import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
from typing import Any, cast
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._cache: dict[str, MemoryCollection] = {}
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        app_home = await dial_client.my_appdata_home()
        if app_home is None:
            raise ValueError("Failed to resolve appdata home path.")
        memory_path = app_home / "__long-memories" / "data.json"
        return f"files/{memory_path.as_posix()}"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        memory_file_path = await self._get_memory_file_path(dial_client)

        cached = self._cache.get(memory_file_path)
        if cached is not None:
            return cached

        try:
            download_response = await dial_client.files.download(memory_file_path)
            file_content = download_response.get_content().decode('utf-8')
            content_json = json.loads(file_content)
            memories = MemoryCollection.model_validate(content_json)
        except Exception:
            memories = MemoryCollection()

        self._cache[memory_file_path] = memories
        return memories

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        memory_file_path = await self._get_memory_file_path(dial_client)
        memories.updated_at = datetime.now(UTC)
        content = memories.model_dump_json()
        await dial_client.files.upload(url=memory_file_path, file=content.encode('utf-8'))
        self._cache[memory_file_path] = memories

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""
        memories = await self._load_memories(api_key)
        embedding = self.model.encode([content])[0].tolist()

        memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics,
            ),
            embedding=embedding,
        )

        memories.memories.append(memory)
        await self._save_memories(api_key, memories)
        return "Memory stored successfully."

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        collection = await self._load_memories(api_key)
        if not collection.memories:
            return []

        if self._needs_deduplication(collection):
            collection = await self._deduplicate_and_save(api_key, collection)

        embeddings = np.array([m.embedding for m in collection.memories], dtype='float32')
        if embeddings.size == 0:
            return []

        faiss_any = cast(Any, faiss)
        faiss_any.normalize_L2(embeddings)
        index: Any = faiss_any.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        query_embedding = self.model.encode([query]).astype('float32')
        faiss_any.normalize_L2(query_embedding)
        k = min(max(int(top_k), 1), len(collection.memories))
        _, indices = index.search(query_embedding, k=k)

        results: list[MemoryData] = []
        for idx in indices[0]:
            if idx < 0:
                continue
            results.append(collection.memories[int(idx)].data)
        return results

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        if len(collection.memories) <= 10:
            return False

        if collection.last_deduplicated_at is None:
            return True

        now = datetime.now(UTC)
        return now - collection.last_deduplicated_at > timedelta(hours=self.DEDUP_INTERVAL_HOURS)

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        collection.memories = self._deduplicate_fast(collection.memories)
        collection.last_deduplicated_at = datetime.now(UTC)
        await self._save_memories(api_key, collection)
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        if len(memories) <= 1:
            return memories

        embeddings = np.array([m.embedding for m in memories], dtype='float32')
        faiss_any = cast(Any, faiss)
        faiss_any.normalize_L2(embeddings)

        index: Any = faiss_any.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        k = min(10, len(memories))
        similarities, neighbors = index.search(embeddings, k=k)

        order = sorted(
            range(len(memories)),
            key=lambda i: (memories[i].data.importance, memories[i].data.id),
            reverse=True
        )
        rank = {idx: pos for pos, idx in enumerate(order)}

        removed: set[int] = set()
        for i in order:
            if i in removed:
                continue

            for pos, j in enumerate(neighbors[i]):
                if j == i:
                    continue
                if similarities[i][pos] <= 0.75:
                    continue

                if j in removed:
                    continue

                importance_i = memories[i].data.importance
                importance_j = memories[j].data.importance
                if importance_j < importance_i:
                    removed.add(int(j))
                elif importance_j == importance_i and rank[j] > rank[i]:
                    removed.add(int(j))

        return [mem for idx, mem in enumerate(memories) if idx not in removed]

    async def delete_all_memories(self, api_key: str, ) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        memory_file_path = await self._get_memory_file_path(dial_client)
        await dial_client.files.delete(url=memory_file_path)
        self._cache.pop(memory_file_path, None)
        return "All long-term memories have been deleted."
