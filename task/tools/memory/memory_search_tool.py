import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        return "memory_search_tool"

    @property
    def description(self) -> str:
        return (
            "Searches the user's long-term memories using a semantic query. Use when you need to recall stored user facts "
            "(preferences, personal info, goals, plans, context). Do NOT use for general knowledge or current session details. "
            "If nothing relevant exists, it returns no memories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

        results = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k,
        )

        if not results:
            final_result = "No memories found."
        else:
            final_result = self._format_results(results)

        stage = tool_call_params.stage
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Query**: {query}\n\r")
        stage.append_content(f"**Top K**: {top_k}\n\r")
        stage.append_content("## Response: \n")
        stage.append_content(f"{final_result}\n")

        return final_result

    @staticmethod
    def _format_results(results: list[MemoryData]) -> str:
        lines = []
        for memory in results:
            topics = ", ".join(memory.topics) if memory.topics else None
            if topics:
                lines.append(
                    f"- **Content**: {memory.content} | **Category**: {memory.category} | **Topics**: {topics}"
                )
            else:
                lines.append(
                    f"- **Content**: {memory.content} | **Category**: {memory.category}"
                )
        return "\n".join(lines)
