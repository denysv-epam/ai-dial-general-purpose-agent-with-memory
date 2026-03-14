import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class DeleteMemoryTool(BaseTool):
    """
    Tool for deleting all long-term memories about the user.

    This permanently removes all stored memories from the system.
    Use with caution - this action cannot be undone.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "memory_delete_tool"

    @property
    def description(self) -> str:
        return (
            "Deletes long-term memories. Use a query to delete specific facts (e.g., 'forget my name'). "
            "Only delete ALL memories when the user explicitly asks to erase/reset everything."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to forget. Use for targeted deletion."
                },
                "delete_all": {
                    "type": "boolean",
                    "description": "Set true ONLY when the user asks to delete all memories.",
                    "default": False
                }
            }
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = tool_call_params.tool_call.function.arguments
        payload = json.loads(arguments) if arguments else {}
        query = payload.get("query")
        delete_all = payload.get("delete_all", False)

        if delete_all:
            result = await self.memory_store.delete_all_memories(api_key=tool_call_params.api_key)
        elif query:
            result = await self.memory_store.delete_memories(
                api_key=tool_call_params.api_key,
                query=query,
            )
        else:
            result = "No deletion performed. Provide a query or set delete_all to true."

        stage = tool_call_params.stage
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"```json\n\r{json.dumps(payload, indent=2)}\n\r```\n\r")
        stage.append_content("## Response: \n")
        stage.append_content(f"{result}\n")

        return result
