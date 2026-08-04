from __future__ import annotations

import uuid
from typing import Any

from data_harness.llm.types import ToolSpec


class Planner:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._turns_since_update: int = 0

    def add(self, items: list[str]) -> str:
        for text in items:
            self._items.append(
                {
                    "id": str(uuid.uuid4())[:8],
                    "text": text,
                    "status": "pending",
                }
            )
        self._turns_since_update = 0
        return self.list()

    def update(self, id: str, status: str) -> str:
        for item in self._items:
            if item["id"] == id:
                item["status"] = status
                self._turns_since_update = 0
                return f"Updated {id!r} to {status!r}"
        return f"Item {id!r} not found"

    def list(self) -> str:
        if not self._items:
            return "Todo list is empty."
        lines = []
        for item in self._items:
            lines.append(f"[{item['id']}] ({item['status']}) {item['text']}")
        return "\n".join(lines)

    def reminder_hook(self, current_turn: int, max_turns: int) -> str | None:
        pending = [i for i in self._items if i["status"] == "pending"]
        n = self._turns_since_update
        self._turns_since_update += 1

        if not pending:
            return None

        if n >= 12:
            return (
                f"URGENT: You have {len(pending)} pending todo item(s) "
                f"that haven't been updated in {n} turns. Address them immediately."
            )
        if n >= 8:
            return (
                f"WARNING: {len(pending)} pending todo item(s) remain "
                f"with no updates for {n} turns. Please make progress on your plan."
            )
        if n >= 4:
            return (
                f"Reminder: You have {len(pending)} pending todo item(s). "
                f"Consider updating your plan."
            )
        return None

    def make_tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="planner__add",
                description="Add items to your todo list.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of task descriptions to add.",
                        }
                    },
                    "required": ["items"],
                },
                handler=self.add,
            ),
            ToolSpec(
                name="planner__update",
                description="Update the status of a todo item.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Item ID"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done", "blocked"],
                        },
                    },
                    "required": ["id", "status"],
                },
                handler=self.update,
            ),
            ToolSpec(
                name="planner__list",
                description="List all todo items and their statuses.",
                input_schema={"type": "object", "properties": {}},
                handler=self.list,
            ),
        ]
