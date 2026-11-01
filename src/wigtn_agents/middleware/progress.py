"""ProgressMiddleware — tool activity on the ``custom`` stream, safe to show a user.

Each tool call emits ``{"type": "progress", "kind": "tool", "tool", "call_id", "status"}`` when it
starts and when it ends (``completed`` or ``error``). Arguments, results and error text are never
part of the event: a client can render progress without seeing data. Clients receive it with
``stream_mode=["custom", ...]``. Outside a streaming run the events are dropped.

Implemented on ``wrap_tool_call`` only, so it adds no graph node.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Union

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

ToolResult = Union[ToolMessage, Command[Any]]
ToolHandler = Callable[[ToolCallRequest], ToolResult]
AsyncToolHandler = Callable[[ToolCallRequest], Awaitable[ToolResult]]
Status = Literal["running", "completed", "error"]


def emit_progress(event: dict[str, Any]) -> None:
    """Write one event to the custom stream; no-op outside a run."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except (RuntimeError, ImportError):
        return
    if writer is not None:
        writer(event)


def _event(request: ToolCallRequest, status: Status) -> dict[str, Any]:
    return {
        "type": "progress",
        "kind": "tool",
        "tool": request.tool_call["name"],
        "call_id": request.tool_call.get("id"),
        "status": status,
    }


def _final_status(result: ToolResult) -> Status:
    return "error" if isinstance(result, ToolMessage) and result.status == "error" else "completed"


class ProgressMiddleware(AgentMiddleware):
    """Emit tool start/end markers on the custom stream."""

    def wrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolResult:
        """Start marker, call, end marker."""
        emit_progress(_event(request, "running"))
        try:
            result = handler(request)
        except GraphBubbleUp:
            raise  # an interrupt inside a tool is a pause, not a failure
        except BaseException:
            emit_progress(_event(request, "error"))
            raise
        emit_progress(_event(request, _final_status(result)))
        return result

    async def awrap_tool_call(self, request: ToolCallRequest, handler: AsyncToolHandler) -> ToolResult:
        """Async twin."""
        emit_progress(_event(request, "running"))
        try:
            result = await handler(request)
        except GraphBubbleUp:
            raise
        except BaseException:
            emit_progress(_event(request, "error"))
            raise
        emit_progress(_event(request, _final_status(result)))
        return result
