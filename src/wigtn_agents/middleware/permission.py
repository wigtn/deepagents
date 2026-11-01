"""PermissionFilterMiddleware — deny tool calls outside the manifest allowlist.

The allowlist is ``manifest.permissions.allow_tools`` **plus** the deepagents built-in tools
(todo · filesystem · task · async-task). Without that union the harness blocks its own
planning and file idioms. Built-in *write* tools are restricted separately via
``FilesystemPermission`` in the assembly, not here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional, Union

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

ToolResult = Union[ToolMessage, Command[Any]]
ToolHandler = Callable[[ToolCallRequest], ToolResult]
AsyncToolHandler = Callable[[ToolCallRequest], Awaitable[ToolResult]]
Authorizer = Callable[[str, dict[str, Any]], bool]

# deepagents 0.7.x built-in tool names. Pinned here on purpose; tests/test_shape.py checks them
# against the installed package (FsToolName) so a rename fails loudly instead of silently blocking.
# Observed on 0.7.17: the bare agent binds ls·read_file·write_file·edit_file·delete·glob·grep·task;
# write_todos is no longer bound by default and execute only appears with a sandbox backend.
DEEPAGENTS_BUILTIN_TOOLS: frozenset[str] = frozenset(
    {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "task",
        "start_async_task",
        "check_async_task",
        "update_async_task",
        "cancel_async_task",
        "list_async_tasks",
    }
)


class PermissionFilterMiddleware(AgentMiddleware):
    """Block calls to tools that are not allowed. The authorization seam sits here too."""

    def __init__(
        self,
        allowed_tool_names: set[str],
        *,
        authorize: Optional[Authorizer] = None,
        include_builtins: bool = True,
    ) -> None:
        super().__init__()
        self._allowed = set(allowed_tool_names) | (set(DEEPAGENTS_BUILTIN_TOOLS) if include_builtins else set())
        self._authorize = authorize
        self.denied: list[str] = []

    @property
    def allowed(self) -> frozenset[str]:
        """Effective allowlist."""
        return frozenset(self._allowed)

    def _deny(self, request: ToolCallRequest, reason: str) -> ToolMessage:
        name = request.tool_call["name"]
        self.denied.append(name)
        return ToolMessage(
            content=f"차단: {reason} ({name})",
            tool_call_id=request.tool_call["id"],
            name=name,
            status="error",
            additional_kwargs={"denied": "permission"},
        )

    def _check(self, request: ToolCallRequest) -> Optional[ToolMessage]:
        name = request.tool_call["name"]
        args = request.tool_call.get("args", {}) or {}
        if name not in self._allowed:
            return self._deny(request, "허용 목록 밖 도구 호출")
        if self._authorize is not None and not self._authorize(name, args):
            return self._deny(request, "권한 없음")
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolResult:
        """Deny before the tool runs; allowed calls pass through unchanged."""
        denied = self._check(request)
        return denied if denied is not None else handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: AsyncToolHandler) -> ToolResult:
        """Async twin of ``wrap_tool_call``."""
        denied = self._check(request)
        return denied if denied is not None else await handler(request)
