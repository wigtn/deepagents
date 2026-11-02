"""ScopeGuardMiddleware — the LLM never supplies a data-scope identifier.

For tools registered in ``SCOPE_PARAMS`` the named arguments are overwritten with the
canonical value from the run context. No canonical value → deny (fail-closed), never pass the
LLM's value through. The original tool call stays untouched in history for audit; only the
request handed to the tool is rewritten.

The assembly passes the server-built run context (``worker.build_run_context``) in as
``static_context``, and that value wins: the main agent, its subagents (separate graphs that do
not receive the parent's runtime context) and the action lane then all use one canonical scope,
even when a client sends different values in ``context`` and ``configurable``. The runtime
context is only read when no static context is given.

A denial carries ``additional_kwargs={"denied": ...}`` so the run record can tell a gate apart
from a tool that failed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Optional, Union, cast

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolCall, ToolMessage
from langgraph.types import Command

DENIED_KEY = "denied"
ToolResult = Union[ToolMessage, Command[Any]]
ToolHandler = Callable[[ToolCallRequest], ToolResult]
AsyncToolHandler = Callable[[ToolCallRequest], Awaitable[ToolResult]]

# tool name → (arg name, context key). Registration is the protection; tests check coverage.
_COMPANY = (("company", "entity"),)
SCOPE_PARAMS: dict[str, tuple[tuple[str, str], ...]] = {
    # tests/fixtures/catalog
    "record-read": (("entity", "entity"),),
    "record-write": (("entity", "entity"),),
    "view-read": (("entity", "entity"),),
    # catalog/ (sample ERP): the company code is the run's entity
    **{
        name: _COMPANY
        for name in (
            "erp-period-status",
            "erp-po-read",
            "erp-invoice-read",
            "erp-three-way-match",
            "erp-stock-read",
            "erp-ar-aging",
            "erp-trial-balance",
            "erp-close-checklist",
            "erp-journal-post",
            "erp-payment-block",
            "erp-requisition-create",
        )
    },
}


def _context_of(request: ToolCallRequest) -> Mapping[str, Any]:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    if isinstance(context, Mapping):
        return context
    return {}


class ScopeGuardMiddleware(AgentMiddleware):
    """Overwrite scope arguments with the run context's canonical values."""

    def __init__(
        self,
        scope_params: Optional[dict[str, tuple[tuple[str, str], ...]]] = None,
        *,
        static_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._params = scope_params if scope_params is not None else SCOPE_PARAMS
        self._static = dict(static_context or {})
        self.overrides: list[dict[str, Any]] = []

    def _rewrite(self, request: ToolCallRequest) -> Union[ToolCallRequest, ToolMessage]:
        name = request.tool_call["name"]
        bindings = self._params.get(name)
        if not bindings:
            return request
        context = _context_of(request)
        injected: dict[str, Any] = {}
        for arg_name, ctx_key in bindings:
            canonical = self._static.get(ctx_key) if self._static else context.get(ctx_key)
            if canonical is None:
                return ToolMessage(
                    content=f"차단: 스코프 정본 없음 ({name}.{arg_name})",
                    tool_call_id=request.tool_call["id"],
                    name=name,
                    status="error",
                    additional_kwargs={DENIED_KEY: "scope"},
                )
            injected[arg_name] = canonical
        self.overrides.append({"tool": name, **injected})
        new_call = {**request.tool_call, "args": {**(request.tool_call.get("args") or {}), **injected}}
        return request.override(tool_call=cast(ToolCall, new_call))

    def wrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolResult:
        """Inject canonical scope values or deny."""
        rewritten = self._rewrite(request)
        return rewritten if isinstance(rewritten, ToolMessage) else handler(rewritten)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: AsyncToolHandler) -> ToolResult:
        """Async twin of ``wrap_tool_call``."""
        rewritten = self._rewrite(request)
        return rewritten if isinstance(rewritten, ToolMessage) else await handler(rewritten)
