"""EventRecorderMiddleware — the Observability Contract implementation point (platform capability F12).

Every run emits ``run.start · llm.call · tool.call · run.end`` with manifest / asset / model
version stamps, ``session_type`` and ``batch_id`` (event-schema-v0 common fields). Sinks are
pluggable: an in-memory list for tests, a JSONL file by default. The sink never touches the
runtime's own tables.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import Command

from wigtn_agents.context import PlatformState
from wigtn_agents.manifest import Manifest

ToolResult = Union[ToolMessage, Command]
ToolHandler = Callable[[ToolCallRequest], ToolResult]
AsyncToolHandler = Callable[[ToolCallRequest], Awaitable[ToolResult]]
ModelHandler = Callable[[ModelRequest], ModelResponse]
AsyncModelHandler = Callable[[ModelRequest], Awaitable[ModelResponse]]
EVENT_LOG_ENV = "WIGTN_AGENT_EVENT_LOG"


class EventSink(Protocol):
    """Anything that accepts one event dict."""

    def append(self, event: dict[str, Any]) -> None:  # pragma: no cover - protocol
        """Persist one event."""


class JsonlSink:
    """Append-only JSONL file sink."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or Path(os.environ.get(EVENT_LOG_ENV, "var/events.jsonl"))

    def append(self, event: dict[str, Any]) -> None:
        """Write one line."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _configurable() -> dict[str, Any]:
    """Current run's ``configurable`` (thread_id · run_id injected by the runtime), if any."""
    try:
        from langgraph.config import get_config

        return dict(get_config().get("configurable", {}))
    except Exception:  # noqa: BLE001 - outside a graph run there is no config
        return {}


class EventRecorderMiddleware(AgentMiddleware):
    """Emit common-schema execution events with version stamps."""

    state_schema = PlatformState

    def __init__(
        self,
        sink: EventSink,
        manifest: Manifest,
        *,
        session_type: str = "production",
        batch_id: Optional[str] = None,
        principal: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        approval_tools: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__()
        # Tools the human-approval step stops before: a model turn that calls one ends the
        # run as interrupted (the approval middleware sits outside this one).
        self._approval_tools = approval_tools
        self._sink = sink
        self._manifest = manifest
        self._session_type = session_type
        self._batch_id = batch_id
        self._principal = principal
        self._parent_run_id = parent_run_id
        self._fallback_run_id = f"local-{uuid.uuid4().hex[:12]}"

    def _emit(self, event_type: str, state: PlatformState, **fields: Any) -> None:
        cfg = _configurable()
        meta = state.get("run_meta") or {}
        self._sink.append(
            {
                "event_id": uuid.uuid4().hex,
                "ts": _now(),
                "thread_id": cfg.get("thread_id"),
                "run_id": cfg.get("run_id") or self._fallback_run_id,
                "parent_run_id": self._parent_run_id,
                "batch_id": self._batch_id,
                "agent_id": self._manifest.agent_id,
                "manifest_version": self._manifest.metadata.version,
                "asset_versions": dict(state.get("asset_versions") or {}),
                "image_version": meta.get("image_version", "dev"),
                "model_id": self._manifest.model.ref,
                "event_type": event_type,
                "session_type": self._session_type,
                "actor_type": "tester" if self._session_type == "evaluation" else "user",
                "actor_id": self._principal,
                **fields,
            }
        )

    # -- hooks ---------------------------------------------------------------------------------

    def before_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """run.start"""
        self._emit("run.start", state)
        return None

    async def abefore_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Async twin."""
        return self.before_agent(state, runtime)

    def after_model(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """llm.call"""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        self._emit("llm.call", state, tool_call_count=len(tool_calls))
        return None

    def _answered(self, state: PlatformState, response: ModelResponse) -> ModelResponse:
        """run.end ``interrupted`` when the turn calls a tool that waits for approval.

        Checked on the model response: the approval step's ``after_model`` runs before this
        middleware's and stops the run, so ``after_model`` here never sees that turn.
        """
        calls = [c for m in response.result for c in (getattr(m, "tool_calls", None) or [])]
        gated = sorted({c["name"] for c in calls} & self._approval_tools)
        if gated:
            self._emit("run.end", state, status="interrupted", waiting_for=gated)
        return response

    def _failed(self, state: PlatformState, error: BaseException) -> None:
        """run.error then run.end ``error``; the class only, the message may hold user data."""
        if isinstance(error, GraphBubbleUp):
            return  # an interrupt or a handoff is not a failure
        self._emit("run.error", state, error_class=type(error).__name__)
        self._emit("run.end", state, status="error")

    def wrap_model_call(self, request: ModelRequest, handler: ModelHandler) -> ModelResponse:
        """A model call that raises ends the run with an error."""
        try:
            response = handler(request)
        except BaseException as exc:
            self._failed(request.state, exc)
            raise
        return self._answered(request.state, response)

    async def awrap_model_call(self, request: ModelRequest, handler: AsyncModelHandler) -> ModelResponse:
        """Async twin."""
        try:
            response = await handler(request)
        except BaseException as exc:
            self._failed(request.state, exc)
            raise
        return self._answered(request.state, response)

    async def aafter_model(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Async twin."""
        return self.after_model(state, runtime)

    def _record_tool(self, request: ToolCallRequest, result: ToolResult, started: float) -> ToolResult:
        latency_ms = int((time.perf_counter() - started) * 1000)
        allowed = not (isinstance(result, ToolMessage) and result.status == "error")
        self._emit(
            "tool.call",
            request.state,
            tool={"name": request.tool_call["name"], "allowed": allowed, "latency_ms": latency_ms},
            skill=self._manifest.skills[0] if self._manifest.skills else None,
        )
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolResult:
        """tool.call (records denials too — the filter returns an error ToolMessage)."""
        started = time.perf_counter()
        try:
            result = handler(request)
        except BaseException as exc:
            self._failed(request.state, exc)
            raise
        return self._record_tool(request, result, started)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: AsyncToolHandler) -> ToolResult:
        """Async twin."""
        started = time.perf_counter()
        try:
            result = await handler(request)
        except BaseException as exc:
            self._failed(request.state, exc)
            raise
        return self._record_tool(request, result, started)

    def after_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """run.end"""
        last = state["messages"][-1]
        outcome = last.text if isinstance(last, AIMessage) else ""
        self._emit("run.end", state, status="ok", outcome=outcome[:200])
        return None

    async def aafter_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Async twin."""
        return self.after_agent(state, runtime)
