"""EventRecorderMiddleware — the run record every attached system reads.

Every run emits ``run.start · llm.call · tool.call · run.end`` (and ``run.error``) stamped with
the manifest, pinned asset versions, model alias / id / provider, middleware preset, extra
middleware names, ``session_type`` and ``batch_id``. ``llm.call`` carries the provider-reported
token usage. Both lanes (agent and action) and subagents share the stamp (``RunStamp``), so one
JSONL file holds comparable rows. Sinks are pluggable: an in-memory list for tests, a JSONL file
by default. The sink never touches the runtime's own tables.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, Protocol, Union, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import Command

from wigtn_agents.context import PlatformState, RunContext
from wigtn_agents.manifest import AnyManifest, Manifest
from wigtn_agents.models import ResolvedModel

ToolResult = Union[ToolMessage, Command[Any]]
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


@dataclass(frozen=True)
class RunStamp:
    """Fields every event of one run carries."""

    agent_id: str
    kind: str
    manifest_version: int
    model_alias: str
    model_id: str
    model_provider: str
    preset: str
    extra_middleware: tuple[str, ...] = ()
    skill: Optional[str] = None
    role: str = "main"
    session_type: str = "production"
    batch_id: Optional[str] = None
    principal: Optional[str] = None
    parent_run_id: Optional[str] = None
    fallback_run_id: str = field(default_factory=lambda: f"local-{uuid.uuid4().hex[:12]}")

    @classmethod
    def for_manifest(
        cls,
        manifest: AnyManifest,
        ctx: RunContext,
        model: ResolvedModel,
        *,
        extra_middleware: tuple[str, ...] = (),
        role: str = "main",
    ) -> RunStamp:
        """Stamp of one manifest in one run."""
        return cls(
            agent_id=manifest.agent_id,
            kind="agent" if isinstance(manifest, Manifest) else "action",
            manifest_version=manifest.metadata.version,
            model_alias=model.alias,
            model_id=model.model_id,
            model_provider=model.provider,
            preset=manifest.harness_preset,
            extra_middleware=extra_middleware,
            skill=manifest.skills[0] if manifest.skills else None,
            role=role,
            session_type=ctx.get("session_type", "production"),
            batch_id=ctx.get("batch_id"),
            principal=ctx.get("principal"),
            parent_run_id=ctx.get("parent_run_id"),
        )


def build_event(
    stamp: RunStamp,
    event_type: str,
    *,
    asset_versions: Mapping[str, str],
    image_version: str = "dev",
    **fields: Any,
) -> dict[str, Any]:
    """One event row in the common shape."""
    cfg = _configurable()
    row = asdict(stamp)
    row.pop("fallback_run_id")
    principal = row.pop("principal")
    return {
        "event_id": uuid.uuid4().hex,
        "ts": _now(),
        "thread_id": cfg.get("thread_id"),
        "run_id": cfg.get("run_id") or stamp.fallback_run_id,
        **row,
        "extra_middleware": list(stamp.extra_middleware),
        "asset_versions": dict(asset_versions),
        "image_version": image_version,
        "event_type": event_type,
        "actor_type": "tester" if stamp.session_type == "evaluation" else "user",
        "actor_id": principal,
        **fields,
    }


def usage_of(message: Any) -> Optional[dict[str, int]]:
    """Provider-reported token usage of one model message, if any."""
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    inputs = usage.get("input_token_details") or {}
    outputs = usage.get("output_token_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_tokens": int(inputs.get("cache_read") or 0),
        "reasoning_tokens": int(outputs.get("reasoning") or 0),
    }


def denial_of(result: Any) -> Optional[str]:
    """Which gate refused a tool call (``permission`` · ``scope`` · ``filesystem``), or ``None``."""
    if not isinstance(result, ToolMessage) or result.status != "error":
        return None
    marked = (result.additional_kwargs or {}).get("denied")
    if marked:
        return str(marked)
    if result.text.startswith("Error: permission denied"):
        return "filesystem"  # deepagents FilesystemPermission
    return None


class EventRecorderMiddleware(AgentMiddleware):
    """Emit common-schema execution events with version stamps."""

    state_schema = PlatformState

    def __init__(
        self,
        sink: EventSink,
        stamp: RunStamp,
        *,
        approval_tools: frozenset[str] = frozenset(),
        asset_versions: Optional[Mapping[str, str]] = None,
        lifecycle: bool = True,
    ) -> None:
        super().__init__()
        # False inside an action's reason loop: the action records run.start/run.end itself.
        self._lifecycle = lifecycle
        # Tools the human-approval step stops before: a model turn that calls one ends the
        # run as interrupted (the approval middleware sits outside this one).
        self._approval_tools = approval_tools
        self._sink = sink
        self.stamp = stamp
        # Subagents have no pinned versions in their own state; the parent passes them in.
        self._static_versions = dict(asset_versions) if asset_versions is not None else None
        # Runs already closed with an error: parallel tool calls that fail together end it once.
        self._failed_runs: set[str] = set()

    def _emit(self, event_type: str, state: PlatformState, **fields: Any) -> None:
        meta = state.get("run_meta") or {}
        versions = self._static_versions if self._static_versions is not None else state.get("asset_versions") or {}
        self._sink.append(
            build_event(
                self.stamp,
                event_type,
                asset_versions=versions,
                image_version=meta.get("image_version", "dev"),
                **fields,
            )
        )

    # -- hooks ---------------------------------------------------------------------------------

    def before_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """run.start"""
        if self._lifecycle:
            self._emit("run.start", state)
        return None

    async def abefore_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """Async twin."""
        return self.before_agent(state, runtime)

    def after_model(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """llm.call"""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        self._emit("llm.call", state, tool_call_count=len(tool_calls), usage=usage_of(last))
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
        if isinstance(error, GraphBubbleUp) or not self._lifecycle:
            return  # an interrupt or a handoff is not a failure; an action records its own end
        run_id = str(_configurable().get("run_id") or self.stamp.fallback_run_id)
        if run_id in self._failed_runs:
            return
        self._failed_runs.add(run_id)
        self._emit("run.error", state, error_class=type(error).__name__)
        self._emit("run.end", state, status="error")

    def wrap_model_call(self, request: ModelRequest, handler: ModelHandler) -> ModelResponse:
        """A model call that raises ends the run with an error."""
        try:
            response = handler(request)
        except BaseException as exc:
            self._failed(cast(PlatformState, request.state), exc)
            raise
        return self._answered(cast(PlatformState, request.state), response)

    async def awrap_model_call(self, request: ModelRequest, handler: AsyncModelHandler) -> ModelResponse:
        """Async twin."""
        try:
            response = await handler(request)
        except BaseException as exc:
            self._failed(cast(PlatformState, request.state), exc)
            raise
        return self._answered(cast(PlatformState, request.state), response)

    async def aafter_model(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """Async twin."""
        return self.after_model(state, runtime)

    def _record_tool(self, request: ToolCallRequest, result: ToolResult, started: float) -> ToolResult:
        latency_ms = int((time.perf_counter() - started) * 1000)
        denied = denial_of(result)
        failed = isinstance(result, ToolMessage) and result.status == "error"
        self._emit(
            "tool.call",
            request.state,
            tool={
                "name": request.tool_call["name"],
                "allowed": denied is None,
                "denied_by": denied,
                "status": "error" if failed else "ok",
                "latency_ms": latency_ms,
            },
        )
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolResult:
        """tool.call (records denials too — the filter returns an error ToolMessage)."""
        started = time.perf_counter()
        try:
            result = handler(request)
        except BaseException as exc:
            self._failed(cast(PlatformState, request.state), exc)
            raise
        return self._record_tool(request, result, started)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: AsyncToolHandler) -> ToolResult:
        """Async twin."""
        started = time.perf_counter()
        try:
            result = await handler(request)
        except BaseException as exc:
            self._failed(cast(PlatformState, request.state), exc)
            raise
        return self._record_tool(request, result, started)

    def after_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """run.end"""
        if not self._lifecycle:
            return None
        last = state["messages"][-1]
        outcome = last.text if isinstance(last, AIMessage) else ""
        self._emit("run.end", state, status="ok", outcome=outcome[:200])
        return None

    async def aafter_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """Async twin."""
        return self.after_agent(state, runtime)
