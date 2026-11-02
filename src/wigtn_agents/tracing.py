"""Tracing: one set of OpenTelemetry spans per run, sent to Langfuse, Phoenix and LangSmith.

The runtime traces what an agent did and hands the traces to the tools; it does not read or
analyze them. Spans come from one source, the OpenInference LangChain instrumentor, and fan out
to every target in ``OTEL_TARGETS``:

* ``LANGFUSE`` and ``PHOENIX`` are Aegra's own targets (``LANGFUSE_*``, ``PHOENIX_*`` variables);
* ``LANGSMITH`` is added here and sends the same spans to LangSmith's OTLP endpoint
  (``LANGSMITH_API_KEY``, optional ``LANGSMITH_PROJECT``; the endpoint is ``LANGSMITH_OTEL_ENDPOINT``,
  else ``LANGSMITH_ENDPOINT`` + ``/otel/v1/traces`` for a regional or self-hosted LangSmith).

LangSmith's own LangChain tracer (``LANGSMITH_TRACING=true``) is refused while any OTLP target is
on: LangChain wraps middleware ``wrap_*`` hooks in LangSmith ``@traceable`` runs, which the
OpenInference tracer never sees, so every model and tool call inside a hook would start a new
trace in the other tools. Sending LangSmith the OTLP spans keeps one trace per run everywhere.

The Aegra server sets the tracer provider up at startup; ``ensure_server_targets`` (called when
the worker module is imported) adds the LangSmith target to it. In-process runs
(``scripts/run.py --local``, ``scripts/eval.py``) call ``start_local_tracing`` and
``stop_local_tracing``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Optional

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

logger = logging.getLogger(__name__)

TARGETS_ENV = "OTEL_TARGETS"
LANGSMITH = "LANGSMITH"
KNOWN_TARGETS = ("LANGFUSE", "PHOENIX", LANGSMITH)
LANGSMITH_OTEL_ENDPOINT = "https://api.smith.langchain.com/otel/v1/traces"
_TRUE = ("1", "true", "yes")


class TracingConfigError(ValueError):
    """A tracing setup that would send broken or no traces."""


def configured_targets(environ: Optional[Mapping[str, str]] = None) -> list[str]:
    """Target names in ``OTEL_TARGETS``, upper-cased, unknown names refused."""
    env = os.environ if environ is None else environ
    names = [n.strip().upper() for n in env.get(TARGETS_ENV, "").split(",") if n.strip()]
    unknown = [n for n in names if n not in (*KNOWN_TARGETS, "GENERIC", "OTLP", "DEFAULT")]
    if unknown:
        raise TracingConfigError(
            f"{TARGETS_ENV}에 알 수 없는 대상이 있습니다: {unknown} (쓸 수 있는 값: {KNOWN_TARGETS})"
        )
    return names


def check_config(environ: Optional[Mapping[str, str]] = None) -> list[str]:
    """Targets to use; refuses LangSmith's own tracer next to OTLP targets."""
    env = os.environ if environ is None else environ
    targets = configured_targets(env)
    native = env.get("LANGSMITH_TRACING", env.get("LANGCHAIN_TRACING_V2", "")).strip().lower() in _TRUE
    if targets and native:
        raise TracingConfigError(
            "LANGSMITH_TRACING과 OTEL_TARGETS를 함께 켜면 Langfuse·Phoenix에서 run 하나가 trace 여러 개로 갈라집니다. "
            f"LANGSMITH_TRACING을 끄고 {TARGETS_ENV}에 {LANGSMITH}를 넣으세요"
        )
    phoenix = env.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    if "PHOENIX" in targets and phoenix and not phoenix.rstrip("/").endswith("/v1/traces"):
        # Aegra posts to this URL as given; a project URL without the OTLP path answers 404/405.
        raise TracingConfigError(
            f"PHOENIX_COLLECTOR_ENDPOINT는 OTLP 경로까지 적어야 합니다: {phoenix.rstrip('/')}/v1/traces"
        )
    return targets


def langsmith_endpoint(environ: Optional[Mapping[str, str]] = None) -> str:
    """LangSmith's OTLP traces URL: explicit, derived from ``LANGSMITH_ENDPOINT``, or the default."""
    env = os.environ if environ is None else environ
    explicit = env.get("LANGSMITH_OTEL_ENDPOINT", "").strip()
    if explicit:
        return explicit
    api = env.get("LANGSMITH_ENDPOINT", "").strip().rstrip("/")
    return f"{api}/otel/v1/traces" if api else LANGSMITH_OTEL_ENDPOINT


def langsmith_exporter(environ: Optional[Mapping[str, str]] = None) -> Optional[SpanExporter]:
    """OTLP exporter for LangSmith, or ``None`` without an API key."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    env = os.environ if environ is None else environ
    key = env.get("LANGSMITH_API_KEY", "").strip()
    if not key:
        logger.warning("LangSmith 대상: LANGSMITH_API_KEY가 비어 있어 보내지 않습니다")
        return None
    headers = {"x-api-key": key}
    project = env.get("LANGSMITH_PROJECT", "").strip()
    if project:
        headers["Langsmith-Project"] = project
    return OTLPSpanExporter(endpoint=langsmith_endpoint(env), headers=headers)


def _aegra_target(name: str) -> Any:
    from aegra_api.observability.targets import GenericOtelTarget, LangfuseTarget, PhoenixTarget

    if name == "LANGFUSE":
        return LangfuseTarget()
    if name == "PHOENIX":
        return PhoenixTarget()
    return GenericOtelTarget()


def _langsmith_target() -> Any:
    from aegra_api.observability.targets import BaseOtelTarget

    class LangSmithTarget(BaseOtelTarget):  # type: ignore[misc]
        """LangSmith's OTLP endpoint as an Aegra target."""

        @property
        def name(self) -> str:
            return "LangSmith"

        def get_exporter(self) -> Optional[SpanExporter]:
            return langsmith_exporter()

    return LangSmithTarget()


def exporters_for(targets: Sequence[str]) -> list[tuple[str, SpanExporter]]:
    """Exporters for the named targets; a target missing its settings is skipped with a warning."""
    built: list[tuple[str, SpanExporter]] = []
    for name in targets:
        exporter = langsmith_exporter() if name == LANGSMITH else _aegra_target(name).get_exporter()
        if exporter is None:
            logger.warning("추적 대상 %s: 설정이 없어 보내지 않습니다", name)
            continue
        built.append((name, exporter))
    return built


# -- server ------------------------------------------------------------------------------------

_server_lock = threading.Lock()
_server_done = {"value": False}


def ensure_server_targets() -> None:
    """Check the tracing settings and add the LangSmith target to Aegra's tracer provider.

    Called on worker import. Outside the Aegra server (no ``aegra_api``) it does nothing.
    """
    with _server_lock:
        if _server_done["value"]:
            return
        _server_done["value"] = True
        targets = check_config()
        if LANGSMITH not in targets:
            return
        try:
            from aegra_api.observability.base import get_observability_manager
            from aegra_api.observability.otel import otel_provider
        except ImportError:  # pragma: no cover - aegra is a runtime dependency
            return
        otel_provider.add_custom_target(_langsmith_target())
        # Aegra registers the provider only if it was enabled at startup; with LANGSMITH as the
        # only target it was not, so register and set it up here (both are idempotent).
        get_observability_manager().register_provider(otel_provider)
        otel_provider.setup()


# -- in-process runs ---------------------------------------------------------------------------

_local: dict[str, Any] = {"provider": None}


def start_local_tracing(
    *,
    exporters: Optional[Sequence[SpanExporter]] = None,
    batch: bool = True,
) -> Optional[TracerProvider]:
    """Instrument LangChain in this process and export to the configured targets.

    Args:
        exporters: Exporters to use instead of ``OTEL_TARGETS`` (tests).
        batch: Batch span export (off in tests, so spans are visible right away).

    Returns:
        The tracer provider, or ``None`` when no target is configured.

    """
    from aegra_api.observability.span_enrichment import SpanEnrichmentProcessor
    from openinference.instrumentation.langchain import LangChainInstrumentor

    if _local["provider"] is not None:
        return _local["provider"]  # type: ignore[no-any-return]
    if exporters is None:
        named = exporters_for(check_config())
        exporters = [e for _, e in named]
        if named:
            logger.info("추적 대상: %s", ", ".join(n for n, _ in named))
    if not exporters:
        return None
    provider = TracerProvider()
    provider.add_span_processor(SpanEnrichmentProcessor())
    for exporter in exporters:
        provider.add_span_processor(BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter))
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _local["provider"] = provider
    return provider


def stop_local_tracing() -> None:
    """Flush and shut down in-process tracing (before a script exits)."""
    from openinference.instrumentation.langchain import LangChainInstrumentor

    provider = _local["provider"]
    if provider is None:
        return
    LangChainInstrumentor().uninstrument()
    provider.force_flush()
    provider.shutdown()
    _local["provider"] = None


@contextmanager
def run_attributes(
    *,
    thread_id: str,
    principal: Optional[str],
    name: str,
    metadata: dict[str, Any],
) -> Iterator[None]:
    """The same per-run span attributes Aegra sets on the server (session, user, name, metadata)."""
    from aegra_api.observability.span_enrichment import _trace_attrs, merge_run_metadata, set_trace_context

    token = _trace_attrs.set(None)
    try:
        set_trace_context(
            user_id=principal,
            session_id=thread_id,
            trace_name=name,
            metadata=merge_run_metadata(metadata, {"thread_id": thread_id}),
        )
        yield
    finally:
        _trace_attrs.reset(token)
