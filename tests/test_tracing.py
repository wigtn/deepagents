"""Tracing: one span set per run, three targets, and the setup that would split traces is refused."""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from wigtn_agents import tracing
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import default_context
from wigtn_agents.runner import run_local
from wigtn_agents.testing import ScriptedFakeModel
from wigtn_agents.tracing import TracingConfigError, check_config, langsmith_exporter


def test_세_대상을_함께_쓸_수_있다() -> None:
    assert check_config({"OTEL_TARGETS": "langfuse, PHOENIX,LangSmith"}) == ["LANGFUSE", "PHOENIX", "LANGSMITH"]
    assert check_config({}) == []
    assert check_config({"LANGSMITH_TRACING": "true"}) == []  # LangSmith's own tracer alone is fine


def test_LangSmith_자체_추적과_OTLP_대상을_함께_켜면_거부한다() -> None:
    with pytest.raises(TracingConfigError, match="LANGSMITH_TRACING"):
        check_config({"OTEL_TARGETS": "LANGFUSE", "LANGSMITH_TRACING": "true"})
    with pytest.raises(TracingConfigError, match="LANGSMITH_TRACING"):
        check_config({"OTEL_TARGETS": "PHOENIX", "LANGCHAIN_TRACING_V2": "1"})
    with pytest.raises(TracingConfigError, match="알 수 없는"):
        check_config({"OTEL_TARGETS": "LANGFUSE,DATADOG"})


def test_LangSmith_OTLP_엔드포인트와_헤더() -> None:
    assert langsmith_exporter({}) is None
    exporter: Any = langsmith_exporter({"LANGSMITH_API_KEY": "k", "LANGSMITH_PROJECT": "wigtn-agents"})
    assert exporter._endpoint == "https://api.smith.langchain.com/otel/v1/traces"
    assert exporter._headers == {"x-api-key": "k", "Langsmith-Project": "wigtn-agents"}
    eu: Any = langsmith_exporter(
        {"LANGSMITH_API_KEY": "k", "LANGSMITH_OTEL_ENDPOINT": "https://eu.example/otel/v1/traces"}
    )
    assert eu._endpoint == "https://eu.example/otel/v1/traces" and "Langsmith-Project" not in eu._headers
    regional: Any = langsmith_exporter({"LANGSMITH_API_KEY": "k", "LANGSMITH_ENDPOINT": "https://aws.example/"})
    assert regional._endpoint == "https://aws.example/otel/v1/traces"


def test_Phoenix_주소에_OTLP_경로가_없으면_거부한다() -> None:
    with pytest.raises(TracingConfigError, match="/s/me/v1/traces"):
        check_config({"OTEL_TARGETS": "PHOENIX", "PHOENIX_COLLECTOR_ENDPOINT": "https://phoenix.example/s/me/"})
    ok = {"OTEL_TARGETS": "PHOENIX", "PHOENIX_COLLECTOR_ENDPOINT": "https://phoenix.example/s/me/v1/traces"}
    assert check_config(ok) == ["PHOENIX"]


def test_서버는_LangSmith_대상을_Aegra_provider에_붙인다(monkeypatch: pytest.MonkeyPatch) -> None:
    from aegra_api.observability.otel import otel_provider

    added: list[str] = []
    monkeypatch.setattr(otel_provider, "add_custom_target", lambda t: added.append(t.name))
    monkeypatch.setattr(otel_provider, "setup", lambda: None)
    monkeypatch.setitem(tracing._server_done, "value", False)
    monkeypatch.setenv("OTEL_TARGETS", "LANGFUSE,LANGSMITH")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    tracing.ensure_server_targets()
    tracing.ensure_server_targets()  # once per process
    assert added == ["LangSmith"]


async def test_로컬_run은_trace_하나에_run_정보를_싣는다(catalog: Catalog, events: list[dict[str, Any]]) -> None:
    exporter = InMemorySpanExporter()
    provider = tracing.start_local_tracing(exporters=[exporter], batch=False)
    assert provider is not None
    try:
        ctx = default_context("records-guide", entity="1000", principal="u-1", batch_id="b-1", session_type="test")
        result = await run_local(
            "records-guide", "8월 기록 점검", ctx, catalog=catalog, model=ScriptedFakeModel(), sink=events
        )
    finally:
        tracing.stop_local_tracing()
    spans = exporter.get_finished_spans()
    assert spans and len({s.context.trace_id for s in spans if s.context}) == 1
    root = next(s for s in spans if s.parent is None)
    attrs = dict(root.attributes or {})
    assert attrs["session.id"] == result.thread_id and attrs["user.id"] == "u-1"
    assert attrs["langfuse.trace.metadata.agent_id"] == "records-guide"
    metadata = json.loads(str(attrs["metadata"]))
    assert metadata["agent_id"] == "records-guide" and metadata["batch_id"] == "b-1"
    run_ids = {e["run_id"] for e in events}
    assert run_ids == {metadata["run_id"]} == {attrs["langfuse.trace.metadata.run_id"]}
    assert any(s.name == "record-read" for s in spans)


def test_대상이_없으면_로컬_추적을_켜지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_TARGETS", raising=False)
    assert tracing.start_local_tracing() is None
