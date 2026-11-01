"""Gates, limits and records hold on every path a call can take."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException
from langgraph.checkpoint.memory import InMemorySaver

from tests.conftest import scripted_model, thread, user_turn
from wigtn_agents.actions import build_action_graph
from wigtn_agents.actions.chassis import BOUNDED_REACT_ROUNDS, GatherToolError
from wigtn_agents.assembly import GENERAL_PURPOSE_NAME, assemble, build_backend, build_subagents, resolve_run_model
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import Asset
from wigtn_agents.context import default_context
from wigtn_agents.middleware.extra import register_middleware, unregister_middleware
from wigtn_agents.middleware.repeated import RepeatedToolCallMiddleware
from wigtn_agents.tools.gateway import ToolClients, ToolGatewayClient, make_tool
from wigtn_agents.tools.http import HttpToolClient


def _read(i: int, period: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "record-read", "args": {"period": period}, "id": f"c{i}"}])


def test_스코프는_서버가_만든_run_context_하나만_쓴다(catalog: Catalog) -> None:
    # Given: the server-built context says 1000, the runtime context a client sent says 2222
    ctx = default_context("records-guide", entity="1000")
    gateway = ToolGatewayClient()
    agent = assemble(
        catalog.manifest("records-guide"),
        catalog,
        ctx,
        model=scripted_model(_read(1, "2026-08"), AIMessage(content="끝")),
        clients=ToolClients(gateway=gateway),
        sink=[],
    )
    agent.invoke(user_turn("점검"), context={**ctx, "entity": "2222"})
    assert [c["payload"]["entity"] for c in gateway.calls] == ["1000"]


async def test_bounded_react는_도구_라운드_8회를_다_쓸_수_있다(catalog: Catalog) -> None:
    ctx = default_context("records-review", entity="1000")
    reads = [_read(i, f"2026-{i:02d}") for i in range(1, BOUNDED_REACT_ROUNDS + 1)]
    final = AIMessage(content="", tool_calls=[{"name": "ActionResult", "args": {"verdict": "ok"}, "id": "r"}])
    action = catalog.action("records-review")
    graph = build_action_graph(
        action, catalog, ctx, resolved=resolve_run_model(action, ctx), model=scripted_model(*reads, final), sink=[]
    )
    state = await graph.ainvoke({"input": {}})
    assert state["result"]["verdict"] == "ok"


async def test_라운드를_넘기면_재귀_오류가_아니라_분명한_오류로_끝난다(catalog: Catalog) -> None:
    ctx = default_context("records-review", entity="1000")
    reads = [_read(i, f"2026-{i:02d}") for i in range(1, BOUNDED_REACT_ROUNDS + 3)]
    action = catalog.action("records-review")
    graph = build_action_graph(
        action, catalog, ctx, resolved=resolve_run_model(action, ctx), model=scripted_model(*reads), sink=[]
    )
    with pytest.raises(ValueError, match="라운드 상한 8회"):
        await graph.ainvoke({"input": {}})


async def test_gather_도구_오류는_데이터가_아니라_실패다(catalog: Catalog, events: list[dict[str, Any]]) -> None:
    def down(operation: Any, payload: dict[str, Any]) -> dict[str, Any]:
        raise ToolException("HTTP 503: upstream")

    gateway = ToolGatewayClient({"record-read": down})
    ctx = default_context("records-summary", entity="1000")
    action = catalog.action("records-summary").model_copy(update={"writeback": "none"})
    graph = build_action_graph(
        action,
        catalog,
        ctx,
        resolved=resolve_run_model(action, ctx),
        model=scripted_model(),
        clients=ToolClients(gateway=gateway),
        sink=events,
    )
    with pytest.raises(GatherToolError):
        await graph.ainvoke({"input": {"period": "2026-08", "view": "v"}})
    rows = [(e["event_type"], e.get("tool", {}).get("status")) for e in events]
    assert ("tool.call", "error") in rows and rows[-1] == ("run.end", None)


def test_http_경로_인자는_한_구간으로_인코딩된다(catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIGTN_SECRET_RECORDS_BASE_URL", "https://records.example")
    seen: list[str] = []
    transport = httpx.MockTransport(lambda r: (seen.append(r.url.raw_path.decode()), httpx.Response(200, json={}))[1])
    asset = Asset.model_validate(
        {
            "kind": "tool",
            "name": "entity-record",
            "version": 1,
            "status": "active",
            "owner": "t",
            "body": {
                "input_schema": {
                    "type": "object",
                    "properties": {"entity": {"type": "string"}, "record_id": {"type": "string"}},
                    "required": ["entity", "record_id"],
                },
                "side_effect": "read",
                "idempotent": True,
                "binding": {
                    "type": "http",
                    "api": "api://records-api@1",
                    "operation": "GET /entities/{entity}/records/{record_id}",
                },
            },
        }
    )
    tool = make_tool(asset, ToolClients(http=HttpToolClient(transport=transport)), catalog)
    tool.invoke({"entity": "1000", "record_id": "../../9999/records/R-1"})
    assert seen == ["/entities/1000/records/..%2F..%2F9999%2Frecords%2FR-1"]
    out = tool.invoke(
        {"type": "tool_call", "id": "x", "name": "entity-record", "args": {"entity": "1000", "record_id": ".."}}
    )
    assert out.status == "error" and len(seen) == 1


def test_서브에이전트는_run의_extra와_더_엄격한_프리셋을_받는다(catalog: Catalog) -> None:
    register_middleware("probe", lambda spec: RepeatedToolCallMiddleware(warn_after=1, stop_after=2))
    try:
        ctx = default_context("records-writer", entity="1000", principal="u-1", middleware=["probe"])
        parent = catalog.manifest("records-writer").model_copy(update={"harness_preset": "strict"})
        specs = build_subagents(parent, catalog, ctx, sink=[], model=resolve_run_model(parent, ctx))
    finally:
        unregister_middleware("probe")
    guide = specs[0]
    names = [type(m).__name__ for m in guide["middleware"]]
    assert names.count("RepeatedToolCallMiddleware") == 2  # the run's extra + the preset's own
    assert "ModelCallLimitMiddleware" in names  # strict from the parent
    recorder = guide["middleware"][0]
    assert recorder.stamp.preset == "strict" and recorder.stamp.extra_middleware == ("probe",)
    assert any("/memories" in p.paths[0] for p in guide["permissions"])


def test_대체한_general_purpose도_기록된다(catalog: Catalog) -> None:
    ctx = default_context("records-guide", entity="1000")
    events: list[dict[str, Any]] = []
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "task", "args": {"description": "요약", "subagent_type": GENERAL_PURPOSE_NAME}, "id": "t"}
            ],
        ),
        AIMessage(content="요약함"),
        AIMessage(content="끝"),
    )
    assemble(catalog.manifest("records-guide"), catalog, ctx, model=model, sink=events).invoke(
        user_turn("x"), context=ctx
    )
    gp_rows = [e for e in events if e["agent_id"] == GENERAL_PURPOSE_NAME]
    assert (
        gp_rows and {e["role"] for e in gp_rows} == {"subagent"} and any(e["event_type"] == "llm.call" for e in gp_rows)
    )


def test_structured_action에는_extra를_붙일_수_없다(catalog: Catalog) -> None:
    ctx = default_context("records-summary", entity="1000")
    action = catalog.action("records-summary")
    with pytest.raises(ValueError, match="structured"):
        build_action_graph(action, catalog, ctx, resolved=resolve_run_model(action, ctx), extra_names=("x",), sink=[])


def test_병렬_도구가_함께_실패해도_run_end는_한_번이다(catalog: Catalog) -> None:
    def broken(operation: Any, payload: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("down")

    events: list[dict[str, Any]] = []
    ctx = default_context("records-guide", entity="1000")
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "record-read", "args": {"period": "2026-08"}, "id": "a"},
                {"name": "view-read", "args": {"view": "v", "period": "2026-08"}, "id": "b"},
            ],
        )
    )
    gateway = ToolGatewayClient({"record-read": broken, "db-read": broken})
    agent = assemble(
        catalog.manifest("records-guide"), catalog, ctx, model=model, clients=ToolClients(gateway=gateway), sink=events
    )
    with pytest.raises(ConnectionError):
        agent.invoke(user_turn("x"), context=ctx)
    assert [e["event_type"] for e in events].count("run.end") == 1


def test_익명_주체에게는_기억을_붙이지_않는다(catalog: Catalog) -> None:
    writer = catalog.manifest("records-writer")
    assert build_backend(writer, default_context("records-writer", principal="anonymous")) is None
    assert build_backend(writer, default_context("records-writer", principal="u-1")) is not None


def test_production_run은_fake_모델을_고를_수_없다(catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIGTN_AGENT_FAKE_MODEL", raising=False)
    manifest = catalog.manifest("records-guide")
    with pytest.raises(ValueError, match="fake"):
        resolve_run_model(manifest, default_context("records-guide", model="models/fake"))
    assert (
        resolve_run_model(manifest, default_context("records-guide", model="models/fake", session_type="test")).provider
        == "fake"
    )


def test_도구_오류와_게이트_차단은_기록에서_구분된다(catalog: Catalog) -> None:
    def failing(operation: Any, payload: dict[str, Any]) -> dict[str, Any]:
        raise ToolException("HTTP 500")

    events: list[dict[str, Any]] = []
    ctx = default_context("records-guide", entity="1000")
    model = scripted_model(_read(1, "2026-08"), AIMessage(content="끝"))
    gateway = ToolGatewayClient({"record-read": failing})
    agent = assemble(
        catalog.manifest("records-guide"), catalog, ctx, model=model, clients=ToolClients(gateway=gateway), sink=events
    )
    agent.copy(update={"checkpointer": InMemorySaver()}).invoke(user_turn("x"), config=thread("t-err"), context=ctx)
    row = next(e["tool"] for e in events if e["event_type"] == "tool.call")
    assert row["allowed"] is True and row["denied_by"] is None and row["status"] == "error"
