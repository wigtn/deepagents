"""Action lane: gather (code) → reason (model, three levels) → writeback (code)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.conftest import scripted_model, thread
from wigtn_agents.actions import ACTION_NODES, build_action_graph, register_gatherer, schema_only_action_graph
from wigtn_agents.assembly import resolve_run_model
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import RunContext, default_context
from wigtn_agents.manifest import ActionManifest
from wigtn_agents.testing import ScriptedFakeModel
from wigtn_agents.tools.gateway import ToolClients, ToolGatewayClient
from wigtn_agents.worker import action as action_factory


@pytest.fixture
def action_ctx() -> RunContext:
    return default_context("records-summary", principal="u-1", entity="1000", batch_id="b-1")


@pytest.fixture(autouse=True)
def _action_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "actions.jsonl"
    monkeypatch.setenv("WIGTN_AGENT_ACTION_LOG", str(path))
    return path


def _graph(action: ActionManifest, catalog: Catalog, ctx: RunContext, **kwargs: Any) -> Any:
    return build_action_graph(action, catalog, ctx, resolved=resolve_run_model(action, ctx), **kwargs)


def _result_call(**args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "ActionResult", "args": args, "id": "res"}])


async def test_structured_action은_코드가_모은_입력으로_한_번_판단하고_port가_저장한다(
    catalog: Catalog, action_ctx: RunContext, events: list[dict[str, Any]], _action_log: Path
) -> None:
    # Given: default-plan gather calls record-read then view-read; the input tries another scope
    gateway = ToolGatewayClient()
    model = scripted_model(_result_call(summary="미처리 2건", open_items=2, next_steps=["확인"]))
    graph = _graph(
        catalog.action("records-summary"),
        catalog,
        action_ctx,
        model=model,
        clients=ToolClients(gateway=gateway),
        sink=events,
    ).copy(update={"checkpointer": InMemorySaver()})

    # When
    state = await graph.ainvoke(
        {"input": {"period": "2026-08", "view": "open-items", "entity": "9999"}}, config=thread("a-1")
    )

    # Then
    assert set(state["gathered"]) == {"record-read", "view-read"}
    assert [c["payload"]["entity"] for c in gateway.calls] == ["1000", "1000"]
    assert state["result"] == {"summary": "미처리 2건", "open_items": 2, "next_steps": ["확인"]}
    assert state["writeback"]["state"] == "written"
    assert state["asset_versions"]["skill://example/checklist@active"].endswith("@3")
    row = json.loads(_action_log.read_text(encoding="utf-8").splitlines()[0])
    assert row["action"] == "records-summary" and row["batch_id"] == "b-1" and row["result"]["open_items"] == 2
    assert [e["event_type"] for e in events] == ["run.start", "tool.call", "tool.call", "llm.call", "run.end"]
    gather_rows = [e for e in events if e["event_type"] == "tool.call"]
    assert [(e["step"], e["tool"]["name"], e["tool"]["status"]) for e in gather_rows] == [
        ("gather", "record-read", "ok"),
        ("gather", "view-read", "ok"),
    ]
    assert state["gathered"]["record-read"]["count"] == 2  # the tool's dict, not its text
    assert {e["kind"] for e in events} == {"action"} and events[-1]["writeback"] == "written"
    assert events[0]["model_alias"] == "models/default" and events[0]["model_id"] == "openai:gpt-6-luna"


async def test_bounded_react는_읽기_도구를_쓰고_형식대로_답한다(catalog: Catalog, events: list[dict[str, Any]]) -> None:
    ctx = default_context("records-review", entity="1000")
    model = scripted_model(
        AIMessage(content="", tool_calls=[{"name": "record-read", "args": {"period": "2026-08"}, "id": "r1"}]),
        _result_call(verdict="needs-review", reasons=["R-0001 금액 큼"]),
    )
    graph = _graph(catalog.action("records-review"), catalog, ctx, model=model, sink=events)
    state = await graph.ainvoke({"input": {"period": "2026-08"}})

    assert state["result"]["verdict"] == "needs-review"
    assert state["writeback"] == {"state": "skipped"}
    tool_rows = [e for e in events if e["event_type"] == "tool.call"]
    # the structured-response tool is handled by the agent itself, not as a tool call
    assert [r["tool"]["name"] for r in tool_rows] == ["record-read"]
    assert [e["event_type"] for e in events].count("run.start") == 1  # the loop does not restart the run


async def test_scoped_agent도_같은_계약으로_끝난다(catalog: Catalog) -> None:
    ctx = default_context("records-review", entity="1000")
    action = catalog.action("records-review").model_copy(update={"level": "scoped-agent"})
    model = scripted_model(_result_call(verdict="ok", reasons=[]))
    state = await _graph(action, catalog, ctx, model=model, sink=[]).ainvoke({"input": {}})
    assert state["result"] == {"verdict": "ok", "reasons": []}


@pytest.mark.parametrize("level", ["structured", "bounded-react", "scoped-agent"])
async def test_오프라인_fake_모델로_세_수준이_모두_돈다(catalog: Catalog, level: str) -> None:
    ctx = default_context("records-review", entity="1000")
    action = catalog.action("records-review").model_copy(update={"level": level})
    state = await _graph(action, catalog, ctx, model=ScriptedFakeModel(), sink=[]).ainvoke(
        {"input": {"period": "2026-08"}}
    )
    assert state["result"]["verdict"] == "(fake)"


def test_쓰기_도구에_닿는_action은_조립하지_않는다(catalog: Catalog, action_ctx: RunContext) -> None:
    action = catalog.action("records-summary").model_copy(update={"skill": "skill://example/record-update@active"})
    with pytest.raises(ValueError, match="writeback port"):
        _graph(action, catalog, action_ctx, model=scripted_model(), sink=[])


def test_등록되지_않은_port는_조립에서_실패한다(catalog: Catalog, action_ctx: RunContext) -> None:
    action = catalog.action("records-summary").model_copy(update={"gather": "nowhere"})
    with pytest.raises(KeyError, match="nowhere"):
        _graph(action, catalog, action_ctx, model=scripted_model(), sink=[])


async def test_gather가_실패하면_오류와_종료가_남고_run이_실패한다(
    catalog: Catalog, action_ctx: RunContext, events: list[dict[str, Any]]
) -> None:
    def broken(request: Any) -> dict[str, Any]:
        raise ConnectionError("source down: user kim@example.com")

    register_gatherer("broken-source", broken)
    action = catalog.action("records-summary").model_copy(update={"gather": "broken-source"})
    graph = _graph(action, catalog, action_ctx, model=scripted_model(), sink=events)
    with pytest.raises(ConnectionError):
        await graph.ainvoke({"input": {}})
    assert [(e["event_type"], e.get("status") or e.get("error_class")) for e in events] == [
        ("run.start", None),
        ("run.error", "ConnectionError"),
        ("run.end", "error"),
    ]
    assert events[1]["step"] == "gather"
    assert "kim@example.com" not in json.dumps(events, ensure_ascii=False)


def test_action_팩토리는_agent_id가_없으면_같은_모양의_읽기_전용_그래프를_준다(catalog: Catalog) -> None:
    reader = action_factory({"configurable": {"thread_id": "t"}}, None)  # type: ignore[arg-type]
    assert [n for n in reader.nodes if not n.startswith("__")] == list(ACTION_NODES)
    real = action_factory({"configurable": {"agent_id": "records-summary", "entity": "1000"}}, None)  # type: ignore[arg-type]
    assert set(real.nodes) == set(reader.nodes)
    assert set(real.channels) <= set(schema_only_action_graph("x").channels)


def test_action_팩토리는_run의_모델_별칭을_따른다(catalog: Catalog) -> None:
    with pytest.raises(KeyError, match="models/nope"):
        action_factory({"configurable": {"agent_id": "records-summary", "model": "models/nope"}}, None)  # type: ignore[arg-type]
