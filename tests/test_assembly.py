"""Assembly on top of create_deep_agent: pinning, events, prompt composition, permission, HITL."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.conftest import CATALOG_DIR, scripted_model, thread, user_turn
from wigtn_agents.assembly import assemble
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import Asset
from wigtn_agents.context import RunContext, default_context
from wigtn_agents.manifest import Manifest
from wigtn_agents.tools.gateway import ToolGatewayClient


def _with_memory(graph: Any) -> Any:
    """Inject a checkpointer the way the worker runtime does (graph.copy), never via assembly."""
    return graph.copy(update={"checkpointer": InMemorySaver()})


def test_write_도구가_있는_react는_조립_실패(catalog: Catalog, ctx: RunContext) -> None:
    data = yaml.safe_load((CATALOG_DIR / "manifests" / "records-writer.yaml").read_text(encoding="utf-8"))
    data["flow"] = {"template": "react"}
    data.pop("interrupts")
    manifest = Manifest.model_validate(data)
    with pytest.raises(ValueError):
        assemble(manifest, catalog, ctx, model=scripted_model(), sink=[])


def test_react_1턴_실행이_버전을_고정하고_이벤트를_남긴다(
    catalog: Catalog, ctx: RunContext, gateway: ToolGatewayClient, events: list[dict[str, Any]]
) -> None:
    # Given
    manifest = catalog.manifest("records-guide")
    model = scripted_model(AIMessage(content="미처리 기록 12건 · 확인 대상 3건"))
    agent = _with_memory(assemble(manifest, catalog, ctx, model=model, gateway=gateway, sink=events))

    # When
    result = agent.invoke(user_turn("8월 기록 점검 (범위 1000)"), config=thread("t-1"), context=ctx)

    # Then: pinned versions live in thread state, events carry the stamps
    assert result["asset_versions"]["skill://example/checklist@active"] == "skill://example/checklist@3"
    assert result["asset_versions"]["prompt://records/system@active"] == "prompt://records/system@5"
    types = [e["event_type"] for e in events]
    assert types[0] == "run.start" and types[-1] == "run.end" and "llm.call" in types
    run_end = events[-1]
    assert run_end["manifest_version"] == 2 and run_end["actor_id"] == "u-1" and run_end["session_type"] == "production"
    assert run_end["asset_versions"]["skill://example/checklist@active"].endswith("@3")
    assert "미처리 기록" in run_end["outcome"]


def _system_text(model: Any) -> str:
    first = model.seen_messages[0][0]
    return first.text if isinstance(first, SystemMessage) else ""


def test_시스템_프롬프트는_플랫폼_자산을_앞에_붙이고_하네스_지시를_보존한다(
    catalog: Catalog, ctx: RunContext, events: list[dict[str, Any]]
) -> None:
    # Given: baseline = what deepagents itself sends without any platform middleware
    from deepagents import create_deep_agent

    baseline_model = scripted_model(AIMessage(content="ok"))
    create_deep_agent(model=baseline_model, subagents=[]).invoke(user_turn("안녕"))
    harness_text = _system_text(baseline_model)

    model = scripted_model(AIMessage(content="ok"))
    agent = _with_memory(assemble(catalog.manifest("records-guide"), catalog, ctx, model=model, sink=events))

    # When
    agent.invoke(user_turn("안녕"), config=thread("t-prompt"), context=ctx)

    # Then: platform text first, harness text (whatever the installed version emits) intact after it
    text = _system_text(model)
    assert text.startswith("당신은 기록 관리 담당자를 돕는 에이전트입니다.")
    assert "## 스킬 · example/checklist (v3)" in text
    if harness_text.strip():
        assert text.endswith(harness_text.strip()), "deepagents 하네스 프롬프트가 잘렸다"


def test_모델에_바인딩된_도구는_전부_허용_목록_안에_있다(
    catalog: Catalog, ctx: RunContext, events: list[dict[str, Any]]
) -> None:
    from wigtn_agents.middleware import DEEPAGENTS_BUILTIN_TOOLS

    manifest = catalog.manifest("records-guide")
    model = scripted_model(AIMessage(content="ok"))
    agent = _with_memory(assemble(manifest, catalog, ctx, model=model, sink=events))
    agent.invoke(user_turn("안녕"), config=thread("t-tools"), context=ctx)

    bound = {name for names in model.bound_tool_names for name in names}
    manifest_tools = {catalog.resolve(r).tool_name() for r in manifest.permissions.allow_tools}
    assert bound, "도구가 바인딩되지 않았다"
    assert bound <= (manifest_tools | DEEPAGENTS_BUILTIN_TOOLS), bound - (manifest_tools | DEEPAGENTS_BUILTIN_TOOLS)
    assert {"read_file", "edit_file", "task"} <= bound, "deepagents 내장 툴이 없다"
    assert "record-write" not in bound


def test_허용_목록_밖_도구_호출은_차단되고_이벤트에_남는다(
    catalog: Catalog, ctx: RunContext, gateway: ToolGatewayClient, events: list[dict[str, Any]]
) -> None:
    # Given: records-guide(read only) 인데 모델이 record-write 를 부른다
    model = scripted_model(
        AIMessage(
            content="", tool_calls=[{"name": "record-write", "args": {"document": {}, "entity": "9"}, "id": "c1"}]
        ),
        AIMessage(content="차단되어 중단합니다"),
    )
    agent = _with_memory(
        assemble(catalog.manifest("records-guide"), catalog, ctx, model=model, gateway=gateway, sink=events)
    )

    # When
    result = agent.invoke(user_turn("반영해줘"), config=thread("t-deny"), context=ctx)

    # Then
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and tool_msgs[0].status == "error" and tool_msgs[0].content.startswith("차단")
    assert gateway.calls == [], "게이트웨이까지 도달하면 안 된다"
    tool_events = [e for e in events if e["event_type"] == "tool.call"]
    assert tool_events and tool_events[0]["tool"] == {
        "name": "record-write",
        "allowed": False,
        "denied_by": "permission",
        "status": "error",
        "latency_ms": tool_events[0]["tool"]["latency_ms"],
    }


def test_스코프_가드는_LLM이_준_entity를_컨텍스트_정본으로_덮어쓴다(
    catalog: Catalog, ctx: RunContext, gateway: ToolGatewayClient, events: list[dict[str, Any]]
) -> None:
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[{"name": "record-read", "args": {"period": "2026-08", "entity": "9999"}, "id": "c1"}],
        ),
        AIMessage(content="조회 완료"),
    )
    agent = _with_memory(
        assemble(catalog.manifest("records-guide"), catalog, ctx, model=model, gateway=gateway, sink=events)
    )

    agent.invoke(user_turn("8월 조회"), config=thread("t-scope"), context=ctx)

    assert gateway.calls[0]["payload"]["entity"] == "1000", gateway.calls
    # 원본 tool_call 은 히스토리에 그대로 남는다 (감사)
    state = agent.get_state(thread("t-scope")).values
    ai = [m for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls][0]
    assert ai.tool_calls[0]["args"]["entity"] == "9999"


def test_스코프_정본이_없으면_거부한다(
    catalog: Catalog, gateway: ToolGatewayClient, events: list[dict[str, Any]]
) -> None:
    ctx_no_entity = default_context("records-guide", principal="u-1")  # entity=None
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[{"name": "record-read", "args": {"period": "2026-08", "entity": "9999"}, "id": "c1"}],
        ),
        AIMessage(content="끝"),
    )
    agent = _with_memory(
        assemble(catalog.manifest("records-guide"), catalog, ctx_no_entity, model=model, gateway=gateway, sink=events)
    )

    result = agent.invoke(user_turn("조회"), config=thread("t-noscope"), context=ctx_no_entity)

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error" and "스코프 정본 없음" in tool_msgs[0].content
    assert gateway.calls == []


def test_재개는_고정된_버전을_유지한다(catalog: Catalog, ctx: RunContext, events: list[dict[str, Any]]) -> None:
    # Given: 첫 실행에서 skill v3 로 고정
    manifest = catalog.manifest("records-guide")
    model = scripted_model(AIMessage(content="1차"), AIMessage(content="2차"))
    agent = _with_memory(assemble(manifest, catalog, ctx, model=model, sink=events))
    agent.invoke(user_turn("시작"), config=thread("t-pin"), context=ctx)

    # When: 대기 중 skill v4 가 활성이 되고 같은 thread 에서 이어서 실행
    v3 = catalog.resolve("skill://example/checklist@3")
    v4 = v3.model_dump(mode="json")
    v4["version"] = 4
    catalog.add(Asset.model_validate(v4))
    catalog.set_active("skill", "example/checklist", 4)
    result = agent.invoke(user_turn("이어서"), config=thread("t-pin"), context=ctx)

    # Then: 실행 중인 thread 는 v3, 새 thread 는 v4
    assert result["asset_versions"]["skill://example/checklist@active"].endswith("@3")
    assert catalog.pin("skill://example/checklist@active").endswith("@4")


def test_approval_gated는_write_도구_앞에서_멈추고_승인_후_이어간다(
    catalog: Catalog, gateway: ToolGatewayClient, events: list[dict[str, Any]]
) -> None:
    # Given
    ctx = default_context("records-writer", principal="u-1", entity="1000")
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[{"name": "record-write", "args": {"document": {"amt": 1}, "entity": "1000"}, "id": "c1"}],
        ),
        AIMessage(content="기록 R-0077 반영 완료"),
    )
    agent = _with_memory(
        assemble(catalog.manifest("records-writer"), catalog, ctx, model=model, gateway=gateway, sink=events)
    )

    # When: 1차 실행은 interrupt 로 멈춘다
    first = agent.invoke(user_turn("기록 수정 반영"), config=thread("t-hitl"), context=ctx)

    # Then
    assert "__interrupt__" in first, first.keys()
    assert gateway.calls == [], "승인 전에는 게이트웨이를 부르면 안 된다"

    # When: 승인 후 재개
    resumed = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=thread("t-hitl"), context=ctx)

    # Then
    assert gateway.calls and gateway.calls[0]["route"] == "record-write"
    assert "R-0077" in resumed["messages"][-1].text
    assert resumed["asset_versions"]["tool://record-write@active"].endswith("@1")
