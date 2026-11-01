"""Presets, the extra-middleware slot, memory, and the schema-only reader as their superset."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import pytest
from langchain.agents.middleware import AgentMiddleware, AgentState, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from tests.conftest import scripted_model, thread, user_turn
from wigtn_agents.assembly import assemble, memory_namespace
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import RunContext, default_context
from wigtn_agents.manifest import Manifest
from wigtn_agents.middleware.extra import MiddlewareSpec, register_middleware, unregister_middleware
from wigtn_agents.worker import schema_only_graph


class TagToolCalls(AgentMiddleware):
    """A research middleware stand-in: notes every tool call it sees."""

    seen: list[str] = []

    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        TagToolCalls.seen.append(request.tool_call["name"])
        return handler(request)


class CountsTurns(AgentMiddleware):
    """An extra middleware with a node-creating hook (before_model)."""

    def before_model(self, state: AgentState, runtime: Runtime) -> Optional[dict[str, Any]]:
        return None


@pytest.fixture
def extras() -> Iterator[None]:
    TagToolCalls.seen = []
    register_middleware("tag-tools", lambda spec: TagToolCalls())
    register_middleware("counts-turns", lambda spec: CountsTurns())
    yield
    unregister_middleware("tag-tools")
    unregister_middleware("counts-turns")


def _preset(catalog: Catalog, agent_id: str, preset: str, **update: Any) -> Manifest:
    return catalog.manifest(agent_id).model_copy(update={"harness_preset": preset, **update})


def _middleware_names(graph: Any) -> list[str]:
    return [n.split(".")[0] for n in graph.nodes]


def test_extra_미들웨어는_기록기_바로_안쪽에_붙고_기록에_이름이_남는다(
    catalog: Catalog, extras: None, events: list[dict[str, Any]]
) -> None:
    ctx = default_context("records-guide", principal="u-1", entity="1000", middleware=["tag-tools"])
    model = scripted_model(
        AIMessage(content="", tool_calls=[{"name": "record-read", "args": {"period": "2026-08"}, "id": "c1"}]),
        AIMessage(content="끝"),
    )
    agent = assemble(catalog.manifest("records-guide"), catalog, ctx, model=model, sink=events)
    agent.invoke(user_turn("점검"), context=ctx)

    assert TagToolCalls.seen == ["record-read"]
    assert {tuple(e["extra_middleware"]) for e in events} == {("tag-tools",)}


def test_등록되지_않은_미들웨어_이름은_조립에서_실패한다(catalog: Catalog) -> None:
    ctx = default_context("records-guide", middleware=["not-installed"])
    with pytest.raises(KeyError, match="not-installed"):
        assemble(catalog.manifest("records-guide"), catalog, ctx, model=scripted_model(), sink=[])


def test_strict는_파일_쓰기를_막는다(catalog: Catalog, ctx: RunContext) -> None:
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[{"name": "write_file", "args": {"file_path": "/notes.md", "content": "x"}, "id": "w1"}],
        ),
        AIMessage(content="끝"),
    )
    agent = assemble(_preset(catalog, "records-guide", "strict"), catalog, ctx, model=model, sink=[])
    result = agent.invoke(user_turn("메모 남겨"), context=ctx)
    denied = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "write_file"]
    assert denied[0].status == "error" and "permission denied" in denied[0].text
    assert "/notes.md" not in (result.get("files") or {})


def test_observe는_권한_필터_없이_기록만_한다(catalog: Catalog, ctx: RunContext, events: list[dict[str, Any]]) -> None:
    model = scripted_model(
        AIMessage(
            content="", tool_calls=[{"name": "record-read", "args": {"period": "2026-08", "entity": "9"}, "id": "c"}]
        ),
        AIMessage(content="끝"),
    )
    agent = assemble(_preset(catalog, "records-guide", "observe"), catalog, ctx, model=model, sink=events)
    agent.invoke(user_turn("점검"), context=ctx)
    names = _middleware_names(agent)
    assert "ScopeGuardMiddleware" not in names and "RepeatedToolCallMiddleware" not in names
    assert [e["tool"]["name"] for e in events if e["event_type"] == "tool.call"] == ["record-read"]
    assert {e["preset"] for e in events} == {"observe"}


def test_memory는_주체별_영속_경로를_마운트한다(catalog: Catalog) -> None:
    ctx = default_context("records-writer", principal="u-7", entity="1000")
    store = InMemoryStore()
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "write_file", "args": {"file_path": "/memories/AGENTS.md", "content": "선호: 표"}, "id": "m"}
            ],
        ),
        AIMessage(content="기억함"),
    )
    agent = assemble(catalog.manifest("records-writer"), catalog, ctx, model=model, sink=[]).copy(
        update={"checkpointer": InMemorySaver(), "store": store}
    )
    agent.invoke(user_turn("앞으로 표로 정리해"), config=thread("t-mem"), context=ctx)
    namespace = memory_namespace(catalog.manifest("records-writer"), ctx)
    assert namespace == ("memories", "records-writer", "u-7")
    assert store.get(namespace, "/AGENTS.md") is not None


@pytest.mark.parametrize("preset", ["observe", "standard", "strict"])
@pytest.mark.parametrize("agent_id", ["records-guide", "records-writer", "router"])
def test_스키마_전용_조립은_모든_프리셋과_extra의_상위집합이다(
    catalog: Catalog, ctx: RunContext, extras: None, agent_id: str, preset: str
) -> None:
    run_ctx = default_context(agent_id, entity="1000", middleware=["counts-turns"])
    real = assemble(_preset(catalog, agent_id, preset), catalog, run_ctx, model=scripted_model(), sink=[])
    reader = schema_only_graph("no agent_id", catalog)
    assert set(real.nodes) <= set(reader.nodes), set(real.nodes) - set(reader.nodes)
    assert set(real.channels) <= set(reader.channels), set(real.channels) - set(reader.channels)


def test_extra_factory는_스키마_전용일_때_그_사실을_받는다(catalog: Catalog) -> None:
    specs: list[MiddlewareSpec] = []

    def factory(spec: MiddlewareSpec) -> AgentMiddleware:
        specs.append(spec)
        return CountsTurns()

    register_middleware("spy", factory)
    try:
        schema_only_graph("x", catalog)
    finally:
        unregister_middleware("spy")
    assert specs and specs[0].schema_only is True
