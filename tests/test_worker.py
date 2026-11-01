"""Worker factory: schema-only assembly must be shape-identical to a real one and never run."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.conftest import scripted_model, thread, user_turn
from wigtn_agents.assembly import assemble
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import RunContext, default_context
from wigtn_agents.worker import build_run_context, resolve_agent_id, schema_only_graph


@pytest.mark.parametrize("agent_id", ["records-guide", "records-writer", "router"])
def test_스키마_전용_조립은_실제_조립과_상태_채널과_노드가_같다(
    catalog: Catalog, ctx: RunContext, agent_id: str
) -> None:
    real = assemble(catalog.manifest(agent_id), catalog, ctx, model=scripted_model(), sink=[])
    schema_only = schema_only_graph("no agent_id", catalog)

    # The reader must be a SUPERSET: every channel/node a real agent can write must exist here.
    assert set(real.channels) <= set(schema_only.channels), set(real.channels) - set(schema_only.channels)
    assert set(real.nodes) <= set(schema_only.nodes), set(real.nodes) - set(schema_only.nodes)
    assert "HumanInTheLoopMiddleware.after_model" in schema_only.nodes


def test_스키마_전용_조립은_실행되면_즉시_실패한다(catalog: Catalog) -> None:
    graph = schema_only_graph("agent_id 없음", catalog).copy(update={"checkpointer": InMemorySaver()})
    with pytest.raises(RuntimeError, match="agent_id 없음"):
        graph.invoke(user_turn("hi"), config=thread("t-schema"), context=default_context("__schema__"))


def test_스키마_전용_조립으로_실제_스레드의_체크포인트를_읽을_수_있다(catalog: Catalog, ctx: RunContext) -> None:
    # Given: a real agent ran on a thread (same checkpointer, as the runtime injects one per process)
    saver = InMemorySaver()
    real = assemble(
        catalog.manifest("records-guide"), catalog, ctx, model=scripted_model(AIMessage(content="ok")), sink=[]
    ).copy(update={"checkpointer": saver})
    real.invoke(user_turn("8월 점검"), config=thread("t-read"), context=ctx)

    # When: the state is read through the schema-only assembly (what threads.get_state does)
    reader = schema_only_graph("no agent_id", catalog).copy(update={"checkpointer": saver})
    values = reader.get_state(thread("t-read")).values

    # Then
    assert values["asset_versions"]["skill://example/checklist@active"].endswith("@3")
    assert values["messages"][-1].text == "ok"


def test_resolve_agent_id는_configurable을_우선하고_없으면_실패한다() -> None:
    assert resolve_agent_id({"agent_id": "a"}, {"agent_id": "b"}) == "a"
    assert resolve_agent_id({}, {"agent_id": "b"}) == "b"
    with pytest.raises(LookupError):
        resolve_agent_id({}, {})


def test_principal은_langgraph_auth_user에서만_온다() -> None:
    ctx = build_run_context(
        {"agent_id": "records-guide", "user_id": "spoofed", "langgraph_auth_user": {"identity": "real-user"}}, None
    )
    assert ctx["principal"] == "real-user"
    assert build_run_context({"agent_id": "records-guide", "user_id": "spoofed"}, None)["principal"] == "anonymous"
