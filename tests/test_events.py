"""Run events: the real end of a run."""

from __future__ import annotations

from typing import Any, Optional

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from tests.conftest import scripted_model, thread, user_turn
from wigtn_agents.assembly import assemble
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import RunContext, default_context
from wigtn_agents.tools.gateway import ToolGatewayClient


class BrokenModel(GenericFakeChatModel):
    """A provider that fails; the message carries text that must not reach the events."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> BrokenModel:  # type: ignore[override]
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        raise RuntimeError("upstream refused: user kim@example.com")


def ends(events: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    return [
        (e["event_type"], e.get("status") or e.get("error_class"))
        for e in events
        if e["event_type"] in ("run.error", "run.end")
    ]


def test_a_failed_run_leaves_an_error_and_an_error_end(
    catalog: Catalog, ctx: RunContext, events: list[dict[str, Any]]
) -> None:
    agent = assemble(catalog.manifest("records-guide"), catalog, ctx, model=BrokenModel(messages=iter([])), sink=events)
    with pytest.raises(RuntimeError):
        agent.invoke(user_turn("8월 기록 점검"), context=ctx)
    assert ends(events) == [("run.error", "RuntimeError"), ("run.end", "error")]
    assert "kim@example.com" not in repr(events)


def test_a_run_waiting_for_approval_ends_as_interrupted(
    catalog: Catalog, gateway: ToolGatewayClient, events: list[dict[str, Any]]
) -> None:
    ctx = default_context("records-writer", principal="u-1", entity="1000")
    call = {"name": "record-write", "args": {"document": {"amt": 1}, "entity": "1000"}, "id": "c1"}
    model = scripted_model(AIMessage(content="", tool_calls=[call]))
    agent = assemble(catalog.manifest("records-writer"), catalog, ctx, model=model, gateway=gateway, sink=events)
    agent.checkpointer = InMemorySaver()
    first = agent.invoke(user_turn("기록 수정 반영"), config=thread("t-wait"), context=ctx)
    assert "__interrupt__" in first
    (end,) = [e for e in events if e["event_type"] == "run.end"]
    assert (end["status"], end["waiting_for"]) == ("interrupted", ["record-write"])
