"""Repeated tool-call stop and progress events."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.conftest import scripted_model, user_turn
from wigtn_agents.assembly import assemble
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import RunContext
from wigtn_agents.middleware.repeated import NOTICE_NAME, RepeatedToolCallMiddleware, trailing_repeats


def _read(i: int, entity: str = "1000") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": "record-read", "args": {"period": "2026-08", "entity": entity}, "id": f"c{i}"}]
    )


def _result(i: int) -> ToolMessage:
    return ToolMessage(content="rows", tool_call_id=f"c{i}", name="record-read")


def test_반복_횟수는_다른_호출이_끼면_다시_센다() -> None:
    history = [HumanMessage("x"), _read(1), _result(1), _read(2), _result(2), _read(3, "2000"), _result(3)]
    assert trailing_repeats(history) == (1, "record-read")
    same = [HumanMessage("x"), _read(1), _result(1), _read(2), _result(2), _read(3), _result(3)]
    assert trailing_repeats(same) == (3, "record-read")


def test_같은_호출을_반복하면_한_번_알리고_멈춘다(catalog: Catalog, ctx: RunContext) -> None:
    model = scripted_model(*[_read(i) for i in range(10)])
    agent = assemble(catalog.manifest("records-guide"), catalog, ctx, model=model, sink=[])
    result = agent.invoke(user_turn("점검"), context=ctx)

    notices = [m for m in result["messages"] if getattr(m, "name", None) == NOTICE_NAME]
    calls = [m for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    assert len(notices) == 1
    assert len(calls) == 6
    assert "멈췄습니다" in result["messages"][-1].text


def test_경고_기준은_멈춤_기준보다_작아야_한다() -> None:
    with pytest.raises(ValueError):
        RepeatedToolCallMiddleware(warn_after=6, stop_after=6)


def test_진행_이벤트는_도구_이름과_상태만_싣는다(catalog: Catalog, ctx: RunContext) -> None:
    model = scripted_model(_read(1), AIMessage(content="끝"))
    agent = assemble(catalog.manifest("records-guide"), catalog, ctx, model=model, sink=[])
    chunks: list[Any] = list(agent.stream(user_turn("점검"), context=ctx, stream_mode="custom"))

    tool_events = [c for c in chunks if isinstance(c, dict) and c.get("kind") == "tool"]
    assert [(e["tool"], e["status"]) for e in tool_events] == [("record-read", "running"), ("record-read", "completed")]
    assert all(set(e) == {"type", "kind", "tool", "call_id", "status"} for e in tool_events)


def test_차단된_호출의_진행_이벤트는_error다(catalog: Catalog, ctx: RunContext) -> None:
    model = scripted_model(
        AIMessage(content="", tool_calls=[{"name": "record-write", "args": {"document": {}}, "id": "w"}]),
        AIMessage(content="끝"),
    )
    agent = assemble(catalog.manifest("records-guide"), catalog, ctx, model=model, sink=[])
    chunks = list(agent.stream(user_turn("반영"), context=ctx, stream_mode="custom"))
    statuses = [c["status"] for c in chunks if isinstance(c, dict) and c.get("tool") == "record-write"]
    assert statuses == ["running", "error"]
