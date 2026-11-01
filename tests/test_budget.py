"""Daily model cost cap."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from wigtn_agents import budget as b


def make(tmp_path: Path, day: list[date], cap: float = 1.0) -> b.DailyBudget:
    return b.DailyBudget(
        cap,
        tmp_path / "cost.json",
        tmp_path / "gaps.jsonl",
        today=lambda: day[0],
        now=lambda: datetime.combine(day[0], datetime.min.time(), UTC),
    )


def test_cost_counts_fresh_cached_and_output_tokens() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "input_token_details": {"cache_read": 400_000, "cache_creation": 100_000},
    }
    # 500k fresh x 3 + 400k read x 0.3 + 100k written x 3.75 + 100k out x 15, per million
    assert b.cost_of(usage, (3.0, 15.0)) == pytest.approx(1.5 + 0.12 + 0.375 + 1.5)


def test_the_cap_stops_calls_writes_one_gap_line_a_day_and_resets(tmp_path: Path) -> None:
    day = [date(2026, 11, 1)]
    budget = make(tmp_path, day)
    budget.check()
    budget.charge(0.6)
    budget.charge(0.5)
    for _ in range(3):
        with pytest.raises(b.CostCapReachedError):
            budget.check()
    gaps = [json.loads(line) for line in (tmp_path / "gaps.jsonl").read_text().splitlines()]
    assert [g["reason"] for g in gaps] == ["cost_cap"]
    day[0] = date(2026, 11, 2)
    budget.check()
    assert budget.spent() == 0.0


def test_an_unknown_model_has_no_price_unless_one_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    assert b.price_for("anthropic:claude-sonnet-4-6") == (3.0, 15.0)
    with pytest.raises(LookupError, match="no price"):
        b.price_for("anthropic:some-new-model")
    monkeypatch.setenv(b.PRICE_IN_ENV, "1")
    monkeypatch.setenv(b.PRICE_OUT_ENV, "2")
    assert b.price_for("anthropic:some-new-model") == (1.0, 2.0)


def test_the_callback_charges_a_call_and_refuses_the_next_over_the_cap(tmp_path: Path) -> None:
    budget = make(tmp_path, [date(2026, 11, 1)], cap=0.01)
    reply = AIMessage(content="ok", usage_metadata={"input_tokens": 1000, "output_tokens": 1000, "total_tokens": 2000})
    model = GenericFakeChatModel(messages=iter([reply, reply]), callbacks=[b.BudgetCallback(budget, (3.0, 15.0))])
    model.invoke("hello")
    assert budget.spent() == pytest.approx(0.018)
    with pytest.raises(b.CostCapReachedError):
        model.invoke("hello again")
