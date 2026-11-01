"""Daily model cost cap (10 USD a day by default).

A callback on the real chat model checks the day's spend before every call and refuses the
call once the cap is reached; after every call it adds the cost of the tokens the provider
reported. The run then ends with ``CostCapReachedError`` and the gap file gets one
``cost_cap`` line per day, so the time without runs is never read as a quiet stretch.

Spend is kept per UTC day in a JSON file under a file lock, so several workers share it.
A model without a known price is refused before any call (fail closed).
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

CAP_ENV = "WIGTN_AGENT_DAILY_COST_CAP_USD"
LEDGER_ENV = "WIGTN_AGENT_COST_LEDGER"
GAPS_ENV = "WIGTN_AGENT_GAP_LOG"
PRICE_IN_ENV = "WIGTN_AGENT_PRICE_INPUT_PER_MTOK"
PRICE_OUT_ENV = "WIGTN_AGENT_PRICE_OUTPUT_PER_MTOK"
DEFAULT_CAP_USD: Final = 10.0
# USD per million tokens (input, output), Anthropic first-party rates as of 2026-10.
PRICES: Final[dict[str, tuple[float, float]]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE_READ: Final = 0.1  # of the input price
CACHE_WRITE: Final = 1.25  # of the input price


class CostCapReachedError(RuntimeError):
    """The day's model spend reached the cap; no more model calls until the next UTC day."""


def price_for(model_id: str) -> tuple[float, float]:
    """(input, output) USD per million tokens; environment overrides win, unknown fails."""
    if os.environ.get(PRICE_IN_ENV) and os.environ.get(PRICE_OUT_ENV):
        return float(os.environ[PRICE_IN_ENV]), float(os.environ[PRICE_OUT_ENV])
    name = model_id.split(":", 1)[-1]
    if name not in PRICES:
        raise LookupError(f"no price for {name}; set {PRICE_IN_ENV} and {PRICE_OUT_ENV}")
    return PRICES[name]


def cost_of(usage: dict[str, Any], prices: tuple[float, float]) -> float:
    """USD for one call from LangChain ``usage_metadata``."""
    price_in, price_out = prices
    details = usage.get("input_token_details") or {}
    read = int(details.get("cache_read") or 0)
    write = int(details.get("cache_creation") or 0)
    fresh = max(int(usage.get("input_tokens") or 0) - read - write, 0)
    output = int(usage.get("output_tokens") or 0)
    dollars = fresh * price_in + read * price_in * CACHE_READ + write * price_in * CACHE_WRITE
    return (dollars + output * price_out) / 1_000_000


class DailyBudget:
    """The day's spend against the cap, shared by every worker through one file."""

    def __init__(
        self,
        cap_usd: float,
        ledger: Path,
        gaps: Path,
        *,
        today: Callable[[], date] = lambda: datetime.now(UTC).date(),
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.cap_usd = cap_usd
        self._ledger = ledger
        self._gaps = gaps
        self._today = today
        self._now = now

    def _update(self, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        self._ledger.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            text = handle.read()
            state: dict[str, Any] = json.loads(text) if text.strip() else {}
            day = self._today().isoformat()
            if state.get("day") != day:
                state = {"day": day, "spent_usd": 0.0, "cap_noted": False}
            change(state)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(state))
            return state

    def spent(self) -> float:
        """USD spent so far today (UTC)."""
        return float(self._update(lambda state: None)["spent_usd"])

    def charge(self, usd: float) -> None:
        """Add the cost of one finished call."""

        def add(state: dict[str, Any]) -> None:
            state["spent_usd"] = float(state["spent_usd"]) + usd

        self._update(add)

    def check(self) -> None:
        """Raise when the cap is reached; the first refusal of the day writes a gap line."""
        noted = {"first": False}

        def mark(state: dict[str, Any]) -> None:
            if float(state["spent_usd"]) >= self.cap_usd and not state["cap_noted"]:
                state["cap_noted"] = noted["first"] = True

        state = self._update(mark)
        if noted["first"]:
            self._gaps.parent.mkdir(parents=True, exist_ok=True)
            with self._gaps.open("a", encoding="utf-8") as handle:
                line = {"at": self._now().isoformat(), "reason": "cost_cap", "cap_usd": self.cap_usd}
                handle.write(json.dumps(line) + "\n")
        if float(state["spent_usd"]) >= self.cap_usd:
            raise CostCapReachedError(f"daily model cost cap of {self.cap_usd} USD reached")


def budget_from_env() -> DailyBudget:
    """The budget configured by the environment (cap, ledger file, gap file)."""
    return DailyBudget(
        float(os.environ.get(CAP_ENV, DEFAULT_CAP_USD)),
        Path(os.environ.get(LEDGER_ENV, "var/cost.json")),
        Path(os.environ.get(GAPS_ENV, "var/gaps.jsonl")),
    )


class BudgetCallback(BaseCallbackHandler):
    """Refuses a model call over the cap and charges each finished call."""

    raise_error = True  # an exception here stops the call instead of being logged

    def __init__(self, budget: DailyBudget, prices: tuple[float, float]) -> None:
        self._budget = budget
        self._prices = prices

    def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Refuse the call when the spend of today reached the cap."""
        self._budget.check()

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> None:
        """Charge the tokens the provider reported for this call."""
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self._budget.charge(cost_of(dict(usage), self._prices))
