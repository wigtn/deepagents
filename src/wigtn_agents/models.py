"""Model alias registry. Stands in for the platform LLM gateway (``models/default``).

Models are built lazily, never at import time: the worker runtime imports every registered
graph module at startup, and one missing credential must not take the registry down.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL_ENV = "WIGTN_AGENT_DEFAULT_MODEL"
DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

MODEL_ALIASES: dict[str, Optional[str]] = {
    # None → resolved from the environment at call time
    "models/default": None,
    "models/chat": None,
}


def resolve_model_id(ref: str) -> str:
    """Map a manifest ``model.ref`` alias to a ``provider:model`` id."""
    if ref in MODEL_ALIASES:
        return MODEL_ALIASES[ref] or os.environ.get(DEFAULT_MODEL_ENV, DEFAULT_MODEL)
    if ":" in ref:
        return ref
    raise KeyError(f"알 수 없는 모델 별칭입니다: {ref}")


def build_model(ref: str, *, temperature: float = 0.1, max_tokens: int = 4096) -> BaseChatModel:
    """Instantiate the chat model for a manifest. Imports the provider lazily.

    ``WIGTN_AGENT_FAKE_MODEL=1`` swaps in the scripted offline model (runtime integration runs
    without an LLM key). Never set it in a real deployment.
    """
    if os.environ.get("WIGTN_AGENT_FAKE_MODEL", "").lower() in ("1", "true", "yes"):
        from wigtn_agents.testing import ScriptedFakeModel

        return ScriptedFakeModel()

    from langchain.chat_models import init_chat_model

    from wigtn_agents.budget import BudgetCallback, budget_from_env, price_for

    model_id = resolve_model_id(ref)
    # Daily cost cap on every real model; an unknown price fails here, before any call.
    budget = BudgetCallback(budget_from_env(), price_for(model_id))
    return init_chat_model(model_id, temperature=temperature, max_tokens=max_tokens, callbacks=[budget])
