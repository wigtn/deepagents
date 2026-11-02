"""Model alias registry (``catalog/models.yaml``) and the lazy model builder.

A manifest names an alias, never a provider model id. A run may switch the alias with
``configurable.model`` so the same harness runs on another backbone. The registry is the only
place that knows provider, endpoint variables and prices:

* an alias that is not registered is refused;
* an alias without a price is refused before the first call (the daily cost cap must be able to
  charge every call), a free endpoint says ``price: {input: 0, output: 0}``;
* manifest parameters the entry does not ``accept`` are dropped instead of sent and rejected.

Models are built lazily, never at import time: the worker runtime imports every registered graph
module at startup, and one missing credential must not take the registry down.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Literal, Optional, cast

import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wigtn_agents.paths import catalog_dir

FAKE_MODEL_ENV = "WIGTN_AGENT_FAKE_MODEL"
MODELS_FILE = "models.yaml"
Provider = Literal["openai", "openai-compatible", "anthropic", "fake"]
# Manifest-level parameters a registry entry may accept.
MANIFEST_PARAMS: tuple[str, ...] = ("temperature", "max_tokens", "reasoning_effort")


class Price(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cached_input: Optional[float] = Field(default=None, ge=0)
    cache_write: Optional[float] = Field(default=None, ge=0)

    @property
    def cached_input_rate(self) -> float:
        """Price of a cache-read input token (defaults to the input price)."""
        return self.input if self.cached_input is None else self.cached_input

    @property
    def cache_write_rate(self) -> float:
        """Price of a cache-write input token (defaults to the input price)."""
        return self.input if self.cache_write is None else self.cache_write


class ModelEntry(BaseModel):
    """One registered alias."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str
    api: Literal["responses", "chat"] = "chat"
    # Endpoint: a fixed public URL (``base_url``) or a variable holding it (``base_url_env``, wins).
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    api_key_env: Optional[str] = None
    price: Optional[Price] = None
    accepts: list[str] = Field(default_factory=lambda: ["temperature", "max_tokens"])
    defaults: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> ModelEntry:
        unknown = [p for p in self.accepts if p not in MANIFEST_PARAMS]
        if unknown:
            raise ValueError(f"accepts에 알 수 없는 파라미터가 있습니다: {unknown}")
        if self.provider == "openai-compatible" and not (self.base_url or self.base_url_env):
            raise ValueError("openai-compatible 모델은 base_url이나 base_url_env가 필요합니다")
        if self.api == "responses" and self.provider != "openai":
            raise ValueError("Responses API는 provider openai에서만 씁니다")
        return self


class ResolvedModel(BaseModel):
    """An alias resolved against the registry; what the event log records."""

    alias: str
    entry: ModelEntry

    @property
    def provider(self) -> str:
        """Provider name."""
        return self.entry.provider

    @property
    def model_id(self) -> str:
        """``provider:model``."""
        return f"{self.entry.provider}:{self.entry.model}"


class ModelRegistry(BaseModel):
    """Every alias the runtime can build."""

    model_config = ConfigDict(extra="forbid")

    default: str
    # Default judge for ``llm`` eval cases; an eval refuses a judge that is the backbone itself.
    judge: Optional[str] = None
    models: dict[str, ModelEntry]

    @model_validator(mode="after")
    def _default_registered(self) -> ModelRegistry:
        for key, alias in (("default", self.default), ("judge", self.judge)):
            if alias is not None and alias not in self.models:
                raise ValueError(f"{key} 별칭이 models에 없습니다: {alias}")
        return self

    @classmethod
    def load(cls, path: Path) -> ModelRegistry:
        """Read ``models.yaml``."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def resolve(self, alias: Optional[str] = None) -> ResolvedModel:
        """Alias → entry. ``None`` is the registry default. Unknown aliases fail loudly."""
        name = alias or self.default
        if name not in self.models:
            raise KeyError(f"등록되지 않은 모델 별칭입니다: {name} (catalog/{MODELS_FILE})")
        return ResolvedModel(alias=name, entry=self.models[name])


_registry_lock = threading.Lock()
_registry_cache: dict[str, Any] = {"key": None, "registry": None}


def load_registry(path: Optional[Path] = None) -> ModelRegistry:
    """The registry from ``catalog/models.yaml``, re-read when the file changes."""
    target = path or catalog_dir() / MODELS_FILE
    key = (str(target), target.stat().st_mtime)
    with _registry_lock:
        if _registry_cache["key"] != key:
            _registry_cache["registry"] = ModelRegistry.load(target)
            _registry_cache["key"] = key
        return _registry_cache["registry"]  # type: ignore[no-any-return]


def fake_forced() -> bool:
    """``WIGTN_AGENT_FAKE_MODEL=1`` swaps every alias for the scripted offline model."""
    return os.environ.get(FAKE_MODEL_ENV, "").lower() in ("1", "true", "yes")


def provider_params(entry: ModelEntry, requested: dict[str, Any]) -> dict[str, Any]:
    """Entry defaults, then the manifest's values for the parameters the entry accepts."""
    params = dict(entry.defaults)
    for name in entry.accepts:
        value = requested.get(name)
        if value is not None:
            params[name] = value
    return params


def _env(name: Optional[str], *, required: bool, alias: str) -> Optional[str]:
    if not name:
        return None
    value = os.environ.get(name, "").strip()
    if not value and required:
        raise LookupError(f"{alias}: 환경 변수 {name}가 비어 있습니다")
    return value or None


def build_model(
    alias: Optional[str] = None,
    *,
    registry: Optional[ModelRegistry] = None,
    **requested: Any,
) -> BaseChatModel:
    """Instantiate the chat model for an alias. Imports the provider lazily.

    Args:
        alias: Registry alias (``None`` = registry default).
        registry: Registry override (tests); ``catalog/models.yaml`` otherwise.
        **requested: Manifest parameters (temperature, max_tokens, reasoning_effort).

    Returns:
        A chat model with the daily cost cap attached.

    Raises:
        KeyError: unregistered alias.
        LookupError: missing price or missing endpoint variable.

    """
    resolved = (registry or load_registry()).resolve(alias)
    entry = resolved.entry
    if fake_forced() or entry.provider == "fake":
        from wigtn_agents.testing import ScriptedFakeModel

        return ScriptedFakeModel()
    if entry.price is None:
        raise LookupError(f"{resolved.alias}: 가격이 등록되지 않았습니다. catalog/{MODELS_FILE}에 price를 적으세요")

    from wigtn_agents.budget import BudgetCallback, budget_from_env

    # Daily cost cap on every real model; the price is known at this point.
    callbacks: list[BaseCallbackHandler] = [BudgetCallback(budget_from_env(), entry.price)]
    params = provider_params(entry, requested)

    if entry.provider in ("openai", "openai-compatible"):
        from langchain_openai import ChatOpenAI

        compatible = entry.provider == "openai-compatible"
        connection: dict[str, Any] = {}
        base_url = _env(entry.base_url_env, required=compatible and not entry.base_url, alias=resolved.alias)
        base_url = base_url or entry.base_url
        if base_url:
            connection["base_url"] = base_url
        # A hosted endpoint (fixed base_url) needs its key; a local vLLM accepts any.
        api_key = _env(entry.api_key_env, required=bool(entry.base_url), alias=resolved.alias)
        if api_key:
            connection["api_key"] = api_key
        elif compatible:
            connection["api_key"] = "unused"  # local vLLM accepts any key; the client insists on one
        # use_responses_api is set explicitly: deepagents only turns it on for string model ids.
        return ChatOpenAI(
            model=entry.model,
            use_responses_api=entry.api == "responses",
            stream_usage=True,
            callbacks=callbacks,
            **connection,
            **params,
        )

    from langchain.chat_models import init_chat_model

    return cast(BaseChatModel, init_chat_model(f"anthropic:{entry.model}", callbacks=callbacks, **params))
