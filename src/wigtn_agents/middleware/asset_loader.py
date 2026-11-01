"""AssetLoaderMiddleware — pin ``@active`` refs at run start, compose the system prompt per model call.

Differences from the reference sample (which sat on ``create_agent``):

* The composed platform prompt is **prepended** to whatever system message deepagents already
  built (its base prompt carries the todo/file/task tool guidance). Replacing it would silence
  the harness.
* Sync and async hooks are both implemented; the worker runtime executes graphs with ``ainvoke``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import PlatformState
from wigtn_agents.manifest import Manifest

ModelHandler = Callable[[ModelRequest], ModelResponse]
AsyncModelHandler = Callable[[ModelRequest], Awaitable[ModelResponse]]


class AssetLoaderMiddleware(AgentMiddleware):
    """Resolve asset versions once per thread and inject them into every model call."""

    state_schema = PlatformState

    def __init__(self, manifest: Manifest, catalog: Catalog, *, image_version: str = "dev") -> None:
        super().__init__()
        self._manifest = manifest
        self._catalog = catalog
        self._image_version = image_version

    # -- pinning -----------------------------------------------------------------------------

    def _pin_all(self) -> dict[str, str]:
        pinned: dict[str, str] = {}
        for ref in self._manifest.all_asset_refs():
            pinned[ref] = self._catalog.pin(ref)
        for skill_ref in self._manifest.skills:
            skill = self._catalog.resolve(skill_ref)
            for tool_ref in skill.typed_body.tools:  # type: ignore[attr-defined]
                pinned[tool_ref] = self._catalog.pin(tool_ref)
        return pinned

    def before_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Pin once. A resumed run already carries ``asset_versions`` and is left untouched."""
        if state.get("asset_versions"):
            return None
        return {
            "asset_versions": self._pin_all(),
            "run_meta": {
                "agent_id": self._manifest.agent_id,
                "manifest_version": self._manifest.metadata.version,
                "model_id": self._manifest.model.ref,
                "image_version": self._image_version,
            },
        }

    async def abefore_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Async twin of ``before_agent``."""
        return self.before_agent(state, runtime)

    # -- prompt composition ------------------------------------------------------------------

    def _compose_platform_prompt(self, pinned: dict[str, str]) -> str:
        parts: list[str] = []
        for prompt_ref in self._manifest.prompts.values():
            prompt = self._catalog.resolve(pinned.get(prompt_ref, prompt_ref))
            parts.append(prompt.typed_body.template.strip())  # type: ignore[attr-defined]
        for skill_ref in self._manifest.skills:
            skill = self._catalog.resolve(pinned.get(skill_ref, skill_ref))
            body = skill.typed_body
            plan = " → ".join(
                self._catalog.resolve(pinned.get(t, t)).tool_name()
                for t in body.default_plan  # type: ignore[attr-defined]
            )
            parts.append(
                f"## 스킬 · {skill.name} (v{skill.version})\n"
                f"사용 시점: {body.trigger}\n"  # type: ignore[attr-defined]
                f"절차:\n{body.instruction.strip()}\n"  # type: ignore[attr-defined]
                f"도구 기본 순서: {plan or '없음'}"
            )
        return "\n\n".join(parts)

    def _with_platform_prompt(self, request: ModelRequest) -> ModelRequest:
        pinned = request.state.get("asset_versions") or {}
        platform_text = self._compose_platform_prompt(pinned)
        existing = request.system_message
        harness_text = existing.text if existing is not None else ""
        merged = f"{platform_text}\n\n{harness_text}".strip() if harness_text else platform_text
        return request.override(system_message=SystemMessage(content=merged))

    def wrap_model_call(self, request: ModelRequest, handler: ModelHandler) -> ModelResponse:
        """Prepend the pinned prompt + skill instructions to the harness system message."""
        return handler(self._with_platform_prompt(request))

    async def awrap_model_call(self, request: ModelRequest, handler: AsyncModelHandler) -> ModelResponse:
        """Async twin of ``wrap_model_call``."""
        return await handler(self._with_platform_prompt(request))
