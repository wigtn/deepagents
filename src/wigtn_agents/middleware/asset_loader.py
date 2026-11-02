"""AssetLoaderMiddleware — pin ``@active`` refs at run start, compose the system prompt per model call.

Differences from the reference sample (which sat on ``create_agent``):

* The composed platform prompt is **prepended** to whatever system message deepagents already
  built (its base prompt carries the todo/file/task tool guidance). Replacing it would silence
  the harness.
* Sync and async hooks are both implemented; the worker runtime executes graphs with ``ainvoke``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional, cast

import yaml
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import PlatformState
from wigtn_agents.manifest import AnyManifest
from wigtn_agents.models import ResolvedModel

ModelHandler = Callable[[ModelRequest], ModelResponse]
AsyncModelHandler = Callable[[ModelRequest], Awaitable[ModelResponse]]


def pin_assets(manifest: AnyManifest, catalog: Catalog) -> dict[str, str]:
    """``@active`` → integer version for every asset the manifest reaches (skills, prompts, tools)."""
    pinned: dict[str, str] = {}
    for ref in manifest.all_asset_refs():
        pinned[ref] = catalog.pin(ref)
    for skill_ref in manifest.skills:
        skill = catalog.resolve(skill_ref)
        for tool_ref in skill.typed_body.tools:  # type: ignore[attr-defined]
            pinned[tool_ref] = catalog.pin(tool_ref)
    return pinned


def compose_prompt(manifest: AnyManifest, catalog: Catalog, pinned: dict[str, str]) -> str:
    """Prompt assets, then one section per skill (trigger, procedure, default tool order)."""
    parts: list[str] = []
    for prompt_ref in manifest.prompts.values():
        prompt = catalog.resolve(pinned.get(prompt_ref, prompt_ref))
        parts.append(prompt.typed_body.template.strip())  # type: ignore[attr-defined]
    for skill_ref in manifest.skills:
        skill = catalog.resolve(pinned.get(skill_ref, skill_ref))
        body = skill.typed_body
        plan = " → ".join(
            catalog.resolve(pinned.get(t, t)).tool_name()
            for t in body.default_plan  # type: ignore[attr-defined]
        )
        section = (
            f"## 스킬 · {skill.name} (v{skill.version})\n"
            f"사용 시점: {body.trigger}\n"  # type: ignore[attr-defined]
            f"절차:\n{body.instruction.strip()}\n"  # type: ignore[attr-defined]
            f"도구 기본 순서: {plan or '없음'}"
        )
        # Fields a skill adds to its body (a routing table, thresholds) are part of the procedure.
        extra = {k: v for k, v in (body.model_extra or {}).items() if v is not None}
        if extra:
            section += "\n참고 자료:\n" + yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip()
        parts.append(section)
    return "\n\n".join(parts)


class AssetLoaderMiddleware(AgentMiddleware):
    """Resolve asset versions once per thread and inject them into every model call."""

    state_schema = PlatformState

    def __init__(
        self,
        manifest: AnyManifest,
        catalog: Catalog,
        *,
        model: Optional[ResolvedModel] = None,
        image_version: str = "dev",
    ) -> None:
        super().__init__()
        self._manifest = manifest
        self._catalog = catalog
        self._model = model
        self._image_version = image_version

    def before_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """Pin once. A resumed run already carries ``asset_versions`` and is left untouched."""
        if state.get("asset_versions"):
            return None
        return {
            "asset_versions": pin_assets(self._manifest, self._catalog),
            "run_meta": {
                "agent_id": self._manifest.agent_id,
                "manifest_version": self._manifest.metadata.version,
                "model_alias": self._model.alias if self._model else self._manifest.model.ref,
                "model_id": self._model.model_id if self._model else None,
                "image_version": self._image_version,
            },
        }

    async def abefore_agent(self, state: PlatformState, runtime: Runtime) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """Async twin of ``before_agent``."""
        return self.before_agent(state, runtime)

    def _with_platform_prompt(self, request: ModelRequest) -> ModelRequest:
        pinned = cast(dict[str, str], request.state.get("asset_versions") or {})
        platform_text = compose_prompt(self._manifest, self._catalog, pinned)
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
