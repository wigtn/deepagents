"""카탈로그. 자산 YAML 로드 · 참조 해석 · 활성 버전 포인터."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import yaml

from wigtn_agents.assets.schema import Asset, AssetKind, AssetRef, AssetStatus, EvalCase
from wigtn_agents.manifest import ActionManifest, AnyManifest, Manifest


class Catalog:
    """자산과 매니페스트의 정본 저장소(인메모리 샘플)."""

    def __init__(self) -> None:
        self._assets: dict[tuple[str, str, int], Asset] = {}
        self._active: dict[tuple[str, str], int] = {}
        self._manifests: dict[str, Manifest] = {}
        self._actions: dict[str, ActionManifest] = {}

    @classmethod
    def load(cls, assets_dir: Path, manifests_dir: Optional[Path] = None) -> Catalog:
        """디렉터리의 YAML을 읽어 카탈로그를 만듭니다."""
        catalog = cls()
        for path in sorted(assets_dir.glob("*.yaml")):
            catalog.add(Asset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))
        if manifests_dir is not None:
            for path in sorted(manifests_dir.glob("*.yaml")):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if data.get("kind") == "action":
                    catalog.add_manifest(ActionManifest.model_validate(data))
                else:
                    catalog.add_manifest(Manifest.model_validate(data))
        catalog.validate_links()
        return catalog

    def add(self, asset: Asset) -> None:
        """자산 버전을 등록합니다. 같은 버전은 덮어쓰지 않습니다(불변)."""
        key = (asset.kind.value, asset.name, asset.version)
        if key in self._assets:
            raise ValueError(f"이미 존재하는 버전입니다: {asset.ref}")
        self._assets[key] = asset
        if asset.status == AssetStatus.ACTIVE:
            self._active[(asset.kind.value, asset.name)] = asset.version

    def add_manifest(self, manifest: AnyManifest) -> None:
        """매니페스트를 등록합니다. 에이전트와 action은 이름 공간을 같이 씁니다."""
        if manifest.agent_id in self._manifests or manifest.agent_id in self._actions:
            raise ValueError(f"이미 등록된 이름입니다: {manifest.agent_id}")
        if isinstance(manifest, ActionManifest):
            self._actions[manifest.agent_id] = manifest
        else:
            self._manifests[manifest.agent_id] = manifest

    def manifest(self, agent_id: str) -> Manifest:
        """agent_id로 에이전트 매니페스트를 조회합니다."""
        try:
            return self._manifests[agent_id]
        except KeyError as exc:
            raise KeyError(f"등록되지 않은 에이전트입니다: {agent_id}") from exc

    def action(self, action_id: str) -> ActionManifest:
        """Action 매니페스트를 조회합니다."""
        try:
            return self._actions[action_id]
        except KeyError as exc:
            raise KeyError(f"등록되지 않은 action입니다: {action_id}") from exc

    def has_agent(self, agent_id: str) -> bool:
        """등록된 에이전트인지."""
        return agent_id in self._manifests

    def agent_ids(self) -> list[str]:
        """등록된 에이전트 이름."""
        return sorted(self._manifests)

    def action_ids(self) -> list[str]:
        """등록된 action 이름."""
        return sorted(self._actions)

    def validate_links(self) -> None:
        """서브에이전트 이름이 전부 등록된 에이전트인지 확인합니다(로드 시점에 실패)."""
        for manifest in self._manifests.values():
            missing = [name for name in manifest.subagents if name not in self._manifests]
            if missing:
                raise ValueError(f"{manifest.agent_id}: 등록되지 않은 서브에이전트 {missing}")

    def active_version(self, kind: str, name: str) -> int:
        """활성 버전 번호."""
        try:
            return self._active[(kind, name)]
        except KeyError as exc:
            raise KeyError(f"활성 버전이 없습니다: {kind}://{name}") from exc

    def set_active(self, kind: str, name: str, version: int) -> Optional[int]:
        """활성 포인터를 옮기고 이전 활성 버전을 돌려줍니다."""
        previous = self._active.get((kind, name))
        self._active[(kind, name)] = version
        return previous

    def get(self, kind: str, name: str, version: int) -> Asset:
        """정수 버전으로 조회합니다."""
        return self._assets[(kind, name, version)]

    def pin(self, ref: str) -> str:
        """`@active`를 정수 버전으로 고정한 참조 문자열을 돌려줍니다."""
        parsed = AssetRef.parse(ref)
        if parsed.version == "active":
            return str(parsed.pinned(self.active_version(parsed.kind, parsed.name)))
        return ref

    def resolve(self, ref: str) -> Asset:
        """참조를 자산으로 해석합니다."""
        parsed = AssetRef.parse(self.pin(ref))
        return self.get(parsed.kind, parsed.name, int(parsed.version))

    def versions(self, kind: str, name: str) -> list[Asset]:
        """이름의 모든 버전."""
        return sorted(
            (a for (k, n, _), a in self._assets.items() if k == kind and n == name),
            key=lambda a: a.version,
        )

    def eval_cases_for(self, refs: Iterable[str]) -> list[tuple[str, EvalCase]]:
        """참조 목록의 (고정 참조, eval_case) 쌍을 모읍니다."""
        collected: list[tuple[str, EvalCase]] = []
        for ref in refs:
            asset = self.resolve(ref)
            collected.extend((asset.ref, case) for case in asset.eval_cases)
        return collected

    def tools_for_manifest(self, manifest: AnyManifest) -> list[Asset]:
        """매니페스트의 skill이 참조하는 tool 자산(고정 버전)."""
        tools: dict[str, Asset] = {}
        for skill_ref in manifest.skills:
            skill = self.resolve(skill_ref)
            for tool_ref in skill.typed_body.tools:  # type: ignore[attr-defined]
                tool = self.resolve(tool_ref)
                tools[tool.ref] = tool
        return list(tools.values())

    def kinds(self) -> list[AssetKind]:
        """등록된 자산 종류."""
        return sorted({AssetKind(k) for (k, _, _) in self._assets}, key=lambda k: k.value)
