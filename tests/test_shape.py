"""Structural tests: things that keep working when they are silently missing."""

from __future__ import annotations

import ast
from typing import get_args

import pytest

from tests.conftest import ROOT
from wigtn_agents.assembly import MIDDLEWARE_ORDER, build_middleware, build_subagents, resolve_run_model
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import RunContext
from wigtn_agents.middleware import DEEPAGENTS_BUILTIN_TOOLS, SCOPE_PARAMS

SRC = ROOT / "src" / "wigtn_agents"


@pytest.mark.parametrize("preset", ["observe", "standard", "strict"])
def test_미들웨어_순서가_프리셋_결정과_같다(catalog: Catalog, ctx: RunContext, preset: str) -> None:
    manifest = catalog.manifest("records-guide").model_copy(update={"harness_preset": preset})
    stack = build_middleware(manifest, catalog, ctx, sink=[], model=resolve_run_model(manifest, ctx))
    assert tuple(type(m).__name__ for m in stack) == MIDDLEWARE_ORDER[preset]


def test_general_purpose_서브에이전트는_같은_이름과_빈_도구로_억제된다(catalog: Catalog) -> None:
    subagents = build_subagents(catalog.manifest("records-guide"))
    gp = [s for s in subagents if s["name"] == "general-purpose"]
    assert len(gp) == 1 and gp[0]["tools"] == []


def test_내장_도구_이름_핀은_설치된_deepagents와_맞는다() -> None:
    from deepagents import FsToolName

    fs_names = set(get_args(FsToolName))
    assert fs_names <= DEEPAGENTS_BUILTIN_TOOLS, fs_names - DEEPAGENTS_BUILTIN_TOOLS


def test_스코프_인자를_받는_도구는_전부_가드_레지스트리에_있다(catalog: Catalog) -> None:
    for asset in (a for a in catalog._assets.values() if a.kind.value == "tool"):
        props = asset.typed_body.input_schema.get("properties", {})  # type: ignore[attr-defined]
        if "entity" in props:
            assert asset.tool_name() in SCOPE_PARAMS, f"{asset.tool_name()} 미등재 = 격리 갭"


def test_import_시점에_모델이나_에이전트를_만들지_않는다() -> None:
    banned = {
        "build_model",
        "init_chat_model",
        "create_deep_agent",
        "create_agent",
        "assemble",
        "get_client",
        "ChatAnthropic",
        "ChatOpenAI",
        "load_registry",
    }
    for path in (
        SRC / "worker.py",
        SRC / "assembly.py",
        SRC / "models.py",
        SRC / "presets.py",
        SRC / "actions" / "chassis.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            for call in ast.walk(node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    break
                if isinstance(call, ast.Call):
                    name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
                    assert name not in banned, f"{path.name}: import-time call to {name}"
