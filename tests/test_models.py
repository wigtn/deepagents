"""Model alias registry: what can be built, with which parameters, at which price."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.conftest import CATALOG_DIR
from wigtn_agents.models import ModelRegistry, build_model, load_registry, provider_params
from wigtn_agents.testing import ScriptedFakeModel


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.load(CATALOG_DIR / "models.yaml")


@pytest.fixture(autouse=True)
def _no_fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WIGTN_AGENT_FAKE_MODEL", raising=False)
    monkeypatch.setenv("WIGTN_AGENT_COST_LEDGER", str(tmp_path / "cost.json"))
    monkeypatch.setenv("WIGTN_AGENT_GAP_LOG", str(tmp_path / "gaps.jsonl"))


def test_기본_별칭은_gpt_6_luna_Responses_API다(registry: ModelRegistry) -> None:
    resolved = registry.resolve(None)
    assert resolved.alias == "models/default"
    assert resolved.model_id == "openai:gpt-6-luna"
    assert resolved.entry.api == "responses"
    assert resolved.entry.price is not None and resolved.entry.price.cached_input_rate == 0.01


def test_등록되지_않은_별칭은_거부한다(registry: ModelRegistry) -> None:
    with pytest.raises(KeyError, match="등록되지 않은 모델 별칭"):
        registry.resolve("models/nope")
    with pytest.raises(KeyError):
        registry.resolve("openai:gpt-6-luna")  # provider ids are not aliases


LOCAL = {"provider": "openai-compatible", "model": "local-m", "base_url_env": "WIGTN_AGENT_LOCAL_BASE_URL"}


def _with_local(price: object) -> ModelRegistry:
    data = yaml.safe_load((CATALOG_DIR / "models.yaml").read_text(encoding="utf-8"))
    data["models"]["models/local"] = {**LOCAL, "price": price}
    return ModelRegistry.model_validate(data)


def test_가격이_없는_별칭은_호출_전에_거부한다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIGTN_AGENT_LOCAL_BASE_URL", "http://localhost:8000/v1")
    with pytest.raises(LookupError, match="가격"):
        build_model("models/local", registry=_with_local(None))


def test_받지_않는_파라미터는_보내지_않는다(registry: ModelRegistry) -> None:
    entry = registry.resolve("models/default").entry
    params = provider_params(entry, {"temperature": 0.2, "max_tokens": 512, "reasoning_effort": None})
    assert params == {"max_tokens": 512, "reasoning_effort": "medium"}  # temperature dropped, default kept


def test_gpt_6_luna는_Responses_API로_만든다(registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = build_model("models/default", registry=registry, temperature=0.3, max_tokens=256, reasoning_effort="low")
    assert type(model).__name__ == "ChatOpenAI"
    assert model.use_responses_api is True  # type: ignore[attr-defined]
    assert model.model_name == "gpt-6-luna"  # type: ignore[attr-defined]
    assert model.temperature is None  # type: ignore[attr-defined]
    assert model.reasoning_effort == "low" and model.max_tokens == 256  # type: ignore[attr-defined]
    assert any(type(c).__name__ == "BudgetCallback" for c in model.callbacks or [])  # type: ignore[union-attr]


def test_openai_호환_엔드포인트는_주소_변수가_필요하다(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _with_local({"input": 0, "output": 0})
    monkeypatch.delenv("WIGTN_AGENT_LOCAL_BASE_URL", raising=False)
    with pytest.raises(LookupError, match="WIGTN_AGENT_LOCAL_BASE_URL"):
        build_model("models/local", registry=registry)
    monkeypatch.setenv("WIGTN_AGENT_LOCAL_BASE_URL", "http://localhost:8000/v1")
    model = build_model("models/local", registry=registry, temperature=0.0)
    assert model.use_responses_api is False  # type: ignore[attr-defined]
    assert str(model.openai_api_base) == "http://localhost:8000/v1"  # type: ignore[attr-defined]
    assert model.temperature == 0.0  # type: ignore[attr-defined]


def test_MARA_Cloud_모델은_고정_주소와_키로_만든다(registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    mara = [a for a, e in registry.models.items() if e.base_url == "https://api.cloud.mara.com/v1"]
    assert len(mara) == 5 and all(registry.models[a].price is not None for a in mara)
    monkeypatch.delenv("MARACLOUD_API_KEY", raising=False)
    with pytest.raises(LookupError, match="MARACLOUD_API_KEY"):
        build_model("models/gpt-oss-120b", registry=registry)
    monkeypatch.setenv("MARACLOUD_API_KEY", "k-test")
    model = build_model("models/gpt-oss-120b", registry=registry, reasoning_effort="low")
    assert str(model.openai_api_base) == "https://api.cloud.mara.com/v1"  # type: ignore[attr-defined]
    assert model.max_retries == 4 and model.reasoning_effort == "low"  # type: ignore[attr-defined]


def test_기본_채점_모델은_기본_백본과_다른_개발사_모델이다(registry: ModelRegistry) -> None:
    assert registry.judge is not None
    judge, backbone = registry.resolve(registry.judge), registry.resolve(None)
    assert judge.model_id != backbone.model_id and "gpt" not in judge.entry.model


def test_fake_별칭과_환경_변수는_오프라인_모델을_만든다(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(build_model("models/fake", registry=registry), ScriptedFakeModel)
    monkeypatch.setenv("WIGTN_AGENT_FAKE_MODEL", "1")
    assert isinstance(build_model("models/default", registry=registry), ScriptedFakeModel)


def test_레지스트리_규칙_위반은_로드에서_실패한다() -> None:
    base = {"default": "a", "models": {"a": {"provider": "openai-compatible", "model": "m"}}}
    with pytest.raises(ValueError, match="base_url"):
        ModelRegistry.model_validate(base)
    with pytest.raises(ValueError, match="default"):
        ModelRegistry.model_validate({"default": "b", "models": {"a": {"provider": "fake", "model": "m"}}})
    with pytest.raises(ValueError, match="Responses"):
        ModelRegistry.model_validate(
            {"default": "a", "models": {"a": {"provider": "anthropic", "model": "m", "api": "responses"}}}
        )


def test_카탈로그_레지스트리는_파일이_바뀌면_다시_읽는다(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("default: a\nmodels:\n  a: {provider: fake, model: m, price: {input: 0, output: 0}}\n")
    assert load_registry(path).resolve(None).alias == "a"
    path.write_text("default: b\nmodels:\n  b: {provider: fake, model: m, price: {input: 0, output: 0}}\n")
    import os

    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
    assert load_registry(path).resolve(None).alias == "b"
