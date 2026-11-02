"""Shared fixtures: catalog, offline fake model that records what the agent sends it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import RunContext, default_context
from wigtn_agents.tools.gateway import ToolGatewayClient

ROOT = Path(__file__).resolve().parents[1]
# The runtime tests run on a small fixed catalog (an example "records" domain) so they do not
# change when the shipped catalog (catalog/, the ERP agents) does. models.yaml is a link to the
# shipped registry. tests/test_erp_catalog.py covers the shipped catalog.
CATALOG_DIR = ROOT / "tests" / "fixtures" / "catalog"
SHIPPED_CATALOG_DIR = ROOT / "catalog"


@pytest.fixture(autouse=True)
def _fixture_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from wigtn_agents.sample_erp import shared_erp

    monkeypatch.setenv("WIGTN_AGENT_CATALOG_DIR", str(CATALOG_DIR))
    shared_erp().reset()  # the process-wide sample ERP: every test starts from the seed data
    yield


class RecordingFakeModel(GenericFakeChatModel):
    """Fake chat model: replays scripted AIMessages, ignores tool binding, records inputs."""

    seen_messages: list[list[BaseMessage]] = []
    bound_tool_names: list[list[str]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> RecordingFakeModel:  # type: ignore[override]
        names = [getattr(t, "name", None) or getattr(t, "__name__", None) or str(t) for t in tools]
        self.bound_tool_names.append(names)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted_model(*replies: AIMessage) -> RecordingFakeModel:
    """Model that answers with the given messages in order."""
    model = RecordingFakeModel(messages=iter(replies))
    model.seen_messages = []
    model.bound_tool_names = []
    return model


@pytest.fixture
def catalog() -> Catalog:
    return Catalog.load(CATALOG_DIR / "assets", CATALOG_DIR / "manifests")


@pytest.fixture
def gateway() -> ToolGatewayClient:
    return ToolGatewayClient()


@pytest.fixture
def ctx() -> RunContext:
    return default_context("records-guide", principal="u-1", entity="1000", data_scope=["view:default"])


@pytest.fixture
def events() -> list[dict[str, Any]]:
    return []


def user_turn(text: str) -> dict[str, Any]:
    """Input payload for one user message."""
    return {"messages": [{"role": "user", "content": text}]}


def thread(thread_id: str) -> dict[str, Any]:
    """Config with a thread id."""
    return {"configurable": {"thread_id": thread_id}}


def iter_replies(*replies: AIMessage) -> Iterator[AIMessage]:
    """Helper kept for readability in tests."""
    return iter(replies)
