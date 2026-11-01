"""Scripted fake chat model for end-to-end runs without an LLM key.

Enabled with ``WIGTN_AGENT_FAKE_MODEL=1``. The fake plays a deterministic script driven by the
bound tools and the last human message, so the runtime integration (factory call, checkpointer
injection, HITL interrupt/resume, version pinning across restarts) can be exercised offline.
It is not a substitute for real-model e2e runs; those need ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

FAKE_MODEL_ENV = "WIGTN_AGENT_FAKE_MODEL"


class ScriptedFakeModel(BaseChatModel):
    """Deterministic script: (1) call a read tool, (2) call record-write if the user asks to post, (3) answer."""

    bound: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "wigtn-agents-scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedFakeModel:  # type: ignore[override]
        """Record the bound tool names; the script decides calls from them."""
        self.bound = [getattr(t, "name", None) or str(t) for t in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        tools_done = [m.name for m in messages if isinstance(m, ToolMessage)]
        last_user = humans[-1].text if humans else ""
        wants_post = any(k in last_user for k in ("반영", "post", "update"))

        if "record-read" in self.bound and "record-read" not in tools_done:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "record-read", "args": {"period": "2026-08", "entity": "0000"}, "id": "call-read"}
                ],
            )
        if wants_post and "record-write" in self.bound and "record-write" not in tools_done:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "record-write",
                        "args": {"document": {"period": "2026-08", "lines": 2}, "entity": "0000"},
                        "id": "call-write",
                    }
                ],
            )
        posted = [m for m in messages if isinstance(m, ToolMessage) and m.name == "record-write"]
        if posted:
            return AIMessage(content=f"반영 완료. 결과: {posted[-1].content}")
        read = [m for m in messages if isinstance(m, ToolMessage) and m.name == "record-read"]
        if read:
            return AIMessage(content=f"조회 완료. 미처리 기록 요약: {read[-1].content[:120]}")
        return AIMessage(content=f"(fake) 요청을 받았습니다: {last_user[:80]}")
