"""OutputValidatorMiddleware — catch empty or oversized final answers."""

from __future__ import annotations

from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime


class OutputValidatorMiddleware(AgentMiddleware):
    """Replace an empty answer with a retry marker and truncate over-long text."""

    def __init__(self, max_chars: int = 8000) -> None:
        super().__init__()
        self._max_chars = max_chars

    def _validate(self, state: AgentState) -> Optional[dict[str, Any]]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or (getattr(last, "tool_calls", None) or []):
            return None
        text = last.text
        if not text.strip():
            return {"messages": [AIMessage(content="(빈 응답 · 재시도 필요)", id=last.id)]}
        if len(text) > self._max_chars:
            return {"messages": [AIMessage(content=text[: self._max_chars], id=last.id)]}
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Validate the final assistant message."""
        return self._validate(state)

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Async twin."""
        return self._validate(state)
