"""RepeatedToolCallMiddleware — stop an agent that keeps sending the same tool call.

A model that has already seen a result and sends the identical call again (same tool, same
arguments) gains nothing from another identical result, and each round grows the context by one
call/result pair. The middleware counts identical calls at the tail of the conversation:

* at ``warn_after`` it adds one notice that the result will not change;
* at ``stop_after`` it ends the run instead of calling the model again. The last tool result is
  already in the history, so the conversation closes without a dangling tool call.

A different call in between resets the count: calling the same write again after fixing the
input is not a loop.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

WARN_AFTER: Final = 3
STOP_AFTER: Final = 6
NOTICE_NAME: Final = "repeated_tool_call_notice"

Signature = tuple[tuple[str, str], ...]


def call_signature(message: AIMessage) -> Signature:
    """Order-independent (name, arguments) of every tool call in one model turn."""
    return tuple(
        sorted(
            (call["name"], json.dumps(call.get("args") or {}, sort_keys=True, ensure_ascii=False, default=str))
            for call in message.tool_calls
        )
    )


def trailing_repeats(messages: Sequence[BaseMessage]) -> tuple[int, str]:
    """How many model turns at the end sent the same calls, and the tool names involved."""
    count = 0
    last: Optional[Signature] = None
    for message in reversed(messages):
        if isinstance(message, ToolMessage) or getattr(message, "name", None) == NOTICE_NAME:
            continue
        if not isinstance(message, AIMessage) or not message.tool_calls:
            break
        signature = call_signature(message)
        if last is not None and signature != last:
            break
        last = signature
        count += 1
    return count, ", ".join(name for name, _ in last) if last else ""


class RepeatedToolCallMiddleware(AgentMiddleware):
    """Warn once, then end the run, when identical tool calls repeat."""

    def __init__(self, *, warn_after: int = WARN_AFTER, stop_after: int = STOP_AFTER) -> None:
        super().__init__()
        if not 0 < warn_after < stop_after:
            raise ValueError("0 < warn_after < stop_after 이어야 합니다")
        self._warn_after = warn_after
        self._stop_after = stop_after

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Count identical trailing calls before the next model call."""
        count, names = trailing_repeats(state["messages"])
        if count >= self._stop_after:
            return {
                "jump_to": "end",
                "messages": [
                    AIMessage(
                        content=(
                            f"같은 도구 호출({names})이 {count}번 반복되어 실행을 멈췄습니다. 결과는 바뀌지 않았습니다."
                        )
                    )
                ],
            }
        if count == self._warn_after:
            notice = (
                f"같은 도구 호출({names})을 인자까지 같게 {count}번 보냈습니다. 결과는 바뀌지 않습니다. "
                f"받은 결과로 다음 단계를 진행하세요. {self._stop_after}번째에는 실행이 멈춥니다."
            )
            return {"messages": [HumanMessage(content=notice, name=NOTICE_NAME)]}
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> Optional[dict[str, Any]]:
        """Async twin."""
        return self.before_model(state, runtime)
