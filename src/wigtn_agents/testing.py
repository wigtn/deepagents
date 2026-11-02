"""Scripted fake chat model for end-to-end runs without an LLM key.

Enabled with ``WIGTN_AGENT_FAKE_MODEL=1`` or the ``models/fake`` alias. The fake plays a
deterministic script driven by the bound tools and the last human message, so the runtime
integration (factory call, checkpointer injection, HITL interrupt/resume, version pinning across
restarts, the action lane) can be exercised offline. It is not a substitute for real-model runs.

Script: (1) call ``record-read`` if bound, (2) call ``record-write`` if the user asks to post,
(3) if an action result tool is bound, call it with placeholder values, (4) answer in text.
On the sample ERP catalog the same script reads with the first bound tool of ``ERP_READS`` and,
when asked to post, posts the first journal proposal of the close checklist.
Structured-output calls return placeholder values for every field of the schema.
"""

from __future__ import annotations

import types
import typing
from typing import Any, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

FAKE_MODEL_ENV = "WIGTN_AGENT_FAKE_MODEL"
RESULT_TOOL = "ActionResult"
_PLACEHOLDERS: dict[Any, Any] = {str: "(fake)", int: 0, float: 0.0, bool: False, dict: {}, list: []}


def _placeholder(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _placeholder(args[0]) if args else None
    if origin is not None:
        annotation = origin
    value = _PLACEHOLDERS.get(annotation, "(fake)")
    return type(value)() if isinstance(value, (dict, list)) else value


def placeholder_values(schema: Any) -> dict[str, Any]:
    """A value for every field of a Pydantic model or a JSON Schema dict."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return {name: _placeholder(info.annotation) for name, info in schema.model_fields.items()}
    if isinstance(schema, dict):
        return {name: _placeholder(_json_type(spec)) for name, spec in schema.get("properties", {}).items()}
    return {}


_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _json_type(spec: dict[str, Any]) -> type:
    """Python type of a JSON Schema property; ``anyOf`` takes its first non-null branch."""
    if "type" in spec and spec["type"] != "null":
        return _JSON_TYPES.get(spec["type"], str)
    for branch in spec.get("anyOf", []):
        if branch.get("type") not in (None, "null"):
            return _JSON_TYPES.get(branch["type"], str)
    return str


ERP_READS: tuple[str, ...] = (
    "erp-close-checklist",
    "erp-three-way-match",
    "erp-stock-read",
    "erp-ar-aging",
    "erp-period-status",
)
POST_WORDS: tuple[str, ...] = ("반영", "전기", "올려", "post", "update")


def _journal_proposal(messages: list[BaseMessage]) -> dict[str, Any]:
    """Arguments for ``erp-journal-post`` from the last close checklist the run read."""
    import json

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "erp-close-checklist":
            try:
                checks = json.loads(message.text).get("checks", [])
            except ValueError:
                break
            for check in checks:
                if check.get("proposal"):
                    return {"text": check["detail"][:60], "lines": check["proposal"]["lines"]}
    return {
        "text": "Monthly depreciation",
        "lines": [{"account": "6300", "debit": 2_400_000}, {"account": "1590", "credit": 2_400_000}],
    }


class ScriptedFakeModel(BaseChatModel):
    """Deterministic script over the bound tools; see the module docstring."""

    bound: list[str] = []
    bound_schemas: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "wigtn-agents-scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedFakeModel:
        """Record the bound tool names (and schemas); the script decides calls from them."""
        self.bound = []
        self.bound_schemas = {}
        for tool in tools:
            name = getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None) or str(tool)
            self.bound.append(name)
            self.bound_schemas[name] = getattr(tool, "args_schema", None) or (
                tool.get("parameters") if isinstance(tool, dict) else None
            )
        return self

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any) -> Runnable[Any, Any]:
        """Placeholder instance of the schema (or dict for a JSON Schema)."""

        def produce(_: Any) -> Any:
            values = placeholder_values(schema)
            parsed: Any = schema(**values) if isinstance(schema, type) and issubclass(schema, BaseModel) else values
            if include_raw:
                return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}
            return parsed

        return RunnableLambda(produce)

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
        wants_post = any(k in last_user for k in POST_WORDS)
        erp_read = next((n for n in ERP_READS if n in self.bound), None)
        if erp_read and erp_read not in tools_done:
            return AIMessage(content="", tool_calls=[{"name": erp_read, "args": {}, "id": "call-erp-read"}])
        if wants_post and "erp-journal-post" in self.bound and "erp-journal-post" not in tools_done:
            args = _journal_proposal(messages)
            return AIMessage(content="", tool_calls=[{"name": "erp-journal-post", "args": args, "id": "call-erp-post"}])

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
        if RESULT_TOOL in self.bound and RESULT_TOOL not in tools_done:
            args = placeholder_values(self.bound_schemas.get(RESULT_TOOL))
            return AIMessage(content="", tool_calls=[{"name": RESULT_TOOL, "args": args, "id": "call-result"}])
        posted = [m for m in messages if isinstance(m, ToolMessage) and m.name in ("record-write", "erp-journal-post")]
        if posted:
            return AIMessage(content=f"반영 완료. 결과: {posted[-1].content}")
        read = [m for m in messages if isinstance(m, ToolMessage) and m.name in ("record-read", *ERP_READS)]
        if read:
            return AIMessage(content=f"조회 완료. 미처리 기록 요약: {read[-1].content[:120]}")
        return AIMessage(content=f"(fake) 요청을 받았습니다: {last_user[:80]}")
