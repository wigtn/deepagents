"""Manifest schemas (agent and action lanes) and the flow decision table.

Two kinds share the catalog:

* ``kind: agent`` — a conversational deep agent assembled per run (graph ``worker``).
* ``kind: action`` — a fixed three-step run, gather → reason → writeback (graph ``action``).
  Inputs are gathered by code, the model only reasons, and only code writes. How much the model
  may explore is the ``level``: one structured call, a bounded ReAct loop over read tools, or a
  scoped deep agent. Control flow is never configured in the manifest; an action that needs its
  own flow is a code extension.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wigtn_agents.assets.schema import AssetRef


class WorkerClass(StrEnum):
    """워커 클래스."""

    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    SEL = "sel"


class FlowTemplate(StrEnum):
    """flow 템플릿."""

    REACT = "react"
    PLAN_EXECUTE = "plan-execute"
    APPROVAL_GATED = "approval-gated"
    EXTENSION = "extension"


class Metadata(BaseModel):
    """식별 정보."""

    name: str
    version: int = Field(ge=1)
    owner: str
    worker_class: WorkerClass
    description: str = ""


PRESETS: tuple[str, ...] = ("observe", "standard", "strict")
Preset = Literal["observe", "standard", "strict"]
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
ActionLevel = Literal["structured", "bounded-react", "scoped-agent"]


class ModelSpec(BaseModel):
    """Model alias from ``catalog/models.yaml`` and the parameters asked of it.

    A parameter the registry entry does not accept is dropped, not sent.
    """

    ref: str = "models/default"
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4096, ge=1)
    reasoning_effort: Optional[ReasoningEffort] = None

    def params(self) -> dict[str, Any]:
        """Parameters handed to the model builder."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }


class Flow(BaseModel):
    """flow 템플릿 선택."""

    template: FlowTemplate
    extension: Optional[str] = None

    @model_validator(mode="after")
    def _extension_iff_template(self) -> Flow:
        if (self.template == FlowTemplate.EXTENSION) != (self.extension is not None):
            raise ValueError("flow.extension은 template이 extension일 때만, 그때는 반드시 지정합니다")
        return self


class Interrupts(BaseModel):
    """실행 승인 지점."""

    approve_before: list[str] = Field(default_factory=list)


class Permissions(BaseModel):
    """권한. 생성 단계에서 사전 차단. SEL이 고칠 수 없다."""

    allow_tools: list[str]
    data_scope: list[str] = Field(default_factory=list)


class SelConfig(BaseModel):
    """매니페스트 수준 SEL 조정 범위."""

    mutable_by_sel: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    """에이전트 매니페스트."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    api_version: Literal["wigtn.agents/v1"] = Field(alias="apiVersion")
    kind: Literal["agent"]
    metadata: Metadata
    model: ModelSpec = Field(default_factory=ModelSpec)
    harness_preset: Preset = "standard"
    extra_middleware: list[str] = Field(default_factory=list)
    skills: list[str]
    prompts: dict[str, str]
    flow: Flow
    interrupts: Interrupts = Field(default_factory=Interrupts)
    permissions: Permissions
    subagents: list[str] = Field(default_factory=list)
    memory: bool = False
    extensions: list[str] = Field(default_factory=list)
    sel: SelConfig = Field(default_factory=SelConfig)

    @field_validator("skills")
    @classmethod
    def _skills_are_skill_refs(cls, value: list[str]) -> list[str]:
        for ref in value:
            if AssetRef.parse(ref).kind != "skill":
                raise ValueError(f"skills에는 skill 참조만 올 수 있습니다: {ref}")
        return value

    @field_validator("prompts")
    @classmethod
    def _prompts_are_prompt_refs(cls, value: dict[str, str]) -> dict[str, str]:
        for ref in value.values():
            if AssetRef.parse(ref).kind != "prompt":
                raise ValueError(f"prompts에는 prompt 참조만 올 수 있습니다: {ref}")
        return value

    @model_validator(mode="after")
    def _cross_rules(self) -> Manifest:
        allowed = set(self.permissions.allow_tools)
        outside = [ref for ref in self.interrupts.approve_before if ref not in allowed]
        if outside:
            raise ValueError(f"approve_before는 allow_tools의 부분집합이어야 합니다: {outside}")
        if self.interrupts.approve_before and self.flow.template not in (
            FlowTemplate.APPROVAL_GATED,
            FlowTemplate.EXTENSION,
        ):
            raise ValueError("실행 승인 지점이 있으면 flow는 approval-gated 또는 extension이어야 합니다")
        for path in self.sel.mutable_by_sel:
            if path.split(".")[0] in ("permissions", "interrupts", "metadata", "flow"):
                raise ValueError(f"SEL이 고칠 수 없는 매니페스트 필드입니다: {path}")
        if self.flow.template == FlowTemplate.EXTENSION and self.flow.extension not in self.extensions:
            raise ValueError("flow.extension은 extensions 목록에 있어야 합니다")
        if self.metadata.name in self.subagents:
            raise ValueError("자기 자신을 서브에이전트로 둘 수 없습니다")
        return self

    def all_asset_refs(self) -> list[str]:
        """이 매니페스트가 직접 참조하는 자산 목록(skill · prompt)."""
        return [*self.skills, *self.prompts.values()]

    @property
    def agent_id(self) -> str:
        """카탈로그 식별자."""
        return self.metadata.name


class ActionMetadata(BaseModel):
    """Identity of an action."""

    name: str
    version: int = Field(ge=1)
    owner: str
    description: str = ""


class ActionManifest(BaseModel):
    """A gather → reason → writeback run. Gather and writeback are code, selected by name."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    api_version: Literal["wigtn.agents/v1"] = Field(alias="apiVersion")
    kind: Literal["action"]
    metadata: ActionMetadata
    model: ModelSpec = Field(default_factory=ModelSpec)
    level: ActionLevel
    harness_preset: Preset = "standard"
    extra_middleware: list[str] = Field(default_factory=list)
    skill: str
    prompts: dict[str, str] = Field(default_factory=dict)
    gather: str = "input"
    writeback: str = "none"
    output_schema: dict[str, Any]

    @field_validator("skill")
    @classmethod
    def _skill_ref(cls, value: str) -> str:
        if AssetRef.parse(value).kind != "skill":
            raise ValueError(f"skill에는 skill 참조만 올 수 있습니다: {value}")
        return value

    @field_validator("prompts")
    @classmethod
    def _prompt_refs(cls, value: dict[str, str]) -> dict[str, str]:
        for ref in value.values():
            if AssetRef.parse(ref).kind != "prompt":
                raise ValueError(f"prompts에는 prompt 참조만 올 수 있습니다: {ref}")
        return value

    @field_validator("output_schema")
    @classmethod
    def _object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object" or not value.get("properties"):
            raise ValueError("output_schema는 properties가 있는 object 스키마여야 합니다")
        return value

    @property
    def skills(self) -> list[str]:
        """Same accessor as ``Manifest.skills`` (one skill per action)."""
        return [self.skill]

    def all_asset_refs(self) -> list[str]:
        """Assets this action references directly."""
        return [self.skill, *self.prompts.values()]

    @property
    def agent_id(self) -> str:
        """Catalog identifier."""
        return self.metadata.name


AnyManifest = Union[Manifest, ActionManifest]


def decide_flow(
    *,
    has_write_tool: bool,
    multi_step: bool,
    needs_intermediate_state: bool,
    expressible: bool = True,
) -> FlowTemplate:
    """Flow 판단표.

    Args:
        has_write_tool: write side-effect 도구를 참조하는가
        multi_step: 다단계인가
        needs_intermediate_state: 중간 상태가 필요한가
        expressible: 템플릿으로 표현 가능한가

    Returns:
        선택된 flow 템플릿

    """
    if not expressible:
        return FlowTemplate.EXTENSION
    if has_write_tool:
        return FlowTemplate.APPROVAL_GATED
    if multi_step and needs_intermediate_state:
        return FlowTemplate.PLAN_EXECUTE
    return FlowTemplate.REACT
