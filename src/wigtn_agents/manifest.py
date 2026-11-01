"""매니페스트 스키마와 flow 판단표."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Optional

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


class ModelSpec(BaseModel):
    """모델 지정. LLM 게이트웨이 경유."""

    ref: str
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=1)


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
    model: ModelSpec
    harness_preset: str
    skills: list[str]
    prompts: dict[str, str]
    flow: Flow
    interrupts: Interrupts = Field(default_factory=Interrupts)
    permissions: Permissions
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
        return self

    def all_asset_refs(self) -> list[str]:
        """이 매니페스트가 직접 참조하는 자산 목록(skill · prompt)."""
        return [*self.skills, *self.prompts.values()]

    @property
    def agent_id(self) -> str:
        """카탈로그 식별자."""
        return self.metadata.name


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
