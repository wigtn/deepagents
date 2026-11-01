"""자산 프로토콜 스키마. 공통 envelope · kind별 body · 참조 문법 · 상태."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REF_PATTERN = re.compile(
    r"^(?P<kind>prompt|skill|tool|api|agent|evalset|rubric)://"
    r"(?P<name>[A-Za-z0-9_./-]+)@(?P<version>active|\d+)$"
)

# SEL이 어떤 경우에도 고칠 수 없는 필드 경로. 권한은 생성 단계에서 사전 차단한다.
FROZEN_PATHS: tuple[str, ...] = (
    "kind",
    "name",
    "version",
    "status",
    "owner",
    "permission_scope",
    "provenance",
    "mutable_by_sel",
    "body.tools",
    "body.binding",
    "body.credential_ref",
    "body.base_url_ref",
    "body.openapi_ref",
    "body.side_effect",
)


class AssetKind(StrEnum):
    """자산 종류."""

    PROMPT = "prompt"
    SKILL = "skill"
    TOOL = "tool"
    API = "api"


class AssetStatus(StrEnum):
    """자산 버전 상태. CI는 루프 안, CD는 ITSM."""

    PROPOSED = "proposed"
    REVIEW = "review"
    APPROVED = "approved"
    DEPLOY_REQUESTED = "deploy_requested"
    ACTIVE = "active"
    RETIRED = "retired"


class AssetRef(BaseModel):
    """`kind://name@version` 참조."""

    kind: str
    name: str
    version: Union[int, Literal["active"]]

    @classmethod
    def parse(cls, ref: str) -> AssetRef:
        """문자열 참조를 파싱합니다.

        Args:
            ref: `kind://name@version` 형식 문자열

        Returns:
            파싱된 참조

        Raises:
            ValueError: 형식이 맞지 않을 때

        """
        match = REF_PATTERN.match(ref)
        if match is None:
            raise ValueError(f"잘못된 자산 참조: {ref}")
        version = match.group("version")
        return cls(
            kind=match.group("kind"),
            name=match.group("name"),
            version="active" if version == "active" else int(version),
        )

    def pinned(self, version: int) -> AssetRef:
        """정수 버전으로 고정한 새 참조를 돌려줍니다."""
        return AssetRef(kind=self.kind, name=self.name, version=version)

    def __str__(self) -> str:
        return f"{self.kind}://{self.name}@{self.version}"


class Provenance(BaseModel):
    """어느 주체 · 어느 SEL run이 만들었는지."""

    created_by: Literal["human", "sel"] = "human"
    sel_run_id: Optional[str] = None
    source_version: Optional[int] = None


class EvalCase(BaseModel):
    """자산이 자기 안에 들고 다니는 테스트 케이스."""

    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    judge: Literal["exact", "contains", "llm"] = "contains"


class PromptBody(BaseModel):
    """prompt body."""

    template: str
    variables: dict[str, Any] = Field(default_factory=dict)
    model_constraints: dict[str, Any] = Field(default_factory=dict)


class SkillBody(BaseModel):
    """skill body. 에이전트가 보는 유일한 단위."""

    model_config = ConfigDict(extra="allow")

    trigger: str
    instruction: str
    tools: list[str]
    default_plan: list[str] = Field(default_factory=list)
    examples: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("tools")
    @classmethod
    def _tools_are_tool_refs(cls, value: list[str]) -> list[str]:
        for ref in value:
            if AssetRef.parse(ref).kind != "tool":
                raise ValueError(f"skill.tools에는 tool 참조만 올 수 있습니다: {ref}")
        return value

    @model_validator(mode="after")
    def _plan_subset_of_tools(self) -> SkillBody:
        missing = [ref for ref in self.default_plan if ref not in self.tools]
        if missing:
            raise ValueError(f"default_plan은 tools의 부분집합이어야 합니다: {missing}")
        return self


HTTP_METHODS: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE")


class ToolBinding(BaseModel):
    """Where a tool call goes. The worker never holds connection values, only references.

    * ``gateway`` — a named route on the tool gateway client (offline stub by default).
    * ``http`` — ``operation: "METHOD /path/{param}"`` on the ``api`` asset's base URL;
      base URL and credential are ``secret://`` references resolved from the environment.
    * ``mcp`` — remote tool ``tool`` on server ``server`` from ``catalog/mcp.yaml``.
    """

    type: Literal["gateway", "http", "mcp"] = "gateway"
    gateway_route: Optional[str] = None
    api: Optional[str] = None
    operation: Optional[str] = None
    server: Optional[str] = None
    tool: Optional[str] = None

    @field_validator("api")
    @classmethod
    def _api_is_api_ref(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and AssetRef.parse(value).kind != "api":
            raise ValueError(f"binding.api에는 api 참조만 올 수 있습니다: {value}")
        return value

    @model_validator(mode="after")
    def _fields_for_type(self) -> ToolBinding:
        if self.type == "gateway" and not self.gateway_route:
            raise ValueError("gateway 바인딩은 gateway_route가 필요합니다")
        if self.type == "http":
            if not self.api or not self.operation:
                raise ValueError("http 바인딩은 api와 operation이 필요합니다")
            method = self.operation.split(" ", 1)[0]
            if method not in HTTP_METHODS or " /" not in self.operation:
                raise ValueError(f"http operation은 'METHOD /path' 형식이어야 합니다: {self.operation}")
        if self.type == "mcp" and (not self.server or not self.tool):
            raise ValueError("mcp 바인딩은 server와 tool이 필요합니다")
        return self


class ToolBody(BaseModel):
    """tool body."""

    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = Field(default_factory=dict)
    side_effect: Literal["read", "write"]
    idempotent: bool
    binding: ToolBinding


class ApiBody(BaseModel):
    """api body. 참조만 두고 값은 절대 넣지 않는다."""

    openapi_ref: str
    base_url_ref: str
    credential_ref: str
    rate_limit: dict[str, Any] = Field(default_factory=dict)

    @field_validator("credential_ref", "base_url_ref")
    @classmethod
    def _secret_refs_only(cls, value: str) -> str:
        if not value.startswith("secret://"):
            raise ValueError("크리덴셜 · 접속 정보는 secret:// 참조만 허용합니다")
        return value


BODY_TYPES: dict[AssetKind, type[BaseModel]] = {
    AssetKind.PROMPT: PromptBody,
    AssetKind.SKILL: SkillBody,
    AssetKind.TOOL: ToolBody,
    AssetKind.API: ApiBody,
}


class Asset(BaseModel):
    """자산 envelope."""

    model_config = ConfigDict(extra="forbid")

    kind: AssetKind
    name: str
    version: int = Field(ge=1)
    status: AssetStatus
    owner: str
    description: str = ""
    inputs: Optional[dict[str, Any]] = None
    outputs: Optional[dict[str, Any]] = None
    permission_scope: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    eval_cases: list[EvalCase] = Field(default_factory=list)
    mutable_by_sel: list[str] = Field(default_factory=list)
    body: dict[str, Any]

    @model_validator(mode="after")
    def _validate_body_and_mutability(self) -> Asset:
        BODY_TYPES[self.kind].model_validate(self.body)
        for path in self.mutable_by_sel:
            if any(path == frozen or path.startswith(frozen + ".") for frozen in FROZEN_PATHS):
                raise ValueError(f"mutable_by_sel에 넣을 수 없는 경로입니다: {path}")
            if not path.startswith("body."):
                raise ValueError(f"mutable_by_sel은 body 하위 경로만 허용합니다: {path}")
        if self.kind == AssetKind.SKILL and self.inputs is None:
            raise ValueError("skill은 inputs 스키마가 필수입니다")
        return self

    @property
    def typed_body(self) -> BaseModel:
        """kind에 맞는 body 모델."""
        return BODY_TYPES[self.kind].model_validate(self.body)

    @property
    def ref(self) -> str:
        """이 버전의 정수 참조."""
        return f"{self.kind.value}://{self.name}@{self.version}"

    def tool_name(self) -> str:
        """LLM에 노출하는 도구 이름. 참조 이름의 `/`를 `_`로 바꾼다."""
        return self.name.replace("/", "_")
