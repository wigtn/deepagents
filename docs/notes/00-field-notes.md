# Field notes — 설치본 기준 실측 (증거 트레일)

> 설계 판단의 근거를 설치된 패키지 소스로 다시 확인한 기록.
> 기준: deepagents **0.7.17** · aegra-api **0.10.5** · langchain 1.4.2 · langgraph 1.2.12 · langgraph-sdk 0.4.5 · Python 3.12 (2026-09-23). 11-02 행부터 langchain-core 1.7.3 · langchain-openai 1.7.3 · langchain-mcp-adapters 0.3.2

## deepagents 0.7.17

| # | 확인 사항 | 근거 (site-packages 기준) | 영향 |
|---|---|---|---|
| 1 | `create_deep_agent` 인자: `model, tools, system_prompt, middleware, subagents, skills, memory, permissions, backend, interrupt_on, response_format, state_schema, context_schema, checkpointer, store, debug, name, cache` | `inspect.signature(create_deep_agent)` | `checkpointer=None, store=None` 전달 가능 |
| 2 | 미들웨어 스택 순서: `[Skills?] Filesystem, SubAgent?, Summarization, PatchToolCalls, AsyncSubAgent?, profile extras, PromptCaching, Memory?, HITL` → **그 뒤에 사용자 middleware** → `_ToolExclusion?` | `deepagents/graph.py` `deepagent_middleware` 조립부 (`_apply_custom_middleware(..., core_names=_main_core_names)`) | 사용자 미들웨어는 HITL보다 **안쪽**. interrupt는 우리 필터보다 먼저 일어난다 |
| 3 | 기본 시스템 메시지는 **빈 문자열**. `base_prompt = _apply_profile_prompt(_profile, "")`, 기본 하네스 프로필 `base_system_prompt=None`. 툴 사용 지시는 툴 description에 있다 | 빈 `create_deep_agent(model=fake)` 실행 후 모델이 받은 첫 메시지 `SystemMessage(len=0)` | prepend 로직은 유효하되 "하네스 텍스트가 사라진다"는 우려 자체는 0.7.17에서 해당 없음. 테스트는 "baseline이 있으면 보존"으로 작성 |
| 4 | 기본 바인딩 툴: `ls, read_file, write_file, edit_file, delete, glob, grep, task`. **`write_todos` 없음** | fake 모델 `bind_tools` 기록 | 스킬 레퍼런스의 "write_todos 항상 포함"은 구버전 설명. 허용 목록 핀에 `delete` 추가 |
| 5 | `FsToolName = Literal['ls','read_file','write_file','edit_file','delete','glob','grep','execute']` | `deepagents.FsToolName` | `tests/test_shape.py`가 핀과 대조 |
| 6 | `FilesystemPermission(operations: list['read'|'write'], paths: list[str], mode: 'allow'|'deny'|'interrupt'='allow')` | `inspect.signature` | 읽기전용 에이전트의 내장 쓰기 제한은 `permissions=[FilesystemPermission(["write"], ["/"], mode="deny")]` 로 가능 (미적용) |
| 7 | `SubAgent` 키: `name, description, tools, model, middleware, interrupt_on, skills, permissions, response_format, system_prompt, mode`. `AsyncSubAgent` 키: `name, description, graph_id, url, headers` | TypedDict `__annotations__` | gp 억제(`tools: []`), AsyncSubAgent |
| 8 | `HarnessProfile(base_system_prompt, system_prompt_suffix, tool_description_overrides, excluded_tools, excluded_middleware, extra_middleware, general_purpose_subagent)` · `GeneralPurposeSubagentProfile(enabled, description, system_prompt)` | `deepagents.profiles` | gp 억제의 대안 경로(프로필 `enabled=False`). 공개 인자만 쓰기로 해서 미사용 |
| 9 | `AgentMiddleware` 훅: `before_agent/abefore_agent, before_model/abefore_model, after_model/aafter_model, after_agent/aafter_agent, wrap_model_call/awrap_model_call, wrap_tool_call/awrap_tool_call, state_schema` | `dir(AgentMiddleware)` | 런타임은 `ainvoke` → 모든 미들웨어에 async 쌍 구현 |
| 10 | `ModelRequest` 필드: `model, messages, system_message, tool_choice, tools, response_format, state, runtime, model_settings`. `ToolCallRequest`: `tool_call, tool, state, runtime` | dataclass fields | ScopeGuard는 `request.runtime.context`에서 정본을 읽는다 |

## aegra-api 0.10.5

| # | 확인 사항 | 근거 | 영향 |
|---|---|---|---|
| 1 | 팩토리 시그니처는 **타입 어노테이션**으로 분류 (`RunnableConfig` / `ServerRuntime`). 요청마다 호출, 캐시 없음 | `services/graph_factory.py` | `worker.graph(config: RunnableConfig, runtime: ServerRuntime[RunContext])` |
| 2 | run config = `_merge_jsonb(assistant.config, run.config)` = `dict.update` **얕은 병합**, run config가 이김 | `services/run_preparation.py`, `utils/run_utils.py` | 함정: `configurable`을 보내면 assistant의 `agent_id`가 사라진다 → 클라이언트가 항상 포함 |
| 3 | 서버가 `configurable`에 주입하는 키: `langgraph_auth_user`, `thread_id`, `run_id`. 클라이언트의 `thread_id/run_id`는 제거 | `services/langgraph_service.py:684,721-722`, `utils/run_utils.py:11-16` | principal = `langgraph_auth_user.identity`만 신뢰 |
| 4 | `context`가 비면 `configurable.copy()`가 `runtime.context`로 | `run_preparation.py` | RunContext 키를 configurable에 flat으로 보내면 둘 다 채워진다 |
| 5 | 체크포인터·스토어 주입: 팩토리 결과에 `graph.copy(update={"checkpointer", "store"})`. 실패 시 경고 후 무영속 실행 | `services/langgraph_service.py` | 테스트도 같은 방식으로 주입 |
| 6 | 초기 마이그레이션에 `CREATE EXTENSION` 없음. pgvector는 시맨틱 스토어용 | `alembic/versions/20250817172544_initial_schema.py` | 로컬 PG로도 부팅 가능(미실측). 컨테이너는 pgvector 0.8.6 포함 |
| 7 | 설정: `DATABASE_URL` 우선, 없으면 `POSTGRES_*`. `AUTH_TYPE` 기본 `noop`. `PORT` 2026 | `settings.py:75-147` | `.env.example` |
| 8 | CLI(`aegra-cli` 0.10.5): `init · dev · serve · up · down · db · version` | `aegra --help` | `aegra db upgrade` 후 `aegra dev` |

## 이 프로젝트에서 밟은 것

| 날짜 | 증상 | 원인 | 대응 |
|---|---|---|---|
| 09-23 | 차단된 도구 호출이 `tool.call` 이벤트에 안 남음 | `PermissionFilter`가 `EventRecorder`보다 **바깥**이라 short-circuit 시 안쪽 미들웨어가 호출되지 않음. | 순서를 AssetLoader → **EventRecorder** → PermissionFilter → ScopeGuard → OutputValidator 로 변경. "게이트가 먼저 막고 통과분에만 정본 주입"(Permission → Scope)은 유지 |
| 09-23 | `uv run python -c "import wigtn_agents"` 실패 | `[tool.uv] package=false` 라 `src`가 sys.path에 없음 | 스크립트는 `PYTHONPATH=src`, pytest는 `pythonpath=["src"]`, Aegra는 `dependencies: ["./src"]` |
| 09-23 | `threads.get_state`에 `asset_versions`가 없고 interrupted run인데 `tasks`가 비어 있음 | 상태 조회 경로(`api/threads.py:433`)가 팩토리를 `{"configurable": {"thread_id"}}`만으로 호출한다(`access_context="threads.read"`). thread 메타데이터에 `assistant_id·graph_id`가 있지만 팩토리에는 안 넘긴다. 자리표시 그래프(노드 1개)로 체크포인트를 읽으니 채널·태스크가 안 맞았다 | **런타임이 읽기 경로에서 "어느 매니페스트인지"를 표현하지 못한다.** 대응: 상태 스키마·노드 이름은 매니페스트와 무관하므로 `agent_id` 부재 시 **스키마 동형 조립**(자산 0, 가드 미들웨어가 실행 시 즉시 예외)을 반환. 대안(미채택): 매니페스트마다 graph_id를 정적 등록 → 읽기는 자연스럽지만 에이전트 추가가 재기동이 된다 |
| 09-23 | 스키마 동형 조립으로 읽어도 interrupted thread의 `tasks`·`interrupts`가 비어 있음 | `HumanInTheLoopMiddleware`는 `after_model` **노드**와 `hitl_edited_tool_calls` 채널을 만든다. `interrupt_on={}`는 deepagents가 무시해 HITL이 빠진다. 읽기용 그래프에 그 노드가 없으면 대기 태스크가 매핑되지 않는다 | 읽기용 조립을 **상위집합**으로: sentinel 승인 지점(`__schema_only_never_called__`)으로 HITL을 포함. 가드는 노드를 만들지 않는 `wrap_model_call`로. 테스트 `real.nodes ⊆ schema_only.nodes` |
| 09-23 | (검증) interrupt 중 서버 재시작 → `command={"resume": {"decisions": [{"type": "approve"}]}}` 로 같은 thread에 **새 run** 생성 → success. interrupt 페이로드는 `action_requests[{name,args,description}]` + `review_configs` | 체크포인트가 프로세스 밖(Postgres)에 있다. 재개는 새 run이다 | 재시작 뒤 재개 확인. "다른 워커"에서의 재개는 Redis + 2인스턴스로 확인할 일 |
| 09-23 | `POST /assistants`가 400 "Failed to load graph: agent_id가 없습니다" | Aegra `assistant_service.create_assistant`가 `get_graph_for_validation(graph_id)`를 **config 없이** 호출한다(`services/langgraph_service.py:432-`, `access_context="assistants.read"`, store 없음). 팩토리가 run 밖에서도 호출된다는 뜻 — Studio 토폴로지·스키마 읽기도 같은 경로 | `worker.graph`가 `agent_id` 부재 시 예외 대신 **자리표시 그래프**(노드 1개, 실행되면 RuntimeError)를 반환. 검증은 통과하고 실제 run은 loud fail. `runtime.access_context`로 검증/실행을 구분할 수도 있다 |
| 11-02 | 서브에이전트의 도구 호출이 부모의 ScopeGuard를 거치지 않고, 서브에이전트 안에 같은 가드를 걸어도 `runtime.context`가 비어 있어 모두 차단됨 | deepagents `task` 도구가 서브에이전트를 `subagent.invoke(state, {"configurable": {...}})`로 부른다(`middleware/subagents.py`). `context=`를 넘기지 않는다 | 조립이 run마다 일어나므로 run context를 서브에이전트 미들웨어에 정적으로 넘긴다(`ScopeGuardMiddleware(static_context=ctx)`). 런타임 context에 값이 있으면 그쪽이 이긴다 |
| 11-02 | `memory: true` 에이전트가 store 없이 돌면 `MemoryMiddleware.before_agent`에서 `AttributeError` | `StoreBackend._get_store()`가 `get_store()`의 `None`을 그대로 돌려준다. Aegra는 store를 주입하지만 로컬 스크립트·테스트는 아니다 | `RuntimeStoreBackend`가 store가 없으면 프로세스 안 `InMemoryStore`로 대신하고 처음 한 번 경고를 남긴다. 스키마 전용 조립은 실행 전에 가드가 아니라 이 단계에서 멈출 수 있어서도 필요했다 |
| 11-02 | `create_agent(response_format=Model)`의 구조화 응답 도구 호출이 `tool.call` 이벤트에 안 남음 | ToolStrategy는 그 호출을 ToolNode가 아니라 에이전트 안에서 처리한다. `wrap_tool_call`을 지나지 않는다 | action의 `tool.call` 행에는 읽기 도구만 남는다. 결과는 `run.end`와 thread state `result`로 본다 |
| 11-02 | ChatOpenAI 인스턴스를 넘기면 deepagents가 Responses API를 켜지 않음 | deepagents OpenAI 프로필은 모델을 문자열로 받을 때만 `use_responses_api=True`를 넣는다 | `models.py`가 레지스트리의 `api: responses`를 보고 직접 켠다. langchain-openai 1.7.3은 Responses 요청에서 `max_tokens`→`max_output_tokens`, `reasoning_effort`→`reasoning.effort`로 바꿔 보낸다(`chat_models/base.py`) |
| 11-02 | 미들웨어 프리셋별로 그래프 노드가 다름 | `before_*`/`after_*` 훅이 노드를 만든다(RepeatedToolCall·OutputValidator·ModelCallLimit·ToolCallLimit, memory의 MemoryMiddleware) | 스키마 전용 조립을 `strict` + `memory: true` + 등록된 모든 extra 미들웨어로 만든다. 테스트가 프리셋 3개 × 에이전트 3개에서 `real.nodes ⊆ schema_only.nodes`를 확인한다 |

| 11-02 | 클라이언트가 `context.entity`와 `configurable.entity`를 다르게 보내면 메인 에이전트와 서브에이전트·gather가 서로 다른 범위를 씀 | Aegra는 요청의 `context`가 비었을 때만 `configurable`을 복사해 runtime context로 쓴다(`services/run_preparation.py:228`). `build_run_context`는 `configurable`을 먼저 본다 | 모든 ScopeGuard에 서버가 만든 run context를 `static_context`로 넘기고 그 값이 이기게 했다. 범위의 정본은 `build_run_context` 하나다 |
| 11-02 | bounded-react가 "최대 8회"인데 standard에서 도구 2회 만에 `GraphRecursionError` | `recursion_limit`은 그래프 step 수라 미들웨어 훅 노드도 센다 | 라운드는 `ModelCallLimitMiddleware(run_limit=rounds+1, exit_behavior="end")`로 세고, recursion_limit은 넉넉한 안전판(라운드당 25)으로 둔다. 넘기면 "라운드 상한" 오류로 끝난다 |
