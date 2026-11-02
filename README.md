# wigtn/deepagents

An agent runtime for testing LLM agent harnesses. [Aegra](https://github.com/ibbybuilds/aegra) (a self-hosted Agent
Protocol server) runs the workers, and [DeepAgents](https://github.com/langchain-ai/deepagents) assembles each agent
from a manifest. Research systems attach to it at four fixed points (run, tools, middleware, record) without
changing the runtime.

A harness here is everything a manifest fixes around the model: prompts, skills, tools, sub-agents, the approval
points and the middleware stack. Because the manifest is the whole harness, the configuration file is the ground
truth of what the agent is. The research project [wigtn/harness-extractors](https://github.com/wigtn/harness-extractors)
uses this repository as its real-agent testbed.

The catalog ships ERP back-office agents that work against a small in-memory ERP (`src/wigtn_agents/sample_erp/`), so a
harness can be measured on tasks with checkable answers. See [ERP catalog](#erp-catalog).

## Two lanes

| Lane | Graph | Manifest | What runs |
|---|---|---|---|
| agent | `worker` | `kind: agent` | A conversational deep agent: the model plans, picks tools, delegates to sub-agents, stops at approval points |
| action | `action` | `kind: action` | A fixed run: gather (code) → reason (model) → writeback (code) |

The action lane is for work whose inputs and storage are known in advance. Code collects the inputs and stores the
result; the model only decides. How much the model may explore is the action's `level`:

| level | reason step |
|---|---|
| `structured` | one structured-output call, no tools |
| `bounded-react` | a ReAct loop over the skill's read tools, at most 8 tool rounds, ending in the output schema |
| `scoped-agent` | a deep agent scoped to the skill (read tools, scratch files, a tool-less sub-agent), at most 24 tool rounds |

A run that uses up its rounds without the output schema fails with that reason. A gather tool that fails fails the
run; the model never reasons over an error text. Extra middleware needs a tool loop, so a `structured` action refuses
it.

Writing is never an LLM tool in the action lane: an action whose skill reaches a `write` tool does not build.
Control flow is never configured in a manifest; an action that needs its own flow becomes a code extension.

```
client (langgraph_sdk) ──POST /threads/{id}/runs──▶ Aegra (API · queue · checkpointer · store · SSE · HITL)
                                                      │ aegra.json graphs (called per request)
                              ┌───────────────────────┴───────────────────────┐
                              ▼                                               ▼
            worker.graph → assemble(manifest, ctx)          worker.action → build_action_graph(action, ctx)
            create_deep_agent(model, tools,                 gather ─▶ reason (structured | bounded-react |
              middleware = preset stack,                              scoped-agent) ─▶ writeback
              subagents = [catalog agents…, general-purpose suppressed],
              backend = /memories/ on the store, interrupt_on = approve_before)
```

## Run

```bash
uv sync                                  # Python 3.12 · aegra-api 0.10.x · deepagents 0.7.x
cp .env.example .env                     # OPENAI_API_KEY (default model), MARACLOUD_API_KEY (open-weight, judge)
docker compose up -d postgres            # pgvector/pg18 on host port 5433
uv run aegra db upgrade                  # runtime migrations
uv run aegra dev                         # http://localhost:2026/docs
PYTHONPATH=src uv run python scripts/register_assistants.py   # catalog → assistants (both graphs)
```

One run, through the server or in-process:

```bash
PYTHONPATH=src uv run python scripts/run.py erp-assistant "재주문점 아래로 떨어진 자재 있어?"
PYTHONPATH=src uv run python scripts/run.py close-readiness '{}'
PYTHONPATH=src uv run python scripts/run.py close-accountant "10월 감가상각 전표 올려줘" --local --approve
PYTHONPATH=src uv run python scripts/run.py ap-clerk "INV-V200-0917 확인" --local --model models/gpt-oss-120b
```

Eval cases carried by the assets: `PYTHONPATH=src uv run python scripts/eval.py close-accountant --model <alias>`.
A case is graded by `contains` (expected strings appear), `exact` (equal text) or `llm` (a judge model grades the
answer against the case's rubric: criteria with weights and a `pass_score`; the score is computed from the per-criterion
calls, not asked of the judge). The judge (`--judge-model`, or `judge` in `catalog/models.yaml`) must be another
provider model than the backbone; the eval stops before the first run otherwise. `--no-llm` skips `llm` cases.

Offline tests (no model calls): `uv run pytest -q`. Lint: `uv run ruff check .`. Types: `uv run mypy`. CI
(`.github/workflows/ci.yml`) runs the three on every pull request and push to `main`; real-model runs stay manual.

`package = false`, so ad hoc scripts need `PYTHONPATH=src` (Aegra puts the same path on `sys.path` through
`dependencies: ["./src"]` in `aegra.json`).

## Models

`catalog/models.yaml` is the alias registry. A manifest names an alias (`model.ref`); a run can switch it with
`configurable.model`, so the same harness runs on another backbone without editing the manifest.

| Alias | Model | API |
|---|---|---|
| `models/default` | gpt-6-luna | OpenAI Responses API |
| `models/gpt-oss-120b` · `models/deepseek-v3.1` · `models/deepseek-v3.2` · `models/gemma-4-31b` · `models/minimax-m3` | open-weight backbones | MARA Cloud, OpenAI-compatible (`MARACLOUD_API_KEY`) |
| `models/fake` | scripted offline model | none |

The default judge of `llm` eval cases is `models/deepseek-v3.2`, another developer than the default backbone. An eval
of DeepSeek-V3.2 itself needs `--judge-model`. A local vLLM endpoint is an entry with `base_url_env` instead of
`base_url`.

An alias that is not registered is refused, and so is an alias without a price: every real call goes through a daily
cost cap (`WIGTN_AGENT_DAILY_COST_CAP_USD`) that must be able to charge it. Manifest parameters an entry does not `accept` (for
example `temperature` on gpt-6-luna) are dropped instead of sent. `WIGTN_AGENT_FAKE_MODEL=1` swaps every alias for the
scripted model; choosing `models/fake` per run is refused for `session_type: production`, so a scripted result is
never written back as a real one.

## Middleware presets

| `harness_preset` | Stack (first = outermost) |
|---|---|
| `observe` | AssetLoader → EventRecorder → Progress → [extra] |
| `standard` (default) | observe + PermissionFilter → ScopeGuard → RepeatedToolCall → OutputValidator |
| `strict` | standard + ModelCallLimit (40) → ToolCallLimit (60), and no filesystem writes |

- `AssetLoader` pins `@active` asset versions at run start (kept in thread state, so a resumed run uses the same
  versions) and prepends prompts and skills to the system message.
- `EventRecorder` writes the run record (below). It sits outside every gate, so a denied call is still recorded.
- `Progress` emits tool start/end markers on the `custom` stream: tool name and status, never arguments or results.
- `PermissionFilter` denies tools outside the manifest's `allow_tools` (plus the DeepAgents built-ins).
- `ScopeGuard` overwrites scope arguments (the ERP tools' `company`) with the run context's `entity`; no value, no call.
- `RepeatedToolCall` warns once at the third identical tool call in a row and ends the run at the sixth.
- `OutputValidator` replaces an empty final answer and truncates an oversized one.

`observe` measures the bare harness and has no gate between the model and its tools; use it for baselines on test
data. `[extra]` is where research middleware attaches (see [docs/attach.md](docs/attach.md)).

## Attach points

| Point | How |
|---|---|
| Run | Agent Protocol through `langgraph_sdk`, or `scripts/run.py`. Per run: `model`, `middleware`, `batch_id`, `session_type`, `entity` in `configurable` |
| Tools | Tool asset `binding.type`: `gateway` (the sample ERP and offline stubs), `http` (an `api` asset; values from `secret://` references), `mcp` (a server in `catalog/mcp.yaml`) |
| Middleware | Entry point group `wigtn_agents.middleware`, selected by name in the manifest (`extra_middleware`) or per run |
| Record | One JSONL row per event (`WIGTN_AGENT_EVENT_LOG`), plus traces to Langfuse / Phoenix / LangSmith (`OTEL_TARGETS`) |

Action gatherers and writeback ports attach the same way (`wigtn_agents.gatherers`, `wigtn_agents.writebacks`).

## Sub-agents and memory

`subagents: [erp-assistant]` makes another catalog agent a sub-agent: its prompts and skills become its system prompt
and its tools are its own. Sub-agent tool calls never pass through the parent's middleware, so each sub-agent gets a
full stack of its own: the stricter of its preset and the parent's, the extra middleware of both manifests and of the
run, and the parent's run context (scope, principal). Sub-agents use the parent's model, must be read-only, never
write `/memories/` and are one level deep. DeepAgents' auto-added `general-purpose` sub-agent, which would inherit
every parent tool, is always replaced by one without tools; its calls are recorded like any other.

`memory: true` mounts `/memories/` on the runtime store, one namespace per (agent, principal), and loads
`/memories/AGENTS.md` into the prompt at run start. Memory needs an identified principal: with `AUTH_TYPE=noop` every
caller is `anonymous`, one namespace would be shared by everyone, so memory stays off for those runs. Everything else
in the agent's filesystem is per-thread scratch.

## Run record

Every row carries `thread_id`, `run_id`, `batch_id`, `session_type`, `agent_id`, `kind` (`agent` · `action`), `role`
(`main` · `subagent`), `manifest_version`, the pinned `asset_versions`, `model_alias`, `model_id`, `model_provider`,
`preset` and `extra_middleware`. Event types are `run.start`, `llm.call` (with provider-reported token usage),
`tool.call` (name, `allowed`, `denied_by` = `permission` · `scope` · `filesystem`, `status`, latency; gather calls of an
action carry `step: gather`), `run.error` (error class only) and `run.end` (`ok` · `interrupted` · `error` ·
`cancelled`).

## Tracing

Every run is traced to Langfuse, Phoenix and LangSmith, any or all of them: `OTEL_TARGETS=LANGFUSE,PHOENIX,LANGSMITH`.
The spans come from one source, the OpenInference LangChain instrumentor, so the three tools get the same trace of a
run: one trace per run, the model and tool calls of the main agent, sub-agents and action steps under it. Langfuse and
Phoenix are Aegra's targets; LangSmith gets the same spans through its OTLP endpoint (`src/wigtn_agents/tracing.py`).
In-process runs (`scripts/run.py --local`, `scripts/eval.py`) trace to the same targets.

Each trace carries the session (`thread_id`), the user (principal) and metadata (`run_id`, `agent_id`, `batch_id`,
`session_type`, `model_alias`), so a trace and the run record rows of the same run share `run_id`.

LangSmith's own LangChain tracer (`LANGSMITH_TRACING=true`) is refused while an OTLP target is on. LangChain wraps
middleware `wrap_*` hooks in LangSmith runs that the OpenInference tracer never sees, so the calls inside every hook
would land in separate traces in Langfuse and Phoenix. Use `LANGSMITH` in `OTEL_TARGETS` instead.

The runtime traces and records; reading and analyzing traces is the job of the system attached at the record point.

## ERP catalog

The shipped agents cover four back-office areas of an ERP. Their tools are gateway routes on the sample ERP, an
in-memory system with two company codes and one month (October 2026) of data. Reads compute their answers (three-way
match, reorder need, receivables aging, trial balance, close checklist); writes change the state and are refused the
way an ERP refuses them (an unbalanced entry, a closed period, an unknown invoice). The data holds the exceptions an
agent should find, and the eval cases check for them; `tests/test_sample_erp.py` lists them.

| Agent | Lane | Work | Writes (after approval) |
|---|---|---|---|
| `erp-assistant` | agent | Questions about orders, invoices, stock, receivables, the trial balance | none |
| `ap-clerk` | agent | Three-way match of vendor invoices: price variance, quantity over receipt, possible duplicates | payment block |
| `inventory-planner` | agent | Materials below the reorder point once open orders count, quantity to request | purchase requisition |
| `close-accountant` | agent | Month-end checklist: recurring entries, accruals, GR/IR, unblocked exceptions, AR vs ledger | journal entry |
| `router` | agent | Hands a request to one of the above | none |
| `invoice-match-review` | action (`bounded-react`) | Exceptions and invoices to block | none (`jsonl`) |
| `replenishment-proposal` · `collections-priority` · `close-readiness` | action (`structured`) | Reorder proposal · collection order · close readiness | none (`jsonl`) |

The company code is the run's `entity` (`--entity 2000` in the scripts); the scope guard writes it into every ERP tool
call, so the model never chooses it. A server process keeps one ERP: writes of one run are seen by the next until the
process restarts. The same tools can point at a real ERP over HTTP: each tool asset already names an operation on
`api://erp-api@1`, so switching is `binding.type: http` plus `WIGTN_SECRET_ERP_BASE_URL`.

The runtime tests use a smaller fixed catalog (`tests/fixtures/catalog`), so they do not change when this one does.

## Layout

| Path | What |
|---|---|
| `catalog/assets/*.yaml` · `catalog/manifests/*.yaml` | Four asset kinds (prompt · skill · tool · api), five ERP agents and four ERP actions |
| `src/wigtn_agents/sample_erp/` | The in-memory ERP the catalog's tools call: seed data, rules, gateway routes |
| `catalog/models.yaml` · `catalog/mcp.yaml` | Model alias registry · MCP servers (optional file) |
| `src/wigtn_agents/assembly.py` | Agent lane: the only chat-lane `create_deep_agent` call site |
| `src/wigtn_agents/actions/` | Action lane: chassis and ports |
| `src/wigtn_agents/presets.py` · `middleware/` | Preset stacks and the platform middleware |
| `src/wigtn_agents/models.py` · `budget.py` | Model registry and the daily cost cap |
| `src/wigtn_agents/tools/` | Tool bindings (gateway · HTTP · MCP) |
| `src/wigtn_agents/worker.py` | Aegra factories (`graph`, `action`); the runtime injects checkpointer and store |
| `src/wigtn_agents/runner.py` · `scripts/` | In-process runs, run and eval scripts, assistant registration, e2e scripts |
| `docs/attach.md` · `docs/notes/` | How to attach a system · field notes checked against the installed packages |

## Status

Checked on 2026-11-03 with real models: gpt-6-luna and the MARA Cloud models (tool calling and structured output;
DeepSeek-V3.1 and MiniMax-M3 answered 429 that day), the DeepSeek-V3.2 judge, and traces arriving in Langfuse,
Phoenix and LangSmith, one trace per run.

## License

[MIT](LICENSE). Copyright (c) 2026 WIGTN and Harrison Kim.
