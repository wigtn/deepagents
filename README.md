# wigtn/deepagents

An agent runtime for testing LLM agent harnesses. [Aegra](https://github.com/ibbybuilds/aegra) (a self-hosted Agent
Protocol server) runs the workers, and [DeepAgents](https://github.com/langchain-ai/deepagents) assembles each agent
from a manifest. Research systems attach to it at four fixed points (run, tools, middleware, record) without
changing the runtime.

A harness here is everything a manifest fixes around the model: prompts, skills, tools, sub-agents, the approval
points and the middleware stack. Because the manifest is the whole harness, the configuration file is the ground
truth of what the agent is. The research project [wigtn/harness-extractors](https://github.com/wigtn/harness-extractors)
uses this repository as its real-agent testbed.

The catalog ships with a small example domain (reading and updating records) so that the runtime and the tests run
end to end. It is a placeholder, not a target domain.

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
cp .env.example .env                     # fill in OPENAI_API_KEY (default model) or use models/fake
docker compose up -d postgres            # pgvector/pg18 on host port 5433
uv run aegra db upgrade                  # runtime migrations
uv run aegra dev                         # http://localhost:2026/docs
PYTHONPATH=src uv run python scripts/register_assistants.py   # catalog → assistants (both graphs)
```

One run, through the server or in-process:

```bash
PYTHONPATH=src uv run python scripts/run.py records-guide "8월 기록 점검 부탁해"
PYTHONPATH=src uv run python scripts/run.py records-summary '{"period": "2026-08", "view": "open-items"}'
PYTHONPATH=src uv run python scripts/run.py records-writer "8월 기록 반영" --local --model models/fake --approve
```

Eval cases carried by the assets: `PYTHONPATH=src uv run python scripts/eval.py records-guide --model <alias>`.
A case is graded by `contains` (expected strings appear), `exact` (equal text) or `llm` (a judge model grades the
answer against the case's rubric: criteria with weights and a `pass_score`; the score is computed from the per-criterion
calls, not asked of the judge). The judge (`--judge-model`, or `judge` in `catalog/models.yaml`) must be another
provider model than the backbone; the eval stops before the first run otherwise. `--no-llm` skips `llm` cases.

Offline tests (no model calls): `uv run pytest -q`. Lint: `uv run ruff check . && uv run ruff format --check .`.
Types: `uv run mypy`.

`package = false`, so ad hoc scripts need `PYTHONPATH=src` (Aegra puts the same path on `sys.path` through
`dependencies: ["./src"]` in `aegra.json`).

## Models

`catalog/models.yaml` is the alias registry. A manifest names an alias (`model.ref`); a run can switch it with
`configurable.model`, so the same harness runs on another backbone without editing the manifest.

| Alias | Model | API |
|---|---|---|
| `models/default` | gpt-6-luna | OpenAI Responses API |
| `models/qwen3.8-flash` · `models/deepseek-v3.1` · `models/gpt-oss-120b` | open-weight backbones | OpenAI-compatible endpoint (vLLM or a provider API) |
| `models/fake` | scripted offline model | none |

An alias that is not registered is refused, and so is an alias without a price: every real call goes through a daily
cost cap (`WIGTN_AGENT_DAILY_COST_CAP_USD`) that must be able to charge it. The open-weight aliases have no endpoint and
no price yet, so they are refused until both are filled in. Manifest parameters an entry does not `accept` (for
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
- `ScopeGuard` overwrites scope arguments (`entity`) with the run context's value; no value, no call.
- `RepeatedToolCall` warns once at the third identical tool call in a row and ends the run at the sixth.
- `OutputValidator` replaces an empty final answer and truncates an oversized one.

`observe` measures the bare harness and has no gate between the model and its tools; use it for baselines on test
data. `[extra]` is where research middleware attaches (see [docs/attach.md](docs/attach.md)).

## Attach points

| Point | How |
|---|---|
| Run | Agent Protocol through `langgraph_sdk`, or `scripts/run.py`. Per run: `model`, `middleware`, `batch_id`, `session_type`, `entity` in `configurable` |
| Tools | Tool asset `binding.type`: `gateway` (offline stub), `http` (an `api` asset; values from `secret://` references), `mcp` (a server in `catalog/mcp.yaml`) |
| Middleware | Entry point group `wigtn_agents.middleware`, selected by name in the manifest (`extra_middleware`) or per run |
| Record | One JSONL row per event (`WIGTN_AGENT_EVENT_LOG`), plus traces to Langfuse / Phoenix / LangSmith (`OTEL_TARGETS`) |

Action gatherers and writeback ports attach the same way (`wigtn_agents.gatherers`, `wigtn_agents.writebacks`).

## Sub-agents and memory

`subagents: [records-guide]` makes another catalog agent a sub-agent: its prompts and skills become its system prompt
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

## Layout

| Path | What |
|---|---|
| `catalog/assets/*.yaml` · `catalog/manifests/*.yaml` | Four asset kinds (prompt · skill · tool · api), four example agents and two example actions |
| `catalog/models.yaml` · `catalog/mcp.yaml` | Model alias registry · MCP servers (optional file) |
| `src/wigtn_agents/assembly.py` | Agent lane: the only chat-lane `create_deep_agent` call site |
| `src/wigtn_agents/actions/` | Action lane: chassis and ports |
| `src/wigtn_agents/presets.py` · `middleware/` | Preset stacks and the platform middleware |
| `src/wigtn_agents/models.py` · `budget.py` | Model registry and the daily cost cap |
| `src/wigtn_agents/tools/` | Tool bindings (gateway stub · HTTP · MCP) |
| `src/wigtn_agents/worker.py` | Aegra factories (`graph`, `action`); the runtime injects checkpointer and store |
| `src/wigtn_agents/runner.py` · `scripts/` | In-process runs, run and eval scripts, assistant registration, e2e scripts |
| `docs/attach.md` · `docs/notes/` | How to attach a system · field notes checked against the installed packages |

## Status

Not yet done: the open-weight endpoints and prices, and a registered judge model.

## License

[MIT](LICENSE). Copyright (c) 2026 WIGTN and Harrison Kim.
