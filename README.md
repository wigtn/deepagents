# wigtn/deepagents

An agent runtime for testing LLM agent harnesses. [Aegra](https://github.com/ibbybuilds/aegra) (a self-hosted Agent
Protocol server) runs the workers, and `create_deep_agent` from [DeepAgents](https://github.com/langchain-ai/deepagents)
assembles each agent from a manifest. The two meet at one point only, the graph factory in `aegra.json`
(`src/wigtn_agents/worker.py:graph`).

A harness here is everything a manifest fixes around the model: prompts, skills, tools, sub-agents, the approval
points and the middleware stack. Because the manifest is the whole harness, the configuration file is the ground
truth of what the agent is. This is what the research project
[wigtn/harness-extractors](https://github.com/wigtn/harness-extractors) uses as its real-agent testbed.

The catalog ships with a small example domain (reading and updating records) so that the runtime and the tests run
end to end. It is a placeholder, not a target domain.

## Layers

```
client (langgraph_sdk) ──POST /threads/{id}/runs──▶ Aegra (API · queue · checkpointer · SSE · HITL)
                                                      │ aegra.json graphs.worker (called per request)
                                                      ▼
                                 worker.graph(config, runtime) → assemble(manifest, ctx)
                                                      ▼
                 create_deep_agent(model, tools, middleware=[AssetLoader, EventRecorder,
                    PermissionFilter, ScopeGuard, OutputValidator], subagents=[general-purpose suppressed],
                    interrupt_on=approve_before, checkpointer=None, store=None)
```

## Run

```bash
uv sync                                  # Python 3.12 · aegra-api 0.10.x · deepagents 0.7.x
cp .env.example .env                     # fill in a model key
docker compose up -d postgres            # pgvector/pg18 on host port 5433
uv run aegra db upgrade                  # runtime migrations
uv run aegra dev                         # http://localhost:2026/docs
PYTHONPATH=src uv run python scripts/register_assistants.py   # catalog → assistants
```

Offline tests (no model calls): `uv run pytest -q`. Lint: `uv run ruff check . && uv run ruff format --check .`.
Types: `uv run mypy`.

`package = false`, so ad hoc scripts need `PYTHONPATH=src` (Aegra puts the same path on `sys.path` through
`dependencies: ["./src"]` in `aegra.json`). To run without a model key, set `WIGTN_AGENT_FAKE_MODEL=1` to use a
scripted model.

Every real model call goes through a daily cost cap (`WIGTN_AGENT_DAILY_COST_CAP_USD`). A model without a known price
is refused before the first call; set both `WIGTN_AGENT_PRICE_*_PER_MTOK` values for models not listed in
`src/wigtn_agents/budget.py`.

## Observability

Langfuse and Phoenix receive spans through Aegra's OpenTelemetry fan-out (`OTEL_TARGETS`). LangSmith uses LangChain's
own tracer (`LANGSMITH_TRACING`). The settings are in the Observability block of `.env.example`; leaving a tool's
settings empty turns it off.

The three tools do not record the same shape (checked 2026-11-01). LangSmith keeps one run as one trace, including the
middleware `awrap_model_call` and `awrap_tool_call` hooks as spans. The OpenInference records sent to Langfuse and
Phoenix have no span for those hooks, so model and tool calls made inside them lose their parent and become separate
traces (one run, four traces). Each piece still carries the `thread_id` and `run_id` metadata that Aegra adds.

## Layout

| Path | What |
|---|---|
| `catalog/assets/*.yaml` · `catalog/manifests/*.yaml` | Four asset kinds (prompt · skill · tool · api) and four example manifests |
| `src/wigtn_agents/assets/` | Asset schema and catalog (`@active` pinned to an integer version per run) |
| `src/wigtn_agents/manifest.py` | Manifest schema and the flow decision table |
| `src/wigtn_agents/middleware/` | The five platform middlewares |
| `src/wigtn_agents/assembly.py` | The only `create_deep_agent` call site |
| `src/wigtn_agents/worker.py` | Aegra factory; the runtime injects checkpointer and store |
| `src/wigtn_agents/models.py` · `budget.py` | Model alias registry and the daily cost cap |
| `scripts/` | Assistant registration and e2e scripts |
| `docs/notes/` | Field notes checked against the installed packages |

## Status

The model registry connects Anthropic models only. OpenAI-compatible endpoints (vLLM and provider APIs for
open-weight models), a harness instance generator and research domains are planned.

## License

[MIT](LICENSE). Copyright (c) 2026 WIGTN and Harrison Kim.
