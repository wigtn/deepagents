# Attaching a system

A research or product system attaches to this runtime at one of four points. None of them needs a change to the
runtime itself: each one selects or adds code by name, and the run record shows what was attached.

| Point | Use it for | Where it is selected |
|---|---|---|
| Run | Driving agents: batches, backbone sweeps, replays | `configurable` of each run |
| Tools | Giving an agent a real system to call | `binding` of a tool asset |
| Middleware | Code that watches or changes model and tool calls (defenses, probes, rewrites) | `extra_middleware` in the manifest, or `configurable.middleware` |
| Record | Reading what happened (analyzers, dashboards) | `WIGTN_AGENT_EVENT_LOG`, OpenTelemetry targets, LangSmith |

## 1. Run

Agents run on graph `worker`, actions on graph `action`. Register the catalog once
(`scripts/register_assistants.py`); assistant ids are `uuid5` of the agent name (`scripts/e2e_common.py`).

Send the full `configurable` on every run. Aegra merges the assistant's config and the run's config shallowly, so a
partial `configurable` drops `agent_id`.

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2026")
thread = await client.threads.create()
await client.runs.wait(
    thread["thread_id"],
    assistant_id,  # uuid5 of "records-guide"
    input={"messages": [{"role": "user", "content": "8월 기록 점검"}]},
    config={
        "configurable": {
            "agent_id": "records-guide",
            "batch_id": "sweep-01",  # stamped on every event row
            "session_type": "evaluation",  # production | evaluation | test
            "entity": "1000",  # scope the scope guard injects (configurable wins over context)
            "model": "models/gpt-oss-120b",  # optional: another registered alias
            "middleware": ["my-defense"],  # optional: registered extra middleware
        }
    },
)
```

An action takes `input={"input": {...}}` and leaves `result` and `writeback` in the thread state.
`stream_mode=["custom"]` streams progress markers (`{"type": "progress", "kind": "tool" | "step", ...}`).

The same harness on several backbones is one loop over `model`; nothing else changes, and every row carries
`model_alias`, `model_id` and `model_provider`. `scripts/run.py --local` runs the same assembly in-process.

## 2. Tools

A tool asset keeps its own `input_schema`; the binding only says where the call goes.

```yaml
# HTTP: base URL and credential come from the api asset's secret:// references
binding: {type: http, api: api://records-api@1, operation: "GET /records/{record_id}"}

# MCP: a tool on a server listed in catalog/mcp.yaml
binding: {type: mcp, server: docs, tool: search}

# Gateway: a named route on the offline stub (tests, demos)
binding: {type: gateway, gateway_route: record-read}
```

`secret://records/base-url` is read from `WIGTN_SECRET_RECORDS_BASE_URL` at call time, never stored. A credential is
sent as `Authorization: Bearer <value>`. Path parameters come from the tool arguments; the other arguments go to the
query string (GET, DELETE) or the JSON body. An HTTP error comes back to the model as a tool error.

`catalog/mcp.yaml`:

```yaml
servers:
  docs:
    transport: streamable_http          # stdio | streamable_http | sse
    url_env: WIGTN_MCP_DOCS_URL
    headers: {Authorization: "secret://mcp/docs-token"}
  files:
    transport: stdio
    command: uvx
    args: [some-mcp-server]
```

A `write` tool needs an approval point (`interrupts.approve_before`) and the `approval-gated` flow. A tool that takes
a scope argument must be listed in `SCOPE_PARAMS` (`middleware/scope_guard.py`); a test fails otherwise.

## 3. Middleware

Package the middleware and register a factory:

```toml
[project.entry-points."wigtn_agents.middleware"]
my-defense = "my_package.middleware:build"
```

```python
from langchain.agents.middleware import AgentMiddleware
from wigtn_agents.middleware.extra import MiddlewareSpec


class MyDefense(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        ...  # inspect, rewrite or deny, then
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


def build(spec: MiddlewareSpec) -> AgentMiddleware:
    # spec.manifest, spec.ctx (the run context); spec.schema_only is true when the
    # runtime builds the read-only graph, so do not open connections here.
    return MyDefense()
```

Install it into the runtime's environment (`uv add` or `uv pip install -e`), then name it in a manifest
(`extra_middleware: [my-defense]`) or per run (`configurable.middleware`). Unknown names fail the run; nothing is
loaded from a request. Run-level names reach every agent loop of the run: the main agent, its catalog sub-agents and
an action's tool loop. Code steps (an action's gather and writeback) are not wrapped, and a `structured` action has no
loop, so it refuses extra middleware instead of recording it as attached.

The slot is right inside the event recorder and outside the permission gate: the record shows what your middleware
did, and the gate still applies to whatever it lets through. Implement both the sync and the async hook; the runtime
runs graphs with `ainvoke`. Prefer `wrap_model_call` and `wrap_tool_call`: `before_*` and `after_*` hooks add graph
nodes, and the read-only graph then has to include them (it does, for every registered factory, which is why
`schema_only` exists).

Action ports attach the same way:

```toml
[project.entry-points."wigtn_agents.gatherers"]
my-source = "my_package.ports:gather"      # gather(GatherRequest) -> dict
[project.entry-points."wigtn_agents.writebacks"]
my-store = "my_package.ports:write"        # write(WritebackRequest) -> dict
```

## 4. Record

`WIGTN_AGENT_EVENT_LOG` (default `var/events.jsonl`) gets one JSON row per event:

| Field | Meaning |
|---|---|
| `thread_id` · `run_id` · `batch_id` · `parent_run_id` | Where the row belongs |
| `agent_id` · `kind` · `role` | Which manifest, which lane, main agent or sub-agent |
| `manifest_version` · `asset_versions` | What the harness was, pinned at run start |
| `model_alias` · `model_id` · `model_provider` | Which backbone |
| `preset` · `extra_middleware` | Which middleware stack |
| `session_type` · `actor_type` · `actor_id` | Who ran it |
| `event_type` | `run.start` · `llm.call` (`usage`) · `tool.call` (`tool.name`, `tool.allowed`, `tool.denied_by`, `tool.status`, `tool.latency_ms`, `step` for gather calls) · `run.error` (`error_class`) · `run.end` (`status`) |

Error messages are never recorded, only the error class, because a provider message can carry user data. Langfuse and
Phoenix get OpenInference spans through `OTEL_TARGETS`; LangSmith gets LangChain's own trace. Join them to the JSONL
rows on `thread_id` and `run_id`.
