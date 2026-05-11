# Runtime engine

The runtime is the executor a coordination engine dispatches to. The CLI in
`agi.__main__` is one consumer; the HTTP server in `agi.server` is another;
any external orchestrator (a Temporal-style scheduler, a CrewAI-style
multi-agent layer, a per-tenant queue) is another.

This document describes the contract. Read `ARCHITECTURE.md` first if you
want the broader system context.

## The split between scheduler and executor

The runtime is *deliberately small*. It does not:

- persist runs across process restarts
- retry failed runs
- decide *which* run to start next
- own multi-tenant quotas or rate limits

It does exactly four things:

1. Accept a task and start running it in a thread.
2. Honor cooperative cancellation and a cost ceiling.
3. Emit a structured event stream while the run executes.
4. Hand back the final result + cost + usage when done.

Everything above this — persistence, retries, scheduling, fan-out
strategies, rate limiting per tenant — is the coordinator's job. This split
keeps the runtime composable: the same executor sits underneath a Temporal
worker, a webhook endpoint, a long-poll queue worker, or a unit test.

## The core abstractions

```python
from agi import Runtime, RunStatus

rt = Runtime()
run = rt.submit(
    task="summarize ./README.md",
    cost_ceiling_usd=0.50,
    timeout_seconds=300,
    metadata={"tenant": "acme", "trace_id": "..."},
)
# returns immediately — `run.status == RunStatus.RUNNING` (or PENDING briefly)

run.wait(timeout=600)              # block until terminal
print(run.status, run.result, run.cost_usd)

for ev in run.stream(replay=True): # observe events as they happen
    print(ev.type, ev.payload)

rt.cancel(run.id)                  # cooperative cancel
```

### `Run` lifecycle

```
        ┌─────────┐
        │ PENDING │  (briefly, between submit() and thread start)
        └────┬────┘
             ▼
        ┌─────────┐
        │ RUNNING │
        └────┬────┘
             │
   ┌─────────┼──────────────────────────────────────────┐
   ▼         ▼                ▼                          ▼
SUCCEEDED  FAILED         CANCELLED                BUDGET_EXCEEDED / TIMED_OUT
```

Terminal states are sticky. `cancel()` on a terminal run is a no-op.

### Events

Each run carries a per-run `_EventBus`. The bus records history (capped at
1024 events) and fans out to live subscribers via per-subscriber queues
(also capped, oldest-event-dropped if a subscriber falls behind). Slow
subscribers never block the agent.

Event types emitted by the default agent factory:

| Event type            | When                                | Payload                                       |
|-----------------------|-------------------------------------|-----------------------------------------------|
| `run.started`         | Runtime starts the thread           | `{task}`                                      |
| `turn.completed`      | After each LLM turn                 | `{input_tokens, output_tokens, stop_reason, cumulative_cost_usd}` |
| `tool.call`           | Before a tool handler runs          | `{name, id}`                                  |
| `tool.result`         | After a tool handler returns        | `{name, id, is_error}`                        |
| `delegate.spawned`    | A subagent run was submitted        | `{child_id, role, task}`                      |
| `run.succeeded`       | Terminal: success                   | `{result_chars}`                              |
| `run.failed`          | Terminal: unexpected exception      | `{error}`                                     |
| `run.cancelled`       | Terminal: cancellation requested    | `{}`                                          |
| `run.budget_exceeded` | Terminal: cost ceiling crossed      | `{error}`                                     |
| `run.timed_out`       | Terminal: timeout reached           | `{}`                                          |

A coordinator that wants visibility but doesn't want to subscribe live can
poll `run.events()` for the full history at any time.

### Cancellation and budget enforcement are cooperative

Both check between turns (and after each tool dispatch). A stuck tool call
cannot be force-killed — the runtime would have to terminate the thread,
which Python doesn't offer safely. If you need hard time limits, run the
runtime behind a process supervisor that can SIGKILL the whole process.

## HTTP surface

`python -m agi.server --host 0.0.0.0 --port 8000` starts a stdlib HTTP
server. Endpoints under `/v1`:

| Method | Path                       | Body                                                              | Returns                                       |
|--------|----------------------------|-------------------------------------------------------------------|-----------------------------------------------|
| POST   | `/v1/runs`                 | `{"task": str, "cost_ceiling_usd"?: float, "timeout_seconds"?: float, "metadata"?: object}` | 201, the Run as JSON                          |
| GET    | `/v1/runs`                 | —                                                                 | 200, list of Runs                             |
| GET    | `/v1/runs/{id}`            | —                                                                 | 200, the Run as JSON (or 404)                 |
| POST   | `/v1/runs/{id}/cancel`     | —                                                                 | 200, `{"id", "cancelled", "status"}`          |
| GET    | `/v1/runs/{id}/events`     | —                                                                 | 200, `text/event-stream` (SSE)                |
| GET    | `/healthz`                 | —                                                                 | 200, `{"ok": true}`                           |

SSE format: each event is

```
event: <type>
data: {"ts": ..., "run_id": ..., "type": ..., "payload": {...}}

```

Replays history then streams live. The connection closes when the run
reaches a terminal state.

## How a coordination engine uses this

The minimum viable integration is "submit, observe, collect". A richer one
spans multiple runs across many workers:

```
┌─────────────────────────┐
│  Coordination engine     │   owns: task queue, retries, scheduling,
│  (Temporal / Inngest /   │          tenancy, persistence, rate limits
│   custom)                │
└───────────┬─────────────┘
            │ HTTP POST /v1/runs
            │ HTTP GET  /v1/runs/{id}/events
            │ HTTP POST /v1/runs/{id}/cancel
            ▼
┌─────────────────────────┐
│  agi.Runtime (worker)    │   owns: agent loop, tool dispatch, budget,
│                          │          event emission, cooperative cancel
└─────────────────────────┘
```

Concretely:

1. The coordinator pulls a task from its own queue, with whatever
   scheduling logic it wants. Per-tenant quotas, fairness, priorities,
   retries — all the coordinator's problem.
2. It calls `POST /v1/runs` against a worker. The response carries the
   run id.
3. It subscribes to `/v1/runs/{id}/events` (SSE) for live observability.
   Or it polls `GET /v1/runs/{id}` if streaming is overkill.
4. On a terminal event, it persists the result + cost into its own
   datastore. The runtime itself doesn't persist runs across restarts.
5. If the task should be retried, the coordinator re-submits. If a
   deadline is missed, the coordinator cancels via `POST .../cancel`.

The runtime's job ends at "produced a result, charged a cost, emitted
events". The coordinator decides what that means for the system.

## Subagent delegation

A running agent can call `delegate(task=..., role=..., cost_ceiling_usd=...,
timeout_seconds=...)` to spawn a *child run* on the same Runtime. The child
runs in its own thread with its own conversation context, returns its final
text, and the parent agent sees that text as a tool result.

Depth is capped (default 3) to prevent runaway recursion. A `delegate.spawned`
event fires on the parent's bus carrying the child's run id, so a coordinator
that subscribes to the parent can also subscribe to children.

```python
runtime = Runtime()
parent = runtime.submit(
    "research how Postgres handles concurrent migrations, then write a checklist",
    cost_ceiling_usd=1.50,
)

# The agent will probably:
# 1. delegate(task="search the web for Postgres migration safety", role="researcher")
# 2. delegate(task="critique the draft checklist for missing edge cases", role="critic")
# 3. integrate both results and return a final checklist.

for ev in parent.stream():
    if ev.type == "delegate.spawned":
        child_id = ev.payload["child_id"]
        # The coordinator can now also tail /v1/runs/{child_id}/events.
```

This is the smallest credible multi-agent primitive. It is *not* magic — if
the subtasks aren't genuinely separable, decomposition costs more tokens
than it saves. Watch `cost_usd` on rollups, not just pass rate.

## Skills

Skills are markdown files in `~/.agi/skills/` with a simple front-matter
header (`name`, `when`, `tags`). The default agent factory wires four
tools onto the agent:

- `list_skills()` — name + trigger condition for every saved skill
- `search_skills(query, k=3)` — top-K relevant skills (full body)
- `load_skill(name)` — fetch one by name
- `save_skill(name, when, body, tags)` — promote a successful procedure

Skills are the "hours" timescale of adaptation in `ARCHITECTURE.md` —
procedural knowledge the agent can grow across tasks without retraining.
Read that doc for where this sits in the bigger picture and what's
deliberately *not* claimed about it.

## Tool synthesis (opt-in)

A `make_tool(name, description, input_schema, code)` tool lets the agent
write a Python function and register it as a callable tool for the rest of
the session. Imports are restricted to a vetted standard-library
allowlist; the synthesized tool runs with a wall-clock timeout.

**This is not safe by default** — exec'ing model-written Python in-process
has the same trust profile as `run_bash`. It is **not** enabled in the
default agent factory. To enable it, pass your own factory to `Runtime`:

```python
from agi import Runtime
from agi.agent import Agent
from agi.tool_synthesis import make_tool_synthesis

def factory_with_synthesis(run, runtime):
    agent = Agent(
        verbose=False,
        cancel_event=run._cancel,
        cost_ceiling_usd=run.cost_ceiling_usd,
        runtime=runtime,
        run_id=run.id,
    )
    schema, handler = make_tool_synthesis(agent.tool_schemas, agent.handlers)
    agent.tool_schemas.append(schema)
    agent.handlers["make_tool"] = handler
    return agent

rt = Runtime(agent_factory=factory_with_synthesis)
```

Long-term: synthesized tools should run in a subprocess or container.

## What's deliberately not in here

In keeping with the rest of this repo, here is what the runtime *does not*
solve:

- **Hard-killing stuck tools.** Cooperative only; a tight infinite loop in a
  synthesized tool will exceed timeout but leave the thread alive until the
  process restarts.
- **Cross-process state.** The runtime is process-local. Coordinators that
  need durable run state implement it on top.
- **Authentication / authorization.** The HTTP server is meant to live
  behind a coordinator that owns auth.
- **Streaming the agent's *text output* to the caller.** The event stream
  surfaces metadata (turns, tool calls, status). For the model's text, the
  caller fetches `run.result` after `run.succeeded`. (We could add a
  `text.delta` event later if a coordinator needs it.)
- **Adapter hot-swap during a run.** Adapters are session-scoped per the
  learner architecture; mid-run weight swaps are out of scope.

Naming these keeps the contract honest.
