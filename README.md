# agi

An agent harness and **runtime engine** — built on Claude Opus 4.7, with
tools to act on the world, persistent memory, structured event streams,
self-extension at runtime, and a stable API surface so an outer
**coordination engine** can drive it as a worker.

Honest framing: this is not AGI — that remains an unsolved scientific
problem. It is what you can credibly ship today: a capable agent runtime
with the right primitives to plug into a larger system.

## What's in here

```
agi/
  agent.py            streaming agent loop — adaptive thinking + tool dispatch
  runtime.py          task lifecycle, parallel workers, capability manifest
  server.py           JSON-RPC HTTP server (the runtime over the wire)
  events.py           structured runtime events + EventBus
  budget.py           per-task USD / token / wall-clock ceilings
  capabilities.py     discovery manifest (tools, skills, roles, features)
  skills.py           procedural-memory library (markdown SOPs)
  sandbox.py          AST-restricted Python exec for tool synthesis
  synth_registry.py   session-scoped & persistent synthesized tools
  roles.py            planner / executor / critic / researcher / coder / summarizer
  reflect.py          per-task lesson distillation into long-term memory
  memory.py           persistent JSONL memory store
  costs.py            per-turn + cumulative token usage and $ tracking
  tools.py            file, shell, web, memory, skills, synth, delegate tools
  __main__.py         CLI: repl | oneshot | serve | manifest | task

learner/              learning track — small open base + LoRA loop
  traces.py           append-only JSONL trace logger
  filter.py           quality gates
  critic.py           trace-quality critic (CPU, no torch needed if unused)
  train.py            LoRA SFT script (GPU)

evals/                eval suite + runner
tests/                unit tests for everything (no API key required)
ARCHITECTURE.md       design: timescales, components, what we don't do
PLAN.md               stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Run it

```sh
python -m agi                          # interactive REPL
python -m agi "summarize ./README.md"  # one-shot
python -m agi task "find the prime factors of 91" --budget 0.05
python -m agi manifest                 # capability manifest as JSON
python -m agi serve --port 8765        # JSON-RPC server
python evals/run.py                    # run the eval suite
```

## What the agent can do

- Read/write files, run shell commands
- Search the live web (`web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions and recall by keyword
- Plan with adaptive thinking; show summarized thinking while streaming
- Track per-turn and cumulative token usage with $ cost
- **Define new tools at runtime** (sandboxed Python; promote to persistent)
- **Save and reuse procedural skills** (markdown SOPs; loaded by relevance)
- **Delegate to specialist subagents** (planner / executor / critic / …)
- **Reflect** after a task and write durable lessons to memory

## Runtime API for coordination engines

The runtime is the layer above the agent that a coordination engine drives.
It owns task lifecycle, budgets, the event bus, the skill library, the
synthesized-tool registry, and parallel workers.

### Why a runtime, not just an agent?

A coordination engine is responsible for *what work runs, in what order,
under what budget, across which workers*. The runtime is the worker it
talks to. Three things make that integration tractable:

1. **Structured events.** Every state change emits a typed `Event` —
   `task.started`, `task.tool_call`, `task.subagent_spawned`, `task.completed`,
   etc. The coordinator subscribes and reacts (route, retry, escalate, log).
2. **A capability manifest.** Before sending work, the coordinator asks
   "what can you do?" and gets back a stable JSON description: tools (with
   schemas), skills (with usage counts), roles (with system prompts),
   features (events / skills / synth / delegation / reflection / traces),
   limits (max workers).
3. **Bounded tasks.** Every submission carries a `Budget` (USD, tokens,
   wall-seconds). The runtime enforces it between agent turns and emits
   `task.budget_exceeded` when it trips. A coordinator that schedules
   thousands of tasks per hour needs this; without it the system stalls
   on one runaway task.

### Python API

```python
from agi import Runtime, Budget

rt = Runtime(max_workers=4, enable_reflection=True)

# Discovery
manifest = rt.manifest()         # CapabilityManifest
print(manifest.to_dict())

# Submit a task
handle = rt.submit(
    "find the prime factors of 91",
    role="executor",
    budget=Budget(max_usd=0.05, max_wall_seconds=30),
)

# Subscribe to events
rt.subscribe(lambda ev: print(ev.kind, ev.data))

# Wait for it (or poll handle.status())
snap = handle.wait(timeout=60)
print(snap["status"], snap["result"], snap["cost_usd"])

# Parallel
handles = rt.submit_batch(["task A", "task B", "task C"], role="executor")
results = rt.wait_all([h.id for h in handles], timeout=120)

rt.shutdown()
```

### HTTP / JSON-RPC API

For coordinators running in another process or another language:

```
GET  /healthz                liveness
GET  /manifest               capability manifest
POST /tasks                  submit (returns {id}, 202)
GET  /tasks                  list tasks (filter ?status=running)
GET  /tasks/{id}             status snapshot
GET  /tasks/{id}/wait        block until done (?timeout=60)
POST /tasks/{id}/cancel      cancel a running task
GET  /events                 SSE stream of runtime events
```

```sh
python -m agi serve --port 8765 &

curl -s localhost:8765/manifest | jq '.tools | length'
# 16

curl -X POST localhost:8765/tasks \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "say hi", "budget": {"max_usd": 0.01}}'
# {"id": "abc123...", "status": "pending"}

curl -N localhost:8765/events
# event: task.submitted
# data: {"kind": "task.submitted", ...}
# event: task.tool_call
# data: ...
```

### Self-extension at runtime

The agent can extend its own capabilities mid-task:

- **`define_tool(name, description, source, input_schema)`** — write a
  Python function the runtime compiles in a restricted AST sandbox
  (no imports beyond a pre-loaded allowlist of math/json/re/etc., no
  `exec`/`eval`, no dunder attribute access, 5s timeout). Session-scoped
  by default. `promote_tool(name)` persists it across runtime restarts.
- **`add_skill(name, when_to_use, procedure, failure_modes)`** — save a
  procedural skill (markdown SOP). On the next task with overlapping
  keywords, the top-K relevant skills are loaded into the system prompt.
- **`delegate(role, task)`** — spawn a child agent in a named role
  (planner / executor / critic / researcher / coder / summarizer). Each
  role has its own system prompt and default model so the coordinator
  routes cheap work to Haiku and reserves Opus for hard reasoning.

### Reflection (optional)

When `enable_reflection=True`, the runtime runs a small Haiku call after
each non-trivial task asking the model "what's the durable lesson worth
remembering?" Useful lessons get saved to long-term memory tagged
`lesson` and surface in future tasks via memory search. Trivial tasks are
gated out so the journal stays signal-rich.

## Architecture

A coordination engine orchestrates multiple agi runtimes; each runtime
orchestrates multiple agents; each agent uses a frozen Opus reasoning
core with tools, memory, skills, and (optionally) the learning track's
LoRA adapter as a second reasoning core. The boundaries are clean:

```
            ┌────────────────────────────────────┐
            │       Coordination Engine          │
            │  (out of scope: yours, not ours)   │
            └────────┬──────────────┬────────────┘
                     │ JSON-RPC     │ JSON-RPC
                     ▼              ▼
            ┌──────────────┐  ┌──────────────┐
            │  agi Runtime │  │  agi Runtime │   ← horizontally scalable
            └──┬──────┬────┘  └──────────────┘
               │      │
               │      ▼ events
               │   ┌──────────┐
               │   │ Subscriber│ ← your dashboard / queue / planner
               │   └──────────┘
               ▼
        ┌─────────────────────────────────────┐
        │ Agent (Opus 4.7) + Memory + Skills  │
        │ + Synth tools + Subagents + Budgets │
        └─────────────────────────────────────┘
```

See `ARCHITECTURE.md` for the dual-track (frozen-Opus + learning-LoRA)
design and the honest list of what this architecture does *not* do.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement at the weight
level, robust transfer, grounded world models, durable goals over weeks
of unsupervised operation. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable
improvement through weight updates on a small open base. The runtime
shipped here is the substrate those weights would run inside.

## Tests

```sh
python -m unittest discover tests/
```

75 tests, no API key required. The Runtime / Server tests stub the
`Agent` class with a fake so the full task-lifecycle path is exercised
without API calls.
