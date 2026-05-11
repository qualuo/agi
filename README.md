# agi

An agent **runtime engine** for coordination engines to drive. Not AGI —
that remains an unsolved scientific problem. This is what you can credibly
build today:

- A capable agent on Claude Opus 4.7 with tools, persistent memory, and an
  eval harness.
- A **Runtime** that exposes the agent as a session/task substrate other
  systems compose: sessions, structured tasks, budgets, cancellation,
  streamed events, delegation to subagents, durable skills.
- An **HTTP+SSE transport** so a coordination engine in any language can
  drive the runtime cross-process.
- A learning track (`learner/`) that trains a small open base with LoRA on
  filtered traces — the only honest path to durable improvement on top of
  a frozen frontier model.

## What's in here

```
agi/
  agent.py        # streaming agent loop — adaptive thinking + tool dispatch
  budget.py       # token / $ / time / iteration ceilings
  costs.py        # per-turn + cumulative token usage + $ tracking
  memory.py       # persistent JSONL memory store (keyword search; embeddings later)
  reflection.py   # post-task auto-reflection → memory notes
  runtime.py      # Runtime engine: sessions, tasks, events, delegation
  server.py       # HTTP + SSE transport (stdlib-only)
  skills.py       # durable procedural memory (markdown SOPs, eval-gated promotion)
  tools.py        # filesystem, shell, web search/fetch, memory tools
  __main__.py     # CLI: REPL / serve / caps / skills
learner/          # learning track — small open base + LoRA loop + trace critic
evals/            # task suite + runner
tests/            # smoke + runtime + critic tests
ARCHITECTURE.md   # full design
PLAN.md           # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Use it as a single agent

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

## Use it as a runtime for a coordination engine

In-process (Python coordinator):

```python
from agi import Runtime, SessionConfig, Budget

runtime = Runtime()                                 # one substrate
session = runtime.open_session(SessionConfig(
    model="claude-opus-4-7",
    budget=Budget(max_usd=2.0, max_seconds=120),
    skill_ids=["refactor-python-module"],           # always-on skills
))

result = runtime.run_task(
    session.id,
    "rename methodName to method_name across this repo",
    budget=Budget(max_usd=0.50),                    # per-task cap
)
print(result.status, result.cost_usd, result.output)

runtime.close_session(session.id)
```

Cross-process (any-language coordinator over HTTP+SSE):

```sh
python -m agi serve --host 127.0.0.1 --port 8088    # start the runtime
# optional: export AGI_RUNTIME_TOKEN=secret           # then send Bearer auth
```

```sh
# 1. Inspect capabilities
curl -s http://127.0.0.1:8088/v1/capabilities

# 2. Open a session
curl -s -X POST http://127.0.0.1:8088/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-opus-4-7","budget":{"max_usd":2.0}}'
# -> {"id": "abc123...", ...}

# 3. Run a task (synchronous)
curl -s -X POST http://127.0.0.1:8088/v1/sessions/abc123.../tasks \
  -H 'Content-Type: application/json' \
  -d '{"input":"summarize README.md"}'

# 4. Run a task asynchronously and stream events
curl -s -X POST 'http://127.0.0.1:8088/v1/sessions/abc123.../tasks?async=1' \
  -H 'Content-Type: application/json' \
  -d '{"input":"refactor module X"}'
curl -N http://127.0.0.1:8088/v1/sessions/abc123.../tasks/<tid>/events
```

### Why this shape for a coordination engine

A coordinator above the runtime wants four things, and the API exposes each
as a first-class primitive:

| Coordinator need | Runtime primitive |
|---|---|
| Pick the right worker for a job | `GET /v1/capabilities` — tools, skills, models, supports |
| Bound the cost of any single subtask | `Budget` per task or per session — tokens, $, seconds, iterations |
| Observe progress and intervene | SSE event stream + `POST .../cancel` |
| Compose subtasks into a workflow | `delegate` tool + parent/child sessions with rolled-up accounting |

The runtime is intentionally **not** a queue, scheduler, or router. Those
belong above this layer in the coordination engine.

## What it can do today

- **Agent capability** — read/write files, run shell, web search/fetch, persistent
  memory, adaptive thinking, summarized streaming.
- **Runtime substrate** — multiple concurrent sessions, structured task lifecycle,
  budgets (tokens / $ / time / iterations), cancellation, SSE event streaming,
  capability introspection.
- **Subagent delegation** — the `delegate` tool spawns a fresh child session
  scoped by `max_usd` / `max_seconds`; usage rolls up to the parent's view.
- **Skill library** — markdown SOPs versionable in git; matched on triggers,
  eval-gated promotion before injection into the system prompt.
- **Auto-reflection** — after each task the runtime writes a 1-line lesson to
  long-term memory on failures and on recovered runs.
- **Trace-quality critic** — a tiny CPU model gates outputs below a confidence
  threshold (opt-in).

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See [ARCHITECTURE.md](ARCHITECTURE.md).
