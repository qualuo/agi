# agi

An agent runtime. Not AGI in the textbook sense — that remains an unsolved
scientific problem. What this *is*: a goal-directed runtime engine that a
coordination engine can drive. Frozen Claude Opus 4.7 as the reasoning core,
plus durable memory, a skill library, a world model, multi-agent
delegation, tool synthesis, a typed task DAG executor, and an HTTP API.

## What's in here

```
agi/
  agent.py             # streaming agent loop — role-aware, skill-aware
  costs.py             # per-turn + cumulative token usage and $ tracking
  memory.py            # persistent JSONL memory store with keyword search
  world_model.py       # observed-entity tracker (files, urls, commands)
  planner.py           # turns a goal into a typed task DAG
  tools.py             # filesystem, shell, web search/fetch, memory tools
  tools_extension.py   # delegate, make_tool, plan_graph, invoke_skill, observe
  skills/library.py    # markdown SOPs, retrieve by query, render with args
  runtime/             # coordination-engine runtime
    capabilities.py    # machine-readable descriptor of what the runtime can do
    tasks.py           # task lifecycle, idempotent submission, state machine
    events.py          # pub/sub event bus with bounded history + SSE replay
    graph.py           # typed task DAG executor with parameter substitution
    worker.py          # agent-backed handlers for chat/plan/critique/skill/tool
    server.py          # stdlib HTTP server: POST /tasks, /graphs, GET /events
  coordination/        # reference coordination engine
    coordinator.py     # plan → dispatch → verify → revise loop
    __main__.py        # CLI demo
learner/               # learning track — small open base + LoRA loop
  ...
skills_library/        # version-controlled markdown skills (shipped)
evals/                 # eval suite + runner
tests/                 # pytest suite (40+ tests, no API required)
ARCHITECTURE.md        # full design
PLAN.md                # roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Use it

### As a CLI agent

```sh
python -m agi                            # interactive REPL
python -m agi "summarize ./README.md"    # one-shot
```

### As a coordination-engine runtime

The runtime exposes a stable HTTP protocol that any coordination engine
(yours, ours, or someone else's) can drive:

```sh
# Terminal A — start the runtime
python -m agi.runtime.server
# agi-runtime listening on http://127.0.0.1:7777

# Terminal B — submit a task
curl -s -X POST http://127.0.0.1:7777/tasks \
  -H 'Content-Type: application/json' \
  -d '{"kind":"chat","input":{"message":"summarize ./README.md"}}'

# Subscribe to the live event stream (SSE)
curl -N http://127.0.0.1:7777/events

# Discover what this runtime can do
curl -s http://127.0.0.1:7777/capabilities | jq .
```

Submit a typed task DAG and have the runtime execute it:

```sh
curl -s -X POST http://127.0.0.1:7777/graphs \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "research-and-draft",
    "nodes": [
      {"id":"research","kind":"chat","input":{"message":"find facts about X"}},
      {"id":"draft","kind":"chat","input":{"message":"draft using ${research.text}"},
       "depends_on":["research"]},
      {"id":"verify","kind":"critique","input":{"prompt":"X","response":"${draft.text}"},
       "depends_on":["draft"]}
    ]
  }'
```

### With the reference coordinator

```sh
python -m agi.coordination "research the Voyager 1 launch date, then write a tweet"
```

Drives an in-process runtime through plan → execute → verify → (revise) →
report, streaming live events to your terminal.

## What it can do

- **Goal → Plan → Execute → Verify** loop. Given a goal, the planner role
  produces a typed task DAG (`GraphSpec`). The executor role runs each node;
  the critic role gates the output. If the critic rejects, the coordinator
  revises and re-dispatches.
- **Typed task DAG** with parameter substitution (`${node.field}`) and
  per-node failure policies (`fail_graph` / `skip` / `retry:N`). Independent
  branches execute in parallel across the worker pool.
- **Multi-agent delegation.** A running agent can spawn focused subagents
  via the `delegate` tool. Token usage rolls up.
- **Skill library.** Markdown SOPs in `skills_library/` (and `~/.agi/skills/`)
  are retrieved by keyword match at task start and prepended to context.
- **Tool synthesis.** `make_tool(name, description, code)` lets the agent
  define new tools mid-session. AST guardrails block imports, `eval`,
  `exec`, `open`, and dunder access. **Not a security boundary — run in a
  sandbox in untrusted contexts.**
- **World model.** `~/.agi/world.jsonl` records every observed entity
  (file/url/command) with action + outcome, so future tasks can answer
  "have I done this before?"
- **Web search + fetch** (server-side `web_search_20260209` /
  `web_fetch_20260209`), file/shell, persistent memory.
- **Streaming + cost tracking + critic gating** on every chat turn.
- **HTTP API** exposing all of the above to a coordination engine, with
  idempotent submission (`dedup_key`), cancellation, per-task budgets,
  and a Server-Sent-Events stream.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models built from interaction, durable goals over months.
The reasoning core is frozen; the system improves the *runtime around* the
model — memory, skills, observed-entity model, trace-fed adapters — not the
model itself. The `learner/` track is the path to durable improvement via
LoRA on a small open base; that loop runs offline.

See `ARCHITECTURE.md` for the design and what's research vs. engineering.

## Running the tests

```sh
pip install pytest
python -m pytest tests/ --ignore=tests/test_critic.py \
                       --ignore=tests/test_learner.py \
                       --ignore=tests/test_critic_gate.py
```

40+ tests cover the runtime, task graph, event bus, HTTP API, skills,
world model, and tool synthesis — none require an API key.
