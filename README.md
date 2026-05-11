# agi

An agent runtime engine. Not AGI — that remains an unsolved scientific problem.
What this is: a capable agent on top of Claude Opus 4.7, packaged as a typed,
observable **runtime** that a higher-level coordination engine can drive.

The Agent class is the local loop. The **Runtime** is the multi-tenant,
session-aware, event-emitting, cost-accounting surface that a planner or
workflow orchestrator binds against. An HTTP/SSE server exposes the same
surface to other processes and languages.

## What's in here

```
agi/
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  events.py        # typed Event + EventBus; pub/sub between agent and consumers
  runtime.py       # multi-session Runtime: capabilities, idempotency,
                   # cancellation, usage roll-up, event subscription
  server.py        # stdlib HTTP/JSON + SSE surface over Runtime
  coordinator.py   # example coordination engine that plans + dispatches
  skills.py        # markdown SOPs; retrievable procedural memory
  costs.py         # per-turn + cumulative token usage and $ tracking
  memory.py        # persistent JSONL memory store with keyword search
  tools.py         # filesystem, shell, web search/fetch, memory tools
  __main__.py      # CLI: python -m agi
learner/           # learning track — small open base + LoRA loop
  traces.py        # append-only JSONL trace logger
  filter.py        # quality gates: eval-pass, score threshold, thumbs
  train.py         # LoRA SFT script (HF transformers + PEFT, GPU)
  critic.py        # tiny critic that learns trace quality (CPU-tractable)
evals/
  tasks.jsonl      # eval tasks (math, file ops, recall, search)
  run.py           # eval runner — reports pass/fail per task
tests/             # 74 tests; no network needed for the runtime surface
ARCHITECTURE.md    # full design — read this for direction
PLAN.md            # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Three ways to use it

### 1. Direct CLI

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

### 2. In-process runtime

```python
from agi import Runtime, SkillLibrary

rt = Runtime(skills=SkillLibrary())
sid = rt.create_session()
result = rt.run(sid, "What's in ./README.md? Be brief.")
print(result.output_text)
print(f"cost: ${result.cost_usd:.4f}, tokens: {result.usage}")
```

A coordination engine inside the same process drives the runtime through
this typed API. Sessions are addressable, events are observable, runs
are idempotent if you pass an `idempotency_key`.

### 3. HTTP/SSE server (cross-process or cross-language)

```sh
AGI_RUNTIME_PORT=7777 python -m agi.server
```

Then from any HTTP client:

```sh
# Discover what the runtime can do
curl http://127.0.0.1:7777/v1/capabilities

# Create a session, run a turn
sid=$(curl -s -X POST http://127.0.0.1:7777/v1/sessions -d '{}' | jq -r .session_id)
curl -X POST http://127.0.0.1:7777/v1/sessions/$sid/run \
  -H 'content-type: application/json' \
  -d '{"prompt": "hello", "idempotency_key": "k1"}'

# Subscribe to events as they happen
curl -N http://127.0.0.1:7777/v1/sessions/$sid/events
```

Set `AGI_RUNTIME_TOKEN=secret` to require `Authorization: Bearer secret`.

## Runtime API (v1)

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | Liveness + uptime + session count |
| GET | `/v1/capabilities` | Tools, skills, supported events, features |
| GET | `/v1/usage` | Aggregated token/cost telemetry across sessions |
| POST | `/v1/sessions` | Create a session |
| GET | `/v1/sessions` | List sessions |
| GET | `/v1/sessions/{id}` | Session info: runs, cumulative usage, cost |
| DELETE | `/v1/sessions/{id}` | Destroy a session |
| POST | `/v1/sessions/{id}/run` | Run a turn (sync) |
| POST | `/v1/sessions/{id}/cancel` | Request cancellation |
| GET | `/v1/sessions/{id}/events` | SSE event stream |
| GET | `/v1/sessions/{id}/events/recent` | Recent events (replay) |
| GET | `/v1/skills` | List installed skills |
| POST | `/v1/skills` | Upsert a skill (markdown SOP) |

Events emitted: `run.started`, `run.finished`, `turn.started`, `turn.finished`,
`thinking.started`, `thinking.delta`, `text.started`, `text.delta`,
`tool.requested`, `tool.result`, `server_tool.requested`, `critic.scored`,
`skills.injected`, `cancelled`, `error`.

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Load procedural skills (`~/.agi/skills/*.md`) when the prompt matches
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output, thinking summaries, and tool calls as typed events
- Track per-turn and cumulative token usage with $ cost
- Cancel a running session, retry idempotently, replay recent events

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals over weeks. The Opus harness is a frozen
system; it doesn't learn. The `learner/` track is the path toward durable
improvement through weight updates on a small open base — see `ARCHITECTURE.md`
for the dual-track design.
