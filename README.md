# agi

An AGI **runtime engine** — not AGI, but the substrate a coordination engine
drives to get there. Capable agent on top of Claude Opus 4.7 with tools to act
on the world, persistent memory + skill library across sessions, runtime tool
synthesis, sub-agent delegation, a typed event stream, an HTTP/SSE control
plane, and an eval harness to measure capability.

## Why "runtime engine"

A coordination engine — whatever decides what to do when, plans across runs,
or schedules a population of agents — needs a runtime it can drive. This repo
is that runtime:

- **Submit work** programmatically (`Runtime.submit`) or over HTTP (`POST /v1/runs`).
- **Stream events** as they happen (thinking, tool-call, tool-result, text,
  subrun, reflection, critic-score, usage, done, error, cancelled).
- **Control** runs: cancel cooperatively at the next safe point.
- **Compose** runs into trees: `delegate(task, role)` from inside an agent
  spawns a child run with rolled-up cost accounting.
- **Inspect** state: live runs, skill library, recent memory, cost metrics.
- **Persist** durable state: long-term memory, skill library, trace logs.

The same primitives drive `python -m agi`, the REPL, the HTTP server, and any
external orchestrator you wire on top.

## What's in here

```
agi/
  agent.py        # streaming agent loop — adaptive thinking + tool dispatch
                  #   + critic gate + event emission + delegate/make_tool
  runtime.py      # Runtime, RunRequest, RunHandle, RunResult — the engine API
  events.py       # typed Event records (the wire format)
  server.py       # stdlib HTTP + SSE adapter (no FastAPI needed)
  skills.py       # markdown skill library w/ retrieval and uses tracking
  reflection.py   # post-task lesson writer (writes to memory under `lesson`)
  synthesis.py    # runtime tool synthesis — sandboxed, smoke-tested
  memory.py       # persistent JSONL memory store with keyword search
  tools.py        # filesystem, shell, web search/fetch, memory tools
  costs.py        # per-turn + cumulative token usage and $ tracking
  __main__.py     # CLI: REPL, one-shot, `serve`
learner/          # learning track — small open base + LoRA loop
  traces.py       # append-only JSONL trace logger
  filter.py       # quality gates: eval-pass, score threshold, thumbs
  critic.py       # first specialist: trace-quality MLP
  train.py        # LoRA SFT script (HF transformers + PEFT, GPU)
evals/
  tasks.jsonl     # eval tasks (math, file ops, recall, search)
  run.py          # eval runner — reports pass/fail per task
tests/            # 51 unit tests (no API key needed)
ARCHITECTURE.md   # full design — read this for direction
PLAN.md           # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Use it

### As a library

```python
from agi import Runtime, RunRequest

rt = Runtime()                            # one runtime per process
run = rt.submit(RunRequest("research the weather in Tokyo"))
for evt in run.events():                  # blocking iterator of typed Events
    print(evt.type, evt.data)
run.wait()
print(run.result.text, run.result.cost_usd)
```

### As a CLI

```sh
python -m agi                             # interactive REPL on the runtime
python -m agi "summarize ./README.md"     # one-shot
python -m agi serve 8088                  # HTTP/SSE control plane
python evals/run.py                       # run the eval suite
```

### As a service (HTTP/SSE)

```sh
python -m agi serve 8088
```

Then from any coordination engine:

```sh
# Start a run
curl -s -XPOST localhost:8088/v1/runs \
     -H 'content-type: application/json' \
     -d '{"task":"hello"}'
# → {"run_id":"...","stream":"/v1/runs/.../events"}

# Stream events (SSE)
curl -N localhost:8088/v1/runs/<id>/events

# Cancel, list runs, inspect skills/memory, metrics
curl -XPOST localhost:8088/v1/runs/<id>/cancel
curl       localhost:8088/v1/runs
curl       localhost:8088/v1/skills
curl       'localhost:8088/v1/memory?q=teal'
curl       localhost:8088/v1/metrics
```

The wire format is one JSON-encoded `Event` per SSE message. Compatible with
`EventSource` in any browser, or any SSE client.

## What it can do

- Read/write files, run shell commands.
- Search the live web (server-side `web_search_20260209`) and fetch URLs.
- **Remember** across sessions (`~/.agi/memory.jsonl`) — durable lessons get
  tagged `lesson` and surface in future related tasks via search.
- **Skill library** at `~/.agi/skills/` — markdown SOPs retrieved by keyword,
  injected into the system prompt for relevant tasks; usage counts track which
  skills actually pay off.
- **Sub-agents**: `delegate(task, role)` spawns a child run with its own
  context window; cost rolls up to the parent and a coordination engine sees
  the tree shape via `subrun_started` / `subrun_completed` events.
- **Tool synthesis**: `make_tool(name, description, code, input_schema, ...)`
  authors a Python tool at runtime, statically checks it against an allow-list,
  smoke-tests it, and registers it for the rest of the session.
- **Critic gate**: optional learned critic scores responses and annotates
  low-confidence outputs before they reach the user.
- **Trace logging**: every run is logged to `~/.agi/traces.jsonl`; the LoRA
  loop trains on filtered traces.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals over weeks of unsupervised operation. The
Opus harness is a frozen system; it doesn't learn from interaction at the
weights level — the `learner/` track is the path toward durable improvement
through weight updates on a small open base. See `ARCHITECTURE.md`.

## How a coordination engine drives it

The runtime is intentionally event-shaped and stateless-on-top. A coordinator
runs a control loop like:

1. Pull a unit of work from its queue.
2. `POST /v1/runs` to start a run on this runtime.
3. Subscribe to the event stream and react: surface progress to the user,
   pipe tool-calls into observability, watch for `critic_score` to decide
   if regeneration is needed, cancel on timeout.
4. On `done`, persist the result, decide what's next.

Sub-runs spawned via `delegate` ride the same machinery — they appear as
their own `run_id` with `parent_id` set, and the parent sees them via
`subrun_started` / `subrun_completed` events.

## Tests

```sh
python -m unittest discover -s tests -v
```

51 unit tests, none need an API key.
