# agi

An agent harness *and* a runtime engine. Not AGI — that remains an unsolved
scientific problem. What this is: a capable Claude-Opus-4.7 agent with tools,
memory, and evals, plus a small **runtime** an external coordination engine
can drive concurrently over HTTP/SSE with budgets, plans, and structured
event streams.

## What's in here

```
agi/                # agent + runtime engine
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  budget.py         # cost / token / iteration / wall-time caps
  coordinator.py    # reference coordinator over the runtime
  costs.py          # per-turn + cumulative token usage and $ tracking
  events.py         # structured event types emitted during a session
  memory.py         # persistent JSONL memory store with keyword search
  plan.py           # Plan / Subgoal DAG with {{ name }} substitution
  runtime.py        # concurrent Session manager driving Agents on a thread pool
  server.py         # HTTP + SSE surface for external coordinators
  tools.py          # filesystem, shell, web search/fetch, memory tools
  __main__.py       # CLI: python -m agi [serve|plan|<prompt>]
learner/            # learning track — small open base + LoRA loop
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
  README.md         # how to run training
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner — reports pass/fail per task
tests/
  test_smoke.py     # smoke tests for the agi/ package
  test_runtime.py   # runtime + budget + plan + HTTP server tests (no API)
  test_learner.py   # smoke tests for learner/ (GPU-free pieces)
ARCHITECTURE.md     # full design — read this for direction
PLAN.md             # stage roadmap (being rewritten against ARCHITECTURE.md)
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Use it

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost

## Runtime engine

The same Agent can run as a single CLI process **or** as a runtime engine
driven by an external coordinator:

```python
from agi import Runtime, Budget, SimpleCoordinator, Plan, Subgoal

rt = Runtime(max_concurrent=4)
coord = SimpleCoordinator(rt)

# Single goal with a hard $ cap.
summary = coord.run("draft a release note for v0.2", budget=Budget(max_cost_usd=0.25))
print(summary.status, summary.final_text, f"${summary.cost_usd:.4f}")

# Plan: a DAG of subgoals; independents fan out, dependents pull upstream text in.
plan = Plan(name="research", subgoals=[
    Subgoal("market",  "summarize the agent-runtime market"),
    Subgoal("rivals",  "list 5 competing agent runtimes with one-line takes"),
    Subgoal("brief",   "write a 1-page brief combining: {{ market }} | {{ rivals }}",
            depends_on=["market", "rivals"]),
])
results = coord.run_plan(plan)
```

Run it as a service for an external coordinator (LangGraph, an orchestrator
service, your own workflow engine):

```sh
python -m agi serve --host 0.0.0.0 --port 8765 --concurrent 8
# AGI_API_TOKEN=... to require a Bearer token
```

HTTP surface:

```
POST /v1/sessions              { "goal": "...", "budget": {...} } -> { "session_id" }
GET  /v1/sessions              -> [ {record}, ... ]
GET  /v1/sessions/{id}         -> {record}                # status, cost, tokens, final_text
POST /v1/sessions/{id}/cancel  -> { "ok": true }
GET  /v1/sessions/{id}/events  -> Server-Sent Events stream  # text deltas, tool calls, ...
POST /v1/plans                 { "name", "subgoals": [...] } -> { "results": {...} }
GET  /v1/health                -> { "ok": true, "concurrent": N }
```

Why this shape: a coordination engine doesn't need a chat UI, it needs
**concurrent sessions, hard budgets, observable event streams, and plan
fan-out**. That's exactly what the runtime exposes, and nothing more.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design.
