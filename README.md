# agi

An agent harness. Not AGI — that remains an unsolved scientific problem. This
is what you can credibly build today: a capable agent on top of Claude Opus 4.7
with tools to act on the world, persistent memory across sessions, and an eval
harness to measure capability.

## What's in here

```
agi/                # frozen-Opus harness (capability track)
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  costs.py          # per-turn + cumulative token usage and $ tracking
  memory.py         # persistent JSONL memory store with keyword search
  tools.py          # filesystem, shell, web search/fetch, memory tools
  runtime.py        # Runtime/Run/RunStatus — the executor a coordinator dispatches to
  server.py         # stdlib HTTP server: POST /v1/runs, GET /v1/runs/{id}/events (SSE)
  delegation.py     # `delegate` tool — agents spawn child runs on the same Runtime
  skills.py         # markdown skill library + retrieval tools
  tool_synthesis.py # `make_tool` — agent writes Python, gets a new tool (opt-in)
  __main__.py       # CLI: python -m agi
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
  test_runtime.py   # Runtime + event bus + cancellation + budget enforcement
  test_server.py    # HTTP API + SSE end-to-end
  test_delegation.py
  test_skills.py
  test_tool_synthesis.py
  test_learner.py   # smoke tests for learner/ (GPU-free pieces)
ARCHITECTURE.md     # full design — read this for direction
RUNTIME.md          # runtime engine contract — what a coordinator calls
PLAN.md             # stage roadmap (being rewritten against ARCHITECTURE.md)
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Use it

As a CLI:

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

As a Python runtime — a coordinator submits tasks and observes them:

```python
from agi import Runtime

rt = Runtime()
run = rt.submit("write a haiku about Postgres", cost_ceiling_usd=0.10)
run.wait(timeout=120)
print(run.status, run.result, f"${run.cost_usd:.4f}")
```

As an HTTP service (stdlib only, no framework deps):

```sh
python -m agi.server --host 0.0.0.0 --port 8000
# POST /v1/runs, GET /v1/runs/{id}, GET /v1/runs/{id}/events (SSE), POST .../cancel
```

See `RUNTIME.md` for the full executor contract — including subagent
delegation, cooperative cancellation, cost ceilings, and the event stream
schema a coordination engine subscribes to.

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost
- Run as a service: HTTP API with structured events, cancellation, and budget ceilings
- Delegate subtasks to child agent runs on the same Runtime (capped depth)
- Persist named skills (markdown procedures) and retrieve them by query
- (Opt-in) Synthesize new tools at runtime — agent writes Python, registers it

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design.
