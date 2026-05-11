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
  __main__.py       # CLI: python -m agi
runtime/            # runtime engine — HTTP/JSON+SSE surface for coordinators
  runtime.py        # in-process Runtime: sessions, jobs, budgets, metrics
  sessions.py       # SessionManager — per-session Agent + Memory + budget
  jobs.py           # async job queue (threadpool) + event stream
  budgets.py        # token/USD/turn caps, enforced before each turn
  capabilities.py   # machine-readable capability manifest
  metrics.py        # counters, gauges, latency histograms
  server.py         # stdlib HTTP server (JSON + SSE)
  client.py         # stdlib Python client for the HTTP API
  mock_agent.py     # offline drop-in for tests/demos (no API key needed)
  __main__.py       # CLI: python -m runtime
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
  test_runtime.py   # full runtime engine tests (no API key required)
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

## Runtime engine (for coordination engines)

The harness also runs as a managed service that a coordination engine can drive
over HTTP. Sessions are isolated, jobs run async on a threadpool, budgets are
enforced before each turn, and the capability manifest is machine-readable so a
coordinator can route tasks across heterogeneous runtimes.

```sh
python -m runtime --backend opus            # production: needs ANTHROPIC_API_KEY
python -m runtime --backend mock            # offline: no key, deterministic
```

```python
from runtime.client import Client

c = Client("http://localhost:8765")
print(c.capabilities()["tools"])             # discover what this runtime can do
sid = c.create_session(budget={"max_usd": 1.0})["id"]
print(c.chat(sid, "2 + 2")["text"])          # synchronous

job = c.submit_job(sid, "summarize ./README.md")
for ev in c.stream(job["id"]):               # SSE: text_delta, tool_use, done
    print(ev)
print(c.wait_job(job["id"])["result_text"])
```

Endpoints (`GET /v1/capabilities` for the live list):

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | liveness + uptime + counts |
| GET | `/v1/capabilities` | tools, models, pricing, features, protocol version |
| GET | `/v1/metrics` | counters, gauges, p50/p95/p99 latencies |
| POST | `/v1/sessions` | create a session with optional budget |
| GET | `/v1/sessions` | list active sessions |
| GET / DELETE | `/v1/sessions/{sid}` | inspect / terminate |
| POST | `/v1/sessions/{sid}/messages` | sync chat (blocks, returns text + usage) |
| POST | `/v1/sessions/{sid}/jobs` | async chat (returns job id) |
| GET | `/v1/jobs/{jid}` | poll status + result |
| GET | `/v1/jobs/{jid}/stream` | SSE event stream |
| POST | `/v1/jobs/{jid}/cancel` | cooperative cancel |
| GET / POST | `/v1/sessions/{sid}/memory` | search or save notes |

Auth is a bearer token via `AGI_RUNTIME_TOKEN` (or `--token`); unset means open
(local dev only).

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design.
