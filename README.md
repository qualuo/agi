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
  tools.py          # filesystem, shell, web search/fetch, memory, delegate
  runtime.py        # control plane: sessions, jobs, budgets, cancel, events
  coordinator.py    # in-process DAG executor over the runtime
  server.py         # stdlib HTTP control plane for non-Python coordinators
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
  test_smoke.py        # smoke tests for the agi/ package
  test_runtime.py      # runtime: sessions, jobs, budgets, cancel, events
  test_coordinator.py  # DAG executor over the runtime
  test_critic_gate.py  # critic-as-output-gate integration
  test_learner.py      # smoke tests for learner/ (GPU-free pieces)
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
- Run as a **runtime** — sessions, parallel jobs, per-job $ budgets, cooperative
  cancel, live event streams, snapshot/restore, metrics
- **Coordinate** itself — the agent has a `delegate` tool; an in-process
  `Coordinator` runs DAGs of jobs in parallel; an HTTP control plane lets a
  coordinator written in any language drive the runtime

## Use it as a runtime

The `Runtime` is the stable surface that an external coordination engine — or
the in-process `Coordinator`, or your own scheduler — calls into. It owns
sessions and jobs, enforces budgets between turns, cancels on demand, and
streams structured events.

```python
from agi import Runtime

rt = Runtime(max_workers=8)
sess = rt.create_session(role="researcher")
job  = rt.submit(sess.id, "summarize https://example.com",
                 budget_usd=0.10, max_iterations=15)

for ev in rt.stream(job.id):
    print(ev.kind, ev.payload)        # text_delta, tool_use, status, ...

rec = rt.await_job(job.id, timeout=60)
print(rec.status, rec.cost_usd, rec.output)
print(rt.metrics())
```

Compose multi-step plans with the `Coordinator` (DAG of jobs, parallel
where deps allow, with `{upstream_name}` rendered into downstream prompts):

```python
from agi import Runtime, Coordinator, Node

rt = Runtime()
plan = Coordinator(rt, [
    Node("research",  "Find three recent papers on RAG.",                    role="researcher"),
    Node("summarize", "Two sentences each:\n{research}", depends_on=["research"], role="writer"),
    Node("critique",  "Weaknesses?\n{summarize}",       depends_on=["summarize"], role="critic"),
])
print(plan.run(timeout=120)["critique"].output)
```

Drive it from a non-Python coordinator over HTTP (stdlib only, no extra deps):

```python
from agi import Runtime
from agi.server import serve
serve(Runtime(), host="127.0.0.1", port=8765)  # background thread
```

```
GET  /healthz                          GET  /metrics
POST /v1/sessions                      GET  /v1/sessions  GET /v1/sessions/{id}
POST /v1/sessions/{id}/jobs            GET  /v1/jobs      GET /v1/jobs/{id}
POST /v1/jobs/{id}/cancel              GET  /v1/jobs/{id}/events   (SSE)
```

The agent itself is also a coordinator: under a Runtime it gets a `delegate`
tool for spawning child sessions/jobs. Costs roll up; the call graph is
recoverable from `parent_session_id` metadata.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design.
