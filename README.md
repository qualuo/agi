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
  protocol.py       # Job/JobResult/RuntimeCapabilities — coordinator wire format
  runtime.py        # Runtime — wraps Agent in submit/snapshot/resume contract
  __main__.py       # CLI: python -m agi
coord/              # coordination engine over Runtimes
  coordinator.py    # routing, race, global budget enforcement
learner/            # learning track — small open base + LoRA loop
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
  README.md         # how to run training
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner — reports pass/fail per task
examples/
  coordinator_demo.py  # runnable: 2 runtimes, 1 coordinator, 5 patterns
tests/
  test_smoke.py     # smoke tests for the agi/ package
  test_runtime.py   # tests for protocol, runtime, coordinator (no API)
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
python examples/coordinator_demo.py   # runtime + coordinator demo
```

## As a runtime engine

The Agent is the executor. The **Runtime** is the contract a coordination
engine depends on — it wraps the Agent in a uniform `submit(job) → result`
shape with budgets, snapshots, and capability introspection. The
**Coordinator** routes jobs across runtimes (cheapest-first, by tag,
race-and-take-best) and enforces a global spend ceiling.

```python
from agi import Job, opus_runtime, haiku_runtime
from coord import Coordinator, RoutingPolicy
from coord.coordinator import CoordinatorBudget

c = Coordinator(budget=CoordinatorBudget(max_total_usd=1.00))
c.register(haiku_runtime(tags=["fast", "cheap"]))
c.register(opus_runtime(tags=["frontier"]))

# cheapest runtime that can do the work
result = c.run(Job(prompt="What is 17 + 25?", max_cost_usd=0.05),
               policy=RoutingPolicy.CHEAPEST)

# explicit capability requirement
result = c.run(Job(prompt="Hard reasoning task", max_cost_usd=0.50),
               required_tags=["frontier"])

# fan out and take the first acceptable answer
winner = c.race(Job(prompt="Summarize: ...", max_cost_usd=0.10),
                runtime_ids=[r.runtime_id for r in c.runtimes.values()],
                accept=lambda r: r.succeeded and len(r.output) > 100)
```

Every `JobResult` carries `cost_usd`, `iterations`, `session_id`, and a
`trace_id` linking back to the durable trace log — so a coordinator can
audit, retry, or branch any prior run. See `examples/coordinator_demo.py`.

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
