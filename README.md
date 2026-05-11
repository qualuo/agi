# agi

An agent harness, wrapped in a **runtime engine** that a coordination engine
can drive. Not AGI — that remains an unsolved scientific problem. This is what
you can credibly build today and ship into production:

- A capable agent on top of Claude Opus 4.7 with tools (file/shell/web/memory)
- A **runtime engine** that exposes the agent as a structured task service with
  lifecycle, budgets, structured events, cancellation, and **task delegation**
  (agents spawn child tasks; coordinators walk the resulting tree)
- A **skill library** — markdown SOPs auto-loaded into agent prompts based on
  the current task. Stage 3 of `ARCHITECTURE.md`, now real.
- An **HTTP API** (`python -m runtime`) so a coordinator in any language can
  submit tasks, stream events, cancel, query skills.
- A **learning track** that distills traces into a small open base + LoRA
  adapter, evaluated head-to-head against the frozen model.

## What's in here

```
agi/                # frozen-Opus agent (capability track)
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  costs.py          # per-turn + cumulative token usage and $ tracking
  memory.py         # persistent JSONL memory store with keyword search
  tools.py          # filesystem, shell, web search/fetch, memory tools
  __main__.py       # CLI: python -m agi
runtime/            # runtime engine — drives the agent as a callable service
  task.py           # Task / TaskStatus / Budget / TaskEvent
  engine.py         # concurrent orchestrator + delegate tool + skill injection
  backend.py        # LLM backend interface; AnthropicBackend + MockBackend
  server.py         # stdlib HTTP server (no FastAPI dep)
  client.py         # tiny Python client for the HTTP API
  __main__.py       # CLI: python -m runtime
learner/            # learning track — small open base + LoRA loop
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
  critic.py         # trace-quality critic (optional, requires torch)
  skills.py         # skill library — markdown SOPs with keyword retrieval
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner — reports pass/fail per task
examples/
  coordinator.py    # tiny coordination engine driving the runtime
tests/
  test_smoke.py     # smoke tests for agi/
  test_runtime.py   # engine lifecycle, events, cancellation, delegation, budgets
  test_skills.py    # skill library + engine integration
  test_server.py    # HTTP server end-to-end via the client
  test_learner.py   # smoke tests for learner/ (GPU-free pieces)
  test_critic*.py   # critic tests (skip if torch not installed)
ARCHITECTURE.md     # full design — read this for direction
PLAN.md             # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Use it

```sh
python -m agi                          # interactive REPL (single agent)
python -m agi "summarize ./README.md"  # one-shot
python -m runtime                      # HTTP runtime, real Anthropic backend
python -m runtime --mock               # HTTP runtime, deterministic mock (no API key)
python evals/run.py                    # run the eval suite
python examples/coordinator.py --mock  # tiny coordinator dispatching parallel subtasks
```

### Driving the runtime from a coordinator

```python
from runtime.client import RuntimeClient

rc = RuntimeClient("http://127.0.0.1:8765")
task = rc.submit("research X and summarize", budget={"max_cost_usd": 1.0, "max_turns": 10})
for event in rc.stream(task["id"]):
    print(event["kind"], event["data"])
final = rc.get(task["id"])
print(final["result"])
```

Or in-process:

```python
from runtime import Engine, Budget
from runtime.backend import AnthropicBackend
from learner.skills import SkillLibrary

engine = Engine(backend=AnthropicBackend(), skill_library=SkillLibrary())
task = engine.submit("research X", budget=Budget(max_cost_usd=1.0))
task.wait()
print(task.result)
```

### Decomposition via delegation

The agent has a `delegate(instruction, max_turns, max_cost_usd)` tool that
spawns a child task through the engine. Children get their own ids, their own
budgets, their own event streams, and a parent pointer — so a coordinator can
walk the tree with `GET /tasks/{id}/tree`. A task can recursively decompose
without the coordinator having to plan the decomposition itself.

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Load relevant skills from `~/.agi/skills/*.md` into the system prompt
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost
- Run multiple tasks concurrently in the engine
- Delegate subtasks to child agents
- Enforce per-task budgets (cost, tokens, turns, wall-clock)
- Expose all of the above over HTTP with structured events (SSE)

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design and
honest limitations.
