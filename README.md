# agi

An agent harness and **runtime engine**. Not AGI — that remains an unsolved
scientific problem. This is what you can credibly build today: a capable
agent on Claude Opus 4.7 with tools to act on the world, persistent memory,
a skill library, subagent delegation, structured events, and a JSON-lines
protocol so a coordination engine can drive it as a substrate.

## What's in here

```
agi/
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  runtime.py        # Runtime / Run / RunSpec / Budget — the coordinator-facing API
  events.py         # structured event types emitted by a Run
  serve.py          # JSON-lines protocol server (stdin/stdout)
  skills.py         # markdown skill library, keyword-searchable
  memory.py         # persistent JSONL memory store
  tools.py          # file/shell/web/memory + delegate, make_tool, skill I/O
  costs.py          # per-turn + cumulative token usage and $ tracking
  __main__.py       # CLI: python -m agi
learner/            # learning track — small open base + LoRA loop (optional)
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner — reports pass/fail per task
tests/              # 60+ tests, no API calls required
ARCHITECTURE.md     # full design — read this for direction
PLAN.md             # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Two ways to use it

### As an interactive agent

```sh
python -m agi                         # REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

### As a runtime engine for a coordinator

```python
from agi import Runtime, RunSpec, Budget

rt = Runtime()
run = rt.submit(RunSpec(
    prompt="Build me a fizzbuzz script and run it",
    budget=Budget(max_usd=0.50, max_turns=10),
    skills=["run-fizzbuzz"],          # preload relevant skills if any
))

# Stream structured events as they happen
for event in run.iter_events():
    print(event.type, event.data)

print(run.result.text, f"${run.result.cost_usd:.4f}")
```

Or talk to it over a line-oriented protocol any orchestrator can speak:

```sh
python -m agi.serve
> {"cmd": "submit", "prompt": "...", "budget": {"max_usd": 0.5}, "subscribe": true}
< {"type": "submitted", "run_id": "abc123def456", ...}
< {"type": "event", "run_id": "abc...", "event": {"type": "run_started", ...}}
< {"type": "event", "run_id": "abc...", "event": {"type": "text_delta", ...}}
< {"type": "event", "run_id": "abc...", "event": {"type": "run_completed", ...}}
```

Other commands: `events`, `status`, `list`, `cancel`, `shutdown`. See
`agi/serve.py` for the full protocol.

## What the runtime gives a coordinator

- **Run handles** with `id`, `status`, `result`, `usage`, `cost_usd`.
- **Structured events**: `run_started`, `task_started`, `thinking_delta`,
  `text_delta`, `tool_call`, `tool_result`, `turn_completed`,
  `task_completed`, `run_completed`, `run_failed`, `run_cancelled`,
  `budget_exceeded`, `child_run_started`, `child_run_completed`.
- **Hard budgets**: USD, input/output tokens, turns, wall-clock seconds.
  Trips end the run as `cancelled` with a `budget_exceeded` event first.
- **Cooperative cancellation** via `Run.cancel(reason)` from outside.
- **Durable run registry** in `~/.agi/runs/<id>.json` — re-load by id.
- **Subagent delegation**. The agent has a `delegate(prompt, role, max_usd,
  max_turns)` tool. Roles: `planner`, `executor`, `critic`, `researcher`,
  `general`. Child token + dollar cost rolls up to the parent run.
- **Self-extension**:
  - `save_skill(name, description, body, tags)` — write a reusable procedure
    to `~/.agi/skills/`. Auto-loaded by relevance on future tasks.
  - `make_tool(name, description, code)` — compile a Python function and
    register it as a new tool for the rest of this session.

## What it can do

- Read/write files, run shell commands
- Search the web (server-side `web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost
- Spawn subagents with rolled-up cost accounting
- Persist and retrieve named skills; extend its tool set at runtime

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design.
