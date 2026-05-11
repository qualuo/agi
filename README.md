# agi

An agent harness. Not AGI — that remains an unsolved scientific problem. This
is what you can credibly build today: a capable agent on top of Claude Opus 4.7
with tools to act on the world, persistent memory across sessions, an eval
harness to measure capability, and a **runtime API** designed to be driven by
a parent coordination engine.

## What's in here

```
agi/                # frozen-Opus harness (capability track)
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  runtime.py        # Runtime / Task / Result / Event — coordination API
  costs.py          # per-turn + cumulative token usage and $ tracking
  memory.py         # persistent JSONL memory store with keyword search
  tools.py          # filesystem, shell, web, memory, skills, delegate, tool-synth
  __main__.py       # CLI: python -m agi
learner/            # learning track — small open base + LoRA loop
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  skills.py         # markdown skill library, atomic writes, keyword retrieval
  critic.py         # trace-quality critic (CPU-tractable specialist)
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner — reports pass/fail per task
tests/              # 72 unit tests — no API key required
ARCHITECTURE.md     # design — read this for direction
PLAN.md             # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Use it interactively

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

## Use it as a runtime engine (for a coordination engine)

The interesting use case isn't a REPL — it's having a parent system (a
planner, a workflow engine, a service mesh) drive this agent as one of its
runtime resources. `agi.Runtime` exposes that surface:

```python
from agi import Runtime, Task

rt = Runtime()                                # process-local, durable memory on disk

# What can this runtime do? (Coordinators introspect before routing work.)
print(rt.describe())
# → {"model": "claude-opus-4-7", "tools": [...], "server_tools": ["web_search_..."],
#    "has_skill_library": True, "active_sessions": []}

# One-shot:
result = rt.execute(Task(
    goal="extract email addresses from inbox.txt",
    deadline_seconds=30,
    max_tokens_budget=8000,
    allowed_tools=["read_file"],     # least-privilege; whitelist for this task only
    reflect=True,                    # write a lesson-tagged memory note after
))
print(result.to_dict())              # JSON-safe envelope

# Conversational (coordinator addresses by id, resumes across calls):
session = rt.spawn(delegate_depth=2, enable_tool_synthesis=True)
r1 = session.run(Task(goal="open the design doc"))
r2 = session.run(Task(goal="critique section 3"))
rt.close(session.session_id)

# Streaming (coordinator subscribes to events for live progress):
for event in rt.stream(Task(goal="research and summarize X")):
    print(event.kind, event.payload)
```

### What the runtime guarantees to a coordinator

- **Structured envelopes.** `Task` in, `Result` out. Both serialize cleanly to
  JSON. No prose-only IO.
- **Budgets and deadlines.** `max_tokens_budget`, `deadline_seconds`, and
  `max_iterations` are checked between turns. A coordinator can cap cost and
  time per task with no extra wrapping logic.
- **Tool whitelists.** `allowed_tools` restricts what the model can call for
  one task; the original surface is restored when the task ends. This is the
  least-privilege primitive a coordinator needs to safely fan out work.
- **Bounded delegation.** A task with `delegate_depth=N` may spawn subagents
  up to that depth and no further. Each spawned subagent is an ephemeral
  session backed by the same runtime, so memory and skills propagate.
- **Capability introspection.** `Runtime.describe()` returns the exact tool
  list, including client tools, server tools, and any dynamically registered
  ones. Coordinators don't have to hardcode capabilities.
- **Streaming.** `Runtime.stream(task)` yields `Event(kind, task_id, payload)`
  objects for `started`, `iteration`, `tool_call`, `finished`, and `result`.
  Useful for live UIs and progress bars in the parent system.
- **Cancellation.** `Session.cancel()` is observed between turns.
- **Cost accounting.** Each `Result` carries token usage and a USD estimate
  derived from the current pricing table.

This is the shape Claude Opus 4.7 plus a small amount of plumbing can be —
treated as one runtime among many that a coordination engine routes work to.

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost
- Recall and save **skills** — named markdown SOPs from `~/.agi/skills/` that
  the agent loads when relevant
- Synthesize new Python tools at runtime in a restricted sandbox (`make_tool`)
- **Delegate** bounded sub-tasks to subagents with depth-limited recursion
- Reflect after each task — write a `lesson`-tagged memory note for future recall

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn at the weight level. The `learner/` track is the path toward
durable improvement through weight updates on a small open base — not a
frontier model, but actually a system that learns. See `ARCHITECTURE.md` for
the design.
