# agi

An agent harness. Not AGI — that remains an unsolved scientific problem. This
is what you can credibly build today: a capable agent on top of Claude Opus 4.7
with tools to act on the world, persistent memory across sessions, an eval
harness to measure capability, and a **runtime layer** that lets external
coordination engines (in any language) drive it over HTTP/SSE.

## What's in here

```
agi/                # frozen-Opus harness (capability track)
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  costs.py          # per-turn + cumulative token usage and $ tracking
  memory.py         # persistent JSONL memory store with keyword search
  tools.py          # filesystem, shell, web search/fetch, memory tools
  events.py         # typed events emitted by the runtime
  runtime.py        # AgentRuntime: sessions, structured events, snapshot/resume
  coordinator.py    # multi-agent Plan→Execute→Verify coordination engine
  server.py         # stdlib HTTP+SSE server — drive sessions from any language
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
  test_runtime.py   # AgentRuntime tests (no API hits)
  test_coordinator.py  # Coordinator tests (fake-runtime driven)
  test_server.py    # HTTP+SSE server tests (loopback)
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

## Runtime API (for coordination engines)

The `Agent` class is one-conversation, one-process, blocking. For anything
that needs to *drive* the agent — a coordinator picking specialists, a UI
that wants tool-approval gates, a multi-tenant service — use `AgentRuntime`:

```python
from agi.runtime import AgentRuntime, SessionConfig

rt = AgentRuntime()
sid = rt.start_session(config=SessionConfig(max_cost_usd=0.50, max_turns=20))

for event in rt.send(sid, "summarize ./README.md"):
    # event is a typed dataclass from agi.events:
    # TextDelta, ThinkingDelta, ToolUseRequested, ToolResult,
    # TurnCompleted, BudgetExceeded, RuntimeError_
    handle(event)

snap = rt.snapshot(sid)   # serialize to JSON-safe dict
rt.close_session(sid)
sid2 = rt.restore(snap)   # resume later, possibly in a new process
```

**Tool interception** lets a coordinator approve, deny, or replace tool
calls before they execute — useful for human-in-the-loop approvals,
sandboxing, or testing:

```python
def approve(session_id, name, inp):
    if name == "run_bash" and "rm " in inp.get("command", ""):
        return "denied by policy"   # replaces the tool result
    return None                      # allow normal execution

sid = rt.start_session(interceptor=approve)
```

## Coordination engine

`agi.coordinator.Coordinator` is the canonical outer loop:
Plan → Execute (parallel sub-tasks) → Verify, with retries and per-goal
cost ceilings.

```python
from agi.coordinator import Coordinator, Goal
from agi.runtime import AgentRuntime

coord = Coordinator(AgentRuntime())
outcome = coord.execute(Goal(
    description="Find the three most-starred Python web frameworks on GitHub and write a 1-paragraph comparison to ./compare.md",
    success_check=lambda txt: "compare.md" in txt,
    max_cost_usd=0.50,
    max_retries=1,
))
print(outcome.success, outcome.total_cost_usd)
```

Specialists (`planner`, `executor`, `verifier`) are pluggable; the verifier
uses Haiku by default for cheap judgment.

## HTTP+SSE server

`python -m agi.server` starts a stdlib HTTP server that exposes the runtime
to any language. A coordination engine in Go, Rust, or Node can drive
sessions over the wire:

```
POST /v1/sessions               → {"session_id": "..."}
POST /v1/sessions/{id}/send     → text/event-stream of typed events
GET  /v1/sessions/{id}/snapshot → JSON snapshot for durable workflows
POST /v1/sessions/{id}/snapshot → restore from snapshot
DELETE /v1/sessions/{id}
GET  /v1/health
```

Set `AGI_RUNTIME_TOKEN` to require `Authorization: Bearer <token>` on every
request. Run with `--host 0.0.0.0 --port 8765` to bind publicly.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design.
