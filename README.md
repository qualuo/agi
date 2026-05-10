# agi

An **agent runtime** designed to be embedded by a coordination engine. Not
AGI — that remains an unsolved scientific problem, and this repo is honest
about that. What this *is*: a capable agent on top of Claude Opus 4.7, with
tools to act on the world, persistent memory across sessions, an embeddable
runtime contract, an HTTP/SSE wire protocol, and an eval harness to measure
capability.

## What's in here

```
agi/                            # runtime
  agent.py                      # streaming agent loop — adaptive thinking + tool dispatch
  runtime.py                    # Session, Runtime, Budget — embeddable contract
  server.py                     # HTTP/JSON + SSE wire shape for a coordinator
  events.py                     # lifecycle event bus
  coordination.py               # delegate (subagents) + reflect (lessons)
  costs.py                      # per-turn + cumulative token usage and $ tracking
  memory.py                     # persistent JSONL memory store with keyword search
  tools.py                      # filesystem, shell, memory, web search/fetch
  __main__.py                   # CLI: python -m agi  /  python -m agi serve
learner/                        # learning track — small open base + LoRA loop
  traces.py                     # append-only JSONL trace logger
  filter.py                     # quality gates: eval-pass, score threshold, thumbs
  critic.py                     # trace-quality critic (CPU-trainable specialist)
  skills.py                     # markdown skill library + tools
  train.py                      # LoRA SFT script (HF transformers + PEFT, GPU)
  README.md                     # how to run training
evals/
  tasks.jsonl                   # eval tasks (math, file ops, recall, search)
  run.py                        # eval runner — reports pass/fail per task
tests/                          # 98 unit tests; no API calls
ARCHITECTURE.md                 # full design — read this for direction
PLAN.md                         # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Use it

### As a CLI

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

### As an embeddable runtime (in-process)

```python
from agi.runtime import Runtime, Budget

rt = Runtime()
session = rt.create_session(role="executor", budget=Budget(max_usd=0.50))
session.bus.subscribe(lambda ev: print(ev.type, ev.data))   # stream events
result = session.step("plan a trip to Tokyo")
print(result.text, result.cost_usd, result.budget_exceeded)
snap = session.snapshot()                                    # park
restored = rt.restore_session(snap)                          # resume later
```

### As a network runtime (for a remote coordination engine)

```sh
python -m agi serve --host 127.0.0.1 --port 8765
```

```sh
curl -s :8765/capabilities                                      # what's exposed
curl -s :8765/sessions -d '{"role":"executor"}' -H content-type:application/json
curl -s :8765/sessions/$ID/step -d '{"input":"plan a trip"}'    # one turn
curl -N :8765/sessions/$ID/events?since=0                       # SSE event stream
curl -s :8765/sessions/$ID/snapshot -X POST                     # park
```

Endpoints, in order of importance to a coordinator:
- `GET  /capabilities` — tools, models, skills, snapshots, streaming
- `POST /sessions` — create (with `budget`, `role`, `agent` overrides)
- `POST /sessions/{id}/step` — drive one turn
- `GET  /sessions/{id}/events` — Server-Sent Events; `?since=<seq>` replays
- `POST /sessions/{id}/snapshot` — opaque resumable state
- `POST /sessions/restore` — reattach a parked session
- `DELETE /sessions/{id}` — close

## What it can do

- **Runtime contract**: `Runtime` / `Session` / `Budget` / `EventBus` for an
  external coordinator to drive in-process or over HTTP; sessions can be
  snapshotted and resumed.
- **Hard budgets**: `max_usd`, `max_turns`, `max_input_tokens`,
  `max_output_tokens` enforced inside the agent loop, not just advisory.
- **Lifecycle events**: every turn, tool call, text/thinking delta, critic
  score and budget hit emits an event a coordinator can subscribe to (in
  process via `bus.subscribe(...)`, or over the network via SSE).
- **Subagent delegation**: a `delegate(role, task)` tool spawns a
  role-specialized child (planner / executor / critic / researcher /
  summarizer); child token usage rolls up into the parent.
- **Skill library**: `~/.agi/skills/*.md` — markdown SOPs the agent can
  read, write (`save_skill`), and search; matched skills get threaded into
  the system prompt at session creation.
- **Reflection**: a `reflect(task, what_worked, what_failed, lesson)` tool
  writes a tagged note to long-term memory at the end of a task.
- **Trace-quality critic**: an opt-in CPU-trainable specialist that gates
  output by predicted P(passed).
- **Tool surface**: read/write files, run shell, search/fetch the web,
  save/search/recent memory, list/read/save/search skills, delegate, reflect.
- **Cost & token accounting**: per-turn deltas + cumulative; pricing table
  for Opus 4.7 / Sonnet 4.6 / Haiku 4.5 with cache write/read multipliers.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus runtime is a frozen reasoning
core; the `learner/` track is the path toward durable improvement through
weight updates on a small open base. See `ARCHITECTURE.md` for the design
and the open research problems we are not solving.
