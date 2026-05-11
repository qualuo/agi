# agi

An agent **runtime**, not AGI — that remains an unsolved scientific problem.
What you get today: a capable agent on top of Claude Opus 4.7 with tools to act
on the world, persistent memory across sessions, an eval harness to measure
capability, **and a programmable runtime layer so a coordination engine can
drive many concurrent agents through one stable streaming interface.**

## What's in here

```
agi/                # frozen-Opus harness + runtime
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  events.py         # typed event stream emitted during every turn
  runtime.py        # addressable, multi-session, streaming runtime
  coord.py          # reference Coordinator: declarative tasks over the runtime
  server.py         # HTTP/SSE server exposing the runtime
  costs.py          # per-turn + cumulative token usage and $ tracking
  memory.py         # persistent JSONL memory store with keyword search
  tools.py          # filesystem, shell, web search/fetch, memory tools
  __main__.py       # CLI: python -m agi
learner/            # learning track — small open base + LoRA loop
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
  critic.py         # tiny CPU-trainable critic; opt-in output gate
  README.md         # how to run training
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner — reports pass/fail per task
tests/
  test_smoke.py     # smoke tests for memory/tools/usage
  test_runtime.py   # event stream + session lifecycle + budgets
  test_coord.py     # Coordinator: leaves, parallel, subtasks, cost roll-up
  test_server.py    # HTTP/SSE wire format
  test_critic*.py   # critic train + gate (requires torch)
  test_learner.py   # learner package smoke (GPU-free parts)
  fake_client.py    # scripted Anthropic stub — drives the agent offline
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
python -m agi.server --port 8765      # start the HTTP/SSE runtime
```

## Drive it as a runtime

The runtime turns the agent into a service a coordination engine can call. In-
process Python:

```python
from agi import Runtime
rt = Runtime(default_budget_usd=2.00)
sid = rt.open()                          # addressable session id
rt.send(sid, "summarize ./README.md")
for ev in rt.stream(sid):
    print(ev.type, ev.to_dict())         # TurnStart, TextDelta, ToolUseStart,
                                         # ToolUseResult, UsageDelta, TurnEnd, ...
    if ev.type == "TurnEnd":
        break
rt.close(sid)
```

Same surface over HTTP:

```sh
# create a session
curl -X POST localhost:8765/v1/sessions -d '{"budget_usd": 2.0}'
# -> {"session_id":"a1b2c3d4e5f6"}

# send a message (returns immediately; events flow over SSE)
curl -X POST localhost:8765/v1/sessions/a1b2c3d4e5f6/messages \
  -d '{"input":"summarize ./README.md"}'

# stream typed events
curl -N localhost:8765/v1/sessions/a1b2c3d4e5f6/events
# event: TurnStart
# data: {"session_id":"...","seq":1,"user_input":"..."}
# event: TextDelta
# data: {"session_id":"...","seq":2,"text":"..."}
# ...
# event: TurnEnd
# data: {"session_id":"...","seq":N,"final_text":"...","cost_usd":0.0123}
```

## Compose it with a coordination engine

`agi.Coordinator` is the reference implementation of an in-process orchestrator
over the runtime. External engines should hit the HTTP API, but the same
patterns apply:

```python
from agi import Coordinator, Runtime, Task
rt = Runtime(default_budget_usd=2.00)
co = Coordinator(rt, max_parallel=4)

# Single task
result = co.run_one(Task(prompt="audit ./agi for security issues", role="executor"))

# Declarative subtask graph: fan out → synthesize
result = co.run_one(Task(
    prompt="report on the state of LoRA fine-tuning frameworks",
    subtasks=[
        Task(prompt="survey trl, peft, unsloth, axolotl", role="researcher"),
        Task(prompt="check recent papers on DoRA vs LoRA", role="researcher"),
        Task(prompt="benchmark numbers for 3B models on consumer GPUs", role="researcher"),
    ],
))
print(result.final_text)
print(f"total spent: ${result.total_cost_usd():.4f} across "
      f"{1 + len(result.children)} sessions")
```

Or hit `POST /v1/tasks` with the same shape from any language.

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost
- Expose every interaction as a **typed event stream** consumable in-process or
  over HTTP/SSE — every TextDelta, ToolUseStart, ToolUseResult, UsageDelta,
  TurnEnd flows through one channel
- Run **many sessions concurrently** with independent memory, budgets, and IDs
- Enforce **per-session $ budgets** that hard-stop further turns when exceeded
- Decompose a task into parallel subtasks with role specialization, then
  synthesize — with **cost roll-up** across the whole task tree

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design.
