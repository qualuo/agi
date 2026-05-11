# agi

An agent **runtime** built on Claude Opus 4.7. Not AGI — that remains an
unsolved scientific problem. This is what you can credibly build today: a
multi-session agent runtime with persistent memory, a skill library
(procedural memory), per-session budget controls, a typed event stream,
and an HTTP control plane that an external coordination engine can drive
in any language.

## What's in here

```
agi/                    # runtime
  agent.py              # streaming agent loop — adaptive thinking + tools
  budget.py             # cost/turn ceilings, raises BudgetExceeded
  costs.py              # per-turn + cumulative token usage and $ tracking
  events.py             # typed event bus (pub-sub)
  memory.py             # persistent JSONL memory store with keyword search
  runtime.py            # multi-session manager + capability manifest
  server.py             # stdlib HTTP control plane + SSE event stream
  session.py            # one isolated agent + memory + skills + budget
  skills.py             # procedural memory: markdown SOPs retrieved on overlap
  tools.py              # filesystem, shell, web, memory, skill, reflect, delegate
  __main__.py           # CLI: python -m agi [prompt | serve | manifest]
coordinator/            # reference coordination engine — drives the runtime
  dag.py                # tiny DAG executor: plan → executor → critic, etc.
learner/                # learning track — small open base + LoRA loop
  traces.py             # append-only JSONL trace logger
  filter.py             # quality gates: eval-pass, score threshold, thumbs
  critic.py             # tiny MLP trace-quality critic (CPU-tractable)
  train.py              # LoRA SFT script (HF transformers + PEFT, GPU)
evals/
  tasks.jsonl           # eval tasks (math, file ops, recall, search)
  run.py                # eval runner — reports pass/fail per task
tests/                  # 71 tests, none require an API key
ARCHITECTURE.md         # full design — read this for direction
PLAN.md                 # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Three ways to use it

### 1. CLI — single agent, ad-hoc

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python -m agi manifest                # dump capability descriptor
python evals/run.py                   # run the eval suite
```

### 2. Runtime, in-process — multi-session, drivable from Python

```python
from agi import Runtime, Budget, filter_types

rt = Runtime(default_budget=Budget(max_usd=0.50, max_turns=15))

# Spin up a session, chat, close.
s = rt.open_session(role="executor")
result = s.chat("read pyproject.toml and tell me the version")
print(result["text"], "—", result["usage"]["cost_usd"])
rt.close_session(s.id)

# See what happened on the event bus.
for evt in filter_types(rt.history(), "tool_call", "tool_result"):
    print(evt["type"], evt.get("name"))
```

### 3. HTTP service — drivable from any language

```sh
python -m agi serve --host 0.0.0.0 --port 8765
# optional auth: export AGI_API_TOKEN=secret  →  Authorization: Bearer secret
```

```sh
curl localhost:8765/v1/manifest                                    # capabilities
curl -X POST localhost:8765/v1/sessions \
     -H 'content-type: application/json' \
     -d '{"role":"executor","max_usd":0.25}'                       # → {"id": "<sid>", ...}
curl -X POST localhost:8765/v1/sessions/<sid>/chat \
     -H 'content-type: application/json' \
     -d '{"input":"summarize README.md in 3 bullets"}'             # → {"text": ..., "usage": ...}
curl -N localhost:8765/v1/events                                   # SSE: turn_*, tool_*, etc.
curl -X DELETE localhost:8765/v1/sessions/<sid>                    # close
```

The event stream is the live observability hook a coordination engine
subscribes to. Every session emits `session_opened`, `turn_started`,
`tool_call`, `tool_result`, `text`, `critic_score`, `turn_finished`,
`session_closed`, and `delegate_*` for subagent spawns.

## Capabilities the runtime exposes

- **Roles** — `general`, `planner`, `executor`, `critic`, `researcher` (each
  preloads a role-specific system prompt).
- **Tools** — `read_file`, `write_file`, `list_dir`, `run_bash`,
  `save_memory`, `search_memory`, `recent_memory`, `reflect`,
  `save_skill`, `list_skills`, `delegate`, plus server-side `web_search`
  and `web_fetch`.
- **Skill library** — markdown SOPs the agent writes via `save_skill` and
  the runtime auto-injects on relevant prompts. The medium-timescale
  learning channel: a task family gets cheaper the second time.
- **Budgets** — every session has hard `max_usd` / `max_turns` ceilings.
  Breach raises `BudgetExceeded`; the agent loop unwinds cleanly.
- **Subagents** — the `delegate` tool spawns a child session with its own
  budget; usage rolls up via the event bus.
- **Critic gate** — opt-in: a learned trace-quality critic scores the
  output and annotates if confidence is below threshold.
- **Trace logger** — every turn writes a JSONL record for the LoRA
  learning loop in `learner/`.

## Coordinator demo

`coordinator/` is a reference orchestrator. The DAG executor opens a
fresh session per node, threads parent outputs into child prompts, and
respects per-node budgets:

```python
from agi import Runtime, Budget
from coordinator import DAG, Node, run

rt = Runtime()
dag = DAG([
    Node("plan",  role="planner",  prompt="Plan: {ask}"),
    Node("do",    role="executor", prompt="Execute: {plan}",  deps=["plan"]),
    Node("grade", role="critic",   prompt="Did this satisfy {ask}? {do}",
         deps=["plan", "do"]),
])
results = run(dag, rt,
              inputs={"ask": "summarize README.md in 3 bullets"},
              budget=Budget(max_usd=0.20))
for r in results:
    print(r.name, r.stop_reason, f"${r.cost_usd:.4f}")
```

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust
transfer, grounded world models, durable goals. The Opus-backed runtime
is a frozen system; it doesn't update its own weights. The `learner/`
track is the path toward durable improvement through weight updates on a
small open base — not a frontier model, but actually a system that
learns. See `ARCHITECTURE.md` for the design.

## Tests

```sh
python -m unittest discover -s tests
```

71 tests; none require an API key (the runtime tests use a fake Anthropic
client). The torch-dependent learner tests need `pip install -e .[learner]`.
