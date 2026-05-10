# agi

A learning agent harness with a runtime engine. Not AGI — that remains an
unsolved scientific problem. This is what you can credibly build today: a
capable agent on top of Claude Opus 4.7, plus a runtime layer designed
to be driven by an external coordination engine over a stable HTTP/JSON
contract or in-process Python API.

## What's in here

```
agi/                     # core agent + runtime
  agent.py               # streaming agent loop with lifecycle hooks
  costs.py               # per-turn + cumulative token usage and $ tracking
  memory.py              # persistent JSONL memory store with keyword search
  skills.py              # procedural memory: markdown skills + retrieval
  reflection.py          # auto-distill durable lessons after each task
  sandbox.py             # restricted Python exec for agent-authored tools
  tools.py               # tool registry: file/shell/web/memory/skills/synth/delegate
  runtime.py             # Runtime + Session: multi-session manager with roles
  server.py              # HTTP/JSON façade for the runtime
  __main__.py            # CLI: python -m agi
learner/                 # learning track — small open base + LoRA loop
  traces.py              # append-only JSONL trace logger
  filter.py              # quality gates: eval-pass, score threshold, thumbs
  critic.py              # tiny MLP that learns to predict trace quality
  goals.py / synth.py    # goal abstraction + synthetic data generators
  train.py               # LoRA SFT script (HF transformers + PEFT, GPU)
  train_critic.py        # CLI to train the critic on synthetic data
evals/
  tasks.jsonl            # eval tasks: math, file ops, recall, skills, synth
  run.py                 # eval runner — reports pass/fail per task
tests/                   # 90 unit tests across smoke, critic, skills,
                         # sandbox, reflection, runtime, server, learner
ARCHITECTURE.md          # full design — read this for direction
PLAN.md                  # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Three ways to use it

### 1. CLI / REPL — solo task

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

### 2. In-process runtime — Python coordination engines

```python
from agi import Runtime

rt = Runtime(enable_reflection=True)

# Plan with a focused planner sub-agent
plan = rt.create_session(role="planner", goal="Refactor module X")
plan_text = plan.step("Refactor agi/tools.py to support async dispatch").text
plan.end()

# Then have an executor carry out one step under coordination control
exec_sess = rt.create_session(role="executor", goal="Implement plan step 1")
result = exec_sess.step(f"Plan:\n{plan_text}\n\nDo step 1.")
print(result.text, result.usage.cost_usd("claude-opus-4-7"))

# Finally critique the result
critique = rt.create_session(role="critic")
critique.step(f"Review this work:\n{result.text}")

# Aggregate accounting across sessions
print(rt.stats())
```

### 3. HTTP runtime — out-of-process coordination engines

```sh
AGI_API_TOKEN=secret python -m agi.server --host 0.0.0.0 --port 8765
```

```sh
# Discover capabilities
curl -H 'Authorization: Bearer secret' http://localhost:8765/v1/capabilities

# Create a session
curl -X POST -H 'Authorization: Bearer secret' \
     -H 'Content-Type: application/json' \
     -d '{"role":"general","goal":"summarize ./README.md"}' \
     http://localhost:8765/v1/sessions

# Step the session
curl -X POST -H 'Authorization: Bearer secret' \
     -H 'Content-Type: application/json' \
     -d '{"input":"summarize ./README.md"}' \
     http://localhost:8765/v1/sessions/<id>/step

# Inject an environment observation between steps
curl -X POST -H 'Authorization: Bearer secret' \
     -H 'Content-Type: application/json' \
     -d '{"text":"the deploy completed"}' \
     http://localhost:8765/v1/sessions/<id>/inject
```

The full route table lives in `agi/server.py`. All routes return JSON;
errors look like `{"error": "..."}`.

## What it can do

- **Read/write files, run shell commands, search/fetch the web**
- **Persistent memory** — `~/.agi/memory.jsonl` with keyword search
- **Skill library** — `~/.agi/skills/*.md`; agent retrieves top-K relevant
  skills into the system prompt for each turn, can author new ones
- **Reflection** (opt-in) — Haiku call per turn distills durable lessons
  back into long-term memory tagged `lesson`
- **Self-extension** — `make_tool` lets the agent author Python functions
  in a restricted sandbox (whitelisted stdlib, no I/O, no eval) and
  register them as callable tools for the rest of the session
- **Multi-agent delegation** — `delegate(role, task)` spawns a sub-agent
  with role-specific prompt and tool subset (planner / executor / critic)
- **Lifecycle hooks** — text/thinking deltas, tool calls, tool results,
  and step completion stream to user-provided callbacks for the
  coordination engine
- **Critic gate** (opt-in) — the trace-quality critic from `learner/`
  scores responses below a threshold and annotates them
- **Capability descriptors** — `Runtime.capabilities()` returns a stable
  JSON shape describing roles, tools, and policy knobs

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust
transfer, grounded world models, durable goals across weeks. The Opus
harness is a frozen system; only the LoRA loop in `learner/` produces
weight updates. The roadmap in `ARCHITECTURE.md` is honest about which
problems are research-open and which are tractable engineering.
