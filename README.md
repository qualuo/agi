# agi

An agent runtime engine. Not AGI — that remains an unsolved scientific
problem. This is what you can credibly build today: a capable agent on top
of Claude Opus 4.7 with tools to act on the world, persistent memory and
skills across sessions, sandboxed self-extension via tool synthesis,
subagent delegation, and an event-driven runtime surface that a higher-level
coordination engine can drive.

## What's in here

```
agi/                # runtime + agent + reference coordinator
  runtime.py        # Runtime, Session, SessionConfig — the engine surface
  events.py         # EventBus + typed Event kinds (the coordination signal)
  server.py         # HTTP+SSE server exposing Runtime (stdlib only)
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  coordinator.py    # reference Coordinator + Goal/Plan/PlanStep abstractions
  autoloop.py       # AutonomousLoop — retry-with-lessons until goal accepted
  fork.py           # SessionFork — race N variants, pick winner by critic
  capabilities.py   # observed-performance routing — learn which roles win where
  skillmine.py      # mine reusable skills from successful trace patterns
  skills.py         # markdown skill library with retrieval (procedural memory)
  reflection.py     # per-task lessons-to-memory loop (medium-timescale learning)
  world_model.py    # observed-entity tracker (file/url/command + outcomes)
  toolsynth.py      # sandboxed Python tool synthesis (subprocess isolated)
  tasks.py          # Task / TaskQueue / TaskRunner — scheduled work
  persistence.py    # checkpoint sessions to disk and rehydrate
  memory.py         # persistent JSONL memory store + namespacing (multi-tenant)
  costs.py          # per-turn + cumulative token usage and $ tracking
  tools.py          # builtin tools: file, shell, web, memory (+ world auto-record)
  __main__.py       # CLI: python -m agi
learner/            # learning track — small open base + LoRA loop
  critic.py         # trace-quality critic (small MLP, trains on CPU)
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  goals.py          # Goal abstraction; Addition is the first concrete goal
  synth.py          # synthetic labeled data for critic warm-start
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner
tests/              # 128 unit tests, all run without an API key
ARCHITECTURE.md     # full design — read this for direction
PLAN.md             # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## CLI

```sh
python -m agi                          # interactive REPL
python -m agi "summarize ./README.md"  # one-shot
python evals/run.py                    # run the eval suite
python -m agi.server --port 8765       # start HTTP runtime
```

## Coordinator — the reference driver

The `Coordinator` is a reference implementation of a coordination engine
sitting *above* the Runtime. It accepts a `Goal` (declarative intent +
budget), runs it through a pluggable `decomposer` to produce a `Plan`
of dependent `PlanStep`s, dispatches each step as a `Task` against the
runtime queue, and aggregates step results.

```python
from agi.coordinator import Coordinator, Goal, Plan, PlanStep
from agi.runtime import Runtime

def planner(goal):
    return Plan(steps=[
        PlanStep(id="plan",       role="planner",    prompt=f"plan: {goal.intent}"),
        PlanStep(id="gather",     role="researcher", prompt=f"gather: {goal.intent}", depends_on=["plan"]),
        PlanStep(id="synthesize", role="writer",     prompt=f"summarize: {goal.intent}", depends_on=["gather"]),
    ])

result = Coordinator(Runtime(), decomposer=planner).run(
    Goal(intent="summarize LoRA adapters in production", budget_usd=1.0)
)
print(result.final_text)
```

The Coordinator talks to the Runtime only through its public API
(`create_session`, `chat`, `bus.subscribe`, `metrics`) — any other
planner can use the same surface. See `examples/coordinator_e2e.py`
for a full run including skill mining.

### Three drivers ship on top of the Runtime

These are coordination patterns shipped as small modules — a higher-level
coordination engine composes them or rolls its own:

- **`AutonomousLoop`** (`agi/autoloop.py`) — pursues a `Goal` across many
  attempts. Each failed attempt distills a lesson that is prepended to the
  next attempt's prompt; on success it mines a `SkillCandidate` from the
  winning trajectory. Halts on success, budget exhaustion, deadline, or
  iteration cap. Records every iteration to a `CapabilityRegistry` for
  downstream routing.

  ```python
  from agi.autoloop import AutonomousLoop, promote_skill
  from agi.capabilities import CapabilityRegistry

  caps = CapabilityRegistry()
  loop = AutonomousLoop(Coordinator(rt), max_iterations=4, capabilities=caps)
  result = loop.pursue(Goal(intent="…", acceptance=lambda t: "42" in t, budget_usd=1.0))
  if result.success and result.skill_candidate:
      promote_skill(rt, result.skill_candidate)  # graduate into the skill library
  ```

- **`SessionFork`** (`agi/fork.py`) — races N `SessionConfig` variants of the
  same prompt in parallel against the runtime's task queue and picks a
  winner via a pluggable `judge` (default: critic score, then succeeded,
  then cost). The cheapest way to lift pass rate on hard prompts.

  ```python
  from agi.fork import SessionFork, ForkVariant

  fork = SessionFork(rt, max_workers=4)
  race = fork.race("hard question", [
      ForkVariant("careful", SessionConfig(effort="high", role="planner")),
      ForkVariant("fast",    SessionConfig(effort="medium", role="executor")),
      ForkVariant("opus",    SessionConfig(model="claude-opus-4-7", role="reviewer")),
  ])
  print(race.winner.variant.name, race.winner.result)
  ```

- **`CapabilityRegistry`** (`agi/capabilities.py`) — append-only JSONL store
  of `(prompt_tokens, role, model, skills_used, success, cost, latency,
  critic_score)`. `recommend(prompt, budget_usd=…)` returns the best
  `(role, model)` bucket by similarity-weighted success rate, with a
  budget penalty. A coordinator queries this *before* dispatching work
  so each step picks the most-likely-to-succeed config.

  ```python
  rec = caps.recommend("compile this regex", budget_usd=0.05)
  cfg = rec.to_session_config(base=SessionConfig(max_tokens=8000))
  ```

See `examples/agi_demo.py` for an end-to-end narrated run that wires
all three together without an API key.

## Runtime API — for a coordination engine

The `Runtime` is the integration point. A coordination engine (orchestrator,
planner, scheduler — anything sitting above) creates sessions, drives them,
observes the event stream, enforces budgets, and queries capabilities.

```python
from agi.runtime import Runtime, SessionConfig

rt = Runtime()

# Discover what this runtime can do
caps = rt.capabilities()
# → {models, skills, synthesized_tools, active_sessions, ...}

# Subscribe to the event stream before running anything
rt.subscribe(lambda e: print(e.kind, e.data))

# Spawn a session with a per-session budget
sid = rt.create_session(SessionConfig(
    model="claude-opus-4-7",
    effort="high",
    enable_tool_synthesis=True,   # agent can write new tools at runtime
    enable_delegation=True,        # agent can spawn subagents
    use_skills=True,               # relevant skills auto-loaded into prompt
    cost_ceiling_usd=5.00,         # session ends when budget is hit
))

result = rt.chat(sid, "Plan and execute: …")

# State + accounting available any time
print(rt.get_session(sid).to_dict())

# Persist a learned procedure as a durable skill
from agi.skills import Skill
rt.save_skill(Skill(
    name="bisect_by_test",
    description="locate a regression by running the test against bisected commits",
    body="1. Identify last-known-good commit.\n2. git bisect run …",
    tags=["debugging", "git"],
))
```

## HTTP / SSE surface

`python -m agi.server` exposes the Runtime over HTTP for out-of-process
coordinators:

| Method | Path                              | Purpose                                       |
|--------|-----------------------------------|-----------------------------------------------|
| GET    | `/healthz`                        | Liveness check                                |
| GET    | `/capabilities`                   | What the runtime offers right now             |
| GET    | `/metrics`                        | Counters + totals for SLO/observability       |
| GET    | `/sessions`                       | List sessions                                 |
| POST   | `/sessions`                       | Create a session (body = SessionConfig + `namespace`) |
| GET    | `/sessions/{id}`                  | Inspect state                                 |
| POST   | `/sessions/{id}/chat`             | One turn; returns `{final_text, session}`     |
| POST   | `/sessions/{id}/cancel`           | Cancel between turns                          |
| POST   | `/sessions/{id}/reset`            | Clear conversation, keep session              |
| POST   | `/sessions/{id}/checkpoint`       | Persist session to the session store          |
| POST   | `/sessions/restore`               | `{session_id}` → reload from store            |
| DELETE | `/sessions/{id}`                  | End                                           |
| GET    | `/events`                         | SSE stream of all events                      |
| GET    | `/events?session_id=…&kind=…`     | Filtered SSE                                  |
| GET    | `/events/history`                 | Replay past events                            |
| GET    | `/tasks`                          | List queued/running/done tasks                |
| POST   | `/tasks`                          | Submit a task (prompt + budget + deadline)    |
| GET    | `/tasks/{id}`                     | Inspect a task                                |
| POST   | `/tasks/drain`                    | Run queued tasks (synchronous; `max_ticks`)   |
| POST   | `/skills`                         | Save a skill                                  |
| POST   | `/tools`                          | Synthesize a sandboxed tool                   |

Optional bearer-token auth via `AGI_AUTH_TOKEN` env var or `--auth-token`.

## Event kinds (the coordination contract)

The bus emits typed events. Coordinators pattern-match on `kind`:

- `session.created` / `session.ended`
- `chat.started` / `chat.completed`
- `usage.updated` — running token + cost totals
- `skill.loaded` — a skill was injected into a prompt
- `subagent.started` / `subagent.completed` — delegation
- `tool.synthesized` — agent extended itself
- `critic.scored` — gate fired with a confidence score
- `autoloop.iteration_started` / `autoloop.iteration_completed`
- `autoloop.completed` / `autoloop.failed` / `autoloop.budget_exhausted`
- `autoloop.skill_promoted` — a winning trajectory graduated into the library
- `fork.race_started` / `fork.race_completed`
- `error` — including `CostCeilingExceeded` when budget runs out

Subagent token usage rolls up into the parent session for honest accounting.

## What it can do

- Read/write files, run shell commands
- Search the live web (`web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`)
- Load **relevant skills automatically** before answering (procedural memory)
- **Synthesize new tools mid-session** in a sandboxed subprocess (AST scan +
  banned imports + smoke test + per-call timeout)
- **Delegate subtasks to specialist subagents** with cost roll-up
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output and emit a typed event for every state transition
- Enforce per-session **cost ceilings** at the runtime layer
- Critic gate: scores final responses and annotates low-confidence ones

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement on the base
model, robust out-of-distribution transfer, grounded world models, durable
goal pursuit across weeks of autonomy. The Opus reasoning core is frozen.
The `learner/` track is the path toward durable improvement via LoRA on a
small open base. See `ARCHITECTURE.md` for the design and what's open
research vs. tractable engineering.

## Testing

```sh
python -m unittest discover tests
# 165+ tests across events, skills, toolsynth, runtime, server, persistence,
# tasks, coordinator, autoloop, fork, capabilities, skillmine, agent, learner
```

All tests run without an API key; they exercise the runtime, sandbox, and
HTTP server via a `FakeAgent` factory so CI doesn't burn budget.
