# agi — AI runtime engine

An **addressable runtime** for AI agent execution. A coordination engine
upstream dispatches typed tasks; this runtime executes them with planning,
tool use, persistent memory, skill retrieval, and budget enforcement, then
streams structured progress events back. Not AGI — that remains an unsolved
scientific problem. This is the kind of substrate a coordinator needs to
run real workloads.

## What you get

- **Typed Task / TaskResult / Event** — a stable JSON contract a
  coordination engine can speak. No "send a string, hope for the best."
- **RuntimeEngine** — executes tasks synchronously or in the background,
  enforces token / cost / wall-clock / iteration budgets, isolates each
  task in a fresh agent so state never bleeds.
- **Event stream** — real-time `started → iteration → critic → succeeded`
  (or `failed` / `budget_exceeded` / `cancelled`) per task, both via an
  in-process subscriber and over HTTP Server-Sent Events.
- **Capability introspection** — `/v1/capabilities` returns the model,
  pricing, tools, available skills, and default budgets so the coordinator
  can route work to the right runtime.
- **Skill library** — durable procedural memory (markdown SOPs) loaded
  into the prompt by name or keyword. Persists across processes.
- **HTTP server** — stdlib only, no new deps. Run `agi-runtime` and any
  coordinator that speaks HTTP/JSON can drive it.
- **Frozen-Opus reasoning core** — Claude Opus 4.7 with adaptive thinking,
  streaming output, prompt caching, web search and web fetch, file/shell/
  memory tools.
- **Trace + critic learning track** — every interaction is logged; a
  tiny CPU-trainable critic gates output and produces a dense reward
  signal for the LoRA loop on the local-base track (see `ARCHITECTURE.md`).

## Repo layout

```
agi/
  runtime.py        # Task / TaskResult / Event / RuntimeEngine
  capabilities.py   # Capabilities + ToolSpec — what the runtime advertises
  skills.py         # SkillLibrary — procedural memory
  server.py         # HTTP/SSE server, stdlib only
  agent.py          # streaming agent loop (Opus + tools + memory)
  memory.py         # persistent JSONL memory + keyword search
  tools.py          # file / shell / memory tools
  costs.py          # token & $ accounting
  __main__.py       # CLI REPL
learner/            # trace logger, critic specialist, LoRA training (optional)
evals/              # eval suite that gates progress
tests/              # unittest; runtime + skills + capabilities + server are fully covered
ARCHITECTURE.md     # full design — read this for direction
PLAN.md             # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Drive the runtime from a coordination engine

### In-process (Python coordinator)

```python
from agi import RuntimeEngine, Task, Budget
from agi.agent import Agent

engine = RuntimeEngine(agent_factory=lambda: Agent(verbose=False))

result = engine.execute(Task(
    instruction="Read ./README.md and summarize it in 3 bullets.",
    skills=["summarize"],
    budget=Budget(max_iterations=10, max_tokens=50_000, max_cost_usd=0.50, deadline_s=120),
))
print(result.status)        # "succeeded" | "budget_exceeded" | "failed" | "cancelled"
print(result.output)
print(result.cost_usd, result.iterations, result.critic_score)
for e in result.events:
    print(e.type, e.payload)
```

Concurrent dispatch (each task gets a fresh isolated agent):

```python
ids = [engine.submit(Task(instruction=t)) for t in tasks]
results = [engine.await_result(i) for i in ids]
```

Live event subscription:

```python
def on_event(ev):
    print(ev.type, ev.payload)

task_id = engine.submit(Task(instruction="..."), on_event=on_event)
```

### Over HTTP (any-language coordinator)

```sh
agi-runtime --host 127.0.0.1 --port 8765
```

```sh
# What can this runtime do?
curl localhost:8765/v1/capabilities

# Submit
curl -X POST localhost:8765/v1/tasks \
  -H 'content-type: application/json' \
  -d '{"instruction":"summarize ./README.md","budget":{"max_cost_usd":0.50}}'
# → {"task_id": "abc123...", "status": "running"}

# Poll
curl localhost:8765/v1/tasks/abc123

# Or stream events
curl -N localhost:8765/v1/tasks/abc123/events

# Cancel
curl -X POST localhost:8765/v1/tasks/abc123/cancel
```

Endpoints:

| Method | Path                          | Purpose                           |
|--------|-------------------------------|-----------------------------------|
| GET    | `/v1/health`                  | liveness                          |
| GET    | `/v1/capabilities`            | model, tools, skills, budgets     |
| POST   | `/v1/tasks`                   | submit a Task                     |
| GET    | `/v1/tasks/{id}`              | current TaskResult                |
| GET    | `/v1/tasks/{id}/events`       | SSE event stream                  |
| POST   | `/v1/tasks/{id}/cancel`       | cooperative cancellation          |

## Standalone REPL

The original interactive REPL still works for hands-on use:

```sh
python -m agi                         # interactive REPL
python -m agi "summarize ./README.md" # one-shot
python evals/run.py                   # run the eval suite
```

## What this can actually do

- Read/write files, run shell commands
- Search the live web (`web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Load procedural skills (`~/.agi/skills/*.md`) on demand
- Plan with adaptive thinking on hard tasks
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost
- Hard-enforce per-task budgets (tokens, $, deadline, iterations)
- Expose all of the above to a coordination engine over HTTP/JSON

## What this can't do

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals over weeks. The Opus core is frozen; the
runtime makes it *useful as infrastructure*, not smarter. The `learner/` track
is the path toward durable improvement through weight updates on a small open
base. See `ARCHITECTURE.md` for the design.
