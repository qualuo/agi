# agi

An agent harness. Not AGI — that remains an unsolved scientific problem. This
is what you can credibly build today: a capable agent on top of Claude Opus 4.7
with tools to act on the world, persistent memory across sessions, an eval
harness to measure capability, and a JSON HTTP runtime so an external
coordination engine can drive it.

## What's in here

```
agi/                # frozen-Opus harness (capability track)
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  costs.py          # per-turn + cumulative token usage and $ tracking
  memory.py         # persistent JSONL memory store with keyword search
  tools.py          # filesystem, shell, web search/fetch, memory tools
  runtime.py        # stateful, multi-session wrapper for external drivers
  server.py         # stdlib HTTP server exposing the runtime as JSON
  __main__.py       # CLI: python -m agi [prompt | serve]
learner/            # learning track — small open base + LoRA loop
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
  critic.py         # trace-quality critic (small CPU-trainable specialist)
  skills.py         # markdown skill library — medium-timescale procedural memory
  README.md         # how to run training
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner — reports pass/fail per task
tests/              # offline smoke + integration tests (no API calls required)
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
python -m agi serve --port 8088       # HTTP runtime for an external driver
python evals/run.py                   # run the eval suite
```

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Load relevant procedural skills from `~/.agi/skills/` into the system prompt per turn
- Spawn focused sub-agents via the optional `delegate` tool, with usage rolled up
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost
- Serve as a stateful runtime: sessions, structured turn results, capability discovery

## Runtime / coordination engine surface

The harness is designed to be **driven** by an external coordination engine
(another agent, an orchestrator, an IDE plugin). `agi.Runtime` is the in-process
contract; `python -m agi serve` exposes it over HTTP/JSON.

### Endpoints

```
GET    /v1/health                       liveness
GET    /v1/describe                     model, tools, skills, pricing
GET    /v1/sessions                     list active sessions
POST   /v1/sessions                     create a session (body: {"session_id"?})
GET    /v1/sessions/{id}                session info + cumulative usage
DELETE /v1/sessions/{id}                end session
POST   /v1/sessions/{id}/reset          clear history, keep session id
POST   /v1/sessions/{id}/turn           run one turn (body: {"input": "..."})
```

A `turn` returns a structured result that a coordinator can route on:

```json
{
  "text": "...",
  "usage": {"input_tokens": ..., "output_tokens": ..., ...},
  "cost_usd": 0.0123,
  "critic_score": 0.81,
  "skills_used": ["solve-quadratic"],
  "finish_reason": "ok",
  "elapsed_seconds": 4.2,
  "error": null
}
```

Set `AGI_AUTH_TOKEN` (or `--auth-token`) to require `Authorization: Bearer ...`
on every request. Auth is intentionally minimal — wrap behind your own gateway
if you need more.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn from its own weight updates. The `learner/` track is the path
toward durable improvement through LoRA training on a small open base. The
skill library and the critic give us medium-timescale learning channels that
don't require GPU. See `ARCHITECTURE.md` for the design and the explicit list
of research-open problems we are *not* claiming to solve.
