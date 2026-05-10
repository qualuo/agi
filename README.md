# agi

An agent harness. Not AGI — that remains an unsolved scientific problem. This
is what you can credibly build today: a capable agent on top of Claude Opus 4.7
with tools to act on the world, persistent memory across sessions, and an eval
harness to measure capability.

## What's in here

```
agi/                # frozen-Opus harness (capability track)
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  costs.py          # per-turn + cumulative token usage and $ tracking
  memory.py         # persistent JSONL memory store with keyword search
  skills.py         # markdown skill library; named, retrievable procedures
  tools.py          # filesystem, shell, web search/fetch, memory + skill tools
  runtime.py        # JSON-line stdio protocol — drive the agent as a subprocess
  __main__.py       # CLI: python -m agi [--runtime]
coord/              # coordination clients (drive runtimes over JSON-line pipe)
  client.py         # RuntimeClient: spawn + request/response/stream
  demo.py           # demo coordinator: pool N runtimes, fan out tasks
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
  test_skills.py    # skill library + tool integration
  test_runtime.py   # runtime protocol with an injected fake agent
  test_runtime_e2e.py # end-to-end: real subprocess driven over a pipe
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
python -m agi                          # interactive REPL
python -m agi "summarize ./README.md"  # one-shot
python -m agi --runtime                # JSON-line stdio runtime (see below)
python coord/demo.py "what's 2+2?"     # demo coordinator that drives a runtime
python coord/demo.py --pool 3 a b c    # pool 3 runtimes, fan out 3 tasks
python evals/run.py                    # run the eval suite
```

## What it can do

- Read/write files, run shell commands
- Search the live web (server-side `web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Accumulate skills (`~/.agi/skills/*.md`): the agent saves successful procedures
  as named markdown SOPs, then retrieves them on similar future tasks
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output as Claude generates it; show summarized thinking
- Track per-turn and cumulative token usage with $ cost
- Run as a **subprocess driven by an external coordination engine** over a
  JSON-line stdio protocol. Memory, skills, chat, and streamed events all
  exposed (see `agi/runtime.py` for the wire format)

## Runtime protocol — agent as a swappable building block

The agent ships with a JSON-line stdio protocol so a coordination engine can
drive it as a subprocess. Each line of stdin is a JSON request; each line of
stdout is a JSON event or terminal result. This lets a coordinator:

- Pool many runtimes (different models / memory scopes / skill libraries) and
  route work between them
- Get crash isolation — a misbehaving agent doesn't take down the coordinator
- Be written in any language (Go, Rust, TS) without depending on the Python
  agent code

```sh
# Start a runtime
python -m agi --runtime
# (writes {"type":"ready",...} on stdout)
# Then feed it requests:
{"id":"1","type":"capabilities"}
{"id":"2","type":"chat","input":"what's 2+2?"}
{"id":"3","type":"shutdown"}
```

The Python `coord.client.RuntimeClient` is a reference implementation of the
client side; `coord/demo.py` shows pooling and fan-out. See `agi/runtime.py`
for the full request/response catalog.

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. The Opus harness is a frozen system; it
doesn't learn. The `learner/` track is the path toward durable improvement
through weight updates on a small open base — not a frontier model, but
actually a system that learns. See `ARCHITECTURE.md` for the design.
