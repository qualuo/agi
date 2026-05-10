# agi

An agent scaffold. Not AGI — that remains an unsolved scientific problem. This
is what you can credibly build today: a capable agent on top of Claude Opus 4.7
with tools to act on the world, persistent memory across sessions, and an eval
harness to measure capability.

## What's in here

```
agi/
  agent.py      # core agent — adaptive thinking + tool runner loop
  memory.py     # persistent JSONL memory store with keyword search
  tools.py      # filesystem, shell, web search, memory tools
  __main__.py   # CLI: python -m agi
evals/
  tasks.jsonl   # eval tasks (math, file ops, recall, search)
  run.py        # eval runner — reports pass/fail per task
tests/
  test_smoke.py # smoke tests
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
- Search the live web (server-side `web_search_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`) and recall by keyword
- Plan with adaptive thinking on hard tasks (`effort: high`)

## What it can't do

Everything that makes AGI AGI: open-ended self-improvement, robust transfer,
grounded world models, durable goals. This is a useful tool, not a mind.
