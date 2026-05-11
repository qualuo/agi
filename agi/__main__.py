"""CLI entry: `python -m agi [prompt | serve | runs]`.

Modes:
  python -m agi                     interactive REPL on top of the runtime
  python -m agi "<prompt>"          one-shot run, prints final text + cost
  python -m agi serve [port]        start the runtime HTTP/SSE server
  python -m agi runs                list active/recent runs (requires server URL)
"""
from __future__ import annotations

import json
import os
import sys

from agi.runtime import Runtime, RunRequest


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)


def _serve(argv: list[str]) -> int:
    from agi.server import run_forever
    port = int(argv[0]) if argv else 8088
    _check_api_key()
    run_forever(port=port)
    return 0


def _one_shot(prompt: str) -> int:
    _check_api_key()
    rt = Runtime()
    run = rt.submit(RunRequest(task=prompt))
    for evt in run.events():
        if evt.type in {"text_delta"}:
            sys.stdout.write(evt.data.get("text", ""))
            sys.stdout.flush()
        elif evt.type == "tool_call":
            sys.stdout.write(f"\n[tool: {evt.data.get('name')}]\n")
        elif evt.type == "subrun_started":
            sys.stdout.write(f"\n[delegate → {evt.data.get('child_id')}: {evt.data.get('task')}]\n")
        elif evt.type == "reflection":
            sys.stdout.write(f"\n[lesson: {evt.data.get('text')}]\n")
        elif evt.type == "usage":
            print(f"\n[cost: ${evt.data.get('cost_usd', 0):.4f}]", flush=True)
    print()
    return 0


def _repl() -> int:
    _check_api_key()
    rt = Runtime()
    print("agi runtime — type a prompt. /skills lists skills, /runs lists runs, Ctrl-D to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/skills":
            for s in rt.skills.all():
                print(f"  {s.name}  ({s.uses} uses)  {s.description}")
            continue
        if line == "/runs":
            print(json.dumps(rt.list_runs(), indent=2, default=str))
            continue
        try:
            run = rt.submit(RunRequest(task=line))
            for evt in run.events():
                if evt.type == "text_delta":
                    sys.stdout.write(evt.data.get("text", ""))
                    sys.stdout.flush()
                elif evt.type == "tool_call":
                    sys.stdout.write(f"\n[tool: {evt.data.get('name')}]\n")
                elif evt.type == "reflection":
                    sys.stdout.write(f"\n[lesson: {evt.data.get('text')}]\n")
                elif evt.type == "usage":
                    print(f"\n[cost: ${evt.data.get('cost_usd', 0):.4f}]", flush=True)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        return _serve(sys.argv[2:])
    if len(sys.argv) > 1:
        return _one_shot(" ".join(sys.argv[1:]))
    return _repl()


if __name__ == "__main__":
    sys.exit(main())
