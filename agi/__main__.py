"""CLI entry: `python -m agi [subcommand] [args]`.

Subcommands:
  repl                          interactive REPL (default if no args)
  oneshot "<prompt>"            run one prompt and exit
  serve [--host H] [--port P]   start the JSON-RPC HTTP server
  manifest                      print the capability manifest as JSON
  task "<prompt>" [--role R] [--budget USD]
                                run a single task via the Runtime

Back-compat: `python -m agi "prompt"` (no subcommand) → oneshot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from agi.agent import Agent
from agi.budget import Budget
from agi.runtime import Runtime
from agi.server import serve_blocking


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)


def _cmd_repl(_: argparse.Namespace) -> int:
    _require_api_key()
    agent = Agent()
    print("agi — type a prompt and hit enter. /reset clears history. Ctrl-D to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line == "/reset":
            agent.reset()
            print("(history cleared)")
            continue
        if line in ("/quit", "/exit"):
            return 0
        try:
            agent.chat(line)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue


def _cmd_oneshot(args: argparse.Namespace) -> int:
    _require_api_key()
    agent = Agent()
    agent.chat(args.prompt)
    print()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    _require_api_key()
    rt = Runtime(
        max_workers=args.workers,
        enable_reflection=args.reflect,
    )
    serve_blocking(rt, host=args.host, port=args.port)
    return 0


def _cmd_manifest(_: argparse.Namespace) -> int:
    rt = Runtime()
    try:
        print(json.dumps(rt.manifest().to_dict(), indent=2, default=str))
    finally:
        rt.shutdown(wait=False)
    return 0


def _cmd_task(args: argparse.Namespace) -> int:
    _require_api_key()
    rt = Runtime(enable_reflection=args.reflect)
    budget = Budget(max_usd=args.budget) if args.budget else None

    def _printer(ev) -> None:
        if ev.kind == "task.usage":
            return  # spammy
        print(f"[{ev.kind}] {json.dumps(ev.data, default=str)[:200]}", flush=True)

    rt.subscribe(_printer)
    try:
        handle = rt.submit(args.prompt, role=args.role, budget=budget)
        snap = handle.wait(timeout=args.timeout)
        print(f"\n--- result (status={snap['status']}, ${snap.get('cost_usd', 0):.4f}) ---")
        print(snap.get("result") or snap.get("error") or "(no output)")
        return 0 if snap["status"] == "succeeded" else 1
    finally:
        rt.shutdown(wait=False)


def main() -> int:
    # Back-compat: bare `python -m agi` → REPL, `python -m agi "prompt"` → oneshot.
    known = {"repl", "oneshot", "serve", "manifest", "task", "-h", "--help"}
    if len(sys.argv) == 1:
        return _cmd_repl(argparse.Namespace())
    if sys.argv[1] not in known:
        return _cmd_oneshot(argparse.Namespace(prompt=" ".join(sys.argv[1:])))

    parser = argparse.ArgumentParser(prog="agi", description="agi runtime engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_repl = sub.add_parser("repl", help="interactive REPL")
    p_repl.set_defaults(func=_cmd_repl)

    p_one = sub.add_parser("oneshot", help="run one prompt and exit")
    p_one.add_argument("prompt")
    p_one.set_defaults(func=_cmd_oneshot)

    p_serve = sub.add_parser("serve", help="start the JSON-RPC HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--workers", type=int, default=4)
    p_serve.add_argument("--reflect", action="store_true", help="enable reflection")
    p_serve.set_defaults(func=_cmd_serve)

    p_mani = sub.add_parser("manifest", help="print capability manifest")
    p_mani.set_defaults(func=_cmd_manifest)

    p_task = sub.add_parser("task", help="run one task via the runtime")
    p_task.add_argument("prompt")
    p_task.add_argument("--role", default="executor")
    p_task.add_argument("--budget", type=float, default=None, help="USD ceiling")
    p_task.add_argument("--timeout", type=float, default=300)
    p_task.add_argument("--reflect", action="store_true")
    p_task.set_defaults(func=_cmd_task)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
