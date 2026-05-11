"""CLI entry: `python -m agi [subcommand] [...]`.

  python -m agi                     # REPL on a single Agent (legacy)
  python -m agi "summarize foo.md"  # one-shot
  python -m agi serve [--host H] [--port P] [--concurrent N]
                                    # run the HTTP+SSE runtime server
  python -m agi plan PATH.json      # execute a plan file and print results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agi.agent import Agent


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)


def _cmd_serve(args: argparse.Namespace) -> int:
    _check_api_key()
    from agi.runtime import Runtime
    from agi.server import serve

    rt = Runtime(max_concurrent=args.concurrent)
    serve(rt, host=args.host, port=args.port, auth_token=args.token)
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    _check_api_key()
    from agi.coordinator import SimpleCoordinator
    from agi.plan import plan_from_dict
    from agi.runtime import Runtime

    plan_dict = json.loads(Path(args.path).read_text())
    plan = plan_from_dict(plan_dict)
    rt = Runtime(max_concurrent=args.concurrent)
    coord = SimpleCoordinator(rt)
    results = coord.run_plan(plan, timeout=args.timeout)
    out = {k: {
        "status": v.status,
        "session_id": v.session_id,
        "cost_usd": v.cost_usd,
        "elapsed_s": v.elapsed_s,
        "final_text": v.final_text,
    } for k, v in results.items()}
    print(json.dumps(out, indent=2, default=str))
    rt.shutdown()
    return 0


def _cmd_repl_or_oneshot(prompt_parts: list[str]) -> int:
    _check_api_key()
    agent = Agent()

    if prompt_parts:
        prompt = " ".join(prompt_parts)
        agent.chat(prompt)
        print()
        return 0

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


def main() -> int:
    # Treat first argv as a subcommand only if it's a known one. Otherwise
    # fall through to the legacy "REPL or one-shot prompt" behaviour, so
    # `python -m agi "summarize ./foo"` still works.
    subcommands = {"serve", "plan"}
    argv = sys.argv[1:]
    if argv and argv[0] in subcommands:
        parser = argparse.ArgumentParser(prog="agi")
        sub = parser.add_subparsers(dest="cmd", required=True)

        p_serve = sub.add_parser("serve", help="Run the runtime engine over HTTP+SSE")
        p_serve.add_argument("--host", default="127.0.0.1")
        p_serve.add_argument("--port", type=int, default=8765)
        p_serve.add_argument("--concurrent", type=int, default=4)
        p_serve.add_argument("--token", default=None, help="Bearer token (else $AGI_API_TOKEN)")
        p_serve.set_defaults(func=_cmd_serve)

        p_plan = sub.add_parser("plan", help="Execute a plan JSON file")
        p_plan.add_argument("path")
        p_plan.add_argument("--concurrent", type=int, default=4)
        p_plan.add_argument("--timeout", type=float, default=None)
        p_plan.set_defaults(func=_cmd_plan)

        args = parser.parse_args(argv)
        return args.func(args)

    return _cmd_repl_or_oneshot(argv)


if __name__ == "__main__":
    sys.exit(main())
