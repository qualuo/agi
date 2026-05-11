"""CLI entry: `python -m agi [prompt | serve | manifest]`.

Modes:
  python -m agi                  REPL on a single Agent
  python -m agi "do X"           one-shot
  python -m agi serve [--host H] [--port P]
                                 start the runtime HTTP server
  python -m agi manifest         dump the capability descriptor as JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from agi.agent import Agent


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)


def _serve(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="agi serve", description="Run the agi runtime HTTP server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", default=None, help="Bearer token (default: AGI_API_TOKEN env)")
    args = p.parse_args(argv)
    _check_api_key()
    from agi.runtime import Runtime
    from agi.server import serve
    runtime = Runtime()
    serve(runtime, host=args.host, port=args.port, auth_token=args.token)
    return 0


def _manifest() -> int:
    from agi.runtime import capability_manifest
    print(json.dumps(capability_manifest(), indent=2))
    return 0


def _repl() -> int:
    _check_api_key()
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


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        return _serve(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == "manifest":
        return _manifest()
    if len(sys.argv) > 1:
        _check_api_key()
        agent = Agent()
        prompt = " ".join(sys.argv[1:])
        agent.chat(prompt)
        print()
        return 0
    return _repl()


if __name__ == "__main__":
    sys.exit(main())
