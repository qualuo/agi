"""CLI entry: `python -m agi [prompt]`.

No prompt → REPL. With a prompt → one-shot.
"""
from __future__ import annotations

import os
import sys

from agi.agent import Agent


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    _check_api_key()
    agent = Agent()

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
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


if __name__ == "__main__":
    sys.exit(main())
