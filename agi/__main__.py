"""CLI entry.

Three modes:

    python -m agi                  # interactive REPL
    python -m agi "<prompt>"       # one-shot prompt
    python -m agi --runtime        # JSON-line stdio runtime (for coordinators)

See agi.runtime for the runtime protocol.
"""
from __future__ import annotations

import os
import sys

from agi.agent import Agent
from agi.skills import SkillLibrary


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    args = sys.argv[1:]

    if args and args[0] == "--runtime":
        # Don't require an API key just to start the runtime — it may be
        # used for memory/skills-only operations. Chat will fail clearly
        # if the key is missing.
        from agi.runtime import Runtime
        return Runtime().serve()

    _check_api_key()
    agent = Agent(skills=SkillLibrary())

    if args:
        prompt = " ".join(args)
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
