"""CLI entry: `python -m agi [subcommand] [args]`.

Subcommands:

    python -m agi                              # interactive REPL (Opus)
    python -m agi "<prompt>"                   # one-shot
    python -m agi serve [--host H --port P]    # run the Runtime HTTP server
    python -m agi caps                         # print runtime capabilities (JSON)
    python -m agi skills list                  # list installed skills
    python -m agi skills add <desc> <body>     # add a skill
    python -m agi skills promote <id>          # promote a skill into the active library

No subcommand and no prompt → REPL.
"""
from __future__ import annotations

import json
import os
import sys

from agi.agent import Agent
from agi.runtime import Runtime
from agi.skills import SkillLibrary


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)


def _cmd_serve(argv: list[str]) -> int:
    from agi.server import serve_forever
    host = "127.0.0.1"
    port = 8088
    i = 0
    while i < len(argv):
        if argv[i] == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            i += 2
        elif argv[i] == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 2
        else:
            print(f"unknown serve arg: {argv[i]}", file=sys.stderr)
            return 2
    _check_api_key()
    serve_forever(host=host, port=port)
    return 0


def _cmd_caps() -> int:
    runtime = Runtime()
    print(json.dumps(runtime.capabilities(), indent=2, default=str))
    return 0


def _cmd_skills(argv: list[str]) -> int:
    lib = SkillLibrary()
    if not argv:
        argv = ["list"]
    op = argv[0]
    if op == "list":
        for s in lib.all():
            badge = "promoted" if s.promoted else "draft"
            print(f"[{badge}] {s.id}\n  {s.description}")
        return 0
    if op == "add":
        if len(argv) < 3:
            print("usage: skills add <description> <body>", file=sys.stderr)
            return 2
        skill = lib.add(argv[1], argv[2])
        print(f"added {skill.id}")
        return 0
    if op == "promote":
        if len(argv) < 2:
            print("usage: skills promote <id>", file=sys.stderr)
            return 2
        s = lib.promote(argv[1])
        if s is None:
            print(f"no skill: {argv[1]}", file=sys.stderr)
            return 1
        print(f"promoted {s.id}")
        return 0
    if op == "remove":
        if len(argv) < 2:
            print("usage: skills remove <id>", file=sys.stderr)
            return 2
        ok = lib.remove(argv[1])
        print("removed" if ok else "not found")
        return 0 if ok else 1
    print(f"unknown skills op: {op}", file=sys.stderr)
    return 2


def main() -> int:
    argv = sys.argv[1:]

    if argv and argv[0] == "serve":
        return _cmd_serve(argv[1:])
    if argv and argv[0] == "caps":
        return _cmd_caps()
    if argv and argv[0] == "skills":
        return _cmd_skills(argv[1:])

    _check_api_key()
    agent = Agent()

    if argv:
        prompt = " ".join(argv)
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
