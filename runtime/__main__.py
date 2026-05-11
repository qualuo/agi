"""Launch the runtime server.

    python -m runtime                       # default host/port
    python -m runtime --host 0.0.0.0 --port 8765
    python -m runtime --mock                # MockBackend (no API key required)

The mock flag is handy for letting a coordinator wire up its own integration
without burning Anthropic credits.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agi runtime engine HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a deterministic mock backend (no ANTHROPIC_API_KEY required).",
    )
    parser.add_argument(
        "--mock-text",
        default="ok",
        help="Text the mock backend echoes for every task (with --mock).",
    )
    parser.add_argument(
        "--skills",
        default=None,
        help="Path to the skill library directory. Default: ~/.agi/skills",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.mock and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "error: ANTHROPIC_API_KEY not set. Pass --mock to run with a deterministic backend.",
            file=sys.stderr,
        )
        return 2

    from runtime.backend import AnthropicBackend, MockBackend
    from runtime.engine import Engine
    from runtime.server import serve
    from learner.skills import SkillLibrary

    backend = MockBackend.echo(args.mock_text) if args.mock else AnthropicBackend()
    skill_library = SkillLibrary(args.skills) if args.skills else SkillLibrary()
    engine = Engine(
        backend=backend,
        max_concurrent=args.max_concurrent,
        skill_library=skill_library,
    )
    server = serve(engine, host=args.host, port=args.port, skill_library=skill_library)

    print(f"runtime listening on http://{args.host}:{args.port}", flush=True)
    print(f"backend: {type(backend).__name__}, skills: {skill_library.path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...", flush=True)
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
