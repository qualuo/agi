"""CLI: `python -m runtime` starts the runtime engine HTTP server.

Flags:
    --host HOST          bind address (default 127.0.0.1)
    --port PORT          tcp port (default 8765)
    --workers N          job thread-pool size (default 8)
    --backend opus|mock  agent backend; mock needs no API key (default opus)
    --root PATH          state directory (default ~/.agi/runtime)
    --token TOKEN        require Authorization: Bearer TOKEN
                         (or set AGI_RUNTIME_TOKEN)

Environment:
    ANTHROPIC_API_KEY    required when --backend=opus
    AGI_RUNTIME_BACKEND  default backend if --backend omitted
    AGI_RUNTIME_TOKEN    default auth token if --token omitted
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from runtime.runtime import Runtime
from runtime.server import serve_forever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m runtime")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=["opus", "mock"], default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.backend == "opus" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set; pass --backend=mock for offline.", file=sys.stderr)
        return 2

    if args.backend is not None:
        os.environ["AGI_RUNTIME_BACKEND"] = args.backend

    runtime = Runtime(root=args.root, max_workers=args.workers)
    serve_forever(host=args.host, port=args.port, auth_token=args.token, runtime=runtime)
    return 0


if __name__ == "__main__":
    sys.exit(main())
