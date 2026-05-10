"""Demo coordinator: spawn one or more runtimes, route work, print events.

What this shows:
  1. A runtime is a subprocess driven over a JSON-line pipe.
  2. The same coordinator can pool multiple runtimes (different memory/skill
     scopes, in principle different models) and route requests by capability.
  3. Streamed events (text deltas, tool calls) flow back to the coordinator
     in real time.

Usage:

    python coord/demo.py "what's 2+2?"             # one runtime, one task
    python coord/demo.py --pool 3 task1 task2 task3  # spawn 3, fan out

This is a demonstration; a production coordinator would do load balancing,
retries, capability-based routing, persistence, etc.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coord.client import RuntimeClient


_print_lock = threading.Lock()


def _printer(prefix: str):
    def on_event(event: dict) -> None:
        kind = event.get("kind", "")
        with _print_lock:
            if kind == "text_delta":
                sys.stdout.write(event.get("text", ""))
                sys.stdout.flush()
            elif kind == "tool_call":
                sys.stdout.write(f"\n[{prefix} tool {event.get('name')}({event.get('input')})]\n")
                sys.stdout.flush()
            elif kind == "tool_result":
                output = (event.get("output") or "").strip().splitlines()
                head = output[0][:120] if output else ""
                sys.stdout.write(f"[{prefix} -> {head}]\n")
                sys.stdout.flush()

    return on_event


def run_one(prefix: str, prompt: str) -> dict:
    with RuntimeClient() as rt:
        with _print_lock:
            print(f"\n=== runtime {prefix} ===\n> {prompt}\n", flush=True)
        result = rt.chat(prompt, on_event=_printer(prefix))
        with _print_lock:
            print(
                f"\n[{prefix} done — ${result['usage']['cost_usd']:.4f}, "
                f"{result['usage']['turns']} turns]"
            )
        return {"prefix": prefix, "prompt": prompt, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="+", help="One or more prompts to dispatch.")
    parser.add_argument("--pool", type=int, default=1, help="Max concurrent runtimes.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    if len(args.prompts) == 1:
        run_one("rt0", args.prompts[0])
        return 0

    print(f"dispatching {len(args.prompts)} tasks across pool of {args.pool}")
    with ThreadPoolExecutor(max_workers=args.pool) as ex:
        futures = [
            ex.submit(run_one, f"rt{i}", p) for i, p in enumerate(args.prompts)
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"task failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
