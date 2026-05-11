"""Example: a tiny coordination engine that drives the runtime.

A coordination engine is the thing that decides *what to do*. The runtime
(this repo) is the thing that knows *how to do it*. This example shows a
coordinator that:

  1. Takes a high-level goal
  2. Asks the runtime to plan it (a single agent task)
  3. Watches the plan come back, decomposes it into subtasks
  4. Submits the subtasks in parallel
  5. Aggregates results, then submits a final task to summarize

Two run modes:

    python examples/coordinator.py --mock          # in-process, deterministic
    python examples/coordinator.py                 # in-process, real Anthropic API

A real coordinator would more likely talk to a remote runtime over HTTP
(`runtime.client.RuntimeClient`). This example uses the in-process Engine
for simplicity; the HTTP path is exercised by `tests/test_server.py`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.backend import AnthropicBackend, MockBackend
from runtime.engine import Engine
from runtime.task import Budget


def coordinate(engine: Engine, goal: str) -> dict:
    """Decompose `goal` into N parallel subtasks, run them, summarize."""

    # Step 1: planning task. We don't actually need the model to plan in this
    # demo — a real coordinator might use a small fast model to draft a plan.
    # Here we hard-code a trivial decomposition: ask the runtime to handle
    # three sub-questions in parallel.
    subgoals = [
        f"Step 1 of: {goal}. Be concise.",
        f"Step 2 of: {goal}. Be concise.",
        f"Step 3 of: {goal}. Be concise.",
    ]

    # Step 2: dispatch in parallel.
    parallel_budget = Budget(max_turns=5, max_cost_usd=0.50, deadline_seconds=120)
    children = [
        engine.submit(sg, budget=parallel_budget, metadata={"phase": "decompose", "i": i})
        for i, sg in enumerate(subgoals)
    ]
    for c in children:
        print(f"  -> submitted child {c.id}: {c.instruction[:60]}")

    # Step 3: wait for all of them. A coordinator might prefer event streams
    # for live observability; here we just block.
    results: list[str] = []
    for c in children:
        c.wait(timeout=120)
        snap = c.snapshot()
        results.append(f"[{snap.id}] {snap.result or snap.error}")
        print(f"  <- child {c.id} finished: {snap.status.value} (${snap.cost_usd:.4f})")

    # Step 4: aggregation. Hand the partial results back to the runtime to
    # produce a unified answer.
    aggregation_prompt = (
        f"Original goal: {goal}\n\n"
        f"Partial results from parallel subtasks:\n\n" + "\n\n".join(results) + "\n\n"
        f"Produce a single concise final answer."
    )
    final = engine.submit(
        aggregation_prompt,
        budget=Budget(max_turns=5, max_cost_usd=0.50),
        metadata={"phase": "aggregate"},
    )
    final.wait(timeout=120)
    final_snap = final.snapshot()

    return {
        "goal": goal,
        "subtasks": [c.snapshot().as_dict() for c in children],
        "final": final_snap.as_dict(),
        "total_cost_usd": round(
            sum(c.snapshot().cost_usd for c in children) + final_snap.cost_usd, 6
        ),
    }


def build_mock_engine() -> Engine:
    """A mock engine for offline runs. The mock backend echoes a per-task
    deterministic string so the coordinator demo can run end-to-end."""

    def responder(messages):
        from runtime.backend import MockBlock, MockMessage, MockUsage
        # Find the most recent user message to echo back something relevant.
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    last_user = content
                    break
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            last_user = str(block.get("content", ""))
                            break
                        if hasattr(block, "type") and block.type == "tool_result":
                            last_user = str(getattr(block, "content", ""))
                            break
                break
        return MockMessage(
            content=[MockBlock(type="text", text=f"mock answer to: {last_user[:80]}")],
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=10, output_tokens=20),
        )

    backend = MockBackend(responder=responder)
    return Engine(backend=backend, max_concurrent=4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default="Explain how this repository's runtime engine differs from a frozen agent harness.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock backend.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.mock or not os.environ.get("ANTHROPIC_API_KEY"):
        if not args.mock:
            print("(no ANTHROPIC_API_KEY — falling back to --mock)")
        engine = build_mock_engine()
    else:
        engine = Engine(backend=AnthropicBackend(), max_concurrent=4)

    print(f"\n=== coordinator dispatching: {args.goal!r} ===\n")
    t0 = time.time()
    with engine:
        result = coordinate(engine, args.goal)
    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed:.1f}s, total ${result['total_cost_usd']:.4f} ===\n")
    print("Final:", result["final"]["result"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
