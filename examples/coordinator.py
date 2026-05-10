"""Reference coordination engine.

This is a *minimal* example showing how an external coordination engine
would drive the agi runtime: it creates role-specialized sessions, dispatches
turns to them in parallel, observes their lifecycle events, and produces a
rolled-up summary with aggregate cost.

It is intentionally not the runtime itself — the runtime lives in
`agi.runtime`. A coordination engine is a *consumer* of that runtime's
contract. Production coordinators would add: routing policy across many
runtimes, persistence, retry, fan-out/fan-in patterns, durable workflow
state, auth, etc. None of that is in scope here. The point of this file
is to show the contract surface is real and drivable.

Usage:

    # In-process (what this script does):
    python examples/coordinator.py "summarize the README"

    # Network: start `python -m agi serve` in one terminal and a
    # coordination engine on another machine talks the same shape over HTTP.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Make the agi package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.events import Event
from agi.runtime import Budget, Runtime, Session


@dataclass
class WorkItem:
    """A piece of work the coordinator routes to a runtime session."""

    role: str
    prompt: str
    budget_usd: float = 0.25


@dataclass
class TurnRecord:
    work: WorkItem
    text: str
    cost_usd: float
    duration_s: float
    error: str | None = None


class CoordinatorObserver:
    """Subscribes to a session's bus, prints a compact live trace.

    A real coordinator would persist these events, drive metric pipelines,
    or feed a UI. This one prints to stdout."""

    def __init__(self, work: WorkItem) -> None:
        self.work = work
        self._lock = threading.Lock()

    def __call__(self, ev: Event) -> None:
        with self._lock:
            t = ev.type
            tag = f"[{self.work.role}]"
            if t == "turn.started":
                print(f"{tag} ▶ start", flush=True)
            elif t == "tool.invoked":
                print(f"{tag}   tool {ev.data.get('name')}", flush=True)
            elif t == "tool.errored":
                print(f"{tag}   tool {ev.data.get('name')} errored: {ev.data.get('error')}", flush=True)
            elif t == "budget.exceeded":
                print(f"{tag}   budget: {ev.data.get('reason')}", flush=True)
            elif t == "turn.completed":
                cost = ev.data.get("cost_usd", 0)
                print(f"{tag} ◀ done (${cost:.4f})", flush=True)
            elif t == "turn.errored":
                print(f"{tag} ✗ {ev.data.get('error')}", flush=True)


def run_one(runtime: Runtime, work: WorkItem) -> TurnRecord:
    """Drive a single work item end-to-end through a fresh session.

    A real coordinator would reuse sessions across related items (so memory
    and conversation context carry over) and only create a new one when
    starting a new task tree. This example uses one-session-per-item for
    clarity."""
    session = runtime.create_session(role=work.role, budget=Budget(max_usd=work.budget_usd))
    session.bus.subscribe(CoordinatorObserver(work))
    t0 = time.time()
    result = session.step(work.prompt)
    duration = time.time() - t0
    record = TurnRecord(
        work=work,
        text=result.text,
        cost_usd=result.cost_usd,
        duration_s=duration,
        error=result.error,
    )
    runtime.close(session.id)
    return record


def coordinate(prompt: str, runtime: Runtime, max_workers: int = 3) -> dict:
    """Decompose `prompt` into role-specialized parallel work, run it,
    aggregate.

    Decomposition policy here is hard-coded for demo purposes. A real
    coordinator might dispatch a planner first and use its plan to derive
    further work items dynamically.
    """
    plan = [
        WorkItem(role="planner", prompt=f"Decompose this task into 3-5 numbered steps:\n\n{prompt}"),
        WorkItem(role="researcher", prompt=f"What background information would help with this task?\n\n{prompt}"),
        WorkItem(role="executor", prompt=prompt),
    ]

    records: list[TurnRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, runtime, w): w for w in plan}
        for f in as_completed(futures):
            records.append(f.result())

    total_cost = sum(r.cost_usd for r in records)
    return {
        "prompt": prompt,
        "fanout": len(plan),
        "total_cost_usd": total_cost,
        "results": [
            {
                "role": r.work.role,
                "duration_s": round(r.duration_s, 2),
                "cost_usd": round(r.cost_usd, 4),
                "error": r.error,
                "text_preview": r.text[:300],
            }
            for r in records
        ],
        "metrics": runtime.metrics.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reference coordination engine for the agi runtime.")
    p.add_argument("prompt", nargs="?", help="The task to coordinate. If omitted, runs a fixed demo.")
    p.add_argument("--max-workers", type=int, default=3)
    args = p.parse_args(argv)

    prompt = args.prompt or "Write a one-paragraph summary of what an agent runtime is."
    runtime = Runtime()
    summary = coordinate(prompt, runtime, max_workers=args.max_workers)

    print()
    print("=" * 70)
    print(f"fanout: {summary['fanout']}  total cost: ${summary['total_cost_usd']:.4f}")
    print(f"runtime metrics: {summary['metrics']['turns']}  cost: {summary['metrics']['cost']}")
    print("=" * 70)
    for r in summary["results"]:
        print(f"\n[{r['role']}] ${r['cost_usd']:.4f}  {r['duration_s']:.1f}s")
        if r["error"]:
            print(f"  error: {r['error']}")
        print(f"  preview: {r['text_preview']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
