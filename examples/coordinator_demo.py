"""Coordinator demo — drive two Runtimes from one Coordinator.

Run with:  python examples/coordinator_demo.py

Requires ANTHROPIC_API_KEY. Spends ~$0.10 — the Job budgets are tight.

Demonstrates:
  1. Capability introspection: coordinator lists what each runtime can do.
  2. Cheapest-first routing: cheap question goes to Haiku; hard one to Opus.
  3. Race: two runtimes attempt the same job; first acceptable answer wins.
  4. Budget enforcement: a deliberately tiny budget aborts mid-job.
  5. Snapshot + resume: branch a session, continue it on a different prompt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.protocol import Job, ProgressEvent
from agi.runtime import haiku_runtime, opus_runtime
from coord import Coordinator, RoutingPolicy
from coord.coordinator import CoordinatorBudget


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    coord = Coordinator(budget=CoordinatorBudget(max_total_usd=0.50))

    cheap = haiku_runtime(tags=["fast", "cheap"])
    smart = opus_runtime(tags=["frontier", "smart"])

    # 1. Capabilities --------------------------------------------------------
    coord.register(cheap)
    coord.register(smart)
    print("registered runtimes:")
    for cap in coord.capabilities():
        tools = ", ".join(t.name for t in cap.tools)
        print(f"  {cap.runtime_id}  model={cap.model}  tags={cap.tags}")
        print(f"    output rate: ${cap.cost_per_1m_output_usd}/1M  tools: {tools}")

    # Subscribe to mid-flight events on both runtimes for visibility.
    def on_event(e: ProgressEvent) -> None:
        if e.kind in ("job_started", "job_finished", "budget_check"):
            print(f"  [event] {e.kind} {e.payload}")
    cheap.subscribe(on_event)
    smart.subscribe(on_event)

    # 2. Route by policy -----------------------------------------------------
    print("\n-- cheapest routing for a trivial job --")
    r1 = coord.run(
        Job(prompt="What is 17 + 25? Reply with just the number.", max_cost_usd=0.05),
        policy=RoutingPolicy.CHEAPEST,
    )
    print(f"  status={r1.status.value}  output={r1.output[:80]!r}  cost=${r1.cost_usd:.4f}")

    # 3. Race ----------------------------------------------------------------
    print("\n-- race: cheap and smart both attempt; first acceptable wins --")
    winner = coord.race(
        Job(prompt="In one sentence, what is the central limit theorem?",
            max_cost_usd=0.05),
        runtime_ids=[cheap.runtime_id, smart.runtime_id],
        accept=lambda r: r.succeeded and len(r.output) > 30,
    )
    print(f"  winner runtime had session={winner.session_id}  cost=${winner.cost_usd:.4f}")
    print(f"  output: {winner.output[:160]}")

    # 4. Budget enforcement --------------------------------------------------
    print("\n-- tiny budget should trip the per-job ceiling --")
    starved = coord.run(
        Job(prompt="Write a 2000-word essay on the history of fungi.",
            max_cost_usd=0.0005),
        policy=RoutingPolicy.CHEAPEST,
    )
    print(f"  status={starved.status.value}  cost=${starved.cost_usd:.4f}  err={starved.error}")

    # 5. Snapshot + resume ---------------------------------------------------
    print("\n-- snapshot + resume: branch a session --")
    setup = coord.run(
        Job(prompt="Remember the codeword 'thalassa'. Acknowledge briefly.",
            max_cost_usd=0.05),
        policy=RoutingPolicy.CHEAPEST,
    )
    snap_rt = coord.runtimes[next(iter(coord.stats.by_runtime))]
    snap = snap_rt.snapshot(setup.session_id)
    branched_session = snap_rt.resume(snap)
    recall = snap_rt.submit(Job(
        prompt="What was the codeword I gave you?",
        session_id=branched_session,
        max_cost_usd=0.05,
    ))
    print(f"  recall in branched session: {recall.output[:120]!r}")

    # 6. Stats ---------------------------------------------------------------
    print("\nfinal stats:")
    print(f"  submitted={coord.stats.submitted}  succeeded={coord.stats.succeeded}  "
          f"failed={coord.stats.failed}  rejected_no_budget={coord.stats.rejected_no_budget}")
    print(f"  by_runtime={coord.stats.by_runtime}")
    print(f"  total spend=${coord.budget.spent_usd:.4f} / ${coord.budget.max_total_usd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
