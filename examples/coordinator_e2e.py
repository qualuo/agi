"""End-to-end: a Coordinator decomposes a Goal, drives the Runtime, mines
a skill from the successful pattern.

This is the full loop a coordination engine would run:
  1. Receive a Goal (user intent + budget).
  2. Decompose it into a Plan with multiple PlanSteps (parallel where
     possible, sequenced where there are dependencies).
  3. Dispatch each step as a Task to the Runtime's queue.
  4. Aggregate step results into a final answer.
  5. Mine a Skill candidate from successful similar runs so the *next*
     instance of this Goal is cheaper.

Uses FakeAgent so it runs without an API key. Swap agent_factory=None
to run against real Opus.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.coordinator import Coordinator, Goal, Plan, PlanStep
from agi.runtime import Runtime
from agi.skillmine import mine_skills
from tests.test_runtime import FakeAgent


def research_decomposer(goal: Goal) -> Plan:
    """A planner that decomposes a research request into three roles."""
    return Plan(
        rationale="Standard research workflow: plan → gather → synthesize.",
        steps=[
            PlanStep(id="plan", role="planner",
                     prompt=f"Outline a research approach for: {goal.intent}"),
            PlanStep(id="gather", role="researcher", depends_on=["plan"],
                     prompt=f"Gather sources relevant to: {goal.intent}"),
            PlanStep(id="synthesize", role="writer", depends_on=["gather"],
                     prompt=f"Synthesize a 3-bullet summary of: {goal.intent}"),
        ],
    )


def main() -> None:
    runtime = Runtime(agent_factory=FakeAgent)

    # Observe every coordination decision.
    runtime.subscribe(lambda e: print(f"[{e.kind}] {e.data}") if "coordinator" in e.kind or "task" in e.kind else None)

    coordinator = Coordinator(runtime, decomposer=research_decomposer)

    # Simulate three similar Goals — what a coordinator sees in steady state.
    pairs: list[tuple[str, str]] = []
    for topic in ("LoRA fine-tuning", "LoRA adapters in production", "LoRA quantization"):
        goal = Goal(intent=f"summarize the state of {topic}", budget_usd=1.0)
        result = coordinator.run(goal)
        print(f"\n→ goal={goal.intent!r} success={result.success} cost=${result.total_cost_usd:.4f}")
        for outcome in result.outcomes:
            pairs.append((outcome.step_id, outcome.result or ""))

    # After three successful runs, mine the pattern into skill candidates
    # the coordinator can show to a human for approval.
    candidates = mine_skills(pairs, min_cluster_size=2)
    print(f"\nmined {len(candidates)} skill candidate(s):")
    for c in candidates:
        print(f"  - {c.suggested_name}: {c.suggested_description}")

    print(f"\nruntime metrics: {runtime.metrics()}")


if __name__ == "__main__":
    main()
