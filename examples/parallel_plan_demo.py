"""Fan-out / fan-in plan execution against the AGI runtime.

A coordination engine — anything that authors plans — can hand the
runtime a DAG of work and watch live events as it runs. This demo
constructs a small research-style plan:

    research_a   research_b   research_c
              \\      |      /
               synthesize
                   |
                  review

`research_*` steps run in parallel up to the scheduler's
`max_concurrent_steps`. `synthesize` waits for all three. `review`
waits for `synthesize`. Live events stream to stdout.

Run with a fake agent (no API): `python examples/parallel_plan_demo.py`
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.coordinator import Plan, PlanStep
from agi.memory import Memory
from agi.runtime import Runtime
from agi.scheduler import (
    PLAN_COMPLETED,
    PLAN_STEP_COMPLETED,
    PLAN_STEP_READY,
    PLAN_STEP_RUNNING,
    ParallelScheduler,
    RetryPolicy,
    SchedulerConfig,
)
from agi.skills import SkillLibrary


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.turns = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    """Stand-in that simulates a working agent without API calls."""

    def __init__(self, **kw) -> None:
        self.memory = kw.get("memory")
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.extra_system: str | None = None

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        time.sleep(0.05)  # simulate work
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        # Pretend the role hint changes what we return.
        for hint in ("research", "synthesize", "review"):
            if hint in prompt:
                return f"[{hint}] summary line"
        return "ok"

    def attach_tool_synth(self, *a, **k): pass
    def attach_delegation(self, *a, **k): pass
    def reset(self): self.usage = FakeUsage()


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    runtime = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
    )

    def log(e):
        kind = e.kind.replace("plan.", "")
        sid = e.data.get("step_id", "")
        print(f"  [{kind:12s}] step={sid}  data={ {k: v for k, v in e.data.items() if k != 'execution_id'} }")

    for kind in (PLAN_STEP_READY, PLAN_STEP_RUNNING, PLAN_STEP_COMPLETED, PLAN_COMPLETED):
        runtime.subscribe(log, kind=kind)

    scheduler = ParallelScheduler(
        runtime,
        config=SchedulerConfig(
            max_concurrent_steps=3,
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.1),
        ),
    )

    plan = Plan(
        rationale="Research three angles in parallel, synthesize, then review.",
        steps=[
            PlanStep(id="research_a", prompt="research angle A", role="researcher"),
            PlanStep(id="research_b", prompt="research angle B", role="researcher"),
            PlanStep(id="research_c", prompt="research angle C", role="researcher"),
            PlanStep(
                id="synthesize",
                prompt="synthesize the three research outputs",
                role="synthesize",
                depends_on=["research_a", "research_b", "research_c"],
            ),
            PlanStep(
                id="review",
                prompt="review the synthesis",
                role="review",
                depends_on=["synthesize"],
            ),
        ],
    )

    print(f"Submitting plan with {len(plan.steps)} steps. "
          f"concurrency={scheduler.config.max_concurrent_steps}")
    started = time.time()
    result = scheduler.run(plan, budget_usd=10.0)
    elapsed = time.time() - started

    print()
    print(f"Plan status:        {result.status}")
    print(f"Wall-clock seconds: {elapsed:.3f}")
    print(f"Total cost:         ${result.cost_usd:.4f}")
    print(f"Per-step results:")
    for sid in (s.id for s in plan.steps):
        o = result.outcomes[sid]
        print(f"  - {sid:12s} {o.status:8s} cost=${o.cost_usd:.4f} duration={o.duration_seconds:.3f}s")


if __name__ == "__main__":
    main()
