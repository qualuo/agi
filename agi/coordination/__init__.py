"""Reference coordination engine.

This is the *other side* of the runtime: a small orchestrator that consumes
the runtime's HTTP API to plan, dispatch, and aggregate work. It exists for
two reasons:

1. To prove the runtime's protocol is self-sufficient — anyone can replace
   this coordinator with their own and the runtime still works.
2. To give investors a concrete demo: "feed it a goal, watch it decompose,
   dispatch, verify, and report."

The coordinator is intentionally not exotic. It's a planning loop:

    while not done:
        plan = runtime.plan(goal)
        result = runtime.execute(plan)
        if result.failed and budget_left:
            goal = revise(goal, result)
            continue
        return result
"""
from agi.coordination.coordinator import Coordinator, CoordinatorReport, RuntimeClient

__all__ = ["Coordinator", "CoordinatorReport", "RuntimeClient"]
