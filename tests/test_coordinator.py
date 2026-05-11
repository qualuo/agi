"""Tests for the reference Coordinator + Goal/Plan/Step."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.coordinator import (
    Coordinator,
    Goal,
    Plan,
    PlanStep,
    StepOutcome,
    label_aggregator,
    single_step_decomposer,
)
from agi.events import Event
from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent


def _make_runtime() -> Runtime:
    tmp = tempfile.mkdtemp()
    return Runtime(
        memory=Memory(path=Path(tmp) / "m.jsonl"),
        skills=SkillLibrary(path=Path(tmp) / "skills"),
        agent_factory=FakeAgent,
    )


class TestDefaultDecomposer(unittest.TestCase):
    def test_single_step_decomposer_produces_one_step(self):
        plan = single_step_decomposer(Goal(intent="do the thing"))
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].prompt, "do the thing")


class TestCoordinator(unittest.TestCase):
    def test_run_single_step_goal(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        events: list[Event] = []
        rt.subscribe(events.append, kind="coordinator.completed")

        result = coord.run(Goal(intent="hello world"))
        self.assertTrue(result.success)
        self.assertIn("ok", result.final_text)
        self.assertEqual(len(result.outcomes), 1)
        self.assertEqual(result.outcomes[0].status, "done")
        self.assertEqual(len(events), 1)

    def test_multi_step_plan_with_dependencies(self):
        def planner(goal: Goal) -> Plan:
            return Plan(steps=[
                PlanStep(id="a", prompt="step a", role="planner"),
                PlanStep(id="b", prompt="step b", role="executor", depends_on=["a"]),
                PlanStep(id="c", prompt="step c", role="critic", depends_on=["b"]),
            ])

        rt = _make_runtime()
        coord = Coordinator(rt, decomposer=planner)

        order: list[str] = []
        rt.subscribe(
            lambda e: order.append(e.data.get("task_id", "")) if e.kind == "task.started" else None,
            kind="task.started",
        )

        result = coord.run(Goal(intent="multi-step"))
        self.assertTrue(result.success)
        self.assertEqual(len(result.outcomes), 3)
        # Order in outcomes should respect deps
        outcome_ids = [o.step_id for o in result.outcomes]
        self.assertEqual(outcome_ids, ["a", "b", "c"])

    def test_aggregator_concatenates_steps(self):
        def planner(_goal: Goal) -> Plan:
            return Plan(steps=[
                PlanStep(id="s1", prompt="first", role="r"),
                PlanStep(id="s2", prompt="second", role="r"),
            ])

        rt = _make_runtime()
        coord = Coordinator(rt, decomposer=planner)
        result = coord.run(Goal(intent="combine"))
        self.assertIn("Step s1", result.final_text)
        self.assertIn("Step s2", result.final_text)

    def test_budget_stops_further_dispatch(self):
        def planner(_goal: Goal) -> Plan:
            return Plan(steps=[
                PlanStep(id=str(i), prompt=f"p{i}", role="r") for i in range(5)
            ])

        rt = _make_runtime()
        coord = Coordinator(rt, decomposer=planner)
        # Each FakeAgent chat costs ~$0.0018 (100 in + 50 out at fake rates).
        # Budget should cut us off after ~1 step.
        events: list[Event] = []
        rt.subscribe(events.append, kind="coordinator.budget_exhausted")
        result = coord.run(Goal(intent="x", budget_usd=0.001))

        # Either no steps ran (budget hit immediately) or 1 step ran then
        # the budget cut in. Both are acceptable; the invariant is that
        # we did NOT run all 5.
        self.assertLess(len(result.outcomes), 5)
        if len(result.outcomes) >= 1:
            self.assertEqual(len(events), 1)

    def test_acceptance_callback_runs(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        result_yes = coord.run(Goal(intent="x", acceptance=lambda t: "ok" in t))
        self.assertTrue(result_yes.success)
        result_no = coord.run(Goal(intent="x", acceptance=lambda t: "NOPE" in t))
        self.assertFalse(result_no.success)

    def test_cyclic_or_missing_dep_stops_safely(self):
        def planner(_goal: Goal) -> Plan:
            return Plan(steps=[
                PlanStep(id="a", prompt="a", role="r", depends_on=["b"]),
                PlanStep(id="b", prompt="b", role="r", depends_on=["a"]),
            ])

        rt = _make_runtime()
        coord = Coordinator(rt, decomposer=planner)
        result = coord.run(Goal(intent="cycle"))
        # Neither step ran because each depends on the other
        self.assertEqual(len(result.outcomes), 0)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
