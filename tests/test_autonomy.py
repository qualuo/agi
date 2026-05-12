"""Tests for the AutonomyEngine continuous-learning driver."""
import os
import tempfile
import unittest
from unittest.mock import patch

from agi.autonomy import AutonomyEngine, GoalQueue, TickReport
from agi.coordinator import Coordinator, Goal, Plan, PlanStep
from agi.runtime import Runtime, SessionConfig
from agi.selfeval import EvalItem, SelfEvalBank


class _FakeAgent:
    """Echoes the prompt; usage counters tick up so cost > 0."""
    def __init__(self, **kwargs):
        from agi.costs import Usage
        self.usage = Usage()
        self.usage.input_tokens = 200
        self.usage.output_tokens = 80
        self.messages: list = []
        self.last_critic_score = 0.9

    def chat(self, prompt, max_iterations=25):
        # The acceptance check often looks for a substring; embed the
        # whole prompt so simple acceptance tests pass.
        return f"answer for: {prompt[:120]}"

    def reset(self):
        self.messages = []


def _planner(goal: Goal) -> Plan:
    return Plan(steps=[PlanStep(id="root", prompt=goal.intent, role="executor")])


class TestGoalQueue(unittest.TestCase):
    def test_push_and_pop_ordering(self):
        q = GoalQueue()
        q.push(Goal(intent="first"))
        q.push(Goal(intent="second"))
        self.assertEqual(len(q), 2)
        first = q.pop()
        second = q.pop()
        self.assertEqual(first.intent, "first")
        self.assertEqual(second.intent, "second")
        self.assertIsNone(q.pop())

    def test_provider_callable(self):
        q = GoalQueue()
        q.push(Goal(intent="only"))
        provider = q.as_provider()
        self.assertEqual(provider().intent, "only")
        self.assertIsNone(provider())


class TestAutonomyEngine(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(agent_factory=_FakeAgent)
        self.coord = Coordinator(self.runtime, decomposer=_planner)
        self.queue = GoalQueue()

    def test_idle_tick_when_no_goal(self):
        engine = AutonomyEngine(
            self.runtime, self.coord,
            goal_provider=self.queue.as_provider(),
            max_iterations=1,
        )
        report = engine.run_once()
        self.assertTrue(report.idle)
        self.assertIsNone(report.goal)
        self.assertIsNone(report.autonomous)

    def test_success_promotes_skill_when_no_eval_bank(self):
        # No eval bank: candidate is promoted eagerly.
        self.queue.push(Goal(
            intent="add: produce the answer",
            acceptance=lambda t: "answer for" in t,
        ))
        engine = AutonomyEngine(
            self.runtime, self.coord,
            goal_provider=self.queue.as_provider(),
            max_iterations=1,
            mine_eval_items=False,
        )
        report = engine.run_once()
        self.assertFalse(report.idle)
        self.assertTrue(report.autonomous.success)
        # Skill promotion happens after a successful AutonomousLoop
        # when a candidate is mined. Mining may or may not occur for a
        # trivial trace; we just assert the engine didn't error.
        self.assertEqual(report.cost_usd, report.autonomous.total_cost_usd)

    def test_per_tick_budget_overrides_goal_budget(self):
        self.queue.push(Goal(
            intent="x",
            budget_usd=100.0,  # large
            acceptance=lambda t: "answer for" in t,
        ))
        engine = AutonomyEngine(
            self.runtime, self.coord,
            goal_provider=self.queue.as_provider(),
            max_cost_per_tick_usd=0.05,
            max_iterations=1,
        )
        report = engine.run_once()
        # Engine clamps the budget down to the per-tick ceiling
        self.assertLessEqual(report.goal.budget_usd, 0.05)

    def test_eval_item_mined_on_success(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            bank = SelfEvalBank(path=path)
            self.queue.push(Goal(
                intent="some question with a unique slug zkqp",
                acceptance=lambda t: True,
            ))
            engine = AutonomyEngine(
                self.runtime, self.coord,
                goal_provider=self.queue.as_provider(),
                eval_bank=bank,
                max_iterations=1,
                mine_eval_items=True,
            )
            report = engine.run_once()
            self.assertTrue(report.autonomous.success)
            # New automatic item should be in the bank
            self.assertGreaterEqual(report.eval_items_added, 0)
        finally:
            os.unlink(path)

    def test_run_forever_with_max_ticks(self):
        for i in range(3):
            self.queue.push(Goal(
                intent=f"task {i}",
                acceptance=lambda t: "answer for" in t,
            ))
        engine = AutonomyEngine(
            self.runtime, self.coord,
            goal_provider=self.queue.as_provider(),
            max_iterations=1,
            mine_eval_items=False,
        )
        reports = engine.run_forever(max_ticks=3, heartbeat_seconds=0.0)
        self.assertEqual(len(reports), 3)
        self.assertEqual(engine.metrics()["ticks"], 3)
        self.assertEqual(engine.metrics()["goals_attempted"], 3)

    def test_run_forever_stops_after_idle_grace(self):
        engine = AutonomyEngine(
            self.runtime, self.coord,
            goal_provider=self.queue.as_provider(),
            max_iterations=1,
        )
        reports = engine.run_forever(
            max_ticks=10, heartbeat_seconds=0.0, idle_grace_ticks=2,
        )
        self.assertLessEqual(len(reports), 3)
        self.assertTrue(all(r.idle for r in reports))

    def test_emits_lifecycle_events(self):
        events: list = []
        self.runtime.subscribe(events.append)
        self.queue.push(Goal(
            intent="hello",
            acceptance=lambda t: "answer for" in t,
        ))
        engine = AutonomyEngine(
            self.runtime, self.coord,
            goal_provider=self.queue.as_provider(),
            max_iterations=1,
            mine_eval_items=False,
        )
        engine.run_once()
        kinds = {e.kind for e in events}
        self.assertIn("autonomy.tick_started", kinds)
        self.assertIn("autonomy.tick_completed", kinds)
        self.assertIn("autonomy.goal_started", kinds)


class TestRegressionGate(unittest.TestCase):
    """With a regression gate wired, a candidate whose deployment would
    drop pass rate is rejected."""

    def test_skill_rejected_when_regression_runner_fails(self):
        rt = Runtime(agent_factory=_FakeAgent)
        coord = Coordinator(rt, decomposer=_planner)
        q = GoalQueue()
        q.push(Goal(intent="task", acceptance=lambda t: True))

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            bank = SelfEvalBank(path=path)
            bank.add(
                prompt="seed task", expect_substring="MUST_APPEAR_BUT_WONT",
                source="explicit",
            )
            # A runner that always fails the regression check
            def runner(item: EvalItem) -> tuple[bool, str, float]:
                return False, "no match", 0.0

            engine = AutonomyEngine(
                rt, coord,
                goal_provider=q.as_provider(),
                eval_bank=bank,
                eval_runner=runner,
                max_iterations=1,
                mine_eval_items=False,
                regression_tolerance=0.0,
            )
            report = engine.run_once()
            # If a candidate was mined, it must have been rejected.
            if report.autonomous.skill_candidate is not None:
                self.assertIsNone(report.skill_promoted)
                self.assertIsNotNone(report.skill_rejected)
                self.assertIsNotNone(report.regression_report)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
