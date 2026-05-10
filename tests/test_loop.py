"""Tests for the autonomous goal-loop primitive (no API calls)."""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import StepResult
from agi.costs import Usage
from agi.loop import GoalBudget, GoalRunResult, is_done, run_goal


def _step(text: str, *, error: str | None = None) -> StepResult:
    return StepResult(
        text=text,
        usage=Usage(),
        tool_calls=[],
        iterations=1,
        stop_reason="end_turn",
        duration_seconds=0.0,
        critic_score=None,
        error=error,
    )


class FakeSession:
    """Minimal Session-shaped stand-in for loop tests."""

    def __init__(self, scripted_responses: list[StepResult], cost_per_step: float = 0.05):
        self._responses = list(scripted_responses)
        self._cost_per_step = cost_per_step
        self.calls: list[str] = []
        # Mock agent surface used by run_goal
        self.agent = MagicMock()
        self.agent.model = "claude-opus-4-7"
        self._cumulative_cost = 0.0
        self.agent.usage = MagicMock()
        self.agent.usage.cost_usd = lambda model: self._cumulative_cost

    def step(self, user_input: str, max_iterations: int = 25) -> StepResult:
        self.calls.append(user_input)
        if not self._responses:
            return _step("(no more scripted responses) GOAL_COMPLETE")
        self._cumulative_cost += self._cost_per_step
        return self._responses.pop(0)


class TestIsDone(unittest.TestCase):
    def test_recognizes_done_marker(self):
        self.assertTrue(is_done("doing the thing\nGOAL_COMPLETE"))
        self.assertTrue(is_done("GOAL_COMPLETE"))
        self.assertTrue(is_done("...DONE.\n"))
        self.assertTrue(is_done("TASK_COMPLETE"))

    def test_rejects_partial(self):
        self.assertFalse(is_done("almost done"))
        self.assertFalse(is_done("complete the task next"))
        self.assertFalse(is_done(""))


class TestRunGoal(unittest.TestCase):
    def test_completes_on_done_marker(self):
        sess = FakeSession([
            _step("step 1 in progress"),
            _step("step 2 done\nGOAL_COMPLETE"),
        ])
        result = run_goal(sess, "do the thing", budget=GoalBudget(max_steps=8))
        self.assertTrue(result.completed)
        self.assertEqual(result.stop_reason, "done")
        self.assertEqual(len(result.steps), 2)
        # Kickoff message includes the goal verbatim
        self.assertIn("do the thing", sess.calls[0])

    def test_stops_at_max_steps(self):
        sess = FakeSession([_step(f"step {i}") for i in range(10)], cost_per_step=0.0)
        result = run_goal(sess, "endless task", budget=GoalBudget(max_steps=3, max_cost_usd=10))
        self.assertFalse(result.completed)
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(len(result.steps), 3)

    def test_stops_at_max_cost(self):
        sess = FakeSession([_step(f"step {i}") for i in range(10)], cost_per_step=0.30)
        result = run_goal(sess, "expensive task", budget=GoalBudget(max_steps=10, max_cost_usd=0.50))
        # First step costs 0.30 (< 0.50, continue), second costs another 0.30
        # making cumulative 0.60 (>= 0.50, stop).
        self.assertEqual(result.stop_reason, "max_cost")
        self.assertEqual(len(result.steps), 2)

    def test_predicate_stop(self):
        sess = FakeSession([_step("a"), _step("b"), _step("c")])
        def stop(s, step):
            return "b" in step.text
        result = run_goal(sess, "x", budget=GoalBudget(max_steps=8), stop=stop)
        self.assertEqual(result.stop_reason, "predicate")
        self.assertEqual(len(result.steps), 2)

    def test_step_error_stops_loop(self):
        sess = FakeSession([_step("step 1"), _step("boom", error="BOOM: bad things")])
        result = run_goal(sess, "x")
        self.assertEqual(result.stop_reason, "error")
        self.assertEqual(result.error, "BOOM: bad things")

    def test_continue_message_includes_budget_hint(self):
        sess = FakeSession([_step("step 1"), _step("step 2\nGOAL_COMPLETE")])
        run_goal(sess, "x", budget=GoalBudget(max_steps=8, max_cost_usd=2.0))
        # The second user message should include the budget hint
        self.assertIn("remaining", sess.calls[1])

    def test_summary_shape(self):
        sess = FakeSession([_step("done\nGOAL_COMPLETE")])
        result = run_goal(sess, "trivial")
        s = result.summary()
        for k in ("goal", "completed", "stop_reason", "steps", "cost_usd", "final_text"):
            self.assertIn(k, s)


if __name__ == "__main__":
    unittest.main()
