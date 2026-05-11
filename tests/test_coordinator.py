"""Coordinator tests with FakeAgent runtime."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests._fakes import FakeAgent


def _patched_runtime():
    from agi import runtime as runtime_module
    patcher = patch.object(runtime_module, "Agent", FakeAgent)
    patcher.start()
    rt = runtime_module.Runtime()
    return rt, patcher


class TestRuleBasedDecomposition(unittest.TestCase):
    def setUp(self):
        from agi.coordinator import RuleBasedCoordinator
        self.rt, self._patcher = _patched_runtime()
        self.coord = RuleBasedCoordinator(self.rt)

    def tearDown(self):
        self._patcher.stop()

    def test_numbered_list(self):
        plan = self.coord.plan("Goal:\n1. fetch data\n2. summarize it\n3. write a report")
        self.assertEqual(len(plan), 3)
        self.assertIn("fetch data", plan[0].prompt)
        self.assertIn("write a report", plan[2].prompt)

    def test_bullet_list(self):
        plan = self.coord.plan("Goal:\n- step a\n- step b")
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0].prompt, "step a")

    def test_and_then(self):
        plan = self.coord.plan("read the file and then summarize it")
        self.assertEqual(len(plan), 2)

    def test_single_goal_passthrough(self):
        plan = self.coord.plan("just answer this question")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].prompt, "just answer this question")


class TestExecution(unittest.TestCase):
    def setUp(self):
        from agi.coordinator import RuleBasedCoordinator
        self.rt, self._patcher = _patched_runtime()
        self.coord = RuleBasedCoordinator(self.rt)

    def tearDown(self):
        self._patcher.stop()

    def test_execute_aggregates_results(self):
        observed = []
        result = self.coord.execute(
            "1. task one\n2. task two",
            on_event=lambda k, d: observed.append(k),
        )
        self.assertEqual(len(result.results), 2)
        self.assertIn("task one", result.summary)
        self.assertIn("task two", result.summary)
        self.assertGreater(result.total_cost_usd, 0)
        self.assertIn("subtask.started", observed)
        self.assertIn("subtask.finished", observed)

    def test_sessions_are_cleaned_up_after_execution(self):
        self.coord.execute("- a\n- b")
        # The coordinator destroys sub-sessions when done.
        self.assertEqual(len(self.rt.list_sessions()), 0)


class TestPlanParsing(unittest.TestCase):
    def test_valid_plan(self):
        from agi.coordinator import _parse_plan
        text = '{"subtasks": [{"id": "s0", "prompt": "do thing", "depends_on": []}]}'
        plan = _parse_plan(text)
        self.assertIsNotNone(plan)
        self.assertEqual(plan[0].prompt, "do thing")

    def test_plan_with_dependencies(self):
        from agi.coordinator import _parse_plan
        text = (
            '{"subtasks": ['
            '{"id":"s0","prompt":"first","depends_on":[]},'
            '{"id":"s1","prompt":"second","depends_on":["s0"]}'
            "]}"
        )
        plan = _parse_plan(text)
        self.assertEqual(plan[1].depends_on, ["s0"])

    def test_garbage_returns_none(self):
        from agi.coordinator import _parse_plan
        self.assertIsNone(_parse_plan("not json"))
        self.assertIsNone(_parse_plan(""))

    def test_extracts_object_from_wrapped_text(self):
        from agi.coordinator import _parse_plan
        text = 'Here is the plan: {"subtasks":[{"id":"x","prompt":"y"}]} thanks!'
        plan = _parse_plan(text)
        self.assertIsNotNone(plan)


if __name__ == "__main__":
    unittest.main()
