"""Tests for agi.metrics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import (
    BUDGET_EXCEEDED,
    CRITIC_SCORED,
    EventBus,
    SESSION_CLOSED,
    SESSION_CREATED,
    TOOL_ERRORED,
    TOOL_INVOKED,
    TURN_COMPLETED,
    TURN_ERRORED,
    TURN_STARTED,
)
from agi.metrics import RuntimeMetrics
from agi.runtime import Budget, Runtime, Session
from tests.test_runtime import FakeAgent


class TestRuntimeMetricsAggregator(unittest.TestCase):
    def test_aggregates_session_lifecycle(self):
        m = RuntimeMetrics()
        bus = EventBus()
        m.attach(bus, role="executor")
        bus.emit(SESSION_CREATED, {})
        bus.emit(TURN_STARTED, {"prompt": "x"})
        bus.emit(TOOL_INVOKED, {"name": "read_file"})
        bus.emit(TOOL_INVOKED, {"name": "read_file"})
        bus.emit(TOOL_INVOKED, {"name": "run_bash"})
        bus.emit(TOOL_ERRORED, {"name": "run_bash", "error": "boom"})
        bus.emit(CRITIC_SCORED, {"score": 0.8})
        bus.emit(TURN_COMPLETED, {"text": "ok", "cost_usd": 0.0123, "input_tokens": 100, "output_tokens": 50})
        bus.emit(SESSION_CLOSED, {})

        d = m.to_dict()
        self.assertEqual(d["sessions"]["created"], 1)
        self.assertEqual(d["sessions"]["closed"], 1)
        self.assertEqual(d["sessions"]["active"], 0)
        self.assertEqual(d["turns"]["started"], 1)
        self.assertEqual(d["turns"]["completed"], 1)
        self.assertAlmostEqual(d["cost"]["usd"], 0.0123, places=4)
        self.assertEqual(d["cost"]["input_tokens"], 100)
        self.assertEqual(d["tool_invocations"]["read_file"], 2)
        self.assertEqual(d["tool_invocations"]["run_bash"], 1)
        self.assertEqual(d["tool_errors"]["run_bash"], 1)
        self.assertAlmostEqual(d["critic"]["average_score"], 0.8)
        self.assertIn("executor", d["by_role"])
        self.assertEqual(d["by_role"]["executor"]["turns_completed"], 1)

    def test_budget_exceeded_counted(self):
        m = RuntimeMetrics()
        bus = EventBus()
        m.attach(bus)
        bus.emit(BUDGET_EXCEEDED, {"reason": "max_usd"})
        bus.emit(BUDGET_EXCEEDED, {"reason": "max_turns"})
        d = m.to_dict()
        self.assertEqual(d["budgets_exceeded"], 2)

    def test_turn_errored_counted(self):
        m = RuntimeMetrics()
        bus = EventBus()
        m.attach(bus, role="researcher")
        bus.emit(TURN_STARTED, {})
        bus.emit(TURN_ERRORED, {"error": "boom"})
        d = m.to_dict()
        self.assertEqual(d["turns"]["errored"], 1)
        self.assertEqual(d["by_role"]["researcher"]["turns_errored"], 1)


class TestRuntimeMetricsIntegration(unittest.TestCase):
    """Verify Runtime threads metrics correctly through Session lifecycle."""

    def test_runtime_records_session_creation_and_steps(self):
        rt = Runtime(agent_factory=FakeAgent)
        s1 = rt.create_session(role="planner")
        s2 = rt.create_session(role="executor")
        s1.step("task a")
        s2.step("task b")
        s2.step("task c")
        d = rt.metrics.to_dict()
        self.assertEqual(d["sessions"]["created"], 2)
        self.assertEqual(d["sessions"]["active"], 2)
        self.assertEqual(d["turns"]["completed"], 3)
        self.assertEqual(d["by_role"]["planner"]["turns_completed"], 1)
        self.assertEqual(d["by_role"]["executor"]["turns_completed"], 2)

        rt.close(s1.id)
        d2 = rt.metrics.to_dict()
        self.assertEqual(d2["sessions"]["active"], 1)
        self.assertEqual(d2["sessions"]["closed"], 1)


if __name__ == "__main__":
    unittest.main()
