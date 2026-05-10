"""Tests for the Coordinator.

Reuses the StubAgent from test_runtime so the DAG executor can be tested
without API calls. We feed a recording stub: it captures the prompts it
receives so we can assert that upstream outputs were rendered into
downstream prompts.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.coordinator import Coordinator, CoordinatorError, Node
from agi.runtime import Runtime, SessionRecord
from tests.test_runtime import StubAgent


class _RecordingFactory:
    """Per-call factory: each session gets a fresh StubAgent that returns
    a value keyed off the session's role."""

    def __init__(self, responses: dict[str, str], sleep_s: float = 0.0) -> None:
        self.responses = responses
        self.sleep_s = sleep_s
        self.received_prompts: dict[str, str] = {}

    def __call__(self, session: SessionRecord):
        role = session.role or "default"
        response = self.responses.get(role, "ok")
        agent = _CapturingStub(response, self.sleep_s, self.received_prompts, role)
        return agent


class _CapturingStub(StubAgent):
    def __init__(self, response, sleep_s, sink, role):
        super().__init__(response=response, sleep_s=sleep_s)
        self._sink = sink
        self._role = role

    def chat_controlled(self, prompt, **kw):
        # Last write wins per role; the coordinator uses one session per node
        # so this is unique per node.
        self._sink[self._role] = prompt
        return super().chat_controlled(prompt, **kw)


class TestCoordinatorValidation(unittest.TestCase):
    def test_unknown_dependency_raises(self):
        rt = Runtime(agent_factory=_RecordingFactory({}))
        try:
            with self.assertRaises(CoordinatorError):
                Coordinator(rt, [Node("a", "p", depends_on=["nope"])])
        finally:
            rt.shutdown()

    def test_cycle_raises(self):
        rt = Runtime(agent_factory=_RecordingFactory({}))
        try:
            with self.assertRaises(CoordinatorError):
                Coordinator(rt, [
                    Node("a", "p", depends_on=["b"]),
                    Node("b", "p", depends_on=["a"]),
                ])
        finally:
            rt.shutdown()

    def test_duplicate_node_raises(self):
        rt = Runtime(agent_factory=_RecordingFactory({}))
        try:
            with self.assertRaises(CoordinatorError):
                Coordinator(rt, [Node("a", "p"), Node("a", "p")])
        finally:
            rt.shutdown()


class TestCoordinatorExecution(unittest.TestCase):
    def test_simple_dag_runs_in_order(self):
        factory = _RecordingFactory({
            "researcher": "fact: water boils at 100C",
            "writer":     "drafted summary",
        })
        rt = Runtime(agent_factory=factory)
        try:
            plan = Coordinator(rt, [
                Node("research", "Find a fact.", role="researcher"),
                Node("write", "Use this: {research}", depends_on=["research"], role="writer"),
            ])
            results = plan.run(timeout=10)
            self.assertEqual(results["research"].status, "succeeded")
            self.assertEqual(results["write"].status, "succeeded")
            # Downstream prompt was rendered with upstream output
            self.assertIn("water boils", factory.received_prompts["writer"])
        finally:
            rt.shutdown()

    def test_parallel_independent_nodes(self):
        # Three independent nodes should run concurrently
        factory = _RecordingFactory({"a": "A", "b": "B", "c": "C"}, sleep_s=0.4)
        rt = Runtime(agent_factory=factory, max_workers=4)
        try:
            plan = Coordinator(rt, [
                Node("a", "p", role="a"),
                Node("b", "p", role="b"),
                Node("c", "p", role="c"),
            ])
            t0 = time.time()
            results = plan.run(timeout=10)
            elapsed = time.time() - t0
            self.assertLess(elapsed, 1.0, f"DAG ran serially in {elapsed:.2f}s")
            self.assertTrue(all(r.status == "succeeded" for r in results.values()))
        finally:
            rt.shutdown()

    def test_upstream_failure_short_circuits_downstream(self):
        from agi.runtime import SessionRecord

        def factory(session: SessionRecord):
            if session.role == "fail":
                return StubAgent(raise_in_chat=RuntimeError("nope"))
            return StubAgent(response="ok")

        rt = Runtime(agent_factory=factory)
        try:
            plan = Coordinator(rt, [
                Node("upstream", "p", role="fail"),
                Node("downstream", "needs {upstream}", depends_on=["upstream"]),
            ])
            results = plan.run(timeout=10)
            self.assertEqual(results["upstream"].status, "failed")
            self.assertEqual(results["downstream"].status, "failed")
            self.assertIn("upstream", results["downstream"].error or "")
        finally:
            rt.shutdown()


if __name__ == "__main__":
    unittest.main()
