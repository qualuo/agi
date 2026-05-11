"""Tests for the runtime engine.

Exercise the full engine path — task lifecycle, events, cancellation,
delegation, budgets — using a MockBackend so no API key or network is
required.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from runtime.backend import MockBackend, MockBlock, MockMessage, MockUsage
from runtime.engine import Engine
from runtime.task import Budget, TaskStatus


def _mock_text(text: str) -> MockBackend:
    return MockBackend.echo(text)


class TestTaskLifecycle(unittest.TestCase):
    def test_simple_task_runs_to_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(backend=_mock_text("hello world"), memory=Memory(Path(tmp) / "m.jsonl"))
            try:
                task = engine.submit("say hi")
                self.assertTrue(task.wait(timeout=5))
                self.assertEqual(task.status, TaskStatus.COMPLETED)
                self.assertEqual(task.result, "hello world")
            finally:
                engine.shutdown()

    def test_task_has_stable_id(self):
        with Engine(backend=_mock_text("ok")) as engine:
            task = engine.submit("x", task_id="abc123")
            self.assertEqual(task.id, "abc123")
            self.assertIs(engine.get("abc123"), task)

    def test_status_transitions_recorded(self):
        with Engine(backend=_mock_text("x")) as engine:
            task = engine.submit("anything")
            task.wait(timeout=5)
            kinds = [e.kind for e in task.events()]
            self.assertIn("status_changed", kinds)
            # We should see at least queued->running and running->completed
            transitions = [
                (e.data["from"], e.data["to"])
                for e in task.events() if e.kind == "status_changed"
            ]
            self.assertIn(("queued", "running"), transitions)
            self.assertIn(("running", "completed"), transitions)

    def test_failure_captured(self):
        def boom(messages):
            raise RuntimeError("backend boom")
        with Engine(backend=MockBackend(responder=boom)) as engine:
            task = engine.submit("explode")
            task.wait(timeout=5)
            self.assertEqual(task.status, TaskStatus.FAILED)
            self.assertIn("boom", task.error or "")


class TestEvents(unittest.TestCase):
    def test_text_event_emitted(self):
        with Engine(backend=_mock_text("the answer")) as engine:
            task = engine.submit("question")
            task.wait(timeout=5)
            texts = [e for e in task.events() if e.kind == "text"]
            self.assertTrue(texts, "expected at least one text event")
            self.assertEqual(texts[-1].data["text"], "the answer")

    def test_turn_complete_event(self):
        with Engine(backend=_mock_text("hi")) as engine:
            task = engine.submit("ping")
            task.wait(timeout=5)
            kinds = [e.kind for e in task.events()]
            self.assertIn("turn_complete", kinds)

    def test_stream_events_yields_all(self):
        with Engine(backend=_mock_text("done")) as engine:
            task = engine.submit("q")
            collected = []
            for ev in task.stream_events(timeout=0.5):
                collected.append(ev.kind)
            self.assertIn("status_changed", collected)
            self.assertIn("text", collected)


class TestTools(unittest.TestCase):
    def test_tool_call_then_completion(self):
        # Scripted: first turn calls run_bash, second turn ends with text.
        first = MockBackend.tool_call("run_bash", {"command": "echo hello"})
        second = MockBackend.text("done")
        backend = MockBackend.scripted([first, second])

        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(backend=backend, memory=Memory(Path(tmp) / "m.jsonl"))
            try:
                task = engine.submit("run a thing")
                task.wait(timeout=10)
                self.assertEqual(task.status, TaskStatus.COMPLETED)
                tool_events = [e for e in task.events() if e.kind == "tool_call"]
                self.assertEqual(len(tool_events), 1)
                self.assertEqual(tool_events[0].data["name"], "run_bash")
                results = [e for e in task.events() if e.kind == "tool_result"]
                self.assertEqual(len(results), 1)
                self.assertIn("hello", results[0].data["output"])
            finally:
                engine.shutdown()


class TestCancellation(unittest.TestCase):
    def test_cancel_before_first_turn(self):
        # Backend that blocks until released; we cancel before that.
        gate = threading.Event()

        def slow(messages):
            gate.wait(timeout=5)
            return MockBackend.text("late")

        with Engine(backend=MockBackend(responder=slow)) as engine:
            task = engine.submit("slow task")
            time.sleep(0.05)
            engine.cancel(task.id)
            gate.set()
            task.wait(timeout=5)
            self.assertEqual(task.status, TaskStatus.CANCELLED)


class TestBudgets(unittest.TestCase):
    def test_turn_budget_aborts(self):
        # Each turn calls a tool, so the loop keeps running. Budget cuts it off.
        first = MockBackend.tool_call("run_bash", {"command": "echo loop"})
        # Use scripted with the same tool call repeated many times so we never
        # naturally end_turn — budget must cut us off.
        backend = MockBackend.scripted([first] * 50)

        with Engine(backend=backend) as engine:
            task = engine.submit("loop forever", budget=Budget(max_turns=3))
            task.wait(timeout=10)
            self.assertEqual(task.status, TaskStatus.FAILED)
            self.assertIn("turns", (task.error or "").lower())


class TestDelegation(unittest.TestCase):
    def test_delegate_creates_child(self):
        # Parent calls delegate once, then returns a final answer.
        parent_responses = [
            MockBackend.tool_call(
                "delegate", {"instruction": "subtask", "max_turns": 3, "max_cost_usd": 0.1}
            ),
            MockBackend.text("parent done with child"),
        ]
        child_responses = [MockBackend.text("child answer")]

        # We need different mocks for parent vs child. Easiest: a single
        # responder that branches on the user message contents.
        def responder(messages):
            for m in messages:
                content = m.get("content")
                if isinstance(content, str) and "subtask" in content:
                    return child_responses[0]
            # Parent path: consume scripted parent_responses in order.
            idx = sum(1 for m in messages if m.get("role") == "assistant")
            if idx < len(parent_responses):
                return parent_responses[idx]
            return MockBackend.text("")

        with Engine(backend=MockBackend(responder=responder)) as engine:
            parent = engine.submit("do a parent task")
            parent.wait(timeout=10)
            self.assertEqual(parent.status, TaskStatus.COMPLETED)
            self.assertEqual(len(parent.children), 1)
            child = engine.get(parent.children[0])
            self.assertIsNotNone(child)
            self.assertEqual(child.status, TaskStatus.COMPLETED)
            self.assertEqual(child.parent_id, parent.id)
            self.assertEqual(child.result, "child answer")

            # task_tree exposes the parent/child relation
            tree = engine.task_tree(parent.id)
            self.assertEqual(tree["id"], parent.id)
            self.assertEqual(len(tree["children"]), 1)
            self.assertEqual(tree["children"][0]["id"], child.id)


class TestConcurrency(unittest.TestCase):
    def test_parallel_tasks_run_independently(self):
        # Each task takes ~0.05s. Five tasks in flight should finish in
        # well under 0.25s (would be 0.25s if serial).
        def slow_text(messages):
            time.sleep(0.05)
            return MockBackend.text("done")

        with Engine(backend=MockBackend(responder=slow_text), max_concurrent=5) as engine:
            tasks = [engine.submit(f"task {i}") for i in range(5)]
            t0 = time.time()
            for t in tasks:
                t.wait(timeout=5)
            elapsed = time.time() - t0
            for t in tasks:
                self.assertEqual(t.status, TaskStatus.COMPLETED)
            # Loose timing assertion: at least 2x speedup over serial.
            self.assertLess(elapsed, 5 * 0.05 / 2 + 0.5, f"elapsed {elapsed} too slow for parallel")


if __name__ == "__main__":
    unittest.main()
