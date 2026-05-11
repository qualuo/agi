"""Tests for the task queue + runner."""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import Event
from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary
from agi.tasks import (
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_STARTED,
    Task,
    TaskQueue,
    TaskRunner,
    submit_task,
)
from tests.test_runtime import FakeAgent


def _make_runtime() -> Runtime:
    tmp = tempfile.mkdtemp()
    return Runtime(
        memory=Memory(path=Path(tmp) / "m.jsonl"),
        skills=SkillLibrary(path=Path(tmp) / "skills"),
        agent_factory=FakeAgent,
    )


class TestTaskQueue(unittest.TestCase):
    def test_submit_and_get(self):
        q = TaskQueue()
        tid = submit_task(q, prompt="hello")
        self.assertEqual(q.get(tid).prompt, "hello")

    def test_priority_ordering(self):
        q = TaskQueue()
        low = submit_task(q, prompt="low", priority=10)
        high = submit_task(q, prompt="high", priority=0)
        first = q.next_runnable()
        self.assertEqual(first.id, high)
        second = q.next_runnable()
        self.assertEqual(second.id, low)

    def test_deadline_skipped(self):
        q = TaskQueue()
        # past deadline
        submit_task(q, prompt="expired", deadline_ts=time.time() - 1)
        live = submit_task(q, prompt="live")
        chosen = q.next_runnable()
        self.assertEqual(chosen.id, live)
        # past-deadline task remains queued but is never selected
        self.assertIsNone(q.next_runnable())

    def test_cancel_running_task(self):
        q = TaskQueue()
        tid = submit_task(q, prompt="x")
        self.assertTrue(q.cancel(tid))
        self.assertEqual(q.get(tid).status, "cancelled")

    def test_filter_by_tag(self):
        q = TaskQueue()
        submit_task(q, prompt="a", tag="batch-1")
        submit_task(q, prompt="b", tag="batch-2")
        results = q.list(tag="batch-1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].prompt, "a")


class TestTaskRunner(unittest.TestCase):
    def test_runs_a_task_end_to_end(self):
        rt = _make_runtime()
        q = TaskQueue()
        events: list[Event] = []
        rt.subscribe(events.append)

        tid = submit_task(q, prompt="solve this", session_config=SessionConfig(use_skills=False))
        runner = TaskRunner(rt, q)
        count = runner.run_until_empty(max_ticks=5)

        self.assertEqual(count, 1)
        task = q.get(tid)
        self.assertEqual(task.status, "done")
        self.assertEqual(task.result, "ok")
        self.assertIsNotNone(task.session_id)

        kinds = [e.kind for e in events]
        self.assertIn(TASK_STARTED, kinds)
        self.assertIn(TASK_COMPLETED, kinds)

    def test_failed_task_marked_failed_when_max_attempts_reached(self):
        # Agent factory that always raises
        class ExplodingAgent:
            def __init__(self, **kw):
                self.usage = type("U", (), {
                    "input_tokens": 0, "output_tokens": 0,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                    "cost_usd": lambda self, m: 0.0,
                })()
                self.last_critic_score = None
                self.extra_system = None
                self.messages = []

            def chat(self, prompt, max_iterations=25):
                raise RuntimeError("boom")

            def attach_tool_synth(self, *a, **kw): pass
            def attach_delegation(self, *a, **kw): pass
            def reset(self): pass

        rt = Runtime(
            memory=Memory(path=Path(tempfile.mkdtemp()) / "m.jsonl"),
            skills=SkillLibrary(path=Path(tempfile.mkdtemp())),
            agent_factory=ExplodingAgent,
        )
        q = TaskQueue()
        events: list[Event] = []
        rt.subscribe(events.append, kind=TASK_FAILED)

        tid = submit_task(q, prompt="x", max_attempts=1, session_config=SessionConfig(use_skills=False))
        runner = TaskRunner(rt, q)
        runner.run_until_empty()
        task = q.get(tid)
        self.assertEqual(task.status, "failed")
        self.assertIn("boom", task.error)
        self.assertEqual(len(events), 1)

    def test_priority_runs_high_first(self):
        rt = _make_runtime()
        q = TaskQueue()
        order: list[str] = []
        rt.subscribe(
            lambda e: order.append(e.data.get("task_id", "")) if e.kind == TASK_STARTED else None,
            kind=TASK_STARTED,
        )

        low = submit_task(q, prompt="low", priority=10, session_config=SessionConfig(use_skills=False))
        high = submit_task(q, prompt="high", priority=0, session_config=SessionConfig(use_skills=False))

        runner = TaskRunner(rt, q)
        runner.run_until_empty()
        self.assertEqual(order, [high, low])


if __name__ == "__main__":
    unittest.main()
