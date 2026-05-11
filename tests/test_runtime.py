"""Runtime tests — no API calls.

Uses a fake agent factory so the runtime is exercised end-to-end without
touching Anthropic. The fake agent supports cancellation, simulated cost
ceilings, and raising arbitrary errors so all paths are covered.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.runtime import (
    BudgetExceeded,
    Cancelled,
    Run,
    Runtime,
    RunStatus,
)


class _FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 100
        self.output_tokens = 200
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return 0.0025  # arbitrary


class FakeAgent:
    """Minimal agent shape the Runtime needs: .chat() + .usage + .model."""

    def __init__(
        self,
        run: Run,
        *,
        delay: float = 0.0,
        raise_exc: Exception | None = None,
        respect_cancel: bool = True,
        output: str = "done",
    ) -> None:
        self.run = run
        self.delay = delay
        self.raise_exc = raise_exc
        self.respect_cancel = respect_cancel
        self.output = output
        self.usage = _FakeUsage()
        self.model = "claude-opus-4-7"

    def chat(self, task: str) -> str:
        deadline = time.monotonic() + self.delay
        while time.monotonic() < deadline:
            if self.respect_cancel and self.run._cancel.is_set():
                raise Cancelled("cancelled")
            time.sleep(0.01)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.output


def _factory(**fake_kwargs):
    def f(run: Run, runtime: Runtime) -> FakeAgent:
        return FakeAgent(run, **fake_kwargs)

    return f


class TestRuntimeSuccess(unittest.TestCase):
    def test_run_succeeds_and_reports_usage(self):
        rt = Runtime(agent_factory=_factory(output="hi"))
        run = rt.submit("say hi")
        run.wait(timeout=2.0)
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertEqual(run.result, "hi")
        self.assertEqual(run.input_tokens, 100)
        self.assertEqual(run.output_tokens, 200)
        self.assertGreater(run.cost_usd, 0)
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.ended_at)

    def test_get_and_list(self):
        rt = Runtime(agent_factory=_factory())
        r1 = rt.submit("a")
        r2 = rt.submit("b")
        for r in (r1, r2):
            r.wait(timeout=2.0)
        all_runs = {r.id for r in rt.list_runs()}
        self.assertEqual(all_runs, {r1.id, r2.id})
        self.assertIs(rt.get(r1.id), r1)
        self.assertIsNone(rt.get("nonexistent"))

    def test_to_public_dict_is_json_safe(self):
        import json

        rt = Runtime(agent_factory=_factory())
        run = rt.submit("x")
        run.wait(timeout=2.0)
        # Round-trips cleanly.
        round_tripped = json.loads(json.dumps(run.to_public_dict()))
        self.assertEqual(round_tripped["id"], run.id)
        self.assertEqual(round_tripped["status"], "succeeded")


class TestRuntimeCancellation(unittest.TestCase):
    def test_cancel_running_task(self):
        rt = Runtime(agent_factory=_factory(delay=5.0))
        run = rt.submit("long task")
        # Give the thread a moment to start
        time.sleep(0.05)
        self.assertEqual(run.status, RunStatus.RUNNING)
        self.assertTrue(rt.cancel(run.id))
        run.wait(timeout=2.0)
        self.assertEqual(run.status, RunStatus.CANCELLED)

    def test_cancel_unknown_run(self):
        rt = Runtime(agent_factory=_factory())
        self.assertFalse(rt.cancel("not-a-real-id"))

    def test_cancel_after_terminal_is_noop(self):
        rt = Runtime(agent_factory=_factory())
        run = rt.submit("fast")
        run.wait(timeout=2.0)
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertFalse(run.cancel())


class TestRuntimeFailure(unittest.TestCase):
    def test_unexpected_exception_marks_failed(self):
        rt = Runtime(agent_factory=_factory(raise_exc=RuntimeError("boom")))
        run = rt.submit("x")
        run.wait(timeout=2.0)
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertIn("boom", run.error)

    def test_budget_exceeded_status(self):
        rt = Runtime(agent_factory=_factory(raise_exc=BudgetExceeded("over")))
        run = rt.submit("x", cost_ceiling_usd=0.01)
        run.wait(timeout=2.0)
        self.assertEqual(run.status, RunStatus.BUDGET_EXCEEDED)
        self.assertIn("over", run.error)


class TestRuntimeTimeout(unittest.TestCase):
    def test_timeout_marks_timed_out(self):
        rt = Runtime(agent_factory=_factory(delay=5.0))
        run = rt.submit("long", timeout_seconds=0.1)
        run.wait(timeout=2.0)
        self.assertEqual(run.status, RunStatus.TIMED_OUT)


class TestEventBus(unittest.TestCase):
    def test_history_replay_for_late_subscriber(self):
        rt = Runtime(agent_factory=_factory(output="hi"))
        run = rt.submit("x")
        run.wait(timeout=2.0)
        events = list(run.stream(replay=True))
        types = [e.type for e in events]
        self.assertIn("run.started", types)
        self.assertIn("run.succeeded", types)

    def test_live_subscriber_receives_events_during_run(self):
        rt = Runtime(agent_factory=_factory(delay=0.2, output="done"))
        run = rt.submit("x")
        seen: list[str] = []
        finished = threading.Event()

        def consumer():
            for ev in run.stream(replay=True):
                seen.append(ev.type)
            finished.set()

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        run.wait(timeout=2.0)
        finished.wait(timeout=2.0)
        self.assertIn("run.started", seen)
        self.assertIn("run.succeeded", seen)


class TestRunSerialization(unittest.TestCase):
    def test_to_public_dict_excludes_internals(self):
        rt = Runtime(agent_factory=_factory())
        run = rt.submit("x")
        run.wait(timeout=2.0)
        d = run.to_public_dict()
        # Internal threading primitives must not leak.
        self.assertNotIn("_bus", d)
        self.assertNotIn("_cancel", d)
        self.assertNotIn("_done", d)
        self.assertNotIn("_thread", d)


if __name__ == "__main__":
    unittest.main()
