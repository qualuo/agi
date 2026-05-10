"""Tests for the Runtime control plane.

Uses a stub agent so we don't hit the Anthropic API. The stub honors the
same control-signal contract that the real Agent.chat_controlled implements:
cancel_event, budget_usd, event_sink. A separate integration test would run
the real Agent end-to-end.
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
    JobCanceled,
    Runtime,
    SessionRecord,
)


class _Usage:
    """Mimics the slice of agi.costs.Usage that the runtime reads."""

    def __init__(self, in_tok=0, out_tok=0, cost=0.0) -> None:
        self.input_tokens = in_tok
        self.output_tokens = out_tok
        self._cost = cost

    def cost_usd(self, model: str) -> float:
        return self._cost


class StubAgent:
    """Test-only agent. Ignores the prompt and returns a configured response.

    Honors cancel_event, budget_usd, and event_sink so we can exercise the
    full Runtime contract without an API call.
    """

    def __init__(
        self,
        response: str = "ok",
        sleep_s: float = 0.0,
        emit_text: bool = True,
        raise_in_chat: Exception | None = None,
        cost_per_call: float = 0.001,
    ) -> None:
        self.response = response
        self.sleep_s = sleep_s
        self.emit_text = emit_text
        self.raise_in_chat = raise_in_chat
        self.cost_per_call = cost_per_call
        self.usage = _Usage()
        self.calls: list[str] = []

    def chat_controlled(
        self,
        prompt: str,
        *,
        cancel_event: threading.Event | None = None,
        budget_usd: float | None = None,
        event_sink=None,
        max_iterations: int = 25,
    ) -> str:
        self.calls.append(prompt)

        # Simulate work in small slices so cancel checks fire.
        slice_s = 0.02
        elapsed = 0.0
        while elapsed < self.sleep_s:
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("canceled")
            time.sleep(slice_s)
            elapsed += slice_s

        if self.raise_in_chat:
            raise self.raise_in_chat

        # Bump usage so the runtime can compute cost deltas
        self.usage.input_tokens += 100
        self.usage.output_tokens += 50
        self.usage._cost += self.cost_per_call

        if budget_usd is not None and self.usage._cost > budget_usd:
            raise BudgetExceeded(f"over budget {self.usage._cost} > {budget_usd}")

        if self.emit_text and event_sink:
            event_sink("text_delta", {"text": self.response})
        return self.response


def make_factory(**kwargs):
    """Return an agent factory that builds StubAgent(**kwargs) for any session."""
    def factory(session: SessionRecord):
        return StubAgent(**kwargs)
    return factory


class TestRuntimeBasics(unittest.TestCase):
    def test_create_and_get_session(self):
        rt = Runtime(agent_factory=make_factory(response="hi"))
        try:
            s = rt.create_session(role="researcher")
            self.assertEqual(s.role, "researcher")
            again = rt.get_session(s.id)
            self.assertEqual(again.id, s.id)
        finally:
            rt.shutdown()

    def test_submit_and_await(self):
        rt = Runtime(agent_factory=make_factory(response="42"))
        try:
            s = rt.create_session()
            job = rt.submit(s.id, "the prompt")
            rec = rt.await_job(job.id, timeout=5)
            self.assertEqual(rec.status, "succeeded")
            self.assertEqual(rec.output, "42")
            self.assertGreaterEqual(rec.cost_usd, 0.0)
        finally:
            rt.shutdown()

    def test_unknown_session(self):
        rt = Runtime(agent_factory=make_factory())
        try:
            with self.assertRaises(KeyError):
                rt.submit("does-not-exist", "p")
        finally:
            rt.shutdown()


class TestRuntimeCancellation(unittest.TestCase):
    def test_cancel_running_job(self):
        rt = Runtime(agent_factory=make_factory(response="x", sleep_s=2.0))
        try:
            s = rt.create_session()
            job = rt.submit(s.id, "long task")
            # Give the worker a beat to start
            time.sleep(0.1)
            rt.cancel(job.id)
            rec = rt.await_job(job.id, timeout=2)
            self.assertEqual(rec.status, "canceled")
        finally:
            rt.shutdown()


class TestRuntimeBudget(unittest.TestCase):
    def test_budget_exceeded_marks_failed(self):
        # cost_per_call=0.10 with budget 0.05 → the StubAgent itself raises.
        rt = Runtime(agent_factory=make_factory(response="x", cost_per_call=0.10))
        try:
            s = rt.create_session()
            job = rt.submit(s.id, "p", budget_usd=0.05)
            rec = rt.await_job(job.id, timeout=5)
            self.assertEqual(rec.status, "failed")
            self.assertIn("budget", rec.error or "")
        finally:
            rt.shutdown()


class TestRuntimeFailure(unittest.TestCase):
    def test_agent_exception_marks_failed(self):
        boom = RuntimeError("kaboom")
        rt = Runtime(agent_factory=make_factory(raise_in_chat=boom))
        try:
            s = rt.create_session()
            job = rt.submit(s.id, "p")
            rec = rt.await_job(job.id, timeout=5)
            self.assertEqual(rec.status, "failed")
            self.assertIn("kaboom", rec.error or "")
        finally:
            rt.shutdown()


class TestRuntimeEvents(unittest.TestCase):
    def test_stream_yields_events_and_terminates(self):
        rt = Runtime(agent_factory=make_factory(response="hello"))
        try:
            s = rt.create_session()
            job = rt.submit(s.id, "p")
            kinds = [ev.kind for ev in rt.stream(job.id, timeout=5)]
            # We expect at least: status running, text_delta, status succeeded.
            self.assertIn("text_delta", kinds)
            self.assertIn("status", kinds)
        finally:
            rt.shutdown()


class TestRuntimeMetrics(unittest.TestCase):
    def test_metrics_count_jobs(self):
        rt = Runtime(agent_factory=make_factory(response="x"))
        try:
            s = rt.create_session()
            jobs = [rt.submit(s.id, f"p{i}") for i in range(3)]
            for j in jobs:
                rt.await_job(j.id, timeout=5)
            m = rt.metrics()
            self.assertEqual(m["jobs_total"], 3)
            self.assertEqual(m["jobs_by_status"].get("succeeded", 0), 3)
            self.assertEqual(m["sessions"], 1)
        finally:
            rt.shutdown()


class TestRuntimeConcurrency(unittest.TestCase):
    def test_jobs_in_different_sessions_run_in_parallel(self):
        rt = Runtime(agent_factory=make_factory(response="x", sleep_s=0.4), max_workers=4)
        try:
            sessions = [rt.create_session() for _ in range(3)]
            t0 = time.time()
            jobs = [rt.submit(s.id, "p") for s in sessions]
            for j in jobs:
                rt.await_job(j.id, timeout=5)
            elapsed = time.time() - t0
            # If serialized: 3 * 0.4 = 1.2s. Parallel: ~0.4s. Generous bound.
            self.assertLess(elapsed, 1.0)
        finally:
            rt.shutdown()

    def test_jobs_in_same_session_serialize(self):
        # Per-session lock means same-session jobs do NOT overlap.
        rt = Runtime(agent_factory=make_factory(response="x", sleep_s=0.2), max_workers=4)
        try:
            s = rt.create_session()
            jobs = [rt.submit(s.id, "p") for _ in range(3)]
            t0 = time.time()
            for j in jobs:
                rt.await_job(j.id, timeout=5)
            elapsed = time.time() - t0
            # Serialized lower bound: 3 * 0.2 = 0.6s.
            self.assertGreater(elapsed, 0.5)
        finally:
            rt.shutdown()


class TestRuntimeSnapshot(unittest.TestCase):
    def test_snapshot_round_trips(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            rt1 = Runtime(agent_factory=make_factory(response="x"), snapshot_path=path)
            try:
                s = rt1.create_session(role="r")
                j = rt1.submit(s.id, "p")
                rt1.await_job(j.id, timeout=5)
            finally:
                rt1.shutdown()

            rt2 = Runtime(agent_factory=make_factory(response="x"), snapshot_path=path)
            try:
                rec = rt2.get_session(s.id)
                self.assertEqual(rec.role, "r")
                # The completed job should be readable post-restore
                jobs = rt2.list_jobs(session_id=s.id)
                self.assertEqual(len(jobs), 1)
            finally:
                rt2.shutdown()


if __name__ == "__main__":
    unittest.main()
