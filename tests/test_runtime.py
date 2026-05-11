"""Runtime tests with a fake Agent injected.

The real Agent talks to the Anthropic API; here we inject a deterministic
fake so we can verify event ordering, status transitions, budget enforcement,
cancellation, persistence, and subagent delegation without spending money.
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

from agi import events as ev
from agi.costs import Usage
from agi.runtime import Budget, BudgetExceeded, Run, RunCancelled, RunSpec, Runtime


class FakeUsageBlob:
    def __init__(self, input_tokens=10, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class FakeAgent:
    """Deterministic stand-in for agi.agent.Agent.

    `script` is a function `(prompt, agent) -> str` so a single test can
    control behaviour: emit events, accumulate usage, raise BudgetExceeded
    or RunCancelled to exercise those paths.
    """
    def __init__(
        self,
        *,
        memory=None,
        model="fake",
        effort="high",
        verbose=False,
        tracer=None,
        event_sink=None,
        budget=None,
        runtime=None,
        depth=0,
        role_addendum="",
        preload_skills=None,
        **kw,
    ):
        self.memory = memory
        self.model = model
        self.effort = effort
        self.event_sink = event_sink
        self.budget = budget
        self.runtime = runtime
        self.depth = depth
        self.role_addendum = role_addendum
        self.preload_skills = preload_skills or []
        self.usage = Usage()
        self.last_critic_score: float | None = None
        self.handlers: dict = {}
        self.tool_schemas: list = []

    def _rid(self) -> str:
        return self.event_sink.id if self.event_sink is not None else "local"

    def _emit(self, event):
        if self.event_sink is not None:
            self.event_sink.emit(event)

    # Script hook — tests overwrite this.
    def chat(self, prompt, max_iterations=25):
        self._emit(ev.task_started(self._rid(), prompt))
        self.usage.add(FakeUsageBlob(100, 200))
        self._emit(ev.task_completed(self._rid(), f"echo: {prompt}", None))
        return f"echo: {prompt}"


class TestRuntimeBasics(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.rt = Runtime(
            registry_dir=Path(self._tmp.name) / "runs",
            agent_factory=FakeAgent,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_submit_returns_run_with_id(self):
        run = self.rt.submit(RunSpec(prompt="hello"))
        run.wait(timeout=2)
        self.assertTrue(run.id)
        self.assertEqual(run.status, Run.STATUS_COMPLETED)
        self.assertIsNotNone(run.result)
        assert run.result is not None
        self.assertEqual(run.result.text, "echo: hello")

    def test_submit_accepts_plain_string(self):
        run = self.rt.submit("hi")
        run.wait(timeout=2)
        self.assertEqual(run.spec.prompt, "hi")
        self.assertEqual(run.status, Run.STATUS_COMPLETED)

    def test_emits_run_started_and_run_completed(self):
        run = self.rt.submit("x")
        run.wait(timeout=2)
        types = [e.type for e in run.events()]
        self.assertEqual(types[0], "run_started")
        self.assertEqual(types[-1], "run_completed")
        self.assertIn("task_started", types)
        self.assertIn("task_completed", types)

    def test_iter_events_streams_until_terminal(self):
        run = self.rt.submit("x")
        # Drain via iter_events — should terminate cleanly.
        seen = list(run.iter_events(timeout=2))
        self.assertTrue(seen)
        self.assertEqual(seen[-1].type, "run_completed")

    def test_event_sequence_numbers_increment(self):
        run = self.rt.submit("x")
        run.wait(timeout=2)
        seqs = [e.seq for e in run.events()]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(seqs[0], 1)

    def test_get_lookup_by_id(self):
        run = self.rt.submit("x")
        run.wait(timeout=2)
        self.assertIs(self.rt.get(run.id), run)

    def test_persisted_record_on_disk(self):
        run = self.rt.submit("x")
        run.wait(timeout=2)
        record = self.rt.load_record(run.id)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["id"], run.id)
        self.assertEqual(record["status"], "completed")

    def test_cost_rolled_into_result(self):
        run = self.rt.submit("x")
        run.wait(timeout=2)
        assert run.result is not None
        self.assertGreaterEqual(run.result.usage["input_tokens"], 100)


class TestBudget(unittest.TestCase):
    def test_check_input_token_cap(self):
        b = Budget(max_input_tokens=100)
        u = Usage(input_tokens=50)
        b.check(u, "claude-opus-4-7", turns=1, elapsed_s=1.0)
        u.input_tokens = 200
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check(u, "claude-opus-4-7", turns=1, elapsed_s=1.0)
        self.assertEqual(ctx.exception.kind, "input_tokens")

    def test_check_turns_cap(self):
        b = Budget(max_turns=3)
        u = Usage()
        b.check(u, "claude-opus-4-7", turns=3, elapsed_s=0)
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check(u, "claude-opus-4-7", turns=4, elapsed_s=0)
        self.assertEqual(ctx.exception.kind, "turns")

    def test_check_usd_cap_uses_model_pricing(self):
        # opus-4-7 input is $5/M; 1M input tokens = $5
        b = Budget(max_usd=1.0)
        u = Usage(input_tokens=1_000_000)
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check(u, "claude-opus-4-7", turns=1, elapsed_s=0)
        self.assertEqual(ctx.exception.kind, "usd")

    def test_no_limit_means_no_check(self):
        b = Budget()
        u = Usage(input_tokens=10**9, output_tokens=10**9)
        b.check(u, "claude-opus-4-7", turns=10**6, elapsed_s=10**6)  # should not raise


class TestRuntimeBudgetEnforcement(unittest.TestCase):
    def test_budget_exceeded_finishes_as_cancelled_with_event(self):
        class BudgetBlowingAgent(FakeAgent):
            def chat(self, prompt, max_iterations=25):
                self._emit(ev.task_started(self._rid(), prompt))
                raise BudgetExceeded("usd", 1.0, 5.0)

        rt = Runtime(
            registry_dir=tempfile.mkdtemp(),
            agent_factory=BudgetBlowingAgent,
        )
        run = rt.submit(RunSpec(prompt="big task", budget=Budget(max_usd=1.0)))
        run.wait(timeout=2)
        self.assertEqual(run.status, Run.STATUS_CANCELLED)
        types = [e.type for e in run.events()]
        self.assertIn("budget_exceeded", types)
        self.assertIn("run_cancelled", types)


class TestRuntimeCancellation(unittest.TestCase):
    def test_cancel_stops_a_running_agent(self):
        gate = threading.Event()
        cancelled = threading.Event()

        class SlowAgent(FakeAgent):
            def chat(self, prompt, max_iterations=25):
                self._emit(ev.task_started(self._rid(), prompt))
                # Wait until the test calls cancel(), then check.
                gate.wait(timeout=3)
                try:
                    self.event_sink.check_cancelled()
                except RunCancelled:
                    cancelled.set()
                    raise
                return "should not get here"

        rt = Runtime(
            registry_dir=tempfile.mkdtemp(),
            agent_factory=SlowAgent,
        )
        run = rt.submit("slow")
        time.sleep(0.05)
        run.cancel("test")
        gate.set()
        run.wait(timeout=3)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(run.status, Run.STATUS_CANCELLED)


class TestRuntimeFailure(unittest.TestCase):
    def test_unexpected_exception_marks_failed(self):
        class BrokenAgent(FakeAgent):
            def chat(self, prompt, max_iterations=25):
                raise RuntimeError("boom")

        rt = Runtime(
            registry_dir=tempfile.mkdtemp(),
            agent_factory=BrokenAgent,
        )
        run = rt.submit("oops")
        run.wait(timeout=2)
        self.assertEqual(run.status, Run.STATUS_FAILED)
        assert run.error is not None
        self.assertIn("boom", run.error)
        types = [e.type for e in run.events()]
        self.assertIn("run_failed", types)


class TestSubagentDelegation(unittest.TestCase):
    def test_subagent_depth_cap(self):
        # An agent that tries to delegate at depth >= max should fail.
        rt = Runtime(
            registry_dir=tempfile.mkdtemp(),
            agent_factory=FakeAgent,
            max_subagent_depth=2,
        )
        with self.assertRaises(RuntimeError):
            rt.submit(RunSpec(prompt="x"), depth=3)


class TestRunSpecSerialization(unittest.TestCase):
    def test_spec_to_dict_round_trips_budget(self):
        spec = RunSpec(prompt="x", budget=Budget(max_usd=2.0, max_turns=5))
        d = spec.to_dict()
        self.assertEqual(d["budget"]["max_usd"], 2.0)
        self.assertEqual(d["budget"]["max_turns"], 5)


if __name__ == "__main__":
    unittest.main()
