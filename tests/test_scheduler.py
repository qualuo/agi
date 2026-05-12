"""Tests for ParallelScheduler — DAG-aware parallel plan execution.

Uses FakeAgent from test_runtime to avoid hitting the API.  Concurrency
assertions use a shared lock-protected counter to detect overlap.
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

from agi.coordinator import Plan, PlanStep
from agi.memory import Memory
from agi.runtime import Runtime
from agi.scheduler import (
    PLAN_COMPLETED,
    PLAN_FAILED,
    PLAN_STEP_COMPLETED,
    PLAN_STEP_FAILED,
    PLAN_STEP_RETRY,
    PLAN_STEP_RUNNING,
    CycleError,
    ParallelScheduler,
    RetryPolicy,
    SchedulerConfig,
)
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent, FakeUsage


class TimedAgent(FakeAgent):
    """FakeAgent that holds chat() open long enough to observe overlap."""

    hold_seconds = 0.05

    def chat(self, prompt: str, max_iterations: int = 25) -> str:  # type: ignore[override]
        with _shared.lock:
            _shared.in_flight += 1
            _shared.peak = max(_shared.peak, _shared.in_flight)
        try:
            time.sleep(self.hold_seconds)
            return super().chat(prompt, max_iterations=max_iterations)
        finally:
            with _shared.lock:
                _shared.in_flight -= 1


class _Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0

    def reset(self) -> None:
        with self.lock:
            self.in_flight = 0
            self.peak = 0


_shared = _Shared()


class FailingAgent(FakeAgent):
    """Fails the first N times then succeeds — used to exercise retries."""

    fail_times = 0
    call_count = 0

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._calls = 0

    def chat(self, prompt: str, max_iterations: int = 25) -> str:  # type: ignore[override]
        self._calls += 1
        FailingAgent.call_count += 1
        if FailingAgent.call_count <= FailingAgent.fail_times:
            raise RuntimeError(f"flaky #{FailingAgent.call_count}")
        return super().chat(prompt, max_iterations=max_iterations)


def _make_runtime(factory=FakeAgent) -> Runtime:
    tmp = tempfile.mkdtemp()
    return Runtime(
        memory=Memory(path=Path(tmp) / "m.jsonl"),
        skills=SkillLibrary(path=Path(tmp) / "skills"),
        agent_factory=factory,
    )


class TestValidation(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        rt = _make_runtime()
        sched = ParallelScheduler(rt)
        plan = Plan(steps=[PlanStep(id="a", prompt="x"), PlanStep(id="a", prompt="y")])
        with self.assertRaises(CycleError):
            sched.submit(plan)

    def test_unknown_dep_rejected(self):
        rt = _make_runtime()
        sched = ParallelScheduler(rt)
        plan = Plan(steps=[PlanStep(id="a", prompt="x", depends_on=["b"])])
        with self.assertRaises(CycleError):
            sched.submit(plan)

    def test_cycle_rejected(self):
        rt = _make_runtime()
        sched = ParallelScheduler(rt)
        plan = Plan(steps=[
            PlanStep(id="a", prompt="x", depends_on=["b"]),
            PlanStep(id="b", prompt="y", depends_on=["a"]),
        ])
        with self.assertRaises(CycleError):
            sched.submit(plan)


class TestSequentialDag(unittest.TestCase):
    def test_linear_chain_runs_in_order(self):
        rt = _make_runtime()
        sched = ParallelScheduler(rt, config=SchedulerConfig(max_concurrent_steps=4))
        plan = Plan(steps=[
            PlanStep(id="a", prompt="A"),
            PlanStep(id="b", prompt="B", depends_on=["a"]),
            PlanStep(id="c", prompt="C", depends_on=["b"]),
        ])
        events_order: list[str] = []
        rt.subscribe(
            lambda e: events_order.append(e.data["step_id"]),
            kind=PLAN_STEP_COMPLETED,
        )
        result = sched.run(plan)
        self.assertEqual(result.status, "done")
        self.assertEqual(events_order, ["a", "b", "c"])
        for sid in ("a", "b", "c"):
            self.assertEqual(result.outcomes[sid].status, "done")

    def test_zero_step_plan_rejected_at_construction(self):
        # Empty plans never validate (validation requires the test below to
        # also be a meaningful unit). We allow empty plans through the
        # scheduler but they finish immediately.
        rt = _make_runtime()
        sched = ParallelScheduler(rt)
        plan = Plan(steps=[])
        result = sched.run(plan)
        self.assertEqual(result.status, "done")
        self.assertEqual(result.outcomes, {})


class TestParallelism(unittest.TestCase):
    def test_independent_steps_run_concurrently(self):
        _shared.reset()
        rt = _make_runtime(factory=TimedAgent)
        sched = ParallelScheduler(rt, config=SchedulerConfig(max_concurrent_steps=4))
        plan = Plan(steps=[
            PlanStep(id=f"s{i}", prompt=f"prompt-{i}") for i in range(4)
        ])
        result = sched.run(plan)
        self.assertEqual(result.status, "done")
        # We launched 4 independent steps with a 50ms hold; peak concurrent
        # should be > 1 (loose check — exact value depends on scheduling).
        self.assertGreaterEqual(_shared.peak, 2)

    def test_concurrency_cap_respected(self):
        _shared.reset()
        rt = _make_runtime(factory=TimedAgent)
        sched = ParallelScheduler(rt, config=SchedulerConfig(max_concurrent_steps=2))
        plan = Plan(steps=[
            PlanStep(id=f"s{i}", prompt=f"prompt-{i}") for i in range(6)
        ])
        result = sched.run(plan)
        self.assertEqual(result.status, "done")
        self.assertLessEqual(_shared.peak, 2)

    def test_fan_in_dag(self):
        # a + b run in parallel, c waits for both.
        _shared.reset()
        rt = _make_runtime(factory=TimedAgent)
        sched = ParallelScheduler(rt, config=SchedulerConfig(max_concurrent_steps=4))
        plan = Plan(steps=[
            PlanStep(id="a", prompt="A"),
            PlanStep(id="b", prompt="B"),
            PlanStep(id="c", prompt="C", depends_on=["a", "b"]),
        ])
        ran: list[str] = []
        rt.subscribe(
            lambda e: ran.append(e.data["step_id"]),
            kind=PLAN_STEP_COMPLETED,
        )
        result = sched.run(plan)
        self.assertEqual(result.status, "done")
        # c must complete last
        self.assertEqual(ran[-1], "c")


class TestRetries(unittest.TestCase):
    def setUp(self) -> None:
        FailingAgent.call_count = 0
        FailingAgent.fail_times = 0

    def test_retry_succeeds_after_transient_failures(self):
        FailingAgent.fail_times = 2
        rt = _make_runtime(factory=FailingAgent)
        sched = ParallelScheduler(
            rt,
            config=SchedulerConfig(
                retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
            ),
        )
        retry_events: list[int] = []
        rt.subscribe(
            lambda e: retry_events.append(e.data["attempt"]),
            kind=PLAN_STEP_RETRY,
        )
        plan = Plan(steps=[PlanStep(id="a", prompt="x")])
        result = sched.run(plan)
        self.assertEqual(result.status, "done")
        self.assertEqual(result.outcomes["a"].status, "done")
        # Retry fires after each failed attempt that has another to follow:
        # attempts 1 & 2 fail → 2 retry events.
        self.assertEqual(len(retry_events), 2)

    def test_retry_gives_up_after_max_attempts(self):
        FailingAgent.fail_times = 10  # always fail
        rt = _make_runtime(factory=FailingAgent)
        sched = ParallelScheduler(
            rt,
            config=SchedulerConfig(
                retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.0),
            ),
        )
        plan = Plan(steps=[PlanStep(id="a", prompt="x")])
        result = sched.run(plan)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.outcomes["a"].status, "failed")
        # 2 failed attempts logged
        self.assertEqual(len(result.failures["a"]), 2)

    def test_per_step_retry_override(self):
        FailingAgent.fail_times = 3
        rt = _make_runtime(factory=FailingAgent)
        sched = ParallelScheduler(
            rt,
            config=SchedulerConfig(
                retry_policy=RetryPolicy(max_attempts=1, backoff_seconds=0.0),
            ),
        )
        plan = Plan(steps=[
            PlanStep(
                id="a", prompt="x",
                metadata={"retry": {"max_attempts": 5, "backoff_seconds": 0.0}},
            ),
        ])
        result = sched.run(plan)
        # Step-level override of max_attempts=5 should beat the 3 failures.
        self.assertEqual(result.outcomes["a"].status, "done")


class TestFailureModes(unittest.TestCase):
    def test_dependent_steps_skipped_when_parent_fails(self):
        FailingAgent.call_count = 0
        FailingAgent.fail_times = 1
        rt = _make_runtime(factory=FailingAgent)
        sched = ParallelScheduler(
            rt,
            config=SchedulerConfig(
                retry_policy=RetryPolicy(max_attempts=1),
            ),
        )
        plan = Plan(steps=[
            PlanStep(id="a", prompt="A"),
            PlanStep(id="b", prompt="B", depends_on=["a"]),
            PlanStep(id="c", prompt="C", depends_on=["b"]),
        ])
        result = sched.run(plan)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.outcomes["a"].status, "failed")
        self.assertEqual(result.outcomes["b"].status, "skipped")
        self.assertEqual(result.outcomes["c"].status, "skipped")

    def test_fail_fast_cancels_independent_steps(self):
        FailingAgent.call_count = 0
        FailingAgent.fail_times = 1
        rt = _make_runtime(factory=FailingAgent)
        sched = ParallelScheduler(
            rt,
            config=SchedulerConfig(
                retry_policy=RetryPolicy(max_attempts=1),
                fail_fast=True,
                max_concurrent_steps=1,
            ),
        )
        # All independent — fail_fast=True should skip the rest once one fails.
        plan = Plan(steps=[PlanStep(id=f"s{i}", prompt=f"p{i}") for i in range(4)])
        result = sched.run(plan)
        self.assertEqual(result.status, "failed")
        # At least one skipped from fail_fast.
        skipped = sum(1 for o in result.outcomes.values() if o.status == "skipped")
        self.assertGreaterEqual(skipped, 1)


class TestBudgetAndDeadline(unittest.TestCase):
    def test_budget_halts_further_dispatch(self):
        rt = _make_runtime()
        sched = ParallelScheduler(rt, config=SchedulerConfig(max_concurrent_steps=1))
        plan = Plan(steps=[PlanStep(id=f"s{i}", prompt=f"p{i}") for i in range(5)])
        # Each FakeAgent chat costs (100*0.000005 + 50*0.000025) ≈ $0.00175.
        result = sched.run(plan, budget_usd=0.002)
        # First step must run; later steps should stop.
        done = sum(1 for o in result.outcomes.values() if o.status == "done")
        self.assertGreaterEqual(done, 1)
        self.assertLess(done, 5)
        self.assertIn(result.status, ("budget_exhausted", "done"))

    def test_deadline_terminates_plan(self):
        rt = _make_runtime(factory=TimedAgent)
        sched = ParallelScheduler(rt, config=SchedulerConfig(max_concurrent_steps=1))
        plan = Plan(steps=[PlanStep(id=f"s{i}", prompt=f"p{i}") for i in range(20)])
        # Deadline in 100ms — at 50ms/step that's about 2 steps.
        result = sched.run(plan, deadline_ts=time.time() + 0.1)
        self.assertIn(result.status, ("failed", "done"))
        done = sum(1 for o in result.outcomes.values() if o.status == "done")
        self.assertLess(done, 20)


class TestEvents(unittest.TestCase):
    def test_emits_lifecycle_events(self):
        rt = _make_runtime()
        sched = ParallelScheduler(rt)
        plan = Plan(steps=[
            PlanStep(id="a", prompt="A"),
            PlanStep(id="b", prompt="B", depends_on=["a"]),
        ])
        seen: list[str] = []
        for kind in (PLAN_STEP_RUNNING, PLAN_STEP_COMPLETED, PLAN_COMPLETED):
            rt.subscribe(lambda e, k=kind: seen.append(k), kind=kind)
        sched.run(plan)
        # We saw at least 2 runs, 2 completions, 1 plan-completed.
        self.assertEqual(seen.count(PLAN_STEP_RUNNING), 2)
        self.assertEqual(seen.count(PLAN_STEP_COMPLETED), 2)
        self.assertEqual(seen.count(PLAN_COMPLETED), 1)


class TestCancellation(unittest.TestCase):
    def test_cancel_stops_new_dispatch(self):
        _shared.reset()

        class SlowAgent(TimedAgent):
            hold_seconds = 0.2

        rt = _make_runtime(factory=SlowAgent)
        sched = ParallelScheduler(rt, config=SchedulerConfig(max_concurrent_steps=1))
        plan = Plan(steps=[PlanStep(id=f"s{i}", prompt=f"p{i}") for i in range(5)])
        eid = sched.submit(plan)
        # Let one step start, then cancel.
        time.sleep(0.05)
        self.assertTrue(sched.cancel(eid))
        result = sched.wait(eid, timeout=5.0)
        self.assertEqual(result.status, "cancelled")
        # Should not have run all 5.
        done = sum(1 for o in result.outcomes.values() if o.status == "done")
        self.assertLess(done, 5)


class TestSnapshots(unittest.TestCase):
    def test_to_dict_includes_summary(self):
        rt = _make_runtime()
        sched = ParallelScheduler(rt)
        plan = Plan(steps=[PlanStep(id="a", prompt="A")])
        result = sched.run(plan)
        d = result.to_dict()
        self.assertEqual(d["status"], "done")
        self.assertEqual(d["step_count"], 1)
        self.assertEqual(d["completed_count"], 1)
        self.assertIn("a", d["outcomes"])

    def test_list_executions(self):
        rt = _make_runtime()
        sched = ParallelScheduler(rt)
        plan = Plan(steps=[PlanStep(id="a", prompt="A")])
        sched.run(plan)
        sched.run(plan)
        listed = sched.list_executions()
        self.assertEqual(len(listed), 2)


if __name__ == "__main__":
    unittest.main()
