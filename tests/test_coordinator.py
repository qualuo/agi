"""Tests for the Coordinator using a fake AgentRuntime.

The coordinator's contract is `runtime.send(sid, prompt) -> Iterator[Event]
ending in a TurnCompleted`. We satisfy that contract with a callable-driven
fake so we can script specialist responses without touching the API.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from typing import Callable

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.coordinator import Coordinator, Goal, Specialist, _parse_numbered_list
from agi.events import TurnCompleted


class FakeRuntime:
    """Minimal runtime that returns scripted responses.

    `responder` is called with (system_prompt, prompt) and returns the
    final text. Cost defaults to 0; override per-instance if needed.
    """

    def __init__(self, responder: Callable[[str, str], str]):
        self.responder = responder
        self.sessions: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self._next = 0

    def start_session(self, session_id=None, config=None, *, interceptor=None, agent=None):
        self._next += 1
        sid = session_id or f"sess-{self._next}"
        self.sessions[sid] = {"config": config, "system_prompt": getattr(config, "system_prompt", "")}
        return sid

    def close_session(self, sid):
        self.sessions.pop(sid, None)

    def send(self, sid, prompt, max_iterations=25):
        system_prompt = self.sessions[sid]["system_prompt"]
        self.calls.append((system_prompt, prompt))
        text = self.responder(system_prompt, prompt)
        yield TurnCompleted(
            session_id=sid,
            seq=1,
            text=text,
            stop_reason="end_turn",
            cost_usd=0.001,
        )


class TestParseNumberedList(unittest.TestCase):
    def test_basic(self):
        text = "1. first\n2. second\n3. third"
        self.assertEqual(_parse_numbered_list(text), ["first", "second", "third"])

    def test_with_paren(self):
        text = "1) one\n2) two"
        self.assertEqual(_parse_numbered_list(text), ["one", "two"])

    def test_ignores_non_list_lines(self):
        text = "Here is the plan:\n1. a\nblah blah\n2. b\n"
        self.assertEqual(_parse_numbered_list(text), ["a", "b"])

    def test_empty_when_no_list(self):
        self.assertEqual(_parse_numbered_list("no list here"), [])


class TestSingleSubtaskFlow(unittest.TestCase):
    def test_direct_plan_skips_planner(self):
        def respond(system, prompt):
            # Verifier sees a 'PASS' if our executor returns a sentinel
            if "verification" in system.lower():
                return "PASS"
            return "executor-output"

        rt = FakeRuntime(respond)
        coord = Coordinator(rt)
        goal = Goal(description="do the thing", plan="direct")
        outcome = coord.execute(goal)
        self.assertTrue(outcome.success)
        self.assertEqual(len(outcome.subtasks), 1)
        self.assertEqual(outcome.subtasks[0].text, "executor-output")

    def test_success_check_bypasses_verifier(self):
        def respond(system, prompt):
            return "result containing the answer 42"

        rt = FakeRuntime(respond)
        coord = Coordinator(rt)
        goal = Goal(
            description="produce 42",
            plan="direct",
            success_check=lambda t: "42" in t,
        )
        outcome = coord.execute(goal)
        self.assertTrue(outcome.success)
        # Verifier specialist should never have been invoked
        verifier_calls = [
            c for c in rt.calls if "verification specialist" in c[0].lower()
        ]
        self.assertEqual(verifier_calls, [])

    def test_failure_when_verifier_rejects(self):
        def respond(system, prompt):
            if "verification" in system.lower():
                return "FAIL\nresult was missing the answer"
            return "executor-output"

        rt = FakeRuntime(respond)
        coord = Coordinator(rt)
        goal = Goal(description="do it", plan="direct", max_retries=0)
        outcome = coord.execute(goal)
        self.assertFalse(outcome.success)
        self.assertIn("verifier rejected", outcome.failure_reason or "")


class TestPlanThenExecute(unittest.TestCase):
    def test_planner_decomposes_then_executor_runs_each(self):
        executor_calls: list[str] = []

        def respond(system, prompt):
            sys_lower = system.lower()
            if "planning specialist" in sys_lower:
                return "1. download the file\n2. parse the data\n3. summarize"
            if "verification" in sys_lower:
                return "PASS"
            # executor
            executor_calls.append(prompt)
            return f"done: {prompt}"

        rt = FakeRuntime(respond)
        coord = Coordinator(rt, max_parallel=3)
        goal = Goal(description="research and summarize")
        outcome = coord.execute(goal)
        self.assertTrue(outcome.success)
        self.assertEqual(len(outcome.subtasks), 3)
        # All three subtasks got run
        self.assertEqual(
            sorted(executor_calls),
            sorted(["download the file", "parse the data", "summarize"]),
        )

    def test_planner_empty_list_falls_back_to_single_task(self):
        def respond(system, prompt):
            if "planning specialist" in system.lower():
                return "(no decomposition needed)"
            if "verification" in system.lower():
                return "PASS"
            return "did it"

        rt = FakeRuntime(respond)
        coord = Coordinator(rt)
        goal = Goal(description="trivial task")
        outcome = coord.execute(goal)
        self.assertTrue(outcome.success)
        self.assertEqual(len(outcome.subtasks), 1)
        self.assertEqual(outcome.subtasks[0].task, "trivial task")


class TestRetry(unittest.TestCase):
    def test_retries_on_failure(self):
        attempts = {"n": 0}

        def respond(system, prompt):
            sys_lower = system.lower()
            if "verification" in sys_lower:
                attempts["n"] += 1
                return "PASS" if attempts["n"] >= 2 else "FAIL"
            return "result"

        rt = FakeRuntime(respond)
        coord = Coordinator(rt)
        goal = Goal(description="x", plan="direct", max_retries=2)
        outcome = coord.execute(goal)
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.retries_used, 1)


class TestCostAccumulation(unittest.TestCase):
    def test_total_cost_sums_subtasks_plus_verify(self):
        def respond(system, prompt):
            if "planning specialist" in system.lower():
                return "1. a\n2. b"
            if "verification" in system.lower():
                return "PASS"
            return "ok"

        rt = FakeRuntime(respond)
        coord = Coordinator(rt)
        goal = Goal(description="x")
        outcome = coord.execute(goal)
        # planner contributes to subtasks=0 (planner runs as a one-shot inside
        # _plan, but its cost isn't tracked in subtasks; that's intentional —
        # see _plan implementation). 2 executors + 1 verifier = 0.003
        self.assertAlmostEqual(outcome.total_cost_usd, 0.001 * 2 + 0.001, places=6)


class TestParallelism(unittest.TestCase):
    def test_independent_subtasks_run_concurrently(self):
        # Sleep on every executor call; with parallelism=N we expect total
        # wall time to be much less than N * sleep_seconds.
        per_task_sleep = 0.1
        n_tasks = 4

        def respond(system, prompt):
            if "planning specialist" in system.lower():
                return "\n".join(f"{i+1}. task{i}" for i in range(n_tasks))
            if "verification" in system.lower():
                return "PASS"
            time.sleep(per_task_sleep)
            return f"done {prompt}"

        rt = FakeRuntime(respond)
        coord = Coordinator(rt, max_parallel=n_tasks)
        goal = Goal(description="run n in parallel")
        t0 = time.time()
        outcome = coord.execute(goal)
        elapsed = time.time() - t0
        self.assertTrue(outcome.success)
        # Sequential would be n_tasks * per_task_sleep = 0.4s. Parallel should
        # be ~per_task_sleep. Allow generous headroom for slow CI.
        self.assertLess(elapsed, per_task_sleep * (n_tasks - 1))


if __name__ == "__main__":
    unittest.main()
