"""Tests for the runtime engine.

These never call the Anthropic API — they use a FakeAgent that satisfies
the AgentLike protocol from agi.runtime. That isolates the engine logic
(budgets, events, concurrency, cancellation) from network and pricing
volatility, while still proving the contract the real Agent must meet.
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

from agi.costs import Usage
from agi.runtime import Budget, Event, RuntimeEngine, Task, TaskResult


class FakeAgent:
    """Implements the minimum surface that RuntimeEngine consumes."""

    def __init__(
        self,
        *,
        output: str = "done",
        input_tokens: int = 100,
        output_tokens: int = 50,
        sleep_s: float = 0.0,
        raises: Exception | None = None,
        model: str = "claude-opus-4-7",
        critic_score: float | None = None,
    ):
        self.model = model
        self.usage = Usage()
        self.messages: list[dict] = []
        self.last_critic_score = critic_score
        self.tool_schemas: list[dict] = []
        self.handlers: dict = {}
        self.critic = None
        self.tracer = None
        self._output = output
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._sleep_s = sleep_s
        self._raises = raises
        self.last_prompt: str | None = None

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.last_prompt = user_input
        if self._sleep_s:
            time.sleep(self._sleep_s)
        if self._raises is not None:
            raise self._raises
        # Mimic a single turn.
        class _U:
            def __init__(self, i, o):
                self.input_tokens = i
                self.output_tokens = o
                self.cache_creation_input_tokens = 0
                self.cache_read_input_tokens = 0
        self.usage.add(_U(self._input_tokens, self._output_tokens))
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": self._output})
        return self._output

    def reset(self) -> None:
        self.usage = Usage()
        self.messages = []


def factory(**kwargs):
    return lambda: FakeAgent(**kwargs)


class TestRuntimeExecute(unittest.TestCase):
    def test_happy_path(self):
        engine = RuntimeEngine(factory(output="hello"))
        result = engine.execute(Task(instruction="say hi"))
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.output, "hello")
        self.assertGreater(result.iterations, 0)
        self.assertGreater(result.usage["input_tokens"], 0)

    def test_failure_caught(self):
        engine = RuntimeEngine(factory(raises=RuntimeError("boom")))
        result = engine.execute(Task(instruction="break it"))
        self.assertEqual(result.status, "failed")
        self.assertIn("boom", result.error or "")

    def test_invalid_budget_is_a_failure(self):
        engine = RuntimeEngine(factory())
        result = engine.execute(Task(instruction="x", budget=Budget(max_iterations=0)))
        self.assertEqual(result.status, "failed")

    def test_token_budget_exceeded(self):
        # 1000 in + 1000 out total = 2000 used; budget 100 → fail.
        engine = RuntimeEngine(factory(input_tokens=1000, output_tokens=1000))
        result = engine.execute(Task(
            instruction="big",
            budget=Budget(max_tokens=100, max_cost_usd=1000, max_iterations=25, deadline_s=60),
        ))
        self.assertEqual(result.status, "budget_exceeded")
        self.assertIn("tokens", result.error or "")

    def test_cost_budget_exceeded(self):
        # 1M input + 1M output on opus-4-7 = $30; cap at $0.01 → fail.
        engine = RuntimeEngine(factory(input_tokens=1_000_000, output_tokens=1_000_000))
        result = engine.execute(Task(
            instruction="expensive",
            budget=Budget(max_tokens=10_000_000, max_cost_usd=0.01, max_iterations=25, deadline_s=60),
        ))
        self.assertEqual(result.status, "budget_exceeded")
        self.assertIn("cost", result.error or "")

    def test_critic_score_surfaced(self):
        engine = RuntimeEngine(factory(critic_score=0.42))
        result = engine.execute(Task(instruction="ok"))
        self.assertEqual(result.critic_score, 0.42)

    def test_inputs_rendered_into_prompt(self):
        agent = FakeAgent(output="ok")
        engine = RuntimeEngine(lambda: agent)
        engine.execute(Task(instruction="do thing", inputs={"a": 1, "b": "two"}))
        assert agent.last_prompt is not None
        self.assertIn("do thing", agent.last_prompt)
        self.assertIn('"a": 1', agent.last_prompt)
        self.assertIn('"b": "two"', agent.last_prompt)


class TestRuntimeEvents(unittest.TestCase):
    def test_events_recorded_in_order(self):
        engine = RuntimeEngine(factory())
        result = engine.execute(Task(instruction="hi"))
        types = [e.type for e in result.events]
        self.assertEqual(types[0], "started")
        self.assertEqual(types[-1], "succeeded")
        self.assertIn("iteration", types)

    def test_subscriber_receives_events_live(self):
        seen: list[Event] = []
        engine = RuntimeEngine(factory(sleep_s=0.02))
        task_id = engine.submit(Task(instruction="hi"), on_event=seen.append)
        engine.await_result(task_id, timeout=5.0)
        types = [e.type for e in seen]
        self.assertIn("started", types)
        self.assertIn("succeeded", types)

    def test_stream_events_yields_until_terminal(self):
        engine = RuntimeEngine(factory(sleep_s=0.05))
        task_id = engine.submit(Task(instruction="hi"))
        events = list(engine.stream_events(task_id))
        types = [e.type for e in events]
        self.assertEqual(types[0], "started")
        self.assertIn("succeeded", types)

    def test_failed_event_emitted(self):
        engine = RuntimeEngine(factory(raises=ValueError("nope")))
        result = engine.execute(Task(instruction="hi"))
        types = [e.type for e in result.events]
        self.assertIn("failed", types)


class TestRuntimeSubmit(unittest.TestCase):
    def test_submit_and_await(self):
        engine = RuntimeEngine(factory())
        task_id = engine.submit(Task(instruction="bg"))
        result = engine.await_result(task_id, timeout=5.0)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.task_id, task_id)

    def test_get_result_before_completion_shows_running(self):
        engine = RuntimeEngine(factory(sleep_s=0.1))
        task_id = engine.submit(Task(instruction="bg"))
        # We can't reliably catch "running" without a race; either running or done is fine.
        result = engine.get_result(task_id)
        self.assertIn(result.status, ("running", "pending", "succeeded"))
        engine.await_result(task_id, timeout=5.0)

    def test_unknown_task_id_raises(self):
        engine = RuntimeEngine(factory())
        with self.assertRaises(KeyError):
            engine.get_result("nope")
        with self.assertRaises(KeyError):
            engine.await_result("nope", timeout=0.1)

    def test_explicit_id_honored(self):
        engine = RuntimeEngine(factory())
        task_id = engine.submit(Task(id="custom-1234", instruction="x"))
        self.assertEqual(task_id, "custom-1234")
        engine.await_result(task_id, timeout=5.0)


class TestSkillInjection(unittest.TestCase):
    def test_explicit_skills_loaded_into_prompt(self):
        tmpdir = tempfile.mkdtemp()
        try:
            # Save a skill in a non-default library by monkeypatching at the
            # point of import inside _assemble_prompt. We use the real
            # library but place it where the engine will look — point home
            # at the temp dir.
            import os
            old = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                from agi.skills import SkillLibrary
                lib = SkillLibrary()
                lib.save("greet", "How to greet politely", "Always say hello first.")

                agent = FakeAgent(output="ok")
                engine = RuntimeEngine(lambda: agent)
                engine.execute(Task(instruction="greet user", skills=["greet"]))
                assert agent.last_prompt is not None
                self.assertIn("skill: greet", agent.last_prompt)
                self.assertIn("Always say hello first.", agent.last_prompt)
            finally:
                if old is not None:
                    os.environ["HOME"] = old
                else:
                    del os.environ["HOME"]
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestConcurrent(unittest.TestCase):
    def test_two_tasks_run_independently(self):
        # Each task gets a fresh agent via the factory, so usage doesn't
        # bleed between them. This is the core invariant for safe
        # coordination dispatch.
        results: dict[str, TaskResult] = {}
        engine = RuntimeEngine(factory(sleep_s=0.05))
        ids = [engine.submit(Task(instruction=f"t{i}")) for i in range(3)]
        for tid in ids:
            results[tid] = engine.await_result(tid, timeout=5.0)
        statuses = {r.status for r in results.values()}
        self.assertEqual(statuses, {"succeeded"})


if __name__ == "__main__":
    unittest.main()
