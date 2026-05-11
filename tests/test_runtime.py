"""Runtime engine tests — exercise the coordination API without hitting Claude.

We stub the client at the agent layer with a fake that yields a canned
response sequence. This lets us verify wiring (events, tool dispatch,
budget/deadline gating, capability introspection) deterministically.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.runtime import Runtime, Task
from agi.tools import _compile_dynamic_tool, _validate_tool_source


# ----- fake Anthropic stream that returns scripted responses -----


class FakeBlock(SimpleNamespace):
    pass


class FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class FakeResponse:
    def __init__(self, content, stop_reason="end_turn", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or FakeUsage()


class FakeStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([])

    def get_final_message(self):
        return self.response


class FakeAnthropic:
    """A scripted client. Each call to messages.stream(...) pops the next
    response off the queue."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **_kwargs):
        if not self.responses:
            r = FakeResponse([FakeBlock(type="text", text="")])
        else:
            r = self.responses.pop(0)
        self.calls += 1
        return FakeStream(r)


class FakeAnthropicTestCase(unittest.TestCase):
    """Patches anthropic.Anthropic for every test in the subclass.

    Each test calls `self._set_responses(...)` to enqueue scripted
    responses. The same fake instance is returned for every Agent built
    during the test, so spawn()ed and delegated sub-sessions share it.
    """

    def setUp(self):
        self.fake = FakeAnthropic([])
        self._patcher = patch("agi.agent.anthropic.Anthropic", return_value=self.fake)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _set_responses(self, responses):
        self.fake.responses = list(responses)

    def _new_runtime(self, **kwargs):
        return Runtime(verbose=False, **kwargs)


# ----- tests -----


class TestRuntimeBasic(FakeAnthropicTestCase):
    def test_execute_returns_ok_result(self):
        self._set_responses([FakeResponse([FakeBlock(type="text", text="hello world")])])
        rt = self._new_runtime()
        result = rt.execute(Task(goal="say hi"))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output, "hello world")
        self.assertGreater(result.iterations, 0)
        self.assertGreaterEqual(result.cost_usd, 0.0)

    def test_to_dict_round_trips(self):
        self._set_responses([FakeResponse([FakeBlock(type="text", text="x")])])
        rt = self._new_runtime()
        d = rt.execute(Task(goal="x")).to_dict()
        self.assertEqual(d["status"], "ok")
        self.assertIn("usage", d)
        self.assertIn("input_tokens", d["usage"])


class TestRuntimeEvents(FakeAnthropicTestCase):
    def test_events_emitted_in_order(self):
        self._set_responses([FakeResponse([FakeBlock(type="text", text="done")])])
        rt = self._new_runtime()
        events = []
        rt.execute(Task(goal="go"), on_event=events.append)
        kinds = [e.kind for e in events]
        self.assertEqual(kinds[0], "started")
        self.assertEqual(kinds[-1], "finished")
        self.assertIn("iteration", kinds)

    def test_subscriber_exception_does_not_crash_run(self):
        self._set_responses([FakeResponse([FakeBlock(type="text", text="ok")])])
        rt = self._new_runtime()

        def broken(_ev):
            raise RuntimeError("subscriber blew up")

        result = rt.execute(Task(goal="x"), on_event=broken)
        self.assertEqual(result.status, "ok")


class TestRuntimeBudget(FakeAnthropicTestCase):
    def test_token_budget_enforced_between_turns(self):
        big_usage = FakeUsage(input_tokens=1000, output_tokens=1000)
        self._set_responses([
            FakeResponse(
                [FakeBlock(type="text", text="thinking...")],
                stop_reason="tool_use",
                usage=big_usage,
            ),
            FakeResponse([FakeBlock(type="text", text="final")]),
        ])
        rt = self._new_runtime()
        result = rt.execute(Task(goal="big", max_tokens_budget=100))
        self.assertEqual(result.status, "budget_exceeded")


class TestRuntimeDeadline(FakeAnthropicTestCase):
    def test_deadline_fires_between_turns(self):
        self._set_responses([
            FakeResponse(
                [FakeBlock(type="text", text="...")],
                stop_reason="tool_use",
            ),
        ] * 5)
        rt = self._new_runtime()
        result = rt.execute(Task(goal="slow", deadline_seconds=-1.0))
        self.assertEqual(result.status, "deadline_exceeded")


class TestRuntimeCancel(FakeAnthropicTestCase):
    def test_cancel_observed_between_turns(self):
        self._set_responses([
            FakeResponse(
                [FakeBlock(type="text", text="...")],
                stop_reason="tool_use",
            ),
            FakeResponse([FakeBlock(type="text", text="never reached")]),
        ])
        rt = self._new_runtime()
        session = rt.spawn()
        session.cancel()
        result = session.run(Task(goal="x"))
        self.assertEqual(result.status, "cancelled")


class TestRuntimeIntrospection(FakeAnthropicTestCase):
    def test_describe_reports_tools(self):
        rt = self._new_runtime()
        caps = rt.describe()
        self.assertIn("read_file", caps["tools"])
        self.assertIn("save_memory", caps["tools"])
        # Skills are enabled by default if learner is installed.
        self.assertIn("recall_skill", caps["tools"])
        # Tool synthesis is enabled in the describe-probe agent.
        self.assertIn("make_tool", caps["tools"])
        # Server-side tools are reported separately.
        self.assertTrue(any("web_" in t for t in caps["server_tools"]))


class TestToolWhitelist(FakeAnthropicTestCase):
    def test_only_allowed_tools_remain(self):
        self._set_responses([FakeResponse([FakeBlock(type="text", text="ok")])])
        rt = self._new_runtime()
        session = rt.spawn()
        before = {s.get("name") for s in session.agent.tool_schemas if s.get("name")}
        session.run(Task(goal="x", allowed_tools=["read_file"]))
        after = {s.get("name") for s in session.agent.tool_schemas if s.get("name")}
        self.assertEqual(before, after)


class TestToolSynthesisCompiler(unittest.TestCase):
    """These tests don't need the anthropic stub — they exercise the
    sandbox helpers directly."""

    def test_compiles_simple_function(self):
        fn = _compile_dynamic_tool(
            "double",
            "def double(text):\n    return text + text\n",
        )
        self.assertEqual(fn("ab"), "abab")

    def test_rejects_imports(self):
        with self.assertRaises(ValueError):
            _validate_tool_source("import os\ndef f(text): return text", "f")

    def test_rejects_dunder_attr(self):
        with self.assertRaises(ValueError):
            _validate_tool_source("def f(text):\n    return text.__class__\n", "f")

    def test_rejects_exec(self):
        with self.assertRaises(ValueError):
            _validate_tool_source("def f(text):\n    exec(text)\n", "f")

    def test_requires_named_function(self):
        with self.assertRaises(ValueError):
            _validate_tool_source("def other(text): return text", "expected")


class TestToolSynthesisIntegration(FakeAnthropicTestCase):
    def test_runtime_error_in_user_tool_is_caught(self):
        self._set_responses([FakeResponse([FakeBlock(type="text", text="x")])])
        rt = self._new_runtime()
        session = rt.spawn(enable_tool_synthesis=True)
        make_tool = session.agent.handlers["make_tool"]
        out = make_tool(
            name="boom",
            description="raises",
            source="def boom(text):\n    return 1/0\n",
        )
        self.assertIn("registered", out)
        boom = session.agent.handlers["boom"]
        result = boom(text="")
        self.assertIn("error:", result)

    def test_make_tool_registers_with_agent(self):
        self._set_responses([FakeResponse([FakeBlock(type="text", text="x")])])
        rt = self._new_runtime()
        session = rt.spawn(enable_tool_synthesis=True)
        make_tool = session.agent.handlers["make_tool"]
        make_tool(
            name="upper",
            description="uppercases input",
            source="def upper(text):\n    return text.upper()\n",
        )
        self.assertIn("upper", session.agent.handlers)
        self.assertIn("upper", session.agent._dynamic_tool_names)
        self.assertEqual(session.agent.handlers["upper"](text="ab"), "AB")


class TestDelegateBounded(FakeAnthropicTestCase):
    def test_delegate_at_depth_one_spawns_subagent(self):
        self._set_responses([
            # Parent turn 1: delegate tool_use.
            FakeResponse(
                [
                    FakeBlock(
                        type="tool_use",
                        id="t1",
                        name="delegate",
                        input={"task": "sub-task"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Subagent: a single text response.
            FakeResponse([FakeBlock(type="text", text="child-result")]),
            # Parent turn 2: final text.
            FakeResponse([FakeBlock(type="text", text="parent-final")]),
        ])
        rt = self._new_runtime()
        result = rt.execute(Task(goal="root", delegate_depth=1))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output, "parent-final")


class TestReflection(FakeAnthropicTestCase):
    def test_reflection_writes_lesson_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            self._set_responses([FakeResponse([FakeBlock(type="text", text="done")])])
            rt = self._new_runtime(memory=mem)
            rt.execute(Task(goal="hello", reflect=True))
            results = mem.search("lesson")
            self.assertGreaterEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
