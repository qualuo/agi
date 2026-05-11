"""Tests for the new tools: skill I/O, make_tool, delegate.

`delegate` is exercised against a fake Agent factory so we can verify the
roll-up semantics without API calls.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi import events as ev
from agi.costs import Usage
from agi.memory import Memory
from agi.runtime import Budget, RunSpec, Runtime
from agi.skills import SkillLibrary
from agi.tools import make_tools


class FakeAgent:
    """Minimal stand-in matching what tools need."""
    def __init__(self, *, event_sink=None, model="claude-opus-4-7", effort="high", depth=0, **kw):
        self.event_sink = event_sink
        self.model = model
        self.effort = effort
        self.depth = depth
        self.usage = Usage()
        self.handlers: dict = {}
        self.tool_schemas: list = []

    def _rid(self):
        return self.event_sink.id if self.event_sink is not None else "local"

    def _emit(self, event):
        if self.event_sink is not None:
            self.event_sink.emit(event)

    def chat(self, prompt, max_iterations=25):
        # Fake some token usage and a deterministic reply
        class U:
            input_tokens = 50
            output_tokens = 30
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0
        self.usage.add(U())
        self._emit(ev.task_started(self._rid(), prompt))
        text = f"[fake:{prompt[:40]}]"
        self._emit(ev.task_completed(self._rid(), text, None))
        return text


class TestSkillTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=self._tmp.name)
        self.mem = Memory(path=Path(self._tmp.name) / "mem.jsonl")
        self.schemas, self.handlers = make_tools(self.mem, skill_library=self.lib)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_skill_persists(self):
        out = self.handlers["save_skill"](
            name="foo", description="when to use foo", body="do foo", tags=["bar"],
        )
        self.assertIn("foo", out)
        self.assertIsNotNone(self.lib.load("foo"))

    def test_search_skills_returns_matches(self):
        self.handlers["save_skill"](name="fizz", description="fizzbuzz recipe", body="...")
        out = self.handlers["search_skills"](query="fizzbuzz")
        self.assertIn("fizz", out)

    def test_search_skills_empty(self):
        self.assertEqual(self.handlers["search_skills"](query="nothing-here"), "no skills match")

    def test_load_skill_missing(self):
        self.assertIn("error", self.handlers["load_skill"](name="missing"))

    def test_no_skill_library_returns_error(self):
        _, handlers = make_tools(self.mem)  # no skill_library
        self.assertNotIn("save_skill", handlers)


class TestMakeTool(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(path=Path(self._tmp.name) / "mem.jsonl")
        self.agent = FakeAgent()
        self.schemas, self.handlers = make_tools(self.mem, current_agent=self.agent)
        # Mirror what Agent.__init__ does so the tool can see the live state.
        self.agent.tool_schemas = list(self.schemas)
        self.agent.handlers = self.handlers

    def tearDown(self):
        self._tmp.cleanup()

    def test_register_and_call_new_tool(self):
        code = "def square(n):\n    return str(n * n)\n"
        out = self.handlers["make_tool"](
            name="square",
            description="Square a number.",
            code=code,
            input_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
        )
        self.assertIn("registered", out)
        self.assertIn("square", self.agent.handlers)
        self.assertEqual(self.agent.handlers["square"](n=5), "25")

    def test_reject_invalid_identifier(self):
        out = self.handlers["make_tool"](name="123nope", description="d", code="def x(): return 1")
        self.assertIn("error", out)

    def test_reject_duplicate_name(self):
        out = self.handlers["make_tool"](
            name="read_file", description="d", code="def read_file(): return ''"
        )
        self.assertIn("already exists", out)

    def test_reject_syntax_error(self):
        out = self.handlers["make_tool"](name="bad", description="d", code="def bad(:\n")
        self.assertIn("syntax error", out)

    def test_reject_wrong_function_name(self):
        out = self.handlers["make_tool"](
            name="foo", description="d", code="def bar(): return 1"
        )
        self.assertIn("exactly one function", out)

    def test_reject_multiple_functions(self):
        out = self.handlers["make_tool"](
            name="foo", description="d", code="def foo(): pass\ndef bar(): pass"
        )
        self.assertIn("exactly one function", out)


class TestDelegateTool(unittest.TestCase):
    def test_delegate_spawns_child_and_rolls_up_cost(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rt = Runtime(
            registry_dir=Path(tmp.name) / "runs",
            agent_factory=FakeAgent,
        )
        # Build a parent agent wired to the runtime.
        parent = FakeAgent()

        class _RunStub:
            id = "parent"
            child_run_ids: list = []
            def emit(self, e): self.events_seen.append(e)  # type: ignore[attr-defined]
            events_seen: list = []

        parent.event_sink = _RunStub()
        _, handlers = make_tools(
            Memory(path=Path(tmp.name) / "mem.jsonl"),
            runtime=rt,
            current_agent=parent,
        )
        result = handlers["delegate"](prompt="sub-task", role="executor")
        self.assertIn("subagent", result)
        self.assertIn("[fake:sub-task]", result)
        # Cost rolled up into parent
        self.assertGreater(parent.usage.input_tokens, 0)
        # Parent received child_run_started + child_run_completed events
        types = [e.type for e in parent.event_sink.events_seen]
        self.assertIn("child_run_started", types)
        self.assertIn("child_run_completed", types)

    def test_delegate_requires_runtime(self):
        _, handlers = make_tools(Memory(path=Path(tempfile.mkdtemp()) / "m"))
        self.assertNotIn("delegate", handlers)


if __name__ == "__main__":
    unittest.main()
