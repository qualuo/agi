"""Tests for the coordination tools (delegate + reflect).

`delegate` requires building a real Agent to test end-to-end (since it
spawns a child Agent), so we exercise just the schema + handler shape.
The reflection tool is fully exercisable without API calls.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.coordination import (
    ROLE_PROMPTS,
    make_coordination_tools,
    make_delegate_tool,
    make_reflection_tool,
)
from agi.costs import Usage
from agi.memory import Memory


class TestReflectTool(unittest.TestCase):
    def test_writes_lesson_tagged_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            schemas, handlers = make_reflection_tool(mem)
            self.assertEqual([s["name"] for s in schemas], ["reflect"])
            out = handlers["reflect"](
                task="set up postgres",
                what_worked="docker compose with named volume",
                lesson="prefer named volumes over bind mounts",
            )
            self.assertIn("recorded lesson", out)
            notes = mem.search("postgres")
            self.assertEqual(len(notes), 1)
            self.assertIn("lesson", notes[0].tags)
            self.assertIn("docker compose", notes[0].text)

    def test_reflect_handles_minimal_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            _, handlers = make_reflection_tool(mem)
            out = handlers["reflect"](task="x", what_worked="y")
            self.assertIn("recorded", out)


class TestDelegateTool(unittest.TestCase):
    def test_schema_lists_known_roles(self):
        usage = Usage()
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            schemas, handlers = make_delegate_tool(
                parent_usage=usage,
                parent_memory=mem,
                parent_model="claude-haiku-4-5",
            )
        self.assertEqual([s["name"] for s in schemas], ["delegate"])
        desc = schemas[0]["description"]
        for role in ("planner", "executor", "critic", "researcher", "summarizer"):
            self.assertIn(role, desc)

    def test_max_depth_returns_no_tool(self):
        usage = Usage()
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            schemas, handlers = make_delegate_tool(
                parent_usage=usage,
                parent_memory=mem,
                parent_model="claude-haiku-4-5",
                max_depth=2,
                current_depth=2,
            )
        self.assertEqual(schemas, [])
        self.assertEqual(handlers, {})

    def test_role_prompts_are_distinct(self):
        seen = set()
        for k, v in ROLE_PROMPTS.items():
            self.assertNotIn(v, seen)
            seen.add(v)


class TestMakeCoordinationTools(unittest.TestCase):
    def test_combines_delegate_and_reflect(self):
        usage = Usage()
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            schemas, handlers = make_coordination_tools(
                parent_usage=usage, parent_memory=mem, parent_model="claude-haiku-4-5"
            )
        names = {s["name"] for s in schemas}
        self.assertEqual(names, {"delegate", "reflect"})
        self.assertEqual(set(handlers.keys()), {"delegate", "reflect"})

    def test_can_disable_each(self):
        usage = Usage()
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            schemas, _ = make_coordination_tools(
                parent_usage=usage,
                parent_memory=mem,
                parent_model="claude-haiku-4-5",
                enable_delegate=False,
            )
            self.assertEqual([s["name"] for s in schemas], ["reflect"])


if __name__ == "__main__":
    unittest.main()
