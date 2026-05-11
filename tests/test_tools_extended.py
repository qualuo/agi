"""Tests for the new tools: reflect, save_skill, list_skills, delegate."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.skills import SkillLibrary
from agi.tools import make_tools, make_delegate_tool


class TestReflect(unittest.TestCase):
    def test_reflect_writes_lesson_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            schemas, handlers = make_tools(mem)
            self.assertIn("reflect", handlers)
            handlers["reflect"](lesson="always read the file before writing", tags=["files"])
            note = mem.recent(1)[0]
            self.assertIn("lesson", note.tags)
            self.assertIn("files", note.tags)


class TestSkillTools(unittest.TestCase):
    def test_skill_tools_only_present_when_library_provided(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            schemas_no, handlers_no = make_tools(mem)
            self.assertNotIn("save_skill", handlers_no)

            lib = SkillLibrary(path=Path(tmp) / "s")
            schemas, handlers = make_tools(mem, skills=lib)
            self.assertIn("save_skill", handlers)
            self.assertIn("list_skills", handlers)

    def test_save_skill_and_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=Path(tmp) / "s")
            mem = Memory(path=Path(tmp) / "m.jsonl")
            _, handlers = make_tools(mem, skills=lib)
            handlers["save_skill"](
                name="git-rebase",
                description="how to rebase a branch onto main",
                body="git fetch; git rebase origin/main; resolve; git push -f.",
                tags=["git"],
            )
            out = handlers["list_skills"](query="rebase onto main")
            self.assertIn("git-rebase", out)


class _FakeRuntime:
    """Minimal Runtime stand-in to test the delegate-tool wrapper."""

    def __init__(self):
        self.model = "claude-opus-4-7"
        self.calls: list[dict] = []

    def delegate(self, *, parent_id, task, role, budget, tags=None):
        self.calls.append(dict(parent_id=parent_id, task=task, role=role, budget=budget))
        return {
            "session_id": "child-id",
            "text": f"did: {task}",
            "stop_reason": "ok",
            "usage": {"cost_usd": 0.0123},
        }


class TestDelegateTool(unittest.TestCase):
    def test_delegate_tool_invokes_runtime_and_formats(self):
        rt = _FakeRuntime()
        schema, handler = make_delegate_tool(rt, parent_id="p1")
        self.assertEqual(schema["name"], "delegate")
        out = handler(task="check the build", role="executor", max_usd=0.1)
        self.assertEqual(len(rt.calls), 1)
        self.assertEqual(rt.calls[0]["parent_id"], "p1")
        self.assertEqual(rt.calls[0]["role"], "executor")
        self.assertIsNotNone(rt.calls[0]["budget"])
        self.assertIn("child child-id", out)
        self.assertIn("did: check the build", out)

    def test_delegate_tool_no_budget_when_max_usd_none(self):
        rt = _FakeRuntime()
        _, handler = make_delegate_tool(rt, parent_id="p1")
        handler(task="x", max_usd=None)
        self.assertIsNone(rt.calls[0]["budget"])


if __name__ == "__main__":
    unittest.main()
