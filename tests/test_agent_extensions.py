"""Tests for the optional Agent extensions (skills, delegate).

These don't make API calls; they only check the wiring around the chat
loop — that the system prompt is augmented with skills, that the delegate
tool is registered when requested, and that selected skills surface on
the agent for traceability.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import Agent
from agi.memory import Memory
from learner.skills import SkillLibrary


class _NoApiAgent(Agent):
    """Agent subclass that doesn't initialize an Anthropic client."""

    def __init__(self, **kwargs) -> None:
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = object()
            super().__init__(**kwargs)


class TestSkillSelection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=self._tmp.name)
        self.lib.write(
            name="solve-quadratic",
            description="Solve quadratic equations using the quadratic formula.",
            body="Apply x = (-b ± sqrt(b^2 - 4ac)) / 2a.",
            tags=["math"],
        )
        self.lib.write(
            name="fetch-and-summarize",
            description="Fetch a URL and summarize the page.",
            body="Use web_fetch then write a one-paragraph summary.",
            tags=["web"],
        )
        memory = Memory(path=Path(self._tmp.name) / "m.jsonl")
        self.agent = _NoApiAgent(memory=memory, skills=self.lib, verbose=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_relevant_skill_block_includes_match(self):
        block, names = self.agent._select_skills("Solve a quadratic equation for me")
        self.assertEqual(names, ["solve-quadratic"])
        self.assertIn("Skill: solve-quadratic", block)
        self.assertIn("quadratic formula", block)

    def test_irrelevant_query_returns_no_block(self):
        block, names = self.agent._select_skills("xyzzy plover frobnicate")
        self.assertEqual(block, "")
        self.assertEqual(names, [])

    def test_no_library_no_block(self):
        memory = Memory(path=Path(self._tmp.name) / "m2.jsonl")
        agent = _NoApiAgent(memory=memory, skills=None, verbose=False)
        block, names = agent._select_skills("Solve a quadratic")
        self.assertEqual(block, "")
        self.assertEqual(names, [])


class TestDelegateTool(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_delegate_registered_when_enabled(self):
        memory = Memory(path=Path(self._tmp.name) / "m.jsonl")
        agent = _NoApiAgent(memory=memory, enable_delegate=True, verbose=False)
        names = {s.get("name") for s in agent.tool_schemas}
        self.assertIn("delegate", names)
        self.assertIn("delegate", agent.handlers)

    def test_delegate_not_registered_by_default(self):
        memory = Memory(path=Path(self._tmp.name) / "m2.jsonl")
        agent = _NoApiAgent(memory=memory, verbose=False)
        names = {s.get("name") for s in agent.tool_schemas}
        self.assertNotIn("delegate", names)


if __name__ == "__main__":
    unittest.main()
