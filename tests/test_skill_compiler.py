"""Tests for the skill compiler proposal parser and trace summarizer."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skill_compiler import _parse_proposals, _summarize_trace, collect_summaries
from agi.skills import SkillLibrary


class FakeTrace:
    def __init__(self, messages, *, metadata=None):
        self.messages = messages
        self.metadata = metadata or {}


class TestSummarize(unittest.TestCase):
    def test_summary_includes_user_first_and_assistant_last(self):
        t = FakeTrace([
            {"role": "user", "content": "summarize foo.py"},
            {"role": "assistant", "content": "ok let me read it"},
            {"role": "assistant", "content": "summary: foo.py defines bar()"},
        ])
        s = _summarize_trace(t)
        self.assertIn("summarize foo.py", s)
        self.assertIn("foo.py defines bar", s)

    def test_summary_with_block_messages(self):
        t = FakeTrace([
            {"role": "user", "content": "do x"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "name": "run_bash", "input": {"command": "ls"}},
            ]},
        ])
        s = _summarize_trace(t)
        self.assertIn("run_bash", s)


class TestParseProposals(unittest.TestCase):
    def test_strict_parse(self):
        text = """
        {"proposals": [
          {
            "title": "Read-then-edit",
            "when": "Editing a file you have not read this session",
            "procedure": "1. read the file\\n2. propose the edit\\n3. write the change",
            "failure_modes": "stale assumptions about file contents",
            "triggers": ["edit", "file"]
          }
        ]}
        """
        props = _parse_proposals(text)
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0].title, "Read-then-edit")
        self.assertEqual(props[0].triggers, ["edit", "file"])

    def test_skips_invalid_entries(self):
        text = '{"proposals": [{"title": "ok", "when": "x", "procedure": "y"}, {"title": "no when"}]}'
        props = _parse_proposals(text)
        self.assertEqual(len(props), 1)

    def test_caps_to_five(self):
        proposals = [
            {"title": f"t{i}", "when": "x", "procedure": "y"} for i in range(8)
        ]
        text = '{"proposals": ' + str(proposals).replace("'", '"') + '}'
        props = _parse_proposals(text)
        self.assertEqual(len(props), 5)

    def test_unparseable(self):
        self.assertEqual(_parse_proposals(""), [])
        self.assertEqual(_parse_proposals("no json"), [])

    def test_commit_writes_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=Path(tmp))
            text = '{"proposals": [{"title": "T", "when": "x", "procedure": "y", "triggers": ["a"]}]}'
            props = _parse_proposals(text)
            skill = props[0].commit(lib)
            self.assertEqual(skill.title, "T")
            self.assertIsNotNone(lib.get(skill.id))


class TestCollect(unittest.TestCase):
    def test_collects_recent(self):
        traces = [
            FakeTrace([{"role": "user", "content": f"task {i}"}, {"role": "assistant", "content": f"answer {i}"}])
            for i in range(50)
        ]
        summaries = collect_summaries(traces, max_traces=10)
        self.assertEqual(len(summaries), 10)
        self.assertIn("task 49", summaries[-1])
        self.assertIn("task 40", summaries[0])


if __name__ == "__main__":
    unittest.main()
