"""Tests for agi.reflect."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.reflect import parse_lesson, should_reflect


class TestParseLesson(unittest.TestCase):
    def test_no_lesson_returns_none(self) -> None:
        self.assertIsNone(parse_lesson("NO_LESSON"))
        self.assertIsNone(parse_lesson("no_lesson"))
        self.assertIsNone(parse_lesson(""))

    def test_strips_prefix(self) -> None:
        self.assertEqual(parse_lesson("Lesson: be terse"), "be terse")
        self.assertEqual(parse_lesson("lesson: validate inputs"), "validate inputs")

    def test_truncates_long_lessons(self) -> None:
        long = "x" * 500
        out = parse_lesson(long)
        self.assertLess(len(out), 245)


class TestShouldReflect(unittest.TestCase):
    def test_skips_trivial_round_trip(self) -> None:
        self.assertFalse(should_reflect(n_turns=1, n_tool_calls=0, response_chars=10))

    def test_runs_on_meaningful_task(self) -> None:
        self.assertTrue(should_reflect(n_turns=3, n_tool_calls=2, response_chars=200))

    def test_skips_very_short_response(self) -> None:
        self.assertFalse(should_reflect(n_turns=4, n_tool_calls=2, response_chars=10))


if __name__ == "__main__":
    unittest.main()
