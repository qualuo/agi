"""Tests for the reflection JSON parser. The Anthropic call itself is
not exercised — we drive the parser directly with model-output strings.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.reflection import _parse_lessons


class TestParseLessons(unittest.TestCase):
    def test_strict_json(self):
        text = '{"lessons": [{"text": "use list_dir before reading", "tags": ["filesystem"]}]}'
        lessons = _parse_lessons(text)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["text"], "use list_dir before reading")
        self.assertEqual(lessons[0]["tags"], ["filesystem"])

    def test_json_inside_prose(self):
        text = (
            "Sure, here is the JSON:\n"
            '{"lessons": [{"text": "always verify file writes by reading back"}]}\n'
            "End."
        )
        lessons = _parse_lessons(text)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["tags"], [])

    def test_empty_output(self):
        self.assertEqual(_parse_lessons(""), [])
        self.assertEqual(_parse_lessons('{"lessons": []}'), [])

    def test_caps_to_three(self):
        text = (
            '{"lessons": ['
            '{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}, {"text": "e"}'
            ']}'
        )
        lessons = _parse_lessons(text)
        self.assertEqual(len(lessons), 3)

    def test_skips_malformed_entries(self):
        text = '{"lessons": [{"text": "ok"}, "bad", {"no_text_field": true}]}'
        lessons = _parse_lessons(text)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["text"], "ok")

    def test_truncates_long_text(self):
        long = "x" * 1000
        lessons = _parse_lessons('{"lessons": [{"text": "' + long + '"}]}')
        self.assertEqual(len(lessons), 1)
        self.assertLessEqual(len(lessons[0]["text"]), 300)

    def test_unparseable_returns_empty(self):
        self.assertEqual(_parse_lessons("not json at all"), [])


if __name__ == "__main__":
    unittest.main()
