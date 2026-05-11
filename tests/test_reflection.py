"""Tests for the reflection JSON parser. The Anthropic call itself is
not exercised — we drive the parser directly with model-output strings.
The wiring into Session is tested separately with a fake reflector.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.reflection import Reflector, _parse_lessons


class TestParseLessons(unittest.TestCase):
    def test_strict_json(self) -> None:
        text = '{"lessons": [{"text": "use list_dir before reading", "tags": ["filesystem"]}]}'
        lessons = _parse_lessons(text)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["text"], "use list_dir before reading")
        self.assertEqual(lessons[0]["tags"], ["filesystem"])

    def test_json_inside_prose(self) -> None:
        text = (
            "Sure, here is the JSON:\n"
            '{"lessons": [{"text": "always verify file writes by reading back"}]}\n'
            "End."
        )
        lessons = _parse_lessons(text)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["tags"], [])

    def test_empty_outputs(self) -> None:
        self.assertEqual(_parse_lessons(""), [])
        self.assertEqual(_parse_lessons("   "), [])
        self.assertEqual(_parse_lessons('{"lessons": []}'), [])

    def test_missing_or_wrong_shape(self) -> None:
        self.assertEqual(_parse_lessons('{"other": []}'), [])
        self.assertEqual(_parse_lessons('{"lessons": "string-not-list"}'), [])
        self.assertEqual(_parse_lessons("garbage"), [])

    def test_caps_to_three(self) -> None:
        text = (
            '{"lessons": ['
            '{"text": "a"},'
            '{"text": "b"},'
            '{"text": "c"},'
            '{"text": "d"},'
            '{"text": "e"}'
            ']}'
        )
        self.assertEqual(len(_parse_lessons(text)), 3)

    def test_drops_invalid_entries(self) -> None:
        text = (
            '{"lessons": ['
            '{"text": "good"},'
            '{"text": ""},'
            '"not-a-dict",'
            '{"text": null}'
            ']}'
        )
        lessons = _parse_lessons(text)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["text"], "good")

    def test_truncates_long_text(self) -> None:
        long = "x" * 1000
        out = _parse_lessons(f'{{"lessons": [{{"text": "{long}"}}]}}')
        self.assertEqual(len(out[0]["text"]), 300)

    def test_caps_tags(self) -> None:
        tags = ["t" + str(i) for i in range(20)]
        import json as _json
        text = _json.dumps({"lessons": [{"text": "x", "tags": tags}]})
        out = _parse_lessons(text)
        self.assertEqual(len(out[0]["tags"]), 6)


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class TestReflectorWithFakeClient(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.memory = Memory(path=Path(self._tmp.name) / "m.jsonl")
        self.client = MagicMock()
        self.reflector = Reflector(self.memory, client=self.client)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_saves_lessons_to_memory(self) -> None:
        self.client.messages.create.return_value = _FakeResp(
            '{"lessons": [{"text": "always read back files you wrote", "tags": ["fs"]}]}'
        )
        result = self.reflector.reflect(
            user_prompt="write a file",
            final_text="wrote 5 bytes to /tmp/x",
            tools_used=["write_file"],
        )
        self.assertEqual(result.lessons_saved, 1)
        self.assertIsNone(result.error)
        notes = self.memory.search("read back")
        self.assertEqual(len(notes), 1)
        self.assertIn("lesson", notes[0].tags)
        self.assertIn("fs", notes[0].tags)

    def test_empty_final_text_skips_call(self) -> None:
        result = self.reflector.reflect(user_prompt="x", final_text="")
        self.assertEqual(result.lessons_saved, 0)
        self.client.messages.create.assert_not_called()

    def test_api_error_falls_open(self) -> None:
        self.client.messages.create.side_effect = RuntimeError("network")
        result = self.reflector.reflect(user_prompt="x", final_text="y")
        self.assertEqual(result.lessons_saved, 0)
        self.assertIsNotNone(result.error)
        self.assertIn("network", result.error)


if __name__ == "__main__":
    unittest.main()
