"""Reflection journal tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.reflection import Reflector


class TestReflector(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(path=Path(self._tmp.name) / "m.jsonl")

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_none_with_no_completer(self):
        r = Reflector(self.mem).reflect("task", "response")
        self.assertIsNone(r)

    def test_writes_lesson_to_memory(self):
        def fake(system: str, user: str) -> str:
            return "When deleting files, double-check the path."
        r = Reflector(self.mem, complete=fake).reflect("delete /tmp/foo", "deleted /tmp/foo", passed=True)
        self.assertIsNotNone(r)
        notes = self.mem.search("deleting files")
        self.assertEqual(len(notes), 1)
        self.assertIn("lesson", notes[0].tags)
        self.assertIn("passed", notes[0].tags)

    def test_none_response_skips_write(self):
        def fake(system: str, user: str) -> str:
            return "NONE"
        r = Reflector(self.mem, complete=fake).reflect("trivial", "ok")
        self.assertIsNone(r)
        self.assertEqual(len(self.mem.all()), 0)

    def test_failed_completer_does_not_crash(self):
        def boom(system: str, user: str) -> str:
            raise RuntimeError("network down")
        r = Reflector(self.mem, complete=boom).reflect("x", "y")
        self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()
