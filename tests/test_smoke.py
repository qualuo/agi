"""Smoke tests that don't hit the API.

Run with `python -m unittest tests/test_smoke.py` or `pytest tests/`.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.tools import make_tools


class TestMemory(unittest.TestCase):
    def test_save_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            mem.save("the user's favorite color is teal", tags=["pref"])
            mem.save("project foo uses postgres", tags=["proj"])
            results = mem.search("color")
            self.assertEqual(len(results), 1)
            self.assertIn("teal", results[0].text)

    def test_search_ranks_by_term_frequency(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            mem.save("apple banana")
            mem.save("apple apple cherry")
            results = mem.search("apple")
            self.assertEqual(len(results), 2)
            self.assertIn("cherry", results[0].text)  # higher count first

    def test_recent_returns_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            mem.save("first")
            mem.save("second")
            mem.save("third")
            results = mem.recent(2)
            self.assertEqual(results[0].text, "third")
            self.assertEqual(results[1].text, "second")

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            Memory(path=path).save("durable")
            results = Memory(path=path).search("durable")
            self.assertEqual(len(results), 1)


class TestTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.memory = Memory(path=self.tmp / "m.jsonl")
        self.schemas, self.handlers = make_tools(self.memory)

    def tearDown(self):
        self._tmp.cleanup()

    def test_schema_set_matches_handlers(self):
        schema_names = {s["name"] for s in self.schemas}
        self.assertEqual(schema_names, set(self.handlers))

    def test_write_then_read(self):
        path = str(self.tmp / "hello.txt")
        self.handlers["write_file"](path=path, content="hi")
        self.assertEqual(self.handlers["read_file"](path=path), "hi")

    def test_read_missing_file(self):
        result = self.handlers["read_file"](path=str(self.tmp / "nope.txt"))
        self.assertIn("does not exist", result)

    def test_run_bash_echoes(self):
        result = self.handlers["run_bash"](command="echo agi")
        self.assertIn("agi", result)
        self.assertIn("[exit 0]", result)

    def test_run_bash_timeout(self):
        result = self.handlers["run_bash"](command="sleep 5", timeout_seconds=1)
        self.assertIn("timed out", result)

    def test_memory_tools_round_trip(self):
        self.handlers["save_memory"](text="i like pizza", tags=["food"])
        result = self.handlers["search_memory"](query="pizza")
        self.assertIn("pizza", result)


if __name__ == "__main__":
    unittest.main()
