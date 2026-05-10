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

from agi.costs import Usage, PRICING
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
        self.registry = make_tools(self.memory)
        self.schemas = self.registry.schemas
        self.handlers = self.registry.handlers

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

    def test_make_tool_is_present_by_default(self):
        self.assertIn("make_tool", self.handlers)

    def test_dispatch_records_call_log(self):
        self.handlers["save_memory"](text="x")
        # Direct call doesn't log; dispatch does.
        out, is_error = self.registry.dispatch("save_memory", {"text": "y"})
        self.assertFalse(is_error)
        self.assertIn("save_memory", self.registry.call_log)


class FakeResponseUsage:
    """Mimics anthropic.types.Usage for the Usage.add() interface."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestUsage(unittest.TestCase):
    def test_add_accumulates(self):
        u = Usage()
        u.add(FakeResponseUsage(input_tokens=100, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=0))
        u.add(FakeResponseUsage(input_tokens=50, output_tokens=80, cache_creation_input_tokens=10, cache_read_input_tokens=20))
        self.assertEqual(u.input_tokens, 150)
        self.assertEqual(u.output_tokens, 280)
        self.assertEqual(u.cache_creation_input_tokens, 10)
        self.assertEqual(u.cache_read_input_tokens, 20)
        self.assertEqual(u.turns, 2)

    def test_add_handles_missing_attrs(self):
        u = Usage()
        u.add(FakeResponseUsage(input_tokens=10))  # no output_tokens etc.
        self.assertEqual(u.input_tokens, 10)
        self.assertEqual(u.output_tokens, 0)

    def test_add_handles_none_values(self):
        # The SDK can return None for cache fields when not used
        u = Usage()
        u.add(FakeResponseUsage(input_tokens=10, output_tokens=20, cache_creation_input_tokens=None, cache_read_input_tokens=None))
        self.assertEqual(u.cache_creation_input_tokens, 0)
        self.assertEqual(u.cache_read_input_tokens, 0)

    def test_cost_opus_4_7(self):
        # 1M input + 1M output on opus-4-7 = $5 + $25 = $30
        u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(u.cost_usd("claude-opus-4-7"), 30.0)

    def test_cost_cache_read_is_cheap(self):
        # 1M cache reads on opus-4-7 = $5 * 0.1 = $0.50
        u = Usage(cache_read_input_tokens=1_000_000)
        self.assertAlmostEqual(u.cost_usd("claude-opus-4-7"), 0.50)

    def test_cost_cache_write_premium(self):
        # 1M cache writes on opus-4-7 = $5 * 1.25 = $6.25
        u = Usage(cache_creation_input_tokens=1_000_000)
        self.assertAlmostEqual(u.cost_usd("claude-opus-4-7"), 6.25)

    def test_cost_unknown_model_returns_zero(self):
        u = Usage(input_tokens=1000)
        self.assertEqual(u.cost_usd("unknown-model"), 0.0)

    def test_format_includes_dollar_amount(self):
        u = Usage(input_tokens=1000, output_tokens=500)
        out = u.format("claude-opus-4-7")
        self.assertIn("1,000 in", out)
        self.assertIn("500 out", out)
        self.assertIn("$", out)


if __name__ == "__main__":
    unittest.main()
