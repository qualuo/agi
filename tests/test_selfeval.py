"""Tests for SelfEvalBank — agent-generated regression suite."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.selfeval import EvalItem, SelfEvalBank
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent


def _bank() -> SelfEvalBank:
    tmp = Path(tempfile.mkdtemp())
    return SelfEvalBank(path=tmp / "se.jsonl")


class TestItemPredicate(unittest.TestCase):
    def test_substring_only(self):
        item = EvalItem(id="x", prompt="p", expect_substring="hello")
        self.assertTrue(item.predicate()("say hello world"))
        self.assertFalse(item.predicate()("nothing here"))

    def test_substring_case_insensitive(self):
        item = EvalItem(id="x", prompt="p", expect_substring="Hello")
        self.assertTrue(item.predicate()("HELLO there"))

    def test_regex_match(self):
        item = EvalItem(id="x", prompt="p", expect_regex=r"\b42\b")
        self.assertTrue(item.predicate()("the answer is 42"))
        self.assertFalse(item.predicate()("the answer is 420"))

    def test_min_length(self):
        item = EvalItem(id="x", prompt="p", expect_min_length=10)
        self.assertFalse(item.predicate()("short"))
        self.assertTrue(item.predicate()("this is plenty long"))

    def test_combined_predicates(self):
        item = EvalItem(id="x", prompt="p", expect_substring="ok",
                        expect_min_length=5)
        self.assertFalse(item.predicate()("ok"))      # too short
        self.assertFalse(item.predicate()("longer"))  # missing substring
        self.assertTrue(item.predicate()("ok long enough"))


class TestBankAddDedup(unittest.TestCase):
    def test_add_persists(self):
        bank = _bank()
        item = bank.add(prompt="q", expect_substring="a")
        self.assertIsNotNone(item)
        bank2 = SelfEvalBank(path=bank.path)
        self.assertEqual(len(bank2.all()), 1)

    def test_add_dedupes(self):
        bank = _bank()
        bank.add(prompt="q", expect_substring="a")
        again = bank.add(prompt="q", expect_substring="a")
        self.assertIsNone(again)

    def test_add_with_no_predicate_rejected(self):
        bank = _bank()
        self.assertIsNone(bank.add(prompt="q"))


class TestAutoMine(unittest.TestCase):
    def test_mines_when_critic_confident(self):
        bank = _bank()
        item = bank.auto_mine(
            prompt="What is 2+2?", final_text="The answer is 4.",
            critic_score=0.9,
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.source, "automatic")

    def test_skips_when_critic_low(self):
        bank = _bank()
        item = bank.auto_mine(
            prompt="bad", final_text="garbage", critic_score=0.1,
        )
        self.assertIsNone(item)

    def test_skips_when_no_text(self):
        bank = _bank()
        item = bank.auto_mine(prompt="q", final_text="", critic_score=0.95)
        self.assertIsNone(item)


class TestBankRun(unittest.TestCase):
    def test_run_updates_counters(self):
        bank = _bank()
        bank.add(prompt="q1", expect_substring="ok")
        bank.add(prompt="q2", expect_substring="missing")

        def runner(item):
            text = "ok response"
            return item.predicate()(text), text, 0.0

        report = bank.run(runner)
        self.assertEqual(report.total, 2)
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.failed, 1)
        self.assertAlmostEqual(report.pass_rate, 0.5)
        # Counters persisted
        items = bank.all()
        self.assertTrue(any(i.passes == 1 for i in items))
        self.assertTrue(any(i.passes == 0 and i.runs == 1 for i in items))

    def test_run_handles_runner_exceptions(self):
        bank = _bank()
        bank.add(prompt="q", expect_substring="ok")

        def runner(item):
            raise RuntimeError("boom")

        report = bank.run(runner)
        self.assertEqual(report.passed, 0)
        self.assertEqual(report.failed, 1)


class TestGate(unittest.TestCase):
    def test_gate_blocks_regression(self):
        bank = _bank()
        bank.add(prompt="q", expect_substring="must_have")

        def good_runner(item):
            return True, "this has must_have inside", 0.0

        def bad_runner(item):
            return False, "missing", 0.0

        ok, _ = bank.gate_promotion(good_runner, baseline_pass_rate=1.0)
        self.assertTrue(ok)
        ok, _ = bank.gate_promotion(bad_runner, baseline_pass_rate=1.0)
        self.assertFalse(ok)


class TestRuntimeRunner(unittest.TestCase):
    def test_runtime_runner_drives_chat(self):
        tmp = Path(tempfile.mkdtemp())
        rt = Runtime(
            memory=Memory(path=tmp / "m.jsonl"),
            skills=SkillLibrary(path=tmp / "skills"),
            agent_factory=FakeAgent,
        )
        bank = _bank()
        bank.add(prompt="hi", expect_substring="ok")  # FakeAgent returns "ok"
        report = bank.run(bank.runtime_runner(rt))
        self.assertEqual(report.total, 1)
        self.assertEqual(report.passed, 1)
        # Cost was tracked through the runtime
        self.assertGreater(report.total_cost_usd, 0.0)


class TestStats(unittest.TestCase):
    def test_stats_summarizes_bank(self):
        bank = _bank()
        bank.add(prompt="q1", expect_substring="a")
        bank.add(prompt="q2", expect_substring="b", source="explicit")
        s = bank.stats()
        self.assertEqual(s["items"], 2)
        self.assertEqual(s["by_source"]["automatic"], 1)
        self.assertEqual(s["by_source"]["explicit"], 1)


if __name__ == "__main__":
    unittest.main()
