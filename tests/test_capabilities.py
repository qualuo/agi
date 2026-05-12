"""Tests for the CapabilityRegistry — observed-performance routing."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.capabilities import CapabilityRegistry
from agi.runtime import SessionConfig


def _make_registry(**kw):
    tmp = tempfile.mkdtemp()
    return CapabilityRegistry(path=Path(tmp) / "caps.jsonl", **kw)


class TestEmptyRegistry(unittest.TestCase):
    def test_recommend_returns_defaults_when_empty(self):
        reg = _make_registry(default_role="executor", default_model="claude-opus-4-7")
        rec = reg.recommend("sum two integers")
        self.assertEqual(rec.role, "executor")
        self.assertEqual(rec.model, "claude-opus-4-7")
        self.assertEqual(rec.confidence, 0.0)
        self.assertEqual(rec.evidence_count, 0)

    def test_stats_empty(self):
        reg = _make_registry()
        s = reg.stats()
        self.assertEqual(s["records"], 0)


class TestRecording(unittest.TestCase):
    def test_record_persists_to_disk(self):
        reg = _make_registry()
        reg.record(
            prompt="add 2 and 3",
            role="math_executor",
            model="claude-opus-4-7",
            skills_used=["add"],
            success=True,
            cost_usd=0.001,
            duration_seconds=1.5,
        )
        reg2 = CapabilityRegistry(path=reg.path)
        self.assertEqual(len(reg2.all()), 1)
        self.assertEqual(reg2.all()[0].role, "math_executor")

    def test_stats_summarizes_by_role(self):
        reg = _make_registry()
        for _ in range(3):
            reg.record(
                prompt="add two numbers",
                role="math",
                model="claude-opus-4-7",
                success=True,
                cost_usd=0.001,
                duration_seconds=1.0,
            )
        for _ in range(2):
            reg.record(
                prompt="write a poem",
                role="writer",
                model="claude-opus-4-7",
                success=False,
                cost_usd=0.005,
                duration_seconds=4.0,
            )
        s = reg.stats()
        self.assertEqual(s["records"], 5)
        self.assertAlmostEqual(s["success_rate"], 3 / 5)
        self.assertEqual(s["by_role"]["math"]["successes"], 3)
        self.assertAlmostEqual(s["by_role"]["writer"]["success_rate"], 0.0)


class TestRecommend(unittest.TestCase):
    def test_recommends_higher_success_role(self):
        reg = _make_registry()
        # Math role succeeds at math
        for _ in range(5):
            reg.record(
                prompt="add two integers please",
                role="math",
                model="claude-opus-4-7",
                success=True,
                cost_usd=0.001,
                duration_seconds=1.0,
            )
        # Writer role fails at math
        for _ in range(3):
            reg.record(
                prompt="add two integers please",
                role="writer",
                model="claude-opus-4-7",
                success=False,
                cost_usd=0.002,
                duration_seconds=2.0,
            )
        rec = reg.recommend("add two integers please")
        self.assertEqual(rec.role, "math")
        self.assertGreater(rec.confidence, 0.0)
        self.assertGreater(rec.expected_success_rate, 0.5)

    def test_unrelated_prompt_falls_back(self):
        reg = _make_registry(default_role="default", default_model="m")
        for _ in range(3):
            reg.record(
                prompt="add integers",
                role="math",
                model="m",
                success=True,
                cost_usd=0.001,
                duration_seconds=1.0,
            )
        # Unrelated tokens
        rec = reg.recommend("zzzzz qqqqq")
        self.assertEqual(rec.role, "default")
        self.assertEqual(rec.confidence, 0.0)

    def test_budget_penalty_redirects(self):
        reg = _make_registry()
        # Expensive but successful
        for _ in range(3):
            reg.record(
                prompt="solve task xyz",
                role="opus_planner",
                model="claude-opus-4-7",
                success=True,
                cost_usd=0.10,
                duration_seconds=2.0,
            )
        # Cheaper, slightly worse but ok
        for _ in range(3):
            reg.record(
                prompt="solve task xyz",
                role="haiku_executor",
                model="claude-haiku-4-5",
                success=True,
                cost_usd=0.001,
                duration_seconds=1.0,
            )
        # Tight budget should prefer cheaper
        rec = reg.recommend("solve task xyz", budget_usd=0.005)
        self.assertEqual(rec.role, "haiku_executor")

    def test_to_session_config_overrides_role_and_model(self):
        reg = _make_registry()
        for _ in range(3):
            reg.record(
                prompt="convert json data",
                role="parser",
                model="claude-haiku-4-5",
                success=True,
                cost_usd=0.001,
                duration_seconds=0.5,
            )
        rec = reg.recommend("convert json data")
        cfg = rec.to_session_config()
        self.assertEqual(cfg.role, "parser")
        self.assertEqual(cfg.model, "claude-haiku-4-5")

    def test_to_session_config_preserves_base_fields(self):
        reg = _make_registry()
        for _ in range(2):
            reg.record(
                prompt="do thing",
                role="r",
                model="m",
                success=True,
                cost_usd=0.001,
                duration_seconds=1.0,
            )
        rec = reg.recommend("do thing")
        base = SessionConfig(max_tokens=42, enable_web_search=False)
        cfg = rec.to_session_config(base)
        self.assertEqual(cfg.max_tokens, 42)
        self.assertFalse(cfg.enable_web_search)


class TestSkillsHint(unittest.TestCase):
    def test_skills_hint_extracted_from_successes(self):
        reg = _make_registry()
        for _ in range(4):
            reg.record(
                prompt="parse json blob",
                role="parser",
                model="m",
                skills_used=["json_parse", "validate"],
                success=True,
                cost_usd=0.001,
                duration_seconds=1.0,
            )
        rec = reg.recommend("parse json blob")
        self.assertIn("json_parse", rec.skills_hint)
        self.assertIn("validate", rec.skills_hint)


if __name__ == "__main__":
    unittest.main()
