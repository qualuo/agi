"""Tests for the contextual Thompson-sampling policy router."""
from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.capabilities import CapabilityRegistry
from agi.policy import (
    Arm,
    ArmPosterior,
    PolicyRouter,
    RoutingDecision,
    recommend_with_policy,
)


def _registry(tmp: Path) -> CapabilityRegistry:
    return CapabilityRegistry(path=tmp / "caps.jsonl")


class TestArmPosterior(unittest.TestCase):
    def test_priors(self):
        p = ArmPosterior(arm=Arm(role="executor", model="m"))
        # With a uniform prior, expected success = 0.5 and expected cost = 0.
        self.assertAlmostEqual(p.expected_success(), 0.5)
        self.assertEqual(p.expected_cost(), 0.0)
        self.assertEqual(p.evidence_count, 0)

    def test_update_moves_posterior(self):
        p = ArmPosterior(arm=Arm(role="r", model="m"))
        for _ in range(10):
            p.update(success=True, cost_usd=0.01)
        self.assertEqual(p.evidence_count, 10)
        self.assertGreater(p.expected_success(), 0.9)
        self.assertAlmostEqual(p.expected_cost(), 0.01, places=5)

    def test_sample_in_unit_interval(self):
        p = ArmPosterior(arm=Arm(role="r", model="m"))
        rng = random.Random(0)
        for _ in range(50):
            s = p.sample(rng)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class TestPolicyRouterSeed(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.reg = _registry(self.tmp)

    def test_empty_seed_uses_defaults(self):
        router = PolicyRouter(self.reg)
        self.assertGreater(len(router.arms), 0)
        posteriors = router.seed("plan and execute X")
        self.assertEqual(len(posteriors), len(router.arms))
        for p in posteriors.values():
            self.assertEqual(p.evidence_count, 0)

    def test_seed_folds_similar_records(self):
        for _ in range(8):
            self.reg.record(
                prompt="summarize the docs about lora adapters",
                role="writer",
                model="claude-sonnet-4-6",
                success=True,
                cost_usd=0.01,
                duration_seconds=1.0,
            )
        for _ in range(2):
            self.reg.record(
                prompt="summarize the docs about lora adapters",
                role="executor",
                model="claude-opus-4-7",
                success=False,
                cost_usd=0.1,
                duration_seconds=1.0,
            )
        router = PolicyRouter(self.reg, epsilon=0.0)
        posteriors = router.seed("summarize the lora adapter docs")
        writer = posteriors[("writer", "claude-sonnet-4-6", "high")]
        exec_ = posteriors[("executor", "claude-opus-4-7", "high")]
        self.assertGreater(writer.evidence_count, 0)
        self.assertGreater(exec_.evidence_count, 0)
        # Writer has 8 successes vs executor's 2 failures — expected success higher.
        self.assertGreater(writer.expected_success(), exec_.expected_success())


class TestPolicyRouterDecide(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.reg = _registry(self.tmp)

    def test_decide_returns_arm(self):
        router = PolicyRouter(self.reg, epsilon=0.0)
        d = router.decide("write a haiku")
        self.assertIsInstance(d, RoutingDecision)
        self.assertIn((d.arm.role, d.arm.model, d.arm.effort),
                      {a.key() for a in router.arms})

    def test_force_arm(self):
        router = PolicyRouter(self.reg)
        target = Arm(role="planner", model="claude-opus-4-7", effort="high")
        d = router.decide("any prompt", force_arm=target)
        self.assertEqual(d.arm.key(), target.key())
        self.assertFalse(d.explored)

    def test_epsilon_one_always_explores(self):
        # Force a tiny rng so we can confirm an exploratory pick happens.
        router = PolicyRouter(self.reg, epsilon=1.0)
        d = router.decide("anything")
        self.assertTrue(d.explored)

    def test_learning_converges_to_better_arm(self):
        # Two arms; one is strictly better. Run the loop and check that
        # picks concentrate on the winning arm.
        for _ in range(50):
            self.reg.record(
                prompt="alpha beta gamma",
                role="winner",
                model="claude-sonnet-4-6",
                success=True,
                cost_usd=0.005,
                duration_seconds=1.0,
            )
        for _ in range(50):
            self.reg.record(
                prompt="alpha beta gamma",
                role="loser",
                model="claude-sonnet-4-6",
                success=False,
                cost_usd=0.005,
                duration_seconds=1.0,
            )
        router = PolicyRouter(
            self.reg,
            arms=[
                Arm(role="winner", model="claude-sonnet-4-6"),
                Arm(role="loser", model="claude-sonnet-4-6"),
            ],
            epsilon=0.0,
        )
        # Deterministic rng so the test is stable.
        router._rng = random.Random(42)
        wins = 0
        for _ in range(40):
            d = router.decide("alpha beta gamma")
            if d.arm.role == "winner":
                wins += 1
        # Should pick the winner most of the time (allow some randomness).
        self.assertGreater(wins, 30)

    def test_cost_weight_penalises_expensive_arms(self):
        # Two arms with similar success but very different cost.
        for _ in range(20):
            self.reg.record(
                prompt="cost test", role="cheap", model="claude-sonnet-4-6",
                success=True, cost_usd=0.001, duration_seconds=1.0,
            )
            self.reg.record(
                prompt="cost test", role="expensive", model="claude-opus-4-7",
                success=True, cost_usd=0.5, duration_seconds=1.0,
            )
        router = PolicyRouter(
            self.reg,
            arms=[
                Arm(role="cheap", model="claude-sonnet-4-6"),
                Arm(role="expensive", model="claude-opus-4-7"),
            ],
            epsilon=0.0,
            cost_weight=10.0,
        )
        router._rng = random.Random(0)
        picks = {"cheap": 0, "expensive": 0}
        for _ in range(40):
            d = router.decide("cost test")
            picks[d.arm.role] += 1
        self.assertGreater(picks["cheap"], picks["expensive"])

    def test_observe_persists_to_registry(self):
        router = PolicyRouter(self.reg, epsilon=0.0)
        decision = router.decide("hello world")
        rec = router.observe(
            prompt="hello world", decision=decision, success=True,
            cost_usd=0.01, duration_seconds=0.5,
        )
        self.assertEqual(rec.role, decision.arm.role)
        self.assertEqual(rec.model, decision.arm.model)
        self.assertEqual(len(self.reg.all()), 1)


class TestRecommendShim(unittest.TestCase):
    def test_recommend_with_policy_returns_recommendation(self):
        tmp = Path(tempfile.mkdtemp())
        reg = _registry(tmp)
        for _ in range(5):
            reg.record(
                prompt="some recurring prompt",
                role="executor", model="claude-opus-4-7",
                success=True, cost_usd=0.02, duration_seconds=1.0,
            )
        rec = recommend_with_policy(reg, "some recurring prompt", budget_usd=1.0)
        self.assertIsNotNone(rec.role)
        self.assertGreaterEqual(rec.confidence, 0.0)
        self.assertLessEqual(rec.confidence, 1.0)


class TestPolicyStats(unittest.TestCase):
    def test_stats_with_prompt_includes_arms(self):
        tmp = Path(tempfile.mkdtemp())
        reg = _registry(tmp)
        router = PolicyRouter(reg)
        s = router.stats(prompt="hello")
        self.assertIn("arms", s)
        self.assertGreater(len(s["arms"]), 0)

    def test_stats_marginal_aggregates_records(self):
        tmp = Path(tempfile.mkdtemp())
        reg = _registry(tmp)
        for _ in range(3):
            reg.record(prompt="x", role="r", model="m", success=True,
                       cost_usd=0.01, duration_seconds=1.0)
        router = PolicyRouter(reg)
        s = router.stats()
        self.assertEqual(len(s["arms"]), 1)
        self.assertEqual(s["arms"][0]["n"], 3)
        self.assertEqual(s["arms"][0]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
