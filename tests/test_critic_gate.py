"""Tests for the critic-gate integration in agi.Agent.

The gate logic is testable without API calls: build a bare Agent with
__new__ (skipping __init__), wire in a duck-typed mock critic, exercise
_apply_critic_gate. Real Critic integration is verified separately by
running `python -m learner.train_critic` and using the saved model.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import Agent


class FixedCritic:
    """Returns a fixed score regardless of input. Test fixture."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls: list[tuple[str, str]] = []

    def predict_proba(self, prompt: str, response: str) -> float:
        self.calls.append((prompt, response))
        return self.score


def _bare_agent(critic=None, threshold: float = 0.5, verbose: bool = False) -> Agent:
    """Build an Agent without calling __init__ (skips API client setup)."""
    a = Agent.__new__(Agent)
    a.critic = critic
    a.critic_threshold = threshold
    a.verbose = verbose
    return a


class TestCriticGate(unittest.TestCase):
    def test_no_critic_passes_through(self):
        a = _bare_agent(critic=None)
        out, score = a._apply_critic_gate("p", "r")
        self.assertEqual(out, "r")
        self.assertIsNone(score)

    def test_high_score_no_annotation(self):
        critic = FixedCritic(0.9)
        a = _bare_agent(critic=critic, threshold=0.5)
        out, score = a._apply_critic_gate("12+5=", "17")
        self.assertEqual(out, "17")
        self.assertEqual(score, 0.9)

    def test_low_score_appends_warning(self):
        critic = FixedCritic(0.2)
        a = _bare_agent(critic=critic, threshold=0.5)
        out, score = a._apply_critic_gate("12+5=", "wrong")
        self.assertIn("critic confidence", out)
        self.assertIn("0.20", out)
        self.assertTrue(out.startswith("wrong"))
        self.assertEqual(score, 0.2)

    def test_score_at_threshold_does_not_annotate(self):
        # Threshold is strict <; equal score should pass
        critic = FixedCritic(0.5)
        a = _bare_agent(critic=critic, threshold=0.5)
        out, score = a._apply_critic_gate("p", "r")
        self.assertEqual(out, "r")
        self.assertEqual(score, 0.5)

    def test_critic_receives_prompt_and_response(self):
        critic = FixedCritic(0.9)
        a = _bare_agent(critic=critic)
        a._apply_critic_gate("the prompt", "the response")
        self.assertEqual(critic.calls, [("the prompt", "the response")])

    def test_threshold_is_configurable(self):
        critic = FixedCritic(0.6)
        # Strict threshold: 0.6 < 0.8 → should annotate
        a_strict = _bare_agent(critic=critic, threshold=0.8)
        out_strict, _ = a_strict._apply_critic_gate("p", "r")
        self.assertIn("critic confidence", out_strict)

        # Relaxed threshold: 0.6 not less than 0.3 → no annotation
        a_relaxed = _bare_agent(critic=FixedCritic(0.6), threshold=0.3)
        out_relaxed, _ = a_relaxed._apply_critic_gate("p", "r")
        self.assertEqual(out_relaxed, "r")


try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed (optional [learner] extra)")
class TestRealCriticIntegration(unittest.TestCase):
    """End-to-end: train a real Critic on a tiny dataset, plug into the
    gate, verify it discriminates obvious good vs bad."""

    def test_trained_critic_distinguishes(self):
        from learner.critic import Critic, CriticConfig
        from learner.synth import addition_examples

        examples = addition_examples(n=400, max_n=20, seed=7)
        critic = Critic(CriticConfig(n_buckets=512, hidden=32))
        critic.fit(examples, epochs=15, lr=2e-3, verbose=False)

        a = _bare_agent(critic=critic, threshold=0.5)

        _, hedge_score = a._apply_critic_gate("12+5=", "I don't know")
        _, garbage_score = a._apply_critic_gate("12+5=", "asdf")
        _, correct_score = a._apply_critic_gate("12+5=", "17")

        # Real critic learns hedging and garbage are bad
        self.assertLess(hedge_score, 0.3)
        self.assertLess(garbage_score, 0.3)
        # Correct answer should score notably higher than the bad ones,
        # even if the absolute number is moderate (surface features can't
        # do arithmetic — the critic recognizes "looks like a plausible
        # numeric answer to an arithmetic prompt")
        self.assertGreater(correct_score, hedge_score + 0.2)
        self.assertGreater(correct_score, garbage_score + 0.2)


if __name__ == "__main__":
    unittest.main()
