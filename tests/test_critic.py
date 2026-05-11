"""Tests for the trace-quality critic (CPU, fast).

Covers featurizer behavior, model wiring, save/load, and that training
actually moves loss in the right direction. Does NOT verify accuracy on a
real distribution — that's what `python -m learner.train_critic` is for.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed (optional [learner] extra)")

from learner.critic import CharHashFeaturizer, Critic, CriticConfig
from learner.synth import addition_examples


class TestFeaturizer(unittest.TestCase):
    def test_output_shape(self):
        feat = CharHashFeaturizer(CriticConfig(n_buckets=512))
        v = feat.featurize("hello")
        self.assertEqual(v.shape, (512,))

    def test_l2_normalized(self):
        feat = CharHashFeaturizer(CriticConfig(n_buckets=512))
        v = feat.featurize("the quick brown fox jumps")
        self.assertAlmostEqual(v.norm().item(), 1.0, places=5)

    def test_empty_string_is_safe(self):
        feat = CharHashFeaturizer(CriticConfig(n_buckets=64))
        v = feat.featurize("")
        # All zeros, but normalize() shouldn't NaN — clamped denominator
        self.assertFalse(torch.isnan(v).any())

    def test_different_inputs_produce_different_features(self):
        feat = CharHashFeaturizer(CriticConfig(n_buckets=4096))
        v1 = feat.featurize("12+5=17")
        v2 = feat.featurize("12+5=99")
        self.assertGreater((v1 - v2).norm().item(), 0.0)

    def test_batch_matches_individual(self):
        feat = CharHashFeaturizer(CriticConfig(n_buckets=512))
        texts = ["a", "ab", "abc"]
        batched = feat.featurize_batch(texts)
        for i, t in enumerate(texts):
            self.assertTrue(torch.allclose(batched[i], feat.featurize(t)))


class TestCritic(unittest.TestCase):
    def test_predict_returns_probability(self):
        critic = Critic(CriticConfig(n_buckets=128, hidden=16))
        p = critic.predict_proba("12+5=", "17")
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_fit_reduces_loss(self):
        # Tiny synthetic set: critic should overfit and drive loss down sharply
        examples = addition_examples(n=200, max_n=20, seed=0)
        critic = Critic(CriticConfig(n_buckets=512, hidden=32))
        history = critic.fit(examples, epochs=10, lr=2e-3, batch_size=32, verbose=False)
        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertGreater(history["acc"][-1], 0.7)  # should fit well on small set

    def test_evaluate_returns_metrics(self):
        examples = addition_examples(n=100, max_n=20, seed=1)
        critic = Critic(CriticConfig(n_buckets=256, hidden=16))
        critic.fit(examples, epochs=5, verbose=False)
        m = critic.evaluate(examples)
        self.assertEqual(m["n"], 100)
        for k in ("accuracy", "precision", "recall"):
            self.assertGreaterEqual(m[k], 0.0)
            self.assertLessEqual(m[k], 1.0)

    def test_save_and_load_round_trip(self):
        critic = Critic(CriticConfig(n_buckets=128, hidden=16))
        examples = addition_examples(n=50, max_n=10, seed=2)
        critic.fit(examples, epochs=3, verbose=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            critic.save(path)
            loaded = Critic.load(path)

            for prompt, response, _ in examples[:5]:
                self.assertAlmostEqual(
                    critic.predict_proba(prompt, response),
                    loaded.predict_proba(prompt, response),
                    places=5,
                )


class TestSynth(unittest.TestCase):
    def test_addition_examples_balanced(self):
        examples = addition_examples(n=200, seed=0, pos_frac=0.5)
        n_pos = sum(y for _, _, y in examples)
        # Stochastic; should be roughly balanced
        self.assertGreater(n_pos, 60)
        self.assertLess(n_pos, 140)

    def test_positive_examples_have_correct_answer(self):
        examples = addition_examples(n=500, seed=0)
        for prompt, response, label in examples:
            if label == 1:
                # Parse "a+b="
                a, rest = prompt.split("+")
                b = rest.rstrip("=")
                self.assertEqual(int(response), int(a) + int(b))

    def test_negative_examples_are_wrong(self):
        examples = addition_examples(n=500, seed=0)
        for prompt, response, label in examples:
            if label == 0:
                a, rest = prompt.split("+")
                b = rest.rstrip("=")
                expected = str(int(a) + int(b))
                # Either response is not the bare correct number, or it's
                # the correct number plus extra junk
                self.assertNotEqual(response, expected)


if __name__ == "__main__":
    unittest.main()
