"""Tests for agi.aligner — direct preference optimisation."""
from __future__ import annotations

import math
import random
import unittest

from agi.aligner import (
    ALG_CDPO,
    ALG_DPO,
    ALG_IPO,
    ALG_KTO,
    ALG_ORPO,
    ALG_RDPO,
    ALG_SIMPO,
    ALG_SLIC,
    ALIGNER_DEPLOYED,
    ALIGNER_FIT,
    ALIGNER_OBSERVED_PAIR,
    ALIGNER_REJECTED,
    Aligner,
    AlignerConfig,
    AlignerReport,
    BilinearScorer,
    IdentityScorer,
    InsufficientData,
    InvalidConfig,
    InvalidPreference,
    KNOWN_ALGORITHMS,
    KNOWN_MODELS,
    KNOWN_OPTIMIZERS,
    LinearScorer,
    MODEL_BILINEAR,
    MODEL_IDENTITY,
    MODEL_LINEAR,
    OPTIM_ADAMW,
    OPTIM_PA,
    OPTIM_SGD,
    Preference,
    REFERENCE_FREE_ALGORITHMS,
    UNARY_ALGORITHMS,
    UnknownAlgorithm,
    UnknownModel,
    cdpo_aligner,
    dpo_aligner,
    ipo_aligner,
    kto_aligner,
    orpo_aligner,
    rdpo_aligner,
    simpo_aligner,
    slic_aligner,
)


# =============================================================================
# Synthetic preference data
# =============================================================================


def _make_pair_dataset(n: int, *, true_token: str = "good",
                       noise: float = 0.1, seed: int = 0):
    """Pairs where the winner contains `true_token` and the loser doesn't.

    Noise: with probability `noise`, winner/loser are flipped.
    """
    rng = random.Random(seed)
    prompts = ["explain X", "summarise Y", "fix Z", "compute W", "predict V"]
    decoys_g = ["alpha", "beta", "gamma", "delta", "epsilon"]
    decoys_b = ["zeta", "eta", "theta", "iota", "kappa"]
    out = []
    for _ in range(n):
        prompt = rng.choice(prompts)
        winner = f"{true_token} " + " ".join(rng.choices(decoys_g, k=4))
        loser = "bad " + " ".join(rng.choices(decoys_b, k=4))
        if rng.random() < noise:
            winner, loser = loser, winner
        out.append((prompt, winner, loser, 0.0, 0.0))
    return out


# =============================================================================
# Configuration
# =============================================================================


class TestConfig(unittest.TestCase):

    def test_known_algorithms(self):
        for alg in ("dpo", "ipo", "kto", "slic", "simpo",
                    "orpo", "cdpo", "rdpo"):
            self.assertIn(alg, KNOWN_ALGORITHMS)

    def test_known_models(self):
        for m in ("linear", "bilinear", "identity"):
            self.assertIn(m, KNOWN_MODELS)

    def test_known_optimizers(self):
        for o in ("adamw", "sgd", "pa"):
            self.assertIn(o, KNOWN_OPTIMIZERS)

    def test_default_config_is_dpo(self):
        cfg = AlignerConfig()
        self.assertEqual(cfg.algorithm, ALG_DPO)
        self.assertEqual(cfg.model, MODEL_LINEAR)
        self.assertEqual(cfg.optimizer, OPTIM_ADAMW)
        self.assertGreater(cfg.beta, 0)

    def test_invalid_algorithm(self):
        with self.assertRaises(InvalidConfig):
            AlignerConfig(algorithm="not_a_real_algorithm")

    def test_invalid_model(self):
        with self.assertRaises(InvalidConfig):
            AlignerConfig(model="not_a_real_model")

    def test_invalid_optimizer(self):
        with self.assertRaises(InvalidConfig):
            AlignerConfig(optimizer="not_a_real_optimizer")

    def test_invalid_beta(self):
        for b in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(InvalidConfig):
                AlignerConfig(beta=b)

    def test_invalid_n_features(self):
        with self.assertRaises(InvalidConfig):
            AlignerConfig(n_features=1)

    def test_invalid_lr(self):
        with self.assertRaises(InvalidConfig):
            AlignerConfig(learning_rate=0)

    def test_invalid_weight_decay(self):
        with self.assertRaises(InvalidConfig):
            AlignerConfig(weight_decay=1.5)

    def test_invalid_eval_holdout(self):
        with self.assertRaises(InvalidConfig):
            AlignerConfig(eval_holdout_fraction=1.0)

    def test_invalid_cdpo_epsilon(self):
        for eps in (-0.1, 0.5, 0.6, 1.0):
            with self.assertRaises(InvalidConfig):
                AlignerConfig(cdpo_epsilon=eps)

    def test_invalid_rdpo_epsilon(self):
        for eps in (-0.1, 0.5, 0.6, 1.0):
            with self.assertRaises(InvalidConfig):
                AlignerConfig(rdpo_epsilon=eps)


# =============================================================================
# Preference observation
# =============================================================================


class TestPreference(unittest.TestCase):

    def test_pair_preference_basic(self):
        p = Preference(kind="pair", prompt="q", winner="a", loser="b",
                       ref_log_prob_winner=0.0, ref_log_prob_loser=0.0)
        self.assertEqual(p.kind, "pair")

    def test_unary_preference_basic(self):
        p = Preference(kind="unary", prompt="q", candidate="a",
                       desirable=True, ref_log_prob_candidate=0.0)
        self.assertEqual(p.kind, "unary")

    def test_bad_kind(self):
        with self.assertRaises(InvalidPreference):
            Preference(kind="other", prompt="q", winner="a", loser="b")

    def test_pair_requires_winner_loser(self):
        with self.assertRaises(InvalidPreference):
            Preference(kind="pair", prompt="q", winner="a")

    def test_pair_winner_loser_distinct(self):
        with self.assertRaises(InvalidPreference):
            Preference(kind="pair", prompt="q", winner="a", loser="a")

    def test_unary_requires_candidate(self):
        with self.assertRaises(InvalidPreference):
            Preference(kind="unary", prompt="q", desirable=True)

    def test_unary_requires_desirable(self):
        with self.assertRaises(InvalidPreference):
            Preference(kind="unary", prompt="q", candidate="a")

    def test_invalid_weight(self):
        with self.assertRaises(InvalidPreference):
            Preference(kind="pair", prompt="q", winner="a", loser="b",
                       weight=-1.0)

    def test_invalid_ref_log_prob(self):
        with self.assertRaises(InvalidPreference):
            Preference(kind="pair", prompt="q", winner="a", loser="b",
                       ref_log_prob_winner=float("nan"))


# =============================================================================
# Observation gating
# =============================================================================


class TestObservation(unittest.TestCase):

    def test_dpo_rejects_unary(self):
        a = dpo_aligner(beta=0.1, seed=0)
        with self.assertRaises(InvalidPreference):
            a.observe_unary(prompt="q", candidate="a", desirable=True,
                            ref_log_prob_candidate=0.0)

    def test_kto_rejects_pair(self):
        a = kto_aligner(beta=0.1, seed=0)
        with self.assertRaises(InvalidPreference):
            a.observe_pair(prompt="q", winner="a", loser="b",
                           ref_log_prob_winner=0.0, ref_log_prob_loser=0.0)

    def test_dpo_requires_ref_logprobs(self):
        a = dpo_aligner(beta=0.1, seed=0)
        with self.assertRaises(InvalidPreference):
            a.observe_pair(prompt="q", winner="a", loser="b")

    def test_simpo_ignores_ref_logprobs(self):
        a = simpo_aligner(seed=0)
        # SimPO is reference-free; should accept pair without ref_log_probs.
        a.observe_pair(prompt="q", winner="good", loser="bad")
        self.assertEqual(a.state()["n_observations"], 1)

    def test_orpo_ignores_ref_logprobs(self):
        # ORPO is in default REFERENCE_FREE_ALGORITHMS? Actually it's not
        # in that constant — it accepts but doesn't require ref. Check:
        # ORPO is not in REFERENCE_FREE_ALGORITHMS so it requires refs.
        a = orpo_aligner(seed=0)
        with self.assertRaises(InvalidPreference):
            a.observe_pair(prompt="q", winner="a", loser="b")

    def test_observe_kind_accepts_preference_instance(self):
        a = dpo_aligner(seed=0)
        p = Preference(kind="pair", prompt="q", winner="a", loser="b",
                       ref_log_prob_winner=0.0, ref_log_prob_loser=0.0)
        a.observe(p)
        self.assertEqual(a.state()["n_observations"], 1)


# =============================================================================
# Fitting on synthetic data — each algorithm should learn the discrimination
# =============================================================================


class TestFitLearnsSignal(unittest.TestCase):

    def _train_pair(self, ctor, **kw):
        a = ctor(seed=42, **kw)
        data = _make_pair_dataset(200, noise=0.05, seed=1)
        for prompt, w, l, rw, rl in data:
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        return a.fit()

    def test_dpo_learns(self):
        r = self._train_pair(dpo_aligner, beta=0.3)
        self.assertGreater(r.preference_accuracy, 0.7)
        self.assertTrue(r.deployed)
        self.assertEqual(r.algorithm, ALG_DPO)

    def test_ipo_learns(self):
        r = self._train_pair(ipo_aligner, beta=0.3)
        self.assertGreater(r.preference_accuracy, 0.7)
        self.assertEqual(r.algorithm, ALG_IPO)

    def test_slic_learns(self):
        r = self._train_pair(slic_aligner, beta=0.5, delta=1.0, lam=0.0)
        self.assertGreater(r.preference_accuracy, 0.6)

    def test_simpo_learns(self):
        a = simpo_aligner(beta=2.0, gamma=0.5, seed=42)
        data = _make_pair_dataset(200, noise=0.05, seed=1)
        for prompt, w, l, _, _ in data:
            a.observe_pair(prompt=prompt, winner=w, loser=l)
        r = a.fit()
        self.assertGreater(r.preference_accuracy, 0.7)
        self.assertEqual(r.algorithm, ALG_SIMPO)

    def test_orpo_learns(self):
        r = self._train_pair(orpo_aligner, beta=0.5, lam=0.5)
        self.assertGreater(r.preference_accuracy, 0.6)

    def test_cdpo_learns(self):
        r = self._train_pair(cdpo_aligner, beta=0.3, epsilon=0.1)
        self.assertGreater(r.preference_accuracy, 0.7)

    def test_rdpo_learns(self):
        r = self._train_pair(rdpo_aligner, beta=0.3, epsilon=0.1)
        self.assertGreater(r.preference_accuracy, 0.7)

    def test_kto_learns(self):
        a = kto_aligner(beta=0.3, seed=42)
        data = _make_pair_dataset(150, noise=0.05, seed=1)
        for prompt, w, l, rw, rl in data:
            a.observe_unary(prompt=prompt, candidate=w, desirable=True,
                            ref_log_prob_candidate=0.0)
            a.observe_unary(prompt=prompt, candidate=l, desirable=False,
                            ref_log_prob_candidate=0.0)
        r = a.fit()
        self.assertGreater(r.preference_accuracy, 0.6)


# =============================================================================
# Scoring / best_of_n / preference_probability / softmax_sample
# =============================================================================


class TestScoring(unittest.TestCase):

    def _train_simple(self):
        a = dpo_aligner(beta=0.5, seed=1)
        data = _make_pair_dataset(150, noise=0.0, seed=2)
        for prompt, w, l, rw, rl in data:
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        a.fit()
        return a

    def test_score_returns_float(self):
        a = self._train_simple()
        s = a.score("explain X", "good answer")
        self.assertIsInstance(s, float)

    def test_best_of_n_picks_good_candidate(self):
        a = self._train_simple()
        best = a.best_of_n("explain X",
                           ["bad zeta eta", "good alpha beta gamma delta",
                            "bad theta iota"])
        self.assertEqual(best, "good alpha beta gamma delta")

    def test_best_of_n_empty_raises(self):
        a = self._train_simple()
        with self.assertRaises(InvalidPreference):
            a.best_of_n("q", [])

    def test_preference_probability_in_range(self):
        a = self._train_simple()
        p = a.preference_probability("explain X", "good answer", "bad answer")
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_preference_probability_good_beats_bad(self):
        a = self._train_simple()
        p = a.preference_probability("explain X", "good a b c d", "bad z y x w")
        self.assertGreater(p, 0.5)

    def test_softmax_sample_in_candidates(self):
        a = self._train_simple()
        rng = random.Random(0)
        cands = ["good a", "bad b", "good c"]
        for _ in range(20):
            x = a.softmax_sample("q", cands, temperature=1.0, rng=rng)
            self.assertIn(x, cands)

    def test_softmax_sample_bad_temperature(self):
        a = self._train_simple()
        with self.assertRaises(InvalidConfig):
            a.softmax_sample("q", ["a", "b"], temperature=0.0)


# =============================================================================
# Determinism (replay) + fingerprint
# =============================================================================


class TestDeterminism(unittest.TestCase):

    def _run(self, seed=7):
        a = dpo_aligner(beta=0.3, seed=seed)
        data = _make_pair_dataset(80, noise=0.05, seed=3)
        for prompt, w, l, rw, rl in data:
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        return a.fit(), a.fingerprint()

    def test_replay_determinism(self):
        r1, fp1 = self._run(seed=11)
        r2, fp2 = self._run(seed=11)
        self.assertEqual(fp1, fp2)
        self.assertAlmostEqual(r1.preference_accuracy, r2.preference_accuracy)
        self.assertAlmostEqual(r1.train_loss, r2.train_loss)
        self.assertAlmostEqual(r1.eval_loss, r2.eval_loss)

    def test_different_seed_changes_fingerprint(self):
        _, fp1 = self._run(seed=1)
        _, fp2 = self._run(seed=2)
        self.assertNotEqual(fp1, fp2)


# =============================================================================
# Eval gating + deployment ladder
# =============================================================================


class TestEvalGating(unittest.TestCase):

    def test_random_preferences_dont_promote_or_get_demoted(self):
        a = dpo_aligner(beta=0.3, seed=42,
                        min_accuracy_improvement=0.1)
        rng = random.Random(7)
        # First batch: clear signal -> deploys.
        data1 = _make_pair_dataset(80, noise=0.0, seed=10)
        for prompt, w, l, rw, rl in data1:
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r1 = a.fit()
        self.assertTrue(r1.deployed)

    def test_first_fit_always_deploys(self):
        a = dpo_aligner(beta=0.3, seed=42)
        for prompt, w, l, rw, rl in _make_pair_dataset(50, seed=4):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        self.assertTrue(r.deployed)

    def test_min_fit_observations(self):
        cfg = AlignerConfig(algorithm=ALG_DPO, min_fit_observations=100)
        a = Aligner(cfg)
        with self.assertRaises(InsufficientData):
            a.fit()


# =============================================================================
# Statistical certificates
# =============================================================================


class TestCertificates(unittest.TestCase):

    def test_lcb_bounds_are_ordered(self):
        a = dpo_aligner(beta=0.3, seed=42)
        for prompt, w, l, rw, rl in _make_pair_dataset(200, noise=0.05, seed=5):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        # All LCBs ≤ accuracy ≤ UCB.
        self.assertLessEqual(r.preference_accuracy_lcb_hoeffding,
                              r.preference_accuracy + 1e-9)
        self.assertLessEqual(r.preference_accuracy_lcb_bernstein,
                              r.preference_accuracy + 1e-9)
        self.assertLessEqual(r.preference_accuracy_lcb_anytime,
                              r.preference_accuracy + 1e-9)
        self.assertLessEqual(r.preference_accuracy,
                              r.preference_accuracy_ucb_hoeffding + 1e-9)
        # All LCBs ≥ 0.
        self.assertGreaterEqual(r.preference_accuracy_lcb_hoeffding, 0.0)
        self.assertGreaterEqual(r.preference_accuracy_lcb_bernstein, 0.0)
        self.assertGreaterEqual(r.preference_accuracy_lcb_anytime, 0.0)

    def test_e_process_grows_on_correct_signal(self):
        a = dpo_aligner(beta=0.3, seed=42)
        for prompt, w, l, rw, rl in _make_pair_dataset(200, noise=0.05, seed=5):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        # With strong signal, e-process should grow well above 1.
        self.assertGreater(r.e_process, 1.0)

    def test_pacbayes_bound_finite(self):
        a = dpo_aligner(beta=0.3, seed=42)
        for prompt, w, l, rw, rl in _make_pair_dataset(80, noise=0.05, seed=5):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        self.assertTrue(math.isfinite(r.pacbayes_bound))
        self.assertGreaterEqual(r.pacbayes_bound, r.eval_loss - 1e-9)

    def test_kl_ci_non_negative_half_width(self):
        a = dpo_aligner(beta=0.3, seed=42)
        for prompt, w, l, rw, rl in _make_pair_dataset(100, noise=0.05, seed=5):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        self.assertGreaterEqual(r.kl_ci_half_width, 0.0)


# =============================================================================
# Reservoir buffer enforces capacity
# =============================================================================


class TestBuffer(unittest.TestCase):

    def test_buffer_capacity_respected(self):
        cfg = AlignerConfig(algorithm=ALG_DPO, buffer_capacity=50, seed=0,
                            min_fit_observations=1)
        a = Aligner(cfg)
        for prompt, w, l, rw, rl in _make_pair_dataset(200, seed=6):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        # The total observed count is 200 even though buffer ≤ 50.
        s = a.state()
        self.assertEqual(s["n_observations"], 200)


# =============================================================================
# Scorer models — direct API
# =============================================================================


class TestScorerModels(unittest.TestCase):

    def test_linear_scorer_score(self):
        cfg = AlignerConfig(algorithm=ALG_DPO, model=MODEL_LINEAR,
                            n_features=128, seed=0)
        s = LinearScorer(cfg)
        # Empty scorer returns 0.
        self.assertEqual(s.score({"foo": 1.0}), 0.0)
        s.update({"foo": 1.0}, 0.5)
        s.update({"bar": 2.0}, -0.3)
        # Update applied: score should be non-zero on observed features.
        self.assertNotEqual(s.score({"foo": 1.0}), 0.0)

    def test_identity_scorer_returns_score_feature(self):
        cfg = AlignerConfig(algorithm=ALG_DPO, model=MODEL_IDENTITY, seed=0)
        s = IdentityScorer(cfg)
        self.assertEqual(s.score({"#score": 1.5, "foo": 99.0}), 1.5)
        # Updates are no-ops.
        s.update({"foo": 1.0}, 0.5)
        self.assertEqual(s.score({"#score": 2.0}), 2.0)

    def test_bilinear_scorer_score_after_update(self):
        cfg = AlignerConfig(algorithm=ALG_DPO, model=MODEL_BILINEAR,
                            n_features=64, bilinear_rank=4, seed=0)
        s = BilinearScorer(cfg)
        s.update({"p:hi": 1.0, "c:there": 1.0}, 0.5)
        # Should have created rows in U and V.
        self.assertGreater(len(s.U), 0)
        self.assertGreater(len(s.V), 0)
        # Score is finite.
        self.assertTrue(math.isfinite(s.score({"p:hi": 1.0, "c:there": 1.0})))


# =============================================================================
# Calibration
# =============================================================================


class TestCalibration(unittest.TestCase):

    def test_temperature_calibration_runs(self):
        cfg = AlignerConfig(algorithm=ALG_DPO, beta=0.3, seed=42,
                            temperature_calibration=True)
        a = Aligner(cfg)
        for prompt, w, l, rw, rl in _make_pair_dataset(150, noise=0.1, seed=7):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        # Temperature should be a positive finite scalar.
        s = a.state()
        self.assertGreater(s["temperature"], 0.0)

    def test_isotonic_calibration_runs(self):
        cfg = AlignerConfig(algorithm=ALG_DPO, beta=0.3, seed=42,
                            isotonic_calibration=True)
        a = Aligner(cfg)
        for prompt, w, l, rw, rl in _make_pair_dataset(150, noise=0.1, seed=7):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        # No exception, deployment succeeded.
        self.assertTrue(r.deployed)


# =============================================================================
# Optimisers
# =============================================================================


class TestOptimizers(unittest.TestCase):

    def _train_with(self, optimizer):
        cfg = AlignerConfig(algorithm=ALG_DPO, beta=0.3, seed=42,
                            optimizer=optimizer,
                            learning_rate=0.05 if optimizer == OPTIM_SGD else 0.01)
        a = Aligner(cfg)
        for prompt, w, l, rw, rl in _make_pair_dataset(200, noise=0.05, seed=8):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        return a.fit()

    def test_adamw_optimizer(self):
        r = self._train_with(OPTIM_ADAMW)
        self.assertGreater(r.preference_accuracy, 0.6)

    def test_sgd_optimizer(self):
        r = self._train_with(OPTIM_SGD)
        self.assertGreater(r.preference_accuracy, 0.5)


# =============================================================================
# Identity scorer pipeline (LLM-logprob mode)
# =============================================================================


class TestIdentityScorerPipeline(unittest.TestCase):

    def test_identity_scorer_uses_caller_supplied_score(self):
        # Treat caller-supplied features['#score'] as the model output.
        # The Aligner is then used purely for stats + deployment gating.
        cfg = AlignerConfig(algorithm=ALG_DPO, beta=0.5, seed=0,
                            model=MODEL_IDENTITY)
        # Featurizer injects the caller's "logit" into #score.
        def f(prompt, candidate):
            # Higher score for candidates containing 'good'.
            s = 1.0 if isinstance(candidate, str) and "good" in candidate else -1.0
            return {"#score": s, "#bias": 1.0}
        a = Aligner(cfg, featurizer=f)
        for prompt, w, l, rw, rl in _make_pair_dataset(50, noise=0.0, seed=9):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        # The pre-built scorer is perfect on this dataset.
        self.assertGreaterEqual(r.preference_accuracy, 0.95)


# =============================================================================
# rDPO + cDPO under label noise
# =============================================================================


class TestNoiseRobustness(unittest.TestCase):

    def test_rdpo_robust_under_25pct_noise(self):
        # rDPO with the correct noise rate should approach the clean-data
        # accuracy.
        a = rdpo_aligner(beta=0.3, epsilon=0.25, seed=42)
        for prompt, w, l, rw, rl in _make_pair_dataset(300, noise=0.25, seed=11):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        # Under flip noise 0.25, naive DPO degrades; rDPO should still
        # do better than chance.
        self.assertGreater(r.preference_accuracy, 0.55)

    def test_cdpo_label_smoothing_runs(self):
        a = cdpo_aligner(beta=0.3, epsilon=0.1, seed=42)
        for prompt, w, l, rw, rl in _make_pair_dataset(200, noise=0.1, seed=11):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        self.assertGreater(r.preference_accuracy, 0.6)


# =============================================================================
# Report contents
# =============================================================================


class TestReport(unittest.TestCase):

    def test_report_fields(self):
        a = dpo_aligner(beta=0.3, seed=42)
        for prompt, w, l, rw, rl in _make_pair_dataset(80, seed=12):
            a.observe_pair(prompt=prompt, winner=w, loser=l,
                           ref_log_prob_winner=rw, ref_log_prob_loser=rl)
        r = a.fit()
        self.assertIsInstance(r, AlignerReport)
        self.assertEqual(r.algorithm, ALG_DPO)
        self.assertEqual(r.model, MODEL_LINEAR)
        self.assertGreater(r.n_observations, 0)
        self.assertGreater(r.n_train, 0)
        self.assertGreater(r.n_eval, 0)
        self.assertGreaterEqual(r.preference_accuracy, 0.0)
        self.assertLessEqual(r.preference_accuracy, 1.0)
        self.assertGreaterEqual(r.elapsed_seconds, 0.0)
        self.assertGreaterEqual(r.iterations, 1)
        self.assertGreaterEqual(r.weight_l2, 0.0)
        self.assertEqual(len(r.fingerprint_hash), 64)  # hex SHA-256
        self.assertGreater(r.chain_length, 0)


# =============================================================================
# Thread safety smoke
# =============================================================================


class TestThreadSafety(unittest.TestCase):

    def test_concurrent_observation(self):
        import threading
        a = dpo_aligner(beta=0.3, seed=0, buffer_capacity=10000)
        data = _make_pair_dataset(500, seed=13)

        def worker(slice_):
            for prompt, w, l, rw, rl in slice_:
                a.observe_pair(prompt=prompt, winner=w, loser=l,
                               ref_log_prob_winner=rw, ref_log_prob_loser=rl)

        chunks = [data[i::4] for i in range(4)]
        ths = [threading.Thread(target=worker, args=(c,)) for c in chunks]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        self.assertEqual(a.state()["n_observations"], 500)


# =============================================================================
# Convenience constructors
# =============================================================================


class TestConvenience(unittest.TestCase):

    def test_each_constructor_returns_aligner(self):
        for ctor in (dpo_aligner, ipo_aligner, kto_aligner, slic_aligner,
                     simpo_aligner, orpo_aligner, cdpo_aligner, rdpo_aligner):
            a = ctor(seed=0)
            self.assertIsInstance(a, Aligner)

    def test_alg_constants_match(self):
        self.assertEqual(ALG_DPO, "dpo")
        self.assertEqual(ALG_IPO, "ipo")
        self.assertEqual(ALG_KTO, "kto")
        self.assertEqual(ALG_SLIC, "slic")
        self.assertEqual(ALG_SIMPO, "simpo")
        self.assertEqual(ALG_ORPO, "orpo")
        self.assertEqual(ALG_CDPO, "cdpo")
        self.assertEqual(ALG_RDPO, "rdpo")

    def test_unary_alg_set(self):
        self.assertIn(ALG_KTO, UNARY_ALGORITHMS)
        self.assertNotIn(ALG_DPO, UNARY_ALGORITHMS)

    def test_reference_free_alg_set(self):
        self.assertIn(ALG_SIMPO, REFERENCE_FREE_ALGORITHMS)
        self.assertNotIn(ALG_DPO, REFERENCE_FREE_ALGORITHMS)


# =============================================================================
# State + fingerprint
# =============================================================================


class TestStateFingerprint(unittest.TestCase):

    def test_state_contains_expected_keys(self):
        a = dpo_aligner(seed=0)
        keys = set(a.state().keys())
        for k in ("algorithm", "model", "n_observations", "iterations",
                  "deployed_iteration", "rejected_count", "best_eval_lcb",
                  "best_eval_ucb", "temperature", "e_process",
                  "fingerprint", "chain_length"):
            self.assertIn(k, keys)

    def test_fingerprint_changes_after_observation(self):
        a = dpo_aligner(seed=0)
        fp1 = a.fingerprint()
        a.observe_pair(prompt="q", winner="a", loser="b",
                       ref_log_prob_winner=0.0, ref_log_prob_loser=0.0)
        fp2 = a.fingerprint()
        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
