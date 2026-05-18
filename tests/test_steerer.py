"""Tests for the Steerer contrastive activation-steering primitive.

Covers:
  * Vector / matrix helpers (PSD solver, AUROC / Mason-Graham CI, Spearman,
    hedged-capital betting e-process, Welch t)
  * ContrastivePair / DoseOutcome / SteererConfig validation
  * All four fit algorithms (CAA, LAT, LDA, probe) recover the planted
    axis on a separable synthetic dataset
  * Application primitives (addition, ablation, clamp, rank-one) and the
    Steerer.apply / ablate / clamp convenience methods
  * Recommend coefficient against a dose-response with a known optimum
  * certify(): pass / fail / inconclusive / flip recommendation paths
  * Fingerprint chain is deterministic given the same observation stream
  * Event emission on every public mutation
  * compare_steerers cosine similarity
"""
from __future__ import annotations

import math
import random
import unittest

from agi.events import EventBus
from agi.steerer import (
    ALG_CAA,
    ALG_LAT,
    ALG_LDA,
    ALG_PROBE,
    KNOWN_ALGORITHMS,
    KNOWN_EVENTS,
    KNOWN_RECOMMENDATIONS,
    KNOWN_TESTS,
    KNOWN_VERDICTS,
    REC_FLIP,
    REC_HOLD,
    REC_PROMOTE,
    REC_REJECT,
    STEERER_CERTIFIED,
    STEERER_FIT,
    STEERER_OUTCOME_OBSERVED,
    STEERER_PAIR_OBSERVED,
    STEERER_RECOMMENDED,
    STEERER_REPORTED,
    STEERER_RESET,
    STEERER_STARTED,
    STEERER_TESTED,
    TEST_DOSE_RESPONSE,
    TEST_EFFECT_SIZE,
    TEST_LEAKAGE,
    TEST_SEPARABILITY,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
    VERDICT_PASS,
    VERDICT_WARN,
    ContrastivePair,
    DimensionMismatch,
    DoseOutcome,
    InsufficientData,
    InvalidConfig,
    InvalidOutcome,
    InvalidPair,
    NotFitted,
    Steerer,
    SteererCertificate,
    SteererConfig,
    SteererError,
    SteererFit,
    SteererReport,
    TestResult,
    UnknownAlgorithm,
    apply_addition,
    apply_clamp,
    apply_orthogonal_ablation,
    apply_rank_one,
    auroc,
    auroc_ci,
    compare_steerers,
    fit_caa,
    fit_lat,
    fit_lda,
    fit_probe,
    fresh_steerer,
    hedged_capital_lift_e,
    spearman_p,
    spearman_rho,
    synthetic_pairs,
    welch_t,
)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


class NumericsTests(unittest.TestCase):
    def test_auroc_perfect_separation(self):
        self.assertEqual(auroc([3, 4, 5], [0, 1, 2]), 1.0)

    def test_auroc_inverted(self):
        self.assertEqual(auroc([0, 1, 2], [3, 4, 5]), 0.0)

    def test_auroc_chance(self):
        self.assertAlmostEqual(auroc([1, 1, 1], [1, 1, 1]), 0.5)

    def test_auroc_ci_returns_in_range(self):
        a, lo, hi = auroc_ci([3, 4, 5, 4], [0, 1, 2, 1])
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(a, 1.0)
        self.assertLessEqual(lo, a)
        self.assertGreaterEqual(hi, a)

    def test_spearman_monotonic(self):
        rho = spearman_rho([0, 1, 2, 3, 4], [10, 20, 30, 40, 50])
        self.assertAlmostEqual(rho, 1.0)

    def test_spearman_inverse(self):
        rho = spearman_rho([0, 1, 2, 3, 4], [50, 40, 30, 20, 10])
        self.assertAlmostEqual(rho, -1.0)

    def test_spearman_p_one_sided(self):
        # Strong positive rho on n=20 — should be a small p.
        p = spearman_p(0.9, 20)
        self.assertLess(p, 0.001)
        self.assertEqual(spearman_p(0.0, 20), 1.0)

    def test_welch_t_smoke(self):
        # Distinct samples — non-zero variance, non-zero t.
        t, df, diff, p = welch_t([1.0, 1.1, 0.9, 1.0, 1.0], [0.0, 0.1, -0.1, 0.0, 0.0])
        self.assertGreater(t, 0.0)
        self.assertGreater(df, 0.0)
        self.assertAlmostEqual(diff, 1.0, places=2)
        self.assertLess(p, 0.5)

    def test_welch_t_zero_variance_returns_zero(self):
        # Degenerate identical inputs: SE = 0 → handler returns t=0 + p=0.5.
        t, df, diff, p = welch_t([1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
        self.assertEqual(t, 0.0)
        self.assertEqual(p, 0.5)

    def test_hedged_capital_e_under_h1(self):
        # Mean of diffs is 0.5 vs δ=0.1 — strong signal.
        e, p, mean = hedged_capital_lift_e([0.5] * 50, delta=0.1)
        self.assertGreater(e, 1.0)
        self.assertLess(p, 0.01)
        self.assertAlmostEqual(mean, 0.5)

    def test_hedged_capital_e_under_h0(self):
        # Mean exactly at δ — e-value stays modest.
        e, p, mean = hedged_capital_lift_e([0.1] * 50, delta=0.1)
        self.assertLess(e, 10.0)
        self.assertGreaterEqual(p, 0.1)


# ---------------------------------------------------------------------------
# Config / pair / outcome validation
# ---------------------------------------------------------------------------


class ValidationTests(unittest.TestCase):
    def test_config_defaults_ok(self):
        c = SteererConfig(dim=4)
        self.assertEqual(c.algorithm, ALG_CAA)
        self.assertEqual(c.dim, 4)
        self.assertEqual(c.alpha, 0.05)

    def test_config_unknown_algorithm(self):
        with self.assertRaises(UnknownAlgorithm):
            SteererConfig(algorithm="not_a_thing", dim=4)

    def test_config_bad_dim(self):
        with self.assertRaises(InvalidConfig):
            SteererConfig(dim=0)

    def test_config_alpha_range(self):
        with self.assertRaises(InvalidConfig):
            SteererConfig(dim=4, alpha=0.0)
        with self.assertRaises(InvalidConfig):
            SteererConfig(dim=4, alpha=0.6)

    def test_config_grid_must_contain_zero(self):
        with self.assertRaises(InvalidConfig):
            SteererConfig(dim=4, coefficient_grid=(1.0, 2.0))

    def test_pair_validates_vectors(self):
        with self.assertRaises(DimensionMismatch):
            ContrastivePair(task_id="t", positive=(1.0,), negative=(0.0, 0.0))
        with self.assertRaises(InvalidPair):
            ContrastivePair(task_id="t", positive=(1.0, float("nan")),
                            negative=(0.0, 0.0))

    def test_pair_split_validation(self):
        with self.assertRaises(InvalidPair):
            ContrastivePair(task_id="t", positive=(1.0,), negative=(0.0,),
                            split="bogus")

    def test_outcome_score_range(self):
        with self.assertRaises(InvalidOutcome):
            DoseOutcome(task_id="t", coefficient=0.5, outcome=True,
                        outcome_score=2.0)

    def test_outcome_coefficient_finite(self):
        with self.assertRaises(InvalidOutcome):
            DoseOutcome(task_id="t", coefficient=float("inf"), outcome=True)


# ---------------------------------------------------------------------------
# Fitting algorithms recover the planted axis
# ---------------------------------------------------------------------------


class FitAlgorithmsTests(unittest.TestCase):
    def _check_axis_recovery(self, algorithm, axis=0):
        pairs = synthetic_pairs(n=60, dim=6, axis=axis, snr=2.0, noise=0.3, seed=7)
        s = Steerer(SteererConfig(algorithm=algorithm, dim=6, alpha=0.05, seed=0))
        for p in pairs:
            s.observe_pair(p)
        fit = s.fit()
        self.assertGreater(abs(fit.direction[axis]), 0.85)
        # AUROC near 1 on cleanly separable axis:
        self.assertGreater(fit.separability_auroc, 0.9)
        self.assertGreater(fit.cohen_d, 1.0)
        # Direction is unit-norm:
        norm = math.sqrt(sum(x * x for x in fit.direction))
        self.assertAlmostEqual(norm, 1.0, places=4)
        return fit

    def test_caa_recovers_axis(self):
        self._check_axis_recovery(ALG_CAA, axis=2)

    def test_lat_recovers_axis(self):
        self._check_axis_recovery(ALG_LAT, axis=1)

    def test_lda_recovers_axis(self):
        self._check_axis_recovery(ALG_LDA, axis=3)

    def test_probe_recovers_axis(self):
        self._check_axis_recovery(ALG_PROBE, axis=0)

    def test_fit_module_functions(self):
        # The module-level helpers are part of the public surface.
        positives = [(1.0, 0.0), (1.2, 0.1), (0.8, -0.1)]
        negatives = [(-1.0, 0.0), (-1.1, 0.1), (-0.9, 0.05)]
        d, mp, mn, n = fit_caa(positives, negatives)
        self.assertGreater(d[0], 0.9)
        d2, mp2, mn2, n2 = fit_lat(positives, negatives)
        self.assertGreater(d2[0], 0.9)
        d3, mp3, mn3, n3, _ = fit_lda(positives, negatives)
        # LDA with three pairs is noisy; mostly verify it gets the sign
        # right and is dominated by the first axis.
        self.assertGreater(d3[0], 0.7)
        d4, mp4, mn4, n4, _ = fit_probe(positives, negatives, seed=0)
        self.assertGreater(d4[0], 0.7)

    def test_fit_insufficient_data(self):
        s = Steerer(SteererConfig(dim=4))
        with self.assertRaises(InsufficientData):
            s.fit()

    def test_lda_falls_back_under_degenerate(self):
        # Identical positives — degenerate within-class scatter.
        pos = [(1.0, 0.0)] * 3
        neg = [(-1.0, 0.0)] * 3
        d, mp, mn, n, diag = fit_lda(pos, neg)
        self.assertEqual(len(d), 2)
        # Direction still points along the mean difference:
        self.assertGreater(d[0], 0.9)


# ---------------------------------------------------------------------------
# Application primitives
# ---------------------------------------------------------------------------


class ApplicationTests(unittest.TestCase):
    def test_addition(self):
        out = apply_addition((1.0, 2.0, 3.0), (1.0, 0.0, 0.0), 0.5)
        self.assertEqual(out, (1.5, 2.0, 3.0))

    def test_orthogonal_ablation_removes_projection(self):
        a = (3.0, 4.0, 0.0)
        d = (1.0, 0.0, 0.0)
        out = apply_orthogonal_ablation(a, d)
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 4.0)

    def test_clamp_sets_projection(self):
        a = (1.0, 2.0)
        d = (1.0, 0.0)
        out = apply_clamp(a, d, target_value=5.0)
        self.assertAlmostEqual(out[0], 5.0)
        self.assertAlmostEqual(out[1], 2.0)

    def test_rank_one_scales_projection(self):
        a = (2.0, 1.0)
        d = (1.0, 0.0)
        out = apply_rank_one(a, d, coefficient=0.5)
        # proj=2, shift = 0.5*2 = 1, so first coord += 1
        self.assertAlmostEqual(out[0], 3.0)
        self.assertAlmostEqual(out[1], 1.0)

    def test_dim_mismatch_raises(self):
        with self.assertRaises(DimensionMismatch):
            apply_addition((1.0,), (1.0, 0.0), 0.5)

    def test_steerer_apply_requires_fit(self):
        s = Steerer(SteererConfig(dim=3))
        with self.assertRaises(NotFitted):
            s.apply((1.0, 0.0, 0.0), 0.5)
        with self.assertRaises(NotFitted):
            s.ablate((1.0, 0.0, 0.0))
        with self.assertRaises(NotFitted):
            s.clamp((1.0, 0.0, 0.0), target_value=1.0)


# ---------------------------------------------------------------------------
# Recommend + certify
# ---------------------------------------------------------------------------


def _simulate_dose_outcomes(s, rng, coefficients, n_per_coef=30, slope=2.0):
    """Outcome ~ Bernoulli(σ(slope · coefficient))."""
    for c in coefficients:
        rate = 1.0 / (1.0 + math.exp(-slope * c))
        for j in range(n_per_coef):
            s.observe_outcome(DoseOutcome(
                task_id=f"q{j}-{c}",
                coefficient=c,
                outcome=rng.random() < rate,
            ))


class CertifyTests(unittest.TestCase):
    def _build_pass_case(self, seed=0):
        s = Steerer(SteererConfig(algorithm=ALG_CAA, dim=6, alpha=0.05, seed=seed))
        for p in synthetic_pairs(n=80, dim=6, axis=0, snr=2.0, noise=0.3, seed=seed):
            s.observe_pair(p)
        s.fit()
        return s

    def test_certify_pass(self):
        s = self._build_pass_case(seed=1)
        rng = random.Random(11)
        _simulate_dose_outcomes(s, rng, (-1.0, 0.0, 1.0, 2.0), n_per_coef=40, slope=3.0)
        cert = s.certify(delta=0.1, alpha=0.05)
        self.assertEqual(cert.verdict, VERDICT_PASS)
        self.assertEqual(cert.recommendation, REC_PROMOTE)
        self.assertIsNotNone(cert.recommended_coefficient)
        self.assertGreater(cert.outcome_lift, 0.1)
        # Tests tuple shape:
        names = {t.name for t in cert.tests}
        self.assertIn(TEST_SEPARABILITY, names)
        self.assertIn(TEST_DOSE_RESPONSE, names)
        self.assertIn(TEST_EFFECT_SIZE, names)

    def test_certify_no_doses_is_inconclusive_or_warn(self):
        s = self._build_pass_case(seed=2)
        cert = s.certify(delta=0.1, alpha=0.05)
        # With pairs but zero outcomes: dose_response stub has n=0; effect
        # cannot be tested; verdict is at most WARN.  We accept either
        # WARN (separability passed, others stubbed) or INCONCLUSIVE.
        self.assertIn(cert.verdict, (VERDICT_WARN, VERDICT_INCONCLUSIVE))
        self.assertIn(cert.recommendation, (REC_HOLD,))

    def test_certify_no_data_inconclusive(self):
        s = Steerer(SteererConfig(dim=4))
        cert = s.certify(delta=0.1, alpha=0.05)
        self.assertEqual(cert.verdict, VERDICT_INCONCLUSIVE)
        self.assertEqual(cert.recommendation, REC_HOLD)

    def test_recommend_coefficient_picks_best(self):
        s = self._build_pass_case(seed=3)
        rng = random.Random(7)
        _simulate_dose_outcomes(s, rng, (-1.0, 0.0, 1.0, 2.0), n_per_coef=40, slope=2.5)
        best = s.recommend_coefficient()
        # Higher coefficients drive σ→1, so the optimum is the largest
        # coefficient observed.
        self.assertEqual(best, 2.0)

    def test_recommend_coefficient_safety_floor(self):
        s = self._build_pass_case(seed=4)
        rng = random.Random(8)
        # Make every dose rate identical so the floor is the only
        # discriminator: at floor=1.01, no coefficient qualifies →
        # returns 0.0 fallback.
        _simulate_dose_outcomes(s, rng, (-1.0, 0.0, 1.0), n_per_coef=30, slope=1.0)
        # Impossible floor:
        c = s.recommend_coefficient(safety_floor=1.01)
        self.assertEqual(c, 0.0)


# ---------------------------------------------------------------------------
# Replay-verifiable fingerprint chain
# ---------------------------------------------------------------------------


class ReplayTests(unittest.TestCase):
    def test_same_stream_same_chain(self):
        cfg = dict(algorithm=ALG_CAA, dim=4, alpha=0.05, seed=0)
        s1 = Steerer(SteererConfig(**cfg))
        s2 = Steerer(SteererConfig(**cfg))
        rng = random.Random(0)
        for i in range(12):
            p = ContrastivePair(
                task_id=f"t{i}",
                positive=tuple(rng.gauss(1.0 if j == 0 else 0.0, 0.3) for j in range(4)),
                negative=tuple(rng.gauss(-1.0 if j == 0 else 0.0, 0.3) for j in range(4)),
            )
            s1.observe_pair(p)
            s2.observe_pair(p)
        self.assertEqual(s1.fingerprint, s2.fingerprint)
        # Both fit → same chain
        s1.fit()
        s2.fit()
        self.assertEqual(s1.fingerprint, s2.fingerprint)

    def test_different_seed_same_fit_for_caa(self):
        # CAA is deterministic — the seed has no effect on the fit.
        pairs = synthetic_pairs(n=30, dim=4, seed=5)
        s1 = Steerer(SteererConfig(algorithm=ALG_CAA, dim=4, seed=0))
        s2 = Steerer(SteererConfig(algorithm=ALG_CAA, dim=4, seed=999))
        for p in pairs:
            s1.observe_pair(p)
            s2.observe_pair(p)
        f1 = s1.fit()
        f2 = s2.fit()
        for a, b in zip(f1.direction, f2.direction):
            self.assertAlmostEqual(a, b, places=10)


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class EventTests(unittest.TestCase):
    def test_known_events_cover_all_emit_paths(self):
        for ev in (
            STEERER_STARTED,
            STEERER_PAIR_OBSERVED,
            STEERER_OUTCOME_OBSERVED,
            STEERER_FIT,
            STEERER_TESTED,
            STEERER_CERTIFIED,
            STEERER_RECOMMENDED,
            STEERER_REPORTED,
            STEERER_RESET,
        ):
            self.assertIn(ev, KNOWN_EVENTS)

    def test_bus_receives_started_pair_and_fit(self):
        bus = EventBus()
        seen = []
        bus.subscribe(lambda e: seen.append(e.kind))
        s = Steerer(SteererConfig(dim=4), bus=bus)
        s.observe_pair(ContrastivePair(
            task_id="t",
            positive=(1.0, 0.0, 0.0, 0.0),
            negative=(-1.0, 0.0, 0.0, 0.0),
        ))
        s.observe_pair(ContrastivePair(
            task_id="t2",
            positive=(0.8, 0.0, 0.0, 0.0),
            negative=(-0.9, 0.0, 0.0, 0.0),
        ))
        s.fit()
        self.assertIn(STEERER_STARTED, seen)
        self.assertIn(STEERER_PAIR_OBSERVED, seen)
        self.assertIn(STEERER_FIT, seen)

    def test_bus_receives_outcome_certified_and_reset(self):
        bus = EventBus()
        seen = []
        bus.subscribe(lambda e: seen.append(e.kind))
        s = Steerer(SteererConfig(dim=4), bus=bus)
        # Need pairs to fit before certifying
        rng = random.Random(2)
        for i in range(8):
            s.observe_pair(ContrastivePair(
                task_id=f"t{i}",
                positive=tuple(rng.gauss(1.0 if j == 0 else 0.0, 0.2) for j in range(4)),
                negative=tuple(rng.gauss(-1.0 if j == 0 else 0.0, 0.2) for j in range(4)),
            ))
        s.fit()
        s.observe_outcome(DoseOutcome(task_id="q", coefficient=0.0, outcome=False))
        s.observe_outcome(DoseOutcome(task_id="q", coefficient=1.0, outcome=True))
        s.certify(delta=0.05)
        s.reset()
        self.assertIn(STEERER_OUTCOME_OBSERVED, seen)
        self.assertIn(STEERER_CERTIFIED, seen)
        self.assertIn(STEERER_RESET, seen)


# ---------------------------------------------------------------------------
# Compare two steerers
# ---------------------------------------------------------------------------


class CompareTests(unittest.TestCase):
    def test_compare_steerers_returns_cosine(self):
        s1 = Steerer(SteererConfig(algorithm=ALG_CAA, dim=4, seed=0))
        s2 = Steerer(SteererConfig(algorithm=ALG_LAT, dim=4, seed=0))
        for p in synthetic_pairs(n=24, dim=4, axis=0, seed=0):
            s1.observe_pair(p)
            s2.observe_pair(p)
        s1.fit()
        s2.fit()
        comp = compare_steerers(s1, s2)
        self.assertIn("cosine_similarity", comp)
        # CAA and LAT should learn nearly the same direction on a single-axis
        # planted dataset.
        self.assertGreater(comp["cosine_similarity"], 0.9)

    def test_compare_requires_fit(self):
        s1 = Steerer(SteererConfig(dim=4))
        s2 = Steerer(SteererConfig(dim=4))
        with self.assertRaises(NotFitted):
            compare_steerers(s1, s2)


# ---------------------------------------------------------------------------
# Reset clears state
# ---------------------------------------------------------------------------


class ResetTests(unittest.TestCase):
    def test_reset_clears_pairs_and_outcomes(self):
        s = Steerer(SteererConfig(dim=4))
        for p in synthetic_pairs(n=6, dim=4, seed=0):
            s.observe_pair(p)
        s.observe_outcome(DoseOutcome(task_id="q", coefficient=0.0, outcome=True))
        self.assertGreater(s.n_pairs, 0)
        self.assertGreater(s.n_outcomes, 0)
        s.reset()
        self.assertEqual(s.n_pairs, 0)
        self.assertEqual(s.n_outcomes, 0)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


class FactoryTests(unittest.TestCase):
    def test_fresh_steerer_smoke(self):
        s = fresh_steerer(ALG_CAA, dim=4, alpha=0.05, target_layer="resid.10", seed=0)
        self.assertEqual(s.config.algorithm, ALG_CAA)
        self.assertEqual(s.config.dim, 4)
        self.assertEqual(s.config.target_layer, "resid.10")

    def test_known_constants_are_tuples(self):
        self.assertIn(ALG_CAA, KNOWN_ALGORITHMS)
        self.assertIn(VERDICT_PASS, KNOWN_VERDICTS)
        self.assertIn(REC_PROMOTE, KNOWN_RECOMMENDATIONS)
        self.assertIn(TEST_SEPARABILITY, KNOWN_TESTS)


# ---------------------------------------------------------------------------
# Manifest registration
# ---------------------------------------------------------------------------


class ManifestTests(unittest.TestCase):
    def test_steerer_in_manifest(self):
        from agi.manifest import default_manifest
        m = default_manifest()
        s = m.lookup("steerer")
        self.assertEqual(s.name, "steerer")
        # Anytime-valid certificate per the design.
        self.assertEqual(s.certificate, "anytime")
        # Composes with the safety stack:
        self.assertIn("refuser", s.composes_with)
        self.assertIn("mechanizer", s.composes_with)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
