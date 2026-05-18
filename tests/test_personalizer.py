"""Tests for the Personalizer online per-user preference-learning primitive.

Covers:
  * Renyi-DP conversion + budget guard
  * Pairwise / Unary signal validation, config validation
  * BTL recovers a planted user-specific axis on synthetic data
  * KTO (unary) recovers a planted user-specific axis
  * Score returns CIs and a trust verdict that escalates with samples
  * predict() returns a calibrated P(A > B)
  * promote_to_global moves the global prior
  * remove_user erases an adapter (GDPR Article 17)
  * LRU eviction when max_users is hit
  * Fingerprint chain is deterministic
  * EventBus integration is tolerant of stand-in Event classes
"""
from __future__ import annotations

import math
import random
import unittest

from agi.events import EventBus
from agi.personalizer import (
    ALG_BTL,
    ALG_KTO,
    ALG_LOGISTIC,
    KNOWN_ALGORITHMS,
    KNOWN_EVENTS,
    KNOWN_TRUSTS,
    PERS_OBSERVED,
    PERS_PROMOTED,
    PERS_REPORTED,
    PERS_RESET,
    PERS_SCORED,
    PERS_STARTED,
    PERS_USER_REMOVED,
    TRUST_BLEND,
    TRUST_FALLBACK,
    TRUST_PROMOTE,
    CandidateScore,
    DimensionMismatch,
    InvalidConfig,
    InvalidSignal,
    PairwisePreference,
    Personalizer,
    PersonalizerConfig,
    PersonalizerReport,
    PrivacyBudgetExceeded,
    UnarySignal,
    UnknownAlgorithm,
    UnknownUser,
    UserSummary,
    fresh_personalizer,
    renyi_epsilon,
    synthetic_users,
)


# ---------------------------------------------------------------------------
# DP accountant
# ---------------------------------------------------------------------------


class DPTests(unittest.TestCase):
    def test_renyi_epsilon_monotone_in_steps(self):
        a = renyi_epsilon(sigma=1.0, steps=1, delta=1e-6)
        b = renyi_epsilon(sigma=1.0, steps=100, delta=1e-6)
        self.assertLess(a, b)

    def test_renyi_epsilon_monotone_in_inv_sigma(self):
        a = renyi_epsilon(sigma=2.0, steps=10, delta=1e-6)
        b = renyi_epsilon(sigma=0.5, steps=10, delta=1e-6)
        self.assertLess(a, b)

    def test_renyi_epsilon_zero_sigma_inf(self):
        self.assertEqual(renyi_epsilon(sigma=0.0, steps=1, delta=1e-6), float("inf"))


# ---------------------------------------------------------------------------
# Config / signal validation
# ---------------------------------------------------------------------------


class ValidationTests(unittest.TestCase):
    def test_config_defaults(self):
        c = PersonalizerConfig(dim=4)
        self.assertEqual(c.algorithm, ALG_BTL)
        self.assertEqual(c.dim, 4)

    def test_unknown_algorithm(self):
        with self.assertRaises(UnknownAlgorithm):
            PersonalizerConfig(algorithm="dpo", dim=4)

    def test_bad_dim(self):
        with self.assertRaises(InvalidConfig):
            PersonalizerConfig(dim=0)

    def test_bad_lr(self):
        with self.assertRaises(InvalidConfig):
            PersonalizerConfig(dim=4, learning_rate=-1.0)

    def test_pair_dim_mismatch(self):
        with self.assertRaises(DimensionMismatch):
            p = Personalizer(PersonalizerConfig(dim=4))
            p.observe_pair(PairwisePreference(
                user_id="u",
                features_winner=(1.0, 0.0),
                features_loser=(0.0, 1.0),
            ))

    def test_pair_empty_user_id(self):
        with self.assertRaises(InvalidSignal):
            PairwisePreference(user_id="", features_winner=(1.0,), features_loser=(0.0,))

    def test_unary_confidence_range(self):
        with self.assertRaises(InvalidSignal):
            UnarySignal(user_id="u", features=(1.0,), desirable=True, confidence=1.5)

    def test_observe_pair_requires_btl(self):
        with self.assertRaises(InvalidSignal):
            p = Personalizer(PersonalizerConfig(algorithm=ALG_KTO, dim=2))
            p.observe_pair(PairwisePreference(
                user_id="u",
                features_winner=(1.0, 0.0),
                features_loser=(0.0, 1.0),
            ))


# ---------------------------------------------------------------------------
# BTL recovers per-user axis
# ---------------------------------------------------------------------------


class BTLLearningTests(unittest.TestCase):
    def _build(self, axis, n_prefs=80, seed=0):
        p = Personalizer(PersonalizerConfig(
            algorithm=ALG_BTL, dim=4,
            learning_rate=0.1, ridge=0.001,
            min_observations_for_trust=10,
            ci_half_width_promote=0.3,
            seed=seed,
        ))
        rng = random.Random(seed)
        for _ in range(n_prefs):
            v1 = tuple(rng.gauss(0, 1) for _ in range(4))
            v2 = tuple(rng.gauss(0, 1) for _ in range(4))
            if v1[axis] > v2[axis]:
                p.observe_pair(PairwisePreference(
                    user_id="u", features_winner=v1, features_loser=v2,
                ))
            else:
                p.observe_pair(PairwisePreference(
                    user_id="u", features_winner=v2, features_loser=v1,
                ))
        return p

    def test_learns_axis_0(self):
        p = self._build(axis=0, seed=1)
        scores = p.score("u", [(1.0, 0, 0, 0), (-1.0, 0, 0, 0)])
        self.assertGreater(scores[0].mean, scores[1].mean)
        self.assertGreater(scores[0].mean, 0.6)

    def test_learns_axis_3(self):
        p = self._build(axis=3, seed=2)
        scores = p.score("u", [(0, 0, 0, 1.0), (0, 0, 0, -1.0)])
        self.assertGreater(scores[0].mean, scores[1].mean)

    def test_trust_escalates_with_samples(self):
        p = Personalizer(PersonalizerConfig(
            algorithm=ALG_BTL, dim=2,
            min_observations_for_trust=5,
            ci_half_width_promote=0.4,
            seed=0,
        ))
        # No observations
        self.assertEqual(p.trust("u"), TRUST_FALLBACK)
        rng = random.Random(0)
        for _ in range(60):
            v1 = tuple(rng.gauss(0, 1) for _ in range(2))
            v2 = tuple(rng.gauss(0, 1) for _ in range(2))
            if v1[0] > v2[0]:
                p.observe_pair(PairwisePreference("u", v1, v2))
            else:
                p.observe_pair(PairwisePreference("u", v2, v1))
        self.assertIn(p.trust("u"), (TRUST_BLEND, TRUST_PROMOTE))

    def test_score_returns_candidate_scores_with_ci(self):
        p = self._build(axis=0, seed=3)
        scores = p.score("u", [(1.0, 0, 0, 0), (0, 0, 0, 0)])
        for s in scores:
            self.assertIsInstance(s, CandidateScore)
            self.assertLessEqual(s.ci_low, s.mean)
            self.assertGreaterEqual(s.ci_high, s.mean)
            self.assertIn(s.trust, KNOWN_TRUSTS)

    def test_predict_pairwise(self):
        p = self._build(axis=0, seed=4)
        prob, lo, hi, trust = p.predict("u", (1.0, 0, 0, 0), (-1.0, 0, 0, 0))
        self.assertGreater(prob, 0.5)
        self.assertLessEqual(lo, prob)
        self.assertGreaterEqual(hi, prob)


# ---------------------------------------------------------------------------
# KTO / unary learning
# ---------------------------------------------------------------------------


class UnaryLearningTests(unittest.TestCase):
    def test_kto_learns_axis(self):
        p = Personalizer(PersonalizerConfig(
            algorithm=ALG_KTO, dim=3, learning_rate=0.1, ridge=0.001, seed=0,
        ))
        rng = random.Random(0)
        for _ in range(60):
            f = tuple(rng.gauss(0, 1) for _ in range(3))
            desirable = f[1] > 0
            p.observe_unary(UnarySignal(user_id="u", features=f, desirable=desirable))
        s = p.score("u", [(0, 1.0, 0), (0, -1.0, 0)])
        self.assertGreater(s[0].mean, s[1].mean)


# ---------------------------------------------------------------------------
# Promote and remove
# ---------------------------------------------------------------------------


class PromoteRemoveTests(unittest.TestCase):
    def _build(self):
        p = Personalizer(PersonalizerConfig(
            algorithm=ALG_BTL, dim=3, learning_rate=0.1, ridge=0.001, seed=0,
        ))
        rng = random.Random(0)
        for _ in range(40):
            v1 = tuple(rng.gauss(0, 1) for _ in range(3))
            v2 = tuple(rng.gauss(0, 1) for _ in range(3))
            if v1[0] > v2[0]:
                p.observe_pair(PairwisePreference("u", v1, v2))
            else:
                p.observe_pair(PairwisePreference("u", v2, v1))
        return p

    def test_promote_moves_global(self):
        p = self._build()
        old_global = p.global_theta
        new_global = p.promote_to_global("u", blend=1.0)
        self.assertEqual(new_global, p.user_summary("u").theta)
        self.assertNotEqual(old_global, new_global)

    def test_remove_user_existing(self):
        p = self._build()
        self.assertTrue(p.remove_user("u"))
        with self.assertRaises(UnknownUser):
            p.user_summary("u")

    def test_remove_user_unknown_returns_false(self):
        p = Personalizer(PersonalizerConfig(dim=3))
        self.assertFalse(p.remove_user("ghost"))

    def test_promote_unknown_user_raises(self):
        p = Personalizer(PersonalizerConfig(dim=3))
        with self.assertRaises(UnknownUser):
            p.promote_to_global("ghost")

    def test_promote_bad_blend(self):
        p = self._build()
        with self.assertRaises(InvalidSignal):
            p.promote_to_global("u", blend=1.5)


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class LRUTests(unittest.TestCase):
    def test_lru_eviction_when_max_users_exceeded(self):
        p = Personalizer(PersonalizerConfig(dim=2, max_users=3))
        for uid in ("a", "b", "c", "d"):
            p.observe_pair(PairwisePreference(
                uid, (1.0, 0.0), (0.0, 1.0),
            ))
        self.assertEqual(p.n_users, 3)
        # `a` should have been evicted (oldest).
        with self.assertRaises(UnknownUser):
            p.user_summary("a")


# ---------------------------------------------------------------------------
# Fingerprint chain
# ---------------------------------------------------------------------------


class ReplayTests(unittest.TestCase):
    def test_same_observations_same_chain(self):
        cfg = dict(algorithm=ALG_BTL, dim=2, seed=0)
        p1 = Personalizer(PersonalizerConfig(**cfg))
        p2 = Personalizer(PersonalizerConfig(**cfg))
        for i in range(6):
            pref = PairwisePreference(
                user_id="u",
                features_winner=(1.0 + i * 0.1, 0.0),
                features_loser=(0.0, 1.0 - i * 0.1),
            )
            p1.observe_pair(pref)
            p2.observe_pair(pref)
        self.assertEqual(p1.fingerprint, p2.fingerprint)


# ---------------------------------------------------------------------------
# DP budget
# ---------------------------------------------------------------------------


class DPBudgetTests(unittest.TestCase):
    def test_dp_increments_epsilon(self):
        p = Personalizer(PersonalizerConfig(
            algorithm=ALG_BTL, dim=2, dp_sigma=1.0, dp_delta=1e-6, seed=0,
        ))
        # No DP exception, but epsilon climbs.
        for _ in range(10):
            p.observe_pair(PairwisePreference(
                "u", (1.0, 0.0), (0.0, 1.0),
            ))
        eps = p.user_summary("u").epsilon_spent
        self.assertGreater(eps, 0)

    def test_dp_budget_exceeded_raises(self):
        p = Personalizer(PersonalizerConfig(
            algorithm=ALG_BTL, dim=2,
            dp_sigma=0.5, dp_epsilon_target=0.5, dp_delta=1e-6, seed=0,
        ))
        with self.assertRaises(PrivacyBudgetExceeded):
            for _ in range(50):
                p.observe_pair(PairwisePreference(
                    "u", (1.0, 0.0), (0.0, 1.0),
                ))


# ---------------------------------------------------------------------------
# EventBus integration
# ---------------------------------------------------------------------------


class EventTests(unittest.TestCase):
    def test_known_events_cover_emit_paths(self):
        for ev in (PERS_STARTED, PERS_OBSERVED, PERS_SCORED, PERS_PROMOTED,
                   PERS_REPORTED, PERS_RESET, PERS_USER_REMOVED):
            self.assertIn(ev, KNOWN_EVENTS)

    def test_bus_receives_events(self):
        bus = EventBus()
        seen = []
        bus.subscribe(lambda e: seen.append(e.kind))
        p = Personalizer(PersonalizerConfig(dim=2), bus=bus)
        p.observe_pair(PairwisePreference("u", (1.0, 0.0), (0.0, 1.0)))
        p.score("u", [(1.0, 0.0)])
        p.report()
        p.reset()
        self.assertIn(PERS_STARTED, seen)
        self.assertIn(PERS_OBSERVED, seen)
        self.assertIn(PERS_SCORED, seen)
        self.assertIn(PERS_REPORTED, seen)
        self.assertIn(PERS_RESET, seen)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class ReportTests(unittest.TestCase):
    def test_report_shape(self):
        p = Personalizer(PersonalizerConfig(dim=2, seed=0))
        p.observe_pair(PairwisePreference("u", (1.0, 0.0), (0.0, 1.0)))
        r = p.report()
        self.assertIsInstance(r, PersonalizerReport)
        self.assertEqual(r.n_users, 1)
        self.assertEqual(r.total_observations, 1)
        self.assertEqual(len(r.per_user), 1)
        self.assertIsInstance(r.per_user[0], UserSummary)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


class FactoryTests(unittest.TestCase):
    def test_fresh_personalizer_smoke(self):
        p = fresh_personalizer(ALG_BTL, dim=4)
        self.assertEqual(p.config.algorithm, ALG_BTL)
        self.assertEqual(p.config.dim, 4)

    def test_synthetic_users_returns_well_formed_prefs(self):
        prefs, truths = synthetic_users(n_users=3, n_prefs_per_user=10, dim=4, seed=0)
        self.assertEqual(len(prefs), 30)
        self.assertEqual(len(truths), 3)
        # Every preference references some u<idx>
        ids = {pp.user_id for pp in prefs}
        self.assertTrue(all(uid.startswith("u") for uid in ids))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class ManifestTests(unittest.TestCase):
    def test_personalizer_in_manifest(self):
        from agi.manifest import default_manifest
        m = default_manifest()
        s = m.lookup("personalizer")
        self.assertEqual(s.name, "personalizer")
        self.assertIn("aligner", s.composes_with)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
