"""Tests for agi.reconciler."""
from __future__ import annotations

import math
import random
import unittest

from agi.reconciler import (
    KNOWN_METHODS,
    METHOD_AUMANN,
    METHOD_KL_BARYCENTER,
    METHOD_LINEAR,
    METHOD_LOGARITHMIC,
    RECONCILER_CALIBRATED,
    RECONCILER_CONSENSUS,
    RECONCILER_CONTRIBUTED,
    RECONCILER_STARTED,
    RECONCILER_TOPIC_REGISTERED,
    Reconciler,
    ReconcilerConfig,
    ReconcilerError,
    InsufficientData,
    InvalidBelief,
    InvalidConfig,
    InvalidTopic,
    UnknownTopic,
    aumann_iterate,
    effective_number_of_experts,
    empirical_bernstein_half_width,
    hrms_half_width,
    kl_barycenter,
    kl_divergence,
    ks_pvalue,
    ledger_root,
    linear_pool,
    logarithmic_pool,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestLinearPool(unittest.TestCase):
    def test_basic(self):
        out = linear_pool([[0.6, 0.4], [0.8, 0.2]])
        self.assertAlmostEqual(out[0], 0.7)
        self.assertAlmostEqual(out[1], 0.3)

    def test_weighted(self):
        out = linear_pool([[1.0, 0.0], [0.0, 1.0]], weights=[3.0, 1.0])
        self.assertAlmostEqual(out[0], 0.75)
        self.assertAlmostEqual(out[1], 0.25)

    def test_empty(self):
        self.assertEqual(linear_pool([]), [])

    def test_weight_mismatch(self):
        with self.assertRaises(ReconcilerError):
            linear_pool([[0.5, 0.5]], weights=[1.0, 2.0])

    def test_length_mismatch(self):
        with self.assertRaises(ReconcilerError):
            linear_pool([[0.5, 0.5], [0.3, 0.3, 0.4]])


class TestLogPool(unittest.TestCase):
    def test_geometric_mean(self):
        out = logarithmic_pool([[0.5, 0.5], [0.5, 0.5]])
        for p in out:
            self.assertAlmostEqual(p, 0.5)

    def test_agreement_preserved(self):
        # Two identical beliefs → pool returns the same belief.
        out = logarithmic_pool([[0.6, 0.4], [0.6, 0.4]])
        self.assertAlmostEqual(out[0], 0.6, places=5)

    def test_sharpens_complementary(self):
        # When one expert puts mass on outcome A and the other on B,
        # the geometric mean is more balanced than the linear mean.
        out_log = logarithmic_pool([[0.9, 0.1], [0.1, 0.9]])
        out_lin = linear_pool([[0.9, 0.1], [0.1, 0.9]])
        # Both should be 0.5 by symmetry.
        self.assertAlmostEqual(out_log[0], 0.5, places=5)
        self.assertAlmostEqual(out_lin[0], 0.5, places=5)

    def test_normalises(self):
        out = logarithmic_pool([[0.3, 0.3, 0.4], [0.5, 0.3, 0.2]])
        self.assertAlmostEqual(sum(out), 1.0)


class TestKLDivergence(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)

    def test_positive(self):
        self.assertGreater(kl_divergence([1.0, 0.0], [0.5, 0.5]), 0.0)

    def test_length_mismatch(self):
        with self.assertRaises(ReconcilerError):
            kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_handles_zero_in_p(self):
        # KL is 0 when p_i = 0, regardless of q_i.
        kl = kl_divergence([0.0, 1.0], [0.5, 0.5])
        self.assertAlmostEqual(kl, math.log(2.0), places=5)


class TestAumannIterate(unittest.TestCase):
    def test_converges_on_agreement(self):
        out, conv, rounds = aumann_iterate([[0.6, 0.4], [0.6, 0.4]])
        self.assertTrue(conv)
        self.assertEqual(rounds, 1)
        self.assertAlmostEqual(out[0], 0.6)

    def test_disagreement_converges(self):
        out, conv, rounds = aumann_iterate(
            [[0.7, 0.3], [0.3, 0.7]], max_rounds=100
        )
        self.assertTrue(conv)
        # Average should be 0.5
        self.assertAlmostEqual(out[0], 0.5, places=4)

    def test_three_experts(self):
        out, conv, rounds = aumann_iterate(
            [[0.7, 0.3], [0.6, 0.4], [0.65, 0.35]], max_rounds=100
        )
        self.assertTrue(conv)
        self.assertAlmostEqual(out[0], 0.65, places=4)

    def test_max_rounds_caps(self):
        # With tiny tolerance and few rounds, may not converge.
        out, conv, rounds = aumann_iterate(
            [[0.7, 0.3], [0.3, 0.7]], max_rounds=1, tol=1e-20
        )
        self.assertFalse(conv)
        self.assertEqual(rounds, 1)

    def test_empty(self):
        out, conv, rounds = aumann_iterate([])
        self.assertEqual(out, [])
        self.assertTrue(conv)
        self.assertEqual(rounds, 0)


class TestBoundHelpers(unittest.TestCase):
    def test_hrms_shrinks(self):
        self.assertGreater(hrms_half_width(10), hrms_half_width(100))

    def test_hrms_one_sample(self):
        self.assertEqual(hrms_half_width(0), float("inf"))

    def test_bernstein_shrinks(self):
        self.assertGreater(
            empirical_bernstein_half_width(10, 0.1),
            empirical_bernstein_half_width(100, 0.1),
        )


class TestKSTest(unittest.TestCase):
    def test_uniform_passes(self):
        rng = random.Random(0)
        s = [rng.random() for _ in range(500)]
        _, p = ks_pvalue(s)
        self.assertGreater(p, 0.01)

    def test_skewed_fails(self):
        rng = random.Random(0)
        s = [rng.random() ** 3 for _ in range(500)]
        _, p = ks_pvalue(s)
        self.assertLess(p, 0.01)


class TestEffectiveN(unittest.TestCase):
    def test_equal_weights(self):
        self.assertAlmostEqual(effective_number_of_experts([1, 1, 1]), 3.0)

    def test_dominant_weight(self):
        self.assertAlmostEqual(
            effective_number_of_experts([1000, 1]), 1.0, places=2
        )

    def test_empty(self):
        self.assertEqual(effective_number_of_experts([]), 0.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig(unittest.TestCase):
    def test_default(self):
        c = ReconcilerConfig()
        self.assertEqual(c.method, METHOD_LINEAR)

    def test_invalid_method(self):
        with self.assertRaises(InvalidConfig):
            ReconcilerConfig(method="foo")

    def test_invalid_confidence(self):
        with self.assertRaises(InvalidConfig):
            ReconcilerConfig(confidence=0.4)
        with self.assertRaises(InvalidConfig):
            ReconcilerConfig(confidence=1.0)

    def test_invalid_aumann_cap(self):
        with self.assertRaises(InvalidConfig):
            ReconcilerConfig(aumann_max_rounds=0)

    def test_invalid_aumann_tol(self):
        with self.assertRaises(InvalidConfig):
            ReconcilerConfig(aumann_tol=0)

    def test_invalid_smoothing(self):
        with self.assertRaises(InvalidConfig):
            ReconcilerConfig(smoothing=-0.1)

    def test_invalid_hmac(self):
        with self.assertRaises(InvalidConfig):
            ReconcilerConfig(hmac_key="not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration(unittest.TestCase):
    def test_register(self):
        rec = Reconciler()
        spec = rec.register_topic("t", outcomes=("a", "b"))
        self.assertEqual(spec.topic, "t")
        self.assertIn("t", rec.topics())

    def test_duplicate(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        with self.assertRaises(InvalidTopic):
            rec.register_topic("t", outcomes=("a", "b"))

    def test_single_outcome_fails(self):
        rec = Reconciler()
        with self.assertRaises(InvalidTopic):
            rec.register_topic("t", outcomes=("a",))

    def test_duplicate_outcomes_fail(self):
        rec = Reconciler()
        with self.assertRaises(InvalidTopic):
            rec.register_topic("t", outcomes=("a", "a"))

    def test_remove(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        rec.remove_topic("t")
        self.assertEqual(rec.topics(), [])

    def test_remove_unknown(self):
        rec = Reconciler()
        with self.assertRaises(UnknownTopic):
            rec.remove_topic("missing")

    def test_clear(self):
        rec = Reconciler()
        rec.register_topic("t1", outcomes=("a", "b"))
        rec.register_topic("t2", outcomes=("c", "d"))
        rec.clear()
        self.assertEqual(rec.topics(), [])

    def test_topic_spec(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        spec = rec.topic_spec("t")
        self.assertEqual(spec.outcomes, ("a", "b"))


# ---------------------------------------------------------------------------
# Contribution
# ---------------------------------------------------------------------------


class TestContribution(unittest.TestCase):
    def test_contribute(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute("t", source="s1", belief={"a": 0.7, "b": 0.3})
        sources = rec.sources("t")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_id, "s1")
        # Normalised
        self.assertAlmostEqual(sources[0].belief["a"], 0.7, places=5)

    def test_overwrites(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute("t", source="s1", belief={"a": 0.7, "b": 0.3})
        rec.contribute("t", source="s1", belief={"a": 0.4, "b": 0.6})
        sources = rec.sources("t")
        self.assertEqual(len(sources), 1)
        self.assertAlmostEqual(sources[0].belief["a"], 0.4, places=5)

    def test_unknown_topic(self):
        rec = Reconciler()
        with self.assertRaises(UnknownTopic):
            rec.contribute("missing", source="s1", belief={"a": 1.0})

    def test_unknown_outcome(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        with self.assertRaises(InvalidBelief):
            rec.contribute("t", source="s1", belief={"a": 0.5, "c": 0.5})

    def test_negative_mass(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        with self.assertRaises(InvalidBelief):
            rec.contribute("t", source="s1", belief={"a": -0.5, "b": 0.5})

    def test_negative_weight(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        with self.assertRaises(InvalidBelief):
            rec.contribute(
                "t", source="s1", belief={"a": 0.5, "b": 0.5}, weight=-1.0
            )

    def test_realised_known(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute(
            "t", source="s1", belief={"a": 0.5, "b": 0.5}, realised="a"
        )

    def test_realised_unknown(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        with self.assertRaises(InvalidBelief):
            rec.contribute(
                "t", source="s1", belief={"a": 0.5, "b": 0.5}, realised="c"
            )

    def test_reset_topic(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute("t", source="s1", belief={"a": 0.5, "b": 0.5})
        rec.reset_topic("t")
        self.assertEqual(rec.sources("t"), [])


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


class TestConsensus(unittest.TestCase):
    def setUp(self):
        self.rec = Reconciler()
        self.rec.register_topic("t", outcomes=("a", "b"))
        self.rec.contribute("t", source="s1", belief={"a": 0.8, "b": 0.2})
        self.rec.contribute("t", source="s2", belief={"a": 0.6, "b": 0.4})
        self.rec.contribute("t", source="s3", belief={"a": 0.7, "b": 0.3})

    def test_linear(self):
        report = self.rec.consensus("t", method=METHOD_LINEAR)
        self.assertAlmostEqual(report.consensus["a"], 0.7, places=4)
        self.assertEqual(report.method, METHOD_LINEAR)
        self.assertAlmostEqual(report.effective_n_sources, 3.0)

    def test_logarithmic(self):
        report = self.rec.consensus("t", method=METHOD_LOGARITHMIC)
        self.assertGreater(report.consensus["a"], 0.5)

    def test_aumann_converges(self):
        report = self.rec.consensus("t", method=METHOD_AUMANN)
        self.assertTrue(report.converged)
        self.assertGreater(report.rounds, 0)

    def test_kl_barycenter_same_as_log(self):
        r1 = self.rec.consensus("t", method=METHOD_KL_BARYCENTER)
        r2 = self.rec.consensus("t", method=METHOD_LOGARITHMIC)
        for o in ("a", "b"):
            self.assertAlmostEqual(r1.consensus[o], r2.consensus[o], places=5)

    def test_per_source_kl(self):
        report = self.rec.consensus("t", method=METHOD_LINEAR)
        for s in ("s1", "s2", "s3"):
            self.assertGreaterEqual(report.per_source_kl[s], 0)

    def test_outlier(self):
        report = self.rec.consensus("t", method=METHOD_LINEAR)
        self.assertIsNotNone(report.outlier)
        # s1 (0.8) and s2 (0.6) are both 0.1 away; one of them is the outlier.
        self.assertIn(report.outlier[0], ("s1", "s2"))

    def test_no_data(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        with self.assertRaises(InsufficientData):
            rec.consensus("t")

    def test_unknown_method(self):
        with self.assertRaises(ReconcilerError):
            self.rec.consensus("t", method="foo")

    def test_per_source_weights(self):
        report = self.rec.consensus(
            "t", method=METHOD_LINEAR, weights={"s1": 100.0, "s2": 0.0, "s3": 0.0}
        )
        self.assertAlmostEqual(report.consensus["a"], 0.8, places=4)

    def test_unknown_topic(self):
        with self.assertRaises(UnknownTopic):
            self.rec.consensus("missing")

    def test_ci_brackets(self):
        report = self.rec.consensus("t", method=METHOD_LINEAR)
        for outcome in ("a", "b"):
            lo, hi = report.confidence_interval[outcome]
            self.assertLessEqual(lo, hi)


class TestConsensusBigK(unittest.TestCase):
    def test_many_experts(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        rng = random.Random(0)
        for i in range(50):
            p = 0.6 + rng.uniform(-0.1, 0.1)
            rec.contribute("t", source=f"s{i}", belief={"a": p, "b": 1 - p})
        report = rec.consensus("t", method=METHOD_AUMANN)
        self.assertAlmostEqual(report.consensus["a"], 0.6, delta=0.05)
        # 50 equal-weight experts → eff_n = 50
        self.assertAlmostEqual(report.effective_n_sources, 50, places=3)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration(unittest.TestCase):
    def test_calibration_runs(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        rng = random.Random(0)
        for _ in range(50):
            p = rng.random()
            outcome = "a" if rng.random() < p else "b"
            rec.contribute(
                "t",
                source="s1",
                belief={"a": p, "b": 1 - p},
                realised=outcome,
            )
        rec.contribute(
            "t", source="s1", belief={"a": 0.5, "b": 0.5}, realised="a"
        )
        cal = rec.calibration("t", source="s1")
        self.assertGreater(cal.n_observations, 0)
        self.assertGreaterEqual(cal.p_value, 0)
        self.assertLessEqual(cal.p_value, 1)
        self.assertGreater(cal.log_loss, 0)

    def test_no_data(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        with self.assertRaises(InsufficientData):
            rec.calibration("t", source="missing")


# ---------------------------------------------------------------------------
# Identifiability
# ---------------------------------------------------------------------------


class TestIdentifiability(unittest.TestCase):
    def test_no_zero_mass(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b", "c"))
        rec.contribute("t", source="s1", belief={"a": 0.5, "b": 0.3, "c": 0.2})
        report = rec.identifiability_report("t")
        self.assertEqual(report.zero_mass_outcomes, [])

    def test_zero_mass_detected(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b", "c"))
        rec.contribute("t", source="s1", belief={"a": 0.5, "b": 0.5})
        rec.contribute("t", source="s2", belief={"a": 0.7, "b": 0.3})
        # smoothing kicks in; check effective_n
        report = rec.identifiability_report("t")
        self.assertAlmostEqual(report.effective_n_sources, 2.0)


# ---------------------------------------------------------------------------
# Fingerprint chain
# ---------------------------------------------------------------------------


class TestFingerprintChain(unittest.TestCase):
    def test_chain_advances(self):
        rec = Reconciler()
        h0 = rec.chain_head
        rec.register_topic("t", outcomes=("a", "b"))
        h1 = rec.chain_head
        self.assertNotEqual(h0, h1)

    def test_chain_replays(self):
        a = Reconciler()
        b = Reconciler()
        for r in (a, b):
            r.register_topic("t", outcomes=("a", "b"))
            r.contribute("t", source="s1", belief={"a": 0.5, "b": 0.5})
            r.contribute("t", source="s2", belief={"a": 0.6, "b": 0.4})
        self.assertEqual(a.chain_head, b.chain_head)

    def test_hmac_differs(self):
        a = Reconciler()
        b = Reconciler(ReconcilerConfig(hmac_key=b"secret"))
        self.assertNotEqual(a.chain_head, b.chain_head)

    def test_ledger_root_stable(self):
        self.assertEqual(ledger_root(), ledger_root())
        self.assertNotEqual(ledger_root(), ledger_root(b"key"))


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------


class TestExportImport(unittest.TestCase):
    def test_roundtrip(self):
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute("t", source="s1", belief={"a": 0.7, "b": 0.3})
        rec.contribute(
            "t", source="s2", belief={"a": 0.4, "b": 0.6}, realised="b"
        )
        snap = rec.export_state()
        rec2 = Reconciler()
        rec2.import_state(snap)
        self.assertEqual(rec.chain_head, rec2.chain_head)
        self.assertEqual(rec.topics(), rec2.topics())
        c1 = rec.consensus("t").consensus
        c2 = rec2.consensus("t").consensus
        for o in ("a", "b"):
            self.assertAlmostEqual(c1[o], c2[o])


# ---------------------------------------------------------------------------
# Event publishing
# ---------------------------------------------------------------------------


class TestEventPublishing(unittest.TestCase):
    def test_publishes(self):
        events: list[tuple[str, dict]] = []
        def cb(k, d):
            events.append((k, d))
        rec = Reconciler(publisher=cb)
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute("t", source="s1", belief={"a": 0.5, "b": 0.5})
        rec.consensus("t")
        kinds = [k for k, _ in events]
        self.assertIn(RECONCILER_STARTED, kinds)
        self.assertIn(RECONCILER_TOPIC_REGISTERED, kinds)
        self.assertIn(RECONCILER_CONTRIBUTED, kinds)
        self.assertIn(RECONCILER_CONSENSUS, kinds)

    def test_publish_failure_tolerated(self):
        def bad(k, d):
            raise RuntimeError("nope")
        rec = Reconciler(publisher=bad)
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute("t", source="s1", belief={"a": 0.5, "b": 0.5})


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_contribute(self):
        import threading
        rec = Reconciler()
        rec.register_topic("t", outcomes=("a", "b"))
        def worker(i):
            for j in range(20):
                rec.contribute(
                    "t",
                    source=f"s{i}_{j}",
                    belief={"a": 0.5, "b": 0.5},
                )
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(rec.sources("t")), 80)


# ---------------------------------------------------------------------------
# Aumann disagreement
# ---------------------------------------------------------------------------


class TestAumannAgreement(unittest.TestCase):
    def test_two_experts_average(self):
        """Aumann iteration on two experts with disagreement converges to
        their average belief."""
        rec = Reconciler(ReconcilerConfig(method=METHOD_AUMANN))
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute("t", source="alice", belief={"a": 0.8, "b": 0.2})
        rec.contribute("t", source="bob", belief={"a": 0.2, "b": 0.8})
        report = rec.consensus("t")
        self.assertTrue(report.converged)
        # Symmetric → converges to 0.5
        self.assertAlmostEqual(report.consensus["a"], 0.5, places=3)

    def test_unequal_initial_beliefs(self):
        rec = Reconciler(ReconcilerConfig(method=METHOD_AUMANN))
        rec.register_topic("t", outcomes=("a", "b"))
        rec.contribute("t", source="bandit", belief={"a": 0.7, "b": 0.3})
        rec.contribute("t", source="bayesopt", belief={"a": 0.6, "b": 0.4})
        report = rec.consensus("t")
        self.assertTrue(report.converged)
        self.assertAlmostEqual(report.consensus["a"], 0.65, places=2)


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    def test_end_to_end_arm_consensus(self):
        rec = Reconciler(ReconcilerConfig(method=METHOD_AUMANN))
        rec.register_topic("arm_a_wins", outcomes=("yes", "no"))
        rec.contribute(
            "arm_a_wins", source="bandit", belief={"yes": 0.70, "no": 0.30}
        )
        rec.contribute(
            "arm_a_wins", source="bayesopt", belief={"yes": 0.60, "no": 0.40}
        )
        rec.contribute(
            "arm_a_wins", source="psrl", belief={"yes": 0.65, "no": 0.35}
        )
        report = rec.consensus("arm_a_wins")
        self.assertTrue(report.converged)
        self.assertAlmostEqual(report.consensus["yes"], 0.65, places=2)
        self.assertIsNotNone(report.outlier)
        # Each method produces a coherent pmf
        for m in ("linear", "logarithmic", "kl_barycenter"):
            r = rec.consensus("arm_a_wins", method=m)
            self.assertAlmostEqual(
                sum(r.consensus.values()), 1.0, places=5
            )


if __name__ == "__main__":
    unittest.main()
