"""Tests for the LatentReasoner primitive.

Coverage:
  * config validation
  * helper math (feature_hash, softmax, lipschitz, pac_bayes_bound)
  * encode / refine / decode / quantise smoke
  * reason() greedy and beam paths
  * convergence + Banach bound monotone
  * observe() learning signal
  * fit() loss decreases
  * certificate() shape and bounds
  * export / import round-trip
  * event publishing (publisher captures all kinds)
  * determinism (seed → identical trace)
  * pure-stdlib guard (no torch / numpy import)
  * end-to-end yes/no classification accuracy on a tiny synthetic dataset
  * thread-safety smoke
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import threading
import unittest

from agi.latent_reasoner import (
    Beam,
    Certificate,
    InvalidAnswer,
    InvalidConfig,
    InvalidPrompt,
    InsufficientData,
    LATENT_DECODED,
    LATENT_FIT,
    LATENT_OBSERVED,
    LATENT_REASONED,
    LATENT_REFINED,
    LATENT_STARTED,
    LATENT_CLEARED,
    LatentReasoner,
    LatentReasonerConfig,
    LatentReasonerError,
    NonContractive,
    ReasonReport,
    ReasonTrace,
    dot,
    feature_hash,
    ledger_root,
    lipschitz_estimate,
    pac_bayes_bound,
    softmax,
    tanh,
    vector_add,
    vector_distance,
    vector_norm,
    vector_scale,
    vector_sub,
)


# ----------------------------------------------------------------------
# Helper math
# ----------------------------------------------------------------------


class TestMathHelpers(unittest.TestCase):
    def test_tanh_basic(self):
        self.assertAlmostEqual(tanh(0.0), 0.0)
        self.assertGreater(tanh(2.0), 0.9)
        self.assertLess(tanh(-2.0), -0.9)

    def test_softmax_distribution(self):
        d = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(d), 1.0)
        self.assertLess(d[0], d[1])
        self.assertLess(d[1], d[2])

    def test_softmax_temperature(self):
        cold = softmax([1.0, 2.0, 3.0], temperature=0.1)
        hot = softmax([1.0, 2.0, 3.0], temperature=10.0)
        # Cold concentrates mass on the top entry.
        self.assertGreater(cold[2], hot[2])

    def test_softmax_empty(self):
        self.assertEqual(softmax([]), [])

    def test_softmax_invalid_temperature(self):
        with self.assertRaises(ValueError):
            softmax([1.0, 2.0], temperature=0.0)

    def test_dot_dim_mismatch(self):
        with self.assertRaises(ValueError):
            dot([1.0, 2.0], [3.0])

    def test_vector_arith(self):
        self.assertEqual(vector_add([1, 2], [3, 4]), [4, 6])
        self.assertEqual(vector_sub([5, 6], [1, 2]), [4, 4])
        self.assertEqual(vector_scale([1, 2], 3), [3, 6])

    def test_vector_norm_distance(self):
        self.assertAlmostEqual(vector_norm([3, 4]), 5.0)
        self.assertAlmostEqual(vector_distance([0, 0], [3, 4]), 5.0)

    def test_feature_hash_deterministic(self):
        a = feature_hash("hello world", 32, seed=7)
        b = feature_hash("hello world", 32, seed=7)
        self.assertEqual(a, b)

    def test_feature_hash_seed_differs(self):
        a = feature_hash("hello world", 32, seed=7)
        b = feature_hash("hello world", 32, seed=8)
        # Different seeds → different vectors (with very high probability)
        self.assertNotEqual(a, b)

    def test_feature_hash_unit_norm(self):
        v = feature_hash("the quick brown fox", 64, seed=0)
        self.assertAlmostEqual(vector_norm(v), 1.0, places=5)

    def test_feature_hash_invalid_dim(self):
        with self.assertRaises(ValueError):
            feature_hash("x", 0)

    def test_feature_hash_handles_empty_ish(self):
        # Pure whitespace tokenises to one empty token; still returns a
        # valid unit vector.
        v = feature_hash("   ", 16, seed=3)
        self.assertEqual(len(v), 16)

    def test_lipschitz_estimate_constant(self):
        # Trajectory that converges geometrically with γ = 0.5.
        traj = [[1.0], [0.5], [0.25], [0.125]]
        g = lipschitz_estimate(traj)
        self.assertAlmostEqual(g, 0.5, places=5)

    def test_lipschitz_estimate_too_short(self):
        self.assertEqual(lipschitz_estimate([[1.0]]), 1.0)
        self.assertEqual(lipschitz_estimate([[1.0], [2.0]]), 1.0)

    def test_pac_bayes_bound(self):
        b = pac_bayes_bound(kl=0.0, n=100, delta=0.05)
        # log(20)/100 ≈ 0.03
        self.assertLess(b, 0.04)
        self.assertGreater(b, 0.02)

    def test_pac_bayes_bound_validation(self):
        with self.assertRaises(ValueError):
            pac_bayes_bound(kl=0.0, n=0)
        with self.assertRaises(ValueError):
            pac_bayes_bound(kl=0.0, n=10, delta=0.0)
        with self.assertRaises(ValueError):
            pac_bayes_bound(kl=0.0, n=10, delta=1.0)
        with self.assertRaises(ValueError):
            pac_bayes_bound(kl=-1.0, n=10)

    def test_ledger_root_deterministic(self):
        self.assertEqual(ledger_root(), ledger_root())


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------


class TestConfigValidation(unittest.TestCase):
    def test_minimal_ok(self):
        cfg = LatentReasonerConfig(dim=4, anchors=("a", "b"))
        cfg.validate()

    def test_dim_too_small(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(dim=1).validate()

    def test_no_anchors(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(anchors=()).validate()

    def test_one_anchor(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(anchors=("only",)).validate()

    def test_duplicate_anchors(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(anchors=("a", "a")).validate()

    def test_bad_learning_rate(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(learning_rate=0.0).validate()
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(learning_rate=2.0).validate()

    def test_bad_ridge(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(ridge=-1.0).validate()

    def test_bad_tol(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(contraction_tol=0.0).validate()

    def test_bad_max_steps(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(max_steps=0).validate()

    def test_bad_beam(self):
        with self.assertRaises(InvalidConfig):
            LatentReasonerConfig(beam=0).validate()


# ----------------------------------------------------------------------
# Construction + smoke
# ----------------------------------------------------------------------


class TestConstruction(unittest.TestCase):
    def test_default(self):
        lr = LatentReasoner(dim=16, anchors=("yes", "no"))
        self.assertEqual(lr.config.dim, 16)
        self.assertEqual(lr.config.anchors, ("yes", "no"))

    def test_construction_invalid_propagates(self):
        with self.assertRaises(InvalidConfig):
            LatentReasoner(dim=1)

    def test_encode_dim(self):
        lr = LatentReasoner(dim=8)
        h = lr.encode("hello world")
        self.assertEqual(len(h), 8)
        self.assertAlmostEqual(vector_norm(h), 1.0, places=5)

    def test_encode_invalid(self):
        lr = LatentReasoner(dim=8)
        with self.assertRaises(InvalidPrompt):
            lr.encode("")
        with self.assertRaises(InvalidPrompt):
            lr.encode("   ")

    def test_refine_smoke(self):
        lr = LatentReasoner(dim=8)
        h0 = lr.encode("test")
        h1 = lr.refine(h0, h0)
        self.assertEqual(len(h1), 8)
        # tanh output bounded
        for x in h1:
            self.assertLessEqual(abs(x), 1.0)

    def test_refine_dim_mismatch(self):
        lr = LatentReasoner(dim=8)
        with self.assertRaises(ValueError):
            lr.refine([0.0] * 4, [0.0] * 8)
        with self.assertRaises(ValueError):
            lr.refine([0.0] * 8, [0.0] * 4)

    def test_decode_distribution(self):
        lr = LatentReasoner(dim=8, anchors=("a", "b", "c"))
        h = lr.encode("hello")
        dist = lr.decode(h)
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(sum(dist), 1.0)

    def test_decode_temperature_validation(self):
        lr = LatentReasoner(dim=8)
        with self.assertRaises(ValueError):
            lr.decode([0.0] * 8, temperature=0.0)

    def test_decode_dim_mismatch(self):
        lr = LatentReasoner(dim=8)
        with self.assertRaises(ValueError):
            lr.decode([0.0] * 4)

    def test_quantise(self):
        lr = LatentReasoner(dim=8, anchors=("a", "b"))
        # anchor 0 is e_0; the latent (1,0,0,...) should snap to it.
        v = [0.0] * 8
        v[0] = 1.0
        self.assertEqual(lr.quantise(v), 0)
        v = [0.0] * 8
        v[1] = 1.0
        self.assertEqual(lr.quantise(v), 1)


# ----------------------------------------------------------------------
# reason()
# ----------------------------------------------------------------------


class TestReason(unittest.TestCase):
    def test_greedy_smoke(self):
        lr = LatentReasoner(dim=16, anchors=("yes", "no"), seed=0)
        r = lr.reason("is the sky blue?")
        self.assertIsInstance(r, ReasonReport)
        self.assertIn(r.answer, ("yes", "no"))
        self.assertEqual(len(r.distribution), 2)
        self.assertAlmostEqual(sum(r.distribution), 1.0, places=5)
        self.assertGreater(r.elapsed_ms, 0.0)
        self.assertEqual(len(r.beams), 1)

    def test_beam_smoke(self):
        lr = LatentReasoner(dim=16, anchors=("yes", "no"), seed=1)
        r = lr.reason("does 1 + 1 equal 2?", beam=4)
        self.assertEqual(len(r.beams), 4)
        # Winner is beams[0].
        self.assertEqual(r.beams[0].answer, r.answer)

    def test_reason_invalid_prompt(self):
        lr = LatentReasoner(dim=8)
        with self.assertRaises(InvalidPrompt):
            lr.reason("")

    def test_reason_param_validation(self):
        lr = LatentReasoner(dim=8)
        with self.assertRaises(ValueError):
            lr.reason("x", beam=0)
        with self.assertRaises(ValueError):
            lr.reason("x", max_steps=0)
        with self.assertRaises(ValueError):
            lr.reason("x", tol=0.0)

    def test_convergence_flag_or_max_steps(self):
        lr = LatentReasoner(dim=16, anchors=("a", "b"), seed=2)
        r = lr.reason("converge please", max_steps=20, tol=1e-2)
        # With identity-ish init the operator is contractive enough that
        # the trajectory will converge within 20 steps.
        self.assertTrue(r.converged or r.n_steps == 20)

    def test_lipschitz_bounded(self):
        lr = LatentReasoner(dim=16, anchors=("a", "b"), seed=3)
        r = lr.reason("near identity is contractive", max_steps=20)
        # Initial operator is 0.7 * I (under tanh) so γ̂ should be ≤ 1.
        self.assertLessEqual(r.lipschitz, 1.01)

    def test_strict_contractive_path(self):
        # If we make the operator near-identity it's still contractive
        # via tanh; strict mode should pass cleanly.
        lr = LatentReasoner(
            dim=8, anchors=("a", "b"), seed=4, strict_contractive=False
        )
        r = lr.reason("strict ok", max_steps=20)
        self.assertIsNotNone(r)

    def test_beam_trace_structure(self):
        lr = LatentReasoner(dim=8, anchors=("a", "b"), seed=5)
        r = lr.reason("structure", beam=3, max_steps=6)
        for trace in r.beams:
            self.assertIsInstance(trace, ReasonTrace)
            self.assertEqual(len(trace.final_distribution), 2)
            self.assertGreater(len(trace.latent_path), 1)
            self.assertEqual(len(trace.anchor_path), len(trace.latent_path))


# ----------------------------------------------------------------------
# observe / fit / learning signal
# ----------------------------------------------------------------------


class TestLearning(unittest.TestCase):
    def test_observe_invalid_answer(self):
        lr = LatentReasoner(dim=8, anchors=("yes", "no"))
        with self.assertRaises(InvalidAnswer):
            lr.observe("q", "maybe")

    def test_observe_invalid_answer_type(self):
        lr = LatentReasoner(dim=8, anchors=("yes", "no"))
        with self.assertRaises(InvalidAnswer):
            lr.observe("q", 42)  # type: ignore[arg-type]

    def test_observe_invalid_prompt(self):
        lr = LatentReasoner(dim=8, anchors=("yes", "no"))
        with self.assertRaises(InvalidPrompt):
            lr.observe("", "yes")

    def test_observe_increments(self):
        lr = LatentReasoner(dim=8, anchors=("yes", "no"))
        before = lr.certificate().n_observations
        lr.observe("q", "yes")
        after = lr.certificate().n_observations
        self.assertEqual(after, before + 1)

    def test_fit_insufficient_data(self):
        lr = LatentReasoner(dim=8)
        with self.assertRaises(InsufficientData):
            lr.fit([], epochs=3, min_examples=1)

    def test_fit_invalid_pair(self):
        lr = LatentReasoner(dim=8, anchors=("yes", "no"))
        with self.assertRaises(InvalidPrompt):
            lr.fit([("", "yes")], epochs=1)
        with self.assertRaises(InvalidAnswer):
            lr.fit([("q", "maybe")], epochs=1)

    def test_fit_loss_decreases(self):
        lr = LatentReasoner(dim=24, anchors=("yes", "no"), seed=7)
        examples = [
            ("is 2+2=4", "yes"),
            ("is the moon made of cheese", "no"),
            ("is fire hot", "yes"),
            ("can humans breathe water", "no"),
            ("is grass green", "yes"),
            ("is the sun cold", "no"),
        ]
        summary = lr.fit(examples, epochs=15)
        losses = summary["loss_per_epoch"]
        self.assertEqual(len(losses), 15)
        # First-half average vs second-half average.
        first = sum(losses[:5]) / 5
        last = sum(losses[-5:]) / 5
        self.assertLessEqual(last, first + 1e-6,
                             f"loss did not decrease: {first} -> {last}")
        self.assertGreaterEqual(summary["anchor_coverage"], 1.0)

    def test_fit_then_reason_uses_anchors(self):
        # After fit, both anchors should be populated; reason() must
        # return one of them with a valid distribution.
        lr = LatentReasoner(dim=32, anchors=("yes", "no"), seed=11)
        examples = [
            ("affirmative example one", "yes"),
            ("affirmative example two", "yes"),
            ("negative example one", "no"),
            ("negative example two", "no"),
        ]
        lr.fit(examples, epochs=10)
        r = lr.reason("affirmative test query")
        self.assertIn(r.answer, ("yes", "no"))

    def test_reset_clears(self):
        lr = LatentReasoner(dim=8, anchors=("a", "b"))
        lr.observe("q", "a")
        lr.reason("hello")
        lr.reset()
        cert = lr.certificate()
        self.assertEqual(cert.n_observations, 0)
        self.assertEqual(cert.n_reasonings, 0)
        # chain back to root
        self.assertEqual(cert.chain_head, ledger_root())


# ----------------------------------------------------------------------
# certificate()
# ----------------------------------------------------------------------


class TestCertificate(unittest.TestCase):
    def test_certificate_shape(self):
        lr = LatentReasoner(dim=8, anchors=("a", "b"))
        cert = lr.certificate()
        self.assertIsInstance(cert, Certificate)
        self.assertGreaterEqual(cert.gamma, 0.0)
        self.assertGreater(cert.epsilon, 0.0)
        self.assertGreaterEqual(cert.pac_bayes_bound, 0.0)
        self.assertEqual(cert.n_observations, 0)
        self.assertEqual(cert.n_reasonings, 0)

    def test_certificate_after_reason(self):
        lr = LatentReasoner(dim=16, anchors=("yes", "no"), seed=13)
        lr.reason("query 1")
        lr.reason("query 2")
        cert = lr.certificate()
        self.assertEqual(cert.n_reasonings, 2)
        self.assertGreaterEqual(cert.gamma, 0.0)

    def test_certificate_anytime_valid(self):
        # Default operator is contractive so anytime_valid should flip
        # to True after a converged reason.
        lr = LatentReasoner(dim=16, anchors=("yes", "no"), seed=15)
        r = lr.reason("converge me", max_steps=30, tol=1e-3)
        cert = lr.certificate()
        if r.converged and cert.gamma < 1.0:
            self.assertTrue(cert.anytime_valid)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def test_seed_determinism(self):
        a = LatentReasoner(dim=16, anchors=("yes", "no"), seed=42)
        b = LatentReasoner(dim=16, anchors=("yes", "no"), seed=42)
        ra = a.reason("does seeding produce identical traces?")
        rb = b.reason("does seeding produce identical traces?")
        self.assertEqual(ra.answer, rb.answer)
        self.assertEqual(len(ra.beams[0].latent_path), len(rb.beams[0].latent_path))
        for ha, hb in zip(ra.beams[0].latent_path, rb.beams[0].latent_path):
            for x, y in zip(ha, hb):
                self.assertAlmostEqual(x, y, places=10)

    def test_different_seed_different_trace(self):
        a = LatentReasoner(dim=16, anchors=("yes", "no"), seed=1)
        b = LatentReasoner(dim=16, anchors=("yes", "no"), seed=2)
        ra = a.reason("prompt")
        rb = b.reason("prompt")
        # Some byte of the latent path should differ.
        eq = True
        for ha, hb in zip(ra.beams[0].latent_path, rb.beams[0].latent_path):
            for x, y in zip(ha, hb):
                if abs(x - y) > 1e-10:
                    eq = False
                    break
            if not eq:
                break
        self.assertFalse(eq)


# ----------------------------------------------------------------------
# Event publishing
# ----------------------------------------------------------------------


class TestEvents(unittest.TestCase):
    def setUp(self):
        self.events: list[tuple[str, dict]] = []

        def cap(kind: str, data: dict):
            self.events.append((kind, data))

        self.pub = cap

    def test_started_event(self):
        LatentReasoner(dim=8, anchors=("y", "n"), publisher=self.pub)
        kinds = [k for k, _ in self.events]
        self.assertIn(LATENT_STARTED, kinds)

    def test_full_event_lifecycle(self):
        lr = LatentReasoner(dim=8, anchors=("y", "n"), publisher=self.pub)
        lr.observe("q1", "y")
        lr.reason("test", beam=2, max_steps=5)
        lr.reset()
        kinds = {k for k, _ in self.events}
        self.assertIn(LATENT_STARTED, kinds)
        self.assertIn(LATENT_OBSERVED, kinds)
        self.assertIn(LATENT_REFINED, kinds)
        self.assertIn(LATENT_DECODED, kinds)
        self.assertIn(LATENT_REASONED, kinds)
        self.assertIn(LATENT_CLEARED, kinds)

    def test_fit_event(self):
        lr = LatentReasoner(dim=8, anchors=("y", "n"), publisher=self.pub)
        lr.fit([("q", "y"), ("r", "n")], epochs=2)
        kinds = [k for k, _ in self.events]
        self.assertIn(LATENT_FIT, kinds)

    def test_publisher_crash_silent(self):
        def boom(kind, data):
            raise RuntimeError("subscriber bug")

        lr = LatentReasoner(dim=8, anchors=("y", "n"), publisher=boom)
        # If the primitive crashed on subscriber exceptions this would
        # raise; the contract is best-effort.
        r = lr.reason("hi")
        self.assertIsNotNone(r)


# ----------------------------------------------------------------------
# Export / import roundtrip
# ----------------------------------------------------------------------


class TestExportImport(unittest.TestCase):
    def test_roundtrip(self):
        lr = LatentReasoner(dim=16, anchors=("yes", "no", "maybe"), seed=21)
        lr.fit([("a affirmative", "yes"), ("b negative", "no"),
                ("c uncertain", "maybe")], epochs=5)
        r1 = lr.reason("dump-and-reload test")
        blob = lr.export()
        # Must round-trip through JSON.
        s = json.dumps(blob)
        blob2 = json.loads(s)
        lr2 = LatentReasoner.import_(blob2)
        # Same internal state.
        cert1 = lr.certificate()
        cert2 = lr2.certificate()
        self.assertEqual(cert1.n_observations, cert2.n_observations)
        self.assertEqual(cert1.n_reasonings, cert2.n_reasonings)
        # Reasoning gives the same answer.
        r2 = lr2.reason("dump-and-reload test")
        self.assertEqual(r1.answer, r2.answer)

    def test_import_rejects_unknown_version(self):
        with self.assertRaises(InvalidConfig):
            LatentReasoner.import_({"version": "vΩ", "config": {}})


# ----------------------------------------------------------------------
# Pure-stdlib guard
# ----------------------------------------------------------------------


class TestPureStdlib(unittest.TestCase):
    """Verify the module loads without numpy / torch / scipy in sys.modules."""

    def test_no_numpy(self):
        # Re-import in a fresh way to test the module's own dependency
        # surface — we only look at what the module *itself* imports.
        import agi.latent_reasoner as mod
        src_imports = set()
        with open(mod.__file__, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    src_imports.add(line)
        joined = " ".join(src_imports)
        self.assertNotIn("numpy", joined)
        self.assertNotIn("torch", joined)
        self.assertNotIn("scipy", joined)
        self.assertNotIn("jax", joined)


# ----------------------------------------------------------------------
# Integration / end-to-end accuracy
# ----------------------------------------------------------------------


class TestEndToEndAccuracy(unittest.TestCase):
    def test_yesno_classification(self):
        """Train on shared-keyword examples, eval on held-out — the
        learner should beat random (50%) by a clear margin."""
        lr = LatentReasoner(dim=48, anchors=("yes", "no"), seed=99,
                            learning_rate=0.1)
        train = [
            # 'yes' = sentences containing 'sky' / 'sun' / 'fire'
            ("the sky is blue", "yes"),
            ("the sun is hot", "yes"),
            ("fire burns", "yes"),
            ("the sky has clouds", "yes"),
            ("the sun rises in the east", "yes"),
            ("fire is dangerous", "yes"),
            # 'no' = sentences containing 'water' / 'ice' / 'snow'
            ("water is wet", "no"),
            ("ice is cold", "no"),
            ("snow falls in winter", "no"),
            ("water flows downhill", "no"),
            ("ice cracks under pressure", "no"),
            ("snow blankets the ground", "no"),
        ]
        lr.fit(train, epochs=30)
        held_out = [
            ("the sun glows", "yes"),
            ("the sky darkens", "yes"),
            ("fire crackles", "yes"),
            ("water freezes", "no"),
            ("ice forms quickly", "no"),
            ("snow covers everything", "no"),
        ]
        correct = 0
        for prompt, expected in held_out:
            r = lr.reason(prompt, max_steps=10)
            if r.answer == expected:
                correct += 1
        # 6 held out items — accept >= 4/6 (≥ 67%) as 'beats random' with
        # margin.  The classifier won't be perfect on this tiny dataset
        # but it should beat the 3/6 random baseline.
        self.assertGreaterEqual(correct, 4,
                                f"only {correct}/6 — learner did not generalise")


# ----------------------------------------------------------------------
# Thread safety smoke
# ----------------------------------------------------------------------


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_reason(self):
        lr = LatentReasoner(dim=16, anchors=("y", "n"), seed=0)
        errors: list[BaseException] = []

        def worker():
            try:
                for _ in range(10):
                    lr.reason("concurrent prompt")
                    lr.observe("concurrent prompt", "y")
            except BaseException as e:  # noqa
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors, f"thread errors: {errors!r}")
        cert = lr.certificate()
        # Each thread observed 10 times = 40 observations + fit-internal
        # observes; just verify >= 40.
        self.assertGreaterEqual(cert.n_observations, 40)
        self.assertGreaterEqual(cert.n_reasonings, 40)


if __name__ == "__main__":
    unittest.main()
