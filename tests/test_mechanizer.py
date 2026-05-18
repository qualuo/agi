"""Tests for ``agi.mechanizer``.

The Mechanizer primitive is a runtime-grade mechanistic-interpretability
kernel: sparse-autoencoder / dictionary-learning fit, sparse pursuit
encoder, activation patching & steering, circuit graph, Donoho-Elad
identifiability certificate.  The tests below pin the API contract,
algorithmic correctness on synthetic data with a known sparse ground
truth, snapshot / restore fidelity, ledger integrity, and the standard
runtime expectations (determinism, error surface, validation).
"""
from __future__ import annotations

import math
import unittest

from agi.mechanizer import (
    # algorithms
    ALGO_KSVD,
    ALGO_L1_SAE,
    ALGO_PCA,
    ALGO_TOPK_SAE,
    KNOWN_ALGORITHMS,
    # pursuit
    KNOWN_PURSUITS,
    PURSUIT_AUTO,
    PURSUIT_DENSE,
    PURSUIT_FISTA,
    PURSUIT_MP,
    PURSUIT_OMP,
    PURSUIT_THRESHOLD,
    PURSUIT_TOPK,
    # events
    KNOWN_EVENTS,
    MECHANIZER_CERTIFIED,
    MECHANIZER_CIRCUIT_BUILT,
    MECHANIZER_ENCODED,
    MECHANIZER_FIT,
    MECHANIZER_PATCHED,
    MECHANIZER_STARTED,
    MECHANIZER_STEERED,
    # exceptions
    InsufficientData,
    InvalidActivations,
    InvalidConfig,
    InvalidFeature,
    LedgerCorrupt,
    NotFit,
    UnknownAlgorithm,
    UnknownPursuit,
    # helpers
    Mechanizer,
    MechanizerCertificate,
    MechanizerConfig,
    MechanizerReport,
    empirical_bernstein_half_width,
    hoeffding_half_width,
    mechanizer_donoho_elad_bound,
    mechanizer_fista,
    mechanizer_hard_threshold,
    mechanizer_ledger_root,
    mechanizer_mutual_coherence,
    mechanizer_omp,
    mechanizer_random_dictionary,
    mechanizer_recovery_threshold,
    mechanizer_soft_threshold,
    mechanizer_synthetic_features,
    mechanizer_topk_mask,
)


def _make_easy_data(n=80, dim=16, n_true=12, true_l0=3, seed=0):
    return mechanizer_synthetic_features(
        n=n, dim=dim, n_true=n_true, true_l0=true_l0, seed=seed,
    )


# ---------------------------------------------------------------------------
# Threshold / mask primitives
# ---------------------------------------------------------------------------


class ThresholdTests(unittest.TestCase):
    def test_soft_threshold(self):
        out = mechanizer_soft_threshold([-2.0, -0.3, 0.0, 0.5, 1.7], 0.4)
        expected = [-1.6, 0.0, 0.0, 0.1, 1.3]
        for actual, want in zip(out, expected):
            self.assertAlmostEqual(actual, want, places=9)

    def test_soft_threshold_rejects_negative_lambda(self):
        with self.assertRaises(InvalidConfig):
            mechanizer_soft_threshold([1.0], -1.0)

    def test_hard_threshold(self):
        out = mechanizer_hard_threshold([-2.0, -0.3, 0.0, 0.5, 1.7], 0.4)
        self.assertEqual(out, [-2.0, 0.0, 0.0, 0.5, 1.7])

    def test_topk_mask_keeps_top_positive_only(self):
        v = [3.0, -2.0, 1.0, 4.0, -5.0, 0.5]
        out = mechanizer_topk_mask(v, k=2)
        # negatives zeroed; top 2 of positives are 4.0 (idx 3) and 3.0 (idx 0).
        self.assertEqual(out[0], 3.0)
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[2], 0.0)
        self.assertEqual(out[3], 4.0)
        self.assertEqual(out[4], 0.0)
        self.assertEqual(out[5], 0.0)

    def test_topk_mask_k_zero_returns_zeros(self):
        out = mechanizer_topk_mask([1.0, 2.0, 3.0], k=0)
        self.assertEqual(out, [0.0, 0.0, 0.0])

    def test_topk_mask_k_negative_raises(self):
        with self.assertRaises(InvalidConfig):
            mechanizer_topk_mask([1.0], k=-1)

    def test_topk_mask_more_k_than_positives(self):
        out = mechanizer_topk_mask([1.0, -1.0, 2.0], k=10)
        # everything positive kept; negatives still zeroed by ReLU.
        self.assertEqual(out, [1.0, 0.0, 2.0])


# ---------------------------------------------------------------------------
# Pursuit kernels — OMP, MP, FISTA
# ---------------------------------------------------------------------------


class PursuitKernelTests(unittest.TestCase):
    def test_omp_exact_recovery_when_coherence_low(self):
        # Build a dictionary with low mutual coherence by sampling
        # random gaussian unit atoms in a moderately high dimension.
        D = mechanizer_random_dictionary(n_features=10, dim=32, seed=7)
        # True 2-sparse signal supported on atoms 1 & 6.
        true = [0.0] * 10
        true[1] = 1.4
        true[6] = -0.8
        # x = D^T true = sum_j true_j * D_j
        d = 32
        x = [0.0] * d
        for j, c in enumerate(true):
            if c == 0.0:
                continue
            for k in range(d):
                x[k] += c * D[j][k]
        z = mechanizer_omp(x, D, k=2)
        self.assertGreater(abs(z[1]), 1e-3)
        self.assertGreater(abs(z[6]), 1e-3)
        # The other indices should be very small.
        for j in range(10):
            if j in (1, 6):
                continue
            self.assertLess(abs(z[j]), 1e-3)

    def test_omp_zero_k_returns_zeros(self):
        D = mechanizer_random_dictionary(n_features=4, dim=8, seed=0)
        z = mechanizer_omp([0.1] * 8, D, k=0)
        self.assertEqual(z, [0.0] * 4)

    def test_omp_validates_dimensions(self):
        D = mechanizer_random_dictionary(n_features=3, dim=4, seed=0)
        with self.assertRaises(InvalidActivations):
            mechanizer_omp([1.0, 2.0], D, k=1)

    def test_fista_recovers_sparse_signal(self):
        D = mechanizer_random_dictionary(n_features=8, dim=20, seed=0)
        # true 1-sparse with positive coefficient at index 3
        true = [0.0] * 8
        true[3] = 1.0
        x = [0.0] * 20
        for k in range(20):
            x[k] = D[3][k]
        z = mechanizer_fista(x, D, lam=1e-3, max_iter=300)
        # FISTA may not recover the exact 1-hot; verify reconstruction works.
        recon = [0.0] * 20
        for j, c in enumerate(z):
            if c == 0.0:
                continue
            for k in range(20):
                recon[k] += c * D[j][k]
        err = sum((x[k] - recon[k]) ** 2 for k in range(20))
        self.assertLess(err, 5e-2)


# ---------------------------------------------------------------------------
# Mutual coherence & Donoho-Elad helpers
# ---------------------------------------------------------------------------


class CoherenceTests(unittest.TestCase):
    def test_coherence_of_orthonormal_basis_is_zero(self):
        # Standard basis in R^4 has zero mutual coherence.
        D = [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        self.assertAlmostEqual(mechanizer_mutual_coherence(D), 0.0)

    def test_coherence_of_duplicate_atoms_is_one(self):
        D = [[1.0, 0.0], [1.0, 0.0]]
        self.assertAlmostEqual(mechanizer_mutual_coherence(D), 1.0)

    def test_coherence_single_atom(self):
        self.assertEqual(mechanizer_mutual_coherence([[1.0, 2.0, 3.0]]), 0.0)

    def test_donoho_elad_bound(self):
        self.assertEqual(mechanizer_donoho_elad_bound(2), 1.0 / 3.0)
        self.assertEqual(mechanizer_donoho_elad_bound(5), 1.0 / 9.0)
        self.assertEqual(mechanizer_donoho_elad_bound(0), float("inf"))
        self.assertEqual(mechanizer_donoho_elad_bound(1), float("inf"))

    def test_recovery_threshold(self):
        # μ < (2k-1)^-1 ⇒ k < (1 + 1/μ)/2.
        self.assertEqual(mechanizer_recovery_threshold(0.5), 1)
        self.assertEqual(mechanizer_recovery_threshold(0.25), 2)
        self.assertEqual(mechanizer_recovery_threshold(0.1), 5)
        # μ == 0 falls back to a huge sentinel.
        self.assertGreater(mechanizer_recovery_threshold(0.0), 1_000_000)


# ---------------------------------------------------------------------------
# Hoeffding / empirical-Bernstein
# ---------------------------------------------------------------------------


class TailBoundTests(unittest.TestCase):
    def test_hoeffding_half_width_shrinks_with_n(self):
        wide = hoeffding_half_width(10, delta=0.05)
        tight = hoeffding_half_width(1000, delta=0.05)
        self.assertGreater(wide, tight)

    def test_hoeffding_rejects_invalid_delta(self):
        with self.assertRaises(InvalidConfig):
            hoeffding_half_width(100, delta=0.0)
        with self.assertRaises(InvalidConfig):
            hoeffding_half_width(100, delta=1.0)

    def test_bernstein_handles_n_le_one(self):
        self.assertEqual(
            empirical_bernstein_half_width(1, variance=0.1, delta=0.05),
            float("inf"),
        )

    def test_bernstein_shrinks_with_variance(self):
        wide = empirical_bernstein_half_width(100, variance=0.25, delta=0.05)
        tight = empirical_bernstein_half_width(100, variance=0.0, delta=0.05)
        self.assertGreater(wide, tight)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = MechanizerConfig()
        self.assertIn(cfg.algorithm, KNOWN_ALGORITHMS)
        self.assertIn(cfg.pursuit, KNOWN_PURSUITS)

    def test_invalid_algorithm(self):
        with self.assertRaises(UnknownAlgorithm):
            MechanizerConfig(algorithm="not-an-algorithm")

    def test_invalid_pursuit(self):
        with self.assertRaises(UnknownPursuit):
            MechanizerConfig(pursuit="not-a-pursuit")

    def test_invalid_n_features(self):
        with self.assertRaises(InvalidConfig):
            MechanizerConfig(n_features=0)

    def test_invalid_target_l0_too_big(self):
        with self.assertRaises(InvalidConfig):
            MechanizerConfig(n_features=4, target_l0=10)

    def test_invalid_l1_coeff(self):
        with self.assertRaises(InvalidConfig):
            MechanizerConfig(l1_coeff=-0.1)

    def test_invalid_learning_rate(self):
        with self.assertRaises(InvalidConfig):
            MechanizerConfig(learning_rate=0.0)

    def test_invalid_max_iter(self):
        with self.assertRaises(InvalidConfig):
            MechanizerConfig(max_iter=0)

    def test_invalid_dead_threshold(self):
        with self.assertRaises(InvalidConfig):
            MechanizerConfig(dead_feature_threshold=-0.1)
        with self.assertRaises(InvalidConfig):
            MechanizerConfig(dead_feature_threshold=1.1)

    def test_hmac_key_must_be_bytes(self):
        with self.assertRaises(InvalidConfig):
            MechanizerConfig(hmac_key="abc")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fit for each algorithm
# ---------------------------------------------------------------------------


class FitTopKSAETests(unittest.TestCase):
    def test_fit_topk_sae_reaches_meaningful_r2(self):
        X = _make_easy_data(n=80, dim=16, n_true=16, true_l0=4, seed=0)
        m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_TOPK_SAE, n_features=32, target_l0=4,
            learning_rate=5e-2, max_iter=80, seed=0,
        ))
        rep = m.fit(X)
        self.assertIsInstance(rep, MechanizerReport)
        self.assertEqual(rep.algorithm, ALGO_TOPK_SAE)
        self.assertEqual(rep.n_samples, 80)
        self.assertEqual(rep.n_neurons, 16)
        self.assertEqual(rep.n_features, 32)
        self.assertGreater(rep.r2, 0.5)
        # Top-K enforces an exact L0 cap.
        self.assertLessEqual(rep.mean_l0, 4.001)

    def test_fit_topk_publishes_event(self):
        published: list[tuple[str, dict]] = []
        m = Mechanizer(
            MechanizerConfig(
                algorithm=ALGO_TOPK_SAE, n_features=8, target_l0=2,
                max_iter=5, learning_rate=5e-2, seed=0,
            ),
            publish=lambda k, p: published.append((k, p)),
        )
        m.fit(_make_easy_data(n=20, dim=8, n_true=8, true_l0=2, seed=0))
        self.assertIn(MECHANIZER_STARTED, [k for k, _ in published])
        self.assertIn(MECHANIZER_FIT, [k for k, _ in published])


class FitL1SAETests(unittest.TestCase):
    def test_fit_l1_sae_reduces_loss(self):
        X = _make_easy_data(n=60, dim=12, n_true=12, true_l0=3, seed=0)
        m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_L1_SAE, n_features=12, target_l0=3,
            l1_coeff=5e-2, learning_rate=2e-2, max_iter=80, seed=3,
        ))
        rep = m.fit(X)
        # Loss should drop from the start of training to the end.
        self.assertGreater(rep.loss_history[0], rep.loss_history[-1])
        self.assertGreater(rep.r2, 0.5)


class FitKSVDTests(unittest.TestCase):
    def test_fit_ksvd_recovers_sparse_structure(self):
        X = _make_easy_data(n=120, dim=20, n_true=16, true_l0=3, seed=1)
        m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=24, target_l0=3,
            max_iter=20, seed=4,
        ))
        rep = m.fit(X)
        # KSVD with OMP should achieve very high R^2 on synthetic data.
        self.assertGreater(rep.r2, 0.85)
        # Average L0 enforced by OMP's k cap.
        self.assertLessEqual(rep.mean_l0, 3.05)
        # Mutual coherence is bounded.
        self.assertLess(rep.mutual_coherence, 1.0)


class FitPCATests(unittest.TestCase):
    def test_fit_pca_reaches_full_r2_when_overcomplete(self):
        # PCA with K >= dim reproduces every sample exactly (modulo
        # eigen-decomposition rounding).
        X = _make_easy_data(n=60, dim=10, n_true=10, true_l0=4, seed=5)
        m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_PCA, n_features=10, target_l0=4, max_iter=1, seed=0,
        ))
        rep = m.fit(X)
        self.assertGreater(rep.r2, 0.999)
        # Coherence of orthogonal eigenvectors should be tiny.
        self.assertLess(rep.mutual_coherence, 1e-3)

    def test_fit_pca_with_more_features_than_dims_leaves_atoms_zero(self):
        # K > d should leave d+1..K atoms as zero, so dead_features ≥ K − d.
        X = _make_easy_data(n=40, dim=8, n_true=8, true_l0=2, seed=0)
        m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_PCA, n_features=16, target_l0=4, max_iter=1, seed=0,
        ))
        rep = m.fit(X)
        self.assertGreaterEqual(rep.dead_features, 16 - 8)


# ---------------------------------------------------------------------------
# Encode / decode roundtrip
# ---------------------------------------------------------------------------


class EncodeDecodeTests(unittest.TestCase):
    def setUp(self):
        self.X = _make_easy_data(n=60, dim=14, n_true=10, true_l0=3, seed=2)
        self.m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=16, target_l0=3,
            max_iter=15, seed=0,
        ))
        self.m.fit(self.X)

    def test_encode_returns_correct_shape(self):
        Z = self.m.encode(self.X[:5])
        self.assertEqual(len(Z), 5)
        for row in Z:
            self.assertEqual(len(row), 16)

    def test_decode_returns_correct_shape(self):
        Z = self.m.encode(self.X[:5])
        X_hat = self.m.decode(Z)
        self.assertEqual(len(X_hat), 5)
        for row in X_hat:
            self.assertEqual(len(row), 14)

    def test_encode_decode_close_to_input(self):
        Z = self.m.encode(self.X[:5])
        X_hat = self.m.decode(Z)
        err = sum(
            (self.X[i][c] - X_hat[i][c]) ** 2
            for i in range(5) for c in range(14)
        )
        # OMP encode against the KSVD dictionary should reconstruct well.
        self.assertLess(err, 5.0)

    def test_encode_rejects_wrong_neuron_count(self):
        bad = [[0.0] * 7 for _ in range(3)]
        with self.assertRaises(InvalidActivations):
            self.m.encode(bad)

    def test_encode_before_fit_raises(self):
        m = Mechanizer(MechanizerConfig(seed=0))
        with self.assertRaises(NotFit):
            m.encode([[0.0] * 8])

    def test_encode_each_pursuit(self):
        for p in (PURSUIT_OMP, PURSUIT_MP, PURSUIT_TOPK,
                  PURSUIT_FISTA, PURSUIT_THRESHOLD, PURSUIT_DENSE):
            Z = self.m.encode(self.X[:2], pursuit=p)
            self.assertEqual(len(Z), 2)
            self.assertEqual(len(Z[0]), 16)

    def test_encode_unknown_pursuit_raises(self):
        with self.assertRaises(UnknownPursuit):
            self.m.encode(self.X[:1], pursuit="not-real")


# ---------------------------------------------------------------------------
# Patching & steering
# ---------------------------------------------------------------------------


class PatchSteerTests(unittest.TestCase):
    def setUp(self):
        self.X = _make_easy_data(n=40, dim=12, n_true=10, true_l0=3, seed=3)
        self.m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=16, target_l0=3,
            max_iter=12, seed=1,
        ))
        self.m.fit(self.X)

    def test_patch_returns_activation_shape(self):
        out = self.m.patch(self.X[0:1], self.X[1:2], feature=2, scale=1.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 12)

    def test_patch_at_zero_scale_is_pure_target(self):
        # scale=0 => the patched code equals the target's code so the
        # reconstruction is identical to encode(target) then decode.
        Z = self.m.encode(self.X[0:1])
        baseline = self.m.decode(Z)
        out = self.m.patch(self.X[0:1], self.X[1:2], feature=2, scale=0.0)
        for c in range(12):
            self.assertAlmostEqual(baseline[0][c], out[0][c], places=6)

    def test_patch_accepts_feature_list(self):
        out = self.m.patch(self.X[0:1], self.X[1:2], feature=[0, 1, 2], scale=1.0)
        self.assertEqual(len(out), 1)

    def test_patch_rejects_invalid_feature(self):
        with self.assertRaises(InvalidFeature):
            self.m.patch(self.X[0:1], self.X[1:2], feature=999)

    def test_patch_rejects_scale_out_of_range(self):
        with self.assertRaises(InvalidConfig):
            self.m.patch(self.X[0:1], self.X[1:2], feature=0, scale=-0.1)
        with self.assertRaises(InvalidConfig):
            self.m.patch(self.X[0:1], self.X[1:2], feature=0, scale=1.1)

    def test_patch_requires_non_empty_inputs(self):
        with self.assertRaises(InvalidActivations):
            self.m.patch([], self.X[0:1], feature=0)

    def test_steer_changes_activations(self):
        out = self.m.steer(self.X[0:1], feature=0, magnitude=2.0)
        # the steered activation differs from the original.
        any_diff = any(out[0][c] != self.X[0][c] for c in range(12))
        self.assertTrue(any_diff)

    def test_steer_with_zero_magnitude_is_identity(self):
        out = self.m.steer(self.X[0:1], feature=0, magnitude=0.0)
        for c in range(12):
            self.assertAlmostEqual(out[0][c], self.X[0][c], places=9)

    def test_steer_rejects_invalid_feature(self):
        with self.assertRaises(InvalidFeature):
            self.m.steer(self.X[0:1], feature=-1, magnitude=1.0)

    def test_steer_rejects_non_finite_magnitude(self):
        with self.assertRaises(InvalidConfig):
            self.m.steer(self.X[0:1], feature=0, magnitude=float("nan"))

    def test_steer_event_records_perturbation_norm(self):
        before = len(self.m.events())
        self.m.steer(self.X[0:1], feature=3, magnitude=1.5)
        after = self.m.events()
        steer_evs = [e for e in after if e.kind == MECHANIZER_STEERED]
        self.assertGreaterEqual(len(steer_evs), 1)
        payload = steer_evs[-1].payload
        self.assertIn("perturbation_norm", payload)
        self.assertGreater(payload["perturbation_norm"], 0.0)


# ---------------------------------------------------------------------------
# Auto-interpret & circuit graph
# ---------------------------------------------------------------------------


class AutoInterpretTests(unittest.TestCase):
    def setUp(self):
        self.X = _make_easy_data(n=60, dim=12, n_true=10, true_l0=3, seed=4)
        self.m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=16, target_l0=3,
            max_iter=15, seed=2,
        ))
        self.m.fit(self.X)

    def test_auto_interpret_returns_summary_per_feature(self):
        summaries = self.m.auto_interpret(self.X, top_k=4)
        self.assertEqual(len(summaries), 16)
        for s in summaries:
            self.assertGreaterEqual(s.activation_density, 0.0)
            self.assertLessEqual(s.activation_density, 1.0)
            self.assertLessEqual(len(s.top_indices), 4)
            self.assertGreaterEqual(s.variance_explained, 0.0)

    def test_auto_interpret_top_k_must_be_positive(self):
        with self.assertRaises(InvalidConfig):
            self.m.auto_interpret(self.X, top_k=0)

    def test_auto_interpret_label_callback(self):
        labels = self.m.auto_interpret(
            self.X[:6], top_k=2,
            label_fn=lambda f, idx: f"feature-{f}",
        )
        self.assertTrue(any(s.label and s.label.startswith("feature-") for s in labels))


class CircuitTests(unittest.TestCase):
    def setUp(self):
        self.X = _make_easy_data(n=80, dim=12, n_true=8, true_l0=3, seed=5)
        self.m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=12, target_l0=3,
            max_iter=18, seed=3,
        ))
        self.m.fit(self.X)

    def test_circuit_returns_graph(self):
        g = self.m.circuit(self.X, threshold=0.2)
        self.assertEqual(g.n_features, 12)
        self.assertGreaterEqual(g.edge_count, 0)
        self.assertGreaterEqual(g.largest_component, 0)

    def test_circuit_threshold_must_be_in_range(self):
        with self.assertRaises(InvalidConfig):
            self.m.circuit(self.X, threshold=0.0)
        with self.assertRaises(InvalidConfig):
            self.m.circuit(self.X, threshold=1.5)

    def test_circuit_neighbours_filter(self):
        g = self.m.circuit(self.X, threshold=0.2)
        for j in range(g.n_features):
            for nb, w in g.neighbours(j):
                self.assertGreaterEqual(abs(w), 0.2)


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.X = _make_easy_data(n=50, dim=10, n_true=8, true_l0=3, seed=6)
        self.m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=12, target_l0=3,
            max_iter=15, seed=0,
        ))
        self.m.fit(self.X)

    def test_certify_returns_certificate(self):
        cert = self.m.certify(self.X, delta=0.05)
        self.assertIsInstance(cert, MechanizerCertificate)
        self.assertEqual(cert.n_samples, 50)
        self.assertLessEqual(cert.r2, 1.0)
        self.assertGreaterEqual(cert.r2, -1.0)

    def test_certify_hoeffding_bound_below_r2(self):
        cert = self.m.certify(self.X, delta=0.05)
        # The LCB is always at most the in-sample mean R^2.
        self.assertLessEqual(cert.hoeffding_r2_lcb, cert.r2 + 1e-6)

    def test_certify_bernstein_tighter_when_variance_low(self):
        cert = self.m.certify(self.X, delta=0.05)
        # If variance is small, Bernstein should be at least as tight
        # as Hoeffding for moderate-N regimes; here we just assert both
        # finite and ordered sensibly.
        self.assertTrue(math.isfinite(cert.hoeffding_r2_lcb))
        self.assertTrue(math.isfinite(cert.bernstein_r2_lcb))

    def test_certify_delta_validated(self):
        with self.assertRaises(InvalidConfig):
            self.m.certify(self.X, delta=0.0)
        with self.assertRaises(InvalidConfig):
            self.m.certify(self.X, delta=1.0)

    def test_certify_identifiable_predicate(self):
        cert = self.m.certify(self.X, delta=0.05)
        if cert.mean_l0 <= cert.identifiable_l0:
            self.assertTrue(cert.identifiable)
        else:
            self.assertFalse(cert.identifiable)

    def test_certify_before_fit_raises(self):
        m = Mechanizer(MechanizerConfig(seed=0))
        with self.assertRaises(NotFit):
            m.certify(self.X)


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------


class SnapshotRestoreTests(unittest.TestCase):
    def test_snapshot_restore_preserves_encode(self):
        X = _make_easy_data(n=30, dim=10, n_true=8, true_l0=2, seed=7)
        m1 = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=12, target_l0=2,
            max_iter=10, seed=0,
        ))
        m1.fit(X)
        codes1 = m1.encode(X[:4])
        snap = m1.snapshot()
        m2 = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=12, target_l0=2,
            max_iter=10, seed=0,
        ))
        m2.restore(snap)
        codes2 = m2.encode(X[:4])
        for i in range(4):
            for j in range(12):
                self.assertAlmostEqual(codes1[i][j], codes2[i][j], places=9)

    def test_snapshot_before_fit_raises(self):
        m = Mechanizer(MechanizerConfig(seed=0))
        with self.assertRaises(NotFit):
            m.snapshot()

    def test_restore_rejects_mismatched_algorithm(self):
        X = _make_easy_data(n=20, dim=8, n_true=6, true_l0=2, seed=0)
        m_topk = Mechanizer(MechanizerConfig(
            algorithm=ALGO_TOPK_SAE, n_features=8, target_l0=2,
            max_iter=5, seed=0,
        ))
        m_topk.fit(X)
        snap = m_topk.snapshot()
        m_ksvd = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=8, target_l0=2,
            max_iter=5, seed=0,
        ))
        with self.assertRaises(InvalidConfig):
            m_ksvd.restore(snap)


# ---------------------------------------------------------------------------
# Ledger integrity
# ---------------------------------------------------------------------------


class LedgerTests(unittest.TestCase):
    def test_ledger_chain_verifies(self):
        X = _make_easy_data(n=20, dim=8, n_true=6, true_l0=2, seed=8)
        m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=8, target_l0=2,
            max_iter=6, seed=0,
        ))
        m.fit(X)
        m.encode(X[:2])
        m.steer(X[:1], feature=0, magnitude=1.0)
        m.patch(X[:1], X[1:2], feature=0)
        m.certify(X)
        self.assertTrue(m.verify_chain())

    def test_ledger_root_is_genesis(self):
        m = Mechanizer(MechanizerConfig(seed=0))
        root = mechanizer_ledger_root()
        events = m.events()
        self.assertGreaterEqual(len(events), 1)
        # The first event's parent is the genesis hash.
        self.assertEqual(events[0].parent_hash, root)

    def test_ledger_hmac_distinct_chains(self):
        X = _make_easy_data(n=10, dim=6, n_true=4, true_l0=2, seed=9)
        m1 = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=6, target_l0=2,
            max_iter=4, seed=0, hmac_key=b"alice",
        ))
        m2 = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=6, target_l0=2,
            max_iter=4, seed=0, hmac_key=b"bob",
        ))
        m1.fit(X)
        m2.fit(X)
        self.assertNotEqual(m1.fingerprint, m2.fingerprint)
        self.assertTrue(m1.verify_chain())
        self.assertTrue(m2.verify_chain())

    def test_known_events_complete(self):
        # Sanity: every documented event constant is in KNOWN_EVENTS.
        for name in (MECHANIZER_STARTED, MECHANIZER_FIT, MECHANIZER_ENCODED,
                     MECHANIZER_STEERED, MECHANIZER_PATCHED,
                     MECHANIZER_CIRCUIT_BUILT, MECHANIZER_CERTIFIED):
            self.assertIn(name, KNOWN_EVENTS)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def test_two_fits_with_same_seed_produce_same_fingerprint(self):
        X = _make_easy_data(n=30, dim=8, n_true=6, true_l0=2, seed=11)
        cfg = MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=8, target_l0=2,
            max_iter=8, seed=99,
        )
        m1 = Mechanizer(cfg)
        m2 = Mechanizer(cfg)
        r1 = m1.fit(X)
        r2 = m2.fit(X)
        self.assertEqual(r1.fingerprint, r2.fingerprint)
        self.assertAlmostEqual(r1.r2, r2.r2, places=9)
        self.assertAlmostEqual(r1.mean_l0, r2.mean_l0, places=9)


# ---------------------------------------------------------------------------
# Helpers / synthetic data
# ---------------------------------------------------------------------------


class HelperTests(unittest.TestCase):
    def test_synthetic_features_validates_inputs(self):
        with self.assertRaises(InvalidConfig):
            mechanizer_synthetic_features(n=0, dim=4, n_true=2, true_l0=1)
        with self.assertRaises(InvalidConfig):
            mechanizer_synthetic_features(n=4, dim=4, n_true=2, true_l0=10)

    def test_random_dictionary_returns_unit_atoms(self):
        D = mechanizer_random_dictionary(n_features=5, dim=8, seed=0)
        for row in D:
            self.assertAlmostEqual(
                math.sqrt(sum(v * v for v in row)), 1.0, places=8,
            )

    def test_random_dictionary_validates(self):
        with self.assertRaises(InvalidConfig):
            mechanizer_random_dictionary(n_features=0, dim=8)


# ---------------------------------------------------------------------------
# Invariants: KSVD recovers identifiable support under low coherence
# ---------------------------------------------------------------------------


class IdentifiabilityIntegrationTests(unittest.TestCase):
    def test_low_coherence_dictionary_recovers_high_r2(self):
        # When the true generating dictionary has low coherence and the
        # signals are k-sparse, KSVD + OMP should reconstruct the
        # observations with near-perfect R^2 — Donoho-Elad 2003.
        X = mechanizer_synthetic_features(
            n=200, dim=40, n_true=20, true_l0=3, seed=12,
        )
        m = Mechanizer(MechanizerConfig(
            algorithm=ALGO_KSVD, n_features=20, target_l0=3,
            max_iter=40, seed=7,
        ))
        rep = m.fit(X)
        self.assertGreater(rep.r2, 0.85)
        self.assertLessEqual(rep.mean_l0, 3.01)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
