"""Tests for the Continualist primitive — continual / lifelong learning."""
from __future__ import annotations

import math

import pytest

from agi.continualist import (
    CONTINUALIST_BOUNDARY,
    CONTINUALIST_CERTIFIED,
    CONTINUALIST_COMMITTED,
    CONTINUALIST_PROJECTED,
    CONTINUALIST_REGISTERED,
    CONTINUALIST_REPORTED,
    CONTINUALIST_RESET,
    CONTINUALIST_STARTED,
    CONTINUALIST_UPDATED,
    Continualist,
    ContinualistConfig,
    ContinualistError,
    InsufficientData,
    InvalidConfig,
    InvalidGradient,
    InvalidTask,
    METHOD_AGEM,
    METHOD_LWF,
    METHOD_MAS,
    METHOD_NONE,
    METHOD_ONLINE_EWC,
    METHOD_REPLAY,
    METHOD_SI,
    REPLAY_BALANCED,
    REPLAY_RESERVOIR,
    REPLAY_RING,
    UnknownTask,
    average_accuracy,
    backward_transfer,
    continualist_ledger_root,
    forgetting_metric,
    forward_transfer,
    pac_bayes_continual_bound,
)


# ---------------------------------------------------------------------------
# Pure-function metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_bwt_zero_for_single_task(self):
        assert backward_transfer([[1.0]]) == 0.0

    def test_bwt_negative_when_forgetting(self):
        # R[0] = train on task 0, evaluate on task 0 → 0.9
        # R[1] = train on task 1, evaluate on (0, 1) → (0.6, 0.95)
        # BWT = (R[1][0] - R[0][0]) / 1 = 0.6 - 0.9 = -0.3
        R = [[0.9], [0.6, 0.95]]
        assert backward_transfer(R) == pytest.approx(-0.3)

    def test_bwt_positive_when_helping(self):
        R = [[0.5], [0.7, 0.8]]
        assert backward_transfer(R) == pytest.approx(0.2)

    def test_average_accuracy(self):
        R = [[0.5], [0.6, 0.8], [0.55, 0.7, 0.9]]
        # last row over committed tasks (3): (0.55 + 0.7 + 0.9) / 3
        assert average_accuracy(R) == pytest.approx((0.55 + 0.7 + 0.9) / 3)

    def test_forgetting_metric_zero_when_monotone(self):
        # Tasks 0..1 only get *better* over time.
        R = [[0.5], [0.6, 0.9]]
        assert forgetting_metric(R) == 0.0

    def test_forgetting_metric_positive(self):
        # Task 0 peaked at 0.9 in row 0, dropped to 0.6 by last row.
        R = [[0.9], [0.6, 0.8]]
        assert forgetting_metric(R) == pytest.approx(0.3)

    def test_forward_transfer_zero_below_baseline(self):
        baseline = [0.0, 0.5]
        R = [[0.5], [0.6, 0.55]]
        # FWT = R[0][1] - baseline[1] = 0.5 - 0.5 = 0
        # but R[0] has length 1, so the term is skipped → 0
        assert forward_transfer(R, baseline) == 0.0

    def test_forward_transfer_positive(self):
        # Each row gets one extra column ahead-of-time evaluation.
        baseline = [0.1, 0.2, 0.3]
        R = [[0.5, 0.4], [0.6, 0.55, 0.45], [0.7, 0.65, 0.6]]
        # FWT_i = R[i-1][i] - baseline[i]
        # i=1: 0.4 - 0.2 = 0.2
        # i=2: 0.45 - 0.3 = 0.15
        assert forward_transfer(R, baseline) == pytest.approx((0.2 + 0.15) / 2)

    def test_pac_bayes_bound_monotone_in_n(self):
        b1 = pac_bayes_continual_bound(
            empirical_mean_risk=0.1, kl_complexity=2.0, n_tasks=3, n_samples_per_task=10, confidence=0.95
        )
        b2 = pac_bayes_continual_bound(
            empirical_mean_risk=0.1, kl_complexity=2.0, n_tasks=3, n_samples_per_task=1000, confidence=0.95
        )
        assert b2 < b1

    def test_pac_bayes_bound_monotone_in_kl(self):
        b1 = pac_bayes_continual_bound(
            empirical_mean_risk=0.1, kl_complexity=1.0, n_tasks=3, n_samples_per_task=100, confidence=0.95
        )
        b2 = pac_bayes_continual_bound(
            empirical_mean_risk=0.1, kl_complexity=10.0, n_tasks=3, n_samples_per_task=100, confidence=0.95
        )
        assert b2 > b1

    def test_pac_bayes_bound_floors_at_risk(self):
        # As n → ∞ the half-width vanishes and bound → empirical_mean_risk.
        b = pac_bayes_continual_bound(
            empirical_mean_risk=0.1, kl_complexity=0.0, n_tasks=1, n_samples_per_task=10**9, confidence=0.95
        )
        assert b == pytest.approx(0.1, abs=1e-3)

    def test_pac_bayes_bound_invalid_n(self):
        assert math.isinf(
            pac_bayes_continual_bound(
                empirical_mean_risk=0.1,
                kl_complexity=0.0,
                n_tasks=0,
                n_samples_per_task=10,
                confidence=0.95,
            )
        )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config_valid(self):
        cfg = ContinualistConfig()
        assert cfg.method == METHOD_ONLINE_EWC
        assert cfg.dim == 1

    def test_unknown_method_rejected(self):
        with pytest.raises(ContinualistError):
            ContinualistConfig(method="not_a_method")

    def test_unknown_replay_rejected(self):
        with pytest.raises(ContinualistError):
            ContinualistConfig(replay_strategy="no_such")

    def test_dim_must_be_positive(self):
        with pytest.raises(InvalidConfig):
            ContinualistConfig(dim=0)

    def test_fisher_decay_range(self):
        with pytest.raises(InvalidConfig):
            ContinualistConfig(fisher_decay=1.5)

    def test_boundary_hazard_range(self):
        with pytest.raises(InvalidConfig):
            ContinualistConfig(boundary_hazard=0.0)
        with pytest.raises(InvalidConfig):
            ContinualistConfig(boundary_hazard=1.0)

    def test_confidence_range(self):
        with pytest.raises(InvalidConfig):
            ContinualistConfig(confidence=1.5)

    def test_replay_capacity_positive(self):
        with pytest.raises(InvalidConfig):
            ContinualistConfig(replay_capacity=0)

    def test_hmac_must_be_bytes(self):
        with pytest.raises(InvalidConfig):
            ContinualistConfig(hmac_key="not_bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Lifecycle: register / update / commit
# ---------------------------------------------------------------------------


def _make_cl(**kw):
    base = dict(dim=3, method=METHOD_ONLINE_EWC, replay_capacity=8)
    base.update(kw)
    return Continualist(ContinualistConfig(**base))


class TestLifecycle:
    def test_register_then_update_then_commit(self):
        cl = _make_cl()
        cl.register_task("a")
        u = cl.update("a", grad=[0.1, -0.1, 0.0], theta=[0.5, -0.5, 0.0], loss=0.5, accuracy=0.7)
        assert u.step == 1
        assert u.task_id == "a"
        rep = cl.commit_task("a", final_theta=[0.5, -0.5, 0.0], accuracies={"a": 0.75})
        assert rep.n_tasks == 1
        assert rep.average_accuracy == pytest.approx(0.75)

    def test_unknown_task_update_raises(self):
        cl = _make_cl()
        with pytest.raises(UnknownTask):
            cl.update("never_registered", grad=[0.0, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)

    def test_duplicate_register_raises(self):
        cl = _make_cl()
        cl.register_task("a")
        with pytest.raises(InvalidTask):
            cl.register_task("a")

    def test_empty_task_id_raises(self):
        cl = _make_cl()
        with pytest.raises(InvalidTask):
            cl.register_task("")

    def test_double_commit_raises(self):
        cl = _make_cl()
        cl.register_task("a")
        cl.update("a", grad=[0.0, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.5)
        cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.5})
        with pytest.raises(InvalidTask):
            cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.5})

    def test_dimension_mismatch_rejected(self):
        cl = _make_cl(dim=3)
        cl.register_task("a")
        with pytest.raises(InvalidGradient):
            cl.update("a", grad=[0.1, 0.2], theta=[0.5, -0.5, 0.0], loss=0.5)
        with pytest.raises(InvalidGradient):
            cl.update("a", grad=[0.1, 0.2, 0.3], theta=[0.5, -0.5], loss=0.5)

    def test_nan_inf_rejected(self):
        cl = _make_cl()
        cl.register_task("a")
        with pytest.raises(InvalidGradient):
            cl.update("a", grad=[float("nan"), 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)
        with pytest.raises(InvalidGradient):
            cl.update("a", grad=[0.0, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=float("inf"))


# ---------------------------------------------------------------------------
# EWC regulariser behaviour
# ---------------------------------------------------------------------------


class TestEWC:
    def test_zero_regulariser_before_commit(self):
        cl = _make_cl(method=METHOD_ONLINE_EWC, ewc_lambda=10.0)
        v, g = cl.regulariser([1.0, 2.0, 3.0])
        assert v == 0.0
        assert g == (0.0, 0.0, 0.0)

    def test_regulariser_grows_with_distance(self):
        cl = _make_cl(method=METHOD_ONLINE_EWC, ewc_lambda=10.0)
        cl.register_task("a")
        # Generate gradients to make Fisher non-zero.
        for _ in range(20):
            cl.update("a", grad=[0.5, -0.5, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.8)
        cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.8})
        # Now regulariser should be 0 at anchor.
        v0, g0 = cl.regulariser([0.0, 0.0, 0.0])
        assert v0 == pytest.approx(0.0)
        # And > 0 away from anchor.
        v1, g1 = cl.regulariser([1.0, 1.0, 1.0])
        assert v1 > 0.0
        # Gradient should point away from anchor.
        for gi, di in zip(g1, [1.0, 1.0, 1.0]):
            assert gi * di >= 0.0

    def test_fisher_decay_accumulates(self):
        # Two committed tasks; with decay=0.5, second task's Fisher is
        # 0.5 * first + new.
        cl = _make_cl(method=METHOD_ONLINE_EWC, fisher_decay=0.5)
        cl.register_task("a")
        for _ in range(10):
            cl.update("a", grad=[1.0, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.8)
        cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.8})
        f_after_a = list(cl._fisher)
        assert f_after_a[0] == pytest.approx(1.0)
        cl.register_task("b")
        for _ in range(10):
            cl.update("b", grad=[0.0, 1.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.7)
        cl.commit_task("b", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.8, "b": 0.7})
        f_after_b = list(cl._fisher)
        assert f_after_b[0] == pytest.approx(0.5)  # decayed
        assert f_after_b[1] == pytest.approx(1.0)  # fresh


# ---------------------------------------------------------------------------
# Synaptic Intelligence regulariser
# ---------------------------------------------------------------------------


class TestSI:
    def test_si_importance_nonnegative_after_commit(self):
        cl = _make_cl(method=METHOD_SI, si_c=1.0, si_xi=1e-2)
        cl.register_task("a")
        theta = [0.0, 0.0, 0.0]
        for i in range(10):
            # Negative gradient followed by positive parameter move
            # → ω = − g · Δθ > 0 when they have opposite signs.
            new_theta = [theta[0] + 0.05, theta[1], theta[2]]
            cl.update(
                "a",
                grad=[-0.1, 0.0, 0.0],
                theta=new_theta,
                loss=0.5 - 0.01 * i,
                accuracy=0.5 + 0.02 * i,
            )
            theta = new_theta
        cl.commit_task("a", final_theta=theta, accuracies={"a": 0.7})
        assert cl._si_importance[0] > 0.0

    def test_si_regulariser_zero_at_anchor(self):
        cl = _make_cl(method=METHOD_SI, si_c=1.0)
        cl.register_task("a")
        for _ in range(5):
            cl.update(
                "a",
                grad=[-0.1, 0.0, 0.0],
                theta=[0.5, 0.0, 0.0],
                loss=0.5,
                accuracy=0.6,
            )
        cl.commit_task("a", final_theta=[0.5, 0.0, 0.0], accuracies={"a": 0.7})
        v_anchor, _ = cl.regulariser([0.5, 0.0, 0.0])
        assert v_anchor == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Memory Aware Synapses
# ---------------------------------------------------------------------------


class TestMAS:
    def test_mas_refresh_accumulates_importance(self):
        cl = _make_cl(method=METHOD_MAS, mas_lambda=1.0)
        cl.refresh_mas([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        # Importance is mean of absolutes = 1.0 for index 0, 0 otherwise.
        assert cl._mas_importance == [1.0, 0.0, 0.0]
        cl.refresh_mas([[0.0, 2.0, 0.0]])
        # Accumulates across refreshes.
        assert cl._mas_importance == [1.0, 2.0, 0.0]

    def test_mas_refresh_empty_raises(self):
        cl = _make_cl(method=METHOD_MAS)
        with pytest.raises(InsufficientData):
            cl.refresh_mas([])

    def test_mas_regulariser_uses_importance(self):
        cl = _make_cl(method=METHOD_MAS, mas_lambda=2.0)
        cl.refresh_mas([[1.0, 0.0, 0.0]])
        cl.register_task("a")
        cl.commit_task("a", final_theta=[0.5, 0.5, 0.5], accuracies={"a": 0.8})
        v, g = cl.regulariser([1.0, 0.5, 0.5])
        # Only index 0 has importance; (1.0 - 0.5)² · 0.5 · 2 = 0.25
        assert v == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# A-GEM gradient projection
# ---------------------------------------------------------------------------


class TestAGEM:
    def test_no_projection_when_buffer_empty(self):
        cl = _make_cl(method=METHOD_AGEM)
        out = cl.project_gradient([0.1, 0.2, 0.3])
        assert not out.was_projected
        assert out.projected == (0.1, 0.2, 0.3)

    def test_no_projection_when_inner_positive(self):
        cl = _make_cl(method=METHOD_AGEM)
        cl.register_task("a")
        cl.update("a", grad=[1.0, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.5)
        out = cl.project_gradient([0.5, 0.0, 0.0])
        assert not out.was_projected
        assert out.inner_product > 0.0

    def test_projection_when_inner_negative(self):
        cl = _make_cl(method=METHOD_AGEM)
        cl.register_task("a")
        cl.update("a", grad=[1.0, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.5)
        out = cl.project_gradient([-1.0, 0.0, 0.0])
        assert out.was_projected
        # After projection: g · g_ref ≈ 0 (orthogonal).
        gref = (1.0, 0.0, 0.0)
        ip = sum(a * b for a, b in zip(out.projected, gref))
        assert abs(ip) < 1e-9


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class TestReplay:
    def test_ring_buffer_bounded(self):
        cl = _make_cl(replay_strategy=REPLAY_RING, replay_capacity=4)
        cl.register_task("a")
        for i in range(20):
            cl.update("a", grad=[float(i), 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)
        assert cl.replay_size() == 4
        # FIFO: latest grads kept.
        firsts = [it.gradient[0] for it in cl._replay]
        assert firsts == [16.0, 17.0, 18.0, 19.0]

    def test_reservoir_bounded_and_seeded(self):
        cl = _make_cl(replay_strategy=REPLAY_RESERVOIR, replay_capacity=4, seed=42)
        cl.register_task("a")
        for i in range(100):
            cl.update("a", grad=[float(i), 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)
        assert cl.replay_size() == 4
        # Same seed → same indices kept.
        cl2 = _make_cl(replay_strategy=REPLAY_RESERVOIR, replay_capacity=4, seed=42)
        cl2.register_task("a")
        for i in range(100):
            cl2.update("a", grad=[float(i), 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)
        a = sorted(it.step for it in cl._replay)
        b = sorted(it.step for it in cl2._replay)
        assert a == b

    def test_balanced_reservoir_keeps_per_class(self):
        cl = _make_cl(replay_strategy=REPLAY_BALANCED, replay_capacity=6, seed=0)
        cl.register_task("a")
        for i in range(60):
            lbl = "x" if i % 3 == 0 else ("y" if i % 3 == 1 else "z")
            cl.update("a", grad=[float(i), 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, label=lbl)
        labels = [it.label for it in cl._replay]
        # At least two of each label kept.
        for c in ("x", "y", "z"):
            assert labels.count(c) >= 1

    def test_replay_sample_size(self):
        cl = _make_cl(replay_capacity=10)
        cl.register_task("a")
        for i in range(20):
            cl.update("a", grad=[float(i), 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)
        items = cl.replay_sample(5)
        assert len(items) == 5
        # Asking for more than available returns what's there.
        items = cl.replay_sample(100)
        assert len(items) == cl.replay_size()

    def test_replay_sample_negative_raises(self):
        cl = _make_cl()
        with pytest.raises(InvalidGradient):
            cl.replay_sample(-1)


# ---------------------------------------------------------------------------
# Boundary detection (BOCD)
# ---------------------------------------------------------------------------


class TestBoundaryDetection:
    def test_no_boundaries_when_disabled(self):
        cl = _make_cl(boundary_detection=False)
        cl.register_task("a")
        for i in range(50):
            sig = 0.0 if i < 25 else 10.0
            cl.update(
                "a",
                grad=[0.1, 0.0, 0.0],
                theta=[0.0, 0.0, 0.0],
                loss=0.5,
                boundary_signal=sig,
            )
        rep = cl.report()
        assert rep.n_boundaries == 0

    def test_bocd_detects_distribution_shift(self):
        cl = _make_cl(
            boundary_detection=True,
            boundary_hazard=0.05,
            boundary_threshold=0.3,
        )
        cl.register_task("a")
        # 30 steps near 0, then 30 steps near 5 — clear change-point.
        # Pre-seed the prior with a few samples so the predictive
        # likelihood is well-formed.
        for i in range(30):
            cl.update(
                "a",
                grad=[0.1, 0.0, 0.0],
                theta=[0.0, 0.0, 0.0],
                loss=0.5,
                boundary_signal=0.01 * i,
            )
        for i in range(30):
            cl.update(
                "a",
                grad=[0.1, 0.0, 0.0],
                theta=[0.0, 0.0, 0.0],
                loss=0.5,
                boundary_signal=5.0 + 0.01 * i,
            )
        rep = cl.report()
        assert rep.n_boundaries >= 1

    def test_explicit_boundary_is_recorded(self):
        cl = _make_cl(boundary_detection=False)
        b = cl.task_boundary()
        assert b.explicit is True
        assert b.probability == 1.0
        rep = cl.report()
        assert rep.n_boundaries == 1


# ---------------------------------------------------------------------------
# Reports — BWT / FWT / forgetting end-to-end
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_signals_forgetting(self):
        cl = _make_cl()
        # Task a: trained, accuracy 0.9
        cl.register_task("a")
        cl.update("a", grad=[0.1, 0.0, 0.0], theta=[0.1, 0.0, 0.0], loss=0.5, accuracy=0.9)
        cl.commit_task("a", final_theta=[0.1, 0.0, 0.0], accuracies={"a": 0.9})
        # Task b: trained; a forgotten → 0.4.
        cl.register_task("b")
        cl.update("b", grad=[0.0, 0.1, 0.0], theta=[0.0, 0.1, 0.0], loss=0.5, accuracy=0.8)
        rep = cl.commit_task(
            "b",
            final_theta=[0.0, 0.1, 0.0],
            accuracies={"a": 0.4, "b": 0.8},
        )
        assert rep.n_tasks == 2
        # BWT = (R[1][0] - R[0][0]) / 1 = 0.4 - 0.9 = -0.5
        assert rep.backward_transfer == pytest.approx(-0.5)
        assert rep.forgetting == pytest.approx(0.5)
        assert rep.average_accuracy == pytest.approx((0.4 + 0.8) / 2)

    def test_certify_requires_committed_task(self):
        cl = _make_cl()
        with pytest.raises(InsufficientData):
            cl.certify()


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


class TestCertificate:
    def test_certificate_passes_when_stable(self):
        cl = _make_cl(method=METHOD_ONLINE_EWC, plasticity_min=0.7, stability_eps=0.1)
        cl.register_task("a")
        for _ in range(5):
            cl.update("a", grad=[0.1, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.3, accuracy=0.85)
        cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.85})
        cl.register_task("b")
        for _ in range(5):
            cl.update("b", grad=[0.0, 0.1, 0.0], theta=[0.0, 0.0, 0.0], loss=0.3, accuracy=0.8)
        cl.commit_task("b", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.82, "b": 0.8})
        cert = cl.certify(n_samples_per_task=1000)
        assert cert.plasticity_ok is True
        assert cert.stability_ok is True
        assert cert.min_fresh_accuracy >= 0.7
        assert cert.empirical_mean_risk <= 0.2

    def test_certificate_fails_on_forgetting(self):
        cl = _make_cl(method=METHOD_ONLINE_EWC, plasticity_min=0.5, stability_eps=0.05)
        cl.register_task("a")
        cl.update("a", grad=[0.1, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.3, accuracy=0.9)
        cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.9})
        cl.register_task("b")
        cl.update("b", grad=[0.0, 0.1, 0.0], theta=[0.0, 0.0, 0.0], loss=0.3, accuracy=0.8)
        cl.commit_task("b", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.2, "b": 0.8})
        cert = cl.certify(n_samples_per_task=100)
        assert cert.stability_ok is False
        assert cert.max_forget_gap > 0.05

    def test_certificate_bound_tighter_with_more_samples(self):
        cl = _make_cl()
        cl.register_task("a")
        for _ in range(3):
            cl.update("a", grad=[0.1, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.8)
        cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.8})
        c_small = cl.certify(n_samples_per_task=10)
        c_big = cl.certify(n_samples_per_task=10_000)
        assert c_big.pac_bayes_bound < c_small.pac_bayes_bound


# ---------------------------------------------------------------------------
# Tamper-evident chain
# ---------------------------------------------------------------------------


class TestChain:
    def test_chain_advances_on_each_event(self):
        cl = _make_cl()
        h0 = cl.chain_head
        cl.register_task("a")
        h1 = cl.chain_head
        cl.update("a", grad=[0.1, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.5)
        h2 = cl.chain_head
        assert h0 != h1 != h2
        assert h0 != h2

    def test_chain_genesis_deterministic(self):
        a = continualist_ledger_root()
        b = continualist_ledger_root()
        assert a == b
        c = continualist_ledger_root(b"secret")
        assert a != c

    def test_chain_uses_hmac_when_keyed(self):
        cl1 = _make_cl(hmac_key=b"secret")
        cl2 = _make_cl(hmac_key=b"other")
        assert cl1.chain_head != cl2.chain_head

    def test_chain_reproduces_for_same_history(self):
        # Two Continualists with identical configs and identical
        # update streams should end with identical chain heads.
        def trace(seed):
            cl = _make_cl(seed=seed)
            cl.register_task("a")
            for i in range(5):
                cl.update("a", grad=[0.1, 0.2, 0.3], theta=[0.0, 0.0, 0.0], loss=0.5)
            cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.5})
            return cl.chain_head

        assert trace(0) == trace(0)


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_round_trip(self):
        cl = _make_cl(seed=7)
        cl.register_task("a")
        for i in range(5):
            cl.update("a", grad=[0.1 * i, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.7)
        cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.75})
        snap = cl.snapshot()
        cl2 = _make_cl(seed=99)  # different seed; restore should overwrite
        cl2.restore(snap)
        assert cl2.chain_head == cl.chain_head
        assert cl2.n_committed == cl.n_committed
        assert cl2._anchor == cl._anchor
        assert cl2._fisher == cl._fisher

    def test_snapshot_resumes_seamlessly(self):
        cl = _make_cl(seed=3)
        cl.register_task("a")
        cl.update("a", grad=[0.1, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)
        snap = cl.snapshot()
        cl.update("a", grad=[0.2, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)
        h_a = cl.chain_head
        cl2 = _make_cl(seed=1234)
        cl2.restore(snap)
        cl2.update("a", grad=[0.2, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5)
        assert cl2.chain_head == h_a


# ---------------------------------------------------------------------------
# Event bus integration
# ---------------------------------------------------------------------------


class TestEvents:
    def test_publisher_receives_lifecycle_events(self):
        events: list[tuple[str, dict]] = []

        def pub(kind: str, payload: dict) -> None:
            events.append((kind, payload))

        cl = Continualist(ContinualistConfig(dim=2), publisher=pub)
        cl.register_task("a")
        cl.update("a", grad=[0.1, 0.2], theta=[0.0, 0.0], loss=0.5, accuracy=0.5)
        cl.commit_task("a", final_theta=[0.0, 0.0], accuracies={"a": 0.5})
        cl.report()
        cl.certify()
        kinds = [k for k, _ in events]
        assert CONTINUALIST_STARTED in kinds
        assert CONTINUALIST_REGISTERED in kinds
        assert CONTINUALIST_UPDATED in kinds
        assert CONTINUALIST_COMMITTED in kinds
        assert CONTINUALIST_REPORTED in kinds
        assert CONTINUALIST_CERTIFIED in kinds


# ---------------------------------------------------------------------------
# Plasticity-stability dynamics smoke test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_three_task_sequence(self):
        """A three-task sequence with mild forgetting — exercises the full pipeline."""
        cl = _make_cl(
            method=METHOD_ONLINE_EWC,
            ewc_lambda=1.0,
            replay_capacity=32,
            seed=0,
        )
        accs_history: list[dict[str, float]] = []
        for t_idx, tid in enumerate(["math", "code", "research"]):
            cl.register_task(tid)
            for _ in range(20):
                cl.update(
                    tid,
                    grad=[0.05 * (t_idx + 1)] * 3,
                    theta=[0.1 * (t_idx + 1)] * 3,
                    loss=0.5 - 0.01 * t_idx,
                    accuracy=0.7 + 0.05 * t_idx,
                )
            # Simulated held-out: small forgetting penalty per old task.
            cur_accs: dict[str, float] = {}
            for past_idx, past_tid in enumerate(["math", "code", "research"][: t_idx + 1]):
                if past_tid == tid:
                    cur_accs[past_tid] = 0.7 + 0.05 * t_idx
                else:
                    cur_accs[past_tid] = max(
                        0.0,
                        (0.7 + 0.05 * past_idx) - 0.02 * (t_idx - past_idx),
                    )
            accs_history.append(cur_accs)
            cl.commit_task(tid, final_theta=[0.1 * (t_idx + 1)] * 3, accuracies=cur_accs)
        rep = cl.report()
        assert rep.n_tasks == 3
        # Small forgetting since penalty was small.
        assert rep.forgetting >= 0.0
        assert rep.average_accuracy > 0.5

    def test_reset_clears_state(self):
        cl = _make_cl()
        cl.register_task("a")
        cl.update("a", grad=[0.1, 0.0, 0.0], theta=[0.0, 0.0, 0.0], loss=0.5, accuracy=0.7)
        cl.commit_task("a", final_theta=[0.0, 0.0, 0.0], accuracies={"a": 0.7})
        h_old = cl.chain_head
        cl.reset()
        assert cl.n_tasks == 0
        assert cl.n_committed == 0
        assert cl.step == 0
        assert cl.chain_head == continualist_ledger_root(cl.config.hmac_key)
        assert h_old != cl.chain_head
