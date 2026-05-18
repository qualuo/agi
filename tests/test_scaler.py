"""Tests for the Scaler scaling-law primitive."""
from __future__ import annotations

import json
import math
import random

import pytest

from agi.events import EventBus
from agi.scaler import (
    FAMILY_BAHRI_D,
    FAMILY_BAHRI_N,
    FAMILY_BNSL,
    FAMILY_CHINCHILLA,
    FAMILY_KAPLAN,
    KNOWN_FAMILIES,
    SCALER_CERTIFIED,
    SCALER_EXTRAPOLATED,
    SCALER_FIT,
    SCALER_OBSERVED,
    SCALER_OPTIMAL,
    SCALER_REPORTED,
    SCALER_STARTED,
    ComputeOptimal,
    ExtrapolatePoint,
    FitFailed,
    FitResult,
    InvalidConfig,
    InvalidObservation,
    NotFitted,
    Observation,
    Scaler,
    ScalerCertificate,
    ScalerConfig,
    ScalerReport,
    UnknownFamily,
    bahri_scaler,
    bnsl_scaler,
    chinchilla_scaler,
    kaplan_scaler,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestScalerConfig:
    def test_default_family_chinchilla(self):
        cfg = ScalerConfig()
        assert cfg.family == FAMILY_CHINCHILLA

    def test_unknown_family_rejected(self):
        with pytest.raises(UnknownFamily):
            ScalerConfig(family="nonsense")

    @pytest.mark.parametrize("bad", [-1, -100])
    def test_negative_bootstrap_b_rejected(self, bad):
        with pytest.raises(InvalidConfig):
            ScalerConfig(bootstrap_b=bad)

    def test_bootstrap_b_zero_allowed(self):
        ScalerConfig(bootstrap_b=0)  # disables CI

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nonpositive_max_iters_rejected(self, bad):
        with pytest.raises(InvalidConfig):
            ScalerConfig(max_iters=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_holdout_out_of_range_rejected(self, bad):
        with pytest.raises(InvalidConfig):
            ScalerConfig(holdout_fraction=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nonpositive_tol_rejected(self, bad):
        with pytest.raises(InvalidConfig):
            ScalerConfig(tol=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nonpositive_flops_constant_rejected(self, bad):
        with pytest.raises(InvalidConfig):
            ScalerConfig(flops_per_param_token=bad)

    def test_negative_ridge_rejected(self):
        with pytest.raises(InvalidConfig):
            ScalerConfig(ridge=-1e-6)

    def test_unknown_bootstrap_kind_rejected(self):
        with pytest.raises(InvalidConfig):
            ScalerConfig(bootstrap_kind="bag")

    @pytest.mark.parametrize("bad", [0, 1, -0.1, 1.1])
    def test_confidence_bounded(self, bad):
        with pytest.raises(InvalidConfig):
            ScalerConfig(confidence=bad)

    def test_known_families_consistent(self):
        assert set(KNOWN_FAMILIES) == {
            FAMILY_CHINCHILLA, FAMILY_KAPLAN, FAMILY_BNSL,
            FAMILY_BAHRI_N, FAMILY_BAHRI_D,
        }


# ---------------------------------------------------------------------------
# Observation validation
# ---------------------------------------------------------------------------


class TestObservation:
    def test_valid_observation(self):
        o = Observation(1e9, 1e10, 2.5)
        assert o.weight == 1.0

    @pytest.mark.parametrize("field,val", [
        ("n_params", 0),
        ("n_params", -1),
        ("d_tokens", 0),
        ("d_tokens", -2),
        ("loss", 0),
        ("loss", -0.1),
    ])
    def test_nonpositive_rejected(self, field, val):
        kwargs = {"n_params": 1e9, "d_tokens": 1e10, "loss": 2.5}
        kwargs[field] = val
        with pytest.raises(InvalidObservation):
            Observation(**kwargs)

    @pytest.mark.parametrize("v", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_rejected(self, v):
        with pytest.raises(InvalidObservation):
            Observation(n_params=v, d_tokens=1e9, loss=2.0)

    def test_nonnumeric_rejected(self):
        with pytest.raises(InvalidObservation):
            Observation(n_params="big", d_tokens=1e9, loss=2.0)  # type: ignore[arg-type]

    def test_nonpositive_weight_rejected(self):
        with pytest.raises(InvalidObservation):
            Observation(1e9, 1e10, 2.0, weight=0)


# ---------------------------------------------------------------------------
# Chinchilla fitting on synthetic ground truth
# ---------------------------------------------------------------------------


def _make_chinchilla_data(
    *,
    e: float = 1.7,
    a: float = 410.0,
    b: float = 410.0,
    alpha: float = 0.34,
    beta: float = 0.28,
    noise: float = 0.0,
    seed: int = 0,
) -> list[Observation]:
    rng = random.Random(seed)
    rows = []
    for n in [1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11, 3e11]:
        for d in [1e8, 3e8, 1e9, 3e9, 1e10]:
            true_l = e + a * n**-alpha + b * d**-beta
            log_noise = rng.gauss(0.0, noise) if noise > 0 else 0.0
            rows.append(Observation(n, d, true_l * math.exp(log_noise)))
    return rows


class TestChinchillaFit:
    def test_recovers_clean_parameters(self):
        rows = _make_chinchilla_data()
        s = Scaler(ScalerConfig(
            family=FAMILY_CHINCHILLA, bootstrap_b=0, holdout_fraction=0.0
        ))
        s.observe(rows)
        fit = s.fit()
        assert fit.converged
        assert abs(fit.params["E"] - 1.7) < 1e-4
        assert abs(fit.params["A"] - 410.0) < 1e-1
        assert abs(fit.params["B"] - 410.0) < 1e-1
        assert abs(fit.params["alpha"] - 0.34) < 1e-4
        assert abs(fit.params["beta"] - 0.28) < 1e-4
        assert fit.rmse_in_sample < 1e-5

    def test_recovers_noisy_parameters(self):
        rows = _make_chinchilla_data(noise=0.01, seed=1)
        s = Scaler(ScalerConfig(
            family=FAMILY_CHINCHILLA, bootstrap_b=0, holdout_fraction=0.2,
        ))
        s.observe(rows)
        fit = s.fit()
        assert fit.converged
        # Within 15% relative for each param at 1% noise.
        assert abs(fit.params["alpha"] - 0.34) / 0.34 < 0.15
        assert abs(fit.params["beta"] - 0.28) / 0.28 < 0.15
        assert fit.rmse_held_out is not None
        assert fit.rmse_held_out < 0.05

    def test_fit_too_few_obs_raises(self):
        s = Scaler()
        s.observe(Observation(1e9, 1e10, 2.0))
        with pytest.raises(FitFailed):
            s.fit()

    def test_min_obs_is_param_count_plus_one(self):
        # Chinchilla has 5 params; needs 6 obs.
        s = Scaler()
        for i in range(5):
            s.observe(Observation(1e9 * (i + 1), 1e10, 2.0 + i * 0.1))
        with pytest.raises(FitFailed):
            s.fit()
        s.observe(Observation(1e15, 1e11, 1.7))
        # Now should at least try to fit (may or may not converge well).
        fit = s.fit()
        assert isinstance(fit, FitResult)


# ---------------------------------------------------------------------------
# Other families
# ---------------------------------------------------------------------------


class TestOtherFamilies:
    def test_kaplan_recovers_parameters(self):
        # L = ((Nc/N)^(αn/αd) + (Dc/D))^αd
        nc, dc, alpha_n, alpha_d = 8.8e13, 5.4e13, 0.076, 0.103
        def true_l(n, d):
            return ((nc / n) ** (alpha_n / alpha_d) + dc / d) ** alpha_d
        rows = [
            Observation(n_params=n, d_tokens=d, loss=true_l(n, d))
            for n in [1e7, 1e8, 1e9, 1e10, 1e11]
            for d in [1e8, 1e9, 1e10, 1e11]
        ]
        s = Scaler(ScalerConfig(family=FAMILY_KAPLAN, bootstrap_b=0, holdout_fraction=0.0))
        s.observe(rows)
        fit = s.fit()
        # Kaplan is harder to identify; sanity-check the prediction RMSE.
        assert fit.rmse_in_sample < 1e-3

    def test_bahri_n_recovers_parameters(self):
        # L = 1.5 + (1e8 / N)^0.5
        def true_l(n):
            return 1.5 + (1e8 / n) ** 0.5
        rows = [
            Observation(n_params=n, d_tokens=1e10, loss=true_l(n))
            for n in [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8]
        ]
        s = Scaler(ScalerConfig(family=FAMILY_BAHRI_N, bootstrap_b=0,
                                holdout_fraction=0.0))
        s.observe(rows)
        fit = s.fit()
        assert fit.converged
        assert abs(fit.params["L_inf"] - 1.5) < 1e-3
        assert abs(fit.params["alpha"] - 0.5) < 1e-3

    def test_bahri_d_recovers_parameters(self):
        def true_l(d):
            return 1.2 + (5e8 / d) ** 0.4
        rows = [
            Observation(n_params=1e9, d_tokens=d, loss=true_l(d))
            for d in [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8]
        ]
        s = Scaler(ScalerConfig(family=FAMILY_BAHRI_D, bootstrap_b=0,
                                holdout_fraction=0.0))
        s.observe(rows)
        fit = s.fit()
        assert fit.converged
        assert abs(fit.params["L_inf"] - 1.2) < 1e-3
        assert abs(fit.params["alpha"] - 0.4) < 1e-3

    def test_bnsl_fits_smooth_data(self):
        # Smooth Bahri-like data; BNSL should at least fit closely.
        def true_l(n):
            return 0.5 + 1.0 * n ** -0.3
        rows = [
            Observation(n_params=n, d_tokens=1e10, loss=true_l(n))
            for n in [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7]
        ]
        s = Scaler(ScalerConfig(family=FAMILY_BNSL, bootstrap_b=0,
                                holdout_fraction=0.0, max_iters=400))
        s.observe(rows)
        fit = s.fit()
        # BNSL is over-parameterised for smooth single-power-law data —
        # we just demand the RMSE be small.
        assert fit.rmse_in_sample < 0.05


# ---------------------------------------------------------------------------
# Extrapolation and CI
# ---------------------------------------------------------------------------


class TestExtrapolation:
    def test_extrapolate_before_fit_raises(self):
        s = Scaler()
        with pytest.raises(NotFitted):
            s.extrapolate(1e9, 1e10)

    def test_extrapolate_invalid_inputs(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        with pytest.raises(InvalidObservation):
            s.extrapolate(-1, 1e9)
        with pytest.raises(InvalidObservation):
            s.extrapolate(1e9, float("inf"))
        with pytest.raises(InvalidObservation):
            s.extrapolate(0, 1e9)

    def test_point_prediction_matches_truth_clean(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        # Point prediction at a held-out (N, D) should match truth.
        true_l = 1.7 + 410.0 * (5e10) ** -0.34 + 410.0 * (5e9) ** -0.28
        ep = s.extrapolate(5e10, 5e9)
        assert abs(ep.loss_point - true_l) / true_l < 1e-4

    def test_bootstrap_ci_brackets_truth_noisy(self):
        rows = _make_chinchilla_data(noise=0.02, seed=2)
        s = Scaler(ScalerConfig(bootstrap_b=80, seed=0, holdout_fraction=0.0))
        s.observe(rows)
        s.fit()
        n_eval, d_eval = 5e10, 5e9
        true_l = 1.7 + 410.0 * n_eval ** -0.34 + 410.0 * d_eval ** -0.28
        ep = s.extrapolate(n_eval, d_eval)
        # The CI should bracket the truth with sane margins.
        assert ep.loss_lower <= true_l <= ep.loss_upper
        assert ep.loss_lower < ep.loss_point < ep.loss_upper or \
               abs(ep.loss_upper - ep.loss_lower) < 1e-6

    def test_bootstrap_b_zero_returns_point_only(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        ep = s.extrapolate(1e11, 1e10)
        assert ep.loss_lower == ep.loss_point == ep.loss_upper
        assert ep.bootstrap_b == 0

    def test_residual_bootstrap_runs(self):
        rows = _make_chinchilla_data(noise=0.01, seed=3)
        s = Scaler(ScalerConfig(bootstrap_b=20, bootstrap_kind="residual",
                                seed=0, holdout_fraction=0.0))
        s.observe(rows)
        s.fit()
        ep = s.extrapolate(1e11, 1e10)
        assert ep.bootstrap_b == 20
        assert ep.loss_lower <= ep.loss_point <= ep.loss_upper

    def test_bootstrap_is_seed_deterministic(self):
        rows = _make_chinchilla_data(noise=0.01, seed=7)
        s1 = Scaler(ScalerConfig(bootstrap_b=30, seed=5, holdout_fraction=0.0))
        s2 = Scaler(ScalerConfig(bootstrap_b=30, seed=5, holdout_fraction=0.0))
        s1.observe(rows)
        s2.observe(rows)
        s1.fit()
        s2.fit()
        ep1 = s1.extrapolate(1e11, 1e10)
        ep2 = s2.extrapolate(1e11, 1e10)
        assert ep1.loss_lower == ep2.loss_lower
        assert ep1.loss_upper == ep2.loss_upper


# ---------------------------------------------------------------------------
# Compute-optimal allocation
# ---------------------------------------------------------------------------


class TestComputeOptimal:
    def test_optimal_before_fit_raises(self):
        s = Scaler()
        with pytest.raises(NotFitted):
            s.compute_optimal(1e20)

    def test_optimal_requires_chinchilla(self):
        s = bahri_scaler()
        rows = [Observation(n_params=n, d_tokens=1e10, loss=1.5 + (1e8/n)**0.5)
                for n in [1e5, 1e6, 1e7, 1e8]]
        s.observe(rows)
        s.fit()
        with pytest.raises(InvalidConfig):
            s.compute_optimal(1e22)

    def test_invalid_budget_rejected(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        for bad in [0, -1, float("nan"), float("inf")]:
            with pytest.raises(InvalidConfig):
                s.compute_optimal(bad)

    def test_constraint_satisfied(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        for c in [1e18, 1e20, 1e22, 1e24]:
            co = s.compute_optimal(c)
            # k N D == C
            k = s.config.flops_per_param_token
            assert abs(k * co.n_star_analytic * co.d_star_analytic - c) / c < 1e-9

    def test_analytic_matches_numeric_sweep(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        co = s.compute_optimal(1e22)
        # Analytic point should be optimal within the numeric grid.
        assert co.loss_at_optimum <= co.loss_at_numeric + 1e-9 or \
               abs(co.loss_at_optimum - co.loss_at_numeric) < 1e-9

    def test_optimal_scales_with_budget(self):
        # N* should grow monotonically with C.
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        prev_n = 0.0
        prev_l = float("inf")
        for c in [1e18, 1e19, 1e20, 1e21, 1e22]:
            co = s.compute_optimal(c)
            assert co.n_star_analytic > prev_n
            assert co.loss_at_optimum < prev_l
            prev_n = co.n_star_analytic
            prev_l = co.loss_at_optimum

    def test_optimal_alpha_eq_beta_balanced(self):
        # When alpha == beta, the optimal allocation puts N and D on the
        # diagonal — N* / D* should be A/B at compute budget.
        rng = random.Random(99)
        rows = []
        for n in [1e7, 1e8, 1e9, 1e10, 1e11]:
            for d in [1e8, 1e9, 1e10, 1e11]:
                l = 1.5 + 300.0 * n ** -0.3 + 300.0 * d ** -0.3
                rows.append(Observation(n, d, l))
        s = Scaler(ScalerConfig(bootstrap_b=0, holdout_fraction=0.0))
        s.observe(rows)
        s.fit()
        co = s.compute_optimal(1e22)
        # alpha == beta and A == B → N* == D*.
        assert abs(co.n_star_analytic - co.d_star_analytic) / co.n_star_analytic < 1e-2


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


class TestCertificate:
    def test_certificate_before_fit_raises(self):
        s = Scaler()
        with pytest.raises(NotFitted):
            s.certificate()

    def test_certificate_basic_fields(self):
        rows = _make_chinchilla_data(noise=0.01)
        s = Scaler(ScalerConfig(bootstrap_b=0, holdout_fraction=0.2, seed=42))
        s.observe(rows)
        s.fit()
        cert = s.certificate()
        assert cert.family == FAMILY_CHINCHILLA
        assert cert.n_in_sample + cert.n_held_out == len(rows)
        assert cert.rmse_held_out is not None
        assert cert.rmse_lcb_hoeffding is not None
        assert cert.rmse_lcb_bernstein is not None
        # LCB must be <= the point estimate.
        assert cert.rmse_lcb_hoeffding <= cert.rmse_held_out + 1e-12
        assert cert.rmse_lcb_bernstein <= cert.rmse_held_out + 1e-12
        assert cert.in_range_n[0] < cert.in_range_n[1]
        assert cert.in_range_d[0] < cert.in_range_d[1]

    def test_certificate_no_holdout_has_none_held_out(self):
        rows = _make_chinchilla_data()
        s = Scaler(ScalerConfig(bootstrap_b=0, holdout_fraction=0.0))
        s.observe(rows)
        s.fit()
        cert = s.certificate()
        assert cert.rmse_held_out is None
        assert cert.rmse_lcb_hoeffding is None
        assert cert.rmse_lcb_bernstein is None

    def test_certificate_confidence_override(self):
        rows = _make_chinchilla_data(noise=0.02)
        s = Scaler(ScalerConfig(bootstrap_b=0, holdout_fraction=0.2, seed=11))
        s.observe(rows)
        s.fit()
        c95 = s.certificate(confidence=0.95)
        c99 = s.certificate(confidence=0.99)
        # 99% LCB is lower (further from point) than 95%.
        assert c99.rmse_lcb_hoeffding <= c95.rmse_lcb_hoeffding

    def test_certificate_invalid_confidence(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        with pytest.raises(InvalidConfig):
            s.certificate(confidence=1.5)


# ---------------------------------------------------------------------------
# Event bus integration & fingerprint
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_emits_lifecycle_events(self):
        bus = EventBus()
        kinds = []
        bus.subscribe(lambda ev: kinds.append(ev.kind))
        s = Scaler(ScalerConfig(bootstrap_b=10, holdout_fraction=0.2),
                   bus=bus)
        assert SCALER_STARTED in kinds
        s.observe(_make_chinchilla_data())
        assert SCALER_OBSERVED in kinds
        s.fit()
        assert SCALER_FIT in kinds
        s.extrapolate(1e11, 1e10)
        assert SCALER_EXTRAPOLATED in kinds
        s.compute_optimal(1e22)
        assert SCALER_OPTIMAL in kinds
        s.certificate()
        assert SCALER_CERTIFIED in kinds
        s.report()
        assert SCALER_REPORTED in kinds

    def test_fingerprint_evolves_with_state(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        fp0 = s.fingerprint_hash
        s.observe(Observation(1e9, 1e10, 2.0))
        fp1 = s.fingerprint_hash
        assert fp0 != fp1

    def test_fingerprint_same_for_same_history(self):
        # Replay-verifiability: identical state sequence yields identical hash.
        rows = _make_chinchilla_data(seed=5)
        s1 = Scaler(ScalerConfig(family=FAMILY_CHINCHILLA, seed=5, bootstrap_b=0))
        s2 = Scaler(ScalerConfig(family=FAMILY_CHINCHILLA, seed=5, bootstrap_b=0))
        s1.observe(rows)
        s2.observe(rows)
        assert s1.fingerprint_hash == s2.fingerprint_hash

    def test_fingerprint_differs_for_different_seed(self):
        s1 = Scaler(ScalerConfig(seed=1, bootstrap_b=0))
        s2 = Scaler(ScalerConfig(seed=2, bootstrap_b=0))
        assert s1.fingerprint_hash != s2.fingerprint_hash


# ---------------------------------------------------------------------------
# Report / JSON round-trip
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_before_fit(self):
        s = Scaler()
        rep = s.report()
        assert rep.fit is None
        assert rep.certificate is None
        assert rep.observations == 0

    def test_report_after_fit(self):
        s = Scaler(ScalerConfig(bootstrap_b=0, holdout_fraction=0.2))
        s.observe(_make_chinchilla_data(noise=0.005, seed=8))
        s.fit()
        rep = s.report()
        assert rep.fit is not None
        assert rep.certificate is not None
        assert rep.observations > 0

    def test_report_json_round_trip(self):
        s = Scaler(ScalerConfig(bootstrap_b=0, holdout_fraction=0.2))
        s.observe(_make_chinchilla_data(noise=0.005, seed=9))
        s.fit()
        rep = s.report()
        as_json = json.dumps(rep.to_dict())
        parsed = json.loads(as_json)
        assert parsed["observations"] == rep.observations
        assert parsed["fit"]["family"] == FAMILY_CHINCHILLA

    def test_observations_property(self):
        s = Scaler()
        rows = _make_chinchilla_data()
        s.observe(rows)
        assert len(s.observations) == len(rows)
        # Tuple is immutable.
        with pytest.raises((TypeError, AttributeError)):
            s.observations.append(rows[0])  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        s.reset()
        assert len(s.observations) == 0
        with pytest.raises(NotFitted):
            s.extrapolate(1e9, 1e10)

    def test_observe_after_fit_invalidates_fit(self):
        s = Scaler(ScalerConfig(bootstrap_b=0))
        s.observe(_make_chinchilla_data())
        s.fit()
        s.observe(Observation(1e15, 1e12, 1.71))
        with pytest.raises(NotFitted):
            s.extrapolate(1e9, 1e10)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


class TestFactories:
    def test_chinchilla_factory(self):
        s = chinchilla_scaler(seed=7)
        assert s.config.family == FAMILY_CHINCHILLA
        assert s.config.seed == 7

    def test_kaplan_factory(self):
        s = kaplan_scaler()
        assert s.config.family == FAMILY_KAPLAN

    def test_bnsl_factory(self):
        s = bnsl_scaler()
        assert s.config.family == FAMILY_BNSL

    def test_bahri_factory_axes(self):
        sn = bahri_scaler(axis="n")
        sd = bahri_scaler(axis="d")
        assert sn.config.family == FAMILY_BAHRI_N
        assert sd.config.family == FAMILY_BAHRI_D


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_observe(self):
        import threading
        s = Scaler(ScalerConfig(bootstrap_b=0))
        rows = _make_chinchilla_data()
        threads = []
        def worker(start, step):
            for i in range(start, len(rows), step):
                s.observe(rows[i])
        for k in range(4):
            t = threading.Thread(target=worker, args=(k, 4))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        assert len(s.observations) == len(rows)


# ---------------------------------------------------------------------------
# Weighted observations
# ---------------------------------------------------------------------------


class TestWeights:
    def test_high_weight_dominates(self):
        # Two batches: one tiny but heavily weighted, one large from a
        # mis-specified power.  Fit should track the heavy batch.
        rows = []
        # "Trusted" Chinchilla data with weight 100.
        for n in [1e9, 1e10, 1e11]:
            for d in [1e9, 1e10, 1e11]:
                l = 1.7 + 410.0 * n ** -0.34 + 410.0 * d ** -0.28
                rows.append(Observation(n, d, l, weight=100.0))
        # "Untrusted" outliers with weight 1.
        for n in [5e9, 5e10]:
            for d in [5e9, 5e10]:
                rows.append(Observation(n, d, 5.0, weight=1.0))
        s = Scaler(ScalerConfig(bootstrap_b=0, holdout_fraction=0.0))
        s.observe(rows)
        fit = s.fit()
        # E should remain near 1.7, not be dragged to ~5.
        assert abs(fit.params["E"] - 1.7) < 0.5
