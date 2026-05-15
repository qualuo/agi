"""Tests for the Filterer runtime primitive — Bayesian state-space filtering."""
from __future__ import annotations

import math
import random
import pytest

from agi.filterer import (
    APF,
    BOOTSTRAP,
    EKF,
    FFBSI,
    FILTERER_CLEARED,
    FILTERER_PREDICTED,
    FILTERER_REPORT,
    FILTERER_RESAMPLED,
    FILTERER_STARTED,
    FILTERER_UPDATED,
    FilterDegenerate,
    FilterReport,
    Filterer,
    FiltererConfig,
    FiltererError,
    GaussianBelief,
    GenericConfigError,
    INFORMATION_FILTER,
    InvalidDimension,
    InvalidMatrix,
    InvalidObservation,
    InvalidParticles,
    KALMAN_FAMILY,
    KF,
    KNOWN_EVENTS,
    KNOWN_FILTERS,
    KNOWN_RESAMPLERS,
    KNOWN_SMOOTHERS,
    MULTINOMIAL,
    NonPositiveDefinite,
    PARTICLE_FAMILY,
    ParticleBelief,
    PredictionResult,
    RESIDUAL,
    RTS,
    ResampleResult,
    SIR,
    SQRT_KF,
    STRATIFIED,
    SYSTEMATIC,
    SmoothResult,
    StateSpaceModel,
    UKF,
    UnknownFilter,
    UnknownResampler,
    UnknownSmoother,
    UpdateResult,
    crisan_doucet_mse_bound,
    effective_sample_size,
    ekf_predict,
    ekf_update,
    innovation_whiteness_stat,
    kalman_predict,
    kalman_update,
    massart_dkw_band,
    nis_chi2_threshold,
    rts_smooth,
    ukf_predict,
    ukf_update,
)


# =====================================================================
# Validation
# =====================================================================


class TestValidation:
    def test_known_filters(self):
        assert KF in KNOWN_FILTERS
        assert EKF in KNOWN_FILTERS
        assert UKF in KNOWN_FILTERS
        assert SIR in KNOWN_FILTERS
        assert APF in KNOWN_FILTERS
        assert BOOTSTRAP in KNOWN_FILTERS

    def test_known_resamplers(self):
        for r in (MULTINOMIAL, SYSTEMATIC, STRATIFIED, RESIDUAL):
            assert r in KNOWN_RESAMPLERS

    def test_known_smoothers(self):
        for s in (RTS, FFBSI):
            assert s in KNOWN_SMOOTHERS

    def test_known_events(self):
        for e in (FILTERER_STARTED, FILTERER_PREDICTED, FILTERER_UPDATED,
                  FILTERER_RESAMPLED, FILTERER_REPORT, FILTERER_CLEARED):
            assert e in KNOWN_EVENTS

    def test_unknown_filter_rejected(self):
        cfg = FiltererConfig(filter_name="bogus")
        with pytest.raises(UnknownFilter):
            Filterer(cfg, mean=[0.0], cov=[[1.0]])

    def test_invalid_init_no_mean(self):
        with pytest.raises(GenericConfigError):
            cfg = FiltererConfig(filter_name=KF)
            Filterer(cfg)

    def test_invalid_kalman_matrix_shapes(self):
        # H has wrong column count.
        with pytest.raises(InvalidDimension):
            Filterer.kalman(
                F=[[1.0, 1.0], [0.0, 1.0]],
                H=[[1.0]],
                Q=[[1.0e-4, 0.0], [0.0, 1.0e-4]],
                R=[[0.1]],
                x0=[0.0, 0.0],
                P0=[[1.0, 0.0], [0.0, 1.0]],
            )

    def test_invalid_non_spd_Q(self):
        with pytest.raises((InvalidMatrix, NonPositiveDefinite)):
            Filterer.kalman(
                F=[[1.0]],
                H=[[1.0]],
                Q=[[-1.0]],
                R=[[0.1]],
                x0=[0.0],
                P0=[[1.0]],
            )

    def test_nan_observation_rejected(self):
        f = _make_simple_kf()
        with pytest.raises(InvalidObservation):
            f.update([float("nan")])


# =====================================================================
# Helpers
# =====================================================================


def _make_simple_kf(seed=0):
    return Filterer.kalman(
        F=[[1.0, 1.0], [0.0, 1.0]],
        H=[[1.0, 0.0]],
        Q=[[1.0e-4, 0.0], [0.0, 1.0e-3]],
        R=[[0.1]],
        x0=[0.0, 0.0],
        P0=[[1.0, 0.0], [0.0, 1.0]],
        seed=seed,
    )


# =====================================================================
# Bounds + diagnostics
# =====================================================================


class TestBoundsAndDiagnostics:
    def test_chi2_threshold_m_1(self):
        # χ²(1) 95% quantile ≈ 3.841459.
        t = nis_chi2_threshold(1, 0.05)
        assert 3.8 < t < 3.9

    def test_chi2_threshold_m_2(self):
        # χ²(2) 95% quantile ≈ 5.99146.
        t = nis_chi2_threshold(2, 0.05)
        assert 5.7 < t < 6.2

    def test_chi2_threshold_invalid(self):
        with pytest.raises(InvalidDimension):
            nis_chi2_threshold(0)
        with pytest.raises(FiltererError):
            nis_chi2_threshold(1, 0.0)
        with pytest.raises(FiltererError):
            nis_chi2_threshold(1, 1.0)

    def test_crisan_doucet_bound_decreasing_in_N(self):
        a = crisan_doucet_mse_bound(100)
        b = crisan_doucet_mse_bound(1000)
        assert b < a
        assert math.isclose(b * 10, a)

    def test_dkw_band(self):
        # Massart 1990 verified value: sqrt(log(40)/200) at δ=0.05, N=100.
        b = massart_dkw_band(100, 0.05)
        expected = math.sqrt(math.log(2.0 / 0.05) / 200.0)
        assert math.isclose(b, expected, rel_tol=1e-12)

    def test_ess_uniform_is_N(self):
        weights = [1.0 / 4] * 4
        assert math.isclose(effective_sample_size(weights), 4.0)

    def test_ess_degenerate_is_1(self):
        weights = [1.0] + [0.0] * 9
        assert math.isclose(effective_sample_size(weights), 1.0)

    def test_ess_negative_rejected(self):
        with pytest.raises(InvalidParticles):
            effective_sample_size([0.5, -0.5])

    def test_ess_zero_sum_rejected(self):
        with pytest.raises(FilterDegenerate):
            effective_sample_size([0.0, 0.0])

    def test_innovation_whiteness_constant(self):
        # Whiteness statistic on a constant sequence is small (≈0).
        # Variance of zero gives statistic = 0 by convention.
        s, df = innovation_whiteness_stat([1.0] * 50, max_lag=2)
        assert df == 2
        assert s == 0.0

    def test_innovation_whiteness_iid(self):
        rng = random.Random(42)
        innov = [rng.gauss(0, 1) for _ in range(200)]
        s, df = innovation_whiteness_stat(innov, max_lag=3)
        # Under H0 stat ∼ χ²(3); 99th percentile ≈ 11.35.  We just check
        # it's not pathologically large.
        assert 0 <= s < 30.0
        assert df == 3


# =====================================================================
# Linear-Gaussian Kalman filter
# =====================================================================


class TestKalmanFilter:
    def test_one_step_predict_update(self):
        f = _make_simple_kf()
        f.predict()
        r = f.update([1.0])
        assert isinstance(r, UpdateResult)
        assert isinstance(r.belief, GaussianBelief)
        assert r.belief.dim == 2
        assert r.n_step == 1
        assert r.nis >= 0.0
        # Posterior position should be pulled toward 1.0.
        mu = r.belief.mean
        assert 0.0 < mu[0] < 1.0

    def test_tracks_constant_velocity(self):
        # True trajectory: position = t * 0.5 + noise.
        f = _make_simple_kf(seed=1)
        rng = random.Random(7)
        positions = []
        for t in range(100):
            true_pos = 0.5 * t
            y = true_pos + rng.gauss(0, math.sqrt(0.1))
            f.predict()
            r = f.update([y])
            positions.append(r.belief.mean[0])
        # After warmup, the filtered position should track the truth.
        err = abs(positions[-1] - 0.5 * 99)
        assert err < 1.5  # well within filter covariance

    def test_log_marginal_strictly_finite(self):
        f = _make_simple_kf()
        rng = random.Random(0)
        for _ in range(20):
            f.predict()
            f.update([rng.gauss(0, 1)])
        assert math.isfinite(f.log_marginal)

    def test_nis_near_chi2_mean(self):
        # Under correct specification, E[NIS] = m = 1.
        rng = random.Random(123)
        f = Filterer.kalman(
            F=[[1.0]],
            H=[[1.0]],
            Q=[[1e-3]],
            R=[[0.1]],
            x0=[0.0],
            P0=[[0.1]],
            seed=0,
        )
        x = 0.0
        for _ in range(400):
            x += rng.gauss(0, math.sqrt(1e-3))
            y = x + rng.gauss(0, math.sqrt(0.1))
            f.predict()
            f.update([y])
        report = f.report()
        # Sample mean of χ²(1) is ≈ 1 with std ≈ √(2/T) ≈ 0.07 at T=400.
        assert 0.5 < report.mean_nis < 2.0

    def test_report_structure(self):
        f = _make_simple_kf()
        for _ in range(10):
            f.predict()
            f.update([0.5])
        r = f.report()
        assert isinstance(r, FilterReport)
        assert r.n_steps == 10
        assert r.filter_name == KF
        assert r.n_state == 2
        assert r.n_obs == 1
        assert r.nis_chi2_threshold > 0.0
        assert math.isfinite(r.log_marginal)

    def test_fingerprint_chains(self):
        f = _make_simple_kf()
        fp_0 = f.fingerprint
        f.predict()
        fp_1 = f.fingerprint
        f.update([0.0])
        fp_2 = f.fingerprint
        # Must be a chain — every step changes the fingerprint.
        assert fp_0 != fp_1 != fp_2

    def test_replay_determinism(self):
        # Same seed + same input → same fingerprint.
        rng = random.Random(99)
        ys = [rng.gauss(0, 1) for _ in range(15)]
        f1 = _make_simple_kf(seed=0)
        f2 = _make_simple_kf(seed=0)
        for y in ys:
            f1.predict(); f1.update([y])
            f2.predict(); f2.update([y])
        assert f1.fingerprint == f2.fingerprint
        assert math.isclose(f1.log_marginal, f2.log_marginal)


# =====================================================================
# Extended Kalman filter
# =====================================================================


class TestEKF:
    def test_ekf_pendulum_step(self):
        # Non-linear pendulum-ish: x_t = sin(x_{t-1}); observe x.
        def f_dyn(x, _u):
            return [math.sin(x[0])]

        def f_jac(x, _u):
            return [[math.cos(x[0])]]

        def h_obs(x):
            return [x[0]]

        def h_jac(x):
            return [[1.0]]

        filt = Filterer.ekf(
            dynamics_fn=f_dyn,
            obs_fn=h_obs,
            dynamics_jac=f_jac,
            obs_jac=h_jac,
            Q=[[1e-3]],
            R=[[0.05]],
            x0=[0.1],
            P0=[[0.5]],
        )
        rng = random.Random(0)
        x = 0.1
        for _ in range(50):
            x = math.sin(x) + rng.gauss(0, math.sqrt(1e-3))
            y = x + rng.gauss(0, math.sqrt(0.05))
            filt.predict()
            filt.update([y])
        # EKF mean should track the (decaying) true state.
        assert abs(filt.mean[0] - x) < 0.5

    def test_ekf_linear_matches_kf(self):
        # When f and h are linear, EKF reduces exactly to KF.
        F = [[1.0, 1.0], [0.0, 1.0]]
        H = [[1.0, 0.0]]
        Q = [[1e-4, 0.0], [0.0, 1e-4]]
        R = [[0.1]]

        def f_dyn(x, _u):
            return [F[0][0] * x[0] + F[0][1] * x[1],
                    F[1][0] * x[0] + F[1][1] * x[1]]

        def f_jac(x, _u):
            return F

        def h_obs(x):
            return [H[0][0] * x[0] + H[0][1] * x[1]]

        def h_jac(x):
            return H

        ekf = Filterer.ekf(
            dynamics_fn=f_dyn, obs_fn=h_obs,
            dynamics_jac=f_jac, obs_jac=h_jac,
            Q=Q, R=R, x0=[0.0, 0.0], P0=[[1.0, 0.0], [0.0, 1.0]],
        )
        kf = Filterer.kalman(
            F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=[[1.0, 0.0], [0.0, 1.0]],
        )
        for y in (0.5, -0.3, 1.2, 0.8):
            ekf.predict(); ekf.update([y])
            kf.predict(); kf.update([y])
        for a, b in zip(ekf.mean, kf.mean):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


# =====================================================================
# Unscented Kalman filter
# =====================================================================


class TestUKF:
    def test_ukf_linear_matches_kf(self):
        # On a linear-Gaussian system UKF should match KF closely (third-
        # order match for any symmetric distribution; quadratic in the
        # linear case is exact).
        F = [[1.0, 1.0], [0.0, 1.0]]
        H = [[1.0, 0.0]]
        Q = [[1e-4, 0.0], [0.0, 1e-4]]
        R = [[0.1]]

        def f_dyn(x, _u):
            return [F[0][0] * x[0] + F[0][1] * x[1],
                    F[1][0] * x[0] + F[1][1] * x[1]]

        def h_obs(x):
            return [H[0][0] * x[0] + H[0][1] * x[1]]

        ukf = Filterer.ukf(
            dynamics_fn=f_dyn, obs_fn=h_obs,
            Q=Q, R=R, x0=[0.0, 0.0], P0=[[1.0, 0.0], [0.0, 1.0]],
        )
        kf = Filterer.kalman(
            F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=[[1.0, 0.0], [0.0, 1.0]],
        )
        for y in (0.5, -0.3, 1.2, 0.8):
            ukf.predict(); ukf.update([y])
            kf.predict(); kf.update([y])
        for a, b in zip(ukf.mean, kf.mean):
            assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)

    def test_ukf_nonlinear_tracking(self):
        def f_dyn(x, _u):
            return [0.9 * x[0] + 0.1 * math.sin(x[0])]

        def h_obs(x):
            return [x[0] * x[0]]

        filt = Filterer.ukf(
            dynamics_fn=f_dyn, obs_fn=h_obs,
            Q=[[1e-3]], R=[[0.05]],
            x0=[0.5], P0=[[0.5]],
        )
        rng = random.Random(0)
        x = 0.5
        for _ in range(40):
            x = 0.9 * x + 0.1 * math.sin(x) + rng.gauss(0, math.sqrt(1e-3))
            y = x * x + rng.gauss(0, math.sqrt(0.05))
            filt.predict()
            filt.update([y])
        # We only require the filter to remain finite and the report to be valid.
        r = filt.report()
        assert math.isfinite(r.log_marginal)
        assert r.n_steps == 40


# =====================================================================
# Particle filter
# =====================================================================


class TestParticleFilter:
    def test_pf_init_requires_particles(self):
        with pytest.raises(GenericConfigError):
            cfg = FiltererConfig(filter_name=SIR, n_particles=10)
            Filterer(cfg, propose_fn=lambda x, u, r: x,
                     likelihood_fn=lambda y, x: 0.0)

    def test_pf_tracks_random_walk(self):
        # x_t = x_{t-1} + N(0, σ²);  y_t = x_t + N(0, σ_obs²).
        sigma_proc = 0.1
        sigma_obs = 0.2

        def propose(p, _u, rng):
            return [p[0] + rng.gauss(0, sigma_proc)]

        def loglik(y, p):
            d = y[0] - p[0]
            return -0.5 * (d / sigma_obs) ** 2 - math.log(sigma_obs) - 0.5 * math.log(2 * math.pi)

        rng = random.Random(0)
        N = 200
        init = [[rng.gauss(0, 1.0)] for _ in range(N)]
        filt = Filterer.particle(
            propose_fn=propose,
            likelihood_fn=loglik,
            initial_particles=init,
            filter_name=SIR,
            resampler=SYSTEMATIC,
            seed=42,
        )
        x = 0.0
        rng2 = random.Random(7)
        for _ in range(50):
            x += rng2.gauss(0, sigma_proc)
            y = x + rng2.gauss(0, sigma_obs)
            filt.predict()
            r = filt.update([y])
            assert r.ess is not None
            assert r.ess > 0
        assert abs(filt.mean[0] - x) < 1.0

    def test_pf_resample_makes_ess_N(self):
        sigma_proc = 0.1
        sigma_obs = 0.2

        def propose(p, _u, rng):
            return [p[0] + rng.gauss(0, sigma_proc)]

        def loglik(y, p):
            d = y[0] - p[0]
            return -0.5 * (d / sigma_obs) ** 2

        N = 50
        init = [[0.0] for _ in range(N)]
        filt = Filterer.particle(
            propose_fn=propose,
            likelihood_fn=loglik,
            initial_particles=init,
            seed=0,
            ess_threshold=0.9,  # force resampling
        )
        filt.predict()
        filt.update([5.0])  # very informative obs → big ESS drop
        # ESS should be ≈ N after the auto-resample.
        ess = effective_sample_size(list(filt.belief.weights))
        assert ess > N * 0.5

    def test_pf_manual_resample(self):
        def propose(p, _u, rng):
            return [p[0]]

        def loglik(y, p):
            return -((y[0] - p[0]) ** 2)

        init = [[float(i)] for i in range(20)]
        filt = Filterer.particle(
            propose_fn=propose,
            likelihood_fn=loglik,
            initial_particles=init,
            seed=0,
            ess_threshold=0.0,  # don't auto-resample
        )
        filt.predict()
        filt.update([0.0])
        r = filt.resample(scheme=STRATIFIED)
        assert isinstance(r, ResampleResult)
        assert r.scheme == STRATIFIED

    def test_pf_resample_invalid_for_kf(self):
        f = _make_simple_kf()
        with pytest.raises(FiltererError):
            f.resample()


# =====================================================================
# Resampling schemes
# =====================================================================


class TestResamplers:
    def test_each_scheme_returns_valid_indices(self):
        for scheme in (MULTINOMIAL, SYSTEMATIC, STRATIFIED, RESIDUAL):
            sigma_proc = 0.1

            def propose(p, _u, rng):
                return [p[0] + rng.gauss(0, sigma_proc)]

            def loglik(y, p):
                return -((y[0] - p[0]) ** 2)

            init = [[float(i)] for i in range(30)]
            filt = Filterer.particle(
                propose_fn=propose,
                likelihood_fn=loglik,
                initial_particles=init,
                resampler=scheme,
                seed=0,
                ess_threshold=1.0,
            )
            filt.predict()
            filt.update([0.5])
            r = filt.resample(scheme=scheme)
            assert r.scheme == scheme
            assert math.isclose(r.ess_after, 30.0)

    def test_unknown_resampler_rejected(self):
        with pytest.raises(UnknownResampler):
            cfg = FiltererConfig(filter_name=SIR, n_particles=5,
                                 resampler="bogus")
            Filterer(cfg, particles=[[0.0]] * 5,
                     propose_fn=lambda p, u, r: p,
                     likelihood_fn=lambda y, x: 0.0)


# =====================================================================
# RTS smoother
# =====================================================================


class TestRTSSmoother:
    def test_smoother_reduces_variance(self):
        # Standard rule: smoothed P ≤ filtered P (PSD ordering).  We verify
        # diagonal trace strictly decreases on a tracking problem.
        f = _make_simple_kf(seed=0)
        rng = random.Random(1)
        for _ in range(20):
            f.predict()
            f.update([rng.gauss(0, 1)])
        sm = f.smooth(RTS)
        assert isinstance(sm, SmoothResult)
        assert sm.smoother == RTS
        assert len(sm.beliefs) == 20
        # Compare trace(P_filt) and trace(P_smooth) at the *middle* index.
        mid = 10
        trace_filt = sum(f._filtered_covs[mid][i][i] for i in range(2))
        trace_smooth = sum(sm.beliefs[mid].cov[i][i] for i in range(2))
        # Smoothed must be smaller (within numerical tolerance).
        assert trace_smooth <= trace_filt + 1e-9

    def test_smoother_requires_kalman(self):
        def propose(p, _u, rng): return p
        def loglik(y, p): return 0.0
        filt = Filterer.particle(
            propose_fn=propose, likelihood_fn=loglik,
            initial_particles=[[0.0]] * 4, seed=0,
        )
        filt.predict()
        filt.update([0.0])
        with pytest.raises(FiltererError):
            filt.smooth(RTS)

    def test_smoother_needs_two_steps(self):
        f = _make_simple_kf()
        f.predict()
        f.update([0.0])
        with pytest.raises(FiltererError):
            f.smooth(RTS)


# =====================================================================
# Snapshot + clear
# =====================================================================


class TestStateManagement:
    def test_snapshot_roundtrip(self):
        f = _make_simple_kf()
        for y in (0.1, 0.2, 0.3):
            f.predict()
            f.update([y])
        snap = f.snapshot()
        assert snap["n_step"] == 3
        assert len(snap["mean"]) == 2
        assert len(snap["cov"]) == 2
        assert math.isfinite(snap["log_marginal"])

    def test_clear_resets(self):
        f = _make_simple_kf()
        f.predict()
        f.update([0.5])
        f.clear()
        assert f.n_step == 0
        assert f.log_marginal == 0.0


# =====================================================================
# Composition smoke
# =====================================================================


class TestCompositionSmoke:
    """Verify the primitive composes properly with the rest of the runtime."""

    def test_event_emission(self):
        events = []

        class StubBus:
            def emit(self, kind, payload):
                events.append((kind, dict(payload)))

        f = Filterer.kalman(
            F=[[1.0]], H=[[1.0]],
            Q=[[1e-3]], R=[[0.1]],
            x0=[0.0], P0=[[1.0]],
            event_bus=StubBus(),
        )
        f.predict()
        f.update([1.0])
        f.report()
        kinds = [e[0] for e in events]
        assert FILTERER_STARTED in kinds
        assert FILTERER_PREDICTED in kinds
        assert FILTERER_UPDATED in kinds
        assert FILTERER_REPORT in kinds

    def test_log_marginal_decomposes(self):
        # The report's log_marginal equals the sum of per-step log_evidence.
        f = _make_simple_kf()
        total = 0.0
        rng = random.Random(2025)
        for _ in range(10):
            f.predict()
            r = f.update([rng.gauss(0, 1)])
            total += r.log_evidence
        report = f.report()
        assert math.isclose(report.log_marginal, total, rel_tol=1e-9)

    def test_pf_log_marginal_decomposes(self):
        def propose(p, _u, rng):
            return [p[0] + rng.gauss(0, 0.1)]

        def loglik(y, p):
            return -0.5 * ((y[0] - p[0]) / 0.2) ** 2

        init = [[0.0] for _ in range(100)]
        filt = Filterer.particle(
            propose_fn=propose, likelihood_fn=loglik,
            initial_particles=init, seed=0,
        )
        total = 0.0
        rng = random.Random(0)
        for _ in range(8):
            filt.predict()
            r = filt.update([rng.gauss(0, 0.2)])
            total += r.log_evidence
        report = filt.report()
        assert math.isclose(report.log_marginal, total, rel_tol=1e-9)

    def test_whiteness_test_on_clean_kf(self):
        f = _make_simple_kf(seed=11)
        rng = random.Random(11)
        for _ in range(60):
            f.predict()
            f.update([rng.gauss(0, 1)])
        stat, df, thr = f.whiteness_test(max_lag=3)
        assert df == 3
        assert thr > 0
        assert stat >= 0.0


# =====================================================================
# Primitive function tests (kalman_predict / kalman_update etc.)
# =====================================================================


class TestPrimitiveFunctions:
    def test_kalman_predict_deterministic(self):
        m, P = kalman_predict(
            mean=[0.0, 1.0],
            cov=[[1.0, 0.0], [0.0, 1.0]],
            F=[[1.0, 1.0], [0.0, 1.0]],
            Q=[[0.0, 0.0], [0.0, 0.0]],
        )
        assert m == [1.0, 1.0]
        # P' = F P F^T + Q = [[2, 1], [1, 1]].
        assert math.isclose(P[0][0], 2.0)
        assert math.isclose(P[0][1], 1.0)
        assert math.isclose(P[1][0], 1.0)
        assert math.isclose(P[1][1], 1.0)

    def test_kalman_update_zero_innov(self):
        # When y equals Hx the innovation is zero and the mean does not move.
        mean = [3.0]
        cov = [[2.0]]
        H = [[1.0]]
        R = [[0.5]]
        new_mean, new_cov, logev, innov, S, nis = kalman_update(
            mean, cov, [3.0], H, R,
        )
        assert math.isclose(new_mean[0], 3.0)
        assert math.isclose(innov[0], 0.0)
        assert nis == 0.0
        # Posterior variance: P' = (I − K H) P  with K = 2/(2+0.5)=0.8 →
        # P' = 0.2 * 2 = 0.4 (Joseph form).  We allow numerical tolerance.
        assert math.isclose(new_cov[0][0], 0.4, rel_tol=1e-7)

    def test_kalman_log_evidence_at_mean(self):
        # log N(0 | 0, 0.5) = −0.5 log(2π · 0.5).
        _, _, logev, _, _, _ = kalman_update(
            mean=[0.0], cov=[[0.0]], y=[0.0], H=[[1.0]], R=[[0.5]],
        )
        expected = -0.5 * (math.log(2 * math.pi) + math.log(0.5))
        assert math.isclose(logev, expected, rel_tol=1e-9, abs_tol=1e-9)

    def test_rts_smooth_two_steps(self):
        # Trivial: F=1, no noise.  Smoother should return last point at t=0.
        fm = [[1.0], [2.0]]
        fc = [[[1.0]], [[0.5]]]
        pm = [[1.0]]
        pc = [[[2.0]]]
        F = [[1.0]]
        sm, sc = rts_smooth(fm, fc, pm, pc, F)
        assert len(sm) == 2
        assert len(sc) == 2
        # Last smoothed point equals last filtered point.
        assert sm[-1] == [2.0]
        # Smoothed variance at t=0 cannot exceed filtered at t=0.
        assert sc[0][0][0] <= fc[0][0][0] + 1e-12
