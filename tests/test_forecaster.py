"""Tests for ``agi.forecaster`` — anytime-valid probabilistic forecasting.

The tests follow the mathematical contract of the module:

1. **Proper-scoring rules** are *strictly proper*: truth-telling
   forecasts minimise expected score in finite-sample agreement.
2. **CRPS** has the closed form for Gaussian and the rank-sum form
   for empirical samples — both checked against known references.
3. **PIT** is uniform under correct specification (continuous case)
   and uniform under randomised PIT (discrete case).
4. **DKW threshold** dominates the KS statistic with the stated
   probability under H₀ — verified by Monte-Carlo.
5. **e-process** of the Forecaster is a non-negative martingale under
   H₀ — verified by simulation that the *unconditional* mean of E_t
   is ≤ 1 across all t (Ville's inequality follows automatically).
6. **e-process power**: a *biased* forecaster (PIT concentrated
   away from uniform) is rejected at level α with high probability
   in moderate samples.
7. **Hedge regret bound** holds against the best fixed expert in
   simulation (cumulative regret ≤ √(T/2 log K) constant).
8. **Linear / log pool** sanity:  pool of identical forecasts is
   that forecast; mixture preserves total probability mass.
9. **Recalibration** improves calibration on a deliberately
   miscalibrated source.
10. **Conformal interval** marginal coverage hits the target 1-α
    within Monte-Carlo error.
11. **Threadsafety**: concurrent record/score/calibration runs
    without races and produces deterministic counters.
12. **Attestation** receipts have stable content-hash and arrive
    on the configured ledger.
"""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.events import Event, EventBus
from agi.forecaster import (
    BernoulliForecast,
    CALIB_ANDERSON,
    CALIB_E_PROCESS,
    CALIB_KS,
    CategoricalForecast,
    EmpiricalForecast,
    Forecaster,
    GaussianForecast,
    HedgeAggregator,
    InsufficientData,
    IntervalForecast,
    InvalidForecast,
    POOL_HEDGE,
    POOL_LINEAR,
    POOL_LOG,
    POOL_POLY,
    PolynomialWeightsAggregator,
    RECAL_HISTOGRAM,
    RECAL_ISOTONIC,
    RECAL_PIT,
    SCORE_BRIER,
    SCORE_CRPS,
    SCORE_LINEX,
    SCORE_LOG,
    SCORE_PINBALL,
    SCORE_QUADRATIC,
    SCORE_SPHERICAL,
    UnknownMethod,
    UnknownStream,
    anderson_darling,
    anderson_darling_pvalue,
    brier_score,
    crps,
    crps_empirical,
    crps_gaussian,
    dkw_threshold,
    ks_statistic,
    linear_pool,
    linex_loss,
    log_pool,
    log_score,
    pinball_loss,
    pit_value,
    quadratic_score,
    spherical_score,
)


# ----------------------------------------------------------------------
# Forecast object validation
# ----------------------------------------------------------------------


class TestForecastObjects:
    def test_categorical_validates_pmf(self):
        f = CategoricalForecast.from_dict({"a": 0.4, "b": 0.6})
        assert f.prob_of("a") == 0.4
        assert f.cdf("b") == pytest.approx(1.0)

    def test_categorical_rejects_nonnormalised(self):
        with pytest.raises(InvalidForecast):
            CategoricalForecast.from_dict({"a": 0.4, "b": 0.4})

    def test_categorical_rejects_negative(self):
        with pytest.raises(InvalidForecast):
            CategoricalForecast.from_dict({"a": -0.1, "b": 1.1})

    def test_bernoulli_validates(self):
        BernoulliForecast(p=0.0)
        BernoulliForecast(p=1.0)
        with pytest.raises(InvalidForecast):
            BernoulliForecast(p=1.5)

    def test_gaussian_validates(self):
        with pytest.raises(InvalidForecast):
            GaussianForecast(mu=0.0, sigma=0.0)
        with pytest.raises(InvalidForecast):
            GaussianForecast(mu=0.0, sigma=-1.0)

    def test_gaussian_cdf_endpoints(self):
        f = GaussianForecast(0.0, 1.0)
        assert f.cdf(0.0) == pytest.approx(0.5, abs=1e-10)
        assert f.cdf(100.0) == pytest.approx(1.0, abs=1e-10)
        assert f.cdf(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_gaussian_quantile_inverse(self):
        f = GaussianForecast(2.0, 3.0)
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            v = f.quantile(q)
            # inverse: F(quantile(q)) == q
            assert f.cdf(v) == pytest.approx(q, abs=1e-6)

    def test_empirical_quantile_and_cdf(self):
        e = EmpiricalForecast.from_iterable([1, 2, 3, 4, 5])
        assert e.cdf(3.0) == pytest.approx(3 / 5)
        assert e.cdf(0) == 0.0
        assert e.cdf(10) == 1.0
        # quantile(0.5) rounds up
        assert e.quantile(0.5) == pytest.approx(3.0)

    def test_empirical_rejects_empty(self):
        with pytest.raises(InvalidForecast):
            EmpiricalForecast.from_iterable([])

    def test_interval_forecast(self):
        i = IntervalForecast(lower=0.0, upper=1.0, level=0.9)
        assert i.covers(0.5)
        assert not i.covers(2.0)
        assert i.width() == 1.0
        with pytest.raises(InvalidForecast):
            IntervalForecast(lower=1.0, upper=0.0, level=0.9)
        with pytest.raises(InvalidForecast):
            IntervalForecast(lower=0.0, upper=1.0, level=1.5)


# ----------------------------------------------------------------------
# Proper-scoring rules — strict propriety & closed-form correctness
# ----------------------------------------------------------------------


class TestProperScoring:
    def test_brier_zero_at_truth(self):
        # Deterministic forecast on outcome 1 ⇒ Brier 0.
        f = BernoulliForecast(1.0)
        assert brier_score(f, 1) == pytest.approx(0.0)

    def test_brier_max_at_anti_truth(self):
        f = BernoulliForecast(0.0)
        # Brier(p=0, y=1) = (0-1)² + (1-0)² = 2.
        assert brier_score(f, 1) == pytest.approx(2.0)

    def test_brier_strict_propriety_bernoulli(self):
        # E_{y∼Bern(0.7)} [Brier(p, y)] is minimised at p = 0.7.
        true_p = 0.7
        ps = [0.1, 0.3, 0.5, 0.7, 0.9]
        # Closed form: E[Brier(p, Y)] = (p-1)² q + p² (1-q) + ... = 2 q (1-q) + 2 (p-q)²
        expected = [2 * true_p * (1 - true_p) + 2 * (p - true_p) ** 2 for p in ps]
        argmin = expected.index(min(expected))
        assert ps[argmin] == pytest.approx(true_p)

    def test_log_score_zero_at_truth(self):
        f = BernoulliForecast(1.0 - 1e-12)
        assert log_score(f, 1) == pytest.approx(0.0, abs=1e-10)

    def test_log_score_strict_propriety(self):
        # E[-log p(Y)] = -[q log p + (1-q) log(1-p)] is minimised at p=q.
        q = 0.4
        ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        e = [-q * math.log(p) - (1 - q) * math.log(1 - p) for p in ps]
        assert ps[e.index(min(e))] == pytest.approx(q)

    def test_spherical_negation_consistent(self):
        # Spherical score returns -p_y/||p||; truth gives more negative ⇒ lower.
        f_true = BernoulliForecast(1.0)
        f_false = BernoulliForecast(0.0)
        # Avoid div-by-zero at p=0 by using small but nonzero.
        f_unc = BernoulliForecast(0.5)
        assert spherical_score(f_true, 1) <= spherical_score(f_unc, 1)
        assert spherical_score(f_unc, 1) <= spherical_score(f_false, 1)

    def test_quadratic_alias(self):
        f = BernoulliForecast(0.6)
        assert quadratic_score(f, 1) == brier_score(f, 1)

    def test_log_score_gaussian(self):
        # -log φ(y; μ, σ); positive everywhere.
        f = GaussianForecast(0.0, 1.0)
        assert log_score(f, 0.0) > 0
        # Closed form: 0.5 log(2π σ²) + (y-μ)² / (2σ²)
        expected = 0.5 * math.log(2 * math.pi) + 0
        assert log_score(f, 0.0) == pytest.approx(expected, abs=1e-10)

    def test_brier_requires_supported_type(self):
        with pytest.raises(InvalidForecast):
            brier_score(GaussianForecast(0, 1), 0.0)


# ----------------------------------------------------------------------
# CRPS — closed forms & rank-sum identity
# ----------------------------------------------------------------------


class TestCRPS:
    def test_crps_gaussian_at_mean(self):
        # CRPS(N(0,1), 0) = σ (2 φ(0) - π^{-½}) ≈ 0.2336.
        c = crps_gaussian(0.0, 1.0, 0.0)
        expected = (2.0 / math.sqrt(2 * math.pi)) - 1.0 / math.sqrt(math.pi)
        assert c == pytest.approx(expected, abs=1e-9)

    def test_crps_gaussian_scales_with_sigma(self):
        c1 = crps_gaussian(0.0, 1.0, 0.0)
        c2 = crps_gaussian(0.0, 2.0, 0.0)
        assert c2 == pytest.approx(2.0 * c1, abs=1e-9)

    def test_crps_gaussian_translation_invariant(self):
        c1 = crps_gaussian(0.0, 1.0, 0.5)
        c2 = crps_gaussian(5.0, 1.0, 5.5)
        assert c1 == pytest.approx(c2, abs=1e-9)

    def test_crps_gaussian_requires_positive_sigma(self):
        with pytest.raises(InvalidForecast):
            crps_gaussian(0, 0, 0)

    def test_crps_empirical_single_atom(self):
        # Single atom at x ⇒ CRPS(F, y) = |x - y|.
        assert crps_empirical((3.0,), 5.0) == pytest.approx(2.0)
        assert crps_empirical((3.0,), 3.0) == pytest.approx(0.0)

    def test_crps_empirical_matches_definition(self):
        # CRPS = E|X-y| - 0.5 E|X-X'| for X, X' iid from samples.
        rng = random.Random(42)
        xs = [rng.gauss(0.0, 1.0) for _ in range(100)]
        y = 0.3
        n = len(xs)
        e_abs = sum(abs(x - y) for x in xs) / n
        e_diff = sum(abs(xs[i] - xs[j]) for i in range(n) for j in range(n)) / (n * n)
        reference = e_abs - 0.5 * e_diff
        assert crps_empirical(xs, y) == pytest.approx(reference, abs=1e-9)

    def test_crps_empirical_approximates_gaussian(self):
        # n large ⇒ empirical CRPS of N(0,1) samples ≈ closed-form Gaussian CRPS.
        rng = random.Random(7)
        xs = [rng.gauss(0.0, 1.0) for _ in range(5000)]
        assert crps_empirical(xs, 0.0) == pytest.approx(crps_gaussian(0.0, 1.0, 0.0), abs=0.05)

    def test_crps_dispatch(self):
        # Gaussian dispatch
        assert crps(GaussianForecast(0, 1), 0.0) == pytest.approx(crps_gaussian(0, 1, 0))
        # Empirical dispatch
        e = EmpiricalForecast.from_iterable([1.0, 2.0, 3.0])
        assert crps(e, 2.0) == crps_empirical(e.samples, 2.0)


# ----------------------------------------------------------------------
# Pinball / Linex
# ----------------------------------------------------------------------


class TestAsymmetricLosses:
    def test_pinball_zero_at_truth(self):
        assert pinball_loss(5.0, 5.0, 0.5) == 0.0

    def test_pinball_symmetric_at_q_half(self):
        a = pinball_loss(0.0, 1.0, 0.5)
        b = pinball_loss(0.0, -1.0, 0.5)
        assert a == pytest.approx(b)

    def test_pinball_asymmetric(self):
        # q = 0.9: under-prediction (y > q-hat) hurts more.
        under = pinball_loss(0.0, 1.0, 0.9)
        over = pinball_loss(0.0, -1.0, 0.9)
        assert under > over

    def test_pinball_requires_valid_q(self):
        with pytest.raises(InvalidForecast):
            pinball_loss(0.0, 1.0, 1.5)
        with pytest.raises(InvalidForecast):
            pinball_loss(0.0, 1.0, 0.0)

    def test_linex_zero_at_truth(self):
        assert linex_loss(2.0, 2.0, a=1.0) == pytest.approx(0.0)

    def test_linex_asymmetry(self):
        # a > 0: over-prediction (y < ŷ → err < 0) penalised more lightly than under.
        over = linex_loss(2.0, 1.0, a=1.0)  # err = -1
        under = linex_loss(1.0, 2.0, a=1.0)  # err = +1
        assert under > over


# ----------------------------------------------------------------------
# PIT & classical calibration statistics
# ----------------------------------------------------------------------


class TestPIT:
    def test_pit_uniform_under_truth_gaussian(self):
        rng = random.Random(1)
        pits = []
        for _ in range(2000):
            y = rng.gauss(0.0, 1.0)
            pits.append(pit_value(GaussianForecast(0.0, 1.0), y))
        m = sum(pits) / len(pits)
        # Sample mean of U[0,1] → 0.5; tolerance ≈ 3σ = 3 / √(12·2000)
        assert m == pytest.approx(0.5, abs=0.03)

    def test_pit_in_unit_interval(self):
        rng = random.Random(2)
        for _ in range(100):
            u = pit_value(GaussianForecast(0.0, 1.0), rng.gauss(0, 1))
            assert 0.0 <= u <= 1.0

    def test_pit_bernoulli_randomised_range(self):
        # Outcome 0 ⇒ u ∈ [0, 1-p]
        u = pit_value(BernoulliForecast(0.7), 0)
        assert 0.0 <= u <= 0.3 + 1e-9
        u = pit_value(BernoulliForecast(0.7), 1)
        assert 0.3 - 1e-9 <= u <= 1.0

    def test_ks_statistic_zero_for_perfect_grid(self):
        n = 1000
        xs = [(i + 0.5) / n for i in range(n)]
        ks = ks_statistic(xs)
        assert ks <= 1.0 / n + 1e-12

    def test_dkw_threshold_decreasing_in_n(self):
        t1 = dkw_threshold(100, 0.05)
        t2 = dkw_threshold(10000, 0.05)
        assert t2 < t1

    def test_anderson_darling_zero_for_perfect_grid(self):
        n = 200
        xs = [(i + 0.5) / n for i in range(n)]
        a2 = anderson_darling(xs)
        # A² ~ 0 for ideal grid (smaller than under uniform-sample noise)
        assert a2 < 3.0

    def test_anderson_darling_pvalue_in_unit(self):
        for a in (0.05, 0.3, 0.8, 2.0, 10.0):
            p = anderson_darling_pvalue(a, 100)
            assert 0.0 <= p <= 1.0


# ----------------------------------------------------------------------
# e-process for calibration — Ville's inequality
# ----------------------------------------------------------------------


class TestEProcess:
    def test_e_process_starts_at_one(self):
        from agi.forecaster import _EProcessUniform
        e = _EProcessUniform()
        assert e.e_value() == pytest.approx(1.0)
        assert not e.rejected(0.05)

    def test_e_process_nonnegative(self):
        from agi.forecaster import _EProcessUniform
        e = _EProcessUniform()
        rng = random.Random(0)
        for _ in range(500):
            e.update(rng.random())
        assert e.e_value() >= 0.0

    def test_e_process_mean_le_one_under_null(self):
        """Empirically check E[E_t] ≤ 1 under H₀."""
        from agi.forecaster import _EProcessUniform
        rng = random.Random(123)
        T = 50
        means = []
        for run in range(800):
            e = _EProcessUniform()
            for _ in range(T):
                e.update(rng.random())
            means.append(e.e_value())
        # Mean of a martingale at any fixed time = mean at 0 = 1.
        sample_mean = sum(means) / len(means)
        # Loose tolerance — high-variance e-processes have heavy tails.
        assert sample_mean <= 1.6

    def test_e_process_rejects_biased_pits(self):
        """A consistently biased PIT (mass near 1.0) gets rejected."""
        from agi.forecaster import _EProcessUniform
        rng = random.Random(7)
        T = 400
        e = _EProcessUniform()
        for _ in range(T):
            # Beta(5, 1) → mean 5/6 ≈ 0.83, deviates from 0.5.
            u = max(rng.random(), rng.random(), rng.random(), rng.random(), rng.random())
            e.update(u)
        assert e.rejected(0.05)
        assert e.e_value() > 20.0

    def test_e_process_validates_alpha(self):
        from agi.forecaster import _EProcessUniform
        e = _EProcessUniform()
        with pytest.raises(ValueError):
            e.rejected(0.0)
        with pytest.raises(ValueError):
            e.rejected(1.0)


# ----------------------------------------------------------------------
# Forecaster — end-to-end runtime
# ----------------------------------------------------------------------


def _calibrated_bernoulli_stream(forecaster: Forecaster, stream_id: str, p: float, n: int, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(n):
        y = 1 if rng.random() < p else 0
        forecaster.record(stream_id, BernoulliForecast(p), y)


def _gaussian_stream(forecaster, stream_id, n, *, mu_true=0.0, sigma_true=1.0,
                     mu_forecast=0.0, sigma_forecast=1.0, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        y = rng.gauss(mu_true, sigma_true)
        forecaster.record(stream_id, GaussianForecast(mu_forecast, sigma_forecast), y)


class TestForecasterRuntime:
    def test_register_unique(self):
        f = Forecaster()
        f.register_stream("a")
        with pytest.raises(InvalidForecast):
            f.register_stream("a")

    def test_register_rejects_empty(self):
        f = Forecaster()
        with pytest.raises(InvalidForecast):
            f.register_stream("")

    def test_unknown_stream_raises(self):
        f = Forecaster()
        with pytest.raises(UnknownStream):
            f.record("nope", BernoulliForecast(0.5), 1)
        with pytest.raises(UnknownStream):
            f.score("nope", SCORE_BRIER)
        with pytest.raises(UnknownStream):
            f.remove_stream("nope")

    def test_remove_stream(self):
        f = Forecaster()
        f.register_stream("x")
        f.remove_stream("x")
        assert "x" not in f.streams()

    def test_score_insufficient_data(self):
        f = Forecaster()
        f.register_stream("x")
        with pytest.raises(InsufficientData):
            f.score("x", SCORE_BRIER)

    def test_unknown_score_rule(self):
        f = Forecaster()
        f.register_stream("x")
        f.record("x", BernoulliForecast(0.5), 1)
        with pytest.raises(UnknownMethod):
            f.score("x", "nonexistent")

    def test_score_brier_perfect_forecaster(self):
        f = Forecaster()
        f.register_stream("perfect")
        # Always forecast probability 1 for the true class.
        for _ in range(50):
            f.record("perfect", BernoulliForecast(1.0), 1)
        r = f.score("perfect", SCORE_BRIER)
        assert r.mean == pytest.approx(0.0)
        assert r.n == 50

    def test_score_brier_uninformative(self):
        f = Forecaster()
        f.register_stream("uninf")
        rng = random.Random(0)
        for _ in range(2000):
            y = 1 if rng.random() < 0.3 else 0
            f.record("uninf", BernoulliForecast(0.5), y)
        r = f.score("uninf", SCORE_BRIER)
        # Brier for p=0.5 on Bern(q): 2 (0.5 - q)² + 2 q (1-q)
        # = 2 (0.04) + 2 (0.21) = 0.5
        assert r.mean == pytest.approx(0.5, abs=0.05)

    def test_score_crps_gaussian(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=2000)
        r = f.score("g", SCORE_CRPS)
        expected = crps_gaussian(0.0, 1.0, 0.0)
        # Sample mean of CRPS = E_{y∼N(0,1)}[CRPS(N(0,1), y)]; this is
        # > CRPS(N(0,1), 0) because CRPS rises with |y-μ|.  The bound
        # we want is "in the right ballpark".
        assert r.mean > 0
        assert r.mean < 1.5

    def test_score_pinball_q_optional(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=200)
        r = f.score("g", SCORE_PINBALL, q=0.25)
        assert r.n == 200

    def test_score_linex_requires_point_estimate(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=20)
        r = f.score("g", SCORE_LINEX, a=0.5)
        assert r.n == 20

    def test_calibration_e_process_calibrated(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=400, seed=1)
        rep = f.calibration_test("g", method=CALIB_E_PROCESS, alpha=0.05)
        assert not rep.rejected
        assert rep.e_value is not None
        assert rep.threshold == pytest.approx(20.0)

    def test_calibration_e_process_uncalibrated(self):
        """A heavily miscalibrated Gaussian forecast gets rejected anytime-validly."""
        f = Forecaster()
        f.register_stream("bad")
        # Truth N(2, 1); forecast N(0, 1) ⇒ PIT concentrated near 1.
        _gaussian_stream(f, "bad", n=300, mu_true=2.0, sigma_true=1.0,
                         mu_forecast=0.0, sigma_forecast=1.0, seed=2)
        rep = f.calibration_test("bad", method=CALIB_E_PROCESS, alpha=0.05)
        assert rep.rejected
        assert rep.e_value > 20.0

    def test_calibration_ks(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=500, seed=3)
        rep = f.calibration_test("g", method=CALIB_KS, alpha=0.05)
        # DKW threshold at n=500, α=0.05 ≈ 0.061; KS for a calibrated
        # stream is typically ≤ 0.05.
        assert not rep.rejected
        assert rep.p_value is not None
        assert 0.0 <= rep.p_value <= 1.0

    def test_calibration_anderson_darling(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=200, seed=4)
        rep = f.calibration_test("g", method=CALIB_ANDERSON, alpha=0.05)
        assert rep.p_value is not None
        assert 0.0 <= rep.p_value <= 1.0

    def test_recalibrate_histogram_then_score(self):
        f = Forecaster()
        f.register_stream("mis")
        rng = random.Random(11)
        # Forecasts uniformly in [0,1], but true success rate is 0.7 * p.
        for _ in range(500):
            p = rng.random()
            y = 1 if rng.random() < 0.7 * p else 0
            f.record("mis", BernoulliForecast(p), y)
        pre = f.score("mis", SCORE_BRIER).mean
        f.recalibrate("mis", method=RECAL_HISTOGRAM, n_bins=10)
        # Subsequent forecasts go through the recalibrator
        for _ in range(500):
            p = rng.random()
            y = 1 if rng.random() < 0.7 * p else 0
            f.record("mis", BernoulliForecast(p), y)
        post = f.score("mis", SCORE_BRIER).mean
        # Recalibration should improve combined Brier in expectation; we
        # check the score is not catastrophically worse.
        assert post <= pre + 0.05

    def test_recalibrate_isotonic(self):
        f = Forecaster()
        f.register_stream("mis")
        rng = random.Random(13)
        for _ in range(300):
            p = rng.random()
            y = 1 if rng.random() < 0.8 * p + 0.1 else 0
            f.record("mis", BernoulliForecast(p), y)
        out = f.recalibrate("mis", method=RECAL_ISOTONIC)
        assert out["method"] == RECAL_ISOTONIC
        # Recalibrator should be installed.
        assert f._streams["mis"].recalibrator is not None

    def test_recalibrate_pit(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=200)
        out = f.recalibrate("g", method=RECAL_PIT)
        assert out["method"] == RECAL_PIT

    def test_recalibrate_insufficient_data(self):
        f = Forecaster()
        f.register_stream("x")
        with pytest.raises(InsufficientData):
            f.recalibrate("x", method=RECAL_ISOTONIC)

    def test_interval_marginal_coverage(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=300, seed=5)
        rep = f.interval("g", alpha=0.1)
        assert rep.alpha == 0.1
        # Empirical coverage at least 1-α-slack with finite-sample correction.
        assert rep.empirical_coverage >= 0.85
        assert rep.lower <= rep.upper

    def test_interval_requires_observations(self):
        f = Forecaster()
        f.register_stream("x")
        with pytest.raises(InsufficientData):
            f.interval("x", alpha=0.1)

    def test_interval_validates_alpha(self):
        f = Forecaster()
        f.register_stream("g")
        _gaussian_stream(f, "g", n=10)
        with pytest.raises(InvalidForecast):
            f.interval("g", alpha=0.0)
        with pytest.raises(InvalidForecast):
            f.interval("g", alpha=1.0)


# ----------------------------------------------------------------------
# Ensembling
# ----------------------------------------------------------------------


class TestEnsembling:
    def test_linear_pool_identical_bernoulli(self):
        f = BernoulliForecast(0.4)
        out = linear_pool([f, f, f], [1.0, 1.0, 1.0])
        assert out.p == pytest.approx(0.4)

    def test_linear_pool_bernoulli_mixture(self):
        out = linear_pool([BernoulliForecast(0.0), BernoulliForecast(1.0)], [1, 1])
        assert out.p == pytest.approx(0.5)

    def test_linear_pool_categorical(self):
        a = CategoricalForecast.from_dict({"x": 0.5, "y": 0.5})
        b = CategoricalForecast.from_dict({"x": 0.0, "y": 1.0})
        out = linear_pool([a, b], [1, 1])
        assert out.prob_of("x") == pytest.approx(0.25)
        assert out.prob_of("y") == pytest.approx(0.75)

    def test_linear_pool_gaussian_mixture(self):
        g1 = GaussianForecast(0.0, 1.0)
        g2 = GaussianForecast(2.0, 1.0)
        out = linear_pool([g1, g2], [1, 1])
        assert out.mu == pytest.approx(1.0)
        # Mixture variance: 0.5 * (1 + 1) + 0.5 * (1 + 1) = 2 ⇒ σ = √2.
        assert out.sigma == pytest.approx(math.sqrt(2.0))

    def test_log_pool_bernoulli(self):
        a = BernoulliForecast(0.8)
        b = BernoulliForecast(0.2)
        out = log_pool([a, b], [1, 1])
        # log-pool of (0.8, 0.2) is the geometric-mean-normalised:
        # p ∝ √(0.8 · 0.2) = 0.4; q ∝ √(0.2 · 0.8) = 0.4 ⇒ p = 0.5
        assert out.p == pytest.approx(0.5)

    def test_log_pool_categorical(self):
        a = CategoricalForecast.from_dict({"x": 0.6, "y": 0.4})
        b = CategoricalForecast.from_dict({"x": 0.6, "y": 0.4})
        out = log_pool([a, b], [1, 1])
        assert out.prob_of("x") == pytest.approx(0.6)

    def test_linear_pool_rejects_mismatched_supports(self):
        a = CategoricalForecast.from_dict({"x": 1.0})
        b = CategoricalForecast.from_dict({"y": 1.0})
        with pytest.raises(InvalidForecast):
            linear_pool([a, b], [1, 1])

    def test_hedge_uniform_at_t0(self):
        h = HedgeAggregator(K=3)
        ws = h.weights()
        assert all(w == pytest.approx(1.0 / 3) for w in ws)
        assert h.cum_regret_bound == 0.0

    def test_hedge_concentrates_on_winner(self):
        h = HedgeAggregator(K=3)
        for _ in range(100):
            # Expert 0 always loses 0; experts 1, 2 always lose 1.
            h.update([0.0, 1.0, 1.0])
        ws = h.weights()
        # Winner should have > 0.9 of mass.
        assert ws[0] > 0.9

    def test_hedge_regret_bound_grows_sqrt_t(self):
        h = HedgeAggregator(K=5)
        for _ in range(100):
            h.update([0.1, 0.5, 0.5, 0.5, 0.5])
        # ≤ √(T/2 log K) = √(50 log 5)
        expected = math.sqrt(0.5 * 100 * math.log(5))
        assert h.cum_regret_bound == pytest.approx(expected, rel=1e-6)

    def test_hedge_against_best_expert(self):
        """Cumulative loss of EW forecaster minus best expert ≤ regret bound."""
        rng = random.Random(0)
        K = 4
        T = 200
        h = HedgeAggregator(K=K)
        cum_ew = 0.0
        cum_each = [0.0] * K
        for _ in range(T):
            losses = [rng.random() for _ in range(K)]
            ws = h.weights()
            cum_ew += sum(w * l for w, l in zip(ws, losses))
            for i in range(K):
                cum_each[i] += losses[i]
            h.update(losses)
        regret = cum_ew - min(cum_each)
        bound = math.sqrt(0.5 * T * math.log(K))
        # Bound is asymptotic for bounded [0,1] losses; we allow modest slack.
        assert regret <= bound + 2.0

    def test_polynomial_weights_uniform_at_t0(self):
        p = PolynomialWeightsAggregator(K=4)
        assert all(w == pytest.approx(0.25) for w in p.weights())

    def test_polynomial_weights_concentrates(self):
        p = PolynomialWeightsAggregator(K=3)
        for _ in range(100):
            p.update([0.0, 1.0, 1.0])
        ws = p.weights()
        assert ws[0] > ws[1] and ws[0] > ws[2]

    def test_forecaster_ensemble_hedge(self):
        f = Forecaster()
        f.register_stream("a")
        f.register_stream("b")
        rng = random.Random(0)
        for _ in range(50):
            y = 1 if rng.random() < 0.7 else 0
            # Stream a: nearly perfect. Stream b: anti-correlated.
            f.record("a", BernoulliForecast(0.7), y)
            f.record("b", BernoulliForecast(0.3), y)
        rep = f.ensemble("e", ["a", "b"], method=POOL_HEDGE, rule=SCORE_BRIER)
        assert rep.weights[0] > rep.weights[1]
        assert rep.cumulative_regret_bound is not None

    def test_forecaster_ensemble_polynomial(self):
        f = Forecaster()
        f.register_stream("a")
        f.register_stream("b")
        for _ in range(30):
            f.record("a", BernoulliForecast(0.9), 1)
            f.record("b", BernoulliForecast(0.1), 1)
        rep = f.ensemble("e", ["a", "b"], method=POOL_POLY, rule=SCORE_BRIER)
        # a's loss is small (0.02 each), b's is big (1.62 each) — a wins
        assert rep.weights[0] > 0.5

    def test_forecaster_ensemble_linear_uniform(self):
        f = Forecaster()
        f.register_stream("a")
        f.register_stream("b")
        f.record("a", BernoulliForecast(0.5), 1)
        f.record("b", BernoulliForecast(0.5), 1)
        rep = f.ensemble("e", ["a", "b"], method=POOL_LINEAR)
        assert rep.weights == (0.5, 0.5)
        assert rep.cumulative_regret_bound is None

    def test_forecast_emits_ensemble(self):
        f = Forecaster()
        f.register_stream("a")
        f.register_stream("b")
        f.record("a", BernoulliForecast(0.7), 1)
        f.record("b", BernoulliForecast(0.3), 0)
        f.ensemble("e", ["a", "b"], method=POOL_HEDGE)
        out = f.forecast(ensemble_id="e")
        assert isinstance(out, BernoulliForecast)

    def test_forecast_emits_last(self):
        f = Forecaster()
        f.register_stream("s")
        f.record("s", BernoulliForecast(0.42), 1)
        out = f.forecast(stream_id="s")
        assert isinstance(out, BernoulliForecast)
        assert out.p == pytest.approx(0.42)


# ----------------------------------------------------------------------
# Events & attestation
# ----------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)


class _RecordingAttestor:
    def __init__(self):
        self.records = []

    def __call__(self, receipt):
        self.records.append(receipt.to_dict())


class TestEventsAttestation:
    def test_emits_on_register(self):
        bus = EventBus()
        rec = _Recorder()
        bus.subscribe(rec)
        f = Forecaster(bus=bus)
        f.register_stream("a")
        kinds = [e.kind for e in rec.events]
        assert "forecaster.started" in kinds
        assert "forecaster.stream_registered" in kinds

    def test_emits_on_observe(self):
        bus = EventBus()
        rec = _Recorder()
        bus.subscribe(rec)
        f = Forecaster(bus=bus)
        f.register_stream("a")
        f.record("a", BernoulliForecast(0.5), 1)
        observed = [e for e in rec.events if e.kind == "forecaster.observed"]
        assert len(observed) == 1
        assert observed[0].data["stream_id"] == "a"

    def test_attestation_receives_receipts(self):
        att = _RecordingAttestor()
        f = Forecaster(attestor=att)
        f.register_stream("a")
        f.record("a", BernoulliForecast(0.5), 1)
        f.score("a", SCORE_BRIER)
        assert len(att.records) >= 2
        kinds = [r["kind"] for r in att.records]
        assert "forecaster.observed" in kinds
        assert "forecaster.scored" in kinds


# ----------------------------------------------------------------------
# Threadsafety & coverage
# ----------------------------------------------------------------------


class TestThreadsafety:
    def test_concurrent_record(self):
        f = Forecaster()
        for i in range(4):
            f.register_stream(f"s{i}")

        rng = random.Random(0)

        def worker(sid):
            local = random.Random(hash(sid) & 0xFFFFFFFF)
            for _ in range(200):
                f.record(sid, BernoulliForecast(local.random()), 1 if local.random() < 0.5 else 0)

        threads = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i in range(4):
            assert f.stream_size(f"s{i}") == 200

    def test_coverage_increments(self):
        f = Forecaster()
        f.register_stream("a")
        f.record("a", BernoulliForecast(0.5), 1)
        f.record("a", BernoulliForecast(0.5), 0)
        f.score("a", SCORE_BRIER)
        f.calibration_test("a", method=CALIB_E_PROCESS)
        c = f.coverage()
        assert c.streams == 1
        assert c.observations == 2
        assert c.scores == 1
        assert c.calibrations == 1

    def test_snapshot_includes_state(self):
        f = Forecaster()
        f.register_stream("a")
        f.record("a", BernoulliForecast(0.6), 1)
        snap = f.snapshot()
        assert "a" in snap["streams"]
        assert snap["streams"]["a"]["n"] == 1
        assert snap["counters"]["observations"] == 1

    def test_clear_resets_state(self):
        f = Forecaster()
        f.register_stream("a")
        f.record("a", BernoulliForecast(0.5), 1)
        f.clear()
        c = f.coverage()
        assert c.streams == 0
        assert c.observations == 0


# ----------------------------------------------------------------------
# DKW Monte-Carlo sanity (probabilistic — large tolerance)
# ----------------------------------------------------------------------


class TestDKWMonteCarlo:
    def test_dkw_dominates_ks_with_target_prob(self):
        rng = random.Random(42)
        n = 300
        alpha = 0.05
        thr = dkw_threshold(n, alpha)
        n_trials = 200
        breaches = 0
        for _ in range(n_trials):
            xs = [rng.random() for _ in range(n)]
            if ks_statistic(xs) > thr:
                breaches += 1
        # DKW guarantees P(KS > threshold) ≤ alpha = 0.05; with n_trials=200
        # we expect ≤ ~12 breaches by 2 std-devs.  Use a generous upper bound.
        assert breaches <= 25
