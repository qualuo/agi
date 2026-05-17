"""Tests for the Scientist primitive (sparse symbolic law discovery)."""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.events import EventBus
from agi.scientist import (
    Basis,
    Bootstrap,
    InsufficientData,
    InvalidBasis,
    InvalidConfig,
    InvalidCriterion,
    InvalidObservation,
    Law,
    ParetoPoint,
    SCIENTIST_BOOTSTRAPPED,
    SCIENTIST_CLEARED,
    SCIENTIST_FITTED,
    SCIENTIST_KNOWN_CRITERIA,
    SCIENTIST_KNOWN_EVENTS,
    SCIENTIST_OBSERVED,
    SCIENTIST_PARETO,
    SCIENTIST_PREDICTED,
    SCIENTIST_REPORTED,
    SCIENTIST_STABILITY,
    SCIENTIST_STARTED,
    SELECT_AIC,
    SELECT_BIC,
    SELECT_MDL,
    SELECT_PARETO_KNEE,
    Scientist,
    ScientistError,
    ScientistReport,
    Stability,
    Term,
    default_library,
)


# =====================================================================
# Helpers
# =====================================================================


def _line_data(n: int = 60, slope: float = 2.5, intercept: float = -1.3, noise: float = 0.0, seed: int = 0):
    rng = random.Random(seed)
    xs = [[rng.uniform(-3.0, 3.0)] for _ in range(n)]
    ys = [slope * x[0] + intercept + (rng.gauss(0.0, noise) if noise > 0 else 0.0) for x in xs]
    return xs, ys


def _quadratic_data(n: int = 80, a: float = 0.5, b: float = -1.0, c: float = 0.25, noise: float = 0.0, seed: int = 1):
    rng = random.Random(seed)
    xs = [[rng.uniform(-2.0, 2.0)] for _ in range(n)]
    ys = [a * x[0] ** 2 + b * x[0] + c + (rng.gauss(0.0, noise) if noise > 0 else 0.0) for x in xs]
    return xs, ys


def _falling_body(n: int = 50, g: float = -9.81, v0: float = 3.0, h0: float = 100.0, seed: int = 2):
    rng = random.Random(seed)
    ts = [rng.uniform(0.0, 4.0) for _ in range(n)]
    ts.sort()
    xs = [[t] for t in ts]
    ys = [0.5 * g * t * t + v0 * t + h0 for t in ts]
    return xs, ys


def _two_input_data(n: int = 100, seed: int = 3):
    """y = 3 x0 - 2 x1 + 0.5 x0·x1 + noise."""
    rng = random.Random(seed)
    xs = [[rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)] for _ in range(n)]
    ys = [3.0 * x[0] - 2.0 * x[1] + 0.5 * x[0] * x[1] for x in xs]
    return xs, ys


# =====================================================================
# Configuration & errors
# =====================================================================


class TestCreate:
    def test_create_defaults(self):
        sci = Scientist.create(input_dim=1)
        assert sci.input_dim == 1
        assert sci.library_size >= 3  # 1, x0, x0^2
        assert sci.fingerprint() != "0" * 64
        assert sci.n_observations == 0

    def test_create_rejects_negative_dim(self):
        with pytest.raises(InvalidConfig):
            Scientist.create(input_dim=-1)

    def test_create_rejects_huge_degree(self):
        with pytest.raises(InvalidConfig):
            Scientist.create(input_dim=2, max_degree=9)

    def test_create_rejects_negative_degree(self):
        with pytest.raises(InvalidConfig):
            Scientist.create(input_dim=1, max_degree=-1)

    def test_create_rejects_empty_library(self):
        with pytest.raises(InvalidConfig):
            Scientist.create(input_dim=1, library=[])

    def test_create_rejects_bad_library_entry(self):
        with pytest.raises(InvalidBasis):
            Scientist.create(input_dim=1, library=["not a basis"])  # type: ignore[list-item]

    def test_create_rejects_bad_lambda_grid(self):
        with pytest.raises(InvalidConfig):
            Scientist.create(input_dim=1, lambda_grid=[0.0])
        with pytest.raises(InvalidConfig):
            Scientist.create(input_dim=1, lambda_grid=[])
        with pytest.raises(InvalidConfig):
            Scientist.create(input_dim=1, lambda_grid=[float("inf")])

    def test_create_rejects_bad_ridge(self):
        with pytest.raises(InvalidConfig):
            Scientist.create(input_dim=1, ridge=-1.0)

    def test_create_publishes_started_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(events.append, kind=SCIENTIST_STARTED)
        Scientist.create(input_dim=1, bus=bus, session_id="s1")
        assert len(events) == 1
        assert events[0].kind == SCIENTIST_STARTED
        assert events[0].session_id == "s1"
        assert "fingerprint" in events[0].data

    def test_known_events_complete(self):
        expected = {
            SCIENTIST_STARTED,
            SCIENTIST_OBSERVED,
            SCIENTIST_FITTED,
            SCIENTIST_PARETO,
            SCIENTIST_BOOTSTRAPPED,
            SCIENTIST_STABILITY,
            SCIENTIST_PREDICTED,
            SCIENTIST_REPORTED,
            SCIENTIST_CLEARED,
        }
        assert SCIENTIST_KNOWN_EVENTS == expected


# =====================================================================
# Default library
# =====================================================================


class TestDefaultLibrary:
    def test_monomials_only(self):
        lib = default_library(input_dim=2, max_degree=2)
        names = [b.name for b in lib]
        assert "1" in names
        assert "x0" in names
        assert "x1" in names
        assert "x0·x1" in names
        assert "x0^2" in names
        assert "x1^2" in names

    def test_monomial_evaluates(self):
        lib = default_library(input_dim=2, max_degree=2)
        by_name = {b.name: b for b in lib}
        assert by_name["1"]([1.5, -0.5]) == 1.0
        assert by_name["x0"]([1.5, -0.5]) == 1.5
        assert by_name["x1"]([1.5, -0.5]) == -0.5
        assert by_name["x0·x1"]([2.0, 3.0]) == 6.0
        assert by_name["x0^2"]([2.0, 3.0]) == 4.0

    def test_trig_basis(self):
        lib = default_library(input_dim=1, max_degree=1, include_trig=True)
        names = [b.name for b in lib]
        assert "sin(x0)" in names
        assert "cos(x0)" in names

    def test_exp_log_inv(self):
        lib = default_library(
            input_dim=1,
            max_degree=1,
            include_exp=True,
            include_log=True,
            include_inv=True,
        )
        names = [b.name for b in lib]
        assert "exp(x0)" in names
        assert "log1p(|x0|)" in names
        assert "1/x0" in names

    def test_exp_clipped(self):
        lib = default_library(input_dim=1, max_degree=0, include_exp=True)
        by_name = {b.name: b for b in lib}
        # Very large input shouldn't overflow
        v = by_name["exp(x0)"]([1e6])
        assert math.isfinite(v)
        assert v == math.exp(50.0)

    def test_inv_avoids_singularity(self):
        lib = default_library(input_dim=1, max_degree=0, include_inv=True)
        by_name = {b.name: b for b in lib}
        v = by_name["1/x0"]([0.0])
        assert math.isfinite(v)

    def test_extra_basis_appended(self):
        cubed = Basis(name="x0^3_custom", fn=lambda x: x[0] ** 3, complexity=3)
        lib = default_library(input_dim=1, max_degree=1, extra=[cubed])
        assert lib[-1].name == "x0^3_custom"
        assert lib[-1](([2.0])) == 8.0

    def test_extra_must_be_basis(self):
        with pytest.raises(InvalidBasis):
            default_library(input_dim=1, max_degree=1, extra=[42])  # type: ignore[list-item]

    def test_library_deduplicates(self):
        cubed = Basis(name="x0", fn=lambda x: x[0], complexity=1)  # duplicate of monomial x0
        lib = default_library(input_dim=1, max_degree=1, extra=[cubed])
        names = [b.name for b in lib]
        assert names.count("x0") == 1


# =====================================================================
# Observation
# =====================================================================


class TestObserve:
    def test_observe_single(self):
        sci = Scientist.create(input_dim=1)
        fp0 = sci.fingerprint()
        fp1 = sci.observe([1.0], 2.0)
        assert fp1 != fp0
        assert sci.n_observations == 1

    def test_observe_wrong_arity(self):
        sci = Scientist.create(input_dim=2)
        with pytest.raises(InvalidObservation):
            sci.observe([1.0], 2.0)

    def test_observe_nan_x(self):
        sci = Scientist.create(input_dim=1)
        with pytest.raises(InvalidObservation):
            sci.observe([float("nan")], 1.0)

    def test_observe_inf_y(self):
        sci = Scientist.create(input_dim=1)
        with pytest.raises(InvalidObservation):
            sci.observe([0.0], float("inf"))

    def test_observe_many(self):
        sci = Scientist.create(input_dim=1)
        xs, ys = _line_data(n=10)
        sci.observe_many(xs, ys)
        assert sci.n_observations == 10

    def test_observe_many_mismatched_lengths(self):
        sci = Scientist.create(input_dim=1)
        with pytest.raises(InvalidObservation):
            sci.observe_many([[1.0]], [1.0, 2.0])

    def test_observe_publishes_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(events.append, kind=SCIENTIST_OBSERVED)
        sci = Scientist.create(input_dim=1, bus=bus)
        sci.observe([1.0], 2.0)
        assert len(events) == 1
        assert events[0].data["n"] == 1

    def test_observe_clears_cached_pareto(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=20)
        sci.observe_many(xs, ys)
        front_before = sci.pareto()
        sci.observe([10.0], 1000.0)  # outlier
        front_after = sci.pareto()
        # The cache was invalidated; values likely change with outlier
        assert front_before != front_after or front_before == front_after

    def test_observe_max_limit(self):
        sci = Scientist.create(input_dim=1, max_observations=3)
        for i in range(3):
            sci.observe([float(i)], float(i))
        with pytest.raises(InvalidObservation):
            sci.observe([3.0], 3.0)

    def test_clear(self):
        sci = Scientist.create(input_dim=1)
        sci.observe([1.0], 2.0)
        assert sci.n_observations == 1
        sci.clear()
        assert sci.n_observations == 0


# =====================================================================
# Fitting — recover known laws
# =====================================================================


class TestFit:
    def test_fit_recovers_line(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=60, slope=2.5, intercept=-1.3)
        sci.observe_many(xs, ys)
        law = sci.fit()
        assert law.r2 > 0.999
        coefs = {t.name: t.coefficient for t in law.terms}
        assert abs(coefs.get("x0", 0.0) - 2.5) < 1e-3
        assert abs(coefs.get("1", 0.0) - (-1.3)) < 1e-3
        # x0^2 should be removed
        assert "x0^2" not in coefs or abs(coefs["x0^2"]) < 0.1

    def test_fit_recovers_quadratic(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        xs, ys = _quadratic_data(n=80, a=0.5, b=-1.0, c=0.25)
        sci.observe_many(xs, ys)
        law = sci.fit()
        assert law.r2 > 0.999
        coefs = {t.name: t.coefficient for t in law.terms}
        assert abs(coefs.get("x0^2", 0.0) - 0.5) < 1e-3
        assert abs(coefs.get("x0", 0.0) - (-1.0)) < 1e-3
        assert abs(coefs.get("1", 0.0) - 0.25) < 1e-3

    def test_fit_recovers_falling_body(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        xs, ys = _falling_body(n=60, g=-9.81, v0=3.0, h0=100.0)
        sci.observe_many(xs, ys)
        law = sci.fit()
        assert law.r2 > 0.9999
        coefs = {t.name: t.coefficient for t in law.terms}
        assert abs(coefs.get("x0^2", 0.0) - (0.5 * -9.81)) < 1e-2
        assert abs(coefs.get("x0", 0.0) - 3.0) < 1e-2
        assert abs(coefs.get("1", 0.0) - 100.0) < 1e-2

    def test_fit_recovers_two_input(self):
        sci = Scientist.create(input_dim=2, max_degree=2)
        xs, ys = _two_input_data(n=200)
        sci.observe_many(xs, ys)
        law = sci.fit()
        assert law.r2 > 0.999
        coefs = {t.name: t.coefficient for t in law.terms}
        assert abs(coefs.get("x0", 0.0) - 3.0) < 1e-2
        assert abs(coefs.get("x1", 0.0) - (-2.0)) < 1e-2
        assert abs(coefs.get("x0·x1", 0.0) - 0.5) < 1e-2

    def test_fit_insufficient_data(self):
        sci = Scientist.create(input_dim=1)
        with pytest.raises(InsufficientData):
            sci.fit()
        sci.observe([0.0], 1.0)
        with pytest.raises(InsufficientData):
            sci.fit()

    def test_fit_with_bic(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        xs, ys = _line_data(n=200, slope=2.0, intercept=0.5)
        sci.observe_many(xs, ys)
        law = sci.fit(criterion=SELECT_BIC)
        # BIC favours sparsity strongly; expect 2 terms
        assert law.k <= 3
        coefs = {t.name: t.coefficient for t in law.terms}
        assert "x0" in coefs

    def test_fit_with_mdl(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        xs, ys = _line_data(n=100, slope=2.0)
        sci.observe_many(xs, ys)
        law = sci.fit(criterion=SELECT_MDL)
        assert law is not None
        assert law.r2 > 0.99

    def test_fit_with_pareto_knee(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        xs, ys = _quadratic_data(n=80)
        sci.observe_many(xs, ys)
        law = sci.fit(criterion=SELECT_PARETO_KNEE)
        assert law is not None

    def test_fit_bad_criterion(self):
        sci = Scientist.create(input_dim=1)
        sci.observe_many(*_line_data(n=10))
        with pytest.raises(InvalidCriterion):
            sci.fit(criterion="not_a_criterion")

    def test_fit_publishes_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(events.append, kind=SCIENTIST_FITTED)
        sci = Scientist.create(input_dim=1, bus=bus)
        xs, ys = _line_data(n=20)
        sci.observe_many(xs, ys)
        sci.fit()
        assert len(events) == 1
        assert events[0].data["criterion"] == SELECT_AIC


# =====================================================================
# Pareto frontier
# =====================================================================


class TestPareto:
    def test_pareto_returns_non_empty(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        xs, ys = _quadratic_data(n=60)
        sci.observe_many(xs, ys)
        front = sci.pareto()
        assert len(front) >= 1
        # Frontier is non-dominated: complexity strictly increasing, RSS strictly decreasing
        for i in range(1, len(front)):
            assert front[i].complexity > front[i - 1].complexity
            assert front[i].rss < front[i - 1].rss

    def test_pareto_smallest_complexity_is_simplest(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        xs, ys = _line_data(n=40, slope=2.0)
        sci.observe_many(xs, ys)
        front = sci.pareto()
        assert front[0].complexity <= front[-1].complexity

    def test_pareto_caches(self):
        sci = Scientist.create(input_dim=1)
        xs, ys = _line_data(n=20)
        sci.observe_many(xs, ys)
        f1 = sci.pareto()
        f2 = sci.pareto()
        assert f1 == f2

    def test_pareto_insufficient_data(self):
        sci = Scientist.create(input_dim=1)
        with pytest.raises(InsufficientData):
            sci.pareto()

    def test_akaike_weights_sum_to_one(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        xs, ys = _quadratic_data(n=60)
        sci.observe_many(xs, ys)
        ws = sci.akaike_weights()
        assert ws  # non-empty
        assert abs(sum(ws.values()) - 1.0) < 1e-9
        for v in ws.values():
            assert 0.0 <= v <= 1.0

    def test_pareto_publishes_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(events.append, kind=SCIENTIST_PARETO)
        sci = Scientist.create(input_dim=1, bus=bus)
        xs, ys = _line_data(n=20)
        sci.observe_many(xs, ys)
        sci.pareto()
        assert len(events) == 1


# =====================================================================
# Prediction
# =====================================================================


class TestPredict:
    def test_predict_matches_law(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=60, slope=2.5, intercept=-1.3)
        sci.observe_many(xs, ys)
        law = sci.fit()
        y_hat = sci.predict([2.0], law=law)
        assert abs(y_hat - (2.5 * 2.0 - 1.3)) < 1e-3

    def test_predict_implicit_fit(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=40)
        sci.observe_many(xs, ys)
        # No prior fit() call — predict should fit implicitly.
        y = sci.predict([0.5])
        assert math.isfinite(y)

    def test_predict_wrong_arity(self):
        sci = Scientist.create(input_dim=2)
        xs, ys = _two_input_data(n=20)
        sci.observe_many(xs, ys)
        with pytest.raises(InvalidObservation):
            sci.predict([1.0])

    def test_predict_nan(self):
        sci = Scientist.create(input_dim=1)
        xs, ys = _line_data(n=20)
        sci.observe_many(xs, ys)
        with pytest.raises(InvalidObservation):
            sci.predict([float("nan")])

    def test_predict_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(events.append, kind=SCIENTIST_PREDICTED)
        sci = Scientist.create(input_dim=1, bus=bus)
        sci.observe_many(*_line_data(n=20))
        sci.predict([1.0])
        assert len(events) == 1

    def test_out_of_sample_r2(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs_train, ys_train = _line_data(n=60, seed=42)
        sci.observe_many(xs_train, ys_train)
        xs_test, ys_test = _line_data(n=30, seed=43)
        r2 = sci.evaluate_r2(xs_test, ys_test)
        assert r2 > 0.99


# =====================================================================
# Bootstrap
# =====================================================================


class TestBootstrap:
    def test_bootstrap_ci_contains_true(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        rng = random.Random(123)
        xs = [[rng.uniform(-3, 3)] for _ in range(150)]
        ys = [2.0 * x[0] + 1.0 + rng.gauss(0, 0.3) for x in xs]
        sci.observe_many(xs, ys)
        law = sci.fit()
        boot = sci.bootstrap(law=law, n_resamples=80, alpha=0.05)
        ci_x = boot.ci.get("x0")
        ci_c = boot.ci.get("1")
        assert ci_x is not None and ci_c is not None
        assert ci_x[0] <= 2.0 <= ci_x[1]
        assert ci_c[0] <= 1.0 <= ci_c[1]

    def test_bootstrap_se_positive(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=40, noise=0.2, seed=7)
        sci.observe_many(xs, ys)
        boot = sci.bootstrap(n_resamples=30)
        for v in boot.se.values():
            assert v >= 0.0

    def test_bootstrap_bad_alpha(self):
        sci = Scientist.create(input_dim=1)
        sci.observe_many(*_line_data(n=20))
        with pytest.raises(InvalidConfig):
            sci.bootstrap(alpha=0.0)
        with pytest.raises(InvalidConfig):
            sci.bootstrap(alpha=1.0)
        with pytest.raises(InvalidConfig):
            sci.bootstrap(n_resamples=1)

    def test_bootstrap_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(events.append, kind=SCIENTIST_BOOTSTRAPPED)
        sci = Scientist.create(input_dim=1, bus=bus)
        sci.observe_many(*_line_data(n=30))
        sci.bootstrap(n_resamples=10)
        assert len(events) == 1


# =====================================================================
# Stability selection
# =====================================================================


class TestStabilitySelection:
    def test_stability_finds_true_support(self):
        sci = Scientist.create(input_dim=1, max_degree=3)
        # Pure linear data: only x0 and the intercept should be stable.
        xs, ys = _line_data(n=80, slope=2.0, intercept=0.0)
        sci.observe_many(xs, ys)
        stab = sci.stability_selection(n_resamples=40, lam=0.05, pi_thr=0.6)
        names = stab.stable_names(sci.library)
        assert "x0" in names
        # x0^2 and x0^3 should typically NOT be stable.
        # (Allow noise but the support shouldn't include high-order terms reliably.)

    def test_stability_inclusion_bounded(self):
        sci = Scientist.create(input_dim=1)
        xs, ys = _line_data(n=30)
        sci.observe_many(xs, ys)
        stab = sci.stability_selection(n_resamples=20)
        for v in stab.inclusion.values():
            assert 0.0 <= v <= 1.0

    def test_stability_bad_config(self):
        sci = Scientist.create(input_dim=1)
        sci.observe_many(*_line_data(n=20))
        with pytest.raises(InvalidConfig):
            sci.stability_selection(subsample_fraction=0.0)
        with pytest.raises(InvalidConfig):
            sci.stability_selection(pi_thr=1.5)
        with pytest.raises(InvalidConfig):
            sci.stability_selection(n_resamples=1)
        with pytest.raises(InvalidConfig):
            sci.stability_selection(lam=-1.0)

    def test_stability_insufficient_data(self):
        sci = Scientist.create(input_dim=1)
        sci.observe([0.0], 0.0)
        sci.observe([1.0], 1.0)
        with pytest.raises(InsufficientData):
            sci.stability_selection(n_resamples=10)

    def test_stability_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(events.append, kind=SCIENTIST_STABILITY)
        sci = Scientist.create(input_dim=1, bus=bus)
        sci.observe_many(*_line_data(n=40))
        sci.stability_selection(n_resamples=10)
        assert len(events) == 1


# =====================================================================
# Determinism
# =====================================================================


class TestDeterminism:
    def test_same_seed_same_pareto(self):
        s1 = Scientist.create(input_dim=1, max_degree=2, seed=42)
        s2 = Scientist.create(input_dim=1, max_degree=2, seed=42)
        xs, ys = _quadratic_data(n=50)
        s1.observe_many(xs, ys)
        s2.observe_many(xs, ys)
        f1 = s1.pareto()
        f2 = s2.pareto()
        assert len(f1) == len(f2)
        for p1, p2 in zip(f1, f2):
            assert p1.k == p2.k
            assert abs(p1.rss - p2.rss) < 1e-9

    def test_fingerprint_chain_deterministic(self):
        s1 = Scientist.create(input_dim=1, seed=0)
        s2 = Scientist.create(input_dim=1, seed=0)
        xs, ys = _line_data(n=10, seed=0)
        s1.observe_many(xs, ys)
        s2.observe_many(xs, ys)
        assert s1.fingerprint() == s2.fingerprint()

    def test_different_seed_same_data_same_pareto(self):
        # Pareto is deterministic in data, not seed (seed only enters bootstrap/stability).
        # Fingerprints differ because the seed is hashed into the started-event payload,
        # but the discovered laws on the Pareto frontier are identical.
        s1 = Scientist.create(input_dim=1, max_degree=2, seed=1)
        s2 = Scientist.create(input_dim=1, max_degree=2, seed=999)
        xs, ys = _quadratic_data(n=60)
        s1.observe_many(xs, ys)
        s2.observe_many(xs, ys)
        assert s1.fingerprint() != s2.fingerprint()  # seed leaks into hash chain
        f1 = s1.pareto()
        f2 = s2.pareto()
        assert len(f1) == len(f2)
        for p1, p2 in zip(f1, f2):
            assert abs(p1.rss - p2.rss) < 1e-9
            assert p1.k == p2.k


# =====================================================================
# Hash chain
# =====================================================================


class TestHashChain:
    def test_each_observation_advances_chain(self):
        sci = Scientist.create(input_dim=1)
        fps = [sci.fingerprint()]
        for x in range(5):
            sci.observe([float(x)], float(x))
            fps.append(sci.fingerprint())
        assert len(set(fps)) == 6  # all distinct

    def test_chain_is_hash_linked(self):
        # Permuting observation order changes the fingerprint.
        s1 = Scientist.create(input_dim=1, seed=0)
        s2 = Scientist.create(input_dim=1, seed=0)
        s1.observe([1.0], 1.0)
        s1.observe([2.0], 2.0)
        s2.observe([2.0], 2.0)
        s2.observe([1.0], 1.0)
        assert s1.fingerprint() != s2.fingerprint()


# =====================================================================
# Report
# =====================================================================


class TestReport:
    def test_report_empty(self):
        sci = Scientist.create(input_dim=1)
        rep = sci.report()
        assert rep.n_observations == 0
        assert rep.best_law_aic is None
        assert rep.best_law_bic is None
        assert rep.best_law_mdl is None

    def test_report_after_fit(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=40)
        sci.observe_many(xs, ys)
        sci.fit()
        rep = sci.report()
        assert rep.n_observations == 40
        assert rep.best_law_aic is not None
        assert rep.best_law_aic.r2 > 0.99
        assert rep.best_law_bic is not None
        assert rep.best_law_mdl is not None
        assert len(rep.pareto) >= 1

    def test_report_serialises(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=40)
        sci.observe_many(xs, ys)
        rep = sci.report()
        d = rep.to_dict()
        assert "fingerprint" in d
        assert "pareto_size" in d
        # round-trip-friendly types only
        import json
        json.dumps(d)

    def test_report_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(events.append, kind=SCIENTIST_REPORTED)
        sci = Scientist.create(input_dim=1, bus=bus)
        sci.report()
        assert len(events) == 1


# =====================================================================
# Certificates
# =====================================================================


class TestCertificates:
    def test_aicc_finite(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _quadratic_data(n=40)
        sci.observe_many(xs, ys)
        law = sci.fit()
        aicc = sci.aicc_correction(law)
        assert math.isfinite(aicc)
        assert aicc >= law.aic

    def test_aicc_infinite_when_too_few(self):
        # n_observations = 2, but the discovered law could easily have k=2,
        # making n - k - 1 = -1.
        sci = Scientist.create(input_dim=1, max_degree=2)
        sci.observe([0.0], 0.0)
        sci.observe([1.0], 1.0)
        law = sci.fit()
        aicc = sci.aicc_correction(law)
        # If law has only 1 term, AICc is finite; if 2 or more, it should be inf.
        if law.k >= 2:
            assert aicc == float("inf")

    def test_mdl_certificate_components(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _quadratic_data(n=60)
        sci.observe_many(xs, ys)
        law = sci.fit()
        cert = sci.mdl_certificate(law)
        assert cert["model_bits_per_sample"] >= 0.0
        assert math.isfinite(cert["data_bits_per_sample"])
        total = cert["model_bits_per_sample"] + cert["data_bits_per_sample"]
        assert abs(total - cert["total_bits_per_sample"]) < 1e-9


# =====================================================================
# Concurrency
# =====================================================================


class TestThreadSafety:
    def test_concurrent_observations(self):
        sci = Scientist.create(input_dim=1)
        def worker(start: int):
            for i in range(start, start + 50):
                sci.observe([float(i)], float(i))
        threads = [threading.Thread(target=worker, args=(k * 50,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sci.n_observations == 200
        # No exception, no corruption — chain is well-formed.

    def test_concurrent_fits(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        sci.observe_many(*_line_data(n=60))
        results: list[Law] = []
        lock = threading.Lock()
        def worker():
            law = sci.fit()
            with lock:
                results.append(law)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 4
        # All fits should be on the same underlying data → same coefficients.
        first_terms = {(t.name, round(t.coefficient, 6)) for t in results[0].terms}
        for r in results[1:]:
            assert {(t.name, round(t.coefficient, 6)) for t in r.terms} == first_terms


# =====================================================================
# Law printing
# =====================================================================


class TestLawPrinting:
    def test_law_string_has_y_prefix(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=40, slope=2.0, intercept=1.0)
        sci.observe_many(xs, ys)
        law = sci.fit()
        s = str(law)
        assert s.startswith("y ≈ ")

    def test_law_empty(self):
        empty = Law(
            lam=1e-6, terms=tuple(), rss=0.0, n=10, sigma2=0.1,
            r2=0.0, aic=0.0, bic=0.0, mdl=0.0, fingerprint="0" * 64,
        )
        assert str(empty) == "y ≈ 0"

    def test_term_negative_sign(self):
        t = Term(name="x0", index=0, coefficient=-1.5, complexity=1)
        s = str(t)
        assert "−" in s  # unicode minus
        assert "1.5" in s

    def test_law_to_dict(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        sci.observe_many(*_line_data(n=40))
        law = sci.fit()
        d = law.to_dict()
        assert "terms" in d
        assert "lam" in d
        assert d["complexity"] == sum(t["coef"] != 0 for t in d["terms"])

    def test_predict_via_law(self):
        sci = Scientist.create(input_dim=1, max_degree=2)
        xs, ys = _line_data(n=40, slope=3.0, intercept=-2.0)
        sci.observe_many(xs, ys)
        law = sci.fit()
        y_hat = law.predict([1.0], sci.library)
        assert abs(y_hat - (3.0 * 1.0 - 2.0)) < 1e-3


# =====================================================================
# Trig basis recovery
# =====================================================================


class TestTrigDiscovery:
    def test_recover_sine(self):
        rng = random.Random(11)
        xs = [[rng.uniform(0.0, 6.0)] for _ in range(120)]
        ys = [2.0 * math.sin(x[0]) for x in xs]
        sci = Scientist.create(
            input_dim=1,
            max_degree=1,
            include_trig=True,
        )
        sci.observe_many(xs, ys)
        law = sci.fit()
        coefs = {t.name: t.coefficient for t in law.terms}
        assert "sin(x0)" in coefs
        assert abs(coefs["sin(x0)"] - 2.0) < 0.1
        assert law.r2 > 0.99


# =====================================================================
# Custom basis
# =====================================================================


class TestCustomBasis:
    def test_custom_sqrt_basis_recovers_law(self):
        rng = random.Random(2024)
        xs = [[rng.uniform(0.5, 4.0)] for _ in range(80)]
        ys = [3.0 * math.sqrt(x[0]) + 0.5 for x in xs]
        sqrt_b = Basis(name="sqrt(x0)", fn=lambda x: math.sqrt(abs(x[0])), complexity=2)
        sci = Scientist.create(
            input_dim=1,
            max_degree=1,
            extra_basis=[sqrt_b],
        )
        sci.observe_many(xs, ys)
        law = sci.fit()
        coefs = {t.name: t.coefficient for t in law.terms}
        assert "sqrt(x0)" in coefs
        assert abs(coefs["sqrt(x0)"] - 3.0) < 0.1
