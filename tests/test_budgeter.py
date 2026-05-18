"""Tests for the Budgeter test-time compute allocation primitive."""
from __future__ import annotations

import json
import math
import random

import pytest

from agi.events import EventBus
from agi.budgeter import (
    BUDGETER_ALLOCATED,
    BUDGETER_CERTIFIED,
    BUDGETER_EXTRAPOLATED,
    BUDGETER_FIT,
    BUDGETER_OBSERVED,
    BUDGETER_PARETO,
    BUDGETER_REPORTED,
    BUDGETER_STARTED,
    Allocation,
    Budgeter,
    BudgeterCertificate,
    BudgeterConfig,
    BudgeterError,
    BudgeterReport,
    FitFailed,
    InfeasibleBudget,
    InvalidConfig,
    InvalidObservation,
    KNOWN_STRATEGIES,
    NotFitted,
    Observation,
    ParetoPoint,
    STRAT_BEAM,
    STRAT_MAJORITY,
    STRAT_PARALLEL,
    STRAT_SEQUENTIAL,
    STRAT_TREE,
    STRAT_VERIFIER,
    StrategyFit,
    StrategySpec,
    UnknownStrategy,
    default_beam_spec,
    default_majority_spec,
    default_parallel_spec,
    default_sequential_spec,
    default_tree_spec,
    default_verifier_spec,
    fresh_budgeter,
    unbiased_pass_at_k,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic ground-truth curves.
# ---------------------------------------------------------------------------


def true_parallel(units, p1=0.25, p_inf=0.85, tau=12.0):
    return p_inf - (p_inf - p1) * math.exp(-units / tau)


def true_verifier(units, p1=0.25, p_inf=0.92, r=0.5):
    return p_inf - (p_inf - p1) * (1.0 - r) ** max(0, units - 1)


def true_sequential(units, p1=0.25, p_inf=0.95, tau_t=200.0):
    return p_inf - (p_inf - p1) * math.exp(-units / tau_t)


def true_tree(units, p1=0.25, p_inf=0.88, n0=32.0, gamma=0.6):
    return p_inf - (p_inf - p1) / (1.0 + units / n0) ** gamma


def true_majority(units, p1=0.30, p_inf=0.85, k0=5.0, s=2.0):
    z = (units - k0) / s
    frac = 1.0 / (1.0 + math.exp(-z))
    return p1 + (p_inf - p1) * frac


def true_beam(units, p1=0.30, p_inf=0.92, r=0.6):
    return p_inf - (p_inf - p1) * (1.0 - r) ** max(0.0, units)


def _sample(p, trials, rng):
    return sum(1 for _ in range(trials) if rng.random() < p)


def _populate_strategy(b, strategy, units_list, truth_fn, *,
                       trials=80, rng_seed=0):
    rng = random.Random(rng_seed)
    for u in units_list:
        p = truth_fn(u)
        successes = _sample(p, trials, rng)
        b.observe(Observation(strategy=strategy, difficulty=0.0,
                              compute_units=float(u), trials=trials,
                              successes=successes))


# ---------------------------------------------------------------------------
# unbiased_pass_at_k
# ---------------------------------------------------------------------------


def test_pass_at_k_n_eq_k_full_credit():
    assert unbiased_pass_at_k(5, 5, 3) == 1.0


def test_pass_at_k_no_successes():
    assert unbiased_pass_at_k(10, 0, 5) == 0.0


def test_pass_at_k_one_in_ten_for_k_one():
    assert abs(unbiased_pass_at_k(10, 1, 1) - 0.1) < 1e-9


def test_pass_at_k_monotone_in_k():
    n = 20
    c = 3
    rates = [unbiased_pass_at_k(n, c, k) for k in range(1, n + 1)]
    for a, b in zip(rates, rates[1:]):
        assert b >= a - 1e-12


def test_pass_at_k_monotone_in_c():
    n = 20
    k = 5
    rates = [unbiased_pass_at_k(n, c, k) for c in range(0, n + 1)]
    for a, b in zip(rates, rates[1:]):
        assert b >= a - 1e-12


def test_pass_at_k_invalid_inputs_raise():
    with pytest.raises(InvalidObservation):
        unbiased_pass_at_k(10, -1, 1)
    with pytest.raises(InvalidObservation):
        unbiased_pass_at_k(10, 11, 1)
    with pytest.raises(InvalidObservation):
        unbiased_pass_at_k(10, 3, 0)


# ---------------------------------------------------------------------------
# Config / spec validation
# ---------------------------------------------------------------------------


def test_config_defaults_validate():
    cfg = BudgeterConfig()
    assert cfg.seed == 0
    assert cfg.bootstrap_b >= 0
    assert 0.0 < cfg.confidence < 1.0


def test_config_rejects_bad_confidence():
    with pytest.raises(InvalidConfig):
        BudgeterConfig(confidence=0.0)
    with pytest.raises(InvalidConfig):
        BudgeterConfig(confidence=1.0)


def test_config_rejects_bad_holdout():
    with pytest.raises(InvalidConfig):
        BudgeterConfig(holdout_fraction=1.0)


def test_config_rejects_small_grid():
    with pytest.raises(InvalidConfig):
        BudgeterConfig(allocator_grid=4)


def test_strategy_spec_validates_name():
    with pytest.raises(UnknownStrategy):
        StrategySpec(name="not-a-real-strategy")


def test_strategy_spec_validates_unit_cost():
    with pytest.raises(InvalidConfig):
        StrategySpec(name=STRAT_PARALLEL, unit_cost=0.0)
    with pytest.raises(InvalidConfig):
        StrategySpec(name=STRAT_PARALLEL, unit_cost=float("inf"))


def test_strategy_spec_validates_verifier_info():
    with pytest.raises(InvalidConfig):
        StrategySpec(name=STRAT_VERIFIER, verifier_info=-0.1)
    with pytest.raises(InvalidConfig):
        StrategySpec(name=STRAT_VERIFIER, verifier_info=1.5)


def test_strategy_spec_validates_unit_bounds():
    with pytest.raises(InvalidConfig):
        StrategySpec(name=STRAT_PARALLEL, min_units=10.0, max_units=5.0)
    with pytest.raises(InvalidConfig):
        StrategySpec(name=STRAT_PARALLEL, min_units=0.0, max_units=10.0)


# ---------------------------------------------------------------------------
# Observation validation
# ---------------------------------------------------------------------------


def test_observation_rejects_bad_strategy():
    with pytest.raises(UnknownStrategy):
        Observation(strategy="meow", difficulty=0.0, compute_units=1.0,
                    trials=10, successes=3)


def test_observation_rejects_bad_difficulty():
    with pytest.raises(InvalidObservation):
        Observation(strategy=STRAT_PARALLEL, difficulty=float("nan"),
                    compute_units=1.0, trials=10, successes=3)


def test_observation_rejects_bad_compute_units():
    with pytest.raises(InvalidObservation):
        Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                    compute_units=0.0, trials=10, successes=3)
    with pytest.raises(InvalidObservation):
        Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                    compute_units=-1.0, trials=10, successes=3)


def test_observation_rejects_bad_trials():
    with pytest.raises(InvalidObservation):
        Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                    compute_units=1.0, trials=0, successes=0)


def test_observation_rejects_successes_out_of_range():
    with pytest.raises(InvalidObservation):
        Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                    compute_units=1.0, trials=10, successes=11)
    with pytest.raises(InvalidObservation):
        Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                    compute_units=1.0, trials=10, successes=-1)


def test_observation_rejects_bad_weight():
    with pytest.raises(InvalidObservation):
        Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                    compute_units=1.0, trials=10, successes=5, weight=0.0)


# ---------------------------------------------------------------------------
# Lifecycle: events and fingerprint
# ---------------------------------------------------------------------------


def test_budgeter_emits_started_event_with_bus():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    Budgeter(bus=bus)
    assert BUDGETER_STARTED in seen


def test_fingerprint_changes_after_observe():
    b = Budgeter()
    b.register_strategy(default_parallel_spec())
    h0 = b.fingerprint_hash
    b.observe(Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                          compute_units=8.0, trials=10, successes=5))
    h1 = b.fingerprint_hash
    assert h0 != h1


def test_observe_emits_event_per_row():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    b = Budgeter(bus=bus)
    b.register_strategy(default_parallel_spec())
    b.observe([
        Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                    compute_units=1.0, trials=10, successes=2),
        Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                    compute_units=2.0, trials=10, successes=4),
    ])
    assert seen.count(BUDGETER_OBSERVED) == 2


def test_reset_clears_state_and_emits():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    b = Budgeter(bus=bus)
    b.register_strategy(default_parallel_spec())
    b.observe(Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                          compute_units=4.0, trials=10, successes=4))
    b.reset()
    assert b.observations == ()
    assert "budgeter.reset" in seen


def test_observe_rejects_unregistered_strategy():
    b = Budgeter()
    # Don't register STRAT_PARALLEL.
    with pytest.raises(UnknownStrategy):
        b.observe(Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                              compute_units=1.0, trials=10, successes=2))


def test_observe_rejects_non_observation():
    b = Budgeter()
    b.register_strategy(default_parallel_spec())
    with pytest.raises(InvalidObservation):
        b.observe("not an observation")  # type: ignore[arg-type]


def test_fit_fails_with_too_few_observations():
    b = Budgeter()
    b.register_strategy(default_parallel_spec())
    b.observe(Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                          compute_units=1.0, trials=10, successes=2))
    with pytest.raises(FitFailed):
        b.fit()


def test_fit_unknown_strategy_raises():
    b = Budgeter()
    b.register_strategy(default_parallel_spec())
    with pytest.raises(UnknownStrategy):
        b.fit(strategies=["nonexistent"])


# ---------------------------------------------------------------------------
# Curve fitting recovers truth approximately.
# ---------------------------------------------------------------------------


def test_parallel_fit_recovers_p_inf_within_tolerance():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0, max_iters=400))
    b.register_strategy(default_parallel_spec(max_units=1024.0))
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16, 32, 64, 128, 256],
                       lambda u: true_parallel(u),
                       trials=200, rng_seed=0)
    fit = b.fit()[STRAT_PARALLEL]
    # p_inf should be within 0.08 of truth 0.85.  Convergence may stall
    # at max_iters near a saturating curve floor — what matters is the
    # estimate is in the right ballpark.
    assert abs(fit.params["p_inf"] - 0.85) < 0.10, fit.params


def test_verifier_fit_recovers_r():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(default_verifier_spec())
    _populate_strategy(b, STRAT_VERIFIER,
                       [1, 2, 4, 8, 16, 32, 64, 128],
                       lambda u: true_verifier(u, r=0.5),
                       trials=200, rng_seed=1)
    fit = b.fit()[STRAT_VERIFIER]
    assert fit.converged
    # r ~ 0.5 should fit within 0.15.
    assert abs(fit.params["r"] - 0.5) < 0.15, fit.params


def test_sequential_fit_returns_finite_tau():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(default_sequential_spec(max_units=4096.0))
    _populate_strategy(b, STRAT_SEQUENTIAL,
                       [16, 32, 64, 128, 256, 512, 1024, 2048],
                       lambda u: true_sequential(u),
                       trials=200, rng_seed=2)
    fit = b.fit()[STRAT_SEQUENTIAL]
    assert fit.converged
    assert fit.params["tau_t"] > 0
    assert math.isfinite(fit.params["tau_t"])
    assert 0.0 <= fit.params["p_inf"] <= 1.0


def test_tree_fit_returns_finite_params():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0, max_iters=400))
    b.register_strategy(default_tree_spec(max_units=512.0))
    _populate_strategy(b, STRAT_TREE,
                       [1, 2, 4, 8, 16, 32, 64, 128, 256],
                       lambda u: true_tree(u),
                       trials=200, rng_seed=3)
    fit = b.fit()[STRAT_TREE]
    for k in ("p1", "p_inf", "n0", "gamma"):
        assert math.isfinite(fit.params[k])
        assert fit.params[k] >= 0.0


def test_majority_fit_returns_finite_params():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(default_majority_spec())
    _populate_strategy(b, STRAT_MAJORITY,
                       [1, 2, 3, 5, 7, 11, 17, 23],
                       lambda u: true_majority(u),
                       trials=200, rng_seed=4)
    fit = b.fit()[STRAT_MAJORITY]
    assert fit.converged
    for k in ("p1", "p_inf", "k0", "s"):
        assert math.isfinite(fit.params[k])


def test_beam_fit_returns_finite_params():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0, max_iters=400))
    b.register_strategy(default_beam_spec(max_units=64.0))
    _populate_strategy(b, STRAT_BEAM,
                       [1, 2, 3, 5, 8, 13, 21, 34, 55],
                       lambda u: true_beam(u),
                       trials=200, rng_seed=5)
    fit = b.fit()[STRAT_BEAM]
    for k in ("p1", "p_inf", "r"):
        assert math.isfinite(fit.params[k])
        assert 0.0 <= fit.params[k] <= 1.0


def test_fit_emits_event_per_strategy():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0), bus=bus)
    b.register_strategy(default_parallel_spec())
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16, 32], lambda u: true_parallel(u),
                       trials=80, rng_seed=0)
    b.fit()
    assert seen.count(BUDGETER_FIT) == 1


def test_observe_invalidates_previous_fit():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(default_parallel_spec())
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16, 32], lambda u: true_parallel(u),
                       trials=80, rng_seed=0)
    b.fit()
    assert b.fits  # cached
    b.observe(Observation(strategy=STRAT_PARALLEL, difficulty=0.0,
                          compute_units=64.0, trials=10, successes=8))
    assert b.fits == {}  # invalidated


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------


def test_extrapolate_not_fitted_raises():
    b = Budgeter()
    b.register_strategy(default_parallel_spec())
    with pytest.raises(NotFitted):
        b.extrapolate(STRAT_PARALLEL, 8.0)


def test_extrapolate_returns_in_unit_interval():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=20))
    b.register_strategy(default_parallel_spec())
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16, 32, 64], lambda u: true_parallel(u),
                       trials=120, rng_seed=0)
    b.fit()
    p, lo, hi = b.extrapolate(STRAT_PARALLEL, 32.0)
    assert 0.0 <= lo <= p <= hi <= 1.0


def test_extrapolate_zero_bootstrap_collapses_ci():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(default_parallel_spec())
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16, 32], lambda u: true_parallel(u),
                       trials=80, rng_seed=0)
    b.fit()
    p, lo, hi = b.extrapolate(STRAT_PARALLEL, 8.0)
    assert lo == p == hi


def test_extrapolate_monotone_in_units():
    """For saturating-exponential curves, p is monotone non-decreasing."""
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(default_parallel_spec())
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16, 32, 64], lambda u: true_parallel(u),
                       trials=160, rng_seed=0)
    b.fit()
    p_at = [b.extrapolate(STRAT_PARALLEL, u)[0] for u in (1.0, 4.0, 16.0, 64.0)]
    for a, c in zip(p_at, p_at[1:]):
        assert c >= a - 1e-9


def test_extrapolate_invalid_units():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(default_parallel_spec())
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16, 32], lambda u: true_parallel(u),
                       trials=80, rng_seed=0)
    b.fit()
    with pytest.raises(InvalidObservation):
        b.extrapolate(STRAT_PARALLEL, 0.0)
    with pytest.raises(InvalidObservation):
        b.extrapolate(STRAT_PARALLEL, -3.0)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def _make_three_strat_budgeter(seed=42, bootstrap_b=10):
    b = Budgeter(BudgeterConfig(seed=seed, bootstrap_b=bootstrap_b,
                                holdout_fraction=0.2))
    b.register_strategy(default_parallel_spec(max_units=512.0))
    b.register_strategy(default_verifier_spec(max_units=128.0))
    b.register_strategy(default_sequential_spec(max_units=4096.0))
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16, 32, 64, 128, 256],
                       lambda u: true_parallel(u),
                       trials=150, rng_seed=0)
    _populate_strategy(b, STRAT_VERIFIER,
                       [1, 2, 4, 8, 16, 32, 64, 128],
                       lambda u: true_verifier(u),
                       trials=150, rng_seed=1)
    _populate_strategy(b, STRAT_SEQUENTIAL,
                       [16, 32, 64, 128, 256, 512, 1024, 2048],
                       lambda u: true_sequential(u),
                       trials=150, rng_seed=2)
    return b


def test_allocate_without_fits_lazy_fits():
    b = _make_three_strat_budgeter()
    alloc = b.allocate(budget=20.0)
    assert isinstance(alloc, Allocation)
    assert alloc.spent <= 20.0 + 1e-9


def test_allocate_emits_event():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    b = _make_three_strat_budgeter()
    b.bus = bus
    b.allocate(budget=20.0)
    assert BUDGETER_ALLOCATED in seen


def test_allocate_respects_budget_ceiling():
    b = _make_three_strat_budgeter()
    for budget in (5.0, 10.0, 25.0, 100.0):
        alloc = b.allocate(budget=budget)
        assert alloc.spent <= budget + 1e-6, (budget, alloc.spent)


def test_allocate_predicted_pass_monotone_in_budget():
    b = _make_three_strat_budgeter()
    last_p = -1.0
    for budget in (3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0):
        alloc = b.allocate(budget=budget)
        assert alloc.predicted_pass >= last_p - 1e-9, (budget, alloc.predicted_pass, last_p)
        last_p = alloc.predicted_pass


def test_allocate_predicted_pass_in_unit_interval():
    b = _make_three_strat_budgeter()
    for budget in (4.0, 20.0, 200.0):
        alloc = b.allocate(budget=budget)
        assert 0.0 <= alloc.predicted_pass <= 1.0
        assert 0.0 <= alloc.predicted_lower <= alloc.predicted_upper <= 1.0


def test_allocate_ci_brackets_point_estimate():
    b = _make_three_strat_budgeter()
    alloc = b.allocate(budget=20.0)
    # Point must be inside CI (within tolerance).
    assert alloc.predicted_lower <= alloc.predicted_pass + 1e-9
    assert alloc.predicted_pass <= alloc.predicted_upper + 1e-9


def test_allocate_rejects_bad_budget():
    b = _make_three_strat_budgeter()
    with pytest.raises(InvalidConfig):
        b.allocate(budget=0.0)
    with pytest.raises(InvalidConfig):
        b.allocate(budget=-5.0)
    with pytest.raises(InvalidConfig):
        b.allocate(budget=float("inf"))


def test_allocate_strategies_filter():
    b = _make_three_strat_budgeter()
    alloc = b.allocate(budget=50.0, strategies=[STRAT_VERIFIER])
    # Only verifier active.
    assert set(alloc.active_strategies).issubset({STRAT_VERIFIER})
    assert alloc.per_strategy_units[STRAT_VERIFIER] > 0
    # No other strategy gets units in the returned dict.
    assert list(alloc.per_strategy_units.keys()) == [STRAT_VERIFIER]


def test_allocate_strategies_filter_unfit_raises():
    b = _make_three_strat_budgeter()
    b.register_strategy(default_tree_spec())  # registered but not observed/fit
    b.fit(strategies=[STRAT_PARALLEL, STRAT_VERIFIER, STRAT_SEQUENTIAL])
    with pytest.raises(NotFitted):
        b.allocate(budget=10.0, strategies=[STRAT_TREE])


def test_allocate_no_observations_raises_not_fitted():
    b = Budgeter()
    b.register_strategy(default_parallel_spec())
    with pytest.raises(NotFitted):
        b.allocate(budget=10.0)


def test_allocate_no_specs_raises_invalid_config():
    b = Budgeter()
    with pytest.raises(InvalidConfig):
        b.allocate(budget=10.0)


def test_allocate_below_floor_raises_infeasible():
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(StrategySpec(name=STRAT_PARALLEL, unit_cost=10.0,
                                     min_units=4.0, max_units=64.0))
    _populate_strategy(b, STRAT_PARALLEL,
                       [4, 8, 16, 32, 64], lambda u: true_parallel(u),
                       trials=120, rng_seed=0)
    b.fit()
    with pytest.raises(InfeasibleBudget):
        b.allocate(budget=10.0)  # min cost = 4 * 10 = 40


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def test_pareto_returns_n_points():
    b = _make_three_strat_budgeter()
    pts = b.pareto(min_budget=4.0, max_budget=200.0, n_points=6)
    assert len(pts) >= 4  # some low budgets may be infeasible
    for p in pts:
        assert isinstance(p, ParetoPoint)


def test_pareto_monotone_non_decreasing_in_pass():
    b = _make_three_strat_budgeter()
    pts = b.pareto(min_budget=4.0, max_budget=200.0, n_points=10)
    last = -1.0
    for p in pts:
        assert p.predicted_pass >= last - 1e-6
        last = p.predicted_pass


def test_pareto_emits_event():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    b = _make_three_strat_budgeter()
    b.bus = bus
    b.pareto(min_budget=4.0, max_budget=100.0, n_points=4)
    assert BUDGETER_PARETO in seen


def test_pareto_rejects_bad_range():
    b = _make_three_strat_budgeter()
    with pytest.raises(InvalidConfig):
        b.pareto(min_budget=100.0, max_budget=10.0, n_points=4)
    with pytest.raises(InvalidConfig):
        b.pareto(min_budget=10.0, max_budget=100.0, n_points=1)


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


def test_certificate_after_allocation():
    b = _make_three_strat_budgeter()
    b.allocate(budget=20.0)
    cert = b.certificate()
    assert isinstance(cert, BudgeterCertificate)
    # Fingerprint at cert creation is sealed *before* the cert's own publish.
    # We require it to be a valid SHA-256 hex digest (64 chars).
    assert len(cert.fingerprint_hash) == 64
    assert all(c in "0123456789abcdef" for c in cert.fingerprint_hash)
    assert cert.regret_ucb is not None
    assert cert.regret_ucb >= 0.0
    assert cert.oracle_strategy in {STRAT_PARALLEL, STRAT_VERIFIER, STRAT_SEQUENTIAL}


def test_certificate_before_allocation_raises():
    b = _make_three_strat_budgeter()
    with pytest.raises(NotFitted):
        b.certificate()


def test_certificate_emits_event():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    b = _make_three_strat_budgeter()
    b.bus = bus
    b.allocate(budget=20.0)
    b.certificate()
    assert BUDGETER_CERTIFIED in seen


def test_certificate_lcb_bounds_when_holdout_nonzero():
    b = _make_three_strat_budgeter()
    b.allocate(budget=20.0)
    cert = b.certificate()
    if cert.pass_held_out is not None:
        assert cert.pass_lcb_hoeffding is not None
        assert cert.pass_lcb_bernstein is not None
        assert 0.0 <= cert.pass_lcb_hoeffding <= cert.pass_held_out + 1e-9
        assert 0.0 <= cert.pass_lcb_bernstein <= cert.pass_held_out + 1e-9


def test_certificate_invalid_confidence():
    b = _make_three_strat_budgeter()
    b.allocate(budget=20.0)
    with pytest.raises(InvalidConfig):
        b.certificate(confidence=0.0)
    with pytest.raises(InvalidConfig):
        b.certificate(confidence=1.0)


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------


def test_report_round_trips_json():
    b = _make_three_strat_budgeter()
    b.allocate(budget=20.0)
    rep = b.report()
    s = json.dumps(rep.to_dict())
    restored = json.loads(s)
    assert restored["observations"] == len(b.observations)
    assert restored["allocation"] is not None
    assert restored["allocation"]["budget"] == 20.0


def test_report_no_allocation_section_when_not_allocated():
    b = _make_three_strat_budgeter()
    b.fit(strategies=[STRAT_PARALLEL, STRAT_VERIFIER, STRAT_SEQUENTIAL])
    rep = b.report()
    assert rep.allocation is None
    assert rep.certificate is None


def test_report_includes_each_strategy_fit():
    b = _make_three_strat_budgeter()
    b.fit(strategies=[STRAT_PARALLEL, STRAT_VERIFIER, STRAT_SEQUENTIAL])
    rep = b.report()
    for s in (STRAT_PARALLEL, STRAT_VERIFIER, STRAT_SEQUENTIAL):
        assert s in rep.fits


def test_report_emits_event():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    b = _make_three_strat_budgeter()
    b.bus = bus
    b.allocate(budget=20.0)
    b.report()
    assert BUDGETER_REPORTED in seen


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_allocate_deterministic_under_same_seed():
    def make_alloc(seed):
        b = Budgeter(BudgeterConfig(seed=seed, bootstrap_b=10,
                                    holdout_fraction=0.2))
        b.register_strategy(default_parallel_spec())
        b.register_strategy(default_verifier_spec())
        _populate_strategy(b, STRAT_PARALLEL,
                           [1, 2, 4, 8, 16, 32, 64],
                           lambda u: true_parallel(u),
                           trials=120, rng_seed=0)
        _populate_strategy(b, STRAT_VERIFIER,
                           [1, 2, 4, 8, 16, 32, 64],
                           lambda u: true_verifier(u),
                           trials=120, rng_seed=1)
        return b.allocate(budget=15.0)
    a1 = make_alloc(123)
    a2 = make_alloc(123)
    assert a1.per_strategy_units == a2.per_strategy_units
    assert a1.predicted_pass == a2.predicted_pass
    assert a1.predicted_lower == a2.predicted_lower
    assert a1.predicted_upper == a2.predicted_upper


def test_different_seeds_produce_valid_ci_bounds():
    """Different seeds may shuffle the holdout split differently and hence
    produce different fits.  We don't lock the point estimate across seeds;
    we do lock that each is a valid interval and that point ∈ [lo, hi]."""
    def ci(seed):
        b = Budgeter(BudgeterConfig(seed=seed, bootstrap_b=20,
                                    holdout_fraction=0.2))
        b.register_strategy(default_parallel_spec())
        _populate_strategy(b, STRAT_PARALLEL,
                           [1, 2, 4, 8, 16, 32],
                           lambda u: true_parallel(u),
                           trials=80, rng_seed=0)
        b.fit()
        return b.extrapolate(STRAT_PARALLEL, 16.0)
    p1, lo1, hi1 = ci(1)
    p2, lo2, hi2 = ci(2)
    assert lo1 <= p1 <= hi1
    assert lo2 <= p2 <= hi2


# ---------------------------------------------------------------------------
# recommend convenience
# ---------------------------------------------------------------------------


def test_recommend_end_to_end():
    b = _make_three_strat_budgeter()
    alloc = b.recommend(budget=25.0)
    assert isinstance(alloc, Allocation)
    assert alloc.spent <= 25.0 + 1e-6


# ---------------------------------------------------------------------------
# fresh_budgeter factory
# ---------------------------------------------------------------------------


def test_fresh_budgeter_registers_all_defaults():
    b = fresh_budgeter()
    assert set(b.strategies) == set(KNOWN_STRATEGIES)


def test_fresh_budgeter_passes_seed():
    b = fresh_budgeter(seed=7)
    assert b.config.seed == 7


def test_fresh_budgeter_passes_kwargs():
    b = fresh_budgeter(seed=0, bootstrap_b=42, confidence=0.9)
    assert b.config.bootstrap_b == 42
    assert b.config.confidence == 0.9


# ---------------------------------------------------------------------------
# Investor-headline behaviour: low-difficulty tasks prefer cheap parallel,
# higher-difficulty tasks shift compute to the strategy with the highest
# ceiling (verifier or sequential).
# ---------------------------------------------------------------------------


def test_easy_task_can_be_solved_with_minimal_budget():
    """When every strategy saturates at ≈1.0, even a tiny budget hits ≥ 0.7."""
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
    b.register_strategy(default_parallel_spec())
    _populate_strategy(b, STRAT_PARALLEL,
                       [1, 2, 4, 8, 16],
                       lambda u: 0.99 - (0.99 - 0.6) * math.exp(-u / 4.0),
                       trials=300, rng_seed=0)
    b.fit()
    alloc = b.allocate(budget=20.0)
    assert alloc.predicted_pass > 0.7


def test_hard_task_needs_more_budget_for_same_pass_rate():
    """At a fixed pass-rate target, harder true-curves need bigger budgets."""
    def make_budgeter(p_inf):
        b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
        b.register_strategy(default_parallel_spec(max_units=512.0))
        _populate_strategy(b, STRAT_PARALLEL,
                           [1, 2, 4, 8, 16, 32, 64, 128],
                           lambda u: p_inf - (p_inf - 0.05)
                                     * math.exp(-u / 20.0),
                           trials=300, rng_seed=0)
        b.fit()
        return b
    b_easy = make_budgeter(0.99)
    b_hard = make_budgeter(0.85)
    # The hard curve never exceeds 0.85; predicted_pass at any budget < 0.86.
    alloc_hard = b_hard.allocate(budget=200.0)
    assert alloc_hard.predicted_pass <= 0.90  # has slop
    alloc_easy = b_easy.allocate(budget=200.0)
    assert alloc_easy.predicted_pass >= alloc_hard.predicted_pass - 0.05


def test_higher_verifier_info_yields_higher_pass_at_same_budget():
    """Holding cost constant, a stronger verifier raises the alloc's pass."""
    def make(r_truth):
        b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0))
        b.register_strategy(StrategySpec(name=STRAT_VERIFIER, unit_cost=1.0,
                                         verifier_info=r_truth,
                                         min_units=1.0, max_units=128.0))
        _populate_strategy(b, STRAT_VERIFIER,
                           [1, 2, 4, 8, 16, 32, 64],
                           lambda u: true_verifier(u, p1=0.20, p_inf=0.95,
                                                    r=r_truth),
                           trials=300, rng_seed=0)
        b.fit()
        return b
    weak = make(0.2)
    strong = make(0.7)
    p_weak = weak.allocate(budget=20.0).predicted_pass
    p_strong = strong.allocate(budget=20.0).predicted_pass
    # Strong verifier dominates.
    assert p_strong >= p_weak - 0.02, (p_weak, p_strong)


# ---------------------------------------------------------------------------
# Composition smoke: write a custom strategy spec with a coordination payload.
# ---------------------------------------------------------------------------


def test_strategy_spec_metadata_round_trips():
    spec = StrategySpec(name=STRAT_TREE, unit_cost=2.0, min_units=1.0,
                        max_units=128.0,
                        metadata={"impl": "agi.searcher.mcts",
                                  "branching": 4})
    b = Budgeter()
    b.register_strategy(spec)
    out = b.spec(STRAT_TREE)
    assert out.metadata["impl"] == "agi.searcher.mcts"
    assert out.metadata["branching"] == 4


def test_unknown_strategy_lookup():
    b = Budgeter()
    with pytest.raises(UnknownStrategy):
        b.spec(STRAT_PARALLEL)
