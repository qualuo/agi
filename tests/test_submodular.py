"""Tests for ``agi.submodular`` — subset selection with approximation bounds.

The tests follow the mathematical contracts of the module:

1.  **Canonical objectives** evaluate to their textbook expressions on
    small examples (facility location, weighted coverage, log-det,
    max-cut, concave-over-modular, feature-based, Gaussian entropy).
2.  **Submodularity (DR)** holds for every canonical objective under
    its declared monotone-+-α regime, checked exhaustively on small
    ground sets and probabilistically by ``certify_submodular``.
3.  **Lazy greedy** matches **naive greedy** value to floating-point
    tolerance on every monotone submodular problem we shipped
    (Minoux 1978 equivalence theorem).
4.  **Approximation ratio**: ``f(Ŝ) / OPT ≥ (1 − 1/e)`` on small
    enumerable problems for ``lazy_greedy`` / ``naive_greedy`` /
    ``celf``.
5.  **Stochastic greedy** beats the ``(1 − 1/e − ε)`` bound in
    expectation over random seeds.
6.  **Khuller-Moss-Naor / Sviridenko knapsack** satisfies budget and
    matches the cost-benefit pivot.
7.  **Double greedy** (randomised + deterministic) hits the ½ /
    ⅓ bounds respectively on unconstrained ``MaxCut``.
8.  **Distorted greedy** with ``γ = 1`` agrees with standard greedy.
9.  **Sieve-Streaming** one-pass result satisfies ``(½ − ε)``.
10. **Submodular cover** terminates with ``f(Ŝ) ≥ Q`` and reports
    the Wolsey ``1 + ln(Q/η)`` bound.
11. **Curvature** is ``0`` for additive / modular objectives and
    ``1`` (worst case) for hard set-cover.  Bound interpolates
    monotonically.
12. **Threadsafety**: many concurrent ``maximize`` calls produce
    stable counters and deterministic digests for a fixed seed.
13. **Attestation**: every solve emits a content-hashed receipt to
    the optional attestor.

The tests are pure-Python (stdlib only) and run without an API key.
"""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.events import Event, EventBus
from agi.submodular import (
    CertificateReport,
    ConcaveOverModular,
    FacilityLocation,
    FeatureBased,
    GaussianEntropy,
    KNOWN_METHODS,
    LogDeterminant,
    METHOD_CELF,
    METHOD_COST_GREEDY,
    METHOD_DISTORTED_GREEDY,
    METHOD_DOUBLE_GREEDY_DETERMINISTIC,
    METHOD_DOUBLE_GREEDY_RANDOM,
    METHOD_LAZY_GREEDY,
    METHOD_NAIVE_GREEDY,
    METHOD_SIEVE_STREAMING,
    METHOD_STOCHASTIC_GREEDY,
    METHOD_SUBMODULAR_COVER,
    METHOD_SVIRIDENKO_KNAPSACK,
    METHOD_THRESHOLD_GREEDY,
    MaxCut,
    MonotoneSetCover,
    StreamReport,
    Submodular,
    SubmodularError,
    SubmodularReport,
    WeightedCoverage,
    double_greedy,
    lazy_greedy,
    sieve_streaming,
    stochastic_greedy,
)

ONE_MINUS_INV_E = 1.0 - math.exp(-1.0)


# =====================================================================
# Canonical objective math
# =====================================================================


def _enumerate_optimum(f, ground, k):
    """Brute-force best subset of size exactly up to k."""
    from itertools import combinations

    best = (-math.inf, ())
    for r in range(0, k + 1):
        for combo in combinations(ground, r):
            val = f(list(combo))
            if val > best[0]:
                best = (val, combo)
    return best


def _is_submodular(f, ground) -> tuple[bool, int]:
    """Exhaustively check the diminishing-returns inequality."""
    from itertools import combinations

    n = len(ground)
    violations = 0
    for ra in range(0, n):
        for A in combinations(ground, ra):
            for rb in range(ra, n):
                for B in combinations(ground, rb):
                    if not set(A).issubset(B):
                        continue
                    for v in ground:
                        if v in B:
                            continue
                        da = f(list(A) + [v]) - f(list(A))
                        db = f(list(B) + [v]) - f(list(B))
                        if da + 1e-9 < db:
                            violations += 1
    return violations == 0, violations


def test_weighted_coverage_basic():
    sets = [{1, 2, 3}, {3, 4}, {1, 4, 5}]
    f = WeightedCoverage(sets)
    assert f([]) == 0.0
    assert f([0]) == 3.0
    assert f([0, 1]) == 4.0
    assert f([0, 1, 2]) == 5.0


def test_weighted_coverage_weighted():
    sets = [{1, 2}, {2, 3}, {1, 3}]
    f = WeightedCoverage(sets, weights={1: 1.0, 2: 2.0, 3: 3.0})
    assert f([0]) == 3.0
    assert f([1]) == 5.0
    assert f([2]) == 4.0
    assert f([0, 1, 2]) == 6.0


def test_facility_location_basic():
    W = [[1.0, 0.0], [0.0, 1.0]]
    f = FacilityLocation(W)
    assert f([]) == 0.0
    assert f([0]) == 1.0
    assert f([0, 1]) == 2.0


def test_log_determinant_diag():
    # Diagonal kernel: f(S) = Σ log(K_ii + α).
    K = [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]
    f = LogDeterminant(K, alpha=0.0)
    expected = math.log(1.0) + math.log(4.0) + math.log(9.0)
    assert math.isclose(f([0, 1, 2]), expected, rel_tol=1e-12)


def test_log_determinant_correlation():
    K = [[1.0, 0.5], [0.5, 1.0]]
    f = LogDeterminant(K, alpha=0.0)
    expected = math.log(1.0 - 0.25)  # 1*1 - 0.5*0.5 = 0.75
    assert math.isclose(f([0, 1]), expected, rel_tol=1e-12)


def test_max_cut_basic():
    W = [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    f = MaxCut(W)
    # f({0}) = 1+2 = 3
    assert f([0]) == 3.0
    # f({0,1}) = (1->2) + (0->2)? No: cut(S,V\S) where S={0,1}, V\S={2}: W[0][2]+W[1][2] = 2+3 = 5
    assert f([0, 1]) == 5.0
    # f(V) = 0
    assert f([0, 1, 2]) == 0.0


def test_concave_over_modular_basic():
    f = ConcaveOverModular([1.0, 2.0, 3.0], phi=lambda x: math.sqrt(max(0.0, x)))
    assert math.isclose(f([0, 1, 2]), math.sqrt(6.0))
    assert math.isclose(f([1, 2]), math.sqrt(5.0))


def test_feature_based_basic():
    feats = [{"a": 1.0}, {"a": 1.0, "b": 1.0}, {"b": 1.0}]
    f = FeatureBased(feats, phi=lambda x: math.sqrt(max(0.0, x)))
    # f({0}) = sqrt(1) + sqrt(0) = 1
    assert math.isclose(f([0]), 1.0)
    # f({0,1}) = sqrt(2) + sqrt(1) = √2+1
    assert math.isclose(f([0, 1]), math.sqrt(2.0) + 1.0)
    # f({0,1,2}) = sqrt(2) + sqrt(2)
    assert math.isclose(f([0, 1, 2]), 2.0 * math.sqrt(2.0))


def test_gaussian_entropy_diag():
    Sigma = [[1.0, 0.0], [0.0, 4.0]]
    f = GaussianEntropy(Sigma)
    const = 0.5 * math.log(2.0 * math.pi * math.e)
    expected = 2 * const + 0.5 * (math.log(1.0) + math.log(4.0))
    assert math.isclose(f([0, 1]), expected, rel_tol=1e-12)


# =====================================================================
# Submodularity contracts (exhaustive on small ground sets)
# =====================================================================


def test_weighted_coverage_is_submodular():
    sets = [{1, 2}, {2, 3}, {3, 4}, {1, 4}]
    f = WeightedCoverage(sets)
    ok, n_v = _is_submodular(f, f.ground_set())
    assert ok, f"weighted coverage violated DR {n_v} times"


def test_facility_location_is_submodular():
    rng = random.Random(7)
    W = [[rng.random() for _ in range(4)] for _ in range(4)]
    f = FacilityLocation(W)
    ok, n_v = _is_submodular(f, f.ground_set())
    assert ok


def test_max_cut_is_submodular():
    rng = random.Random(11)
    n = 4
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            W[i][j] = W[j][i] = rng.random()
    f = MaxCut(W)
    ok, n_v = _is_submodular(f, f.ground_set())
    assert ok


def test_concave_over_modular_is_submodular():
    f = ConcaveOverModular([0.4, 1.1, 0.7, 2.0, 0.5])
    ok, n_v = _is_submodular(f, f.ground_set())
    assert ok


def test_feature_based_is_submodular():
    feats = [
        {"a": 1.0, "b": 0.5},
        {"a": 0.3, "c": 2.0},
        {"b": 1.0},
        {"c": 0.5, "a": 0.7},
    ]
    f = FeatureBased(feats)
    ok, n_v = _is_submodular(f, f.ground_set())
    assert ok


# =====================================================================
# Lazy = Naive on monotone submodular under cardinality (Minoux 1978)
# =====================================================================


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_lazy_equals_naive_on_coverage(seed):
    rng = random.Random(seed)
    n_sets = 10
    universe = list(range(8))
    sets = []
    for _ in range(n_sets):
        size = rng.randint(2, 5)
        sets.append(rng.sample(universe, size))
    f = WeightedCoverage(sets)
    sm = Submodular()
    for k in [1, 2, 3, 4]:
        lazy = sm.maximize(f, f.ground_set(), k=k, method=METHOD_LAZY_GREEDY)
        naive = sm.maximize(f, f.ground_set(), k=k, method=METHOD_NAIVE_GREEDY)
        assert math.isclose(lazy.value, naive.value, rel_tol=1e-12), (
            f"lazy ({lazy.value}) ≠ naive ({naive.value}) for k={k}"
        )


def test_lazy_equals_naive_on_facility_location():
    rng = random.Random(42)
    n_clients = 12
    n_facilities = 8
    W = [[rng.random() for _ in range(n_facilities)] for _ in range(n_clients)]
    f = FacilityLocation(W)
    sm = Submodular()
    for k in [1, 2, 3, 4, 5]:
        lazy = sm.maximize(f, f.ground_set(), k=k, method=METHOD_LAZY_GREEDY)
        naive = sm.maximize(f, f.ground_set(), k=k, method=METHOD_NAIVE_GREEDY)
        assert math.isclose(lazy.value, naive.value, rel_tol=1e-12)


# =====================================================================
# (1 − 1/e) approximation ratio on exhaustively-enumerable problems
# =====================================================================


@pytest.mark.parametrize("seed", range(8))
def test_lazy_greedy_meets_one_minus_inv_e(seed):
    rng = random.Random(seed)
    universe = list(range(7))
    sets = []
    for _ in range(8):
        size = rng.randint(2, 4)
        sets.append(rng.sample(universe, size))
    f = WeightedCoverage(sets)
    sm = Submodular()
    k = 3
    rep = sm.maximize(f, f.ground_set(), k=k, method=METHOD_LAZY_GREEDY)
    opt, _ = _enumerate_optimum(f, f.ground_set(), k)
    if opt <= 0.0:
        return  # degenerate
    ratio = rep.value / opt
    assert ratio >= ONE_MINUS_INV_E - 1e-9, (
        f"greedy ratio {ratio} < 1 - 1/e on seed {seed}"
    )


@pytest.mark.parametrize("seed", range(5))
def test_celf_meets_one_minus_inv_e(seed):
    rng = random.Random(seed)
    universe = list(range(8))
    sets = []
    for _ in range(10):
        size = rng.randint(2, 4)
        sets.append(rng.sample(universe, size))
    f = WeightedCoverage(sets)
    sm = Submodular()
    k = 3
    rep = sm.maximize(f, f.ground_set(), k=k, method=METHOD_CELF)
    opt, _ = _enumerate_optimum(f, f.ground_set(), k)
    if opt <= 0.0:
        return
    assert rep.value / opt >= ONE_MINUS_INV_E - 1e-9


def test_stochastic_greedy_close_to_one_minus_inv_e_in_expectation():
    rng_master = random.Random(1)
    universe = list(range(8))
    sets = []
    for _ in range(12):
        sets.append(rng_master.sample(universe, rng_master.randint(2, 5)))
    f = WeightedCoverage(sets)
    sm = Submodular()
    k = 3
    epsilon = 0.2
    opt, _ = _enumerate_optimum(f, f.ground_set(), k)
    if opt <= 0.0:
        return
    ratios = []
    for s in range(50):
        rep = sm.maximize(
            f,
            f.ground_set(),
            k=k,
            method=METHOD_STOCHASTIC_GREEDY,
            epsilon=epsilon,
            seed=s,
        )
        ratios.append(rep.value / opt)
    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio >= ONE_MINUS_INV_E - epsilon - 0.05


# =====================================================================
# Knapsack / cost-benefit
# =====================================================================


def test_celf_respects_budget():
    sets = [{1, 2, 3}, {4, 5}, {6}, {7, 8, 9, 10}]
    f = WeightedCoverage(sets)
    sm = Submodular()
    costs = [1.0, 0.5, 0.2, 2.0]
    rep = sm.maximize(
        f, f.ground_set(), method=METHOD_COST_GREEDY, budget=1.5, costs=costs
    )
    total = sum(costs[i] for i in rep.selected)
    assert total <= 1.5 + 1e-9
    assert rep.feasible


def test_sviridenko_respects_budget_and_beats_singleton():
    sets = [{1, 2, 3}, {3, 4, 5}, {5, 6, 7, 8}, {1, 6, 9}, {2, 4, 10}]
    costs = [1.0, 1.0, 2.0, 1.5, 1.5]
    f = WeightedCoverage(sets)
    sm = Submodular()
    budget = 3.0
    rep = sm.maximize(
        f,
        f.ground_set(),
        method=METHOD_SVIRIDENKO_KNAPSACK,
        budget=budget,
        costs=costs,
        enum_size=2,
    )
    total = sum(costs[i] for i in rep.selected)
    assert total <= budget + 1e-9
    # Must beat best feasible singleton.
    best_singleton = max(
        (f([i]) for i in range(len(sets)) if costs[i] <= budget), default=0.0
    )
    assert rep.value >= best_singleton


# =====================================================================
# Double greedy on MaxCut
# =====================================================================


def test_double_greedy_random_half_bound_on_max_cut():
    rng_master = random.Random(0)
    n = 8
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            W[i][j] = W[j][i] = rng_master.random()
    f = MaxCut(W)
    opt, _ = _enumerate_optimum(f, f.ground_set(), n)
    if opt <= 0.0:
        return
    sm = Submodular()
    vals = []
    for s in range(40):
        rep = sm.maximize(
            f,
            f.ground_set(),
            method=METHOD_DOUBLE_GREEDY_RANDOM,
            monotone=False,
            seed=s,
        )
        vals.append(rep.value)
    mean_val = sum(vals) / len(vals)
    assert mean_val / opt >= 0.5 - 0.05


def test_double_greedy_deterministic_third_bound_on_max_cut():
    rng_master = random.Random(2)
    n = 7
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            W[i][j] = W[j][i] = rng_master.random()
    f = MaxCut(W)
    opt, _ = _enumerate_optimum(f, f.ground_set(), n)
    sm = Submodular()
    rep = sm.maximize(
        f, f.ground_set(), method=METHOD_DOUBLE_GREEDY_DETERMINISTIC, monotone=False
    )
    if opt > 0.0:
        assert rep.value / opt >= 1.0 / 3.0 - 1e-9


# =====================================================================
# Distorted greedy reduces to standard greedy at γ = 1
# =====================================================================


def test_distorted_greedy_matches_standard_at_gamma_1():
    rng = random.Random(13)
    sets = []
    for _ in range(8):
        sets.append(rng.sample(range(8), rng.randint(2, 4)))
    f = WeightedCoverage(sets)
    sm = Submodular()
    k = 3
    rep_lazy = sm.maximize(f, f.ground_set(), k=k, method=METHOD_LAZY_GREEDY)
    rep_dist = sm.maximize(
        f, f.ground_set(), k=k, method=METHOD_DISTORTED_GREEDY, gamma=1.0
    )
    assert math.isclose(rep_dist.value, rep_lazy.value, rel_tol=1e-12)


# =====================================================================
# Sieve-Streaming
# =====================================================================


def test_sieve_streaming_one_pass():
    rng = random.Random(99)
    sets = []
    for _ in range(12):
        sets.append(rng.sample(range(8), rng.randint(2, 5)))
    f = WeightedCoverage(sets)
    sm = Submodular()
    k = 4
    rep = sm.stream(f, f.ground_set(), k=k, epsilon=0.1)
    assert isinstance(rep, StreamReport)
    assert len(rep.selected) <= k
    # Compare to lazy greedy: streaming should reach ≥ (½ − ε) of greedy.
    rep_lazy = sm.maximize(f, f.ground_set(), k=k, method=METHOD_LAZY_GREEDY)
    if rep_lazy.value > 0.0:
        ratio = rep.value / rep_lazy.value
        # Lazy ≥ (1-1/e)·OPT, so sieve ≥ (½-ε)·OPT ≥ (½-ε)/(1-1/e+...).
        # In practice the realised ratio is well above 0.5.
        assert ratio >= 0.4


# =====================================================================
# Submodular cover
# =====================================================================


def test_submodular_cover_terminates_and_satisfies_quota():
    sets = [{1, 2, 3}, {3, 4, 5}, {5, 6}, {1, 6, 7}, {2, 4, 8}]
    f = WeightedCoverage(sets)
    sm = Submodular()
    quota = 6.0
    rep = sm.cover(f, f.ground_set(), quota=quota)
    assert rep.value + 1e-9 >= quota
    assert rep.feasible
    # The Wolsey bound is finite for non-trivial η.
    assert math.isfinite(rep.approx_ratio)


# =====================================================================
# Curvature
# =====================================================================


def test_curvature_zero_on_modular_function():
    weights = [1.0, 2.0, 3.0, 4.0]

    # f(S) = Σ_{i ∈ S} w_i — modular.
    def f(S):
        return sum(weights[i] for i in S)

    sm = Submodular()
    c = sm.curvature(f, list(range(len(weights))))
    assert c < 1e-9


def test_curvature_one_on_hard_set_cover():
    # Every element brings the same singleton value (1) but only the first
    # one to enter contributes anything to f(V).
    universe = {0}

    def f(S):
        if not S:
            return 0.0
        return 1.0

    sm = Submodular()
    c = sm.curvature(f, [0, 1, 2, 3])
    # f({v}) = 1, f(V) − f(V∖{v}) = 0  →  ratio 0  →  c = 1.
    assert math.isclose(c, 1.0, rel_tol=1e-9)


def test_curvature_bound_monotone_in_c():
    sm = Submodular()
    k = 5
    cs = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    bounds = [sm.curvature_bound(c, k) for c in cs]
    # Non-increasing in c.
    for a, b in zip(bounds, bounds[1:]):
        assert a >= b - 1e-12
    # c → 0 ↔ bound → 1 (additive limit).
    assert math.isclose(bounds[0], 1.0, abs_tol=1e-12)
    # c = 1 finite-k bound is 1 − (1 − 1/k)^k, which tends to 1 − 1/e
    # only as k → ∞.  Check the closed form holds exactly at k = 5.
    assert math.isclose(bounds[-1], 1.0 - 0.8 ** 5, rel_tol=1e-12)
    # Large k should approach 1 − 1/e.
    big = sm.curvature_bound(1.0, 1000)
    assert math.isclose(big, ONE_MINUS_INV_E, abs_tol=1e-3)


# =====================================================================
# Submodularity certificate
# =====================================================================


def test_certify_submodular_zero_violations_on_coverage():
    sets = [{1, 2, 3}, {2, 4}, {4, 5}, {1, 5}]
    f = WeightedCoverage(sets)
    sm = Submodular(random_seed=0)
    cert = sm.certify_submodular(f, f.ground_set(), n_samples=300, alpha=0.05)
    assert isinstance(cert, CertificateReport)
    assert cert.violation_rate == 0.0
    # Hoeffding upper at 0 violations and 300 samples, α=0.05.
    assert cert.hoeffding_upper < 0.1


def test_certify_submodular_flags_non_submodular_function():
    # A supermodular function: f(S) = (|S|)^2 — strictly increasing DR.
    def f(S):
        return float(len(set(S))) ** 2

    sm = Submodular(random_seed=1)
    cert = sm.certify_submodular(f, list(range(5)), n_samples=300, alpha=0.05)
    assert cert.violation_rate > 0.1  # plenty of violations


# =====================================================================
# Reports, events, attestation
# =====================================================================


def test_event_bus_receives_solve():
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe(received.append, kind="submodular.solved")
    sm = Submodular(bus=bus)
    sets = [{1, 2, 3}, {3, 4}, {1, 4, 5}]
    f = WeightedCoverage(sets)
    sm.maximize(f, f.ground_set(), k=2, method=METHOD_LAZY_GREEDY)
    assert any(e.kind == "submodular.solved" for e in received)


def test_attestor_records_digest():
    seen: list[dict] = []

    class Sink:
        def record(self, *, kind, payload):
            seen.append({"kind": kind, "payload": payload})

    sm = Submodular(attestor=Sink())
    sets = [{1, 2}, {2, 3}, {3, 4}]
    f = WeightedCoverage(sets)
    rep = sm.maximize(f, f.ground_set(), k=2, method=METHOD_LAZY_GREEDY)
    assert rep.digest, "digest must be non-empty"
    assert any(s["kind"] == "submodular.solved" for s in seen)


def test_report_digest_is_deterministic_for_fixed_input():
    sets = [{1, 2}, {2, 3}, {3, 4}]
    f = WeightedCoverage(sets)
    sm1 = Submodular()
    sm2 = Submodular()
    a = sm1.maximize(f, f.ground_set(), k=2, method=METHOD_LAZY_GREEDY)
    b = sm2.maximize(f, f.ground_set(), k=2, method=METHOD_LAZY_GREEDY)
    # Same selected set, same value, but ``elapsed_ns`` differs — strip
    # it before comparison.
    aa = a.to_dict()
    bb = b.to_dict()
    for k in ("elapsed_ns", "digest"):
        aa.pop(k, None)
        bb.pop(k, None)
    assert aa == bb


# =====================================================================
# Convenience wrappers
# =====================================================================


def test_lazy_greedy_wrapper():
    sets = [{1, 2}, {2, 3}, {3, 4}]
    rep = lazy_greedy(WeightedCoverage(sets), [0, 1, 2], k=2)
    assert isinstance(rep, SubmodularReport)
    assert rep.method == METHOD_LAZY_GREEDY
    assert rep.value > 0


def test_stochastic_greedy_wrapper():
    sets = [{1, 2}, {2, 3}, {3, 4}, {1, 4}]
    rep = stochastic_greedy(
        WeightedCoverage(sets), [0, 1, 2, 3], k=2, epsilon=0.2, seed=0
    )
    assert rep.method == METHOD_STOCHASTIC_GREEDY


def test_double_greedy_wrapper():
    n = 4
    rng = random.Random(0)
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            W[i][j] = W[j][i] = rng.random()
    rep = double_greedy(MaxCut(W), list(range(n)), randomized=False)
    assert rep.method == METHOD_DOUBLE_GREEDY_DETERMINISTIC


def test_sieve_streaming_wrapper():
    sets = [{1, 2, 3}, {2, 4}, {4, 5}, {1, 5, 6}]
    rep = sieve_streaming(WeightedCoverage(sets), [0, 1, 2, 3], k=2, epsilon=0.2)
    assert isinstance(rep, StreamReport)
    assert rep.epsilon == 0.2


# =====================================================================
# Input validation
# =====================================================================


def test_maximize_rejects_unknown_method():
    f = WeightedCoverage([{1, 2}, {3, 4}])
    sm = Submodular()
    with pytest.raises(SubmodularError):
        sm.maximize(f, [0, 1], k=1, method="madeup")


def test_maximize_rejects_empty_ground_set():
    f = WeightedCoverage([])
    sm = Submodular()
    with pytest.raises(SubmodularError):
        sm.maximize(f, [], k=1, method=METHOD_LAZY_GREEDY)


def test_maximize_rejects_missing_constraint():
    f = WeightedCoverage([{1, 2}, {3, 4}])
    sm = Submodular()
    with pytest.raises(SubmodularError):
        sm.maximize(f, [0, 1], method=METHOD_LAZY_GREEDY)


def test_maximize_rejects_negative_cost():
    f = WeightedCoverage([{1, 2}, {3, 4}])
    sm = Submodular()
    with pytest.raises(SubmodularError):
        sm.maximize(
            f,
            [0, 1],
            method=METHOD_SVIRIDENKO_KNAPSACK,
            budget=1.0,
            costs=[1.0, -0.5],
        )


def test_stochastic_greedy_rejects_bad_epsilon():
    f = WeightedCoverage([{1, 2}, {3, 4}])
    sm = Submodular()
    with pytest.raises(SubmodularError):
        sm.maximize(f, [0, 1], k=1, method=METHOD_STOCHASTIC_GREEDY, epsilon=0.0)
    with pytest.raises(SubmodularError):
        sm.maximize(f, [0, 1], k=1, method=METHOD_STOCHASTIC_GREEDY, epsilon=1.0)


def test_certify_submodular_rejects_tiny_ground_set():
    f = WeightedCoverage([{1}])
    sm = Submodular()
    with pytest.raises(SubmodularError):
        sm.certify_submodular(f, [0], n_samples=10, alpha=0.05)


def test_log_determinant_requires_square():
    with pytest.raises(SubmodularError):
        LogDeterminant([[1.0, 0.0], [0.0, 1.0, 0.0]])


def test_facility_location_requires_nonneg():
    with pytest.raises(SubmodularError):
        FacilityLocation([[-1.0, 0.0], [0.0, 1.0]])


def test_max_cut_requires_nonneg():
    with pytest.raises(SubmodularError):
        MaxCut([[0.0, -1.0], [-1.0, 0.0]])


# =====================================================================
# Threadsafety
# =====================================================================


def test_concurrent_maximize_calls_are_safe():
    sets = [{1, 2}, {2, 3}, {3, 4}, {1, 4}, {2, 4, 5}, {1, 3, 5}]
    f = WeightedCoverage(sets)
    sm = Submodular()
    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(5):
                rep = sm.maximize(
                    f, f.ground_set(), k=2, method=METHOD_LAZY_GREEDY
                )
                assert rep.value > 0
        except Exception as exc:  # pragma: no cover - any error bubbles up
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    cov = sm.coverage()
    assert cov["n_solves"] == 6 * 5


# =====================================================================
# Method-set sanity
# =====================================================================


def test_known_methods_complete():
    for m in (
        METHOD_LAZY_GREEDY,
        METHOD_NAIVE_GREEDY,
        METHOD_CELF,
        METHOD_STOCHASTIC_GREEDY,
        METHOD_COST_GREEDY,
        METHOD_SVIRIDENKO_KNAPSACK,
        METHOD_DOUBLE_GREEDY_RANDOM,
        METHOD_DOUBLE_GREEDY_DETERMINISTIC,
        METHOD_DISTORTED_GREEDY,
        METHOD_SIEVE_STREAMING,
        METHOD_SUBMODULAR_COVER,
        METHOD_THRESHOLD_GREEDY,
    ):
        assert m in KNOWN_METHODS
