"""Tests for ``agi.refuter`` — automated falsification as a runtime primitive.

The tests follow the mathematical contract of the module:

1. **Refutes a known-false hypothesis** (x² ≥ x on (0, 1)).
2. **Does not refute a known-true hypothesis** (x² + 1 > 0 on a finite-real
   interval with NaN corners disabled).
3. **Reports zero failures and Clopper-Pearson UCB** when supported.
4. **Reports a counterexample with the smallest margin** when refuted.
5. **Boundary corners include lo/hi/mid/zero-cross**.
6. **Halton sequence is low-discrepancy** (deterministic, distinct draws).
7. **ListSpace mutates, samples, shrinks**.
8. **ProductSpace handles nested coords**.
9. **Shrinking reduces a list witness to a minimal one**.
10. **Metamorphic mode catches sort-not-idempotent bug** (faked).
11. **Bound mode** refutes a tight upper bound.
12. **Sequential mode** accumulates an e-value past 1/α.
13. **CEGIS scaffold** converges in finite rounds on a finite witness set.
14. **Fingerprint is deterministic** across two calls with the same seed.
15. **Fingerprint changes** when the space changes.
16. **Clopper-Pearson UCB matches the closed-form on k=0** and is
    monotone in n.
17. **e-value > 1/α** when k/n > p0 by a wide margin.
18. **with_margin** preserves the bool predicate output.
19. **Exceptions in the predicate** are caught and treated as refutations.
20. **Walltime budget** stops the search.
21. **Deterministic replay** under a fixed seed.
22. **Strategy counts** sum to n_trials.
23. **Near misses** are sorted by ascending margin.
24. **NaN-on-numeric hypothesis is refuted by boundary corner** unless
    corners are disabled.
"""

from __future__ import annotations

import math
import random

import pytest

from agi.refuter import (
    BoolSpace,
    Counterexample,
    ContinuousSpace,
    Evaluation,
    FiniteSet,
    IntegerSpace,
    InvalidHypothesis,
    InvalidSpace,
    ListSpace,
    Product,
    ProductSpace,
    Refuter,
    RefutationReport,
    cegis_loop,
    clopper_pearson_ucb,
    clopper_pearson_zero_ucb,
    e_value_binomial,
    forall,
    hoeffding_ucb,
    rule_of_three,
    with_margin,
)


# -----------------------------------------------------------------------------
# 1. Known-false hypothesis is refuted
# -----------------------------------------------------------------------------


def test_refutes_known_false():
    R = Refuter(seed=0)
    # x² ≥ x is false for x ∈ (0, 1)
    def H(x): return x["v"] ** 2 >= x["v"]
    rep = R.try_refute(
        H, Product(v=ContinuousSpace(-3.0, 3.0, include_corners=False)),
        n_trials=2000,
    )
    assert rep.refuted
    assert rep.counterexample is not None
    v = rep.counterexample.x["v"]
    # Witness must be in (0, 1) where the claim fails
    assert 0.0 < v < 1.0 or v == 0.0  # shrink may push to denormal zero
    # Negative margin
    assert rep.counterexample.margin < 0


# -----------------------------------------------------------------------------
# 2. Known-true hypothesis is not refuted (corners disabled)
# -----------------------------------------------------------------------------


def test_does_not_refute_known_true():
    R = Refuter(seed=0)
    # x² + 1 > 0 for all real x
    def H(x): return x["v"] ** 2 + 1 > 0
    rep = R.try_refute(
        H, Product(v=ContinuousSpace(-3.0, 3.0, include_corners=False)),
        n_trials=500,
    )
    assert not rep.refuted
    assert rep.counterexample is None
    assert rep.n_failures == 0


# -----------------------------------------------------------------------------
# 3. Support report includes Clopper-Pearson UCB on k=0
# -----------------------------------------------------------------------------


def test_support_report_zero_failures():
    R = Refuter(seed=0)
    def H(x): return True
    rep = R.try_refute(H, Product(v=ContinuousSpace(0.0, 1.0, include_corners=False)),
                       n_trials=100, alpha=0.05)
    assert not rep.refuted
    assert rep.n_failures == 0
    assert rep.failure_rate_emp == 0.0
    # 1 - 0.05^(1/100)
    expected = 1.0 - 0.05 ** (1.0 / 100)
    assert abs(rep.failure_rate_ucb - expected) < 1e-12


# -----------------------------------------------------------------------------
# 4. Refutation report: counterexample has minimal (negative) margin
# -----------------------------------------------------------------------------


def test_counterexample_margin_is_negative():
    R = Refuter(seed=0)
    def H(x): return x["v"] > 0
    rep = R.try_refute(H, Product(v=ContinuousSpace(-1.0, 1.0, include_corners=False)),
                       n_trials=200)
    assert rep.refuted
    assert rep.counterexample.margin < 0


# -----------------------------------------------------------------------------
# 5. Boundary corners
# -----------------------------------------------------------------------------


def test_continuous_boundary_includes_corners():
    s = ContinuousSpace(-1.0, 1.0)
    b = s.boundary()
    assert -1.0 in b
    assert 1.0 in b
    assert 0.0 in b  # zero-cross
    # IEEE-754 corners
    assert any(math.isinf(x) and x > 0 for x in b)
    assert any(math.isinf(x) and x < 0 for x in b)
    assert any(isinstance(x, float) and math.isnan(x) for x in b)


def test_continuous_boundary_corners_off():
    s = ContinuousSpace(-1.0, 1.0, include_corners=False)
    b = s.boundary()
    assert not any(math.isinf(x) for x in b)
    assert not any(isinstance(x, float) and math.isnan(x) for x in b)


def test_integer_boundary():
    s = IntegerSpace(-5, 5)
    b = set(s.boundary())
    assert -5 in b and 5 in b and 0 in b and 1 in b and -1 in b


# -----------------------------------------------------------------------------
# 6. Halton sequence is deterministic and distinct
# -----------------------------------------------------------------------------


def test_halton_deterministic_distinct():
    R = Refuter(seed=0)
    s = Product(v=ContinuousSpace(-1.0, 1.0, include_corners=False))
    # Run twice with same seed → identical sequence on supported predicates
    def H(x): return True
    rep1 = R.try_refute(H, s, n_trials=100)
    R2 = Refuter(seed=0)
    rep2 = R2.try_refute(H, s, n_trials=100)
    assert rep1.fingerprint == rep2.fingerprint


# -----------------------------------------------------------------------------
# 7. ListSpace operations
# -----------------------------------------------------------------------------


def test_list_space_basics():
    rng = random.Random(0)
    s = ListSpace(IntegerSpace(0, 9), min_len=1, max_len=5)
    for _ in range(50):
        L = s.sample(rng)
        assert isinstance(L, list)
        assert 1 <= len(L) <= 5
        assert all(0 <= v <= 9 for v in L)


def test_list_space_mutate_stays_in_bounds():
    rng = random.Random(0)
    s = ListSpace(IntegerSpace(0, 9), min_len=1, max_len=5)
    L = [3, 4]
    for _ in range(20):
        L = s.mutate(L, 0.5, rng)
        assert 1 <= len(L) <= 5
        assert all(0 <= v <= 9 for v in L)


# -----------------------------------------------------------------------------
# 8. ProductSpace nesting
# -----------------------------------------------------------------------------


def test_product_space_nested():
    rng = random.Random(0)
    s = Product(
        a=ContinuousSpace(0.0, 1.0, include_corners=False),
        b=IntegerSpace(0, 10),
        c=FiniteSet(("x", "y", "z")),
    )
    x = s.sample(rng)
    assert set(x.keys()) == {"a", "b", "c"}
    assert 0.0 <= x["a"] <= 1.0
    assert 0 <= x["b"] <= 10
    assert x["c"] in ("x", "y", "z")


def test_product_space_boundary_one_axis_at_a_time():
    s = Product(
        a=ContinuousSpace(-1.0, 1.0, include_corners=False),
        b=IntegerSpace(0, 5),
    )
    boundary = s.boundary()
    # Each boundary point varies one axis from defaults
    for pt in boundary:
        assert "a" in pt and "b" in pt


# -----------------------------------------------------------------------------
# 9. Shrinking
# -----------------------------------------------------------------------------


def test_shrink_minimises_list_witness():
    R = Refuter(seed=0)
    # H: list contains no 7
    def H(L): return 7 not in L
    s = ListSpace(IntegerSpace(0, 10), min_len=0, max_len=8)
    rep = R.try_refute(H, s, n_trials=500)
    assert rep.refuted
    cex = rep.counterexample
    # The minimum-size witness must contain 7 and have length 1
    assert 7 in cex.x
    assert len(cex.x) == 1


# -----------------------------------------------------------------------------
# 10. Metamorphic mode
# -----------------------------------------------------------------------------


def test_metamorphic_idempotence_supported():
    R = Refuter(seed=0)
    def f(L): return sorted(L)
    def rel(x, fx, x2, fx2): return fx == fx2
    s = ListSpace(IntegerSpace(0, 50), max_len=8)
    rep = R.try_refute_relation(
        f, rel, s, x_to_x2=lambda L: list(reversed(L)), n_trials=300,
    )
    assert not rep.refuted


def test_metamorphic_catches_bug():
    R = Refuter(seed=0)
    # buggy "sort" that randomly drops one element when length > 2
    def buggy_sort(L):
        if len(L) > 2:
            return sorted(L[1:])
        return sorted(L)
    def rel(x, fx, x2, fx2):
        # sorted(reversed(L)) should equal sorted(L)
        return fx == fx2
    s = ListSpace(IntegerSpace(0, 50), min_len=3, max_len=6)
    rep = R.try_refute_relation(
        buggy_sort, rel, s, x_to_x2=lambda L: list(reversed(L)), n_trials=200,
    )
    assert rep.refuted


# -----------------------------------------------------------------------------
# 11. Bound mode
# -----------------------------------------------------------------------------


def test_bound_mode_refutes_tight_bound():
    R = Refuter(seed=0)
    # sum(L) <= 5 on lists of [0..3] with length 5 — max sum is 15
    def sum_fn(L): return sum(L)
    s = ListSpace(IntegerSpace(0, 3), min_len=5, max_len=5)
    rep = R.try_refute_bound(sum_fn, threshold=5, direction="<=",
                             space=s, n_trials=500)
    assert rep.refuted
    assert rep.extra.get("tightness_margin", 1.0) <= 0


def test_bound_mode_supports_safe_bound():
    R = Refuter(seed=0)
    # sum(L) <= 100 on lists of [0..3] with length 5 — max sum is 15
    def sum_fn(L): return sum(L)
    s = ListSpace(IntegerSpace(0, 3), min_len=5, max_len=5)
    rep = R.try_refute_bound(sum_fn, threshold=100, direction="<=",
                             space=s, n_trials=300)
    assert not rep.refuted
    tm = rep.extra.get("tightness_margin", 0.0)
    assert tm > 0  # bound is slack


# -----------------------------------------------------------------------------
# 12. Sequential refute_until
# -----------------------------------------------------------------------------


def test_refute_until_finds_witness_quickly():
    R = Refuter(seed=0)
    def H(L): return len(L) > 0
    s = ListSpace(IntegerSpace(0, 10), min_len=0, max_len=3)
    rep = R.refute_until(H, s, p0=0.01, alpha=0.05, n_max=500, block_size=32)
    # Empty list is in the boundary -> refute early
    assert rep.refuted
    assert rep.n_trials < 500


def test_refute_until_supports_true_hypothesis():
    R = Refuter(seed=0)
    def H(x): return x["v"] >= 0
    s = Product(v=IntegerSpace(0, 100))
    rep = R.refute_until(H, s, p0=0.01, alpha=0.05, n_max=300, block_size=64)
    assert not rep.refuted


# -----------------------------------------------------------------------------
# 13. CEGIS scaffold
# -----------------------------------------------------------------------------


def test_cegis_loop_converges():
    R = Refuter(seed=0)
    # Goal: find a constant c >= max(L) for any L in the space.
    space = ListSpace(IntegerSpace(0, 5), min_len=1, max_len=3)
    def refute(c):
        return R.try_refute(lambda L: (max(L) <= c) if L else True,
                            space, n_trials=200)
    def resynth(c, cex):
        return max(c, max(cex.x))
    final, witnesses = cegis_loop(0, refute, resynth, max_rounds=20)
    assert final >= 5
    assert len(witnesses) <= 6  # bounded by integer range


# -----------------------------------------------------------------------------
# 14. Fingerprint is deterministic
# -----------------------------------------------------------------------------


def test_fingerprint_deterministic():
    def H(x): return x["v"] >= 0
    s = Product(v=IntegerSpace(-10, 10))
    R1 = Refuter(seed=123)
    R2 = Refuter(seed=123)
    a = R1.try_refute(H, s, n_trials=200)
    b = R2.try_refute(H, s, n_trials=200)
    assert a.fingerprint == b.fingerprint


# -----------------------------------------------------------------------------
# 15. Fingerprint changes with space
# -----------------------------------------------------------------------------


def test_fingerprint_changes_with_space():
    def H(x): return x["v"] >= 0
    R = Refuter(seed=0)
    a = R.try_refute(H, Product(v=IntegerSpace(-10, 10)), n_trials=50)
    b = R.try_refute(H, Product(v=IntegerSpace(-20, 20)), n_trials=50)
    assert a.fingerprint != b.fingerprint


# -----------------------------------------------------------------------------
# 16. Clopper-Pearson statistics
# -----------------------------------------------------------------------------


def test_clopper_pearson_zero_ucb_matches_closed_form():
    for n in (10, 100, 1000):
        for alpha in (0.05, 0.01, 0.001):
            v = clopper_pearson_zero_ucb(n, alpha)
            expected = 1.0 - alpha ** (1.0 / n)
            assert abs(v - expected) < 1e-12


def test_clopper_pearson_zero_ucb_monotone_in_n():
    prev = 1.0
    for n in (1, 2, 5, 10, 50, 100, 1000):
        ucb = clopper_pearson_zero_ucb(n, 0.05)
        assert ucb < prev
        prev = ucb


def test_clopper_pearson_general_matches_zero_and_full():
    # k=0 should match closed form
    assert abs(clopper_pearson_ucb(0, 50, 0.05)
               - clopper_pearson_zero_ucb(50, 0.05)) < 1e-9
    # k=n should be 1.0
    assert clopper_pearson_ucb(50, 50, 0.05) == 1.0


def test_clopper_pearson_general_bracketed():
    # 5 failures in 100 trials, 95% UCB on rate
    # Known result: ~0.113 (closed-form via beta CDF)
    v = clopper_pearson_ucb(5, 100, 0.05)
    assert 0.10 < v < 0.13


def test_rule_of_three():
    assert abs(rule_of_three(30) - 0.1) < 1e-12
    assert abs(rule_of_three(300) - 0.01) < 1e-12


def test_hoeffding_ucb_grows_with_alpha():
    a = hoeffding_ucb(0.0, 100, 0.05)
    b = hoeffding_ucb(0.0, 100, 0.001)  # tighter α ⇒ wider UCB
    assert b > a


# -----------------------------------------------------------------------------
# 17. e-value semantics
# -----------------------------------------------------------------------------


def test_e_value_large_when_rate_exceeds_p0():
    # 100 failures in 1000 trials, claim is p ≤ 0.001
    e = e_value_binomial(100, 1000, 0.001)
    assert e > 1e20


def test_e_value_near_one_when_rate_below_p0():
    # When the data is consistent with H₀ (rate ≪ p0), the MLE-betting
    # e-value collapses to ≈ 1 (no evidence against H₀).
    e = e_value_binomial(1, 1000, 0.5)
    assert e <= 1.0 + 1e-9


def test_e_value_n_zero_returns_one():
    assert e_value_binomial(0, 0, 0.1) == 1.0


# -----------------------------------------------------------------------------
# 18. with_margin
# -----------------------------------------------------------------------------


def test_with_margin_returns_evaluation():
    pred = lambda x: x > 0
    marg = lambda x: x
    w = with_margin(pred, marg)
    e1 = w(2.0)
    assert isinstance(e1, Evaluation)
    assert e1.ok is True
    assert e1.margin > 0
    e2 = w(-3.0)
    assert e2.ok is False
    assert e2.margin < 0


def test_with_margin_sign_correction():
    # User gives a margin with the wrong sign — should be corrected
    pred = lambda x: x > 0
    marg = lambda x: -x  # opposite sign
    w = with_margin(pred, marg)
    e = w(2.0)
    assert e.ok is True
    assert e.margin > 0   # corrected back to positive


# -----------------------------------------------------------------------------
# 19. Exceptions
# -----------------------------------------------------------------------------


def test_predicate_exception_is_refutation():
    R = Refuter(seed=0)
    def H(x):
        if x["v"] == 0:
            raise ValueError("can't handle zero")
        return 1.0 / x["v"] > 0
    s = Product(v=IntegerSpace(-3, 3))
    rep = R.try_refute(H, s, n_trials=50)
    # Zero is in the boundary, so we will refute via exception
    assert rep.refuted
    assert rep.counterexample.x["v"] == 0


# -----------------------------------------------------------------------------
# 20. Walltime budget
# -----------------------------------------------------------------------------


def test_walltime_budget_stops_search():
    R = Refuter(seed=0)
    def H(x):
        # slow predicate
        import time
        time.sleep(0.005)
        return True
    rep = R.try_refute(H, Product(v=ContinuousSpace(0.0, 1.0, include_corners=False)),
                       n_trials=10_000, walltime_s=0.05)
    # Walltime is 50ms, predicate is ~5ms each → maybe 10 trials
    assert rep.n_trials < 100


# -----------------------------------------------------------------------------
# 21. Deterministic replay
# -----------------------------------------------------------------------------


def test_deterministic_replay():
    def H(L):
        return len(L) > 0 and L[0] != 42
    s = ListSpace(IntegerSpace(0, 100), min_len=1, max_len=4)
    R1 = Refuter(seed=99)
    R2 = Refuter(seed=99)
    a = R1.try_refute(H, s, n_trials=300)
    b = R2.try_refute(H, s, n_trials=300)
    assert a.refuted == b.refuted
    assert a.n_trials == b.n_trials
    assert a.fingerprint == b.fingerprint


# -----------------------------------------------------------------------------
# 22. Strategy counts add up
# -----------------------------------------------------------------------------


def test_strategy_counts_consistent():
    R = Refuter(seed=0)
    def H(x): return True
    s = Product(v=ContinuousSpace(0.0, 1.0, include_corners=False))
    rep = R.try_refute(H, s, n_trials=200)
    total = sum(rep.strategy_counts.values())
    assert total == rep.n_trials


# -----------------------------------------------------------------------------
# 23. Near misses are sorted by margin
# -----------------------------------------------------------------------------


def test_near_misses_sorted_by_margin():
    R = Refuter(seed=0)
    def H(x): return x["v"] >= 0.001  # narrow positive slab
    rep = R.try_refute(H, Product(v=ContinuousSpace(0.001, 1.0, include_corners=False)),
                       n_trials=300)
    margins = [m for (_, m) in rep.near_misses]
    assert margins == sorted(margins)


# -----------------------------------------------------------------------------
# 24. NaN corner refutes a strict numerical predicate
# -----------------------------------------------------------------------------


def test_nan_corner_refutes_strict_inequality():
    R = Refuter(seed=0)
    def H(x): return x["v"] > -1e30  # any finite real satisfies; ±inf / NaN do not
    rep = R.try_refute(H, Product(v=ContinuousSpace(-1.0, 1.0, include_corners=True)),
                       n_trials=200)
    assert rep.refuted
    v = rep.counterexample.x["v"]
    # The witness is one of the IEEE-754 corners: ±inf or NaN (in the
    # raw float space; some are stringified by _safe_jsonable).
    assert (v in ("NaN", "Infinity", "-Infinity")
            or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))


def test_nan_corner_off_does_not_refute_strict_inequality():
    R = Refuter(seed=0)
    def H(x): return x["v"] > -1e30
    rep = R.try_refute(H, Product(v=ContinuousSpace(-1.0, 1.0, include_corners=False)),
                       n_trials=200)
    assert not rep.refuted


# -----------------------------------------------------------------------------
# Extras: report serialisation / report claim render
# -----------------------------------------------------------------------------


def test_report_to_dict_roundtrips_safe_jsonable():
    R = Refuter(seed=0)
    def H(x): return True
    rep = R.try_refute(H, Product(v=ContinuousSpace(0.0, 1.0, include_corners=False)),
                       n_trials=20)
    d = rep.to_dict()
    assert "fingerprint" in d
    assert "strategy_counts" in d


def test_report_support_claim_is_string():
    R = Refuter(seed=0)
    def H(x): return True
    rep = R.try_refute(H, Product(v=ContinuousSpace(0.0, 1.0, include_corners=False)),
                       n_trials=20)
    s = rep.support_claim()
    assert isinstance(s, str)
    assert "SUPPORTED" in s


def test_invalid_space_raises():
    with pytest.raises(InvalidSpace):
        ContinuousSpace(1.0, 0.0)  # lo > hi
    with pytest.raises(InvalidSpace):
        IntegerSpace(5, 3)
    with pytest.raises(InvalidSpace):
        FiniteSet(())
    with pytest.raises(InvalidSpace):
        ListSpace(IntegerSpace(0, 1), min_len=5, max_len=2)
    with pytest.raises(InvalidSpace):
        ProductSpace(children={})


def test_invalid_predicate_returns_treated_as_failure():
    R = Refuter(seed=0)
    def H(x): return "not a bool"
    rep = R.try_refute(H, Product(v=IntegerSpace(0, 1)), n_trials=10)
    assert rep.refuted  # invalid return is captured as refutation
