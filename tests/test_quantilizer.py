"""Tests for the Quantilizer runtime primitive."""
from __future__ import annotations

import math
import random
import pytest

from agi.quantilizer import (
    ANYTIME,
    BERNSTEIN,
    BudgetInfeasible,
    CostBound,
    DKW,
    HARD,
    HOEFFDING,
    InsufficientData,
    InvalidDistribution,
    InvalidQuantile,
    InvalidSamples,
    InvalidUtility,
    KNOWN_ALGORITHMS,
    KNOWN_EVENTS,
    KNOWN_LCB_METHODS,
    Observation,
    QUANTILIZER_OBSERVED,
    QUANTILIZER_QUANTILIZED,
    QUANTILIZER_REPORT,
    QUANTILIZER_SELECTED,
    QUANTILIZER_STARTED,
    QuantilizedDistribution,
    QuantileEstimate,
    Quantilizer,
    QuantilizerError,
    QuantilizerReport,
    SAMPLE,
    SOFT,
    SampleQuantilization,
    Selection,
    TOP_K,
    UnknownAlgorithm,
    UnknownLCBMethod,
    UtilityBound,
    anytime_lcb,
    anytime_ucb,
    bretagnolle_huber_tv_from_kl,
    cost_amplification,
    dkw_band,
    empirical_bernstein_lcb,
    empirical_bernstein_ucb,
    hoeffding_lcb,
    hoeffding_ucb,
    kl_bound_from_quantile,
    kl_kl_bernoulli,
    le_cam_overlap_from_tv,
    pinsker_tv_from_kl,
    quantile_lcb_dkw,
    quantilize_bandit_distribution,
    quantilize_discrete,
    quantilize_policy_improvement,
    quantilize_samples,
    quantilize_top_k,
    quantilizer_from_spec,
    sample_from_distribution,
    soft_quantilize,
    soft_quantilize_with_beta,
    tv_bound_from_quantile,
)


# =====================================================================
# Information-theoretic bound helpers
# =====================================================================


class TestKLBoundFromQuantile:
    def test_q_one_gives_zero(self):
        assert kl_bound_from_quantile(1.0) == 0.0

    def test_q_half(self):
        assert kl_bound_from_quantile(0.5) == pytest.approx(math.log(2.0))

    def test_q_tenth(self):
        assert kl_bound_from_quantile(0.1) == pytest.approx(math.log(10.0))

    def test_q_hundredth(self):
        assert kl_bound_from_quantile(0.01) == pytest.approx(math.log(100.0))

    def test_q_zero_rejected(self):
        with pytest.raises(InvalidQuantile):
            kl_bound_from_quantile(0.0)

    def test_q_above_one_rejected(self):
        with pytest.raises(InvalidQuantile):
            kl_bound_from_quantile(1.1)

    def test_nan_rejected(self):
        with pytest.raises(InvalidQuantile):
            kl_bound_from_quantile(float("nan"))


class TestCostAmplification:
    def test_q_one_one(self):
        assert cost_amplification(1.0) == 1.0

    def test_q_tenth(self):
        assert cost_amplification(0.1) == pytest.approx(10.0)

    def test_q_hundredth(self):
        assert cost_amplification(0.01) == pytest.approx(100.0)


class TestTVBoundFromQuantile:
    def test_q_one_zero(self):
        assert tv_bound_from_quantile(1.0) == 0.0

    def test_q_half_half(self):
        assert tv_bound_from_quantile(0.5) == 0.5

    def test_q_tenth(self):
        assert tv_bound_from_quantile(0.1) == pytest.approx(0.9)


class TestDivergenceConversions:
    def test_pinsker_zero(self):
        assert pinsker_tv_from_kl(0.0) == 0.0

    def test_pinsker_log2(self):
        assert pinsker_tv_from_kl(math.log(2.0)) == pytest.approx(
            math.sqrt(math.log(2.0) / 2.0))

    def test_pinsker_negative_kl_rejected(self):
        with pytest.raises(QuantilizerError):
            pinsker_tv_from_kl(-0.1)

    def test_bretagnolle_huber_zero(self):
        assert bretagnolle_huber_tv_from_kl(0.0) == 0.0

    def test_bretagnolle_huber_log2(self):
        # 1 - exp(-log 2) = 0.5; sqrt = sqrt(0.5)
        assert bretagnolle_huber_tv_from_kl(math.log(2.0)) == pytest.approx(
            math.sqrt(0.5))

    def test_bretagnolle_huber_large_kl_close_to_one(self):
        v = bretagnolle_huber_tv_from_kl(10.0)
        assert 0.99 < v <= 1.0

    def test_le_cam_overlap(self):
        assert le_cam_overlap_from_tv(0.0) == 1.0
        assert le_cam_overlap_from_tv(0.5) == 0.5
        assert le_cam_overlap_from_tv(1.0) == 0.0

    def test_kl_bernoulli_symmetric_zero(self):
        assert kl_kl_bernoulli(0.5, 0.5) == pytest.approx(0.0)

    def test_kl_bernoulli_endpoint_infinity(self):
        assert kl_kl_bernoulli(0.5, 0.0) == float("inf")
        assert kl_kl_bernoulli(0.5, 1.0) == float("inf")

    def test_kl_bernoulli_positive_when_different(self):
        assert kl_kl_bernoulli(0.3, 0.7) > 0.0


# =====================================================================
# Discrete (exact) hard quantilizer
# =====================================================================


class TestQuantilizeDiscrete:
    def test_uniform_base_q_one_returns_base(self):
        base = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        utility = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.7}
        d = quantilize_discrete(base, utility, 1.0)
        assert d.q == 1.0
        for k in base:
            assert d.probs[k] == pytest.approx(base[k], abs=1e-9)
        assert d.realised_kl == pytest.approx(0.0, abs=1e-9)
        assert d.realised_tv == pytest.approx(0.0, abs=1e-9)
        assert d.kl_bound == pytest.approx(0.0, abs=1e-9)
        assert d.tv_bound == pytest.approx(0.0, abs=1e-9)

    def test_uniform_base_q_quarter_picks_argmax(self):
        base = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        utility = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.7}
        d = quantilize_discrete(base, utility, 0.25)
        # The top 25% (= 1 atom) is "c" (utility 0.9).
        assert d.probs == {"c": pytest.approx(1.0, abs=1e-9)}
        assert d.realised_kl == pytest.approx(math.log(4.0), abs=1e-9)
        assert d.kl_bound == pytest.approx(math.log(4.0), abs=1e-9)
        assert d.threshold == pytest.approx(0.9)

    def test_uniform_base_q_half_picks_top_two(self):
        base = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        utility = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.7}
        d = quantilize_discrete(base, utility, 0.5)
        # The top half is c (0.9) and d (0.7), each with renormalised mass 0.5
        assert set(d.probs.keys()) == {"c", "d"}
        for v in d.probs.values():
            assert v == pytest.approx(0.5, abs=1e-9)
        assert d.realised_kl == pytest.approx(math.log(2.0), abs=1e-9)

    def test_skewed_base_q_quarter(self):
        base = {"a": 0.6, "b": 0.3, "c": 0.05, "d": 0.05}
        utility = {"a": 0.1, "b": 0.2, "c": 0.99, "d": 0.95}
        d = quantilize_discrete(base, utility, 0.25)
        # Sorted by utility descending: c (0.99, mass .05), d (0.95, mass .05),
        # b (0.2, mass .3), a (0.1, mass .6).
        # Walking: c kept (cum .05), d kept (cum .1), b would be .4 > .25, so
        # take partial mass 0.15 of b.
        assert set(d.probs.keys()) == {"c", "d", "b"}
        assert d.probs["c"] == pytest.approx(0.05 / 0.25, abs=1e-9)
        assert d.probs["d"] == pytest.approx(0.05 / 0.25, abs=1e-9)
        assert d.probs["b"] == pytest.approx(0.15 / 0.25, abs=1e-9)
        # Sum to 1
        assert sum(d.probs.values()) == pytest.approx(1.0, abs=1e-9)

    def test_kl_bound_respected(self):
        random.seed(0)
        for _ in range(50):
            n = random.randint(2, 12)
            base = [random.random() for _ in range(n)]
            s = sum(base)
            base = {f"a{i}": p / s for i, p in enumerate(base)}
            utility = {a: random.random() for a in base}
            q = random.uniform(0.01, 1.0)
            d = quantilize_discrete(base, utility, q)
            assert d.realised_kl <= d.kl_bound + 1e-9
            assert d.realised_tv <= d.tv_bound + 1e-9
            assert sum(d.probs.values()) == pytest.approx(1.0, abs=1e-7)

    def test_kl_bound_tight_when_top_mass_is_q(self):
        # When the highest-utility atom has base mass exactly q, the
        # quantilizer puts all its mass on that atom: KL = log(1/q) exactly.
        base = {"a": 0.1, "b": 0.9}
        utility = {"a": 1.0, "b": 0.0}
        d = quantilize_discrete(base, utility, 0.1)
        assert d.realised_kl == pytest.approx(math.log(10.0), abs=1e-9)
        assert d.kl_bound == pytest.approx(math.log(10.0), abs=1e-9)

    def test_invalid_q_rejected(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        with pytest.raises(InvalidQuantile):
            quantilize_discrete(base, utility, 0.0)
        with pytest.raises(InvalidQuantile):
            quantilize_discrete(base, utility, 1.1)

    def test_missing_utility_rejected(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0}
        with pytest.raises(InvalidUtility):
            quantilize_discrete(base, utility, 0.5)

    def test_negative_mass_rejected(self):
        base = {"a": -0.1, "b": 1.1}
        utility = {"a": 1.0, "b": 0.0}
        with pytest.raises(InvalidDistribution):
            quantilize_discrete(base, utility, 0.5)

    def test_empty_base_rejected(self):
        with pytest.raises(InvalidDistribution):
            quantilize_discrete({}, {}, 0.5)

    def test_renormalisation_within_tolerance(self):
        # Sum slightly off from 1; should renormalise quietly.
        base = {"a": 0.5 + 1e-12, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = quantilize_discrete(base, utility, 0.5)
        assert abs(sum(d.probs.values()) - 1.0) < 1e-7

    def test_deterministic_tie_break(self):
        # When utilities tie, the tie-break is on SHA-256(action) — stable
        # across calls.
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 0.5, "b": 0.5}
        d1 = quantilize_discrete(base, utility, 0.5)
        d2 = quantilize_discrete(base, utility, 0.5)
        assert d1.fingerprint == d2.fingerprint

    def test_cost_amplification_field(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = quantilize_discrete(base, utility, 0.1)
        assert d.cost_amplification == pytest.approx(10.0)


class TestQuantilizeTopK:
    def test_k_one_picks_argmax(self):
        base = {"a": 0.5, "b": 0.3, "c": 0.2}
        utility = {"a": 0.1, "b": 0.9, "c": 0.5}
        d = quantilize_top_k(base, utility, 1)
        assert list(d.probs.keys()) == ["b"]
        assert d.probs["b"] == pytest.approx(1.0)

    def test_k_all_returns_base(self):
        base = {"a": 0.5, "b": 0.3, "c": 0.2}
        utility = {"a": 0.1, "b": 0.9, "c": 0.5}
        d = quantilize_top_k(base, utility, 3)
        for k, p in base.items():
            assert d.probs[k] == pytest.approx(p, abs=1e-9)
        assert d.realised_kl == pytest.approx(0.0, abs=1e-9)

    def test_k_overflow_clipped(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 0.1, "b": 0.9}
        d = quantilize_top_k(base, utility, 100)
        assert d.n_kept == 2

    def test_k_zero_rejected(self):
        with pytest.raises(QuantilizerError):
            quantilize_top_k({"a": 1.0}, {"a": 0.0}, 0)

    def test_kl_bound_from_kept_mass(self):
        base = {"a": 0.7, "b": 0.2, "c": 0.1}
        utility = {"a": 0.1, "b": 0.9, "c": 0.5}
        d = quantilize_top_k(base, utility, 2)
        # kept = {b, c}; kept mass = 0.3; KL bound = log(1/0.3)
        assert d.kl_bound == pytest.approx(-math.log(0.3), abs=1e-9)


# =====================================================================
# Soft (Boltzmann) quantilizer
# =====================================================================


class TestSoftQuantilize:
    def test_zero_budget_returns_base(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = soft_quantilize(base, utility, kl_budget=0.0)
        for k, p in base.items():
            assert d.probs[k] == pytest.approx(p, abs=1e-9)
        assert d.realised_kl == pytest.approx(0.0, abs=1e-9)

    def test_positive_budget_shifts_mass_to_higher_utility(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = soft_quantilize(base, utility, kl_budget=0.3)
        assert d.probs["a"] > d.probs["b"]
        assert d.probs["a"] > 0.5

    def test_kl_lands_on_budget(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        for B in (0.05, 0.1, 0.3, 0.5):
            d = soft_quantilize(base, utility, kl_budget=B)
            assert d.realised_kl == pytest.approx(B, abs=1e-6)
            assert d.kl_bound == pytest.approx(B, abs=1e-9)

    def test_max_budget_rejected(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        # Max achievable on this base is log(1 / min_base_mass) = log(2).
        with pytest.raises(BudgetInfeasible):
            soft_quantilize(base, utility, kl_budget=10.0)

    def test_negative_budget_rejected(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        with pytest.raises(QuantilizerError):
            soft_quantilize(base, utility, kl_budget=-0.1)

    def test_constant_utility_rejected(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 0.5, "b": 0.5}
        with pytest.raises(BudgetInfeasible):
            soft_quantilize(base, utility, kl_budget=0.1)

    def test_with_beta_zero_is_base(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = soft_quantilize_with_beta(base, utility, beta=0.0)
        for k, p in base.items():
            assert d.probs[k] == pytest.approx(p, abs=1e-9)

    def test_with_beta_large_picks_argmax(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = soft_quantilize_with_beta(base, utility, beta=50.0)
        assert d.probs["a"] > 0.99

    def test_with_beta_negative_rejected(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        with pytest.raises(QuantilizerError):
            soft_quantilize_with_beta(base, utility, beta=-1.0)


# =====================================================================
# Sample-based quantilizer
# =====================================================================


class TestQuantilizeSamples:
    def test_basic(self):
        samples = list(range(100))
        utility = lambda x: float(x)
        sq = quantilize_samples(samples, utility, 0.1)
        # top 10% = top 10 = [90..99]
        assert sq.n_kept == 10
        assert sq.kept == [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
        assert sq.threshold == 90
        assert sq.dkw_band > 0

    def test_q_one_keeps_all(self):
        samples = list(range(10))
        sq = quantilize_samples(samples, float, 1.0)
        assert sq.n_kept == 10

    def test_dkw_band_decreases_with_n(self):
        b_small = dkw_band(10, delta=0.05)
        b_large = dkw_band(10_000, delta=0.05)
        assert b_small > b_large
        assert b_large > 0

    def test_empty_samples_rejected(self):
        with pytest.raises(InvalidSamples):
            quantilize_samples([], float, 0.1)

    def test_non_numeric_utility_rejected(self):
        with pytest.raises(InvalidUtility):
            quantilize_samples([1, 2, 3], lambda x: "bad", 0.1)

    def test_quantile_lcb_within_band(self):
        random.seed(0)
        samples = [random.random() for _ in range(1000)]
        lo, hi = quantile_lcb_dkw(samples, 0.1, delta=0.05)
        # True 90th percentile of U(0,1) is 0.9.
        # Empirical should be near 0.9, and band should bracket it.
        emp = sorted(samples)[int(0.9 * 1000) - 1]
        assert lo <= emp <= hi


# =====================================================================
# Finite-sample LCB / UCB
# =====================================================================


class TestHoeffdingLCB:
    def test_zero_with_unit_range(self):
        # At δ = 1, log(1/δ) = 0 → half-width 0.
        v = hoeffding_lcb(0.5, 100, delta=0.999999)
        assert v == pytest.approx(0.5, abs=0.01)

    def test_bound_shrinks_with_n(self):
        v_small = hoeffding_lcb(0.5, 10, delta=0.05)
        v_large = hoeffding_lcb(0.5, 10_000, delta=0.05)
        assert v_small < v_large
        assert v_large < 0.5

    def test_invalid_delta(self):
        with pytest.raises(QuantilizerError):
            hoeffding_lcb(0.5, 100, delta=0.0)
        with pytest.raises(QuantilizerError):
            hoeffding_lcb(0.5, 100, delta=1.0)

    def test_ucb_symmetric_to_lcb(self):
        mean = 0.5
        n = 200
        delta = 0.05
        lcb = hoeffding_lcb(mean, n, delta=delta)
        ucb = hoeffding_ucb(mean, n, delta=delta)
        assert (mean - lcb) == pytest.approx(ucb - mean, abs=1e-9)


class TestBernsteinLCB:
    def test_n_one_rejected(self):
        with pytest.raises(InsufficientData):
            empirical_bernstein_lcb(0.5, 0.1, 1, delta=0.05)

    def test_zero_var_bound(self):
        v = empirical_bernstein_lcb(0.5, 0.0, 100, delta=0.05)
        # With var=0 only the range term survives.
        assert v < 0.5

    def test_smaller_var_gives_tighter_bound(self):
        # Higher var should give a lower (looser) LCB.
        v_low_var = empirical_bernstein_lcb(0.5, 0.01, 100, delta=0.05)
        v_high_var = empirical_bernstein_lcb(0.5, 0.25, 100, delta=0.05)
        assert v_high_var < v_low_var


class TestAnytimeLCB:
    def test_basic(self):
        v = anytime_lcb(0.5, 100, delta=0.05)
        assert v < 0.5

    def test_looser_than_hoeffding_for_fixed_n(self):
        # Anytime bounds pay a (1 + 1/n) and log√n+1 surcharge.
        h = hoeffding_lcb(0.5, 100, delta=0.05)
        a = anytime_lcb(0.5, 100, delta=0.05)
        assert a < h

    def test_ucb_basic(self):
        v = anytime_ucb(0.5, 100, delta=0.05)
        assert v > 0.5


class TestDKWBand:
    def test_band_decreases_with_n(self):
        b1 = dkw_band(10, delta=0.05)
        b2 = dkw_band(1000, delta=0.05)
        assert b1 > b2

    def test_band_increases_as_delta_shrinks(self):
        b1 = dkw_band(100, delta=0.5)
        b2 = dkw_band(100, delta=0.001)
        assert b2 > b1


# =====================================================================
# Sampling from a quantilized distribution
# =====================================================================


class TestSampleFromDistribution:
    def test_deterministic_seed(self):
        base = {"a": 0.5, "b": 0.3, "c": 0.2}
        utility = {"a": 0.1, "b": 0.9, "c": 0.5}
        d = quantilize_discrete(base, utility, 0.5)
        a1, p1 = sample_from_distribution(d, seed=42)
        a2, p2 = sample_from_distribution(d, seed=42)
        assert a1 == a2
        assert p1 == p2

    def test_different_seed_can_differ(self):
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = quantilize_discrete(base, utility, 1.0)  # full base
        outs = set()
        for s in range(20):
            a, _ = sample_from_distribution(d, seed=s)
            outs.add(a)
        # With q=1, both atoms are reachable.
        assert outs == {"a", "b"}

    def test_top_k_is_uniform(self):
        # Uniform base + top-2-by-utility → each kept atom mass 0.5.
        random.seed(0)
        base = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        utility = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.7}
        d = quantilize_discrete(base, utility, 0.5)
        counts = {"c": 0, "d": 0}
        for s in range(5000):
            a, _ = sample_from_distribution(d, seed=s)
            counts[a] += 1
        # Expect roughly 50/50.
        assert abs(counts["c"] - counts["d"]) < 250  # within ~5%


# =====================================================================
# Quantilizer class
# =====================================================================


class TestQuantilizerClass:
    def test_default_q(self):
        Q = Quantilizer(q=0.05)
        assert Q.q == 0.05
        assert Q.fingerprint != ""

    def test_invalid_q(self):
        with pytest.raises(InvalidQuantile):
            Quantilizer(q=0.0)
        with pytest.raises(InvalidQuantile):
            Quantilizer(q=1.5)

    def test_invalid_method(self):
        with pytest.raises(UnknownLCBMethod):
            Quantilizer(default_lcb_method="not_a_method")

    def test_quantilize_records(self):
        Q = Quantilizer(q=0.5)
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = Q.quantilize(base, utility)
        assert isinstance(d, QuantilizedDistribution)
        assert len(Q.quantizations()) == 1
        assert Q.quantizations()[0] is d

    def test_quantilize_top_k(self):
        Q = Quantilizer()
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = Q.quantilize(base, utility, algorithm=TOP_K, K=1)
        assert d.algorithm == TOP_K
        assert list(d.probs.keys()) == ["a"]

    def test_quantilize_soft(self):
        Q = Quantilizer()
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        d = Q.quantilize(base, utility, algorithm=SOFT, kl_budget=0.2)
        assert d.algorithm == SOFT
        assert d.realised_kl == pytest.approx(0.2, abs=1e-6)

    def test_quantilize_soft_requires_budget_or_beta(self):
        Q = Quantilizer()
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        with pytest.raises(QuantilizerError):
            Q.quantilize(base, utility, algorithm=SOFT)

    def test_quantilize_top_k_requires_k(self):
        Q = Quantilizer()
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        with pytest.raises(QuantilizerError):
            Q.quantilize(base, utility, algorithm=TOP_K)

    def test_unknown_algorithm_rejected(self):
        Q = Quantilizer()
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        with pytest.raises(UnknownAlgorithm):
            Q.quantilize(base, utility, algorithm="weird")

    def test_select_deterministic_seed(self):
        Q = Quantilizer(q=1.0)
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        s1 = Q.select(base, utility, seed=11)
        Q.clear()
        s2 = Q.select(base, utility, seed=11)
        assert s1.action == s2.action

    def test_select_records_certificates(self):
        Q = Quantilizer(q=0.25)
        base = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        utility = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.7}
        s = Q.select(base, utility, seed=0)
        assert s.kl_bound == pytest.approx(math.log(4.0))
        assert s.cost_amplification == pytest.approx(4.0)
        assert s.tv_bound == pytest.approx(0.75)
        assert s.utility == pytest.approx(0.9)
        assert s.action == "c"

    def test_select_advances_fingerprint(self):
        Q = Quantilizer()
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        fp_initial = Q.fingerprint
        Q.select(base, utility, seed=1)
        fp_after = Q.fingerprint
        assert fp_initial != fp_after

    def test_replay_determinism(self):
        # Two Quantilizers with the same selections should land on the
        # same fingerprint.
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        Q1 = Quantilizer(q=0.5)
        Q2 = Quantilizer(q=0.5)
        Q1.select(base, utility, seed=7)
        Q2.select(base, utility, seed=7)
        assert Q1.fingerprint == Q2.fingerprint

    def test_observe_advances_counters(self):
        Q = Quantilizer()
        Q.observe("a", 0.5)
        Q.observe("b", 0.7)
        r = Q.report()
        assert r.n_observations == 2

    def test_observe_invalid_utility_rejected(self):
        Q = Quantilizer()
        with pytest.raises(InvalidUtility):
            Q.observe("a", float("nan"))

    def test_clear_resets_fingerprint(self):
        Q = Quantilizer()
        Q.observe("a", 0.5)
        fp = Q.fingerprint
        Q.clear()
        assert Q.fingerprint != fp
        assert Q.report().n_observations == 0

    def test_report_contents(self):
        Q = Quantilizer(q=0.25)
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        Q.select(base, utility, seed=0)
        Q.observe("a", 0.9)
        r = Q.report()
        assert r.n_selections == 1
        assert r.n_observations == 1
        assert r.n_quantilized == 1
        assert r.last_selection is not None
        assert r.last_quantilized is not None
        assert r.config["q"] == 0.25

    def test_divergence_certificates(self):
        Q = Quantilizer(q=0.1)
        certs = Q.divergence_certificates()
        assert certs["kl_nats"] == pytest.approx(math.log(10.0))
        assert certs["kl_bits"] == pytest.approx(math.log2(10.0))
        assert certs["tv"] == pytest.approx(0.9)
        assert certs["cost_amplification"] == pytest.approx(10.0)

    def test_quantile_estimate_requires_base_observations(self):
        Q = Quantilizer()
        with pytest.raises(InsufficientData):
            Q.quantile_estimate()

    def test_quantile_estimate_from_base_observations(self):
        Q = Quantilizer(q=0.1)
        random.seed(0)
        for _ in range(500):
            u = random.random()
            Q.observe("a", u, came_from_quantilizer=False)
        est = Q.quantile_estimate()
        # True 0.9-quantile of U(0, 1) is 0.9.
        assert 0.85 < est.estimate < 0.95
        assert est.lcb <= est.estimate <= est.ucb

    def test_expected_utility_lcb_requires_data(self):
        Q = Quantilizer()
        with pytest.raises(InsufficientData):
            Q.expected_utility_lcb()

    def test_expected_utility_lcb_after_observations(self):
        Q = Quantilizer()
        for _ in range(50):
            Q.observe("a", 0.7)
        b = Q.expected_utility_lcb(delta=0.05)
        assert b.mean == pytest.approx(0.7, abs=1e-9)
        assert b.lcb < 0.7
        assert b.ucb > 0.7
        assert b.n == 50

    def test_bernstein_method(self):
        Q = Quantilizer(default_lcb_method=BERNSTEIN)
        random.seed(0)
        for _ in range(50):
            Q.observe("a", random.random())
        b = Q.expected_utility_lcb()
        assert b.method == BERNSTEIN
        assert b.variance > 0.0

    def test_anytime_method(self):
        Q = Quantilizer(default_lcb_method=ANYTIME)
        for _ in range(50):
            Q.observe("a", 0.5)
        b = Q.expected_utility_lcb()
        assert b.method == ANYTIME

    def test_dkw_unsupported_for_utility_lcb(self):
        Q = Quantilizer()
        for _ in range(50):
            Q.observe("a", 0.5)
        with pytest.raises(UnknownLCBMethod):
            Q.expected_utility_lcb(method=DKW)

    def test_cost_ucb(self):
        Q = Quantilizer(q=0.1)
        c = Q.cost_ucb(0.05)
        assert isinstance(c, CostBound)
        assert c.base_cost_ucb == 0.05
        assert c.amplification == pytest.approx(10.0)
        assert c.quantilizer_cost_ucb == pytest.approx(0.5)

    def test_cost_ucb_negative_rejected(self):
        Q = Quantilizer()
        with pytest.raises(QuantilizerError):
            Q.cost_ucb(-1.0)

    def test_emit_to_sink(self):
        events = []
        Q = Quantilizer(q=0.5,
                         sink=lambda k, p: events.append((k, dict(p))))
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        Q.select(base, utility, seed=0)
        Q.observe("a", 0.9)
        Q.report()
        kinds = {e[0] for e in events}
        assert QUANTILIZER_STARTED in kinds
        assert QUANTILIZER_QUANTILIZED in kinds
        assert QUANTILIZER_SELECTED in kinds
        assert QUANTILIZER_OBSERVED in kinds
        assert QUANTILIZER_REPORT in kinds

    def test_sink_exception_does_not_propagate(self):
        def bad(_kind, _payload):
            raise RuntimeError("sink broke")
        Q = Quantilizer(sink=bad)
        # Should still work.
        Q.observe("a", 0.5)
        assert Q.report().n_observations == 1


# =====================================================================
# Spec factory
# =====================================================================


class TestQuantilizerFromSpec:
    def test_basic(self):
        Q = quantilizer_from_spec({
            "q": 0.05,
            "default_lcb_method": "bernstein",
            "default_delta": 0.01,
            "utility_lower": -1.0,
            "utility_upper": 1.0,
            "default_seed": 7,
        })
        assert Q.q == 0.05

    def test_minimal(self):
        Q = quantilizer_from_spec({})
        assert Q.q == 0.1

    def test_invalid_spec(self):
        with pytest.raises(QuantilizerError):
            quantilizer_from_spec("not a mapping")


# =====================================================================
# Composition adapters
# =====================================================================


class TestComposition:
    def test_quantilize_bandit(self):
        arm_probs = {"A": 0.5, "B": 0.3, "C": 0.2}
        rewards = {"A": 0.4, "B": 0.7, "C": 0.6}
        d = quantilize_bandit_distribution(arm_probs, rewards, 0.5)
        assert sum(d.probs.values()) == pytest.approx(1.0)
        assert d.realised_kl <= d.kl_bound + 1e-9

    def test_quantilize_policy_improvement(self):
        deployed = {"a": 0.6, "b": 0.4}
        score = {"a": 0.3, "b": 0.8}
        d = quantilize_policy_improvement(deployed, score, kl_budget=0.1)
        assert d.realised_kl == pytest.approx(0.1, abs=1e-6)
        assert d.probs["b"] > d.probs["a"] / 0.6 * 0.4  # shifted toward b


# =====================================================================
# Constants and known events
# =====================================================================


class TestConstants:
    def test_algorithms(self):
        assert HARD in KNOWN_ALGORITHMS
        assert SOFT in KNOWN_ALGORITHMS
        assert TOP_K in KNOWN_ALGORITHMS
        assert SAMPLE in KNOWN_ALGORITHMS

    def test_lcb_methods(self):
        for m in (HOEFFDING, BERNSTEIN, ANYTIME, DKW):
            assert m in KNOWN_LCB_METHODS

    def test_known_events(self):
        for e in (QUANTILIZER_STARTED, QUANTILIZER_SELECTED,
                  QUANTILIZER_QUANTILIZED, QUANTILIZER_OBSERVED,
                  QUANTILIZER_REPORT):
            assert e in KNOWN_EVENTS


# =====================================================================
# End-to-end safety scenario
# =====================================================================


class TestGoodhartScenario:
    """A Goodhart-style scenario: proxy utility correlates with true
    utility most of the time but flips sign on a small slice of the
    base distribution.  The argmax selects the flipped slice.  The
    quantilizer with q ≥ flipped-slice-mass dilutes the bad action.
    """

    def test_safe_quantilizer_dilutes_goodhart_action(self):
        # 100-atom support, base uniform.
        # Proxy utility = 1.0 for atom 0 (the Goodhart trap),
        # ranks all others by an id-based score.
        # True utility = 0.0 for atom 0, 1.0 for others.
        n = 100
        base = {f"a{i}": 1.0 / n for i in range(n)}
        proxy = {f"a{i}": (10.0 if i == 0 else i / n) for i in range(n)}
        true_u = {f"a{i}": (0.0 if i == 0 else 1.0) for i in range(n)}

        # q = 0.01: pick only the argmax — falls into the Goodhart trap.
        d_unsafe = quantilize_discrete(base, proxy, 0.01)
        assert list(d_unsafe.probs.keys())[0] == "a0"
        exp_true_unsafe = sum(p * true_u[a] for a, p in d_unsafe.probs.items())
        assert exp_true_unsafe == pytest.approx(0.0)

        # q = 0.1: dilutes 10x, true utility recovers.
        d_safe = quantilize_discrete(base, proxy, 0.1)
        exp_true_safe = sum(p * true_u[a] for a, p in d_safe.probs.items())
        assert exp_true_safe > 0.5

    def test_cost_amplification_bound_certified(self):
        # Quantizer's worst-case hidden cost is 1/q × base cost.
        n = 50
        base = {f"a{i}": 1.0 / n for i in range(n)}
        proxy = {f"a{i}": float(i) for i in range(n)}
        for q in (0.5, 0.2, 0.1, 0.05, 0.02):
            d = quantilize_discrete(base, proxy, q)
            assert d.cost_amplification == pytest.approx(1.0 / q)
            assert d.kl_bound == pytest.approx(-math.log(q))


# =====================================================================
# Thread-safety smoke test
# =====================================================================


class TestConcurrency:
    def test_concurrent_observations(self):
        import threading
        Q = Quantilizer()
        def worker(seed: int):
            r = random.Random(seed)
            for _ in range(100):
                Q.observe(f"a{r.randint(0, 4)}", r.random())
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert Q.report().n_observations == 400


# =====================================================================
# End-to-end: realistic workflow
# =====================================================================


class TestEndToEnd:
    def test_full_workflow(self):
        # Coordinator:
        # 1. Builds a base distribution over plan variants
        # 2. Scores them with a proxy (e.g., LLM-judged quality)
        # 3. Quantilizes with q = 0.1 → KL bound = log(10) nats
        # 4. Selects one plan, executes, observes the true outcome
        # 5. Asks for a finite-sample LCB on quantilizer's expected utility
        Q = Quantilizer(q=0.1, default_lcb_method=ANYTIME)
        plans = {f"plan_{i}": 1.0 / 10 for i in range(10)}
        proxy_scores = {f"plan_{i}": random.Random(i).random() for i in range(10)}
        for trial in range(50):
            s = Q.select(plans, proxy_scores, seed=trial)
            # Simulate "true" outcome.
            true_u = 0.5 + 0.4 * random.Random(s.seed * 7 + 1).random()
            Q.observe(s.action, true_u)
        b = Q.expected_utility_lcb()
        assert b.mean > 0.4 and b.mean < 1.0
        assert b.lcb < b.mean < b.ucb
        r = Q.report()
        assert r.n_selections == 50
        assert r.n_observations == 50
        # Cumulative KL bound = 50 × log(10) nats; realised could be smaller.
        assert r.cumulative_kl_bound <= 50 * math.log(10.0) + 1e-6

    def test_fingerprint_chain_chains(self):
        Q = Quantilizer(q=0.5)
        base = {"a": 0.5, "b": 0.5}
        utility = {"a": 1.0, "b": 0.0}
        fps = [Q.fingerprint]
        for s in range(5):
            Q.select(base, utility, seed=s)
            fps.append(Q.fingerprint)
        # Each fingerprint is unique.
        assert len(set(fps)) == 6
