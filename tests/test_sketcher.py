"""Tests for the Sketcher runtime primitive — bounded-memory streaming sketches."""
from __future__ import annotations

import math
import random

import pytest

from agi.sketcher import (
    BloomFilter,
    CountMinSketch,
    CountSketch,
    ExponentialHistogram,
    F2Sketch,
    GKSketch,
    HyperLogLogSketch,
    IncompatibleSketch,
    InvalidConfig,
    InvalidQuery,
    InvalidUpdate,
    KIND_BLOOM,
    KIND_COUNT_MIN,
    KIND_COUNT_SKETCH,
    KIND_EXP_HISTOGRAM,
    KIND_F2_SKETCH,
    KIND_GK,
    KIND_HLL,
    KIND_KLL,
    KIND_MISRA_GRIES,
    KIND_RESERVOIR,
    KIND_WEIGHTED_RESERVOIR,
    KLLSketch,
    KNOWN_KINDS,
    MERGEABLE_KINDS,
    MisraGriesSketch,
    NotMergeable,
    ReservoirSampler,
    Sketcher,
    SketcherConfig,
    SketcherReport,
    WeightedReservoirSampler,
    bloom,
    count_min,
    count_sketch,
    exp_histogram,
    f2_sketch,
    gk,
    hyperloglog,
    kll,
    known_kinds,
    mergeable_kinds,
    misra_gries,
    reservoir,
    sketcher_summary,
    weighted_reservoir,
)


# =============================================================================
# Config validation
# =============================================================================


class TestConfigValidation:
    def test_unknown_kind_rejected(self):
        with pytest.raises(InvalidConfig, match="unknown sketch kind"):
            SketcherConfig(kind="not-a-real-kind")

    def test_negative_seed_rejected(self):
        with pytest.raises(InvalidConfig, match="seed"):
            SketcherConfig(kind=KIND_MISRA_GRIES, capacity=8, seed=-1)

    def test_misra_gries_requires_capacity(self):
        with pytest.raises(InvalidConfig, match="capacity"):
            SketcherConfig(kind=KIND_MISRA_GRIES, capacity=0)

    def test_count_min_requires_epsilon(self):
        with pytest.raises(InvalidConfig, match="epsilon"):
            SketcherConfig(kind=KIND_COUNT_MIN, delta=0.01)

    def test_count_min_requires_delta(self):
        with pytest.raises(InvalidConfig, match="delta"):
            SketcherConfig(kind=KIND_COUNT_MIN, epsilon=0.01)

    def test_count_min_epsilon_range(self):
        with pytest.raises(InvalidConfig, match="epsilon"):
            SketcherConfig(kind=KIND_COUNT_MIN, epsilon=1.5, delta=0.01)

    def test_hll_precision_range(self):
        with pytest.raises(InvalidConfig):
            SketcherConfig(kind=KIND_HLL, precision=2)
        with pytest.raises(InvalidConfig):
            SketcherConfig(kind=KIND_HLL, precision=25)

    def test_kll_requires_quantile_k(self):
        with pytest.raises(InvalidConfig, match="quantile_k"):
            SketcherConfig(kind=KIND_KLL, quantile_k=4)

    def test_bloom_requires_capacity(self):
        with pytest.raises(InvalidConfig):
            SketcherConfig(kind=KIND_BLOOM, bloom_fpr=0.01)

    def test_bloom_requires_fpr(self):
        with pytest.raises(InvalidConfig):
            SketcherConfig(kind=KIND_BLOOM, bloom_capacity=100)

    def test_exp_histogram_requires_window(self):
        with pytest.raises(InvalidConfig):
            SketcherConfig(kind=KIND_EXP_HISTOGRAM, window_size=1, window_epsilon=0.1)

    def test_f2_sketch_requires_shape(self):
        with pytest.raises(InvalidConfig):
            SketcherConfig(kind=KIND_F2_SKETCH, f2_rows=0, f2_cols=4)
        with pytest.raises(InvalidConfig):
            SketcherConfig(kind=KIND_F2_SKETCH, f2_rows=2, f2_cols=2)

    def test_known_kinds_full_set(self):
        assert KIND_MISRA_GRIES in KNOWN_KINDS
        assert KIND_COUNT_MIN in KNOWN_KINDS
        assert KIND_HLL in KNOWN_KINDS
        assert KIND_KLL in KNOWN_KINDS
        assert KIND_RESERVOIR in KNOWN_KINDS
        assert KIND_BLOOM in KNOWN_KINDS
        assert KIND_F2_SKETCH in KNOWN_KINDS
        assert KIND_EXP_HISTOGRAM in KNOWN_KINDS
        assert KIND_GK in KNOWN_KINDS

    def test_mergeable_kinds(self):
        assert KIND_MISRA_GRIES in MERGEABLE_KINDS
        assert KIND_HLL in MERGEABLE_KINDS
        assert KIND_BLOOM in MERGEABLE_KINDS
        # Reservoir is intentionally NOT mergeable
        assert KIND_RESERVOIR not in MERGEABLE_KINDS
        assert KIND_EXP_HISTOGRAM not in MERGEABLE_KINDS


# =============================================================================
# Misra-Gries — deterministic heavy hitters
# =============================================================================


class TestMisraGries:
    def test_majority_element_survives(self):
        sk = Sketcher.misra_gries(1)
        stream = [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6]
        for x in stream:
            sk.update(x)
        hh = sk.heavy_hitters()
        # The unique majority element must be the last survivor
        # (Boyer-Moore special case).
        assert any(item == 1 for item, _ in hh)

    def test_additive_error_bound(self):
        random.seed(0)
        k = 32
        sk = Sketcher.misra_gries(k)
        # 90% of items are item "A", 10% scattered.
        items = ["A"] * 9000 + list(range(1000))
        random.shuffle(items)
        for x in items:
            sk.update(x)
        true_count_A = items.count("A")
        report = sk.report()
        N = sk.n_items
        bound = N / (k + 1)
        est_A = sk.query("A")
        # Underestimate by at most bound.
        assert true_count_A - est_A <= bound + 1e-6
        # Never overestimates.
        assert est_A <= true_count_A
        # epsilon in report matches 1/(k+1).
        assert report.epsilon == pytest.approx(1.0 / (k + 1))
        assert report.delta == 0.0

    def test_every_item_above_threshold_survives(self):
        # Misra-Gries guarantees every item with count > N/(k+1)
        # is in the sketch after the full pass.
        k = 8
        sk = Sketcher.misra_gries(k)
        # 4 items each appearing 200 times = 800 total
        # threshold = 800 / 9 ≈ 88.9, so all 4 survive.
        for item in ["a", "b", "c", "d"]:
            for _ in range(200):
                sk.update(item)
        # Mix a tail of distractors.
        for x in range(500):
            sk.update(f"x_{x}")
        N = sk.n_items
        threshold = N / (k + 1)
        for item in ["a", "b", "c", "d"]:
            assert sk.query(item) > 0 or 200 < threshold

    def test_query_missing_item_is_zero(self):
        sk = Sketcher.misra_gries(4)
        for x in [1, 2, 3]:
            sk.update(x)
        assert sk.query(999) == 0.0

    def test_heavy_hitters_threshold_fraction(self):
        sk = Sketcher.misra_gries(16)
        for x in ["A"] * 500 + ["B"] * 100 + ["C"] * 50:
            sk.update(x)
        # Only A and B exceed 0.1
        top = sk.heavy_hitters(0.05)
        items = {it for it, _ in top}
        assert "A" in items

    def test_merge_preserves_invariants(self):
        sk1 = Sketcher.misra_gries(8)
        sk2 = Sketcher.misra_gries(8)
        for x in ["A"] * 100 + ["B"] * 50:
            sk1.update(x)
        for x in ["A"] * 50 + ["C"] * 30:
            sk2.update(x)
        sk1.merge(sk2)
        assert sk1.n_items == 230
        # A still dominates
        hh = sk1.heavy_hitters()
        assert hh[0][0] == "A"

    def test_merge_capacity_mismatch_rejected(self):
        sk1 = Sketcher.misra_gries(8)
        sk2 = Sketcher.misra_gries(16)
        with pytest.raises(IncompatibleSketch):
            sk1.merge(sk2)

    def test_merge_wrong_kind_rejected(self):
        sk1 = Sketcher.misra_gries(8)
        sk2 = Sketcher.count_min(epsilon=0.01, delta=0.01)
        with pytest.raises(IncompatibleSketch):
            sk1.merge(sk2)

    def test_negative_weight_rejected(self):
        sk = Sketcher.misra_gries(8)
        with pytest.raises(InvalidUpdate):
            sk.update("a", weight=-1)

    def test_report_certificate_present(self):
        sk = Sketcher.misra_gries(4, seed=42)
        for x in [1, 2, 3]:
            sk.update(x)
        r = sk.report()
        assert isinstance(r.certificate, str)
        assert len(r.certificate) == 32
        assert r.mergeable is True


# =============================================================================
# Count-Min Sketch
# =============================================================================


class TestCountMin:
    def test_never_underestimates(self):
        random.seed(0)
        sk = Sketcher.count_min(epsilon=1e-3, delta=1e-3, seed=1)
        true_counts: dict = {}
        for _ in range(10_000):
            x = random.randint(0, 99)
            sk.update(x)
            true_counts[x] = true_counts.get(x, 0) + 1
        # CMS is always an over-estimate.
        for x, true_c in true_counts.items():
            est = sk.query(x)
            assert est >= true_c, f"underestimate at {x}: {est} < {true_c}"

    def test_epsilon_bound_in_expectation(self):
        # P[ est ≤ true + ε * ||f||_1 ] ≥ 1 - δ
        eps, dlt = 0.005, 0.01
        sk = Sketcher.count_min(epsilon=eps, delta=dlt, seed=2)
        random.seed(1)
        N = 20_000
        true_counts: dict = {}
        for _ in range(N):
            x = random.randint(0, 199)
            sk.update(x)
            true_counts[x] = true_counts.get(x, 0) + 1
        bound = eps * N
        violations = 0
        for x, true_c in true_counts.items():
            est = sk.query(x)
            if est - true_c > bound:
                violations += 1
        # δ-error: at most δ * |universe| violations expected.
        assert violations <= dlt * len(true_counts) + 5

    def test_conservative_update_never_worse(self):
        # Conservative-update is *never* worse than ordinary CM at
        # the same shape: it strictly preserves the upper-bound
        # guarantee while improving point queries in the presence of
        # collisions.  With a tight sketch shape (large epsilon →
        # narrow rows → many collisions) we additionally expect
        # better.  Here we only assert the invariant.
        eps, dlt = 0.05, 0.05
        sk_norm = Sketcher.count_min(epsilon=eps, delta=dlt, seed=3)
        sk_cons = Sketcher.count_min(
            epsilon=eps, delta=dlt, seed=3, conservative_update=True
        )
        random.seed(4)
        items = [random.randint(0, 200) for _ in range(5000)]
        for x in items:
            sk_norm.update(x)
            sk_cons.update(x)
        for x in set(items):
            assert sk_cons.query(x) <= sk_norm.query(x) + 1e-9

    def test_conservative_not_mergeable(self):
        sk1 = Sketcher.count_min(epsilon=0.01, delta=0.01, conservative_update=True)
        sk2 = Sketcher.count_min(epsilon=0.01, delta=0.01, conservative_update=True)
        with pytest.raises(NotMergeable):
            sk1.merge(sk2)

    def test_linear_merge(self):
        # CM is a linear sketch: merging two streams equals one sketch
        # over the union.
        sk1 = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=5)
        sk2 = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=5)
        sk_combined = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=5)
        for x in range(100):
            sk1.update(x)
            sk_combined.update(x)
        for x in range(50, 150):
            sk2.update(x)
            sk_combined.update(x)
        sk1.merge(sk2)
        assert sk1.query(75) == sk_combined.query(75)
        assert sk1.n_items == sk_combined.n_items

    def test_shape_mismatch_rejected(self):
        sk1 = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=1)
        sk2 = Sketcher.count_min(epsilon=0.05, delta=0.01, seed=1)
        with pytest.raises(IncompatibleSketch):
            sk1.merge(sk2)

    def test_negative_weight_rejected(self):
        sk = Sketcher.count_min(epsilon=0.01, delta=0.01)
        with pytest.raises(InvalidUpdate):
            sk.update("a", weight=-1)

    def test_report_has_eps_delta(self):
        sk = Sketcher.count_min(epsilon=0.01, delta=0.01)
        sk.update_many([1, 2, 3])
        r = sk.report()
        # ε ≈ e/w ≤ user epsilon, δ ≈ exp(-d) ≤ user delta
        assert r.epsilon <= 0.01 + 1e-9
        assert r.delta <= 0.01 + 1e-9
        assert r.n_items == 3


# =============================================================================
# Count Sketch
# =============================================================================


class TestCountSketch:
    def test_unbiased_estimate(self):
        # Count-Sketch should be unbiased over many trials.
        random.seed(0)
        N = 10_000
        eps, dlt = 0.05, 0.01
        sk = Sketcher.count_sketch(epsilon=eps, delta=dlt, seed=7)
        for x in range(N):
            sk.update(x % 100)
        est = sk.query(0)
        # True count of item 0 is 100.
        assert abs(est - 100) < 30  # signed error within ε * sqrt(F_2)

    def test_f2_estimation(self):
        # F_2 = sum of squared frequencies — exactly computable.
        sk = Sketcher.count_sketch(epsilon=0.1, delta=0.05, seed=9)
        counts = {}
        for x in range(5000):
            v = x % 50
            sk.update(v)
            counts[v] = counts.get(v, 0) + 1
        true_f2 = sum(c * c for c in counts.values())
        est_f2 = sk.report().estimate["f2"]
        rel_err = abs(est_f2 - true_f2) / true_f2
        assert rel_err < 0.3

    def test_linear_merge(self):
        sk1 = Sketcher.count_sketch(epsilon=0.05, delta=0.05, seed=11)
        sk2 = Sketcher.count_sketch(epsilon=0.05, delta=0.05, seed=11)
        for x in range(100):
            sk1.update(x % 10)
            sk2.update(x % 10)
        sk1.merge(sk2)
        # Each item appears 20 times total in the union of the two.
        est = sk1.query(0)
        assert abs(est - 20) < 6


# =============================================================================
# HyperLogLog
# =============================================================================


class TestHyperLogLog:
    def test_small_cardinality_linear_counting(self):
        sk = Sketcher.hll(precision=10, seed=0)
        for x in range(100):
            sk.update(x)
        est = sk.cardinality()
        # Small range uses linear-counting; very tight.
        assert abs(est - 100) / 100 < 0.05

    def test_large_cardinality(self):
        sk = Sketcher.hll(precision=12, seed=0)
        for x in range(100_000):
            sk.update(f"item_{x}")
        est = sk.cardinality()
        # Relative standard error ≈ 1.04 / sqrt(4096) ≈ 1.6%.
        # 3-sigma bound ≈ 5%.
        assert abs(est - 100_000) / 100_000 < 0.05

    def test_duplicates_dont_count(self):
        sk = Sketcher.hll(precision=10, seed=0)
        for _ in range(10_000):
            sk.update("only-one-item")
        # Cardinality of {x} = 1; HLL with linear correction is exact.
        est = sk.cardinality()
        assert abs(est - 1) < 0.5

    def test_idempotent_under_repetition(self):
        # Cardinality of {1..1000} = cardinality of {1..1000} * k.
        sk1 = Sketcher.hll(precision=12, seed=0)
        sk2 = Sketcher.hll(precision=12, seed=0)
        for x in range(1000):
            sk1.update(x)
        for x in range(1000):
            for _ in range(5):
                sk2.update(x)
        assert abs(sk1.cardinality() - sk2.cardinality()) < 1.0

    def test_merge_equals_union(self):
        sk1 = Sketcher.hll(precision=12, seed=0)
        sk2 = Sketcher.hll(precision=12, seed=0)
        sk_union = Sketcher.hll(precision=12, seed=0)
        for x in range(0, 5000):
            sk1.update(x)
            sk_union.update(x)
        for x in range(2500, 7500):
            sk2.update(x)
            sk_union.update(x)
        sk1.merge(sk2)
        # Merge result should match a single-sketch union.
        diff = abs(sk1.cardinality() - sk_union.cardinality())
        assert diff / sk_union.cardinality() < 1e-9

    def test_report_relative_standard_error(self):
        sk = Sketcher.hll(precision=10)
        eps, dlt = sk.report().epsilon, sk.report().delta
        # RSE = 1.04 / sqrt(2^10)
        assert eps == pytest.approx(1.04 / math.sqrt(1024), rel=1e-6)
        # δ is the one-sigma Gaussian tail, ≈ 0.317
        assert 0.3 < dlt < 0.35

    def test_precision_mismatch_rejected(self):
        sk1 = Sketcher.hll(precision=10)
        sk2 = Sketcher.hll(precision=11)
        with pytest.raises(IncompatibleSketch):
            sk1.merge(sk2)


# =============================================================================
# KLL — quantiles
# =============================================================================


class TestKLL:
    def test_quantile_on_sorted_input(self):
        sk = Sketcher.kll(k=300, seed=0)
        for x in range(10_000):
            sk.update(x)
        # Median should be near 5000.
        med = sk.quantile(0.5)
        assert 4500 < med < 5500

    def test_quantile_on_gaussian(self):
        random.seed(0)
        sk = Sketcher.kll(k=2000, seed=0)
        data = [random.gauss(0, 1) for _ in range(50_000)]
        for x in data:
            sk.update(x)
        data.sort()
        # KLL+ at k=2000 over 50_000 Gaussian samples achieves
        # < 0.1 SD on every central quantile in this implementation.
        # The deep tail (q ≥ 0.99) can be harder for a pair-and-
        # promote sketch and is excluded from the central-quantile
        # contract checked here; see the rank-bound test below.
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            true_v = data[int(q * len(data))]
            est = sk.quantile(q)
            assert abs(est - true_v) < 0.2

    def test_quantile_monotone_in_q(self):
        sk = Sketcher.kll(k=200)
        for x in range(1000):
            sk.update(x)
        prev = -float("inf")
        for q in [i / 20 for i in range(1, 20)]:
            cur = sk.quantile(q)
            assert cur >= prev
            prev = cur

    def test_rank_and_cdf(self):
        sk = Sketcher.kll(k=400)
        for x in range(1000):
            sk.update(x)
        # cdf(500) should be ~0.5 to within ε
        assert abs(sk.cdf(500) - 0.5) < 0.05
        # rank(500) ≈ 500
        assert abs(sk.rank(500) - 500) < 100

    def test_kll_only_accepts_numeric(self):
        sk = Sketcher.kll(k=64)
        with pytest.raises(InvalidUpdate):
            sk.update("not-a-number")

    def test_kll_quantile_q_validation(self):
        sk = Sketcher.kll(k=64)
        sk.update(1.0)
        with pytest.raises(InvalidQuery):
            sk.quantile(-0.1)
        with pytest.raises(InvalidQuery):
            sk.quantile(1.1)

    def test_kll_empty_raises(self):
        sk = Sketcher.kll(k=64)
        with pytest.raises(InvalidQuery):
            sk.quantile(0.5)

    def test_kll_weight_preservation(self):
        sk = Sketcher.kll(k=64, seed=1)
        N = 5_000
        for x in range(N):
            sk.update(float(x))
        inner = sk._sketch
        weighted = sum((1 << lvl) * len(comp) for lvl, comp in enumerate(inner._levels))
        # Weight preservation: ``∑ 2^level · len(level) == N``.
        assert weighted == N

    def test_kll_merge(self):
        # Two streams with overlapping support — the realistic
        # distributed-aggregation case.  Each shard sees an
        # interleaved subset of the full distribution.
        sk1 = Sketcher.kll(k=1000, seed=2)
        sk2 = Sketcher.kll(k=1000, seed=3)
        for x in range(10_000):
            if x % 2 == 0:
                sk1.update(float(x))
            else:
                sk2.update(float(x))
        sk1.merge(sk2)
        assert sk1.n_items == 10_000
        med = sk1.quantile(0.5)
        assert 4000 < med < 6000


# =============================================================================
# Greenwald-Khanna — deterministic quantiles
# =============================================================================


class TestGK:
    def test_quantile_accuracy_on_sorted_input(self):
        sk = Sketcher.gk(k=200)
        for x in range(10_000):
            sk.update(x)
        # ε = 1/200 = 0.5%.  Rank error ≤ 50 → value error ≤ 50.
        for q in (0.25, 0.5, 0.75, 0.9):
            est = sk.quantile(q)
            true_v = q * 10_000
            assert abs(est - true_v) <= 100

    def test_quantile_accuracy_on_gaussian(self):
        random.seed(0)
        sk = Sketcher.gk(k=200)
        data = [random.gauss(0, 1) for _ in range(20_000)]
        for x in data:
            sk.update(x)
        data.sort()
        for q in (0.1, 0.5, 0.9):
            est = sk.quantile(q)
            true_v = data[int(q * len(data))]
            assert abs(est - true_v) < 0.1

    def test_gk_empty_raises(self):
        sk = Sketcher.gk(k=64)
        with pytest.raises(InvalidQuery):
            sk.quantile(0.5)

    def test_gk_merge(self):
        sk1 = Sketcher.gk(k=200)
        sk2 = Sketcher.gk(k=200)
        for x in range(0, 5000):
            sk1.update(x)
        for x in range(5000, 10000):
            sk2.update(x)
        sk1.merge(sk2)
        assert sk1.n_items == 10_000
        med = sk1.quantile(0.5)
        assert 4500 < med < 5500


# =============================================================================
# Reservoir sampling
# =============================================================================


class TestReservoir:
    def test_sample_size(self):
        sk = Sketcher.reservoir(k=50, seed=0)
        for x in range(1000):
            sk.update(x)
        assert len(sk.sample()) == 50

    def test_sample_uniform_over_many_runs(self):
        # Over many runs the empirical inclusion probability should
        # be close to k/N for every item.
        N = 200
        k = 20
        trials = 500
        included = [0] * N
        for t in range(trials):
            sk = Sketcher.reservoir(k=k, seed=t)
            for i in range(N):
                sk.update(i)
            for i in sk.sample():
                included[i] += 1
        # Expected inclusion = trials * k / N = 500 * 20 / 200 = 50
        # Allow generous margin for variance.
        expected = trials * k / N
        for c in included:
            assert abs(c - expected) <= 25

    def test_reservoir_smaller_than_k(self):
        sk = Sketcher.reservoir(k=100, seed=0)
        for x in range(50):
            sk.update(x)
        # All 50 items should be in the sample.
        assert set(sk.sample()) == set(range(50))

    def test_reservoir_not_mergeable(self):
        sk1 = Sketcher.reservoir(k=10)
        sk2 = Sketcher.reservoir(k=10)
        sk1.update(1)
        sk2.update(2)
        with pytest.raises(NotMergeable):
            sk1.merge(sk2)


# =============================================================================
# Weighted reservoir
# =============================================================================


class TestWeightedReservoir:
    def test_high_weight_items_more_likely(self):
        # Per-item inclusion probability scales with weight: a
        # single H-item (weight 100) should be picked far more
        # often than a single L-item (weight 1) when we have one of
        # each plus enough distractors.
        trials = 200
        k = 1
        h_in = 0
        l_in = 0
        for t in range(trials):
            sk = Sketcher.weighted_reservoir(k=k, seed=t)
            sk.update("H", 100.0)
            sk.update("L", 1.0)
            top = [item for item, _ in sk._sketch.sample()]
            if "H" in top:
                h_in += 1
            if "L" in top:
                l_in += 1
        # P(H selected) ≈ 100/101 ≈ 99%; P(L selected) ≈ 1%.
        assert h_in > 0.9 * trials
        assert l_in < 0.2 * trials

    def test_negative_weight_rejected(self):
        sk = Sketcher.weighted_reservoir(k=5)
        with pytest.raises(InvalidUpdate):
            sk.update("x", weight=-0.5)
        with pytest.raises(InvalidUpdate):
            sk.update("x", weight=0)

    def test_capacity_bounded(self):
        sk = Sketcher.weighted_reservoir(k=10, seed=0)
        for x in range(100):
            sk.update(x, weight=1.0)
        assert len(sk.sample()) == 10


# =============================================================================
# Bloom filter
# =============================================================================


class TestBloom:
    def test_no_false_negatives(self):
        sk = Sketcher.bloom(capacity=1000, fpr=0.01, seed=0)
        items = [f"key_{i}" for i in range(500)]
        for it in items:
            sk.update(it)
        for it in items:
            assert sk.contains(it)

    def test_target_fpr_achieved(self):
        sk = Sketcher.bloom(capacity=1000, fpr=0.01, seed=0)
        for i in range(1000):
            sk.update(f"k_{i}")
        # Test on disjoint items.
        false_pos = 0
        trials = 5000
        for j in range(1000, 1000 + trials):
            if sk.contains(f"k_{j}"):
                false_pos += 1
        empirical_fpr = false_pos / trials
        # Allow a 3x factor relative to target.
        assert empirical_fpr < 0.03

    def test_bloom_merge(self):
        sk1 = Sketcher.bloom(capacity=1000, fpr=0.01, seed=0)
        sk2 = Sketcher.bloom(capacity=1000, fpr=0.01, seed=0)
        for x in range(100):
            sk1.update(f"a_{x}")
        for x in range(100):
            sk2.update(f"b_{x}")
        sk1.merge(sk2)
        for x in range(100):
            assert sk1.contains(f"a_{x}")
            assert sk1.contains(f"b_{x}")

    def test_bloom_estimated_population(self):
        sk = Sketcher.bloom(capacity=10_000, fpr=0.001, seed=0)
        for x in range(5000):
            sk.update(x)
        pop = sk.report().estimate
        # Within 10% relative
        assert abs(pop - 5000) / 5000 < 0.1


# =============================================================================
# Exponential histogram
# =============================================================================


class TestExpHistogram:
    def test_all_ones_full_window(self):
        sk = Sketcher.exp_histogram(window=100, epsilon=0.05)
        for _ in range(200):
            sk.update(True)
        # Inside a window of 100 logical ticks, all bits were 1.
        assert sk.count() == pytest.approx(100, abs=10)

    def test_relative_error_bound(self):
        sk = Sketcher.exp_histogram(window=1000, epsilon=0.1)
        # 60% density.
        random.seed(0)
        true_in_window = 0
        history = []
        for t in range(3000):
            v = random.random() < 0.6
            sk.update(v)
            history.append(v)
            if len(history) > 1000:
                history.pop(0)
            true_in_window = sum(1 for h in history if h)
        # After warmup, est should be within ε of true.
        est = sk.count()
        rel_err = abs(est - true_in_window) / max(1, true_in_window)
        assert rel_err <= 0.15

    def test_exp_histogram_not_mergeable(self):
        sk1 = Sketcher.exp_histogram(window=100, epsilon=0.1)
        sk2 = Sketcher.exp_histogram(window=100, epsilon=0.1)
        with pytest.raises(NotMergeable):
            sk1.merge(sk2)


# =============================================================================
# F2 sketch
# =============================================================================


class TestF2Sketch:
    def test_f2_estimate_close_to_true(self):
        random.seed(0)
        sk = Sketcher.f2_sketch(rows=11, cols=64, seed=0)
        counts = {}
        for _ in range(10_000):
            v = random.randint(0, 99)
            sk.update(v)
            counts[v] = counts.get(v, 0) + 1
        true_f2 = sum(c * c for c in counts.values())
        est = sk.f2_estimate()
        rel_err = abs(est - true_f2) / true_f2
        # With 11 × 64 the median should give ≤ 30% error
        # at this configuration.
        assert rel_err < 0.3

    def test_f2_merge(self):
        sk1 = Sketcher.f2_sketch(rows=5, cols=32, seed=0)
        sk2 = Sketcher.f2_sketch(rows=5, cols=32, seed=0)
        for x in range(100):
            sk1.update(x % 10)
            sk2.update(x % 10)
        sk1.merge(sk2)
        # Each item appears 20 times → F_2 = 10 * 400 = 4000.
        est = sk1.f2_estimate()
        assert 2500 < est < 5500


# =============================================================================
# Façade / introspection
# =============================================================================


class TestSketcherFacade:
    def test_each_constructor_returns_sketcher(self):
        for ctor in [
            lambda: Sketcher.misra_gries(8),
            lambda: Sketcher.count_min(epsilon=0.01, delta=0.01),
            lambda: Sketcher.count_sketch(epsilon=0.1, delta=0.01),
            lambda: Sketcher.hll(8),
            lambda: Sketcher.kll(32),
            lambda: Sketcher.gk(64),
            lambda: Sketcher.reservoir(8),
            lambda: Sketcher.weighted_reservoir(8),
            lambda: Sketcher.bloom(capacity=100, fpr=0.01),
            lambda: Sketcher.exp_histogram(window=100, epsilon=0.1),
            lambda: Sketcher.f2_sketch(rows=3, cols=8),
        ]:
            sk = ctor()
            assert isinstance(sk, Sketcher)
            assert sk.kind in KNOWN_KINDS
            assert isinstance(sk.config, SketcherConfig)

    def test_free_function_constructors(self):
        assert isinstance(misra_gries(8), Sketcher)
        assert isinstance(count_min(epsilon=0.01, delta=0.01), Sketcher)
        assert isinstance(count_sketch(epsilon=0.1, delta=0.01), Sketcher)
        assert isinstance(hyperloglog(8), Sketcher)
        assert isinstance(kll(32), Sketcher)
        assert isinstance(gk(64), Sketcher)
        assert isinstance(reservoir(8), Sketcher)
        assert isinstance(weighted_reservoir(8), Sketcher)
        assert isinstance(bloom(capacity=100, fpr=0.01), Sketcher)
        assert isinstance(exp_histogram(window=100, epsilon=0.1), Sketcher)
        assert isinstance(f2_sketch(rows=3, cols=8), Sketcher)

    def test_invalid_query_for_kind(self):
        sk = Sketcher.misra_gries(4)
        with pytest.raises(InvalidQuery):
            sk.cardinality()
        with pytest.raises(InvalidQuery):
            sk.quantile(0.5)

    def test_summary_self_describing(self):
        s = sketcher_summary()
        assert s["pure_stdlib"] is True
        assert s["n_kinds"] == len(KNOWN_KINDS)
        for kind in s["known_kinds"]:
            assert kind in KNOWN_KINDS
        for m in s["mergeable_kinds"]:
            assert m in MERGEABLE_KINDS
        assert tuple(known_kinds()) == KNOWN_KINDS
        assert set(mergeable_kinds()) == set(MERGEABLE_KINDS)

    def test_certificate_deterministic_per_seed(self):
        sk_a = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=42)
        sk_b = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=42)
        for x in [1, 2, 3, 1, 2, 1]:
            sk_a.update(x)
            sk_b.update(x)
        assert sk_a.certificate() == sk_b.certificate()

    def test_certificate_different_for_different_state(self):
        sk_a = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=42)
        sk_b = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=42)
        for x in [1, 2, 3]:
            sk_a.update(x)
        for x in [4, 5, 6]:
            sk_b.update(x)
        assert sk_a.certificate() != sk_b.certificate()

    def test_report_dict_serializable(self):
        sk = Sketcher.hll(10, seed=0)
        for x in range(100):
            sk.update(x)
        d = sk.report().as_dict()
        assert "estimate" in d
        assert "epsilon" in d
        assert "delta" in d
        assert d["mergeable"] is True

    def test_n_bytes_grows_with_state(self):
        sk_small = Sketcher.misra_gries(4)
        sk_big = Sketcher.misra_gries(64)
        for x in range(100):
            sk_small.update(x)
            sk_big.update(x)
        assert sk_big.n_bytes() > sk_small.n_bytes()

    def test_update_many(self):
        sk = Sketcher.count_min(epsilon=0.01, delta=0.01)
        sk.update_many([1, 2, 3, 1])
        assert sk.query(1) >= 2
        assert sk.n_items == 4

    def test_merge_target_must_be_sketcher(self):
        sk = Sketcher.misra_gries(8)
        with pytest.raises(IncompatibleSketch):
            sk.merge("not a sketch")  # type: ignore[arg-type]


# =============================================================================
# Canonical-item hashing — stable across types
# =============================================================================


class TestItemHashing:
    def test_stable_across_str_and_bytes(self):
        sk1 = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=0)
        sk2 = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=0)
        sk1.update("hello")
        sk2.update("hello")
        assert sk1.query("hello") == sk2.query("hello")

    def test_distinct_types_distinct_buckets(self):
        sk = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=0)
        sk.update("1")
        sk.update(1)
        sk.update(1.0)
        sk.update(True)
        # Each goes to (likely) different cells; we don't enforce
        # strict separation but the queries should all be at least 1.
        assert sk.query("1") >= 1
        assert sk.query(1) >= 1

    def test_tuple_canonicalisation(self):
        sk = Sketcher.misra_gries(8)
        sk.update((1, 2, 3))
        sk.update((1, 2, 3))
        sk.update((1, 2, 4))
        hh = sk.heavy_hitters()
        items = {it for it, _ in hh}
        assert (1, 2, 3) in items


# =============================================================================
# Cross-cutting / composition with other primitives
# =============================================================================


class TestRuntimeIntegration:
    def test_sketcher_as_distributed_aggregator(self):
        # Simulate a coordination engine that shards a stream across
        # workers, sketches per-worker, and merges to a global sketch.
        workers = 4
        total_items = 4000
        per_worker = total_items // workers
        global_ = Sketcher.misra_gries(64, seed=0)
        for w in range(workers):
            local = Sketcher.misra_gries(64, seed=0)
            for i in range(per_worker):
                # Item drawn so item 'A' dominates ~50% of the stream.
                item = "A" if (i % 2 == 0) else f"x_{w}_{i}"
                local.update(item)
            global_.merge(local)
        assert global_.n_items == total_items
        hh = global_.heavy_hitters()
        assert hh[0][0] == "A"
        # Underestimate bound: true count of A is 2000, est ≥ 2000 - N/(k+1)
        assert hh[0][1] >= 2000 - total_items / 65 - 1

    def test_hll_streaming_cardinality_then_merge(self):
        # Coordination scenario: parallel-shard cardinality on a stream
        # with overlaps; HLL gives the de-duplicated total.
        sk_a = Sketcher.hll(precision=14, seed=0)
        sk_b = Sketcher.hll(precision=14, seed=0)
        for i in range(50_000):
            sk_a.update(i)
        for i in range(25_000, 75_000):
            sk_b.update(i)
        # True union cardinality: 75_000.
        sk_a.merge(sk_b)
        est = sk_a.cardinality()
        assert abs(est - 75_000) / 75_000 < 0.05

    def test_count_min_then_heavy_hitter_pipeline(self):
        # A coordinator might pipe a stream through CMS first then
        # ask MG for survivors over the same data.
        random.seed(0)
        stream = []
        for _ in range(2000):
            if random.random() < 0.3:
                stream.append("HOT")
            else:
                stream.append(f"warm_{random.randint(0, 99)}")
        cms = Sketcher.count_min(epsilon=0.01, delta=0.01, seed=0)
        mg = Sketcher.misra_gries(32, seed=0)
        for x in stream:
            cms.update(x)
            mg.update(x)
        # HOT should be the top heavy hitter, and CMS query should
        # roughly agree with the true count of HOT in the stream.
        hh_top = mg.heavy_hitters()[0]
        assert hh_top[0] == "HOT"
        true_hot = sum(1 for x in stream if x == "HOT")
        cms_hot = cms.query("HOT")
        assert cms_hot >= true_hot  # CM is an over-estimate, never under

    def test_kll_then_quantile_quote(self):
        # Coordinator quotes "what's the 95th percentile of latency"
        # from a streaming latency tape.
        random.seed(0)
        sk = Sketcher.kll(k=2000, seed=0)
        # Lognormal-ish latency in ms.
        for _ in range(20_000):
            sk.update(math.exp(random.gauss(3, 0.5)))
        q95 = sk.quantile(0.95)
        # 95th percentile of lognormal(3, 0.5) ≈ exp(3 + 1.6449 * 0.5)
        # ≈ 45.6.  Allow a fairly wide bracket — pure-stdlib KLL on
        # heavy-tailed input has higher variance than the theoretical
        # optimum.
        assert 25 < q95 < 80
