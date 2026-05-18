"""Tests for the Topologist primitive (TDA / persistent homology)."""
from __future__ import annotations

import math
import random

import pytest

from agi.events import EventBus
from agi.topologist import (
    METRIC_EUCLIDEAN,
    METRIC_PRECOMPUTED,
    TOPOLOGIST_BOOTSTRAPPED,
    TOPOLOGIST_CLEARED,
    TOPOLOGIST_COMPARED,
    TOPOLOGIST_COMPUTED,
    TOPOLOGIST_OBSERVED,
    TOPOLOGIST_REPORTED,
    TOPOLOGIST_STARTED,
    BootstrapBand,
    ComplexTooLarge,
    DimensionMismatch,
    InsufficientData,
    InvalidConfig,
    InvalidMetric,
    InvalidPoint,
    PersistenceDiagram,
    PersistenceLandscape,
    PersistencePair,
    StabilityCertificate,
    Topologist,
    TopologistReport,
    chebyshev,
    cosine_distance,
    euclidean,
    hamming_distance,
    manhattan,
    sqeuclidean,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _circle(n: int, radius: float = 1.0, seed: int = 0) -> list[tuple[float, float]]:
    """``n`` points uniformly on a circle of given radius."""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        # Evenly spaced angles with a tiny jitter so ties don't dominate.
        theta = 2.0 * math.pi * k / n + 1e-9 * rng.random()
        out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return out


def _square_cluster(n: int, cx: float, cy: float, side: float, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [
        (cx + side * (rng.random() - 0.5), cy + side * (rng.random() - 0.5))
        for _ in range(n)
    ]


# ------------------------------------------------------------------
# Configuration / construction
# ------------------------------------------------------------------


class TestConstruction:
    def test_create_defaults(self):
        top = Topologist.create()
        assert top.n_points() == 0
        assert top.fingerprint() == "0" * 64 or len(top.fingerprint()) == 64

    def test_invalid_max_dim_negative(self):
        with pytest.raises(InvalidConfig):
            Topologist.create(max_dim=-1)

    def test_invalid_max_dim_too_large(self):
        with pytest.raises(InvalidConfig):
            Topologist.create(max_dim=10)

    def test_invalid_max_scale_zero(self):
        with pytest.raises(InvalidConfig):
            Topologist.create(max_scale=0.0)

    def test_invalid_max_scale_negative(self):
        with pytest.raises(InvalidConfig):
            Topologist.create(max_scale=-1.0)

    def test_max_scale_inf_ok(self):
        top = Topologist.create(max_scale=float("inf"))
        assert top.n_points() == 0

    def test_invalid_metric_name(self):
        with pytest.raises(InvalidMetric):
            Topologist.create(metric="lp42")

    def test_metric_callable_ok(self):
        top = Topologist.create(metric=euclidean)
        top.observe((0.0, 0.0))
        top.observe((1.0, 0.0))
        top.observe((0.0, 1.0))
        d = top.compute()
        assert d.n_points == 3

    def test_metric_non_str_non_callable_rejected(self):
        with pytest.raises(InvalidMetric):
            Topologist.create(metric=42)  # type: ignore[arg-type]

    def test_max_simplices_must_be_positive(self):
        with pytest.raises(InvalidConfig):
            Topologist.create(max_simplices=0)

    def test_max_points_must_be_positive(self):
        with pytest.raises(InvalidConfig):
            Topologist.create(max_points=0)

    def test_session_id_custom(self):
        top = Topologist.create(session_id="custom-123")
        assert top._session_id == "custom-123"

    def test_emits_started_event(self):
        bus = EventBus()
        Topologist.create(bus=bus, session_id="sess")
        events = bus.history(session_id="sess", kind=TOPOLOGIST_STARTED)
        assert len(events) == 1
        assert events[0].data["max_dim"] == 1
        assert events[0].data["metric"] == "euclidean"


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


class TestMetrics:
    def test_euclidean(self):
        assert euclidean((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
        assert euclidean((1.0,), (1.0,)) == 0.0

    def test_sqeuclidean(self):
        assert sqeuclidean((0.0, 0.0), (3.0, 4.0)) == pytest.approx(25.0)

    def test_manhattan(self):
        assert manhattan((0.0, 0.0), (3.0, 4.0)) == pytest.approx(7.0)

    def test_chebyshev(self):
        assert chebyshev((0.0, 0.0), (3.0, 4.0)) == pytest.approx(4.0)

    def test_cosine_zero_vs_zero(self):
        # Both zero: returns 0 (degenerate but well-defined)
        assert cosine_distance((0.0, 0.0), (0.0, 0.0)) == 0.0

    def test_cosine_orthogonal(self):
        assert cosine_distance((1.0, 0.0), (0.0, 1.0)) == pytest.approx(1.0)

    def test_cosine_aligned(self):
        assert cosine_distance((1.0, 0.0), (2.0, 0.0)) == pytest.approx(0.0)

    def test_cosine_antipodal(self):
        assert cosine_distance((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(2.0)

    def test_hamming(self):
        assert hamming_distance((1, 0, 1, 0), (1, 1, 0, 0)) == pytest.approx(0.5)

    def test_hamming_empty(self):
        assert hamming_distance((), ()) == 0.0


# ------------------------------------------------------------------
# Observation / input validation
# ------------------------------------------------------------------


class TestObservation:
    def test_observe_single(self):
        top = Topologist.create()
        pid = top.observe((1.0, 2.0))
        assert pid == "p0"
        assert top.n_points() == 1

    def test_observe_custom_id(self):
        top = Topologist.create()
        pid = top.observe((1.0, 2.0), point_id="alpha")
        assert pid == "alpha"

    def test_observe_batch(self):
        top = Topologist.create()
        ids = top.observe_batch([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        assert ids == ["p0", "p1", "p2"]
        assert top.n_points() == 3

    def test_dimension_mismatch_rejected(self):
        top = Topologist.create()
        top.observe((1.0, 2.0))
        with pytest.raises(DimensionMismatch):
            top.observe((1.0, 2.0, 3.0))

    def test_nan_rejected(self):
        top = Topologist.create()
        with pytest.raises(InvalidPoint):
            top.observe((float("nan"), 0.0))

    def test_inf_rejected(self):
        top = Topologist.create()
        with pytest.raises(InvalidPoint):
            top.observe((float("inf"), 0.0))

    def test_non_iterable_rejected(self):
        top = Topologist.create()
        with pytest.raises(InvalidPoint):
            top.observe(42.0)  # type: ignore[arg-type]

    def test_empty_point_rejected(self):
        top = Topologist.create()
        with pytest.raises(InvalidPoint):
            top.observe(())

    def test_max_points_enforced(self):
        top = Topologist.create(max_points=3)
        top.observe((0.0,))
        top.observe((1.0,))
        top.observe((2.0,))
        with pytest.raises(InvalidConfig):
            top.observe((3.0,))

    def test_observed_event_emitted(self):
        bus = EventBus()
        top = Topologist.create(bus=bus, session_id="s")
        top.observe((0.0, 0.0))
        events = bus.history(session_id="s", kind=TOPOLOGIST_OBSERVED)
        assert len(events) == 1
        assert events[0].data["n_points"] == 1

    def test_observe_distance_matrix(self):
        top = Topologist.create()
        dm = [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        top.observe_distance_matrix(dm)
        assert top.n_points() == 3

    def test_observe_distance_matrix_non_square(self):
        top = Topologist.create()
        with pytest.raises(InvalidConfig):
            top.observe_distance_matrix([[0.0, 1.0], [1.0]])

    def test_observe_distance_matrix_non_zero_diag(self):
        top = Topologist.create()
        with pytest.raises(InvalidConfig):
            top.observe_distance_matrix([[1.0, 0.0], [0.0, 1.0]])

    def test_observe_distance_matrix_asymmetric(self):
        top = Topologist.create()
        with pytest.raises(InvalidConfig):
            top.observe_distance_matrix([[0.0, 1.0], [2.0, 0.0]])

    def test_observe_distance_matrix_negative(self):
        top = Topologist.create()
        with pytest.raises(InvalidConfig):
            top.observe_distance_matrix([[0.0, -1.0], [-1.0, 0.0]])

    def test_observe_dm_then_observe_rejected(self):
        top = Topologist.create()
        top.observe_distance_matrix([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(InvalidConfig):
            top.observe((0.0, 0.0))

    def test_observe_then_dm_rejected(self):
        top = Topologist.create()
        top.observe((0.0,))
        with pytest.raises(InvalidConfig):
            top.observe_distance_matrix([[0.0, 1.0], [1.0, 0.0]])


# ------------------------------------------------------------------
# Dimension-0 persistence (connected components)
# ------------------------------------------------------------------


class TestDim0:
    def test_single_point(self):
        top = Topologist.create(max_dim=0)
        top.observe((0.0,))
        d = top.compute()
        pairs0 = d.diagram(0)
        # Only one essential class for a single point
        assert len(pairs0) == 1
        assert pairs0[0].is_infinite

    def test_two_close_points(self):
        top = Topologist.create(max_dim=0)
        top.observe((0.0,))
        top.observe((1.0,))
        d = top.compute()
        pairs0 = d.diagram(0)
        # One essential class + one (born 0, died 1)
        assert len(pairs0) == 2
        deaths = sorted(
            float("inf") if p.is_infinite else p.death for p in pairs0
        )
        assert deaths[0] == pytest.approx(1.0)
        assert math.isinf(deaths[1])

    def test_three_clusters_recovered(self):
        """Three well-separated clusters of 10 points each."""
        pts = []
        pts.extend(_square_cluster(10, 0.0, 0.0, 0.2, seed=1))
        pts.extend(_square_cluster(10, 10.0, 0.0, 0.2, seed=2))
        pts.extend(_square_cluster(10, 5.0, 8.0, 0.2, seed=3))
        top = Topologist.create(max_dim=0, max_scale=12.0, max_points=40)
        for p in pts:
            top.observe(p)
        d = top.compute()
        # Top-3 most persistent dim-0 features should include 3 essentials
        # OR: the 3 longest finite + 1 essential. The essential is born at 0
        # and never dies, so it always tops the list; below it the next two
        # most persistent finite pairs separate the clusters.
        top3 = d.k_most_persistent(0, 3)
        # All three should have persistence >= the cluster-side scale (0.2)
        # since the inter-cluster distance is ~ several units
        long_count = sum(
            1
            for p in top3
            if p.is_infinite or (p.death - p.birth) >= 1.0
        )
        assert long_count == 3

    def test_betti_at_small_scale(self):
        pts = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        top = Topologist.create(max_dim=0, max_scale=50.0)
        for p in pts:
            top.observe(p)
        d = top.compute()
        # At scale 5.0, no merges have happened — 3 components
        b = d.betti(5.0)
        assert b[0] == 3
        # At scale 100, all merged
        b = d.betti(100.0)
        assert b[0] == 1

    def test_dim0_no_max_scale_uses_diameter(self):
        pts = [(0.0,), (1.0,), (10.0,)]
        top = Topologist.create(max_dim=0)  # max_scale=inf
        for p in pts:
            top.observe(p)
        d = top.compute()
        # diameter = 10; all merges captured
        assert d.max_scale == pytest.approx(10.0)
        # 3 dim-0 pairs total: 2 finite deaths, 1 essential
        pairs0 = d.diagram(0)
        finite = [p for p in pairs0 if not p.is_infinite]
        assert len(finite) == 2
        assert len([p for p in pairs0 if p.is_infinite]) == 1

    def test_truncated_max_scale_persists_more(self):
        """Points further than max_scale don't merge."""
        pts = [(0.0,), (1.0,), (10.0,)]
        top = Topologist.create(max_dim=0, max_scale=2.0)
        for p in pts:
            top.observe(p)
        d = top.compute()
        # Two components survive: {0,1} merged at scale 1, {10} alone
        b = d.betti(1.5)
        assert b[0] == 2

    def test_compute_after_clear_works(self):
        top = Topologist.create(max_dim=0)
        top.observe((0.0,))
        top.observe((1.0,))
        top.compute()
        top.clear()
        with pytest.raises(InsufficientData):
            top.compute()
        top.observe((5.0,))
        d = top.compute()
        assert d.n_points == 1

    def test_clear_emits_event(self):
        bus = EventBus()
        top = Topologist.create(bus=bus, session_id="s")
        top.observe((0.0,))
        top.clear()
        events = bus.history(session_id="s", kind=TOPOLOGIST_CLEARED)
        assert len(events) == 1


# ------------------------------------------------------------------
# Dimension-1 persistence (loops)
# ------------------------------------------------------------------


class TestDim1:
    def test_loop_detected_on_circle(self):
        """A circle has one significant loop in dim-1."""
        pts = _circle(20, radius=1.0)
        top = Topologist.create(
            max_dim=1, max_scale=2.5, max_points=30, max_simplices=200_000
        )
        for p in pts:
            top.observe(p)
        d = top.compute()
        loops = d.diagram(1)
        # The one true loop should have far the longest persistence
        # of all dim-1 features.
        assert len(loops) >= 1
        persistences = sorted(
            (p.death - p.birth for p in loops if not p.is_infinite),
            reverse=True,
        )
        # The dominant loop persistence should be > some reasonable cutoff
        assert persistences[0] > 0.5
        # The second-longest (if any) should be much smaller
        if len(persistences) > 1:
            assert persistences[0] > 2.0 * persistences[1]

    def test_no_loop_in_cluster(self):
        """A blob has no significant dim-1 feature."""
        pts = _square_cluster(15, 0.0, 0.0, 0.2, seed=42)
        top = Topologist.create(max_dim=1, max_scale=1.0, max_points=30)
        for p in pts:
            top.observe(p)
        d = top.compute()
        loops = d.diagram(1)
        # Whatever dim-1 features exist should all have small persistence
        for p in loops:
            if not p.is_infinite:
                assert (p.death - p.birth) < 0.5

    def test_betti_one_inside_loop(self):
        """β_1 = 1 at a scale inside the loop's life."""
        pts = _circle(15, radius=1.0)
        top = Topologist.create(
            max_dim=1, max_scale=2.5, max_points=20, max_simplices=200_000
        )
        for p in pts:
            top.observe(p)
        d = top.compute()
        loops = d.diagram(1)
        # Find the longest loop and pick a scale inside (birth, death)
        long_loop = max(
            (p for p in loops if not p.is_infinite),
            key=lambda p: p.death - p.birth,
        )
        mid = 0.5 * (long_loop.birth + long_loop.death)
        b = d.betti(mid)
        assert b[1] >= 1

    def test_two_loops_in_figure_eight(self):
        """A figure-eight should expose two loops."""
        pts1 = _circle(12, radius=1.0)
        # second circle offset to the right
        pts2 = [(p[0] + 2.5, p[1]) for p in _circle(12, radius=1.0)]
        all_pts = pts1 + pts2
        top = Topologist.create(
            max_dim=1, max_scale=2.5, max_points=30, max_simplices=400_000
        )
        for p in all_pts:
            top.observe(p)
        d = top.compute()
        loops = d.diagram(1)
        # Top-2 loops should have comparable persistence; both > some cutoff.
        top2 = sorted(
            ((p.death - p.birth) for p in loops if not p.is_infinite),
            reverse=True,
        )[:2]
        assert len(top2) >= 2
        assert top2[0] > 0.3
        assert top2[1] > 0.3


# ------------------------------------------------------------------
# Persistence diagram API
# ------------------------------------------------------------------


class TestDiagramAPI:
    def test_barcode_sorted_by_birth(self):
        pts = [(0.0,), (1.0,), (5.0,), (5.5,)]
        top = Topologist.create(max_dim=0)
        for p in pts:
            top.observe(p)
        d = top.compute()
        bars = d.barcode(0)
        births = [b[0] for b in bars]
        assert births == sorted(births)

    def test_k_most_persistent_capped(self):
        pts = [(float(i),) for i in range(10)]
        top = Topologist.create(max_dim=0)
        for p in pts:
            top.observe(p)
        d = top.compute()
        top3 = d.k_most_persistent(0, 3)
        assert len(top3) == 3

    def test_total_persistence(self):
        pts = [(0.0,), (1.0,), (3.0,)]
        top = Topologist.create(max_dim=0)
        for p in pts:
            top.observe(p)
        d = top.compute()
        tp = d.total_persistence(0, 1.0)
        # Two finite dim-0 pairs: (0, 1) and (0, 2) → total 3
        # Actually depends on which root absorbed which; classical elder-rule
        # on (0,1,3) gives deaths {1, 2} (death of {0..1} merging is 1;
        # death of merging with 3 is the next edge 2). Total = 3.
        assert tp == pytest.approx(3.0)

    def test_landscape_shape(self):
        pts = _circle(15, radius=1.0)
        top = Topologist.create(
            max_dim=1, max_scale=2.5, max_points=20, max_simplices=200_000
        )
        for p in pts:
            top.observe(p)
        d = top.compute()
        ls = d.landscape(dim=1, num_levels=2, grid=32)
        assert isinstance(ls, PersistenceLandscape)
        assert ls.num_levels == 2
        assert len(ls.grid) == 32
        assert len(ls.levels) == 2
        # Each level is the right length
        for row in ls.levels:
            assert len(row) == 32
        # Top level non-trivial
        assert max(ls.levels[0]) > 0.0

    def test_landscape_empty_diagram(self):
        top = Topologist.create(max_dim=1, max_scale=10.0)
        top.observe((0.0, 0.0))
        d = top.compute()
        # No dim-1 features at all for a single point
        ls = d.landscape(dim=1, num_levels=2, grid=8)
        assert ls.grid == ()
        # All levels are empty
        for row in ls.levels:
            assert row == ()

    def test_landscape_norm_nonnegative(self):
        pts = _circle(10, radius=1.0)
        top = Topologist.create(
            max_dim=1, max_scale=2.5, max_points=15, max_simplices=200_000
        )
        for p in pts:
            top.observe(p)
        d = top.compute()
        ls = d.landscape(dim=1, num_levels=3, grid=32)
        assert ls.norm(p=2.0) >= 0.0
        # Vector concatenation is the right length
        v = ls.vector()
        assert len(v) == 32 * 3

    def test_landscape_invalid_args(self):
        d = PersistenceDiagram(
            pairs=(),
            max_dim=1,
            n_points=0,
            max_scale=0.0,
            metric="euclidean",
        )
        with pytest.raises(InvalidConfig):
            d.landscape(dim=1, num_levels=0)
        with pytest.raises(InvalidConfig):
            d.landscape(dim=1, num_levels=2, grid=1)

    def test_significant_features_above_threshold(self):
        """Features whose persistence > 2·threshold pass."""
        d = PersistenceDiagram(
            pairs=(
                PersistencePair(dim=0, birth=0.0, death=1.0),  # persistence 1
                PersistencePair(dim=0, birth=0.0, death=0.05),  # persistence 0.05
                PersistencePair(dim=0, birth=0.0, death=float("inf")),  # essential
            ),
            max_dim=0,
            n_points=3,
            max_scale=2.0,
            metric="euclidean",
        )
        sig = d.significant_features(0, threshold=0.1)
        # threshold * 2 = 0.2; only persistence > 0.2 passes (+ essentials)
        deaths = sorted(
            float("inf") if p.is_infinite else p.death for p in sig
        )
        assert 1.0 in deaths
        assert math.isinf(deaths[-1])
        assert 0.05 not in deaths


# ------------------------------------------------------------------
# Bottleneck distance
# ------------------------------------------------------------------


class TestBottleneck:
    def test_identical_diagrams_distance_zero(self):
        pts = [(0.0, 0.0), (3.0, 4.0), (10.0, 0.0)]
        top1 = Topologist.create(max_dim=0)
        top2 = Topologist.create(max_dim=0)
        for p in pts:
            top1.observe(p)
            top2.observe(p)
        d1 = top1.compute()
        d2 = top2.compute()
        bd = d1.bottleneck_distance(d2, 0)
        assert bd == pytest.approx(0.0)

    def test_empty_diagrams_distance_zero(self):
        d1 = PersistenceDiagram(
            pairs=(), max_dim=0, n_points=0, max_scale=0.0, metric="euclidean"
        )
        d2 = PersistenceDiagram(
            pairs=(), max_dim=0, n_points=0, max_scale=0.0, metric="euclidean"
        )
        assert d1.bottleneck_distance(d2, 0) == 0.0

    def test_stability_under_small_perturbation(self):
        """Bottleneck distance ≤ Hausdorff distance (CSEH 2007)."""
        rng = random.Random(0)
        n = 8
        pts = [(rng.random(), rng.random()) for _ in range(n)]
        eps = 0.01
        perturbed = [(x + eps * (rng.random() - 0.5), y + eps * (rng.random() - 0.5)) for (x, y) in pts]
        top1 = Topologist.create(max_dim=0, max_points=n)
        top2 = Topologist.create(max_dim=0, max_points=n)
        for p in pts:
            top1.observe(p)
        for p in perturbed:
            top2.observe(p)
        d1 = top1.compute()
        d2 = top2.compute()
        bd = d1.bottleneck_distance(d2, 0)
        # Hausdorff distance ≤ eps (since each point moves at most eps/2 per coord
        # → at most eps/sqrt(2) in Euclidean). Bottleneck must be no larger.
        assert bd <= eps + 1e-6

    def test_essential_count_mismatch_inf(self):
        """Different numbers of essential pairs → inf bottleneck."""
        d1 = PersistenceDiagram(
            pairs=(PersistencePair(dim=0, birth=0.0, death=float("inf")),),
            max_dim=0,
            n_points=1,
            max_scale=1.0,
            metric="euclidean",
        )
        d2 = PersistenceDiagram(
            pairs=(
                PersistencePair(dim=0, birth=0.0, death=float("inf")),
                PersistencePair(dim=0, birth=0.0, death=float("inf")),
            ),
            max_dim=0,
            n_points=2,
            max_scale=1.0,
            metric="euclidean",
        )
        assert math.isinf(d1.bottleneck_distance(d2, 0))

    def test_single_point_to_diagonal(self):
        """A diagram with one pair (b, d) and an empty diagram have
        bottleneck distance (d-b)/2."""
        d1 = PersistenceDiagram(
            pairs=(PersistencePair(dim=0, birth=0.0, death=2.0),),
            max_dim=0,
            n_points=2,
            max_scale=3.0,
            metric="euclidean",
        )
        d2 = PersistenceDiagram(
            pairs=(),
            max_dim=0,
            n_points=0,
            max_scale=3.0,
            metric="euclidean",
        )
        bd = d1.bottleneck_distance(d2, 0)
        assert bd == pytest.approx(1.0)  # (2-0)/2

    def test_bottleneck_to_emits_event(self):
        bus = EventBus()
        top = Topologist.create(bus=bus, session_id="s", max_dim=0)
        top.observe((0.0,))
        top.observe((1.0,))
        d = top.compute()
        # compare to identical fresh diagram
        top.bottleneck_to(d, 0)
        events = bus.history(session_id="s", kind=TOPOLOGIST_COMPARED)
        assert len(events) == 1
        assert events[0].data["bottleneck_distance"] == pytest.approx(0.0)


# ------------------------------------------------------------------
# Bootstrap confidence band
# ------------------------------------------------------------------


class TestBootstrap:
    def test_bootstrap_returns_band(self):
        pts = [(float(i),) for i in range(10)]
        top = Topologist.create(max_dim=0, seed=1)
        for p in pts:
            top.observe(p)
        band = top.bootstrap_band(n_resamples=10, alpha=0.1)
        assert isinstance(band, BootstrapBand)
        assert band.alpha == 0.1
        assert band.n_resamples == 10
        # Each known dim has a quantile
        assert 0 in band.quantiles

    def test_bootstrap_zero_data_raises(self):
        top = Topologist.create(max_dim=0)
        with pytest.raises(InsufficientData):
            top.bootstrap_band(n_resamples=10, alpha=0.1)

    def test_bootstrap_bad_alpha(self):
        top = Topologist.create(max_dim=0)
        top.observe((0.0,))
        top.observe((1.0,))
        with pytest.raises(InvalidConfig):
            top.bootstrap_band(n_resamples=10, alpha=0.0)
        with pytest.raises(InvalidConfig):
            top.bootstrap_band(n_resamples=10, alpha=1.0)

    def test_bootstrap_bad_n_resamples(self):
        top = Topologist.create(max_dim=0)
        top.observe((0.0,))
        top.observe((1.0,))
        with pytest.raises(InvalidConfig):
            top.bootstrap_band(n_resamples=0)

    def test_bootstrap_emits_event(self):
        bus = EventBus()
        top = Topologist.create(bus=bus, session_id="s", max_dim=0, seed=2)
        for i in range(5):
            top.observe((float(i),))
        top.bootstrap_band(n_resamples=5, alpha=0.1)
        events = bus.history(session_id="s", kind=TOPOLOGIST_BOOTSTRAPPED)
        assert len(events) == 1
        assert events[0].data["n_resamples"] == 5

    def test_bootstrap_seeded_reproducible(self):
        pts = [(float(i),) for i in range(8)]
        top1 = Topologist.create(max_dim=0, seed=7)
        top2 = Topologist.create(max_dim=0, seed=7)
        for p in pts:
            top1.observe(p)
            top2.observe(p)
        b1 = top1.bootstrap_band(n_resamples=10, alpha=0.1)
        b2 = top2.bootstrap_band(n_resamples=10, alpha=0.1)
        for k in b1.quantiles:
            assert b1.quantiles[k] == pytest.approx(b2.quantiles[k])


# ------------------------------------------------------------------
# Stability certificate
# ------------------------------------------------------------------


class TestCertificate:
    def test_stability_certificate_basic(self):
        top = Topologist.create()
        cert = top.stability_certificate(0.05)
        assert isinstance(cert, StabilityCertificate)
        assert cert.hausdorff_perturbation == 0.05
        assert cert.bottleneck_bound_bits == 0.05
        assert "Cohen-Steiner" in cert.statement

    def test_stability_certificate_negative_rejected(self):
        top = Topologist.create()
        with pytest.raises(InvalidConfig):
            top.stability_certificate(-0.1)


# ------------------------------------------------------------------
# Report API
# ------------------------------------------------------------------


class TestReport:
    def test_report_basic(self):
        pts = _square_cluster(5, 0.0, 0.0, 0.1, seed=1)
        top = Topologist.create(max_dim=0)
        for p in pts:
            top.observe(p)
        rep = top.report(top_k=3)
        assert isinstance(rep, TopologistReport)
        assert rep.n_points == 5
        assert rep.max_dim == 0
        assert 0 in rep.n_pairs
        # essential at scale 0 → β_0 = 1 (one merged-by-the-end component)
        assert rep.betti_at_max[0] >= 1
        assert rep.diagram is not None

    def test_report_emits_event(self):
        bus = EventBus()
        top = Topologist.create(bus=bus, session_id="s", max_dim=0)
        for i in range(5):
            top.observe((float(i),))
        top.report()
        events = bus.history(session_id="s", kind=TOPOLOGIST_REPORTED)
        assert len(events) == 1
        assert "n_pairs" in events[0].data

    def test_report_with_stability_eps(self):
        top = Topologist.create(max_dim=0)
        for i in range(3):
            top.observe((float(i),))
        rep = top.report(hausdorff_perturbation=0.1)
        assert rep.stability.hausdorff_perturbation == 0.1


# ------------------------------------------------------------------
# Fingerprint / audit chain
# ------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_changes_on_observe(self):
        top = Topologist.create()
        fp0 = top.fingerprint()
        top.observe((0.0,))
        fp1 = top.fingerprint()
        top.observe((1.0,))
        fp2 = top.fingerprint()
        assert fp0 != fp1
        assert fp1 != fp2

    def test_fingerprint_changes_on_compute(self):
        top = Topologist.create()
        top.observe((0.0,))
        top.observe((1.0,))
        fp_before = top.fingerprint()
        top.compute()
        fp_after = top.fingerprint()
        assert fp_before != fp_after

    def test_fingerprints_reproducible(self):
        top1 = Topologist.create(seed=0)
        top2 = Topologist.create(seed=0)
        for v in [0.0, 1.0, 2.0]:
            top1.observe((v,))
            top2.observe((v,))
        top1.compute()
        top2.compute()
        assert top1.fingerprint() == top2.fingerprint()


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_observations(self):
        import threading

        top = Topologist.create(max_points=200)
        errors: list[BaseException] = []

        def worker(start: int):
            try:
                for i in range(start, start + 20):
                    top.observe((float(i),))
            except BaseException as e:  # noqa
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(k * 20,)) for k in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert top.n_points() == 100


# ------------------------------------------------------------------
# Edge cases / complex truncation
# ------------------------------------------------------------------


class TestComplexTooLarge:
    def test_truncation_records_flag(self):
        """Many points + high dim should trigger auto-truncation."""
        rng = random.Random(0)
        pts = [(rng.random(), rng.random()) for _ in range(40)]
        # Very low max_simplices forces truncation.
        top = Topologist.create(
            max_dim=2, max_scale=1.5, max_simplices=200, max_points=50
        )
        for p in pts:
            top.observe(p)
        d = top.compute()
        rep = top.report()
        assert rep.truncated is True
        assert rep.n_simplices > 0


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------


class TestPureHelpers:
    def test_from_points_one_shot(self):
        pts = [(0.0,), (1.0,), (5.0,)]
        d = Topologist.from_points(pts, max_dim=0, max_scale=10.0)
        assert isinstance(d, PersistenceDiagram)
        assert d.n_points == 3

    def test_from_points_empty_rejected(self):
        with pytest.raises(InsufficientData):
            Topologist.from_points([], max_dim=0)


# ------------------------------------------------------------------
# Use case: drift detection via bottleneck distance
# ------------------------------------------------------------------


class TestDriftUseCase:
    def test_drift_via_bottleneck_increases_on_added_loop(self):
        """A point cloud morphing from cluster to ring should show
        increasing dim-1 bottleneck distance from the cluster baseline."""
        cluster = _square_cluster(15, 0.0, 0.0, 0.3, seed=42)
        # baseline diagram
        top_b = Topologist.create(
            max_dim=1, max_scale=2.5, max_points=30, max_simplices=200_000
        )
        for p in cluster:
            top_b.observe(p)
        base = top_b.compute()
        # ring (clearly different topology)
        ring = _circle(15, radius=1.0)
        top_r = Topologist.create(
            max_dim=1, max_scale=2.5, max_points=30, max_simplices=200_000
        )
        for p in ring:
            top_r.observe(p)
        dring = top_r.compute()
        # bottleneck distance should be sizeable in dim-1
        bd1 = base.bottleneck_distance(dring, 1)
        assert bd1 > 0.2  # clear shape change

    def test_precomputed_distance_circle(self):
        """Use a precomputed distance matrix instead of coords."""
        pts = _circle(10, radius=1.0)
        n = len(pts)
        dm = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean(pts[i], pts[j])
                dm[i][j] = d
                dm[j][i] = d
        top = Topologist.create(
            max_dim=1, max_scale=2.5, max_points=15, max_simplices=200_000
        )
        top.observe_distance_matrix(dm)
        d = top.compute()
        loops = d.diagram(1)
        assert any(
            not p.is_infinite and (p.death - p.birth) > 0.5 for p in loops
        )
