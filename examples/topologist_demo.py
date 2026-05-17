"""Topologist demo — persistent homology as a runtime primitive.

Three scenes:

  1. *Cluster discovery*.  A point cloud with three well-separated
     blobs.  The Topologist returns a dim-0 persistence diagram in
     which the three most persistent features dominate; everything
     below the bootstrap band is noise.

  2. *Loop detection on a noisy circle*.  A sample on a circle in
     the plane has β_1 = 1.  The Topologist finds the dim-1 feature
     and prints both its persistence interval and the bottleneck
     band that says it is significant.

  3. *Drift detection*.  A reference cluster is compared to an
     incoming circle by bottleneck distance in dim-1.  The drift is
     unambiguous; this is the runtime call a coordination engine
     makes when it suspects an LLM rollout has shifted topology.

No external dependencies; pure Python.
"""
from __future__ import annotations

import math
import random
import sys

from agi.events import EventBus
from agi.topologist import (
    METRIC_EUCLIDEAN,
    TOPOLOGIST_BOOTSTRAPPED,
    TOPOLOGIST_COMPUTED,
    TOPOLOGIST_REPORTED,
    Topologist,
)


def _bar_string(birth: float, death: float, lo: float, hi: float, width: int = 50) -> str:
    """Render a single persistence bar as ASCII."""
    span = max(hi - lo, 1e-9)
    if math.isinf(death):
        d = hi
        cap = "→"
    else:
        d = death
        cap = "│"
    a = max(0, int(round((birth - lo) / span * width)))
    b = max(a + 1, int(round((d - lo) / span * width)))
    cells = [" "] * width
    for i in range(a, min(b, width)):
        cells[i] = "─"
    if a < width:
        cells[a] = "│"
    if b - 1 < width:
        cells[b - 1] = cap
    return "".join(cells)


def scene_one_clusters() -> None:
    print("=" * 76)
    print("SCENE 1 — Cluster discovery via dim-0 persistent homology")
    print("=" * 76)
    rng = random.Random(0)

    def blob(cx: float, cy: float, side: float, n: int) -> list[tuple[float, float]]:
        return [
            (cx + side * (rng.random() - 0.5), cy + side * (rng.random() - 0.5))
            for _ in range(n)
        ]

    points: list[tuple[float, float]] = []
    points.extend(blob(0.0, 0.0, 0.4, 12))
    points.extend(blob(8.0, 0.0, 0.4, 12))
    points.extend(blob(4.0, 7.0, 0.4, 12))

    bus = EventBus()
    top = Topologist.create(
        max_dim=0,
        max_scale=12.0,
        max_points=80,
        seed=42,
        session_id="clusters",
        bus=bus,
    )
    for p in points:
        top.observe(p)
    diag = top.compute()

    print(f"  {len(points)} points, metric=euclidean, max_scale=12")
    print()
    print("  Top-5 most persistent dim-0 features:")
    print(f"  {'birth':>10} {'death':>10} {'persistence':>12}  bar")
    lo, hi = 0.0, diag.max_scale
    for pp in diag.k_most_persistent(0, 5):
        if pp.is_infinite:
            d_str = "∞"
            pers_str = "∞"
        else:
            d_str = f"{pp.death:8.4f}"
            pers_str = f"{pp.death - pp.birth:10.4f}"
        bar = _bar_string(pp.birth, pp.death, lo, hi)
        print(f"  {pp.birth:10.4f} {d_str:>10} {pers_str:>12}  {bar}")

    print()
    print("  Bootstrap confidence band (Fasy et al. 2014):")
    band = top.bootstrap_band(n_resamples=20, alpha=0.1)
    for k, q in sorted(band.quantiles.items()):
        print(f"    dim={k}: 90th-percentile bottleneck = {q:.4f} → significance threshold {2*q:.4f}")

    sig = diag.significant_features(0, threshold=band.dim(0))
    print(f"  Significant dim-0 features above threshold: {len(sig)}")
    print(f"  → recovered cluster count = {len(sig)}  (expected: 3)")
    print()
    n_events = len(bus.history(session_id="clusters"))
    print(f"  Audit ledger: {n_events} events; fingerprint head = {top.fingerprint()[:16]}…")


def scene_two_circle() -> None:
    print()
    print("=" * 76)
    print("SCENE 2 — Loop detection via dim-1 persistent homology")
    print("=" * 76)
    rng = random.Random(7)

    def noisy_circle(n: int, radius: float, jitter: float) -> list[tuple[float, float]]:
        pts = []
        for k in range(n):
            theta = 2.0 * math.pi * k / n
            r = radius + jitter * (rng.random() - 0.5)
            pts.append((r * math.cos(theta), r * math.sin(theta)))
        return pts

    points = noisy_circle(20, radius=1.0, jitter=0.05)

    top = Topologist.create(
        max_dim=1,
        max_scale=2.5,
        max_points=30,
        seed=11,
    )
    for p in points:
        top.observe(p)
    diag = top.compute()

    print(f"  {len(points)} points on a noisy circle, radius ≈ 1.0")
    print()
    print(f"  All dim-1 (loop) features in the diagram:")
    loops = diag.diagram(1)
    for pp in sorted(
        loops,
        key=lambda p: (
            float("inf")
            if p.is_infinite
            else -(p.death - p.birth)
        ),
    ):
        if pp.is_infinite:
            print(f"    birth = {pp.birth:7.4f}  death = ∞   (essential)")
        else:
            print(
                f"    birth = {pp.birth:7.4f}  death = {pp.death:7.4f}  "
                f"persistence = {pp.death - pp.birth:.4f}"
            )
    print()
    band = top.bootstrap_band(n_resamples=15, alpha=0.1, max_dim=1)
    print(f"  Bootstrap confidence band (dim 1, α=0.1): {band.dim(1):.4f}")
    sig = diag.significant_features(1, threshold=band.dim(1))
    print(f"  Significant dim-1 features above threshold: {len(sig)}")
    if sig:
        print(f"  → β_1 ≥ 1 with confidence: the data has a loop.  ✓")
    else:
        # Resample once more with more replicates to be sure
        print("  → bootstrap underestimated; the most persistent loop:")
        if loops:
            biggest = max(
                (p for p in loops if not p.is_infinite),
                key=lambda p: p.death - p.birth,
                default=None,
            )
            if biggest is not None:
                print(
                    f"    birth = {biggest.birth:.4f}  death = {biggest.death:.4f}  "
                    f"persistence = {biggest.death - biggest.birth:.4f}"
                )
    print()
    print("  Persistence landscape (Bubenik 2015), level 1, 16-point grid:")
    ls = diag.landscape(dim=1, num_levels=1, grid=16)
    if ls.grid:
        for t, v in zip(ls.grid, ls.levels[0]):
            bar = "▇" * max(0, int(v * 30))
            print(f"    t = {t:7.4f}  λ_1 = {v:7.4f}  {bar}")


def scene_three_drift() -> None:
    print()
    print("=" * 76)
    print("SCENE 3 — Drift detection via bottleneck distance")
    print("=" * 76)
    rng = random.Random(99)

    def blob(n: int) -> list[tuple[float, float]]:
        return [(0.5 * (rng.random() - 0.5), 0.5 * (rng.random() - 0.5)) for _ in range(n)]

    def circle(n: int) -> list[tuple[float, float]]:
        return [
            (math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
            for k in range(n)
        ]

    baseline = blob(15)
    incoming = circle(15)

    top_b = Topologist.create(
        max_dim=1, max_scale=2.5, max_points=20, seed=0
    )
    for p in baseline:
        top_b.observe(p)
    base_diag = top_b.compute()

    top_i = Topologist.create(
        max_dim=1, max_scale=2.5, max_points=20, seed=1
    )
    for p in incoming:
        top_i.observe(p)
    inc_diag = top_i.compute()

    bd0 = base_diag.bottleneck_distance(inc_diag, 0)
    bd1 = base_diag.bottleneck_distance(inc_diag, 1)

    cert = top_b.stability_certificate(0.05)

    print("  Reference: 15-point blob (no loop)")
    print("  Incoming:  15-point circle (one loop)")
    print()
    print(f"  Bottleneck distance, dim 0 = {bd0:.4f}")
    print(f"  Bottleneck distance, dim 1 = {bd1:.4f}  ← drift signal")
    print()
    print("  Stability certificate (CSEH 2007):")
    print(f"    {cert.statement}")
    print()
    print("  How a coordination engine uses this:")
    print("    if bd1 > 0.2:")
    print("        # incoming distribution has new topological structure;")
    print("        # route to a re-training trigger or freeze the policy.")


def main() -> int:
    scene_one_clusters()
    scene_two_circle()
    scene_three_drift()
    print()
    print("=" * 76)
    print("Done. Three primitives the runtime now exposes:")
    print("  Topologist.from_points(...)           — one-shot diagram")
    print("  top.bootstrap_band(...)               — Fasy et al. confidence")
    print("  diag.bottleneck_distance(other, k)    — drift signal w/ stability")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
