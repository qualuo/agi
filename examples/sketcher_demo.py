"""Sketcher demo — bounded-memory streaming sketches as a runtime primitive.

Five scenarios show how a coordination engine actually drives the
sketcher at runtime, with each scenario producing an explicit
``(ε, δ)`` error certificate and the measured state footprint:

  1. Heavy hitters at scale — Misra-Gries finds the top items in a
     1M-event stream using a few KB of state, with a deterministic
     additive-error bound.

  2. Approximate frequencies — Count-Min sketches the same stream
     with a configurable ε-additive (one-sided) bound; conservative
     update is shown alongside ordinary update.

  3. Distinct-count under hash collisions — HyperLogLog estimates
     stream cardinality with O(2^p) registers and ≈ 1.04/√(2^p)
     relative standard error.

  4. Streaming quantiles — KLL gives an ε-additive-rank approximation
     of every quantile, and GreenwaldKhanna gives a deterministic
     version of the same.

  5. Distributed merge — two workers sketch independent shards of a
     stream, the coordinator merges them, and the merged sketch
     answers as well as a single serial sketch over the union.

Run::

    python examples/sketcher_demo.py
"""
from __future__ import annotations

import math
import random
import time

from agi.sketcher import Sketcher, sketcher_summary


def banner(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


# ----------------------------------------------------------------
# Scenario 1 — heavy hitters at scale
# ----------------------------------------------------------------


def scenario_heavy_hitters() -> None:
    banner("Scenario 1 — Misra-Gries heavy hitters (deterministic)")
    random.seed(0)
    # 1M events: 40% are HOT_i for i in 0..9 (each ~4% of stream, well
    # above the survival threshold N/(k+1)), rest are random distractors.
    N = 1_000_000
    sk = Sketcher.misra_gries(k=128)
    t0 = time.time()
    for _ in range(N):
        r = random.random()
        if r < 0.4:
            sk.update(f"HOT_{int(r * 25) % 10}")
        else:
            sk.update(f"d_{random.randint(0, 99_999)}")
    dt = time.time() - t0
    report = sk.report()
    print(f"  Stream length: {sk.n_items:,}")
    print(f"  Sketch state:  {report.n_bytes:,} bytes")
    print(f"  Wall time:     {dt:.2f} s")
    print(f"  Additive bound: ε = {report.epsilon:.5f}  (i.e. ≤ {report.epsilon * sk.n_items:,.0f} per item)")
    print(f"  Certificate:    {report.certificate}")
    print(f"  Top 10 heavy hitters (counter is an underestimate):")
    for item, count in sk.heavy_hitters()[:10]:
        print(f"    {item!s:>14}  ≥ {count:>8.0f}")


# ----------------------------------------------------------------
# Scenario 2 — Count-Min, ordinary vs conservative
# ----------------------------------------------------------------


def scenario_count_min() -> None:
    banner("Scenario 2 — Count-Min vs Count-Min-Conservative-Update")
    random.seed(1)
    N = 200_000
    true_counts: dict = {}
    sk_norm = Sketcher.count_min(epsilon=1e-3, delta=1e-3, seed=7)
    sk_cons = Sketcher.count_min(
        epsilon=1e-3, delta=1e-3, seed=7, conservative_update=True,
    )
    for _ in range(N):
        x = random.randint(0, 9_999)
        true_counts[x] = true_counts.get(x, 0) + 1
        sk_norm.update(x)
        sk_cons.update(x)
    print(f"  Stream length: {sk_norm.n_items:,}")
    norm_r, cons_r = sk_norm.report(), sk_cons.report()
    print(
        f"  CMS:      {norm_r.n_bytes:>7,} B, "
        f"ε={norm_r.epsilon:.5f}, δ={norm_r.delta:.5f}, "
        f"shape={norm_r.extra['rows']}×{norm_r.extra['cols']}"
    )
    print(
        f"  CMS-CU:   {cons_r.n_bytes:>7,} B, "
        f"ε={cons_r.epsilon:.5f}, δ={cons_r.delta:.5f}, "
        f"shape={cons_r.extra['rows']}×{cons_r.extra['cols']}"
    )
    # Compare per-item over-estimate (≥ 0).
    overs_n = []
    overs_c = []
    for x, true_c in true_counts.items():
        overs_n.append(sk_norm.query(x) - true_c)
        overs_c.append(sk_cons.query(x) - true_c)
    avg_n = sum(overs_n) / len(overs_n)
    avg_c = sum(overs_c) / len(overs_c)
    max_n = max(overs_n)
    max_c = max(overs_c)
    print(f"  CMS    avg over-estimate: {avg_n:>6.2f}    max: {max_n:>5.0f}")
    print(f"  CMS-CU avg over-estimate: {avg_c:>6.2f}    max: {max_c:>5.0f}")
    print("  (Conservative update never under-estimates and is "
          "almost always tighter.)")


# ----------------------------------------------------------------
# Scenario 3 — HyperLogLog distinct count
# ----------------------------------------------------------------


def scenario_hll() -> None:
    banner("Scenario 3 — HyperLogLog cardinality at multiple precisions")
    random.seed(2)
    true_card = 200_000
    print(f"  True cardinality: {true_card:,}")
    print(f"  Precision  Registers   Bytes   Estimate   Rel-error")
    for p in (8, 10, 12, 14, 16):
        sk = Sketcher.hll(precision=p, seed=42)
        for x in range(true_card):
            sk.update(f"item_{x}")
        est = sk.cardinality()
        rel_err = abs(est - true_card) / true_card
        bytes_ = sk.report().n_bytes
        regs = 1 << p
        print(
            f"     p={p}       {regs:>5d}   {bytes_:>6,d}  {est:>10,.0f}    {rel_err:.4f}"
        )


# ----------------------------------------------------------------
# Scenario 4 — Streaming quantiles
# ----------------------------------------------------------------


def scenario_quantiles() -> None:
    banner("Scenario 4 — Streaming quantiles (KLL + GreenwaldKhanna)")
    random.seed(3)
    N = 100_000
    data = [math.exp(random.gauss(3, 0.5)) for _ in range(N)]
    kll = Sketcher.kll(k=1024, seed=0)
    gk = Sketcher.gk(k=200)
    t0 = time.time()
    for x in data:
        kll.update(x)
        gk.update(x)
    dt = time.time() - t0
    data.sort()
    print(f"  Stream length: {N:,}  (lognormal latency in ms)")
    print(f"  KLL  state: {kll.report().n_bytes:>5,} bytes")
    print(f"  GK   state: {gk.report().n_bytes:>5,} bytes")
    print(f"  Sketched both in {dt:.2f} s")
    print(f"  Quantile  True       KLL est    GK est")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        idx = int(q * N)
        print(
            f"  q={q:.2f}   {data[idx]:>8.2f}   "
            f"{kll.quantile(q):>8.2f}   {gk.quantile(q):>8.2f}"
        )


# ----------------------------------------------------------------
# Scenario 5 — Distributed merge
# ----------------------------------------------------------------


def scenario_distributed_merge() -> None:
    banner("Scenario 5 — Distributed merge of mergeable sketches")
    random.seed(4)
    n_workers = 8
    per_worker = 50_000
    # Each worker sketches an independent shard with HOT items
    # appearing throughout.
    workers = [
        Sketcher.misra_gries(k=64, seed=0) for _ in range(n_workers)
    ]
    for w_idx, worker in enumerate(workers):
        for _ in range(per_worker):
            r = random.random()
            if r < 0.1:
                worker.update("HOT")
            elif r < 0.15:
                worker.update("WARM")
            else:
                worker.update(f"cold_{random.randint(0, 9_999)}")
    # Coordinator merges into a single sketch.
    global_sk = Sketcher.misra_gries(k=64, seed=0)
    for w in workers:
        global_sk.merge(w)
    print(
        f"  {n_workers} workers × {per_worker:,} events = "
        f"{n_workers * per_worker:,} total"
    )
    print(f"  Global sketch state: {global_sk.report().n_bytes:,} bytes")
    print(f"  Global sketch n_items: {global_sk.n_items:,}")
    print(f"  Global heavy hitters (top 5):")
    for item, count in global_sk.heavy_hitters()[:5]:
        print(f"    {item!s:>16}  ≥ {count:>8.0f}")
    # And a parallel HLL distinct-count merge.
    hlls = [Sketcher.hll(precision=14, seed=0) for _ in range(n_workers)]
    random.seed(5)
    for hll in hlls:
        for _ in range(per_worker):
            hll.update(random.randint(0, 1_000_000))
    union = Sketcher.hll(precision=14, seed=0)
    for hll in hlls:
        union.merge(hll)
    print(
        f"  Global HLL distinct estimate (merged): "
        f"{union.cardinality():,.0f}"
    )


# ----------------------------------------------------------------
# Scenario 6 — Composition with other primitives
# ----------------------------------------------------------------


def scenario_composition() -> None:
    banner("Scenario 6 — Sketcher composition with the rest of the runtime")
    print("  Composition recipes a coordination engine plugs in:")
    print("    • CountMin → DriftSentinel: low-memory drift over identifiers")
    print("    • HyperLogLog → Auditor: distinct-counts in compliance logs")
    print("    • MisraGries → Compressor: heavy-hitter MDL prior on streams")
    print("    • KLL → Forecaster: streaming quantile-binned calibration")
    print("    • Reservoir → ExperimentDesigner: unbiased eval-pool sampling")
    print("    • Bloom → ToolSynth: candidate-dedup over synthesised programs")
    print("    • F2Sketch → CausalDiscoverer: streaming mutual information")
    print("    • ExpHistogram → Forecaster: sliding-window event counts")
    s = sketcher_summary()
    print(f"  Sketch kinds available: {len(s['known_kinds'])}")
    print(f"  Mergeable kinds:        {len(s['mergeable_kinds'])}")
    print(f"  Pure-stdlib build:      {s['pure_stdlib']}")


def main() -> None:
    scenario_heavy_hitters()
    scenario_count_min()
    scenario_hll()
    scenario_quantiles()
    scenario_distributed_merge()
    scenario_composition()
    print()


if __name__ == "__main__":
    main()
