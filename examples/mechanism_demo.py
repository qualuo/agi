"""End-to-end demo of `agi.mechanism` — revenue-optimal mechanism design as a
runtime primitive.

Six scenes:

  1. Vickrey vs first-price — the same bids, two prices. First-price
     leaves money on the table because the auctioneer doesn't know the
     true valuations; Vickrey collects them honestly.
  2. Myerson optimal — closed-form revenue gain over Vickrey at the cost
     of needing a prior. We show the gain on i.i.d. U[0,1].
  3. Data-driven Myerson — fit the prior *from samples*, then run Myerson.
     Cole-Roughgarden sample complexity bounds the gap to true-optimal.
  4. VCG for multi-item allocation — three GPUs, two tenants with
     additive valuations. Pivot payments make every tenant want to tell
     the truth.
  5. Online posted-price — Kleinberg-Leighton against an opaque buyer
     with a fixed acceptance threshold. We show convergence to the
     optimal price in O(T^{2/3}) regret.
  6. Bulow-Klemperer — "one more bidder beats any reserve."
"""
from __future__ import annotations

import random
import statistics
import time

from agi.events import EventBus
from agi.mechanism import (
    KIND_FIRST_PRICE,
    KIND_MYERSON,
    KIND_VICKREY,
    MechanismDesigner,
    UniformDistribution,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def scene_1_vickrey_vs_first_price(md: MechanismDesigner) -> None:
    banner("SCENE 1 — Vickrey vs First-Price (same bids, two prices)")
    bids = {"haiku": 0.62, "sonnet": 0.81, "opus": 0.93}
    v = md.vickrey_auction(bids)
    f = md.first_price_auction(bids)
    print(f"  bids:               {bids}")
    print(f"  Vickrey:  winner={v.winner} pays={v.payment:.3f} revenue={v.revenue:.3f}")
    print(f"  1st-price:winner={f.winner} pays={f.payment:.3f} revenue={f.revenue:.3f}")
    # DSIC certificates
    dsic_v = md.certify_dsic(KIND_VICKREY, bids_truthful=bids)
    dsic_f = md.certify_dsic(KIND_FIRST_PRICE, bids_truthful=bids)
    print(f"  Vickrey DSIC:        {dsic_v.is_dsic}  (worst gain {dsic_v.worst_gain:.4f})")
    print(f"  1st-price DSIC:      {dsic_f.is_dsic}  (worst gain {dsic_f.worst_gain:.4f}, failing={dsic_f.failing_bidder})")
    print("  ⇒ Vickrey is DSIC; first-price is NOT — bidders gain by shading.")


def scene_2_myerson_revenue_gain(md: MechanismDesigner) -> None:
    banner("SCENE 2 — Myerson optimal vs Vickrey (revenue gain from a known prior)")
    rng = random.Random(2026)
    N, K = 5000, 3
    dists = {f"b{i}": UniformDistribution(0, 1) for i in range(K)}
    rev_myerson, rev_vickrey = 0.0, 0.0
    for _ in range(N):
        bids = {f"b{i}": rng.uniform(0, 1) for i in range(K)}
        rev_myerson += md.myerson_auction(bids, dists).revenue
        rev_vickrey += md.vickrey_auction(bids).revenue
    mu_m, mu_v = rev_myerson / N, rev_vickrey / N
    gain = (mu_m / mu_v - 1.0) * 100 if mu_v > 0 else float("inf")
    print(f"  3 bidders i.i.d. U[0,1], N={N} draws")
    print(f"  E[rev | Myerson]:  {mu_m:.4f}")
    print(f"  E[rev | Vickrey]:  {mu_v:.4f}")
    print(f"  Revenue gain:      {gain:+.1f}% from designing the reserve")


def scene_3_data_driven_myerson(md: MechanismDesigner) -> None:
    banner("SCENE 3 — Data-driven Myerson (fit prior from samples)")
    rng = random.Random(3)
    K, N_SAMPLES = 3, 400
    samples = {f"b{i}": [rng.uniform(0, 1) for _ in range(N_SAMPLES)] for i in range(K)}

    # Fit empirical reserve per bidder, report the LCB
    for b, s in samples.items():
        rp = md.empirical_reserve(s, method="monopoly", delta=0.05)
        print(
            f"  {b}: empirical reserve = {rp.reserve:.3f}  "
            f"E[rev]≈{rp.revenue_mean:.3f}  LCB={rp.revenue_lcb:.3f}  "
            f"(true monopoly = 0.500)"
        )

    # Run myerson_from_samples on a single round
    bids = {"b0": 0.62, "b1": 0.81, "b2": 0.55}
    a = md.myerson_from_samples(bids, samples)
    print(f"  myerson_from_samples bids={bids} ⇒ winner={a.winner}, pays={a.payment:.3f}")

    # Sample complexity to be within ε of optimal
    for eps in (0.10, 0.05, 0.02):
        n = md.sample_complexity(epsilon=eps, delta=0.05, k_bidders=K)
        print(f"  Sample complexity: ε={eps:.2f} ⇒ n ≥ {n:,} samples per bidder")


def scene_4_vcg_multi_item(md: MechanismDesigner) -> None:
    banner("SCENE 4 — VCG for multi-item allocation (3 GPUs, 2 tenants)")
    items = ["gpu-0", "gpu-1", "gpu-2"]
    bids = {
        "tenantA": {"gpu-0": 10.0, "gpu-1": 8.0, "gpu-2": 6.0},
        "tenantB": {"gpu-0":  7.0, "gpu-1": 9.0, "gpu-2": 5.0},
    }
    out = md.vcg_allocation(items=items, bids=bids, capacity={"tenantA": 2, "tenantB": 1})
    print(f"  total welfare:  {out.total_welfare:.2f}")
    print(f"  total revenue:  {out.total_revenue:.2f}")
    for a in out.assignments:
        print(f"    {a.bidder_id} ← {a.item_id}  (value {a.value:.2f}, pay {a.payment:.2f})")
    ir = md.certify_ir(out)
    print(f"  IR certified:   {ir.is_ir}  worst utility = {ir.worst_utility:.3f}")


def scene_5_online_posted_price(md: MechanismDesigner) -> None:
    banner("SCENE 5 — Online posted-price (Kleinberg-Leighton EXP3 grid)")
    threshold = 0.65

    def buyer(price: float) -> bool:
        return price <= threshold

    out = md.online_posted_price(feedback=buyer, T=4000, v_max=1.0, delta=0.05)
    # Optimal revenue/round = threshold.
    print(f"  buyer threshold:        {threshold}")
    print(f"  T = {out.T},  v_max = {out.v_max},  grid size = {out.n_grid}")
    print(f"  total revenue:          {out.revenue:.2f}")
    print(f"  mean revenue / round:   {out.revenue_mean:.4f}   (optimum = {threshold})")
    print(f"  pseudo-regret UB:       {out.pseudo_regret_ub:.1f}")
    # Empirical hit rate of the buyer threshold across the last 25% of rounds
    tail = out.accept_history[-len(out.accept_history) // 4 :]
    print(f"  acceptance rate (last quartile): {sum(tail) / len(tail):.3f}")


def scene_6_bulow_klemperer(md: MechanismDesigner) -> None:
    banner("SCENE 6 — Bulow-Klemperer ('one more bidder ≥ optimal reserve')")
    rng = random.Random(6)
    samples = [rng.uniform(0, 1) for _ in range(1000)]
    for n in (1, 2, 4, 8):
        bk = md.bulow_klemperer(samples, n=n, trials=2000)
        print(
            f"  n={n}: Vickrey-no-reserve(n+1)={bk.revenue_vickrey_n_plus_1:.4f}  "
            f"Myerson-with-reserve(n)={bk.revenue_myerson_n:.4f}  "
            f"ratio={bk.ratio:.3f}"
        )
    print("  ⇒ Inviting one more bidder beats designing the reserve (BK 1996).")


def main() -> None:
    bus = EventBus()
    md = MechanismDesigner(bus=bus, random_seed=2026)

    print("MechanismDesigner — revenue-optimal mechanism design as a runtime primitive.")
    print(f"Event bus attached; {len(bus.history())} events in history.")

    scene_1_vickrey_vs_first_price(md)
    scene_2_myerson_revenue_gain(md)
    scene_3_data_driven_myerson(md)
    scene_4_vcg_multi_item(md)
    scene_5_online_posted_price(md)
    scene_6_bulow_klemperer(md)

    print()
    print("=" * 72)
    print("  Designer stats:")
    for k, v in md.stats().items():
        print(f"    {k:>20} = {v}")
    print(f"  Events recorded:  {len(bus.history())}")


if __name__ == "__main__":
    main()
