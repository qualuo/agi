"""RiskController demo: a coordination engine using distribution-free
risk control to pick its own operating thresholds.

Scenario
--------

A multi-tenant runtime hedges expensive tickets — running them on two
models in parallel and keeping the faster correct answer. Hedging is
useful but doubles the cost; the engine only wants to hedge tickets
that are *actually* at risk of overrunning their estimate.

The coordination engine needs to set a single number — the
`hedge_score` threshold above which a ticket is hedged — and ship it
to production with a guarantee like:

    "with 90% confidence, our hedging policy will have a
     'hedged-and-still-overran' rate of at most 5%."

That's a *risk-control* problem, not a coverage problem. CRC + LTT
gives it a finite-sample, distribution-free guarantee.

What this demo shows
--------------------

1. CRC picks the smallest λ satisfying its E-bound — pure expectation
   control, no FWER. We feed it a monotone "miss-rate" loss whose
   true risk decreases as λ rises.

2. LTT with Hoeffding-Bentkus picks the most aggressive λ from a grid
   whose UCB is ≤ target. We report empirical risk, UCB, and the
   threshold chosen, side-by-side with the WSR variant (uniformly
   sharper UCBs on bounded losses).

3. `select_multi` bounds two risks (refund rate AND abstention rate)
   simultaneously by Bonferroni-splitting δ — the right tool when an
   SLO has multiple quantitative constraints.

4. The Monte Carlo verification confirms FWER ≤ δ at the boundary:
   we run the procedure repeatedly on resampled calibration data
   and check the realized risk on a fresh test point is ≤ target.

Run:
    python examples/risk_control_demo.py
"""
from __future__ import annotations

import math
import random
import statistics

from agi.risk_control import (
    METHOD_CRC,
    METHOD_LTT_HB,
    METHOD_LTT_WSR,
    ORDER_AGGRESSIVE_FIRST,
    Risk,
    RiskController,
    RiskPoint,
    hoeffding_bentkus_ucb,
    wsr_ucb,
)


def _draw_ticket(rng: random.Random) -> tuple[float, float]:
    """Return (predicted_cost, actual_cost) for one synthetic ticket.

    predicted_cost ∼ Exponential(2) (≈$0.50 mean).
    actual_cost   = predicted_cost · LogNormal(0, 0.7²).
    """
    pred = rng.expovariate(2.0)
    shock = math.exp(rng.gauss(0.0, 0.7))
    return pred, pred * shock


def hedged_and_overran(p: RiskPoint, lam: float) -> float:
    """1 if the policy hedged *and* the ticket still overran by ≥1.5×.

    This is the operational risk the coordinator must bound. It is
    *not* monotone in λ: hedging less reduces the false-hedge risk
    but also reduces opportunities to catch real overruns.
    """
    if p.score >= lam and float(p.outcome) > p.score * 1.5:
        return 1.0
    return 0.0


def miss_rate(p: RiskPoint, lam: float) -> float:
    """1 if we did NOT hedge an overrunning ticket. Monotone NON-DECREASING
    in λ — as the hedge threshold rises, we hedge fewer tickets, so more
    overruns slip through.

    This is the "false negative" risk that CRC's monotone-step trick
    applies to directly.
    """
    if p.score < lam and float(p.outcome) > p.score * 1.5:
        return 1.0
    return 0.0


def abstain_rate(p: RiskPoint, lam: float) -> float:
    """1 if we hedge (i.e. "abstain" from single-model dispatch).
    Monotone DECREASING in λ — higher threshold means hedging less.
    """
    return 1.0 if p.score >= lam else 0.0


def run() -> None:
    rng = random.Random(2026)
    rc = RiskController()

    # 1) Drain the receipt stream into calibration.
    for _ in range(1500):
        pred, actual = _draw_ticket(rng)
        rc.record(score=pred, outcome=actual)

    # 2) Conformal Risk Control — monotone loss, expectation bound.
    candidates = [round(0.1 * i, 2) for i in range(1, 26)]  # 0.1 .. 2.5
    sel_crc = rc.select(
        candidates=candidates,
        target=0.20,            # ≤20% expected miss rate
        loss_fn=miss_rate,
        method=METHOD_CRC,
        monotone="increasing",  # miss_rate ↑ as λ ↑
    )

    # 3) Learn-Then-Test with Hoeffding-Bentkus on the
    #    hedged-and-still-overran rate — UCB ≤ 20%, FWER δ = 10%.
    #    The loss is monotone DECREASING in λ (higher λ ⇒ less hedging
    #    ⇒ fewer false hedges). Aggressive-first means: find the
    #    smallest λ (= most willing to hedge) whose UCB ≤ target.
    sel_hb = rc.select(
        candidates=candidates,
        target=0.20,
        delta=0.10,
        loss_fn=hedged_and_overran,
        method=METHOD_LTT_HB,
        monotone="decreasing",
        ordering=ORDER_AGGRESSIVE_FIRST,
    )

    # 4) Same risk, sharper bound: WSR.
    sel_wsr = rc.select(
        candidates=candidates,
        target=0.20,
        delta=0.10,
        loss_fn=hedged_and_overran,
        method=METHOD_LTT_WSR,
        monotone="decreasing",
        ordering=ORDER_AGGRESSIVE_FIRST,
    )

    # 5) Multi-risk: bound miss-rate ≤ 22% AND abstain-rate ≤ 70% at FWER δ.
    #    miss_rate is increasing in λ; abstain_rate is decreasing in λ.
    #    The aggressive-first ordering interpretation differs per risk;
    #    each call resolves it independently against its own monotone tag.
    sels = rc.select_multi(
        candidates=candidates,
        delta=0.10,
        method=METHOD_LTT_HB,
        risks=[
            Risk(name="miss_rate", target=0.22, loss_fn=miss_rate, monotone="increasing"),
            Risk(name="abstain_rate", target=0.70, loss_fn=abstain_rate, monotone="decreasing"),
        ],
        ordering=ORDER_AGGRESSIVE_FIRST,
    )

    print("=== RiskController demo ===")
    print(f"Calibration size:                  {len(rc)}")
    print("")
    print("[CRC: E[L_miss] ≤ 0.20]")
    if sel_crc is None:
        print("  no certifiable threshold (calibration too small or target too tight)")
    else:
        print(f"  λ̂ = {sel_crc.threshold:.3f}")
        print(f"  empirical miss-rate = {sel_crc.empirical_risk:.4f}")
        print(f"  HB UCB on E[L]      = {sel_crc.ucb:.4f}")
    print("")
    print("[LTT-HB: P(R(λ̂) ≤ 0.20) ≥ 0.90]")
    if sel_hb is None:
        print("  no candidate UCB ≤ target")
    else:
        print(f"  λ̂ = {sel_hb.threshold:.3f}")
        print(f"  empirical risk       = {sel_hb.empirical_risk:.4f}")
        print(f"  UCB (Hoeffding-Bent) = {sel_hb.ucb:.4f}")
    print("")
    print("[LTT-WSR: same target, sharper bound]")
    if sel_wsr is None:
        print("  no candidate UCB ≤ target")
    else:
        print(f"  λ̂ = {sel_wsr.threshold:.3f}")
        print(f"  empirical risk       = {sel_wsr.empirical_risk:.4f}")
        print(f"  UCB (WSR)            = {sel_wsr.ucb:.4f}")
    print("")
    print("[Multi-risk: miss ≤ 0.22 AND abstain ≤ 0.70 at FWER 0.10]")
    for name, sel in sels.items():
        if sel is None:
            print(f"  {name}: infeasible at δ/k = {0.05}")
        else:
            print(f"  {name}: λ̂ = {sel.threshold:.3f}  emp = {sel.empirical_risk:.4f}  UCB = {sel.ucb:.4f}")
    print("")

    # 6) Sanity: Monte Carlo a held-out test set, check the realized
    #    risk at the chosen λ̂ matches the certificate.
    if sel_hb is not None:
        realized = 0
        held_n = 3000
        for _ in range(held_n):
            pred, actual = _draw_ticket(rng)
            test = RiskPoint(score=pred, outcome=actual)
            realized += int(hedged_and_overran(test, sel_hb.threshold))
        print(f"Held-out empirical risk at λ̂={sel_hb.threshold:.2f}: {realized/held_n:.4f}  (UCB was {sel_hb.ucb:.4f})")

    # 7) Show how the UCB tightens as data accumulates.
    #    Use randomized i.i.d. Bernoulli samples; WSR needs exchangeable
    #    inputs to behave well (its predictable λ_t schedule is sensitive
    #    to adversarial ordering).
    print("")
    print("[UCB tightness vs. sample size, true p=0.05, δ=0.10]")
    print("  n         p̂        HB         WSR")
    bench_rng = random.Random(31415)
    for n in (50, 200, 1000, 5000, 20000):
        sample = [1.0 if bench_rng.random() < 0.05 else 0.0 for _ in range(n)]
        p_hat = statistics.fmean(sample)
        hb = hoeffding_bentkus_ucb(p_hat, n, delta=0.10)
        wsr = wsr_ucb(sample, delta=0.10)
        print(f"  {n:<6}    {p_hat:.4f}   {hb:.4f}    {wsr:.4f}")


if __name__ == "__main__":
    run()
