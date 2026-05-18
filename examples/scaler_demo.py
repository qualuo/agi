"""Scaler demo: fit a Chinchilla scaling law and read off the compute-optimal
allocation for a coordination engine.

The pitch in one runnable script (no API key required):

  1. A coordination engine has run completions at a handful of small-scale
     ``(model_size, data_tokens)`` settings and observed loss.
  2. It asks the runtime: *"If I spend C = 1e22 FLOPs on a next-gen run,
     what (N*, D*) should I pick and what loss should I expect?"*
  3. The :class:`Scaler` primitive fits a Chinchilla-style scaling law,
     extrapolates loss with a bootstrap-percentile confidence interval,
     and returns the closed-form ``(N*, D*, L*)`` allocation under
     ``C = 6 N D``.
  4. Every step writes a fingerprinted event to the runtime EventBus so
     any auditor can replay the decision later.

Run:  python examples/scaler_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.scaler import (
    FAMILY_CHINCHILLA,
    Observation,
    Scaler,
    ScalerConfig,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def true_loss(n: float, d: float) -> float:
    """A 'ground-truth' Chinchilla-shaped law for the demo."""
    return 1.69 + 406.4 * n**-0.34 + 410.7 * d**-0.28


def synthesize_grid(noise: float = 0.015, seed: int = 0) -> list[Observation]:
    """Synthetic (N, D, L) measurements at the 'small-scale' grid."""
    rng = random.Random(seed)
    rows: list[Observation] = []
    # 8 N x 5 D = 40 observations spanning ~6 orders of magnitude in N
    # and ~3 in D — representative of a real grid-fit budget.
    for n in [1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10]:
        for d in [3e8, 1e9, 3e9, 1e10, 3e10]:
            l = true_loss(n, d)
            l *= math.exp(rng.gauss(0.0, noise))  # multiplicative noise
            rows.append(Observation(n_params=n, d_tokens=d, loss=l))
    return rows


def main() -> int:
    bus = EventBus()
    events_seen: list[str] = []
    bus.subscribe(lambda ev: events_seen.append(ev.kind))

    banner("1. Coordination engine has 40 small-scale measurements")
    rows = synthesize_grid(noise=0.015, seed=42)
    print(f"   observations: {len(rows)}")
    print(f"   N range: {min(o.n_params for o in rows):.0e} ... "
          f"{max(o.n_params for o in rows):.0e}")
    print(f"   D range: {min(o.d_tokens for o in rows):.0e} ... "
          f"{max(o.d_tokens for o in rows):.0e}")
    print(f"   L range: {min(o.loss for o in rows):.4f} ... "
          f"{max(o.loss for o in rows):.4f}")

    banner("2. Scaler fits a Chinchilla law with PAC certificate")
    s = Scaler(ScalerConfig(
        family=FAMILY_CHINCHILLA,
        bootstrap_b=100,
        seed=42,
        holdout_fraction=0.2,
        confidence=0.95,
    ), bus=bus)
    s.observe(rows)
    fit = s.fit()
    print(f"   converged: {fit.converged} in {fit.iters} iters")
    print(f"   parameters (true in parentheses):")
    truth = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}
    for name, val in fit.params.items():
        true = truth.get(name, "?")
        stderr = fit.parameter_stderr.get(name, 0.0)
        rel_err = abs(val - true) / true if isinstance(true, float) else float("nan")
        print(f"     {name:>6} = {val:>10.4f}  (true {true!s:>8}, "
              f"|err|/true={rel_err:.2%}, ±stderr={stderr:.4f})")
    print(f"   RMSE in-sample = {fit.rmse_in_sample:.4f}")
    print(f"   RMSE held-out  = {fit.rmse_held_out:.4f}")

    banner("3. Extrapolate to a 5× larger model and 10× more data")
    ep = s.extrapolate(1.5e11, 3e11)
    target_truth = true_loss(1.5e11, 3e11)
    print(f"   query: N=1.5e11, D=3.0e11")
    print(f"   predicted loss: {ep.loss_point:.4f}")
    print(f"   95% CI:        [{ep.loss_lower:.4f}, {ep.loss_upper:.4f}]")
    print(f"   ground truth:   {target_truth:.4f}")
    in_ci = ep.loss_lower <= target_truth <= ep.loss_upper
    print(f"   truth ∈ CI:     {in_ci}")

    banner("4. Closed-form compute-optimal allocation for C = 1e22 FLOPs")
    co = s.compute_optimal(budget_c=1e22)
    print(f"   budget C = {co.compute_budget:.2e} FLOPs")
    print(f"   analytic N* = {co.n_star_analytic:.3e}")
    print(f"   analytic D* = {co.d_star_analytic:.3e}")
    print(f"   loss at (N*, D*) = {co.loss_at_optimum:.4f}")
    print(f"   numeric sweep agrees: "
          f"N*={co.n_star_numeric:.3e} D*={co.d_star_numeric:.3e} "
          f"L={co.loss_at_numeric:.4f}")

    banner("5. Sweep compute budgets — the scaling frontier")
    print(f"   {'C (FLOPs)':>12}  {'N* (params)':>14}  {'D* (tokens)':>14}  {'L*':>8}")
    for log_c in [18, 19, 20, 21, 22, 23, 24, 25, 26]:
        c = 10.0 ** log_c
        co = s.compute_optimal(budget_c=c)
        print(f"   {c:>12.1e}  {co.n_star_analytic:>14.3e}  "
              f"{co.d_star_analytic:>14.3e}  {co.loss_at_optimum:>8.4f}")

    banner("6. Replay-verifiable certificate")
    cert = s.certificate()
    print(f"   family:                {cert.family}")
    print(f"   in-sample / held-out:  {cert.n_in_sample} / {cert.n_held_out}")
    print(f"   RMSE held-out:         {cert.rmse_held_out:.4f}")
    print(f"   95% LCB (Hoeffding):   {cert.rmse_lcb_hoeffding:.4f}")
    print(f"   95% LCB (Bernstein):   {cert.rmse_lcb_bernstein:.4f}")
    print(f"   N range observed:      {cert.in_range_n[0]:.2e} ... "
          f"{cert.in_range_n[1]:.2e}")
    print(f"   D range observed:      {cert.in_range_d[0]:.2e} ... "
          f"{cert.in_range_d[1]:.2e}")
    print(f"   extrapolation factor:  N×{cert.extrapolation_factor_n:.1f}, "
          f"D×{cert.extrapolation_factor_d:.1f}")
    print(f"   fingerprint:           {cert.fingerprint_hash[:24]}...")

    banner("7. Event stream (replay-verifiable)")
    summary: dict[str, int] = {}
    for k in events_seen:
        summary[k] = summary.get(k, 0) + 1
    for k in sorted(summary):
        print(f"   {k:<30}  ×{summary[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
