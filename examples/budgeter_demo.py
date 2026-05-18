"""Budgeter demo: compute-optimal test-time inference allocation.

The pitch in one runnable script (no API key required):

  1. A coordination engine has logs from running five inference
     strategies (parallel best-of-N, self-consistency majority vote,
     verifier-guided rerank, MCTS-of-thoughts, and sequential "think
     longer" CoT) at a handful of compute budgets per strategy.
  2. It asks the runtime: *"If I have C = $20 / query of inference
     compute, what mix should I run and what pass rate can I promise?"*
  3. The :class:`Budgeter` primitive fits per-strategy pass@k curves
     (saturating-exponential, verifier order-statistic, saturating
     power-law, self-consistency logistic), Lagrangian water-fills
     the budget across them, and returns the predicted pass with a
     bootstrap-percentile CI.
  4. A PAC-style certificate bounds the regret vs the per-budget
     oracle (best single strategy at the same total spend), and the
     entire trace is sealed in a SHA-256 fingerprint chain.

Run:  python examples/budgeter_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.budgeter import (
    Budgeter,
    BudgeterConfig,
    Observation,
    STRAT_PARALLEL,
    STRAT_VERIFIER,
    STRAT_MAJORITY,
    STRAT_TREE,
    STRAT_SEQUENTIAL,
    default_majority_spec,
    default_parallel_spec,
    default_sequential_spec,
    default_tree_spec,
    default_verifier_spec,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Ground-truth pass curves used to *synthesize* observations.  In production
# these would be replaced by measurements logged by the coordinator.
# ---------------------------------------------------------------------------


def truth_parallel(k):    return 0.85 - (0.85 - 0.25) * math.exp(-k / 12.0)
def truth_verifier(k):    return 0.92 - (0.92 - 0.25) * (1 - 0.55) ** max(0, k - 1)
def truth_sequential(t):  return 0.95 - (0.95 - 0.25) * math.exp(-t / 220.0)
def truth_tree(n):        return 0.88 - (0.88 - 0.25) / (1.0 + n / 32.0) ** 0.6
def truth_majority(k):    return 0.30 + (0.85 - 0.30) / (1 + math.exp(-(k - 5.0) / 2.0))


def synth(strategy, units_grid, truth_fn, *, trials=120, seed=0):
    rng = random.Random(seed)
    rows = []
    for u in units_grid:
        p = truth_fn(u)
        successes = sum(1 for _ in range(trials) if rng.random() < p)
        rows.append(Observation(strategy=strategy, difficulty=0.0,
                                compute_units=float(u), trials=trials,
                                successes=successes))
    return rows


def main() -> int:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda ev: seen.append(ev.kind))

    banner("1. Coordination engine has 5 strategies × ~8 units each")
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=80,
                                 holdout_fraction=0.2, confidence=0.95),
                 bus=bus)
    # Cost model — sensible defaults; substitute real $$ per call.
    # Sequential is per-token so unit_cost is much smaller (e.g. 0.05);
    # tree/MCTS is heavier because each rollout expands many tokens.
    b.register_strategy(default_parallel_spec(unit_cost=1.00, max_units=512))
    b.register_strategy(default_majority_spec(unit_cost=1.00, max_units=64))
    b.register_strategy(default_verifier_spec(unit_cost=1.20, max_units=128))
    b.register_strategy(default_tree_spec(unit_cost=2.00, max_units=128))
    b.register_strategy(default_sequential_spec(unit_cost=0.05, max_units=4096))

    grids = {
        STRAT_PARALLEL:   [1, 2, 4, 8, 16, 32, 64, 128, 256],
        STRAT_MAJORITY:   [1, 3, 5, 7, 11, 17, 23, 31],
        STRAT_VERIFIER:   [1, 2, 4, 8, 16, 32, 64, 128],
        STRAT_TREE:       [1, 2, 4, 8, 16, 32, 64, 128],
        STRAT_SEQUENTIAL: [16, 32, 64, 128, 256, 512, 1024, 2048],
    }
    truths = {
        STRAT_PARALLEL:   truth_parallel,
        STRAT_MAJORITY:   truth_majority,
        STRAT_VERIFIER:   truth_verifier,
        STRAT_TREE:       truth_tree,
        STRAT_SEQUENTIAL: truth_sequential,
    }

    total = 0
    for i, (s, grid) in enumerate(grids.items()):
        rows = synth(s, grid, truths[s], trials=120, seed=i)
        b.observe(rows)
        total += len(rows)
    print(f"   observations: {total} across {len(grids)} strategies")

    banner("2. Fit per-strategy pass curves with bootstrap-CI machinery")
    fits = b.fit()
    print(f"   {'strategy':>12}  {'p1':>6}  {'p_inf':>6}  {'shape':>14}  "
          f"{'logloss_in':>11}  {'logloss_out':>12}")
    for s in sorted(fits):
        f = fits[s]
        # Shape column changes meaning by strategy.
        shape = ""
        if "tau" in f.params:
            shape = f"tau={f.params['tau']:.2f}"
        elif "tau_t" in f.params:
            shape = f"tau_t={f.params['tau_t']:.0f}"
        elif "r" in f.params:
            shape = f"r={f.params['r']:.3f}"
        elif "n0" in f.params:
            shape = f"n0={f.params['n0']:.1f},γ={f.params['gamma']:.2f}"
        elif "k0" in f.params:
            shape = f"k0={f.params['k0']:.2f}"
        out_str = (f"{f.log_loss_held_out:.4f}"
                   if f.log_loss_held_out is not None else "---")
        print(f"   {s:>12}  {f.params['p1']:6.3f}  {f.params['p_inf']:6.3f}  "
              f"{shape:>14}  {f.log_loss_in_sample:11.4f}  {out_str:>12}")

    banner("3. Allocate a $20 budget — compute-optimal mix")
    alloc = b.allocate(budget=20.0, difficulty=0.0)
    print(f"   budget=${alloc.budget:.2f}  spent=${alloc.spent:.2f}  "
          f"predicted_pass={alloc.predicted_pass:.3f}  "
          f"95% CI=[{alloc.predicted_lower:.3f}, {alloc.predicted_upper:.3f}]")
    print(f"   active: {', '.join(alloc.active_strategies)}")
    print(f"   {'strategy':>12}  {'units':>10}  {'cost':>8}  {'p(alone)':>10}")
    for s in sorted(alloc.per_strategy_units):
        u = alloc.per_strategy_units[s]
        c = alloc.per_strategy_cost[s]
        p = alloc.per_strategy_pass[s]
        if u > 0:
            print(f"   {s:>12}  {u:10.2f}  ${c:6.2f}  {p:10.3f}")

    banner("4. Pareto frontier — $ vs P(pass)")
    pts = b.pareto(min_budget=2.0, max_budget=200.0, n_points=10)
    print(f"   {'budget ($)':>12}  {'spent ($)':>10}  {'P(pass)':>9}  "
          f"{'active strategies':>40}")
    for p in pts:
        active = ",".join(sorted(p.active_strategies)) or "(none)"
        print(f"   {p.budget:>12.2f}  {p.spent:>10.2f}  {p.predicted_pass:>9.3f}  "
              f"{active:>40}")

    banner("5. Replay-verifiable certificate")
    cert = b.certificate()
    print(f"   oracle strategy:     {cert.oracle_strategy}")
    if cert.oracle_pass is not None and cert.regret_ucb is not None:
        print(f"   oracle pass:         {cert.oracle_pass:.4f}")
        print(f"   regret UCB:          {cert.regret_ucb:.4f}")
    print(f"   held-out trials:     {cert.n_held_out}")
    if cert.pass_held_out is not None:
        print(f"   held-out pass rate:  {cert.pass_held_out:.4f}")
        print(f"   95% LCB Hoeffding:   {cert.pass_lcb_hoeffding:.4f}")
        print(f"   95% LCB Bernstein:   {cert.pass_lcb_bernstein:.4f}")
    print(f"   extrapolation factor: ×{cert.extrapolation_factor:.2f}  "
          f"(>1 means querying outside observed range)")
    print(f"   fingerprint:         {cert.fingerprint_hash[:24]}...")

    banner("6. Event stream (replay-verifiable)")
    summary: dict[str, int] = {}
    for k in seen:
        summary[k] = summary.get(k, 0) + 1
    for k in sorted(summary):
        print(f"   {k:<28}  ×{summary[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
