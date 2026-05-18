"""Budgeter ⇆ coordination engine — three roles a coordinator plays with
the Budgeter primitive.

The coordination engine wraps the runtime.  It decides, per task,
*which* strategies to draw on and *how much* compute to spend.  This
demo walks three concrete coordinator moves on top of Budgeter:

  A.  *Per-difficulty routing.*  Easy tasks should not pay an MCTS bill;
      hard tasks should not be left with three parallel samples.  The
      Budgeter exposes a difficulty-aware ``recommend(budget, difficulty)``
      that a router can dispatch on.

  B.  *Service-level guarantee.*  Given a target P(pass) the coordinator
      promises to a customer, find the *minimum* budget that hits it on
      the Pareto frontier.  Headline unit-economic question: how much
      does an extra 1% of accuracy cost?

  C.  *Composition with Scaler and Anticipator.*  Budgeter answers the
      *online* test-time axis.  Scaler answers the *training* axis.
      Anticipator answers the *offline* sleep-time axis.  Together they
      span the three places a coordinator can spend a marginal dollar
      and tell it which spend has the steepest ΔP per $.

Run:  python examples/budgeter_coordination_demo.py
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
    STRAT_SEQUENTIAL,
    default_parallel_spec,
    default_sequential_spec,
    default_verifier_spec,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# Ground-truth curves parameterised by difficulty d ∈ [0, 1].
# Easy (d=0):   parallel saturates at 0.99, verifier at 0.995, seq at 1.0
# Hard (d=1):   parallel saturates at 0.55, verifier at 0.78, seq at 0.85
def truth_parallel(units, d):
    p_inf = 0.99 - 0.44 * d
    tau = 6.0 + 18.0 * d
    return p_inf - (p_inf - 0.10) * math.exp(-units / tau)


def truth_verifier(units, d):
    p_inf = 0.995 - 0.215 * d
    r = 0.7 - 0.35 * d
    return p_inf - (p_inf - 0.10) * (1.0 - r) ** max(0, units - 1)


def truth_sequential(units, d):
    p_inf = 1.0 - 0.15 * d
    tau_t = 80.0 + 320.0 * d
    return p_inf - (p_inf - 0.10) * math.exp(-units / tau_t)


def _populate(b, difficulty: float, seed: int = 0):
    rng = random.Random(seed + int(difficulty * 100))
    grids = {
        STRAT_PARALLEL:   [1, 2, 4, 8, 16, 32, 64, 128],
        STRAT_VERIFIER:   [1, 2, 4, 8, 16, 32, 64],
        STRAT_SEQUENTIAL: [16, 32, 64, 128, 256, 512, 1024, 2048],
    }
    truths = {
        STRAT_PARALLEL:   lambda u: truth_parallel(u, difficulty),
        STRAT_VERIFIER:   lambda u: truth_verifier(u, difficulty),
        STRAT_SEQUENTIAL: lambda u: truth_sequential(u, difficulty),
    }
    for s, grid in grids.items():
        for u in grid:
            p = truths[s](u)
            trials = 150
            succ = sum(1 for _ in range(trials) if rng.random() < p)
            b.observe(Observation(strategy=s, difficulty=difficulty,
                                  compute_units=float(u), trials=trials,
                                  successes=succ))


# ---------------------------------------------------------------------------
# A. Per-difficulty routing
# ---------------------------------------------------------------------------


def part_a_routing():
    banner("A.  Per-difficulty routing — coordinator dispatches by difficulty")
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda ev: seen.append(ev.kind))

    # Three Budgeters — one per difficulty bucket.  In production a single
    # Budgeter could pool with `difficulty_kernel_bandwidth > 0`, but for
    # clarity we segregate.
    budgeters = {}
    for label, d in (("EASY", 0.0), ("MEDIUM", 0.5), ("HARD", 1.0)):
        b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0,
                                     holdout_fraction=0.2), bus=bus)
        b.register_strategy(default_parallel_spec(unit_cost=1.00,
                                                  max_units=256))
        b.register_strategy(default_verifier_spec(unit_cost=1.20,
                                                  max_units=128))
        b.register_strategy(default_sequential_spec(unit_cost=0.05,
                                                    max_units=4096))
        _populate(b, d, seed=hash(label) & 0xFFFF)
        b.fit()
        budgeters[label] = b

    budget = 10.0
    print(f"   query budget = ${budget:.2f} per task")
    print(f"   {'difficulty':>12}  {'pass':>6}  {'active':<40}  "
          f"{'oracle':>10}  {'regret':>8}")
    for label, b in budgeters.items():
        alloc = b.allocate(budget=budget)
        cert = b.certificate()
        active = ",".join(alloc.active_strategies) or "(none)"
        regret = f"{cert.regret_ucb:.3f}" if cert.regret_ucb is not None else "n/a"
        oracle = cert.oracle_strategy or "n/a"
        print(f"   {label:>12}  {alloc.predicted_pass:6.3f}  "
              f"{active:<40}  {oracle:>10}  {regret:>8}")

    print()
    print(f"   → Note how the optimal MIX flips: easy tasks lean on cheap")
    print(f"     verifier-rerank, hard tasks shift compute into")
    print(f"     sequential 'think longer' — the Snell 2024 finding.")


# ---------------------------------------------------------------------------
# B. Service-level guarantee — minimum budget that hits target P(pass).
# ---------------------------------------------------------------------------


def part_b_sla():
    banner("B.  Service-level guarantee — what does +1% accuracy cost?")
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0,
                                 holdout_fraction=0.2))
    b.register_strategy(default_parallel_spec(unit_cost=1.00, max_units=256))
    b.register_strategy(default_verifier_spec(unit_cost=1.20, max_units=128))
    b.register_strategy(default_sequential_spec(unit_cost=0.05, max_units=4096))
    _populate(b, difficulty=0.5, seed=7)
    b.fit()

    targets = [0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94]
    # Sweep a fine Pareto frontier and pick the cheapest point clearing
    # each target.
    pts = b.pareto(min_budget=1.0, max_budget=80.0, n_points=24)
    print(f"   {'target P(pass)':>14}  {'min $':>8}  {'achieved P':>11}  "
          f"{'active strategies':>40}")
    last_cost = None
    for t in targets:
        cheapest = None
        for p in pts:
            if p.predicted_pass + 1e-9 >= t and (cheapest is None or
                                                   p.spent < cheapest.spent):
                cheapest = p
        if cheapest is None:
            print(f"   {t:>14.3f}  {'∞':>8}  {'unreachable':>11}  {'':>40}")
        else:
            delta = (f"(+${cheapest.spent - last_cost:.2f})"
                     if last_cost is not None else "")
            active = ",".join(cheapest.active_strategies)
            print(f"   {t:>14.3f}  ${cheapest.spent:>6.2f}  "
                  f"{cheapest.predicted_pass:>11.3f}  {active:>40}  {delta}")
            last_cost = cheapest.spent

    print()
    print(f"   → Each row is a row in your pricing book.  '94% accuracy =")
    print(f"     $X / query' is a sellable SLA underpinned by a PAC bound.")


# ---------------------------------------------------------------------------
# C. Where to spend a marginal dollar — Budgeter vs Scaler vs Anticipator.
# ---------------------------------------------------------------------------


def part_c_three_axes():
    banner("C.  Where to spend a marginal dollar — three compute axes")
    print("   Budgeter  — test-time online compute (samples, votes, search).")
    print("   Scaler    — training-time compute (model size × tokens).")
    print("   Anticipator — sleep-time compute (precompute idle queries).")
    print()
    print("   The three primitives expose the same currency: predicted ΔP(pass)")
    print("   per $.  A coordination engine asks each, picks the steepest.")
    print()

    # Budgeter side: how much does +$1 buy at the current operating point?
    b = Budgeter(BudgeterConfig(seed=42, bootstrap_b=0,
                                 holdout_fraction=0.2))
    b.register_strategy(default_parallel_spec(unit_cost=1.00))
    b.register_strategy(default_verifier_spec(unit_cost=1.20))
    b.register_strategy(default_sequential_spec(unit_cost=0.05))
    _populate(b, difficulty=0.5, seed=11)
    b.fit()

    for current_budget in (4.0, 12.0, 30.0, 80.0):
        alloc_a = b.allocate(budget=current_budget)
        alloc_b = b.allocate(budget=current_budget * 1.10)
        dp = alloc_b.predicted_pass - alloc_a.predicted_pass
        dc = alloc_b.spent - alloc_a.spent
        marginal = dp / dc if dc > 0 else 0.0
        print(f"   at ${current_budget:>5.2f}  →  +10% budget yields "
              f"ΔP={dp:+.4f} for Δ$={dc:+.2f}  =>  ΔP/$ = {marginal:+.5f}")

    print()
    print(f"   → Decreasing marginal returns confirm the curve is concave;")
    print(f"     above ~$30 the test-time axis saturates and the coordinator")
    print(f"     should redirect spend to Scaler (train a better model) or")
    print(f"     Anticipator (precompute the answer at sleep-time).")


def main() -> int:
    part_a_routing()
    part_b_sla()
    part_c_three_axes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
