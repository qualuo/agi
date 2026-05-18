"""Annealer demo — combinatorial optimisation as a runtime primitive.

Walks through five canonical NP-hard benchmarks and shows the same
:class:`Annealer` instance solving all of them with a PAC certificate
on the best-cost gap:

  1. Travelling Salesman (Euclidean, 12 cities)
  2. MaxCut on a random weighted graph
  3. MaxSAT on a 3-SAT instance
  4. 0/1 Knapsack with capacity constraint
  5. Karmarkar-Karp number partitioning

Then shows three algorithmic variants on the same TSP:

  - single-chain Simulated Annealing
  - 4-replica Parallel Tempering
  - Luby restart-wrapped SA

Run from repo root::

    python examples/annealer_demo.py
"""
from __future__ import annotations

import math
import random

from agi.annealer import (
    ALGO_PT,
    ALGO_RESTART,
    ALGO_SA,
    Annealer,
    AnnealerConfig,
    annealer_knapsack,
    annealer_max_cut,
    annealer_max_sat,
    annealer_number_partition,
    annealer_tsp,
)


def section(title: str) -> None:
    print()
    print("─" * 64)
    print(title)
    print("─" * 64)


def show_run(name: str, an: Annealer, prob, *, delta: float = 0.05) -> None:
    rep = an.run(prob)
    cert = an.certify(rep, delta=delta, problem=prob)
    print(f"  problem        : {name}")
    print(f"  algorithm      : {rep.algorithm}")
    print(f"  best cost      : {rep.best_cost:.6f}")
    print(f"  final cost     : {rep.final_cost:.6f}")
    print(f"  acceptance     : {rep.acceptance_rate:.3f}")
    print(f"  iterations     : {rep.iterations}")
    print(f"  proposals      : {rep.proposals}")
    if rep.swaps_attempted > 0:
        print(f"  swap accept    : {rep.swaps_accepted}/{rep.swaps_attempted}")
    if rep.restarts_taken > 0:
        print(f"  restarts taken : {rep.restarts_taken}")
    print(f"  lower bound    : {cert.lower_bound}")
    print(f"  Hoeffding gap  : {cert.gap_hoeffding:.4f} (δ={delta})")
    print(f"  Bernstein gap  : {cert.gap_bernstein:.4f}")
    print(f"  P(globalopt) ≥ : {cert.p_global_opt:.4f}")
    print(f"  chain head     : {rep.chain_head[:16]}…")


def demo_tsp() -> None:
    section("1. Travelling Salesman — 12-city Euclidean instance")
    rng = random.Random(1)
    # Twelve cities arranged in two clusters.
    pts = []
    for _ in range(6):
        pts.append((rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)))
    for _ in range(6):
        pts.append((rng.uniform(4.0, 5.0), rng.uniform(4.0, 5.0)))
    prob = annealer_tsp(pts, seed=0)
    an = Annealer(AnnealerConfig(max_iter=5000, t_init=2.0, t_final=1e-3, seed=0))
    show_run("TSP", an, prob)


def demo_max_cut() -> None:
    section("2. MaxCut — 10-vertex random weighted graph")
    rng = random.Random(2)
    n = 10
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                edges.append((i, j, rng.uniform(0.1, 1.0)))
    prob = annealer_max_cut(edges, n_vertices=n)
    an = Annealer(AnnealerConfig(max_iter=3000, t_init=0.5, t_final=1e-3, seed=0))
    show_run("MaxCut", an, prob)


def demo_max_sat() -> None:
    section("3. MaxSAT — random 3-SAT, 10 vars, 30 clauses")
    rng = random.Random(3)
    n_vars = 10
    clauses = []
    for _ in range(30):
        clause = []
        vs = rng.sample(range(1, n_vars + 1), 3)
        for v in vs:
            clause.append(v if rng.random() < 0.5 else -v)
        clauses.append(clause)
    prob = annealer_max_sat(clauses, n_vars=n_vars)
    an = Annealer(AnnealerConfig(max_iter=4000, t_init=1.0, t_final=1e-3, seed=0))
    show_run("MaxSAT", an, prob)


def demo_knapsack() -> None:
    section("4. 0/1 Knapsack — 15 items, capacity 50")
    rng = random.Random(4)
    weights = [rng.uniform(5.0, 20.0) for _ in range(15)]
    values = [rng.uniform(1.0, 10.0) for _ in range(15)]
    prob = annealer_knapsack(weights, values, capacity=50.0)
    an = Annealer(AnnealerConfig(max_iter=3000, t_init=10.0, t_final=1e-2, seed=0))
    show_run("Knapsack", an, prob)


def demo_partition() -> None:
    section("5. Number Partitioning — 12 weights, target balance = 0")
    rng = random.Random(5)
    w = [rng.uniform(1.0, 20.0) for _ in range(12)]
    prob = annealer_number_partition(w)
    an = Annealer(AnnealerConfig(max_iter=3000, t_init=5.0, t_final=1e-3, seed=0))
    show_run("Partition", an, prob)


def demo_algorithms() -> None:
    section("6. Same TSP — three algorithms")
    rng = random.Random(7)
    pts = [(rng.uniform(0.0, 5.0), rng.uniform(0.0, 5.0)) for _ in range(15)]
    prob = annealer_tsp(pts, seed=0)

    for algo, label in (
        (ALGO_SA, "Simulated Annealing"),
        (ALGO_PT, "Parallel Tempering (K=4)"),
        (ALGO_RESTART, "Luby-restart SA"),
    ):
        an = Annealer(
            AnnealerConfig(
                algorithm=algo,
                max_iter=2000,
                n_replicas=4,
                swap_every=25,
                luby_unit=64,
                t_init=2.0,
                t_final=1e-3,
                seed=0,
            )
        )
        rep = an.run(prob)
        print(f"  {label:30s}  best={rep.best_cost:.4f}   "
              f"accept={rep.acceptance_rate:.2f}   iter={rep.iterations}")


def demo_certificate_tightening() -> None:
    section("7. Certificate tightens with more samples")
    prob = annealer_number_partition([5.0, 4.0, 3.0, 2.0, 6.0, 7.0, 1.0])
    for max_iter in (200, 1000, 5000):
        an = Annealer(AnnealerConfig(max_iter=max_iter, t_init=2.0, t_final=1e-4, seed=0))
        rep = an.run(prob)
        cert = an.certify(rep, delta=0.05, problem=prob)
        print(
            f"  max_iter={max_iter:5d}  best={rep.best_cost:.4f}  "
            f"Bernstein-gap={cert.gap_bernstein:.4f}  "
            f"P_gopt≥{cert.p_global_opt:.3f}"
        )


def main() -> None:
    print()
    print("=" * 64)
    print("Annealer demo — combinatorial optimisation as a runtime primitive")
    print("=" * 64)
    demo_tsp()
    demo_max_cut()
    demo_max_sat()
    demo_knapsack()
    demo_partition()
    demo_algorithms()
    demo_certificate_tightening()
    print()
    print("done.")


if __name__ == "__main__":
    main()
