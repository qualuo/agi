"""Curator — runtime primitive demos.

Closes the AlphaGo-Zero-style self-improvement loop together with
Searcher and Distiller:

    Curator   →  proposes new tasks at the ZPD frontier
    Searcher  →  solves them
    Distiller →  compiles solutions into a fast student
    student   →  becomes the policy_prior for the next Searcher call
    Cartographer (not shown) → picks among the Curator's proposals

Runs four scenarios:
  1. ZPD strategy: pinpoint the difficulty band where the agent's
     success probability is near target.
  2. Learning-progress strategy: prioritise tasks where the agent is
     *getting better fastest*.
  3. Thompson-LP: posterior-sampling version of (2).
  4. Composition with Searcher: solve a parameterised maze whose size
     is the Curator's difficulty knob.
"""
from __future__ import annotations

import random
import time

from agi.curator import (
    Cell,
    Curator,
    CuratorConfig,
    learning_progress_curator,
    thompson_lp_curator,
    zpd_curator,
)
from agi.searcher import Searcher, SearcherConfig, ALGORITHM_ASTAR


# -----------------------------------------------------------------------------
# Synthetic competence oracle
# -----------------------------------------------------------------------------


def competence_curve(theta_val: float) -> float:
    """Simulated agent competence: hard-coded sigmoid.

    The agent succeeds with probability sigmoid(-5(theta - 0.5)).
    """
    import math
    return 1.0 / (1.0 + math.exp(5 * (theta_val - 0.5)))


# -----------------------------------------------------------------------------
# Demo 1: ZPD frontier
# -----------------------------------------------------------------------------


def demo_zpd() -> None:
    print("=" * 78)
    print("Demo 1 — ZPD strategy: proposals concentrate at the difficulty")
    print("         where the agent's success probability ≈ target_competence.")
    print("=" * 78)
    rng = random.Random(0)
    cur = zpd_curator(param_lo=(0.0,), param_hi=(1.0,), n_buckets=(20,),
                      target_competence=0.5, target_tolerance=0.1, seed=0)

    def gen(theta): return f"task(theta={theta[0]:.2f})"

    for round_idx in range(20):
        proposals = cur.propose(n=8, generator=gen)
        for p in proposals:
            success = rng.random() < competence_curve(p.theta[0])
            cur.observe(theta=p.theta, success=success,
                        predicted_competence=p.predicted_competence)

    # Final report
    rep = cur.report()
    front = cur.frontier()
    front_thetas = [cur.cell_centre(cell)[0] for cell in front]
    print(f"  After 20 rounds, {rep.n_observations} observations, "
          f"{rep.n_cells_explored}/{rep.n_cells_total} cells touched.")
    print(f"  Frontier (CI brackets target=0.5±0.1): "
          f"theta ∈ {sorted([round(t, 2) for t in front_thetas])}")
    print(f"  Most competent cell: theta={cur.cell_centre(Cell(indices=tuple(rep.most_competent_cell)))[0]:.2f} "
          f"(value={rep.most_competent_value:.2f})")
    bs = rep.brier_score
    if bs is not None:
        print(f"  Brier score over last 64: {bs:.3f}  (lower is better-calibrated)")
    print()


# -----------------------------------------------------------------------------
# Demo 2: Learning-progress
# -----------------------------------------------------------------------------


def demo_learning_progress() -> None:
    print("=" * 78)
    print("Demo 2 — Learning-progress strategy: proposals concentrate where")
    print("         the agent is *getting better fastest*.")
    print("=" * 78)
    rng = random.Random(0)
    cur = learning_progress_curator(
        param_lo=(0.0,), param_hi=(1.0,), n_buckets=(10,),
        target_competence=0.5, target_tolerance=0.2,
        recent_window=5, prev_window=5, seed=0,
    )

    def gen(theta): return f"task(theta={theta[0]:.2f})"

    # Simulate: agent slowly improves on theta=0.3..0.5 over time
    for round_idx in range(15):
        proposals = cur.propose(n=6, generator=gen)
        for p in proposals:
            t = p.theta[0]
            # Base competence + slow learning bonus near 0.4
            base = competence_curve(t)
            bonus = 0.05 * round_idx if 0.3 < t < 0.5 else 0.0
            p_success = min(1.0, base + bonus)
            success = rng.random() < p_success
            cur.observe(theta=p.theta, success=success,
                        predicted_competence=p.predicted_competence)

    # Show LP per cell
    print("  Learning progress per cell:")
    for cell in sorted(cur.cells(), key=lambda c: c.indices):
        est = cur.cells()[cell]
        if est.n == 0:
            continue
        lp = est.learning_progress()
        theta_c = cur.cell_centre(cell)[0]
        marker = " ←high LP" if lp > 0.1 else ""
        print(f"    theta={theta_c:.2f}  n={est.n:>3}  mean={est.mean:.2f}  LP={lp:.3f}{marker}")
    print()


# -----------------------------------------------------------------------------
# Demo 3: Thompson over LP
# -----------------------------------------------------------------------------


def demo_thompson_lp() -> None:
    print("=" * 78)
    print("Demo 3 — Thompson-LP: posterior-sampled LP-based proposals")
    print("=" * 78)
    rng = random.Random(0)
    cur = thompson_lp_curator(
        param_lo=(0.0,), param_hi=(1.0,), n_buckets=(10,),
        target_competence=0.5, target_tolerance=0.2,
        recent_window=8, prev_window=8, seed=0,
    )

    def gen(theta): return f"task(theta={theta[0]:.2f})"

    visits_per_cell = {}
    for round_idx in range(20):
        proposals = cur.propose(n=4, generator=gen)
        for p in proposals:
            visits_per_cell[p.cell.indices] = visits_per_cell.get(p.cell.indices, 0) + 1
            success = rng.random() < competence_curve(p.theta[0])
            cur.observe(theta=p.theta, success=success,
                        predicted_competence=p.predicted_competence)

    print("  Visits per cell (Thompson exploration):")
    for cell_idx in sorted(visits_per_cell):
        theta_c = (cell_idx[0] + 0.5) / 10
        n = visits_per_cell[cell_idx]
        bars = "█" * n
        print(f"    theta={theta_c:.2f}  visits={n:>2} {bars}")
    print()


# -----------------------------------------------------------------------------
# Demo 4: Composition with Searcher — maze whose size is the difficulty
# -----------------------------------------------------------------------------


def demo_composition() -> None:
    print("=" * 78)
    print("Demo 4 — Curator + Searcher: maze size as difficulty knob.")
    print("         Difficulty θ ∈ [0,1] maps to maze size n = 4..16.")
    print("         Agent (A*) succeeds if it finds a path under a fixed budget.")
    print("=" * 78)

    rng = random.Random(0)
    cur = zpd_curator(param_lo=(0.0,), param_hi=(1.0,), n_buckets=(13,),
                      target_competence=0.5, target_tolerance=0.2, seed=0)

    def gen(theta):
        size = int(4 + theta[0] * 12)  # 4..16
        return ("maze", size)

    def oracle(task) -> bool:
        _, size = task
        # Agent solves via A* with iteration budget that *barely* covers
        # the optimal path for small mazes.
        goal = (size - 1, size - 1)
        budget = 50
        def acts(_s): return ["N", "S", "E", "W"]
        def app(s, a):
            x, y = s
            if a == "N": y -= 1
            elif a == "S": y += 1
            elif a == "E": x += 1
            elif a == "W": x -= 1
            if not (0 <= x < size and 0 <= y < size):
                return s
            return (x, y)
        def term(s): return s == goal
        def heur(s): return float(abs(goal[0] - s[0]) + abs(goal[1] - s[1]))
        s = Searcher(SearcherConfig(algorithm=ALGORITHM_ASTAR,
                                    max_iterations=budget, seed=0))
        rep = s.search((0, 0), actions=acts, apply=app, terminal=term,
                       heuristic=heur, key=lambda s: s)
        return rep.optimal_cost is not None

    print(f"  {'round':>5}  {'proposal_thetas':<40}  successes_so_far")
    print("  " + "-" * 78)
    successes = 0
    total = 0
    for round_idx in range(6):
        proposals = cur.propose(n=4, generator=gen)
        thetas_str = ",".join(f"{p.theta[0]:.2f}" for p in proposals)
        for p in proposals:
            ok = oracle(p.task)
            total += 1
            if ok:
                successes += 1
            cur.observe(theta=p.theta, success=ok,
                        predicted_competence=p.predicted_competence)
        print(f"  {round_idx+1:>5}  {thetas_str:<40}  {successes}/{total}")

    rep = cur.report()
    print()
    print(f"  Final frontier: {[cur.cell_centre(c)[0] for c in cur.frontier()]}")
    if rep.most_competent_value is not None:
        idx = tuple(rep.most_competent_cell)
        print(f"  Most competent maze size: theta="
              f"{cur.cell_centre(Cell(indices=idx))[0]:.2f}, "
              f"competence={rep.most_competent_value:.2f}")
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print()
    print("=" * 78)
    print(" Curator — automated curriculum generation demo")
    print("=" * 78)
    print()

    demo_zpd()
    demo_learning_progress()
    demo_thompson_lp()
    demo_composition()

    print("=" * 78)
    print(" Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
