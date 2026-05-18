"""Searcher — runtime primitive demos.

Runs each of the supported algorithms on a representative problem:
  * A* / weighted A* / IDA* / B&B on a 2D grid with the Manhattan
    heuristic.
  * UCT / PUCT on the same grid, comparing convergence by visit count.
  * Alpha-beta on a small zero-sum game (subtraction game / Nim).
  * Beam search with a learned value function.
  * Composition: PUCT with a hand-coded "policy prior" that biases
    moves toward the goal.

Each demo prints the principal variation, the best value, the budget
the algorithm consumed, and the certificate hash that pins the run.
"""
from __future__ import annotations

import time

from agi.searcher import (
    Searcher,
    SearcherConfig,
    alphabeta,
    astar,
    beam_search,
    branch_and_bound,
    ida_star,
    make_evaluator,
    puct,
    uct,
    verify_certificate,
    ALGORITHM_ASTAR,
    ALGORITHM_PUCT,
    ALGORITHM_UCT,
)


# -----------------------------------------------------------------------------
# Grid world
# -----------------------------------------------------------------------------


def grid_world(size: int = 6, walls: set | None = None,
               step_reward: float = -1.0, goal_reward: float = 100.0,
               goal: tuple[int, int] | None = None):
    if walls is None:
        walls = set()
    if goal is None:
        goal = (size - 1, size - 1)

    def actions(_s):
        return ["N", "S", "E", "W"]

    def apply_(s, a):
        x, y = s
        if a == "N":
            y -= 1
        elif a == "S":
            y += 1
        elif a == "E":
            x += 1
        elif a == "W":
            x -= 1
        if not (0 <= x < size and 0 <= y < size):
            return s
        if (x, y) in walls:
            return s
        return (x, y)

    def terminal(s):
        return s == goal

    def reward(s):
        return goal_reward if s == goal else step_reward

    def heuristic(s):
        return float(abs(goal[0] - s[0]) + abs(goal[1] - s[1]))

    def value(s):
        # Cheap admissible negative-cost value estimate (used as MCTS leaf).
        return -heuristic(s)

    return dict(actions=actions, apply=apply_, terminal=terminal,
                reward=reward, heuristic=heuristic, value=value,
                key=lambda s: s)


def print_report(name: str, rep) -> None:
    pv = rep.principal_variation
    print(f"  {name:<22}  best_action={rep.best_action!r:<6}  "
          f"value={rep.best_value:+7.2f}  iters={rep.iterations:>4}  "
          f"nodes={rep.budget_used.nodes:>4}  "
          f"cert={rep.certificate[:10]}  PV={'/'.join(map(str, pv))[:32]}")


# -----------------------------------------------------------------------------
# Demo 1: classical search on a grid
# -----------------------------------------------------------------------------


def _classical_args(gw):
    return dict(actions=gw["actions"], apply=gw["apply"],
                terminal=gw["terminal"], heuristic=gw["heuristic"],
                key=gw["key"])


def demo_grid_classical() -> None:
    print("=" * 78)
    print("Demo 1 — Classical search on a 6×6 grid (start=(0,0), goal=(5,5))")
    print("=" * 78)

    gw = grid_world(size=6)

    rep = astar((0, 0), **_classical_args(gw), max_iterations=5_000)
    print_report("A*", rep)

    rep = ida_star((0, 0), **_classical_args(gw), max_iterations=200)
    print_report("IDA*", rep)

    rep_w = astar((0, 0), **_classical_args(gw), weighted=2.0,
                  max_iterations=5_000)
    print_report("Weighted A* (w=2)", rep_w)
    print(f"      → w-suboptimality bound: {rep_w.suboptimality_bound}")

    rep = branch_and_bound((0, 0), **_classical_args(gw),
                           max_iterations=5_000)
    print_report("Branch & Bound", rep)
    print()


# -----------------------------------------------------------------------------
# Demo 2: classical search with obstacles
# -----------------------------------------------------------------------------


def demo_grid_with_walls() -> None:
    print("=" * 78)
    print("Demo 2 — Grid with a wall.  Optimal cost is 10 (forced detour).")
    print("=" * 78)
    walls = {(2, 0), (2, 1), (2, 2), (2, 4), (2, 5)}
    gw = grid_world(size=6, walls=walls)
    rep = astar((0, 0), **_classical_args(gw), max_iterations=10_000)
    print_report("A*", rep)
    print(f"      → walls forced cost {rep.optimal_cost}  (unobstructed = 10)")
    print()


# -----------------------------------------------------------------------------
# Demo 3: UCT / PUCT on the grid
# -----------------------------------------------------------------------------


def demo_mcts_grid() -> None:
    print("=" * 78)
    print("Demo 3 — MCTS variants on the grid (anytime; no heuristic).")
    print("=" * 78)
    gw = grid_world(size=6)

    rep = uct((0, 0), actions=gw["actions"], apply=gw["apply"],
              terminal=gw["terminal"], reward=gw["reward"],
              key=gw["key"], max_iterations=1_000, c_puct=1.4, seed=42)
    print_report("UCT (c=1.4)", rep)
    print(f"      → root visits: {rep.root_visits_by_action}")

    # PUCT with goal-biased prior
    def prior(s, A):
        # Bias toward actions that reduce Manhattan distance.
        from collections import defaultdict
        scores = {}
        for a in A:
            ns = gw["apply"](s, a)
            scores[a] = -gw["heuristic"](ns)
        # Softmax with temperature 1
        import math
        mx = max(scores.values())
        exps = {a: math.exp(scores[a] - mx) for a in A}
        Z = sum(exps.values())
        return {a: exps[a] / Z for a in A}

    rep = puct((0, 0), actions=gw["actions"], apply=gw["apply"],
               terminal=gw["terminal"], reward=gw["reward"],
               policy_prior=prior, key=gw["key"],
               max_iterations=1_000, c_puct=1.25, seed=42)
    print_report("PUCT + goal prior", rep)
    print(f"      → root visits: {rep.root_visits_by_action}")
    print(f"      → root priors: {rep.root_priors_by_action}")

    # PUCT with leaf value (no rollouts needed)
    rep = puct((0, 0), actions=gw["actions"], apply=gw["apply"],
               terminal=gw["terminal"], value=gw["value"],
               policy_prior=prior, key=gw["key"],
               max_iterations=500, c_puct=1.25, seed=42)
    print_report("PUCT + value", rep)
    print()


# -----------------------------------------------------------------------------
# Demo 4: alpha-beta on a subtraction game
# -----------------------------------------------------------------------------


def demo_alphabeta_nim() -> None:
    print("=" * 78)
    print("Demo 4 — Alpha-beta on standard Nim (take 1-3, last stone wins).")
    print("=" * 78)

    def acts(s):
        stones, _ = s
        return [k for k in (1, 2, 3) if k <= stones]

    def app(s, a):
        stones, player = s
        return (stones - a, 1 - player)

    def term(s):
        return s[0] == 0

    def rew(s):
        return -1.0 if s[0] == 0 else 0.0  # side-to-move at 0 lost

    def key(s):
        return s

    for n in (4, 5, 8, 12, 21):
        rep = alphabeta((n, 0), actions=acts, apply=app, terminal=term,
                        reward=rew, key=key, depth=min(n + 1, 16),
                        iterative_deepening=True)
        outcome = "WIN" if rep.best_value > 0 else "LOSS"
        print(f"  Nim(n={n:>2}) → best move = {rep.best_action} "
              f"({outcome}, value={rep.best_value:+.0f})  "
              f"depth={rep.bound}  nodes={rep.budget_used.nodes}  "
              f"cert={rep.certificate[:10]}")
    print()


# -----------------------------------------------------------------------------
# Demo 5: beam search with a value function
# -----------------------------------------------------------------------------


def demo_beam_search() -> None:
    print("=" * 78)
    print("Demo 5 — Beam search on the grid with a value function.")
    print("=" * 78)
    gw = grid_world(size=8)
    for width in (1, 4, 16):
        rep = beam_search((0, 0), actions=gw["actions"], apply=gw["apply"],
                          terminal=gw["terminal"], value=gw["value"],
                          key=gw["key"], width=width, score="value",
                          max_iterations=40)
        print(f"  width={width:>2}  best_action={rep.best_action}  "
              f"value={rep.best_value:+.2f}  PV-len={len(rep.principal_variation)}"
              f"  nodes={rep.budget_used.nodes}")
    print()


# -----------------------------------------------------------------------------
# Demo 6: anytime + budget control
# -----------------------------------------------------------------------------


def demo_anytime() -> None:
    print("=" * 78)
    print("Demo 6 — Anytime: bound the search by wall-clock seconds.")
    print("=" * 78)
    gw = grid_world(size=10)
    for ms in (5, 20, 100):
        seconds = ms / 1000.0
        cfg = SearcherConfig(algorithm=ALGORITHM_UCT,
                             max_iterations=None,
                             max_seconds=seconds, seed=0)
        s = Searcher(cfg)
        t0 = time.time()
        rep = s.search((0, 0), actions=gw["actions"], apply=gw["apply"],
                       terminal=gw["terminal"], reward=gw["reward"],
                       key=gw["key"])
        elapsed = (time.time() - t0) * 1000
        print(f"  budget={ms:>4}ms  used={elapsed:>5.1f}ms  "
              f"iters={rep.iterations:>5}  bound_hit={rep.bound_hit:<14}  "
              f"best={rep.best_action}")
    print()


# -----------------------------------------------------------------------------
# Demo 7: certificate replay (tamper-evidence)
# -----------------------------------------------------------------------------


def demo_certificate_replay() -> None:
    print("=" * 78)
    print("Demo 7 — Certificate replay: two identical runs produce the same")
    print("         certificate; a single byte change breaks it.")
    print("=" * 78)
    gw = grid_world(size=4)
    ev = make_evaluator(actions=gw["actions"], apply=gw["apply"],
                        terminal=gw["terminal"], reward=gw["reward"],
                        key=gw["key"])
    s = Searcher(SearcherConfig(algorithm=ALGORITHM_UCT,
                                max_iterations=200, seed=42))
    rep1 = s.search((0, 0), evaluator=ev)
    print(f"  Run 1 certificate: {rep1.certificate}")
    rep2 = s.search((0, 0), evaluator=ev)
    print(f"  Run 2 certificate: {rep2.certificate}")
    print(f"  identical?  {rep1.certificate == rep2.certificate}")
    print(f"  verify_certificate(rep1, ...): {verify_certificate(rep1, s, (0,0), ev)}")
    s_seeded = Searcher(SearcherConfig(algorithm=ALGORITHM_UCT,
                                       max_iterations=200, seed=43))
    rep3 = s_seeded.search((0, 0), evaluator=ev)
    print(f"  Run 3 (different seed=43) certificate: {rep3.certificate}")
    print(f"  matches Run 1? {rep1.certificate == rep3.certificate}")
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print()
    print("=" * 78)
    print(" Searcher — bounded, anytime, certified tree search demo")
    print("=" * 78)
    print()

    demo_grid_classical()
    demo_grid_with_walls()
    demo_mcts_grid()
    demo_alphabeta_nim()
    demo_beam_search()
    demo_anytime()
    demo_certificate_replay()

    print("=" * 78)
    print(" Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
