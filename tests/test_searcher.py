"""Tests for agi.searcher — bounded-anytime certified tree search."""
from __future__ import annotations

import math
import random
import time
import unittest

from agi.searcher import (
    ALGORITHM_ALPHABETA,
    ALGORITHM_ASTAR,
    ALGORITHM_AUTO,
    ALGORITHM_BEAM,
    ALGORITHM_BNB,
    ALGORITHM_IDA_STAR,
    ALGORITHM_PUCT,
    ALGORITHM_UCT,
    BudgetUsed,
    Evaluator,
    InvalidConfig,
    InvalidEvaluator,
    KNOWN_ALGORITHMS,
    SearchNode,
    SearchReport,
    Searcher,
    SearcherConfig,
    StopCondition,
    UnknownAlgorithm,
    alphabeta,
    astar,
    beam_search,
    branch_and_bound,
    ida_star,
    make_evaluator,
    puct,
    uct,
    verify_certificate,
)


# =============================================================================
# Reusable problem fixtures
# =============================================================================


def grid_actions(_s):
    return ["N", "S", "E", "W"]


def grid_apply(s, a, *, size=4):
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
    return (x, y)


def make_grid_problem(size=4, goal=None, terminal_reward=100.0, step_reward=-1.0):
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
        return (x, y)

    def terminal(s):
        return s == goal

    def reward(s):
        return terminal_reward if s == goal else step_reward

    def heuristic(s):
        return float(abs(goal[0] - s[0]) + abs(goal[1] - s[1]))

    def key(s):
        return s

    return actions, apply_, terminal, reward, heuristic, key


def nim_problem(misere=False):
    """Standard nim: take 1–3 stones, last stone wins.  If misere, loses."""
    def acts(s):
        stones, _ = s
        return [k for k in (1, 2, 3) if k <= stones]

    def app(s, a):
        stones, player = s
        return (stones - a, 1 - player)

    def term(s):
        return s[0] == 0

    def rew(s):
        if s[0] == 0:
            # In standard nim, the side-to-move at 0 LOST (opponent took last).
            # In misère, they WON.
            return 1.0 if misere else -1.0
        return 0.0

    def key(s):
        return s

    return acts, app, term, rew, key


# =============================================================================
# Configuration validation
# =============================================================================


class TestSearcherConfig(unittest.TestCase):

    def test_known_algorithms_includes_all(self):
        for a in ("astar", "ida_star", "uct", "puct", "alphabeta", "beam", "bnb", "auto"):
            self.assertIn(a, KNOWN_ALGORITHMS)

    def test_default_config_is_puct(self):
        cfg = SearcherConfig()
        self.assertEqual(cfg.algorithm, ALGORITHM_PUCT)
        self.assertEqual(cfg.max_iterations, 1024)

    def test_invalid_algorithm_rejected(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(algorithm="totally_not_an_algorithm")

    def test_invalid_c_puct_rejected(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(c_puct=0)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(c_puct=-1)

    def test_invalid_rollout_params_rejected(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(rollout_depth=-1)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(rollouts_per_leaf=0)

    def test_invalid_dirichlet_rejected(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(dirichlet_alpha=-0.1)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(dirichlet_epsilon=1.1)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(dirichlet_epsilon=-0.1)

    def test_invalid_widen_rejected(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(widen_alpha=0)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(widen_alpha=1.1)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(widen_k=0)

    def test_invalid_weighted_astar_rejected(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(weighted_astar=0.5)

    def test_invalid_ab_depth_rejected(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(ab_depth=0)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(aspiration_window=-0.1)

    def test_invalid_beam_rejected(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(beam_width=0)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(beam_score="random")

    def test_max_iterations_must_be_positive_or_none(self):
        SearcherConfig(max_iterations=None)
        SearcherConfig(max_iterations=1)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(max_iterations=0)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(max_iterations=-5)

    def test_max_seconds_must_be_positive(self):
        SearcherConfig(max_seconds=0.01)
        SearcherConfig(max_seconds=None)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(max_seconds=0)
        with self.assertRaises(InvalidConfig):
            SearcherConfig(max_seconds=-1)


# =============================================================================
# Stop-condition
# =============================================================================


class TestStopCondition(unittest.TestCase):

    def test_empty_stop_condition(self):
        s = StopCondition()
        self.assertTrue(s.is_empty())

    def test_nonempty_stop_condition(self):
        self.assertFalse(StopCondition(max_iterations=10).is_empty())
        self.assertFalse(StopCondition(max_seconds=0.1).is_empty())

    def test_negative_deadline_rejected(self):
        with self.assertRaises(InvalidConfig):
            StopCondition(deadline=-1)


# =============================================================================
# A*
# =============================================================================


class TestAStar(unittest.TestCase):

    def test_astar_finds_optimal_path_in_grid(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep = astar((0, 0), actions=a, apply=app, terminal=term,
                    heuristic=h, key=k, max_iterations=1000)
        self.assertEqual(rep.optimal_cost, 6.0)  # Manhattan distance
        self.assertEqual(len(rep.principal_variation), 6)
        self.assertIn(rep.best_action, ("S", "E"))  # symmetric optima

    def test_astar_finds_path_in_5x5(self):
        a, app, term, _r, h, k = make_grid_problem(size=5)
        rep = astar((0, 0), actions=a, apply=app, terminal=term,
                    heuristic=h, key=k, max_iterations=2000)
        self.assertEqual(rep.optimal_cost, 8.0)

    def test_astar_certificate_deterministic(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep1 = astar((0, 0), actions=a, apply=app, terminal=term,
                     heuristic=h, key=k, max_iterations=1000, seed=0)
        rep2 = astar((0, 0), actions=a, apply=app, terminal=term,
                     heuristic=h, key=k, max_iterations=1000, seed=0)
        self.assertEqual(rep1.certificate, rep2.certificate)
        self.assertEqual(rep1.principal_variation, rep2.principal_variation)

    def test_astar_without_heuristic_is_dijkstra(self):
        a, app, term, _r, _h, k = make_grid_problem(size=4)
        rep = astar((0, 0), actions=a, apply=app, terminal=term,
                    key=k, max_iterations=1000)
        self.assertEqual(rep.optimal_cost, 6.0)

    def test_astar_no_solution_returns_no_action(self):
        # Wall everything: actions return [], so no successors.
        rep = astar(0, actions=lambda s: [], apply=lambda s, a: s,
                    terminal=lambda s: False, key=lambda s: s,
                    max_iterations=100)
        self.assertIsNone(rep.best_action)
        self.assertIsNone(rep.optimal_cost)

    def test_astar_requires_terminal(self):
        with self.assertRaises(InvalidEvaluator):
            astar(0, actions=lambda s: [],
                  apply=lambda s, a: s,
                  terminal=None, max_iterations=10)

    def test_weighted_astar_records_suboptimality(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep = astar((0, 0), actions=a, apply=app, terminal=term,
                    heuristic=h, key=k, weighted=2.0, max_iterations=1000)
        self.assertEqual(rep.suboptimality_bound, 2.0)

    def test_astar_budget_respected(self):
        a, app, term, _r, h, k = make_grid_problem(size=10)
        rep = astar((0, 0), actions=a, apply=app, terminal=term,
                    heuristic=h, key=k, max_iterations=5)
        self.assertLessEqual(rep.iterations, 5)

    def test_astar_negative_cost_rejected(self):
        def bad_cost(s, a, ss): return -1.0
        a, app, term, _r, h, k = make_grid_problem(size=4)
        with self.assertRaises(InvalidEvaluator):
            Searcher(SearcherConfig(algorithm=ALGORITHM_ASTAR)).search(
                (0, 0), actions=a, apply=app, terminal=term, heuristic=h,
                cost=bad_cost, key=k,
            )

    def test_astar_negative_heuristic_rejected(self):
        def bad_h(s): return -1.0
        a, app, term, _r, _h, k = make_grid_problem(size=4)
        with self.assertRaises(InvalidEvaluator):
            Searcher(SearcherConfig(algorithm=ALGORITHM_ASTAR)).search(
                (0, 0), actions=a, apply=app, terminal=term, heuristic=bad_h,
                key=k,
            )


# =============================================================================
# IDA*
# =============================================================================


class TestIDAStar(unittest.TestCase):

    def test_ida_star_finds_optimal_path(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep = ida_star((0, 0), actions=a, apply=app, terminal=term,
                       heuristic=h, key=k, max_iterations=200)
        self.assertEqual(rep.optimal_cost, 6.0)
        self.assertEqual(len(rep.principal_variation), 6)

    def test_ida_star_certificate_deterministic(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep1 = ida_star((0, 0), actions=a, apply=app, terminal=term,
                        heuristic=h, key=k, max_iterations=200, seed=1)
        rep2 = ida_star((0, 0), actions=a, apply=app, terminal=term,
                        heuristic=h, key=k, max_iterations=200, seed=1)
        self.assertEqual(rep1.certificate, rep2.certificate)


# =============================================================================
# Branch and bound
# =============================================================================


class TestBranchAndBound(unittest.TestCase):

    def test_bnb_optimal_cost(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep = branch_and_bound((0, 0), actions=a, apply=app, terminal=term,
                               heuristic=h, key=k, max_iterations=1000)
        self.assertEqual(rep.optimal_cost, 6.0)

    def test_bnb_handles_multiple_goals(self):
        def acts(s): return ["A", "B"]
        def app(s, a): return s + 1
        def term(s): return s >= 3
        rep = branch_and_bound(0, actions=acts, apply=app, terminal=term,
                               key=lambda s: s, max_iterations=100)
        self.assertEqual(rep.optimal_cost, 3.0)


# =============================================================================
# UCT
# =============================================================================


class TestUCT(unittest.TestCase):

    def test_uct_returns_legal_action(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = uct((0, 0), actions=a, apply=app, terminal=term,
                  reward=r, key=k, max_iterations=200, c_puct=1.4, seed=42)
        self.assertIn(rep.best_action, ("N", "S", "E", "W"))
        # certificate consistent
        self.assertEqual(len(rep.certificate), 64)  # SHA-256 hex

    def test_uct_certificate_deterministic_under_seed(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep1 = uct((0, 0), actions=a, apply=app, terminal=term,
                   reward=r, key=k, max_iterations=200, c_puct=1.4, seed=7)
        rep2 = uct((0, 0), actions=a, apply=app, terminal=term,
                   reward=r, key=k, max_iterations=200, c_puct=1.4, seed=7)
        self.assertEqual(rep1.certificate, rep2.certificate)
        self.assertEqual(rep1.best_action, rep2.best_action)

    def test_uct_certificate_differs_on_different_seed(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep1 = uct((0, 0), actions=a, apply=app, terminal=term,
                   reward=r, key=k, max_iterations=200, seed=1)
        rep2 = uct((0, 0), actions=a, apply=app, terminal=term,
                   reward=r, key=k, max_iterations=200, seed=2)
        self.assertNotEqual(rep1.certificate, rep2.certificate)

    def test_uct_root_q_visits_dicts_populated(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = uct((0, 0), actions=a, apply=app, terminal=term,
                  reward=r, key=k, max_iterations=400, seed=0)
        self.assertEqual(set(rep.root_visits_by_action.keys()),
                         {"N", "S", "E", "W"})
        self.assertEqual(set(rep.root_q_by_action.keys()),
                         {"N", "S", "E", "W"})

    def test_uct_regret_bound_reported(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = uct((0, 0), actions=a, apply=app, terminal=term,
                  reward=r, key=k, max_iterations=200, seed=0)
        self.assertIsNotNone(rep.regret_bound)
        self.assertGreater(rep.regret_bound, 0)

    def test_uct_two_armed_finds_better_arm(self):
        # Two actions: A always gives high reward, B always gives low.
        def acts(_s): return ["A", "B"]
        def app(s, a): return (s[0] + (1 if a == "A" else 0),
                               s[1] + (1 if a == "B" else 0),
                               s[2] - 1)
        def term(s): return s[2] <= 0
        def rew(s): return s[0] - s[1]  # higher when more A's were chosen
        rep = uct((0, 0, 4), actions=acts, apply=app, terminal=term,
                  reward=rew, key=lambda s: s, max_iterations=400, c_puct=0.8,
                  seed=0)
        self.assertEqual(rep.best_action, "A")
        # most visits should be on A
        self.assertGreater(rep.root_visits_by_action["A"],
                           rep.root_visits_by_action["B"])

    def test_uct_history_increasing(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = uct((0, 0), actions=a, apply=app, terminal=term,
                  reward=r, key=k, max_iterations=200, seed=0)
        # history populated, with iterations strictly increasing
        prev = 0
        for it, _act, _val in rep.history:
            self.assertGreater(it, prev)
            prev = it

    def test_uct_no_reward_no_value_gives_zero_estimates(self):
        # When neither value nor reward is supplied, leaf eval returns 0
        a, app, term, _r, _h, k = make_grid_problem(size=4)
        rep = Searcher(SearcherConfig(
            algorithm=ALGORITHM_UCT, max_iterations=20, seed=0,
        )).search((0, 0), actions=a, apply=app, terminal=term, key=k)
        self.assertEqual(rep.best_value, 0.0)


# =============================================================================
# PUCT
# =============================================================================


class TestPUCT(unittest.TestCase):

    def test_puct_runs(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = puct((0, 0), actions=a, apply=app, terminal=term,
                   reward=r, key=k, max_iterations=200, c_puct=1.25, seed=0)
        self.assertIn(rep.best_action, ("N", "S", "E", "W"))

    def test_puct_with_uniform_prior_matches_uct(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        uniform = lambda s, A: {act: 1.0 / len(A) for act in A}
        rep_p = puct((0, 0), actions=a, apply=app, terminal=term,
                     reward=r, key=k, max_iterations=400, policy_prior=uniform,
                     seed=42)
        # PUCT with uniform prior should pick something reasonable
        self.assertIn(rep_p.best_action, ("S", "E"))

    def test_puct_uses_supplied_prior(self):
        # Strong prior for one action — should drive its visit count up.
        # Each action leads to a distinct state so transposition doesn't
        # conflate visit counts.
        def acts(s): return ["A", "B", "C"] if s == 0 else []
        def app(s, a): return ("A", 1) if a == "A" else (("B", 1) if a == "B" else ("C", 1))
        def term(s): return s != 0
        def rew(s): return 0.0
        def prior(_s, A): return {"A": 0.9, "B": 0.05, "C": 0.05}
        rep = puct(0, actions=acts, apply=app, terminal=term,
                   reward=rew, policy_prior=prior, key=lambda s: s,
                   max_iterations=200, c_puct=2.5, seed=0)
        # The prior should drive A to the highest visit count.
        self.assertEqual(rep.best_action, "A")
        self.assertGreater(rep.root_visits_by_action["A"],
                           rep.root_visits_by_action["B"])

    def test_puct_dirichlet_root_noise(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        # Two different seeds for Dirichlet should give different priors
        rep_a = puct((0, 0), actions=a, apply=app, terminal=term,
                     reward=r, key=k, max_iterations=100, c_puct=1.25,
                     dirichlet_alpha=0.3, dirichlet_epsilon=0.5, seed=1)
        rep_b = puct((0, 0), actions=a, apply=app, terminal=term,
                     reward=r, key=k, max_iterations=100, c_puct=1.25,
                     dirichlet_alpha=0.3, dirichlet_epsilon=0.5, seed=2)
        # Certificates differ (noise differs).
        self.assertNotEqual(rep_a.certificate, rep_b.certificate)

    def test_puct_priors_populated_in_report(self):
        def acts(_s): return ["A", "B"]
        def app(s, a): return s + 1
        def term(s): return s >= 1
        def prior(_s, A): return {"A": 0.7, "B": 0.3}
        def rew(_s): return 0.0
        rep = puct(0, actions=acts, apply=app, terminal=term, reward=rew,
                   policy_prior=prior, key=lambda s: s,
                   max_iterations=50, c_puct=1.25, seed=0)
        self.assertAlmostEqual(rep.root_priors_by_action["A"], 0.7, places=6)
        self.assertAlmostEqual(rep.root_priors_by_action["B"], 0.3, places=6)


# =============================================================================
# Alpha-beta
# =============================================================================


class TestAlphaBeta(unittest.TestCase):

    def test_alphabeta_nim_wins_when_winning(self):
        acts, app, term, rew, key = nim_problem()
        # Standard nim, take 1-3, last stone wins.  5 stones: P1 wins
        # by taking 1, leaving 4.
        rep = alphabeta((5, 0), actions=acts, apply=app, terminal=term,
                        reward=rew, key=key, depth=6)
        self.assertEqual(rep.best_action, 1)
        self.assertEqual(rep.best_value, 1.0)

    def test_alphabeta_nim_loses_when_losing(self):
        acts, app, term, rew, key = nim_problem()
        # 4 stones is losing for side-to-move (any move 1/2/3 leaves 1/2/3
        # → opponent clears).
        rep = alphabeta((4, 0), actions=acts, apply=app, terminal=term,
                        reward=rew, key=key, depth=6)
        self.assertEqual(rep.best_value, -1.0)

    def test_alphabeta_certificate_deterministic(self):
        acts, app, term, rew, key = nim_problem()
        rep1 = alphabeta((10, 0), actions=acts, apply=app, terminal=term,
                         reward=rew, key=key, depth=11, seed=0)
        rep2 = alphabeta((10, 0), actions=acts, apply=app, terminal=term,
                         reward=rew, key=key, depth=11, seed=0)
        self.assertEqual(rep1.certificate, rep2.certificate)

    def test_alphabeta_requires_reward_and_terminal(self):
        with self.assertRaises(InvalidEvaluator):
            Searcher(SearcherConfig(algorithm=ALGORITHM_ALPHABETA, ab_depth=2)).search(
                0, actions=lambda s: [],
                apply=lambda s, a: s,
                terminal=lambda s: True, reward=None,
            )

    def test_alphabeta_iterative_deepening_records_depth(self):
        acts, app, term, rew, key = nim_problem()
        rep = alphabeta((10, 0), actions=acts, apply=app, terminal=term,
                        reward=rew, key=key, depth=8, iterative_deepening=True)
        self.assertIsNotNone(rep.bound)
        self.assertGreater(rep.bound, 0)

    def test_alphabeta_transposition_table_used(self):
        # The transposition table should make repeated searches cheap.
        acts, app, term, rew, key = nim_problem()
        rep = alphabeta((15, 0), actions=acts, apply=app, terminal=term,
                        reward=rew, key=key, depth=5)
        self.assertGreater(rep.budget_used.nodes, 0)


# =============================================================================
# Beam search
# =============================================================================


class TestBeamSearch(unittest.TestCase):

    def test_beam_finds_a_path(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = beam_search((0, 0), actions=a, apply=app, terminal=term,
                          reward=r, value=None, key=k, width=8, score="value",
                          max_iterations=50)
        self.assertIsNotNone(rep.best_action)

    def test_beam_with_value_evaluator(self):
        def acts(_s): return ["A", "B"]
        def app(s, a): return s + (1 if a == "A" else -1)
        def term(s): return abs(s) >= 5
        def val(s): return float(s)
        rep = beam_search(0, actions=acts, apply=app, terminal=term,
                          value=val, key=lambda s: s, width=4, score="value",
                          max_iterations=20)
        self.assertEqual(rep.best_action, "A")

    def test_beam_with_cost_score(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        # 'cost' score requires cost or heuristic.
        rep = beam_search((0, 0), actions=a, apply=app, terminal=term,
                          heuristic=h, key=k, width=8, score="cost",
                          max_iterations=50)
        self.assertIsNotNone(rep.best_action)

    def test_beam_invalid_score(self):
        with self.assertRaises(InvalidConfig):
            SearcherConfig(beam_score="other")

    def test_beam_width_one_is_greedy(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep = beam_search((0, 0), actions=a, apply=app, terminal=term,
                          heuristic=h, key=k, width=1, score="cost",
                          max_iterations=20)
        self.assertIsNotNone(rep.best_action)


# =============================================================================
# Auto-pick
# =============================================================================


class TestAuto(unittest.TestCase):

    def test_auto_picks_astar_when_heuristic_given(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        s = Searcher(SearcherConfig(algorithm=ALGORITHM_AUTO))
        rep = s.search((0, 0), actions=a, apply=app, terminal=term,
                       heuristic=h, key=k)
        self.assertEqual(rep.algorithm, ALGORITHM_ASTAR)

    def test_auto_picks_puct_when_prior_given(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        prior = lambda s, A: {act: 1.0 / len(A) for act in A}
        s = Searcher(SearcherConfig(algorithm=ALGORITHM_AUTO))
        rep = s.search((0, 0), actions=a, apply=app, terminal=term,
                       reward=r, policy_prior=prior, key=k)
        self.assertEqual(rep.algorithm, ALGORITHM_PUCT)

    def test_auto_picks_uct_when_only_reward(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        s = Searcher(SearcherConfig(algorithm=ALGORITHM_AUTO))
        rep = s.search((0, 0), actions=a, apply=app, terminal=term,
                       reward=r, key=k)
        self.assertEqual(rep.algorithm, ALGORITHM_UCT)

    def test_auto_raises_when_no_evaluator_suffices(self):
        # No heuristic, no policy_prior, no value, no reward, no terminal.
        s = Searcher(SearcherConfig(algorithm=ALGORITHM_AUTO))
        with self.assertRaises(InvalidEvaluator):
            s.search(0, actions=lambda s: [], apply=lambda s, a: s)


# =============================================================================
# Budgets & bounds
# =============================================================================


class TestBudgets(unittest.TestCase):

    def test_max_iterations_respected(self):
        a, app, term, r, _h, k = make_grid_problem(size=10)
        rep = uct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                  key=k, max_iterations=37, seed=0)
        self.assertLessEqual(rep.iterations, 37)
        # Bound was hit
        if rep.iterations == 37:
            self.assertEqual(rep.bound_hit, "max_iterations")

    def test_max_seconds_respected(self):
        # A cycle that never terminates and has lots of actions.
        def acts(_s): return list(range(8))
        def app(s, a): return s
        def term(_s): return False
        def rew(_s): return 0.0
        start = time.time()
        rep = uct(0, actions=acts, apply=app, terminal=term, reward=rew,
                  key=lambda s: s, max_iterations=10 ** 9, max_seconds=0.1,
                  seed=0)
        self.assertLessEqual(time.time() - start, 1.0)

    def test_deadline_respected(self):
        def acts(_s): return list(range(4))
        def app(s, a): return s
        def term(_s): return False
        def rew(_s): return 0.0
        cfg = SearcherConfig(algorithm=ALGORITHM_UCT,
                             max_iterations=None,
                             max_seconds=None,
                             deadline=time.time() + 0.1,
                             seed=0)
        start = time.time()
        rep = Searcher(cfg).search(0, actions=acts, apply=app, terminal=term,
                                   reward=rew, key=lambda s: s)
        self.assertLessEqual(time.time() - start, 1.0)


# =============================================================================
# Certificates
# =============================================================================


class TestCertificates(unittest.TestCase):

    def test_certificate_changes_with_iterations(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep1 = uct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                   key=k, max_iterations=100, seed=0)
        rep2 = uct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                   key=k, max_iterations=200, seed=0)
        self.assertNotEqual(rep1.certificate, rep2.certificate)

    def test_certificate_hex_length(self):
        # SHA-256 = 64 hex chars
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = uct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                  key=k, max_iterations=50, seed=0)
        self.assertEqual(len(rep.certificate), 64)
        int(rep.certificate, 16)  # must parse as hex

    def test_certificate_with_secret_key_differs(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        cfg = SearcherConfig(algorithm=ALGORITHM_UCT, max_iterations=50,
                             seed=0, secret_key=b"key-a")
        rep_a = Searcher(cfg).search((0, 0), actions=a, apply=app,
                                     terminal=term, reward=r, key=k)
        cfg = SearcherConfig(algorithm=ALGORITHM_UCT, max_iterations=50,
                             seed=0, secret_key=b"key-b")
        rep_b = Searcher(cfg).search((0, 0), actions=a, apply=app,
                                     terminal=term, reward=r, key=k)
        self.assertNotEqual(rep_a.certificate, rep_b.certificate)

    def test_verify_certificate_replays(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        ev = make_evaluator(actions=a, apply=app, terminal=term, reward=r,
                            key=k)
        searcher = Searcher(SearcherConfig(
            algorithm=ALGORITHM_UCT, max_iterations=50, seed=42,
        ))
        rep = searcher.search((0, 0), evaluator=ev)
        self.assertTrue(verify_certificate(rep, searcher, (0, 0), ev))

    def test_verify_certificate_detects_tampering(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        ev = make_evaluator(actions=a, apply=app, terminal=term, reward=r,
                            key=k)
        searcher = Searcher(SearcherConfig(
            algorithm=ALGORITHM_UCT, max_iterations=50, seed=42,
        ))
        rep = searcher.search((0, 0), evaluator=ev)
        # Forge a certificate
        rep_tampered = SearchReport(
            algorithm=rep.algorithm,
            best_action=rep.best_action,
            best_value=rep.best_value,
            principal_variation=rep.principal_variation,
            iterations=rep.iterations,
            budget_used=rep.budget_used,
            certificate="0" * 64,
            seed=rep.seed,
            finished=rep.finished,
            bound_hit=rep.bound_hit,
        )
        self.assertFalse(verify_certificate(rep_tampered, searcher, (0, 0), ev))


# =============================================================================
# Reports
# =============================================================================


class TestReports(unittest.TestCase):

    def test_report_as_dict_serializes(self):
        import json as _json
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = uct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                  key=k, max_iterations=100, seed=0)
        d = rep.as_dict()
        s = _json.dumps(d)  # must be JSON-serializable
        self.assertIn("algorithm", s)
        self.assertIn("certificate", s)

    def test_budget_used_records_iterations_and_nodes(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = uct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                  key=k, max_iterations=20, seed=0)
        self.assertEqual(rep.budget_used.iterations, 20)
        self.assertGreater(rep.budget_used.nodes, 0)


# =============================================================================
# Searcher orchestrator
# =============================================================================


class TestSearcher(unittest.TestCase):

    def test_default_searcher_works(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        s = Searcher()
        rep = s.search((0, 0), actions=a, apply=app, terminal=term,
                       reward=r, key=k)
        self.assertIn(rep.best_action, ("N", "S", "E", "W"))

    def test_search_with_evaluator_object(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        ev = make_evaluator(actions=a, apply=app, terminal=term, reward=r,
                            key=k)
        s = Searcher(SearcherConfig(algorithm=ALGORITHM_UCT,
                                    max_iterations=50, seed=0))
        rep = s.search((0, 0), evaluator=ev)
        self.assertIn(rep.best_action, ("N", "S", "E", "W"))

    def test_search_algorithm_override(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        s = Searcher(SearcherConfig(algorithm=ALGORITHM_UCT, max_iterations=50))
        rep = s.search((0, 0), actions=a, apply=app, terminal=term, reward=r,
                       heuristic=lambda x: float(abs(3-x[0]) + abs(3-x[1])),
                       key=k, algorithm=ALGORITHM_ASTAR)
        self.assertEqual(rep.algorithm, ALGORITHM_ASTAR)

    def test_missing_evaluator_raises(self):
        s = Searcher()
        with self.assertRaises(InvalidEvaluator):
            s.search(0)  # neither evaluator nor actions/apply

    def test_unknown_algorithm_at_dispatch(self):
        # Construct config bypassing validation by patch — unlikely
        # in practice; the validation in SearcherConfig is the first
        # line of defence.
        with self.assertRaises(InvalidConfig):
            SearcherConfig(algorithm="not_a_thing")


# =============================================================================
# Composition with other primitives (illustrative; via duck-typing)
# =============================================================================


class TestComposition(unittest.TestCase):

    def test_evaluator_with_subset_of_callables(self):
        # A search problem with only actions, apply, and terminal — should
        # work with A* (no heuristic) as uniform-cost search.
        def acts(s): return [s + 1, s + 2]
        def app(s, a): return a
        def term(s): return s == 5
        rep = astar(0, actions=acts, apply=app, terminal=term,
                    key=lambda s: s, max_iterations=100)
        self.assertEqual(rep.optimal_cost, 3.0)  # 0->1->3->5 or 0->2->4->5

    def test_evaluator_with_value_function_only(self):
        # MCTS leaf eval via supplied value, no rewards.
        def acts(s): return ["A", "B"]
        def app(s, a): return s + 1
        def term(s): return s >= 3
        def val(s): return float(s)
        rep = uct(0, actions=acts, apply=app, terminal=term,
                  value=val, key=lambda s: s, max_iterations=200, seed=0)
        self.assertIn(rep.best_action, ("A", "B"))


# =============================================================================
# Edge-case stability
# =============================================================================


class TestEdgeCases(unittest.TestCase):

    def test_root_already_terminal_returns_no_action(self):
        rep = astar(0, actions=lambda s: [],
                    apply=lambda s, a: s,
                    terminal=lambda s: True,
                    key=lambda s: s, max_iterations=10)
        self.assertEqual(rep.optimal_cost, 0.0)

    def test_uct_with_root_terminal(self):
        rep = uct(0, actions=lambda s: [],
                  apply=lambda s, a: s,
                  terminal=lambda s: True,
                  reward=lambda s: 5.0,
                  key=lambda s: s, max_iterations=10, seed=0)
        # No actions available, so best_action is None
        self.assertIsNone(rep.best_action)

    def test_self_loop_actions_dont_starve_siblings(self):
        # An action that doesn't change state should not blow up.
        def acts(s): return ["loop", "advance"]
        def app(s, a):
            if a == "loop":
                return s
            return s + 1
        def term(s): return s >= 3
        def rew(s): return float(s)
        rep = uct(0, actions=acts, apply=app, terminal=term,
                  reward=rew, key=lambda s: s, max_iterations=300, seed=0)
        # "advance" should win
        self.assertEqual(rep.best_action, "advance")

    def test_empty_action_set_at_root(self):
        # Non-terminal root with no actions — search should terminate.
        rep = uct(0, actions=lambda s: [],
                  apply=lambda s, a: s,
                  terminal=lambda s: False,
                  reward=lambda s: 0.0,
                  key=lambda s: s, max_iterations=10, seed=0)
        self.assertIsNone(rep.best_action)


# =============================================================================
# Free-function shortcuts
# =============================================================================


class TestFreeFunctions(unittest.TestCase):

    def test_astar_shortcut(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep = astar((0, 0), actions=a, apply=app, terminal=term, heuristic=h,
                    key=k)
        self.assertEqual(rep.optimal_cost, 6.0)

    def test_uct_shortcut(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = uct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                  key=k, max_iterations=100, seed=0)
        self.assertIsNotNone(rep.best_action)

    def test_puct_shortcut(self):
        a, app, term, r, _h, k = make_grid_problem(size=4)
        rep = puct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                   key=k, max_iterations=100, seed=0)
        self.assertIsNotNone(rep.best_action)

    def test_alphabeta_shortcut(self):
        acts, app, term, rew, key = nim_problem()
        rep = alphabeta((5, 0), actions=acts, apply=app, terminal=term,
                        reward=rew, key=key, depth=6)
        self.assertEqual(rep.best_value, 1.0)

    def test_branch_and_bound_shortcut(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep = branch_and_bound((0, 0), actions=a, apply=app, terminal=term,
                               heuristic=h, key=k, max_iterations=200)
        self.assertEqual(rep.optimal_cost, 6.0)

    def test_beam_search_shortcut(self):
        a, app, term, _r, h, k = make_grid_problem(size=4)
        rep = beam_search((0, 0), actions=a, apply=app, terminal=term,
                          heuristic=h, key=k, width=3, score="cost",
                          max_iterations=30)
        self.assertIsNotNone(rep.best_action)


# =============================================================================
# Larger correctness checks
# =============================================================================


class TestLarger(unittest.TestCase):

    def test_astar_random_grid_with_obstacles(self):
        # 5x5 grid with a wall at column 2 (except a hole at row 2).
        size = 5
        goal = (4, 4)
        obstacles = {(2, y) for y in range(size) if y != 2}

        def acts(_s): return ["N", "S", "E", "W"]
        def app(s, a):
            x, y = s
            if a == "N": y -= 1
            elif a == "S": y += 1
            elif a == "E": x += 1
            elif a == "W": x -= 1
            if not (0 <= x < size and 0 <= y < size):
                return s
            if (x, y) in obstacles:
                return s
            return (x, y)
        def term(s): return s == goal
        def heur(s):
            return float(abs(goal[0]-s[0]) + abs(goal[1]-s[1]))

        rep = astar((0, 0), actions=acts, apply=app, terminal=term,
                    heuristic=heur, key=lambda s: s, max_iterations=10_000)
        # The path must traverse the hole at (2, 2): cost = 4 down/right
        # to (2,2) + 4 down/right to (4,4) = 8 (Manhattan from (0,0) is 8).
        self.assertEqual(rep.optimal_cost, 8.0)

    def test_uct_converges_on_simple_chain(self):
        # Chain: state ∈ {0..10}, action 'inc' = +1.  Terminal at 10
        # with reward 10.  UCT should pick 'inc' over nothing.
        def acts(_s): return ["inc"]
        def app(s, _a): return s + 1
        def term(s): return s >= 10
        def rew(s): return 10.0 if s >= 10 else 0.0
        rep = uct(0, actions=acts, apply=app, terminal=term, reward=rew,
                  key=lambda s: s, max_iterations=200, c_puct=1.4, seed=0)
        self.assertEqual(rep.best_action, "inc")

    def test_alphabeta_nim_misere(self):
        acts, app, term, rew, key = nim_problem(misere=True)
        # Misère nim: 5 stones, take the last loses.  P1 wants to leave
        # 1 stone, so take 4 over two turns.  Position 5 is losing for P1
        # under perfect play? Actually it depends on N.
        rep = alphabeta((5, 0), actions=acts, apply=app, terminal=term,
                        reward=rew, key=key, depth=7)
        self.assertIn(rep.best_value, (-1.0, 1.0))


# =============================================================================
# Determinism stress
# =============================================================================


class TestDeterminism(unittest.TestCase):

    def test_uct_repeatable_certificate(self):
        a, app, term, r, _h, k = make_grid_problem(size=5)
        certs = set()
        for _ in range(5):
            rep = uct((0, 0), actions=a, apply=app, terminal=term, reward=r,
                      key=k, max_iterations=150, c_puct=1.4, seed=123)
            certs.add(rep.certificate)
        self.assertEqual(len(certs), 1)

    def test_astar_repeatable_certificate(self):
        a, app, term, _r, h, k = make_grid_problem(size=6)
        certs = set()
        for _ in range(5):
            rep = astar((0, 0), actions=a, apply=app, terminal=term,
                        heuristic=h, key=k, max_iterations=1000)
            certs.add(rep.certificate)
        self.assertEqual(len(certs), 1)


if __name__ == "__main__":
    unittest.main()
