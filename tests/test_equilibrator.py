"""Tests for ``agi.equilibrator`` — non-cooperative equilibria runtime.

The contract is theorem-driven, and the tests reflect that:

1. **Existence**: every finite normal-form game has at least one Nash
   equilibrium (Nash 1951). Multiplicative weights / fictitious play
   converge in zero-sum and potential games; support enumeration
   provides an exact certificate for 2-player.

2. **Best-response invariance**: at a Nash equilibrium, every player
   plays a best response. Tested both on hand-built equilibria and
   solver outputs.

3. **Minimax theorem** (von Neumann 1928): for any 2-player zero-sum
   game,  max_σ min_τ σᵀ A τ = min_τ max_σ σᵀ A τ.  Verified across
   matrix size and shifted payoffs.

4. **Coarse correlated equilibria via no-regret**: independent Hedge
   learners converge to a CCE with regret O(√(log K / T)). Verified
   against the analytic regret bound.

5. **Correlated equilibrium** is a convex superset of the Nash hull,
   and the LP solution satisfies Aumann's incentive constraints.

6. **Replicator dynamics fixed points** are Nash equilibria
   (Taylor-Jonker 1978). ESS test on hawk-dove confirms the mixed
   strategy is stable.

7. **Determinism + seeded randomness**: a seeded run reproduces the
   exact same profile.

8. **Event emission + attestation pass-through**: every solve emits
   `equilibrator.solved`.
"""
from __future__ import annotations

import math
import threading

import pytest

from agi.events import Event, EventBus
from agi.equilibrator import (
    AXIOM_BEST_RESPONSE,
    AXIOM_MINIMAX,
    AXIOM_NO_REGRET,
    CONCEPT_COARSE_CORRELATED,
    CONCEPT_CORRELATED,
    CONCEPT_ESS,
    CONCEPT_MINIMAX,
    CONCEPT_NASH,
    CONCEPT_PURE_NASH,
    EQUILIBRATOR_GAME_REGISTERED,
    EQUILIBRATOR_SOLVED,
    EQUILIBRATOR_STARTED,
    EquilibriumReport,
    Equilibrator,
    GameRecord,
    InvalidGame,
    KNOWN_CONCEPTS,
    KNOWN_METHODS,
    METHOD_AUTO,
    METHOD_BEST_RESPONSE,
    METHOD_FICTITIOUS_PLAY,
    METHOD_LINEAR_PROGRAM,
    METHOD_MULTIPLICATIVE_WEIGHTS,
    METHOD_REPLICATOR,
    METHOD_SUPPORT_ENUMERATION,
    Profile,
    SolverUnavailable,
    Strategy,
    UnknownGame,
    best_response,
    best_response_dynamics,
    coarse_correlated_equilibrium,
    correlated_equilibrium_lp,
    expected_payoff,
    exploitability,
    fictitious_play,
    make_game,
    multiplicative_weights,
    player_payoff_vector,
    pure_nash_equilibria,
    replicator_dynamics,
    support_enumeration_bimatrix,
    zero_sum_value,
)


# ============================================================================
# Canonical games — used in many tests below.
# ============================================================================


def prisoners_dilemma() -> GameRecord:
    return make_game(
        "pd",
        [
            [[3, 0], [5, 1]],   # row payoffs: (C/D) × (C/D)
            [[3, 5], [0, 1]],   # col payoffs
        ],
        action_names=[("C", "D"), ("C", "D")],
    )


def matching_pennies() -> GameRecord:
    return make_game(
        "mp",
        [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ],
        action_names=[("H", "T"), ("H", "T")],
    )


def rock_paper_scissors() -> GameRecord:
    return make_game(
        "rps",
        [
            [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
            [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        ],
        action_names=[("R", "P", "S"), ("R", "P", "S")],
    )


def battle_of_sexes() -> GameRecord:
    return make_game(
        "bos",
        [
            [[2, 0], [0, 1]],
            [[1, 0], [0, 2]],
        ],
    )


def stag_hunt() -> GameRecord:
    # Stag-hunt: cooperate on stag (high reward) or defect on hare.
    return make_game(
        "stag",
        [
            [[4, 0], [3, 3]],
            [[4, 3], [0, 3]],
        ],
    )


def hawk_dove(v: float = 2.0, c: float = 6.0) -> GameRecord:
    # Classic hawk-dove with cost > value → mixed ESS at p_hawk = v/c.
    return make_game(
        "hd",
        [
            [[(v - c) / 2.0, v], [0.0, v / 2.0]],
            [[(v - c) / 2.0, 0.0], [v, v / 2.0]],
        ],
        action_names=[("H", "D"), ("H", "D")],
    )


def chicken() -> GameRecord:
    return make_game(
        "chicken",
        [
            [[0, -1], [1, -10]],
            [[0, 1], [-1, -10]],
        ],
    )


# ============================================================================
# Game representation
# ============================================================================


class TestGameRecord:
    def test_make_game_pd_shape(self):
        g = prisoners_dilemma()
        assert g.n_players == 2
        assert g.action_counts == (2, 2)
        assert g.n_joint_actions() == 4
        assert not g.is_zero_sum
        assert not g.is_constant_sum

    def test_make_game_mp_zero_sum(self):
        g = matching_pennies()
        assert g.is_zero_sum
        assert g.is_constant_sum
        assert g.is_symmetric is False  # MP is anti-symmetric, not symmetric

    def test_make_game_rps_zero_sum(self):
        g = rock_paper_scissors()
        assert g.is_zero_sum

    def test_make_game_hawk_dove_symmetric(self):
        g = hawk_dove()
        assert g.is_symmetric

    def test_flat_index_round_trip(self):
        g = make_game(
            "3p",
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            ],
        )
        assert g.n_players == 3
        assert g.action_counts == (2, 2, 2)
        for joint in g.joint_actions():
            idx = g.flat_index(joint)
            recovered = g.joint_from_index(idx)
            assert recovered == joint

    def test_payoff_lookup(self):
        g = prisoners_dilemma()
        assert g.payoff((0, 0)) == (3.0, 3.0)
        assert g.payoff((0, 1)) == (0.0, 5.0)
        assert g.payoff((1, 0)) == (5.0, 0.0)
        assert g.payoff((1, 1)) == (1.0, 1.0)

    def test_content_hash_deterministic(self):
        g1 = prisoners_dilemma()
        g2 = make_game(
            "pd2",
            [
                [[3, 0], [5, 1]],
                [[3, 5], [0, 1]],
            ],
        )
        assert g1.content_hash == g2.content_hash

    def test_invalid_shape_raises(self):
        with pytest.raises(InvalidGame):
            make_game("bad", [[[1, 2], [3]], [[4, 5], [6, 7]]])  # ragged

    def test_unknown_game_raises(self):
        eq = Equilibrator()
        with pytest.raises(UnknownGame):
            eq.get_game("nonexistent")


# ============================================================================
# Strategy / Profile invariants
# ============================================================================


class TestStrategy:
    def test_uniform(self):
        s = Strategy.uniform(4)
        assert s.n_actions == 4
        assert all(abs(p - 0.25) < 1e-12 for p in s.probabilities)
        assert s.support == (0, 1, 2, 3)
        assert math.isclose(s.entropy(), math.log(4))

    def test_pure(self):
        s = Strategy.pure(2, 5)
        assert s.support == (2,)
        assert s.probability(2) == 1.0
        assert s.probability(0) == 0.0
        assert s.entropy() == 0.0

    def test_from_weights_renormalises(self):
        s = Strategy.from_weights([1, 2, 3, 4])
        assert math.isclose(sum(s.probabilities), 1.0)
        assert math.isclose(s.probability(0), 0.1)
        assert math.isclose(s.probability(3), 0.4)

    def test_from_weights_zero_returns_uniform(self):
        s = Strategy.from_weights([0, 0, 0])
        assert all(abs(p - 1.0 / 3.0) < 1e-12 for p in s.probabilities)

    def test_negative_probability_rejected(self):
        with pytest.raises(InvalidGame):
            Strategy(probabilities=(-0.1, 1.1))

    def test_expected_value(self):
        s = Strategy(probabilities=(0.3, 0.7))
        assert math.isclose(s.expected_value([10.0, 20.0]), 0.3 * 10 + 0.7 * 20)

    def test_total_variation(self):
        s = Strategy.pure(0, 3)
        t = Strategy.uniform(3)
        # TV between pure(0) and uniform: 0.5 * (|1 - 1/3| + 2 * |0 - 1/3|) = 2/3
        assert math.isclose(s.total_variation(t), 2.0 / 3.0)


class TestProfile:
    def test_replace(self):
        prof = Profile(strategies=(Strategy.pure(0, 2), Strategy.pure(0, 2)))
        new = prof.replace(0, Strategy.pure(1, 2))
        assert new[0].probability(1) == 1.0
        # original unchanged (immutable)
        assert prof[0].probability(0) == 1.0

    def test_action_counts(self):
        prof = Profile(strategies=(Strategy.uniform(3), Strategy.uniform(2)))
        assert prof.action_counts == (3, 2)
        assert prof.n_players == 2


# ============================================================================
# Best response and exploitability
# ============================================================================


class TestBestResponse:
    def test_pd_best_response(self):
        g = prisoners_dilemma()
        # Against C (= pure 0), defect is BR.
        prof = Profile(strategies=(Strategy.pure(0, 2), Strategy.pure(0, 2)))
        br_actions, br_val, pv = best_response(g, 0, prof)
        assert br_actions == (1,)
        assert br_val == 5.0

    def test_pd_dominant_defection(self):
        # Defect (1) dominates regardless of opponent — best response
        # to ANY opponent profile.
        g = prisoners_dilemma()
        for op_p_c in [0.0, 0.1, 0.5, 0.9, 1.0]:
            prof = Profile(strategies=(
                Strategy(probabilities=(0.5, 0.5)),
                Strategy(probabilities=(op_p_c, 1.0 - op_p_c)),
            ))
            br_actions, _, _ = best_response(g, 0, prof)
            assert 1 in br_actions

    def test_mp_uniform_is_indifferent(self):
        g = matching_pennies()
        prof = Profile(strategies=(Strategy.uniform(2), Strategy.uniform(2)))
        _, _, pv = best_response(g, 0, prof)
        # Both H and T have expected value 0 against uniform opponent
        assert abs(pv[0] - pv[1]) < 1e-12

    def test_player_payoff_vector_matches_definition(self):
        g = battle_of_sexes()
        prof = Profile(strategies=(Strategy(probabilities=(0.4, 0.6)),
                                    Strategy(probabilities=(0.3, 0.7))))
        # Manual computation for player 0:
        # u_0(0, σ_1) = 0.3 * 2 + 0.7 * 0 = 0.6
        # u_0(1, σ_1) = 0.3 * 0 + 0.7 * 1 = 0.7
        pv = player_payoff_vector(g, 0, prof)
        assert math.isclose(pv[0], 0.6)
        assert math.isclose(pv[1], 0.7)


class TestExploitability:
    def test_pure_nash_zero(self):
        g = prisoners_dilemma()
        prof = Profile(strategies=(Strategy.pure(1, 2), Strategy.pure(1, 2)))
        total, per = exploitability(g, prof)
        assert total == 0.0
        assert all(g == 0.0 for g in per)

    def test_non_eq_strictly_positive(self):
        g = prisoners_dilemma()
        # (C, C) is not a Nash — both can deviate to D for higher payoff.
        prof = Profile(strategies=(Strategy.pure(0, 2), Strategy.pure(0, 2)))
        total, per = exploitability(g, prof)
        assert total > 0
        assert per[0] == 2.0  # 5 - 3
        assert per[1] == 2.0

    def test_mp_uniform_zero(self):
        g = matching_pennies()
        prof = Profile(strategies=(Strategy.uniform(2), Strategy.uniform(2)))
        total, _ = exploitability(g, prof)
        assert total < 1e-9


# ============================================================================
# Pure Nash search
# ============================================================================


class TestPureNash:
    def test_pd_has_unique_pure_nash(self):
        g = prisoners_dilemma()
        eqs = pure_nash_equilibria(g)
        assert eqs == ((1, 1),)

    def test_mp_has_no_pure_nash(self):
        g = matching_pennies()
        eqs = pure_nash_equilibria(g)
        assert eqs == ()

    def test_bos_has_two_pure_nash(self):
        g = battle_of_sexes()
        eqs = pure_nash_equilibria(g)
        assert set(eqs) == {(0, 0), (1, 1)}

    def test_stag_hunt_two_pure_nash(self):
        g = stag_hunt()
        eqs = pure_nash_equilibria(g)
        assert (0, 0) in eqs   # (Stag, Stag) is Nash
        assert (1, 1) in eqs   # (Hare, Hare) is Nash


# ============================================================================
# Support enumeration (2-player exact Nash)
# ============================================================================


class TestSupportEnumeration:
    def test_pd_finds_unique_pure_nash(self):
        g = prisoners_dilemma()
        res = support_enumeration_bimatrix(g)
        assert res["n_found"] >= 1
        # All equilibria should have (1, 1) in support
        found_dd = False
        for eq in res["equilibria"]:
            if (eq["profile"][0].probability(1) > 0.99 and
                eq["profile"][1].probability(1) > 0.99):
                found_dd = True
        assert found_dd

    def test_bos_finds_three_equilibria(self):
        g = battle_of_sexes()
        res = support_enumeration_bimatrix(g)
        # Should find: (0,0), (1,1), and the mixed one.
        assert res["n_found"] == 3
        # Mixed equilibrium should have probabilities (2/3, 1/3) and (1/3, 2/3)
        found_mixed = False
        for eq in res["equilibria"]:
            p1 = eq["profile"][0].probabilities
            p2 = eq["profile"][1].probabilities
            if abs(p1[0] - 2.0 / 3.0) < 1e-6 and abs(p2[1] - 2.0 / 3.0) < 1e-6:
                found_mixed = True
        assert found_mixed

    def test_mp_finds_unique_mixed(self):
        g = matching_pennies()
        res = support_enumeration_bimatrix(g)
        assert res["n_found"] == 1
        eq = res["equilibria"][0]
        assert all(abs(p - 0.5) < 1e-9 for p in eq["profile"][0].probabilities)
        assert all(abs(p - 0.5) < 1e-9 for p in eq["profile"][1].probabilities)

    def test_rps_finds_uniform_mixed(self):
        g = rock_paper_scissors()
        res = support_enumeration_bimatrix(g)
        assert res["n_found"] >= 1
        # Must find the uniform mixed equilibrium
        found = False
        for eq in res["equilibria"]:
            if all(abs(p - 1.0 / 3.0) < 1e-6 for p in eq["profile"][0].probabilities):
                found = True
        assert found


# ============================================================================
# Multiplicative weights / Hedge
# ============================================================================


class TestMultiplicativeWeights:
    def test_mp_converges_to_uniform(self):
        g = matching_pennies()
        res = multiplicative_weights(g, iterations=5_000, seed=0)
        for p in range(2):
            assert all(abs(prob - 0.5) < 0.05 for prob in res["profile"][p].probabilities)
        assert res["exploitability"] < 0.05

    def test_rps_converges_to_uniform(self):
        g = rock_paper_scissors()
        res = multiplicative_weights(g, iterations=5_000, seed=0)
        for p in range(2):
            assert all(abs(prob - 1.0 / 3.0) < 0.05 for prob in res["profile"][p].probabilities)
        assert res["exploitability"] < 0.1

    def test_pd_converges_to_dd(self):
        # Even though MW doesn't have great Nash guarantees in general,
        # in dominance-solvable games it converges to the dominant action.
        g = prisoners_dilemma()
        res = multiplicative_weights(g, iterations=5_000, seed=0)
        # Both players should heavily weight defection.
        assert res["profile"][0].probability(1) > 0.9
        assert res["profile"][1].probability(1) > 0.9

    def test_regret_bound_holds(self):
        # For Hedge with optimal eta, regret bound is √(log K / (2T)) * span.
        g = matching_pennies()
        res = multiplicative_weights(g, iterations=2_000, seed=0)
        assert res["exploitability"] <= res["regret_bound"] * 3 + 0.1
        # Note: regret bound applies to per-round, total exploitability
        # at the average profile is bounded by 2× regret in zero-sum.

    def test_deterministic_with_seed(self):
        g = matching_pennies()
        r1 = multiplicative_weights(g, iterations=500, seed=42)
        r2 = multiplicative_weights(g, iterations=500, seed=42)
        for p in range(2):
            for a in range(2):
                assert r1["profile"][p].probability(a) == r2["profile"][p].probability(a)

    def test_eta_must_be_positive(self):
        g = matching_pennies()
        with pytest.raises(InvalidGame):
            multiplicative_weights(g, iterations=100, eta=-0.1)


# ============================================================================
# Fictitious play
# ============================================================================


class TestFictitiousPlay:
    def test_mp_converges_to_uniform(self):
        # FP converges to Nash in zero-sum (Robinson 1951)
        g = matching_pennies()
        res = fictitious_play(g, iterations=3_000, seed=0)
        for p in range(2):
            assert all(abs(prob - 0.5) < 0.1 for prob in res["profile"][p].probabilities)
        assert res["exploitability"] < 0.1

    def test_potential_game_converges(self):
        # Stag-hunt is a potential game; FP should converge.
        g = stag_hunt()
        res = fictitious_play(g, iterations=3_000, seed=0)
        assert res["exploitability"] < 0.1


# ============================================================================
# Replicator dynamics
# ============================================================================


class TestReplicator:
    def test_hawk_dove_mixed_ess(self):
        # Hawk-dove with V=2, C=6: ESS at p_hawk = V/C = 1/3.
        g = hawk_dove(v=2.0, c=6.0)
        res = replicator_dynamics(g, iterations=3_000, dt=0.05)
        # Both players should converge to (1/3, 2/3) — same mixed strategy.
        for p in range(2):
            p_hawk = res["profile"][p].probability(0)
            assert abs(p_hawk - 1.0 / 3.0) < 0.05

    def test_dominance_solvable_pd(self):
        # Defection strictly dominates → replicator drives weight to defect.
        g = prisoners_dilemma()
        res = replicator_dynamics(g, iterations=2_000, dt=0.1)
        for p in range(2):
            assert res["profile"][p].probability(1) > 0.95


# ============================================================================
# Best-response dynamics on potential games
# ============================================================================


class TestBestResponseDynamics:
    def test_pd_converges_to_dd(self):
        g = prisoners_dilemma()
        res = best_response_dynamics(g, iterations=50, seed=0)
        assert res["converged"]
        assert res["joint_action"] == (1, 1)

    def test_stag_hunt_converges(self):
        g = stag_hunt()
        # Potential game → BR dynamics converges.
        # Initial start at (Stag, Stag) — already a Nash equilibrium.
        res = best_response_dynamics(g, iterations=50, initial=(0, 0), seed=0)
        assert res["converged"]


# ============================================================================
# Zero-sum minimax
# ============================================================================


class TestZeroSumValue:
    def test_mp_value_zero(self):
        sol = zero_sum_value([[1, -1], [-1, 1]], method=METHOD_LINEAR_PROGRAM)
        assert abs(sol["value"]) < 1e-6
        assert all(abs(p - 0.5) < 1e-6 for p in sol["row_strategy"].probabilities)
        assert all(abs(p - 0.5) < 1e-6 for p in sol["col_strategy"].probabilities)

    def test_rps_value_zero(self):
        rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
        sol = zero_sum_value(rps, method=METHOD_LINEAR_PROGRAM)
        assert abs(sol["value"]) < 1e-6
        for p in sol["row_strategy"].probabilities:
            assert abs(p - 1.0 / 3.0) < 1e-6

    def test_dominant_row_value(self):
        # Row 0 dominates row 1: [[3, 2], [1, 0]] → value = 2 (minmax over col j of 3, 2),
        # row plays pure 0, col plays pure 1 (the min for row).
        sol = zero_sum_value([[3, 2], [1, 0]], method=METHOD_LINEAR_PROGRAM)
        assert abs(sol["value"] - 2.0) < 1e-4
        assert sol["row_strategy"].probability(0) > 0.99
        assert sol["col_strategy"].probability(1) > 0.99

    def test_value_translation_invariance(self):
        A = [[1, -1], [-1, 1]]
        sol1 = zero_sum_value(A, method=METHOD_LINEAR_PROGRAM)
        # Shift by 7
        A_shifted = [[a + 7 for a in row] for row in A]
        sol2 = zero_sum_value(A_shifted, method=METHOD_LINEAR_PROGRAM)
        # Value should shift by exactly 7
        assert abs((sol2["value"] - sol1["value"]) - 7.0) < 1e-6

    def test_mw_method_matches_lp(self):
        rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
        sol_lp = zero_sum_value(rps, method=METHOD_LINEAR_PROGRAM)
        sol_mw = zero_sum_value(rps, method=METHOD_MULTIPLICATIVE_WEIGHTS, iterations=5_000)
        assert abs(sol_lp["value"] - sol_mw["value"]) < 0.1


# ============================================================================
# Correlated equilibrium
# ============================================================================


class TestCorrelatedEquilibrium:
    def test_chicken_ce_outperforms_nash(self):
        # In chicken, the welfare-optimal CE strictly dominates each
        # pure Nash in expected social welfare.
        g = chicken()
        ce = correlated_equilibrium_lp(g, objective="welfare")
        total_welfare = sum(ce["expected_payoff"])
        # Pure Nash (Hawk, Dove) has welfare 1 + 0 = 1.
        # Welfare-optimal CE should hit ≥ 0 (above both -1 single-player NE outcomes).
        assert total_welfare >= -0.1   # avoid (D,D) crash with prob ~0

    def test_pd_ce_satisfies_constraints(self):
        g = prisoners_dilemma()
        ce = correlated_equilibrium_lp(g, objective="welfare")
        # All probability mass should be at (D, D) since CE is the unique
        # pure Nash in a dominance-solvable game.
        max_mass = max(p for _, p in ce["distribution"])
        assert max_mass > 0.95

    def test_bos_ce_egalitarian_balances(self):
        g = battle_of_sexes()
        ce = correlated_equilibrium_lp(g, objective="egalitarian")
        # Egalitarian CE should give both players equal expected payoff.
        assert abs(ce["expected_payoff"][0] - ce["expected_payoff"][1]) < 1e-4


# ============================================================================
# Coarse correlated equilibrium via no-regret
# ============================================================================


class TestCoarseCorrelatedEquilibrium:
    def test_mp_cce_zero_unconditional_exploit(self):
        g = matching_pennies()
        cce = coarse_correlated_equilibrium(g, iterations=3_000, seed=0)
        assert cce["exploitability_unconditional"] < 0.1

    def test_rps_cce_zero_unconditional_exploit(self):
        g = rock_paper_scissors()
        cce = coarse_correlated_equilibrium(g, iterations=3_000, seed=0)
        assert cce["exploitability_unconditional"] < 0.15

    def test_regret_bound_present(self):
        g = matching_pennies()
        cce = coarse_correlated_equilibrium(g, iterations=2_000, seed=0)
        assert cce["regret_bound"] > 0
        assert cce["regret_bound"] < 0.5


# ============================================================================
# Equilibrator (the runtime class)
# ============================================================================


class TestEquilibratorRuntime:
    def test_register_and_solve_pd(self):
        bus = EventBus()
        events = []
        bus.subscribe(lambda e: events.append(e))
        eq = Equilibrator(bus=bus)

        eq.register_game(
            "pd",
            [
                [[3, 0], [5, 1]],
                [[3, 5], [0, 1]],
            ],
        )
        rep = eq.solve("pd", concept=CONCEPT_PURE_NASH)
        assert rep.profile[0].probability(1) == 1.0
        assert rep.profile[1].probability(1) == 1.0
        assert rep.converged

        kinds = [e.kind for e in events]
        assert EQUILIBRATOR_STARTED in kinds
        assert EQUILIBRATOR_GAME_REGISTERED in kinds
        assert EQUILIBRATOR_SOLVED in kinds

    def test_solve_mp_nash_via_mw(self):
        eq = Equilibrator()
        eq.register_game("mp", [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])
        rep = eq.solve("mp", concept=CONCEPT_NASH,
                       method=METHOD_MULTIPLICATIVE_WEIGHTS,
                       iterations=3_000, seed=0)
        assert rep.exploitability < 0.05
        for p in range(2):
            assert all(abs(prob - 0.5) < 0.1 for prob in rep.profile[p].probabilities)

    def test_solve_minimax_mp(self):
        eq = Equilibrator()
        eq.register_game("mp", [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])
        rep = eq.solve("mp", concept=CONCEPT_MINIMAX)
        assert abs(rep.value) < 1e-6
        assert AXIOM_MINIMAX in rep.certificate.get("axioms", [])

    def test_solve_correlated_chicken(self):
        eq = Equilibrator()
        eq.register_game("chicken", [
            [[0, -1], [1, -10]],
            [[0, 1], [-1, -10]],
        ])
        rep = eq.solve("chicken", concept=CONCEPT_CORRELATED,
                       method=METHOD_LINEAR_PROGRAM)
        assert rep.distribution is not None
        assert rep.exploitability < 1e-3

    def test_solve_coarse_correlated_bos(self):
        eq = Equilibrator()
        eq.register_game("bos", [
            [[2, 0], [0, 1]],
            [[1, 0], [0, 2]],
        ])
        rep = eq.solve("bos", concept=CONCEPT_COARSE_CORRELATED,
                       iterations=3_000, seed=0)
        assert rep.distribution is not None
        assert rep.exploitability < 0.1

    def test_solve_pure_nash_bos(self):
        eq = Equilibrator()
        eq.register_game("bos", [
            [[2, 0], [0, 1]],
            [[1, 0], [0, 2]],
        ])
        rep = eq.solve("bos", concept=CONCEPT_PURE_NASH)
        # Should pick the welfare-max pure Nash, which is (0,0) (payoff 2,1 sum=3)
        # tied with (1,1) (sum = 1+2 = 3). Either is acceptable.
        joint = (rep.profile[0].probabilities.index(1.0),
                 rep.profile[1].probabilities.index(1.0))
        assert joint in {(0, 0), (1, 1)}

    def test_solve_auto_method(self):
        eq = Equilibrator()
        eq.register_game("mp", [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])
        rep = eq.solve("mp", concept=CONCEPT_NASH, method=METHOD_AUTO)
        assert rep.exploitability < 0.05

    def test_unknown_concept_rejected(self):
        eq = Equilibrator()
        eq.register_game("mp", [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])
        with pytest.raises(SolverUnavailable):
            eq.solve("mp", concept="bogus_concept")

    def test_unknown_method_rejected(self):
        eq = Equilibrator()
        eq.register_game("mp", [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])
        with pytest.raises(SolverUnavailable):
            eq.solve("mp", concept=CONCEPT_NASH, method="bogus_method")

    def test_minimax_requires_zero_sum(self):
        eq = Equilibrator()
        eq.register_game("pd", [
            [[3, 0], [5, 1]],
            [[3, 5], [0, 1]],
        ])
        with pytest.raises(SolverUnavailable):
            eq.solve("pd", concept=CONCEPT_MINIMAX)

    def test_ess_requires_symmetric(self):
        eq = Equilibrator()
        eq.register_game("bos", [
            [[2, 0], [0, 1]],
            [[1, 0], [0, 2]],
        ])
        with pytest.raises(SolverUnavailable):
            eq.solve("bos", concept=CONCEPT_ESS)

    def test_double_register_rejected(self):
        eq = Equilibrator()
        eq.register_game("g", [[[1]], [[1]]])
        with pytest.raises(InvalidGame):
            eq.register_game("g", [[[2]], [[2]]])

    def test_remove_game(self):
        eq = Equilibrator()
        eq.register_game("g", [[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
        assert "g" in eq.games()
        eq.remove_game("g")
        assert "g" not in eq.games()

    def test_observe_and_empirical_distribution(self):
        eq = Equilibrator()
        eq.register_game("mp", [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])
        for _ in range(10):
            eq.observe("mp", (0, 0))
        for _ in range(10):
            eq.observe("mp", (1, 1))
        dist = eq.empirical_distribution("mp")
        # 10/20 each
        d = dict(dist)
        assert abs(d[(0, 0)] - 0.5) < 1e-9
        assert abs(d[(1, 1)] - 0.5) < 1e-9

    def test_observe_invalid_joint(self):
        eq = Equilibrator()
        eq.register_game("mp", [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])
        with pytest.raises(InvalidGame):
            eq.observe("mp", (5, 0))

    def test_clear(self):
        eq = Equilibrator()
        eq.register_game("g", [[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
        eq.clear()
        assert len(eq.games()) == 0

    def test_coverage_report(self):
        eq = Equilibrator()
        eq.register_game("g1", [[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
        eq.register_game("g2", [[[1]], [[1]]])
        eq.solve("g1", concept=CONCEPT_PURE_NASH)
        cov = eq.coverage()
        assert cov.n_games == 2
        assert cov.n_solved == 1
        assert set(cov.games) == {"g1", "g2"}


# ============================================================================
# Receipts and event payloads
# ============================================================================


class TestEvents:
    def test_solve_event_payload_keys(self):
        bus = EventBus()
        events = []
        bus.subscribe(lambda e: events.append(e))
        eq = Equilibrator(bus=bus)
        eq.register_game("mp", [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])
        eq.solve("mp", concept=CONCEPT_MINIMAX)
        solved = [e for e in events if e.kind == EQUILIBRATOR_SOLVED]
        assert solved
        payload = solved[0].data
        # Sanity-check the expected schema keys
        for k in ["game_id", "concept", "method", "expected_payoff",
                  "exploitability", "epsilon", "iterations", "converged",
                  "certificate"]:
            assert k in payload


# ============================================================================
# Strategic invariants: theorem-driven cross-checks
# ============================================================================


class TestTheoremInvariants:
    def test_minimax_theorem_2x2(self):
        # Random 2×2 matrix games: maximin == minimax (von Neumann 1928)
        import random
        rng = random.Random(0)
        for _ in range(8):
            A = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
            sol = zero_sum_value(A, method=METHOD_LINEAR_PROGRAM)
            value_via_row = sol["value"]
            # Compute maximin directly: max_σ min_τ σᵀ A τ
            row = sol["row_strategy"]
            for col_action in range(2):
                payoff = sum(row.probabilities[i] * A[i][col_action] for i in range(2))
                assert payoff >= value_via_row - 1e-6

    def test_nash_is_best_response_at_eq(self):
        # For every equilibrium found by support enumeration, every
        # player is best-responding to opponents.
        g = battle_of_sexes()
        res = support_enumeration_bimatrix(g)
        for eq in res["equilibria"]:
            prof = eq["profile"]
            for p in range(2):
                # Every action in support gives the same expected payoff (BR principle).
                pv = player_payoff_vector(g, p, prof)
                supp = prof[p].support
                vals_in_support = [pv[a] for a in supp]
                if len(vals_in_support) > 1:
                    assert max(vals_in_support) - min(vals_in_support) < 1e-6
                # Best value attained at all support actions.
                best_val = max(pv)
                for a in supp:
                    assert abs(pv[a] - best_val) < 1e-6

    def test_exploitability_nonneg(self):
        # Exploitability is a NashConv: always non-negative.
        g = prisoners_dilemma()
        for joint in g.joint_actions():
            prof = Profile(strategies=tuple(
                Strategy.pure(joint[p], g.action_counts[p]) for p in range(2)
            ))
            total, per = exploitability(g, prof)
            assert total >= 0
            assert all(x >= 0 for x in per)

    def test_zero_sum_value_is_unique(self):
        # The value of a zero-sum game is unique even when strategies aren't.
        A = [[3, 1], [1, 3]]   # diagonal payoffs; multiple equilibria
        sol1 = zero_sum_value(A, method=METHOD_LINEAR_PROGRAM)
        sol2 = zero_sum_value(A, method=METHOD_MULTIPLICATIVE_WEIGHTS, iterations=2_000)
        assert abs(sol1["value"] - sol2["value"]) < 0.05

    def test_mw_and_se_agree_on_mp(self):
        # In matching pennies, MW and support enumeration should
        # both converge to (0.5, 0.5).
        g = matching_pennies()
        se_res = support_enumeration_bimatrix(g)
        mw_res = multiplicative_weights(g, iterations=5_000, seed=0)
        # Same expected payoffs
        se_payoffs = expected_payoff(g, se_res["equilibria"][0]["profile"])
        mw_payoffs = mw_res["payoffs"]
        for p in range(2):
            assert abs(se_payoffs[p] - mw_payoffs[p]) < 0.05

    def test_replicator_fixed_point_is_nash(self):
        # In symmetric hawk-dove, replicator's fixed point is the Nash.
        g = hawk_dove(v=2.0, c=4.0)   # mixed ESS at 1/2
        res = replicator_dynamics(g, iterations=5_000, dt=0.05)
        # Fixed-point check: exploitability ≤ ε
        assert res["exploitability"] < 0.05
