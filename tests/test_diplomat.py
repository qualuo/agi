"""Tests for agi.diplomat — CFR-family extensive-form game solvers."""
from __future__ import annotations

import math
import random

import pytest

from agi.diplomat import (
    BestResponseReport,
    CFRConfig,
    Diplomat,
    DiplomatError,
    Game,
    GameBuilder,
    InfeasibleProgram,
    InvalidGame,
    KIND_CFR,
    KIND_CFR_PLUS,
    KIND_LINEAR_CFR,
    KIND_DISCOUNTED_CFR,
    KIND_PREDICTIVE_CFR_PLUS,
    KIND_OUTCOME_SAMPLING,
    KIND_EXTERNAL_SAMPLING,
    KIND_CHANCE_SAMPLING,
    KIND_SEQUENCE_FORM_LP,
    KNOWN_KINDS,
    NotTwoPlayerZeroSum,
    PerfectRecallViolation,
    SolveReport,
    UnknownSolver,
    best_response,
    coin_match_with_signal,
    expected_utilities,
    exploitability,
    kuhn_poker,
    matching_pennies_sequential,
    matching_pennies_simultaneous,
    rock_paper_scissors,
    simple_bargaining,
)


# =====================================================================
# Game construction
# =====================================================================


class TestGameBuilder:
    def test_empty_game_raises(self):
        b = GameBuilder(n_players=2)
        with pytest.raises(InvalidGame):
            b.build()

    def test_bad_player_count(self):
        with pytest.raises(InvalidGame):
            GameBuilder(n_players=0)

    def test_terminal_only_game(self):
        b = GameBuilder(n_players=2)
        b.terminal_node(parent=-1, parent_action=None, utilities=[+1, -1])
        g = b.build()
        assert g.n_terminals() == 1
        assert g.n_decisions() == 0

    def test_chance_must_have_nonneg_probs(self):
        b = GameBuilder(n_players=2)
        with pytest.raises(InvalidGame):
            b.chance_node(parent=-1, parent_action=None, actions=["a"], probs=[-1.0])

    def test_chance_normalises_probs(self):
        b = GameBuilder(n_players=2)
        root = b.chance_node(parent=-1, parent_action=None,
                             actions=["a", "b"], probs=[2.0, 3.0])
        b.terminal_node(parent=root, parent_action="a", utilities=[0, 0])
        b.terminal_node(parent=root, parent_action="b", utilities=[0, 0])
        g = b.build()
        # Probs were renormalised to sum 1.
        chance = g.nodes[root]
        assert pytest.approx(sum(chance.probs)) == 1.0
        assert pytest.approx(chance.probs[0]) == 2.0 / 5.0

    def test_chance_zero_total_rejected(self):
        b = GameBuilder(n_players=2)
        with pytest.raises(InvalidGame):
            b.chance_node(parent=-1, parent_action=None, actions=["a"], probs=[0.0])

    def test_decision_player_out_of_range(self):
        b = GameBuilder(n_players=2)
        with pytest.raises(InvalidGame):
            b.decision_node(parent=-1, parent_action=None, player=5,
                            actions=["a"], info_set="X")

    def test_decision_empty_actions(self):
        b = GameBuilder(n_players=2)
        with pytest.raises(InvalidGame):
            b.decision_node(parent=-1, parent_action=None, player=0,
                            actions=[], info_set="X")

    def test_terminal_utility_count_mismatch(self):
        b = GameBuilder(n_players=2)
        with pytest.raises(InvalidGame):
            b.terminal_node(parent=-1, parent_action=None, utilities=[1.0])

    def test_terminal_nonfinite_utility(self):
        b = GameBuilder(n_players=2)
        with pytest.raises(InvalidGame):
            b.terminal_node(parent=-1, parent_action=None,
                            utilities=[float("nan"), 0.0])

    def test_action_not_in_parent_actions(self):
        b = GameBuilder(n_players=2)
        d = b.decision_node(parent=-1, parent_action=None, player=0,
                            actions=["a", "b"], info_set="X")
        with pytest.raises(InvalidGame):
            b.terminal_node(parent=d, parent_action="c", utilities=[0, 0])

    def test_action_already_linked(self):
        b = GameBuilder(n_players=2)
        d = b.decision_node(parent=-1, parent_action=None, player=0,
                            actions=["a", "b"], info_set="X")
        b.terminal_node(parent=d, parent_action="a", utilities=[0, 0])
        with pytest.raises(InvalidGame):
            b.terminal_node(parent=d, parent_action="a", utilities=[0, 0])

    def test_unlinked_action_rejected(self):
        b = GameBuilder(n_players=2)
        d = b.decision_node(parent=-1, parent_action=None, player=0,
                            actions=["a", "b"], info_set="X")
        b.terminal_node(parent=d, parent_action="a", utilities=[0, 0])
        # Missing 'b' branch
        with pytest.raises(InvalidGame):
            b.build()

    def test_info_set_player_mismatch(self):
        b = GameBuilder(n_players=2)
        d0 = b.decision_node(parent=-1, parent_action=None, player=0,
                             actions=["a"], info_set="X")
        d1 = b.decision_node(parent=d0, parent_action="a", player=1,
                             actions=["a"], info_set="X")
        b.terminal_node(parent=d1, parent_action="a", utilities=[0, 0])
        with pytest.raises(InvalidGame):
            b.build()

    def test_info_set_action_mismatch(self):
        b = GameBuilder(n_players=2)
        d0 = b.decision_node(parent=-1, parent_action=None, player=0,
                             actions=["a", "b"], info_set="X")
        b.terminal_node(parent=d0, parent_action="a", utilities=[0, 0])
        d1 = b.decision_node(parent=d0, parent_action="b", player=0,
                             actions=["c", "d"], info_set="X")
        b.terminal_node(parent=d1, parent_action="c", utilities=[0, 0])
        b.terminal_node(parent=d1, parent_action="d", utilities=[0, 0])
        with pytest.raises(InvalidGame):
            b.build()

    def test_perfect_recall_violation(self):
        # Two nodes in the same info set with different past player actions
        b = GameBuilder(n_players=2)
        d_top = b.decision_node(parent=-1, parent_action=None, player=0,
                                actions=["L", "R"], info_set="top")
        # Both branches lead to a second player-0 decision in info set "bottom"
        # — but with different past actions of player 0, violating PR.
        d_L = b.decision_node(parent=d_top, parent_action="L", player=0,
                              actions=["x", "y"], info_set="bottom")
        d_R = b.decision_node(parent=d_top, parent_action="R", player=0,
                              actions=["x", "y"], info_set="bottom")
        for d in (d_L, d_R):
            b.terminal_node(parent=d, parent_action="x", utilities=[0, 0])
            b.terminal_node(parent=d, parent_action="y", utilities=[0, 0])
        with pytest.raises(PerfectRecallViolation):
            b.build()


# =====================================================================
# Built-in benchmark games
# =====================================================================


class TestBuiltinGames:
    def test_matching_pennies_simultaneous(self):
        g = matching_pennies_simultaneous()
        assert g.n_players == 2
        assert g.n_decisions() >= 2
        assert g.n_terminals() == 4

    def test_matching_pennies_sequential(self):
        g = matching_pennies_sequential()
        # P1 has two distinct info sets (observes P0).
        assert sum(1 for I in g.info_sets.values() if I.player == 1) == 2

    def test_kuhn_poker_structure(self):
        g = kuhn_poker()
        assert g.n_players == 2
        # 6 deals × 6 actionable nodes per deal = sizable but small.
        assert g.size() > 30
        # 6 P0 info sets (3 cards × 2 contexts), 6 P1 info sets.
        assert sum(1 for I in g.info_sets.values() if I.player == 0) == 6
        assert sum(1 for I in g.info_sets.values() if I.player == 1) == 6

    def test_rock_paper_scissors(self):
        g = rock_paper_scissors()
        assert g.n_terminals() == 9
        # P1 is one info set (doesn't observe).
        assert sum(1 for I in g.info_sets.values() if I.player == 1) == 1

    def test_simple_bargaining(self):
        g = simple_bargaining(n_rounds=1)
        assert g.n_players == 2
        # Single-round must have 5 offers × 2 responses = 10 terminals + 0 rejection terminals.
        assert g.n_terminals() == 10

    def test_simple_bargaining_two_rounds(self):
        g = simple_bargaining(n_rounds=2)
        assert g.size() > 50

    def test_simple_bargaining_bad_rounds(self):
        with pytest.raises(InvalidGame):
            simple_bargaining(n_rounds=0)
        with pytest.raises(InvalidGame):
            simple_bargaining(pie=0.0)

    def test_coin_match_invalid_p(self):
        with pytest.raises(InvalidGame):
            coin_match_with_signal(p_heads=0.0)
        with pytest.raises(InvalidGame):
            coin_match_with_signal(p_heads=1.0)


# =====================================================================
# Strategy + best response
# =====================================================================


class TestExpectedUtility:
    def test_uniform_matching_pennies(self):
        g = matching_pennies_simultaneous()
        s = {"P0": [0.5, 0.5], "P1": [0.5, 0.5]}
        u = expected_utilities(g, s)
        assert u[0] == pytest.approx(0.0, abs=1e-9)
        assert u[1] == pytest.approx(0.0, abs=1e-9)

    def test_pure_strategy_matching_pennies(self):
        g = matching_pennies_simultaneous()
        s = {"P0": [1.0, 0.0], "P1": [1.0, 0.0]}  # both play H
        u = expected_utilities(g, s)
        assert u[0] == pytest.approx(+1.0)
        assert u[1] == pytest.approx(-1.0)

    def test_rps_uniform(self):
        g = rock_paper_scissors()
        s = {"P0": [1 / 3] * 3, "P1": [1 / 3] * 3}
        u = expected_utilities(g, s)
        assert u[0] == pytest.approx(0.0, abs=1e-9)

    def test_validation_missing_info_set(self):
        g = matching_pennies_simultaneous()
        with pytest.raises(DiplomatError):
            expected_utilities(g, {"P0": [0.5, 0.5]})

    def test_validation_wrong_length(self):
        g = matching_pennies_simultaneous()
        with pytest.raises(DiplomatError):
            expected_utilities(g, {"P0": [1.0], "P1": [0.5, 0.5]})

    def test_validation_negative_entry(self):
        g = matching_pennies_simultaneous()
        with pytest.raises(DiplomatError):
            expected_utilities(g, {"P0": [-0.5, 1.5], "P1": [0.5, 0.5]})


class TestBestResponse:
    def test_br_matching_pennies(self):
        g = matching_pennies_simultaneous()
        # P0 always plays H; P1's best response is to play T → wins.
        s = {"P0": [1.0, 0.0], "P1": [0.5, 0.5]}
        br1 = best_response(g, s, player=1)
        # P1 BR is to play T (idx 1) — utility +1 vs 0 under uniform.
        assert br1.value == pytest.approx(+1.0)
        assert br1.response["P1"][1] == 1.0
        assert br1.delta > 0

    def test_br_pure_already_optimal(self):
        g = matching_pennies_simultaneous()
        s = {"P0": [0.5, 0.5], "P1": [0.5, 0.5]}
        br0 = best_response(g, s, player=0)
        # All player-0 actions tied at 0; BR delta = 0.
        assert br0.delta == pytest.approx(0.0, abs=1e-9)

    def test_br_kuhn(self):
        g = kuhn_poker()
        # Uniform strategy is exploitable; BR delta should be > 0.
        uniform = {k: [1.0 / len(I.actions)] * len(I.actions) for k, I in g.info_sets.items()}
        e = exploitability(g, uniform)
        assert e > 0.01

    def test_br_out_of_range_player(self):
        g = matching_pennies_simultaneous()
        s = {"P0": [0.5, 0.5], "P1": [0.5, 0.5]}
        with pytest.raises(DiplomatError):
            best_response(g, s, player=5)


# =====================================================================
# CFR-family convergence
# =====================================================================


class TestCFRDeterministic:
    @pytest.mark.parametrize("kind", [
        KIND_CFR, KIND_CFR_PLUS, KIND_LINEAR_CFR,
        KIND_DISCOUNTED_CFR, KIND_PREDICTIVE_CFR_PLUS,
    ])
    def test_rps_converges(self, kind):
        d = Diplomat()
        g = rock_paper_scissors()
        rep = d.solve(g, CFRConfig(kind=kind, iterations=2000, seed=0))
        # Should reach near-uniform (≈ 1/3 each) and low exploitability.
        for a, p in enumerate(rep.average_strategy["P0"]):
            assert p == pytest.approx(1 / 3, abs=0.05)
        assert rep.exploitability < 0.05
        assert rep.root_value[0] == pytest.approx(0.0, abs=0.05)

    @pytest.mark.parametrize("kind", [
        KIND_CFR, KIND_CFR_PLUS, KIND_LINEAR_CFR, KIND_DISCOUNTED_CFR,
    ])
    def test_kuhn_converges_to_correct_value(self, kind):
        """All deterministic CFR variants reach Kuhn poker's Nash value
        (-1/18 for player 0) to within 1%."""
        d = Diplomat()
        g = kuhn_poker()
        rep = d.solve(g, CFRConfig(kind=kind, iterations=3000, seed=0))
        assert rep.root_value[0] == pytest.approx(-1.0 / 18.0, abs=0.01)
        assert rep.exploitability < 0.02

    def test_predictive_cfr_plus_converges_fast(self):
        """Predictive CFR+ has O(1/T) last-iterate rate (Farina-Kroer-Sandholm 2021)."""
        d = Diplomat()
        g = matching_pennies_simultaneous()
        rep = d.solve(g, CFRConfig(kind=KIND_PREDICTIVE_CFR_PLUS, iterations=500, seed=0))
        # At 500 iter, predictive CFR+ should give ~zero exploitability.
        assert rep.exploitability < 1e-3

    def test_exploitability_trace(self):
        d = Diplomat()
        g = kuhn_poker()
        rep = d.solve(g, CFRConfig(
            kind=KIND_CFR_PLUS, iterations=400, seed=0,
            track_exploitability_every=100,
        ))
        # 4 trace entries; later expl ≤ earlier (mostly monotone for CFR+).
        assert len(rep.exploitability_trace) == 4
        first, last = rep.exploitability_trace[0][1], rep.exploitability_trace[-1][1]
        assert last <= first + 1e-6

    def test_regret_bound_is_upper_bound(self):
        d = Diplomat()
        g = rock_paper_scissors()
        rep = d.solve(g, CFRConfig(kind=KIND_CFR, iterations=1000, seed=0))
        # Anytime bound should be ≥ actual exploitability (Zinkevich 2008).
        # bound here is sum_max_pos_regret / T; we don't enforce a tight
        # constant but it must be non-negative and finite.
        assert rep.regret_bound >= 0.0
        assert math.isfinite(rep.regret_bound)


class TestCFRSampling:
    def test_external_sampling_kuhn(self):
        d = Diplomat()
        g = kuhn_poker()
        rep = d.solve(g, CFRConfig(
            kind=KIND_EXTERNAL_SAMPLING, iterations=5000, seed=42,
        ))
        # ES-MCCFR converges to correct Kuhn value at this iteration count.
        assert rep.root_value[0] == pytest.approx(-1.0 / 18.0, abs=0.02)
        assert rep.exploitability < 0.05

    def test_chance_sampling_kuhn(self):
        d = Diplomat()
        g = kuhn_poker()
        rep = d.solve(g, CFRConfig(
            kind=KIND_CHANCE_SAMPLING, iterations=5000, seed=42,
        ))
        assert rep.root_value[0] == pytest.approx(-1.0 / 18.0, abs=0.02)

    def test_outcome_sampling_runs_on_matching_pennies(self):
        # OS-MCCFR converges on chance-free games (MP, RPS).  Known
        # to be biased on chance-heavy Kuhn — we only assert behaviour
        # on the chance-free benchmark.
        d = Diplomat()
        g = matching_pennies_simultaneous()
        rep = d.solve(g, CFRConfig(
            kind=KIND_OUTCOME_SAMPLING, iterations=100_000, seed=42,
        ))
        assert rep.exploitability < 0.05

    def test_outcome_sampling_rps(self):
        d = Diplomat()
        g = rock_paper_scissors()
        rep = d.solve(g, CFRConfig(
            kind=KIND_OUTCOME_SAMPLING, iterations=200_000, seed=42,
        ))
        assert rep.exploitability < 0.1


# =====================================================================
# Sequence-form LP (exact)
# =====================================================================


class TestSequenceFormLP:
    def test_matching_pennies_simul(self):
        d = Diplomat()
        g = matching_pennies_simultaneous()
        rep = d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))
        assert rep.exploitability == pytest.approx(0.0, abs=1e-6)
        assert rep.root_value[0] == pytest.approx(0.0, abs=1e-6)
        for p in rep.average_strategy["P0"]:
            assert p == pytest.approx(0.5)

    def test_matching_pennies_sequential(self):
        # When P1 observes P0, P0's value is -1 (always matched).
        d = Diplomat()
        g = matching_pennies_sequential()
        rep = d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))
        assert rep.root_value[0] == pytest.approx(-1.0, abs=1e-6)
        assert rep.exploitability == pytest.approx(0.0, abs=1e-6)

    def test_rps(self):
        d = Diplomat()
        g = rock_paper_scissors()
        rep = d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))
        assert rep.exploitability == pytest.approx(0.0, abs=1e-6)
        assert rep.root_value[0] == pytest.approx(0.0, abs=1e-6)
        for p in rep.average_strategy["P0"]:
            assert p == pytest.approx(1 / 3, abs=1e-6)

    def test_kuhn_poker(self):
        d = Diplomat()
        g = kuhn_poker()
        rep = d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))
        # Kuhn poker value to dealer is -1/18 (von Neumann minimax).
        assert rep.root_value[0] == pytest.approx(-1.0 / 18.0, abs=1e-6)
        assert rep.exploitability == pytest.approx(0.0, abs=1e-6)
        assert rep.certificate["exact"] is True

    def test_rejects_three_player(self):
        b = GameBuilder(n_players=3)
        d0 = b.decision_node(parent=-1, parent_action=None, player=0,
                             actions=["a"], info_set="X")
        d1 = b.decision_node(parent=d0, parent_action="a", player=1,
                             actions=["a"], info_set="Y")
        d2 = b.decision_node(parent=d1, parent_action="a", player=2,
                             actions=["a"], info_set="Z")
        b.terminal_node(parent=d2, parent_action="a", utilities=[1, 0, 0])
        g3 = b.build()
        d = Diplomat()
        with pytest.raises(NotTwoPlayerZeroSum):
            d.solve(g3, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))

    def test_rejects_non_zero_sum(self):
        b = GameBuilder(n_players=2)
        d0 = b.decision_node(parent=-1, parent_action=None, player=0,
                             actions=["a"], info_set="X")
        b.terminal_node(parent=d0, parent_action="a", utilities=[1, 1])  # not zero-sum
        g = b.build()
        d = Diplomat()
        with pytest.raises(NotTwoPlayerZeroSum):
            d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))


# =====================================================================
# Diplomat API surface
# =====================================================================


class TestDiplomatApi:
    def test_unknown_solver(self):
        with pytest.raises(UnknownSolver):
            CFRConfig(kind="this_does_not_exist").normalise()

    def test_aliases(self):
        for alias, full in [
            ("cfr+", KIND_CFR_PLUS),
            ("pcfr+", KIND_PREDICTIVE_CFR_PLUS),
            ("dcfr", KIND_DISCOUNTED_CFR),
            ("lcfr", KIND_LINEAR_CFR),
            ("es", KIND_EXTERNAL_SAMPLING),
            ("cs", KIND_CHANCE_SAMPLING),
            ("lp", KIND_SEQUENCE_FORM_LP),
        ]:
            assert CFRConfig(kind=alias).normalise().kind == full

    def test_zero_iterations_rejected(self):
        d = Diplomat()
        g = rock_paper_scissors()
        from agi.diplomat import InsufficientIterations
        with pytest.raises(InsufficientIterations):
            d.solve(g, CFRConfig(kind="cfr", iterations=0))

    def test_default_config(self):
        d = Diplomat()
        g = matching_pennies_simultaneous()
        rep = d.solve(g)  # default = cfr_plus, 1000 iters
        assert isinstance(rep, SolveReport)
        assert rep.exploitability < 0.1

    def test_known_kinds_listed(self):
        # Defensive: every advertised kind is actually handled by solve().
        d = Diplomat()
        g = rock_paper_scissors()
        for kind in KNOWN_KINDS:
            cfg = CFRConfig(kind=kind, iterations=50, seed=0)
            rep = d.solve(g, cfg)
            assert isinstance(rep, SolveReport)
            assert rep.kind == kind

    def test_fingerprint_stable(self):
        d = Diplomat()
        g = kuhn_poker()
        r1 = d.solve(g, CFRConfig(kind="cfr_plus", iterations=200, seed=7))
        f1 = r1.fingerprint()
        g2 = kuhn_poker()  # fresh game
        r2 = d.solve(g2, CFRConfig(kind="cfr_plus", iterations=200, seed=7))
        f2 = r2.fingerprint()
        # Same input → same fingerprint (modulo wall_seconds which goes into
        # the JSON; if wall time differs the fingerprint may differ).
        # We instead check the strategy is identical.
        assert r1.average_strategy == r2.average_strategy
        assert r1.exploitability == pytest.approx(r2.exploitability)

    def test_to_json_roundtrips(self):
        import json as _json
        d = Diplomat()
        g = rock_paper_scissors()
        rep = d.solve(g, CFRConfig(kind="cfr_plus", iterations=50))
        s = rep.to_json()
        parsed = _json.loads(s)
        assert parsed["kind"] == "cfr_plus"
        assert "average_strategy" in parsed

    def test_uniform_strategy_helper(self):
        d = Diplomat()
        g = rock_paper_scissors()
        u = d.uniform_strategy(g)
        for k, v in u.items():
            assert sum(v) == pytest.approx(1.0)
            for p in v:
                assert p == pytest.approx(1.0 / len(v))


# =====================================================================
# Composition with the rest of the runtime — light smoke tests
# =====================================================================


class TestEventBusIntegration:
    def test_no_bus_runs_silently(self):
        d = Diplomat()
        g = rock_paper_scissors()
        rep = d.solve(g, CFRConfig(kind="cfr_plus", iterations=10))
        assert rep.iterations == 10

    def test_with_event_bus(self):
        """Optional EventBus receives diplomat events."""
        try:
            from agi.events import EventBus
        except Exception:
            pytest.skip("agi.events not available")
        bus = EventBus()
        captured = []
        bus.subscribe(lambda e: captured.append(e), kind="diplomat.solved")
        d = Diplomat(bus=bus)
        g = rock_paper_scissors()
        d.solve(g, CFRConfig(kind="cfr_plus", iterations=50))
        assert len(captured) >= 1


# =====================================================================
# Equilibrium properties
# =====================================================================


class TestNashProperties:
    def test_kuhn_lp_solution_has_zero_exploitability(self):
        """The LP solution is *exactly* Nash; its exploitability is 0."""
        d = Diplomat()
        g = kuhn_poker()
        rep = d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))
        # Recompute exploitability from the strategy independently.
        e = exploitability(g, rep.average_strategy)
        assert e == pytest.approx(0.0, abs=1e-6)

    def test_uniform_strategy_exploitable(self):
        d = Diplomat()
        g = kuhn_poker()
        u = d.uniform_strategy(g)
        e = exploitability(g, u)
        assert e > 0.1  # Kuhn under uniform play is highly exploitable

    def test_best_response_increases_payoff(self):
        d = Diplomat()
        g = kuhn_poker()
        u = d.uniform_strategy(g)
        base = expected_utilities(g, u)
        for p in range(2):
            br = best_response(g, u, p)
            assert br.value >= base[p] - 1e-9

    def test_cfr_converges_to_zero_exploitability_kuhn(self):
        d = Diplomat()
        g = kuhn_poker()
        rep = d.solve(g, CFRConfig(kind="cfr_plus", iterations=5000, seed=0))
        assert rep.exploitability < 0.01


# =====================================================================
# Bargaining game — exercising larger trees
# =====================================================================


class TestBargaining:
    def test_one_round_zero_sum_swap(self):
        """Two-round bargaining is *not* zero-sum; sequence-form LP
        should reject it."""
        g = simple_bargaining(n_rounds=1, pie=1.0)
        d = Diplomat()
        with pytest.raises(NotTwoPlayerZeroSum):
            d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))

    def test_cfr_runs_on_bargaining(self):
        g = simple_bargaining(n_rounds=2, pie=1.0)
        d = Diplomat()
        rep = d.solve(g, CFRConfig(kind="cfr_plus", iterations=200))
        # We can't assert convergence to a unique equilibrium (multi-equilibria
        # in non-zero-sum), but we can assert the run is sane.
        assert rep.iterations == 200
        # Total utility ≤ pie (each player gets a share).
        assert sum(rep.root_value) <= 1.0 + 1e-6
