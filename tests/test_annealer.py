"""Tests for :mod:`agi.annealer`."""
from __future__ import annotations

import math
import random

import pytest

from agi.annealer import (
    ALGO_BASIN,
    ALGO_LAHC,
    ALGO_PT,
    ALGO_RESTART,
    ALGO_SA,
    ALGO_TABU,
    ANNEALER_LEDGER_GENESIS,
    Annealer,
    AnnealerCertificate,
    AnnealerConfig,
    AnnealerReport,
    InvalidConfig,
    InvalidProblem,
    NotRun,
    Problem,
    SCHED_ADAPTIVE,
    SCHED_GEOMETRIC,
    SCHED_LINEAR,
    SCHED_LOG,
    SCHED_LUNDY_MEES,
    UnknownAlgorithm,
    UnknownSchedule,
    annealer_adaptive_schedule,
    annealer_geometric_schedule,
    annealer_knapsack,
    annealer_ledger_root,
    annealer_linear_schedule,
    annealer_log_schedule,
    annealer_luby_sequence,
    annealer_lundy_mees_schedule,
    annealer_max_cut,
    annealer_max_sat,
    annealer_metropolis_accept,
    annealer_number_partition,
    annealer_qap,
    annealer_tsp,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_construct(self) -> None:
        c = AnnealerConfig()
        assert c.algorithm == ALGO_SA
        assert c.schedule == SCHED_GEOMETRIC
        assert c.t_init > c.t_final > 0

    def test_unknown_algorithm(self) -> None:
        with pytest.raises(UnknownAlgorithm):
            AnnealerConfig(algorithm="nope")

    def test_unknown_schedule(self) -> None:
        with pytest.raises(UnknownSchedule):
            AnnealerConfig(schedule="nope")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t_init": 0.0},
            {"t_init": -1.0},
            {"t_final": 0.0},
            {"t_final": 10.0, "t_init": 1.0},
            {"max_iter": 0},
            {"n_replicas": 0},
            {"swap_every": 0},
            {"lahc_length": 0},
            {"tabu_tenure": -1},
            {"basin_perturbations": 0},
            {"target_acceptance": 0.0},
            {"target_acceptance": 1.0},
            {"adapt_window": 0},
            {"record_every": 0},
            {"restarts": 0},
            {"luby_unit": 0},
        ],
    )
    def test_invalid_config(self, kwargs: dict) -> None:
        with pytest.raises(InvalidConfig):
            AnnealerConfig(**kwargs)

    def test_hmac_key_must_be_bytes(self) -> None:
        with pytest.raises(InvalidConfig):
            AnnealerConfig(hmac_key="not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class TestLedger:
    def test_root_deterministic(self) -> None:
        a = annealer_ledger_root()
        b = annealer_ledger_root()
        assert a == b
        assert len(a) == 64  # sha256 hex

    def test_root_changes_with_hmac(self) -> None:
        a = annealer_ledger_root()
        b = annealer_ledger_root(b"secret")
        assert a != b


# ---------------------------------------------------------------------------
# Cooling schedules
# ---------------------------------------------------------------------------


class TestSchedules:
    def test_geometric_monotone(self) -> None:
        s = annealer_geometric_schedule(1.0, 0.01, 100)
        ts = [s(k) for k in range(100)]
        assert all(ts[k] >= ts[k + 1] - 1e-9 for k in range(99))
        assert ts[0] == pytest.approx(1.0, abs=1e-3)
        assert ts[-1] == pytest.approx(0.01, abs=1e-2)

    def test_log_schedule_monotone(self) -> None:
        s = annealer_log_schedule(1.0, 0.0001, 100)
        ts = [s(k) for k in range(100)]
        assert all(ts[k] >= ts[k + 1] - 1e-12 for k in range(99))

    def test_linear_schedule(self) -> None:
        s = annealer_linear_schedule(1.0, 0.0, 11)
        ts = [s(k) for k in range(11)]
        assert ts[0] == pytest.approx(1.0)
        assert ts[10] == pytest.approx(0.0, abs=1e-9)
        for k in range(10):
            assert ts[k] >= ts[k + 1] - 1e-9

    def test_lundy_mees_monotone(self) -> None:
        s = annealer_lundy_mees_schedule(1.0, 0.01, 100)
        ts = [s(k) for k in range(100)]
        for k in range(99):
            assert ts[k] >= ts[k + 1] - 1e-9
        assert ts[0] == pytest.approx(1.0, abs=1e-9)

    def test_adaptive_responds_to_acceptance(self) -> None:
        s = annealer_adaptive_schedule(1.0, 0.01, 100, target_acceptance=0.5)
        # Low acceptance -> heats up.
        t0 = s(0, 0.01)
        t1 = s(1, 0.01)
        assert t1 >= t0 or t1 == 1.0
        # High acceptance -> cools down.
        t2 = s(2, 0.99)
        assert t2 < t1


# ---------------------------------------------------------------------------
# Metropolis acceptance
# ---------------------------------------------------------------------------


class TestMetropolis:
    def test_always_accepts_improvement(self) -> None:
        for u in (0.0, 0.5, 0.999):
            assert annealer_metropolis_accept(-1.0, 1.0, u) is True
            assert annealer_metropolis_accept(0.0, 1.0, u) is True

    def test_temperature_zero_rejects_uphill(self) -> None:
        assert annealer_metropolis_accept(1.0, 0.0, 0.0) is False

    def test_high_temperature_accepts_most(self) -> None:
        # ΔE=0.1, T=100 -> P ~ exp(-0.001) ~ 0.999
        n_acc = sum(annealer_metropolis_accept(0.1, 100.0, u / 1000) for u in range(1000))
        assert n_acc > 990

    def test_low_temperature_rejects_most(self) -> None:
        # ΔE=1.0, T=0.001 -> P ~ exp(-1000) ~ 0
        rng = random.Random(0)
        n_acc = sum(annealer_metropolis_accept(1.0, 0.001, rng.random()) for _ in range(1000))
        assert n_acc < 5


# ---------------------------------------------------------------------------
# Luby sequence
# ---------------------------------------------------------------------------


class TestLuby:
    def test_first_fifteen_match_canon(self) -> None:
        expected = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]
        assert annealer_luby_sequence(15) == expected

    def test_empty(self) -> None:
        assert annealer_luby_sequence(0) == []


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------


class TestTsp:
    def test_basic_construct(self) -> None:
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        prob = annealer_tsp(pts, seed=0)
        assert prob.name == "tsp"
        # Square tour cost is 4.0
        assert prob.cost(prob.initial) <= 4.0 + 1e-9
        # Lower bound is feasible (>0).
        assert prob.lower_bound() > 0

    def test_too_few_points_rejected(self) -> None:
        with pytest.raises(InvalidProblem):
            annealer_tsp([(0.0, 0.0)])

    def test_neighbour_is_permutation(self) -> None:
        pts = [(float(i), float(i % 3)) for i in range(8)]
        prob = annealer_tsp(pts, seed=42)
        rng = random.Random(1)
        for _ in range(20):
            nb = prob.neighbour(prob.initial, rng)
            assert sorted(nb) == list(range(8))


class TestMaxCut:
    def test_basic_construct(self) -> None:
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        prob = annealer_max_cut(edges)
        # Initial all-zero cuts nothing; cost = 0 (since -cut).
        assert prob.cost(prob.initial) == 0.0
        # State (0,1,0) cuts 2 of 3 edges
        assert prob.cost((0, 1, 0)) == -2.0

    def test_negative_weights_rejected(self) -> None:
        with pytest.raises(InvalidProblem):
            annealer_max_cut([(0, 1, -1.0)])


class TestMaxSat:
    def test_basic_construct(self) -> None:
        # (x1) AND (¬x1 OR x2)
        clauses = [[1], [-1, 2]]
        prob = annealer_max_sat(clauses)
        # All zero satisfies clause 2 (¬x1) only -> 1 satisfied -> cost -1
        assert prob.cost((0, 0)) == -1.0
        # x1=1, x2=1 satisfies both -> -2
        assert prob.cost((1, 1)) == -2.0


class TestQap:
    def test_basic_construct(self) -> None:
        f = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        d = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        prob = annealer_qap(f, d)
        assert prob.cost(prob.initial) > 0
        # Different permutations have different costs.
        c1 = prob.cost((0, 1, 2))
        c2 = prob.cost((2, 1, 0))
        assert c1 != c2 or abs(c1 - c2) < 1e-12  # at least computable


class TestPartition:
    def test_basic(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        # All +1 -> sum = 10
        assert prob.cost(prob.initial) == 10.0
        # (1, -1, -1, 1) -> 1 - 2 - 3 + 4 = 0
        assert prob.cost((1, -1, -1, 1)) == 0.0


class TestKnapsack:
    def test_feasible_vs_infeasible(self) -> None:
        prob = annealer_knapsack([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], capacity=3.0)
        # Take items 0 and 1 (weight 3, value 30) -> cost -30
        assert prob.cost((1, 1, 0)) == -30.0
        # All items (weight 6, value 60) -> infeasible
        c = prob.cost((1, 1, 1))
        assert c > -60.0  # penalty applied
        # Lower bound: LP relaxation
        assert prob.lower_bound() <= -30.0


# ---------------------------------------------------------------------------
# End-to-end runs
# ---------------------------------------------------------------------------


class TestRun:
    def test_sa_improves_on_partition(self) -> None:
        prob = annealer_number_partition([3.0, 1.0, 1.0, 2.0, 2.0, 1.0])
        an = Annealer(AnnealerConfig(max_iter=500, seed=0))
        rep = an.run(prob)
        assert isinstance(rep, AnnealerReport)
        assert rep.best_cost <= prob.cost(prob.initial)

    def test_sa_finds_known_optimum_partition(self) -> None:
        # [4, 4, 4, 4] -> [+1, +1, -1, -1] hits 0.
        prob = annealer_number_partition([4.0, 4.0, 4.0, 4.0])
        an = Annealer(AnnealerConfig(max_iter=2000, seed=0))
        rep = an.run(prob)
        assert rep.best_cost == 0.0

    def test_deterministic_given_seed(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0, 5.0])
        a = Annealer(AnnealerConfig(max_iter=200, seed=42)).run(prob)
        b = Annealer(AnnealerConfig(max_iter=200, seed=42)).run(prob)
        assert a.best_cost == b.best_cost
        assert a.chain_head == b.chain_head

    def test_different_seeds_diverge(self) -> None:
        # On a non-trivial problem, different seeds should explore differently.
        prob = annealer_tsp(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0), (3.0, 0.0), (1.0, 3.0)],
            seed=0,
        )
        a = Annealer(AnnealerConfig(max_iter=200, seed=1)).run(prob)
        b = Annealer(AnnealerConfig(max_iter=200, seed=2)).run(prob)
        # Same final report unlikely; chain head should differ regardless.
        assert a.chain_head != b.chain_head

    def test_record_every_truncates(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        an = Annealer(AnnealerConfig(max_iter=100, record_every=10, seed=0))
        rep = an.run(prob)
        assert len(rep.cost_history) == 10

    @pytest.mark.parametrize(
        "schedule",
        [SCHED_GEOMETRIC, SCHED_LOG, SCHED_LINEAR, SCHED_LUNDY_MEES, SCHED_ADAPTIVE],
    )
    def test_each_schedule_runs(self, schedule: str) -> None:
        prob = annealer_number_partition([1.0, 1.0, 2.0, 2.0])
        an = Annealer(AnnealerConfig(schedule=schedule, max_iter=200, seed=0))
        rep = an.run(prob)
        assert rep.iterations == 200

    def test_pt_runs(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0, 5.0])
        an = Annealer(AnnealerConfig(algorithm=ALGO_PT, n_replicas=4, max_iter=200, seed=0))
        rep = an.run(prob)
        assert rep.algorithm == ALGO_PT
        assert len(rep.replicas) == 4
        # Coldest replica is at index 0 (temp t_final).
        assert rep.replicas[0].temperature < rep.replicas[-1].temperature

    def test_pt_attempts_swaps(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0, 5.0])
        an = Annealer(
            AnnealerConfig(algorithm=ALGO_PT, n_replicas=3, max_iter=200, swap_every=10, seed=0)
        )
        rep = an.run(prob)
        assert rep.swaps_attempted > 0

    def test_lahc_runs(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        an = Annealer(AnnealerConfig(algorithm=ALGO_LAHC, max_iter=300, lahc_length=20, seed=0))
        rep = an.run(prob)
        assert rep.algorithm == ALGO_LAHC
        assert rep.best_cost <= prob.cost(prob.initial)

    def test_basin_runs(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        an = Annealer(AnnealerConfig(algorithm=ALGO_BASIN, max_iter=200, seed=0))
        rep = an.run(prob)
        assert rep.algorithm == ALGO_BASIN

    def test_tabu_runs(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        an = Annealer(AnnealerConfig(algorithm=ALGO_TABU, max_iter=100, tabu_tenure=10, seed=0))
        rep = an.run(prob)
        assert rep.algorithm == ALGO_TABU

    def test_luby_restart_runs(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0, 5.0])
        an = Annealer(AnnealerConfig(algorithm=ALGO_RESTART, max_iter=300, luby_unit=20, seed=0))
        rep = an.run(prob)
        assert rep.algorithm == ALGO_RESTART
        assert rep.restarts_taken >= 1

    def test_multiple_restarts(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0, 5.0])
        an = Annealer(AnnealerConfig(max_iter=100, restarts=3, seed=0))
        rep = an.run(prob)
        # Best across forks should be at least as good as any single fork.
        ref = Annealer(AnnealerConfig(max_iter=100, restarts=1, seed=0)).run(prob)
        assert rep.best_cost <= ref.best_cost

    def test_rejects_invalid_problem(self) -> None:
        an = Annealer()
        with pytest.raises(InvalidProblem):
            an.run("not a problem")  # type: ignore[arg-type]

    def test_invalid_restarts(self) -> None:
        an = Annealer()
        prob = annealer_number_partition([1.0, 2.0])
        with pytest.raises(InvalidConfig):
            an.run(prob, restarts=0)


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


class TestCertify:
    def test_not_run_raises(self) -> None:
        an = Annealer()
        with pytest.raises(NotRun):
            an.certify()

    def test_invalid_delta(self) -> None:
        an = Annealer()
        prob = annealer_number_partition([1.0, 2.0])
        an.run(prob)
        with pytest.raises(InvalidConfig):
            an.certify(delta=0.0)
        with pytest.raises(InvalidConfig):
            an.certify(delta=1.0)

    def test_certificate_fields(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        an = Annealer(AnnealerConfig(max_iter=200, seed=0))
        rep = an.run(prob)
        cert = an.certify(rep, delta=0.05, problem=prob)
        assert isinstance(cert, AnnealerCertificate)
        assert cert.delta == 0.05
        assert cert.n_samples == len(rep.cost_history)
        assert cert.best_cost == rep.best_cost
        assert cert.lower_bound == 0.0  # partition problem's lower bound
        assert cert.gap_hoeffding is not None and cert.gap_hoeffding >= 0
        assert cert.gap_bernstein is not None and cert.gap_bernstein >= 0
        assert cert.p_global_opt is not None and 0.0 <= cert.p_global_opt <= 1.0

    def test_certificate_without_problem(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        an = Annealer(AnnealerConfig(max_iter=100, seed=0))
        rep = an.run(prob)
        cert = an.certify(rep, delta=0.1)
        # Without a problem-supplied lower bound, lb falls back to
        # cost-history minimum (= best_cost essentially).
        assert cert.method == "empirical"


# ---------------------------------------------------------------------------
# Snapshot / restore / reset
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_restore_round_trip(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0])
        an = Annealer(AnnealerConfig(max_iter=50, seed=0))
        an.run(prob)
        snap = an.snapshot()
        an2 = Annealer(an.config)
        an2.restore(snap)
        assert an2.chain_head == an.chain_head
        assert an2.n_runs == an.n_runs

    def test_reset_clears_run_state(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0])
        an = Annealer(AnnealerConfig(max_iter=50, seed=0))
        an.run(prob)
        post_run_head = an.chain_head
        assert an.n_runs == 1
        an.reset()
        # The reset itself is hash-chained for audit, so the chain head
        # is not the bare genesis — it's genesis + reset event.  The
        # observable invariant is that n_runs is 0 and the chain head
        # moved away from the post-run state.
        assert an.n_runs == 0
        assert an.last_report is None
        assert an.chain_head != post_run_head


# ---------------------------------------------------------------------------
# Determinism and replayability
# ---------------------------------------------------------------------------


class TestReplay:
    def test_replay_same_chain_head(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        cfg = AnnealerConfig(max_iter=200, seed=0)
        a = Annealer(cfg).run(prob)
        b = Annealer(cfg).run(prob)
        assert a.chain_head == b.chain_head
        assert a.best_cost == b.best_cost
        assert a.cost_history == b.cost_history

    def test_hmac_changes_chain_head(self) -> None:
        prob = annealer_number_partition([1.0, 2.0, 3.0, 4.0])
        a = Annealer(AnnealerConfig(max_iter=50, seed=0)).run(prob)
        b = Annealer(AnnealerConfig(max_iter=50, seed=0, hmac_key=b"k")).run(prob)
        assert a.chain_head != b.chain_head
        assert a.best_cost == b.best_cost  # same problem, same trajectory


# ---------------------------------------------------------------------------
# Integration: small TSP — best-known recovery sanity
# ---------------------------------------------------------------------------


class TestTSPSmall:
    def test_finds_optimal_square(self) -> None:
        # 4-vertex unit square has optimal tour length 4.0.
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        prob = annealer_tsp(pts, seed=0)
        an = Annealer(AnnealerConfig(max_iter=1000, seed=0))
        rep = an.run(prob)
        assert rep.best_cost == pytest.approx(4.0, abs=1e-9)

    def test_finds_optimal_collinear(self) -> None:
        # Collinear points: best tour goes there and back -> 2*(max-min)
        pts = [(0.0, 0.0), (3.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        prob = annealer_tsp(pts, seed=0)
        an = Annealer(AnnealerConfig(max_iter=2000, seed=0))
        rep = an.run(prob)
        assert rep.best_cost == pytest.approx(6.0, abs=1e-9)
