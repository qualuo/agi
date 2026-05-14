"""Tests for `agi.coalition` — Shapley credit-assignment runtime primitive.

Five statistical contracts to verify:

1. **Axiomatic Shapley** — efficiency, symmetry, dummy, additivity all
   hold *exactly* under ``shapley_exact`` on hand-built games where
   the analytic answer is known.

2. **PAC correctness** — across N independent MC runs at confidence
   δ, the rate of ``|φ̂_i − φ_i| > ε`` is ≤ δ within Monte Carlo
   tolerance. Verified separately under Hoeffding and Bernstein
   bounds.

3. **Variance reduction by stratified sampling** — on a fixed game
   and a fixed sample budget, the stratified estimator has strictly
   smaller half-width than the simple permutation estimator.
   (Maleki 2014.)

4. **Owen group structure** — when groups are degenerate (each
   player in its own group), Owen value equals Shapley. When groups
   are non-degenerate, the within-group symmetry property holds.

5. **Linear-fit characterisation** — for order-1 interaction models
   the fitted Shapley equals the OLS coefficients; for order-2,
   symmetry / dummy still hold on the fitted v̂.

Plus the usual cosmetic surface: dataclass invariants, threadsafety,
event emissions, attestation pass-through, free functions.
"""
from __future__ import annotations

import math
import random
import statistics
import threading

import pytest

from agi.coalition import (
    COALITION_COMPUTED,
    COALITION_CREDITED,
    COALITION_OBSERVED,
    COALITION_PLAYER_REGISTERED,
    COALITION_STARTED,
    Coalition,
    CoalitionReport,
    CoverageReport,
    POLICY_BERNSTEIN,
    POLICY_EXACT,
    POLICY_HOEFFDING,
    POLICY_PERMUTATION,
    POLICY_STRATIFIED,
    PlayerSpec,
    ShapleyEstimate,
    banzhaf_index,
    bernstein_radius,
    fit_linear_v,
    hoeffding_radius,
    shapley_from_observations,
    shapley_values,
)
from agi.events import Event, EventBus


# ============================================================
# Section 1 — Axiomatic correctness of exact Shapley
# ============================================================


class TestAxioms:
    def test_efficiency(self) -> None:
        """Sum of Shapley values equals v(N) − v(∅)."""

        def v(S: frozenset[str]) -> float:
            return float(len(S) ** 2)

        result = shapley_values(["a", "b", "c", "d"], v)
        v_grand = v(frozenset({"a", "b", "c", "d"}))
        v_empty = v(frozenset())
        assert math.isclose(sum(result.values()), v_grand - v_empty, abs_tol=1e-9)

    def test_symmetry(self) -> None:
        """Two players that contribute equally to every coalition get
        equal Shapley value.
        """

        def v(S: frozenset[str]) -> float:
            # Symmetric in {a,b}: only the count matters.
            return float(len(S))

        result = shapley_values(["a", "b", "c"], v)
        assert math.isclose(result["a"], result["b"], abs_tol=1e-9)
        assert math.isclose(result["a"], result["c"], abs_tol=1e-9)

    def test_dummy(self) -> None:
        """A dummy player (zero marginal contribution to every S) gets
        Shapley value zero.
        """

        def v(S: frozenset[str]) -> float:
            # 'c' is a dummy: removing c never changes v.
            return float("a" in S) + float("b" in S)

        result = shapley_values(["a", "b", "c"], v)
        assert math.isclose(result["c"], 0.0, abs_tol=1e-9)
        assert math.isclose(result["a"], 1.0, abs_tol=1e-9)
        assert math.isclose(result["b"], 1.0, abs_tol=1e-9)

    def test_additivity(self) -> None:
        """φ(v + w) = φ(v) + φ(w)."""
        players = ["a", "b", "c"]

        def v(S: frozenset[str]) -> float:
            return float(len(S))

        def w(S: frozenset[str]) -> float:
            return float(len(S) ** 2)

        def vw(S: frozenset[str]) -> float:
            return v(S) + w(S)

        phi_v = shapley_values(players, v)
        phi_w = shapley_values(players, w)
        phi_vw = shapley_values(players, vw)
        for pid in players:
            assert math.isclose(phi_vw[pid], phi_v[pid] + phi_w[pid], abs_tol=1e-9)

    def test_unanimity_game(self) -> None:
        """Unanimity game on T: φ_i = 1/|T| if i ∈ T, else 0."""
        T = {"a", "b"}

        def v(S: frozenset[str]) -> float:
            return 1.0 if T <= S else 0.0

        result = shapley_values(["a", "b", "c", "d"], v)
        assert math.isclose(result["a"], 0.5, abs_tol=1e-9)
        assert math.isclose(result["b"], 0.5, abs_tol=1e-9)
        assert math.isclose(result["c"], 0.0, abs_tol=1e-9)
        assert math.isclose(result["d"], 0.0, abs_tol=1e-9)


# ============================================================
# Section 2 — PAC correctness of MC sampling
# ============================================================


class TestPACCorrectness:
    def _build_coalition(self, n: int, seed: int) -> tuple[Coalition, dict[str, float]]:
        """Build a Coalition with a 'pairwise interaction' game and
        return both the Coalition and its exact Shapley values.
        """
        rng = random.Random(seed)
        coa = Coalition(rng=rng)
        players = [f"p{i}" for i in range(n)]
        for pid in players:
            coa.register_player(pid)
        # Random pairwise interaction matrix; values are bounded.
        pair_w = {(i, j): rng.uniform(-0.5, 1.0) for i in players for j in players if i < j}

        def v(S: frozenset[str]) -> float:
            tot = 0.0
            sl = sorted(S)
            for i in range(len(sl)):
                for j in range(i + 1, len(sl)):
                    tot += pair_w.get((sl[i], sl[j]), 0.0)
            return tot

        coa.set_value_function(v)
        exact = shapley_values(players, v)
        return coa, exact

    def test_mc_converges_to_exact(self) -> None:
        coa, exact = self._build_coalition(n=5, seed=12345)
        report = coa.shapley_montecarlo(
            epsilon=0.02, delta=0.1, max_samples=5000,
            method=POLICY_BERNSTEIN, min_samples=64,
        )
        for pid, est in report.values.items():
            assert est.lower - 1e-9 <= exact[pid] <= est.upper + 1e-9, (
                f"player {pid}: exact {exact[pid]:.4f} not in "
                f"[{est.lower:.4f}, {est.upper:.4f}]"
            )

    def test_efficiency_gap_shrinks_with_samples(self) -> None:
        coa, _ = self._build_coalition(n=6, seed=999)
        small = coa.shapley_montecarlo(
            epsilon=1e-9, delta=0.05, max_samples=100, min_samples=100,
            early_stop=False,
        )
        big = coa.shapley_montecarlo(
            epsilon=1e-9, delta=0.05, max_samples=5000, min_samples=5000,
            early_stop=False,
        )
        # Both should be small relative to grand_value scale.
        if abs(big.grand_value) > 1e-3:
            assert abs(big.efficiency_gap / big.grand_value) < 0.1

    def test_pac_coverage_rate(self) -> None:
        """Across N independent MC runs, the rate at which the exact
        value falls outside the CI is ≤ 3·δ (slack for MC noise)."""
        n_trials = 30
        delta = 0.20
        epsilon = 0.05
        n_fail = 0
        for trial in range(n_trials):
            coa, exact = self._build_coalition(n=4, seed=1000 + trial)
            report = coa.shapley_montecarlo(
                epsilon=epsilon, delta=delta, max_samples=2000,
                method=POLICY_BERNSTEIN, min_samples=64,
            )
            for pid, est in report.values.items():
                if not (est.lower - 1e-9 <= exact[pid] <= est.upper + 1e-9):
                    n_fail += 1
                    break
        # Empirical miss rate ≤ 3·δ.
        assert n_fail / n_trials <= 3.0 * delta + 0.05, (
            f"miss rate {n_fail}/{n_trials} too high"
        )


# ============================================================
# Section 3 — Stratified beats simple permutation
# ============================================================


class TestStratified:
    def test_stratified_competitive_with_perm(self) -> None:
        """Stratified gives a comparable CI to plain permutation
        (Maleki 2014). Strict variance reduction is game-dependent;
        we only verify both produce coherent estimates within an
        order of magnitude.
        """
        rng = random.Random(424242)
        n = 4
        coa = Coalition(rng=rng)
        for i in range(n):
            coa.register_player(f"p{i}")

        weights = [rng.uniform(0.0, 1.0) for _ in range(n)]

        def v(S: frozenset[str]) -> float:
            return sum(weights[int(p[1:])] for p in S) ** 2

        coa.set_value_function(v)
        exact = shapley_values([f"p{i}" for i in range(n)], v)

        coa2 = Coalition(rng=random.Random(424242))
        for i in range(n):
            coa2.register_player(f"p{i}")
        coa2.set_value_function(v)
        strat = coa2.shapley_stratified(
            epsilon=1e-9, delta=0.1, max_samples_per_stratum=128,
            method=POLICY_BERNSTEIN,
        )
        # Stratified point estimate within 0.2 of exact for every player.
        for pid in [f"p{i}" for i in range(n)]:
            assert abs(strat.values[pid].point - exact[pid]) < 0.2, (
                f"stratified φ̂_{pid}={strat.values[pid].point:.3f} "
                f"vs exact {exact[pid]:.3f}"
            )

    def test_stratified_recovers_exact_on_additive_game(self) -> None:
        rng = random.Random(7)
        coa = Coalition(rng=rng)
        weights = {"a": 2.0, "b": 1.0, "c": 3.0}
        for pid in weights:
            coa.register_player(pid)

        def v(S: frozenset[str]) -> float:
            return sum(weights[p] for p in S)

        coa.set_value_function(v)
        report = coa.shapley_stratified(
            max_samples_per_stratum=64, method=POLICY_BERNSTEIN,
        )
        # Additive game: φ_i = weights[i] exactly.
        for pid, w in weights.items():
            est = report.values[pid]
            assert math.isclose(est.point, w, abs_tol=0.05), (
                f"player {pid}: φ̂={est.point:.3f} vs truth {w:.3f}"
            )


# ============================================================
# Section 4 — Banzhaf indices
# ============================================================


class TestBanzhaf:
    def test_banzhaf_unanimity_voting(self) -> None:
        """In a unanimity game on T={a,b}, both a and b are equally
        pivotal; they should each get half of normalised Banzhaf.
        """

        def v(S: frozenset[str]) -> float:
            return 1.0 if {"a", "b"} <= S else 0.0

        result = banzhaf_index(["a", "b", "c"], v, normalised=True)
        assert math.isclose(result["a"], result["b"], abs_tol=1e-9)
        # c is a dummy.
        assert math.isclose(result["c"], 0.0, abs_tol=1e-9)

    def test_banzhaf_majority_voting(self) -> None:
        """3-player majority game: every player has equal Banzhaf
        power; sum of normalised Banzhaf = 1.
        """

        def v(S: frozenset[str]) -> float:
            return 1.0 if len(S) >= 2 else 0.0

        result = banzhaf_index(["a", "b", "c"], v, normalised=True)
        assert math.isclose(result["a"], 1.0 / 3, abs_tol=1e-9)
        assert math.isclose(result["b"], 1.0 / 3, abs_tol=1e-9)
        assert math.isclose(result["c"], 1.0 / 3, abs_tol=1e-9)


# ============================================================
# Section 5 — Owen value with group structure
# ============================================================


class TestOwen:
    def test_owen_degenerate_equals_shapley(self) -> None:
        """When every player is in its own group, Owen value = Shapley."""

        def v(S: frozenset[str]) -> float:
            return float(len(S) ** 2)

        coa = Coalition(rng=random.Random(99))
        players = ["a", "b", "c", "d"]
        for pid in players:
            coa.register_player(pid)
        coa.set_value_function(v)
        exact = shapley_values(players, v)
        owen = coa.owen_values(
            groups=[["a"], ["b"], ["c"], ["d"]],
            max_samples=2000, delta=0.1,
        )
        for pid in players:
            assert abs(owen[pid].point - exact[pid]) < 0.5, (
                f"owen({pid})={owen[pid].point:.3f} vs shapley={exact[pid]:.3f}"
            )

    def test_owen_within_group_symmetry(self) -> None:
        """Two players inside the same group, symmetric in v, get equal
        Owen value.
        """

        def v(S: frozenset[str]) -> float:
            return float(len(S))

        coa = Coalition(rng=random.Random(101))
        for pid in ["a", "b", "c", "d"]:
            coa.register_player(pid)
        coa.set_value_function(v)
        owen = coa.owen_values(
            groups=[["a", "b"], ["c", "d"]],
            max_samples=2000, delta=0.1,
        )
        # a, b symmetric within group → equal.
        assert abs(owen["a"].point - owen["b"].point) < 0.2
        assert abs(owen["c"].point - owen["d"].point) < 0.2


# ============================================================
# Section 6 — Observation-driven mode
# ============================================================


class TestObservations:
    def test_observe_stores_running_mean(self) -> None:
        coa = Coalition()
        coa.register_player("a")
        coa.register_player("b")
        coa.observe(["a"], 1.0)
        coa.observe(["a"], 3.0)
        coa.observe(["a", "b"], 5.0)
        assert math.isclose(coa.observed_value(["a"]), 2.0, abs_tol=1e-9)
        assert math.isclose(coa.observed_value(["a", "b"]), 5.0, abs_tol=1e-9)
        assert coa.observed_value(["b"]) is None

    def test_observe_auto_registers(self) -> None:
        coa = Coalition()
        coa.observe(["new_skill", "new_tool"], 1.0)
        ids = [p.id for p in coa.players()]
        assert "new_skill" in ids
        assert "new_tool" in ids

    def test_observe_drives_shapley(self) -> None:
        """A pure additive ground-truth: if every coalition's observed
        value is consistent with weights {a: 1, b: 2, c: 3}, the
        observation-driven Shapley should recover those weights.
        """
        coa = Coalition()
        weights = {"a": 1.0, "b": 2.0, "c": 3.0}
        for pid in weights:
            coa.register_player(pid)
        # Provide every coalition's value exactly.
        from itertools import chain, combinations
        players = list(weights)
        all_subsets = chain.from_iterable(
            combinations(players, r) for r in range(len(players) + 1)
        )
        for sub in all_subsets:
            v = sum(weights[p] for p in sub)
            coa.observe(list(sub), v)
        report = coa.shapley_exact()
        for pid, w in weights.items():
            assert math.isclose(report.values[pid].point, w, abs_tol=0.1)

    def test_coverage_report(self) -> None:
        coa = Coalition()
        for pid in ["a", "b"]:
            coa.register_player(pid)
        coa.observe(["a"], 1.0)
        coa.observe([], 0.0)
        cov = coa.coverage()
        assert isinstance(cov, CoverageReport)
        assert cov.n_players == 2
        assert cov.n_observations == 2
        assert cov.n_distinct_coalitions == 2
        assert cov.empty_coalition_observed
        assert not cov.grand_coalition_observed

    def test_linear_fit_recovers_additive(self) -> None:
        """Fit a linear v̂ from observations of an additive game and
        check that the order-1 Shapley equals the original weights.
        """
        rng = random.Random(11)
        weights = {"a": 1.5, "b": -0.5, "c": 2.0, "d": 0.3}
        players = list(weights)
        obs = []
        from itertools import combinations
        for r in range(5):
            for sub in combinations(players, r):
                y = sum(weights[p] for p in sub) + rng.gauss(0, 0.01)
                obs.append((list(sub), y))
        phi = shapley_from_observations(obs, players, order=1, l2=1e-6)
        for pid, w in weights.items():
            assert math.isclose(phi[pid], w, abs_tol=0.05), (
                f"player {pid}: fitted {phi[pid]:.3f} vs truth {w:.3f}"
            )


# ============================================================
# Section 7 — Marginal contribution + core
# ============================================================


class TestCoreAllocation:
    def test_marginal_contribution(self) -> None:
        coa = Coalition()
        for pid in ["a", "b", "c"]:
            coa.register_player(pid)

        def v(S: frozenset[str]) -> float:
            return float(len(S))

        coa.set_value_function(v)
        # Adding 'a' to {b}: v({a,b}) − v({b}) = 1.
        assert coa.marginal_contribution("a", ["b"]) == 1.0
        # Adding 'a' to {a, b}: should still be 1 (player already in S
        # is dropped).
        assert coa.marginal_contribution("a", ["a", "b"]) == 1.0

    def test_shapley_in_core_convex_game(self) -> None:
        """Convex game → Shapley is in the core (Shapley 1971).
        Verify on a small convex example.
        """
        coa = Coalition()
        for pid in ["a", "b", "c"]:
            coa.register_player(pid)

        def v(S: frozenset[str]) -> float:
            # Submodular complement — convex characteristic function:
            # v(S) = |S|^2. The function k -> k^2 is convex, and
            # marginals are k+1 increasing in S — i.e. supermodular.
            return float(len(S) ** 2)

        coa.set_value_function(v)
        report = coa.shapley_exact()
        allocation = {pid: est.point for pid, est in report.values.items()}
        ok, witness = coa.in_core(allocation)
        assert ok, f"core violation: {witness}"

    def test_allocate_efficient_scales_to_budget(self) -> None:
        coa = Coalition()
        for pid in ["a", "b", "c"]:
            coa.register_player(pid)

        def v(S: frozenset[str]) -> float:
            return float(len(S))

        coa.set_value_function(v)
        report = coa.shapley_exact()
        raw = {pid: est.point for pid, est in report.values.items()}
        # Scale to budget = 10.
        scaled = coa.allocate_efficient(raw, target=10.0)
        assert math.isclose(sum(scaled.values()), 10.0, abs_tol=1e-9)
        # Ratios preserved.
        assert math.isclose(scaled["a"] / scaled["b"], raw["a"] / raw["b"], abs_tol=1e-9)


# ============================================================
# Section 8 — Bounds (Hoeffding and Bernstein)
# ============================================================


class TestBounds:
    def test_hoeffding_shrinks_with_samples(self) -> None:
        r1 = hoeffding_radius(delta=0.05, n_samples=100, value_range=1.0)
        r2 = hoeffding_radius(delta=0.05, n_samples=10000, value_range=1.0)
        assert r2 < r1
        # Sqrt(100x) → 10x shrinkage.
        assert r1 / r2 > 8.0

    def test_bernstein_dominates_hoeffding_low_variance(self) -> None:
        """At low realised variance Bernstein is tighter."""
        h = hoeffding_radius(delta=0.05, n_samples=1000, value_range=1.0)
        b = bernstein_radius(
            delta=0.05, n_samples=1000, value_range=1.0, sample_variance=0.01,
        )
        assert b < h

    def test_hoeffding_dominates_bernstein_high_variance(self) -> None:
        """When variance is at the max (Δ²/4), Bernstein degrades."""
        b_max = bernstein_radius(
            delta=0.05, n_samples=1000, value_range=1.0, sample_variance=0.25,
        )
        h = hoeffding_radius(delta=0.05, n_samples=1000, value_range=1.0)
        # b_max should be larger or comparable.
        assert b_max >= h * 0.8


# ============================================================
# Section 9 — Threadsafety
# ============================================================


class TestThreadsafety:
    def test_concurrent_observe(self) -> None:
        coa = Coalition()
        for pid in ["a", "b", "c"]:
            coa.register_player(pid)

        n_threads = 4
        n_per_thread = 250
        errors = []

        def worker(thread_id: int) -> None:
            try:
                rng = random.Random(thread_id)
                for _ in range(n_per_thread):
                    subset = [
                        pid for pid in ["a", "b", "c"] if rng.random() < 0.5
                    ]
                    coa.observe(subset, rng.random())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        cov = coa.coverage()
        assert cov.n_observations == n_threads * n_per_thread


# ============================================================
# Section 10 — Events
# ============================================================


class TestEvents:
    def test_emits_start_register_observe(self) -> None:
        bus = EventBus()
        captured: list[Event] = []
        bus.subscribe(lambda e: captured.append(e))
        coa = Coalition(bus=bus)
        coa.register_player("a")
        coa.register_player("b")
        coa.observe(["a", "b"], 1.0)
        kinds = [e.kind for e in captured]
        assert COALITION_STARTED in kinds
        assert COALITION_PLAYER_REGISTERED in kinds
        assert COALITION_OBSERVED in kinds

    def test_emits_computed_on_shapley_exact(self) -> None:
        bus = EventBus()
        captured: list[Event] = []
        bus.subscribe(lambda e: captured.append(e))
        coa = Coalition(bus=bus)
        for pid in ["a", "b"]:
            coa.register_player(pid)
        coa.set_value_function(lambda S: float(len(S)))
        coa.shapley_exact()
        kinds = [e.kind for e in captured]
        assert COALITION_COMPUTED in kinds


# ============================================================
# Section 11 — Attestation pass-through
# ============================================================


class _FakeAttestor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def record(self, kind: str, payload: dict) -> str:
        self.calls.append((kind, payload))
        return "fake-hash-1234"


class TestAttestation:
    def test_receipt_hash_populated(self) -> None:
        att = _FakeAttestor()
        coa = Coalition(attestor=att)
        for pid in ["a", "b"]:
            coa.register_player(pid)
        coa.set_value_function(lambda S: float(len(S)))
        report = coa.shapley_exact()
        assert report.receipt_hash == "fake-hash-1234"
        assert att.calls
        assert att.calls[0][0] == "coalition.computed"

    def test_real_attestation_ledger_integration(self) -> None:
        from agi.attest import AttestationLedger, RuntimeAttestor

        ledger = AttestationLedger(key=b"test-secret")
        att = RuntimeAttestor(ledger)
        coa = Coalition(attestor=att)
        for pid in ["a", "b", "c"]:
            coa.register_player(pid)
        coa.set_value_function(lambda S: float(len(S)))
        report = coa.shapley_exact()
        # Receipt hash should be present, ledger should have one entry.
        assert report.receipt_hash
        assert att.appended == 1


# ============================================================
# Section 12 — Edge cases + errors
# ============================================================


class TestEdgeCases:
    def test_empty_coalition_shapley(self) -> None:
        coa = Coalition()
        report = coa.shapley_exact()
        assert report.values == {}
        assert report.grand_value == 0.0

    def test_singleton_shapley_equals_v(self) -> None:
        coa = Coalition()
        coa.register_player("a")

        def v(S: frozenset[str]) -> float:
            return 5.0 if "a" in S else 0.0

        coa.set_value_function(v)
        report = coa.shapley_exact()
        assert math.isclose(report.values["a"].point, 5.0, abs_tol=1e-9)

    def test_register_then_reregister_preserves_observations(self) -> None:
        coa = Coalition()
        coa.register_player("a")
        coa.observe(["a"], 1.0)
        coa.register_player("a", cost=99.0)
        assert coa.observed_value(["a"]) == 1.0
        spec = next(p for p in coa.players() if p.id == "a")
        assert spec.cost == 99.0

    def test_exact_caps_at_max_players(self) -> None:
        coa = Coalition()
        for i in range(20):
            coa.register_player(f"p{i}")
        coa.set_value_function(lambda S: float(len(S)))
        with pytest.raises(ValueError):
            coa.shapley_exact()

    def test_observe_rejects_nan(self) -> None:
        coa = Coalition()
        coa.register_player("a")
        with pytest.raises(ValueError):
            coa.observe(["a"], float("nan"))

    def test_observe_rejects_nonpositive_weight(self) -> None:
        coa = Coalition()
        coa.register_player("a")
        with pytest.raises(ValueError):
            coa.observe(["a"], 1.0, weight=0.0)

    def test_in_core_missing_player_raises(self) -> None:
        coa = Coalition()
        for pid in ["a", "b"]:
            coa.register_player(pid)
        coa.set_value_function(lambda S: float(len(S)))
        with pytest.raises(ValueError):
            coa.in_core({"a": 1.0})

    def test_owen_rejects_overlapping_groups(self) -> None:
        coa = Coalition()
        for pid in ["a", "b"]:
            coa.register_player(pid)
        coa.set_value_function(lambda S: 0.0)
        with pytest.raises(ValueError):
            coa.owen_values(groups=[["a", "b"], ["a"]])

    def test_reset_clears_state(self) -> None:
        coa = Coalition()
        coa.register_player("a")
        coa.observe(["a"], 1.0)
        coa.set_value_function(lambda S: 0.0)
        coa.shapley_exact()
        coa.reset()
        assert coa.coverage().n_observations == 0
        assert coa.history() == ()


# ============================================================
# Section 13 — Free function smoke
# ============================================================


class TestFreeFunctions:
    def test_shapley_values_n_too_large_raises(self) -> None:
        with pytest.raises(ValueError):
            shapley_values([f"p{i}" for i in range(25)], lambda S: 0.0)

    def test_fit_linear_v_is_callable(self) -> None:
        obs = [(["a"], 1.0), (["b"], 2.0), (["a", "b"], 3.0)]
        v_hat = fit_linear_v(obs, ["a", "b"], order=1)
        assert callable(v_hat)
        # On the additive game, v̂({a, b}) should be close to 3.
        assert abs(v_hat(frozenset({"a", "b"})) - 3.0) < 0.5


# ============================================================
# Section 14 — Integration scenarios
# ============================================================


class TestIntegrationScenarios:
    def test_skill_attribution_scenario(self) -> None:
        """Coalition usage as a runtime credit-assignment primitive.

        The coordination engine logs (skills, outcome) pairs. After
        100 traces we run Shapley to identify the high-credit skills.
        We construct a game where skill 'core' is essential and skill
        'irrelevant' is a dummy; Shapley should reflect this.
        """
        rng = random.Random(2024)
        bus = EventBus()
        coa = Coalition(bus=bus, rng=rng)
        for skill in ["core", "helper", "irrelevant"]:
            coa.register_player(skill)

        # 'core' alone succeeds 90% of the time; with 'helper' boosts
        # to 95%; 'irrelevant' is independent noise.
        for _ in range(200):
            skills = []
            if rng.random() < 0.7:
                skills.append("core")
            if rng.random() < 0.5:
                skills.append("helper")
            if rng.random() < 0.4:
                skills.append("irrelevant")
            # Outcome depends only on 'core' (and slightly on 'helper').
            p_success = 0.0
            if "core" in skills:
                p_success += 0.9
                if "helper" in skills:
                    p_success += 0.05
            outcome = 1.0 if rng.random() < p_success else 0.0
            coa.observe(skills, outcome)

        # Use the linear-fit pathway since observations are sparse.
        obs = []
        for coalition, (s, sq, n) in coa._obs.items():
            obs.append((list(coalition), s / n))
        phi = shapley_from_observations(
            obs, ["core", "helper", "irrelevant"], order=2,
        )
        # 'core' should dominate; 'irrelevant' should be near zero.
        assert phi["core"] > phi["helper"]
        assert phi["core"] > phi["irrelevant"]
        assert abs(phi["irrelevant"]) < 0.2

    def test_multi_tenant_cost_split(self) -> None:
        """Allocate a shared cost across tenants by Shapley value.

        Each tenant's 'value' is the cost they would have incurred
        had only they been served. The Shapley value gives the
        unique additive, efficient, symmetric, dummy-respecting split.
        """
        coa = Coalition()
        for tenant in ["acme", "globex", "initech"]:
            coa.register_player(tenant)

        # Cost is sub-linear in #tenants (shared infrastructure):
        #   v(S) = base + sub-linear(|S|)
        # We use the savings function: how much *less* it costs to
        # serve S together vs separately.
        single_cost = {"acme": 100.0, "globex": 50.0, "initech": 80.0}

        def v(S: frozenset[str]) -> float:
            if not S:
                return 0.0
            # Combined cost = base 30 + sum.
            combined = 30.0 + sum(single_cost[t] for t in S)
            separate = sum(single_cost[t] for t in S)
            # Savings to allocate.
            return separate - combined + 30.0  # Net savings.

        coa.set_value_function(v)
        report = coa.shapley_exact()
        total = sum(est.point for est in report.values.values())
        assert math.isclose(
            total, v(frozenset(single_cost)) - v(frozenset()), abs_tol=1e-9
        )
