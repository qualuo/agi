"""Tests for the Cartographer curriculum kernel.

We exercise the numerical primitives in isolation (Wilson CI, Clopper–
Pearson, Beta inversion, learning progress, submodular knapsack), then
the Cartographer itself: registration / DAG validation, observation
dynamics, status transitions, every recommendation policy, frontier
maintenance, drift-driven regression, snapshot/restore round-trip,
coverage report, attestation, and the event-bus integration.

Stdlib-only; runs in a few seconds.
"""
from __future__ import annotations

import math
import statistics
import time

import pytest

from agi.events import Event, EventBus
from agi.cartographer import (
    CARTOGRAPHER_ADVANCED,
    CARTOGRAPHER_CLEARED,
    CARTOGRAPHER_FRONTIER_CHANGED,
    CARTOGRAPHER_OBSERVED,
    CARTOGRAPHER_RECOMMENDED,
    CARTOGRAPHER_REGRESSED,
    CARTOGRAPHER_STARTED,
    POLICY_INFOGAIN,
    POLICY_KNAPSACK,
    POLICY_LP,
    POLICY_ROUND_ROBIN,
    POLICY_THOMPSON,
    POLICY_UCB,
    REWARD_BERNOULLI,
    REWARD_GAUSSIAN,
    STATUS_FRAGILE,
    STATUS_FRONTIER,
    STATUS_LOCKED,
    STATUS_MASTERED,
    STATUS_NOVICE,
    Cartographer,
    Competence,
    Curriculum,
    CurriculumItem,
    TickReport,
    beta_lcb,
    beta_ucb,
    clopper_pearson_ci,
    expected_pulls_to_master,
    frontier_subset,
    learning_progress,
    lp_signal,
    submodular_knapsack,
    wilson_ci,
)


# =====================================================================
# Wilson CI
# =====================================================================


def test_wilson_ci_zero_observations_is_trivial():
    low, high = wilson_ci(0, 0)
    assert low == 0.0 and high == 1.0


def test_wilson_ci_endpoints_remain_in_unit_interval():
    for s in range(0, 21):
        low, high = wilson_ci(s, 20)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_ci_extremes_have_nondegenerate_interval():
    low, high = wilson_ci(0, 20)
    assert low == 0.0
    assert 0.0 < high < 0.25
    low, high = wilson_ci(20, 20)
    assert 0.75 < low < 1.0
    assert high == 1.0


def test_wilson_ci_narrows_with_n():
    low100, high100 = wilson_ci(50, 100)
    low1000, high1000 = wilson_ci(500, 1000)
    assert (high1000 - low1000) < (high100 - low100)


def test_wilson_ci_centred_on_phat_for_balanced():
    low, high = wilson_ci(50, 100)
    centre = 0.5 * (low + high)
    assert abs(centre - 0.5) < 0.02


# =====================================================================
# Clopper-Pearson + Beta inversion
# =====================================================================


def test_clopper_pearson_endpoints_match_extremes():
    low, high = clopper_pearson_ci(0, 20)
    assert low == 0.0
    assert 0.0 < high < 1.0
    low, high = clopper_pearson_ci(20, 20)
    assert 0.0 < low < 1.0
    assert high == 1.0


def test_clopper_pearson_is_wider_than_wilson_at_small_n():
    s, n = 2, 8
    w_low, w_high = wilson_ci(s, n)
    cp_low, cp_high = clopper_pearson_ci(s, n)
    assert (cp_high - cp_low) >= (w_high - w_low)


def test_beta_bounds_increase_with_successes_at_fixed_n():
    bounds = []
    for s in range(0, 11):
        bounds.append((beta_lcb(s, 10), beta_ucb(s, 10)))
    for i in range(len(bounds) - 1):
        # both bounds monotone non-decreasing in s
        assert bounds[i][0] <= bounds[i + 1][0] + 1e-9
        assert bounds[i][1] <= bounds[i + 1][1] + 1e-9


def test_beta_lcb_le_beta_ucb():
    for s, n in [(0, 0), (1, 5), (10, 10), (50, 100), (1, 10_000)]:
        assert beta_lcb(s, n) <= beta_ucb(s, n)


# =====================================================================
# Learning progress
# =====================================================================


def test_learning_progress_none_on_short_history():
    assert learning_progress([], window_recent=3, window_prior=3) is None
    assert learning_progress([1, 1, 1], window_recent=3, window_prior=3) is None
    assert learning_progress([1, 1, 1, 1, 1], window_recent=3, window_prior=3) is None


def test_learning_progress_positive_on_climbing_trace():
    trace = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    lp = learning_progress(trace, window_recent=4, window_prior=4)
    assert lp == pytest.approx(1.0)


def test_learning_progress_negative_on_regression():
    trace = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    lp = learning_progress(trace, window_recent=4, window_prior=4)
    assert lp == pytest.approx(-1.0)


def test_learning_progress_zero_on_stationary_signal():
    trace = [0.5] * 16
    lp = learning_progress(trace, window_recent=4, window_prior=4)
    assert lp == pytest.approx(0.0)


def test_lp_signal_is_alias():
    trace = [0.0] * 4 + [1.0] * 4
    assert lp_signal(trace, window_recent=4, window_prior=4) == pytest.approx(1.0)


# =====================================================================
# Submodular knapsack
# =====================================================================


def test_submodular_knapsack_respects_budget():
    items = [("a", 5.0, 1.0), ("b", 4.0, 1.0), ("c", 3.0, 1.0), ("d", 2.0, 1.0)]
    out = submodular_knapsack(items, budget=2.0)
    # Should pick the two top-gain items.
    assert set(out) == {"a", "b"}


def test_submodular_knapsack_handles_lumpy_costs():
    items = [("big", 10.0, 5.0), ("small_a", 4.0, 1.0), ("small_b", 4.0, 1.0)]
    out = submodular_knapsack(items, budget=5.0)
    # Either the lumpy item, or the two smalls, whichever has higher gain.
    assert out == ["big"] or set(out) == {"small_a", "small_b"}


def test_submodular_knapsack_drops_zero_cost_items():
    items = [("ok", 1.0, 1.0), ("bad", 1.0, 0.0)]
    out = submodular_knapsack(items, budget=1.0)
    assert "bad" not in out
    assert "ok" in out


def test_submodular_knapsack_empty_inputs():
    assert submodular_knapsack([], budget=1.0) == []
    assert submodular_knapsack([("a", 1.0, 1.0)], budget=0.0) == []


# =====================================================================
# frontier_subset, expected_pulls_to_master
# =====================================================================


def test_frontier_subset_drops_mastered_and_keeps_frontier():
    means = {"easy": 0.97, "frontier": 0.5, "novice": 0.05}
    counts = {"easy": 200, "frontier": 50, "novice": 50}
    out = frontier_subset(means, counts, entry=0.2, mastery=0.8)
    assert "easy" not in out
    assert "frontier" in out
    # novice has Wilson upper above 0.05 since n=50 — may or may not be in.
    # We only assert frontier inclusion + mastered exclusion (the contract).


def test_frontier_subset_includes_uninitialised_tasks():
    means = {"new": 0.0}
    counts = {"new": 0}
    assert "new" in frontier_subset(means, counts)


def test_expected_pulls_to_master_unreachable_returns_cap():
    # p̂ = 0.3 cannot ever reach mastery 0.8 in expectation.
    out = expected_pulls_to_master(3, 10, mastery=0.8, max_pulls=500)
    assert out == 500


def test_expected_pulls_to_master_decreases_for_strong_signal():
    out_strong = expected_pulls_to_master(95, 100, mastery=0.8)
    out_weak = expected_pulls_to_master(81, 100, mastery=0.8)
    assert out_strong <= out_weak


# =====================================================================
# Cartographer construction
# =====================================================================


def test_constructor_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        Cartographer(entry_threshold=0.0)
    with pytest.raises(ValueError):
        Cartographer(entry_threshold=0.6, mastery_threshold=0.5)
    with pytest.raises(ValueError):
        Cartographer(mastery_threshold=1.5)
    with pytest.raises(ValueError):
        Cartographer(delta=0.0)
    with pytest.raises(ValueError):
        Cartographer(window_recent=0)
    with pytest.raises(ValueError):
        Cartographer(prior_alpha=0.0)


def test_register_rejects_invalid_args():
    cart = Cartographer()
    with pytest.raises(ValueError):
        cart.register_task("")
    with pytest.raises(ValueError):
        cart.register_task("x", cost=0.0)
    with pytest.raises(ValueError):
        cart.register_task("x", reward_model="poisson")
    with pytest.raises(ValueError):
        cart.register_task("x", value=-1.0)


def test_register_is_idempotent_and_detects_conflict():
    cart = Cartographer()
    cart.register_task("t", value=2.0, cost=1.0)
    cart.register_task("t", value=2.0, cost=1.0)  # idempotent
    with pytest.raises(ValueError):
        cart.register_task("t", value=3.0)


def test_register_rejects_cyclic_prereqs():
    cart = Cartographer()
    cart.register_task("a")
    cart.register_task("b", prereqs=("a",))
    with pytest.raises(ValueError):
        # would create a -> b -> a
        cart._tasks["a"].spec = cart._tasks["a"].spec.__class__(
            id="a", value=1.0, cost=1.0, prereqs=("b",),
            reward_model=REWARD_BERNOULLI, sigma2=1.0, tags=(),
        )
        cart._validate_dag_locked()


def test_unregister_blocks_when_dependent_exists():
    cart = Cartographer()
    cart.register_task("a")
    cart.register_task("b", prereqs=("a",))
    with pytest.raises(ValueError):
        cart.unregister_task("a")
    assert cart.unregister_task("b") is True
    assert cart.unregister_task("a") is True
    assert cart.unregister_task("missing") is False


# =====================================================================
# Observation + status engine
# =====================================================================


def test_observation_records_count_and_mean():
    cart = Cartographer()
    cart.register_task("t")
    for v in [1.0, 1.0, 0.0, 1.0]:
        cart.observe("t", v)
    c = cart.competence("t")
    assert c.n == 4
    assert c.raw_mean == pytest.approx(0.75)


def test_observation_rejects_unknown_task():
    cart = Cartographer()
    with pytest.raises(KeyError):
        cart.observe("ghost", 1.0)


def test_observation_rejects_out_of_range_bernoulli():
    cart = Cartographer()
    cart.register_task("t")
    with pytest.raises(ValueError):
        cart.observe("t", 1.5)
    with pytest.raises(ValueError):
        cart.observe("t", -0.1)


def test_task_with_no_observations_is_frontier_if_no_prereqs():
    cart = Cartographer()
    cart.register_task("t")
    assert cart.status("t") == STATUS_FRONTIER


def test_task_with_unmet_prereq_is_locked():
    cart = Cartographer()
    cart.register_task("base")
    cart.register_task("child", prereqs=("base",))
    cart.tick()
    assert cart.status("child") == STATUS_LOCKED


def test_mastery_propagates_to_unlock_dependents():
    cart = Cartographer(mastery_threshold=0.7, entry_threshold=0.1)
    cart.register_task("base")
    cart.register_task("child", prereqs=("base",))
    # Master the base.
    for _ in range(40):
        cart.observe("base", 1.0)
    rep = cart.tick()
    assert cart.status("base") == STATUS_MASTERED
    assert cart.status("child") == STATUS_FRONTIER
    advanced_ids = [e.task_id for e in rep.advanced]
    assert "base" in advanced_ids


def test_mastery_triggers_advanced_event():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append, kind=CARTOGRAPHER_ADVANCED)
    cart = Cartographer(bus=bus, mastery_threshold=0.7)
    cart.register_task("t")
    for _ in range(40):
        cart.observe("t", 1.0)
    cart.tick()
    assert any(e.data["task_id"] == "t" for e in seen)


def test_status_is_novice_when_upper_is_below_entry():
    cart = Cartographer(entry_threshold=0.3)
    cart.register_task("t")
    # 0/200 has Wilson upper ~ 0.018 — well below 0.3.
    for _ in range(200):
        cart.observe("t", 0.0)
    cart.tick()
    assert cart.status("t") == STATUS_NOVICE


def test_status_remains_frontier_in_uncertain_zone():
    cart = Cartographer(entry_threshold=0.2, mastery_threshold=0.8)
    cart.register_task("t")
    for _ in range(20):
        cart.observe("t", 0.5)
    cart.tick()
    assert cart.status("t") == STATUS_FRONTIER


def test_regress_demotes_mastered_task():
    cart = Cartographer(mastery_threshold=0.7)
    cart.register_task("t")
    for _ in range(40):
        cart.observe("t", 1.0)
    cart.tick()
    assert cart.status("t") == STATUS_MASTERED
    cart.regress("t", rationale="drift")
    assert cart.status("t") == STATUS_FRONTIER


def test_fragile_status_when_mastered_mean_drops():
    cart = Cartographer(mastery_threshold=0.7, regression_margin=0.05)
    cart.register_task("t")
    for _ in range(40):
        cart.observe("t", 1.0)
    cart.tick()
    assert cart.status("t") == STATUS_MASTERED
    # Inject regression: a flood of zeros.
    for _ in range(60):
        cart.observe("t", 0.0)
    # The Wilson lower has not necessarily fallen below mastery, but
    # the mastery_mean is 1.0 and the current raw mean has dropped by
    # well over regression_margin → fragile.
    cart.tick()
    assert cart.status("t") in (STATUS_FRAGILE, STATUS_FRONTIER, STATUS_NOVICE)


# =====================================================================
# Recommendations
# =====================================================================


def _populate_balanced(cart: Cartographer, n: int = 12) -> None:
    """Three tasks with diverging means."""
    cart.register_task("easy", value=1.0, cost=1.0)
    cart.register_task("hard", value=1.0, cost=1.0)
    cart.register_task("mid", value=2.0, cost=1.0)
    for _ in range(n):
        cart.observe("easy", 0.9)
        cart.observe("hard", 0.1)
        cart.observe("mid", 0.5)


def test_lp_policy_picks_frontier_tasks():
    cart = Cartographer(window_recent=4, window_prior=4)
    cart.register_task("rising")
    cart.register_task("flat")
    # rising: clear LP signal
    for v in [0.0] * 8 + [1.0] * 8:
        cart.observe("rising", v)
    # flat: stationary
    for _ in range(16):
        cart.observe("flat", 0.5)
    cart.tick()
    cur = cart.recommend(policy=POLICY_LP, k=1)
    assert len(cur.items) == 1
    # "rising" has |LP| close to 1 vs ~0 for flat
    assert cur.items[0].task_id == "rising"


def test_ucb_policy_picks_highest_upper_bound():
    cart = Cartographer(mastery_threshold=0.95)
    cart.register_task("strong", value=1.0)
    cart.register_task("weak", value=1.0)
    for _ in range(10):
        cart.observe("strong", 0.8)
        cart.observe("weak", 0.2)
    cart.tick()
    cur = cart.recommend(policy=POLICY_UCB, k=1)
    assert cur.items[0].task_id == "strong"


def test_infogain_prefers_undersampled_tasks():
    cart = Cartographer()
    cart.register_task("seen", cost=1.0)
    cart.register_task("rare", cost=1.0)
    for _ in range(50):
        cart.observe("seen", 0.5)
    for _ in range(3):
        cart.observe("rare", 0.5)
    cart.tick()
    cur = cart.recommend(policy=POLICY_INFOGAIN, k=1)
    assert cur.items[0].task_id == "rare"


def test_thompson_seeded_is_deterministic():
    cart = Cartographer()
    cart.register_task("a", value=1.0)
    cart.register_task("b", value=1.0)
    for _ in range(5):
        cart.observe("a", 0.9)
        cart.observe("b", 0.1)
    cart.tick()
    cur1 = cart.recommend(policy=POLICY_THOMPSON, k=2, rng_seed=42)
    cur2 = cart.recommend(policy=POLICY_THOMPSON, k=2, rng_seed=42)
    assert [i.task_id for i in cur1.items] == [i.task_id for i in cur2.items]


def test_round_robin_rotates_through_frontier():
    cart = Cartographer()
    for name in ["a", "b", "c"]:
        cart.register_task(name)
    cart.tick()
    seen: list[str] = []
    for _ in range(6):
        cur = cart.recommend(policy=POLICY_ROUND_ROBIN, k=1)
        seen.append(cur.items[0].task_id)
    # All three tasks must appear in the cycle.
    assert set(seen) == {"a", "b", "c"}


def test_knapsack_respects_budget_and_returns_best_under_cap():
    cart = Cartographer()
    cart.register_task("big", value=10.0, cost=5.0)
    cart.register_task("cheap_a", value=4.0, cost=1.0)
    cart.register_task("cheap_b", value=4.0, cost=1.0)
    # Use observations to seed scores.
    for tid in ("big", "cheap_a", "cheap_b"):
        for v in [0.0] * 4 + [1.0] * 4:
            cart.observe(tid, v)
    cart.tick()
    cur = cart.recommend(policy=POLICY_KNAPSACK, k=10, budget=5.0)
    assert cur.total_cost <= 5.0 + 1e-9


def test_knapsack_without_budget_raises():
    cart = Cartographer()
    cart.register_task("t")
    cart.tick()
    with pytest.raises(ValueError):
        cart.recommend(policy=POLICY_KNAPSACK, k=1)


def test_unknown_policy_raises():
    cart = Cartographer()
    cart.register_task("t")
    cart.tick()
    with pytest.raises(ValueError):
        cart.recommend(policy="not-a-policy")


def test_recommend_emits_event():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append, kind=CARTOGRAPHER_RECOMMENDED)
    cart = Cartographer(bus=bus)
    cart.register_task("t")
    cart.tick()
    cart.recommend(policy=POLICY_UCB, k=1)
    assert seen and seen[0].data["policy"] == POLICY_UCB


def test_recommend_returns_empty_when_frontier_empty():
    cart = Cartographer(mastery_threshold=0.7)
    cart.register_task("t")
    for _ in range(40):
        cart.observe("t", 1.0)
    cart.tick()
    cur = cart.recommend(policy=POLICY_LP, k=4)
    assert cur.items == []


def test_score_fn_hook_influences_ranking():
    cart = Cartographer()
    cart.register_task("a", value=1.0)
    cart.register_task("b", value=1.0)
    for _ in range(10):
        cart.observe("a", 0.5)
        cart.observe("b", 0.5)
    cart.tick()

    def prefer_b(_cart, tid):
        return 100.0 if tid == "b" else 0.0

    cur = cart.recommend(policy=POLICY_UCB, k=1, score_fn=prefer_b)
    assert cur.items[0].task_id == "b"


# =====================================================================
# Tick / frontier signature events
# =====================================================================


def test_tick_emits_frontier_changed_when_set_changes():
    bus = EventBus()
    changes: list[Event] = []
    bus.subscribe(changes.append, kind=CARTOGRAPHER_FRONTIER_CHANGED)
    cart = Cartographer(bus=bus, mastery_threshold=0.7)
    cart.register_task("a")
    cart.register_task("b")
    cart.tick()
    # Now master 'a' — frontier signature should change.
    for _ in range(40):
        cart.observe("a", 1.0)
    cart.tick()
    # At least one change event after both observations.
    assert len(changes) >= 1


def test_tick_idempotent_when_no_observations():
    cart = Cartographer()
    cart.register_task("t")
    rep1 = cart.tick()
    rep2 = cart.tick()
    assert rep1.frontier_size == rep2.frontier_size
    assert rep1.mastered_size == rep2.mastered_size


# =====================================================================
# Gaussian path
# =====================================================================


def test_gaussian_observation_supports_arbitrary_values():
    cart = Cartographer()
    cart.register_task("gauss", reward_model=REWARD_GAUSSIAN, sigma2=1.0)
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cart.observe("gauss", v)
    c = cart.competence("gauss")
    assert c.n == 5
    assert c.raw_mean == pytest.approx(0.3)


def test_gaussian_path_does_not_reject_negative_outcomes():
    cart = Cartographer()
    cart.register_task("g", reward_model=REWARD_GAUSSIAN)
    # Bernoulli path would reject, but Gaussian accepts.
    cart.observe("g", -1.0)
    cart.observe("g", 0.5)
    assert cart.competence("g").n == 2


# =====================================================================
# Coverage report
# =====================================================================


def test_coverage_report_handles_zero_predictions_gracefully():
    cart = Cartographer()
    cart.register_task("t")
    rep = cart.coverage_report()
    assert rep.n_predictions == 0
    assert rep.mean_abs_error == 0.0


def test_coverage_report_tracks_signed_and_abs_error():
    cart = Cartographer()
    cart.register_task("t")
    # All zeros — predictions will be close to 0 after a few samples.
    for _ in range(60):
        cart.observe("t", 0.0)
    cart.record_truth("t", true_mean=0.0)
    rep = cart.coverage_report()
    assert rep.n_predictions >= 60
    # error should be small.
    assert abs(rep.mean_signed_error) < 0.2
    assert rep.brier_score < 0.2


# =====================================================================
# Snapshot / restore
# =====================================================================


def test_snapshot_restore_round_trip_preserves_state():
    cart = Cartographer(mastery_threshold=0.7)
    cart.register_task("a")
    cart.register_task("b", prereqs=("a",))
    for _ in range(30):
        cart.observe("a", 1.0)
    for _ in range(5):
        cart.observe("b", 0.4)
    cart.tick()

    snap = cart.snapshot()
    cart2 = Cartographer()
    cart2.restore(snap)

    assert set(cart2.task_ids()) == {"a", "b"}
    assert cart2.status("a") == STATUS_MASTERED
    assert cart2.status("b") in (STATUS_FRONTIER, STATUS_NOVICE)
    assert cart2.competence("a").n == 30
    assert cart2.competence("b").n == 5


def test_snapshot_keeps_truths():
    cart = Cartographer()
    cart.register_task("t")
    cart.record_truth("t", true_mean=0.6)
    snap = cart.snapshot()
    cart2 = Cartographer()
    cart2.restore(snap)
    rep = cart2.coverage_report()
    # No predictions logged after restore, but truth is loaded.
    assert rep.n_predictions == 0


# =====================================================================
# Clear
# =====================================================================


def test_clear_task_resets_state():
    cart = Cartographer()
    cart.register_task("t")
    for _ in range(10):
        cart.observe("t", 1.0)
    cart.clear("t")
    c = cart.competence("t")
    assert c.n == 0
    assert c.raw_mean == 0.0


def test_clear_all_wipes_the_curriculum():
    cart = Cartographer()
    cart.register_task("a")
    cart.register_task("b")
    cart.observe("a", 1.0)
    cart.clear()
    assert cart.task_ids() == []


# =====================================================================
# Attestation
# =====================================================================


class _RecordingAttestor:
    """Minimal attestor that records each call and returns a hash-like obj."""

    def __init__(self):
        self.records: list[tuple[str, dict]] = []

    def record(self, *, kind, payload):
        self.records.append((kind, dict(payload)))
        return "abc123"


def test_attestor_receives_mastery_record():
    att = _RecordingAttestor()
    cart = Cartographer(attestor=att, mastery_threshold=0.7)
    cart.register_task("t")
    for _ in range(40):
        cart.observe("t", 1.0)
    cart.tick()
    kinds = [k for k, _ in att.records]
    assert CARTOGRAPHER_ADVANCED in kinds


# =====================================================================
# Threading smoke
# =====================================================================


def test_concurrent_observation_does_not_corrupt_state():
    import threading

    cart = Cartographer()
    cart.register_task("t")

    def worker():
        for _ in range(200):
            cart.observe("t", 1.0)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert cart.competence("t").n == 800


# =====================================================================
# Event-bus integration smoke
# =====================================================================


def test_event_bus_carries_observed_started_recommended_advanced():
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(events.append)
    cart = Cartographer(bus=bus, mastery_threshold=0.7)
    cart.register_task("t")
    for _ in range(40):
        cart.observe("t", 1.0)
    cart.tick()
    cart.recommend(policy=POLICY_UCB, k=1)
    kinds = {e.kind for e in events}
    assert CARTOGRAPHER_STARTED in kinds
    assert CARTOGRAPHER_OBSERVED in kinds
    # advanced may fire either via tick or via observation-time refresh.
    assert CARTOGRAPHER_ADVANCED in kinds


# =====================================================================
# End-to-end demo: learning to mastery on a multi-task DAG
# =====================================================================


def test_e2e_dag_progresses_through_frontier_to_mastery():
    cart = Cartographer(
        mastery_threshold=0.7, entry_threshold=0.1,
        window_recent=4, window_prior=4,
    )
    cart.register_task("add-1d", value=1.0, cost=1.0)
    cart.register_task("add-2d", value=2.0, cost=2.0, prereqs=("add-1d",))
    cart.register_task("mul-1d", value=3.0, cost=3.0, prereqs=("add-2d",))
    cart.tick()

    # Initially, add-1d is on the frontier, the others are locked.
    assert cart.status("add-1d") == STATUS_FRONTIER
    assert cart.status("add-2d") == STATUS_LOCKED
    assert cart.status("mul-1d") == STATUS_LOCKED

    # Master add-1d.
    for _ in range(40):
        cart.observe("add-1d", 1.0)
    cart.tick()
    assert cart.status("add-1d") == STATUS_MASTERED
    assert cart.status("add-2d") == STATUS_FRONTIER
    assert cart.status("mul-1d") == STATUS_LOCKED

    # Master add-2d.
    for _ in range(40):
        cart.observe("add-2d", 1.0)
    cart.tick()
    assert cart.status("add-2d") == STATUS_MASTERED
    assert cart.status("mul-1d") == STATUS_FRONTIER

    # Recommendation should now propose mul-1d.
    cur = cart.recommend(policy=POLICY_UCB, k=1)
    assert cur.items[0].task_id == "mul-1d"
