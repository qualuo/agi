"""Tests for agi.policy_lab — off-policy evaluation lab.

The math sanity-check strategy: generate synthetic logged data where the
true expected reward of a target policy is known analytically, then
confirm the estimators recover it inside their confidence intervals on
reasonable N. We do not test exact numbers (estimators are stochastic);
we test coverage, bounds, and the algorithmic structure of each method.
"""
from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path

import pytest

from agi.events import Event, EventBus
from agi.policy_lab import (
    KNOWN_METHODS,
    LAB_COMPARED,
    LAB_EVALUATED,
    LAB_RECOMMENDED,
    METHOD_DM,
    METHOD_DR,
    METHOD_IPS,
    METHOD_SNIPS,
    METHOD_SWITCH_DR,
    REC_INCONCLUSIVE,
    REC_KILL,
    REC_SHIP,
    Estimate,
    LinearRewardModel,
    LoggedEvent,
    PerActionMeanRewardModel,
    PolicyCandidate,
    PolicyLab,
    _bernstein_radius,
    _normalize_policy,
    _pareto_frontier,
    _solve,
)


# ----- synthetic worlds ----------------------------------------------


def make_world(
    n: int,
    actions=("cheap", "smart"),
    *,
    seed: int = 0,
    propensity_smart: float = 0.5,
):
    """Two-action world.

    The true reward depends on a context feature `difficulty` ∈ [0,1]:
    - `cheap`: reward = 1 - difficulty + noise
    - `smart`: reward = 0.9 + 0.05*difficulty + noise (always near 0.9)

    Logging policy picks `smart` with prob `propensity_smart` independent
    of context.

    True expected reward of a candidate policy can be computed exactly.
    """
    rng = random.Random(seed)
    events = []
    for _ in range(n):
        diff = rng.random()
        ctx = {"difficulty": diff}
        if rng.random() < propensity_smart:
            a = "smart"
            prop = propensity_smart
            r = 0.9 + 0.05 * diff + rng.gauss(0, 0.05)
        else:
            a = "cheap"
            prop = 1.0 - propensity_smart
            r = (1.0 - diff) + rng.gauss(0, 0.05)
        events.append(
            LoggedEvent(context=ctx, action=a, propensity=prop, reward=r)
        )
    return events


def true_expected_reward(policy_fn, n_grid: int = 200) -> float:
    """Compute true expected reward by integrating over difficulty."""
    s = 0.0
    for i in range(n_grid):
        diff = (i + 0.5) / n_grid
        p = policy_fn({"difficulty": diff})
        r_cheap = 1.0 - diff
        r_smart = 0.9 + 0.05 * diff
        s += p.get("cheap", 0.0) * r_cheap + p.get("smart", 0.0) * r_smart
    return s / n_grid


# ----- numerical primitives ------------------------------------------


def test_solve_2x2():
    # 2x + 3y = 8, 5x + 4y = 13 → x=1, y=2
    M = [[2.0, 3.0], [5.0, 4.0]]
    b = [8.0, 13.0]
    x = _solve(M, b)
    assert abs(x[0] - 1.0) < 1e-9
    assert abs(x[1] - 2.0) < 1e-9


def test_solve_singular_raises():
    from agi.policy_lab import _SingularMatrix
    M = [[1.0, 2.0], [2.0, 4.0]]
    b = [3.0, 6.0]
    with pytest.raises(_SingularMatrix):
        _solve(M, b)


def test_normalize_policy_handles_garbage():
    assert _normalize_policy({"a": -1.0, "b": 0.0}) == {}
    p = _normalize_policy({"a": 2.0, "b": 1.0})
    assert abs(p["a"] - 2 / 3) < 1e-12
    assert abs(p["b"] - 1 / 3) < 1e-12


def test_pareto_frontier_basic():
    # Three points in (reward, -cost) space; (0.9, -0.1) dominates the others.
    pts = [("a", 0.5, -0.5), ("b", 0.9, -0.1), ("c", 0.6, -0.2)]
    front = _pareto_frontier(pts)
    assert "b" in front
    # "c" is dominated by "b" (higher reward, lower cost).
    # "a" has lowest cost (-0.5 is biggest y), so still on frontier?
    # Actually y is -cost so a has y=-0.5, b has y=-0.1, c has y=-0.2.
    # a is dominated by b because b.x > a.x and b.y > a.y. So front = {b}.
    assert front == ["b"]


def test_bernstein_radius_shrinks_with_n():
    xs_small = [0.5, 0.6, 0.55, 0.45]
    xs_large = [0.5 + 0.05 * math.sin(i / 100) for i in range(1000)]
    r_small = _bernstein_radius(xs_small, 0.95)
    r_large = _bernstein_radius(xs_large, 0.95)
    assert r_large < r_small


# ----- reward models -------------------------------------------------


def test_per_action_mean_recovers_action_means():
    # No prior — pure sample mean.
    rm = PerActionMeanRewardModel(prior_n=0.0)
    events = [
        LoggedEvent(context={}, action="a", propensity=0.5, reward=1.0),
        LoggedEvent(context={}, action="a", propensity=0.5, reward=3.0),
        LoggedEvent(context={}, action="b", propensity=0.5, reward=10.0),
    ]
    rm.fit(events)
    assert rm.predict({}, "a") == pytest.approx(2.0, abs=1e-9)
    assert rm.predict({}, "b") == pytest.approx(10.0, abs=1e-9)


def test_per_action_mean_prior_pulls_toward_prior_mean():
    rm = PerActionMeanRewardModel(prior_n=2.0, prior_mean=0.0)
    events = [
        LoggedEvent(context={}, action="a", propensity=1.0, reward=4.0),
        LoggedEvent(context={}, action="a", propensity=1.0, reward=4.0),
    ]
    rm.fit(events)
    # (4+4 + 2*0) / (2 + 2) = 2.0
    assert rm.predict({}, "a") == pytest.approx(2.0, abs=1e-9)


def test_linear_reward_model_learns_slope():
    # reward = 5 + 2 * x for action 'a'.
    rm = LinearRewardModel(ridge=0.01)
    events = []
    for x in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        events.append(
            LoggedEvent(context={"x": x}, action="a", propensity=1.0, reward=5 + 2 * x)
        )
    rm.fit(events)
    assert abs(rm.predict({"x": 0.0}, "a") - 5.0) < 0.2
    assert abs(rm.predict({"x": 2.0}, "a") - 9.0) < 0.4
    # Unknown action falls back.
    assert rm.predict({"x": 0.0}, "unseen") == pytest.approx(rm._fallback, abs=1e-9)


def test_linear_reward_model_unknown_feature_treated_zero():
    rm = LinearRewardModel(ridge=0.1)
    events = [
        LoggedEvent(context={"x": 1.0}, action="a", propensity=1.0, reward=2.0),
        LoggedEvent(context={"x": 2.0}, action="a", propensity=1.0, reward=4.0),
    ]
    rm.fit(events)
    # New feature ignored.
    v_known = rm.predict({"x": 1.0}, "a")
    v_extra = rm.predict({"x": 1.0, "unused": 99.0}, "a")
    assert v_known == pytest.approx(v_extra, abs=1e-9)


def test_linear_reward_model_empty_events_is_safe():
    rm = LinearRewardModel()
    rm.fit([])
    assert rm.predict({"x": 1.0}, "a") == 0.0


# ----- core ingest ---------------------------------------------------


def test_lab_records_and_caps():
    lab = PolicyLab(max_events=10)
    for i in range(25):
        lab.record(LoggedEvent(context={}, action="a", propensity=1.0, reward=float(i)))
    assert len(lab) == 10
    # Oldest dropped: last event reward must be 24.
    evs = lab.events()
    assert evs[-1].reward == 24.0
    assert evs[0].reward == 15.0


def test_lab_record_batch_returns_count():
    lab = PolicyLab()
    n = lab.record_batch(
        LoggedEvent(context={}, action="a", propensity=1.0, reward=0.0)
        for _ in range(7)
    )
    assert n == 7
    assert len(lab) == 7


def test_lab_invalid_method_raises():
    lab = PolicyLab()
    lab.record(LoggedEvent(context={}, action="a", propensity=1.0, reward=1.0))
    with pytest.raises(ValueError):
        lab.evaluate(lambda c: {"a": 1.0}, method="nonsense")


def test_lab_invalid_confidence_raises():
    lab = PolicyLab()
    lab.record(LoggedEvent(context={}, action="a", propensity=1.0, reward=1.0))
    with pytest.raises(ValueError):
        lab.evaluate(lambda c: {"a": 1.0}, confidence=1.5)


def test_lab_empty_returns_empty_estimate():
    lab = PolicyLab()
    est = lab.evaluate(lambda c: {"a": 1.0})
    assert est.n == 0
    assert math.isinf(est.se)
    assert math.isinf(est.ci_high)


# ----- estimator math ------------------------------------------------


@pytest.fixture
def lab_with_world():
    lab = PolicyLab()
    for e in make_world(n=4000, seed=42, propensity_smart=0.5):
        lab.record(e)
    return lab


def test_ips_unbiased_on_uniform_target(lab_with_world):
    """When target == logging policy, IPS should recover logged mean."""
    lab = lab_with_world
    logged_mean = sum(e.reward for e in lab.events()) / len(lab)
    est = lab.evaluate(lambda c: {"cheap": 0.5, "smart": 0.5}, method=METHOD_IPS)
    # IPS unbiased; tolerance based on std error.
    assert abs(est.value - logged_mean) < 5 * est.se + 0.05


def test_snips_lower_variance_than_ips(lab_with_world):
    lab = lab_with_world
    target = lambda c: {"cheap": 0.5, "smart": 0.5}
    ips = lab.evaluate(target, method=METHOD_IPS)
    snips = lab.evaluate(target, method=METHOD_SNIPS)
    # SNIPS should reduce SE on this setup.
    assert snips.se <= ips.se * 1.1


def test_dm_recovers_per_action_means_when_no_context(lab_with_world):
    lab = lab_with_world
    target = lambda c: {"smart": 1.0}
    est = lab.evaluate(target, method=METHOD_DM)
    truth = true_expected_reward(lambda c: {"smart": 1.0})
    assert abs(est.value - truth) < 0.1


def test_dr_close_to_truth_for_uniform(lab_with_world):
    lab = lab_with_world
    target = lambda c: {"cheap": 0.5, "smart": 0.5}
    est = lab.evaluate(target, method=METHOD_DR)
    truth = true_expected_reward(target)
    # 95% CI should usually cover the truth.
    assert est.ci_low <= truth <= est.ci_high + 0.05


def test_dr_with_linear_model_for_context_aware_policy():
    lab = PolicyLab(reward_model=LinearRewardModel(ridge=0.1))
    for e in make_world(n=4000, seed=7, propensity_smart=0.5):
        lab.record(e)
    target = lambda c: (
        {"cheap": 1.0, "smart": 0.0}
        if c.get("difficulty", 0.5) < 0.5
        else {"cheap": 0.0, "smart": 1.0}
    )
    truth = true_expected_reward(target)
    est = lab.evaluate(target, method=METHOD_DR)
    # Smart policy should beat the random baseline truth ≈ 0.7.
    assert truth > 0.7
    # Estimator should be close.
    assert abs(est.value - truth) < 0.1


def test_switch_dr_lowers_max_weight_dependence():
    """SWITCH-DR falls back to DM when w_i > τ; should not crash on tiny props."""
    lab = PolicyLab(switch_tau=2.0)
    rng = random.Random(0)
    for _ in range(500):
        prop = rng.choice([0.001, 0.5])  # mix of tiny and normal propensities
        a = "rare" if prop == 0.001 else "common"
        lab.record(
            LoggedEvent(
                context={"x": rng.random()},
                action=a,
                propensity=prop,
                reward=rng.gauss(0, 1),
            )
        )
    target = lambda c: {"rare": 0.5, "common": 0.5}
    est = lab.evaluate(target, method=METHOD_SWITCH_DR)
    assert "switch_dr_fallback_frac" in est.diagnostics
    # Some fraction should have fallen back, but not all.
    frac = est.diagnostics["switch_dr_fallback_frac"]
    assert 0.0 < frac < 1.0


def test_weight_clip_is_honored():
    lab = PolicyLab(weight_clip=5.0)
    lab.record(LoggedEvent(context={}, action="a", propensity=0.01, reward=1.0))
    # With clip=5, w_max == 5.
    lab.evaluate(lambda c: {"a": 1.0}, method=METHOD_IPS)
    # Re-evaluate and inspect diagnostic via compare path.
    cmp = lab.compare(
        target=PolicyCandidate("t", lambda c: {"a": 1.0}),
        baseline=PolicyCandidate("b", lambda c: {"a": 1.0}),
        method=METHOD_IPS,
    )
    assert cmp.target.diagnostics["max_weight"] <= 5.0 + 1e-9


def test_all_methods_runnable(lab_with_world):
    lab = lab_with_world
    target = lambda c: {"cheap": 0.3, "smart": 0.7}
    for m in KNOWN_METHODS:
        est = lab.evaluate(target, method=m)
        assert est.method == m
        assert est.n > 0
        assert math.isfinite(est.value)


# ----- compare -------------------------------------------------------


def test_compare_recommends_ship_when_target_wins():
    lab = PolicyLab()
    for e in make_world(n=4000, seed=11, propensity_smart=0.5):
        lab.record(e)
    # Smart-only is uniformly mediocre (~0.925); cheap-only varies but
    # averages 0.5; "smart-only" target should beat "cheap-only" baseline.
    target = lambda c: {"smart": 1.0}
    baseline = lambda c: {"cheap": 1.0}
    cmp = lab.compare(
        target=PolicyCandidate("smart_only", target),
        baseline=PolicyCandidate("cheap_only", baseline),
        method=METHOD_DR,
    )
    assert cmp.recommend == REC_SHIP
    assert cmp.lift > 0
    assert cmp.lift_ci_low > 0


def test_compare_recommends_kill_when_target_loses():
    lab = PolicyLab()
    for e in make_world(n=4000, seed=12, propensity_smart=0.5):
        lab.record(e)
    cmp = lab.compare(
        target=PolicyCandidate("cheap_only", lambda c: {"cheap": 1.0}),
        baseline=PolicyCandidate("smart_only", lambda c: {"smart": 1.0}),
        method=METHOD_DR,
    )
    assert cmp.recommend == REC_KILL


def test_compare_inconclusive_when_policies_equal():
    lab = PolicyLab()
    for e in make_world(n=2000, seed=13, propensity_smart=0.5):
        lab.record(e)
    target = lambda c: {"cheap": 0.5, "smart": 0.5}
    cmp = lab.compare(
        target=PolicyCandidate("a", target),
        baseline=PolicyCandidate("b", target),
        method=METHOD_DR,
    )
    assert cmp.recommend == REC_INCONCLUSIVE
    assert abs(cmp.lift) < 0.05


def test_compare_with_min_lift_floors_ship_threshold():
    lab = PolicyLab()
    for e in make_world(n=2000, seed=14, propensity_smart=0.5):
        lab.record(e)
    target = lambda c: {"smart": 0.55, "cheap": 0.45}
    baseline = lambda c: {"smart": 0.5, "cheap": 0.5}
    cmp = lab.compare(
        target=PolicyCandidate("t", target),
        baseline=PolicyCandidate("b", baseline),
        method=METHOD_DR,
        min_lift=0.5,  # absurdly high
    )
    assert cmp.recommend == REC_INCONCLUSIVE


# ----- recommend -----------------------------------------------------


def test_recommend_picks_top_candidate():
    lab = PolicyLab()
    for e in make_world(n=3000, seed=23, propensity_smart=0.5):
        lab.record(e)
    cands = [
        PolicyCandidate("cheap_only", lambda c: {"cheap": 1.0}),
        PolicyCandidate("smart_only", lambda c: {"smart": 1.0}),
        PolicyCandidate(
            "context_aware",
            lambda c: (
                {"cheap": 1.0}
                if c.get("difficulty", 0.5) < 0.5
                else {"smart": 1.0}
            ),
        ),
    ]
    rec = lab.recommend(cands, method=METHOD_DR)
    # The context-aware policy is optimal here (truth ≈ 0.875).
    assert rec.best in {"context_aware", "smart_only"}
    assert "summary" not in rec.summary  # not literal — just sanity.
    assert "PolicyLab recommend" in rec.summary


def test_recommend_with_cost_picks_pareto_frontier():
    lab = PolicyLab()
    for e in make_world(n=2000, seed=24, propensity_smart=0.5):
        lab.record(e)
    cands = [
        PolicyCandidate("cheap", lambda c: {"cheap": 1.0}),
        PolicyCandidate("smart", lambda c: {"smart": 1.0}),
    ]
    rec = lab.recommend(
        cands,
        method=METHOD_DR,
        cost_per_action={"cheap": 0.01, "smart": 1.0},
    )
    assert {p.name for p in rec.frontier} == {"cheap", "smart"}
    smart_pt = next(p for p in rec.frontier if p.name == "smart")
    cheap_pt = next(p for p in rec.frontier if p.name == "cheap")
    assert smart_pt.expected_cost > cheap_pt.expected_cost
    # Both are on the frontier (one wins reward, other wins cost) so
    # neither is dominated.
    assert not smart_pt.dominated_by
    assert not cheap_pt.dominated_by


def test_recommend_empty_candidates_raises():
    lab = PolicyLab()
    with pytest.raises(ValueError):
        lab.recommend([])


# ----- events --------------------------------------------------------


def test_event_bus_emissions():
    bus = EventBus()
    got: list[Event] = []
    bus.subscribe(lambda e: got.append(e))
    lab = PolicyLab(event_bus=bus)
    for e in make_world(n=200, seed=99):
        lab.record(e)
    lab.evaluate(lambda c: {"cheap": 0.5, "smart": 0.5}, method=METHOD_IPS)
    lab.compare(
        target=PolicyCandidate("t", lambda c: {"smart": 1.0}),
        baseline=PolicyCandidate("b", lambda c: {"cheap": 1.0}),
    )
    lab.recommend(
        [
            PolicyCandidate("a", lambda c: {"cheap": 1.0}),
            PolicyCandidate("b", lambda c: {"smart": 1.0}),
        ]
    )
    kinds = {e.kind for e in got}
    assert LAB_EVALUATED in kinds
    assert LAB_COMPARED in kinds
    assert LAB_RECOMMENDED in kinds


def test_attach_to_bus_drains_events():
    bus = EventBus()
    lab = PolicyLab()
    unsub = lab.attach_to_bus(bus, kinds=("logged",))
    bus.publish(
        Event(
            kind="logged",
            data={
                "context": {"x": 1.0},
                "action": "a",
                "propensity": 0.5,
                "reward": 2.0,
            },
        )
    )
    bus.publish(Event(kind="other", data={"context": {}, "action": "a", "propensity": 1.0, "reward": 0.0}))
    assert len(lab) == 1
    unsub()


# ----- driver attach -------------------------------------------------


class _FakeReceipt:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeDriver:
    def __init__(self):
        self._subs: list = []

    def subscribe_receipts(self, cb):
        self._subs.append(cb)
        return lambda: self._subs.remove(cb)

    def emit(self, receipt):
        for cb in list(self._subs):
            cb(receipt)


def test_attach_to_driver_records_receipts():
    lab = PolicyLab()
    drv = _FakeDriver()
    unsub = lab.attach_to_driver(drv)
    drv.emit(
        _FakeReceipt(
            verdict="admit",
            propensity=0.7,
            ev_realized=1.25,
            estimated_cost_usd=0.01,
            estimated_duration_s=1.0,
            estimated_p_success=0.9,
            priority=1,
        )
    )
    assert len(lab) == 1
    ev = lab.events()[0]
    assert ev.action == "admit"
    assert ev.propensity == 0.7
    assert ev.reward == 1.25
    assert ev.context.get("estimated_p_success") == 0.9
    unsub()


def test_attach_to_driver_bad_receipt_is_swallowed():
    lab = PolicyLab()
    drv = _FakeDriver()
    lab.attach_to_driver(drv)

    class _Bad:
        @property
        def verdict(self):
            raise RuntimeError("boom")

    drv.emit(_Bad())  # should not raise
    assert len(lab) == 0


# ----- persistence ---------------------------------------------------


def test_snapshot_roundtrip():
    lab = PolicyLab(weight_clip=12.0, switch_tau=3.5)
    for e in make_world(n=100, seed=1):
        lab.record(e)
    snap = lab.snapshot()
    other = PolicyLab()
    other.restore(snap)
    assert len(other) == len(lab)
    assert other.weight_clip == 12.0
    assert other.switch_tau == 3.5
    # Estimates should match within numerical precision.
    target = lambda c: {"cheap": 0.5, "smart": 0.5}
    e1 = lab.evaluate(target, method=METHOD_IPS)
    e2 = other.evaluate(target, method=METHOD_IPS)
    assert abs(e1.value - e2.value) < 1e-9


def test_save_load_roundtrip(tmp_path: Path):
    lab = PolicyLab()
    for e in make_world(n=50, seed=2):
        lab.record(e)
    fp = tmp_path / "snap.json"
    lab.save(fp)
    other = PolicyLab()
    other.load(fp)
    assert len(other) == 50


def test_restore_rejects_unknown_version():
    lab = PolicyLab()
    with pytest.raises(ValueError):
        lab.restore({"version": 99, "events": []})


# ----- estimate serialisation ----------------------------------------


def test_estimate_to_dict_is_json_safe():
    import json as _json
    lab = PolicyLab()
    for e in make_world(n=200, seed=3):
        lab.record(e)
    est = lab.evaluate(lambda c: {"smart": 1.0}, method=METHOD_DR)
    s = _json.dumps(est.to_dict())
    parsed = _json.loads(s)
    assert parsed["policy_name"] == "target"
    assert parsed["method"] == METHOD_DR


def test_comparison_to_dict_is_json_safe():
    import json as _json
    lab = PolicyLab()
    for e in make_world(n=500, seed=4):
        lab.record(e)
    cmp = lab.compare(
        target=PolicyCandidate("a", lambda c: {"smart": 1.0}),
        baseline=PolicyCandidate("b", lambda c: {"cheap": 1.0}),
        method=METHOD_DR,
    )
    s = _json.dumps(cmp.to_dict())
    parsed = _json.loads(s)
    assert parsed["target"]["policy_name"] == "a"
    assert parsed["recommend"] in {REC_SHIP, REC_KILL, REC_INCONCLUSIVE}
