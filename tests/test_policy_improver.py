"""Tests for `agi.policy_improver` — safe off-policy policy optimization.

The contract verified by these tests:

  1. **Optimization**: on data where one action dominates, the softmax
     improver concentrates mass on that action.
  2. **Contextual**: on data where the best action depends on x, the
     improver learns a non-degenerate context-conditional policy.
  3. **HCPI safety gate**: with a high baseline, no candidate can clear
     LCB > baseline; verdict must be UNSAFE or UNCERTAIN, never SAFE.
  4. **Small-data conservatism**: with ~tens of events the bound is
     wide and verdict is UNCERTAIN.
  5. **Bernstein bound** is a valid lower bound — empirically observed
     IPS sample mean must be within [lcb, ucb] in repeated trials.
  6. **Determinism**: same seed → same result.
  7. **Composability**: a PolicyImprover can ingest a PolicyLab's
     events and the resulting policy is callable as a `PolicyLab`
     `PolicyCandidate`.
  8. **Diagnostics**: ESS / coverage / clipped fraction reflect the
     truth of the underlying weighted-reward stream.
  9. **MixturePolicySpace** monotonically improves with α when target
     dominates baseline.
 10. **EpsilonGreedyPolicySpace** lower-bounds exploration and tunes
     the ε scalar correctly.
 11. **safety_check** on an arbitrary callable policy works.
 12. **Event emission** through the EventBus matches the documented
     kinds.
 13. **Input validation** — bad parameters raise, not silently miscompute.
"""

from __future__ import annotations

import math
import random

import pytest

from agi.events import EventBus
from agi.policy_improver import (
    EpsilonGreedyPolicySpace,
    HCPIReport,
    IMP_CHECKED,
    IMP_OPTIMIZED,
    IMP_PROMOTED,
    IMP_RECORDED,
    IMP_REJECTED,
    Improvement,
    KNOWN_OBJECTIVES,
    MixturePolicySpace,
    OBJ_CLIPPED_IPS,
    OBJ_CRM_VAR,
    OBJ_SNIPS,
    OptimizationDiagnostics,
    PolicyImprover,
    SAFE,
    SoftmaxPolicySpace,
    UNCERTAIN,
    UNSAFE,
    empirical_bernstein_bound,
    normal_lcb,
    to_policy_candidate,
)
from agi.policy_lab import (
    LoggedEvent,
    PolicyCandidate,
    PolicyLab,
)


# =====================================================================
# Test fixtures
# =====================================================================


def _logged_uniform_policy_data(
    *,
    rewards: dict[str, float],
    n: int,
    seed: int,
    noise: float = 0.05,
    context_features: dict[str, callable] | None = None,
) -> list[LoggedEvent]:
    rng = random.Random(seed)
    actions = list(rewards.keys())
    out: list[LoggedEvent] = []
    for _ in range(n):
        ctx = {}
        if context_features:
            for name, fn in context_features.items():
                ctx[name] = fn(rng)
        a = rng.choice(actions)
        mean = rewards[a]
        if callable(mean):
            mean = mean(ctx)
        r = max(0.0, min(1.0, rng.gauss(mean, noise)))
        out.append(LoggedEvent(context=ctx, action=a, propensity=1 / len(actions), reward=r))
    return out


# =====================================================================
# 1–2. Optimization recovers the best action
# =====================================================================


def test_softmax_concentrates_on_best_action():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9},
        n=2000,
        seed=0,
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=3, n_iters=120, learning_rate=0.5)
    assert isinstance(result, Improvement)
    assert result.verdict == SAFE
    pi = imp.policy_space.to_policy(result.parameters)
    p = pi({})
    # 'c' should win with high mass.
    assert p["c"] > p["a"] and p["c"] > p["b"]
    assert p["c"] > 0.8


def test_softmax_learns_context_dependent_policy():
    def best_mean(ctx):
        x = ctx.get("x", 0.0)
        return {"a": 0.9 if x < 0 else 0.2, "b": 0.5, "c": 0.2 if x < 0 else 0.9}

    rng = random.Random(0)
    actions = ["a", "b", "c"]
    events: list[LoggedEvent] = []
    for _ in range(3000):
        ctx = {"x": rng.gauss(0.0, 1.0)}
        a = rng.choice(actions)
        r = max(0.0, min(1.0, rng.gauss(best_mean(ctx)[a], 0.05)))
        events.append(
            LoggedEvent(context=ctx, action=a, propensity=1 / 3, reward=r)
        )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(actions, feature_names=["x"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=3, n_iters=120, learning_rate=0.5)
    pi = imp.policy_space.to_policy(result.parameters)
    p_neg = pi({"x": -2.0})
    p_pos = pi({"x": +2.0})
    # On negative x, 'a' should win.
    assert max(p_neg, key=p_neg.get) == "a"
    # On positive x, 'c' should win.
    assert max(p_pos, key=p_pos.get) == "c"
    # The policy is genuinely contextual.
    assert abs(p_neg["a"] - p_pos["a"]) > 0.5
    assert result.verdict == SAFE


# =====================================================================
# 3. HCPI: high baseline → not SAFE
# =====================================================================


def test_unbeatable_baseline_is_never_safe():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.7},
        n=2000,
        seed=0,
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.99,  # above the best action's value
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=3, n_iters=80, learning_rate=0.5)
    assert result.verdict in (UNSAFE, UNCERTAIN)
    assert result.safe is False
    assert result.improvement_lcb <= 0.0


def test_easy_baseline_is_safe():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9},
        n=3000,
        seed=0,
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.3,  # well below the best
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=3, n_iters=100, learning_rate=0.5)
    assert result.verdict == SAFE
    assert result.safe is True
    assert result.improvement_lcb > 0.0


# =====================================================================
# 4. Small-data conservatism
# =====================================================================


def test_tiny_data_is_uncertain():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9},
        n=15,
        seed=0,
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=2, n_iters=50)
    # CI half-width should be huge with this little data.
    assert result.value_ucb - result.value_lcb > 1.0
    assert result.verdict == UNCERTAIN


# =====================================================================
# 5. Bernstein bound covers the truth in repeated trials
# =====================================================================


def test_bernstein_bound_covers_in_repeated_trials():
    """Empirical Bernstein on iid samples should cover the mean in
    well-above (1-2δ)·100% of trials."""
    rng = random.Random(0)
    delta = 0.1
    truth = 0.5
    trials = 200
    cover = 0
    for _ in range(trials):
        xs = [
            max(0.0, min(1.0, rng.gauss(truth, 0.3))) for _ in range(80)
        ]
        mean, half = empirical_bernstein_bound(xs, delta=delta, value_range=1.0)
        if mean - half <= truth <= mean + half:
            cover += 1
    # Bernstein is *conservative*; nominal coverage 1-2δ=0.80 should be
    # comfortably exceeded.
    assert cover / trials >= 0.85


def test_bernstein_n0_returns_infinite_half():
    mean, half = empirical_bernstein_bound([], delta=0.05, value_range=1.0)
    assert math.isinf(half)


def test_bernstein_n1_returns_value_range_half():
    mean, half = empirical_bernstein_bound([0.42], delta=0.05, value_range=1.0)
    assert mean == pytest.approx(0.42)
    assert half == pytest.approx(1.0)


def test_normal_lcb_is_mean_minus_z_sigma():
    lcb = normal_lcb(mean=0.5, se=0.1, delta=0.025)
    # z_{0.975} ≈ 1.96 → 0.5 - 0.196 ≈ 0.304
    assert lcb == pytest.approx(0.5 - 1.95996 * 0.1, abs=1e-3)


# =====================================================================
# 6. Determinism with seed
# =====================================================================


def test_deterministic_with_seed():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9},
        n=500,
        seed=1,
    )
    args = dict(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"], feature_names=[]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=42,
    )
    a = PolicyImprover(**args)
    a.record_batch(events)
    ra = a.improve(n_restarts=3, n_iters=50)

    b = PolicyImprover(**args)
    b.record_batch(events)
    rb = b.improve(n_restarts=3, n_iters=50)

    assert ra.parameters == rb.parameters
    assert ra.value == rb.value
    assert ra.value_lcb == rb.value_lcb


# =====================================================================
# 7. Composability with PolicyLab
# =====================================================================


def test_pulls_events_from_policy_lab_and_round_trips():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9},
        n=2000,
        seed=0,
    )
    lab = PolicyLab()
    for e in events:
        lab.record(e)

    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    n = imp.ingest_from_lab(lab)
    assert n == len(events)
    assert len(imp) == len(events)
    result = imp.improve(n_restarts=2, n_iters=80, learning_rate=0.5)
    # Round-trip: PolicyLab should be able to evaluate the result.
    cand = to_policy_candidate(imp, result, name="improved")
    est = lab.evaluate(cand, method="snips")
    # PolicyLab's SNIPS evaluation should agree to within a few SEs.
    assert abs(est.value - result.value) < 0.05


def test_to_policy_candidate_wraps_policy():
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["x", "y"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(
        _logged_uniform_policy_data(rewards={"x": 0.3, "y": 0.7}, n=200, seed=0)
    )
    result = imp.improve(n_restarts=1, n_iters=30)
    cand = to_policy_candidate(imp, result, name="rolled-up")
    assert isinstance(cand, PolicyCandidate)
    assert cand.name == "rolled-up"
    p = cand.policy({})
    assert abs(sum(p.values()) - 1.0) < 1e-6


# =====================================================================
# 8. Diagnostics — ESS, coverage, clipped fraction
# =====================================================================


def test_diagnostics_reflect_truth():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9},
        n=1000,
        seed=0,
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.5,
        weight_clip=5.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=2, n_iters=80, learning_rate=0.5)
    d = result.diagnostics
    assert d.n == 1000
    assert 0.0 <= d.coverage <= 1.0
    assert d.coverage > 0.9  # softmax never zeroes any action
    assert d.n_eff > 0
    assert d.n_eff <= d.n
    assert d.max_weight <= imp.weight_clip
    assert d.mean_weight > 0
    # Diagnostics dict should serialize.
    assert isinstance(d.to_dict(), dict)


# =====================================================================
# 9. MixturePolicySpace — monotone improvement in α
# =====================================================================


def test_mixture_picks_high_alpha_when_target_dominates():
    events = _logged_uniform_policy_data(
        rewards={"good": 0.9, "ok": 0.5, "bad": 0.2},
        n=2000,
        seed=0,
    )

    def baseline_pi(ctx):
        return {"good": 1 / 3, "ok": 1 / 3, "bad": 1 / 3}

    def target_pi(ctx):
        return {"good": 1.0, "ok": 0.0, "bad": 0.0}

    imp = PolicyImprover(
        policy_space=MixturePolicySpace(baseline_pi, target_pi, ["good", "ok", "bad"]),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=4, n_iters=80, learning_rate=0.05)
    alpha = result.parameters[0]
    assert 0.0 <= alpha <= 1.0
    # Target dominates → high α.
    assert alpha > 0.6
    assert result.verdict == SAFE


def test_mixture_picks_low_alpha_when_target_is_worse():
    events = _logged_uniform_policy_data(
        rewards={"good": 0.9, "ok": 0.5, "bad": 0.2},
        n=2000,
        seed=0,
    )

    def baseline_pi(ctx):
        return {"good": 1.0, "ok": 0.0, "bad": 0.0}

    def target_pi(ctx):
        return {"good": 0.0, "ok": 0.0, "bad": 1.0}  # always pick the worst

    imp = PolicyImprover(
        policy_space=MixturePolicySpace(baseline_pi, target_pi, ["good", "ok", "bad"]),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=4, n_iters=80, learning_rate=0.05)
    alpha = result.parameters[0]
    # Target is worse than baseline → α should be near 0.
    assert alpha < 0.4


# =====================================================================
# 10. EpsilonGreedyPolicySpace — tunes ε
# =====================================================================


def test_epsilon_greedy_tunes_eps_toward_min_when_inner_is_good():
    events = _logged_uniform_policy_data(
        rewards={"good": 0.9, "bad": 0.1},
        n=1500,
        seed=0,
    )

    def inner(ctx):
        return {"good": 1.0, "bad": 0.0}

    imp = PolicyImprover(
        policy_space=EpsilonGreedyPolicySpace(
            inner=inner, actions=["good", "bad"], eps_min=0.0, eps_max=0.5
        ),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=3, n_iters=60, learning_rate=0.5)
    # Since the inner deterministic policy is perfect, ε should shrink.
    theta = result.parameters[0]
    eps = 0.0 + (0.5 - 0.0) * (1.0 / (1.0 + math.exp(-theta)))
    assert eps < 0.25


def test_epsilon_greedy_floor_holds():
    """Even if the inner policy is bad, the floor exploration prevents
    π(a|x) from collapsing to 0; coverage stays full."""
    events = _logged_uniform_policy_data(
        rewards={"good": 0.9, "bad": 0.1},
        n=500,
        seed=0,
    )

    def bad_inner(ctx):
        return {"good": 0.0, "bad": 1.0}

    imp = PolicyImprover(
        policy_space=EpsilonGreedyPolicySpace(
            inner=bad_inner, actions=["good", "bad"], eps_min=0.2, eps_max=0.5
        ),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=2, n_iters=40, learning_rate=0.3)
    d = result.diagnostics
    assert d.coverage == 1.0


# =====================================================================
# 11. safety_check on arbitrary policies
# =====================================================================


def test_safety_check_certifies_oracle_policy():
    def oracle(ctx):
        return {"a": 0.0, "b": 0.0, "c": 1.0}

    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9},
        n=2000,
        seed=0,
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    rpt = imp.safety_check(oracle, name="oracle")
    assert isinstance(rpt, HCPIReport)
    assert rpt.policy_name == "oracle"
    assert rpt.verdict == SAFE
    assert rpt.safe is True


def test_safety_check_rejects_worse_than_baseline():
    def bad(ctx):
        return {"a": 1.0, "b": 0.0, "c": 0.0}

    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9},
        n=2000,
        seed=0,
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    rpt = imp.safety_check(bad, name="all_a")
    assert rpt.verdict in (UNSAFE, UNCERTAIN)
    assert rpt.safe is False


def test_safety_check_with_policy_candidate():
    def oracle(ctx):
        return {"a": 0.0, "b": 0.0, "c": 1.0}

    cand = PolicyCandidate(name="my_oracle", policy=oracle)
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9}, n=1000, seed=0
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    rpt = imp.safety_check(cand)
    assert rpt.policy_name == "my_oracle"


def test_safety_check_with_override_baseline():
    def oracle(ctx):
        return {"a": 0.0, "b": 0.0, "c": 1.0}

    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9}, n=1500, seed=0
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    rpt1 = imp.safety_check(oracle)
    rpt2 = imp.safety_check(oracle, baseline_value=0.99)
    assert rpt1.baseline_value == 0.4
    assert rpt2.baseline_value == 0.99
    assert rpt1.verdict == SAFE
    assert rpt2.verdict in (UNSAFE, UNCERTAIN)


# =====================================================================
# 12. Event emission
# =====================================================================


def test_emits_recorded_optimized_promoted():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9}, n=300, seed=0
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        event_bus=bus,
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=1, n_iters=30, learning_rate=0.3)
    assert IMP_RECORDED in seen
    assert IMP_OPTIMIZED in seen
    assert (IMP_PROMOTED in seen) or (IMP_REJECTED in seen)
    if result.safe:
        assert IMP_PROMOTED in seen
    else:
        assert IMP_REJECTED in seen


def test_emits_checked_on_safety_check():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda ev: seen.append(ev.kind))
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b"]),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        event_bus=bus,
        seed=0,
    )
    imp.record_batch(
        _logged_uniform_policy_data(rewards={"a": 0.3, "b": 0.7}, n=200, seed=0)
    )

    def oracle(ctx):
        return {"a": 0.0, "b": 1.0}

    imp.safety_check(oracle)
    assert IMP_CHECKED in seen


# =====================================================================
# 13. Input validation
# =====================================================================


def test_invalid_objective_raises():
    with pytest.raises(ValueError):
        PolicyImprover(
            policy_space=SoftmaxPolicySpace(["a", "b"]),
            baseline_value=0.5,
            objective="not_an_objective",
        )


def test_invalid_clip_raises():
    with pytest.raises(ValueError):
        PolicyImprover(
            policy_space=SoftmaxPolicySpace(["a", "b"]),
            baseline_value=0.5,
            weight_clip=0.0,
        )


def test_invalid_delta_raises():
    with pytest.raises(ValueError):
        PolicyImprover(
            policy_space=SoftmaxPolicySpace(["a", "b"]),
            baseline_value=0.5,
            delta=1.0,
        )


def test_invalid_reward_range_raises():
    with pytest.raises(ValueError):
        PolicyImprover(
            policy_space=SoftmaxPolicySpace(["a", "b"]),
            baseline_value=0.5,
            reward_range=(1.0, 0.0),
        )


def test_improve_with_bad_args_raises():
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b"]),
        baseline_value=0.5,
    )
    with pytest.raises(ValueError):
        imp.improve(n_restarts=0)
    with pytest.raises(ValueError):
        imp.improve(n_iters=0)
    with pytest.raises(ValueError):
        imp.improve(learning_rate=0.0)


def test_softmax_requires_actions():
    with pytest.raises(ValueError):
        SoftmaxPolicySpace([])


def test_softmax_requires_unique_actions():
    with pytest.raises(ValueError):
        SoftmaxPolicySpace(["a", "a", "b"])


def test_epsilon_greedy_validates_eps_range():
    def inner(ctx):
        return {"x": 1.0}

    with pytest.raises(ValueError):
        EpsilonGreedyPolicySpace(inner=inner, actions=["x"], eps_min=0.5, eps_max=0.5)
    with pytest.raises(ValueError):
        EpsilonGreedyPolicySpace(inner=inner, actions=["x"], eps_min=-0.1, eps_max=0.5)
    with pytest.raises(ValueError):
        EpsilonGreedyPolicySpace(inner=inner, actions=[], eps_min=0.0, eps_max=0.5)


def test_mixture_requires_actions():
    with pytest.raises(ValueError):
        MixturePolicySpace(lambda c: {}, lambda c: {}, [])


def test_known_objectives_constants_complete():
    assert set(KNOWN_OBJECTIVES) == {OBJ_CLIPPED_IPS, OBJ_SNIPS, OBJ_CRM_VAR}


# =====================================================================
# Misc: SNIPS-objective and CRM-var objectives both run end-to-end
# =====================================================================


@pytest.mark.parametrize("obj", [OBJ_CLIPPED_IPS, OBJ_SNIPS, OBJ_CRM_VAR])
def test_all_objectives_run(obj):
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.5, "c": 0.9}, n=600, seed=0
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b", "c"]),
        baseline_value=0.4,
        objective=obj,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=2, n_iters=40, learning_rate=0.3)
    pi = imp.policy_space.to_policy(result.parameters)
    p = pi({})
    # 'c' should still win for all three objectives.
    assert max(p, key=p.get) == "c"


def test_no_events_yields_zero_value():
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
    )
    result = imp.improve(n_restarts=1, n_iters=10)
    assert result.diagnostics.n == 0
    assert result.value == 0.0


def test_to_dict_roundtrips_for_serialization():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.9}, n=400, seed=0
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b"]),
        baseline_value=0.3,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    imp.record_batch(events)
    result = imp.improve(n_restarts=1, n_iters=30)
    d = result.to_dict()
    assert d["verdict"] in (SAFE, UNSAFE, UNCERTAIN)
    assert "diagnostics" in d
    assert isinstance(d["parameters"], list)
    assert "n" in d["diagnostics"]


def test_record_batch_returns_count():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.5, "b": 0.5}, n=37, seed=0
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
    )
    n = imp.record_batch(events)
    assert n == 37
    assert len(imp) == 37


def test_max_events_drops_oldest_fifo():
    events = _logged_uniform_policy_data(
        rewards={"a": 0.5, "b": 0.5}, n=100, seed=0
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b"]),
        baseline_value=0.5,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        max_events=10,
    )
    imp.record_batch(events)
    assert len(imp) == 10
    # The oldest 90 should have been dropped.
    remaining = imp.events()
    assert remaining == events[-10:]


def test_hcpi_report_to_dict():
    def oracle(ctx):
        return {"a": 0.0, "b": 1.0}

    events = _logged_uniform_policy_data(
        rewards={"a": 0.2, "b": 0.8}, n=400, seed=0
    )
    imp = PolicyImprover(
        policy_space=SoftmaxPolicySpace(["a", "b"]),
        baseline_value=0.4,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
    )
    imp.record_batch(events)
    rpt = imp.safety_check(oracle)
    d = rpt.to_dict()
    assert d["verdict"] in (SAFE, UNSAFE, UNCERTAIN)
    assert "diagnostics" in d
    assert isinstance(d["diagnostics"], dict)


def test_optimization_diagnostics_is_picklable_dict():
    diag = OptimizationDiagnostics(
        n=10,
        n_eff=5.0,
        max_weight=2.0,
        mean_weight=1.0,
        clipped_fraction=0.1,
        coverage=1.0,
        iterations=42,
        restarts=3,
        objective_value=0.5,
        converged=True,
    )
    d = diag.to_dict()
    assert d["n"] == 10
    assert d["converged"] is True
