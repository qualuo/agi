"""Tests for the Intender runtime primitive — IRL / preference learning."""
from __future__ import annotations

import math
import random
import pytest

from agi.intender import (
    ANYTIME,
    APPRENTICESHIP,
    BEHAVIORAL_CLONING,
    BERNSTEIN,
    BIRL,
    BIRLChain,
    BehavioralCloningModel,
    FitCertificate,
    GenericConfigError,
    HOEFFDING,
    IdentifiabilityReport,
    INTENDER_CLEARED,
    INTENDER_FIT,
    INTENDER_PREFERENCE,
    INTENDER_REPORT,
    INTENDER_SAMPLED,
    INTENDER_STARTED,
    INTENDER_TRAJECTORY,
    InsufficientData,
    Intender,
    IntenderConfig,
    IntenderError,
    IntenderReport,
    InvalidMDP,
    InvalidPreference,
    InvalidTrajectory,
    InvalidWeights,
    KNOWN_ALGORITHMS,
    KNOWN_BOUND_METHODS,
    KNOWN_EVENTS,
    MAXENT,
    MaxEntFit,
    MDPSchema,
    PREFERENCE,
    PreferenceFit,
    SoftValue,
    UnknownAlgorithm,
    UnknownBoundMethod,
    anytime_half_width,
    behavioral_cloning,
    empirical_bernstein_half_width,
    empirical_feature_expectations,
    fit_apprenticeship_step,
    fit_birl,
    fit_maxent,
    fit_preference,
    half_width,
    hoeffding_half_width,
    identifiability_report,
    policy_kl_divergence,
    quick_gridworld_fixture,
    soft_feature_expectations,
    soft_q_iteration,
    trajectory_return,
)


# =====================================================================
# Constants and schema validation
# =====================================================================


def test_known_algorithms_complete():
    expected = {MAXENT, BIRL, PREFERENCE, APPRENTICESHIP, BEHAVIORAL_CLONING}
    assert KNOWN_ALGORITHMS == expected


def test_known_bound_methods_complete():
    assert KNOWN_BOUND_METHODS == {HOEFFDING, BERNSTEIN, ANYTIME}


def test_event_kinds_exposed():
    assert INTENDER_STARTED in KNOWN_EVENTS
    assert INTENDER_TRAJECTORY in KNOWN_EVENTS
    assert INTENDER_PREFERENCE in KNOWN_EVENTS
    assert INTENDER_FIT in KNOWN_EVENTS
    assert INTENDER_SAMPLED in KNOWN_EVENTS
    assert INTENDER_REPORT in KNOWN_EVENTS
    assert INTENDER_CLEARED in KNOWN_EVENTS


def test_invalid_algorithm_raises():
    cfg = IntenderConfig(algorithm="nonsense")
    with pytest.raises(UnknownAlgorithm):
        cfg.validate()


def test_invalid_bound_method_raises():
    cfg = IntenderConfig(bound_method="nope")
    with pytest.raises(UnknownBoundMethod):
        cfg.validate()


def test_invalid_delta_raises():
    cfg = IntenderConfig(delta=1.5)
    with pytest.raises(GenericConfigError):
        cfg.validate()


def test_schema_rejects_empty_states():
    with pytest.raises(InvalidMDP):
        # Empty states list.
        Intender.maxent(
            states=[], actions=["a"], features=lambda s, a: [1.0],
            transitions={},
        )


def test_schema_rejects_bad_gamma():
    with pytest.raises(InvalidMDP):
        Intender.maxent(
            states=["s"], actions=["a"], features=lambda s, a: [1.0],
            transitions={("s", "a"): [("s", 1.0)]},
            gamma=1.0,
        )


def test_schema_rejects_missing_transition():
    with pytest.raises(InvalidMDP):
        Intender.maxent(
            states=["s1", "s2"], actions=["a"],
            features=lambda s, a: [1.0],
            transitions={("s1", "a"): [("s1", 1.0)]},   # missing s2/a
        )


def test_schema_rejects_unnormalized_transition():
    with pytest.raises(InvalidMDP):
        Intender.maxent(
            states=["s"], actions=["a"],
            features=lambda s, a: [1.0],
            transitions={("s", "a"): [("s", 0.5)]},
        )


def test_schema_rejects_nonfinite_feature():
    with pytest.raises(InvalidMDP):
        Intender.maxent(
            states=["s"], actions=["a"],
            features=lambda s, a: [float("inf")],
            transitions={("s", "a"): [("s", 1.0)]},
        )


def test_schema_rejects_duplicate_states():
    with pytest.raises(InvalidMDP):
        Intender.maxent(
            states=["s", "s"], actions=["a"],
            features=lambda s, a: [1.0],
            transitions={("s", "a"): [("s", 1.0)]},
        )


# =====================================================================
# Soft Q-iteration
# =====================================================================


def test_soft_q_iteration_converges_on_gridworld():
    schema, _ = quick_gridworld_fixture()
    theta = [1.0, -0.05, 0.0, 0.0]  # reward goal, mild step cost
    soft = soft_q_iteration(schema, theta, vi_iters=500, vi_tol=1.0e-6)
    assert soft.converged is True
    # The policy at every state is a proper distribution over actions.
    for s in schema.states:
        total = sum(soft.policy[(s, a)] for a in schema.actions)
        assert abs(total - 1.0) < 1.0e-6
    # Goal-adjacent states should put substantial mass on moves toward goal.
    east_from_origin = soft.policy[((0, 0), "E")]
    north_from_origin = soft.policy[((0, 0), "N")]
    west_from_origin = soft.policy[((0, 0), "W")]
    south_from_origin = soft.policy[((0, 0), "S")]
    assert east_from_origin > west_from_origin
    assert north_from_origin > south_from_origin


def test_soft_q_iteration_zero_reward_gives_uniform_policy_single_state():
    # On a single-state self-loop MDP with zero reward, every action has the
    # same Q-value ⇒ the soft policy is exactly uniform.
    states = ["s"]
    actions = ["a", "b", "c"]
    transitions = {("s", a): [("s", 1.0)] for a in actions}
    intender = Intender.maxent(
        states=states, actions=actions,
        features=lambda s, a: [0.0],
        transitions=transitions, gamma=0.5,
    )
    soft = soft_q_iteration(intender.schema, [0.0], vi_iters=200, vi_tol=1.0e-12)
    for a in actions:
        assert abs(soft.policy[("s", a)] - 1.0 / 3) < 1.0e-9


def test_soft_q_iteration_rejects_wrong_dim():
    schema, _ = quick_gridworld_fixture()
    with pytest.raises(InvalidWeights):
        soft_q_iteration(schema, [1.0])


# =====================================================================
# Empirical feature expectations
# =====================================================================


def test_empirical_feature_expectations_correct_average():
    schema, _ = quick_gridworld_fixture()
    tau1 = [((0, 0), "E"), ((1, 0), "E"), ((2, 0), "N")]
    tau2 = [((0, 1), "E"), ((1, 1), "E"), ((2, 1), "N")]
    mu = empirical_feature_expectations(schema, [tau1, tau2])
    assert len(mu) == schema.feature_dim
    # Each trajectory has step-cost=1 at every step; discounted sum is bounded.
    assert mu[1] > 0.0


def test_empirical_feature_expectations_rejects_empty():
    schema, _ = quick_gridworld_fixture()
    with pytest.raises(InsufficientData):
        empirical_feature_expectations(schema, [])


def test_empirical_feature_expectations_rejects_unknown_state():
    schema, _ = quick_gridworld_fixture()
    with pytest.raises(InvalidTrajectory):
        empirical_feature_expectations(schema, [[(("zzz",), "N")]])


# =====================================================================
# MaxEnt IRL
# =====================================================================


def _expert_trajectory_toward_goal(schema, start=(0, 0), goal=(2, 2)):
    """Greedy Manhattan-walk trajectory toward goal."""
    s = start
    out = []
    while s != goal:
        x, y = s
        gx, gy = goal
        if x < gx:
            a = "E"
        elif x > gx:
            a = "W"
        elif y < gy:
            a = "N"
        else:
            a = "S"
        out.append((s, a))
        # Apply deterministic step.
        if a == "E":
            s = (x + 1, y)
        elif a == "W":
            s = (x - 1, y)
        elif a == "N":
            s = (x, y + 1)
        else:
            s = (x, y - 1)
    out.append((goal, "X"))
    return out


def test_fit_maxent_recovers_positive_goal_weight():
    schema, _ = quick_gridworld_fixture()
    trajs = [_expert_trajectory_toward_goal(schema) for _ in range(8)]
    fit = fit_maxent(
        schema, trajs,
        lr=0.5, tol=1.0e-3, max_iters=300, l2=1.0e-3,
    )
    # Goal reward θ[0] should be the dominant positive weight; step cost θ[1] negative.
    assert fit.theta[0] > 0.0
    assert fit.theta[1] < fit.theta[0]
    # Residual should be small.
    assert fit.residual_norm < 1.0


def test_fit_maxent_converges_under_l2():
    schema, _ = quick_gridworld_fixture()
    trajs = [_expert_trajectory_toward_goal(schema) for _ in range(5)]
    fit = fit_maxent(schema, trajs, max_iters=200, l2=0.1, tol=1.0e-3)
    # With strong L2, gradient norm history should be monotone-ish small at the end.
    assert fit.history[-1] < fit.history[0] + 1.0
    # θ stays in a bounded region.
    assert max(abs(x) for x in fit.theta) < 50.0


def test_fit_maxent_rejects_empty():
    schema, _ = quick_gridworld_fixture()
    with pytest.raises(InsufficientData):
        fit_maxent(schema, [])


def test_fit_maxent_rejects_bad_init_dim():
    schema, _ = quick_gridworld_fixture()
    trajs = [_expert_trajectory_toward_goal(schema)]
    with pytest.raises(InvalidWeights):
        fit_maxent(schema, trajs, init_theta=[1.0])


# =====================================================================
# Preference learning
# =====================================================================


def test_fit_preference_recovers_winner_direction():
    schema, _ = quick_gridworld_fixture()
    # Winner trajectory reaches the goal; loser stays put.
    winner = _expert_trajectory_toward_goal(schema)
    loser = [((0, 0), "X") for _ in range(5)]
    prefs = [(winner, loser) for _ in range(20)]
    fit = fit_preference(schema, prefs, max_iters=300, tol=1.0e-3, lr=0.5)
    # Winner has larger return: θᵀ (Φ(w) − Φ(l)) > 0 under fitted θ.
    return_w = trajectory_return(schema, fit.theta, winner)
    return_l = trajectory_return(schema, fit.theta, loser)
    assert return_w > return_l
    assert fit.agreement_rate > 0.9


def test_fit_preference_rejects_empty():
    schema, _ = quick_gridworld_fixture()
    with pytest.raises(InsufficientData):
        fit_preference(schema, [])


def test_fit_preference_rejects_invalid_pair():
    schema, _ = quick_gridworld_fixture()
    winner = _expert_trajectory_toward_goal(schema)
    with pytest.raises(InvalidPreference):
        fit_preference(schema, [(winner, [])])


# =====================================================================
# Behavioural cloning
# =====================================================================


def test_behavioral_cloning_recovers_visited_action():
    schema, _ = quick_gridworld_fixture()
    tau = [((0, 0), "E"), ((1, 0), "E"), ((2, 0), "N")]
    bc = behavioral_cloning(schema, [tau], alpha=0.0)
    # At (0,0) action E was the only one ever seen — α=0 ⇒ probability 1.
    assert bc.policy[((0, 0), "E")] == pytest.approx(1.0)
    assert bc.policy[((0, 0), "N")] == pytest.approx(0.0)


def test_behavioral_cloning_smoothing():
    schema, _ = quick_gridworld_fixture()
    tau = [((0, 0), "E")]
    bc = behavioral_cloning(schema, [tau], alpha=1.0)
    # Laplace smoothing: every action has positive probability.
    for a in schema.actions:
        assert bc.policy[((0, 0), a)] > 0.0
    # E should still dominate.
    assert bc.policy[((0, 0), "E")] > bc.policy[((0, 0), "N")]


def test_behavioral_cloning_rejects_empty():
    schema, _ = quick_gridworld_fixture()
    with pytest.raises(InsufficientData):
        behavioral_cloning(schema, [])


# =====================================================================
# Identifiability
# =====================================================================


def test_identifiability_full_rank_features():
    schema, _ = quick_gridworld_fixture()
    rep = identifiability_report(schema)
    assert rep.feature_dim == 4
    # All four features are linearly independent in our gridworld fixture.
    assert rep.rank == 4
    assert rep.nullity == 0


def test_identifiability_redundant_features():
    # An MDP whose features are colinear should report nullity > 0.
    def features(s, a):
        return [1.0, 1.0]  # both features identical
    states = ["s"]
    actions = ["a"]
    transitions = {("s", "a"): [("s", 1.0)]}
    intender = Intender.maxent(
        states=states, actions=actions, features=features,
        transitions=transitions, gamma=0.5,
    )
    rep = identifiability_report(intender.schema)
    assert rep.feature_dim == 2
    assert rep.rank == 1
    assert rep.nullity == 1


# =====================================================================
# Bound primitives (sanity)
# =====================================================================


def test_hoeffding_half_width_decreases_with_n():
    a = hoeffding_half_width(10, 0.05)
    b = hoeffding_half_width(100, 0.05)
    assert b < a


def test_anytime_half_width_decreases_with_n():
    a = anytime_half_width(10, 0.05)
    b = anytime_half_width(100, 0.05)
    assert b < a


def test_anytime_dominates_hoeffding_at_some_n():
    # Anytime is wider than Hoeffding (it covers all stopping times).
    # We just check it's positive and finite.
    hw = anytime_half_width(100, 0.05)
    assert 0.0 < hw < 1.0


def test_empirical_bernstein_with_zero_variance():
    hw = empirical_bernstein_half_width(100, 0.0, 0.05)
    # When variance is 0, the bound is dominated by the Bennett-style residual.
    assert hw > 0


def test_half_width_dispatch():
    assert half_width(HOEFFDING, 10, 0.05) > 0
    assert half_width(ANYTIME, 10, 0.05) > 0
    assert half_width(BERNSTEIN, 10, 0.05, sample_variance=0.1) > 0
    with pytest.raises(UnknownBoundMethod):
        half_width("bogus", 10, 0.05)


def test_bernstein_requires_variance():
    with pytest.raises(IntenderError):
        half_width(BERNSTEIN, 10, 0.05)


# =====================================================================
# End-to-end Intender lifecycle
# =====================================================================


class _DummyBus:
    def __init__(self):
        self.events = []

    def publish(self, kind, payload):
        self.events.append((kind, dict(payload)))


def test_intender_maxent_end_to_end():
    schema, _ = quick_gridworld_fixture()
    bus = _DummyBus()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        bus=bus, lr=0.5, max_iters=200, tol=1.0e-3, l2=1.0e-3,
    )
    trajs = [_expert_trajectory_toward_goal(schema) for _ in range(6)]
    for tau in trajs:
        intender.observe_trajectory(tau)
    fit = intender.fit()
    assert isinstance(fit, MaxEntFit)
    report = intender.report()
    assert isinstance(report, IntenderReport)
    assert report.algorithm == MAXENT
    assert report.feature_dim == 4
    assert report.n_trajectories == 6
    assert report.feature_residual_norm is not None
    assert report.kl_to_bc is not None
    # Bus should have at least started, n trajectories, fit, report events.
    kinds = [k for k, _ in bus.events]
    assert INTENDER_STARTED in kinds
    assert INTENDER_TRAJECTORY in kinds
    assert INTENDER_FIT in kinds
    assert INTENDER_REPORT in kinds


def test_intender_preference_end_to_end():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.preference(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        lr=0.5, max_iters=300, tol=1.0e-3,
    )
    winner = _expert_trajectory_toward_goal(schema)
    loser = [((0, 0), "X") for _ in range(5)]
    for _ in range(15):
        intender.observe_preference(winner, loser)
    fit = intender.fit()
    assert isinstance(fit, PreferenceFit)
    report = intender.report()
    assert report.algorithm == PREFERENCE
    assert report.agreement_rate is not None
    assert "preference_agreement" in report.certificates
    cert = report.certificates["preference_agreement"]
    assert isinstance(cert, FitCertificate)
    # cert.upper is the raw estimate + half_width and may exceed 1; the
    # *clipped* version is exposed by evaluate_preferences.
    assert cert.lower <= cert.estimate <= cert.upper
    assert cert.half_width > 0.0


def test_intender_birl_end_to_end():
    schema, _ = quick_gridworld_fixture(width=2, height=2)
    intender = Intender.birl(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        beta=2.0, n_steps=50, burn_in=50, proposal_scale=0.3, seed=7,
        vi_iters=40,
    )
    # A few short trajectories that head toward (1, 1).
    intender.observe_trajectory([((0, 0), "E"), ((1, 0), "N"), ((1, 1), "X")])
    intender.observe_trajectory([((0, 0), "N"), ((0, 1), "E"), ((1, 1), "X")])
    intender.observe_trajectory([((0, 1), "E"), ((1, 1), "X")])
    chain = intender.fit()
    assert isinstance(chain, BIRLChain)
    assert len(chain.samples) > 0
    report = intender.report()
    assert report.algorithm == BIRL
    assert report.posterior_mean is not None
    assert report.posterior_lower is not None
    assert report.posterior_upper is not None
    # Posterior intervals respect ordering.
    for lo, hi in zip(report.posterior_lower, report.posterior_upper):
        assert lo <= hi
    assert report.acceptance_rate is not None
    assert 0.0 <= report.acceptance_rate <= 1.0


def test_intender_clear_resets_state():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
    )
    intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    intender.fit()
    intender.clear()
    assert intender.n_trajectories == 0
    assert intender.n_preferences == 0


def test_intender_rejects_unknown_state_trajectory():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
    )
    with pytest.raises(InvalidTrajectory):
        intender.observe_trajectory([("not-a-state", "N")])


def test_intender_rejects_empty_trajectory():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
    )
    with pytest.raises(InvalidTrajectory):
        intender.observe_trajectory([])


def test_intender_held_out_preference_evaluation():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.preference(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        max_iters=300, tol=1.0e-3,
    )
    winner = _expert_trajectory_toward_goal(schema)
    loser = [((0, 0), "X") for _ in range(5)]
    for _ in range(20):
        intender.observe_preference(winner, loser)
    intender.fit()
    # Build a held-out preference set.
    held_out = [(winner, loser) for _ in range(10)]
    stats = intender.evaluate_preferences(held_out)
    assert stats["agreement_rate"] >= 0.5
    assert 0.0 <= stats["agreement_lower"] <= stats["agreement_upper"] <= 1.0


def test_intender_reward_function():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        max_iters=200, tol=1.0e-3,
    )
    for _ in range(5):
        intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    intender.fit()
    # Reward at the goal moving away should be lower than staying at goal.
    r_at_goal = intender.reward((2, 2), "X")
    r_off_goal = intender.reward((0, 0), "W")
    assert r_at_goal > r_off_goal


def test_intender_policy_method():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        max_iters=100, tol=1.0e-3,
    )
    for _ in range(5):
        intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    intender.fit()
    policy = intender.policy()
    # Policy is a proper conditional distribution.
    for s in schema.states:
        total = sum(policy[(s, a)] for a in schema.actions)
        assert abs(total - 1.0) < 1.0e-6


def test_intender_sample_posterior_birl():
    schema, _ = quick_gridworld_fixture(width=2, height=2)
    intender = Intender.birl(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        n_steps=40, burn_in=20, vi_iters=30, seed=11,
    )
    intender.observe_trajectory([((0, 0), "E"), ((1, 0), "N"), ((1, 1), "X")])
    intender.fit()
    samples = intender.sample_posterior(n=10)
    assert len(samples) <= 10
    assert all(len(s) == intender.schema.feature_dim for s in samples)


def test_intender_sample_posterior_maxent_returns_point_mass():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        max_iters=50, tol=1.0e-3,
    )
    intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    intender.fit()
    samples = intender.sample_posterior(n=5)
    assert len(samples) == 5


def test_intender_sample_posterior_before_fit_raises():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
    )
    with pytest.raises(GenericConfigError):
        intender.sample_posterior(n=5)


def test_intender_fingerprint_chains_deterministically():
    """Tamper-evident chain advances on every observation."""
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
    )
    fp0 = intender.fingerprint
    intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    fp1 = intender.fingerprint
    intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    fp2 = intender.fingerprint
    assert fp0 != fp1 != fp2
    # Fingerprints are SHA-256 hex strings.
    assert all(len(fp) == 64 for fp in (fp0, fp1, fp2))


def test_intender_set_initial_distribution_validates():
    schema, _ = quick_gridworld_fixture()
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
    )
    intender.set_initial_distribution({(0, 0): 1.0})
    # An unknown state in the distribution raises.
    with pytest.raises(InvalidMDP):
        intender.set_initial_distribution({"zzz": 1.0})


def test_intender_behavioral_cloning_algorithm():
    schema, _ = quick_gridworld_fixture()
    intender = Intender(
        schema=schema,
        config=IntenderConfig(algorithm=BEHAVIORAL_CLONING),
    )
    for _ in range(3):
        intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    fit = intender.fit()
    assert isinstance(fit, BehavioralCloningModel)
    policy = intender.policy()
    assert len(policy) == len(schema.states) * len(schema.actions)


def test_intender_apprenticeship_step():
    schema, _ = quick_gridworld_fixture()
    intender = Intender(
        schema=schema,
        config=IntenderConfig(algorithm=APPRENTICESHIP),
    )
    intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    fit = intender.fit()
    assert len(fit.theta) == schema.feature_dim
    # θ is a unit vector when there are no seen mus.
    norm = math.sqrt(sum(t * t for t in fit.theta))
    assert abs(norm - 1.0) < 1.0e-6


# =====================================================================
# Policy KL composition with Quantilizer-style safe-deployment
# =====================================================================


def test_policy_kl_divergence_zero_when_identical():
    schema, _ = quick_gridworld_fixture()
    p = {(s, a): 1.0 / len(schema.actions) for s in schema.states for a in schema.actions}
    kl = policy_kl_divergence(schema, p, p)
    assert abs(kl) < 1.0e-9


def test_policy_kl_divergence_infinite_on_zero_support():
    schema, _ = quick_gridworld_fixture()
    p = {(s, a): 1.0 / len(schema.actions) for s in schema.states for a in schema.actions}
    q = dict(p)
    # Zero out one action in q at (0, 0).
    q[((0, 0), "X")] = 0.0
    z_total = sum(q[((0, 0), a)] for a in schema.actions if a != "X")
    for a in schema.actions:
        if a != "X":
            q[((0, 0), a)] /= z_total
    # p has positive mass on X at (0,0) but q has zero → KL is +inf.
    kl = policy_kl_divergence(schema, p, q)
    assert kl == float("inf")


# =====================================================================
# Trajectory return composition
# =====================================================================


def test_trajectory_return_is_discounted():
    schema, _ = quick_gridworld_fixture()
    theta = [1.0, 0.0, 0.0, 0.0]
    # A single-step trajectory at the goal: return = γ^0 · 1.
    tau = [((2, 2), "X")]
    assert trajectory_return(schema, theta, tau) == pytest.approx(1.0)
    # Two-step trajectory both at goal: return = 1 + γ · 1.
    tau = [((2, 2), "X"), ((2, 2), "X")]
    assert trajectory_return(schema, theta, tau) == pytest.approx(1.0 + schema.gamma)


def test_trajectory_return_rejects_bad_state():
    schema, _ = quick_gridworld_fixture()
    theta = [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(InvalidTrajectory):
        trajectory_return(schema, theta, [("nope", "X")])


def test_trajectory_return_rejects_bad_theta_dim():
    schema, _ = quick_gridworld_fixture()
    with pytest.raises(InvalidWeights):
        trajectory_return(schema, [1.0], [((0, 0), "X")])
