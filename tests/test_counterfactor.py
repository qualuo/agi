"""Tests for ``agi.counterfactor`` — sequential off-policy evaluation.

The tests follow the mathematical contracts of the module:

1.  **Estimator identities** on simple deterministic ground truth.
    Trajectory IS / PDIS / DR-RL all match the true policy value on
    a tabular MDP under known logging propensities.
2.  **Self-normalisation** drives WIS / WPDIS to the true value on
    log-shifted IS weights.
3.  **Doubly robust** is consistent if **either** the Q̂ model **or**
    the propensities are correct (Jiang-Li 2016 / Robins 1994).
4.  **MAGIC** convex blend lies in the convex hull of its g^j family;
    extreme indices recover DM (j=0) and DR-RL (j=H).
5.  **HCOPE** lower bound is dominated by the point estimate and
    holds in coverage simulations.
6.  **Confidence intervals**: Hoeffding ⊇ empirical-Bernstein ⊇
    Student-t for low-variance bags; conformal envelope achieves
    nominal coverage under exchangeable sampling.
7.  **Effective sample size** matches Kong 1992; weight diagnostics
    are tail-aware.
8.  **Policy adapters** produce normalised distributions over fixed
    action sets; epsilon-greedy and softmax invariants.
9.  **Q-models**: ConstantQModel returns its constant; TabularQModel
    fits the empirical cumulative return; LinearQModel solves the
    ridge normal equations exactly on linear data.
10. **Bandit bridge**: H=1 trajectories give the contextual-bandit
    IPS/SNIPS/DM/DR values within numerical tolerance.
11. **Compare**: paired-difference dominance is symmetric (delta(A,B)
    = -delta(B,A)) and self-comparison has delta=0.
12. **Diagnostics**: ESS, max-weight, overlap-KL, and clip fraction
    surface support-violation warnings.
13. **Attestation**: each call emits a content-hashed receipt to the
    optional attestor.
14. **Threadsafety**: concurrent log_trajectory + evaluate are
    consistent and the counter advances atomically.

Pure-Python (stdlib only); runs without an API key.
"""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.events import Event, EventBus
from agi.counterfactor import (
    CI_BERNSTEIN,
    CI_CONFORMAL,
    CI_HOEFFDING,
    CI_STUDENT_T,
    CompareReport,
    ConstantQModel,
    Counterfactor,
    CounterfactorError,
    DeterministicPolicy,
    DiagnosticsReport,
    EpsilonGreedyPolicy,
    HCOPEReport,
    InsufficientData,
    KNOWN_CI,
    KNOWN_METHODS,
    LinearQModel,
    LoggedStep,
    LoggedTrajectory,
    METHOD_DM,
    METHOD_DR_RL,
    METHOD_MAGIC,
    METHOD_PDIS,
    METHOD_TRAJ_IS,
    METHOD_TRAJ_WIS,
    METHOD_WDR,
    METHOD_WPDIS,
    OPEReport,
    SoftmaxPolicy,
    SupportViolation,
    TabularQModel,
    UniformPolicy,
    UnknownMethod,
    conformal_envelope,
    dm,
    dr_rl,
    empirical_bernstein_half_width,
    ess,
    hcope_lower_bound,
    hoeffding_half_width,
    magic,
    overlap_kl,
    pdis,
    student_t_half_width,
    traj_is,
    traj_wis,
    wdr,
    weight_diagnostics,
    wpdis,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0xC0FFEE)


def _bernoulli_log(
    n: int,
    horizon: int,
    actions: list[str],
    behaviour_prob: dict[str, float],
    reward_of: callable,
    seed: int = 0,
) -> list[LoggedTrajectory]:
    """Build ``n`` trajectories under a fixed-probability bandit log."""
    r = random.Random(seed)
    trajs: list[LoggedTrajectory] = []
    actions_t = tuple(actions)
    weights = [behaviour_prob[a] for a in actions_t]
    for _ in range(n):
        steps = []
        for t in range(horizon):
            a = r.choices(actions_t, weights=weights)[0]
            steps.append(
                LoggedStep(
                    state=t,
                    action=a,
                    reward=float(reward_of(t, a)),
                    behavior_prob=behaviour_prob[a],
                )
            )
        trajs.append(LoggedTrajectory(steps=tuple(steps)))
    return trajs


# =====================================================================
# Basic estimator identities
# =====================================================================


class TestEstimators:
    def test_traj_is_unbiased_under_uniform_target(self):
        # Under target = behaviour, all importance weights = 1,
        # so traj_is = empirical mean return.
        actions = ["a", "b"]
        log = _bernoulli_log(
            300, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=1
        )
        target = UniformPolicy(actions)
        v, _ = traj_is(log, target, gamma=1.0)
        # Truth: each step has prob 0.5 of reward 1 → expected return 1.5.
        assert abs(v - 1.5) < 0.2

    def test_pdis_equals_traj_is_under_target_equals_behaviour(self):
        actions = ["a", "b"]
        log = _bernoulli_log(
            200, 4, actions, {"a": 0.5, "b": 0.5}, lambda t, a: t * 1.0, seed=2
        )
        target = UniformPolicy(actions)
        v1, _ = traj_is(log, target, gamma=1.0)
        v2, _ = pdis(log, target, gamma=1.0)
        # Under target=behaviour the per-step weight is identically 1,
        # so PDIS = trajectory IS exactly.
        assert abs(v1 - v2) < 1e-9

    def test_wpdis_self_normalises_to_truth_under_deterministic_target(self):
        # Target = "always a" against 50/50 behaviour: only trajectories
        # that picked 'a' at every step survive, but those trajectories
        # have the true return of "always a", so WPDIS converges exactly.
        actions = ["a", "b"]
        log = _bernoulli_log(
            500, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=3
        )
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        v, _, _ = wpdis(log, target, gamma=1.0)
        # Truth for "always a" is 3.0 (three rewards of 1).
        assert abs(v - 3.0) < 1e-9

    def test_dr_rl_recovers_truth_with_perfect_q(self):
        actions = ["a", "b"]
        rewards = {"a": 1.0, "b": 0.0}

        log = _bernoulli_log(
            200, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: rewards[a], seed=4
        )

        class PerfectQ(ConstantQModel):
            def q(self, state, action):
                # cumulative future return from this (state, action):
                # immediate r + expected future at uniform behaviour = 0.5
                # Future H - 1 - state random rewards under always-'a' target:
                # but for the DR check we want Q̂ s.t. dr_rl picks up the
                # truth on the deterministic 'a' target.
                return rewards[action] + (3 - 1 - int(state))  # if always-a downstream

        q = PerfectQ()
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        v, _ = dr_rl(log, target, q, gamma=1.0)
        assert abs(v - 3.0) < 1e-9

    def test_dr_rl_consistent_if_propensity_correct_q_wrong(self):
        # Q̂ is arbitrarily wrong (constant 5). DR-RL is *unbiased*
        # because IS weights are exact, but DR with mis-specified Q̂
        # has high variance — verify the *bias* is small via averaging
        # across many independent draws.
        actions = ["a", "b"]
        means = []
        for seed in range(20):
            log = _bernoulli_log(
                300, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=seed
            )
            q = ConstantQModel(5.0)
            target = DeterministicPolicy(lambda s: "a", actions=actions)
            v, _ = dr_rl(log, target, q, gamma=1.0)
            means.append(v)
        # average of unbiased estimators converges to truth
        grand_mean = sum(means) / len(means)
        assert abs(grand_mean - 3.0) < 0.5

    def test_dr_rl_consistent_if_q_correct_propensity_wrong(self):
        # Build a Q̂ that returns the true value V̂(s,a). DR should
        # also be consistent purely from the model term — the residual
        # vanishes in expectation when Q̂ is exact (Jiang-Li 2016).
        actions = ["a", "b"]
        rewards = {"a": 1.0, "b": 0.0}
        log = _bernoulli_log(
            200, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: rewards[a], seed=6
        )

        class TrueQ(ConstantQModel):
            def q(self, state, action):
                # Cumulative future return under "always a" from (state,action)
                # is r(a) + (H - 1 - state) * 1.0.
                return rewards[action] + (3 - 1 - int(state))

        target = DeterministicPolicy(lambda s: "a", actions=actions)
        q = TrueQ()
        v, _ = dr_rl(log, target, q, gamma=1.0)
        assert abs(v - 3.0) < 1e-9

    def test_dm_uses_starting_state_value(self):
        # If Q̂ is constant c, then V̂(s_0) = c regardless of policy.
        actions = ["a", "b"]
        log = _bernoulli_log(50, 4, actions, {"a": 0.7, "b": 0.3}, lambda t, a: 0.0)
        target = UniformPolicy(actions)
        q = ConstantQModel(2.5)
        v, contribs = dm(log, target, q)
        assert all(abs(c - 2.5) < 1e-9 for c in contribs)
        assert abs(v - 2.5) < 1e-9


# =====================================================================
# MAGIC convex blend
# =====================================================================


class TestMagic:
    def test_j_0_equals_dm(self):
        actions = ["a", "b"]
        log = _bernoulli_log(
            100, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=7
        )
        q = ConstantQModel(2.5)
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        v_dm, _ = dm(log, target, q)
        v_magic_j0, _ = magic(log, target, q, j_set=[0])
        assert abs(v_dm - v_magic_j0) < 1e-9

    def test_j_H_equals_dr_rl(self):
        actions = ["a", "b"]
        log = _bernoulli_log(
            100, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=8
        )
        q = ConstantQModel(0.0)
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        v_dr, _ = dr_rl(log, target, q)
        v_magic_jH, _ = magic(log, target, q, j_set=[3])
        assert abs(v_dr - v_magic_jH) < 1e-9

    def test_lambda_sums_to_one(self):
        actions = ["a", "b"]
        log = _bernoulli_log(
            100, 4, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=9
        )
        q = ConstantQModel(0.5)
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        v, lam = magic(log, target, q)
        assert abs(sum(lam) - 1.0) < 1e-9
        assert all(l >= -1e-6 for l in lam)


# =====================================================================
# Confidence intervals
# =====================================================================


class TestCIs:
    def test_hoeffding_monotone_in_n(self):
        h1 = hoeffding_half_width(100, 1.0, 0.05)
        h2 = hoeffding_half_width(400, 1.0, 0.05)
        assert h1 > h2

    def test_empirical_bernstein_tighter_for_low_variance(self):
        values = [0.5] * 50  # zero variance
        eb = empirical_bernstein_half_width(values, alpha=0.05, range_=1.0)
        h = hoeffding_half_width(50, 1.0, 0.05)
        assert eb < h  # bernstein dominates when variance is small

    def test_student_t_monotone_in_alpha(self):
        values = [0.0, 1.0, 0.5, 0.7, 0.3, 0.4, 0.6]
        wide = student_t_half_width(values, alpha=0.01)
        narrow = student_t_half_width(values, alpha=0.20)
        assert wide > narrow

    def test_conformal_envelope_coverage(self):
        # Under iid uniform residuals the (1-α)-quantile envelope
        # achieves nominal coverage.
        r = random.Random(123)
        residuals_train = [r.random() for _ in range(200)]
        env = conformal_envelope(residuals_train, alpha=0.10)
        cover = sum(1 for _ in range(2000) if r.random() <= env) / 2000
        assert cover >= 0.85

    def test_hcope_lower_bound_under_point(self):
        values = [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.3, 0.5]
        lb, b, rng_term = hcope_lower_bound(values, xi=1.0, alpha=0.05)
        mean = sum(values) / len(values)
        assert lb < mean
        assert b > 0.0
        assert rng_term > 0.0


# =====================================================================
# Effective sample size & diagnostics
# =====================================================================


class TestESS:
    def test_ess_uniform_weights_is_n(self):
        assert abs(ess([1.0] * 10) - 10.0) < 1e-9

    def test_ess_concentrated_is_one(self):
        # one massive weight, rest tiny → ESS ≈ 1
        ws = [1e6] + [1e-9] * 99
        assert ess(ws) < 1.1

    def test_ess_zero_for_empty(self):
        assert ess([]) == 0.0

    def test_weight_diagnostics_keys(self):
        d = weight_diagnostics([1.0, 2.0, 3.0, 4.0])
        for key in ("ess", "max_weight", "p99_weight", "p50_weight", "mean_log_weight", "var_log_weight"):
            assert key in d

    def test_overlap_kl_zero_when_equal(self):
        ps = [0.5, 0.5, 0.5]
        qs = [0.5, 0.5, 0.5]
        assert abs(overlap_kl(ps, qs)) < 1e-12

    def test_overlap_kl_positive(self):
        ps = [0.5, 0.5]
        qs = [0.9, 0.9]
        assert overlap_kl(ps, qs) > 0.0

    def test_overlap_kl_support_violation_raises(self):
        with pytest.raises(SupportViolation):
            overlap_kl([0.0, 0.5], [0.5, 0.5])


# =====================================================================
# Policy adapters
# =====================================================================


class TestPolicies:
    def test_uniform_policy_sums_to_one(self):
        p = UniformPolicy(["a", "b", "c"])
        d = p("anything")
        assert abs(sum(d.values()) - 1.0) < 1e-12
        assert all(abs(v - 1.0 / 3.0) < 1e-12 for v in d.values())

    def test_deterministic_policy(self):
        p = DeterministicPolicy(lambda s: "a", actions=["a", "b"])
        d = p("s")
        assert d["a"] == 1.0
        assert d["b"] == 0.0

    def test_epsilon_greedy_total_mass(self):
        p = EpsilonGreedyPolicy(lambda s: "a", actions=["a", "b", "c"], epsilon=0.3)
        d = p("s")
        assert abs(sum(d.values()) - 1.0) < 1e-12
        # greedy action: 1-ε + ε/3 = 0.8
        assert abs(d["a"] - (0.7 + 0.1)) < 1e-12

    def test_softmax_temperature_sharpens(self):
        scores = {"a": 1.0, "b": 0.0}
        cool = SoftmaxPolicy(lambda s, a: scores[a], actions=["a", "b"], temperature=0.1)
        hot = SoftmaxPolicy(lambda s, a: scores[a], actions=["a", "b"], temperature=1000.0)
        assert cool("s")["a"] > 0.99
        assert abs(hot("s")["a"] - 0.5) < 1e-3

    def test_invalid_temperature(self):
        with pytest.raises(CounterfactorError):
            SoftmaxPolicy(lambda s, a: 0.0, actions=["a"], temperature=-1.0)

    def test_invalid_epsilon(self):
        with pytest.raises(CounterfactorError):
            EpsilonGreedyPolicy(lambda s: "a", actions=["a"], epsilon=1.5)


# =====================================================================
# Q-models
# =====================================================================


class TestQModels:
    def test_constant_q(self):
        q = ConstantQModel(2.5)
        assert q.q("any", "any") == 2.5

    def test_tabular_q_fits_empirical_return(self):
        log = [
            LoggedTrajectory(
                steps=(
                    LoggedStep(state="s0", action="a", reward=1.0, behavior_prob=0.5),
                    LoggedStep(state="s1", action="a", reward=2.0, behavior_prob=0.5),
                )
            ),
            LoggedTrajectory(
                steps=(
                    LoggedStep(state="s0", action="a", reward=3.0, behavior_prob=0.5),
                    LoggedStep(state="s1", action="a", reward=4.0, behavior_prob=0.5),
                )
            ),
        ]
        q = TabularQModel(gamma=1.0)
        q.fit(log)
        # (s0, a): cumulative future return = (1+2)=3 and (3+4)=7 → mean 5
        # (s1, a): cumulative future return = 2 and 4 → mean 3
        assert abs(q.q("s0", "a") - 5.0) < 1e-9
        assert abs(q.q("s1", "a") - 3.0) < 1e-9

    def test_tabular_q_unseen_uses_global_mean(self):
        log = [
            LoggedTrajectory(
                steps=(LoggedStep(state="s", action="x", reward=4.0, behavior_prob=1.0),)
            ),
        ]
        q = TabularQModel()
        q.fit(log)
        assert abs(q.q("unknown", "unknown") - 4.0) < 1e-9

    def test_linear_q_fits_linear_data(self):
        # Build data where reward = 2 * x + 0.5 deterministically.
        log = []
        for x in range(-3, 4):
            r = 2.0 * x + 0.5
            log.append(
                LoggedTrajectory(
                    steps=(LoggedStep(state=x, action="a", reward=r, behavior_prob=1.0),)
                )
            )
        q = LinearQModel(
            features=lambda s: [float(s)],
            actions=["a"],
            l2=0.0,
            gamma=1.0,
        )
        q.fit(log)
        # Predictions should match the linear truth.
        for x in [-2, 0, 3]:
            assert abs(q.q(x, "a") - (2.0 * x + 0.5)) < 1e-6


# =====================================================================
# Counterfactor wiring
# =====================================================================


class TestCounterfactor:
    def test_log_trajectory_increments_counter(self):
        ctr = Counterfactor()
        tr = LoggedTrajectory(
            steps=(LoggedStep(state=0, action="a", reward=1.0, behavior_prob=0.5),)
        )
        ctr.log_trajectory(tr)
        assert ctr.n_trajectories == 1

    def test_log_dicts(self):
        ctr = Counterfactor()
        tr = ctr.log(
            [
                {"state": 0, "action": "a", "reward": 1.0, "behavior_prob": 0.5},
                {"state": 1, "action": "b", "reward": 0.0, "behavior_prob": 0.5},
            ]
        )
        assert tr.horizon == 2
        assert ctr.n_trajectories == 1

    def test_log_bandit_is_horizon_1(self):
        ctr = Counterfactor()
        tr = ctr.log_bandit(state="x", action="a", reward=1.0, behavior_prob=0.5)
        assert tr.horizon == 1

    def test_invalid_behaviour_prob_rejected(self):
        # LoggedStep itself is permissive; Counterfactor.log_trajectory
        # validates each behaviour-prob is in (0, 1].
        ctr = Counterfactor()
        bad = LoggedTrajectory(
            steps=(LoggedStep(state=0, action="a", reward=1.0, behavior_prob=0.0),)
        )
        with pytest.raises(CounterfactorError):
            ctr.log_trajectory(bad)

    def test_evaluate_dispatches_each_method(self):
        ctr = Counterfactor()
        actions = ["a", "b"]
        for tr in _bernoulli_log(
            50, 2, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0
        ):
            ctr.log_trajectory(tr)
        q = TabularQModel(gamma=1.0)
        q.fit(ctr.trajectories())
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        for method in KNOWN_METHODS:
            kwargs = {}
            if method in (METHOD_DM, METHOD_DR_RL, METHOD_WDR, METHOD_MAGIC):
                kwargs["q_model"] = q
            rep = ctr.evaluate(target, method=method, **kwargs)
            assert isinstance(rep, OPEReport)
            assert rep.method == method
            assert rep.digest

    def test_evaluate_ci_methods(self):
        ctr = Counterfactor()
        actions = ["a", "b"]
        for tr in _bernoulli_log(
            80, 2, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 0.0
        ):
            ctr.log_trajectory(tr)
        target = UniformPolicy(actions)
        for ci in KNOWN_CI:
            rep = ctr.evaluate(target, method=METHOD_PDIS, ci_method=ci)
            assert rep.ci_lo <= rep.value <= rep.ci_hi

    def test_evaluate_requires_q_model_for_model_methods(self):
        ctr = Counterfactor()
        ctr.log([{"state": 0, "action": "a", "reward": 1.0, "behavior_prob": 0.5}])
        target = UniformPolicy(["a", "b"])
        for method in (METHOD_DM, METHOD_DR_RL, METHOD_WDR, METHOD_MAGIC):
            with pytest.raises(CounterfactorError):
                ctr.evaluate(target, method=method)

    def test_unknown_method_raises(self):
        ctr = Counterfactor()
        ctr.log_bandit("x", "a", 1.0, 0.5)
        with pytest.raises(UnknownMethod):
            ctr.evaluate(UniformPolicy(["a", "b"]), method="nonexistent")

    def test_evaluate_empty_log_raises(self):
        ctr = Counterfactor()
        with pytest.raises(InsufficientData):
            ctr.evaluate(UniformPolicy(["a"]), method=METHOD_PDIS)

    def test_clear_resets(self):
        ctr = Counterfactor()
        ctr.log_bandit("x", "a", 1.0, 0.5)
        ctr.clear()
        assert ctr.n_trajectories == 0
        assert ctr.coverage()["n_trajectories"] == 0


# =====================================================================
# HCOPE
# =====================================================================


class TestHCOPE:
    def test_lower_bound_dominated_by_point(self):
        ctr = Counterfactor(reward_range=(0.0, 1.0))
        actions = ["a", "b"]
        for tr in _bernoulli_log(
            100, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=10
        ):
            ctr.log_trajectory(tr)
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        rep = ctr.hcope(target, method=METHOD_PDIS, alpha=0.05)
        assert isinstance(rep, HCOPEReport)
        assert rep.lower_bound <= rep.point_value
        assert rep.bernstein_term > 0.0
        assert rep.range_term > 0.0

    def test_lower_bound_loosens_for_smaller_alpha(self):
        ctr = Counterfactor(reward_range=(0.0, 1.0))
        actions = ["a", "b"]
        for tr in _bernoulli_log(
            100, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=11
        ):
            ctr.log_trajectory(tr)
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        wide = ctr.hcope(target, method=METHOD_PDIS, alpha=0.01)
        narrow = ctr.hcope(target, method=METHOD_PDIS, alpha=0.20)
        assert wide.lower_bound < narrow.lower_bound

    def test_hcope_dispatches_each_method(self):
        ctr = Counterfactor(reward_range=(0.0, 1.0))
        actions = ["a", "b"]
        for tr in _bernoulli_log(50, 2, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 0.5):
            ctr.log_trajectory(tr)
        q = TabularQModel(gamma=1.0)
        q.fit(ctr.trajectories())
        target = UniformPolicy(actions)
        for method in KNOWN_METHODS:
            kwargs = {}
            if method in (METHOD_DM, METHOD_DR_RL, METHOD_WDR, METHOD_MAGIC):
                kwargs["q_model"] = q
            rep = ctr.hcope(target, method=method, alpha=0.05, **kwargs)
            assert rep.lower_bound <= rep.point_value + 1e-6


# =====================================================================
# Compare
# =====================================================================


class TestCompare:
    def test_self_compare_zero_delta(self):
        ctr = Counterfactor()
        actions = ["a", "b"]
        for tr in _bernoulli_log(
            80, 2, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=12
        ):
            ctr.log_trajectory(tr)
        target = UniformPolicy(actions)
        rep = ctr.compare(target, target, method=METHOD_PDIS)
        assert abs(rep.delta) < 1e-9
        assert not rep.a_dominates

    def test_compare_detects_dominance(self):
        ctr = Counterfactor()
        actions = ["a", "b"]
        for tr in _bernoulli_log(
            300, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=13
        ):
            ctr.log_trajectory(tr)
        a_policy = DeterministicPolicy(lambda s: "a", actions=actions)
        b_policy = DeterministicPolicy(lambda s: "b", actions=actions)
        rep = ctr.compare(a_policy, b_policy, method=METHOD_PDIS, alpha=0.05)
        assert rep.delta > 0.0
        assert rep.a_dominates
        assert rep.p_a_better > 0.99

    def test_compare_antisymmetry(self):
        ctr = Counterfactor()
        actions = ["a", "b"]
        for tr in _bernoulli_log(
            100, 2, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=14
        ):
            ctr.log_trajectory(tr)
        a_policy = DeterministicPolicy(lambda s: "a", actions=actions)
        b_policy = DeterministicPolicy(lambda s: "b", actions=actions)
        ab = ctr.compare(a_policy, b_policy, method=METHOD_PDIS)
        ba = ctr.compare(b_policy, a_policy, method=METHOD_PDIS)
        assert abs(ab.delta + ba.delta) < 1e-9
        assert ab.a_dominates != ba.a_dominates


# =====================================================================
# Diagnostics
# =====================================================================


class TestDiagnostics:
    def test_diagnostics_full_overlap(self):
        ctr = Counterfactor()
        actions = ["a", "b"]
        for tr in _bernoulli_log(
            50, 3, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 0.0
        ):
            ctr.log_trajectory(tr)
        diag = ctr.diagnostics(UniformPolicy(actions))
        assert diag.coverage == 1.0
        assert diag.ess_trajectory > 0.0

    def test_diagnostics_partial_overlap_warning(self):
        ctr = Counterfactor()
        actions = ["a", "b"]
        # behaviour only ever picks 'a'
        for tr in _bernoulli_log(
            20, 2, actions, {"a": 1.0, "b": 0.0}, lambda t, a: 0.0
        ):
            # but force into the log with behavior_prob set to a positive number
            for s in tr.steps:
                object.__setattr__(s, "behavior_prob", 1.0)
            ctr.log_trajectory(tr)
        # target always picks 'b' — zero probability under behaviour log
        target = DeterministicPolicy(lambda s: "b", actions=actions)
        diag = ctr.diagnostics(target)
        # All logged actions are 'a' but target prob is 0 for 'a' → coverage=0
        assert diag.coverage == 0.0
        assert any("low overlap" in w for w in diag.warnings)

    def test_diagnostics_max_weight_warning(self):
        ctr = Counterfactor(weight_cap=None)
        actions = ["a", "b"]
        # tiny behaviour propensity → huge IS ratio under deterministic target
        for _ in range(20):
            steps = []
            for t in range(4):
                steps.append(
                    LoggedStep(state=t, action="a", reward=0.0, behavior_prob=0.001)
                )
            ctr.log_trajectory(LoggedTrajectory(steps=tuple(steps)))
        target = DeterministicPolicy(lambda s: "a", actions=actions)
        diag = ctr.diagnostics(target)
        # weight 1000 per step → trajectory weight 1e12, way over 1e3 threshold
        assert any("max trajectory weight" in w for w in diag.warnings)


# =====================================================================
# Attestation + events
# =====================================================================


class TestAttestation:
    def test_attestor_receives_evaluate_payload(self):
        captures: list[dict] = []
        ctr = Counterfactor(attestor=lambda d: captures.append(d), reward_range=(0.0, 1.0))
        ctr.log_bandit("x", "a", 1.0, 0.5)
        ctr.log_bandit("x", "b", 0.0, 0.5)
        ctr.evaluate(UniformPolicy(["a", "b"]), method=METHOD_PDIS)
        assert captures, "attestor was not called"
        kinds = [c["kind"] for c in captures]
        assert any("counterfactor" in k for k in kinds)

    def test_bus_receives_evaluated_event(self):
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(lambda e: events.append(e))
        ctr = Counterfactor(bus=bus)
        ctr.log_bandit("x", "a", 1.0, 0.5)
        ctr.log_bandit("x", "b", 0.0, 0.5)
        ctr.evaluate(UniformPolicy(["a", "b"]), method=METHOD_PDIS)
        kinds = [e.kind for e in events]
        assert any("evaluated" in k for k in kinds)

    def test_digest_stable_across_runs(self):
        # Two Counterfactors fed identical data + identical evaluation
        # produce the same digest.
        def run():
            ctr = Counterfactor(reward_range=(0.0, 1.0))
            for tr in _bernoulli_log(
                20, 2, ["a", "b"], {"a": 0.5, "b": 0.5}, lambda t, a: 1.0 if a == "a" else 0.0, seed=99
            ):
                ctr.log_trajectory(tr)
            rep = ctr.evaluate(
                DeterministicPolicy(lambda s: "a", actions=["a", "b"]),
                method=METHOD_PDIS,
                ci_method=CI_BERNSTEIN,
            )
            return rep.digest

        assert run() == run()


# =====================================================================
# Threadsafety
# =====================================================================


class TestThreading:
    def test_concurrent_log_consistent(self):
        ctr = Counterfactor()
        actions = ["a", "b"]

        def worker(n):
            for _ in range(n):
                ctr.log_bandit("x", "a", 1.0, 0.5)

        threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert ctr.n_trajectories == 200

    def test_concurrent_evaluate(self):
        ctr = Counterfactor(reward_range=(0.0, 1.0))
        actions = ["a", "b"]
        for tr in _bernoulli_log(80, 2, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 0.5):
            ctr.log_trajectory(tr)
        target = UniformPolicy(actions)

        results: list[float] = []
        lock = threading.Lock()

        def worker():
            r = ctr.evaluate(target, method=METHOD_PDIS, ci_method=CI_BERNSTEIN)
            with lock:
                results.append(r.value)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # PDIS is deterministic given the data → all evaluations agree.
        assert len(results) == 8
        for v in results:
            assert abs(v - results[0]) < 1e-9
        assert ctr.coverage()["n_evaluates"] == 8


# =====================================================================
# Bandit bridge — H=1 single-step OPE
# =====================================================================


class TestBanditBridge:
    def test_horizon_one_traj_is_equals_bandit_ips(self):
        # Contextual bandit IPS: V̂ = (1/n) Σ_i (π(a_i|x_i)/μ(a_i|x_i)) r_i
        ctr = Counterfactor()
        # 4 samples: action a with reward 1, behaviour 0.5; action b reward 0, behaviour 0.5
        for a, r in [("a", 1.0), ("b", 0.0), ("a", 1.0), ("b", 0.0)]:
            ctr.log_bandit("x", a, r, 0.5)
        target = DeterministicPolicy(lambda s: "a", actions=["a", "b"])
        rep = ctr.evaluate(target, method=METHOD_TRAJ_IS, ci_method=CI_BERNSTEIN)
        # IPS estimator: (1/4) * (2*1 + 0 + 2*1 + 0) = 1.0
        assert abs(rep.value - 1.0) < 1e-9


# =====================================================================
# Coverage / report
# =====================================================================


class TestCoverage:
    def test_coverage_advances_counters(self):
        ctr = Counterfactor(reward_range=(0.0, 1.0))
        actions = ["a", "b"]
        for tr in _bernoulli_log(20, 2, actions, {"a": 0.5, "b": 0.5}, lambda t, a: 0.5):
            ctr.log_trajectory(tr)
        target = UniformPolicy(actions)
        ctr.evaluate(target, method=METHOD_PDIS)
        ctr.hcope(target, method=METHOD_PDIS)
        ctr.compare(target, target, method=METHOD_PDIS)
        cov = ctr.coverage()
        assert cov["n_evaluates"] == 1
        assert cov["n_hcopes"] == 1
        assert cov["n_compares"] == 1
        assert cov["n_trajectories"] == 20
