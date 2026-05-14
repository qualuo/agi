"""Tests for the ActiveInferencer runtime primitive.

Covers:
  - numerical helpers (softmax, kl, entropy)
  - model validation
  - variational free energy: discrete + linear-gaussian
  - belief propagation B(a)
  - expected free energy decomposition + invariants
  - policy posterior softmax
  - planning / acting end-to-end
  - learning (Dirichlet A/B) + consolidation
  - Bayesian model averaging
  - bayesian surprise
  - PAC bounds (Hoeffding, empirical Bernstein)
  - composition / setters
  - threadsafety
  - quick two-armed bandit converges to winning arm
"""

from __future__ import annotations

import math
import random
import threading
from typing import Any

import pytest

from agi.active_inference import (
    AI_AGENT_REGISTERED,
    AI_PLANNED,
    ActiveInferenceError,
    ActiveInferencer,
    AgentSnapshot,
    CategoricalBelief,
    CoverageReport,
    DiscreteGenerativeModel,
    EFEReport,
    FreeEnergyReport,
    GaussianBelief,
    InsufficientData,
    InvalidModel,
    InvalidPolicy,
    KIND_DISCRETE,
    KIND_LINEAR_GAUSSIAN,
    LinearGaussianGenerativeModel,
    Policy,
    PolicySelection,
    SELECT_ARGMAX,
    SELECT_HABIT_ONLY,
    SELECT_RANDOM,
    SELECT_SOFTMAX,
    UnknownAgent,
    UnknownKind,
    UtilityBound,
    bayesian_model_average_belief,
    bayesian_surprise_discrete,
    empirical_bernstein_half_width,
    enumerate_policies,
    expected_free_energy_discrete,
    expected_free_energy_linear_gaussian,
    expected_utility_bound,
    hoeffding_half_width,
    policy_posterior,
    predict_belief_discrete,
    predicted_observation_distribution,
    quick_two_armed_bandit,
    variational_free_energy_discrete,
    variational_free_energy_linear_gaussian,
)


# ---------- helpers ----------


def _trivial_model() -> DiscreteGenerativeModel:
    """2 states, 2 obs, 2 actions; perfect observation; identity transition."""
    A = [[1.0, 0.0], [0.0, 1.0]]
    B = [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
    C = [0.0, 1.0]
    D = [0.5, 0.5]
    return DiscreteGenerativeModel(A=A, B=B, C=C, D=D)


def _noisy_model() -> DiscreteGenerativeModel:
    """Same shape but with observation noise; preference for outcome=1."""
    A = [[0.7, 0.2], [0.3, 0.8]]
    B = [[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]]
    C = [-1.0, 1.0]
    D = [0.5, 0.5]
    return DiscreteGenerativeModel(A=A, B=B, C=C, D=D)


# =====================================================================
# Model validation
# =====================================================================


class TestModelValidation:
    def test_discrete_valid_passes(self):
        m = _trivial_model()
        m.validate()

    def test_discrete_bad_A_columns(self):
        m = _trivial_model()
        m.A = [[0.5, 0.5], [0.0, 0.0]]  # second column zero
        with pytest.raises(InvalidModel):
            m.validate()

    def test_discrete_D_not_pmf(self):
        m = _trivial_model()
        m.D = [0.3, 0.3]
        with pytest.raises(InvalidModel):
            m.validate()

    def test_discrete_empty(self):
        with pytest.raises(InvalidModel):
            DiscreteGenerativeModel(A=[], B=[], C=[], D=[]).validate()

    def test_discrete_B_bad_action(self):
        m = _trivial_model()
        m.B = [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.0, 0.0]]]
        with pytest.raises(InvalidModel):
            m.validate()

    def test_linear_gaussian_valid(self):
        m = LinearGaussianGenerativeModel(
            F=[[1.0, 0.0], [0.0, 1.0]],
            b=[0.0, 0.0],
            H=[[1.0, 0.0]],
            Q_diag=[0.1, 0.1],
            R_diag=[0.5],
            mu0=[0.0, 0.0],
            Sigma0_diag=[1.0, 1.0],
            n_actions=1,
        )
        m.validate()

    def test_linear_gaussian_bad_dims(self):
        with pytest.raises(InvalidModel):
            LinearGaussianGenerativeModel(
                F=[[1.0, 0.0], [0.0, 1.0]],
                b=None,
                H=[[1.0]],  # wrong width
                Q_diag=[0.1, 0.1],
                R_diag=[0.5],
                mu0=[0.0, 0.0],
                Sigma0_diag=[1.0, 1.0],
            ).validate()

    def test_linear_gaussian_negative_var(self):
        with pytest.raises(InvalidModel):
            LinearGaussianGenerativeModel(
                F=[[1.0, 0.0], [0.0, 1.0]],
                b=None,
                H=[[1.0, 0.0]],
                Q_diag=[-0.1, 0.1],  # negative variance
                R_diag=[0.5],
                mu0=[0.0, 0.0],
                Sigma0_diag=[1.0, 1.0],
            ).validate()


# =====================================================================
# Variational free energy — discrete
# =====================================================================


class TestDiscreteVFE:
    def test_perfect_likelihood_collapses_belief(self):
        m = _trivial_model()
        post, report = variational_free_energy_discrete(
            m, prior=[0.5, 0.5], observation=0
        )
        # With A = identity, observing o=0 → q(s)=[1,0].
        assert post.probs[0] > 0.999
        assert post.probs[1] < 0.001
        # F = complexity - accuracy.  Complexity = KL(q || p) = log 2,
        # accuracy = log 1 = 0.
        assert abs(report.complexity - math.log(2.0)) < 1e-6
        assert abs(report.accuracy) < 1e-6
        assert abs(report.F - math.log(2.0)) < 1e-6

    def test_noisy_likelihood_updates_belief(self):
        m = _noisy_model()
        post, _ = variational_free_energy_discrete(m, prior=[0.5, 0.5], observation=1)
        # A[1] = (0.3, 0.8) → likes state 1.
        assert post.probs[1] > post.probs[0]
        # Bayes-optimal posterior is proportional to A[1,s] D[s]
        z = 0.5 * 0.3 + 0.5 * 0.8
        assert abs(post.probs[1] - (0.5 * 0.8) / z) < 1e-9

    def test_invalid_observation(self):
        m = _trivial_model()
        with pytest.raises(InvalidModel):
            variational_free_energy_discrete(m, prior=[0.5, 0.5], observation=5)

    def test_invalid_prior(self):
        m = _trivial_model()
        with pytest.raises(InvalidModel):
            variational_free_energy_discrete(m, prior=[0.3, 0.3], observation=0)

    def test_surprise_matches_evidence(self):
        # surprise should equal − log Z where Z = Σ_s A[o,s] D[s]
        m = _noisy_model()
        _, r = variational_free_energy_discrete(m, prior=[0.5, 0.5], observation=1)
        Z = 0.5 * 0.3 + 0.5 * 0.8
        assert abs(r.surprise - (-math.log(Z))) < 1e-9


class TestPredictBelief:
    def test_identity_transition(self):
        m = _trivial_model()
        b = CategoricalBelief(probs=[0.7, 0.3])
        b2 = predict_belief_discrete(m, b, 0)
        assert abs(b2.probs[0] - 0.7) < 1e-9 and abs(b2.probs[1] - 0.3) < 1e-9

    def test_swap_transition(self):
        m = _trivial_model()
        b = CategoricalBelief(probs=[0.7, 0.3])
        # action 1 swaps states
        b2 = predict_belief_discrete(m, b, 1)
        assert abs(b2.probs[0] - 0.3) < 1e-9 and abs(b2.probs[1] - 0.7) < 1e-9

    def test_invalid_action(self):
        m = _trivial_model()
        b = CategoricalBelief(probs=[0.5, 0.5])
        with pytest.raises(InvalidPolicy):
            predict_belief_discrete(m, b, 99)

    def test_predicted_observation(self):
        m = _noisy_model()
        b = CategoricalBelief(probs=[1.0, 0.0])
        qo = predicted_observation_distribution(m, b)
        # belief in state 0 → q(o=0)=0.7, q(o=1)=0.3
        assert abs(qo[0] - 0.7) < 1e-9 and abs(qo[1] - 0.3) < 1e-9


# =====================================================================
# Expected free energy
# =====================================================================


class TestDiscreteEFE:
    def test_efe_nonnegative_ambiguity(self):
        m = _noisy_model()
        b = CategoricalBelief(probs=[0.5, 0.5])
        p = Policy(actions=(0, 0))
        r = expected_free_energy_discrete(m, p, b)
        assert r.ambiguity >= -1e-9

    def test_efe_perfect_obs_zero_ambiguity(self):
        m = _trivial_model()  # A = identity → H(P(o|s))=0 for every s
        b = CategoricalBelief(probs=[0.5, 0.5])
        p = Policy(actions=(0, 0))
        r = expected_free_energy_discrete(m, p, b)
        assert abs(r.ambiguity) < 1e-9

    def test_efe_pragmatic_preference(self):
        m = _trivial_model()
        # Outcome 1 is preferred (C = [0, 1])
        b_high_for_one = CategoricalBelief(probs=[0.01, 0.99])
        b_low_for_one = CategoricalBelief(probs=[0.99, 0.01])
        p = Policy(actions=(0,))  # identity action
        r_high = expected_free_energy_discrete(m, p, b_high_for_one)
        r_low = expected_free_energy_discrete(m, p, b_low_for_one)
        # The belief that expects the preferred outcome has higher pragmatic
        # value (less risky).
        assert r_high.pragmatic_value > r_low.pragmatic_value
        assert r_high.G < r_low.G

    def test_efe_invalid_action(self):
        m = _trivial_model()
        b = CategoricalBelief(probs=[0.5, 0.5])
        p = Policy(actions=(99,))
        with pytest.raises(InvalidPolicy):
            expected_free_energy_discrete(m, p, b)

    def test_efe_epistemic_value_zero_when_no_ambiguity(self):
        # Identity A → q(s|o,π)=delta → epistemic value = entropy reduction
        m = _trivial_model()
        b = CategoricalBelief(probs=[0.5, 0.5])
        p = Policy(actions=(0,))
        r = expected_free_energy_discrete(m, p, b)
        # Under identity A from a uniform prior, epistemic value should be
        # log(2): we learn the state perfectly from the observation.
        assert abs(r.epistemic_value - math.log(2.0)) < 1e-6

    def test_efe_accumulates_over_horizon(self):
        m = _noisy_model()
        b = CategoricalBelief(probs=[0.5, 0.5])
        r1 = expected_free_energy_discrete(m, Policy(actions=(0,)), b)
        r2 = expected_free_energy_discrete(m, Policy(actions=(0, 0)), b)
        assert r2.ambiguity >= r1.ambiguity - 1e-9


class TestEnumerationPosterior:
    def test_enumerate(self):
        p = enumerate_policies(2, 3)
        assert len(p) == 8
        assert Policy(actions=(0, 1, 0)) in p

    def test_policy_posterior_sums_to_one(self):
        m = _noisy_model()
        b = CategoricalBelief(probs=[0.5, 0.5])
        efe = [
            expected_free_energy_discrete(m, p, b)
            for p in enumerate_policies(m.n_actions, 2)
        ]
        q = policy_posterior(efe, gamma=2.0)
        assert abs(sum(q) - 1.0) < 1e-9
        assert all(x >= 0 for x in q)

    def test_policy_posterior_low_G_dominates(self):
        m = _noisy_model()
        b = CategoricalBelief(probs=[0.5, 0.5])
        # Plant two policies with very different G manually.
        efe = [
            EFEReport(policy=Policy(actions=(0,)), G=10.0, risk=0, ambiguity=0,
                      epistemic_value=0, pragmatic_value=0),
            EFEReport(policy=Policy(actions=(1,)), G=0.0, risk=0, ambiguity=0,
                      epistemic_value=0, pragmatic_value=0),
        ]
        q = policy_posterior(efe, gamma=5.0)
        assert q[1] > q[0]

    def test_policy_posterior_habit(self):
        efe = [
            EFEReport(policy=Policy(actions=(0,)), G=0.0, risk=0, ambiguity=0,
                      epistemic_value=0, pragmatic_value=0),
            EFEReport(policy=Policy(actions=(1,)), G=0.0, risk=0, ambiguity=0,
                      epistemic_value=0, pragmatic_value=0),
        ]
        # Equal G → habit dominates.
        q = policy_posterior(efe, gamma=1.0, habit_E=[1.0, 9.0])
        assert q[1] > q[0]


# =====================================================================
# Linear-Gaussian inference + planning
# =====================================================================


def _lg_model() -> LinearGaussianGenerativeModel:
    return LinearGaussianGenerativeModel(
        F=[[1.0, 0.0], [0.0, 1.0]],
        b=[0.0, 0.0],
        H=[[1.0, 0.0]],
        Q_diag=[0.01, 0.01],
        R_diag=[0.1],
        mu0=[0.0, 0.0],
        Sigma0_diag=[1.0, 1.0],
        C=[0.0],
        n_actions=2,
        F_per_action=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.9, 0.1], [0.0, 1.0]],
        ],
    )


class TestLinearGaussianVFE:
    def test_kalman_update_pulls_belief(self):
        m = _lg_model()
        prior = GaussianBelief(mu=[0.0, 0.0], var_diag=[1.0, 1.0])
        post, _ = variational_free_energy_linear_gaussian(m, prior, [3.0])
        # The observed dim has its mean pulled toward 3.
        assert post.mu[0] > 0.5
        # Posterior variance is smaller than prior variance.
        assert post.var_diag[0] < prior.var_diag[0]

    def test_kalman_no_obs_dim_unaffected(self):
        m = _lg_model()
        prior = GaussianBelief(mu=[0.0, 0.0], var_diag=[1.0, 1.0])
        post, _ = variational_free_energy_linear_gaussian(m, prior, [3.0])
        # The un-observed dimension's mean is unchanged.
        assert abs(post.mu[1]) < 1e-9
        assert post.var_diag[1] == pytest.approx(1.0)

    def test_efe_lg_epistemic_positive(self):
        m = _lg_model()
        prior = GaussianBelief(mu=[0.0, 0.0], var_diag=[1.0, 1.0])
        p = Policy(actions=(0,))
        r = expected_free_energy_linear_gaussian(m, p, prior)
        assert r.epistemic_value > 0.0  # observing reduces variance

    def test_efe_lg_bad_obs_dim(self):
        m = _lg_model()
        prior = GaussianBelief(mu=[0.0, 0.0], var_diag=[1.0, 1.0])
        with pytest.raises(InvalidModel):
            variational_free_energy_linear_gaussian(m, prior, [1.0, 2.0])


# =====================================================================
# ActiveInferencer end-to-end
# =====================================================================


class TestInferencerLifecycle:
    def test_register_and_snapshot(self):
        inf = ActiveInferencer(random_seed=42)
        inf.register_agent("a", _trivial_model(), gamma=2.0, horizon=2)
        snap = inf.snapshot("a")
        assert snap.kind == KIND_DISCRETE
        assert snap.horizon == 2
        assert snap.gamma == 2.0
        assert snap.n_obs_seen == 0
        assert snap.belief_entropy == pytest.approx(math.log(2.0), abs=1e-9)

    def test_unknown_agent_raises(self):
        inf = ActiveInferencer()
        with pytest.raises(UnknownAgent):
            inf.snapshot("nope")
        with pytest.raises(UnknownAgent):
            inf.belief("nope")
        with pytest.raises(UnknownAgent):
            inf.step("nope", 0)

    def test_duplicate_name_raises(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        with pytest.raises(InvalidModel):
            inf.register_agent("a", _trivial_model())

    def test_remove_agent(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        inf.remove_agent("a")
        assert "a" not in inf.list_agents()
        inf.remove_agent("a")  # idempotent

    def test_register_bad_gamma(self):
        inf = ActiveInferencer()
        with pytest.raises(InvalidModel):
            inf.register_agent("a", _trivial_model(), gamma=0.0)

    def test_register_bad_horizon(self):
        inf = ActiveInferencer()
        with pytest.raises(InvalidModel):
            inf.register_agent("a", _trivial_model(), horizon=0)

    def test_unknown_kind_at_registration(self):
        inf = ActiveInferencer()
        with pytest.raises(UnknownKind):
            inf.register_agent("a", "not a model")  # type: ignore


class TestStepAndBelief:
    def test_step_discrete_updates_belief(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _noisy_model())
        r = inf.step("a", 1)
        assert isinstance(r, FreeEnergyReport)
        b = inf.belief("a")
        assert isinstance(b, CategoricalBelief)
        assert b.probs[1] > b.probs[0]
        assert inf.snapshot("a").n_obs_seen == 1

    def test_step_linear_gaussian(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _lg_model())
        r = inf.step("a", [2.0])
        b = inf.belief("a")
        assert isinstance(b, GaussianBelief)
        assert b.mu[0] > 0.1
        assert r.F == r.complexity - r.accuracy

    def test_repeated_observation_concentrates(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _noisy_model())
        h0 = inf.snapshot("a").belief_entropy
        for _ in range(8):
            inf.step("a", 1)
        h1 = inf.snapshot("a").belief_entropy
        assert h1 < h0  # entropy decreases with information

    def test_observe_alias(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        r = inf.observe("a", 0)
        assert isinstance(r, FreeEnergyReport)


class TestPlanAndAct:
    def test_plan_enumerates(self):
        inf = ActiveInferencer(random_seed=0)
        inf.register_agent("a", _trivial_model(), horizon=2)
        sel = inf.plan("a", horizon=2)
        assert len(sel.candidates) == 4  # 2 actions ^ 2 horizon
        assert abs(sum(sel.q_pi) - 1.0) < 1e-9
        assert 0 <= sel.best < 4

    def test_plan_with_explicit_candidates(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model(), horizon=2)
        cands = [Policy(actions=(0, 1)), Policy(actions=(1, 0))]
        sel = inf.plan("a", horizon=2, candidate_policies=cands)
        assert len(sel.candidates) == 2

    def test_plan_too_big_raises(self):
        # Build a model with many actions to exceed enumeration limit.
        m = _trivial_model()
        m.B = m.B + m.B + m.B + m.B + m.B  # 10 actions
        inf = ActiveInferencer()
        inf.register_agent("a", m, horizon=4)
        with pytest.raises(InvalidPolicy):
            inf.plan("a", horizon=4)

    def test_plan_invalid_policy_horizon(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model(), horizon=2)
        with pytest.raises(InvalidPolicy):
            inf.plan(
                "a",
                horizon=2,
                candidate_policies=[Policy(actions=(0,))],
            )

    def test_plan_invalid_action(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model(), horizon=1)
        with pytest.raises(InvalidPolicy):
            inf.plan(
                "a", horizon=1, candidate_policies=[Policy(actions=(99,))]
            )

    def test_act_advances_belief(self):
        inf = ActiveInferencer(random_seed=0)
        inf.register_agent("a", _trivial_model(), horizon=1)
        b0 = inf.belief("a")
        a = inf.act("a", mode=SELECT_ARGMAX)
        assert isinstance(a, int)
        b1 = inf.belief("a")
        # action 1 swaps the belief; action 0 does not.
        if a == 0:
            assert b1.probs == pytest.approx(b0.probs)
        else:
            assert b1.probs[0] == pytest.approx(b0.probs[1])

    def test_act_unknown_mode(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        with pytest.raises(InvalidPolicy):
            inf.act("a", mode="bogus")

    def test_act_random(self):
        inf = ActiveInferencer(random_seed=0)
        inf.register_agent("a", _trivial_model(), horizon=1)
        a = inf.act("a", mode=SELECT_RANDOM)
        assert 0 <= a < 2

    def test_act_habit_only(self):
        inf = ActiveInferencer(random_seed=0)
        inf.register_agent(
            "a",
            _trivial_model(),
            horizon=1,
            habit_E=[1.0, 1000.0],  # strongly prefer action 1
        )
        a = inf.act("a", mode=SELECT_HABIT_ONLY)
        assert a == 1

    def test_act_habit_only_without_habit(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model(), horizon=1)
        with pytest.raises(InvalidPolicy):
            inf.act("a", mode=SELECT_HABIT_ONLY)


class TestLearning:
    def test_learn_A_counts(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _noisy_model())
        # Observe outcome 0 while belief is concentrated on state 0
        inf.learn(
            "a",
            observation=0,
            prev_state_belief=[1.0, 0.0],
            lr=1.0,
        )
        inf.learn(
            "a",
            observation=0,
            prev_state_belief=[1.0, 0.0],
            lr=1.0,
        )
        snap = inf.snapshot("a")
        assert snap.n_obs_seen == 0  # learn does not advance step counter
        # consolidate moves counts into A
        inf.consolidate_learning("a", smoothing=0.5)
        # Now P(o=0|s=0) should have risen.
        # We re-run a step with prior centred on s=0 and outcome 0 to verify.
        report = inf.step("a", 0)
        assert report.F < math.log(2.0) + 1e-6  # surprise dropped

    def test_learn_B_counts(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        inf.learn(
            "a",
            action=0,
            prev_state_belief=[1.0, 0.0],
            next_state_belief=[0.0, 1.0],
        )
        inf.consolidate_learning("a")
        # The agent should now believe action 0 sends s=0 to s=1
        # (overcoming the original identity prior).
        b0 = CategoricalBelief(probs=[1.0, 0.0])
        b1 = predict_belief_discrete(
            inf._agent("a").model_discrete, b0, 0  # type: ignore
        )
        assert b1.probs[1] > b1.probs[0]

    def test_learn_bad_args(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        with pytest.raises(InvalidModel):
            inf.learn("a", observation=99, prev_state_belief=[1.0, 0.0])
        with pytest.raises(InvalidModel):
            inf.learn("a", action=99, prev_state_belief=[1.0, 0.0], next_state_belief=[0.0, 1.0])

    def test_learn_wrong_kind(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _lg_model())
        with pytest.raises(UnknownKind):
            inf.learn("a", observation=0, prev_state_belief=[1.0, 0.0])
        with pytest.raises(UnknownKind):
            inf.consolidate_learning("a")


class TestComposition:
    def test_set_likelihood(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        inf.set_likelihood("a", [[0.9, 0.1], [0.1, 0.9]])
        assert inf._agent("a").model_discrete.A == [[0.9, 0.1], [0.1, 0.9]]  # type: ignore

    def test_set_likelihood_bad(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        with pytest.raises(InvalidModel):
            inf.set_likelihood("a", [[0.5, 0.5], [0.0, 0.0]])

    def test_set_transition(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        new_B = [[[0.7, 0.3], [0.3, 0.7]], [[0.4, 0.6], [0.6, 0.4]]]
        inf.set_transition("a", new_B)
        assert inf._agent("a").model_discrete.B == new_B  # type: ignore

    def test_set_preferences(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        inf.set_preferences("a", [1.0, -1.0])
        m = inf._agent("a").model_discrete
        assert m.C == [1.0, -1.0]  # type: ignore

    def test_set_preferences_time_varying(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        inf.set_preferences("a", [[0.0, 1.0], [1.0, 0.0]])
        m = inf._agent("a").model_discrete
        assert m.preferences_at(0) == [0.0, 1.0]  # type: ignore
        assert m.preferences_at(1) == [1.0, 0.0]  # type: ignore
        assert m.preferences_at(99) == [1.0, 0.0]  # type: ignore  # clamp

    def test_set_prior_resets_belief(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        inf.set_prior("a", [0.9, 0.1])
        b = inf.belief("a")
        assert b.probs == pytest.approx([0.9, 0.1])


class TestBMA:
    def test_bma_returns_mixture(self):
        b1 = CategoricalBelief(probs=[0.9, 0.1])
        b2 = CategoricalBelief(probs=[0.1, 0.9])
        out = bayesian_model_average_belief([b1, b2], log_evidence=[0.0, 0.0])
        # Equal weight → ~uniform mixture
        assert abs(out.probs[0] - 0.5) < 1e-9

    def test_bma_skew_with_evidence(self):
        b1 = CategoricalBelief(probs=[1.0, 0.0])
        b2 = CategoricalBelief(probs=[0.0, 1.0])
        out = bayesian_model_average_belief([b1, b2], log_evidence=[10.0, 0.0])
        assert out.probs[0] > 0.99

    def test_bma_dimension_mismatch(self):
        b1 = CategoricalBelief(probs=[0.5, 0.5])
        b2 = CategoricalBelief(probs=[0.3, 0.3, 0.4])
        with pytest.raises(InvalidModel):
            bayesian_model_average_belief([b1, b2], log_evidence=[0.0, 0.0])

    def test_inferencer_bma(self):
        inf = ActiveInferencer()
        inf.register_agent("a1", _noisy_model())
        inf.register_agent("a2", _noisy_model())
        inf.step("a1", 1)
        inf.step("a2", 0)
        out = inf.bayesian_model_average(["a1", "a2"])
        assert isinstance(out, CategoricalBelief)

    def test_bma_unknown_agent(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        with pytest.raises(UnknownAgent):
            inf.bayesian_model_average(["a", "b"])


class TestBayesianSurprise:
    def test_zero_when_likelihood_uninformative(self):
        m = _trivial_model()
        m.A = [[0.5, 0.5], [0.5, 0.5]]
        s = bayesian_surprise_discrete(m, [0.3, 0.7], 0)
        assert s == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_likelihood_informative(self):
        m = _noisy_model()
        s = bayesian_surprise_discrete(m, [0.5, 0.5], 1)
        assert s > 0.0


# =====================================================================
# PAC bounds + counterfactuals
# =====================================================================


class TestBounds:
    def test_hoeffding_width(self):
        w = hoeffding_half_width(1000, delta=0.05, range_=1.0)
        assert w > 0
        assert w < 0.1  # should be tight at n=1000

    def test_hoeffding_invalid(self):
        with pytest.raises(InsufficientData):
            hoeffding_half_width(0, delta=0.05)
        with pytest.raises(ValueError):
            hoeffding_half_width(10, delta=0.0)
        with pytest.raises(ValueError):
            hoeffding_half_width(10, delta=0.05, range_=-1)

    def test_empirical_bernstein_low_variance_tighter(self):
        # Constant samples → bernstein gives smaller width
        const = [0.5] * 100
        eps_hoef = hoeffding_half_width(100, delta=0.05)
        eps_eb = empirical_bernstein_half_width(const, delta=0.05)
        assert eps_eb < eps_hoef

    def test_empirical_bernstein_min_samples(self):
        with pytest.raises(InsufficientData):
            empirical_bernstein_half_width([0.5], delta=0.05)

    def test_expected_utility_bound_report(self):
        b = expected_utility_bound([0.5] * 100, delta=0.05)
        assert isinstance(b, UtilityBound)
        assert b.n == 100
        assert b.mean == pytest.approx(0.5)
        assert b.lcb <= b.mean <= b.ucb
        assert b.half_width >= 0

    def test_expected_utility_bound_hoeffding(self):
        b = expected_utility_bound([0.3] * 100, delta=0.1, method="hoeffding")
        assert b.method == "hoeffding"

    def test_expected_utility_bound_falls_back(self):
        b = expected_utility_bound([0.5], delta=0.1, method="empirical_bernstein")
        assert b.method == "hoeffding"  # fell back

    def test_expected_utility_bound_zero(self):
        with pytest.raises(InsufficientData):
            expected_utility_bound([], delta=0.05)

    def test_expected_utility_bound_unknown_method(self):
        with pytest.raises(ValueError):
            expected_utility_bound([0.5] * 5, delta=0.1, method="bogus")


class TestCounterfactual:
    def test_counterfactual_rollouts(self):
        inf = ActiveInferencer(random_seed=0)
        inf.register_agent("a", _trivial_model())
        utils = inf.counterfactual(
            "a", Policy(actions=(0, 0)), n_rollouts=50
        )
        assert len(utils) == 50

    def test_counterfactual_lg_raises(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _lg_model())
        with pytest.raises(UnknownKind):
            inf.counterfactual("a", Policy(actions=(0,)), n_rollouts=2)

    def test_expected_utility_bound_method(self):
        inf = ActiveInferencer(random_seed=0)
        inf.register_agent("a", _trivial_model())
        b = inf.expected_utility_bound(
            "a", Policy(actions=(0, 0)), n_rollouts=200, delta=0.05
        )
        assert isinstance(b, UtilityBound)
        assert b.n == 200


# =====================================================================
# Convergence: two-armed bandit picks the good arm
# =====================================================================


class TestQuickBandit:
    def test_bandit_picks_good_arm(self):
        # Arm 1 has p=0.9, arm 0 has p=0.1.  Over 50 simulated trials the
        # agent should choose arm 1 most of the time once it has seen a
        # few outcomes from each arm.
        true_p = (0.1, 0.9)
        inf, name = quick_two_armed_bandit(
            arm_means=true_p, horizon=2, gamma=8.0, random_seed=7
        )
        rng = random.Random(0)
        chosen = []
        for _ in range(60):
            a = inf.act(name, mode=SELECT_ARGMAX, advance_belief=True)
            # simulate outcome from the true arm
            o = 1 if rng.random() < true_p[a] else 0
            inf.step(name, o)
            chosen.append(a)
        # After a few exploration trials, the agent should prefer arm 1.
        n_arm_1 = sum(1 for a in chosen[20:] if a == 1)
        assert n_arm_1 >= 25  # at least 25 of the last 40 trials


# =====================================================================
# Coverage + cleared
# =====================================================================


class TestCoverage:
    def test_coverage_counts(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model(), horizon=2)
        inf.step("a", 0)
        inf.plan("a", horizon=2)
        inf.act("a", mode=SELECT_ARGMAX)
        cov = inf.coverage()
        assert isinstance(cov, CoverageReport)
        assert cov.agents == 1
        assert cov.inferences >= 1
        assert cov.plans >= 2  # act() internally calls plan() once
        assert cov.acts == 1

    def test_clear(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model())
        inf.step("a", 0)
        inf.clear()
        assert inf.list_agents() == []
        cov = inf.coverage()
        assert cov.agents == 0
        assert cov.inferences == 0


# =====================================================================
# Threadsafety
# =====================================================================


class TestThreadsafety:
    def test_concurrent_steps(self):
        inf = ActiveInferencer(random_seed=1)
        for i in range(4):
            inf.register_agent(f"a{i}", _noisy_model())
        errors: list[Any] = []

        def worker(idx: int) -> None:
            try:
                for _ in range(20):
                    inf.step(f"a{idx}", idx % 2)
                    inf.plan(f"a{idx}", horizon=2)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert inf.coverage().inferences == 80


# =====================================================================
# Event bus integration (smoke)
# =====================================================================


class TestEventBus:
    def test_emits_events_when_bus_given(self):
        recorded: list[tuple[str, dict]] = []

        class _Bus:
            def publish(self, event: Any) -> None:
                recorded.append((event.kind, dict(event.data)))

        try:
            from agi.events import Event  # noqa
        except Exception:
            pytest.skip("agi.events not importable")
        inf = ActiveInferencer(bus=_Bus())
        inf.register_agent("a", _trivial_model())
        inf.step("a", 0)
        kinds = [k for (k, _) in recorded]
        assert "active_inference.started" in kinds
        assert "active_inference.agent_registered" in kinds
        assert "active_inference.inferred" in kinds


# =====================================================================
# Attestation receipts
# =====================================================================


class TestAttestation:
    def test_planning_returns_digest(self):
        inf = ActiveInferencer()
        inf.register_agent("a", _trivial_model(), horizon=2)
        sel = inf.plan("a", horizon=2)
        assert isinstance(sel.digest, str)
        assert len(sel.digest) == 64  # sha256 hex

    def test_attestor_receives_records(self):
        recs: list[tuple[str, dict]] = []

        class _Attestor:
            def record(self, kind: str, payload: dict) -> None:
                recs.append((kind, dict(payload)))

        inf = ActiveInferencer(attestor=_Attestor())
        inf.register_agent("a", _trivial_model())
        inf.step("a", 0)
        inf.plan("a", horizon=1)
        kinds = {k for (k, _) in recs}
        assert "active_inference.agent_registered" in kinds
        assert "active_inference.inferred" in kinds
        assert "active_inference.planned" in kinds
