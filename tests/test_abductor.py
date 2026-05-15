"""Tests for the Abductor runtime primitive — Bayesian abductive inference."""
from __future__ import annotations

import math
import random

import pytest

from agi.abductor import (
    ABDUCTOR_AVERAGED,
    ABDUCTOR_CLEARED,
    ABDUCTOR_CONTRASTED,
    ABDUCTOR_DESIGNED,
    ABDUCTOR_OBSERVED,
    ABDUCTOR_REGISTERED,
    ABDUCTOR_REPORTED,
    ABDUCTOR_SCORED,
    ABDUCTOR_SELECTED,
    ABDUCTOR_STARTED,
    Abductor,
    AbductorError,
    AbductorReport,
    BERNOULLI_BETA,
    CATEGORICAL_DIRICHLET,
    CUSTOM_POINT,
    Contrastive,
    EXPONENTIAL_GAMMA,
    EProcess,
    GAUSSIAN_KNOWN_VAR,
    GAUSSIAN_NIG,
    HypothesisSpec,
    IdentifiabilityReport,
    InformationGain,
    InsufficientData,
    InvalidHypothesis,
    InvalidObservation,
    JEFFREYS_DECISIVE,
    JEFFREYS_INSUBSTANTIAL,
    JEFFREYS_STRONG,
    JEFFREYS_SUBSTANTIAL,
    JEFFREYS_VERY_STRONG,
    KNOWN_EVENTS,
    KNOWN_HYPOTHESES,
    KNOWN_SELECTORS,
    POINT_BERNOULLI,
    POINT_CATEGORICAL,
    POINT_GAUSSIAN,
    POINT_POISSON,
    POISSON_GAMMA,
    Posterior,
    RobustnessReport,
    SELECT_MAP,
    SELECT_MIN_RISK,
    Selection,
    UnknownHypothesis,
    UnknownMethod,
    abductor_from_spec,
    empirical_bernstein_half_width,
    hoeffding_half_width,
    jeffreys_label,
    quick_two_hypothesis_coin,
    ville_threshold,
)


# -----------------------------------------------------------------------
# Basic structural and registration tests
# -----------------------------------------------------------------------


def test_constants_complete():
    assert {"point_bernoulli", "bernoulli_beta", "poisson_gamma",
            "gaussian_nig", "categorical_dirichlet"} <= KNOWN_HYPOTHESES
    assert SELECT_MAP in KNOWN_SELECTORS
    assert SELECT_MIN_RISK in KNOWN_SELECTORS
    for k in (ABDUCTOR_STARTED, ABDUCTOR_REGISTERED, ABDUCTOR_OBSERVED,
              ABDUCTOR_SCORED, ABDUCTOR_SELECTED, ABDUCTOR_AVERAGED,
              ABDUCTOR_CONTRASTED, ABDUCTOR_DESIGNED, ABDUCTOR_REPORTED,
              ABDUCTOR_CLEARED):
        assert k in KNOWN_EVENTS


def test_register_validates_kind_and_params():
    abd = Abductor()
    with pytest.raises(UnknownHypothesis):
        abd.register("h", "no_such_kind")
    with pytest.raises(InvalidHypothesis):
        abd.register("h", POINT_BERNOULLI, p=1.5)
    with pytest.raises(InvalidHypothesis):
        abd.register("h", POINT_BERNOULLI, p=0.0)
    with pytest.raises(InvalidHypothesis):
        abd.register("", POINT_BERNOULLI, p=0.5)
    abd.register("h", POINT_BERNOULLI, p=0.5)
    with pytest.raises(InvalidHypothesis):
        abd.register("h", POINT_BERNOULLI, p=0.5)  # duplicate
    assert "h" in abd
    assert len(abd) == 1


def test_register_emits_event_and_advances_fingerprint():
    abd = Abductor()
    before = abd.fingerprint
    abd.register("h", POINT_BERNOULLI, p=0.5)
    after = abd.fingerprint
    assert before != after
    kinds = [e["kind"] for e in abd.events()]
    assert ABDUCTOR_STARTED in kinds
    assert ABDUCTOR_REGISTERED in kinds


def test_observe_rejects_string_data():
    abd = Abductor()
    abd.register("h", POINT_BERNOULLI, p=0.5)
    with pytest.raises(InvalidObservation):
        abd.observe("011")  # type: ignore[arg-type]


def test_observe_before_register():
    abd = Abductor()
    with pytest.raises(AbductorError):
        abd.observe([0, 1])


# -----------------------------------------------------------------------
# Posterior computation — point hypotheses
# -----------------------------------------------------------------------


def test_posterior_point_bernoulli_unanimous_data_favors_loaded_coin():
    abd = Abductor()
    abd.register("fair", POINT_BERNOULLI, p=0.5)
    abd.register("loaded", POINT_BERNOULLI, p=0.9)
    abd.observe([1] * 10)
    post = abd.posterior()
    p = post.posterior_probs()
    assert p["loaded"] > p["fair"]
    # With 10 ones: p(D|loaded)=0.9^10 ≈ 0.349; p(D|fair) = 0.5^10 ≈ 9.77e-4.
    # Equal priors → posterior ratio ≈ 357.
    ratio = p["loaded"] / p["fair"]
    assert 300 < ratio < 400


def test_posterior_normalizes_to_one():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.3)
    abd.register("b", POINT_BERNOULLI, p=0.7)
    abd.register("c", POINT_BERNOULLI, p=0.5)
    abd.observe([1, 0, 1, 1, 0])
    post = abd.posterior()
    total = sum(post.posterior_probs().values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_posterior_under_zero_observations_equals_prior():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5, prior_weight=3.0)
    abd.register("b", POINT_BERNOULLI, p=0.5, prior_weight=1.0)
    post = abd.posterior()
    p = post.posterior_probs()
    assert math.isclose(p["a"], 0.75, abs_tol=1e-9)
    assert math.isclose(p["b"], 0.25, abs_tol=1e-9)


def test_prior_weights_propagate():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5, prior_weight=1.0)
    abd.register("b", POINT_BERNOULLI, p=0.5, prior_weight=99.0)
    abd.observe([0, 1, 0, 1, 1])
    post = abd.posterior()
    p = post.posterior_probs()
    # Likelihoods identical; posterior should track prior.
    assert p["b"] > 0.95
    assert math.isclose(p["a"] + p["b"], 1.0, abs_tol=1e-9)


# -----------------------------------------------------------------------
# Posterior computation — conjugate families (closed-form check)
# -----------------------------------------------------------------------


def test_bernoulli_beta_closed_form_evidence():
    """log p(D | Beta(1,1)) for D = [1,0] equals log B(2,2)/B(1,1) = log(1/6) − log(1) − ?

    With Beta(1,1) prior (= uniform) and one 1 and one 0:
        p(D) = B(2,2) / B(1,1) = (Γ(2)Γ(2)/Γ(4)) / (Γ(1)Γ(1)/Γ(2)) = (1·1/6) / (1·1/1) = 1/6.
    """
    abd = Abductor()
    abd.register("u", BERNOULLI_BETA, alpha=1.0, beta=1.0)
    abd.observe([1, 0])
    assert math.isclose(math.exp(abd.log_evidence("u")), 1.0 / 6.0, abs_tol=1e-9)


def test_categorical_dirichlet_evidence_2outcomes_matches_beta_bernoulli():
    """The Dirichlet-multinomial reduces to Beta-Bernoulli for k=2."""
    abd_a = Abductor()
    abd_a.register("h", BERNOULLI_BETA, alpha=1.0, beta=1.0)
    abd_a.observe([1, 0, 1, 0, 1])
    abd_b = Abductor()
    abd_b.register("h", CATEGORICAL_DIRICHLET, concentration=[1.0, 1.0])
    # Categorical uses index 0/1 too, treat 0 as 0 and 1 as 1.
    abd_b.observe([1, 0, 1, 0, 1])
    assert math.isclose(abd_a.log_evidence("h"), abd_b.log_evidence("h"), abs_tol=1e-9)


def test_gaussian_known_var_evidence_decreases_with_extreme_data():
    abd = Abductor()
    abd.register("near0", GAUSSIAN_KNOWN_VAR, mu0=0.0, tau0=1.0, sigma=1.0)
    abd.register("near5", GAUSSIAN_KNOWN_VAR, mu0=5.0, tau0=1.0, sigma=1.0)
    abd.observe([4.8, 5.1, 4.9, 5.05])  # data clearly near 5
    p = abd.posterior().posterior_probs()
    assert p["near5"] > p["near0"]


def test_gaussian_nig_recovers_data_centered_posterior_mean():
    abd = Abductor()
    abd.register("vague", GAUSSIAN_NIG, mu0=0.0, kappa0=1e-3, alpha0=1.0, beta0=1.0)
    data = [10.0] * 50
    abd.observe(data)
    # posterior predictive mean should be near 10 with very weak prior
    pred = abd.predict()
    assert math.isclose(pred, 10.0, rel_tol=1e-2)


def test_poisson_gamma_evidence_finite_and_consistent():
    abd = Abductor()
    abd.register("a", POISSON_GAMMA, alpha=2.0, beta=1.0)
    abd.observe([3, 4, 2, 5, 3])
    e = abd.log_evidence("a")
    assert math.isfinite(e)


def test_exponential_gamma_evidence_decreases_for_wrong_prior():
    abd = Abductor()
    abd.register("fast", EXPONENTIAL_GAMMA, alpha=2.0, beta=0.1)   # prior rate small
    abd.register("slow", EXPONENTIAL_GAMMA, alpha=2.0, beta=10.0)  # prior rate small / much
    abd.observe([0.05, 0.04, 0.06, 0.03])  # short waits → high rate
    p = abd.posterior().posterior_probs()
    assert p["fast"] > p["slow"]


# -----------------------------------------------------------------------
# Selection — MAP and min-risk
# -----------------------------------------------------------------------


def test_select_map_returns_highest_posterior_hypothesis():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.1)
    abd.register("b", POINT_BERNOULLI, p=0.9)
    abd.observe([1, 1, 1, 1, 1, 1])  # clearly b
    sel = abd.select()
    assert sel.winner == "b"
    assert sel.runner_up == "a"
    assert sel.log_bayes_factor > 0.0
    assert sel.posterior_prob > 0.9


def test_select_records_jeffreys_label():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.1)
    abd.register("b", POINT_BERNOULLI, p=0.9)
    abd.observe([1] * 20)
    sel = abd.select()
    assert sel.jeffreys_label == JEFFREYS_DECISIVE


def test_select_unknown_method_raises():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    with pytest.raises(UnknownMethod):
        abd.select(method="banana")  # type: ignore[arg-type]


def test_select_min_risk_with_loss():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.5)
    abd.observe([0, 1])
    sel = abd.select(method=SELECT_MIN_RISK, loss={"a": 0.1, "b": 10.0})
    # Both hypotheses have equal evidence; min-risk picks the one with
    # smaller loss weighted by posterior.
    assert sel.winner == "a"
    assert sel.expected_loss is not None


def test_select_min_risk_requires_loss():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    with pytest.raises(AbductorError):
        abd.select(method=SELECT_MIN_RISK)


# -----------------------------------------------------------------------
# Bayes factor / weight of evidence
# -----------------------------------------------------------------------


def test_bayes_factor_symmetric_inverse():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.3)
    abd.register("b", POINT_BERNOULLI, p=0.7)
    abd.observe([1, 0, 1, 1])
    bf_ab = abd.bayes_factor("a", "b")
    bf_ba = abd.bayes_factor("b", "a")
    assert math.isclose(bf_ab * bf_ba, 1.0, rel_tol=1e-9)


def test_weight_of_evidence_unit_conversions():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.3)
    abd.register("b", POINT_BERNOULLI, p=0.7)
    abd.observe([1, 0, 1, 1])
    w_nat = abd.weight_of_evidence("a", "b", base="nat")
    w_bit = abd.weight_of_evidence("a", "b", base="bit")
    w_ban = abd.weight_of_evidence("a", "b", base="ban")
    assert math.isclose(w_bit, w_nat / math.log(2.0), abs_tol=1e-9)
    assert math.isclose(w_ban, w_nat / math.log(10.0), abs_tol=1e-9)


def test_weight_of_evidence_unknown_base():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.5)
    with pytest.raises(AbductorError):
        abd.weight_of_evidence("a", "b", base="dits")


# -----------------------------------------------------------------------
# Jeffreys scale
# -----------------------------------------------------------------------


def test_jeffreys_label_thresholds():
    assert jeffreys_label(0.0) == JEFFREYS_INSUBSTANTIAL
    assert jeffreys_label(0.4) == JEFFREYS_INSUBSTANTIAL
    assert jeffreys_label(0.5) == JEFFREYS_SUBSTANTIAL
    assert jeffreys_label(0.99) == JEFFREYS_SUBSTANTIAL
    assert jeffreys_label(1.0) == JEFFREYS_STRONG
    assert jeffreys_label(1.4) == JEFFREYS_STRONG
    assert jeffreys_label(1.5) == JEFFREYS_VERY_STRONG
    assert jeffreys_label(1.99) == JEFFREYS_VERY_STRONG
    assert jeffreys_label(2.0) == JEFFREYS_DECISIVE
    assert jeffreys_label(5.0) == JEFFREYS_DECISIVE
    assert jeffreys_label(-5.0) == JEFFREYS_DECISIVE


# -----------------------------------------------------------------------
# BMA / prediction
# -----------------------------------------------------------------------


def test_bma_predict_lies_between_hypothesis_means():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.1)
    abd.register("b", POINT_BERNOULLI, p=0.9)
    abd.observe([1, 0])  # 1 of each — posterior nearly uniform
    pred = abd.predict()
    assert 0.1 <= pred <= 0.9


def test_bma_predict_proba_sums_to_one_for_bernoulli():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.3)
    abd.register("b", POINT_BERNOULLI, p=0.7)
    abd.observe([1, 0, 1])
    p0 = abd.predict_proba(0)
    p1 = abd.predict_proba(1)
    assert math.isclose(p0 + p1, 1.0, abs_tol=1e-9)


def test_bma_average_emits_event():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.5)
    abd.observe([0, 1])
    abd.average("predictive_mean")
    assert any(e["kind"] == ABDUCTOR_AVERAGED for e in abd.events())


def test_bma_custom_functional():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.2)
    abd.register("b", POINT_BERNOULLI, p=0.8)
    abd.observe([1])
    val = abd.average(lambda spec, state: 0.0 if spec.name == "a" else 1.0)
    # The value should be the posterior prob of "b".
    p_b = abd.posterior().posterior_probs()["b"]
    assert math.isclose(val, p_b, abs_tol=1e-9)


# -----------------------------------------------------------------------
# Contrastive / counterfactual
# -----------------------------------------------------------------------


def test_contrastive_per_obs_cumulates_to_total_log_bf():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.3)
    abd.register("b", POINT_BERNOULLI, p=0.7)
    data = [1, 0, 1, 1, 0]
    abd.observe(data)
    c = abd.contrastive("a", "b", data=data)
    assert c.n_observations == 5
    # Cumulative sum of per-observation log-BF equals final log-BF.
    assert math.isclose(sum(c.per_obs_log_bf), c.final_log_bf, abs_tol=1e-9)
    # Final log-BF in the contrast equals direct calculation.
    assert math.isclose(c.final_log_bf, abd.log_bayes_factor("a", "b"), abs_tol=1e-9)


def test_contrastive_emits_event():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.6)
    abd.observe([1, 0])
    abd.contrastive("a", "b", data=[1, 0])
    assert any(e["kind"] == ABDUCTOR_CONTRASTED for e in abd.events())


def test_counterfactual_posterior_doesnt_mutate_state():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.3)
    abd.register("b", POINT_BERNOULLI, p=0.7)
    abd.observe([1, 1, 1])
    p_real = abd.posterior().posterior_probs()
    cf = abd.counterfactual_posterior([0, 0, 0])
    p_after = abd.posterior().posterior_probs()
    assert p_real == p_after
    p_cf = {n: math.exp(lp) for n, lp in zip(cf.names, cf.log_posteriors)}
    # Under reversed observations, hypothesis "a" (p=0.3 of 1) wins.
    assert p_cf["a"] > p_cf["b"]


# -----------------------------------------------------------------------
# Identifiability
# -----------------------------------------------------------------------


def test_identifiability_groups_indistinguishable_hypotheses():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.5)
    abd.register("c", POINT_BERNOULLI, p=0.9)
    abd.observe([1, 0, 1, 0])
    ident = abd.identifiability()
    flat = [n for cls in ident.classes for n in cls]
    assert set(flat) == {"a", "b", "c"}
    # a and b are exactly identical → same class.
    cls_set = [set(cls) for cls in ident.classes]
    assert {"a", "b"} in cls_set


def test_identifiability_distinguishes_when_likelihoods_differ():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.3)
    abd.register("b", POINT_BERNOULLI, p=0.7)
    abd.observe([1, 1, 0])
    ident = abd.identifiability()
    cls_set = [set(cls) for cls in ident.classes]
    assert {"a"} in cls_set or {"b"} in cls_set


# -----------------------------------------------------------------------
# Prior robustness
# -----------------------------------------------------------------------


def test_prior_robustness_returns_finite_kl_for_close_call():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.4)
    abd.register("b", POINT_BERNOULLI, p=0.6)
    abd.observe([1, 0, 1])  # marginal preference; not decisive
    rob = abd.prior_robustness()
    assert rob.current_winner in ("a", "b")
    assert math.isfinite(rob.max_kl_perturbation)
    assert rob.max_kl_perturbation >= 0.0


def test_prior_robustness_with_single_hypothesis_returns_infinity():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.observe([1])
    rob = abd.prior_robustness()
    assert rob.max_kl_perturbation == float("inf")


# -----------------------------------------------------------------------
# Expected information gain / next-experiment design
# -----------------------------------------------------------------------


def test_expected_information_gain_nonnegative_for_uniform_posterior():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.1)
    abd.register("b", POINT_BERNOULLI, p=0.9)
    ig = abd.expected_information_gain([0, 1])
    assert ig.expected_gain_nats >= 0.0
    assert math.isclose(sum(ig.per_outcome_prob.values()), 1.0, abs_tol=1e-9)


def test_expected_information_gain_zero_when_posterior_already_certain():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.99)
    abd.observe([1] * 500)  # decisive — b wins
    ig = abd.expected_information_gain([0, 1])
    # Posterior is already concentrated; a single new datum shouldn't
    # help much (≤ 0.01 nat is "negligible" for this state).
    assert ig.expected_gain_nats < 0.05


def test_design_next_experiment_picks_higher_eig():
    abd = Abductor()
    abd.register("a", POINT_CATEGORICAL, probs=[0.5, 0.5])
    abd.register("b", POINT_CATEGORICAL, probs=[0.95, 0.05])
    name, ig = abd.design_next_experiment({
        "binary": [0, 1],
        "ternary_collapsed": [0, 0, 0],  # never discriminates → EIG ≈ 0
    })
    assert name == "binary"
    assert ig.expected_gain_nats > 0.0


def test_design_next_experiment_requires_non_empty():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    with pytest.raises(AbductorError):
        abd.design_next_experiment({})


# -----------------------------------------------------------------------
# E-process (anytime-valid testing)
# -----------------------------------------------------------------------


def test_e_process_is_a_running_likelihood_ratio():
    abd = Abductor()
    abd.register("h0", POINT_BERNOULLI, p=0.5)
    abd.register("h1", POINT_BERNOULLI, p=0.9)
    abd.observe([1] * 5)
    e = abd.e_process("h1", "h0", delta=0.05)
    # 0.9^5 / 0.5^5 = 0.59049 / 0.03125 = 18.895
    assert math.isclose(e.e_value, 0.9 ** 5 / 0.5 ** 5, rel_tol=1e-9)
    assert e.threshold_log is not None
    # threshold = log(20), log_e = log(18.895) < log(20), so not yet crossed.
    assert e.crossed_at is None


def test_e_process_crosses_threshold_for_extreme_data():
    abd = Abductor()
    abd.register("h0", POINT_BERNOULLI, p=0.5)
    abd.register("h1", POINT_BERNOULLI, p=0.9)
    abd.observe([1] * 20)
    e = abd.e_process("h1", "h0", delta=0.05)
    assert e.crossed_at == 20


def test_ville_threshold_inverts_delta():
    assert math.isclose(ville_threshold(0.05), 20.0, abs_tol=1e-9)
    assert math.isclose(ville_threshold(0.5), 2.0, abs_tol=1e-9)


def test_e_process_rejects_invalid_delta():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.5)
    with pytest.raises(AbductorError):
        abd.e_process("a", "b", delta=0.0)
    with pytest.raises(AbductorError):
        abd.e_process("a", "b", delta=1.5)


# -----------------------------------------------------------------------
# PAC-Bayes / Hoeffding / empirical Bernstein
# -----------------------------------------------------------------------


def test_pac_bayes_bound_dominates_empirical_mean():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.5)
    abd.observe([1] * 50)
    bound = abd.pac_bayes_bound({"a": 0.2, "b": 0.3}, delta=0.05)
    # bound >= mean weighted by posterior
    p = abd.posterior().posterior_probs()
    mean = p["a"] * 0.2 + p["b"] * 0.3
    assert bound >= mean


def test_pac_bayes_requires_full_loss_map():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.register("b", POINT_BERNOULLI, p=0.5)
    abd.observe([1])
    with pytest.raises(AbductorError):
        abd.pac_bayes_bound({"a": 0.0})


def test_pac_bayes_rejects_out_of_range_loss():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.observe([1])
    with pytest.raises(AbductorError):
        abd.pac_bayes_bound({"a": 1.5})


def test_hoeffding_half_width_shrinks_with_n():
    h_10 = hoeffding_half_width(10, delta=0.05)
    h_1000 = hoeffding_half_width(1000, delta=0.05)
    assert h_10 > h_1000 > 0.0


def test_empirical_bernstein_zero_variance_shrinks_to_constant_term():
    # Variance = 0, so the leading sqrt term vanishes; only the b·log/3(n-1) term remains.
    n = 100
    h = empirical_bernstein_half_width(n, sample_variance=0.0, delta=0.05)
    assert h < 0.5  # extremely tight when variance is zero


def test_empirical_bernstein_rejects_small_n():
    with pytest.raises(InsufficientData):
        empirical_bernstein_half_width(1, sample_variance=0.1, delta=0.05)


def test_empirical_bernstein_returns_ci_on_observed_mean():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    lo, hi = abd.empirical_bernstein([0.4, 0.6, 0.5, 0.55, 0.45], delta=0.05)
    mean = 0.5
    assert lo < mean < hi


# -----------------------------------------------------------------------
# Custom point hypothesis
# -----------------------------------------------------------------------


def test_custom_point_hypothesis_evaluates_user_log_likelihood():
    log_lik = lambda x: math.log(0.7) if int(x) == 1 else math.log(0.3)
    abd = Abductor()
    abd.register("custom", CUSTOM_POINT, log_likelihood=log_lik, signature="biased_07")
    abd.register("fair", POINT_BERNOULLI, p=0.5)
    abd.observe([1, 1, 0, 1, 1])
    p = abd.posterior().posterior_probs()
    assert p["custom"] > p["fair"]


def test_custom_point_requires_signature():
    abd = Abductor()
    with pytest.raises(InvalidHypothesis):
        abd.register("c", CUSTOM_POINT, log_likelihood=lambda x: 0.0)


def test_custom_point_cannot_be_registered_after_observations():
    abd = Abductor()
    abd.register("h", POINT_BERNOULLI, p=0.5)
    abd.observe([1])
    with pytest.raises(InvalidHypothesis):
        abd.register(
            "late", CUSTOM_POINT, log_likelihood=lambda x: 0.0, signature="sig"
        )


# -----------------------------------------------------------------------
# Streaming / online behaviour
# -----------------------------------------------------------------------


def test_streaming_observe_matches_batch_observe():
    a = Abductor()
    a.register("a", POINT_BERNOULLI, p=0.3)
    a.register("b", POINT_BERNOULLI, p=0.7)
    data = [1, 0, 1, 1, 0, 1, 0]
    a.observe(data)
    p_batch = a.posterior().posterior_probs()
    b = Abductor()
    b.register("a", POINT_BERNOULLI, p=0.3)
    b.register("b", POINT_BERNOULLI, p=0.7)
    for x in data:
        b.observe([x])
    p_stream = b.posterior().posterior_probs()
    assert math.isclose(p_batch["a"], p_stream["a"], rel_tol=1e-9)
    assert math.isclose(p_batch["b"], p_stream["b"], rel_tol=1e-9)


def test_streaming_bernoulli_beta_matches_batch():
    a = Abductor()
    a.register("h", BERNOULLI_BETA, alpha=1.5, beta=2.5)
    data = [random.Random(0).randint(0, 1) for _ in range(20)]
    a.observe(data)
    b = Abductor()
    b.register("h", BERNOULLI_BETA, alpha=1.5, beta=2.5)
    for x in data:
        b.observe([x])
    assert math.isclose(a.log_evidence("h"), b.log_evidence("h"), abs_tol=1e-9)


def test_categorical_observation_out_of_range_raises():
    abd = Abductor()
    abd.register("h", POINT_CATEGORICAL, probs=[0.3, 0.4, 0.3])
    with pytest.raises(InvalidObservation):
        abd.observe([3])


def test_poisson_rejects_negative():
    abd = Abductor()
    abd.register("h", POINT_POISSON, lam=2.0)
    with pytest.raises(InvalidObservation):
        abd.observe([-1])


def test_gaussian_rejects_non_numeric():
    abd = Abductor()
    abd.register("h", POINT_GAUSSIAN, mu=0.0, sigma=1.0)
    with pytest.raises(InvalidObservation):
        abd.observe(["nope"])


# -----------------------------------------------------------------------
# Fingerprint / event log
# -----------------------------------------------------------------------


def test_fingerprint_is_path_dependent():
    a = Abductor()
    a.register("h", POINT_BERNOULLI, p=0.5)
    a.observe([1, 0])
    b = Abductor()
    b.register("h", POINT_BERNOULLI, p=0.5)
    b.observe([0, 1])
    # Different observation order → different fingerprint chain.
    assert a.fingerprint != b.fingerprint


def test_fingerprint_replays_identically():
    a = Abductor()
    a.register("h", POINT_BERNOULLI, p=0.5)
    a.observe([1, 0, 1])
    b = Abductor()
    b.register("h", POINT_BERNOULLI, p=0.5)
    b.observe([1, 0, 1])
    assert a.fingerprint == b.fingerprint


def test_clear_resets_fingerprint_to_genesis():
    a = Abductor()
    a.register("h", POINT_BERNOULLI, p=0.5)
    a.observe([1])
    fp1 = a.fingerprint
    a.clear()
    fp2 = a.fingerprint
    assert fp1 != fp2
    # Two abductors that have each been freshly cleared produce identical
    # fingerprints — the chain is deterministic and path-independent at this
    # state.
    b = Abductor()
    b.register("ignored", POINT_BERNOULLI, p=0.5)
    b.clear()
    assert a.fingerprint == b.fingerprint


# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------


def test_report_contains_posterior_and_selection():
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.3)
    abd.register("b", POINT_BERNOULLI, p=0.7)
    abd.observe([1, 1, 0])
    abd.posterior()
    abd.select()
    rep = abd.report()
    assert rep.n_observations == 3
    assert rep.posterior is not None
    assert rep.last_selection is not None
    assert rep.last_selection.winner in ("a", "b")
    assert rep.n_events > 0


def test_report_to_dict_is_jsonable():
    import json
    abd = Abductor()
    abd.register("a", POINT_BERNOULLI, p=0.5)
    abd.observe([1])
    abd.posterior()
    abd.select()
    rep_d = abd.report().to_dict()
    # Round-trip through JSON.
    s = json.dumps(rep_d)
    rep_d_2 = json.loads(s)
    assert rep_d_2["n_observations"] == 1


# -----------------------------------------------------------------------
# Spec factory
# -----------------------------------------------------------------------


def test_abductor_from_spec_round_trip():
    spec = {
        "hypotheses": [
            {"name": "fair", "kind": "point_bernoulli", "params": {"p": 0.5}},
            {"name": "biased", "kind": "bernoulli_beta",
             "params": {"alpha": 2.0, "beta": 2.0}, "prior_weight": 2.0},
        ],
    }
    abd = abductor_from_spec(spec)
    assert "fair" in abd
    assert "biased" in abd
    abd.observe([1, 1, 0])
    p = abd.posterior().posterior_probs()
    assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-9)


def test_abductor_from_spec_rejects_bad_shape():
    with pytest.raises(AbductorError):
        abductor_from_spec("not a dict")  # type: ignore[arg-type]
    with pytest.raises(InvalidHypothesis):
        abductor_from_spec({"hypotheses": [{"kind": "point_bernoulli"}]})


def test_quick_two_hypothesis_coin_smoke():
    abd = quick_two_hypothesis_coin()
    abd.observe([1] * 5)
    p = abd.posterior().posterior_probs()
    # "biased" (Beta(2,2)) should compete with "fair" — with 5 heads,
    # both are non-trivial; main check is normalisation + names.
    assert set(p.keys()) == {"fair", "biased"}
    assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-9)


# -----------------------------------------------------------------------
# Integration / composition smoke tests
# -----------------------------------------------------------------------


def test_full_workflow_diagnoses_skewed_coin():
    """End-to-end: register competing coins, observe, score, select,
    contrast, design next experiment, e-process at α=0.05."""
    abd = Abductor()
    abd.register("fair", POINT_BERNOULLI, p=0.5)
    abd.register("strongly_biased", POINT_BERNOULLI, p=0.95)
    abd.register("uniform_prior_biased", BERNOULLI_BETA, alpha=1.0, beta=1.0)
    abd.observe([1] * 9 + [0])
    post = abd.posterior()
    assert post.map_name() in ("strongly_biased", "uniform_prior_biased")
    sel = abd.select()
    assert sel.posterior_prob > 0.4
    c = abd.contrastive("strongly_biased", "fair", data=[1] * 9 + [0])
    assert c.final_log_bf > 0.0
    ig = abd.expected_information_gain([0, 1])
    assert ig.expected_gain_nats >= 0.0
    e = abd.e_process("strongly_biased", "fair", delta=0.05)
    assert e.e_value > 1.0
    rep = abd.report()
    assert rep.posterior is not None
