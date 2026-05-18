"""Tests for the Attributor runtime primitive — data attribution / influence functions."""
from __future__ import annotations

import math
import random

import pytest

from agi.attributor import (
    ATTRIBUTOR_CLEARED,
    ATTRIBUTOR_COUNTERFACTUAL,
    ATTRIBUTOR_DECISION_FLIP,
    ATTRIBUTOR_FIT,
    ATTRIBUTOR_GROUP_LOO,
    ATTRIBUTOR_INFLUENCE_COMPUTED,
    ATTRIBUTOR_REPORTED,
    ATTRIBUTOR_STARTED,
    ATTRIBUTOR_TRACIN_COMPUTED,
    Attributor,
    AttributorError,
    AttributorReport,
    BootstrapBand,
    COOKS_DISTANCE,
    CounterfactualReport,
    CUSTOM,
    ConvergenceError,
    DecisionFlipReport,
    DFBETAS_METHOD,
    EXACT_LOO,
    FitReport,
    INFLUENCE_FUNCTION,
    InfluenceReport,
    InsufficientData,
    InvalidData,
    InvalidQuery,
    KNOWN_EVENTS,
    KNOWN_KINDS,
    KNOWN_METHODS,
    KNOWN_QUERIES,
    LEVERAGE,
    LINEAR,
    LOGISTIC,
    LinearDiagnostics,
    NotFit,
    PRESS,
    QUERY_CUSTOM,
    QUERY_LOSS,
    QUERY_PARAMETER,
    QUERY_PREDICTION,
    RIDGE,
    STUDENTIZED,
    SingularMatrix,
    TRACIN_IDEAL,
    TRAK_METHOD,
    TracInReport,
    UnknownKind,
    UnknownMethod,
    UnknownQuery,
    attributor_from_spec,
    empirical_bernstein_half_width,
    hoeffding_half_width,
    quick_linear_attribution,
    quick_logistic_attribution,
)


# -----------------------------------------------------------------------
# Constants and structural tests
# -----------------------------------------------------------------------


def test_constants_known_sets():
    assert {LINEAR, LOGISTIC, RIDGE, CUSTOM} <= KNOWN_KINDS
    assert {EXACT_LOO, INFLUENCE_FUNCTION, COOKS_DISTANCE,
            DFBETAS_METHOD, LEVERAGE, STUDENTIZED, PRESS,
            TRACIN_IDEAL, TRAK_METHOD} <= KNOWN_METHODS
    assert {QUERY_LOSS, QUERY_PREDICTION, QUERY_PARAMETER,
            QUERY_CUSTOM} <= KNOWN_QUERIES
    for ev in (ATTRIBUTOR_STARTED, ATTRIBUTOR_FIT,
               ATTRIBUTOR_INFLUENCE_COMPUTED,
               ATTRIBUTOR_COUNTERFACTUAL, ATTRIBUTOR_DECISION_FLIP,
               ATTRIBUTOR_TRACIN_COMPUTED, ATTRIBUTOR_GROUP_LOO,
               ATTRIBUTOR_REPORTED, ATTRIBUTOR_CLEARED):
        assert ev in KNOWN_EVENTS


def test_fresh_attributor_has_clean_chain():
    a = Attributor()
    assert a.verify_chain()
    assert a.names() == []
    assert a.fingerprint != ""
    # Exactly one event after init: ATTRIBUTOR_STARTED.
    evs = a.events()
    assert len(evs) == 1
    assert evs[0].kind == ATTRIBUTOR_STARTED


def test_unknown_name_raises():
    a = Attributor()
    with pytest.raises(UnknownKind):
        a.leverage("nope")
    with pytest.raises(UnknownKind):
        a.influence("nope")


def test_unknown_kind_raises():
    a = Attributor()
    with pytest.raises(UnknownKind):
        a.fit("m", "no-such-kind", X=[[1.0]], y=[1.0])


def test_unknown_method_raises():
    a = quick_linear_attribution([[1, 1], [1, 2]], [1.0, 2.0])
    with pytest.raises(UnknownMethod):
        a.influence("model", method="no-such-method",
                    query=QUERY_PREDICTION, test_point=[1, 1])


def test_unknown_query_raises():
    a = quick_linear_attribution([[1, 1], [1, 2]], [1.0, 2.0])
    with pytest.raises(UnknownQuery):
        a.influence("model", query="no-such-query")


def test_concentration_helpers():
    # Hoeffding monotone in n, finite values
    h_small = hoeffding_half_width(10, delta=0.05)
    h_big = hoeffding_half_width(1000, delta=0.05)
    assert h_small > h_big > 0
    eb = empirical_bernstein_half_width(100, sample_variance=0.25, delta=0.05)
    assert eb > 0
    with pytest.raises(InsufficientData):
        hoeffding_half_width(0, delta=0.05)
    with pytest.raises(AttributorError):
        hoeffding_half_width(10, delta=1.5)


# -----------------------------------------------------------------------
# Linear regression: closed-form correctness against a hand-computed example
# -----------------------------------------------------------------------


def test_linear_fit_recovers_known_coefficients():
    # y = 1 + 0.5x  exactly
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
    y = [1.5, 2.0, 2.5, 3.0, 3.5]
    a = Attributor()
    fit = a.fit("clean", LINEAR, X=X, y=y)
    assert fit.kind == LINEAR
    assert abs(fit.theta[0] - 1.0) < 1e-5
    assert abs(fit.theta[1] - 0.5) < 1e-5
    assert fit.residual_sum_squares < 1e-9
    assert fit.converged
    assert fit.iterations == 1


def test_linear_fit_intercept_only_recovers_mean():
    X = [[1.0], [1.0], [1.0], [1.0]]
    y = [1.0, 2.0, 3.0, 4.0]
    fit = quick_linear_attribution(X, y, name="m").fit  # no-op, just check
    a = Attributor()
    fit = a.fit("m", LINEAR, X=X, y=y)
    assert abs(fit.theta[0] - 2.5) < 1e-5


def test_invalid_data_shapes():
    a = Attributor()
    with pytest.raises(InvalidData):
        a.fit("bad", LINEAR, X=[[1, 1], [1, 2]], y=[1.0])  # n mismatch
    with pytest.raises(InvalidData):
        a.fit("bad", LINEAR, X=[[1, 1], [1, 2, 3]], y=[1.0, 2.0])
    with pytest.raises(InsufficientData):
        a.fit("bad", LINEAR, X=[], y=[])


def test_press_residuals_match_exact_loo():
    # PRESS_i = ε̂_i / (1 − h_ii)  is the *exact* LOO residual for OLS.
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 4.0], [1.0, 7.0], [1.0, 9.0]]
    y = [1.2, 1.9, 3.7, 6.6, 8.7]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    press = a.press_residuals("m")
    # Refit without each point and recompute residual at that x_i.
    for i in range(len(X)):
        kept_X = [X[k] for k in range(len(X)) if k != i]
        kept_y = [y[k] for k in range(len(y)) if k != i]
        a2 = Attributor()
        a2.fit("loo", LINEAR, X=kept_X, y=kept_y)
        # The LOO prediction at the removed x_i:
        beta_loo = a2._state("loo").theta
        pred_loo = sum(X[i][j] * beta_loo[j] for j in range(len(X[i])))
        loo_residual = y[i] - pred_loo
        assert abs(press[i] - loo_residual) < 1e-8, (
            f"PRESS mismatch at {i}: {press[i]} vs LOO {loo_residual}"
        )


def test_leverage_diagonal_sum_equals_p():
    # Standard identity: Σ_i h_ii = p for OLS (trace of the projection).
    X = [[1.0, x] for x in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    y = [-3.0, -1.0, 1.0, 3.0, 5.0]  # y = 1 + 2x
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    h = a.leverage("m")
    assert abs(sum(h) - len(X[0])) < 1e-5


def test_cooks_distance_flags_outliers():
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 10.0]]
    y = [1.5, 2.0, 2.5, 3.0, 1000.0]  # last is severe outlier + high leverage
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    cd = a.cooks_distance("m")
    assert all(v >= 0.0 for v in cd)
    # Outlier index dominates: well above the conventional 4/n threshold.
    assert cd[-1] > 4.0 / len(X)
    assert cd[-1] > max(cd[:-1])


def test_dfbetas_signed_change_matches_loo_coefficient():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    df = a.dfbetas("m")
    # Per-row vector dimension matches p.
    for row in df:
        assert len(row) == 2
    # Sign matches the direction of the LOO coefficient change.
    full_beta = list(a._state("m").theta)
    for i in range(len(X)):
        kept_X = [X[k] for k in range(len(X)) if k != i]
        kept_y = [y[k] for k in range(len(y)) if k != i]
        a2 = Attributor()
        a2.fit("loo", LINEAR, X=kept_X, y=kept_y)
        loo_beta = list(a2._state("loo").theta)
        for j in range(2):
            actual_sign = (full_beta[j] - loo_beta[j])
            df_sign = df[i][j]
            if abs(actual_sign) > 1e-9 and abs(df_sign) > 1e-9:
                assert (actual_sign > 0) == (df_sign > 0), (
                    f"DFBETAS sign mismatch at i={i} j={j}: "
                    f"actual {actual_sign} vs df {df_sign}"
                )


def test_studentized_residuals_t_distributed_under_null():
    # On clean linear data, |t_i| should be small.
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 6)]
    y = [1.0 + 0.5 * x[1] for x in X]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    stud = a.studentized_residuals("m")
    for v in stud:
        # On a perfect fit, studentized is 0 or NaN; we just check it isn't huge.
        if not math.isnan(v):
            assert abs(v) < 1.0


# -----------------------------------------------------------------------
# Influence-function approximation matches direction of exact LOO
# -----------------------------------------------------------------------


def test_if_and_loo_signs_match_on_well_conditioned_data():
    rnd = random.Random(0xCAFE)
    n, p = 30, 3
    X = []
    y = []
    true_beta = [0.5, -1.2, 0.8]
    for _ in range(n):
        x = [1.0] + [rnd.gauss(0, 1) for _ in range(p - 1)]
        eps = rnd.gauss(0, 0.1)
        y.append(sum(x[j] * true_beta[j] for j in range(p)) + eps)
        X.append(x)
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    test_x = [1.0, 0.3, -0.2]
    if_inf = a.influence("m", query=QUERY_PREDICTION, test_point=test_x)
    loo_inf = a.influence("m", query=QUERY_PREDICTION, test_point=test_x,
                          method=EXACT_LOO)
    assert len(if_inf.per_point) == n
    assert len(loo_inf.per_point) == n
    # Direction agreement on well-conditioned data: nearly all signs match.
    match = sum(
        1 for a_, b in zip(if_inf.per_point, loo_inf.per_point)
        if (a_ >= 0) == (b >= 0)
    )
    assert match >= 0.85 * n


def test_if_for_loss_query_matches_taylor_expansion():
    # On the *training* data, the loss-IF should approximately equal the
    # difference (Q(θ̂_{-i}) − Q(θ̂)) — exact would require 2nd-order.
    X = [[1.0, x] for x in (-2, -1, 0, 1, 2)]
    y = [-4.0, -2.0, 0.0, 2.0, 4.0]  # y = 2x
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    # Pick a separate test point.
    test = ([1.0, 0.5], 1.0)
    if_inf = a.influence("m", query=QUERY_LOSS, test_point=test)
    loo_inf = a.influence("m", query=QUERY_LOSS, test_point=test,
                          method=EXACT_LOO)
    # Signs of nonzero entries should mostly agree.
    for i, (a_, b) in enumerate(zip(if_inf.per_point, loo_inf.per_point)):
        if abs(a_) > 1e-7 and abs(b) > 1e-7:
            assert (a_ > 0) == (b > 0), f"sign mismatch at i={i}"


def test_influence_report_ranking_helpers():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    inf = a.influence("m", query=QUERY_PREDICTION, test_point=[1.0, 6.0])
    top = inf.most_influential(3)
    assert len(top) == 3
    assert all(isinstance(t, tuple) for t in top)
    abs_vals = [abs(v) for _, v in top]
    assert abs_vals == sorted(abs_vals, reverse=True)
    assert inf.top_indices(2) == [t[0] for t in top[:2]]
    assert inf.most_influential(0) == []


def test_parameter_query():
    X = [[1.0, x] for x in (-1, 0, 1, 2)]
    y = [0.0, 1.0, 2.0, 3.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    inf = a.influence("m", query=QUERY_PARAMETER, param_index=1)
    assert len(inf.per_point) == 4
    # Custom should error without param_index.
    with pytest.raises(InvalidQuery):
        a.influence("m", query=QUERY_PARAMETER, param_index=42)


# -----------------------------------------------------------------------
# Group LOO and Sherman-Morrison-Woodbury consistency
# -----------------------------------------------------------------------


def test_group_loo_singleton_matches_loo():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    loo = a.influence("m", query=QUERY_PREDICTION, test_point=[1.0, 6.0],
                      method=EXACT_LOO)
    for i in range(len(X)):
        g = a.group_loo("m", indices=[i],
                        query=QUERY_PREDICTION, test_point=[1.0, 6.0])
        assert abs(g - loo.per_point[i]) < 1e-7, (
            f"group_loo singleton mismatch at i={i}: {g} vs {loo.per_point[i]}"
        )


def test_group_loo_equals_refit():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10, 12)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0, 150.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    grp = a.group_loo("m", indices=[5, 6],
                      query=QUERY_PREDICTION, test_point=[1.0, 6.0])
    cf = a.counterfactual_refit("m", remove=[5, 6],
                                query=QUERY_PREDICTION, test_point=[1.0, 6.0])
    assert abs(grp - cf.delta_q) < 1e-7


def test_group_loo_rejects_bad_indices():
    a = quick_linear_attribution([[1, 1], [1, 2]], [1.0, 2.0])
    with pytest.raises(AttributorError):
        a.group_loo("model", indices=[5],
                    query=QUERY_PREDICTION, test_point=[1.0, 1.0])
    with pytest.raises(AttributorError):
        a.group_loo("model", indices=[0, 0],
                    query=QUERY_PREDICTION, test_point=[1.0, 1.0])


def test_group_loo_empty_set_is_zero():
    a = quick_linear_attribution([[1, 1], [1, 2]], [1.0, 2.0])
    g = a.group_loo("model", indices=[],
                    query=QUERY_PREDICTION, test_point=[1.0, 1.5])
    assert g == 0.0


# -----------------------------------------------------------------------
# Counterfactual refit
# -----------------------------------------------------------------------


def test_counterfactual_refit_flips_outlier_dominated_fit():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    full_beta = list(a._state("m").theta)
    cf = a.counterfactual_refit("m", remove=[5],
                                query=QUERY_PREDICTION, test_point=[1.0, 6.0])
    # The clean fit on the remaining points should give β ≈ (1, 0.5).
    assert abs(cf.theta_counterfactual[0] - 1.0) < 1e-5
    assert abs(cf.theta_counterfactual[1] - 0.5) < 1e-5
    # full_q − counterfactual_q should be big (≥ 30 in magnitude).
    assert abs(cf.delta_q) > 30.0
    assert cf.removed == [5]
    assert isinstance(cf, CounterfactualReport)


def test_counterfactual_refit_cannot_remove_all_points():
    a = quick_linear_attribution([[1, 1], [1, 2]], [1.0, 2.0])
    with pytest.raises(InsufficientData):
        a.counterfactual_refit("model", remove=[0, 1],
                               query=QUERY_PREDICTION, test_point=[1.0, 1.0])


# -----------------------------------------------------------------------
# Decision flip certificate
# -----------------------------------------------------------------------


def test_decision_flip_certificate_finds_flipping_set():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    # Decision: "is slope > 5?"  Yes on the outlier-dominated fit.
    flip = a.decision_flip(
        "m",
        decision_fn=lambda theta: "yes" if theta[1] > 5 else "no",
        budget_k=3,
        query=QUERY_PREDICTION, test_point=[1.0, 6.0],
    )
    assert isinstance(flip, DecisionFlipReport)
    assert flip.flipped
    assert flip.decision_full == "yes"
    assert flip.decision_after == "no"
    assert 5 in flip.minimal_set
    # E-value is a Bayes factor between counterfactual and full fits.
    assert flip.e_value > 0.0
    assert math.isfinite(flip.log10_bayes_factor)


def test_decision_flip_respects_ranking():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    # If we force a ranking that doesn't contain the outlier, the flip
    # may not happen within budget.
    flip = a.decision_flip(
        "m",
        decision_fn=lambda theta: "yes" if theta[1] > 5 else "no",
        budget_k=2,
        ranking=[0, 1],   # don't try index 5
        query=QUERY_PREDICTION, test_point=[1.0, 6.0],
    )
    assert not flip.flipped


def test_decision_flip_rejects_bad_budget():
    a = quick_linear_attribution([[1, 1], [1, 2]], [1.0, 2.0])
    with pytest.raises(AttributorError):
        a.decision_flip("model",
                        decision_fn=lambda t: "x",
                        budget_k=0,
                        query=QUERY_PREDICTION, test_point=[1.0, 1.0])
    with pytest.raises(AttributorError):
        a.decision_flip("model",
                        decision_fn=lambda t: "x",
                        budget_k=100,
                        query=QUERY_PREDICTION, test_point=[1.0, 1.0])


# -----------------------------------------------------------------------
# Logistic regression
# -----------------------------------------------------------------------


def test_logistic_fit_converges_on_well_separated_data():
    # Mostly separable but with one label-noise example to avoid the
    # MLE diverging to infinity.
    X = [[1.0, -2.0], [1.0, -1.0], [1.0, -0.2],
         [1.0, 0.2], [1.0, 1.0], [1.0, 2.0]]
    y = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    a = Attributor()
    fit = a.fit("clf", LOGISTIC, X=X, y=y, ridge=1e-3)
    assert fit.converged
    # β_1 should be positive (class-separation in the same direction).
    assert fit.theta[1] > 0.5


def test_logistic_leverage_in_unit_interval():
    rnd = random.Random(0xBEEF)
    X = [[1.0, rnd.gauss(0, 1)] for _ in range(20)]
    y = [1.0 if x[1] > 0 else 0.0 for x in X]
    a = Attributor()
    a.fit("clf", LOGISTIC, X=X, y=y, ridge=1e-3)
    lev = a.leverage("clf")
    for h in lev:
        assert 0.0 <= h <= 1.0 + 1e-6


def test_logistic_targets_must_be_binary():
    a = Attributor()
    with pytest.raises(InvalidData):
        a.fit("bad", LOGISTIC, X=[[1.0, 1.0]], y=[0.5])


def test_logistic_if_loo_sign_match():
    rnd = random.Random(0xFEED)
    X = [[1.0, rnd.gauss(0, 1), rnd.gauss(0, 1)] for _ in range(30)]
    # logistic ground-truth.
    true_beta = [0.0, 1.5, -1.0]
    y = []
    for x in X:
        z = sum(x[j] * true_beta[j] for j in range(3))
        p = 1.0 / (1.0 + math.exp(-z))
        y.append(1.0 if rnd.random() < p else 0.0)
    a = Attributor()
    a.fit("clf", LOGISTIC, X=X, y=y, ridge=1e-3)
    test = ([1.0, 0.5, -0.5], 1.0)
    if_inf = a.influence("clf", query=QUERY_LOSS, test_point=test)
    loo_inf = a.influence("clf", query=QUERY_LOSS, test_point=test,
                          method=EXACT_LOO)
    # Sign agreement on well-conditioned points.
    sig = 0
    matches = 0
    for a_, b in zip(if_inf.per_point, loo_inf.per_point):
        if abs(a_) > 1e-5 and abs(b) > 1e-5:
            sig += 1
            if (a_ >= 0) == (b >= 0):
                matches += 1
    assert sig > 0
    assert matches >= int(0.7 * sig)


# -----------------------------------------------------------------------
# Ridge regression
# -----------------------------------------------------------------------


def test_ridge_shrinks_coefficients():
    X = [[1.0, x] for x in (-2, -1, 0, 1, 2)]
    y = [-4.0, -2.0, 0.0, 2.0, 4.0]  # y = 2x
    a = Attributor()
    fit_ols = a.fit("ols", LINEAR, X=X, y=y)
    fit_ridge = a.fit("rid", RIDGE, X=X, y=y, ridge=10.0)
    assert abs(fit_ridge.theta[1]) < abs(fit_ols.theta[1])


def test_ridge_negative_lambda_rejected():
    a = Attributor()
    with pytest.raises(AttributorError):
        a.fit("m", RIDGE, X=[[1.0]], y=[1.0], ridge=-0.1)


# -----------------------------------------------------------------------
# Custom-kind interface (caller-supplied gradient + HVP)
# -----------------------------------------------------------------------


def test_custom_kind_influence_via_cg():
    # Synthetic quadratic loss: L(θ) = ½ Σ ((θ - z_i)^⊤ A (θ - z_i))
    # with the same A for all points → H = n·A, ∇L_i = A(θ - z_i).
    A = [[2.0, 0.5], [0.5, 1.0]]
    Z = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]]
    theta0 = [0.0, 0.0]
    n = len(Z)

    def grad_fn(i, theta):
        d = [theta[k] - Z[i][k] for k in range(2)]
        return [A[0][0] * d[0] + A[0][1] * d[1],
                A[1][0] * d[0] + A[1][1] * d[1]]

    def hvp_fn(theta, v):
        # H = n · A
        return [n * (A[0][0] * v[0] + A[0][1] * v[1]),
                n * (A[1][0] * v[0] + A[1][1] * v[1])]

    a = Attributor()
    a.fit("q", CUSTOM, X=[[0.0, 0.0]] * n, y=[0.0] * n,
          grad_fn=grad_fn, hvp_fn=hvp_fn, theta_init=theta0)
    # Custom query: Q(θ) = θ_0 + θ_1.  ∇Q = (1, 1).
    inf = a.influence("q", query=QUERY_CUSTOM,
                      custom_query=lambda t: t[0] + t[1],
                      custom_query_grad=lambda t: [1.0, 1.0],
                      method=INFLUENCE_FUNCTION)
    assert len(inf.per_point) == n
    # All values finite.
    assert all(math.isfinite(v) for v in inf.per_point)


def test_custom_kind_requires_init_and_grad():
    a = Attributor()
    with pytest.raises(AttributorError):
        a.fit("q", CUSTOM, X=[[0.0]], y=[0.0])  # missing theta_init
    with pytest.raises(AttributorError):
        a.fit("q", CUSTOM, X=[[0.0]], y=[0.0], theta_init=[0.0])  # missing grad


def test_custom_kind_exact_loo_unsupported():
    a = Attributor()
    a.fit("q", CUSTOM, X=[[0.0]], y=[0.0],
          grad_fn=lambda i, t: [0.0],
          hvp_fn=lambda t, v: [1.0 * v[0]],
          theta_init=[0.0])
    with pytest.raises(AttributorError):
        a.influence("q", query=QUERY_CUSTOM,
                    custom_query=lambda t: t[0],
                    custom_query_grad=lambda t: [1.0],
                    method=EXACT_LOO)


# -----------------------------------------------------------------------
# TracIn ideal
# -----------------------------------------------------------------------


def test_tracin_ideal_runs_and_has_correct_length():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    # Trajectory: 3 checkpoints — just θ̂ each time for simplicity.
    theta = list(a._state("m").theta)
    traj = [(theta, 0.1), (theta, 0.1), (theta, 0.1)]
    rep = a.tracin_ideal(
        "m",
        trajectory=traj,
        grad_query=lambda theta: [1.0, 0.5],
    )
    assert isinstance(rep, TracInReport)
    assert len(rep.per_point) == 5
    assert rep.n_checkpoints == 3
    top = rep.most_influential(2)
    assert len(top) == 2


def test_tracin_rejects_empty_trajectory():
    a = quick_linear_attribution([[1, 1]], [1.0])
    with pytest.raises(InvalidData):
        a.tracin_ideal("model",
                       trajectory=[],
                       grad_query=lambda t: [0.0] * len(t))


def test_tracin_custom_uses_registered_grad_fn():
    # A CUSTOM-kind model fit with grad_fn supplied uses that grad as
    # the built-in per-point gradient for TracIn.
    a = Attributor()
    a.fit("q", CUSTOM, X=[[0.0]] * 3, y=[0.0] * 3,
          grad_fn=lambda i, t: [float(i), float(i) * 0.5],
          hvp_fn=lambda t, v: [v[0] * 2.0, v[1] * 2.0],
          theta_init=[0.0, 0.0])
    traj = [([0.1, 0.2], 0.05), ([0.05, 0.1], 0.05)]
    rep = a.tracin_ideal(
        "q", trajectory=traj,
        grad_query=lambda theta: [1.0, 1.0],
    )
    # i = 0 contributes 0 to all steps (grad is 0); i > 0 contributes positively.
    assert rep.per_point[0] == 0.0
    assert rep.per_point[1] > 0.0
    assert rep.per_point[2] > rep.per_point[1]


# -----------------------------------------------------------------------
# TRAK random projection
# -----------------------------------------------------------------------


def test_trak_returns_per_point_scores():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    scores = a.trak("m", proj_dim=32, seed=1, query_x=[1.0, 4.0])
    assert len(scores) == 5
    for s in scores:
        assert math.isfinite(s)


def test_trak_dim_validation():
    a = quick_linear_attribution([[1, 1]], [1.0])
    with pytest.raises(AttributorError):
        a.trak("model", proj_dim=0, query_x=[1.0, 1.0])


# -----------------------------------------------------------------------
# Bootstrap confidence band
# -----------------------------------------------------------------------


def test_bootstrap_band_returns_proper_envelope():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 6, 7, 8)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    band = a.bootstrap_influence_band(
        "m", query=QUERY_PREDICTION, test_point=[1.0, 4.0],
        delta=0.10, n_resamples=80, seed=42,
    )
    assert isinstance(band, BootstrapBand)
    assert len(band.per_point_lower) == 8
    assert len(band.per_point_upper) == 8
    for lo, hi in zip(band.per_point_lower, band.per_point_upper):
        assert lo <= hi + 1e-9


def test_bootstrap_band_rejects_bad_delta():
    a = quick_linear_attribution([[1, 1]], [1.0])
    with pytest.raises(AttributorError):
        a.bootstrap_influence_band("model", query=QUERY_PREDICTION,
                                   test_point=[1, 1], delta=1.5)
    with pytest.raises(AttributorError):
        a.bootstrap_influence_band("model", query=QUERY_PREDICTION,
                                   test_point=[1, 1], n_resamples=1)


# -----------------------------------------------------------------------
# Attestation chain
# -----------------------------------------------------------------------


def test_chain_remains_valid_through_typical_workflow():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    fp0 = a.fingerprint
    a.fit("m", LINEAR, X=X, y=y)
    fp1 = a.fingerprint
    assert fp1 != fp0
    a.cooks_distance("m")
    a.influence("m", query=QUERY_PREDICTION, test_point=[1.0, 6.0])
    a.counterfactual_refit("m", remove=[5],
                           query=QUERY_PREDICTION, test_point=[1.0, 6.0])
    a.linear_diagnostics("m")
    a.report("m")
    assert a.verify_chain()
    assert a.fingerprint != fp1


def test_chain_breaks_on_tamper():
    X = [[1.0, x] for x in (1, 2, 3, 4)]
    y = [1.0, 2.0, 3.0, 4.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    assert a.verify_chain()
    # Tamper with an internal event's payload — chain must reject.
    a._events[1].payload["tampered"] = True
    assert not a.verify_chain()


def test_clear_resets_state_but_keeps_chain_valid():
    a = quick_linear_attribution([[1, 1], [1, 2]], [1.0, 2.0])
    a.clear()
    assert a.names() == []
    assert a.verify_chain()


# -----------------------------------------------------------------------
# Report (aggregate)
# -----------------------------------------------------------------------


def test_attributor_report_carries_diagnostics_for_linear():
    X = [[1.0, x] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    a.fit("m", LINEAR, X=X, y=y)
    rep = a.report("m")
    assert isinstance(rep, AttributorReport)
    assert rep.diagnostics is not None
    assert len(rep.diagnostics.cooks_distance) == 6


def test_attributor_report_for_logistic_has_no_linear_diag():
    X = [[1.0, x] for x in (-1.0, 0.0, 1.0, 2.0)]
    y = [0.0, 0.0, 1.0, 1.0]
    a = Attributor()
    a.fit("clf", LOGISTIC, X=X, y=y, ridge=1e-3)
    rep = a.report("clf")
    assert rep.diagnostics is None


# -----------------------------------------------------------------------
# Spec-based construction
# -----------------------------------------------------------------------


def test_attributor_from_spec_builds_multiple_hypotheses():
    spec = {
        "models": [
            {"name": "price", "kind": "linear",
             "X": [[1, 1], [1, 2], [1, 3]],
             "y": [1.0, 2.0, 3.0]},
            {"name": "ridge", "kind": "ridge",
             "X": [[1, 1], [1, 2], [1, 3]],
             "y": [1.0, 2.0, 3.0],
             "ridge": 0.1},
        ]
    }
    a = attributor_from_spec(spec)
    assert sorted(a.names()) == ["price", "ridge"]


def test_attributor_from_spec_rejects_malformed():
    with pytest.raises(AttributorError):
        attributor_from_spec("not a mapping")
    with pytest.raises(AttributorError):
        attributor_from_spec({"models": "not a list"})
    with pytest.raises(AttributorError):
        attributor_from_spec({"models": [{"kind": "linear"}]})


# -----------------------------------------------------------------------
# Quick constructors
# -----------------------------------------------------------------------


def test_quick_linear_constructor():
    a = quick_linear_attribution([[1, 1], [1, 2]], [1.0, 2.0])
    assert "model" in a.names()


def test_quick_logistic_constructor():
    a = quick_logistic_attribution([[1, -1], [1, 1]], [0, 1])
    assert "model" in a.names()
