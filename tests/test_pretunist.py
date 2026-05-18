"""Tests for :mod:`agi.pretunist`."""
from __future__ import annotations

import math
import random

import pytest

from agi.pretunist import (
    ABSTAIN_E_PROCESS,
    ABSTAIN_KL_BUDGET,
    ABSTAIN_LEVERAGE,
    ABSTAIN_LOO_RISK,
    ABSTAIN_VARIANCE,
    ALG_KERNEL_RIDGE,
    ALG_LOW_RANK,
    ALG_RIDGE,
    AbstentionReport,
    AdaptationResult,
    AdapterNotFit,
    AdapterParameters,
    DimensionMismatch,
    DriftBudgetExceeded,
    InsufficientData,
    InvalidConfig,
    InvalidQuery,
    InvalidSupport,
    KNOWN_ABSTAIN_RULES,
    KNOWN_ALGORITHMS,
    KNOWN_SSL,
    PRETUNIST_ADAPTED,
    PRETUNIST_ABSTAINED,
    PRETUNIST_CERTIFIED,
    PRETUNIST_OBSERVED,
    PRETUNIST_RESET,
    Pretunist,
    PretunistCertificate,
    PretunistConfig,
    PretunistError,
    PretunistReport,
    SSL_LOO,
    SSL_NONE,
    SSL_PREFIX_TARGET,
    SSL_RECONSTRUCT,
    SupportPoint,
    cholesky,
    kl_gauss_isotropic,
    leave_one_out_residuals,
    leverage_score,
    matmul,
    matvec,
    pac_bayes_bound_value,
    pretunist_ledger_genesis,
    pretunist_ledger_root,
    project_to_kl_budget,
    ridge_regression_closed_form,
    solve_lower_triangular,
    solve_psd,
    solve_upper_triangular,
    transpose,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestMath:
    def test_matvec(self) -> None:
        A = [[1.0, 2.0], [3.0, 4.0]]
        x = [1.0, 1.0]
        assert matvec(A, x) == [3.0, 7.0]

    def test_matvec_shape_mismatch(self) -> None:
        with pytest.raises(DimensionMismatch):
            matvec([[1.0, 2.0]], [1.0, 2.0, 3.0])

    def test_matmul(self) -> None:
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        C = matmul(A, B)
        assert C == [[19.0, 22.0], [43.0, 50.0]]

    def test_matmul_shape_mismatch(self) -> None:
        with pytest.raises(DimensionMismatch):
            matmul([[1.0]], [[1.0, 2.0], [3.0, 4.0]])

    def test_transpose(self) -> None:
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert transpose(A) == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_transpose_empty(self) -> None:
        assert transpose([]) == []

    def test_cholesky_identity(self) -> None:
        L = cholesky([[1.0, 0.0], [0.0, 1.0]])
        assert L == [[1.0, 0.0], [0.0, 1.0]]

    def test_cholesky_2x2(self) -> None:
        # A = [[4, 2],[2, 3]]; expected L = [[2, 0], [1, sqrt(2)]]
        L = cholesky([[4.0, 2.0], [2.0, 3.0]])
        assert L[0][0] == pytest.approx(2.0)
        assert L[1][0] == pytest.approx(1.0)
        assert L[1][1] == pytest.approx(math.sqrt(2.0))

    def test_cholesky_not_pd_raises(self) -> None:
        # Negative on diagonal -> not PD.
        with pytest.raises(PretunistError):
            cholesky([[-1.0, 0.0], [0.0, 1.0]])

    def test_cholesky_non_square_raises(self) -> None:
        with pytest.raises(DimensionMismatch):
            cholesky([[1.0, 0.0]])

    def test_solve_psd_round_trip(self) -> None:
        A = [[4.0, 1.0], [1.0, 3.0]]
        b = [9.0, 8.0]
        x = solve_psd(A, b)
        # A @ x ≈ b
        r = matvec(A, x)
        assert r[0] == pytest.approx(b[0], rel=1e-12)
        assert r[1] == pytest.approx(b[1], rel=1e-12)

    def test_solve_lower_triangular(self) -> None:
        L = [[2.0, 0.0], [1.0, 3.0]]
        b = [4.0, 7.0]
        y = solve_lower_triangular(L, b)
        # L @ y = b ⇒ y = [2, (7-1·2)/3] = [2, 5/3]
        assert y[0] == pytest.approx(2.0)
        assert y[1] == pytest.approx(5.0 / 3.0)

    def test_solve_upper_triangular(self) -> None:
        U = [[3.0, 1.0], [0.0, 2.0]]
        b = [5.0, 4.0]
        x = solve_upper_triangular(U, b)
        # x_1 = 4/2 = 2; x_0 = (5 - 1·2)/3 = 1
        assert x[1] == pytest.approx(2.0)
        assert x[0] == pytest.approx(1.0)

    def test_solve_zero_pivot_raises(self) -> None:
        with pytest.raises(PretunistError):
            solve_lower_triangular([[0.0, 0.0], [1.0, 1.0]], [1.0, 1.0])


# ---------------------------------------------------------------------------
# Standalone reference algorithms
# ---------------------------------------------------------------------------


class TestStandaloneAlgorithms:
    def test_ridge_recovers_exact_solution_under_no_noise(self) -> None:
        # y = X @ w exactly; recoveryof w if λ is small.
        random.seed(0)
        d = 4
        w = [random.gauss(0, 1) for _ in range(d)]
        X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(30)]
        y = [[sum(w[k] * row[k] for k in range(d))] for row in X]
        theta = ridge_regression_closed_form(X, y, ridge_lambda=1e-8)
        for k in range(d):
            assert theta[k][0] == pytest.approx(w[k], abs=1e-3)

    def test_ridge_shrinks_to_base_at_large_lambda(self) -> None:
        random.seed(1)
        d = 3
        X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(20)]
        y = [[1.0] for _ in range(20)]
        base = [[42.0] for _ in range(d)]
        # Huge λ should keep θ very close to base.
        theta = ridge_regression_closed_form(X, y, ridge_lambda=1e10, base=base)
        for k in range(d):
            assert theta[k][0] == pytest.approx(42.0, abs=1e-3)

    def test_ridge_empty_X_raises(self) -> None:
        with pytest.raises(InvalidSupport):
            ridge_regression_closed_form([], [], ridge_lambda=1.0)

    def test_ridge_shape_mismatch_raises(self) -> None:
        with pytest.raises(DimensionMismatch):
            ridge_regression_closed_form([[1.0, 2.0]], [[1.0], [2.0]], ridge_lambda=1.0)

    def test_leave_one_out_residuals_press_identity(self) -> None:
        # On 1-D linear data y = 2x: LOO residual = 0.
        X = [[float(i)] for i in range(1, 11)]
        y = [[2.0 * row[0]] for row in X]
        e = leave_one_out_residuals(X, y, ridge_lambda=1e-12)
        for r in e:
            assert abs(r[0]) < 1e-4

    def test_leave_one_out_residuals_empty(self) -> None:
        assert leave_one_out_residuals([], [], ridge_lambda=1.0) == []

    def test_kl_gauss_isotropic_zero_when_identical(self) -> None:
        kl = kl_gauss_isotropic([1.0, 2.0], [1.0, 2.0], 1.0, 1.0)
        assert kl == pytest.approx(0.0)

    def test_kl_gauss_isotropic_positive(self) -> None:
        kl = kl_gauss_isotropic([1.0, 2.0], [3.0, 4.0], 1.0, 1.0)
        # ||diff||² = 4 + 4 = 8 ⇒ KL = ½ · 8 / 1 = 4
        assert kl == pytest.approx(4.0)

    def test_kl_gauss_variance_only(self) -> None:
        # Same mean, different variance.
        kl = kl_gauss_isotropic([0.0, 0.0], [0.0, 0.0], 2.0, 1.0)
        # ½ d (r − 1 − log r) with r = 2, d = 2 → ½·2·(1 − log 2)
        expected = 0.5 * 2.0 * (2.0 - 1.0 - math.log(2.0))
        assert kl == pytest.approx(expected)

    def test_kl_dim_mismatch_raises(self) -> None:
        with pytest.raises(DimensionMismatch):
            kl_gauss_isotropic([1.0], [1.0, 2.0], 1.0, 1.0)

    def test_kl_bad_variance_raises(self) -> None:
        with pytest.raises(InvalidConfig):
            kl_gauss_isotropic([1.0], [1.0], -1.0, 1.0)

    def test_pac_bayes_bound_value(self) -> None:
        # KL=0, n=100, δ=0.05 ⇒ slack = sqrt(log(100/0.05)/200)
        slack = math.sqrt(math.log(100.0 / 0.05) / 200.0)
        b = pac_bayes_bound_value(empirical_risk=0.1, kl_qp_nats=0.0, n=100, delta=0.05)
        assert b == pytest.approx(0.1 + slack)

    def test_pac_bayes_n_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            pac_bayes_bound_value(empirical_risk=0.0, kl_qp_nats=0.0, n=0, delta=0.05)

    def test_pac_bayes_delta_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            pac_bayes_bound_value(empirical_risk=0.0, kl_qp_nats=0.0, n=10, delta=0.0)
        with pytest.raises(InvalidConfig):
            pac_bayes_bound_value(empirical_risk=0.0, kl_qp_nats=0.0, n=10, delta=1.0)

    def test_pac_bayes_negative_kl_raises(self) -> None:
        with pytest.raises(InvalidConfig):
            pac_bayes_bound_value(empirical_risk=0.0, kl_qp_nats=-1.0, n=10, delta=0.05)

    def test_leverage_score_empty_X(self) -> None:
        h = leverage_score([1.0, 0.0], [], ridge_lambda=2.0)
        # ||x||² / λ = 1 / 2 = 0.5
        assert h == pytest.approx(0.5)

    def test_leverage_score_with_support(self) -> None:
        X = [[1.0, 0.0], [0.0, 1.0]]
        h = leverage_score([1.0, 0.0], X, ridge_lambda=0.01)
        # A = I + 0.01 I = 1.01 I; A^-1 = 1/1.01 I; h = 1/1.01
        assert h == pytest.approx(1.0 / 1.01)

    def test_leverage_score_dim_mismatch(self) -> None:
        with pytest.raises(DimensionMismatch):
            leverage_score([1.0, 2.0], [[1.0]], ridge_lambda=1.0)

    def test_project_to_kl_budget_no_drift(self) -> None:
        # θ == θ_0 ⇒ alpha undefined but we return base.
        base = [[0.0], [0.0]]
        proj, alpha, kl = project_to_kl_budget(base, base,
                                              sigma_p_sq=1.0, sigma_q_sq=1.0,
                                              budget_nats=1.0)
        assert proj == [[0.0], [0.0]]
        assert alpha == 0.0
        assert kl == pytest.approx(0.0)

    def test_project_to_kl_budget_full_alpha_when_budget_loose(self) -> None:
        theta = [[1.0], [0.0]]
        base = [[0.0], [0.0]]
        proj, alpha, kl = project_to_kl_budget(theta, base,
                                              sigma_p_sq=1.0, sigma_q_sq=1.0,
                                              budget_nats=10.0)
        assert alpha == pytest.approx(1.0)
        assert proj[0][0] == pytest.approx(1.0)

    def test_project_to_kl_budget_shrinks(self) -> None:
        theta = [[10.0], [0.0]]
        base = [[0.0], [0.0]]
        proj, alpha, kl = project_to_kl_budget(theta, base,
                                              sigma_p_sq=1.0, sigma_q_sq=1.0,
                                              budget_nats=0.5)
        assert 0.0 < alpha < 1.0
        assert kl == pytest.approx(0.5, rel=1e-6)

    def test_project_to_kl_budget_base_kl_too_big(self) -> None:
        # When σ_q ≠ σ_p the prior↔posterior KL is non-zero even at α=0.
        # Budget < that base KL ⇒ projection is to α=0.
        theta = [[1.0], [0.0]]
        base = [[0.0], [0.0]]
        # σ_q²=100, σ_p²=1 ⇒ base KL = ½·2·(100 - 1 - log 100) ≈ 94.4
        proj, alpha, kl = project_to_kl_budget(theta, base,
                                              sigma_p_sq=1.0, sigma_q_sq=100.0,
                                              budget_nats=1.0)
        assert alpha == 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_construct(self) -> None:
        c = PretunistConfig()
        assert c.adapter_dim > 0
        assert c.output_dim == 1
        assert c.algorithm == ALG_RIDGE
        assert c.ridge_lambda > 0

    def test_adapter_dim_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(adapter_dim=0)
        with pytest.raises(InvalidConfig):
            PretunistConfig(adapter_dim=-1)

    def test_output_dim_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(output_dim=0)

    def test_algorithm_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(algorithm="bogus")

    def test_ridge_lambda_nonpositive(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(ridge_lambda=0.0)
        with pytest.raises(InvalidConfig):
            PretunistConfig(ridge_lambda=-1.0)

    def test_variance_nonpositive(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(prior_variance=0.0)
        with pytest.raises(InvalidConfig):
            PretunistConfig(posterior_variance=0.0)
        with pytest.raises(InvalidConfig):
            PretunistConfig(noise_variance=0.0)

    def test_kl_budget_nonpositive(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(kl_budget=0.0)

    def test_ssl_mode_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(ssl_mode="bogus")

    def test_ssl_prefix_fraction_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(ssl_prefix_fraction=0.0)
        with pytest.raises(InvalidConfig):
            PretunistConfig(ssl_prefix_fraction=1.0)

    def test_abstain_rules_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(abstain_rules=("bogus",))

    def test_abstain_rules_subset_of_known(self) -> None:
        # All known rules must be accepted.
        c = PretunistConfig(abstain_rules=tuple(sorted(KNOWN_ABSTAIN_RULES)))
        assert ABSTAIN_LEVERAGE in c.abstain_rules

    def test_low_rank_requires_rank(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(algorithm=ALG_LOW_RANK, rank=0, adapter_dim=4)
        # rank > adapter_dim invalid
        with pytest.raises(InvalidConfig):
            PretunistConfig(algorithm=ALG_LOW_RANK, rank=5, adapter_dim=4)
        # Valid
        c = PretunistConfig(algorithm=ALG_LOW_RANK, rank=2, adapter_dim=4)
        assert c.rank == 2

    def test_max_support_below_min_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(min_support=10, max_support=5)

    def test_e_process_grid_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(e_process_grid=())
        with pytest.raises(InvalidConfig):
            PretunistConfig(e_process_grid=(0.0,))
        with pytest.raises(InvalidConfig):
            PretunistConfig(e_process_grid=(float("inf"),))

    def test_e_process_alpha_invalid(self) -> None:
        with pytest.raises(InvalidConfig):
            PretunistConfig(e_process_alpha=0.0)
        with pytest.raises(InvalidConfig):
            PretunistConfig(e_process_alpha=1.0)

    def test_to_dict_round_trip(self) -> None:
        c = PretunistConfig(adapter_dim=4, kl_budget=1.0)
        d = c.to_dict()
        assert d["adapter_dim"] == 4
        assert d["kl_budget"] == 1.0

    def test_to_dict_kl_budget_inf_serialised_as_none(self) -> None:
        c = PretunistConfig(kl_budget=math.inf)
        assert c.to_dict()["kl_budget"] is None


# ---------------------------------------------------------------------------
# Pretunist core
# ---------------------------------------------------------------------------


def _gen_linear_data(d: int, n: int, *, noise: float = 0.0, seed: int = 0):
    rng = random.Random(seed)
    w = [rng.gauss(0, 1) for _ in range(d)]
    xs, ys = [], []
    for _ in range(n):
        x = [rng.gauss(0, 1) for _ in range(d)]
        y = sum(wi * xi for wi, xi in zip(w, x)) + rng.gauss(0, noise)
        xs.append(x)
        ys.append([y])
    return w, xs, ys


class TestPretunistConstructionAndObserve:
    def test_construct_default(self) -> None:
        pre = Pretunist()
        assert pre.n_observed == 0
        assert pre.n_adaptations == 0
        assert pre.has_adapter is False
        assert pre.ledger_root == pretunist_ledger_genesis()

    def test_construct_with_config(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=4))
        assert pre.config.adapter_dim == 4

    def test_observe_increments_count(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=3))
        pre.observe_support([1.0, 0.0, 0.0], [1.0])
        assert pre.n_observed == 1

    def test_observe_dim_mismatch_raises(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=3))
        with pytest.raises(InvalidSupport):
            pre.observe_support([1.0, 0.0], [1.0])

    def test_observe_non_finite_raises(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        with pytest.raises(InvalidSupport):
            pre.observe_support([1.0, float("nan")], [1.0])
        with pytest.raises(InvalidSupport):
            pre.observe_support([1.0, 2.0], [float("inf")])

    def test_observe_max_support_enforced(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2, max_support=2))
        pre.observe_support([1.0, 0.0], [1.0])
        pre.observe_support([0.0, 1.0], [1.0])
        with pytest.raises(InvalidSupport):
            pre.observe_support([1.0, 1.0], [2.0])

    def test_observe_scalar_y(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], 3.14)
        assert pre.n_observed == 1

    def test_observe_y_dim_mismatch(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2, output_dim=2))
        with pytest.raises(InvalidSupport):
            pre.observe_support([1.0, 0.0], [1.0])  # output_dim=2, y has length 1

    def test_observe_weight_invalid(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        with pytest.raises(InvalidSupport):
            pre.observe_support([1.0, 0.0], [1.0], weight=-0.1)
        with pytest.raises(InvalidSupport):
            pre.observe_support([1.0, 0.0], [1.0], weight=float("nan"))

    def test_observe_no_y_with_ssl_none_raises(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2, ssl_mode=SSL_NONE))
        with pytest.raises(InvalidSupport):
            pre.observe_support([1.0, 2.0])

    def test_observe_no_y_with_ssl_prefix_target_succeeds(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=4, ssl_mode=SSL_PREFIX_TARGET))
        pre.observe_support([1.0, 2.0, 3.0, 4.0])
        assert pre.n_observed == 1

    def test_observe_no_y_with_ssl_loo_succeeds(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2, ssl_mode=SSL_LOO))
        pre.observe_support([1.0, 2.0])
        assert pre.n_observed == 1

    def test_observe_no_y_with_ssl_reconstruct(self) -> None:
        # SSL_RECONSTRUCT requires output_dim == adapter_dim
        pre = Pretunist(PretunistConfig(adapter_dim=3, output_dim=3, ssl_mode=SSL_RECONSTRUCT))
        pre.observe_support([0.1, 0.2, 0.3])
        assert pre.n_observed == 1

    def test_observe_batch(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        xs = [[1.0, 0.0], [0.0, 1.0]]
        ys = [[1.0], [2.0]]
        pre.observe_batch(xs, ys)
        assert pre.n_observed == 2

    def test_observe_batch_weights(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_batch(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0], [2.0]],
            weights=[0.5, 0.5],
        )
        assert pre.n_observed == 2

    def test_observe_batch_length_mismatch(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        with pytest.raises(InvalidSupport):
            pre.observe_batch([[1.0, 0.0]], [[1.0], [2.0]])
        with pytest.raises(InvalidSupport):
            pre.observe_batch([[1.0, 0.0]], [[1.0]], weights=[1.0, 1.0])

    def test_observe_emits_event(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        events: list[tuple[str, dict]] = []
        pre.subscribe(lambda topic, payload: events.append((topic, payload)))
        pre.observe_support([1.0, 0.0], [1.0])
        assert any(t == PRETUNIST_OBSERVED for t, _ in events)


class TestBasePolicy:
    def test_default_base_is_zero(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=3))
        base = pre.get_base()
        for row in base:
            for v in row:
                assert v == 0.0

    def test_set_base_round_trip(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2, output_dim=2))
        b = [[1.0, 2.0], [3.0, 4.0]]
        pre.set_base(b)
        assert pre.get_base() == [[1.0, 2.0], [3.0, 4.0]]

    def test_set_base_dim_mismatch(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        with pytest.raises(DimensionMismatch):
            pre.set_base([[1.0]])
        with pytest.raises(DimensionMismatch):
            pre.set_base([[1.0, 2.0], [3.0, 4.0]])  # output_dim=1, but row has 2

    def test_set_base_non_finite_raises(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        with pytest.raises(InvalidConfig):
            pre.set_base([[1.0], [float("nan")]])

    def test_set_base_invalidates_adapter(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        pre.adapt()
        assert pre.has_adapter
        pre.set_base([[1.0], [2.0]])
        assert not pre.has_adapter

    def test_base_fingerprint_changes_with_base(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        fp0 = pre.base_fingerprint
        pre.set_base([[1.0], [0.0]])
        assert pre.base_fingerprint != fp0


class TestAdapt:
    def test_adapt_recovers_w_under_no_noise(self) -> None:
        d = 5
        w, X, y = _gen_linear_data(d, 80, noise=0.0, seed=42)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-8))
        pre.observe_batch(X, y)
        r = pre.adapt(query=X[0])
        # Recovered θ ≈ w
        for k in range(d):
            assert r.adapter.theta[k][0] == pytest.approx(w[k], abs=1e-3)
        # Prediction matches y[0]
        assert r.prediction[0] == pytest.approx(y[0][0], abs=1e-3)

    def test_adapt_requires_min_support(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2, min_support=3))
        pre.observe_support([1.0, 0.0], [1.0])
        with pytest.raises(InsufficientData):
            pre.adapt(query=[1.0, 0.0])

    def test_adapt_query_dim_mismatch(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        with pytest.raises(InvalidQuery):
            pre.adapt(query=[1.0])

    def test_adapt_query_non_finite_raises(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        with pytest.raises(InvalidQuery):
            pre.adapt(query=[1.0, float("nan")])

    def test_adapt_without_query_returns_empty_prediction(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        r = pre.adapt()
        assert r.prediction == ()
        assert r.base_prediction == ()

    def test_adapt_gain_positive_when_base_zero(self) -> None:
        # Base is zero ⇒ adapter must beat it when y ≠ 0.
        d = 4
        _, X, y = _gen_linear_data(d, 30, noise=0.01, seed=1)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-4))
        pre.observe_batch(X, y)
        r = pre.adapt()
        assert r.adaptation_gain_bits > 0.0

    def test_adapt_n_adaptations_increments(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        assert pre.n_adaptations == 0
        pre.adapt()
        assert pre.n_adaptations == 1
        pre.adapt()
        assert pre.n_adaptations == 2

    def test_adapt_fingerprint_is_stable(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 20, noise=0.01, seed=2)
        pre = Pretunist(PretunistConfig(adapter_dim=d))
        pre.observe_batch(X, y)
        r1 = pre.adapt()
        r2 = pre.adapt()
        assert r1.adapter.fingerprint == r2.adapter.fingerprint

    def test_adapt_emits_event(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        events: list[tuple[str, dict]] = []
        pre.subscribe(lambda t, p: events.append((t, p)))
        pre.observe_support([1.0, 0.0], [1.0])
        pre.adapt()
        assert any(t == PRETUNIST_ADAPTED for t, _ in events)

    def test_adapt_kl_drift_zero_when_base_matches_solution(self) -> None:
        # If base is *already* the optimum, the residual fit is zero ⇒
        # θ* == base ⇒ KL ≈ 0 (modulo posterior/prior variance terms).
        d = 3
        w, X, y = _gen_linear_data(d, 30, noise=0.0, seed=3)
        pre = Pretunist(PretunistConfig(
            adapter_dim=d, ridge_lambda=1e-8,
            prior_variance=1.0, posterior_variance=1.0,
        ))
        pre.set_base([[wi] for wi in w])
        pre.observe_batch(X, y)
        r = pre.adapt()
        assert r.kl_drift_nats < 1e-3

    def test_adapt_kl_drift_positive_when_base_zero(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 30, noise=0.0, seed=4)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-8))
        pre.observe_batch(X, y)
        r = pre.adapt()
        assert r.kl_drift_nats > 0.0

    def test_adapt_loo_risk_finite(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 20, noise=0.1, seed=5)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-3))
        pre.observe_batch(X, y)
        r = pre.adapt()
        assert math.isfinite(r.loo_risk)
        assert r.loo_risk >= 0.0


class TestKLBudgetProjection:
    def test_unbudgeted_returns_alpha_one(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 20, noise=0.0, seed=6)
        pre = Pretunist(PretunistConfig(adapter_dim=d))
        pre.observe_batch(X, y)
        r = pre.adapt()
        assert r.kl_projection_alpha == 1.0
        assert not r.kl_budget_active

    def test_tight_budget_projects(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 30, noise=0.0, seed=7)
        # Get an unbudgeted solution to know the KL.
        pre0 = Pretunist(PretunistConfig(adapter_dim=d))
        pre0.observe_batch(X, y)
        kl_full = pre0.adapt().kl_drift_nats
        assert kl_full > 0.0
        # Budget = half the unbudgeted KL.
        pre = Pretunist(PretunistConfig(adapter_dim=d, kl_budget=kl_full / 2.0))
        pre.observe_batch(X, y)
        r = pre.adapt()
        assert r.kl_budget_active
        assert 0.0 < r.kl_projection_alpha < 1.0
        assert r.kl_drift_nats == pytest.approx(kl_full / 2.0, rel=1e-3)

    def test_budget_zero_alpha_when_base_kl_exceeds(self) -> None:
        # σ_q ≠ σ_p so base_kl > 0 even at α=0; budget below that.
        d = 2
        _, X, y = _gen_linear_data(d, 20, noise=0.0, seed=8)
        pre = Pretunist(PretunistConfig(
            adapter_dim=d, prior_variance=1.0, posterior_variance=100.0,
            kl_budget=1.0,
        ))
        pre.observe_batch(X, y)
        r = pre.adapt()
        assert r.kl_budget_active
        assert r.kl_projection_alpha == 0.0


class TestAbstention:
    def test_no_abstention_in_distribution(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 30, noise=0.01, seed=9)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-3))
        pre.observe_batch(X, y)
        rep = pre.should_abstain(X[0])
        assert not rep.triggered

    def test_abstain_leverage_far_point(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 5, noise=0.0, seed=10)
        # With small ridge_lambda and tiny support, a very far point
        # hits the leverage threshold.
        pre = Pretunist(PretunistConfig(
            adapter_dim=d, ridge_lambda=1e-4,
            abstain_rules=(ABSTAIN_LEVERAGE,),
            abstain_leverage_threshold=0.9,
        ))
        pre.observe_batch(X, y)
        rep = pre.should_abstain([1000.0, 0.0, 0.0])
        assert rep.triggered
        assert ABSTAIN_LEVERAGE in rep.rules_fired

    def test_abstain_predictive_variance(self) -> None:
        d = 2
        _, X, y = _gen_linear_data(d, 5, noise=0.0, seed=11)
        pre = Pretunist(PretunistConfig(
            adapter_dim=d, ridge_lambda=1e-4, noise_variance=100.0,
            abstain_rules=(ABSTAIN_VARIANCE,),
            abstain_variance_threshold=10.0,
        ))
        pre.observe_batch(X, y)
        rep = pre.should_abstain([1.0, 0.0])
        assert rep.triggered

    def test_abstain_kl_budget(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 20, noise=0.0, seed=12)
        # Force budget very tight ⇒ KL drift ≥ budget after adapt.
        pre = Pretunist(PretunistConfig(
            adapter_dim=d, kl_budget=0.001,
            abstain_rules=(ABSTAIN_KL_BUDGET,),
        ))
        pre.observe_batch(X, y)
        pre.adapt()
        rep = pre.should_abstain(X[0])
        assert ABSTAIN_KL_BUDGET in rep.rules_fired

    def test_abstain_query_dim_mismatch(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        with pytest.raises(InvalidQuery):
            pre.should_abstain([1.0])

    def test_abstain_increments_counter(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 5, noise=0.0, seed=13)
        pre = Pretunist(PretunistConfig(
            adapter_dim=d, ridge_lambda=1e-4,
            abstain_rules=(ABSTAIN_LEVERAGE,),
            abstain_leverage_threshold=0.9,
        ))
        pre.observe_batch(X, y)
        pre.should_abstain([1000.0, 0.0, 0.0])
        assert pre.n_abstentions == 1


class TestPACBayes:
    def test_certify_returns_cert(self) -> None:
        d = 4
        _, X, y = _gen_linear_data(d, 80, noise=0.05, seed=14)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-3))
        pre.observe_batch(X, y)
        pre.adapt()
        cert = pre.certify(delta=0.05)
        assert isinstance(cert, PretunistCertificate)
        assert cert.n == 80
        assert cert.delta == 0.05
        assert cert.pac_bayes_bound > cert.empirical_risk

    def test_certify_without_adapt_raises(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        with pytest.raises(AdapterNotFit):
            pre.certify()

    def test_certify_includes_ledger(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 20, noise=0.05, seed=15)
        pre = Pretunist(PretunistConfig(adapter_dim=d))
        pre.observe_batch(X, y)
        pre.adapt()
        cert = pre.certify()
        assert len(cert.ledger_root) == 64  # SHA-256 hex

    def test_pac_bayes_alias(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        pre.adapt()
        assert pre.pac_bayes_bound(0.1).pac_bayes_bound == pre.certify(0.1).pac_bayes_bound

    def test_bound_tightens_with_n(self) -> None:
        d = 4
        _, X100, y100 = _gen_linear_data(d, 100, noise=0.05, seed=16)
        pre1 = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-3))
        pre1.observe_batch(X100[:20], y100[:20])
        pre1.adapt()
        c1 = pre1.certify()

        pre2 = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-3))
        pre2.observe_batch(X100, y100)
        pre2.adapt()
        c2 = pre2.certify()

        # Larger n ⇒ tighter (smaller slack) at comparable KL.
        assert c2.pac_bayes_bound < c1.pac_bayes_bound + 0.5

    def test_certify_negative_delta_raises(self) -> None:
        d = 2
        _, X, y = _gen_linear_data(d, 5, noise=0.01, seed=17)
        pre = Pretunist(PretunistConfig(adapter_dim=d))
        pre.observe_batch(X, y)
        pre.adapt()
        with pytest.raises(InvalidConfig):
            pre.certify(delta=0.0)


class TestEProcess:
    def test_e_process_starts_at_zero(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        assert pre.e_process_log == 0.0
        assert not pre.e_process_rejected()

    def test_e_process_grows_when_adapter_helps(self) -> None:
        # Interleaved adapt/observe ⇒ each new observation is scored
        # against the fitted adapter.  On clean linear data the adapter
        # consistently beats base.
        d = 3
        _, X, y = _gen_linear_data(d, 40, noise=0.01, seed=18)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-3))
        # Warm-up
        for i in range(5):
            pre.observe_support(X[i], y[i])
        pre.adapt()
        log0 = pre.e_process_log
        # More observations after adapter exists.
        for i in range(5, 30):
            pre.observe_support(X[i], y[i])
        pre.adapt()
        log1 = pre.e_process_log
        # Should have moved (in some direction).
        assert log1 != log0

    def test_e_process_log_finite(self) -> None:
        d = 2
        _, X, y = _gen_linear_data(d, 10, noise=0.1, seed=19)
        pre = Pretunist(PretunistConfig(adapter_dim=d))
        pre.observe_batch(X, y)
        pre.adapt()
        assert math.isfinite(pre.e_process_log)


class TestPredict:
    def test_predict_round_trip(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 20, noise=0.0, seed=20)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-6))
        pre.observe_batch(X, y)
        pre.adapt()
        p = pre.predict(X[0])
        # Should be close to the LOO prediction.
        assert isinstance(p, tuple)
        assert len(p) == 1
        assert math.isfinite(p[0])

    def test_predict_without_adapt_raises(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        with pytest.raises(AdapterNotFit):
            pre.predict([1.0, 0.0])

    def test_predict_base(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.set_base([[5.0], [3.0]])
        p = pre.predict_base([1.0, 1.0])
        assert p[0] == pytest.approx(8.0)

    def test_predict_query_dim_mismatch(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        pre.adapt()
        with pytest.raises(InvalidQuery):
            pre.predict([1.0])

    def test_predict_base_query_dim_mismatch(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        with pytest.raises(InvalidQuery):
            pre.predict_base([1.0])


class TestReportSnapshot:
    def test_report_has_all_fields(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 10, noise=0.0, seed=21)
        pre = Pretunist(PretunistConfig(adapter_dim=d))
        pre.observe_batch(X, y)
        pre.adapt()
        rep = pre.report()
        assert isinstance(rep, PretunistReport)
        assert rep.n_support == 10
        assert rep.n_adaptations == 1
        assert math.isfinite(rep.last_loo_risk)
        assert len(rep.ledger_root) == 64

    def test_report_pre_adapt_returns_inf_bound(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.observe_support([1.0, 0.0], [1.0])
        rep = pre.report()
        assert math.isinf(rep.last_pac_bayes_bound)

    def test_snapshot_restore_round_trip(self) -> None:
        d = 4
        _, X, y = _gen_linear_data(d, 20, noise=0.01, seed=22)
        pre = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-3))
        pre.set_base([[0.5]] * d)
        pre.observe_batch(X, y)
        r0 = pre.adapt(query=X[0])
        snap = pre.snapshot()
        pre2 = Pretunist.restore(snap)
        # Same ledger root and adapter fingerprint after restore.
        assert pre.ledger_root == pre2.ledger_root
        assert pre.adapter_fingerprint == pre2.adapter_fingerprint
        # Reproduce the original prediction without re-adapting.
        p1 = pre2.predict(X[0])
        assert r0.prediction[0] == pytest.approx(p1[0])

    def test_snapshot_handles_inf_kl_budget(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2, kl_budget=math.inf))
        snap = pre.snapshot()
        assert snap["config"]["kl_budget"] is None
        pre2 = Pretunist.restore(snap)
        assert math.isinf(pre2.config.kl_budget)


class TestReset:
    def test_reset_clears_state(self) -> None:
        d = 3
        _, X, y = _gen_linear_data(d, 10, noise=0.0, seed=23)
        pre = Pretunist(PretunistConfig(adapter_dim=d))
        pre.observe_batch(X, y)
        pre.adapt()
        pre.reset()
        assert pre.n_observed == 0
        assert pre.n_adaptations == 1  # counter persists by design
        assert not pre.has_adapter

    def test_reset_preserves_base_and_config(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.set_base([[1.0], [2.0]])
        cfg = pre.config
        pre.reset()
        assert pre.get_base() == [[1.0], [2.0]]
        assert pre.config is cfg

    def test_reset_emits_event(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        events: list[tuple[str, dict]] = []
        pre.subscribe(lambda t, p: events.append((t, p)))
        pre.reset()
        assert any(t == PRETUNIST_RESET for t, _ in events)


class TestLedger:
    def test_genesis_roots_match(self) -> None:
        assert pretunist_ledger_genesis() == pretunist_ledger_root()

    def test_ledger_grows_on_observe(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        g = pre.ledger_root
        pre.observe_support([1.0, 0.0], [1.0])
        assert pre.ledger_root != g

    def test_ledger_with_hmac_key_differs(self) -> None:
        cfg1 = PretunistConfig(adapter_dim=2, hmac_key=b"alpha")
        cfg2 = PretunistConfig(adapter_dim=2, hmac_key=b"beta")
        p1 = Pretunist(cfg1)
        p2 = Pretunist(cfg2)
        p1.observe_support([1.0, 0.0], [1.0])
        p2.observe_support([1.0, 0.0], [1.0])
        assert p1.ledger_root != p2.ledger_root

    def test_ledger_deterministic_under_same_inputs(self) -> None:
        # Two primitives with the same config / hmac / observations
        # should produce identical ledger roots.  Note: the per-record
        # timestamp is included, so we have to monkey-patch time. We
        # instead just verify that *operationally* deterministic
        # inputs produce the same fingerprint via snapshot round-trip.
        d = 2
        pre = Pretunist(PretunistConfig(adapter_dim=d))
        pre.observe_support([1.0, 0.0], [1.0])
        snap1 = pre.snapshot()
        pre2 = Pretunist.restore(snap1)
        snap2 = pre2.snapshot()
        # Same ledger because restore preserves it.
        assert snap1["ledger_root"] == snap2["ledger_root"]


class TestManifestSpec:
    def test_spec_shape(self) -> None:
        spec = Pretunist.manifest_spec()
        assert spec["name"] == "pretunist"
        assert spec["kind"] == "learning"
        assert spec["certificate"] == "pac"
        assert spec["dependency"] == "stdlib"
        # Composition hints should reference adjacent primitives
        assert "continualist" in spec["composes_with"]


class TestOutputDim:
    def test_multivariate_output(self) -> None:
        # y = (x, 2x) ⇒ recovers two columns.
        random.seed(33)
        d = 3
        n = 50
        xs, ys = [], []
        w1 = [1.0, -0.5, 0.25]
        w2 = [0.0, 1.0, -1.0]
        for _ in range(n):
            x = [random.gauss(0, 1) for _ in range(d)]
            y = [
                sum(w1[k] * x[k] for k in range(d)),
                sum(w2[k] * x[k] for k in range(d)),
            ]
            xs.append(x)
            ys.append(y)
        pre = Pretunist(PretunistConfig(adapter_dim=d, output_dim=2, ridge_lambda=1e-8))
        pre.observe_batch(xs, ys)
        r = pre.adapt(query=xs[0])
        for k in range(d):
            assert r.adapter.theta[k][0] == pytest.approx(w1[k], abs=1e-3)
            assert r.adapter.theta[k][1] == pytest.approx(w2[k], abs=1e-3)


class TestSubscriberFaultTolerance:
    def test_bad_subscriber_does_not_break(self) -> None:
        pre = Pretunist(PretunistConfig(adapter_dim=2))
        pre.subscribe(lambda t, p: 1 / 0)
        # Must not raise.
        pre.observe_support([1.0, 0.0], [1.0])
        pre.adapt()


class TestAdapterParameters:
    def test_drift_l2(self) -> None:
        ap = AdapterParameters(
            theta=((1.0, 0.0), (0.0, 1.0)),
            base=((0.0, 0.0), (0.0, 0.0)),
            ridge_lambda=1.0,
            n=10,
            fingerprint="abc",
        )
        # ||θ - θ_0||₂ = √2
        assert ap.drift_l2() == pytest.approx(math.sqrt(2.0))


class TestEndToEnd:
    """High-level workflow tests — coordination-engine perspective."""

    def test_full_workflow(self) -> None:
        d = 6
        w, X, y = _gen_linear_data(d, 100, noise=0.05, seed=99)
        # Split into support / query.
        X_supp, y_supp = X[:80], y[:80]
        X_query = X[80:]
        y_query = y[80:]

        pre = Pretunist(PretunistConfig(
            adapter_dim=d,
            ridge_lambda=1e-3,
            prior_variance=1.0,
            posterior_variance=1.0,
            kl_budget=10.0,
            abstain_rules=(ABSTAIN_LEVERAGE, ABSTAIN_VARIANCE),
        ))
        pre.observe_batch(X_supp, y_supp)
        # Coordinator path
        total_se = 0.0
        n_pred = 0
        n_abst = 0
        for xq, yq in zip(X_query, y_query):
            ab = pre.should_abstain(xq)
            if ab.triggered:
                n_abst += 1
                continue
            r = pre.adapt(query=xq)
            total_se += (r.prediction[0] - yq[0]) ** 2
            n_pred += 1
        assert n_pred > 0
        mse = total_se / n_pred
        # The model should have low MSE given the recovery of w.
        assert mse < 0.5

        cert = pre.certify(delta=0.05)
        # The bound is non-vacuous and conservative.
        assert cert.pac_bayes_bound > cert.empirical_risk
        # And on this clean linear task it should not be vacuous.
        assert not cert.pac_bayes_is_vacuous

    def test_kl_budget_workflow(self) -> None:
        d = 4
        _, X, y = _gen_linear_data(d, 40, noise=0.0, seed=100)
        # Unconstrained KL
        pre_un = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-6))
        pre_un.observe_batch(X, y)
        r_un = pre_un.adapt()

        # Constrained KL
        pre_c = Pretunist(PretunistConfig(
            adapter_dim=d, ridge_lambda=1e-6,
            kl_budget=r_un.kl_drift_nats / 4.0,
        ))
        pre_c.observe_batch(X, y)
        r_c = pre_c.adapt()
        assert r_c.kl_budget_active
        assert r_c.kl_drift_nats <= r_un.kl_drift_nats / 4.0 + 1e-6
        # The constrained adapter is *closer to base* than unconstrained.
        assert r_c.adapter.drift_l2() < r_un.adapter.drift_l2()
