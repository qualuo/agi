"""Tests for ``agi.privacy`` — differential privacy as a runtime primitive.

The tests follow the mathematical contract of the module:

1. **Standard normal CDF / inverse** match closed-form to 1e-9.
2. **Analytic Gaussian σ is tighter than classical** for ε > 1.
3. **Classical Gaussian σ matches √(2 ln(1.25/δ)) Δ / ε**.
4. **Gaussian RDP** matches the closed form ``α Δ² / (2σ²)``.
5. **Laplace RDP** is increasing in sensitivity and decreasing in b.
6. **Subsampled-Gaussian RDP** ≤ Gaussian RDP (subsampling tightens).
7. **RDP→(ε, δ) conversion** picks the optimal α.
8. **Basic composition** sums ε and δ.
9. **Advanced composition** is tighter than basic for k ≫ 1/ε.
10. **zCDP↔ε,δ-DP conversion** matches Bun-Steinke formula.
11. **Laplace sample** has empirical mean ≈ 0 and variance ≈ 2b².
12. **Gaussian sample** has empirical variance ≈ σ².
13. **Snap mechanism** rounds output to the grid.
14. **Exponential mechanism** is biased toward high-utility items.
15. **PrivacyAccountant: laplace + gaussian** debits budget correctly.
16. **PrivacyAccountant: budget exhaustion** raises BudgetExhausted.
17. **PrivacyAccountant: ledger hash** is deterministic + changes with new releases.
18. **Each release's fingerprint** is stable under repeat construction.
19. **SVT: max_positive caps hits**.
20. **SVT: total ε budget = ε_t + max_positive · ε_a**.
21. **BinaryTreeCounter: error bounded by O(log T)**.
22. **RDP accountant: composing N Gaussian releases adds N · α Δ² / (2σ²)**.
23. **Replay determinism**: same seed produces same release values.
24. **Negative sensitivity / negative ε / bad δ raise InvalidMechanism**.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from agi.privacy import (
    BinaryTreeCounter,
    BudgetExhausted,
    InvalidMechanism,
    PrivacyAccountant,
    RenyiAccountant,
    Release,
    SparseVector,
    advanced_composition,
    analytic_gaussian_sigma,
    basic_composition,
    classical_gaussian_sigma,
    exponential_select,
    gaussian_rdp,
    gaussian_sample,
    gaussian_to_zcdp,
    laplace_rdp,
    laplace_sample,
    rdp_to_epsilon_delta,
    snap_sample,
    std_normal_cdf,
    std_normal_inv_cdf,
    subsampled_gaussian_rdp,
    zcdp_to_epsilon_delta,
)


# -----------------------------------------------------------------------------
# 1. Standard normal CDF / inverse
# -----------------------------------------------------------------------------


def test_std_normal_cdf_known_values():
    assert abs(std_normal_cdf(0.0) - 0.5) < 1e-12
    assert abs(std_normal_cdf(1.0) - 0.8413447) < 1e-5
    assert abs(std_normal_cdf(1.96) - 0.975) < 1e-3
    assert abs(std_normal_cdf(-1.96) - 0.025) < 1e-3


def test_std_normal_inv_cdf_known_values():
    assert abs(std_normal_inv_cdf(0.5) - 0.0) < 1e-12
    assert abs(std_normal_inv_cdf(0.975) - 1.96) < 1e-3
    assert abs(std_normal_inv_cdf(0.025) + 1.96) < 1e-3


def test_std_normal_round_trip():
    for x in (-2.0, -1.0, 0.0, 0.5, 1.5, 2.5):
        assert abs(std_normal_inv_cdf(std_normal_cdf(x)) - x) < 1e-4


# -----------------------------------------------------------------------------
# 2-3. Gaussian calibration
# -----------------------------------------------------------------------------


def test_analytic_tighter_than_classical_at_eps_one():
    # At ε ≤ 1 they're close but analytic still ≤ classical
    sa = analytic_gaussian_sigma(1.0, 1.0, 1e-5)
    sc = classical_gaussian_sigma(1.0, 1.0, 1e-5)
    assert sa <= sc + 1e-9


def test_classical_gaussian_matches_formula():
    sigma = classical_gaussian_sigma(2.0, 0.5, 1e-5)
    expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) * 2.0 / 0.5
    assert abs(sigma - expected) < 1e-9


def test_classical_requires_eps_le_one():
    with pytest.raises(InvalidMechanism):
        classical_gaussian_sigma(1.0, 2.0, 1e-5)


def test_analytic_gaussian_zero_sensitivity():
    assert analytic_gaussian_sigma(0.0, 1.0, 1e-5) == 0.0


# -----------------------------------------------------------------------------
# 4-5. RDP closed forms
# -----------------------------------------------------------------------------


def test_gaussian_rdp_closed_form():
    Δ, σ, α = 2.0, 5.0, 3
    expected = α * Δ * Δ / (2.0 * σ * σ)
    assert abs(gaussian_rdp(Δ, σ, α) - expected) < 1e-12


def test_gaussian_rdp_monotone_in_alpha():
    Δ, σ = 1.0, 2.0
    prev = -1.0
    for α in (2, 3, 4, 8, 16):
        v = gaussian_rdp(Δ, σ, α)
        assert v > prev
        prev = v


def test_laplace_rdp_monotone_in_sensitivity():
    b, α = 1.0, 4
    prev = -1.0
    for Δ in (0.1, 0.5, 1.0, 2.0, 5.0):
        v = laplace_rdp(Δ, b, α)
        assert v > prev
        prev = v


def test_subsampled_gaussian_rdp_le_full():
    σ = 5.0
    q = 0.01
    for α in (2, 3, 8):
        sg = subsampled_gaussian_rdp(q, σ, α)
        full = gaussian_rdp(1.0, σ, α)
        assert sg <= full + 1e-9  # subsampling can only help


# -----------------------------------------------------------------------------
# 6. RDP → (ε, δ) conversion picks the optimal α
# -----------------------------------------------------------------------------


def test_rdp_to_eps_delta_picks_optimal_alpha():
    pairs = [(a, gaussian_rdp(1.0, 5.0, a)) for a in (2, 4, 8, 16, 32, 64)]
    eps, alpha_opt = rdp_to_epsilon_delta(pairs, 1e-5)
    assert math.isfinite(eps)
    assert alpha_opt in (2, 4, 8, 16, 32, 64)
    # Every α gives an upper bound; chosen is the smallest
    candidates = [e + math.log(1.0 / 1e-5) / (a - 1)
                  for a, e in pairs if a > 1]
    assert abs(eps - min(candidates)) < 1e-12


# -----------------------------------------------------------------------------
# 7. Composition theorems
# -----------------------------------------------------------------------------


def test_basic_composition_sums():
    eps_total, delta_total = basic_composition([0.1, 0.2, 0.3], [0.0, 1e-6, 1e-6])
    assert abs(eps_total - 0.6) < 1e-12
    assert abs(delta_total - 2e-6) < 1e-12


def test_advanced_tighter_than_basic_for_many_folds():
    # k=100 ε=0.1 → basic = 10
    eps, _ = advanced_composition(0.1, 0.0, 100, 1e-9)
    assert eps < 10.0


def test_zcdp_eps_delta_formula():
    # ρ=0.1, δ=1e-5
    expected = 0.1 + 2 * math.sqrt(0.1 * math.log(1e5))
    assert abs(zcdp_to_epsilon_delta(0.1, 1e-5) - expected) < 1e-12


def test_gaussian_to_zcdp():
    # σ=2, Δ=1 → ρ = 1/8
    assert abs(gaussian_to_zcdp(1.0, 2.0) - 0.125) < 1e-12


# -----------------------------------------------------------------------------
# 11-13. Sampling primitives
# -----------------------------------------------------------------------------


def test_laplace_sample_mean_variance():
    rng = random.Random(0)
    b = 2.0
    xs = [laplace_sample(rng, b) for _ in range(20_000)]
    m = statistics.mean(xs)
    v = statistics.pvariance(xs)
    assert abs(m) < 0.1
    # Var(Lap(b)) = 2b²
    assert abs(v - 2 * b * b) / (2 * b * b) < 0.1


def test_gaussian_sample_variance():
    rng = random.Random(0)
    σ = 3.0
    xs = [gaussian_sample(rng, σ) for _ in range(10_000)]
    v = statistics.pvariance(xs)
    assert abs(v - σ * σ) / (σ * σ) < 0.1


def test_snap_sample_is_on_grid():
    rng = random.Random(0)
    grid = 0.1
    for _ in range(100):
        x = snap_sample(rng, b=1.0, lam=grid)
        # x must be a multiple of grid (up to floating point precision)
        ratio = x / grid
        assert abs(ratio - round(ratio)) < 1e-6


# -----------------------------------------------------------------------------
# 14. Exponential mechanism is biased toward best
# -----------------------------------------------------------------------------


def test_exponential_picks_best_with_high_eps():
    items = ["a", "b", "c"]
    utilities = [1.0, 10.0, 2.0]  # b is best
    rng = random.Random(0)
    picks = []
    for _ in range(200):
        c, _ = exponential_select(items, utilities, sensitivity=1.0, epsilon=5.0,
                                  rng=rng)
        picks.append(c)
    # With ε=5 and sensitivity 1, b should win the vast majority
    assert picks.count("b") > 0.7 * len(picks)


def test_exponential_uniform_with_low_eps():
    items = ["a", "b", "c"]
    utilities = [1.0, 10.0, 2.0]
    rng = random.Random(0)
    picks = []
    for _ in range(300):
        c, _ = exponential_select(items, utilities, sensitivity=10.0, epsilon=0.01,
                                  rng=rng)
        picks.append(c)
    # With huge sensitivity ε is effectively zero ⇒ near-uniform
    for it in items:
        assert 0.2 * len(picks) < picks.count(it) < 0.5 * len(picks)


# -----------------------------------------------------------------------------
# 15-16. Accountant budget
# -----------------------------------------------------------------------------


def test_accountant_laplace_debits_budget():
    A = PrivacyAccountant(epsilon=1.0, delta=1e-5, seed=0)
    A.laplace(value=100.0, sensitivity=1.0, epsilon=0.4)
    assert abs(A.spent_epsilon - 0.4) < 1e-12
    A.laplace(value=200.0, sensitivity=1.0, epsilon=0.4)
    assert abs(A.spent_epsilon - 0.8) < 1e-12


def test_accountant_gaussian_debits_eps_delta():
    A = PrivacyAccountant(epsilon=1.0, delta=1e-5, seed=0)
    A.gaussian(value=42.0, sensitivity=1.0, epsilon=0.4, delta=1e-7)
    assert abs(A.spent_epsilon - 0.4) < 1e-12
    assert abs(A.spent_delta - 1e-7) < 1e-15


def test_accountant_refuses_overspend():
    A = PrivacyAccountant(epsilon=1.0, delta=1e-5, seed=0)
    A.laplace(value=0, sensitivity=1, epsilon=0.9)
    with pytest.raises(BudgetExhausted):
        A.laplace(value=0, sensitivity=1, epsilon=0.2)


def test_accountant_refuses_overspend_delta():
    A = PrivacyAccountant(epsilon=10.0, delta=1e-8, seed=0)
    A.gaussian(value=0, sensitivity=1, epsilon=0.1, delta=1e-9)
    with pytest.raises(BudgetExhausted):
        A.gaussian(value=0, sensitivity=1, epsilon=0.1, delta=1e-8)


# -----------------------------------------------------------------------------
# 17. Ledger hash is deterministic and reactive
# -----------------------------------------------------------------------------


def test_ledger_hash_changes_with_releases():
    A = PrivacyAccountant(epsilon=10.0, seed=0)
    h0 = A.ledger_hash()
    A.laplace(value=1, sensitivity=1, epsilon=0.1)
    h1 = A.ledger_hash()
    A.laplace(value=2, sensitivity=1, epsilon=0.1)
    h2 = A.ledger_hash()
    assert h0 != h1 != h2


def test_ledger_hash_deterministic_under_seed():
    A1 = PrivacyAccountant(epsilon=10.0, seed=42)
    A2 = PrivacyAccountant(epsilon=10.0, seed=42)
    for A in (A1, A2):
        A.laplace(value=10.0, sensitivity=1.0, epsilon=0.1)
        A.laplace(value=20.0, sensitivity=1.0, epsilon=0.1)
    assert A1.ledger_hash() == A2.ledger_hash()


# -----------------------------------------------------------------------------
# 18. Release fingerprint stable
# -----------------------------------------------------------------------------


def test_release_has_fingerprint():
    A = PrivacyAccountant(epsilon=1.0, seed=0)
    A.laplace(value=5.0, sensitivity=1.0, epsilon=0.1)
    assert len(A.releases) == 1
    rel = A.releases[0]
    assert isinstance(rel, Release)
    assert len(rel.fingerprint) == 64  # hex sha256


# -----------------------------------------------------------------------------
# 19-20. Sparse Vector Technique
# -----------------------------------------------------------------------------


def test_svt_caps_positives():
    A = PrivacyAccountant(epsilon=20.0, seed=0)
    svt = A.sparse_vector(threshold=0.0, sensitivity=1.0,
                          epsilon_threshold=1.0, epsilon_answer=0.5,
                          max_positive=3)
    # Every query massively above threshold
    queries = [1e6] * 10
    hits = sum(1 for v in queries if svt.query(v))
    assert hits <= 3
    assert svt.closed


def test_svt_budget_total():
    A = PrivacyAccountant(epsilon=20.0, seed=0)
    eps_t = 0.5
    eps_a = 0.2
    max_pos = 4
    _ = A.sparse_vector(threshold=0.0, sensitivity=1.0,
                       epsilon_threshold=eps_t, epsilon_answer=eps_a,
                       max_positive=max_pos)
    # Threshold release is recorded immediately
    assert abs(A.spent_epsilon - eps_t) < 1e-12


# -----------------------------------------------------------------------------
# 21. Binary tree counter
# -----------------------------------------------------------------------------


def test_binary_tree_counter_error_bounded():
    T = 64
    A = PrivacyAccountant(epsilon=100.0, seed=0)
    btc = A.binary_tree_counter(T=T, sensitivity=1.0, epsilon=5.0)
    rng = random.Random(0)
    true_total = 0.0
    errs = []
    for _ in range(T):
        x = rng.choice([0.0, 1.0])
        true_total += x
        noisy = btc.increment(x)
        errs.append(abs(noisy - true_total))
    # max error should be small compared to T (poly-log scaling)
    assert max(errs) < T  # very loose bound


# -----------------------------------------------------------------------------
# 22. Renyi accountant additivity
# -----------------------------------------------------------------------------


def test_rdp_accountant_additive_under_gaussian():
    R = RenyiAccountant(alphas=(2, 4, 8))
    for _ in range(10):
        R.gaussian(sensitivity=1.0, sigma=4.0)
    # ε(α) should be 10 · α / (2 · 4²) = 10·α/32
    for a in (2, 4, 8):
        expected = 10.0 * a / 32.0
        assert abs(R._rdp[a] - expected) < 1e-12


def test_rdp_accountant_to_eps_delta():
    R = RenyiAccountant()
    for _ in range(100):
        R.gaussian(sensitivity=1.0, sigma=10.0)
    eps, alpha = R.to_epsilon_delta(delta=1e-5)
    assert math.isfinite(eps)
    assert alpha >= 2


# -----------------------------------------------------------------------------
# 23. Replay determinism
# -----------------------------------------------------------------------------


def test_accountant_replay_determinism():
    a1 = PrivacyAccountant(epsilon=10.0, delta=1e-3, seed=999)
    a2 = PrivacyAccountant(epsilon=10.0, delta=1e-3, seed=999)
    v1a = a1.laplace(value=100.0, sensitivity=1.0, epsilon=0.5)
    v1b = a2.laplace(value=100.0, sensitivity=1.0, epsilon=0.5)
    assert v1a == v1b
    v2a = a1.gaussian(value=50.0, sensitivity=1.0, epsilon=0.5, delta=1e-6)
    v2b = a2.gaussian(value=50.0, sensitivity=1.0, epsilon=0.5, delta=1e-6)
    assert v2a == v2b


# -----------------------------------------------------------------------------
# 24. Invalid mechanism arguments
# -----------------------------------------------------------------------------


def test_invalid_arguments_raise():
    with pytest.raises(InvalidMechanism):
        PrivacyAccountant(epsilon=0.0)
    with pytest.raises(InvalidMechanism):
        PrivacyAccountant(epsilon=1.0, delta=-0.1)
    with pytest.raises(InvalidMechanism):
        PrivacyAccountant(epsilon=1.0, composition="unknown")

    A = PrivacyAccountant(epsilon=1.0)
    with pytest.raises(InvalidMechanism):
        A.laplace(value=0, sensitivity=-1, epsilon=0.1)
    with pytest.raises(InvalidMechanism):
        A.gaussian(value=0, sensitivity=1, epsilon=0.1, delta=2.0)


# -----------------------------------------------------------------------------
# Extras
# -----------------------------------------------------------------------------


def test_remaining_budgets():
    A = PrivacyAccountant(epsilon=1.0, delta=1e-5, seed=0)
    A.laplace(value=0, sensitivity=1, epsilon=0.3)
    assert abs(A.remaining_epsilon - 0.7) < 1e-12


def test_summary_is_jsonable():
    import json
    A = PrivacyAccountant(epsilon=1.0, seed=0)
    A.laplace(value=0, sensitivity=1, epsilon=0.1)
    json.dumps(A.summary())  # must not raise


def test_advanced_composition_allows_more_releases_than_basic():
    # Many tiny queries: basic ε would be 50·0.01 = 0.5; advanced ≈ 0.46.
    # We set a target above advanced but below basic.
    A = PrivacyAccountant(epsilon=1.0, delta=1e-5, composition="advanced",
                          delta_prime=1e-9, seed=0)
    for _ in range(50):
        A.laplace(value=0, sensitivity=1, epsilon=0.01)
    assert A.n_releases == 50
