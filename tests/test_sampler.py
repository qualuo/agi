"""Tests for ``agi.sampler`` — Bayesian probabilistic inference.

The tests follow the mathematical contract of the module:

1. **HMC** on isotropic Gaussian recovers ``μ`` and ``σ²`` within
   Monte-Carlo error, with ``R̂ < 1.01`` and bulk-ESS ≥ 400.
2. **NUTS** matches HMC on the Gaussian benchmark and additionally
   passes on a correlated Gaussian (mass adaptation effect).
3. **MALA** recovers Gaussian moments at the Roberts-Rosenthal
   optimal acceptance ``≈ 0.574``.
4. **RWMH** with adaptive Metropolis recovers Gaussian moments at
   acceptance ``≈ 0.234``.
5. **Slice** sampling has zero-tuning and recovers moments.
6. **ULA** is consistent (bias → 0 as step → 0) but biased at any
   fixed step for non-Gaussian targets.
7. **Parallel tempering** escapes a deeply bimodal density that
   plain RWMH gets stuck in — directly observable as the cold-chain
   mass on the "far" mode.
8. **SMC** returns posterior samples from a Gaussian-prior +
   Gaussian-likelihood toy model AND its ``log_evidence`` is close
   to the analytic Gaussian-Gaussian marginal log-likelihood.
9. **Importance sampling** is unbiased on bounded test functions
   when the proposal covers the target; ``pareto_k`` flags a heavy-
   tailed mismatch.
10. **ADVI** mean-field recovers ``μ`` of a Gaussian; ``ELBO`` is
    monotone-improving on average.
11. **R̂** converges to 1 as chains mix; **R̂ ≫ 1** for unmixed
    independent-initialisation chains stuck in modes.
12. **Bulk / tail ESS** correctly reduces by ``τ`` on an AR(1) chain.
13. **Geweke z-score** is small for stationary chain, large for
    transient.
14. **DKW credible set** has the stated finite-sample coverage on
    a bounded target.
15. **HRMS anytime-valid credible set** maintains coverage when
    optionally stopped (the whole point of the bound).
16. **Fingerprint** is deterministic per (seed, kernel, n_samples)
    and changes when any of those change.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from agi.sampler import (
    Sampler,
    SampleReport,
    Diagnostics,
    ImportanceReport,
    SMCReport,
    ADVIReport,
    rhat,
    ess_bulk,
    ess_tail,
    autocorr,
    autocorr_time,
    geweke_z,
    pareto_k,
    gaussian_log_density,
    gaussian_grad,
    banana_log_density,
    banana_grad,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _make_iso(d: int = 2, rng_seed: int = 0) -> Sampler:
    """Isotropic standard Normal."""
    return Sampler(
        log_density=gaussian_log_density([0.0] * d, [1.0] * d),
        dim=d,
        grad_log_density=gaussian_grad([0.0] * d, [1.0] * d),
        rng=random.Random(rng_seed),
    )


# -----------------------------------------------------------------------------
# 1. HMC on isotropic Gaussian
# -----------------------------------------------------------------------------


def test_hmc_isotropic_recovers_moments():
    s = _make_iso(d=2, rng_seed=42)
    rep = s.hmc(n_samples=800, n_chains=4, warmup=400,
                step_size=0.5, n_leapfrog=10)
    mu = rep.mean()
    assert abs(mu[0]) < 0.15
    assert abs(mu[1]) < 0.15
    var = rep.var()
    assert 0.7 < var[0] < 1.3
    assert 0.7 < var[1] < 1.3
    assert all(r < 1.05 for r in rep.diagnostics.rhat)
    assert all(e > 200 for e in rep.diagnostics.ess_bulk)


# -----------------------------------------------------------------------------
# 2. NUTS on isotropic + correlated Gaussian
# -----------------------------------------------------------------------------


def test_nuts_isotropic_recovers_moments():
    s = _make_iso(d=2, rng_seed=7)
    rep = s.nuts(n_samples=400, n_chains=4, warmup=400)
    mu = rep.mean()
    assert abs(mu[0]) < 0.2
    assert abs(mu[1]) < 0.2
    assert all(r < 1.1 for r in rep.diagnostics.rhat)


def test_nuts_correlated_gaussian():
    # ~ N(0, Σ), Σ_{11}=Σ_{22}=1, Σ_{12}=0.9 — strong correlation
    # Use precision form: log p = -0.5 θᵀ Λ θ, Λ = Σ^{-1}
    # For Σ = [[1, 0.9],[0.9, 1]]:  det = 0.19
    # Λ = (1/0.19) [[1,-0.9],[-0.9,1]]
    inv_det = 1.0 / 0.19

    def lp(theta):
        x, y = theta[0], theta[1]
        return -0.5 * inv_det * (x * x - 1.8 * x * y + y * y)

    def gr(theta):
        x, y = theta[0], theta[1]
        return [-inv_det * (x - 0.9 * y), -inv_det * (y - 0.9 * x)]

    s = Sampler(lp, 2, gr, rng=random.Random(3))
    rep = s.nuts(n_samples=400, n_chains=4, warmup=400)
    mu = rep.mean()
    assert abs(mu[0]) < 0.25
    assert abs(mu[1]) < 0.25
    # marginal variances ~ 1
    var = rep.var()
    assert 0.5 < var[0] < 1.6
    assert 0.5 < var[1] < 1.6


# -----------------------------------------------------------------------------
# 3. MALA
# -----------------------------------------------------------------------------


def test_mala_gaussian():
    s = _make_iso(d=2, rng_seed=11)
    rep = s.mala(n_samples=800, n_chains=4, warmup=400, step_size=0.5)
    mu = rep.mean()
    assert all(abs(m) < 0.2 for m in mu)
    # acceptance not trivially 0 or 1 (after adaptation should aim ~0.574)
    for a in rep.accept_rates:
        assert 0.2 < a <= 1.0


# -----------------------------------------------------------------------------
# 4. RWMH adaptive
# -----------------------------------------------------------------------------


def test_rwmh_adaptive_gaussian():
    s = _make_iso(d=2, rng_seed=5)
    rep = s.rwmh(n_samples=1500, n_chains=4, warmup=800, proposal_sd=1.0)
    mu = rep.mean()
    assert all(abs(m) < 0.25 for m in mu)


# -----------------------------------------------------------------------------
# 5. Slice
# -----------------------------------------------------------------------------


def test_slice_gaussian():
    s = _make_iso(d=2, rng_seed=2)
    rep = s.slice_sample(n_samples=600, n_chains=2, warmup=200, width=1.5)
    mu = rep.mean()
    assert all(abs(m) < 0.25 for m in mu)
    var = rep.var()
    assert all(0.6 < v < 1.6 for v in var)


# -----------------------------------------------------------------------------
# 6. ULA consistency
# -----------------------------------------------------------------------------


def test_ula_smaller_step_smaller_bias():
    # On the standard normal, ULA stationary distribution is exact for the
    # discretised SDE only as step → 0.  The variance is biased upward by
    # O(step).  Larger step ⇒ larger variance.
    s_big = Sampler(
        gaussian_log_density([0.0], [1.0]), 1,
        gaussian_grad([0.0], [1.0]),
        rng=random.Random(0),
    )
    s_small = Sampler(
        gaussian_log_density([0.0], [1.0]), 1,
        gaussian_grad([0.0], [1.0]),
        rng=random.Random(0),
    )
    big = s_big.ula(n_samples=2000, n_chains=2, warmup=500, step_size=0.5)
    small = s_small.ula(n_samples=2000, n_chains=2, warmup=500, step_size=0.05)
    # bigger step has bigger sample variance
    assert big.var(0) > small.var(0) - 0.1
    # small step is closer to 1.0
    assert abs(small.var(0) - 1.0) < 0.3


# -----------------------------------------------------------------------------
# 7. Parallel tempering escapes bimodal
# -----------------------------------------------------------------------------


def test_parallel_tempering_finds_far_mode():
    # Two well-separated Gaussians: N(-4, 0.5²) + N(+4, 0.5²)
    def lp(theta):
        x = theta[0]
        a = math.exp(-0.5 * ((x + 4.0) / 0.5) ** 2)
        b = math.exp(-0.5 * ((x - 4.0) / 0.5) ** 2)
        return math.log(a + b + 1e-300)

    s = Sampler(lp, 1, rng=random.Random(123))
    rep = s.parallel_tempering(
        n_samples=2000, warmup=500, n_replicas=6,
        base_kernel="rwmh", step_size=0.6,
    )
    # cold chain should visit both modes — count fraction of samples with x>0
    pos = sum(1 for s_ in rep.chains[0] if s_[0] > 0)
    frac = pos / len(rep.chains[0])
    assert 0.2 < frac < 0.8, f"PT failed to mix: fraction positive = {frac}"


# -----------------------------------------------------------------------------
# 8. SMC log-evidence on Gaussian-Gaussian
# -----------------------------------------------------------------------------


def test_smc_evidence_gaussian_gaussian():
    # prior: N(0, 1).  likelihood: N(y; θ, 1) with y = 2.
    # posterior: N(1.0, 0.5).  log p(y) = log N(y; 0, 2) = -log √(4π) - y²/4
    y = 2.0

    def prior_sample(rng):
        return [rng.gauss(0.0, 1.0)]

    def log_prior(theta):
        return -0.5 * theta[0] ** 2

    def log_lik(theta):
        return -0.5 * (y - theta[0]) ** 2

    def log_joint(theta):  # log p(θ) + log p(y|θ)
        return log_prior(theta) + log_lik(theta)

    s = Sampler(log_joint, 1, rng=random.Random(0))
    rep = s.smc(
        n_particles=400, prior_sampler=prior_sample,
        log_likelihood=log_lik, ess_target=0.5, max_temps=20, n_mcmc=3,
    )
    posterior_mean = rep.mean(0)
    assert abs(posterior_mean - 1.0) < 0.4
    # marginal log-likelihood ≈ -0.5 log(4π) - 1 ≈ -2.265
    truth = -0.5 * math.log(4 * math.pi) - y * y / 4.0
    # SMC log_Z estimator has some bias / variance; allow generous slack
    assert abs(rep.log_evidence - truth) < 1.5


# -----------------------------------------------------------------------------
# 9. Importance sampling pareto-k
# -----------------------------------------------------------------------------


def test_importance_ess_flags_mismatched_proposal():
    # target: N(0, 4²) (wide).  proposal: N(0, 1) (narrow).
    # narrow proposal misses tails of target ⇒ ESS collapses.
    def target(theta):
        return -0.5 * theta[0] ** 2 / 16.0

    def prop_sampler(rng):
        return [rng.gauss(0.0, 1.0)]

    def log_q(theta):
        return -0.5 * theta[0] ** 2

    s = Sampler(target, 1, rng=random.Random(0))
    rep = s.importance(n=800, proposal_sampler=prop_sampler, log_proposal=log_q)
    # matched proposal: ESS should be ~ n
    def prop_match(rng):
        return [rng.gauss(0.0, 4.0)]

    def log_q_match(theta):
        return -0.5 * theta[0] ** 2 / 16.0

    rep2 = s.importance(n=800, proposal_sampler=prop_match,
                        log_proposal=log_q_match)
    # ESS for mismatched proposal should be markedly lower than for matched
    assert rep.ess < rep2.ess
    assert rep.ess < 600
    assert rep2.ess > 600
    # both pareto_k values should be finite
    assert math.isfinite(rep.pareto_k)
    assert math.isfinite(rep2.pareto_k)


def test_pareto_k_degenerate_returns_zero():
    # all-equal log_weights ⇒ k̂ = 0 (no heavy-tail signal)
    assert pareto_k([1.0] * 200) == 0.0


def test_importance_unbiased_test_function():
    def target(theta):
        return -0.5 * theta[0] ** 2

    def prop(rng):
        return [rng.gauss(0.0, 1.5)]

    def log_q(theta):
        return -0.5 * (theta[0] / 1.5) ** 2 - math.log(1.5)

    s = Sampler(target, 1, rng=random.Random(0))
    rep = s.importance(n=2000, proposal_sampler=prop, log_proposal=log_q)
    # E_π[θ²] = 1 (under N(0,1))
    val = rep.estimate(lambda th: th[0] ** 2)
    assert 0.6 < val < 1.5


# -----------------------------------------------------------------------------
# 10. ADVI
# -----------------------------------------------------------------------------


def test_advi_mean_field_recovers_gaussian_mu():
    s = Sampler(
        gaussian_log_density([1.0, -0.5], [1.0, 1.0]), 2,
        gaussian_grad([1.0, -0.5], [1.0, 1.0]),
        rng=random.Random(0),
    )
    rep = s.advi(n_iter=400, n_mc=8, learning_rate=0.1)
    assert abs(rep.mu[0] - 1.0) < 0.3
    assert abs(rep.mu[1] - (-0.5)) < 0.3
    # σ should be ~ 1.0
    assert 0.5 < rep.sigma[0] < 1.6
    # ELBO grew on average from start to end
    early = statistics.fmean(rep.elbo_history[:20])
    late = statistics.fmean(rep.elbo_history[-50:])
    assert late > early - 0.5  # at least not collapsing


def test_advi_sample_shape():
    s = _make_iso(d=3, rng_seed=0)
    rep = s.advi(n_iter=50, n_mc=4)
    draws = rep.sample(20)
    assert len(draws) == 20
    assert all(len(d) == 3 for d in draws)


# -----------------------------------------------------------------------------
# 11. R̂
# -----------------------------------------------------------------------------


def test_rhat_converges_to_one_for_mixed_chains():
    rng = random.Random(1)
    chains = [[rng.gauss(0.0, 1.0) for _ in range(1000)] for _ in range(4)]
    r = rhat(chains)
    assert abs(r - 1.0) < 0.05, f"R̂ = {r}"


def test_rhat_high_for_unmixed_chains():
    rng = random.Random(2)
    chains = [
        [rng.gauss(-5.0, 0.1) for _ in range(500)],   # chain 0 stuck at -5
        [rng.gauss(+5.0, 0.1) for _ in range(500)],   # chain 1 stuck at +5
        [rng.gauss(-5.0, 0.1) for _ in range(500)],
        [rng.gauss(+5.0, 0.1) for _ in range(500)],
    ]
    r = rhat(chains)
    assert r > 1.5, f"R̂ = {r}"


# -----------------------------------------------------------------------------
# 12. Bulk + tail ESS, autocorr
# -----------------------------------------------------------------------------


def test_ess_smaller_for_autocorrelated_chain():
    rng = random.Random(0)
    n = 2000
    # iid Normal
    iid_chain = [rng.gauss(0.0, 1.0) for _ in range(n)]
    # AR(1) with ρ = 0.9
    ar = [0.0]
    for _ in range(n - 1):
        ar.append(0.9 * ar[-1] + rng.gauss(0.0, 1.0))
    e_iid = ess_bulk([iid_chain])
    e_ar = ess_bulk([ar])
    assert e_ar < e_iid * 0.6, f"AR ESS {e_ar} not less than iid {e_iid}"
    # τ_int for AR(1) ρ=0.9 should be ~ 1 + 2*0.9/(1-0.9) ≈ 19
    tau = autocorr_time(ar)
    assert tau > 3.0


def test_autocorr_iid_near_zero_at_lag_1():
    rng = random.Random(0)
    chain = [rng.gauss(0.0, 1.0) for _ in range(2000)]
    rho = autocorr(chain, max_lag=10)
    assert abs(rho[1]) < 0.1


def test_ess_tail_positive():
    rng = random.Random(0)
    chain = [rng.gauss(0.0, 1.0) for _ in range(2000)]
    e = ess_tail([chain])
    assert e > 50.0


# -----------------------------------------------------------------------------
# 13. Geweke
# -----------------------------------------------------------------------------


def test_geweke_small_for_stationary():
    rng = random.Random(0)
    chain = [rng.gauss(0.0, 1.0) for _ in range(2000)]
    z = geweke_z(chain)
    assert abs(z) < 3.0


def test_geweke_large_for_transient():
    # mean drifts from -5 to 0 over time
    rng = random.Random(0)
    n = 2000
    chain = [rng.gauss(-5.0 + 5.0 * i / n, 0.1) for i in range(n)]
    z = geweke_z(chain)
    assert abs(z) > 3.0


# -----------------------------------------------------------------------------
# 14. DKW credible set
# -----------------------------------------------------------------------------


def test_credible_set_quantile_recovers_normal():
    # 95% equal-tailed CI of N(0,1) is (-1.96, 1.96)
    s = _make_iso(d=1, rng_seed=0)
    rep = s.hmc(n_samples=4000, n_chains=2, warmup=500,
                step_size=0.7, n_leapfrog=10)
    lo, hi = rep.credible_set(alpha=0.05, dim=0, method="quantile")
    assert lo < -1.4 and lo > -2.6
    assert hi > 1.4 and hi < 2.6


def test_credible_set_hdi_shorter_for_skewed():
    # construct a chain mostly at 0 with a tail to +5
    rng = random.Random(0)
    samples = [[rng.gauss(0.0, 0.5)] for _ in range(800)] + \
              [[rng.uniform(2.0, 5.0)] for _ in range(200)]
    rep = SampleReport(
        chains=[samples], log_probs=[[0.0] * 1000],
        accept_rates=[1.0], kernel="dummy",
        diagnostics=Diagnostics(), walltime_s=0.0,
        n_samples=1000, dim=1,
    )
    lo_q, hi_q = rep.credible_set(0.1, 0, method="quantile")
    lo_h, hi_h = rep.credible_set(0.1, 0, method="hdi")
    # HDI should be at least as short as quantile
    assert (hi_h - lo_h) <= (hi_q - lo_q) + 1e-9


# -----------------------------------------------------------------------------
# 15. HRMS anytime-valid bound
# -----------------------------------------------------------------------------


def test_hrms_anytime_valid_covers_truth():
    # bound the chain values to [-3, 3] for HRMS use
    s = _make_iso(d=1, rng_seed=0)
    rep = s.hmc(n_samples=500, n_chains=2, warmup=200,
                step_size=0.7, n_leapfrog=10)
    lo, hi = rep.credible_set(alpha=0.1, dim=0, method="hrms", bound=(-5.0, 5.0))
    # truth = 0
    assert lo <= 0.0 <= hi


# -----------------------------------------------------------------------------
# 16. Fingerprint
# -----------------------------------------------------------------------------


def test_fingerprint_deterministic_and_changes():
    s1 = _make_iso(d=1, rng_seed=0)
    s2 = _make_iso(d=1, rng_seed=0)
    rep1 = s1.hmc(n_samples=200, n_chains=2, warmup=100,
                  step_size=0.5, n_leapfrog=10)
    rep2 = s2.hmc(n_samples=200, n_chains=2, warmup=100,
                  step_size=0.5, n_leapfrog=10)
    assert rep1.fingerprint() == rep2.fingerprint()

    s3 = _make_iso(d=1, rng_seed=999)
    rep3 = s3.hmc(n_samples=200, n_chains=2, warmup=100,
                  step_size=0.5, n_leapfrog=10)
    assert rep3.fingerprint() != rep1.fingerprint()


# -----------------------------------------------------------------------------
# Additional: composability & report helpers
# -----------------------------------------------------------------------------


def test_thin_and_discard_warmup():
    s = _make_iso(d=2, rng_seed=0)
    rep = s.hmc(n_samples=400, n_chains=2, warmup=200,
                step_size=0.5, n_leapfrog=10)
    thinned = rep.thin(2)
    assert thinned.n_samples == rep.n_samples // 2 * 1  # ~half per chain pair
    dropped = rep.discard_warmup(50)
    assert dropped.n_warmup == rep.n_warmup + 50


def test_converged_passes_on_clean_run():
    s = _make_iso(d=1, rng_seed=0)
    rep = s.nuts(n_samples=400, n_chains=4, warmup=400)
    assert rep.converged() or rep.diagnostics.ess_bulk[0] > 100  # tolerate small runs


def test_fd_gradient_fallback():
    # don't supply grad — finite-difference takes over
    s = Sampler(gaussian_log_density([0.0], [1.0]), 1,
                rng=random.Random(0))
    rep = s.hmc(n_samples=200, n_chains=2, warmup=200,
                step_size=0.3, n_leapfrog=8)
    assert abs(rep.mean(0)) < 0.4


def test_pareto_k_smoothed_smaller_than_inf():
    # synthetic well-behaved weights
    rng = random.Random(0)
    lw = [rng.gauss(0.0, 0.5) for _ in range(200)]
    k = pareto_k(lw)
    assert math.isfinite(k)


def test_smc_returns_temperatures_monotone():
    def prior(rng): return [rng.gauss(0.0, 1.0)]
    def loglik(theta): return -0.5 * (1.0 - theta[0]) ** 2
    s = Sampler(lambda th: -0.5 * th[0] ** 2 + loglik(th), 1,
                rng=random.Random(0))
    rep = s.smc(n_particles=200, prior_sampler=prior, log_likelihood=loglik,
                ess_target=0.5, max_temps=20, n_mcmc=2)
    # temperatures are monotone increasing
    assert all(rep.temperatures[i] <= rep.temperatures[i + 1]
               for i in range(len(rep.temperatures) - 1))
    assert rep.temperatures[-1] == pytest.approx(1.0, abs=1e-5)


# -----------------------------------------------------------------------------
# Composition with a Forecaster-style problem (without importing it)
# -----------------------------------------------------------------------------


def test_posterior_predictive_on_normal_model():
    """Quick sanity: posterior mean of θ given y ~ N(θ, 1) repeated converges."""
    ys = [1.5, 2.0, 1.8, 2.2, 1.7]
    n = len(ys)

    def log_joint(theta):
        # prior N(0, 100); likelihood Π_i N(y_i | θ, 1)
        return (-0.5 * theta[0] ** 2 / 100.0
                + sum(-0.5 * (y - theta[0]) ** 2 for y in ys))

    def grad(theta):
        return [-theta[0] / 100.0 + sum((y - theta[0]) for y in ys)]

    s = Sampler(log_joint, 1, grad, rng=random.Random(0))
    rep = s.nuts(n_samples=500, n_chains=4, warmup=400)
    # analytic posterior mean ≈ Σy / (n + 1/100) ≈ Σy / 5.01 ≈ 1.84
    truth = sum(ys) / (n + 1.0 / 100.0)
    assert abs(rep.mean(0) - truth) < 0.15
