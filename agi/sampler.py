r"""Sampler — Bayesian probabilistic inference as a runtime primitive.

Every primitive that already lives in this runtime — ``Forecaster`` (a
calibrated belief), ``Causal`` (a posterior over a treatment effect),
``ActiveInferencer`` (a belief over hidden state), ``PolicyImprover`` (a
posterior over policy parameters), ``Persuader`` (Bayesian persuasion
with an uncertain prior), ``CausalDiscoverer`` (a posterior over DAGs),
``Strategist`` (Thompson sampling on a posterior) — *secretly assumes*
that someone, somewhere, can draw samples from a high-dimensional
unnormalised density.  Until now that "someone" was a static prior or
a closed-form conjugate update.  The moment the runtime wants to do
real Bayesian inference on a non-conjugate posterior — and a
coordination engine that wants to *combine* primitives will — it needs
a workhorse posterior sampler.

``Sampler`` is that workhorse.  Given a black-box log-density
``log p̃(θ)`` (unnormalised) on ``R^d`` it returns calibrated samples,
finite-sample-valid convergence diagnostics, and anytime-valid credible
sets — the *give-me-the-posterior-and-prove-it* primitive.

Mathematical roots
------------------

The runtime supports the entire family.  Each kernel below is a
generator of a ``π``-invariant Markov chain on ``R^d`` for the target
``π(θ) ∝ p̃(θ)``.

  * **Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller,
    A. H., Teller, E. (1953).**  *Random-walk Metropolis* (RWM): symmetric
    proposal ``q(θ' | θ) = q(θ | θ')`` accepted with probability
    ``α(θ → θ') = min(1, π(θ')/π(θ))``.

  * **Hastings, W. K. (1970).**  Asymmetric proposal generalisation::

        α(θ → θ')  =  min(1, [π(θ') q(θ | θ')] / [π(θ) q(θ' | θ)]).

  * **Haario, H., Saksman, E., Tamminen, J. (2001).**  *Adaptive
    Metropolis* (AM).  Proposal covariance ``Σ_t`` is updated online
    from the empirical chain covariance::

        q_t(· | θ)  =  N(θ,  s_d Σ̂_t  +  s_d ε I),  s_d = (2.38)² / d.

    AM is **diminishing-adaptation ergodic** (Roberts-Rosenthal 2007)
    when adaptation decays — ``Sampler.rwmh`` ships AM by default.

  * **Roberts, G. O., Tweedie, R. L. (1996) — "Exponential convergence
    of Langevin distributions and their discrete approximations."**
    *Metropolis-adjusted Langevin algorithm* (MALA): drift-corrected
    Gaussian proposal::

        θ'  =  θ  +  (τ/2) ∇ log π(θ)  +  √τ  ξ,  ξ ~ N(0, I),

    accepted with Hastings ratio.  Optimal scaling ``τ ∝ d^{-1/3}``
    (Roberts-Rosenthal 1998); ``Sampler.mala`` adapts via dual averaging.

  * **Durmus, A., Moulines, É. (2017) — "Nonasymptotic convergence
    analysis for the unadjusted Langevin algorithm."**  *ULA*: drop the
    Metropolis correction.  Biased, but for strongly log-concave
    ``π`` the bias is ``O(√τ)`` in 2-Wasserstein.  ``Sampler.ula``
    targets fast, unbiased-in-the-limit warm-up.

  * **Duane, S., Kennedy, A. D., Pendleton, B. J., Roweth, D. (1987)
    — "Hybrid Monte Carlo."**  *HMC*: augment ``θ`` with momentum
    ``r ~ N(0, M)`` and simulate Hamiltonian ``H(θ, r) = -log π(θ) +
    ½ r^T M^{-1} r`` for ``L`` leapfrog steps of size ``ε``, then
    Metropolis-accept::

        α  =  min(1, exp(H(θ, r) - H(θ', r'))).

    Leapfrog is **symplectic** and **time-reversible**, the two
    invariants Metropolis correction needs.

  * **Neal, R. M. (2011) — "MCMC using Hamiltonian dynamics" (Handbook
    of MCMC, chap. 5).**  The canonical reference.  Mass-matrix
    adaptation ``M ≈ Cov_π(θ)^{-1}`` decorrelates dimensions; the
    optimal acceptance is ``≈ 0.65`` for univariate, ``≈ 0.8`` typical.

  * **Hoffman, M. D., Gelman, A. (2014) — "The No-U-Turn Sampler:
    adaptively setting path lengths in Hamiltonian Monte Carlo."**
    *NUTS*: build a binary tree of leapfrog states by recursive
    doubling, stop when the trajectory makes a U-turn::

        (θ⁺ - θ⁻) · r⁻ < 0  or  (θ⁺ - θ⁻) · r⁺ < 0,

    sample uniformly from the slice ``{(θ, r) : u ≤ exp(-H)}``.  The
    Hoffman-Gelman paper additionally introduces **dual-averaging
    step-size adaptation** (Nesterov 2009 primal-dual averaging on
    the log-step) — ``Sampler.nuts`` ships both.  Stan's reference
    sampler is NUTS-with-dual-averaging.

  * **Nesterov, Y. (2009) — "Primal-dual subgradient methods for
    convex problems."**  Dual averaging on ``log ε`` with target
    acceptance ``δ``::

        H̄_m  =  (1 - 1/(m+t₀))H̄_{m-1} + (δ - α_m)/(m+t₀),
        log ε_m   =  μ - √m / γ  · H̄_m,
        log ε̄_m  =  m^{-κ} log ε_m + (1 - m^{-κ}) log ε̄_{m-1},

    with ``γ = 0.05``, ``t₀ = 10``, ``κ = 0.75``, ``μ = log(10ε₀)``
    (Hoffman-Gelman Algorithm 6).  Convergence after ``M_adapt``
    warmup iterations is guaranteed by Nesterov's analysis.

  * **Neal, R. M. (2003) — "Slice sampling" (Annals of Statistics
    31:705).**  Auxiliary uniform on ``[0, π(θ)]``, then sample
    ``θ'`` uniformly from the slice.  *Stepping-out* + *shrinkage*
    procedure gives a kernel with **no tuning parameters**::

        u  ~  Uniform(0, π(θ)),
        θ' ~  Uniform({θ : π(θ) ≥ u}).

    Doss-coordinate update across dimensions makes it Gibbs-like for
    high dimensions.

  * **Geman, S., Geman, D. (1984) — "Stochastic relaxation, Gibbs
    distributions, and the Bayesian restoration of images."**  *Gibbs*:
    cycle through coordinates, sample each from its full conditional
    ``π(θ_i | θ_{-i})`` — slice sampling along each axis gives a
    **gradient-free** Gibbs when the conditionals are not available
    in closed form.

  * **Earl, D. J., Deem, M. W. (2005) — "Parallel tempering: theory,
    applications, and new perspectives."**  *Replica exchange* runs
    chains at inverse temperatures ``1 = β_0 > β_1 > … > β_K`` against
    targets ``π_k(θ) ∝ π(θ)^{β_k}``; pairs of adjacent chains swap
    with probability::

        α_swap  =  min(1, exp((β_k - β_{k+1})(log π(θ_k) - log π(θ_{k+1})))).

    Cold chain (``β_0 = 1``) samples ``π``; hot chains explore
    multimodal landscapes.  ``Sampler.parallel_tempering`` runs an
    arbitrary base kernel with an automatic geometric temperature
    ladder (Atchadé-Roberts-Rosenthal 2011 optimal-spacing rule).

  * **Doucet, A., de Freitas, N., Gordon, N. (2001) — "Sequential
    Monte Carlo methods in practice."**  *SMC* tempers from prior
    ``π_0`` to posterior ``π_N`` along a sequence of intermediate
    targets ``π_n(θ) ∝ π_0(θ) · L(θ)^{β_n}``, ``0 = β_0 < β_1 <
    … < β_N = 1``.  Iterate (i) reweight, (ii) resample if
    ``ESS < N/2``, (iii) move with an MCMC kernel.  Returns an
    unbiased estimate of the *normalising constant*::

        Ẑ  =  ∏_{n=1}^{N} (1/M) Σ_m w_n^{(m)} / (1/M Σ_m w_{n-1}^{(m)}),

    which ``Forecaster`` and ``Causal`` need for Bayes factors.

  * **Owen, A. B. (2013) — *Monte Carlo theory, methods and examples*,
    chap. 9.**  *Importance sampling* with proposal ``q``::

        Î_IS = (1/N) Σ_n  [π(θ_n)/q(θ_n)] f(θ_n),  θ_n ~ q.

    The **Pareto-smoothed importance sampling** diagnostic ``k̂``
    (Vehtari, Simpson, Gelman, Yao, Gabry 2024) flags ``k̂ > 0.7``
    as untrustworthy — ``Sampler.importance`` ships it.

  * **Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., Blei, D. M.
    (2017) — "Automatic Differentiation Variational Inference."**
    *ADVI*: mean-field ``q_φ(θ) = ∏_d N(μ_d, σ_d²)`` in an unconstrained
    space, maximised by stochastic gradient ascent on the ELBO::

        ℒ(φ)  =  E_{q_φ}[ log p̃(T^{-1}(η)) + log |det J_{T^{-1}}(η)| ]
                 + 0.5 Σ_d log(2π e σ_d²).

    Reparameterisation ``η = μ + σ ⊙ ε``, ``ε ~ N(0, I)``, decouples
    the gradient.  ``Sampler.advi`` ships mean-field + full-rank
    Gaussian variants with AdaGrad step.

Convergence diagnostics — anytime-valid because each is monotone or has
explicit error bars
-------------------

  * **Gelman, A., Rubin, D. B. (1992).**  ``R̂``: pooled-over-chains
    variance / within-chain variance.  ``R̂ → 1`` as chains mix.

  * **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner,
    P.-C. (2021) — "Rank-normalisation, folding, and localisation: an
    improved R̂."**  *Split-R̂* on rank-normalised + folded chains is
    the modern recommendation; ``R̂ > 1.01`` is the **non-convergence
    flag** Stan uses by default.  Ships in ``rhat()``.

  * **Vehtari et al. 2021.**  *Bulk-ESS* (rank-normalised) and
    *tail-ESS* (computed on indicators ``θ < q_α`` and ``θ > q_{1-α}``)
    — the *quantile* ESS, the right diagnostic for credible intervals.

  * **Geyer, C. J. (1992) — "Practical Markov chain Monte Carlo."**
    Initial monotone sequence estimator for the integrated
    autocorrelation time ``τ_int``; the asymptotic-variance estimator
    behind all MCMC ESS computations.

  * **Geweke, J. (1992) — "Evaluating the accuracy of sampling-based
    approaches to the calculation of posterior moments."**  ``Z``-test
    on first 10% vs. last 50% of a chain; ``|z| > 2`` flags
    non-stationarity.

  * **Heidelberger, P., Welch, P. D. (1983) — "Simulation run length
    control in the presence of an initial transient."**  Cramér-von
    Mises stationarity test with a half-width relative-precision
    criterion.

  * **Vehtari, Simpson, Gelman, Yao, Gabry (2024) — "Pareto-smoothed
    importance sampling."**  Generalised-Pareto tail-shape ``k̂``
    diagnoses unreliable IS / VI estimates.  Ships in ``pareto_k()``.

Anytime-valid credible sets
---------------------------

Once we have ``N`` post-warm-up samples ``{θ_n}`` with ESS ``ess > 0``,
the empirical ``α``-quantile of any bounded ``f(θ) ∈ [a, b]`` admits a
**finite-sample, distribution-free** confidence interval via the
Massart-tightened DKW inequality on the empirical CDF::

    P( sup_x |F̂_n(x) - F(x)|  ≥  ε )  ≤  2 exp(-2 ε² n),

— and *anytime-valid* sequential refinement via the Howard-Ramdas-
McAuliffe-Sekhon (2021) bounded-mean confidence sequence::

    μ ∈ X̄_n  ±  (b - a) √( 2 log log(2n) + log(2/δ) ) / √n.

Both ship in ``SampleReport.credible_set(α, dim, method=…)`` — *the
posterior CI you can quote to the regulator*.

Composition
-----------

  * **Forecaster** — uses ``Sampler.smc`` to compute posterior
    predictive scoring and Bayes factor for ensemble selection.
  * **Causal** — uses ``Sampler.nuts`` to draw posterior treatment
    effects from non-conjugate prior + likelihood, then composes the
    standard ``BLP / Qini`` CIs on top of *posterior* draws (not just
    point estimates).
  * **ActiveInferencer** — uses ``Sampler.smc`` (particle filter
    variant ``smc.particle_filter``) for sequential belief update.
  * **PolicyImprover** — uses ``Sampler.advi`` to amortise the safe
    policy posterior across many tickets.
  * **Strategist** — uses ``Sampler.nuts`` posterior draws as the
    *Thompson sampling* source, with the runtime guarantee that
    ``R̂ < 1.01`` and ``ess > 400`` before any draw is used.
  * **Persuader** — uses ``Sampler.smc`` to draw uncertain priors
    over receiver types in robust persuasion.
  * **CausalDiscoverer** — uses ``Sampler.parallel_tempering`` to
    sample DAGs from the BGe / BDeu posterior across multimodal
    graph space.
  * **AttestationLedger** — every ``SampleReport`` ships a tamper-
    evident fingerprint over kernel, seed, ``N``, ``R̂``, ``ess`` so
    the receipt is *reproducible*.

Public API
----------

::

    >>> def log_p(theta): return -0.5 * sum(t*t for t in theta)    # N(0, I)
    >>> S = Sampler(log_p, dim=2)
    >>> rep = S.nuts(n_samples=2000, n_chains=4, warmup=1000)
    >>> rep.mean()            # ~ [0, 0]
    >>> rep.diagnostics.rhat  # ~ [1.00, 1.00]
    >>> rep.credible_set(alpha=0.05, dim=0)  # ~ (-1.96, 1.96)

All randomness flows through a user-supplied ``random.Random`` (default
``random.Random(0)`` for reproducibility).  All gradients are computed
analytically from a user-supplied ``grad_log_density`` callable or fall
back to central finite differences with adaptive step.  Pure stdlib —
no NumPy / no SciPy / no autograd.

The primitive is **deliberately conservative**: dual-averaging, R̂
gating, Pareto-k flagging, ESS thresholding, divergence counting.  When
``Sampler`` says ``converged=True`` it means *all four* of (R̂ < 1.01,
bulk-ESS ≥ 400, divergence rate < 1%, no max-tree-depth-saturation).
"""

from __future__ import annotations

import math
import random
import statistics
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


# =============================================================================
# Linear algebra (stdlib)
# =============================================================================

Vec = list[float]
Mat = list[list[float]]


def _vec_add(a: Sequence[float], b: Sequence[float]) -> Vec:
    return [x + y for x, y in zip(a, b)]


def _vec_sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return [x - y for x, y in zip(a, b)]


def _vec_scale(a: Sequence[float], s: float) -> Vec:
    return [x * s for x in a]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _zeros(d: int) -> Vec:
    return [0.0] * d


def _eye(d: int) -> Mat:
    return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]


def _mat_vec(M: Mat, v: Sequence[float]) -> Vec:
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _diag_mat_vec(diag: Sequence[float], v: Sequence[float]) -> Vec:
    return [d * x for d, x in zip(diag, v)]


def _cholesky(A: Mat) -> Mat:
    """Lower-triangular Cholesky factor; raises on non-PD."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 0:
                    raise ValueError("non-positive-definite matrix")
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _solve_lower(L: Mat, b: Sequence[float]) -> Vec:
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]
    return x


# =============================================================================
# Gradient (analytic or central finite difference)
# =============================================================================


def _fd_grad(
    f: Callable[[Sequence[float]], float],
    theta: Sequence[float],
    eps_rel: float = 1e-5,
) -> Vec:
    """Central-difference gradient with adaptive per-dim step."""
    d = len(theta)
    g = [0.0] * d
    for i in range(d):
        h = eps_rel * max(1.0, abs(theta[i]))
        tp = list(theta); tp[i] += h
        tm = list(theta); tm[i] -= h
        g[i] = (f(tp) - f(tm)) / (2.0 * h)
    return g


# =============================================================================
# Public dataclasses
# =============================================================================


@dataclass
class Diagnostics:
    """Per-dimension convergence + per-kernel divergence diagnostics."""
    rhat: list[float] = field(default_factory=list)
    ess_bulk: list[float] = field(default_factory=list)
    ess_tail: list[float] = field(default_factory=list)
    geweke_z: list[float] = field(default_factory=list)
    autocorr_time: list[float] = field(default_factory=list)
    divergences: int = 0
    max_tree_depth_hits: int = 0
    swap_accept_rate: float | None = None
    pareto_k: float | None = None
    ess_normalising: float | None = None  # SMC

    def converged(
        self,
        rhat_tol: float = 1.01,
        min_bulk_ess: float = 400.0,
        max_div_rate: float = 0.01,
        n_samples: int = 1,
    ) -> bool:
        """Return True iff all conventional diagnostics pass."""
        if any(r > rhat_tol or r != r for r in self.rhat):
            return False
        if any(e < min_bulk_ess for e in self.ess_bulk):
            return False
        if self.divergences / max(1, n_samples) > max_div_rate:
            return False
        if self.max_tree_depth_hits / max(1, n_samples) > 0.05:
            return False
        return True


@dataclass
class SampleReport:
    """The deliverable: chains, log-densities, diagnostics, fingerprint.

    ``chains[c][n][d]`` is the ``d``-th coordinate of the ``n``-th draw
    of chain ``c``.  ``warmup`` is the number of *already-discarded*
    warmup iterations.
    """
    chains: list[list[Vec]]
    log_probs: list[list[float]]
    accept_rates: list[float]
    kernel: str
    diagnostics: Diagnostics
    walltime_s: float
    n_warmup: int = 0
    step_size: float | None = None
    n_leapfrog: int | float | None = None
    mass: list[float] | None = None  # diagonal mass
    seed: int = 0
    n_samples: int = 0
    dim: int = 0

    # ---------------------------------------------------------------
    # post-processing helpers
    # ---------------------------------------------------------------

    def flat(self) -> list[Vec]:
        out: list[Vec] = []
        for ch in self.chains:
            out.extend(ch)
        return out

    def thin(self, k: int) -> "SampleReport":
        if k <= 1:
            return self
        new_chains = [ch[::k] for ch in self.chains]
        new_lps = [lp[::k] for lp in self.log_probs]
        return SampleReport(
            chains=new_chains, log_probs=new_lps,
            accept_rates=self.accept_rates, kernel=self.kernel + f"+thin({k})",
            diagnostics=_diagnose(new_chains),
            walltime_s=self.walltime_s, n_warmup=self.n_warmup,
            step_size=self.step_size, n_leapfrog=self.n_leapfrog,
            mass=self.mass, seed=self.seed,
            n_samples=sum(len(c) for c in new_chains),
            dim=self.dim,
        )

    def discard_warmup(self, n: int) -> "SampleReport":
        new_chains = [ch[n:] for ch in self.chains]
        new_lps = [lp[n:] for lp in self.log_probs]
        return SampleReport(
            chains=new_chains, log_probs=new_lps,
            accept_rates=self.accept_rates, kernel=self.kernel,
            diagnostics=_diagnose(new_chains),
            walltime_s=self.walltime_s,
            n_warmup=self.n_warmup + n,
            step_size=self.step_size, n_leapfrog=self.n_leapfrog,
            mass=self.mass, seed=self.seed,
            n_samples=sum(len(c) for c in new_chains),
            dim=self.dim,
        )

    # ---------------------------------------------------------------
    # moments
    # ---------------------------------------------------------------

    def mean(self, dim: int | None = None) -> Any:
        flat = self.flat()
        if not flat:
            return [0.0] * self.dim if dim is None else 0.0
        if dim is None:
            return [statistics.fmean(s[i] for s in flat) for i in range(self.dim)]
        return statistics.fmean(s[dim] for s in flat)

    def var(self, dim: int | None = None) -> Any:
        flat = self.flat()
        n = len(flat)
        if n < 2:
            return [0.0] * self.dim if dim is None else 0.0
        mu = self.mean()
        if dim is None:
            mu_l = mu if isinstance(mu, list) else [mu]
            return [
                sum((s[i] - mu_l[i]) ** 2 for s in flat) / (n - 1)
                for i in range(self.dim)
            ]
        return sum((s[dim] - mu[dim]) ** 2 for s in flat) / (n - 1)

    def std(self, dim: int | None = None) -> Any:
        v = self.var(dim)
        if isinstance(v, list):
            return [math.sqrt(x) for x in v]
        return math.sqrt(v)

    def quantile(self, q: float, dim: int) -> float:
        flat = sorted(s[dim] for s in self.flat())
        if not flat:
            return float("nan")
        if len(flat) == 1:
            return flat[0]
        h = q * (len(flat) - 1)
        lo = int(h)
        hi = min(lo + 1, len(flat) - 1)
        return flat[lo] * (hi - h) + flat[hi] * (h - lo)

    def credible_set(
        self,
        alpha: float,
        dim: int,
        method: str = "quantile",
        bound: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Posterior ``1-α`` credible interval for coordinate ``dim``.

        ``method``:
          * ``"quantile"`` — equal-tailed quantile CI.
          * ``"hdi"``      — highest-density (shortest) interval.
          * ``"dkw"``      — Massart-DKW finite-sample CDF band on
                             the empirical quantile (requires ``bound``).
          * ``"hrms"``     — Howard-Ramdas-McAuliffe-Sekhon anytime-valid
                             bounded-mean confidence sequence on the mean
                             (requires ``bound``).
        """
        flat = sorted(s[dim] for s in self.flat())
        n = len(flat)
        if n == 0:
            return (float("nan"), float("nan"))
        a = alpha / 2.0
        if method == "quantile":
            return (self.quantile(a, dim), self.quantile(1 - a, dim))
        if method == "hdi":
            return _hdi(flat, alpha)
        if method == "dkw":
            if bound is None:
                raise ValueError("dkw requires bound=(lo, hi)")
            lo, hi = bound
            # tightened DKW
            eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
            lo_idx = max(0, int((a - eps) * n))
            hi_idx = min(n - 1, int(math.ceil((1 - a + eps) * n)))
            return (max(lo, flat[lo_idx]), min(hi, flat[hi_idx]))
        if method == "hrms":
            if bound is None:
                raise ValueError("hrms requires bound=(lo, hi)")
            lo, hi = bound
            mu = self.mean(dim)
            half = (hi - lo) * math.sqrt(
                (2.0 * math.log(math.log(max(2.0, 2 * n)))
                 + math.log(2.0 / alpha)) / max(1, n)
            )
            return (mu - half, mu + half)
        raise ValueError(f"unknown method: {method}")

    # ---------------------------------------------------------------
    # diagnostics shortcuts
    # ---------------------------------------------------------------

    def ess(self, dim: int, kind: str = "bulk") -> float:
        if kind == "bulk":
            return self.diagnostics.ess_bulk[dim]
        if kind == "tail":
            return self.diagnostics.ess_tail[dim]
        raise ValueError(f"unknown ess kind: {kind}")

    def rhat(self, dim: int) -> float:
        return self.diagnostics.rhat[dim]

    def converged(self, **kw) -> bool:
        return self.diagnostics.converged(n_samples=self.n_samples, **kw)

    # ---------------------------------------------------------------
    # tamper-evident fingerprint
    # ---------------------------------------------------------------

    def fingerprint(self) -> str:
        payload = json.dumps({
            "kernel": self.kernel,
            "seed": self.seed,
            "n_samples": self.n_samples,
            "n_warmup": self.n_warmup,
            "dim": self.dim,
            "step_size": self.step_size,
            "n_leapfrog": self.n_leapfrog,
            "rhat": [round(r, 6) for r in self.diagnostics.rhat],
            "ess_bulk": [round(e, 3) for e in self.diagnostics.ess_bulk],
            "divergences": self.diagnostics.divergences,
            "max_tree_depth_hits": self.diagnostics.max_tree_depth_hits,
            "mean": [round(m, 6) for m in
                     (self.mean() if isinstance(self.mean(), list)
                      else [self.mean()])],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ImportanceReport:
    log_weights: list[float]
    ess: float
    pareto_k: float
    log_norm_const_estimate: float
    samples: list[Vec]

    def estimate(self, f: Callable[[Sequence[float]], float]) -> float:
        """Self-normalised IS estimate of ``E_π[f]``."""
        w = _normalise_log_weights(self.log_weights)
        return sum(wi * f(s) for wi, s in zip(w, self.samples))


@dataclass
class SMCReport:
    """SMC tempering result: posterior samples + log marginal likelihood."""
    samples: list[Vec]
    log_weights: list[float]
    log_evidence: float
    ess_history: list[float]
    temperatures: list[float]
    walltime_s: float
    n_resample: int
    diagnostics: Diagnostics

    def mean(self, dim: int | None = None) -> Any:
        w = _normalise_log_weights(self.log_weights)
        d = len(self.samples[0])
        if dim is None:
            return [sum(wi * s[i] for wi, s in zip(w, self.samples))
                    for i in range(d)]
        return sum(wi * s[dim] for wi, s in zip(w, self.samples))

    def var(self, dim: int) -> float:
        mu = self.mean(dim)
        w = _normalise_log_weights(self.log_weights)
        return sum(wi * (s[dim] - mu) ** 2 for wi, s in zip(w, self.samples))


@dataclass
class ADVIReport:
    mu: Vec
    sigma: Vec
    L_lowrank: Mat | None  # full-rank cholesky factor (None for mean-field)
    elbo_history: list[float]
    walltime_s: float
    n_iter: int

    def sample(self, n: int, rng: random.Random | None = None) -> list[Vec]:
        rng = rng or random.Random()
        d = len(self.mu)
        out: list[Vec] = []
        for _ in range(n):
            eps = [rng.gauss(0.0, 1.0) for _ in range(d)]
            if self.L_lowrank is None:
                out.append([m + s * e for m, s, e in zip(self.mu, self.sigma, eps)])
            else:
                # mu + L @ eps
                z = [sum(self.L_lowrank[i][j] * eps[j] for j in range(d))
                     for i in range(d)]
                out.append([m + zi for m, zi in zip(self.mu, z)])
        return out

    def mean(self) -> Vec:
        return list(self.mu)


# =============================================================================
# Diagnostics (rank-normalised split-R̂, bulk/tail-ESS, Geweke, …)
# =============================================================================


def _split_chains(chains: list[list[Vec]]) -> list[list[Vec]]:
    """Vehtari et al. split each chain in half (improves R̂ sensitivity)."""
    out: list[list[Vec]] = []
    for ch in chains:
        n = len(ch)
        if n < 4:
            out.append(ch)
            continue
        mid = n // 2
        out.append(ch[:mid])
        out.append(ch[mid:])
    return out


def _rank_normalize(values: list[float]) -> list[float]:
    """Rank-normalise (Vehtari 2021): rank, then map to N(0,1) ICDF."""
    n = len(values)
    if n == 0:
        return values
    idx = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    # average ranks for ties
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    # invert normal CDF on (r - 3/8) / (n + 1/4) — Blom plotting position
    return [_inv_norm_cdf((r - 3.0 / 8.0) / (n + 0.25)) for r in ranks]


def _inv_norm_cdf(p: float) -> float:
    """Beasley-Springer-Moro inverse normal CDF."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def _rhat_single(chains: list[list[float]]) -> float:
    """Gelman-Rubin R̂ for one coordinate.  Returns NaN for singletons."""
    if not chains or len(chains) < 2:
        return float("nan")
    n = min(len(c) for c in chains)
    if n < 2:
        return float("nan")
    m = len(chains)
    chain_means = [statistics.fmean(c[:n]) for c in chains]
    chain_vars = [
        sum((x - chain_means[i]) ** 2 for x in c[:n]) / (n - 1)
        for i, c in enumerate(chains)
    ]
    W = statistics.fmean(chain_vars)
    grand = statistics.fmean(chain_means)
    B = n * sum((cm - grand) ** 2 for cm in chain_means) / (m - 1)
    if W <= 0:
        return float("nan")
    var_plus = (n - 1) / n * W + B / n
    return math.sqrt(var_plus / W)


def rhat(chains: list[list[float]], split: bool = True,
         rank_normalize: bool = True) -> float:
    """Modern split-rank-R̂ (Vehtari et al. 2021)."""
    chs = chains
    if split:
        chs = []
        for c in chains:
            n = len(c)
            if n < 4:
                chs.append(c)
            else:
                m = n // 2
                chs.append(c[:m]); chs.append(c[m:])
    if rank_normalize:
        flat = [x for c in chs for x in c]
        if len(set(flat)) <= 1:
            return float("nan")
        ranks = _rank_normalize(flat)
        # split back
        chs2 = []
        i = 0
        for c in chs:
            chs2.append(ranks[i:i + len(c)])
            i += len(c)
        chs = chs2
    return _rhat_single(chs)


def autocorr(chain: list[float], max_lag: int | None = None) -> list[float]:
    """Biased autocorrelation function (Geyer initial monotone)."""
    n = len(chain)
    if n < 2:
        return [1.0]
    if max_lag is None:
        max_lag = min(n - 1, 1000)
    mu = statistics.fmean(chain)
    var = sum((x - mu) ** 2 for x in chain) / n
    if var <= 0:
        return [1.0]
    out = [1.0]
    for lag in range(1, max_lag + 1):
        s = sum((chain[t] - mu) * (chain[t + lag] - mu) for t in range(n - lag))
        out.append(s / (n * var))
    return out


def autocorr_time(chain: list[float]) -> float:
    """Geyer (1992) initial monotone-positive-sequence estimator.

    Sum ``Γ_k = ρ_{2k} + ρ_{2k+1}`` pairs until the first ``Γ_k ≤ 0``;
    enforce monotone decreasing Γ_k thereafter.  Returns ``τ_int =
    1 + 2 Σ_{lag=1} ρ_lag``.
    """
    n = len(chain)
    if n < 4:
        return 1.0
    rho = autocorr(chain, max_lag=min(n - 1, 1000))
    if len(rho) < 3:
        return 1.0
    # build initial-monotone-sequence cap
    gammas: list[float] = []
    k = 0
    while 2 * k + 2 < len(rho):
        g = rho[2 * k + 1] + rho[2 * k + 2]
        if g <= 0:
            break
        if gammas:
            g = min(g, gammas[-1])  # monotone
        gammas.append(g)
        k += 1
    tau = -1.0 + 2.0 * sum(gammas) + 2.0 * rho[0]  # rho[0] = 1
    return max(1.0, tau)


def ess_bulk(chains: list[list[float]]) -> float:
    """Rank-normalised bulk-ESS (Vehtari 2021).  Min over chains × n
    divided by the integrated autocorrelation time."""
    if not chains:
        return 0.0
    flat = [x for c in chains for x in c]
    if len(set(flat)) <= 1:
        return float("nan")
    ranks = _rank_normalize(flat)
    chs2 = []
    i = 0
    for c in chains:
        chs2.append(ranks[i:i + len(c)])
        i += len(c)
    n = min(len(c) for c in chs2)
    m = len(chs2)
    total = m * n
    # combined autocorr time (per-chain mean of τ)
    taus = [autocorr_time(c[:n]) for c in chs2]
    tau = statistics.fmean(taus)
    return total / max(1.0, tau)


def ess_tail(chains: list[list[float]], alpha: float = 0.05) -> float:
    """Tail-ESS: ESS of indicator below 5%-quantile and above 95%-quantile,
    take the minimum (Vehtari 2021)."""
    if not chains:
        return 0.0
    flat = sorted(x for c in chains for x in c)
    if len(flat) < 4:
        return 0.0
    n = len(flat)
    q_lo = flat[max(0, int(alpha * n))]
    q_hi = flat[min(n - 1, int((1 - alpha) * n))]
    ind_lo = [[1.0 if x <= q_lo else 0.0 for x in c] for c in chains]
    ind_hi = [[1.0 if x >= q_hi else 0.0 for x in c] for c in chains]
    # rank-normalisation collapses to constant for {0, 1}; skip and use raw ess
    n_per = min(len(c) for c in chains)
    m = len(chains)
    total = m * n_per

    def _raw_ess(chs):
        taus = [autocorr_time(c[:n_per]) for c in chs]
        tau = statistics.fmean(taus)
        return total / max(1.0, tau)

    return min(_raw_ess(ind_lo), _raw_ess(ind_hi))


def geweke_z(chain: list[float],
             first_frac: float = 0.1, last_frac: float = 0.5) -> float:
    """Geweke (1992) stationarity z-score."""
    n = len(chain)
    n1 = max(2, int(first_frac * n))
    n2 = max(2, int(last_frac * n))
    a = chain[:n1]
    b = chain[-n2:]
    m1 = statistics.fmean(a); m2 = statistics.fmean(b)
    # spectral variance ~ var * τ
    tau1 = autocorr_time(a); tau2 = autocorr_time(b)
    v1 = statistics.pvariance(a) * tau1 / len(a) if len(a) > 1 else 1.0
    v2 = statistics.pvariance(b) * tau2 / len(b) if len(b) > 1 else 1.0
    se = math.sqrt(max(1e-12, v1 + v2))
    return (m1 - m2) / se


def _diagnose(chains: list[list[Vec]]) -> Diagnostics:
    """Build a full ``Diagnostics`` object from raw chains."""
    if not chains or not chains[0]:
        return Diagnostics()
    d = len(chains[0][0])
    rhats: list[float] = []
    bulks: list[float] = []
    tails: list[float] = []
    gews: list[float] = []
    taus: list[float] = []
    for i in range(d):
        per = [[s[i] for s in ch] for ch in chains]
        rhats.append(rhat(per))
        bulks.append(ess_bulk(per))
        tails.append(ess_tail(per))
        gews.append(geweke_z(per[0]) if per else float("nan"))
        taus.append(autocorr_time(per[0]) if per[0] else 1.0)
    return Diagnostics(
        rhat=rhats, ess_bulk=bulks, ess_tail=tails,
        geweke_z=gews, autocorr_time=taus,
    )


# =============================================================================
# Highest-density interval
# =============================================================================


def _hdi(sorted_flat: list[float], alpha: float) -> tuple[float, float]:
    n = len(sorted_flat)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (sorted_flat[0], sorted_flat[0])
    k = max(1, int(math.ceil((1 - alpha) * n)))
    best_lo, best_hi = sorted_flat[0], sorted_flat[-1]
    best_width = float("inf")
    for i in range(n - k + 1):
        lo = sorted_flat[i]
        hi = sorted_flat[i + k - 1]
        if hi - lo < best_width:
            best_width = hi - lo
            best_lo, best_hi = lo, hi
    return (best_lo, best_hi)


# =============================================================================
# Pareto-smoothed importance sampling (Vehtari et al. 2024)
# =============================================================================


def _normalise_log_weights(lws: Sequence[float]) -> list[float]:
    if not lws:
        return []
    m = max(lws)
    raw = [math.exp(lw - m) for lw in lws]
    s = sum(raw)
    if s <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [r / s for r in raw]


def _ess_is(lws: Sequence[float]) -> float:
    w = _normalise_log_weights(lws)
    return 1.0 / sum(wi * wi for wi in w) if w else 0.0


def _gpd_fit_pwm(tail: list[float]) -> tuple[float, float]:
    """Hosking-Wallis 1987 Probability-Weighted-Moments GPD fit.  Returns
    ``(k_hat, sigma_hat)`` in Vehtari's PSIS convention where ``k̂ > 0`` is a
    *heavy-tailed* signal.  Robust to small samples; defined for all ξ."""
    n = len(tail)
    if n < 3:
        return (float("inf"), 1.0)
    x = sorted(tail)
    # shift so origin at 0
    if x[0] < 0:
        x = [v - x[0] + 1e-12 for v in x]
    # degenerate: all values equal
    if x[-1] - x[0] < 1e-12 * max(1.0, abs(x[-1])):
        return (0.0, max(1e-12, x[-1]))
    # PWM: a₀ = mean(x); a₁ = (1/n) Σ x_i (1 - (i - 0.5)/n)
    a0 = statistics.fmean(x)
    a1 = statistics.fmean(x[i] * (1 - (i + 0.5) / n) for i in range(n))
    denom = a0 - 2 * a1
    if abs(denom) < 1e-12:
        return (0.0, max(1e-12, a0))
    k = a0 / denom - 2.0
    sigma = 2 * a0 * a1 / denom
    if sigma <= 0:
        sigma = max(1e-12, a0)
    return (k, sigma)


def pareto_k(log_weights: Sequence[float]) -> float:
    """Pareto-smoothed-IS shape parameter ``k̂`` (Vehtari et al. 2024).
    ``k̂ > 0.7`` is the reliability cutoff: above it, the IS / VI estimate
    is *not* trustworthy.  Returns ``0`` for degenerate (equal) weights."""
    if len(log_weights) < 5:
        return float("inf")
    sorted_lw = sorted(log_weights, reverse=True)
    n = len(sorted_lw)
    m = max(5, int(min(n / 5.0, 3.0 * math.sqrt(n))))
    # degenerate: all log-weights equal
    if sorted_lw[0] - sorted_lw[-1] < 1e-12:
        return 0.0
    # exponentiate tail above threshold = (m+1)-th largest weight
    threshold_lw = sorted_lw[m]
    tail = [math.exp(lw - threshold_lw) for lw in sorted_lw[:m]]
    # shift so smallest tail value is at 0
    tmin = min(tail)
    tail = [t - tmin + 1e-12 for t in tail]
    k_hat, _ = _gpd_fit_pwm(tail)
    return k_hat


# =============================================================================
# Core sampler
# =============================================================================


class Sampler:
    """Black-box Bayesian sampler over an arbitrary log-density.

    Construct with a callable ``log_density(theta) -> float`` returning
    ``log p̃(θ)`` (unnormalised) and the parameter dimension.  Pass an
    optional ``grad_log_density(theta) -> list[float]`` to skip the
    finite-difference fallback for gradient-based kernels (HMC, NUTS,
    MALA, ULA, ADVI).
    """

    def __init__(
        self,
        log_density: Callable[[Sequence[float]], float],
        dim: int,
        grad_log_density: Callable[[Sequence[float]], list[float]] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.log_density = log_density
        self.dim = dim
        self.rng = rng or random.Random(0)
        self._user_grad = grad_log_density

    def grad_log_density(self, theta: Sequence[float]) -> Vec:
        if self._user_grad is not None:
            g = self._user_grad(theta)
            if len(g) != self.dim:
                raise ValueError("grad returned wrong dimension")
            return list(g)
        return _fd_grad(self.log_density, theta)

    # ===============================================================
    # Random-walk Metropolis (Haario adaptive)
    # ===============================================================

    def rwmh(
        self,
        n_samples: int,
        n_chains: int = 4,
        warmup: int | None = None,
        init: Sequence[Sequence[float]] | None = None,
        proposal_sd: float = 0.5,
        adapt: bool = True,
        target_accept: float = 0.234,
    ) -> SampleReport:
        """Random-walk Metropolis with Haario-Saksman-Tamminen 2001
        adaptive proposal covariance.  Optimal acceptance ``≈ 0.234``."""
        t0 = time.time()
        if warmup is None:
            warmup = n_samples // 2

        inits = self._init_chains(init, n_chains)
        chains: list[list[Vec]] = [[] for _ in range(n_chains)]
        log_probs: list[list[float]] = [[] for _ in range(n_chains)]
        accept_counts = [0] * n_chains
        # per-chain proposal sd (Haario)
        sd = [proposal_sd] * n_chains
        # running mean & covariance (Welford) per chain
        run_mean = [list(inits[c]) for c in range(n_chains)]
        run_M2 = [[[0.0] * self.dim for _ in range(self.dim)]
                  for _ in range(n_chains)]
        run_n = [0] * n_chains

        total = warmup + n_samples

        for c in range(n_chains):
            theta = list(inits[c])
            lp = self.log_density(theta)
            for it in range(total):
                # Welford update
                run_n[c] += 1
                delta = _vec_sub(theta, run_mean[c])
                run_mean[c] = _vec_add(run_mean[c],
                                        _vec_scale(delta, 1.0 / run_n[c]))
                delta2 = _vec_sub(theta, run_mean[c])
                for i in range(self.dim):
                    for j in range(self.dim):
                        run_M2[c][i][j] += delta[i] * delta2[j]
                # proposal: isotropic for warmup/3, then AM
                use_am = adapt and it >= max(20, warmup // 3) and run_n[c] > 2 * self.dim
                if use_am:
                    cov = [[run_M2[c][i][j] / (run_n[c] - 1)
                            for j in range(self.dim)] for i in range(self.dim)]
                    s_d = (2.38 ** 2) / self.dim
                    cov = [[s_d * cov[i][j] + (s_d * 1e-6 if i == j else 0.0)
                            for j in range(self.dim)] for i in range(self.dim)]
                    try:
                        L = _cholesky(cov)
                    except ValueError:
                        L = None
                    if L is not None:
                        eps = [self.rng.gauss(0.0, 1.0) for _ in range(self.dim)]
                        prop = _vec_add(theta, _mat_vec(L, eps))
                    else:
                        eps = [self.rng.gauss(0.0, sd[c]) for _ in range(self.dim)]
                        prop = _vec_add(theta, eps)
                else:
                    eps = [self.rng.gauss(0.0, sd[c]) for _ in range(self.dim)]
                    prop = _vec_add(theta, eps)
                try:
                    lp_prop = self.log_density(prop)
                except (ValueError, OverflowError):
                    lp_prop = float("-inf")
                log_alpha = lp_prop - lp
                if math.log(self.rng.random() + 1e-300) < log_alpha:
                    theta = prop
                    lp = lp_prop
                    if it >= warmup:
                        accept_counts[c] += 1
                    elif adapt and not use_am:
                        sd[c] *= math.exp(min(0.1, 1.0 / math.sqrt(it + 1)))
                else:
                    if it < warmup and adapt and not use_am:
                        sd[c] *= math.exp(-min(0.1, target_accept /
                                                math.sqrt(it + 1)))
                if it >= warmup:
                    chains[c].append(list(theta))
                    log_probs[c].append(lp)

        return SampleReport(
            chains=chains, log_probs=log_probs,
            accept_rates=[accept_counts[c] / max(1, n_samples)
                          for c in range(n_chains)],
            kernel="rwmh", diagnostics=_diagnose(chains),
            walltime_s=time.time() - t0,
            n_warmup=warmup, step_size=statistics.fmean(sd),
            seed=self.rng.randint(0, 1 << 30),
            n_samples=n_samples * n_chains, dim=self.dim,
        )

    # ===============================================================
    # MALA (Metropolis-adjusted Langevin)
    # ===============================================================

    def mala(
        self,
        n_samples: int,
        n_chains: int = 4,
        warmup: int | None = None,
        step_size: float = 0.1,
        init: Sequence[Sequence[float]] | None = None,
        adapt: bool = True,
        target_accept: float = 0.574,
    ) -> SampleReport:
        """Metropolis-adjusted Langevin algorithm (Roberts-Tweedie 1996).
        Optimal acceptance ``≈ 0.574`` (Roberts-Rosenthal 1998)."""
        t0 = time.time()
        if warmup is None:
            warmup = n_samples // 2
        eps = step_size
        inits = self._init_chains(init, n_chains)
        chains: list[list[Vec]] = [[] for _ in range(n_chains)]
        log_probs: list[list[float]] = [[] for _ in range(n_chains)]
        accept_counts = [0] * n_chains
        # dual averaging on log-eps
        da = _DualAveraging(target_accept=target_accept,
                            init_log_step=math.log(eps))

        for c in range(n_chains):
            theta = list(inits[c])
            lp = self.log_density(theta)
            grad = self.grad_log_density(theta)
            for it in range(warmup + n_samples):
                tau = eps if it >= warmup else math.exp(da.log_step_smoothed)
                drift = _vec_scale(grad, tau / 2.0)
                noise = [self.rng.gauss(0.0, math.sqrt(tau)) for _ in range(self.dim)]
                prop = _vec_add(_vec_add(theta, drift), noise)
                try:
                    lp_prop = self.log_density(prop)
                    grad_prop = self.grad_log_density(prop)
                except (ValueError, OverflowError):
                    lp_prop = float("-inf"); grad_prop = grad
                # log q(theta | prop) - log q(prop | theta)
                drift_back = _vec_scale(grad_prop, tau / 2.0)
                d_fwd = _vec_sub(prop, _vec_add(theta, drift))
                d_bwd = _vec_sub(theta, _vec_add(prop, drift_back))
                log_q_fb = (-_dot(d_bwd, d_bwd) / (2 * tau)
                            + _dot(d_fwd, d_fwd) / (2 * tau))
                log_alpha = lp_prop - lp + log_q_fb
                alpha = min(1.0, math.exp(min(0.0, log_alpha)))
                if math.log(self.rng.random() + 1e-300) < log_alpha:
                    theta = prop; lp = lp_prop; grad = grad_prop
                    if it >= warmup:
                        accept_counts[c] += 1
                if it < warmup and adapt:
                    da.update(alpha)
                if it >= warmup:
                    chains[c].append(list(theta))
                    log_probs[c].append(lp)
        return SampleReport(
            chains=chains, log_probs=log_probs,
            accept_rates=[accept_counts[c] / max(1, n_samples)
                          for c in range(n_chains)],
            kernel="mala", diagnostics=_diagnose(chains),
            walltime_s=time.time() - t0, n_warmup=warmup,
            step_size=eps,
            seed=self.rng.randint(0, 1 << 30),
            n_samples=n_samples * n_chains, dim=self.dim,
        )

    # ===============================================================
    # ULA (no Metropolis correction)
    # ===============================================================

    def ula(
        self,
        n_samples: int,
        n_chains: int = 4,
        warmup: int | None = None,
        step_size: float = 0.01,
        init: Sequence[Sequence[float]] | None = None,
    ) -> SampleReport:
        """Unadjusted Langevin (biased but fast)."""
        t0 = time.time()
        if warmup is None:
            warmup = n_samples // 2
        tau = step_size
        inits = self._init_chains(init, n_chains)
        chains: list[list[Vec]] = [[] for _ in range(n_chains)]
        log_probs: list[list[float]] = [[] for _ in range(n_chains)]
        for c in range(n_chains):
            theta = list(inits[c])
            for it in range(warmup + n_samples):
                g = self.grad_log_density(theta)
                theta = _vec_add(
                    _vec_add(theta, _vec_scale(g, tau / 2.0)),
                    [self.rng.gauss(0.0, math.sqrt(tau)) for _ in range(self.dim)],
                )
                if it >= warmup:
                    chains[c].append(list(theta))
                    log_probs[c].append(self.log_density(theta))
        return SampleReport(
            chains=chains, log_probs=log_probs,
            accept_rates=[1.0] * n_chains,
            kernel="ula", diagnostics=_diagnose(chains),
            walltime_s=time.time() - t0, n_warmup=warmup,
            step_size=tau,
            seed=self.rng.randint(0, 1 << 30),
            n_samples=n_samples * n_chains, dim=self.dim,
        )

    # ===============================================================
    # HMC (Hamiltonian Monte Carlo)
    # ===============================================================

    def hmc(
        self,
        n_samples: int,
        n_chains: int = 4,
        warmup: int | None = None,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        mass: Sequence[float] | None = None,
        init: Sequence[Sequence[float]] | None = None,
        adapt: bool = True,
        target_accept: float = 0.8,
        max_energy_change: float = 1000.0,
    ) -> SampleReport:
        """Hamiltonian Monte Carlo with leapfrog integrator (Neal 2011).
        Mass matrix is diagonal; pass per-dimension masses."""
        t0 = time.time()
        if warmup is None:
            warmup = n_samples // 2
        M = list(mass) if mass is not None else [1.0] * self.dim
        inv_M = [1.0 / m for m in M]
        eps = step_size
        L = n_leapfrog
        inits = self._init_chains(init, n_chains)
        chains: list[list[Vec]] = [[] for _ in range(n_chains)]
        log_probs: list[list[float]] = [[] for _ in range(n_chains)]
        accept_counts = [0] * n_chains
        divergences = 0
        da = _DualAveraging(target_accept=target_accept,
                            init_log_step=math.log(eps))

        for c in range(n_chains):
            theta = list(inits[c])
            lp = self.log_density(theta)
            grad = self.grad_log_density(theta)
            for it in range(warmup + n_samples):
                step = math.exp(da.log_step_smoothed) if it < warmup else eps
                r = [self.rng.gauss(0.0, math.sqrt(m)) for m in M]
                H0 = -lp + 0.5 * sum(inv_M[i] * r[i] * r[i] for i in range(self.dim))
                theta_new = list(theta); grad_new = list(grad); r_new = list(r)
                # leapfrog
                # half-step momentum
                r_new = [r_new[i] + 0.5 * step * grad_new[i] for i in range(self.dim)]
                divergent = False
                for k in range(L):
                    theta_new = [theta_new[i] + step * inv_M[i] * r_new[i]
                                 for i in range(self.dim)]
                    try:
                        grad_new = self.grad_log_density(theta_new)
                    except (ValueError, OverflowError):
                        divergent = True
                        break
                    if k < L - 1:
                        r_new = [r_new[i] + step * grad_new[i] for i in range(self.dim)]
                if not divergent:
                    r_new = [r_new[i] + 0.5 * step * grad_new[i] for i in range(self.dim)]
                    try:
                        lp_new = self.log_density(theta_new)
                    except (ValueError, OverflowError):
                        divergent = True
                if divergent:
                    log_alpha = float("-inf"); alpha = 0.0
                    divergences += 1 if it >= warmup else 0
                else:
                    H1 = -lp_new + 0.5 * sum(inv_M[i] * r_new[i] * r_new[i]
                                             for i in range(self.dim))
                    if abs(H1 - H0) > max_energy_change:
                        divergent = True
                        divergences += 1 if it >= warmup else 0
                    log_alpha = H0 - H1
                    alpha = min(1.0, math.exp(min(0.0, log_alpha)))
                if not divergent and math.log(self.rng.random() + 1e-300) < log_alpha:
                    theta = theta_new; lp = lp_new; grad = grad_new
                    if it >= warmup:
                        accept_counts[c] += 1
                if it < warmup and adapt:
                    da.update(alpha)
                if it >= warmup:
                    chains[c].append(list(theta))
                    log_probs[c].append(lp)

        diag = _diagnose(chains)
        diag.divergences = divergences
        return SampleReport(
            chains=chains, log_probs=log_probs,
            accept_rates=[accept_counts[c] / max(1, n_samples)
                          for c in range(n_chains)],
            kernel="hmc", diagnostics=diag,
            walltime_s=time.time() - t0, n_warmup=warmup,
            step_size=eps, n_leapfrog=L, mass=M,
            seed=self.rng.randint(0, 1 << 30),
            n_samples=n_samples * n_chains, dim=self.dim,
        )

    # ===============================================================
    # NUTS (No-U-Turn Sampler, Hoffman-Gelman 2014)
    # ===============================================================

    def nuts(
        self,
        n_samples: int,
        n_chains: int = 4,
        warmup: int | None = None,
        max_tree_depth: int = 10,
        step_size: float = 1.0,
        target_accept: float = 0.8,
        init: Sequence[Sequence[float]] | None = None,
        mass: Sequence[float] | None = None,
        adapt: bool = True,
        max_energy_change: float = 1000.0,
    ) -> SampleReport:
        """No-U-Turn Sampler with dual-averaging step-size adaptation.

        Algorithm 6 of Hoffman-Gelman (2014).  Step-size is initialised
        by Algorithm 4 (find ε that halves the acceptance probability)."""
        t0 = time.time()
        if warmup is None:
            warmup = n_samples
        M = list(mass) if mass is not None else [1.0] * self.dim
        inv_M = [1.0 / m for m in M]
        inits = self._init_chains(init, n_chains)
        chains: list[list[Vec]] = [[] for _ in range(n_chains)]
        log_probs: list[list[float]] = [[] for _ in range(n_chains)]
        accept_means = [0.0] * n_chains
        divergences = 0
        depth_hits = 0

        def L(theta: Vec) -> float:
            try:
                return self.log_density(theta)
            except (ValueError, OverflowError):
                return float("-inf")

        def grad(theta: Vec) -> Vec:
            return self.grad_log_density(theta)

        def leapfrog(theta: Vec, r: Vec, grd: Vec, eps: float
                     ) -> tuple[Vec, Vec, Vec, float]:
            r2 = [r[i] + 0.5 * eps * grd[i] for i in range(self.dim)]
            theta2 = [theta[i] + eps * inv_M[i] * r2[i] for i in range(self.dim)]
            grd2 = grad(theta2)
            r3 = [r2[i] + 0.5 * eps * grd2[i] for i in range(self.dim)]
            return theta2, r3, grd2, L(theta2)

        def find_reasonable_eps(theta0: Vec, grd0: Vec, lp0: float) -> float:
            # Hoffman-Gelman Algorithm 4
            eps = 1.0
            r = [self.rng.gauss(0.0, math.sqrt(m)) for m in M]
            theta_p, r_p, _, lp_p = leapfrog(theta0, r, grd0, eps)
            H0 = -lp0 + 0.5 * sum(inv_M[i] * r[i] * r[i] for i in range(self.dim))
            H1 = -lp_p + 0.5 * sum(inv_M[i] * r_p[i] * r_p[i] for i in range(self.dim))
            log_alpha = H0 - H1
            a = 1 if log_alpha > math.log(0.5) else -1
            it = 0
            while a * log_alpha > -a * math.log(2.0) and it < 50:
                eps = eps * (2.0 ** a)
                theta_p, r_p, _, lp_p = leapfrog(theta0, r, grd0, eps)
                H1 = -lp_p + 0.5 * sum(inv_M[i] * r_p[i] * r_p[i]
                                       for i in range(self.dim))
                log_alpha = H0 - H1
                it += 1
            return eps

        def build_tree(theta: Vec, r: Vec, grd: Vec, log_u: float,
                       v: int, j: int, eps: float, H0: float):
            """Hoffman-Gelman Algorithm 3.  Returns
            ``(theta_-, r_-, grd_-, theta_+, r_+, grd_+, theta', n', s',
              alpha, n_alpha)``."""
            if j == 0:
                theta_p, r_p, grd_p, lp_p = leapfrog(theta, r, grd, v * eps)
                H1 = -lp_p + 0.5 * sum(inv_M[i] * r_p[i] * r_p[i]
                                       for i in range(self.dim))
                n_p = 1 if log_u <= -H1 else 0
                s_p = 1 if log_u < max_energy_change + (-H1) else 0
                alpha = min(1.0, math.exp(min(0.0, H0 - H1))) if math.isfinite(H1) else 0.0
                return (theta_p, r_p, grd_p, theta_p, r_p, grd_p,
                        theta_p, n_p, s_p, alpha, 1)
            (th_m, r_m, g_m, th_p, r_p, g_p, th_pr, n_pr, s_pr, alpha, n_alpha
             ) = build_tree(theta, r, grd, log_u, v, j - 1, eps, H0)
            if s_pr == 1:
                if v == -1:
                    (th_m, r_m, g_m, _, _, _, th_pr2, n_pr2, s_pr2, alpha2, n_alpha2
                     ) = build_tree(th_m, r_m, g_m, log_u, v, j - 1, eps, H0)
                else:
                    (_, _, _, th_p, r_p, g_p, th_pr2, n_pr2, s_pr2, alpha2, n_alpha2
                     ) = build_tree(th_p, r_p, g_p, log_u, v, j - 1, eps, H0)
                if n_pr + n_pr2 > 0 and self.rng.random() < n_pr2 / (n_pr + n_pr2):
                    th_pr = th_pr2
                alpha += alpha2; n_alpha += n_alpha2
                # U-turn check
                dtheta = _vec_sub(th_p, th_m)
                if _dot(dtheta, [inv_M[i] * r_m[i] for i in range(self.dim)]) < 0:
                    s_pr2 = 0
                if _dot(dtheta, [inv_M[i] * r_p[i] for i in range(self.dim)]) < 0:
                    s_pr2 = 0
                s_pr = s_pr * s_pr2
                n_pr += n_pr2
            return (th_m, r_m, g_m, th_p, r_p, g_p, th_pr, n_pr, s_pr,
                    alpha, n_alpha)

        for c in range(n_chains):
            theta = list(inits[c])
            lp = L(theta); grd = grad(theta)
            eps0 = find_reasonable_eps(theta, grd, lp)
            da = _DualAveraging(target_accept=target_accept,
                                init_log_step=math.log(eps0))
            for it in range(warmup + n_samples):
                eps = math.exp(da.log_step_smoothed) if it < warmup else \
                      math.exp(da.log_step_smoothed)
                if it == warmup and adapt:
                    # freeze step size at smoothed value
                    eps_frozen = math.exp(da.log_step_smoothed)
                r = [self.rng.gauss(0.0, math.sqrt(m)) for m in M]
                H0 = -lp + 0.5 * sum(inv_M[i] * r[i] * r[i] for i in range(self.dim))
                # slice variable
                log_u = -H0 + math.log(self.rng.random() + 1e-300)
                th_m = list(theta); r_m = list(r); g_m = list(grd)
                th_p = list(theta); r_p = list(r); g_p = list(grd)
                theta_new = list(theta); lp_new = lp; grd_new = list(grd)
                n = 1; s = 1; j = 0
                alpha = 0.0; n_alpha = 0
                while s == 1 and j < max_tree_depth:
                    v = -1 if self.rng.random() < 0.5 else 1
                    if v == -1:
                        (th_m, r_m, g_m, _, _, _, th_pr, n_pr, s_pr, alpha_j, n_alpha_j
                         ) = build_tree(th_m, r_m, g_m, log_u, v, j, eps, H0)
                    else:
                        (_, _, _, th_p, r_p, g_p, th_pr, n_pr, s_pr, alpha_j, n_alpha_j
                         ) = build_tree(th_p, r_p, g_p, log_u, v, j, eps, H0)
                    if s_pr == 1 and n_pr > 0 and self.rng.random() < min(1.0, n_pr / n):
                        theta_new = list(th_pr)
                        lp_new = L(th_pr); grd_new = grad(th_pr)
                    n += n_pr
                    s = s_pr
                    dtheta = _vec_sub(th_p, th_m)
                    if _dot(dtheta, [inv_M[i] * r_m[i] for i in range(self.dim)]) < 0:
                        s = 0
                    if _dot(dtheta, [inv_M[i] * r_p[i] for i in range(self.dim)]) < 0:
                        s = 0
                    alpha = alpha_j; n_alpha = max(1, n_alpha_j)
                    j += 1
                if j >= max_tree_depth and it >= warmup:
                    depth_hits += 1
                avg_alpha = alpha / max(1, n_alpha)
                theta = theta_new; lp = lp_new; grd = grd_new
                if it < warmup and adapt:
                    da.update(avg_alpha)
                if not math.isfinite(lp) and it >= warmup:
                    divergences += 1
                if it >= warmup:
                    chains[c].append(list(theta))
                    log_probs[c].append(lp)
                    accept_means[c] += avg_alpha
        accept_means = [a / max(1, n_samples) for a in accept_means]
        diag = _diagnose(chains)
        diag.divergences = divergences
        diag.max_tree_depth_hits = depth_hits
        return SampleReport(
            chains=chains, log_probs=log_probs,
            accept_rates=accept_means,
            kernel="nuts", diagnostics=diag,
            walltime_s=time.time() - t0, n_warmup=warmup,
            step_size=math.exp(da.log_step_smoothed),
            n_leapfrog=None, mass=M,
            seed=self.rng.randint(0, 1 << 30),
            n_samples=n_samples * n_chains, dim=self.dim,
        )

    # ===============================================================
    # Slice sampling (Neal 2003)
    # ===============================================================

    def slice_sample(
        self,
        n_samples: int,
        n_chains: int = 4,
        warmup: int | None = None,
        init: Sequence[Sequence[float]] | None = None,
        width: float = 1.0,
        max_steps_out: int = 100,
    ) -> SampleReport:
        """Slice sampling with axis-aligned stepping-out + shrinkage."""
        t0 = time.time()
        if warmup is None:
            warmup = n_samples // 2
        inits = self._init_chains(init, n_chains)
        chains: list[list[Vec]] = [[] for _ in range(n_chains)]
        log_probs: list[list[float]] = [[] for _ in range(n_chains)]

        def slice_along(theta: Vec, lp: float, dim: int) -> tuple[Vec, float]:
            log_y = lp + math.log(self.rng.random() + 1e-300)
            w = width
            u = self.rng.random()
            lo = theta[dim] - u * w
            hi = theta[dim] + (1 - u) * w
            k = max_steps_out
            test_lo = list(theta); test_lo[dim] = lo
            test_hi = list(theta); test_hi[dim] = hi
            while k > 0 and self.log_density(test_lo) > log_y:
                lo -= w; test_lo[dim] = lo; k -= 1
            k = max_steps_out
            while k > 0 and self.log_density(test_hi) > log_y:
                hi += w; test_hi[dim] = hi; k -= 1
            for _ in range(200):
                cand = lo + self.rng.random() * (hi - lo)
                test = list(theta); test[dim] = cand
                lp_cand = self.log_density(test)
                if lp_cand > log_y:
                    return test, lp_cand
                if cand < theta[dim]:
                    lo = cand
                else:
                    hi = cand
            return theta, lp

        for c in range(n_chains):
            theta = list(inits[c])
            lp = self.log_density(theta)
            for it in range(warmup + n_samples):
                for d in range(self.dim):
                    theta, lp = slice_along(theta, lp, d)
                if it >= warmup:
                    chains[c].append(list(theta))
                    log_probs[c].append(lp)
        return SampleReport(
            chains=chains, log_probs=log_probs,
            accept_rates=[1.0] * n_chains,
            kernel="slice", diagnostics=_diagnose(chains),
            walltime_s=time.time() - t0, n_warmup=warmup,
            seed=self.rng.randint(0, 1 << 30),
            n_samples=n_samples * n_chains, dim=self.dim,
        )

    # ===============================================================
    # Parallel tempering / replica exchange
    # ===============================================================

    def parallel_tempering(
        self,
        n_samples: int,
        warmup: int | None = None,
        n_replicas: int = 4,
        temperatures: Sequence[float] | None = None,
        base_kernel: str = "rwmh",
        swap_every: int = 5,
        **kwargs: Any,
    ) -> SampleReport:
        """Replica exchange with a geometric temperature ladder.

        Returns the cold-chain (β = 1) samples.  ``base_kernel`` ∈
        ``{rwmh, mala, hmc}``."""
        t0 = time.time()
        if warmup is None:
            warmup = n_samples // 2
        if temperatures is None:
            # geometric ladder β_k = β_max^{k/(K-1)}, β_max = 0.01
            temperatures = [
                math.exp(-k / max(1, n_replicas - 1) * math.log(100.0))
                for k in range(n_replicas)
            ]
        # initialise replicas
        replicas = [
            [self.rng.gauss(0.0, 1.0) for _ in range(self.dim)]
            for _ in range(n_replicas)
        ]
        lps = [self.log_density(r) for r in replicas]
        chains_cold: list[Vec] = []
        lps_cold: list[float] = []
        accept_swap = 0
        attempt_swap = 0
        accept_chain = [0] * n_replicas
        attempt_chain = [0] * n_replicas
        step_sizes = [kwargs.get("step_size", 0.5)] * n_replicas

        def kernel_step(rep_idx: int) -> None:
            beta = temperatures[rep_idx]
            theta = replicas[rep_idx]
            lp = lps[rep_idx]
            ss = step_sizes[rep_idx] / math.sqrt(max(1e-6, beta))
            if base_kernel == "rwmh":
                prop = [theta[i] + self.rng.gauss(0.0, ss) for i in range(self.dim)]
                try:
                    lp_prop = self.log_density(prop)
                except (ValueError, OverflowError):
                    lp_prop = float("-inf")
                log_alpha = beta * (lp_prop - lp)
                if math.log(self.rng.random() + 1e-300) < log_alpha:
                    replicas[rep_idx] = prop
                    lps[rep_idx] = lp_prop
                    accept_chain[rep_idx] += 1
            elif base_kernel == "mala":
                g = self.grad_log_density(theta)
                drift = _vec_scale(g, beta * ss / 2.0)
                prop = _vec_add(
                    _vec_add(theta, drift),
                    [self.rng.gauss(0.0, math.sqrt(ss)) for _ in range(self.dim)],
                )
                try:
                    lp_prop = self.log_density(prop)
                except (ValueError, OverflowError):
                    lp_prop = float("-inf")
                log_alpha = beta * (lp_prop - lp)
                if math.log(self.rng.random() + 1e-300) < log_alpha:
                    replicas[rep_idx] = prop
                    lps[rep_idx] = lp_prop
                    accept_chain[rep_idx] += 1
            elif base_kernel == "hmc":
                # short HMC trajectory for tempered target
                M = [1.0] * self.dim
                r = [self.rng.gauss(0.0, 1.0) for _ in range(self.dim)]
                theta_n = list(theta)
                lp_n = lp
                grad_n = self.grad_log_density(theta_n)
                L_steps = kwargs.get("n_leapfrog", 5)
                r_n = [r[i] + 0.5 * ss * beta * grad_n[i] for i in range(self.dim)]
                for k in range(L_steps):
                    theta_n = [theta_n[i] + ss * r_n[i] for i in range(self.dim)]
                    grad_n = self.grad_log_density(theta_n)
                    if k < L_steps - 1:
                        r_n = [r_n[i] + ss * beta * grad_n[i] for i in range(self.dim)]
                r_n = [r_n[i] + 0.5 * ss * beta * grad_n[i] for i in range(self.dim)]
                try:
                    lp_new = self.log_density(theta_n)
                except (ValueError, OverflowError):
                    lp_new = float("-inf")
                H0 = -beta * lp + 0.5 * sum(rr * rr for rr in r)
                H1 = -beta * lp_new + 0.5 * sum(rr * rr for rr in r_n)
                if math.log(self.rng.random() + 1e-300) < (H0 - H1):
                    replicas[rep_idx] = theta_n
                    lps[rep_idx] = lp_new
                    accept_chain[rep_idx] += 1
            attempt_chain[rep_idx] += 1

        for it in range(warmup + n_samples):
            for k in range(n_replicas):
                kernel_step(k)
            if it % swap_every == 0:
                for k in range(n_replicas - 1):
                    db = temperatures[k] - temperatures[k + 1]
                    dlp = lps[k] - lps[k + 1]
                    log_alpha = db * dlp * -1  # actually (β_k - β_{k+1})(lp_{k+1} - lp_k) flipped
                    # Correct PT swap: α = exp( (β_k - β_{k+1}) (lp(θ_{k+1}) - lp(θ_k)) )
                    log_alpha = (temperatures[k] - temperatures[k + 1]) * (lps[k + 1] - lps[k])
                    if math.log(self.rng.random() + 1e-300) < log_alpha:
                        replicas[k], replicas[k + 1] = replicas[k + 1], replicas[k]
                        lps[k], lps[k + 1] = lps[k + 1], lps[k]
                        if it >= warmup and k == 0:
                            accept_swap += 1
                    if it >= warmup and k == 0:
                        attempt_swap += 1
            if it >= warmup:
                chains_cold.append(list(replicas[0]))
                lps_cold.append(lps[0])
        chains = [chains_cold]
        diag = _diagnose(chains)
        diag.swap_accept_rate = accept_swap / max(1, attempt_swap)
        return SampleReport(
            chains=chains, log_probs=[lps_cold],
            accept_rates=[accept_chain[0] / max(1, attempt_chain[0])],
            kernel=f"pt[{base_kernel}]", diagnostics=diag,
            walltime_s=time.time() - t0, n_warmup=warmup,
            step_size=step_sizes[0],
            seed=self.rng.randint(0, 1 << 30),
            n_samples=n_samples, dim=self.dim,
        )

    # ===============================================================
    # SMC (Sequential Monte Carlo tempering)
    # ===============================================================

    def smc(
        self,
        n_particles: int,
        prior_sampler: Callable[[random.Random], Vec],
        log_likelihood: Callable[[Sequence[float]], float],
        ess_target: float = 0.5,
        max_temps: int = 50,
        n_mcmc: int = 5,
        mcmc_step: float = 0.3,
    ) -> SMCReport:
        """SMC sampler with adaptive geometric tempering (Del Moral 2006).

        Draws ``n_particles`` from prior, then adaptively chooses
        ``β_{n+1}`` such that ``ESS(β_{n+1}) = ess_target · ESS(β_n)``,
        reweights, resamples (multinomial when ESS drops below target),
        and moves with a Metropolis kernel targeted at ``π_0 · L^{β_{n+1}}``.

        Returns posterior samples *and an unbiased estimate of the log
        marginal likelihood* (Del Moral-Doucet-Jasra 2006 thm. 1)."""
        t0 = time.time()
        particles = [prior_sampler(self.rng) for _ in range(n_particles)]
        log_lik = [log_likelihood(p) for p in particles]
        log_w = [0.0] * n_particles
        beta = 0.0
        ess_history: list[float] = [float(n_particles)]
        betas = [0.0]
        log_Z = 0.0
        n_resample = 0
        target = ess_target * n_particles

        def ess_at(beta_new: float) -> float:
            lw = [log_w[i] + (beta_new - beta) * log_lik[i] for i in range(n_particles)]
            w = _normalise_log_weights(lw)
            return 1.0 / sum(wi * wi for wi in w)

        while beta < 1.0 and len(betas) < max_temps:
            # bisection for next β
            lo, hi = beta, 1.0
            if ess_at(1.0) > target:
                beta_new = 1.0
            else:
                for _ in range(50):
                    mid = (lo + hi) / 2.0
                    if ess_at(mid) > target:
                        lo = mid
                    else:
                        hi = mid
                    if hi - lo < 1e-6:
                        break
                beta_new = lo
            beta_new = min(1.0, max(beta + 1e-6, beta_new))
            # update log_Z: log E_{π_n}[L^{(β_{n+1} - β_n)}]
            inc = [(beta_new - beta) * log_lik[i] for i in range(n_particles)]
            m = max(_vec_add(log_w, inc))
            log_Z += math.log(
                sum(math.exp(log_w[i] + inc[i] - m) for i in range(n_particles))
            ) + m - math.log(
                sum(math.exp(lw - max(log_w)) for lw in log_w)
            ) - max(log_w)
            log_w = _vec_add(log_w, inc)
            beta = beta_new
            betas.append(beta)
            # resample if ESS drops
            ess_now = ess_at(beta)
            ess_history.append(ess_now)
            if ess_now < target:
                w = _normalise_log_weights(log_w)
                cum = [0.0] * n_particles
                s = 0.0
                for i, wi in enumerate(w):
                    s += wi; cum[i] = s
                new_particles = []
                new_loglik = []
                for _ in range(n_particles):
                    u = self.rng.random()
                    lo_idx, hi_idx = 0, n_particles - 1
                    while lo_idx < hi_idx:
                        mid = (lo_idx + hi_idx) // 2
                        if cum[mid] < u:
                            lo_idx = mid + 1
                        else:
                            hi_idx = mid
                    new_particles.append(list(particles[lo_idx]))
                    new_loglik.append(log_lik[lo_idx])
                particles = new_particles
                log_lik = new_loglik
                log_w = [0.0] * n_particles
                n_resample += 1
            # MCMC move at current tempered target
            for i in range(n_particles):
                for _ in range(n_mcmc):
                    prop = [particles[i][j] + self.rng.gauss(0.0, mcmc_step)
                            for j in range(self.dim)]
                    try:
                        lp_prop = self.log_density(prop)
                        ll_prop = log_likelihood(prop)
                    except (ValueError, OverflowError):
                        continue
                    log_alpha = beta * ll_prop - beta * log_lik[i]
                    if math.log(self.rng.random() + 1e-300) < log_alpha:
                        particles[i] = prop
                        log_lik[i] = ll_prop
        diag = Diagnostics(
            rhat=[float("nan")] * self.dim,
            ess_bulk=[ess_history[-1]] * self.dim,
            ess_tail=[ess_history[-1]] * self.dim,
            geweke_z=[float("nan")] * self.dim,
            autocorr_time=[1.0] * self.dim,
            ess_normalising=ess_history[-1],
        )
        return SMCReport(
            samples=particles, log_weights=log_w,
            log_evidence=log_Z, ess_history=ess_history,
            temperatures=betas, walltime_s=time.time() - t0,
            n_resample=n_resample, diagnostics=diag,
        )

    # ===============================================================
    # Importance sampling
    # ===============================================================

    def importance(
        self,
        n: int,
        proposal_sampler: Callable[[random.Random], Vec],
        log_proposal: Callable[[Sequence[float]], float],
    ) -> ImportanceReport:
        """Self-normalised IS estimate, with Pareto-k tail diagnostic."""
        samples: list[Vec] = []
        log_w: list[float] = []
        for _ in range(n):
            x = proposal_sampler(self.rng)
            try:
                lpx = self.log_density(x); lqx = log_proposal(x)
            except (ValueError, OverflowError):
                continue
            samples.append(list(x))
            log_w.append(lpx - lqx)
        if not log_w:
            return ImportanceReport(
                log_weights=[], ess=0.0, pareto_k=float("inf"),
                log_norm_const_estimate=float("nan"), samples=[],
            )
        m = max(log_w)
        log_Z = math.log(sum(math.exp(lw - m) for lw in log_w) / len(log_w)) + m
        return ImportanceReport(
            log_weights=log_w,
            ess=_ess_is(log_w),
            pareto_k=pareto_k(log_w),
            log_norm_const_estimate=log_Z,
            samples=samples,
        )

    # ===============================================================
    # ADVI (Automatic Differentiation Variational Inference)
    # ===============================================================

    def advi(
        self,
        n_iter: int = 1000,
        n_mc: int = 8,
        learning_rate: float = 0.05,
        full_rank: bool = False,
        init_mu: Sequence[float] | None = None,
        init_log_sigma: float = -1.0,
        adagrad_eps: float = 1e-8,
    ) -> ADVIReport:
        """Mean-field or full-rank Gaussian VI on ``R^d`` (Kucukelbir 2017).
        AdaGrad on the natural log-sigma parameterisation."""
        t0 = time.time()
        d = self.dim
        mu = list(init_mu) if init_mu is not None else [0.0] * d
        log_sigma = [init_log_sigma] * d
        L = None
        if full_rank:
            L = _eye(d)
            for i in range(d):
                L[i][i] = math.exp(init_log_sigma)
        elbo_history: list[float] = []
        # AdaGrad accumulators
        acc_mu = [0.0] * d
        acc_log_sigma = [0.0] * d
        acc_L = [[0.0] * d for _ in range(d)] if full_rank else None

        for it in range(n_iter):
            grad_mu = [0.0] * d
            grad_log_sigma = [0.0] * d
            grad_L = [[0.0] * d for _ in range(d)] if full_rank else None
            elbo = 0.0
            for _ in range(n_mc):
                eps = [self.rng.gauss(0.0, 1.0) for _ in range(d)]
                if full_rank:
                    z = [sum(L[i][j] * eps[j] for j in range(d)) for i in range(d)]
                    theta = [mu[i] + z[i] for i in range(d)]
                else:
                    sig = [math.exp(ls) for ls in log_sigma]
                    theta = [mu[i] + sig[i] * eps[i] for i in range(d)]
                try:
                    lp = self.log_density(theta)
                    g = self.grad_log_density(theta)
                except (ValueError, OverflowError):
                    continue
                if not math.isfinite(lp):
                    continue
                # entropy contribution: 0.5 d log(2π e) + Σ log σ  or  log|L|
                if full_rank:
                    ent = 0.5 * d * (math.log(2 * math.pi) + 1)
                    for i in range(d):
                        ent += math.log(max(1e-12, abs(L[i][i])))
                else:
                    ent = 0.5 * d * (math.log(2 * math.pi) + 1) + sum(log_sigma)
                elbo += (lp + ent) / n_mc
                # gradients via reparameterisation
                for i in range(d):
                    grad_mu[i] += g[i] / n_mc
                if full_rank:
                    for i in range(d):
                        for j in range(i + 1):
                            grad_L[i][j] += g[i] * eps[j] / n_mc
                        # entropy term contributes 1/L[i][i] on the diagonal
                        grad_L[i][i] += 1.0 / max(1e-12, abs(L[i][i])) / n_mc
                else:
                    for i in range(d):
                        sig = math.exp(log_sigma[i])
                        # dELBO / dlog_sigma  =  g_i * sig * eps_i + 1
                        grad_log_sigma[i] += (g[i] * sig * eps[i] + 1.0) / n_mc
            elbo_history.append(elbo)
            for i in range(d):
                acc_mu[i] += grad_mu[i] ** 2
                mu[i] += learning_rate * grad_mu[i] / (math.sqrt(acc_mu[i]) + adagrad_eps)
            if full_rank:
                for i in range(d):
                    for j in range(i + 1):
                        acc_L[i][j] += grad_L[i][j] ** 2
                        L[i][j] += learning_rate * grad_L[i][j] / (math.sqrt(acc_L[i][j]) + adagrad_eps)
                    # keep diagonal positive
                    if L[i][i] <= 0:
                        L[i][i] = 1e-6
            else:
                for i in range(d):
                    acc_log_sigma[i] += grad_log_sigma[i] ** 2
                    log_sigma[i] += learning_rate * grad_log_sigma[i] / (
                        math.sqrt(acc_log_sigma[i]) + adagrad_eps)
        return ADVIReport(
            mu=mu, sigma=[math.exp(ls) for ls in log_sigma],
            L_lowrank=L, elbo_history=elbo_history,
            walltime_s=time.time() - t0, n_iter=n_iter,
        )

    # ===============================================================
    # initialisation
    # ===============================================================

    def _init_chains(
        self,
        init: Sequence[Sequence[float]] | None,
        n_chains: int,
    ) -> list[Vec]:
        if init is not None:
            inits = [list(x) for x in init]
            if len(inits) != n_chains:
                raise ValueError("init must have length n_chains")
            for x in inits:
                if len(x) != self.dim:
                    raise ValueError("init has wrong dimension")
            return inits
        return [
            [self.rng.gauss(0.0, 1.0) for _ in range(self.dim)]
            for _ in range(n_chains)
        ]


# =============================================================================
# Dual averaging on log-step (Hoffman-Gelman Algorithm 6)
# =============================================================================


class _DualAveraging:
    """Nesterov primal-dual averaging on ``log ε`` with target accept ``δ``."""

    def __init__(
        self,
        target_accept: float = 0.8,
        init_log_step: float = 0.0,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ) -> None:
        self.delta = target_accept
        self.mu = math.log(10.0) + init_log_step
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.log_step = init_log_step
        self.log_step_smoothed = init_log_step
        self.H_bar = 0.0
        self.m = 0

    def update(self, alpha: float) -> None:
        self.m += 1
        eta = 1.0 / (self.m + self.t0)
        self.H_bar = (1 - eta) * self.H_bar + eta * (self.delta - alpha)
        self.log_step = self.mu - math.sqrt(self.m) / self.gamma * self.H_bar
        eta_smooth = self.m ** (-self.kappa)
        self.log_step_smoothed = (eta_smooth * self.log_step
                                  + (1 - eta_smooth) * self.log_step_smoothed)


# =============================================================================
# Convenience builders for common targets
# =============================================================================


def gaussian_log_density(
    mu: Sequence[float], cov_diag: Sequence[float]
) -> Callable[[Sequence[float]], float]:
    """``log N(θ; μ, diag(σ²))`` minus constant."""
    inv = [1.0 / c for c in cov_diag]

    def f(theta: Sequence[float]) -> float:
        return -0.5 * sum(inv[i] * (theta[i] - mu[i]) ** 2 for i in range(len(mu)))
    return f


def gaussian_grad(
    mu: Sequence[float], cov_diag: Sequence[float]
) -> Callable[[Sequence[float]], list[float]]:
    inv = [1.0 / c for c in cov_diag]

    def g(theta: Sequence[float]) -> list[float]:
        return [-inv[i] * (theta[i] - mu[i]) for i in range(len(mu))]
    return g


def banana_log_density(b: float = 0.1) -> Callable[[Sequence[float]], float]:
    """Rosenbrock-style banana — the classical MCMC stress test."""
    def f(theta: Sequence[float]) -> float:
        x, y = theta[0], theta[1]
        return -0.5 * x * x - 0.5 * (y - b * (x * x - 100.0)) ** 2 / 1.0
    return f


def banana_grad(b: float = 0.1) -> Callable[[Sequence[float]], list[float]]:
    def g(theta: Sequence[float]) -> list[float]:
        x, y = theta[0], theta[1]
        r = y - b * (x * x - 100.0)
        return [-x + r * 2 * b * x, -r]
    return g


__all__ = [
    "Sampler",
    "SampleReport",
    "Diagnostics",
    "ImportanceReport",
    "SMCReport",
    "ADVIReport",
    "rhat",
    "ess_bulk",
    "ess_tail",
    "autocorr",
    "autocorr_time",
    "geweke_z",
    "pareto_k",
    "gaussian_log_density",
    "gaussian_grad",
    "banana_log_density",
    "banana_grad",
]
