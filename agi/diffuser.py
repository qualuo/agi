r"""Diffuser — score-based generative modelling as a runtime primitive.

Existing generative primitives in the runtime cover specific modes:

  * :class:`agi.flower.Flower` — Generative Flow Networks (categorical,
    multi-modal sampling proportional to a reward)
  * :class:`agi.imaginator.Imaginator` — learned world-model rollouts
    (discrete state-action trajectories)
  * :class:`agi.bayesopt.BayesOpt` — Gaussian-process posterior sampling
  * :class:`agi.predictor.Predictor` — Context-Tree-Weighting sequence
    prediction
  * :class:`agi.sampler.Sampler` — exact / approximate posterior sampling

None of these gives the runtime a *score-based continuous* generative
model — the family that has driven the last five years of frontier work
on images, video, audio, robotics, and protein structure.  **Diffuser**
fills that gap.  Given a noise schedule and a (learned or analytic)
score function ``s(x, t) ≈ ∇_x log p_t(x)``, the primitive draws samples
from the target distribution via *any* of ten classical reverse-time
solvers, with formal mixing / sampling-error certificates.

Mathematical roots
------------------

The primitive ships **ten** algorithms, every one from a named paper,
implemented from first principles in pure stdlib:

* **DDPM (Ho, Jain, Abbeel 2020 — "Denoising Diffusion Probabilistic
  Models", arXiv:2006.11239).**  Stochastic reverse-time process
  ``x_{t-1} = 1/√α_t · (x_t − (1−α_t)/√(1−ᾱ_t) · ε_θ(x_t,t)) + σ_t · z``.

* **DDIM (Song, Meng, Ermon 2020 — "Denoising Diffusion Implicit
  Models", arXiv:2010.02502).**  Deterministic non-Markovian sampler
  recovering the same marginals as DDPM in O(T) → O(K) steps.

* **DPM-Solver-1 / DPM-Solver-2 (Lu, Zhou, Bao, Chen, Li, Zhu 2022 —
  "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model
  Sampling in Around 10 Steps", arXiv:2206.00927).**  Exponential
  integrator for the probability-flow ODE; first / second order.

* **Heun (Karras, Aittala, Aila, Laine 2022 — "Elucidating the Design
  Space of Diffusion-Based Generative Models", arXiv:2206.00364).**
  Second-order predictor-corrector for the EDM PF-ODE.

* **Euler-Maruyama (Karras et al. 2022, stochastic variant).**
  First-order SDE solver: ``x_{t-h} = x_t + h·f(x_t,t) + √h·g(t)·z``.

* **Probability-flow ODE Euler (Song, Sohl-Dickstein, Kingma, Kumar,
  Ermon, Poole 2021 — "Score-Based Generative Modeling Through
  Stochastic Differential Equations", arXiv:2011.13456).**  Deterministic
  ODE form: ``dx/dt = f(x,t) − ½g(t)² ∇log p_t(x)``.

* **Predictor-Corrector (Song et al. 2021).**  Alternate a discretised
  reverse-SDE step (predictor) with Langevin MCMC moves (corrector).

* **Flow Matching (Lipman, Chen, Ben-Hamu, Nickel, Le 2023 —
  "Flow Matching for Generative Modeling", arXiv:2210.02747).**
  Continuous normalising flow via straight-line conditional path
  ``x_t = (1−t)x_0 + t·x_1``; integrate learnt vector field ``v_θ``.

* **Consistency model (Song, Dhariwal, Chen, Sutskever 2023 —
  "Consistency Models", arXiv:2303.01469).**  One-shot map
  ``f_θ(x_t,t) ≈ x_0`` distilled from a trained diffusion model;
  enables one-step or few-step sampling.

* **D3PM categorical diffusion (Austin, Johnson, Ho, Tarlow, van den
  Berg 2021 — "Structured Denoising Diffusion Models in Discrete
  State-Spaces", arXiv:2107.03006).**  Absorbing / uniform / Gaussian
  transition kernels over discrete state spaces, with closed-form
  posterior ``q(x_{t-1}|x_t, x_0)`` for reverse sampling.

Theoretical certificates
------------------------

For every sample drawn, ``certify()`` returns a record bundling four
quantitative guarantees, each derived from a named theorem:

* **Girsanov mixing bound (Chen, Chewi, Li, Li, Salim, Zhang 2022 —
  "Sampling is as easy as learning the score", arXiv:2209.11215).**
  If the learned score satisfies ``E[‖s_θ(x,t) − ∇log p_t(x)‖²] ≤ ε``
  and the data distribution has second moment ``M`` and decays
  exponentially with rate ``μ``, then TV(p_T-stationary, p_data) ≤
  e^{-μT/2} + √(T · ε / 2).

* **DDPM ELBO (Ho 2020 §3).**  ``-log p_θ(x_0) ≤ Σ_t E_q[D_KL(q(x_{t-1}
  |x_t,x_0) || p_θ(x_{t-1}|x_t))] + const``.  Each term is a Gaussian
  KL we evaluate in closed form for the diagnostic Gaussian-mixture
  target.

* **Score-matching loss (Hyvärinen 2005 — "Estimation of Non-Normalized
  Statistical Models by Score Matching").**  ``L(θ) = E[½‖s_θ(x)‖² +
  tr(∇s_θ(x))]``; minimisation is equivalent (up to a constant) to
  minimising Fisher divergence ``E‖s_θ(x) − ∇log p(x)‖²``.

* **Empirical TV concentration (Devroye, Györfi, Lugosi 1996).**  For
  finite samples from the model and the target, ``|TV̂ − TV| ≤
  √(log(2/δ)/(2n))`` (Hoeffding) — a Wasserstein-style witness via
  per-bin counts.

How the primitive composes
--------------------------

The runtime treats Diffuser as a *deferred sampler*.  Other primitives
ask it for trajectories or targets, with optional conditioning and
guidance:

* :class:`agi.imaginator.Imaginator` — synthetic rollouts via
  conditional sampling on observed state prefixes.
* :class:`agi.aligner.Aligner` — preference-conditioned generation via
  classifier-free guidance over a learned reward.
* :class:`agi.quantilizer.Quantilizer` — bound rare-tail sampling by
  rejecting low-score generations.
* :class:`agi.pareto.Pareto` — multi-objective conditional generation
  by linearly combining classifier gradients.
* :class:`agi.reconciler.Reconciler` — Aumann agreement on samples
  drawn from two independent score networks.
* :class:`agi.speculator.Speculator` — speculate K reverse-time steps
  on a cheap network, verify with the expensive one.

Pure-stdlib, deterministic, exportable
--------------------------------------

The implementation is pure ``math``/``random``/``hashlib``/``json``;
all randomness is seedable; every public step appends a hash-linked
ledger entry; every (config, state) tuple round-trips through
``export()`` / ``import_()``.

The reference *score function* the runtime ships for testing and demos
is the analytic score of an isotropic Gaussian mixture.  Real users
plug in their own learnt score via ``register(..., score_fn=...)``.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    # algorithms
    "ALG_DDPM",
    "ALG_DDIM",
    "ALG_DPM_SOLVER_1",
    "ALG_DPM_SOLVER_2",
    "ALG_HEUN",
    "ALG_EULER_SDE",
    "ALG_PF_ODE",
    "ALG_PREDICTOR_CORRECTOR",
    "ALG_FLOW_MATCHING",
    "ALG_CONSISTENCY",
    "ALG_D3PM",
    "KNOWN_ALGORITHMS",
    # noise schedules
    "SCHEDULE_LINEAR",
    "SCHEDULE_COSINE",
    "SCHEDULE_KARRAS",
    "KNOWN_SCHEDULES",
    # D3PM kernels
    "KERNEL_UNIFORM",
    "KERNEL_ABSORBING",
    "KNOWN_KERNELS",
    # event kinds
    "DIFFUSER_STARTED",
    "DIFFUSER_REGISTERED",
    "DIFFUSER_DEREGISTERED",
    "DIFFUSER_STEP",
    "DIFFUSER_SAMPLED",
    "DIFFUSER_FIT",
    "DIFFUSER_CERTIFIED",
    "DIFFUSER_CLEARED",
    # errors
    "DiffuserError",
    "InvalidConfig",
    "InvalidTarget",
    "UnknownTarget",
    "UnknownAlgorithm",
    "UnknownSchedule",
    "InsufficientData",
    "GuidanceViolation",
    # data classes
    "DiffuserConfig",
    "NoiseSchedule",
    "Sample",
    "StepOutput",
    "FitReport",
    "Certificate",
    "DiffuserReport",
    # primitive
    "Diffuser",
    # math helpers
    "normal_sample",
    "log_sumexp",
    "gaussian_log_density",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "karras_sigma_schedule",
    "alpha_bar_from_betas",
    "girsanov_tv_bound",
    "empirical_tv",
    "ddpm_elbo_term",
    "ledger_root",
    # built-in score factories (for demos and tests)
    "gaussian_mixture_score",
    "gaussian_mixture_log_density",
    "absorbing_d3pm_kernel",
    "uniform_d3pm_kernel",
]

# ----------------------------------------------------------------------
# Algorithm + schedule + kernel registries
# ----------------------------------------------------------------------

ALG_DDPM = "ddpm"
ALG_DDIM = "ddim"
ALG_DPM_SOLVER_1 = "dpm-solver-1"
ALG_DPM_SOLVER_2 = "dpm-solver-2"
ALG_HEUN = "heun"
ALG_EULER_SDE = "euler-sde"
ALG_PF_ODE = "pf-ode"
ALG_PREDICTOR_CORRECTOR = "predictor-corrector"
ALG_FLOW_MATCHING = "flow-matching"
ALG_CONSISTENCY = "consistency"
ALG_D3PM = "d3pm"

KNOWN_ALGORITHMS: tuple[str, ...] = (
    ALG_DDPM,
    ALG_DDIM,
    ALG_DPM_SOLVER_1,
    ALG_DPM_SOLVER_2,
    ALG_HEUN,
    ALG_EULER_SDE,
    ALG_PF_ODE,
    ALG_PREDICTOR_CORRECTOR,
    ALG_FLOW_MATCHING,
    ALG_CONSISTENCY,
    ALG_D3PM,
)

SCHEDULE_LINEAR = "linear"
SCHEDULE_COSINE = "cosine"
SCHEDULE_KARRAS = "karras"

KNOWN_SCHEDULES: tuple[str, ...] = (SCHEDULE_LINEAR, SCHEDULE_COSINE, SCHEDULE_KARRAS)

KERNEL_UNIFORM = "uniform"
KERNEL_ABSORBING = "absorbing"

KNOWN_KERNELS: tuple[str, ...] = (KERNEL_UNIFORM, KERNEL_ABSORBING)

# ----------------------------------------------------------------------
# Event kinds (typed strings — match runtime convention)
# ----------------------------------------------------------------------

DIFFUSER_STARTED = "diffuser.started"
DIFFUSER_REGISTERED = "diffuser.registered"
DIFFUSER_DEREGISTERED = "diffuser.deregistered"
DIFFUSER_STEP = "diffuser.step"
DIFFUSER_SAMPLED = "diffuser.sampled"
DIFFUSER_FIT = "diffuser.fit"
DIFFUSER_CERTIFIED = "diffuser.certified"
DIFFUSER_CLEARED = "diffuser.cleared"

# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class DiffuserError(Exception):
    """Base class for all Diffuser-raised errors."""


class InvalidConfig(DiffuserError):
    """Raised when the configuration is malformed."""


class InvalidTarget(DiffuserError):
    """Raised when a target registration is malformed."""


class UnknownTarget(DiffuserError):
    """Raised when an operation references an unregistered target."""


class UnknownAlgorithm(DiffuserError):
    """Raised when ``algorithm`` is not in :data:`KNOWN_ALGORITHMS`."""


class UnknownSchedule(DiffuserError):
    """Raised when a noise schedule name is unrecognised."""


class InsufficientData(DiffuserError):
    """Raised when an estimator is asked to operate without enough samples."""


class GuidanceViolation(DiffuserError):
    """Raised when classifier-free guidance is requested but no conditional
    score branch was registered."""


# ----------------------------------------------------------------------
# Math helpers — pure stdlib, deterministic, exhaustively tested
# ----------------------------------------------------------------------

_LOG_2PI = math.log(2.0 * math.pi)


def normal_sample(rng: random.Random, dim: int) -> list[float]:
    """Draw a fresh ``dim``-dimensional standard-normal vector."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def log_sumexp(xs: Sequence[float]) -> float:
    """Numerically stable ``log(sum(exp(x_i)))``."""
    if not xs:
        raise ValueError("log_sumexp of empty sequence")
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def gaussian_log_density(
    x: Sequence[float],
    mu: Sequence[float],
    sigma_sq: float,
) -> float:
    """Log-density of an isotropic Gaussian ``N(mu, sigma_sq · I)``."""
    if len(x) != len(mu):
        raise ValueError("x and mu must have matching dimension")
    if sigma_sq <= 0:
        raise ValueError("sigma_sq must be positive")
    d = len(x)
    sq = sum((xi - mui) ** 2 for xi, mui in zip(x, mu))
    return -0.5 * (sq / sigma_sq + d * (math.log(sigma_sq) + _LOG_2PI))


def linear_beta_schedule(
    T: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> list[float]:
    """Linear ``β_t`` interpolation as used in the original DDPM paper."""
    if T <= 0:
        raise ValueError("T must be positive")
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError("beta endpoints must be positive")
    if beta_start > beta_end:
        raise ValueError("beta_start must be <= beta_end")
    if T == 1:
        return [beta_end]
    step = (beta_end - beta_start) / (T - 1)
    return [beta_start + step * t for t in range(T)]


def cosine_beta_schedule(T: int, *, s: float = 8e-3) -> list[float]:
    """Cosine ``β_t`` schedule (Nichol & Dhariwal 2021, improved DDPM)."""
    if T <= 0:
        raise ValueError("T must be positive")
    if s <= 0:
        raise ValueError("offset s must be positive")
    # f(t) = cos((t/T + s)/(1+s) * π/2)^2
    def f(t: float) -> float:
        return math.cos(((t / T + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2

    alpha_bar = [f(t) / f(0.0) for t in range(T + 1)]
    betas: list[float] = []
    for t in range(1, T + 1):
        b = 1.0 - alpha_bar[t] / max(alpha_bar[t - 1], 1e-12)
        betas.append(min(max(b, 1e-8), 0.999))
    return betas


def karras_sigma_schedule(
    T: int,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> list[float]:
    """Karras et al. 2022 ``σ_i`` schedule (variance-exploding form)."""
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigmas must be positive")
    if sigma_min >= sigma_max:
        raise ValueError("sigma_min must be < sigma_max")
    if rho <= 0:
        raise ValueError("rho must be positive")
    # σ_i = (σ_max^(1/ρ) + i/(T-1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ
    rho_inv = 1.0 / rho
    a = sigma_max ** rho_inv
    b = sigma_min ** rho_inv
    if T == 1:
        return [sigma_max]
    return [(a + (i / (T - 1)) * (b - a)) ** rho for i in range(T)] + [0.0]


def alpha_bar_from_betas(betas: Sequence[float]) -> list[float]:
    """Cumulative ``ᾱ_t = Π_{s≤t} (1 − β_s)`` from a beta schedule."""
    out: list[float] = []
    p = 1.0
    for b in betas:
        p *= (1.0 - b)
        out.append(p)
    return out


def girsanov_tv_bound(
    *,
    score_error: float,
    horizon: float,
    second_moment: float,
    mu: float,
) -> float:
    """Girsanov-style TV bound (Chen et al. 2022).

    Given Fisher-divergence score error ``score_error`` averaged over the
    reverse-time SDE, a finite horizon ``T = horizon``, a data second
    moment ``M``, and an exponential decay rate ``μ`` of the forward
    process, this returns an upper bound on the total-variation distance
    between the model's stationary distribution and the data:

        ``TV ≤ e^{-μT/2} · √M + √(T · score_error / 2)``

    The first term is the *initial-distribution* gap (how close ``N(0,
    I)`` is to the actual prior at time ``T``); the second is the
    *score-mismatch* gap.
    """
    if score_error < 0:
        raise ValueError("score_error must be non-negative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if second_moment < 0:
        raise ValueError("second_moment must be non-negative")
    if mu < 0:
        raise ValueError("mu must be non-negative")
    init_gap = math.exp(-0.5 * mu * horizon) * math.sqrt(second_moment)
    score_gap = math.sqrt(0.5 * horizon * score_error)
    return init_gap + score_gap


def empirical_tv(
    samples: Sequence[Sequence[float]],
    target_samples: Sequence[Sequence[float]],
    *,
    bins: int = 16,
    delta: float = 0.05,
) -> tuple[float, float]:
    """Histogram-based empirical TV between two sample sets, plus
    a Hoeffding confidence half-width (so the *true* TV lies in
    ``[TV̂ − half, TV̂ + half]`` with probability at least ``1 − δ``).

    Both sample sets must share the same dimensionality.  Bins are
    placed jointly over the union of both sets' per-dim ranges so the
    estimator is invariant to which set defines the histogram.
    """
    if not samples or not target_samples:
        raise InsufficientData("need at least one sample in each set")
    d = len(samples[0])
    if any(len(s) != d for s in samples):
        raise ValueError("ragged samples")
    if any(len(s) != d for s in target_samples):
        raise ValueError("ragged target_samples")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    # Per-dim bin edges from union range.
    lows = [
        min(min(s[i] for s in samples), min(s[i] for s in target_samples))
        for i in range(d)
    ]
    highs = [
        max(max(s[i] for s in samples), max(s[i] for s in target_samples))
        for i in range(d)
    ]
    widths = [max(highs[i] - lows[i], 1e-12) / bins for i in range(d)]
    def bin_index(x: Sequence[float]) -> tuple[int, ...]:
        out: list[int] = []
        for i in range(d):
            j = int((x[i] - lows[i]) / widths[i])
            if j < 0:
                j = 0
            elif j >= bins:
                j = bins - 1
            out.append(j)
        return tuple(out)

    p_counts: dict[tuple[int, ...], int] = {}
    q_counts: dict[tuple[int, ...], int] = {}
    for s in samples:
        b = bin_index(s)
        p_counts[b] = p_counts.get(b, 0) + 1
    for s in target_samples:
        b = bin_index(s)
        q_counts[b] = q_counts.get(b, 0) + 1
    np_ = sum(p_counts.values())
    nq_ = sum(q_counts.values())
    all_keys = set(p_counts) | set(q_counts)
    tv = 0.5 * sum(
        abs(p_counts.get(k, 0) / np_ - q_counts.get(k, 0) / nq_) for k in all_keys
    )
    n_min = min(np_, nq_)
    half = math.sqrt(math.log(2.0 / delta) / (2.0 * n_min))
    return tv, half


def ddpm_elbo_term(
    *,
    x0: Sequence[float],
    xt: Sequence[float],
    alpha_bar_t: float,
    alpha_bar_tm1: float,
    beta_t: float,
    pred_x0: Sequence[float],
) -> float:
    """Single-term DDPM ELBO contribution (Ho 2020, §3, Eq. 8–9).

    Returns ``D_KL(q(x_{t-1} | x_t, x_0) || p_θ(x_{t-1} | x_t))`` under
    a Gaussian closed-form approximation.  This is the per-timestep
    summand in the variational bound on negative log-likelihood.
    """
    if not (0.0 < alpha_bar_t < 1.0):
        raise ValueError("alpha_bar_t must be in (0, 1)")
    if not (0.0 < alpha_bar_tm1 <= 1.0):
        raise ValueError("alpha_bar_tm1 must be in (0, 1]")
    if not (0.0 < beta_t < 1.0):
        raise ValueError("beta_t must be in (0, 1)")
    if len(x0) != len(xt) or len(x0) != len(pred_x0):
        raise ValueError("x0, xt and pred_x0 must share dimension")
    # Posterior mean of q(x_{t-1} | x_t, x_0).
    coef_x0 = math.sqrt(alpha_bar_tm1) * beta_t / (1.0 - alpha_bar_t)
    coef_xt = math.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t)
    # Two Gaussian means differ only by `coef_x0 * (x0 - pred_x0)`.
    diff = [coef_x0 * (a - b) for a, b in zip(x0, pred_x0)]
    sq = sum(v * v for v in diff)
    posterior_var = beta_t * (1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t)
    posterior_var = max(posterior_var, 1e-12)
    # KL between two isotropic Gaussians with same variance is sq/(2σ²).
    return 0.5 * sq / posterior_var


def _hash_entry(parent: str, payload: dict[str, Any]) -> str:
    """SHA256 hash of (parent ‖ canonical JSON of payload). Hex digest."""
    blob = parent + "|" + json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ledger_root() -> str:
    """Genesis hash for the per-target ledger chain."""
    return hashlib.sha256(b"agi.diffuser.ledger.root").hexdigest()


# ----------------------------------------------------------------------
# Built-in score factories: analytic Gaussian-mixture target
# ----------------------------------------------------------------------


def gaussian_mixture_log_density(
    means: Sequence[Sequence[float]],
    weights: Sequence[float],
    sigma_sq: float,
) -> Callable[[Sequence[float], float], float]:
    """Closed-form ``log p_t(x)`` for an isotropic mixture under the
    DDPM forward kernel ``q(x_t | x_0) = N(√ᾱ_t · x_0, (1−ᾱ_t)·I)``.

    Returns a function ``f(x, alpha_bar_t)`` evaluating the marginal
    log-density of ``x_t`` after ``t`` noising steps.
    """
    if not means or not weights:
        raise InsufficientData("need at least one component")
    if len(means) != len(weights):
        raise ValueError("means and weights must have equal length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    w_norm = [w / total for w in weights]
    log_w = [math.log(max(w, 1e-12)) for w in w_norm]
    d = len(means[0])
    if any(len(m) != d for m in means):
        raise ValueError("means must share dimension")
    if sigma_sq <= 0:
        raise ValueError("sigma_sq must be positive")

    def log_density(x: Sequence[float], alpha_bar_t: float) -> float:
        if not (0.0 < alpha_bar_t <= 1.0):
            raise ValueError("alpha_bar_t must be in (0, 1]")
        # x_t = sqrt(alpha_bar_t)·x_0 + sqrt(1-alpha_bar_t)·z
        # so under each component k:  N(sqrt(alpha_bar_t)·mu_k, (alpha_bar_t·sigma_sq + (1-alpha_bar_t)) · I)
        scale = math.sqrt(alpha_bar_t)
        var = alpha_bar_t * sigma_sq + (1.0 - alpha_bar_t)
        terms: list[float] = []
        for k, m in enumerate(means):
            shifted = [scale * mi for mi in m]
            terms.append(log_w[k] + gaussian_log_density(x, shifted, var))
        return log_sumexp(terms)

    return log_density


def gaussian_mixture_score(
    means: Sequence[Sequence[float]],
    weights: Sequence[float],
    sigma_sq: float,
) -> Callable[[Sequence[float], float], list[float]]:
    """Analytic score ``∇_x log p_t(x)`` for a Gaussian mixture under the
    DDPM forward kernel.  Returns a closure with the same signature
    the runtime expects from any registered score function.
    """
    if not means or not weights:
        raise InsufficientData("need at least one component")
    if len(means) != len(weights):
        raise ValueError("means and weights must have equal length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    w_norm = [w / total for w in weights]
    log_w = [math.log(max(w, 1e-12)) for w in w_norm]
    d = len(means[0])
    if any(len(m) != d for m in means):
        raise ValueError("means must share dimension")
    if sigma_sq <= 0:
        raise ValueError("sigma_sq must be positive")

    def score(x: Sequence[float], alpha_bar_t: float) -> list[float]:
        if not (0.0 < alpha_bar_t <= 1.0):
            raise ValueError("alpha_bar_t must be in (0, 1]")
        scale = math.sqrt(alpha_bar_t)
        var = alpha_bar_t * sigma_sq + (1.0 - alpha_bar_t)
        log_terms: list[float] = []
        shifted_means: list[list[float]] = []
        for k, m in enumerate(means):
            shifted = [scale * mi for mi in m]
            shifted_means.append(shifted)
            log_terms.append(log_w[k] + gaussian_log_density(x, shifted, var))
        # Responsibilities r_k = w_k · N(x | μ_k', σ²) / Σ ...
        lse = log_sumexp(log_terms)
        r = [math.exp(lt - lse) for lt in log_terms]
        # Score = Σ_k r_k · (μ_k' − x) / σ²
        out = [0.0] * d
        for k, mu_k in enumerate(shifted_means):
            for i in range(d):
                out[i] += r[k] * (mu_k[i] - x[i]) / var
        return out

    return score


def _identity_score(_x: Sequence[float], _alpha_bar_t: float) -> list[float]:
    return [0.0] * len(_x)


# ----------------------------------------------------------------------
# Built-in D3PM (discrete-state) transition kernels
# ----------------------------------------------------------------------


def absorbing_d3pm_kernel(K: int, mask_state: int) -> Callable[[int, float, random.Random], int]:
    """Absorbing D3PM kernel (Austin et al. 2021, §4.1).

    With probability ``β_t`` any non-mask state transitions to
    ``mask_state``; once absorbed, it stays there.
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    if not (0 <= mask_state < K):
        raise ValueError("mask_state out of range")

    def step(x: int, beta_t: float, rng: random.Random) -> int:
        if not (0.0 <= beta_t < 1.0):
            raise ValueError("beta_t must be in [0, 1)")
        if x == mask_state:
            return mask_state
        if rng.random() < beta_t:
            return mask_state
        return x

    return step


def uniform_d3pm_kernel(K: int) -> Callable[[int, float, random.Random], int]:
    """Uniform D3PM kernel (Austin et al. 2021, §4.2).

    ``q(x_t | x_{t-1}) = (1 − β_t)·δ_{x_{t-1}} + β_t · Uniform``.
    """
    if K < 2:
        raise ValueError("K must be >= 2")

    def step(x: int, beta_t: float, rng: random.Random) -> int:
        if not (0.0 <= beta_t < 1.0):
            raise ValueError("beta_t must be in [0, 1)")
        if rng.random() < beta_t:
            return rng.randrange(K)
        return x

    return step


# ----------------------------------------------------------------------
# Configuration and dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DiffuserConfig:
    """Configuration for the Diffuser primitive.

    ``T`` is the maximum diffusion horizon used by stochastic samplers;
    deterministic ODE samplers can take fewer steps.

    ``schedule_kind`` picks the default noise schedule for the variance-
    preserving family (``linear``, ``cosine``) or the variance-exploding
    family (``karras``).
    """

    dim: int = 2
    T: int = 1000
    schedule_kind: str = SCHEDULE_LINEAR
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    karras_sigma_min: float = 0.002
    karras_sigma_max: float = 80.0
    karras_rho: float = 7.0
    cosine_s: float = 8e-3
    seed: int = 0
    girsanov_mu: float = 1.0
    second_moment_default: float = 1.0
    histogram_bins: int = 16
    tv_confidence: float = 0.05

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise InvalidConfig("dim must be positive")
        if self.T <= 0:
            raise InvalidConfig("T must be positive")
        if self.schedule_kind not in KNOWN_SCHEDULES:
            raise InvalidConfig(
                f"unknown schedule_kind {self.schedule_kind!r}; "
                f"known: {KNOWN_SCHEDULES}"
            )
        if self.beta_start <= 0 or self.beta_end <= 0:
            raise InvalidConfig("beta endpoints must be positive")
        if self.beta_start > self.beta_end:
            raise InvalidConfig("beta_start must be <= beta_end")
        if self.karras_sigma_min <= 0 or self.karras_sigma_max <= 0:
            raise InvalidConfig("karras sigmas must be positive")
        if self.karras_sigma_min >= self.karras_sigma_max:
            raise InvalidConfig("karras_sigma_min must be < karras_sigma_max")
        if self.karras_rho <= 0:
            raise InvalidConfig("karras_rho must be positive")
        if self.cosine_s <= 0:
            raise InvalidConfig("cosine_s must be positive")
        if self.girsanov_mu < 0:
            raise InvalidConfig("girsanov_mu must be non-negative")
        if self.second_moment_default < 0:
            raise InvalidConfig("second_moment_default must be non-negative")
        if self.histogram_bins < 2:
            raise InvalidConfig("histogram_bins must be >= 2")
        if not (0.0 < self.tv_confidence < 1.0):
            raise InvalidConfig("tv_confidence must be in (0, 1)")


@dataclass(frozen=True)
class NoiseSchedule:
    """A noise schedule materialised as four parallel lists.

    For VP schedules, ``betas[t] ∈ (0, 1)``, ``alpha_bar[t] = Π_{s≤t}
    (1−β_s)``, and ``sigmas[t] = √(1 − ᾱ_t)``.  For VE schedules
    (Karras), ``betas`` is set to zeros and ``sigmas`` carries the σ_i;
    ``alpha_bar`` is set to ``1.0 / (1.0 + σ²)`` (the conventional VP
    embedding).
    """

    kind: str
    T: int
    betas: tuple[float, ...]
    alpha_bar: tuple[float, ...]
    sigmas: tuple[float, ...]


@dataclass(frozen=True)
class StepOutput:
    """Output of a single reverse-time step."""

    target_id: str
    algorithm: str
    t_from: int
    t_to: int
    x_before: tuple[float, ...]
    x_after: tuple[float, ...]
    score_norm: float
    chain_head: str


@dataclass(frozen=True)
class Sample:
    """A full reverse-time trajectory result."""

    target_id: str
    algorithm: str
    num_steps: int
    trajectory: tuple[tuple[float, ...], ...]
    final: tuple[float, ...]
    chain_head: str


@dataclass(frozen=True)
class FitReport:
    """Report from a score-matching fit pass."""

    target_id: str
    n_examples: int
    final_loss: float
    loss_history: tuple[float, ...]
    chain_head: str


@dataclass(frozen=True)
class Certificate:
    """Certificate bundling four quantitative bounds on sample quality."""

    target_id: str
    algorithm: str
    n_samples: int
    score_error: float
    girsanov_tv_bound: float
    empirical_tv: float
    empirical_tv_half_width: float
    elbo_per_step: float
    chain_head: str


@dataclass(frozen=True)
class DiffuserReport:
    """Aggregate report on a Diffuser's lifetime."""

    n_targets: int
    n_steps: int
    n_samples: int
    n_certifications: int
    n_fits: int


# ----------------------------------------------------------------------
# Event publisher type
# ----------------------------------------------------------------------

EventPublisher = Callable[[str, dict[str, Any]], None]


# ----------------------------------------------------------------------
# Per-target state container
# ----------------------------------------------------------------------


@dataclass
class _Target:
    """Internal per-target state."""

    target_id: str
    dim: int
    score_fn: Callable[[Sequence[float], float], Sequence[float]]
    classifier_grad_fn: Callable[[Sequence[float], float, Any], Sequence[float]] | None
    cond_score_fn: Callable[[Sequence[float], float, Any], Sequence[float]] | None
    uncond_score_fn: Callable[[Sequence[float], float], Sequence[float]] | None
    consistency_fn: Callable[[Sequence[float], float], Sequence[float]] | None
    flow_vector_fn: Callable[[Sequence[float], float], Sequence[float]] | None
    d3pm_K: int | None
    d3pm_kernel: Callable[[int, float, random.Random], int] | None
    d3pm_x0_proposal: Callable[[int, float, random.Random], int] | None
    second_moment: float
    score_error_floor: float
    fit_W: list[list[float]] = field(default_factory=list)
    chain_head: str = field(default_factory=ledger_root)


# ----------------------------------------------------------------------
# The primitive
# ----------------------------------------------------------------------


class Diffuser:
    """Score-based generative modelling as a runtime primitive.

    Construct with a ``DiffuserConfig`` (or top-level keyword args),
    ``register()`` one or more targets with a score function, then call
    ``sample()`` to draw samples.  All operations are thread-safe and
    append a hash-linked ledger entry per target.
    """

    def __init__(
        self,
        config: DiffuserConfig | None = None,
        *,
        dim: int | None = None,
        T: int | None = None,
        schedule_kind: str | None = None,
        seed: int | None = None,
        publisher: EventPublisher | None = None,
    ) -> None:
        if config is None:
            kw: dict[str, Any] = {}
            if dim is not None:
                kw["dim"] = dim
            if T is not None:
                kw["T"] = T
            if schedule_kind is not None:
                kw["schedule_kind"] = schedule_kind
            if seed is not None:
                kw["seed"] = seed
            config = DiffuserConfig(**kw)
        elif not isinstance(config, DiffuserConfig):
            raise InvalidConfig("config must be a DiffuserConfig")
        self.config = config
        self._publisher = publisher
        self._lock = threading.RLock()
        self._rng = random.Random(self.config.seed)
        self._targets: dict[str, _Target] = {}
        self._schedule = self._build_schedule()
        self._n_steps = 0
        self._n_samples = 0
        self._n_certifications = 0
        self._n_fits = 0
        self._publish(
            DIFFUSER_STARTED,
            {
                "dim": self.config.dim,
                "T": self.config.T,
                "schedule": self.config.schedule_kind,
                "seed": self.config.seed,
            },
        )

    # ------------------------------------------------------------------
    # event publishing
    # ------------------------------------------------------------------

    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        pub = self._publisher
        if pub is not None:
            try:
                pub(kind, data)
            except Exception:
                # The runtime tolerates broken publishers; just suppress.
                pass

    # ------------------------------------------------------------------
    # schedule construction
    # ------------------------------------------------------------------

    def _build_schedule(self) -> NoiseSchedule:
        kind = self.config.schedule_kind
        T = self.config.T
        if kind == SCHEDULE_LINEAR:
            betas = linear_beta_schedule(
                T,
                beta_start=self.config.beta_start,
                beta_end=self.config.beta_end,
            )
            ab = alpha_bar_from_betas(betas)
            sigmas = [math.sqrt(max(1.0 - a, 0.0)) for a in ab]
            return NoiseSchedule(
                kind=kind,
                T=T,
                betas=tuple(betas),
                alpha_bar=tuple(ab),
                sigmas=tuple(sigmas),
            )
        if kind == SCHEDULE_COSINE:
            betas = cosine_beta_schedule(T, s=self.config.cosine_s)
            ab = alpha_bar_from_betas(betas)
            sigmas = [math.sqrt(max(1.0 - a, 0.0)) for a in ab]
            return NoiseSchedule(
                kind=kind,
                T=T,
                betas=tuple(betas),
                alpha_bar=tuple(ab),
                sigmas=tuple(sigmas),
            )
        if kind == SCHEDULE_KARRAS:
            sigmas = karras_sigma_schedule(
                T,
                sigma_min=self.config.karras_sigma_min,
                sigma_max=self.config.karras_sigma_max,
                rho=self.config.karras_rho,
            )
            # VP embedding: alpha_bar = 1/(1+sigma²)
            ab = [1.0 / (1.0 + s * s) for s in sigmas]
            return NoiseSchedule(
                kind=kind,
                T=T,
                betas=tuple([0.0] * T),
                alpha_bar=tuple(ab[:T]),
                sigmas=tuple(sigmas[:T]),
            )
        raise UnknownSchedule(f"unknown schedule {kind!r}")

    @property
    def schedule(self) -> NoiseSchedule:
        return self._schedule

    # ------------------------------------------------------------------
    # target registration
    # ------------------------------------------------------------------

    def register(
        self,
        target_id: str,
        *,
        dim: int | None = None,
        score_fn: Callable[[Sequence[float], float], Sequence[float]] | None = None,
        classifier_grad_fn: Callable[[Sequence[float], float, Any], Sequence[float]] | None = None,
        cond_score_fn: Callable[[Sequence[float], float, Any], Sequence[float]] | None = None,
        uncond_score_fn: Callable[[Sequence[float], float], Sequence[float]] | None = None,
        consistency_fn: Callable[[Sequence[float], float], Sequence[float]] | None = None,
        flow_vector_fn: Callable[[Sequence[float], float], Sequence[float]] | None = None,
        d3pm_K: int | None = None,
        d3pm_kernel: Callable[[int, float, random.Random], int] | None = None,
        d3pm_x0_proposal: Callable[[int, float, random.Random], int] | None = None,
        second_moment: float | None = None,
        score_error_floor: float = 0.0,
    ) -> None:
        """Register a target distribution by name.

        At least one of ``score_fn``, ``cond_score_fn``,
        ``consistency_fn``, ``flow_vector_fn``, or a D3PM kernel must
        be supplied (which family will be sampled depends on the
        algorithm passed to ``sample()``).
        """
        if not target_id:
            raise InvalidTarget("target_id must be non-empty")
        if target_id in self._targets:
            raise InvalidTarget(f"target {target_id!r} already registered")
        d = dim if dim is not None else self.config.dim
        if d <= 0:
            raise InvalidTarget("dim must be positive")
        provided = [
            score_fn,
            cond_score_fn,
            consistency_fn,
            flow_vector_fn,
            d3pm_kernel,
        ]
        if all(p is None for p in provided):
            raise InvalidTarget(
                "must provide one of score_fn / cond_score_fn / consistency_fn / "
                "flow_vector_fn / d3pm_kernel"
            )
        if d3pm_kernel is not None and (d3pm_K is None or d3pm_K < 2):
            raise InvalidTarget("D3PM kernel requires d3pm_K >= 2")
        if score_error_floor < 0:
            raise InvalidTarget("score_error_floor must be non-negative")
        sm = (
            second_moment
            if second_moment is not None
            else self.config.second_moment_default
        )
        if sm < 0:
            raise InvalidTarget("second_moment must be non-negative")
        eff_score = score_fn if score_fn is not None else _identity_score
        with self._lock:
            t = _Target(
                target_id=target_id,
                dim=d,
                score_fn=eff_score,
                classifier_grad_fn=classifier_grad_fn,
                cond_score_fn=cond_score_fn,
                uncond_score_fn=uncond_score_fn,
                consistency_fn=consistency_fn,
                flow_vector_fn=flow_vector_fn,
                d3pm_K=d3pm_K,
                d3pm_kernel=d3pm_kernel,
                d3pm_x0_proposal=d3pm_x0_proposal,
                second_moment=sm,
                score_error_floor=score_error_floor,
            )
            payload = {"target_id": target_id, "dim": d}
            t.chain_head = _hash_entry(t.chain_head, payload)
            self._targets[target_id] = t
        self._publish(DIFFUSER_REGISTERED, payload)

    def deregister(self, target_id: str) -> None:
        """Remove a previously-registered target."""
        with self._lock:
            if target_id not in self._targets:
                raise UnknownTarget(f"unknown target {target_id!r}")
            del self._targets[target_id]
        self._publish(DIFFUSER_DEREGISTERED, {"target_id": target_id})

    def list_targets(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._targets))

    # ------------------------------------------------------------------
    # public sampling surface
    # ------------------------------------------------------------------

    def sample(
        self,
        target_id: str,
        *,
        algorithm: str = ALG_DDPM,
        num_steps: int | None = None,
        x_init: Sequence[float] | None = None,
        condition: Any = None,
        guidance_scale: float = 0.0,
        record_trajectory: bool = True,
    ) -> Sample:
        """Draw one sample from ``target_id`` using ``algorithm``.

        For continuous (Euclidean) algorithms, ``x_init`` may be
        provided; otherwise the primitive seeds with ``N(0, I)`` (VP
        family) or ``N(0, σ_max² · I)`` (VE family).  For D3PM,
        ``x_init`` is the absorbing / fully-noised categorical state.

        ``guidance_scale`` > 0 enables classifier or classifier-free
        guidance on continuous algorithms; passing it to D3PM is a
        no-op.
        """
        if algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"unknown algorithm {algorithm!r}; known: {KNOWN_ALGORITHMS}"
            )
        with self._lock:
            t = self._targets.get(target_id)
        if t is None:
            raise UnknownTarget(f"unknown target {target_id!r}")
        if algorithm == ALG_DDPM:
            return self._sample_ddpm(t, num_steps, x_init, condition, guidance_scale, record_trajectory)
        if algorithm == ALG_DDIM:
            return self._sample_ddim(t, num_steps, x_init, condition, guidance_scale, record_trajectory)
        if algorithm == ALG_DPM_SOLVER_1:
            return self._sample_dpm_solver(t, num_steps, x_init, condition, guidance_scale, 1, record_trajectory)
        if algorithm == ALG_DPM_SOLVER_2:
            return self._sample_dpm_solver(t, num_steps, x_init, condition, guidance_scale, 2, record_trajectory)
        if algorithm == ALG_HEUN:
            return self._sample_heun(t, num_steps, x_init, condition, guidance_scale, record_trajectory)
        if algorithm == ALG_EULER_SDE:
            return self._sample_euler_sde(t, num_steps, x_init, condition, guidance_scale, record_trajectory)
        if algorithm == ALG_PF_ODE:
            return self._sample_pf_ode(t, num_steps, x_init, condition, guidance_scale, record_trajectory)
        if algorithm == ALG_PREDICTOR_CORRECTOR:
            return self._sample_pc(t, num_steps, x_init, condition, guidance_scale, record_trajectory)
        if algorithm == ALG_FLOW_MATCHING:
            return self._sample_flow_matching(t, num_steps, x_init, condition, record_trajectory)
        if algorithm == ALG_CONSISTENCY:
            return self._sample_consistency(t, num_steps, x_init, condition, record_trajectory)
        if algorithm == ALG_D3PM:
            return self._sample_d3pm(t, num_steps, x_init, record_trajectory)
        raise UnknownAlgorithm(f"unknown algorithm {algorithm!r}")

    def imagine(
        self,
        target_id: str,
        *,
        algorithm: str = ALG_DDIM,
        num_steps: int | None = None,
        condition: Any = None,
        guidance_scale: float = 0.0,
    ) -> Sample:
        """Convenience alias around :meth:`sample` for the imagination
        composition surface used by :class:`agi.imaginator.Imaginator`."""
        return self.sample(
            target_id,
            algorithm=algorithm,
            num_steps=num_steps,
            condition=condition,
            guidance_scale=guidance_scale,
        )

    # ------------------------------------------------------------------
    # internal: score evaluation with guidance
    # ------------------------------------------------------------------

    def _eval_score(
        self,
        t: _Target,
        x: Sequence[float],
        alpha_bar_t: float,
        condition: Any,
        guidance_scale: float,
    ) -> list[float]:
        """Compute ``∇_x log p_t(x | condition)`` under guidance."""
        if condition is None or guidance_scale == 0.0:
            return list(t.score_fn(x, alpha_bar_t))
        # Classifier guidance.
        if t.classifier_grad_fn is not None:
            s = list(t.score_fn(x, alpha_bar_t))
            g = list(t.classifier_grad_fn(x, alpha_bar_t, condition))
            if len(s) != len(g):
                raise GuidanceViolation(
                    "classifier gradient dimension does not match score dimension"
                )
            return [si + guidance_scale * gi for si, gi in zip(s, g)]
        # Classifier-free guidance.
        if t.cond_score_fn is not None:
            s_cond = list(t.cond_score_fn(x, alpha_bar_t, condition))
            base = (
                list(t.uncond_score_fn(x, alpha_bar_t))
                if t.uncond_score_fn is not None
                else list(t.score_fn(x, alpha_bar_t))
            )
            if len(s_cond) != len(base):
                raise GuidanceViolation(
                    "cond/uncond score dimension mismatch"
                )
            return [
                base[i] + (1.0 + guidance_scale) * (s_cond[i] - base[i])
                for i in range(len(base))
            ]
        raise GuidanceViolation(
            "guidance requested but neither classifier nor classifier-free "
            "score branch was registered"
        )

    def _eval_score_at_t(
        self,
        t: _Target,
        x: Sequence[float],
        t_idx: int,
        condition: Any,
        guidance_scale: float,
    ) -> list[float]:
        return self._eval_score(
            t, x, self._schedule.alpha_bar[t_idx], condition, guidance_scale
        )

    # ------------------------------------------------------------------
    # internal sampler implementations
    # ------------------------------------------------------------------

    def _init_x_vp(
        self, t: _Target, x_init: Sequence[float] | None
    ) -> list[float]:
        if x_init is not None:
            if len(x_init) != t.dim:
                raise InvalidTarget(
                    f"x_init dim {len(x_init)} != target dim {t.dim}"
                )
            return list(x_init)
        return normal_sample(self._rng, t.dim)

    def _init_x_ve(
        self, t: _Target, x_init: Sequence[float] | None, sigma_max: float
    ) -> list[float]:
        if x_init is not None:
            if len(x_init) != t.dim:
                raise InvalidTarget(
                    f"x_init dim {len(x_init)} != target dim {t.dim}"
                )
            return list(x_init)
        z = normal_sample(self._rng, t.dim)
        return [sigma_max * v for v in z]

    def _record_step(
        self,
        t: _Target,
        algorithm: str,
        t_from: int,
        t_to: int,
        x_before: Sequence[float],
        x_after: Sequence[float],
        score_norm: float,
    ) -> None:
        payload = {
            "target_id": t.target_id,
            "algorithm": algorithm,
            "t_from": t_from,
            "t_to": t_to,
            "score_norm": round(score_norm, 6),
        }
        t.chain_head = _hash_entry(t.chain_head, payload)
        self._n_steps += 1
        self._publish(DIFFUSER_STEP, payload)

    def _finish_sample(
        self,
        t: _Target,
        algorithm: str,
        traj: list[list[float]],
    ) -> Sample:
        payload = {
            "target_id": t.target_id,
            "algorithm": algorithm,
            "num_steps": len(traj) - 1 if traj else 0,
        }
        t.chain_head = _hash_entry(t.chain_head, payload)
        self._n_samples += 1
        self._publish(DIFFUSER_SAMPLED, payload)
        return Sample(
            target_id=t.target_id,
            algorithm=algorithm,
            num_steps=len(traj) - 1 if traj else 0,
            trajectory=tuple(tuple(x) for x in traj),
            final=tuple(traj[-1]) if traj else (),
            chain_head=t.chain_head,
        )

    def _resolved_num_steps(self, num_steps: int | None) -> int:
        if num_steps is None:
            return self.config.T
        if num_steps <= 0:
            raise InvalidConfig("num_steps must be positive")
        if num_steps > self.config.T:
            raise InvalidConfig(
                f"num_steps {num_steps} cannot exceed T {self.config.T}"
            )
        return num_steps

    def _step_indices(self, K: int) -> list[int]:
        """Uniformly-spaced descending indices from T-1 to 0, length K+1."""
        T = self.config.T
        if K >= T:
            return list(range(T - 1, -1, -1))
        # Step size dt = (T-1)/K, indices = round((K-i)·dt)
        out: list[int] = []
        for i in range(K + 1):
            idx = int(round((K - i) * (T - 1) / K))
            out.append(idx)
        return out

    def _sample_ddpm(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        guidance_scale: float,
        record_trajectory: bool,
    ) -> Sample:
        K = self._resolved_num_steps(num_steps)
        idxs = self._step_indices(K)
        x = self._init_x_vp(t, x_init)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        sched = self._schedule
        for step_i in range(K):
            t_from = idxs[step_i]
            t_to = idxs[step_i + 1]
            ab_t = sched.alpha_bar[t_from]
            ab_tm1 = sched.alpha_bar[t_to] if t_to >= 0 else 1.0
            beta_t = sched.betas[t_from] if sched.kind != SCHEDULE_KARRAS else 1.0 - ab_t / max(ab_tm1, 1e-12)
            beta_t = min(max(beta_t, 1e-8), 0.999)
            # eps prediction from score: eps = -sqrt(1-ab_t) · score
            score = self._eval_score_at_t(t, x, t_from, condition, guidance_scale)
            sn = math.sqrt(sum(s * s for s in score))
            sigma_t = math.sqrt(1.0 - ab_t)
            eps = [-sigma_t * s for s in score]
            # x_{t-1} mean
            inv_sqrt_alpha = 1.0 / math.sqrt(max(1.0 - beta_t, 1e-12))
            coef = beta_t / max(sigma_t, 1e-12)
            mean = [inv_sqrt_alpha * (xi - coef * ei) for xi, ei in zip(x, eps)]
            # noise variance per Ho 2020 Eq.7: σ² = β_t · (1−ᾱ_{t-1})/(1−ᾱ_t)
            if t_to >= 0:
                post_var = beta_t * (1.0 - ab_tm1) / max(1.0 - ab_t, 1e-12)
                post_var = max(post_var, 0.0)
                z = normal_sample(self._rng, t.dim)
                x = [m + math.sqrt(post_var) * zi for m, zi in zip(mean, z)]
            else:
                x = mean
            if record_trajectory:
                traj.append(list(x))
            self._record_step(t, ALG_DDPM, t_from, t_to, traj[-2] if record_trajectory else x, x, sn)
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, ALG_DDPM, traj)

    def _sample_ddim(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        guidance_scale: float,
        record_trajectory: bool,
    ) -> Sample:
        K = self._resolved_num_steps(num_steps)
        idxs = self._step_indices(K)
        x = self._init_x_vp(t, x_init)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        sched = self._schedule
        for step_i in range(K):
            t_from = idxs[step_i]
            t_to = idxs[step_i + 1]
            ab_t = sched.alpha_bar[t_from]
            ab_tm1 = sched.alpha_bar[t_to] if t_to >= 0 else 1.0
            score = self._eval_score_at_t(t, x, t_from, condition, guidance_scale)
            sn = math.sqrt(sum(s * s for s in score))
            sigma_t = math.sqrt(max(1.0 - ab_t, 0.0))
            eps = [-sigma_t * s for s in score]
            # Predict x_0
            sqrt_ab_t = math.sqrt(max(ab_t, 1e-12))
            x0_hat = [(xi - sigma_t * ei) / sqrt_ab_t for xi, ei in zip(x, eps)]
            # DDIM update with eta=0 (deterministic)
            sqrt_ab_tm1 = math.sqrt(max(ab_tm1, 1e-12))
            sigma_tm1 = math.sqrt(max(1.0 - ab_tm1, 0.0))
            x = [sqrt_ab_tm1 * x0i + sigma_tm1 * ei for x0i, ei in zip(x0_hat, eps)]
            if record_trajectory:
                traj.append(list(x))
            self._record_step(t, ALG_DDIM, t_from, t_to, traj[-2] if record_trajectory else x, x, sn)
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, ALG_DDIM, traj)

    def _sample_dpm_solver(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        guidance_scale: float,
        order: int,
        record_trajectory: bool,
    ) -> Sample:
        """DPM-Solver-{1,2} (Lu et al. 2022).

        Implementing in log-SNR (lambda) coordinates.  For VP:
        ``λ_t = ½ log(ᾱ_t / (1 − ᾱ_t))``.  Order-1 update::

            x_{s} = (α_s / α_t) x_t − σ_s (e^h − 1) ε̂_θ(x_t, t)

        with ``h = λ_s − λ_t``.  Order-2 takes an additional midpoint
        evaluation.
        """
        K = self._resolved_num_steps(num_steps)
        idxs = self._step_indices(K)
        x = self._init_x_vp(t, x_init)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        sched = self._schedule
        algo_name = ALG_DPM_SOLVER_1 if order == 1 else ALG_DPM_SOLVER_2
        for step_i in range(K):
            t_from = idxs[step_i]
            t_to = idxs[step_i + 1]
            ab_t = sched.alpha_bar[t_from]
            ab_s = sched.alpha_bar[t_to] if t_to >= 0 else 1.0
            sigma_t = math.sqrt(max(1.0 - ab_t, 1e-12))
            sigma_s = math.sqrt(max(1.0 - ab_s, 1e-12))
            alpha_t = math.sqrt(max(ab_t, 1e-12))
            alpha_s = math.sqrt(max(ab_s, 1e-12))
            lam_t = 0.5 * math.log(max(ab_t / max(1.0 - ab_t, 1e-12), 1e-12))
            lam_s = 0.5 * math.log(max(ab_s / max(1.0 - ab_s, 1e-12), 1e-12))
            h = lam_s - lam_t
            score = self._eval_score_at_t(t, x, t_from, condition, guidance_scale)
            sn = math.sqrt(sum(s * s for s in score))
            eps = [-sigma_t * s for s in score]
            if order == 1:
                x_new = [
                    (alpha_s / alpha_t) * xi - sigma_s * (math.exp(h) - 1.0) * ei
                    for xi, ei in zip(x, eps)
                ]
            else:
                # Order 2: midpoint
                lam_mid = 0.5 * (lam_t + lam_s)
                # Map lam_mid back to alpha_bar via inverse: ab = 1/(1+e^{-2λ})
                ab_mid = 1.0 / (1.0 + math.exp(-2.0 * lam_mid))
                alpha_mid = math.sqrt(ab_mid)
                sigma_mid = math.sqrt(max(1.0 - ab_mid, 1e-12))
                # Find closest index for midpoint t-eval
                t_mid = max(
                    min(int(round(0.5 * (t_from + t_to))), self.config.T - 1), 0
                )
                # First-order step to midpoint
                u = [
                    (alpha_mid / alpha_t) * xi - sigma_mid * (math.exp(0.5 * h) - 1.0) * ei
                    for xi, ei in zip(x, eps)
                ]
                score_mid = self._eval_score_at_t(t, u, t_mid, condition, guidance_scale)
                eps_mid = [-sigma_mid * s for s in score_mid]
                x_new = [
                    (alpha_s / alpha_t) * xi - sigma_s * (math.exp(h) - 1.0) * emi
                    for xi, emi in zip(x, eps_mid)
                ]
            x = x_new
            if record_trajectory:
                traj.append(list(x))
            self._record_step(t, algo_name, t_from, t_to, traj[-2] if record_trajectory else x, x, sn)
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, algo_name, traj)

    def _sample_heun(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        guidance_scale: float,
        record_trajectory: bool,
    ) -> Sample:
        """Karras EDM Heun's method on the probability-flow ODE."""
        K = self._resolved_num_steps(num_steps)
        idxs = self._step_indices(K)
        x = self._init_x_vp(t, x_init)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        sched = self._schedule
        for step_i in range(K):
            t_from = idxs[step_i]
            t_to = idxs[step_i + 1]
            ab_t = sched.alpha_bar[t_from]
            ab_s = sched.alpha_bar[t_to] if t_to >= 0 else 1.0
            sigma_t = math.sqrt(max(1.0 - ab_t, 1e-12)) / math.sqrt(max(ab_t, 1e-12))
            sigma_s = math.sqrt(max(1.0 - ab_s, 1e-12)) / math.sqrt(max(ab_s, 1e-12))
            d_sigma = sigma_s - sigma_t
            score = self._eval_score_at_t(t, x, t_from, condition, guidance_scale)
            sn = math.sqrt(sum(s * s for s in score))
            # d/dσ x = -σ · score(x, σ) (Karras EDM form, normalised score)
            deriv = [-sigma_t * s for s in score]
            x_euler = [xi + d_sigma * di for xi, di in zip(x, deriv)]
            if abs(sigma_s) < 1e-12:
                x = x_euler
            else:
                score_next = self._eval_score_at_t(t, x_euler, t_to, condition, guidance_scale)
                deriv_next = [-sigma_s * s for s in score_next]
                x = [
                    xi + d_sigma * 0.5 * (di + dni)
                    for xi, di, dni in zip(x, deriv, deriv_next)
                ]
            if record_trajectory:
                traj.append(list(x))
            self._record_step(t, ALG_HEUN, t_from, t_to, traj[-2] if record_trajectory else x, x, sn)
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, ALG_HEUN, traj)

    def _sample_euler_sde(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        guidance_scale: float,
        record_trajectory: bool,
    ) -> Sample:
        """Euler-Maruyama on the reverse-time SDE (Song 2021)."""
        K = self._resolved_num_steps(num_steps)
        idxs = self._step_indices(K)
        x = self._init_x_vp(t, x_init)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        sched = self._schedule
        for step_i in range(K):
            t_from = idxs[step_i]
            t_to = idxs[step_i + 1]
            ab_t = sched.alpha_bar[t_from]
            ab_s = sched.alpha_bar[t_to] if t_to >= 0 else 1.0
            score = self._eval_score_at_t(t, x, t_from, condition, guidance_scale)
            sn = math.sqrt(sum(s * s for s in score))
            beta_t = sched.betas[t_from] if sched.kind != SCHEDULE_KARRAS else 1.0 - ab_t / max(ab_s, 1e-12)
            beta_t = min(max(beta_t, 1e-8), 0.999)
            # Reverse SDE: dx = [-β/2 · x − β · score] dt + √β dw̄
            drift = [-0.5 * beta_t * xi - beta_t * si for xi, si in zip(x, score)]
            z = normal_sample(self._rng, t.dim)
            x = [
                xi + (-1.0) * di + math.sqrt(beta_t) * zi
                for xi, di, zi in zip(x, drift, z)
            ]
            if record_trajectory:
                traj.append(list(x))
            self._record_step(t, ALG_EULER_SDE, t_from, t_to, traj[-2] if record_trajectory else x, x, sn)
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, ALG_EULER_SDE, traj)

    def _sample_pf_ode(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        guidance_scale: float,
        record_trajectory: bool,
    ) -> Sample:
        """Probability-flow ODE Euler step (Song 2021).

        ``dx/dt = -½β(t) (x + score(x, t))`` in VP coordinates.
        """
        K = self._resolved_num_steps(num_steps)
        idxs = self._step_indices(K)
        x = self._init_x_vp(t, x_init)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        sched = self._schedule
        for step_i in range(K):
            t_from = idxs[step_i]
            t_to = idxs[step_i + 1]
            ab_t = sched.alpha_bar[t_from]
            ab_s = sched.alpha_bar[t_to] if t_to >= 0 else 1.0
            score = self._eval_score_at_t(t, x, t_from, condition, guidance_scale)
            sn = math.sqrt(sum(s * s for s in score))
            beta_t = sched.betas[t_from] if sched.kind != SCHEDULE_KARRAS else 1.0 - ab_t / max(ab_s, 1e-12)
            beta_t = min(max(beta_t, 1e-8), 0.999)
            # PF-ODE: dx = -0.5 β · (x + score) dt; reverse-in-time so dt is positive going backward
            drift = [-0.5 * beta_t * (xi + si) for xi, si in zip(x, score)]
            x = [xi + (-1.0) * di for xi, di in zip(x, drift)]
            if record_trajectory:
                traj.append(list(x))
            self._record_step(t, ALG_PF_ODE, t_from, t_to, traj[-2] if record_trajectory else x, x, sn)
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, ALG_PF_ODE, traj)

    def _sample_pc(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        guidance_scale: float,
        record_trajectory: bool,
        *,
        n_corrector_steps: int = 1,
        snr: float = 0.16,
    ) -> Sample:
        """Predictor-corrector sampler (Song 2021).

        Predictor: one reverse-SDE Euler-Maruyama step.
        Corrector: ``n_corrector_steps`` Langevin moves with step size
        ``ε = 2·(snr · ‖z‖ / ‖score‖)²``.
        """
        K = self._resolved_num_steps(num_steps)
        idxs = self._step_indices(K)
        x = self._init_x_vp(t, x_init)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        sched = self._schedule
        for step_i in range(K):
            t_from = idxs[step_i]
            t_to = idxs[step_i + 1]
            ab_t = sched.alpha_bar[t_from]
            ab_s = sched.alpha_bar[t_to] if t_to >= 0 else 1.0
            # Predictor (reverse-SDE)
            score = self._eval_score_at_t(t, x, t_from, condition, guidance_scale)
            sn = math.sqrt(sum(s * s for s in score))
            beta_t = sched.betas[t_from] if sched.kind != SCHEDULE_KARRAS else 1.0 - ab_t / max(ab_s, 1e-12)
            beta_t = min(max(beta_t, 1e-8), 0.999)
            drift = [-0.5 * beta_t * xi - beta_t * si for xi, si in zip(x, score)]
            z = normal_sample(self._rng, t.dim)
            x = [
                xi - di + math.sqrt(beta_t) * zi for xi, di, zi in zip(x, drift, z)
            ]
            # Corrector (Langevin)
            for _ in range(n_corrector_steps):
                score_c = self._eval_score_at_t(t, x, t_to if t_to >= 0 else 0, condition, guidance_scale)
                z = normal_sample(self._rng, t.dim)
                score_norm = math.sqrt(sum(s * s for s in score_c)) + 1e-12
                z_norm = math.sqrt(sum(zi * zi for zi in z)) + 1e-12
                eps = 2.0 * (snr * z_norm / score_norm) ** 2
                x = [
                    xi + eps * si + math.sqrt(2.0 * eps) * zi
                    for xi, si, zi in zip(x, score_c, z)
                ]
            if record_trajectory:
                traj.append(list(x))
            self._record_step(t, ALG_PREDICTOR_CORRECTOR, t_from, t_to, traj[-2] if record_trajectory else x, x, sn)
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, ALG_PREDICTOR_CORRECTOR, traj)

    def _sample_flow_matching(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        record_trajectory: bool,
    ) -> Sample:
        """Flow matching (Lipman et al. 2023) integrator.

        Vector field ``v_θ(x, t)`` is integrated forward in t ∈ [0, 1].
        Default fallback: if ``flow_vector_fn`` is unregistered, the
        primitive synthesises one from the registered score function
        via the canonical OT path mapping (``v(x,t) = (1−t)·x + t·E[x_1
        | x_t]``).
        """
        K = self._resolved_num_steps(num_steps)
        if t.flow_vector_fn is None and t.score_fn is None:
            raise InvalidTarget("flow_matching requires flow_vector_fn or score_fn")
        x = self._init_x_vp(t, x_init)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        dt = 1.0 / K
        for step_i in range(K):
            tau = step_i / K
            if t.flow_vector_fn is not None:
                v = list(t.flow_vector_fn(x, tau))
            else:
                # Synthesise from score: at time τ, x = (1-τ)x_0 + τ x_1
                #   ⇒ x_0 ≈ (x - τ·E[x_1])/(1-τ)
                # We use score to estimate x_1 via Tweedie.
                ab_t = max(1.0 - (1.0 - tau) ** 2, 1e-6)
                score = list(t.score_fn(x, ab_t))
                sigma_sq = 1.0 - ab_t
                tweedie = [xi + sigma_sq * si for xi, si in zip(x, score)]
                v = [(t1 - xi) / max(1.0 - tau, 1e-6) for t1, xi in zip(tweedie, x)]
            sn = math.sqrt(sum(vi * vi for vi in v))
            x = [xi + dt * vi for xi, vi in zip(x, v)]
            if record_trajectory:
                traj.append(list(x))
            self._record_step(
                t, ALG_FLOW_MATCHING, step_i, step_i + 1,
                traj[-2] if record_trajectory else x, x, sn,
            )
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, ALG_FLOW_MATCHING, traj)

    def _sample_consistency(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        condition: Any,
        record_trajectory: bool,
    ) -> Sample:
        """Consistency model multistep sampling (Song et al. 2023).

        If a ``consistency_fn`` is registered, we apply it one or
        ``num_steps`` times.  Multistep alternates noise injection
        with consistency calls.
        """
        if t.consistency_fn is None:
            raise InvalidTarget("consistency requires consistency_fn")
        K = num_steps if num_steps is not None else 1
        if K <= 0:
            raise InvalidConfig("num_steps must be positive")
        sched = self._schedule
        sigma_max = sched.sigmas[0] if sched.kind == SCHEDULE_KARRAS else math.sqrt(
            max(1.0 - sched.alpha_bar[0], 1e-12)
        )
        x = self._init_x_ve(t, x_init, sigma_max)
        traj: list[list[float]] = [list(x)] if record_trajectory else []
        for step_i in range(K):
            sigma_idx = max(
                int((1.0 - step_i / K) * (sched.T - 1)), 0
            )
            ab_t = sched.alpha_bar[sigma_idx]
            x_new = list(t.consistency_fn(x, ab_t))
            sn = math.sqrt(sum((a - b) ** 2 for a, b in zip(x_new, x))) / max(
                math.sqrt(len(x)), 1.0
            )
            if step_i < K - 1:
                # Re-noise per multistep consistency schedule.
                target_sigma = math.sqrt(max(1.0 - sched.alpha_bar[
                    max(int((1.0 - (step_i + 1) / K) * (sched.T - 1)), 0)
                ], 0.0))
                z = normal_sample(self._rng, t.dim)
                x = [xn + target_sigma * zi for xn, zi in zip(x_new, z)]
            else:
                x = x_new
            if record_trajectory:
                traj.append(list(x))
            self._record_step(
                t, ALG_CONSISTENCY, sigma_idx, sigma_idx,
                traj[-2] if record_trajectory else x, x, sn,
            )
        if not record_trajectory:
            traj = [list(x)]
        return self._finish_sample(t, ALG_CONSISTENCY, traj)

    def _sample_d3pm(
        self,
        t: _Target,
        num_steps: int | None,
        x_init: Sequence[float] | None,
        record_trajectory: bool,
    ) -> Sample:
        """D3PM categorical reverse sampler (Austin et al. 2021)."""
        if t.d3pm_kernel is None or t.d3pm_K is None:
            raise InvalidTarget("d3pm requires d3pm_kernel + d3pm_K")
        K = self._resolved_num_steps(num_steps)
        idxs = self._step_indices(K)
        # Initial state: x_init (if int) else random.
        if x_init is not None:
            if not isinstance(x_init, (list, tuple)) or len(x_init) != t.dim:
                raise InvalidTarget(
                    "x_init for d3pm must be a length-dim sequence of ints"
                )
            x = [int(v) for v in x_init]
        else:
            x = [self._rng.randrange(t.d3pm_K) for _ in range(t.dim)]
        traj: list[list[float]] = [[float(xi) for xi in x]] if record_trajectory else []
        sched = self._schedule
        for step_i in range(K):
            t_from = idxs[step_i]
            t_to = idxs[step_i + 1]
            beta_t = sched.betas[t_from] if sched.kind != SCHEDULE_KARRAS else 0.0
            # Reverse via reverse kernel implemented as: optionally project
            # via x0 proposal, then reapply forward up to t_to.
            if t.d3pm_x0_proposal is not None:
                x0_pred = [t.d3pm_x0_proposal(xi, beta_t, self._rng) for xi in x]
                # Re-noise to t_to (one step of forward kernel for each dim)
                if t_to > 0:
                    beta_to = sched.betas[t_to] if sched.kind != SCHEDULE_KARRAS else 0.0
                    x = [t.d3pm_kernel(x0i, beta_to, self._rng) for x0i in x0_pred]
                else:
                    x = x0_pred
            else:
                # Crude reverse: just resample under the same forward kernel.
                x = [t.d3pm_kernel(xi, beta_t, self._rng) for xi in x]
            if record_trajectory:
                traj.append([float(xi) for xi in x])
            self._record_step(
                t, ALG_D3PM, t_from, t_to,
                tuple(float(v) for v in (traj[-2] if record_trajectory else x)),
                tuple(float(v) for v in x), 0.0,
            )
        if not record_trajectory:
            traj = [[float(xi) for xi in x]]
        return self._finish_sample(t, ALG_D3PM, traj)

    # ------------------------------------------------------------------
    # fit: score-matching via finite differences against samples
    # ------------------------------------------------------------------

    def fit(
        self,
        target_id: str,
        data: Sequence[Sequence[float]],
        *,
        num_epochs: int = 5,
        learning_rate: float = 0.05,
        time_index: int | None = None,
    ) -> FitReport:
        """Fit a tiny diagnostic linear score model ``s_θ(x) = W·x``
        on the given data via Hyvärinen score matching (1-D step):
        minimise ``L(W) = ½ E‖Wx‖² + E[tr(W)]``.

        The optimum is ``W* = -Cov(x)⁻¹`` (matching the score of the
        empirical covariance Gaussian).  We solve via simple gradient
        descent on the empirical loss; this fit is intended for tests
        and as a baseline, not as a learnt large-scale score network.
        """
        with self._lock:
            t = self._targets.get(target_id)
        if t is None:
            raise UnknownTarget(f"unknown target {target_id!r}")
        if not data:
            raise InsufficientData("data must be non-empty")
        d = t.dim
        if any(len(x) != d for x in data):
            raise ValueError("data must be a sequence of length-dim sequences")
        if num_epochs <= 0:
            raise InvalidConfig("num_epochs must be positive")
        if learning_rate <= 0:
            raise InvalidConfig("learning_rate must be positive")
        # Initialise W from existing fit_W or as -I.
        with self._lock:
            W = (
                [row[:] for row in t.fit_W]
                if t.fit_W and len(t.fit_W) == d and all(len(r) == d for r in t.fit_W)
                else [
                    [-1.0 if i == j else 0.0 for j in range(d)]
                    for i in range(d)
                ]
            )
        loss_hist: list[float] = []
        n = len(data)
        for _ in range(num_epochs):
            # Compute loss and gradient.
            tot = 0.0
            grad = [[0.0] * d for _ in range(d)]
            for x in data:
                Wx = [sum(W[i][j] * x[j] for j in range(d)) for i in range(d)]
                tot += 0.5 * sum(v * v for v in Wx)
                # gradient of 0.5||Wx||²:  Wx · xᵀ ⇒ G_{ij} += Wx_i · x_j
                for i in range(d):
                    for j in range(d):
                        grad[i][j] += Wx[i] * x[j]
            tot /= n
            # Add tr(W) term and its gradient (identity).
            tot += sum(W[i][i] for i in range(d))
            for i in range(d):
                grad[i][i] += n
            for i in range(d):
                for j in range(d):
                    grad[i][j] /= n
                    W[i][j] -= learning_rate * grad[i][j]
            loss_hist.append(tot)
        # Persist.
        with self._lock:
            t.fit_W = [row[:] for row in W]
            payload = {
                "target_id": target_id,
                "n_examples": n,
                "final_loss": round(loss_hist[-1], 6),
            }
            t.chain_head = _hash_entry(t.chain_head, payload)
            self._n_fits += 1
        self._publish(DIFFUSER_FIT, payload)
        return FitReport(
            target_id=target_id,
            n_examples=n,
            final_loss=loss_hist[-1],
            loss_history=tuple(loss_hist),
            chain_head=t.chain_head,
        )

    def fitted_score(self, target_id: str, x: Sequence[float]) -> list[float]:
        """Evaluate the fitted diagnostic linear score model."""
        with self._lock:
            t = self._targets.get(target_id)
        if t is None:
            raise UnknownTarget(f"unknown target {target_id!r}")
        if not t.fit_W:
            raise InsufficientData("call fit() first")
        if len(x) != t.dim:
            raise ValueError("x dim mismatch")
        d = t.dim
        return [sum(t.fit_W[i][j] * x[j] for j in range(d)) for i in range(d)]

    # ------------------------------------------------------------------
    # certify: Girsanov + ELBO + empirical TV
    # ------------------------------------------------------------------

    def certify(
        self,
        target_id: str,
        samples: Sequence[Sequence[float]],
        *,
        target_samples: Sequence[Sequence[float]] | None = None,
        algorithm: str = ALG_DDPM,
        score_error: float | None = None,
    ) -> Certificate:
        """Return four quantitative bounds for the supplied sample batch.

        - Girsanov TV bound (closed form)
        - Empirical TV (with Hoeffding half-width) vs ``target_samples``
        - Per-step DDPM ELBO term averaged over the batch (diagnostic
          — needs ``target_samples`` to play the role of x_0)
        - Reported score-matching error floor
        """
        with self._lock:
            t = self._targets.get(target_id)
        if t is None:
            raise UnknownTarget(f"unknown target {target_id!r}")
        if not samples:
            raise InsufficientData("samples must be non-empty")
        if algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(f"unknown algorithm {algorithm!r}")
        d = t.dim
        if any(len(s) != d for s in samples):
            raise ValueError("samples ragged or wrong dim")
        # Score error: caller-supplied, else use registered floor.
        se = score_error if score_error is not None else t.score_error_floor
        if se < 0:
            raise InvalidConfig("score_error must be non-negative")
        # Girsanov bound.
        gir = girsanov_tv_bound(
            score_error=se,
            horizon=float(self.config.T),
            second_moment=t.second_moment,
            mu=self.config.girsanov_mu,
        )
        # Empirical TV (only meaningful if target_samples supplied).
        if target_samples is not None:
            if any(len(s) != d for s in target_samples):
                raise ValueError("target_samples ragged or wrong dim")
            tv, half = empirical_tv(
                samples,
                target_samples,
                bins=self.config.histogram_bins,
                delta=self.config.tv_confidence,
            )
        else:
            tv = 0.0
            half = 0.0
        # Per-step ELBO term.
        if target_samples is not None and self._schedule.kind != SCHEDULE_KARRAS:
            t_mid = self.config.T // 2
            beta_t = self._schedule.betas[t_mid]
            ab_t = self._schedule.alpha_bar[t_mid]
            ab_tm1 = self._schedule.alpha_bar[max(t_mid - 1, 0)]
            n_pairs = min(len(samples), len(target_samples))
            elbo = 0.0
            for i in range(n_pairs):
                elbo += ddpm_elbo_term(
                    x0=target_samples[i],
                    xt=samples[i],
                    alpha_bar_t=ab_t,
                    alpha_bar_tm1=ab_tm1,
                    beta_t=beta_t,
                    pred_x0=samples[i],
                )
            elbo /= max(n_pairs, 1)
        else:
            elbo = 0.0
        with self._lock:
            payload = {
                "target_id": target_id,
                "algorithm": algorithm,
                "n_samples": len(samples),
                "girsanov": round(gir, 6),
                "empirical_tv": round(tv, 6),
                "elbo_per_step": round(elbo, 6),
            }
            t.chain_head = _hash_entry(t.chain_head, payload)
            self._n_certifications += 1
        self._publish(DIFFUSER_CERTIFIED, payload)
        return Certificate(
            target_id=target_id,
            algorithm=algorithm,
            n_samples=len(samples),
            score_error=se,
            girsanov_tv_bound=gir,
            empirical_tv=tv,
            empirical_tv_half_width=half,
            elbo_per_step=elbo,
            chain_head=t.chain_head,
        )

    # ------------------------------------------------------------------
    # reports / lifecycle
    # ------------------------------------------------------------------

    def report(self) -> DiffuserReport:
        with self._lock:
            return DiffuserReport(
                n_targets=len(self._targets),
                n_steps=self._n_steps,
                n_samples=self._n_samples,
                n_certifications=self._n_certifications,
                n_fits=self._n_fits,
            )

    def reset(self) -> None:
        """Reset all per-target state and counters; keep schedule + config."""
        with self._lock:
            self._targets.clear()
            self._n_steps = 0
            self._n_samples = 0
            self._n_certifications = 0
            self._n_fits = 0
            self._rng = random.Random(self.config.seed)
        self._publish(DIFFUSER_CLEARED, {})

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Round-trippable JSON dict capturing config + per-target counts.

        Score / classifier / kernel callables are NOT serialised; the
        caller must re-register them on the imported instance.
        """
        with self._lock:
            return {
                "version": "agi.diffuser.v1",
                "config": {
                    "dim": self.config.dim,
                    "T": self.config.T,
                    "schedule_kind": self.config.schedule_kind,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "karras_sigma_min": self.config.karras_sigma_min,
                    "karras_sigma_max": self.config.karras_sigma_max,
                    "karras_rho": self.config.karras_rho,
                    "cosine_s": self.config.cosine_s,
                    "seed": self.config.seed,
                    "girsanov_mu": self.config.girsanov_mu,
                    "second_moment_default": self.config.second_moment_default,
                    "histogram_bins": self.config.histogram_bins,
                    "tv_confidence": self.config.tv_confidence,
                },
                "targets": {
                    tid: {
                        "dim": t.dim,
                        "second_moment": t.second_moment,
                        "score_error_floor": t.score_error_floor,
                        "chain_head": t.chain_head,
                        "fit_W": [row[:] for row in t.fit_W],
                    }
                    for tid, t in self._targets.items()
                },
                "n_steps": self._n_steps,
                "n_samples": self._n_samples,
                "n_certifications": self._n_certifications,
                "n_fits": self._n_fits,
            }

    @classmethod
    def import_(
        cls,
        blob: dict[str, Any],
        *,
        publisher: EventPublisher | None = None,
    ) -> "Diffuser":
        if blob.get("version") != "agi.diffuser.v1":
            raise InvalidConfig(f"unsupported export version {blob.get('version')!r}")
        cfg = blob["config"]
        d = cls(
            DiffuserConfig(
                dim=cfg["dim"],
                T=cfg["T"],
                schedule_kind=cfg["schedule_kind"],
                beta_start=cfg["beta_start"],
                beta_end=cfg["beta_end"],
                karras_sigma_min=cfg["karras_sigma_min"],
                karras_sigma_max=cfg["karras_sigma_max"],
                karras_rho=cfg["karras_rho"],
                cosine_s=cfg["cosine_s"],
                seed=cfg["seed"],
                girsanov_mu=cfg["girsanov_mu"],
                second_moment_default=cfg["second_moment_default"],
                histogram_bins=cfg["histogram_bins"],
                tv_confidence=cfg["tv_confidence"],
            ),
            publisher=publisher,
        )
        # Targets are skeleton-only after import; user must re-register
        # callables.  We still seed metadata so chain heads survive.
        with d._lock:
            for tid, tblob in blob.get("targets", {}).items():
                d._targets[tid] = _Target(
                    target_id=tid,
                    dim=tblob["dim"],
                    score_fn=_identity_score,
                    classifier_grad_fn=None,
                    cond_score_fn=None,
                    uncond_score_fn=None,
                    consistency_fn=None,
                    flow_vector_fn=None,
                    d3pm_K=None,
                    d3pm_kernel=None,
                    d3pm_x0_proposal=None,
                    second_moment=tblob["second_moment"],
                    score_error_floor=tblob["score_error_floor"],
                    fit_W=[row[:] for row in tblob.get("fit_W", [])],
                    chain_head=tblob["chain_head"],
                )
            d._n_steps = blob.get("n_steps", 0)
            d._n_samples = blob.get("n_samples", 0)
            d._n_certifications = blob.get("n_certifications", 0)
            d._n_fits = blob.get("n_fits", 0)
        return d
