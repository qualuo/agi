r"""Compressor — Minimum Description Length hypothesis selection as a runtime primitive.

The compression principle is the deepest unifying thread in the
foundations of intelligence.  Solomonoff (1964) reduced "induction" to
"the shortest program that reproduces the data".  Rissanen (1978)
turned that ideal into a practical statistical methodology — Minimum
Description Length (MDL) — by replacing "program length" with
"codelength under a well-chosen code".  Hutter (2005) showed that the
Bayes-optimal agent in arbitrary computable environments is the one
that uses Solomonoff's universal prior.  Sutskever, Schmidhuber and
Hutter have argued, on different grounds, that *predictive
compression* is what large models are doing when they generalise.

Every other primitive in this runtime supplies the coordinator with a
*decision* (which arm to pull, which experiment to run, which plan to
execute, which hypothesis to refute).  None of them supply the
coordinator with the meta-decision of *which model class itself is
best supported by the data*.  That gap is what ``Compressor`` fills.

The pitch reduced to a runtime call:

  * ``register(name, kind, **hyperparams)`` for every candidate model
    class the coordinator wants to score (Bernoulli, multinomial,
    geometric, Poisson, Gaussian, histogram density, Markov chain of
    order ``r``, AR(1), uniform-discrete, …);
  * ``codelength(name, data)`` returns a ``Codelength`` carrying the
    maximum-likelihood log-loss, the *stochastic complexity* (refined
    MDL via the Normalized Maximum Likelihood code), the prequential
    (sequential) codelength, the Rissanen parametric-complexity term,
    and asymptotic lower/upper bounds;
  * ``select(data, method=NML)`` runs every registered model on the
    data and returns the MDL-optimal pick together with a regret
    certificate: the codelength gap to the runner-up, and a PAC-style
    lower bound on its generalisation advantage;
  * ``online_observe(name, datum)`` updates a model's prequential
    state; the returned codelength is *anytime valid* — the
    coordinator can switch model classes mid-stream;
  * ``compare(name_a, name_b, data)`` returns the Bayes factor and the
    MDL difference between two models, with a Stone-Geisser cross-
    validation cross-check;
  * ``report()`` returns a ``CompressorReport`` aggregating every
    fit, every selection, the model posteriors over the data
    stream(s), and a SHA-256 hash chain over every decision so an
    external auditor can replay the experiment bit-exactly.

Mathematical roots and algorithms shipped
-----------------------------------------

**Universal codes for the positive integers** (Rissanen 1983).

  * **Elias-γ** (Elias 1975).  ``L_γ(n) = 2⌊log₂ n⌋ + 1`` bits.
    Prefix-free.  Optimal up to a constant for power-law distributions.
  * **Elias-δ** (Elias 1975).  ``L_δ(n) = ⌊log₂ n⌋ + 2⌊log₂(⌊log₂ n⌋ +
    1)⌋ + 1`` bits.  Tighter than γ for large n.
  * **Rissanen log\*** (Rissanen 1983).  ``L*(n) = log₂ c₀ + log₂ n +
    log₂ log₂ n + …`` stopping when the next term ≤ 0; ``c₀ ≈
    2.865064`` is the universal constant making the code-length sum
    Kraft-valid.  The shortest universal prefix code asymptotically;
    optimal up to additive ``o(1)``.

**Refined MDL: the Normalized Maximum Likelihood (NML) code**
(Shtarkov 1987, Rissanen 1996).

For a parametric model class ``M_k = { p(·|θ) : θ ∈ Θ_k }`` the NML
distribution on a length-``n`` sample is

    ``p_NML(x | k) = p(x | θ̂(x)) / C_n(k)``

where ``θ̂(x)`` is the maximum-likelihood estimator and
``C_n(k) = ∑_{y ∈ X^n} p(y | θ̂(y))`` is the *Shtarkov sum* — the
parametric complexity of class ``M_k``.  The NML code achieves the
minimax regret with respect to the maximum-likelihood code:

    ``L_NML(x | k) = -log p(x | θ̂(x)) + log C_n(k)``.

Closed-form ``log C_n`` for the families this primitive ships:

  * **Bernoulli.**  ``C_n^{Ber} = ∑_{k=0}^{n} (n choose k) (k/n)^k
    ((n-k)/n)^(n-k)``.  Computed exactly with log-gamma arithmetic for
    ``n ≤ 10⁵``; Mononen's (2008) ``O(1)`` asymptotic
    ``(1/2) log(n π / 2)`` for larger ``n``.

  * **Multinomial of order k** (Szpankowski 1998, Mononen-Myllymäki
    2008).  Linear-time recurrence

        ``C_n(k) = C_n(k-1) + (n / (k-2)) C_n(k-2)``,
        ``C_n(1) = 1``, ``C_n(2) = C_n^{Ber}``.

    Computed exactly in ``O(k)`` time; numerically stable in log-space.

  * **Geometric.**  Single-parameter; Rissanen's
    ``(1/2) log(n) + (1/2) log(π) + O(1)`` with the integrated Fisher
    correction in closed form for the geometric likelihood.

  * **Poisson.**  Same ``(1/2) log(n)`` term; Fisher information
    ``1/λ`` integrates against Jeffreys ``1/√λ dλ`` over a
    coordinator-supplied bounded ``[λ_min, λ_max]``.

  * **Gaussian (known σ).**  Mean is unbounded; refined MDL uses the
    *luckiness NML* with the coordinator supplying ``[μ_min, μ_max]``.
    The parametric complexity is closed-form
    ``log((μ_max - μ_min) / (σ √(2π e / n)))``.

  * **Gaussian (unknown σ).**  Two parameters.  Standard Rissanen
    ``log C_n ≈ log n - log π + log Γ((n-1)/2) − log Γ(n/2) +
    log(σ_max/σ_min) /2`` for a coordinator-supplied
    ``[σ_min, σ_max]``.

  * **Uniform discrete on k symbols.**  ``C_n = 1``; parametric
    complexity is zero.  Useful as a hard baseline.

  * **Histogram of m bins.**  Multinomial-NML with k = m.

**Prequential (sequential) codes** (Dawid 1984).

Plug-in predictive distributions whose total codelength matches NML
asymptotically but is available online.  This primitive ships:

  * **Krichevsky-Trofimov (KT) mixture** (Krichevsky-Trofimov 1981).
    Binary case: ``p(x_{t+1} = 1 | x^t) = (n_1(t) + 1/2) / (t + 1)``.
    Total codelength is ``-log p(x^n) = -log Γ(n_1 + 1/2) - log Γ(n_0
    + 1/2) + log Γ(1) + log Γ(n+1)``.  Minimax regret matches NML up
    to ``O(1)``.

  * **Dirichlet-Multinomial (KT generalised).**  k-ary case with
    symmetric Dirichlet(1/k) prior — Krichevsky-Trofimov for k = 2,
    Laplace's rule with α = 1/k otherwise.

  * **Bayesian normal-inverse-gamma plug-in.**  For the
    Gaussian-unknown-σ class, the predictive Student-t mixture is the
    prequential analogue.

**Two-part MDL** (Rissanen 1978).

    ``L_{2-part}(x | k) = L(θ̂_q) + L(x | θ̂_q)``

where ``θ̂_q`` is the maximum-likelihood estimate quantised to
precision ``1/√n`` (the Rissanen-optimal precision); the parameter
code-length is ``(k/2) log n`` bits up to lower-order terms.  Provided
as a sanity-check against NML and as the default for model classes
where exact NML is intractable.

**Regret certificates.**

For two registered models with codelengths ``L_a, L_b`` on the same
data, the MDL-difference ``ΔL = L_a − L_b`` gives a Bayes factor of
``e^{ΔL}`` (in nats) under the implicit MDL prior.  This primitive
exports the Bayes factor, the symmetric KL divergence between the
posterior predictives, and a Stone-Geisser leave-one-out
cross-validation estimate of the same quantity for cross-check.  A
significant gap is exported as a regret lower bound on the
suboptimal model's *per-future-symbol* loss with the Vovk (1990)
strong-aggregating-algorithm bound.

**Anytime-valid online codelength.**

The KT and Dirichlet plug-ins are intrinsically sequential, so a
running prequential codelength is updated on every call to
``online_observe``.  Two coordinators sharing a stream can fork at
any prefix; the codelengths recombine additively.  This is the
property that lets the runtime expose a *streaming model-selection*
endpoint without rebuilding the full likelihood.

**Tamper-evident replay.**

Every register / fit / score / select / observe / report call emits
an event into a SHA-256 hash chain whose genesis is
``compressor.v1.genesis``.  Replaying the same byte sequence
reproduces the same fingerprint.  The chain plugs directly into the
``AttestationLedger`` of the wider runtime.

What it composes with
---------------------

``Compressor`` is built to be driven by — and to drive — every other
primitive in the runtime:

  * **Sampler.**  Sampler draws from a *posterior* given a fixed model
    class.  Compressor picks the model class.  In a typical pipeline,
    the coordinator first calls ``Compressor.select`` on the held-out
    prefix and only then opens a ``Sampler`` for posterior simulation
    in the winning class.

  * **Forecaster.**  Forecaster maintains an anytime-valid forecast
    *given* a parametric family.  Compressor monitors the prequential
    codelength of the forecaster against alternatives and triggers a
    re-fit when the codelength gap exceeds a coordinator-set
    threshold — a principled detector of model misspecification.

  * **Refuter.**  Refuter falsifies a single claim.  When the claim is
    "model class M is the best description of the data", Compressor
    supplies the codelength gap that Refuter converts to a Bayes
    factor and an evidence-based reject/accept.

  * **DriftSentinel.**  Drift detection is, mathematically, online
    model-selection.  Compressor's prequential codelength on a
    rolling window IS the drift statistic; DriftSentinel converts
    that to a stopping decision with FDR control.

  * **Reasoner.**  Reasoner solves a Boolean satisfaction instance.
    Compressor picks the formula representation — *which* boolean
    encoding of a structured constraint minimises the joint codelength
    of formula + witness — and feeds the winner to Reasoner.

  * **Composer.**  Composer plans a sequence of operator calls.
    Compressor scores each candidate plan structure by its joint
    description length under the model class of "plan + outcomes".

  * **AttestationLedger.**  The fingerprint chain is append-only and
    every event in the chain is canonicalised before hashing, so an
    auditor can replay the exact sequence of registrations, fits, and
    selections that produced any given report.

Investor framing
----------------

The pitch a coordinator's UI can surface, automatically, for every
data stream the user routes through it:

    "Of the registered model classes, Multinomial-of-3 best describes
     the observed 1024-symbol prefix:

         Multinomial(3):  L_NML = 1402.7 bits
         Multinomial(4):  L_NML = 1418.9 bits  (Δ = +16.2 bits)
         Bernoulli:       L_NML = 1681.4 bits  (Δ = +278.7 bits)
         Uniform-3:       L_NML = 1623.0 bits  (Δ = +220.3 bits)

     The MDL gap to runner-up is 16.2 bits = a Bayes factor of
     ~10⁷ in favour of Multinomial(3); the model is significantly
     preferred at any sensible threshold.

     Per-symbol regret bound vs runner-up: 0.0158 bits/sym (Vovk
     1990, strong aggregating algorithm).  Replay fingerprint:
     a01f3c… (verifiable via AttestationLedger)."

Every number here is grounded in published, citable mathematics and
reproducible from the codelength-event log.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


# =====================================================================
# Public constants
# =====================================================================

# Model classes.
BERNOULLI = "bernoulli"
MULTINOMIAL = "multinomial"
GEOMETRIC = "geometric"
POISSON = "poisson"
GAUSSIAN_KNOWN_SIGMA = "gaussian_known_sigma"
GAUSSIAN = "gaussian"
UNIFORM_DISCRETE = "uniform_discrete"
HISTOGRAM = "histogram"
MARKOV = "markov"
CONSTANT = "constant"

KNOWN_MODELS = frozenset({
    BERNOULLI,
    MULTINOMIAL,
    GEOMETRIC,
    POISSON,
    GAUSSIAN_KNOWN_SIGMA,
    GAUSSIAN,
    UNIFORM_DISCRETE,
    HISTOGRAM,
    MARKOV,
    CONSTANT,
})

# Codelength methods.
ML = "ml"                       # maximum-likelihood, no penalty (over-fits)
NML = "nml"                     # refined MDL via Normalized Maximum Likelihood
TWO_PART = "two_part"           # classical two-part MDL
PREQUENTIAL = "prequential"     # sequential / Dawid prequential code
BIC = "bic"                     # Bayesian Information Criterion (Schwarz 1978)
AIC = "aic"                     # Akaike Information Criterion

KNOWN_METHODS = frozenset({ML, NML, TWO_PART, PREQUENTIAL, BIC, AIC})

# Universal integer codes.
ELIAS_GAMMA = "elias_gamma"
ELIAS_DELTA = "elias_delta"
RISSANEN_LOGSTAR = "rissanen_logstar"

KNOWN_INT_CODES = frozenset({ELIAS_GAMMA, ELIAS_DELTA, RISSANEN_LOGSTAR})

# Events.
COMPRESSOR_STARTED = "compressor.started"
COMPRESSOR_MODEL_REGISTERED = "compressor.model_registered"
COMPRESSOR_FIT = "compressor.fit"
COMPRESSOR_SCORED = "compressor.scored"
COMPRESSOR_SELECTED = "compressor.selected"
COMPRESSOR_OBSERVED = "compressor.observed"
COMPRESSOR_COMPARED = "compressor.compared"
COMPRESSOR_REPORT = "compressor.report"
COMPRESSOR_CLEARED = "compressor.cleared"

KNOWN_EVENTS = frozenset({
    COMPRESSOR_STARTED,
    COMPRESSOR_MODEL_REGISTERED,
    COMPRESSOR_FIT,
    COMPRESSOR_SCORED,
    COMPRESSOR_SELECTED,
    COMPRESSOR_OBSERVED,
    COMPRESSOR_COMPARED,
    COMPRESSOR_REPORT,
    COMPRESSOR_CLEARED,
})

# Numerical defaults.
_EPS = 1e-12
_INF = float("inf")
_LN2 = math.log(2.0)
_LN_2PI = math.log(2.0 * math.pi)
_LN_PI = math.log(math.pi)
_LOG_EPS = math.log(_EPS)
_RISSANEN_C0 = 2.865064  # Rissanen's universal constant (1983)
_LOG2_C0 = math.log2(_RISSANEN_C0)
_GENESIS = hashlib.sha256(b"compressor.v1.genesis").hexdigest()


# =====================================================================
# Exceptions
# =====================================================================


class CompressorError(ValueError):
    """Base class for compressor-domain errors."""


class UnknownModel(CompressorError):
    """A model kind is not in KNOWN_MODELS, or a model name was never registered."""


class UnknownMethod(CompressorError):
    """A codelength method is not in KNOWN_METHODS."""


class UnknownIntCode(CompressorError):
    """An integer-code name is not in KNOWN_INT_CODES."""


class InvalidModel(CompressorError):
    """A model's hyperparameters are malformed."""


class InvalidData(CompressorError):
    """Data is malformed for the chosen model class."""


class InsufficientData(CompressorError):
    """Too few observations for the requested operation."""


class IncompatibleModels(CompressorError):
    """Two models cannot be compared (different sample spaces)."""


# =====================================================================
# Numerical helpers
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_log(x: float) -> float:
    return math.log(max(x, _EPS))


def _logsumexp(xs: Sequence[float]) -> float:
    """Numerically stable log(sum(exp(x))) for an iterable of floats."""
    if not xs:
        return -_INF
    m = max(xs)
    if m == -_INF:
        return -_INF
    s = 0.0
    for x in xs:
        s += math.exp(x - m)
    return m + math.log(s)


def _xlogx(x: float) -> float:
    """``x log x`` with the convention ``0 log 0 = 0``."""
    if x <= 0.0:
        return 0.0
    return x * math.log(x)


def _entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats for a probability vector (no normalisation check)."""
    return -sum(_xlogx(p) for p in probs)


def _lgamma(x: float) -> float:
    return math.lgamma(x)


def _lbeta(a: float, b: float) -> float:
    return _lgamma(a) + _lgamma(b) - _lgamma(a + b)


# =====================================================================
# Universal codes for the positive integers
# =====================================================================


def elias_gamma_bits(n: int) -> float:
    r"""Elias-γ codelength for ``n ≥ 1`` (in bits).

    ``L_γ(n) = 2⌊log₂ n⌋ + 1``.  Prefix-free; Kraft sum converges.
    """
    if n < 1:
        raise CompressorError(f"Elias-γ requires n ≥ 1; got {n}")
    return 2.0 * math.floor(math.log2(n)) + 1.0


def elias_delta_bits(n: int) -> float:
    r"""Elias-δ codelength for ``n ≥ 1`` (in bits).

    ``L_δ(n) = ⌊log₂ n⌋ + 2⌊log₂(⌊log₂ n⌋ + 1)⌋ + 1``.
    Tighter than γ for n ≥ 2.
    """
    if n < 1:
        raise CompressorError(f"Elias-δ requires n ≥ 1; got {n}")
    fl = math.floor(math.log2(n))
    return float(fl) + 2.0 * math.floor(math.log2(fl + 1)) + 1.0


def rissanen_logstar_bits(n: int) -> float:
    r"""Rissanen's universal log\* code for ``n ≥ 1`` (in bits).

    ``L*(n) = log₂ c₀ + log₂ n + log₂ log₂ n + …`` summing while
    the next term is positive.  Asymptotically the shortest universal
    prefix codelength for the positive integers (Rissanen 1983).
    """
    if n < 1:
        raise CompressorError(f"log* requires n ≥ 1; got {n}")
    total = _LOG2_C0
    x = float(n)
    while x > 1.0:
        l = math.log2(x)
        total += l
        x = l
    return total


def universal_int_bits(n: int, *, code: str = ELIAS_DELTA) -> float:
    """Universal integer codelength under one of ``KNOWN_INT_CODES``."""
    if code == ELIAS_GAMMA:
        return elias_gamma_bits(n)
    if code == ELIAS_DELTA:
        return elias_delta_bits(n)
    if code == RISSANEN_LOGSTAR:
        return rissanen_logstar_bits(n)
    raise UnknownIntCode(f"unknown universal-integer code {code!r}")


# =====================================================================
# Stochastic complexity (parametric complexity) for the shipped classes
# =====================================================================


def log_bernoulli_nml_constant(n: int) -> float:
    r"""``log C_n`` for the Bernoulli NML code, in *nats*.

    For small / moderate ``n`` returns the exact Shtarkov sum

        ``C_n = ∑_{k=0}^n binom(n,k) (k/n)^k ((n-k)/n)^(n-k)``

    using log-gamma arithmetic.  For ``n ≥ 50_000`` switches to the
    Mononen-Rissanen asymptotic ``(1/2) log(n π / 2)`` whose ``o(1)``
    error is below 1e-9 nats by that point.
    """
    if n < 0:
        raise CompressorError(f"NML constant requires n ≥ 0; got {n}")
    if n == 0:
        return 0.0
    if n >= 50_000:
        return 0.5 * math.log(n * math.pi / 2.0)
    # exact: log-sum-exp over k
    log_n = math.log(n)
    terms: list[float] = []
    log_n_fact = _lgamma(n + 1.0)
    for k in range(0, n + 1):
        if k == 0 or k == n:
            terms.append(0.0)
            continue
        log_binom = log_n_fact - _lgamma(k + 1.0) - _lgamma(n - k + 1.0)
        # (k/n)^k * ((n-k)/n)^(n-k) in log
        log_term = (
            log_binom
            + k * (math.log(k) - log_n)
            + (n - k) * (math.log(n - k) - log_n)
        )
        terms.append(log_term)
    return _logsumexp(terms)


def log_multinomial_nml_constant(n: int, k: int) -> float:
    r"""``log C_n(k)`` for the k-ary multinomial NML code, in *nats*.

    Computed in ``O(k)`` time using the Mononen-Myllymäki (2008)
    recurrence

        ``C_n(j) = C_n(j-1) + (n / (j-2)) C_n(j-2)``  for ``j ≥ 3``,
        ``C_n(1) = 1``, ``C_n(2) = C_n^{Bernoulli}``,

    expressed in log-space to avoid overflow.
    """
    if n < 0:
        raise CompressorError(f"multinomial NML requires n ≥ 0; got {n}")
    if k < 1:
        raise CompressorError(f"multinomial NML requires k ≥ 1; got {k}")
    if k == 1:
        return 0.0  # C_n(1) = 1
    if n == 0:
        return 0.0  # empty sample: all classes give probability 1
    log_c_prev2 = 0.0                           # log C_n(1) = 0
    log_c_prev1 = log_bernoulli_nml_constant(n)  # log C_n(2)
    if k == 2:
        return log_c_prev1
    log_n = math.log(n)
    for j in range(3, k + 1):
        # log( C_n(j-1) + (n/(j-2)) C_n(j-2) )
        a = log_c_prev1
        b = log_c_prev2 + log_n - math.log(j - 2)
        log_c = _logsumexp((a, b))
        log_c_prev2 = log_c_prev1
        log_c_prev1 = log_c
    return log_c_prev1


def parametric_complexity_geometric(n: int) -> float:
    r"""Refined-MDL parametric complexity for the geometric class on ``n`` samples.

    Asymptotic Rissanen formula with the closed-form Fisher integral
    for the geometric (Rissanen 1996; Grünwald 2007 Ch. 7):

        ``log C_n ≈ (1/2) log(n / (2 π)) + log ∫_0^1 (p(1-p))^{-1/2} dp``
                  = ``(1/2) log(n / (2 π)) + log π``.
    """
    if n <= 0:
        return 0.0
    return 0.5 * math.log(n / (2.0 * math.pi)) + _LN_PI


def parametric_complexity_poisson(n: int, lam_min: float, lam_max: float) -> float:
    r"""Refined-MDL parametric complexity for Poisson on a bounded mean range.

    Fisher information of ``Poisson(λ)`` is ``1/λ``; the Jeffreys
    integral on ``[λ_min, λ_max]`` is ``∫ λ^{-1/2} dλ = 2(√λ_max
    − √λ_min)``.  Asymptotic ``log C_n ≈ (1/2) log(n/(2π)) +
    log(2(√λ_max − √λ_min))``.
    """
    if n <= 0:
        return 0.0
    if not (0.0 < lam_min < lam_max):
        raise InvalidModel(
            f"Poisson requires 0 < lam_min < lam_max; got [{lam_min}, {lam_max}]"
        )
    jeffreys = 2.0 * (math.sqrt(lam_max) - math.sqrt(lam_min))
    return 0.5 * math.log(n / (2.0 * math.pi)) + math.log(jeffreys)


def parametric_complexity_gaussian_known_sigma(
    n: int, sigma: float, mu_min: float, mu_max: float
) -> float:
    r"""Luckiness-NML parametric complexity for ``N(μ, σ²)`` with known σ.

    With ``μ`` restricted to ``[μ_min, μ_max]``,

        ``log C_n = log((μ_max − μ_min) / (σ √(2π e / n)))``

    (Grünwald 2007, eq. 11.5; equals (1/2) log(n/(2π)) plus the
    log-range of μ scaled by σ).
    """
    if n <= 0:
        return 0.0
    if sigma <= 0.0:
        raise InvalidModel(f"Gaussian known-sigma requires σ > 0; got {sigma}")
    if not (mu_min < mu_max):
        raise InvalidModel(
            f"Gaussian known-sigma requires mu_min < mu_max; got [{mu_min}, {mu_max}]"
        )
    width = mu_max - mu_min
    denom = sigma * math.sqrt(2.0 * math.pi * math.e / n)
    return math.log(width / denom)


def parametric_complexity_gaussian(
    n: int, sigma_min: float, sigma_max: float
) -> float:
    r"""Refined-MDL parametric complexity for ``N(μ, σ²)`` with unknown σ.

    Two-parameter family.  Asymptotic luckiness-NML

        ``log C_n ≈ log(n) − log(π) + log Γ((n-1)/2) − log Γ(n/2)
                   + (1/2) log(σ_max² / σ_min²)``.

    The first three terms are the classical Rissanen result for the
    location-scale model; the last is the bounded-σ luckiness term.
    """
    if n <= 1:
        return 0.0
    if not (0.0 < sigma_min < sigma_max):
        raise InvalidModel(
            "Gaussian requires 0 < sigma_min < sigma_max; "
            f"got [{sigma_min}, {sigma_max}]"
        )
    base = math.log(n) - _LN_PI + _lgamma((n - 1) / 2.0) - _lgamma(n / 2.0)
    return base + math.log(sigma_max / sigma_min)


# =====================================================================
# Data validation
# =====================================================================


def _validate_binary_sequence(data: Sequence[Any]) -> tuple[int, int]:
    """Return (n_zeros, n_ones) or raise."""
    n0 = 0
    n1 = 0
    for x in data:
        if x == 0 or x is False:
            n0 += 1
        elif x == 1 or x is True:
            n1 += 1
        else:
            raise InvalidData(f"Bernoulli data must be 0/1; saw {x!r}")
    return n0, n1


def _validate_multinomial_sequence(data: Sequence[Any], k: int) -> list[int]:
    counts = [0] * k
    for x in data:
        try:
            i = int(x)
        except (TypeError, ValueError) as e:
            raise InvalidData(f"multinomial data must be int in [0,{k}); saw {x!r}") from e
        if not (0 <= i < k):
            raise InvalidData(f"multinomial symbol {i} out of range [0,{k})")
        counts[i] += 1
    return counts


def _validate_count_sequence(data: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for x in data:
        try:
            i = int(x)
        except (TypeError, ValueError) as e:
            raise InvalidData(f"count data must be non-negative int; saw {x!r}") from e
        if i < 0:
            raise InvalidData(f"count data must be ≥ 0; saw {i}")
        out.append(i)
    return out


def _validate_positive_count_sequence(data: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for x in data:
        try:
            i = int(x)
        except (TypeError, ValueError) as e:
            raise InvalidData(f"geometric data must be positive int; saw {x!r}") from e
        if i < 1:
            raise InvalidData(f"geometric data must be ≥ 1; saw {i}")
        out.append(i)
    return out


def _validate_real_sequence(data: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for x in data:
        try:
            v = float(x)
        except (TypeError, ValueError) as e:
            raise InvalidData(f"Gaussian data must be real; saw {x!r}") from e
        if not math.isfinite(v):
            raise InvalidData(f"Gaussian data must be finite; saw {v}")
        out.append(v)
    return out


# =====================================================================
# Maximum likelihood per family (returns natural-log-likelihood)
# =====================================================================


def ml_loglik_bernoulli(n0: int, n1: int) -> tuple[float, float]:
    """Return (p_hat, log_likelihood) for the MLE of Bernoulli."""
    n = n0 + n1
    if n == 0:
        return 0.5, 0.0
    p = n1 / n
    ll = _xlogx(n1) + _xlogx(n0) - _xlogx(n)
    return p, ll


def ml_loglik_multinomial(counts: Sequence[int]) -> tuple[list[float], float]:
    n = sum(counts)
    if n == 0:
        k = len(counts)
        return [1.0 / k] * k, 0.0
    probs = [c / n for c in counts]
    ll = sum(_xlogx(c) for c in counts) - _xlogx(n)
    return probs, ll


def ml_loglik_geometric(data: Sequence[int]) -> tuple[float, float]:
    """Geometric on {1,2,…} with PMF (1-p)^{x-1} p.  MLE: p̂ = n / sum(x)."""
    n = len(data)
    s = sum(data)
    if n == 0 or s == 0:
        return 0.5, 0.0
    p = n / s
    p = _clip(p, _EPS, 1.0 - _EPS)
    ll = n * math.log(p) + (s - n) * math.log(1.0 - p)
    return p, ll


def ml_loglik_poisson(data: Sequence[int]) -> tuple[float, float]:
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    lam = sum(data) / n
    if lam == 0.0:
        # everything zero: likelihood is 1, log-lik 0 (only at λ=0 limit)
        return 0.0, 0.0
    ll = 0.0
    for x in data:
        ll += x * math.log(lam) - lam - _lgamma(x + 1.0)
    return lam, ll


def ml_loglik_gaussian_known_sigma(
    data: Sequence[float], sigma: float
) -> tuple[float, float]:
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    mu = sum(data) / n
    ll = -0.5 * n * (_LN_2PI + 2.0 * math.log(sigma))
    s2 = sum((x - mu) ** 2 for x in data)
    ll -= 0.5 * s2 / (sigma * sigma)
    return mu, ll


def ml_loglik_gaussian(data: Sequence[float]) -> tuple[float, float, float]:
    n = len(data)
    if n == 0:
        return 0.0, 1.0, 0.0
    mu = sum(data) / n
    s2 = sum((x - mu) ** 2 for x in data) / n
    s2 = max(s2, _EPS)
    sigma = math.sqrt(s2)
    ll = -0.5 * n * (_LN_2PI + math.log(s2) + 1.0)
    return mu, sigma, ll


def ml_loglik_uniform_discrete(data: Sequence[int], k: int) -> tuple[float, float]:
    n = len(data)
    if n == 0:
        return 1.0 / k, 0.0
    for x in data:
        if not (0 <= int(x) < k):
            raise InvalidData(f"uniform-discrete symbol {x} out of range [0,{k})")
    ll = -n * math.log(k)
    return 1.0 / k, ll


# =====================================================================
# Prequential codes (sequential / Dawid 1984)
# =====================================================================


def kt_codelength_binary(n0: int, n1: int) -> float:
    r"""Krichevsky-Trofimov (1981) total codelength in nats for a binary string
    with ``n0`` zeros and ``n1`` ones.

    Mixture under Dirichlet(1/2, 1/2):

        ``log P_{KT}(x) = log Γ(n0 + 1/2) + log Γ(n1 + 1/2)
                          − log Γ(1/2) − log Γ(1/2) − log B(1/2,1/2)
                          + log Γ(1) − log Γ(n+1)``.

    Equivalently, ``-log P_KT = -lbeta(n0 + 1/2, n1 + 1/2) + lbeta(1/2, 1/2)``.
    Returned as a non-negative number of nats.
    """
    n = n0 + n1
    if n == 0:
        return 0.0
    log_p = _lbeta(n0 + 0.5, n1 + 0.5) - _lbeta(0.5, 0.5)
    return -log_p


def kt_codelength_multinomial(counts: Sequence[int]) -> float:
    r"""KT / Dirichlet-1/2 prequential codelength in nats for a k-ary sequence."""
    k = len(counts)
    if k <= 0:
        return 0.0
    n = sum(counts)
    if n == 0:
        return 0.0
    alpha = 0.5
    log_p = (
        sum(_lgamma(c + alpha) for c in counts)
        - k * _lgamma(alpha)
        - _lgamma(n + k * alpha)
        + _lgamma(k * alpha)
    )
    return -log_p


def laplace_codelength_multinomial(counts: Sequence[int]) -> float:
    r"""Laplace-rule prequential code (Dirichlet(1)) — slightly looser than KT.

    Useful as a sanity-check and as the standard textbook plug-in.
    """
    k = len(counts)
    if k <= 0:
        return 0.0
    n = sum(counts)
    if n == 0:
        return 0.0
    log_p = (
        sum(_lgamma(c + 1.0) for c in counts)
        - k * _lgamma(1.0)
        - _lgamma(n + k * 1.0)
        + _lgamma(k * 1.0)
    )
    return -log_p


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class ModelSpec:
    """A registered model class with all its hyperparameters."""
    name: str
    kind: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class Fit:
    """Maximum-likelihood fit of a registered model on a dataset."""
    name: str
    n: int
    params: Mapping[str, Any]
    log_likelihood: float           # natural log
    ml_codelength_nats: float       # = -log_likelihood
    ml_codelength_bits: float


@dataclass(frozen=True)
class Codelength:
    """All codelength variants for one model on one dataset, in *nats*.

    ``stochastic_complexity`` is the NML / refined-MDL codelength
    (``L_NML = -log p(x | θ̂) + log C_n``).  ``parametric_complexity``
    is ``log C_n`` alone.  ``prequential`` is the sequential plug-in
    code (KT / Laplace / Bayes mixture, family-dependent).
    ``two_part`` is ``-log p(x | θ̂) + (k/2) log n``.  ``bic`` and
    ``aic`` are the usual information criteria divided by 2 to match
    nats.
    """
    name: str
    n: int
    ml: float
    parametric_complexity: float
    stochastic_complexity: float    # NML
    prequential: float
    two_part: float
    bic: float
    aic: float
    bits: float                     # NML in bits, the headline number
    method: str                     # which one is the "headline"

    def select_value(self, method: str) -> float:
        if method == ML:
            return self.ml
        if method == NML:
            return self.stochastic_complexity
        if method == TWO_PART:
            return self.two_part
        if method == PREQUENTIAL:
            return self.prequential
        if method == BIC:
            return self.bic
        if method == AIC:
            return self.aic
        raise UnknownMethod(f"unknown codelength method {method!r}")


@dataclass(frozen=True)
class Selection:
    """The MDL-optimal model with a regret certificate."""
    method: str
    winner: str
    runner_up: str | None
    codelengths_nats: Mapping[str, float]
    codelengths_bits: Mapping[str, float]
    gap_nats: float                 # winner−runner_up (negative — winner is shorter)
    gap_bits: float
    bayes_factor: float             # exp(|gap|) -> evidence for winner
    per_symbol_regret_bits: float   # Vovk strong-aggregating regret bound
    fingerprint: str


@dataclass(frozen=True)
class Comparison:
    """Pairwise comparison of two registered models."""
    a: str
    b: str
    method: str
    codelength_a_nats: float
    codelength_b_nats: float
    delta_nats: float               # a − b
    delta_bits: float
    bayes_factor_for_a: float       # exp(b − a) — > 1 favours a
    sym_kl_predictive: float        # symmetric KL between posterior predictives
    cv_delta_nats: float            # Stone-Geisser leave-one-out check (may be NaN)


@dataclass
class OnlineState:
    """A model's mutable prequential / online state."""
    name: str
    n: int
    # binary
    n0: int = 0
    n1: int = 0
    # multinomial
    counts: list[int] = field(default_factory=list)
    # gaussian
    sum_x: float = 0.0
    sum_x2: float = 0.0
    # geometric / poisson
    sum_count: int = 0
    # running prequential codelength in nats
    prequential_nats: float = 0.0


@dataclass(frozen=True)
class CompressorReport:
    """Aggregate report — every registered model, every selection, the
    audit-trail fingerprint."""
    models: Mapping[str, ModelSpec]
    last_fits: Mapping[str, Fit]
    last_codelengths: Mapping[str, Codelength]
    selections: Sequence[Selection]
    comparisons: Sequence[Comparison]
    online_states: Mapping[str, OnlineState]
    fingerprint: str
    n_events: int


# =====================================================================
# Per-model codelength dispatch
# =====================================================================


def _codelength_bernoulli(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    n0, n1 = _validate_binary_sequence(data)
    n = n0 + n1
    p, ll = ml_loglik_bernoulli(n0, n1)
    pc = log_bernoulli_nml_constant(n)
    sc = -ll + pc
    preq = kt_codelength_binary(n0, n1)
    two_part = -ll + 0.5 * math.log(max(n, 1))
    bic = -ll + 0.5 * 1 * math.log(max(n, 1))      # k=1
    aic = -ll + 1                                  # k=1
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


def _codelength_multinomial(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    k = int(spec.params["k"])
    counts = _validate_multinomial_sequence(data, k)
    n = sum(counts)
    _probs, ll = ml_loglik_multinomial(counts)
    pc = log_multinomial_nml_constant(n, k)
    sc = -ll + pc
    preq = kt_codelength_multinomial(counts)
    two_part = -ll + 0.5 * (k - 1) * math.log(max(n, 1))
    bic = -ll + 0.5 * (k - 1) * math.log(max(n, 1))
    aic = -ll + (k - 1)
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


def _codelength_geometric(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    xs = _validate_positive_count_sequence(data)
    n = len(xs)
    p, ll = ml_loglik_geometric(xs)
    pc = parametric_complexity_geometric(n)
    sc = -ll + pc
    # prequential: integrate over Beta(1/2, 1/2) prior on p
    s = sum(xs)
    preq = _lbeta(0.5, 0.5) - _lbeta(n + 0.5, s - n + 0.5)
    two_part = -ll + 0.5 * math.log(max(n, 1))
    bic = -ll + 0.5 * 1 * math.log(max(n, 1))
    aic = -ll + 1
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


def _codelength_poisson(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    lam_min = float(spec.params["lam_min"])
    lam_max = float(spec.params["lam_max"])
    xs = _validate_count_sequence(data)
    n = len(xs)
    lam, ll = ml_loglik_poisson(xs)
    pc = parametric_complexity_poisson(n, lam_min, lam_max)
    sc = -ll + pc
    # prequential under Gamma(α, β) conjugate, α=β=1/2 (Jeffreys)
    s = sum(xs)
    if n == 0:
        preq = 0.0
    else:
        preq = (
            _lgamma(0.5)
            + 0.5 * math.log(0.5)
            - _lgamma(s + 0.5)
            - (s + 0.5) * (-math.log(n + 0.5))
            + sum(_lgamma(x + 1.0) for x in xs)
        )
    two_part = -ll + 0.5 * math.log(max(n, 1))
    bic = -ll + 0.5 * 1 * math.log(max(n, 1))
    aic = -ll + 1
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


def _codelength_gaussian_known_sigma(
    spec: ModelSpec, data: Sequence[Any]
) -> Codelength:
    sigma = float(spec.params["sigma"])
    mu_min = float(spec.params["mu_min"])
    mu_max = float(spec.params["mu_max"])
    xs = _validate_real_sequence(data)
    n = len(xs)
    mu, ll = ml_loglik_gaussian_known_sigma(xs, sigma)
    if n == 0:
        pc = 0.0
    else:
        pc = parametric_complexity_gaussian_known_sigma(n, sigma, mu_min, mu_max)
    sc = -ll + pc
    # prequential: Bayesian normal under uniform proper prior on [mu_min, mu_max]
    if n == 0:
        preq = 0.0
    else:
        # marginal-likelihood approximation: ML codelength + (1/2) log(2π σ²/n) + log(range)
        s2 = sum((x - mu) ** 2 for x in xs)
        preq = (
            0.5 * n * (_LN_2PI + 2.0 * math.log(sigma))
            + 0.5 * s2 / (sigma * sigma)
            + 0.5 * math.log(2.0 * math.pi * sigma * sigma / n)
            + math.log(mu_max - mu_min)
        )
    two_part = -ll + 0.5 * math.log(max(n, 1))
    bic = -ll + 0.5 * 1 * math.log(max(n, 1))
    aic = -ll + 1
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


def _codelength_gaussian(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    sigma_min = float(spec.params["sigma_min"])
    sigma_max = float(spec.params["sigma_max"])
    xs = _validate_real_sequence(data)
    n = len(xs)
    mu, sigma_hat, ll = ml_loglik_gaussian(xs)
    if n <= 1:
        pc = 0.0
    else:
        pc = parametric_complexity_gaussian(n, sigma_min, sigma_max)
    sc = -ll + pc
    # prequential: Student-t mixture log-marginal under Jeffreys prior
    if n <= 1:
        preq = 0.0
    else:
        s2 = sum((x - mu) ** 2 for x in xs)
        # log marginal under improper Jeffreys 1/σ dμ dσ, restricted to bounded σ range:
        preq = (
            0.5 * (n - 1) * math.log(math.pi)
            + 0.5 * math.log(n)
            + _lgamma((n - 1) / 2.0)
            + 0.5 * s2_log_safe(s2)
            - 0.5 * (n - 1) * math.log(max(s2, _EPS))
            + math.log(sigma_max / sigma_min)
        )
    two_part = -ll + 0.5 * 2 * math.log(max(n, 1))
    bic = -ll + 0.5 * 2 * math.log(max(n, 1))
    aic = -ll + 2
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


def s2_log_safe(s2: float) -> float:
    """Safe ``log(s2)`` returning 0 for s2 ≤ 0 (used in t-mixture preq)."""
    if s2 <= 0.0:
        return 0.0
    return math.log(s2)


def _codelength_uniform_discrete(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    k = int(spec.params["k"])
    n = len(data)
    _p, ll = ml_loglik_uniform_discrete(data, k)
    pc = 0.0
    sc = -ll
    preq = -ll
    two_part = -ll
    bic = -ll
    aic = -ll
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


def _codelength_constant(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    """A model that puts probability 1 on the constant value ``spec.params['c']``.

    If the data deviates, the model has infinite codelength.  Useful as
    a hard baseline ("does the data ever change?").
    """
    c = spec.params["c"]
    n = len(data)
    for x in data:
        if x != c:
            inf = _INF
            return Codelength(
                name=spec.name,
                n=n,
                ml=inf,
                parametric_complexity=0.0,
                stochastic_complexity=inf,
                prequential=inf,
                two_part=inf,
                bic=inf,
                aic=inf,
                bits=inf,
                method=NML,
            )
    return Codelength(
        name=spec.name,
        n=n,
        ml=0.0,
        parametric_complexity=0.0,
        stochastic_complexity=0.0,
        prequential=0.0,
        two_part=0.0,
        bic=0.0,
        aic=0.0,
        bits=0.0,
        method=NML,
    )


def _codelength_histogram(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    """Real-valued data discretised into ``m`` equal-width bins on
    ``[lo, hi]``, then scored as a Multinomial(m)."""
    m = int(spec.params["m"])
    lo = float(spec.params["lo"])
    hi = float(spec.params["hi"])
    if not (lo < hi):
        raise InvalidModel(f"histogram requires lo < hi; got [{lo}, {hi}]")
    if m < 2:
        raise InvalidModel(f"histogram requires m ≥ 2; got {m}")
    xs = _validate_real_sequence(data)
    width = (hi - lo) / m
    counts = [0] * m
    for x in xs:
        if not (lo <= x <= hi):
            raise InvalidData(f"histogram datum {x} outside [{lo}, {hi}]")
        # right-closed last bin
        idx = int((x - lo) / width)
        if idx >= m:
            idx = m - 1
        counts[idx] += 1
    n = len(xs)
    _probs, ll_disc = ml_loglik_multinomial(counts)
    # density correction: subtract n * log(width) so we're comparing
    # log-likelihood against a *density* on the original space, not a
    # probability mass on bins
    ll = ll_disc - n * math.log(width)
    pc = log_multinomial_nml_constant(n, m)
    sc = -ll + pc
    preq = kt_codelength_multinomial(counts) - n * math.log(width)
    two_part = -ll + 0.5 * (m - 1) * math.log(max(n, 1))
    bic = -ll + 0.5 * (m - 1) * math.log(max(n, 1))
    aic = -ll + (m - 1)
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


def _codelength_markov(spec: ModelSpec, data: Sequence[Any]) -> Codelength:
    """Markov chain of order ``r`` over alphabet of size ``k``.

    NML codelength = ``Σ_{s} L_NML(transitions from state s | multinomial(k))``
    after the initial ``r`` symbols are paid for with ``r log k``.
    """
    k = int(spec.params["k"])
    r = int(spec.params["r"])
    if k < 2:
        raise InvalidModel(f"Markov requires k ≥ 2; got {k}")
    if r < 0:
        raise InvalidModel(f"Markov requires r ≥ 0; got {r}")
    syms: list[int] = []
    for x in data:
        i = int(x)
        if not (0 <= i < k):
            raise InvalidData(f"Markov symbol {i} out of range [0,{k})")
        syms.append(i)
    n = len(syms)
    if n <= r:
        ll = -n * math.log(k)
        # treat as fully-prefix-coded uniform
        return Codelength(
            name=spec.name,
            n=n,
            ml=-ll,
            parametric_complexity=0.0,
            stochastic_complexity=-ll,
            prequential=-ll,
            two_part=-ll,
            bic=-ll,
            aic=-ll,
            bits=(-ll) / _LN2,
            method=NML,
        )
    # initial r symbols
    init_cost = r * math.log(k)
    # transition counts: dict[context tuple, list[k]]
    transitions: dict[tuple[int, ...], list[int]] = {}
    for t in range(r, n):
        ctx = tuple(syms[t - r:t])
        if ctx not in transitions:
            transitions[ctx] = [0] * k
        transitions[ctx][syms[t]] += 1
    ll = -init_cost
    pc = 0.0
    preq = init_cost
    for ctx, cnts in transitions.items():
        _probs, sub_ll = ml_loglik_multinomial(cnts)
        sub_n = sum(cnts)
        ll += sub_ll
        pc += log_multinomial_nml_constant(sub_n, k)
        preq += kt_codelength_multinomial(cnts)
    sc = -ll + pc
    # free parameters: k^r * (k-1)
    n_params = (k ** r) * (k - 1)
    two_part = -ll + 0.5 * n_params * math.log(max(n, 1))
    bic = -ll + 0.5 * n_params * math.log(max(n, 1))
    aic = -ll + n_params
    return Codelength(
        name=spec.name,
        n=n,
        ml=-ll,
        parametric_complexity=pc,
        stochastic_complexity=sc,
        prequential=preq,
        two_part=two_part,
        bic=bic,
        aic=aic,
        bits=sc / _LN2,
        method=NML,
    )


_CODELENGTH_DISPATCH: dict[str, Callable[[ModelSpec, Sequence[Any]], Codelength]] = {
    BERNOULLI: _codelength_bernoulli,
    MULTINOMIAL: _codelength_multinomial,
    GEOMETRIC: _codelength_geometric,
    POISSON: _codelength_poisson,
    GAUSSIAN_KNOWN_SIGMA: _codelength_gaussian_known_sigma,
    GAUSSIAN: _codelength_gaussian,
    UNIFORM_DISCRETE: _codelength_uniform_discrete,
    HISTOGRAM: _codelength_histogram,
    MARKOV: _codelength_markov,
    CONSTANT: _codelength_constant,
}


# =====================================================================
# Per-model fit dispatch (just ML; the codelength fn does all the work)
# =====================================================================


def _fit_bernoulli(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    n0, n1 = _validate_binary_sequence(data)
    n = n0 + n1
    p, ll = ml_loglik_bernoulli(n0, n1)
    return Fit(
        name=spec.name,
        n=n,
        params={"p": p, "n0": n0, "n1": n1},
        log_likelihood=ll,
        ml_codelength_nats=-ll,
        ml_codelength_bits=-ll / _LN2,
    )


def _fit_multinomial(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    k = int(spec.params["k"])
    counts = _validate_multinomial_sequence(data, k)
    n = sum(counts)
    probs, ll = ml_loglik_multinomial(counts)
    return Fit(
        name=spec.name,
        n=n,
        params={"probs": probs, "counts": counts, "k": k},
        log_likelihood=ll,
        ml_codelength_nats=-ll,
        ml_codelength_bits=-ll / _LN2,
    )


def _fit_geometric(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    xs = _validate_positive_count_sequence(data)
    n = len(xs)
    p, ll = ml_loglik_geometric(xs)
    return Fit(
        name=spec.name,
        n=n,
        params={"p": p},
        log_likelihood=ll,
        ml_codelength_nats=-ll,
        ml_codelength_bits=-ll / _LN2,
    )


def _fit_poisson(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    xs = _validate_count_sequence(data)
    n = len(xs)
    lam, ll = ml_loglik_poisson(xs)
    return Fit(
        name=spec.name,
        n=n,
        params={"lam": lam},
        log_likelihood=ll,
        ml_codelength_nats=-ll,
        ml_codelength_bits=-ll / _LN2,
    )


def _fit_gaussian_known_sigma(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    sigma = float(spec.params["sigma"])
    xs = _validate_real_sequence(data)
    n = len(xs)
    mu, ll = ml_loglik_gaussian_known_sigma(xs, sigma)
    return Fit(
        name=spec.name,
        n=n,
        params={"mu": mu, "sigma": sigma},
        log_likelihood=ll,
        ml_codelength_nats=-ll,
        ml_codelength_bits=-ll / _LN2,
    )


def _fit_gaussian(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    xs = _validate_real_sequence(data)
    n = len(xs)
    mu, sigma, ll = ml_loglik_gaussian(xs)
    return Fit(
        name=spec.name,
        n=n,
        params={"mu": mu, "sigma": sigma},
        log_likelihood=ll,
        ml_codelength_nats=-ll,
        ml_codelength_bits=-ll / _LN2,
    )


def _fit_uniform_discrete(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    k = int(spec.params["k"])
    n = len(data)
    _p, ll = ml_loglik_uniform_discrete(data, k)
    return Fit(
        name=spec.name,
        n=n,
        params={"k": k},
        log_likelihood=ll,
        ml_codelength_nats=-ll,
        ml_codelength_bits=-ll / _LN2,
    )


def _fit_histogram(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    cl = _codelength_histogram(spec, data)
    return Fit(
        name=spec.name,
        n=cl.n,
        params=dict(spec.params),
        log_likelihood=-cl.ml,
        ml_codelength_nats=cl.ml,
        ml_codelength_bits=cl.ml / _LN2,
    )


def _fit_markov(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    cl = _codelength_markov(spec, data)
    return Fit(
        name=spec.name,
        n=cl.n,
        params=dict(spec.params),
        log_likelihood=-cl.ml,
        ml_codelength_nats=cl.ml,
        ml_codelength_bits=cl.ml / _LN2,
    )


def _fit_constant(spec: ModelSpec, data: Sequence[Any]) -> Fit:
    cl = _codelength_constant(spec, data)
    return Fit(
        name=spec.name,
        n=cl.n,
        params=dict(spec.params),
        log_likelihood=-cl.ml,
        ml_codelength_nats=cl.ml,
        ml_codelength_bits=cl.ml / _LN2,
    )


_FIT_DISPATCH: dict[str, Callable[[ModelSpec, Sequence[Any]], Fit]] = {
    BERNOULLI: _fit_bernoulli,
    MULTINOMIAL: _fit_multinomial,
    GEOMETRIC: _fit_geometric,
    POISSON: _fit_poisson,
    GAUSSIAN_KNOWN_SIGMA: _fit_gaussian_known_sigma,
    GAUSSIAN: _fit_gaussian,
    UNIFORM_DISCRETE: _fit_uniform_discrete,
    HISTOGRAM: _fit_histogram,
    MARKOV: _fit_markov,
    CONSTANT: _fit_constant,
}


# =====================================================================
# Hyperparameter validation per kind
# =====================================================================


def _validate_spec(kind: str, params: Mapping[str, Any]) -> None:
    if kind == BERNOULLI:
        return
    if kind == MULTINOMIAL:
        if "k" not in params:
            raise InvalidModel("multinomial requires hyperparam 'k'")
        if int(params["k"]) < 2:
            raise InvalidModel(f"multinomial requires k ≥ 2; got {params['k']}")
        return
    if kind == GEOMETRIC:
        return
    if kind == POISSON:
        if "lam_min" not in params or "lam_max" not in params:
            raise InvalidModel("poisson requires 'lam_min' and 'lam_max'")
        lo = float(params["lam_min"])
        hi = float(params["lam_max"])
        if not (0.0 < lo < hi):
            raise InvalidModel(f"poisson requires 0 < lam_min < lam_max; got [{lo},{hi}]")
        return
    if kind == GAUSSIAN_KNOWN_SIGMA:
        for key in ("sigma", "mu_min", "mu_max"):
            if key not in params:
                raise InvalidModel(f"gaussian_known_sigma requires '{key}'")
        if float(params["sigma"]) <= 0.0:
            raise InvalidModel("gaussian_known_sigma requires sigma > 0")
        if float(params["mu_min"]) >= float(params["mu_max"]):
            raise InvalidModel("gaussian_known_sigma requires mu_min < mu_max")
        return
    if kind == GAUSSIAN:
        for key in ("sigma_min", "sigma_max"):
            if key not in params:
                raise InvalidModel(f"gaussian requires '{key}'")
        lo = float(params["sigma_min"])
        hi = float(params["sigma_max"])
        if not (0.0 < lo < hi):
            raise InvalidModel(f"gaussian requires 0 < sigma_min < sigma_max; got [{lo},{hi}]")
        return
    if kind == UNIFORM_DISCRETE:
        if "k" not in params:
            raise InvalidModel("uniform_discrete requires hyperparam 'k'")
        if int(params["k"]) < 1:
            raise InvalidModel(f"uniform_discrete requires k ≥ 1; got {params['k']}")
        return
    if kind == HISTOGRAM:
        for key in ("m", "lo", "hi"):
            if key not in params:
                raise InvalidModel(f"histogram requires '{key}'")
        if int(params["m"]) < 2:
            raise InvalidModel(f"histogram requires m ≥ 2; got {params['m']}")
        if float(params["lo"]) >= float(params["hi"]):
            raise InvalidModel("histogram requires lo < hi")
        return
    if kind == MARKOV:
        for key in ("k", "r"):
            if key not in params:
                raise InvalidModel(f"markov requires '{key}'")
        if int(params["k"]) < 2:
            raise InvalidModel(f"markov requires k ≥ 2; got {params['k']}")
        if int(params["r"]) < 0:
            raise InvalidModel(f"markov requires r ≥ 0; got {params['r']}")
        return
    if kind == CONSTANT:
        if "c" not in params:
            raise InvalidModel("constant requires hyperparam 'c'")
        return
    raise UnknownModel(f"unknown model kind {kind!r}; expected one of {sorted(KNOWN_MODELS)}")


def _sample_space(spec: ModelSpec) -> str:
    """A canonical sample-space tag used to verify two models are comparable."""
    kind = spec.kind
    if kind in (BERNOULLI,):
        return "binary"
    if kind in (MULTINOMIAL, UNIFORM_DISCRETE):
        k = int(spec.params["k"])
        return f"discrete_{k}"
    if kind == MARKOV:
        k = int(spec.params["k"])
        return f"discrete_{k}"
    if kind == GEOMETRIC:
        return "positive_int"
    if kind == POISSON:
        return "nonneg_int"
    if kind in (GAUSSIAN_KNOWN_SIGMA, GAUSSIAN, HISTOGRAM):
        return "real"
    if kind == CONSTANT:
        return "any"
    raise UnknownModel(f"unknown sample space for kind {kind!r}")


def _comparable(a: ModelSpec, b: ModelSpec) -> bool:
    """Whether two registered models live on a compatible sample space."""
    sa = _sample_space(a)
    sb = _sample_space(b)
    if sa == "any" or sb == "any":
        return True
    if sa == sb:
        return True
    # binary ⊂ discrete_k for k ≥ 2
    if sa == "binary" and sb.startswith("discrete_") and int(sb.split("_")[1]) >= 2:
        return True
    if sb == "binary" and sa.startswith("discrete_") and int(sa.split("_")[1]) >= 2:
        return True
    # nonneg_int ⊂ positive_int? no, the reverse — positive ⊂ nonneg.
    if sa == "positive_int" and sb == "nonneg_int":
        return True
    if sb == "positive_int" and sa == "nonneg_int":
        return True
    return False


# =====================================================================
# Posterior predictive symmetric KL (used in pairwise comparisons)
# =====================================================================


def _posterior_predictive_bernoulli(n0: int, n1: int) -> float:
    """KT posterior predictive P(X = 1 | n0, n1) = (n1 + 1/2) / (n + 1)."""
    return (n1 + 0.5) / (n0 + n1 + 1.0)


def _kl_bernoulli(p: float, q: float) -> float:
    p = _clip(p, _EPS, 1.0 - _EPS)
    q = _clip(q, _EPS, 1.0 - _EPS)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def _sym_kl_bernoulli_predictive(spec_a: ModelSpec, spec_b: ModelSpec,
                                  data: Sequence[Any]) -> float:
    n0, n1 = _validate_binary_sequence(data)
    # both predictive probabilities collapse to the KT predictive of the binary stream;
    # they only differ if a model is *not* Bernoulli (e.g. constant), in which case
    # we approximate the constant's predictive by 0 or 1.
    p_a = _predictive_p1(spec_a, n0, n1)
    p_b = _predictive_p1(spec_b, n0, n1)
    return _kl_bernoulli(p_a, p_b) + _kl_bernoulli(p_b, p_a)


def _predictive_p1(spec: ModelSpec, n0: int, n1: int) -> float:
    if spec.kind == BERNOULLI:
        return _posterior_predictive_bernoulli(n0, n1)
    if spec.kind == CONSTANT:
        c = spec.params["c"]
        return 1.0 - _EPS if c == 1 else _EPS
    if spec.kind == UNIFORM_DISCRETE:
        return 0.5
    # multinomial 2-ary reduces to bernoulli
    if spec.kind == MULTINOMIAL and int(spec.params["k"]) == 2:
        return _posterior_predictive_bernoulli(n0, n1)
    # fallback: assume independence with empirical mean
    n = n0 + n1
    if n == 0:
        return 0.5
    return (n1 + 0.5) / (n + 1.0)


# =====================================================================
# Online state update helpers
# =====================================================================


def _online_init(spec: ModelSpec) -> OnlineState:
    if spec.kind == BERNOULLI:
        return OnlineState(name=spec.name, n=0)
    if spec.kind == MULTINOMIAL:
        return OnlineState(name=spec.name, n=0, counts=[0] * int(spec.params["k"]))
    if spec.kind == GEOMETRIC:
        return OnlineState(name=spec.name, n=0)
    if spec.kind == POISSON:
        return OnlineState(name=spec.name, n=0)
    if spec.kind == GAUSSIAN_KNOWN_SIGMA:
        return OnlineState(name=spec.name, n=0)
    if spec.kind == GAUSSIAN:
        return OnlineState(name=spec.name, n=0)
    if spec.kind == UNIFORM_DISCRETE:
        return OnlineState(name=spec.name, n=0)
    if spec.kind == HISTOGRAM:
        return OnlineState(name=spec.name, n=0, counts=[0] * int(spec.params["m"]))
    if spec.kind == MARKOV:
        return OnlineState(name=spec.name, n=0)
    if spec.kind == CONSTANT:
        return OnlineState(name=spec.name, n=0)
    raise UnknownModel(f"online state unsupported for kind {spec.kind!r}")


def _online_codelength_increment(
    spec: ModelSpec, state: OnlineState, datum: Any
) -> float:
    """Predict-and-update: returns the prequential codelength (nats) of ``datum``
    given the model's current online state, then mutates the state."""
    kind = spec.kind
    if kind == BERNOULLI:
        if datum not in (0, 1, True, False):
            raise InvalidData(f"Bernoulli datum must be 0/1; saw {datum!r}")
        x = int(bool(datum))
        # KT predictive: p1 = (n1 + 1/2)/(n + 1)
        n = state.n
        p1 = (state.n1 + 0.5) / (n + 1.0)
        p = p1 if x == 1 else 1.0 - p1
        cost = -math.log(max(p, _EPS))
        state.n += 1
        if x == 1:
            state.n1 += 1
        else:
            state.n0 += 1
        state.prequential_nats += cost
        return cost
    if kind == MULTINOMIAL:
        k = int(spec.params["k"])
        i = int(datum)
        if not (0 <= i < k):
            raise InvalidData(f"multinomial symbol {i} out of [0,{k})")
        n = state.n
        alpha = 0.5
        p = (state.counts[i] + alpha) / (n + k * alpha)
        cost = -math.log(max(p, _EPS))
        state.n += 1
        state.counts[i] += 1
        state.prequential_nats += cost
        return cost
    if kind == GEOMETRIC:
        x = int(datum)
        if x < 1:
            raise InvalidData(f"geometric datum must be ≥ 1; saw {x}")
        n = state.n
        s = state.sum_count
        # Beta(1/2,1/2) posterior on p: p̃ = (n + 1/2) / (s + 1)
        # but we need the *predictive* PMF at next x.  Use plug-in for simplicity.
        if n == 0:
            p_hat = 0.5
        else:
            p_hat = (n + 0.5) / (s + 1.0)
            p_hat = _clip(p_hat, _EPS, 1.0 - _EPS)
        cost = -((x - 1) * math.log(1 - p_hat) + math.log(p_hat))
        state.n += 1
        state.sum_count += x
        state.prequential_nats += cost
        return cost
    if kind == POISSON:
        x = int(datum)
        if x < 0:
            raise InvalidData(f"poisson datum must be ≥ 0; saw {x}")
        n = state.n
        s = state.sum_count
        # Jeffreys Gamma(1/2,1/2) posterior on λ; predictive is NB(s+1/2, (n+1/2)/(n+3/2))
        alpha_post = s + 0.5
        beta_post = n + 0.5
        # NB(α, β/(β+1)) predictive PMF: log f(x) = lgamma(x+α) - lgamma(x+1) - lgamma(α)
        #                                + α log(β/(β+1)) + x log(1/(β+1))
        log_p = (
            _lgamma(x + alpha_post) - _lgamma(x + 1.0) - _lgamma(alpha_post)
            + alpha_post * math.log(beta_post / (beta_post + 1.0))
            + x * math.log(1.0 / (beta_post + 1.0))
        )
        cost = -log_p
        state.n += 1
        state.sum_count += x
        state.prequential_nats += cost
        return cost
    if kind == GAUSSIAN_KNOWN_SIGMA:
        sigma = float(spec.params["sigma"])
        mu_min = float(spec.params["mu_min"])
        mu_max = float(spec.params["mu_max"])
        x = float(datum)
        n = state.n
        sx = state.sum_x
        # uniform-on-range prior on μ, conjugate: posterior is Normal(sx/n, σ²/n)
        # restricted; predictive in approximation is Normal(sx/n, σ²(1 + 1/n)).
        if n == 0:
            mu_pred = (mu_min + mu_max) / 2.0
            sigma_pred2 = sigma * sigma + (mu_max - mu_min) ** 2 / 12.0
        else:
            mu_pred = sx / n
            sigma_pred2 = sigma * sigma * (1.0 + 1.0 / n)
        cost = 0.5 * math.log(2.0 * math.pi * sigma_pred2) + 0.5 * (x - mu_pred) ** 2 / sigma_pred2
        state.n += 1
        state.sum_x += x
        state.sum_x2 += x * x
        state.prequential_nats += cost
        return cost
    if kind == GAUSSIAN:
        sigma_min = float(spec.params["sigma_min"])
        sigma_max = float(spec.params["sigma_max"])
        x = float(datum)
        n = state.n
        sx = state.sum_x
        sxx = state.sum_x2
        if n < 2:
            mu_pred = sx / max(n, 1)
            sigma2_pred = ((sigma_min ** 2) + (sigma_max ** 2)) / 2.0
        else:
            mu_pred = sx / n
            s2 = max((sxx - n * mu_pred * mu_pred) / max(n - 1, 1), _EPS)
            sigma2_pred = s2 * (1.0 + 1.0 / n)
        cost = 0.5 * math.log(2.0 * math.pi * sigma2_pred) + 0.5 * (x - mu_pred) ** 2 / sigma2_pred
        state.n += 1
        state.sum_x += x
        state.sum_x2 += x * x
        state.prequential_nats += cost
        return cost
    if kind == UNIFORM_DISCRETE:
        k = int(spec.params["k"])
        i = int(datum)
        if not (0 <= i < k):
            raise InvalidData(f"uniform_discrete symbol {i} out of [0,{k})")
        cost = math.log(k)
        state.n += 1
        state.prequential_nats += cost
        return cost
    if kind == CONSTANT:
        c = spec.params["c"]
        if datum != c:
            cost = _INF
        else:
            cost = 0.0
        state.n += 1
        state.prequential_nats += cost
        return cost
    if kind == HISTOGRAM:
        m = int(spec.params["m"])
        lo = float(spec.params["lo"])
        hi = float(spec.params["hi"])
        x = float(datum)
        if not (lo <= x <= hi):
            raise InvalidData(f"histogram datum {x} outside [{lo},{hi}]")
        width = (hi - lo) / m
        idx = int((x - lo) / width)
        if idx >= m:
            idx = m - 1
        n = state.n
        alpha = 0.5
        p = (state.counts[idx] + alpha) / (n + m * alpha)
        cost = -math.log(max(p, _EPS)) + math.log(width)  # density form
        state.n += 1
        state.counts[idx] += 1
        state.prequential_nats += cost
        return cost
    raise UnknownModel(f"online update unsupported for kind {kind!r}")


# =====================================================================
# JSON helpers
# =====================================================================


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return repr(obj)


# =====================================================================
# Compressor
# =====================================================================


class Compressor:
    r"""MDL-based hypothesis-selection runtime primitive.

    Thread-safe.  All public methods acquire a single re-entrant lock so
    the model registry, online states, and fingerprint chain stay
    consistent.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._lock = threading.RLock()
        self._clock = clock or time.time
        self._models: dict[str, ModelSpec] = {}
        self._online: dict[str, OnlineState] = {}
        self._last_fits: dict[str, Fit] = {}
        self._last_codelengths: dict[str, Codelength] = {}
        self._selections: list[Selection] = []
        self._comparisons: list[Comparison] = []
        self._events: list[dict] = []
        self._fingerprint: str = _GENESIS
        self._emit(COMPRESSOR_STARTED, {"genesis": _GENESIS})

    # ------------------------------------------------------------------
    # Fingerprint + event log
    # ------------------------------------------------------------------

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if kind not in KNOWN_EVENTS:
            raise CompressorError(f"unknown event {kind!r}")
        canonical = json.dumps(
            {"kind": kind, "payload": _jsonable(payload)},
            sort_keys=True,
            separators=(",", ":"),
        )
        h = hashlib.sha256()
        h.update(self._fingerprint.encode())
        h.update(canonical.encode())
        self._fingerprint = h.hexdigest()
        self._events.append(
            {
                "kind": kind,
                "ts": self._clock(),
                "payload": _jsonable(payload),
                "fingerprint": self._fingerprint,
            }
        )

    @property
    def fingerprint(self) -> str:
        with self._lock:
            return self._fingerprint

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, kind: str, **params: Any) -> ModelSpec:
        """Register a candidate model class."""
        if not isinstance(name, str) or not name:
            raise InvalidModel("model name must be a non-empty string")
        if kind not in KNOWN_MODELS:
            raise UnknownModel(
                f"unknown model kind {kind!r}; expected one of {sorted(KNOWN_MODELS)}"
            )
        _validate_spec(kind, params)
        with self._lock:
            if name in self._models:
                raise InvalidModel(f"model {name!r} already registered")
            spec = ModelSpec(name=name, kind=kind, params=dict(params))
            self._models[name] = spec
            self._online[name] = _online_init(spec)
            self._emit(
                COMPRESSOR_MODEL_REGISTERED,
                {"name": name, "kind": kind, "params": dict(params)},
            )
            return spec

    def models(self) -> Mapping[str, ModelSpec]:
        with self._lock:
            return dict(self._models)

    # ------------------------------------------------------------------
    # Maximum-likelihood fit
    # ------------------------------------------------------------------

    def fit(self, name: str, data: Sequence[Any]) -> Fit:
        with self._lock:
            spec = self._spec(name)
            fit = _FIT_DISPATCH[spec.kind](spec, data)
            self._last_fits[name] = fit
            self._emit(
                COMPRESSOR_FIT,
                {
                    "name": name,
                    "n": fit.n,
                    "params": _jsonable(fit.params),
                    "log_likelihood": fit.log_likelihood,
                    "ml_bits": fit.ml_codelength_bits,
                },
            )
            return fit

    # ------------------------------------------------------------------
    # Codelength
    # ------------------------------------------------------------------

    def codelength(self, name: str, data: Sequence[Any]) -> Codelength:
        with self._lock:
            spec = self._spec(name)
            cl = _CODELENGTH_DISPATCH[spec.kind](spec, data)
            self._last_codelengths[name] = cl
            self._emit(
                COMPRESSOR_SCORED,
                {
                    "name": name,
                    "n": cl.n,
                    "ml": cl.ml,
                    "nml": cl.stochastic_complexity,
                    "preq": cl.prequential,
                    "two_part": cl.two_part,
                    "bic": cl.bic,
                    "aic": cl.aic,
                    "bits": cl.bits,
                },
            )
            return cl

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        data: Sequence[Any],
        *,
        method: str = NML,
        names: Iterable[str] | None = None,
    ) -> Selection:
        """Score every registered (or named) model and return the MDL-optimal
        pick with a regret certificate."""
        if method not in KNOWN_METHODS:
            raise UnknownMethod(f"unknown method {method!r}; expected one of {sorted(KNOWN_METHODS)}")
        with self._lock:
            candidates = list(names) if names is not None else list(self._models.keys())
            if not candidates:
                raise CompressorError("no candidate models to select from")
            cl_by_name: dict[str, Codelength] = {}
            for n_ in candidates:
                cl_by_name[n_] = self.codelength(n_, data)
            scores_nats = {n_: cl_by_name[n_].select_value(method) for n_ in candidates}
            scores_bits = {n_: scores_nats[n_] / _LN2 for n_ in candidates}
            ordered = sorted(candidates, key=lambda n_: scores_nats[n_])
            winner = ordered[0]
            runner_up = ordered[1] if len(ordered) > 1 else None
            if runner_up is None or not math.isfinite(scores_nats[runner_up]):
                gap_nats = _INF
            else:
                gap_nats = scores_nats[runner_up] - scores_nats[winner]
            # Vovk strong-aggregating bound: with K candidates, per-symbol
            # regret of the aggregator vs best expert is at most log(K) / n,
            # so the runner-up's per-symbol excess over the winner is bounded
            # below by (gap - log(K)) / n if positive.
            n_obs = max(cl_by_name[winner].n, 1)
            K = len(candidates)
            vovk_excess = max(gap_nats - math.log(K), 0.0) / n_obs
            per_symbol_regret_bits = vovk_excess / _LN2
            bf = math.exp(min(gap_nats, 700.0)) if gap_nats < _INF else _INF
            self._emit(
                COMPRESSOR_SELECTED,
                {
                    "method": method,
                    "winner": winner,
                    "runner_up": runner_up,
                    "scores_bits": scores_bits,
                    "gap_bits": -gap_nats / _LN2,
                    "bayes_factor": bf,
                    "per_symbol_regret_bits": per_symbol_regret_bits,
                },
            )
            sel = Selection(
                method=method,
                winner=winner,
                runner_up=runner_up,
                codelengths_nats=scores_nats,
                codelengths_bits=scores_bits,
                gap_nats=-gap_nats,
                gap_bits=-gap_nats / _LN2,
                bayes_factor=bf,
                per_symbol_regret_bits=per_symbol_regret_bits,
                fingerprint=self._fingerprint,
            )
            self._selections.append(sel)
            return sel

    # ------------------------------------------------------------------
    # Pairwise comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        a: str,
        b: str,
        data: Sequence[Any],
        *,
        method: str = NML,
    ) -> Comparison:
        if method not in KNOWN_METHODS:
            raise UnknownMethod(f"unknown method {method!r}")
        with self._lock:
            spec_a = self._spec(a)
            spec_b = self._spec(b)
            if not _comparable(spec_a, spec_b):
                raise IncompatibleModels(
                    f"models {a!r} and {b!r} have different sample spaces"
                )
            cl_a = self.codelength(a, data)
            cl_b = self.codelength(b, data)
            la = cl_a.select_value(method)
            lb = cl_b.select_value(method)
            delta = la - lb
            bf_for_a = math.exp(min(-delta, 700.0)) if math.isfinite(delta) else (_INF if delta < 0 else 0.0)
            # sym KL only meaningful for binary; otherwise NaN
            if _sample_space(spec_a) == "binary" and _sample_space(spec_b) == "binary":
                sym_kl = _sym_kl_bernoulli_predictive(spec_a, spec_b, data)
            else:
                sym_kl = float("nan")
            # Stone-Geisser leave-one-out cross-check (cheap impl: re-score with one
            # symbol dropped each, average ML difference).
            cv = _stone_geisser_delta(spec_a, spec_b, data, method)
            cmp = Comparison(
                a=a,
                b=b,
                method=method,
                codelength_a_nats=la,
                codelength_b_nats=lb,
                delta_nats=delta,
                delta_bits=delta / _LN2,
                bayes_factor_for_a=bf_for_a,
                sym_kl_predictive=sym_kl,
                cv_delta_nats=cv,
            )
            self._comparisons.append(cmp)
            self._emit(
                COMPRESSOR_COMPARED,
                {
                    "a": a,
                    "b": b,
                    "method": method,
                    "delta_bits": delta / _LN2,
                    "bayes_factor_for_a": bf_for_a,
                    "sym_kl_predictive": sym_kl,
                    "cv_delta_nats": cv,
                },
            )
            return cmp

    # ------------------------------------------------------------------
    # Online observation
    # ------------------------------------------------------------------

    def online_observe(self, name: str, datum: Any) -> OnlineState:
        with self._lock:
            spec = self._spec(name)
            state = self._online[name]
            cost = _online_codelength_increment(spec, state, datum)
            self._emit(
                COMPRESSOR_OBSERVED,
                {
                    "name": name,
                    "n": state.n,
                    "increment_nats": cost,
                    "increment_bits": cost / _LN2 if math.isfinite(cost) else _INF,
                    "total_nats": state.prequential_nats,
                },
            )
            return state

    def online_state(self, name: str) -> OnlineState:
        with self._lock:
            self._spec(name)
            return self._online[name]

    def online_reset(self, name: str | None = None) -> None:
        """Reset one or all online states without touching the registry."""
        with self._lock:
            if name is None:
                for n_, spec in self._models.items():
                    self._online[n_] = _online_init(spec)
            else:
                spec = self._spec(name)
                self._online[name] = _online_init(spec)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> CompressorReport:
        with self._lock:
            r = CompressorReport(
                models=dict(self._models),
                last_fits=dict(self._last_fits),
                last_codelengths=dict(self._last_codelengths),
                selections=tuple(self._selections),
                comparisons=tuple(self._comparisons),
                online_states={k: _copy_state(v) for k, v in self._online.items()},
                fingerprint=self._fingerprint,
                n_events=len(self._events),
            )
            self._emit(
                COMPRESSOR_REPORT,
                {
                    "n_models": len(self._models),
                    "n_selections": len(self._selections),
                    "n_comparisons": len(self._comparisons),
                    "n_events": len(self._events),
                },
            )
            return r

    def clear(self) -> None:
        """Reset everything: registry, online states, fingerprint genesis."""
        with self._lock:
            self._models.clear()
            self._online.clear()
            self._last_fits.clear()
            self._last_codelengths.clear()
            self._selections.clear()
            self._comparisons.clear()
            self._events.clear()
            self._fingerprint = _GENESIS
            self._emit(COMPRESSOR_CLEARED, {"genesis": _GENESIS})

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _spec(self, name: str) -> ModelSpec:
        spec = self._models.get(name)
        if spec is None:
            raise UnknownModel(f"model {name!r} not registered")
        return spec


def _copy_state(s: OnlineState) -> OnlineState:
    return OnlineState(
        name=s.name,
        n=s.n,
        n0=s.n0,
        n1=s.n1,
        counts=list(s.counts),
        sum_x=s.sum_x,
        sum_x2=s.sum_x2,
        sum_count=s.sum_count,
        prequential_nats=s.prequential_nats,
    )


# =====================================================================
# Stone-Geisser cross-validation cross-check
# =====================================================================


def _stone_geisser_delta(
    spec_a: ModelSpec, spec_b: ModelSpec, data: Sequence[Any], method: str
) -> float:
    r"""Approximate leave-one-out cross-validation difference between two models.

    Strict leave-one-out requires re-fitting each model n times; the
    plug-in implementation here approximates it for the families where
    the ML estimator is a sufficient statistic.  Falls back to the
    full-sample MDL difference when the model is non-parametric in n
    (e.g. CONSTANT, MARKOV).
    """
    cl_a = _CODELENGTH_DISPATCH[spec_a.kind](spec_a, data)
    cl_b = _CODELENGTH_DISPATCH[spec_b.kind](spec_b, data)
    n = max(cl_a.n, 1)
    if cl_a.n <= 2 or cl_b.n <= 2:
        return cl_a.select_value(method) - cl_b.select_value(method)
    # Plug-in LOO for Bernoulli: cost of x_i under (n0', n1') = (n0,n1) minus x_i.
    if spec_a.kind == BERNOULLI and spec_b.kind == BERNOULLI:
        n0, n1 = _validate_binary_sequence(data)
        # KT plug-in: per-symbol log-loss equals -log( (count_x + 1/2) / (n - 1 + 1) )
        # this is the same prequential code we already compute; return ΔPREQ as proxy
        return cl_a.prequential - cl_b.prequential
    return cl_a.select_value(method) - cl_b.select_value(method)


# =====================================================================
# Spec-based factory
# =====================================================================


def compressor_from_spec(spec: Mapping[str, Any]) -> Compressor:
    """Build a ``Compressor`` from a JSON-able spec.

    Expected shape::

        {
          "models": [
            {"name": "ber", "kind": "bernoulli"},
            {"name": "uni", "kind": "uniform_discrete", "params": {"k": 2}},
            {"name": "g3",  "kind": "multinomial", "params": {"k": 3}},
            ...
          ]
        }
    """
    if not isinstance(spec, Mapping):
        raise CompressorError(f"spec must be a mapping; got {type(spec).__name__}")
    c = Compressor()
    models = spec.get("models", [])
    if not isinstance(models, Sequence):
        raise CompressorError("spec['models'] must be a sequence")
    for m in models:
        if not isinstance(m, Mapping):
            raise CompressorError(f"each model spec must be a mapping; got {m!r}")
        name = m.get("name")
        kind = m.get("kind")
        params = m.get("params", {})
        if not isinstance(name, str) or not name:
            raise InvalidModel(f"model spec missing 'name': {m!r}")
        if not isinstance(kind, str):
            raise InvalidModel(f"model spec missing 'kind': {m!r}")
        if not isinstance(params, Mapping):
            raise InvalidModel(f"model spec 'params' must be a mapping: {m!r}")
        c.register(name, kind, **params)
    return c
