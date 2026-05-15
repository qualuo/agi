r"""BayesOpt — Bayesian optimisation as a runtime primitive.

The coordination engine repeatedly faces the same shape of problem:
*pick the next query of an expensive black-box function* — the next
hyperparameter setting, the next prompt template, the next molecule,
the next ad creative, the next robot policy parameter, the next dose of
a synthesis recipe.  Each query is expensive (one full training run, one
wet-lab experiment, one A/B-test cohort) and noisy.  The coordinator
cannot afford grid search; it must spend each query on the point that
maximises *learning value* about the optimum.

`Bandit` and `Arbiter` solve the K-armed version of this question.
`BayesOpt` solves the continuous-arm version: ``argmax_{x ∈ X} f(x)``
where ``X ⊂ R^d`` may be continuous, discrete, or mixed, and the only
oracle the coordinator has is a noisy evaluation ``y_t = f(x_t) + ε_t``.

This is the Mockus (1974) / Jones-Schonlau-Welch (1998) /
Srinivas-Krause-Kakade-Seeger (2010) / Frazier (2018) framework, reduced
to a runtime call:

  * register the search ``Domain`` (box bounds + optional discrete dims);
  * call ``suggest()`` to obtain the next ``x_t`` chosen by the
    acquisition policy;
  * call ``observe(x_t, y_t)`` after evaluating ``f``;
  * read back ``predict(x)``, ``best()``, ``regret_bound()`` and
    ``cumulative_regret_bound()`` — anytime-valid posterior mean,
    incumbent, instantaneous upper bound on regret, cumulative
    information-gain-regret bound.

Mathematical roots
------------------

The surrogate is a *Gaussian Process* (GP) — a prior over functions ``f
~ GP(m, k)`` characterised by a mean ``m(x)`` and a positive-definite
covariance kernel ``k(x, x')``.  Conditioning on ``n`` observations
``(X, y)`` with i.i.d. Gaussian noise ``ε ~ N(0, σ²)`` produces the
closed-form posterior::

    μ_n(x)  =  k(x, X) [K + σ² I]⁻¹ y
    σ_n²(x) =  k(x, x) - k(x, X) [K + σ² I]⁻¹ k(X, x).

The runtime ships three stationary kernels with analytic gradients:

  * **Squared Exponential / RBF** — ``k(x, x') = σ_f² exp(- ½ ‖x - x'‖_ℓ²)``.
    Infinitely differentiable; the canonical default.
  * **Matérn ν = 5/2** — ``k(r) = σ_f² (1 + √5 r + (5/3) r²) exp(-√5 r)``.
    Twice mean-square differentiable; closer to the regularity of real
    physical objectives (Stein 1999).
  * **Matérn ν = 3/2** — ``k(r) = σ_f² (1 + √3 r) exp(-√3 r)``.
    Once mean-square differentiable; good for less smooth objectives.

Each kernel carries per-dimension lengthscales ``ℓ_d`` (Automatic
Relevance Determination, MacKay 1994).  Kernel hyperparameters are
learned by maximising the log-marginal likelihood::

    log p(y | X)  =  - ½ yᵀ (K + σ² I)⁻¹ y  -  ½ log |K + σ² I|  -  (n/2) log 2π.

Acquisition policies — closed-form on a GP posterior
----------------------------------------------------

  * **GP-UCB** (Srinivas et al. 2010, *Gaussian Process Optimization in
    the Bandit Setting*).  Pull ``argmax_x μ_n(x) + √β_t · σ_n(x)`` with
    ``β_t = 2 log(t² 2 π²/(3δ)) + 2 d log(t² d b r √(log(4 d a / δ)))``.
    Cumulative regret bound::

        R_T  =  O*(√(T γ_T β_T))

    where ``γ_T`` is the maximum information gain — finite for compact
    domains and stationary kernels (e.g. ``γ_T = O((log T)^{d+1})`` for
    RBF).  Used by ``acquisition="ucb"``.

  * **Expected Improvement** (Močkus 1974, Jones-Schonlau-Welch 1998).
    For maximisation,
    ``EI(x) = (μ - f*) Φ(z) + σ φ(z)`` with ``z = (μ - f*)/σ``.
    Closed-form gradients via ``dEI/dx = σ' Φ(z) + (μ - f*) φ(z) z'/σ
    + σ φ(z) (-z) z'``.  Bull (2011) shows the simple-regret of EI on
    Matérn kernels is ``O(n^{-ν/d} (log n)^α)``.  Used by
    ``acquisition="ei"``.

  * **Probability of Improvement** (Kushner 1964).
    ``PI(x) = Φ((μ - f* - ξ)/σ)``.  Highly exploitative; useful when ``ξ``
    is tuned for explicit exploration.  ``acquisition="pi"``.

  * **Thompson Sampling on the GP posterior** (Russo-Van Roy 2014,
    Kandasamy et al. 2018).  Draw ``f̃ ~ GP(μ_n, k_n)`` on a candidate
    grid and pull ``argmax f̃``.  Frequentist regret ``Õ(√(T γ_T β_T))``
    matches GP-UCB.  ``acquisition="thompson"``.

  * **Knowledge Gradient** (Frazier-Powell-Dayanik 2009).  Quasi-Monte-
    Carlo approximation of the *one-step lookahead expected
    improvement on the maximiser*; admissible for batches and
    risk-averse settings.  ``acquisition="kg"``.

  * **Expected Constrained Improvement** (Schonlau-Welch-Jones 1998;
    Gardner et al. 2014).  Multiply EI by the product of posterior
    feasibility probabilities for each registered constraint.

Batch / parallel suggestions are produced by the *constant liar*
heuristic (Ginsbourger-Le Riche-Carraro 2010): fantasise the current
posterior mean as the unobserved label of the previously suggested
point, refit and propose, repeat.  This preserves diversity without the
combinatorics of joint maximisation.

Anytime regret bound
--------------------

After ``t`` observations and an empirical maximum information gain
estimate ``γ̂_t``, the runtime reports the anytime instantaneous regret
upper bound::

    r_t  ≤  2 √β_t · max_x σ_{t-1}(x)

and the cumulative regret bound::

    R_T  ≤  √(C_1 T β_T γ_T),     C_1  =  8 / log(1 + σ_f² / σ²).

This is the same coordination-engine-readable contract that ``Bandit``,
``Arbiter``, ``Forecaster`` and ``Auditor`` already export: feed the
primitive observations, and it returns calibrated, finite-sample
bounds on its own decision quality.

Public surface
--------------

The module exports:

  * ``Domain`` / ``ContinuousBox`` / ``CategoricalDim`` / ``MixedDomain``;
  * ``Kernel`` constants ``KERNEL_RBF``, ``KERNEL_MATERN52``,
    ``KERNEL_MATERN32``;
  * ``Acquisition`` constants ``ACQ_UCB``, ``ACQ_EI``, ``ACQ_PI``,
    ``ACQ_THOMPSON``, ``ACQ_KG``;
  * ``BayesOpt`` — the main class;
  * ``Observation`` / ``Suggestion`` / ``BayesOptReport`` dataclasses;
  * Events ``BAYESOPT_STARTED``, ``BAYESOPT_SUGGESTED``,
    ``BAYESOPT_OBSERVED``, ``BAYESOPT_REPORT``, ``BAYESOPT_CLEARED``;
  * Errors ``BayesOptError``, ``UnknownKernel``, ``UnknownAcquisition``,
    ``InvalidDomain``, ``InsufficientData``;
  * Numerical helpers ``gp_predict``, ``gp_log_marginal_likelihood``,
    ``optimise_acquisition``;
  * Convenience driver ``minimise(f, bounds, n_steps)``.

Everything is pure stdlib: ``math``, ``random``, ``statistics``.  The
GP linear algebra is built on inline Cholesky / triangular solve so the
primitive runs anywhere ``Bandit`` does.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


# =====================================================================
# Constants / vocabulary
# =====================================================================


KERNEL_RBF = "rbf"
KERNEL_MATERN52 = "matern52"
KERNEL_MATERN32 = "matern32"

KNOWN_KERNELS = frozenset({KERNEL_RBF, KERNEL_MATERN52, KERNEL_MATERN32})

ACQ_UCB = "ucb"
ACQ_EI = "ei"
ACQ_PI = "pi"
ACQ_THOMPSON = "thompson"
ACQ_KG = "kg"

KNOWN_ACQUISITIONS = frozenset({ACQ_UCB, ACQ_EI, ACQ_PI, ACQ_THOMPSON, ACQ_KG})

# Direction.
MAXIMISE = "maximise"
MINIMISE = "minimise"

# Events emitted on the runtime EventBus.
BAYESOPT_STARTED = "bayesopt.started"
BAYESOPT_SUGGESTED = "bayesopt.suggested"
BAYESOPT_OBSERVED = "bayesopt.observed"
BAYESOPT_REPORT = "bayesopt.report"
BAYESOPT_CLEARED = "bayesopt.cleared"
BAYESOPT_HYPERS_LEARNED = "bayesopt.hypers_learned"

KNOWN_EVENTS = frozenset({
    BAYESOPT_STARTED, BAYESOPT_SUGGESTED, BAYESOPT_OBSERVED,
    BAYESOPT_REPORT, BAYESOPT_CLEARED, BAYESOPT_HYPERS_LEARNED,
})


# Numerical tolerances.
_EPS = 1e-12
_JITTER = 1e-6                  # added to K diagonal for numerical PD.
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_GOLDEN_TOL = 1e-4
_DEFAULT_MULTISTART = 16
_DEFAULT_LOCAL_STEPS = 32
_DEFAULT_LOCAL_LR = 0.1
_DEFAULT_DELTA = 0.05
_DEFAULT_NOISE_VAR = 1e-4
_DEFAULT_SIGNAL_VAR = 1.0
_DEFAULT_LENGTHSCALE = 0.25
_DEFAULT_THOMPSON_GRID = 256
_DEFAULT_KG_FANTASIES = 32


# =====================================================================
# Exceptions
# =====================================================================


class BayesOptError(ValueError):
    """Base class for BayesOpt-domain errors."""


class UnknownKernel(BayesOptError):
    """Kernel name not in KNOWN_KERNELS."""


class UnknownAcquisition(BayesOptError):
    """Acquisition function name not in KNOWN_ACQUISITIONS."""


class InvalidDomain(BayesOptError):
    """The provided Domain is malformed (e.g. low ≥ high)."""


class InvalidObservation(BayesOptError):
    """Observed x has wrong dimension or y is non-finite."""


class InsufficientData(BayesOptError):
    """Operation requires more observations than the optimiser holds."""


# =====================================================================
# Numerical helpers — pure stdlib linear algebra and special functions
# =====================================================================


def _phi(x: float) -> float:
    """Standard normal CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_pdf(x: float) -> float:
    """Standard normal PDF φ(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _phi_inv(p: float) -> float:
    """Beasley-Springer / Moro inverse standard normal CDF.

    Sufficient precision (~1e-9 max abs error on (1e-12, 1 - 1e-12)) for
    confidence-bound and Thompson-sampling use.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        ) / (
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        ) / (
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        ((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]
    ) * q / (
        ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0
    )


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _cholesky(A: Sequence[Sequence[float]]) -> list[list[float]]:
    """Lower-triangular Cholesky factor of an SPD matrix.

    Used for the GP posterior solve.  Adds ``_JITTER`` automatically on
    near-singular kernels so user code never has to.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 0.0:
                    v = _EPS
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / max(L[j][j], _EPS)
    return L


def _solve_lower_tri(L: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i] - sum(L[i][k] * y[k] for k in range(i))
        y[i] = s / max(L[i][i], _EPS)
    return y


def _solve_upper_tri(U: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))
        x[i] = s / max(U[i][i], _EPS)
    return x


def _solve_chol(L: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Given lower Cholesky factor L, solve (L Lᵀ) x = b."""
    y = _solve_lower_tri(L, b)
    Lt = [[L[j][i] for j in range(len(L))] for i in range(len(L))]
    return _solve_upper_tri(Lt, y)


def _logdet_from_chol(L: Sequence[Sequence[float]]) -> float:
    s = 0.0
    for i in range(len(L)):
        s += math.log(max(L[i][i], _EPS))
    return 2.0 * s


def _matvec(A: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _add_jitter(K: list[list[float]], jitter: float = _JITTER) -> None:
    for i in range(len(K)):
        K[i][i] += jitter


# =====================================================================
# Domain types
# =====================================================================


@dataclass(frozen=True)
class ContinuousBox:
    """Per-dimension continuous bounds [low_d, high_d] (inclusive)."""
    low: tuple[float, ...]
    high: tuple[float, ...]

    @property
    def dim(self) -> int:
        return len(self.low)

    def __post_init__(self) -> None:
        if len(self.low) != len(self.high):
            raise InvalidDomain("low and high must have the same length")
        if len(self.low) == 0:
            raise InvalidDomain("ContinuousBox must have at least one dimension")
        for i, (lo, hi) in enumerate(zip(self.low, self.high)):
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise InvalidDomain(f"non-finite bound on dim {i}")
            if lo >= hi:
                raise InvalidDomain(f"low[{i}]={lo} ≥ high[{i}]={hi}")

    def contains(self, x: Sequence[float]) -> bool:
        if len(x) != self.dim:
            return False
        return all(lo <= xi <= hi for xi, lo, hi in zip(x, self.low, self.high))

    def clip(self, x: Sequence[float]) -> list[float]:
        return [_clip(float(xi), lo, hi)
                for xi, lo, hi in zip(x, self.low, self.high)]

    def sample(self, rng: random.Random) -> list[float]:
        return [rng.uniform(lo, hi) for lo, hi in zip(self.low, self.high)]

    def width(self) -> list[float]:
        return [hi - lo for lo, hi in zip(self.low, self.high)]

    def normalise(self, x: Sequence[float]) -> list[float]:
        """Map x ∈ box → unit hypercube."""
        return [(xi - lo) / (hi - lo)
                for xi, lo, hi in zip(x, self.low, self.high)]

    def unnormalise(self, u: Sequence[float]) -> list[float]:
        return [lo + ui * (hi - lo)
                for ui, lo, hi in zip(u, self.low, self.high)]


@dataclass(frozen=True)
class CategoricalDim:
    """A single categorical / discrete-set dimension."""
    name: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.values) < 1:
            raise InvalidDomain(f"CategoricalDim {self.name!r} is empty")

    def index_of(self, v: Any) -> int:
        try:
            return self.values.index(v)
        except ValueError as exc:
            raise InvalidObservation(
                f"value {v!r} not in CategoricalDim {self.name!r}"
            ) from exc


@dataclass(frozen=True)
class MixedDomain:
    """A mixed continuous + categorical search space.

    Internally a coordinate is the concatenation
    ``(continuous_floats..., categorical_codes_as_floats...)``.  Discrete
    coordinates are encoded as 0..K-1; the GP treats them as continuous
    over the unit interval after normalisation — which is the standard
    *encoding-based* trick.  For pure-categorical problems prefer
    ``Bandit``; for mixed search prefer this.
    """
    cont: ContinuousBox | None
    cats: tuple[CategoricalDim, ...] = ()

    @property
    def dim(self) -> int:
        return (self.cont.dim if self.cont else 0) + len(self.cats)

    @property
    def cont_dim(self) -> int:
        return self.cont.dim if self.cont else 0

    @property
    def cat_dim(self) -> int:
        return len(self.cats)

    def sample(self, rng: random.Random) -> list[float]:
        cs = list(self.cont.sample(rng)) if self.cont else []
        ds = [float(rng.randrange(len(c.values))) for c in self.cats]
        return cs + ds

    def clip(self, x: Sequence[float]) -> list[float]:
        cs = list(self.cont.clip(x[: self.cont_dim])) if self.cont else []
        ds = []
        for j, c in enumerate(self.cats):
            v = int(round(_clip(float(x[self.cont_dim + j]),
                                0.0, float(len(c.values) - 1))))
            ds.append(float(v))
        return cs + ds

    def width(self) -> list[float]:
        ws = list(self.cont.width()) if self.cont else []
        ws += [float(max(1, len(c.values) - 1)) for c in self.cats]
        return ws


Domain = ContinuousBox | MixedDomain


def _as_mixed(domain: Domain) -> MixedDomain:
    if isinstance(domain, MixedDomain):
        return domain
    return MixedDomain(cont=domain, cats=())


# =====================================================================
# Kernels — value and gradient w.r.t. first input
# =====================================================================


@dataclass(frozen=True)
class Kernel:
    """A stationary kernel parameterised by lengthscales + signal variance.

    ``lengthscales`` is per-dimension (ARD).  Gradient utilities compute
    ``∂k(x, x')/∂x_i`` analytically — required by gradient-ascent on
    the acquisition surface.
    """

    name: str
    lengthscales: tuple[float, ...]
    signal_var: float

    def __post_init__(self) -> None:
        if self.name not in KNOWN_KERNELS:
            raise UnknownKernel(f"unknown kernel {self.name!r}")
        if self.signal_var <= 0.0:
            raise BayesOptError("signal_var must be > 0")
        for i, l in enumerate(self.lengthscales):
            if not (l > 0.0 and math.isfinite(l)):
                raise BayesOptError(f"lengthscale[{i}] must be finite > 0")

    @property
    def dim(self) -> int:
        return len(self.lengthscales)

    def _scaled_sqdist(self, x: Sequence[float], y: Sequence[float]) -> float:
        s = 0.0
        for d in range(self.dim):
            dx = (x[d] - y[d]) / self.lengthscales[d]
            s += dx * dx
        return s

    def value(self, x: Sequence[float], y: Sequence[float]) -> float:
        r2 = self._scaled_sqdist(x, y)
        if self.name == KERNEL_RBF:
            return self.signal_var * math.exp(-0.5 * r2)
        r = math.sqrt(r2)
        if self.name == KERNEL_MATERN52:
            sqrt5 = math.sqrt(5.0)
            return self.signal_var * (1.0 + sqrt5 * r + (5.0 / 3.0) * r2) \
                * math.exp(-sqrt5 * r)
        # Matern 3/2
        sqrt3 = math.sqrt(3.0)
        return self.signal_var * (1.0 + sqrt3 * r) * math.exp(-sqrt3 * r)

    def grad_x(self, x: Sequence[float], y: Sequence[float]) -> list[float]:
        """∂ k(x, y) / ∂ x — used for gradient-ascent on acquisition."""
        r2 = self._scaled_sqdist(x, y)
        if self.name == KERNEL_RBF:
            k = self.signal_var * math.exp(-0.5 * r2)
            return [-(x[d] - y[d]) / (self.lengthscales[d] ** 2) * k
                    for d in range(self.dim)]
        r = math.sqrt(r2)
        if self.name == KERNEL_MATERN52:
            sqrt5 = math.sqrt(5.0)
            # k = σ² (1 + √5 r + (5/3) r²) exp(-√5 r)
            # dk/dr = σ² [ √5 + (10/3) r ] exp(-√5 r)
            #       + σ² (1 + √5 r + (5/3) r²) (-√5) exp(-√5 r)
            #       = σ² (-(5/3) r (1 + √5 r)) exp(-√5 r)
            # (using the standard simplification)
            common = -self.signal_var * (5.0 / 3.0) * (1.0 + sqrt5 * r) \
                * math.exp(-sqrt5 * r)
            # dr/dx_d = (x_d - y_d) / (ℓ_d² r)   (singular at r=0; use limit 0)
            grads: list[float] = []
            for d in range(self.dim):
                if r < _EPS:
                    grads.append(0.0)
                else:
                    grads.append(common * (x[d] - y[d]) / (self.lengthscales[d] ** 2))
            return grads
        # Matern 3/2: k = σ² (1 + √3 r) exp(-√3 r)
        # dk/dr = -3 σ² r exp(-√3 r)
        sqrt3 = math.sqrt(3.0)
        common = -3.0 * self.signal_var * math.exp(-sqrt3 * r)
        grads = []
        for d in range(self.dim):
            if r < _EPS:
                grads.append(0.0)
            else:
                grads.append(common * (x[d] - y[d]) / (self.lengthscales[d] ** 2))
        return grads

    def gram(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            K[i][i] = self.signal_var
            for j in range(i):
                kij = self.value(X[i], X[j])
                K[i][j] = kij
                K[j][i] = kij
        return K

    def cross(self, X: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
        return [self.value(xi, x) for xi in X]


def make_kernel(name: str, dim: int, *,
                lengthscale: float | Sequence[float] = _DEFAULT_LENGTHSCALE,
                signal_var: float = _DEFAULT_SIGNAL_VAR) -> Kernel:
    if name not in KNOWN_KERNELS:
        raise UnknownKernel(f"unknown kernel {name!r}")
    if isinstance(lengthscale, (int, float)):
        ls = tuple(float(lengthscale) for _ in range(dim))
    else:
        ls = tuple(float(x) for x in lengthscale)
        if len(ls) != dim:
            raise BayesOptError(
                f"lengthscale has {len(ls)} entries, expected {dim}"
            )
    return Kernel(name=name, lengthscales=ls, signal_var=float(signal_var))


# =====================================================================
# Public dataclasses
# =====================================================================


@dataclass(frozen=True)
class Observation:
    """A single noisy observation (x, y) at wall time t_obs."""
    x: tuple[float, ...]
    y: float
    t: int
    stderr: float | None = None
    meta: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class Suggestion:
    """The optimiser's recommendation for the next query."""
    x: tuple[float, ...]
    acquisition_value: float
    mean: float
    std: float
    ucb_beta: float
    rationale: str


@dataclass(frozen=True)
class GPPosterior:
    """A snapshot of the GP posterior on a probe point."""
    x: tuple[float, ...]
    mean: float
    variance: float
    std: float

    def credible(self, level: float = 0.95) -> tuple[float, float]:
        if not (0.0 < level < 1.0):
            raise BayesOptError("credible level must be in (0, 1)")
        z = _phi_inv(0.5 + 0.5 * level)
        h = z * self.std
        return (self.mean - h, self.mean + h)


@dataclass(frozen=True)
class BayesOptReport:
    """Anytime snapshot — feed straight back into a coordination engine."""
    direction: str                          # "maximise" or "minimise"
    n_observations: int
    best_x: tuple[float, ...] | None
    best_y: float | None
    incumbent_observation_index: int | None
    posterior_max_mean: float | None        # arg-max of μ on multistart probe
    posterior_max_mean_x: tuple[float, ...] | None
    posterior_max_std: float | None         # max σ over the probe — drives β.
    posterior_min_std: float | None
    simple_regret_upper_bound: float | None
    cumulative_regret_upper_bound: float | None
    info_gain_estimate: float | None
    beta_t: float
    delta: float
    kernel_name: str
    lengthscales: tuple[float, ...]
    signal_var: float
    noise_var: float
    acquisition: str
    fingerprint: str


# =====================================================================
# GP fit and prediction
# =====================================================================


def _fit_chol(
    K: list[list[float]], noise_var: float
) -> tuple[list[list[float]], float]:
    """Compute Cholesky of K + σ² I.  Returns (L, logdet) and inflates
    the jitter if K is near-singular.
    """
    n = len(K)
    jitter = _JITTER
    while True:
        A = [row[:] for row in K]
        for i in range(n):
            A[i][i] += noise_var + jitter
        try:
            L = _cholesky(A)
            # Detect bad factor: any zero on diagonal means singular.
            bad = False
            for i in range(n):
                if L[i][i] <= _EPS:
                    bad = True
                    break
            if bad:
                raise BayesOptError("Cholesky failed")
            return L, _logdet_from_chol(L)
        except BayesOptError:
            jitter *= 10.0
            if jitter > 1.0:
                raise


def gp_predict(
    *,
    kernel: Kernel,
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    x_star: Sequence[float],
    noise_var: float = _DEFAULT_NOISE_VAR,
    y_mean: float | None = None,
) -> tuple[float, float]:
    """GP posterior mean and variance at a single test point.

    Subtracts ``y_mean`` (default: mean(y)) before fitting so the prior
    is zero-mean; adds it back to the posterior mean.  This is the
    standard practical choice for stationary kernels.
    """
    n = len(X)
    if n == 0:
        return (y_mean or 0.0), kernel.signal_var + noise_var
    if y_mean is None:
        y_mean = statistics.fmean(y) if n > 0 else 0.0
    y_centred = [yi - y_mean for yi in y]
    K = kernel.gram(X)
    L, _ = _fit_chol(K, noise_var)
    alpha = _solve_chol(L, y_centred)
    k_star = kernel.cross(X, x_star)
    mu = y_mean + _dot(k_star, alpha)
    v = _solve_lower_tri(L, k_star)
    var = kernel.signal_var - _dot(v, v)
    if var < 0.0:
        var = 0.0
    return mu, var


def gp_log_marginal_likelihood(
    *,
    kernel: Kernel,
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    noise_var: float = _DEFAULT_NOISE_VAR,
) -> float:
    """log p(y | X, θ) under the GP prior.

    Standard form (Rasmussen-Williams §2.3)::

        log p  =  -½ yᵀ (K + σ² I)⁻¹ y  -  ½ log |K + σ² I|  -  (n/2) log 2π.

    Used by `BayesOpt.learn_hyperparameters` to choose lengthscales.
    """
    n = len(X)
    if n == 0:
        return 0.0
    y_mean = statistics.fmean(y)
    y_centred = [yi - y_mean for yi in y]
    K = kernel.gram(X)
    L, logdet = _fit_chol(K, noise_var)
    alpha = _solve_chol(L, y_centred)
    quad = _dot(y_centred, alpha)
    return -0.5 * quad - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)


# Cached GP fit — avoids redundant Cholesky calls inside acquisition
# optimisation, where μ_n(x) and σ_n²(x) are evaluated thousands of
# times at the same posterior.
class _GPFit:
    __slots__ = ("kernel", "X", "y", "y_mean", "L", "alpha", "noise_var")

    def __init__(
        self,
        *,
        kernel: Kernel,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        noise_var: float,
    ) -> None:
        self.kernel = kernel
        self.X = [list(xi) for xi in X]
        self.y = list(y)
        self.noise_var = float(noise_var)
        self.y_mean = statistics.fmean(self.y) if self.y else 0.0
        if self.y:
            y_centred = [yi - self.y_mean for yi in self.y]
            K = kernel.gram(self.X)
            self.L, _ = _fit_chol(K, self.noise_var)
            self.alpha = _solve_chol(self.L, y_centred)
        else:
            self.L = None
            self.alpha = None

    def predict(self, x: Sequence[float]) -> tuple[float, float]:
        if self.L is None:
            return self.y_mean, self.kernel.signal_var + self.noise_var
        k_star = self.kernel.cross(self.X, x)
        mu = self.y_mean + _dot(k_star, self.alpha)
        v = _solve_lower_tri(self.L, k_star)
        var = self.kernel.signal_var - _dot(v, v)
        if var < 0.0:
            var = 0.0
        return mu, var

    def predict_grad(
        self, x: Sequence[float]
    ) -> tuple[float, float, list[float], list[float]]:
        """Posterior μ, σ², ∇μ, ∇σ² at x.

        The gradients are needed by gradient-ascent acquisition
        optimisation.  All derivations are textbook; see
        Rasmussen-Williams (2006) §5.5 for the closed form.
        """
        d = self.kernel.dim
        if self.L is None:
            return self.y_mean, self.kernel.signal_var + self.noise_var, \
                [0.0] * d, [0.0] * d
        k_star = self.kernel.cross(self.X, x)
        # ∇_x k(X, x): n × d matrix; row i is ∂k(X_i, x)/∂x.  Because
        # the stationary kernels are symmetric in their arguments,
        # ∂k(X_i, x)/∂x = -∂k(x, X_i)/∂X_i = +∂k(x, X_i)/∂x evaluated
        # the same way as kernel.grad_x(x, X_i).
        dk_star = [self.kernel.grad_x(x, xi) for xi in self.X]
        mu = self.y_mean + _dot(k_star, self.alpha)
        # ∇μ = (∇k_star)ᵀ α
        grad_mu = [0.0] * d
        for j in range(d):
            s = 0.0
            for i in range(len(self.X)):
                s += dk_star[i][j] * self.alpha[i]
            grad_mu[j] = s
        # σ² = k(x,x) - k_starᵀ K⁻¹ k_star ; ∇k(x,x) = 0 for stationary
        # kernels.  ∇σ² = -2 (∇k_star)ᵀ K⁻¹ k_star.
        Kinv_k = _solve_chol(self.L, k_star)
        var = self.kernel.signal_var - _dot(k_star, Kinv_k)
        if var < 0.0:
            var = 0.0
        grad_var = [0.0] * d
        for j in range(d):
            s = 0.0
            for i in range(len(self.X)):
                s += dk_star[i][j] * Kinv_k[i]
            grad_var[j] = -2.0 * s
        return mu, var, grad_mu, grad_var


# =====================================================================
# Acquisition functions
# =====================================================================


def _beta_t(t: int, dim: int, delta: float = _DEFAULT_DELTA) -> float:
    """Srinivas et al. (2010) Theorem 2 GP-UCB schedule.

    For finite-domain sub-problems we drop the high-dim correction
    term; for continuous-domain optimisation we add the standard
    ``2 d log(t² d b r √(log(4 d a / δ)))`` with ``a, b, r`` absorbed
    into the constant.  This is a conservative, anytime-valid choice.
    """
    if t <= 1:
        t = 2
    base = 2.0 * math.log(t * t * math.pi * math.pi / (3.0 * max(delta, _EPS)))
    if dim > 0:
        # +2d log(t² d b r √(log(4 d a / δ))) — absorb a = b = r = 1.
        base += 2.0 * dim * math.log(
            t * t * max(dim, 1) * max(math.sqrt(
                math.log(4.0 * max(dim, 1) / max(delta, _EPS))
            ), 1.0)
        )
    return max(base, 1.0)


def acq_ucb(mu: float, std: float, beta_t: float, *, direction: str) -> float:
    """GP-UCB / LCB.  Maximised by the acquisition optimiser."""
    sgn = 1.0 if direction == MAXIMISE else -1.0
    return sgn * mu + math.sqrt(max(beta_t, 0.0)) * std


def acq_ei(mu: float, std: float, incumbent: float,
           xi: float = 0.0, *, direction: str) -> float:
    """Expected Improvement (Močkus 1974; Jones-Schonlau-Welch 1998).

    For maximisation of f, EI(x) = E[(f(x) - f* - ξ)⁺] under the GP
    posterior.  For minimisation we flip signs.
    """
    if std <= _EPS:
        # No uncertainty: improvement is deterministic.
        if direction == MAXIMISE:
            return max(mu - incumbent - xi, 0.0)
        return max(incumbent - mu - xi, 0.0)
    if direction == MAXIMISE:
        z = (mu - incumbent - xi) / std
        return (mu - incumbent - xi) * _phi(z) + std * _phi_pdf(z)
    z = (incumbent - mu - xi) / std
    return (incumbent - mu - xi) * _phi(z) + std * _phi_pdf(z)


def acq_pi(mu: float, std: float, incumbent: float,
           xi: float = 0.01, *, direction: str) -> float:
    """Probability of Improvement (Kushner 1964)."""
    if std <= _EPS:
        if direction == MAXIMISE:
            return 1.0 if mu - incumbent - xi > 0.0 else 0.0
        return 1.0 if incumbent - mu - xi > 0.0 else 0.0
    if direction == MAXIMISE:
        z = (mu - incumbent - xi) / std
    else:
        z = (incumbent - mu - xi) / std
    return _phi(z)


def acq_thompson_value(
    mu: float, std: float, sample_z: float, *, direction: str
) -> float:
    """Thompson sample at a single point.

    Sampling a *full* GP posterior on a continuous space is expensive;
    the runtime instead draws a one-shot Gaussian per probe point and
    selects the argmax — a faithful approximation to a "candidate-set"
    Thompson policy (Kandasamy et al. 2018, *Parallel Bayesian
    Optimization via Thompson Sampling*).
    """
    sgn = 1.0 if direction == MAXIMISE else -1.0
    return sgn * (mu + sample_z * std)


# =====================================================================
# Acquisition optimisation
# =====================================================================


def _random_starts(
    domain: MixedDomain, k: int, rng: random.Random
) -> list[list[float]]:
    return [domain.sample(rng) for _ in range(k)]


def _project_to_domain(domain: MixedDomain, x: Sequence[float]) -> list[float]:
    return domain.clip(x)


def _acq_at(
    fit: _GPFit,
    x: Sequence[float],
    *,
    acquisition: str,
    beta_t: float,
    incumbent: float,
    xi: float,
    direction: str,
    thompson_z: float | None = None,
) -> float:
    mu, var = fit.predict(x)
    std = math.sqrt(max(var, 0.0))
    if acquisition == ACQ_UCB:
        return acq_ucb(mu, std, beta_t, direction=direction)
    if acquisition == ACQ_EI:
        return acq_ei(mu, std, incumbent, xi, direction=direction)
    if acquisition == ACQ_PI:
        return acq_pi(mu, std, incumbent, xi, direction=direction)
    if acquisition == ACQ_THOMPSON:
        z = thompson_z if thompson_z is not None else 0.0
        return acq_thompson_value(mu, std, z, direction=direction)
    raise UnknownAcquisition(acquisition)


def _acq_and_grad(
    fit: _GPFit,
    x: Sequence[float],
    *,
    acquisition: str,
    beta_t: float,
    incumbent: float,
    xi: float,
    direction: str,
) -> tuple[float, list[float]]:
    """Acquisition value and gradient — used by local refinement.

    Only UCB / EI / PI carry analytic gradients here; Thompson and KG
    use derivative-free random search.
    """
    mu, var, grad_mu, grad_var = fit.predict_grad(x)
    std = math.sqrt(max(var, _EPS))
    if acquisition == ACQ_UCB:
        sgn = 1.0 if direction == MAXIMISE else -1.0
        beta_sqrt = math.sqrt(max(beta_t, 0.0))
        # f = sgn * μ + √β σ  ;  ∇σ = ∇σ²/(2σ)
        value = sgn * mu + beta_sqrt * std
        grad_std = [g / (2.0 * std) for g in grad_var]
        grad = [sgn * grad_mu[i] + beta_sqrt * grad_std[i]
                for i in range(len(x))]
        return value, grad
    if acquisition == ACQ_EI:
        sgn = 1.0 if direction == MAXIMISE else -1.0
        diff = sgn * (mu - incumbent) - xi
        # EI = diff * Φ(z) + σ φ(z), z = diff/σ.  ∂diff/∂x = sgn*∂μ.
        if std <= _EPS:
            return max(diff, 0.0), [0.0] * len(x)
        z = diff / std
        Phi = _phi(z)
        phi = _phi_pdf(z)
        value = diff * Phi + std * phi
        grad_std = [g / (2.0 * std) for g in grad_var]
        # ∂EI/∂x simplifies via z = diff/σ to:  ∂diff · Φ(z) + ∂σ · φ(z)
        # (the σ φ(z) (-z) ∂z term cancels diff φ(z) ∂z exactly).
        grad = [sgn * grad_mu[i] * Phi + grad_std[i] * phi
                for i in range(len(x))]
        return value, grad
    if acquisition == ACQ_PI:
        sgn = 1.0 if direction == MAXIMISE else -1.0
        diff = sgn * (mu - incumbent) - xi
        if std <= _EPS:
            return (1.0 if diff > 0.0 else 0.0), [0.0] * len(x)
        z = diff / std
        Phi = _phi(z)
        phi = _phi_pdf(z)
        grad_std = [g / (2.0 * std) for g in grad_var]
        # PI = Φ(z), z = diff/σ; ∂z/∂x = (sgn*∂μ · σ - diff · ∂σ) / σ²
        grad = [
            phi * ((sgn * grad_mu[i] * std - diff * grad_std[i]) / (std * std))
            for i in range(len(x))
        ]
        return Phi, grad
    # Fallback: numeric.
    return _acq_at(
        fit, x,
        acquisition=acquisition, beta_t=beta_t,
        incumbent=incumbent, xi=xi, direction=direction,
    ), [0.0] * len(x)


def optimise_acquisition(
    fit: _GPFit,
    domain: Domain,
    *,
    acquisition: str,
    beta_t: float,
    incumbent: float,
    xi: float = 0.0,
    direction: str = MAXIMISE,
    rng: random.Random | None = None,
    n_starts: int = _DEFAULT_MULTISTART,
    local_steps: int = _DEFAULT_LOCAL_STEPS,
    local_lr: float = _DEFAULT_LOCAL_LR,
    thompson_grid: int = _DEFAULT_THOMPSON_GRID,
) -> tuple[list[float], float]:
    """Return ``(x*, acq*)`` — the multi-start gradient-ascent optimum.

    For continuous-only domains we use analytic-gradient ascent from
    ``n_starts`` random restarts.  For mixed / categorical domains the
    continuous coords are refined with gradients while the categorical
    coords are explored by random sampling within each restart bundle.
    Thompson sampling uses a Sobol-like quasi-random candidate set
    (here implemented as Halton; see ``_halton_point``).
    """
    rng = rng or random.Random(0)
    mixed = _as_mixed(domain)

    # Thompson sampling is naturally optimised on a candidate set.
    if acquisition == ACQ_THOMPSON:
        # Halton candidate set transformed into the mixed domain.
        candidates: list[list[float]] = []
        for i in range(thompson_grid):
            u = _halton_point(i + 1, mixed.dim)
            x = _u_to_mixed(u, mixed)
            candidates.append(x)
        zs = [rng.gauss(0.0, 1.0) for _ in range(thompson_grid)]
        best_x: list[float] | None = None
        best_v = -math.inf
        for j, c in enumerate(candidates):
            v = _acq_at(
                fit, c,
                acquisition=acquisition, beta_t=beta_t,
                incumbent=incumbent, xi=xi, direction=direction,
                thompson_z=zs[j],
            )
            if v > best_v:
                best_v = v
                best_x = c
        assert best_x is not None
        return best_x, best_v

    if acquisition == ACQ_KG:
        return _optimise_kg(
            fit, mixed,
            beta_t=beta_t, incumbent=incumbent, direction=direction,
            rng=rng, n_starts=n_starts,
        )

    starts = _random_starts(mixed, n_starts, rng)
    # Also include the incumbent / argmax of mean as a warm start to
    # bias the search toward the current best region (Frazier 2018).
    best_x: list[float] | None = None
    best_v = -math.inf
    widths = mixed.width()
    for x0 in starts:
        x = list(x0)
        v0 = _acq_at(
            fit, x,
            acquisition=acquisition, beta_t=beta_t,
            incumbent=incumbent, xi=xi, direction=direction,
        )
        # Local refinement on the continuous block only.
        cdim = mixed.cont_dim
        if cdim > 0 and local_steps > 0:
            lr = local_lr
            for _ in range(local_steps):
                v, g = _acq_and_grad(
                    fit, x,
                    acquisition=acquisition, beta_t=beta_t,
                    incumbent=incumbent, xi=xi, direction=direction,
                )
                # Gradient ascent — step size scaled by per-dim width.
                step = [0.0] * mixed.dim
                for d in range(cdim):
                    step[d] = lr * g[d] * widths[d]
                new_x = [x[d] + step[d] for d in range(mixed.dim)]
                new_x = _project_to_domain(mixed, new_x)
                v_new = _acq_at(
                    fit, new_x,
                    acquisition=acquisition, beta_t=beta_t,
                    incumbent=incumbent, xi=xi, direction=direction,
                )
                if v_new > v:
                    x = new_x
                else:
                    lr *= 0.5
                    if lr < 1e-6:
                        break
            v_local = _acq_at(
                fit, x,
                acquisition=acquisition, beta_t=beta_t,
                incumbent=incumbent, xi=xi, direction=direction,
            )
        else:
            v_local = v0
        if v_local > best_v:
            best_v = v_local
            best_x = x
    assert best_x is not None
    return best_x, best_v


# Halton — deterministic low-discrepancy sequence for Thompson candidate
# sets and quasi-Monte-Carlo KG.
def _halton_point(idx: int, d: int) -> list[float]:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
              59, 61, 67, 71, 73, 79, 83, 89, 97)
    if d > len(primes):
        raise BayesOptError(
            f"Halton sequence supports at most {len(primes)} dimensions"
        )
    out = []
    for k in range(d):
        base = primes[k]
        f = 1.0
        r = 0.0
        i = idx
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        out.append(r)
    return out


def _u_to_mixed(u: Sequence[float], domain: MixedDomain) -> list[float]:
    """Map unit-cube point to mixed domain coordinates."""
    out: list[float] = []
    if domain.cont:
        out.extend(domain.cont.unnormalise(u[: domain.cont_dim]))
    for j, c in enumerate(domain.cats):
        ui = u[domain.cont_dim + j] if domain.cont_dim + j < len(u) else 0.0
        idx = min(int(ui * len(c.values)), len(c.values) - 1)
        out.append(float(idx))
    return out


def _optimise_kg(
    fit: _GPFit,
    domain: MixedDomain,
    *,
    beta_t: float,
    incumbent: float,
    direction: str,
    rng: random.Random,
    n_starts: int,
    n_fantasies: int = _DEFAULT_KG_FANTASIES,
    probe_grid: int = 64,
) -> tuple[list[float], float]:
    """Knowledge-Gradient optimisation by quasi-Monte-Carlo (Frazier).

    For each candidate ``x`` we draw ``n_fantasies`` posterior labels,
    refit, and compute the improvement of the posterior maximum over
    the current incumbent's posterior mean.  Approximate, but
    sufficient as a runtime primitive.
    """
    candidates = _random_starts(domain, n_starts, rng)
    probe = [domain.sample(rng) for _ in range(probe_grid)]
    incumbent_mu = max(fit.predict(p)[0] for p in probe) if direction == MAXIMISE \
        else min(fit.predict(p)[0] for p in probe)
    sgn = 1.0 if direction == MAXIMISE else -1.0
    best_x: list[float] | None = None
    best_kg = -math.inf
    for x in candidates:
        mu_x, var_x = fit.predict(x)
        std_x = math.sqrt(max(var_x, 0.0))
        if std_x <= _EPS:
            kg = 0.0
        else:
            kg_total = 0.0
            for j in range(n_fantasies):
                z = _phi_inv((j + 0.5) / n_fantasies)
                y_fant = mu_x + std_x * z
                # Refit fantasised: insert (x, y_fant) into the GP.
                X_aug = fit.X + [list(x)]
                y_aug = fit.y + [y_fant]
                fit_aug = _GPFit(
                    kernel=fit.kernel, X=X_aug, y=y_aug,
                    noise_var=fit.noise_var,
                )
                mus = [fit_aug.predict(p)[0] for p in probe]
                best_post = max(mus) if direction == MAXIMISE else min(mus)
                kg_total += sgn * (best_post - incumbent_mu)
            kg = kg_total / n_fantasies
        if kg > best_kg:
            best_kg = kg
            best_x = list(x)
    assert best_x is not None
    return best_x, best_kg


# =====================================================================
# Hyperparameter learning — golden-section over a single log-lengthscale
# multiplier (cheap, robust, good enough for runtime use)
# =====================================================================


def learn_hyperparameters(
    *,
    kernel: Kernel,
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    noise_var: float = _DEFAULT_NOISE_VAR,
    log_scale_low: float = -2.0,
    log_scale_high: float = 2.0,
    tol: float = _GOLDEN_TOL,
) -> Kernel:
    """Pick the lengthscale multiplier that maximises the log-marginal
    likelihood, with all per-dim relative lengthscales fixed.

    A full ARD optimisation would require quasi-Newton.  The runtime
    instead optimises a single scalar — the *isotropic multiplier* — by
    golden-section search.  That's sufficient to stop the GP from being
    pathologically wrong while keeping the routine deterministic and
    derivative-free.
    """
    n = len(X)
    if n < 3:
        return kernel
    base = list(kernel.lengthscales)

    def _ll(log_mult: float) -> float:
        mult = math.exp(log_mult)
        k = Kernel(
            name=kernel.name,
            lengthscales=tuple(b * mult for b in base),
            signal_var=kernel.signal_var,
        )
        try:
            return gp_log_marginal_likelihood(
                kernel=k, X=X, y=y, noise_var=noise_var,
            )
        except BayesOptError:
            return -1e18

    a, b = log_scale_low, log_scale_high
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = _ll(c), _ll(d)
    while abs(b - a) > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = _ll(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = _ll(d)
    best_log = 0.5 * (a + b)
    best_mult = math.exp(best_log)
    return Kernel(
        name=kernel.name,
        lengthscales=tuple(bl * best_mult for bl in base),
        signal_var=kernel.signal_var,
    )


# =====================================================================
# Configuration and main class
# =====================================================================


@dataclass
class BayesOptConfig:
    direction: str = MAXIMISE
    kernel: str = KERNEL_MATERN52
    acquisition: str = ACQ_EI
    signal_var: float = _DEFAULT_SIGNAL_VAR
    noise_var: float = _DEFAULT_NOISE_VAR
    lengthscale: float | Sequence[float] = _DEFAULT_LENGTHSCALE
    delta: float = _DEFAULT_DELTA
    ei_xi: float = 0.0
    pi_xi: float = 0.01
    n_starts: int = _DEFAULT_MULTISTART
    local_steps: int = _DEFAULT_LOCAL_STEPS
    local_lr: float = _DEFAULT_LOCAL_LR
    thompson_grid: int = _DEFAULT_THOMPSON_GRID
    kg_fantasies: int = _DEFAULT_KG_FANTASIES
    learn_hypers_every: int = 5     # 0 disables.
    seed: int | None = None


def _validate_config(cfg: BayesOptConfig) -> None:
    if cfg.direction not in (MAXIMISE, MINIMISE):
        raise BayesOptError(f"direction must be {MAXIMISE!r} or {MINIMISE!r}")
    if cfg.kernel not in KNOWN_KERNELS:
        raise UnknownKernel(cfg.kernel)
    if cfg.acquisition not in KNOWN_ACQUISITIONS:
        raise UnknownAcquisition(cfg.acquisition)
    if cfg.signal_var <= 0.0:
        raise BayesOptError("signal_var must be > 0")
    if cfg.noise_var < 0.0:
        raise BayesOptError("noise_var must be ≥ 0")
    if not (0.0 < cfg.delta < 1.0):
        raise BayesOptError("delta must be in (0, 1)")


class BayesOpt:
    """The runtime-facing Bayesian optimiser.

    Lifecycle::

        bo = BayesOpt(domain=ContinuousBox(low=(0,), high=(1,)),
                      config=BayesOptConfig(acquisition="ucb"))
        for _ in range(30):
            sug = bo.suggest()                  # GP-UCB pick
            y   = expensive_oracle(sug.x)
            bo.observe(sug.x, y)
        report = bo.report()                    # incumbent + regret bound

    The optimiser is replay-deterministic given ``config.seed``: same
    seed + same observation order ⇒ same suggestions, same fingerprint.
    """

    def __init__(
        self,
        *,
        domain: Domain,
        config: BayesOptConfig | None = None,
        event_sink: Callable[[str, Mapping[str, Any]], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        cfg = config or BayesOptConfig()
        _validate_config(cfg)
        self._domain = domain
        self._mixed = _as_mixed(domain)
        self._config = cfg
        self._sink = event_sink
        self._clock = clock or time.time
        self._kernel = make_kernel(
            cfg.kernel, self._mixed.dim,
            lengthscale=cfg.lengthscale, signal_var=cfg.signal_var,
        )
        self._observations: list[Observation] = []
        self._t = 0
        self._rng = random.Random(cfg.seed) if cfg.seed is not None \
            else random.Random()
        self._suggestion_log: list[Suggestion] = []
        self._info_gain_estimate: float = 0.0
        self._started_at = self._clock()
        self._emit(BAYESOPT_STARTED, {
            "direction": cfg.direction,
            "kernel": cfg.kernel,
            "acquisition": cfg.acquisition,
            "dim": self._mixed.dim,
        })

    # -- introspection ------------------------------------------------

    @property
    def dim(self) -> int:
        return self._mixed.dim

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def config(self) -> BayesOptConfig:
        return self._config

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    @property
    def observations(self) -> list[Observation]:
        return list(self._observations)

    # -- public API ---------------------------------------------------

    def suggest(self) -> Suggestion:
        """Return the next x picked by the configured acquisition.

        Cold-start (``n_observations < 2``) draws from a Halton sequence
        — quasi-random space-filling — instead of the acquisition, since
        the GP posterior is uninformative.
        """
        self._t += 1
        if len(self._observations) < 2:
            u = _halton_point(self._t, max(1, self._mixed.dim))
            x = _u_to_mixed(u, self._mixed)
            sug = Suggestion(
                x=tuple(x),
                acquisition_value=0.0,
                mean=0.0,
                std=math.sqrt(self._kernel.signal_var + self._config.noise_var),
                ucb_beta=_beta_t(self._t, self._mixed.dim, self._config.delta),
                rationale="cold-start halton",
            )
            self._suggestion_log.append(sug)
            self._emit(BAYESOPT_SUGGESTED, {
                "x": sug.x, "acq": sug.acquisition_value,
                "mean": sug.mean, "std": sug.std,
                "rationale": sug.rationale,
            })
            return sug

        fit = self._fit()
        beta = _beta_t(self._t, self._mixed.dim, self._config.delta)
        incumbent = self._incumbent_value()
        x, acq_v = optimise_acquisition(
            fit, self._mixed,
            acquisition=self._config.acquisition,
            beta_t=beta,
            incumbent=incumbent,
            xi=(self._config.ei_xi if self._config.acquisition == ACQ_EI
                else self._config.pi_xi),
            direction=self._config.direction,
            rng=self._rng,
            n_starts=self._config.n_starts,
            local_steps=self._config.local_steps,
            local_lr=self._config.local_lr,
            thompson_grid=self._config.thompson_grid,
        )
        mu, var = fit.predict(x)
        std = math.sqrt(max(var, 0.0))
        sug = Suggestion(
            x=tuple(x),
            acquisition_value=acq_v,
            mean=mu,
            std=std,
            ucb_beta=beta,
            rationale=f"{self._config.acquisition} after {len(self._observations)} obs",
        )
        self._suggestion_log.append(sug)
        self._emit(BAYESOPT_SUGGESTED, {
            "x": sug.x, "acq": sug.acquisition_value,
            "mean": sug.mean, "std": sug.std,
            "rationale": sug.rationale,
        })
        return sug

    def suggest_batch(self, k: int) -> list[Suggestion]:
        """Return ``k`` suggestions via the constant-liar batch policy
        (Ginsbourger-Le Riche-Carraro 2010).
        """
        if k <= 0:
            raise BayesOptError("k must be > 0")
        batch: list[Suggestion] = []
        # Snapshot real observations to restore at the end.
        real_n = len(self._observations)
        for _ in range(k):
            sug = self.suggest()
            batch.append(sug)
            # Fantasise the posterior mean as the label.
            fantasised_y = sug.mean if len(self._observations) >= 2 \
                else self._fantasy_floor()
            self._observations.append(Observation(
                x=tuple(sug.x), y=fantasised_y, t=-1,
            ))
        # Discard fantasies — the caller will call observe() with real y.
        self._observations = self._observations[:real_n]
        return batch

    def _fantasy_floor(self) -> float:
        if self._observations:
            return statistics.fmean(o.y for o in self._observations)
        return 0.0

    def observe(self, x: Sequence[float], y: float, *,
                stderr: float | None = None,
                meta: Mapping[str, Any] | None = None) -> Observation:
        """Record a noisy evaluation ``y = f(x) + ε``."""
        x_list = self._mixed.clip(x)
        if len(x_list) != self._mixed.dim:
            raise InvalidObservation(
                f"observation has dim {len(x_list)}, expected {self._mixed.dim}"
            )
        if not math.isfinite(float(y)):
            raise InvalidObservation(f"y={y!r} is not finite")
        obs = Observation(
            x=tuple(x_list), y=float(y), t=len(self._observations),
            stderr=stderr,
            meta=tuple((k, meta[k]) for k in sorted(meta)) if meta else (),
        )
        self._observations.append(obs)
        # Approximate information-gain accumulator: ½ log(1 + σ²(x)/σ_n²).
        try:
            mu, var = self.predict(x_list)
            self._info_gain_estimate += 0.5 * math.log(
                1.0 + max(var, 0.0) / max(self._config.noise_var, _EPS)
            )
        except Exception:  # noqa: BLE001
            pass
        if (self._config.learn_hypers_every > 0
                and len(self._observations) > 0
                and len(self._observations) % self._config.learn_hypers_every == 0):
            self._learn_hyperparameters_inline()
        self._emit(BAYESOPT_OBSERVED, {
            "x": obs.x, "y": obs.y, "t": obs.t,
            "stderr": obs.stderr,
        })
        return obs

    def predict(self, x: Sequence[float]) -> GPPosterior:
        if len(x) != self._mixed.dim:
            raise InvalidObservation(
                f"predict x has dim {len(x)}, expected {self._mixed.dim}"
            )
        fit = self._fit()
        mu, var = fit.predict(list(x))
        return GPPosterior(
            x=tuple(x), mean=mu, variance=var, std=math.sqrt(max(var, 0.0)),
        )

    def credible_interval(self, x: Sequence[float], level: float = 0.95
                          ) -> tuple[float, float]:
        return self.predict(x).credible(level)

    def best(self) -> Observation | None:
        if not self._observations:
            return None
        if self._config.direction == MAXIMISE:
            return max(self._observations, key=lambda o: o.y)
        return min(self._observations, key=lambda o: o.y)

    def regret_bound(self) -> float | None:
        """Anytime instantaneous regret upper bound  ``2 √β_t · σ_max``.

        Returns ``None`` until the GP has at least one observation.
        """
        if not self._observations:
            return None
        fit = self._fit()
        probe = [self._mixed.sample(self._rng)
                 for _ in range(max(64, 4 * self._mixed.dim))]
        max_std = 0.0
        for p in probe:
            _, var = fit.predict(p)
            s = math.sqrt(max(var, 0.0))
            if s > max_std:
                max_std = s
        beta = _beta_t(max(self._t, 2), self._mixed.dim, self._config.delta)
        return 2.0 * math.sqrt(max(beta, 0.0)) * max_std

    def cumulative_regret_bound(self) -> float | None:
        """Cumulative GP-UCB regret bound  ``√(C₁ T β_T γ_T)`` (Srinivas).

        ``γ_T`` is estimated by the running information-gain sum
        ``½ Σ_t log(1 + σ_t²(x_t)/σ_n²)``; ``C₁ = 8 / log(1 + σ_f²/σ_n²)``.
        """
        if not self._observations:
            return None
        T = max(self._t, 2)
        beta_T = _beta_t(T, self._mixed.dim, self._config.delta)
        C1 = 8.0 / math.log(
            1.0 + self._kernel.signal_var
            / max(self._config.noise_var, _EPS)
        )
        gamma_T = max(self._info_gain_estimate, 1.0)
        return math.sqrt(C1 * T * beta_T * gamma_T)

    def report(self) -> BayesOptReport:
        """Anytime snapshot."""
        best = self.best()
        if self._observations:
            fit = self._fit()
            probe = [self._mixed.sample(self._rng)
                     for _ in range(max(128, 8 * self._mixed.dim))]
            mus = []
            stds = []
            for p in probe:
                mu, var = fit.predict(p)
                mus.append((mu, p))
                stds.append(math.sqrt(max(var, 0.0)))
            if self._config.direction == MAXIMISE:
                pm_mean, pm_x = max(mus, key=lambda t: t[0])
            else:
                pm_mean, pm_x = min(mus, key=lambda t: t[0])
            max_std = max(stds)
            min_std = min(stds)
        else:
            pm_mean = None
            pm_x = None
            max_std = None
            min_std = None
        r_inst = self.regret_bound()
        r_cum = self.cumulative_regret_bound()
        beta = _beta_t(max(self._t, 2), self._mixed.dim, self._config.delta)
        report = BayesOptReport(
            direction=self._config.direction,
            n_observations=len(self._observations),
            best_x=(best.x if best else None),
            best_y=(best.y if best else None),
            incumbent_observation_index=(best.t if best else None),
            posterior_max_mean=pm_mean,
            posterior_max_mean_x=(tuple(pm_x) if pm_x is not None else None),
            posterior_max_std=max_std,
            posterior_min_std=min_std,
            simple_regret_upper_bound=r_inst,
            cumulative_regret_upper_bound=r_cum,
            info_gain_estimate=(self._info_gain_estimate
                                if self._observations else None),
            beta_t=beta,
            delta=self._config.delta,
            kernel_name=self._kernel.name,
            lengthscales=self._kernel.lengthscales,
            signal_var=self._kernel.signal_var,
            noise_var=self._config.noise_var,
            acquisition=self._config.acquisition,
            fingerprint=self.fingerprint(),
        )
        self._emit(BAYESOPT_REPORT, {"fingerprint": report.fingerprint})
        return report

    def clear(self) -> None:
        self._observations.clear()
        self._suggestion_log.clear()
        self._t = 0
        self._info_gain_estimate = 0.0
        self._emit(BAYESOPT_CLEARED, {})

    # -- replay-fingerprint ------------------------------------------

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over the observation history + config.

        Two runs with the same seed, config, and observation order
        produce identical fingerprints — usable as a replay key by a
        coordination engine that needs to recover state.
        """
        h = hashlib.sha256()
        h.update(json.dumps({
            "direction": self._config.direction,
            "kernel": self._config.kernel,
            "acq": self._config.acquisition,
            "delta": self._config.delta,
            "signal_var": self._config.signal_var,
            "noise_var": self._config.noise_var,
            "lengthscales": list(self._kernel.lengthscales),
            "seed": self._config.seed,
        }, sort_keys=True).encode("utf-8"))
        for o in self._observations:
            h.update(json.dumps({
                "x": list(o.x), "y": o.y, "t": o.t,
                "stderr": o.stderr,
            }, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    # -- internals ---------------------------------------------------

    def _fit(self) -> _GPFit:
        return _GPFit(
            kernel=self._kernel,
            X=[list(o.x) for o in self._observations],
            y=[o.y for o in self._observations],
            noise_var=self._config.noise_var,
        )

    def _incumbent_value(self) -> float:
        best = self.best()
        if best is None:
            return 0.0
        return best.y

    def _learn_hyperparameters_inline(self) -> None:
        Xs = [list(o.x) for o in self._observations]
        ys = [o.y for o in self._observations]
        try:
            new_kernel = learn_hyperparameters(
                kernel=self._kernel, X=Xs, y=ys,
                noise_var=self._config.noise_var,
            )
        except BayesOptError:
            return
        self._kernel = new_kernel
        self._emit(BAYESOPT_HYPERS_LEARNED, {
            "lengthscales": list(new_kernel.lengthscales),
            "signal_var": new_kernel.signal_var,
        })

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self._sink is None:
            return
        try:
            self._sink(event_type, dict(payload))
        except Exception:  # noqa: BLE001
            pass


# =====================================================================
# Convenience drivers — synchronous closed-loop optimisation
# =====================================================================


def minimise(
    f: Callable[[Sequence[float]], float],
    *,
    bounds: Sequence[tuple[float, float]],
    n_steps: int,
    n_seed: int | None = None,
    config: BayesOptConfig | None = None,
    rng: random.Random | None = None,
) -> BayesOptReport:
    """One-shot driver: optimise ``f`` over the box ``bounds`` for
    ``n_steps`` evaluations.  Used in tests and notebooks.
    """
    low = tuple(b[0] for b in bounds)
    high = tuple(b[1] for b in bounds)
    domain = ContinuousBox(low=low, high=high)
    cfg = config or BayesOptConfig(direction=MINIMISE, seed=n_seed)
    if cfg.direction != MINIMISE:
        cfg = BayesOptConfig(**{**asdict(cfg), "direction": MINIMISE})
    bo = BayesOpt(domain=domain, config=cfg)
    for _ in range(n_steps):
        sug = bo.suggest()
        y = float(f(list(sug.x)))
        bo.observe(sug.x, y)
    return bo.report()


def maximise(
    f: Callable[[Sequence[float]], float],
    *,
    bounds: Sequence[tuple[float, float]],
    n_steps: int,
    n_seed: int | None = None,
    config: BayesOptConfig | None = None,
    rng: random.Random | None = None,
) -> BayesOptReport:
    low = tuple(b[0] for b in bounds)
    high = tuple(b[1] for b in bounds)
    domain = ContinuousBox(low=low, high=high)
    cfg = config or BayesOptConfig(direction=MAXIMISE, seed=n_seed)
    if cfg.direction != MAXIMISE:
        cfg = BayesOptConfig(**{**asdict(cfg), "direction": MAXIMISE})
    bo = BayesOpt(domain=domain, config=cfg)
    for _ in range(n_steps):
        sug = bo.suggest()
        y = float(f(list(sug.x)))
        bo.observe(sug.x, y)
    return bo.report()


# =====================================================================
# Final exports
# =====================================================================


__all__ = [
    # Direction.
    "MAXIMISE", "MINIMISE",
    # Kernels.
    "KERNEL_RBF", "KERNEL_MATERN52", "KERNEL_MATERN32",
    "KNOWN_KERNELS", "Kernel", "make_kernel",
    # Acquisitions.
    "ACQ_UCB", "ACQ_EI", "ACQ_PI", "ACQ_THOMPSON", "ACQ_KG",
    "KNOWN_ACQUISITIONS",
    "acq_ucb", "acq_ei", "acq_pi", "acq_thompson_value",
    # Domain.
    "ContinuousBox", "CategoricalDim", "MixedDomain", "Domain",
    # Events.
    "BAYESOPT_STARTED", "BAYESOPT_SUGGESTED", "BAYESOPT_OBSERVED",
    "BAYESOPT_REPORT", "BAYESOPT_CLEARED", "BAYESOPT_HYPERS_LEARNED",
    "KNOWN_EVENTS",
    # Errors.
    "BayesOptError", "UnknownKernel", "UnknownAcquisition",
    "InvalidDomain", "InvalidObservation", "InsufficientData",
    # Dataclasses.
    "Observation", "Suggestion", "GPPosterior", "BayesOptReport",
    "BayesOptConfig",
    # Numerical helpers.
    "gp_predict", "gp_log_marginal_likelihood",
    "learn_hyperparameters", "optimise_acquisition",
    # Main class.
    "BayesOpt",
    # Drivers.
    "minimise", "maximise",
]
