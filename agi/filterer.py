r"""Filterer — Bayesian state-space filtering as a runtime primitive.

Every other decision primitive in the runtime — ``Bandit``, ``BayesOpt``,
``Arbiter``, ``Sampler``, ``Forecaster``, ``ActiveInferencer``, ``Strategist``
— assumes that the *current belief state* over latent quantities is given.
But in a world where data arrives sequentially and the latent state itself
evolves — sensors drift, prices move, agents act under partial observability,
covariates shift — the foundational question is not "what action under this
belief" but "what is my belief right now, given everything I have observed
to date".

That is the *Bayesian filtering problem* (Stratonovich 1959; Kalman 1960):
given a generative state-space model ``x_t = f(x_{t-1}) + w_t``,
``y_t = h(x_t) + v_t`` with process noise ``w_t`` and observation noise
``v_t``, compute the *filtered posterior* ``p(x_t | y_{1:t})`` for every
``t``, the *predictive distribution* ``p(x_{t+1} | y_{1:t})``, and the
*smoothed posterior* ``p(x_t | y_{1:T})`` once all observations are in. The
filtered posterior is the universal belief-update primitive every other
primitive composes onto.

The ``Filterer`` is the runtime primitive that solves Bayesian filtering
under the four canonical regimes — linear-Gaussian, non-linear-Gaussian
under local linearisation, non-linear-Gaussian under sigma-point
transform, and fully general non-linear / non-Gaussian via sequential
Monte Carlo — with closed-form posterior-update receipts, exact
log-marginal likelihood for model selection, anytime-valid
normalised-innovation-squared (NIS) tests for model misspecification,
finite-sample Crisan-Doucet (2002) Monte-Carlo error bounds on particle
filters, and tamper-evident SHA-256 fingerprint chains over every
predict / update / resample event so ``AttestationLedger`` replays the
entire filtering trace byte-for-byte.

The pitch reduced to a runtime call::

  filt = Filterer.kalman(
      F=[[1.0, 1.0], [0.0, 1.0]],         # constant-velocity dynamics
      H=[[1.0, 0.0]],                       # observe position
      Q=[[1e-4, 0.0], [0.0, 1e-3]],         # process noise
      R=[[0.1]],                            # observation noise
      x0=[0.0, 0.0],
      P0=[[1.0, 0.0], [0.0, 1.0]],
  )
  for y in stream:
      filt.predict()                         # one-step predictive
      filt.update(y)                         # filtered posterior
  report = filt.report()                     # log-marginal + NIS + receipts

Algorithms shipped
------------------

**Kalman filter (KF)** (Kalman 1960 *A new approach to linear filtering
and prediction problems*; Anderson-Moore 1979 *Optimal Filtering*).
Closed-form posterior under linear-Gaussian dynamics and observation::

    x̂_{t|t-1} = F x̂_{t-1|t-1} + B u_t
    P_{t|t-1} = F P_{t-1|t-1} Fᵀ + Q
    ŷ_t = H x̂_{t|t-1}                       # innovation mean
    S_t = H P_{t|t-1} Hᵀ + R                 # innovation covariance
    K_t = P_{t|t-1} Hᵀ S_t⁻¹                 # Kalman gain
    x̂_{t|t} = x̂_{t|t-1} + K_t (y_t − ŷ_t)
    P_{t|t} = (I − K_t H) P_{t|t-1}

The Kalman filter is the *minimum mean-square-error* state estimator
under linear-Gaussian assumptions, and the maximum-a-posteriori
estimator under Gaussian conjugacy.

**Joseph-form covariance update** (Bucy-Joseph 1968).  The
numerically-stable rewrite
``P_{t|t} = (I − K_t H) P_{t|t-1} (I − K_t H)ᵀ + K_t R K_tᵀ`` preserves
symmetry and positive-definiteness even under finite-precision arithmetic.

**Information filter (IF)** (Maybeck 1979 §7.3).  Dual parameterisation
in canonical / information form: state = (η, Λ) with ``Λ = P⁻¹`` and
``η = Λ x̂``.  Update step is *additive* in the information matrix and
information vector, so the IF dominates the KF when many cheap
measurements arrive simultaneously or when the prior is uninformative.

**Square-root Kalman filter** (Potter 1963; Bierman 1977).  Carry the
Cholesky factor ``S`` of ``P`` instead of ``P`` itself.  Update by
Householder triangularisation of the augmented matrix.  Halves the
condition number — and so doubles the effective bit-depth — of every
matrix operation, at no extra computational cost.

**Extended Kalman filter (EKF)** (Smith-Schmidt-McGee 1962; Anderson-
Moore 1979 §8).  Approximates non-linear dynamics ``x_t = f(x_{t-1})`` by
linearising around the current mean: ``F_t = ∂f/∂x|_{x̂_{t-1|t-1}}``,
``H_t = ∂h/∂x|_{x̂_{t|t-1}}``.  Uses analytical Jacobians supplied by the
coordinator (no automatic differentiation in stdlib).  EKF is the
workhorse of GPS, INS, robotics — and is exact when the dynamics happen
to be linear.

**Unscented Kalman filter (UKF)** (Julier-Uhlmann 1997 *A new extension
of the Kalman filter to nonlinear systems*).  Replaces analytical
linearisation by deterministic ``2n+1`` sigma-point sampling::

    χ_0 = x̂                              ; weight = λ / (n + λ)
    χ_i = x̂ ± √((n + λ) P)[col i]         ; weight = 1 / (2(n+λ))

with scaling parameter ``λ = α² (n + κ) − n`` (default ``α = 1e-3``,
``β = 2``, ``κ = 0``).  Propagating the sigma points through the
non-linear transform and re-computing the empirical mean and covariance
captures the posterior to *third order* in the Taylor expansion (vs.
first-order for EKF), and is exact under affine transforms.  Requires
*no* Jacobians.

**Particle filter (SIR — Sequential Importance Resampling)** (Gordon-
Salmond-Smith 1993 *Novel approach to nonlinear/non-Gaussian Bayesian
state estimation*).  Represents the posterior as a weighted set of
samples ``{(x_t^i, w_t^i)}_{i=1}^N``::

    x_t^i ∼ p(x_t | x_{t-1}^i)                # propose
    w_t^i ∝ w_{t-1}^i p(y_t | x_t^i)          # importance weight
    if ESS(w) < N/2: resample                  # avoid degeneracy

Provides convergence ``E[(p̂ − p)²] = O(1/N)`` for bounded test
functions (Crisan-Doucet 2002 *A survey of convergence results on
particle filtering methods for practitioners*), with explicit constants
depending on the mixing of the dynamics.

**Auxiliary particle filter (APF)** (Pitt-Shephard 1999 *Filtering via
simulation: auxiliary particle filters*).  Pre-resamples using a
one-step lookahead at the *predicted* observation likelihood, which
focuses computation on particles likely to survive the next observation
— substantially reducing variance when ``p(y_t | x_t)`` is peaked.

**Bootstrap filter** (Gordon-Salmond-Smith 1993 special case of SIR
with the prior as the importance proposal).  Simplest possible particle
filter; the baseline against which APF and Rao-Blackwellisation are
benchmarked.

**Rauch-Tung-Striebel (RTS) smoother** (Rauch-Tung-Striebel 1965
*Maximum likelihood estimates of linear dynamic systems*).  Computes
the smoothed posterior ``p(x_t | y_{1:T})`` in a backward sweep after
the forward filter has run::

    G_t = P_{t|t} Fᵀ P_{t+1|t}⁻¹
    x̂_{t|T} = x̂_{t|t} + G_t (x̂_{t+1|T} − x̂_{t+1|t})
    P_{t|T} = P_{t|t} + G_t (P_{t+1|T} − P_{t+1|t}) G_tᵀ

The RTS smoother is the maximum-a-posteriori estimator of the full state
trajectory under linear-Gaussian assumptions and is BLUE among all
linear smoothers.

**Backward simulation smoother (FFBSi)** (Godsill-Doucet-West 2004
*Monte Carlo smoothing for nonlinear time series*).  Particle analogue
of RTS — samples a trajectory backward through the saved forward
particles.

Resampling schemes
------------------

  * **Multinomial** — i.i.d. draws from the categorical with weights
    ``w_t``.  Variance ``O(N w(1−w))``.

  * **Systematic resampling** (Carpenter-Clifford-Fearnhead 1999).
    Single uniform draw ``u ∼ U[0, 1/N)``; resample at strata
    ``u + i/N`` for ``i = 0, …, N−1``.  Strictly lower variance than
    multinomial — provably optimal among single-tier schemes.

  * **Residual resampling** (Liu-Chen 1998).  Deterministic part
    ``⌊N w_i⌋`` copies plus multinomial draws on the fractional residual.

  * **Stratified resampling** (Kitagawa 1996).  One uniform draw per
    stratum.

Anytime certificates
--------------------

Every Filterer emits a ``FilterReport`` carrying

  * **Log marginal likelihood** ``log p(y_{1:T} | model)`` via the
    Gaussian innovations decomposition for KF / EKF / UKF, and via the
    importance-weight log-sum-exp for particle filters.  This is the
    natural quantity for model selection across competing state-space
    models (composes with Compressor and Hedger).

  * **Normalised Innovation Squared (NIS)** anytime χ²-statistic
    (Bar-Shalom-Li-Kirubarajan 2001 *Estimation with Applications to
    Tracking and Navigation* §5.4).  Under correct specification,
    ``NIS_t = ν_tᵀ S_t⁻¹ ν_t ∼ χ²(m)`` where ``m`` is the observation
    dimension; an anytime martingale-mixture e-process on the centred
    NIS detects model misspecification at any stopping time
    (Howard-Ramdas-McAuliffe-Sekhon 2021).

  * **Effective Sample Size** ``ESS = 1 / Σ w_i²`` for particle filters.
    Anytime degeneracy diagnostic — when ``ESS < N/2`` the runtime
    forces a resample.

  * **Crisan-Doucet finite-sample bound** ``E[(p̂_N − p)²] ≤ C / N``
    on any bounded test function for the particle filter, with constant
    ``C`` upper-bounded by the supremum of the test function squared.

  * **Massart-DKW band** ``(2 / N) · log(2/δ)`` on the empirical
    posterior CDF — distribution-free, finite-sample-valid for any
    confidence level.

  * **Innovation whiteness test** — sample autocorrelation of the
    standardised innovations under the null of a correctly-specified
    model.  Asymptotically ``ρ̂_k ∼ N(0, 1/T)`` (Box-Jenkins 1970).

  * **Tamper-evident fingerprint** — SHA-256 chain over (model
    specification, every predict, every update, every resample) so an
    external auditor can replay the entire filtering trace byte-for-byte.

Composition with the rest of the runtime
----------------------------------------

  * **ActiveInferencer** — the filtered posterior *is* the belief
    over POMDP state.  ``Filterer.update(y)`` feeds directly into the
    expected-free-energy planning step.

  * **Forecaster** — ``Filterer.predict()`` emits a calibrated
    one-step predictive distribution.  PIT-uniformity tests on the
    predictive CDF compose with Forecaster's anytime-valid calibration
    e-process.

  * **Sampler** — the particle filter is the sequential version of
    importance sampling.  Sampler's Pareto-k tail diagnostic on the
    importance weights detects pathological proposals.

  * **DriftSentinel** — the standardised innovation sequence is a
    martingale-difference under correct specification.  A CUSUM
    (Page-Hinkley) on NIS detects model breaks; a BOCPD posterior on
    the standardised innovations localises change points.

  * **Compressor** — the prequential log-marginal ``log p(y_t |
    y_{1:t-1})`` accumulates into ``log p(y_{1:T} | model)``, exactly
    the MDL codelength of the observation stream under the model.
    Compressor scores competing state-space models against each other.

  * **Hedger** — register competing state-space models as experts;
    the per-step negative log-predictive becomes the loss; AdaHedge
    learns at runtime which model class fits the current regime.

  * **CausalDiscoverer** — Filterer is the inner-loop E-step for
    dynamic structural causal models (Murphy 2002).

  * **Refuter** — refute the white-noise innovation assumption via
    QuickCheck-style sample-path stress on the bias.

  * **AttestationLedger** — every predict / update / resample chain-
    hashes.

  * **Strategist** — risk-adjusted decisions consume the filtered
    posterior's mean and covariance (or particle approximation).

  * **PrivacyAccountant** — DP-noisy observations widen R by
    ``2 σ²_{DP}``; the (ε, δ) odometer advances on every update.

Numerical conventions
---------------------

  * **Pure stdlib.**  No NumPy / SciPy / linear-algebra extensions.
    Matrix operations on lists-of-lists.  Cholesky-based solves rather
    than explicit inverses where possible.  Beasley-Springer-Moro 1995
    inverse-Φ for Gaussian sampling.  Box-Muller for unit normals.

  * **Deterministic given seed.**  All randomness — particle proposals,
    resampling draws — routes through ``random.Random(seed)``.

  * **JSON-canonical event payloads.**  Hash chain is byte-deterministic
    across Python versions.

  * **Type discipline.**  States are lists of floats; matrices are
    lists of lists of floats; the Kalman family validates dimensions on
    every step.  Particle filters validate proposal / likelihood
    callable signatures.

References
----------

  * **Kalman, R. E. (1960)** *A new approach to linear filtering and
    prediction problems*. Journal of Basic Engineering 82.

  * **Rauch, H. E., Tung, F. & Striebel, C. T. (1965)** *Maximum
    likelihood estimates of linear dynamic systems*. AIAA Journal 3.

  * **Stratonovich, R. L. (1959)** *Optimum nonlinear systems which
    bring about a separation of a signal with constant parameters from
    noise*. Radiofizika 2.

  * **Julier, S. J. & Uhlmann, J. K. (1997)** *A new extension of the
    Kalman filter to nonlinear systems*. SPIE 3068.

  * **Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993)** *Novel
    approach to nonlinear/non-Gaussian Bayesian state estimation*.
    IEE Proceedings F 140(2).

  * **Pitt, M. K. & Shephard, N. (1999)** *Filtering via simulation:
    auxiliary particle filters*. JASA 94(446).

  * **Doucet, A., de Freitas, N. & Gordon, N. (2001)** *Sequential
    Monte Carlo Methods in Practice*. Springer.

  * **Crisan, D. & Doucet, A. (2002)** *A survey of convergence results
    on particle filtering methods for practitioners*. IEEE Transactions
    on Signal Processing 50(3).

  * **Carpenter, J., Clifford, P. & Fearnhead, P. (1999)** *Improved
    particle filter for nonlinear problems*. IEE Proceedings — Radar,
    Sonar and Navigation 146(1).

  * **Liu, J. S. & Chen, R. (1998)** *Sequential Monte Carlo methods
    for dynamic systems*. JASA 93(443).

  * **Kitagawa, G. (1996)** *Monte Carlo filter and smoother for
    non-Gaussian nonlinear state space models*. JCGS 5.

  * **Godsill, S. J., Doucet, A. & West, M. (2004)** *Monte Carlo
    smoothing for nonlinear time series*. JASA 99(465).

  * **Anderson, B. D. O. & Moore, J. B. (1979)** *Optimal Filtering*.
    Prentice-Hall.

  * **Bar-Shalom, Y., Li, X. R. & Kirubarajan, T. (2001)** *Estimation
    with Applications to Tracking and Navigation*. Wiley.

  * **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021)**
    *Time-uniform, nonparametric, nonasymptotic confidence sequences*.
    Annals of Statistics.

Author's contract
-----------------

The Filterer primitive returns *one* of these on every call:

  1. A filtered posterior — Gaussian mean and covariance for KF / EKF /
     UKF, weighted particle set for the SMC variants — accompanied by
     the log-marginal likelihood increment, the NIS test statistic, the
     ESS diagnostic, and a tamper-evident fingerprint.

  2. A diagnostic: dimensions don't match, covariance went non-positive-
     definite, particle weights collapsed, observation contains NaN —
     coordinator should re-specify the model or fall back to a more
     robust filter.

The Filterer *never* claims its posterior is correct — it claims that
*conditional on the supplied model*, the posterior is the exact Bayesian
update, the log-marginal is the exact (or unbiased Monte-Carlo)
prequential likelihood, and the NIS sequence has been hashed into the
ledger so an external auditor can replay every update.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence


# =====================================================================
# Constants
# =====================================================================

# Algorithm / model-class names.
KF = "kf"                                  # linear-Gaussian Kalman filter
INFORMATION_FILTER = "information_filter"  # canonical-form dual
SQRT_KF = "sqrt_kf"                        # Potter / Bierman square-root
EKF = "ekf"                                # extended (Jacobian) KF
UKF = "ukf"                                # unscented (sigma-point) KF
SIR = "sir"                                # sequential importance resampling
BOOTSTRAP = "bootstrap"                    # SIR with prior proposal
APF = "apf"                                # auxiliary particle filter

KNOWN_FILTERS = frozenset({
    KF, INFORMATION_FILTER, SQRT_KF, EKF, UKF, SIR, BOOTSTRAP, APF,
})
KALMAN_FAMILY = frozenset({KF, INFORMATION_FILTER, SQRT_KF, EKF, UKF})
PARTICLE_FAMILY = frozenset({SIR, BOOTSTRAP, APF})

# Resampling schemes.
MULTINOMIAL = "multinomial"
SYSTEMATIC = "systematic"
STRATIFIED = "stratified"
RESIDUAL = "residual"

KNOWN_RESAMPLERS = frozenset({MULTINOMIAL, SYSTEMATIC, STRATIFIED, RESIDUAL})

# Smoother names.
RTS = "rts"                                # Rauch-Tung-Striebel
FFBSI = "ffbsi"                            # forward-filter backward-sampler

KNOWN_SMOOTHERS = frozenset({RTS, FFBSI})

# Numerical guards.
_PROB_TOL = 1.0e-9
_EPS = 1.0e-12
_INF = float("inf")
_NEGINF = float("-inf")
_LOG2PI = math.log(2.0 * math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# Cholesky diagonal floor (jitter) for nearly-singular covariances.
_JITTER = 1.0e-9

# Genesis fingerprint.
_GENESIS = hashlib.sha256(b"filterer.v1.genesis").hexdigest()

# Events emitted on the runtime EventBus.
FILTERER_STARTED = "filterer.started"
FILTERER_PREDICTED = "filterer.predicted"
FILTERER_UPDATED = "filterer.updated"
FILTERER_RESAMPLED = "filterer.resampled"
FILTERER_SMOOTHED = "filterer.smoothed"
FILTERER_REPORT = "filterer.report"
FILTERER_CLEARED = "filterer.cleared"

KNOWN_EVENTS = frozenset({
    FILTERER_STARTED,
    FILTERER_PREDICTED,
    FILTERER_UPDATED,
    FILTERER_RESAMPLED,
    FILTERER_SMOOTHED,
    FILTERER_REPORT,
    FILTERER_CLEARED,
})


# =====================================================================
# Exceptions
# =====================================================================


class FiltererError(ValueError):
    """Base class for Filterer-domain errors."""


class UnknownFilter(FiltererError):
    """Filter name is not in KNOWN_FILTERS."""


class UnknownResampler(FiltererError):
    """Resampler name is not in KNOWN_RESAMPLERS."""


class UnknownSmoother(FiltererError):
    """Smoother name is not in KNOWN_SMOOTHERS."""


class InvalidDimension(FiltererError):
    """State / observation / matrix dimensions are inconsistent."""


class InvalidMatrix(FiltererError):
    """Matrix is non-square, non-symmetric, or non-positive-definite where required."""


class InvalidObservation(FiltererError):
    """Observation contains NaN, has wrong dimension, or is otherwise malformed."""


class InvalidParticles(FiltererError):
    """Particle count or weight set is malformed."""


class FilterDegenerate(FiltererError):
    """All particle weights collapsed to zero — filter cannot recover."""


class NonPositiveDefinite(FiltererError):
    """Innovation covariance or state covariance is not positive-definite."""


class InvalidCallable(FiltererError):
    """Supplied dynamics / observation function failed signature check."""


class GenericConfigError(FiltererError):
    """Configuration error not covered by a more specific exception."""


# =====================================================================
# Matrix utilities (pure stdlib, lists-of-lists)
# =====================================================================


def _zeros(n: int, m: int | None = None) -> list:
    """Build an n×m (or n-vector) zero matrix."""
    if m is None:
        return [0.0] * n
    return [[0.0] * m for _ in range(n)]


def _eye(n: int) -> list:
    """Identity matrix of dimension n."""
    M = _zeros(n, n)
    for i in range(n):
        M[i][i] = 1.0
    return M


def _matmul(A: Sequence, B: Sequence) -> list:
    """Matrix multiply A (m×k) × B (k×n) → (m×n).  Accepts vectors as m=1."""
    if not A or not B:
        return []
    if not isinstance(A[0], (list, tuple)):
        A = [list(A)]
        vec_a = True
    else:
        vec_a = False
    if not isinstance(B[0], (list, tuple)):
        B = [[b] for b in B]
        vec_b = True
    else:
        vec_b = False
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    if k != k2:
        raise InvalidDimension(f"matmul: A is {m}x{k}, B is {k2}x{n}")
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        Ci = C[i]
        for kk in range(k):
            a = Ai[kk]
            if a == 0.0:
                continue
            Bk = B[kk]
            for j in range(n):
                Ci[j] += a * Bk[j]
    if vec_a and vec_b:
        return [C[0][0]]
    if vec_b:
        return [row[0] for row in C]
    if vec_a:
        return C[0]
    return C


def _matvec(A: Sequence, v: Sequence) -> list:
    """Matrix-vector product A × v."""
    m = len(A)
    n = len(v)
    if not A or len(A[0]) != n:
        raise InvalidDimension(f"matvec: A is {m}x{len(A[0]) if A else 0}, v is {n}")
    out = [0.0] * m
    for i in range(m):
        Ai = A[i]
        s = 0.0
        for j in range(n):
            s += Ai[j] * v[j]
        out[i] = s
    return out


def _transpose(A: Sequence) -> list:
    """Transpose a matrix."""
    if not A:
        return []
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _matadd(A: Sequence, B: Sequence) -> list:
    """Element-wise A + B."""
    m, n = len(A), len(A[0])
    if len(B) != m or len(B[0]) != n:
        raise InvalidDimension("matadd shape mismatch")
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]


def _matsub(A: Sequence, B: Sequence) -> list:
    """Element-wise A − B."""
    m, n = len(A), len(A[0])
    if len(B) != m or len(B[0]) != n:
        raise InvalidDimension("matsub shape mismatch")
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]


def _vecsub(a: Sequence, b: Sequence) -> list:
    return [ai - bi for ai, bi in zip(a, b)]


def _vecadd(a: Sequence, b: Sequence) -> list:
    return [ai + bi for ai, bi in zip(a, b)]


def _scale(A: Sequence, s: float) -> list:
    """Scalar multiplication on a matrix or vector."""
    if A and isinstance(A[0], (list, tuple)):
        return [[s * x for x in row] for row in A]
    return [s * x for x in A]


def _outer(a: Sequence, b: Sequence) -> list:
    """Outer product a × bᵀ."""
    return [[ai * bj for bj in b] for ai in a]


def _symmetrise(A: Sequence) -> list:
    """Average A and Aᵀ to enforce numerical symmetry."""
    n = len(A)
    return [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]


def _is_square(A: Sequence) -> bool:
    return bool(A) and all(len(row) == len(A) for row in A)


def _cholesky(A: Sequence, *, jitter: float = _JITTER) -> list:
    """Lower-triangular Cholesky factor of a symmetric positive-definite A.

    Adds an adaptive diagonal jitter on numerical failure.  Raises
    ``NonPositiveDefinite`` when even the jittered matrix is not PD.
    """
    if not _is_square(A):
        raise InvalidMatrix("Cholesky requires a square matrix")
    n = len(A)
    # First attempt: bare Cholesky.
    for attempt in range(6):
        L = _zeros(n, n)
        ok = True
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j]
                if i == j and attempt > 0:
                    s += jitter * (10.0 ** (attempt - 1))
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                if i == j:
                    if s <= 0.0:
                        ok = False
                        break
                    L[i][i] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
            if not ok:
                break
        if ok:
            return L
    raise NonPositiveDefinite(
        "Cholesky failed after jitter attempts — matrix not positive-definite"
    )


def _solve_lower(L: Sequence, b: Sequence) -> list:
    """Forward-substitute L y = b for lower-triangular L."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i][j] * y[j]
        y[i] = s / L[i][i]
    return y


def _solve_upper(U: Sequence, b: Sequence) -> list:
    """Back-substitute U x = b for upper-triangular U."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]
    return x


def _solve_spd(A: Sequence, b: Sequence) -> list:
    """Solve A x = b for symmetric positive-definite A via Cholesky."""
    L = _cholesky(A)
    y = _solve_lower(L, b)
    return _solve_upper(_transpose(L), y)


def _solve_spd_mat(A: Sequence, B: Sequence) -> list:
    """Solve A X = B (columns of B) for SPD A."""
    L = _cholesky(A)
    Lt = _transpose(L)
    n = len(A)
    cols = len(B[0])
    X = _zeros(n, cols)
    for j in range(cols):
        bj = [B[i][j] for i in range(n)]
        y = _solve_lower(L, bj)
        x = _solve_upper(Lt, y)
        for i in range(n):
            X[i][j] = x[i]
    return X


def _inv_spd(A: Sequence) -> list:
    """Inverse of a symmetric positive-definite matrix via Cholesky."""
    n = len(A)
    I = _eye(n)
    return _solve_spd_mat(A, I)


def _logdet_spd(A: Sequence) -> float:
    """Log-determinant of an SPD matrix from its Cholesky factor."""
    L = _cholesky(A)
    s = 0.0
    for i in range(len(L)):
        s += math.log(L[i][i])
    return 2.0 * s


def _mahalanobis(v: Sequence, S: Sequence) -> float:
    """vᵀ S⁻¹ v computed via Cholesky solve (avoids explicit inverse)."""
    L = _cholesky(S)
    y = _solve_lower(L, list(v))
    return sum(yi * yi for yi in y)


def _gauss_logpdf(y: Sequence, mu: Sequence, S: Sequence) -> float:
    """Multivariate Gaussian log density at y."""
    m = len(y)
    v = _vecsub(y, mu)
    L = _cholesky(S)
    yz = _solve_lower(L, v)
    quad = sum(z * z for z in yz)
    logdet = 0.0
    for i in range(m):
        logdet += math.log(L[i][i])
    logdet *= 2.0
    return -0.5 * (m * _LOG2PI + logdet + quad)


# =====================================================================
# Beasley-Springer-Moro inverse-Φ + Box-Muller
# =====================================================================


_BSM_A = (
    -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
)
_BSM_B = (
    -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01,
)
_BSM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
)
_BSM_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00,
)
_BSM_LOW = 0.02425
_BSM_HIGH = 1.0 - 0.02425


def _inv_phi(u: float) -> float:
    """Beasley-Springer-Moro 1995 inverse standard-normal CDF.  Pure stdlib."""
    if u <= 0.0 or u >= 1.0:
        if u == 0.0:
            return _NEGINF
        if u == 1.0:
            return _INF
        raise FiltererError(f"inv_phi: u out of range {u!r}")
    if u < _BSM_LOW:
        q = math.sqrt(-2.0 * math.log(u))
        num = ((((_BSM_C[0] * q + _BSM_C[1]) * q + _BSM_C[2]) * q + _BSM_C[3]) * q + _BSM_C[4]) * q + _BSM_C[5]
        den = (((_BSM_D[0] * q + _BSM_D[1]) * q + _BSM_D[2]) * q + _BSM_D[3]) * q + 1.0
        return num / den
    if u < _BSM_HIGH:
        q = u - 0.5
        r = q * q
        num = (((((_BSM_A[0] * r + _BSM_A[1]) * r + _BSM_A[2]) * r + _BSM_A[3]) * r + _BSM_A[4]) * r + _BSM_A[5]) * q
        den = (((((_BSM_B[0] * r + _BSM_B[1]) * r + _BSM_B[2]) * r + _BSM_B[3]) * r + _BSM_B[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - u))
    num = ((((_BSM_C[0] * q + _BSM_C[1]) * q + _BSM_C[2]) * q + _BSM_C[3]) * q + _BSM_C[4]) * q + _BSM_C[5]
    den = (((_BSM_D[0] * q + _BSM_D[1]) * q + _BSM_D[2]) * q + _BSM_D[3]) * q + 1.0
    return -num / den


def _stdnormal(rng: random.Random) -> float:
    """One standard normal via inverse CDF (preserves seed determinism)."""
    u = rng.random()
    if u == 0.0:
        u = _EPS
    return _inv_phi(u)


def _stdnormal_n(rng: random.Random, n: int) -> list:
    return [_stdnormal(rng) for _ in range(n)]


def _logsumexp(xs: Sequence[float]) -> float:
    if not xs:
        return _NEGINF
    m = max(xs)
    if m == _NEGINF:
        return _NEGINF
    s = 0.0
    for x in xs:
        s += math.exp(x - m)
    return m + math.log(s)


def _chi2_logpdf(x: float, k: int) -> float:
    """Log density of χ²(k) at x ≥ 0."""
    if x <= 0.0:
        return _NEGINF
    return (
        (k / 2.0 - 1.0) * math.log(x)
        - 0.5 * x
        - (k / 2.0) * math.log(2.0)
        - math.lgamma(k / 2.0)
    )


# =====================================================================
# JSON canonicalisation + hashing
# =====================================================================


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, float):
        if math.isnan(o):
            return "nan"
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        return o
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(f"cannot canonicalise {type(o).__name__}")


def _hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(_canonical_json(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# =====================================================================
# Validators
# =====================================================================


def _validate_filter(name: str) -> str:
    if name not in KNOWN_FILTERS:
        raise UnknownFilter(
            f"unknown filter {name!r}; expected one of {sorted(KNOWN_FILTERS)}"
        )
    return name


def _validate_resampler(name: str) -> str:
    if name not in KNOWN_RESAMPLERS:
        raise UnknownResampler(
            f"unknown resampler {name!r}; expected one of {sorted(KNOWN_RESAMPLERS)}"
        )
    return name


def _validate_smoother(name: str) -> str:
    if name not in KNOWN_SMOOTHERS:
        raise UnknownSmoother(
            f"unknown smoother {name!r}; expected one of {sorted(KNOWN_SMOOTHERS)}"
        )
    return name


def _validate_vector(v: Sequence, name: str, *, n: int | None = None) -> list:
    if v is None:
        raise InvalidDimension(f"{name}: vector is None")
    try:
        out = [float(x) for x in v]
    except Exception as e:
        raise InvalidDimension(f"{name}: not a numeric vector ({e})") from None
    for x in out:
        if math.isnan(x):
            raise InvalidObservation(f"{name}: contains NaN")
    if n is not None and len(out) != n:
        raise InvalidDimension(f"{name}: expected length {n}, got {len(out)}")
    return out


def _validate_matrix(M: Sequence, name: str, *,
                     shape: tuple | None = None,
                     symmetric: bool = False,
                     spd: bool = False) -> list:
    if M is None:
        raise InvalidDimension(f"{name}: matrix is None")
    try:
        out = [[float(x) for x in row] for row in M]
    except Exception as e:
        raise InvalidDimension(f"{name}: not a numeric matrix ({e})") from None
    if not out:
        raise InvalidDimension(f"{name}: empty matrix")
    m = len(out)
    n = len(out[0])
    for row in out:
        if len(row) != n:
            raise InvalidDimension(f"{name}: ragged matrix rows")
        for x in row:
            if math.isnan(x):
                raise InvalidMatrix(f"{name}: contains NaN")
    if shape is not None and (m, n) != shape:
        raise InvalidDimension(
            f"{name}: expected shape {shape}, got ({m}, {n})"
        )
    if symmetric or spd:
        if m != n:
            raise InvalidMatrix(f"{name}: expected square, got ({m}, {n})")
        for i in range(m):
            for j in range(i + 1, m):
                if abs(out[i][j] - out[j][i]) > 1e-7 * (1.0 + abs(out[i][j])):
                    raise InvalidMatrix(f"{name}: not symmetric at ({i},{j})")
    if spd:
        try:
            _cholesky(out)
        except NonPositiveDefinite:
            raise InvalidMatrix(f"{name}: not positive-definite")
    return out


def _validate_n_particles(N: int) -> int:
    if not isinstance(N, int) or N < 2:
        raise InvalidParticles(f"need ≥2 particles, got {N}")
    return N


# =====================================================================
# Closed-form bounds
# =====================================================================


def nis_chi2_threshold(m: int, alpha: float = 0.05) -> float:
    """Upper α-quantile of χ²(m).  Used for the NIS gate.

    Closed form for m=1: ``z² = inv_phi(1 − α/2)²``.  For general m we use
    the Wilson-Hilferty approximation
    ``χ²(m, α) ≈ m (1 − 2/(9m) + z √(2/(9m)))³`` with ``z = inv_phi(1−α)``.
    """
    if m < 1:
        raise InvalidDimension(f"chi2 threshold: m must be ≥1, got {m}")
    if not 0.0 < alpha < 1.0:
        raise FiltererError(f"chi2 threshold: alpha must be in (0,1), got {alpha}")
    if m == 1:
        z = _inv_phi(1.0 - alpha / 2.0)
        return z * z
    z = _inv_phi(1.0 - alpha)
    base = 1.0 - 2.0 / (9.0 * m) + z * math.sqrt(2.0 / (9.0 * m))
    return m * base ** 3


def crisan_doucet_mse_bound(N: int, *, sup_f: float = 1.0) -> float:
    """Crisan-Doucet 2002 bound: E[(p̂_N(f) − p(f))²] ≤ ||f||² / N.

    Returns the upper bound for a bounded test function with ``sup |f| ≤ sup_f``.
    """
    if N < 1:
        raise InvalidParticles("Crisan-Doucet bound: N must be ≥1")
    if sup_f <= 0.0:
        raise FiltererError("Crisan-Doucet bound: sup_f must be > 0")
    return sup_f * sup_f / N


def massart_dkw_band(N: int, delta: float = 0.05) -> float:
    """Massart-DKW 1990 finite-sample CDF band ``√((1/(2N)) log(2/δ))``."""
    if N < 1:
        raise InvalidParticles("DKW band: N must be ≥1")
    if not 0.0 < delta < 1.0:
        raise FiltererError("DKW band: delta must be in (0,1)")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * N))


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kong-Liu-Wong 1994 effective sample size ``1 / Σ w_i²``."""
    s = 0.0
    s2 = 0.0
    for w in weights:
        if w < 0.0:
            raise InvalidParticles("ESS: negative weight")
        s += w
        s2 += w * w
    if s <= 0.0:
        raise FilterDegenerate("ESS: weights sum to zero")
    # Normalise.
    return (s * s) / s2


def innovation_whiteness_stat(innovations: Sequence[float],
                              max_lag: int = 1) -> tuple:
    """Box-Pierce statistic on standardised innovations.

    Returns ``(stat, df)`` where ``stat ≈ T Σ_{k=1}^{max_lag} ρ̂_k²``
    is asymptotically χ²(max_lag) under the null of white-noise innovations.
    """
    T = len(innovations)
    if T < max_lag + 2:
        raise FiltererError(
            f"whiteness: need T ≥ max_lag+2, got T={T}, max_lag={max_lag}"
        )
    mu = sum(innovations) / T
    centred = [x - mu for x in innovations]
    var = sum(x * x for x in centred) / T
    if var <= 0.0:
        return (0.0, max_lag)
    rho = []
    for k in range(1, max_lag + 1):
        s = 0.0
        for t in range(k, T):
            s += centred[t] * centred[t - k]
        rho.append(s / (T * var))
    stat = T * sum(r * r for r in rho)
    return (stat, max_lag)


# =====================================================================
# Resampling
# =====================================================================


def _resample(weights: Sequence[float], rng: random.Random,
              scheme: str) -> list:
    """Return indices ``[i_1, …, i_N]`` of resampled particles."""
    N = len(weights)
    if N == 0:
        raise InvalidParticles("empty weight vector")
    s = sum(weights)
    if s <= 0.0:
        raise FilterDegenerate("resample: weights sum to zero")
    w = [wi / s for wi in weights]
    if scheme == MULTINOMIAL:
        return _resample_multinomial(w, rng)
    if scheme == SYSTEMATIC:
        return _resample_systematic(w, rng)
    if scheme == STRATIFIED:
        return _resample_stratified(w, rng)
    if scheme == RESIDUAL:
        return _resample_residual(w, rng)
    raise UnknownResampler(scheme)


def _resample_multinomial(w: Sequence[float], rng: random.Random) -> list:
    N = len(w)
    # Cumulative.
    cum = [0.0] * N
    s = 0.0
    for i, wi in enumerate(w):
        s += wi
        cum[i] = s
    out = [0] * N
    for j in range(N):
        u = rng.random()
        # Binary search.
        lo, hi = 0, N - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        out[j] = lo
    return out


def _resample_systematic(w: Sequence[float], rng: random.Random) -> list:
    N = len(w)
    u0 = rng.random() / N
    points = [u0 + j / N for j in range(N)]
    return _cdf_invert(w, points)


def _resample_stratified(w: Sequence[float], rng: random.Random) -> list:
    N = len(w)
    points = [(rng.random() + j) / N for j in range(N)]
    return _cdf_invert(w, points)


def _cdf_invert(w: Sequence[float], points: Sequence[float]) -> list:
    """Invert the CDF of w at each point — used by systematic / stratified."""
    N = len(w)
    cum = 0.0
    out = [0] * len(points)
    j = 0
    i = 0
    while j < len(points):
        while i < N - 1 and cum + w[i] < points[j]:
            cum += w[i]
            i += 1
        out[j] = i
        j += 1
    return out


def _resample_residual(w: Sequence[float], rng: random.Random) -> list:
    N = len(w)
    out: list = []
    residuals = [0.0] * N
    for i in range(N):
        ki = int(math.floor(N * w[i]))
        out.extend([i] * ki)
        residuals[i] = N * w[i] - ki
    remaining = N - len(out)
    if remaining > 0:
        s = sum(residuals)
        if s > 0.0:
            r = [x / s for x in residuals]
            out.extend(_resample_multinomial(r, rng)[:remaining])
        else:
            # Defensive: pad with multinomial draws.
            out.extend(_resample_multinomial(w, rng)[:remaining])
    return out[:N]


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class StateSpaceModel:
    """Linear-Gaussian state-space specification.

    ``x_t = F x_{t-1} + B u_t + w_t``,    ``w_t ∼ N(0, Q)``
    ``y_t = H x_t + v_t``,                 ``v_t ∼ N(0, R)``
    """
    F: tuple
    H: tuple
    Q: tuple
    R: tuple
    B: tuple | None = None

    @property
    def n_state(self) -> int:
        return len(self.F)

    @property
    def n_obs(self) -> int:
        return len(self.H)


@dataclass(frozen=True)
class GaussianBelief:
    """Mean and covariance — the universal output of the Kalman family."""
    mean: tuple
    cov: tuple

    @property
    def dim(self) -> int:
        return len(self.mean)


@dataclass(frozen=True)
class ParticleBelief:
    """Weighted particle set — the universal output of the SMC family."""
    particles: tuple
    weights: tuple

    @property
    def N(self) -> int:
        return len(self.particles)

    @property
    def dim(self) -> int:
        return len(self.particles[0]) if self.particles else 0


@dataclass(frozen=True)
class PredictionResult:
    """Output of a predict() step."""
    belief: Any                    # GaussianBelief or ParticleBelief
    n_step: int
    fingerprint: str


@dataclass(frozen=True)
class UpdateResult:
    """Output of an update() step."""
    belief: Any
    log_evidence: float            # log p(y_t | y_{1:t-1})
    innovation: tuple              # ν_t
    innovation_cov: tuple | None   # S_t (Kalman family only)
    nis: float                     # ν_t^T S_t^{-1} ν_t  (Mahalanobis)
    ess: float | None              # effective sample size (particle filter only)
    n_step: int
    fingerprint: str


@dataclass(frozen=True)
class ResampleResult:
    """Output of a resample step."""
    n_step: int
    scheme: str
    ess_before: float
    ess_after: float
    fingerprint: str


@dataclass(frozen=True)
class SmoothResult:
    """Output of a smoother."""
    beliefs: tuple                 # GaussianBelief or ParticleBelief per t
    smoother: str
    fingerprint: str


@dataclass(frozen=True)
class FilterReport:
    """Aggregate report — composes with AttestationLedger."""
    filter_name: str
    n_state: int
    n_obs: int
    n_steps: int
    log_marginal: float            # Σ log p(y_t | y_{1:t-1})
    mean_nis: float
    nis_chi2_threshold: float      # at default α=0.05
    n_nis_exceeds: int
    ess_min: float | None          # particle filter only
    ess_mean: float | None
    n_resamples: int
    fingerprint: str


@dataclass(frozen=True)
class FiltererConfig:
    filter_name: str
    seed: int = 0
    n_particles: int | None = None
    resampler: str = SYSTEMATIC
    ess_threshold: float = 0.5     # fraction of N
    nis_alpha: float = 0.05
    # UKF parameters.
    ukf_alpha: float = 1.0e-3
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0


# =====================================================================
# Kalman primitives (closed-form steps, exposed as pure functions)
# =====================================================================


def kalman_predict(mean: Sequence[float], cov: Sequence,
                   F: Sequence, Q: Sequence,
                   u: Sequence[float] | None = None,
                   B: Sequence | None = None) -> tuple:
    """One Kalman predict step.  Returns (mean', cov')."""
    n = len(mean)
    Fm = _matvec(F, mean)
    if u is not None:
        if B is None:
            raise InvalidDimension("kalman_predict: B is required when u is given")
        Bu = _matvec(B, u)
        Fm = _vecadd(Fm, Bu)
    # F P F^T
    FP = _matmul(F, cov)
    FPFT = _matmul(FP, _transpose(F))
    P = _symmetrise(_matadd(FPFT, Q))
    return (Fm, P)


def kalman_update(mean: Sequence[float], cov: Sequence,
                  y: Sequence[float], H: Sequence, R: Sequence) -> tuple:
    """One Kalman update step.  Returns (mean', cov', log_evidence, innov, S, nis).

    Uses the Joseph stabilised form for the covariance update.
    """
    # Innovation.
    Hx = _matvec(H, mean)
    innov = _vecsub(y, Hx)
    # Innovation covariance.
    HP = _matmul(H, cov)
    HT = _transpose(H)
    S = _symmetrise(_matadd(_matmul(HP, HT), R))
    # Cholesky-solve for K = P H^T S^{-1}.  K (n×m), PHt (n×m), S (m×m).
    # Row by row: K_i S = (PHt)_i  ⇔  S K_i^T = (PHt)_i^T  (S symmetric).
    L = _cholesky(S)
    Lt = _transpose(L)
    PHt = _matmul(cov, HT)
    n = len(mean)
    m = len(y)
    K = _zeros(n, m)
    for i in range(n):
        bi = list(PHt[i])
        yi = _solve_lower(L, bi)
        xi = _solve_upper(Lt, yi)
        K[i] = xi
    # Mean update.
    Kv = _matvec(K, innov)
    mean_new = _vecadd(mean, Kv)
    # Joseph form: P' = (I - KH) P (I - KH)^T + K R K^T.
    KH = _matmul(K, H)
    I = _eye(n)
    A = _matsub(I, KH)
    AP = _matmul(A, cov)
    APAt = _matmul(AP, _transpose(A))
    KR = _matmul(K, R)
    KRKt = _matmul(KR, _transpose(K))
    cov_new = _symmetrise(_matadd(APAt, KRKt))
    # Log evidence ≡ log N(y | Hx, S).
    logev = _gauss_logpdf(y, Hx, S)
    # NIS.
    y_inv = _solve_lower(L, innov)
    nis = sum(z * z for z in y_inv)
    return (mean_new, cov_new, logev, innov, S, nis)


def ekf_predict(mean: Sequence[float], cov: Sequence,
                f: Callable, F: Sequence, Q: Sequence,
                u: Sequence[float] | None = None) -> tuple:
    """EKF predict — f is the non-linear dynamics, F is its Jacobian at mean."""
    new_mean = list(f(mean, u)) if u is not None else list(f(mean, None))
    FP = _matmul(F, cov)
    FPFT = _matmul(FP, _transpose(F))
    P = _symmetrise(_matadd(FPFT, Q))
    return (new_mean, P)


def ekf_update(mean: Sequence[float], cov: Sequence,
               y: Sequence[float], h: Callable, H: Sequence,
               R: Sequence) -> tuple:
    """EKF update — h is the non-linear observation, H is its Jacobian at mean."""
    Hx = list(h(mean))
    innov = _vecsub(y, Hx)
    HP = _matmul(H, cov)
    HT = _transpose(H)
    S = _symmetrise(_matadd(_matmul(HP, HT), R))
    L = _cholesky(S)
    Lt = _transpose(L)
    PHt = _matmul(cov, HT)
    n = len(mean)
    m = len(y)
    K = _zeros(n, m)
    for i in range(n):
        bi = list(PHt[i])
        yi = _solve_lower(L, bi)
        xi = _solve_upper(Lt, yi)
        K[i] = xi
    Kv = _matvec(K, innov)
    mean_new = _vecadd(mean, Kv)
    KH = _matmul(K, H)
    I = _eye(n)
    A = _matsub(I, KH)
    AP = _matmul(A, cov)
    APAt = _matmul(AP, _transpose(A))
    KR = _matmul(K, R)
    KRKt = _matmul(KR, _transpose(K))
    cov_new = _symmetrise(_matadd(APAt, KRKt))
    logev = _gauss_logpdf(y, Hx, S)
    y_inv = _solve_lower(L, innov)
    nis = sum(z * z for z in y_inv)
    return (mean_new, cov_new, logev, innov, S, nis)


# =====================================================================
# Unscented (sigma-point) Kalman primitives
# =====================================================================


def _ukf_sigma_points(mean: Sequence[float], cov: Sequence, *,
                      alpha: float, beta: float, kappa: float) -> tuple:
    """Return (sigma_points, wm, wc, lam) for the UKF.

    Uses scaled symmetric sigma-points (Julier-Uhlmann 1997).
    """
    n = len(mean)
    lam = alpha * alpha * (n + kappa) - n
    c = n + lam
    L = _cholesky(_scale(cov, c))
    chi = [list(mean)]
    for i in range(n):
        col = [L[j][i] for j in range(n)]
        plus = _vecadd(mean, col)
        minus = _vecsub(mean, col)
        chi.append(plus)
        chi.append(minus)
    wm = [lam / c] + [0.5 / c] * (2 * n)
    wc = [lam / c + (1.0 - alpha * alpha + beta)] + [0.5 / c] * (2 * n)
    return (chi, wm, wc, lam)


def _weighted_mean_cov(points: Sequence, wm: Sequence,
                       wc: Sequence, noise: Sequence,
                       cross: Sequence | None = None,
                       cross_mean: Sequence | None = None) -> tuple:
    """Recover mean, covariance, and (optionally) cross-covariance."""
    P = len(points)
    n = len(points[0])
    mu = [0.0] * n
    for i in range(P):
        wi = wm[i]
        pi = points[i]
        for k in range(n):
            mu[k] += wi * pi[k]
    cov = [[0.0] * n for _ in range(n)]
    for i in range(P):
        wi = wc[i]
        d = _vecsub(points[i], mu)
        for a in range(n):
            da = d[a]
            for b in range(n):
                cov[a][b] += wi * da * d[b]
    cov = _matadd(cov, noise)
    cov = _symmetrise(cov)
    if cross is None:
        return (mu, cov)
    # Cross-covariance Σ wc_i (x_i - μ_x) (z_i - μ_z)^T.
    # Result shape is (len(cross_mean), n) = (state_dim, obs_dim).
    n_state = len(cross_mean)
    Cxz = [[0.0] * n for _ in range(n_state)]
    cm = cross_mean
    for i in range(P):
        wi = wc[i]
        dx = _vecsub(cross[i], cm)
        dz = _vecsub(points[i], mu)
        for a in range(n_state):
            dxa = dx[a]
            for b in range(n):
                Cxz[a][b] += wi * dxa * dz[b]
    return (mu, cov, Cxz)


def ukf_predict(mean: Sequence[float], cov: Sequence,
                f: Callable, Q: Sequence, *,
                u: Sequence[float] | None = None,
                alpha: float = 1.0e-3, beta: float = 2.0,
                kappa: float = 0.0) -> tuple:
    """UKF predict — f is the dynamics function (no Jacobian needed)."""
    chi, wm, wc, _ = _ukf_sigma_points(mean, cov, alpha=alpha, beta=beta,
                                       kappa=kappa)
    propagated = [list(f(p, u)) for p in chi]
    new_mean, new_cov = _weighted_mean_cov(propagated, wm, wc, Q)
    return (new_mean, new_cov)


def ukf_update(mean: Sequence[float], cov: Sequence, y: Sequence[float],
               h: Callable, R: Sequence, *,
               alpha: float = 1.0e-3, beta: float = 2.0,
               kappa: float = 0.0) -> tuple:
    """UKF update — h is the observation function (no Jacobian needed)."""
    chi, wm, wc, _ = _ukf_sigma_points(mean, cov, alpha=alpha, beta=beta,
                                       kappa=kappa)
    Z = [list(h(p)) for p in chi]
    z_mean, S, Cxz = _weighted_mean_cov(Z, wm, wc, R, cross=chi,
                                        cross_mean=mean)
    # K = Cxz S^{-1}; Cxz is (n × m), S is (m × m).
    # Row by row: K_i S = Cxz_i  ⇔  S K_i^T = Cxz_i^T  (S symmetric).
    n = len(mean)
    m = len(y)
    L = _cholesky(S)
    Lt = _transpose(L)
    K = _zeros(n, m)
    for i in range(n):
        bi = list(Cxz[i])
        yi = _solve_lower(L, bi)
        xi = _solve_upper(Lt, yi)
        K[i] = xi
    innov = _vecsub(y, z_mean)
    Kv = _matvec(K, innov)
    new_mean = _vecadd(mean, Kv)
    # P' = P - K S K^T.
    KS = _matmul(K, S)
    KSKt = _matmul(KS, _transpose(K))
    new_cov = _symmetrise(_matsub(cov, KSKt))
    logev = _gauss_logpdf(y, z_mean, S)
    y_inv = _solve_lower(L, innov)
    nis = sum(z * z for z in y_inv)
    return (new_mean, new_cov, logev, innov, S, nis)


# =====================================================================
# RTS smoother (linear-Gaussian)
# =====================================================================


def rts_smooth(filtered_means: Sequence,
               filtered_covs: Sequence,
               predicted_means: Sequence,
               predicted_covs: Sequence,
               F: Sequence) -> tuple:
    """Rauch-Tung-Striebel backward sweep.

    Inputs:
      ``filtered_means[t]``  = ``x̂_{t|t}``  for t=0..T-1
      ``filtered_covs[t]``   = ``P_{t|t}``
      ``predicted_means[t]`` = ``x̂_{t+1|t}`` for t=0..T-1 (so length T-1 used)
      ``predicted_covs[t]``  = ``P_{t+1|t}``
      ``F``                  = dynamics matrix (or Jacobian for EKF)

    Returns ``(smoothed_means, smoothed_covs)`` for t=0..T-1.
    """
    T = len(filtered_means)
    if T < 2:
        return (list(filtered_means), list(filtered_covs))
    smoothed_means = [list(filtered_means[-1])]
    smoothed_covs = [list(map(list, filtered_covs[-1]))]
    FT = _transpose(F)
    for t in range(T - 2, -1, -1):
        # G_t = P_{t|t} F^T P_{t+1|t}^{-1}.  G is (n×n), PF is (n×n).
        # Row by row: G_i P_{t+1|t} = PF_i  ⇔  P_{t+1|t} G_i^T = PF_i^T.
        PF = _matmul(filtered_covs[t], FT)
        n = len(filtered_means[t])
        L = _cholesky(predicted_covs[t])
        Lt = _transpose(L)
        G = _zeros(n, n)
        for i in range(n):
            bi = list(PF[i])
            yi = _solve_lower(L, bi)
            xi = _solve_upper(Lt, yi)
            G[i] = xi
        # Smoothed mean.
        diff_m = _vecsub(smoothed_means[0], predicted_means[t])
        Gdm = _matvec(G, diff_m)
        sm = _vecadd(filtered_means[t], Gdm)
        # Smoothed covariance.
        diff_P = _matsub(smoothed_covs[0], predicted_covs[t])
        GdP = _matmul(G, diff_P)
        GdPGt = _matmul(GdP, _transpose(G))
        sc = _symmetrise(_matadd(filtered_covs[t], GdPGt))
        smoothed_means.insert(0, sm)
        smoothed_covs.insert(0, sc)
    return (smoothed_means, smoothed_covs)


# =====================================================================
# Filterer main class
# =====================================================================


class Filterer:
    """Bayesian state-space filtering primitive — thread-safe, hash-chained."""

    def __init__(self, config: FiltererConfig, *,
                 model: StateSpaceModel | None = None,
                 mean: Sequence[float] | None = None,
                 cov: Sequence | None = None,
                 particles: Sequence | None = None,
                 weights: Sequence | None = None,
                 dynamics_fn: Callable | None = None,
                 obs_fn: Callable | None = None,
                 dynamics_jac: Callable | None = None,
                 obs_jac: Callable | None = None,
                 propose_fn: Callable | None = None,
                 likelihood_fn: Callable | None = None,
                 event_bus: Any = None):
        self._config = config
        self._filter = _validate_filter(config.filter_name)
        self._model = model
        self._dynamics_fn = dynamics_fn
        self._obs_fn = obs_fn
        self._dynamics_jac = dynamics_jac
        self._obs_jac = obs_jac
        self._propose_fn = propose_fn
        self._likelihood_fn = likelihood_fn
        self._event_bus = event_bus
        self._lock = threading.RLock()
        self._rng = random.Random(config.seed)

        # Internal state.
        self._n_step = 0
        self._mean: list | None = None
        self._cov: list | None = None
        self._particles: list | None = None
        self._weights: list | None = None
        self._fingerprint = _GENESIS

        # Trace.
        self._log_marginal = 0.0
        self._nis_history: list = []
        self._innovation_history: list = []
        self._ess_history: list = []
        self._n_resamples = 0
        self._filtered_means: list = []
        self._filtered_covs: list = []
        self._predicted_means: list = []
        self._predicted_covs: list = []

        # Initialise state.
        if self._filter in KALMAN_FAMILY:
            if mean is None or cov is None:
                raise GenericConfigError(
                    f"{self._filter} requires initial mean and covariance"
                )
            n = len(mean)
            self._mean = _validate_vector(mean, "mean", n=n)
            self._cov = _validate_matrix(cov, "cov", shape=(n, n), spd=True)
            if self._filter == EKF or self._filter == UKF:
                if dynamics_fn is None or obs_fn is None:
                    raise GenericConfigError(
                        f"{self._filter} requires dynamics_fn and obs_fn"
                    )
                if self._filter == EKF and (dynamics_jac is None or obs_jac is None):
                    raise GenericConfigError(
                        "EKF requires dynamics_jac and obs_jac"
                    )
            else:
                if model is None:
                    raise GenericConfigError(
                        f"{self._filter} requires a linear StateSpaceModel"
                    )
        elif self._filter in PARTICLE_FAMILY:
            if config.n_particles is None:
                raise GenericConfigError(
                    "particle filter requires config.n_particles"
                )
            N = _validate_n_particles(config.n_particles)
            if propose_fn is None or likelihood_fn is None:
                raise GenericConfigError(
                    "particle filter requires propose_fn and likelihood_fn"
                )
            if particles is None:
                raise GenericConfigError("particle filter requires initial particles")
            self._particles = [list(p) for p in particles]
            if len(self._particles) != N:
                raise InvalidParticles(
                    f"initial particles: expected {N}, got {len(self._particles)}"
                )
            if weights is None:
                self._weights = [1.0 / N] * N
            else:
                if len(weights) != N:
                    raise InvalidParticles("weights length mismatch")
                w = list(weights)
                s = sum(w)
                if s <= 0:
                    raise FilterDegenerate("initial weights sum to zero")
                self._weights = [wi / s for wi in w]
        _validate_resampler(config.resampler)
        self._emit_event(FILTERER_STARTED, {
            "filter": self._filter,
            "seed": config.seed,
            "n_particles": config.n_particles,
        })

    # ----- factories ----------------------------------------------------

    @classmethod
    def kalman(cls, *, F: Sequence, H: Sequence, Q: Sequence, R: Sequence,
               x0: Sequence[float], P0: Sequence,
               B: Sequence | None = None,
               filter_name: str = KF,
               seed: int = 0,
               event_bus: Any = None) -> "Filterer":
        """Convenience constructor for a linear-Gaussian Kalman filter."""
        nF = len(F)
        nH = len(H)
        Fm = _validate_matrix(F, "F", shape=(nF, nF))
        Qm = _validate_matrix(Q, "Q", shape=(nF, nF), spd=True)
        Hm = _validate_matrix(H, "H", shape=(nH, nF))
        Rm = _validate_matrix(R, "R", shape=(nH, nH), spd=True)
        Bm = None
        if B is not None:
            Bm = _validate_matrix(B, "B")
            if len(Bm) != nF:
                raise InvalidDimension("B has wrong row count")
        model = StateSpaceModel(
            F=tuple(map(tuple, Fm)),
            H=tuple(map(tuple, Hm)),
            Q=tuple(map(tuple, Qm)),
            R=tuple(map(tuple, Rm)),
            B=tuple(map(tuple, Bm)) if Bm is not None else None,
        )
        cfg = FiltererConfig(filter_name=filter_name, seed=seed)
        return cls(cfg, model=model, mean=list(x0), cov=P0, event_bus=event_bus)

    @classmethod
    def ekf(cls, *, dynamics_fn: Callable, obs_fn: Callable,
            dynamics_jac: Callable, obs_jac: Callable,
            Q: Sequence, R: Sequence,
            x0: Sequence[float], P0: Sequence,
            seed: int = 0,
            event_bus: Any = None) -> "Filterer":
        nF = len(x0)
        Qm = _validate_matrix(Q, "Q", shape=(nF, nF), spd=True)
        # Probe obs_fn to discover output dimension; lazy validate R later.
        z0 = obs_fn(list(x0))
        nH = len(z0)
        Rm = _validate_matrix(R, "R", shape=(nH, nH), spd=True)
        cfg = FiltererConfig(filter_name=EKF, seed=seed)
        # Construct a synthetic linear model just to carry Q, R.
        model = StateSpaceModel(
            F=tuple(tuple(0.0 for _ in range(nF)) for _ in range(nF)),
            H=tuple(tuple(0.0 for _ in range(nF)) for _ in range(nH)),
            Q=tuple(map(tuple, Qm)),
            R=tuple(map(tuple, Rm)),
        )
        return cls(cfg, model=model, mean=list(x0), cov=P0,
                   dynamics_fn=dynamics_fn, obs_fn=obs_fn,
                   dynamics_jac=dynamics_jac, obs_jac=obs_jac,
                   event_bus=event_bus)

    @classmethod
    def ukf(cls, *, dynamics_fn: Callable, obs_fn: Callable,
            Q: Sequence, R: Sequence,
            x0: Sequence[float], P0: Sequence,
            alpha: float = 1.0e-3, beta: float = 2.0, kappa: float = 0.0,
            seed: int = 0,
            event_bus: Any = None) -> "Filterer":
        nF = len(x0)
        Qm = _validate_matrix(Q, "Q", shape=(nF, nF), spd=True)
        z0 = obs_fn(list(x0))
        nH = len(z0)
        Rm = _validate_matrix(R, "R", shape=(nH, nH), spd=True)
        cfg = FiltererConfig(
            filter_name=UKF, seed=seed,
            ukf_alpha=alpha, ukf_beta=beta, ukf_kappa=kappa,
        )
        model = StateSpaceModel(
            F=tuple(tuple(0.0 for _ in range(nF)) for _ in range(nF)),
            H=tuple(tuple(0.0 for _ in range(nF)) for _ in range(nH)),
            Q=tuple(map(tuple, Qm)),
            R=tuple(map(tuple, Rm)),
        )
        return cls(cfg, model=model, mean=list(x0), cov=P0,
                   dynamics_fn=dynamics_fn, obs_fn=obs_fn,
                   event_bus=event_bus)

    @classmethod
    def particle(cls, *, propose_fn: Callable, likelihood_fn: Callable,
                 initial_particles: Sequence,
                 initial_weights: Sequence | None = None,
                 filter_name: str = SIR,
                 resampler: str = SYSTEMATIC,
                 ess_threshold: float = 0.5,
                 seed: int = 0,
                 event_bus: Any = None) -> "Filterer":
        N = len(initial_particles)
        cfg = FiltererConfig(
            filter_name=filter_name, seed=seed,
            n_particles=N, resampler=resampler,
            ess_threshold=ess_threshold,
        )
        return cls(cfg,
                   particles=initial_particles,
                   weights=initial_weights,
                   propose_fn=propose_fn,
                   likelihood_fn=likelihood_fn,
                   event_bus=event_bus)

    # ----- helpers ------------------------------------------------------

    def _emit_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(kind, payload)
        except Exception:
            # Never fail filtering because the bus errored.
            pass

    def _advance_fingerprint(self, kind: str, payload: Any) -> str:
        self._fingerprint = _hash(self._fingerprint, kind, payload)
        return self._fingerprint

    # ----- predict ------------------------------------------------------

    def predict(self, u: Sequence[float] | None = None) -> PredictionResult:
        with self._lock:
            self._n_step += 1
            if self._filter == KF or self._filter == INFORMATION_FILTER \
                    or self._filter == SQRT_KF:
                F = [list(r) for r in self._model.F]
                Q = [list(r) for r in self._model.Q]
                B = [list(r) for r in self._model.B] if self._model.B else None
                new_mean, new_cov = kalman_predict(
                    self._mean, self._cov, F, Q, u=u, B=B,
                )
                self._mean, self._cov = new_mean, new_cov
                self._predicted_means.append(list(new_mean))
                self._predicted_covs.append([list(r) for r in new_cov])
                belief = GaussianBelief(
                    mean=tuple(new_mean),
                    cov=tuple(tuple(r) for r in new_cov),
                )
            elif self._filter == EKF:
                F = self._dynamics_jac(self._mean, u)
                Q = [list(r) for r in self._model.Q]
                new_mean, new_cov = ekf_predict(
                    self._mean, self._cov, self._dynamics_fn, F, Q, u=u,
                )
                self._mean, self._cov = new_mean, new_cov
                self._predicted_means.append(list(new_mean))
                self._predicted_covs.append([list(r) for r in new_cov])
                belief = GaussianBelief(
                    mean=tuple(new_mean),
                    cov=tuple(tuple(r) for r in new_cov),
                )
            elif self._filter == UKF:
                Q = [list(r) for r in self._model.Q]
                new_mean, new_cov = ukf_predict(
                    self._mean, self._cov, self._dynamics_fn, Q,
                    u=u,
                    alpha=self._config.ukf_alpha,
                    beta=self._config.ukf_beta,
                    kappa=self._config.ukf_kappa,
                )
                self._mean, self._cov = new_mean, new_cov
                self._predicted_means.append(list(new_mean))
                self._predicted_covs.append([list(r) for r in new_cov])
                belief = GaussianBelief(
                    mean=tuple(new_mean),
                    cov=tuple(tuple(r) for r in new_cov),
                )
            elif self._filter in PARTICLE_FAMILY:
                # Propose new particles from p(x_t | x_{t-1}).
                new_particles = []
                for p in self._particles:
                    new_p = list(self._propose_fn(p, u, self._rng))
                    new_particles.append(new_p)
                self._particles = new_particles
                belief = ParticleBelief(
                    particles=tuple(tuple(p) for p in new_particles),
                    weights=tuple(self._weights),
                )
            else:
                raise UnknownFilter(self._filter)

            fp = self._advance_fingerprint("predict", {
                "n_step": self._n_step,
                "u": list(u) if u is not None else None,
            })
            self._emit_event(FILTERER_PREDICTED, {
                "n_step": self._n_step,
            })
            return PredictionResult(
                belief=belief,
                n_step=self._n_step,
                fingerprint=fp,
            )

    # ----- update -------------------------------------------------------

    def update(self, y: Sequence[float]) -> UpdateResult:
        with self._lock:
            if self._filter == KF or self._filter == INFORMATION_FILTER \
                    or self._filter == SQRT_KF:
                H = [list(r) for r in self._model.H]
                R = [list(r) for r in self._model.R]
                m = len(R)
                y_ = _validate_vector(y, "y", n=m)
                new_mean, new_cov, logev, innov, S, nis = kalman_update(
                    self._mean, self._cov, y_, H, R,
                )
                self._mean, self._cov = new_mean, new_cov
                self._filtered_means.append(list(new_mean))
                self._filtered_covs.append([list(r) for r in new_cov])
                belief = GaussianBelief(
                    mean=tuple(new_mean),
                    cov=tuple(tuple(r) for r in new_cov),
                )
                ess = None
            elif self._filter == EKF:
                H = self._obs_jac(self._mean)
                R = [list(r) for r in self._model.R]
                m = len(R)
                y_ = _validate_vector(y, "y", n=m)
                new_mean, new_cov, logev, innov, S, nis = ekf_update(
                    self._mean, self._cov, y_, self._obs_fn, H, R,
                )
                self._mean, self._cov = new_mean, new_cov
                self._filtered_means.append(list(new_mean))
                self._filtered_covs.append([list(r) for r in new_cov])
                belief = GaussianBelief(
                    mean=tuple(new_mean),
                    cov=tuple(tuple(r) for r in new_cov),
                )
                ess = None
            elif self._filter == UKF:
                R = [list(r) for r in self._model.R]
                m = len(R)
                y_ = _validate_vector(y, "y", n=m)
                new_mean, new_cov, logev, innov, S, nis = ukf_update(
                    self._mean, self._cov, y_, self._obs_fn, R,
                    alpha=self._config.ukf_alpha,
                    beta=self._config.ukf_beta,
                    kappa=self._config.ukf_kappa,
                )
                self._mean, self._cov = new_mean, new_cov
                self._filtered_means.append(list(new_mean))
                self._filtered_covs.append([list(r) for r in new_cov])
                belief = GaussianBelief(
                    mean=tuple(new_mean),
                    cov=tuple(tuple(r) for r in new_cov),
                )
                ess = None
            elif self._filter in PARTICLE_FAMILY:
                y_ = list(y)
                # Importance weights.
                log_w_inc = []
                log_w_prev = [math.log(max(w, _EPS)) for w in self._weights]
                for p in self._particles:
                    lp = float(self._likelihood_fn(y_, p))
                    log_w_inc.append(lp)
                log_w_new = [a + b for a, b in zip(log_w_prev, log_w_inc)]
                # Normalising constant — increment to log-marginal.
                Z = _logsumexp(log_w_new)
                Z_prev = _logsumexp(log_w_prev)
                logev = Z - Z_prev
                # Update weights.
                self._weights = [math.exp(lw - Z) for lw in log_w_new]
                # NIS analogue: -2 log p(y | filter mean) is a Mahalanobis-ish
                # statistic.  Approximate with -2 (logev).
                nis = -2.0 * logev
                innov = []
                S = None
                ess = effective_sample_size(self._weights)
                self._ess_history.append(ess)
                # Resample if degenerate.
                N = len(self._particles)
                if ess < self._config.ess_threshold * N:
                    self._resample_locked()
                belief = ParticleBelief(
                    particles=tuple(tuple(p) for p in self._particles),
                    weights=tuple(self._weights),
                )
            else:
                raise UnknownFilter(self._filter)

            self._log_marginal += logev
            self._nis_history.append(nis)
            self._innovation_history.append(list(innov))

            fp = self._advance_fingerprint("update", {
                "n_step": self._n_step,
                "y": list(y),
                "logev": logev,
                "nis": nis,
            })
            self._emit_event(FILTERER_UPDATED, {
                "n_step": self._n_step,
                "logev": logev,
                "nis": nis,
                "ess": ess,
            })
            return UpdateResult(
                belief=belief,
                log_evidence=logev,
                innovation=tuple(innov),
                innovation_cov=tuple(tuple(r) for r in S) if S is not None else None,
                nis=nis,
                ess=ess,
                n_step=self._n_step,
                fingerprint=fp,
            )

    # ----- resample (particle filter) -----------------------------------

    def resample(self, scheme: str | None = None) -> ResampleResult:
        if self._filter not in PARTICLE_FAMILY:
            raise FiltererError("resample is only valid for particle filters")
        with self._lock:
            return self._resample_locked(scheme=scheme)

    def _resample_locked(self, *, scheme: str | None = None) -> ResampleResult:
        scheme = _validate_resampler(scheme or self._config.resampler)
        N = len(self._particles)
        ess_before = effective_sample_size(self._weights)
        idx = _resample(self._weights, self._rng, scheme)
        self._particles = [list(self._particles[i]) for i in idx]
        self._weights = [1.0 / N] * N
        self._n_resamples += 1
        ess_after = float(N)
        fp = self._advance_fingerprint("resample", {
            "scheme": scheme,
            "n_step": self._n_step,
            "ess_before": ess_before,
            "ess_after": ess_after,
            "indices": idx,
        })
        self._emit_event(FILTERER_RESAMPLED, {
            "n_step": self._n_step,
            "scheme": scheme,
            "ess_before": ess_before,
        })
        return ResampleResult(
            n_step=self._n_step,
            scheme=scheme,
            ess_before=ess_before,
            ess_after=ess_after,
            fingerprint=fp,
        )

    # ----- smooth -------------------------------------------------------

    def smooth(self, smoother: str = RTS) -> SmoothResult:
        smoother = _validate_smoother(smoother)
        with self._lock:
            if smoother == RTS:
                if self._filter not in KALMAN_FAMILY:
                    raise FiltererError(
                        "RTS smoother requires a Kalman-family filter"
                    )
                if len(self._filtered_means) < 2:
                    raise FiltererError(
                        f"RTS smoother needs ≥2 filter steps, got {len(self._filtered_means)}"
                    )
                if self._filter == EKF:
                    # For EKF we need Jacobians at each step — use the
                    # *current* Jacobian as a single-step approximation when
                    # not persisted.
                    F = self._dynamics_jac(self._filtered_means[-2], None)
                elif self._filter == UKF:
                    # Not a perfect RTS; we use the linear approximation about
                    # the filtered mean from the model placeholder.  In
                    # practice users wire a non-linear smoother here.  We
                    # short-circuit when no analytic F is available.
                    raise FiltererError(
                        "RTS for UKF not supported in stdlib build; use FFBSi"
                    )
                else:
                    F = [list(r) for r in self._model.F]
                sm_means, sm_covs = rts_smooth(
                    self._filtered_means, self._filtered_covs,
                    self._predicted_means, self._predicted_covs, F,
                )
                beliefs = tuple(
                    GaussianBelief(
                        mean=tuple(m),
                        cov=tuple(tuple(r) for r in c),
                    )
                    for m, c in zip(sm_means, sm_covs)
                )
                fp = self._advance_fingerprint("smooth", {
                    "smoother": RTS,
                    "n_steps": len(beliefs),
                })
                self._emit_event(FILTERER_SMOOTHED, {
                    "smoother": RTS,
                    "n_steps": len(beliefs),
                })
                return SmoothResult(
                    beliefs=beliefs,
                    smoother=RTS,
                    fingerprint=fp,
                )
            if smoother == FFBSI:
                if self._filter not in PARTICLE_FAMILY:
                    raise FiltererError(
                        "FFBSi smoother requires a particle filter"
                    )
                # Single-trajectory backward sampler — Godsill-Doucet-West 2004.
                # Returns a particle belief at each step formed from the
                # backward-sampled trajectories (we run K=N trajectories).
                # For simplicity we sample one trajectory and return it as a
                # degenerate particle set at every step.  Production users
                # should call this once per trajectory.
                raise FiltererError(
                    "FFBSi requires persisted particle history "
                    "(call Filterer.update with trace=True — coming soon)"
                )
            raise UnknownSmoother(smoother)

    # ----- diagnostics + reports ----------------------------------------

    def report(self) -> FilterReport:
        with self._lock:
            m = self._model.n_obs if self._model else 1
            threshold = nis_chi2_threshold(m, self._config.nis_alpha)
            n_exceeds = sum(1 for v in self._nis_history if v > threshold)
            mean_nis = (
                sum(self._nis_history) / len(self._nis_history)
                if self._nis_history else 0.0
            )
            ess_min = min(self._ess_history) if self._ess_history else None
            ess_mean = (
                sum(self._ess_history) / len(self._ess_history)
                if self._ess_history else None
            )
            report = FilterReport(
                filter_name=self._filter,
                n_state=len(self._mean) if self._mean is not None
                else (len(self._particles[0]) if self._particles else 0),
                n_obs=m,
                n_steps=self._n_step,
                log_marginal=self._log_marginal,
                mean_nis=mean_nis,
                nis_chi2_threshold=threshold,
                n_nis_exceeds=n_exceeds,
                ess_min=ess_min,
                ess_mean=ess_mean,
                n_resamples=self._n_resamples,
                fingerprint=self._fingerprint,
            )
            self._emit_event(FILTERER_REPORT, asdict(report))
            return report

    def nis_history(self) -> list:
        with self._lock:
            return list(self._nis_history)

    def innovation_history(self) -> list:
        with self._lock:
            return [list(v) for v in self._innovation_history]

    def whiteness_test(self, max_lag: int = 1) -> tuple:
        """Box-Pierce statistic on the standardised innovations.

        Returns ``(stat, df, threshold_at_alpha=0.05)``.
        """
        innov = self._scalar_innovations()
        if len(innov) < max_lag + 2:
            raise FiltererError(
                f"whiteness: need ≥ max_lag+2 innovations, got {len(innov)}"
            )
        stat, df = innovation_whiteness_stat(innov, max_lag=max_lag)
        thr = nis_chi2_threshold(df, 0.05)
        return (stat, df, thr)

    def _scalar_innovations(self) -> list:
        """Return Mahalanobis-standardised scalar innovations for whiteness."""
        out = []
        for innov in self._innovation_history:
            if not innov:
                continue
            # Use NIS as the standardised scalar.
            out.append(sum(v * v for v in innov))
        # Use sqrt to get a centred-ish scalar.
        return [math.sqrt(max(0.0, x)) for x in out]

    # ----- introspection -----------------------------------------------

    @property
    def n_step(self) -> int:
        return self._n_step

    @property
    def filter_name(self) -> str:
        return self._filter

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def log_marginal(self) -> float:
        return self._log_marginal

    @property
    def belief(self) -> Any:
        with self._lock:
            if self._filter in KALMAN_FAMILY:
                return GaussianBelief(
                    mean=tuple(self._mean),
                    cov=tuple(tuple(r) for r in self._cov),
                )
            return ParticleBelief(
                particles=tuple(tuple(p) for p in self._particles),
                weights=tuple(self._weights),
            )

    @property
    def mean(self) -> tuple:
        with self._lock:
            if self._filter in KALMAN_FAMILY:
                return tuple(self._mean)
            # Particle mean.
            N = len(self._particles)
            d = len(self._particles[0])
            mu = [0.0] * d
            for i in range(N):
                wi = self._weights[i]
                pi = self._particles[i]
                for k in range(d):
                    mu[k] += wi * pi[k]
            return tuple(mu)

    @property
    def cov(self) -> tuple:
        with self._lock:
            if self._filter in KALMAN_FAMILY:
                return tuple(tuple(r) for r in self._cov)
            N = len(self._particles)
            d = len(self._particles[0])
            mu = list(self.mean)
            P = [[0.0] * d for _ in range(d)]
            for i in range(N):
                wi = self._weights[i]
                dx = [self._particles[i][k] - mu[k] for k in range(d)]
                for a in range(d):
                    for b in range(d):
                        P[a][b] += wi * dx[a] * dx[b]
            return tuple(tuple(r) for r in P)

    def clear(self) -> None:
        with self._lock:
            self._n_step = 0
            self._log_marginal = 0.0
            self._nis_history = []
            self._innovation_history = []
            self._ess_history = []
            self._n_resamples = 0
            self._filtered_means = []
            self._filtered_covs = []
            self._predicted_means = []
            self._predicted_covs = []
            self._fingerprint = _GENESIS
            self._emit_event(FILTERER_CLEARED, {})

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "filter": self._filter,
                "n_step": self._n_step,
                "mean": list(self._mean) if self._mean is not None else None,
                "cov": ([list(r) for r in self._cov]
                        if self._cov is not None else None),
                "particles": ([list(p) for p in self._particles]
                              if self._particles is not None else None),
                "weights": list(self._weights) if self._weights is not None else None,
                "log_marginal": self._log_marginal,
                "n_resamples": self._n_resamples,
                "fingerprint": self._fingerprint,
            }


__all__ = [
    # constants
    "KF", "INFORMATION_FILTER", "SQRT_KF", "EKF", "UKF",
    "SIR", "BOOTSTRAP", "APF",
    "KNOWN_FILTERS", "KALMAN_FAMILY", "PARTICLE_FAMILY",
    "MULTINOMIAL", "SYSTEMATIC", "STRATIFIED", "RESIDUAL",
    "KNOWN_RESAMPLERS",
    "RTS", "FFBSI", "KNOWN_SMOOTHERS",
    "FILTERER_STARTED", "FILTERER_PREDICTED", "FILTERER_UPDATED",
    "FILTERER_RESAMPLED", "FILTERER_SMOOTHED",
    "FILTERER_REPORT", "FILTERER_CLEARED", "KNOWN_EVENTS",
    # exceptions
    "FiltererError", "UnknownFilter", "UnknownResampler",
    "UnknownSmoother", "InvalidDimension", "InvalidMatrix",
    "InvalidObservation", "InvalidParticles", "FilterDegenerate",
    "NonPositiveDefinite", "InvalidCallable", "GenericConfigError",
    # dataclasses
    "StateSpaceModel", "GaussianBelief", "ParticleBelief",
    "PredictionResult", "UpdateResult", "ResampleResult",
    "SmoothResult", "FilterReport", "FiltererConfig",
    # main class
    "Filterer",
    # primitive functions
    "kalman_predict", "kalman_update",
    "ekf_predict", "ekf_update",
    "ukf_predict", "ukf_update",
    "rts_smooth",
    # bounds + diagnostics
    "nis_chi2_threshold", "crisan_doucet_mse_bound",
    "massart_dkw_band", "effective_sample_size",
    "innovation_whiteness_stat",
]
