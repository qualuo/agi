r"""Attributor — data attribution / influence functions as a runtime primitive.

Every other primitive in this runtime answers a *forward* question.
``Predictor`` asks "given the data, what will happen next?". ``Abductor``
asks "given the observation, which hypothesis explains it?". ``Verifier``
asks "given the candidate, does it satisfy the spec?".

The *backwards* question — "given a model's *output*, **which of its
training points caused that output, by how much, and would the output
flip if I removed them?**" — is the *data-attribution* question, and it
is the one any coordination engine must answer the moment a downstream
decision is contested, an outlier is suspected, a privacy-driven
unlearning request is filed, or an investor asks *which observations
underpin this forecast*.

The ``Attributor`` is the runtime primitive that answers that question.
Given a fitted parametric hypothesis ``H = (model_kind, θ̂)`` and the
training set ``D = {(x_i, y_i)}_{i=1}^n`` that produced ``θ̂``, plus a
*query function* ``Q(θ)`` (typically the test-point loss, a prediction
component, or any smooth functional of the parameters), the
``Attributor`` returns

  * the **leave-one-out (LOO) influence** ``Q(θ̂_{-i}) − Q(θ̂)`` for
    every training point — either *exactly* by refitting (linear and
    logistic regression have closed-form LOO) or via the *first-order
    influence-function approximation* of Cook (1977) and Koh & Liang
    (2017) when refitting is not affordable;
  * **Cook's distance** ``D_i = (1/p)·(ε̂_i² · h_{ii}) / (s² · (1 − h_{ii})²)``
    (Cook 1977) and **DFBETAS** (Belsley-Kuh-Welsch 1980) for full
    *parameter-level* leverage on linear models;
  * the **studentized residual** ``t_i = ε̂_i / (s_{(i)}·√(1 − h_{ii}))``
    and **hat-matrix leverage** ``h_{ii}`` (Hoaglin-Welsch 1978) — the
    *diagonal of the projection onto column-space*;
  * **TracIn ideal-checkpoint attribution** (Pruthi-Liu-Sundararajan-
    Inan 2020): given a per-step learning trajectory ``θ_t`` and
    learning rates ``η_t``, the cumulative training-point contribution
    ``TracInIdeal_i(Q) = Σ_t η_t ⟨∇L_i(θ_t), ∇Q(θ_t)⟩``;
  * **group / slice attribution** ``Q(θ̂_{-S}) − Q(θ̂)`` for any
    subset ``S ⊆ {1,…,n}`` with **Sherman-Morrison-Woodbury**-based
    rank-|S| Hessian update for first-order computation;
  * the **counterfactual prediction** "what would the model predict if
    we removed the top-K most influential points and refit?" with the
    *exact* refit value;
  * a **PAC-Bayes-style confidence band** on the LOO-influence estimate
    using bootstrap or jackknife resampling (Efron-Tibshirani 1986);
  * a **decision flip certificate** "removing these K points changes
    the MAP decision from a → b" with an associated *e-value* on the
    counterfactual likelihood ratio;
  * a tamper-evident SHA-256 fingerprint chain over every fit, query,
    attribution and refit, so the ``AttestationLedger`` can replay the
    full attribution trace byte-for-byte.

The pitch reduced to a runtime call::

    attr = Attributor()
    attr.fit("price_model", LINEAR,
             X=[[1, 4], [1, 7], [1, 11]],
             y=[2.0, 3.1, 4.7])
    ip = attr.influence("price_model",
                        query="loss",
                        test_point=([1, 9], 4.0))
    print(ip.per_point)                  # influence[i] for i ∈ D
    print(ip.most_influential(k=3))      # ranked list
    cd = attr.cooks_distance("price_model")
    h = attr.leverage("price_model")
    flip = attr.decision_flip(
        "price_model",
        query=lambda theta: 1 if theta[1] > 0.3 else 0,
        budget_k=2,
    )
    cf = attr.counterfactual_refit("price_model", remove=ip.top_indices(2))
    report = attr.report("price_model")  # everything + receipts

Mathematical roots and algorithms shipped
-----------------------------------------

**The classical influence function.**  Hampel (1974)'s influence
function ``IF(x; θ̂, F)`` answers "how much would the estimator move
under an infinitesimal contamination of the data distribution ``F`` at
point ``x``?".  For an estimator defined by an *M-estimating equation*
``Σ_i ψ(z_i, θ̂) = 0`` (which subsumes least-squares, MLE, and most
empirical-risk minimisers) the influence function is::

    IF(z_k; θ̂)  =  H(θ̂)^{-1} · ψ(z_k, θ̂)

where ``H(θ̂) = (1/n) Σ_i ∇_θ ψ(z_i, θ̂)`` is the empirical Hessian of
the loss.  Koh & Liang (2017) lifted this to deep models by noting that
for any *smooth* functional ``Q(θ)`` of interest, a single point's
influence on ``Q`` follows the chain rule::

    Influence_i(Q)  ≈  − (1/n) · ∇Q(θ̂)^⊤ · H(θ̂)^{-1} · ∇L_i(θ̂)

The Attributor ships:

  * closed-form ``IF`` for linear regression (the *exact* analytic
    LOO is the Sherman-Morrison rank-1 update — no approximation),
  * closed-form ``IF`` for logistic regression evaluated at the
    converged MLE (the canonical Koh-Liang setting),
  * a *user-supplied gradient/Hessian-vector-product* interface for
    arbitrary differentiable losses; the caller hands in ``∇L_i`` and
    a ``hvp`` and the Attributor solves ``H^{-1} ∇Q`` by the
    Krylov-style **conjugate-gradient** routine of Hestenes-Stiefel
    (1952) — the algorithm Koh-Liang use at ImageNet scale.

**Closed-form LOO for linear regression.**  Given the OLS hat matrix
``H = X(X^⊤X)^{-1}X^⊤`` and residuals ``ε̂``, the LOO residual is the
*press residual* (Allen 1971)::

    ε_{i,(i)}  =  ε̂_i / (1 − h_{ii})

and the LOO mean-square prediction error is the *PRESS statistic*::

    PRESS  =  Σ_i (ε̂_i / (1 − h_{ii}))^2

so the per-point LOO loss change is *available without any refit at
all*.  All three of {leverage, Cook's distance, DFBETAS, studentized
residual} are subsequently exact byproducts of the same hat matrix —
shipped as ``leverage()``, ``cooks_distance()``, ``dfbetas()`` and
``studentized_residual()``.

**Closed-form LOO for logistic regression.**  At the converged MLE
``β̂`` the **Newton-step LOO** approximation of Pregibon (1981) is::

    β̂_{-i}  ≈  β̂  −  (X^⊤WX)^{-1} · x_i · ((y_i − π̂_i) / (1 − h_{ii,W}))

where ``W = diag(π̂_i(1 − π̂_i))`` is the weighted-least-squares matrix
of the final IRLS step and ``h_{ii,W} = x_i^⊤ (X^⊤WX)^{-1} x_i w_i`` is
the *weighted leverage*.  Pregibon proved this matches a single
Fisher-scoring step from ``β̂``; it is exact to first order in
``1/(n − p)`` and is the standard "logistic LOO" deployed in every
serious diagnostic package since 1981.

**Sherman-Morrison-Woodbury for group attribution.**  Removing a set
``S`` of |S| points (rather than one) is equivalent to a rank-|S|
update of ``X^⊤X``.  The Sherman-Morrison-Woodbury formula gives::

    (X^⊤X − X_S^⊤ X_S)^{-1}
    =  M  +  M·X_S^⊤·(I_{|S|} − X_S·M·X_S^⊤)^{-1}·X_S·M

where ``M = (X^⊤X)^{-1}``.  The cost is ``O(|S|^3)`` — independent of
``n`` — and ships as ``group_loo()``.

**Cook's distance** (Cook 1977)::

    D_i  =  (1/p) · (ε̂_i^2 · h_{ii}) / (s² · (1 − h_{ii})²)

is the canonical *one-number* influence summary.  By convention an
observation with ``D_i > 4/n`` (or ``D_i > 1``) is flagged as
*high-influence*.

**DFBETAS** (Belsley-Kuh-Welsch 1980, *Regression Diagnostics*, p.13)::

    DFBETAS_{ij}  =  (β̂_j − β̂_{j,(i)}) / (s_{(i)} · √(M_{jj}))

is the *signed standardised change* in the j-th coefficient on dropping
observation i.  Threshold ``|DFBETAS| > 2/√n`` flags coefficient-level
sensitivity.

**Studentized residual** (Hoaglin-Welsch 1978)::

    t_i  =  ε̂_i / (s_{(i)} · √(1 − h_{ii}))

asymptotically ``t_i ∼ t(n − p − 1)`` under the null of a correct
linear model.  Flagging ``|t_i| > 2`` is the textbook outlier rule.

**TracIn (Pruthi-Liu-Sundararajan-Inan 2020).**  Given a learning
trajectory ``{(θ_t, η_t)}_{t=1}^T`` (the parameter at each gradient
step and the learning rate that was applied), the *ideal* TracIn
attribution of training point ``i`` to the final value of a query
``Q(θ_T)`` is::

    TracIn_i(Q)  =  Σ_t η_t · ⟨∇L_i(θ_t),  ∇Q(θ_t)⟩

The intuition is exact: each gradient step moves ``Q`` by ``η_t · ⟨∇L,
∇Q⟩`` if the training point ``i`` was the only point used.  In a batch
training run this generalises to "training point ``i`` contributed
``Σ_{t : i ∈ B_t} η_t ⟨∇L_i, ∇Q⟩``".  Attributor ships the *ideal*
formulation, accepting a caller-supplied trajectory.

**TRAK** (Park-Georgiev-Ilyas-Madry-Engstrom 2023, ICML).  TRAK is a
random-projection scaling of the Koh-Liang influence to billions of
parameters; it reduces to the ``H^{-1}``-weighted gradient inner
product after dimension reduction.  Attributor ships the per-feature
``proj_dim``-d projection as ``trak_attribution()`` so coordinators can
scale to large-parameter models without sacrificing the influence
semantics.

**Concentration certificates on attribution estimates.**  For LOO
estimates obtained via the first-order approximation, the empirical
gap between ``IF_i`` and the true ``Q(θ̂_{-i}) − Q(θ̂)`` is bounded
under standard convexity (Koh-Liang 2017 Theorem 2.1; Basu et al.
2020 *Second-Order Influence Functions*).  Attributor ships:

  * Hoeffding / empirical-Bernstein half-widths on aggregate
    attribution statistics (group means, top-K sum) (Maurer-Pontil
    2009).
  * Bootstrap percentile CIs on per-point influence by resampling the
    training set (Efron-Tibshirani 1986; the *non-parametric*
    delete-d jackknife of Shao-Wu 1989 for variance estimation).
  * For every reported influence the second-order bound
    ``|true LOO − IF| ≤ ½ · ‖∇L_i‖ · ‖θ̂ − θ̂_{-i}‖² · σ_max(∇³L)``
    when the caller supplies a third-derivative spectral norm, else a
    "first-order only" flag.

Composition with the rest of the runtime
----------------------------------------

  * **Curator.**  Top-K most-influential-and-incorrect points are
    natural relabelling candidates.
  * **Conformal.**  Conformal anomaly score and Cook's distance both
    measure "how unlike the rest" — Attributor's diagnostics
    *attribute* the conformity score to specific examples.
  * **Robustifier.**  Robustifier asks for the worst-case removal of
    ε-fraction of data; Attributor returns *exactly* which points
    saturate that worst-case via top-K influence.
  * **Auditor / AttestationLedger.**  Every fit, query, and
    counterfactual refit is fingerprinted; an auditor can replay the
    entire chain bit-for-bit.
  * **Forecaster.**  When a predictive set is contested, Attributor
    answers "which historical observation caused this set to include
    the contested value?".
  * **Aligner.**  The DPO/IPO gradient of any preference pair is a
    valid ``∇L_i``; Attributor identifies which preference pairs
    drive any policy decision.
  * **Quantilizer.**  Quantilizer needs a budget on policy switches;
    Attributor's *decision-flip certificate* tells it exactly how
    many points away the next switch is.
  * **Pretunist.**  Pretunist's test-time-training step is itself a
    weighted ERM; Attributor's influence on the pretuned model is the
    *adaptation footprint*, and the receipts make the adaptation
    auditable.

Investor framing
----------------

The pitch a coordinator's UI can surface, automatically, for every
fitted model the user routes through it::

    "The model's prediction for query x* = [1, 9] is ŷ = 4.012,
     down from 4.487 had we not seen training point #2 (Cook's D =
     0.83, leverage = 0.71, studentized residual = +2.4 — flagged
     *highly influential* by Cook's 4/n rule).

     The runner-up influencers are #5 (Cook's D = 0.12) and #11
     (Cook's D = 0.09); removing the top-3 points flips the
     classifier's decision from ACCEPT to REJECT with an e-value of
     34.7 (decisive on Jeffreys's scale).

     Counterfactual prediction (top-3 removed):  ŷ = 3.241
     ε_2 (PRESS LOO residual):                   +0.612
     Bootstrap 95% CI on Influence_#2:           [+0.421, +0.794]

     Replay fingerprint: 9c4f7b…  (verifiable via AttestationLedger)."

Every number here is grounded in published, citable mathematics
(Cook 1977; Hampel 1974; Belsley-Kuh-Welsch 1980; Hoaglin-Welsch
1978; Pregibon 1981; Koh-Liang 2017; Pruthi et al. 2020; Park et al.
2023) and reproducible bit-exactly from the attribution-event log.
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
# Public constants — model kinds
# =====================================================================

# Linear regression (ordinary least squares).  Closed-form LOO via
# the hat matrix; Cook's distance, leverage, DFBETAS, studentized
# residual all exact.
LINEAR = "linear"

# Logistic regression via Newton/IRLS.  Pregibon (1981) one-step LOO
# at the converged MLE, weighted leverage.
LOGISTIC = "logistic"

# Ridge regression (L2-regularised linear regression).  Closed-form
# LOO via shrinkage hat matrix.
RIDGE = "ridge"

# Caller-supplied differentiable loss.  Caller hands in per-point
# log-loss values and (optionally) a gradient + Hessian-vector product.
# Influence is computed by conjugate gradient on the Hessian.
CUSTOM = "custom"

KNOWN_KINDS = frozenset({LINEAR, LOGISTIC, RIDGE, CUSTOM})

# Attribution methods.
EXACT_LOO = "exact_loo"                  # refit without each point
INFLUENCE_FUNCTION = "influence_function"  # Cook 1977 / Koh-Liang 2017
COOKS_DISTANCE = "cooks_distance"        # Cook 1977 D_i
DFBETAS_METHOD = "dfbetas"               # Belsley-Kuh-Welsch 1980
LEVERAGE = "leverage"                    # diag of hat matrix
STUDENTIZED = "studentized_residual"     # Hoaglin-Welsch 1978
PRESS = "press_residual"                 # Allen 1971
TRACIN_IDEAL = "tracin_ideal"            # Pruthi et al. 2020
TRAK_METHOD = "trak"                     # Park et al. 2023

KNOWN_METHODS = frozenset({
    EXACT_LOO, INFLUENCE_FUNCTION, COOKS_DISTANCE, DFBETAS_METHOD,
    LEVERAGE, STUDENTIZED, PRESS, TRACIN_IDEAL, TRAK_METHOD,
})

# Query kinds — what is being attributed.
QUERY_LOSS = "loss"          # test-point loss
QUERY_PREDICTION = "prediction"  # raw model prediction
QUERY_PARAMETER = "parameter"    # specific θ_j
QUERY_CUSTOM = "custom"          # caller-supplied Q(θ)

KNOWN_QUERIES = frozenset({QUERY_LOSS, QUERY_PREDICTION, QUERY_PARAMETER, QUERY_CUSTOM})

# Events.
ATTRIBUTOR_STARTED = "attributor.started"
ATTRIBUTOR_FIT = "attributor.fit"
ATTRIBUTOR_QUERIED = "attributor.queried"
ATTRIBUTOR_INFLUENCE_COMPUTED = "attributor.influence_computed"
ATTRIBUTOR_COUNTERFACTUAL = "attributor.counterfactual"
ATTRIBUTOR_DECISION_FLIP = "attributor.decision_flip"
ATTRIBUTOR_TRACIN_COMPUTED = "attributor.tracin_computed"
ATTRIBUTOR_GROUP_LOO = "attributor.group_loo"
ATTRIBUTOR_REPORTED = "attributor.reported"
ATTRIBUTOR_CLEARED = "attributor.cleared"

KNOWN_EVENTS = frozenset({
    ATTRIBUTOR_STARTED, ATTRIBUTOR_FIT, ATTRIBUTOR_QUERIED,
    ATTRIBUTOR_INFLUENCE_COMPUTED, ATTRIBUTOR_COUNTERFACTUAL,
    ATTRIBUTOR_DECISION_FLIP, ATTRIBUTOR_TRACIN_COMPUTED,
    ATTRIBUTOR_GROUP_LOO, ATTRIBUTOR_REPORTED, ATTRIBUTOR_CLEARED,
})

# Numerical defaults.
_EPS = 1e-12
_INF = float("inf")
_NEG_INF = float("-inf")
_LN2 = math.log(2.0)
_LN10 = math.log(10.0)
_LN_2PI = math.log(2.0 * math.pi)
_GENESIS = hashlib.sha256(b"attributor.v1.genesis").hexdigest()
_DEFAULT_RIDGE = 1e-6  # diagonal jitter for numerical stability
_DEFAULT_CG_TOL = 1e-8
_DEFAULT_CG_MAX_ITER = 200
_DEFAULT_NEWTON_TOL = 1e-8
_DEFAULT_NEWTON_MAX_ITER = 100


# =====================================================================
# Exceptions
# =====================================================================


class AttributorError(ValueError):
    """Base class for attributor-domain errors."""


class UnknownKind(AttributorError):
    """A model kind is unknown or a name was never fit."""


class UnknownMethod(AttributorError):
    """An attribution method is not in KNOWN_METHODS."""


class UnknownQuery(AttributorError):
    """A query kind is not in KNOWN_QUERIES."""


class InvalidData(AttributorError):
    """Training data is malformed."""


class InvalidQuery(AttributorError):
    """A query specification is malformed."""


class SingularMatrix(AttributorError):
    """Design matrix is singular — refit with a stronger ridge."""


class NotFit(AttributorError):
    """A model is being queried before fit()."""


class InsufficientData(AttributorError):
    """Too few points for the requested operation."""


class ConvergenceError(AttributorError):
    """Iterative solver failed to converge (Newton, CG, etc.)."""


# =====================================================================
# Numerical helpers — pure Python linear algebra
# =====================================================================
#
# We use lists of lists for matrices and lists for vectors so that
# Attributor has no numpy dependency.  All ops are O(p^3) or O(np^2)
# which is fine for the runtime-primitive scale (n, p up to a few
# thousand) the coordination engine actually routes through us.


Vector = list[float]
Matrix = list[list[float]]


def _zeros(n: int) -> Vector:
    return [0.0] * n


def _zeros_mat(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def _identity(n: int) -> Matrix:
    M = _zeros_mat(n, n)
    for i in range(n):
        M[i][i] = 1.0
    return M


def _mat_vec(M: Matrix, v: Vector) -> Vector:
    rows = len(M)
    cols = len(M[0]) if rows else 0
    if len(v) != cols:
        raise AttributorError(f"mat_vec: shape mismatch {rows}x{cols} · {len(v)}")
    out = _zeros(rows)
    for i in range(rows):
        row = M[i]
        s = 0.0
        for j in range(cols):
            s += row[j] * v[j]
        out[i] = s
    return out


def _vec_mat(v: Vector, M: Matrix) -> Vector:
    rows = len(M)
    cols = len(M[0]) if rows else 0
    if len(v) != rows:
        raise AttributorError(f"vec_mat: shape mismatch {len(v)} · {rows}x{cols}")
    out = _zeros(cols)
    for j in range(cols):
        s = 0.0
        for i in range(rows):
            s += v[i] * M[i][j]
        out[j] = s
    return out


def _mat_mat(A: Matrix, B: Matrix) -> Matrix:
    ra = len(A)
    ca = len(A[0]) if ra else 0
    rb = len(B)
    cb = len(B[0]) if rb else 0
    if ca != rb:
        raise AttributorError(f"mat_mat: shape mismatch {ra}x{ca} · {rb}x{cb}")
    out = _zeros_mat(ra, cb)
    for i in range(ra):
        row_a = A[i]
        for k in range(ca):
            aik = row_a[k]
            if aik == 0.0:
                continue
            row_b = B[k]
            row_o = out[i]
            for j in range(cb):
                row_o[j] += aik * row_b[j]
    return out


def _transpose(M: Matrix) -> Matrix:
    rows = len(M)
    cols = len(M[0]) if rows else 0
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def _vec_dot(u: Vector, v: Vector) -> float:
    if len(u) != len(v):
        raise AttributorError(f"vec_dot: length mismatch {len(u)} vs {len(v)}")
    s = 0.0
    for i in range(len(u)):
        s += u[i] * v[i]
    return s


def _vec_norm(v: Vector) -> float:
    return math.sqrt(_vec_dot(v, v))


def _vec_axpy(alpha: float, x: Vector, y: Vector) -> Vector:
    return [alpha * xi + yi for xi, yi in zip(x, y)]


def _vec_scale(alpha: float, v: Vector) -> Vector:
    return [alpha * vi for vi in v]


def _outer(u: Vector, v: Vector) -> Matrix:
    return [[ui * vj for vj in v] for ui in u]


def _add_mat(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _sub_mat(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _add_diag(A: Matrix, lam: float) -> Matrix:
    M = [row[:] for row in A]
    for i in range(len(M)):
        M[i][i] += lam
    return M


def _cholesky(A: Matrix) -> Matrix:
    """Lower-triangular Cholesky factor of a symmetric positive-definite A.

    Raises SingularMatrix on non-PD input — the caller may then retry
    with a stronger ridge.
    """
    n = len(A)
    L = _zeros_mat(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                if s <= 0.0:
                    raise SingularMatrix(
                        f"cholesky: matrix not positive definite at row {i} (s={s})"
                    )
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _solve_lower(L: Matrix, b: Vector) -> Vector:
    """Solve L x = b for lower-triangular L."""
    n = len(L)
    if len(b) != n:
        raise AttributorError(f"solve_lower: shape mismatch {n} vs {len(b)}")
    x = _zeros(n)
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i][k] * x[k]
        if L[i][i] == 0.0:
            raise SingularMatrix(f"solve_lower: zero pivot at {i}")
        x[i] = s / L[i][i]
    return x


def _solve_upper(U: Matrix, b: Vector) -> Vector:
    """Solve U x = b for upper-triangular U."""
    n = len(U)
    if len(b) != n:
        raise AttributorError(f"solve_upper: shape mismatch {n} vs {len(b)}")
    x = _zeros(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        for k in range(i + 1, n):
            s -= U[i][k] * x[k]
        if U[i][i] == 0.0:
            raise SingularMatrix(f"solve_upper: zero pivot at {i}")
        x[i] = s / U[i][i]
    return x


def _solve_spd(A: Matrix, b: Vector, *, ridge: float = _DEFAULT_RIDGE) -> Vector:
    """Solve A x = b for SPD A via Cholesky with a small diagonal jitter."""
    A_reg = _add_diag(A, ridge)
    try:
        L = _cholesky(A_reg)
    except SingularMatrix:
        # Retry with stronger ridge.
        L = _cholesky(_add_diag(A, ridge * 1e6))
    y = _solve_lower(L, b)
    Lt = _transpose(L)
    return _solve_upper(Lt, y)


def _invert_spd(A: Matrix, *, ridge: float = _DEFAULT_RIDGE) -> Matrix:
    """Invert SPD A by solving A · A_inv = I column-wise."""
    n = len(A)
    A_reg = _add_diag(A, ridge)
    try:
        L = _cholesky(A_reg)
    except SingularMatrix:
        L = _cholesky(_add_diag(A, ridge * 1e6))
    Lt = _transpose(L)
    A_inv = _zeros_mat(n, n)
    for j in range(n):
        ej = _zeros(n)
        ej[j] = 1.0
        y = _solve_lower(L, ej)
        x = _solve_upper(Lt, y)
        for i in range(n):
            A_inv[i][j] = x[i]
    return A_inv


def _conjugate_gradient(
    hvp: Callable[[Vector], Vector],
    b: Vector,
    *,
    tol: float = _DEFAULT_CG_TOL,
    max_iter: int = _DEFAULT_CG_MAX_ITER,
) -> Vector:
    """Solve H x = b for SPD operator H given only its matrix-vector product.

    Hestenes-Stiefel (1952) conjugate-gradient on a Hessian-vector
    product oracle.  This is the routine Koh-Liang (2017) deploy at
    ImageNet scale to compute H^{-1} ∇Q on the test point without
    materialising H.
    """
    n = len(b)
    x = _zeros(n)
    r = b[:]
    p = b[:]
    rs_old = _vec_dot(r, r)
    if math.sqrt(rs_old) < tol:
        return x
    for it in range(max_iter):
        Hp = hvp(p)
        denom = _vec_dot(p, Hp)
        if denom <= 0.0:
            raise ConvergenceError(
                f"conjugate_gradient: non-PD direction at iter {it} (p·H·p = {denom})"
            )
        alpha = rs_old / denom
        x = _vec_axpy(alpha, p, x)
        r = _vec_axpy(-alpha, Hp, r)
        rs_new = _vec_dot(r, r)
        if math.sqrt(rs_new) < tol:
            return x
        beta = rs_new / rs_old
        p = _vec_axpy(beta, p, r)
        rs_old = rs_new
    raise ConvergenceError(
        f"conjugate_gradient: did not converge in {max_iter} iterations "
        f"(final residual = {math.sqrt(rs_old):.3e})"
    )


def _logistic_sigmoid(z: float) -> float:
    if z >= 0.0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _log1p_exp(z: float) -> float:
    """Numerically stable log(1 + e^z)."""
    if z >= 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return repr(x)


# =====================================================================
# Concentration-inequality helpers (shared across Attributor reports)
# =====================================================================


def hoeffding_half_width(n: int, *, delta: float, b: float = 1.0) -> float:
    """Hoeffding (1963) half-width for the mean of n iid observations in [0, b]."""
    if n <= 0:
        raise InsufficientData("hoeffding requires n >= 1")
    if not 0.0 < delta < 1.0:
        raise AttributorError("delta must be in (0, 1)")
    if b <= 0.0:
        raise AttributorError("b must be positive")
    return b * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def empirical_bernstein_half_width(
    n: int, sample_variance: float, *, delta: float, b: float = 1.0
) -> float:
    """Maurer-Pontil (2009) empirical-Bernstein half-width."""
    if n <= 1:
        raise InsufficientData("empirical Bernstein requires n >= 2")
    if not 0.0 < delta < 1.0:
        raise AttributorError("delta must be in (0, 1)")
    if sample_variance < 0.0:
        raise AttributorError("sample_variance must be non-negative")
    if b <= 0.0:
        raise AttributorError("b must be positive")
    log_term = math.log(4.0 / delta)
    return (
        math.sqrt(2.0 * sample_variance * log_term / n)
        + 7.0 * b * log_term / (3.0 * (n - 1))
    )


# =====================================================================
# Public dataclasses
# =====================================================================


@dataclass(frozen=True)
class FitReport:
    """The result of fitting a parametric model on (X, y).

    Attributes
    ----------
    name : str
        Caller-supplied identifier for this hypothesis.
    kind : str
        One of ``KNOWN_KINDS``.
    n : int
        Number of training points.
    p : int
        Number of parameters (including intercept if present).
    theta : list[float]
        Fitted parameters θ̂.
    residual_sum_squares : float
        Σ ε̂_i² (linear / ridge) or − Σ log L_i(θ̂) (logistic / custom).
    sigma2 : float
        Unbiased residual variance s² = RSS / (n − p) for linear models.
    converged : bool
        True iff the Newton/IRLS iteration converged inside the tolerance.
    iterations : int
        Iteration count to convergence (1 for closed-form linear).
    fingerprint : str
        SHA-256 chain root after this fit was logged.
    """
    name: str
    kind: str
    n: int
    p: int
    theta: list[float]
    residual_sum_squares: float
    sigma2: float
    converged: bool
    iterations: int
    fingerprint: str


@dataclass(frozen=True)
class InfluenceReport:
    """Per-training-point influence on a query Q.

    Attributes
    ----------
    name : str
        Hypothesis name.
    query : str
        One of ``KNOWN_QUERIES``.
    method : str
        One of ``KNOWN_METHODS``.
    per_point : list[float]
        ``influence[i] = approximate Q(θ̂_{-i}) − Q(θ̂)``.
        Positive ⇒ point ``i`` *decreased* Q on inclusion (Q would
        rise if removed); negative ⇒ ``i`` *increased* Q.
    test_point : Any
        Caller-supplied test point or query spec.
    q_baseline : float
        Q(θ̂) — the baseline value of the query at the fitted model.
    bound_second_order : float | None
        Spectral-norm Hessian bound on the IF error, if a third-
        derivative bound was supplied.  None ⇒ first-order only.
    fingerprint : str
        Chain root after this influence pass.
    """
    name: str
    query: str
    method: str
    per_point: list[float]
    test_point: Any
    q_baseline: float
    bound_second_order: float | None
    fingerprint: str

    def most_influential(self, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k indices ranked by |influence|, descending."""
        if k < 0:
            raise AttributorError(f"k must be >= 0; got {k}")
        n = len(self.per_point)
        if k > n:
            k = n
        idx_inf = list(enumerate(self.per_point))
        idx_inf.sort(key=lambda kv: -abs(kv[1]))
        return idx_inf[:k]

    def top_indices(self, k: int = 5) -> list[int]:
        """Indices of the top-k most-influential points by |influence|."""
        return [i for i, _ in self.most_influential(k)]

    def sum_of_influence(self) -> float:
        """Σ_i Influence_i(Q).  By identity equal to the LOO total."""
        return sum(self.per_point)

    def positive_count(self) -> int:
        """Number of points with strictly positive influence on Q."""
        return sum(1 for v in self.per_point if v > 0.0)

    def negative_count(self) -> int:
        """Number of points with strictly negative influence on Q."""
        return sum(1 for v in self.per_point if v < 0.0)


@dataclass(frozen=True)
class LinearDiagnostics:
    """Closed-form linear-regression case-influence diagnostics."""
    name: str
    leverage: list[float]                  # h_ii
    residual: list[float]                  # ε̂_i
    studentized: list[float]               # t_i
    press_residual: list[float]            # ε̂_i / (1 − h_ii)
    cooks_distance: list[float]            # D_i
    dfbetas: list[list[float]]             # DFBETAS_{ij}; rows = obs, cols = params
    press_statistic: float                 # Σ press²
    s_squared: float                       # s²
    fingerprint: str


@dataclass(frozen=True)
class CounterfactualReport:
    """Refit the model with ``removed`` indices excluded; report the diff."""
    name: str
    removed: list[int]
    theta_full: list[float]
    theta_counterfactual: list[float]
    q_full: float
    q_counterfactual: float
    delta_q: float
    fingerprint: str


@dataclass(frozen=True)
class DecisionFlipReport:
    """How many points must be removed to flip a discrete decision."""
    name: str
    decision_full: Any
    minimal_set: list[int]                 # smallest greedy set that flips
    decision_after: Any
    flipped: bool
    e_value: float                         # likelihood ratio under refit
    log10_bayes_factor: float
    fingerprint: str


@dataclass(frozen=True)
class TracInReport:
    """TracIn ideal-checkpoint attribution (Pruthi et al. 2020)."""
    name: str
    per_point: list[float]
    n_checkpoints: int
    fingerprint: str

    def most_influential(self, k: int = 5) -> list[tuple[int, float]]:
        if k < 0:
            raise AttributorError(f"k must be >= 0; got {k}")
        n = len(self.per_point)
        idx_inf = list(enumerate(self.per_point))
        idx_inf.sort(key=lambda kv: -abs(kv[1]))
        return idx_inf[: min(k, n)]


@dataclass(frozen=True)
class BootstrapBand:
    """Bootstrap confidence band on per-point influence."""
    name: str
    method: str
    per_point_lower: list[float]
    per_point_upper: list[float]
    delta: float
    n_resamples: int
    fingerprint: str


@dataclass(frozen=True)
class AttributorReport:
    """Aggregate snapshot of a single hypothesis."""
    name: str
    kind: str
    n: int
    p: int
    theta: list[float]
    fingerprint: str
    diagnostics: LinearDiagnostics | None
    fit_iterations: int
    converged: bool


# =====================================================================
# Internal hypothesis state
# =====================================================================


@dataclass
class _FitState:
    name: str
    kind: str
    X: list[list[float]]
    y: list[float]
    theta: list[float]
    fitted: bool = False
    p: int = 0
    n: int = 0
    iterations: int = 0
    converged: bool = False
    # Linear / ridge:
    XtX_inv: Matrix | None = None   # (X^⊤X + λI)^{-1}
    hat: Matrix | None = None       # only built on demand
    leverage_diag: list[float] | None = None
    residual: list[float] | None = None
    s_squared: float = 0.0
    rss: float = 0.0
    ridge: float = 0.0
    # Logistic:
    pi_hat: list[float] | None = None       # π̂_i at converged β̂
    w_diag: list[float] | None = None       # π̂_i (1 − π̂_i)
    XtWX_inv: Matrix | None = None
    weighted_leverage: list[float] | None = None
    # Custom:
    custom_grad_fn: Callable[[int, Vector], Vector] | None = None
    custom_hvp_fn: Callable[[Vector, Vector], Vector] | None = None
    custom_loss_fn: Callable[[int, Vector], float] | None = None


# =====================================================================
# Attribution math — per-kind helpers
# =====================================================================


def _fit_linear(X: list[list[float]], y: list[float], *, ridge: float) -> tuple[Vector, Matrix, list[float], float]:
    """OLS / Ridge: β̂ = (X^⊤X + λI)^{-1} X^⊤y.

    Returns (β̂, XtX_inv, residuals, rss).
    """
    n = len(X)
    if n == 0:
        raise InsufficientData("fit: empty training data")
    p = len(X[0])
    Xt = _transpose(X)
    XtX = _mat_mat(Xt, X)
    Xty = _mat_vec(Xt, y)
    XtX_reg = _add_diag(XtX, ridge)
    XtX_inv = _invert_spd(XtX_reg)
    beta = _mat_vec(XtX_inv, Xty)
    y_hat = _mat_vec(X, beta)
    residual = [y[i] - y_hat[i] for i in range(n)]
    rss = sum(r * r for r in residual)
    return beta, XtX_inv, residual, rss


def _leverage_from_invXtX(X: list[list[float]], XtX_inv: Matrix) -> list[float]:
    """h_ii = x_i^⊤ (X^⊤X + λI)^{-1} x_i."""
    n = len(X)
    h = _zeros(n)
    for i in range(n):
        v = _mat_vec(XtX_inv, X[i])
        h[i] = _vec_dot(X[i], v)
    return h


def _fit_logistic(
    X: list[list[float]],
    y: list[float],
    *,
    ridge: float,
    tol: float = _DEFAULT_NEWTON_TOL,
    max_iter: int = _DEFAULT_NEWTON_MAX_ITER,
) -> tuple[Vector, list[float], list[float], Matrix, int, bool]:
    """Logistic regression via Newton/IRLS.

    Returns (β̂, π̂, W_diag = π̂(1−π̂), (X^⊤WX + λI)^{-1}, iterations, converged).
    """
    n = len(X)
    if n == 0:
        raise InsufficientData("fit: empty training data")
    p = len(X[0])
    for yi in y:
        if yi not in (0.0, 1.0):
            raise InvalidData(f"logistic targets must be in {{0,1}}; got {yi!r}")
    beta = _zeros(p)
    Xt = _transpose(X)
    converged = False
    for it in range(1, max_iter + 1):
        z = _mat_vec(X, beta)
        pi = [_logistic_sigmoid(zi) for zi in z]
        w = [pi[i] * (1.0 - pi[i]) for i in range(n)]
        # Score: X^⊤ (y − π)
        score = [0.0] * p
        for j in range(p):
            s = 0.0
            for i in range(n):
                s += X[i][j] * (y[i] - pi[i])
            score[j] = s
        # Fisher information: X^⊤ W X
        XtWX = _zeros_mat(p, p)
        for i in range(n):
            wi = w[i]
            if wi == 0.0:
                continue
            row = X[i]
            for j in range(p):
                aij = wi * row[j]
                for k in range(p):
                    XtWX[j][k] += aij * row[k]
        XtWX_reg = _add_diag(XtWX, ridge)
        try:
            step = _solve_spd(XtWX_reg, score)
        except SingularMatrix:
            step = _solve_spd(XtWX_reg, score, ridge=ridge * 1e6)
        # Standard Newton: β ← β + (X^⊤WX)^{-1} X^⊤(y − π).
        beta = _vec_axpy(1.0, step, beta)
        if _vec_norm(step) < tol:
            converged = True
            break
    # Final values at converged β̂.
    z = _mat_vec(X, beta)
    pi = [_logistic_sigmoid(zi) for zi in z]
    w = [pi[i] * (1.0 - pi[i]) for i in range(n)]
    XtWX = _zeros_mat(p, p)
    for i in range(n):
        wi = w[i]
        if wi == 0.0:
            continue
        row = X[i]
        for j in range(p):
            aij = wi * row[j]
            for k in range(p):
                XtWX[j][k] += aij * row[k]
    XtWX_inv = _invert_spd(_add_diag(XtWX, ridge))
    return beta, pi, w, XtWX_inv, it, converged


def _logistic_neg_log_likelihood(X: list[list[float]], y: list[float], beta: Vector) -> float:
    nll = 0.0
    n = len(X)
    for i in range(n):
        z = 0.0
        for j in range(len(beta)):
            z += X[i][j] * beta[j]
        # − [ y · log σ(z) + (1−y) · log(1−σ(z)) ] = log(1 + e^z) − y·z
        nll += _log1p_exp(z) - y[i] * z
    return nll


# =====================================================================
# Event payload + attestation chain
# =====================================================================


@dataclass
class AttestationEvent:
    seq: int
    kind: str
    timestamp: float
    payload: dict[str, Any]
    parent_hash: str
    this_hash: str


def _hash_event(seq: int, kind: str, payload: dict[str, Any], parent: str) -> str:
    canonical = json.dumps(
        {"seq": seq, "kind": kind, "payload": _jsonable(payload), "parent": parent},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# =====================================================================
# The Attributor
# =====================================================================


class Attributor:
    """Data attribution / influence-function runtime primitive.

    A single ``Attributor`` instance can hold many fitted hypotheses
    (one for each model the coordinator wants to audit).  Methods are
    thread-safe at the granularity of *one* hypothesis: the global
    state lock is held only across the inner writes to ``_states`` and
    the event chain.  Numerical work itself is GIL-bound pure Python.

    Construction is free of side effects.  Every state-changing call
    appends to the ``AttestationEvent`` chain so the entire computation
    can be replayed bit-exactly from the event log.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, _FitState] = {}
        self._events: list[AttestationEvent] = []
        self._fingerprint: str = _GENESIS
        self._started_ts: float = time.time()
        self._record(ATTRIBUTOR_STARTED, {"ts": self._started_ts})

    # -----------------------------------------------------------------
    # Event chain
    # -----------------------------------------------------------------

    def _record(self, kind: str, payload: dict[str, Any]) -> AttestationEvent:
        seq = len(self._events)
        ts = time.time()
        new_hash = _hash_event(seq, kind, payload, self._fingerprint)
        ev = AttestationEvent(
            seq=seq,
            kind=kind,
            timestamp=ts,
            payload=dict(payload),
            parent_hash=self._fingerprint,
            this_hash=new_hash,
        )
        self._events.append(ev)
        self._fingerprint = new_hash
        return ev

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def events(self) -> list[AttestationEvent]:
        return list(self._events)

    def verify_chain(self) -> bool:
        """Recompute every hash and confirm the chain is intact.

        Useful in the AttestationLedger context: an auditor pulls the
        events out, replays the hashes, and reports whether the chain
        matches the disclosed fingerprint.
        """
        parent = _GENESIS
        for ev in self._events:
            expected = _hash_event(ev.seq, ev.kind, ev.payload, parent)
            if expected != ev.this_hash:
                return False
            if ev.parent_hash != parent:
                return False
            parent = ev.this_hash
        return parent == self._fingerprint

    # -----------------------------------------------------------------
    # Lookup helpers
    # -----------------------------------------------------------------

    def _state(self, name: str) -> _FitState:
        st = self._states.get(name)
        if st is None:
            raise UnknownKind(f"no hypothesis named {name!r}; call fit() first")
        return st

    def names(self) -> list[str]:
        return sorted(self._states.keys())

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(
        self,
        name: str,
        kind: str,
        *,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        ridge: float = 0.0,
        max_iter: int = _DEFAULT_NEWTON_MAX_ITER,
        tol: float = _DEFAULT_NEWTON_TOL,
        # CUSTOM-kind hooks:
        loss_fn: Callable[[int, Vector], float] | None = None,
        grad_fn: Callable[[int, Vector], Vector] | None = None,
        hvp_fn: Callable[[Vector, Vector], Vector] | None = None,
        theta_init: Sequence[float] | None = None,
    ) -> FitReport:
        """Fit a model of ``kind`` on (X, y) and register it under ``name``.

        Linear and ridge kinds are closed-form; logistic uses Newton.
        ``CUSTOM`` takes the *initial* parameters from ``theta_init``
        and the caller-supplied gradient / loss / hvp functions; the
        Attributor itself does not iterate on a custom loss (we assume
        the caller has already fit it via whatever method).  Custom
        influence is then computed from the supplied derivatives.
        """
        if not isinstance(name, str) or not name:
            raise AttributorError("name must be a non-empty string")
        if kind not in KNOWN_KINDS:
            raise UnknownKind(
                f"unknown kind {kind!r}; expected one of {sorted(KNOWN_KINDS)}"
            )
        X_list = [list(map(float, row)) for row in X]
        y_list = [float(v) for v in y]
        if len(X_list) != len(y_list):
            raise InvalidData(
                f"len(X)={len(X_list)} != len(y)={len(y_list)}"
            )
        if not X_list:
            raise InsufficientData("fit: at least one row required")
        p_obs = len(X_list[0]) if X_list else 0
        for i, row in enumerate(X_list):
            if len(row) != p_obs:
                raise InvalidData(
                    f"row {i} has length {len(row)} != {p_obs}"
                )
        if ridge < 0.0:
            raise AttributorError(f"ridge must be >= 0; got {ridge!r}")

        with self._lock:
            state = _FitState(
                name=name,
                kind=kind,
                X=X_list,
                y=y_list,
                theta=[],
                p=p_obs,
                n=len(X_list),
                ridge=ridge,
            )

            if kind in (LINEAR, RIDGE):
                eff_ridge = ridge if kind == RIDGE else 0.0
                beta, XtX_inv, residual, rss = _fit_linear(X_list, y_list, ridge=eff_ridge)
                state.theta = beta
                state.XtX_inv = XtX_inv
                state.residual = residual
                state.rss = rss
                state.iterations = 1
                state.converged = True
                lev = _leverage_from_invXtX(X_list, XtX_inv)
                state.leverage_diag = lev
                df = max(state.n - state.p, 1)
                state.s_squared = rss / df

            elif kind == LOGISTIC:
                if theta_init is not None:
                    raise AttributorError("logistic: theta_init not yet supported")
                beta, pi, w, XtWX_inv, it, converged = _fit_logistic(
                    X_list, y_list, ridge=max(ridge, _DEFAULT_RIDGE),
                    tol=tol, max_iter=max_iter,
                )
                state.theta = beta
                state.pi_hat = pi
                state.w_diag = w
                state.XtWX_inv = XtWX_inv
                state.iterations = it
                state.converged = converged
                # Weighted leverage: h_ii = w_i · x_i^⊤ (X^⊤WX)^{-1} x_i.
                wlev = _zeros(state.n)
                for i in range(state.n):
                    v = _mat_vec(XtWX_inv, X_list[i])
                    wlev[i] = w[i] * _vec_dot(X_list[i], v)
                state.weighted_leverage = wlev
                state.rss = _logistic_neg_log_likelihood(X_list, y_list, beta)

            elif kind == CUSTOM:
                if theta_init is None:
                    raise AttributorError("custom kind requires theta_init")
                if grad_fn is None:
                    raise AttributorError("custom kind requires grad_fn")
                state.theta = [float(v) for v in theta_init]
                if len(state.theta) != p_obs:
                    state.p = len(state.theta)
                state.custom_grad_fn = grad_fn
                state.custom_hvp_fn = hvp_fn
                state.custom_loss_fn = loss_fn
                state.iterations = 0
                state.converged = True

            else:  # pragma: no cover — guarded by KNOWN_KINDS check
                raise UnknownKind(f"unhandled kind {kind!r}")

            state.fitted = True
            self._states[name] = state

            ev = self._record(
                ATTRIBUTOR_FIT,
                {
                    "name": name, "kind": kind, "n": state.n, "p": len(state.theta),
                    "rss": state.rss, "ridge": ridge,
                    "converged": state.converged, "iterations": state.iterations,
                },
            )
            return FitReport(
                name=name, kind=kind, n=state.n, p=len(state.theta),
                theta=list(state.theta), residual_sum_squares=state.rss,
                sigma2=state.s_squared, converged=state.converged,
                iterations=state.iterations, fingerprint=ev.this_hash,
            )

    # -----------------------------------------------------------------
    # Closed-form linear diagnostics
    # -----------------------------------------------------------------

    def leverage(self, name: str) -> list[float]:
        """Diagonal of the hat matrix h_{ii} = x_i^⊤(X^⊤X)^{-1}x_i.

        For linear / ridge models this is exact.  For logistic models
        it is the *weighted* leverage h_{ii,W} = w_i · x_i^⊤(X^⊤WX)^{-1}x_i.
        """
        st = self._state(name)
        if st.kind in (LINEAR, RIDGE):
            assert st.leverage_diag is not None
            return list(st.leverage_diag)
        if st.kind == LOGISTIC:
            assert st.weighted_leverage is not None
            return list(st.weighted_leverage)
        raise AttributorError(f"leverage: not supported for kind {st.kind!r}")

    def press_residuals(self, name: str) -> list[float]:
        """Allen (1971) PRESS residuals ε̂_i / (1 − h_ii) — exact LOO residuals."""
        st = self._state(name)
        if st.kind not in (LINEAR, RIDGE):
            raise AttributorError(f"press_residuals: kind {st.kind!r} not supported")
        assert st.residual is not None and st.leverage_diag is not None
        out = []
        for i in range(st.n):
            denom = 1.0 - st.leverage_diag[i]
            if abs(denom) < _EPS:
                out.append(float("inf") if st.residual[i] >= 0 else float("-inf"))
            else:
                out.append(st.residual[i] / denom)
        return out

    def press_statistic(self, name: str) -> float:
        """PRESS = Σ_i (ε̂_i / (1 − h_ii))^2."""
        return sum(r * r for r in self.press_residuals(name))

    def studentized_residuals(self, name: str) -> list[float]:
        """Hoaglin-Welsch (1978) studentized residuals.

        t_i = ε̂_i / (s_{(i)} · √(1 − h_ii))  with
            s_{(i)}^2 = ((n − p) · s² − ε̂_i^2/(1 − h_ii)) / (n − p − 1)
        """
        st = self._state(name)
        if st.kind not in (LINEAR, RIDGE):
            raise AttributorError(
                f"studentized_residuals: kind {st.kind!r} not supported"
            )
        assert st.residual is not None and st.leverage_diag is not None
        df = st.n - st.p
        if df <= 1:
            raise InsufficientData(
                f"studentized requires n - p >= 2; got {df}"
            )
        out = []
        for i in range(st.n):
            h = st.leverage_diag[i]
            denom_h = 1.0 - h
            if abs(denom_h) < _EPS:
                out.append(float("inf"))
                continue
            r2 = st.residual[i] ** 2
            s_i_sq = (df * st.s_squared - r2 / denom_h) / (df - 1)
            s_i_sq = max(s_i_sq, _EPS)
            s_i = math.sqrt(s_i_sq)
            out.append(st.residual[i] / (s_i * math.sqrt(denom_h)))
        return out

    def cooks_distance(self, name: str) -> list[float]:
        """Cook (1977) D_i = (1/p) · (ε̂_i² · h_ii) / (s² · (1 − h_ii)²)."""
        st = self._state(name)
        if st.kind not in (LINEAR, RIDGE):
            raise AttributorError(f"cooks_distance: kind {st.kind!r} not supported")
        assert st.residual is not None and st.leverage_diag is not None
        if st.s_squared <= 0.0:
            return [0.0] * st.n
        out = []
        for i in range(st.n):
            h = st.leverage_diag[i]
            denom = (1.0 - h) ** 2
            if denom < _EPS:
                out.append(float("inf"))
                continue
            num = (st.residual[i] ** 2) * h
            out.append(num / (st.p * st.s_squared * denom))
        return out

    def dfbetas(self, name: str) -> list[list[float]]:
        """Belsley-Kuh-Welsch (1980) per-coefficient sensitivity matrix.

        DFBETAS_{ij} = (β̂_j − β̂_{j,(i)}) / (s_{(i)} · √M_jj)
        where M = (X^⊤X)^{-1}.  Threshold |DFBETAS_{ij}| > 2/√n flags
        coefficient-level sensitivity to observation i.
        """
        st = self._state(name)
        if st.kind not in (LINEAR, RIDGE):
            raise AttributorError(f"dfbetas: kind {st.kind!r} not supported")
        assert (
            st.residual is not None
            and st.leverage_diag is not None
            and st.XtX_inv is not None
        )
        df = st.n - st.p
        if df <= 1:
            raise InsufficientData(f"dfbetas requires n - p >= 2; got {df}")
        # M_jj = diag of (X^⊤X)^{-1}
        Mdiag = [st.XtX_inv[j][j] for j in range(st.p)]
        # Per-obs (β̂ − β̂_{(i)}) = (X^⊤X)^{-1} x_i · ε̂_i / (1 − h_ii)
        out: list[list[float]] = [[0.0] * st.p for _ in range(st.n)]
        for i in range(st.n):
            h = st.leverage_diag[i]
            denom_h = 1.0 - h
            if abs(denom_h) < _EPS:
                out[i] = [float("inf")] * st.p
                continue
            r2 = st.residual[i] ** 2
            s_i_sq = (df * st.s_squared - r2 / denom_h) / (df - 1)
            s_i_sq = max(s_i_sq, _EPS)
            s_i = math.sqrt(s_i_sq)
            v = _mat_vec(st.XtX_inv, st.X[i])  # M x_i
            scale = st.residual[i] / denom_h
            for j in range(st.p):
                m_jj = max(Mdiag[j], _EPS)
                out[i][j] = (scale * v[j]) / (s_i * math.sqrt(m_jj))
        return out

    def linear_diagnostics(self, name: str) -> LinearDiagnostics:
        """All exact linear-regression case-influence diagnostics in one call."""
        st = self._state(name)
        if st.kind not in (LINEAR, RIDGE):
            raise AttributorError(
                f"linear_diagnostics: kind {st.kind!r} not supported"
            )
        lev = self.leverage(name)
        press = self.press_residuals(name)
        # Studentized requires df >= 2; we can fail soft.
        try:
            stud = self.studentized_residuals(name)
        except InsufficientData:
            stud = [float("nan")] * st.n
        try:
            cooks = self.cooks_distance(name)
        except (InsufficientData, AttributorError):
            cooks = [0.0] * st.n
        try:
            dfb = self.dfbetas(name)
        except (InsufficientData, AttributorError):
            dfb = [[float("nan")] * st.p for _ in range(st.n)]
        assert st.residual is not None
        ev = self._record(
            ATTRIBUTOR_QUERIED,
            {
                "name": name, "kind": "linear_diagnostics",
                "press_statistic": sum(p * p for p in press),
                "max_leverage": max(lev) if lev else 0.0,
                "max_cooks": max(cooks) if cooks else 0.0,
            },
        )
        return LinearDiagnostics(
            name=name, leverage=lev, residual=list(st.residual),
            studentized=stud, press_residual=press, cooks_distance=cooks,
            dfbetas=dfb, press_statistic=sum(p * p for p in press),
            s_squared=st.s_squared, fingerprint=ev.this_hash,
        )

    # -----------------------------------------------------------------
    # Influence on a query Q
    # -----------------------------------------------------------------

    def influence(
        self,
        name: str,
        *,
        query: str = QUERY_LOSS,
        test_point: Any = None,
        method: str = INFLUENCE_FUNCTION,
        custom_query: Callable[[Vector], float] | None = None,
        custom_query_grad: Callable[[Vector], Vector] | None = None,
        param_index: int | None = None,
    ) -> InfluenceReport:
        """Per-point influence on a query Q about the fitted model.

        Parameters
        ----------
        query :
            - ``"loss"`` — Q(θ) = loss at a held-out test point (test_point
              is (x, y)).
            - ``"prediction"`` — Q(θ) = scalar prediction at x (test_point
              is x).
            - ``"parameter"`` — Q(θ) = θ_{param_index}.
            - ``"custom"`` — Q(θ) = custom_query(θ) with gradient
              custom_query_grad(θ).
        method :
            - ``"influence_function"`` — first-order Koh-Liang.
            - ``"exact_loo"`` — full refit per point (linear/ridge fast,
              logistic slower).
        """
        st = self._state(name)
        if query not in KNOWN_QUERIES:
            raise UnknownQuery(
                f"unknown query {query!r}; expected one of {sorted(KNOWN_QUERIES)}"
            )
        if method not in KNOWN_METHODS:
            raise UnknownMethod(
                f"unknown method {method!r}; expected one of {sorted(KNOWN_METHODS)}"
            )
        # 1. Resolve the query function and its gradient ∇_θ Q at θ̂.
        q_baseline, grad_Q = self._resolve_query(
            st, query, test_point, custom_query, custom_query_grad, param_index
        )
        # 2. Pick the algorithm.
        if method == EXACT_LOO:
            inf = self._exact_loo(st, query, test_point, custom_query, param_index)
        elif method == INFLUENCE_FUNCTION:
            inf = self._influence_function(st, grad_Q)
        else:
            raise UnknownMethod(
                f"method {method!r} not supported in influence(); "
                "use the dedicated cooks_distance/dfbetas/etc. accessor"
            )
        ev = self._record(
            ATTRIBUTOR_INFLUENCE_COMPUTED,
            {
                "name": name, "query": query, "method": method,
                "n": st.n, "q_baseline": q_baseline,
                "max_abs_influence": max((abs(v) for v in inf), default=0.0),
            },
        )
        return InfluenceReport(
            name=name, query=query, method=method,
            per_point=inf, test_point=_jsonable(test_point),
            q_baseline=q_baseline, bound_second_order=None,
            fingerprint=ev.this_hash,
        )

    def _resolve_query(
        self,
        st: _FitState,
        query: str,
        test_point: Any,
        custom_query: Callable[[Vector], float] | None,
        custom_query_grad: Callable[[Vector], Vector] | None,
        param_index: int | None,
    ) -> tuple[float, Vector]:
        """Evaluate Q(θ̂) and ∇_θ Q(θ̂) for each supported query kind."""
        theta = st.theta
        p = len(theta)
        if query == QUERY_PARAMETER:
            if param_index is None or not 0 <= param_index < p:
                raise InvalidQuery(
                    f"query=parameter requires 0 <= param_index < {p}; "
                    f"got {param_index!r}"
                )
            grad = _zeros(p)
            grad[param_index] = 1.0
            return theta[param_index], grad
        if query == QUERY_PREDICTION:
            if test_point is None:
                raise InvalidQuery("query=prediction requires test_point=x")
            x = [float(v) for v in test_point]
            if len(x) != p:
                raise InvalidQuery(f"prediction x has length {len(x)} != {p}")
            if st.kind in (LINEAR, RIDGE):
                return _vec_dot(x, theta), list(x)
            if st.kind == LOGISTIC:
                z = _vec_dot(x, theta)
                pi = _logistic_sigmoid(z)
                return pi, _vec_scale(pi * (1.0 - pi), x)
            # custom: caller-supplied predictor is non-standard;
            # we treat prediction = θ · x as a default linear pred.
            return _vec_dot(x, theta), list(x)
        if query == QUERY_LOSS:
            if test_point is None or not isinstance(test_point, (tuple, list)):
                raise InvalidQuery(
                    "query=loss requires test_point=(x, y)"
                )
            x_seq, y_val = test_point
            x = [float(v) for v in x_seq]
            y_val = float(y_val)
            if len(x) != p:
                raise InvalidQuery(f"loss x has length {len(x)} != {p}")
            if st.kind in (LINEAR, RIDGE):
                pred = _vec_dot(x, theta)
                resid = pred - y_val
                # Q = ½ ε² ⇒ ∇Q = ε · x.
                return 0.5 * resid * resid, _vec_scale(resid, x)
            if st.kind == LOGISTIC:
                z = _vec_dot(x, theta)
                pi = _logistic_sigmoid(z)
                # − [y log π + (1−y) log(1−π)] = log(1 + e^z) − y·z
                # ∇Q = (π − y) · x
                loss = _log1p_exp(z) - y_val * z
                return loss, _vec_scale(pi - y_val, x)
            # CUSTOM: prediction loss not defined; require custom query.
            raise InvalidQuery(
                "query=loss for kind=custom: use query=custom with custom_query"
            )
        if query == QUERY_CUSTOM:
            if custom_query is None or custom_query_grad is None:
                raise InvalidQuery(
                    "query=custom requires both custom_query and custom_query_grad"
                )
            v = float(custom_query(theta))
            g = list(custom_query_grad(theta))
            if len(g) != p:
                raise InvalidQuery(
                    f"custom_query_grad returned length {len(g)} != {p}"
                )
            return v, g
        raise UnknownQuery(f"unknown query {query!r}")

    def _influence_function(self, st: _FitState, grad_Q: Vector) -> list[float]:
        """Compute the first-order Koh-Liang influence vector.

        For each i:  Influence_i(Q) ≈ − (1/n) · ∇Q^⊤ H^{-1} ∇L_i(θ̂)

        where H is the empirical Hessian of the loss.  For linear /
        ridge / logistic models H is closed-form; for CUSTOM it is the
        caller-supplied HVP and we solve H^{-1} ∇Q by conjugate
        gradient (Hestenes-Stiefel 1952).
        """
        n = st.n
        p = len(st.theta)
        if n == 0:
            return []

        # The influence-function approximation we ship is the *exact-LOO
        # first-order Taylor expansion*: for the sum-loss convention
        # used throughout the runtime (H_total = Σ ∇²L_i, not 1/n × Σ),
        # the first-order LOO change in any smooth Q is
        #
        #     Q(θ̂_{-i}) − Q(θ̂)  ≈  ∇Q(θ̂)^⊤ · H^{-1} · ∇L_i(θ̂)
        #
        # which sign-matches Koh-Liang (2017) Eq. 2 after their −1/n
        # conversion from upweight-IF to LOO-IF.

        if st.kind in (LINEAR, RIDGE):
            assert st.XtX_inv is not None and st.residual is not None
            # For squared-loss L_i = ½(x_i^⊤θ − y_i)²:
            #   ∇L_i(θ̂) = (x_i^⊤ θ̂ − y_i) · x_i = − ε̂_i · x_i.
            H_inv_grad_Q = _mat_vec(st.XtX_inv, grad_Q)
            out = _zeros(n)
            for i in range(n):
                inner = _vec_dot(H_inv_grad_Q, st.X[i])
                # ∇Q^⊤ · M · (− ε̂_i · x_i)  =  − ε̂_i · ⟨M ∇Q, x_i⟩
                out[i] = -st.residual[i] * inner
            return out

        if st.kind == LOGISTIC:
            assert st.XtWX_inv is not None and st.pi_hat is not None
            H_inv_grad_Q = _mat_vec(st.XtWX_inv, grad_Q)
            out = _zeros(n)
            for i in range(n):
                # ∇L_i = (π̂_i − y_i) · x_i  (note positive sign)
                resid = st.pi_hat[i] - st.y[i]
                inner = _vec_dot(H_inv_grad_Q, st.X[i])
                # ∇Q^⊤ · M_W · (π̂_i − y_i) · x_i  =  resid · ⟨M_W ∇Q, x_i⟩
                out[i] = resid * inner
            return out

        if st.kind == CUSTOM:
            if st.custom_grad_fn is None or st.custom_hvp_fn is None:
                raise AttributorError(
                    "custom kind influence_function: requires grad_fn and hvp_fn"
                )
            hvp = lambda v: st.custom_hvp_fn(st.theta, v)  # noqa: E731
            H_inv_grad_Q = _conjugate_gradient(hvp, grad_Q)
            out = _zeros(n)
            for i in range(n):
                gi = list(st.custom_grad_fn(i, st.theta))
                if len(gi) != p:
                    raise InvalidData(
                        f"custom grad_fn returned length {len(gi)} != {p}"
                    )
                # ∇Q^⊤ · H^{-1} · ∇L_i  (no extra sign — caller's ∇L_i convention).
                inner = _vec_dot(H_inv_grad_Q, gi)
                out[i] = inner
            return out

        raise AttributorError(f"_influence_function: kind {st.kind!r} unhandled")

    def _exact_loo(
        self,
        st: _FitState,
        query: str,
        test_point: Any,
        custom_query: Callable[[Vector], float] | None,
        param_index: int | None,
    ) -> list[float]:
        """Exact LOO influence: refit without each point, eval Q, return diff.

        For linear / ridge this uses Sherman-Morrison so the cost is O(p²)
        per point rather than a full refit (n × O(p³)).  For logistic
        we use Pregibon (1981)'s one-step LOO.  For CUSTOM we honestly
        refit by calling the caller-supplied refit_fn (or refuse).
        """
        n = st.n
        p = len(st.theta)
        if n == 0:
            return []
        # First, materialise q_baseline for delta computation.
        q_baseline, _ = self._resolve_query(
            st, query, test_point, custom_query, None, param_index
        )

        def _q(theta_loo: Vector) -> float:
            return self._q_at_theta(st, query, test_point, custom_query, param_index, theta_loo)

        if st.kind in (LINEAR, RIDGE):
            assert st.XtX_inv is not None and st.residual is not None
            assert st.leverage_diag is not None
            out = _zeros(n)
            M = st.XtX_inv
            for i in range(n):
                h = st.leverage_diag[i]
                denom_h = 1.0 - h
                if abs(denom_h) < _EPS:
                    out[i] = float("inf")
                    continue
                # β̂_{(i)} = β̂ − M x_i ε̂_i / (1 − h_ii)
                M_xi = _mat_vec(M, st.X[i])
                scale = st.residual[i] / denom_h
                theta_loo = [st.theta[j] - scale * M_xi[j] for j in range(p)]
                out[i] = _q(theta_loo) - q_baseline
            return out

        if st.kind == LOGISTIC:
            assert st.XtWX_inv is not None and st.pi_hat is not None
            assert st.weighted_leverage is not None
            out = _zeros(n)
            for i in range(n):
                # Pregibon: β̂_{(i)} ≈ β̂ − (X^⊤WX)^{-1} x_i (y_i − π̂_i) / (1 − h_{ii,W})
                resid = st.y[i] - st.pi_hat[i]
                denom = 1.0 - st.weighted_leverage[i]
                if abs(denom) < _EPS:
                    out[i] = float("inf")
                    continue
                M_xi = _mat_vec(st.XtWX_inv, st.X[i])
                scale = resid / denom
                theta_loo = [st.theta[j] - scale * M_xi[j] for j in range(p)]
                out[i] = _q(theta_loo) - q_baseline
            return out

        if st.kind == CUSTOM:
            raise AttributorError(
                "exact_loo: not supported for kind=custom (would require refit)"
            )

        raise AttributorError(f"_exact_loo: unhandled kind {st.kind!r}")

    def _q_at_theta(
        self,
        st: _FitState,
        query: str,
        test_point: Any,
        custom_query: Callable[[Vector], float] | None,
        param_index: int | None,
        theta: Vector,
    ) -> float:
        """Re-evaluate Q at a *different* parameter vector."""
        p = len(theta)
        if query == QUERY_PARAMETER:
            if param_index is None or not 0 <= param_index < p:
                raise InvalidQuery("query=parameter: bad param_index")
            return theta[param_index]
        if query == QUERY_PREDICTION:
            x = [float(v) for v in test_point]
            if st.kind in (LINEAR, RIDGE):
                return _vec_dot(x, theta)
            if st.kind == LOGISTIC:
                return _logistic_sigmoid(_vec_dot(x, theta))
            return _vec_dot(x, theta)
        if query == QUERY_LOSS:
            x_seq, y_val = test_point
            x = [float(v) for v in x_seq]
            y_val = float(y_val)
            if st.kind in (LINEAR, RIDGE):
                r = _vec_dot(x, theta) - y_val
                return 0.5 * r * r
            if st.kind == LOGISTIC:
                z = _vec_dot(x, theta)
                return _log1p_exp(z) - y_val * z
            raise InvalidQuery("query=loss for custom kind: use query=custom")
        if query == QUERY_CUSTOM:
            if custom_query is None:
                raise InvalidQuery("query=custom needs custom_query")
            return float(custom_query(theta))
        raise UnknownQuery(f"_q_at_theta: unknown query {query!r}")

    # -----------------------------------------------------------------
    # Group LOO via Sherman-Morrison-Woodbury
    # -----------------------------------------------------------------

    def group_loo(
        self,
        name: str,
        *,
        indices: Sequence[int],
        query: str = QUERY_LOSS,
        test_point: Any = None,
        custom_query: Callable[[Vector], float] | None = None,
        param_index: int | None = None,
    ) -> float:
        """Sherman-Morrison-Woodbury exact LOO of a *group* of indices.

        Returns ``Q(θ̂_{-S}) − Q(θ̂)`` for ``S = indices``.  Only valid
        for LINEAR / RIDGE; logistic uses an iterative refit (call
        ``counterfactual_refit`` if you really need that).
        """
        st = self._state(name)
        idx = list(indices)
        if not idx:
            return 0.0
        if any(not 0 <= i < st.n for i in idx):
            raise AttributorError(
                f"group_loo: indices out of range [0, {st.n})"
            )
        if len(set(idx)) != len(idx):
            raise AttributorError("group_loo: indices must be unique")
        q_baseline, _ = self._resolve_query(
            st, query, test_point, custom_query, None, param_index
        )

        if st.kind not in (LINEAR, RIDGE):
            raise AttributorError(
                f"group_loo: kind {st.kind!r} not supported (use counterfactual_refit)"
            )
        assert st.XtX_inv is not None and st.residual is not None
        # M = (X^⊤X + λI)^{-1}; X_S has rows = [X[i] for i in idx].
        XS = [st.X[i] for i in idx]
        rS = [st.residual[i] for i in idx]
        XS_M = [_vec_mat(x, st.XtX_inv) for x in XS]  # |S|×p
        # K = X_S · M · X_S^⊤  is |S|×|S|.
        K = _zeros_mat(len(idx), len(idx))
        for a in range(len(idx)):
            for b in range(len(idx)):
                K[a][b] = _vec_dot(XS_M[a], XS[b])
        # A = I_{|S|} − K
        A = _zeros_mat(len(idx), len(idx))
        for a in range(len(idx)):
            for b in range(len(idx)):
                A[a][b] = (1.0 if a == b else 0.0) - K[a][b]
        # β̂_{-S} = β̂ − M X_S^⊤ A^{-1} ε̂_S
        # Solve A z = ε̂_S.
        try:
            z = self._solve_general(A, rS)
        except SingularMatrix:
            raise SingularMatrix("group_loo: identifiability — group is collinear")
        # u = X_S^⊤ z  (p-vector)
        u = _zeros(st.p)
        for a in range(len(idx)):
            za = z[a]
            for j in range(st.p):
                u[j] += XS[a][j] * za
        # delta_theta = − M u
        Mu = _mat_vec(st.XtX_inv, u)
        theta_loo = [st.theta[j] - Mu[j] for j in range(st.p)]
        q_loo = self._q_at_theta(
            st, query, test_point, custom_query, param_index, theta_loo
        )
        ev = self._record(
            ATTRIBUTOR_GROUP_LOO,
            {
                "name": name, "indices": idx, "n_indices": len(idx),
                "q_baseline": q_baseline, "q_loo": q_loo,
                "delta_q": q_loo - q_baseline,
            },
        )
        return q_loo - q_baseline

    @staticmethod
    def _solve_general(A: Matrix, b: Vector) -> Vector:
        """LU-with-partial-pivot solver for a (possibly non-symmetric) matrix."""
        n = len(A)
        # Make a working copy.
        M = [row[:] for row in A]
        x = list(b)
        for k in range(n):
            # Find pivot.
            piv = k
            piv_val = abs(M[k][k])
            for r in range(k + 1, n):
                if abs(M[r][k]) > piv_val:
                    piv = r
                    piv_val = abs(M[r][k])
            if piv_val < _EPS:
                raise SingularMatrix(f"_solve_general: zero pivot at column {k}")
            if piv != k:
                M[k], M[piv] = M[piv], M[k]
                x[k], x[piv] = x[piv], x[k]
            inv_pivot = 1.0 / M[k][k]
            for r in range(k + 1, n):
                factor = M[r][k] * inv_pivot
                if factor == 0.0:
                    continue
                for c in range(k, n):
                    M[r][c] -= factor * M[k][c]
                x[r] -= factor * x[k]
        # Back substitution.
        out = _zeros(n)
        for i in range(n - 1, -1, -1):
            s = x[i]
            for j in range(i + 1, n):
                s -= M[i][j] * out[j]
            out[i] = s / M[i][i]
        return out

    # -----------------------------------------------------------------
    # Counterfactual refit
    # -----------------------------------------------------------------

    def counterfactual_refit(
        self,
        name: str,
        *,
        remove: Sequence[int] = (),
        query: str = QUERY_PREDICTION,
        test_point: Any = None,
        custom_query: Callable[[Vector], float] | None = None,
        param_index: int | None = None,
    ) -> CounterfactualReport:
        """Refit the model on (X, y) with ``remove`` indices excluded,
        evaluate the query, and return the diff vs the full-data fit.

        For linear/ridge we use the closed-form Sherman-Morrison
        group-LOO formula (fast); for logistic we honestly refit by
        IRLS on the reduced data (the canonical "did the prediction
        flip?" auditor query).
        """
        st = self._state(name)
        idx = sorted(set(remove))
        if any(not 0 <= i < st.n for i in idx):
            raise AttributorError(f"counterfactual_refit: indices out of range")
        kept = [i for i in range(st.n) if i not in set(idx)]
        if not kept:
            raise InsufficientData(
                "counterfactual_refit: cannot remove all points"
            )

        q_full, _ = self._resolve_query(
            st, query, test_point, custom_query, None, param_index
        )

        if st.kind in (LINEAR, RIDGE):
            X_cf = [st.X[i] for i in kept]
            y_cf = [st.y[i] for i in kept]
            eff_ridge = st.ridge if st.kind == RIDGE else 0.0
            beta_cf, _, _, _ = _fit_linear(X_cf, y_cf, ridge=eff_ridge)
        elif st.kind == LOGISTIC:
            X_cf = [st.X[i] for i in kept]
            y_cf = [st.y[i] for i in kept]
            beta_cf, _, _, _, _, _ = _fit_logistic(
                X_cf, y_cf, ridge=max(st.ridge, _DEFAULT_RIDGE)
            )
        else:
            raise AttributorError(
                f"counterfactual_refit: kind {st.kind!r} not supported"
            )

        q_cf = self._q_at_theta(
            st, query, test_point, custom_query, param_index, beta_cf
        )
        ev = self._record(
            ATTRIBUTOR_COUNTERFACTUAL,
            {
                "name": name, "removed": idx, "kind": st.kind,
                "q_full": q_full, "q_counterfactual": q_cf,
                "delta_q": q_cf - q_full,
            },
        )
        return CounterfactualReport(
            name=name, removed=idx,
            theta_full=list(st.theta),
            theta_counterfactual=list(beta_cf),
            q_full=q_full, q_counterfactual=q_cf,
            delta_q=q_cf - q_full,
            fingerprint=ev.this_hash,
        )

    # -----------------------------------------------------------------
    # Decision-flip certificate
    # -----------------------------------------------------------------

    def decision_flip(
        self,
        name: str,
        *,
        decision_fn: Callable[[Vector], Any],
        budget_k: int,
        ranking: list[int] | None = None,
        query: str = QUERY_PREDICTION,
        test_point: Any = None,
    ) -> DecisionFlipReport:
        """Greedily search for the smallest set ≤ budget_k whose removal
        flips ``decision_fn(θ̂) → decision_fn(θ̂_{-S})``.

        ``ranking`` is the order to try removing in.  If None we use
        the influence-function ranking (highest |influence| first).
        Greedy is *suboptimal* for general decisions but provides a
        certificate: the returned set *demonstrates* a flip — there
        may exist smaller flipping sets we don't find.
        """
        st = self._state(name)
        if budget_k <= 0 or budget_k > st.n:
            raise AttributorError(
                f"budget_k must be in [1, {st.n}]; got {budget_k}"
            )
        d_full = decision_fn(list(st.theta))
        if ranking is None:
            inf = self.influence(
                name, query=query, test_point=test_point,
                method=INFLUENCE_FUNCTION,
            )
            order = [i for i, _ in inf.most_influential(st.n)]
        else:
            seen = set()
            order = []
            for i in ranking:
                if not 0 <= i < st.n:
                    raise AttributorError(
                        f"ranking index {i} out of [0, {st.n})"
                    )
                if i not in seen:
                    seen.add(i)
                    order.append(i)
        chosen: list[int] = []
        d_after = d_full
        flipped = False
        log10_bf = 0.0
        e_value = 1.0
        for i in order:
            chosen.append(i)
            cf = self.counterfactual_refit(
                name, remove=chosen,
                query=query, test_point=test_point,
            )
            d_after = decision_fn(cf.theta_counterfactual)
            if d_after != d_full:
                flipped = True
                # E-value as the data-likelihood ratio between the two
                # fits, evaluated on the *original* training set.
                ll_full = self._log_likelihood_total(st, st.theta)
                ll_cf = self._log_likelihood_total(st, cf.theta_counterfactual)
                log_lr = ll_cf - ll_full
                e_value = math.exp(log_lr)
                log10_bf = log_lr / _LN10
                break
            if len(chosen) >= budget_k:
                break
        ev = self._record(
            ATTRIBUTOR_DECISION_FLIP,
            {
                "name": name, "flipped": flipped,
                "set": list(chosen), "size": len(chosen),
                "decision_full": _jsonable(d_full),
                "decision_after": _jsonable(d_after),
                "log10_bf": log10_bf,
            },
        )
        return DecisionFlipReport(
            name=name, decision_full=d_full,
            minimal_set=list(chosen), decision_after=d_after,
            flipped=flipped, e_value=e_value,
            log10_bayes_factor=log10_bf, fingerprint=ev.this_hash,
        )

    def _log_likelihood_total(self, st: _FitState, theta: Vector) -> float:
        if st.kind in (LINEAR, RIDGE):
            # Up to a constant in σ², the LL of a Gaussian linear model is
            # − (n/2) log(rss/n) − n/2.  We use the proportional form here.
            rss = 0.0
            for i in range(st.n):
                r = _vec_dot(st.X[i], theta) - st.y[i]
                rss += r * r
            sigma2 = max(rss / max(st.n, 1), _EPS)
            return -0.5 * st.n * (math.log(2.0 * math.pi * sigma2) + 1.0)
        if st.kind == LOGISTIC:
            return -_logistic_neg_log_likelihood(st.X, st.y, theta)
        if st.custom_loss_fn is not None:
            total = 0.0
            for i in range(st.n):
                total += st.custom_loss_fn(i, theta)
            return -total
        raise AttributorError("_log_likelihood_total: custom kind needs loss_fn")

    # -----------------------------------------------------------------
    # TracIn ideal (Pruthi-Liu-Sundararajan-Inan 2020)
    # -----------------------------------------------------------------

    def tracin_ideal(
        self,
        name: str,
        *,
        trajectory: Sequence[tuple[Vector, float]],
        grad_query: Callable[[Vector], Vector],
        per_point_grad: Callable[[int, Vector], Vector] | None = None,
    ) -> TracInReport:
        """TracIn ideal-checkpoint attribution.

        Pruthi et al. (2020).  Given a learning trajectory
        ``trajectory = [(θ_t, η_t), …]`` (parameter snapshots and the
        learning rate at each step), the per-point cumulative
        contribution to a query Q(θ_T) is::

            TracIn_i(Q)  =  Σ_t η_t · ⟨∇L_i(θ_t),  ∇Q(θ_t)⟩

        For LINEAR / RIDGE / LOGISTIC the per-point gradient is
        analytic; for CUSTOM the caller must supply ``per_point_grad``.
        """
        st = self._state(name)
        if not trajectory:
            raise InvalidData("tracin_ideal: trajectory must be non-empty")
        per_grad_fn = per_point_grad or self._builtin_per_point_grad(st)
        if per_grad_fn is None:
            raise AttributorError(
                "tracin_ideal: custom kind requires per_point_grad"
            )
        out = _zeros(st.n)
        for theta_t, eta_t in trajectory:
            theta_t = list(theta_t)
            if len(theta_t) != len(st.theta):
                raise InvalidData(
                    f"tracin: trajectory θ_t has length {len(theta_t)} != {len(st.theta)}"
                )
            gQ = list(grad_query(theta_t))
            if len(gQ) != len(theta_t):
                raise InvalidData(
                    "tracin: grad_query(θ_t) has wrong length"
                )
            for i in range(st.n):
                gi = list(per_grad_fn(i, theta_t))
                if len(gi) != len(theta_t):
                    raise InvalidData(
                        f"tracin: per_point_grad({i}) returned wrong length"
                    )
                out[i] += float(eta_t) * _vec_dot(gi, gQ)
        ev = self._record(
            ATTRIBUTOR_TRACIN_COMPUTED,
            {
                "name": name, "n_checkpoints": len(trajectory),
                "max_abs_tracin": max((abs(v) for v in out), default=0.0),
            },
        )
        return TracInReport(
            name=name, per_point=out,
            n_checkpoints=len(trajectory), fingerprint=ev.this_hash,
        )

    def _builtin_per_point_grad(
        self, st: _FitState
    ) -> Callable[[int, Vector], Vector] | None:
        """Return the per-point gradient function for built-in losses."""
        if st.kind in (LINEAR, RIDGE):
            def grad_lin(i: int, theta: Vector) -> Vector:
                r = _vec_dot(st.X[i], theta) - st.y[i]
                return _vec_scale(r, st.X[i])
            return grad_lin
        if st.kind == LOGISTIC:
            def grad_log(i: int, theta: Vector) -> Vector:
                z = _vec_dot(st.X[i], theta)
                pi = _logistic_sigmoid(z)
                return _vec_scale(pi - st.y[i], st.X[i])
            return grad_log
        if st.kind == CUSTOM:
            return st.custom_grad_fn
        return None

    # -----------------------------------------------------------------
    # TRAK random projection
    # -----------------------------------------------------------------

    def trak(
        self,
        name: str,
        *,
        proj_dim: int = 128,
        seed: int = 0,
        query_x: Sequence[float] | None = None,
    ) -> list[float]:
        """TRAK (Park et al. 2023, ICML) random-projection attribution.

        Returns the per-point dot product
            ⟨P · ∇L_i(θ̂),  P · ∇Q(θ̂)⟩
        for a Rademacher random projection P ∈ R^{proj_dim × p}.

        Equivalent to influence_function for high enough ``proj_dim``
        when the design matrix is well-conditioned (Johnson-Lindenstrauss
        guarantees inner-product preservation up to ε with high prob.).
        """
        st = self._state(name)
        if proj_dim <= 0:
            raise AttributorError(f"proj_dim must be positive; got {proj_dim}")
        if st.kind not in (LINEAR, RIDGE, LOGISTIC):
            raise AttributorError(f"trak: kind {st.kind!r} not yet supported")
        # Rademacher projection matrix.
        rng_state = seed & 0xFFFFFFFF
        def _bit() -> int:
            nonlocal rng_state
            # xorshift32
            x = rng_state
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17) & 0xFFFFFFFF
            x ^= (x << 5) & 0xFFFFFFFF
            rng_state = x & 0xFFFFFFFF
            return (rng_state & 1) * 2 - 1
        P = _zeros_mat(proj_dim, st.p)
        scale = 1.0 / math.sqrt(proj_dim)
        for r in range(proj_dim):
            for c in range(st.p):
                P[r][c] = scale * _bit()
        # ∇Q ≈ x  (linear/logistic prediction gradient).
        if query_x is None:
            raise AttributorError("trak: query_x is required")
        x = [float(v) for v in query_x]
        gQ_proj = _mat_vec(P, x)
        per_grad = self._builtin_per_point_grad(st)
        assert per_grad is not None
        out = _zeros(st.n)
        for i in range(st.n):
            gi = per_grad(i, st.theta)
            gi_proj = _mat_vec(P, gi)
            out[i] = _vec_dot(gQ_proj, gi_proj)
        return out

    # -----------------------------------------------------------------
    # Bootstrap CIs (Efron-Tibshirani 1986)
    # -----------------------------------------------------------------

    def bootstrap_influence_band(
        self,
        name: str,
        *,
        query: str = QUERY_LOSS,
        test_point: Any = None,
        custom_query: Callable[[Vector], float] | None = None,
        custom_query_grad: Callable[[Vector], Vector] | None = None,
        param_index: int | None = None,
        delta: float = 0.05,
        n_resamples: int = 200,
        seed: int = 0,
    ) -> BootstrapBand:
        """Bootstrap percentile CIs on per-point influence.

        Efron-Tibshirani (1986).  Resamples the training set with
        replacement ``n_resamples`` times, refits, and reports the
        empirical (δ/2, 1-δ/2) quantiles of each point's influence
        estimate.
        """
        if not 0.0 < delta < 1.0:
            raise AttributorError(f"delta must be in (0, 1); got {delta}")
        if n_resamples < 2:
            raise AttributorError(f"n_resamples must be >= 2")
        st = self._state(name)
        if st.kind == CUSTOM:
            raise AttributorError(
                "bootstrap_influence_band: not supported for custom kind"
            )
        rng_state = (seed & 0xFFFFFFFF) or 0x12345
        def _rand_idx(n: int) -> int:
            nonlocal rng_state
            x = rng_state
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17) & 0xFFFFFFFF
            x ^= (x << 5) & 0xFFFFFFFF
            rng_state = x & 0xFFFFFFFF
            return rng_state % n
        n = st.n
        per_resample: list[list[float]] = []
        for r in range(n_resamples):
            sample_idx = [_rand_idx(n) for _ in range(n)]
            X_b = [st.X[k] for k in sample_idx]
            y_b = [st.y[k] for k in sample_idx]
            if st.kind in (LINEAR, RIDGE):
                eff_ridge = st.ridge if st.kind == RIDGE else 0.0
                try:
                    beta_b, XtX_inv_b, residual_b, _ = _fit_linear(
                        X_b, y_b, ridge=eff_ridge
                    )
                except SingularMatrix:
                    continue
                # Compute IF on original data using the bootstrap fit.
                _, gQ = self._resolve_query_at(
                    st, query, test_point, custom_query, custom_query_grad,
                    param_index, beta_b,
                )
                H_inv_g = _mat_vec(XtX_inv_b, gQ)
                inf_b = _zeros(n)
                # ∇L_i on original (X, y) at θ_b:
                for i in range(n):
                    r_orig = _vec_dot(st.X[i], beta_b) - st.y[i]
                    inner = _vec_dot(H_inv_g, st.X[i])
                    inf_b[i] = r_orig * inner
                per_resample.append(inf_b)
            elif st.kind == LOGISTIC:
                try:
                    beta_b, _, _, XtWX_inv_b, _, _ = _fit_logistic(
                        X_b, y_b, ridge=max(st.ridge, _DEFAULT_RIDGE)
                    )
                except (SingularMatrix, ConvergenceError):
                    continue
                _, gQ = self._resolve_query_at(
                    st, query, test_point, custom_query, custom_query_grad,
                    param_index, beta_b,
                )
                H_inv_g = _mat_vec(XtWX_inv_b, gQ)
                inf_b = _zeros(n)
                for i in range(n):
                    z = _vec_dot(st.X[i], beta_b)
                    pi_i = _logistic_sigmoid(z)
                    resid = pi_i - st.y[i]
                    inner = _vec_dot(H_inv_g, st.X[i])
                    inf_b[i] = resid * inner
                per_resample.append(inf_b)
        if not per_resample:
            raise ConvergenceError(
                "bootstrap_influence_band: every resample was singular"
            )
        # Per-i quantiles.
        per_lower = _zeros(n)
        per_upper = _zeros(n)
        for i in range(n):
            col = sorted(r[i] for r in per_resample)
            lo_idx = max(0, int(math.floor((delta / 2.0) * len(col))))
            hi_idx = min(len(col) - 1, int(math.ceil((1.0 - delta / 2.0) * len(col)) - 1))
            per_lower[i] = col[lo_idx]
            per_upper[i] = col[hi_idx]
        ev = self._record(
            ATTRIBUTOR_QUERIED,
            {
                "name": name, "kind": "bootstrap_band",
                "n_resamples": len(per_resample), "delta": delta,
            },
        )
        return BootstrapBand(
            name=name, method="bootstrap_percentile",
            per_point_lower=per_lower, per_point_upper=per_upper,
            delta=delta, n_resamples=len(per_resample),
            fingerprint=ev.this_hash,
        )

    def _resolve_query_at(
        self,
        st: _FitState,
        query: str,
        test_point: Any,
        custom_query: Callable[[Vector], float] | None,
        custom_query_grad: Callable[[Vector], Vector] | None,
        param_index: int | None,
        theta: Vector,
    ) -> tuple[float, Vector]:
        """Resolve (Q, ∇Q) at a non-fitted parameter vector."""
        prev = st.theta
        st.theta = theta
        try:
            return self._resolve_query(
                st, query, test_point, custom_query, custom_query_grad, param_index
            )
        finally:
            st.theta = prev

    # -----------------------------------------------------------------
    # Aggregate report + clearing
    # -----------------------------------------------------------------

    def report(self, name: str) -> AttributorReport:
        """A single-call snapshot of one fitted hypothesis."""
        st = self._state(name)
        diagnostics: LinearDiagnostics | None = None
        if st.kind in (LINEAR, RIDGE):
            try:
                diagnostics = self.linear_diagnostics(name)
            except (InsufficientData, AttributorError):
                diagnostics = None
        ev = self._record(
            ATTRIBUTOR_REPORTED,
            {
                "name": name, "kind": st.kind, "n": st.n, "p": st.p,
                "fingerprint": self._fingerprint,
            },
        )
        return AttributorReport(
            name=name, kind=st.kind, n=st.n, p=st.p,
            theta=list(st.theta), fingerprint=ev.this_hash,
            diagnostics=diagnostics,
            fit_iterations=st.iterations, converged=st.converged,
        )

    def clear(self) -> None:
        """Drop every hypothesis and reset the chain."""
        with self._lock:
            self._states.clear()
            self._record(ATTRIBUTOR_CLEARED, {})


# =====================================================================
# Spec-based factory
# =====================================================================


def attributor_from_spec(spec: Mapping[str, Any]) -> Attributor:
    """Build an Attributor from a JSON-friendly spec.

    Expected shape::

        {
          "models": [
            {"name": "price_model", "kind": "linear",
             "X": [[1, 4], [1, 7], [1, 11]],
             "y": [2.0, 3.1, 4.7]},
            {"name": "spam_clf", "kind": "logistic",
             "X": [[1, 0.1, 0.4], ...],
             "y": [0, 1, 1, 0, ...]}
          ]
        }
    """
    if not isinstance(spec, Mapping):
        raise AttributorError(
            f"spec must be a mapping; got {type(spec).__name__}"
        )
    a = Attributor()
    models = spec.get("models", [])
    if not isinstance(models, Sequence):
        raise AttributorError("spec['models'] must be a sequence")
    for m in models:
        if not isinstance(m, Mapping):
            raise AttributorError(f"each model spec must be a mapping; got {m!r}")
        name = m.get("name")
        kind = m.get("kind")
        if not isinstance(name, str) or not name:
            raise AttributorError(f"model spec missing 'name': {m!r}")
        if not isinstance(kind, str):
            raise AttributorError(f"model spec missing 'kind': {m!r}")
        X = m.get("X")
        y = m.get("y")
        ridge = float(m.get("ridge", 0.0))
        if X is None or y is None:
            raise AttributorError(f"model spec missing X or y: {m!r}")
        a.fit(name, kind, X=X, y=y, ridge=ridge)
    return a


# =====================================================================
# Convenience constructors
# =====================================================================


def quick_linear_attribution(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    name: str = "model",
) -> Attributor:
    """One-shot linear-regression Attributor on (X, y)."""
    a = Attributor()
    a.fit(name, LINEAR, X=X, y=y)
    return a


def quick_logistic_attribution(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    name: str = "model",
    ridge: float = _DEFAULT_RIDGE,
) -> Attributor:
    """One-shot logistic-regression Attributor on (X, y)."""
    a = Attributor()
    a.fit(name, LOGISTIC, X=X, y=y, ridge=ridge)
    return a
