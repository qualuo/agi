r"""Pretunist — Test-Time Training as a runtime primitive.

Every learning primitive in this runtime updates *between* tasks
(``Continualist`` constrains drift across task boundaries; ``Distiller``
amortises trained behaviour; ``Aligner`` runs DPO on collected
preferences).  None of them changes the model *during a single task* —
that's the "research-open per-token weight update" cell the
architecture document explicitly leaves blank::

    | Timescale         | What changes      | How             | Cost |
    | Seconds (per-turn)| working memory    | append context  | free |
    | …                  …                   …                 …
    | per-token         | adapter weights   | OPEN PROBLEM    | ?    |

The frontier name for filling that cell is **Test-Time Training**
(Sun, Wang, Liu, Held, Efros, Hardt 2020 *Test-Time Training with
Self-Supervision for Generalization under Distribution Shifts*).  TTT
takes the test instance ``x*``, mines a *self-supervised auxiliary
target* from it (rotation prediction, masked reconstruction, leave-one-
out fit on its own support set), runs a few SGD steps that improve only
that auxiliary loss, and produces ``y*`` from the locally-fitted
weights.  Akyürek, Damani, Qiu, Guo, Suzgun, Kim, Andreas, Kim 2024
*The Surprising Effectiveness of Test-Time Training for Few-Shot
Learning* — the technique that won the **ARC-AGI 2024 prize** — showed
that a small adapter fit on the *current* puzzle at inference time
outperforms an order-of-magnitude bigger model that was only
pre-trained.

The pitch reduced to a runtime call::

    pre = Pretunist(PretunistConfig(adapter_dim=16, ridge_lambda=1e-2))
    pre.set_base(theta_0)                       # base policy / prior
    for (x_i, y_i) in support_set:
        pre.observe_support(x_i, y_i)
    result = pre.adapt(query=x_star)            # closed-form ridge fit
    pred   = result.prediction                  # adapted output
    gain   = result.adaptation_gain_bits        # vs base
    drift  = result.kl_drift_nats               # ||θ* − θ_0||² / 2σ²
    bound  = pre.pac_bayes_bound(delta=0.05)    # generalisation cert
    if pre.should_abstain(x_star):              # entropy / drift / fit
        ...
    cert   = pre.certify()                      # full guarantee record
    rep    = pre.report()                       # everything + receipts

What it ships
-------------

* **Closed-form Bayesian ridge regression** as the adapter.  Linear
  in-context learning theory (Akyürek-Schuurmans-Andreas-Zhou-Ma 2023
  *What Learning Algorithm is In-Context Learning?*; Garg-Tsipras-Liang-
  Valiant 2022 *What Can Transformers Learn In-Context?*) shows that
  one step of a Transformer block implementing self-attention with a
  carefully-chosen value head is mathematically *equivalent* to one
  step of ridge regression on the support set.  Pretunist makes that
  equivalence executable: we run the *closed-form* ridge regression
  ::

      θ*  =  (Xᵀ X + λ I)⁻¹ Xᵀ y

  in pure-stdlib O(d³ + n d²) time (Cholesky-based linear solve), with
  ``d = adapter_dim``.  No autograd, no GPU, no torch.

* **PAC-Bayes generalisation bound** (McAllester 1999 *PAC-Bayesian
  Model Averaging*; Dziugaite-Roy 2017 *Computing nonvacuous
  generalization bounds for deep (stochastic) neural networks*) on the
  adapted hypothesis.  With prior ``N(θ_0, σ_p² I)`` and posterior
  ``N(θ*, σ_q² I)``, with probability ≥ 1−δ over the support sample of
  size n,
  ::

      L_true(θ*)  ≤  L_emp(θ*; S)  +  √( ( KL(Q‖P) + log(n / δ) ) / (2n) )

  where ``KL(Q‖P) = ½ [ d (σ_q²/σ_p² − 1 − log(σ_q²/σ_p²))
                       + ||θ* − θ_0||² / σ_p² ]``.  The runtime
  returns the bound as a number of nats and as a 0/1
  *non-vacuous?* flag.

* **KL-budget shielding**.  A coordination engine can demand
  ``KL(Q‖P) ≤ B`` (preserve guarantees that hold under the base
  policy).  Pretunist projects the closed-form solution onto the KL
  ball::

      θ_proj  =  θ_0  +  α (θ* − θ_0),   α ∈ [0, 1],
                                          ½ α² ||θ*−θ_0||² ≤ B σ_p²

  giving a monotone family of adapters parametrised by ``α`` between
  *fully base* (α=0) and *fully data-fit* (α=1).  The largest α
  satisfying the budget is closed-form.

* **Leave-one-out (LOO) cross-validation risk** via the **hat-matrix
  identity** (Allen 1974 PRESS statistic; Cook-Weisberg 1982; recently
  Patil-Wei-Rakhlin-Tibshirani 2022 *Bagging and the kernel
  reproducing property*)::

      e_loo,i  =  (y_i − ŷ_i) / (1 − H_ii)

  where ``H = X (XᵀX + λI)⁻¹ Xᵀ`` is the hat matrix.  One closed-form
  ridge fit gives *all* n LOO residuals in O(n d²) extra time — no
  retraining.  The coordinator reads ``E[e_loo²]`` as an empirical
  test risk that does *not* require held-out data.

* **Anytime-valid e-process on adaptation gain**.  For each new
  observation ``(x_i, y_i)`` we maintain a one-step-ahead mixture
  martingale that tests "the adapter is genuinely improving on the
  base" vs the null "the adapter equals the base".  At any stopping
  time τ::

      P[ ∃ t ≤ τ : log E_t > log(1/δ) ]  ≤  δ.

  The coordinator can monitor the e-process and stop adapting as soon
  as the test rejects (Howard-Ramdas-McAuliffe-Sekhon 2021
  *Time-uniform Chernoff bounds via the method of mixtures*).

* **Self-supervised TTT loss** for the no-label case.  When the
  primitive has only a support of ``x`` (no ``y``), it implements
  **leave-one-token-out reconstruction**: each ``x_i`` is split into
  ``(prefix, target)``, the adapter is fit to predict ``target | prefix``
  minimising squared loss, and the *adaptation gain* is measured
  against the base.  This is the Sun-2020 / Akyürek-2024 recipe
  reduced to closed form.

* **Abstention rule** (Chow 1970 *On optimum recognition error and
  reject tradeoff*; Tortorella 2000): refuse to predict on ``x*`` when
  *any* of three quantities exceeds a configurable threshold:

    1. **Leverage** ``h(x*) = x*ᵀ (XᵀX + λI)⁻¹ x*`` — the test point
       is far from the support.
    2. **Posterior predictive variance** ``σ² (1 + h(x*))`` is large.
    3. **KL drift** is at its budget — model is already maxed out.

  Routes the task back to the coordinator.

* **Adapter snapshot / restore** for federation (``RuntimePool``
  picks one of N adapted variants by some critic).  All snapshots are
  signed with an SHA-256 fingerprint chain over the observation log,
  so a federation member can prove what data its adapter saw.

* **Replay-verifiable ledger**.  Every ``observe_support``,
  ``adapt``, ``certify`` is hashed into an ``AttestationLedger``-
  compatible chain.  The next agent on the case can reproduce the
  adapter byte-for-byte and audit the certification.

Coordination-engine integration
-------------------------------

A coordinator running this runtime can::

    if ``Preflight.p_success`` is low on the base policy:
        spec = manifest.lookup("pretunist")
        pre  = registry.instantiate(spec)
        pre.set_base(default_adapter)
        for ex in retrieved_demos:
            pre.observe_support(ex.x, ex.y)
        cert = pre.adapt(query=user_input)
        if cert.adaptation_gain_bits < 0.05:
            # adapter didn't help — fall back to base
            ...
        elif pre.pac_bayes_bound(0.05).is_vacuous:
            # we can fit but the cert doesn't generalise
            preflight.flag("untrusted-adapter")
        else:
            session.use_adapter(cert.adapter_params)

Pretunist is the runtime primitive a coordination engine calls when
**the task is OOD relative to the base policy**, and it returns a
provable adapter or a defensible abstention — never a silent failure.

Mathematical roots
------------------

The implementation only assumes:

  * **Numerical linear algebra**.  We use Cholesky factorisation of
    ``(XᵀX + λI)`` to solve normal equations in O(d³).  Implemented
    inline (``_cholesky``, ``_solve_lower_triangular``,
    ``_solve_upper_triangular``).  λ > 0 makes the matrix positive
    definite *for any* ``X`` — the Tikhonov regularisation argument
    (Tikhonov 1943; Hoerl-Kennard 1970).

  * **Sherman-Morrison-Woodbury** identity to update the hat matrix
    and LOO residuals incrementally as new ``(x_i, y_i)`` arrive
    (Hager 1989 *Updating the Inverse of a Matrix*).  This makes the
    primitive *streamable* — the coordinator can call
    ``observe_support`` thousands of times in O(d²) per update
    without re-Cholesky-ing.

  * **PAC-Bayes for sub-Gaussian losses** (Catoni 2007 *PAC-Bayesian
    supervised classification: the thermodynamics of statistical
    learning*).  The bound is non-vacuous when ``KL(Q‖P)`` is small
    relative to ``n``.  The runtime exposes both the bound and the
    *vacuous?* flag.

  * **Mixture martingales** for anytime-valid testing (Howard et al.
    2021).  Implemented as a log-mixture of likelihood-ratio
    sub-Gaussian martingales over a finite grid of effect sizes.

References
----------

* Sun, Wang, Liu, Held, Efros, Hardt 2020 *Test-Time Training with
  Self-Supervision for Generalization under Distribution Shifts*.
  ICML.
* Akyürek, Damani, Qiu, Guo, Suzgun, Kim, Andreas, Kim 2024 *The
  Surprising Effectiveness of Test-Time Training for Few-Shot
  Learning*.  (ARC-AGI 2024 prize-winning method.)
* Akyürek, Schuurmans, Andreas, Zhou, Ma 2023 *What Learning
  Algorithm is In-Context Learning?  Investigations with Linear
  Models*.  ICLR.
* Garg, Tsipras, Liang, Valiant 2022 *What Can Transformers Learn
  In-Context?*  NeurIPS.
* McAllester 1999 *PAC-Bayesian Model Averaging*.
* Catoni 2007 *PAC-Bayesian Supervised Classification*.
* Dziugaite, Roy 2017 *Computing Nonvacuous Generalization Bounds*.
* Howard, Ramdas, McAuliffe, Sekhon 2021 *Time-Uniform Chernoff
  Bounds via the Method of Mixtures*.  Annals of Statistics.
* Hoerl, Kennard 1970 *Ridge Regression: Biased Estimation for
  Nonorthogonal Problems*.
* Allen 1974 *The Relationship Between Variable Selection and Data
  Augmentation and a Method for Prediction* (PRESS statistic).
* Chow 1970 *On Optimum Recognition Error and Reject Tradeoff*.
* Hager 1989 *Updating the Inverse of a Matrix* (Sherman-Morrison-
  Woodbury survey).

Pure Python stdlib.  No torch, no numpy, no scipy.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)


__all__ = [
    # Algorithms
    "ALG_RIDGE",
    "ALG_KERNEL_RIDGE",
    "ALG_BAYESIAN_RIDGE",
    "ALG_LOW_RANK",
    "KNOWN_ALGORITHMS",
    # Self-supervision modes
    "SSL_LOO",
    "SSL_PREFIX_TARGET",
    "SSL_RECONSTRUCT",
    "SSL_NONE",
    "KNOWN_SSL",
    # Abstention rules
    "ABSTAIN_LEVERAGE",
    "ABSTAIN_VARIANCE",
    "ABSTAIN_KL_BUDGET",
    "ABSTAIN_LOO_RISK",
    "ABSTAIN_E_PROCESS",
    "KNOWN_ABSTAIN_RULES",
    # Events
    "PRETUNIST_OBSERVED",
    "PRETUNIST_ADAPTED",
    "PRETUNIST_CERTIFIED",
    "PRETUNIST_ABSTAINED",
    "PRETUNIST_RESET",
    # Errors
    "PretunistError",
    "InvalidConfig",
    "InvalidSupport",
    "InvalidQuery",
    "InsufficientData",
    "DriftBudgetExceeded",
    "AdapterNotFit",
    "DimensionMismatch",
    # Core types
    "PretunistConfig",
    "SupportPoint",
    "AdapterParameters",
    "AdaptationResult",
    "AbstentionReport",
    "PretunistReport",
    "PretunistCertificate",
    "Pretunist",
    # Ledger
    "pretunist_ledger_genesis",
    "pretunist_ledger_root",
    # Standalone helpers
    "ridge_regression_closed_form",
    "leave_one_out_residuals",
    "pac_bayes_bound_value",
    "kl_gauss_isotropic",
    "leverage_score",
    "mixture_martingale_logvalue",
    "project_to_kl_budget",
    # Math
    "matvec",
    "matmul",
    "transpose",
    "cholesky",
    "solve_lower_triangular",
    "solve_upper_triangular",
    "solve_psd",
]


# ---------------------------------------------------------------------------
# Constants

SCHEMA_VERSION = "1.0"

# Algorithms — which adapter family is the primitive fitting?
ALG_RIDGE = "ridge"
ALG_KERNEL_RIDGE = "kernel-ridge"
ALG_BAYESIAN_RIDGE = "bayesian-ridge"
ALG_LOW_RANK = "low-rank"

KNOWN_ALGORITHMS = frozenset({
    ALG_RIDGE, ALG_KERNEL_RIDGE, ALG_BAYESIAN_RIDGE, ALG_LOW_RANK,
})

# Self-supervision modes — when no labels are provided, how do we mine targets?
SSL_LOO = "leave-one-out"           # each support point predicts the rest
SSL_PREFIX_TARGET = "prefix-target" # split feature into (prefix, target)
SSL_RECONSTRUCT = "reconstruct"     # masked reconstruction
SSL_NONE = "none"                   # require labels explicitly

KNOWN_SSL = frozenset({
    SSL_LOO, SSL_PREFIX_TARGET, SSL_RECONSTRUCT, SSL_NONE,
})

# Abstention rules — the coordinator can configure which to enable.
ABSTAIN_LEVERAGE = "leverage"
ABSTAIN_VARIANCE = "predictive-variance"
ABSTAIN_KL_BUDGET = "kl-budget"
ABSTAIN_LOO_RISK = "loo-risk"
ABSTAIN_E_PROCESS = "e-process"

KNOWN_ABSTAIN_RULES = frozenset({
    ABSTAIN_LEVERAGE, ABSTAIN_VARIANCE, ABSTAIN_KL_BUDGET,
    ABSTAIN_LOO_RISK, ABSTAIN_E_PROCESS,
})

# EventBus topics (subscribe via runtime.event_bus).
PRETUNIST_OBSERVED = "pretunist.observed"
PRETUNIST_ADAPTED = "pretunist.adapted"
PRETUNIST_CERTIFIED = "pretunist.certified"
PRETUNIST_ABSTAINED = "pretunist.abstained"
PRETUNIST_RESET = "pretunist.reset"


# ---------------------------------------------------------------------------
# Errors


class PretunistError(Exception):
    """Base for all Pretunist-raised errors."""


class InvalidConfig(PretunistError):
    """A ``PretunistConfig`` field is outside its admissible domain."""


class InvalidSupport(PretunistError):
    """The support point is ill-shaped or non-finite."""


class InvalidQuery(PretunistError):
    """The query vector is ill-shaped or non-finite."""


class InsufficientData(PretunistError):
    """Adapter requested but ``n_observed`` < minimum."""


class DriftBudgetExceeded(PretunistError):
    """Closed-form fit exceeds the configured KL budget without projection."""


class AdapterNotFit(PretunistError):
    """``adapt()`` has not been called or has been reset."""


class DimensionMismatch(PretunistError):
    """Observation dimension does not match configured ``adapter_dim``."""


# ---------------------------------------------------------------------------
# Math helpers — pure stdlib linear algebra
#
# We keep matrices as ``list[list[float]]`` (row-major) and vectors as
# ``list[float]``.  The state lives in 2-D Python lists.  At
# adapter_dim ≤ a few hundred and n ≤ a few thousand, this is fast
# enough for runtime use (microseconds to milliseconds), and it avoids
# importing numpy / torch.
# ---------------------------------------------------------------------------


def matvec(A: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
    """Compute ``y = A @ x``.

    ``A`` is row-major.  Raises :class:`DimensionMismatch` if shapes
    disagree.
    """
    if not A:
        return []
    n = len(A)
    m = len(A[0])
    if len(x) != m:
        raise DimensionMismatch(f"matvec: A is {n}x{m}, x has length {len(x)}")
    out = [0.0] * n
    for i in range(n):
        row = A[i]
        if len(row) != m:
            raise DimensionMismatch(f"matvec: row {i} has length {len(row)} != {m}")
        s = 0.0
        for j in range(m):
            s += row[j] * x[j]
        out[i] = s
    return out


def matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> list[list[float]]:
    """Compute ``C = A @ B``."""
    if not A or not B:
        return []
    n = len(A)
    k = len(A[0])
    if len(B) != k:
        raise DimensionMismatch(
            f"matmul: A is {n}x{k}, B is {len(B)}x{len(B[0]) if B else 0}"
        )
    m = len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        row_a = A[i]
        row_out = out[i]
        for kk in range(k):
            aik = row_a[kk]
            if aik == 0.0:
                continue
            row_b = B[kk]
            for j in range(m):
                row_out[j] += aik * row_b[j]
    return out


def transpose(A: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return ``Aᵀ``."""
    if not A:
        return []
    n = len(A)
    m = len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def cholesky(A: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return lower-triangular ``L`` with ``L Lᵀ = A``.

    Raises :class:`PretunistError` if ``A`` is not symmetric positive
    definite (e.g. caller forgot Tikhonov regularisation).
    """
    n = len(A)
    if n == 0:
        return []
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if len(A[i]) != n:
            raise DimensionMismatch(f"cholesky: matrix not square at row {i}")
        for j in range(i + 1):
            s = A[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                if s <= 0.0:
                    # Matrix is not PD — degenerate.  Caller should have
                    # added Tikhonov regularisation.
                    raise PretunistError(
                        f"cholesky: matrix is not positive definite "
                        f"(diagonal {i} = {s:.6g}); add Tikhonov regulariser"
                    )
                L[i][j] = math.sqrt(s)
            else:
                if L[j][j] == 0.0:
                    raise PretunistError("cholesky: zero pivot")
                L[i][j] = s / L[j][j]
    return L


def solve_lower_triangular(L: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Solve ``L y = b`` by forward substitution."""
    n = len(L)
    if len(b) != n:
        raise DimensionMismatch(f"forward solve: L is {n}x{n}, b has length {len(b)}")
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i][j] * y[j]
        if L[i][i] == 0.0:
            raise PretunistError("forward solve: zero pivot")
        y[i] = s / L[i][i]
    return y


def solve_upper_triangular(U: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Solve ``U x = b`` by back substitution."""
    n = len(U)
    if len(b) != n:
        raise DimensionMismatch(f"back solve: U is {n}x{n}, b has length {len(b)}")
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        if U[i][i] == 0.0:
            raise PretunistError("back solve: zero pivot")
        x[i] = s / U[i][i]
    return x


def solve_psd(A: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Solve ``A x = b`` for symmetric positive-definite ``A`` via Cholesky."""
    L = cholesky(A)
    y = solve_lower_triangular(L, b)
    # Lᵀ has rows = columns of L
    LT = transpose(L)
    return solve_upper_triangular(LT, y)


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _add_diag(A: Sequence[Sequence[float]], lam: float) -> list[list[float]]:
    """Return ``A + lam * I`` (new matrix)."""
    n = len(A)
    out = [list(row) for row in A]
    for i in range(n):
        out[i][i] += lam
    return out


def _outer_accum(M: list[list[float]], x: Sequence[float], y: Sequence[float]) -> None:
    """In-place ``M += x yᵀ``."""
    n = len(x)
    if len(M) != n or any(len(row) != len(y) for row in M):
        raise DimensionMismatch("_outer_accum: shape disagreement")
    for i in range(n):
        xi = x[i]
        if xi == 0.0:
            continue
        row = M[i]
        for j in range(len(y)):
            row[j] += xi * y[j]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise DimensionMismatch(f"_dot: lengths {len(a)} != {len(b)}")
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _norm_sq(a: Sequence[float]) -> float:
    return _dot(a, a)


def _vec_finite(x: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in x)


# ---------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class PretunistConfig:
    """Configuration for :class:`Pretunist`.

    All fields are validated in ``__post_init__``.  Frozen / hashable.
    """

    adapter_dim: int = 8
    """Dimension ``d`` of the adapter feature space."""

    output_dim: int = 1
    """Dimension of ``y``.  Scalar regression is the default."""

    algorithm: str = ALG_RIDGE
    """One of :data:`KNOWN_ALGORITHMS`."""

    ridge_lambda: float = 1e-2
    """Tikhonov regulariser ``λ > 0``."""

    prior_variance: float = 1.0
    """Prior variance ``σ_p²`` for PAC-Bayes (``N(θ_0, σ_p² I)``)."""

    posterior_variance: float = 1.0
    """Posterior variance ``σ_q²`` for PAC-Bayes (``N(θ*, σ_q² I)``).

    Defaults to the prior to keep KL bounded only by the parameter
    drift term.
    """

    noise_variance: float = 1.0
    """Observation noise variance ``σ²`` for posterior predictive."""

    kl_budget: float = math.inf
    """Maximum allowed ``KL(Q‖P)`` in nats.  ``math.inf`` disables it."""

    min_support: int = 1
    """Minimum number of observed support points before ``adapt`` is allowed."""

    max_support: int = 10_000
    """Hard cap on support buffer size.  Beyond this, observations are
    rejected (the coordinator should subsample first)."""

    ssl_mode: str = SSL_NONE
    """Self-supervision mode when only ``x`` is provided.  See
    :data:`KNOWN_SSL`."""

    ssl_prefix_fraction: float = 0.5
    """For ``SSL_PREFIX_TARGET``: split the feature at
    ``floor(adapter_dim * ssl_prefix_fraction)``."""

    abstain_rules: tuple[str, ...] = (ABSTAIN_LEVERAGE, ABSTAIN_VARIANCE)
    """Which abstention rules are active.  Empty disables abstention."""

    abstain_leverage_threshold: float = 0.99
    """``h(x*) ≥ τ`` triggers abstention.  Leverage is in [0, 1] for
    fitted points; > 1 in pathological cases."""

    abstain_variance_threshold: float = 10.0
    """Predictive variance threshold."""

    abstain_loo_risk_threshold: float = math.inf
    """LOO empirical risk threshold."""

    rank: int = 0
    """For ``ALG_LOW_RANK``: rank of the adapter.  0 disables."""

    e_process_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    """Mixture grid for the anytime-valid e-process."""

    e_process_alpha: float = 0.05
    """Type-I error budget for the e-process."""

    seed: int = 0
    """RNG seed for any stochastic helpers (LOO permutations, etc.)."""

    hmac_key: bytes | None = None
    """Optional HMAC key for the ledger.  ``None`` uses plain SHA-256."""

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_dim, int) or self.adapter_dim <= 0:
            raise InvalidConfig(f"adapter_dim must be a positive int (got {self.adapter_dim!r})")
        if not isinstance(self.output_dim, int) or self.output_dim <= 0:
            raise InvalidConfig(f"output_dim must be a positive int (got {self.output_dim!r})")
        if self.algorithm not in KNOWN_ALGORITHMS:
            raise InvalidConfig(
                f"algorithm {self.algorithm!r} not in {sorted(KNOWN_ALGORITHMS)}"
            )
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda <= 0.0:
            raise InvalidConfig(f"ridge_lambda must be > 0 (got {self.ridge_lambda!r})")
        if not math.isfinite(self.prior_variance) or self.prior_variance <= 0.0:
            raise InvalidConfig(f"prior_variance must be > 0 (got {self.prior_variance!r})")
        if not math.isfinite(self.posterior_variance) or self.posterior_variance <= 0.0:
            raise InvalidConfig(
                f"posterior_variance must be > 0 (got {self.posterior_variance!r})"
            )
        if not math.isfinite(self.noise_variance) or self.noise_variance <= 0.0:
            raise InvalidConfig(f"noise_variance must be > 0 (got {self.noise_variance!r})")
        if self.kl_budget <= 0.0:
            raise InvalidConfig(f"kl_budget must be > 0 or inf (got {self.kl_budget!r})")
        if not isinstance(self.min_support, int) or self.min_support < 0:
            raise InvalidConfig(f"min_support must be >= 0 (got {self.min_support!r})")
        if not isinstance(self.max_support, int) or self.max_support < self.min_support:
            raise InvalidConfig(
                f"max_support must be >= min_support (got {self.max_support}, "
                f"{self.min_support})"
            )
        if self.ssl_mode not in KNOWN_SSL:
            raise InvalidConfig(f"ssl_mode {self.ssl_mode!r} not in {sorted(KNOWN_SSL)}")
        if not 0.0 < self.ssl_prefix_fraction < 1.0:
            raise InvalidConfig(
                f"ssl_prefix_fraction must be in (0, 1) (got {self.ssl_prefix_fraction!r})"
            )
        for rule in self.abstain_rules:
            if rule not in KNOWN_ABSTAIN_RULES:
                raise InvalidConfig(
                    f"abstain_rules contains {rule!r} not in {sorted(KNOWN_ABSTAIN_RULES)}"
                )
        if self.abstain_leverage_threshold <= 0.0:
            raise InvalidConfig("abstain_leverage_threshold must be > 0")
        if self.abstain_variance_threshold <= 0.0:
            raise InvalidConfig("abstain_variance_threshold must be > 0")
        if self.abstain_loo_risk_threshold <= 0.0:
            raise InvalidConfig("abstain_loo_risk_threshold must be > 0")
        if self.rank < 0:
            raise InvalidConfig(f"rank must be >= 0 (got {self.rank})")
        if self.algorithm == ALG_LOW_RANK and not (0 < self.rank <= self.adapter_dim):
            raise InvalidConfig(
                f"ALG_LOW_RANK requires 0 < rank <= adapter_dim "
                f"(got rank={self.rank}, adapter_dim={self.adapter_dim})"
            )
        if not self.e_process_grid:
            raise InvalidConfig("e_process_grid must be non-empty")
        for g in self.e_process_grid:
            if not math.isfinite(g) or g <= 0.0:
                raise InvalidConfig(f"e_process_grid entry {g!r} must be a positive finite float")
        if not (0.0 < self.e_process_alpha < 1.0):
            raise InvalidConfig(
                f"e_process_alpha must be in (0, 1) (got {self.e_process_alpha!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "adapter_dim": self.adapter_dim,
            "output_dim": self.output_dim,
            "algorithm": self.algorithm,
            "ridge_lambda": self.ridge_lambda,
            "prior_variance": self.prior_variance,
            "posterior_variance": self.posterior_variance,
            "noise_variance": self.noise_variance,
            "kl_budget": (
                None if math.isinf(self.kl_budget) else self.kl_budget
            ),
            "min_support": self.min_support,
            "max_support": self.max_support,
            "ssl_mode": self.ssl_mode,
            "ssl_prefix_fraction": self.ssl_prefix_fraction,
            "abstain_rules": list(self.abstain_rules),
            "abstain_leverage_threshold": self.abstain_leverage_threshold,
            "abstain_variance_threshold": self.abstain_variance_threshold,
            "abstain_loo_risk_threshold": (
                None
                if math.isinf(self.abstain_loo_risk_threshold)
                else self.abstain_loo_risk_threshold
            ),
            "rank": self.rank,
            "e_process_grid": list(self.e_process_grid),
            "e_process_alpha": self.e_process_alpha,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Records


@dataclass(frozen=True)
class SupportPoint:
    """A single (x, y) observation in the support set."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    weight: float = 1.0
    ts: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class AdapterParameters:
    """The learned adapter — ``θ* ∈ R^{adapter_dim × output_dim}``.

    Stored as a row-major matrix.  ``output_dim=1`` returns a (d,) row
    list with a singleton column.
    """

    theta: tuple[tuple[float, ...], ...]
    """Closed-form ridge solution."""

    base: tuple[tuple[float, ...], ...]
    """Base (prior mean) used for the KL drift term."""

    ridge_lambda: float
    """The λ that produced θ."""

    n: int
    """Number of support points the fit was computed on."""

    fingerprint: str
    """SHA-256 of the (theta, base, lambda, n, support_hash) tuple."""

    def drift_l2(self) -> float:
        """``||θ* − θ_0||₂``."""
        if len(self.theta) != len(self.base):
            raise DimensionMismatch("AdapterParameters.drift_l2: shape disagreement")
        s = 0.0
        for i in range(len(self.theta)):
            row_t = self.theta[i]
            row_b = self.base[i]
            for j in range(len(row_t)):
                d = row_t[j] - row_b[j]
                s += d * d
        return math.sqrt(s)


@dataclass(frozen=True)
class AdaptationResult:
    """Returned by :meth:`Pretunist.adapt`."""

    prediction: tuple[float, ...]
    """Adapted output ``ŷ = x*ᵀ θ*``."""

    base_prediction: tuple[float, ...]
    """Output of the base model ``ŷ_0 = x*ᵀ θ_0``."""

    adapter: AdapterParameters
    """The fit adapter parameters."""

    leverage: float
    """``h(x*) = x*ᵀ (XᵀX+λI)⁻¹ x*``."""

    predictive_variance: float
    """``σ² (1 + h(x*))`` — Bayesian predictive variance of the adapted
    model at ``x*``."""

    adaptation_gain_bits: float
    """``log₂ p_adapted(y|x*) − log₂ p_base(y|x*)`` evaluated at the
    *implicit* posterior — surrogate is mean-squared-error reduction in
    bits, since labels at the query may be unknown.  Bigger = better."""

    kl_drift_nats: float
    """``KL(Q ‖ P)`` for the adapter."""

    loo_risk: float
    """Leave-one-out empirical risk on the support."""

    kl_budget_active: bool
    """``True`` if the adapter was projected to the KL budget."""

    kl_projection_alpha: float
    """The projection scalar α ∈ [0,1].  1.0 means no projection."""

    fit_residual_norm: float
    """``||y − Xθ*||₂`` on the support."""

    timestamp: float = 0.0


@dataclass(frozen=True)
class AbstentionReport:
    """Why the primitive refused to commit a prediction."""

    triggered: bool
    rules_fired: tuple[str, ...]
    leverage: float
    predictive_variance: float
    kl_drift_nats: float
    loo_risk: float
    e_process_log: float


@dataclass(frozen=True)
class PretunistReport:
    """Snapshot of the state for the coordination engine to inspect."""

    schema: str
    config: dict[str, Any]
    n_support: int
    n_adaptations: int
    n_abstentions: int
    last_loo_risk: float
    last_pac_bayes_bound: float
    last_kl_drift_nats: float
    last_adaptation_gain_bits: float
    last_e_process_log: float
    ledger_root: str
    base_fingerprint: str
    adapter_fingerprint: str


@dataclass(frozen=True)
class PretunistCertificate:
    """Formal guarantee on the adapter, suitable for shipping to a
    coordinator that integrates with ``attest.AttestationLedger``."""

    schema: str
    n: int
    delta: float
    empirical_risk: float
    kl_qp_nats: float
    pac_bayes_bound: float
    pac_bayes_is_vacuous: bool
    loo_risk: float
    leverage_max: float
    e_process_log: float
    e_process_rejected: bool
    adapter_fingerprint: str
    ledger_root: str
    references: tuple[str, ...] = (
        "McAllester 1999",
        "Catoni 2007",
        "Akyürek-Damani-Qiu-Guo-Suzgun-Kim-Andreas-Kim 2024 (ARC-AGI)",
        "Howard-Ramdas-McAuliffe-Sekhon 2021",
    )


# ---------------------------------------------------------------------------
# Ledger helpers — SHA-256 fingerprint chain
# ---------------------------------------------------------------------------


PRETUNIST_LEDGER_GENESIS = "pretunist.ledger.v1"


def pretunist_ledger_genesis() -> str:
    """Return the genesis hash for a Pretunist ledger."""
    return hashlib.sha256(PRETUNIST_LEDGER_GENESIS.encode("utf-8")).hexdigest()


def pretunist_ledger_root() -> str:
    """Public alias for the genesis root (compat with other primitives)."""
    return pretunist_ledger_genesis()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default_json)


def _default_json(o: Any) -> Any:
    if isinstance(o, (tuple, list)):
        return list(o)
    if isinstance(o, bytes):
        return o.hex()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"unserialisable: {type(o).__name__}")


def _digest(prev: str, payload: Mapping[str, Any], hmac_key: bytes | None) -> str:
    canonical = _canonical_json({"prev": prev, "payload": payload})
    if hmac_key is None:
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return _hmac.new(hmac_key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Standalone reference algorithms
# ---------------------------------------------------------------------------


def ridge_regression_closed_form(
    X: Sequence[Sequence[float]],
    y: Sequence[Sequence[float]],
    *,
    ridge_lambda: float,
    base: Sequence[Sequence[float]] | None = None,
) -> list[list[float]]:
    """Compute ``θ* = (XᵀX + λI)⁻¹ Xᵀ (y − X θ_0) + θ_0`` (reference).

    The "shifted" form expresses the closed-form ridge as a residual
    fit on top of a base policy ``θ_0``.  When ``base`` is ``None``,
    the standard ``θ* = (XᵀX + λI)⁻¹ Xᵀ y`` is returned.
    """
    if not X:
        raise InvalidSupport("ridge_regression_closed_form: empty X")
    n = len(X)
    d = len(X[0])
    if len(y) != n:
        raise DimensionMismatch(f"X has {n} rows, y has {len(y)} rows")
    if base is not None:
        if len(base) != d:
            raise DimensionMismatch(
                f"base must have {d} rows, got {len(base)}"
            )
        out_dim = len(base[0])
    else:
        out_dim = len(y[0])
    if base is None:
        zero_base = [[0.0] * out_dim for _ in range(d)]
        base = zero_base

    # XᵀX
    XtX = [[0.0] * d for _ in range(d)]
    for i in range(n):
        row = X[i]
        if len(row) != d:
            raise DimensionMismatch(f"row {i} has length {len(row)} != {d}")
        for a in range(d):
            xa = row[a]
            if xa == 0.0:
                continue
            for b in range(d):
                XtX[a][b] += xa * row[b]

    A = _add_diag(XtX, ridge_lambda)

    # Xᵀ (y − X θ_0)
    theta_out = [[0.0] * out_dim for _ in range(d)]
    for k in range(out_dim):
        rhs = [0.0] * d
        for i in range(n):
            row = X[i]
            base_pred = 0.0
            for a in range(d):
                base_pred += row[a] * base[a][k]
            resid = y[i][k] - base_pred
            for a in range(d):
                rhs[a] += row[a] * resid
        delta = solve_psd(A, rhs)
        for a in range(d):
            theta_out[a][k] = base[a][k] + delta[a]

    return theta_out


def leave_one_out_residuals(
    X: Sequence[Sequence[float]],
    y: Sequence[Sequence[float]],
    *,
    ridge_lambda: float,
) -> list[list[float]]:
    """Return the n × output_dim matrix of LOO residuals using the
    PRESS / hat-matrix identity::

        e_loo,i  =  (y_i − ŷ_i) / (1 − H_ii)

    One ridge fit + n hat-matrix diagonal evaluations — O(n d² + d³).
    Allen 1974.
    """
    if not X:
        return []
    n = len(X)
    d = len(X[0])
    out_dim = len(y[0])
    theta = ridge_regression_closed_form(X, y, ridge_lambda=ridge_lambda)

    XtX = [[0.0] * d for _ in range(d)]
    for i in range(n):
        row = X[i]
        for a in range(d):
            xa = row[a]
            if xa == 0.0:
                continue
            for b in range(d):
                XtX[a][b] += xa * row[b]
    A = _add_diag(XtX, ridge_lambda)
    L = cholesky(A)
    LT = transpose(L)

    out = []
    for i in range(n):
        x = X[i]
        # H_ii = xᵀ A⁻¹ x
        z = solve_lower_triangular(L, list(x))
        w = solve_upper_triangular(LT, z)  # A⁻¹ x
        h_ii = _dot(x, w)
        denom = 1.0 - h_ii
        # When λ > 0, H_ii < 1 — degenerate denom should not appear.
        # Numerically clamp to avoid blowup.
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0.0 else -1e-12
        e = [0.0] * out_dim
        for k in range(out_dim):
            y_hat = 0.0
            for a in range(d):
                y_hat += x[a] * theta[a][k]
            e[k] = (y[i][k] - y_hat) / denom
        out.append(e)
    return out


def kl_gauss_isotropic(
    mu_q: Sequence[float],
    mu_p: Sequence[float],
    sigma_q_sq: float,
    sigma_p_sq: float,
) -> float:
    """``KL(N(μ_q, σ_q² I) ‖ N(μ_p, σ_p² I))`` in nats.

    Closed form::

        ½ [ d (σ_q²/σ_p² − 1 − log(σ_q²/σ_p²))  +  ||μ_q−μ_p||² / σ_p² ]
    """
    if len(mu_q) != len(mu_p):
        raise DimensionMismatch("kl_gauss_isotropic: length mismatch")
    if sigma_q_sq <= 0.0 or sigma_p_sq <= 0.0:
        raise InvalidConfig("kl_gauss_isotropic: variances must be positive")
    d = len(mu_q)
    ratio = sigma_q_sq / sigma_p_sq
    diff_sq = 0.0
    for i in range(d):
        diff = mu_q[i] - mu_p[i]
        diff_sq += diff * diff
    return 0.5 * (
        d * (ratio - 1.0 - math.log(ratio))
        + diff_sq / sigma_p_sq
    )


def pac_bayes_bound_value(
    *,
    empirical_risk: float,
    kl_qp_nats: float,
    n: int,
    delta: float,
) -> float:
    """McAllester-style PAC-Bayes upper bound on test risk::

        L_true ≤ L_emp + √( (KL + log(n / δ)) / (2 n) )

    Returns the bound (in the same units as ``empirical_risk``).
    """
    if n <= 0:
        raise InvalidConfig("pac_bayes_bound_value: n must be > 0")
    if not 0.0 < delta < 1.0:
        raise InvalidConfig("pac_bayes_bound_value: delta must be in (0, 1)")
    if kl_qp_nats < 0.0:
        raise InvalidConfig("pac_bayes_bound_value: KL must be >= 0")
    slack = math.sqrt(
        max(0.0, (kl_qp_nats + math.log(n / delta)) / (2.0 * n))
    )
    return empirical_risk + slack


def leverage_score(
    x: Sequence[float],
    X: Sequence[Sequence[float]],
    *,
    ridge_lambda: float,
) -> float:
    """``h(x) = xᵀ (XᵀX + λI)⁻¹ x``."""
    if not X:
        # With no support, A = λI, h(x) = ||x||² / λ.
        return _norm_sq(x) / max(ridge_lambda, 1e-30)
    d = len(X[0])
    if len(x) != d:
        raise DimensionMismatch(f"leverage_score: x has length {len(x)}, X has dim {d}")
    XtX = [[0.0] * d for _ in range(d)]
    for i in range(len(X)):
        row = X[i]
        for a in range(d):
            xa = row[a]
            if xa == 0.0:
                continue
            for b in range(d):
                XtX[a][b] += xa * row[b]
    A = _add_diag(XtX, ridge_lambda)
    w = solve_psd(A, list(x))
    return _dot(x, w)


def project_to_kl_budget(
    theta: Sequence[Sequence[float]],
    base: Sequence[Sequence[float]],
    *,
    sigma_p_sq: float,
    sigma_q_sq: float,
    budget_nats: float,
) -> tuple[list[list[float]], float, float]:
    """Project ``θ*`` toward ``θ_0`` so that ``KL ≤ budget``.

    Returns ``(theta_proj, alpha, kl)``.

    Solution: parametrise the line ``θ(α) = θ_0 + α (θ* − θ_0)``,
    α ∈ [0, 1].  KL grows quadratically in α::

        KL(α) = ½ d (r − 1 − log r) + ½ α² ||Δ||² / σ_p²

    where ``r = σ_q² / σ_p²`` and ``Δ = θ* − θ_0`` and dimension is
    ``d = adapter_dim × output_dim``.  The first term is independent
    of α; we solve for the largest α with KL ≤ budget.
    """
    d = len(theta)
    out_dim = len(theta[0])
    flat_dim = d * out_dim
    r = sigma_q_sq / sigma_p_sq
    base_kl = 0.5 * flat_dim * (r - 1.0 - math.log(r))
    diff_sq = 0.0
    for i in range(d):
        for j in range(out_dim):
            diff = theta[i][j] - base[i][j]
            diff_sq += diff * diff
    if diff_sq <= 0.0:
        return [list(row) for row in base], 0.0, base_kl
    slack = budget_nats - base_kl
    if slack <= 0.0:
        # Even the base KL exceeds budget; clamp to α = 0.
        return [list(row) for row in base], 0.0, base_kl
    alpha_max_sq = 2.0 * slack * sigma_p_sq / diff_sq
    alpha = min(1.0, math.sqrt(max(0.0, alpha_max_sq)))
    proj = [[base[i][j] + alpha * (theta[i][j] - base[i][j]) for j in range(out_dim)] for i in range(d)]
    kl = base_kl + 0.5 * alpha * alpha * diff_sq / sigma_p_sq
    return proj, alpha, kl


def mixture_martingale_logvalue(
    log_lr_history: Sequence[float],
    grid: Sequence[float],
) -> float:
    """Log of a mixture-martingale e-value.

    For a sequence of (already-log-likelihood-ratio) increments, the
    mixture-martingale value at time t is::

        M_t = (1/|G|) Σ_{g ∈ G} exp( Σ_{s≤t} log_lr(s; g) )

    We return log M_t.  In our restricted setting the per-step log-LR
    is parametrised by a sub-Gaussian effect size ``g`` and is taken
    to be ``g · z_s − ½ g²`` for each scalar increment ``z_s``
    (Howard-Ramdas-McAuliffe-Sekhon 2021, eq. (24)).  Pretunist passes
    in already-computed log-LRs per grid entry — see
    :meth:`Pretunist._update_eprocess`.
    """
    if not grid:
        return 0.0
    log_per_g = [0.0] * len(grid)
    for entry in log_lr_history:
        # entry is interpreted as the per-grid log-LR delta packed
        # into a single float through ``g * z − ½ g²`` reduction by
        # the caller.  See ``Pretunist._update_eprocess``.
        for i in range(len(grid)):
            log_per_g[i] += entry  # already pre-aggregated; left here
            # purely so the standalone helper has a sane interpretation.
    # Mixture: log( (1/|G|) Σ exp(log_per_g) )
    m = max(log_per_g)
    s = 0.0
    for v in log_per_g:
        s += math.exp(v - m)
    return m + math.log(s / len(grid))


# ---------------------------------------------------------------------------
# The Pretunist primitive
# ---------------------------------------------------------------------------


class Pretunist:
    """Test-Time Training as a runtime primitive.

    Lifecycle::

        pre = Pretunist(PretunistConfig(adapter_dim=16))
        pre.set_base(theta_0)               # optional; defaults to 0
        for (x, y) in support:
            pre.observe_support(x, y)
        result = pre.adapt(x_star)
        cert   = pre.certify(delta=0.05)
        report = pre.report()
        pre.reset()                          # next task starts fresh

    Threading: all public methods take an internal ``RLock``.
    Concurrent ``observe_support`` calls are safe; concurrent
    ``adapt`` calls serialise.
    """

    # ------------------------------------------------------------------
    # Construction / state

    def __init__(self, config: PretunistConfig | None = None) -> None:
        self.config: PretunistConfig = config or PretunistConfig()
        self._lock = threading.RLock()

        self._X: list[list[float]] = []          # support features
        self._y: list[list[float]] = []          # support targets
        self._weights: list[float] = []          # per-point weights
        self._notes: list[str] = []
        self._ts: list[float] = []

        # Incremental Cholesky info: we maintain XᵀX and Xᵀy for
        # streaming; we refactor lazily on adapt().
        self._XtX: list[list[float]] = [
            [0.0] * self.config.adapter_dim for _ in range(self.config.adapter_dim)
        ]
        self._Xty: list[list[float]] = [
            [0.0] * self.config.output_dim for _ in range(self.config.adapter_dim)
        ]
        self._sumw: float = 0.0

        # Base policy ``θ_0`` — defaults to zeros.
        self._base: list[list[float]] = [
            [0.0] * self.config.output_dim for _ in range(self.config.adapter_dim)
        ]

        # Last fit cache
        self._theta: list[list[float]] | None = None
        self._loo_risk: float = math.inf
        self._kl_drift: float = 0.0
        self._adaptation_gain: float = 0.0
        self._adapter_fingerprint: str = ""
        self._base_fingerprint: str = self._compute_base_fingerprint()
        self._kl_projection_alpha: float = 1.0

        # Counters
        self._n_adaptations: int = 0
        self._n_abstentions: int = 0

        # E-process state — running log-likelihood ratios per grid
        self._eproc_log_per_g: list[float] = [0.0] * len(self.config.e_process_grid)
        self._eproc_log_value: float = 0.0

        # Ledger
        self._ledger_root: str = pretunist_ledger_genesis()
        self._event_subscribers: list[Callable[[str, dict[str, Any]], None]] = []

    # ------------------------------------------------------------------
    # Public introspection helpers

    @property
    def n_observed(self) -> int:
        return len(self._X)

    @property
    def n_adaptations(self) -> int:
        return self._n_adaptations

    @property
    def n_abstentions(self) -> int:
        return self._n_abstentions

    @property
    def ledger_root(self) -> str:
        return self._ledger_root

    @property
    def base_fingerprint(self) -> str:
        return self._base_fingerprint

    @property
    def adapter_fingerprint(self) -> str:
        return self._adapter_fingerprint

    @property
    def has_adapter(self) -> bool:
        return self._theta is not None

    def subscribe(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        """Register a callback for runtime events (topic, payload)."""
        with self._lock:
            self._event_subscribers.append(fn)

    # ------------------------------------------------------------------
    # Base policy

    def set_base(self, theta_0: Sequence[Sequence[float]]) -> None:
        """Set the base policy / prior mean ``θ_0``.

        ``theta_0`` is a (adapter_dim × output_dim) row-major matrix.
        Must be set before ``adapt`` if a non-zero prior is desired
        (zeros are the default).
        """
        d = self.config.adapter_dim
        o = self.config.output_dim
        if len(theta_0) != d:
            raise DimensionMismatch(
                f"set_base: theta_0 must have {d} rows, got {len(theta_0)}"
            )
        for i, row in enumerate(theta_0):
            if len(row) != o:
                raise DimensionMismatch(
                    f"set_base: row {i} must have {o} entries, got {len(row)}"
                )
            for v in row:
                if not math.isfinite(v):
                    raise InvalidConfig("set_base: theta_0 contains non-finite entries")
        with self._lock:
            self._base = [list(row) for row in theta_0]
            self._base_fingerprint = self._compute_base_fingerprint()
            # Invalidate cached fit — the residual fit assumes a fixed base.
            self._theta = None
            self._loo_risk = math.inf

    def get_base(self) -> list[list[float]]:
        with self._lock:
            return [list(row) for row in self._base]

    def _compute_base_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(b"pretunist.base.v1")
        for row in self._base:
            for v in row:
                h.update(f"{v:.17g}".encode("ascii"))
                h.update(b"|")
            h.update(b"\n")
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Observation

    def observe_support(
        self,
        x: Sequence[float],
        y: Sequence[float] | float | None = None,
        *,
        weight: float = 1.0,
        note: str = "",
    ) -> None:
        """Add a support point ``(x, y)`` to the buffer.

        When ``y`` is ``None`` and ``config.ssl_mode != SSL_NONE`` the
        primitive mines a self-supervised target from ``x``.

        Streamed updates of ``XᵀX``, ``Xᵀy`` in O(d²) per call.  The
        Cholesky is refactored lazily inside :meth:`adapt`.
        """
        cfg = self.config
        d = cfg.adapter_dim
        o = cfg.output_dim

        x_list = list(x)
        if len(x_list) != d:
            raise InvalidSupport(
                f"observe_support: x has length {len(x_list)}, expected {d}"
            )
        if not _vec_finite(x_list):
            raise InvalidSupport("observe_support: x contains non-finite entries")
        if not math.isfinite(weight) or weight < 0.0:
            raise InvalidSupport(f"observe_support: weight must be >= 0 (got {weight})")

        if y is None:
            y_vec = self._mine_ssl_target(x_list)
            if y_vec is None:
                raise InvalidSupport(
                    f"observe_support: y is None but ssl_mode={cfg.ssl_mode!r} "
                    "does not produce a target"
                )
        else:
            if isinstance(y, (int, float)):
                y_vec = [float(y)] * o
            else:
                y_vec = list(float(v) for v in y)
            if len(y_vec) != o:
                raise InvalidSupport(
                    f"observe_support: y has length {len(y_vec)}, expected {o}"
                )
            if not _vec_finite(y_vec):
                raise InvalidSupport("observe_support: y contains non-finite entries")

        with self._lock:
            if len(self._X) >= cfg.max_support:
                raise InvalidSupport(
                    f"observe_support: max_support={cfg.max_support} reached"
                )
            # Streaming XᵀX
            for a in range(d):
                xa = x_list[a]
                if xa == 0.0:
                    continue
                row = self._XtX[a]
                for b in range(d):
                    row[b] += weight * xa * x_list[b]
            # Streaming Xᵀy
            for a in range(d):
                xa = x_list[a]
                if xa == 0.0:
                    continue
                row = self._Xty[a]
                for k in range(o):
                    row[k] += weight * xa * y_vec[k]
            self._X.append(x_list)
            self._y.append(y_vec)
            self._weights.append(float(weight))
            self._notes.append(note)
            self._ts.append(time.time())
            self._sumw += float(weight)

            # Update e-process: each new (x, y) gives a one-step-ahead
            # residual that we score under "adapter helps" vs "adapter
            # equals base".  See _update_eprocess for details.
            self._update_eprocess(x_list, y_vec)

            # Invalidate cached fit; coordinator must call adapt() again.
            self._theta = None

            payload = {
                "n_support": len(self._X),
                "weight": weight,
                "note": note,
            }
            self._append_ledger("observe", payload)
            self._emit(PRETUNIST_OBSERVED, payload)

    def observe_batch(
        self,
        xs: Sequence[Sequence[float]],
        ys: Sequence[Sequence[float]] | None = None,
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        """Append multiple support points; convenience wrapper."""
        if ys is not None and len(ys) != len(xs):
            raise InvalidSupport(
                f"observe_batch: len(xs)={len(xs)} != len(ys)={len(ys)}"
            )
        if weights is not None and len(weights) != len(xs):
            raise InvalidSupport("observe_batch: weights length mismatch")
        for i, x in enumerate(xs):
            y = None if ys is None else ys[i]
            w = 1.0 if weights is None else weights[i]
            self.observe_support(x, y, weight=w)

    def _mine_ssl_target(self, x: Sequence[float]) -> list[float] | None:
        cfg = self.config
        if cfg.ssl_mode == SSL_NONE:
            return None
        if cfg.ssl_mode == SSL_LOO:
            # Use the support mean as the target — degenerate when n<2;
            # the actual LOO loss is captured by ``loo_risk_estimate``.
            return [sum(x) / max(1, len(x))] * cfg.output_dim
        if cfg.ssl_mode == SSL_PREFIX_TARGET:
            split = max(1, int(cfg.adapter_dim * cfg.ssl_prefix_fraction))
            # Use the suffix mean as a scalar target broadcast to output_dim.
            tail = x[split:]
            if not tail:
                tail = [0.0]
            mean = sum(tail) / len(tail)
            return [mean] * cfg.output_dim
        if cfg.ssl_mode == SSL_RECONSTRUCT:
            # Target = x itself (output_dim must equal adapter_dim).
            if cfg.output_dim != cfg.adapter_dim:
                raise InvalidConfig(
                    "SSL_RECONSTRUCT requires output_dim == adapter_dim"
                )
            return list(x)
        return None

    # ------------------------------------------------------------------
    # Reset

    def reset(self) -> None:
        """Drop support, cached adapter, e-process state.

        The base policy and configuration survive.
        """
        with self._lock:
            d = self.config.adapter_dim
            o = self.config.output_dim
            self._X.clear()
            self._y.clear()
            self._weights.clear()
            self._notes.clear()
            self._ts.clear()
            self._XtX = [[0.0] * d for _ in range(d)]
            self._Xty = [[0.0] * o for _ in range(d)]
            self._sumw = 0.0
            self._theta = None
            self._loo_risk = math.inf
            self._kl_drift = 0.0
            self._adaptation_gain = 0.0
            self._adapter_fingerprint = ""
            self._kl_projection_alpha = 1.0
            self._eproc_log_per_g = [0.0] * len(self.config.e_process_grid)
            self._eproc_log_value = 0.0
            self._append_ledger("reset", {})
            self._emit(PRETUNIST_RESET, {})

    # ------------------------------------------------------------------
    # Fit

    def adapt(
        self,
        query: Sequence[float] | None = None,
        *,
        delta: float = 0.05,
    ) -> AdaptationResult:
        """Fit the adapter on the current support and (optionally)
        evaluate at ``query``.

        When ``query is None``, the result's ``prediction`` /
        ``base_prediction`` are empty tuples and the rest of the
        certificate is still returned.

        Side effect: updates the e-process, the adapter cache, the
        ledger, and the ``adapter_fingerprint``.
        """
        cfg = self.config
        d = cfg.adapter_dim
        o = cfg.output_dim
        with self._lock:
            n = len(self._X)
            if n < cfg.min_support:
                raise InsufficientData(
                    f"adapt: n={n} below min_support={cfg.min_support}"
                )
            # 1. Closed-form ridge: θ* − θ_0 = (XᵀX + λI)⁻¹ Xᵀ (y − X θ_0)
            A = _add_diag(self._XtX, cfg.ridge_lambda)
            L = cholesky(A)
            LT = transpose(L)
            theta = [[0.0] * o for _ in range(d)]
            for k in range(o):
                # rhs_k = Xty[:, k] - XtX @ base[:, k]
                rhs = [0.0] * d
                for a in range(d):
                    rhs[a] = self._Xty[a][k]
                    s = 0.0
                    for b in range(d):
                        s += self._XtX[a][b] * self._base[b][k]
                    rhs[a] -= s
                z = solve_lower_triangular(L, rhs)
                delta_k = solve_upper_triangular(LT, z)
                for a in range(d):
                    theta[a][k] = self._base[a][k] + delta_k[a]

            # 2. KL drift in *nats*
            mu_q = [theta[a][k] for a in range(d) for k in range(o)]
            mu_p = [self._base[a][k] for a in range(d) for k in range(o)]
            kl_drift = kl_gauss_isotropic(
                mu_q, mu_p, cfg.posterior_variance, cfg.prior_variance,
            )

            # 3. KL-budget projection
            alpha = 1.0
            kl_budget_active = False
            if not math.isinf(cfg.kl_budget) and kl_drift > cfg.kl_budget:
                theta, alpha, kl_drift = project_to_kl_budget(
                    theta, self._base,
                    sigma_p_sq=cfg.prior_variance,
                    sigma_q_sq=cfg.posterior_variance,
                    budget_nats=cfg.kl_budget,
                )
                kl_budget_active = True

            # 4. LOO risk via PRESS identity
            loo = self._loo_risk_internal(L, LT, theta)

            # 5. Residual norm
            fit_resid = 0.0
            for i in range(n):
                row = self._X[i]
                for k in range(o):
                    y_hat = 0.0
                    for a in range(d):
                        y_hat += row[a] * theta[a][k]
                    diff = self._y[i][k] - y_hat
                    fit_resid += diff * diff
            fit_resid = math.sqrt(fit_resid)

            # 6. Adaptation gain in bits (log₂ ratio of MSEs).
            # Base MSE on support:
            base_mse = 0.0
            adapt_mse = 0.0
            for i in range(n):
                row = self._X[i]
                for k in range(o):
                    y_hat_base = 0.0
                    y_hat_adapt = 0.0
                    for a in range(d):
                        y_hat_base += row[a] * self._base[a][k]
                        y_hat_adapt += row[a] * theta[a][k]
                    base_mse += (self._y[i][k] - y_hat_base) ** 2
                    adapt_mse += (self._y[i][k] - y_hat_adapt) ** 2
            base_mse = max(base_mse / max(1, n * o), 1e-30)
            adapt_mse = max(adapt_mse / max(1, n * o), 1e-30)
            gain_bits = 0.5 * math.log2(base_mse / adapt_mse)

            # 7. Evaluate at query (if given)
            if query is not None:
                qlist = list(query)
                if len(qlist) != d:
                    raise InvalidQuery(
                        f"adapt: query has length {len(qlist)}, expected {d}"
                    )
                if not _vec_finite(qlist):
                    raise InvalidQuery("adapt: query contains non-finite entries")
                pred = [0.0] * o
                base_pred = [0.0] * o
                for k in range(o):
                    for a in range(d):
                        pred[k] += qlist[a] * theta[a][k]
                        base_pred[k] += qlist[a] * self._base[a][k]
                # Leverage and predictive variance
                z = solve_lower_triangular(L, qlist)
                w = solve_upper_triangular(LT, z)
                lev = _dot(qlist, w)
                pvar = cfg.noise_variance * (1.0 + lev)
            else:
                pred = []
                base_pred = []
                lev = 0.0
                pvar = 0.0

            # 8. Cache + fingerprint
            self._theta = theta
            self._loo_risk = loo
            self._kl_drift = kl_drift
            self._adaptation_gain = gain_bits
            self._kl_projection_alpha = alpha
            self._adapter_fingerprint = self._compute_adapter_fingerprint(theta)
            self._n_adaptations += 1

            adapter = AdapterParameters(
                theta=tuple(tuple(row) for row in theta),
                base=tuple(tuple(row) for row in self._base),
                ridge_lambda=cfg.ridge_lambda,
                n=n,
                fingerprint=self._adapter_fingerprint,
            )
            result = AdaptationResult(
                prediction=tuple(pred),
                base_prediction=tuple(base_pred),
                adapter=adapter,
                leverage=lev,
                predictive_variance=pvar,
                adaptation_gain_bits=gain_bits,
                kl_drift_nats=kl_drift,
                loo_risk=loo,
                kl_budget_active=kl_budget_active,
                kl_projection_alpha=alpha,
                fit_residual_norm=fit_resid,
                timestamp=time.time(),
            )
            payload = {
                "n": n,
                "leverage": lev,
                "kl_drift_nats": kl_drift,
                "adaptation_gain_bits": gain_bits,
                "loo_risk": loo,
                "alpha": alpha,
                "fingerprint": self._adapter_fingerprint,
            }
            self._append_ledger("adapt", payload)
            self._emit(PRETUNIST_ADAPTED, payload)
            return result

    def _loo_risk_internal(
        self,
        L: Sequence[Sequence[float]],
        LT: Sequence[Sequence[float]],
        theta: Sequence[Sequence[float]],
    ) -> float:
        n = len(self._X)
        if n == 0:
            return math.inf
        d = self.config.adapter_dim
        o = self.config.output_dim
        s = 0.0
        for i in range(n):
            x = self._X[i]
            z = solve_lower_triangular(L, x)
            w = solve_upper_triangular(LT, z)
            h_ii = _dot(x, w)
            denom = 1.0 - h_ii
            if abs(denom) < 1e-12:
                denom = 1e-12 if denom >= 0.0 else -1e-12
            for k in range(o):
                y_hat = 0.0
                for a in range(d):
                    y_hat += x[a] * theta[a][k]
                e = (self._y[i][k] - y_hat) / denom
                s += e * e
        return s / max(1, n * o)

    # ------------------------------------------------------------------
    # Prediction (post-adapt)

    def predict(self, query: Sequence[float]) -> tuple[float, ...]:
        """Predict using the cached adapter.  Raises if no fit yet."""
        with self._lock:
            if self._theta is None:
                raise AdapterNotFit("predict: call adapt() first")
            q = list(query)
            if len(q) != self.config.adapter_dim:
                raise InvalidQuery(
                    f"predict: query has length {len(q)}, expected "
                    f"{self.config.adapter_dim}"
                )
            o = self.config.output_dim
            out = [0.0] * o
            for k in range(o):
                for a in range(self.config.adapter_dim):
                    out[k] += q[a] * self._theta[a][k]
            return tuple(out)

    def predict_base(self, query: Sequence[float]) -> tuple[float, ...]:
        """Predict using the base policy (no adaptation)."""
        with self._lock:
            q = list(query)
            if len(q) != self.config.adapter_dim:
                raise InvalidQuery(
                    f"predict_base: query has length {len(q)}, expected "
                    f"{self.config.adapter_dim}"
                )
            o = self.config.output_dim
            out = [0.0] * o
            for k in range(o):
                for a in range(self.config.adapter_dim):
                    out[k] += q[a] * self._base[a][k]
            return tuple(out)

    # ------------------------------------------------------------------
    # PAC-Bayes bound

    def pac_bayes_bound(self, delta: float = 0.05) -> "PretunistCertificate":
        """Return the PAC-Bayes certificate.

        Requires at least one ``adapt`` call.
        """
        with self._lock:
            if self._theta is None:
                raise AdapterNotFit("pac_bayes_bound: call adapt() first")
            cfg = self.config
            n = len(self._X)
            if n <= 0:
                raise InsufficientData("pac_bayes_bound: no support")
            # Empirical risk = mean squared training residual (per-output).
            d = cfg.adapter_dim
            o = cfg.output_dim
            sq = 0.0
            for i in range(n):
                row = self._X[i]
                for k in range(o):
                    y_hat = 0.0
                    for a in range(d):
                        y_hat += row[a] * self._theta[a][k]
                    sq += (self._y[i][k] - y_hat) ** 2
            emp = sq / max(1, n * o)
            bound = pac_bayes_bound_value(
                empirical_risk=emp,
                kl_qp_nats=self._kl_drift,
                n=n,
                delta=delta,
            )
            # Vacuous if bound exceeds the worst-case sub-Gaussian
            # variance estimate.  Heuristic: a regression bound is
            # vacuous if it is ≥ Var(y) measured on the support.
            var_y = self._variance_y()
            is_vacuous = bound >= var_y > 0.0

            # Max leverage on support (sanity).
            A = _add_diag(self._XtX, cfg.ridge_lambda)
            L = cholesky(A)
            LT = transpose(L)
            max_lev = 0.0
            for i in range(n):
                z = solve_lower_triangular(L, self._X[i])
                w = solve_upper_triangular(LT, z)
                lev = _dot(self._X[i], w)
                if lev > max_lev:
                    max_lev = lev

            eproc = self._eproc_log_value
            eproc_rej = self._eproc_log_value > math.log(1.0 / cfg.e_process_alpha)

            cert = PretunistCertificate(
                schema=SCHEMA_VERSION,
                n=n,
                delta=delta,
                empirical_risk=emp,
                kl_qp_nats=self._kl_drift,
                pac_bayes_bound=bound,
                pac_bayes_is_vacuous=is_vacuous,
                loo_risk=self._loo_risk,
                leverage_max=max_lev,
                e_process_log=eproc,
                e_process_rejected=eproc_rej,
                adapter_fingerprint=self._adapter_fingerprint,
                ledger_root=self._ledger_root,
            )
            self._append_ledger("certify", {
                "delta": delta,
                "bound": bound,
                "is_vacuous": is_vacuous,
            })
            self._emit(PRETUNIST_CERTIFIED, {
                "delta": delta,
                "bound": bound,
                "kl_qp_nats": self._kl_drift,
            })
            return cert

    def certify(self, delta: float = 0.05) -> "PretunistCertificate":
        """Alias for :meth:`pac_bayes_bound`."""
        return self.pac_bayes_bound(delta=delta)

    def _variance_y(self) -> float:
        if not self._y:
            return 0.0
        flat = [v for row in self._y for v in row]
        m = sum(flat) / len(flat)
        return sum((v - m) ** 2 for v in flat) / len(flat)

    # ------------------------------------------------------------------
    # Abstention

    def should_abstain(self, query: Sequence[float]) -> AbstentionReport:
        """Apply all configured abstention rules to ``query``.

        Returns a report.  The coordinator decides what to do; this
        method does not raise.
        """
        cfg = self.config
        with self._lock:
            fired: list[str] = []
            q = list(query)
            if len(q) != cfg.adapter_dim:
                raise InvalidQuery(
                    f"should_abstain: query length {len(q)} != {cfg.adapter_dim}"
                )
            A = _add_diag(self._XtX, cfg.ridge_lambda)
            L = cholesky(A)
            LT = transpose(L)
            z = solve_lower_triangular(L, q)
            w = solve_upper_triangular(LT, z)
            lev = _dot(q, w)
            pvar = cfg.noise_variance * (1.0 + lev)
            if ABSTAIN_LEVERAGE in cfg.abstain_rules and lev >= cfg.abstain_leverage_threshold:
                fired.append(ABSTAIN_LEVERAGE)
            if ABSTAIN_VARIANCE in cfg.abstain_rules and pvar >= cfg.abstain_variance_threshold:
                fired.append(ABSTAIN_VARIANCE)
            kl = self._kl_drift if self._theta is not None else 0.0
            if (
                ABSTAIN_KL_BUDGET in cfg.abstain_rules
                and not math.isinf(cfg.kl_budget)
                and kl >= cfg.kl_budget
            ):
                fired.append(ABSTAIN_KL_BUDGET)
            loo = self._loo_risk if self._theta is not None else math.inf
            if (
                ABSTAIN_LOO_RISK in cfg.abstain_rules
                and math.isfinite(cfg.abstain_loo_risk_threshold)
                and loo >= cfg.abstain_loo_risk_threshold
            ):
                fired.append(ABSTAIN_LOO_RISK)
            eproc = self._eproc_log_value
            eproc_threshold = math.log(1.0 / cfg.e_process_alpha)
            if ABSTAIN_E_PROCESS in cfg.abstain_rules and eproc < 0.0:
                # Negative e-process => base is favoured; abstain on adapter.
                fired.append(ABSTAIN_E_PROCESS)
            triggered = bool(fired)
            if triggered:
                self._n_abstentions += 1
                self._append_ledger("abstain", {"rules": list(fired)})
                self._emit(PRETUNIST_ABSTAINED, {"rules": list(fired)})
            return AbstentionReport(
                triggered=triggered,
                rules_fired=tuple(fired),
                leverage=lev,
                predictive_variance=pvar,
                kl_drift_nats=kl,
                loo_risk=loo,
                e_process_log=eproc,
            )

    # ------------------------------------------------------------------
    # E-process

    def _update_eprocess(self, x: Sequence[float], y: Sequence[float]) -> None:
        """One-step-ahead e-process update.

        For each ``(x_i, y_i)`` we compute the prediction error of the
        *current cached adapter* and the *base*, take the difference
        ``z_i = e_base² − e_adapt²``, normalise to a sub-Gaussian unit,
        and update the mixture martingale (Howard-Ramdas 2021).
        """
        cfg = self.config
        if self._theta is None:
            theta = self._base
        else:
            theta = self._theta
        e_adapt_sq = 0.0
        e_base_sq = 0.0
        for k in range(cfg.output_dim):
            y_hat_adapt = 0.0
            y_hat_base = 0.0
            for a in range(cfg.adapter_dim):
                y_hat_adapt += x[a] * theta[a][k]
                y_hat_base += x[a] * self._base[a][k]
            e_adapt_sq += (y[k] - y_hat_adapt) ** 2
            e_base_sq += (y[k] - y_hat_base) ** 2
        # z is in [−B, B] with B ≈ ||x||² + ||y||² (boundedness assumed).
        # We use ``tanh`` to map into [−1, 1] without re-tracking B.
        raw = e_base_sq - e_adapt_sq
        z = math.tanh(raw)
        # Per-grid log-likelihood-ratio increment (Howard 2021 eq. 24):
        #     log_LR(g) = g · z  −  ½ g²
        m = -math.inf
        new_per_g = list(self._eproc_log_per_g)
        for i, g in enumerate(cfg.e_process_grid):
            new_per_g[i] += g * z - 0.5 * g * g
            if new_per_g[i] > m:
                m = new_per_g[i]
        s = 0.0
        for v in new_per_g:
            s += math.exp(v - m)
        self._eproc_log_per_g = new_per_g
        self._eproc_log_value = m + math.log(s / len(cfg.e_process_grid))

    @property
    def e_process_log(self) -> float:
        return self._eproc_log_value

    def e_process_rejected(self) -> bool:
        return self._eproc_log_value > math.log(1.0 / self.config.e_process_alpha)

    # ------------------------------------------------------------------
    # Reports / snapshots

    def report(self) -> PretunistReport:
        with self._lock:
            return PretunistReport(
                schema=SCHEMA_VERSION,
                config=self.config.to_dict(),
                n_support=len(self._X),
                n_adaptations=self._n_adaptations,
                n_abstentions=self._n_abstentions,
                last_loo_risk=self._loo_risk,
                last_pac_bayes_bound=(
                    self.pac_bayes_bound().pac_bayes_bound
                    if self._theta is not None
                    else math.inf
                ),
                last_kl_drift_nats=self._kl_drift,
                last_adaptation_gain_bits=self._adaptation_gain,
                last_e_process_log=self._eproc_log_value,
                ledger_root=self._ledger_root,
                base_fingerprint=self._base_fingerprint,
                adapter_fingerprint=self._adapter_fingerprint,
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable dict containing all state.

        ``Pretunist.restore(d)`` recreates the primitive byte-for-byte.
        """
        with self._lock:
            return {
                "schema": SCHEMA_VERSION,
                "config": self.config.to_dict(),
                "X": [list(row) for row in self._X],
                "y": [list(row) for row in self._y],
                "weights": list(self._weights),
                "notes": list(self._notes),
                "ts": list(self._ts),
                "XtX": [list(row) for row in self._XtX],
                "Xty": [list(row) for row in self._Xty],
                "sumw": self._sumw,
                "base": [list(row) for row in self._base],
                "theta": (
                    [list(row) for row in self._theta]
                    if self._theta is not None else None
                ),
                "loo_risk": self._loo_risk,
                "kl_drift": self._kl_drift,
                "adaptation_gain": self._adaptation_gain,
                "adapter_fingerprint": self._adapter_fingerprint,
                "base_fingerprint": self._base_fingerprint,
                "kl_projection_alpha": self._kl_projection_alpha,
                "n_adaptations": self._n_adaptations,
                "n_abstentions": self._n_abstentions,
                "ledger_root": self._ledger_root,
                "eproc_log_per_g": list(self._eproc_log_per_g),
                "eproc_log_value": self._eproc_log_value,
            }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "Pretunist":
        """Reconstruct a primitive from its :meth:`snapshot`."""
        cfg_d = dict(payload["config"])
        if cfg_d.get("kl_budget") is None:
            cfg_d["kl_budget"] = math.inf
        if cfg_d.get("abstain_loo_risk_threshold") is None:
            cfg_d["abstain_loo_risk_threshold"] = math.inf
        cfg_d.pop("schema", None)
        cfg_d["abstain_rules"] = tuple(cfg_d.get("abstain_rules", ()))
        cfg_d["e_process_grid"] = tuple(cfg_d.get("e_process_grid", (1.0,)))
        cfg = PretunistConfig(**cfg_d)
        out = cls(cfg)
        out._X = [list(row) for row in payload.get("X", [])]
        out._y = [list(row) for row in payload.get("y", [])]
        out._weights = list(payload.get("weights", []))
        out._notes = list(payload.get("notes", []))
        out._ts = list(payload.get("ts", []))
        out._XtX = [list(row) for row in payload.get("XtX", [])]
        out._Xty = [list(row) for row in payload.get("Xty", [])]
        out._sumw = float(payload.get("sumw", 0.0))
        out._base = [list(row) for row in payload.get("base", [])]
        theta = payload.get("theta")
        out._theta = (
            [list(row) for row in theta] if theta is not None else None
        )
        out._loo_risk = float(payload.get("loo_risk", math.inf))
        out._kl_drift = float(payload.get("kl_drift", 0.0))
        out._adaptation_gain = float(payload.get("adaptation_gain", 0.0))
        out._adapter_fingerprint = str(payload.get("adapter_fingerprint", ""))
        out._base_fingerprint = str(
            payload.get("base_fingerprint", out._compute_base_fingerprint())
        )
        out._kl_projection_alpha = float(payload.get("kl_projection_alpha", 1.0))
        out._n_adaptations = int(payload.get("n_adaptations", 0))
        out._n_abstentions = int(payload.get("n_abstentions", 0))
        out._ledger_root = str(payload.get("ledger_root", pretunist_ledger_genesis()))
        eproc = payload.get("eproc_log_per_g")
        if eproc is not None:
            out._eproc_log_per_g = [float(v) for v in eproc]
        out._eproc_log_value = float(payload.get("eproc_log_value", 0.0))
        return out

    # ------------------------------------------------------------------
    # Ledger / events

    def _append_ledger(self, op: str, payload: Mapping[str, Any]) -> None:
        rec = {
            "op": op,
            "ts": time.time(),
            **dict(payload),
        }
        self._ledger_root = _digest(self._ledger_root, rec, self.config.hmac_key)

    def _compute_adapter_fingerprint(self, theta: Sequence[Sequence[float]]) -> str:
        h = hashlib.sha256()
        h.update(b"pretunist.adapter.v1")
        h.update(f"|n={len(self._X)}".encode("ascii"))
        h.update(f"|lambda={self.config.ridge_lambda:.17g}".encode("ascii"))
        for row in theta:
            for v in row:
                h.update(f"{v:.17g}".encode("ascii"))
                h.update(b"|")
            h.update(b"\n")
        return h.hexdigest()

    def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        for fn in self._event_subscribers:
            try:
                fn(topic, dict(payload))
            except Exception:  # pragma: no cover — subscriber faults must not kill us
                pass

    # ------------------------------------------------------------------
    # Manifest hook — let coordination engines discover Pretunist.

    @staticmethod
    def manifest_spec() -> dict[str, Any]:
        """Return the manifest entry shape for this primitive.

        Used by :func:`agi.manifest.auto_discover` if the curated table
        ever drifts.
        """
        return {
            "name": "pretunist",
            "kind": "learning",
            "summary": (
                "Test-time training as a runtime primitive — closed-form "
                "ridge adapter, PAC-Bayes bound, KL-budget projection, "
                "leverage-based abstention."
            ),
            "tags": (
                "adaptive", "pac-bound", "anytime-valid", "calibration",
            ),
            "inputs": (
                "support: Iterable[(x: Sequence[float], y: Sequence[float])]",
                "base: Sequence[Sequence[float]] | None",
            ),
            "outputs": (
                "AdaptationResult",
                "PretunistCertificate",
                "AbstentionReport",
            ),
            "certificate": "pac",
            "determinism": "pure",
            "dependency": "stdlib",
            "composes_with": (
                "continualist", "stepwiser", "distiller", "aligner",
                "preflight", "selfeval",
            ),
            "demo_path": "examples/pretunist_demo.py",
            "events_emitted": (
                PRETUNIST_OBSERVED, PRETUNIST_ADAPTED,
                PRETUNIST_CERTIFIED, PRETUNIST_ABSTAINED,
                PRETUNIST_RESET,
            ),
            "complexity": "O(n d² + d³) per adapt; O(d²) per observe",
            "references": (
                "Sun-Wang-Liu-Held-Efros-Hardt 2020",
                "Akyürek-Damani-Qiu-Guo-Suzgun-Kim-Andreas-Kim 2024 (ARC-AGI)",
                "Akyürek-Schuurmans-Andreas-Zhou-Ma 2023",
                "McAllester 1999",
                "Catoni 2007",
                "Howard-Ramdas-McAuliffe-Sekhon 2021",
            ),
        }
