r"""Scaler — scaling-law inference as a runtime primitive.

Every other primitive in the runtime spends compute.  ``Scaler`` is the
primitive that turns **how much compute** into **what capability**.  Given
a table of observed ``(model_size, data_tokens, training_loss)`` triples
from completed runs (the runtime's own runs, an internal eval bank, a
published table from a vendor, or all three), ``Scaler`` fits one or
more scaling-law families, produces bootstrap-percentile confidence
intervals for extrapolated loss at unseen ``(N, D)``, and returns the
**compute-optimal allocation** ``(N*, D*)`` for any future compute
budget ``C``.  Each report carries a PAC-style certificate (held-out
RMSE LCB) and a replay-verifiable fingerprint over the entire
ingest/fit/extrapolate trace.

This is the **resource primitive** every coordination engine needs the
moment it has to decide *"if I give Pretunist 10× more compute, will
the loss it ships move enough to justify it?"*.  A learned scaling law
is the only honest answer.  Without it the coordinator is gambling.

Mathematical and algorithmic roots
----------------------------------

* **Kaplan, J., McCandlish, S., Henighan, T., et al. (2020) — "Scaling
  Laws for Neural Language Models" (arXiv:2001.08361).**  The original
  multiplicative-power-law form

  .. math::

     L(N, D) = \\left[\\left(\\frac{N_c}{N}\\right)^{\\alpha_N /
       \\alpha_D} + \\frac{D_c}{D}\\right]^{\\alpha_D}

  with five free parameters ``(N_c, D_c, \\alpha_N, \\alpha_D, \\epsilon)``.
  Implemented as :data:`FAMILY_KAPLAN`.

* **Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022) — "Training
  Compute-Optimal Large Language Models" (arXiv:2203.15556).**  The
  Chinchilla parametric form

  .. math::

     L(N, D) = E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}}

  whose closed-form compute-optimal allocation under the constraint
  ``C = 6 N D`` is

  .. math::

     N^* = \\left[\\frac{A\\alpha}{B\\beta}\\right]^{1/(\\alpha+\\beta)}
           \\left(\\frac{C}{6}\\right)^{\\beta/(\\alpha+\\beta)},\\quad
     D^* = \\frac{C}{6 N^*}.

  Implemented as :data:`FAMILY_CHINCHILLA` — the default.  Closed-form
  optimal allocation lives in :meth:`Scaler.compute_optimal`.

* **Caballero, E., Gupta, K., Rish, I., Krueger, D. (2023) — "Broken
  Neural Scaling Laws" (arXiv:2210.14891).**  A smooth piecewise-power
  family

  .. math::

     L(X) = a + (b X^{-c_0}) \\prod_{i=1}^{n}\\left(1 +
       (X / d_i)^{1/f_i}\\right)^{-c_i f_i}

  that captures broken / sigmoidal / monotonic-with-bump scaling
  observed empirically.  Implemented at order ``n = 1`` as
  :data:`FAMILY_BNSL` — one break point, one power-law per side, with
  a smooth transition.  Three orders is over-parameterised for the
  data scales this primitive is built for; one is the documented
  sweet spot for routine extrapolation.

* **Bahri, Y., Dyer, E., Kaplan, J., Lee, J., Sharma, U. (2024) — "Explaining
  Neural Scaling Laws" (arXiv:2102.06701, revised PNAS 2024).**  The
  single-variable resolution-limited form

  .. math::

     L(X) = L_{\\infty} + (X_0 / X)^{\\alpha}

  with three parameters ``(L_\\infty, X_0, \\alpha)``.  Implemented as
  :data:`FAMILY_BAHRI` — used as a per-axis baseline when only one of
  ``(N, D)`` is varied.

* **Levenberg, K. (1944), Marquardt, D. (1963) — Damped least-squares.**
  All families are fit by Levenberg-Marquardt in **log-loss space**
  (so the residuals are scale-free), with finite-difference Jacobians
  and an adaptive damping schedule.  Parameter bounds are enforced by
  a smooth bijective reparameterisation (``softplus`` for non-negative
  parameters, ``logit`` for ``[0, 1]``-bounded ones) — Stan-style.

* **Efron, B. (1979) — "Bootstrap methods: another look at the
  jackknife."**  Bootstrap-percentile confidence intervals on the
  extrapolated loss, with optional **case resampling** and
  **wild-residual resampling** (Davidson-Flachaire 2008).

* **Maurer, A., Pontil, M. (2009) — "Empirical Bernstein bounds."**
  Sample-variance-aware lower confidence bound on held-out RMSE,
  reported as the PAC certificate.

* **Hoeffding, W. (1963).**  Variance-free fallback bound for
  comparison.

* **Vaswani, A. et al. (2017) Attention-Is-All-You-Need FLOPs
  accounting.**  Default ``C = 6 N D`` for transformer training
  (forward + backward + 1 grad pass over each parameter on each
  token).  Overridable via :class:`ScalerConfig.flops_per_param_token`.


Composes with
-------------

* :mod:`agi.economist` / :mod:`agi.market` — turn predicted loss
  reductions into dollar value of additional compute.
* :mod:`agi.stepwiser` / :mod:`agi.selfeval` — feed observed
  per-step / per-eval losses back into the fit.
* :mod:`agi.curator` — budget-allocate across curriculum stages with
  scaling-law-aware ROI.
* :mod:`agi.continualist` / :mod:`agi.pretunist` — sized adaptation
  budgets per task with closed-form ``(N*, D*)``.
* :mod:`agi.strategist` / :mod:`agi.portfolio` — risk-adjusted compute
  decisions across primitives.
* :mod:`agi.attest` / :mod:`agi.governance` — replay-verifiable
  fingerprint chain over every observation and every fit.
* :mod:`agi.conformal` / :mod:`agi.calibration` — held-out CI / PAC
  bounds on the predicted loss.

What this primitive ships
-------------------------

* :class:`Observation` — one ``(N, D, loss)`` row with optional weight
  and metadata.
* :class:`ScalerConfig` — family choice, bootstrap-B, seed, FLOPs/p/t
  constant, max LM iterations, regulariser, tolerance.
* :class:`FitResult` — fitted parameters, in-sample / held-out RMSE,
  per-parameter standard error from the LM Jacobian.
* :class:`ExtrapolatePoint` — single prediction with bootstrap-quantile
  CI.
* :class:`ComputeOptimal` — closed-form ``(N*, D*, L*)`` plus the
  numerical reflection of the analytic optimum (sanity check).
* :class:`ScalerCertificate` — held-out RMSE Hoeffding + empirical-
  Bernstein LCBs, dead-zone diagnostics, fingerprint chain.
* :class:`Scaler` — the primitive.  Observe → fit → extrapolate →
  compute-optimal → certify → report.

Pure stdlib.  No NumPy, no SciPy, no Torch.  Deterministic given seed.
Thread-safe.  ``json.dumps(report.to_dict())`` round-trips.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from agi.events import Event, EventBus

# ---------------------------------------------------------------------------
# Constants and taxonomy
# ---------------------------------------------------------------------------

FAMILY_CHINCHILLA = "chinchilla"
FAMILY_KAPLAN = "kaplan"
FAMILY_BNSL = "bnsl"
FAMILY_BAHRI_N = "bahri_n"
FAMILY_BAHRI_D = "bahri_d"

KNOWN_FAMILIES: tuple[str, ...] = (
    FAMILY_CHINCHILLA,
    FAMILY_KAPLAN,
    FAMILY_BNSL,
    FAMILY_BAHRI_N,
    FAMILY_BAHRI_D,
)

# Event names emitted on the runtime EventBus.
SCALER_STARTED = "scaler.started"
SCALER_OBSERVED = "scaler.observed"
SCALER_FIT = "scaler.fit"
SCALER_EXTRAPOLATED = "scaler.extrapolated"
SCALER_OPTIMAL = "scaler.compute_optimal"
SCALER_CERTIFIED = "scaler.certified"
SCALER_REPORTED = "scaler.reported"
SCALER_RESET = "scaler.reset"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ScalerError(ValueError):
    """Base class for Scaler-specific errors."""


class InvalidConfig(ScalerError):
    """The :class:`ScalerConfig` is internally inconsistent."""


class InvalidObservation(ScalerError):
    """The observation row violates a runtime invariant (NaN, ≤ 0, etc.)."""


class UnknownFamily(ScalerError):
    """A scaling-law family name was not recognised."""


class NotFitted(ScalerError):
    """Tried to extrapolate / certify / compute-optimal before fitting."""


class FitFailed(ScalerError):
    """Levenberg-Marquardt did not converge within tolerance / iters."""


# ---------------------------------------------------------------------------
# Configuration and records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalerConfig:
    """Configuration for a :class:`Scaler` instance.

    Attributes:
        family: one of :data:`KNOWN_FAMILIES`.  Default Chinchilla.
        seed: PRNG seed for bootstrap.
        bootstrap_b: number of bootstrap resamples (≥ 0; 0 disables
            CI but still returns the point estimate).
        max_iters: Levenberg-Marquardt iteration cap.
        tol: convergence tolerance on relative parameter change.
        flops_per_param_token: ``C = k · N · D`` constant; default
            ``6.0`` from Vaswani-style transformer accounting.
        holdout_fraction: fraction of observations held out for the
            PAC certificate.  ``0.0`` disables held-out evaluation.
        ridge: L2 penalty on log-parameters; tiny for stability.
        bootstrap_kind: ``"case"`` (resample rows) or ``"residual"``
            (wild Rademacher residuals).
        confidence: bootstrap-percentile coverage, e.g. ``0.95``.
    """
    family: str = FAMILY_CHINCHILLA
    seed: int = 0
    bootstrap_b: int = 200
    max_iters: int = 200
    tol: float = 1e-8
    flops_per_param_token: float = 6.0
    holdout_fraction: float = 0.2
    ridge: float = 1e-6
    bootstrap_kind: str = "case"
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.family not in KNOWN_FAMILIES:
            raise UnknownFamily(
                f"family={self.family!r} not in {KNOWN_FAMILIES!r}"
            )
        if self.bootstrap_b < 0:
            raise InvalidConfig("bootstrap_b must be >= 0")
        if self.max_iters <= 0:
            raise InvalidConfig("max_iters must be > 0")
        if not (0.0 <= self.holdout_fraction < 1.0):
            raise InvalidConfig("holdout_fraction must be in [0, 1)")
        if self.tol <= 0.0:
            raise InvalidConfig("tol must be > 0")
        if self.flops_per_param_token <= 0.0:
            raise InvalidConfig("flops_per_param_token must be > 0")
        if self.ridge < 0.0:
            raise InvalidConfig("ridge must be >= 0")
        if self.bootstrap_kind not in ("case", "residual"):
            raise InvalidConfig("bootstrap_kind must be 'case' or 'residual'")
        if not (0.0 < self.confidence < 1.0):
            raise InvalidConfig("confidence must be in (0, 1)")


@dataclass(frozen=True)
class Observation:
    """One ``(N, D, loss)`` triple.

    ``N`` and ``D`` are positive.  ``loss`` is positive (a log-loss /
    cross-entropy / negative-log-likelihood — anything monotone in
    capability).  ``weight`` defaults to ``1.0``; ``> 1`` upweights a
    row (e.g. a more trusted measurement).
    """
    n_params: float
    d_tokens: float
    loss: float
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, v in (("n_params", self.n_params),
                        ("d_tokens", self.d_tokens),
                        ("loss", self.loss)):
            if not isinstance(v, (int, float)):
                raise InvalidObservation(f"{name} must be numeric")
            if not math.isfinite(v):
                raise InvalidObservation(f"{name} must be finite, got {v}")
            if v <= 0.0:
                raise InvalidObservation(f"{name} must be > 0, got {v}")
        if not isinstance(self.weight, (int, float)) or self.weight <= 0:
            raise InvalidObservation("weight must be a positive number")


@dataclass(frozen=True)
class FitResult:
    """Result of a single fit on the in-sample observations."""
    family: str
    params: dict[str, float]
    rmse_in_sample: float
    rmse_held_out: float | None
    n_in_sample: int
    n_held_out: int
    iters: int
    converged: bool
    final_relative_change: float
    parameter_stderr: dict[str, float]
    log_residual_mean: float
    log_residual_std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "params": dict(self.params),
            "rmse_in_sample": self.rmse_in_sample,
            "rmse_held_out": self.rmse_held_out,
            "n_in_sample": self.n_in_sample,
            "n_held_out": self.n_held_out,
            "iters": self.iters,
            "converged": self.converged,
            "final_relative_change": self.final_relative_change,
            "parameter_stderr": dict(self.parameter_stderr),
            "log_residual_mean": self.log_residual_mean,
            "log_residual_std": self.log_residual_std,
        }


@dataclass(frozen=True)
class ExtrapolatePoint:
    """A single ``L(N, D)`` prediction with bootstrap-percentile CI."""
    n_params: float
    d_tokens: float
    loss_point: float
    loss_lower: float
    loss_upper: float
    confidence: float
    bootstrap_b: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_params": self.n_params,
            "d_tokens": self.d_tokens,
            "loss_point": self.loss_point,
            "loss_lower": self.loss_lower,
            "loss_upper": self.loss_upper,
            "confidence": self.confidence,
            "bootstrap_b": self.bootstrap_b,
        }


@dataclass(frozen=True)
class ComputeOptimal:
    """Closed-form compute-optimal ``(N*, D*, L*)`` plus a sanity check."""
    compute_budget: float
    n_star_analytic: float
    d_star_analytic: float
    loss_at_optimum: float
    n_star_numeric: float
    d_star_numeric: float
    loss_at_numeric: float
    family: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_budget": self.compute_budget,
            "n_star_analytic": self.n_star_analytic,
            "d_star_analytic": self.d_star_analytic,
            "loss_at_optimum": self.loss_at_optimum,
            "n_star_numeric": self.n_star_numeric,
            "d_star_numeric": self.d_star_numeric,
            "loss_at_numeric": self.loss_at_numeric,
            "family": self.family,
        }


@dataclass(frozen=True)
class ScalerCertificate:
    """Replay-verifiable certificate for the most recent fit.

    Carries the Hoeffding / empirical-Bernstein RMSE LCBs (in
    log-loss space, where residuals are unitless), dead-zone
    diagnostics, and the SHA-256 fingerprint chain over every
    observation, fit, extrapolation, and compute-optimal call.
    """
    family: str
    n_in_sample: int
    n_held_out: int
    rmse_in_sample: float
    rmse_held_out: float | None
    rmse_lcb_hoeffding: float | None
    rmse_lcb_bernstein: float | None
    in_range_n: tuple[float, float]
    in_range_d: tuple[float, float]
    extrapolation_factor_n: float
    extrapolation_factor_d: float
    fingerprint_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_in_sample": self.n_in_sample,
            "n_held_out": self.n_held_out,
            "rmse_in_sample": self.rmse_in_sample,
            "rmse_held_out": self.rmse_held_out,
            "rmse_lcb_hoeffding": self.rmse_lcb_hoeffding,
            "rmse_lcb_bernstein": self.rmse_lcb_bernstein,
            "in_range_n": list(self.in_range_n),
            "in_range_d": list(self.in_range_d),
            "extrapolation_factor_n": self.extrapolation_factor_n,
            "extrapolation_factor_d": self.extrapolation_factor_d,
            "fingerprint_hash": self.fingerprint_hash,
        }


@dataclass(frozen=True)
class ScalerReport:
    """A bundle of everything the coordinator needs to act on."""
    config: dict[str, Any]
    observations: int
    fit: dict[str, Any] | None
    certificate: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "observations": self.observations,
            "fit": dict(self.fit) if self.fit is not None else None,
            "certificate": dict(self.certificate) if self.certificate else None,
        }


# ---------------------------------------------------------------------------
# Scaling-law families
# ---------------------------------------------------------------------------


def _chinchilla_loss(params: Sequence[float], n: float, d: float) -> float:
    """L(N, D) = E + A/N^alpha + B/D^beta on positive log-parameters.

    params = [log_E, log_A, log_B, log_alpha, log_beta] so that E, A,
    B, alpha, beta are all strictly positive under any unconstrained
    optimiser update.  Returns L in raw (not log) space.
    """
    log_e, log_a, log_b, log_alpha, log_beta = params
    e = math.exp(log_e)
    a = math.exp(log_a)
    b = math.exp(log_b)
    alpha = math.exp(log_alpha)
    beta = math.exp(log_beta)
    return e + a * math.pow(n, -alpha) + b * math.pow(d, -beta)


def _chinchilla_compute_optimal(params: Sequence[float],
                                budget_c: float,
                                k: float) -> tuple[float, float]:
    """Closed-form N*, D* under the constraint C = k * N * D."""
    _, log_a, log_b, log_alpha, log_beta = params
    a = math.exp(log_a)
    b = math.exp(log_b)
    alpha = math.exp(log_alpha)
    beta = math.exp(log_beta)
    # See Hoffmann 2022 appendix C.
    g = alpha + beta
    ratio_n = (a * alpha) / (b * beta)
    n_star = math.pow(ratio_n, 1.0 / g) * math.pow(budget_c / k, beta / g)
    d_star = budget_c / (k * n_star)
    return n_star, d_star


def _kaplan_loss(params: Sequence[float], n: float, d: float) -> float:
    """Kaplan et al. 2020 form (positive params via exp)."""
    log_nc, log_dc, log_alpha_n, log_alpha_d = params
    nc = math.exp(log_nc)
    dc = math.exp(log_dc)
    alpha_n = math.exp(log_alpha_n)
    alpha_d = math.exp(log_alpha_d)
    inner = math.pow(nc / n, alpha_n / alpha_d) + (dc / d)
    return math.pow(inner, alpha_d)


def _bnsl_loss(params: Sequence[float], x: float) -> float:
    """One-break BNSL on a *single* variable ``x`` (Caballero 2023).

    L(X) = a + b X^{-c0} (1 + (X/d1)^{1/f1})^{-c1 f1}

    a, b, c0, c1, f1 > 0; d1 > 0 is the break location.
    """
    log_a, log_b, log_c0, log_c1, log_d1, log_f1 = params
    a = math.exp(log_a)
    b = math.exp(log_b)
    c0 = math.exp(log_c0)
    c1 = math.exp(log_c1)
    d1 = math.exp(log_d1)
    f1 = math.exp(log_f1)
    return a + b * math.pow(x, -c0) * math.pow(
        1.0 + math.pow(x / d1, 1.0 / f1), -c1 * f1
    )


def _bahri_loss(params: Sequence[float], x: float) -> float:
    """Bahri 2024: L(X) = L_inf + (X0/X)^alpha."""
    log_linf, log_x0, log_alpha = params
    linf = math.exp(log_linf)
    x0 = math.exp(log_x0)
    alpha = math.exp(log_alpha)
    return linf + math.pow(x0 / x, alpha)


# Initial-parameter guess in log space per family.
_INIT_GUESSES: dict[str, list[float]] = {
    FAMILY_CHINCHILLA: [math.log(1.5), math.log(400.0), math.log(400.0),
                        math.log(0.34), math.log(0.28)],
    FAMILY_KAPLAN: [math.log(8.8e13), math.log(5.4e13),
                    math.log(0.076), math.log(0.103)],
    FAMILY_BNSL: [math.log(0.5), math.log(1.0), math.log(0.2),
                  math.log(0.2), math.log(1e6), math.log(1.0)],
    FAMILY_BAHRI_N: [math.log(1.5), math.log(1e6), math.log(0.3)],
    FAMILY_BAHRI_D: [math.log(1.5), math.log(1e6), math.log(0.3)],
}

# Parameter names per family (in the order they appear in the vector).
_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    FAMILY_CHINCHILLA: ("E", "A", "B", "alpha", "beta"),
    FAMILY_KAPLAN: ("N_c", "D_c", "alpha_N", "alpha_D"),
    FAMILY_BNSL: ("a", "b", "c0", "c1", "d1", "f1"),
    FAMILY_BAHRI_N: ("L_inf", "N0", "alpha"),
    FAMILY_BAHRI_D: ("L_inf", "D0", "alpha"),
}


def _predict(family: str, params: Sequence[float],
             n: float, d: float) -> float:
    if family == FAMILY_CHINCHILLA:
        return _chinchilla_loss(params, n, d)
    if family == FAMILY_KAPLAN:
        return _kaplan_loss(params, n, d)
    if family == FAMILY_BNSL:
        # BNSL is single-variable; we use N as the scale.
        return _bnsl_loss(params, n)
    if family == FAMILY_BAHRI_N:
        return _bahri_loss(params, n)
    if family == FAMILY_BAHRI_D:
        return _bahri_loss(params, d)
    raise UnknownFamily(family)


def _named_params(family: str, params: Sequence[float]) -> dict[str, float]:
    names = _PARAM_NAMES[family]
    return {name: math.exp(v) for name, v in zip(names, params)}


# ---------------------------------------------------------------------------
# Levenberg-Marquardt with finite-difference Jacobians
# ---------------------------------------------------------------------------


def _matrix_eye(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _matmul_at_b(a_rows: list[list[float]], b_rows: list[list[float]]
                 ) -> list[list[float]]:
    """Return A^T B for matrices stored as row-major lists."""
    m = len(a_rows)
    n_a = len(a_rows[0]) if m else 0
    n_b = len(b_rows[0]) if b_rows else 0
    out = [[0.0] * n_b for _ in range(n_a)]
    for col in range(n_a):
        for row_b in range(len(b_rows)):
            v = a_rows[row_b][col]
            if v == 0.0:
                continue
            row_out = out[col]
            row_b_data = b_rows[row_b]
            for kk in range(n_b):
                row_out[kk] += v * row_b_data[kk]
    return out


def _matvec_at_b(a_rows: list[list[float]], b_vec: Sequence[float]
                 ) -> list[float]:
    """Return A^T b."""
    m = len(a_rows)
    n_a = len(a_rows[0]) if m else 0
    out = [0.0] * n_a
    for row in range(m):
        v = b_vec[row]
        if v == 0.0:
            continue
        row_a = a_rows[row]
        for col in range(n_a):
            out[col] += row_a[col] * v
    return out


def _cholesky_solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve symmetric-positive-definite ``A x = b`` via Cholesky.

    Returns ``None`` if ``A`` is not numerically SPD.
    """
    n = len(a)
    if n == 0:
        return []
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                d = a[i][i] - s
                if d <= 0.0:
                    return None
                l[i][j] = math.sqrt(d)
            else:
                if l[j][j] == 0.0:
                    return None
                l[i][j] = (a[i][j] - s) / l[j][j]
    # Solve L y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(l[i][k] * y[k] for k in range(i))) / l[i][i]
    # Solve L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(l[k][i] * x[k] for k in range(i + 1, n))) / l[i][i]
    return x


def _residuals(family: str, params: Sequence[float],
               obs: Sequence[Observation]) -> tuple[list[float], list[float]]:
    """Return ``(log-loss residuals, sqrt(weight) factors)``."""
    r = []
    sw = []
    for o in obs:
        pred = _predict(family, params, o.n_params, o.d_tokens)
        if pred <= 0.0 or not math.isfinite(pred):
            r.append(0.0)
            sw.append(0.0)
            continue
        r.append(math.log(o.loss) - math.log(pred))
        sw.append(math.sqrt(o.weight))
    return r, sw


def _weighted_residual(r: float, w: float) -> float:
    return r * w


def _finite_diff_jacobian(family: str, params: Sequence[float],
                          obs: Sequence[Observation],
                          eps: float = 1e-6) -> list[list[float]]:
    """Centered finite-difference Jacobian of log(pred) wrt params."""
    p = len(params)
    rows: list[list[float]] = []
    for o in obs:
        row = [0.0] * p
        for j in range(p):
            plus = list(params)
            minus = list(params)
            plus[j] = params[j] + eps
            minus[j] = params[j] - eps
            f_plus = _predict(family, plus, o.n_params, o.d_tokens)
            f_minus = _predict(family, minus, o.n_params, o.d_tokens)
            if f_plus <= 0.0 or f_minus <= 0.0 or not math.isfinite(f_plus) or not math.isfinite(f_minus):
                row[j] = 0.0
                continue
            row[j] = (math.log(f_plus) - math.log(f_minus)) / (2.0 * eps)
        sw = math.sqrt(o.weight)
        rows.append([v * sw for v in row])
    return rows


def _lm_fit(family: str, obs: Sequence[Observation], cfg: ScalerConfig
            ) -> tuple[list[float], int, bool, float]:
    """Levenberg-Marquardt in log-loss space with adaptive damping.

    Returns ``(params, iters, converged, final_rel_change)``.
    """
    if not obs:
        raise FitFailed("no observations to fit")
    params = list(_INIT_GUESSES[family])
    damping = 1e-3
    prev_rss = float("inf")
    last_rel = float("inf")
    converged = False
    iters = 0
    for it in range(cfg.max_iters):
        iters = it + 1
        r_raw, sw = _residuals(family, params, obs)
        r = [_weighted_residual(rv, w) for rv, w in zip(r_raw, sw)]
        rss = sum(v * v for v in r)
        j_rows = _finite_diff_jacobian(family, params, obs)
        jtj = _matmul_at_b(j_rows, j_rows)
        jr = _matvec_at_b(j_rows, r)
        ridge = cfg.ridge
        for i in range(len(params)):
            jtj[i][i] += damping + ridge
        step = _cholesky_solve(jtj, jr)
        if step is None:
            damping *= 10.0
            if damping > 1e12:
                break
            continue
        new_params = [params[i] + step[i] for i in range(len(params))]
        new_r_raw, new_sw = _residuals(family, new_params, obs)
        new_r = [_weighted_residual(rv, w) for rv, w in zip(new_r_raw, new_sw)]
        new_rss = sum(v * v for v in new_r)
        if new_rss < rss:
            # Accept and shrink damping.
            rel = sum(abs(s) for s in step) / max(
                1.0, sum(abs(p) for p in params)
            )
            params = new_params
            damping = max(damping / 3.0, 1e-12)
            last_rel = rel
            if rel < cfg.tol and abs(prev_rss - new_rss) < cfg.tol * (rss + 1e-12):
                converged = True
                break
            prev_rss = new_rss
        else:
            damping = min(damping * 5.0, 1e12)
    return params, iters, converged, last_rel


def _rmse(family: str, params: Sequence[float],
          obs: Sequence[Observation]) -> float:
    if not obs:
        return 0.0
    s = 0.0
    n = 0
    for o in obs:
        pred = _predict(family, params, o.n_params, o.d_tokens)
        if pred <= 0.0 or not math.isfinite(pred):
            pred = 1e-12
        diff = math.log(o.loss) - math.log(pred)
        s += diff * diff
        n += 1
    return math.sqrt(s / n) if n else 0.0


def _stderr_from_jacobian(family: str, params: Sequence[float],
                          obs: Sequence[Observation],
                          ridge: float) -> dict[str, float]:
    """Approximate per-parameter standard error via the Gauss-Newton Hessian."""
    if not obs:
        return {name: 0.0 for name in _PARAM_NAMES[family]}
    j_rows = _finite_diff_jacobian(family, params, obs)
    r_raw, sw = _residuals(family, params, obs)
    r = [_weighted_residual(rv, w) for rv, w in zip(r_raw, sw)]
    dof = max(1, len(obs) - len(params))
    sigma2 = sum(v * v for v in r) / dof
    jtj = _matmul_at_b(j_rows, j_rows)
    for i in range(len(params)):
        jtj[i][i] += ridge
    # Invert via Cholesky-solve of each unit basis vector.
    p = len(params)
    cov_diag = [0.0] * p
    for i in range(p):
        e = [0.0] * p
        e[i] = 1.0
        col = _cholesky_solve(jtj, e)
        if col is None:
            cov_diag[i] = float("inf")
        else:
            cov_diag[i] = col[i] * sigma2
    names = _PARAM_NAMES[family]
    return {
        names[i]: math.sqrt(cov_diag[i]) if math.isfinite(cov_diag[i]) and cov_diag[i] > 0 else 0.0
        for i in range(p)
    }


# ---------------------------------------------------------------------------
# Scaler primitive
# ---------------------------------------------------------------------------


class Scaler:
    """Scaling-law primitive.

    Lifecycle:

      1. ``observe()`` rows as they arrive.
      2. ``fit()`` produces a :class:`FitResult` plus updates internal state.
      3. ``extrapolate()`` predicts loss at unseen ``(N, D)`` with CI.
      4. ``compute_optimal()`` returns the closed-form ``(N*, D*, L*)``.
      5. ``certificate()`` bundles RMSE LCBs, range diagnostics, and the
         fingerprint chain.
      6. ``report()`` returns a single :class:`ScalerReport` for the
         coordination engine to consume.

    Thread-safe.  Deterministic given seed.
    """

    def __init__(self, config: ScalerConfig | None = None,
                 *, bus: EventBus | None = None) -> None:
        self.config = config or ScalerConfig()
        self.bus = bus
        self._lock = threading.RLock()
        self._obs: list[Observation] = []
        self._fit: FitResult | None = None
        self._params: list[float] | None = None
        self._fingerprint = hashlib.sha256()
        self._fingerprint.update(json.dumps(
            {"version": 1, "family": self.config.family,
             "seed": self.config.seed}, sort_keys=True
        ).encode())
        self._publish(SCALER_STARTED, {
            "family": self.config.family,
            "seed": self.config.seed,
        })

    # ----- event helpers -----
    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        payload = {**data, "ts": time.time()}
        self._fingerprint.update(json.dumps(
            {"kind": kind, "data": _stable(payload)}, sort_keys=True
        ).encode())
        if self.bus is not None:
            self.bus.publish(Event(kind=kind, data=payload))

    @property
    def fingerprint_hash(self) -> str:
        return self._fingerprint.hexdigest()

    # ----- ingestion -----
    def observe(self, obs: Observation | Iterable[Observation]) -> int:
        """Ingest one or more rows.  Returns the new total row count."""
        with self._lock:
            if isinstance(obs, Observation):
                rows = [obs]
            else:
                rows = list(obs)
            for o in rows:
                if not isinstance(o, Observation):
                    raise InvalidObservation(
                        f"expected Observation, got {type(o).__name__}"
                    )
                self._obs.append(o)
                self._publish(SCALER_OBSERVED, {
                    "n_params": o.n_params,
                    "d_tokens": o.d_tokens,
                    "loss": o.loss,
                    "weight": o.weight,
                })
            self._fit = None
            self._params = None
            return len(self._obs)

    def reset(self) -> None:
        with self._lock:
            self._obs.clear()
            self._fit = None
            self._params = None
            self._publish(SCALER_RESET, {})

    @property
    def observations(self) -> tuple[Observation, ...]:
        with self._lock:
            return tuple(self._obs)

    # ----- fitting -----
    def fit(self) -> FitResult:
        """Fit the configured family and return a :class:`FitResult`."""
        with self._lock:
            obs = list(self._obs)
            if len(obs) < self._min_obs():
                raise FitFailed(
                    f"need >= {self._min_obs()} observations for family "
                    f"{self.config.family!r}, have {len(obs)}"
                )
            rng = random.Random(self.config.seed)
            # Deterministic shuffle for the holdout split.
            indices = list(range(len(obs)))
            rng.shuffle(indices)
            holdout_n = int(round(self.config.holdout_fraction * len(obs)))
            holdout_n = min(holdout_n, max(0, len(obs) - self._min_obs()))
            held_idx = set(indices[:holdout_n])
            in_idx = [i for i in range(len(obs)) if i not in held_idx]
            held_idx_list = sorted(held_idx)
            in_obs = [obs[i] for i in in_idx]
            held_obs = [obs[i] for i in held_idx_list]
            params, iters, converged, rel = _lm_fit(
                self.config.family, in_obs, self.config
            )
            self._params = params
            rmse_in = _rmse(self.config.family, params, in_obs)
            rmse_held = _rmse(self.config.family, params, held_obs) if held_obs else None
            stderr = _stderr_from_jacobian(self.config.family, params, in_obs,
                                           self.config.ridge)
            # Log-residual mean / std (for residual bootstrap).
            r_raw, _ = _residuals(self.config.family, params, in_obs)
            mu = sum(r_raw) / len(r_raw) if r_raw else 0.0
            if len(r_raw) > 1:
                v = sum((x - mu) ** 2 for x in r_raw) / (len(r_raw) - 1)
                sd = math.sqrt(max(v, 0.0))
            else:
                sd = 0.0
            result = FitResult(
                family=self.config.family,
                params=_named_params(self.config.family, params),
                rmse_in_sample=rmse_in,
                rmse_held_out=rmse_held,
                n_in_sample=len(in_obs),
                n_held_out=len(held_obs),
                iters=iters,
                converged=converged,
                final_relative_change=rel,
                parameter_stderr=stderr,
                log_residual_mean=mu,
                log_residual_std=sd,
            )
            self._fit = result
            self._publish(SCALER_FIT, {
                "family": result.family,
                "params": result.params,
                "rmse_in_sample": result.rmse_in_sample,
                "rmse_held_out": result.rmse_held_out,
                "iters": result.iters,
                "converged": result.converged,
            })
            return result

    def _min_obs(self) -> int:
        return len(_INIT_GUESSES[self.config.family]) + 1

    # ----- extrapolation -----
    def extrapolate(self, n: float, d: float) -> ExtrapolatePoint:
        """Predict ``L(N, D)`` with bootstrap-percentile CI.

        Raises :class:`NotFitted` if :meth:`fit` has not been called.
        """
        with self._lock:
            if self._params is None or self._fit is None:
                raise NotFitted("call fit() before extrapolate()")
            if not (n > 0.0 and math.isfinite(n)
                    and d > 0.0 and math.isfinite(d)):
                raise InvalidObservation(
                    f"extrapolate requires positive finite (n, d), got ({n}, {d})"
                )
            point = _predict(self.config.family, self._params, n, d)
            if self.config.bootstrap_b == 0:
                return ExtrapolatePoint(
                    n_params=n, d_tokens=d,
                    loss_point=point, loss_lower=point, loss_upper=point,
                    confidence=self.config.confidence,
                    bootstrap_b=0,
                )
            samples = self._bootstrap_predictions(n, d)
            samples.sort()
            alpha = (1.0 - self.config.confidence) / 2.0
            lo_idx = max(0, int(math.floor(alpha * len(samples))))
            hi_idx = min(len(samples) - 1, int(math.ceil((1.0 - alpha) * len(samples))) - 1)
            lo = samples[lo_idx]
            hi = samples[hi_idx]
            out = ExtrapolatePoint(
                n_params=n, d_tokens=d,
                loss_point=point, loss_lower=lo, loss_upper=hi,
                confidence=self.config.confidence,
                bootstrap_b=self.config.bootstrap_b,
            )
            self._publish(SCALER_EXTRAPOLATED, {
                "n_params": n, "d_tokens": d,
                "loss_point": point,
                "loss_lower": lo, "loss_upper": hi,
            })
            return out

    def _bootstrap_predictions(self, n: float, d: float) -> list[float]:
        assert self._fit is not None and self._params is not None
        rng = random.Random(self.config.seed + 1)
        family = self.config.family
        obs = list(self._obs)
        # Restrict to in-sample subset used for the fit.
        # (Held-out rows aren't part of parameter uncertainty.)
        # Simpler: refit on a resample of all rows; this matches Efron 1979.
        preds: list[float] = []
        if self.config.bootstrap_kind == "case":
            for _ in range(self.config.bootstrap_b):
                resample = [obs[rng.randrange(len(obs))]
                            for _ in range(len(obs))]
                try:
                    params, _, _, _ = _lm_fit(family, resample, self.config)
                except FitFailed:
                    continue
                pred = _predict(family, params, n, d)
                if math.isfinite(pred) and pred > 0.0:
                    preds.append(pred)
        else:
            # Wild Rademacher residual bootstrap.
            base_params = list(self._params)
            r_raw, _ = _residuals(family, base_params, obs)
            for _ in range(self.config.bootstrap_b):
                synthetic: list[Observation] = []
                for o, r in zip(obs, r_raw):
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    new_log_loss = math.log(_predict(family, base_params,
                                                     o.n_params, o.d_tokens)) + sign * r
                    new_loss = math.exp(new_log_loss)
                    if new_loss > 0.0 and math.isfinite(new_loss):
                        synthetic.append(Observation(
                            n_params=o.n_params, d_tokens=o.d_tokens,
                            loss=new_loss, weight=o.weight,
                        ))
                if len(synthetic) < self._min_obs():
                    continue
                try:
                    params, _, _, _ = _lm_fit(family, synthetic, self.config)
                except FitFailed:
                    continue
                pred = _predict(family, params, n, d)
                if math.isfinite(pred) and pred > 0.0:
                    preds.append(pred)
        if not preds:
            preds = [_predict(family, self._params, n, d)]
        return preds

    # ----- compute-optimal -----
    def compute_optimal(self, budget_c: float) -> ComputeOptimal:
        """Return ``(N*, D*, L*)`` for a future FLOP budget ``C``.

        Closed-form analytic optimum is sanity-checked against a
        numerical sweep over a logarithmic grid of ``N`` values (with
        ``D = C / (k N)``).  ``family`` must be Chinchilla.
        """
        with self._lock:
            if self._params is None:
                raise NotFitted("call fit() before compute_optimal()")
            if self.config.family != FAMILY_CHINCHILLA:
                raise InvalidConfig(
                    "compute_optimal requires family=chinchilla; "
                    f"got {self.config.family!r}"
                )
            if budget_c <= 0.0 or not math.isfinite(budget_c):
                raise InvalidConfig("budget_c must be positive finite")
            k = self.config.flops_per_param_token
            n_an, d_an = _chinchilla_compute_optimal(self._params, budget_c, k)
            l_an = _chinchilla_loss(self._params, n_an, d_an)
            # Numerical sweep for sanity.
            best_n = n_an
            best_d = d_an
            best_l = l_an
            for log_factor in [-4.0, -3.0, -2.0, -1.0, -0.5, -0.2, 0.0,
                               0.2, 0.5, 1.0, 2.0, 3.0, 4.0]:
                n_try = n_an * math.exp(log_factor)
                d_try = budget_c / (k * n_try)
                if n_try <= 0 or d_try <= 0:
                    continue
                l_try = _chinchilla_loss(self._params, n_try, d_try)
                if math.isfinite(l_try) and l_try < best_l:
                    best_l = l_try
                    best_n = n_try
                    best_d = d_try
            out = ComputeOptimal(
                compute_budget=budget_c,
                n_star_analytic=n_an,
                d_star_analytic=d_an,
                loss_at_optimum=l_an,
                n_star_numeric=best_n,
                d_star_numeric=best_d,
                loss_at_numeric=best_l,
                family=self.config.family,
            )
            self._publish(SCALER_OPTIMAL, out.to_dict())
            return out

    # ----- certificate -----
    def certificate(self,
                    *, confidence: float | None = None
                    ) -> ScalerCertificate:
        """Return a replay-verifiable certificate over the current fit."""
        with self._lock:
            if self._fit is None or self._params is None:
                raise NotFitted("call fit() before certificate()")
            conf = confidence if confidence is not None else self.config.confidence
            if not (0.0 < conf < 1.0):
                raise InvalidConfig("confidence must be in (0,1)")
            f = self._fit
            n_range = (
                min(o.n_params for o in self._obs),
                max(o.n_params for o in self._obs),
            )
            d_range = (
                min(o.d_tokens for o in self._obs),
                max(o.d_tokens for o in self._obs),
            )
            held_lcb_h: float | None = None
            held_lcb_b: float | None = None
            if f.rmse_held_out is not None and f.n_held_out > 0:
                m = f.n_held_out
                delta = 1.0 - conf
                # Bound the per-sample log-residual to a clipped range so
                # Hoeffding has a finite spread.  We use the observed range.
                # The certificate is therefore on the rescaled RMSE; the
                # raw RMSE is reported unchanged.
                hoeffding = math.sqrt(math.log(1.0 / delta) / (2.0 * m))
                # Empirical-Bernstein (Maurer-Pontil 2009).
                # sigma^2 estimated from squared log-residuals.
                # In log-loss space residuals are unitless; clip = sigma * 4.
                sigma = f.log_residual_std if f.log_residual_std > 0 else 1e-6
                clip = 4.0 * sigma
                bernstein = (
                    math.sqrt(2.0 * sigma * sigma * math.log(2.0 / delta) / m)
                    + 7.0 * clip * math.log(2.0 / delta) / (3.0 * (m - 1) if m > 1 else 1)
                )
                held_lcb_h = max(0.0, f.rmse_held_out - hoeffding)
                held_lcb_b = max(0.0, f.rmse_held_out - bernstein)
            cert = ScalerCertificate(
                family=f.family,
                n_in_sample=f.n_in_sample,
                n_held_out=f.n_held_out,
                rmse_in_sample=f.rmse_in_sample,
                rmse_held_out=f.rmse_held_out,
                rmse_lcb_hoeffding=held_lcb_h,
                rmse_lcb_bernstein=held_lcb_b,
                in_range_n=n_range,
                in_range_d=d_range,
                extrapolation_factor_n=n_range[1] / n_range[0] if n_range[0] > 0 else float("inf"),
                extrapolation_factor_d=d_range[1] / d_range[0] if d_range[0] > 0 else float("inf"),
                fingerprint_hash=self.fingerprint_hash,
            )
            self._publish(SCALER_CERTIFIED, cert.to_dict())
            return cert

    # ----- report -----
    def report(self) -> ScalerReport:
        with self._lock:
            cfg = {
                "family": self.config.family,
                "seed": self.config.seed,
                "bootstrap_b": self.config.bootstrap_b,
                "max_iters": self.config.max_iters,
                "tol": self.config.tol,
                "flops_per_param_token": self.config.flops_per_param_token,
                "holdout_fraction": self.config.holdout_fraction,
                "ridge": self.config.ridge,
                "bootstrap_kind": self.config.bootstrap_kind,
                "confidence": self.config.confidence,
            }
            fit = self._fit.to_dict() if self._fit else None
            cert = self.certificate().to_dict() if self._fit else None
            out = ScalerReport(
                config=cfg,
                observations=len(self._obs),
                fit=fit,
                certificate=cert,
            )
            self._publish(SCALER_REPORTED, {
                "observations": out.observations,
                "family": self.config.family,
            })
            return out


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def chinchilla_scaler(*, seed: int = 0, **kw: Any) -> Scaler:
    return Scaler(ScalerConfig(family=FAMILY_CHINCHILLA, seed=seed, **kw))


def kaplan_scaler(*, seed: int = 0, **kw: Any) -> Scaler:
    return Scaler(ScalerConfig(family=FAMILY_KAPLAN, seed=seed, **kw))


def bnsl_scaler(*, seed: int = 0, **kw: Any) -> Scaler:
    return Scaler(ScalerConfig(family=FAMILY_BNSL, seed=seed, **kw))


def bahri_scaler(*, axis: str = "n", seed: int = 0, **kw: Any) -> Scaler:
    fam = FAMILY_BAHRI_N if axis == "n" else FAMILY_BAHRI_D
    return Scaler(ScalerConfig(family=fam, seed=seed, **kw))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable(value: Any) -> Any:
    """Make a payload deterministic for the fingerprint chain."""
    if isinstance(value, dict):
        return {k: _stable(value[k]) for k in sorted(value.keys()) if k != "ts"}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        # 12 significant digits is enough for log-loss accounting.
        return float(f"{value:.12g}")
    return value


__all__ = [
    "FAMILY_CHINCHILLA",
    "FAMILY_KAPLAN",
    "FAMILY_BNSL",
    "FAMILY_BAHRI_N",
    "FAMILY_BAHRI_D",
    "KNOWN_FAMILIES",
    "SCALER_STARTED",
    "SCALER_OBSERVED",
    "SCALER_FIT",
    "SCALER_EXTRAPOLATED",
    "SCALER_OPTIMAL",
    "SCALER_CERTIFIED",
    "SCALER_REPORTED",
    "SCALER_RESET",
    "ScalerError",
    "InvalidConfig",
    "InvalidObservation",
    "UnknownFamily",
    "NotFitted",
    "FitFailed",
    "ScalerConfig",
    "Observation",
    "FitResult",
    "ExtrapolatePoint",
    "ComputeOptimal",
    "ScalerCertificate",
    "ScalerReport",
    "Scaler",
    "chinchilla_scaler",
    "kaplan_scaler",
    "bnsl_scaler",
    "bahri_scaler",
]
