r"""Scientist — sparse symbolic law discovery as a runtime primitive.

Every learning primitive in this runtime fits a model whose *form* is
fixed before fitting begins.  ``Forecaster`` fits a parametric mean
process to time series.  ``Predictor`` mixes a fixed exponential class
of variable-order Markov models.  ``Filterer`` runs Bayesian state-
estimation against a *known* linear-Gaussian dynamics.  ``CausalDiscoverer``
recovers a graph but each edge is still a parametric coefficient.  None
of them produces an interpretable, closed-form *law* — a finite
arithmetic expression an investor, a domain expert, or a downstream
verifier could read on a slide.

The ``Scientist`` is the runtime primitive that closes that gap.
Given a stream of pairs ``(x ∈ ℝᵈ, y ∈ ℝ)`` it discovers a sparse
linear combination of *symbolic basis functions* — monomials,
inverses, cross-products, sines/cosines, exponentials, logarithms,
plus any user-supplied callable — that explains ``y`` as a function
of ``x``, and returns it as a ``Law`` object carrying:

* a printable closed-form expression (``"y ≈ 0.500·x0² − 9.810·1"``);
* per-coefficient bootstrap confidence intervals (Efron 1979);
* per-term stability-selection inclusion frequencies
  (Meinshausen-Bühlmann 2010);
* AIC, BIC, MDL ranking against the empty-model null;
* in-sample R² and (Akaike-corrected) out-of-sample generalisation
  bound;
* the full **Pareto frontier** of (complexity, NMSE) so the
  coordination engine can route on the bias/sparsity tradeoff;
* a SHA-256 fingerprint chain over every ``observe`` / ``fit`` /
  ``report`` call, compatible with ``AttestationLedger``.

The pitch reduced to a runtime call::

    sci = Scientist.create(input_dim=1, max_degree=2, seed=0)
    for x, y in falling_body_data:
        sci.observe([x], y)
    law = sci.fit()           # → "y ≈ -4.905·t² + v0·t + h0"
    front = sci.pareto()      # complexity ↔ residual trade-off
    rep = sci.report()

Mathematical roots
------------------

* **Brunton-Proctor-Kutz 2016 — Sparse Identification of Nonlinear
  Dynamics (SINDy).**  Fit ``ẋ = Θ(x) ξ`` by alternating
  least-squares and hard thresholding (STLSQ): solve OLS, zero
  coefficients with ``|ξ_j| < λ``, re-fit on the kept columns,
  iterate until the support stabilises.  The fixed point is the
  ℓ⁰-constrained projection of the OLS solution onto the basis
  subset that survives the threshold; under mild restricted-isometry
  assumptions on Θ this is the global ℓ⁰ optimum on the kept basis
  (Hastie-Tibshirani-Wainwright 2015 §2.5).  ``Scientist`` implements
  this exactly, with ``lambda_grid`` sweeping λ to expose the full
  Pareto frontier in a single fit call.

* **Akaike 1973 — AIC.**  ``AIC = 2k − 2 log L̂``, where ``k`` is the
  number of free parameters (kept basis terms + 1 for the residual
  variance) and ``log L̂`` is the maximised Gaussian log-likelihood
  ``-n/2 · log(2π e σ²)``.  Under Gaussian residuals this reduces to
  ``AIC = n log(RSS/n) + 2k + const`` (Burnham-Anderson 2002 §2.2).
  Models are *Akaike-weighted*: ``w_i = exp(-ΔAIC_i / 2) /
  Σ exp(-ΔAIC_j / 2)``, giving an evidence ratio between any two
  laws on the Pareto frontier.

* **Schwarz 1978 — BIC.**  ``BIC = k log n − 2 log L̂``; under
  Gaussian residuals ``BIC = n log(RSS/n) + k log n + const``.
  Consistent for the true sparse support as ``n → ∞`` if the support
  is contained in the library.

* **Rissanen 1978 — MDL (two-part code).**  Encode the model
  (support pattern + quantised coefficients) and the residuals
  separately; the total bit-length is minimised at the *parsimonious*
  law.  ``Scientist`` returns the **normalised description length**

  .. math::

      \frac{L(\xi) + L(y \mid \xi)}{n}
      \;=\; \tfrac{1}{n}\bigl(
        k \log_2 p + \tfrac{k}{2}\log_2 n
        + \tfrac{n}{2}\log_2(2\pi e \hat\sigma^2)
        \bigr)

  in *bits per sample* — a length-independent quantity directly
  comparable across stream lengths and library sizes.

* **Efron 1979 — Bootstrap.**  For each kept basis function we
  bootstrap-resample ``(x_i, y_i)`` pairs, refit OLS on the *same*
  selected support, and report the empirical ``(α/2, 1−α/2)``
  percentile CI on each coefficient.  The same resamples drive
  **stability selection** (Meinshausen-Bühlmann 2010): for each
  resample we re-run STLSQ end-to-end and count how often each basis
  function is kept.  Inclusion frequency ≥ ``π_thr`` defines a
  *stable* support that controls per-family error under mild
  exchangeability (Theorem 1 of MB 2010).

* **Pareto frontier.**  Each ``λ ∈ lambda_grid`` produces one law of
  complexity ``k(λ)`` and residual-sum-of-squares ``RSS(λ)``.  The
  pairs ``(k, RSS)`` define the *complexity-error frontier*.  The
  non-dominated points — those for which no law of lower complexity
  has lower RSS — form the Pareto front returned by ``pareto()``.
  AIC, BIC and MDL each select one element of this front; the user
  / coordinator can pick by any other criterion (e.g. "smallest law
  whose RSS is within 5 % of the best").

Why is this the right primitive
-------------------------------

A coordination engine that can call

    law = Scientist.fit(observation_stream)

closes a loop none of the other primitives close: from
*observations* to *interpretable mechanism*.  ``Forecaster`` predicts
the next number; ``Filterer`` tracks a latent state; ``CausalDiscoverer``
finds an arrow.  Only ``Scientist`` returns the *formula*.  That
formula is then:

* an audit artefact a regulator can read;
* a hypothesis ``Refuter`` can try to break;
* a closed-form prior ``Filterer`` can plug into its dynamics;
* a typed program ``Synthesizer`` can lift into a tool;
* a step in a ``KnowledgeGraph`` fact whose edges carry numerical
  coefficients;
* the parsimony selector that prevents ``Composer`` from greedily
  enumerating an explosion of basis terms.

All of which makes the runtime more — to a domain expert — *legible*.

Implementation notes
--------------------

* Pure stdlib — ``math``, ``hashlib``, ``random``, ``struct``,
  ``threading``.  No NumPy, no SciPy, no SymPy.  Linear systems are
  solved via Cholesky on the (small) Gram matrix ``Φ_kᵀΦ_k``;
  ridge ``γ·I`` is added when the Gram is rank-deficient.  Cost is
  ``O(n·p + k³)`` per STLSQ iteration with ``k`` ≤ ``p`` the kept
  support size.

* Deterministic given ``seed``.  Every ``observe`` / ``fit`` /
  ``report`` event hashes into a SHA-256 chain consumed by
  ``AttestationLedger``.

* Thread-safe via a re-entrant lock.

* Crash-safe: ``observe`` appends to in-memory buffers only; no
  on-disk state.  A caller can wrap with ``SessionStore`` for
  persistence.

This file is intentionally self-contained; the only intra-package
dependency is the event bus.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds — published when a bus is supplied
# =====================================================================

SCIENTIST_STARTED = "scientist.started"
SCIENTIST_OBSERVED = "scientist.observed"
SCIENTIST_FITTED = "scientist.fitted"
SCIENTIST_PARETO = "scientist.pareto"
SCIENTIST_BOOTSTRAPPED = "scientist.bootstrapped"
SCIENTIST_STABILITY = "scientist.stability"
SCIENTIST_PREDICTED = "scientist.predicted"
SCIENTIST_REPORTED = "scientist.reported"
SCIENTIST_CLEARED = "scientist.cleared"

SCIENTIST_KNOWN_EVENTS = frozenset(
    {
        SCIENTIST_STARTED,
        SCIENTIST_OBSERVED,
        SCIENTIST_FITTED,
        SCIENTIST_PARETO,
        SCIENTIST_BOOTSTRAPPED,
        SCIENTIST_STABILITY,
        SCIENTIST_PREDICTED,
        SCIENTIST_REPORTED,
        SCIENTIST_CLEARED,
    }
)


# Selection criteria
SELECT_AIC = "aic"
SELECT_BIC = "bic"
SELECT_MDL = "mdl"
SELECT_PARETO_KNEE = "pareto_knee"

SCIENTIST_KNOWN_CRITERIA = frozenset(
    {SELECT_AIC, SELECT_BIC, SELECT_MDL, SELECT_PARETO_KNEE}
)


# =====================================================================
# Errors
# =====================================================================


class ScientistError(Exception):
    """Base error for the Scientist primitive."""


class InvalidConfig(ScientistError):
    """Configuration values out of range."""


class InvalidObservation(ScientistError):
    """Observation rejected (wrong arity, NaN, etc.)."""


class InsufficientData(ScientistError):
    """Operation requires more observations than have been collected."""


class NotYetFitted(ScientistError):
    """A method that requires a prior ``fit()`` was called before fitting."""


class InvalidBasis(ScientistError):
    """A user-supplied basis function is malformed."""


class InvalidCriterion(ScientistError):
    """Selection criterion not in SCIENTIST_KNOWN_CRITERIA."""


# =====================================================================
# Constants
# =====================================================================

_GENESIS = "0" * 64
_EPS = 1e-12
_INF = float("inf")
_DEFAULT_LAMBDA_GRID: tuple[float, ...] = (
    1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0,
)
_DEFAULT_RIDGE = 1e-8
_DEFAULT_BOOTSTRAP_SAMPLES = 200
_LOG2 = math.log(2.0)


# =====================================================================
# Hash chain
# =====================================================================


def _hash_link(prev: str, payload: str) -> str:
    """SHA-256 chain: ``H(prev || 0x1f || payload)``."""
    h = hashlib.sha256()
    h.update(prev.encode("ascii"))
    h.update(b"\x1f")
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def _payload_repr(obj: Any) -> str:
    """Canonical deterministic string repr for hashing."""
    if isinstance(obj, dict):
        keys = sorted(obj.keys(), key=str)
        return "{" + ",".join(f"{k}={_payload_repr(obj[k])}" for k in keys) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_payload_repr(x) for x in obj) + "]"
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return f"{obj:.17g}"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    return repr(obj)


# =====================================================================
# Linear algebra — pure stdlib
# =====================================================================


def _cholesky(A: Sequence[Sequence[float]], ridge: float = _DEFAULT_RIDGE) -> list[list[float]]:
    """Lower-triangular Cholesky factor with auto-jitter on rank deficiency.

    Adds ``ridge·I`` and increases by powers of 10 on failure.  Returns
    the factor for the *jittered* Gram matrix; the caller can read the
    effective ridge from ``L[-1][-1]`` only indirectly, so for safety
    we cap the jitter at ``ridge·1e8`` and raise ``ScientistError`` if
    the matrix is still indefinite (which should never happen for a
    Gram matrix of real features).
    """
    n = len(A)
    jit = ridge
    for _ in range(9):  # try up to 9 orders of magnitude of jitter
        L = [[0.0] * n for _ in range(n)]
        ok = True
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    v = A[i][i] + jit - s
                    if v <= 0.0:
                        ok = False
                        break
                    L[i][j] = math.sqrt(v)
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], _EPS)
            if not ok:
                break
        if ok:
            return L
        jit *= 10.0
    raise ScientistError("Cholesky failed even with maximum jitter")


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


def _solve_normal_eq(
    Phi: Sequence[Sequence[float]],
    y: Sequence[float],
    ridge: float = _DEFAULT_RIDGE,
) -> list[float]:
    """Solve ``min_ξ ‖Φ ξ - y‖²`` via the ridge-regularised normal equations.

    Forms ``G = Φᵀ Φ + ridge·I`` (with auto-jitter on rank-deficiency)
    and ``r = Φᵀ y``; returns ``ξ = G⁻¹ r``.  Costs ``O(n p² + p³)``
    where ``n = len(y)`` and ``p`` is the number of features.
    """
    n = len(Phi)
    if n == 0:
        return []
    p = len(Phi[0])
    if p == 0:
        return []
    # Gram matrix Φᵀ Φ
    G = [[0.0] * p for _ in range(p)]
    for i in range(n):
        row = Phi[i]
        for a in range(p):
            ra = row[a]
            if ra == 0.0:
                continue
            for b in range(a, p):
                G[a][b] += ra * row[b]
    for a in range(p):
        for b in range(a):
            G[a][b] = G[b][a]
    # Right-hand side Φᵀ y
    r = [0.0] * p
    for i in range(n):
        yi = y[i]
        row = Phi[i]
        for a in range(p):
            r[a] += row[a] * yi
    # Cholesky solve
    L = _cholesky(G, ridge=ridge)
    z = _solve_lower_tri(L, r)
    Lt = [[L[j][i] for j in range(p)] for i in range(p)]
    xi = _solve_upper_tri(Lt, z)
    return xi


# =====================================================================
# Basis library
# =====================================================================


@dataclass(frozen=True)
class Basis:
    """A single basis function ``φ_j(x) → ℝ`` with a pretty name.

    ``fn`` is a deterministic, side-effect-free callable from a
    ``Sequence[float]`` of length ``input_dim`` to a real number.
    ``name`` is the *symbolic* form printed in laws and used as the
    deterministic identifier in the hash chain.  ``complexity`` is the
    integer cost charged for this term in the Pareto / MDL accounting
    (typically 1 for a single multiplication, 2 for transcendental
    functions, etc.).
    """

    name: str
    fn: Callable[[Sequence[float]], float]
    complexity: int = 1

    def __call__(self, x: Sequence[float]) -> float:
        return float(self.fn(x))


def _monomial_basis(input_dim: int, max_degree: int) -> list[Basis]:
    """Total-degree monomials ``x0^a0 · x1^a1 · …`` with ``Σa_i ≤ max_degree``.

    The constant ``1`` (the empty product) is included as the first
    basis; ``Σa_i = 0`` is the intercept.  For ``input_dim = d`` and
    ``max_degree = D`` this yields ``C(d+D, D)`` terms.  We enumerate
    by ascending degree, then lexicographic on exponent tuples, so the
    library order is canonical.
    """
    if input_dim < 0:
        raise InvalidConfig("input_dim must be >= 0")
    if max_degree < 0:
        raise InvalidConfig("max_degree must be >= 0")

    def _enum(rem: int, dim_left: int) -> list[tuple[int, ...]]:
        if dim_left == 0:
            return [tuple()]
        out: list[tuple[int, ...]] = []
        for k in range(rem + 1):
            for tail in _enum(rem - k, dim_left - 1):
                out.append((k,) + tail)
        return out

    bases: list[Basis] = []
    for deg in range(max_degree + 1):
        # exponents that sum to exactly `deg`
        for exps in _enum(deg, input_dim):
            if sum(exps) != deg:
                continue
            if all(e == 0 for e in exps):
                bases.append(Basis(name="1", fn=lambda x: 1.0, complexity=1))
                continue
            # build the lambda capturing exps
            name_parts: list[str] = []
            for i, e in enumerate(exps):
                if e == 0:
                    continue
                if e == 1:
                    name_parts.append(f"x{i}")
                else:
                    name_parts.append(f"x{i}^{e}")
            name = "·".join(name_parts) if name_parts else "1"
            complexity = max(1, sum(exps))

            def _make_fn(exps_=exps):
                def _fn(x: Sequence[float]) -> float:
                    p = 1.0
                    for j, ej in enumerate(exps_):
                        if ej == 0:
                            continue
                        v = x[j]
                        if ej == 1:
                            p *= v
                        else:
                            p *= v ** ej
                    return p
                return _fn

            bases.append(Basis(name=name, fn=_make_fn(), complexity=complexity))
    return bases


def _trig_basis(input_dim: int, frequencies: Sequence[float] = (1.0,)) -> list[Basis]:
    """Sin/cos of each input coordinate at each given frequency."""
    out: list[Basis] = []
    for i in range(input_dim):
        for w in frequencies:
            w_ = float(w)

            def _msin(idx=i, ww=w_):
                def _fn(x: Sequence[float]) -> float:
                    return math.sin(ww * x[idx])
                return _fn

            def _mcos(idx=i, ww=w_):
                def _fn(x: Sequence[float]) -> float:
                    return math.cos(ww * x[idx])
                return _fn

            tag = "" if abs(w_ - 1.0) < 1e-12 else f"{w_:g}·"
            out.append(Basis(name=f"sin({tag}x{i})", fn=_msin(), complexity=2))
            out.append(Basis(name=f"cos({tag}x{i})", fn=_mcos(), complexity=2))
    return out


def _exp_basis(input_dim: int) -> list[Basis]:
    """``exp(x_i)`` for each coordinate.

    Caller is responsible for ensuring inputs don't overflow; we
    silently clip the exponent argument to [-50, 50] to keep finite
    values everywhere.
    """
    out: list[Basis] = []
    for i in range(input_dim):
        def _mexp(idx=i):
            def _fn(x: Sequence[float]) -> float:
                v = x[idx]
                if v > 50.0:
                    v = 50.0
                elif v < -50.0:
                    v = -50.0
                return math.exp(v)
            return _fn
        out.append(Basis(name=f"exp(x{i})", fn=_mexp(), complexity=2))
    return out


def _log_basis(input_dim: int) -> list[Basis]:
    """``log(1 + |x_i|)`` for each coordinate — domain-safe log."""
    out: list[Basis] = []
    for i in range(input_dim):
        def _mlog(idx=i):
            def _fn(x: Sequence[float]) -> float:
                return math.log1p(abs(x[idx]))
            return _fn
        out.append(Basis(name=f"log1p(|x{i}|)", fn=_mlog(), complexity=2))
    return out


def _inv_basis(input_dim: int, regulariser: float = 1e-6) -> list[Basis]:
    """``1/(x_i + ε)`` with a small constant to avoid singularities."""
    out: list[Basis] = []
    for i in range(input_dim):
        def _minv(idx=i, reg=regulariser):
            def _fn(x: Sequence[float]) -> float:
                v = x[idx]
                denom = v if abs(v) > reg else (reg if v >= 0 else -reg)
                return 1.0 / denom
            return _fn
        out.append(Basis(name=f"1/x{i}", fn=_minv(), complexity=2))
    return out


def default_library(
    input_dim: int,
    *,
    max_degree: int = 2,
    include_trig: bool = False,
    trig_frequencies: Sequence[float] = (1.0,),
    include_exp: bool = False,
    include_log: bool = False,
    include_inv: bool = False,
    extra: Sequence[Basis] = (),
) -> list[Basis]:
    """Convenience: assemble the canonical default basis library.

    The default library is monomials of total degree ≤ ``max_degree``;
    transcendental and inverse families are opt-in.  ``extra`` is a
    list of user-supplied :class:`Basis` objects appended at the end
    of the library — their order is preserved so the deterministic
    hash chain is stable across runs.
    """
    if input_dim < 0:
        raise InvalidConfig("input_dim must be >= 0")
    lib: list[Basis] = _monomial_basis(input_dim, max_degree)
    if include_trig:
        lib.extend(_trig_basis(input_dim, trig_frequencies))
    if include_exp:
        lib.extend(_exp_basis(input_dim))
    if include_log:
        lib.extend(_log_basis(input_dim))
    if include_inv:
        lib.extend(_inv_basis(input_dim))
    for b in extra:
        if not isinstance(b, Basis):
            raise InvalidBasis(f"extra entry {b!r} is not a Basis")
        lib.append(b)
    # Dedupe by name — preserve first occurrence.
    seen: set[str] = set()
    out: list[Basis] = []
    for b in lib:
        if b.name in seen:
            continue
        seen.add(b.name)
        out.append(b)
    return out


# =====================================================================
# Result types
# =====================================================================


@dataclass(frozen=True)
class Term:
    """A single (basis, coefficient) pair in a discovered law."""

    name: str
    index: int
    coefficient: float
    complexity: int

    def __str__(self) -> str:
        coef = self.coefficient
        sign = "−" if coef < 0 else "+"
        a = abs(coef)
        if self.name == "1":
            return f"{sign} {a:.6g}"
        return f"{sign} {a:.6g}·{self.name}"


@dataclass(frozen=True)
class Law:
    """A discovered closed-form law.

    Attributes
    ----------
    lam :
        The STLSQ threshold ``λ`` at which this law was selected.
    terms :
        Non-zero ``(basis, coefficient)`` pairs in library order.
    rss :
        Residual sum of squares on the training set.
    n :
        Number of observations the law was fit on.
    sigma2 :
        Maximum-likelihood residual variance ``RSS/n``.
    r2 :
        Coefficient of determination ``1 − RSS / TSS`` on training data.
    aic :
        ``n·log(RSS/n) + 2k`` (additive const dropped).
    bic :
        ``n·log(RSS/n) + k·log(n)`` (additive const dropped).
    mdl :
        Two-part description length, in **bits per sample**.
    fingerprint :
        Hash chain head at the moment of selection.
    """

    lam: float
    terms: tuple[Term, ...]
    rss: float
    n: int
    sigma2: float
    r2: float
    aic: float
    bic: float
    mdl: float
    fingerprint: str

    @property
    def complexity(self) -> int:
        return sum(t.complexity for t in self.terms)

    @property
    def k(self) -> int:
        return len(self.terms)

    def predict(self, x: Sequence[float], library: Sequence[Basis]) -> float:
        """Evaluate the law on a single input vector ``x``."""
        s = 0.0
        for t in self.terms:
            s += t.coefficient * library[t.index](x)
        return s

    def __str__(self) -> str:
        if not self.terms:
            return "y ≈ 0"
        parts = [str(self.terms[0])]
        # Drop leading "+" if present
        if parts[0].startswith("+ "):
            parts[0] = parts[0][2:]
        for t in self.terms[1:]:
            parts.append(str(t))
        body = " ".join(parts)
        return f"y ≈ {body}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "lam": self.lam,
            "terms": [
                {"name": t.name, "index": t.index, "coef": t.coefficient}
                for t in self.terms
            ],
            "rss": self.rss,
            "n": self.n,
            "sigma2": self.sigma2,
            "r2": self.r2,
            "aic": self.aic,
            "bic": self.bic,
            "mdl": self.mdl,
            "complexity": self.complexity,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ParetoPoint:
    """One point on the complexity / RSS Pareto frontier."""

    lam: float
    k: int
    complexity: int
    rss: float
    aic: float
    bic: float
    mdl: float
    r2: float
    law: Law


@dataclass(frozen=True)
class Bootstrap:
    """Bootstrap CIs for the coefficients of a given law."""

    law: Law
    n_resamples: int
    alpha: float
    # name -> (lo, hi)
    ci: dict[str, tuple[float, float]]
    # name -> standard error (sample stddev of bootstrap estimates)
    se: dict[str, float]

    def contains_zero(self, name: str) -> bool:
        lo, hi = self.ci[name]
        return lo <= 0.0 <= hi


@dataclass(frozen=True)
class Stability:
    """Stability-selection inclusion frequencies (Meinshausen-Bühlmann 2010)."""

    lam: float
    n_resamples: int
    # name -> π̂_j ∈ [0, 1]
    inclusion: dict[str, float]
    # support of terms with inclusion >= pi_thr (in library order)
    stable_support: tuple[int, ...]
    pi_thr: float

    def stable_names(self, library: Sequence[Basis]) -> tuple[str, ...]:
        return tuple(library[i].name for i in self.stable_support)


@dataclass(frozen=True)
class ScientistReport:
    """Top-level state report for an attestation snapshot."""

    n_observations: int
    input_dim: int
    library_size: int
    n_fits: int
    n_observe_calls: int
    fingerprint: str
    best_law_aic: Law | None
    best_law_bic: Law | None
    best_law_mdl: Law | None
    pareto: tuple[ParetoPoint, ...]
    last_bootstrap: Bootstrap | None
    last_stability: Stability | None
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "input_dim": self.input_dim,
            "library_size": self.library_size,
            "n_fits": self.n_fits,
            "n_observe_calls": self.n_observe_calls,
            "fingerprint": self.fingerprint,
            "best_law_aic": None if self.best_law_aic is None else self.best_law_aic.to_dict(),
            "best_law_bic": None if self.best_law_bic is None else self.best_law_bic.to_dict(),
            "best_law_mdl": None if self.best_law_mdl is None else self.best_law_mdl.to_dict(),
            "pareto_size": len(self.pareto),
            "wall_time_s": self.wall_time_s,
        }


# =====================================================================
# Core fitting helpers
# =====================================================================


def _design_matrix(
    library: Sequence[Basis],
    xs: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Evaluate the basis library on every input row.  Cost ``O(n·p)``."""
    Phi = [[0.0] * len(library) for _ in range(len(xs))]
    for i, x in enumerate(xs):
        for j, b in enumerate(library):
            v = b(x)
            if not math.isfinite(v):
                v = 0.0
            Phi[i][j] = v
    return Phi


def _rss(
    Phi: Sequence[Sequence[float]],
    y: Sequence[float],
    xi: Sequence[float],
    keep: Sequence[int],
) -> float:
    """Residual sum of squares ``Σ_i (y_i − Σ_{j ∈ keep} ξ_j Φ_{ij})²``."""
    s = 0.0
    for i in range(len(y)):
        row = Phi[i]
        pred = 0.0
        for k_idx, j in enumerate(keep):
            pred += xi[k_idx] * row[j]
        d = y[i] - pred
        s += d * d
    return s


def _stlsq_one(
    Phi: Sequence[Sequence[float]],
    y: Sequence[float],
    lam: float,
    *,
    max_iter: int = 20,
    ridge: float = _DEFAULT_RIDGE,
    initial_keep: Sequence[int] | None = None,
) -> tuple[list[int], list[float]]:
    """Sequential Thresholded Least Squares for one ``λ``.

    Returns ``(keep_indices, coefficients_on_keep)``.  The coefficients
    are in *library order*: ``len(coefs) == len(keep_indices)`` and
    ``coefs[k]`` multiplies the basis at library index ``keep_indices[k]``.
    """
    p = len(Phi[0]) if Phi else 0
    if initial_keep is None:
        keep = list(range(p))
    else:
        keep = sorted(set(initial_keep))
    for _ in range(max_iter):
        if not keep:
            return [], []
        sub = [[Phi[i][j] for j in keep] for i in range(len(Phi))]
        xi = _solve_normal_eq(sub, y, ridge=ridge)
        # Threshold
        new_keep: list[int] = []
        for k_idx, j in enumerate(keep):
            if abs(xi[k_idx]) >= lam:
                new_keep.append(j)
        if new_keep == keep:
            return keep, xi
        keep = new_keep
    if not keep:
        return [], []
    sub = [[Phi[i][j] for j in keep] for i in range(len(Phi))]
    xi = _solve_normal_eq(sub, y, ridge=ridge)
    return keep, xi


def _tss(y: Sequence[float]) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    mean = sum(y) / n
    return sum((yi - mean) ** 2 for yi in y)


def _make_law(
    library: Sequence[Basis],
    keep: Sequence[int],
    xi: Sequence[float],
    lam: float,
    rss: float,
    n: int,
    tss: float,
    fingerprint: str,
) -> Law:
    """Materialise a Law dataclass from raw fit outputs.

    AIC/BIC/MDL all use the Gaussian-residual log-likelihood with the
    *MLE* of ``σ²``, and drop additive constants that cancel when laws
    are compared on the same data.
    """
    k = len(keep)
    if n == 0:
        sigma2 = 0.0
    else:
        sigma2 = max(rss / n, _EPS)
    if tss <= 0.0:
        r2 = 1.0 if rss <= _EPS else 0.0
    else:
        r2 = 1.0 - rss / tss
    if r2 < -1e9:  # absurd: probably degenerate data
        r2 = 0.0
    if n == 0 or sigma2 <= 0.0:
        aic = 0.0
        bic = 0.0
        mdl = 0.0
    else:
        ln_sig2 = math.log(sigma2)
        aic = n * ln_sig2 + 2 * k
        bic = n * ln_sig2 + k * (math.log(n) if n > 1 else 0.0)
        # Two-part MDL in bits per sample.
        log2_p = (math.log(max(len(library), 1)) / _LOG2)
        mdl_total = (
            k * log2_p
            + 0.5 * k * (math.log(max(n, 1)) / _LOG2 if n > 1 else 0.0)
            + 0.5 * n * (math.log(2 * math.pi * math.e * sigma2) / _LOG2)
        )
        mdl = mdl_total / max(n, 1)
    terms = tuple(
        Term(
            name=library[j].name,
            index=j,
            coefficient=float(xi[idx]),
            complexity=library[j].complexity,
        )
        for idx, j in enumerate(keep)
    )
    return Law(
        lam=float(lam),
        terms=terms,
        rss=float(rss),
        n=int(n),
        sigma2=float(sigma2),
        r2=float(r2),
        aic=float(aic),
        bic=float(bic),
        mdl=float(mdl),
        fingerprint=fingerprint,
    )


def _akaike_weights(aics: Sequence[float]) -> list[float]:
    """Akaike weights ``w_i = e^{-ΔAIC_i/2} / Σ_j e^{-ΔAIC_j/2}``."""
    if not aics:
        return []
    best = min(aics)
    es = [math.exp(-(a - best) / 2.0) for a in aics]
    s = sum(es)
    if s <= 0.0:
        return [1.0 / len(aics)] * len(aics)
    return [e / s for e in es]


def _pareto_frontier(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return the non-dominated subset, sorted by complexity ascending.

    A point ``(k_a, RSS_a)`` is dominated by ``(k_b, RSS_b)`` iff
    ``k_b <= k_a`` and ``RSS_b < RSS_a``, or ``k_b < k_a`` and
    ``RSS_b <= RSS_a``.  We resolve ties on complexity by keeping
    the smallest RSS for each unique ``k``.
    """
    by_complexity: dict[int, ParetoPoint] = {}
    for p in points:
        cur = by_complexity.get(p.complexity)
        if cur is None or p.rss < cur.rss:
            by_complexity[p.complexity] = p
    ordered = sorted(by_complexity.values(), key=lambda q: q.complexity)
    front: list[ParetoPoint] = []
    best_rss = _INF
    for p in ordered:
        if p.rss < best_rss - 1e-15:
            front.append(p)
            best_rss = p.rss
    return front


def _pareto_knee(front: Sequence[ParetoPoint]) -> ParetoPoint | None:
    """Maximum-distance-from-chord knee point (the elbow rule).

    Given Pareto points ordered by complexity, draw the chord between
    the first and last point and pick the point with the maximum
    perpendicular distance to that chord — a classical elbow heuristic
    (Satopää et al. 2011 *Finding a Kneedle in a Haystack*).  Returns
    ``None`` if the frontier has fewer than 3 points (no interior).
    """
    n = len(front)
    if n == 0:
        return None
    if n == 1 or n == 2:
        return front[0]  # simplest law on the frontier
    x0, y0 = front[0].complexity, math.log1p(front[0].rss)
    xn, yn = front[-1].complexity, math.log1p(front[-1].rss)
    dx, dy = xn - x0, yn - y0
    norm = math.hypot(dx, dy)
    if norm <= 0.0:
        return front[0]
    best_d = -_INF
    best_p = front[0]
    for p in front[1:-1]:
        x, y = p.complexity, math.log1p(p.rss)
        # Perpendicular distance to chord
        d = abs(dy * (x - x0) - dx * (y - y0)) / norm
        if d > best_d:
            best_d = d
            best_p = p
    return best_p


# =====================================================================
# Scientist class — the runtime primitive
# =====================================================================


class Scientist:
    """Sparse symbolic law discovery as a runtime primitive.

    Stream observations in with :meth:`observe`, call :meth:`fit` to
    discover laws across a ``λ`` grid, inspect :meth:`pareto`, and
    request :meth:`bootstrap` or :meth:`stability_selection` for
    coefficient confidence intervals or support stability.  All state
    is in-memory; every state-mutating call advances a SHA-256 hash
    chain and (optionally) publishes a typed event on an
    :class:`EventBus`.

    Thread-safe via a re-entrant lock.  Deterministic given ``seed``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        library: Sequence[Basis],
        lambda_grid: Sequence[float],
        seed: int,
        bus: EventBus | None,
        ridge: float,
        stlsq_max_iter: int,
        max_observations: int,
    ) -> None:
        self._input_dim = int(input_dim)
        self._library: tuple[Basis, ...] = tuple(library)
        self._lambda_grid: tuple[float, ...] = tuple(sorted(lambda_grid))
        self._seed = int(seed)
        self._bus = bus
        self._ridge = float(ridge)
        self._stlsq_max_iter = int(stlsq_max_iter)
        self._max_observations = int(max_observations)
        self._t_started = time.monotonic()

        # Streaming data buffers.
        self._xs: list[tuple[float, ...]] = []
        self._ys: list[float] = []

        # Cached fit state.
        self._pareto_cache: tuple[ParetoPoint, ...] | None = None
        self._best_aic: Law | None = None
        self._best_bic: Law | None = None
        self._best_mdl: Law | None = None
        self._last_bootstrap: Bootstrap | None = None
        self._last_stability: Stability | None = None

        # Counters
        self._n_fits = 0
        self._n_observe_calls = 0
        self._n_predict_calls = 0

        # Audit chain.
        self._fingerprint = _GENESIS
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        input_dim: int,
        library: Sequence[Basis] | None = None,
        max_degree: int = 2,
        include_trig: bool = False,
        trig_frequencies: Sequence[float] = (1.0,),
        include_exp: bool = False,
        include_log: bool = False,
        include_inv: bool = False,
        extra_basis: Sequence[Basis] = (),
        lambda_grid: Sequence[float] | None = None,
        seed: int = 0,
        bus: EventBus | None = None,
        ridge: float = _DEFAULT_RIDGE,
        stlsq_max_iter: int = 20,
        max_observations: int = 1_000_000,
        session_id: str | None = None,
    ) -> "Scientist":
        """Construct a :class:`Scientist`.

        Either supply a fully-formed ``library`` or use the convenience
        knobs (``max_degree``, ``include_trig``, …) that build the
        default monomial-and-friends library via
        :func:`default_library`.
        """
        if input_dim < 0:
            raise InvalidConfig("input_dim must be >= 0")
        if max_degree < 0:
            raise InvalidConfig("max_degree must be >= 0")
        if max_degree > 8:
            raise InvalidConfig("max_degree must be <= 8 (combinatorial blowup)")
        if ridge <= 0.0:
            raise InvalidConfig("ridge must be > 0")
        if stlsq_max_iter <= 0:
            raise InvalidConfig("stlsq_max_iter must be > 0")
        if max_observations <= 0:
            raise InvalidConfig("max_observations must be > 0")
        if library is None:
            library = default_library(
                input_dim,
                max_degree=max_degree,
                include_trig=include_trig,
                trig_frequencies=trig_frequencies,
                include_exp=include_exp,
                include_log=include_log,
                include_inv=include_inv,
                extra=extra_basis,
            )
        else:
            if not library:
                raise InvalidConfig("library must be non-empty")
            for b in library:
                if not isinstance(b, Basis):
                    raise InvalidBasis(f"library entry {b!r} is not a Basis")
        if lambda_grid is None:
            lambda_grid = _DEFAULT_LAMBDA_GRID
        if not lambda_grid:
            raise InvalidConfig("lambda_grid must be non-empty")
        for lam in lambda_grid:
            if not (math.isfinite(lam) and lam > 0.0):
                raise InvalidConfig("lambda_grid entries must be finite and > 0")

        sci = cls(
            input_dim=input_dim,
            library=library,
            lambda_grid=lambda_grid,
            seed=seed,
            bus=bus,
            ridge=ridge,
            stlsq_max_iter=stlsq_max_iter,
            max_observations=max_observations,
        )
        payload = {
            "event": "started",
            "input_dim": input_dim,
            "library_size": len(sci._library),
            "library_hash": _library_hash(sci._library),
            "lambda_grid": list(sci._lambda_grid),
            "seed": int(seed),
            "ridge": float(ridge),
            "stlsq_max_iter": int(stlsq_max_iter),
        }
        sci._fingerprint = _hash_link(_GENESIS, _payload_repr(payload))
        if bus is not None:
            bus.publish(
                Event(
                    kind=SCIENTIST_STARTED,
                    session_id=session_id,
                    data={**payload, "fingerprint": sci._fingerprint},
                )
            )
        return sci

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def library(self) -> tuple[Basis, ...]:
        return self._library

    @property
    def library_size(self) -> int:
        return len(self._library)

    @property
    def lambda_grid(self) -> tuple[float, ...]:
        return self._lambda_grid

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def n_observations(self) -> int:
        with self._lock:
            return len(self._ys)

    def fingerprint(self) -> str:
        with self._lock:
            return self._fingerprint

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------

    def observe(
        self,
        x: Sequence[float],
        y: float,
        *,
        session_id: str | None = None,
    ) -> str:
        """Append one (``x``, ``y``) pair to the buffer.

        Invalidates any cached fit (next call to :meth:`fit` will refit
        from scratch).  Returns the new fingerprint.

        Raises :class:`InvalidObservation` if ``len(x) != input_dim``
        or any value is non-finite.
        """
        if not isinstance(x, (list, tuple)):
            try:
                x = list(x)
            except TypeError as exc:
                raise InvalidObservation(f"x must be a sequence of floats: {exc}")
        if len(x) != self._input_dim:
            raise InvalidObservation(
                f"x has length {len(x)} but input_dim is {self._input_dim}"
            )
        xv: list[float] = []
        for v in x:
            fv = float(v)
            if not math.isfinite(fv):
                raise InvalidObservation(f"x contains non-finite value: {v!r}")
            xv.append(fv)
        yv = float(y)
        if not math.isfinite(yv):
            raise InvalidObservation(f"y is non-finite: {y!r}")

        with self._lock:
            if len(self._ys) >= self._max_observations:
                raise InvalidObservation(
                    f"max_observations ({self._max_observations}) reached"
                )
            self._xs.append(tuple(xv))
            self._ys.append(yv)
            self._n_observe_calls += 1
            self._pareto_cache = None
            self._best_aic = None
            self._best_bic = None
            self._best_mdl = None
            payload = {
                "event": "observed",
                "n": len(self._ys),
                "x": xv,
                "y": yv,
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=SCIENTIST_OBSERVED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return fp

    def observe_many(
        self,
        xs: Sequence[Sequence[float]],
        ys: Sequence[float],
        *,
        session_id: str | None = None,
    ) -> str:
        """Bulk ingestion. Returns final fingerprint."""
        if len(xs) != len(ys):
            raise InvalidObservation("xs and ys must have the same length")
        fp = self._fingerprint
        for x, y in zip(xs, ys):
            fp = self.observe(x, y, session_id=session_id)
        return fp

    def clear(self, *, session_id: str | None = None) -> str:
        """Drop all observations and cached fits."""
        with self._lock:
            self._xs.clear()
            self._ys.clear()
            self._pareto_cache = None
            self._best_aic = None
            self._best_bic = None
            self._best_mdl = None
            self._last_bootstrap = None
            self._last_stability = None
            payload = {"event": "cleared"}
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=SCIENTIST_CLEARED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return fp

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        *,
        criterion: str = SELECT_AIC,
        session_id: str | None = None,
    ) -> Law:
        """Run STLSQ across the ``λ`` grid and select the best law.

        ``criterion`` ∈ {AIC, BIC, MDL, PARETO_KNEE}.  Caches the full
        Pareto frontier on the first call; subsequent calls (with the
        same observations) only re-select.  Raises
        :class:`InsufficientData` if fewer than 2 observations have
        been ingested.
        """
        if criterion not in SCIENTIST_KNOWN_CRITERIA:
            raise InvalidCriterion(
                f"criterion must be one of {sorted(SCIENTIST_KNOWN_CRITERIA)}"
            )
        with self._lock:
            n = len(self._ys)
            if n < 2:
                raise InsufficientData(
                    f"fit requires at least 2 observations (have {n})"
                )
            self._refit_pareto_locked()
            front = self._pareto_cache or ()
            if not front:
                raise InsufficientData("Pareto frontier is empty after fit")
            best: ParetoPoint
            if criterion == SELECT_AIC:
                best = min(front, key=lambda p: p.aic)
                self._best_aic = best.law
            elif criterion == SELECT_BIC:
                best = min(front, key=lambda p: p.bic)
                self._best_bic = best.law
            elif criterion == SELECT_MDL:
                best = min(front, key=lambda p: p.mdl)
                self._best_mdl = best.law
            elif criterion == SELECT_PARETO_KNEE:
                knee = _pareto_knee(front)
                assert knee is not None  # front non-empty
                best = knee
            else:
                raise InvalidCriterion(criterion)
            self._n_fits += 1
            payload = {
                "event": "fitted",
                "criterion": criterion,
                "n": n,
                "k": best.k,
                "rss": best.rss,
                "r2": best.r2,
                "aic": best.aic,
                "bic": best.bic,
                "mdl": best.mdl,
                "lam": best.lam,
                "law_terms": [(t.name, t.coefficient) for t in best.law.terms],
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            law = Law(
                lam=best.law.lam,
                terms=best.law.terms,
                rss=best.law.rss,
                n=best.law.n,
                sigma2=best.law.sigma2,
                r2=best.law.r2,
                aic=best.law.aic,
                bic=best.law.bic,
                mdl=best.law.mdl,
                fingerprint=self._fingerprint,
            )
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=SCIENTIST_FITTED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return law

    def _refit_pareto_locked(self) -> None:
        """Re-run STLSQ across the ``λ`` grid; cache the Pareto frontier."""
        if self._pareto_cache is not None:
            return
        n = len(self._ys)
        if n < 2:
            self._pareto_cache = ()
            return
        xs = self._xs
        ys = self._ys
        Phi = _design_matrix(self._library, xs)
        tss = _tss(ys)
        seen_supports: dict[tuple[int, ...], ParetoPoint] = {}
        for lam in self._lambda_grid:
            keep, xi = _stlsq_one(
                Phi,
                ys,
                lam,
                max_iter=self._stlsq_max_iter,
                ridge=self._ridge,
            )
            support = tuple(keep)
            if support in seen_supports:
                # Identical support → identical fit; keep the smaller λ
                # so the *least aggressive* threshold owns this support.
                existing = seen_supports[support]
                if lam < existing.lam:
                    seen_supports[support] = ParetoPoint(
                        lam=float(lam),
                        k=existing.k,
                        complexity=existing.complexity,
                        rss=existing.rss,
                        aic=existing.aic,
                        bic=existing.bic,
                        mdl=existing.mdl,
                        r2=existing.r2,
                        law=existing.law,
                    )
                continue
            rss = _rss(Phi, ys, xi, keep)
            law = _make_law(
                self._library, keep, xi, lam, rss, n, tss, self._fingerprint
            )
            seen_supports[support] = ParetoPoint(
                lam=float(lam),
                k=law.k,
                complexity=law.complexity,
                rss=law.rss,
                aic=law.aic,
                bic=law.bic,
                mdl=law.mdl,
                r2=law.r2,
                law=law,
            )
        all_points = list(seen_supports.values())
        front = tuple(_pareto_frontier(all_points))
        self._pareto_cache = front

    def pareto(
        self,
        *,
        session_id: str | None = None,
    ) -> tuple[ParetoPoint, ...]:
        """Return the Pareto frontier across the ``λ`` grid.

        Caches across calls; cleared by :meth:`observe` and
        :meth:`clear`.  Raises :class:`InsufficientData` if fewer than
        2 observations exist.
        """
        with self._lock:
            n = len(self._ys)
            if n < 2:
                raise InsufficientData("pareto requires at least 2 observations")
            self._refit_pareto_locked()
            front = self._pareto_cache or ()
            payload = {
                "event": "pareto",
                "n_points": len(front),
                "min_k": (min((p.k for p in front), default=0)),
                "max_k": (max((p.k for p in front), default=0)),
                "min_rss": (min((p.rss for p in front), default=0.0)),
                "max_rss": (max((p.rss for p in front), default=0.0)),
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=SCIENTIST_PARETO,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return front

    def akaike_weights(self) -> dict[float, float]:
        """Akaike weight per ``λ`` on the Pareto frontier.

        Returns ``{lam: w}`` where weights sum to 1.  Empty if no fit.
        """
        with self._lock:
            self._refit_pareto_locked()
            front = self._pareto_cache or ()
            if not front:
                return {}
            ws = _akaike_weights([p.aic for p in front])
            return {p.lam: w for p, w in zip(front, ws)}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        x: Sequence[float],
        *,
        law: Law | None = None,
        criterion: str = SELECT_AIC,
        session_id: str | None = None,
    ) -> float:
        """Predict ``y`` at ``x`` using ``law`` or the best-by-``criterion``."""
        if law is None:
            law = self.fit(criterion=criterion, session_id=session_id)
        if len(x) != self._input_dim:
            raise InvalidObservation(
                f"x has length {len(x)} but input_dim is {self._input_dim}"
            )
        for v in x:
            if not math.isfinite(float(v)):
                raise InvalidObservation(f"x contains non-finite value: {v!r}")
        y_hat = law.predict(list(x), self._library)
        with self._lock:
            self._n_predict_calls += 1
            payload = {
                "event": "predicted",
                "x": list(x),
                "y_hat": float(y_hat),
                "law_fp": law.fingerprint,
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=SCIENTIST_PREDICTED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return float(y_hat)

    def evaluate_r2(self, xs: Sequence[Sequence[float]], ys: Sequence[float], *, law: Law | None = None) -> float:
        """Out-of-sample R² on a held-out set."""
        if law is None:
            law = self.fit()
        if len(xs) != len(ys):
            raise InvalidObservation("xs and ys must have the same length")
        if not ys:
            return 1.0
        ss_res = 0.0
        mean_y = sum(ys) / len(ys)
        ss_tot = sum((yi - mean_y) ** 2 for yi in ys)
        for x, y in zip(xs, ys):
            r = y - law.predict(list(x), self._library)
            ss_res += r * r
        if ss_tot <= 0.0:
            return 1.0 if ss_res <= _EPS else 0.0
        return 1.0 - ss_res / ss_tot

    # ------------------------------------------------------------------
    # Bootstrap and stability selection
    # ------------------------------------------------------------------

    def bootstrap(
        self,
        *,
        law: Law | None = None,
        n_resamples: int = _DEFAULT_BOOTSTRAP_SAMPLES,
        alpha: float = 0.05,
        session_id: str | None = None,
    ) -> Bootstrap:
        """Per-coefficient percentile bootstrap CIs at the law's support.

        Holds the support fixed and refits OLS on resamples.  ``alpha``
        is the two-sided significance level; default 0.05 → 95 % CI.
        """
        if not (0.0 < alpha < 1.0):
            raise InvalidConfig("alpha must be in (0, 1)")
        if n_resamples < 5:
            raise InvalidConfig("n_resamples must be >= 5")
        with self._lock:
            if law is None:
                law = self.fit(session_id=session_id)
            n = len(self._ys)
            if n < 2:
                raise InsufficientData("bootstrap requires at least 2 observations")
            xs = list(self._xs)
            ys = list(self._ys)
            keep = [t.index for t in law.terms]
            names = [t.name for t in law.terms]
            if not keep:
                empty = Bootstrap(
                    law=law,
                    n_resamples=int(n_resamples),
                    alpha=float(alpha),
                    ci={},
                    se={},
                )
                self._last_bootstrap = empty
                return empty
            rng = random.Random(self._seed ^ 0xB007517A47)
            Phi_full = _design_matrix(self._library, xs)
            estimates: list[list[float]] = [[] for _ in keep]
            for _ in range(n_resamples):
                idx = [rng.randrange(n) for _ in range(n)]
                sub_Phi = [[Phi_full[i][j] for j in keep] for i in idx]
                sub_y = [ys[i] for i in idx]
                try:
                    xi = _solve_normal_eq(sub_Phi, sub_y, ridge=self._ridge)
                except ScientistError:
                    continue
                for k, c in enumerate(xi):
                    estimates[k].append(c)
            ci: dict[str, tuple[float, float]] = {}
            se: dict[str, float] = {}
            lo_q = alpha / 2.0
            hi_q = 1.0 - alpha / 2.0
            for name, ests in zip(names, estimates):
                if not ests:
                    ci[name] = (float("nan"), float("nan"))
                    se[name] = float("nan")
                    continue
                ests_sorted = sorted(ests)
                m = len(ests_sorted)
                lo_idx = max(0, min(m - 1, int(math.floor(lo_q * (m - 1)))))
                hi_idx = max(0, min(m - 1, int(math.ceil(hi_q * (m - 1)))))
                ci[name] = (ests_sorted[lo_idx], ests_sorted[hi_idx])
                mean = sum(ests) / m
                var = sum((e - mean) ** 2 for e in ests) / max(m - 1, 1)
                se[name] = math.sqrt(max(var, 0.0))
            boot = Bootstrap(
                law=law,
                n_resamples=int(n_resamples),
                alpha=float(alpha),
                ci=ci,
                se=se,
            )
            self._last_bootstrap = boot
            payload = {
                "event": "bootstrapped",
                "n_resamples": int(n_resamples),
                "alpha": float(alpha),
                "k": len(keep),
                "ci": {k: list(v) for k, v in ci.items()},
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=SCIENTIST_BOOTSTRAPPED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return boot

    def stability_selection(
        self,
        *,
        lam: float | None = None,
        n_resamples: int = 100,
        subsample_fraction: float = 0.5,
        pi_thr: float = 0.6,
        session_id: str | None = None,
    ) -> Stability:
        """Meinshausen-Bühlmann stability selection.

        Draw subsamples of size ⌊n·``subsample_fraction``⌋ without
        replacement, re-run STLSQ at ``lam`` (default: the smallest λ
        in the grid → maximally generous support), and count the
        fraction of subsamples in which each basis is retained.
        Returns the *stable support* — basis indices whose inclusion
        frequency is ≥ ``pi_thr``.
        """
        if not (0.0 < subsample_fraction <= 1.0):
            raise InvalidConfig("subsample_fraction must be in (0, 1]")
        if not (0.0 <= pi_thr <= 1.0):
            raise InvalidConfig("pi_thr must be in [0, 1]")
        if n_resamples < 5:
            raise InvalidConfig("n_resamples must be >= 5")
        with self._lock:
            n = len(self._ys)
            if n < 4:
                raise InsufficientData(
                    "stability_selection requires at least 4 observations"
                )
            if lam is None:
                lam_ = self._lambda_grid[0]
            else:
                if not (math.isfinite(lam) and lam > 0.0):
                    raise InvalidConfig("lam must be finite and > 0")
                lam_ = float(lam)
            xs = list(self._xs)
            ys = list(self._ys)
            Phi_full = _design_matrix(self._library, xs)
            sub_n = max(2, int(math.floor(n * subsample_fraction)))
            rng = random.Random(self._seed ^ 0x57AB14554)
            p = len(self._library)
            counts = [0] * p
            for _ in range(n_resamples):
                idx = rng.sample(range(n), sub_n)
                sub_Phi = [Phi_full[i] for i in idx]
                sub_y = [ys[i] for i in idx]
                keep, _xi = _stlsq_one(
                    sub_Phi,
                    sub_y,
                    lam_,
                    max_iter=self._stlsq_max_iter,
                    ridge=self._ridge,
                )
                for j in keep:
                    counts[j] += 1
            inclusion = {
                self._library[j].name: counts[j] / n_resamples for j in range(p)
            }
            stable = tuple(
                j for j in range(p) if counts[j] / n_resamples >= pi_thr
            )
            stab = Stability(
                lam=lam_,
                n_resamples=int(n_resamples),
                inclusion=inclusion,
                stable_support=stable,
                pi_thr=float(pi_thr),
            )
            self._last_stability = stab
            payload = {
                "event": "stability",
                "lam": lam_,
                "n_resamples": int(n_resamples),
                "subsample_fraction": float(subsample_fraction),
                "pi_thr": float(pi_thr),
                "stable_support": list(stable),
                "inclusion": inclusion,
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=SCIENTIST_STABILITY,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return stab

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, *, session_id: str | None = None) -> ScientistReport:
        """Top-level snapshot for attestation."""
        with self._lock:
            n = len(self._ys)
            front: tuple[ParetoPoint, ...]
            if n >= 2:
                self._refit_pareto_locked()
                front = self._pareto_cache or ()
            else:
                front = ()
            best_aic = (
                min(front, key=lambda p: p.aic).law if front else None
            )
            best_bic = (
                min(front, key=lambda p: p.bic).law if front else None
            )
            best_mdl = (
                min(front, key=lambda p: p.mdl).law if front else None
            )
            rep = ScientistReport(
                n_observations=n,
                input_dim=self._input_dim,
                library_size=len(self._library),
                n_fits=self._n_fits,
                n_observe_calls=self._n_observe_calls,
                fingerprint=self._fingerprint,
                best_law_aic=best_aic,
                best_law_bic=best_bic,
                best_law_mdl=best_mdl,
                pareto=front,
                last_bootstrap=self._last_bootstrap,
                last_stability=self._last_stability,
                wall_time_s=time.monotonic() - self._t_started,
            )
            payload = {
                "event": "reported",
                "n": n,
                "n_fits": self._n_fits,
                "pareto_size": len(front),
                "best_aic_terms": (
                    [(t.name, t.coefficient) for t in best_aic.terms]
                    if best_aic is not None
                    else []
                ),
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
            rep = ScientistReport(
                n_observations=rep.n_observations,
                input_dim=rep.input_dim,
                library_size=rep.library_size,
                n_fits=rep.n_fits,
                n_observe_calls=rep.n_observe_calls,
                fingerprint=fp,
                best_law_aic=rep.best_law_aic,
                best_law_bic=rep.best_law_bic,
                best_law_mdl=rep.best_law_mdl,
                pareto=rep.pareto,
                last_bootstrap=rep.last_bootstrap,
                last_stability=rep.last_stability,
                wall_time_s=rep.wall_time_s,
            )
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=SCIENTIST_REPORTED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return rep

    # ------------------------------------------------------------------
    # Certificates — finite-sample bounds the coordinator can route on
    # ------------------------------------------------------------------

    def aicc_correction(self, law: Law) -> float:
        """Akaike's small-sample correction ``AICc = AIC + 2k(k+1)/(n-k-1)``.

        For ``n - k - 1 ≤ 0`` returns ``+∞`` (the law cannot be
        validated on so few observations).
        """
        n = law.n
        k = law.k
        if n - k - 1 <= 0:
            return _INF
        return law.aic + 2.0 * k * (k + 1) / (n - k - 1)

    def mdl_certificate(self, law: Law) -> dict[str, float]:
        """Return the components of the two-part MDL code length.

        ``model_bits`` is the cost of describing the support and the
        quantised coefficients; ``data_bits`` is the cost of encoding
        the residuals under a Gaussian noise model with the MLE
        variance.  Total ``= model_bits + data_bits``, in bits per
        sample.  An empty law has ``model_bits = 0`` and ``data_bits =
        0.5 log₂(2πeσ²)``.
        """
        n = max(law.n, 1)
        k = law.k
        log2_p = math.log(max(len(self._library), 1)) / _LOG2
        model_bits = (k * log2_p + 0.5 * k * (math.log(n) / _LOG2 if n > 1 else 0.0)) / n
        if law.sigma2 <= 0.0:
            data_bits = 0.0
        else:
            data_bits = 0.5 * math.log(2 * math.pi * math.e * law.sigma2) / _LOG2
        return {
            "model_bits_per_sample": model_bits,
            "data_bits_per_sample": data_bits,
            "total_bits_per_sample": model_bits + data_bits,
            "library_log2_p": log2_p,
        }


# ---------------------------------------------------------------------
# Internal: deterministic hash of a basis library
# ---------------------------------------------------------------------


def _library_hash(library: Sequence[Basis]) -> str:
    """SHA-256 of (name, complexity) tuples in library order.

    The fingerprint of a Scientist depends on the *names* of its basis
    functions (which are deterministic strings) and their complexities,
    not on the function objects themselves (Python lambdas are not
    hashable in a stable way across runs).
    """
    h = hashlib.sha256()
    for b in library:
        h.update(b.name.encode("utf-8"))
        h.update(b"\x1e")
        h.update(struct.pack("<i", int(b.complexity)))
        h.update(b"\x1d")
    return h.hexdigest()


__all__ = [
    "Basis",
    "Bootstrap",
    "InsufficientData",
    "InvalidBasis",
    "InvalidConfig",
    "InvalidCriterion",
    "InvalidObservation",
    "Law",
    "NotYetFitted",
    "ParetoPoint",
    "SCIENTIST_BOOTSTRAPPED",
    "SCIENTIST_CLEARED",
    "SCIENTIST_FITTED",
    "SCIENTIST_KNOWN_CRITERIA",
    "SCIENTIST_KNOWN_EVENTS",
    "SCIENTIST_OBSERVED",
    "SCIENTIST_PARETO",
    "SCIENTIST_PREDICTED",
    "SCIENTIST_REPORTED",
    "SCIENTIST_STABILITY",
    "SCIENTIST_STARTED",
    "SELECT_AIC",
    "SELECT_BIC",
    "SELECT_MDL",
    "SELECT_PARETO_KNEE",
    "Scientist",
    "ScientistError",
    "ScientistReport",
    "Stability",
    "Term",
    "default_library",
]
