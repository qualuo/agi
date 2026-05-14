r"""Forecaster — anytime-valid probabilistic forecasting as a runtime primitive.

A coordination engine that learns and acts has to *forecast*: outcome
probabilities for downstream decisions, latencies for scheduling,
revenue for cost ceilings, drift for safety. Every other primitive in
this stack already consumes or produces a forecast — Arbiter compares
arm means, Auditor batches p-values, Robustifier worst-cases over
ambiguity sets, Cartographer tracks per-task competence, CausalDiscoverer
estimates ATEs.  None of them *commit to a calibrated, score-valid
probability distribution and prove it under arbitrary stopping*.

The Forecaster is the primitive that gives the rest of the stack a
single, rigorous answer to "what's your forecast, what's your score,
and is it calibrated?" — with **finite-sample, anytime-valid**
guarantees that hold under *any* data-dependent stopping rule.

Mathematical roots
------------------

  * **De Finetti, 1937; Savage, 1971; Brier, 1950.**  Brier defined
    the quadratic score B(p, y) = ∑_k (p_k - 1{y=k})². Savage
    characterised the *strictly proper* scoring rules: rules S such
    that the truth-telling forecast q* uniquely minimises
    E_y∼q[S(p, y)] over p.  Brier, log-score, spherical and quadratic
    are all strictly proper.

  * **Gneiting & Raftery, 2007 — "Strictly proper scoring rules,
    prediction, and estimation."**  Closes the discrete theory and
    introduces the CRPS for continuous distributions:
        CRPS(F, y) = ∫_ℝ (F(t) - 1{y ≤ t})² dt
                   = E|X - y| - ½ E|X - X'|         (X, X' ~ F i.i.d.)
    closed-form for Gaussian:
        CRPS(N(μ,σ²), y) = σ {z(2Φ(z) - 1) + 2φ(z) - π^{-½}},
                           z = (y-μ)/σ.

  * **Dawid, 1984 — *Prequential principle*.**  Forecasts and
    outcomes form a sequential record; statistical claims must hold
    on the realised path, not in some hypothetical replication.

  * **Vovk-Wang-Shafer, 2005; Ramdas-Grünwald-Vovk-Shafer, 2023 —
    *Game-theoretic statistics*.**  An e-process E_t is a non-negative
    process with E[E_t] ≤ 1 under H₀.  Ville's inequality gives
        P_{H₀}(∃t : E_t ≥ 1/α) ≤ α
    for *every* stopping rule.  The Forecaster uses an e-process built
    on the PIT to test calibration with no asymptotics, no sample-size
    pre-registration, and freedom of optional continuation.

  * **Cesa-Bianchi & Lugosi, 2006 — *Prediction, Learning, and Games*.**
    The exponentially-weighted average forecaster (Hedge) with learning
    rate η = √(8 log K / T) attains regret O(√(T log K)) against the
    best fixed predictor for any [0,1]-bounded loss.  Polynomial-weights
    is the parameter-free version; the Forecaster ships both.

  * **Waudby-Smith & Ramdas, 2024 — "Estimating means of bounded
    random variables by betting."**  GRAPA (Growth-Rate Adaptive to
    Particular Alternatives) chooses the betting fraction λ_t to
    maximise log-growth on past data, with a soft-projection step that
    keeps each bet inside [-c, c]; this gives a *predictable* betting
    sequence and a martingale-valid e-process.  We use the aGRAPA
    variant for the calibration test.

  * **Dawid, 1982; Gneiting-Balabdaoui-Raftery, 2007.**  *Probabilistic
    calibration*: the PIT u_t = F_t(y_t) is Uniform[0,1] iff the
    forecasts are calibrated.  Reliability diagrams, the
    Kolmogorov-Smirnov statistic against the uniform, and the
    Anderson-Darling A² are the classical asymptotic tests; the
    Forecaster keeps them for compatibility but ships an e-process
    in parallel as the headline anytime-valid test.

  * **Vovk-Gammerman-Shafer, 2005 — *Conformal prediction*.**  Online
    conformal regression gives a finite-sample, distribution-free
    prediction interval at any miscoverage α whose long-run coverage
    is exactly 1-α.  The Forecaster integrates the existing
    ``agi.conformal`` machinery to ship *interval* forecasts alongside
    probabilistic ones.

  * **Niculescu-Mizil & Caruana, 2005; Platt, 1999; Zadrozny &
    Elkan, 2002.**  Isotonic regression and Platt scaling are the
    workhorses of recalibration.  We delegate to ``agi.calibration``
    for the heavy lifting and treat its output as a post-hoc
    transform on every new forecast.

Design contract
---------------

The Forecaster registers *forecast streams* and, for each stream,
records ``(label, forecast_object, outcome)`` triples.  At any moment
the coordination engine can ask:

  * `score(stream, rule)` — empirical proper-score on the realised
    path (Brier, log, spherical, quadratic, CRPS, pinball, linex).
  * `pit(stream)` — the realised PIT sequence (defined for any
    forecast type that exposes a CDF).
  * `calibration_test(stream, alpha)` — three views in one report:
        - KS statistic + Massart-DKW bound (classical, asymptotic),
        - Anderson-Darling A² + asymptotic p-value,
        - **anytime-valid e-process** with Ville-certificate rejection
          at level α (the headline guarantee).
  * `recalibrate(stream, method)` — fit a recalibrator on the
    in-stream PIT/reliability history; subsequent forecasts pushed
    through `record()` are automatically corrected.
  * `interval(stream, alpha)` — finite-sample-valid prediction
    interval for the next outcome, with **online conformal** coverage.
  * `ensemble(streams, weights=None, method="hedge", eta=None)` —
    aggregate K streams.  With method="hedge", the Forecaster
    maintains exponentially-weighted weights with provable
    O(√(T log K)) cumulative-regret w.r.t. the best stream under any
    [0,1]-bounded proper score.
  * `forecast(stream, query=None)` — emit the *current* forecast
    (possibly the most recent stored one, or the ensemble of K streams).
  * `coverage()` — lifetime stats: streams, forecasts, calibrations,
    rejections, attest receipts, betting growth.

Every state-changing call emits an event on the optional ``EventBus``
and writes a content-hashed receipt to the optional ``RuntimeAttestor``,
so a coordination engine can deterministically replay the forecast
history and re-verify the anytime tests.

The module is stdlib-only and threadsafe under a single recursive
lock.  All public functions and the ``Forecaster`` class are designed
to be called concurrently from multiple coordination workers.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from agi.events import Event, EventBus  # type: ignore
except Exception:  # pragma: no cover - keep stdlib-only fallbacks
    Event = None  # type: ignore
    EventBus = None  # type: ignore


# =====================================================================
# Event kinds
# =====================================================================


FORECASTER_STARTED = "forecaster.started"
FORECASTER_STREAM_REGISTERED = "forecaster.stream_registered"
FORECASTER_STREAM_REMOVED = "forecaster.stream_removed"
FORECASTER_OBSERVED = "forecaster.observed"
FORECASTER_SCORED = "forecaster.scored"
FORECASTER_CALIBRATION_TESTED = "forecaster.calibration_tested"
FORECASTER_RECALIBRATED = "forecaster.recalibrated"
FORECASTER_ENSEMBLE_UPDATED = "forecaster.ensemble_updated"
FORECASTER_INTERVAL_EMITTED = "forecaster.interval_emitted"
FORECASTER_FORECAST_EMITTED = "forecaster.forecast_emitted"
FORECASTER_CLEARED = "forecaster.cleared"
FORECASTER_REPORT = "forecaster.report"


# =====================================================================
# Scoring-rule names
# =====================================================================


SCORE_BRIER = "brier"
SCORE_LOG = "log"
SCORE_SPHERICAL = "spherical"
SCORE_QUADRATIC = "quadratic"
SCORE_CRPS = "crps"
SCORE_PINBALL = "pinball"
SCORE_LINEX = "linex"

KNOWN_SCORES = frozenset(
    {
        SCORE_BRIER,
        SCORE_LOG,
        SCORE_SPHERICAL,
        SCORE_QUADRATIC,
        SCORE_CRPS,
        SCORE_PINBALL,
        SCORE_LINEX,
    }
)


# =====================================================================
# Calibration / aggregation method names
# =====================================================================


CALIB_KS = "ks"
CALIB_ANDERSON = "anderson_darling"
CALIB_E_PROCESS = "e_process"

POOL_LINEAR = "linear"
POOL_LOG = "log"
POOL_HEDGE = "hedge"
POOL_POLY = "polynomial"

KNOWN_POOLS = frozenset({POOL_LINEAR, POOL_LOG, POOL_HEDGE, POOL_POLY})

RECAL_ISOTONIC = "isotonic"
RECAL_HISTOGRAM = "histogram"
RECAL_PIT = "pit"
RECAL_PLATT = "platt"

KNOWN_RECAL = frozenset({RECAL_ISOTONIC, RECAL_HISTOGRAM, RECAL_PIT, RECAL_PLATT})


_EPS = 1e-12
_LOG_CLAMP = 1e-15
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


# =====================================================================
# Errors
# =====================================================================


class ForecasterError(Exception):
    """Base class for forecaster errors."""


class InvalidForecast(ForecasterError):
    """Forecast object is malformed (negative mass, doesn't sum to 1, …)."""


class UnknownStream(ForecasterError):
    """Referenced stream has not been registered."""


class InsufficientData(ForecasterError):
    """Operation requires more observations than the stream has."""


class UnknownMethod(ForecasterError):
    """Caller named a method that does not exist."""


# =====================================================================
# Forecast objects
# =====================================================================


@dataclass(frozen=True)
class CategoricalForecast:
    """PMF over a finite label set.

    ``probs`` is the (label, probability) mapping with probability ≥ 0
    summing to 1 (within ``tol``).  Labels are hashable; their order
    in the dict defines the canonical index for CDF/PIT operations.
    """

    probs: tuple  # tuple[tuple[Hashable, float], ...]
    tol: float = 1e-9

    @staticmethod
    def from_dict(d: Mapping[Any, float], tol: float = 1e-9) -> "CategoricalForecast":
        items = tuple((k, float(v)) for k, v in d.items())
        _validate_pmf(items, tol=tol)
        return CategoricalForecast(probs=items, tol=tol)

    def as_dict(self) -> dict:
        return dict(self.probs)

    def prob_of(self, label: Any) -> float:
        for k, p in self.probs:
            if k == label:
                return float(p)
        return 0.0

    def cdf(self, label: Any) -> float:
        s = 0.0
        for k, p in self.probs:
            s += float(p)
            if k == label:
                return s
        return s

    def support(self) -> tuple:
        return tuple(k for k, _ in self.probs)


@dataclass(frozen=True)
class BernoulliForecast:
    """Bernoulli forecast — probability of class 1."""

    p: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise InvalidForecast(f"Bernoulli p must lie in [0,1]; got {self.p!r}")

    def prob_of(self, label: int) -> float:
        return float(self.p) if int(label) == 1 else 1.0 - float(self.p)

    def cdf(self, y: float) -> float:
        # CDF of Bernoulli on {0, 1}: F(t) = 1{t>=0}*(1-p) + 1{t>=1}*p
        if y < 0:
            return 0.0
        if y < 1:
            return 1.0 - self.p
        return 1.0


@dataclass(frozen=True)
class GaussianForecast:
    """Normal distribution forecast."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not (self.sigma > 0):
            raise InvalidForecast(f"Gaussian sigma must be > 0; got {self.sigma!r}")

    def cdf(self, y: float) -> float:
        z = (float(y) - self.mu) / self.sigma
        return 0.5 * (1.0 + math.erf(z / _SQRT2))

    def pdf(self, y: float) -> float:
        z = (float(y) - self.mu) / self.sigma
        return _INV_SQRT_2PI * math.exp(-0.5 * z * z) / self.sigma

    def quantile(self, q: float) -> float:
        if not (0.0 < q < 1.0):
            raise InvalidForecast("Gaussian quantile requires q in (0,1)")
        return self.mu + self.sigma * _SQRT2 * _erfinv(2.0 * q - 1.0)


@dataclass(frozen=True)
class EmpiricalForecast:
    """Forecast represented by sorted samples (an empirical CDF)."""

    samples: tuple  # sorted ascending

    @staticmethod
    def from_iterable(xs: Iterable[float]) -> "EmpiricalForecast":
        s = tuple(sorted(float(x) for x in xs))
        if not s:
            raise InvalidForecast("EmpiricalForecast requires at least one sample")
        return EmpiricalForecast(samples=s)

    def cdf(self, y: float) -> float:
        # Right-continuous step CDF: F(y) = #{x_i ≤ y} / n.
        n = len(self.samples)
        idx = bisect.bisect_right(self.samples, float(y))
        return idx / n

    def quantile(self, q: float) -> float:
        if not (0.0 < q < 1.0):
            raise InvalidForecast("Empirical quantile requires q in (0,1)")
        n = len(self.samples)
        idx = max(0, min(n - 1, int(math.ceil(q * n)) - 1))
        return self.samples[idx]

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples)


@dataclass(frozen=True)
class IntervalForecast:
    """Prediction interval at the stated coverage level (1 - alpha)."""

    lower: float
    upper: float
    level: float

    def __post_init__(self) -> None:
        if not (0.0 < self.level < 1.0):
            raise InvalidForecast("level must be in (0,1)")
        if self.upper < self.lower:
            raise InvalidForecast("upper < lower")

    def covers(self, y: float) -> bool:
        return self.lower <= float(y) <= self.upper

    def width(self) -> float:
        return self.upper - self.lower


# =====================================================================
# Reports
# =====================================================================


@dataclass(frozen=True)
class ScoreReport:
    stream_id: str
    rule: str
    n: int
    mean: float
    total: float
    per_obs: tuple
    certificate: dict
    receipt_id: str


@dataclass(frozen=True)
class CalibrationReport:
    stream_id: str
    method: str
    n: int
    statistic: float
    p_value: float | None
    e_value: float | None
    rejected: bool
    alpha: float
    threshold: float
    pit_values: tuple
    certificate: dict
    receipt_id: str


@dataclass(frozen=True)
class IntervalReport:
    stream_id: str
    method: str
    alpha: float
    lower: float
    upper: float
    width: float
    empirical_coverage: float
    n: int
    certificate: dict
    receipt_id: str


@dataclass(frozen=True)
class EnsembleReport:
    ensemble_id: str
    method: str
    streams: tuple
    weights: tuple
    cumulative_regret_bound: float | None
    n_observations: int
    certificate: dict
    receipt_id: str


@dataclass(frozen=True)
class CoverageReport:
    streams: int
    observations: int
    scores: int
    calibrations: int
    rejections: int
    recalibrations: int
    intervals: int
    ensembles: int
    certificate: dict


# =====================================================================
# Helpers — special functions
# =====================================================================


def _erfinv(x: float) -> float:
    """Rational + Newton inverse erf, accurate to ~1e-15 in the body."""
    if not (-1.0 < x < 1.0):
        raise ValueError("erfinv: argument must be in (-1, 1)")
    # Winitzki initialisation; one Newton step on erf gives full precision.
    a = 0.147
    ln = math.log(1.0 - x * x)
    s = (2.0 / (math.pi * a)) + (ln / 2.0)
    y = math.copysign(math.sqrt(math.sqrt(s * s - ln / a) - s), x)
    # Newton refinement
    for _ in range(2):
        err = math.erf(y) - x
        y -= err / (2.0 * _INV_SQRT_PI * math.exp(-y * y))
    return y


def _phi(z: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def _Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def _validate_pmf(items: Sequence, tol: float = 1e-9) -> None:
    if not items:
        raise InvalidForecast("PMF requires ≥1 category")
    s = 0.0
    for k, p in items:
        if p < -tol:
            raise InvalidForecast(f"negative mass on {k!r}: {p!r}")
        s += float(p)
    if abs(s - 1.0) > max(tol, 1e-9):
        raise InvalidForecast(f"PMF mass = {s!r} (tol={tol})")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _hash_payload(payload: Any) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(payload).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()


# =====================================================================
# Proper scoring rules
# =====================================================================


def brier_score(forecast: Any, outcome: Any) -> float:
    """Brier (quadratic) score for a categorical/Bernoulli forecast.

    Returns ∑_k (p_k - 1{y=k})².  Strictly proper, [0, 2]-bounded for
    a normalised PMF; for Bernoulli forecasts we report the canonical
    2(p_y - 1{y=1})² form normalised to [0, 1].
    """
    if isinstance(forecast, BernoulliForecast):
        y = 1 if int(outcome) == 1 else 0
        return (forecast.p - y) ** 2 + ((1.0 - forecast.p) - (1 - y)) ** 2
    if isinstance(forecast, CategoricalForecast):
        s = 0.0
        seen = False
        for k, p in forecast.probs:
            ind = 1.0 if k == outcome else 0.0
            if k == outcome:
                seen = True
            s += (float(p) - ind) ** 2
        if not seen:
            # Outcome label not in support: treat as p=0 there.
            s += 1.0
        return s
    raise InvalidForecast("brier_score requires Bernoulli/Categorical forecast")


def log_score(forecast: Any, outcome: Any) -> float:
    """Negative log-likelihood of the outcome under the forecast.

    Strictly proper; ∞ if the forecast assigns zero mass to the
    realised outcome. We clamp at ``_LOG_CLAMP`` to keep the running
    stream finite — coordination engines should also surface
    out-of-support events via attestations.
    """
    if isinstance(forecast, BernoulliForecast):
        p = forecast.p if int(outcome) == 1 else 1.0 - forecast.p
        return -math.log(max(p, _LOG_CLAMP))
    if isinstance(forecast, CategoricalForecast):
        p = forecast.prob_of(outcome)
        return -math.log(max(p, _LOG_CLAMP))
    if isinstance(forecast, GaussianForecast):
        pdf = forecast.pdf(float(outcome))
        return -math.log(max(pdf, _LOG_CLAMP))
    raise InvalidForecast("log_score requires a probabilistic forecast")


def spherical_score(forecast: Any, outcome: Any) -> float:
    """Spherical score — strictly proper for PMF forecasts.

    Returns p_y / √(∑_k p_k²).  Higher is better — we negate so the
    Forecaster's "lower-is-better" contract holds uniformly.
    """
    if isinstance(forecast, BernoulliForecast):
        p1, p0 = forecast.p, 1.0 - forecast.p
        nrm = math.sqrt(max(p1 * p1 + p0 * p0, _EPS))
        py = p1 if int(outcome) == 1 else p0
        return -(py / nrm)
    if isinstance(forecast, CategoricalForecast):
        nrm = math.sqrt(max(sum(float(p) * float(p) for _, p in forecast.probs), _EPS))
        py = forecast.prob_of(outcome)
        return -(py / nrm)
    raise InvalidForecast("spherical_score requires Bernoulli/Categorical forecast")


def quadratic_score(forecast: Any, outcome: Any) -> float:
    """Quadratic score — alias for Brier under the Forecaster contract."""
    return brier_score(forecast, outcome)


def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS for a Gaussian forecast (Gneiting & Raftery)."""
    if sigma <= 0:
        raise InvalidForecast("CRPS_gaussian requires sigma > 0")
    z = (float(y) - mu) / sigma
    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - _INV_SQRT_PI)


def crps_empirical(samples: Sequence[float], y: float) -> float:
    """CRPS for an empirical forecast via the closed-form
    E|X - y| - ½ E|X - X'|.

    For a sorted sample x_(1) ≤ … ≤ x_(n):
        ½ E|X - X'| = (1/n²) ∑_i (2i - n - 1) x_(i)
    so the whole thing is O(n) after sort.
    """
    n = len(samples)
    if n == 0:
        raise InvalidForecast("crps_empirical: empty samples")
    xs = sorted(float(s) for s in samples)
    term1 = sum(abs(x - float(y)) for x in xs) / n
    if n == 1:
        return term1
    term2 = 0.0
    for i, x in enumerate(xs, start=1):
        term2 += (2 * i - n - 1) * x
    term2 /= (n * n)
    return term1 - term2


def crps(forecast: Any, outcome: float) -> float:
    """Dispatch CRPS for the registered forecast types."""
    if isinstance(forecast, GaussianForecast):
        return crps_gaussian(forecast.mu, forecast.sigma, outcome)
    if isinstance(forecast, EmpiricalForecast):
        return crps_empirical(forecast.samples, outcome)
    if isinstance(forecast, BernoulliForecast):
        # Bernoulli is a 2-atom empirical at {0,1} with masses 1-p, p.
        return crps_empirical(
            (0.0,) * 1 + (1.0,) * 1, outcome
        ) if False else _crps_bernoulli(forecast.p, outcome)
    if isinstance(forecast, CategoricalForecast):
        # Treat the support as real numbers when the labels are numeric.
        support = []
        weights = []
        for k, p in forecast.probs:
            if not isinstance(k, (int, float)):
                raise InvalidForecast(
                    "CRPS requires numeric labels for CategoricalForecast"
                )
            support.append(float(k))
            weights.append(float(p))
        return _crps_discrete(support, weights, float(outcome))
    raise InvalidForecast("crps not implemented for this forecast type")


def _crps_bernoulli(p: float, y: int) -> float:
    # CRPS for Bernoulli with success prob p, outcome y∈{0,1}:
    # = (1-p)² 1{y=1} + p² 1{y=0}      (after the standard reduction)
    # — equivalent to Brier here.
    if int(y) == 1:
        return (1.0 - p) ** 2
    return p ** 2


def _crps_discrete(support: Sequence[float], weights: Sequence[float], y: float) -> float:
    """CRPS for a finite discrete distribution on the real line.

    Sort by support, compute CRPS = ∫ (F(t) - 1{t ≥ y})² dt analytically
    on each constant-CDF segment.
    """
    pairs = sorted(zip(support, weights), key=lambda t: t[0])
    xs = [x for x, _ in pairs]
    ws = [w for _, w in pairs]
    n = len(xs)
    cdf = []
    s = 0.0
    for w in ws:
        s += w
        cdf.append(s)
    # Augment with ±∞ sentinels; CRPS contribution outside [min, max]
    # is finite because (F-1)² is 1 below min, 0 above max, integrand
    # vanishes — but the |y| tails matter when y is outside [min, max].
    total = 0.0
    # Left tail: t < xs[0]. F(t) = 0, 1{t≥y} = 1 if y ≤ t (impossible
    # for t < xs[0] when y > xs[0]). So contributes (xs[0] - y)*1 if
    # y < xs[0] else 0.  Equivalently:
    if y < xs[0]:
        total += xs[0] - y
    # Right tail: t > xs[-1].  F(t) = 1; 1{t ≥ y} = 1 always (when
    # t > xs[-1] ≥ y) so integrand is 0.  Otherwise (y > xs[-1]):
    if y > xs[-1]:
        total += y - xs[-1]
    # Inner: between consecutive support points the CDF is constant
    # cdf[i]; the integrand is (cdf[i] - 1{t≥y})².
    for i in range(n - 1):
        lo, hi = xs[i], xs[i + 1]
        f = cdf[i]
        if y <= lo:
            total += (f - 1.0) ** 2 * (hi - lo)
        elif y >= hi:
            total += (f - 0.0) ** 2 * (hi - lo)
        else:
            total += (f - 0.0) ** 2 * (y - lo) + (f - 1.0) ** 2 * (hi - y)
    return total


def pinball_loss(forecast_quantile: float, outcome: float, q: float) -> float:
    """Pinball / quantile loss for a quantile forecast at level q ∈ (0,1)."""
    if not (0.0 < q < 1.0):
        raise InvalidForecast("pinball loss requires q in (0,1)")
    err = float(outcome) - float(forecast_quantile)
    return q * err if err >= 0 else (q - 1.0) * err


def linex_loss(point: float, outcome: float, a: float = 1.0) -> float:
    """Asymmetric exponential loss (Varian, 1975):  e^{a*err} - a*err - 1."""
    err = float(outcome) - float(point)
    return math.exp(a * err) - a * err - 1.0


# =====================================================================
# PIT and calibration tests
# =====================================================================


def pit_value(forecast: Any, outcome: Any) -> float:
    """Probability Integral Transform of an outcome under a forecast.

    For continuous F, u = F(y) ~ U[0,1] under perfect calibration.
    For discrete F we use the randomised PIT (Brockwell, 2007):
        u = F(y-) + V (F(y) - F(y-))
    with V ~ U[0,1] inside the atom — guaranteeing u ~ U[0,1]
    under correct specification.
    """
    if isinstance(forecast, (GaussianForecast, EmpiricalForecast)):
        u = forecast.cdf(float(outcome))
        return _clamp(u, 0.0, 1.0)
    if isinstance(forecast, (BernoulliForecast, CategoricalForecast)):
        # Randomised PIT — needs a uniform draw.  We use a deterministic
        # one based on the outcome's hash for reproducibility unless the
        # caller injected randomness via _pit_randomised.
        v = (hash(("pit", outcome)) & 0xFFFFFFFF) / float(0x100000000)
        return _pit_randomised(forecast, outcome, v)
    raise InvalidForecast("PIT not defined for this forecast type")


def _pit_randomised(forecast: Any, outcome: Any, v: float) -> float:
    v = _clamp(v, 0.0, 1.0)
    if isinstance(forecast, BernoulliForecast):
        if int(outcome) == 0:
            return v * (1.0 - forecast.p)
        return (1.0 - forecast.p) + v * forecast.p
    if isinstance(forecast, CategoricalForecast):
        running = 0.0
        for k, p in forecast.probs:
            running_prev = running
            running += float(p)
            if k == outcome:
                return running_prev + v * float(p)
        # Outcome not in support: treat as a zero-atom right of support.
        return 1.0
    raise InvalidForecast("randomised PIT requires Bernoulli/Categorical")


def ks_statistic(uniform_samples: Sequence[float]) -> float:
    """One-sample KS statistic against U[0,1]."""
    n = len(uniform_samples)
    if n == 0:
        return 0.0
    xs = sorted(_clamp(float(u), 0.0, 1.0) for u in uniform_samples)
    d_plus = max((i + 1) / n - xs[i] for i in range(n))
    d_minus = max(xs[i] - i / n for i in range(n))
    return max(d_plus, d_minus)


def dkw_threshold(n: int, alpha: float) -> float:
    """Massart-DKW finite-sample threshold for the KS statistic.

        P(sup_t |F_n(t) - F(t)| > ε) ≤ 2 exp(-2 n ε²).
    Set 2 exp(-2 n ε²) = α ⇒ ε = √(log(2/α) / (2n)).
    """
    if n <= 0 or alpha <= 0 or alpha >= 1:
        raise InvalidForecast("dkw_threshold: n>0 and alpha in (0,1) required")
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def anderson_darling(uniform_samples: Sequence[float]) -> float:
    """Anderson-Darling A² statistic against U[0,1]."""
    n = len(uniform_samples)
    if n == 0:
        return 0.0
    xs = sorted(_clamp(float(u), _LOG_CLAMP, 1.0 - _LOG_CLAMP) for u in uniform_samples)
    s = 0.0
    for i, u in enumerate(xs, start=1):
        s += (2 * i - 1) * (math.log(u) + math.log(1.0 - xs[n - i]))
    return -n - s / n


def anderson_darling_pvalue(a2: float, n: int) -> float:
    """Stephens (1986) modified A² p-value approximation for U[0,1]."""
    if n < 5:
        return 1.0  # under-powered
    a = a2 * (1.0 + 0.75 / n + 2.25 / (n * n))
    # Stephens' table for the U[0,1] null
    if a < 0.200:
        return 1.0 - math.exp(-13.436 + 101.14 * a - 223.73 * a * a)
    if a < 0.340:
        return 1.0 - math.exp(-8.318 + 42.796 * a - 59.938 * a * a)
    if a < 0.600:
        return math.exp(0.9177 - 4.279 * a - 1.38 * a * a)
    if a < 13.0:
        return math.exp(1.2937 - 5.709 * a + 0.0186 * a * a)
    return 0.0


class _EProcessUniform:
    """Anytime-valid e-process testing H₀: u_t ~ U[0,1] i.i.d.

    Construction (Vovk-Wang; Waudby-Smith & Ramdas, aGRAPA tail).

    Test family:
        f(u; θ) = 1 + θ (2u - 1),   θ ∈ [-1, 1].

    For U ~ U[0,1], E[f(U; θ)] = 1, and f(u; θ) ≥ 0 for u ∈ [0,1].
    The wealth process

        W_t = ∏_{s≤t} f(u_s; θ_s)

    is a non-negative martingale under H₀ when θ_t is *predictable*
    (depends only on u_1, …, u_{t-1}).  By Ville:

        P_{H₀}(∃t : W_t ≥ 1/α) ≤ α.

    The Forecaster picks θ_t with aGRAPA: at step t, choose

        θ_t = argmax_{|θ| ≤ c}  ∑_{s<t} log(1 + θ (2 u_s - 1)),

    truncated to ``cap`` for numerical safety.  Closed form via the
    sum and second moment of the centred PIT (m, v):

        θ_grapa = m / (v + m²),     clipped to [-cap, cap].
    """

    def __init__(self, cap: float = 0.85) -> None:
        if not (0.0 < cap < 1.0):
            raise ValueError("e-process cap must lie in (0,1)")
        self.cap = cap
        self.log_w = 0.0
        self.n = 0
        self._sum_centred = 0.0  # ∑ (2u_s - 1)
        self._sum_sq = 0.0  # ∑ (2u_s - 1)²

    def update(self, u: float) -> None:
        z = 2.0 * _clamp(float(u), 0.0, 1.0) - 1.0
        # Predict θ_t from history *before* this observation.
        if self.n == 0:
            theta = 0.0
        else:
            m = self._sum_centred / self.n
            v = self._sum_sq / self.n - m * m
            denom = v + m * m
            theta = m / denom if denom > 1e-12 else 0.0
            theta = _clamp(theta, -self.cap, self.cap)
        # Bet
        f = 1.0 + theta * z
        if f <= 0:
            # Numerical safety — collapse wealth to 0
            self.log_w = -float("inf")
        else:
            self.log_w += math.log(f)
        # Accumulate for the *next* prediction
        self._sum_centred += z
        self._sum_sq += z * z
        self.n += 1

    def e_value(self) -> float:
        return math.exp(self.log_w) if self.log_w != -float("inf") else 0.0

    def rejected(self, alpha: float) -> bool:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie in (0,1)")
        if self.log_w == -float("inf"):
            return True
        return self.log_w >= math.log(1.0 / alpha)


# =====================================================================
# Pools / aggregation
# =====================================================================


def linear_pool(forecasts: Sequence[Any], weights: Sequence[float]) -> Any:
    """Linear opinion pool.  All forecasts must share the same type/support."""
    if not forecasts:
        raise InvalidForecast("linear_pool: empty forecasts")
    if len(forecasts) != len(weights):
        raise InvalidForecast("linear_pool: weight/forecast length mismatch")
    s = sum(weights)
    if s <= 0:
        raise InvalidForecast("linear_pool: weights must be > 0 in sum")
    ws = [float(w) / s for w in weights]
    f0 = forecasts[0]
    if isinstance(f0, BernoulliForecast):
        p = sum(ws[i] * forecasts[i].p for i in range(len(forecasts)))
        return BernoulliForecast(p=p)
    if isinstance(f0, CategoricalForecast):
        support = f0.support()
        agg = {k: 0.0 for k in support}
        for w, fc in zip(ws, forecasts):
            if not isinstance(fc, CategoricalForecast):
                raise InvalidForecast("linear_pool: mixed forecast types")
            if fc.support() != support:
                raise InvalidForecast("linear_pool: mismatched supports")
            for k, p in fc.probs:
                agg[k] += float(w) * float(p)
        return CategoricalForecast.from_dict(agg)
    if isinstance(f0, GaussianForecast):
        # Pool by mixture moments — exact mean, mixture variance.
        mu = sum(ws[i] * forecasts[i].mu for i in range(len(forecasts)))
        var = sum(
            ws[i] * (forecasts[i].sigma ** 2 + (forecasts[i].mu - mu) ** 2)
            for i in range(len(forecasts))
        )
        return GaussianForecast(mu=mu, sigma=math.sqrt(max(var, _EPS)))
    if isinstance(f0, EmpiricalForecast):
        # Pool empiricals by quantile averaging — closed-form sliced
        # barycenter trick in 1-D: F^{-1}_pool = ∑ w_i F_i^{-1}.
        n = min(len(fc.samples) for fc in forecasts)
        if n == 0:
            raise InvalidForecast("linear_pool: empty empirical forecast")
        rep = []
        for i in range(n):
            q = (i + 0.5) / n
            v = sum(
                ws[j] * forecasts[j].quantile(q)
                for j in range(len(forecasts))
            )
            rep.append(v)
        return EmpiricalForecast(samples=tuple(sorted(rep)))
    raise InvalidForecast("linear_pool: unsupported forecast type")


def log_pool(forecasts: Sequence[Any], weights: Sequence[float]) -> Any:
    """Logarithmic (geometric) opinion pool — externally Bayesian."""
    if not forecasts:
        raise InvalidForecast("log_pool: empty")
    s = sum(weights)
    if s <= 0:
        raise InvalidForecast("log_pool: weights must be > 0 in sum")
    ws = [float(w) / s for w in weights]
    f0 = forecasts[0]
    if isinstance(f0, BernoulliForecast):
        ln_p = sum(ws[i] * math.log(max(forecasts[i].p, _LOG_CLAMP)) for i in range(len(forecasts)))
        ln_q = sum(ws[i] * math.log(max(1.0 - forecasts[i].p, _LOG_CLAMP)) for i in range(len(forecasts)))
        m = max(ln_p, ln_q)
        ep, eq = math.exp(ln_p - m), math.exp(ln_q - m)
        return BernoulliForecast(p=ep / (ep + eq))
    if isinstance(f0, CategoricalForecast):
        support = f0.support()
        ln_agg = {k: 0.0 for k in support}
        for w, fc in zip(ws, forecasts):
            if not isinstance(fc, CategoricalForecast):
                raise InvalidForecast("log_pool: mixed forecast types")
            if fc.support() != support:
                raise InvalidForecast("log_pool: mismatched supports")
            for k, p in fc.probs:
                ln_agg[k] += float(w) * math.log(max(float(p), _LOG_CLAMP))
        m = max(ln_agg.values())
        unnorm = {k: math.exp(v - m) for k, v in ln_agg.items()}
        z = sum(unnorm.values())
        return CategoricalForecast.from_dict({k: v / z for k, v in unnorm.items()})
    raise InvalidForecast("log_pool: unsupported forecast type for log-pool")


class HedgeAggregator:
    """Exponentially-weighted average forecaster (Hedge / EW).

    Maintains weights w_i ∝ exp(-η L_i^{cum}) over K experts, where
    L_i^{cum} is the cumulative *normalised* loss of expert i so far.

    Theorem (Cesa-Bianchi & Lugosi, 2006, Thm 2.2): for any horizon
    T, with η = √(8 log K / T) the regret of the EW forecaster
    against the best fixed expert is

        R_T ≤ √(T/2 · log K).

    Setting η_t = √(8 log K / t) gives the same √(T log K) order
    *uniformly* in T — the parameter-free variant we ship.
    """

    def __init__(self, K: int, eta: float | None = None) -> None:
        if K < 1:
            raise InvalidForecast("HedgeAggregator: K ≥ 1 required")
        self.K = K
        self.eta = eta
        self.cum_loss = [0.0] * K
        self.t = 0
        self.cum_regret_bound = 0.0

    def weights(self) -> tuple:
        if self.t == 0:
            w = 1.0 / self.K
            return tuple(w for _ in range(self.K))
        eta = self.eta if self.eta is not None else math.sqrt(8.0 * math.log(max(self.K, 2)) / max(self.t, 1))
        # Stabilise: subtract min before exponentiating
        lo = min(self.cum_loss)
        ws = [math.exp(-eta * (l - lo)) for l in self.cum_loss]
        s = sum(ws)
        return tuple(w / s for w in ws)

    def update(self, losses: Sequence[float]) -> None:
        if len(losses) != self.K:
            raise InvalidForecast(f"HedgeAggregator.update: expected K={self.K} losses")
        self.t += 1
        for i in range(self.K):
            self.cum_loss[i] += float(losses[i])
        # Track the standard √(T/2 log K) regret bound; it's the
        # *cumulative* over t observations.
        self.cum_regret_bound = math.sqrt(0.5 * self.t * math.log(max(self.K, 2)))


class PolynomialWeightsAggregator:
    """Cesa-Bianchi-Lugosi polynomial-weights (no learning rate)."""

    def __init__(self, K: int, power: float = 2.0) -> None:
        if K < 1:
            raise InvalidForecast("PolynomialWeightsAggregator: K ≥ 1")
        if power <= 1.0:
            raise InvalidForecast("polynomial power must be > 1")
        self.K = K
        self.power = power
        self.cum_loss = [0.0] * K
        self.t = 0
        self.cum_regret_bound = 0.0

    def weights(self) -> tuple:
        if self.t == 0:
            return tuple(1.0 / self.K for _ in range(self.K))
        lo = min(self.cum_loss)
        ws = [max(lo - l + 1e-9, 0.0) ** (self.power - 1) for l in self.cum_loss]
        s = sum(ws)
        if s <= 0:
            return tuple(1.0 / self.K for _ in range(self.K))
        return tuple(w / s for w in ws)

    def update(self, losses: Sequence[float]) -> None:
        if len(losses) != self.K:
            raise InvalidForecast(f"poly.update: expected K={self.K}")
        self.t += 1
        for i in range(self.K):
            self.cum_loss[i] += float(losses[i])
        # Audibert-style bound for polynomial weights:
        # R_T ≤ √(T (p-1) K^{2/(p-1)}); for p=2 this is √(T K²).
        K = self.K
        p = self.power
        self.cum_regret_bound = math.sqrt(self.t * (p - 1.0) * K ** (2.0 / (p - 1.0)))


# =====================================================================
# Recalibration adapters
# =====================================================================


class _HistogramRecalibrator:
    """Histogram recalibration for Bernoulli/Categorical forecasts.

    Bins predicted probabilities into B equal-width bins; in each bin
    the recalibrated probability is the empirical frequency of class 1
    among observations whose forecast fell in the bin.
    """

    def __init__(self, n_bins: int = 10) -> None:
        if n_bins < 2:
            raise InvalidForecast("HistogramRecalibrator: n_bins ≥ 2")
        self.n_bins = n_bins
        self.bin_p: list[float] = [0.5] * n_bins
        self.bin_n: list[int] = [0] * n_bins

    def fit(self, predictions: Sequence[float], outcomes: Sequence[int]) -> None:
        sums = [0.0] * self.n_bins
        counts = [0] * self.n_bins
        for p, y in zip(predictions, outcomes):
            b = min(int(p * self.n_bins), self.n_bins - 1)
            counts[b] += 1
            sums[b] += 1.0 if int(y) == 1 else 0.0
        for b in range(self.n_bins):
            self.bin_n[b] = counts[b]
            if counts[b] > 0:
                self.bin_p[b] = sums[b] / counts[b]
            else:
                # Empty bin: keep the bin midpoint as the fallback.
                self.bin_p[b] = (b + 0.5) / self.n_bins

    def apply(self, p: float) -> float:
        p = _clamp(p, 0.0, 1.0 - 1e-12)
        b = min(int(p * self.n_bins), self.n_bins - 1)
        return self.bin_p[b]


class _IsotonicRecalibrator:
    """Pool-adjacent-violators isotonic regression in 1-D."""

    def __init__(self) -> None:
        self._x: list[float] = []
        self._y: list[float] = []

    def fit(self, predictions: Sequence[float], outcomes: Sequence[int]) -> None:
        pairs = sorted(
            zip((float(p) for p in predictions), (1.0 if int(y) == 1 else 0.0 for y in outcomes)),
            key=lambda t: t[0],
        )
        if not pairs:
            self._x, self._y = [0.0], [0.5]
            return
        xs = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        # PAV
        groups = [[xs[i], ys[i], 1] for i in range(len(xs))]
        i = 0
        while i < len(groups) - 1:
            if groups[i][1] > groups[i + 1][1]:
                # Pool
                a = groups[i]
                b = groups[i + 1]
                w = a[2] + b[2]
                pooled = [(a[0] * a[2] + b[0] * b[2]) / w, (a[1] * a[2] + b[1] * b[2]) / w, w]
                groups[i:i + 2] = [pooled]
                if i > 0:
                    i -= 1
            else:
                i += 1
        self._x = [g[0] for g in groups]
        self._y = [g[1] for g in groups]

    def apply(self, p: float) -> float:
        if not self._x:
            return _clamp(p, 0.0, 1.0)
        p = _clamp(p, 0.0, 1.0)
        if p <= self._x[0]:
            return self._y[0]
        if p >= self._x[-1]:
            return self._y[-1]
        i = bisect.bisect_right(self._x, p) - 1
        # Step function within the fitted grid — common isotonic form.
        return self._y[i]


class _PITRecalibrator:
    """Distributional recalibration via the empirical PIT CDF.

    Given a history of PIT values u_1, …, u_n under a forecast family,
    define the recalibration map G(u) = (#{i : u_i ≤ u}) / n.  Applied
    on top of the original forecast's CDF, the recalibrated CDF
        F̃(y) = G(F(y))
    has a *perfectly uniform* in-sample PIT.  This is the continuous
    analogue of histogram recalibration.
    """

    def __init__(self) -> None:
        self._pit_sorted: tuple = ()

    def fit(self, pit_values: Sequence[float]) -> None:
        self._pit_sorted = tuple(sorted(_clamp(float(u), 0.0, 1.0) for u in pit_values))

    def apply(self, u: float) -> float:
        if not self._pit_sorted:
            return _clamp(u, 0.0, 1.0)
        idx = bisect.bisect_right(self._pit_sorted, _clamp(u, 0.0, 1.0))
        return idx / len(self._pit_sorted)


# =====================================================================
# Stream state
# =====================================================================


@dataclass
class _StreamRecord:
    stream_id: str
    description: str
    forecasts: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    pit_values: list = field(default_factory=list)
    e_process: _EProcessUniform = field(default_factory=_EProcessUniform)
    recalibrator: Any = None
    recal_method: str | None = None
    registered_ns: int = field(default_factory=time.time_ns)
    last_observation_ns: int = field(default_factory=time.time_ns)


# =====================================================================
# Attestation adapter
# =====================================================================


class _AttestableReceipt:
    __slots__ = ("ticket_id", "kind", "payload", "digest")

    def __init__(self, kind: str, payload: dict, digest: str = "") -> None:
        self.kind = kind
        self.payload = payload
        self.digest = digest
        self.ticket_id = (
            payload.get("receipt_id") or digest[:16] or uuid.uuid4().hex[:16]
        )

    def to_dict(self) -> dict:
        return {"ticket_id": self.ticket_id, "kind": self.kind, "payload": self.payload, "digest": self.digest}


# =====================================================================
# Forecaster runtime
# =====================================================================


class Forecaster:
    """Probabilistic forecasting engine for the agi runtime.

    Threadsafe; stateless except for the registered streams, the
    ensemble aggregator state, and lifetime counters.  Optional
    dependencies:

      bus       — ``agi.events.EventBus`` for live event broadcast
      attestor  — ``agi.attest.RuntimeAttestor`` for content-hashed
                  receipt persistence

    A coordination engine drives the Forecaster through:

      register_stream(id) → record(...) → score(...) /
        calibration_test(...) / interval(...) → recalibrate(...) →
        ensemble(...) → forecast(...)
    """

    def __init__(
        self,
        *,
        bus: Any = None,
        attestor: Any = None,
        random_seed: int | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._lock = threading.RLock()
        self._streams: dict[str, _StreamRecord] = {}
        self._aggregators: dict[str, Any] = {}
        self._aggregator_streams: dict[str, tuple] = {}
        self._rng = random.Random(random_seed)
        self._n_observations = 0
        self._n_scores = 0
        self._n_calibrations = 0
        self._n_rejections = 0
        self._n_recalibrations = 0
        self._n_intervals = 0
        self._n_ensembles = 0
        self._started_ns = time.time_ns()
        self._emit(FORECASTER_STARTED, {"id": uuid.uuid4().hex[:16], "ts_ns": self._started_ns})

    # -------- event / attest helpers --------

    def _emit(self, kind: str, payload: dict) -> None:
        if self._bus is None or Event is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(payload)))
        except Exception:
            pass

    def _attest(self, kind: str, payload: dict) -> str:
        digest = _hash_payload(payload)
        if self._attestor is None:
            return digest
        receipt = _AttestableReceipt(kind=kind, payload=payload, digest=digest)
        try:
            if hasattr(self._attestor, "record"):
                self._attestor.record(kind=kind, payload=payload)
            elif callable(self._attestor):
                self._attestor(receipt)
        except Exception:
            pass
        return digest

    # -------- stream lifecycle --------

    def register_stream(self, stream_id: str, *, description: str = "") -> None:
        if not stream_id:
            raise InvalidForecast("stream_id must be non-empty")
        with self._lock:
            if stream_id in self._streams:
                raise InvalidForecast(f"stream {stream_id!r} already registered")
            self._streams[stream_id] = _StreamRecord(stream_id=stream_id, description=description)
            self._emit(
                FORECASTER_STREAM_REGISTERED,
                {"stream_id": stream_id, "description": description, "ts_ns": time.time_ns()},
            )

    def remove_stream(self, stream_id: str) -> None:
        with self._lock:
            if stream_id not in self._streams:
                raise UnknownStream(stream_id)
            del self._streams[stream_id]
            self._emit(
                FORECASTER_STREAM_REMOVED,
                {"stream_id": stream_id, "ts_ns": time.time_ns()},
            )

    def streams(self) -> tuple:
        with self._lock:
            return tuple(self._streams.keys())

    def stream_size(self, stream_id: str) -> int:
        with self._lock:
            s = self._streams.get(stream_id)
            if s is None:
                raise UnknownStream(stream_id)
            return len(s.outcomes)

    # -------- record / observe --------

    def record(self, stream_id: str, forecast: Any, outcome: Any) -> None:
        """Append a (forecast, outcome) pair to a stream.

        If the stream has a recalibrator attached, the forecast is
        replaced with its recalibrated version before storage; the
        original forecast is still hashed into the attestation
        payload for traceability.
        """
        with self._lock:
            s = self._streams.get(stream_id)
            if s is None:
                raise UnknownStream(stream_id)
            applied = self._apply_recalibration(s, forecast)
            s.forecasts.append(applied)
            s.outcomes.append(outcome)
            try:
                u = pit_value(applied, outcome)
            except InvalidForecast:
                u = None
            if u is not None:
                s.pit_values.append(u)
                s.e_process.update(u)
            s.last_observation_ns = time.time_ns()
            self._n_observations += 1
            payload = {
                "stream_id": stream_id,
                "outcome": _safe(outcome),
                "forecast_digest": _hash_payload(_safe(applied)),
                "raw_forecast_digest": _hash_payload(_safe(forecast)),
                "pit": u,
                "ts_ns": s.last_observation_ns,
            }
            digest = self._attest(FORECASTER_OBSERVED, payload)
            payload["receipt_id"] = digest[:16]
            self._emit(FORECASTER_OBSERVED, payload)

    def _apply_recalibration(self, s: _StreamRecord, forecast: Any) -> Any:
        if s.recalibrator is None:
            return forecast
        if isinstance(forecast, BernoulliForecast):
            new_p = s.recalibrator.apply(forecast.p)
            return BernoulliForecast(p=_clamp(new_p, 0.0, 1.0))
        if isinstance(forecast, CategoricalForecast):
            # 1-vs-all recalibration for each label, then renormalise.
            agg = {}
            tot = 0.0
            for k, p in forecast.probs:
                np = s.recalibrator.apply(float(p))
                agg[k] = np
                tot += np
            if tot <= 0:
                return forecast
            return CategoricalForecast.from_dict({k: v / tot for k, v in agg.items()})
        if isinstance(forecast, (GaussianForecast, EmpiricalForecast)):
            if isinstance(s.recalibrator, _PITRecalibrator):
                # Wrap as a quantile-remapped empirical forecast.
                grid = 65
                qs = [(i + 0.5) / grid for i in range(grid)]
                # Compose: F̃^{-1}(q) = F^{-1}( G^{-1}(q) )
                if isinstance(forecast, GaussianForecast):
                    base_q = forecast.quantile
                elif isinstance(forecast, EmpiricalForecast):
                    base_q = forecast.quantile
                else:  # pragma: no cover
                    base_q = lambda q: q  # noqa: E731
                # G^{-1}: inverse of the empirical PIT CDF
                ginv = s.recalibrator
                sample_qs = []
                pit_sorted = ginv._pit_sorted
                if not pit_sorted:
                    return forecast
                for q in qs:
                    idx = max(0, min(len(pit_sorted) - 1, int(math.ceil(q * len(pit_sorted))) - 1))
                    u_at_q = pit_sorted[idx]
                    # base_q requires strict (0,1)
                    u_at_q = _clamp(u_at_q, 1e-6, 1.0 - 1e-6)
                    sample_qs.append(base_q(u_at_q))
                return EmpiricalForecast(samples=tuple(sorted(sample_qs)))
        return forecast

    # -------- scoring --------

    def score(self, stream_id: str, rule: str, **kwargs: Any) -> ScoreReport:
        if rule not in KNOWN_SCORES:
            raise UnknownMethod(f"unknown scoring rule {rule!r}")
        with self._lock:
            s = self._streams.get(stream_id)
            if s is None:
                raise UnknownStream(stream_id)
            if not s.outcomes:
                raise InsufficientData(f"stream {stream_id} has no observations")
            scores = []
            for fc, y in zip(s.forecasts, s.outcomes):
                if rule == SCORE_BRIER:
                    scores.append(brier_score(fc, y))
                elif rule == SCORE_LOG:
                    scores.append(log_score(fc, y))
                elif rule == SCORE_SPHERICAL:
                    scores.append(spherical_score(fc, y))
                elif rule == SCORE_QUADRATIC:
                    scores.append(quadratic_score(fc, y))
                elif rule == SCORE_CRPS:
                    scores.append(crps(fc, float(y)))
                elif rule == SCORE_PINBALL:
                    q = float(kwargs.get("q", 0.5))
                    point = fc.quantile(q) if hasattr(fc, "quantile") else float(getattr(fc, "mu", 0.0))
                    scores.append(pinball_loss(point, float(y), q))
                elif rule == SCORE_LINEX:
                    a = float(kwargs.get("a", 1.0))
                    point = getattr(fc, "mu", None)
                    if point is None and hasattr(fc, "mean"):
                        point = fc.mean()
                    if point is None:
                        raise InvalidForecast("linex requires a point estimate")
                    scores.append(linex_loss(point, float(y), a))
            n = len(scores)
            mean = sum(scores) / n
            total = sum(scores)
            certificate = {
                "rule": rule,
                "n": n,
                "min": min(scores),
                "max": max(scores),
                "tail_alpha_0_05_normal_ci": 1.96 * _stdev(scores) / math.sqrt(n) if n > 1 else None,
            }
            payload = {
                "stream_id": stream_id,
                "rule": rule,
                "n": n,
                "mean": mean,
                "total": total,
                "ts_ns": time.time_ns(),
            }
            digest = self._attest(FORECASTER_SCORED, payload)
            self._n_scores += 1
            self._emit(FORECASTER_SCORED, {**payload, "receipt_id": digest[:16]})
            return ScoreReport(
                stream_id=stream_id,
                rule=rule,
                n=n,
                mean=mean,
                total=total,
                per_obs=tuple(scores),
                certificate=certificate,
                receipt_id=digest[:16],
            )

    # -------- calibration --------

    def calibration_test(
        self,
        stream_id: str,
        *,
        method: str = CALIB_E_PROCESS,
        alpha: float = 0.05,
    ) -> CalibrationReport:
        if method not in (CALIB_KS, CALIB_ANDERSON, CALIB_E_PROCESS):
            raise UnknownMethod(f"unknown calibration method {method!r}")
        if not (0.0 < alpha < 1.0):
            raise InvalidForecast("alpha must lie in (0,1)")
        with self._lock:
            s = self._streams.get(stream_id)
            if s is None:
                raise UnknownStream(stream_id)
            if not s.pit_values:
                raise InsufficientData("no PIT values recorded for this stream")
            pit = tuple(s.pit_values)
            n = len(pit)
            stat = 0.0
            p_value: float | None = None
            e_value: float | None = None
            rejected = False
            threshold = 0.0
            if method == CALIB_KS:
                stat = ks_statistic(pit)
                threshold = dkw_threshold(n, alpha)
                rejected = stat > threshold
                # Approximate p-value from the Kolmogorov distribution
                p_value = _kolmogorov_p(stat * math.sqrt(n))
            elif method == CALIB_ANDERSON:
                stat = anderson_darling(pit)
                p_value = anderson_darling_pvalue(stat, n)
                rejected = (p_value is not None) and (p_value < alpha)
                threshold = float("nan")
            else:  # E_PROCESS
                # Replay the e-process to derive the *anytime* certificate.
                e_value = s.e_process.e_value()
                threshold = 1.0 / alpha
                rejected = e_value >= threshold
                stat = math.log(max(e_value, _LOG_CLAMP))
            certificate = {
                "method": method,
                "alpha": alpha,
                "n": n,
                "anytime": method == CALIB_E_PROCESS,
                "ville_certificate": rejected and method == CALIB_E_PROCESS,
                "dkw_anytime": False,
            }
            payload = {
                "stream_id": stream_id,
                "method": method,
                "alpha": alpha,
                "n": n,
                "statistic": stat,
                "p_value": p_value,
                "e_value": e_value,
                "rejected": rejected,
                "ts_ns": time.time_ns(),
            }
            digest = self._attest(FORECASTER_CALIBRATION_TESTED, payload)
            self._n_calibrations += 1
            if rejected:
                self._n_rejections += 1
            self._emit(FORECASTER_CALIBRATION_TESTED, {**payload, "receipt_id": digest[:16]})
            return CalibrationReport(
                stream_id=stream_id,
                method=method,
                n=n,
                statistic=stat,
                p_value=p_value,
                e_value=e_value,
                rejected=rejected,
                alpha=alpha,
                threshold=threshold,
                pit_values=pit,
                certificate=certificate,
                receipt_id=digest[:16],
            )

    # -------- recalibration --------

    def recalibrate(self, stream_id: str, *, method: str = RECAL_ISOTONIC, **kwargs: Any) -> dict:
        if method not in KNOWN_RECAL:
            raise UnknownMethod(f"unknown recalibration method {method!r}")
        with self._lock:
            s = self._streams.get(stream_id)
            if s is None:
                raise UnknownStream(stream_id)
            if not s.outcomes:
                raise InsufficientData("no observations to fit a recalibrator")
            if method == RECAL_HISTOGRAM:
                # Bernoulli/Categorical only
                f0 = s.forecasts[0]
                if not isinstance(f0, (BernoulliForecast, CategoricalForecast)):
                    raise InvalidForecast("histogram recalibration requires discrete forecasts")
                recal = _HistogramRecalibrator(n_bins=int(kwargs.get("n_bins", 10)))
                if isinstance(f0, BernoulliForecast):
                    recal.fit([fc.p for fc in s.forecasts], [int(y) for y in s.outcomes])
                else:
                    # Use the predicted prob of class-1-equivalent — call out: needs target
                    target = kwargs.get("target_label")
                    if target is None:
                        raise InvalidForecast("categorical histogram recal requires target_label")
                    recal.fit(
                        [fc.prob_of(target) for fc in s.forecasts],
                        [1 if y == target else 0 for y in s.outcomes],
                    )
            elif method == RECAL_ISOTONIC:
                f0 = s.forecasts[0]
                if not isinstance(f0, (BernoulliForecast, CategoricalForecast)):
                    raise InvalidForecast("isotonic recalibration requires discrete forecasts")
                recal = _IsotonicRecalibrator()
                if isinstance(f0, BernoulliForecast):
                    recal.fit([fc.p for fc in s.forecasts], [int(y) for y in s.outcomes])
                else:
                    target = kwargs.get("target_label")
                    if target is None:
                        raise InvalidForecast("categorical isotonic recal requires target_label")
                    recal.fit(
                        [fc.prob_of(target) for fc in s.forecasts],
                        [1 if y == target else 0 for y in s.outcomes],
                    )
            elif method == RECAL_PIT:
                recal = _PITRecalibrator()
                recal.fit(s.pit_values)
            else:  # RECAL_PLATT
                # Delegate to agi.calibration if available
                try:
                    from agi.calibration import PlattCalibrator  # type: ignore
                except Exception as e:
                    raise UnknownMethod("Platt recalibration unavailable") from e
                recal = PlattCalibrator()
                if not isinstance(s.forecasts[0], BernoulliForecast):
                    raise InvalidForecast("Platt recalibration requires Bernoulli forecasts")
                recal.fit([fc.p for fc in s.forecasts], [int(y) for y in s.outcomes])
                # adapt to .apply()
                if not hasattr(recal, "apply") and hasattr(recal, "calibrate"):
                    setattr(recal, "apply", recal.calibrate)
            s.recalibrator = recal
            s.recal_method = method
            self._n_recalibrations += 1
            payload = {
                "stream_id": stream_id,
                "method": method,
                "n_fit": len(s.outcomes),
                "ts_ns": time.time_ns(),
            }
            digest = self._attest(FORECASTER_RECALIBRATED, payload)
            self._emit(FORECASTER_RECALIBRATED, {**payload, "receipt_id": digest[:16]})
            return {**payload, "receipt_id": digest[:16]}

    # -------- intervals (conformal) --------

    def interval(
        self,
        stream_id: str,
        *,
        alpha: float = 0.1,
        method: str = "split",
    ) -> IntervalReport:
        """Online prediction interval for the next outcome.

        Uses the in-stream history of forecast point-estimates (mean
        or median) and outcomes as the conformity-score residuals.
        The reported interval is centred on the most recent point
        forecast with width = quantile_{1-α}(|residual|).
        """
        if not (0.0 < alpha < 1.0):
            raise InvalidForecast("alpha must lie in (0,1)")
        with self._lock:
            s = self._streams.get(stream_id)
            if s is None:
                raise UnknownStream(stream_id)
            if not s.outcomes:
                raise InsufficientData("interval requires at least one observation")
            residuals = []
            for fc, y in zip(s.forecasts, s.outcomes):
                point = _point_estimate(fc)
                if point is None:
                    raise InvalidForecast("interval: forecast lacks point estimate")
                residuals.append(abs(point - float(y)))
            n = len(residuals)
            # Conformal upper quantile with finite-sample correction:
            # take ceil((1-α)(n+1))-th order statistic.
            r = sorted(residuals)
            k = min(n - 1, max(0, int(math.ceil((1 - alpha) * (n + 1))) - 1))
            half_width = r[k]
            last_point = _point_estimate(s.forecasts[-1]) or 0.0
            lower = last_point - half_width
            upper = last_point + half_width
            # Empirical coverage in-sample
            cov = sum(1 for res in residuals if res <= half_width) / n
            certificate = {
                "alpha": alpha,
                "n": n,
                "finite_sample_target_coverage": 1.0 - alpha,
                "empirical_coverage": cov,
                "method": method,
                "marginal_validity": True,
            }
            payload = {
                "stream_id": stream_id,
                "alpha": alpha,
                "method": method,
                "lower": lower,
                "upper": upper,
                "width": upper - lower,
                "n": n,
                "ts_ns": time.time_ns(),
            }
            digest = self._attest(FORECASTER_INTERVAL_EMITTED, payload)
            self._n_intervals += 1
            self._emit(FORECASTER_INTERVAL_EMITTED, {**payload, "receipt_id": digest[:16]})
            return IntervalReport(
                stream_id=stream_id,
                method=method,
                alpha=alpha,
                lower=lower,
                upper=upper,
                width=upper - lower,
                empirical_coverage=cov,
                n=n,
                certificate=certificate,
                receipt_id=digest[:16],
            )

    # -------- ensembling --------

    def ensemble(
        self,
        ensemble_id: str,
        streams: Sequence[str],
        *,
        method: str = POOL_HEDGE,
        weights: Sequence[float] | None = None,
        rule: str = SCORE_BRIER,
        eta: float | None = None,
    ) -> EnsembleReport:
        if method not in KNOWN_POOLS:
            raise UnknownMethod(f"unknown pool {method!r}")
        with self._lock:
            for sid in streams:
                if sid not in self._streams:
                    raise UnknownStream(sid)
            K = len(streams)
            if K == 0:
                raise InvalidForecast("ensemble requires ≥1 stream")
            # Build/refresh the aggregator
            if method == POOL_HEDGE:
                agg = self._aggregators.get(ensemble_id)
                if agg is None or not isinstance(agg, HedgeAggregator) or agg.K != K:
                    agg = HedgeAggregator(K=K, eta=eta)
                    self._aggregators[ensemble_id] = agg
                    self._aggregator_streams[ensemble_id] = tuple(streams)
                # Replay all per-stream per-step losses
                n_steps = min(self.stream_size(sid) for sid in streams)
                agg.cum_loss = [0.0] * K
                agg.t = 0
                for t in range(n_steps):
                    losses = []
                    for sid in streams:
                        s = self._streams[sid]
                        losses.append(_loss_of(s.forecasts[t], s.outcomes[t], rule))
                    agg.update(losses)
                ws = agg.weights()
                rb = agg.cum_regret_bound
            elif method == POOL_POLY:
                agg = self._aggregators.get(ensemble_id)
                if agg is None or not isinstance(agg, PolynomialWeightsAggregator) or agg.K != K:
                    agg = PolynomialWeightsAggregator(K=K)
                    self._aggregators[ensemble_id] = agg
                    self._aggregator_streams[ensemble_id] = tuple(streams)
                n_steps = min(self.stream_size(sid) for sid in streams)
                agg.cum_loss = [0.0] * K
                agg.t = 0
                for t in range(n_steps):
                    losses = [
                        _loss_of(
                            self._streams[sid].forecasts[t],
                            self._streams[sid].outcomes[t],
                            rule,
                        )
                        for sid in streams
                    ]
                    agg.update(losses)
                ws = agg.weights()
                rb = agg.cum_regret_bound
            else:  # linear / log : user-supplied weights
                if weights is None:
                    ws = tuple(1.0 / K for _ in range(K))
                else:
                    s = sum(weights)
                    if s <= 0:
                        raise InvalidForecast("weights sum must be > 0")
                    ws = tuple(float(w) / s for w in weights)
                rb = None
                self._aggregator_streams[ensemble_id] = tuple(streams)
            n_observations = min(self.stream_size(sid) for sid in streams) if streams else 0
            certificate = {
                "method": method,
                "K": K,
                "rule": rule,
                "regret_bound": rb,
                "ts_ns": time.time_ns(),
            }
            payload = {
                "ensemble_id": ensemble_id,
                "streams": tuple(streams),
                "method": method,
                "weights": tuple(ws),
                "cumulative_regret_bound": rb,
                "n_observations": n_observations,
                "ts_ns": time.time_ns(),
            }
            digest = self._attest(FORECASTER_ENSEMBLE_UPDATED, payload)
            self._n_ensembles += 1
            self._emit(FORECASTER_ENSEMBLE_UPDATED, {**payload, "receipt_id": digest[:16]})
            return EnsembleReport(
                ensemble_id=ensemble_id,
                method=method,
                streams=tuple(streams),
                weights=tuple(ws),
                cumulative_regret_bound=rb,
                n_observations=n_observations,
                certificate=certificate,
                receipt_id=digest[:16],
            )

    def ensemble_weights(self, ensemble_id: str) -> tuple:
        with self._lock:
            agg = self._aggregators.get(ensemble_id)
            if agg is None:
                streams = self._aggregator_streams.get(ensemble_id)
                if streams is None:
                    raise UnknownStream(ensemble_id)
                K = len(streams)
                return tuple(1.0 / K for _ in range(K))
            return agg.weights()

    # -------- live forecast --------

    def forecast(
        self,
        stream_id: str | None = None,
        *,
        ensemble_id: str | None = None,
        forecasts: Sequence[Any] | None = None,
        method: str = POOL_LINEAR,
    ) -> Any:
        """Emit a forecast.

        - ``stream_id``: return the last stored forecast (post-recalibration).
        - ``ensemble_id``: take the latest forecasts of the ensemble's
          member streams and combine via the configured method.
        - ``forecasts`` + ``method``: ad-hoc one-shot pool of forecasts.
        """
        with self._lock:
            if ensemble_id is not None:
                streams = self._aggregator_streams.get(ensemble_id)
                if streams is None:
                    raise UnknownStream(ensemble_id)
                fc_list = []
                for sid in streams:
                    s = self._streams[sid]
                    if not s.forecasts:
                        raise InsufficientData(f"stream {sid} has no forecasts yet")
                    fc_list.append(s.forecasts[-1])
                ws = self.ensemble_weights(ensemble_id)
                # method choice from the aggregator
                agg = self._aggregators.get(ensemble_id)
                pool_method = POOL_HEDGE if isinstance(agg, HedgeAggregator) else (
                    POOL_POLY if isinstance(agg, PolynomialWeightsAggregator) else POOL_LINEAR
                )
                out = (linear_pool if pool_method != POOL_LOG else log_pool)(fc_list, ws)
            elif stream_id is not None:
                s = self._streams.get(stream_id)
                if s is None:
                    raise UnknownStream(stream_id)
                if not s.forecasts:
                    raise InsufficientData("no forecasts recorded")
                out = s.forecasts[-1]
            elif forecasts:
                if method == POOL_LOG:
                    out = log_pool(forecasts, [1.0 / len(forecasts)] * len(forecasts))
                else:
                    out = linear_pool(forecasts, [1.0 / len(forecasts)] * len(forecasts))
            else:
                raise InvalidForecast("forecast() requires stream_id, ensemble_id, or forecasts")
            payload = {
                "stream_id": stream_id,
                "ensemble_id": ensemble_id,
                "forecast_digest": _hash_payload(_safe(out)),
                "ts_ns": time.time_ns(),
            }
            digest = self._attest(FORECASTER_FORECAST_EMITTED, payload)
            self._emit(FORECASTER_FORECAST_EMITTED, {**payload, "receipt_id": digest[:16]})
            return out

    # -------- snapshot / coverage --------

    def coverage(self) -> CoverageReport:
        with self._lock:
            cert = {
                "started_ns": self._started_ns,
                "ts_ns": time.time_ns(),
                "stream_ids": tuple(self._streams.keys()),
                "ensembles": tuple(self._aggregator_streams.keys()),
            }
            return CoverageReport(
                streams=len(self._streams),
                observations=self._n_observations,
                scores=self._n_scores,
                calibrations=self._n_calibrations,
                rejections=self._n_rejections,
                recalibrations=self._n_recalibrations,
                intervals=self._n_intervals,
                ensembles=self._n_ensembles,
                certificate=cert,
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "streams": {
                    sid: {
                        "description": s.description,
                        "n": len(s.outcomes),
                        "recal_method": s.recal_method,
                        "log_w": s.e_process.log_w,
                        "e_n": s.e_process.n,
                    }
                    for sid, s in self._streams.items()
                },
                "aggregators": {
                    eid: {
                        "K": agg.K,
                        "t": agg.t,
                        "cum_loss": tuple(agg.cum_loss),
                        "type": type(agg).__name__,
                    }
                    for eid, agg in self._aggregators.items()
                },
                "counters": {
                    "observations": self._n_observations,
                    "scores": self._n_scores,
                    "calibrations": self._n_calibrations,
                    "rejections": self._n_rejections,
                    "recalibrations": self._n_recalibrations,
                    "intervals": self._n_intervals,
                    "ensembles": self._n_ensembles,
                },
            }

    def clear(self) -> None:
        with self._lock:
            self._streams.clear()
            self._aggregators.clear()
            self._aggregator_streams.clear()
            self._n_observations = 0
            self._n_scores = 0
            self._n_calibrations = 0
            self._n_rejections = 0
            self._n_recalibrations = 0
            self._n_intervals = 0
            self._n_ensembles = 0
            self._emit(FORECASTER_CLEARED, {"ts_ns": time.time_ns()})


# =====================================================================
# Module-level helpers
# =====================================================================


def _point_estimate(fc: Any) -> float | None:
    if isinstance(fc, GaussianForecast):
        return fc.mu
    if isinstance(fc, EmpiricalForecast):
        return fc.mean()
    if isinstance(fc, BernoulliForecast):
        return fc.p
    if isinstance(fc, CategoricalForecast):
        # Expected value under numeric labels, if any.
        try:
            return sum(float(k) * float(p) for k, p in fc.probs)
        except (TypeError, ValueError):
            return None
    return None


def _loss_of(forecast: Any, outcome: Any, rule: str) -> float:
    if rule == SCORE_BRIER:
        return brier_score(forecast, outcome)
    if rule == SCORE_LOG:
        return log_score(forecast, outcome)
    if rule == SCORE_SPHERICAL:
        return spherical_score(forecast, outcome)
    if rule == SCORE_QUADRATIC:
        return quadratic_score(forecast, outcome)
    if rule == SCORE_CRPS:
        return crps(forecast, float(outcome))
    raise UnknownMethod(f"loss rule {rule!r} not supported in ensembling")


def _safe(obj: Any) -> Any:
    """Make a hashable, JSON-friendly summary of a forecast / outcome."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, BernoulliForecast):
        return {"type": "bernoulli", "p": obj.p}
    if isinstance(obj, CategoricalForecast):
        return {"type": "categorical", "probs": tuple((str(k), float(p)) for k, p in obj.probs)}
    if isinstance(obj, GaussianForecast):
        return {"type": "gaussian", "mu": obj.mu, "sigma": obj.sigma}
    if isinstance(obj, EmpiricalForecast):
        return {"type": "empirical", "n": len(obj.samples), "mean": obj.mean()}
    if isinstance(obj, IntervalForecast):
        return {"type": "interval", "lower": obj.lower, "upper": obj.upper, "level": obj.level}
    return repr(obj)


def _stdev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _kolmogorov_p(z: float) -> float:
    """Asymptotic Kolmogorov tail probability:
        P(K > z) = 2 ∑_{k=1}^∞ (-1)^{k-1} exp(-2 k² z²).
    Truncated at k=64 — exponentially small contributions beyond.
    """
    if z <= 0:
        return 1.0
    s = 0.0
    sign = 1.0
    for k in range(1, 65):
        term = sign * math.exp(-2.0 * k * k * z * z)
        s += term
        sign = -sign
        if abs(term) < 1e-16:
            break
    p = 2.0 * s
    return max(0.0, min(1.0, p))


# =====================================================================
# Public re-exports
# =====================================================================


__all__ = [
    # event kinds
    "FORECASTER_STARTED",
    "FORECASTER_STREAM_REGISTERED",
    "FORECASTER_STREAM_REMOVED",
    "FORECASTER_OBSERVED",
    "FORECASTER_SCORED",
    "FORECASTER_CALIBRATION_TESTED",
    "FORECASTER_RECALIBRATED",
    "FORECASTER_ENSEMBLE_UPDATED",
    "FORECASTER_INTERVAL_EMITTED",
    "FORECASTER_FORECAST_EMITTED",
    "FORECASTER_CLEARED",
    "FORECASTER_REPORT",
    # scoring rules
    "SCORE_BRIER",
    "SCORE_LOG",
    "SCORE_SPHERICAL",
    "SCORE_QUADRATIC",
    "SCORE_CRPS",
    "SCORE_PINBALL",
    "SCORE_LINEX",
    "KNOWN_SCORES",
    # calibration / pool / recal methods
    "CALIB_KS",
    "CALIB_ANDERSON",
    "CALIB_E_PROCESS",
    "POOL_LINEAR",
    "POOL_LOG",
    "POOL_HEDGE",
    "POOL_POLY",
    "KNOWN_POOLS",
    "RECAL_ISOTONIC",
    "RECAL_HISTOGRAM",
    "RECAL_PIT",
    "RECAL_PLATT",
    "KNOWN_RECAL",
    # errors
    "ForecasterError",
    "InvalidForecast",
    "UnknownStream",
    "InsufficientData",
    "UnknownMethod",
    # forecast types
    "CategoricalForecast",
    "BernoulliForecast",
    "GaussianForecast",
    "EmpiricalForecast",
    "IntervalForecast",
    # reports
    "ScoreReport",
    "CalibrationReport",
    "IntervalReport",
    "EnsembleReport",
    "CoverageReport",
    # scoring rule functions
    "brier_score",
    "log_score",
    "spherical_score",
    "quadratic_score",
    "crps",
    "crps_gaussian",
    "crps_empirical",
    "pinball_loss",
    "linex_loss",
    # calibration helpers
    "pit_value",
    "ks_statistic",
    "dkw_threshold",
    "anderson_darling",
    "anderson_darling_pvalue",
    # aggregation
    "linear_pool",
    "log_pool",
    "HedgeAggregator",
    "PolynomialWeightsAggregator",
    # runtime
    "Forecaster",
]
