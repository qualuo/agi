"""ConformalPredictor — distribution-free, finite-sample-valid prediction
intervals as a runtime primitive.

Every other forecaster in this runtime emits a point estimate plus, at
best, a Gaussian-flavoured standard error. `PreflightEstimator` says
"this ticket will cost $0.42 ± $0.08"; `CalibrationEngine` says "the
forecaster's `p_success` is well-calibrated in aggregate." Neither of
those gives a coordination engine what it actually needs to make an
admission decision under hard constraints:

    "with 95% probability, regardless of the data distribution, this
     ticket will cost no more than X."

Asymptotic CIs cannot deliver that. They depend on Gaussian assumptions
that break for fat-tailed costs, on plug-in variances that are wrong
when the underlying forecaster is biased, and on enough data to make
the asymptotics kick in.

`ConformalPredictor` does deliver that. It implements **conformal
prediction** (Vovk-Gammerman-Shafer 2005) — a distribution-free,
finite-sample-valid framework that wraps *any* base forecaster and
produces prediction sets with provable marginal coverage:

    P( y_test ∈ Ĉ(x_test) ) ≥ 1 − α     for any distribution of (x, y).

The only assumption is **exchangeability** between calibration and
test points. That is dramatically weaker than i.i.d., and it holds
naturally for the receipts a runtime emits during steady-state
operation. When it stops holding (regime change, new model rolled
out), we detect drift and refit.

What this module implements
---------------------------

Split / inductive conformal (Papadopoulos 2008)
    Fit a base predictor on a proper training set, score it on a held-
    out calibration set, take the empirical (1-α) quantile of the
    nonconformity scores as a threshold, and return the test-point
    prediction set induced by that threshold. Marginal coverage holds
    exactly for any exchangeable distribution.

Conformalized Quantile Regression — CQR (Romano-Patterson-Candès 2019)
    Use quantile-regression predictions of the α/2 and 1-α/2 levels as
    the base, score by the worst-side miss, and inflate symmetrically.
    Heteroscedasticity-aware: intervals shrink where the conditional
    response is concentrated and widen where it is dispersed. The
    state of the art for *adaptive* width.

Mondrian conformal prediction (Vovk-Lindsay-Nouretdinov-Gammerman 2003)
    Partition the calibration set by group (tenant / model / task
    class) and apply split conformal separately per group. Trades
    marginal coverage for group-conditional coverage so a coordination
    engine can guarantee no tenant is systematically under-covered.

Adaptive conformal inference — ACI (Gibbs-Candès 2021)
    Online learning rate on α: update α_{t+1} = α_t + γ(α* − err_t),
    where err_t = 1{y_t ∉ Ĉ_t}. Recovers long-run coverage even under
    arbitrary distribution shift. The runtime can run this as a
    *guardrail* on top of split conformal: when ACI drives α below
    nominal, raise a CONFORMAL_DRIFT event so the coordinator knows
    the calibration set has gone stale.

Jackknife+ (Barber-Candès-Ramdas-Tibshirani 2021)
    Leave-one-out aggregate over a single training set. Coverage
    bound of 1 − 2α (vs. 1 − α for split) but no calibration/training
    split, so you keep your sample. Useful when n is small.

Classification: regularized adaptive prediction sets — RAPS (Angelopoulos-Bates-Jordan-Malik 2021)
    For multi-class problems where the base forecaster emits a score
    vector, build sets by sorting and accumulating until the
    cumulative softmax mass exceeds a conformal threshold. Tight
    intervals on confident points; conservative ones on hard points.

Diagnostics
    Empirical marginal and group-conditional coverage on a held-out
    sample (`measure_coverage`), mean and median interval widths,
    coverage gap vs. nominal. The coverage gap is what a coordination
    engine actually monitors.

Surface a coordination engine drives
------------------------------------

    cp = ConformalPredictor(target_coverage=0.95)

    # warm-up: drain the receipt stream into the calibration set
    for r in driver.tickets():
        cp.record(
            features={"model": r.model, "estimated_cost": r.estimated_cost_usd},
            prediction=r.estimated_cost_usd,
            outcome=r.actual_cost_usd,
            group=r.tenant_id,
        )

    # query: what is the 95% upper bound on cost for this candidate?
    pi = cp.predict_interval(
        features={"model": "claude-opus-4-7", "estimated_cost": 0.42},
        prediction=0.42,
        method="cqr",
        group="tenant-a",
    )
    if pi.upper > tenant.budget_remaining:
        coordinator.defer(ticket)

    # online guardrail: detect drift via ACI
    cp.update_adaptive(outcome=actual_cost, last_interval=pi)
    if cp.report().drift_detected:
        coordinator.pause_forecaster("preflight")

Events
    conformal.observed   — one (features, prediction, outcome) recorded
    conformal.fit        — calibrator refit (n samples used)
    conformal.predicted  — one interval/set returned (sampled)
    conformal.drift      — ACI α drifted outside band
    conformal.report     — periodic coverage report

Honest about limits
-------------------

Exchangeability is the assumption. If receipts during the morning are
systematically different from receipts at midnight, marginal coverage
still holds *over the whole day* but conditional coverage on time-of-
day will not. The Mondrian variant lets you swap marginal for
conditional on any group you can name. The ACI variant lets you keep
long-run marginal coverage even when exchangeability breaks; it
cannot promise it on any finite window.

The lab is stdlib-only. All quantile / inverse operations are exact
on the empirical CDF; no sampling, no numerical optimisation.
"""
from __future__ import annotations

import bisect
import json
import math
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# ----- event kinds ----------------------------------------------------

CONFORMAL_OBSERVED = "conformal.observed"
CONFORMAL_FIT = "conformal.fit"
CONFORMAL_PREDICTED = "conformal.predicted"
CONFORMAL_DRIFT = "conformal.drift"
CONFORMAL_REPORT = "conformal.report"


# ----- methods --------------------------------------------------------

METHOD_SPLIT = "split"           # absolute-residual split conformal
METHOD_CQR = "cqr"               # conformalized quantile regression
METHOD_MONDRIAN = "mondrian"     # group-conditional split
METHOD_JACKKNIFE_PLUS = "jk+"    # Barber-Candès-Ramdas-Tibshirani
METHOD_RAPS = "raps"             # regularized adaptive prediction sets

KNOWN_REGRESSION_METHODS = (
    METHOD_SPLIT,
    METHOD_CQR,
    METHOD_MONDRIAN,
    METHOD_JACKKNIFE_PLUS,
)
KNOWN_CLASSIFICATION_METHODS = (METHOD_RAPS,)


# ----- numerical primitives ------------------------------------------


def _empirical_quantile_ceiling(values: Sequence[float], level: float) -> float:
    """Conformal quantile: ⌈(n+1)·level⌉ / n empirical quantile.

    This is the *exact* finite-sample threshold that gives marginal
    coverage ≥ level. Different from the standard empirical quantile
    by a +1 in the numerator — the +1 accounts for the test point.
    Vovk 2012, Lei-G'Sell-Rinaldo-Tibshirani-Wasserman 2018.
    """
    n = len(values)
    if n == 0:
        return math.inf
    if not 0.0 < level < 1.0:
        if level >= 1.0:
            return max(values)
        return min(values)
    sorted_values = sorted(values)
    # rank index = ⌈(n+1) * level⌉, clipped to [1, n], then 0-indexed.
    rank = math.ceil((n + 1) * level)
    if rank > n:
        # No finite threshold gives the requested coverage with n samples;
        # return +inf so the prediction set is vacuous (correctly conservative).
        return math.inf
    return sorted_values[rank - 1]


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _hash_features(features: Mapping[str, Any]) -> str:
    """Stable feature fingerprint for caching."""
    try:
        return json.dumps(features, sort_keys=True, default=str)
    except Exception:
        return repr(sorted(features.items())) if features else ""


# ----- dataclasses ---------------------------------------------------


@dataclass(frozen=True)
class CalibrationPoint:
    """One (features, base prediction, realized outcome) triple.

    Regression: `prediction` is the base point forecast and `outcome`
    is the realized response.

    CQR: `prediction` is a 2-tuple (lower_q, upper_q) of quantile-
    regression predictions; pass it through `prediction_lo` / `prediction_hi`.

    Classification: `prediction` is a mapping label → score (need not
    be normalized) and `outcome` is the true label as a string.
    """
    features: Mapping[str, Any]
    outcome: Any
    prediction: Any = None
    prediction_lo: float | None = None
    prediction_hi: float | None = None
    group: str = ""
    ts: float = field(default_factory=time.time)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0.0 or not math.isfinite(self.weight):
            raise ValueError(f"weight must be finite ≥ 0, got {self.weight}")


@dataclass(frozen=True)
class PredictionInterval:
    """Conformal prediction interval for a regression query.

    `lower` ≤ `upper`. `target_coverage` is the requested marginal
    coverage; `method` is the estimator used; `width` is convenience.

    `n_cal` is the number of calibration points the threshold rests
    on; `effective_alpha` is the actual α the interval was built at
    (may differ from requested under adaptive conformal).
    """
    lower: float
    upper: float
    target_coverage: float
    method: str
    n_cal: int
    effective_alpha: float
    point: float | None = None
    group: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, y: float) -> bool:
        return self.lower <= y <= self.upper

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionSet:
    """Conformal prediction set for a classification query.

    `labels` is the sorted tuple of in-set labels. `coverage` is the
    requested level. `scores` keeps the raw score for each in-set
    label for downstream introspection.
    """
    labels: tuple[str, ...]
    target_coverage: float
    method: str
    n_cal: int
    scores: dict[str, float] = field(default_factory=dict)
    group: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.labels)

    def contains(self, y: str) -> bool:
        return y in self.labels

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageReport:
    n: int
    target_coverage: float
    empirical_coverage: float
    mean_width: float
    median_width: float
    coverage_gap: float
    drift_detected: bool
    method: str
    per_group: dict[str, "GroupCoverage"] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_group"] = {k: asdict(v) for k, v in self.per_group.items()}
        return d


@dataclass(frozen=True)
class GroupCoverage:
    group: str
    n: int
    empirical_coverage: float
    mean_width: float


# ----- core: split conformal regression ------------------------------


def _residuals_absolute(points: Sequence[CalibrationPoint]) -> list[float]:
    out = []
    for p in points:
        if p.prediction is None:
            raise ValueError("split conformal needs a base prediction on every sample")
        try:
            yhat = float(p.prediction)
            y = float(p.outcome)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric sample: {exc}") from exc
        out.append(abs(y - yhat))
    return out


def _residuals_cqr(points: Sequence[CalibrationPoint]) -> list[float]:
    """CQR nonconformity: max(lo - y, y - hi). Negative ⇒ inside the
    quantile band, positive ⇒ outside on one side.

    Romano-Patterson-Candès 2019, eq. (5).
    """
    out = []
    for p in points:
        if p.prediction_lo is None or p.prediction_hi is None:
            raise ValueError("CQR needs prediction_lo and prediction_hi on every sample")
        y = float(p.outcome)
        out.append(max(p.prediction_lo - y, y - p.prediction_hi))
    return out


def _split_threshold(residuals: Sequence[float], target_coverage: float) -> float:
    return _empirical_quantile_ceiling(residuals, target_coverage)


# ----- jackknife+ ----------------------------------------------------


def _jackknife_plus_interval(
    loo_predictions: Sequence[float],
    loo_residuals: Sequence[float],
    target_coverage: float,
) -> tuple[float, float]:
    """Jackknife+ interval from leave-one-out predictions + residuals.

    Barber-Candès-Ramdas-Tibshirani 2021, eq. (4): the lower endpoint
    is the (⌊α(n+1)⌋)-th order statistic of (μ̂_{−i}(x) − R_i), and
    the upper is the (⌈(1-α)(n+1)⌉)-th of (μ̂_{−i}(x) + R_i).
    """
    n = len(loo_predictions)
    if n != len(loo_residuals):
        raise ValueError("jk+ predictions and residuals must align")
    if n == 0:
        return (-math.inf, math.inf)
    alpha = 1.0 - target_coverage
    lows = sorted(p - r for p, r in zip(loo_predictions, loo_residuals))
    highs = sorted(p + r for p, r in zip(loo_predictions, loo_residuals))
    lo_rank = max(1, math.floor(alpha * (n + 1)))
    hi_rank = min(n, math.ceil((1.0 - alpha) * (n + 1)))
    return lows[lo_rank - 1], highs[hi_rank - 1]


# ----- RAPS classification -------------------------------------------


def _raps_nonconformity(
    scores: Mapping[str, float],
    true_label: str,
    *,
    k_reg: int = 1,
    lam: float = 0.01,
    rng_eps: float = 0.0,
) -> float:
    """Regularized Adaptive Prediction Sets nonconformity score.

    Angelopoulos-Bates-Jordan-Malik 2021. Sort softmax-normalized
    scores in decreasing order, walk down the ranking, accumulate
    softmax mass + a regularization term (lam·max(0, rank−k_reg+1)),
    and stop at the true label. Higher score ⇒ more atypical.
    """
    # softmax-normalize so scores compose meaningfully across samples.
    vals = list(scores.values())
    if not vals:
        return math.inf
    m = max(vals)
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    sm = {k: exps[k] / z for k in exps}
    if true_label not in sm:
        return math.inf
    ranked = sorted(sm.items(), key=lambda kv: kv[1], reverse=True)
    cum = 0.0
    for i, (lab, p) in enumerate(ranked):
        cum += p + lam * max(0, i + 1 - k_reg)
        if lab == true_label:
            # randomized tie-breaking is optional; deterministic by default.
            return cum - rng_eps * p
    return cum


def _raps_predict_set(
    scores: Mapping[str, float],
    threshold: float,
    *,
    k_reg: int = 1,
    lam: float = 0.01,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Return labels whose RAPS cumulative score ≤ threshold."""
    if not scores:
        return (), {}
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    sm = {k: exps[k] / z for k in exps}
    ranked = sorted(sm.items(), key=lambda kv: kv[1], reverse=True)
    in_set: list[str] = []
    in_scores: dict[str, float] = {}
    cum = 0.0
    for i, (lab, p) in enumerate(ranked):
        cum += p + lam * max(0, i + 1 - k_reg)
        in_set.append(lab)
        in_scores[lab] = p
        if cum >= threshold:
            break
    return tuple(sorted(in_set)), in_scores


# ----- adaptive conformal (Gibbs-Candès 2021) ------------------------


@dataclass
class _ACIState:
    """State for Adaptive Conformal Inference.

    α_t evolves to maintain long-run coverage:
        α_{t+1} = α_t + γ (α* − err_t),   err_t ∈ {0, 1}.

    γ controls how aggressively we react; α_t lives in (0, 1) and the
    *effective* coverage at time t is 1 − α_t. We clamp into a sane
    band to avoid degenerate intervals.
    """
    alpha_star: float
    alpha_t: float
    gamma: float = 0.05
    band_lo: float = 0.001
    band_hi: float = 0.999
    history: list[bool] = field(default_factory=list)  # err_t per step

    def update(self, miss: bool) -> None:
        err = 1.0 if miss else 0.0
        self.alpha_t = _clip(
            self.alpha_t + self.gamma * (self.alpha_star - err),
            self.band_lo,
            self.band_hi,
        )
        self.history.append(miss)
        if len(self.history) > 10000:
            self.history = self.history[-5000:]


# ----- main predictor ------------------------------------------------


class ConformalPredictor:
    """Distribution-free, finite-sample-valid prediction intervals.

    Build one per (response-type, group-policy) pair. Calibration is
    cheap (O(n log n)) and incremental — record() appends to a ring
    buffer; predict_*() reads the current state.

    Thread-safe via a coarse lock; reads return immutable dataclasses.
    """

    def __init__(
        self,
        *,
        target_coverage: float = 0.9,
        max_history: int = 10000,
        bus: EventBus | None = None,
        adaptive: bool = False,
        adaptive_gamma: float = 0.05,
        drift_threshold: float = 0.05,
        seed: int = 0,
    ) -> None:
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(f"target_coverage must be in (0,1), got {target_coverage}")
        if max_history < 1:
            raise ValueError("max_history must be positive")

        self.target_coverage = float(target_coverage)
        self.max_history = int(max_history)
        self.bus = bus
        self.drift_threshold = float(drift_threshold)
        self.seed = int(seed)

        self._lock = threading.RLock()
        self._points: list[CalibrationPoint] = []
        self._aci: _ACIState | None = None
        if adaptive:
            self._aci = _ACIState(
                alpha_star=1.0 - self.target_coverage,
                alpha_t=1.0 - self.target_coverage,
                gamma=float(adaptive_gamma),
            )

        # Coverage stream for drift detection — appended on update_adaptive
        # OR when measure_coverage is called with a held-out slice. The
        # detector compares ECE on the recent tail to the bulk.
        self._coverage_stream: list[bool] = []  # True = hit (covered)
        self._drift_flag = False

    # ----- recording -------------------------------------------------

    def record(
        self,
        *,
        features: Mapping[str, Any] | None = None,
        prediction: Any = None,
        outcome: Any = None,
        prediction_lo: float | None = None,
        prediction_hi: float | None = None,
        group: str = "",
        weight: float = 1.0,
    ) -> None:
        point = CalibrationPoint(
            features=dict(features or {}),
            outcome=outcome,
            prediction=prediction,
            prediction_lo=prediction_lo,
            prediction_hi=prediction_hi,
            group=group,
            weight=float(weight),
        )
        with self._lock:
            self._points.append(point)
            if len(self._points) > self.max_history:
                # ring-buffer: drop oldest so exchangeability of the
                # *current* window holds against a stationary regime.
                self._points = self._points[-self.max_history :]
        if self.bus is not None:
            self.bus.publish(Event(
                kind=CONFORMAL_OBSERVED,
                data={
                    "group": group,
                    "has_outcome": outcome is not None,
                    "n_total": len(self._points),
                },
            ))

    def record_many(self, points: Iterable[CalibrationPoint]) -> None:
        with self._lock:
            for p in points:
                self._points.append(p)
            if len(self._points) > self.max_history:
                self._points = self._points[-self.max_history :]

    def __len__(self) -> int:
        with self._lock:
            return len(self._points)

    # ----- prediction: regression ------------------------------------

    def predict_interval(
        self,
        *,
        prediction: float | None = None,
        prediction_lo: float | None = None,
        prediction_hi: float | None = None,
        features: Mapping[str, Any] | None = None,
        method: str = METHOD_SPLIT,
        group: str = "",
        loo_predictor: Callable[[Sequence[CalibrationPoint], Mapping[str, Any]], tuple[float, float]] | None = None,
    ) -> PredictionInterval:
        """Return a prediction interval for a single test point.

        method=split:     prediction is the base point forecast.
        method=cqr:       prediction_lo / prediction_hi are quantile
                          forecasts at α/2 and 1−α/2.
        method=mondrian:  same as split but threshold uses only the
                          calibration points with the same `group`.
        method=jk+:       `loo_predictor(samples, features)` should
                          return (mean_prediction, residual_sample)
                          for the test point. Costly; use sparingly.
        """
        if method not in KNOWN_REGRESSION_METHODS:
            raise ValueError(f"unknown regression method {method!r}")

        with self._lock:
            points = list(self._points)
            alpha = (
                self._aci.alpha_t
                if (self._aci is not None and method != METHOD_JACKKNIFE_PLUS)
                else 1.0 - self.target_coverage
            )

        effective_coverage = 1.0 - alpha

        if method == METHOD_MONDRIAN:
            relevant = [p for p in points if p.group == group]
        else:
            relevant = points

        n = len(relevant)

        if method in (METHOD_SPLIT, METHOD_MONDRIAN):
            if prediction is None:
                raise ValueError("split/mondrian require a `prediction` argument")
            residuals = _residuals_absolute(relevant)
            q = _split_threshold(residuals, effective_coverage)
            lo, hi = prediction - q, prediction + q
        elif method == METHOD_CQR:
            if prediction_lo is None or prediction_hi is None:
                raise ValueError("cqr requires prediction_lo and prediction_hi")
            residuals = _residuals_cqr(relevant)
            q = _split_threshold(residuals, effective_coverage)
            lo, hi = prediction_lo - q, prediction_hi + q
        elif method == METHOD_JACKKNIFE_PLUS:
            if loo_predictor is None:
                raise ValueError("jk+ requires a `loo_predictor` callable")
            loo_preds: list[float] = []
            loo_resid: list[float] = []
            for i in range(n):
                rest = relevant[:i] + relevant[i + 1 :]
                yhat_test, r_i = loo_predictor(rest, dict(features or {}))
                loo_preds.append(yhat_test)
                loo_resid.append(r_i)
            lo, hi = _jackknife_plus_interval(loo_preds, loo_resid, effective_coverage)
        else:
            raise AssertionError(method)

        pi = PredictionInterval(
            lower=lo,
            upper=hi,
            target_coverage=self.target_coverage,
            method=method,
            n_cal=n,
            effective_alpha=alpha,
            point=prediction if prediction is not None else (
                0.5 * (prediction_lo + prediction_hi)
                if (prediction_lo is not None and prediction_hi is not None)
                else None
            ),
            group=group,
            diagnostics={
                "threshold": q if method != METHOD_JACKKNIFE_PLUS else None,
            },
        )
        if self.bus is not None:
            self.bus.publish(Event(
                kind=CONFORMAL_PREDICTED,
                data={
                    "method": method,
                    "lower": lo,
                    "upper": hi,
                    "width": hi - lo,
                    "n_cal": n,
                    "group": group,
                },
            ))
        return pi

    # ----- prediction: classification --------------------------------

    def predict_set(
        self,
        *,
        scores: Mapping[str, float],
        features: Mapping[str, Any] | None = None,
        method: str = METHOD_RAPS,
        group: str = "",
        k_reg: int = 1,
        lam: float = 0.01,
    ) -> PredictionSet:
        if method not in KNOWN_CLASSIFICATION_METHODS:
            raise ValueError(f"unknown classification method {method!r}")
        with self._lock:
            points = [p for p in self._points if isinstance(p.outcome, str)]
            relevant = [p for p in points if p.group == group] if group else points
            alpha = (
                self._aci.alpha_t if self._aci is not None
                else 1.0 - self.target_coverage
            )
        effective_coverage = 1.0 - alpha
        # Compute calibration nonconformity scores.
        cal_scores: list[float] = []
        for p in relevant:
            if not isinstance(p.prediction, Mapping):
                continue
            cal_scores.append(_raps_nonconformity(
                p.prediction, str(p.outcome), k_reg=k_reg, lam=lam,
            ))
        threshold = _empirical_quantile_ceiling(cal_scores, effective_coverage)
        labels, in_scores = _raps_predict_set(
            scores, threshold, k_reg=k_reg, lam=lam,
        )
        ps = PredictionSet(
            labels=labels,
            target_coverage=self.target_coverage,
            method=method,
            n_cal=len(cal_scores),
            scores=in_scores,
            group=group,
            diagnostics={"threshold": threshold},
        )
        if self.bus is not None:
            self.bus.publish(Event(
                kind=CONFORMAL_PREDICTED,
                data={
                    "method": method,
                    "size": len(labels),
                    "n_cal": len(cal_scores),
                    "group": group,
                },
            ))
        return ps

    # ----- adaptive online α update ----------------------------------

    def update_adaptive(
        self,
        *,
        outcome: float,
        last_interval: PredictionInterval,
    ) -> bool:
        """Feed one realized outcome to the adaptive α loop.

        Returns True if the loop adjusted α (i.e., adaptive is on).
        Also pushes a coverage observation into the drift detector.
        """
        miss = not last_interval.contains(outcome)
        with self._lock:
            self._coverage_stream.append(not miss)
            if len(self._coverage_stream) > self.max_history:
                self._coverage_stream = self._coverage_stream[-self.max_history :]
            if self._aci is not None:
                self._aci.update(miss)
                self._check_drift_locked()
                return True
            self._check_drift_locked()
            return False

    def _check_drift_locked(self) -> None:
        # Compare empirical coverage on recent tail to bulk.
        stream = self._coverage_stream
        if len(stream) < 50:
            return
        tail = stream[-min(200, len(stream) // 4 or 1):]
        bulk = stream[:-len(tail)] or stream
        tail_cov = sum(1 for x in tail if x) / len(tail)
        bulk_cov = sum(1 for x in bulk if x) / len(bulk)
        gap = abs(tail_cov - bulk_cov)
        triggered = gap > self.drift_threshold
        if triggered and not self._drift_flag:
            self._drift_flag = True
            if self.bus is not None:
                self.bus.publish(Event(
                    kind=CONFORMAL_DRIFT,
                    data={
                        "tail_coverage": tail_cov,
                        "bulk_coverage": bulk_cov,
                        "gap": gap,
                    },
                ))
        elif not triggered:
            self._drift_flag = False

    # ----- diagnostics -----------------------------------------------

    def measure_coverage(
        self,
        held_out: Sequence[CalibrationPoint],
        *,
        method: str = METHOD_SPLIT,
        loo_predictor: Callable[[Sequence[CalibrationPoint], Mapping[str, Any]], tuple[float, float]] | None = None,
    ) -> CoverageReport:
        """Empirical coverage of `held_out` under the current calibration.

        For regression methods we compute the prediction interval per
        held-out point and check coverage of its true outcome. We
        report marginal coverage, mean/median width, and per-group
        coverage so a coordination engine can spot any tenant being
        under-covered.
        """
        hits = 0
        widths: list[float] = []
        per_group: dict[str, list[bool]] = {}
        per_group_widths: dict[str, list[float]] = {}

        for p in held_out:
            if method == METHOD_CQR:
                pi = self.predict_interval(
                    prediction_lo=p.prediction_lo,
                    prediction_hi=p.prediction_hi,
                    method=METHOD_CQR,
                    group=p.group,
                    features=p.features,
                )
            elif method == METHOD_JACKKNIFE_PLUS:
                pi = self.predict_interval(
                    method=METHOD_JACKKNIFE_PLUS,
                    group=p.group,
                    features=p.features,
                    loo_predictor=loo_predictor,
                )
            else:
                pi = self.predict_interval(
                    prediction=float(p.prediction) if p.prediction is not None else 0.0,
                    method=method,
                    group=p.group,
                    features=p.features,
                )
            y = float(p.outcome)
            covered = pi.contains(y)
            hits += int(covered)
            widths.append(pi.width)
            per_group.setdefault(p.group, []).append(covered)
            per_group_widths.setdefault(p.group, []).append(pi.width)

        n = len(held_out)
        emp = hits / n if n > 0 else 0.0
        mean_w = statistics.mean(widths) if widths else 0.0
        med_w = statistics.median(widths) if widths else 0.0
        gap = self.target_coverage - emp

        groups = {}
        for g, hits_list in per_group.items():
            gn = len(hits_list)
            ws = per_group_widths.get(g, [])
            groups[g] = GroupCoverage(
                group=g,
                n=gn,
                empirical_coverage=(sum(1 for h in hits_list if h) / gn) if gn else 0.0,
                mean_width=(statistics.mean(ws) if ws else 0.0),
            )

        notes = []
        if n < 30:
            notes.append("low_n: held-out set is small; bound is noisy")

        report = CoverageReport(
            n=n,
            target_coverage=self.target_coverage,
            empirical_coverage=emp,
            mean_width=mean_w,
            median_width=med_w,
            coverage_gap=gap,
            drift_detected=self._drift_flag,
            method=method,
            per_group=groups,
            notes=tuple(notes),
        )
        if self.bus is not None:
            self.bus.publish(Event(
                kind=CONFORMAL_REPORT,
                data={
                    "n": n,
                    "empirical_coverage": emp,
                    "coverage_gap": gap,
                    "drift_detected": self._drift_flag,
                },
            ))
        return report

    def report(self) -> CoverageReport:
        """Quick self-report based on the streaming coverage history.

        Does *not* refit anything; only reads the adaptive miss stream.
        Use measure_coverage(held_out) when a real held-out set is
        available (preferred for any guardrail).
        """
        with self._lock:
            stream = list(self._coverage_stream)
            drift = self._drift_flag
        n = len(stream)
        emp = sum(1 for x in stream if x) / n if n > 0 else 0.0
        gap = self.target_coverage - emp
        return CoverageReport(
            n=n,
            target_coverage=self.target_coverage,
            empirical_coverage=emp,
            mean_width=0.0,
            median_width=0.0,
            coverage_gap=gap,
            drift_detected=drift,
            method="stream",
        )

    # ----- integrations ----------------------------------------------

    def attach_to_driver(
        self,
        driver: Any,
        *,
        prediction_of: Callable[[Any], float] | None = None,
        outcome_of: Callable[[Any], float] | None = None,
        features_of: Callable[[Any], Mapping[str, Any]] | None = None,
        group_of: Callable[[Any], str] | None = None,
    ) -> Callable[[], None]:
        """Drain a RuntimeDriver's receipt stream into the calibrator.

        Defaults: prediction=estimated_cost_usd, outcome=actual_cost_usd,
        features={model, intent, tenant_id, estimated_p_success},
        group=tenant_id. Override per domain.
        """
        prediction_of = prediction_of or (lambda r: float(getattr(r, "estimated_cost_usd", 0.0)))
        outcome_of = outcome_of or (lambda r: float(getattr(r, "actual_cost_usd", 0.0)))
        features_of = features_of or (lambda r: {
            "model": getattr(r, "model", None),
            "intent": getattr(r, "intent", None),
            "tenant_id": getattr(r, "tenant_id", None),
            "estimated_p_success": getattr(r, "estimated_p_success", None),
        })
        group_of = group_of or (lambda r: str(getattr(r, "tenant_id", "") or ""))

        def listener(receipt: Any) -> None:
            try:
                self.record(
                    features=features_of(receipt),
                    prediction=prediction_of(receipt),
                    outcome=outcome_of(receipt),
                    group=group_of(receipt),
                )
            except Exception:
                pass

        if hasattr(driver, "subscribe_receipts"):
            return driver.subscribe_receipts(listener)
        if hasattr(driver, "subscribe"):
            return driver.subscribe(listener)
        if hasattr(driver, "tickets"):
            for r in driver.tickets():
                listener(r)
        return lambda: None

    def attach_to_bus(
        self,
        bus: EventBus,
        *,
        kinds: Sequence[str] = (),
        extractor: Callable[[Event], CalibrationPoint | None] | None = None,
    ) -> Callable[[], None]:
        """Subscribe to an EventBus and turn events into CalibrationPoints.

        If `extractor` is None, events whose `.data` carries the keys
        {features, prediction, outcome} (plus optional group) are
        recorded directly.
        """
        def cb(event: Event) -> None:
            try:
                if extractor:
                    p = extractor(event)
                    if p is not None:
                        with self._lock:
                            self._points.append(p)
                            if len(self._points) > self.max_history:
                                self._points = self._points[-self.max_history :]
                    return
                d = event.data or {}
                if "outcome" not in d:
                    return
                self.record(
                    features=d.get("features") or {},
                    prediction=d.get("prediction"),
                    outcome=d.get("outcome"),
                    prediction_lo=d.get("prediction_lo"),
                    prediction_hi=d.get("prediction_hi"),
                    group=str(d.get("group", "")),
                )
            except Exception:
                pass

        if not kinds:
            sid = bus.subscribe(cb)
            return lambda: bus.unsubscribe(sid)
        sids = [bus.subscribe(cb, kind=k) for k in kinds]
        return lambda: [bus.unsubscribe(s) for s in sids]

    # ----- persistence -----------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "target_coverage": self.target_coverage,
                "max_history": self.max_history,
                "drift_threshold": self.drift_threshold,
                "seed": self.seed,
                "points": [
                    {
                        "features": dict(p.features),
                        "outcome": p.outcome,
                        "prediction": p.prediction,
                        "prediction_lo": p.prediction_lo,
                        "prediction_hi": p.prediction_hi,
                        "group": p.group,
                        "ts": p.ts,
                        "weight": p.weight,
                    }
                    for p in self._points
                ],
                "coverage_stream": list(self._coverage_stream),
                "drift_flag": self._drift_flag,
                "aci": (
                    {
                        "alpha_star": self._aci.alpha_star,
                        "alpha_t": self._aci.alpha_t,
                        "gamma": self._aci.gamma,
                        "band_lo": self._aci.band_lo,
                        "band_hi": self._aci.band_hi,
                        "history": list(self._aci.history),
                    }
                    if self._aci is not None
                    else None
                ),
            }

    def restore(self, snap: Mapping[str, Any]) -> None:
        if snap.get("version") != 1:
            raise ValueError("unknown snapshot version")
        with self._lock:
            self.target_coverage = float(snap.get("target_coverage", self.target_coverage))
            self.max_history = int(snap.get("max_history", self.max_history))
            self.drift_threshold = float(snap.get("drift_threshold", self.drift_threshold))
            self.seed = int(snap.get("seed", self.seed))
            self._points = []
            for d in snap.get("points", []):
                self._points.append(CalibrationPoint(
                    features=dict(d.get("features") or {}),
                    outcome=d.get("outcome"),
                    prediction=d.get("prediction"),
                    prediction_lo=d.get("prediction_lo"),
                    prediction_hi=d.get("prediction_hi"),
                    group=str(d.get("group", "")),
                    ts=float(d.get("ts", time.time())),
                    weight=float(d.get("weight", 1.0)),
                ))
            self._coverage_stream = list(snap.get("coverage_stream") or [])
            self._drift_flag = bool(snap.get("drift_flag", False))
            aci_d = snap.get("aci")
            if aci_d is not None:
                self._aci = _ACIState(
                    alpha_star=float(aci_d["alpha_star"]),
                    alpha_t=float(aci_d["alpha_t"]),
                    gamma=float(aci_d.get("gamma", 0.05)),
                    band_lo=float(aci_d.get("band_lo", 0.001)),
                    band_hi=float(aci_d.get("band_hi", 0.999)),
                    history=list(aci_d.get("history", [])),
                )
            else:
                self._aci = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.snapshot()))

    def load(self, path: str | Path) -> None:
        self.restore(json.loads(Path(path).read_text()))
