"""CalibrationEngine — turn raw `p_success` forecasts into trustworthy ones.

Every layer of this runtime emits probabilities: `PreflightEstimator`
returns `p_success`, `TicketOracle` produces counterfactual `p_success`
grids, `TicketSLO.hedged_p_success` composes them. A coordination engine
takes those probabilities and does expected-value math:

    EV = p_success * payoff - (1 - p_success) * refund_cost
    admit iff EV >= 0

That math is only as honest as the probability. A forecaster that says
"0.9" when the empirical rate is 0.6 will produce systematically wrong
admission, hedging, and budget decisions — and nothing else in the
runtime will notice, because every consumer trusts the forecast at
face value.

`CalibrationEngine` closes that loop. It observes (forecast, outcome)
pairs as the receipt stream arrives, fits a monotone recalibration map,
and exposes `calibrate(p_raw) -> p_corrected` for upstream consumers.
It also reports Brier, log loss, ECE, MCE, a reliability diagram, and
a drift score so the coordination engine knows *how much* to trust the
forecaster — not just what the latest number is.

Design choices:

  - **Isotonic regression** by default — non-parametric, makes no
    distributional assumption, monotonic by construction. Pool
    Adjacent Violators (PAV) is exact, deterministic, CPU-only, and
    needs no extra dependency.
  - **Platt scaling** as an alternative for small N where isotonic
    over-fits. A two-parameter logistic fit by Newton-Raphson with
    ridge regularization.
  - **Per-source / per-bucket sub-calibrators**: different forecasters
    (preflight vs. oracle vs. a learned model) and different task
    classes can drift in different directions. The engine maintains
    one calibrator per (source, bucket); reports aggregate up.
  - **Drift detection**: ECE on a rolling tail of recent observations
    compared against ECE on the bulk. Rising-edge transitions into the
    drifted state raise a `CAL_DRIFT` event so the coordinator can
    pause / refit / route around the offending forecaster. Persistent
    drift does not re-emit; recovery resets the flag.
  - **Snapshot / restore** so a long-running runtime can persist
    calibration across restarts without re-burning history.

Surface:

  CalibrationSample      — one (forecast, outcome) observation
  ReliabilityBin         — one row of a reliability diagram
  CalibrationReport      — Brier, log loss, ECE, MCE, bins, drift, ok
  IsotonicCalibrator     — PAV regression with linear interpolation
  PlattCalibrator        — logistic recalibration via Newton-Raphson
  CalibrationEngine      — public face; observe / calibrate / report
  attach_to_driver       — wire to a RuntimeDriver's receipt stream
  attach_to_bus          — wire to an EventBus for ad-hoc events

Events (string constants so external consumers can match without import):

  CAL_OBSERVED           — one (forecast, outcome) recorded
  CAL_FIT                — calibrator refit (n samples used)
  CAL_DRIFT              — drift threshold breached
  CAL_REPORT             — periodic report emitted

Honest about limits: calibration corrects *systematic* miscalibration
(over/under-confidence). It does not improve a forecaster's resolution
— if the raw signal cannot tell easy tasks from hard ones, no monotone
map will fix that. Brier decomposes as reliability + resolution +
irreducible noise; this engine moves the reliability term toward zero
and reports the rest honestly so it shows up on a dashboard.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from agi.events import Event, EventBus


# ----- event kinds ----------------------------------------------------

CAL_OBSERVED = "calibration.observed"
CAL_FIT = "calibration.fit"
CAL_DRIFT = "calibration.drift"
CAL_REPORT = "calibration.report"


# ----- methods --------------------------------------------------------

METHOD_ISOTONIC = "isotonic"
METHOD_PLATT = "platt"
METHOD_IDENTITY = "identity"

KNOWN_METHODS = (METHOD_ISOTONIC, METHOD_PLATT, METHOD_IDENTITY)


# Numerical clamp to keep log loss and logistic fits well-defined.
_EPS = 1e-6


# ----- dataclasses ----------------------------------------------------


@dataclass(frozen=True)
class CalibrationSample:
    """One (forecast, outcome) observation.

    `p_forecast` is the raw probability the runtime emitted before this
    engine touched it. `outcome` is the ground truth (True = success).
    `source` identifies the forecaster (e.g. "preflight", "oracle"),
    `bucket` an optional segment (e.g. task class), and `weight` lets a
    coordinator weight by economic importance (cost, EV, refund) so the
    fit prioritizes the decisions that matter most.
    """

    p_forecast: float
    outcome: bool
    ts: float = field(default_factory=time.time)
    source: str = ""
    bucket: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_forecast <= 1.0:
            raise ValueError(f"p_forecast {self.p_forecast} out of [0,1]")
        if self.weight < 0.0 or not math.isfinite(self.weight):
            raise ValueError(f"weight must be >=0 finite, got {self.weight}")


@dataclass(frozen=True)
class ReliabilityBin:
    p_lo: float
    p_hi: float
    forecast_mean: float
    empirical_rate: float
    n: int
    weight_sum: float


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    weight_sum: float
    brier: float
    log_loss: float
    ece: float
    mce: float
    bins: tuple[ReliabilityBin, ...]
    method: str
    drift_score: float
    ok: bool
    source: str = ""
    bucket: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bins"] = [asdict(b) for b in self.bins]
        return d


# ----- metrics --------------------------------------------------------


def brier_score(samples: Iterable[CalibrationSample]) -> float:
    """Weighted Brier score: mean squared error of the forecast.

    0 is perfect, 0.25 is uninformed (always predicting 0.5), 1 is
    maximally wrong. Lower is better.
    """
    total = 0.0
    wsum = 0.0
    for s in samples:
        y = 1.0 if s.outcome else 0.0
        total += s.weight * (s.p_forecast - y) ** 2
        wsum += s.weight
    return total / wsum if wsum > 0 else 0.0


def log_loss(samples: Iterable[CalibrationSample]) -> float:
    """Weighted binary cross-entropy. Clamped to avoid -inf at p∈{0,1}."""
    total = 0.0
    wsum = 0.0
    for s in samples:
        p = min(max(s.p_forecast, _EPS), 1.0 - _EPS)
        ll = math.log(p) if s.outcome else math.log(1.0 - p)
        total += -s.weight * ll
        wsum += s.weight
    return total / wsum if wsum > 0 else 0.0


def reliability_bins(
    samples: list[CalibrationSample],
    *,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Equal-width binning over [0,1]. Empty bins are dropped."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets: list[list[CalibrationSample]] = [[] for _ in range(n_bins)]
    for s in samples:
        # Right-closed final bin to include p=1.0.
        idx = min(int(s.p_forecast * n_bins), n_bins - 1)
        buckets[idx].append(s)
    out: list[ReliabilityBin] = []
    for i, bs in enumerate(buckets):
        if not bs:
            continue
        wsum = sum(s.weight for s in bs)
        if wsum <= 0:
            continue
        fm = sum(s.weight * s.p_forecast for s in bs) / wsum
        em = sum(s.weight * (1.0 if s.outcome else 0.0) for s in bs) / wsum
        out.append(
            ReliabilityBin(
                p_lo=edges[i],
                p_hi=edges[i + 1],
                forecast_mean=fm,
                empirical_rate=em,
                n=len(bs),
                weight_sum=wsum,
            )
        )
    return out


def expected_calibration_error(bins: list[ReliabilityBin]) -> float:
    """Weighted average of |forecast - empirical| per bin."""
    total_w = sum(b.weight_sum for b in bins)
    if total_w <= 0:
        return 0.0
    return sum(b.weight_sum * abs(b.forecast_mean - b.empirical_rate) for b in bins) / total_w


def max_calibration_error(bins: list[ReliabilityBin], min_weight: float = 0.0) -> float:
    """Max |forecast - empirical| over bins with at least `min_weight`."""
    gaps = [abs(b.forecast_mean - b.empirical_rate) for b in bins if b.weight_sum >= min_weight]
    return max(gaps) if gaps else 0.0


# ----- calibrators ----------------------------------------------------


class IsotonicCalibrator:
    """Pool Adjacent Violators (PAV) isotonic regression.

    Fits the unique monotone non-decreasing step function that minimizes
    the weighted sum of squared deviations from (p_forecast, outcome).
    Linear interpolation between block midpoints; flat extrapolation
    outside the observed range. CPU-only, deterministic, exact.
    """

    method = METHOD_ISOTONIC

    def __init__(self) -> None:
        # Sorted x positions (block midpoints) and matching y values.
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._fitted = False

    def fit(self, samples: list[CalibrationSample]) -> None:
        if not samples:
            self._xs, self._ys, self._fitted = [], [], False
            return
        # Sort by raw forecast ascending.
        ordered = sorted(samples, key=lambda s: s.p_forecast)
        # Pool adjacent violators: each block (x, y, w) where y is the
        # weighted mean of outcomes and x is the weighted mean of forecasts.
        blocks: list[list[float]] = []  # [x, y, w]
        for s in ordered:
            y = 1.0 if s.outcome else 0.0
            blocks.append([s.p_forecast, y, s.weight])
            # Merge backwards while monotonicity is violated.
            while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
                a = blocks.pop()
                b = blocks.pop()
                w = a[2] + b[2]
                if w <= 0:
                    # Both zero-weight — keep the later one to preserve order.
                    blocks.append(a)
                    continue
                merged_x = (a[0] * a[2] + b[0] * b[2]) / w
                merged_y = (a[1] * a[2] + b[1] * b[2]) / w
                blocks.append([merged_x, merged_y, w])
        self._xs = [b[0] for b in blocks]
        self._ys = [b[1] for b in blocks]
        self._fitted = True

    def adjust(self, p_raw: float) -> float:
        if not self._fitted or not self._xs:
            return p_raw
        p_raw = max(0.0, min(1.0, p_raw))
        xs, ys = self._xs, self._ys
        if p_raw <= xs[0]:
            return ys[0]
        if p_raw >= xs[-1]:
            return ys[-1]
        # Linear interpolation between adjacent blocks.
        i = bisect.bisect_right(xs, p_raw)
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        if x1 == x0:
            return y0
        t = (p_raw - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def snapshot(self) -> dict[str, Any]:
        return {"method": self.method, "xs": list(self._xs), "ys": list(self._ys)}

    def restore(self, state: dict[str, Any]) -> None:
        self._xs = [float(x) for x in state.get("xs", [])]
        self._ys = [float(y) for y in state.get("ys", [])]
        self._fitted = bool(self._xs)


class PlattCalibrator:
    """Two-parameter logistic recalibration: p_cal = σ(a * p_raw + b).

    Fit by Newton-Raphson on weighted cross-entropy with a small ridge
    penalty for numerical stability. Useful when N is small or you want
    a smooth, parametric correction.
    """

    method = METHOD_PLATT

    def __init__(self, *, max_iters: int = 50, ridge: float = 1e-3, tol: float = 1e-7) -> None:
        self.max_iters = max_iters
        self.ridge = ridge
        self.tol = tol
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(self, samples: list[CalibrationSample]) -> None:
        if not samples:
            self.a, self.b, self._fitted = 1.0, 0.0, False
            return
        a, b = 1.0, 0.0
        xs = [s.p_forecast for s in samples]
        ys = [1.0 if s.outcome else 0.0 for s in samples]
        ws = [s.weight for s in samples]
        for _ in range(self.max_iters):
            g0 = 0.0  # ∂L/∂a
            g1 = 0.0  # ∂L/∂b
            h00 = 0.0
            h01 = 0.0
            h11 = 0.0
            for x, y, w in zip(xs, ys, ws):
                z = a * x + b
                # σ(z), stable.
                p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
                p = min(max(p, _EPS), 1.0 - _EPS)
                err = (y - p) * w
                g0 += err * x
                g1 += err
                v = p * (1.0 - p) * w
                h00 += v * x * x
                h01 += v * x
                h11 += v
            # Add ridge to Hessian diagonal.
            h00 += self.ridge
            h11 += self.ridge
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-18:
                break
            # Newton step: θ_new = θ + H^{-1} g  (we maximize L)
            da = (h11 * g0 - h01 * g1) / det
            db = (-h01 * g0 + h00 * g1) / det
            a += da
            b += db
            if abs(da) + abs(db) < self.tol:
                break
        self.a, self.b = a, b
        self._fitted = True

    def adjust(self, p_raw: float) -> float:
        if not self._fitted:
            return p_raw
        z = self.a * p_raw + self.b
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def snapshot(self) -> dict[str, Any]:
        return {"method": self.method, "a": float(self.a), "b": float(self.b)}

    def restore(self, state: dict[str, Any]) -> None:
        self.a = float(state.get("a", 1.0))
        self.b = float(state.get("b", 0.0))
        self._fitted = True


class _IdentityCalibrator:
    method = METHOD_IDENTITY

    def fit(self, samples: list[CalibrationSample]) -> None:  # pragma: no cover - trivial
        pass

    def adjust(self, p_raw: float) -> float:
        return max(0.0, min(1.0, p_raw))

    def snapshot(self) -> dict[str, Any]:
        return {"method": self.method}

    def restore(self, state: dict[str, Any]) -> None:  # pragma: no cover
        pass


def _make_calibrator(method: str) -> Any:
    if method == METHOD_ISOTONIC:
        return IsotonicCalibrator()
    if method == METHOD_PLATT:
        return PlattCalibrator()
    if method == METHOD_IDENTITY:
        return _IdentityCalibrator()
    raise ValueError(f"unknown calibration method: {method!r}")


# ----- engine ---------------------------------------------------------


@dataclass
class _SegmentState:
    """Per-(source, bucket) ring buffer + calibrator.

    `total_observed` is the lifetime count of observations seen for this
    segment; it keeps growing past `window` so the auto-fit cadence
    works against a strictly monotonic counter.
    """
    samples: list[CalibrationSample] = field(default_factory=list)
    calibrator: Any = field(default_factory=_IdentityCalibrator)
    total_observed: int = 0
    last_fit_at: int = 0  # value of total_observed at last fit
    last_fit_ts: float = 0.0


class CalibrationEngine:
    """Public face of probability recalibration.

    Workflow:

        eng = CalibrationEngine(method="isotonic")
        for ticket in stream:
            p_raw = preflight.estimate(ticket).p_success
            p_cal = eng.calibrate(p_raw, source="preflight")
            # ... use p_cal for EV / admission / hedging ...
            eng.observe(p_raw, ticket.completed_successfully, source="preflight")
        report = eng.report(source="preflight")
        if not report.ok:
            log.warning("preflight calibration degraded: %s", report)

    Thread-safe. Sub-calibrators per (source, bucket). Auto-refits after
    `refit_every` new observations on a given segment. Emits events on
    the configured bus (if any) so a coordination engine can subscribe.

    Persistence: `snapshot()` and `restore()` round-trip the full state.
    JSONL append-mode persistence is supported for raw observations so
    a fleet of runtimes can share a calibration corpus.
    """

    def __init__(
        self,
        *,
        method: str = METHOD_ISOTONIC,
        window: int = 5000,
        min_samples_to_fit: int = 30,
        refit_every: int = 50,
        ece_threshold: float = 0.05,
        drift_threshold: float = 0.08,
        drift_recent_frac: float = 0.25,
        drift_min_recent: int = 30,
        n_bins: int = 10,
        bus: EventBus | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> None:
        if method not in KNOWN_METHODS:
            raise ValueError(f"unknown calibration method: {method!r}")
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 0.0 < drift_recent_frac < 1.0:
            raise ValueError("drift_recent_frac must be in (0,1)")
        self.method = method
        self.window = window
        self.min_samples_to_fit = min_samples_to_fit
        self.refit_every = max(1, refit_every)
        self.ece_threshold = ece_threshold
        self.drift_threshold = drift_threshold
        self.drift_recent_frac = drift_recent_frac
        self.drift_min_recent = drift_min_recent
        self.n_bins = n_bins
        self.bus = bus
        self.path: Path | None = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
        self._lock = threading.RLock()
        self._segments: dict[tuple[str, str], _SegmentState] = {}
        # Track which segments are currently in a "drifted" state so we
        # only emit on rising edges (entering drift), not on every observe
        # while already drifted. Reset when score falls below threshold.
        self._segment_drifted: dict[tuple[str, str], bool] = {}

    # ---- observation ---------------------------------------------

    def observe(
        self,
        p_forecast: float,
        outcome: bool,
        *,
        source: str = "",
        bucket: str = "",
        weight: float = 1.0,
        ts: float | None = None,
    ) -> CalibrationSample:
        sample = CalibrationSample(
            p_forecast=float(p_forecast),
            outcome=bool(outcome),
            ts=time.time() if ts is None else float(ts),
            source=str(source),
            bucket=str(bucket),
            weight=float(weight),
        )
        with self._lock:
            seg = self._segments.setdefault((sample.source, sample.bucket), _SegmentState())
            seg.samples.append(sample)
            seg.total_observed += 1
            # Trim window.
            if len(seg.samples) > self.window:
                seg.samples = seg.samples[-self.window :]
            self._maybe_persist(sample)
            should_fit = (
                len(seg.samples) >= self.min_samples_to_fit
                and seg.total_observed - seg.last_fit_at >= self.refit_every
            )
        self._emit(CAL_OBSERVED, {
            "source": sample.source,
            "bucket": sample.bucket,
            "p_forecast": sample.p_forecast,
            "outcome": sample.outcome,
            "weight": sample.weight,
        })
        if should_fit:
            self.fit(source=sample.source, bucket=sample.bucket)
        # Drift detection runs on every observe — it's cheap.
        self._check_drift(sample.source, sample.bucket)
        return sample

    def observe_many(self, samples: Iterable[CalibrationSample]) -> None:
        for s in samples:
            self.observe(
                s.p_forecast,
                s.outcome,
                source=s.source,
                bucket=s.bucket,
                weight=s.weight,
                ts=s.ts,
            )

    # ---- adjustment ----------------------------------------------

    def calibrate(self, p_raw: float, *, source: str = "", bucket: str = "") -> float:
        """Map a raw forecast to a calibrated probability.

        Falls back gracefully: if no calibrator has been fit for the
        requested segment, tries (source, "") then ("", "") then
        identity. This keeps the contract simple — the caller never
        has to check whether a fit has happened.
        """
        with self._lock:
            keys = [(source, bucket), (source, ""), ("", "")]
            for key in keys:
                seg = self._segments.get(key)
                if seg is not None and getattr(seg.calibrator, "_fitted", False):
                    return _clamp(seg.calibrator.adjust(p_raw))
            return _clamp(p_raw)

    # ---- fit -----------------------------------------------------

    def fit(self, *, source: str | None = None, bucket: str | None = None) -> None:
        """Refit one segment or all segments.

        Pass `source=None, bucket=None` to refit every known segment.
        """
        with self._lock:
            if source is None and bucket is None:
                keys = list(self._segments.keys())
            else:
                keys = [(source or "", bucket or "")]
            for key in keys:
                seg = self._segments.get(key)
                if seg is None:
                    continue
                if len(seg.samples) < self.min_samples_to_fit:
                    continue
                cal = _make_calibrator(self.method)
                cal.fit(seg.samples)
                seg.calibrator = cal
                seg.last_fit_at = seg.total_observed
                seg.last_fit_ts = time.time()
                self._emit(
                    CAL_FIT,
                    {
                        "source": key[0],
                        "bucket": key[1],
                        "method": self.method,
                        "n": len(seg.samples),
                        "total_observed": seg.total_observed,
                    },
                )

    # ---- reporting -----------------------------------------------

    def report(
        self,
        *,
        source: str | None = None,
        bucket: str | None = None,
    ) -> CalibrationReport:
        """Return a report. With `source=None, bucket=None`, aggregates
        across every segment seen. Otherwise reports on one segment.
        """
        with self._lock:
            if source is None and bucket is None:
                samples: list[CalibrationSample] = []
                for seg in self._segments.values():
                    samples.extend(seg.samples)
                key = ("", "")
                seg_method = self.method
            else:
                key = (source or "", bucket or "")
                seg = self._segments.get(key)
                samples = list(seg.samples) if seg else []
                seg_method = self.method
            drift = self._drift_score(samples) if samples else 0.0
        bins = reliability_bins(samples, n_bins=self.n_bins)
        ece = expected_calibration_error(bins)
        mce = max_calibration_error(bins)
        report = CalibrationReport(
            n=len(samples),
            weight_sum=sum(s.weight for s in samples),
            brier=brier_score(samples),
            log_loss=log_loss(samples),
            ece=ece,
            mce=mce,
            bins=tuple(bins),
            method=seg_method,
            drift_score=drift,
            ok=(
                len(samples) >= self.min_samples_to_fit
                and ece <= self.ece_threshold
                and drift <= self.drift_threshold
            ),
            source=key[0],
            bucket=key[1],
        )
        self._emit(CAL_REPORT, {"report": report.to_dict()})
        return report

    def segments(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._segments.keys())

    # ---- persistence ---------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "method": self.method,
                "window": self.window,
                "segments": [
                    {
                        "source": key[0],
                        "bucket": key[1],
                        "samples": [asdict(s) for s in seg.samples],
                        "calibrator": seg.calibrator.snapshot(),
                        "total_observed": seg.total_observed,
                        "last_fit_at": seg.last_fit_at,
                        "last_fit_ts": seg.last_fit_ts,
                    }
                    for key, seg in self._segments.items()
                ],
            }

    def restore(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._segments.clear()
            for entry in state.get("segments", []):
                src = entry.get("source", "")
                bkt = entry.get("bucket", "")
                seg = _SegmentState()
                for s in entry.get("samples", []):
                    seg.samples.append(CalibrationSample(**s))
                cal_state = entry.get("calibrator", {})
                cal = _make_calibrator(cal_state.get("method", self.method))
                cal.restore(cal_state)
                seg.calibrator = cal
                seg.total_observed = int(entry.get("total_observed", len(seg.samples)))
                seg.last_fit_at = int(entry.get("last_fit_at", entry.get("last_fit_n", 0)))
                seg.last_fit_ts = float(entry.get("last_fit_ts", 0.0))
                self._segments[(src, bkt)] = seg

    def replay_jsonl(self, path: str | os.PathLike[str]) -> int:
        """Load (source, bucket, p_forecast, outcome, ts, weight) rows
        from a JSONL trace and replay them as observations. Returns the
        number of samples accepted. Lines that don't parse are skipped.
        """
        n = 0
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                self.observe(
                    p_forecast=float(row["p_forecast"]),
                    outcome=bool(row["outcome"]),
                    source=str(row.get("source", "")),
                    bucket=str(row.get("bucket", "")),
                    weight=float(row.get("weight", 1.0)),
                    ts=row.get("ts"),
                )
                n += 1
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
        return n

    # ---- private -------------------------------------------------

    def _drift_score(self, samples: list[CalibrationSample]) -> float:
        """ECE on a recent tail minus ECE on the older bulk.

        Returns 0 when there isn't enough recent data to compute a
        meaningful comparison. Bounded above by 1.
        """
        n = len(samples)
        if n < self.drift_min_recent * 2:
            return 0.0
        k = max(self.drift_min_recent, int(self.drift_recent_frac * n))
        if k >= n:
            return 0.0
        recent = samples[-k:]
        older = samples[: n - k]
        if len(older) < self.drift_min_recent:
            return 0.0
        ece_recent = expected_calibration_error(reliability_bins(recent, n_bins=self.n_bins))
        ece_older = expected_calibration_error(reliability_bins(older, n_bins=self.n_bins))
        return min(1.0, abs(ece_recent - ece_older))

    def _check_drift(self, source: str, bucket: str) -> None:
        with self._lock:
            seg = self._segments.get((source, bucket))
            if seg is None:
                return
            drift = self._drift_score(seg.samples)
            key = (source, bucket)
            currently_drifted = self._segment_drifted.get(key, False)
            now_drifted = drift > self.drift_threshold
            self._segment_drifted[key] = now_drifted
            if not now_drifted or currently_drifted:
                # Only emit on rising edge: entering the drifted state.
                # While persistently drifted, the consumer already knows.
                return
        self._emit(
            CAL_DRIFT,
            {"source": source, "bucket": bucket, "drift_score": drift, "threshold": self.drift_threshold},
        )

    def _maybe_persist(self, sample: CalibrationSample) -> None:
        if self.path is None:
            return
        line = json.dumps(asdict(sample), sort_keys=True, separators=(",", ":"))
        with self.path.open("a") as f:
            f.write(line + "\n")

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.bus is None:
            return
        self.bus.publish(Event(kind=kind, data=data))


def _clamp(p: float) -> float:
    if math.isnan(p):
        return 0.5
    return max(0.0, min(1.0, float(p)))


# ----- integration helpers --------------------------------------------


def attach_to_bus(
    engine: CalibrationEngine,
    bus: EventBus,
    *,
    forecast_event: str = "preflight.forecast",
    outcome_event: str = "preflight.outcome",
    forecast_key: str = "p_success",
    outcome_key: str = "success",
    source_key: str = "source",
    bucket_key: str = "bucket",
    correlate_key: str = "ticket_id",
) -> Callable[[], None]:
    """Subscribe `engine` to a paired forecast/outcome event stream.

    Forecast events stash `p_forecast` keyed by `correlate_key`. The
    matching outcome event then triggers `observe()` on the engine.
    Returns an unsubscribe callable.
    """
    pending: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def on_forecast(ev: Event) -> None:
        cid = ev.data.get(correlate_key)
        if cid is None:
            return
        with lock:
            pending[str(cid)] = dict(ev.data)

    def on_outcome(ev: Event) -> None:
        cid = ev.data.get(correlate_key)
        if cid is None:
            return
        with lock:
            forecast = pending.pop(str(cid), None)
        if forecast is None:
            return
        try:
            p = float(forecast[forecast_key])
            y = bool(ev.data.get(outcome_key))
        except (KeyError, TypeError, ValueError):
            return
        engine.observe(
            p,
            y,
            source=str(forecast.get(source_key, "") or ev.data.get(source_key, "")),
            bucket=str(forecast.get(bucket_key, "") or ev.data.get(bucket_key, "")),
        )

    sub_f = bus.subscribe(on_forecast, kind=forecast_event)
    sub_o = bus.subscribe(on_outcome, kind=outcome_event)

    def detach() -> None:
        bus.unsubscribe(sub_f)
        bus.unsubscribe(sub_o)
        with lock:
            pending.clear()

    return detach


def attach_to_driver(
    engine: CalibrationEngine,
    driver: Any,
    *,
    source: str = "preflight",
    success_field: str = "success",
    forecast_field: str = "p_success_forecast",
    bucket_fn: Callable[[Any], str] | None = None,
    weight_fn: Callable[[Any], float] | None = None,
) -> Callable[[], None]:
    """Wire a `RuntimeDriver`'s completion stream into the engine.

    For every receipt the driver emits, pull the forecasted `p_success`
    (from the estimate decision) and the actual outcome, and observe
    them. Returns an unsubscribe callable.

    The integration is best-effort: receipts that don't carry a
    forecast (e.g. rejected tickets) are skipped.
    """
    bus: EventBus | None = getattr(driver, "bus", None) or getattr(driver, "_bus", None)
    if bus is None:
        raise ValueError("driver has no event bus to subscribe to")

    from agi.events import SESSION_ENDED  # local to avoid cycles

    def on_session_ended(ev: Event) -> None:
        receipt = ev.data.get("receipt")
        if not isinstance(receipt, dict):
            return
        p = receipt.get(forecast_field)
        if p is None:
            # Fall back to walking the decision trace.
            for d in receipt.get("decisions", []):
                if d.get("kind") == "estimate":
                    p = (d.get("data") or {}).get("p_success")
                    if p is not None:
                        break
        if p is None:
            return
        try:
            p = float(p)
        except (TypeError, ValueError):
            return
        outcome = bool(receipt.get(success_field, receipt.get("status") == "completed"))
        bucket = bucket_fn(receipt) if bucket_fn else ""
        weight = weight_fn(receipt) if weight_fn else 1.0
        engine.observe(p, outcome, source=source, bucket=bucket, weight=weight)

    sub = bus.subscribe(on_session_ended, kind=SESSION_ENDED)

    def detach() -> None:
        bus.unsubscribe(sub)

    return detach


__all__ = [
    # event kinds
    "CAL_OBSERVED",
    "CAL_FIT",
    "CAL_DRIFT",
    "CAL_REPORT",
    # methods
    "METHOD_ISOTONIC",
    "METHOD_PLATT",
    "METHOD_IDENTITY",
    "KNOWN_METHODS",
    # dataclasses
    "CalibrationSample",
    "ReliabilityBin",
    "CalibrationReport",
    # calibrators
    "IsotonicCalibrator",
    "PlattCalibrator",
    # engine
    "CalibrationEngine",
    # metrics
    "brier_score",
    "log_loss",
    "reliability_bins",
    "expected_calibration_error",
    "max_calibration_error",
    # integration
    "attach_to_bus",
    "attach_to_driver",
]
