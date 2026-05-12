"""Tests for the CalibrationEngine.

Covers:
 - dataclass validation
 - Brier / log loss / ECE / MCE / reliability binning
 - IsotonicCalibrator: monotonicity, interpolation, exact PAV behavior
 - PlattCalibrator: convergence, inversion of miscalibrated streams
 - CalibrationEngine: observe, calibrate (with fallback), auto-fit,
   per-segment isolation, drift detection, snapshot/restore, JSONL
   replay, bus event emission, RuntimeDriver-style integration.
"""
from __future__ import annotations

import json
import math
import random
import tempfile
import threading
from pathlib import Path

import pytest

from agi.calibration import (
    CAL_DRIFT,
    CAL_FIT,
    CAL_OBSERVED,
    CAL_REPORT,
    CalibrationEngine,
    CalibrationReport,
    CalibrationSample,
    IsotonicCalibrator,
    METHOD_ISOTONIC,
    METHOD_PLATT,
    PlattCalibrator,
    ReliabilityBin,
    attach_to_bus,
    attach_to_driver,
    brier_score,
    expected_calibration_error,
    log_loss,
    max_calibration_error,
    reliability_bins,
)
from agi.events import Event, EventBus, SESSION_ENDED


# --- helpers ---------------------------------------------------------


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _make_miscalibrated_stream(n: int, *, bias: float, seed: int = 0):
    """Generate (p_forecast, outcome) pairs where the true rate is
    biased relative to the forecast: y ~ Bernoulli(clip(p + bias))."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        p = rng.random()
        q = max(0.0, min(1.0, p + bias))
        y = rng.random() < q
        samples.append(CalibrationSample(p_forecast=p, outcome=y))
    return samples


def _make_perfect_stream(n: int, seed: int = 0):
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        p = rng.random()
        y = rng.random() < p
        samples.append(CalibrationSample(p_forecast=p, outcome=y))
    return samples


# --- dataclass validation -------------------------------------------


class TestCalibrationSample:
    def test_accepts_valid(self) -> None:
        s = CalibrationSample(p_forecast=0.7, outcome=True, weight=2.0)
        assert s.p_forecast == 0.7
        assert s.outcome is True
        assert s.weight == 2.0

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            CalibrationSample(p_forecast=1.5, outcome=True)
        with pytest.raises(ValueError):
            CalibrationSample(p_forecast=-0.01, outcome=False)

    def test_rejects_bad_weight(self) -> None:
        with pytest.raises(ValueError):
            CalibrationSample(p_forecast=0.5, outcome=True, weight=-1.0)
        with pytest.raises(ValueError):
            CalibrationSample(p_forecast=0.5, outcome=True, weight=float("nan"))


# --- metrics --------------------------------------------------------


class TestBrierScore:
    def test_perfect_prediction(self) -> None:
        s = [
            CalibrationSample(1.0, True),
            CalibrationSample(0.0, False),
        ]
        # Use 1-eps to avoid the validator, then check brier-equivalent.
        assert brier_score(s) == pytest.approx(0.0, abs=1e-9)

    def test_uninformed_is_quarter(self) -> None:
        s = [CalibrationSample(0.5, i % 2 == 0) for i in range(100)]
        assert brier_score(s) == pytest.approx(0.25, abs=1e-9)

    def test_handles_empty(self) -> None:
        assert brier_score([]) == 0.0

    def test_weighted(self) -> None:
        s = [
            CalibrationSample(0.0, True, weight=10.0),  # very wrong, big weight
            CalibrationSample(1.0, True, weight=1.0),   # right, small weight
        ]
        # (10 * 1 + 1 * 0) / 11 ≈ 0.909
        assert brier_score(s) == pytest.approx(10.0 / 11.0, abs=1e-9)


class TestLogLoss:
    def test_perfect_clamped(self) -> None:
        # p=1.0 with y=True → log(1-eps), nearly 0.
        s = [CalibrationSample(1.0, True), CalibrationSample(0.0, False)]
        assert log_loss(s) < 1e-4

    def test_worst_case_clamped(self) -> None:
        s = [CalibrationSample(0.0, True), CalibrationSample(1.0, False)]
        # Should be large but finite.
        ll = log_loss(s)
        assert ll > 10  # roughly -log(1e-6)
        assert math.isfinite(ll)

    def test_empty(self) -> None:
        assert log_loss([]) == 0.0


class TestReliabilityBins:
    def test_basic(self) -> None:
        s = [CalibrationSample(0.05, True), CalibrationSample(0.95, False)]
        bins = reliability_bins(s, n_bins=10)
        # Two non-empty bins (first and last).
        assert len(bins) == 2
        assert bins[0].p_lo == 0.0 and bins[0].p_hi == pytest.approx(0.1)
        assert bins[-1].p_hi == 1.0

    def test_p_one_falls_in_last_bin(self) -> None:
        # Edge case: p=1.0 should not crash and should land in the last bin.
        s = [CalibrationSample(1.0, True)]
        bins = reliability_bins(s, n_bins=10)
        assert len(bins) == 1
        assert bins[0].p_hi == 1.0

    def test_rejects_zero_bins(self) -> None:
        with pytest.raises(ValueError):
            reliability_bins([], n_bins=0)

    def test_weighted_means(self) -> None:
        s = [
            CalibrationSample(0.05, True, weight=4.0),
            CalibrationSample(0.05, False, weight=1.0),
        ]
        bins = reliability_bins(s, n_bins=10)
        assert bins[0].empirical_rate == pytest.approx(0.8)
        assert bins[0].forecast_mean == pytest.approx(0.05)


class TestECE_MCE:
    def test_perfect_calibration(self) -> None:
        samples = _make_perfect_stream(5000, seed=1)
        bins = reliability_bins(samples, n_bins=10)
        ece = expected_calibration_error(bins)
        # Statistical noise around 0.
        assert ece < 0.03

    def test_systematic_overconfidence(self) -> None:
        # Forecasts say high, outcomes are low.
        samples = _make_miscalibrated_stream(2000, bias=-0.3, seed=2)
        bins = reliability_bins(samples, n_bins=10)
        ece = expected_calibration_error(bins)
        # With bias=-0.3, ECE should be large.
        assert ece > 0.15

    def test_mce_at_least_ece(self) -> None:
        samples = _make_miscalibrated_stream(1000, bias=0.2, seed=3)
        bins = reliability_bins(samples, n_bins=10)
        assert max_calibration_error(bins) >= expected_calibration_error(bins)


# --- isotonic calibrator -------------------------------------------


class TestIsotonicCalibrator:
    def test_unfit_returns_input(self) -> None:
        cal = IsotonicCalibrator()
        assert cal.adjust(0.42) == 0.42

    def test_output_is_monotone(self) -> None:
        samples = _make_miscalibrated_stream(500, bias=-0.25, seed=4)
        cal = IsotonicCalibrator()
        cal.fit(samples)
        xs = [i / 100 for i in range(101)]
        ys = [cal.adjust(x) for x in xs]
        for a, b in zip(ys, ys[1:]):
            assert a <= b + 1e-12

    def test_corrects_systematic_overconfidence(self) -> None:
        # Bias=-0.3 means forecasts are too high by ~0.3.
        samples = _make_miscalibrated_stream(3000, bias=-0.3, seed=5)
        cal = IsotonicCalibrator()
        cal.fit(samples)
        # Held-out evaluation: build a fresh stream from the same process.
        test = _make_miscalibrated_stream(3000, bias=-0.3, seed=99)
        raw_ece = expected_calibration_error(reliability_bins(test, n_bins=10))
        adjusted = [
            CalibrationSample(p_forecast=cal.adjust(s.p_forecast), outcome=s.outcome)
            for s in test
        ]
        cal_ece = expected_calibration_error(reliability_bins(adjusted, n_bins=10))
        assert cal_ece < raw_ece * 0.5  # at least halved

    def test_exact_pav_cascading_pool(self) -> None:
        # Three samples (0.1,T), (0.5,F), (0.9,F). After pooling the
        # first violation (T>F at index 0→1) we get a block with mean
        # 0.5; that's still > 0 at index 1→2 so PAV pools again, giving
        # a single block of mean 1/3 over the full range.
        samples = [
            CalibrationSample(0.1, True),
            CalibrationSample(0.5, False),
            CalibrationSample(0.9, False),
        ]
        cal = IsotonicCalibrator()
        cal.fit(samples)
        assert cal.adjust(0.1) == pytest.approx(1.0 / 3.0)
        assert cal.adjust(0.5) == pytest.approx(1.0 / 3.0)
        assert cal.adjust(0.9) == pytest.approx(1.0 / 3.0)

    def test_pav_partial_pool(self) -> None:
        # (0.1, F), (0.3, T), (0.5, F), (0.7, T): violation at 0.3→0.5
        # pools those two to 0.5; the surrounding blocks remain.
        samples = [
            CalibrationSample(0.1, False),
            CalibrationSample(0.3, True),
            CalibrationSample(0.5, False),
            CalibrationSample(0.7, True),
        ]
        cal = IsotonicCalibrator()
        cal.fit(samples)
        # Monotone non-decreasing.
        ys = [cal.adjust(x) for x in (0.1, 0.4, 0.7)]
        for a, b in zip(ys, ys[1:]):
            assert a <= b + 1e-12
        # Endpoints reflect the unpooled samples.
        assert cal.adjust(0.1) == pytest.approx(0.0)
        assert cal.adjust(0.7) == pytest.approx(1.0)

    def test_snapshot_restore_round_trip(self) -> None:
        samples = _make_miscalibrated_stream(200, bias=0.1, seed=6)
        cal = IsotonicCalibrator()
        cal.fit(samples)
        state = cal.snapshot()
        restored = IsotonicCalibrator()
        restored.restore(state)
        for x in [0.0, 0.1, 0.5, 0.9, 1.0]:
            assert restored.adjust(x) == pytest.approx(cal.adjust(x))

    def test_empty_fit_is_identity(self) -> None:
        cal = IsotonicCalibrator()
        cal.fit([])
        assert cal.adjust(0.3) == 0.3

    def test_extrapolation_clamps_to_endpoints(self) -> None:
        # Single block at x=0.5, y=0.7 (one sample True at p=0.5 weight 1).
        samples = [CalibrationSample(0.5, True)]
        cal = IsotonicCalibrator()
        cal.fit(samples)
        assert cal.adjust(0.0) == cal.adjust(0.5)  # flat extrapolation
        assert cal.adjust(1.0) == cal.adjust(0.5)


# --- platt calibrator ----------------------------------------------


class TestPlattCalibrator:
    def test_unfit_returns_sigmoid_identity(self) -> None:
        # Default a=1, b=0; σ(p) is not p, but it should still be a
        # well-defined number in (0, 1).
        cal = PlattCalibrator()
        v = cal.adjust(0.5)
        assert 0.0 < v < 1.0

    def test_fit_perfect_data_pulls_toward_identity_ordering(self) -> None:
        samples = _make_perfect_stream(2000, seed=7)
        cal = PlattCalibrator()
        cal.fit(samples)
        # Output is monotone increasing.
        prev = -1.0
        for x in [i / 50 for i in range(51)]:
            v = cal.adjust(x)
            assert v >= prev - 1e-9
            prev = v

    def test_fit_corrects_overconfidence(self) -> None:
        samples = _make_miscalibrated_stream(2000, bias=-0.25, seed=8)
        cal = PlattCalibrator()
        cal.fit(samples)
        adjusted = [
            CalibrationSample(p_forecast=cal.adjust(s.p_forecast), outcome=s.outcome)
            for s in samples
        ]
        # Brier should drop relative to the raw stream.
        raw_brier = brier_score(samples)
        cal_brier = brier_score(adjusted)
        assert cal_brier < raw_brier

    def test_empty_fit_keeps_defaults(self) -> None:
        cal = PlattCalibrator()
        cal.fit([])
        assert cal.a == 1.0 and cal.b == 0.0

    def test_snapshot_restore(self) -> None:
        cal = PlattCalibrator()
        cal.fit(_make_miscalibrated_stream(500, bias=0.1, seed=9))
        s = cal.snapshot()
        r = PlattCalibrator()
        r.restore(s)
        for x in [0.1, 0.3, 0.6, 0.95]:
            assert r.adjust(x) == pytest.approx(cal.adjust(x), rel=1e-6)


# --- engine ---------------------------------------------------------


class TestEngineBasics:
    def test_observe_and_calibrate_fallback(self) -> None:
        eng = CalibrationEngine(min_samples_to_fit=5, refit_every=1)
        # Before any data, calibrate returns input unchanged.
        assert eng.calibrate(0.7) == 0.7

    def test_auto_fits_after_threshold(self) -> None:
        eng = CalibrationEngine(min_samples_to_fit=10, refit_every=10)
        for s in _make_miscalibrated_stream(100, bias=-0.3, seed=10):
            eng.observe(s.p_forecast, s.outcome)
        report = eng.report()
        assert report.n == 100
        assert report.method == METHOD_ISOTONIC
        # The fit should have happened.
        assert any(
            getattr(seg.calibrator, "_fitted", False)
            for seg in eng._segments.values()
        )

    def test_calibrate_after_fit_reduces_ece(self) -> None:
        eng = CalibrationEngine(min_samples_to_fit=50, refit_every=50)
        train = _make_miscalibrated_stream(500, bias=-0.25, seed=11)
        for s in train:
            eng.observe(s.p_forecast, s.outcome)
        test = _make_miscalibrated_stream(500, bias=-0.25, seed=12)
        raw_ece = expected_calibration_error(reliability_bins(test, n_bins=10))
        adjusted = [
            CalibrationSample(p_forecast=eng.calibrate(s.p_forecast), outcome=s.outcome)
            for s in test
        ]
        cal_ece = expected_calibration_error(reliability_bins(adjusted, n_bins=10))
        assert cal_ece < raw_ece * 0.6

    def test_per_source_segmentation(self) -> None:
        eng = CalibrationEngine(min_samples_to_fit=20, refit_every=20)
        # Two forecasters with opposite biases.
        for s in _make_miscalibrated_stream(200, bias=-0.3, seed=13):
            eng.observe(s.p_forecast, s.outcome, source="preflight")
        for s in _make_miscalibrated_stream(200, bias=+0.3, seed=14):
            eng.observe(s.p_forecast, s.outcome, source="oracle")
        # Calibrators per source should bend in opposite directions.
        p_preflight = eng.calibrate(0.7, source="preflight")
        p_oracle = eng.calibrate(0.7, source="oracle")
        # preflight is overconfident → p_preflight < 0.7
        # oracle is underconfident → p_oracle > 0.7
        assert p_preflight < 0.7
        assert p_oracle > 0.7

    def test_calibrate_falls_back_to_global_then_identity(self) -> None:
        eng = CalibrationEngine(min_samples_to_fit=20, refit_every=20)
        for s in _make_miscalibrated_stream(200, bias=-0.3, seed=15):
            eng.observe(s.p_forecast, s.outcome, source="", bucket="")
        # Unknown source — should fall back to global calibrator.
        assert eng.calibrate(0.7, source="unknown") < 0.7

    def test_window_trims(self) -> None:
        eng = CalibrationEngine(window=100, min_samples_to_fit=10, refit_every=10000)
        for _ in range(250):
            eng.observe(0.5, True)
        seg = next(iter(eng._segments.values()))
        assert len(seg.samples) == 100

    def test_observe_many_round_trip(self) -> None:
        eng = CalibrationEngine(min_samples_to_fit=10, refit_every=10000)
        eng.observe_many(_make_perfect_stream(50, seed=16))
        assert eng.report().n == 50

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError):
            CalibrationEngine(method="banana")

    def test_drift_detection_emits_event(self) -> None:
        bus = EventBus()
        captured: list[Event] = []
        bus.subscribe(lambda ev: captured.append(ev), kind=CAL_DRIFT)
        eng = CalibrationEngine(
            min_samples_to_fit=20,
            refit_every=20,
            drift_threshold=0.05,
            drift_recent_frac=0.3,
            drift_min_recent=30,
            bus=bus,
        )
        # First half: forecasts well calibrated. Second half: severely biased.
        for s in _make_perfect_stream(200, seed=17):
            eng.observe(s.p_forecast, s.outcome)
        for s in _make_miscalibrated_stream(200, bias=-0.4, seed=18):
            eng.observe(s.p_forecast, s.outcome)
        assert len(captured) >= 1
        assert captured[0].data["drift_score"] > 0.05

    def test_no_drift_for_stationary_stream(self) -> None:
        bus = EventBus()
        captured: list[Event] = []
        bus.subscribe(lambda ev: captured.append(ev), kind=CAL_DRIFT)
        # Threshold deliberately loose: small-sample ECE has natural
        # noise around ±0.1. The contract is that *under stationarity*
        # a reasonable threshold doesn't false-alarm, not that the score
        # is exactly zero.
        eng = CalibrationEngine(
            min_samples_to_fit=20,
            refit_every=20,
            drift_threshold=0.20,
            drift_min_recent=50,
            bus=bus,
        )
        for s in _make_perfect_stream(800, seed=19):
            eng.observe(s.p_forecast, s.outcome)
        assert len(captured) == 0

    def test_emits_observed_and_fit_events(self) -> None:
        bus = EventBus()
        obs: list[Event] = []
        fits: list[Event] = []
        bus.subscribe(lambda ev: obs.append(ev), kind=CAL_OBSERVED)
        bus.subscribe(lambda ev: fits.append(ev), kind=CAL_FIT)
        eng = CalibrationEngine(min_samples_to_fit=5, refit_every=5, bus=bus)
        for s in _make_perfect_stream(20, seed=20):
            eng.observe(s.p_forecast, s.outcome, source="preflight")
        assert len(obs) == 20
        assert len(fits) >= 1
        assert fits[0].data["source"] == "preflight"

    def test_report_emits_event(self) -> None:
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(lambda ev: seen.append(ev), kind=CAL_REPORT)
        eng = CalibrationEngine(min_samples_to_fit=5, refit_every=5, bus=bus)
        eng.observe_many(_make_perfect_stream(50, seed=21))
        r = eng.report()
        assert isinstance(r, CalibrationReport)
        assert len(seen) == 1
        assert "report" in seen[0].data


class TestEnginePersistence:
    def test_snapshot_restore_round_trip(self) -> None:
        eng = CalibrationEngine(min_samples_to_fit=10, refit_every=10)
        eng.observe_many(_make_miscalibrated_stream(100, bias=-0.2, seed=22))
        state = eng.snapshot()
        eng2 = CalibrationEngine(min_samples_to_fit=10, refit_every=10)
        eng2.restore(state)
        assert eng2.report().n == eng.report().n
        for x in [0.1, 0.4, 0.7, 0.9]:
            assert eng2.calibrate(x) == pytest.approx(eng.calibrate(x), abs=1e-9)

    def test_jsonl_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.jsonl"
            eng = CalibrationEngine(min_samples_to_fit=10, refit_every=10, path=path)
            eng.observe_many(_make_perfect_stream(50, seed=23))
            # File should contain 50 rows.
            lines = path.read_text().splitlines()
            assert len(lines) == 50
            # Replay into a fresh engine.
            eng2 = CalibrationEngine(min_samples_to_fit=10, refit_every=10)
            assert eng2.replay_jsonl(path) == 50
            assert eng2.report().n == 50

    def test_jsonl_skips_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"p_forecast": 0.5, "outcome": True}),
                        "not json",
                        json.dumps({"missing": "fields"}),
                        json.dumps({"p_forecast": 0.7, "outcome": False, "source": "x"}),
                    ]
                )
                + "\n"
            )
            eng = CalibrationEngine(min_samples_to_fit=2, refit_every=2)
            n = eng.replay_jsonl(path)
            assert n == 2


class TestEngineConcurrency:
    def test_thread_safe_observe(self) -> None:
        eng = CalibrationEngine(min_samples_to_fit=10, refit_every=10)

        def hammer(seed: int) -> None:
            rng = random.Random(seed)
            for _ in range(500):
                eng.observe(rng.random(), rng.random() < 0.5)

        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert eng.report().n == 2000


# --- integration helpers --------------------------------------------


class TestAttachToBus:
    def test_pairs_forecast_with_outcome(self) -> None:
        bus = EventBus()
        eng = CalibrationEngine(min_samples_to_fit=2, refit_every=2)
        detach = attach_to_bus(eng, bus)
        bus.publish(Event(kind="preflight.forecast", data={"ticket_id": "t1", "p_success": 0.8}))
        bus.publish(Event(kind="preflight.outcome", data={"ticket_id": "t1", "success": True}))
        bus.publish(Event(kind="preflight.forecast", data={"ticket_id": "t2", "p_success": 0.3}))
        bus.publish(Event(kind="preflight.outcome", data={"ticket_id": "t2", "success": False}))
        assert eng.report().n == 2
        detach()
        # After detach, more events do not flow.
        bus.publish(Event(kind="preflight.forecast", data={"ticket_id": "t3", "p_success": 0.5}))
        bus.publish(Event(kind="preflight.outcome", data={"ticket_id": "t3", "success": True}))
        assert eng.report().n == 2

    def test_outcome_without_matching_forecast_is_skipped(self) -> None:
        bus = EventBus()
        eng = CalibrationEngine(min_samples_to_fit=2, refit_every=2)
        attach_to_bus(eng, bus)
        bus.publish(Event(kind="preflight.outcome", data={"ticket_id": "orphan", "success": True}))
        assert eng.report().n == 0


class _FakeDriver:
    def __init__(self) -> None:
        self.bus = EventBus()


class TestAttachToDriver:
    def test_consumes_session_ended_receipts(self) -> None:
        driver = _FakeDriver()
        eng = CalibrationEngine(min_samples_to_fit=2, refit_every=2)
        detach = attach_to_driver(eng, driver)
        driver.bus.publish(
            Event(
                kind=SESSION_ENDED,
                data={"receipt": {"p_success_forecast": 0.8, "success": True}},
            )
        )
        driver.bus.publish(
            Event(
                kind=SESSION_ENDED,
                data={"receipt": {"p_success_forecast": 0.2, "success": False}},
            )
        )
        assert eng.report().n == 2
        detach()
        driver.bus.publish(
            Event(
                kind=SESSION_ENDED,
                data={"receipt": {"p_success_forecast": 0.5, "success": True}},
            )
        )
        assert eng.report().n == 2

    def test_falls_back_to_decision_trace(self) -> None:
        driver = _FakeDriver()
        eng = CalibrationEngine(min_samples_to_fit=2, refit_every=2)
        attach_to_driver(eng, driver)
        driver.bus.publish(
            Event(
                kind=SESSION_ENDED,
                data={
                    "receipt": {
                        "status": "completed",
                        "decisions": [
                            {"kind": "estimate", "data": {"p_success": 0.6}},
                        ],
                    }
                },
            )
        )
        assert eng.report().n == 1

    def test_skips_receipt_without_forecast(self) -> None:
        driver = _FakeDriver()
        eng = CalibrationEngine(min_samples_to_fit=2, refit_every=2)
        attach_to_driver(eng, driver)
        driver.bus.publish(
            Event(kind=SESSION_ENDED, data={"receipt": {"status": "rejected"}})
        )
        assert eng.report().n == 0

    def test_rejects_driver_without_bus(self) -> None:
        class NoBus:
            pass

        eng = CalibrationEngine()
        with pytest.raises(ValueError):
            attach_to_driver(eng, NoBus())


# --- end-to-end ------------------------------------------------------


class TestEndToEnd:
    def test_full_loop_overconfidence_then_correction(self) -> None:
        """An overconfident forecaster's mistakes are quantified, then
        corrected by the calibrator. The post-correction report is
        materially better on every metric."""
        eng = CalibrationEngine(min_samples_to_fit=50, refit_every=50)
        train = _make_miscalibrated_stream(1000, bias=-0.3, seed=24)
        for s in train:
            eng.observe(s.p_forecast, s.outcome, source="preflight")
        # Report on raw observations (the engine stores raw forecasts).
        raw = eng.report(source="preflight")
        # Build a "calibrated stream" by running calibrate() over the
        # same forecasts and checking ECE on the recalibrated values.
        test = _make_miscalibrated_stream(1000, bias=-0.3, seed=25)
        adjusted = [
            CalibrationSample(p_forecast=eng.calibrate(s.p_forecast, source="preflight"), outcome=s.outcome)
            for s in test
        ]
        bins_adj = reliability_bins(adjusted, n_bins=10)
        ece_adj = expected_calibration_error(bins_adj)
        brier_adj = brier_score(adjusted)
        # Should be materially better than raw.
        assert ece_adj < raw.ece * 0.5
        assert brier_adj < raw.brier

    def test_report_marks_well_calibrated_as_ok(self) -> None:
        eng = CalibrationEngine(
            min_samples_to_fit=50,
            refit_every=50,
            ece_threshold=0.05,
            drift_threshold=0.10,
        )
        eng.observe_many(_make_perfect_stream(2000, seed=26))
        report = eng.report()
        assert report.ok is True

    def test_report_marks_miscalibrated_as_not_ok(self) -> None:
        eng = CalibrationEngine(
            min_samples_to_fit=50,
            refit_every=10_000,  # never auto-fit
            ece_threshold=0.05,
        )
        eng.observe_many(_make_miscalibrated_stream(500, bias=-0.3, seed=27))
        report = eng.report()
        assert report.ok is False
        assert report.ece > 0.05
