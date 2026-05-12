"""Demo: turning an over-confident forecaster into a trustworthy one.

Coordination engines that drive this runtime make admission, hedging,
and budget decisions on top of `p_success`. If those numbers are
systematically biased, every decision is biased the same way — and
nothing else in the pipeline will notice.

This demo:

  1. Simulates a `preflight` forecaster that's over-confident by ~30%.
  2. Pipes its (forecast, outcome) stream into a `CalibrationEngine`.
  3. Prints a reliability diagram BEFORE and AFTER recalibration.
  4. Reports Brier / log loss / ECE / MCE deltas.
  5. Injects distribution drift and watches the engine raise the alarm.
  6. Refits and shows recovery.

Run:

    python -m examples.calibration_demo
"""
from __future__ import annotations

import random

from agi.calibration import (
    CAL_DRIFT,
    CalibrationEngine,
    CalibrationSample,
    expected_calibration_error,
    log_loss,
    brier_score,
    reliability_bins,
)
from agi.events import EventBus


def _stream(n: int, *, bias: float, seed: int):
    """Emit (forecast, true outcome) pairs where the true rate is
    biased relative to the forecast: y ~ Bernoulli(clip(p + bias))."""
    rng = random.Random(seed)
    for _ in range(n):
        p = rng.random()
        q = max(0.0, min(1.0, p + bias))
        yield CalibrationSample(p_forecast=p, outcome=rng.random() < q)


def _print_reliability(samples, title: str) -> None:
    bins = reliability_bins(samples, n_bins=10)
    ece = expected_calibration_error(bins)
    print(f"\n{title}")
    print("  bin       forecast   empirical   gap     n")
    for b in bins:
        gap = b.empirical_rate - b.forecast_mean
        bar = "█" * int(abs(gap) * 40)
        side = "+" if gap > 0 else "-"
        print(
            f"  [{b.p_lo:.1f}-{b.p_hi:.1f}]   {b.forecast_mean:.3f}      "
            f"{b.empirical_rate:.3f}      {side}{abs(gap):.3f} {bar:<14}  n={b.n}"
        )
    print(f"  ECE = {ece:.4f}   Brier = {brier_score(samples):.4f}   "
          f"LogLoss = {log_loss(samples):.4f}")


def main() -> None:
    bus = EventBus()
    drift_alarms = []
    bus.subscribe(lambda ev: drift_alarms.append(ev), kind=CAL_DRIFT)

    eng = CalibrationEngine(
        method="isotonic",
        # Short window so the demo reaches a saturated state quickly.
        window=1200,
        min_samples_to_fit=100,
        refit_every=100,
        ece_threshold=0.04,
        drift_threshold=0.07,
        drift_recent_frac=0.30,
        drift_min_recent=80,
        bus=bus,
    )

    print("=" * 70)
    print("CalibrationEngine demo — preflight forecaster, bias = -0.30")
    print("=" * 70)

    # --- phase 1: feed the engine 1000 over-confident observations ---
    train = list(_stream(1000, bias=-0.30, seed=42))
    for s in train:
        eng.observe(s.p_forecast, s.outcome, source="preflight")

    # --- phase 2: held-out test set, compare raw vs calibrated ------
    test = list(_stream(1000, bias=-0.30, seed=99))
    _print_reliability(test, "BEFORE calibration (raw preflight forecasts)")
    adjusted = [
        CalibrationSample(p_forecast=eng.calibrate(s.p_forecast, source="preflight"),
                          outcome=s.outcome)
        for s in test
    ]
    _print_reliability(adjusted, "AFTER  calibration (engine-corrected)")

    raw_ece = expected_calibration_error(reliability_bins(test, n_bins=10))
    cal_ece = expected_calibration_error(reliability_bins(adjusted, n_bins=10))
    print(f"\nECE improvement: {raw_ece:.4f}  →  {cal_ece:.4f}  "
          f"({(1 - cal_ece / max(raw_ece, 1e-9)) * 100:.1f}% reduction)")

    # --- phase 3: trigger drift by shifting the bias ---------------
    print("\n" + "=" * 70)
    print("Injecting distribution shift: bias flips to +0.30")
    print("=" * 70)
    drift_alarms.clear()
    for s in _stream(800, bias=+0.30, seed=7):
        eng.observe(s.p_forecast, s.outcome, source="preflight")
    mid_report = eng.report(source="preflight")
    drifted = mid_report.drift_score > eng.drift_threshold
    print(f"\n  drift_score = {mid_report.drift_score:.4f}  "
          f"threshold = {eng.drift_threshold:.2f}  "
          f"→ {'DRIFT DETECTED' if drifted else 'within bounds'}")
    print(f"  CAL_DRIFT events observed on bus: {len(drift_alarms)} "
          "(emitted on rising edge into drifted state)")
    print(f"  Engine ok flag: {mid_report.ok}  "
          "(coordinator should pause / refit / route around this forecaster)")

    # --- phase 4: refit and validate recovery ---------------------
    print("\n" + "=" * 70)
    print("Refitting and validating against the new distribution")
    print("=" * 70)
    # Continue observing under the new regime until the recent window
    # dominates and the calibrator adapts on its own (refit_every=100).
    for s in _stream(2000, bias=+0.30, seed=11):
        eng.observe(s.p_forecast, s.outcome, source="preflight")
    test2 = list(_stream(1000, bias=+0.30, seed=13))
    adjusted2 = [
        CalibrationSample(p_forecast=eng.calibrate(s.p_forecast, source="preflight"),
                          outcome=s.outcome)
        for s in test2
    ]
    raw2 = expected_calibration_error(reliability_bins(test2, n_bins=10))
    cal2 = expected_calibration_error(reliability_bins(adjusted2, n_bins=10))
    print(f"\nPost-shift ECE: raw = {raw2:.4f}, calibrated = {cal2:.4f}")
    report = eng.report(source="preflight")
    print(f"Final engine report: n={report.n}, ECE={report.ece:.4f}, "
          f"Brier={report.brier:.4f}, drift={report.drift_score:.4f}, "
          f"ok={report.ok}")


if __name__ == "__main__":
    main()
