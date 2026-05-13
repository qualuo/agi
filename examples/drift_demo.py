"""DriftSentinel demo — anytime-valid sequential drift detection.

What this shows:

  1. A `DriftSentinel` watches a synthetic stream of `p_success` outcomes
     coming from a routing policy in production.
  2. The stream is stationary at 0.85 for the first 200 ticks (the
     calibrated baseline), then silently shifts to 0.65 at tick 201 —
     the kind of regression that happens when an upstream model is
     swapped, a tool changes its semantics, or a tenant's traffic mix
     turns adversarial.
  3. The sentinel triggers within a small number of post-drift samples,
     reports which detector fired, and emits a `drift.detected` event on
     the runtime bus. Downstream modules (calibration, conformal, policy)
     can subscribe and re-arm.
  4. The sentinel also estimates *when* the drift happened (BOCPD's
     changepoint mode) so the coordinator can roll back stale state to a
     known-good cut.

Run it:
    python examples/drift_demo.py
"""
from __future__ import annotations

import random

from agi.drift import (
    DRIFT_DETECTED,
    DriftSentinel,
    METHOD_BETTING,
    METHOD_BOCPD,
    METHOD_CUSUM,
)
from agi.events import Event, EventBus


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def main() -> None:
    bus = EventBus()
    detected: list[Event] = []
    bus.subscribe(kind=DRIFT_DETECTED, callback=detected.append)

    sentinel = DriftSentinel(
        reference_mean=0.85,        # calibrated baseline p_success
        reference_var=0.02,         # rough scale of stream noise
        value_range=(0.0, 1.0),     # bounded outcome (probability)
        alpha=0.01,                 # ≤ 1% false-alarm rate, uniformly over time
        bus=bus,
        name="p_success",
    )

    random.seed(0)
    stable_n = 200
    drift_n = 60

    print("phase 1: stable baseline (mean=0.85)")
    print("-" * 56)
    for t in range(stable_n):
        x = _clip(random.gauss(0.85, 0.05))
        obs = sentinel.update(x)
        if obs.triggered:
            raise RuntimeError(f"false alarm at t={t+1}")
        if (t + 1) % 50 == 0:
            up, lo = sentinel.betting_log_capital()
            print(
                f"  t={t+1:>3d}  cusum={sentinel.cusum_statistic():.3f}  "
                f"bocpd_alarm={sentinel.bocpd_alarm_mass():.3f}  "
                f"betting_logK_up={up:+.3f}  conf={obs.confidence:.3f}"
            )

    print(f"  → no false alarm across {stable_n} samples at α=0.01")

    print()
    print("phase 2: silent regression at t=201 (mean drops to 0.65)")
    print("-" * 56)

    triggered_at: int | None = None
    method: str | None = None
    changepoint: int | None = None
    for t in range(drift_n):
        x = _clip(random.gauss(0.65, 0.05))
        obs = sentinel.update(x)
        if obs.triggered:
            triggered_at = obs.t
            method = obs.method
            changepoint = obs.changepoint_estimate
            print(
                f"  t={obs.t:>3d}  DRIFT DETECTED via {method}  "
                f"τ̂={changepoint}  confidence={obs.confidence:.3f}"
            )
            for name, det in obs.detectors.items():
                pad = " " * 4
                print(
                    f"{pad}  {name:>18s}  stat={det.statistic:+.4f}  "
                    f"thr={det.threshold:.4f}  fired={det.triggered}"
                )
            break

    assert triggered_at is not None
    print()
    print(
        f"detected after {triggered_at - stable_n} post-drift samples "
        f"out of {drift_n} (faster is better)"
    )
    print(f"detector that fired first: {method}")
    print(
        f"changepoint estimate: t≈{changepoint}  "
        f"(ground truth: t={stable_n + 1})"
    )
    print(
        f"events on the runtime bus: "
        f"{[e.kind for e in detected]}"
    )

    print()
    print("how the coordination engine would react:")
    print("  - calibrator.refit(from=τ̂)")
    print("  - conformal.invalidate_calibration_set()")
    print("  - policy_lab.flag_estimates_stale()")
    print("  - strategist.bias_toward(DEFER) until sentinel.reset()")
    print("  - coordinator.enter_safe_mode(reason=method)")

    print()
    rep = sentinel.report()
    print("report:")
    print(f"  n_samples = {rep.n_samples}")
    print(f"  observed_mean = {rep.observed_mean:.4f}  reference = {rep.reference_mean:.4f}")
    print(f"  rationale = {rep.rationale}")


if __name__ == "__main__":
    main()
