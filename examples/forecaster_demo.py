"""Forecaster demo — calibrated probabilistic forecasting end-to-end.

Scenario
--------

A coordination engine fans incoming jobs out to three models with
*different latency distributions*.  We don't know any of them up
front; what we have is a stream of (forecast distribution, realised
latency_ms) pairs from each model.  We want to:

  1. Score each model with a strictly proper rule (CRPS).
  2. Test whether each model's forecasts are *calibrated* under an
     **anytime-valid e-process** — i.e. with no fixed sample size and
     freedom of optional continuation.
  3. Emit a conformal prediction interval for the next latency.
  4. Mix the three models with **Hedge** so the runtime tracks the
     best-performing model with O(√(T log K)) cumulative regret.
  5. Recalibrate the worst-performing model post-hoc via PIT
     recalibration.

The whole flow is exercised against three artificial generators:
  * `fast`  — truth N(80, 10);   forecast N(80, 10)    (calibrated)
  * `slow`  — truth N(120, 25);  forecast N(100, 20)   (biased)
  * `noisy` — truth N(100, 50);  forecast N(100, 20)   (under-dispersed)

Everything is stdlib-only.  Run with:  `python examples/forecaster_demo.py`.
"""
from __future__ import annotations

import random

from agi.events import Event, EventBus
from agi.forecaster import (
    CALIB_E_PROCESS,
    Forecaster,
    GaussianForecast,
    POOL_HEDGE,
    SCORE_CRPS,
    RECAL_PIT,
)


def _emit(event: Event) -> None:
    if event.kind in (
        "forecaster.calibration_tested",
        "forecaster.ensemble_updated",
        "forecaster.recalibrated",
    ):
        print(f"[event] {event.kind} {event.data.get('stream_id') or event.data.get('ensemble_id')}")


def main() -> None:
    bus = EventBus()
    bus.subscribe(_emit)
    fcst = Forecaster(bus=bus, random_seed=0)

    rng = random.Random(7)
    for sid in ("fast", "slow", "noisy"):
        fcst.register_stream(sid)

    truth_params = {
        "fast": (80.0, 10.0),
        "slow": (120.0, 25.0),
        "noisy": (100.0, 50.0),
    }
    forecast_params = {
        "fast": (80.0, 10.0),
        "slow": (100.0, 20.0),
        "noisy": (100.0, 20.0),
    }

    for _ in range(800):
        for sid in ("fast", "slow", "noisy"):
            mu_t, sg_t = truth_params[sid]
            mu_f, sg_f = forecast_params[sid]
            y = rng.gauss(mu_t, sg_t)
            fcst.record(sid, GaussianForecast(mu_f, sg_f), y)

    print("\n-- proper scoring --")
    for sid in ("fast", "slow", "noisy"):
        s = fcst.score(sid, SCORE_CRPS)
        print(f"  {sid:>6s}  CRPS mean = {s.mean:.3f}  n = {s.n}")

    print("\n-- anytime-valid calibration test (Ville's inequality, α=0.05) --")
    for sid in ("fast", "slow", "noisy"):
        rep = fcst.calibration_test(sid, method=CALIB_E_PROCESS, alpha=0.05)
        verdict = "REJECT" if rep.rejected else "no-reject"
        print(f"  {sid:>6s}  e-value = {rep.e_value:.3e}  ⇒ {verdict}")

    print("\n-- prediction intervals (online conformal, α=0.1) --")
    for sid in ("fast", "slow", "noisy"):
        iv = fcst.interval(sid, alpha=0.1)
        print(
            f"  {sid:>6s}  [{iv.lower:7.2f}, {iv.upper:7.2f}]  width={iv.width:6.2f}  "
            f"emp.cov = {iv.empirical_coverage:.3f}"
        )

    print("\n-- recalibrate the worst (PIT recalibration) --")
    fcst.recalibrate("slow", method=RECAL_PIT)

    print("\n-- ensemble of three streams (Hedge with regret bound) --")
    rep = fcst.ensemble("ensemble", ["fast", "slow", "noisy"], method=POOL_HEDGE, rule=SCORE_CRPS)
    for sid, w in zip(rep.streams, rep.weights):
        print(f"  {sid:>6s}  weight = {w:.3f}")
    print(f"  cumulative regret bound (Cesa-Bianchi & Lugosi): {rep.cumulative_regret_bound:.2f}")

    print("\n-- coverage report --")
    c = fcst.coverage()
    print(
        f"  streams={c.streams}  obs={c.observations}  scores={c.scores}  "
        f"calibrations={c.calibrations}  rejections={c.rejections}  "
        f"intervals={c.intervals}  ensembles={c.ensembles}"
    )

    print("\n-- live forecast (the ensemble's combined output) --")
    out = fcst.forecast(ensemble_id="ensemble")
    print(f"  type={type(out).__name__}  μ={out.mu:.2f}  σ={out.sigma:.2f}")


if __name__ == "__main__":
    main()
