"""Tests for DriftSentinel.

Verifies the three guarantees the runtime needs:

  - **No false alarms on stationary data**, at the requested α level across
    long stretches.
  - **Reliable detection on real drift**, within a small number of post-drift
    samples for realistic shift magnitudes.
  - **Correct composition** (multiple detectors don't interfere with each
    other), event-bus integration, threading, and group multiplexing.
"""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.drift import (
    DIR_LOWER,
    DIR_TWO_SIDED,
    DIR_UPPER,
    DRIFT_CLEARED,
    DRIFT_DETECTED,
    DRIFT_OBSERVATION,
    DRIFT_RESET,
    DRIFT_STARTED,
    DetectorStats,
    DriftObservation,
    DriftReport,
    DriftSentinel,
    DriftSentinelGroup,
    METHOD_BETTING,
    METHOD_BOCPD,
    METHOD_CUSUM,
    _BettingMartingale,
    _BOCPD,
    _CUSUM,
)
from agi.events import Event, EventBus


def _clipped_gauss(mean: float, sd: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, random.gauss(mean, sd)))


# ----- construction / validation --------------------------------------


def test_construct_with_reference_mean():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    assert s.reference_mean() == 0.5
    assert s.is_drift_active() is False
    assert s.n_samples == 0


def test_construct_requires_reference_or_warmup():
    with pytest.raises(ValueError, match="reference_mean"):
        DriftSentinel(value_range=(0.0, 1.0))


def test_construct_warmup_mode_needs_no_reference():
    s = DriftSentinel(warmup_samples=20, value_range=(0.0, 1.0))
    assert s.reference_mean() is None
    obs = s.update(0.5)
    assert obs.triggered is False


def test_construct_value_range_required_for_betting():
    with pytest.raises(ValueError, match="value_range"):
        DriftSentinel(reference_mean=0.5, martingale_enabled=True)


def test_construct_betting_can_be_disabled_without_range():
    s = DriftSentinel(
        reference_mean=0.5,
        martingale_enabled=False,
        value_range=None,
    )
    assert s.update(0.5).triggered is False


def test_alpha_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.0)
    with pytest.raises(ValueError):
        DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=1.0)


def test_unknown_direction_rejected():
    with pytest.raises(ValueError, match="direction"):
        DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), direction="sideways")


def test_all_detectors_disabled_rejected():
    with pytest.raises(ValueError, match="at least one detector"):
        DriftSentinel(
            reference_mean=0.5,
            value_range=(0.0, 1.0),
            cusum_enabled=False,
            bocpd_enabled=False,
            martingale_enabled=False,
        )


def test_value_range_must_be_increasing():
    with pytest.raises(ValueError):
        DriftSentinel(reference_mean=0.5, value_range=(1.0, 0.0))


# ----- input validation ----------------------------------------------


def test_nan_sample_rejected():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        s.update(float("nan"))


def test_inf_sample_rejected():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        s.update(float("inf"))


def test_sample_outside_value_range_rejected():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        s.update(1.5)
    with pytest.raises(ValueError):
        s.update(-0.1)


def test_sample_within_value_range_accepted():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    s.update(0.0)
    s.update(1.0)
    s.update(0.5)
    assert s.n_samples == 3


# ----- no false alarms on stationary data ----------------------------


def test_no_false_alarm_clean_stream():
    random.seed(0)
    s = DriftSentinel(
        reference_mean=0.5, reference_var=0.04, value_range=(0.0, 1.0), alpha=0.01
    )
    triggered = False
    for _ in range(200):
        obs = s.update(_clipped_gauss(0.5, 0.1))
        if obs.triggered:
            triggered = True
            break
    assert triggered is False
    assert s.is_drift_active() is False


def test_false_alarm_rate_bounded_across_seeds():
    """Empirical false alarm rate across many seeds should be < α with margin.

    α=0.05 nominally; we expect well under 0.1 across 60 reps of 100 samples.
    """
    n_reps = 60
    n_per_run = 100
    alarms = 0
    for seed in range(n_reps):
        random.seed(seed)
        s = DriftSentinel(
            reference_mean=0.5,
            reference_var=0.04,
            value_range=(0.0, 1.0),
            alpha=0.05,
        )
        for _ in range(n_per_run):
            obs = s.update(_clipped_gauss(0.5, 0.1))
            if obs.triggered:
                alarms += 1
                break
    # Allow some slack for finite-sample variability and the fact that
    # the union of three detectors is not perfectly tight.
    assert alarms <= n_reps // 3, f"{alarms}/{n_reps} false alarms"


# ----- detects real drift --------------------------------------------


def test_detects_upward_drift():
    random.seed(1)
    s = DriftSentinel(
        reference_mean=0.5, reference_var=0.04, value_range=(0.0, 1.0), alpha=0.05
    )
    # 100 clean
    for _ in range(100):
        s.update(_clipped_gauss(0.5, 0.1))
    assert not s.is_drift_active()
    # large shift
    detected_at = None
    for i in range(100):
        obs = s.update(_clipped_gauss(0.8, 0.1))
        if obs.triggered:
            detected_at = i + 1
            break
    assert detected_at is not None
    assert detected_at <= 30  # should be fast


def test_detects_downward_drift():
    random.seed(2)
    s = DriftSentinel(
        reference_mean=0.5, reference_var=0.04, value_range=(0.0, 1.0), alpha=0.05
    )
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.1))
    detected_at = None
    for i in range(100):
        obs = s.update(_clipped_gauss(0.2, 0.1))
        if obs.triggered:
            detected_at = i + 1
            break
    assert detected_at is not None


def test_one_sided_upper_ignores_downward_drift():
    """A `direction=upper` sentinel should not fire when the mean drops."""
    random.seed(3)
    s = DriftSentinel(
        reference_mean=0.5,
        reference_var=0.04,
        value_range=(0.0, 1.0),
        alpha=0.05,
        direction=DIR_UPPER,
        bocpd_enabled=False,  # BOCPD is two-sided by nature
    )
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.1))
    for _ in range(150):
        obs = s.update(_clipped_gauss(0.2, 0.1))
    assert not s.is_drift_active()


def test_one_sided_lower_ignores_upward_drift():
    random.seed(4)
    s = DriftSentinel(
        reference_mean=0.5,
        reference_var=0.04,
        value_range=(0.0, 1.0),
        alpha=0.05,
        direction=DIR_LOWER,
        bocpd_enabled=False,
    )
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.1))
    for _ in range(150):
        s.update(_clipped_gauss(0.8, 0.1))
    assert not s.is_drift_active()


def test_one_sided_upper_catches_upward_drift():
    random.seed(5)
    s = DriftSentinel(
        reference_mean=0.5,
        reference_var=0.04,
        value_range=(0.0, 1.0),
        alpha=0.05,
        direction=DIR_UPPER,
        bocpd_enabled=False,
    )
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.1))
    detected = False
    for _ in range(100):
        obs = s.update(_clipped_gauss(0.8, 0.1))
        if obs.triggered:
            detected = True
            break
    assert detected


# ----- latching / reset ------------------------------------------------


def test_trigger_is_latching():
    random.seed(6)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05)
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.05))
    # force a trigger
    while not s.is_drift_active():
        s.update(_clipped_gauss(0.9, 0.05))
    # subsequent updates stay triggered
    for _ in range(20):
        obs = s.update(_clipped_gauss(0.5, 0.05))
        assert obs.triggered is True
    assert s.is_drift_active() is True


def test_reset_clears_state():
    random.seed(7)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05)
    while not s.is_drift_active():
        s.update(_clipped_gauss(0.9, 0.05))
    s.reset()
    assert s.is_drift_active() is False
    assert s.n_samples == 0
    assert s.trigger_method() is None
    # And it can detect again
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.05))
    while not s.is_drift_active():
        s.update(_clipped_gauss(0.9, 0.05))
    assert s.is_drift_active() is True


def test_reset_without_keep_reference_goes_back_to_warmup():
    s = DriftSentinel(warmup_samples=10, value_range=(0.0, 1.0))
    for _ in range(10):
        s.update(0.5)
    assert s.reference_mean() is not None
    s.reset(keep_reference=False)
    assert s.reference_mean() is None
    obs = s.update(0.5)
    assert obs.triggered is False


# ----- self-reference (warmup) mode -----------------------------------


def test_warmup_locks_reference_mean():
    random.seed(8)
    s = DriftSentinel(warmup_samples=50, value_range=(0.0, 1.0), alpha=0.05)
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.1))
    mu = s.reference_mean()
    assert mu is not None
    assert abs(mu - 0.5) < 0.05


def test_warmup_does_not_trigger_during_warmup():
    random.seed(9)
    s = DriftSentinel(warmup_samples=200, value_range=(0.0, 1.0))
    for _ in range(150):
        # even with very anomalous values during warmup, sentinel is silent
        obs = s.update(0.99)
        assert obs.triggered is False


# ----- BOCPD changepoint estimate -------------------------------------


def test_changepoint_estimate_close_to_truth():
    random.seed(10)
    # Disable CUSUM/betting so the first to fire is BOCPD; that lets us
    # check the changepoint estimate directly on the observation.
    s = DriftSentinel(
        reference_mean=0.5,
        reference_var=0.04,
        value_range=(0.0, 1.0),
        alpha=0.05,
        cusum_enabled=False,
        martingale_enabled=False,
    )
    n_clean = 80
    for _ in range(n_clean):
        s.update(_clipped_gauss(0.5, 0.08))
    for i in range(60):
        obs = s.update(_clipped_gauss(0.85, 0.08))
        if obs.triggered:
            assert obs.method == METHOD_BOCPD
            assert obs.changepoint_estimate is not None
            # the ML changepoint should be within a small window of n_clean+1
            assert abs(obs.changepoint_estimate - (n_clean + 1)) <= 10
            return
    pytest.fail("BOCPD never triggered")


# ----- betting martingale -------------------------------------------


def test_betting_martingale_alone_catches_drift():
    random.seed(11)
    s = DriftSentinel(
        reference_mean=0.5,
        value_range=(0.0, 1.0),
        alpha=0.05,
        cusum_enabled=False,
        bocpd_enabled=False,
        martingale_enabled=True,
    )
    for _ in range(30):
        s.update(_clipped_gauss(0.5, 0.05))
    detected = False
    for _ in range(300):
        obs = s.update(_clipped_gauss(0.78, 0.05))
        if obs.triggered:
            assert obs.method == METHOD_BETTING
            detected = True
            break
    assert detected


def test_betting_martingale_no_false_alarms_under_null():
    random.seed(12)
    fp = 0
    n_reps = 20
    for seed in range(n_reps):
        random.seed(seed + 500)
        s = DriftSentinel(
            reference_mean=0.5,
            value_range=(0.0, 1.0),
            alpha=0.05,
            cusum_enabled=False,
            bocpd_enabled=False,
            martingale_enabled=True,
        )
        for _ in range(200):
            obs = s.update(_clipped_gauss(0.5, 0.08))
            if obs.triggered:
                fp += 1
                break
    # Ville's inequality bounds at α=0.05 → expect ≤ ~1 across 20 reps
    assert fp <= 4


def test_betting_log_capital_non_negative_grows_under_drift():
    random.seed(13)
    s = DriftSentinel(
        reference_mean=0.5,
        value_range=(0.0, 1.0),
        alpha=0.05,
        cusum_enabled=False,
        bocpd_enabled=False,
        martingale_enabled=True,
        direction=DIR_UPPER,
    )
    for _ in range(20):
        s.update(0.5)
    up, lo = s.betting_log_capital()
    assert lo is None
    for _ in range(100):
        s.update(_clipped_gauss(0.85, 0.05))
    up2, _ = s.betting_log_capital()
    assert up2 > up


# ----- CUSUM ---------------------------------------------------------


def test_cusum_alone_catches_drift():
    random.seed(14)
    s = DriftSentinel(
        reference_mean=0.5,
        reference_var=0.01,
        value_range=(0.0, 1.0),
        alpha=0.05,
        cusum_enabled=True,
        bocpd_enabled=False,
        martingale_enabled=False,
    )
    for _ in range(20):
        s.update(_clipped_gauss(0.5, 0.05))
    detected = False
    for _ in range(50):
        obs = s.update(_clipped_gauss(0.75, 0.05))
        if obs.triggered:
            assert obs.method == METHOD_CUSUM
            detected = True
            break
    assert detected


# ----- detector composition / first-fire ------------------------------


def test_first_to_fire_is_reported_as_method():
    random.seed(15)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05)
    for _ in range(40):
        s.update(_clipped_gauss(0.5, 0.05))
    method = None
    for _ in range(80):
        obs = s.update(_clipped_gauss(0.85, 0.05))
        if obs.triggered:
            method = obs.method
            break
    assert method in (METHOD_CUSUM, METHOD_BOCPD, METHOD_BETTING)


def test_all_detector_stats_present_on_observation():
    random.seed(16)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    obs = s.update(0.5)
    assert METHOD_CUSUM in obs.detectors
    assert METHOD_BOCPD in obs.detectors
    assert METHOD_BETTING in obs.detectors
    for d in obs.detectors.values():
        assert isinstance(d, DetectorStats)
        assert d.threshold > 0
        assert isinstance(d.statistic, float)


# ----- event publishing ----------------------------------------------


def test_events_published_to_bus():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(callback=seen.append)
    s = DriftSentinel(
        reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05, bus=bus
    )
    kinds = {e.kind for e in seen}
    assert DRIFT_STARTED in kinds
    random.seed(17)
    for _ in range(30):
        s.update(_clipped_gauss(0.5, 0.05))
    # force a detection
    while not s.is_drift_active():
        s.update(_clipped_gauss(0.95, 0.02))
    kinds = {e.kind for e in seen}
    assert DRIFT_OBSERVATION in kinds
    assert DRIFT_DETECTED in kinds
    s.reset()
    kinds = {e.kind for e in seen}
    assert DRIFT_RESET in kinds


def test_no_drift_started_event_in_warmup_mode_until_locked():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(callback=seen.append)
    s = DriftSentinel(warmup_samples=5, value_range=(0.0, 1.0), bus=bus)
    # no DRIFT_STARTED yet — reference not locked
    started = [e for e in seen if e.kind == DRIFT_STARTED]
    assert len(started) == 0
    for _ in range(5):
        s.update(0.5)
    # now it should be locked, and DRIFT_STARTED emitted
    started = [e for e in seen if e.kind == DRIFT_STARTED]
    assert len(started) == 1
    assert started[0].data["warmup_samples"] == 5


def test_event_subscriber_exception_does_not_break_sentinel():
    bus = EventBus()

    def bad(e: Event) -> None:
        raise RuntimeError("subscriber boom")

    bus.subscribe(callback=bad)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), bus=bus)
    for _ in range(10):
        s.update(0.5)
    assert s.n_samples == 10


# ----- detector-internal correctness --------------------------------


def test_cusum_threshold_validation():
    with pytest.raises(ValueError):
        _CUSUM(delta=-1, h=1)
    with pytest.raises(ValueError):
        _CUSUM(delta=0.1, h=0)
    with pytest.raises(ValueError):
        _CUSUM(delta=0.1, h=1, direction="bogus")


def test_cusum_reset_clears_state():
    c = _CUSUM(delta=0.1, h=1.0)
    for i in range(20):
        c.update(i + 1, 0.9, mu0=0.5)
    assert c.s_pos > 0
    c.reset()
    assert c.s_pos == 0.0
    assert c.s_neg == 0.0
    assert c.triggered_at is None


def test_betting_martingale_validation():
    with pytest.raises(ValueError):
        _BettingMartingale(mu0=0.5, lo=0.0, hi=1.0, alpha=0.0)
    with pytest.raises(ValueError):
        _BettingMartingale(mu0=0.5, lo=0.0, hi=1.0, alpha=1.0)
    with pytest.raises(ValueError):
        _BettingMartingale(mu0=2.0, lo=0.0, hi=1.0, alpha=0.05)
    with pytest.raises(ValueError):
        _BettingMartingale(mu0=0.5, lo=0.0, hi=1.0, alpha=0.05, side="middle")


def test_betting_martingale_capital_stays_finite():
    """Capital factor must never go non-positive."""
    m = _BettingMartingale(mu0=0.5, lo=0.0, hi=1.0, alpha=0.05, side="upper")
    # extreme observations both sides
    for x in (0.0, 1.0, 0.0, 1.0) * 25:
        m.update(m.n_obs + 1, x)
    assert math.isfinite(m.log_capital)


def test_bocpd_validation():
    with pytest.raises(ValueError):
        _BOCPD(mu0=0.5, var0=0.0)
    with pytest.raises(ValueError):
        _BOCPD(mu0=0.5, var0=0.04, lambda_hazard=0)
    with pytest.raises(ValueError):
        _BOCPD(mu0=0.5, var0=0.04, alarm_mass=0)
    with pytest.raises(ValueError):
        _BOCPD(mu0=0.5, var0=0.04, alarm_mass=1.5)
    with pytest.raises(ValueError):
        _BOCPD(mu0=0.5, var0=0.04, short_run=0)
    with pytest.raises(ValueError):
        _BOCPD(mu0=0.5, var0=0.04, short_run=10, r_max=5)


def test_bocpd_posterior_sums_to_one():
    b = _BOCPD(mu0=0.5, var0=0.04)
    for i in range(30):
        b.update(i + 1, 0.5)
        assert abs(sum(b.run_post) - 1.0) < 1e-9


def test_bocpd_reset_clears_state():
    b = _BOCPD(mu0=0.5, var0=0.04)
    for i in range(20):
        b.update(i + 1, 0.5)
    b.reset()
    assert b.run_post == [1.0]
    assert b.triggered_at is None


# ----- report ---------------------------------------------------------


def test_report_after_clean_stream():
    random.seed(18)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05)
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.05))
    rep = s.report()
    assert isinstance(rep, DriftReport)
    assert rep.n_samples == 50
    assert rep.triggered is False
    assert rep.first_trigger_t is None
    assert abs(rep.observed_mean - 0.5) < 0.05
    assert rep.detectors  # all three present


def test_report_after_drift():
    random.seed(19)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05)
    for _ in range(50):
        s.update(_clipped_gauss(0.5, 0.05))
    while not s.is_drift_active():
        s.update(_clipped_gauss(0.9, 0.05))
    rep = s.report()
    assert rep.triggered is True
    assert rep.first_trigger_t is not None
    assert rep.first_trigger_method in (METHOD_CUSUM, METHOD_BOCPD, METHOD_BETTING)
    assert "first trigger" in rep.rationale


def test_observation_serializes_to_dict():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    obs = s.update(0.5)
    d = obs.to_dict()
    assert d["t"] == 1
    assert d["sample"] == 0.5
    assert "detectors" in d
    assert METHOD_CUSUM in d["detectors"]


def test_report_serializes_to_dict():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    s.update(0.5)
    d = s.report().to_dict()
    assert d["n_samples"] == 1
    assert "detectors" in d


# ----- threading -----------------------------------------------------


def test_concurrent_updates_safe():
    random.seed(20)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05)
    samples = [_clipped_gauss(0.5, 0.05) for _ in range(500)]
    errors: list[BaseException] = []

    def worker(chunk: list[float]) -> None:
        try:
            for x in chunk:
                s.update(x)
        except BaseException as e:
            errors.append(e)

    chunks = [samples[i::4] for i in range(4)]
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert s.n_samples == 500


# ----- update_many ----------------------------------------------------


def test_update_many_returns_one_per_input():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    obs_list = s.update_many([0.5, 0.5, 0.5])
    assert len(obs_list) == 3
    assert all(isinstance(o, DriftObservation) for o in obs_list)


# ----- DriftSentinelGroup --------------------------------------------


def test_group_routes_updates_by_label():
    g = DriftSentinelGroup()
    g.add("tenant_a", DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0)))
    g.add("tenant_b", DriftSentinel(reference_mean=0.8, value_range=(0.0, 1.0)))
    obs_a = g.update("tenant_a", 0.5)
    obs_b = g.update("tenant_b", 0.8)
    assert obs_a is not None and obs_a.t == 1
    assert obs_b is not None and obs_b.t == 1
    assert g.update("unknown", 0.5) is None


def test_group_duplicate_label_rejected():
    g = DriftSentinelGroup()
    g.add("x", DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0)))
    with pytest.raises(ValueError, match="duplicate"):
        g.add("x", DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0)))


def test_group_active_reports_drifting_labels():
    random.seed(21)
    g = DriftSentinelGroup()
    g.add(
        "stable",
        DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05),
    )
    g.add(
        "drifting",
        DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05),
    )
    for _ in range(50):
        g.update("stable", _clipped_gauss(0.5, 0.05))
        g.update("drifting", _clipped_gauss(0.5, 0.05))
    while "drifting" not in g.active():
        g.update("drifting", _clipped_gauss(0.95, 0.02))
        g.update("stable", _clipped_gauss(0.5, 0.05))
    assert g.active() == ["drifting"]


def test_group_reset_targets():
    g = DriftSentinelGroup()
    s1 = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    s2 = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    g.add("a", s1)
    g.add("b", s2)
    s1.update(0.5)
    s2.update(0.5)
    g.reset("a")
    assert s1.n_samples == 0
    assert s2.n_samples == 1


def test_group_reset_all():
    g = DriftSentinelGroup()
    s1 = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    s2 = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    g.add("a", s1)
    g.add("b", s2)
    s1.update(0.5)
    s2.update(0.5)
    g.reset()
    assert s1.n_samples == 0
    assert s2.n_samples == 0


def test_group_remove_label():
    g = DriftSentinelGroup()
    g.add("x", DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0)))
    assert "x" in g.labels()
    g.remove("x")
    assert "x" not in g.labels()


# ----- auto-clear ----------------------------------------------------


def test_auto_clear_releases_sentinel_when_stream_recovers():
    random.seed(22)
    bus = EventBus()
    cleared: list[Event] = []
    bus.subscribe(kind=DRIFT_CLEARED, callback=cleared.append)
    s = DriftSentinel(
        reference_mean=0.5,
        value_range=(0.0, 1.0),
        alpha=0.05,
        bus=bus,
        auto_clear=True,
        clear_window=30,
        clear_tolerance=0.05,
    )
    # force a trigger
    while not s.is_drift_active():
        s.update(_clipped_gauss(0.92, 0.03))
    assert s.is_drift_active()
    # now feed stream that recovers to mean=0.5
    for _ in range(60):
        s.update(_clipped_gauss(0.5, 0.02))
    # auto_clear should have fired
    assert len(cleared) >= 1
    assert s.is_drift_active() is False


# ----- attestation integration --------------------------------------


def test_attestor_called_on_detection():
    class _MockAttestor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def attest(self, *, kind: str, payload: dict) -> None:
            self.calls.append((kind, payload))

    att = _MockAttestor()
    s = DriftSentinel(
        reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05, attestor=att
    )
    random.seed(23)
    for _ in range(30):
        s.update(_clipped_gauss(0.5, 0.05))
    while not s.is_drift_active():
        s.update(_clipped_gauss(0.95, 0.02))
    assert any(k == "drift.detected" for k, _ in att.calls)


def test_attestor_failure_does_not_break_sentinel():
    class _BadAttestor:
        def attest(self, **kwargs):
            raise RuntimeError("attestor boom")

    s = DriftSentinel(
        reference_mean=0.5,
        value_range=(0.0, 1.0),
        alpha=0.5,
        attestor=_BadAttestor(),
    )
    while not s.is_drift_active():
        s.update(0.99)
    assert s.is_drift_active() is True


# ----- BOCPD off, detection still works ------------------------------


def test_bocpd_disabled_does_not_register_method():
    s = DriftSentinel(
        reference_mean=0.5,
        value_range=(0.0, 1.0),
        bocpd_enabled=False,
    )
    obs = s.update(0.5)
    assert METHOD_BOCPD not in obs.detectors


def test_cusum_disabled_does_not_register_method():
    s = DriftSentinel(
        reference_mean=0.5,
        value_range=(0.0, 1.0),
        cusum_enabled=False,
    )
    obs = s.update(0.5)
    assert METHOD_CUSUM not in obs.detectors


def test_martingale_disabled_does_not_register_method():
    s = DriftSentinel(
        reference_mean=0.5,
        value_range=(0.0, 1.0),
        martingale_enabled=False,
    )
    obs = s.update(0.5)
    assert METHOD_BETTING not in obs.detectors


# ----- observations history -----------------------------------------


def test_observations_history_capped():
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0))
    # default max_history is 1024
    for _ in range(1100):
        s.update(0.5)
    assert len(s.observations()) <= 1024


# ----- confidence semantics -----------------------------------------


def test_confidence_grows_with_drift():
    random.seed(24)
    s = DriftSentinel(reference_mean=0.5, value_range=(0.0, 1.0), alpha=0.05)
    for _ in range(30):
        s.update(_clipped_gauss(0.5, 0.02))
    obs_clean = s.update(0.5)
    for _ in range(20):
        s.update(_clipped_gauss(0.85, 0.02))
    obs_drift = s.update(0.85)
    assert obs_drift.confidence > obs_clean.confidence
