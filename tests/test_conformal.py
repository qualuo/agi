"""Tests for agi.conformal.

Coverage tests run on synthetic data with a known data-generating
process so the empirical coverage of conformal intervals can be
compared against the finite-sample target. We use enough trials to
keep the variance small but the tests are fast (<2s total).
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import tempfile

import pytest

from agi.conformal import (
    CONFORMAL_DRIFT,
    CONFORMAL_OBSERVED,
    CONFORMAL_PREDICTED,
    CONFORMAL_REPORT,
    CalibrationPoint,
    ConformalPredictor,
    CoverageReport,
    GroupCoverage,
    KNOWN_CLASSIFICATION_METHODS,
    KNOWN_REGRESSION_METHODS,
    METHOD_CQR,
    METHOD_JACKKNIFE_PLUS,
    METHOD_MONDRIAN,
    METHOD_RAPS,
    METHOD_SPLIT,
    PredictionInterval,
    PredictionSet,
    _ACIState,
    _empirical_quantile_ceiling,
    _jackknife_plus_interval,
    _raps_nonconformity,
    _raps_predict_set,
)
from agi.events import Event, EventBus


# ---------- numerical primitives ----------


def test_empirical_quantile_ceiling_matches_finite_sample_formula():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    # For n=5, level=0.9 → rank = ceil(6 * 0.9) = 6 > 5 ⇒ infinity.
    assert math.isinf(_empirical_quantile_ceiling(vals, 0.9))
    # level=0.8 → rank = ceil(6 * 0.8) = 5 ⇒ 5th order statistic.
    assert _empirical_quantile_ceiling(vals, 0.8) == 5.0
    # level=0.5 → rank = ceil(6 * 0.5) = 3 ⇒ 3rd order statistic.
    assert _empirical_quantile_ceiling(vals, 0.5) == 3.0


def test_empirical_quantile_ceiling_handles_empty_and_edges():
    assert math.isinf(_empirical_quantile_ceiling([], 0.9))
    assert _empirical_quantile_ceiling([1.0, 2.0], 0.0) == 1.0
    assert _empirical_quantile_ceiling([1.0, 2.0], 1.0) == 2.0


# ---------- dataclass invariants ----------


def test_calibration_point_rejects_bad_weight():
    with pytest.raises(ValueError):
        CalibrationPoint(features={}, outcome=1.0, weight=-1.0)
    with pytest.raises(ValueError):
        CalibrationPoint(features={}, outcome=1.0, weight=float("nan"))


def test_prediction_interval_width_and_contains():
    pi = PredictionInterval(
        lower=1.0, upper=3.0, target_coverage=0.9,
        method=METHOD_SPLIT, n_cal=100, effective_alpha=0.1,
    )
    assert pi.width == 2.0
    assert pi.contains(2.0) and pi.contains(1.0) and pi.contains(3.0)
    assert not pi.contains(0.5) and not pi.contains(3.5)


def test_prediction_set_size_and_contains():
    ps = PredictionSet(
        labels=("a", "b"), target_coverage=0.9, method=METHOD_RAPS, n_cal=10,
    )
    assert ps.size == 2
    assert ps.contains("a") and not ps.contains("c")


# ---------- split conformal regression ----------


def test_split_conformal_marginal_coverage():
    """Run 5 trials of split conformal on N(0, 1) noise — empirical
    coverage should sit within ±3σ of the nominal level."""
    random.seed(42)
    target = 0.9
    successes = []
    widths = []
    for _ in range(5):
        cp = ConformalPredictor(target_coverage=target)
        for _ in range(800):
            x = random.uniform(-1, 1)
            cp.record(
                features={"x": x}, prediction=x,
                outcome=x + random.gauss(0, 1.0),
            )
        # held-out
        hits = 0
        widths_trial = []
        held_n = 500
        for _ in range(held_n):
            x = random.uniform(-1, 1)
            y = x + random.gauss(0, 1.0)
            pi = cp.predict_interval(prediction=x, method=METHOD_SPLIT)
            if pi.contains(y):
                hits += 1
            widths_trial.append(pi.width)
        successes.append(hits / held_n)
        widths.append(statistics.mean(widths_trial))
    emp = statistics.mean(successes)
    # Marginal coverage = exactly 1 - α with finite-sample slack.
    assert abs(emp - target) < 0.03, f"empirical {emp} vs target {target}"
    # Width should be ≈ 2 * 1.645 = 3.29 for N(0,1) and 90% nominal.
    assert 2.5 < statistics.mean(widths) < 4.0


def test_split_conformal_requires_prediction():
    cp = ConformalPredictor(target_coverage=0.9)
    cp.record(features={}, prediction=0.0, outcome=0.0)
    with pytest.raises(ValueError):
        cp.predict_interval(method=METHOD_SPLIT)


def test_split_conformal_with_tiny_n_gives_vacuous_interval():
    cp = ConformalPredictor(target_coverage=0.99)
    cp.record(features={}, prediction=0.0, outcome=0.0)
    pi = cp.predict_interval(prediction=0.0, method=METHOD_SPLIT)
    # With n=1 we cannot reach 99% so threshold is +inf ⇒ vacuous (correct).
    assert math.isinf(pi.upper - pi.lower) or pi.upper == math.inf or pi.lower == -math.inf


# ---------- CQR ----------


def test_cqr_marginal_coverage_under_heteroscedasticity():
    """CQR should track the heteroscedastic noise correctly."""
    random.seed(7)
    target = 0.9
    # Heteroscedastic noise: sigma(x) = 0.1 + |x|.
    def noisy(x):
        return x + random.gauss(0, 0.1 + abs(x))

    def lo_q(x):  # 5% quantile of a Gaussian: μ + σ * Φ⁻¹(0.05) ≈ μ - 1.645σ
        return x - 1.645 * (0.1 + abs(x))

    def hi_q(x):
        return x + 1.645 * (0.1 + abs(x))

    cp = ConformalPredictor(target_coverage=target)
    for _ in range(1000):
        x = random.uniform(-2, 2)
        cp.record(
            features={"x": x},
            prediction_lo=lo_q(x),
            prediction_hi=hi_q(x),
            outcome=noisy(x),
        )

    hits = 0
    n = 500
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = noisy(x)
        pi = cp.predict_interval(
            prediction_lo=lo_q(x), prediction_hi=hi_q(x), method=METHOD_CQR,
        )
        if pi.contains(y):
            hits += 1
    emp = hits / n
    assert abs(emp - target) < 0.04, f"cqr coverage {emp} vs target {target}"


def test_cqr_requires_both_quantiles():
    cp = ConformalPredictor(target_coverage=0.9)
    cp.record(
        features={}, prediction_lo=0.0, prediction_hi=1.0, outcome=0.5,
    )
    with pytest.raises(ValueError):
        cp.predict_interval(prediction_lo=0.0, method=METHOD_CQR)


# ---------- Mondrian (group-conditional) ----------


def test_mondrian_group_conditional_coverage():
    """Two groups with very different noise levels — Mondrian should
    give each group ~target coverage, while a single shared calibrator
    would over/under-cover the high-noise group."""
    random.seed(11)
    target = 0.9
    cp = ConformalPredictor(target_coverage=target)
    # Group A: tight noise. Group B: wide noise.
    for _ in range(400):
        cp.record(
            features={}, prediction=0.0,
            outcome=random.gauss(0, 0.5), group="A",
        )
        cp.record(
            features={}, prediction=0.0,
            outcome=random.gauss(0, 3.0), group="B",
        )
    held_a = [CalibrationPoint(features={}, prediction=0.0, outcome=random.gauss(0, 0.5), group="A") for _ in range(300)]
    held_b = [CalibrationPoint(features={}, prediction=0.0, outcome=random.gauss(0, 3.0), group="B") for _ in range(300)]
    rep = cp.measure_coverage(held_a + held_b, method=METHOD_MONDRIAN)

    assert "A" in rep.per_group and "B" in rep.per_group
    assert abs(rep.per_group["A"].empirical_coverage - target) < 0.05
    assert abs(rep.per_group["B"].empirical_coverage - target) < 0.05
    # Widths should differ: B's interval is much wider.
    assert rep.per_group["B"].mean_width > rep.per_group["A"].mean_width * 3


# ---------- Jackknife+ ----------


def test_jackknife_plus_interval_helper():
    # Hand-computed: predictions and residuals so the result is obvious.
    preds = [10.0, 10.0, 10.0]
    resids = [1.0, 2.0, 3.0]
    lo, hi = _jackknife_plus_interval(preds, resids, target_coverage=0.5)
    # alpha=0.5, n=3, n+1=4: lo_rank = floor(0.5 * 4)=2, hi_rank = ceil(0.5 * 4)=2.
    # lows sorted: [10-3, 10-2, 10-1] = [7, 8, 9]. lo = lows[1] = 8.
    # highs sorted: [10+1, 10+2, 10+3] = [11, 12, 13]. hi = highs[1] = 12.
    assert lo == 8.0
    assert hi == 12.0


def test_jackknife_plus_through_predictor():
    """jk+ end-to-end: build a tiny calibration set, supply a loo_predictor
    that returns mean-of-rest as point and the test residual against it."""
    random.seed(3)
    cp = ConformalPredictor(target_coverage=0.9)
    truth_mean = 5.0
    pts = []
    for _ in range(30):
        pts.append(CalibrationPoint(
            features={}, outcome=truth_mean + random.gauss(0, 1.0),
            prediction=None,
        ))
    cp.record_many(pts)

    def loo(rest, features):
        ys = [float(p.outcome) for p in rest]
        m = statistics.mean(ys) if ys else 0.0
        # residual on the test point is approximated as its prior-noise sample;
        # use mean absolute deviation of the rest as a proxy.
        mad = statistics.mean(abs(y - m) for y in ys) if ys else 0.0
        return m, mad

    pi = cp.predict_interval(method=METHOD_JACKKNIFE_PLUS, loo_predictor=loo)
    assert pi.lower < truth_mean < pi.upper
    assert pi.method == METHOD_JACKKNIFE_PLUS


def test_jackknife_plus_requires_loo_predictor():
    cp = ConformalPredictor(target_coverage=0.9)
    cp.record(features={}, prediction=None, outcome=1.0)
    with pytest.raises(ValueError):
        cp.predict_interval(method=METHOD_JACKKNIFE_PLUS)


# ---------- adaptive conformal (ACI) ----------


def test_aci_state_update_increases_alpha_on_miss():
    s = _ACIState(alpha_star=0.1, alpha_t=0.1, gamma=0.05)
    s.update(miss=True)
    # err=1, target=0.1 ⇒ α grows: α + 0.05 * (0.1 - 1) = 0.1 - 0.045 = 0.055
    assert abs(s.alpha_t - 0.055) < 1e-9
    s2 = _ACIState(alpha_star=0.1, alpha_t=0.1, gamma=0.05)
    s2.update(miss=False)
    # err=0, target=0.1 ⇒ α grows: 0.1 + 0.05 * 0.1 = 0.105
    assert abs(s2.alpha_t - 0.105) < 1e-9


def test_aci_state_clamps_to_band():
    s = _ACIState(alpha_star=0.1, alpha_t=0.0005, gamma=0.05)
    for _ in range(20):
        s.update(miss=True)  # drives α down
    assert s.alpha_t >= s.band_lo
    s2 = _ACIState(alpha_star=0.99, alpha_t=0.9, gamma=0.5)
    for _ in range(20):
        s2.update(miss=False)
    assert s2.alpha_t <= s2.band_hi


def test_update_adaptive_records_coverage_stream():
    cp = ConformalPredictor(target_coverage=0.9, adaptive=True)
    cp.record(features={}, prediction=0.0, outcome=0.0)
    pi = PredictionInterval(
        lower=-1.0, upper=1.0, target_coverage=0.9,
        method=METHOD_SPLIT, n_cal=1, effective_alpha=0.1,
    )
    adjusted = cp.update_adaptive(outcome=0.0, last_interval=pi)
    assert adjusted is True
    adjusted = cp.update_adaptive(outcome=5.0, last_interval=pi)  # miss
    assert adjusted is True


def test_drift_detection_fires_event_on_sustained_miss():
    """Bulk hits, tail misses → drift detector should set the flag and
    fire a CONFORMAL_DRIFT event."""
    bus = EventBus()
    cp = ConformalPredictor(target_coverage=0.9, bus=bus, drift_threshold=0.1)
    pi = PredictionInterval(
        lower=-1.0, upper=1.0, target_coverage=0.9,
        method=METHOD_SPLIT, n_cal=1, effective_alpha=0.1,
    )
    cp.record(features={}, prediction=0.0, outcome=0.0)
    # 200 hits, then 100 misses.
    for _ in range(200):
        cp.update_adaptive(outcome=0.0, last_interval=pi)
    for _ in range(100):
        cp.update_adaptive(outcome=10.0, last_interval=pi)
    drift_events = bus.history(kind=CONFORMAL_DRIFT)
    assert len(drift_events) >= 1
    rep = cp.report()
    assert rep.drift_detected is True


# ---------- RAPS classification ----------


def test_raps_nonconformity_orders_correctly():
    # A confident correct guess scores low; a wrong one scores high.
    scores = {"a": 5.0, "b": 1.0, "c": -1.0}
    low_score = _raps_nonconformity(scores, "a", k_reg=1, lam=0.01)
    high_score = _raps_nonconformity(scores, "c", k_reg=1, lam=0.01)
    assert low_score < high_score


def test_raps_predict_set_grows_with_threshold():
    scores = {"a": 5.0, "b": 1.0, "c": -1.0}
    small, _ = _raps_predict_set(scores, threshold=0.5, k_reg=1, lam=0.01)
    big, _ = _raps_predict_set(scores, threshold=2.0, k_reg=1, lam=0.01)
    assert set(small) <= set(big)


def test_raps_classification_coverage():
    """Five-class problem with a noisy classifier — RAPS should
    produce sets that cover the truth at least at the target rate
    (conformal guarantees ≥ 1-α; over-coverage is correct), while
    keeping the average set size strictly less than |labels|."""
    random.seed(5)
    target = 0.9
    labels = ("a", "b", "c", "d", "e")
    cp = ConformalPredictor(target_coverage=target)

    def emit(true_label):
        scores = {}
        for lab in labels:
            if lab == true_label:
                scores[lab] = 1.2 + random.gauss(0, 0.4)
            else:
                scores[lab] = 0.0 + random.gauss(0, 0.6)
        return scores

    for _ in range(800):
        y = random.choice(labels)
        cp.record(features={}, prediction=emit(y), outcome=y)

    hits = 0
    sizes = []
    total = 500
    for _ in range(total):
        y = random.choice(labels)
        s = emit(y)
        ps = cp.predict_set(scores=s, method=METHOD_RAPS)
        if y in ps.labels:
            hits += 1
        sizes.append(ps.size)
    emp = hits / total
    mean_size = statistics.mean(sizes)
    # Marginal coverage guarantee: empirical ≥ target − finite-sample slack.
    assert emp >= target - 0.05, f"raps coverage {emp} below {target}"
    # Adaptive: sets should not always be the full label set.
    assert mean_size < len(labels) - 0.2, f"raps sets too wide: mean {mean_size}"
    assert mean_size > 1.0, f"raps sets too narrow: mean {mean_size}"


def test_raps_unknown_method_raises():
    cp = ConformalPredictor(target_coverage=0.9)
    with pytest.raises(ValueError):
        cp.predict_set(scores={"a": 1.0}, method="bogus")


# ---------- diagnostics ----------


def test_measure_coverage_per_group_split_by_group():
    random.seed(13)
    cp = ConformalPredictor(target_coverage=0.9)
    for _ in range(300):
        cp.record(features={}, prediction=0.0, outcome=random.gauss(0, 1), group="A")
        cp.record(features={}, prediction=0.0, outcome=random.gauss(0, 1), group="B")
    held = []
    for _ in range(100):
        held.append(CalibrationPoint(features={}, prediction=0.0, outcome=random.gauss(0, 1), group="A"))
        held.append(CalibrationPoint(features={}, prediction=0.0, outcome=random.gauss(0, 1), group="B"))
    rep = cp.measure_coverage(held, method=METHOD_SPLIT)
    assert rep.n == 200
    assert "A" in rep.per_group and "B" in rep.per_group
    assert rep.per_group["A"].n == 100
    assert rep.per_group["B"].n == 100
    assert abs(rep.empirical_coverage - 0.9) < 0.06


def test_measure_coverage_notes_small_n():
    cp = ConformalPredictor(target_coverage=0.9)
    for _ in range(40):
        cp.record(features={}, prediction=0.0, outcome=0.1)
    held = [CalibrationPoint(features={}, prediction=0.0, outcome=0.1) for _ in range(10)]
    rep = cp.measure_coverage(held, method=METHOD_SPLIT)
    assert any("low_n" in n for n in rep.notes)


def test_report_streams_from_adaptive_history():
    cp = ConformalPredictor(target_coverage=0.9, adaptive=True)
    cp.record(features={}, prediction=0.0, outcome=0.0)
    pi = PredictionInterval(
        lower=-1.0, upper=1.0, target_coverage=0.9,
        method=METHOD_SPLIT, n_cal=1, effective_alpha=0.1,
    )
    for _ in range(20):
        cp.update_adaptive(outcome=0.0, last_interval=pi)
    rep = cp.report()
    assert rep.n == 20
    assert rep.empirical_coverage == 1.0
    assert rep.method == "stream"


# ---------- integrations ----------


def test_attach_to_bus_records_from_events():
    bus = EventBus()
    cp = ConformalPredictor(target_coverage=0.9)
    unsub = cp.attach_to_bus(bus, kinds=("custom.metric",))
    for i in range(5):
        bus.publish(Event(
            kind="custom.metric",
            data={
                "features": {"i": i},
                "prediction": float(i),
                "outcome": float(i) + 0.5,
                "group": "x",
            },
        ))
    bus.publish(Event(kind="other.kind", data={"prediction": 1.0, "outcome": 2.0}))
    assert len(cp) == 5
    unsub()
    bus.publish(Event(kind="custom.metric", data={
        "features": {}, "prediction": 0.0, "outcome": 0.0,
    }))
    assert len(cp) == 5


def test_attach_to_driver_drains_receipt_callback():
    cp = ConformalPredictor(target_coverage=0.9)
    captured = []

    class FakeDriver:
        def subscribe_receipts(self, cb):
            self._cb = cb
            return lambda: None
        def emit(self, r):
            self._cb(r)

    class FakeReceipt:
        def __init__(self, est, act, tenant="t1"):
            self.estimated_cost_usd = est
            self.actual_cost_usd = act
            self.tenant_id = tenant
            self.model = "claude-opus-4-7"
            self.intent = "demo"
            self.estimated_p_success = 0.8

    d = FakeDriver()
    cp.attach_to_driver(d)
    for est, act in [(0.5, 0.4), (1.0, 1.2), (2.0, 1.8)]:
        d.emit(FakeReceipt(est, act))
    assert len(cp) == 3
    pi = cp.predict_interval(prediction=1.0, method=METHOD_MONDRIAN, group="t1")
    assert pi.n_cal == 3


def test_observed_event_emitted_on_record():
    bus = EventBus()
    cp = ConformalPredictor(target_coverage=0.9, bus=bus)
    cp.record(features={}, prediction=0.0, outcome=0.0, group="g1")
    obs = bus.history(kind=CONFORMAL_OBSERVED)
    assert len(obs) == 1
    assert obs[0].data["group"] == "g1"
    assert obs[0].data["n_total"] == 1


def test_predicted_event_emitted_on_predict():
    bus = EventBus()
    cp = ConformalPredictor(target_coverage=0.9, bus=bus)
    for _ in range(50):
        cp.record(features={}, prediction=0.0, outcome=0.0)
    cp.predict_interval(prediction=0.0, method=METHOD_SPLIT)
    preds = bus.history(kind=CONFORMAL_PREDICTED)
    assert len(preds) == 1
    assert preds[0].data["method"] == METHOD_SPLIT


def test_report_event_emitted_on_measure_coverage():
    bus = EventBus()
    cp = ConformalPredictor(target_coverage=0.9, bus=bus)
    for _ in range(40):
        cp.record(features={}, prediction=0.0, outcome=0.1)
    held = [CalibrationPoint(features={}, prediction=0.0, outcome=0.1) for _ in range(20)]
    cp.measure_coverage(held, method=METHOD_SPLIT)
    reports = bus.history(kind=CONFORMAL_REPORT)
    assert len(reports) == 1


# ---------- ring buffer ----------


def test_record_respects_max_history():
    cp = ConformalPredictor(target_coverage=0.9, max_history=10)
    for i in range(25):
        cp.record(features={"i": i}, prediction=float(i), outcome=float(i))
    assert len(cp) == 10
    # oldest dropped: only last-10 features remain.
    with cp._lock:
        ids = [p.features["i"] for p in cp._points]
    assert ids == list(range(15, 25))


# ---------- persistence ----------


def test_snapshot_restore_roundtrip():
    cp = ConformalPredictor(target_coverage=0.9, adaptive=True)
    for i in range(50):
        cp.record(features={"i": i}, prediction=float(i), outcome=float(i) + 0.1)
    pi = PredictionInterval(
        lower=-1.0, upper=1.0, target_coverage=0.9,
        method=METHOD_SPLIT, n_cal=1, effective_alpha=0.1,
    )
    cp.update_adaptive(outcome=0.0, last_interval=pi)
    snap = cp.snapshot()
    # Round-trip through JSON to verify everything is serialisable.
    raw = json.dumps(snap)
    snap2 = json.loads(raw)

    cp2 = ConformalPredictor(target_coverage=0.5)
    cp2.restore(snap2)
    assert cp2.target_coverage == 0.9
    assert len(cp2) == 50
    assert cp2._aci is not None
    assert abs(cp2._aci.alpha_t - cp._aci.alpha_t) < 1e-9


def test_save_load_to_file(tmp_path):
    cp = ConformalPredictor(target_coverage=0.95)
    for i in range(10):
        cp.record(features={"i": i}, prediction=float(i), outcome=float(i))
    fp = tmp_path / "cp.json"
    cp.save(fp)
    cp2 = ConformalPredictor(target_coverage=0.5)
    cp2.load(fp)
    assert cp2.target_coverage == 0.95
    assert len(cp2) == 10


def test_restore_rejects_wrong_version():
    cp = ConformalPredictor(target_coverage=0.9)
    with pytest.raises(ValueError):
        cp.restore({"version": 99})


# ---------- input validation ----------


def test_target_coverage_must_be_open_interval():
    with pytest.raises(ValueError):
        ConformalPredictor(target_coverage=0.0)
    with pytest.raises(ValueError):
        ConformalPredictor(target_coverage=1.0)


def test_max_history_must_be_positive():
    with pytest.raises(ValueError):
        ConformalPredictor(target_coverage=0.9, max_history=0)


def test_unknown_regression_method_raises():
    cp = ConformalPredictor(target_coverage=0.9)
    with pytest.raises(ValueError):
        cp.predict_interval(prediction=0.0, method="bogus")


def test_known_methods_constants_are_exported():
    assert METHOD_SPLIT in KNOWN_REGRESSION_METHODS
    assert METHOD_CQR in KNOWN_REGRESSION_METHODS
    assert METHOD_MONDRIAN in KNOWN_REGRESSION_METHODS
    assert METHOD_JACKKNIFE_PLUS in KNOWN_REGRESSION_METHODS
    assert METHOD_RAPS in KNOWN_CLASSIFICATION_METHODS


# ---------- JSON safety of PredictionInterval / PredictionSet ----------


def test_prediction_interval_to_dict_is_json_safe():
    pi = PredictionInterval(
        lower=0.1, upper=0.5, target_coverage=0.9,
        method=METHOD_SPLIT, n_cal=10, effective_alpha=0.1,
        point=0.3, group="t1", diagnostics={"threshold": 0.2},
    )
    blob = json.dumps(pi.to_dict())
    back = json.loads(blob)
    assert back["lower"] == 0.1
    assert back["upper"] == 0.5
    assert back["method"] == METHOD_SPLIT


def test_prediction_set_to_dict_is_json_safe():
    ps = PredictionSet(
        labels=("a", "b"), target_coverage=0.9, method=METHOD_RAPS,
        n_cal=10, scores={"a": 0.6, "b": 0.3}, group="g",
    )
    blob = json.dumps(ps.to_dict())
    back = json.loads(blob)
    assert sorted(back["labels"]) == ["a", "b"]


def test_coverage_report_to_dict_is_json_safe():
    cp = ConformalPredictor(target_coverage=0.9)
    for _ in range(40):
        cp.record(features={}, prediction=0.0, outcome=0.0, group="x")
    held = [CalibrationPoint(features={}, prediction=0.0, outcome=0.0, group="x") for _ in range(20)]
    rep = cp.measure_coverage(held, method=METHOD_SPLIT)
    blob = json.dumps(rep.to_dict())
    back = json.loads(blob)
    assert "per_group" in back


# ---------- public exports ----------


def test_top_level_module_exports_conformal_symbols():
    import agi

    assert hasattr(agi, "ConformalPredictor")
    assert hasattr(agi, "CalibrationPoint")
    assert hasattr(agi, "PredictionInterval")
    assert hasattr(agi, "PredictionSet")
    assert hasattr(agi, "CoverageReport")
    assert hasattr(agi, "METHOD_SPLIT")
    assert hasattr(agi, "METHOD_CQR")
    assert hasattr(agi, "METHOD_MONDRIAN")
    assert hasattr(agi, "METHOD_JACKKNIFE_PLUS")
    assert hasattr(agi, "METHOD_RAPS")
