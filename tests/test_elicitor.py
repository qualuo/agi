"""Tests for ``agi.elicitor``."""

from __future__ import annotations

import pytest

from agi.elicitor import (
    EL_ALERTED,
    EL_BUDGET_UPDATED,
    EL_CERTIFIED,
    EL_OBSERVED,
    EL_REPORTED,
    EL_RESET,
    EL_STARTED,
    KNOWN_EVENTS,
    KNOWN_METHODS,
    KNOWN_RECOMMENDATIONS,
    KNOWN_VERDICTS,
    METHOD_BON_64,
    METHOD_COT,
    METHOD_FINETUNE_PROBE,
    METHOD_GREEDY,
    REC_DEPLOY,
    REC_EXPAND_ELICITATION,
    REC_INVESTIGATE_SANDBAG,
    REC_RETIRE,
    VERDICT_ELICIT_MORE,
    VERDICT_FRONTIER,
    VERDICT_SANDBAGGER,
    VERDICT_UNDERPERFORMING,
    ElicitationObservation,
    Elicitor,
    ElicitorCertificate,
    ElicitorConfig,
    ElicitorReport,
    InsufficientData,
    InvalidConfig,
    InvalidObservation,
    MethodReport,
    fresh_elicitor,
    synthetic_frontier_stream,
    synthetic_sandbag_stream,
    synthetic_under_stream,
)


class _MemoryBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, payload=None):
        if payload is None and hasattr(kind, "kind") and hasattr(kind, "payload"):
            self.events.append((kind.kind, dict(kind.payload or {})))
        else:
            self.events.append((str(kind), dict(payload or {})))


# -- Constants ---------------------------------------------------------


def test_constants_lengths():
    assert len(KNOWN_VERDICTS) == 4
    assert len(KNOWN_RECOMMENDATIONS) == 5
    assert len(KNOWN_METHODS) == 8
    assert len(KNOWN_EVENTS) == 7


def test_event_namespace():
    for k in KNOWN_EVENTS:
        assert k.startswith("elicitor.")


# -- Config validation ------------------------------------------------


def test_default_config_valid():
    cfg = ElicitorConfig()
    assert cfg.model_id == "default"
    assert cfg.target_capability == pytest.approx(0.70)


def test_invalid_config_bad_threshold():
    with pytest.raises(InvalidConfig):
        ElicitorConfig(target_capability=1.5)
    with pytest.raises(InvalidConfig):
        ElicitorConfig(delta=0.0)
    with pytest.raises(InvalidConfig):
        ElicitorConfig(min_observations=2)
    with pytest.raises(InvalidConfig):
        ElicitorConfig(min_methods=0)
    with pytest.raises(InvalidConfig):
        ElicitorConfig(rec_retire_threshold=-1.0)


# -- Observation validation ------------------------------------------


def test_observation_validation():
    with pytest.raises(InvalidObservation):
        ElicitationObservation(task_id="", method_id="x", score=0.5)
    with pytest.raises(InvalidObservation):
        ElicitationObservation(task_id="t", method_id="", score=0.5)
    with pytest.raises(InvalidObservation):
        ElicitationObservation(task_id="t", method_id="x", score=1.7)
    with pytest.raises(InvalidObservation):
        ElicitationObservation(task_id="t", method_id="x", score=0.5, cost_usd=-1.0)


# -- Insufficient data -----------------------------------------------


def test_certify_raises_before_min_observations():
    el = fresh_elicitor("m", "b")
    el.observe(ElicitationObservation(task_id="t", method_id="x", score=0.5))
    with pytest.raises(InsufficientData):
        el.certify()


# -- Golden paths ---------------------------------------------------


def test_frontier_stream_yields_deploy_under_tight_bounds():
    el = fresh_elicitor("strong", "aisi", target_capability=0.70)
    # Need enough samples (and lower-variance synthetic) for the
    # empirical-Bernstein LCB to clear 0.70.
    el.observe_many(synthetic_frontier_stream(800, seed=42, method_sigma=0.08))
    cert = el.certify()
    assert isinstance(cert, ElicitorCertificate)
    assert cert.verdict == VERDICT_FRONTIER
    assert cert.recommendation == REC_DEPLOY
    assert cert.elicited_method_id in KNOWN_METHODS
    assert cert.capability_lcb >= 0.70


def test_sandbag_stream_yields_sandbagger():
    el = fresh_elicitor("sand", "aisi", target_capability=0.70)
    el.observe_many(synthetic_sandbag_stream(192, seed=42))
    cert = el.certify()
    assert cert.verdict == VERDICT_SANDBAGGER
    assert cert.recommendation == REC_INVESTIGATE_SANDBAG
    assert cert.sandbag_score > cert.sandbag_threshold


def test_underperforming_stream_yields_retire():
    el = fresh_elicitor("weak", "aisi", target_capability=0.70)
    el.observe_many(synthetic_under_stream(192, seed=42))
    cert = el.certify()
    assert cert.verdict == VERDICT_UNDERPERFORMING
    assert cert.recommendation == REC_RETIRE
    assert cert.capability_ucb < 0.70


def test_under_elicited_yields_expand_elicitation():
    """Only one method, modest sample size — not enough for sign-off."""
    el = fresh_elicitor("partial", "aisi", target_capability=0.70, min_methods=2)
    for i in range(40):
        el.observe(
            ElicitationObservation(
                task_id=f"t{i % 8}",
                method_id=METHOD_GREEDY,
                score=0.75 + 0.05 * ((i % 7) / 7.0 - 0.5),
            )
        )
    cert = el.certify()
    assert cert.verdict == VERDICT_ELICIT_MORE
    assert cert.recommendation == REC_EXPAND_ELICITATION
    assert cert.n_methods == 1


# -- Replay-verifiability -------------------------------------------


def test_replay_verifiability():
    stream = list(synthetic_frontier_stream(96, seed=11))
    a = fresh_elicitor("r1", "b")
    a.observe_many(stream)
    cert_a = a.certify()
    b = fresh_elicitor("r1", "b")
    b.observe_many(stream)
    cert_b = b.certify()
    assert cert_a.fingerprint == cert_b.fingerprint
    assert a.fingerprint == b.fingerprint


def test_fingerprint_advances_each_observation():
    el = fresh_elicitor("chain", "b")
    fps = [el.fingerprint]
    for obs in synthetic_frontier_stream(8, seed=2):
        el.observe(obs)
        fps.append(el.fingerprint)
    assert len(set(fps)) == len(fps)


# -- Event emission --------------------------------------------------


def test_event_lifecycle():
    bus = _MemoryBus()
    el = fresh_elicitor("evt", "b", bus=bus, min_observations=8)
    el.observe_many(synthetic_frontier_stream(32, seed=3))
    el.certify()
    el.report()
    kinds = [ev for ev, _ in bus.events]
    assert EL_STARTED in kinds
    assert EL_OBSERVED in kinds
    assert EL_CERTIFIED in kinds
    assert EL_REPORTED in kinds


def test_alert_on_sandbagger():
    bus = _MemoryBus()
    el = fresh_elicitor("bad", "b", bus=bus, min_observations=8)
    el.observe_many(synthetic_sandbag_stream(96, seed=4))
    el.certify()
    kinds = [ev for ev, _ in bus.events]
    assert EL_ALERTED in kinds


def test_buggy_bus_does_not_break_observe():
    class BoomBus:
        def emit(self, *_, **__):
            raise RuntimeError("bus down")

    el = fresh_elicitor("ok", "b", bus=BoomBus())
    el.observe(ElicitationObservation(task_id="t", method_id="x", score=0.5))


# -- Budget update --------------------------------------------------


def test_update_budget_keeps_state():
    bus = _MemoryBus()
    el = fresh_elicitor("bd", "b", bus=bus)
    el.observe_many(synthetic_frontier_stream(32, seed=6))
    n_before = el.n_observations
    cfg = el.update_budget(target_capability=0.85, delta=0.10)
    assert cfg.target_capability == pytest.approx(0.85)
    assert cfg.delta == pytest.approx(0.10)
    assert el.n_observations == n_before
    kinds = [ev for ev, _ in bus.events]
    assert EL_BUDGET_UPDATED in kinds


# -- Reset ----------------------------------------------------------


def test_reset_clears():
    el = fresh_elicitor("rs", "b")
    el.observe_many(synthetic_frontier_stream(32, seed=7))
    assert el.n_observations == 32
    el.reset()
    assert el.n_observations == 0
    assert el.n_methods == 0
    with pytest.raises(InsufficientData):
        el.certify()


# -- Per-method reports --------------------------------------------


def test_report_contains_per_method_breakdown():
    el = fresh_elicitor("rep", "b", min_observations=24)
    el.observe_many(synthetic_frontier_stream(96, seed=8))
    el.certify()
    rep = el.report()
    assert isinstance(rep, ElicitorReport)
    assert rep.n_observations == 96
    assert len(rep.per_method) >= 2
    # Means descending.
    means = [r.mean for r in rep.per_method]
    assert means == sorted(means, reverse=True)
    for r in rep.per_method:
        assert isinstance(r, MethodReport)
        assert 0.0 <= r.ci_low <= r.mean <= r.ci_high <= 1.0


# -- Certificate serialisability ----------------------------------


def test_certificate_json_serialisable():
    import json
    from dataclasses import asdict

    el = fresh_elicitor("j", "b", min_observations=24)
    el.observe_many(synthetic_frontier_stream(96, seed=9))
    cert = el.certify()
    payload = asdict(cert)
    s = json.dumps(payload, default=lambda o: o.__dict__)
    assert "fingerprint" in s


# -- Concurrency smoke ---------------------------------------------


def test_concurrent_observe_and_certify():
    import threading

    el = fresh_elicitor("c", "b", min_observations=8)
    stop = threading.Event()
    errors: list[BaseException] = []

    def producer():
        try:
            for obs in synthetic_frontier_stream(200, seed=42):
                if stop.is_set():
                    break
                el.observe(obs)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def consumer():
        try:
            for _ in range(50):
                if el.n_observations >= 16:
                    el.certify()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    p = threading.Thread(target=producer)
    c = threading.Thread(target=consumer)
    p.start()
    c.start()
    p.join(timeout=10)
    stop.set()
    c.join(timeout=10)
    assert not errors


# -- Bonferroni: more methods → wider per-method CIs ---------------


def test_bonferroni_correction_active():
    # Same number of total observations.  One run uses one method, one
    # uses many; the multi-method run should have stricter per-method
    # delta and therefore wider CIs.
    el_one = fresh_elicitor("one", "b", min_observations=32)
    for i in range(96):
        el_one.observe(
            ElicitationObservation(task_id=f"t{i % 8}", method_id="m1", score=0.8)
        )
    rep_one = el_one.report()

    el_many = fresh_elicitor("many", "b", min_observations=32)
    methods = ["m1", "m2", "m3", "m4"]
    for i in range(96):
        el_many.observe(
            ElicitationObservation(
                task_id=f"t{i % 8}", method_id=methods[i % 4], score=0.8
            )
        )
    rep_many = el_many.report()

    # Pick the first method in each.  Same mean, same n? Not quite — the
    # many-method run has n=24 vs one-method n=96.  Compare CIs on
    # *equal-n* slices: rerun the one-method version with 24 samples.
    el_eq = fresh_elicitor("eq", "b", min_observations=8)
    for i in range(24):
        el_eq.observe(
            ElicitationObservation(task_id=f"t{i % 8}", method_id="m1", score=0.8)
        )
    rep_eq = el_eq.report()
    # rep_many's first method has the same n=24 as rep_eq; its CI is
    # wider because of the Bonferroni split.
    width_eq = rep_eq.per_method[0].ci_high - rep_eq.per_method[0].ci_low
    width_many = rep_many.per_method[0].ci_high - rep_many.per_method[0].ci_low
    assert width_many > width_eq  # Bonferroni-widened
