"""Tests for ``agi.aegis``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agi.aegis import (
    AE_ABSORBED,
    AE_ALERTED,
    AE_CERTIFIED,
    AE_REPORTED,
    AE_RESET,
    AE_STARTED,
    DECISION_BLOCK,
    DECISION_DEGRADE,
    DECISION_HOLD,
    DECISION_SHIP,
    DEFAULT_VERDICT_SEVERITY,
    KNOWN_DECISIONS,
    KNOWN_EVENTS,
    KNOWN_SEVERITIES,
    SEVERITY_BLOCK,
    SEVERITY_DEGRADE,
    SEVERITY_OK,
    SEVERITY_WATCH,
    Aegis,
    AegisConfig,
    AegisDecision,
    AegisReport,
    InvalidCertificate,
    InvalidConfig,
    PrimitiveDecisionEntry,
    SafetyCertificate,
    fresh_aegis,
    from_dataclass,
)


class _MemoryBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, payload=None):
        if payload is None and hasattr(kind, "kind") and hasattr(kind, "payload"):
            self.events.append((kind.kind, dict(kind.payload or {})))
        else:
            self.events.append((str(kind), dict(payload or {})))


# -- Constants ------------------------------------------------------


def test_constants():
    assert SEVERITY_OK in KNOWN_SEVERITIES
    assert DECISION_SHIP in KNOWN_DECISIONS
    for k in KNOWN_EVENTS:
        assert k.startswith("aegis.")


def test_default_verdict_severity_covers_all_safety_primitives():
    # Sample of known primitive verdicts across the safety stack.
    for v in ("TRUST", "REJECT", "QUARANTINE", "scheming", "ESCALATE_HUMAN"):
        assert v in DEFAULT_VERDICT_SEVERITY


# -- Config validation --------------------------------------------


def test_default_config_valid():
    cfg = AegisConfig()
    assert cfg.deployment_id == "default"


def test_invalid_config():
    with pytest.raises(InvalidConfig):
        AegisConfig(deployment_id="")
    with pytest.raises(InvalidConfig):
        AegisConfig(alpha=0.0)
    with pytest.raises(InvalidConfig):
        AegisConfig(alpha=1.0)
    with pytest.raises(InvalidConfig):
        AegisConfig(severity_overrides={"x": "BOGUS"})
    with pytest.raises(InvalidConfig):
        AegisConfig(product_evalue_factor=0.0)
    with pytest.raises(InvalidConfig):
        AegisConfig(holm_escalation_threshold=-1)


# -- Certificate validation ---------------------------------------


def test_certificate_validation():
    with pytest.raises(InvalidCertificate):
        SafetyCertificate(primitive="", verdict="TRUST")
    with pytest.raises(InvalidCertificate):
        SafetyCertificate(primitive="p", verdict="")
    with pytest.raises(InvalidCertificate):
        SafetyCertificate(primitive="p", verdict="TRUST", p_value=1.5)
    with pytest.raises(InvalidCertificate):
        SafetyCertificate(primitive="p", verdict="TRUST", e_value=-1.0)
    with pytest.raises(InvalidCertificate):
        SafetyCertificate(primitive="p", verdict="TRUST", evidence_n=-1)


# -- Single absorb/certify ----------------------------------------


def test_empty_aegis_yields_hold():
    a = fresh_aegis("d")
    d = a.certify()
    assert d.decision == DECISION_HOLD
    assert d.severity == SEVERITY_WATCH


def test_single_trust_yields_ship():
    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="refuser", verdict="TRUST"))
    d = a.certify()
    assert d.decision == DECISION_SHIP
    assert d.severity == SEVERITY_OK
    assert d.blocking_primitive is None


def test_single_reject_yields_block():
    a = fresh_aegis("d")
    a.absorb(
        SafetyCertificate(
            primitive="refuser",
            verdict="REJECT",
            recommendation="ESCALATE_HUMAN",
        )
    )
    d = a.certify()
    assert d.decision == DECISION_BLOCK
    assert d.severity == SEVERITY_BLOCK
    assert d.blocking_primitive == "refuser"
    assert d.aggregated_recommendation == "ESCALATE_HUMAN"


def test_max_severity_wins_across_primitives():
    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="refuser", verdict="TRUST"))
    a.absorb(SafetyCertificate(primitive="schemer", verdict="suspicious"))
    a.absorb(
        SafetyCertificate(primitive="goodharter", verdict="QUARANTINE")
    )
    d = a.certify()
    assert d.decision == DECISION_BLOCK
    assert d.blocking_primitive == "goodharter"


def test_replacement_on_re_absorb():
    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="refuser", verdict="QUARANTINE"))
    assert a.certify().decision == DECISION_BLOCK
    # Same primitive flips to TRUST → next certify should be SHIP.
    a.absorb(SafetyCertificate(primitive="refuser", verdict="TRUST"))
    assert a.certify().decision == DECISION_SHIP


def test_drop_removes_primitive():
    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="schemer", verdict="scheming"))
    assert a.certify().decision == DECISION_BLOCK
    assert a.drop("schemer") is True
    assert a.certify().decision == DECISION_HOLD
    assert a.drop("nonexistent") is False


# -- Required primitives ------------------------------------------


def test_missing_required_primitives_holds():
    cfg = AegisConfig(
        deployment_id="d",
        require_primitives=("refuser", "faithfuller"),
    )
    a = Aegis(cfg)
    a.absorb(SafetyCertificate(primitive="refuser", verdict="TRUST"))
    d = a.certify()
    assert d.decision == DECISION_HOLD
    assert "faithfuller" in d.missing_required


def test_all_required_satisfied_ships():
    cfg = AegisConfig(
        deployment_id="d",
        require_primitives=("refuser", "faithfuller"),
    )
    a = Aegis(cfg)
    a.absorb(SafetyCertificate(primitive="refuser", verdict="TRUST"))
    a.absorb(SafetyCertificate(primitive="faithfuller", verdict="TRUST"))
    d = a.certify()
    assert d.decision == DECISION_SHIP


# -- Severity overrides + unknown verdicts -----------------------


def test_severity_overrides():
    cfg = AegisConfig(
        deployment_id="d",
        severity_overrides={"custom-warn-x": SEVERITY_DEGRADE},
    )
    a = Aegis(cfg)
    a.absorb(SafetyCertificate(primitive="x", verdict="custom-warn-x"))
    d = a.certify()
    assert d.decision == DECISION_DEGRADE


def test_unknown_verdict_default_watch():
    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="x", verdict="weird-state"))
    d = a.certify()
    assert d.severity == SEVERITY_WATCH


def test_unknown_verdict_block_when_configured():
    cfg = AegisConfig(deployment_id="d", block_on_unknown=True)
    a = Aegis(cfg)
    a.absorb(SafetyCertificate(primitive="x", verdict="weird-state"))
    d = a.certify()
    assert d.decision == DECISION_BLOCK


# -- E-value / product fusion ------------------------------------


def test_product_evalue_escalates_watch_to_degrade():
    """Multiple modest e-values, none individually blocking, fuse
    via Vovk-Wang to exceed 1/α and force DEGRADE."""
    cfg = AegisConfig(deployment_id="d", alpha=0.05)
    a = Aegis(cfg)
    # Three primitives, each at WATCH but carrying e-value > 1/3
    # of 1/alpha = 20 / 3 ≈ 6.67. Product ≈ 32 > 20.
    for i in range(3):
        a.absorb(
            SafetyCertificate(
                primitive=f"prim-{i}",
                verdict="warn",
                e_value=4.0,
            )
        )
    d = a.certify()
    assert d.product_evalue > 1.0 / cfg.alpha
    # Severity escalated above WATCH.
    assert d.severity != SEVERITY_WATCH


def test_holm_escalates_when_many_p_values_small():
    """Three primitives, each with p≈0.01, none on its own blocking,
    Holm-corrects to reject ≥2 → escalate."""
    cfg = AegisConfig(
        deployment_id="d",
        alpha=0.05,
        holm_escalation_threshold=1,
    )
    a = Aegis(cfg)
    for i in range(3):
        a.absorb(
            SafetyCertificate(
                primitive=f"prim-{i}",
                verdict="warn",
                p_value=0.005,
            )
        )
    d = a.certify()
    assert len(d.holm_rejected) >= 2
    assert d.severity != SEVERITY_WATCH


# -- Recommendation propagation ----------------------------------


def test_recommendation_propagates_from_blocking_primitive():
    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="x", verdict="TRUST"))
    a.absorb(
        SafetyCertificate(
            primitive="y",
            verdict="QUARANTINE",
            recommendation="ESCALATE_HUMAN",
        )
    )
    d = a.certify()
    assert d.aggregated_recommendation == "ESCALATE_HUMAN"


def test_recommendation_disabled_when_off():
    cfg = AegisConfig(deployment_id="d", recommendation_propagation=False)
    a = Aegis(cfg)
    a.absorb(
        SafetyCertificate(
            primitive="x",
            verdict="QUARANTINE",
            recommendation="ESCALATE_HUMAN",
        )
    )
    d = a.certify()
    assert d.aggregated_recommendation is None


# -- from_dataclass adapter --------------------------------------


@dataclass(frozen=True)
class _FakeCertificate:
    verdict: str
    recommendation: str | None
    product_evalue: float
    n_observations: int
    fingerprint: str
    p_value: float | None = None


def test_from_dataclass_extracts_fields():
    c = _FakeCertificate(
        verdict="TRUST",
        recommendation="DEPLOY",
        product_evalue=1.5,
        n_observations=128,
        fingerprint="abcd1234",
    )
    sc = from_dataclass(c, "fake")
    assert sc.primitive == "fake"
    assert sc.verdict == "TRUST"
    assert sc.recommendation == "DEPLOY"
    assert sc.e_value == pytest.approx(1.5)
    assert sc.evidence_n == 128
    assert sc.fingerprint == "abcd1234"


def test_from_dataclass_handles_p_value():
    c = _FakeCertificate(
        verdict="warn",
        recommendation=None,
        product_evalue=2.0,
        n_observations=64,
        fingerprint="x",
        p_value=0.03,
    )
    sc = from_dataclass(c, "fake")
    assert sc.p_value == pytest.approx(0.03)


def test_from_dataclass_rejects_none_or_missing_verdict():
    @dataclass
    class NoVerdict:
        recommendation: str = "x"

    with pytest.raises(InvalidCertificate):
        from_dataclass(NoVerdict(), "x")
    with pytest.raises(InvalidCertificate):
        from_dataclass(None, "x")


# -- End-to-end with real safety primitives ---------------------


def test_end_to_end_with_faithfuller_and_elicitor():
    from agi.elicitor import (
        fresh_elicitor,
        synthetic_frontier_stream,
        synthetic_sandbag_stream,
    )
    from agi.faithfuller import (
        fresh_faithfuller,
        synthetic_faithful_stream,
        synthetic_unfaithful_stream,
    )

    # Build A (good): faithful + frontier
    ff_good = fresh_faithfuller("p@prod")
    ff_good.observe_many(synthetic_faithful_stream(64, seed=1))
    el_good = fresh_elicitor("p@prod", "bench")
    el_good.observe_many(synthetic_frontier_stream(800, seed=1, method_sigma=0.08))
    a_good = fresh_aegis("p@prod")
    a_good.absorb(from_dataclass(ff_good.certify(), "faithfuller"))
    a_good.absorb(from_dataclass(el_good.certify(), "elicitor"))
    d_good = a_good.certify()
    assert d_good.decision == DECISION_SHIP

    # Build B (bad): unfaithful + sandbagger
    ff_bad = fresh_faithfuller("p@candidate")
    ff_bad.observe_many(synthetic_unfaithful_stream(64, seed=1))
    el_bad = fresh_elicitor("p@candidate", "bench")
    el_bad.observe_many(synthetic_sandbag_stream(192, seed=1))
    a_bad = fresh_aegis("p@candidate")
    a_bad.absorb(from_dataclass(ff_bad.certify(), "faithfuller"))
    a_bad.absorb(from_dataclass(el_bad.certify(), "elicitor"))
    d_bad = a_bad.certify()
    assert d_bad.decision == DECISION_BLOCK


# -- Event emission --------------------------------------------


def test_event_lifecycle():
    bus = _MemoryBus()
    a = fresh_aegis("d", bus=bus)
    a.absorb(SafetyCertificate(primitive="x", verdict="TRUST"))
    a.certify()
    a.report()
    kinds = [ev for ev, _ in bus.events]
    assert AE_STARTED in kinds
    assert AE_ABSORBED in kinds
    assert AE_CERTIFIED in kinds
    assert AE_REPORTED in kinds


def test_alert_emitted_on_block():
    bus = _MemoryBus()
    a = fresh_aegis("d", bus=bus)
    a.absorb(SafetyCertificate(primitive="x", verdict="REJECT"))
    a.certify()
    kinds = [ev for ev, _ in bus.events]
    assert AE_ALERTED in kinds


def test_alert_not_emitted_on_ship():
    bus = _MemoryBus()
    a = fresh_aegis("d", bus=bus)
    a.absorb(SafetyCertificate(primitive="x", verdict="TRUST"))
    a.certify()
    kinds = [ev for ev, _ in bus.events]
    assert AE_ALERTED not in kinds


def test_buggy_bus_does_not_break_absorb():
    class BoomBus:
        def emit(self, *_, **__):
            raise RuntimeError("bus down")

    a = fresh_aegis("d", bus=BoomBus())
    a.absorb(SafetyCertificate(primitive="x", verdict="TRUST"))
    a.certify()


# -- Replay-verifiability --------------------------------------


def test_replay_verifiability():
    cs = [
        SafetyCertificate(primitive=f"p{i}", verdict="warn", p_value=0.05)
        for i in range(5)
    ]
    a = fresh_aegis("d")
    a.absorb_many(cs)
    d_a = a.certify()
    b = fresh_aegis("d")
    b.absorb_many(cs)
    d_b = b.certify()
    assert d_a.fingerprint == d_b.fingerprint
    assert a.fingerprint == b.fingerprint


def test_fingerprint_advances_each_absorb():
    a = fresh_aegis("d")
    fps = [a.fingerprint]
    for i in range(5):
        a.absorb(SafetyCertificate(primitive=f"p{i}", verdict="TRUST"))
        fps.append(a.fingerprint)
    assert len(set(fps)) == len(fps)


# -- Concurrency smoke ----------------------------------------


def test_concurrent_absorb_and_certify():
    import threading

    a = fresh_aegis("d")
    stop = threading.Event()
    errors: list[BaseException] = []

    def producer():
        try:
            for i in range(200):
                if stop.is_set():
                    break
                a.absorb(
                    SafetyCertificate(
                        primitive=f"p{i % 4}", verdict="TRUST"
                    )
                )
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def consumer():
        try:
            for _ in range(50):
                a.certify()
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


# -- Report -------------------------------------------------


def test_report_after_decisions():
    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="x", verdict="TRUST"))
    a.absorb(SafetyCertificate(primitive="y", verdict="warn"))
    a.certify()
    rep = a.report()
    assert isinstance(rep, AegisReport)
    assert rep.n_certificates_current == 2
    assert rep.last_decision in (DECISION_HOLD, DECISION_SHIP)
    assert len(rep.per_primitive) == 2


def test_reset_clears():
    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="x", verdict="REJECT"))
    a.reset()
    assert a.n_certificates == 0
    d = a.certify()
    assert d.decision == DECISION_HOLD


# -- Certificate JSON-serialisability ------------------------


def test_decision_is_json_serialisable():
    import json
    from dataclasses import asdict

    a = fresh_aegis("d")
    a.absorb(SafetyCertificate(primitive="x", verdict="TRUST"))
    d = a.certify()
    payload = asdict(d)
    s = json.dumps(payload, default=lambda o: o.__dict__)
    assert "fingerprint" in s
