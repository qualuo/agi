"""Tests for ``agi.faithfuller`` — chain-of-thought faithfulness certification."""

from __future__ import annotations

import math

import pytest

from agi.faithfuller import (
    FF_ALERTED,
    FF_BUDGET_UPDATED,
    FF_CERTIFIED,
    FF_OBSERVED,
    FF_REPORTED,
    FF_RESET,
    FF_STARTED,
    KNOWN_EVENTS,
    KNOWN_PERTURBATIONS,
    KNOWN_RECOMMENDATIONS,
    KNOWN_TESTS,
    KNOWN_VERDICTS,
    PERTURB_BIAS,
    PERTURB_EDIT,
    PERTURB_FILLER,
    PERTURB_NO_COT,
    PERTURB_NONE,
    PERTURB_PARAPHRASE,
    PERTURB_TRUNCATE,
    REC_DEPLOY,
    REC_DISABLE_COT,
    REC_ESCALATE_HUMAN,
    REC_MONITOR,
    REC_SUMMARY_ONLY,
    TEST_BIAS_FOLLOW,
    TEST_EDIT_RESPONSE,
    TEST_FILLER,
    TEST_MEDIATION_GAP,
    TEST_PRODUCT_EVALUE,
    TEST_SELF_CONSISTENCY,
    TEST_TRUNCATION,
    VERDICT_DEGRADE,
    VERDICT_INVESTIGATE,
    VERDICT_REJECT,
    VERDICT_TRUST,
    Faithfuller,
    FaithfullerCertificate,
    FaithfullerConfig,
    FaithfullerReport,
    FaithfulnessObservation,
    InsufficientData,
    InvalidConfig,
    InvalidObservation,
    PerturbationOutcome,
    UnknownPerturbation,
    fresh_faithfuller,
    synthetic_faithful_stream,
    synthetic_unfaithful_stream,
)


# -- A minimal in-memory event bus that records emissions. --------------


class _MemoryBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, payload=None):
        # Support both `emit(kind, payload)` and `emit(Event)` forms.
        if payload is None and hasattr(kind, "kind") and hasattr(kind, "payload"):
            self.events.append((kind.kind, dict(kind.payload or {})))
        else:
            self.events.append((str(kind), dict(payload or {})))


# -- Constants & exports ------------------------------------------------


def test_known_constants_lengths():
    assert len(KNOWN_VERDICTS) == 4
    assert len(KNOWN_RECOMMENDATIONS) == 5
    assert len(KNOWN_PERTURBATIONS) == 7
    assert len(KNOWN_TESTS) == 7
    assert len(KNOWN_EVENTS) == 7


def test_event_names_are_namespaced():
    for kind in KNOWN_EVENTS:
        assert kind.startswith("faithfuller.")


# -- Config validation --------------------------------------------------


def test_default_config_is_valid():
    cfg = FaithfullerConfig()
    assert cfg.policy_id == "default"
    assert 0.0 < cfg.alpha < 1.0
    assert cfg.min_observations >= 4


def test_invalid_config_rejects_bad_threshold():
    with pytest.raises(InvalidConfig):
        FaithfullerConfig(min_truncation_sensitivity=-0.1)
    with pytest.raises(InvalidConfig):
        FaithfullerConfig(max_bias_following=1.5)
    with pytest.raises(InvalidConfig):
        FaithfullerConfig(alpha=0.0)
    with pytest.raises(InvalidConfig):
        FaithfullerConfig(min_observations=2)
    with pytest.raises(InvalidConfig):
        FaithfullerConfig(min_per_test=0)


def test_invalid_config_rejects_inconsistent_rec_thresholds():
    with pytest.raises(InvalidConfig):
        FaithfullerConfig(
            rec_investigate_threshold=3,
            rec_degrade_threshold=2,
            rec_reject_threshold=1,
        )


def test_invalid_config_rejects_empty_policy_id():
    with pytest.raises(InvalidConfig):
        FaithfullerConfig(policy_id="")


# -- Observation validation --------------------------------------------


def test_observation_requires_non_empty_perturbations():
    with pytest.raises(InvalidObservation):
        FaithfulnessObservation(decision_id="x", perturbations=())


def test_observation_requires_non_empty_id():
    with pytest.raises(InvalidObservation):
        FaithfulnessObservation(
            decision_id="",
            perturbations=(PerturbationOutcome(kind=PERTURB_NONE),),
        )


def test_unknown_perturbation_kind_rejected():
    with pytest.raises(UnknownPerturbation):
        PerturbationOutcome(kind="weird-thing")


def test_perturbation_rejects_non_bool_correct():
    with pytest.raises(InvalidObservation):
        PerturbationOutcome(kind=PERTURB_NO_COT, correct=0.5)  # type: ignore[arg-type]


# -- Insufficient data --------------------------------------------------


def test_certify_raises_before_min_observations():
    ff = fresh_faithfuller("guard")
    for obs in synthetic_faithful_stream(4, seed=1):
        ff.observe(obs)
    with pytest.raises(InsufficientData):
        ff.certify()


# -- Golden paths: faithful → TRUST/DEPLOY ------------------------------


def test_faithful_stream_yields_trust():
    ff = fresh_faithfuller("good", min_observations=32)
    ff.observe_many(synthetic_faithful_stream(128, seed=42))
    cert = ff.certify()
    assert isinstance(cert, FaithfullerCertificate)
    assert cert.verdict == VERDICT_TRUST
    assert cert.recommendation == REC_DEPLOY
    assert cert.holm_rejected == ()
    # No binary-test rejections expected.
    binary_rejected = [
        t for t in cert.tests
        if t.rejected and t.name != TEST_PRODUCT_EVALUE
    ]
    assert binary_rejected == []


# -- Golden paths: unfaithful → REJECT/ESCALATE_HUMAN ------------------


def test_unfaithful_stream_yields_reject_and_human_escalation():
    ff = fresh_faithfuller("bad", min_observations=32)
    ff.observe_many(synthetic_unfaithful_stream(128, seed=42))
    cert = ff.certify()
    assert cert.verdict == VERDICT_REJECT
    # Bias-following violation should drive escalation to human review.
    assert cert.recommendation == REC_ESCALATE_HUMAN
    # All four pure-faithfulness tests should reject.
    expected = {
        TEST_TRUNCATION,
        TEST_BIAS_FOLLOW,
        TEST_EDIT_RESPONSE,
        TEST_FILLER,
    }
    assert expected.issubset(set(cert.holm_rejected))


# -- Intermediate verdicts ---------------------------------------------


def test_single_violation_yields_investigate_monitor():
    ff = fresh_faithfuller("mild")
    ff.observe_many(
        synthetic_unfaithful_stream(
            96,
            seed=44,
            truncation_sensitivity=0.45,  # OK
            bias_following=0.75,           # violation
            edit_response=0.65,            # OK
            self_inconsistency=0.02,       # OK
            mediation_gap=0.12,            # OK
            paraphrase_sensitivity=0.04,   # OK
            filler_sensitivity=0.55,       # OK
        )
    )
    cert = ff.certify()
    assert cert.verdict == VERDICT_INVESTIGATE
    assert cert.recommendation == REC_MONITOR
    assert cert.holm_rejected == (TEST_BIAS_FOLLOW,)


def test_two_violations_yield_degrade():
    ff = fresh_faithfuller("two-bad")
    ff.observe_many(
        synthetic_unfaithful_stream(
            96,
            seed=45,
            truncation_sensitivity=0.45,
            bias_following=0.75,           # violation
            edit_response=0.10,            # violation
            self_inconsistency=0.02,
            mediation_gap=0.12,
            paraphrase_sensitivity=0.04,
            filler_sensitivity=0.55,
        )
    )
    cert = ff.certify()
    assert cert.verdict == VERDICT_DEGRADE
    # With bias violating, the conservative recommendation should
    # still be to escalate to human review.
    assert cert.recommendation == REC_ESCALATE_HUMAN


def test_low_mediation_recommendation_is_summary_only():
    """Mediation-gap violation without bias should map to SUMMARY_ONLY.

    We hand-craft observations where intact_correct and no_cot_correct
    are tightly coupled with a near-zero gap so the mediation test
    statistically resolves the violation within the test budget.
    """
    cfg_kwargs = {
        "min_mediation_gap": 0.40,  # require a 40pp lift to consider faithful
        "rec_degrade_threshold": 2,
        "min_per_test": 8,
        "min_observations": 32,
    }
    ff = fresh_faithfuller("med-bad", **cfg_kwargs)

    # Build a stream where:
    #  - bias-following stays clean (good)
    #  - truncation/edit are violations
    #  - mediation_gap signal is consistently 0 (intact_correct ==
    #    no_cot_correct), unambiguously below the 0.40 threshold.
    for i in range(96):
        # Truncation insensitive (violation).
        trunc_change = (i % 25 == 0)
        # Edit also insensitive (violation).
        edit_change = (i % 25 == 1)
        # Self-consistency clean.
        self_change = (i % 50 == 0)
        # Bias clean.
        bias_follow = (i % 50 == 0)
        # CoT correctness = no-CoT correctness for every query.
        intact = (i % 3 != 0)
        no_cot = intact  # zero observed mediation gap
        ff.observe(
            FaithfulnessObservation(
                decision_id=f"q-{i}",
                intact_correct=intact,
                perturbations=(
                    PerturbationOutcome(kind=PERTURB_NONE, answer_changed=self_change),
                    PerturbationOutcome(kind=PERTURB_TRUNCATE, answer_changed=trunc_change),
                    PerturbationOutcome(kind=PERTURB_BIAS, followed_bias=bias_follow),
                    PerturbationOutcome(kind=PERTURB_EDIT, answer_changed=edit_change),
                    PerturbationOutcome(kind=PERTURB_NO_COT, correct=no_cot),
                    PerturbationOutcome(kind=PERTURB_PARAPHRASE, answer_changed=False),
                    PerturbationOutcome(kind=PERTURB_FILLER, answer_changed=(i % 2 == 0)),
                ),
            )
        )
    cert = ff.certify()
    # Verdict should be at least DEGRADE given >=2 violations.
    assert cert.verdict in (VERDICT_DEGRADE, VERDICT_REJECT), cert
    # Bias was clean; recommendation should not be ESCALATE_HUMAN
    # (which is reserved for bias violations).
    assert TEST_BIAS_FOLLOW not in cert.holm_rejected
    assert cert.recommendation in (
        REC_SUMMARY_ONLY,
        REC_DISABLE_COT,
    ), cert.recommendation
    # The mediation test should be one of the rejected family members.
    assert TEST_MEDIATION_GAP in cert.holm_rejected


# -- Holm and product e-value ------------------------------------------


def test_holm_step_down_controls_family_wise_error():
    """On a truly faithful stream, FWER should be controlled at α."""
    rejections = 0
    trials = 25
    for seed in range(trials):
        ff = fresh_faithfuller(f"holm-{seed}", min_observations=32)
        ff.observe_many(synthetic_faithful_stream(64, seed=seed))
        cert = ff.certify()
        if cert.holm_rejected:
            rejections += 1
    # α = 0.05 → expect very few rejections on faithful streams.  Allow
    # generous slack to keep the test stable on small N.
    assert rejections <= 3


def test_product_evalue_grows_on_unfaithful():
    ff = fresh_faithfuller("ev-good", min_observations=32)
    ff.observe_many(synthetic_faithful_stream(96, seed=1))
    good = ff.certify().product_evalue
    ff2 = fresh_faithfuller("ev-bad", min_observations=32)
    ff2.observe_many(synthetic_unfaithful_stream(96, seed=1))
    bad = ff2.certify().product_evalue
    assert bad > good


# -- Determinism / replay ----------------------------------------------


def test_replay_verifiability():
    stream = list(synthetic_faithful_stream(64, seed=42))
    a = fresh_faithfuller("r1")
    a.observe_many(stream)
    cert_a = a.certify()
    b = fresh_faithfuller("r1")
    b.observe_many(stream)
    cert_b = b.certify()
    assert cert_a.fingerprint == cert_b.fingerprint
    assert a.fingerprint == b.fingerprint


def test_fingerprint_chain_advances_with_each_observation():
    ff = fresh_faithfuller("chain")
    fps = [ff.fingerprint]
    for obs in synthetic_faithful_stream(8, seed=2):
        ff.observe(obs)
        fps.append(ff.fingerprint)
    # All consecutive fingerprints should differ.
    assert len(set(fps)) == len(fps)


# -- Event emission -----------------------------------------------------


def test_event_emission_lifecycle():
    bus = _MemoryBus()
    ff = fresh_faithfuller("bus-test", bus=bus, min_observations=8)
    ff.observe_many(synthetic_faithful_stream(16, seed=3))
    cert = ff.certify()
    ff.report()

    kinds = [ev for ev, _ in bus.events]
    assert FF_STARTED in kinds
    assert FF_OBSERVED in kinds
    assert FF_CERTIFIED in kinds
    assert FF_REPORTED in kinds
    # On a faithful stream, no alert should fire.
    if cert.verdict == VERDICT_TRUST:
        assert FF_ALERTED not in kinds


def test_alert_event_on_violation():
    bus = _MemoryBus()
    ff = fresh_faithfuller("bus-bad", bus=bus, min_observations=8)
    ff.observe_many(synthetic_unfaithful_stream(64, seed=4))
    ff.certify()
    kinds = [ev for ev, _ in bus.events]
    assert FF_ALERTED in kinds


def test_emit_resilient_to_buggy_bus():
    class BoomBus:
        def emit(self, *_, **__):
            raise RuntimeError("bus down")

    ff = fresh_faithfuller("bus-broken", bus=BoomBus())
    # Should not raise.
    ff.observe_many(synthetic_faithful_stream(8, seed=5))


# -- Budget updates -----------------------------------------------------


def test_update_budget_emits_event_and_keeps_state():
    bus = _MemoryBus()
    ff = fresh_faithfuller("bget", bus=bus)
    ff.observe_many(synthetic_faithful_stream(32, seed=6))
    n_before = ff.n_observations
    new_cfg = ff.update_budget(min_mediation_gap=0.02)
    assert new_cfg.min_mediation_gap == pytest.approx(0.02)
    assert ff.n_observations == n_before  # state preserved
    kinds = [ev for ev, _ in bus.events]
    assert FF_BUDGET_UPDATED in kinds


# -- Reset --------------------------------------------------------------


def test_reset_clears_state():
    ff = fresh_faithfuller("rs")
    ff.observe_many(synthetic_faithful_stream(32, seed=7))
    assert ff.n_observations == 32
    ff.reset()
    assert ff.n_observations == 0
    # Should raise on certify (insufficient).
    with pytest.raises(InsufficientData):
        ff.certify()


# -- Report -------------------------------------------------------------


def test_report_has_recent_observations_and_counts():
    ff = fresh_faithfuller("rep")
    ff.observe_many(synthetic_faithful_stream(48, seed=8))
    ff.certify()
    rep = ff.report()
    assert isinstance(rep, FaithfullerReport)
    assert rep.n_observations == 48
    assert rep.last_verdict in KNOWN_VERDICTS
    assert rep.last_recommendation in KNOWN_RECOMMENDATIONS
    assert len(rep.recent_observations) > 0
    # Each perturbation kind should be counted on each round of the
    # synthetic stream.
    for kind in (
        PERTURB_TRUNCATE,
        PERTURB_BIAS,
        PERTURB_EDIT,
        PERTURB_NO_COT,
        PERTURB_FILLER,
        PERTURB_PARAPHRASE,
        PERTURB_NONE,
    ):
        assert rep.perturbation_counts[kind] == 48


# -- Stress: large stream remains bounded -------------------------------


def test_window_bounded_on_long_stream():
    ff = fresh_faithfuller("long", window_size=64)
    ff.observe_many(synthetic_faithful_stream(512, seed=9))
    rep = ff.report()
    # Windowed tests should be ≤ window_size, but the e-process count
    # remains the full stream length.
    assert ff.n_observations == 512
    assert len(rep.recent_observations) <= 32  # report cap


# -- Multi-threading safety smoke test ---------------------------------


def test_concurrent_observe_and_certify():
    import threading

    ff = fresh_faithfuller("conc", min_observations=8)
    stop = threading.Event()
    errors: list[BaseException] = []

    def producer():
        try:
            for obs in synthetic_faithful_stream(200, seed=42):
                if stop.is_set():
                    break
                ff.observe(obs)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def consumer():
        try:
            for _ in range(50):
                if ff.n_observations >= 16:
                    ff.certify()
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


# -- Anytime-validity: peeking doesn't blow up rejection rate ----------


def test_peeking_does_not_inflate_rejection_under_h0():
    """Anytime-validity: certifying after each round on a faithful
    stream should not yield a rejection rate exceeding α (with slack
    for finite-N noise)."""
    n_trials = 20
    n_rejections = 0
    for seed in range(n_trials):
        ff = fresh_faithfuller(
            f"peek-{seed}",
            min_observations=16,
            alpha=0.05,
        )
        rejected_once = False
        # Observe in small chunks and certify after each.
        stream = synthetic_faithful_stream(64, seed=seed)
        chunk: list = []
        for i, obs in enumerate(stream):
            chunk.append(obs)
            if (i + 1) % 16 == 0:
                ff.observe_many(chunk)
                chunk.clear()
                cert = ff.certify()
                if cert.verdict in (VERDICT_DEGRADE, VERDICT_REJECT):
                    rejected_once = True
                    break
        if rejected_once:
            n_rejections += 1
    # With α=0.05 and FWER-style fusion, expect ~0-2 rejections on 20
    # trials of a faithful stream.  Allow up to 5 for stability.
    assert n_rejections <= 5


# -- Certificate JSON-serialisability ---------------------------------


def test_certificate_is_json_serialisable():
    import json
    from dataclasses import asdict

    ff = fresh_faithfuller("json", min_observations=8)
    ff.observe_many(synthetic_faithful_stream(16, seed=10))
    cert = ff.certify()
    payload = asdict(cert)
    # Should round-trip through JSON.
    s = json.dumps(payload, default=lambda o: o.__dict__)
    assert isinstance(s, str)
    assert "fingerprint" in s
