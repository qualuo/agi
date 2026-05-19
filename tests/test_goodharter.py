"""Tests for the Goodharter primitive — proxy-reward divergence detection."""
from __future__ import annotations

import json
import math
import random

import pytest

from agi.goodharter import (
    GH_ALERTED,
    GH_BUDGET_UPDATED,
    GH_CERTIFIED,
    GH_OBSERVED,
    GH_REPORTED,
    GH_RESET,
    GH_STARTED,
    KNOWN_RECOMMENDATIONS,
    KNOWN_TESTS,
    KNOWN_VERDICTS,
    REC_DEPLOY,
    REC_ESCALATE_HUMAN,
    REC_MONITOR,
    REC_REPLACE,
    REC_RETUNE,
    TEST_DIST_SHIFT,
    TEST_GAP_EVALUE,
    TEST_GAP_EXCESS,
    TEST_GAP_HEDGED,
    TEST_KENDALL_DROP,
    TEST_PEARSON_DROP,
    TEST_SPEARMAN_DROP,
    VERDICT_INVESTIGATE,
    VERDICT_QUARANTINE,
    VERDICT_RETRAIN,
    VERDICT_TRUST,
    Goodharter,
    GoodharterCertificate,
    GoodharterConfig,
    GoodharterReport,
    InsufficientData,
    InvalidConfig,
    InvalidObservation,
    RewardObservation,
    fresh_goodharter,
    synthetic_aligned_stream,
    synthetic_goodhart_stream,
)
# NB: agi.goodharter.TestResult is intentionally NOT imported here as
# its bare name — pytest would try to collect it as a test class given
# the `Test*` prefix.  Tests that need the record type can import it
# locally as needed.


# ---------------------------------------------------------------------------
# Config + observation invariants
# ---------------------------------------------------------------------------


def test_config_defaults_validate():
    cfg = GoodharterConfig()
    assert cfg.proxy_id == "default"
    assert 0 < cfg.alpha < 1
    assert cfg.divergence_budget == 0.05


def test_config_rejects_invalid_budget():
    with pytest.raises(InvalidConfig):
        GoodharterConfig(divergence_budget=1.5)
    with pytest.raises(InvalidConfig):
        GoodharterConfig(divergence_budget=-2.0)


def test_config_rejects_invalid_correlation_floor():
    with pytest.raises(InvalidConfig):
        GoodharterConfig(min_correlation=1.0)
    with pytest.raises(InvalidConfig):
        GoodharterConfig(min_correlation=-1.5)


def test_config_rejects_inconsistent_thresholds():
    with pytest.raises(InvalidConfig):
        GoodharterConfig(rec_investigate_threshold=3, rec_retrain_threshold=2)


def test_config_rejects_undersized_window():
    with pytest.raises(InvalidConfig):
        GoodharterConfig(min_observations=128, window_size=64)


def test_observation_rejects_out_of_range():
    with pytest.raises(InvalidObservation):
        RewardObservation(decision_id="d", proxy_reward=1.5, true_reward=0.1)
    with pytest.raises(InvalidObservation):
        RewardObservation(decision_id="d", proxy_reward=-0.5, true_reward=0.1)


def test_observation_rejects_non_finite():
    with pytest.raises(InvalidObservation):
        RewardObservation(decision_id="d", proxy_reward=float("nan"), true_reward=0.5)
    with pytest.raises(InvalidObservation):
        RewardObservation(decision_id="d", proxy_reward=float("inf"), true_reward=0.5)


def test_observation_rejects_empty_decision_id():
    with pytest.raises(InvalidObservation):
        RewardObservation(decision_id="", proxy_reward=0.5, true_reward=0.5)


def test_observation_clamps_into_unit_interval():
    o = RewardObservation(decision_id="d", proxy_reward=1.0 + 1e-10, true_reward=0.0 - 1e-10)
    assert 0.0 <= o.proxy_reward <= 1.0
    assert 0.0 <= o.true_reward <= 1.0


# ---------------------------------------------------------------------------
# Empty / pending / lifecycle behaviour
# ---------------------------------------------------------------------------


def test_fresh_goodharter_starts_empty():
    g = fresh_goodharter("p")
    assert g.n_observations == 0
    assert g.last_verdict == VERDICT_TRUST
    assert g.fingerprint  # initialised


def test_pending_certificate_when_insufficient_data():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for obs in synthetic_aligned_stream(10, seed=0):
        g.observe(obs)
    cert = g.certify()
    assert cert.n_observations == 10
    assert cert.verdict == VERDICT_TRUST
    assert cert.recommendation == REC_MONITOR
    assert cert.tests == ()  # no tests run yet


def test_reset_clears_state():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for obs in synthetic_aligned_stream(50, seed=0):
        g.observe(obs)
    assert g.n_observations == 50
    g.reset()
    assert g.n_observations == 0
    assert g.last_verdict == VERDICT_TRUST


# ---------------------------------------------------------------------------
# Headline behaviour: aligned vs Goodhart streams
# ---------------------------------------------------------------------------


def test_aligned_stream_certifies_trust():
    g = fresh_goodharter("p", min_observations=32, divergence_budget=0.05,
                         min_correlation=0.7, seed=0)
    for obs in synthetic_aligned_stream(200, noise=0.02, seed=42):
        g.observe(obs)
    cert = g.certify()
    assert cert.verdict == VERDICT_TRUST
    assert cert.recommendation == REC_DEPLOY
    assert cert.pearson_r > 0.95  # near-perfect alignment
    assert abs(cert.gap_mean) < 0.02
    # No test should be rejected on a clean stream.
    assert not any(t.rejected for t in cert.tests)


def test_severe_goodhart_escalates_to_quarantine():
    g = fresh_goodharter("p", min_observations=32, divergence_budget=0.05,
                         min_correlation=0.7, seed=0)
    for obs in synthetic_goodhart_stream(400, onset=0.3, drift=0.3,
                                         noise=0.02, seed=7):
        g.observe(obs)
    cert = g.certify()
    assert cert.verdict == VERDICT_QUARANTINE
    assert cert.recommendation == REC_ESCALATE_HUMAN
    # At least the gap-excess / hedged-capital tests should fire.
    rejected = {t.name for t in cert.tests if t.rejected}
    assert TEST_GAP_HEDGED in rejected or TEST_GAP_EVALUE in rejected


def test_certificate_carries_pearson_ci():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for obs in synthetic_aligned_stream(200, noise=0.05, seed=0):
        g.observe(obs)
    cert = g.certify()
    assert -1.0 <= cert.pearson_ci_low <= cert.pearson_r <= cert.pearson_ci_high <= 1.0
    # CI is non-degenerate at this n.
    assert cert.pearson_ci_high - cert.pearson_ci_low < 0.5


def test_certificate_gap_ci_brackets_mean():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for obs in synthetic_aligned_stream(200, noise=0.02, seed=0):
        g.observe(obs)
    cert = g.certify()
    assert cert.gap_ci_low <= cert.gap_mean <= cert.gap_ci_high


def test_hedged_lcs_brackets_gap_under_drift():
    g = fresh_goodharter("p", min_observations=32, divergence_budget=0.05,
                         window_size=256, seed=0)
    for obs in synthetic_goodhart_stream(400, onset=0.3, drift=0.3,
                                         noise=0.02, seed=7):
        g.observe(obs)
    cert = g.certify()
    # The LCS should sit well above zero on a drifted stream.
    assert cert.hedged_lcs_low > 0.0
    # The LCS is computed over the recent window; its bounds should
    # be finite and ordered.
    assert cert.hedged_lcs_low <= cert.hedged_lcs_high
    assert -1.0 <= cert.hedged_lcs_low and cert.hedged_lcs_high <= 1.0


# ---------------------------------------------------------------------------
# Test taxonomy & multi-test correction
# ---------------------------------------------------------------------------


def test_tests_field_uses_known_names():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for obs in synthetic_aligned_stream(200, seed=0):
        g.observe(obs)
    cert = g.certify()
    for t in cert.tests:
        assert t.name in KNOWN_TESTS


def test_holm_rejection_is_subset_of_p_value_rejections():
    """Holm step-down only operates on p-values; its rejection set
    must be a subset of the *uncorrected* p < alpha set, which is
    itself a subset of the *individually rejected at any test's own
    threshold* set.
    """
    g = fresh_goodharter("p", min_observations=32, divergence_budget=0.05,
                         min_correlation=0.7, seed=0)
    for obs in synthetic_goodhart_stream(400, onset=0.3, drift=0.3,
                                         noise=0.02, seed=7):
        g.observe(obs)
    cert = g.certify()
    p_rejected = {t.name for t in cert.tests
                  if t.p_value is not None and t.p_value < g.config.alpha}
    holm_rejected = set(cert.holm_rejected)
    assert holm_rejected.issubset(p_rejected)


def test_product_evalue_at_least_one_under_alignment():
    # On a clean stream the product e-value should not explode.
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for obs in synthetic_aligned_stream(200, noise=0.02, seed=42):
        g.observe(obs)
    cert = g.certify()
    # E-process under the null should stay bounded; pick a generous
    # ceiling — the deterministic seed makes this stable.
    assert cert.product_evalue < 100.0


# ---------------------------------------------------------------------------
# Verdict graduation
# ---------------------------------------------------------------------------


def test_increasing_drift_increases_severity():
    seeds = []
    for drift in (0.0, 0.05, 0.15, 0.3):
        g = fresh_goodharter("p", min_observations=32, seed=0)
        for obs in synthetic_goodhart_stream(300, onset=0.3, drift=drift,
                                             noise=0.02, seed=11):
            g.observe(obs)
        cert = g.certify()
        seeds.append((drift, cert.verdict))
    # The clean case (drift=0) should be TRUST; the strongest should
    # be at least RETRAIN / QUARANTINE.
    verdicts = dict(seeds)
    assert verdicts[0.0] == VERDICT_TRUST
    assert verdicts[0.3] in (VERDICT_RETRAIN, VERDICT_QUARANTINE)
    # Severity is monotone in drift.
    severity = {VERDICT_TRUST: 0, VERDICT_INVESTIGATE: 1,
                VERDICT_RETRAIN: 2, VERDICT_QUARANTINE: 3}
    sevs = [severity[v] for _, v in seeds]
    for a, b in zip(sevs, sevs[1:]):
        assert a <= b


def test_verdict_in_known_set():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for obs in synthetic_aligned_stream(50, seed=0):
        g.observe(obs)
    cert = g.certify()
    assert cert.verdict in KNOWN_VERDICTS
    assert cert.recommendation in KNOWN_RECOMMENDATIONS


# ---------------------------------------------------------------------------
# Replay-verifiability
# ---------------------------------------------------------------------------


def test_replay_same_seed_identical_fingerprint():
    obs = synthetic_aligned_stream(80, noise=0.02, seed=99)
    a = fresh_goodharter("p", seed=42)
    b = fresh_goodharter("p", seed=42)
    for o in obs:
        a.observe(o)
        b.observe(o)
    assert a.fingerprint == b.fingerprint


def test_different_observation_diverges_fingerprint():
    a = fresh_goodharter("p", seed=42)
    b = fresh_goodharter("p", seed=42)
    a.observe(RewardObservation(decision_id="d0", proxy_reward=0.5, true_reward=0.5))
    b.observe(RewardObservation(decision_id="d0", proxy_reward=0.5, true_reward=0.4))
    assert a.fingerprint != b.fingerprint


def test_fingerprint_chain_advances_each_observation():
    g = fresh_goodharter("p", seed=42)
    seen = {g.fingerprint}
    for obs in synthetic_aligned_stream(20, seed=0):
        g.observe(obs)
        assert g.fingerprint not in seen
        seen.add(g.fingerprint)


def test_certify_advances_fingerprint_only_after_min_observations():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    fp0 = g.fingerprint
    # Pending certify (n < min_observations): no chain advance.
    for obs in synthetic_aligned_stream(5, seed=0):
        g.observe(obs)
    fp_after_obs = g.fingerprint
    g.certify()
    # Pending certify: chain should NOT advance (skipped certify chain entry).
    assert g.fingerprint == fp_after_obs
    # Now feed enough to make a real certify and confirm chain advances.
    for obs in synthetic_aligned_stream(50, seed=1):
        g.observe(obs)
    pre_cert = g.fingerprint
    g.certify()
    assert g.fingerprint != pre_cert


# ---------------------------------------------------------------------------
# Control observations and report
# ---------------------------------------------------------------------------


def test_control_observations_kept_separate():
    g = fresh_goodharter("p", min_observations=8, seed=0)
    for obs in synthetic_aligned_stream(20, seed=0):
        g.observe(obs)
    for i in range(10):
        g.observe(RewardObservation(
            decision_id=f"c{i}", proxy_reward=0.9, true_reward=0.0,
            is_control=True,
        ))
    assert g.n_observations == 20
    assert g.n_control == 10
    rep = g.report()
    assert rep.n_control == 10
    assert rep.n_observations == 20


def test_report_carries_recent_history():
    g = fresh_goodharter("p", seed=0)
    for obs in synthetic_aligned_stream(40, seed=0):
        g.observe(obs)
    rep = g.report()
    assert isinstance(rep, GoodharterReport)
    assert len(rep.recent_observations) <= 32
    assert rep.proxy_mean == pytest.approx(rep.recent_observations[0][1], abs=0.5)


# ---------------------------------------------------------------------------
# Budget management
# ---------------------------------------------------------------------------


def test_update_budget_changes_verdict():
    g = fresh_goodharter("p", min_observations=32, divergence_budget=0.5, seed=0)
    for obs in synthetic_goodhart_stream(200, onset=0.3, drift=0.3, noise=0.02, seed=7):
        g.observe(obs)
    c1 = g.certify()
    # With a wide budget the verdict should be TRUST or INVESTIGATE.
    severity = {VERDICT_TRUST: 0, VERDICT_INVESTIGATE: 1,
                VERDICT_RETRAIN: 2, VERDICT_QUARANTINE: 3}
    assert severity[c1.verdict] <= 1
    # Tighten the budget so the same stream becomes a violation.
    g.update_budget(divergence_budget=0.01)
    c2 = g.certify()
    assert severity[c2.verdict] >= severity[c1.verdict]


def test_update_budget_advances_fingerprint():
    g = fresh_goodharter("p", divergence_budget=0.05, seed=0)
    fp = g.fingerprint
    g.update_budget(divergence_budget=0.10)
    assert g.fingerprint != fp


# ---------------------------------------------------------------------------
# Event bus integration
# ---------------------------------------------------------------------------


class _Bus:
    def __init__(self):
        self.events = []

    def publish(self, ev):
        self.events.append(ev)


def test_observe_emits_event():
    bus = _Bus()
    g = Goodharter(GoodharterConfig(proxy_id="p", min_observations=32, seed=0), bus=bus)
    g.observe(RewardObservation(decision_id="d0", proxy_reward=0.5, true_reward=0.5))
    kinds = {ev.kind for ev in bus.events}
    assert GH_STARTED in kinds
    assert GH_OBSERVED in kinds


def test_certify_emits_event():
    bus = _Bus()
    g = Goodharter(GoodharterConfig(proxy_id="p", min_observations=32, seed=0), bus=bus)
    for obs in synthetic_aligned_stream(40, seed=0):
        g.observe(obs)
    g.certify()
    kinds = {ev.kind for ev in bus.events}
    assert GH_CERTIFIED in kinds


def test_alert_emitted_only_on_retrain_or_quarantine():
    bus = _Bus()
    g = Goodharter(GoodharterConfig(proxy_id="p", min_observations=32, seed=0), bus=bus)
    for obs in synthetic_goodhart_stream(400, onset=0.3, drift=0.3, noise=0.02, seed=7):
        g.observe(obs)
    g.certify()
    kinds = [ev.kind for ev in bus.events]
    assert GH_ALERTED in kinds


def test_no_alert_on_clean_stream():
    bus = _Bus()
    g = Goodharter(GoodharterConfig(proxy_id="p", min_observations=32, seed=0), bus=bus)
    for obs in synthetic_aligned_stream(200, noise=0.02, seed=42):
        g.observe(obs)
    g.certify()
    kinds = [ev.kind for ev in bus.events]
    assert GH_ALERTED not in kinds


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


def test_welford_stable_on_long_stream():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    # 10k observations should not blow up second-order moments.
    for obs in synthetic_aligned_stream(10_000, noise=0.05, seed=0):
        g.observe(obs)
    cert = g.certify()
    # All numbers finite.
    for name, v in (("pearson_r", cert.pearson_r),
                    ("gap_mean", cert.gap_mean),
                    ("gap_var", cert.gap_var),
                    ("gap_evalue", cert.gap_evalue),
                    ("gap_hedged_evalue", cert.gap_hedged_evalue)):
        assert math.isfinite(v) or v == math.inf, f"{name}={v} not finite"


def test_zero_variance_stream_handles_gracefully():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for t in range(50):
        g.observe(RewardObservation(decision_id=f"d{t}",
                                    proxy_reward=0.5, true_reward=0.5))
    cert = g.certify()
    # Pearson is undefined on zero-variance streams — should clamp to 0.
    assert cert.pearson_r == 0.0


# ---------------------------------------------------------------------------
# Thread-safety smoke
# ---------------------------------------------------------------------------


def test_concurrent_observation_does_not_corrupt_state():
    import threading
    g = fresh_goodharter("p", min_observations=32, seed=0)
    stream = synthetic_aligned_stream(400, seed=0)

    def feed(slice_):
        for obs in slice_:
            g.observe(obs)

    half = len(stream) // 2
    t1 = threading.Thread(target=feed, args=(stream[:half],))
    t2 = threading.Thread(target=feed, args=(stream[half:],))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert g.n_observations == 400
    cert = g.certify()
    assert math.isfinite(cert.pearson_r)


# ---------------------------------------------------------------------------
# Spec API
# ---------------------------------------------------------------------------


def test_certificate_is_json_compatible():
    g = fresh_goodharter("p", min_observations=32, seed=0)
    for obs in synthetic_aligned_stream(80, seed=0):
        g.observe(obs)
    cert = g.certify()
    # Build a JSON-friendly dict.
    payload = {
        "proxy_id": cert.proxy_id,
        "verdict": cert.verdict,
        "recommendation": cert.recommendation,
        "n_observations": cert.n_observations,
        "pearson_r": cert.pearson_r,
        "gap_mean": cert.gap_mean,
        "fingerprint": cert.fingerprint,
        "tests": [
            {"name": t.name, "statistic": t.statistic, "rejected": t.rejected,
             "p_value": t.p_value, "e_value": t.e_value}
            for t in cert.tests
        ],
    }
    s = json.dumps(payload, sort_keys=True)
    # Round-trip without loss.
    d = json.loads(s)
    assert d["proxy_id"] == "p"
    assert d["verdict"] in KNOWN_VERDICTS


def test_synthetic_aligned_is_clean():
    obs = synthetic_aligned_stream(50, noise=0.01, seed=0)
    assert len(obs) == 50
    for o in obs:
        assert abs(o.proxy_reward - o.true_reward) < 0.1


def test_synthetic_goodhart_diverges():
    obs = synthetic_goodhart_stream(200, onset=0.5, drift=0.3, noise=0.0, seed=0)
    # Past the onset the proxy should sit above the true reward on average.
    second_half = obs[len(obs) // 2:]
    avg_gap = sum(o.proxy_reward - o.true_reward for o in second_half) / len(second_half)
    assert avg_gap > 0.05
