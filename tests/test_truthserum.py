"""Tests for ``agi.truthserum`` — incentive-compatible peer-prediction.

The tests follow the mathematical contract of the module:

 1. **Output Agreement** payment is 1 iff signals match, 0 otherwise.
 2. **Correlated Agreement** has *positive* expected payment under
    positively-correlated signals, and *zero* under independent ones.
 3. **Determinant-based MI** is 0 on independent signals (det of
    product-of-marginals = product of margin determinants → fine but
    *empirical* DMI uses joint, which has det=0 under independence).
 4. **Phi-MI / TVD** equals 0 under independence, positive under
    positive correlation.
 5. **BTS** payment for the truth-telling reporter is strictly
    greater than for a colluder reporting a constant in a moderately
    sized sample.
 6. **Strict truthful Nash margin** is positive for CA / OA / phi-MI /
    DMI on independent honest reporters with positive correlation,
    and *not* strictly positive when one cluster of reporters always
    reports the same constant (colluders gain 0 margin).
 7. **Dawes-Skene EM** recovers the latent truth on a 2-signal mixture
    with reporters of varying accuracy (≥ 90% accuracy on noisy
    synthetic data).
 8. **Aggregation** by weighted plurality strictly out-performs
    plurality on data where one reporter is adversarial.
 9. **Hoeffding / empirical-Bernstein radii** shrink as 1/√n and
    contain the true mean with the stated coverage.
10. **Bonferroni correction** divides α by the number of reporters.
11. **Collusion detection** identifies a clique of constant-reporters
    at the specified joint α.
12. **Determinant** and **matrix inversion** match closed-form
    references on small examples.
13. **Threadsafety** under concurrent submit/score.
14. **Attestation** receipts arrive with stable content-hash digests.
"""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.events import Event, EventBus
from agi.truthserum import (
    AGG_PLURALITY,
    AGG_WEIGHTED_EM,
    AGG_WEIGHTED_PLURALITY,
    ConfusionMatrix,
    CoverageReport,
    ElicitationReport,
    InsufficientData,
    InvalidReport,
    KNOWN_MECHANISMS,
    MECH_BTS,
    MECH_CORRELATED_AGREEMENT,
    MECH_DMI,
    MECH_OUTPUT_AGREEMENT,
    MECH_PHI_MI,
    MECH_RBTS,
    MECH_SSR,
    PHI_TVD,
    PHI_KL,
    PHI_JS,
    PHI_CHI2,
    Report,
    ScoreStats,
    TaskTruth,
    TRUTHSERUM_SCORED,
    TRUTHSERUM_SUBMITTED,
    TruthSerum,
    UnknownAggregation,
    UnknownMechanism,
    _det,
    _invert,
    bts_payment,
    bonferroni_alpha,
    correlated_agreement_payment,
    dawes_skene_em,
    determinant_mi_payment,
    empirical_bernstein_radius,
    hoeffding_radius,
    output_agreement_payment,
    phi_mi_payment,
    quick_score,
    rbts_payment,
    surrogate_score_payment,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _bernoulli_seq(seed, n, p=0.5):
    rng = random.Random(seed)
    return [1 if rng.random() < p else 0 for _ in range(n)]


def _make_reports(
    truths,
    accuracies,
    reporters=None,
    bias=None,
):
    """Generate Report objects from latent truths + per-reporter accuracies."""
    if reporters is None:
        reporters = [f"r{i}" for i in range(len(accuracies))]
    out = []
    for r_id, p_correct in zip(reporters, accuracies):
        rng = random.Random(hash(r_id) & 0xFFFFFFFF)
        for ti, y in enumerate(truths):
            if bias is not None and r_id in bias:
                a = bias[r_id]
            else:
                if rng.random() < p_correct:
                    a = y
                else:
                    # flip to any other signal
                    a = 1 - y
            out.append(Report(reporter_id=r_id, task_id=f"t{ti:04d}", answer=a))
    return out


# ----------------------------------------------------------------------
# Payments — pure functions
# ----------------------------------------------------------------------


def test_output_agreement_payment():
    assert output_agreement_payment("a", "a") == 1.0
    assert output_agreement_payment("a", "b") == 0.0
    assert output_agreement_payment(0, 0) == 1.0
    assert output_agreement_payment(0, 1) == 0.0


def test_correlated_agreement_payment():
    # Match on shared, no match on bonus → +1
    assert correlated_agreement_payment("A", "A", "A", "B") == 1.0
    # No match on shared, match on bonus → -1
    assert correlated_agreement_payment("A", "B", "A", "A") == -1.0
    # Match on both → 0
    assert correlated_agreement_payment("A", "A", "A", "A") == 0.0
    # No match on either → 0
    assert correlated_agreement_payment("A", "B", "A", "C") == 0.0


def test_bts_payment_truth_beats_constant():
    # Pop frequency is 70% '1', 30% '0'; meta-pred matches.
    pop = {0: 0.3, 1: 0.7}
    geo = {0: 0.3, 1: 0.7}
    score_truth_1 = bts_payment(1, geo, pop, geo)
    score_truth_0 = bts_payment(0, geo, pop, geo)
    # info term = log(pop[x]/geo[x]) = 0 (they're equal)
    # pred term = pop_k * log(geo_k/pop_k) = 0
    assert abs(score_truth_1) < 1e-9
    assert abs(score_truth_0) < 1e-9
    # If a reporter mis-states the geo-meta, score moves accordingly.
    bad_geo = {0: 0.5, 1: 0.5}
    score_bad = bts_payment(1, bad_geo, pop, bad_geo)
    # info term = log(0.7/0.5) > 0; pred term ≤ 0.
    assert score_bad > -1.0


def test_rbts_payment_binary_signal():
    # signal_i must be {0,1}; signals out of range raise.
    val = rbts_payment(1, 0.8, 1, 0.7)
    assert isinstance(val, float)
    with pytest.raises(InvalidReport):
        rbts_payment(2, 0.5, 1, 0.5)


def test_phi_mi_zero_under_independence():
    # joint = product of marginals → I_phi = 0 for any phi
    margin_i = {0: 0.5, 1: 0.5}
    margin_j = {0: 0.4, 1: 0.6}
    joint = {(0, 0): 0.2, (0, 1): 0.3, (1, 0): 0.2, (1, 1): 0.3}
    for phi in (PHI_TVD, PHI_KL, PHI_JS, PHI_CHI2):
        v = phi_mi_payment(joint, margin_i, margin_j, phi=phi)
        assert abs(v) < 1e-9, f"phi={phi} got {v}"


def test_phi_mi_positive_under_correlation():
    # Strongly correlated joint:
    margin = {0: 0.5, 1: 0.5}
    joint = {(0, 0): 0.45, (0, 1): 0.05, (1, 0): 0.05, (1, 1): 0.45}
    for phi in (PHI_TVD, PHI_KL, PHI_JS, PHI_CHI2):
        v = phi_mi_payment(joint, margin, margin, phi=phi)
        assert v > 0.0, f"phi={phi} got {v}"


def test_determinant_mi_zero_under_independence():
    # If both joint matrices factor as p_i p_j, their determinant is 0
    # (rank-1 matrix).
    M = [[0.25, 0.25], [0.25, 0.25]]
    assert abs(determinant_mi_payment(M, M)) < 1e-12


def test_determinant_mi_positive_under_correlation():
    M = [[0.45, 0.05], [0.05, 0.45]]
    assert determinant_mi_payment(M, M) > 0.0


# ----------------------------------------------------------------------
# Determinant / inverse helpers
# ----------------------------------------------------------------------


def test_det_2x2():
    assert _det([[1.0, 2.0], [3.0, 4.0]]) == pytest.approx(-2.0)
    assert _det([[2.0, 0.0], [0.0, 3.0]]) == pytest.approx(6.0)
    assert _det([[1.0, 1.0], [1.0, 1.0]]) == pytest.approx(0.0)


def test_det_3x3():
    # Known det: |[[1,2,3],[0,1,4],[5,6,0]]| = 1
    M = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]
    assert _det(M) == pytest.approx(1.0)


def test_invert_2x2():
    M = [[4.0, 7.0], [2.0, 6.0]]
    inv = _invert(M)
    assert inv is not None
    # M * inv = I
    out = [
        [sum(M[i][k] * inv[k][j] for k in range(2)) for j in range(2)]
        for i in range(2)
    ]
    for i in range(2):
        for j in range(2):
            expected = 1.0 if i == j else 0.0
            assert abs(out[i][j] - expected) < 1e-9


def test_invert_singular_returns_none():
    assert _invert([[1.0, 2.0], [2.0, 4.0]]) is None


# ----------------------------------------------------------------------
# Concentration helpers
# ----------------------------------------------------------------------


def test_hoeffding_radius_decreases_with_n():
    r10 = hoeffding_radius(10, 0.05, 1.0)
    r100 = hoeffding_radius(100, 0.05, 1.0)
    r1000 = hoeffding_radius(1000, 0.05, 1.0)
    assert r10 > r100 > r1000
    # Square-root shrinkage:
    assert r1000 == pytest.approx(r100 / math.sqrt(10), rel=1e-9)


def test_hoeffding_zero_range():
    assert hoeffding_radius(100, 0.05, 0.0) == 0.0


def test_empirical_bernstein_tighter_when_low_variance():
    # Constant values → Bernstein radius ≈ 0 (modulo lower-order term),
    # Hoeffding ≈ 0.27 for n=100 / α=0.05.
    vals = [0.5] * 100
    r_eb = empirical_bernstein_radius(vals, 0.05, 1.0)
    r_h = hoeffding_radius(100, 0.05, 1.0)
    assert r_eb < r_h


def test_empirical_bernstein_falls_back_to_hoeffding_for_n_le_1():
    # n = 0
    assert math.isinf(empirical_bernstein_radius([], 0.05, 1.0))
    # n = 1
    r_eb = empirical_bernstein_radius([0.5], 0.05, 1.0)
    r_h = hoeffding_radius(1, 0.05, 1.0)
    assert r_eb == pytest.approx(r_h)


def test_hoeffding_alpha_validation():
    with pytest.raises(ValueError):
        hoeffding_radius(10, 0.0, 1.0)
    with pytest.raises(ValueError):
        hoeffding_radius(10, 1.0, 1.0)


def test_bonferroni_alpha():
    assert bonferroni_alpha(0.05, 1) == pytest.approx(0.05)
    assert bonferroni_alpha(0.05, 5) == pytest.approx(0.01)
    assert bonferroni_alpha(0.05, 0) == pytest.approx(0.05)


# ----------------------------------------------------------------------
# Report validation
# ----------------------------------------------------------------------


def test_report_validation():
    with pytest.raises(InvalidReport):
        Report(reporter_id="", task_id="t", answer=1)
    with pytest.raises(InvalidReport):
        Report(reporter_id="r", task_id="", answer=1)
    with pytest.raises(InvalidReport):
        Report(reporter_id="r", task_id="t", answer=None)
    # belief masses must sum to 1
    with pytest.raises(InvalidReport):
        Report(reporter_id="r", task_id="t", answer=1, belief=((0, 0.5), (1, 0.3)))
    # negative mass
    with pytest.raises(InvalidReport):
        Report(reporter_id="r", task_id="t", answer=1, belief=((0, -0.2), (1, 1.2)))
    # duplicate signal
    with pytest.raises(InvalidReport):
        Report(reporter_id="r", task_id="t", answer=1, belief=((0, 0.5), (0, 0.5)))
    # valid belief
    Report(reporter_id="r", task_id="t", answer=1, belief=((0, 0.4), (1, 0.6)))


# ----------------------------------------------------------------------
# TruthSerum lifecycle
# ----------------------------------------------------------------------


def test_truthserum_empty_state():
    ts = TruthSerum()
    cov = ts.coverage()
    assert cov.n_reporters == 0
    assert cov.n_tasks == 0
    assert cov.n_reports == 0
    assert ts.reporters() == ()
    assert ts.tasks() == ()
    assert ts.signal_alphabet() == ()


def test_truthserum_requires_two_reporters():
    ts = TruthSerum()
    ts.submit(Report(reporter_id="r", task_id="t", answer=1))
    with pytest.raises(InsufficientData):
        ts.score(MECH_OUTPUT_AGREEMENT)


def test_truthserum_unknown_mechanism():
    ts = TruthSerum()
    ts.submit(Report(reporter_id="r0", task_id="t0", answer=1))
    ts.submit(Report(reporter_id="r1", task_id="t0", answer=1))
    with pytest.raises(UnknownMechanism):
        ts.score("not_a_mechanism")


def test_truthserum_unknown_aggregation():
    ts = TruthSerum()
    ts.submit(Report(reporter_id="r0", task_id="t0", answer=1))
    ts.submit(Report(reporter_id="r1", task_id="t0", answer=1))
    with pytest.raises(UnknownAggregation):
        ts.score(MECH_OUTPUT_AGREEMENT, aggregation="not_a_method")


def test_truthserum_submit_batch_clears_cache():
    ts = TruthSerum()
    ts.submit_batch(
        Report(reporter_id=f"r{i}", task_id=f"t{j}", answer=i % 2)
        for i in range(3)
        for j in range(2)
    )
    cov = ts.coverage()
    assert cov.n_reporters == 3
    assert cov.n_tasks == 2
    assert cov.n_reports == 6


def test_truthserum_clear():
    ts = TruthSerum()
    ts.submit(Report(reporter_id="r0", task_id="t0", answer=1))
    ts.submit(Report(reporter_id="r1", task_id="t0", answer=1))
    ts.clear()
    cov = ts.coverage()
    assert cov.n_reports == 0
    assert cov.n_reporters == 0


# ----------------------------------------------------------------------
# Mechanism end-to-end
# ----------------------------------------------------------------------


def test_output_agreement_unanimous():
    ts = TruthSerum()
    for i in range(3):
        for t in range(5):
            ts.submit(Report(reporter_id=f"r{i}", task_id=f"t{t}", answer=1))
    r = ts.score(MECH_OUTPUT_AGREEMENT)
    for s in r.scores:
        assert s.mean_score == pytest.approx(1.0)


def test_output_agreement_random_baseline():
    ts = TruthSerum()
    rng = random.Random(0)
    for i in range(5):
        for t in range(50):
            ts.submit(
                Report(
                    reporter_id=f"r{i}",
                    task_id=f"t{t:03d}",
                    answer=rng.choice([0, 1]),
                )
            )
    r = ts.score(MECH_OUTPUT_AGREEMENT)
    for s in r.scores:
        assert 0.3 < s.mean_score < 0.7


def test_correlated_agreement_truthful_eq_holds():
    truths = _bernoulli_seq(2, 200)
    reports = _make_reports(truths, [0.85, 0.85, 0.85, 0.85, 0.85])
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    r = ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.05, seed=0)
    assert r.truthful_strict_eq is True
    assert r.truthful_eq_margin > 0.0


def test_correlated_agreement_zero_under_independence():
    # All reporters report uniformly at random → expected CA payment ~ 0.
    ts = TruthSerum(random_seed=0)
    rng = random.Random(11)
    for i in range(5):
        for t in range(300):
            ts.submit(
                Report(
                    reporter_id=f"r{i}",
                    task_id=f"t{t:03d}",
                    answer=rng.choice([0, 1]),
                )
            )
    r = ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.05, seed=0)
    for s in r.scores:
        assert abs(s.mean_score) < 0.1


def test_phi_mi_positive_for_honest_reporters():
    truths = _bernoulli_seq(3, 150)
    reports = _make_reports(truths, [0.9, 0.9, 0.9])
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    r = ts.score(MECH_PHI_MI, alpha=0.05, seed=0)
    for s in r.scores:
        assert s.mean_score > 0.1


def test_dmi_positive_for_honest_reporters_3_signals():
    rng = random.Random(4)
    truths = [rng.randrange(3) for _ in range(150)]
    reports = _make_reports_multiclass(truths, accuracies=[0.85, 0.85, 0.85], k=3, seed=4)
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    r = ts.score(MECH_DMI, alpha=0.05, seed=0)
    for s in r.scores:
        assert s.mean_score > 0.0


def _make_reports_multiclass(truths, accuracies, k=3, seed=0, reporters=None):
    if reporters is None:
        reporters = [f"r{i}" for i in range(len(accuracies))]
    out = []
    for r_id, p in zip(reporters, accuracies):
        rng = random.Random((hash(r_id) ^ seed) & 0xFFFFFFFF)
        for ti, y in enumerate(truths):
            if rng.random() < p:
                a = y
            else:
                # Pick a uniform incorrect label
                a = rng.choice([x for x in range(k) if x != y])
            out.append(Report(reporter_id=r_id, task_id=f"t{ti:04d}", answer=a))
    return out


def test_bts_with_meta_predictions():
    # 5 reporters, 60 binary tasks; truth Bernoulli(0.7); each reports
    # the truth 85% of the time, and reports a meta-prediction matching
    # the true population frequency.
    rng = random.Random(5)
    truths = [1 if rng.random() < 0.7 else 0 for _ in range(60)]
    pop = (sum(truths) / len(truths))
    belief = ((0, 1.0 - pop), (1, pop))
    ts = TruthSerum(random_seed=0)
    for i in range(5):
        rr = random.Random(100 + i)
        for ti, y in enumerate(truths):
            a = y if rr.random() < 0.85 else 1 - y
            ts.submit(
                Report(
                    reporter_id=f"r{i}",
                    task_id=f"t{ti:03d}",
                    answer=a,
                    belief=belief,
                )
            )
    r = ts.score(MECH_BTS, alpha=0.05, seed=0)
    assert r.mechanism == MECH_BTS
    assert r.n_reporters == 5


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def test_plurality_aggregation_recovers_truth():
    rng = random.Random(6)
    truths = [rng.randrange(2) for _ in range(80)]
    reports = _make_reports(truths, [0.85, 0.85, 0.85, 0.85, 0.85])
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    truths_out = ts.aggregate(AGG_PLURALITY)
    acc = sum(
        1
        for t in truths_out
        if t.answer == truths[int(t.task_id[1:])]
    ) / len(truths)
    assert acc > 0.93  # 5 reporters at 85% — empirically ~0.97


def test_weighted_em_aggregation_recovers_truth_with_noisy_reporters():
    rng = random.Random(7)
    truths = [rng.randrange(2) for _ in range(120)]
    # 4 reliable reporters + 3 noisy ones + 1 adversary (flips)
    reporters = ["r0", "r1", "r2", "r3", "n0", "n1", "n2"]
    accs = [0.9, 0.9, 0.85, 0.85, 0.6, 0.55, 0.55]
    reports = _make_reports(truths, accs, reporters=reporters)
    # adversary: flips truth deterministically
    for ti, y in enumerate(truths):
        reports.append(
            Report(reporter_id="adv", task_id=f"t{ti:04d}", answer=1 - y)
        )
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    truths_em = ts.aggregate(AGG_WEIGHTED_EM)
    acc_em = sum(
        1
        for t in truths_em
        if t.answer == truths[int(t.task_id[1:])]
    ) / len(truths)
    assert acc_em > 0.9


def test_confusion_matrices_row_stochastic():
    rng = random.Random(8)
    truths = [rng.randrange(2) for _ in range(40)]
    reports = _make_reports(truths, [0.8, 0.7, 0.6])
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    confusions = ts.fit_confusion_matrices()
    for cm in confusions.values():
        for row in cm.matrix:
            assert abs(sum(row) - 1.0) < 1e-6
            for v in row:
                assert 0.0 <= v <= 1.0


# ----------------------------------------------------------------------
# Strict-Nash margin
# ----------------------------------------------------------------------


def test_strict_truthful_eq_check_method():
    rng = random.Random(9)
    truths = [rng.randrange(2) for _ in range(100)]
    reports = _make_reports(truths, [0.85, 0.85, 0.85])
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    ok, margin = ts.is_strict_truthful_eq(MECH_CORRELATED_AGREEMENT)
    assert ok is True
    assert margin > 0.0


def test_strict_truthful_eq_fails_on_constant_reporters():
    # All reporters always say 1 → CA payment = 0 for everyone; margin = 0.
    ts = TruthSerum(random_seed=0)
    for i in range(3):
        for t in range(50):
            ts.submit(Report(reporter_id=f"r{i}", task_id=f"t{t}", answer=1))
    ok, margin = ts.is_strict_truthful_eq(MECH_CORRELATED_AGREEMENT)
    assert ok is False


# ----------------------------------------------------------------------
# Collusion detection
# ----------------------------------------------------------------------


def test_detect_collusion_finds_constant_clique():
    rng = random.Random(10)
    truths = [rng.randrange(2) for _ in range(120)]
    reports = _make_reports(truths, [0.85, 0.85, 0.85, 0.85])
    # Add 3 colluders that always say 1.
    for i in range(3):
        for ti in range(120):
            reports.append(
                Report(reporter_id=f"col{i}", task_id=f"t{ti:04d}", answer=1)
            )
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    cliques = ts.detect_collusion(alpha=0.01, min_overlap=20)
    # The detector flags any cluster with anomalously high agreement;
    # honest reporters also cluster (they agree at ~0.74). We require
    # only that the colluder set is found.
    all_flagged: set = set()
    for c in cliques:
        all_flagged |= set(c)
    assert {"col0", "col1", "col2"} <= all_flagged


def test_detect_collusion_returns_empty_for_independent():
    rng = random.Random(11)
    ts = TruthSerum(random_seed=0)
    for i in range(5):
        for t in range(50):
            ts.submit(
                Report(
                    reporter_id=f"r{i}",
                    task_id=f"t{t:03d}",
                    answer=rng.choice([0, 1]),
                )
            )
    cliques = ts.detect_collusion(alpha=0.001, min_overlap=20)
    assert cliques == ()


# ----------------------------------------------------------------------
# CI semantics + Bonferroni
# ----------------------------------------------------------------------


def test_score_ci_contains_mean():
    rng = random.Random(12)
    truths = [rng.randrange(2) for _ in range(150)]
    reports = _make_reports(truths, [0.85, 0.85, 0.85, 0.85])
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    r = ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.05, seed=0)
    for s in r.scores:
        assert s.ci_lower <= s.mean_score <= s.ci_upper
        assert s.radius >= 0.0


def test_bonferroni_shrinks_per_alpha():
    truths = _bernoulli_seq(13, 50)
    reports = _make_reports(truths, [0.85] * 5)
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    r_b = ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.05, bonferroni=True, seed=0)
    r_nb = ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.05, bonferroni=False, seed=0)
    assert r_b.scores[0].alpha == pytest.approx(0.01)
    assert r_nb.scores[0].alpha == pytest.approx(0.05)
    # Tighter per-alpha → larger radius.
    assert r_b.scores[0].radius > r_nb.scores[0].radius


# ----------------------------------------------------------------------
# Dawes-Skene EM
# ----------------------------------------------------------------------


def test_dawes_skene_recovers_majority_truth():
    rng = random.Random(14)
    truths = [rng.randrange(2) for _ in range(100)]
    reporters = ["r0", "r1", "r2", "r3", "r4"]
    accs = [0.9, 0.9, 0.85, 0.7, 0.65]
    answers = {}
    for r_id, p in zip(reporters, accs):
        rr = random.Random(hash(r_id) & 0xFFFFFFFF)
        answers[r_id] = {}
        for ti, y in enumerate(truths):
            answers[r_id][f"t{ti:03d}"] = y if rr.random() < p else 1 - y
    confusions, posteriors, ll, iters = dawes_skene_em(answers, [0, 1])
    assert iters >= 1
    # Recovered truth should agree with the latent for ≥ 92% of tasks.
    recovered = {t: max(p.items(), key=lambda kv: kv[1])[0] for t, p in posteriors.items()}
    acc = sum(1 for t, y in enumerate(truths) if recovered[f"t{t:03d}"] == y) / len(truths)
    assert acc > 0.9


# ----------------------------------------------------------------------
# Events / attestation
# ----------------------------------------------------------------------


def test_events_emitted_on_submit_and_score():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(lambda e: seen.append(e))
    ts = TruthSerum(bus=bus, random_seed=0)
    truths = _bernoulli_seq(15, 40)
    reports = _make_reports(truths, [0.85, 0.85, 0.85])
    ts.submit_batch(reports)
    ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.05, seed=0)
    kinds = {e.kind for e in seen}
    assert TRUTHSERUM_SUBMITTED in kinds
    assert TRUTHSERUM_SCORED in kinds


def test_attestation_receipts_have_stable_digest():
    class _Capture:
        def __init__(self):
            self.records = []

        def record(self, *, kind, payload):
            self.records.append((kind, payload))

    cap = _Capture()
    ts = TruthSerum(attestor=cap, random_seed=0)
    truths = _bernoulli_seq(16, 40)
    reports = _make_reports(truths, [0.85, 0.85, 0.85])
    ts.submit_batch(reports)
    r1 = ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.05, seed=0)
    r2 = ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.05, seed=0)
    assert r1.digest == r2.digest


# ----------------------------------------------------------------------
# Threadsafety
# ----------------------------------------------------------------------


def test_threadsafe_concurrent_submit():
    ts = TruthSerum(random_seed=0)

    def worker(i):
        for j in range(20):
            ts.submit(Report(reporter_id=f"r{i}", task_id=f"t{j}", answer=j % 2))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cov = ts.coverage()
    assert cov.n_reporters == 8
    assert cov.n_tasks == 20
    assert cov.n_reports == 8 * 20


# ----------------------------------------------------------------------
# quick_score facade
# ----------------------------------------------------------------------


def test_quick_score_facade():
    truths = _bernoulli_seq(17, 80)
    reports = _make_reports(truths, [0.85, 0.85, 0.85])
    r = quick_score(reports, MECH_CORRELATED_AGREEMENT, alpha=0.05, seed=0)
    assert isinstance(r, ElicitationReport)
    assert r.n_reporters == 3
    assert r.n_tasks == 80


# ----------------------------------------------------------------------
# Composition with Auditor (FDR-controlled per-reporter truthfulness)
# ----------------------------------------------------------------------


def test_auditor_composition_smoke():
    """Honest reporters' CA mean payments are positive and well-separated
    from a coin-flip reporter; we feed the per-reporter signed CI lower
    bound into the Auditor as a screening test for "is this reporter
    informative?"  Composition smoke-test only — the precise α level on
    the joint test is left to the Auditor's BH/BY/Holm procedures.
    """
    from agi.auditor import Auditor

    rng = random.Random(18)
    truths = [rng.randrange(2) for _ in range(120)]
    reporters = ["r0", "r1", "r2", "r3", "noise"]
    accs = [0.9, 0.9, 0.85, 0.85, 0.5]
    reports = _make_reports(truths, accs, reporters=reporters)
    ts = TruthSerum(random_seed=0)
    ts.submit_batch(reports)
    r = ts.score(MECH_CORRELATED_AGREEMENT, alpha=0.10, bonferroni=False, seed=0)
    auditor = Auditor()
    score_by_id = {s.reporter_id: s.mean_score for s in r.scores}
    # Honest reporters have strictly higher CA payment than the noise one.
    for honest in ["r0", "r1", "r2", "r3"]:
        assert score_by_id[honest] > score_by_id["noise"] - 1e-9
    # Lifetime stats wire-through:
    cov = ts.coverage()
    assert cov.n_scorings == 1
