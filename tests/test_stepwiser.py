"""Tests for :mod:`agi.stepwiser` — process reward modelling."""
from __future__ import annotations

import math
import random

import pytest

from agi.stepwiser import (
    AGG_LAST,
    AGG_LOGSUMEXP,
    AGG_MEAN,
    AGG_MIN,
    AGG_PROD,
    CAL_ISOTONIC,
    CAL_NONE,
    CAL_PLATT,
    DriftDetected,
    InsufficientData,
    InvalidConfig,
    InvalidStep,
    InvalidTrajectory,
    MODEL_ORM,
    MODEL_PRM,
    MODEL_VALUE,
    NotFitted,
    SHEPHERD_HARD,
    SHEPHERD_SOFT,
    StepScore,
    Stepwiser,
    StepwiserCertificate,
    StepwiserConfig,
    StepwiserReport,
    TrajectoryRecord,
    UnknownAggregator,
    UnknownCalibrator,
    UnknownModel,
    expected_calibration_error,
    hash_step_features,
    isotonic_apply,
    isotonic_fit,
    ks_two_sample,
    platt_apply,
    platt_fit,
    stepwiser_last_aggregate,
    stepwiser_ledger_root,
    stepwiser_logsumexp_aggregate,
    stepwiser_mcts_backup,
    stepwiser_mean_aggregate,
    stepwiser_min_aggregate,
    stepwiser_potential_shaping,
    stepwiser_prod_aggregate,
)


# ---------------------------------------------------------------------------
# Aggregator helpers
# ---------------------------------------------------------------------------


def test_aggregators_basic():
    p = [0.9, 0.7, 0.8]
    assert stepwiser_min_aggregate(p) == pytest.approx(0.7)
    assert stepwiser_mean_aggregate(p) == pytest.approx(0.8)
    assert stepwiser_prod_aggregate(p) == pytest.approx(0.9 * 0.7 * 0.8)
    assert stepwiser_last_aggregate(p) == pytest.approx(0.8)


def test_aggregators_empty():
    assert stepwiser_min_aggregate([]) == 0.0
    assert stepwiser_mean_aggregate([]) == 0.0
    assert stepwiser_prod_aggregate([]) == 0.0
    assert stepwiser_last_aggregate([]) == 0.0
    assert stepwiser_logsumexp_aggregate([]) == 0.0


def test_logsumexp_recovers_min_at_high_beta():
    p = [0.9, 0.6, 0.8]
    soft = stepwiser_logsumexp_aggregate(p, beta=100.0)
    assert abs(soft - min(p)) < 1e-3


def test_logsumexp_invalid_beta():
    with pytest.raises(InvalidConfig):
        stepwiser_logsumexp_aggregate([0.5], beta=0.0)


def test_prod_clamps_out_of_range():
    # negative / >1 entries clamped to [0,1]
    assert stepwiser_prod_aggregate([-0.1, 0.5]) == 0.0
    assert stepwiser_prod_aggregate([1.5, 0.5]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Featurizer
# ---------------------------------------------------------------------------


def test_hash_step_features_deterministic():
    a = hash_step_features(["foo bar"], feature_dim=64)
    b = hash_step_features(["foo bar"], feature_dim=64)
    assert a == b


def test_hash_step_features_index_changes():
    a = hash_step_features(["x"], feature_dim=128, step_index=0)
    b = hash_step_features(["x"], feature_dim=128, step_index=5)
    assert a != b


def test_hash_step_features_validates():
    with pytest.raises(InvalidConfig):
        hash_step_features(["x"], feature_dim=2)
    with pytest.raises(InvalidConfig):
        hash_step_features(["x"], feature_dim=64, ngram_n=0)
    with pytest.raises(InvalidStep):
        hash_step_features([123], feature_dim=64)  # type: ignore[list-item]


def test_hash_step_features_bias_always_present():
    feats = hash_step_features(["x"], feature_dim=64)
    assert 0 in feats


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def test_platt_fit_apply():
    rng = random.Random(0)
    logits = [rng.uniform(-3, 3) for _ in range(200)]
    labels = [1 if z > 0 else 0 for z in logits]
    A, B = platt_fit(logits, labels)
    probs = platt_apply(logits, A, B)
    assert all(0.0 <= p <= 1.0 for p in probs)
    # Mean prediction should track mean label.
    assert abs(sum(probs) / len(probs) - sum(labels) / len(labels)) < 0.15


def test_platt_length_mismatch():
    with pytest.raises(InvalidConfig):
        platt_fit([0.1], [0, 1])


def test_isotonic_pava():
    scores = [0.1, 0.4, 0.35, 0.8]
    labels = [0, 1, 0, 1]
    thresh, vals = isotonic_fit(scores, labels)
    # Output values must be non-decreasing.
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1] - 1e-12


def test_isotonic_apply_monotone():
    thresh = (0.1, 0.3, 0.7)
    vals = (0.0, 0.5, 1.0)
    assert isotonic_apply(0.0, thresh, vals) == 0.0
    assert isotonic_apply(1.0, thresh, vals) == 1.0
    assert 0.0 <= isotonic_apply(0.2, thresh, vals) <= 1.0


def test_isotonic_empty_passthrough():
    assert isotonic_apply(0.5, (), ()) == 0.5


def test_ece_perfectly_calibrated():
    # 100 samples uniformly distributed across the unit interval with
    # labels matching their probability bucket.
    probs = []
    labels = []
    for i in range(100):
        p = (i + 0.5) / 100
        probs.append(p)
        labels.append(1 if i % 2 == 0 else 0)
    ece = expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0


def test_ks_two_sample_identical_is_zero():
    a = [0.1, 0.2, 0.3, 0.4]
    assert ks_two_sample(a, a) == 0.0


def test_ks_two_sample_disjoint():
    a = [0.0, 0.1, 0.2]
    b = [0.8, 0.9, 1.0]
    assert ks_two_sample(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


def test_potential_shaping_terminal_zero():
    vals = [0.5, 0.6]
    shaped = stepwiser_potential_shaping(vals, discount=1.0)
    # r_0 = 1.0 * 0.6 - 0.5 = 0.1; r_1 = 1.0 * 0 - 0.6 = -0.6
    assert shaped[0] == pytest.approx(0.1)
    assert shaped[1] == pytest.approx(-0.6)


def test_potential_shaping_invalid_discount():
    with pytest.raises(InvalidConfig):
        stepwiser_potential_shaping([0.5], discount=1.5)


# ---------------------------------------------------------------------------
# MCTS backup
# ---------------------------------------------------------------------------


def test_mcts_backup_length_mismatch():
    with pytest.raises(InvalidConfig):
        stepwiser_mcts_backup([0.5], [1, 2])


def test_mcts_backup_alpha_one_returns_leaf_values():
    leaves = [0.1, 0.7, 0.9]
    n = [5, 10, 1]
    out = stepwiser_mcts_backup(leaves, n, alpha=1.0)
    assert out == pytest.approx(leaves)


def test_mcts_backup_alpha_zero_uses_weighted_mean():
    leaves = [0.0, 1.0]
    n = [1, 3]
    out = stepwiser_mcts_backup(leaves, n, alpha=0.0)
    weighted = (0.0 * 1 + 1.0 * 3) / 4
    assert all(x == pytest.approx(weighted) for x in out)


def test_mcts_backup_zero_visits_falls_back_to_mean():
    leaves = [0.2, 0.6, 0.4]
    n = [0, 0, 0]
    out = stepwiser_mcts_backup(leaves, n, alpha=0.0)
    avg = sum(leaves) / len(leaves)
    assert all(x == pytest.approx(avg) for x in out)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_defaults_ok():
    StepwiserConfig()


@pytest.mark.parametrize("kw", [
    {"model": "garbage"},
    {"aggregator": "garbage"},
    {"calibrator": "garbage"},
    {"shepherd_estimation": "garbage"},
    {"feature_dim": 0},
    {"ngram_n": 0},
    {"learning_rate": -1.0},
    {"l2": -1.0},
    {"epochs": 0},
    {"momentum": 1.5},
    {"holdout_fraction": 0.0},
    {"max_ece": 2.0},
    {"k_beam": 0},
    {"branch_factor": 0},
    {"max_depth": 0},
    {"quantile": -0.1},
    {"discount": 1.5},
    {"logsumexp_beta": -1.0},
    {"drift_alpha": 0.0},
    {"confidence": 1.0},
    {"rng_seed": "no"},
    {"hmac_key": "not-bytes"},
])
def test_config_rejects_bad(kw):
    with pytest.raises((InvalidConfig, UnknownModel, UnknownAggregator, UnknownCalibrator)):
        StepwiserConfig(**kw)


# ---------------------------------------------------------------------------
# Trajectory validation
# ---------------------------------------------------------------------------


def test_observe_validates():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    with pytest.raises(InvalidTrajectory):
        sw.observe(TrajectoryRecord(steps=(), step_labels=None, outcome=1))
    with pytest.raises(InvalidTrajectory):
        sw.observe(TrajectoryRecord(steps=("a",), step_labels=(0, 1), outcome=1))
    with pytest.raises(InvalidTrajectory):
        sw.observe(TrajectoryRecord(steps=("a",), step_labels=(2,), outcome=1))
    with pytest.raises(InvalidTrajectory):
        sw.observe(TrajectoryRecord(steps=("a",), step_labels=None, outcome=None))
    with pytest.raises(InvalidTrajectory):
        sw.observe(TrajectoryRecord(
            steps=("a",), step_labels=None, outcome=1, weight=0.0,
        ))


def test_observe_many_returns_count():
    sw = Stepwiser(StepwiserConfig(model=MODEL_ORM))
    traj = [
        TrajectoryRecord(steps=("a", "b"), outcome=1),
        TrajectoryRecord(steps=("c",), outcome=0),
    ]
    assert sw.observe_many(traj) == 2
    assert sw.report().n_trajectories_observed == 2


# ---------------------------------------------------------------------------
# Training, scoring, and selection
# ---------------------------------------------------------------------------


def _synthetic_pool(rng: random.Random, n: int = 240) -> list[TrajectoryRecord]:
    """Synthetic reasoning trajectories where 'good' steps contain the
    word 'good' and 'bad' steps contain 'bad'.  PRM should learn this
    quickly."""
    out = []
    for _ in range(n):
        T = rng.randint(2, 5)
        steps = []
        labels = []
        for _ in range(T):
            if rng.random() < 0.6:
                steps.append("a good step taken")
                labels.append(1)
            else:
                steps.append("a bad step indeed")
                labels.append(0)
        outcome = 1 if all(labels) else 0
        out.append(TrajectoryRecord(
            steps=tuple(steps),
            step_labels=tuple(labels),
            outcome=outcome,
        ))
    return out


def test_fit_then_score_separates_good_from_bad():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    rep = sw.fit()
    assert rep.fitted
    assert rep.train_accuracy > 0.7
    good = sw.score(["a good step taken", "a good step taken"])
    bad = sw.score(["a bad step indeed", "a bad step indeed"])
    assert good.aggregated > bad.aggregated


def test_fit_requires_data():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    with pytest.raises(InsufficientData):
        sw.fit()


def test_score_requires_fit():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    sw.observe(TrajectoryRecord(steps=("a",), step_labels=(1,), outcome=1))
    with pytest.raises(NotFitted):
        sw.score(["x"])


def test_score_empty_rejected():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    with pytest.raises(InvalidTrajectory):
        sw.score([])


def test_score_per_step_count_matches():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    sc = sw.score(["a", "b", "c"])
    assert len(sc.per_step) == 3
    for i, ss in enumerate(sc.per_step):
        assert ss.step_index == i
        assert 0.0 <= ss.calibrated_prob <= 1.0


def test_orm_model_trains_on_outcome_only():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_ORM, calibrator=CAL_NONE))
    traj = [
        TrajectoryRecord(steps=("good step", "good step"), outcome=1)
        for _ in range(50)
    ] + [
        TrajectoryRecord(steps=("bad step", "bad step"), outcome=0)
        for _ in range(50)
    ]
    sw.observe_many(traj)
    rep = sw.fit()
    assert rep.train_accuracy > 0.9


def test_value_model_uses_outcome_as_prefix_label():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_VALUE, calibrator=CAL_NONE))
    traj = [
        TrajectoryRecord(steps=("a", "b", "c"), outcome=1)
        for _ in range(50)
    ] + [
        TrajectoryRecord(steps=("x", "y", "z"), outcome=0)
        for _ in range(50)
    ]
    sw.observe_many(traj)
    rep = sw.fit()
    assert rep.fitted


def test_orm_falls_back_to_last_step_label():
    sw = Stepwiser(StepwiserConfig(model=MODEL_ORM, calibrator=CAL_NONE))
    sw.observe(TrajectoryRecord(steps=("a", "b"), step_labels=(1, 0)))
    sw.observe(TrajectoryRecord(steps=("c",), step_labels=(1,)))
    sw.observe(TrajectoryRecord(steps=("d", "e"), step_labels=(0, 1)))
    rep = sw.fit()
    assert rep.fitted


# ---------------------------------------------------------------------------
# Best-of-N selection
# ---------------------------------------------------------------------------


def test_best_of_n_picks_good():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    cands = [
        ["a bad step indeed", "a bad step indeed"],
        ["a good step taken", "a good step taken"],
        ["a bad step indeed", "a good step taken"],
    ]
    sel = sw.best_of_n(cands)
    assert sel.chosen_index == 1
    assert sel.selection_gap >= 0.0
    assert sel.selection_gap_lcb >= 0.0


def test_best_of_n_requires_candidates():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    sw.observe_many(_synthetic_pool(random.Random(0)))
    sw.fit()
    with pytest.raises(InvalidTrajectory):
        sw.best_of_n([])


def test_best_of_n_single_candidate_gap_is_score():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    sel = sw.best_of_n([["a good step taken"]])
    assert sel.chosen_index == 0
    assert sel.selection_gap == pytest.approx(sel.chosen_score.aggregated)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def test_beam_search_prefers_good_path():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, aggregator=AGG_MIN, k_beam=2, branch_factor=2,
        max_depth=3, calibrator=CAL_NONE,
    ))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()

    options = ["a good step taken", "a bad step indeed"]

    def expand(beam):
        return options

    def terminal(beam):
        return len(beam) >= 3

    result = sw.beam_search(("a good step taken",), expand=expand, terminal=terminal)
    assert result.depth >= 1
    top_beam = result.beams[0]
    # The PRM should prefer all-good trajectories.
    assert top_beam.count("a good step taken") >= top_beam.count("a bad step indeed")


def test_beam_search_terminal_immediate():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    result = sw.beam_search(
        ("a",),
        expand=lambda b: ["x"],
        terminal=lambda b: True,
    )
    assert result.depth == 0


def test_beam_search_invalid_params():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    with pytest.raises(InvalidConfig):
        sw.beam_search(("a",), expand=lambda b: ["x"], terminal=lambda b: False, k_beam=0)


def test_beam_search_str_required_from_expand():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    with pytest.raises(InvalidStep):
        sw.beam_search(
            ("a",),
            expand=lambda b: [123],  # type: ignore[list-item]
            terminal=lambda b: len(b) >= 2,
        )


# ---------------------------------------------------------------------------
# Quantilization
# ---------------------------------------------------------------------------


def test_quantilize_step_returns_above_threshold():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, calibrator=CAL_NONE, quantile=0.1,
    ))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    chosen, scored = sw.quantilize_step(
        ["a good step taken"],
        ["a bad step indeed", "a good step taken"],
    )
    assert chosen is not None
    assert len(scored) == 2


def test_quantilize_step_requires_fit():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    sw.observe(TrajectoryRecord(steps=("a",), step_labels=(1,)))
    with pytest.raises(NotFitted):
        sw.quantilize_step(["a"], ["b"])


def test_quantilize_step_requires_candidates():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    with pytest.raises(InvalidStep):
        sw.quantilize_step(["a"], [])


# ---------------------------------------------------------------------------
# Reward shaping integration
# ---------------------------------------------------------------------------


def test_shape_returns_per_step_rewards():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    shaped = sw.shape(["a good step taken", "a good step taken"])
    assert len(shaped) == 2
    assert all(isinstance(r, float) for r in shaped)


# ---------------------------------------------------------------------------
# Calibration with Platt and Isotonic
# ---------------------------------------------------------------------------


def test_platt_calibration_lowers_ece():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, calibrator=CAL_PLATT, max_ece=1.0,
    ))
    sw.observe_many(_synthetic_pool(rng, n=300))
    rep = sw.fit()
    assert rep.calibrated


def test_isotonic_calibration_runs():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, calibrator=CAL_ISOTONIC, max_ece=1.0,
    ))
    sw.observe_many(_synthetic_pool(rng, n=300))
    rep = sw.fit()
    assert rep.calibrated


def test_calibration_none_passthrough():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    sc = sw.score(["a good step taken"])
    # raw == calibrated when CAL_NONE.
    assert sc.per_step[0].raw_prob == pytest.approx(sc.per_step[0].calibrated_prob)


def test_calibration_ece_ceiling_enforced():
    # Set a 0.0 ECE ceiling that the calibrator can't hit; should reject.
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, calibrator=CAL_PLATT, max_ece=0.0,
    ))
    sw.observe_many(_synthetic_pool(rng))
    with pytest.raises(InvalidConfig):
        sw.fit()


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_drift_flags_disjoint_stream():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, calibrator=CAL_NONE,
    ))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    # Long stream of very long prefixes — totally different feature density.
    stream = [["zzzz " * 100] * 5 for _ in range(20)]
    res = sw.drift(stream)
    assert 0.0 <= res.ks_statistic <= 1.0
    assert res.threshold > 0.0


def test_drift_requires_fit():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    with pytest.raises(NotFitted):
        sw.drift([["a"]])


def test_drift_requires_stream():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    with pytest.raises(InvalidTrajectory):
        sw.drift([])


# ---------------------------------------------------------------------------
# Reporting and certification
# ---------------------------------------------------------------------------


def test_report_before_and_after_fit():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    rep_before = sw.report()
    assert not rep_before.fitted
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    rep_after = sw.report()
    assert rep_after.fitted
    assert rep_after.n_trajectories_observed > 0


def test_certify_returns_lcb_below_accuracy():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    cert = sw.certify()
    assert isinstance(cert, StepwiserCertificate)
    assert cert.accuracy_lcb <= cert.holdout_accuracy + 1e-9
    assert cert.hoeffding_half_width > 0.0


def test_certify_requires_fit():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    with pytest.raises(NotFitted):
        sw.certify()


# ---------------------------------------------------------------------------
# Top features (skill mining)
# ---------------------------------------------------------------------------


def test_top_features_returns_top_k():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    top = sw.top_features(k=8)
    assert len(top) <= 8
    # Ordered by |weight| descending.
    for i in range(1, len(top)):
        assert abs(top[i - 1][1]) >= abs(top[i][1])


def test_top_features_requires_fit():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    with pytest.raises(NotFitted):
        sw.top_features()


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------


def test_snapshot_restore_round_trip():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    sc1 = sw.score(["a good step taken"])
    snap = sw.snapshot()
    sw2 = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw2.restore(snap)
    sc2 = sw2.score(["a good step taken"])
    assert sc1.aggregated == pytest.approx(sc2.aggregated)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_state():
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()
    assert sw.report().n_trajectories_observed > 0
    sw.reset()
    rep = sw.report()
    assert rep.n_trajectories_observed == 0
    assert not rep.fitted


# ---------------------------------------------------------------------------
# Ledger chain
# ---------------------------------------------------------------------------


def test_ledger_root_is_deterministic():
    assert stepwiser_ledger_root() == stepwiser_ledger_root()


def test_chain_head_advances_on_observe():
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM))
    h0 = sw.chain_head
    sw.observe(TrajectoryRecord(steps=("a",), step_labels=(1,)))
    h1 = sw.chain_head
    assert h0 != h1


def test_chain_head_replay_byte_equality():
    """Two identical sequences of operations must yield identical chains."""
    rng = random.Random(0)
    pool = _synthetic_pool(rng)
    sw1 = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw2 = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    for t in pool:
        sw1.observe(t)
        sw2.observe(t)
    sw1.fit()
    sw2.fit()
    assert sw1.chain_head == sw2.chain_head


def test_chain_diverges_when_hmac_keys_differ():
    pool = _synthetic_pool(random.Random(0))
    sw1 = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, calibrator=CAL_NONE, hmac_key=b"k1",
    ))
    sw2 = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, calibrator=CAL_NONE, hmac_key=b"k2",
    ))
    for t in pool[:5]:
        sw1.observe(t)
        sw2.observe(t)
    assert sw1.chain_head != sw2.chain_head


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------


def test_thread_safety_smoke():
    import threading
    rng = random.Random(0)
    sw = Stepwiser(StepwiserConfig(model=MODEL_PRM, calibrator=CAL_NONE))
    sw.observe_many(_synthetic_pool(rng))
    sw.fit()

    def worker():
        for _ in range(50):
            sw.score(["a good step taken"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sw.report().n_scored >= 200
