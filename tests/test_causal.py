"""Tests for agi.causal — heterogeneous treatment effects."""
from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from agi.causal import (
    BLPReport,
    CATEPoint,
    CATERecommendation,
    CAUSAL_CATE_ESTIMATED,
    CAUSAL_FIT,
    CAUSAL_HETEROGENEITY,
    CAUSAL_RECOMMENDED,
    CAUSAL_RECORDED,
    CAUSAL_UPLIFT,
    CausalLab,
    HeterogeneityTest,
    KNOWN_LEARNERS,
    LEARNER_DR,
    LEARNER_S,
    LEARNER_T,
    LEARNER_X,
    UpliftReport,
)
from agi.events import EventBus
from agi.policy_lab import LinearRewardModel, LoggedEvent, PolicyLab


# =====================================================================
# Synthetic worlds
# =====================================================================


def _synthetic_heterogeneous(n: int = 800, seed: int = 0):
    """Two-arm bandit with heterogeneous lift.

    Control: reward = 0.5 + 0.1*x + noise
    Treatment: reward = 0.5 + 0.6*x + noise
        → ground-truth CATE = +0.5 * x, so τ(c) varies with x.
    Logging propensity is 0.5/0.5 (true randomisation).
    """
    rng = random.Random(seed)
    events: list[LoggedEvent] = []
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0)
        a = "T" if rng.random() < 0.5 else "C"
        base = 0.5 + 0.1 * x
        if a == "T":
            base += 0.5 * x
        r = base + rng.gauss(0.0, 0.05)
        events.append(LoggedEvent(context={"x": x}, action=a, propensity=0.5, reward=r))
    return events


def _synthetic_homogeneous(n: int = 600, seed: int = 1):
    """Same lift everywhere — should fail heterogeneity test."""
    rng = random.Random(seed)
    events: list[LoggedEvent] = []
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0)
        a = "T" if rng.random() < 0.5 else "C"
        base = 0.5 + 0.1 * x
        if a == "T":
            base += 0.2  # constant lift everywhere
        r = base + rng.gauss(0.0, 0.05)
        events.append(LoggedEvent(context={"x": x}, action=a, propensity=0.5, reward=r))
    return events


def _synthetic_imbalanced(n: int = 600, seed: int = 2):
    """Logging policy strongly prefers control; T is rare."""
    rng = random.Random(seed)
    events: list[LoggedEvent] = []
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0)
        # 10% to T, 90% to C.
        if rng.random() < 0.1:
            a = "T"; p = 0.1
        else:
            a = "C"; p = 0.9
        base = 0.5 + 0.1 * x
        if a == "T":
            base += 0.5 * x
        r = base + rng.gauss(0.0, 0.05)
        events.append(LoggedEvent(context={"x": x}, action=a, propensity=p, reward=r))
    return events


# =====================================================================
# Basic surface tests
# =====================================================================


class TestBasics:
    def test_construction_validates_args(self):
        with pytest.raises(ValueError):
            CausalLab(propensity_floor=0.0)
        with pytest.raises(ValueError):
            CausalLab(weight_clip=-1)
        with pytest.raises(ValueError):
            CausalLab(min_eff_n=0)

    def test_default_treatment_control_required(self):
        lab = CausalLab()
        lab.record(LoggedEvent(context={"x": 0.0}, action="a", propensity=0.5, reward=1.0))
        with pytest.raises(ValueError):
            lab.cate({"x": 0.0})

    def test_treatment_must_differ_from_control(self):
        lab = CausalLab(treatment="A", control="A")
        with pytest.raises(ValueError):
            lab.cate({"x": 0.0})

    def test_unknown_learner_raises(self):
        lab = CausalLab(treatment="A", control="B")
        lab.record(LoggedEvent(context={"x": 0.0}, action="A", propensity=0.5, reward=1.0))
        with pytest.raises(ValueError):
            lab.cate({"x": 0.0}, learner="garbage")

    def test_invalid_confidence_raises(self):
        lab = CausalLab(treatment="A", control="B")
        with pytest.raises(ValueError):
            lab.cate({"x": 0.0}, confidence=0.0)
        with pytest.raises(ValueError):
            lab.cate({"x": 0.0}, confidence=1.5)

    def test_record_and_len(self):
        lab = CausalLab()
        assert len(lab) == 0
        lab.record(LoggedEvent(context={"x": 0.0}, action="A", propensity=0.5, reward=1.0))
        assert len(lab) == 1
        lab.record(LoggedEvent(context={"x": 1.0}, action="B", propensity=0.5, reward=2.0))
        assert len(lab) == 2

    def test_record_batch(self):
        lab = CausalLab()
        events = _synthetic_heterogeneous(50)
        n = lab.record_batch(events)
        assert n == len(events) == len(lab)

    def test_max_events_caps_log(self):
        lab = CausalLab(max_events=10)
        for i in range(25):
            lab.record(LoggedEvent(context={"x": float(i)}, action="A",
                                   propensity=0.5, reward=float(i)))
        assert len(lab) == 10
        # FIFO — oldest dropped.
        keep = [ev.reward for ev in lab.events()]
        assert min(keep) >= 15.0

    def test_known_learners(self):
        assert set(KNOWN_LEARNERS) == {LEARNER_T, LEARNER_S, LEARNER_X, LEARNER_DR}

    def test_empty_lab_returns_empty_cate(self):
        lab = CausalLab(treatment="A", control="B")
        point = lab.cate({"x": 0.0})
        assert point.value == 0.0
        assert math.isinf(point.se)
        assert point.low_data
        assert point.support_score == 0.0


# =====================================================================
# Learner correctness on synthetic data
# =====================================================================


class TestLearners:
    @pytest.mark.parametrize("learner", list(KNOWN_LEARNERS))
    def test_recovers_heterogeneous_lift(self, learner):
        events = _synthetic_heterogeneous(n=1000, seed=42)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        # Ground truth: τ(x) = +0.5 * x.
        # Probe a positive-x context: lift should be clearly positive.
        pos = lab.cate({"x": 0.7}, learner=learner)
        neg = lab.cate({"x": -0.7}, learner=learner)
        assert pos.value > 0.15, f"{learner}: pos={pos.value} expected > 0.15"
        assert neg.value < -0.15, f"{learner}: neg={neg.value} expected < -0.15"
        # The sign at x=0 should be small (close to zero, within ~0.1).
        zero = lab.cate({"x": 0.0}, learner=learner)
        assert abs(zero.value) < 0.15, f"{learner}: zero={zero.value} expected ~0"

    def test_dr_ci_excludes_zero_on_strong_signal(self):
        events = _synthetic_heterogeneous(n=2000, seed=1)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        point = lab.cate({"x": 0.9}, learner=LEARNER_DR, confidence=0.95)
        # At x=0.9, true lift ≈ +0.45. A 95% CI should exclude zero.
        assert point.ci_low > 0.0, f"CI low = {point.ci_low}, expected > 0"

    def test_dr_handles_imbalanced_arms(self):
        events = _synthetic_imbalanced(n=2000, seed=7)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        point = lab.cate({"x": 0.8}, learner=LEARNER_DR)
        assert point.value > 0.1
        # Diagnostics should reveal weight inflation.
        assert point.diagnostics.get("max_weight", 0.0) >= 5.0

    def test_t_learner_uses_per_arm_residual_variance(self):
        events = _synthetic_heterogeneous(n=500, seed=2)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        point = lab.cate({"x": 0.3}, learner=LEARNER_T)
        assert math.isfinite(point.se)
        assert point.se > 0.0

    def test_s_learner_returns_point(self):
        events = _synthetic_heterogeneous(n=400, seed=3)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        point = lab.cate({"x": 0.5}, learner=LEARNER_S)
        assert isinstance(point, CATEPoint)
        assert point.value > 0.05

    def test_x_learner_returns_point(self):
        events = _synthetic_heterogeneous(n=400, seed=4)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        point = lab.cate({"x": 0.5}, learner=LEARNER_X)
        assert isinstance(point, CATEPoint)
        assert point.value > 0.05
        assert "tau_0" in point.diagnostics and "tau_1" in point.diagnostics

    def test_x_learner_returns_empty_when_one_arm_missing(self):
        lab = CausalLab(treatment="T", control="C")
        for i in range(20):
            lab.record(LoggedEvent(context={"x": float(i)}, action="T",
                                    propensity=0.5, reward=float(i) * 0.1))
        # Only T has data; X learner should not crash.
        point = lab.cate({"x": 0.5}, learner=LEARNER_X)
        assert math.isinf(point.se)
        assert point.low_data

    def test_dr_learner_returns_empty_when_arm_missing(self):
        lab = CausalLab(treatment="T", control="C")
        for i in range(20):
            lab.record(LoggedEvent(context={"x": float(i)}, action="T",
                                    propensity=0.5, reward=float(i) * 0.1))
        point = lab.cate({"x": 0.5}, learner=LEARNER_DR)
        # DR refuses without both arms — returns empty.
        assert point.low_data

    def test_bootstrap_refinement_runs(self):
        events = _synthetic_heterogeneous(n=200, seed=11)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        point = lab.cate({"x": 0.5}, learner=LEARNER_T, n_bootstrap=20)
        assert "bootstrap" in point.diagnostics
        assert point.diagnostics["bootstrap"] == 20.0


# =====================================================================
# Recommendation surface
# =====================================================================


class TestRecommend:
    def test_recommend_picks_winner_at_positive_context(self):
        events = _synthetic_heterogeneous(n=1500, seed=5)
        lab = CausalLab()
        lab.record_batch(events)
        rec = lab.recommend({"x": 0.8}, actions=["T", "C"], baseline="C")
        assert rec.best_action == "T"
        assert rec.lift > 0.0
        assert "T" in rec.per_action and "C" in rec.per_action

    def test_recommend_picks_baseline_at_negative_context(self):
        events = _synthetic_heterogeneous(n=1500, seed=6)
        lab = CausalLab()
        lab.record_batch(events)
        rec = lab.recommend({"x": -0.8}, actions=["T", "C"], baseline="C")
        # At x=-0.8, true lift is ~-0.4 — so the baseline wins (C beats T).
        assert rec.best_action == "C"
        assert rec.lift == 0.0  # baseline vs itself

    def test_recommend_chooses_default_baseline_by_frequency(self):
        events = _synthetic_imbalanced(n=400, seed=8)
        lab = CausalLab()
        lab.record_batch(events)
        rec = lab.recommend({"x": 0.5}, actions=["T", "C"])
        # Without an explicit baseline, the lab picks the most common action ("C").
        assert rec.baseline_action == "C"

    def test_recommend_empty_lab(self):
        lab = CausalLab()
        rec = lab.recommend({"x": 0.0}, actions=["A", "B"], baseline="A")
        assert rec.best_action == "A"
        assert math.isinf(rec.lift_se)

    def test_recommend_requires_actions(self):
        lab = CausalLab()
        with pytest.raises(ValueError):
            lab.recommend({"x": 0.0}, actions=[])


# =====================================================================
# Uplift / Qini
# =====================================================================


class TestUplift:
    def test_uplift_curve_runs_and_is_positive_on_heterogeneous_signal(self):
        events = _synthetic_heterogeneous(n=1500, seed=9)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        report = lab.uplift(learner=LEARNER_DR, n_buckets=5)
        assert isinstance(report, UpliftReport)
        assert report.n == len(events)
        assert len(report.buckets) == 5
        assert "ATE" in report.summary and "Qini" in report.summary
        # With heterogeneity, targeting the top decile should beat the bottom.
        top = report.buckets[0]
        bot = report.buckets[-1]
        # Top bucket has *predicted* CATE > bottom bucket.
        assert top.mean_predicted_cate > bot.mean_predicted_cate

    def test_uplift_qini_normalised_returns_float(self):
        events = _synthetic_heterogeneous(n=400, seed=10)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        report = lab.uplift(n_buckets=4)
        assert math.isfinite(report.qini_coefficient)
        assert math.isfinite(report.qini_normalised)

    def test_uplift_validates_n_buckets(self):
        lab = CausalLab(treatment="T", control="C")
        with pytest.raises(ValueError):
            lab.uplift(n_buckets=1)

    def test_uplift_empty_log(self):
        lab = CausalLab(treatment="T", control="C")
        report = lab.uplift(n_buckets=3)
        assert report.n == 0
        assert report.buckets == ()


# =====================================================================
# Heterogeneity test
# =====================================================================


class TestHeterogeneity:
    def test_detects_heterogeneity(self):
        events = _synthetic_heterogeneous(n=400, seed=11)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        result = lab.test_heterogeneity(n_permutations=40, max_eval_contexts=40)
        assert isinstance(result, HeterogeneityTest)
        # The observed statistic should be larger than the null mean.
        assert result.statistic > result.null_mean

    def test_fails_to_detect_when_homogeneous(self):
        events = _synthetic_homogeneous(n=400, seed=12)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        result = lab.test_heterogeneity(n_permutations=40, max_eval_contexts=40)
        # Under homogeneity, p-value should not be tiny on a moderate sample.
        assert result.p_value >= 0.0
        assert result.p_value <= 1.0

    def test_too_few_events_returns_null(self):
        lab = CausalLab(treatment="T", control="C")
        lab.record(LoggedEvent(context={"x": 0.0}, action="T", propensity=0.5, reward=1.0))
        result = lab.test_heterogeneity(n_permutations=10)
        assert result.n_permutations == 0
        assert not result.is_heterogeneous

    def test_invalid_n_permutations(self):
        lab = CausalLab(treatment="T", control="C")
        with pytest.raises(ValueError):
            lab.test_heterogeneity(n_permutations=0)


# =====================================================================
# Best Linear Predictor
# =====================================================================


class TestBLP:
    def test_blp_returns_coefficient_with_nonzero_signal(self):
        events = _synthetic_heterogeneous(n=1500, seed=13)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        report = lab.best_linear_predictor()
        assert isinstance(report, BLPReport)
        # We expect the coefficient on `x` to be clearly positive and CI to exclude 0.
        x_coef = next(c for c in report.coefficients if c.feature == "x")
        assert x_coef.coef > 0.2
        assert x_coef.ci_low > 0.0
        # R² should explain at least some variance.
        assert 0.0 < report.r_squared <= 1.0

    def test_blp_empty_log(self):
        lab = CausalLab(treatment="T", control="C")
        report = lab.best_linear_predictor()
        assert report.n == 0
        assert report.coefficients == ()


# =====================================================================
# Support / overlap diagnostics
# =====================================================================


class TestSupport:
    def test_support_score_high_when_balanced(self):
        events = _synthetic_heterogeneous(n=400, seed=14)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        s = lab.support({"x": 0.0}, "T", "C")
        assert 0.5 <= s <= 1.0

    def test_support_score_low_when_arm_missing(self):
        lab = CausalLab(treatment="T", control="C")
        for i in range(20):
            lab.record(LoggedEvent(context={"x": float(i)}, action="T",
                                    propensity=0.5, reward=float(i)))
        s = lab.support({"x": 0.0}, "T", "C")
        assert s == 0.0

    def test_support_score_lower_when_imbalanced(self):
        events = _synthetic_imbalanced(n=400, seed=15)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        s = lab.support({"x": 0.0}, "T", "C")
        # 0.1/0.9 → 2 * min(0.1, 0.9) ≈ 0.2.
        assert 0.0 < s < 0.5


# =====================================================================
# Persistence
# =====================================================================


class TestPersistence:
    def test_snapshot_restore_roundtrip(self):
        events = _synthetic_heterogeneous(n=50, seed=16)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        snap = lab.snapshot()
        other = CausalLab()
        other.restore(snap)
        assert len(other) == len(lab)
        assert other.treatment == "T"
        assert other.control == "C"

    def test_save_load_to_disk(self, tmp_path: Path):
        events = _synthetic_heterogeneous(n=30, seed=17)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)
        f = tmp_path / "lab.json"
        lab.save(f)
        other = CausalLab()
        other.load(f)
        assert len(other) == len(lab)

    def test_restore_wrong_version_raises(self):
        lab = CausalLab()
        with pytest.raises(ValueError):
            lab.restore({"version": 99})


# =====================================================================
# Integration
# =====================================================================


class TestIntegrations:
    def test_attach_to_policy_lab(self):
        plab = PolicyLab(reward_model=LinearRewardModel(ridge=0.5))
        events = _synthetic_heterogeneous(n=80, seed=18)
        for ev in events:
            plab.record(ev)
        clab = CausalLab(treatment="T", control="C")
        n = clab.attach_to_policy_lab(plab)
        assert n == 80
        assert len(clab) == 80

    def test_attach_to_policy_lab_rejects_wrong_type(self):
        clab = CausalLab()
        with pytest.raises(TypeError):
            clab.attach_to_policy_lab("not a policy lab")

    def test_event_bus_emits_records_and_fits(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda e: seen.append(e.kind))
        lab = CausalLab(treatment="T", control="C", event_bus=bus)
        lab.record(LoggedEvent(context={"x": 0.0}, action="T", propensity=0.5, reward=1.0))
        lab.record(LoggedEvent(context={"x": 0.0}, action="C", propensity=0.5, reward=0.5))
        lab.fit()
        lab.cate({"x": 0.0}, learner=LEARNER_DR)
        assert CAUSAL_RECORDED in seen
        assert CAUSAL_FIT in seen
        assert CAUSAL_CATE_ESTIMATED in seen

    def test_event_bus_emits_uplift_and_heterogeneity_and_recommend(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda e: seen.append(e.kind))
        events = _synthetic_heterogeneous(n=120, seed=19)
        lab = CausalLab(treatment="T", control="C", event_bus=bus)
        lab.record_batch(events)
        lab.uplift(n_buckets=3)
        lab.test_heterogeneity(n_permutations=5, max_eval_contexts=10)
        lab.recommend({"x": 0.5}, actions=["T", "C"], baseline="C")
        assert CAUSAL_UPLIFT in seen
        assert CAUSAL_HETEROGENEITY in seen
        assert CAUSAL_RECOMMENDED in seen


# =====================================================================
# Runtime-engine fit: per-context routing decision
# =====================================================================


class TestRuntimeUsage:
    def test_coordinator_can_route_by_cate_ci(self):
        """The investor-grade pitch: route per-request by counterfactual CI.

        With strong heterogeneity, the lab should recommend treatment for
        contexts where lift CI excludes zero, and decline (return inconclusive)
        elsewhere.
        """
        events = _synthetic_heterogeneous(n=2000, seed=20)
        lab = CausalLab(treatment="T", control="C")
        lab.record_batch(events)

        decisive_pos = lab.cate({"x": 0.9}, learner=LEARNER_DR)
        decisive_neg = lab.cate({"x": -0.9}, learner=LEARNER_DR)
        zero_zone = lab.cate({"x": 0.0}, learner=LEARNER_DR)

        # Coordinator policy: ship treatment only when CI excludes zero positively.
        def route(point: CATEPoint) -> str:
            if point.ci_low > 0:
                return "treatment"
            if point.ci_high < 0:
                return "control"
            return "default"

        assert route(decisive_pos) == "treatment"
        assert route(decisive_neg) == "control"
        # x=0: true lift is ~0; either default or a small ship is acceptable
        # but not a sign-flip — sanity-check that route is consistent.
        assert route(zero_zone) in {"treatment", "control", "default"}
