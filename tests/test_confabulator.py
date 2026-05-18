"""Tests for the Confabulator hallucination-certification primitive."""
from __future__ import annotations

import json
import math

import pytest

from agi.events import EventBus
from agi.confabulator import (
    CONFAB_AUDITED,
    CONFAB_CALIBRATED,
    CONFAB_CERTIFIED,
    CONFAB_GATED,
    CONFAB_REPORTED,
    CONFAB_RESET,
    CONFAB_SCORED,
    CONFAB_STARTED,
    CONFAB_SUBMITTED,
    KNOWN_MODES,
    KNOWN_RECOMMENDATIONS,
    KNOWN_VERDICTS,
    MODE_COMBINED,
    MODE_LEXICAL_ENTROPY,
    MODE_PREDICTIVE_ENTROPY,
    MODE_SELFCHECK,
    MODE_SEMANTIC_ENTROPY,
    NO_LOGPROB,
    REC_ESCALATE,
    REC_QUARANTINE,
    REC_REGENERATE,
    REC_RESTRICT,
    REC_TRUST,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
    VERDICT_PASS,
    VERDICT_WARN,
    AuditReport,
    Confabulator,
    ConfabulatorCertificate,
    ConfabulatorConfig,
    ConfabulatorError,
    ConfabulatorReport,
    InvalidConfig,
    InvalidSample,
    InvalidTrial,
    NotCalibrated,
    NotEnoughTrials,
    Sample,
    ThresholdReport,
    Trial,
    TrialReport,
    UnknownMode,
    confabulator,
    confabulator_with_oracle,
    exact_match_oracle,
    jaccard_oracle,
    synthetic_control_trials,
    synthetic_trials,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfabulatorConfig:
    def test_default_config(self):
        cfg = ConfabulatorConfig()
        assert cfg.sample_budget_k == 5
        assert MODE_SEMANTIC_ENTROPY in cfg.modes
        assert MODE_COMBINED not in cfg.modes
        assert 0.0 < cfg.budget_p0 < 1.0

    def test_empty_modes_rejected(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(modes=())

    def test_combined_mode_rejected(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(modes=(MODE_COMBINED,))

    def test_unknown_mode_rejected(self):
        with pytest.raises(UnknownMode):
            ConfabulatorConfig(modes=("blarg",))

    def test_small_k_rejected(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(sample_budget_k=1)

    @pytest.mark.parametrize("bad", [(-0.1, 0.2, 0.2, 0.2),
                                     (float("inf"), 0, 0, 0)])
    def test_bad_weights_rejected(self, bad):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(weights=bad)

    def test_zero_weights_rejected(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(weights=(0.0, 0.0, 0.0, 0.0))

    def test_wrong_weight_length(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(weights=(1.0, 0.5, 0.5))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
    def test_bad_budget(self, bad):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(budget_p0=bad)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
    def test_bad_alpha(self, bad):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(alpha=bad)

    def test_negative_bootstrap(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(bootstrap_b=-1)

    def test_bad_warn_factor(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(warn_factor=0.0)
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(warn_factor=1.1)

    def test_bad_prior(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(prior_a=0.0)
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(prior_b=-1.0)

    def test_small_max_clusters(self):
        with pytest.raises(InvalidConfig):
            ConfabulatorConfig(max_clusters=1)


# ---------------------------------------------------------------------------
# Sample / Trial validation
# ---------------------------------------------------------------------------


class TestSample:
    def test_minimal_sample(self):
        s = Sample(text="hello")
        assert s.text == "hello"
        assert math.isnan(s.mean_logprob)
        assert s.n_tokens == 0

    def test_empty_text_rejected(self):
        with pytest.raises(InvalidSample):
            Sample(text="   ")

    def test_positive_logprob_rejected(self):
        with pytest.raises(InvalidSample):
            Sample(text="x", mean_logprob=0.5)

    def test_infinite_logprob_rejected(self):
        with pytest.raises(InvalidSample):
            Sample(text="x", mean_logprob=float("-inf"))

    def test_negative_tokens_rejected(self):
        with pytest.raises(InvalidSample):
            Sample(text="x", n_tokens=-1)

    def test_non_string_text(self):
        with pytest.raises(InvalidSample):
            Sample(text=42)  # type: ignore[arg-type]


class TestTrial:
    def _samples(self, n=3):
        return tuple(Sample(text=f"answer_{i}") for i in range(n))

    def test_minimal_trial(self):
        t = Trial(prompt_id="p", samples=self._samples())
        assert t.prompt_id == "p"
        assert len(t.samples) == 3
        assert t.truth is None

    def test_empty_prompt_id_rejected(self):
        with pytest.raises(InvalidTrial):
            Trial(prompt_id="", samples=self._samples())

    def test_few_samples_rejected(self):
        with pytest.raises(InvalidTrial):
            Trial(prompt_id="p", samples=(Sample(text="a"),))

    def test_non_sample_in_samples(self):
        with pytest.raises(InvalidTrial):
            Trial(prompt_id="p", samples=("a", "b"))  # type: ignore[arg-type]

    def test_truth_must_be_bool_or_none(self):
        with pytest.raises(InvalidTrial):
            Trial(prompt_id="p", samples=self._samples(), truth=1)  # type: ignore[arg-type]

    def test_control_with_truth_rejected(self):
        with pytest.raises(InvalidTrial):
            Trial(prompt_id="p", samples=self._samples(),
                  truth=True, control=True)

    def test_reference_index_out_of_range(self):
        with pytest.raises(InvalidTrial):
            Trial(prompt_id="p", samples=self._samples(),
                  reference_chosen_index=99)


# ---------------------------------------------------------------------------
# Equivalence oracles
# ---------------------------------------------------------------------------


class TestOracles:
    def test_exact_match_normalises(self):
        assert exact_match_oracle("Hello, World!", "hello world")
        assert not exact_match_oracle("hello", "goodbye")

    def test_jaccard_threshold_validation(self):
        with pytest.raises(InvalidConfig):
            jaccard_oracle(0.0)
        with pytest.raises(InvalidConfig):
            jaccard_oracle(1.1)

    def test_jaccard_pair(self):
        o = jaccard_oracle(0.5)
        assert o("the cat sat", "cat sat the")  # identical token sets
        assert not o("cats are great", "dogs are great too too")

    def test_jaccard_empty_strings(self):
        o = jaccard_oracle(0.5)
        assert o("", "")
        assert not o("hello", "")


# ---------------------------------------------------------------------------
# Lifecycle: submit / fit / audit / certify / gate
# ---------------------------------------------------------------------------


def _make_samples(label: str, k: int = 5,
                  logprob: float = -0.2) -> tuple[Sample, ...]:
    return tuple(Sample(text=label, mean_logprob=logprob, n_tokens=1)
                 for _ in range(k))


def _make_dispersed(k: int = 5, logprob: float = -1.5) -> tuple[Sample, ...]:
    return tuple(Sample(text=f"answer_{i}", mean_logprob=logprob, n_tokens=1)
                 for i in range(k))


class TestSubmit:
    def test_basic_submit_returns_trial_report(self):
        c = Confabulator()
        rep = c.submit("p1", _make_samples("alpha"))
        assert isinstance(rep, TrialReport)
        assert rep.prompt_id == "p1"
        assert rep.n_samples == 5
        assert rep.n_clusters == 1
        assert rep.semantic_entropy == pytest.approx(0.0, abs=1e-12)
        assert rep.verdict == VERDICT_INCONCLUSIVE

    def test_dispersed_has_higher_semantic_entropy(self):
        c = Confabulator()
        a = c.submit("a", _make_samples("alpha"))
        b = c.submit("b", _make_dispersed())
        assert b.semantic_entropy > a.semantic_entropy
        assert b.n_clusters > a.n_clusters

    def test_predictive_entropy_nan_when_no_logprobs(self):
        c = Confabulator()
        samples = tuple(Sample(text="x") for _ in range(3))
        rep = c.submit("p", samples)
        assert math.isnan(rep.predictive_entropy)
        # combined still finite — falls back to other detectors.
        assert not math.isnan(rep.combined_score)

    def test_submit_increments_n_trials(self):
        c = Confabulator()
        for i in range(3):
            c.submit(f"p{i}", _make_samples("x"))
        assert c.n_trials == 3

    def test_single_sample_rejected(self):
        c = Confabulator()
        with pytest.raises(InvalidTrial):
            c.submit("p", (Sample(text="x"),))

    def test_passing_a_lone_sample_rejected(self):
        c = Confabulator()
        with pytest.raises(InvalidTrial):
            c.submit("p", Sample(text="x"))  # type: ignore[arg-type]

    def test_metadata_round_trips(self):
        c = Confabulator()
        rep = c.submit("p", _make_samples("x"), metadata={"k": "v"})
        assert rep.metadata == {"k": "v"}


class TestFitThreshold:
    def test_fit_threshold_basic(self):
        c = confabulator(seed=0)
        trials = synthetic_trials(n_truthful=20, n_hallucinated=20,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        thr = c.fit_threshold()
        assert isinstance(thr, ThresholdReport)
        assert 0.0 < thr.threshold
        assert 0.5 < thr.auroc <= 1.0
        assert thr.auroc_lower <= thr.auroc <= thr.auroc_upper

    def test_fit_threshold_few_labelled_rejected(self):
        c = confabulator(seed=0)
        c.submit("p", _make_samples("x"), truth=True)
        with pytest.raises(NotEnoughTrials):
            c.fit_threshold()

    def test_fit_threshold_one_class_rejected(self):
        c = confabulator(seed=0)
        for i in range(5):
            c.submit(f"p{i}", _make_samples("x"), truth=True)
        with pytest.raises(NotEnoughTrials):
            c.fit_threshold()

    def test_target_fpr_overrides_youden(self):
        c = confabulator(seed=0)
        trials = synthetic_trials(n_truthful=20, n_hallucinated=20,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        thr_fpr = c.fit_threshold(target_fpr=0.05)
        thr_y = c.fit_threshold()
        # The FPR-constrained threshold should be at least as high
        # as the Youden one (more conservative).
        assert thr_fpr.fpr_at_threshold <= 0.05 + 1e-9 or \
            thr_fpr.fpr_at_threshold == thr_y.fpr_at_threshold

    def test_target_tpr_and_fpr_both_supplied_rejected(self):
        c = confabulator(seed=0)
        trials = synthetic_trials(n_truthful=5, n_hallucinated=5,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        with pytest.raises(InvalidConfig):
            c.fit_threshold(target_fpr=0.1, target_tpr=0.9)

    def test_per_trial_verdicts_refresh(self):
        c = confabulator(seed=0)
        trials = synthetic_trials(n_truthful=10, n_hallucinated=10,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        # Verdicts inconclusive before fit.
        before = [r.verdict for r in c.reports]
        assert all(v == VERDICT_INCONCLUSIVE for v in before)
        c.fit_threshold()
        after = [r.verdict for r in c.reports]
        assert all(v in (VERDICT_PASS, VERDICT_FAIL) for v in after)


class TestAudit:
    def test_audit_empty(self):
        c = Confabulator()
        rep = c.audit()
        assert rep.n_trials_labelled == 0
        assert rep.running_rate == 0.0
        assert not rep.rejected_h0

    def test_audit_rejects_high_rate(self):
        c = confabulator(seed=0, budget_p0=0.05)
        trials = synthetic_trials(n_truthful=5, n_hallucinated=20,
                                  k=5, seed=1)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        audit = c.audit()
        assert audit.running_rate > 0.05
        assert audit.rejected_h0
        assert audit.e_value > 1.0 / 0.05

    def test_audit_clopper_pearson_bounds(self):
        c = confabulator(seed=0)
        for _ in range(10):
            c.submit("t", _make_samples("x"), truth=True)
        audit = c.audit()
        assert audit.n_trials_labelled == 10
        assert audit.n_hallucinations == 0
        # k = 0 lower bound is 0.
        assert audit.rate_lower_clopper_pearson == 0.0
        # And upper is positive.
        assert audit.rate_upper_clopper_pearson > 0.0

    def test_audit_excludes_control_trials(self):
        c = confabulator(seed=0)
        c.submit("p1", _make_samples("x"), truth=True)
        c.submit("c1", _make_samples("y"), control=True)
        audit = c.audit()
        assert audit.n_trials_labelled == 1


class TestCertify:
    def test_certify_with_no_data(self):
        c = Confabulator()
        cert = c.certify()
        assert cert.n_trials == 0
        assert cert.verdict == VERDICT_INCONCLUSIVE
        assert cert.recommendation == REC_RESTRICT

    def test_certify_truthful_pool_passes(self):
        c = confabulator(seed=0, budget_p0=0.10)
        for i in range(40):
            c.submit(f"p{i}", _make_samples(f"alpha"), truth=True)
        cert = c.certify()
        assert cert.hallucination_rate == 0.0
        assert cert.recommendation in (REC_TRUST, REC_RESTRICT)
        assert cert.verdict == VERDICT_PASS

    def test_certify_hallucination_pool_fails(self):
        c = confabulator(seed=0, budget_p0=0.10)
        trials = synthetic_trials(n_truthful=2, n_hallucinated=30,
                                  k=5, seed=2)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        cert = c.certify()
        assert cert.verdict == VERDICT_FAIL
        assert cert.recommendation == REC_ESCALATE

    def test_certify_delta_override(self):
        c = Confabulator()
        # Empty data — certify(delta=0.5) still safe.
        cert = c.certify(delta=0.5)
        assert cert.verdict == VERDICT_INCONCLUSIVE

    def test_certify_invalid_delta(self):
        c = Confabulator()
        with pytest.raises(InvalidConfig):
            c.certify(delta=0.0)
        with pytest.raises(InvalidConfig):
            c.certify(delta=1.5)


class TestGate:
    def test_gate_without_fit_raises(self):
        c = Confabulator()
        with pytest.raises(NotCalibrated):
            c.gate("p", _make_samples("x"))

    def test_gate_trust_for_low_score(self):
        c = confabulator(seed=0)
        # Calibrate.
        trials = synthetic_trials(n_truthful=20, n_hallucinated=20,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        c.fit_threshold()
        rep, rec = c.gate("live_truthful", _make_samples("alpha"))
        assert rep.combined_score < c.threshold.threshold
        assert rec == REC_TRUST

    def test_gate_recommendation_escalate_for_high_score(self):
        c = confabulator(seed=0)
        trials = synthetic_trials(n_truthful=20, n_hallucinated=20,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        c.fit_threshold()
        # Highly dispersed → high score.
        samples = tuple(Sample(text=f"distinct_{i}") for i in range(7))
        rep, rec = c.gate("live_halluc", samples)
        assert rep.combined_score > c.threshold.threshold
        assert rec in (REC_REGENERATE, REC_RESTRICT, REC_ESCALATE)

    def test_gate_quarantine_on_nan_score(self):
        """A degenerate trial whose combined score is NaN routes to quarantine.

        Forcing NaN: configure modes with only PE weights non-zero, and
        send samples with no log-probabilities — PE comes out NaN, the
        other modes are zero-weighted, and combined_score collapses
        to NaN.
        """
        c = Confabulator(ConfabulatorConfig(
            seed=0,
            weights=(0.0, 0.0, 1.0, 0.0),
        ))
        # First, push enough labelled trials WITH log-probs so the
        # threshold fits.
        trials = synthetic_trials(n_truthful=10, n_hallucinated=10,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        c.fit_threshold()
        # Now gate without log-probs.
        samples = tuple(Sample(text="x") for _ in range(3))
        rep, rec = c.gate("live_nan", samples)
        assert math.isnan(rep.combined_score)
        assert rec == REC_QUARANTINE


class TestReportRoundTrip:
    def test_report_is_json_serialisable(self):
        c = confabulator(seed=0)
        trials = synthetic_trials(n_truthful=5, n_hallucinated=5,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        c.fit_threshold()
        out = c.report()
        s = json.dumps(out.to_dict())
        assert isinstance(s, str)
        loaded = json.loads(s)
        assert loaded["n_trials"] == 10
        assert loaded["threshold"]["auroc"] == c.threshold.auroc


# ---------------------------------------------------------------------------
# Event bus integration
# ---------------------------------------------------------------------------


class TestEvents:
    def test_events_fire(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        c = Confabulator(bus=bus)
        c.submit("p", _make_samples("x"), truth=True)
        c.submit("q", _make_samples("y"), truth=False)
        c.submit("r", _make_samples("z"), truth=False)
        c.fit_threshold()
        c.audit()
        c.certify()
        c.reset()
        for kind in [CONFAB_STARTED, CONFAB_SUBMITTED, CONFAB_SCORED,
                     CONFAB_CALIBRATED, CONFAB_AUDITED, CONFAB_CERTIFIED,
                     CONFAB_RESET]:
            assert kind in seen

    def test_gate_event_fires(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        c = Confabulator(bus=bus)
        trials = synthetic_trials(n_truthful=5, n_hallucinated=5,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        c.fit_threshold()
        c.gate("live", _make_samples("alpha"))
        assert CONFAB_GATED in seen


# ---------------------------------------------------------------------------
# Determinism / replay
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_fingerprint(self):
        def run() -> str:
            c = confabulator(seed=42)
            trials = synthetic_trials(n_truthful=10, n_hallucinated=10,
                                      k=5, seed=0)
            for t in trials:
                c.submit(t.prompt_id, t.samples, truth=t.truth)
            c.fit_threshold()
            return c.certify().fingerprint_hash
        assert run() == run()

    def test_different_seeds_differ(self):
        def run(seed: int) -> str:
            c = confabulator(seed=seed)
            trials = synthetic_trials(n_truthful=10, n_hallucinated=10,
                                      k=5, seed=0)
            for t in trials:
                c.submit(t.prompt_id, t.samples, truth=t.truth)
            c.fit_threshold()
            return c.certify().fingerprint_hash
        a = run(1)
        b = run(2)
        # Bootstrap-CI depends on seed so the fingerprint chain
        # differs.
        assert a != b


# ---------------------------------------------------------------------------
# Control-pool noise floor
# ---------------------------------------------------------------------------


class TestControl:
    def test_control_singleton_rate_exact_paraphrase(self):
        c = Confabulator()
        # Identical samples — every control collapses.
        for i in range(10):
            samples = tuple(Sample(text="alpha") for _ in range(4))
            c.submit(f"c{i}", samples, truth=None, control=True)
        assert c.control_singleton_rate() == pytest.approx(1.0)

    def test_control_singleton_rate_none_when_empty(self):
        c = Confabulator()
        assert c.control_singleton_rate() is None

    def test_control_does_not_enter_audit(self):
        c = confabulator(seed=0)
        c.submit("ctrl", _make_samples("x"), control=True)
        audit = c.audit()
        assert audit.n_trials_labelled == 0


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


class TestFactoryHelpers:
    def test_confabulator_sugar(self):
        c = confabulator(seed=7, budget_p0=0.2)
        assert c.config.seed == 7
        assert c.config.budget_p0 == 0.2

    def test_confabulator_with_oracle_uses_supplied(self):
        # An oracle that says everything is equivalent → 1 cluster
        # always.
        c = confabulator_with_oracle(lambda a, b: True, seed=0)
        rep = c.submit("p", _make_dispersed())
        assert rep.n_clusters == 1
        assert rep.semantic_entropy == pytest.approx(0.0, abs=1e-12)

    def test_synthetic_trials_have_labels(self):
        ts = synthetic_trials(n_truthful=3, n_hallucinated=4, seed=0)
        assert sum(1 for t in ts if t.truth is True) == 3
        assert sum(1 for t in ts if t.truth is False) == 4

    def test_synthetic_control_trials(self):
        ts = synthetic_control_trials(n=5, seed=0)
        assert len(ts) == 5
        assert all(t.control for t in ts)
        assert all(t.truth is None for t in ts)


# ---------------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------------


class TestMath:
    def test_clopper_pearson_consistent(self):
        from agi.confabulator import _clopper_pearson
        lo, hi = _clopper_pearson(0, 10, 0.05)
        assert lo == 0.0
        assert 0.0 < hi < 1.0
        lo, hi = _clopper_pearson(10, 10, 0.05)
        assert hi == 1.0
        assert 0.0 < lo < 1.0
        lo, hi = _clopper_pearson(5, 10, 0.05)
        assert 0.0 < lo < 0.5 < hi < 1.0

    def test_clopper_pearson_rejects_bad_k(self):
        from agi.confabulator import _clopper_pearson
        with pytest.raises(ConfabulatorError):
            _clopper_pearson(11, 10, 0.05)

    def test_shannon_entropy_uniform_log_n(self):
        from agi.confabulator import _shannon_entropy
        # Uniform on 4 bins should have entropy log(4).
        h = _shannon_entropy([1, 1, 1, 1])
        assert h == pytest.approx(math.log(4))

    def test_shannon_entropy_zero_on_singleton(self):
        from agi.confabulator import _shannon_entropy
        assert _shannon_entropy([7]) == 0.0
        assert _shannon_entropy([0, 0, 0]) == 0.0

    def test_auroc_perfect_separation(self):
        from agi.confabulator import _auroc
        scores = [0.1, 0.2, 0.9, 0.95]
        labels = [False, False, True, True]
        assert _auroc(scores, labels) == 1.0

    def test_auroc_completely_separated_wrong_way(self):
        from agi.confabulator import _auroc
        scores = [0.9, 0.95, 0.1, 0.2]
        labels = [False, False, True, True]
        assert _auroc(scores, labels) == 0.0

    def test_auroc_random_is_half(self):
        from agi.confabulator import _auroc
        # Symmetric — every positive matched by an equal negative.
        scores = [0.5] * 10
        labels = [True, False] * 5
        assert _auroc(scores, labels) == 0.5

    def test_eprocess_under_h0_stays_low_in_expectation(self):
        from agi.confabulator import _eprocess_one_proportion
        # Under H_0: p == p0 == 0.1, e-value is on average 1.
        # Single trial with no events ⇒ e_value < 1.
        e = _eprocess_one_proportion([False] * 10, p0=0.1,
                                     prior_a=1.0, prior_b=1.0)
        assert e < 1.5  # at most a small constant under H_0 small-n

    def test_eprocess_above_h0_grows(self):
        from agi.confabulator import _eprocess_one_proportion
        e = _eprocess_one_proportion([True] * 10 + [False] * 10,
                                     p0=0.05,
                                     prior_a=1.0, prior_b=1.0)
        assert e > 1.0 / 0.05  # rejects at alpha=0.05

    def test_holm_smallest_is_monotone(self):
        from agi.confabulator import _holm_smallest
        assert _holm_smallest([0.01, 0.04, 0.06], 0.05) == pytest.approx(0.03)
        assert _holm_smallest([], 0.05) == 1.0

    def test_miller_madow_correction(self):
        from agi.confabulator import _miller_madow_correction
        assert _miller_madow_correction(1, 10) == 0.0  # singleton: m-1 = 0
        assert _miller_madow_correction(5, 10) == pytest.approx((5 - 1) / 20)


# ---------------------------------------------------------------------------
# Real-shape end-to-end behaviour
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_classic_workflow(self):
        """Calibrate against a labelled pool, then gate live traffic."""
        c = confabulator(seed=0, budget_p0=0.15)
        # 1. Calibrate
        cal = synthetic_trials(n_truthful=25, n_hallucinated=25,
                               k=5, seed=0)
        for t in cal:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        thr = c.fit_threshold()
        assert thr.auroc > 0.65  # synthetic data is hard but informative
        # 2. Live gating
        live_truthful_recs: list[str] = []
        live_halluc_recs: list[str] = []
        live = synthetic_trials(n_truthful=20, n_hallucinated=20,
                                k=5, seed=99)
        for t in live:
            # Strip label from the live call but record what it should
            # have been.
            _, rec = c.gate(t.prompt_id, t.samples)
            (live_truthful_recs if t.truth else live_halluc_recs).append(rec)
        # The vast majority of truthful trials should land at TRUST or
        # REGENERATE, not ESCALATE.
        assert sum(1 for r in live_truthful_recs if r == REC_TRUST) >= 5
        # And some halluc trials should escalate or restrict.
        bad_recs = sum(1 for r in live_halluc_recs
                       if r in (REC_RESTRICT, REC_ESCALATE, REC_REGENERATE))
        assert bad_recs >= 5

    def test_report_roundtrip(self):
        c = confabulator(seed=0)
        trials = synthetic_trials(n_truthful=10, n_hallucinated=10,
                                  k=5, seed=0)
        for t in trials:
            c.submit(t.prompt_id, t.samples, truth=t.truth)
        c.fit_threshold()
        report = c.report()
        roundtrip = json.loads(json.dumps(report.to_dict()))
        assert roundtrip["n_trials"] == 20
        assert roundtrip["threshold"]["threshold"] == c.threshold.threshold


# ---------------------------------------------------------------------------
# Vocabulary checks
# ---------------------------------------------------------------------------


class TestVocab:
    def test_known_verdicts(self):
        for v in [VERDICT_PASS, VERDICT_WARN, VERDICT_FAIL,
                  VERDICT_INCONCLUSIVE]:
            assert v in KNOWN_VERDICTS

    def test_known_recommendations(self):
        for r in [REC_TRUST, REC_RESTRICT, REC_QUARANTINE,
                  REC_REGENERATE, REC_ESCALATE]:
            assert r in KNOWN_RECOMMENDATIONS

    def test_known_modes(self):
        for m in [MODE_SEMANTIC_ENTROPY, MODE_LEXICAL_ENTROPY,
                  MODE_PREDICTIVE_ENTROPY, MODE_SELFCHECK]:
            assert m in KNOWN_MODES

    def test_no_logprob_is_nan(self):
        assert math.isnan(NO_LOGPROB)
