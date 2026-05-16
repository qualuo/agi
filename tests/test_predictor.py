"""Tests for the Predictor runtime primitive — universal sequence prediction
via Context Tree Weighting (CTW)."""
from __future__ import annotations

import math
import random

import pytest

from agi.events import EventBus
from agi.predictor import (
    PREDICTOR_CLEARED,
    PREDICTOR_KNOWN_EVENTS,
    PREDICTOR_KNOWN_SELECTORS,
    PREDICTOR_OBSERVED,
    PREDICTOR_PREDICTED,
    PREDICTOR_REPORTED,
    PREDICTOR_SELECTED,
    PREDICTOR_STARTED,
    SELECT_BAYES_MEAN,
    SELECT_MAP,
    SELECT_MIN_LOG_LOSS,
    SELECT_SAMPLE,
    EProcess,
    EntropyRate,
    InvalidConfig,
    InvalidObservation,
    InvalidSymbol,
    MAPTree,
    Prediction,
    Predictor,
    PredictorError,
    PredictorReport,
    RedundancyBound,
    TreeNode,
    UnknownSelector,
    compress_binary_sequence,
    kl_divergence_bits,
)


# ---------------------------------------------------------------------
# Construction & config
# ---------------------------------------------------------------------


class TestConstruction:
    def test_default_factory(self) -> None:
        p = Predictor.create()
        assert p.alphabet_size == 2
        assert p.depth == 8
        assert p.n_observations == 0

    def test_invalid_alphabet(self) -> None:
        with pytest.raises(InvalidConfig):
            Predictor.create(alphabet_size=1)
        with pytest.raises(InvalidConfig):
            Predictor.create(alphabet_size=300)

    def test_invalid_depth(self) -> None:
        with pytest.raises(InvalidConfig):
            Predictor.create(depth=-1)
        with pytest.raises(InvalidConfig):
            Predictor.create(depth=100)

    def test_invalid_switching_rate(self) -> None:
        with pytest.raises(InvalidConfig):
            Predictor.create(switching_rate=0.0)
        with pytest.raises(InvalidConfig):
            Predictor.create(switching_rate=1.0)
        with pytest.raises(InvalidConfig):
            Predictor.create(switching_rate=-0.1)
        # valid
        Predictor.create(switching_rate=0.01)


# ---------------------------------------------------------------------
# Observation validation
# ---------------------------------------------------------------------


class TestObservation:
    def test_invalid_symbol_type(self) -> None:
        p = Predictor.create()
        with pytest.raises(InvalidSymbol):
            p.observe("0")  # type: ignore[arg-type]
        with pytest.raises(InvalidSymbol):
            p.observe(1.5)  # type: ignore[arg-type]

    def test_invalid_symbol_range(self) -> None:
        p = Predictor.create(alphabet_size=2)
        with pytest.raises(InvalidSymbol):
            p.observe(2)
        with pytest.raises(InvalidSymbol):
            p.observe(-1)

    def test_observe_advances_n(self) -> None:
        p = Predictor.create()
        for i in range(10):
            p.observe(i % 2)
        assert p.n_observations == 10

    def test_observe_many_advances_n(self) -> None:
        p = Predictor.create()
        p.observe_many([0, 1, 0, 1, 0])
        assert p.n_observations == 5

    def test_bool_treated_as_int(self) -> None:
        p = Predictor.create()
        # bool is a subclass of int — should be accepted
        p.observe(True)
        p.observe(False)
        assert p.n_observations == 2


# ---------------------------------------------------------------------
# Empty predictor / priors
# ---------------------------------------------------------------------


class TestEmpty:
    def test_empty_predict_uniform(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=4)
        pred = p.predict()
        assert pytest.approx(pred.probs[0], abs=1e-9) == 0.5
        assert pytest.approx(pred.probs[1], abs=1e-9) == 0.5

    def test_empty_code_length_zero(self) -> None:
        p = Predictor.create()
        assert abs(p.code_length_bits()) < 1e-9

    def test_empty_e_process_is_one(self) -> None:
        p = Predictor.create()
        e = p.e_process_vs_uniform()
        assert pytest.approx(e.e_value, abs=1e-9) == 1.0
        assert e.p_value_upper_bound == 1.0

    def test_empty_entropy_rate_is_log_a(self) -> None:
        p = Predictor.create(alphabet_size=4)
        er = p.entropy_rate_estimate()
        assert pytest.approx(er.average_log_loss_bits_per_symbol, abs=1e-9) == 2.0
        assert pytest.approx(er.predictive_entropy_bits, abs=1e-9) == 2.0


# ---------------------------------------------------------------------
# Predictive probabilities are a valid distribution
# ---------------------------------------------------------------------


class TestPredictiveValid:
    @pytest.mark.parametrize("A", [2, 3, 4, 8])
    @pytest.mark.parametrize("D", [0, 1, 3])
    def test_predict_sums_to_one(self, A: int, D: int) -> None:
        p = Predictor.create(alphabet_size=A, depth=D)
        random.seed(A * 17 + D)
        seq = [random.randint(0, A - 1) for _ in range(100)]
        p.observe_many(seq)
        pred = p.predict()
        assert pytest.approx(sum(pred.probs), abs=1e-9) == 1.0
        for q in pred.probs:
            assert q >= 0.0

    @pytest.mark.parametrize("A", [2, 4])
    def test_log_probs_consistent_with_probs(self, A: int) -> None:
        p = Predictor.create(alphabet_size=A, depth=3)
        random.seed(7)
        p.observe_many([random.randint(0, A - 1) for _ in range(50)])
        pred = p.predict()
        for prob, lp in zip(pred.probs, pred.log_probs):
            if prob > 0:
                assert pytest.approx(math.log(prob), abs=1e-9) == lp


# ---------------------------------------------------------------------
# CTW recovers structure: alternating sequence
# ---------------------------------------------------------------------


class TestStructureRecovery:
    def test_alternating_converges_to_zero_entropy(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=3)
        p.observe_many([0, 1] * 500)
        er = p.entropy_rate_estimate()
        # Truth: zero entropy.  CTW should approach it.
        assert er.average_log_loss_bits_per_symbol < 0.05

    def test_alternating_predicts_next_correctly(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=3)
        p.observe_many([0, 1] * 100)
        # last symbol is 1; next should be 0 with very high prob
        pred = p.predict()
        assert pred.argmax() == 0
        assert pred.probs[0] > 0.95

    def test_constant_zero_predicts_zero(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=4)
        p.observe_many([0] * 200)
        pred = p.predict()
        assert pred.probs[0] > 0.99
        # entropy rate near zero
        er = p.entropy_rate_estimate()
        assert er.average_log_loss_bits_per_symbol < 0.05

    def test_biased_recovers_entropy(self) -> None:
        random.seed(2026)
        p = Predictor.create(alphabet_size=2, depth=2)
        seq = [1 if random.random() < 0.7 else 0 for _ in range(5000)]
        p.observe_many(seq)
        er = p.entropy_rate_estimate()
        # H(0.7) = -0.7 log2(0.7) - 0.3 log2(0.3) ≈ 0.881 bits
        true_h = -0.7 * math.log2(0.7) - 0.3 * math.log2(0.3)
        assert abs(er.average_log_loss_bits_per_symbol - true_h) < 0.02

    def test_uniform_random_does_not_overfit(self) -> None:
        random.seed(7)
        p = Predictor.create(alphabet_size=2, depth=6)
        seq = [random.randint(0, 1) for _ in range(5000)]
        p.observe_many(seq)
        er = p.entropy_rate_estimate()
        # Truth: 1 bit/symbol. CTW pays small redundancy for the model class.
        assert 0.99 <= er.average_log_loss_bits_per_symbol <= 1.05


# ---------------------------------------------------------------------
# Markov source: CTW outperforms i.i.d.
# ---------------------------------------------------------------------


class TestMarkovOutperformsIID:
    def test_markov_beats_iid(self) -> None:
        random.seed(11)
        # Markov chain with strong dependence: P(x_{t+1}=x_t) = 0.95
        seq = [0]
        for _ in range(2000):
            prev = seq[-1]
            seq.append(prev if random.random() < 0.95 else 1 - prev)
        # Depth-1 predictor should beat depth-0 predictor (i.i.d.)
        p_iid = Predictor.create(alphabet_size=2, depth=0)
        p_markov = Predictor.create(alphabet_size=2, depth=2)
        p_iid.observe_many(seq)
        p_markov.observe_many(seq)
        assert p_markov.code_length_bits() < 0.7 * p_iid.code_length_bits()


# ---------------------------------------------------------------------
# E-process / hypothesis test
# ---------------------------------------------------------------------


class TestEProcess:
    def test_uniform_random_does_not_reject(self) -> None:
        random.seed(101)
        p = Predictor.create(alphabet_size=2, depth=4)
        p.observe_many([random.randint(0, 1) for _ in range(1000)])
        e = p.e_process_vs_uniform()
        # Under H_0, e should be O(1) — definitely not 100+.
        assert e.e_value < 10.0
        assert e.p_value_upper_bound == 1.0  # don't reject H_0

    def test_biased_rejects_uniform(self) -> None:
        random.seed(202)
        p = Predictor.create(alphabet_size=2, depth=3)
        p.observe_many([1 if random.random() < 0.8 else 0 for _ in range(500)])
        e = p.e_process_vs_uniform()
        # Massive evidence against uniform.
        assert e.e_value > 1e10
        assert e.p_value_upper_bound < 1e-9


# ---------------------------------------------------------------------
# Map tree (CTM)
# ---------------------------------------------------------------------


class TestMAPTree:
    def test_constant_yields_single_leaf(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=3)
        p.observe_many([0] * 100)
        mt = p.map_tree()
        # All the data sits in one path; the MAP tree may keep a few unseen
        # phantom leaves but the data-bearing structure should be sparse.
        leaves = mt.leaves()
        with_data = [leaf for leaf in leaves if sum(leaf.counts) > 0]
        # All zeros: only one context path actually accumulates data
        assert len(with_data) <= 2

    def test_map_tree_is_tree(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=3)
        p.observe_many([0, 1, 1, 0, 1, 0, 0, 1] * 30)
        mt = p.map_tree()
        assert isinstance(mt, MAPTree)
        assert mt.alphabet_size == 2
        assert mt.depth == 3
        # leaves count agrees with traversal
        assert mt.n_leaves == len(mt.leaves())

    def test_leaf_probs_are_a_distribution(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=2)
        p.observe_many([0, 1, 1, 0] * 50)
        mt = p.map_tree()
        for leaf in mt.leaves():
            probs = leaf.map_probs()
            assert pytest.approx(sum(probs), abs=1e-9) == 1.0


# ---------------------------------------------------------------------
# Sequence rollouts / log_loss are non-destructive
# ---------------------------------------------------------------------


class TestNonDestructive:
    def test_predict_sequence_does_not_mutate(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=3)
        p.observe_many([0, 1, 0, 1, 0, 1])
        cl_before = p.code_length_bits()
        n_before = p.n_observations
        fp_before = p.fingerprint
        rollout = p.predict_sequence(10)
        assert len(rollout) == 10
        assert p.n_observations == n_before
        assert abs(p.code_length_bits() - cl_before) < 1e-9
        assert p.fingerprint == fp_before

    def test_log_loss_does_not_mutate(self) -> None:
        p = Predictor.create(alphabet_size=3, depth=2)
        p.observe_many([0, 1, 2, 0, 1, 2, 0, 1, 2])
        cl_before = p.code_length_bits()
        loss = p.log_loss([0, 1, 2, 0])
        assert loss >= 0.0
        assert abs(p.code_length_bits() - cl_before) < 1e-9

    def test_most_likely_continuation_matches_argmax(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=2)
        p.observe_many([0, 1] * 50)
        cont = p.most_likely_continuation(6)
        # last observed was 1 → next 0 → next 1 → ...
        assert cont == [0, 1, 0, 1, 0, 1]


# ---------------------------------------------------------------------
# Universal code length: CTW ≤ KT + model redundancy
# ---------------------------------------------------------------------


class TestUniversalCode:
    def test_code_length_nonneg(self) -> None:
        p = Predictor.create()
        p.observe_many([0, 1, 1, 0, 1])
        assert p.code_length_bits() >= 0.0

    def test_code_length_bounded_by_n(self) -> None:
        # CTW on alphabet of size A produces code length ≤ n log2 A + O(1).
        p = Predictor.create(alphabet_size=2, depth=4)
        n = 500
        random.seed(0)
        seq = [random.randint(0, 1) for _ in range(n)]
        p.observe_many(seq)
        # For random data the code length is close to n bits; allow generous slack.
        assert p.code_length_bits() <= n + 50.0

    def test_redundancy_bound_positive(self) -> None:
        p = Predictor.create()
        p.observe_many([0, 1, 0, 1])
        rb = p.redundancy_bound(leaves=2)
        assert rb.parameter_redundancy_bits > 0
        assert rb.model_redundancy_bits >= 0
        assert rb.total_redundancy_bits == (
            rb.parameter_redundancy_bits + rb.model_redundancy_bits
        )


# ---------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------


class TestSelection:
    def test_map_selects_argmax(self) -> None:
        p = Predictor.create()
        p.observe_many([0] * 50)
        assert p.select(SELECT_MAP) == 0

    def test_bayes_mean_matches_map_for_zero_one_loss(self) -> None:
        p = Predictor.create()
        p.observe_many([1] * 30)
        assert p.select(SELECT_BAYES_MEAN) == p.select(SELECT_MAP)

    def test_min_log_loss_matches_map(self) -> None:
        p = Predictor.create()
        p.observe_many([1, 1, 1, 0, 1])
        assert p.select(SELECT_MIN_LOG_LOSS) == p.select(SELECT_MAP)

    def test_sample_is_in_alphabet(self) -> None:
        p = Predictor.create(alphabet_size=4, depth=2, seed=42)
        random.seed(0)
        p.observe_many([random.randint(0, 3) for _ in range(50)])
        for _ in range(20):
            s = p.select(SELECT_SAMPLE)
            assert 0 <= s < 4

    def test_sample_distribution_matches_predict(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=2, seed=99)
        p.observe_many([1] * 100 + [0] * 25)
        pred = p.predict()
        # Take many samples and check empirical frequency
        N = 4000
        counts = [0, 0]
        for _ in range(N):
            counts[p.select(SELECT_SAMPLE)] += 1
        emp = counts[1] / N
        assert abs(emp - pred.probs[1]) < 0.05

    def test_unknown_selector_raises(self) -> None:
        p = Predictor.create()
        with pytest.raises(UnknownSelector):
            p.select("not_a_rule")


# ---------------------------------------------------------------------
# Clear / reset
# ---------------------------------------------------------------------


class TestClear:
    def test_clear_resets_state(self) -> None:
        p = Predictor.create()
        p.observe_many([0, 1, 1, 0])
        p.clear()
        assert p.n_observations == 0
        pred = p.predict()
        assert pytest.approx(pred.probs[0], abs=1e-9) == 0.5


# ---------------------------------------------------------------------
# Event bus integration
# ---------------------------------------------------------------------


class TestEvents:
    def test_emits_started(self) -> None:
        bus = EventBus()
        events: list = []
        bus.subscribe(lambda e: events.append(e))
        Predictor.create(bus=bus)
        kinds = [e.kind for e in events]
        assert PREDICTOR_STARTED in kinds

    def test_emits_observed_predicted_selected(self) -> None:
        bus = EventBus()
        events: list = []
        bus.subscribe(lambda e: events.append(e))
        p = Predictor.create(bus=bus)
        p.observe(0)
        p.predict()
        p.select(SELECT_MAP)
        kinds = {e.kind for e in events}
        assert PREDICTOR_OBSERVED in kinds
        assert PREDICTOR_PREDICTED in kinds
        assert PREDICTOR_SELECTED in kinds

    def test_known_event_kinds(self) -> None:
        assert PREDICTOR_STARTED in PREDICTOR_KNOWN_EVENTS
        assert PREDICTOR_OBSERVED in PREDICTOR_KNOWN_EVENTS
        assert PREDICTOR_REPORTED in PREDICTOR_KNOWN_EVENTS
        assert PREDICTOR_SELECTED in PREDICTOR_KNOWN_EVENTS
        assert PREDICTOR_CLEARED in PREDICTOR_KNOWN_EVENTS
        assert PREDICTOR_PREDICTED in PREDICTOR_KNOWN_EVENTS

    def test_known_selectors(self) -> None:
        assert SELECT_MAP in PREDICTOR_KNOWN_SELECTORS
        assert SELECT_BAYES_MEAN in PREDICTOR_KNOWN_SELECTORS
        assert SELECT_MIN_LOG_LOSS in PREDICTOR_KNOWN_SELECTORS
        assert SELECT_SAMPLE in PREDICTOR_KNOWN_SELECTORS


# ---------------------------------------------------------------------
# Fingerprint chain
# ---------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_changes_on_observe(self) -> None:
        p = Predictor.create()
        fp0 = p.fingerprint
        p.observe(0)
        fp1 = p.fingerprint
        p.observe(1)
        fp2 = p.fingerprint
        assert fp0 != fp1
        assert fp1 != fp2

    def test_fingerprint_reproducible_across_runs(self) -> None:
        # Same construction args produce the same fingerprint chain.
        p1 = Predictor.create(seed=11)
        p2 = Predictor.create(seed=11)
        assert p1.fingerprint == p2.fingerprint
        for s in [0, 1, 1, 0]:
            p1.observe(s)
            p2.observe(s)
        assert p1.fingerprint == p2.fingerprint

    def test_fingerprint_diverges_on_different_data(self) -> None:
        p1 = Predictor.create(seed=11)
        p2 = Predictor.create(seed=11)
        p1.observe(0)
        p2.observe(1)
        assert p1.fingerprint != p2.fingerprint


# ---------------------------------------------------------------------
# Switching CTW
# ---------------------------------------------------------------------


class TestSwitching:
    def test_switching_predictor_works(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=4, switching_rate=0.01)
        # Two regimes: 200 of mostly 0, then 200 of mostly 1.
        random.seed(0)
        seq = []
        for _ in range(200):
            seq.append(0 if random.random() < 0.95 else 1)
        for _ in range(200):
            seq.append(1 if random.random() < 0.95 else 0)
        p.observe_many(seq)
        pred = p.predict()
        # Last regime is mostly 1 → predictor leans 1.
        assert pred.probs[1] > 0.7


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


class TestReport:
    def test_report_includes_everything(self) -> None:
        p = Predictor.create(alphabet_size=2, depth=3)
        p.observe_many([0, 1, 1, 0, 1, 0, 1, 1, 0])
        fp_before = p.fingerprint
        rep = p.report()
        assert isinstance(rep, PredictorReport)
        assert isinstance(rep.prediction, Prediction)
        assert isinstance(rep.redundancy_bound, RedundancyBound)
        assert isinstance(rep.entropy_rate, EntropyRate)
        assert isinstance(rep.e_process, EProcess)
        # report() snapshots the fingerprint before emitting its own event.
        assert rep.fingerprint == fp_before
        assert p.fingerprint != fp_before  # report event advanced it
        assert rep.n_observations == 9
        assert rep.alphabet_size == 2
        assert rep.depth == 3
        assert rep.map_tree_leaves >= 1


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


class TestHelpers:
    def test_compress_binary_sequence(self) -> None:
        bits, p = compress_binary_sequence([0, 1] * 100, depth=4)
        assert bits < 50  # alternating sequence is highly compressible
        assert p.n_observations == 200

    def test_kl_divergence(self) -> None:
        kl = kl_divergence_bits([0.5, 0.5], [0.5, 0.5])
        assert pytest.approx(kl, abs=1e-9) == 0.0
        kl = kl_divergence_bits([1.0, 0.0], [0.5, 0.5])
        assert pytest.approx(kl, abs=1e-9) == 1.0
        kl = kl_divergence_bits([0.5, 0.5], [1.0, 0.0])
        assert math.isinf(kl)


# ---------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------


class TestConcurrency:
    def test_thread_safe_observe(self) -> None:
        import threading

        p = Predictor.create(alphabet_size=2, depth=3)

        def feed(seed: int) -> None:
            rng = random.Random(seed)
            for _ in range(100):
                p.observe(rng.randint(0, 1))

        ts = [threading.Thread(target=feed, args=(i,)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        # All observations recorded
        assert p.n_observations == 400
        # Predict still works
        pred = p.predict()
        assert abs(sum(pred.probs) - 1.0) < 1e-9
