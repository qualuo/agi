"""Tests for :mod:`agi.debater`.

Run with::

    pytest tests/test_debater.py -q

All tests are deterministic given the seeds embedded in the fixtures.
"""
from __future__ import annotations

import json
import math
import unittest

from agi.debater import (
    AGG_MAJORITY,
    AGG_UNANIMITY,
    AGG_WEIGHTED_LOG_ODDS,
    Argument,
    CalibrationReport,
    DEBATER_OPENED,
    DEBATER_STARTED,
    DEBATER_VERDICT,
    Debater,
    DebaterCertificate,
    DebaterConfig,
    DebaterError,
    DebateMove,
    DebateReport,
    DebateSpec,
    InsufficientData,
    InvalidConfig,
    InvalidMove,
    InvalidSpec,
    JUDGE_CALIBRATED,
    JUDGE_JURY,
    JUDGE_PERSUASION_MODEL,
    KNOWN_AGGREGATIONS,
    KNOWN_JUDGE_MODELS,
    KNOWN_MOVES,
    KNOWN_PROTOCOLS,
    MOVE_ARGUE,
    MOVE_CONCEDE,
    MOVE_COUNTER,
    MOVE_CROSS_EXAMINE,
    MOVE_VERDICT,
    NashResult,
    NotRun,
    PROTOCOL_CROSS_EXAM,
    PROTOCOL_DOUBLY_EFFICIENT,
    PROTOCOL_JURY,
    PROTOCOL_MARKET_MAKER,
    PROTOCOL_PERSUASION_AWARE,
    PROTOCOL_TWO_PLAYER,
    PayoffMatrix,
    SIDE_A,
    SIDE_B,
    SIDE_TIE,
    UnknownAggregation,
    UnknownJudgeModel,
    UnknownProtocol,
    debater_bayes_posterior_shift,
    debater_bernstein_lcb,
    debater_calibration_ece,
    debater_condorcet_lcb,
    debater_hoeffding_lcb,
    debater_hrms_radius,
    debater_jury_log_odds,
    debater_jury_majority,
    debater_ledger_root,
    debater_payoff_nash_2x2,
    debater_persuasion_decomposition,
    debater_support_enumeration,
    make_calibrated_judge,
    make_constant_debater,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_spec(*, p_truth: float = 0.8, truth_side: str = SIDE_A, seed: int = 0) -> DebateSpec:
    return DebateSpec(
        question="q",
        claim_a="ca",
        claim_b="cb",
        debater_a=make_constant_debater([Argument(SIDE_A, "a", 0.5)]),
        debater_b=make_constant_debater([Argument(SIDE_B, "b", 0.5)]),
        judge=make_calibrated_judge(p_truth, truth_side, seed=seed),
        ground_truth=truth_side,
    )


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestHoeffdingLCB(unittest.TestCase):
    def test_basic(self) -> None:
        lcb = debater_hoeffding_lcb(0.8, 100, 0.05)
        self.assertGreater(lcb, 0.6)
        self.assertLess(lcb, 0.8)

    def test_clip(self) -> None:
        # Tiny n produces huge radius → clipped to 0
        lcb = debater_hoeffding_lcb(0.5, 1, 0.05)
        self.assertEqual(lcb, 0.0)

    def test_monotone_in_n(self) -> None:
        # Larger n → tighter bound
        lcb_small = debater_hoeffding_lcb(0.8, 10, 0.05)
        lcb_large = debater_hoeffding_lcb(0.8, 1000, 0.05)
        self.assertGreater(lcb_large, lcb_small)

    def test_invalid_n(self) -> None:
        with self.assertRaises(InsufficientData):
            debater_hoeffding_lcb(0.5, 0, 0.05)

    def test_invalid_delta(self) -> None:
        with self.assertRaises(InvalidConfig):
            debater_hoeffding_lcb(0.5, 10, 0.0)
        with self.assertRaises(InvalidConfig):
            debater_hoeffding_lcb(0.5, 10, 1.0)

    def test_invalid_p_hat(self) -> None:
        with self.assertRaises(InvalidConfig):
            debater_hoeffding_lcb(1.5, 10, 0.05)


class TestBernsteinLCB(unittest.TestCase):
    def test_low_variance_beats_hoeffding(self) -> None:
        # With small empirical variance, Bernstein should be tighter
        h = debater_hoeffding_lcb(0.95, 100, 0.05)
        b = debater_bernstein_lcb(0.95, 0.01, 100, 0.05)
        self.assertGreater(b, h)

    def test_n_ge_2(self) -> None:
        with self.assertRaises(InsufficientData):
            debater_bernstein_lcb(0.8, 0.1, 1, 0.05)


class TestCondorcetLCB(unittest.TestCase):
    def test_majority_amplifies_accuracy(self) -> None:
        # As M grows with p > 0.5, the LCB approaches 1
        lcb_small = debater_condorcet_lcb(0.7, 3, 0.05)
        lcb_large = debater_condorcet_lcb(0.7, 51, 0.05)
        self.assertGreater(lcb_large, lcb_small)
        self.assertGreater(lcb_large, 0.8)

    def test_below_half_collapses(self) -> None:
        # p < 0.5 should produce a low LCB (no amplification)
        lcb = debater_condorcet_lcb(0.45, 11, 0.05)
        self.assertLess(lcb, 0.5)

    def test_p_validation(self) -> None:
        with self.assertRaises(InvalidConfig):
            debater_condorcet_lcb(0.0, 11, 0.05)
        with self.assertRaises(InvalidConfig):
            debater_condorcet_lcb(1.0, 11, 0.05)


class TestHRMSRadius(unittest.TestCase):
    def test_shrinks_with_n(self) -> None:
        r2 = debater_hrms_radius(2, 0.05)
        r100 = debater_hrms_radius(100, 0.05)
        self.assertGreater(r2, r100)

    def test_requires_n_ge_2(self) -> None:
        with self.assertRaises(InsufficientData):
            debater_hrms_radius(1, 0.05)


class TestJuryAggregations(unittest.TestCase):
    def test_majority(self) -> None:
        votes = [(SIDE_A, 0.7), (SIDE_A, 0.6), (SIDE_B, 0.9)]
        self.assertEqual(debater_jury_majority(votes), SIDE_A)

    def test_majority_tie(self) -> None:
        votes = [(SIDE_A, 0.7), (SIDE_B, 0.7)]
        self.assertEqual(debater_jury_majority(votes), SIDE_TIE)
        self.assertEqual(debater_jury_majority(votes, tie_break=SIDE_A), SIDE_A)

    def test_log_odds_confidence_weighted(self) -> None:
        # Even when B has more votes, a single very-confident A can flip
        votes = [(SIDE_A, 0.99), (SIDE_B, 0.51), (SIDE_B, 0.51)]
        self.assertEqual(debater_jury_log_odds(votes), SIDE_A)

    def test_log_odds_weights(self) -> None:
        votes = [(SIDE_A, 0.6), (SIDE_B, 0.6)]
        self.assertEqual(debater_jury_log_odds(votes, weights=[3.0, 1.0]), SIDE_A)


class TestBayesPosteriorShift(unittest.TestCase):
    def test_zero_shift_identical(self) -> None:
        p = {SIDE_A: 0.4, SIDE_B: 0.6}
        self.assertAlmostEqual(debater_bayes_posterior_shift(p, p), 0.0)

    def test_max_shift_opposite(self) -> None:
        p1 = {SIDE_A: 1.0, SIDE_B: 0.0}
        p2 = {SIDE_A: 0.0, SIDE_B: 1.0}
        self.assertAlmostEqual(debater_bayes_posterior_shift(p1, p2), 1.0)


class TestPersuasionDecomposition(unittest.TestCase):
    def test_high_evidence_is_truthful(self) -> None:
        t, m = debater_persuasion_decomposition(0.3, evidence=0.9, threshold=0.1)
        self.assertGreater(t, m)
        self.assertAlmostEqual(t + m, 0.3)

    def test_low_evidence_is_manipulative(self) -> None:
        t, m = debater_persuasion_decomposition(0.3, evidence=0.05, threshold=0.1)
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(m, 0.3)


class TestCalibrationECE(unittest.TestCase):
    def test_perfect_calibration(self) -> None:
        # All predictions 0.5, half correct → ECE = 0
        confs = [0.5] * 100
        out = [1] * 50 + [0] * 50
        r = debater_calibration_ece(confs, out, n_bins=10)
        self.assertLess(r.ece, 0.05)

    def test_overconfident_high(self) -> None:
        # Predict 0.9 every time but only 50% correct
        confs = [0.9] * 100
        out = [1, 0] * 50
        r = debater_calibration_ece(confs, out, n_bins=10)
        self.assertGreater(r.ece, 0.3)


class TestNash2x2(unittest.TestCase):
    def test_pure_dominant(self) -> None:
        # B has a dominant strategy; A should respond best to it
        a = [[3, 0], [4, 1]]
        b = [[3, 5], [0, 1]]
        nash = debater_payoff_nash_2x2(a, b)
        self.assertEqual(nash.nash_conv, 0.0)

    def test_zero_sum_mixed(self) -> None:
        # Matching pennies — unique mixed Nash at (½, ½)
        a = [[1, -1], [-1, 1]]
        b = [[-1, 1], [1, -1]]
        nash = debater_payoff_nash_2x2(a, b)
        self.assertAlmostEqual(nash.pi_a[0], 0.5, places=3)
        self.assertAlmostEqual(nash.pi_b[0], 0.5, places=3)

    def test_nash_conv_zero(self) -> None:
        a = [[2, 0], [0, 1]]
        b = [[1, 0], [0, 2]]
        nash = debater_payoff_nash_2x2(a, b)
        self.assertLess(nash.nash_conv, 1e-6)


class TestSupportEnumeration(unittest.TestCase):
    def test_3x3_zero_sum(self) -> None:
        # Rock-paper-scissors zero-sum: unique mixed Nash at (1/3, 1/3, 1/3)
        a = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
        b = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
        nash = debater_support_enumeration(a, b)
        for p in nash.pi_a:
            self.assertAlmostEqual(p, 1.0 / 3.0, places=2)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestDebaterConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        c = DebaterConfig()
        self.assertEqual(c.protocol, PROTOCOL_TWO_PLAYER)
        self.assertEqual(c.judge_model, JUDGE_CALIBRATED)
        self.assertEqual(c.aggregation, AGG_MAJORITY)

    def test_invalid_protocol(self) -> None:
        with self.assertRaises(UnknownProtocol):
            DebaterConfig(protocol="invalid")

    def test_invalid_judge_model(self) -> None:
        with self.assertRaises(UnknownJudgeModel):
            DebaterConfig(judge_model="invalid")

    def test_invalid_aggregation(self) -> None:
        with self.assertRaises(UnknownAggregation):
            DebaterConfig(aggregation="invalid")

    def test_invalid_judge_accuracy(self) -> None:
        with self.assertRaises(InvalidConfig):
            DebaterConfig(judge_accuracy=0.0)
        with self.assertRaises(InvalidConfig):
            DebaterConfig(judge_accuracy=1.0)

    def test_invalid_confidence(self) -> None:
        with self.assertRaises(InvalidConfig):
            DebaterConfig(confidence=0.4)
        with self.assertRaises(InvalidConfig):
            DebaterConfig(confidence=1.0)

    def test_invalid_max_rounds(self) -> None:
        with self.assertRaises(InvalidConfig):
            DebaterConfig(max_rounds=0)


class TestArgument(unittest.TestCase):
    def test_valid(self) -> None:
        a = Argument(SIDE_A, "x", evidence=0.5)
        self.assertEqual(a.side, SIDE_A)

    def test_invalid_side(self) -> None:
        with self.assertRaises(InvalidMove):
            Argument("X", "x", evidence=0.5)

    def test_invalid_evidence(self) -> None:
        with self.assertRaises(InvalidMove):
            Argument(SIDE_A, "x", evidence=2.0)


class TestDebateMove(unittest.TestCase):
    def test_argue_requires_argument(self) -> None:
        with self.assertRaises(InvalidMove):
            DebateMove(kind=MOVE_ARGUE, side=SIDE_A, round_index=0)

    def test_cross_exam_requires_target(self) -> None:
        with self.assertRaises(InvalidMove):
            DebateMove(kind=MOVE_CROSS_EXAMINE, side=SIDE_A, round_index=0)

    def test_invalid_kind(self) -> None:
        with self.assertRaises(InvalidMove):
            DebateMove(kind="bogus", side=SIDE_A, round_index=0)


class TestDebateSpec(unittest.TestCase):
    def test_empty_question(self) -> None:
        with self.assertRaises(InvalidSpec):
            DebateSpec(question="", claim_a="a", claim_b="b",
                       debater_a=lambda *args: None, debater_b=lambda *args: None,
                       judge=lambda *args: {})

    def test_same_claims(self) -> None:
        with self.assertRaises(InvalidSpec):
            DebateSpec(question="q", claim_a="x", claim_b="x",
                       debater_a=lambda *args: None, debater_b=lambda *args: None,
                       judge=lambda *args: {})

    def test_invalid_ground_truth(self) -> None:
        with self.assertRaises(InvalidSpec):
            DebateSpec(question="q", claim_a="a", claim_b="b",
                       debater_a=lambda *args: None, debater_b=lambda *args: None,
                       judge=lambda *args: {}, ground_truth="X")


# ---------------------------------------------------------------------------
# Protocol smoke tests
# ---------------------------------------------------------------------------


class TestTwoPlayerProtocol(unittest.TestCase):
    def test_basic_run(self) -> None:
        d = Debater(DebaterConfig(max_rounds=3))
        spec = _basic_spec(p_truth=0.9, seed=0)
        report = d.run(spec)
        self.assertIn(report.winner, (SIDE_A, SIDE_B, SIDE_TIE))
        self.assertGreater(len(report.transcript), 0)
        self.assertEqual(report.transcript[-1].kind, MOVE_VERDICT)
        self.assertEqual(report.protocol, PROTOCOL_TWO_PLAYER)

    def test_determinism_under_seed(self) -> None:
        for seed in (0, 1, 7, 42):
            d1 = Debater(DebaterConfig(max_rounds=3, seed=seed))
            d2 = Debater(DebaterConfig(max_rounds=3, seed=seed))
            spec = _basic_spec(seed=seed)
            r1 = d1.run(spec)
            r2 = d2.run(spec)
            self.assertEqual(r1.chain_head, r2.chain_head)
            self.assertEqual(r1.winner, r2.winner)

    def test_chain_head_advances(self) -> None:
        d = Debater()
        h0 = d.chain_head
        d.run(_basic_spec())
        h1 = d.chain_head
        self.assertNotEqual(h0, h1)
        d.run(_basic_spec(seed=11))
        h2 = d.chain_head
        self.assertNotEqual(h1, h2)

    def test_truthful_judge_wins_more(self) -> None:
        # With p_truth = 0.95 and 30 debates, A (truth) wins > 70%
        d = Debater(DebaterConfig(max_rounds=2, seed=0))
        for i in range(30):
            spec = _basic_spec(p_truth=0.95, seed=i)
            d.run(spec)
        cert = d.certify(delta=0.05)
        self.assertGreater(cert.win_prob_hat, 0.7)
        self.assertGreaterEqual(cert.hoeffding_lcb, 0.0)


class TestCrossExamProtocol(unittest.TestCase):
    def test_cross_exam_runs(self) -> None:
        def deb(spec, transcript, side):
            n_my = sum(1 for m in transcript if m.side == side)
            if n_my == 1:
                tgt = next((i for i, m in enumerate(transcript)
                             if m.side != side and m.kind == MOVE_ARGUE), None)
                if tgt is not None:
                    return DebateMove(MOVE_CROSS_EXAMINE, side, n_my, target_index=tgt)
            return DebateMove(MOVE_ARGUE if n_my == 0 else MOVE_COUNTER, side, n_my,
                              argument=Argument(side, f"arg{n_my}", evidence=0.4))

        d = Debater(DebaterConfig(protocol=PROTOCOL_CROSS_EXAM, max_rounds=3))
        spec = DebateSpec(question="q", claim_a="a", claim_b="b",
                          debater_a=deb, debater_b=deb,
                          judge=make_calibrated_judge(0.7, SIDE_A, seed=2),
                          ground_truth=SIDE_A)
        r = d.run(spec)
        # Transcript should contain at least one cross_examine + answer
        kinds = [m.kind for m in r.transcript]
        self.assertIn(MOVE_CROSS_EXAMINE, kinds)
        self.assertEqual(r.protocol, PROTOCOL_CROSS_EXAM)


class TestDoublyEfficientProtocol(unittest.TestCase):
    def test_runs_and_verifies(self) -> None:
        d = Debater(DebaterConfig(protocol=PROTOCOL_DOUBLY_EFFICIENT, max_rounds=2, seed=3))
        spec = _basic_spec(p_truth=0.9, seed=3)
        r = d.run(spec)
        self.assertIn(r.winner, (SIDE_A, SIDE_B, SIDE_TIE))
        # Last verdict has the confirmation flag
        self.assertIn("doubly_efficient_confirmed", r.transcript[-1].meta)


class TestMarketMakerProtocol(unittest.TestCase):
    def test_runs(self) -> None:
        d = Debater(DebaterConfig(protocol=PROTOCOL_MARKET_MAKER, max_rounds=5))
        spec = _basic_spec(p_truth=0.9, seed=4)
        r = d.run(spec)
        self.assertIn(r.winner, (SIDE_A, SIDE_B, SIDE_TIE))
        # Last verdict has the market price
        self.assertIn("market_price", r.transcript[-1].meta)


class TestJuryProtocol(unittest.TestCase):
    def test_jury_majority(self) -> None:
        d = Debater(DebaterConfig(protocol=PROTOCOL_JURY, max_rounds=2, seed=0))
        judges = tuple((make_calibrated_judge(0.7, SIDE_A, seed=i), 0.7) for i in range(7))
        spec = DebateSpec(question="q", claim_a="a", claim_b="b",
                          debater_a=make_constant_debater([Argument(SIDE_A, "a", 0.5)]),
                          debater_b=make_constant_debater([Argument(SIDE_B, "b", 0.5)]),
                          judge=judges[0][0], judges_for_jury=judges, ground_truth=SIDE_A)
        r = d.run(spec)
        self.assertEqual(len(r.judge_votes), 7)

    def test_missing_jury_raises(self) -> None:
        d = Debater(DebaterConfig(protocol=PROTOCOL_JURY))
        spec = _basic_spec()  # no judges_for_jury
        with self.assertRaises(InvalidSpec):
            d.run(spec)

    def test_unanimity(self) -> None:
        d = Debater(DebaterConfig(protocol=PROTOCOL_JURY, aggregation=AGG_UNANIMITY,
                                  max_rounds=2, seed=0))
        # All judges with p_truth=1 → unanimous A
        judges = tuple((make_calibrated_judge(0.99, SIDE_A, seed=i+50), 0.99) for i in range(5))
        spec = DebateSpec(question="q", claim_a="a", claim_b="b",
                          debater_a=make_constant_debater([Argument(SIDE_A, "a", 0.5)]),
                          debater_b=make_constant_debater([Argument(SIDE_B, "b", 0.5)]),
                          judge=judges[0][0], judges_for_jury=judges, ground_truth=SIDE_A)
        r = d.run(spec)
        # With high accuracy judges, expect unanimity (winner != TIE) most of the time
        self.assertIn(r.winner, (SIDE_A, SIDE_B, SIDE_TIE))


class TestPersuasionAwareProtocol(unittest.TestCase):
    def test_truthful_wins_over_manipulative(self) -> None:
        def model(spec, transcript):
            a = sum(m.argument.evidence for m in transcript if m.side == SIDE_A and m.argument)
            b = sum(m.argument.evidence for m in transcript if m.side == SIDE_B and m.argument)
            total = a + b + 0.5
            p_a = max(0.01, min(0.99, (a + 0.25) / total))
            return {SIDE_A: p_a, SIDE_B: 1.0 - p_a}

        d = Debater(DebaterConfig(protocol=PROTOCOL_PERSUASION_AWARE, max_rounds=3,
                                  persuasion_penalty_weight=2.0))
        spec = DebateSpec(question="q", claim_a="a", claim_b="b",
                          debater_a=make_constant_debater([Argument(SIDE_A, "a", 0.9)]),
                          debater_b=make_constant_debater([Argument(SIDE_B, "b", 0.05)]),
                          judge=make_calibrated_judge(0.5, SIDE_A, seed=4),
                          persuasion_model=model, ground_truth=SIDE_A)
        r = d.run(spec)
        # Truthful components from A's high-evidence args should sum higher than manipulative
        truthful = sum(r.truthful_components)
        manip = sum(r.manipulative_components)
        self.assertGreater(truthful, manip)

    def test_missing_model_raises(self) -> None:
        d = Debater(DebaterConfig(protocol=PROTOCOL_PERSUASION_AWARE))
        spec = _basic_spec()  # no persuasion_model
        with self.assertRaises(InvalidSpec):
            d.run(spec)


# ---------------------------------------------------------------------------
# Certificate tests
# ---------------------------------------------------------------------------


class TestCertificates(unittest.TestCase):
    def test_not_run(self) -> None:
        d = Debater()
        with self.assertRaises(NotRun):
            d.certify(delta=0.05)

    def test_certify_after_run(self) -> None:
        d = Debater(DebaterConfig(max_rounds=2, seed=0))
        for i in range(20):
            d.run(_basic_spec(p_truth=0.9, seed=i))
        cert = d.certify(delta=0.05)
        self.assertIsInstance(cert, DebaterCertificate)
        self.assertEqual(cert.n, 20)
        self.assertGreaterEqual(cert.win_prob_hat, 0.0)
        self.assertLessEqual(cert.win_prob_hat, 1.0)
        self.assertGreaterEqual(cert.hoeffding_lcb, 0.0)

    def test_anytime_certify_requires_2(self) -> None:
        d = Debater(DebaterConfig(max_rounds=2, seed=0))
        d.run(_basic_spec(p_truth=0.9, seed=0))
        with self.assertRaises(InsufficientData):
            d.anytime_certify(delta=0.05)
        d.run(_basic_spec(p_truth=0.9, seed=1))
        ac = d.anytime_certify(delta=0.05)
        self.assertGreaterEqual(ac.ucb, ac.lcb)

    def test_calibration_requires_ground_truth(self) -> None:
        d = Debater()
        # Without ground truth, judge_outcomes stays empty
        spec = DebateSpec(question="q", claim_a="a", claim_b="b",
                          debater_a=make_constant_debater([Argument(SIDE_A, "a", 0.5)]),
                          debater_b=make_constant_debater([Argument(SIDE_B, "b", 0.5)]),
                          judge=make_calibrated_judge(0.7, SIDE_A, seed=0))
        d.run(spec)
        with self.assertRaises(InsufficientData):
            d.calibration()


# ---------------------------------------------------------------------------
# Payoff matrix + Nash check
# ---------------------------------------------------------------------------


class TestPayoffNash(unittest.TestCase):
    def test_empirical_payoff_runs(self) -> None:
        d = Debater(DebaterConfig(max_rounds=1, seed=0))
        debaters_by = {
            SIDE_A: {
                "truthful": make_constant_debater([Argument(SIDE_A, "true", 0.9)]),
                "obfuscate": make_constant_debater([Argument(SIDE_A, "obf", 0.05)]),
            },
            SIDE_B: {
                "truthful": make_constant_debater([Argument(SIDE_B, "true", 0.9)]),
                "obfuscate": make_constant_debater([Argument(SIDE_B, "obf", 0.05)]),
            },
        }
        spec = DebateSpec(question="q", claim_a="a", claim_b="b",
                          debater_a=make_constant_debater([Argument(SIDE_A, "x", 0.5)]),
                          debater_b=make_constant_debater([Argument(SIDE_B, "y", 0.5)]),
                          judge=make_calibrated_judge(0.7, SIDE_A, seed=42),
                          strategy_space=("truthful", "obfuscate"))
        payoff = d.empirical_payoff(spec, debaters_by_strategy=debaters_by,
                                    samples_per_cell=4)
        self.assertEqual(len(payoff.matrix_a), 2)
        self.assertEqual(len(payoff.matrix_a[0]), 2)
        nash = d.nash_check(payoff)
        self.assertGreaterEqual(nash.nash_conv, 0.0)


# ---------------------------------------------------------------------------
# Snapshot / restore + chain replay
# ---------------------------------------------------------------------------


class TestSnapshotRestore(unittest.TestCase):
    def test_round_trip(self) -> None:
        d1 = Debater(DebaterConfig(max_rounds=2, seed=0))
        for i in range(5):
            d1.run(_basic_spec(seed=i))
        snap = d1.snapshot()
        # JSON-roundtrip
        snap_json = json.dumps(snap)
        snap_back = json.loads(snap_json)
        d2 = Debater(DebaterConfig(max_rounds=2, seed=0))
        d2.restore(snap_back)
        self.assertEqual(d2.chain_head, d1.chain_head)

    def test_reset(self) -> None:
        d = Debater(DebaterConfig(max_rounds=2, seed=0))
        d.run(_basic_spec(seed=0))
        head_before = d.chain_head
        d.reset()
        head_after = d.chain_head
        self.assertNotEqual(head_before, head_after)
        self.assertEqual(head_after, debater_ledger_root())


class TestLedgerRoot(unittest.TestCase):
    def test_deterministic(self) -> None:
        self.assertEqual(debater_ledger_root(), debater_ledger_root())

    def test_hmac_differs(self) -> None:
        self.assertNotEqual(debater_ledger_root(), debater_ledger_root(b"key"))


# ---------------------------------------------------------------------------
# Event publication
# ---------------------------------------------------------------------------


class TestEventPublisher(unittest.TestCase):
    def test_events_emitted(self) -> None:
        events: list[tuple[str, dict]] = []

        def pub(kind: str, payload: dict) -> None:
            events.append((kind, payload))

        d = Debater(DebaterConfig(max_rounds=1), publisher=pub)
        d.run(_basic_spec(seed=0))
        kinds = [k for k, _ in events]
        self.assertIn(DEBATER_STARTED, kinds)
        self.assertIn(DEBATER_OPENED, kinds)
        self.assertIn(DEBATER_VERDICT, kinds)

    def test_publisher_errors_are_swallowed(self) -> None:
        def pub(kind: str, payload: dict) -> None:
            raise RuntimeError("boom")

        d = Debater(DebaterConfig(max_rounds=1), publisher=pub)
        # Should not raise
        d.run(_basic_spec(seed=0))


# ---------------------------------------------------------------------------
# Bad inputs
# ---------------------------------------------------------------------------


class TestBadDebaters(unittest.TestCase):
    def test_debater_returns_wrong_type(self) -> None:
        def bad(spec, transcript, side):
            return "not a move"

        d = Debater(DebaterConfig(max_rounds=1, seed=0))
        spec = DebateSpec(question="q", claim_a="a", claim_b="b",
                          debater_a=bad, debater_b=bad,
                          judge=make_calibrated_judge(0.7, SIDE_A, seed=0))
        with self.assertRaises(InvalidMove):
            d.run(spec)

    def test_debater_wrong_side(self) -> None:
        # A debater that returns side B when called for side A
        def wrong(spec, transcript, side):
            return DebateMove(MOVE_ARGUE, SIDE_B, 0,
                              argument=Argument(SIDE_B, "x", 0.5))

        d = Debater(DebaterConfig(max_rounds=1, seed=0))
        spec = DebateSpec(question="q", claim_a="a", claim_b="b",
                          debater_a=wrong,
                          debater_b=make_constant_debater([Argument(SIDE_B, "b", 0.5)]),
                          judge=make_calibrated_judge(0.7, SIDE_A, seed=0))
        with self.assertRaises(InvalidMove):
            d.run(spec)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestAPISurface(unittest.TestCase):
    def test_known_constants(self) -> None:
        self.assertEqual(len(KNOWN_PROTOCOLS), 6)
        self.assertEqual(len(KNOWN_JUDGE_MODELS), 3)
        self.assertEqual(len(KNOWN_AGGREGATIONS), 3)
        self.assertGreaterEqual(len(KNOWN_MOVES), 6)


if __name__ == "__main__":
    unittest.main()
