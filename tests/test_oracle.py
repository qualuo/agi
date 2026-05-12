"""Tests for `TicketOracle` — counterfactual replay + auto-tuning."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.driver import (
    CANCELLED,
    COMPLETED,
    DEFERRED,
    D_ADMISSION,
    D_ESTIMATE,
    Decision,
    Receipt,
    REJECTED,
    RuntimeDriver,
    TicketRequest,
)
from agi.memory import Memory
from agi.oracle import (
    CounterfactualReport,
    PolicyKnobs,
    Recommendation,
    TicketOracle,
    WhatIfReport,
)
from agi.preflight import ADMIT, DEFER, DOWNGRADE, REJECT, PreflightEstimator
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


# ---------- fakes ----------


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    response = "fake-response"

    def __init__(self, *, memory=None, model="claude-opus-4-7", **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.extra_system: str | None = None
        self.messages: list[Any] = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        return FakeAgent.response

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = FakeUsage()


def _make_driver(**overrides) -> tuple[RuntimeDriver, Path]:
    tmp = Path(tempfile.mkdtemp())
    rt = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
        **overrides,
    )
    return RuntimeDriver(runtime=rt), tmp


def _synth_receipt(
    *,
    ticket_id: str = "t1",
    tenant_id: str | None = None,
    verdict: str = ADMIT,
    status: str = COMPLETED,
    model: str = "claude-opus-4-7",
    est_cost: float = 0.05,
    est_cost_p90: float | None = None,
    est_p_success: float = 0.9,
    actual_cost: float = 0.04,
    confidence: str = "high",
    samples: int = 50,
) -> Receipt:
    """Build a Receipt with a decision trace the oracle can replay."""
    p90 = est_cost_p90 if est_cost_p90 is not None else est_cost * 1.8
    return Receipt(
        ticket_id=ticket_id,
        intent=f"intent for {ticket_id}",
        status=status,
        tenant_id=tenant_id,
        model=model,
        actual_cost_usd=actual_cost,
        estimated_cost_usd=est_cost,
        estimated_p_success=est_p_success,
        decisions=[
            Decision(
                kind=D_ESTIMATE,
                ts=time.time(),
                payload={
                    "cost_usd": est_cost,
                    "cost_p10_usd": est_cost * 0.5,
                    "cost_p90_usd": p90,
                    "duration_s": 5.0,
                    "p_success": est_p_success,
                    "confidence": confidence,
                    "samples": samples,
                    "model": model,
                },
            ),
            Decision(
                kind=D_ADMISSION,
                ts=time.time(),
                payload={"verdict": verdict, "reason": "synthetic"},
            ),
        ],
    )


# ---------- replay basics ----------


class TestReplayBasics(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = PreflightEstimator()
        self.oracle = TicketOracle(self.estimator)

    def test_baseline_admit_replays_to_actual_cost(self):
        r = _synth_receipt(actual_cost=0.07, est_cost=0.05)
        report = self.oracle.replay([r], PolicyKnobs())
        self.assertEqual(report.n_replayable, 1)
        self.assertEqual(report.replays[0].alt_verdict, ADMIT)
        # Under the baseline knobs the alt should match the actual cost.
        self.assertAlmostEqual(report.replays[0].alt_cost_usd, 0.07, places=6)
        self.assertAlmostEqual(report.alt_cost_usd, 0.07, places=6)
        self.assertEqual(report.projected_cost_savings_usd, 0.0)

    def test_lowering_cost_cap_rejects_expensive_admits(self):
        # Three cheap (<$0.04) and three expensive (>$0.10) tickets.
        receipts = [
            _synth_receipt(
                ticket_id=f"cheap-{i}",
                est_cost=0.02,
                est_cost_p90=0.03,
                actual_cost=0.025,
            )
            for i in range(3)
        ] + [
            _synth_receipt(
                ticket_id=f"big-{i}",
                est_cost=0.10,
                est_cost_p90=0.20,
                actual_cost=0.15,
            )
            for i in range(3)
        ]
        knobs = PolicyKnobs(max_cost_per_turn_usd=0.05, allow_downgrade=False)
        report = self.oracle.replay(receipts, knobs)
        # The cheap ones stay ADMIT; the expensive ones flip to REJECT.
        verdicts = [r.alt_verdict for r in report.replays]
        self.assertEqual(verdicts.count(ADMIT), 3)
        self.assertEqual(verdicts.count(REJECT), 3)
        # Savings = expensive baseline cost (~0.45) - cheap baseline (~0.075)
        # alt = cheap kept at 0.025 each, expensive zero'd out.
        self.assertGreater(report.projected_cost_savings_usd, 0.4)

    def test_skips_receipts_without_decisions(self):
        r = Receipt(ticket_id="bad", intent="x", status=COMPLETED)
        report = self.oracle.replay([r], PolicyKnobs())
        self.assertEqual(report.n_replayable, 0)
        self.assertEqual(report.n_skipped, 1)

    def test_verdict_change_buckets_populated(self):
        receipts = [
            _synth_receipt(ticket_id="big", est_cost=0.20, est_cost_p90=0.40,
                           actual_cost=0.30),
        ]
        report = self.oracle.replay(
            receipts,
            PolicyKnobs(max_cost_per_turn_usd=0.10, allow_downgrade=False),
        )
        self.assertEqual(report.verdict_changes, {f"{ADMIT}->{REJECT}": 1})

    def test_low_confidence_skips_psuccess_floor(self):
        # Low-confidence high-cost-cap baseline; bumping the floor up
        # should NOT reject anything because est_confidence='low'.
        r = _synth_receipt(est_p_success=0.30, confidence="low")
        report = self.oracle.replay(
            [r],
            PolicyKnobs(min_p_success=0.80),
        )
        self.assertEqual(report.replays[0].alt_verdict, ADMIT)

    def test_high_confidence_psuccess_floor_rejects(self):
        r = _synth_receipt(est_p_success=0.30, confidence="high", samples=80)
        report = self.oracle.replay(
            [r],
            PolicyKnobs(min_p_success=0.50),
        )
        self.assertEqual(report.replays[0].alt_verdict, REJECT)


# ---------- recommend ----------


class TestRecommend(unittest.TestCase):
    def test_recommend_finds_cheaper_knobs(self):
        oracle = TicketOracle(PreflightEstimator())
        cheap = [
            _synth_receipt(
                ticket_id=f"c-{i}",
                est_cost=0.01,
                est_cost_p90=0.02,
                actual_cost=0.012,
            )
            for i in range(10)
        ]
        expensive = [
            _synth_receipt(
                ticket_id=f"e-{i}",
                est_cost=0.50,
                est_cost_p90=1.20,
                actual_cost=0.70,
            )
            for i in range(10)
        ]
        rec = oracle.recommend(
            cheap + expensive,
            require_no_worse_success=False,
            min_population=5,
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertGreater(rec.improvement.projected_cost_savings_usd, 0.0)
        self.assertIn("savings", rec.summary)
        # The recommended cap should bite the expensive bucket.
        self.assertIsNotNone(rec.knobs.max_cost_per_turn_usd)

    def test_recommend_returns_none_when_insufficient_population(self):
        oracle = TicketOracle(PreflightEstimator())
        rec = oracle.recommend(
            [_synth_receipt()], require_no_worse_success=False, min_population=5,
        )
        self.assertIsNone(rec)

    def test_apply_mutates_advisor(self):
        oracle = TicketOracle(PreflightEstimator())
        from agi.preflight import AdmissionAdvisor
        advisor = AdmissionAdvisor(oracle._estimator)
        before = (advisor._min_p_success, advisor._max_cost_per_turn_usd)
        oracle.apply(
            advisor,
            PolicyKnobs(min_p_success=0.70, max_cost_per_turn_usd=0.05),
        )
        self.assertEqual(advisor._min_p_success, 0.70)
        self.assertEqual(advisor._max_cost_per_turn_usd, 0.05)
        # Baseline rolls forward so subsequent compare() works.
        self.assertEqual(oracle.baseline.min_p_success, 0.70)


# ---------- what-if ----------


class TestWhatIf(unittest.TestCase):
    def test_cost_multiplier_scales_spend(self):
        oracle = TicketOracle(PreflightEstimator())
        receipts = [_synth_receipt(actual_cost=0.10) for _ in range(5)]
        report = oracle.what_if(receipts, cost_multiplier=1.5)
        self.assertEqual(report.n_tickets, 5)
        self.assertAlmostEqual(report.baseline_cost_usd, 0.50, places=6)
        self.assertAlmostEqual(report.shocked_cost_usd, 0.75, places=6)
        self.assertAlmostEqual(report.projected_cost_delta_usd, 0.25, places=6)


# ---------- driver integration ----------


class TestDriverIntegration(unittest.TestCase):
    def test_driver_exposes_oracle(self):
        driver, _ = _make_driver()
        oracle = driver.oracle
        self.assertIsInstance(oracle, TicketOracle)
        # Idempotent / cached.
        self.assertIs(driver.oracle, oracle)

    def test_completed_receipts_feed_oracle_buffer(self):
        driver, _ = _make_driver()
        # Touch the property so the driver knows to mirror receipts.
        _ = driver.oracle
        for i in range(3):
            driver.submit_sync(TicketRequest(intent=f"hi-{i}"), timeout=5.0)
        self.assertEqual(len(driver.oracle.receipts()), 3)

    def test_auto_tune_no_op_below_savings_floor(self):
        driver, _ = _make_driver()
        _ = driver.oracle
        # Submit a few uniform tickets; no policy change should beat
        # the baseline savings floor of $1.00.
        for i in range(6):
            driver.submit_sync(TicketRequest(intent=f"x-{i}"), timeout=5.0)
        rec = driver.oracle.auto_tune(driver, min_savings_usd=1.00)
        self.assertIsNone(rec)

    def test_auto_tune_applies_recommendation_when_above_floor(self):
        driver, _ = _make_driver()
        _ = driver.oracle
        # Mixed population: cheap ones that succeed, plus expensive
        # ones that *failed* — the textbook case where capping cost is
        # a clear win without harming hit rate.
        for i in range(6):
            driver.oracle.record(
                _synth_receipt(
                    ticket_id=f"good-{i}",
                    est_cost=0.02,
                    est_cost_p90=0.03,
                    actual_cost=0.018,
                    est_p_success=0.95,
                    status=COMPLETED,
                )
            )
        for i in range(6):
            driver.oracle.record(
                _synth_receipt(
                    ticket_id=f"bad-{i}",
                    est_cost=0.50,
                    est_cost_p90=1.50,
                    actual_cost=0.80,
                    est_p_success=0.40,
                    confidence="high",
                    samples=80,
                    # FAILED in production — money spent for nothing.
                    status="failed",
                )
            )
        rec = driver.oracle.auto_tune(driver, min_savings_usd=0.10)
        self.assertIsNotNone(rec)
        assert rec is not None
        # The driver's advisor should now reflect the applied knobs.
        self.assertEqual(
            driver.advisor._max_cost_per_turn_usd,
            rec.knobs.max_cost_per_turn_usd,
        )

    def test_recommendation_is_json_serializable(self):
        oracle = TicketOracle(PreflightEstimator())
        receipts = [
            _synth_receipt(
                ticket_id=f"e-{i}",
                est_cost=0.50,
                est_cost_p90=1.20,
                actual_cost=0.70,
            )
            for i in range(6)
        ]
        rec = oracle.recommend(receipts, require_no_worse_success=False)
        self.assertIsNotNone(rec)
        assert rec is not None
        blob = json.dumps(rec.to_dict(), default=str)
        data = json.loads(blob)
        self.assertIn("knobs", data)
        self.assertIn("improvement", data)
        self.assertIn("summary", data)


if __name__ == "__main__":
    unittest.main()
