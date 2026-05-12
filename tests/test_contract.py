"""Tests for the SLO contract surface — TicketSLO, SLOCompiler, SLOTicket,
ComplianceLedger.

These exercise the runtime-engine contract a coordination engine drives:
submit a declarative SLO and get back a hedge-aware ticket whose receipt
carries a compliance verdict. No Anthropic API is required — the existing
FakeAgent pattern from test_driver gives deterministic costs.
"""
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

from agi.contract import (
    DEFAULT_HEDGE_MAX_PARALLEL,
    SLO_BREACHED,
    SLO_COMPLIANT,
    SLO_FAILED,
    SLO_INFEASIBLE,
    STRAT_HEDGE,
    STRAT_SINGLE,
    ComplianceLedger,
    SLOCompiler,
    SLOPlan,
    SLOReceipt,
    SLOTicket,
    TicketSLO,
    evaluate_compliance,
    hedged_p_success,
)
from agi.driver import (
    COMPLETED,
    FAILED,
    REJECTED,
    RuntimeDriver,
    TicketRequest,
)
from agi.memory import Memory
from agi.preflight import PreflightEstimator
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


# ---------- fakes (mirror test_driver) ----------


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
    fail: Exception | None = None
    chat_sleep_s: float = 0.0
    # per-model override hooks for hedge-race tests
    fail_models: tuple[str, ...] = ()
    response_by_model: dict[str, str] = {}

    def __init__(self, *, memory=None, model="claude-opus-4-7", critic_threshold=0.5, **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.extra_system: str | None = None
        self.messages: list[Any] = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        if FakeAgent.chat_sleep_s > 0:
            time.sleep(FakeAgent.chat_sleep_s)
        if self.model in FakeAgent.fail_models:
            raise RuntimeError(f"forced failure for {self.model}")
        if FakeAgent.fail is not None:
            raise FakeAgent.fail
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        return FakeAgent.response_by_model.get(self.model, FakeAgent.response)

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = FakeUsage()


def _reset_fake():
    FakeAgent.response = "fake-response"
    FakeAgent.fail = None
    FakeAgent.chat_sleep_s = 0.0
    FakeAgent.fail_models = ()
    FakeAgent.response_by_model = {}


def _make_runtime(**overrides) -> tuple[Runtime, Path]:
    tmp = Path(tempfile.mkdtemp())
    runtime = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
        **overrides,
    )
    return runtime, tmp


# ============================================================
#  hedged_p_success math
# ============================================================


class TestHedgedMath(unittest.TestCase):

    def test_single_probability_returns_self(self):
        self.assertAlmostEqual(hedged_p_success([0.7]), 0.7)

    def test_two_independent_lift(self):
        # 1 - (1-0.7)(1-0.7) = 0.91
        self.assertAlmostEqual(hedged_p_success([0.7, 0.7]), 0.91)

    def test_three_independent_lift(self):
        # 1 - 0.3^3 = 0.973
        self.assertAlmostEqual(hedged_p_success([0.7, 0.7, 0.7]), 0.973)

    def test_empty_set_is_zero(self):
        self.assertEqual(hedged_p_success([]), 0.0)

    def test_clamps_out_of_range(self):
        self.assertAlmostEqual(hedged_p_success([1.5, -0.2]), 1.0)
        self.assertAlmostEqual(hedged_p_success([0.0]), 0.0)

    def test_monotone_in_n(self):
        ps = [0.6] * 5
        last = 0.0
        for k in range(1, 6):
            v = hedged_p_success(ps[:k])
            self.assertGreaterEqual(v, last)
            last = v


# ============================================================
#  TicketSLO validation
# ============================================================


class TestTicketSLOValidation(unittest.TestCase):

    def test_defaults_are_permissive(self):
        slo = TicketSLO()
        self.assertEqual(slo.min_p_success, 0.0)
        self.assertIsNone(slo.max_cost_usd)
        self.assertEqual(slo.hedge_policy, "auto")
        self.assertEqual(slo.hedge_max_parallel, DEFAULT_HEDGE_MAX_PARALLEL)

    def test_rejects_out_of_range_probability(self):
        with self.assertRaises(ValueError):
            TicketSLO(min_p_success=1.5)
        with self.assertRaises(ValueError):
            TicketSLO(min_p_success=-0.01)

    def test_rejects_negative_cost(self):
        with self.assertRaises(ValueError):
            TicketSLO(max_cost_usd=-1.0)

    def test_rejects_bad_hedge_policy(self):
        with self.assertRaises(ValueError):
            TicketSLO(hedge_policy="maybe")

    def test_rejects_zero_max_parallel(self):
        with self.assertRaises(ValueError):
            TicketSLO(hedge_max_parallel=0)

    def test_rejects_bad_refund_fraction(self):
        with self.assertRaises(ValueError):
            TicketSLO(refund_on_breach=1.5)

    def test_rejects_empty_candidate_models(self):
        with self.assertRaises(ValueError):
            TicketSLO(candidate_models=())


# ============================================================
#  SLOCompiler
# ============================================================


class TestSLOCompiler(unittest.TestCase):

    def setUp(self):
        self.estimator = PreflightEstimator()
        self.compiler = SLOCompiler(self.estimator)

    def test_compiles_single_when_cheap_model_meets_slo(self):
        # Default min_p_success=0; cheapest haiku trivially meets it.
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(max_cost_usd=10.0),
        )
        self.assertEqual(plan.strategy, STRAT_SINGLE)
        self.assertTrue(plan.feasible)
        self.assertEqual(len(plan.candidates), 1)
        self.assertEqual(plan.candidates[0].model, "claude-haiku-4-5")

    def test_picks_more_expensive_model_when_floor_demands_it(self):
        # Haiku's success prior ~0.78; opus ~0.91. Demand 0.9 — haiku
        # should be skipped in favor of the cheapest model that meets 0.9.
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(min_p_success=0.9, max_cost_usd=10.0),
        )
        self.assertEqual(plan.strategy, STRAT_SINGLE)
        self.assertTrue(plan.feasible)
        self.assertNotEqual(plan.candidates[0].model, "claude-haiku-4-5")

    def test_falls_back_to_hedge_when_no_single_model_meets_floor(self):
        # Demand 0.99 — likely above any single model's prior. Should
        # produce a multi-model hedge.
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(min_p_success=0.99, max_cost_usd=2.0),
        )
        self.assertEqual(plan.strategy, STRAT_HEDGE)
        self.assertTrue(plan.feasible)
        self.assertGreater(len(plan.candidates), 1)
        self.assertGreaterEqual(plan.expected_p_success, 0.99)

    def test_hedge_respects_max_parallel(self):
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(
                min_p_success=0.999,
                max_cost_usd=10.0,
                hedge_max_parallel=2,
            ),
        )
        self.assertLessEqual(len(plan.candidates), 2)

    def test_infeasible_when_budget_too_small(self):
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(min_p_success=0.99, max_cost_usd=0.0001),
        )
        self.assertFalse(plan.feasible)

    def test_hedge_off_refuses_to_hedge(self):
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(
                min_p_success=0.99,
                max_cost_usd=2.0,
                hedge_policy="off",
            ),
        )
        self.assertEqual(plan.strategy, STRAT_SINGLE)
        self.assertFalse(plan.feasible)
        self.assertIn("hedging disabled", plan.reason)

    def test_hedge_always_produces_multi_candidate(self):
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(
                min_p_success=0.5,
                max_cost_usd=2.0,
                hedge_policy="always",
            ),
        )
        self.assertEqual(plan.strategy, STRAT_HEDGE)
        self.assertGreaterEqual(len(plan.candidates), 1)

    def test_frontier_is_monotone_in_budget(self):
        rows = self.compiler.frontier(
            TicketRequest(intent="hi"),
            TicketSLO(min_p_success=0.95, hedge_policy="auto"),
            budgets=[0.001, 0.01, 0.05, 0.5, 5.0],
        )
        ps = [r["expected_p_success"] for r in rows]
        for a, b in zip(ps, ps[1:]):
            self.assertLessEqual(a - 1e-9, b)

    def test_custom_candidate_models_honored(self):
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(
                max_cost_usd=10.0,
                candidate_models=("claude-opus-4-7",),
            ),
        )
        self.assertEqual(plan.candidates[0].model, "claude-opus-4-7")

    def test_plan_is_json_serializable(self):
        plan = self.compiler.compile(
            TicketRequest(intent="hi"),
            TicketSLO(min_p_success=0.95, max_cost_usd=2.0),
        )
        text = json.dumps(plan.to_dict(), default=str)
        data = json.loads(text)
        self.assertIn("strategy", data)
        self.assertIn("candidates", data)


# ============================================================
#  evaluate_compliance
# ============================================================


def _toy_plan(feasible: bool = True, strategy: str = STRAT_SINGLE) -> SLOPlan:
    return SLOPlan(
        strategy=strategy,
        feasible=feasible,
        candidates=[],
        expected_cost_usd=0.05,
        expected_p_success=0.9,
        expected_duration_s=5.0,
        reason="toy",
    )


class TestComplianceEval(unittest.TestCase):

    def test_compliant_when_everything_met(self):
        rec = evaluate_compliance(
            TicketSLO(min_p_success=0.5, max_cost_usd=0.10, max_latency_s=10.0),
            _toy_plan(),
            actual_cost_usd=0.05,
            actual_duration_s=4.0,
            success=True,
            ticket_id="t1",
            chosen_model="m",
        )
        self.assertEqual(rec.slo_status, SLO_COMPLIANT)
        self.assertEqual(rec.breaches, [])
        self.assertEqual(rec.refund_usd, 0.0)

    def test_breach_on_cost(self):
        rec = evaluate_compliance(
            TicketSLO(max_cost_usd=0.01, refund_on_breach=1.0),
            _toy_plan(),
            actual_cost_usd=0.05,
            actual_duration_s=1.0,
            success=True,
            ticket_id="t1",
            chosen_model="m",
        )
        self.assertIn("cost", rec.breaches)
        self.assertEqual(rec.slo_status, SLO_BREACHED)
        self.assertAlmostEqual(rec.refund_usd, 0.05)

    def test_breach_on_latency(self):
        rec = evaluate_compliance(
            TicketSLO(max_latency_s=1.0, refund_on_breach=0.5),
            _toy_plan(),
            actual_cost_usd=0.05,
            actual_duration_s=5.0,
            success=True,
            ticket_id="t1",
            chosen_model="m",
        )
        self.assertIn("latency", rec.breaches)
        self.assertEqual(rec.slo_status, SLO_BREACHED)
        self.assertAlmostEqual(rec.refund_usd, 0.025)

    def test_infeasible_plan_marks_breach(self):
        rec = evaluate_compliance(
            TicketSLO(min_p_success=0.99),
            _toy_plan(feasible=False),
            actual_cost_usd=0.0,
            actual_duration_s=0.0,
            success=True,
            ticket_id="t1",
            chosen_model="m",
        )
        self.assertIn("infeasible_plan", rec.breaches)

    def test_failure_without_breach_marks_failed(self):
        rec = evaluate_compliance(
            TicketSLO(),
            _toy_plan(),
            actual_cost_usd=0.0,
            actual_duration_s=0.0,
            success=False,
            ticket_id="t1",
            chosen_model="m",
        )
        self.assertEqual(rec.slo_status, SLO_FAILED)


# ============================================================
#  ComplianceLedger
# ============================================================


class TestComplianceLedger(unittest.TestCase):

    def test_summary_on_empty(self):
        led = ComplianceLedger()
        s = led.summary()
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["compliance_rate"], 0.0)

    def test_records_and_rolls_up(self):
        led = ComplianceLedger()
        led.record(evaluate_compliance(
            TicketSLO(max_cost_usd=0.10),
            _toy_plan(),
            actual_cost_usd=0.05, actual_duration_s=1.0, success=True,
            ticket_id="t1", chosen_model="m",
        ))
        led.record(evaluate_compliance(
            TicketSLO(max_cost_usd=0.01),
            _toy_plan(),
            actual_cost_usd=0.05, actual_duration_s=1.0, success=True,
            ticket_id="t2", chosen_model="m",
        ))
        s = led.summary()
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["compliant"], 1)
        self.assertEqual(s["breached"], 1)
        self.assertEqual(s["by_breach"].get("cost"), 1)

    def test_persists_jsonl(self):
        tmp = Path(tempfile.mkdtemp()) / "led.jsonl"
        led = ComplianceLedger(path=tmp)
        led.record(evaluate_compliance(
            TicketSLO(),
            _toy_plan(),
            actual_cost_usd=0.0, actual_duration_s=0.0, success=True,
            ticket_id="t1", chosen_model="m",
        ))
        lines = tmp.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["ticket_id"], "t1")


# ============================================================
#  RuntimeDriver.submit_with_slo
# ============================================================


class TestDriverSubmitWithSLO(unittest.TestCase):

    def setUp(self):
        _reset_fake()

    def test_single_strategy_dispatches_one_child(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        slo = TicketSLO(max_cost_usd=10.0)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hello"), slo,
        )
        receipt = sticket.result(timeout=10.0)
        self.assertEqual(receipt.status, COMPLETED)
        self.assertEqual(receipt.slo_status, SLO_COMPLIANT)
        self.assertEqual(len(receipt.children), 1)
        self.assertEqual(receipt.plan.strategy, STRAT_SINGLE)

    def test_hedge_dispatches_multiple_children(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        slo = TicketSLO(min_p_success=0.99, max_cost_usd=2.0, hedge_policy="auto")
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hello"), slo,
        )
        receipt = sticket.result(timeout=10.0)
        self.assertEqual(receipt.status, COMPLETED)
        self.assertEqual(receipt.plan.strategy, STRAT_HEDGE)
        self.assertGreater(len(receipt.children), 1)
        # Cost is aggregate across children — every child ran or was cancelled.
        self.assertGreaterEqual(receipt.actual_cost_usd, 0.0)

    def test_hedge_first_winner_supplies_final_text(self):
        rt, _ = _make_runtime()
        FakeAgent.response_by_model = {
            "claude-haiku-4-5": "haiku-answer",
            "claude-sonnet-4-6": "sonnet-answer",
            "claude-opus-4-7": "opus-answer",
        }
        driver = RuntimeDriver(runtime=rt)
        slo = TicketSLO(min_p_success=0.99, max_cost_usd=2.0)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hello"), slo,
        )
        receipt = sticket.result(timeout=10.0)
        self.assertEqual(receipt.status, COMPLETED)
        self.assertIn(
            receipt.final_text,
            {"haiku-answer", "sonnet-answer", "opus-answer"},
        )
        self.assertIsNotNone(receipt.winner_model)

    def test_hedge_survives_one_failing_model(self):
        rt, _ = _make_runtime()
        FakeAgent.fail_models = ("claude-haiku-4-5",)
        FakeAgent.response_by_model = {"claude-opus-4-7": "rescued"}
        driver = RuntimeDriver(runtime=rt)
        slo = TicketSLO(
            min_p_success=0.99, max_cost_usd=2.0, hedge_policy="auto",
        )
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hello"), slo,
        )
        receipt = sticket.result(timeout=10.0)
        # One child fails, another succeeds → overall COMPLETED.
        self.assertEqual(receipt.status, COMPLETED)
        self.assertNotEqual(receipt.winner_model, "claude-haiku-4-5")

    def test_hedge_all_failing_marks_failed(self):
        rt, _ = _make_runtime()
        # Fail every default candidate model.
        FakeAgent.fail_models = (
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-7",
        )
        driver = RuntimeDriver(runtime=rt)
        slo = TicketSLO(min_p_success=0.99, max_cost_usd=2.0)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hello"), slo,
        )
        receipt = sticket.result(timeout=10.0)
        self.assertEqual(receipt.status, FAILED)
        self.assertIn(receipt.slo_status, (SLO_FAILED, SLO_BREACHED))

    def test_infeasible_dispatched_marks_breach(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        # Tiny budget makes the plan infeasible.
        slo = TicketSLO(min_p_success=0.99, max_cost_usd=0.0001)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hi"), slo,
            dispatch_infeasible=True,
        )
        receipt = sticket.result(timeout=10.0)
        self.assertIn("infeasible_plan", receipt.breaches)

    def test_dispatch_infeasible_false_rejects_up_front(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        slo = TicketSLO(min_p_success=0.99, max_cost_usd=0.0001)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hi"), slo,
            dispatch_infeasible=False,
        )
        receipt = sticket.result(timeout=2.0)
        self.assertEqual(receipt.status, REJECTED)
        self.assertEqual(receipt.slo_status, SLO_INFEASIBLE)
        self.assertEqual(receipt.children, [])

    def test_compliance_report_after_runs(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        # Run two SLO submissions: one compliant, one cost-breach.
        sticket_ok = driver.submit_with_slo(
            TicketRequest(intent="ok"),
            TicketSLO(max_cost_usd=10.0),
        )
        sticket_ok.result(timeout=10.0)

        sticket_breach = driver.submit_with_slo(
            TicketRequest(intent="b"),
            TicketSLO(max_cost_usd=0.00001),  # FakeAgent cost is ~0.003
            dispatch_infeasible=True,
        )
        sticket_breach.result(timeout=10.0)

        report = driver.compliance_report()
        self.assertGreaterEqual(report["total"], 2)
        self.assertGreaterEqual(report["compliant"], 1)
        self.assertGreater(report["breached"] + report["infeasible"], 0)

    def test_receipt_is_json_serializable(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hello"),
            TicketSLO(max_cost_usd=10.0),
        )
        receipt = sticket.result(timeout=10.0)
        text = json.dumps(receipt.to_dict(), default=str)
        data = json.loads(text)
        self.assertEqual(data["status"], COMPLETED)
        self.assertIn("plan", data)
        self.assertIn("slo_status", data)

    def test_stream_yields_child_events(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hello"),
            TicketSLO(min_p_success=0.99, max_cost_usd=2.0),
        )
        kinds = [ev.kind for ev in sticket.stream(timeout=5.0)]
        from agi.events import CHAT_STARTED, CHAT_COMPLETED
        self.assertIn(CHAT_STARTED, kinds)
        self.assertIn(CHAT_COMPLETED, kinds)

    def test_frontier_for_slo_via_driver(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        slo = TicketSLO(min_p_success=0.95, hedge_policy="auto")
        rows = driver.frontier_for_slo(
            TicketRequest(intent="hi"),
            slo,
            budgets=[0.001, 0.05, 0.5, 5.0],
        )
        self.assertEqual(len(rows), 4)
        ps = [r["expected_p_success"] for r in rows]
        for a, b in zip(ps, ps[1:]):
            self.assertLessEqual(a - 1e-9, b)

    def test_driver_stats_track_slo_submissions(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hello"),
            TicketSLO(min_p_success=0.99, max_cost_usd=2.0),
        )
        sticket.result(timeout=10.0)
        stats = driver.stats()
        self.assertGreaterEqual(stats["slo_submitted"], 1)
        self.assertGreaterEqual(stats["slo_hedged"], 1)


# ============================================================
#  Persistence integration
# ============================================================


class TestComplianceLedgerWithDriver(unittest.TestCase):

    def setUp(self):
        _reset_fake()

    def test_compliance_persists_to_path(self):
        rt, _ = _make_runtime()
        tmp = Path(tempfile.mkdtemp()) / "compliance.jsonl"
        driver = RuntimeDriver(runtime=rt, compliance_path=tmp)
        sticket = driver.submit_with_slo(
            TicketRequest(intent="hi"),
            TicketSLO(max_cost_usd=10.0),
        )
        sticket.result(timeout=10.0)
        self.assertTrue(tmp.exists())
        text = tmp.read_text()
        self.assertGreater(len(text.splitlines()), 0)
        line = json.loads(text.splitlines()[0])
        self.assertIn("slo_status", line)
        self.assertEqual(line["slo_status"], SLO_COMPLIANT)


if __name__ == "__main__":
    unittest.main()
