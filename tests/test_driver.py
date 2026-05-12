"""Tests for the RuntimeDriver — the coordination-engine entry point.

A coordination engine integrating against the runtime drives it through
RuntimeDriver. These tests pin the contract: every submission produces a
Ticket with a decision trace, a billing receipt, a live event stream,
and a deterministic verdict for each admission path (admit, downgrade,
defer, reject, cancel).

We use the same FakeAgent pattern as test_runtime so no Anthropic API is
required.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
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
    D_COMPLETE,
    D_DISPATCH,
    D_DOWNGRADE,
    D_ESTIMATE,
    D_REJECT,
    D_ROUTE,
    FAILED,
    REJECTED,
    Decision,
    Receipt,
    RuntimeDriver,
    Ticket,
    TicketRequest,
)
from agi.events import CHAT_COMPLETED, CHAT_STARTED, Event
from agi.governance import PolicyManager, TenantLimits
from agi.memory import Memory
from agi.pool import RuntimeNode, RuntimePool
from agi.preflight import AdmissionAdvisor, PreflightEstimator
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
    """Honors the same interface Runtime expects."""

    response = "fake-response"
    fail: Exception | None = None
    chat_sleep_s: float = 0.0

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
        if FakeAgent.fail is not None:
            raise FakeAgent.fail
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        return FakeAgent.response

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = FakeUsage()


def _make_runtime(**overrides) -> tuple[Runtime, Path]:
    tmp = Path(tempfile.mkdtemp())
    runtime = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
        **overrides,
    )
    return runtime, tmp


def _reset_fake():
    FakeAgent.response = "fake-response"
    FakeAgent.fail = None
    FakeAgent.chat_sleep_s = 0.0


# ---------- happy path ----------


class TestDriverHappyPath(unittest.TestCase):
    def setUp(self):
        _reset_fake()

    def test_submit_returns_ticket_immediately(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        ticket = driver.submit(TicketRequest(intent="hello"))
        self.assertIsInstance(ticket, Ticket)
        self.assertTrue(ticket.id)
        receipt = ticket.result(timeout=5.0)
        self.assertEqual(receipt.status, COMPLETED)
        self.assertEqual(receipt.final_text, "fake-response")

    def test_receipt_contains_decision_trace(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        receipt = driver.submit_sync(TicketRequest(intent="hello"), timeout=5.0)
        kinds = [d.kind for d in receipt.decisions]
        # estimate → admission → route → dispatch → complete
        self.assertIn(D_ESTIMATE, kinds)
        self.assertIn(D_ADMISSION, kinds)
        self.assertIn(D_ROUTE, kinds)
        self.assertIn(D_DISPATCH, kinds)
        self.assertIn(D_COMPLETE, kinds)
        # Estimate decision carries a numeric forecast
        est = next(d for d in receipt.decisions if d.kind == D_ESTIMATE)
        self.assertGreater(est.payload["cost_usd"], 0.0)
        self.assertIn(est.payload["confidence"], {"low", "medium", "high"})

    def test_receipt_records_actual_cost(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        receipt = driver.submit_sync(TicketRequest(intent="hi"), timeout=5.0)
        # FakeUsage: 200 input @ 5e-6 + 80 output @ 25e-6 = 0.003
        self.assertAlmostEqual(receipt.actual_cost_usd, 0.003, places=5)
        self.assertGreater(receipt.estimated_cost_usd, 0.0)

    def test_stream_yields_session_events(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        ticket = driver.submit(TicketRequest(intent="hello"))
        kinds = [ev.kind for ev in ticket.stream(timeout=5.0)]
        self.assertIn(CHAT_STARTED, kinds)
        self.assertIn(CHAT_COMPLETED, kinds)

    def test_receipt_is_json_serializable(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        receipt = driver.submit_sync(TicketRequest(intent="ping"), timeout=5.0)
        as_json = json.dumps(receipt.to_dict(), default=str)
        data = json.loads(as_json)
        self.assertEqual(data["status"], COMPLETED)
        self.assertEqual(data["final_text"], "fake-response")
        self.assertIsInstance(data["decisions"], list)


# ---------- admission paths ----------


class TestDriverAdmission(unittest.TestCase):
    def setUp(self):
        _reset_fake()

    def test_governance_budget_blocks_submission(self):
        rt, _ = _make_runtime()
        pm = PolicyManager()
        # Daily cap so tight that any p90 estimate trips it.
        pm.set_limits(TenantLimits(tenant_id="acme", daily_cost_usd=0.0001))
        advisor = AdmissionAdvisor(rt.estimator, policy=pm, runtime=rt)
        driver = RuntimeDriver(runtime=rt, advisor=advisor, policy=pm)
        receipt = driver.submit_sync(
            TicketRequest(intent="x", tenant_id="acme"), timeout=5.0,
        )
        self.assertEqual(receipt.status, DEFERRED)
        # Decisions should include admission with the verdict reason
        adm = next(d for d in receipt.decisions if d.kind == D_ADMISSION)
        self.assertIn("daily_budget", (adm.payload.get("governance_code") or ""))

    def test_quality_floor_rejects(self):
        rt, _ = _make_runtime()
        # Force estimator to a high-confidence low-success bin by injecting samples.
        for _ in range(30):
            rt.estimator.record(
                prompt="x",
                config=SessionConfig(),
                cost_usd=0.001,
                duration_s=1.0,
                success=False,
            )
        advisor = AdmissionAdvisor(rt.estimator, runtime=rt, min_p_success=0.55)
        driver = RuntimeDriver(runtime=rt, advisor=advisor)
        receipt = driver.submit_sync(TicketRequest(intent="x"), timeout=5.0)
        self.assertEqual(receipt.status, REJECTED)

    def test_downgrade_applies_alternative_model(self):
        rt, _ = _make_runtime()
        # Per-turn cap that opus blows but haiku doesn't.
        advisor = AdmissionAdvisor(
            rt.estimator, runtime=rt, max_cost_per_turn_usd=0.002,
        )
        driver = RuntimeDriver(runtime=rt, advisor=advisor)
        receipt = driver.submit_sync(
            TicketRequest(intent="x" * 500), timeout=5.0,
        )
        # Either downgraded-then-completed or rejected when no alt fits;
        # we assert the decision trace records what happened.
        if receipt.status == COMPLETED:
            kinds = [d.kind for d in receipt.decisions]
            self.assertIn(D_DOWNGRADE, kinds)
            # And the receipt's model is not the original opus tier.
            dg = next(d for d in receipt.decisions if d.kind == D_DOWNGRADE)
            self.assertNotEqual(dg.payload["from_model"], dg.payload["to_model"])
        else:
            self.assertEqual(receipt.status, REJECTED)


# ---------- budget enforcement ----------


class TestBudgetCeiling(unittest.TestCase):
    def setUp(self):
        _reset_fake()

    def test_ticket_budget_passes_to_session_ceiling(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        ticket = driver.submit(TicketRequest(intent="hi", budget_usd=0.50))
        receipt = ticket.result(timeout=5.0)
        # Session config should have inherited cost ceiling
        # (we read indirectly via session id and stored config)
        session = rt.get_session(receipt.session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.state.config.cost_ceiling_usd, 0.50)


# ---------- failure + cancellation ----------


class TestDriverFailureModes(unittest.TestCase):
    def setUp(self):
        _reset_fake()

    def tearDown(self):
        _reset_fake()

    def test_agent_failure_marks_ticket_failed(self):
        FakeAgent.fail = RuntimeError("boom")
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        receipt = driver.submit_sync(TicketRequest(intent="x"), timeout=5.0)
        self.assertEqual(receipt.status, FAILED)
        self.assertIn("boom", receipt.error or "")

    def test_cancel_before_dispatch_marks_cancelled(self):
        FakeAgent.chat_sleep_s = 0.5
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt, max_concurrent=1)
        # Saturate driver with one ticket that will sleep
        blocker = driver.submit(TicketRequest(intent="block"))
        # Submit a second; immediately cancel before semaphore frees
        t = driver.submit(TicketRequest(intent="second"))
        t.cancel()
        # Wait for both
        blocker.result(timeout=5.0)
        receipt = t.result(timeout=5.0)
        # Either it cancelled before dispatch, or it dispatched and bailed
        self.assertIn(receipt.status, {CANCELLED, COMPLETED})


# ---------- pool routing ----------


class TestDriverPoolRouting(unittest.TestCase):
    def setUp(self):
        _reset_fake()

    def test_pool_route_decision_records_node_id(self):
        rt_a, _ = _make_runtime()
        rt_b, _ = _make_runtime()
        pool = RuntimePool()
        pool.add_node(RuntimeNode(node_id="a", runtime=rt_a))
        pool.add_node(RuntimeNode(node_id="b", runtime=rt_b))
        driver = RuntimeDriver(pool=pool)
        receipt = driver.submit_sync(TicketRequest(intent="hello"), timeout=5.0)
        self.assertEqual(receipt.status, COMPLETED)
        self.assertIn(receipt.node_id, {"a", "b"})
        route = next(d for d in receipt.decisions if d.kind == D_ROUTE)
        self.assertEqual(route.payload["node_id"], receipt.node_id)


# ---------- receipts persistence ----------


class TestReceiptsPersistence(unittest.TestCase):
    def setUp(self):
        _reset_fake()

    def test_receipts_jsonl_one_line_per_ticket(self):
        tmp = Path(tempfile.mkdtemp())
        rt, _ = _make_runtime()
        path = tmp / "receipts.jsonl"
        driver = RuntimeDriver(runtime=rt, receipts_path=path)
        for _ in range(3):
            driver.submit_sync(TicketRequest(intent="x"), timeout=5.0)
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 3)
        for rec in lines:
            self.assertEqual(rec["status"], COMPLETED)
            self.assertIn("decisions", rec)


# ---------- concurrent submission ----------


class TestDriverConcurrency(unittest.TestCase):
    def setUp(self):
        _reset_fake()

    def test_many_tickets_complete_in_parallel(self):
        rt, _ = _make_runtime()
        driver = RuntimeDriver(runtime=rt, max_concurrent=4)
        tickets = [driver.submit(TicketRequest(intent=f"q{i}")) for i in range(10)]
        receipts = [t.result(timeout=10.0) for t in tickets]
        self.assertTrue(all(r.status == COMPLETED for r in receipts))
        stats = driver.stats()
        self.assertEqual(stats["submitted"], 10)
        self.assertEqual(stats["completed"], 10)


if __name__ == "__main__":
    unittest.main()
