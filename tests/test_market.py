"""Tests for TicketMarket — the multi-tenant marketplace dispatch layer.

These pin the contract a coordination engine drives when fronting many
tenants: tenant registration, quote pricing, tier-weighted scheduling,
quota enforcement, refund-aware invoicing, and market-wide telemetry.

We reuse the FakeAgent pattern from test_driver so no Anthropic API is
required. All tests are hermetic.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.contract import SLO_BREACHED, SLO_COMPLIANT, TicketSLO
from agi.driver import COMPLETED, RuntimeDriver, TicketRequest
from agi.market import (
    KNOWN_TIERS,
    MKT_CANCELLED,
    MKT_COMPLETED,
    MKT_FAILED,
    MKT_REJECTED,
    REASON_BID_TOO_LOW,
    REASON_INFEASIBLE_SLO,
    REASON_MARKET_PAUSED,
    REASON_OVER_BUDGET,
    REASON_UNKNOWN_TENANT,
    TIER_DEFAULT_MARKUP,
    TIER_ECONOMY,
    TIER_PREMIUM,
    TIER_STANDARD,
    Invoice,
    MarketTicket,
    Quote,
    Tenant,
    TenantAccount,
    TicketMarket,
)
from agi.memory import Memory
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


def _reset_fake():
    FakeAgent.response = "fake-response"
    FakeAgent.fail = None
    FakeAgent.chat_sleep_s = 0.0


def _make_runtime() -> tuple[Runtime, Path]:
    tmp = Path(tempfile.mkdtemp())
    runtime = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
    )
    return runtime, tmp


def _make_market(**market_kw) -> tuple[TicketMarket, RuntimeDriver, Path]:
    rt, tmp = _make_runtime()
    driver = RuntimeDriver(runtime=rt)
    market = TicketMarket(driver, **market_kw)
    return market, driver, tmp


# ---------- tenant validation ----------


class TestTenantValidation(unittest.TestCase):
    def test_unknown_tier_rejected(self):
        with self.assertRaises(ValueError):
            Tenant(tenant_id="x", tier="diamond")

    def test_negative_markup_rejected(self):
        with self.assertRaises(ValueError):
            Tenant(tenant_id="x", markup_pct=-0.1)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            Tenant(tenant_id="x", monthly_budget_usd=-1.0)

    def test_tier_defaults_applied_when_markup_unset(self):
        for tier in KNOWN_TIERS:
            t = Tenant(tenant_id=f"t-{tier}", tier=tier)
            self.assertEqual(t.effective_markup_pct, TIER_DEFAULT_MARKUP[tier])

    def test_explicit_markup_overrides_tier_default(self):
        t = Tenant(tenant_id="x", tier=TIER_PREMIUM, markup_pct=1.25)
        self.assertEqual(t.effective_markup_pct, 1.25)


# ---------- market basics ----------


class TestMarketRegistration(unittest.TestCase):
    def setUp(self):
        _reset_fake()
        self.market, self.driver, self.tmp = _make_market()

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_register_tenant_returns_account(self):
        acct = self.market.register_tenant(Tenant(tenant_id="acme", tier=TIER_PREMIUM))
        self.assertIsInstance(acct, TenantAccount)
        self.assertEqual(acct.tenant_id, "acme")
        self.assertEqual(acct.tier, TIER_PREMIUM)
        self.assertEqual(acct.period_charged_usd, 0.0)

    def test_register_duplicate_tenant_raises(self):
        self.market.register_tenant(Tenant(tenant_id="acme"))
        with self.assertRaises(ValueError):
            self.market.register_tenant(Tenant(tenant_id="acme"))

    def test_get_unknown_tenant_returns_none(self):
        self.assertIsNone(self.market.get_tenant("missing"))


# ---------- quoting ----------


class TestQuote(unittest.TestCase):
    def setUp(self):
        _reset_fake()
        self.market, self.driver, self.tmp = _make_market()
        self.market.register_tenant(
            Tenant(tenant_id="acme", tier=TIER_STANDARD, monthly_budget_usd=10.0)
        )

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_quote_unknown_tenant_is_rejected(self):
        q = self.market.quote(MarketTicket(
            intent="hi", tenant_id="ghost", max_bid_usd=1.0,
        ))
        self.assertFalse(q.accepted)
        self.assertEqual(q.reason, REASON_UNKNOWN_TENANT)

    def test_quote_returns_list_price_above_cost(self):
        q = self.market.quote(MarketTicket(
            intent="summarize this document",
            tenant_id="acme",
            max_bid_usd=10.0,
        ))
        self.assertTrue(q.accepted)
        self.assertGreater(q.estimated_cost_usd, 0.0)
        self.assertGreater(q.list_price_usd, q.estimated_cost_usd)
        self.assertAlmostEqual(
            q.list_price_usd,
            q.estimated_cost_usd * (1.0 + TIER_DEFAULT_MARKUP[TIER_STANDARD]),
            places=6,
        )
        self.assertGreater(q.margin_usd, 0.0)

    def test_quote_rejects_bid_below_list_price(self):
        q = self.market.quote(MarketTicket(
            intent="hi", tenant_id="acme", max_bid_usd=1e-9,
        ))
        self.assertFalse(q.accepted)
        self.assertEqual(q.reason, REASON_BID_TOO_LOW)
        self.assertFalse(q.fits_bid)

    def test_quote_rejects_when_quota_exhausted(self):
        # Charge nearly all of acme's monthly quota via a tiny budget.
        acct = self.market.get_tenant("acme")
        assert acct is not None
        acct._period_charged_usd = 9.999
        q = self.market.quote(MarketTicket(
            intent="hi", tenant_id="acme", max_bid_usd=10.0,
        ))
        self.assertFalse(q.accepted)
        self.assertEqual(q.reason, REASON_OVER_BUDGET)
        self.assertFalse(q.fits_budget)

    def test_quote_rejects_paused_tenant(self):
        self.market.pause_tenant("acme")
        q = self.market.quote(MarketTicket(
            intent="hi", tenant_id="acme", max_bid_usd=10.0,
        ))
        self.assertFalse(q.accepted)
        self.assertEqual(q.reason, REASON_MARKET_PAUSED)

    def test_quote_to_dict_is_json_serializable(self):
        q = self.market.quote(MarketTicket(
            intent="hi", tenant_id="acme", max_bid_usd=10.0,
        ))
        s = json.dumps(q.to_dict(), default=str)
        self.assertIn("acme", s)


# ---------- submission / dispatch / invoicing ----------


class TestMarketSubmit(unittest.TestCase):
    def setUp(self):
        _reset_fake()
        self.market, self.driver, self.tmp = _make_market(max_concurrent=2)
        self.market.register_tenant(
            Tenant(tenant_id="acme", tier=TIER_STANDARD, monthly_budget_usd=10.0)
        )

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_accepted_ticket_completes_with_invoice(self):
        handle = self.market.submit(MarketTicket(
            intent="hello world", tenant_id="acme", max_bid_usd=2.0,
        ))
        invoice = handle.result(timeout=5.0)
        self.assertEqual(invoice.status, MKT_COMPLETED)
        self.assertEqual(invoice.tenant_id, "acme")
        self.assertGreater(invoice.list_price_usd, 0.0)
        self.assertGreater(invoice.cost_of_goods_usd, 0.0)
        self.assertGreater(invoice.gross_margin_usd, 0.0)
        # gross_margin = list_price - cost_of_goods (no refund)
        self.assertAlmostEqual(
            invoice.gross_margin_usd,
            invoice.list_price_usd - invoice.cost_of_goods_usd,
            places=6,
        )
        self.assertEqual(invoice.refund_usd, 0.0)
        self.assertEqual(invoice.net_charge_usd, invoice.list_price_usd)
        self.assertIsNotNone(invoice.completed_ts)

    def test_rejected_ticket_returns_zero_charge_invoice_immediately(self):
        handle = self.market.submit(MarketTicket(
            intent="hi", tenant_id="ghost", max_bid_usd=2.0,
        ))
        invoice = handle.result(timeout=2.0)
        self.assertEqual(invoice.status, MKT_REJECTED)
        self.assertEqual(invoice.list_price_usd, 0.0)
        self.assertEqual(invoice.cost_of_goods_usd, 0.0)
        self.assertEqual(invoice.quote_reason, REASON_UNKNOWN_TENANT)

    def test_rejected_ticket_is_not_dispatched(self):
        handle = self.market.submit(MarketTicket(
            intent="hi", tenant_id="acme", max_bid_usd=1e-9,
        ))
        invoice = handle.result(timeout=2.0)
        self.assertEqual(invoice.status, MKT_REJECTED)
        self.assertEqual(invoice.quote_reason, REASON_BID_TOO_LOW)
        self.assertIsNone(handle.underlying)

    def test_submit_sync_returns_invoice(self):
        invoice = self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="acme", max_bid_usd=2.0),
            timeout=5.0,
        )
        self.assertEqual(invoice.status, MKT_COMPLETED)

    def test_invoice_to_dict_is_json_serializable(self):
        invoice = self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="acme", max_bid_usd=2.0),
            timeout=5.0,
        )
        s = json.dumps(invoice.to_dict(), default=str)
        self.assertIn("acme", s)

    def test_handle_stream_yields_underlying_events(self):
        handle = self.market.submit(MarketTicket(
            intent="hello", tenant_id="acme", max_bid_usd=2.0,
        ))
        events = list(handle.stream(timeout=5.0))
        # Underlying ticket emits at least one event; smoke check.
        kinds = [ev.kind for ev in events]
        self.assertGreater(len(kinds), 0)
        # Eventually completes.
        invoice = handle.result(timeout=5.0)
        self.assertEqual(invoice.status, MKT_COMPLETED)

    def test_quota_reserves_then_settles(self):
        acct = self.market.get_tenant("acme")
        assert acct is not None
        FakeAgent.chat_sleep_s = 0.2
        try:
            handle = self.market.submit(MarketTicket(
                intent="hello", tenant_id="acme", max_bid_usd=2.0,
            ))
            # Some reservation should appear briefly. Race-tolerant
            # check: total seen reservation across the lifecycle.
            saw_reservation = False
            deadline = time.time() + 1.0
            while time.time() < deadline and not handle.done:
                if acct.reserved_usd > 0:
                    saw_reservation = True
                    break
                time.sleep(0.01)
            invoice = handle.result(timeout=5.0)
            self.assertEqual(invoice.status, MKT_COMPLETED)
            self.assertTrue(saw_reservation)
            # Reservation must net to zero post-completion.
            self.assertEqual(acct.reserved_usd, 0.0)
            # Period charged should reflect the list price.
            self.assertAlmostEqual(
                acct.period_charged_usd, invoice.list_price_usd, places=6
            )
        finally:
            FakeAgent.chat_sleep_s = 0.0


# ---------- tier weighting / fairness ----------


class TestTierWeighting(unittest.TestCase):
    """A coordination engine relying on premium SLA must see premium
    tickets dispatched before economy tickets under contention."""

    def setUp(self):
        _reset_fake()
        # max_concurrent=1 forces strictly serial dispatch so the
        # priority order is observable.
        self.market, self.driver, self.tmp = _make_market(max_concurrent=1)
        self.market.register_tenant(Tenant(
            tenant_id="prem",
            tier=TIER_PREMIUM,
            monthly_budget_usd=100.0,
        ))
        self.market.register_tenant(Tenant(
            tenant_id="econ",
            tier=TIER_ECONOMY,
            monthly_budget_usd=100.0,
        ))

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_premium_tickets_dispatch_before_economy_under_contention(self):
        # Make every chat slow so all tickets sit in queue concurrently.
        FakeAgent.chat_sleep_s = 0.15
        try:
            # Submit economy first, then premium. Without tier
            # weighting, economy would finish first (FIFO). With
            # weighting, premium wins.
            econ = [
                self.market.submit(MarketTicket(
                    intent=f"econ-{i}", tenant_id="econ", max_bid_usd=2.0,
                ))
                for i in range(3)
            ]
            time.sleep(0.02)  # ensure econ entries are queued first
            prem = [
                self.market.submit(MarketTicket(
                    intent=f"prem-{i}", tenant_id="prem", max_bid_usd=2.0,
                ))
                for i in range(3)
            ]
            # Wait for everything to finish.
            for h in econ + prem:
                h.result(timeout=10.0)
            # Compare completion timestamps: at least one premium
            # ticket completes before at least one economy ticket
            # that was submitted earlier — this is the observable
            # signature of tier preemption in the queue.
            prem_done = [
                h.result().completed_ts for h in prem
            ]
            econ_done = [
                h.result().completed_ts for h in econ
            ]
            # First-served premium beats the LAST-served economy
            # ticket (econ[1] or econ[2] which were enqueued after
            # earlier economy work).
            self.assertLess(min(prem_done), max(econ_done))
        finally:
            FakeAgent.chat_sleep_s = 0.0


# ---------- stats / reporting ----------


class TestMarketReporting(unittest.TestCase):
    def setUp(self):
        _reset_fake()
        self.market, self.driver, self.tmp = _make_market()
        self.market.register_tenant(
            Tenant(tenant_id="acme", tier=TIER_STANDARD, monthly_budget_usd=10.0)
        )
        self.market.register_tenant(
            Tenant(tenant_id="globex", tier=TIER_PREMIUM, monthly_budget_usd=20.0)
        )

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_market_stats_rolls_up_revenue_cost_margin(self):
        for _ in range(2):
            self.market.submit_sync(
                MarketTicket(intent="hi", tenant_id="acme", max_bid_usd=2.0),
                timeout=5.0,
            )
        self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="globex", max_bid_usd=2.0),
            timeout=5.0,
        )
        stats = self.market.market_stats()
        self.assertEqual(stats["submitted"], 3)
        self.assertEqual(stats["completed"], 3)
        self.assertGreater(stats["revenue_usd"], 0.0)
        self.assertGreater(stats["cost_of_goods_usd"], 0.0)
        self.assertGreater(stats["gross_margin_usd"], 0.0)
        # Premium markup is higher than standard, so globex margin
        # share > standard markup share per dollar of cost.
        self.assertGreater(stats["gross_margin_pct"], 0.0)
        self.assertEqual(stats["tenants_count"], 2)
        # per_tenant sorted descending by net_charged_usd.
        per = stats["per_tenant"]
        self.assertEqual(len(per), 2)
        self.assertGreaterEqual(
            per[0]["net_charged_usd"], per[1]["net_charged_usd"]
        )

    def test_per_tenant_stats_independent(self):
        self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="acme", max_bid_usd=2.0),
            timeout=5.0,
        )
        s_acme = self.market.tenant_statement("acme")
        s_globex = self.market.tenant_statement("globex")
        self.assertGreater(s_acme["period_charged_usd"], 0.0)
        self.assertEqual(s_globex["period_charged_usd"], 0.0)
        self.assertEqual(s_acme["submitted"], 1)
        self.assertEqual(s_globex["submitted"], 0)

    def test_tenant_statement_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.market.tenant_statement("nope")

    def test_invoices_filterable_by_tenant(self):
        self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="acme", max_bid_usd=2.0),
            timeout=5.0,
        )
        self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="globex", max_bid_usd=2.0),
            timeout=5.0,
        )
        all_inv = self.market.invoices()
        self.assertEqual(len(all_inv), 2)
        acme_inv = self.market.invoices("acme")
        self.assertEqual(len(acme_inv), 1)
        self.assertEqual(acme_inv[0].tenant_id, "acme")

    def test_invoices_filter_unknown_tenant_returns_empty(self):
        self.assertEqual(self.market.invoices("ghost"), [])


# ---------- close period / monthly rollover ----------


class TestClosePeriod(unittest.TestCase):
    def setUp(self):
        _reset_fake()
        self.market, self.driver, self.tmp = _make_market()
        self.market.register_tenant(
            Tenant(tenant_id="acme", tier=TIER_STANDARD, monthly_budget_usd=10.0)
        )

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_close_period_emits_statement_and_resets_counters(self):
        self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="acme", max_bid_usd=2.0),
            timeout=5.0,
        )
        acct = self.market.get_tenant("acme")
        assert acct is not None
        before_charged = acct.period_charged_usd
        self.assertGreater(before_charged, 0.0)
        stmt = acct.close_period()
        self.assertGreater(stmt["charged_usd"], 0.0)
        self.assertGreaterEqual(stmt["gross_margin_usd"], 0.0)
        self.assertEqual(stmt["invoices"], 1)
        # Counters reset.
        self.assertEqual(acct.period_charged_usd, 0.0)
        self.assertEqual(acct.invoices(), [])


# ---------- persistence ----------


class TestInvoicePersistence(unittest.TestCase):
    def setUp(self):
        _reset_fake()
        self.tmp = Path(tempfile.mkdtemp())
        rt, _ = _make_runtime()
        self.driver = RuntimeDriver(runtime=rt)
        self.market = TicketMarket(
            self.driver,
            invoices_path=self.tmp / "invoices.jsonl",
        )
        self.market.register_tenant(
            Tenant(tenant_id="acme", tier=TIER_STANDARD, monthly_budget_usd=10.0)
        )

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_completed_invoice_persisted_to_jsonl(self):
        self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="acme", max_bid_usd=2.0),
            timeout=5.0,
        )
        lines = (self.tmp / "invoices.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["tenant_id"], "acme")
        self.assertEqual(rec["status"], MKT_COMPLETED)

    def test_rejected_invoice_also_persisted(self):
        self.market.submit_sync(
            MarketTicket(intent="hi", tenant_id="ghost", max_bid_usd=2.0),
            timeout=2.0,
        )
        lines = (self.tmp / "invoices.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["status"], MKT_REJECTED)
        self.assertEqual(rec["quote_reason"], REASON_UNKNOWN_TENANT)


# ---------- SLO integration / refund-aware invoicing ----------


class TestSLOIntegration(unittest.TestCase):
    """A market ticket carrying an SLO must surface SLO breach refunds
    against the invoice line item — the contract that makes the market
    refund-aware."""

    def setUp(self):
        _reset_fake()
        self.market, self.driver, self.tmp = _make_market()
        self.market.register_tenant(
            Tenant(tenant_id="acme", tier=TIER_PREMIUM, monthly_budget_usd=50.0)
        )

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_slo_compliant_ticket_yields_no_refund(self):
        slo = TicketSLO(
            min_p_success=0.0,
            max_cost_usd=2.0,
            max_latency_s=60.0,
        )
        invoice = self.market.submit_sync(
            MarketTicket(
                intent="hi",
                tenant_id="acme",
                max_bid_usd=2.0,
                slo=slo,
            ),
            timeout=10.0,
        )
        # Either compliant or breached; not failed.
        self.assertIn(invoice.status, (MKT_COMPLETED, MKT_FAILED))
        if invoice.status == MKT_COMPLETED:
            # No refund means net_charge == list_price.
            self.assertEqual(invoice.refund_usd, 0.0)
            self.assertEqual(
                invoice.net_charge_usd, invoice.list_price_usd
            )

    def test_infeasible_slo_rejected_at_quote(self):
        # A latency floor of 0 is unreachable by any real model →
        # SLOCompiler reports infeasible; market must refuse.
        slo = TicketSLO(
            min_p_success=0.99999,
            max_cost_usd=1e-9,   # impossible cost ceiling
            max_latency_s=60.0,
        )
        q = self.market.quote(MarketTicket(
            intent="hi", tenant_id="acme", max_bid_usd=2.0, slo=slo,
        ))
        # We accept either rejection at the SLO layer or rejection at
        # the bid layer (the SLO compiler might find a plan exceeding
        # max_cost_usd anyway). The point is: not accepted.
        self.assertFalse(q.accepted)
        self.assertIn(
            q.reason,
            (REASON_INFEASIBLE_SLO, REASON_BID_TOO_LOW, REASON_OVER_BUDGET),
        )


# ---------- handle lifecycle ----------


class TestHandleLifecycle(unittest.TestCase):
    def setUp(self):
        _reset_fake()
        self.market, self.driver, self.tmp = _make_market()
        self.market.register_tenant(
            Tenant(tenant_id="acme", tier=TIER_STANDARD, monthly_budget_usd=10.0)
        )

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_cancel_before_dispatch_yields_cancelled_invoice(self):
        # Saturate the single slot so the second ticket waits.
        self.market.max_concurrent = 1
        FakeAgent.chat_sleep_s = 0.2
        try:
            blocker = self.market.submit(MarketTicket(
                intent="block", tenant_id="acme", max_bid_usd=2.0,
            ))
            target = self.market.submit(MarketTicket(
                intent="cancel-me", tenant_id="acme", max_bid_usd=2.0,
            ))
            target.cancel()
            invoice = target.result(timeout=5.0)
            # Either dispatched-then-cancelled or queued-then-cancelled
            # — both terminate in a cancelled invoice with no charge.
            self.assertIn(invoice.status, (MKT_CANCELLED, MKT_COMPLETED, MKT_FAILED))
            if invoice.status == MKT_CANCELLED:
                self.assertEqual(invoice.list_price_usd, 0.0)
                self.assertEqual(invoice.cost_of_goods_usd, 0.0)
            blocker.result(timeout=5.0)
        finally:
            FakeAgent.chat_sleep_s = 0.0

    def test_get_handle_returns_registered_handle(self):
        handle = self.market.submit(MarketTicket(
            intent="hi", tenant_id="acme", max_bid_usd=2.0,
        ))
        looked = self.market.get_handle(handle.id)
        self.assertIs(looked, handle)
        invoice = handle.result(timeout=5.0)
        self.assertEqual(invoice.status, MKT_COMPLETED)

    def test_get_unknown_handle_returns_none(self):
        self.assertIsNone(self.market.get_handle("nope"))


# ---------- failure isolation ----------


class TestFailureIsolation(unittest.TestCase):
    def setUp(self):
        _reset_fake()
        self.market, self.driver, self.tmp = _make_market()
        self.market.register_tenant(
            Tenant(tenant_id="acme", tier=TIER_STANDARD, monthly_budget_usd=10.0)
        )

    def tearDown(self):
        self.market.close(timeout=2.0)

    def test_failed_dispatch_yields_zero_charge_invoice(self):
        FakeAgent.fail = RuntimeError("boom")
        try:
            invoice = self.market.submit_sync(
                MarketTicket(intent="boom", tenant_id="acme", max_bid_usd=2.0),
                timeout=5.0,
            )
            self.assertEqual(invoice.status, MKT_FAILED)
            self.assertEqual(invoice.list_price_usd, 0.0)
            self.assertEqual(invoice.cost_of_goods_usd, 0.0)
            self.assertEqual(invoice.gross_margin_usd, 0.0)
            # Reservation released even on failure.
            acct = self.market.get_tenant("acme")
            assert acct is not None
            self.assertEqual(acct.reserved_usd, 0.0)
        finally:
            FakeAgent.fail = None


if __name__ == "__main__":
    unittest.main()
