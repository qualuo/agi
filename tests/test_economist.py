"""Tests for TicketEconomist — the closed-loop margin defender.

These pin the contract a coordination engine drives when it hands the
runtime authority over its own marketplace economics: telemetry
rollups, adjustment generation, dry-run vs apply, scenario simulation,
and the auto-pilot control loop.

Fully hermetic — no Anthropic API. Reuses the FakeAgent harness from
test_market.
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

from agi.driver import RuntimeDriver
from agi.economist import (
    ABS_MARKUP_CEILING,
    ABS_MARKUP_FLOOR,
    ADJ_PAUSE_TENANT,
    ADJ_RAISE_MARKUP,
    ADJ_RESUME_TENANT,
    DEFAULT_MARGIN_FLOORS,
    ECON_ADJUSTMENT_APPLIED,
    ECON_HEALTH_REPORTED,
    SEV_CRITICAL,
    SEV_INFO,
    SEV_WARN,
    AppliedAdjustment,
    EconomicWindow,
    HealthReport,
    MarginTarget,
    PolicyAdjustment,
    Scenario,
    SimulationResult,
    TicketEconomist,
)
from agi.events import EventBus
from agi.market import (
    KNOWN_TIERS,
    MKT_COMPLETED,
    MKT_FAILED,
    MKT_REJECTED,
    TIER_ECONOMY,
    TIER_PREMIUM,
    TIER_STANDARD,
    Invoice,
    MarketTicket,
    Tenant,
    TicketMarket,
)
from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


# ---------- fakes (re-used pattern from test_market) ----------


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

    def __init__(self, *, memory=None, model="claude-opus-4-7", **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.extra_system: str | None = None
        self.messages: list[Any] = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        if FakeAgent.fail is not None:
            raise FakeAgent.fail
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        return FakeAgent.response

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = FakeUsage()


def _make_market(**market_kw) -> tuple[TicketMarket, RuntimeDriver, Path]:
    tmp = Path(tempfile.mkdtemp())
    rt = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
    )
    driver = RuntimeDriver(runtime=rt)
    market = TicketMarket(driver, **market_kw)
    return market, driver, tmp


# ---------- direct-invoice helpers ----------
#
# Some tests need to inject invoices into a market without round-
# tripping through the runtime. We do that by registering a tenant
# and pushing an Invoice into its account directly — the economist
# reads only what the public market surface exposes.


def _inject_invoice(
    market: TicketMarket,
    *,
    tenant_id: str,
    tier: str,
    list_price: float,
    refund: float,
    cost: float,
    status: str = MKT_COMPLETED,
    completed_ts: float | None = None,
) -> Invoice:
    acct = market.get_tenant(tenant_id)
    assert acct is not None
    inv = Invoice(
        market_ticket_id=f"inv-{tenant_id}-{time.time_ns()}",
        tenant_id=tenant_id,
        tier=tier,
        intent="test",
        status=status,
        list_price_usd=list_price,
        refund_usd=refund,
        net_charge_usd=max(0.0, list_price - refund),
        cost_of_goods_usd=cost,
        gross_margin_usd=max(0.0, list_price - refund) - cost,
        p_success=1.0,
        submitted_ts=completed_ts or time.time(),
        completed_ts=completed_ts or time.time(),
    )
    acct._record_invoice(inv, reserved=0.0)
    return inv


# ---------- target validation ----------


class TestMarginTargetValidation(unittest.TestCase):
    def test_unknown_tier_rejected(self):
        with self.assertRaises(ValueError):
            MarginTarget(tier="diamond", gross_margin_pct_floor=0.2, refund_rate_ceiling=0.1)

    def test_margin_floor_bounds(self):
        with self.assertRaises(ValueError):
            MarginTarget(tier=TIER_PREMIUM, gross_margin_pct_floor=-0.1, refund_rate_ceiling=0.1)
        with self.assertRaises(ValueError):
            MarginTarget(tier=TIER_PREMIUM, gross_margin_pct_floor=1.1, refund_rate_ceiling=0.1)

    def test_refund_ceiling_bounds(self):
        with self.assertRaises(ValueError):
            MarginTarget(tier=TIER_PREMIUM, gross_margin_pct_floor=0.2, refund_rate_ceiling=-0.1)
        with self.assertRaises(ValueError):
            MarginTarget(tier=TIER_PREMIUM, gross_margin_pct_floor=0.2, refund_rate_ceiling=1.1)

    def test_min_invoices_must_be_positive(self):
        with self.assertRaises(ValueError):
            MarginTarget(
                tier=TIER_PREMIUM,
                gross_margin_pct_floor=0.2,
                refund_rate_ceiling=0.1,
                min_invoices=0,
            )


# ---------- economic window math ----------


class TestEconomicWindow(unittest.TestCase):
    def setUp(self):
        self.market, _, _ = _make_market()
        self.market.register_tenant(Tenant(tenant_id="t1", tier=TIER_STANDARD))
        for _ in range(4):
            _inject_invoice(
                self.market,
                tenant_id="t1",
                tier=TIER_STANDARD,
                list_price=1.0,
                refund=0.1,
                cost=0.5,
            )
        # One refund-bearing failure on top
        _inject_invoice(
            self.market,
            tenant_id="t1",
            tier=TIER_STANDARD,
            list_price=0.0,
            refund=0.0,
            cost=0.2,
            status=MKT_FAILED,
        )
        self.econ = TicketEconomist(self.market)

    def tearDown(self):
        self.econ.close()
        self.market.close(timeout=2.0)

    def test_revenue_refunds_and_cost_roll_up(self):
        report = self.econ.health()
        self.assertAlmostEqual(report.overall.revenue_usd, 4.0, places=6)
        self.assertAlmostEqual(report.overall.refunds_usd, 0.4, places=6)
        self.assertAlmostEqual(report.overall.cost_of_goods_usd, 4 * 0.5 + 0.2, places=6)

    def test_net_revenue_and_margin(self):
        report = self.econ.health()
        self.assertAlmostEqual(report.overall.net_revenue_usd, 3.6, places=6)
        # margin = net - cost = 3.6 - 2.2 = 1.4
        self.assertAlmostEqual(report.overall.gross_margin_usd, 1.4, places=6)

    def test_refund_rate_counts_only_completed(self):
        report = self.econ.health()
        # 4 completed, all of which had refund > 0, so refund_rate = 1.0
        self.assertAlmostEqual(report.overall.refund_rate, 1.0, places=6)

    def test_failure_rate_counts_failures_over_total(self):
        report = self.econ.health()
        self.assertAlmostEqual(report.overall.failure_rate, 1 / 5, places=6)


# ---------- adjustment generation ----------


class TestAdjustmentGeneration(unittest.TestCase):
    def setUp(self):
        self.market, _, _ = _make_market()
        self.market.register_tenant(Tenant(tenant_id="acme", tier=TIER_STANDARD))
        self.econ = TicketEconomist(
            self.market,
            window_s=60.0,
            targets=[
                MarginTarget(
                    tier=TIER_STANDARD,
                    gross_margin_pct_floor=0.30,
                    refund_rate_ceiling=0.20,
                    min_invoices=5,
                ),
            ],
        )

    def tearDown(self):
        self.econ.close()
        self.market.close(timeout=2.0)

    def test_no_adjustments_below_sample_threshold(self):
        for _ in range(4):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.9,  # razor-thin margin
            )
        report = self.econ.health()
        self.assertEqual(report.adjustments, [])

    def test_low_margin_triggers_critical_tier_raise(self):
        for _ in range(6):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.9,
            )
        report = self.econ.health()
        kinds = {(a.kind, a.tier) for a in report.adjustments}
        self.assertIn((ADJ_RAISE_MARKUP, TIER_STANDARD), kinds)
        critical = [a for a in report.adjustments if a.severity == SEV_CRITICAL]
        self.assertTrue(critical, "expected at least one critical adjustment")

    def test_high_refund_triggers_warn(self):
        for i in range(8):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.5 if i < 5 else 0.0, cost=0.3,
            )
        # refund rate among completed = 5/8 = 0.625, well above 0.20 ceiling
        # margin is still healthy at this cost basis
        report = self.econ.health()
        # Should at minimum surface a refund-rate adjustment for the tenant.
        kinds = {a.kind for a in report.adjustments}
        self.assertIn(ADJ_RAISE_MARKUP, kinds)
        self.assertTrue(
            any(a.tenant_id == "acme" or a.tier == TIER_STANDARD for a in report.adjustments)
        )

    def test_unprofitable_tenant_is_paused(self):
        for _ in range(6):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=1.5,  # margin < 0
            )
        report = self.econ.health()
        kinds = {(a.kind, a.tenant_id) for a in report.adjustments}
        self.assertIn((ADJ_PAUSE_TENANT, "acme"), kinds)
        crit = [a for a in report.adjustments
                if a.kind == ADJ_PAUSE_TENANT and a.severity == SEV_CRITICAL]
        self.assertTrue(crit)

    def test_recovered_tenant_recommended_for_resume(self):
        # Pause the tenant up front, then write healthy invoices.
        self.market.pause_tenant("acme")
        for _ in range(6):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.3,
            )
        report = self.econ.health()
        kinds = {(a.kind, a.tenant_id, a.severity) for a in report.adjustments}
        self.assertIn((ADJ_RESUME_TENANT, "acme", SEV_INFO), kinds)


# ---------- apply / dry-run ----------


class TestApply(unittest.TestCase):
    def setUp(self):
        self.market, _, _ = _make_market()
        self.market.register_tenant(Tenant(
            tenant_id="acme",
            tier=TIER_STANDARD,
            markup_pct=0.30,
        ))
        self.econ = TicketEconomist(self.market)

    def tearDown(self):
        self.econ.close()
        self.market.close(timeout=2.0)

    def test_apply_raise_markup_mutates_tenant(self):
        adj = PolicyAdjustment(
            id="x1",
            kind=ADJ_RAISE_MARKUP,
            tenant_id="acme",
            tier=TIER_STANDARD,
            severity=SEV_WARN,
            rationale="test",
            current_markup_pct=0.30,
            suggested_markup_pct=0.55,
        )
        results = self.econ.apply([adj])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].applied)
        self.assertIsNone(results[0].error)
        acct = self.market.get_tenant("acme")
        assert acct is not None
        self.assertAlmostEqual(acct.tenant.markup_pct, 0.55, places=6)

    def test_dry_run_does_not_mutate(self):
        adj = PolicyAdjustment(
            id="x2",
            kind=ADJ_RAISE_MARKUP,
            tenant_id="acme",
            tier=TIER_STANDARD,
            severity=SEV_WARN,
            rationale="test",
            current_markup_pct=0.30,
            suggested_markup_pct=0.55,
        )
        results = self.econ.apply([adj], dry_run=True)
        self.assertFalse(results[0].applied)
        acct = self.market.get_tenant("acme")
        assert acct is not None
        self.assertAlmostEqual(acct.tenant.markup_pct, 0.30, places=6)

    def test_apply_pause_then_resume(self):
        pause = PolicyAdjustment(
            id="p1", kind=ADJ_PAUSE_TENANT, tenant_id="acme",
            tier=TIER_STANDARD, severity=SEV_CRITICAL, rationale="test",
        )
        results = self.econ.apply([pause])
        self.assertTrue(results[0].applied)
        self.assertTrue(self.market.get_tenant("acme").tenant.paused)
        resume = PolicyAdjustment(
            id="r1", kind=ADJ_RESUME_TENANT, tenant_id="acme",
            tier=TIER_STANDARD, severity=SEV_INFO, rationale="test",
        )
        results = self.econ.apply([resume])
        self.assertTrue(results[0].applied)
        self.assertFalse(self.market.get_tenant("acme").tenant.paused)

    def test_unknown_tenant_records_error(self):
        adj = PolicyAdjustment(
            id="x", kind=ADJ_RAISE_MARKUP, tenant_id="ghost",
            tier=TIER_STANDARD, severity=SEV_WARN, rationale="test",
            suggested_markup_pct=0.40,
        )
        res = self.econ.apply([adj])
        self.assertFalse(res[0].applied)
        self.assertIsNotNone(res[0].error)
        self.assertIn("KeyError", res[0].error)

    def test_markup_capped_at_ceiling(self):
        adj = PolicyAdjustment(
            id="x", kind=ADJ_RAISE_MARKUP, tenant_id="acme",
            tier=TIER_STANDARD, severity=SEV_CRITICAL, rationale="test",
            suggested_markup_pct=999.0,
        )
        self.econ.apply([adj])
        acct = self.market.get_tenant("acme")
        self.assertAlmostEqual(acct.tenant.markup_pct, ABS_MARKUP_CEILING, places=6)

    def test_markup_floor_enforced(self):
        adj = PolicyAdjustment(
            id="x", kind="lower_markup", tenant_id="acme",
            tier=TIER_STANDARD, severity=SEV_INFO, rationale="test",
            suggested_markup_pct=-1.0,
        )
        self.econ.apply([adj])
        acct = self.market.get_tenant("acme")
        self.assertAlmostEqual(acct.tenant.markup_pct, ABS_MARKUP_FLOOR, places=6)

    def test_applied_history_records_results(self):
        adj = PolicyAdjustment(
            id="x", kind=ADJ_RAISE_MARKUP, tenant_id="acme",
            tier=TIER_STANDARD, severity=SEV_WARN, rationale="test",
            suggested_markup_pct=0.55,
        )
        self.econ.apply([adj])
        history = self.econ.applied_history()
        self.assertEqual(len(history), 1)
        self.assertTrue(history[0].applied)

    def test_persistence_appends_jsonl(self):
        tmp = Path(tempfile.mkdtemp()) / "adj.jsonl"
        market2, _, _ = _make_market()
        try:
            market2.register_tenant(Tenant(tenant_id="acme", tier=TIER_STANDARD))
            econ = TicketEconomist(market2, adjustments_path=tmp)
            adj = PolicyAdjustment(
                id="x", kind=ADJ_RAISE_MARKUP, tenant_id="acme",
                tier=TIER_STANDARD, severity=SEV_WARN, rationale="test",
                suggested_markup_pct=0.55,
            )
            econ.apply([adj])
            lines = tmp.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertTrue(obj["applied"])
            self.assertEqual(obj["adjustment"]["kind"], ADJ_RAISE_MARKUP)
            econ.close()
        finally:
            market2.close(timeout=2.0)


# ---------- simulation ----------


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.market, _, _ = _make_market()
        self.market.register_tenant(Tenant(
            tenant_id="acme",
            tier=TIER_STANDARD,
            markup_pct=0.30,
        ))
        self.econ = TicketEconomist(self.market, window_s=600.0)

    def tearDown(self):
        self.econ.close()
        self.market.close(timeout=2.0)

    def test_simulation_with_empty_baseline_returns_zeros(self):
        sim = self.econ.simulate(Scenario(duration_s=3600.0))
        self.assertEqual(sim.baseline_invoice_count, 0)
        self.assertEqual(sim.projected_invoice_count, 0)
        self.assertEqual(sim.projected_revenue_usd, 0.0)

    def test_traffic_2x_doubles_projection(self):
        for _ in range(10):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.5,
            )
        single = self.econ.simulate(Scenario(
            traffic_multiplier=1.0,
            duration_s=self.econ.window_s,
        ))
        doubled = self.econ.simulate(Scenario(
            traffic_multiplier=2.0,
            duration_s=self.econ.window_s,
        ))
        self.assertAlmostEqual(
            doubled.projected_revenue_usd,
            single.projected_revenue_usd * 2,
            places=4,
        )

    def test_cost_multiplier_erodes_margin(self):
        for _ in range(10):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.5,
            )
        base = self.econ.simulate(Scenario(
            cost_multiplier=1.0,
            duration_s=self.econ.window_s,
        ))
        bumped = self.econ.simulate(Scenario(
            cost_multiplier=1.5,
            duration_s=self.econ.window_s,
        ))
        self.assertLess(
            bumped.projected_gross_margin_pct,
            base.projected_gross_margin_pct,
        )

    def test_forced_refund_rate_drops_margin(self):
        for _ in range(10):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.5,
            )
        base = self.econ.simulate(Scenario(duration_s=self.econ.window_s))
        breached = self.econ.simulate(Scenario(
            refund_rate=0.5,
            duration_s=self.econ.window_s,
        ))
        self.assertLess(
            breached.projected_gross_margin_pct,
            base.projected_gross_margin_pct,
        )
        self.assertGreater(breached.projected_refunds_usd, base.projected_refunds_usd)

    def test_simulation_returns_per_tier_projections(self):
        for _ in range(10):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.5,
            )
        sim = self.econ.simulate(Scenario(duration_s=self.econ.window_s))
        self.assertIn(TIER_STANDARD, sim.per_tier)
        self.assertGreater(sim.per_tier[TIER_STANDARD]["revenue_usd"], 0)

    def test_simulation_publishes_action_for_underwater_tier(self):
        for _ in range(10):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.95,  # very thin
            )
        sim = self.econ.simulate(Scenario(
            cost_multiplier=1.3,
            duration_s=self.econ.window_s,
        ))
        self.assertTrue(
            any(a.tier == TIER_STANDARD for a in sim.actions_recommended),
            f"expected a tier-{TIER_STANDARD} action, got {sim.actions_recommended}",
        )

    def test_to_dict_round_trips(self):
        for _ in range(6):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=0.5,
            )
        sim = self.econ.simulate(Scenario(duration_s=self.econ.window_s))
        d = sim.to_dict()
        # JSON-serializable end to end
        json.dumps(d, default=str)
        self.assertIn("projected_gross_margin_pct", d)


# ---------- autopilot ----------


class TestAutoPilot(unittest.TestCase):
    def setUp(self):
        self.market, _, _ = _make_market()
        self.market.register_tenant(Tenant(
            tenant_id="acme",
            tier=TIER_STANDARD,
            markup_pct=0.30,
        ))
        self.econ = TicketEconomist(
            self.market,
            window_s=60.0,
            control_interval_s=0.05,
        )

    def tearDown(self):
        self.econ.close()
        self.market.close(timeout=2.0)

    def test_auto_pilot_idempotent_enable(self):
        self.econ.auto_pilot(enable=True)
        first = self.econ._auto_pilot_thread
        self.econ.auto_pilot(enable=True)
        self.assertIs(self.econ._auto_pilot_thread, first)
        self.econ.auto_pilot(enable=False)

    def test_auto_pilot_applies_critical_adjustment(self):
        # Margin underwater on standard tier
        for _ in range(8):
            _inject_invoice(
                self.market, tenant_id="acme", tier=TIER_STANDARD,
                list_price=1.0, refund=0.0, cost=1.5,
            )
        self.econ.auto_pilot(enable=True)
        # Wait briefly for the loop to run at least one full tick.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.market.get_tenant("acme").tenant.paused:
                break
            time.sleep(0.05)
        self.econ.auto_pilot(enable=False)
        self.assertTrue(self.market.get_tenant("acme").tenant.paused)

    def test_close_stops_autopilot(self):
        self.econ.auto_pilot(enable=True)
        self.econ.close(timeout=2.0)
        self.assertFalse(self.econ.autopilot_enabled)


# ---------- event bus integration ----------


class TestEventBusIntegration(unittest.TestCase):
    def test_health_publishes_event_when_bus_attached(self):
        market, _, _ = _make_market()
        try:
            market.register_tenant(Tenant(tenant_id="acme", tier=TIER_STANDARD))
            bus = EventBus()
            received: list[Any] = []
            bus.subscribe(lambda e: received.append(e), kind=ECON_HEALTH_REPORTED)
            econ = TicketEconomist(market, bus=bus)
            econ.health()
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].kind, ECON_HEALTH_REPORTED)
            econ.close()
        finally:
            market.close(timeout=2.0)

    def test_apply_publishes_event(self):
        market, _, _ = _make_market()
        try:
            market.register_tenant(Tenant(
                tenant_id="acme", tier=TIER_STANDARD, markup_pct=0.30,
            ))
            bus = EventBus()
            received: list[Any] = []
            bus.subscribe(lambda e: received.append(e), kind=ECON_ADJUSTMENT_APPLIED)
            econ = TicketEconomist(market, bus=bus)
            adj = PolicyAdjustment(
                id="x", kind=ADJ_RAISE_MARKUP, tenant_id="acme",
                tier=TIER_STANDARD, severity=SEV_WARN, rationale="test",
                suggested_markup_pct=0.55,
            )
            econ.apply([adj])
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].kind, ECON_ADJUSTMENT_APPLIED)
            econ.close()
        finally:
            market.close(timeout=2.0)


# ---------- health score ----------


class TestHealthScore(unittest.TestCase):
    def test_score_high_when_well_above_floor(self):
        market, _, _ = _make_market()
        try:
            market.register_tenant(Tenant(tenant_id="acme", tier=TIER_STANDARD))
            for _ in range(6):
                _inject_invoice(
                    market, tenant_id="acme", tier=TIER_STANDARD,
                    list_price=1.0, refund=0.0, cost=0.1,  # 90% margin
                )
            econ = TicketEconomist(market)
            report = econ.health()
            self.assertTrue(report.healthy)
            self.assertGreater(report.score, 0.5)
            econ.close()
        finally:
            market.close(timeout=2.0)

    def test_score_low_when_below_floor(self):
        market, _, _ = _make_market()
        try:
            market.register_tenant(Tenant(tenant_id="acme", tier=TIER_STANDARD))
            for _ in range(6):
                _inject_invoice(
                    market, tenant_id="acme", tier=TIER_STANDARD,
                    list_price=1.0, refund=0.0, cost=0.95,
                )
            econ = TicketEconomist(market)
            report = econ.health()
            self.assertFalse(report.healthy)
            self.assertLess(report.score, 0.5)
            econ.close()
        finally:
            market.close(timeout=2.0)

    def test_no_data_treated_as_healthy(self):
        market, _, _ = _make_market()
        try:
            econ = TicketEconomist(market)
            report = econ.health()
            self.assertTrue(report.healthy)
            self.assertEqual(report.score, 1.0)
            econ.close()
        finally:
            market.close(timeout=2.0)


# ---------- end-to-end market integration ----------


class TestEndToEndMarketIntegration(unittest.TestCase):
    """A full round-trip: tickets dispatch through the real market, the
    economist sees the real invoices the market wrote, and produces an
    actionable adjustment when the cost model dictates it.

    This catches subtle wiring issues where the economist would miss
    invoices the market actually produced.
    """

    def test_economist_sees_dispatched_invoices(self):
        market, driver, _ = _make_market()
        try:
            market.register_tenant(Tenant(
                tenant_id="acme",
                tier=TIER_STANDARD,
                monthly_budget_usd=100.0,
                markup_pct=0.30,
            ))
            handles = []
            for _ in range(6):
                handles.append(market.submit(MarketTicket(
                    intent="summarize document",
                    tenant_id="acme",
                    max_bid_usd=5.0,
                )))
            for h in handles:
                h.result(timeout=10.0)
            econ = TicketEconomist(market, window_s=300.0)
            report = econ.health()
            self.assertGreater(report.overall.invoices_count, 0)
            econ.close()
        finally:
            market.close(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
