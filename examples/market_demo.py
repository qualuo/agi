"""TicketMarket demo — multi-tenant marketplace on the runtime.

The story this demo tells:

  A coordination engine fronting many tenants doesn't want to wire
  pricing, quotas, fair scheduling, and billing per tenant. It hands
  the work to one `TicketMarket`. The market:

    1. Quotes every ticket — cost forecast × tenant-tier markup.
    2. Refuses business that won't be profitable or that would
       overrun the tenant's monthly quota.
    3. Schedules tier-fairly: premium > standard > economy under
       contention.
    4. Settles refunds from SLO breaches against the invoice line
       automatically.
    5. Reports revenue, cost-of-goods, and gross margin in one call.

Three scenes:

  1.  Quoting before dispatch — surface the price to the tenant.
  2.  Mixed-tier flood        — observe premium preempt economy.
  3.  Refund-aware SLO ticket — an SLO breach refunds against the bill.

Finally we print the market-wide rollup and per-tenant statements —
the dashboard view an operator (and an investor) actually wants.

Uses FakeAgent — no API key required.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.contract import TicketSLO
from agi.driver import RuntimeDriver
from agi.market import (
    MarketTicket,
    Tenant,
    TicketMarket,
    TIER_ECONOMY,
    TIER_PREMIUM,
    TIER_STANDARD,
)
from agi.memory import Memory
from agi.runtime import Runtime
from agi.skills import SkillLibrary


# --- FakeAgent ----------------------------------------------------------


class _FakeUsage:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    chat_sleep_s: float = 0.05

    def __init__(self, *, memory=None, model="claude-opus-4-7", critic_threshold=0.5, **kw):
        self.memory = memory
        self.model = model
        self.usage = _FakeUsage()
        self.last_critic_score = None
        self.extra_system = None
        self.messages = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        time.sleep(FakeAgent.chat_sleep_s)
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        return f"answer:{prompt[:24]}"

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = _FakeUsage()


# --- helpers ------------------------------------------------------------


def _hr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _dollars(x: float) -> str:
    return f"${x:8.5f}"


# --- demo ---------------------------------------------------------------


def main() -> int:
    runtime = Runtime(
        memory=Memory(path=Path("/tmp/market_demo_memory.jsonl")),
        skills=SkillLibrary(path=Path("/tmp/market_demo_skills")),
        agent_factory=FakeAgent,
    )
    driver = RuntimeDriver(runtime=runtime, max_concurrent=4)
    market = TicketMarket(driver, max_concurrent=2)

    # Three tenants on three tiers, each with a monthly budget cap.
    market.register_tenant(Tenant(
        tenant_id="acme",       tier=TIER_PREMIUM,  monthly_budget_usd=10.0,
    ))
    market.register_tenant(Tenant(
        tenant_id="globex",     tier=TIER_STANDARD, monthly_budget_usd=10.0,
    ))
    market.register_tenant(Tenant(
        tenant_id="initech",    tier=TIER_ECONOMY,  monthly_budget_usd=10.0,
    ))

    # --- scene 1: quoting ---------------------------------------------

    _hr("SCENE 1 — quote a ticket before spending")
    for tenant_id in ("acme", "globex", "initech"):
        q = market.quote(MarketTicket(
            intent="summarize Q4 board minutes",
            tenant_id=tenant_id,
            max_bid_usd=0.50,
        ))
        print(
            f"  {tenant_id:8s} tier={q.tier:8s}  "
            f"cost={_dollars(q.estimated_cost_usd)} "
            f"list={_dollars(q.list_price_usd)} "
            f"margin={_dollars(q.margin_usd)}  "
            f"accepted={q.accepted}"
        )

    # --- scene 2: mixed-tier flood ------------------------------------

    _hr("SCENE 2 — flood: economy and premium contend for slots")
    handles = []
    # Submit 3 economy first, then 3 premium. With FIFO they'd finish
    # in submission order. The market's tier-weighted scheduler picks
    # premium first under contention.
    for i in range(3):
        handles.append(market.submit(MarketTicket(
            intent=f"economy-task-{i}", tenant_id="initech", max_bid_usd=0.50,
        )))
    time.sleep(0.02)
    for i in range(3):
        handles.append(market.submit(MarketTicket(
            intent=f"premium-task-{i}", tenant_id="acme", max_bid_usd=0.50,
        )))
    invoices = [h.result(timeout=15.0) for h in handles]
    invoices.sort(key=lambda inv: inv.completed_ts or 0.0)
    print("  Completion order (submitted: 3 econ, 3 prem):")
    for inv in invoices:
        ms = (inv.completed_ts - inv.submitted_ts) * 1000.0
        print(
            f"    t+{ms:6.1f}ms  {inv.tier:8s}  "
            f"{inv.tenant_id:8s}  {inv.intent[:24]:24s}  "
            f"net={_dollars(inv.net_charge_usd)} "
            f"margin={_dollars(inv.gross_margin_usd)}"
        )

    # --- scene 3: refund-aware SLO ticket -----------------------------

    _hr("SCENE 3 — SLO ticket: refund flows into the invoice")
    slo = TicketSLO(
        min_p_success=0.0,
        max_cost_usd=0.50,
        max_latency_s=30.0,
        refund_on_breach=1.0,
    )
    inv = market.submit_sync(
        MarketTicket(
            intent="generate a quarterly forecast",
            tenant_id="globex",
            max_bid_usd=0.50,
            slo=slo,
        ),
        timeout=10.0,
    )
    print(f"  status      = {inv.status}")
    print(f"  slo_status  = {inv.slo_status}")
    print(f"  list_price  = {_dollars(inv.list_price_usd)}")
    print(f"  refund      = {_dollars(inv.refund_usd)}")
    print(f"  net_charge  = {_dollars(inv.net_charge_usd)}")
    print(f"  cost-of-goods = {_dollars(inv.cost_of_goods_usd)}")
    print(f"  gross_margin  = {_dollars(inv.gross_margin_usd)}")

    # --- rollup --------------------------------------------------------

    _hr("MARKET ROLLUP — what an operator dashboards")
    stats = market.market_stats()
    print(f"  submitted        : {stats['submitted']}")
    print(f"  completed        : {stats['completed']}")
    print(f"  rejected         : {stats['rejected']}")
    print(f"  revenue          : {_dollars(stats['revenue_usd'])}")
    print(f"  refunds          : {_dollars(stats['refunds_usd'])}")
    print(f"  net_revenue      : {_dollars(stats['net_revenue_usd'])}")
    print(f"  cost_of_goods    : {_dollars(stats['cost_of_goods_usd'])}")
    print(f"  gross_margin     : {_dollars(stats['gross_margin_usd'])}")
    print(f"  gross_margin_pct : {stats['gross_margin_pct'] * 100.0:5.2f}%")
    print(f"  queued_total     : {stats['queued_total']}")
    print(f"  in_flight_total  : {stats['in_flight_total']}")

    _hr("PER-TENANT STATEMENT")
    for s in stats["per_tenant"]:
        print(
            f"  {s['tenant_id']:8s} tier={s['tier']:8s} "
            f"submitted={s['submitted']:2d}  "
            f"net={_dollars(s['net_charged_usd'])}  "
            f"margin={_dollars(s['gross_margin_usd'])}  "
            f"quota_left={_dollars(s['quota_remaining_usd'])}"
        )

    market.close(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
