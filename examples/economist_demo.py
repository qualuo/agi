"""TicketEconomist demo — closed-loop margin defender + scenario simulator.

The story this demo tells:

  The marketplace prices and bills work. The economist watches the
  marketplace's own economics, recommends adjustments when margins
  erode or refunds climb, and (optionally) applies those adjustments
  back to the market automatically. It also lets an operator stress-
  test the business with a one-line scenario before committing real
  spend.

Three scenes:

  1.  Healthy baseline      — a margin-positive workload, the economist
                              reports green and recommends nothing.
  2.  Margin erosion        — a cost shock pushes a tier underwater;
                              the economist surfaces a critical
                              adjustment and (in auto-pilot mode)
                              raises the markup.
  3.  Stress-test scenario  — what does a 2x traffic surge with a 15%
                              cost spike do to margins?

Uses FakeAgent — no API key required.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.driver import RuntimeDriver
from agi.economist import (
    ABS_MARKUP_CEILING,
    MarginTarget,
    Scenario,
    TicketEconomist,
)
from agi.market import (
    MKT_COMPLETED,
    Invoice,
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


# --- FakeAgent (no API key needed) -------------------------------------


class _FakeUsage:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        # Tunable per scene so we can demonstrate cost-driven margin erosion.
        return (
            self.input_tokens * FakeAgent.input_cost_per_token
            + self.output_tokens * 0.000025
        )


class FakeAgent:
    chat_sleep_s: float = 0.02
    input_cost_per_token: float = 0.000005  # baseline

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
        return "fake-response"

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = _FakeUsage()


# --- helpers -----------------------------------------------------------


def _inject(market, *, tenant_id, tier, list_price, refund, cost, status=MKT_COMPLETED):
    acct = market.get_tenant(tenant_id)
    assert acct is not None
    inv = Invoice(
        market_ticket_id=f"inv-{tenant_id}-{time.time_ns()}",
        tenant_id=tenant_id,
        tier=tier,
        intent="demo",
        status=status,
        list_price_usd=list_price,
        refund_usd=refund,
        net_charge_usd=max(0.0, list_price - refund),
        cost_of_goods_usd=cost,
        gross_margin_usd=max(0.0, list_price - refund) - cost,
        p_success=1.0,
        submitted_ts=time.time(),
        completed_ts=time.time(),
    )
    acct._record_invoice(inv, reserved=0.0)
    return inv


def _print_report(label, report):
    print(f"\n--- {label} ---")
    print(f"  score: {report.score:.2f}    healthy: {report.healthy}")
    overall = report.overall
    print(
        f"  revenue ${overall.revenue_usd:.4f}    "
        f"refunds ${overall.refunds_usd:.4f}    "
        f"cost ${overall.cost_of_goods_usd:.4f}    "
        f"margin ${overall.gross_margin_usd:.4f} "
        f"({overall.gross_margin_pct:.1%})"
    )
    if not report.adjustments:
        print("  no adjustments recommended.")
    for adj in report.adjustments:
        target = adj.tenant_id or adj.tier
        print(
            f"  [{adj.severity:8}] {adj.kind:18} → {target:12}  {adj.rationale}"
        )


# --- demo --------------------------------------------------------------


def main() -> None:
    tmp = Path("/tmp/ticket_economist_demo")
    tmp.mkdir(parents=True, exist_ok=True)
    rt = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
    )
    driver = RuntimeDriver(runtime=rt)
    market = TicketMarket(driver, max_concurrent=4)

    # Two tenants — one premium, one standard.
    market.register_tenant(Tenant(
        tenant_id="acme-corp",
        tier=TIER_PREMIUM,
        monthly_budget_usd=200.0,
        markup_pct=0.50,
    ))
    market.register_tenant(Tenant(
        tenant_id="bluefin",
        tier=TIER_STANDARD,
        monthly_budget_usd=50.0,
        markup_pct=0.30,
    ))

    economist = TicketEconomist(
        market,
        window_s=120.0,
        targets=[
            MarginTarget(tier=TIER_PREMIUM, gross_margin_pct_floor=0.25, refund_rate_ceiling=0.05),
            MarginTarget(tier=TIER_STANDARD, gross_margin_pct_floor=0.15, refund_rate_ceiling=0.12),
            MarginTarget(tier=TIER_ECONOMY, gross_margin_pct_floor=0.05, refund_rate_ceiling=0.20),
        ],
        adjustments_path=tmp / "adjustments.jsonl",
    )

    # --- Scene 1: healthy baseline ---
    print("\n========================================================")
    print(" Scene 1: healthy baseline — margins well above the floor")
    print("========================================================")
    for _ in range(6):
        _inject(market, tenant_id="acme-corp", tier=TIER_PREMIUM,
                list_price=1.0, refund=0.0, cost=0.4)
        _inject(market, tenant_id="bluefin", tier=TIER_STANDARD,
                list_price=0.5, refund=0.0, cost=0.3)
    _print_report("HEALTH (baseline)", economist.health())

    # --- Scene 2: cost shock erodes margin ---
    print("\n========================================================")
    print(" Scene 2: a cost shock pushes premium under its floor")
    print("========================================================")
    for _ in range(20):
        _inject(market, tenant_id="acme-corp", tier=TIER_PREMIUM,
                list_price=1.0, refund=0.0, cost=0.95)
    report = economist.health()
    _print_report("HEALTH (after cost shock)", report)

    # Auto-apply every recommended action.
    print("\n  applying recommended adjustments…")
    applied = economist.apply(report.adjustments)
    for ap in applied:
        target = ap.adjustment.tenant_id or ap.adjustment.tier
        if ap.applied:
            print(
                f"  ✓ {ap.adjustment.kind:14}  {target:14}  "
                f"before={ap.before}  after={ap.after}"
            )
        elif ap.error:
            print(f"  ✗ {ap.adjustment.kind:14}  {target:14}  error={ap.error}")
        else:
            print(
                f"  • {ap.adjustment.kind:14}  {target:14}  "
                f"no-op (target unchanged or dry-run)"
            )
    acme = market.get_tenant("acme-corp")
    print(f"\n  acme-corp.effective_markup_pct is now {acme.tenant.effective_markup_pct:.2f}")

    # --- Scene 3: stress-test scenario ---
    print("\n========================================================")
    print(" Scene 3: stress-test scenarios — what-if without spend")
    print("========================================================")
    for label, scenario in [
        ("baseline", Scenario(traffic_multiplier=1.0, cost_multiplier=1.0, duration_s=120.0)),
        ("2x traffic", Scenario(traffic_multiplier=2.0, cost_multiplier=1.0, duration_s=120.0)),
        ("15% cost shock", Scenario(traffic_multiplier=1.0, cost_multiplier=1.15, duration_s=120.0)),
        ("2x traffic + 15% cost shock", Scenario(
            traffic_multiplier=2.0, cost_multiplier=1.15, duration_s=120.0,
        )),
        ("50% refund rate (SLO outage)", Scenario(
            traffic_multiplier=1.0, cost_multiplier=1.0, refund_rate=0.5,
            duration_s=120.0,
        )),
    ]:
        sim = economist.simulate(scenario)
        print(
            f"\n  [{label:32}] "
            f"revenue ${sim.projected_revenue_usd:7.2f}  "
            f"refunds ${sim.projected_refunds_usd:6.2f}  "
            f"margin ${sim.projected_gross_margin_usd:6.2f} "
            f"({sim.projected_gross_margin_pct:5.1%})  "
            f"actions={len(sim.actions_recommended)}"
        )

    # --- audit ledger ---
    ledger = tmp / "adjustments.jsonl"
    if ledger.exists() and ledger.stat().st_size > 0:
        print("\n========================================================")
        print(f" Audit ledger ({ledger}):")
        print("========================================================")
        for line in ledger.read_text().strip().splitlines()[-5:]:
            obj = json.loads(line)
            adj = obj["adjustment"]
            print(
                f"  [{adj['severity']:8}] {adj['kind']:18}  "
                f"tenant={adj.get('tenant_id'):16}  applied={obj['applied']}"
            )

    economist.close()
    market.close(timeout=2.0)


if __name__ == "__main__":
    main()
