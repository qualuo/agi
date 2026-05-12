"""TicketEconomist — closed-loop margin defender + scenario simulator.

The marketplace prices, schedules, and bills work. The economist is the
control plane that watches the marketplace's own economics and acts on
them. A coordination engine plugs the economist in once and gets:

  - a per-tier and per-tenant rolling read on revenue, refunds, cost of
    goods, gross margin, refund rate, and SLO-breach rate,
  - structured `PolicyAdjustment` recommendations whenever margins
    erode below a configured floor or refund rates climb above a
    configured ceiling,
  - an optional auto-pilot that applies those adjustments back to the
    market (raises a tenant's markup, pauses a chronically lossy
    tenant, tightens default downgrade behavior),
  - a `simulate(scenario)` what-if engine that projects a different
    traffic mix, cost curve, or refund rate forward and reports the
    margin/queue/breach impact before any real spend happens,
  - an append-only JSONL audit ledger of every adjustment the
    economist made, so finance and compliance can reconcile.

    economist = TicketEconomist(market)
    economist.auto_pilot(enable=True)              # run the control loop

    report = economist.health()
    for adj in report.adjustments:
        print(adj.rationale, adj.severity)

    sim = economist.simulate(Scenario(
        traffic_multiplier=2.0,
        cost_multiplier=1.15,
        duration_s=3600.0,
    ))
    print(sim.projected_margin_pct, sim.actions_recommended)

This is the line between "we sell AI" and "we operate a managed AI
business that defends its own gross margin." Without it, every margin
decision requires an on-call human; with it, the runtime polices
itself.

Investor framing
----------------
- **Self-defending margins.** Per-tier floors trigger pricing or
  routing adjustments automatically. The platform never silently runs
  unprofitable.
- **Refund-aware risk premium.** Chronically high-refund tenants pay a
  higher markup. Their cost to us is encoded in their price.
- **What-if capacity planning.** Operators and investors can ask
  "what happens if traffic 2x's and OpenAI raises prices 20%?"
  without booking real spend.
- **Audit ledger.** Every adjustment is JSONL-persisted with the
  evidence that triggered it. Finance/regulators can reconstruct any
  decision.

The economist has no LLM dependency. It reads invoices the market
already wrote, makes deterministic decisions from rolling windows,
and pushes mutations back through the market's existing public
surface (`pause_tenant`, mutating `Tenant.markup_pct`, etc.).
"""
from __future__ import annotations

import bisect
import json
import math
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from agi.events import Event, EventBus
from agi.market import (
    KNOWN_TIERS,
    MKT_COMPLETED,
    MKT_FAILED,
    MKT_REJECTED,
    TIER_DEFAULT_MARKUP,
    TIER_ECONOMY,
    TIER_PREMIUM,
    TIER_STANDARD,
    Invoice,
    Tenant,
    TicketMarket,
)


# --- constants ----------------------------------------------------------

# Adjustment kinds. Strings so a coordination engine can pattern-match
# without importing this module.
ADJ_RAISE_MARKUP = "raise_markup"
ADJ_LOWER_MARKUP = "lower_markup"
ADJ_PAUSE_TENANT = "pause_tenant"
ADJ_RESUME_TENANT = "resume_tenant"
ADJ_INCREASE_BUDGET = "raise_tenant_budget"
ADJ_NO_ACTION = "no_action"

# Severity levels surfaced on every adjustment.
SEV_INFO = "info"
SEV_WARN = "warn"
SEV_CRITICAL = "critical"

# Default per-tier targets a coordination engine inherits unless
# overrides are passed. Premium pays the highest markup, so premium
# expects the strongest floor; economy operates on a thin margin and
# is allowed a higher refund rate.
DEFAULT_MARGIN_FLOORS: dict[str, float] = {
    TIER_PREMIUM: 0.25,
    TIER_STANDARD: 0.15,
    TIER_ECONOMY: 0.05,
}

DEFAULT_REFUND_CEILINGS: dict[str, float] = {
    TIER_PREMIUM: 0.05,
    TIER_STANDARD: 0.12,
    TIER_ECONOMY: 0.20,
}

# How much to bump a markup per critical adjustment. Premium gets the
# biggest absolute bump because it already pays the biggest markup.
DEFAULT_MARKUP_BUMP_PCT = 0.10

# Floor for how low auto-pilot will ever drop a markup, even when a
# tenant is consistently profitable and a price cut would be safe.
ABS_MARKUP_FLOOR = 0.05

# Maximum markup auto-pilot will ever apply, even for chronically
# expensive tenants. Beyond this, a human must approve.
ABS_MARKUP_CEILING = 2.0


# --- event kinds (economist publishes these on the bus) ----------------

ECON_HEALTH_REPORTED = "economist.health_reported"
ECON_ADJUSTMENT_RECOMMENDED = "economist.adjustment_recommended"
ECON_ADJUSTMENT_APPLIED = "economist.adjustment_applied"
ECON_ADJUSTMENT_FAILED = "economist.adjustment_failed"
ECON_AUTOPILOT_TICK = "economist.autopilot_tick"
ECON_SCENARIO_SIMULATED = "economist.scenario_simulated"


# --- dataclasses --------------------------------------------------------


@dataclass
class MarginTarget:
    """Per-tier economic guardrail.

    `gross_margin_pct_floor` is the minimum gross margin (as a
    fraction of net revenue) the economist tolerates over the
    evaluation window before it acts. `refund_rate_ceiling` is the
    maximum fraction of completed invoices that may be refund-bearing
    before the economist acts. Either trigger fires the adjustment.
    """
    tier: str
    gross_margin_pct_floor: float
    refund_rate_ceiling: float
    min_invoices: int = 5  # ignore noise below this sample size

    def __post_init__(self) -> None:
        if self.tier not in KNOWN_TIERS:
            raise ValueError(
                f"unknown tier {self.tier!r}; must be one of {KNOWN_TIERS}"
            )
        if not 0.0 <= self.gross_margin_pct_floor <= 1.0:
            raise ValueError("gross_margin_pct_floor must be in [0,1]")
        if not 0.0 <= self.refund_rate_ceiling <= 1.0:
            raise ValueError("refund_rate_ceiling must be in [0,1]")
        if self.min_invoices < 1:
            raise ValueError("min_invoices must be >= 1")


@dataclass
class EconomicWindow:
    """A read-only roll-up of a slice of invoices.

    All metrics are absolute over the window, not normalized to a
    rate. Coordination engines that want per-hour or per-minute rates
    divide by `window_s` themselves.
    """
    window_s: float
    invoices_count: int
    completed_count: int
    failed_count: int
    rejected_count: int
    refund_bearing_count: int
    revenue_usd: float
    refunds_usd: float
    net_revenue_usd: float
    cost_of_goods_usd: float
    gross_margin_usd: float

    @property
    def gross_margin_pct(self) -> float:
        if self.net_revenue_usd <= 1e-12:
            return 0.0
        return self.gross_margin_usd / self.net_revenue_usd

    @property
    def refund_rate(self) -> float:
        denom = self.completed_count
        if denom == 0:
            return 0.0
        return self.refund_bearing_count / denom

    @property
    def failure_rate(self) -> float:
        denom = self.invoices_count
        if denom == 0:
            return 0.0
        return self.failed_count / denom

    @property
    def rejection_rate(self) -> float:
        denom = self.invoices_count
        if denom == 0:
            return 0.0
        return self.rejected_count / denom

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_s": self.window_s,
            "invoices_count": self.invoices_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "rejected_count": self.rejected_count,
            "refund_bearing_count": self.refund_bearing_count,
            "revenue_usd": self.revenue_usd,
            "refunds_usd": self.refunds_usd,
            "net_revenue_usd": self.net_revenue_usd,
            "cost_of_goods_usd": self.cost_of_goods_usd,
            "gross_margin_usd": self.gross_margin_usd,
            "gross_margin_pct": self.gross_margin_pct,
            "refund_rate": self.refund_rate,
            "failure_rate": self.failure_rate,
            "rejection_rate": self.rejection_rate,
        }


@dataclass
class PolicyAdjustment:
    """An action the economist recommends or has just applied.

    All adjustments are reversible by the market's existing surface;
    none of them require the economist to mutate the market's
    internal state directly. The market sees the same Tenant object
    it always has — only the field values change.
    """
    id: str
    kind: str
    tenant_id: str | None
    tier: str | None
    severity: str
    rationale: str
    current_markup_pct: float | None = None
    suggested_markup_pct: float | None = None
    current_budget_usd: float | None = None
    suggested_budget_usd: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    """Single-shot snapshot of the economist's view."""
    ts: float
    window_s: float
    overall: EconomicWindow
    by_tier: dict[str, EconomicWindow]
    by_tenant: dict[str, EconomicWindow]
    adjustments: list[PolicyAdjustment]
    healthy: bool
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "window_s": self.window_s,
            "overall": self.overall.to_dict(),
            "by_tier": {k: v.to_dict() for k, v in self.by_tier.items()},
            "by_tenant": {k: v.to_dict() for k, v in self.by_tenant.items()},
            "adjustments": [a.to_dict() for a in self.adjustments],
            "healthy": self.healthy,
            "score": self.score,
        }


@dataclass
class AppliedAdjustment:
    """Result of attempting one adjustment.

    `applied=True` means the market's state was mutated; `applied=
    False` with `error` means the adjustment was a no-op or raised.
    """
    adjustment: PolicyAdjustment
    applied: bool
    error: str | None
    before: dict[str, Any]
    after: dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjustment": self.adjustment.to_dict(),
            "applied": self.applied,
            "error": self.error,
            "before": self.before,
            "after": self.after,
            "ts": self.ts,
        }


@dataclass
class Scenario:
    """A what-if input for `TicketEconomist.simulate()`.

    The simulator is deterministic: it takes the current invoice
    history as the baseline and applies the multipliers to produce a
    forward projection. It does *not* spawn real tickets.

    `tenants` (if provided) overrides the current tenant list. This
    lets an operator model "what if we add a new economy tenant"
    without registering them.
    """
    traffic_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    refund_rate: float | None = None     # forced refund rate (per tier)
    duration_s: float = 3600.0
    tenants: Sequence[Tenant] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.traffic_multiplier < 0:
            raise ValueError("traffic_multiplier must be >= 0")
        if self.cost_multiplier <= 0:
            raise ValueError("cost_multiplier must be > 0")
        if self.refund_rate is not None and not 0.0 <= self.refund_rate <= 1.0:
            raise ValueError("refund_rate must be in [0,1]")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be > 0")


@dataclass
class SimulationResult:
    """Projected economics for a Scenario, with recommended actions."""
    scenario: Scenario
    baseline_window_s: float
    baseline_invoice_count: int
    projected_invoice_count: int
    projected_revenue_usd: float
    projected_refunds_usd: float
    projected_cost_of_goods_usd: float
    projected_net_revenue_usd: float
    projected_gross_margin_usd: float
    projected_gross_margin_pct: float
    projected_refund_rate: float
    per_tier: dict[str, dict[str, float]]
    per_tenant: dict[str, dict[str, float]]
    actions_recommended: list[PolicyAdjustment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": asdict(self.scenario) | {
                "tenants": [
                    {"tenant_id": t.tenant_id, "tier": t.tier}
                    for t in (self.scenario.tenants or [])
                ],
            },
            "baseline_window_s": self.baseline_window_s,
            "baseline_invoice_count": self.baseline_invoice_count,
            "projected_invoice_count": self.projected_invoice_count,
            "projected_revenue_usd": self.projected_revenue_usd,
            "projected_refunds_usd": self.projected_refunds_usd,
            "projected_cost_of_goods_usd": self.projected_cost_of_goods_usd,
            "projected_net_revenue_usd": self.projected_net_revenue_usd,
            "projected_gross_margin_usd": self.projected_gross_margin_usd,
            "projected_gross_margin_pct": self.projected_gross_margin_pct,
            "projected_refund_rate": self.projected_refund_rate,
            "per_tier": self.per_tier,
            "per_tenant": self.per_tenant,
            "actions_recommended": [a.to_dict() for a in self.actions_recommended],
        }


# --- the economist -----------------------------------------------------


class TicketEconomist:
    """Margin-aware control plane for a `TicketMarket`.

    Single-process and thread-safe. The economist holds a weak read
    coupling to the market (it does not own the market lifecycle).
    Two operating modes:

      - **Advisory** (default): the coordination engine calls
        `health()` / `evaluate()` on a heartbeat and decides which
        adjustments to `apply()`.
      - **Auto-pilot**: a background thread runs the loop every
        `control_interval_s` and applies every produced adjustment.

    Auto-pilot is opt-in; advisory mode is the safer default and the
    only mode used by tests.
    """

    def __init__(
        self,
        market: TicketMarket,
        *,
        targets: Sequence[MarginTarget] | None = None,
        window_s: float = 300.0,
        control_interval_s: float = 10.0,
        markup_bump_pct: float = DEFAULT_MARKUP_BUMP_PCT,
        adjustments_path: str | Path | None = None,
        bus: EventBus | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        if control_interval_s <= 0:
            raise ValueError("control_interval_s must be > 0")
        if markup_bump_pct <= 0:
            raise ValueError("markup_bump_pct must be > 0")

        self.market = market
        self.window_s = float(window_s)
        self.control_interval_s = float(control_interval_s)
        self.markup_bump_pct = float(markup_bump_pct)
        self.bus = bus
        self._clock = clock or time.time

        self.targets: dict[str, MarginTarget] = {}
        for tier in KNOWN_TIERS:
            self.targets[tier] = MarginTarget(
                tier=tier,
                gross_margin_pct_floor=DEFAULT_MARGIN_FLOORS[tier],
                refund_rate_ceiling=DEFAULT_REFUND_CEILINGS[tier],
            )
        if targets is not None:
            for t in targets:
                self.targets[t.tier] = t

        self.adjustments_path = (
            Path(adjustments_path) if adjustments_path else None
        )
        if self.adjustments_path is not None:
            self.adjustments_path.parent.mkdir(parents=True, exist_ok=True)
            self.adjustments_path.touch(exist_ok=True)

        self._lock = threading.Lock()
        self._applied: list[AppliedAdjustment] = []
        self._reports: list[HealthReport] = []
        self._auto_pilot_enabled = False
        self._auto_pilot_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

    # --- configuration --------------------------------------------

    def set_target(self, target: MarginTarget) -> None:
        with self._lock:
            self.targets[target.tier] = target

    def targets_snapshot(self) -> dict[str, MarginTarget]:
        with self._lock:
            return dict(self.targets)

    # --- read paths -----------------------------------------------

    def health(self) -> HealthReport:
        """Snapshot the current health and produce recommendations."""
        now = self._clock()
        invoices = self.market.invoices()
        window_invoices = _filter_window(invoices, now, self.window_s)

        overall = _window_from_invoices(window_invoices, self.window_s)
        by_tier = {
            tier: _window_from_invoices(
                [inv for inv in window_invoices if inv.tier == tier],
                self.window_s,
            )
            for tier in KNOWN_TIERS
        }
        by_tenant: dict[str, EconomicWindow] = {}
        for inv in window_invoices:
            by_tenant.setdefault(inv.tenant_id, [])  # type: ignore[arg-type]
            by_tenant[inv.tenant_id].append(inv)     # type: ignore[arg-type]
        by_tenant = {
            tid: _window_from_invoices(invs, self.window_s)
            for tid, invs in by_tenant.items()
        }

        adjustments = self._compute_adjustments(by_tier, by_tenant)
        score = _health_score(by_tier, self.targets)
        healthy = (
            score >= 0.5
            and not any(a.severity == SEV_CRITICAL for a in adjustments)
        )

        report = HealthReport(
            ts=now,
            window_s=self.window_s,
            overall=overall,
            by_tier=by_tier,
            by_tenant=by_tenant,
            adjustments=adjustments,
            healthy=healthy,
            score=score,
        )
        with self._lock:
            self._reports.append(report)
            if len(self._reports) > 256:
                self._reports = self._reports[-256:]
        self._publish(ECON_HEALTH_REPORTED, {
            "score": score,
            "healthy": healthy,
            "adjustment_count": len(adjustments),
            "window_s": self.window_s,
        })
        return report

    def evaluate(self) -> list[PolicyAdjustment]:
        """Produce adjustments without taking a fresh health snapshot."""
        return self.health().adjustments

    def last_report(self) -> HealthReport | None:
        with self._lock:
            return self._reports[-1] if self._reports else None

    def applied_history(self) -> list[AppliedAdjustment]:
        with self._lock:
            return list(self._applied)

    # --- write paths ----------------------------------------------

    def apply(
        self,
        adjustments: Iterable[PolicyAdjustment],
        *,
        dry_run: bool = False,
    ) -> list[AppliedAdjustment]:
        """Apply each adjustment in order. Returns one result per input.

        `dry_run=True` produces `AppliedAdjustment(applied=False)` for
        every input without mutating the market — useful for testing a
        coordination engine's reaction without actually moving prices.
        """
        out: list[AppliedAdjustment] = []
        for adj in adjustments:
            res = self._apply_one(adj, dry_run=dry_run)
            out.append(res)
            with self._lock:
                self._applied.append(res)
            if res.applied:
                self._persist_adjustment(res)
                self._publish(ECON_ADJUSTMENT_APPLIED, res.to_dict())
            elif res.error is not None:
                self._publish(ECON_ADJUSTMENT_FAILED, res.to_dict())
        return out

    def auto_pilot(
        self,
        *,
        enable: bool = True,
    ) -> None:
        """Start or stop the background control loop.

        Idempotent: calling `auto_pilot(enable=True)` when already
        running is a no-op. The loop calls `health()` and applies
        every produced adjustment every `control_interval_s`.
        """
        with self._lock:
            if enable and not self._auto_pilot_enabled:
                self._auto_pilot_enabled = True
                self._shutdown.clear()
                self._auto_pilot_thread = threading.Thread(
                    target=self._auto_pilot_loop,
                    name="ticket-economist-autopilot",
                    daemon=True,
                )
                self._auto_pilot_thread.start()
            elif not enable and self._auto_pilot_enabled:
                self._auto_pilot_enabled = False
                self._shutdown.set()

    @property
    def autopilot_enabled(self) -> bool:
        with self._lock:
            return self._auto_pilot_enabled

    def close(self, *, timeout: float | None = 2.0) -> None:
        """Stop the auto-pilot loop. Idempotent."""
        with self._lock:
            self._auto_pilot_enabled = False
        self._shutdown.set()
        thread = self._auto_pilot_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # --- simulation -----------------------------------------------

    def simulate(self, scenario: Scenario) -> SimulationResult:
        """Project current economics forward under the scenario.

        The simulator does *not* hit the market or the runtime. It
        reuses the current invoice history as a behavioral baseline,
        multiplies traffic and per-ticket cost by the scenario's
        multipliers, optionally overrides the refund rate, and rolls
        the result up by tier and tenant. The output carries a list
        of `PolicyAdjustment`s the economist would have produced
        against the projection, so an operator sees what the
        autopilot would do without it actually doing it.
        """
        invoices = self.market.invoices()
        window = _filter_window(invoices, self._clock(), self.window_s)

        # Fall back to all invoices if the window is empty (cold
        # start). Without any history at all, the simulator returns
        # zeros and no adjustments — it doesn't fabricate data.
        if not window:
            window = list(invoices)

        per_tier_baseline = {
            t: [inv for inv in window if inv.tier == t]
            for t in KNOWN_TIERS
        }
        per_tier_proj: dict[str, dict[str, float]] = {}
        total_rev = 0.0
        total_ref = 0.0
        total_cost = 0.0
        total_count = 0

        scale = max(
            0.0,
            scenario.traffic_multiplier * (scenario.duration_s / max(self.window_s, 1e-12)),
        )

        for tier, invs in per_tier_baseline.items():
            if not invs:
                per_tier_proj[tier] = _empty_tier_projection()
                continue
            avg_revenue = statistics.fmean(i.list_price_usd for i in invs)
            avg_cost = statistics.fmean(i.cost_of_goods_usd for i in invs)
            completed = [i for i in invs if i.status == MKT_COMPLETED]
            if completed and scenario.refund_rate is None:
                avg_refund = statistics.fmean(i.refund_usd for i in completed)
                hist_refund_rate = (
                    sum(1 for i in completed if i.refund_usd > 0) / len(completed)
                )
            else:
                avg_refund = 0.0
                hist_refund_rate = 0.0
            forced_rate = scenario.refund_rate
            if forced_rate is not None:
                # Force the refund rate by recomputing refund per
                # ticket proportionally to list price. We use a fixed
                # 100% refund-on-breach assumption (matches market's
                # default SLO policy when no override is set).
                effective_refund_rate = forced_rate
                projected_refund_per_ticket = forced_rate * avg_revenue
            else:
                effective_refund_rate = hist_refund_rate
                projected_refund_per_ticket = avg_refund

            tier_count = len(invs) * scale
            tier_rev = avg_revenue * tier_count
            tier_cost = avg_cost * tier_count * scenario.cost_multiplier
            tier_ref = projected_refund_per_ticket * tier_count
            tier_net = tier_rev - tier_ref
            tier_margin = tier_net - tier_cost
            tier_margin_pct = (tier_margin / tier_net) if tier_net > 1e-12 else 0.0

            per_tier_proj[tier] = {
                "invoice_count": tier_count,
                "revenue_usd": tier_rev,
                "refunds_usd": tier_ref,
                "cost_of_goods_usd": tier_cost,
                "net_revenue_usd": tier_net,
                "gross_margin_usd": tier_margin,
                "gross_margin_pct": tier_margin_pct,
                "refund_rate": effective_refund_rate,
            }
            total_rev += tier_rev
            total_ref += tier_ref
            total_cost += tier_cost
            total_count += tier_count

        per_tenant_baseline: dict[str, list[Invoice]] = {}
        for inv in window:
            per_tenant_baseline.setdefault(inv.tenant_id, []).append(inv)
        per_tenant_proj: dict[str, dict[str, float]] = {}
        for tid, invs in per_tenant_baseline.items():
            avg_revenue = statistics.fmean(i.list_price_usd for i in invs)
            avg_cost = statistics.fmean(i.cost_of_goods_usd for i in invs)
            tier = invs[0].tier
            tier_proj = per_tier_proj.get(tier, _empty_tier_projection())
            ten_count = len(invs) * scale
            ten_rev = avg_revenue * ten_count
            ten_cost = avg_cost * ten_count * scenario.cost_multiplier
            ten_ref = tier_proj["refund_rate"] * ten_rev
            ten_net = ten_rev - ten_ref
            ten_margin = ten_net - ten_cost
            per_tenant_proj[tid] = {
                "tier": tier,
                "invoice_count": ten_count,
                "revenue_usd": ten_rev,
                "refunds_usd": ten_ref,
                "cost_of_goods_usd": ten_cost,
                "net_revenue_usd": ten_net,
                "gross_margin_usd": ten_margin,
                "gross_margin_pct": (
                    ten_margin / ten_net if ten_net > 1e-12 else 0.0
                ),
            }

        projected_net = total_rev - total_ref
        projected_margin = projected_net - total_cost
        projected_margin_pct = (
            projected_margin / projected_net if projected_net > 1e-12 else 0.0
        )
        projected_refund_rate = (
            total_ref / total_rev if total_rev > 1e-12 else 0.0
        )

        # Synthetic windows for adjustment generation against the
        # projection. We reconstruct minimal EconomicWindows from the
        # per-tier projection so we can reuse the production
        # `_compute_adjustments` path. Per-tenant projections aren't
        # rich enough to trigger tenant-level recs — the operator
        # gets tier-level recs in simulations, which is sufficient.
        sim_by_tier = {
            tier: _projection_to_window(per_tier_proj.get(tier, _empty_tier_projection()), self.window_s)
            for tier in KNOWN_TIERS
        }
        sim_actions = self._compute_adjustments(
            sim_by_tier,
            {},
            note_suffix=" (projected)",
        )

        result = SimulationResult(
            scenario=scenario,
            baseline_window_s=self.window_s,
            baseline_invoice_count=len(window),
            projected_invoice_count=int(round(total_count)),
            projected_revenue_usd=total_rev,
            projected_refunds_usd=total_ref,
            projected_cost_of_goods_usd=total_cost,
            projected_net_revenue_usd=projected_net,
            projected_gross_margin_usd=projected_margin,
            projected_gross_margin_pct=projected_margin_pct,
            projected_refund_rate=projected_refund_rate,
            per_tier=per_tier_proj,
            per_tenant=per_tenant_proj,
            actions_recommended=sim_actions,
        )
        self._publish(ECON_SCENARIO_SIMULATED, {
            "scenario_name": scenario.name,
            "projected_margin_pct": projected_margin_pct,
            "projected_revenue_usd": total_rev,
            "action_count": len(sim_actions),
        })
        return result

    # --- internals ------------------------------------------------

    def _compute_adjustments(
        self,
        by_tier: dict[str, EconomicWindow],
        by_tenant: dict[str, EconomicWindow],
        *,
        note_suffix: str = "",
    ) -> list[PolicyAdjustment]:
        out: list[PolicyAdjustment] = []
        # --- tier-level checks ---
        for tier, window in by_tier.items():
            target = self.targets.get(tier)
            if target is None:
                continue
            if window.invoices_count < target.min_invoices:
                continue
            margin_below = (
                window.gross_margin_pct < target.gross_margin_pct_floor
                and window.net_revenue_usd > 0
            )
            refund_above = window.refund_rate > target.refund_rate_ceiling
            if not (margin_below or refund_above):
                continue
            severity = SEV_CRITICAL if margin_below else SEV_WARN
            rationale_bits = []
            if margin_below:
                rationale_bits.append(
                    f"tier {tier!r} gross margin "
                    f"{window.gross_margin_pct:.2%} below floor "
                    f"{target.gross_margin_pct_floor:.2%}"
                )
            if refund_above:
                rationale_bits.append(
                    f"tier {tier!r} refund rate "
                    f"{window.refund_rate:.2%} above ceiling "
                    f"{target.refund_rate_ceiling:.2%}"
                )
            out.append(PolicyAdjustment(
                id=uuid.uuid4().hex[:12],
                kind=ADJ_RAISE_MARKUP,
                tenant_id=None,
                tier=tier,
                severity=severity,
                rationale="; ".join(rationale_bits) + note_suffix,
                current_markup_pct=TIER_DEFAULT_MARKUP[tier],
                suggested_markup_pct=min(
                    TIER_DEFAULT_MARKUP[tier] + self.markup_bump_pct,
                    ABS_MARKUP_CEILING,
                ),
                evidence=window.to_dict(),
                ts=self._clock(),
            ))

        # --- tenant-level checks ---
        for tid, window in by_tenant.items():
            account = self.market.get_tenant(tid)
            if account is None:
                continue
            tier_target = self.targets.get(account.tier)
            if tier_target is None:
                continue
            if window.invoices_count < tier_target.min_invoices:
                continue
            # Chronically unprofitable tenant → pause as critical.
            unprofitable = (
                window.gross_margin_usd < 0
                and window.net_revenue_usd > 0
            )
            high_refund = window.refund_rate > tier_target.refund_rate_ceiling
            margin_below = (
                window.gross_margin_pct < tier_target.gross_margin_pct_floor
                and window.net_revenue_usd > 0
            )
            if unprofitable and not account.tenant.paused:
                out.append(PolicyAdjustment(
                    id=uuid.uuid4().hex[:12],
                    kind=ADJ_PAUSE_TENANT,
                    tenant_id=tid,
                    tier=account.tier,
                    severity=SEV_CRITICAL,
                    rationale=(
                        f"tenant {tid!r} ran at a loss: "
                        f"margin ${window.gross_margin_usd:.4f} "
                        f"on revenue ${window.net_revenue_usd:.4f}"
                    ) + note_suffix,
                    evidence=window.to_dict(),
                    ts=self._clock(),
                ))
            elif margin_below or high_refund:
                current = account.tenant.effective_markup_pct
                bump = self.markup_bump_pct
                if high_refund:
                    # Refund-bearing tenants get a steeper bump — the
                    # platform is bearing the actual loss.
                    bump = self.markup_bump_pct * 1.5
                suggested = min(current + bump, ABS_MARKUP_CEILING)
                rationale_bits = []
                if margin_below:
                    rationale_bits.append(
                        f"tenant {tid!r} margin "
                        f"{window.gross_margin_pct:.2%} below tier floor "
                        f"{tier_target.gross_margin_pct_floor:.2%}"
                    )
                if high_refund:
                    rationale_bits.append(
                        f"tenant {tid!r} refund rate "
                        f"{window.refund_rate:.2%} above tier ceiling "
                        f"{tier_target.refund_rate_ceiling:.2%}"
                    )
                out.append(PolicyAdjustment(
                    id=uuid.uuid4().hex[:12],
                    kind=ADJ_RAISE_MARKUP,
                    tenant_id=tid,
                    tier=account.tier,
                    severity=SEV_WARN,
                    rationale="; ".join(rationale_bits) + note_suffix,
                    current_markup_pct=current,
                    suggested_markup_pct=suggested,
                    evidence=window.to_dict(),
                    ts=self._clock(),
                ))
            elif (
                account.tenant.paused
                and window.invoices_count >= tier_target.min_invoices
                and not unprofitable
                and not high_refund
            ):
                # A previously paused tenant whose recent window is
                # healthy enough → recommend resume.
                out.append(PolicyAdjustment(
                    id=uuid.uuid4().hex[:12],
                    kind=ADJ_RESUME_TENANT,
                    tenant_id=tid,
                    tier=account.tier,
                    severity=SEV_INFO,
                    rationale=(
                        f"tenant {tid!r} recovered: margin "
                        f"{window.gross_margin_pct:.2%}, refund rate "
                        f"{window.refund_rate:.2%}"
                    ) + note_suffix,
                    evidence=window.to_dict(),
                    ts=self._clock(),
                ))
        return out

    def _apply_one(
        self,
        adj: PolicyAdjustment,
        *,
        dry_run: bool,
    ) -> AppliedAdjustment:
        before: dict[str, Any] = {}
        after: dict[str, Any] = {}
        applied = False
        error: str | None = None

        try:
            if adj.kind == ADJ_RAISE_MARKUP:
                if adj.tenant_id is not None:
                    account = self.market.get_tenant(adj.tenant_id)
                    if account is None:
                        raise KeyError(
                            f"tenant {adj.tenant_id!r} not registered"
                        )
                    before["markup_pct"] = account.tenant.effective_markup_pct
                    suggested = max(
                        ABS_MARKUP_FLOOR,
                        min(adj.suggested_markup_pct or 0.0, ABS_MARKUP_CEILING),
                    )
                    if not dry_run:
                        account.tenant.markup_pct = suggested
                    after["markup_pct"] = suggested
                    applied = not dry_run
                elif adj.tier is not None:
                    # Tier-level markup is a recommendation only — the
                    # market's TIER_DEFAULT_MARKUP table is module-
                    # level state and we deliberately don't mutate
                    # it. We bump every tenant on that tier who
                    # doesn't have an explicit override.
                    bumped = []
                    suggested = max(
                        ABS_MARKUP_FLOOR,
                        min(adj.suggested_markup_pct or 0.0, ABS_MARKUP_CEILING),
                    )
                    for account in self.market.tenants():
                        if account.tier != adj.tier:
                            continue
                        if account.tenant.markup_pct is not None:
                            continue
                        prev = account.tenant.effective_markup_pct
                        if not dry_run:
                            account.tenant.markup_pct = suggested
                        bumped.append((account.tenant_id, prev, suggested))
                    before["bumped_tenants"] = [
                        {"tenant_id": tid, "markup_pct": prev}
                        for (tid, prev, _) in bumped
                    ]
                    after["bumped_tenants"] = [
                        {"tenant_id": tid, "markup_pct": new}
                        for (tid, _, new) in bumped
                    ]
                    after["suggested_markup_pct"] = suggested
                    applied = bool(bumped) and not dry_run
                else:
                    raise ValueError("raise_markup requires tenant_id or tier")
            elif adj.kind == ADJ_LOWER_MARKUP:
                if adj.tenant_id is None:
                    raise ValueError("lower_markup requires tenant_id")
                account = self.market.get_tenant(adj.tenant_id)
                if account is None:
                    raise KeyError(f"tenant {adj.tenant_id!r} not registered")
                before["markup_pct"] = account.tenant.effective_markup_pct
                suggested = max(
                    ABS_MARKUP_FLOOR,
                    min(adj.suggested_markup_pct or 0.0, ABS_MARKUP_CEILING),
                )
                if not dry_run:
                    account.tenant.markup_pct = suggested
                after["markup_pct"] = suggested
                applied = not dry_run
            elif adj.kind == ADJ_PAUSE_TENANT:
                if adj.tenant_id is None:
                    raise ValueError("pause_tenant requires tenant_id")
                account = self.market.get_tenant(adj.tenant_id)
                if account is None:
                    raise KeyError(f"tenant {adj.tenant_id!r} not registered")
                before["paused"] = account.tenant.paused
                if not dry_run:
                    self.market.pause_tenant(adj.tenant_id)
                after["paused"] = True
                applied = (not dry_run) and not before["paused"]
            elif adj.kind == ADJ_RESUME_TENANT:
                if adj.tenant_id is None:
                    raise ValueError("resume_tenant requires tenant_id")
                account = self.market.get_tenant(adj.tenant_id)
                if account is None:
                    raise KeyError(f"tenant {adj.tenant_id!r} not registered")
                before["paused"] = account.tenant.paused
                if not dry_run:
                    self.market.resume_tenant(adj.tenant_id)
                after["paused"] = False
                applied = (not dry_run) and before["paused"]
            elif adj.kind == ADJ_INCREASE_BUDGET:
                if adj.tenant_id is None:
                    raise ValueError("raise_tenant_budget requires tenant_id")
                if adj.suggested_budget_usd is None:
                    raise ValueError(
                        "raise_tenant_budget requires suggested_budget_usd"
                    )
                account = self.market.get_tenant(adj.tenant_id)
                if account is None:
                    raise KeyError(f"tenant {adj.tenant_id!r} not registered")
                before["monthly_budget_usd"] = account.tenant.monthly_budget_usd
                if not dry_run:
                    account.tenant.monthly_budget_usd = adj.suggested_budget_usd
                after["monthly_budget_usd"] = adj.suggested_budget_usd
                applied = not dry_run
            elif adj.kind == ADJ_NO_ACTION:
                applied = False
            else:
                raise ValueError(f"unknown adjustment kind {adj.kind!r}")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            applied = False

        return AppliedAdjustment(
            adjustment=adj,
            applied=applied,
            error=error,
            before=before,
            after=after,
            ts=self._clock(),
        )

    def _persist_adjustment(self, applied: AppliedAdjustment) -> None:
        if self.adjustments_path is None:
            return
        try:
            with self.adjustments_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(applied.to_dict(), default=str) + "\n")
        except OSError:
            # Persistence is best-effort; never block the autopilot.
            pass

    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        if self.bus is None:
            return
        try:
            self.bus.publish(Event(kind=kind, data=data, ts=self._clock()))
        except Exception:
            pass

    def _auto_pilot_loop(self) -> None:
        while not self._shutdown.is_set():
            self._publish(ECON_AUTOPILOT_TICK, {"ts": self._clock()})
            try:
                report = self.health()
                if report.adjustments:
                    self.apply(report.adjustments)
            except Exception:
                # Never let a control loop blow up; absorb and continue.
                pass
            if self._shutdown.wait(timeout=self.control_interval_s):
                return


# --- helpers ------------------------------------------------------------


def _filter_window(
    invoices: Sequence[Invoice],
    now: float,
    window_s: float,
) -> list[Invoice]:
    cutoff = now - window_s
    # Invoices are appended dispatch-completion-order; their
    # completed_ts is monotonic with submission only within a single
    # tenant, so we filter explicitly rather than bisecting the whole
    # list. The list is bounded by recent traffic, so a linear filter
    # is fine for production scales.
    return [
        inv for inv in invoices
        if (inv.completed_ts or inv.submitted_ts) >= cutoff
    ]


def _window_from_invoices(
    invoices: Sequence[Invoice],
    window_s: float,
) -> EconomicWindow:
    revenue = 0.0
    refunds = 0.0
    cost = 0.0
    completed = 0
    failed = 0
    rejected = 0
    refund_bearing = 0
    for inv in invoices:
        revenue += inv.list_price_usd
        refunds += inv.refund_usd
        cost += inv.cost_of_goods_usd
        if inv.status == MKT_COMPLETED:
            completed += 1
            if inv.refund_usd > 0:
                refund_bearing += 1
        elif inv.status == MKT_FAILED:
            failed += 1
        elif inv.status == MKT_REJECTED:
            rejected += 1
    net_revenue = revenue - refunds
    return EconomicWindow(
        window_s=window_s,
        invoices_count=len(invoices),
        completed_count=completed,
        failed_count=failed,
        rejected_count=rejected,
        refund_bearing_count=refund_bearing,
        revenue_usd=revenue,
        refunds_usd=refunds,
        net_revenue_usd=net_revenue,
        cost_of_goods_usd=cost,
        gross_margin_usd=net_revenue - cost,
    )


def _empty_tier_projection() -> dict[str, float]:
    return {
        "invoice_count": 0.0,
        "revenue_usd": 0.0,
        "refunds_usd": 0.0,
        "cost_of_goods_usd": 0.0,
        "net_revenue_usd": 0.0,
        "gross_margin_usd": 0.0,
        "gross_margin_pct": 0.0,
        "refund_rate": 0.0,
    }


def _projection_to_window(
    proj: dict[str, float],
    window_s: float,
) -> EconomicWindow:
    """Build a synthetic EconomicWindow from a tier projection so the
    adjustment generator can be reused against simulations."""
    count = int(round(proj["invoice_count"]))
    return EconomicWindow(
        window_s=window_s,
        invoices_count=count,
        completed_count=count,
        failed_count=0,
        rejected_count=0,
        refund_bearing_count=int(round(proj["refund_rate"] * count)),
        revenue_usd=proj["revenue_usd"],
        refunds_usd=proj["refunds_usd"],
        net_revenue_usd=proj["net_revenue_usd"],
        cost_of_goods_usd=proj["cost_of_goods_usd"],
        gross_margin_usd=proj["gross_margin_usd"],
    )


def _health_score(
    by_tier: dict[str, EconomicWindow],
    targets: dict[str, MarginTarget],
) -> float:
    """Aggregate health into 0..1.

    Score is the minimum of (a) per-tier margin headroom over the
    floor (clipped to [0,1]) and (b) per-tier refund-rate headroom
    under the ceiling (clipped to [0,1]). Empty tiers are treated as
    fully healthy. A tier with margin below its floor by 100% lands
    at 0; a tier at exactly its floor lands at 0; a tier with 2x the
    floor lands at 1.
    """
    sub_scores: list[float] = []
    for tier, window in by_tier.items():
        target = targets.get(tier)
        if target is None or window.invoices_count == 0:
            continue
        if window.net_revenue_usd > 0:
            margin_score = max(
                0.0,
                min(
                    1.0,
                    (window.gross_margin_pct - target.gross_margin_pct_floor)
                    / max(target.gross_margin_pct_floor, 1e-3),
                ),
            )
        else:
            margin_score = 0.0
        refund_score = max(
            0.0,
            min(
                1.0,
                (target.refund_rate_ceiling - window.refund_rate)
                / max(target.refund_rate_ceiling, 1e-3),
            ),
        )
        sub_scores.append(min(margin_score, refund_score))
    if not sub_scores:
        return 1.0
    return sum(sub_scores) / len(sub_scores)
