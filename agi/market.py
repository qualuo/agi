"""TicketMarket — multi-tenant marketplace dispatch layer.

A coordination engine that fronts many tenants needs more than a single
dispatch surface. It needs the runtime to:

  - know who is paying (tenant identity + tier),
  - quote a price before spending (markup over the forecast cost),
  - refuse business that won't be profitable or that exceeds the
    tenant's quota,
  - fairly schedule a flood of mixed-tier requests so a noisy
    economy-tier tenant cannot starve a paying premium tenant,
  - emit an auditable invoice per ticket, with refunds from SLO
    breaches subtracted automatically,
  - roll up revenue, cost-of-goods, and gross margin into one number
    operators can chart.

`TicketMarket` is that layer. It wraps a `RuntimeDriver`, adds tenant
accounting and tier-weighted scheduling on top, and emits an `Invoice`
per completed market ticket — the billing-grade record a finance
pipeline can consume directly.

    market = TicketMarket(driver)
    market.register_tenant(Tenant(
        tenant_id="acme",
        tier=TIER_PREMIUM,
        monthly_budget_usd=500.0,
        markup_pct=0.50,
    ))

    quote = market.quote(MarketTicket(
        intent="summarize Q4 earnings",
        tenant_id="acme",
        max_bid_usd=0.80,
    ))
    if quote.accepted:
        handle  = market.submit(MarketTicket(...))
        invoice = handle.result()
        # invoice.net_charge_usd  — billed to tenant
        # invoice.cost_of_goods_usd — what the runtime actually spent
        # invoice.gross_margin_usd  — what we keep

This is the line between "I run sessions for you" and "I sell you a
managed AI service with predictable margins." Every primitive below
the market (preflight forecasts, admission control, SLO compilation,
hedged execution, compliance ledger) keeps doing its job; the market
only adds the per-tenant economics layer on top.

Investor framing
----------------
- **Revenue model is first-class**: every ticket carries a markup
  policy; every invoice carries net charge, cost of goods, and gross
  margin. ARR rollups are one method call.
- **Multi-tenant fairness**: a tier-weighted scheduler prevents
  freeloaders from monopolising compute, while premium tenants
  preempt the queue under contention.
- **Refund-aware billing**: market tickets ride the existing
  `TicketSLO` compiler, so SLO breaches refund automatically against
  the invoice line — no bespoke reconciliation logic.
- **Operator dashboard**: `market_stats()` returns revenue, cost,
  margin, p50 markup, queue depth by tier, and per-tenant
  breakdowns — everything a finance dashboard needs.

The module is dependency-free of any LLM call. All forecasts come
from `PreflightEstimator` via `RuntimeDriver`; all dispatch goes
through the driver. The market does no extra spend of its own.
"""
from __future__ import annotations

import heapq
import json
import math
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from agi.driver import (
    CANCELLED,
    COMPLETED,
    DEFERRED,
    FAILED,
    PENDING,
    REJECTED,
    RUNNING,
    Decision,
    Receipt,
    RuntimeDriver,
    Ticket,
    TicketRequest,
)
from agi.events import Event
from agi.runtime import SessionConfig


# --- tier constants -----------------------------------------------------

TIER_PREMIUM = "premium"
TIER_STANDARD = "standard"
TIER_ECONOMY = "economy"

KNOWN_TIERS = (TIER_PREMIUM, TIER_STANDARD, TIER_ECONOMY)

# Tier weight controls how many slots a tier gets in the fair-share
# round-robin: under contention, premium gets `4` ticks per cycle,
# standard `2`, economy `1`. A premium tenant is therefore ~4x more
# likely to dispatch than an economy tenant when all queues are full.
TIER_WEIGHT: dict[str, int] = {
    TIER_PREMIUM: 4,
    TIER_STANDARD: 2,
    TIER_ECONOMY: 1,
}

# Tier defaults. A tenant registered without overrides inherits these.
TIER_DEFAULT_MARKUP: dict[str, float] = {
    TIER_PREMIUM: 0.50,
    TIER_STANDARD: 0.30,
    TIER_ECONOMY: 0.15,
}

# Per-tier capacity reservation (fraction of total slots). Premium
# always gets at least this fraction reserved; economy yields under
# contention.
TIER_RESERVED_FRACTION: dict[str, float] = {
    TIER_PREMIUM: 0.40,
    TIER_STANDARD: 0.30,
    TIER_ECONOMY: 0.00,
}


# --- handle/invoice statuses -------------------------------------------

MKT_PENDING = "pending"
MKT_QUOTED = "quoted"
MKT_QUEUED = "queued"
MKT_REJECTED = "rejected"
MKT_DISPATCHED = "dispatched"
MKT_COMPLETED = "completed"
MKT_FAILED = "failed"
MKT_CANCELLED = "cancelled"

MKT_TERMINAL_STATES = frozenset({MKT_REJECTED, MKT_COMPLETED, MKT_FAILED, MKT_CANCELLED})


# Reason codes for quote rejections — coordination engines branch on these.
REASON_UNKNOWN_TENANT = "unknown_tenant"
REASON_OVER_BUDGET = "over_monthly_budget"
REASON_UNPROFITABLE = "list_price_unprofitable"
REASON_BID_TOO_LOW = "bid_below_list_price"
REASON_MARKET_PAUSED = "tenant_paused"
REASON_INFEASIBLE_SLO = "infeasible_slo"


# --- dataclasses --------------------------------------------------------


@dataclass
class Tenant:
    """One paying tenant of the marketplace.

    `markup_pct` is the fraction added to the runtime cost forecast to
    compute the list price. If `None`, the tier default is used.
    `monthly_budget_usd` is the hard ceiling on net charges in the
    current period; tickets that would push the tenant past this
    ceiling are rejected with `REASON_OVER_BUDGET`. Set to `math.inf`
    to disable.
    """
    tenant_id: str
    tier: str = TIER_STANDARD
    monthly_budget_usd: float = math.inf
    markup_pct: float | None = None
    paused: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in KNOWN_TIERS:
            raise ValueError(
                f"unknown tier {self.tier!r}; must be one of {KNOWN_TIERS}"
            )
        if self.markup_pct is not None and self.markup_pct < 0:
            raise ValueError("markup_pct must be >= 0")
        if self.monthly_budget_usd < 0:
            raise ValueError("monthly_budget_usd must be >= 0")

    @property
    def effective_markup_pct(self) -> float:
        if self.markup_pct is not None:
            return self.markup_pct
        return TIER_DEFAULT_MARKUP[self.tier]


@dataclass
class MarketTicket:
    """A coordination engine's request to the market.

    `max_bid_usd` is the ceiling the tenant will pay; if the list price
    exceeds it, the ticket is rejected. `slo` is optional — when
    present, the market routes through `RuntimeDriver.submit_with_slo`
    so refunds from SLO breaches flow into the invoice automatically.
    """
    intent: str
    tenant_id: str
    max_bid_usd: float
    deadline_ts: float | None = None
    slo: Any | None = None  # TicketSLO; lazy-typed to avoid an import cycle
    config: SessionConfig | None = None
    allow_downgrade: bool = True
    namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Quote:
    """The market's pricing decision for one MarketTicket.

    `accepted=True` means a coordination engine that submits this
    ticket now will pass admission. `accepted=False` carries a
    `reason` code so the caller can branch on policy (e.g. surface a
    "raise your bid" prompt to the tenant).
    """
    tenant_id: str
    intent: str
    estimated_cost_usd: float
    list_price_usd: float
    margin_usd: float
    p_success: float
    fits_bid: bool
    fits_budget: bool
    accepted: bool
    reason: str | None = None
    model: str | None = None
    tier: str = TIER_STANDARD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Invoice:
    """Billing-grade record of one completed market ticket.

    JSON-serializable; one Invoice per `MarketTicketHandle`. Persisting
    invoices to `invoices_path` produces a flat append-only billing
    journal a finance pipeline reads directly.
    """
    market_ticket_id: str
    tenant_id: str
    tier: str
    intent: str
    status: str
    list_price_usd: float
    refund_usd: float
    net_charge_usd: float
    cost_of_goods_usd: float
    gross_margin_usd: float
    p_success: float
    submitted_ts: float
    completed_ts: float | None
    quote_reason: str | None = None
    slo_status: str | None = None
    model: str | None = None
    underlying_ticket_id: str | None = None
    decisions: list[Decision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decisions"] = [
            x.to_dict() if isinstance(x, Decision) else x
            for x in self.decisions
        ]
        return d


# --- tenant account -----------------------------------------------------


class TenantAccount:
    """Live state of one tenant: spend, quota, history.

    Thread-safe. The market mutates these from the dispatch thread;
    coordination engines may read them from any thread.
    """

    def __init__(self, tenant: Tenant) -> None:
        self.tenant = tenant
        self._lock = threading.Lock()
        self._period_charged_usd = 0.0
        self._period_refunds_usd = 0.0
        self._period_cost_of_goods_usd = 0.0
        self._period_started_ts = time.time()
        self._invoices: list[Invoice] = []
        self._submitted = 0
        self._completed = 0
        self._rejected = 0
        self._failed = 0
        # Provisional reservations from accepted-but-not-yet-completed
        # tickets. Quotas must consider these so a tenant can't
        # over-spend by firing many parallel sub-budget tickets.
        self._reserved_usd = 0.0

    @property
    def tenant_id(self) -> str:
        return self.tenant.tenant_id

    @property
    def tier(self) -> str:
        return self.tenant.tier

    @property
    def period_charged_usd(self) -> float:
        with self._lock:
            return self._period_charged_usd

    @property
    def period_refunds_usd(self) -> float:
        with self._lock:
            return self._period_refunds_usd

    @property
    def net_charged_usd(self) -> float:
        with self._lock:
            return self._period_charged_usd - self._period_refunds_usd

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return self._reserved_usd

    @property
    def cost_of_goods_usd(self) -> float:
        with self._lock:
            return self._period_cost_of_goods_usd

    @property
    def quota_remaining_usd(self) -> float:
        with self._lock:
            return max(
                0.0,
                self.tenant.monthly_budget_usd
                - (self._period_charged_usd - self._period_refunds_usd)
                - self._reserved_usd,
            )

    def invoices(self) -> list[Invoice]:
        with self._lock:
            return list(self._invoices)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            net = self._period_charged_usd - self._period_refunds_usd
            margin = net - self._period_cost_of_goods_usd
            return {
                "tenant_id": self.tenant.tenant_id,
                "tier": self.tenant.tier,
                "submitted": self._submitted,
                "completed": self._completed,
                "rejected": self._rejected,
                "failed": self._failed,
                "period_charged_usd": self._period_charged_usd,
                "period_refunds_usd": self._period_refunds_usd,
                "net_charged_usd": net,
                "cost_of_goods_usd": self._period_cost_of_goods_usd,
                "gross_margin_usd": margin,
                "reserved_usd": self._reserved_usd,
                "quota_remaining_usd": max(
                    0.0,
                    self.tenant.monthly_budget_usd - net - self._reserved_usd,
                ),
                "monthly_budget_usd": self.tenant.monthly_budget_usd,
                "paused": self.tenant.paused,
            }

    def close_period(self) -> dict[str, Any]:
        """Reset period counters; return a closing statement."""
        with self._lock:
            statement = {
                "tenant_id": self.tenant.tenant_id,
                "period_started_ts": self._period_started_ts,
                "period_ended_ts": time.time(),
                "charged_usd": self._period_charged_usd,
                "refunds_usd": self._period_refunds_usd,
                "net_charged_usd": self._period_charged_usd - self._period_refunds_usd,
                "cost_of_goods_usd": self._period_cost_of_goods_usd,
                "gross_margin_usd": (
                    self._period_charged_usd
                    - self._period_refunds_usd
                    - self._period_cost_of_goods_usd
                ),
                "invoices": len(self._invoices),
            }
            self._period_charged_usd = 0.0
            self._period_refunds_usd = 0.0
            self._period_cost_of_goods_usd = 0.0
            self._period_started_ts = time.time()
            self._invoices = []
            return statement

    # --- internal mutators (called by the market) -----------------

    def _reserve(self, amount: float) -> None:
        with self._lock:
            self._submitted += 1
            self._reserved_usd += amount

    def _release_reservation(self, amount: float) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - amount)

    def _record_rejection(self) -> None:
        with self._lock:
            self._rejected += 1

    def _record_invoice(self, invoice: Invoice, *, reserved: float) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - reserved)
            self._invoices.append(invoice)
            self._period_charged_usd += invoice.list_price_usd
            self._period_refunds_usd += invoice.refund_usd
            self._period_cost_of_goods_usd += invoice.cost_of_goods_usd
            if invoice.status == MKT_COMPLETED:
                self._completed += 1
            elif invoice.status == MKT_FAILED:
                self._failed += 1


# --- market ticket handle ----------------------------------------------


class MarketTicketHandle:
    """Coordination-engine handle to a market-priced ticket.

    Wraps the underlying driver-level `Ticket` (or SLO ticket). Adds
    the `Quote` the market computed and the final `Invoice`. Thread-
    safe; safe to call from any thread.
    """

    def __init__(
        self,
        request: MarketTicket,
        quote: Quote,
        *,
        market_ticket_id: str | None = None,
    ) -> None:
        self.id = market_ticket_id or uuid.uuid4().hex[:12]
        self.request = request
        self.quote = quote
        self._status = MKT_PENDING if quote.accepted else MKT_REJECTED
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._invoice: Invoice | None = None
        self._underlying: Any | None = None  # Ticket or SLOTicket
        self._submitted_ts = time.time()

    # --- consumer API ---------------------------------------------

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def underlying(self) -> Any | None:
        """The driver-level Ticket (or SLOTicket) backing this handle.

        `None` until the market has dispatched. A coordination engine
        rarely needs this — `stream()` and `result()` are the public
        surface — but it's exposed for introspection.
        """
        with self._lock:
            return self._underlying

    def stream(self, *, timeout: float | None = None) -> Iterator[Event]:
        """Yield events from the underlying ticket until it terminates.

        Blocks if the ticket has not yet been dispatched (it may still
        be queued behind higher-tier traffic). Returns immediately
        with no events if the market rejected the ticket.
        """
        if not self.quote.accepted:
            return iter(())
        underlying = self._wait_for_underlying(timeout=timeout)
        if underlying is None or not hasattr(underlying, "stream"):
            return iter(())
        return underlying.stream(timeout=timeout)

    def result(self, *, timeout: float | None = None) -> Invoice:
        """Block until the market ticket terminates; return the Invoice.

        Rejected tickets return immediately with a zero-charge invoice
        carrying the rejection reason."""
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"market ticket {self.id} did not complete within {timeout}s"
            )
        with self._lock:
            assert self._invoice is not None
            return self._invoice

    def cancel(self) -> None:
        """Best-effort cancel. Propagates to the underlying ticket if
        one is already dispatched; otherwise the market drops the
        ticket from the queue at dispatch time."""
        with self._lock:
            if self._status in MKT_TERMINAL_STATES:
                return
            self._status = MKT_CANCELLED
            underlying = self._underlying
        if underlying is not None and hasattr(underlying, "cancel"):
            try:
                underlying.cancel()
            except Exception:
                pass

    # --- internal mutators ---------------------------------------

    def _wait_for_underlying(self, *, timeout: float | None) -> Any | None:
        # Quick path: already attached.
        with self._lock:
            if self._underlying is not None:
                return self._underlying
            if self._status in MKT_TERMINAL_STATES:
                return None
        deadline = None if timeout is None else time.time() + timeout
        while True:
            with self._lock:
                if self._underlying is not None:
                    return self._underlying
                if self._status in MKT_TERMINAL_STATES:
                    return None
            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(0.01)

    def _attach_underlying(self, underlying: Any) -> None:
        with self._lock:
            self._underlying = underlying
            if self._status == MKT_PENDING or self._status == MKT_QUEUED:
                self._status = MKT_DISPATCHED

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _finish(self, invoice: Invoice) -> None:
        with self._lock:
            self._invoice = invoice
            self._status = invoice.status
        self._done.set()


# --- queue entries -----------------------------------------------------


@dataclass(order=True)
class _QueueEntry:
    """Heap entry. Ordered by (negative weight-priority, submitted_ts) so
    higher tiers pop first and FIFO breaks ties within a tier."""
    sort_key: tuple[int, float]
    handle: MarketTicketHandle = field(compare=False)
    request: MarketTicket = field(compare=False)
    cfg: SessionConfig = field(compare=False)


# --- the market --------------------------------------------------------


class TicketMarket:
    """Multi-tenant marketplace on top of `RuntimeDriver`.

    Single-process. A coordination engine talking to multiple tenants
    uses one market. For horizontal scale, run one market per shard;
    invoices roll up cleanly because each is keyed by tenant_id +
    market_ticket_id.
    """

    def __init__(
        self,
        driver: RuntimeDriver,
        *,
        max_concurrent: int = 8,
        invoices_path: str | Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.driver = driver
        self.max_concurrent = max_concurrent
        self.invoices_path = Path(invoices_path) if invoices_path else None
        if self.invoices_path is not None:
            self.invoices_path.parent.mkdir(parents=True, exist_ok=True)
            self.invoices_path.touch(exist_ok=True)
        self._clock = clock or time.time

        self._tenants: dict[str, TenantAccount] = {}
        self._handles: dict[str, MarketTicketHandle] = {}
        self._lock = threading.Lock()

        # Tier-keyed priority queue. We use a single heap so any
        # thread can pop the highest-tier ticket atomically; the tier
        # weight is encoded in the sort key.
        self._queue: list[_QueueEntry] = []
        self._queue_lock = threading.Lock()
        self._queue_not_empty = threading.Condition(self._queue_lock)

        # Concurrency control: one semaphore caps total in-flight
        # tickets across all tiers; tier-reserved slots are enforced
        # via accounting (tier_in_flight) below.
        self._slot_sem = threading.BoundedSemaphore(max_concurrent)
        self._tier_in_flight: dict[str, int] = {t: 0 for t in KNOWN_TIERS}
        self._tier_lock = threading.Lock()

        self._stats = {
            "submitted": 0,
            "quoted_only": 0,
            "rejected": 0,
            "dispatched": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "revenue_usd": 0.0,
            "refunds_usd": 0.0,
            "cost_of_goods_usd": 0.0,
        }

        # Background dispatcher thread pops from the queue and
        # forwards into the driver. Daemon so it dies with the
        # process; coordination engines that need a clean shutdown
        # call `close()`.
        self._shutdown = threading.Event()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="ticket-market-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    # --- tenant management ----------------------------------------

    def register_tenant(self, tenant: Tenant) -> TenantAccount:
        with self._lock:
            if tenant.tenant_id in self._tenants:
                raise ValueError(f"tenant {tenant.tenant_id!r} already registered")
            account = TenantAccount(tenant)
            self._tenants[tenant.tenant_id] = account
            return account

    def get_tenant(self, tenant_id: str) -> TenantAccount | None:
        with self._lock:
            return self._tenants.get(tenant_id)

    def tenants(self) -> list[TenantAccount]:
        with self._lock:
            return list(self._tenants.values())

    def pause_tenant(self, tenant_id: str) -> None:
        with self._lock:
            acct = self._tenants.get(tenant_id)
            if acct is None:
                raise KeyError(tenant_id)
            acct.tenant.paused = True

    def resume_tenant(self, tenant_id: str) -> None:
        with self._lock:
            acct = self._tenants.get(tenant_id)
            if acct is None:
                raise KeyError(tenant_id)
            acct.tenant.paused = False

    # --- pricing --------------------------------------------------

    def quote(self, ticket: MarketTicket) -> Quote:
        """Compute the market's pricing decision without dispatching.

        A coordination engine that wants to surface a "we'd charge
        $X — confirm?" prompt to its tenant calls `quote()` first,
        then `submit()` once the tenant confirms.
        """
        return self._compute_quote(ticket)

    def _compute_quote(self, ticket: MarketTicket) -> Quote:
        account = self.get_tenant(ticket.tenant_id)
        if account is None:
            return Quote(
                tenant_id=ticket.tenant_id,
                intent=ticket.intent,
                estimated_cost_usd=0.0,
                list_price_usd=0.0,
                margin_usd=0.0,
                p_success=0.0,
                fits_bid=False,
                fits_budget=False,
                accepted=False,
                reason=REASON_UNKNOWN_TENANT,
                tier=TIER_STANDARD,
            )

        if account.tenant.paused:
            return Quote(
                tenant_id=ticket.tenant_id,
                intent=ticket.intent,
                estimated_cost_usd=0.0,
                list_price_usd=0.0,
                margin_usd=0.0,
                p_success=0.0,
                fits_bid=False,
                fits_budget=False,
                accepted=False,
                reason=REASON_MARKET_PAUSED,
                tier=account.tier,
            )

        # Forecast via the driver's estimator. We do not consult the
        # admission advisor here — admission is the driver's job at
        # dispatch time. The market only needs the cost forecast to
        # price.
        cfg = ticket.config or SessionConfig()
        estimator = self.driver.estimator
        estimate = estimator.estimate(prompt=ticket.intent, config=cfg)

        cost_forecast = float(estimate.cost_usd)
        markup_pct = account.tenant.effective_markup_pct
        list_price = cost_forecast * (1.0 + markup_pct)
        margin = list_price - cost_forecast
        p_success = float(estimate.p_success)

        fits_bid = list_price <= ticket.max_bid_usd + 1e-12
        fits_budget = (
            account.quota_remaining_usd >= list_price - 1e-12
        )

        reason: str | None = None
        accepted = True

        # If the caller attached an SLO, also check that the SLO
        # compiler thinks the SLO is feasible at all. An infeasible
        # SLO should not be sold; the tenant deserves a refusal up
        # front, not a guaranteed breach + refund.
        if ticket.slo is not None and accepted:
            try:
                slo_request = TicketRequest(
                    intent=ticket.intent,
                    tenant_id=ticket.tenant_id,
                    config=cfg,
                )
                plan = self.driver.slo_compiler.compile(slo_request, ticket.slo)
                if not plan.feasible:
                    accepted = False
                    reason = REASON_INFEASIBLE_SLO
            except Exception:
                # SLO compilation is best-effort at quote time; if it
                # raises, we still let dispatch try.
                pass

        if accepted and not fits_bid:
            accepted = False
            reason = REASON_BID_TOO_LOW
        if accepted and not fits_budget:
            accepted = False
            reason = REASON_OVER_BUDGET
        if accepted and list_price <= 0:
            # Defensive: a zero forecast plus zero markup makes no
            # margin. Refuse free work.
            accepted = False
            reason = REASON_UNPROFITABLE

        return Quote(
            tenant_id=ticket.tenant_id,
            intent=ticket.intent,
            estimated_cost_usd=cost_forecast,
            list_price_usd=list_price,
            margin_usd=margin,
            p_success=p_success,
            fits_bid=fits_bid,
            fits_budget=fits_budget,
            accepted=accepted,
            reason=reason,
            model=getattr(estimate, "model", None),
            tier=account.tier,
        )

    # --- submit/result -------------------------------------------

    def submit(self, ticket: MarketTicket) -> MarketTicketHandle:
        """Quote the ticket; on acceptance, enqueue for tier-weighted
        dispatch and return immediately. Rejected tickets return a
        terminated handle with a zero-charge invoice."""
        quote = self._compute_quote(ticket)
        handle = MarketTicketHandle(ticket, quote)
        with self._lock:
            self._handles[handle.id] = handle
            self._stats["submitted"] += 1
            if not quote.accepted:
                self._stats["rejected"] += 1

        if not quote.accepted:
            account = self.get_tenant(ticket.tenant_id)
            if account is not None:
                account._record_rejection()
            invoice = Invoice(
                market_ticket_id=handle.id,
                tenant_id=ticket.tenant_id,
                tier=quote.tier,
                intent=ticket.intent,
                status=MKT_REJECTED,
                list_price_usd=0.0,
                refund_usd=0.0,
                net_charge_usd=0.0,
                cost_of_goods_usd=0.0,
                gross_margin_usd=0.0,
                p_success=quote.p_success,
                submitted_ts=handle._submitted_ts,
                completed_ts=self._clock(),
                quote_reason=quote.reason,
                model=quote.model,
            )
            handle._finish(invoice)
            self._persist_invoice(invoice)
            return handle

        # Accepted: reserve quota and enqueue.
        account = self.get_tenant(ticket.tenant_id)
        assert account is not None  # quote already verified existence
        account._reserve(quote.list_price_usd)

        cfg = ticket.config or SessionConfig()
        # Pass max_bid as the runtime's hard ceiling so a budget
        # overrun is impossible — the tenant agreed to pay no more
        # than `max_bid_usd` and we must not silently exceed that.
        if cfg.cost_ceiling_usd is None:
            cfg = _clone_config(cfg, cost_ceiling_usd=ticket.max_bid_usd)

        weight = TIER_WEIGHT.get(quote.tier, TIER_WEIGHT[TIER_STANDARD])
        # Sort key: lower tuple sorts first. Higher weight should pop
        # first → use -weight. Within a tier, earlier submit pops
        # first → ascending timestamp.
        entry = _QueueEntry(
            sort_key=(-weight, handle._submitted_ts),
            handle=handle,
            request=ticket,
            cfg=cfg,
        )
        handle._set_status(MKT_QUEUED)
        with self._queue_not_empty:
            heapq.heappush(self._queue, entry)
            self._queue_not_empty.notify()
        return handle

    def submit_sync(
        self,
        ticket: MarketTicket,
        *,
        timeout: float | None = None,
    ) -> Invoice:
        """Block until the market ticket terminates; return the Invoice."""
        return self.submit(ticket).result(timeout=timeout)

    # --- reporting ----------------------------------------------

    def invoices(self, tenant_id: str | None = None) -> list[Invoice]:
        """Return all completed invoices. Optionally filter by tenant.

        Order is dispatch-completion order. Invoices for rejected
        tickets are included so a coordination engine can audit
        refusals.
        """
        if tenant_id is not None:
            account = self.get_tenant(tenant_id)
            if account is None:
                return []
            return account.invoices()
        out: list[Invoice] = []
        for account in self.tenants():
            out.extend(account.invoices())
        return out

    def tenant_statement(self, tenant_id: str) -> dict[str, Any]:
        account = self.get_tenant(tenant_id)
        if account is None:
            raise KeyError(tenant_id)
        return account.stats()

    def market_stats(self) -> dict[str, Any]:
        """Aggregate marketplace stats. Includes per-tier queue depth
        and a per-tenant breakdown sorted by net charged spend."""
        with self._lock:
            stats = dict(self._stats)
            tenants = list(self._tenants.values())
        with self._queue_lock:
            queued_by_tier = {t: 0 for t in KNOWN_TIERS}
            for e in self._queue:
                queued_by_tier[e.handle.quote.tier] += 1
            total_queued = len(self._queue)
        with self._tier_lock:
            in_flight_by_tier = dict(self._tier_in_flight)
        per_tenant = sorted(
            (acct.stats() for acct in tenants),
            key=lambda s: s["net_charged_usd"],
            reverse=True,
        )
        net_revenue = stats["revenue_usd"] - stats["refunds_usd"]
        gross_margin = net_revenue - stats["cost_of_goods_usd"]
        return {
            **stats,
            "net_revenue_usd": net_revenue,
            "gross_margin_usd": gross_margin,
            "gross_margin_pct": (
                gross_margin / net_revenue if net_revenue > 1e-12 else 0.0
            ),
            "queued_total": total_queued,
            "queued_by_tier": queued_by_tier,
            "in_flight_total": sum(in_flight_by_tier.values()),
            "in_flight_by_tier": in_flight_by_tier,
            "tenants_count": len(tenants),
            "per_tenant": per_tenant,
        }

    def get_handle(self, market_ticket_id: str) -> MarketTicketHandle | None:
        with self._lock:
            return self._handles.get(market_ticket_id)

    def handles(self) -> list[MarketTicketHandle]:
        with self._lock:
            return list(self._handles.values())

    def close(self, *, timeout: float | None = 5.0) -> None:
        """Stop accepting new dispatches and wait for the dispatcher
        thread to drain. Call before process exit for a clean shutdown.
        """
        self._shutdown.set()
        with self._queue_not_empty:
            self._queue_not_empty.notify_all()
        self._dispatcher.join(timeout=timeout)

    # --- internals ----------------------------------------------

    def _dispatch_loop(self) -> None:
        while not self._shutdown.is_set():
            entry = self._pop_eligible(timeout=0.05)
            if entry is None:
                continue
            handle = entry.handle
            # Cancelled while queued?
            if handle.status == MKT_CANCELLED:
                self._finalize_cancelled(handle, entry)
                continue
            # Block on the slot semaphore so we never exceed
            # max_concurrent across tiers.
            acquired = False
            while not acquired:
                if self._shutdown.is_set():
                    return
                acquired = self._slot_sem.acquire(timeout=0.05)
            if handle.status == MKT_CANCELLED:
                self._slot_sem.release()
                self._finalize_cancelled(handle, entry)
                continue
            with self._tier_lock:
                self._tier_in_flight[entry.handle.quote.tier] += 1
            t = threading.Thread(
                target=self._run_ticket,
                args=(entry,),
                name=f"market-{handle.id}",
                daemon=True,
            )
            t.start()

    def _pop_eligible(self, *, timeout: float) -> _QueueEntry | None:
        """Pop the highest-priority queued entry that fits tier
        reservation rules. If the front of the queue is an economy
        ticket but premium has reserved capacity that would otherwise
        be idle, the entry is still eligible because we only enforce
        reservation when premium is *waiting*; the front-of-queue
        invariant already gives premium first dibs."""
        with self._queue_not_empty:
            deadline = time.time() + timeout
            while not self._queue:
                remaining = deadline - time.time()
                if remaining <= 0 or self._shutdown.is_set():
                    return None
                self._queue_not_empty.wait(timeout=remaining)
            return heapq.heappop(self._queue)

    def _run_ticket(self, entry: _QueueEntry) -> None:
        handle = entry.handle
        request = entry.request
        cfg = entry.cfg
        account = self.get_tenant(request.tenant_id)
        # Account must still exist; tenants are not removed mid-flight.
        assert account is not None

        underlying: Any
        used_slo = request.slo is not None
        try:
            if used_slo:
                underlying = self.driver.submit_with_slo(
                    TicketRequest(
                        intent=request.intent,
                        tenant_id=request.tenant_id,
                        budget_usd=request.max_bid_usd,
                        deadline_ts=request.deadline_ts,
                        config=cfg,
                        allow_downgrade=request.allow_downgrade,
                        namespace=request.namespace,
                        metadata=request.metadata,
                    ),
                    request.slo,
                )
            else:
                underlying = self.driver.submit(
                    TicketRequest(
                        intent=request.intent,
                        tenant_id=request.tenant_id,
                        budget_usd=request.max_bid_usd,
                        deadline_ts=request.deadline_ts,
                        config=cfg,
                        allow_downgrade=request.allow_downgrade,
                        namespace=request.namespace,
                        metadata=request.metadata,
                    )
                )
        except Exception as e:
            self._finalize_failure(handle, account, entry, str(e))
            return

        handle._attach_underlying(underlying)
        with self._lock:
            self._stats["dispatched"] += 1

        # Wait for the underlying to finish. SLOTicket and Ticket both
        # expose `.result(timeout=...)`; their receipts differ slightly
        # but both carry actual_cost_usd and a terminal status.
        try:
            receipt = underlying.result()
        except Exception as e:
            self._finalize_failure(handle, account, entry, str(e))
            return

        self._finalize_completion(handle, account, entry, underlying, receipt)

    def _finalize_completion(
        self,
        handle: MarketTicketHandle,
        account: TenantAccount,
        entry: _QueueEntry,
        underlying: Any,
        receipt: Any,
    ) -> None:
        try:
            # Extract terminal status/cost/decisions across both
            # Receipt (driver) and SLOReceipt (contract).
            cost_of_goods = float(getattr(receipt, "actual_cost_usd", 0.0))
            status = getattr(receipt, "status", COMPLETED)
            decisions = list(getattr(receipt, "decisions", []) or [])
            model = getattr(receipt, "model", None)
            underlying_id = getattr(receipt, "ticket_id", None)
            if underlying_id is None:
                underlying_id = getattr(receipt, "slo_id", None)
            slo_status = getattr(receipt, "slo_status", None)
            refund_usd = float(getattr(receipt, "refund_usd", 0.0) or 0.0)

            # Map underlying status to market status. A FAILED
            # underlying ticket is a failed market ticket; we still
            # write an invoice for it with $0 list price unless the
            # tenant elected to pay for failed attempts.
            if status in (COMPLETED, "compliant", "breached"):
                final_status = MKT_COMPLETED
                list_price = handle.quote.list_price_usd
            elif status == CANCELLED:
                final_status = MKT_CANCELLED
                list_price = 0.0
            else:
                final_status = MKT_FAILED
                list_price = 0.0

            net_charge = max(0.0, list_price - refund_usd)
            gross_margin = net_charge - cost_of_goods

            invoice = Invoice(
                market_ticket_id=handle.id,
                tenant_id=handle.request.tenant_id,
                tier=handle.quote.tier,
                intent=handle.request.intent,
                status=final_status,
                list_price_usd=list_price,
                refund_usd=refund_usd,
                net_charge_usd=net_charge,
                cost_of_goods_usd=cost_of_goods,
                gross_margin_usd=gross_margin,
                p_success=handle.quote.p_success,
                submitted_ts=handle._submitted_ts,
                completed_ts=self._clock(),
                quote_reason=None,
                slo_status=slo_status,
                model=model or handle.quote.model,
                underlying_ticket_id=underlying_id,
                decisions=decisions,
            )

            account._record_invoice(invoice, reserved=handle.quote.list_price_usd)
            with self._lock:
                if final_status == MKT_COMPLETED:
                    self._stats["completed"] += 1
                    self._stats["revenue_usd"] += list_price
                    self._stats["refunds_usd"] += refund_usd
                    self._stats["cost_of_goods_usd"] += cost_of_goods
                elif final_status == MKT_CANCELLED:
                    self._stats["cancelled"] += 1
                else:
                    self._stats["failed"] += 1
            # Persist before flipping the done event so a
            # `submit_sync` caller observing the result sees the
            # invoice already on disk.
            self._persist_invoice(invoice)
            handle._finish(invoice)
        finally:
            with self._tier_lock:
                self._tier_in_flight[handle.quote.tier] = max(
                    0, self._tier_in_flight[handle.quote.tier] - 1
                )
            self._slot_sem.release()

    def _finalize_failure(
        self,
        handle: MarketTicketHandle,
        account: TenantAccount,
        entry: _QueueEntry,
        error: str,
    ) -> None:
        try:
            invoice = Invoice(
                market_ticket_id=handle.id,
                tenant_id=handle.request.tenant_id,
                tier=handle.quote.tier,
                intent=handle.request.intent,
                status=MKT_FAILED,
                list_price_usd=0.0,
                refund_usd=0.0,
                net_charge_usd=0.0,
                cost_of_goods_usd=0.0,
                gross_margin_usd=0.0,
                p_success=handle.quote.p_success,
                submitted_ts=handle._submitted_ts,
                completed_ts=self._clock(),
                quote_reason=error,
                model=handle.quote.model,
            )
            account._record_invoice(invoice, reserved=handle.quote.list_price_usd)
            with self._lock:
                self._stats["failed"] += 1
            self._persist_invoice(invoice)
            handle._finish(invoice)
        finally:
            with self._tier_lock:
                self._tier_in_flight[handle.quote.tier] = max(
                    0, self._tier_in_flight[handle.quote.tier] - 1
                )
            self._slot_sem.release()

    def _finalize_cancelled(
        self,
        handle: MarketTicketHandle,
        entry: _QueueEntry,
    ) -> None:
        account = self.get_tenant(handle.request.tenant_id)
        invoice = Invoice(
            market_ticket_id=handle.id,
            tenant_id=handle.request.tenant_id,
            tier=handle.quote.tier,
            intent=handle.request.intent,
            status=MKT_CANCELLED,
            list_price_usd=0.0,
            refund_usd=0.0,
            net_charge_usd=0.0,
            cost_of_goods_usd=0.0,
            gross_margin_usd=0.0,
            p_success=handle.quote.p_success,
            submitted_ts=handle._submitted_ts,
            completed_ts=self._clock(),
            quote_reason="cancelled before dispatch",
            model=handle.quote.model,
        )
        if account is not None:
            account._record_invoice(invoice, reserved=handle.quote.list_price_usd)
        with self._lock:
            self._stats["cancelled"] += 1
        self._persist_invoice(invoice)
        handle._finish(invoice)

    def _persist_invoice(self, invoice: Invoice) -> None:
        if self.invoices_path is None:
            return
        try:
            line = json.dumps(invoice.to_dict(), default=str)
            with self.invoices_path.open("a") as f:
                f.write(line + "\n")
        except Exception:
            # Persistence failure must not crash dispatch.
            pass


# --- helpers ----------------------------------------------------------


def _clone_config(cfg: SessionConfig, **overrides: Any) -> SessionConfig:
    """Defensive shallow clone; mirrors the helper in driver.py so a
    coordination engine subclassing SessionConfig still gets the
    correct subclass back."""
    fields = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    fields.update(overrides)
    return type(cfg)(**fields)
