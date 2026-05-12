"""Service-level objectives (SLOs) for tickets — the runtime's contract surface.

A coordination engine has, until now, told the runtime *what model to
run on what budget*. That is imperative — the engine has to know the
model catalog, the per-tier price/quality curve, and when to hedge.

SLOs invert that. A coordination engine declares the *objective*:

    slo = TicketSLO(
        min_p_success=0.90,        # I want at least 90% expected success
        max_cost_usd=0.40,         # spend up to 40 cents
        max_latency_s=30.0,        # finish in 30 seconds wall-clock
        hedge_policy="auto",       # parallelize models if needed to hit the floor
        refund_on_breach=1.0,      # full refund credit if we miss
    )
    slo_ticket = driver.submit_with_slo(request, slo)
    receipt    = slo_ticket.result()

The runtime *compiles* that SLO into a concrete execution plan against
the live `PreflightEstimator` forecasts:

  - cheapest single model whose forecast meets the SLO  →  STRAT_SINGLE
  - else a parallel race ("hedge") across the
    fewest models needed to push hedged_p_success over the floor
    within max_cost_usd                                  →  STRAT_HEDGE
  - else the best-effort plan, marked infeasible, with a stated reason
    so the coordination engine decides whether to dispatch anyway

This is the line between "I sell you tokens" and "I sell you outcomes."
Coordination engines write declarative goals; the runtime delivers
auditable, billable contracts.

Investor framing
----------------
- **Declarative outcomes**: every ticket carries an SLO; every receipt
  carries a compliance verdict. Operators chart compliance rate the
  same way a SaaS charts uptime.
- **Speculative quality**: the hedge primitive is the standard low-
  latency / high-quality lever — two independent 0.7-success models in
  parallel deliver 0.91 hedged success on a *single bill line*.
- **Cost frontier**: `frontier_for_slo(intent, slo, budgets)` plots
  expected success vs. spend so an operator can size cost ceilings
  on evidence, not guesswork.
- **Refund-on-breach**: the compliance ledger writes the refund-eligible
  cost back into the receipt so a billing pipeline can honor it
  without bespoke plumbing.

The module is dependency-free of any LLM call. All forecasts come from
`PreflightEstimator`; all dispatch goes through `RuntimeDriver`. It
composes cleanly with `PortfolioOptimizer` (which solves the dual
problem: many tickets, one budget).
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
    Ticket,
    TicketRequest,
)
from agi.events import Event
from agi.portfolio import (
    DEFAULT_CANDIDATE_MODELS,
    PortfolioCandidate,
)
from agi.preflight import PreflightEstimator
from agi.runtime import SessionConfig


# Strategy kinds (recorded on the plan + receipt so a coordination
# engine can replay why a ticket was hedged or single-shot).
STRAT_SINGLE = "single"
STRAT_HEDGE = "hedge"

# SLO compliance status — terminal, per ticket.
SLO_COMPLIANT = "compliant"
SLO_BREACHED = "breached"
SLO_INFEASIBLE = "infeasible"   # dispatched a plan we knew couldn't meet the SLO
SLO_FAILED = "failed"           # dispatch itself failed (no usable outcome)

# Default hedge cap. Three is plenty for almost any practical workload;
# the marginal uplift past three independent 0.7-success models is small.
DEFAULT_HEDGE_MAX_PARALLEL = 3


# ---------- public dataclasses ----------------------------------------


@dataclass
class TicketSLO:
    """Declarative objective attached to a TicketRequest.

    All fields are optional. A TicketSLO() with default zeros admits
    everything — the compiler picks the cheapest forecastable model.

    Fields
    ------
    min_p_success
        Floor on forecast (and post-hedge) success probability.
    max_cost_usd
        Hard ceiling on expected and actual cost. The compiler refuses
        plans whose expected_cost_usd exceeds this; the driver still
        enforces per-session cost_ceiling_usd as a safety net.
    max_latency_s
        Wall-clock ceiling. Missed latency is a breach but not a refusal
        — the runtime cannot preempt a model mid-generation reliably.
    hedge_policy
        "off"    — never hedge; if no single model meets min_p_success
                   under budget, return an infeasible plan.
        "auto"   — hedge only when no single model meets the SLO.
        "always" — always hedge up to hedge_max_parallel for maximum
                   redundancy.
    hedge_max_parallel
        Upper bound on parallel hedged children.
    refund_on_breach
        Fraction (0..1) of actual cost the runtime records as
        refund-eligible if the SLO is breached. A billing pipeline
        reads `ComplianceRecord.refund_usd` and decides how to honor it.
    candidate_models
        Override the default candidate pool (e.g. restrict to internal
        models only, or include a specialty model). Defaults to the
        full `DEFAULT_CANDIDATE_MODELS` set from portfolio.
    """
    min_p_success: float = 0.0
    max_cost_usd: float | None = None
    max_latency_s: float | None = None
    hedge_policy: str = "auto"
    hedge_max_parallel: int = DEFAULT_HEDGE_MAX_PARALLEL
    refund_on_breach: float = 0.0
    candidate_models: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_p_success <= 1.0:
            raise ValueError("min_p_success must be in [0, 1]")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0")
        if self.max_latency_s is not None and self.max_latency_s < 0:
            raise ValueError("max_latency_s must be >= 0")
        if self.hedge_policy not in ("off", "auto", "always"):
            raise ValueError(f"unknown hedge_policy {self.hedge_policy!r}")
        if self.hedge_max_parallel < 1:
            raise ValueError("hedge_max_parallel must be >= 1")
        if not 0.0 <= self.refund_on_breach <= 1.0:
            raise ValueError("refund_on_breach must be in [0, 1]")
        if self.candidate_models is not None and not self.candidate_models:
            raise ValueError("candidate_models, if set, must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.candidate_models is not None:
            d["candidate_models"] = list(self.candidate_models)
        return d


@dataclass
class SLOPlan:
    """Compiled execution plan for one (request, SLO) pair."""
    strategy: str
    feasible: bool
    candidates: list[PortfolioCandidate]
    expected_cost_usd: float
    expected_p_success: float
    expected_duration_s: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "feasible": self.feasible,
            "expected_cost_usd": self.expected_cost_usd,
            "expected_p_success": self.expected_p_success,
            "expected_duration_s": self.expected_duration_s,
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class SLOReceipt:
    """Aggregate receipt for an SLOTicket.

    For STRAT_SINGLE this mirrors the underlying child receipt. For
    STRAT_HEDGE it aggregates across all children — `actual_cost_usd`
    is the *sum* (you paid for losers too), `final_text` is the winner's
    output, `winner_model` records which candidate's result was kept.
    """
    slo_ticket_id: str
    request_intent: str
    slo: TicketSLO
    plan: SLOPlan
    status: str           # one of the driver-level terminal statuses
    slo_status: str       # one of SLO_COMPLIANT / SLO_BREACHED / SLO_INFEASIBLE / SLO_FAILED
    breaches: list[str]
    refund_usd: float
    winner_model: str | None
    winner_ticket_id: str | None
    final_text: str | None
    error: str | None
    actual_cost_usd: float
    actual_duration_s: float
    children: list[Receipt]
    submitted_ts: float
    completed_ts: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slo_ticket_id": self.slo_ticket_id,
            "request_intent": self.request_intent,
            "slo": self.slo.to_dict(),
            "plan": self.plan.to_dict(),
            "status": self.status,
            "slo_status": self.slo_status,
            "breaches": list(self.breaches),
            "refund_usd": self.refund_usd,
            "winner_model": self.winner_model,
            "winner_ticket_id": self.winner_ticket_id,
            "final_text": self.final_text,
            "error": self.error,
            "actual_cost_usd": self.actual_cost_usd,
            "actual_duration_s": self.actual_duration_s,
            "children": [c.to_dict() for c in self.children],
            "submitted_ts": self.submitted_ts,
            "completed_ts": self.completed_ts,
        }


@dataclass
class ComplianceRecord:
    """One row in the compliance ledger."""
    ticket_id: str
    slo: TicketSLO
    plan: SLOPlan
    actual_cost_usd: float
    actual_duration_s: float
    success: bool
    slo_status: str
    breaches: list[str]
    refund_usd: float
    chosen_model: str | None
    finished_ts: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "slo": self.slo.to_dict(),
            "plan": self.plan.to_dict(),
            "actual_cost_usd": self.actual_cost_usd,
            "actual_duration_s": self.actual_duration_s,
            "success": self.success,
            "slo_status": self.slo_status,
            "breaches": list(self.breaches),
            "refund_usd": self.refund_usd,
            "chosen_model": self.chosen_model,
            "finished_ts": self.finished_ts,
        }


# ---------- helpers ----------------------------------------------------


def hedged_p_success(probs: Sequence[float]) -> float:
    """Probability that AT LEAST ONE of N independent attempts succeeds.

    Used for hedge sizing. The independence assumption is approximate
    (correlated failures across models attempting the same prompt are
    real) — but the empirical history in PreflightEstimator naturally
    self-corrects as observed p_success per (model, prompt-bucket)
    bin updates.
    """
    product = 1.0
    for p in probs:
        clipped = max(0.0, min(1.0, float(p)))
        product *= 1.0 - clipped
    return 1.0 - product


def _clone_session_config(cfg: SessionConfig, **overrides: Any) -> SessionConfig:
    data = {**cfg.__dict__, **overrides}
    return type(cfg)(**data)


# ---------- compiler ---------------------------------------------------


class SLOCompiler:
    """Turn a (TicketRequest, TicketSLO) pair into an executable SLOPlan.

    Stateless w.r.t. requests; references a `PreflightEstimator`. Safe
    to share across threads as long as the estimator is.
    """

    def __init__(
        self,
        estimator: PreflightEstimator,
        *,
        candidate_models: Sequence[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.default_models: tuple[str, ...] = tuple(
            candidate_models or DEFAULT_CANDIDATE_MODELS
        )

    def compile(self, request: TicketRequest, slo: TicketSLO) -> SLOPlan:
        models = tuple(slo.candidate_models or self.default_models)
        base_cfg = request.config or SessionConfig()
        budget = slo.max_cost_usd if slo.max_cost_usd is not None else math.inf

        # Forecast every (request, model) cell. Sort cheap → expensive.
        forecasts = self._forecast(request.intent, base_cfg, models)
        forecasts_by_cost = sorted(forecasts, key=lambda c: c.estimated_cost_usd)

        # --- single-model path: cheapest model that meets the SLO ---
        if slo.hedge_policy != "always":
            for c in forecasts_by_cost:
                if c.estimated_cost_usd > budget:
                    continue
                if c.estimated_p_success >= slo.min_p_success:
                    return SLOPlan(
                        strategy=STRAT_SINGLE,
                        feasible=True,
                        candidates=[c],
                        expected_cost_usd=round(c.estimated_cost_usd, 6),
                        expected_p_success=round(c.estimated_p_success, 6),
                        expected_duration_s=round(c.estimated_duration_s, 6),
                        reason="single model meets SLO",
                    )

        # --- hedge path: greedily add by uplift-per-marginal-dollar ---
        if slo.hedge_policy == "off":
            # Refuse to hedge; emit best single-model plan, infeasible.
            best = forecasts_by_cost[0] if forecasts_by_cost else None
            return SLOPlan(
                strategy=STRAT_SINGLE,
                feasible=False,
                candidates=[best] if best is not None else [],
                expected_cost_usd=round(best.estimated_cost_usd, 6) if best else 0.0,
                expected_p_success=round(best.estimated_p_success, 6) if best else 0.0,
                expected_duration_s=round(best.estimated_duration_s, 6) if best else 0.0,
                reason="no single model meets min_p_success within budget; hedging disabled",
            )

        cap = max(1, slo.hedge_max_parallel)
        remaining = list(forecasts)
        chosen: list[PortfolioCandidate] = []
        chosen_cost = 0.0
        # Greedy: maximize marginal uplift / marginal cost until SLO met,
        # hedge_max_parallel reached, or budget exhausted.
        while len(chosen) < cap and remaining:
            current_p = hedged_p_success([c.estimated_p_success for c in chosen])
            best_i = -1
            best_ratio = -1.0
            for i, c in enumerate(remaining):
                if chosen_cost + c.estimated_cost_usd > budget + 1e-12:
                    continue
                trial_p = hedged_p_success(
                    [x.estimated_p_success for x in chosen] + [c.estimated_p_success]
                )
                uplift = trial_p - current_p
                if uplift <= 0:
                    # Adding this candidate doesn't help (e.g. p=0); skip.
                    continue
                if c.estimated_cost_usd <= 0:
                    ratio = uplift * 1e9
                else:
                    ratio = uplift / c.estimated_cost_usd
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_i = i
            if best_i < 0:
                break
            picked = remaining.pop(best_i)
            chosen.append(picked)
            chosen_cost += picked.estimated_cost_usd
            hedged_now = hedged_p_success([c.estimated_p_success for c in chosen])
            if hedged_now >= slo.min_p_success and slo.hedge_policy == "auto":
                break

        hedged_p = hedged_p_success([c.estimated_p_success for c in chosen])
        feasible = (
            len(chosen) > 0
            and hedged_p + 1e-12 >= slo.min_p_success
            and chosen_cost <= budget + 1e-12
        )

        if not chosen:
            # No candidate fit the budget. Surface the cheapest one as
            # an infeasible suggestion so callers see the gap.
            fallback = forecasts_by_cost[:1]
            return SLOPlan(
                strategy=STRAT_SINGLE,
                feasible=False,
                candidates=fallback,
                expected_cost_usd=(
                    round(fallback[0].estimated_cost_usd, 6) if fallback else 0.0
                ),
                expected_p_success=(
                    round(fallback[0].estimated_p_success, 6) if fallback else 0.0
                ),
                expected_duration_s=(
                    round(fallback[0].estimated_duration_s, 6) if fallback else 0.0
                ),
                reason="no candidate fits within max_cost_usd",
            )

        strategy = STRAT_SINGLE if len(chosen) == 1 else STRAT_HEDGE
        if slo.hedge_policy == "always":
            # Honor the operator's intent in the trace, even if one model
            # would have sufficed.
            strategy = STRAT_HEDGE
        # Hedge wall-clock is the max of children (parallel race).
        max_dur = max((c.estimated_duration_s for c in chosen), default=0.0)
        if feasible:
            reason = "hedge meets SLO" if strategy == STRAT_HEDGE else "single model meets SLO"
        elif slo.min_p_success > 0:
            reason = "could not meet min_p_success within budget"
        else:
            reason = "no candidate within budget"

        return SLOPlan(
            strategy=strategy,
            feasible=feasible,
            candidates=list(chosen),
            expected_cost_usd=round(chosen_cost, 6),
            expected_p_success=round(hedged_p, 6),
            expected_duration_s=round(max_dur, 6),
            reason=reason,
        )

    def frontier(
        self,
        request: TicketRequest,
        slo: TicketSLO,
        *,
        budgets: Sequence[float],
    ) -> list[dict[str, Any]]:
        """For each budget, return the SLO-best plan summary.

        Operators chart this to answer "what does another $X buy me?"
        for one ticket — the dual of `PortfolioOptimizer.frontier`,
        which answers it for many tickets.
        """
        rows: list[dict[str, Any]] = []
        for b in budgets:
            scoped = TicketSLO(
                min_p_success=slo.min_p_success,
                max_cost_usd=float(b),
                max_latency_s=slo.max_latency_s,
                hedge_policy=slo.hedge_policy,
                hedge_max_parallel=slo.hedge_max_parallel,
                refund_on_breach=slo.refund_on_breach,
                candidate_models=slo.candidate_models,
            )
            plan = self.compile(request, scoped)
            rows.append({
                "budget_usd": float(b),
                "expected_cost_usd": plan.expected_cost_usd,
                "expected_p_success": plan.expected_p_success,
                "expected_duration_s": plan.expected_duration_s,
                "strategy": plan.strategy,
                "feasible": plan.feasible,
                "models": [c.model for c in plan.candidates],
            })
        return rows

    def _forecast(
        self,
        intent: str,
        base_cfg: SessionConfig,
        models: Sequence[str],
    ) -> list[PortfolioCandidate]:
        out: list[PortfolioCandidate] = []
        for m in models:
            cfg = _clone_session_config(base_cfg, model=m)
            est = self.estimator.estimate(intent, cfg)
            out.append(PortfolioCandidate(
                model=m,
                estimated_cost_usd=float(est.cost_usd),
                estimated_p_success=float(est.p_success),
                estimated_duration_s=float(est.duration_s),
                estimate=est,
                score=float(est.p_success),
            ))
        return out


# ---------- compliance ledger -----------------------------------------


def evaluate_compliance(
    slo: TicketSLO,
    plan: SLOPlan,
    *,
    actual_cost_usd: float,
    actual_duration_s: float,
    success: bool,
    ticket_id: str,
    chosen_model: str | None,
) -> ComplianceRecord:
    """Compare an actual outcome against the SLO. Pure function.

    The probabilistic SLO field `min_p_success` is *not* breached on a
    single failure unless the dispatched plan was infeasible — we
    cannot fairly fault the runtime for a 90%-success forecast that
    landed on the 10% tail. An infeasible-but-dispatched plan IS
    breached because the runtime already told the operator it couldn't
    keep the promise."""
    breaches: list[str] = []

    if not plan.feasible:
        breaches.append("infeasible_plan")
    if slo.max_cost_usd is not None and actual_cost_usd > slo.max_cost_usd + 1e-9:
        breaches.append("cost")
    if slo.max_latency_s is not None and actual_duration_s > slo.max_latency_s + 1e-9:
        breaches.append("latency")

    if not success:
        # An infeasible plan that never produced a success is reported as
        # INFEASIBLE — the runtime told the operator up-front it could
        # not keep the promise. A feasible plan that nonetheless failed
        # is either BREACHED (some breach was recorded) or FAILED.
        if not plan.feasible:
            slo_status = SLO_INFEASIBLE
        elif breaches:
            slo_status = SLO_BREACHED
        else:
            slo_status = SLO_FAILED
    elif breaches:
        slo_status = SLO_BREACHED
    else:
        slo_status = SLO_COMPLIANT

    refund = (slo.refund_on_breach * actual_cost_usd) if breaches else 0.0
    return ComplianceRecord(
        ticket_id=ticket_id,
        slo=slo,
        plan=plan,
        actual_cost_usd=round(actual_cost_usd, 6),
        actual_duration_s=round(actual_duration_s, 6),
        success=success,
        slo_status=slo_status,
        breaches=breaches,
        refund_usd=round(refund, 6),
        chosen_model=chosen_model,
        finished_ts=time.time(),
    )


class ComplianceLedger:
    """Append-only JSONL log of ComplianceRecords. Thread-safe.

    Persisting compliance is what turns "the runtime sometimes does
    well" into "we charge customers an SLO and prove it". A coordination
    engine's billing pipeline reads this file (or watches the in-memory
    list) to bill, refund, and chart compliance over time.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._records: list[ComplianceRecord] = []
        self._path = Path(path) if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.touch(exist_ok=True)

    def record(self, rec: ComplianceRecord) -> None:
        with self._lock:
            self._records.append(rec)
            if self._path is not None:
                try:
                    with self._path.open("a") as f:
                        f.write(json.dumps(rec.to_dict(), default=str))
                        f.write("\n")
                except Exception:
                    pass

    def all(self) -> list[ComplianceRecord]:
        with self._lock:
            return list(self._records)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            recs = list(self._records)
        n = len(recs)
        if n == 0:
            return {
                "total": 0,
                "compliant": 0,
                "breached": 0,
                "infeasible": 0,
                "failed": 0,
                "compliance_rate": 0.0,
                "total_cost_usd": 0.0,
                "total_refund_usd": 0.0,
                "by_breach": {},
                "by_status": {},
            }
        compliant = sum(1 for r in recs if r.slo_status == SLO_COMPLIANT)
        breached = sum(1 for r in recs if r.slo_status == SLO_BREACHED)
        infeasible = sum(1 for r in recs if r.slo_status == SLO_INFEASIBLE)
        failed = sum(1 for r in recs if r.slo_status == SLO_FAILED)
        total_cost = sum(r.actual_cost_usd for r in recs)
        total_refund = sum(r.refund_usd for r in recs)
        by_breach: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for r in recs:
            for b in r.breaches:
                by_breach[b] = by_breach.get(b, 0) + 1
            by_status[r.slo_status] = by_status.get(r.slo_status, 0) + 1
        return {
            "total": n,
            "compliant": compliant,
            "breached": breached,
            "infeasible": infeasible,
            "failed": failed,
            "compliance_rate": round(compliant / n, 6),
            "total_cost_usd": round(total_cost, 6),
            "total_refund_usd": round(total_refund, 6),
            "by_breach": by_breach,
            "by_status": by_status,
        }


# ---------- SLOTicket: the coordination-engine handle -----------------


class SLOTicket:
    """Handle returned by `RuntimeDriver.submit_with_slo`.

    Behaves like a `Ticket` for the coordination engine — has `status`,
    `stream()`, `result()`, `cancel()` — but wraps one or more child
    tickets. For STRAT_HEDGE, the children race; the first to finish
    successfully wins and the rest are cancelled.

    Thread-safe. The race watcher runs on a background thread.
    """

    def __init__(
        self,
        request: TicketRequest,
        slo: TicketSLO,
        plan: SLOPlan,
        children: Sequence[Ticket],
        ledger: ComplianceLedger | None = None,
        ticket_id: str | None = None,
    ) -> None:
        if not children:
            raise ValueError("SLOTicket requires at least one child")
        self.id = ticket_id or "slo_" + uuid.uuid4().hex[:10]
        self.request = request
        self.slo = slo
        self.plan = plan
        self.children: list[Ticket] = list(children)
        self._ledger = ledger
        self._submitted_ts = time.time()
        self._receipt: SLOReceipt | None = None
        self._done = threading.Event()
        self._cancel_requested = threading.Event()
        self._lock = threading.Lock()
        self._status = PENDING
        self._event_q: queue.Queue[Event | None] = queue.Queue()
        self._fan_in_threads: list[threading.Thread] = []
        # Fan-in: every child's stream emits onto our queue, tagged via
        # `event.metadata["slo_child_id"]`.
        for child in self.children:
            t = threading.Thread(
                target=self._fan_in_child, args=(child,), daemon=True,
                name=f"slo-fanin-{child.id}",
            )
            t.start()
            self._fan_in_threads.append(t)
        self._race_thread = threading.Thread(
            target=self._race, daemon=True, name=f"slo-race-{self.id}",
        )
        self._race_thread.start()

    # --- public surface --------------------------------------------

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def stream(self, *, timeout: float | None = None):
        """Yield events as the SLO ticket progresses. Multiplexes events
        from every child. Each event's `metadata["slo_child_id"]` carries
        which child it came from."""
        while True:
            try:
                ev = self._event_q.get(timeout=timeout)
            except queue.Empty:
                return
            if ev is None:
                return
            yield ev

    def result(self, *, timeout: float | None = None) -> SLOReceipt:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError(
                f"slo ticket {self.id} did not complete within {timeout}s"
            )
        with self._lock:
            assert self._receipt is not None
            return self._receipt

    def cancel(self) -> None:
        """Request cancellation across all children. Idempotent."""
        self._cancel_requested.set()
        for c in self.children:
            try:
                c.cancel()
            except Exception:
                pass

    # --- internals -------------------------------------------------

    def _fan_in_child(self, child: Ticket) -> None:
        """Forward events from one child onto the SLOTicket's stream.

        Events pass through unchanged; the child's session_id already
        disambiguates which child emitted what. A coordination engine
        that needs to map session_id back to the child can do so via
        `slo_ticket.children`.
        """
        try:
            for ev in child.stream():
                self._event_q.put(ev)
        except Exception:
            pass

    def _race(self) -> None:
        """Wait for children, declare a winner, cancel losers, finalize.

        Single-strategy path: just block on the one child.
        Hedge path: as soon as any child completes successfully, cancel
        the rest. If all fail, the last failure stands.
        """
        with self._lock:
            self._status = RUNNING

        if self.plan.strategy == STRAT_SINGLE or len(self.children) == 1:
            child = self.children[0]
            child.result()
            self._finalize_from([child])
            return

        # Hedge race: poll children until one succeeds or all finish.
        winner: Ticket | None = None
        try:
            while True:
                if self._cancel_requested.is_set():
                    break
                done_children = [c for c in self.children if c.done]
                if any(c.receipt.status == COMPLETED for c in done_children):
                    winner = next(
                        c for c in self.children
                        if c.done and c.receipt.status == COMPLETED
                    )
                    break
                if len(done_children) == len(self.children):
                    break
                time.sleep(0.005)
        except Exception:
            pass

        if winner is not None:
            for c in self.children:
                if c is not winner and not c.done:
                    try:
                        c.cancel()
                    except Exception:
                        pass
        # Wait for the rest to settle so cost aggregation is accurate.
        for c in self.children:
            try:
                c.result(timeout=60.0)
            except Exception:
                pass
        self._finalize_from(self.children, winner=winner)

    def _finalize_from(
        self,
        children: Sequence[Ticket],
        *,
        winner: Ticket | None = None,
    ) -> None:
        child_receipts = [c.receipt for c in children]
        actual_cost = sum(r.actual_cost_usd for r in child_receipts)
        actual_dur = max((r.actual_duration_s for r in child_receipts), default=0.0)

        if winner is None:
            # Single-strategy or no hedge winner.
            for c in children:
                if c.receipt.status == COMPLETED:
                    winner = c
                    break

        if winner is not None:
            status = COMPLETED
            final_text = winner.receipt.final_text
            error = None
            winner_model = winner.receipt.model
            winner_id = winner.id
        else:
            # No child completed successfully. Surface whichever failure
            # is most informative.
            failures = [c for c in children if c.receipt.error]
            cancelled = [c for c in children if c.receipt.status == CANCELLED]
            if self._cancel_requested.is_set() and cancelled:
                status = CANCELLED
                error = "cancelled"
            elif failures:
                status = FAILED
                error = failures[0].receipt.error
            else:
                # All non-completed, no errors recorded → treat as failed.
                status = FAILED
                error = "all hedge children non-success"
            final_text = None
            winner_model = None
            winner_id = None

        success = winner is not None
        compliance = evaluate_compliance(
            self.slo,
            self.plan,
            actual_cost_usd=actual_cost,
            actual_duration_s=actual_dur,
            success=success,
            ticket_id=self.id,
            chosen_model=winner_model,
        )

        receipt = SLOReceipt(
            slo_ticket_id=self.id,
            request_intent=self.request.intent,
            slo=self.slo,
            plan=self.plan,
            status=status,
            slo_status=compliance.slo_status,
            breaches=list(compliance.breaches),
            refund_usd=compliance.refund_usd,
            winner_model=winner_model,
            winner_ticket_id=winner_id,
            final_text=final_text,
            error=error,
            actual_cost_usd=round(actual_cost, 6),
            actual_duration_s=round(actual_dur, 6),
            children=child_receipts,
            submitted_ts=self._submitted_ts,
            completed_ts=time.time(),
        )

        with self._lock:
            self._receipt = receipt
            self._status = status
        if self._ledger is not None:
            try:
                self._ledger.record(compliance)
            except Exception:
                pass
        self._event_q.put(None)
        self._done.set()
