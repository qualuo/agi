"""RuntimeDriver — the canonical entry point a coordination engine uses.

A coordination engine that wants to drive this runtime end-to-end has, until
now, had to wire several primitives by hand:

  - `PreflightEstimator` to forecast cost/duration/p_success
  - `AdmissionAdvisor` to interpret the forecast against a tenant policy
  - `PolicyManager` to enforce per-tenant budgets and quotas
  - `Runtime` (or `RuntimePool`) to actually create sessions and dispatch chats
  - `EventBus` to observe progress
  - hand-rolled accounting to produce a billing summary

`RuntimeDriver` collapses that into one contract:

    request = TicketRequest(intent="…", tenant_id="acme", budget_usd=0.20)
    ticket  = driver.submit(request)
    for ev in ticket.stream():
        ...                                  # live progress
    receipt = ticket.result()                # billing-grade summary

Every ticket carries a `Decision` trace — every fork the driver took
(estimate, admission, downgrade, defer, route, dispatch, complete) is
recorded in order with timestamps. The trace is what a coordination
engine replays for audit, debugging, or causal inference.

What this is and is not:

  - This is the **runtime-side** contract. A coordination engine (in any
    language, in any process) talks to a `RuntimeDriver` via Python now,
    via JSON-RPC (`agi.protocol`) once exposed remotely.
  - This is not a planner. The driver receives an intent + budget. It
    does not decompose, retry-with-different-plan, or learn over plans.
    That's the coordination engine's job.
  - This is not a scheduler. For DAG-shaped work a coordination engine
    composes many `Ticket`s — or hands its `Plan` to `ParallelScheduler`
    directly. The driver concerns itself with one ticket at a time.

The driver is the line between "an SDK" and "a platform a coordination
engine can stake its product on": every ticket gets cost forecasts,
admission control, optional model downgrade, real-time event streams,
hard budget ceilings, and a JSON-serializable receipt.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agi.events import (
    CHAT_COMPLETED,
    ERROR,
    SESSION_ENDED,
    USAGE_UPDATED,
    Event,
    EventBus,
)
from agi.preflight import (
    ADMIT,
    DEFER,
    DOWNGRADE,
    REJECT,
    AdmissionAdvice,
    AdmissionAdvisor,
    PreflightEstimator,
)
from agi.runtime import Runtime, SessionConfig


# --- ticket status ----------------------------------------------------

PENDING = "pending"
ESTIMATING = "estimating"
DEFERRED = "deferred"
REJECTED = "rejected"
DISPATCHED = "dispatched"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

# --- decision kinds (causal trace) -----------------------------------

D_ESTIMATE = "estimate"
D_ADMISSION = "admission"
D_DOWNGRADE = "downgrade"
D_DEFER = "defer"
D_REJECT = "reject"
D_ROUTE = "route"
D_DISPATCH = "dispatch"
D_COMPLETE = "complete"
D_FAIL = "fail"
D_CANCEL = "cancel"


@dataclass
class TicketRequest:
    """What a coordination engine hands the driver.

    `budget_usd` becomes a hard ceiling on the session's accumulated cost
    (enforced by `Session._enforce_budget`). `deadline_ts` is advisory at
    this layer — a coordination engine waiting on `ticket.result()` is
    expected to honor its own deadline.
    """
    intent: str
    tenant_id: str | None = None
    budget_usd: float | None = None
    deadline_ts: float | None = None
    config: SessionConfig | None = None
    allow_downgrade: bool = True
    namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """One causal step in a ticket's lifecycle. The ordered list of
    decisions IS the explanation for why a ticket cost what it cost and
    landed where it did."""
    kind: str
    ts: float
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Receipt:
    """Billing-grade summary. JSON-serializable; one Receipt per Ticket.

    Persisting receipts gives a coordination engine (and the operator)
    everything needed to: bill a tenant, compute cost variance vs.
    forecast, attribute cost back to which decision tier (forecast,
    downgrade, retry, dispatch) drove the spend, and replay any past
    ticket for audit."""
    ticket_id: str
    intent: str
    status: str
    tenant_id: str | None = None
    model: str | None = None
    node_id: str | None = None
    session_id: str | None = None
    final_text: str | None = None
    error: str | None = None
    estimated_cost_usd: float = 0.0
    estimated_p_success: float = 0.0
    actual_cost_usd: float = 0.0
    actual_duration_s: float = 0.0
    submitted_ts: float = 0.0
    completed_ts: float | None = None
    decisions: list[Decision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decisions"] = [x for x in (decision_to_dict(x) for x in self.decisions)]
        return d


def decision_to_dict(d: Decision | dict[str, Any]) -> dict[str, Any]:
    if isinstance(d, Decision):
        return d.to_dict()
    return d  # already serialized


# --- ticket ----------------------------------------------------------


class Ticket:
    """Handle a coordination engine holds per submitted request.

    Thread-safe. The driver populates state from a worker thread; readers
    (`stream()`, `result()`, `status`) may be called from any thread.
    """

    def __init__(self, request: TicketRequest, ticket_id: str | None = None) -> None:
        self.id = ticket_id or uuid.uuid4().hex[:12]
        self.request = request
        self._status = PENDING
        self._receipt = Receipt(
            ticket_id=self.id,
            intent=request.intent,
            status=PENDING,
            tenant_id=request.tenant_id,
            submitted_ts=time.time(),
        )
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._event_q: queue.Queue[Event | None] = queue.Queue()
        self._cancel_requested = threading.Event()
        # Filled when the driver actually creates a session.
        self.session_id: str | None = None
        self.node_id: str | None = None

    # --- status / receipt -----------------------------------------

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def receipt(self) -> Receipt:
        """Live receipt; mutates until the ticket completes. Callers
        wanting a stable snapshot should call `result()` first."""
        with self._lock:
            return self._receipt

    def decisions(self) -> list[Decision]:
        with self._lock:
            return list(self._receipt.decisions)

    # --- consumer API ---------------------------------------------

    def stream(self, *, timeout: float | None = None) -> Iterator[Event]:
        """Yield events as the ticket progresses. Terminates when the
        ticket completes (or the optional per-event timeout elapses).

        A coordination engine plugs this into its UI or its own observer
        — events are the runtime's first-class progress channel and the
        driver passes them through unchanged so existing tooling works.
        """
        while True:
            try:
                ev = self._event_q.get(timeout=timeout)
            except queue.Empty:
                return
            if ev is None:
                return
            yield ev

    def result(self, *, timeout: float | None = None) -> Receipt:
        """Block until the ticket terminates, then return the receipt.
        Raises TimeoutError if the deadline elapses."""
        if not self._done.wait(timeout=timeout):
            raise TimeoutError(f"ticket {self.id} did not complete within {timeout}s")
        with self._lock:
            return self._receipt

    def cancel(self) -> None:
        """Request cancellation. Best-effort: an in-flight chat completes
        its current turn, then the session ends and the ticket moves to
        CANCELLED. Idempotent."""
        self._cancel_requested.set()

    # --- driver-internal helpers ----------------------------------

    def _push_event(self, event: Event) -> None:
        self._event_q.put(event)

    def _close_stream(self) -> None:
        self._event_q.put(None)

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._receipt.status = status

    def _add_decision(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._receipt.decisions.append(
                Decision(kind=kind, ts=time.time(), payload=payload)
            )

    def _finish(self) -> None:
        with self._lock:
            self._receipt.completed_ts = time.time()
        self._done.set()
        self._close_stream()


# --- driver ----------------------------------------------------------


class RuntimeDriver:
    """Single entry point a coordination engine drives.

    Wraps exactly one of:
      - a `Runtime` (in-process), or
      - a `RuntimePool` (in-process federation; the pool routes per ticket)

    Wires preflight + admission + governance + dispatch + receipts.

    Concurrency: tickets dispatch on a thread pool. `max_concurrent` caps
    in-flight dispatches; excess submissions queue. The driver itself is
    safe to call from any thread.
    """

    def __init__(
        self,
        *,
        runtime: Runtime | None = None,
        pool: Any | None = None,
        advisor: AdmissionAdvisor | None = None,
        estimator: PreflightEstimator | None = None,
        policy: Any | None = None,
        receipts_path: str | os.PathLike[str] | None = None,
        max_concurrent: int = 8,
    ) -> None:
        if runtime is None and pool is None:
            raise ValueError("RuntimeDriver requires either runtime= or pool=")
        if runtime is not None and pool is not None:
            raise ValueError("RuntimeDriver takes runtime= XOR pool=, not both")

        self.runtime = runtime
        self.pool = pool
        self.policy = policy

        # Prefer caller-supplied estimator/advisor; otherwise lean on the
        # runtime's own (the runtime now creates them by default).
        if estimator is None and runtime is not None:
            estimator = getattr(runtime, "estimator", None)
        if estimator is None:
            estimator = PreflightEstimator()
        self.estimator = estimator

        if advisor is None and runtime is not None:
            advisor = getattr(runtime, "admission_advisor", None)
        if advisor is None:
            advisor = AdmissionAdvisor(estimator, policy=policy, runtime=runtime)
        # If a caller supplied an advisor without a policy but we have one,
        # honor the explicit policy here.
        self.advisor = advisor

        self.receipts_path = Path(receipts_path) if receipts_path else None
        if self.receipts_path is not None:
            self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
            self.receipts_path.touch(exist_ok=True)

        self._max_concurrent = max_concurrent
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self._tickets: dict[str, Ticket] = {}
        self._lock = threading.Lock()
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "deferred": 0,
            "downgraded": 0,
            "cancelled": 0,
            "portfolio_batches": 0,
            "portfolio_skipped": 0,
        }
        # Lazy import to avoid an import cycle: portfolio depends on driver.
        self._portfolio_optimizer: Any | None = None

    @property
    def portfolio(self) -> Any:
        """Lazy `PortfolioOptimizer` bound to this driver's estimator.

        A coordination engine that wants to plan-only (no dispatch) can
        call `driver.portfolio.plan(...)` directly. The optimizer
        observes the same forecasts the driver uses for admission, so
        plan and admission decisions stay consistent.
        """
        if self._portfolio_optimizer is None:
            from agi.portfolio import PortfolioOptimizer
            self._portfolio_optimizer = PortfolioOptimizer(self.estimator)
        return self._portfolio_optimizer

    # --- public API -----------------------------------------------

    def submit(self, request: TicketRequest) -> Ticket:
        """Accept a request, run preflight+admission inline, then dispatch
        the chat on a worker thread. Returns immediately with a Ticket.

        The preflight + admission decisions happen synchronously so a
        rejected/deferred ticket surfaces its verdict to the caller
        without spinning up a worker.
        """
        ticket = Ticket(request)
        with self._lock:
            self._tickets[ticket.id] = ticket
            self._stats["submitted"] += 1

        # Preflight + admission inline so rejections are immediate.
        ticket._set_status(ESTIMATING)
        cfg = request.config or SessionConfig()
        if request.budget_usd is not None and cfg.cost_ceiling_usd is None:
            # Pass the ticket budget into the session as a hard ceiling.
            cfg = _clone_config(cfg, cost_ceiling_usd=request.budget_usd)
        advice = self.advisor.advise(
            prompt=request.intent,
            config=cfg,
            tenant_id=request.tenant_id,
        )
        ticket._receipt.estimated_cost_usd = advice.estimate.cost_usd
        ticket._receipt.estimated_p_success = advice.estimate.p_success
        ticket._receipt.model = advice.estimate.model
        ticket._add_decision(D_ESTIMATE, {
            "cost_usd": advice.estimate.cost_usd,
            "cost_p10_usd": advice.estimate.cost_p10_usd,
            "cost_p90_usd": advice.estimate.cost_p90_usd,
            "duration_s": advice.estimate.duration_s,
            "p_success": advice.estimate.p_success,
            "confidence": advice.estimate.confidence,
            "samples": advice.estimate.samples,
            "model": advice.estimate.model,
        })
        ticket._add_decision(D_ADMISSION, {
            "verdict": advice.verdict,
            "reason": advice.reason,
            "governance_code": advice.governance_code,
        })

        # Handle non-admit verdicts.
        if advice.verdict == REJECT:
            return self._terminate_ticket(ticket, REJECTED, error=advice.reason, stat_key="rejected")
        if advice.verdict == DEFER:
            ticket._add_decision(D_DEFER, {
                "reason": advice.reason,
                "retry_after_s": advice.retry_after_s,
            })
            return self._terminate_ticket(ticket, DEFERRED, error=advice.reason, stat_key="deferred")
        if advice.verdict == DOWNGRADE:
            if not request.allow_downgrade or advice.alternative is None:
                return self._terminate_ticket(ticket, REJECTED, error=advice.reason, stat_key="rejected")
            alt = advice.alternative
            cfg = _clone_config(cfg, model=alt["model"])
            ticket._receipt.model = alt["model"]
            ticket._add_decision(D_DOWNGRADE, {
                "from_model": advice.estimate.model,
                "to_model": alt["model"],
                "projected_cost_usd": alt.get("est_cost_usd"),
            })
            with self._lock:
                self._stats["downgraded"] += 1

        # Pick a runtime (pool routing) and stash the choice.
        runtime, node_id = self._pick_runtime(request.intent)
        ticket.node_id = node_id
        ticket._receipt.node_id = node_id
        ticket._add_decision(D_ROUTE, {"node_id": node_id})

        # Spin a worker thread for dispatch.
        ticket._set_status(DISPATCHED)
        t = threading.Thread(
            target=self._run_ticket,
            args=(ticket, runtime, cfg),
            name=f"driver-{ticket.id}",
            daemon=True,
        )
        t.start()
        return ticket

    def submit_sync(self, request: TicketRequest, *, timeout: float | None = None) -> Receipt:
        """Block until the ticket terminates. Convenience for callers
        that don't want to manage the streaming surface."""
        ticket = self.submit(request)
        return ticket.result(timeout=timeout)

    def submit_portfolio(
        self,
        requests: list[TicketRequest],
        *,
        total_budget_usd: float,
        value_weights: list[float] | None = None,
        candidate_models: list[str] | None = None,
        plan_only: bool = False,
        allow_skip: bool = True,
        method: str = "auto",
    ) -> tuple[list[Ticket | None], Any]:
        """Plan + dispatch a batch under a single shared budget.

        Picks one model per request (or "skip") to maximize total
        expected p_success subject to the budget, then submits each
        non-skipped allocation as a normal `Ticket`. Skipped requests
        return `None` in the parallel ticket list so the caller can
        keep input/output indices aligned.

        Returns `(tickets, plan)`. The plan is a `PortfolioPlan`
        carrying the expected_cost / expected_value / per-request
        decisions — durable, JSON-serializable, suitable for a
        coordination engine's audit log.

        Set `plan_only=True` to get a quote without dispatching.
        """
        from agi.portfolio import PortfolioOptimizer

        optimizer = self.portfolio
        if candidate_models is not None or method != "auto":
            # Honor caller overrides without mutating the shared instance.
            optimizer = PortfolioOptimizer(
                self.estimator,
                candidate_models=candidate_models
                or optimizer.candidate_models,
                value_floor=optimizer.value_floor,
            )
        plan = optimizer.plan(
            requests,
            total_budget_usd=total_budget_usd,
            value_weights=value_weights,
            candidate_models=candidate_models,
            allow_skip=allow_skip,
            method=method,
        )
        if plan_only:
            with self._lock:
                self._stats["portfolio_skipped"] += plan.skipped_count
            return [], plan

        tickets: list[Ticket | None] = []
        for alloc in plan.allocations:
            if alloc.skipped:
                tickets.append(None)
                continue
            # Clone the request with the optimizer-selected model.
            cfg = alloc.request.config or SessionConfig()
            cfg = _clone_config(cfg, model=alloc.chosen.model)
            req = TicketRequest(
                intent=alloc.request.intent,
                tenant_id=alloc.request.tenant_id,
                budget_usd=alloc.request.budget_usd,
                deadline_ts=alloc.request.deadline_ts,
                config=cfg,
                allow_downgrade=alloc.request.allow_downgrade,
                namespace=alloc.request.namespace,
                metadata={
                    **alloc.request.metadata,
                    "portfolio_chosen_model": alloc.chosen.model,
                    "portfolio_expected_cost_usd": alloc.chosen.estimated_cost_usd,
                    "portfolio_expected_p_success": alloc.chosen.estimated_p_success,
                },
            )
            tickets.append(self.submit(req))
        with self._lock:
            self._stats["portfolio_batches"] += 1
            self._stats["portfolio_skipped"] += plan.skipped_count
        return tickets, plan

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        with self._lock:
            return self._tickets.get(ticket_id)

    def tickets(self) -> list[Ticket]:
        with self._lock:
            return list(self._tickets.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    # --- worker --------------------------------------------------

    def _run_ticket(self, ticket: Ticket, runtime: Runtime, cfg: SessionConfig) -> None:
        """Run one ticket on a worker thread: create session, subscribe to
        its events, drive a single chat, finalize the receipt."""
        with self._sem:
            if ticket._cancel_requested.is_set():
                self._terminate_ticket(ticket, CANCELLED, error="cancelled before dispatch", stat_key="cancelled")
                return
            try:
                sid = runtime.create_session(cfg, namespace=ticket.request.namespace)
            except Exception as e:
                self._fail(ticket, f"create_session: {type(e).__name__}: {e}")
                return
            ticket.session_id = sid
            ticket._receipt.session_id = sid

            # Tenant accounting (best-effort; policy is None for unscoped).
            if self.policy is not None and ticket.request.tenant_id:
                try:
                    self.policy.session_started(ticket.request.tenant_id, sid)
                except Exception:
                    pass

            # Subscribe to this session's events so the ticket's stream
            # mirrors them. Filter at subscribe time.
            sub_id = runtime.bus.subscribe(ticket._push_event, session_id=sid)
            ticket._set_status(RUNNING)
            ticket._add_decision(D_DISPATCH, {"session_id": sid, "model": cfg.model})

            # If cancel comes in mid-flight we can't preempt the API call,
            # but we set the session's cancel flag so the next iteration
            # boundary bails. Wire a watcher.
            def _watch_cancel():
                ticket._cancel_requested.wait()
                try:
                    session = runtime.get_session(sid)
                    if session is not None:
                        session.cancel()
                except Exception:
                    pass
            watcher = threading.Thread(target=_watch_cancel, daemon=True)
            watcher.start()

            final_text: str | None = None
            error: str | None = None
            try:
                final_text = runtime.chat(sid, ticket.request.intent)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

            # Pull actual cost from the session and tear down.
            duration_s = 0.0
            actual_cost_usd = 0.0
            try:
                session = runtime.get_session(sid)
                if session is not None:
                    actual_cost_usd = session.state.total_cost_usd
                    duration_s = max(0.0, session.state.last_activity_ts - session.state.created_ts)
            except Exception:
                pass

            try:
                runtime.end_session(sid)
            except Exception:
                pass
            try:
                runtime.bus.unsubscribe(sub_id)
            except Exception:
                pass

            # Commit cost back to governance.
            if self.policy is not None and ticket.request.tenant_id:
                try:
                    self.policy.commit(
                        ticket.request.tenant_id,
                        cost_usd=actual_cost_usd,
                        kind="chat",
                    )
                    self.policy.session_ended(ticket.request.tenant_id, sid)
                except Exception:
                    pass

            ticket._receipt.actual_cost_usd = actual_cost_usd
            ticket._receipt.actual_duration_s = duration_s

            if ticket._cancel_requested.is_set():
                ticket._receipt.final_text = final_text
                ticket._add_decision(D_CANCEL, {"after_chat": final_text is not None})
                self._terminate_ticket(ticket, CANCELLED, error="cancelled", stat_key="cancelled")
                return
            if error is not None:
                self._fail(ticket, error)
                return

            ticket._receipt.final_text = final_text
            ticket._add_decision(D_COMPLETE, {
                "actual_cost_usd": actual_cost_usd,
                "duration_s": duration_s,
            })
            self._terminate_ticket(ticket, COMPLETED, stat_key="completed")

    # --- helpers --------------------------------------------------

    def _pick_runtime(self, prompt: str) -> tuple[Runtime, str | None]:
        if self.runtime is not None:
            return self.runtime, None
        # Pool path: ask the pool to select a node, return its runtime.
        node = self.pool.select(prompt=prompt)
        if node is None:
            raise RuntimeError("pool has no healthy nodes")
        return node.runtime, node.node_id

    def _terminate_ticket(
        self,
        ticket: Ticket,
        status: str,
        *,
        error: str | None = None,
        stat_key: str | None = None,
    ) -> Ticket:
        if error is not None:
            ticket._receipt.error = error
        ticket._set_status(status)
        if stat_key is not None:
            with self._lock:
                self._stats[stat_key] = self._stats.get(stat_key, 0) + 1
        # Persist BEFORE signalling done so result() observers see a
        # durable receipt on disk by the time the call returns.
        self._persist_receipt(ticket._receipt)
        ticket._finish()
        return ticket

    def _fail(self, ticket: Ticket, error: str) -> None:
        ticket._add_decision(D_FAIL, {"error": error})
        self._terminate_ticket(ticket, FAILED, error=error, stat_key="failed")

    def _persist_receipt(self, receipt: Receipt) -> None:
        if self.receipts_path is None:
            return
        try:
            with self.receipts_path.open("a") as f:
                f.write(json.dumps(receipt.to_dict(), default=str))
                f.write("\n")
        except Exception:
            # Persistence is best-effort; a coordination engine still has
            # the live receipt object via ticket.result().
            pass


def _clone_config(cfg: SessionConfig, **overrides: Any) -> SessionConfig:
    """Return a new SessionConfig copying `cfg` and applying overrides.

    SessionConfig is a dataclass; we keep this helper independent so a
    coordination engine using a SessionConfig subclass still gets a
    same-typed copy."""
    data = {**cfg.__dict__, **overrides}
    return type(cfg)(**data)
