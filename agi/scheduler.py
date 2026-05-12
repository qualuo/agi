"""Parallel DAG scheduler — the runtime's contract with a coordination engine
when the work has shape, not just length.

The reference `Coordinator` walks a `Plan` step-by-step. That's enough for
demos and for plans that are inherently sequential. A real coordination
engine asking the runtime to execute a fan-out (research -> synthesize ->
review) plan wants three things this scheduler provides:

  1. DAG-aware parallelism. Steps whose `depends_on` are satisfied run
     concurrently up to `max_concurrent_steps`. The shape of the Plan,
     not the order it was authored in, governs scheduling.

  2. Retries with backoff. Transient failures (rate limits, flaky tools)
     are common in agent workloads. A `RetryPolicy` per scheduler is
     applied to every step uniformly; a coordinator can override on a
     per-step basis via `PlanStep.metadata["retry"]`.

  3. Per-plan budget + deadline enforcement, observable. Cost ceilings
     halt new dispatch; in-flight steps complete and then the plan ends
     in `budget_exhausted`. Events on the bus mirror every transition so
     a coordinator sees the same picture the scheduler does.

What this is NOT:
  - A multi-process scheduler. One Python process, one Runtime, one
    thread pool. Multi-process scaling is a coordination concern that
    rides on top of the JSON-RPC protocol.
  - A retry-until-success oracle. Hard failures (e.g. budget exhausted)
    don't retry; they fail fast and surface.

A `ParallelScheduler` is what a coordination engine talks to when it
wants the runtime to *execute a plan it already authored*. Anything
above plan-authoring lives in the coordination engine.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agi.coordinator import Plan, PlanStep, StepOutcome
from agi.events import Event
from agi.runtime import Runtime, SessionConfig


PLAN_SCHEDULED = "plan.scheduled"
PLAN_STEP_READY = "plan.step.ready"
PLAN_STEP_RUNNING = "plan.step.running"
PLAN_STEP_RETRY = "plan.step.retry"
PLAN_STEP_COMPLETED = "plan.step.completed"
PLAN_STEP_FAILED = "plan.step.failed"
PLAN_COMPLETED = "plan.completed"
PLAN_FAILED = "plan.failed"
PLAN_BUDGET_EXHAUSTED = "plan.budget_exhausted"
PLAN_CANCELLED = "plan.cancelled"


@dataclass
class RetryPolicy:
    """How transient step failures are retried.

    `backoff_seconds * (backoff_multiplier ** attempt)` between attempts.
    `max_attempts` is the total number of attempts (1 = no retry).
    """
    max_attempts: int = 1
    backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0


@dataclass
class SchedulerConfig:
    """Tuning for a `ParallelScheduler`.

    - `max_concurrent_steps`: ceiling on simultaneously-running steps,
      regardless of how many are ready. Protects against runaway
      fan-out blowing through token budgets.
    - `retry_policy`: applied uniformly unless a step overrides it.
    - `fail_fast`: if True, the first hard failure cancels remaining
      not-yet-dispatched steps. If False, the plan attempts as much as
      possible and reports per-step outcomes at the end.
    - `default_session_config`: template used to construct each step's
      session unless the step specifies a model.
    """
    max_concurrent_steps: int = 4
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    fail_fast: bool = False
    default_session_config: SessionConfig | None = None


@dataclass
class PlanExecution:
    """Live state of a plan as the scheduler runs it.

    Mutates while running; callers should read via `to_dict()` for a
    point-in-time snapshot. The fields populate over the plan's life:

      - status: queued → running → done | failed | cancelled
      - outcomes: keyed by step.id; populated as steps complete
      - failures: keyed by step.id; transient failures during retry too
      - cost_usd / duration_seconds: rolling, accurate at completion
    """
    id: str
    plan: Plan
    status: str = "queued"
    outcomes: dict[str, StepOutcome] = field(default_factory=dict)
    failures: dict[str, list[str]] = field(default_factory=dict)
    started_ts: float | None = None
    completed_ts: float | None = None
    cost_usd: float = 0.0
    budget_usd: float | None = None
    deadline_ts: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "step_count": len(self.plan.steps),
            "completed_count": sum(1 for o in self.outcomes.values() if o.status == "done"),
            "failed_count": sum(1 for o in self.outcomes.values() if o.status == "failed"),
            "outcomes": {sid: _outcome_to_dict(o) for sid, o in self.outcomes.items()},
            "failures": dict(self.failures),
            "started_ts": self.started_ts,
            "completed_ts": self.completed_ts,
            "cost_usd": self.cost_usd,
            "budget_usd": self.budget_usd,
            "deadline_ts": self.deadline_ts,
            "duration_seconds": (
                (self.completed_ts or time.time()) - self.started_ts
                if self.started_ts is not None
                else 0.0
            ),
            "metadata": dict(self.metadata),
        }


def _outcome_to_dict(o: StepOutcome) -> dict[str, Any]:
    return {
        "step_id": o.step_id,
        "task_id": o.task_id,
        "session_id": o.session_id,
        "status": o.status,
        "result": o.result,
        "error": o.error,
        "duration_seconds": o.duration_seconds,
        "cost_usd": o.cost_usd,
    }


class CycleError(Exception):
    """Plan has a dependency cycle or unresolved depends_on."""


def _validate_plan(plan: Plan) -> None:
    ids = {s.id for s in plan.steps}
    if len(ids) != len(plan.steps):
        raise CycleError("plan has duplicate step ids")
    for s in plan.steps:
        for dep in s.depends_on:
            if dep not in ids:
                raise CycleError(f"step {s.id} depends on unknown step {dep}")
    # Kahn's algorithm for cycle detection.
    indeg = {s.id: len(s.depends_on) for s in plan.steps}
    ready = [sid for sid, n in indeg.items() if n == 0]
    visited = 0
    deps_by: dict[str, list[str]] = {s.id: [] for s in plan.steps}
    for s in plan.steps:
        for dep in s.depends_on:
            deps_by[dep].append(s.id)
    while ready:
        sid = ready.pop()
        visited += 1
        for child in deps_by[sid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
    if visited != len(plan.steps):
        raise CycleError("plan has a dependency cycle")


class ParallelScheduler:
    """DAG-aware executor that drives a `Runtime` from a `Plan`.

    Usage:

        sched = ParallelScheduler(runtime, config=SchedulerConfig(
            max_concurrent_steps=4,
            retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.5),
        ))
        exec_id = sched.submit(plan, budget_usd=5.0, deadline_ts=time.time()+60)
        result = sched.wait(exec_id)

    Two operating modes:
      - submit() + wait(): asynchronous, scheduler thread runs the plan
      - run(): synchronous, blocks until completion or failure

    Both produce the same `PlanExecution`. submit() is the right
    integration for a coordination engine that wants to subscribe to
    events and reason about progress; run() is the right call for a
    script.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        config: SchedulerConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config or SchedulerConfig()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_steps)
        self._executions: dict[str, PlanExecution] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._done: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # --- public API ---------------------------------------------------

    def submit(
        self,
        plan: Plan,
        *,
        budget_usd: float | None = None,
        deadline_ts: float | None = None,
        metadata: dict[str, Any] | None = None,
        execution_id: str | None = None,
    ) -> str:
        """Schedule a plan. Returns execution id immediately; plan runs
        on a background thread."""
        _validate_plan(plan)
        eid = execution_id or uuid.uuid4().hex[:12]
        with self._lock:
            if eid in self._executions:
                raise ValueError(f"execution id already exists: {eid}")
            execution = PlanExecution(
                id=eid,
                plan=plan,
                budget_usd=budget_usd,
                deadline_ts=deadline_ts,
                metadata=metadata or {},
            )
            self._executions[eid] = execution
            self._cancel[eid] = threading.Event()
            self._done[eid] = threading.Event()
        self.runtime.bus.publish(Event(
            kind=PLAN_SCHEDULED,
            data={"execution_id": eid, "step_count": len(plan.steps),
                  "budget_usd": budget_usd, "deadline_ts": deadline_ts},
        ))
        # Run on a dedicated thread so submit returns immediately. The
        # executor's worker pool is for *steps*, not for plan orchestration.
        t = threading.Thread(target=self._drive, args=(eid,), daemon=True, name=f"plan-{eid}")
        t.start()
        return eid

    def run(
        self,
        plan: Plan,
        *,
        budget_usd: float | None = None,
        deadline_ts: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PlanExecution:
        """Synchronous version of submit + wait. Blocks until the plan
        terminates (done / failed / cancelled / budget_exhausted)."""
        eid = self.submit(plan, budget_usd=budget_usd, deadline_ts=deadline_ts, metadata=metadata)
        return self.wait(eid)

    def wait(self, execution_id: str, timeout: float | None = None) -> PlanExecution:
        with self._lock:
            done = self._done.get(execution_id)
        if done is None:
            raise KeyError(f"unknown execution id: {execution_id}")
        finished = done.wait(timeout=timeout)
        if not finished:
            raise TimeoutError(f"plan {execution_id} did not complete within {timeout}s")
        return self._executions[execution_id]

    def cancel(self, execution_id: str) -> bool:
        """Best-effort cancel: prevents new steps from being dispatched.
        In-flight steps complete normally; the plan ends in `cancelled`."""
        with self._lock:
            cancel = self._cancel.get(execution_id)
            execution = self._executions.get(execution_id)
        if cancel is None or execution is None:
            return False
        if execution.status in ("done", "failed", "cancelled", "budget_exhausted"):
            return False
        cancel.set()
        return True

    def get(self, execution_id: str) -> PlanExecution:
        try:
            return self._executions[execution_id]
        except KeyError as ke:
            raise KeyError(f"unknown execution id: {execution_id}") from ke

    def list_executions(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._executions.values()]

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    # --- internals ----------------------------------------------------

    def _drive(self, execution_id: str) -> None:
        execution = self._executions[execution_id]
        cancel = self._cancel[execution_id]
        done_event = self._done[execution_id]
        execution.status = "running"
        execution.started_ts = time.time()

        plan = execution.plan
        remaining: dict[str, PlanStep] = {s.id: s for s in plan.steps}
        completed: set[str] = set()
        failed: set[str] = set()
        # step_id -> Future
        in_flight: dict[str, Future] = {}
        terminal_status: str | None = None

        try:
            while remaining or in_flight:
                if cancel.is_set() and not in_flight:
                    terminal_status = "cancelled"
                    break

                if (execution.deadline_ts is not None
                        and time.time() > execution.deadline_ts
                        and not in_flight):
                    terminal_status = "failed"
                    self.runtime.bus.publish(Event(
                        kind=PLAN_FAILED,
                        data={"execution_id": execution_id, "reason": "deadline_exceeded"},
                    ))
                    break

                # Dispatch all ready steps up to the concurrency ceiling,
                # but only if we haven't been told to stop dispatching.
                dispatching = not cancel.is_set() and (
                    execution.budget_usd is None
                    or execution.cost_usd < execution.budget_usd
                )
                if dispatching:
                    ready = [
                        s for s in remaining.values()
                        if s.id not in in_flight
                        and set(s.depends_on).issubset(completed)
                        # If a dep failed, the step is unreachable.
                        and not set(s.depends_on).intersection(failed)
                    ]
                    ready.sort(key=lambda s: s.priority)
                    for step in ready:
                        if len(in_flight) >= self.config.max_concurrent_steps:
                            break
                        if (execution.budget_usd is not None
                                and execution.cost_usd >= execution.budget_usd):
                            break
                        self.runtime.bus.publish(Event(
                            kind=PLAN_STEP_READY,
                            data={"execution_id": execution_id, "step_id": step.id},
                        ))
                        fut = self._executor.submit(
                            self._execute_step, execution_id, step
                        )
                        in_flight[step.id] = fut

                # Drop unreachable steps (their deps failed). With fail_fast
                # this also covers anything else still queued.
                drop = []
                for sid, s in remaining.items():
                    if sid in in_flight:
                        continue
                    deps_fail = set(s.depends_on).intersection(failed)
                    if deps_fail:
                        drop.append(sid)
                    elif self.config.fail_fast and failed:
                        drop.append(sid)
                for sid in drop:
                    step = remaining.pop(sid)
                    execution.outcomes[sid] = StepOutcome(
                        step_id=sid,
                        task_id="",
                        session_id=None,
                        status="skipped",
                        result=None,
                        error=f"dependency failed: {sorted(set(step.depends_on).intersection(failed)) or 'fail_fast'}",
                        duration_seconds=0.0,
                        cost_usd=0.0,
                    )

                # If nothing's running and nothing's dispatchable, we're
                # either done or stuck (shouldn't happen post-validation).
                if not in_flight:
                    if not remaining:
                        break
                    # Everything left has unsatisfied deps and nothing's
                    # in flight to satisfy them. Mark stuck steps as
                    # skipped and finish.
                    for sid, step in list(remaining.items()):
                        execution.outcomes[sid] = StepOutcome(
                            step_id=sid,
                            task_id="",
                            session_id=None,
                            status="skipped",
                            result=None,
                            error="unreachable: dependencies never completed",
                            duration_seconds=0.0,
                            cost_usd=0.0,
                        )
                        remaining.pop(sid)
                    break

                # Wait for any one in-flight to finish, then loop.
                finished_step = self._wait_any(in_flight)
                outcome = in_flight.pop(finished_step).result()
                execution.outcomes[finished_step] = outcome
                execution.cost_usd += outcome.cost_usd
                remaining.pop(finished_step, None)
                if outcome.status == "done":
                    completed.add(finished_step)
                    self.runtime.bus.publish(Event(
                        kind=PLAN_STEP_COMPLETED,
                        data={
                            "execution_id": execution_id,
                            "step_id": finished_step,
                            "cost_usd": outcome.cost_usd,
                        },
                    ))
                else:
                    failed.add(finished_step)
                    self.runtime.bus.publish(Event(
                        kind=PLAN_STEP_FAILED,
                        data={
                            "execution_id": execution_id,
                            "step_id": finished_step,
                            "error": outcome.error,
                        },
                    ))

                if (execution.budget_usd is not None
                        and execution.cost_usd >= execution.budget_usd
                        and not in_flight):
                    terminal_status = "budget_exhausted"
                    self.runtime.bus.publish(Event(
                        kind=PLAN_BUDGET_EXHAUSTED,
                        data={"execution_id": execution_id, "cost_usd": execution.cost_usd},
                    ))
                    break
        finally:
            execution.completed_ts = time.time()
            if terminal_status is not None:
                execution.status = terminal_status
            elif failed:
                execution.status = "failed"
                self.runtime.bus.publish(Event(
                    kind=PLAN_FAILED,
                    data={"execution_id": execution_id, "failed": sorted(failed)},
                ))
            else:
                execution.status = "done"
                self.runtime.bus.publish(Event(
                    kind=PLAN_COMPLETED,
                    data={
                        "execution_id": execution_id,
                        "step_count": len(plan.steps),
                        "cost_usd": execution.cost_usd,
                    },
                ))
            done_event.set()

    @staticmethod
    def _wait_any(in_flight: dict[str, Future]) -> str:
        """Poll futures for the first one that's done. Polling because
        Future has no native multi-wait without `concurrent.futures.wait`
        whose API doesn't carry our step ids — easier to just poll."""
        while True:
            for sid, fut in in_flight.items():
                if fut.done():
                    return sid
            # Sleep just long enough to avoid pegging a core. Sub-step
            # latency is dominated by the agent call (seconds), not the
            # scheduling tick (ms).
            time.sleep(0.005)

    def _execute_step(self, execution_id: str, step: PlanStep) -> StepOutcome:
        execution = self._executions[execution_id]
        retry = self._effective_retry(step)
        attempt = 0
        last_error: str | None = None
        start = time.time()
        while attempt < retry.max_attempts:
            attempt += 1
            self.runtime.bus.publish(Event(
                kind=PLAN_STEP_RUNNING,
                data={
                    "execution_id": execution_id,
                    "step_id": step.id,
                    "attempt": attempt,
                },
            ))
            session_id: str | None = None
            try:
                cfg = self._build_session_config(execution, step)
                session_id = self.runtime.create_session(cfg, namespace=step.namespace)
                result_text = self.runtime.chat(session_id, step.prompt)
                session = self.runtime.get_session(session_id)
                cost = session.state.total_cost_usd
                try:
                    self.runtime.end_session(session_id)
                except KeyError:
                    pass
                return StepOutcome(
                    step_id=step.id,
                    task_id=f"sched-{execution_id}-{step.id}",
                    session_id=session_id,
                    status="done",
                    result=result_text,
                    error=None,
                    duration_seconds=time.time() - start,
                    cost_usd=cost,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                execution.failures.setdefault(step.id, []).append(last_error)
                # Clean up partial session if any.
                if session_id is not None:
                    try:
                        self.runtime.end_session(session_id)
                    except Exception:
                        pass
                if attempt < retry.max_attempts:
                    self.runtime.bus.publish(Event(
                        kind=PLAN_STEP_RETRY,
                        data={
                            "execution_id": execution_id,
                            "step_id": step.id,
                            "attempt": attempt,
                            "error": last_error,
                        },
                    ))
                    backoff = retry.backoff_seconds * (retry.backoff_multiplier ** (attempt - 1))
                    if backoff > 0:
                        time.sleep(backoff)
        return StepOutcome(
            step_id=step.id,
            task_id=f"sched-{execution_id}-{step.id}",
            session_id=None,
            status="failed",
            result=None,
            error=last_error,
            duration_seconds=time.time() - start,
            cost_usd=0.0,
        )

    def _build_session_config(self, execution: PlanExecution, step: PlanStep) -> SessionConfig:
        base = self.config.default_session_config or SessionConfig(use_skills=True)
        kwargs = dict(base.__dict__)
        kwargs["use_skills"] = step.use_skills
        kwargs["role"] = step.role
        kwargs["system_prompt_extra"] = (
            f"Role: {step.role}. Return a concise final answer."
        )
        # Per-step budget ceiling derived from the plan budget if any.
        if execution.budget_usd is not None:
            # Leave a per-step ceiling at the remaining plan budget so a
            # runaway step can't outspend the plan.
            remaining = max(0.0, execution.budget_usd - execution.cost_usd)
            kwargs["cost_ceiling_usd"] = remaining
        cfg = SessionConfig(**kwargs)
        if step.model:
            cfg.model = step.model
        return cfg

    def _effective_retry(self, step: PlanStep) -> RetryPolicy:
        override = step.metadata.get("retry") if step.metadata else None
        if isinstance(override, RetryPolicy):
            return override
        if isinstance(override, dict):
            return RetryPolicy(**override)
        return self.config.retry_policy
