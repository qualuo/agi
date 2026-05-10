"""Runtime engine.

A stable, programmatic surface that an external **coordination engine** can
drive. The Agent in agent.py is the *worker*; the Runtime here is the *control
plane* around it.

Why this exists: an agent loop is interesting in a REPL but not deployable as
infrastructure. To plug into a coordinator (DAG executor, workflow system, the
Coordinator in coordinator.py, or any external scheduler in another language),
you need:

- **Sessions** — long-lived agent instances with stable IDs, isolated memory,
  per-session role / system prompt overrides, restart-safe.
- **Jobs** — units of work submitted to a session. Each has a budget, a
  cancel signal, an event stream, an outcome record (cost, latency, output).
- **Concurrency** — multiple jobs run in parallel without the coordinator
  having to manage threads itself.
- **Budget enforcement** — per-job $ and iteration ceilings, checked between
  agent turns. A coordinator that submits 1000 jobs needs to know none of
  them will silently spend $50.
- **Cancellation** — cooperative; a coordinator can stop a job whose result
  is no longer needed (the upstream branch failed, the user navigated away,
  a faster competitor returned first).
- **Observability** — events streamed live (text deltas, tool calls, status
  transitions) plus a metrics snapshot for scrape endpoints.
- **Snapshot/restore** — runtime state survives process restarts so a
  coordinator can persist work-in-flight.

This module is dependency-free Python (stdlib only). The HTTP wrapper in
server.py exposes it over the network for non-Python coordinators.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterator


# ---------- public types ----------


class BudgetExceeded(RuntimeError):
    """Raised inside the agent loop when a job exceeds its budget."""


class JobCanceled(RuntimeError):
    """Raised inside the agent loop when a job's cancel signal fires."""


@dataclass
class Event:
    """One observable thing that happened during a job."""
    job_id: str
    ts: float
    kind: str           # status | text_delta | thinking_delta | tool_use | tool_result | error
    payload: dict


@dataclass
class JobRecord:
    """Snapshot of a job's state. Safe to serialize."""
    id: str
    session_id: str
    prompt: str
    status: str         # queued | running | succeeded | failed | canceled
    created_ts: float
    started_ts: float | None = None
    finished_ts: float | None = None
    output: str = ""
    error: str | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    budget_usd: float | None = None
    max_iterations: int = 25
    parent_job_id: str | None = None  # set when spawned via delegate
    metadata: dict = field(default_factory=dict)


@dataclass
class SessionRecord:
    """Snapshot of a session's state. Safe to serialize."""
    id: str
    role: str | None
    system_prompt: str | None
    model: str
    created_ts: float
    job_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ---------- internal handles ----------


class Job:
    """Live handle to a job. Holds the threading primitives the runtime needs;
    the public state is `record`."""

    def __init__(self, record: JobRecord) -> None:
        self.record = record
        self.cancel_event = threading.Event()
        self.done_event = threading.Event()
        self.events: queue.Queue[Event | None] = queue.Queue()
        # When the worker thread exits it pushes None as a sentinel.

    def emit(self, kind: str, payload: dict | None = None) -> None:
        ev = Event(job_id=self.record.id, ts=time.time(), kind=kind, payload=payload or {})
        self.events.put(ev)


class Session:
    """Live handle to a session. Holds the Agent instance and its lock."""

    def __init__(self, record: SessionRecord, agent_factory: Callable[[SessionRecord], Any]) -> None:
        self.record = record
        # The Agent is constructed lazily — many sessions, few hot at a time.
        self._agent = None
        self._agent_factory = agent_factory
        self._lock = threading.Lock()  # serialize jobs within a session

    def agent(self):
        if self._agent is None:
            self._agent = self._agent_factory(self.record)
        return self._agent

    @property
    def lock(self) -> threading.Lock:
        return self._lock


# ---------- the runtime ----------


class Runtime:
    """The control plane. Thread-safe.

    Coordinator usage (sketch):

        rt = Runtime()
        sid = rt.create_session(role="researcher").id
        job = rt.submit(sid, "Summarize https://example.com", budget_usd=0.10)
        for ev in rt.stream(job.id):
            ...                                    # forward to caller
        result = rt.await_job(job.id, timeout=60)
        print(result.output, result.cost_usd)

    The runtime owns a ThreadPoolExecutor; jobs within a session are
    serialized by a per-session lock so the same Agent instance isn't
    re-entered concurrently. Jobs across different sessions run in parallel
    up to the pool size.
    """

    def __init__(
        self,
        *,
        agent_factory: Callable[[SessionRecord], Any] | None = None,
        max_workers: int = 8,
        snapshot_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if agent_factory is None:
            # Default factory binds `self` so the spawned Agent gets a
            # `delegate` tool that can spawn child sessions on this runtime.
            self._agent_factory = lambda session: _default_agent_factory(self, session)
        else:
            self._agent_factory = agent_factory
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agi-job")
        self._sessions: dict[str, Session] = {}
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None
        self._snapshot_lock = threading.Lock()
        if self._snapshot_path and self._snapshot_path.exists():
            self._restore_unlocked()

    # ----- session lifecycle -----

    def create_session(
        self,
        *,
        role: str | None = None,
        system_prompt: str | None = None,
        model: str = "claude-opus-4-7",
        metadata: dict | None = None,
        session_id: str | None = None,
    ) -> SessionRecord:
        record = SessionRecord(
            id=session_id or _new_id("sess"),
            role=role,
            system_prompt=system_prompt,
            model=model,
            created_ts=time.time(),
            metadata=metadata or {},
        )
        with self._lock:
            if record.id in self._sessions:
                raise ValueError(f"session {record.id} already exists")
            self._sessions[record.id] = Session(record, self._agent_factory)
        self._maybe_snapshot()
        return record

    def get_session(self, session_id: str) -> SessionRecord:
        return self._session(session_id).record

    def list_sessions(self) -> list[SessionRecord]:
        with self._lock:
            return [s.record for s in self._sessions.values()]

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        self._maybe_snapshot()

    # ----- job lifecycle -----

    def submit(
        self,
        session_id: str,
        prompt: str,
        *,
        budget_usd: float | None = None,
        max_iterations: int = 25,
        parent_job_id: str | None = None,
        metadata: dict | None = None,
    ) -> JobRecord:
        session = self._session(session_id)
        record = JobRecord(
            id=_new_id("job"),
            session_id=session_id,
            prompt=prompt,
            status="queued",
            created_ts=time.time(),
            budget_usd=budget_usd,
            max_iterations=max_iterations,
            parent_job_id=parent_job_id,
            metadata=metadata or {},
        )
        job = Job(record)
        with self._lock:
            self._jobs[record.id] = job
            session.record.job_ids.append(record.id)
        self._executor.submit(self._run_job, session, job)
        self._maybe_snapshot()
        return record

    def get_job(self, job_id: str) -> JobRecord:
        return self._job(job_id).record

    def list_jobs(self, *, session_id: str | None = None) -> list[JobRecord]:
        with self._lock:
            jobs = list(self._jobs.values())
        records = [j.record for j in jobs]
        if session_id is not None:
            records = [r for r in records if r.session_id == session_id]
        return records

    def cancel(self, job_id: str) -> JobRecord:
        job = self._job(job_id)
        job.cancel_event.set()
        return job.record

    def await_job(self, job_id: str, timeout: float | None = None) -> JobRecord:
        job = self._job(job_id)
        if not job.done_event.wait(timeout=timeout):
            raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
        return job.record

    def stream(self, job_id: str, timeout: float | None = None) -> Iterator[Event]:
        """Yield events as they happen, ending when the job finishes."""
        job = self._job(job_id)
        deadline = time.time() + timeout if timeout else None
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            try:
                ev = job.events.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"stream for job {job_id} timed out")
            if ev is None:
                return
            yield ev

    # ----- observability -----

    def metrics(self) -> dict:
        with self._lock:
            jobs = list(self._jobs.values())
        by_status: dict[str, int] = {}
        total_cost = 0.0
        total_in = 0
        total_out = 0
        for j in jobs:
            by_status[j.record.status] = by_status.get(j.record.status, 0) + 1
            total_cost += j.record.cost_usd
            total_in += j.record.input_tokens
            total_out += j.record.output_tokens
        return {
            "sessions": len(self._sessions),
            "jobs_total": len(jobs),
            "jobs_by_status": by_status,
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
        }

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    # ----- snapshot/restore -----

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "sessions": [asdict(s.record) for s in self._sessions.values()],
                "jobs": [asdict(j.record) for j in self._jobs.values()],
            }

    def _maybe_snapshot(self) -> None:
        if self._snapshot_path is None:
            return
        # Serialize writes so concurrent submitters don't race on the same
        # tmp file. The lock is cheap; snapshot frequency is low.
        with self._snapshot_lock:
            data = self.snapshot()
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._snapshot_path.with_suffix(
                self._snapshot_path.suffix + f".tmp.{uuid.uuid4().hex[:8]}"
            )
            tmp.write_text(json.dumps(data, default=str))
            tmp.replace(self._snapshot_path)

    def _restore_unlocked(self) -> None:
        data = json.loads(self._snapshot_path.read_text())
        for sd in data.get("sessions", []):
            rec = SessionRecord(**sd)
            self._sessions[rec.id] = Session(rec, self._agent_factory)
        for jd in data.get("jobs", []):
            rec = JobRecord(**jd)
            # Jobs that were running when we snapshotted are not resumed —
            # an honest snapshot marks them failed-on-restore so the
            # coordinator can decide what to do.
            if rec.status in ("queued", "running"):
                rec.status = "failed"
                rec.error = "interrupted by runtime restart"
                rec.finished_ts = rec.finished_ts or time.time()
            job = Job(rec)
            job.done_event.set()
            job.events.put(None)
            self._jobs[rec.id] = job

    # ----- internals -----

    def _session(self, session_id: str) -> Session:
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"no session {session_id}")
        return s

    def _job(self, job_id: str) -> Job:
        with self._lock:
            j = self._jobs.get(job_id)
        if j is None:
            raise KeyError(f"no job {job_id}")
        return j

    def _run_job(self, session: Session, job: Job) -> None:
        rec = job.record
        rec.status = "running"
        rec.started_ts = time.time()
        job.emit("status", {"status": "running"})
        try:
            with session.lock:
                agent = session.agent()
                # Drive the agent. The agent honors cancel/budget if it
                # supports them; if not, it'll just run to completion.
                output = self._call_agent(agent, job)
            rec.output = output
            rec.status = "succeeded"
            job.emit("status", {"status": "succeeded"})
        except JobCanceled:
            rec.status = "canceled"
            rec.error = "canceled"
            job.emit("status", {"status": "canceled"})
        except BudgetExceeded as e:
            rec.status = "failed"
            rec.error = f"budget exceeded: {e}"
            job.emit("error", {"error": rec.error})
        except Exception as e:
            rec.status = "failed"
            rec.error = f"{type(e).__name__}: {e}"
            job.emit("error", {"error": rec.error})
        finally:
            rec.finished_ts = time.time()
            # Roll up usage from the agent if available
            self._collect_usage(session, job)
            job.events.put(None)
            job.done_event.set()
            self._maybe_snapshot()

    def _call_agent(self, agent, job: Job) -> str:
        """Drive the agent. Pass through control signals if the agent supports
        them; otherwise call chat() directly. The Agent in agent.py was
        extended to take `cancel_event`, `budget_usd`, `event_sink`."""
        kwargs = {"max_iterations": job.record.max_iterations}
        if hasattr(agent, "chat_controlled"):
            return agent.chat_controlled(
                job.record.prompt,
                cancel_event=job.cancel_event,
                budget_usd=job.record.budget_usd,
                event_sink=lambda kind, payload: job.emit(kind, payload),
                **kwargs,
            )
        return agent.chat(job.record.prompt, **kwargs)

    def _collect_usage(self, session: Session, job: Job) -> None:
        agent = session._agent
        if agent is None or not hasattr(agent, "usage"):
            return
        u = agent.usage
        # The Agent's `usage` is cumulative across the session; we report
        # the *delta* attributable to this job. Sessions track running
        # totals in metadata for that subtraction.
        prev = session.record.metadata.get("usage_prev", {})
        in_delta = u.input_tokens - prev.get("input_tokens", 0)
        out_delta = u.output_tokens - prev.get("output_tokens", 0)
        cost = u.cost_usd(session.record.model) - prev.get("cost_usd", 0.0)
        job.record.input_tokens = max(0, in_delta)
        job.record.output_tokens = max(0, out_delta)
        job.record.cost_usd = max(0.0, round(cost, 6))
        session.record.metadata["usage_prev"] = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cost_usd": u.cost_usd(session.record.model),
        }


# ---------- defaults ----------


def _default_agent_factory(runtime: "Runtime", session: SessionRecord):
    """Build a real Agent for a session. Wires the agent back to its runtime
    so the `delegate` tool is registered and child sessions inherit the
    same control plane. Imported lazily so the runtime module itself doesn't
    require anthropic to be installed for tests."""
    from agi.agent import Agent
    kwargs: dict[str, Any] = {
        "model": session.model,
        "verbose": False,
        "runtime": runtime,
        "session_id": session.id,
    }
    if session.system_prompt is not None:
        kwargs["system_prompt"] = session.system_prompt
    return Agent(**kwargs)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
