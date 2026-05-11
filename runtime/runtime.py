"""Top-level Runtime — the in-process face of the runtime engine.

A coordination engine drives the runtime through this object (in-process) or
through `runtime.server` (out-of-process HTTP). They share the same semantics
so a coordinator can move between them without code changes.

Responsibilities:
  - construct sessions via the configured agent factory
  - submit jobs to the job manager with a runner that calls into the session
    while honoring the session's budget and the job's cancel flag
  - expose health, capability manifest, and metrics
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from agi.costs import Usage
from runtime.budgets import Budget, BudgetError
from runtime.capabilities import build_manifest
from runtime.jobs import Job, JobManager, JobState
from runtime.metrics import Metrics, Timer
from runtime.sessions import AgentLike, Session, SessionManager


def _default_agent_factory(**kwargs) -> AgentLike:
    """Default backend: real Opus agent unless mock is requested."""
    backend = kwargs.pop("backend", os.environ.get("AGI_RUNTIME_BACKEND", "opus"))
    if backend == "mock":
        from runtime.mock_agent import MockAgent
        return MockAgent(**kwargs)
    from agi.agent import Agent
    # The real Agent is verbose by default; the runtime drives it
    # programmatically, so silence it unless explicitly enabled.
    kwargs.setdefault("verbose", False)
    return Agent(**kwargs)


class Runtime:
    def __init__(
        self,
        agent_factory: Callable[..., AgentLike] | None = None,
        *,
        root: Path | str | None = None,
        max_workers: int = 8,
        enable_traces: bool = True,
    ) -> None:
        self.sessions = SessionManager(
            agent_factory or _default_agent_factory,
            root=root,
            enable_traces=enable_traces,
        )
        self.jobs = JobManager(max_workers=max_workers)
        self.metrics = Metrics()
        self._started = time.time()

    def health(self) -> dict:
        return {
            "status": "ok",
            "uptime_seconds": time.time() - self._started,
            "sessions": len(self.sessions.list()),
            "jobs": len(self.jobs.list()),
        }

    def manifest(self) -> dict:
        return build_manifest()

    def metrics_snapshot(self) -> dict:
        return self.metrics.snapshot()

    def create_session(
        self,
        *,
        model: str | None = None,
        budget: Budget | dict | None = None,
        agent_kwargs: dict[str, Any] | None = None,
    ) -> Session:
        if isinstance(budget, dict):
            budget = Budget.from_dict(budget)
        session = self.sessions.create(model=model, budget=budget, agent_kwargs=agent_kwargs)
        self.metrics.incr("sessions_created")
        self.metrics.gauge("sessions_open", len(self.sessions.list()))
        return session

    def delete_session(self, sid: str) -> bool:
        ok = self.sessions.delete(sid)
        if ok:
            self.metrics.incr("sessions_deleted")
            self.metrics.gauge("sessions_open", len(self.sessions.list()))
        return ok

    def chat_sync(self, sid: str, prompt: str, *, max_iterations: int = 25) -> dict:
        """Blocking single-turn chat. Returns {text, usage, session, latency_ms}."""
        session = self.sessions.get(sid)
        if session is None:
            raise KeyError(f"unknown session {sid}")
        before = _usage_dict(session.agent.usage)
        with Timer(self.metrics, "chat_sync_ms"):
            text = session.run_turn(prompt, max_iterations=max_iterations)
        self.metrics.incr("turns_total")
        delta = _usage_diff(before, _usage_dict(session.agent.usage))
        return {
            "text": text,
            "session": session.info().to_dict(),
            "usage_delta": delta,
        }

    def submit_job(self, sid: str, prompt: str, *, max_iterations: int = 25, metadata: dict | None = None) -> Job:
        session = self.sessions.get(sid)
        if session is None:
            raise KeyError(f"unknown session {sid}")
        job = Job(
            session_id=sid,
            prompt=prompt,
            max_iterations=max_iterations,
            metadata=metadata,
        )
        self.metrics.incr("jobs_submitted")
        return self.jobs.submit(job, lambda j: self._run_job(session, j))

    def _run_job(self, session: Session, job: Job) -> None:
        """Job runner. Bridges the cancel flag and streams text chunks as events."""
        before = _usage_dict(session.agent.usage)

        # Hook into the mock agent's chunk callback if available so the SSE
        # stream gets non-trivial event traffic. Real Agent streams to stdout;
        # wiring real-Agent streaming into job events is a larger refactor and
        # out of scope here.
        prior = getattr(session.agent, "on_text_chunk", None)
        if hasattr(session.agent, "on_text_chunk"):
            session.agent.on_text_chunk = lambda c: job.append_event("text_delta", {"text": c})  # type: ignore[attr-defined]

        try:
            if job.is_cancel_requested():
                return
            with Timer(self.metrics, "job_run_ms"):
                text = session.run_turn(job.prompt, max_iterations=job.max_iterations)
            job.result_text = text
            self.metrics.incr("jobs_succeeded")
            self.metrics.incr("turns_total")
        except BudgetError as e:
            self.metrics.incr("budget_exceeded")
            raise
        except Exception:
            self.metrics.incr("jobs_failed")
            raise
        finally:
            if hasattr(session.agent, "on_text_chunk"):
                session.agent.on_text_chunk = prior  # type: ignore[attr-defined]
            job.usage_delta = _usage_diff(before, _usage_dict(session.agent.usage))

    def wait(self, jid: str, timeout: float | None = None) -> Job | None:
        return self.jobs.wait(jid, timeout=timeout)

    def shutdown(self) -> None:
        self.jobs.shutdown(wait=False)


def _usage_dict(u: Usage) -> dict:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": u.cache_creation_input_tokens,
        "cache_read_input_tokens": u.cache_read_input_tokens,
        "turns": u.turns,
    }


def _usage_diff(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before}
