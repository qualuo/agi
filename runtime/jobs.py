"""Async job queue.

A `Job` is a single unit of agent work submitted against a `Session`. Jobs run
on a fixed-size thread pool; the coordination engine polls status, streams
events, or waits synchronously. Jobs carry their own event buffer so multiple
subscribers can replay deltas without racing the producer.

State machine:

    QUEUED → RUNNING → SUCCEEDED
                    ↘ FAILED
                    ↘ CANCELLED   (cooperative — set before next turn)

`cancel()` is best-effort: it flips the flag, but an in-flight SDK call won't
be aborted. The next turn after cancel checks the flag and stops cleanly.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobError(RuntimeError):
    pass


@dataclass
class JobEvent:
    seq: int
    ts: float
    kind: str  # "text_delta" | "tool_use" | "state" | "done" | "error"
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"seq": self.seq, "ts": self.ts, "kind": self.kind, "data": self.data}


class Job:
    def __init__(
        self,
        *,
        session_id: str,
        prompt: str,
        max_iterations: int = 25,
        metadata: dict | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.session_id = session_id
        self.prompt = prompt
        self.max_iterations = max_iterations
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.state = JobState.QUEUED
        self.result_text: str | None = None
        self.error: str | None = None
        self.usage_delta: dict | None = None
        self._cancel_requested = False
        self._event_lock = threading.Lock()
        self._event_cv = threading.Condition(self._event_lock)
        self._events: list[JobEvent] = []
        self._done = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def append_event(self, kind: str, data: dict | None = None) -> JobEvent:
        with self._event_cv:
            ev = JobEvent(seq=len(self._events), ts=time.time(), kind=kind, data=data or {})
            self._events.append(ev)
            self._event_cv.notify_all()
            return ev

    def events_since(self, seq: int, *, timeout: float | None = None) -> list[JobEvent]:
        """Block until events past `seq` are available, or return what we have."""
        with self._event_cv:
            deadline = None if timeout is None else time.time() + timeout
            while len(self._events) <= seq and not self._done.is_set():
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                self._event_cv.wait(timeout=remaining)
            return list(self._events[seq:])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "max_iterations": self.max_iterations,
            "metadata": self.metadata,
            "result_text": self.result_text,
            "error": self.error,
            "usage_delta": self.usage_delta,
            "cancel_requested": self._cancel_requested,
            "event_count": len(self._events),
        }


class JobManager:
    def __init__(self, *, max_workers: int = 8) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agi-job")
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, job: Job, runner: Callable[[Job], None]) -> Job:
        with self._lock:
            self._jobs[job.id] = job

        def _wrapper() -> None:
            job.started_at = time.time()
            job.state = JobState.RUNNING
            job.append_event("state", {"state": job.state.value})
            try:
                if job.is_cancel_requested():
                    job.state = JobState.CANCELLED
                else:
                    runner(job)
                    if job.is_cancel_requested():
                        job.state = JobState.CANCELLED
                    else:
                        job.state = JobState.SUCCEEDED
            except Exception as e:
                job.state = JobState.FAILED
                job.error = f"{type(e).__name__}: {e}"
                job.append_event("error", {"error": job.error})
            finally:
                job.finished_at = time.time()
                job.append_event("state", {"state": job.state.value})
                job.append_event("done", {})
                job._done.set()
                with job._event_cv:
                    job._event_cv.notify_all()

        with self._lock:
            self._futures[job.id] = self._pool.submit(_wrapper)
        return job

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def list(self, *, session_id: str | None = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if session_id is not None:
            jobs = [j for j in jobs if j.session_id == session_id]
        return jobs

    def cancel(self, jid: str) -> bool:
        job = self.get(jid)
        if job is None:
            return False
        job.request_cancel()
        return True

    def wait(self, jid: str, timeout: float | None = None) -> Job | None:
        job = self.get(jid)
        if job is None:
            return None
        job._done.wait(timeout=timeout)
        return job

    def shutdown(self, *, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)
