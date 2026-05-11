"""Runtime engine for the agi harness.

A coordination engine drives one or more `Session`s through this module. Each
session wraps an `Agent` with isolated state (conversation history, memory,
usage), enforces a `Budget`, and exposes work as `Job`s — either synchronous
or queued for background execution.

Public surface lives here so coordination engines depend on a stable shape:

    from runtime import Runtime, Budget, JobState

    rt = Runtime()
    sid = rt.create_session(budget=Budget(max_usd=1.0))
    job = rt.submit(sid, "summarize ./README.md")
    rt.wait(job.id).result_text

`runtime.server` exposes the same surface over HTTP/JSON+SSE for out-of-process
coordinators. `runtime.client.Client` is a thin Python wrapper around it.
"""
from runtime.budgets import Budget, BudgetError
from runtime.capabilities import build_manifest
from runtime.jobs import Job, JobState, JobError
from runtime.runtime import Runtime
from runtime.sessions import Session, SessionInfo

__all__ = [
    "Budget",
    "BudgetError",
    "Job",
    "JobError",
    "JobState",
    "Runtime",
    "Session",
    "SessionInfo",
    "build_manifest",
]
