"""Task lifecycle and store.

A `Task` is the unit of work the runtime executes. The state machine is:

    QUEUED -> RUNNING -> SUCCEEDED
                       \\-> FAILED
                       \\-> CANCELLED

`TaskStore` keeps tasks in memory keyed by id with an idempotency index on
client-supplied `dedup_key`. Tasks persist for the lifetime of the runtime;
durability is the coordination engine's responsibility (it owns the task
graph and can resubmit if the runtime dies).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED)


@dataclass
class TaskSpec:
    """What a caller submits."""

    kind: str  # "chat" | "plan" | "critique" | "skill" | <registered handler>
    input: dict[str, Any]
    dedup_key: str | None = None
    parent_id: str | None = None
    budget_tokens: int | None = None  # cap input+output tokens
    budget_seconds: float | None = None
    role: str | None = None  # planner | executor | critic | None=default
    tags: list[str] = field(default_factory=list)


@dataclass
class Task:
    id: str
    spec: TaskSpec
    status: TaskStatus = TaskStatus.QUEUED
    created_ts: float = field(default_factory=time.time)
    started_ts: float | None = None
    finished_ts: float | None = None
    result: Any | None = None
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    children: list[str] = field(default_factory=list)
    critic_score: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def request_cancel(self) -> None:
        self._cancel.set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "spec": asdict(self.spec),
            "status": self.status.value,
            "created_ts": self.created_ts,
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "result": self.result,
            "error": self.error,
            "usage": dict(self.usage),
            "cost_usd": self.cost_usd,
            "children": list(self.children),
            "critic_score": self.critic_score,
        }


class TaskStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, Task] = {}
        self._dedup: dict[str, str] = {}  # dedup_key -> task_id

    def submit(self, spec: TaskSpec) -> Task:
        with self._lock:
            if spec.dedup_key and spec.dedup_key in self._dedup:
                existing = self._tasks[self._dedup[spec.dedup_key]]
                return existing
            task = Task(id=uuid.uuid4().hex[:16], spec=spec)
            self._tasks[task.id] = task
            if spec.dedup_key:
                self._dedup[spec.dedup_key] = task.id
            if spec.parent_id and spec.parent_id in self._tasks:
                self._tasks[spec.parent_id].children.append(task.id)
            return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def all(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

    def mark_running(self, task_id: str) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return
            t.status = TaskStatus.RUNNING
            t.started_ts = time.time()

    def mark_succeeded(self, task_id: str, result: Any, usage: dict[str, int] | None = None,
                       cost_usd: float = 0.0, critic_score: float | None = None) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return
            t.status = TaskStatus.SUCCEEDED
            t.finished_ts = time.time()
            t.result = result
            t.usage = usage or {}
            t.cost_usd = cost_usd
            t.critic_score = critic_score

    def mark_failed(self, task_id: str, error: str, usage: dict[str, int] | None = None,
                    cost_usd: float = 0.0) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return
            t.status = TaskStatus.FAILED
            t.finished_ts = time.time()
            t.error = error
            t.usage = usage or {}
            t.cost_usd = cost_usd

    def mark_cancelled(self, task_id: str) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return
            t.request_cancel()
            if t.status == TaskStatus.QUEUED:
                t.status = TaskStatus.CANCELLED
                t.finished_ts = time.time()
            # If RUNNING, the worker observes _cancel and finalizes.
