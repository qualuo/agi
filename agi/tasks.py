"""Task scheduler — coordinator-friendly work queue on top of the Runtime.

A coordination engine often wants to submit *work* (a prompt, a budget, a
deadline) and let the runtime figure out which session executes it. Tasks
are the unit of work; the TaskQueue holds them; a TaskRunner drains them
against a Runtime.

Tasks are durable in memory only by default; persistence is opt-in via a
separate store (not implemented here — tasks are short-lived enough that
losing them on restart is acceptable for v1).

Status transitions:

    queued → running → done
                     ↘ failed
                     ↘ cancelled
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from agi.events import Event
from agi.runtime import Runtime, SessionConfig


TASK_QUEUED = "task.queued"
TASK_STARTED = "task.started"
TASK_COMPLETED = "task.completed"
TASK_FAILED = "task.failed"
TASK_CANCELLED = "task.cancelled"


@dataclass
class Task:
    id: str
    prompt: str
    session_config: SessionConfig
    priority: int = 0  # lower = higher priority
    deadline_ts: float | None = None
    status: str = "queued"
    result: str | None = None
    error: str | None = None
    session_id: str | None = None
    created_ts: float = field(default_factory=time.time)
    started_ts: float | None = None
    completed_ts: float | None = None
    attempts: int = 0
    max_attempts: int = 1
    namespace: str | None = None
    tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "priority": self.priority,
            "deadline_ts": self.deadline_ts,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "session_id": self.session_id,
            "created_ts": self.created_ts,
            "started_ts": self.started_ts,
            "completed_ts": self.completed_ts,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "namespace": self.namespace,
            "tag": self.tag,
            "session_config": self.session_config.__dict__,
        }


class TaskQueue:
    """In-memory priority queue. Lower priority value runs first."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def submit(self, task: Task) -> str:
        with self._lock:
            if task.id in self._tasks:
                raise ValueError(f"duplicate task id: {task.id}")
            self._tasks[task.id] = task
        return task.id

    def get(self, task_id: str) -> Task:
        t = self._tasks.get(task_id)
        if t is None:
            raise KeyError(task_id)
        return t

    def list(self, *, status: str | None = None, tag: str | None = None) -> list[Task]:
        out = list(self._tasks.values())
        if status is not None:
            out = [t for t in out if t.status == status]
        if tag is not None:
            out = [t for t in out if t.tag == tag]
        return out

    def next_runnable(self) -> Task | None:
        """Pop a ready task: queued, not past deadline, lowest priority first."""
        now = time.time()
        with self._lock:
            candidates = [
                t for t in self._tasks.values()
                if t.status == "queued" and (t.deadline_ts is None or t.deadline_ts > now)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda t: (t.priority, t.created_ts))
            chosen = candidates[0]
            chosen.status = "running"
            chosen.started_ts = now
            chosen.attempts += 1
            return chosen

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None or t.status not in ("queued", "running"):
                return False
            t.status = "cancelled"
            t.completed_ts = time.time()
            return True


class TaskRunner:
    """Drains a TaskQueue against a Runtime. Single-threaded by default;
    callers can spin up multiple TaskRunner instances for concurrency.

    Each tick:
      1. Pop the next runnable task (priority + deadline aware).
      2. Create a session sized to the task's SessionConfig.
      3. Call runtime.chat() with the task's prompt.
      4. Update task status. End the session.
      5. Emit task.* events on the bus.

    The runner is cooperative — callers control the loop via tick() / run()
    / stop(). Production deployments wrap this in a supervisor (systemd,
    k8s job, etc.).
    """

    def __init__(self, runtime: Runtime, queue: TaskQueue) -> None:
        self.runtime = runtime
        self.queue = queue
        self._stop = threading.Event()

    def tick(self) -> Task | None:
        task = self.queue.next_runnable()
        if task is None:
            return None
        self.runtime.bus.publish(Event(kind=TASK_STARTED, data={"task_id": task.id}))
        try:
            sid = self.runtime.create_session(task.session_config, namespace=task.namespace)
            task.session_id = sid
            result = self.runtime.chat(sid, task.prompt)
        except Exception as e:
            task.status = "failed" if task.attempts >= task.max_attempts else "queued"
            task.error = f"{type(e).__name__}: {e}"
            task.completed_ts = time.time() if task.status == "failed" else None
            self.runtime.bus.publish(Event(
                kind=TASK_FAILED,
                data={"task_id": task.id, "error": task.error, "attempts": task.attempts},
            ))
            return task
        finally:
            if task.session_id is not None:
                # End the session — coordinator can still introspect via runtime.
                try:
                    self.runtime.end_session(task.session_id)
                except KeyError:
                    pass

        task.status = "done"
        task.result = result
        task.completed_ts = time.time()
        self.runtime.bus.publish(Event(
            kind=TASK_COMPLETED,
            data={"task_id": task.id, "session_id": task.session_id},
        ))
        return task

    def run_until_empty(self, *, max_ticks: int = 1000) -> int:
        """Drain the queue. Returns the number of tasks executed."""
        executed = 0
        for _ in range(max_ticks):
            if self._stop.is_set():
                break
            task = self.tick()
            if task is None:
                break
            executed += 1
        return executed

    def stop(self) -> None:
        self._stop.set()


def submit_task(
    queue: TaskQueue,
    *,
    prompt: str,
    session_config: SessionConfig | None = None,
    priority: int = 0,
    deadline_ts: float | None = None,
    max_attempts: int = 1,
    namespace: str | None = None,
    tag: str | None = None,
) -> str:
    """Convenience wrapper around `TaskQueue.submit` that builds a Task."""
    task = Task(
        id=uuid.uuid4().hex[:12],
        prompt=prompt,
        session_config=session_config or SessionConfig(),
        priority=priority,
        deadline_ts=deadline_ts,
        max_attempts=max_attempts,
        namespace=namespace,
        tag=tag,
    )
    return queue.submit(task)
