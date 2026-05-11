"""Task abstraction for the runtime engine.

A `Task` is a unit of work submitted to the engine. It has:

- A stable id (so coordinators can reference it)
- A lifecycle: queued -> running -> {completed, failed, cancelled}
- A budget: cost / tokens / turns / wall-clock — exceeding cancels the task
- An event stream: tool calls, tool results, text, status changes — all observable
- Parent/child relations for delegation (subtasks)
- Result fields populated on terminal status

Tasks are thread-safe data containers; the engine owns mutation.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from queue import Queue, Empty
from typing import Any, Iterator, Optional


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)


class BudgetExceeded(Exception):
    """Raised when a Task has burned through its budget. Engine catches this and
    transitions the task to FAILED (or CANCELLED if the budget was zero from
    the start)."""


@dataclass
class Budget:
    """Per-task resource ceiling. Any cap set to None means no limit.

    The agent loop calls `Budget.check(usage_so_far)` between turns; a cap that
    has been exceeded raises BudgetExceeded.
    """
    max_cost_usd: Optional[float] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_turns: Optional[int] = 25
    deadline_seconds: Optional[float] = None  # wall-clock from task start

    def check(
        self,
        *,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        turns: int,
        elapsed_seconds: float,
    ) -> None:
        if self.max_cost_usd is not None and cost_usd > self.max_cost_usd:
            raise BudgetExceeded(f"cost {cost_usd:.4f} > {self.max_cost_usd}")
        if self.max_input_tokens is not None and input_tokens > self.max_input_tokens:
            raise BudgetExceeded(f"input_tokens {input_tokens} > {self.max_input_tokens}")
        if self.max_output_tokens is not None and output_tokens > self.max_output_tokens:
            raise BudgetExceeded(f"output_tokens {output_tokens} > {self.max_output_tokens}")
        if self.max_turns is not None and turns > self.max_turns:
            raise BudgetExceeded(f"turns {turns} > {self.max_turns}")
        if self.deadline_seconds is not None and elapsed_seconds > self.deadline_seconds:
            raise BudgetExceeded(f"elapsed {elapsed_seconds:.1f}s > {self.deadline_seconds}s")


@dataclass
class TaskEvent:
    """A single observable thing that happened during task execution.

    `kind` is the event type (see below for the vocabulary the runtime emits).
    `data` is event-specific payload. `ts` is monotonic seconds since task start.

    Event vocabulary (stable, coordination-engine-facing):
      - status_changed: {from, to}
      - tool_call: {name, input}
      - tool_result: {name, output, is_error}
      - text: {text}                  # final assistant text for the turn
      - thinking_summary: {text}      # summarized model reasoning
      - turn_complete: {usage}        # per-turn token/cost accounting
      - delegated: {child_task_id, instruction}
      - delegated_complete: {child_task_id, result}
      - error: {type, message}
    """
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "data": self.data, "ts": self.ts}


@dataclass
class TaskRecord:
    """Snapshotable view of a Task. Kept separate from `Task` so the engine
    can hand out immutable-ish snapshots without leaking the live event queue
    or threading primitives."""
    id: str
    parent_id: Optional[str]
    instruction: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    result: Optional[str]
    error: Optional[str]
    cost_usd: float
    input_tokens: int
    output_tokens: int
    turns: int
    children: list[str]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class Task:
    """A live task running (or about to run) in the engine.

    Holds the event queue, cancellation primitive, and final result. Mutated
    only by the worker thread executing the task and by the engine itself
    (for cancellation). Readers should use `snapshot()` for a coherent view.
    """

    def __init__(
        self,
        *,
        instruction: str,
        budget: Optional[Budget] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self.id: str = task_id or uuid.uuid4().hex[:12]
        self.parent_id = parent_id
        self.instruction = instruction
        self.budget = budget or Budget()
        self.metadata: dict[str, Any] = dict(metadata or {})

        self._status: TaskStatus = TaskStatus.QUEUED
        self._lock = threading.Lock()
        self._events: list[TaskEvent] = []
        self._event_subscribers: list[Queue] = []
        self._cancel_event = threading.Event()
        self._done_event = threading.Event()

        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.cost_usd: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.turns: int = 0
        self.children: list[str] = []

    # --- lifecycle ---------------------------------------------------------

    @property
    def status(self) -> TaskStatus:
        with self._lock:
            return self._status

    def set_status(self, new: TaskStatus) -> None:
        with self._lock:
            old = self._status
            self._status = new
        self.emit("status_changed", {"from": old.value, "to": new.value})
        if new is TaskStatus.RUNNING and self.started_at is None:
            self.started_at = time.time()
        if new.terminal:
            self.finished_at = time.time()
            self._done_event.set()

    def cancel(self) -> None:
        """Request cancellation. The worker thread checks `cancel_requested`
        between turns and at tool boundaries."""
        self._cancel_event.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the task reaches a terminal state. Returns True if
        the task finished within `timeout`, False otherwise."""
        return self._done_event.wait(timeout)

    # --- events ------------------------------------------------------------

    def emit(self, kind: str, data: Optional[dict[str, Any]] = None) -> TaskEvent:
        ev = TaskEvent(kind=kind, data=dict(data or {}))
        with self._lock:
            self._events.append(ev)
            subscribers = list(self._event_subscribers)
        for q in subscribers:
            q.put(ev)
        return ev

    def subscribe(self) -> Queue:
        """Get a queue that receives all future events. The caller is
        responsible for draining it. Past events are not replayed — use
        `events()` for that."""
        q: Queue = Queue()
        with self._lock:
            self._event_subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            if q in self._event_subscribers:
                self._event_subscribers.remove(q)

    def events(self) -> list[TaskEvent]:
        with self._lock:
            return list(self._events)

    def stream_events(self, timeout: Optional[float] = None) -> Iterator[TaskEvent]:
        """Yield events until the task reaches a terminal state.

        Includes events that already happened before subscription, then live
        events. Safe to call from any thread.
        """
        q = self.subscribe()
        try:
            seen = 0
            past = self.events()
            for ev in past:
                yield ev
                seen += 1
            while not self._done_event.is_set() or not q.empty():
                try:
                    ev = q.get(timeout=timeout if timeout is not None else 0.5)
                except Empty:
                    if self._done_event.is_set():
                        break
                    continue
                yield ev
        finally:
            self.unsubscribe(q)

    # --- snapshot ----------------------------------------------------------

    def snapshot(self) -> TaskRecord:
        with self._lock:
            return TaskRecord(
                id=self.id,
                parent_id=self.parent_id,
                instruction=self.instruction,
                status=self._status,
                created_at=self.created_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
                result=self.result,
                error=self.error,
                cost_usd=self.cost_usd,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                turns=self.turns,
                children=list(self.children),
                metadata=dict(self.metadata),
            )
