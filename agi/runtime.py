"""Agent runtime engine.

The thing a coordination engine actually calls. The CLI in `agi.__main__`
is one consumer; the HTTP server in `agi.server` is another; any external
orchestrator (Temporal-style scheduler, a CrewAI-style multi-agent layer,
a per-tenant queue) is another.

Contract: a `Runtime` owns a registry of `Run`s. A `Run` is one agent task
with an id, a status, a result, a cost, and an event stream. Submission is
non-blocking; the caller can poll, stream events, or wait. Cancellation and
budget enforcement are cooperative — checked between turns and tool calls.

This file is small on purpose. It is the seam between "an Agent that
streams text" and "a service a coordinator dispatches to".
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Iterator

from agi.costs import Usage
from agi.memory import Memory


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMED_OUT = "timed_out"


TERMINAL = {
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
    RunStatus.BUDGET_EXCEEDED,
    RunStatus.TIMED_OUT,
}


class BudgetExceeded(Exception):
    """Raised by the Agent loop when cost_ceiling_usd is exceeded."""


class Cancelled(Exception):
    """Raised by the Agent loop when its cancel_event is set."""


@dataclass
class Event:
    ts: float
    run_id: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SENTINEL = object()


class _EventBus:
    """Per-run event bus. Stores history (for late subscribers) and fans out
    to live subscribers via per-subscriber queues. Bounded history caps
    memory; bounded subscriber queues drop the oldest event if a slow
    subscriber falls behind (we'd rather lose events than block the agent).
    """

    def __init__(self, history_cap: int = 1024, subscriber_cap: int = 1024) -> None:
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._history_cap = history_cap
        self._subs: list[queue.Queue] = []
        self._subscriber_cap = subscriber_cap
        self._closed = False

    def emit(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_cap:
                # drop oldest; keep most recent N for late subscribers
                del self._history[: len(self._history) - self._history_cap]
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # subscriber is slow; drop the oldest to make room
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(_SENTINEL)
            except queue.Full:
                pass

    def history(self) -> list[Event]:
        with self._lock:
            return list(self._history)

    def subscribe(self, *, replay: bool = True) -> Iterator[Event]:
        q: queue.Queue = queue.Queue(maxsize=self._subscriber_cap)
        with self._lock:
            if replay:
                for ev in self._history:
                    try:
                        q.put_nowait(ev)
                    except queue.Full:
                        pass
            if self._closed:
                try:
                    q.put_nowait(_SENTINEL)
                except queue.Full:
                    pass
                self._subs.append(q)
            else:
                self._subs.append(q)

        def gen() -> Iterator[Event]:
            try:
                while True:
                    item = q.get()
                    if item is _SENTINEL:
                        return
                    yield item  # type: ignore[misc]
            finally:
                with self._lock:
                    if q in self._subs:
                        self._subs.remove(q)

        return gen()


@dataclass
class Run:
    id: str
    task: str
    status: RunStatus = RunStatus.PENDING
    result: str = ""
    error: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_ceiling_usd: float | None = None
    timeout_seconds: float | None = None
    parent_id: str | None = None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime internals — not serialized to JSON.
    _bus: _EventBus = field(default_factory=_EventBus, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cost_ceiling_usd": self.cost_ceiling_usd,
            "timeout_seconds": self.timeout_seconds,
            "parent_id": self.parent_id,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metadata": self.metadata,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    def cancel(self) -> bool:
        if self.is_terminal:
            return False
        self._cancel.set()
        return True

    def wait(self, timeout: float | None = None) -> "Run":
        self._done.wait(timeout=timeout)
        return self

    def stream(self, *, replay: bool = True) -> Iterator[Event]:
        return self._bus.subscribe(replay=replay)

    def events(self) -> list[Event]:
        return self._bus.history()


# Factory signature: takes (Run, Runtime) and returns an object with
# .chat(task) -> str. In production this is `agi.agent.Agent`; tests pass a fake.
AgentFactory = Callable[["Run", "Runtime"], Any]


class Runtime:
    """Owns the registry of runs and the threads that execute them.

    Process-local. A coordination engine wraps this with its own
    persistence + retry + scheduling. The Runtime intentionally does *not*
    persist run state — it's the executor, not the scheduler.
    """

    def __init__(self, agent_factory: AgentFactory | None = None) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._agent_factory = agent_factory or _default_agent_factory

    def submit(
        self,
        task: str,
        *,
        cost_ceiling_usd: float | None = None,
        timeout_seconds: float | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> Run:
        run = Run(
            id=run_id or uuid.uuid4().hex[:12],
            task=task,
            cost_ceiling_usd=cost_ceiling_usd,
            timeout_seconds=timeout_seconds,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        with self._lock:
            self._runs[run.id] = run

        thread = threading.Thread(
            target=self._execute, args=(run,), name=f"agi-run-{run.id}", daemon=True
        )
        run._thread = thread
        thread.start()
        return run

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[Run]:
        with self._lock:
            return list(self._runs.values())

    def cancel(self, run_id: str) -> bool:
        run = self.get(run_id)
        if run is None:
            return False
        return run.cancel()

    def wait(self, run_id: str, timeout: float | None = None) -> Run | None:
        run = self.get(run_id)
        if run is None:
            return None
        return run.wait(timeout=timeout)

    # ---- internals ----

    def _execute(self, run: Run) -> None:
        run.status = RunStatus.RUNNING
        run.started_at = time.time()
        self._emit(run, "run.started", {"task": run.task})

        timer: threading.Timer | None = None
        if run.timeout_seconds is not None:
            timer = threading.Timer(run.timeout_seconds, run._cancel.set)
            timer.daemon = True
            timer.start()

        try:
            agent = self._agent_factory(run, self)
            text = agent.chat(run.task)
            run.result = text
            # Pull final usage if the agent exposes it.
            usage: Usage | None = getattr(agent, "usage", None)
            model: str = getattr(agent, "model", "")
            if usage is not None:
                run.input_tokens = usage.input_tokens
                run.output_tokens = usage.output_tokens
                run.cache_creation_input_tokens = usage.cache_creation_input_tokens
                run.cache_read_input_tokens = usage.cache_read_input_tokens
                if model:
                    run.cost_usd = usage.cost_usd(model)
            run.status = RunStatus.SUCCEEDED
            self._emit(run, "run.succeeded", {"result_chars": len(text)})
        except Cancelled:
            run.status = (
                RunStatus.TIMED_OUT
                if run.timeout_seconds is not None and (time.time() - (run.started_at or 0)) >= run.timeout_seconds
                else RunStatus.CANCELLED
            )
            self._emit(run, f"run.{run.status.value}", {})
        except BudgetExceeded as e:
            run.status = RunStatus.BUDGET_EXCEEDED
            run.error = str(e)
            self._emit(run, "run.budget_exceeded", {"error": run.error})
        except Exception as e:  # noqa: BLE001 — surface every failure
            run.status = RunStatus.FAILED
            run.error = f"{type(e).__name__}: {e}"
            self._emit(run, "run.failed", {"error": run.error})
        finally:
            if timer is not None:
                timer.cancel()
            run.ended_at = time.time()
            run._bus.close()
            run._done.set()

    def _emit(self, run: Run, event_type: str, payload: dict[str, Any]) -> None:
        run._bus.emit(Event(ts=time.time(), run_id=run.id, type=event_type, payload=payload))


def _default_agent_factory(run: Run, runtime: "Runtime") -> Any:
    """Build an Agent wired to this Run's cancel/cost/event signals.

    Imported lazily so the runtime module is importable without the
    anthropic SDK (handy for tests and for tooling that only needs the
    Run/RunStatus types).

    Enables the skill library by default (durable, low-risk). Tool
    synthesis is *not* enabled by default — pass your own factory to
    `Runtime(agent_factory=...)` if you want it.
    """
    from agi.agent import Agent
    from agi.skills import SkillLibrary, make_skill_tools

    bus = run._bus

    def emit(event_type: str, payload: dict[str, Any] | None = None) -> None:
        bus.emit(Event(ts=time.time(), run_id=run.id, type=event_type, payload=payload or {}))

    skill_schemas, skill_handlers = make_skill_tools(SkillLibrary())

    agent = Agent(
        memory=Memory(),
        verbose=False,
        event_callback=emit,
        cancel_event=run._cancel,
        cost_ceiling_usd=run.cost_ceiling_usd,
        runtime=runtime,
        run_id=run.id,
        extra_tools=skill_schemas,
        extra_handlers=skill_handlers,
    )
    return agent
