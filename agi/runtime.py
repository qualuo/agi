"""Agent runtime engine.

A `Runtime` is the substrate a coordination engine drives. It is **not** an
agent itself — it owns Sessions, executes them concurrently on a thread
pool, enforces per-session budgets, broadcasts structured events to
subscribers, and exposes a small set of imperative operations:

    submit(goal, ...) -> session_id
    status(session_id) -> dict
    events(session_id) -> iterator[Event]      (replay + live tail)
    wait(session_id, timeout=...) -> dict
    cancel(session_id) -> None
    list_sessions() -> list[dict]

The Runtime intentionally has no opinions about *what* to coordinate or
*how* to decompose work. That belongs to a Plan (agi.plan) or an external
coordinator. Decoupling those concerns is what makes this composable.

Design notes:
- One thread per active session. Anthropic's client is sync, so this maps
  cleanly. Concurrency caps avoid runaway fan-out.
- Events are durable per session (in-memory deque, optional disk log via
  the existing TraceLogger) AND fanned out to live subscriber queues.
- Cancellation is cooperative: checked between turns. In-flight HTTP
  requests are not aborted (Anthropic's stream client doesn't expose a
  clean cancel) — they finish, then the loop exits.
- The Agent factory is injected so tests can swap in a fake.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from agi.agent import Agent
from agi.budget import Budget
from agi.events import (
    Event,
    SessionFinished,
    SessionStarted,
)
from agi.memory import Memory


# Sentinel that subscribers receive when a session ends; lets live tails exit.
_END = object()


AgentFactory = Callable[..., Agent]


def _default_agent_factory(**kwargs) -> Agent:
    return Agent(**kwargs)


@dataclass
class SessionRecord:
    session_id: str
    goal: str
    status: str = "pending"  # pending | running | ok | budget_exceeded | error | cancelled
    final_text: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turns: int = 0
    model: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "status": self.status,
            "final_text": self.final_text,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "turns": self.turns,
            "model": self.model,
            "tags": self.tags,
            "parent_id": self.parent_id,
        }


class _Session:
    """Internal state for one running session."""

    def __init__(
        self,
        record: SessionRecord,
        agent_kwargs: dict[str, Any],
        budget: Optional[Budget],
        max_iterations: int,
        agent_factory: AgentFactory,
    ) -> None:
        self.record = record
        self.agent_kwargs = agent_kwargs
        self.budget = budget
        self.max_iterations = max_iterations
        self.agent_factory = agent_factory

        self.events: list[Event] = []
        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.cancel_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.agent: Optional[Agent] = None

    def add_event(self, event: Event) -> None:
        with self.lock:
            self.events.append(event)
            for sub in self.subscribers:
                try:
                    sub.put_nowait(event)
                except queue.Full:
                    pass  # subscribers that fall behind drop events

    def subscribe(self, replay: bool = True) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=10000)
        with self.lock:
            if replay:
                for e in self.events:
                    q.put_nowait(e)
            if self.done.is_set():
                q.put_nowait(_END)
            else:
                self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def close(self) -> None:
        with self.lock:
            for sub in self.subscribers:
                try:
                    sub.put_nowait(_END)
                except queue.Full:
                    pass
            self.subscribers.clear()
        self.done.set()


class Runtime:
    """Manages concurrent agent sessions and exposes them to a coordinator."""

    def __init__(
        self,
        max_concurrent: int = 4,
        default_budget: Optional[Budget] = None,
        memory: Optional[Memory] = None,
        agent_factory: AgentFactory = _default_agent_factory,
        agent_defaults: Optional[dict[str, Any]] = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.default_budget = default_budget
        self.shared_memory = memory  # may be None: each session gets its own if so
        self.agent_factory = agent_factory
        self.agent_defaults: dict[str, Any] = {"verbose": False, **(agent_defaults or {})}

        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self._slot = threading.Semaphore(max_concurrent)

    # ----- public surface ----------------------------------------------------

    def submit(
        self,
        goal: str,
        *,
        budget: Optional[Budget] = None,
        memory: Optional[Memory] = None,
        agent_kwargs: Optional[dict[str, Any]] = None,
        max_iterations: int = 25,
        tags: Optional[dict[str, str]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Submit a goal. Returns a session_id immediately; runs in background."""
        sid = uuid.uuid4().hex[:12]
        kwargs = {
            **self.agent_defaults,
            "memory": memory if memory is not None else self.shared_memory,
            **(agent_kwargs or {}),
        }
        eff_budget = budget if budget is not None else self.default_budget

        record = SessionRecord(
            session_id=sid,
            goal=goal,
            model=str(kwargs.get("model", "claude-opus-4-7")),
            tags=tags or {},
            parent_id=parent_id,
        )
        sess = _Session(
            record=record,
            agent_kwargs=kwargs,
            budget=eff_budget,
            max_iterations=max_iterations,
            agent_factory=self.agent_factory,
        )
        with self._lock:
            self._sessions[sid] = sess

        sess.thread = threading.Thread(
            target=self._run, args=(sess,), name=f"agi-session-{sid}", daemon=True
        )
        sess.thread.start()
        return sid

    def status(self, session_id: str) -> dict[str, Any]:
        sess = self._require(session_id)
        return sess.record.to_dict()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.record.to_dict() for s in self._sessions.values()]

    def cancel(self, session_id: str) -> None:
        sess = self._require(session_id)
        sess.cancel_flag.set()

    def wait(self, session_id: str, timeout: Optional[float] = None) -> dict[str, Any]:
        sess = self._require(session_id)
        sess.done.wait(timeout=timeout)
        return sess.record.to_dict()

    def events(
        self,
        session_id: str,
        *,
        replay: bool = True,
        follow: bool = True,
        timeout: Optional[float] = None,
    ) -> Iterator[Event]:
        """Yield events for a session.

        - replay=True: emits historical events first.
        - follow=True: continues yielding new events until the session ends.
        - timeout: per-event timeout when following live; raises StopIteration
          when no new event arrives in that window.
        """
        sess = self._require(session_id)
        if not follow:
            with sess.lock:
                yield from list(sess.events)
            return

        q = sess.subscribe(replay=replay)
        try:
            while True:
                try:
                    item = q.get(timeout=timeout) if timeout else q.get()
                except queue.Empty:
                    return
                if item is _END:
                    return
                yield item  # type: ignore[misc]
        finally:
            sess.unsubscribe(q)

    def gc(self, keep_recent: int = 1000) -> int:
        """Drop completed session records beyond `keep_recent`. Returns count dropped."""
        with self._lock:
            done = [
                (sid, s) for sid, s in self._sessions.items() if s.done.is_set()
            ]
            done.sort(key=lambda kv: kv[1].record.finished_at)
            drop = done[:-keep_recent] if keep_recent > 0 else done
            for sid, _ in drop:
                self._sessions.pop(sid, None)
            return len(drop)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel all sessions and wait briefly for threads to exit."""
        with self._lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            s.cancel_flag.set()
        deadline = time.time() + timeout
        for s in sessions:
            remaining = max(0.0, deadline - time.time())
            s.done.wait(timeout=remaining)

    # ----- internals ---------------------------------------------------------

    def _require(self, session_id: str) -> _Session:
        with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"unknown session: {session_id}")
        return sess

    def _run(self, sess: _Session) -> None:
        # Wait for a concurrency slot.
        self._slot.acquire()
        try:
            sess.record.status = "running"
            sess.record.started_at = time.time()
            sess.add_event(
                SessionStarted(
                    session_id=sess.record.session_id,
                    goal=sess.record.goal,
                    model=sess.record.model,
                )
            )
            agent = sess.agent_factory(
                **sess.agent_kwargs,
                on_event=sess.add_event,
                budget=sess.budget,
                cancel_check=sess.cancel_flag.is_set,
            )
            sess.agent = agent
            try:
                final = agent.chat(sess.record.goal, max_iterations=sess.max_iterations)
            except Exception as e:
                sess.record.status = "error"
                sess.record.error = f"{type(e).__name__}: {e}"
                final = ""
            else:
                if sess.cancel_flag.is_set():
                    sess.record.status = "cancelled"
                elif _hit_budget(sess.events):
                    sess.record.status = "budget_exceeded"
                else:
                    sess.record.status = "ok"

            usage = agent.usage if agent is not None else None
            if usage is not None:
                sess.record.total_cost_usd = usage.cost_usd(sess.record.model)
                sess.record.total_input_tokens = usage.input_tokens
                sess.record.total_output_tokens = usage.output_tokens
                sess.record.turns = usage.turns
            sess.record.final_text = final
            sess.record.finished_at = time.time()

            sess.add_event(
                SessionFinished(
                    session_id=sess.record.session_id,
                    status=sess.record.status,
                    final_text=final,
                    total_cost_usd=sess.record.total_cost_usd,
                    total_input_tokens=sess.record.total_input_tokens,
                    total_output_tokens=sess.record.total_output_tokens,
                    turns=sess.record.turns,
                    error=sess.record.error,
                )
            )
        finally:
            sess.close()
            self._slot.release()


def _hit_budget(events: list[Event]) -> bool:
    for e in events:
        if e.kind == "budget_exceeded":
            return True
    return False
