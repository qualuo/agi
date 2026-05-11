"""Runtime: addressable, streaming, multi-session agent engine.

A coordination engine (whether external, like an orchestration platform, or
internal, like `agi.coord.Coordinator`) needs to drive many concurrent agent
sessions, observe each one's structured event stream, meter cost, enforce
budgets, and tear them down. The CLI in `agi.__main__` is a single-user shell
sitting on top of `Agent.chat` — fine for humans, useless for orchestration.

The Runtime turns the agent into a service:

  runtime = Runtime()
  sid = runtime.open()
  runtime.send(sid, "summarize ./README.md")
  for event in runtime.stream(sid):
      ...   # TextDelta, ToolUseStart, UsageDelta, TurnEnd, ...
  runtime.close(sid)

Each session has a stable id, its own Agent + Memory, its own usage counters,
and a bounded budget the runtime enforces between turns. Sessions execute on a
worker thread per session — Anthropic streaming is synchronous, and the
threading model is the simplest correct way to multiplex without dragging in
asyncio. The event sink writes typed events to a thread-safe queue; consumers
read with `stream()`.

The runtime owns no LLM calls of its own — it's pure plumbing. That keeps the
contract clean: a coordination engine can replace the Agent class with a mock,
a different model, or a different policy, and the runtime stays.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agi.agent import Agent
from agi.events import Event, ErrorEvent, TurnEnd
from agi.memory import Memory


# Sentinel placed on a session's event queue to signal the consumer that no
# more events will arrive (the session was closed). Using a singleton lets
# `stream()` distinguish "queue empty, wait" from "stream ended cleanly".
_STREAM_END = object()


@dataclass
class SessionStatus:
    """Snapshot of a session — what a coordination engine reads to make
    scheduling decisions (is it free? did it crash? has it blown its budget?)."""
    id: str
    state: str  # "idle" | "running" | "closed" | "errored"
    created_at: float
    last_activity: float
    turns: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    budget_usd: float | None
    parent_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """One running agent. The runtime hands these out by id; callers should
    not poke at internals directly."""
    id: str
    agent: Agent
    parent_id: str | None
    budget_usd: float | None
    metadata: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    state: str = "idle"
    turns: int = 0
    # Thread-safe inbox for user messages waiting to be processed.
    _inbox: "queue.Queue[str | object]" = field(default_factory=queue.Queue)
    # Thread-safe outbox of events for consumers.
    _outbox: "queue.Queue[Event | object]" = field(default_factory=queue.Queue)
    _seq: int = 0
    _seq_lock: threading.Lock = field(default_factory=threading.Lock)
    _worker: threading.Thread | None = None
    _last_error: str | None = None

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def status(self) -> SessionStatus:
        return SessionStatus(
            id=self.id,
            state=self.state,
            created_at=self.created_at,
            last_activity=self.last_activity,
            turns=self.turns,
            cost_usd=self.agent.usage.cost_usd(self.agent.model),
            input_tokens=self.agent.usage.input_tokens,
            output_tokens=self.agent.usage.output_tokens,
            budget_usd=self.budget_usd,
            parent_id=self.parent_id,
            metadata=dict(self.metadata),
        )


class BudgetExceededError(RuntimeError):
    """Raised when a session's accumulated spend exceeds its budget."""


class Runtime:
    """Owns the table of live sessions and the worker threads driving them.

    Thread-safe: `open`/`send`/`close`/`stream`/`status` can be called from
    any thread. The internal session table is guarded by a single lock; per-
    session state is owned by its worker thread.
    """

    def __init__(
        self,
        agent_factory: Callable[..., Agent] | None = None,
        default_budget_usd: float | None = None,
        memory_root: Path | None = None,
    ) -> None:
        # Factory pattern lets tests inject fake Agents (e.g. with a stub
        # client) without subclassing the runtime.
        self._agent_factory = agent_factory or _default_agent_factory
        self._default_budget_usd = default_budget_usd
        self._memory_root = memory_root
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def open(
        self,
        *,
        parent_id: str | None = None,
        budget_usd: float | None = None,
        memory: Memory | None = None,
        system_prompt_extra: str = "",
        metadata: dict[str, Any] | None = None,
        agent_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Spin up a new session. Returns its id. The session is idle until
        the first `send`."""
        sid = uuid.uuid4().hex[:12]

        session = Session(
            id=sid,
            agent=None,  # type: ignore[arg-type]  # filled in below
            parent_id=parent_id,
            budget_usd=budget_usd if budget_usd is not None else self._default_budget_usd,
            metadata=dict(metadata or {}),
        )

        def sink(event: Event) -> None:
            event.seq = session.next_seq()
            session._outbox.put(event)

        if memory is None:
            mem_path = None
            if self._memory_root is not None:
                mem_path = self._memory_root / f"{sid}.jsonl"
            memory = Memory(path=mem_path)

        agent = self._agent_factory(
            memory=memory,
            verbose=False,
            event_sink=sink,
            session_id=sid,
            extra_system_prompt=system_prompt_extra,
            **(agent_kwargs or {}),
        )
        session.agent = agent

        worker = threading.Thread(
            target=self._run_session, args=(session,), name=f"agi-session-{sid}", daemon=True
        )
        session._worker = worker
        with self._lock:
            self._sessions[sid] = session
        worker.start()
        return sid

    def send(self, session_id: str, user_input: str) -> None:
        """Enqueue a user message for the session. Returns immediately; events
        come back via `stream()`."""
        session = self._get(session_id)
        if session.state == "closed":
            raise RuntimeError(f"session {session_id} is closed")
        session._inbox.put(user_input)

    def close(self, session_id: str) -> None:
        """Tell the worker to stop after the current turn (if any) and close
        the event stream."""
        session = self._get(session_id)
        if session.state == "closed":
            return
        session._inbox.put(_STREAM_END)

    def status(self, session_id: str) -> SessionStatus:
        return self._get(session_id).status()

    def list_sessions(self) -> list[SessionStatus]:
        with self._lock:
            return [s.status() for s in self._sessions.values()]

    def stream(
        self,
        session_id: str,
        *,
        timeout: float | None = None,
    ) -> Iterator[Event]:
        """Yield events for a session until the stream ends (close()).

        If `timeout` is set, each `get` will wait at most that long; on
        timeout the iterator returns instead of blocking forever. Useful for
        clients that need to disconnect cleanly.
        """
        session = self._get(session_id)
        while True:
            try:
                item = session._outbox.get(timeout=timeout)
            except queue.Empty:
                return
            if item is _STREAM_END:
                return
            yield item  # type: ignore[misc]

    def wait_for_turn_end(
        self, session_id: str, *, timeout: float | None = None
    ) -> TurnEnd | None:
        """Block until the session emits a TurnEnd, return it (or None on
        timeout / stream end). Drains other events into the void — most
        callers want either `stream` OR `wait_for_turn_end`, not both."""
        session = self._get(session_id)
        end_time = None if timeout is None else time.time() + timeout
        while True:
            remaining = None if end_time is None else max(0.0, end_time - time.time())
            try:
                item = session._outbox.get(timeout=remaining)
            except queue.Empty:
                return None
            if item is _STREAM_END:
                return None
            if isinstance(item, TurnEnd):
                return item

    # -------- internal --------

    def _get(self, session_id: str) -> Session:
        with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"unknown session: {session_id}")
        return sess

    def _run_session(self, session: Session) -> None:
        """Worker thread loop: dequeue user messages, run a chat turn for each,
        until close()."""
        while True:
            item = session._inbox.get()
            if item is _STREAM_END:
                break
            user_input = item  # type: ignore[assignment]
            assert isinstance(user_input, str)
            session.state = "running"
            session.last_activity = time.time()

            # Pre-turn budget check. If the session has already spent its
            # budget, refuse to run another turn. The coordination engine can
            # raise the budget if it wants to keep going.
            spend = session.agent.usage.cost_usd(session.agent.model)
            if session.budget_usd is not None and spend >= session.budget_usd:
                err = BudgetExceededError(
                    f"session {session.id}: budget ${session.budget_usd:.4f} "
                    f"already exhausted (${spend:.4f})"
                )
                self._emit_session_error(session, err)
                session.state = "errored"
                session._last_error = str(err)
                continue

            try:
                session.agent.chat(user_input)
                session.turns += 1
                session.state = "idle"
            except Exception as e:
                # ErrorEvent has already been emitted from inside Agent; flag
                # the session as errored but keep the worker alive so the
                # coordinator can inspect status and decide what to do.
                session.state = "errored"
                session._last_error = f"{type(e).__name__}: {e}"
            session.last_activity = time.time()

        session.state = "closed"
        session._outbox.put(_STREAM_END)

    def _emit_session_error(self, session: Session, exc: Exception) -> None:
        ev = ErrorEvent(
            session_id=session.id,
            message=str(exc),
            exc_type=type(exc).__name__,
        )
        ev.seq = session.next_seq()
        session._outbox.put(ev)


def _default_agent_factory(**kwargs) -> Agent:
    """Factory that lets the runtime construct an Agent with the standard
    constructor — but is also the override point for tests / alternate
    deployments."""
    return Agent(**kwargs)
