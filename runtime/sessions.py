"""Session lifecycle.

A `Session` is the durable unit of work for the runtime. It owns:
  - an Agent (frozen Opus, or a mock for testing — pluggable)
  - an isolated Memory (per-session JSONL file under a runtime root dir)
  - an optional TraceLogger
  - a Budget that gates further work
  - usage and job-count accounting

Sessions are created, listed, fetched, and deleted by the SessionManager.
The manager is thread-safe; per-session execution is serialised by an
internal lock so concurrent jobs against the same session can't interleave
turns on the same conversation history.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from agi.costs import Usage
from agi.memory import Memory
from runtime.budgets import Budget, BudgetError

try:
    from learner.traces import TraceLogger
except ImportError:
    TraceLogger = None  # type: ignore


class AgentLike(Protocol):
    """Minimal shape the runtime needs from an agent backend."""

    model: str
    memory: Memory
    messages: list[dict]
    usage: Usage

    def chat(self, user_input: str, max_iterations: int = 25) -> str: ...
    def reset(self) -> None: ...


@dataclass
class SessionInfo:
    id: str
    created_at: float
    model: str
    budget: dict
    usage: dict
    jobs_run: int
    turns: int
    closed: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "model": self.model,
            "budget": self.budget,
            "usage": self.usage,
            "jobs_run": self.jobs_run,
            "turns": self.turns,
            "closed": self.closed,
        }


class Session:
    def __init__(
        self,
        agent: AgentLike,
        *,
        budget: Budget | None = None,
        session_id: str | None = None,
    ) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self.created_at = time.time()
        self.agent = agent
        self.budget = budget or Budget()
        self.jobs_run = 0
        self.closed = False
        self._lock = threading.Lock()  # serialises chat() on this session

    def info(self) -> SessionInfo:
        u = self.agent.usage
        return SessionInfo(
            id=self.id,
            created_at=self.created_at,
            model=self.agent.model,
            budget=self.budget.to_dict(),
            usage={
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_creation_input_tokens": u.cache_creation_input_tokens,
                "cache_read_input_tokens": u.cache_read_input_tokens,
                "cost_usd": u.cost_usd(self.agent.model),
            },
            jobs_run=self.jobs_run,
            turns=u.turns,
            closed=self.closed,
        )

    def run_turn(self, prompt: str, *, max_iterations: int = 25) -> str:
        if self.closed:
            raise BudgetError("session is closed")
        # Pre-flight budget check. The actual SDK call might still push usage
        # over by one turn's worth — acceptable slop; gate the *next* call.
        self.budget.check(self.agent.usage, self.agent.model, self.jobs_run)
        with self._lock:
            text = self.agent.chat(prompt, max_iterations=max_iterations)
            self.jobs_run += 1
            return text

    def close(self) -> None:
        self.closed = True


class SessionManager:
    """Thread-safe registry of live sessions.

    The agent factory is injected so tests, demos, and prod can use different
    backends without touching the manager. Each session gets a fresh Memory
    file under `root/sessions/{sid}/memory.jsonl` so per-session state stays
    isolated.
    """

    def __init__(
        self,
        agent_factory: Callable[..., AgentLike],
        *,
        root: Path | str | None = None,
        enable_traces: bool = True,
    ) -> None:
        self.agent_factory = agent_factory
        self.root = Path(root) if root else Path.home() / ".agi" / "runtime"
        self.root.mkdir(parents=True, exist_ok=True)
        self.enable_traces = enable_traces
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def create(
        self,
        *,
        model: str | None = None,
        budget: Budget | None = None,
        agent_kwargs: dict[str, Any] | None = None,
    ) -> Session:
        sid = uuid.uuid4().hex[:12]
        sdir = self.root / "sessions" / sid
        sdir.mkdir(parents=True, exist_ok=True)
        memory = Memory(path=sdir / "memory.jsonl")
        tracer = TraceLogger(path=sdir / "traces.jsonl") if (self.enable_traces and TraceLogger is not None) else None

        kwargs: dict[str, Any] = dict(agent_kwargs or {})
        kwargs.setdefault("memory", memory)
        if tracer is not None:
            kwargs.setdefault("tracer", tracer)
        if model is not None:
            kwargs.setdefault("model", model)
        agent = self.agent_factory(**kwargs)

        session = Session(agent, budget=budget, session_id=sid)
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, sid: str) -> Session | None:
        with self._lock:
            return self._sessions.get(sid)

    def list(self) -> list[SessionInfo]:
        with self._lock:
            return [s.info() for s in self._sessions.values()]

    def delete(self, sid: str) -> bool:
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is None:
            return False
        session.close()
        return True
