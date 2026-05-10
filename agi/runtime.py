"""Runtime layer — Agent as a service primitive.

A `Runtime` wraps the agent for use by an external driver (a coordination
engine, an HTTP server, an IDE plugin, another agent). Three things change
versus calling `Agent.chat` directly:

1. **Stateful sessions.** Each session has a stable id and its own Agent +
   Memory. The driver routes turns by session id; sessions outlive any
   single request.

2. **Structured turn results.** Instead of returning just the final text,
   each turn returns `TurnResult(text, usage, cost_usd, critic_score,
   skills_used, finish_reason, error)`. This is what a coordination engine
   needs to make scheduling and routing decisions.

3. **Capability discovery.** `describe()` returns the model id, available
   tools, available skills, and the current pricing — so the coordinator
   can pick which runtime to dispatch a task to.

The runtime is deliberately minimal. It does not own task queuing, retries,
or multi-tenant isolation; those belong to the driver. It exposes the smallest
contract a driver needs to drive one agent.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agi.agent import Agent
from agi.costs import PRICING, Usage
from agi.memory import Memory


@dataclass
class TurnResult:
    text: str
    usage: dict[str, int]
    cost_usd: float
    critic_score: float | None
    skills_used: list[str]
    finish_reason: str  # "ok" | "error"
    elapsed_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionInfo:
    id: str
    created_at: float
    last_used_at: float
    turn_count: int
    cumulative_usage: dict[str, int]
    cumulative_cost_usd: float
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Session:
    id: str
    agent: Agent
    created_at: float
    last_used_at: float
    turn_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class Runtime:
    """Stateful, multi-session wrapper over `Agent`.

    Thread-safe: each session has its own lock, so concurrent turns on
    different sessions run in parallel; concurrent turns on the same
    session serialize. The session map itself is guarded by a single lock
    on creation/lookup/deletion.
    """

    def __init__(
        self,
        *,
        agent_factory=None,
        memory_root: str | Path | None = None,
    ) -> None:
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        self._memory_root = Path(memory_root) if memory_root else None
        self._agent_factory = agent_factory or self._default_agent_factory

    def _default_agent_factory(self, session_id: str) -> Agent:
        memory_path: Path | None = None
        if self._memory_root is not None:
            self._memory_root.mkdir(parents=True, exist_ok=True)
            memory_path = self._memory_root / f"{session_id}.jsonl"
        memory = Memory(path=memory_path) if memory_path else Memory()
        return Agent(memory=memory, verbose=False, enable_delegate=True)

    def create_session(self, session_id: str | None = None) -> str:
        sid = session_id or uuid.uuid4().hex[:16]
        with self._sessions_lock:
            if sid in self._sessions:
                raise ValueError(f"session {sid!r} already exists")
            agent = self._agent_factory(sid)
            now = time.time()
            self._sessions[sid] = _Session(
                id=sid, agent=agent, created_at=now, last_used_at=now
            )
        return sid

    def get_session(self, sid: str) -> _Session:
        with self._sessions_lock:
            session = self._sessions.get(sid)
        if session is None:
            raise KeyError(f"unknown session: {sid!r}")
        return session

    def list_sessions(self) -> list[SessionInfo]:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        return [self._session_info(s) for s in sessions]

    def session_info(self, sid: str) -> SessionInfo:
        return self._session_info(self.get_session(sid))

    def _session_info(self, session: _Session) -> SessionInfo:
        a = session.agent
        return SessionInfo(
            id=session.id,
            created_at=session.created_at,
            last_used_at=session.last_used_at,
            turn_count=session.turn_count,
            cumulative_usage={
                "input_tokens": a.usage.input_tokens,
                "output_tokens": a.usage.output_tokens,
                "cache_creation_input_tokens": a.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": a.usage.cache_read_input_tokens,
                "turns": a.usage.turns,
            },
            cumulative_cost_usd=a.usage.cost_usd(a.model),
            model=a.model,
        )

    def delete_session(self, sid: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(sid, None)

    def reset_session(self, sid: str) -> None:
        session = self.get_session(sid)
        with session.lock:
            session.agent.reset()
            session.turn_count = 0

    def turn(self, sid: str, user_input: str, max_iterations: int = 25) -> TurnResult:
        """Run one turn synchronously. Blocks until the agent emits its
        final text or an error is raised."""
        session = self.get_session(sid)
        with session.lock:
            agent = session.agent
            before = Usage(
                input_tokens=agent.usage.input_tokens,
                output_tokens=agent.usage.output_tokens,
                cache_creation_input_tokens=agent.usage.cache_creation_input_tokens,
                cache_read_input_tokens=agent.usage.cache_read_input_tokens,
                turns=agent.usage.turns,
            )
            t0 = time.time()
            try:
                text = agent.chat(user_input, max_iterations=max_iterations)
                error: str | None = None
                finish_reason = "ok"
            except Exception as e:
                text = ""
                error = f"{type(e).__name__}: {e}"
                finish_reason = "error"
            elapsed = time.time() - t0
            after = agent.usage
            session.turn_count += 1
            session.last_used_at = time.time()

        turn_usage = Usage(
            input_tokens=after.input_tokens - before.input_tokens,
            output_tokens=after.output_tokens - before.output_tokens,
            cache_creation_input_tokens=after.cache_creation_input_tokens
            - before.cache_creation_input_tokens,
            cache_read_input_tokens=after.cache_read_input_tokens
            - before.cache_read_input_tokens,
            turns=after.turns - before.turns,
        )
        return TurnResult(
            text=text,
            usage={
                "input_tokens": turn_usage.input_tokens,
                "output_tokens": turn_usage.output_tokens,
                "cache_creation_input_tokens": turn_usage.cache_creation_input_tokens,
                "cache_read_input_tokens": turn_usage.cache_read_input_tokens,
                "turns": turn_usage.turns,
            },
            cost_usd=turn_usage.cost_usd(agent.model),
            critic_score=agent.last_critic_score,
            skills_used=list(agent.last_skills_used),
            finish_reason=finish_reason,
            elapsed_seconds=elapsed,
            error=error,
        )

    def describe(self) -> dict[str, Any]:
        """Capability description for a coordination engine to consume.

        Lists model, tools, skills, and pricing so the coordinator can
        route tasks across multiple runtimes (or runtime versions).
        """
        # Build a probe agent on the fly to inspect available tools without
        # requiring a session. Doesn't make any API calls.
        probe = self._agent_factory("__describe__")
        try:
            tools = [
                {"name": t.get("name"), "type": t.get("type", "client")}
                for t in probe.tool_schemas
            ]
            skills_info: list[dict] = []
            if probe.skills is not None:
                try:
                    skills_info = [
                        {"name": s.name, "description": s.description, "tags": s.tags}
                        for s in probe.skills.all()
                    ]
                except Exception:
                    skills_info = []
            return {
                "model": probe.model,
                "max_tokens": probe.max_tokens,
                "effort": probe.effort,
                "tools": tools,
                "skills": skills_info,
                "pricing": {m: {"input_per_mtok": i, "output_per_mtok": o}
                            for m, (i, o) in PRICING.items()},
                "active_sessions": len(self._sessions),
            }
        finally:
            # The probe doesn't get registered as a session.
            pass
