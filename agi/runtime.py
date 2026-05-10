"""Runtime engine.

What this module is:
- A clean, embeddable Python API for running agents that a *coordination
  engine* (an external orchestrator) can drive without going through the CLI.
- Stateful sessions that can be paused, snapshotted, resumed, migrated.
- Hard budgets enforced in the runtime, not just advisory.
- Lifecycle events emitted on a bus the coordinator can subscribe to.
- Capability declaration so a coordinator knows what each runtime offers.

What this module is not:
- Magic. The Agent class still does the actual reasoning via the Claude API.
  This is the contract surface around it.

Typical use from a coordination engine:

    from agi.runtime import Runtime, Budget

    rt = Runtime()
    session = rt.create_session(budget=Budget(max_usd=0.50))
    session.bus.subscribe(my_handler)        # stream events
    result = session.step("plan a trip to Tokyo")
    snap = session.snapshot()                # park the session
    later = rt.restore_session(snap)         # resume later, possibly elsewhere
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agi.agent import Agent
from agi.costs import Usage
from agi.events import (
    EventBus,
    SESSION_CLOSED,
    SESSION_CREATED,
    TURN_ERRORED,
    new_session_id,
)
from agi.memory import Memory
from agi.metrics import RuntimeMetrics


@dataclass
class Budget:
    """Hard runtime ceilings. None means no limit on that axis.

    Checked before each model call inside a turn; on violation the agent
    aborts mid-turn and the partial output (if any) is returned with a
    `[runtime: ...]` annotation."""

    max_usd: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_turns: int | None = None  # max .step() calls in this session

    def violation(self, usage: Usage, model: str) -> str | None:
        if self.max_usd is not None and usage.cost_usd(model) >= self.max_usd:
            return f"max_usd ({self.max_usd}) reached"
        if self.max_input_tokens is not None and usage.input_tokens >= self.max_input_tokens:
            return f"max_input_tokens ({self.max_input_tokens}) reached"
        if self.max_output_tokens is not None and usage.output_tokens >= self.max_output_tokens:
            return f"max_output_tokens ({self.max_output_tokens}) reached"
        if self.max_turns is not None and usage.turns >= self.max_turns:
            return f"max_turns ({self.max_turns}) reached"
        return None

    def to_dict(self) -> dict:
        return {
            "max_usd": self.max_usd,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_turns": self.max_turns,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Budget":
        if not d:
            return cls()
        return cls(
            max_usd=d.get("max_usd"),
            max_input_tokens=d.get("max_input_tokens"),
            max_output_tokens=d.get("max_output_tokens"),
            max_turns=d.get("max_turns"),
        )


@dataclass
class StepResult:
    text: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    critic_score: float | None
    budget_exceeded: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "critic_score": self.critic_score,
            "budget_exceeded": self.budget_exceeded,
            "error": self.error,
        }


@dataclass
class SessionInfo:
    id: str
    created_at: float
    model: str
    turns: int
    cost_usd: float
    budget: dict
    role: str | None
    status: str  # ready | running | exhausted | closed | errored

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "model": self.model,
            "turns": self.turns,
            "cost_usd": self.cost_usd,
            "budget": self.budget,
            "role": self.role,
            "status": self.status,
        }


class Session:
    """A single conversational thread bound to one Agent instance.

    Thread-safety: `step()` is serialized per-session via an internal lock so
    a coordination engine can have multiple producers calling step()
    concurrently without corrupting the conversation; calls are queued.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        budget: Budget | None = None,
        id: str | None = None,
        role: str | None = None,
    ) -> None:
        self.id = id or new_session_id()
        self.agent = agent
        self.budget = budget or Budget()
        self.role = role
        self.created_at = time.time()
        self.status = "ready"
        self.bus = agent.bus or EventBus()
        self.bus.session_id = self.id
        agent.bus = self.bus
        agent.budget = self.budget
        self._lock = threading.Lock()
        self.bus.emit(SESSION_CREATED, {"role": role, "model": agent.model, "budget": self.budget.to_dict()})

    def step(self, user_input: str) -> StepResult:
        before = Usage(
            input_tokens=self.agent.usage.input_tokens,
            output_tokens=self.agent.usage.output_tokens,
            cache_creation_input_tokens=self.agent.usage.cache_creation_input_tokens,
            cache_read_input_tokens=self.agent.usage.cache_read_input_tokens,
            turns=self.agent.usage.turns,
        )
        # Pre-flight budget check: don't even start a turn we can't afford to run.
        pre = self.budget.violation(self.agent.usage, self.agent.model)
        if pre is not None:
            self.status = "exhausted"
            return StepResult(
                text=f"[runtime: {pre} — turn rejected]",
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                critic_score=None,
                budget_exceeded=True,
                error=pre,
            )

        with self._lock:
            self.status = "running"
            try:
                text = self.agent.chat(user_input)
                error: str | None = None
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                text = ""
                self.bus.emit(TURN_ERRORED, {"error": error})
                self.status = "errored"

        delta = Usage(
            input_tokens=self.agent.usage.input_tokens - before.input_tokens,
            output_tokens=self.agent.usage.output_tokens - before.output_tokens,
            cache_creation_input_tokens=self.agent.usage.cache_creation_input_tokens - before.cache_creation_input_tokens,
            cache_read_input_tokens=self.agent.usage.cache_read_input_tokens - before.cache_read_input_tokens,
            turns=self.agent.usage.turns - before.turns,
        )

        post_violation = self.budget.violation(self.agent.usage, self.agent.model)
        budget_exceeded = post_violation is not None
        if budget_exceeded:
            self.status = "exhausted"
        elif self.status != "errored":
            self.status = "ready"

        return StepResult(
            text=text,
            cost_usd=delta.cost_usd(self.agent.model),
            input_tokens=delta.input_tokens,
            output_tokens=delta.output_tokens,
            cache_read_input_tokens=delta.cache_read_input_tokens,
            cache_creation_input_tokens=delta.cache_creation_input_tokens,
            critic_score=self.agent.last_critic_score,
            budget_exceeded=budget_exceeded,
            error=error,
        )

    def info(self) -> SessionInfo:
        return SessionInfo(
            id=self.id,
            created_at=self.created_at,
            model=self.agent.model,
            turns=self.agent.usage.turns,
            cost_usd=self.agent.usage.cost_usd(self.agent.model),
            budget=self.budget.to_dict(),
            role=self.role,
            status=self.status,
        )

    def snapshot(self) -> dict:
        """Serialize state sufficient to resume in another process.

        Does not snapshot the EventBus subscribers (those are local to the
        process); it does include the buffered event tail so a resumed
        session can replay recent history."""
        return {
            "version": 1,
            "id": self.id,
            "created_at": self.created_at,
            "role": self.role,
            "status": self.status,
            "budget": self.budget.to_dict(),
            "agent": self.agent.snapshot(),
            "events_tail": [e.to_dict() for e in self.bus.replay(0)][-100:],
        }

    def close(self) -> None:
        self.status = "closed"
        self.bus.emit(SESSION_CLOSED, {})


@dataclass
class Capabilities:
    """What this runtime exposes. A coordinator queries this to decide what
    to route here."""

    runtime_version: str = "1"
    default_model: str = "claude-opus-4-7"
    available_models: list[str] = field(default_factory=lambda: [
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ])
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    server_tools: list[str] = field(default_factory=lambda: ["web_search", "web_fetch"])
    snapshots: bool = True
    streaming_events: bool = True

    def to_dict(self) -> dict:
        return {
            "runtime_version": self.runtime_version,
            "default_model": self.default_model,
            "available_models": self.available_models,
            "tools": self.tools,
            "skills": self.skills,
            "server_tools": self.server_tools,
            "snapshots": self.snapshots,
            "streaming_events": self.streaming_events,
        }


AgentFactory = Callable[..., Agent]


class Runtime:
    """In-process registry of sessions. The contract surface a coordination
    engine talks to."""

    def __init__(
        self,
        *,
        agent_factory: AgentFactory | None = None,
        memory: Memory | None = None,
        skill_library=None,
    ) -> None:
        self._agent_factory = agent_factory or Agent
        self._memory = memory
        self._skill_library = skill_library
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.metrics = RuntimeMetrics()

    def capabilities(self) -> Capabilities:
        # Probe a short-lived agent for tool list (no API calls; tools are local).
        probe_memory = self._memory or Memory()
        probe = Agent.__new__(Agent)
        probe.memory = probe_memory
        # use make_tools directly to avoid building a full agent
        from agi.tools import make_tools as _mt
        schemas, handlers = _mt(probe_memory)
        tool_names = [s["name"] for s in schemas]
        skill_names: list[str] = []
        if self._skill_library is not None:
            try:
                skill_names = [s.name for s in self._skill_library.all()]
            except Exception:
                skill_names = []
        return Capabilities(tools=tool_names, skills=skill_names)

    def create_session(
        self,
        *,
        budget: Budget | None = None,
        role: str | None = None,
        agent_kwargs: dict | None = None,
        id: str | None = None,
    ) -> Session:
        kwargs: dict[str, Any] = {"verbose": False}
        if self._memory is not None:
            kwargs["memory"] = self._memory
        if agent_kwargs:
            kwargs.update(agent_kwargs)
        if self._skill_library is not None and "extra_system" not in kwargs:
            extra = self._skill_library.system_prompt_addendum()
            if extra:
                kwargs["extra_system"] = extra

        agent = self._agent_factory(**kwargs)
        # Wire metrics to the bus before Session constructs (so the
        # SESSION_CREATED event it emits is observed).
        bus = EventBus()
        self.metrics.attach(bus, role=role)
        agent.bus = bus
        session = Session(agent, budget=budget, id=id, role=role)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def restore_session(
        self,
        snapshot: dict,
        *,
        agent_kwargs: dict | None = None,
    ) -> Session:
        kwargs: dict[str, Any] = {"verbose": False}
        if self._memory is not None:
            kwargs["memory"] = self._memory
        if agent_kwargs:
            kwargs.update(agent_kwargs)
        agent = self._agent_factory(**kwargs)
        agent.restore(snapshot["agent"])
        role = snapshot.get("role")
        bus = EventBus()
        self.metrics.attach(bus, role=role)
        agent.bus = bus
        session = Session(
            agent,
            budget=Budget.from_dict(snapshot.get("budget")),
            id=snapshot["id"],
            role=role,
        )
        session.created_at = snapshot.get("created_at", session.created_at)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[SessionInfo]:
        with self._lock:
            return [s.info() for s in self._sessions.values()]

    def close(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True
