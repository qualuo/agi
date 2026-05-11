"""Session — one isolated unit of agent state.

A Session bundles everything one logical conversation needs:
  - an Agent (model, tools, conversation history, usage)
  - a Memory (long-term notes; can be shared across sessions or isolated)
  - an optional SkillLibrary (procedural memory)
  - an optional Budget (cost/turn caps)
  - an EventBus (observability; runtime-level bus optional)
  - a TraceLogger (durable record for the learning loop)

Sessions are the primitive a coordination engine schedules against. The
engine creates a session, calls `chat()` zero or more times, and closes
it. Sessions can have parents — useful when one agent delegates to a
subagent — so token usage rolls up.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agi.budget import Budget, BudgetExceeded
from agi.events import EventBus
from agi.memory import Memory
from agi.skills import SkillLibrary


@dataclass
class SessionInfo:
    """Lightweight snapshot of a session's state — safe to serialize."""

    id: str
    role: str
    model: str
    created_at: float
    parent_id: str | None
    turns: int
    cost_usd: float
    last_critic_score: float | None
    closed: bool
    tags: list[str] = field(default_factory=list)


class Session:
    """One conversational thread with isolated state."""

    def __init__(
        self,
        *,
        id: str | None = None,
        role: str = "general",
        model: str = "claude-opus-4-7",
        memory: Memory | None = None,
        skills: SkillLibrary | None = None,
        budget: Budget | None = None,
        bus: EventBus | None = None,
        tracer=None,
        critic=None,
        critic_threshold: float = 0.5,
        parent_id: str | None = None,
        system_extra: str = "",
        effort: str = "high",
        tags: list[str] | None = None,
        verbose: bool = False,
        # injection point for tests; default constructs the real Agent
        agent_factory=None,
    ) -> None:
        # Imported lazily so importing Session doesn't pull anthropic at module
        # load (matters for tools that just want SessionInfo / typing).
        if agent_factory is None:
            from agi.agent import Agent
            agent_factory = Agent

        self.id = id or uuid.uuid4().hex[:12]
        self.role = role
        self.model = model
        self.created_at = time.time()
        self.parent_id = parent_id
        self.tags = list(tags or [])
        self.closed = False
        self.bus = bus
        self.budget = budget
        self.skills = skills
        self.memory = memory or Memory()

        self.agent = agent_factory(
            memory=self.memory,
            model=model,
            effort=effort,
            verbose=verbose,
            tracer=tracer,
            critic=critic,
            critic_threshold=critic_threshold,
            skills=skills,
            budget=budget,
            event_bus=bus,
            session_id=self.id,
            system_extra=system_extra,
        )

        # Pending: a Runtime can call .install_delegate(runtime) to add the
        # `delegate` tool, bound to this session as parent.
        self._delegate_installed = False

        if self.bus is not None:
            self.bus.emit(
                "session_opened",
                session_id=self.id,
                role=self.role,
                model=self.model,
                parent_id=self.parent_id,
                tags=list(self.tags),
            )

    # ---- public API a coordination engine uses ----

    def chat(self, user_input: str, **kwargs: Any) -> dict:
        """Run one turn. Returns a dict with text, usage, critic, budget state."""
        if self.closed:
            raise RuntimeError(f"session {self.id} is closed")
        try:
            text = self.agent.chat(user_input, **kwargs)
            stop_reason = "ok"
        except BudgetExceeded as exc:
            stop_reason = f"budget_exceeded: {exc.reason}"
            text = ""
            if self.bus is not None:
                self.bus.emit(
                    "budget_exceeded",
                    session_id=self.id,
                    reason=exc.reason,
                )
        return {
            "session_id": self.id,
            "text": text,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": self.agent.usage.input_tokens,
                "output_tokens": self.agent.usage.output_tokens,
                "cache_creation_input_tokens": self.agent.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": self.agent.usage.cache_read_input_tokens,
                "turns": self.agent.usage.turns,
                "cost_usd": self.agent.usage.cost_usd(self.model),
            },
            "critic_score": self.agent.last_critic_score,
            "budget": {
                "remaining_usd": self.budget.remaining_usd(self.agent.usage) if self.budget else None,
                "remaining_turns": self.budget.remaining_turns(self.agent.usage) if self.budget else None,
            },
        }

    def reset(self) -> None:
        self.agent.reset()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.bus is not None:
            self.bus.emit(
                "session_closed",
                session_id=self.id,
                turns=self.agent.usage.turns,
                cost_usd=self.agent.usage.cost_usd(self.model),
            )

    def install_delegate(self, runtime) -> None:
        """Attach a `delegate` tool that spawns child sessions on `runtime`.

        Called by Runtime.open_session after the session is registered, so
        the parent_id is known and the runtime can find this session when
        the child returns.
        """
        if self._delegate_installed:
            return
        from agi.tools import make_delegate_tool
        schema, handler = make_delegate_tool(runtime, parent_id=self.id)
        self.agent.tool_schemas.append(schema)
        self.agent.handlers["delegate"] = handler
        self._delegate_installed = True

    def info(self) -> SessionInfo:
        return SessionInfo(
            id=self.id,
            role=self.role,
            model=self.model,
            created_at=self.created_at,
            parent_id=self.parent_id,
            turns=self.agent.usage.turns,
            cost_usd=self.agent.usage.cost_usd(self.model),
            last_critic_score=self.agent.last_critic_score,
            closed=self.closed,
            tags=list(self.tags),
        )
