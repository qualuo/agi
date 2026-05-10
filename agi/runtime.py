"""Runtime engine — the surface a coordination engine drives.

A `Runtime` owns many `Session`s and an `EventBus`. A coordinator (any
caller — orchestrator, web UI, autonomous planner) creates sessions,
submits work, subscribes to the event stream, and reads back metrics.

This module decouples *what an agent is doing right now* (the Session)
from *how a higher-level system schedules and observes work* (the
Runtime). The same Runtime can be wrapped by:

  - a CLI loop (one user, one session)
  - an HTTP/SSE server (many users, many sessions)
  - an orchestrator (one planner, many specialist sessions in parallel)

The Runtime intentionally exposes:
  - capabilities (what tools/skills the runtime offers right now)
  - sessions (state, usage, last activity)
  - event stream (everything happening)
  - control (create / chat / cancel / end)

Sessions are lazy on the Agent: we only instantiate the Anthropic client
when the first chat() lands. This keeps Runtime() free for tests.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agi.events import (
    CHAT_COMPLETED,
    CHAT_STARTED,
    ERROR,
    SESSION_CREATED,
    SESSION_ENDED,
    SKILL_LOADED,
    TOOL_SYNTHESIZED,
    USAGE_UPDATED,
    Event,
    EventBus,
)
from agi.memory import Memory
from agi.skills import Skill, SkillLibrary
from agi.toolsynth import ToolSynthError, ToolSynthRegistry


@dataclass
class SessionConfig:
    model: str = "claude-opus-4-7"
    effort: str = "high"
    max_tokens: int = 16000
    max_iterations: int = 25
    enable_web_search: bool = True
    enable_web_fetch: bool = True
    enable_tool_synthesis: bool = False
    enable_delegation: bool = False
    use_skills: bool = True
    critic_threshold: float = 0.5
    system_prompt_extra: str | None = None
    role: str | None = None
    cost_ceiling_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    id: str
    config: SessionConfig
    created_ts: float = field(default_factory=time.time)
    last_activity_ts: float = field(default_factory=time.time)
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    last_critic_score: float | None = None
    ended: bool = False
    cancelled: bool = False
    parent_session_id: str | None = None


class Session:
    """One conversational session. Wraps an Agent and emits events.

    The Agent itself is lazy: not constructed until `chat()` is called the
    first time. This keeps unit tests that exercise only the Runtime
    bookkeeping (without API access) cheap.
    """

    def __init__(
        self,
        *,
        session_id: str,
        config: SessionConfig,
        bus: EventBus,
        memory: Memory,
        skills: SkillLibrary | None = None,
        tool_synth: ToolSynthRegistry | None = None,
        tracer=None,
        critic=None,
        agent_factory: Callable[..., Any] | None = None,
        parent_session_id: str | None = None,
    ) -> None:
        self.state = SessionState(
            id=session_id, config=config, parent_session_id=parent_session_id
        )
        self._bus = bus
        self._memory = memory
        self._skills = skills
        self._tool_synth = tool_synth
        self._tracer = tracer
        self._critic = critic
        self._agent_factory = agent_factory
        self._agent: Any | None = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        # Tracks last-seen agent.usage cost so we can convert into deltas
        # that survive concurrent subagent rollups.
        self._last_agent_cost: float = 0.0

    # --- public API --------------------------------------------------

    def chat(self, user_input: str) -> str:
        """Run one turn synchronously. Emits events; returns final text."""
        if self.state.ended:
            raise RuntimeError(f"session {self.state.id} has ended")
        if self.state.cancelled:
            self._cancel_event.clear()
            self.state.cancelled = False

        prompt = self._augment_with_skills(user_input)

        self._bus.publish(Event(
            kind=CHAT_STARTED,
            session_id=self.state.id,
            data={"user_input": user_input, "augmented": prompt != user_input, "turn": self.state.turn_count},
        ))

        agent = self._ensure_agent()
        # Snapshot before so delegate-time rollups (which run inside
        # agent.chat) survive the post-chat bookkeeping.
        prev_in = getattr(agent.usage, "input_tokens", 0) if getattr(agent, "usage", None) else 0
        prev_out = getattr(agent.usage, "output_tokens", 0) if getattr(agent, "usage", None) else 0

        try:
            final_text = agent.chat(prompt, max_iterations=self.state.config.max_iterations)
        except Exception as e:
            self._bus.publish(Event(
                kind=ERROR,
                session_id=self.state.id,
                data={"phase": "chat", "type": type(e).__name__, "message": str(e)},
            ))
            raise

        with self._lock:
            self.state.turn_count += 1
            self.state.last_activity_ts = time.time()
            self.state.last_critic_score = getattr(agent, "last_critic_score", None)
            usage = getattr(agent, "usage", None)
            if usage is not None:
                delta_in = usage.input_tokens - prev_in
                delta_out = usage.output_tokens - prev_out
                self.state.total_input_tokens += delta_in
                self.state.total_output_tokens += delta_out
                self.state.total_cost_usd = (
                    self.state.total_cost_usd
                    + (usage.cost_usd(self.state.config.model) - self._last_agent_cost)
                )
                self._last_agent_cost = usage.cost_usd(self.state.config.model)

        self._bus.publish(Event(
            kind=USAGE_UPDATED,
            session_id=self.state.id,
            data={
                "input_tokens": self.state.total_input_tokens,
                "output_tokens": self.state.total_output_tokens,
                "cost_usd": self.state.total_cost_usd,
                "turns": self.state.turn_count,
            },
        ))

        self._bus.publish(Event(
            kind=CHAT_COMPLETED,
            session_id=self.state.id,
            data={
                "final_text": final_text,
                "critic_score": self.state.last_critic_score,
                "turn": self.state.turn_count,
            },
        ))

        self._enforce_budget()
        return final_text

    def cancel(self) -> None:
        """Request the next chat boundary to abort. Best-effort — does not
        kill an in-flight stream mid-token; the Agent loop checks this
        between turns."""
        self.state.cancelled = True
        self._cancel_event.set()

    def end(self) -> None:
        if not self.state.ended:
            self.state.ended = True
            self._bus.publish(Event(
                kind=SESSION_ENDED,
                session_id=self.state.id,
                data={
                    "turn_count": self.state.turn_count,
                    "total_cost_usd": self.state.total_cost_usd,
                },
            ))

    def reset(self) -> None:
        if self._agent is not None and hasattr(self._agent, "reset"):
            self._agent.reset()
        self.state.turn_count = 0
        self.state.total_input_tokens = 0
        self.state.total_output_tokens = 0
        self.state.total_cost_usd = 0.0
        self.state.last_critic_score = None
        self.state.last_activity_ts = time.time()
        self._last_agent_cost = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.state.id,
            "config": self.state.config.__dict__,
            "created_ts": self.state.created_ts,
            "last_activity_ts": self.state.last_activity_ts,
            "turn_count": self.state.turn_count,
            "total_input_tokens": self.state.total_input_tokens,
            "total_output_tokens": self.state.total_output_tokens,
            "total_cost_usd": self.state.total_cost_usd,
            "last_critic_score": self.state.last_critic_score,
            "ended": self.state.ended,
            "cancelled": self.state.cancelled,
            "parent_session_id": self.state.parent_session_id,
        }

    # --- internals ---------------------------------------------------

    def _augment_with_skills(self, user_input: str) -> str:
        if not self.state.config.use_skills or self._skills is None:
            return user_input
        relevant = self._skills.retrieve(user_input, k=3)
        if not relevant:
            return user_input
        for s in relevant:
            self._bus.publish(Event(
                kind=SKILL_LOADED,
                session_id=self.state.id,
                data={"name": s.name, "description": s.description},
            ))
        block = self._skills.format_for_prompt(relevant)
        return f"{block}\n\n# User request\n{user_input}"

    def _ensure_agent(self):
        if self._agent is not None:
            return self._agent
        if self._agent_factory is None:
            # Lazy import: keeps Runtime() usable without the anthropic
            # SDK installed (used by tests that don't hit the API).
            from agi.agent import Agent
            factory = Agent
        else:
            factory = self._agent_factory

        agent = factory(
            memory=self._memory,
            model=self.state.config.model,
            max_tokens=self.state.config.max_tokens,
            effort=self.state.config.effort,
            enable_web_search=self.state.config.enable_web_search,
            enable_web_fetch=self.state.config.enable_web_fetch,
            verbose=False,
            tracer=self._tracer,
            critic=self._critic,
            critic_threshold=self.state.config.critic_threshold,
        )
        if self.state.config.system_prompt_extra and hasattr(agent, "extra_system"):
            agent.extra_system = self.state.config.system_prompt_extra
        if self._tool_synth is not None and self.state.config.enable_tool_synthesis:
            if hasattr(agent, "attach_tool_synth"):
                agent.attach_tool_synth(self._tool_synth, self._bus, self.state.id)
        if self.state.config.enable_delegation and hasattr(agent, "attach_delegation"):
            agent.attach_delegation(self._spawn_subagent, self._bus, self.state.id)
        self._agent = agent
        return agent

    def _spawn_subagent(self, *, task: str, role: str, model: str | None = None) -> str:
        """Delegation hook injected into the Agent. Returns the subagent's
        final text. Costs and events flow through the same Runtime."""
        raise NotImplementedError("Session._spawn_subagent is wired by Runtime")

    def _enforce_budget(self) -> None:
        ceiling = self.state.config.cost_ceiling_usd
        if ceiling is not None and self.state.total_cost_usd >= ceiling:
            self._bus.publish(Event(
                kind=ERROR,
                session_id=self.state.id,
                data={
                    "phase": "budget",
                    "type": "CostCeilingExceeded",
                    "message": f"session cost ${self.state.total_cost_usd:.4f} ≥ ceiling ${ceiling:.4f}",
                },
            ))
            self.end()


class Runtime:
    """Top-level engine. The integration point for coordination engines.

    Surface:
      - capabilities()  → what tools/skills/models are available
      - create_session(config) → session_id
      - chat(session_id, prompt) → str (blocking; emits events)
      - cancel/end/reset
      - sessions() / get_session()
      - subscribe(callback, …) → subscription id
      - synthesize_tool(...) → register a new tool
      - skills / memory exposed as live components

    Concurrency: chat() on different sessions is safe in parallel. chat()
    on the *same* session must be serialized by the caller (we don't lock
    it here because the typical model is "one coordinator turn per
    session at a time"). The Session._lock guards bookkeeping only.
    """

    def __init__(
        self,
        *,
        memory: Memory | None = None,
        skills: SkillLibrary | None = None,
        tool_synth: ToolSynthRegistry | None = None,
        bus: EventBus | None = None,
        tracer=None,
        critic=None,
        agent_factory: Callable[..., Any] | None = None,
        skills_dir: str | Path | None = None,
    ) -> None:
        self.memory = memory or Memory()
        self.skills = skills or SkillLibrary(path=skills_dir)
        self.tool_synth = tool_synth or ToolSynthRegistry()
        self.bus = bus or EventBus()
        self.tracer = tracer
        self.critic = critic
        self._agent_factory = agent_factory
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # --- discovery ---------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        """Inventory for a coordination engine: what this runtime can do."""
        from agi.costs import PRICING
        return {
            "models": list(PRICING.keys()),
            "skills": [
                {"name": s.name, "description": s.description, "tags": s.tags}
                for s in self.skills.all()
            ],
            "synthesized_tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "invocation_count": t.invocation_count,
                    "error_count": t.error_count,
                }
                for t in self.tool_synth.list()
            ],
            "active_sessions": sum(1 for s in self._sessions.values() if not s.state.ended),
            "total_sessions": len(self._sessions),
            "memory_notes": len(self.memory.all()),
        }

    # --- session lifecycle ------------------------------------------

    def create_session(
        self,
        config: SessionConfig | None = None,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> str:
        sid = session_id or uuid.uuid4().hex[:12]
        cfg = config or SessionConfig()
        with self._lock:
            if sid in self._sessions:
                raise ValueError(f"session id already exists: {sid}")
            session = Session(
                session_id=sid,
                config=cfg,
                bus=self.bus,
                memory=self.memory,
                skills=self.skills,
                tool_synth=self.tool_synth,
                tracer=self.tracer,
                critic=self.critic,
                agent_factory=self._agent_factory,
                parent_session_id=parent_session_id,
            )
            session._spawn_subagent = self._spawn_subagent_for(session)  # type: ignore[assignment]
            self._sessions[sid] = session
        self.bus.publish(Event(
            kind=SESSION_CREATED,
            session_id=sid,
            data={"config": cfg.__dict__, "parent_session_id": parent_session_id},
        ))
        return sid

    def chat(self, session_id: str, user_input: str) -> str:
        session = self._require(session_id)
        return session.chat(user_input)

    def cancel(self, session_id: str) -> None:
        self._require(session_id).cancel()

    def end_session(self, session_id: str) -> None:
        self._require(session_id).end()

    def reset_session(self, session_id: str) -> None:
        self._require(session_id).reset()

    def get_session(self, session_id: str) -> Session:
        return self._require(session_id)

    def sessions(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    def _require(self, session_id: str) -> Session:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"unknown session id: {session_id}")
        return s

    # --- events ------------------------------------------------------

    def subscribe(self, callback, *, session_id: str | None = None, kind: str | None = None) -> int:
        return self.bus.subscribe(callback, session_id=session_id, kind=kind)

    def unsubscribe(self, sub_id: int) -> bool:
        return self.bus.unsubscribe(sub_id)

    def events(self, **kwargs) -> list[Event]:
        return self.bus.history(**kwargs)

    # --- skills + memory + tool synthesis (coord surface) ------------

    def save_skill(self, skill: Skill) -> None:
        self.skills.save(skill)

    def synthesize_tool(
        self,
        *,
        name: str,
        description: str,
        code: str,
        input_schema: dict[str, Any] | None = None,
        smoke_test_kwargs: dict[str, Any] | None = None,
    ):
        try:
            tool = self.tool_synth.register(
                name=name,
                description=description,
                code=code,
                input_schema=input_schema,
                smoke_test_kwargs=smoke_test_kwargs,
            )
        except ToolSynthError as e:
            self.bus.publish(Event(kind=ERROR, data={"phase": "tool_synth", "name": name, "message": str(e)}))
            raise
        self.bus.publish(Event(
            kind=TOOL_SYNTHESIZED,
            data={"name": tool.name, "description": tool.description},
        ))
        return tool

    # --- subagent delegation ----------------------------------------

    def _spawn_subagent_for(self, parent: Session):
        """Returns a function `(task, role, model=…) → final_text` that the
        Agent can wire into a `delegate` tool. The child session is a
        first-class Runtime session — coordinators see it in `sessions()`."""
        def spawn(*, task: str, role: str, model: str | None = None) -> str:
            child_cfg = SessionConfig(
                model=model or parent.state.config.model,
                effort=parent.state.config.effort,
                max_tokens=parent.state.config.max_tokens,
                max_iterations=min(parent.state.config.max_iterations, 15),
                enable_web_search=parent.state.config.enable_web_search,
                enable_web_fetch=parent.state.config.enable_web_fetch,
                enable_tool_synthesis=False,  # subagents don't synthesize further
                enable_delegation=False,       # no recursion by default
                use_skills=True,
                system_prompt_extra=f"You are operating as a specialist subagent. Role: {role}. Return a concise final answer.",
                role=role,
                cost_ceiling_usd=parent.state.config.cost_ceiling_usd,
            )
            child_id = self.create_session(child_cfg, parent_session_id=parent.state.id)
            self.bus.publish(Event(
                kind="subagent.started",
                session_id=parent.state.id,
                data={"child_id": child_id, "role": role, "task": task},
            ))
            try:
                result = self.chat(child_id, task)
            finally:
                child = self._sessions.get(child_id)
                if child is not None:
                    # Roll up subagent cost into parent for honest accounting
                    with parent._lock:
                        parent.state.total_cost_usd += child.state.total_cost_usd
                        parent.state.total_input_tokens += child.state.total_input_tokens
                        parent.state.total_output_tokens += child.state.total_output_tokens
                    child.end()
            self.bus.publish(Event(
                kind="subagent.completed",
                session_id=parent.state.id,
                data={"child_id": child_id, "role": role, "final_text": result},
            ))
            return result
        return spawn
