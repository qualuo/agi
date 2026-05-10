"""Runtime engine — what a coordination engine talks to.

The Agent is one task in flight. The Runtime owns many of them. A
coordination engine (workflow orchestrator, planner, multi-tool router,
human in the loop) drives the runtime: spin up a session, push input,
read state, snapshot/resume, tear down. Each session has its own
isolated memory, skill library, and tool registry.

API shape:

    rt = Runtime()
    sess = rt.create_session(role="general", goal="Summarize ./README.md")
    result = sess.step("Summarize ./README.md")
    snap = sess.snapshot()                  # cheap, no API call
    sess.inject_observation("user uploaded a new file")  # context push
    sess.end()                              # finalize, optional reflection

    capabilities = rt.capabilities()        # what this runtime can do
    rt.list_sessions()                      # live + historical session ids

The runtime is intentionally synchronous in v1. Async wrappers
slot in around `step` without changing the data model.

Roles configure the system prompt and the tool subset. The default
roles are:

  general    full toolset, the standard Opus harness
  planner    decompose a task into a numbered plan; no side-effecting tools
  executor   execute one focused step; full toolset
  critic     review work for correctness/risk; read-only tools

Roles are extensible — pass `extra_roles=` to Runtime() to override.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import anthropic

from agi.agent import Agent, Hooks, StepResult
from agi.memory import Memory
from agi.reflection import Reflector
from agi.skills import SkillLibrary
from agi.tools import make_tools

try:
    from learner.traces import TraceLogger
except ImportError:
    TraceLogger = None  # type: ignore


# ---- role registry ----

@dataclass
class Role:
    name: str
    system_prompt: str
    # If set, only the named tools are exposed to the agent.
    allowed_tools: list[str] | None = None
    enable_web_search: bool = True
    enable_web_fetch: bool = True
    enable_synth: bool = True
    can_delegate: bool = False
    skills_top_k: int = 3
    effort: str = "high"


_PLANNER_PROMPT = """\
You are the planner sub-agent. Given a goal, output a numbered plan of 2-7
concrete steps a separate executor can carry out. Do not execute the steps
yourself. Each step must be:
- a single concrete action,
- testable for completion,
- ordered (later steps may depend on earlier ones).

Output the plan as a numbered list. End with a one-line success criterion.
"""

_EXECUTOR_PROMPT = """\
You are the executor sub-agent. You are given one focused step to carry out.
Use tools to do it. Verify before claiming success. Return the concrete
result the parent agent needs. Be terse — do not restate the plan.
"""

_CRITIC_PROMPT = """\
You are the critic sub-agent. You are given a piece of work (a plan, a code
diff, an answer, or an artifact). Review it for: correctness, missing
edge cases, security risks, and unstated assumptions. Output:
- VERDICT: pass | needs_revision | reject
- ISSUES: bullet list (empty if pass)
- SUGGESTED FIX: one sentence (omit if pass)

Be concise. The parent agent decides what to do with your verdict.
"""


DEFAULT_ROLES: dict[str, Role] = {
    "general": Role(
        name="general",
        system_prompt="",  # uses Agent default
        allowed_tools=None,
        can_delegate=True,
    ),
    "planner": Role(
        name="planner",
        system_prompt=_PLANNER_PROMPT,
        allowed_tools=["search_memory", "recent_memory", "list_skills", "read_skill"],
        enable_web_search=True,
        enable_web_fetch=False,
        enable_synth=False,
        can_delegate=False,
        effort="high",
    ),
    "executor": Role(
        name="executor",
        system_prompt=_EXECUTOR_PROMPT,
        allowed_tools=None,
        can_delegate=False,
        effort="medium" if False else "high",
    ),
    "critic": Role(
        name="critic",
        system_prompt=_CRITIC_PROMPT,
        allowed_tools=["read_file", "list_dir", "search_memory", "recent_memory", "list_skills", "read_skill"],
        enable_web_search=True,
        enable_web_fetch=True,
        enable_synth=False,
        can_delegate=False,
        effort="high",
    ),
}


# ---- session ----

@dataclass
class SessionMetadata:
    id: str
    role: str
    goal: str | None
    created_ts: float
    ended_ts: float | None = None
    parent_session_id: str | None = None


class Session:
    """One conversation backed by an Agent. Owned by a Runtime.

    A coordination engine treats this as a long-lived handle: it pushes
    inputs into the session and reads outputs / state, optionally
    snapshotting between steps.
    """

    def __init__(
        self,
        runtime: "Runtime",
        meta: SessionMetadata,
        agent: Agent,
    ) -> None:
        self.runtime = runtime
        self.meta = meta
        self.agent = agent
        self.history: list[StepResult] = []
        self._lock = threading.Lock()

    # ---- public ----

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def is_active(self) -> bool:
        return self.meta.ended_ts is None

    def step(self, user_input: str, *, max_iterations: int = 25) -> StepResult:
        """One user turn. Thread-safe at the session level."""
        with self._lock:
            if not self.is_active:
                raise RuntimeError(f"session {self.id} has been ended")
            result = self.agent.step(user_input, max_iterations=max_iterations)
            self.history.append(result)
            return result

    def inject_observation(self, text: str, role: str = "user") -> None:
        """Add a context message without taking a turn.

        Lets the coordination engine push environment updates ("the
        deploy finished", "user clicked accept") without forcing a
        full inference call.
        """
        with self._lock:
            self.agent.messages.append({"role": role, "content": text})

    def snapshot(self) -> dict:
        snap = self.agent.snapshot()
        snap.update(
            {
                "session_id": self.id,
                "role": self.meta.role,
                "goal": self.meta.goal,
                "active": self.is_active,
                "created_ts": self.meta.created_ts,
                "ended_ts": self.meta.ended_ts,
                "history_steps": len(self.history),
                "parent_session_id": self.meta.parent_session_id,
            }
        )
        return snap

    def transcript(self) -> list[dict]:
        """Plain-dict transcript safe to send across a process boundary."""
        from learner.traces import _serialize_messages  # local import to avoid cycle
        return _serialize_messages(self.agent.messages)

    def end(self, *, reason: str = "complete") -> dict:
        with self._lock:
            if not self.is_active:
                return self.snapshot()
            self.meta.ended_ts = time.time()
            snap = self.snapshot()
            snap["end_reason"] = reason
            return snap


# ---- runtime ----

class Runtime:
    """Top-level container for sessions and shared resources.

    A coordination engine instantiates one Runtime and uses it to manage
    many concurrent sessions. Memory and skill library are shared
    across sessions by default — this is what makes long-running
    learning possible — but can be overridden per-session for isolation.
    """

    def __init__(
        self,
        *,
        memory: Memory | None = None,
        skills: SkillLibrary | None = None,
        tracer=None,
        client: anthropic.Anthropic | None = None,
        reflector: Reflector | None = None,
        enable_reflection: bool = False,
        roles: dict[str, Role] | None = None,
        model: str = "claude-opus-4-7",
    ) -> None:
        self.memory = memory or Memory()
        self.skills = skills or SkillLibrary()
        self.client = client or _maybe_client()
        self.tracer = tracer or (TraceLogger() if TraceLogger else None)
        self.model = model
        self.roles: dict[str, Role] = dict(roles or DEFAULT_ROLES)

        # The reflector is one shared module that writes into shared memory.
        if reflector is not None:
            self.reflector = reflector
        elif enable_reflection and self.client is not None:
            self.reflector = Reflector(self.memory, client=self.client)
        else:
            self.reflector = None

        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ---- session lifecycle ----

    def create_session(
        self,
        *,
        role: str = "general",
        goal: str | None = None,
        memory: Memory | None = None,
        skills: SkillLibrary | None = None,
        hooks: Hooks | None = None,
        parent_session_id: str | None = None,
        model: str | None = None,
        max_tokens: int = 16000,
        verbose: bool = False,
    ) -> Session:
        if role not in self.roles:
            raise ValueError(
                f"unknown role {role!r}; available: {sorted(self.roles)}"
            )
        role_cfg = self.roles[role]
        sess_memory = memory or self.memory
        sess_skills = skills or self.skills
        sess_id = uuid.uuid4().hex[:12]
        meta = SessionMetadata(
            id=sess_id,
            role=role,
            goal=goal,
            created_ts=time.time(),
            parent_session_id=parent_session_id,
        )

        # delegate_fn: bound so a sub-agent runs against this same runtime
        delegate_fn = self._delegate_callable() if role_cfg.can_delegate else None

        registry = make_tools(
            sess_memory,
            skills=sess_skills,
            delegate_fn=delegate_fn,
            enable_synth=role_cfg.enable_synth,
        )
        if role_cfg.allowed_tools is not None:
            allowed = set(role_cfg.allowed_tools)
            for name in list(registry.handlers):
                if name not in allowed:
                    registry.remove(name)

        # Default Agent system prompt; override only when the role declares one.
        from agi.agent import SYSTEM_PROMPT as DEFAULT_SP
        system_prompt = role_cfg.system_prompt or DEFAULT_SP

        agent = Agent(
            memory=sess_memory,
            model=model or self.model,
            max_tokens=max_tokens,
            effort=role_cfg.effort,
            enable_web_search=role_cfg.enable_web_search,
            enable_web_fetch=role_cfg.enable_web_fetch,
            verbose=verbose,
            tracer=self.tracer,
            skills=sess_skills,
            skills_top_k=role_cfg.skills_top_k,
            reflector=self.reflector,
            hooks=hooks,
            registry=registry,
            system_prompt=system_prompt,
            client=self.client,
        )
        sess = Session(self, meta, agent)
        with self._lock:
            self._sessions[sess_id] = sess
        return sess

    def get(self, session_id: str) -> Session:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"no session {session_id!r}")
        return sess

    def list_sessions(self, *, active_only: bool = False) -> list[dict]:
        out = []
        for s in self._sessions.values():
            if active_only and not s.is_active:
                continue
            out.append(
                {
                    "id": s.id,
                    "role": s.meta.role,
                    "goal": s.meta.goal,
                    "active": s.is_active,
                    "created_ts": s.meta.created_ts,
                    "ended_ts": s.meta.ended_ts,
                    "parent": s.meta.parent_session_id,
                    "history_steps": len(s.history),
                }
            )
        return out

    def end_session(self, session_id: str, *, reason: str = "complete") -> dict:
        return self.get(session_id).end(reason=reason)

    # ---- delegation ----

    def _delegate_callable(self) -> Callable[..., str]:
        """Return a callable that spawns a sub-agent for `delegate(role, task)`."""
        def delegate(role: str, task: str) -> str:
            # Sub-sessions get a fresh agent instance with the role's config,
            # share the parent runtime's memory and skills, no further nesting.
            sub = self.create_session(role=role, goal=task)
            try:
                result = sub.step(task)
                return result.text
            finally:
                sub.end(reason=f"delegated_{role}")
        return delegate

    # ---- introspection ----

    def capabilities(self) -> dict:
        """Describe what this runtime offers.

        Coordination engines use this on connect to discover roles,
        tools, and policy knobs. Stable shape; new keys are additive.
        """
        # Build a registry off the shared memory just to enumerate the
        # default tool surface (without committing to any session).
        sample_registry = make_tools(self.memory, skills=self.skills, enable_synth=True)
        return {
            "version": 1,
            "model": self.model,
            "roles": [
                {
                    "name": r.name,
                    "system_prompt_excerpt": (r.system_prompt or "(default)")[:200],
                    "allowed_tools": r.allowed_tools,
                    "can_delegate": r.can_delegate,
                    "enable_synth": r.enable_synth,
                    "enable_web_search": r.enable_web_search,
                    "enable_web_fetch": r.enable_web_fetch,
                    "effort": r.effort,
                }
                for r in self.roles.values()
            ],
            "tools": sample_registry.descriptors() + [
                {"name": "web_search", "description": "Server-side web search."},
                {"name": "web_fetch", "description": "Server-side URL fetch."},
            ],
            "skill_count": len(self.skills.all()),
            "memory_path": str(self.memory.path),
            "tracing_enabled": self.tracer is not None,
            "reflection_enabled": self.reflector is not None,
        }

    def stats(self) -> dict:
        """Aggregate accounting across all live sessions."""
        cost = 0.0
        in_tok = out_tok = 0
        steps = 0
        for s in self._sessions.values():
            cost += s.agent.usage.cost_usd(s.agent.model)
            in_tok += s.agent.usage.input_tokens
            out_tok += s.agent.usage.output_tokens
            steps += s.agent.usage.turns
        return {
            "sessions": len(self._sessions),
            "active_sessions": sum(1 for s in self._sessions.values() if s.is_active),
            "total_cost_usd": cost,
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
            "total_turns": steps,
        }


def _maybe_client() -> anthropic.Anthropic | None:
    """Build a client only if the API key is set. Lets unit tests construct
    a Runtime without credentials for capability/snapshot inspection."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic()
