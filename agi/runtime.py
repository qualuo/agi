"""Runtime — multi-session manager.

A `Runtime` is the top-level object a coordination engine drives. It owns:
  - the shared SkillLibrary (procedural memory, persistent)
  - the shared TraceLogger (durable record for the learning loop)
  - the optional shared Memory (or per-session — your choice)
  - the EventBus (one stream of typed events for everything)
  - the live Session table

A Runtime exposes a small, stable API:
  - `open_session(role=..., budget=..., parent_id=...)`
  - `chat(session_id, text)`
  - `close_session(session_id)`
  - `list_sessions()` / `get_session(id)`
  - `manifest()` — capability descriptor for service discovery
  - `subscribe(callback)` — attach to the event bus

Roles are *labels*, not enforced sandboxes. Coordination engines pick
roles like `planner`, `executor`, `critic` and pass an appropriate
system_extra for each.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from agi.budget import Budget
from agi.events import EventBus
from agi.memory import Memory
from agi.session import Session, SessionInfo
from agi.skills import SkillLibrary

try:
    from learner.traces import TraceLogger
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore


# ---- capability descriptor a coordination engine queries ----


def capability_manifest(model: str = "claude-opus-4-7") -> dict:
    """Static description of what this runtime can do.

    A coordination engine reads this to know which roles, tools, and
    limits exist before it starts scheduling work. Stable shape; add
    fields, don't rename or remove without bumping `version`.
    """
    return {
        "name": "agi-runtime",
        "version": "0.2.0",
        "model": model,
        "roles": [
            {"name": "general", "description": "default; planning + tool use + memory"},
            {"name": "planner", "description": "decompose; emit a plan; do not execute"},
            {"name": "executor", "description": "execute one step of a plan with tools"},
            {"name": "critic", "description": "grade an output; return pass/fail + reason"},
            {"name": "researcher", "description": "web_search + web_fetch + summarize"},
        ],
        "tools": [
            "read_file", "write_file", "list_dir", "run_bash",
            "save_memory", "search_memory", "recent_memory",
            "save_skill", "list_skills",
            "reflect", "delegate",
            "web_search", "web_fetch",
        ],
        "events": [
            "session_opened", "session_closed",
            "turn_started", "turn_finished",
            "thinking", "text",
            "tool_call", "tool_result", "server_tool",
            "budget_warning", "budget_exceeded",
            "critic_score", "skill_loaded",
            "delegate_spawn", "delegate_return",
            "error",
        ],
        "limits": {
            "max_sessions": None,
            "default_budget_usd": None,
            "supports_subagents": True,
            "supports_skills": True,
            "supports_critic": True,
        },
    }


# Per-role system prompt extras. A coordination engine can override by
# passing its own `system_extra` to `open_session`.
ROLE_PROMPTS: dict[str, str] = {
    "general": "",
    "planner": (
        "Role: PLANNER. Your job is to decompose the request into a numbered "
        "list of concrete steps another agent can execute. Output JSON of the "
        "form {\"plan\": [{\"step\": int, \"goal\": str, \"tool_hint\": str}, ...]}. "
        "Do not execute the plan yourself."
    ),
    "executor": (
        "Role: EXECUTOR. You receive one step of a plan. Execute it using "
        "tools. Return a concise result describing what you did and the "
        "outcome. Do not re-plan or expand scope."
    ),
    "critic": (
        "Role: CRITIC. You receive a (task, candidate_output) pair. Decide if "
        "the output satisfies the task. Output JSON {\"pass\": bool, "
        "\"reason\": str, \"score\": float between 0 and 1}."
    ),
    "researcher": (
        "Role: RESEARCHER. Use web_search and web_fetch to gather facts. "
        "Cite sources. Prefer primary sources. Return a brief, factual "
        "summary."
    ),
}


class Runtime:
    """Multi-session agent runtime."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        skill_library: SkillLibrary | None = None,
        memory: Memory | None = None,
        tracer=None,
        bus: EventBus | None = None,
        critic=None,
        critic_threshold: float = 0.5,
        default_budget: Budget | None = None,
        traces_path: str | Path | None = None,
        max_sessions: int | None = None,
        # for tests
        agent_factory=None,
    ) -> None:
        self.model = model
        self.skills = skill_library or SkillLibrary()
        self.memory = memory  # may be None → per-session memory
        self.bus = bus or EventBus(history=256)
        self.critic = critic
        self.critic_threshold = critic_threshold
        self.default_budget = default_budget
        self.max_sessions = max_sessions
        self._agent_factory = agent_factory

        if tracer is not None:
            self.tracer = tracer
        elif TraceLogger is not None:
            self.tracer = TraceLogger(path=traces_path)
        else:
            self.tracer = None

        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._created_at = time.time()

    # ---- service-discovery surface ----

    def manifest(self) -> dict:
        m = capability_manifest(self.model)
        m["limits"]["max_sessions"] = self.max_sessions
        m["limits"]["default_budget_usd"] = (
            self.default_budget.max_usd if self.default_budget else None
        )
        m["live_sessions"] = len(self._sessions)
        m["uptime_seconds"] = time.time() - self._created_at
        return m

    # ---- session lifecycle ----

    def open_session(
        self,
        *,
        role: str = "general",
        model: str | None = None,
        budget: Budget | None = None,
        parent_id: str | None = None,
        system_extra: str | None = None,
        memory: Memory | None = None,
        tags: list[str] | None = None,
        effort: str = "high",
        verbose: bool = False,
    ) -> Session:
        with self._lock:
            if self.max_sessions and len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"runtime at session cap ({self.max_sessions}); close one first"
                )
            sid = uuid.uuid4().hex[:12]

        # Compose the system_extra: per-role prompt + caller addition.
        role_prompt = ROLE_PROMPTS.get(role, "")
        composed_extra = "\n\n".join(p for p in (role_prompt, system_extra or "") if p)

        session = Session(
            id=sid,
            role=role,
            model=model or self.model,
            memory=memory or self.memory or Memory(),
            skills=self.skills,
            budget=budget or self.default_budget,
            bus=self.bus,
            tracer=self.tracer,
            critic=self.critic,
            critic_threshold=self.critic_threshold,
            parent_id=parent_id,
            system_extra=composed_extra,
            effort=effort,
            tags=tags,
            verbose=verbose,
            agent_factory=self._agent_factory,
        )
        with self._lock:
            self._sessions[sid] = session
        # Wire delegation now that the session is discoverable.
        session.install_delegate(self)
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[SessionInfo]:
        with self._lock:
            return [s.info() for s in self._sessions.values()]

    def chat(self, session_id: str, text: str, **kwargs: Any) -> dict:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")
        return session.chat(text, **kwargs)

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True

    def close_all(self) -> int:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.close()
        return len(sessions)

    # ---- delegation: a tool inside one agent spawns a child ----

    def delegate(
        self,
        *,
        parent_id: str,
        task: str,
        role: str = "executor",
        model: str | None = None,
        budget: Budget | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Run a one-shot child session. Returns the child's chat() dict.

        Child usage rolls up to the parent's accounting via the event
        bus; the child is closed automatically.
        """
        parent = self.get_session(parent_id)
        if parent is None:
            raise KeyError(f"unknown parent session: {parent_id}")

        self.bus.emit(
            "delegate_spawn",
            parent_id=parent_id,
            role=role,
            task_preview=task[:120],
        )

        child = self.open_session(
            role=role,
            model=model,
            budget=budget,
            parent_id=parent_id,
            tags=(tags or []) + ["delegated"],
        )
        try:
            result = child.chat(task)
        finally:
            self.close_session(child.id)

        self.bus.emit(
            "delegate_return",
            parent_id=parent_id,
            child_id=child.id,
            stop_reason=result.get("stop_reason"),
            cost_usd=result["usage"]["cost_usd"],
        )
        return result

    # ---- event subscription passthroughs ----

    def subscribe(self, cb: Callable[[dict], None]):
        return self.bus.subscribe(cb)

    def history(self) -> list[dict]:
        return self.bus.history()
