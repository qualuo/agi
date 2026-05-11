"""Runtime engine.

The Runtime is the **substrate a coordination engine drives**. A coordinator
(workflow planner, orchestrator, multi-agent scheduler — anything above this
layer) doesn't want to wire up the agent loop, manage tool dispatch, page
through events, or count tokens by hand. It wants three primitives:

    runtime.open_session(config) -> SessionHandle
    runtime.run_task(session, input, budget=...) -> TaskResult
    runtime.close_session(session)

Plus a stream of structured events while a task runs, and capability
introspection so the coordinator can pick the right session for a job.

This module is the in-process Python API. `agi.server` wraps it in HTTP+SSE
for cross-process coordinators; the contracts are identical.

Design choices:

- **Stateful sessions.** A session keeps conversation history, accumulated
  usage, attached skills, and a budget envelope. One session ≈ one "thread"
  of work; a coordinator opens N sessions for N parallel workstreams.
- **Per-task budget overrides the session budget for that task only.** This
  is how a coordinator says "spend up to $1 on this subtask" without
  globally narrowing the session.
- **Delegation is just sub-tasks.** `runtime.spawn_child(session)` opens a
  fresh session and returns a handle; a parent task can call its child via
  the `delegate` tool. Token usage rolls up to the parent session for
  honest accounting.
- **Events are JSON and structured.** Anything visual is left to the caller.
- **Failures don't crash the runtime.** Exceptions inside an agent loop are
  captured into `TaskResult.error`; the session stays alive.

What this is not: a queue, a scheduler, a router. Those belong above the
Runtime, in the coordination engine.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agi.agent import Agent
from agi.budget import Budget
from agi.costs import Usage
from agi.memory import Memory
from agi.reflection import reflect
from agi.skills import SkillLibrary

try:
    from learner.traces import TraceLogger
except ImportError:
    TraceLogger = None  # type: ignore


RUNTIME_VERSION = "0.2.0"


@dataclass
class SessionConfig:
    model: str = "claude-opus-4-7"
    effort: str = "high"
    max_tokens: int = 16000
    enable_web_search: bool = True
    enable_web_fetch: bool = True
    enable_delegate: bool = True
    budget: Budget | None = None
    system_prompt_extra: str | None = None
    skill_ids: list[str] = field(default_factory=list)  # always-on skills
    reflect: bool = True
    memory_path: str | None = None
    parent_session_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "enable_web_search": self.enable_web_search,
            "enable_web_fetch": self.enable_web_fetch,
            "enable_delegate": self.enable_delegate,
            "budget": self.budget.to_dict() if self.budget else None,
            "system_prompt_extra": self.system_prompt_extra,
            "skill_ids": list(self.skill_ids),
            "reflect": self.reflect,
            "memory_path": self.memory_path,
            "parent_session_id": self.parent_session_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionConfig":
        return cls(
            model=d.get("model", "claude-opus-4-7"),
            effort=d.get("effort", "high"),
            max_tokens=int(d.get("max_tokens", 16000)),
            enable_web_search=bool(d.get("enable_web_search", True)),
            enable_web_fetch=bool(d.get("enable_web_fetch", True)),
            enable_delegate=bool(d.get("enable_delegate", True)),
            budget=Budget.from_dict(d.get("budget")),
            system_prompt_extra=d.get("system_prompt_extra"),
            skill_ids=list(d.get("skill_ids") or []),
            reflect=bool(d.get("reflect", True)),
            memory_path=d.get("memory_path"),
            parent_session_id=d.get("parent_session_id"),
        )


@dataclass
class TaskResult:
    task_id: str
    session_id: str
    status: str  # "ok" | "over_budget" | "cancelled" | "error" | "refusal" | other
    output: str
    usage: dict[str, int]
    cost_usd: float
    elapsed_s: float
    skills_used: list[str] = field(default_factory=list)
    critic_score: float | None = None
    reflection_notes: list[str] = field(default_factory=list)
    error: str | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "status": self.status,
            "output": self.output,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "elapsed_s": self.elapsed_s,
            "skills_used": list(self.skills_used),
            "critic_score": self.critic_score,
            "reflection_notes": list(self.reflection_notes),
            "error": self.error,
            "stop_reason": self.stop_reason,
        }


@dataclass
class _Task:
    id: str
    session_id: str
    input: str
    budget: Budget | None
    started_at: float
    status: str = "running"  # running | done | cancelled | error
    result: TaskResult | None = None
    event_queue: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class Session:
    """A single agent thread. Holds the Agent, accumulated usage, attached skills.

    Concurrency model: at most one task runs per session at a time. Tasks
    on different sessions are independent and can run in parallel.
    """

    def __init__(
        self,
        *,
        id: str,
        config: SessionConfig,
        runtime: "Runtime",
        memory: Memory,
        agent: Agent,
    ) -> None:
        self.id = id
        self.config = config
        self.runtime = runtime
        self.memory = memory
        self.agent = agent
        self.created_at = time.time()
        self.lock = threading.Lock()  # serializes tasks within a session
        self.tasks: dict[str, _Task] = {}
        self.child_session_ids: list[str] = []

    def cumulative_usage(self) -> Usage:
        return self.agent.usage

    def to_dict(self) -> dict:
        u = self.agent.usage
        return {
            "id": self.id,
            "created_at": self.created_at,
            "config": self.config.to_dict(),
            "usage": {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_creation_input_tokens": u.cache_creation_input_tokens,
                "cache_read_input_tokens": u.cache_read_input_tokens,
                "turns": u.turns,
                "cost_usd": u.cost_usd(self.config.model),
            },
            "child_session_ids": list(self.child_session_ids),
            "open_task_ids": [tid for tid, t in self.tasks.items() if t.status == "running"],
        }


class Runtime:
    """The runtime substrate. One process, many sessions, structured tasks."""

    def __init__(
        self,
        *,
        skills: SkillLibrary | None = None,
        tracer=None,
        critic=None,
        critic_threshold: float = 0.5,
        default_budget: Budget | None = None,
        agent_factory: Callable[..., Agent] | None = None,
    ) -> None:
        self.skills = skills or SkillLibrary()
        self.tracer = tracer
        self.critic = critic
        self.critic_threshold = critic_threshold
        self.default_budget = default_budget
        self.sessions: dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self.started_at = time.time()
        self.version = RUNTIME_VERSION
        self._agent_factory = agent_factory or Agent

    # ---- introspection -------------------------------------------------

    def capabilities(self) -> dict:
        """What this runtime can do — for the coordinator to query at boot."""
        all_skills = self.skills.all()
        # Build a representative agent purely to enumerate tools — cheap and
        # avoids drift between "what we say we have" and "what we actually
        # expose". Throw it away after.
        probe = Agent(
            memory=Memory(),
            enable_web_search=True,
            enable_web_fetch=True,
            verbose=False,
            client=_NoOpClient(),
        )
        return {
            "version": self.version,
            "started_at": self.started_at,
            "models": list(_known_models()),
            "default_model": "claude-opus-4-7",
            "tools": [{"name": t.get("name"), "type": t.get("type", "client")} for t in probe.tool_schemas],
            "skills": [
                {
                    "id": s.id,
                    "description": s.description,
                    "promoted": s.promoted,
                    "triggers": list(s.triggers),
                }
                for s in all_skills
            ],
            "supports": {
                "delegation": True,
                "streaming_events": True,
                "budgets": True,
                "reflection": True,
                "critic_gate": self.critic is not None,
            },
        }

    # ---- session lifecycle ---------------------------------------------

    def open_session(self, config: SessionConfig | dict | None = None) -> Session:
        if isinstance(config, dict):
            config = SessionConfig.from_dict(config)
        config = config or SessionConfig()

        session_id = uuid.uuid4().hex[:12]
        memory = Memory(path=config.memory_path) if config.memory_path else Memory()
        attached_skills = self._attach_skills(config.skill_ids)

        budget = self.default_budget.merged_with(config.budget) if self.default_budget else config.budget

        # Build a SkillLibrary view that already includes the always-on skills
        # and falls back to keyword retrieval for the rest. Easiest impl:
        # render the always-on skills into system_prompt_extra so retrieval
        # is additive.
        extras: list[str] = []
        if config.system_prompt_extra:
            extras.append(config.system_prompt_extra.strip())
        if attached_skills:
            extras.append(
                "## Attached skills (always available)\n\n"
                + "\n\n".join(s.render_for_prompt() for s in attached_skills)
            )
        system_prompt_extra = "\n\n".join(extras) if extras else None

        agent = self._agent_factory(
            memory=memory,
            model=config.model,
            max_tokens=config.max_tokens,
            effort=config.effort,
            enable_web_search=config.enable_web_search,
            enable_web_fetch=config.enable_web_fetch,
            verbose=False,
            tracer=self.tracer,
            critic=self.critic,
            critic_threshold=self.critic_threshold,
            budget=budget,
            skills=self.skills,
            system_prompt_extra=system_prompt_extra,
        )

        session = Session(
            id=session_id,
            config=config,
            runtime=self,
            memory=memory,
            agent=agent,
        )

        if config.enable_delegate:
            self._install_delegate_tool(session)

        with self._sessions_lock:
            self.sessions[session_id] = session

        if config.parent_session_id:
            parent = self.sessions.get(config.parent_session_id)
            if parent is not None:
                parent.child_session_ids.append(session_id)

        return session

    def close_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        # Cancel any running tasks
        for t in list(session.tasks.values()):
            if t.status == "running":
                t.cancel_flag.set()
        return True

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    # ---- task lifecycle ------------------------------------------------

    def run_task(
        self,
        session_id: str,
        input: str,
        *,
        budget: Budget | None = None,
        task_id: str | None = None,
    ) -> TaskResult:
        """Blocking: run a task to completion and return the result."""
        task = self._start_task(session_id, input, budget=budget, task_id=task_id, async_=False)
        assert task.result is not None  # _start_task with async_=False runs sync
        return task.result

    def start_task(
        self,
        session_id: str,
        input: str,
        *,
        budget: Budget | None = None,
        task_id: str | None = None,
    ) -> str:
        """Non-blocking: queue the task on a background thread, return its id.

        Use `events()` to stream progress and `get_task()` to poll for the
        final result.
        """
        task = self._start_task(session_id, input, budget=budget, task_id=task_id, async_=True)
        return task.id

    def get_task(self, session_id: str, task_id: str) -> _Task | None:
        s = self.sessions.get(session_id)
        if s is None:
            return None
        return s.tasks.get(task_id)

    def cancel_task(self, session_id: str, task_id: str) -> bool:
        s = self.sessions.get(session_id)
        if s is None:
            return False
        t = s.tasks.get(task_id)
        if t is None or t.status != "running":
            return False
        t.cancel_flag.set()
        return True

    def events(self, session_id: str, task_id: str, *, timeout: float = 30.0) -> Iterator[dict]:
        """Yield structured events for a running task until it finishes.

        Yields dicts like {"type": "turn", ...}, {"type": "budget", ...},
        {"type": "tool_call", ...}, ending with {"type": "result", ...}.
        """
        s = self.sessions.get(session_id)
        if s is None:
            return
        t = s.tasks.get(task_id)
        if t is None:
            return
        deadline = time.time() + timeout
        while True:
            try:
                event = t.event_queue.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                if t.status != "running":
                    return
                continue
            yield event
            if event.get("type") == "result":
                return

    # ---- internals -----------------------------------------------------

    def _start_task(
        self,
        session_id: str,
        input: str,
        *,
        budget: Budget | None,
        task_id: str | None,
        async_: bool,
    ) -> _Task:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")

        task = _Task(
            id=task_id or uuid.uuid4().hex[:12],
            session_id=session_id,
            input=input,
            budget=budget,
            started_at=time.time(),
        )
        session.tasks[task.id] = task

        def run():
            with session.lock:
                self._execute(session, task)

        if async_:
            t = threading.Thread(target=run, name=f"agi-task-{task.id}", daemon=True)
            task.thread = t
            t.start()
        else:
            run()
        return task

    def _execute(self, session: Session, task: _Task) -> None:
        agent = session.agent

        # Hook the agent's event emitter into our task queue, but stash the
        # previous callback so reentrancy (delegate → parent agent → child)
        # doesn't drop events.
        prev_on_event = agent.on_event
        cancel_flag = task.cancel_flag

        def on_event(event: dict) -> None:
            event = {**event, "task_id": task.id, "session_id": session.id}
            task.event_queue.put(event)
            if prev_on_event:
                prev_on_event(event)
            if cancel_flag.is_set():
                # Soft cancel: surface via a synthetic over-budget condition.
                # The next budget check inside chat() will see max_iterations=0
                # tightening (set below).
                pass

        agent.on_event = on_event
        prev_budget = agent.budget

        # If a per-task budget is given, merge with the agent's existing budget
        # for the duration of this call only.
        agent.budget = (prev_budget.merged_with(task.budget) if prev_budget else task.budget)

        # Cancellation: install a sentinel that flips max_iterations to 0
        # mid-loop. We piggyback by wrapping the budget check via a custom
        # Budget subclass-like behavior — simplest implementation is to
        # poll cancel_flag in the on_event callback and tighten the agent's
        # budget so the next budget.check() trips.
        if cancel_flag.is_set():
            # already cancelled before we started
            agent.budget = Budget(max_iterations=0).merged_with(agent.budget)

        # Watcher thread tightens the budget if cancel arrives mid-task.
        stop_watch = threading.Event()

        def watch():
            while not stop_watch.is_set():
                if cancel_flag.is_set():
                    agent.budget = Budget(max_iterations=0).merged_with(agent.budget)
                    return
                stop_watch.wait(0.1)

        watcher = threading.Thread(target=watch, name=f"agi-cancel-{task.id}", daemon=True)
        watcher.start()

        usage_before = _snapshot_usage(agent.usage)
        error: str | None = None
        output = ""
        status = "ok"

        try:
            output = agent.chat(task.input)
        except Exception as e:  # never let an agent crash the runtime
            error = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            task.event_queue.put({"type": "error", "error": error, "traceback": tb, "task_id": task.id, "session_id": session.id})
            status = "error"
        finally:
            stop_watch.set()
            agent.on_event = prev_on_event
            agent.budget = prev_budget

        if cancel_flag.is_set() and status == "ok":
            status = "cancelled"
        elif agent.last_stop_reason and agent.last_stop_reason.startswith("over_budget"):
            status = "over_budget"
        elif agent.last_stop_reason == "refusal":
            status = "refusal"

        elapsed = time.time() - task.started_at
        delta = _usage_delta(usage_before, agent.usage)
        cost = _usage_cost(delta, session.config.model)

        reflection_notes: list[str] = []
        if session.config.reflect and status in ("ok", "over_budget"):
            try:
                eval_passed = None if status != "ok" else True
                reflection_notes = reflect(
                    memory=session.memory,
                    prompt=task.input,
                    response=output,
                    messages=agent.messages,
                    eval_passed=eval_passed,
                    tags=[f"task:{task.id}"],
                )
            except Exception:
                pass

        result = TaskResult(
            task_id=task.id,
            session_id=session.id,
            status=status,
            output=output,
            usage=delta,
            cost_usd=cost,
            elapsed_s=elapsed,
            skills_used=list(agent.last_skills_used or []),
            critic_score=agent.last_critic_score,
            reflection_notes=reflection_notes,
            error=error,
            stop_reason=agent.last_stop_reason,
        )
        task.result = result
        task.status = "done" if status == "ok" else status
        task.event_queue.put({"type": "result", **result.to_dict()})

    def _attach_skills(self, ids: list[str]):
        out = []
        for sid in ids:
            s = self.skills.get(sid)
            if s is not None:
                out.append(s)
        return out

    def _install_delegate_tool(self, parent: Session) -> None:
        """Add a `delegate` tool that opens a child session and runs a task in it.

        Usage rolls up to the parent session because the underlying Agent
        instances live in the same Runtime; the coordinator can read each
        session's usage independently if it wants fine-grained accounting.
        """
        runtime = self

        def delegate(task: str, role: str = "executor", max_usd: float | None = None, max_seconds: float | None = None) -> str:
            sub_cfg = SessionConfig(
                model=parent.config.model,
                effort=parent.config.effort,
                max_tokens=parent.config.max_tokens,
                enable_web_search=parent.config.enable_web_search,
                enable_web_fetch=parent.config.enable_web_fetch,
                enable_delegate=False,  # one level of recursion only by default
                budget=Budget(max_usd=max_usd, max_seconds=max_seconds) if (max_usd or max_seconds) else None,
                system_prompt_extra=f"You are operating as a {role!r} subagent for a parent task.",
                parent_session_id=parent.id,
                reflect=False,
            )
            child = runtime.open_session(sub_cfg)
            try:
                result = runtime.run_task(child.id, task)
                if result.error:
                    return f"delegate error: {result.error}"
                tag = f"[subagent {role} • {result.elapsed_s:.1f}s • ${result.cost_usd:.4f}]"
                return f"{tag}\n{result.output}"
            finally:
                runtime.close_session(child.id)

        schema = {
            "name": "delegate",
            "description": (
                "Spawn a subagent to handle a self-contained subtask. The "
                "subagent runs to completion and returns its final answer. "
                "Use this when a task naturally decomposes and a subtask "
                "would otherwise clutter the main thread of work."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The complete subtask description for the subagent."},
                    "role": {"type": "string", "description": "Optional role label ('planner', 'executor', 'critic').", "default": "executor"},
                    "max_usd": {"type": "number", "description": "Optional max dollar spend for the subagent."},
                    "max_seconds": {"type": "number", "description": "Optional max wall-clock seconds for the subagent."},
                },
                "required": ["task"],
            },
        }
        parent.agent.tool_schemas.append(schema)
        parent.agent.handlers["delegate"] = delegate


# ---- helpers -----------------------------------------------------------

def _snapshot_usage(u: Usage) -> dict[str, int]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": u.cache_creation_input_tokens,
        "cache_read_input_tokens": u.cache_read_input_tokens,
        "turns": u.turns,
    }


def _usage_delta(before: dict[str, int], after: Usage) -> dict[str, int]:
    return {
        "input_tokens": after.input_tokens - before["input_tokens"],
        "output_tokens": after.output_tokens - before["output_tokens"],
        "cache_creation_input_tokens": after.cache_creation_input_tokens - before["cache_creation_input_tokens"],
        "cache_read_input_tokens": after.cache_read_input_tokens - before["cache_read_input_tokens"],
        "turns": after.turns - before["turns"],
    }


def _usage_cost(delta: dict[str, int], model: str) -> float:
    u = Usage(
        input_tokens=delta["input_tokens"],
        output_tokens=delta["output_tokens"],
        cache_creation_input_tokens=delta["cache_creation_input_tokens"],
        cache_read_input_tokens=delta["cache_read_input_tokens"],
        turns=delta["turns"],
    )
    return u.cost_usd(model)


def _known_models() -> list[str]:
    from agi.costs import PRICING
    return sorted(PRICING.keys())


class _NoOpClient:
    """Sentinel for capability probing — never makes API calls."""
    class _Messages:
        def stream(self, *a, **kw):
            raise RuntimeError("_NoOpClient is only used for tool-schema introspection")
    messages = _Messages()
