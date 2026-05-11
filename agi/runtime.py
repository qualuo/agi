"""Runtime engine.

The Agent class is a single conversation loop. The Runtime is what a
*coordination engine* talks to: a process-level multi-tenant runtime
that hosts many agent sessions, advertises capabilities, accounts for
cost, supports idempotent retries, cancellation, event subscription,
and persistent skills.

Why this layer exists, separate from Agent:

  Coordination engines (planners, schedulers, multi-agent orchestrators,
  Temporal-style workflows) don't want a Python class. They want a
  typed, addressable, observable worker. Sessions are the addressable
  unit; capabilities tell the planner what work this runtime can take;
  events let the planner react in real time; usage tells the planner
  what each call cost so it can budget; idempotency keys let it retry
  safely; cancellation lets it abort lost causes.

  This module is also the transport-agnostic core for `agi.server`,
  which exposes the same surface over HTTP.

Sessions are in-memory by default. A coordination engine that needs
durability across runtime restarts should pin sessions to a process
and reconnect, or layer state externally — that's intentionally not
this module's job in v1.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

from agi.agent import Agent
from agi.events import Event, EventBus
from agi.memory import Memory
from agi.skills import Skill, SkillLibrary
from agi.tools import make_tools

try:
    from learner.traces import TraceLogger
except ImportError:
    TraceLogger = None  # type: ignore


# Version of the runtime surface a coordination engine binds against.
# Increment on breaking changes to event names or run-result shape.
RUNTIME_API_VERSION = "1"


@dataclass
class RunResult:
    """Returned from `Runtime.run`. Stable, JSON-serializable."""
    run_id: str
    session_id: str
    output_text: str
    stop_reason: str | None
    cancelled: bool
    usage: dict[str, int]
    cost_usd: float
    critic_score: float | None
    elapsed_s: float
    idempotency_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionInfo:
    session_id: str
    created_ts: float
    last_used_ts: float
    runs: int
    cumulative_usage: dict[str, int]
    cumulative_cost_usd: float
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class _Session:
    """One agent + its event ring buffer + a per-session lock.

    The lock is what makes the runtime safe for concurrent `run`
    requests on the same session: each run takes the lock, the previous
    run finishes, the next begins. A coordination engine that wants
    parallelism creates multiple sessions; a single session is a single
    coherent conversation.
    """

    def __init__(
        self,
        session_id: str,
        agent: Agent,
        metadata: dict[str, Any],
        event_buffer: int = 1000,
    ) -> None:
        self.id = session_id
        self.agent = agent
        self.metadata = metadata
        self.created_ts = time.time()
        self.last_used_ts = self.created_ts
        self.runs = 0
        self.lock = threading.Lock()
        # Bounded event log so a forgotten subscriber can't OOM the process.
        self.events: deque[Event] = deque(maxlen=event_buffer)
        self._subscribers: list[queue.Queue[Event]] = []
        agent.events.subscribe(self._on_event)
        # Replay state for idempotency: idempotency_key -> RunResult
        self.idempotent_results: dict[str, RunResult] = {}

    def _on_event(self, event: Event) -> None:
        # Stamp session id on every event for downstream multiplexing.
        if event.session_id is None:
            event.session_id = self.id
        self.events.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow consumer; drop

    def subscribe(self, replay: bool = False, max_queue: int = 4096) -> "tuple[queue.Queue[Event], Callable[[], None]]":
        q: queue.Queue[Event] = queue.Queue(maxsize=max_queue)
        if replay:
            for evt in list(self.events):
                try:
                    q.put_nowait(evt)
                except queue.Full:
                    break
        self._subscribers.append(q)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

        return q, unsubscribe

    def info(self) -> SessionInfo:
        usage = self.agent.usage
        return SessionInfo(
            session_id=self.id,
            created_ts=self.created_ts,
            last_used_ts=self.last_used_ts,
            runs=self.runs,
            cumulative_usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "turns": usage.turns,
            },
            cumulative_cost_usd=usage.cost_usd(self.agent.model),
            model=self.agent.model,
            metadata=dict(self.metadata),
        )


def default_capability_manifest(
    schemas: list[dict],
    skill_lib: SkillLibrary | None,
    model: str,
) -> dict[str, Any]:
    """The manifest a coordination engine fetches to decide whether to
    route a task here. Stable shape — additive changes only on minor
    version bumps."""
    skills = []
    if skill_lib is not None:
        for s in skill_lib.list():
            skills.append({"name": s.name, "description": s.description, "tags": s.tags})

    return {
        "runtime_api_version": RUNTIME_API_VERSION,
        "name": "agi-runtime",
        "description": "Claude Opus 4.7 agent harness exposed as a typed runtime.",
        "model": model,
        "modalities": ["text"],
        "tools": [{"name": s.get("name") or s.get("type"), "kind": s.get("type", "client")} for s in schemas],
        "skills": skills,
        "features": {
            "streaming_events": True,
            "cancellation": True,
            "idempotency": True,
            "persistent_memory": True,
            "skill_library": True,
            "critic_gate": True,
            "usage_telemetry": True,
        },
        "events": [
            "run.started", "run.finished",
            "turn.started", "turn.finished",
            "thinking.started", "thinking.delta",
            "text.started", "text.delta",
            "tool.requested", "tool.result", "server_tool.requested",
            "critic.scored", "skills.injected",
            "cancelled", "error",
        ],
    }


class Runtime:
    """Hosts sessions. Single-process, thread-safe.

    Typical use from a coordination engine (in-process):

        rt = Runtime()
        sid = rt.create_session()
        result = rt.run(sid, "summarize ./README.md")

    Cross-process: see agi.server.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        effort: str = "high",
        max_tokens: int = 16000,
        enable_web_search: bool = True,
        enable_web_fetch: bool = True,
        skills: SkillLibrary | None = None,
        tracer=None,
        critic=None,
        critic_threshold: float = 0.5,
        memory_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.enable_web_search = enable_web_search
        self.enable_web_fetch = enable_web_fetch
        self.skills = skills
        self.tracer = tracer
        self.critic = critic
        self.critic_threshold = critic_threshold
        self.memory_root = (
            os.fspath(memory_root) if memory_root else None
        )

        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self.started_ts = time.time()
        self.total_runs = 0

    # ---- session lifecycle ----

    def create_session(
        self,
        session_id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        memory: Memory | None = None,
        verbose: bool = False,
    ) -> str:
        sid = session_id or uuid.uuid4().hex[:16]
        with self._lock:
            if sid in self._sessions:
                raise ValueError(f"session {sid!r} already exists")
            if memory is None:
                if self.memory_root:
                    mem_path = os.path.join(self.memory_root, f"{sid}.jsonl")
                    memory = Memory(path=mem_path)
                else:
                    memory = Memory()
            agent = Agent(
                memory=memory,
                model=self.model,
                max_tokens=self.max_tokens,
                effort=self.effort,
                enable_web_search=self.enable_web_search,
                enable_web_fetch=self.enable_web_fetch,
                verbose=verbose,
                tracer=self.tracer,
                critic=self.critic,
                critic_threshold=self.critic_threshold,
                skills=self.skills,
                session_id=sid,
            )
            self._sessions[sid] = _Session(sid, agent, metadata or {})
        return sid

    def destroy_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[SessionInfo]:
        with self._lock:
            return [s.info() for s in self._sessions.values()]

    def get_session(self, session_id: str) -> SessionInfo:
        return self._require(session_id).info()

    def _require(self, session_id: str) -> _Session:
        with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"unknown session {session_id!r}")
        return sess

    # ---- execution ----

    def run(
        self,
        session_id: str,
        prompt: str,
        *,
        max_iterations: int = 25,
        idempotency_key: str | None = None,
        run_id: str | None = None,
    ) -> RunResult:
        sess = self._require(session_id)
        if idempotency_key and idempotency_key in sess.idempotent_results:
            return sess.idempotent_results[idempotency_key]

        rid = run_id or uuid.uuid4().hex[:16]
        with sess.lock:
            # Re-check idempotency under the lock to close the race.
            if idempotency_key and idempotency_key in sess.idempotent_results:
                return sess.idempotent_results[idempotency_key]

            before = _usage_snapshot(sess.agent.usage)
            t0 = time.time()
            try:
                text = sess.agent.chat(prompt, max_iterations=max_iterations, run_id=rid)
            except Exception as e:
                # Surface a structured error; coordinator can decide on retry.
                sess.agent.events.publish(
                    Event(kind="error", data={"message": f"{type(e).__name__}: {e}"}, session_id=session_id, run_id=rid)
                )
                raise
            elapsed = time.time() - t0
            sess.runs += 1
            sess.last_used_ts = time.time()
            self.total_runs += 1
            after = _usage_snapshot(sess.agent.usage)
            turn_usage = _usage_diff(before, after)
            cost = _cost_for(turn_usage, sess.agent.model)

            cancelled = False
            stop_reason: str | None = None
            for evt in reversed(sess.events):
                if evt.kind == "run.finished" and evt.run_id == rid:
                    cancelled = bool(evt.data.get("cancelled"))
                    stop_reason = evt.data.get("stop_reason")
                    break

            result = RunResult(
                run_id=rid,
                session_id=sess.id,
                output_text=text,
                stop_reason=stop_reason,
                cancelled=cancelled,
                usage=turn_usage,
                cost_usd=cost,
                critic_score=sess.agent.last_critic_score,
                elapsed_s=elapsed,
                idempotency_key=idempotency_key,
            )
            if idempotency_key:
                sess.idempotent_results[idempotency_key] = result
            return result

    def cancel(self, session_id: str) -> None:
        sess = self._require(session_id)
        sess.agent.cancel()

    # ---- events ----

    def subscribe(
        self, session_id: str, *, replay: bool = False
    ) -> "tuple[queue.Queue[Event], Callable[[], None]]":
        return self._require(session_id).subscribe(replay=replay)

    def recent_events(self, session_id: str, *, limit: int = 100) -> list[Event]:
        sess = self._require(session_id)
        return list(sess.events)[-limit:]

    # ---- capability discovery ----

    def capabilities(self) -> dict[str, Any]:
        # Tool schemas are static across sessions in v1; build them from a
        # throwaway in-memory store so capability discovery has no side
        # effects on the user's persistent memory file.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            schemas, _ = make_tools(Memory(path=os.path.join(tmp, "m.jsonl")))
        if self.enable_web_search:
            schemas.append({"type": "web_search_20260209", "name": "web_search"})
        if self.enable_web_fetch:
            schemas.append({"type": "web_fetch_20260209", "name": "web_fetch"})
        return default_capability_manifest(schemas, self.skills, self.model)

    def health(self) -> dict[str, Any]:
        with self._lock:
            n_sessions = len(self._sessions)
        return {
            "status": "ok",
            "uptime_s": time.time() - self.started_ts,
            "sessions": n_sessions,
            "total_runs": self.total_runs,
            "runtime_api_version": RUNTIME_API_VERSION,
        }

    def aggregate_usage(self) -> dict[str, Any]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "turns": 0,
            "runs": 0,
            "cost_usd": 0.0,
        }
        with self._lock:
            sessions = list(self._sessions.values())
        for sess in sessions:
            u = sess.agent.usage
            totals["input_tokens"] += u.input_tokens
            totals["output_tokens"] += u.output_tokens
            totals["cache_creation_input_tokens"] += u.cache_creation_input_tokens
            totals["cache_read_input_tokens"] += u.cache_read_input_tokens
            totals["turns"] += u.turns
            totals["runs"] += sess.runs
            totals["cost_usd"] += u.cost_usd(sess.agent.model)
        return totals

    # ---- skill management ----

    def upsert_skill(self, name: str, description: str, body: str, tags: Iterable[str] = ()) -> dict[str, Any]:
        if self.skills is None:
            raise RuntimeError("no skill library is wired to this runtime")
        skill = Skill(name=name, description=description, body=body, tags=list(tags))
        path = self.skills.save(skill)
        return {"name": skill.name, "path": str(path)}


# ---- helpers ----

def _usage_snapshot(u) -> dict[str, int]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": u.cache_creation_input_tokens,
        "cache_read_input_tokens": u.cache_read_input_tokens,
        "turns": u.turns,
    }


def _usage_diff(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: after[k] - before[k] for k in after}


def _cost_for(usage: dict[str, int], model: str) -> float:
    from agi.costs import Usage
    u = Usage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        cache_creation_input_tokens=usage["cache_creation_input_tokens"],
        cache_read_input_tokens=usage["cache_read_input_tokens"],
    )
    return u.cost_usd(model)
