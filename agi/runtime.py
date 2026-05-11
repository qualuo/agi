"""Runtime engine — the agent as a callable primitive.

The Runtime turns an Agent into a service a coordination engine can drive:
submit a Job, get a JobResult, with budgets enforced, sessions snapshottable
and resumable, and capabilities introspectable so a router can pick the right
runtime for the work.

This is the "runtime engine" the architecture document refers to. The Agent
is the *executor*; the Runtime is the *contract* a coordinator depends on.

Why a separate layer? Three reasons:

1. **Separation of concerns.** Agent owns the loop. Runtime owns the SLA
   (budget, idempotency, audit trail, snapshot/resume). Mixing them makes
   both worse.
2. **Substitutability.** A coordinator written against the Runtime/protocol
   surface can swap in a different reasoning core (Opus, a small base model
   from `learner/`, a remote service) without changing the coordinator.
3. **Multi-process / multi-host readiness.** The protocol is JSON-shaped so
   the coordinator and the runtime can live in different processes — even
   different machines — once we wrap a thin RPC layer around `submit`.
   That's the path to a fleet of runtimes coordinated by one engine.
"""
from __future__ import annotations

import copy
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from agi.agent import Agent
from agi.costs import PRICING, Usage
from agi.memory import Memory
from agi.protocol import (
    Job,
    JobResult,
    JobStatus,
    ProgressEvent,
    RuntimeCapabilities,
    ToolDescriptor,
)


_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": "Read a UTF-8 text file.",
    "write_file": "Write content to a file.",
    "list_dir": "List entries in a directory.",
    "run_bash": "Run a bash command and capture stdout/stderr.",
    "save_memory": "Save a note to long-term memory.",
    "search_memory": "Keyword-search long-term memory.",
    "recent_memory": "Return the most recent memory notes.",
    "web_search": "Server-side web search.",
    "web_fetch": "Server-side URL fetch.",
}

_SERVER_SIDE = {"web_search", "web_fetch"}


class Runtime:
    """Wraps an Agent in the coordinator-facing protocol.

    A Runtime owns N sessions. A session is just a (snapshot of) conversation
    history + cumulative usage. A coordinator can:

    - `submit(job)` — run a job to completion within a budget.
    - `capabilities()` — introspect what this runtime can do.
    - `snapshot(session_id)` / `resume(snapshot)` — branch a session, e.g. to
      explore two continuations and pick the best.
    - `cancel(job_id)` — best-effort cancellation (set a flag the in-flight
      loop checks at the next turn boundary).

    The Runtime never raises out of `submit`; failures land in `JobResult.error`.
    Coordinators get a uniform shape for every outcome, which is what makes
    routing logic actually tractable.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Agent] | None = None,
        runtime_id: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.runtime_id = runtime_id or f"rt-{uuid.uuid4().hex[:8]}"
        self.tags = list(tags or [])
        self._agent_factory = agent_factory or (lambda: Agent(verbose=False))
        self._sessions: dict[str, Agent] = {}
        self._results: dict[str, JobResult] = {}
        self._cancelled: set[str] = set()
        self._subscribers: list[Callable[[ProgressEvent], None]] = []
        # Materialise one agent up front so capabilities() works immediately.
        self._template_agent = self._agent_factory()

    # ----- coordinator-facing API --------------------------------------------

    def capabilities(self) -> RuntimeCapabilities:
        a = self._template_agent
        in_rate, out_rate = PRICING.get(a.model, (0.0, 0.0))
        tools: list[ToolDescriptor] = []
        for s in a.tool_schemas:
            name = s.get("name", "?")
            tools.append(
                ToolDescriptor(
                    name=name,
                    description=_TOOL_DESCRIPTIONS.get(name, s.get("description", "")),
                    server_side=name in _SERVER_SIDE,
                )
            )
        return RuntimeCapabilities(
            runtime_id=self.runtime_id,
            model=a.model,
            cost_per_1m_input_usd=in_rate,
            cost_per_1m_output_usd=out_rate,
            tools=tools,
            has_critic=a.critic is not None,
            has_memory=isinstance(a.memory, Memory),
            tags=list(self.tags),
        )

    def submit(self, job: Job) -> JobResult:
        # Idempotent re-submission: prior result wins.
        if job.job_id in self._results:
            return self._results[job.job_id]

        agent = self._get_or_create_session(job.session_id)
        session_id = self._session_id_for(agent)
        prompt = self._render_prompt(job)

        budget = job.max_cost_usd
        # Track Agent.usage delta so we can compute cost-of-this-job.
        baseline_input = agent.usage.input_tokens
        baseline_output = agent.usage.output_tokens
        baseline_cw = agent.usage.cache_creation_input_tokens
        baseline_cr = agent.usage.cache_read_input_tokens

        def should_continue(turn_usage: Usage) -> bool:
            spent = turn_usage.cost_usd(agent.model)
            self._emit(ProgressEvent(
                job_id=job.job_id,
                kind="budget_check",
                payload={"spent_usd": spent, "budget_usd": budget},
            ))
            if job.job_id in self._cancelled:
                return False
            return spent < budget

        self._emit(ProgressEvent(job_id=job.job_id, kind="job_started",
                                 payload={"session_id": session_id}))

        t0 = time.time()
        status = JobStatus.SUCCEEDED
        error: str | None = None
        output = ""
        trace_id: str | None = None
        critic_score: float | None = None
        try:
            output = agent.chat(
                prompt,
                max_iterations=job.max_iterations,
                should_continue=should_continue,
            )
            critic_score = agent.last_critic_score
            if job.job_id in self._cancelled:
                status = JobStatus.CANCELLED
            elif agent.last_aborted:
                # The only abort path today is budget; treat as such.
                status = JobStatus.BUDGET_EXCEEDED
                error = f"aborted after exceeding budget ${budget:.4f}"
        except Exception as e:
            status = JobStatus.FAILED
            error = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0

        # Cost is the delta of agent.usage attributable to this job.
        delta = Usage(
            input_tokens=agent.usage.input_tokens - baseline_input,
            output_tokens=agent.usage.output_tokens - baseline_output,
            cache_creation_input_tokens=agent.usage.cache_creation_input_tokens - baseline_cw,
            cache_read_input_tokens=agent.usage.cache_read_input_tokens - baseline_cr,
        )
        cost = delta.cost_usd(agent.model)

        # Best-effort trace id surface — TraceLogger writes a Trace and could
        # be wired to expose its last id; for now we leave it None unless the
        # Agent's tracer has been extended to remember it.
        if getattr(agent, "tracer", None) is not None:
            trace_id = getattr(agent.tracer, "last_trace_id", None)

        result = JobResult(
            job_id=job.job_id,
            status=status,
            output=output,
            error=error,
            cost_usd=cost,
            budget_remaining_usd=max(0.0, budget - cost),
            elapsed_seconds=elapsed,
            iterations=getattr(agent, "last_iterations", 0),
            trace_id=trace_id,
            critic_score=critic_score,
            session_id=session_id,
        )
        self._results[job.job_id] = result
        self._emit(ProgressEvent(job_id=job.job_id, kind="job_finished",
                                 payload={"status": status.value, "cost_usd": cost}))
        return result

    def cancel(self, job_id: str) -> None:
        """Best-effort cancellation; takes effect at the next turn boundary."""
        self._cancelled.add(job_id)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Serialize a session's conversation + usage so a coordinator can
        fork or persist it. Note: messages are deep-copied; the SDK content
        blocks may not all be JSON-serializable (e.g. thinking blocks), so
        this is a same-process snapshot, not a wire-format trace."""
        if session_id not in self._sessions:
            raise KeyError(f"unknown session {session_id!r}")
        agent = self._sessions[session_id]
        return {
            "session_id": session_id,
            "messages": copy.deepcopy(agent.messages),
            "usage": {
                "input_tokens": agent.usage.input_tokens,
                "output_tokens": agent.usage.output_tokens,
                "cache_creation_input_tokens": agent.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": agent.usage.cache_read_input_tokens,
                "turns": agent.usage.turns,
            },
            "model": agent.model,
        }

    def resume(self, snapshot: dict[str, Any]) -> str:
        """Create a new session populated from a snapshot. Returns the new
        session id so the coordinator can refer to it."""
        agent = self._agent_factory()
        agent.messages = copy.deepcopy(snapshot.get("messages", []))
        u = snapshot.get("usage", {})
        agent.usage.input_tokens = u.get("input_tokens", 0)
        agent.usage.output_tokens = u.get("output_tokens", 0)
        agent.usage.cache_creation_input_tokens = u.get("cache_creation_input_tokens", 0)
        agent.usage.cache_read_input_tokens = u.get("cache_read_input_tokens", 0)
        agent.usage.turns = u.get("turns", 0)
        new_id = f"sess-{uuid.uuid4().hex[:10]}"
        self._sessions[new_id] = agent
        return new_id

    def subscribe(self, fn: Callable[[ProgressEvent], None]) -> None:
        """Register a callback for ProgressEvents. Coordinators use this to
        watch in-flight progress without polling."""
        self._subscribers.append(fn)

    # ----- internals ---------------------------------------------------------

    def _get_or_create_session(self, session_id: str | None) -> Agent:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        agent = self._agent_factory()
        sid = session_id or f"sess-{uuid.uuid4().hex[:10]}"
        self._sessions[sid] = agent
        return agent

    def _session_id_for(self, agent: Agent) -> str:
        for sid, a in self._sessions.items():
            if a is agent:
                return sid
        # Should never happen — defensive.
        sid = f"sess-{uuid.uuid4().hex[:10]}"
        self._sessions[sid] = agent
        return sid

    def _render_prompt(self, job: Job) -> str:
        if not job.output_contract:
            return job.prompt
        return (
            f"{job.prompt}\n\n"
            f"---\nReturn your answer in the following shape:\n{job.output_contract}"
        )

    def _emit(self, event: ProgressEvent) -> None:
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception:
                # A bad subscriber must not take down a job.
                pass


def opus_runtime(tags: list[str] | None = None, **agent_kwargs: Any) -> Runtime:
    """Convenience factory for the default Opus 4.7 runtime."""
    def factory() -> Agent:
        return Agent(verbose=False, **agent_kwargs)
    return Runtime(agent_factory=factory, tags=tags or ["frozen", "frontier"])


def haiku_runtime(tags: list[str] | None = None, **agent_kwargs: Any) -> Runtime:
    """Convenience factory for a cheap Haiku 4.5 runtime — use for fanout."""
    def factory() -> Agent:
        return Agent(model="claude-haiku-4-5", verbose=False, **agent_kwargs)
    return Runtime(agent_factory=factory, tags=tags or ["fast", "cheap"])
