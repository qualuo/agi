"""Runtime engine — a programmatic façade a coordination engine drives.

The agent loop is interactive by default (REPL / one-shot prompt). A
coordination engine (the parent, e.g. a planner, an orchestration framework,
or a service that needs to mix agent work with deterministic steps) wants
something different:

  - Structured inputs:  Task descriptors with goal, constraints, budget,
    tool whitelist, allowed delegation depth, deadline.
  - Structured outputs: Result objects with status, final text, usage,
    cost, capabilities used, and any error info — JSON-serializable.
  - Streaming events:   So a long task can report partial progress.
  - Sessions:           Spawn, address by id, resume, cancel.
  - Introspection:      Ask the runtime what its agents can do before
                        routing work to them.

`Runtime` is what implements that shape. It deliberately stays thin — it
does NOT try to be a planner, scheduler, or queue. Those are coordination-
engine concerns. The runtime exposes one agent (or many sessions) as a
callable resource.

Design notes:

- Sessions hold an Agent + memory + tracer. They live in-process; durable
  state already lives in `~/.agi/memory.jsonl` etc., so processes can come
  and go without losing the long-term knowledge layer.
- Budget enforcement is best-effort: token caps are checked *after* each
  turn (the streaming SDK doesn't expose mid-turn token totals), so a
  single oversize turn can overshoot. The check is still useful: it
  prevents an agent from looping past its budget for many turns.
- Tool whitelist filters at session-init time. Tools the coordinator
  forbids are simply not exposed to the model.
- Delegation: each session knows its remaining "delegate depth". A
  delegated subtask spawns a new ephemeral session with depth - 1, so
  recursion is bounded. depth=0 means no further delegation allowed.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from agi.agent import Agent
from agi.costs import Usage
from agi.memory import Memory

try:
    from learner.skills import SkillLibrary
except ImportError:  # learner is optional
    SkillLibrary = None  # type: ignore

try:
    from learner.traces import TraceLogger
except ImportError:
    TraceLogger = None  # type: ignore


# ----- Task / Result / Event types: stable wire format for coordinators -----


@dataclass
class Task:
    """A unit of work for the runtime engine.

    `goal` is the natural-language instruction. Everything else is optional
    structure that lets a coordination engine constrain or contextualize it.
    """

    goal: str
    context: str = ""                           # extra system-prompt content
    max_tokens_budget: int | None = None        # input+output cap; soft enforcement
    max_iterations: int = 25                    # tool-use loop ceiling
    deadline_seconds: float | None = None       # wall-clock cap
    allowed_tools: list[str] | None = None      # whitelist; None = all
    delegate_depth: int = 0                     # 0 = no delegation; >0 enables subagents
    enable_tool_synthesis: bool = False
    reflect: bool = False
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """The outcome of running a Task."""

    task_id: str
    status: str  # "ok" | "error" | "budget_exceeded" | "deadline_exceeded" | "cancelled"
    output: str
    elapsed_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    iterations: int = 0
    capabilities_used: list[str] = field(default_factory=list)
    critic_score: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "output": self.output,
            "elapsed_seconds": self.elapsed_seconds,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_creation_input_tokens": self.cache_creation_input_tokens,
                "cache_read_input_tokens": self.cache_read_input_tokens,
            },
            "cost_usd": self.cost_usd,
            "iterations": self.iterations,
            "capabilities_used": self.capabilities_used,
            "critic_score": self.critic_score,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Event:
    """A streaming event a coordination engine can subscribe to.

    Kinds: "started", "tool_call", "text_delta", "iteration", "finished",
    "error". Payload depends on kind — coordinators should treat unknown
    keys as forward-compatible additions.
    """

    kind: str
    task_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


# ----- Session: one addressable agent instance -----


class Session:
    """An addressable agent instance.

    Coordinators that need conversational continuity hold onto a session id
    and call `run` repeatedly. Stateless callers can prefer `Runtime.execute`
    which creates a one-shot session.

    The session is responsible for budget/deadline checks; the underlying
    Agent doesn't know about them.
    """

    def __init__(
        self,
        session_id: str,
        agent: Agent,
        *,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.session_id = session_id
        self.agent = agent
        self.on_event = on_event
        self.created_at = time.time()
        self.cancelled = False

    def cancel(self) -> None:
        """Request cancellation. Checked between turns (not mid-turn)."""
        self.cancelled = True

    def capabilities(self) -> dict[str, Any]:
        caps = self.agent.capabilities()
        caps["session_id"] = self.session_id
        caps["created_at"] = self.created_at
        return caps

    def run(self, task: Task) -> Result:
        """Execute one Task against this session.

        Budget/deadline checks happen between turns. The Agent loop runs to
        completion or to a stop condition; the runtime then checks limits
        and returns the appropriate status.
        """
        t0 = time.time()
        baseline = _snapshot_usage(self.agent.usage)
        self._emit("started", task.task_id, goal=task.goal)

        # Temporarily restrict tools if the task whitelists them.
        restore = None
        if task.allowed_tools is not None:
            restore = self._restrict_tools(task.allowed_tools)

        try:
            try:
                output = self._run_with_limits(task)
                status = "ok"
                err = None
            except _BudgetExceeded as e:
                output = str(e)
                status = "budget_exceeded"
                err = None
            except _DeadlineExceeded as e:
                output = str(e)
                status = "deadline_exceeded"
                err = None
            except _Cancelled as e:
                output = str(e)
                status = "cancelled"
                err = None
            except Exception as e:
                output = ""
                status = "error"
                err = f"{type(e).__name__}: {e}"
        finally:
            if restore is not None:
                restore()

        elapsed = time.time() - t0
        turn_usage = _delta_usage(baseline, self.agent.usage)
        capabilities_used = self._capabilities_used(turn_usage.turns)
        result = Result(
            task_id=task.task_id,
            status=status,
            output=output,
            elapsed_seconds=elapsed,
            input_tokens=turn_usage.input_tokens,
            output_tokens=turn_usage.output_tokens,
            cache_creation_input_tokens=turn_usage.cache_creation_input_tokens,
            cache_read_input_tokens=turn_usage.cache_read_input_tokens,
            cost_usd=turn_usage.cost_usd(self.agent.model),
            iterations=turn_usage.turns,
            capabilities_used=capabilities_used,
            critic_score=self.agent.last_critic_score,
            error=err,
            metadata=dict(task.metadata),
        )
        self._emit("finished", task.task_id, status=status, cost_usd=result.cost_usd)
        return result

    def stream(self, task: Task) -> Iterator[Event]:
        """Yield events as the task runs. Implemented as a thin wrapper:
        captures emitted events into a queue we drain after each turn.

        For now this completes the task in the calling thread and yields
        the buffered events. Concurrent streaming requires a thread/queue
        and is intentionally deferred — coordinators that want true
        concurrency can run multiple Sessions.
        """
        buf: list[Event] = []
        prev = self.on_event
        self.on_event = buf.append
        try:
            result = self.run(task)
        finally:
            self.on_event = prev
        for ev in buf:
            yield ev
        yield Event(
            kind="result",
            task_id=task.task_id,
            payload=result.to_dict(),
        )

    # ---- internals ----

    def _emit(self, kind: str, task_id: str, **payload: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(Event(kind=kind, task_id=task_id, payload=payload))
        except Exception:
            # An exception in the subscriber must not poison the session.
            pass

    def _run_with_limits(self, task: Task) -> str:
        """Run the agent loop with between-turn budget/deadline/cancel checks.

        We re-implement the outer iteration loop here (instead of delegating
        to Agent.chat) so we can inspect state between turns. The inner work
        — streaming + tool dispatch — still lives in Agent for a single
        source of truth.
        """
        agent = self.agent
        deadline = (time.time() + task.deadline_seconds) if task.deadline_seconds else None
        agent.messages.append({"role": "user", "content": _compose_input(task)})
        last_text = ""
        turn_usage = Usage()

        for i in range(task.max_iterations):
            if self.cancelled:
                raise _Cancelled("session cancelled")
            if deadline is not None and time.time() > deadline:
                raise _DeadlineExceeded(f"deadline exceeded after {i} iterations")

            response = agent._stream_one()
            agent.messages.append({"role": "assistant", "content": response.content})
            agent.usage.add(response.usage)
            turn_usage.add(response.usage)
            self._emit(
                "iteration",
                task.task_id,
                iteration=i + 1,
                input_tokens=turn_usage.input_tokens,
                output_tokens=turn_usage.output_tokens,
            )

            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text
                elif block.type == "tool_use":
                    self._emit("tool_call", task.task_id, name=block.name)

            if task.max_tokens_budget is not None:
                spent = turn_usage.input_tokens + turn_usage.output_tokens
                if spent > task.max_tokens_budget:
                    raise _BudgetExceeded(
                        f"token budget {task.max_tokens_budget} exceeded ({spent} spent)"
                    )

            if response.stop_reason == "end_turn":
                break
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason == "tool_use":
                tool_results = agent._dispatch_tool_calls(response.content)
                if not tool_results:
                    break
                agent.messages.append({"role": "user", "content": tool_results})
                continue
            break  # max_tokens / refusal / context exceeded / etc.

        last_text, critic_score = agent._apply_critic_gate(_compose_input(task), last_text)
        agent.last_critic_score = critic_score
        if agent.reflect:
            agent._write_reflection(task.goal, last_text)
        if agent.tracer is not None:
            agent.tracer.log(
                model=agent.model,
                messages=agent.messages,
                final_text=last_text,
                usage={
                    "input_tokens": turn_usage.input_tokens,
                    "output_tokens": turn_usage.output_tokens,
                    "cache_creation_input_tokens": turn_usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": turn_usage.cache_read_input_tokens,
                },
                metadata=({"critic_score": critic_score} if critic_score is not None else {}),
            )
        return last_text

    def _restrict_tools(self, allowed: list[str]) -> Callable[[], None]:
        """Hide tools not in `allowed`; return a function to restore them."""
        allowed_set = set(allowed)
        saved_schemas = list(self.agent.tool_schemas)
        saved_handlers = dict(self.agent.handlers)
        self.agent.tool_schemas = [
            s for s in saved_schemas
            if s.get("name") in allowed_set or s.get("type", "").startswith("web_") and s.get("name") in allowed_set
        ]
        self.agent.handlers = {n: h for n, h in saved_handlers.items() if n in allowed_set}

        def restore() -> None:
            self.agent.tool_schemas = saved_schemas
            self.agent.handlers = saved_handlers

        return restore

    def _capabilities_used(self, iterations: int) -> list[str]:
        """Best-effort scan of the most-recent N turns for tool names used."""
        names: list[str] = []
        # Walk back from end to find recent tool_use blocks until we've covered
        # the iterations we just ran. Cheaper than re-walking the entire
        # history.
        budget = max(iterations * 2, 4)
        for m in reversed(self.agent.messages[-budget:]):
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                bname = getattr(b, "name", None) if not isinstance(b, dict) else b.get("name")
                btype = getattr(b, "type", None) if not isinstance(b, dict) else b.get("type")
                if btype == "tool_use" and bname and bname not in names:
                    names.append(bname)
        return names


# ----- Runtime: the top-level handle a coordinator holds onto -----


class Runtime:
    """A pool of sessions backed by a shared memory + skill library + tracer.

    Typical coordinator usage:

        runtime = Runtime()                       # uses defaults from disk
        result = runtime.execute(Task(goal=...))  # one-shot

        # or, for conversational continuity:
        session = runtime.spawn()
        r1 = session.run(Task(goal=...))
        r2 = session.run(Task(goal=...))          # remembers turn 1
        runtime.close(session.session_id)

    The runtime is intentionally process-local. Cross-process / cross-host
    coordination is the parent system's job — it can call this runtime from
    each worker.
    """

    def __init__(
        self,
        memory: Memory | None = None,
        skills=None,
        tracer=None,
        model: str = "claude-opus-4-7",
        verbose: bool = False,
    ) -> None:
        self.memory = memory or Memory()
        if skills is None and SkillLibrary is not None:
            skills = SkillLibrary()
        self.skills = skills
        self.tracer = tracer
        self.model = model
        self.verbose = verbose
        self._sessions: dict[str, Session] = {}

    # ----- capability introspection at the runtime level -----

    def describe(self) -> dict[str, Any]:
        """Return what this runtime offers. A coordinator polls this once,
        before deciding what kinds of work to send."""
        probe = Agent(
            memory=self.memory,
            model=self.model,
            verbose=False,
            skills=self.skills,
            delegate_runtime=self._make_delegate_callable(depth=1),
            enable_tool_synthesis=True,
        )
        caps = probe.capabilities()
        caps["active_sessions"] = list(self._sessions)
        return caps

    # ----- session lifecycle -----

    def spawn(
        self,
        *,
        delegate_depth: int = 1,
        enable_tool_synthesis: bool = False,
        reflect: bool = False,
        memory: Memory | None = None,
        critic=None,
        critic_threshold: float = 0.5,
        on_event: Callable[[Event], None] | None = None,
        system_prompt_extra: str = "",
    ) -> Session:
        session_id = uuid.uuid4().hex[:12]
        agent = Agent(
            memory=memory or self.memory,
            model=self.model,
            verbose=self.verbose,
            tracer=self.tracer,
            critic=critic,
            critic_threshold=critic_threshold,
            skills=self.skills,
            delegate_runtime=self._make_delegate_callable(depth=delegate_depth),
            enable_tool_synthesis=enable_tool_synthesis,
            reflect=reflect,
            system_prompt_extra=system_prompt_extra,
        )
        session = Session(session_id=session_id, agent=agent, on_event=on_event)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def close(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    # ----- one-shot execute (no session persistence) -----

    def execute(
        self,
        task: Task,
        *,
        on_event: Callable[[Event], None] | None = None,
        critic=None,
    ) -> Result:
        """Execute a single Task in an ephemeral session. The session is
        destroyed after the call returns. Use `spawn`+`run` for continuity."""
        session = self.spawn(
            delegate_depth=task.delegate_depth,
            enable_tool_synthesis=task.enable_tool_synthesis,
            reflect=task.reflect,
            on_event=on_event,
            critic=critic,
        )
        try:
            return session.run(task)
        finally:
            self.close(session.session_id)

    def stream(self, task: Task) -> Iterator[Event]:
        """One-shot streaming variant."""
        session = self.spawn(
            delegate_depth=task.delegate_depth,
            enable_tool_synthesis=task.enable_tool_synthesis,
            reflect=task.reflect,
        )
        try:
            yield from session.stream(task)
        finally:
            self.close(session.session_id)

    # ----- delegate plumbing -----

    def _make_delegate_callable(self, depth: int) -> Callable[[str], str] | None:
        """Return a delegate function bound to a max recursion depth.

        Returning None disables the `delegate` tool. depth=0 means no
        further delegation allowed; recursion is bounded by the task's
        delegate_depth at spawn time.
        """
        if depth <= 0:
            return None

        def _delegate(task_text: str) -> str:
            sub_task = Task(
                goal=task_text,
                delegate_depth=depth - 1,
                # Inherit nothing else by default — keep subagents bounded.
                max_iterations=15,
            )
            sub_result = self.execute(sub_task)
            if sub_result.status != "ok":
                return f"[subagent {sub_result.status}] {sub_result.output or sub_result.error or ''}"
            return sub_result.output

        return _delegate


# ----- internal helpers -----


def _compose_input(task: Task) -> str:
    if not task.context:
        return task.goal
    return f"{task.context}\n\n---\n\n{task.goal}"


def _snapshot_usage(u: Usage) -> Usage:
    return Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_creation_input_tokens=u.cache_creation_input_tokens,
        cache_read_input_tokens=u.cache_read_input_tokens,
        turns=u.turns,
    )


def _delta_usage(before: Usage, after: Usage) -> Usage:
    return Usage(
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
        cache_creation_input_tokens=after.cache_creation_input_tokens - before.cache_creation_input_tokens,
        cache_read_input_tokens=after.cache_read_input_tokens - before.cache_read_input_tokens,
        turns=after.turns - before.turns,
    )


class _BudgetExceeded(Exception):
    pass


class _DeadlineExceeded(Exception):
    pass


class _Cancelled(Exception):
    pass
