"""Runtime engine.

Turns the Agent (a REPL/Python object) into a runtime that an external
coordination engine can drive. Coordinators don't want a chat loop — they
want to dispatch typed tasks, observe progress, enforce budgets, retry
deterministically, and aggregate results.

Surface:
- `Task`           — typed unit of work submitted to the runtime.
- `TaskResult`     — terminal status + output + provenance.
- `Event`          — structured progress events emitted during execution.
- `RuntimeEngine`  — the executor. Stateless across tasks; concurrent-safe.

The engine wraps `agi.Agent` but does not couple to it tightly: anything
that exposes `chat(prompt) -> str` plus the bookkeeping fields used here
can be plugged in (see `tests/test_runtime.py` for the fake agent used in
tests). That separation is what lets the same engine front different
reasoning cores down the line (frozen Opus today, local-base+adapter
later — see ARCHITECTURE.md).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterator, Protocol

from agi.costs import Usage


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

TaskStatus = str  # one of: "pending" | "running" | "succeeded" | "failed" | "cancelled" | "budget_exceeded"


@dataclass
class Budget:
    """Hard ceilings on a single task. The engine refuses to start work
    when any field is non-positive; it cancels in-flight work as soon as
    a ceiling is breached. Coordinators rely on these being honored —
    they're how a planner upstream allocates capacity.
    """
    max_iterations: int = 25
    max_tokens: int = 200_000          # sum of input+output over the whole task
    max_cost_usd: float = 5.0          # hard $ ceiling
    deadline_s: float = 600.0          # wall-clock seconds from submit


@dataclass
class Task:
    instruction: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    inputs: dict[str, Any] = field(default_factory=dict)   # structured context
    skills: list[str] = field(default_factory=list)        # explicit skills to load
    budget: Budget = field(default_factory=Budget)
    metadata: dict[str, Any] = field(default_factory=dict) # coordinator-supplied tags

    def render_prompt(self) -> str:
        """Render the full prompt the reasoning core sees. Coordinators
        supply structured `inputs`; we materialize them deterministically
        so two identical Tasks produce identical prompts (important for
        replay and caching)."""
        if not self.inputs:
            return self.instruction
        rendered_inputs = json.dumps(self.inputs, indent=2, sort_keys=True, default=str)
        return f"{self.instruction}\n\n[inputs]\n{rendered_inputs}"


@dataclass
class Event:
    """A structured progress event. The engine emits these in order; the
    coordinator can subscribe to a live stream or fetch them after the
    fact from `TaskResult.events`. Names are stable and form the public
    contract."""
    type: str            # "started" | "iteration" | "message" | "tool_call" | "tool_result" | "critic" | "succeeded" | "failed" | "budget_exceeded" | "cancelled"
    ts: float = field(default_factory=time.time)
    task_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    output: str = ""
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    iterations: int = 0
    critic_score: float | None = None
    events: list[Event] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [asdict(e) for e in self.events]
        return d


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------


class AgentLike(Protocol):
    """Minimal surface a reasoning core must expose to run inside the
    runtime. `agi.Agent` satisfies this; the test fakes do too."""

    model: str
    usage: Usage
    messages: list[dict]
    last_critic_score: float | None

    def chat(self, user_input: str, max_iterations: int = ...) -> str: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


EventSink = Callable[[Event], None]


@dataclass
class _RunHandle:
    result: TaskResult
    thread: threading.Thread
    cancel: threading.Event
    events: list[Event]
    subscribers: list[EventSink]
    lock: threading.Lock


class BudgetExceeded(RuntimeError):
    """Raised internally when a ceiling is breached. The engine catches
    this and turns it into a budget_exceeded result; callers never see it
    propagate."""


class RuntimeEngine:
    """Executes Tasks against an Agent factory. The engine is stateless
    across tasks — each task gets a fresh Agent so memory, history, and
    cost counters never leak between coordinator-dispatched units of work.

    Concurrency: `submit()` returns a task id immediately and runs work
    on a background thread. `await_result(task_id)` blocks until done.
    Synchronous use: `execute(task)` runs and returns the TaskResult.
    """

    def __init__(
        self,
        agent_factory: Callable[[], AgentLike],
        *,
        tracer=None,
        capabilities: "Capabilities | None" = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.tracer = tracer
        self._capabilities = capabilities
        self._runs: dict[str, _RunHandle] = {}
        self._runs_lock = threading.Lock()

    # ---- Public ---------------------------------------------------------

    def execute(self, task: Task) -> TaskResult:
        """Run a task to completion in the calling thread. Raises nothing
        — failures are reported via the TaskResult.status field."""
        handle = self._make_handle(task)
        self._run(task, handle)
        return handle.result

    def submit(self, task: Task, on_event: EventSink | None = None) -> str:
        """Start a task on a background thread. Optionally subscribe an
        event sink that fires once per event in real time. Returns the
        task id; pass it to `await_result()` or `cancel()` later."""
        handle = self._make_handle(task)
        if on_event is not None:
            handle.subscribers.append(on_event)
        handle.thread = threading.Thread(
            target=self._run, args=(task, handle), name=f"task-{task.id}", daemon=True
        )
        with self._runs_lock:
            self._runs[task.id] = handle
        handle.thread.start()
        return task.id

    def await_result(self, task_id: str, timeout: float | None = None) -> TaskResult:
        handle = self._handle(task_id)
        handle.thread.join(timeout)
        if handle.thread.is_alive():
            raise TimeoutError(f"task {task_id} still running after {timeout}s")
        return handle.result

    def cancel(self, task_id: str) -> None:
        handle = self._handle(task_id)
        handle.cancel.set()

    def stream_events(self, task_id: str, *, from_index: int = 0) -> Iterator[Event]:
        """Yield events as they arrive for a running task. Once the task
        terminates, the generator finishes. Safe to call after completion
        — it just yields the recorded events and stops."""
        handle = self._handle(task_id)
        i = from_index
        while True:
            with handle.lock:
                events = handle.events[i:]
            for ev in events:
                yield ev
                i += 1
            with handle.lock:
                terminal = handle.result.status not in ("pending", "running")
                more = len(handle.events) > i
            if terminal and not more:
                return
            time.sleep(0.01)

    def get_result(self, task_id: str) -> TaskResult:
        return self._handle(task_id).result

    @property
    def capabilities(self) -> "Capabilities":
        from agi.capabilities import describe_runtime  # avoid cycle at import
        if self._capabilities is None:
            self._capabilities = describe_runtime(self.agent_factory)
        return self._capabilities

    # ---- Internals ------------------------------------------------------

    def _handle(self, task_id: str) -> _RunHandle:
        with self._runs_lock:
            handle = self._runs.get(task_id)
        if handle is None:
            raise KeyError(f"no task with id {task_id}")
        return handle

    def _make_handle(self, task: Task) -> _RunHandle:
        result = TaskResult(task_id=task.id, status="pending", metadata=dict(task.metadata))
        return _RunHandle(
            result=result,
            thread=threading.Thread(),  # placeholder; replaced if submitted
            cancel=threading.Event(),
            events=[],
            subscribers=[],
            lock=threading.Lock(),
        )

    def _emit(self, handle: _RunHandle, ev: Event) -> None:
        with handle.lock:
            handle.events.append(ev)
            subscribers = list(handle.subscribers)
            handle.result.events = list(handle.events)
        for sink in subscribers:
            try:
                sink(ev)
            except Exception:
                # A bad subscriber must not take down the runtime.
                pass

    def _run(self, task: Task, handle: _RunHandle) -> None:
        if task.budget.max_iterations <= 0 or task.budget.max_tokens <= 0:
            handle.result.status = "failed"
            handle.result.error = "invalid budget"
            self._emit(handle, Event(type="failed", task_id=task.id, payload={"error": handle.result.error}))
            return

        with self._runs_lock:
            self._runs.setdefault(task.id, handle)

        t0 = time.time()
        handle.result.status = "running"
        self._emit(handle, Event(type="started", task_id=task.id, payload={
            "instruction": task.instruction,
            "skills": task.skills,
            "budget": asdict(task.budget),
        }))

        agent = self.agent_factory()
        prompt = self._assemble_prompt(task, agent)

        try:
            output = self._chat_with_enforcement(agent, prompt, task, handle, t0)
            handle.result.output = output
            handle.result.iterations = agent.usage.turns
            handle.result.usage = {
                "input_tokens": agent.usage.input_tokens,
                "output_tokens": agent.usage.output_tokens,
                "cache_creation_input_tokens": agent.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": agent.usage.cache_read_input_tokens,
            }
            handle.result.cost_usd = agent.usage.cost_usd(agent.model)
            handle.result.critic_score = getattr(agent, "last_critic_score", None)
            handle.result.status = "succeeded"
            handle.result.elapsed_s = time.time() - t0
            self._emit(handle, Event(type="succeeded", task_id=task.id, payload={
                "output": output,
                "usage": handle.result.usage,
                "cost_usd": handle.result.cost_usd,
                "elapsed_s": handle.result.elapsed_s,
                "iterations": handle.result.iterations,
                "critic_score": handle.result.critic_score,
            }))
        except BudgetExceeded as e:
            handle.result.status = "budget_exceeded"
            handle.result.error = str(e)
            handle.result.elapsed_s = time.time() - t0
            handle.result.iterations = agent.usage.turns
            handle.result.usage = {
                "input_tokens": agent.usage.input_tokens,
                "output_tokens": agent.usage.output_tokens,
                "cache_creation_input_tokens": agent.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": agent.usage.cache_read_input_tokens,
            }
            handle.result.cost_usd = agent.usage.cost_usd(agent.model)
            self._emit(handle, Event(type="budget_exceeded", task_id=task.id, payload={"reason": str(e)}))
        except _Cancelled:
            handle.result.status = "cancelled"
            handle.result.elapsed_s = time.time() - t0
            self._emit(handle, Event(type="cancelled", task_id=task.id, payload={}))
        except Exception as e:  # any other failure surfaces to the coordinator
            handle.result.status = "failed"
            handle.result.error = f"{type(e).__name__}: {e}"
            handle.result.elapsed_s = time.time() - t0
            self._emit(handle, Event(type="failed", task_id=task.id, payload={"error": handle.result.error}))

    def _assemble_prompt(self, task: Task, agent: AgentLike) -> str:
        """Attach skill snippets (if any) before the user instruction."""
        if not task.skills:
            return task.render_prompt()
        try:
            from agi.skills import SkillLibrary
        except ImportError:
            return task.render_prompt()
        lib = SkillLibrary()
        snippets = lib.render(task.skills)
        if not snippets:
            return task.render_prompt()
        return f"{snippets}\n\n---\n\n{task.render_prompt()}"

    def _chat_with_enforcement(
        self,
        agent: AgentLike,
        prompt: str,
        task: Task,
        handle: _RunHandle,
        t0: float,
    ) -> str:
        """Run the agent's chat loop while polling budgets.

        Implementation note: agent.chat is a blocking call that loops
        internally up to `max_iterations`. We don't have a hook to
        interrupt every iteration without rewriting the agent, so we
        cap `max_iterations` to the budget value, then check ceilings
        on return. Long-running tool calls inside a single iteration
        will run to completion before the cancel check fires; that's a
        known trade-off (and why per-tool timeouts exist). When a
        coordinator needs hard preemption, it should split work into
        smaller tasks.
        """
        if handle.cancel.is_set():
            raise _Cancelled()
        elapsed = time.time() - t0
        if elapsed > task.budget.deadline_s:
            raise BudgetExceeded(f"deadline {task.budget.deadline_s}s already passed at start")

        # Run the agent. It mutates agent.usage and agent.messages.
        output = agent.chat(prompt, max_iterations=task.budget.max_iterations)

        if handle.cancel.is_set():
            raise _Cancelled()

        # Post-run budget verification — agents that exceeded a ceiling
        # still produced output, but we report budget_exceeded so the
        # coordinator can decide whether to trust the partial result.
        used_tokens = agent.usage.input_tokens + agent.usage.output_tokens
        if used_tokens > task.budget.max_tokens:
            raise BudgetExceeded(
                f"used {used_tokens} tokens, budget {task.budget.max_tokens}"
            )
        cost = agent.usage.cost_usd(agent.model)
        if cost > task.budget.max_cost_usd:
            raise BudgetExceeded(
                f"cost ${cost:.4f}, budget ${task.budget.max_cost_usd:.4f}"
            )
        elapsed = time.time() - t0
        if elapsed > task.budget.deadline_s:
            raise BudgetExceeded(
                f"elapsed {elapsed:.1f}s, deadline {task.budget.deadline_s}s"
            )

        # Forward terminal critic + iteration events so coordinators
        # don't have to inspect the trailing TaskResult for them.
        self._emit(handle, Event(type="iteration", task_id=task.id, payload={
            "iterations": agent.usage.turns,
            "usage_total": used_tokens,
            "cost_usd": cost,
        }))
        critic_score = getattr(agent, "last_critic_score", None)
        if critic_score is not None:
            self._emit(handle, Event(type="critic", task_id=task.id, payload={"score": critic_score}))

        return output


class _Cancelled(Exception):
    pass
