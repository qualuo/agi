"""Engine — the runtime orchestrator.

The engine accepts tasks, executes them concurrently in a thread pool, and
exposes their state and events. The coordination engine (external) decides
*what* to run; this layer is responsible for *running* it.

Responsibilities:

- Maintain a registry of tasks (by id) so coordinators can query/cancel them
- Spin up a worker per task, run the Agent loop inside it, catch failures
- Enforce per-task budgets (cost/tokens/turns/deadline)
- Provide a `delegate` primitive — the agent can spawn child tasks through
  the engine, building a tree the coordinator can walk
- Stream events to subscribers
- Persist trace + cost to the existing learner pipeline

The engine is intentionally synchronous to its callers: `submit()` returns
immediately, `wait()` blocks, `cancel()` is non-blocking. Multiple tasks run
concurrently because each gets its own thread.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Optional

from agi.memory import Memory
from runtime.backend import Backend, AnthropicBackend
from runtime.task import (
    Budget,
    BudgetExceeded,
    Task,
    TaskEvent,
    TaskRecord,
    TaskStatus,
)

logger = logging.getLogger("runtime.engine")


class Engine:
    """Concurrent task runtime.

    Construct once per process. Submit tasks with `submit()`. The engine
    runs each task on a worker thread; tasks are independent and can run
    in parallel (the only shared mutable state is the task registry).

    Args:
        backend: LLM backend. Defaults to AnthropicBackend(), which requires
            ANTHROPIC_API_KEY. Pass MockBackend() for tests.
        memory: shared persistent memory. One per engine — tasks see each
            other's notes. Pass per-task memory by overriding `agent_factory`.
        max_concurrent: cap on simultaneous running tasks. Excess tasks
            sit in QUEUED until a slot opens.
        tracer: optional TraceLogger. If set, every task logs a trace on
            completion (consumed by learner/train.py).
        agent_factory: advanced — override how the Agent for a task is built.
            Signature: (task, engine) -> Agent.
    """

    def __init__(
        self,
        *,
        backend: Optional[Backend] = None,
        memory: Optional[Memory] = None,
        max_concurrent: int = 4,
        tracer=None,
        agent_factory: Optional[Callable[["Task", "Engine"], Any]] = None,
        skill_library=None,
    ) -> None:
        self.backend: Backend = backend or AnthropicBackend()
        self.memory = memory or Memory()
        self.tracer = tracer
        self.skill_library = skill_library
        self.agent_factory = agent_factory or _default_agent_factory

        self._executor = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="runtime")
        self._tasks: dict[str, Task] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    # --- public API --------------------------------------------------------

    def submit(
        self,
        instruction: str,
        *,
        budget: Optional[Budget] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Task:
        """Submit a new task. Returns immediately with the Task handle.
        Use `task.wait()` or `task.stream_events()` to observe progress.
        """
        if self._shutdown:
            raise RuntimeError("engine is shut down")

        task = Task(
            instruction=instruction,
            budget=budget,
            parent_id=parent_id,
            metadata=metadata,
            task_id=task_id,
        )
        with self._lock:
            self._tasks[task.id] = task
            if parent_id and parent_id in self._tasks:
                self._tasks[parent_id].children.append(task.id)

        future = self._executor.submit(self._run_task, task)
        with self._lock:
            self._futures[task.id] = future
        return task

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Request cancellation. Returns True if the task exists and was
        running; False otherwise. Cancellation is cooperative — the agent
        loop checks between turns. Tasks blocked in a long tool call will
        finish that call first."""
        task = self.get(task_id)
        if task is None:
            return False
        if task.status.terminal:
            return False
        task.cancel()
        return True

    def list_tasks(self, *, status: Optional[TaskStatus] = None) -> list[TaskRecord]:
        with self._lock:
            tasks = list(self._tasks.values())
        records = [t.snapshot() for t in tasks]
        if status is not None:
            records = [r for r in records if r.status is status]
        records.sort(key=lambda r: r.created_at)
        return records

    def task_tree(self, root_id: str) -> dict[str, Any]:
        """Return the task subtree rooted at `root_id` as nested dicts.
        Useful for a coordinator UI."""
        root = self.get(root_id)
        if root is None:
            return {}

        def build(t: Task) -> dict[str, Any]:
            snap = t.snapshot().as_dict()
            snap["children"] = [
                build(child) for cid in t.children if (child := self.get(cid)) is not None
            ]
            return snap

        return build(root)

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown = True
        with self._lock:
            for t in self._tasks.values():
                if not t.status.terminal:
                    t.cancel()
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # --- internals ---------------------------------------------------------

    def _run_task(self, task: Task) -> None:
        """Worker thread entry point. Owns the task's lifecycle transitions."""
        try:
            task.set_status(TaskStatus.RUNNING)
            agent = self.agent_factory(task, self)
            try:
                result = agent.chat(task.instruction)
            finally:
                # Pull cost/usage from the agent regardless of success/failure
                # so partial work is still accounted for.
                _absorb_usage(task, agent)

            if task.cancel_requested:
                task.set_status(TaskStatus.CANCELLED)
                return

            task.result = result or ""
            task.emit("completed", {"result": task.result})
            task.set_status(TaskStatus.COMPLETED)
        except BudgetExceeded as e:
            task.error = f"budget exceeded: {e}"
            task.emit("error", {"type": "BudgetExceeded", "message": str(e)})
            task.set_status(TaskStatus.FAILED)
        except _CancelledMid:
            task.set_status(TaskStatus.CANCELLED)
        except Exception as e:  # noqa: BLE001 — engine has to catch everything
            logger.exception("task %s failed", task.id)
            task.error = f"{type(e).__name__}: {e}"
            task.emit("error", {"type": type(e).__name__, "message": str(e)})
            task.set_status(TaskStatus.FAILED)
        finally:
            if self.tracer is not None:
                self._log_trace(task)

    def _log_trace(self, task: Task) -> None:
        try:
            self.tracer.log(
                model="runtime",
                messages=[{"role": "user", "content": task.instruction}],
                final_text=task.result or "",
                usage={
                    "input_tokens": task.input_tokens,
                    "output_tokens": task.output_tokens,
                },
                metadata={
                    "task_id": task.id,
                    "parent_id": task.parent_id,
                    "status": task.status.value,
                    "cost_usd": task.cost_usd,
                    "turns": task.turns,
                    "error": task.error,
                    **task.metadata,
                },
            )
        except Exception:
            logger.exception("trace logging failed for task %s", task.id)


# ---------------------------------------------------------------------------
# Internal: factory that builds an Agent bound to a specific Task
# ---------------------------------------------------------------------------


class _CancelledMid(Exception):
    """Raised inside the agent loop when cancellation is detected mid-task.
    Caught in `Engine._run_task` and turned into a TaskStatus.CANCELLED."""


def _default_agent_factory(task: Task, engine: Engine):
    """Build the Agent for a task, wired to the engine.

    Imports `agi.agent` lazily so the runtime package can be imported in
    environments without the `anthropic` package (when using MockBackend).
    """
    from agi.agent import Agent

    # Build a delegate tool that lets *this* agent spawn child tasks.
    delegate_schema, delegate_handler = _build_delegate_tool(task, engine)

    # Build the skill-injection system suffix.
    skill_suffix = ""
    if engine.skill_library is not None:
        relevant = engine.skill_library.retrieve(task.instruction, k=3)
        if relevant:
            skill_suffix = (
                "\n\nRelevant skills from your library (prefer these procedures "
                "when they apply):\n\n" + "\n\n---\n\n".join(
                    f"# {s.name}\n{s.body}" for s in relevant
                )
            )
            task.emit(
                "skills_loaded",
                {"names": [s.name for s in relevant]},
            )

    agent = Agent(
        memory=engine.memory,
        backend=engine.backend,
        verbose=False,
        event_sink=task.emit,
        cancel_check=_make_cancel_check(task),
        extra_tools=[(delegate_schema, delegate_handler)],
        budget_check=_make_budget_check(task),
        system_suffix=skill_suffix,
        tracer=None,  # engine logs the rolled-up trace
    )
    return agent


def _make_cancel_check(task: Task) -> Callable[[], None]:
    def check() -> None:
        if task.cancel_requested:
            raise _CancelledMid()
    return check


def _make_budget_check(task: Task):
    def check(*, cost_usd, input_tokens, output_tokens, turns, elapsed_seconds):
        task.cost_usd = cost_usd
        task.input_tokens = input_tokens
        task.output_tokens = output_tokens
        task.turns = turns
        task.budget.check(
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            turns=turns,
            elapsed_seconds=elapsed_seconds,
        )
    return check


def _absorb_usage(task: Task, agent) -> None:
    """Copy final usage from an agent into the task. Tolerant of partial state."""
    usage = getattr(agent, "usage", None)
    if usage is None:
        return
    task.input_tokens = getattr(usage, "input_tokens", task.input_tokens)
    task.output_tokens = getattr(usage, "output_tokens", task.output_tokens)
    model = getattr(agent, "model", "claude-opus-4-7")
    try:
        task.cost_usd = usage.cost_usd(model)
    except Exception:
        pass
    task.turns = max(task.turns, getattr(usage, "turns", task.turns))


# ---------------------------------------------------------------------------
# Delegate tool: agent -> child task via the engine
# ---------------------------------------------------------------------------


def _build_delegate_tool(parent_task: Task, engine: Engine) -> tuple[dict, Callable[..., str]]:
    """Create a delegate tool bound to (parent_task, engine).

    The agent calls `delegate(instruction, ...)` and gets the child's final
    text back. The child runs as a real engine task: it has its own id, its
    own events, its own budget, and a parent pointer. Coordinators can walk
    the resulting tree.
    """
    schema = {
        "name": "delegate",
        "description": (
            "Spawn a child task to handle a focused subproblem. Use this for "
            "decomposition: hand off a self-contained sub-question to a fresh "
            "agent with its own context, and you get back its final answer. "
            "Useful for parallel sub-questions, expensive research detours, "
            "and bounded exploration. Each child runs with its own budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The sub-task. Be specific and self-contained.",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Cap on agent turns for the child (default 10).",
                    "default": 10,
                },
                "max_cost_usd": {
                    "type": "number",
                    "description": "Cap on USD cost for the child (default 1.0).",
                    "default": 1.0,
                },
            },
            "required": ["instruction"],
        },
    }

    def handler(instruction: str, max_turns: int = 10, max_cost_usd: float = 1.0) -> str:
        child_budget = Budget(max_turns=max_turns, max_cost_usd=max_cost_usd)
        child = engine.submit(
            instruction=instruction,
            budget=child_budget,
            parent_id=parent_task.id,
        )
        parent_task.emit("delegated", {"child_task_id": child.id, "instruction": instruction})
        # Block this agent turn until the child is done. We pick a generous
        # ceiling so a stuck child can't hang the parent forever — the child's
        # own budget will trip first in normal operation.
        finished = child.wait(timeout=3600)
        if not finished:
            return f"error: child task {child.id} did not finish within 1 hour"
        snap = child.snapshot()
        parent_task.emit(
            "delegated_complete",
            {"child_task_id": child.id, "status": snap.status.value, "result": snap.result},
        )
        if snap.status is TaskStatus.COMPLETED:
            return snap.result or ""
        return f"error: child task {child.id} ended in status {snap.status.value}: {snap.error or ''}"

    return schema, handler


# ---------------------------------------------------------------------------
# Trace metadata helper for the HTTP server
# ---------------------------------------------------------------------------


def event_to_jsonable(ev: TaskEvent) -> dict:
    """JSON-safe form of a TaskEvent for the HTTP / SSE layer."""
    return {"kind": ev.kind, "data": _scrub(ev.data), "ts": ev.ts}


def _scrub(value):
    """Make a value JSON-serializable. Best-effort, never raises."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_scrub(v) for v in value]
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        return repr(value)
