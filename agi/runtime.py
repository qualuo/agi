"""Runtime — the orchestration layer above Agent.

A coordination engine drives the runtime via this surface:

    rt = Runtime()
    t = rt.submit("write a haiku about idempotent retries",
                  budget=Budget(max_usd=0.10, max_wall_seconds=60))
    rt.subscribe(lambda ev: print(ev.kind, ev.data))
    result = rt.wait(t.id, timeout=120)

Or for streaming consumers (a web UI, an outer planner, a queue worker),
subscribe to the event bus and consume `task.text`, `task.tool_call`,
`task.completed` as they happen.

Design contracts:

1. **Tasks are first-class.** Each `submit` returns a `TaskHandle` with a
   stable id. State lives in the Runtime, not in the caller. The lifecycle
   is: PENDING → RUNNING → (SUCCEEDED | FAILED | CANCELLED).
2. **Parallel by default.** Tasks run on a thread pool. Use `submit_batch`
   to launch N tasks; `wait_all` to join.
3. **Budgets are honored.** Per-task `Budget` (USD / tokens / wall) is
   enforced between agent turns and surfaces as `task.budget_exceeded`.
4. **Subagent delegation rolls up.** When an agent calls `delegate(role, ...)`,
   the runtime spawns a child task with parent_task_id set; events carry
   the parent linkage so a coordinator can reconstruct the tree.
5. **Capability manifest is the contract.** `rt.manifest()` returns a stable
   JSON-serializable description of what this runtime can do. A coordination
   engine inspects it to decide whether to route work here.

What the runtime is NOT:
- A multi-tenant queue (no auth, no rate limiting). Wrap it in a real
  job server (Celery / Temporal / your favorite) for production.
- A scheduler with priorities / preemption. Tasks run FIFO on the pool.
- An auto-retry harness. Failures surface to the coordinator unmodified.
  Retries are policy, not mechanism.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from agi.agent import Agent
from agi.budget import Budget
from agi.capabilities import (
    CapabilityManifest,
    RoleDescriptor,
    SkillDescriptor,
    ToolDescriptor,
)
from agi.costs import PRICING
from agi.events import Event, EventBus, new_task_id
from agi.memory import Memory
from agi import roles
from agi.skills import SkillLibrary
from agi.synth_registry import SynthToolRegistry

try:
    from learner.traces import TraceLogger
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore


RUNTIME_VERSION = "0.2.0"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskRecord:
    id: str
    prompt: str
    role: str
    model: str
    parent_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    submitted_ts: float = field(default_factory=time.time)
    started_ts: float | None = None
    finished_ts: float | None = None
    result: str | None = None
    error: str | None = None
    cost_usd: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "role": self.role,
            "model": self.model,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "submitted_ts": self.submitted_ts,
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "result": self.result,
            "error": self.error,
            "cost_usd": self.cost_usd,
            "elapsed_seconds": (
                (self.finished_ts or time.time()) - self.started_ts
                if self.started_ts else 0.0
            ),
        }


@dataclass
class TaskHandle:
    id: str
    runtime: "Runtime"

    def status(self) -> dict[str, Any]:
        return self.runtime.status(self.id)

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        return self.runtime.wait(self.id, timeout=timeout)

    def cancel(self) -> bool:
        return self.runtime.cancel(self.id)


class Runtime:
    """The runtime engine.

    All public methods are thread-safe. The runtime owns one event bus, one
    skill library, one synthesized-tool registry, and one trace logger;
    each task gets its own Agent instance, Memory (or shared, if injected),
    and budget.
    """

    def __init__(
        self,
        *,
        default_model: str = "claude-opus-4-7",
        max_workers: int = 4,
        memory_path: str | os.PathLike[str] | None = None,
        skill_root: str | os.PathLike[str] | None = None,
        synth_root: str | os.PathLike[str] | None = None,
        trace_path: str | os.PathLike[str] | None = None,
        enable_reflection: bool = False,
        enable_traces: bool = True,
        shared_memory: bool = True,
    ) -> None:
        self.default_model = default_model
        self.enable_reflection = enable_reflection
        self.shared_memory = shared_memory

        self.bus = EventBus()
        self.skills = SkillLibrary(root=skill_root)
        self.synth = SynthToolRegistry(root=synth_root)
        self._memory_path = memory_path
        self._shared_memory_instance: Memory | None = (
            Memory(path=memory_path) if shared_memory else None
        )
        self.tracer = TraceLogger(path=trace_path) if (enable_traces and TraceLogger is not None) else None

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agi-task")
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.RLock()

    # --------------------------------------------------------------- public
    def submit(
        self,
        prompt: str,
        *,
        role: str = "executor",
        model: str | None = None,
        budget: Budget | None = None,
        parent_id: str | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 25,
    ) -> TaskHandle:
        """Queue a task. Returns immediately with a TaskHandle.

        `role` selects the system prompt and default model; pass "" or
        "raw" to use the runtime's default model + base system prompt
        and a custom `system_prompt`.
        """
        task_id = new_task_id()
        role_obj = roles.get(role) if role else None
        chosen_model = model or (role_obj.default_model if role_obj else self.default_model)
        record = TaskRecord(
            id=task_id,
            prompt=prompt,
            role=role or "default",
            model=chosen_model,
            parent_id=parent_id,
        )
        with self._lock:
            self._tasks[task_id] = record

        self.bus.emit(
            "task.submitted", task_id, parent_task_id=parent_id,
            role=role, model=chosen_model, prompt=prompt,
        )

        def _runner() -> None:
            self._run_task(record, role_obj, budget, system_prompt, max_iterations)

        record.future = self._executor.submit(_runner)
        return TaskHandle(id=task_id, runtime=self)

    def submit_batch(self, prompts: list[str], *, role: str = "executor", **kwargs) -> list[TaskHandle]:
        return [self.submit(p, role=role, **kwargs) for p in prompts]

    def status(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                raise KeyError(task_id)
            return rec.to_dict()

    def list_tasks(self, *, status: TaskStatus | None = None) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._tasks.values())
        if status is not None:
            records = [r for r in records if r.status == status]
        return [r.to_dict() for r in records]

    def wait(self, task_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                raise KeyError(task_id)
            fut = rec.future
        if fut is not None:
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass  # error captured on the record
        return self.status(task_id)

    def wait_all(self, task_ids: list[str], *, timeout: float | None = None) -> list[dict[str, Any]]:
        deadline = (time.time() + timeout) if timeout is not None else None
        out: list[dict[str, Any]] = []
        for tid in task_ids:
            remaining = (deadline - time.time()) if deadline is not None else None
            if remaining is not None and remaining <= 0:
                out.append(self.status(tid))
            else:
                out.append(self.wait(tid, timeout=remaining))
        return out

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None or rec.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False
        rec.cancel_event.set()
        return True

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        return self.bus.subscribe(fn)

    def manifest(self) -> CapabilityManifest:
        """Capability manifest — the discovery contract for coordinators."""
        # Inspect the schemas a default Agent would expose, without standing
        # one up (no API call required).
        from agi.tools import make_tools

        tmp_mem = self._shared_memory_instance or Memory(path=self._memory_path)
        schemas, _ = make_tools(
            tmp_mem,
            skills=self.skills,
            synth=self.synth,
            delegate_fn=lambda *a, **kw: "",
        )

        tools: list[ToolDescriptor] = []
        for s in schemas:
            origin = "builtin"
            tools.append(ToolDescriptor(
                name=s.get("name", "?"),
                description=s.get("description", ""),
                input_schema=s.get("input_schema", {}),
                origin=origin,
            ))
        for name, t in self.synth.all().items():
            tools.append(ToolDescriptor(
                name=name,
                description=t.description,
                input_schema=t.input_schema or {},
                origin="synthesized",
            ))
        # Mention server tools that get added by the Agent when enabled.
        tools.append(ToolDescriptor(name="web_search", description="Server-side web search.",
                                    input_schema={}, origin="server"))
        tools.append(ToolDescriptor(name="web_fetch", description="Server-side URL fetch.",
                                    input_schema={}, origin="server"))

        skill_descs = [
            SkillDescriptor(name=s.name, when_to_use=s.when_to_use, usage_count=s.usage_count)
            for s in self.skills.all()
        ]
        role_descs = [
            RoleDescriptor(name=r.name, description=r.description,
                           system_prompt=r.system_prompt, model=r.default_model)
            for r in roles.all_roles()
        ]
        return CapabilityManifest(
            runtime_version=RUNTIME_VERSION,
            models=sorted(PRICING.keys()),
            tools=tools,
            skills=skill_descs,
            roles=role_descs,
            limits={
                "max_workers": self._executor._max_workers,
            },
            features={
                "events": True,
                "skills": True,
                "synth_tools": True,
                "delegation": True,
                "reflection": self.enable_reflection,
                "traces": self.tracer is not None,
            },
        )

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        self.bus.close()

    # ---------------------------------------------------------------- worker
    def _run_task(
        self,
        record: TaskRecord,
        role_obj,
        budget: Budget | None,
        system_prompt: str | None,
        max_iterations: int,
    ) -> None:
        record.status = TaskStatus.RUNNING
        record.started_ts = time.time()
        if record.cancel_event.is_set():
            record.status = TaskStatus.CANCELLED
            record.finished_ts = time.time()
            self.bus.emit("task.cancelled", record.id, parent_task_id=record.parent_id, reason="pre-start")
            return

        # Pick memory: shared (default) or per-task isolated.
        memory = self._shared_memory_instance if self.shared_memory else Memory(path=self._memory_path)

        sys_prompt = system_prompt or (role_obj.system_prompt if role_obj else None)

        agent = Agent(
            memory=memory,
            model=record.model,
            verbose=False,
            tracer=self.tracer,
            event_bus=self.bus,
            task_id=record.id,
            parent_task_id=record.parent_id,
            skills=self.skills,
            synth=self.synth,
            delegate_fn=self._make_delegate_fn(record.id),
            budget=budget,
            system_prompt_override=sys_prompt,
            reflect=self.enable_reflection,
            cancel_event=record.cancel_event,
        )

        try:
            result = agent.chat(record.prompt, max_iterations=max_iterations)
            record.result = result
            record.cost_usd = agent.usage.cost_usd(record.model)
            if record.cancel_event.is_set():
                record.status = TaskStatus.CANCELLED
            else:
                record.status = TaskStatus.SUCCEEDED
        except Exception as e:  # noqa: BLE001 — surface any agent crash
            record.error = f"{type(e).__name__}: {e}"
            record.status = TaskStatus.FAILED
            self.bus.emit(
                "task.failed", record.id, parent_task_id=record.parent_id,
                error=record.error,
            )
        finally:
            record.finished_ts = time.time()

    def _make_delegate_fn(self, parent_id: str):
        """Return a callable the agent's `delegate` tool will invoke."""
        def _delegate(role: str, task: str, opts: dict | None) -> str:
            opts = opts or {}
            budget: Budget | None = None
            if "max_usd" in opts and opts["max_usd"] is not None:
                budget = Budget(max_usd=float(opts["max_usd"]))
            self.bus.emit(
                "task.subagent_spawned", parent_id,
                parent_task_id=None, role=role, task=task,
            )
            handle = self.submit(task, role=role, parent_id=parent_id, budget=budget)
            # Block until the subagent finishes — delegation is sync from
            # the parent agent's perspective.
            status = handle.wait(timeout=600)
            self.bus.emit(
                "task.subagent_finished", parent_id,
                parent_task_id=None,
                child_task_id=handle.id,
                status=status["status"],
                cost_usd=status.get("cost_usd", 0.0),
            )
            return status.get("result") or status.get("error") or "(no result)"
        return _delegate
