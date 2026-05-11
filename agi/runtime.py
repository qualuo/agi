"""Runtime layer.

This module turns the bare `Agent` into a runtime engine that an external
coordinator (or a parent agent) can drive. The unit of work is a `Run`:
a single task with a `RunSpec` (prompt + budget + role), a structured
event stream, a status field, and a persistent record on disk.

A coordination engine talks to `Runtime`:

    rt = Runtime()
    run = rt.submit(RunSpec(prompt="summarize ./README.md"))
    for event in run.iter_events():
        ...        # forward to the coordinator's bus
    print(run.result.text, run.result.cost_usd)

Or, fire-and-forget with persistence — the coordinator picks up later by id:

    rid = rt.submit(spec).id
    ...
    run = rt.get(rid)
    print(run.result, run.status)

Budgets are enforced inside the agent loop via a `Budget.consumed_*` callback
that the runtime updates after each turn. If a budget is exceeded mid-run,
the agent raises `BudgetExceeded`, which the run records as a cancellation.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agi import events as ev
from agi.costs import PRICING, CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, Usage
from agi.memory import Memory


class BudgetExceeded(Exception):
    """Raised by the budget checker; caught by the runtime."""
    def __init__(self, kind: str, limit: float, actual: float):
        super().__init__(f"budget exceeded: {kind} actual={actual} > limit={limit}")
        self.kind = kind
        self.limit = limit
        self.actual = actual


class RunCancelled(Exception):
    """Raised when the coordinator cancels a run via `Run.cancel()`."""


@dataclass
class Budget:
    """Hard caps on a run.

    All limits are optional; `None` means no cap. Limits are checked between
    turns and after each tool call. A run that trips a cap finishes with
    status='cancelled' and a `budget_exceeded` event.
    """
    max_usd: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_turns: int | None = None
    max_wallclock_s: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def check(self, usage: Usage, model: str, turns: int, elapsed_s: float) -> None:
        if self.max_turns is not None and turns > self.max_turns:
            raise BudgetExceeded("turns", self.max_turns, turns)
        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            raise BudgetExceeded("input_tokens", self.max_input_tokens, usage.input_tokens)
        if self.max_output_tokens is not None and usage.output_tokens > self.max_output_tokens:
            raise BudgetExceeded("output_tokens", self.max_output_tokens, usage.output_tokens)
        if self.max_wallclock_s is not None and elapsed_s > self.max_wallclock_s:
            raise BudgetExceeded("wallclock_s", self.max_wallclock_s, elapsed_s)
        if self.max_usd is not None:
            spent = usage.cost_usd(model)
            if spent > self.max_usd:
                raise BudgetExceeded("usd", self.max_usd, spent)


@dataclass
class RunSpec:
    """Inputs to a run.

    `role` selects a system-prompt preset (planner/executor/critic/general).
    `parent_run_id` is set for subagent runs so cost/events roll up to the
    parent. `metadata` is a free-form bag the coordinator can attach for
    later attribution.
    """
    prompt: str
    role: str = "general"
    model: str = "claude-opus-4-7"
    effort: str = "high"
    max_iterations: int = 25
    budget: Budget = field(default_factory=Budget)
    parent_run_id: str | None = None
    skills: list[str] = field(default_factory=list)  # names of skills to preload
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["budget"] = self.budget.to_dict()
        return d


@dataclass
class RunResult:
    text: str
    usage: dict
    cost_usd: float
    critic_score: float | None
    child_run_ids: list[str] = field(default_factory=list)


class Run:
    """A single submitted task plus its live state.

    A Run is created by `Runtime.submit`. The runtime spawns a worker thread
    that drives the Agent and pushes events into this Run's queue. Callers
    consume events via `iter_events()`. After the run finishes, `result`,
    `usage`, `cost_usd`, `status`, and `error` are populated.
    """

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    def __init__(self, spec: RunSpec, *, run_id: str | None = None) -> None:
        self.id = run_id or uuid.uuid4().hex[:12]
        self.spec = spec
        self.status: str = self.STATUS_PENDING
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None
        self.result: RunResult | None = None
        self.usage = Usage()
        self._events: queue.Queue[ev.Event | None] = queue.Queue()
        self._buffered: list[ev.Event] = []
        self._seq = 0
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.child_run_ids: list[str] = []

    @property
    def is_terminal(self) -> bool:
        return self.status in (self.STATUS_COMPLETED, self.STATUS_FAILED, self.STATUS_CANCELLED)

    def emit(self, event: ev.Event) -> None:
        """Push an event onto the run's queue. Used by the agent loop."""
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            event.run_id = self.id
            self._buffered.append(event)
        self._events.put(event)

    def cancel(self, reason: str = "cancelled by caller") -> None:
        """Signal cancellation. The agent checks this between turns."""
        self._cancel.set()
        # We don't synthesize the run_cancelled event here; the worker thread
        # does it once the agent loop unwinds, so usage is accurate.
        self._cancel_reason = reason  # type: ignore[attr-defined]

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise RunCancelled(getattr(self, "_cancel_reason", "cancelled"))

    def events(self) -> list[ev.Event]:
        """All events emitted so far (snapshot)."""
        with self._lock:
            return list(self._buffered)

    def iter_events(self, timeout: float | None = None) -> Iterator[ev.Event]:
        """Stream events as they happen. Stops after a terminal event."""
        while True:
            try:
                event = self._events.get(timeout=timeout)
            except queue.Empty:
                return
            if event is None:  # sentinel for end-of-stream
                return
            yield event
            if event.type in ("run_completed", "run_failed", "run_cancelled"):
                return

    def _close(self) -> None:
        """Signal end-of-stream. Called by the runtime when the worker exits."""
        self._events.put(None)

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "spec": self.spec.to_dict(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": asdict(self.result) if self.result else None,
            "usage": asdict(self.usage),
            "cost_usd": self.usage.cost_usd(self.spec.model),
            "child_run_ids": list(self.child_run_ids),
        }


ROLE_PROMPTS: dict[str, str] = {
    "general": "",  # use the base Agent system prompt
    "planner": (
        "You are the planner. Decompose the task into a numbered list of"
        " concrete steps a downstream executor can run without further"
        " clarification. Do not execute the steps yourself; just plan."
    ),
    "executor": (
        "You are the executor. You will be handed a concrete step. Run it"
        " using your tools and report the result. Do not re-plan; if a step"
        " is ambiguous, surface the ambiguity and stop."
    ),
    "critic": (
        "You are the critic. You will be shown a task and a candidate"
        " answer. Report concrete defects, missing verifications, and a"
        " final verdict: PASS or FAIL with one-line reason."
    ),
    "researcher": (
        "You are the researcher. Use web_search and web_fetch to gather"
        " evidence. Cite URLs inline. Do not speculate; if you cannot find"
        " a source, say so."
    ),
}


class Runtime:
    """Submit, track, and replay runs.

    The runtime owns:
      - the per-run worker threads
      - the on-disk run registry (`~/.agi/runs/<id>.json`)
      - the trace logger (optional; same one used by learner/)
      - default Memory (shared across runs unless a run brings its own)
    """

    def __init__(
        self,
        *,
        memory: Memory | None = None,
        tracer=None,
        registry_dir: str | os.PathLike[str] | None = None,
        agent_factory: Callable[..., Any] | None = None,
        max_subagent_depth: int = 3,
    ) -> None:
        self.memory = memory or Memory()
        self.tracer = tracer
        self.registry_dir = Path(registry_dir) if registry_dir else Path.home() / ".agi" / "runs"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self.max_subagent_depth = max_subagent_depth
        # agent_factory is injected for testing — defaults to the real Agent.
        self._agent_factory = agent_factory

    def submit(
        self,
        spec: RunSpec | str,
        *,
        depth: int = 0,
        parent_run_id: str | None = None,
        wait: bool = False,
    ) -> Run:
        if isinstance(spec, str):
            spec = RunSpec(prompt=spec)
        if parent_run_id is not None:
            spec.parent_run_id = parent_run_id
        if depth > self.max_subagent_depth:
            raise RuntimeError(
                f"subagent depth {depth} exceeds max {self.max_subagent_depth}"
            )

        run = Run(spec=spec)
        with self._lock:
            self._runs[run.id] = run
        self._persist(run)

        thread = threading.Thread(
            target=self._drive,
            args=(run, depth),
            daemon=True,
            name=f"run-{run.id}",
        )
        run._thread = thread
        thread.start()
        if wait:
            run.wait()
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[Run]:
        return list(self._runs.values())

    def load_record(self, run_id: str) -> dict | None:
        path = self.registry_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _persist(self, run: Run) -> None:
        path = self.registry_dir / f"{run.id}.json"
        path.write_text(json.dumps(run.to_dict(), default=str, indent=2))

    def _drive(self, run: Run, depth: int) -> None:
        """Worker thread: build an Agent, run the chat, capture results."""
        spec = run.spec
        run.status = Run.STATUS_RUNNING
        run.started_at = time.time()
        run.emit(ev.run_started(run.id, spec.prompt, spec.model, spec.budget.to_dict()))

        try:
            agent = self._build_agent(run, depth)
            text = agent.chat(spec.prompt, max_iterations=spec.max_iterations)
            run.usage = agent.usage
            critic_score = getattr(agent, "last_critic_score", None)
            run.result = RunResult(
                text=text,
                usage=asdict(agent.usage),
                cost_usd=agent.usage.cost_usd(spec.model),
                critic_score=critic_score,
                child_run_ids=list(run.child_run_ids),
            )
            run.status = Run.STATUS_COMPLETED
            run.emit(ev.run_completed(
                run.id,
                text,
                asdict(agent.usage),
                agent.usage.cost_usd(spec.model),
            ))
        except RunCancelled as e:
            run.status = Run.STATUS_CANCELLED
            run.error = str(e)
            run.emit(ev.run_cancelled(run.id, str(e), asdict(run.usage)))
        except BudgetExceeded as e:
            run.status = Run.STATUS_CANCELLED
            run.error = str(e)
            run.emit(ev.budget_exceeded(run.id, e.kind, e.limit, e.actual))
            run.emit(ev.run_cancelled(run.id, str(e), asdict(run.usage)))
        except Exception as e:  # pragma: no cover - defensive
            run.status = Run.STATUS_FAILED
            run.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            run.emit(ev.run_failed(run.id, run.error, asdict(run.usage)))
        finally:
            run.finished_at = time.time()
            self._persist(run)
            run._close()

    def _build_agent(self, run: Run, depth: int):
        """Instantiate an Agent wired to this run's event sink and budget."""
        from agi.agent import Agent  # local import — avoid cycles

        factory = self._agent_factory or Agent
        role_addendum = ROLE_PROMPTS.get(run.spec.role, "")

        agent = factory(
            memory=self.memory,
            model=run.spec.model,
            effort=run.spec.effort,
            verbose=False,
            tracer=self.tracer,
            event_sink=run,
            budget=run.spec.budget,
            runtime=self,
            depth=depth,
            role_addendum=role_addendum,
            preload_skills=list(run.spec.skills),
        )
        return agent
