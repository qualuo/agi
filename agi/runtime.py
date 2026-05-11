"""AGI runtime engine.

A coordination engine drives the runtime: it submits tasks, streams events,
controls (cancel/pause), inspects state, and composes runs into trees via
delegation. The runtime owns shared, durable state — memory, skills, traces,
adapters — so individual runs are cheap and stateless on top.

Shape:

    rt = Runtime()                        # one per process, owns shared state
    run = rt.submit(RunRequest("..."))    # returns RunHandle
    for evt in run.events():              # blocking iterator, typed Events
        ...
    run.cancel()                          # cooperative cancellation
    run.wait().result                     # final RunResult once done

The Agent emits events through a sink; the Runtime hooks every Agent into a
sink that pushes into the handle's queue. This keeps Agent's own API surface
unchanged (you can still call `agent.chat(...)` directly without the runtime).
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agi.events import (
    Event,
    make,
    RUN_STARTED, DONE, ERROR, CANCELLED, USAGE, REFLECTION, CRITIC_SCORE,
    SKILLS_LOADED, SUBRUN_STARTED, SUBRUN_COMPLETED,
)
from agi.memory import Memory
from agi.skills import SkillLibrary
from agi.reflection import Reflector


# Sentinel pushed onto a run's event queue to signal end-of-stream.
_END = object()


@dataclass
class RunRequest:
    task: str
    skills: bool = True             # auto-load skill library hits
    reflect: bool = True            # write a lesson on completion
    max_iterations: int = 25
    parent_id: str | None = None    # set when this is a sub-run
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    text: str
    passed: bool | None = None
    critic_score: float | None = None
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    cancelled: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RunHandle:
    """Handle on a single executing run. Thread-safe."""

    def __init__(
        self,
        run_id: str,
        request: RunRequest,
        runtime: "Runtime",
    ) -> None:
        self.run_id = run_id
        self.request = request
        self.runtime = runtime
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._events: list[Event] = []
        self._events_lock = threading.Lock()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self.result: RunResult | None = None
        self.t0 = time.time()

    # --- producer side --------------------------------------------------

    def emit(self, event: Event) -> None:
        with self._events_lock:
            self._events.append(event)
        self._queue.put(event)

    def finish(self, result: RunResult) -> None:
        self.result = result
        self._done.set()
        self._queue.put(_END)

    # --- consumer side --------------------------------------------------

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def events(self, timeout: float | None = None) -> Iterator[Event]:
        """Block-iterate events until the run finishes.

        If `timeout` is set, individual `get` calls time out and stop iteration
        (caller can call again to resume). Without a timeout, iterates to the
        end of the stream.
        """
        while True:
            try:
                item = self._queue.get(timeout=timeout) if timeout else self._queue.get()
            except queue.Empty:
                return
            if item is _END:
                return
            yield item

    def replay(self) -> list[Event]:
        """Snapshot of all events emitted so far. Safe to call concurrently."""
        with self._events_lock:
            return list(self._events)

    def wait(self, timeout: float | None = None) -> "RunHandle":
        self._done.wait(timeout=timeout)
        return self


# ---------------------------------------------------------------------------


class Runtime:
    """The runtime engine: shared state + per-run scheduling.

    Configurable so a coordination engine can wire up its own backends:

    - `agent_factory(runtime, request, on_event)` returns a configured
      Agent (or any object with `.chat(task, max_iterations)` and the
      attributes the runtime reads back: `.last_critic_score`,
      `.usage`, `.model`).
    - `memory`, `skills`, `traces` default to local on-disk stores at
      `~/.agi/`.

    `submit` is non-blocking; each run executes on its own daemon thread.
    """

    def __init__(
        self,
        *,
        memory: Memory | None = None,
        skills: SkillLibrary | None = None,
        traces: Any = None,
        agent_factory: "Callable[[Runtime, RunHandle], Any] | None" = None,
        root_dir: str | Path | None = None,
    ) -> None:
        root = Path(root_dir) if root_dir else Path.home() / ".agi"
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.memory = memory or Memory(path=root / "memory.jsonl")
        self.skills = skills or SkillLibrary(path=root / "skills")
        self.traces = traces or _default_tracer(root / "traces.jsonl")
        self.agent_factory = agent_factory or _default_agent_factory
        self._handles: dict[str, RunHandle] = {}
        self._handles_lock = threading.Lock()

    # --- run management -------------------------------------------------

    def submit(self, request: RunRequest) -> RunHandle:
        run_id = uuid.uuid4().hex[:12]
        handle = RunHandle(run_id, request, self)
        with self._handles_lock:
            self._handles[run_id] = handle
        thread = threading.Thread(target=self._execute, args=(handle,), daemon=True)
        handle._thread = thread
        thread.start()
        return handle

    def get(self, run_id: str) -> RunHandle | None:
        with self._handles_lock:
            return self._handles.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._handles_lock:
            out = []
            for h in self._handles.values():
                out.append({
                    "run_id": h.run_id,
                    "task": h.request.task,
                    "parent_id": h.request.parent_id,
                    "done": h.is_done(),
                    "cancelled": h.is_cancelled(),
                    "elapsed_seconds": time.time() - h.t0,
                    "result": h.result and {
                        "text": h.result.text,
                        "cost_usd": h.result.cost_usd,
                        "passed": h.result.passed,
                        "critic_score": h.result.critic_score,
                    },
                })
            return out

    # --- delegation -----------------------------------------------------

    def submit_child(self, parent: RunHandle, request: RunRequest) -> RunHandle:
        """Spawn a sub-run; emit subrun events on the parent so a coordination
        engine sees the tree shape."""
        request.parent_id = parent.run_id
        child = self.submit(request)
        parent.emit(make(
            SUBRUN_STARTED, parent.run_id,
            child_id=child.run_id, task=request.task,
        ))

        def watch() -> None:
            child.wait()
            result = child.result
            parent.emit(make(
                SUBRUN_COMPLETED, parent.run_id,
                child_id=child.run_id,
                text=result.text if result else "",
                cost_usd=result.cost_usd if result else 0.0,
                cancelled=result.cancelled if result else False,
                error=result.error if result else None,
            ))

        threading.Thread(target=watch, daemon=True).start()
        return child

    # --- execution ------------------------------------------------------

    def _execute(self, handle: RunHandle) -> None:
        handle.emit(make(RUN_STARTED, handle.run_id, task=handle.request.task,
                         parent_id=handle.request.parent_id))
        try:
            if handle.request.skills:
                hits = self.skills.search(handle.request.task, k=3)
                if hits:
                    handle.emit(make(
                        SKILLS_LOADED, handle.run_id,
                        skills=[{"name": s.name, "description": s.description} for s in hits],
                    ))

            agent = self.agent_factory(self, handle)

            if handle.is_cancelled():
                self._finalize_cancelled(handle, "")
                return

            text = agent.chat(handle.request.task, max_iterations=handle.request.max_iterations)

            if handle.is_cancelled():
                self._finalize_cancelled(handle, text)
                return

            critic_score = getattr(agent, "last_critic_score", None)
            if critic_score is not None:
                handle.emit(make(CRITIC_SCORE, handle.run_id, score=critic_score))

            usage = getattr(agent, "usage", None)
            model = getattr(agent, "model", "")
            usage_d, cost = _summarize_usage(usage, model)
            handle.emit(make(USAGE, handle.run_id, **usage_d, model=model, cost_usd=cost))

            if handle.request.reflect:
                self._maybe_reflect(handle, agent, text)

            handle.emit(make(DONE, handle.run_id, text=text, cost_usd=cost))
            handle.finish(RunResult(
                run_id=handle.run_id,
                text=text,
                critic_score=critic_score,
                usage=usage_d,
                cost_usd=cost,
                elapsed_seconds=time.time() - handle.t0,
                metadata=dict(handle.request.metadata),
            ))
        except Exception as e:
            handle.emit(make(ERROR, handle.run_id, error=f"{type(e).__name__}: {e}"))
            handle.finish(RunResult(
                run_id=handle.run_id,
                text="",
                error=f"{type(e).__name__}: {e}",
                elapsed_seconds=time.time() - handle.t0,
                metadata=dict(handle.request.metadata),
            ))

    def _finalize_cancelled(self, handle: RunHandle, text: str) -> None:
        handle.emit(make(CANCELLED, handle.run_id))
        handle.finish(RunResult(
            run_id=handle.run_id,
            text=text,
            cancelled=True,
            elapsed_seconds=time.time() - handle.t0,
            metadata=dict(handle.request.metadata),
        ))

    def _maybe_reflect(self, handle: RunHandle, agent: Any, text: str) -> None:
        complete = getattr(agent, "complete", None)
        if not callable(complete):
            return
        reflector = Reflector(self.memory, complete=complete)
        r = reflector.reflect(handle.request.task, text)
        if r is not None:
            handle.emit(make(REFLECTION, handle.run_id, text=r.text))


# ---------------------------------------------------------------------------


def _summarize_usage(usage: Any, model: str) -> tuple[dict[str, int], float]:
    if usage is None:
        return {}, 0.0
    d = {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "turns": getattr(usage, "turns", 0),
    }
    cost_fn = getattr(usage, "cost_usd", None)
    cost = float(cost_fn(model)) if callable(cost_fn) and model else 0.0
    return d, cost


def _default_tracer(path: Path):
    try:
        from learner.traces import TraceLogger
    except ImportError:
        return None
    return TraceLogger(path)


def _default_agent_factory(runtime: "Runtime", handle: RunHandle):
    # Lazy import to avoid a top-level dependency on the SDK in tests.
    from agi.agent import Agent
    skill_prompt = ""
    if handle.request.skills:
        skill_prompt = runtime.skills.render_prompt(handle.request.task, k=3)
    return Agent(
        memory=runtime.memory,
        tracer=runtime.traces,
        on_event=handle.emit,
        runtime=runtime,
        run_id=handle.run_id,
        extra_system_prompt=skill_prompt,
        verbose=False,
    )
