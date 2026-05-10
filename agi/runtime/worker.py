"""Worker that executes tasks against the runtime.

A `Worker` owns an `Agent` instance (or a per-task Agent depending on
configuration) and pulls tasks from a queue. It emits events to the bus as
the task progresses. The worker is the bridge between the abstract `Task`
in the store and a concrete LLM call.

The worker supports several `kind` values:
- `chat`        — full agent turn returning final text
- `plan`        — decompose into a GraphSpec
- `critique`    — score a candidate response
- `skill.invoke`— run a named skill with args
- `tool`        — single tool invocation (bypass LLM; e.g., read_file)
- `noop`        — synchronously succeed with the provided result; useful in tests

Workers are intentionally simple: one thread pulls tasks, runs them
synchronously, and writes back. A coordination engine that needs parallelism
spawns multiple workers.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from typing import Any, Callable

from agi.runtime.events import EventBus
from agi.runtime.tasks import Task, TaskSpec, TaskStatus, TaskStore


HandlerFn = Callable[["WorkerContext", Task], Any]


class WorkerContext:
    """Per-worker dependencies passed to handlers. Held weakly by the worker."""

    def __init__(self, bus: EventBus, store: TaskStore, registry: dict[str, HandlerFn]) -> None:
        self.bus = bus
        self.store = store
        self.registry = registry

    def submit_child(self, parent: Task, spec: TaskSpec) -> Task:
        spec.parent_id = parent.id
        child = self.store.submit(spec)
        self.bus.publish(f"task.{child.id}", "task.queued", {"task": child.to_dict()})
        return child


class Worker:
    def __init__(
        self,
        store: TaskStore,
        bus: EventBus,
        registry: dict[str, HandlerFn],
        *,
        name: str = "worker-0",
    ) -> None:
        self.store = store
        self.bus = bus
        self.registry = registry
        self.name = name
        self._inbox: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.ctx = WorkerContext(bus=bus, store=store, registry=registry)

    def enqueue(self, task_id: str) -> None:
        self._inbox.put(task_id)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, *, drain: bool = False) -> None:
        self._stop.set()
        if drain:
            self._inbox.put("__SENTINEL__")
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                task_id = self._inbox.get(timeout=0.25)
            except queue.Empty:
                continue
            if task_id == "__SENTINEL__":
                return
            self._execute(task_id)

    def _execute(self, task_id: str) -> None:
        task = self.store.get(task_id)
        if task is None:
            return
        if task.cancel_requested:
            self.store.mark_cancelled(task.id)
            self.bus.publish(f"task.{task.id}", "task.cancelled", {"task": task.to_dict()})
            return
        handler = self.registry.get(task.spec.kind)
        if handler is None:
            self.store.mark_failed(task.id, error=f"unknown task kind: {task.spec.kind}")
            self.bus.publish(f"task.{task.id}", "task.failed",
                             {"task": task.to_dict(), "error": task.error})
            return
        self.store.mark_running(task.id)
        self.bus.publish(f"task.{task.id}", "task.started", {"task": task.to_dict()})
        try:
            t0 = time.time()
            result = handler(self.ctx, task)
            elapsed = time.time() - t0
            # Handlers may return either a plain value or a dict with a
            # `result` key (and optional `usage`, `cost_usd`, `critic_score`).
            if isinstance(result, dict) and "result" in result:
                payload = result
            else:
                payload = {"result": result}
            self.store.mark_succeeded(
                task.id,
                result=payload["result"],
                usage=payload.get("usage", {}),
                cost_usd=payload.get("cost_usd", 0.0),
                critic_score=payload.get("critic_score"),
            )
            self.bus.publish(
                f"task.{task.id}",
                "task.succeeded",
                {"task": self.store.get(task.id).to_dict(), "elapsed": elapsed},
            )
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self.store.mark_failed(task.id, error=err)
            self.bus.publish(f"task.{task.id}", "task.failed",
                             {"task": self.store.get(task.id).to_dict(), "error": err})


def make_default_registry(agent_factory: Callable[[str | None], Any] | None = None) -> dict[str, HandlerFn]:
    """Build the default task-kind registry.

    `agent_factory(role)` returns an `Agent` configured for the role. Tests
    can pass a stub factory. None falls back to importing agi.Agent lazily.
    """

    def _factory(role: str | None):
        if agent_factory is not None:
            return agent_factory(role)
        from agi.agent import Agent
        return Agent(verbose=False, role=role)

    def chat_handler(ctx: WorkerContext, task: Task) -> dict:
        inp = task.spec.input
        agent = _factory(task.spec.role)
        # Stream tool-use events to the bus.
        original = getattr(agent, "_dispatch_tool_calls", None)
        if original is not None:
            def hooked(content):
                for block in content:
                    if getattr(block, "type", None) == "tool_use":
                        ctx.bus.publish(
                            f"task.{task.id}",
                            "task.tool_use",
                            {"name": block.name, "id": getattr(block, "id", None)},
                        )
                return original(content)
            agent._dispatch_tool_calls = hooked  # type: ignore[attr-defined]

        skills = inp.get("skills") or []
        if skills:
            # Inject as a system suffix the first time. Idempotent because
            # we add as a single message-role user note here.
            text = "\n\n[skills loaded: " + ", ".join(skills) + "]"
            agent.messages.append({"role": "user", "content": text})

        text = agent.chat(inp["message"], max_iterations=int(inp.get("max_iterations", 25)))
        usage = {
            "input_tokens": agent.usage.input_tokens,
            "output_tokens": agent.usage.output_tokens,
            "cache_creation_input_tokens": agent.usage.cache_creation_input_tokens,
            "cache_read_input_tokens": agent.usage.cache_read_input_tokens,
        }
        return {
            "result": {"text": text},
            "usage": usage,
            "cost_usd": getattr(agent.usage, "cost_usd", lambda *_: 0.0)(agent.model)
                if hasattr(agent.usage, "cost_usd") else 0.0,
            "critic_score": getattr(agent, "last_critic_score", None),
        }

    def plan_handler(ctx: WorkerContext, task: Task) -> dict:
        from agi.planner import propose_graph
        graph = propose_graph(goal=task.spec.input["goal"],
                              constraints=task.spec.input.get("constraints", ""),
                              agent_factory=_factory)
        return {"result": {"graph": graph}}

    def critique_handler(ctx: WorkerContext, task: Task) -> dict:
        prompt = task.spec.input["prompt"]
        response = task.spec.input["response"]
        # Cheap path: use the trained critic if available, else LLM judge.
        try:
            from learner.critic import Critic
            from pathlib import Path
            ckpt = Path.home() / ".agi" / "critics" / "addition.pt"
            if ckpt.exists():
                c = Critic.load(str(ckpt))
                score = c.predict_proba(prompt, response)
                return {"result": {"score": float(score), "explanation": "learned-critic"}}
        except Exception:
            pass
        agent = _factory("critic")
        rubric = (
            "Score the following candidate answer for correctness and "
            "trustworthiness on a 0.0-1.0 scale. Reply with JSON only: "
            '{"score": <float>, "explanation": "<one sentence>"}.\n\n'
            f"PROMPT:\n{prompt}\n\nCANDIDATE:\n{response}\n"
        )
        text = agent.chat(rubric)
        try:
            parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
            score = float(parsed.get("score", 0.5))
            expl = parsed.get("explanation", "")
        except Exception:
            score, expl = 0.5, "judge-parse-failed"
        return {"result": {"score": score, "explanation": expl}}

    def skill_handler(ctx: WorkerContext, task: Task) -> dict:
        from agi.skills.library import SkillLibrary
        skill_name = task.spec.input["skill"]
        args = task.spec.input.get("args", {})
        lib = SkillLibrary()
        skill = lib.get(skill_name)
        if skill is None:
            raise RuntimeError(f"unknown skill: {skill_name}")
        # Render the skill SOP and dispatch as a chat task.
        rendered = skill.render(args)
        agent = _factory(task.spec.role)
        text = agent.chat(rendered)
        return {"result": {"text": text, "skill": skill_name}}

    def tool_handler(ctx: WorkerContext, task: Task) -> dict:
        """Run a single registered tool by name. Bypasses the LLM."""
        from agi.memory import Memory
        from agi.tools import make_tools
        _, handlers = make_tools(Memory())
        name = task.spec.input["name"]
        args = task.spec.input.get("args", {})
        if name not in handlers:
            raise RuntimeError(f"unknown tool: {name}")
        out = handlers[name](**args)
        return {"result": {"output": out}}

    def noop_handler(ctx: WorkerContext, task: Task) -> dict:
        # Used in tests and as a synthetic node in graphs.
        return {"result": task.spec.input.get("result", "ok")}

    return {
        "chat": chat_handler,
        "plan": plan_handler,
        "critique": critique_handler,
        "skill.invoke": skill_handler,
        "tool": tool_handler,
        "noop": noop_handler,
    }
