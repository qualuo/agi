"""Typed task DAG executor.

A `GraphSpec` is a list of `NodeSpec` entries. Each node has an id, a task
kind, an input dict, and a set of `depends_on` node ids. Inputs can reference
outputs of upstream nodes using `${node_id.field}` placeholders, which the
executor resolves before dispatch.

The executor schedules nodes as their dependencies become satisfied, runs
them through the worker pool, and emits events on the bus. It's the
in-process counterpart of what a coordination engine does across a fleet —
but the same graph format is what an external coordinator sends in over HTTP.

Failure policy is per-node: `on_failure ∈ {fail_graph, skip, retry:N}`.
Default is `fail_graph`. Cancellation propagates: if the graph is cancelled,
in-flight node tasks receive cancel requests.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

from agi.runtime.events import EventBus
from agi.runtime.tasks import TaskSpec, TaskStatus, TaskStore
from agi.runtime.worker import Worker


@dataclass
class NodeSpec:
    id: str
    kind: str
    input: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    on_failure: str = "fail_graph"  # fail_graph | skip | retry:N
    role: str | None = None
    budget_tokens: int | None = None
    budget_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSpec:
    nodes: list[NodeSpec]
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, d: dict) -> "GraphSpec":
        return cls(
            name=d.get("name"),
            nodes=[NodeSpec(**n) for n in d["nodes"]],
        )


@dataclass
class GraphResult:
    graph_id: str
    name: str | None
    status: str  # succeeded | failed | cancelled
    outputs: dict[str, Any]
    errors: dict[str, str]
    elapsed: float
    total_cost_usd: float
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z0-9_.-]+)\}")


def _resolve(value: Any, outputs: dict[str, Any]) -> Any:
    """Replace ${node.field} references with concrete values from outputs."""
    if isinstance(value, str):
        # If the whole string is one placeholder, substitute the object directly.
        m = _PLACEHOLDER.fullmatch(value.strip())
        if m:
            return _lookup(m.group(1), outputs, default=value)
        # Otherwise do string substitution.
        def repl(m: re.Match) -> str:
            v = _lookup(m.group(1), outputs, default=m.group(0))
            return v if isinstance(v, str) else json.dumps(v, default=str)
        return _PLACEHOLDER.sub(repl, value)
    if isinstance(value, list):
        return [_resolve(v, outputs) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, outputs) for k, v in value.items()}
    return value


def _lookup(path: str, outputs: dict[str, Any], default: Any) -> Any:
    parts = path.split(".")
    head = outputs.get(parts[0])
    if head is None:
        return default
    cur: Any = head
    for p in parts[1:]:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


class GraphExecutor:
    """Schedules a graph against a pool of workers."""

    def __init__(
        self,
        store: TaskStore,
        bus: EventBus,
        workers: list[Worker],
    ) -> None:
        self.store = store
        self.bus = bus
        self.workers = workers
        self._next_worker = 0
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, graph: GraphSpec) -> str:
        graph_id = uuid.uuid4().hex[:16]
        self.bus.publish(f"graph.{graph_id}", "graph.submitted",
                         {"graph_id": graph_id, "graph": graph.to_dict()})
        thread = threading.Thread(target=self._run, args=(graph_id, graph), daemon=True)
        thread.start()
        return graph_id

    def run(self, graph: GraphSpec, *, timeout: float | None = None) -> GraphResult:
        """Blocking convenience: submit and wait for the result."""
        graph_id = self.submit(graph)
        return self.wait(graph_id, timeout=timeout)

    def wait(self, graph_id: str, *, timeout: float | None = None) -> GraphResult:
        deadline = time.time() + timeout if timeout else None
        for event in self.bus.stream(f"graph.{graph_id}", timeout=0.5):
            if event.kind in ("graph.completed", "graph.failed", "graph.cancelled"):
                return GraphResult(**event.payload["result"])
            if deadline and time.time() > deadline:
                raise TimeoutError(f"graph {graph_id} did not complete within {timeout}s")
        raise TimeoutError(f"graph {graph_id} event stream ended unexpectedly")

    def cancel(self, graph_id: str) -> None:
        with self._lock:
            self._cancelled.add(graph_id)
        self.bus.publish(f"graph.{graph_id}", "graph.cancelled_requested", {})

    def _is_cancelled(self, graph_id: str) -> bool:
        with self._lock:
            return graph_id in self._cancelled

    def _pick_worker(self) -> Worker:
        with self._lock:
            w = self.workers[self._next_worker % len(self.workers)]
            self._next_worker += 1
        return w

    def _run(self, graph_id: str, graph: GraphSpec) -> None:
        t0 = time.time()
        # Validate.
        node_by_id = {n.id: n for n in graph.nodes}
        if len(node_by_id) != len(graph.nodes):
            self._finalize(graph_id, graph, status="failed",
                           outputs={}, errors={"_graph": "duplicate node ids"},
                           t0=t0)
            return
        for n in graph.nodes:
            for dep in n.depends_on:
                if dep not in node_by_id:
                    self._finalize(graph_id, graph, status="failed",
                                   outputs={}, errors={n.id: f"unknown dep {dep}"},
                                   t0=t0)
                    return

        outputs: dict[str, Any] = {}
        errors: dict[str, str] = {}
        remaining = {n.id for n in graph.nodes}
        running: dict[str, str] = {}  # node_id -> task_id
        retries: dict[str, int] = {n.id: 0 for n in graph.nodes}
        total_cost = 0.0
        total_tokens = 0

        def deps_satisfied(n: NodeSpec) -> bool:
            return all(d in outputs for d in n.depends_on)

        def ready_nodes() -> list[NodeSpec]:
            return [
                node_by_id[i] for i in remaining
                if i not in running and deps_satisfied(node_by_id[i])
            ]

        def dispatch(n: NodeSpec) -> str:
            inp = _resolve(n.input, outputs)
            spec = TaskSpec(
                kind=n.kind,
                input=inp,
                role=n.role,
                budget_tokens=n.budget_tokens,
                budget_seconds=n.budget_seconds,
                parent_id=None,
                tags=[f"graph:{graph_id}", f"node:{n.id}"],
            )
            task = self.store.submit(spec)
            self._pick_worker().enqueue(task.id)
            self.bus.publish(f"graph.{graph_id}", "graph.node_ready",
                             {"graph_id": graph_id, "node_id": n.id, "task_id": task.id})
            return task.id

        # Kick off initially-ready nodes.
        for n in ready_nodes():
            running[n.id] = dispatch(n)

        # Subscribe before draining — race-free.
        sub = self.bus.subscribe(prefix=f"task.")
        try:
            while remaining:
                if self._is_cancelled(graph_id):
                    for tid in running.values():
                        self.store.mark_cancelled(tid)
                    self._finalize(graph_id, graph, status="cancelled",
                                   outputs=outputs, errors=errors, t0=t0,
                                   total_cost=total_cost, total_tokens=total_tokens)
                    return
                # Poll terminal tasks via task store (event subscription drives wake-ups
                # but polling the store gives us a simple, race-free check).
                progressed = False
                for node_id, task_id in list(running.items()):
                    t = self.store.get(task_id)
                    if t is None or not t.status.terminal:
                        continue
                    progressed = True
                    running.pop(node_id, None)
                    total_cost += t.cost_usd
                    total_tokens += sum(t.usage.values()) if t.usage else 0
                    if t.status == TaskStatus.SUCCEEDED:
                        outputs[node_id] = t.result
                        remaining.discard(node_id)
                    else:
                        node = node_by_id[node_id]
                        policy = node.on_failure or "fail_graph"
                        if policy.startswith("retry:"):
                            limit = int(policy.split(":", 1)[1])
                            if retries[node_id] < limit:
                                retries[node_id] += 1
                                running[node_id] = dispatch(node)
                                continue
                        if policy == "skip":
                            outputs[node_id] = {"skipped": True, "error": t.error}
                            remaining.discard(node_id)
                            continue
                        errors[node_id] = t.error or t.status.value
                        # On fail_graph, finalize as failed but include partial outputs.
                        self._finalize(graph_id, graph, status="failed",
                                       outputs=outputs, errors=errors, t0=t0,
                                       total_cost=total_cost, total_tokens=total_tokens)
                        return
                # Schedule newly-ready nodes.
                for n in ready_nodes():
                    running[n.id] = dispatch(n)
                if not progressed and running:
                    # Block waiting for an event so we don't busy-loop.
                    try:
                        assert sub.queue is not None
                        sub.queue.get(timeout=0.25)
                    except Exception:
                        pass
                elif not running and remaining and not progressed:
                    # Deadlock: ready set was empty but remaining is non-empty.
                    remaining_ids = list(remaining)
                    errors["_graph"] = f"unsatisfiable deps for {remaining_ids}"
                    self._finalize(graph_id, graph, status="failed",
                                   outputs=outputs, errors=errors, t0=t0,
                                   total_cost=total_cost, total_tokens=total_tokens)
                    return
        finally:
            self.bus.unsubscribe(sub)

        self._finalize(graph_id, graph, status="succeeded",
                       outputs=outputs, errors=errors, t0=t0,
                       total_cost=total_cost, total_tokens=total_tokens)

    def _finalize(self, graph_id: str, graph: GraphSpec, *, status: str,
                  outputs: dict[str, Any], errors: dict[str, str], t0: float,
                  total_cost: float = 0.0, total_tokens: int = 0) -> None:
        result = GraphResult(
            graph_id=graph_id,
            name=graph.name,
            status=status,
            outputs=outputs,
            errors=errors,
            elapsed=time.time() - t0,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
        )
        kind = {"succeeded": "graph.completed", "failed": "graph.failed",
                "cancelled": "graph.cancelled"}[status]
        self.bus.publish(f"graph.{graph_id}", kind, {"result": result.to_dict()})
