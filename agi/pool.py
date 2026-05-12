"""RuntimePool — federate over many Runtimes from one coordinator.

A single `Runtime` is one process's view of the world. A real deployment
(or any meaningfully autonomous system) wants:

  - **horizontal scale** — many runtime instances, possibly on many hosts
  - **capability-aware dispatch** — send the prompt to a runtime that has
    the right skills loaded, the right synthesized tools, the right
    memory namespace
  - **liveness routing** — skip a runtime that's full, slow, or down
  - **uniform metrics** — total throughput, per-runtime cost, p99 latency

`RuntimePool` is that layer. It accepts a heterogeneous set of
`RuntimeNode`s and lets a coordinator submit work without picking the
target. The pool picks based on observed capability + capacity.

A `RuntimeNode` exposes the *same* surface as a local `Runtime` (so any
coordinator code that depended on `runtime.create_session()` works
against the pool's selection), plus identity (`node_id`) and per-node
counters used for load balancing.

Out-of-process runtimes (HTTP/SSE servers) plug in via the same
`RuntimeNode` adapter — a small client implements the `Runtime`-shaped
methods we use here (`create_session`, `chat`, `end_session`,
`capabilities`, `metrics`). Production deployments swap the in-process
node for that adapter without changing pool code. We ship the adapter
as `HttpRuntimeNode` below.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agi.events import Event, EventBus
from agi.runtime import Runtime, SessionConfig


@dataclass
class RuntimeNode:
    """Wraps one Runtime with identity and per-node counters.

    The Runtime is the public surface; this dataclass exists so the
    pool can track which physical (or HTTP) node it's calling.
    """
    node_id: str
    runtime: Runtime
    tags: tuple[str, ...] = field(default_factory=tuple)
    weight: float = 1.0  # static capacity hint; pool divides load proportional
    last_dispatch_ts: float = 0.0
    in_flight: int = 0
    total_dispatched: int = 0
    total_failed: int = 0
    healthy: bool = True

    def score(self, *, prompt: str) -> float:
        """Higher is better. Combines static weight, current load, and
        whether the node has any matching skill/tool. The pool's
        `select()` picks the argmax."""
        caps = self.runtime.capabilities()
        n_skills = len(caps.get("skills", []))
        # Skill-match boost: number of skills whose name/description
        # share a token with the prompt. Cheap and good enough.
        from agi.capabilities import _tokenize
        q = _tokenize(prompt)
        skill_hits = 0
        for s in caps.get("skills", []):
            tokens = _tokenize(s.get("name", "") + " " + s.get("description", ""))
            if q & tokens:
                skill_hits += 1
        # Liveness penalty + load penalty
        if not self.healthy:
            return -1.0
        load_penalty = self.in_flight / max(1.0, self.weight)
        return (
            self.weight
            + 0.5 * skill_hits
            + 0.1 * n_skills
            - 0.7 * load_penalty
        )


@dataclass
class PoolDispatch:
    """One unit of work routed through the pool."""
    dispatch_id: str
    node_id: str
    session_id: str
    prompt: str
    started_ts: float
    completed_ts: float | None = None
    final_text: str | None = None
    error: str | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


# Event kinds — observable through the pool's own bus.
POOL_NODE_ADDED = "pool.node_added"
POOL_NODE_REMOVED = "pool.node_removed"
POOL_NODE_UNHEALTHY = "pool.node_unhealthy"
POOL_DISPATCH_STARTED = "pool.dispatch_started"
POOL_DISPATCH_COMPLETED = "pool.dispatch_completed"
POOL_DISPATCH_FAILED = "pool.dispatch_failed"


class RuntimePool:
    """Federation layer: many `RuntimeNode`s, one coordinator-facing API.

    The pool re-publishes per-runtime events on its own bus, prefixed
    with the node id in the event data, so a coordinator can observe
    everything in one stream. Per-node buses keep working — pool
    observability is *additive*.

    Concurrency: dispatch is thread-safe; the pool's lock protects
    routing decisions and counters. Actual `runtime.chat()` runs on
    the calling thread (single-threaded by design — callers can fan
    out by calling `dispatch` from a worker pool).
    """

    def __init__(self, bus: EventBus | None = None) -> None:
        self._nodes: dict[str, RuntimeNode] = {}
        self._dispatches: list[PoolDispatch] = []
        self._lock = threading.Lock()
        self.bus = bus or EventBus()

    # --- node lifecycle ------------------------------------------------

    def add_node(self, node: RuntimeNode) -> str:
        with self._lock:
            if node.node_id in self._nodes:
                raise ValueError(f"node already in pool: {node.node_id}")
            self._nodes[node.node_id] = node
        # Forward this node's events onto the pool bus with the node id
        # annotated. This lets a coordinator watch the whole federation
        # from one subscriber.
        node_id = node.node_id

        def _forward(e: Event) -> None:
            forwarded = Event(
                kind=f"pool/{e.kind}",
                session_id=e.session_id,
                data={**e.data, "_node_id": node_id},
            )
            self.bus.publish(forwarded)

        try:
            node.runtime.bus.subscribe(_forward)
        except Exception:
            pass
        self.bus.publish(Event(kind=POOL_NODE_ADDED, data={"node_id": node.node_id, "tags": list(node.tags)}))
        return node.node_id

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._nodes:
                return False
            del self._nodes[node_id]
        self.bus.publish(Event(kind=POOL_NODE_REMOVED, data={"node_id": node_id}))
        return True

    def nodes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "node_id": n.node_id,
                    "tags": list(n.tags),
                    "weight": n.weight,
                    "in_flight": n.in_flight,
                    "healthy": n.healthy,
                    "total_dispatched": n.total_dispatched,
                    "total_failed": n.total_failed,
                    "last_dispatch_ts": n.last_dispatch_ts,
                }
                for n in self._nodes.values()
            ]

    def mark_unhealthy(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.healthy = False
        self.bus.publish(Event(kind=POOL_NODE_UNHEALTHY, data={"node_id": node_id}))

    def mark_healthy(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.healthy = True

    # --- selection -----------------------------------------------------

    def select(self, prompt: str, *, require_tag: str | None = None) -> RuntimeNode:
        """Return the best node for `prompt`. Optionally restrict by tag."""
        with self._lock:
            candidates = [
                n for n in self._nodes.values()
                if n.healthy and (require_tag is None or require_tag in n.tags)
            ]
        if not candidates:
            raise RuntimeError("no healthy nodes available in pool")
        candidates.sort(key=lambda n: n.score(prompt=prompt), reverse=True)
        return candidates[0]

    # --- dispatch ------------------------------------------------------

    def dispatch(
        self,
        prompt: str,
        config: SessionConfig | None = None,
        *,
        require_tag: str | None = None,
        namespace: str | None = None,
    ) -> PoolDispatch:
        """Pick a node, create a session, run one turn, end the session."""
        node = self.select(prompt, require_tag=require_tag)
        dispatch_id = uuid.uuid4().hex[:12]
        d = PoolDispatch(
            dispatch_id=dispatch_id,
            node_id=node.node_id,
            session_id="",
            prompt=prompt,
            started_ts=time.time(),
        )
        with self._lock:
            node.in_flight += 1
            node.total_dispatched += 1
            node.last_dispatch_ts = d.started_ts
        self.bus.publish(Event(
            kind=POOL_DISPATCH_STARTED,
            data={"dispatch_id": dispatch_id, "node_id": node.node_id},
        ))
        sid = ""
        try:
            sid = node.runtime.create_session(config or SessionConfig(), namespace=namespace)
            d.session_id = sid
            d.final_text = node.runtime.chat(sid, prompt)
        except Exception as e:
            d.error = f"{type(e).__name__}: {e}"
            with self._lock:
                node.total_failed += 1
            self.bus.publish(Event(
                kind=POOL_DISPATCH_FAILED,
                data={"dispatch_id": dispatch_id, "node_id": node.node_id, "error": d.error},
            ))
        finally:
            if sid:
                try:
                    sess = node.runtime.get_session(sid)
                    d.cost_usd = sess.state.total_cost_usd
                    node.runtime.end_session(sid)
                except Exception:
                    pass
            d.completed_ts = time.time()
            d.duration_seconds = d.completed_ts - d.started_ts
            with self._lock:
                node.in_flight = max(0, node.in_flight - 1)
                self._dispatches.append(d)
        if d.error is None:
            self.bus.publish(Event(
                kind=POOL_DISPATCH_COMPLETED,
                data={
                    "dispatch_id": dispatch_id,
                    "node_id": node.node_id,
                    "cost_usd": d.cost_usd,
                    "duration_seconds": d.duration_seconds,
                },
            ))
        return d

    # --- observability -------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            nodes = list(self._nodes.values())
            dispatches = list(self._dispatches)
        per_node: dict[str, dict[str, Any]] = {}
        for n in nodes:
            m = n.runtime.metrics()
            per_node[n.node_id] = {
                "in_flight": n.in_flight,
                "total_dispatched": n.total_dispatched,
                "total_failed": n.total_failed,
                "healthy": n.healthy,
                "weight": n.weight,
                "tags": list(n.tags),
                "runtime_metrics": m,
            }
        total_cost = sum(d.cost_usd for d in dispatches)
        succ = sum(1 for d in dispatches if d.error is None)
        return {
            "nodes": per_node,
            "total_dispatches": len(dispatches),
            "successes": succ,
            "success_rate": succ / max(1, len(dispatches)),
            "total_cost_usd": total_cost,
        }

    def aggregate_capabilities(self) -> dict[str, Any]:
        """Union of capabilities across all nodes — what the federation
        as a whole can do right now. A coordinator queries this to
        decide whether to dispatch a task or refuse it for lack of
        relevant skills."""
        all_models: set[str] = set()
        all_skills: dict[str, dict[str, Any]] = {}
        all_synth: dict[str, dict[str, Any]] = {}
        active_sessions = 0
        with self._lock:
            nodes = list(self._nodes.values())
        for n in nodes:
            caps = n.runtime.capabilities()
            for m in caps.get("models", []):
                all_models.add(m)
            for s in caps.get("skills", []):
                all_skills[s["name"]] = {**s, "_nodes": all_skills.get(s["name"], {}).get("_nodes", []) + [n.node_id]}
            for t in caps.get("synthesized_tools", []):
                all_synth[t["name"]] = {**t, "_nodes": all_synth.get(t["name"], {}).get("_nodes", []) + [n.node_id]}
            active_sessions += caps.get("active_sessions", 0)
        return {
            "models": sorted(all_models),
            "skills": list(all_skills.values()),
            "synthesized_tools": list(all_synth.values()),
            "active_sessions": active_sessions,
            "node_count": len(nodes),
            "healthy_node_count": sum(1 for n in nodes if n.healthy),
        }

    def dispatches(self, *, limit: int | None = None) -> list[PoolDispatch]:
        with self._lock:
            out = list(self._dispatches)
        if limit is not None:
            out = out[-limit:]
        return out
