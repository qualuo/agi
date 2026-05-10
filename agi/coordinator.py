"""Coordinator — DAG of jobs over the Runtime.

The Runtime gives you sessions and jobs. The Coordinator gives you a way to
*compose* jobs: declare a graph of nodes, each a (role, prompt-template,
dependencies), and the coordinator runs ready nodes in parallel, feeds
upstream outputs into downstream prompts, and reports the rolled-up result.

This is the simplest useful coordination pattern. Real workflow systems
(Airflow, Temporal, Prefect, etc.) can drive the Runtime directly via its
Python API or the HTTP control plane in server.py — this coordinator is for
in-process orchestration and as a reference for what a coordinator needs.

Example:

    from agi.runtime import Runtime
    from agi.coordinator import Coordinator, Node

    rt = Runtime()
    plan = Coordinator(rt, [
        Node("research",  "Find three recent papers on retrieval-augmented "
                          "generation. Return as a numbered list."),
        Node("summarize", "Summarize each paper in two sentences:\\n{research}",
                          depends_on=["research"]),
        Node("critique",  "Identify weaknesses in this summary:\\n{summarize}",
                          depends_on=["summarize"], role="critic"),
    ])
    results = plan.run()
    print(results["critique"].output)
"""
from __future__ import annotations

import string
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agi.runtime import JobRecord, Runtime


@dataclass
class Node:
    """One step in the plan.

    `prompt` may contain `{name}` placeholders that resolve to the `output`
    of the named upstream node at run time.
    """
    name: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    role: str | None = None
    budget_usd: float | None = None
    max_iterations: int = 15
    metadata: dict = field(default_factory=dict)


class CoordinatorError(RuntimeError):
    pass


class Coordinator:
    def __init__(
        self,
        runtime: "Runtime",
        plan: list[Node],
        *,
        on_node_done: callable | None = None,
    ) -> None:
        self.runtime = runtime
        self.nodes: dict[str, Node] = {}
        for n in plan:
            if n.name in self.nodes:
                raise CoordinatorError(f"duplicate node name {n.name!r}")
            self.nodes[n.name] = n
        self._validate_dag()
        self.on_node_done = on_node_done
        self._results: dict[str, "JobRecord"] = {}
        self._lock = threading.Lock()

    def _validate_dag(self) -> None:
        # All deps must reference existing nodes.
        for n in self.nodes.values():
            for d in n.depends_on:
                if d not in self.nodes:
                    raise CoordinatorError(f"node {n.name!r} depends on unknown node {d!r}")
        # No cycles (Kahn's algorithm).
        indeg = {name: 0 for name in self.nodes}
        for n in self.nodes.values():
            for d in n.depends_on:
                indeg[n.name] += 1
        ready = [name for name, k in indeg.items() if k == 0]
        seen = 0
        while ready:
            name = ready.pop()
            seen += 1
            for other in self.nodes.values():
                if name in other.depends_on:
                    indeg[other.name] -= 1
                    if indeg[other.name] == 0:
                        ready.append(other.name)
        if seen != len(self.nodes):
            raise CoordinatorError("plan has a cycle")

    def run(self, timeout: float | None = None) -> dict[str, "JobRecord"]:
        """Execute the plan. Blocks until all nodes finish, fail, or the
        timeout elapses. Returns the JobRecord for each node by name."""
        deadline = time.time() + timeout if timeout else None
        pending = set(self.nodes)
        in_flight: dict[str, str] = {}  # node_name -> job_id
        sessions: dict[str, str] = {}   # node_name -> session_id (one session per node)

        while pending or in_flight:
            # Schedule ready nodes
            for name in list(pending):
                n = self.nodes[name]
                if all(dep in self._results and self._results[dep].status == "succeeded" for dep in n.depends_on):
                    prompt = self._render_prompt(n)
                    s = self.runtime.create_session(role=n.role, metadata={"coordinator_node": name})
                    sessions[name] = s.id
                    job = self.runtime.submit(
                        s.id,
                        prompt,
                        budget_usd=n.budget_usd,
                        max_iterations=n.max_iterations,
                        metadata={"coordinator_node": name, **n.metadata},
                    )
                    in_flight[name] = job.id
                    pending.discard(name)
                elif any(dep in self._results and self._results[dep].status != "succeeded" for dep in n.depends_on):
                    # Upstream failed — short-circuit this node as failed.
                    self._results[name] = _synthetic_failed(name, n, "upstream node failed")
                    pending.discard(name)

            if not in_flight:
                # Either everything is done or the remaining pending nodes are
                # blocked on already-failed deps that we just short-circuited.
                if not pending:
                    break
                continue

            # Wait for any in-flight job to finish, with a per-iteration tick
            # so we can re-check the deadline.
            tick = 0.25 if deadline is None else min(0.25, max(0.0, deadline - time.time()))
            done_now: list[str] = []
            for name, jid in list(in_flight.items()):
                try:
                    rec = self.runtime.await_job(jid, timeout=tick)
                except TimeoutError:
                    continue
                self._results[name] = rec
                done_now.append(name)
                if self.on_node_done:
                    try:
                        self.on_node_done(name, rec)
                    except Exception:
                        pass
            for name in done_now:
                in_flight.pop(name, None)

            if deadline is not None and time.time() > deadline:
                # Cancel still-running jobs and surface a partial result.
                for name, jid in in_flight.items():
                    self.runtime.cancel(jid)
                    self._results[name] = _synthetic_failed(name, self.nodes[name], "coordinator timeout")
                in_flight.clear()
                pending.clear()
                break

        return dict(self._results)

    def _render_prompt(self, node: Node) -> str:
        """Substitute {dep_name} placeholders with the upstream node's output."""
        if not node.depends_on:
            return node.prompt
        mapping = {dep: self._results[dep].output for dep in node.depends_on}
        # Tolerate stray { } in user prompts by using a Template-like substitution
        # only for explicit dep names; everything else stays literal.
        out = node.prompt
        for dep, value in mapping.items():
            out = out.replace("{" + dep + "}", value)
        return out


def _synthetic_failed(name: str, node: Node, reason: str) -> "JobRecord":
    from agi.runtime import JobRecord
    now = time.time()
    return JobRecord(
        id=f"synthetic_{name}",
        session_id="",
        prompt=node.prompt,
        status="failed",
        created_ts=now,
        started_ts=now,
        finished_ts=now,
        error=reason,
        metadata={"coordinator_node": name, "synthetic": True},
    )
