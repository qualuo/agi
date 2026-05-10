"""Coordination engine.

`Coordinator` drives a runtime — either an in-process `Runtime` instance or
a remote runtime over HTTP via `RuntimeClient`. The coordinator's job is to
take an *external* goal and turn it into a *finished outcome* by:

1. Asking the runtime to plan (kind=`plan`).
2. Submitting the resulting task graph (POST /graphs).
3. Streaming graph events for live observability.
4. Verifying the result with a critic node (or LLM judge).
5. If verification fails and the budget allows, revising the plan and
   re-dispatching.

The coordinator never executes work itself; the runtime is the worker.
This separation is what makes the system composable: multiple coordinators
can target the same runtime, and one coordinator can target many runtimes.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from agi.runtime.graph import GraphResult, GraphSpec, NodeSpec
from agi.runtime.tasks import TaskSpec


@dataclass
class CoordinatorReport:
    goal: str
    final_text: str
    graph_id: str | None
    iterations: int
    status: str
    total_cost_usd: float
    total_tokens: int
    elapsed: float
    critic_score: float | None = None
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeClient:
    """HTTP client for the runtime. Mirrors the in-process Runtime API."""

    def __init__(self, base_url: str = "http://127.0.0.1:7777", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None,
                 *, raw: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        resp = urllib.request.urlopen(req, timeout=self.timeout)
        if raw:
            return resp
        with resp:
            return json.loads(resp.read().decode("utf-8") or "{}")

    def capabilities(self) -> dict:
        return self._request("GET", "/capabilities")

    def submit_task(self, spec: TaskSpec) -> str:
        body = {
            "kind": spec.kind, "input": spec.input,
            "role": spec.role, "dedup_key": spec.dedup_key,
            "parent_id": spec.parent_id,
            "budget_tokens": spec.budget_tokens,
            "budget_seconds": spec.budget_seconds, "tags": spec.tags,
        }
        return self._request("POST", "/tasks", body)["task_id"]

    def task(self, task_id: str) -> dict:
        return self._request("GET", f"/tasks/{task_id}")

    def submit_graph(self, graph: GraphSpec) -> str:
        return self._request("POST", "/graphs", graph.to_dict())["graph_id"]

    def stream(self, prefix: str, *, timeout: float = 600.0) -> Iterable[dict]:
        """SSE stream of events under `prefix`."""
        url = f"{self.base_url}/{prefix}/stream" if prefix.startswith(("tasks/", "graphs/")) else f"{self.base_url}/events"
        if prefix.startswith(("task.", "graph.")):
            kind, _, ident = prefix.partition(".")
            url = f"{self.base_url}/{kind}s/{ident}/stream"
        req = urllib.request.Request(url, method="GET",
                                     headers={"Accept": "text/event-stream"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        event: dict[str, str] = {}
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith(":"):
                continue
            if line == "":
                if "data" in event:
                    try:
                        yield json.loads(event["data"])
                    except json.JSONDecodeError:
                        pass
                event = {}
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                event[key.strip()] = val.strip()


class Coordinator:
    """Reference orchestrator. Talks to a Runtime (in-proc or remote)."""

    def __init__(self, runtime, *, max_iterations: int = 2,
                 verify: bool = True, verify_threshold: float = 0.6) -> None:
        # runtime is either agi.runtime.server.Runtime or RuntimeClient
        self.runtime = runtime
        self.max_iterations = max_iterations
        self.verify = verify
        self.verify_threshold = verify_threshold

    # ---- runtime-agnostic adapters ----
    def _submit_graph(self, graph: GraphSpec) -> str:
        if isinstance(self.runtime, RuntimeClient):
            return self.runtime.submit_graph(graph)
        return self.runtime.graph.submit(graph)

    def _wait_graph(self, graph_id: str, *, timeout: float = 600.0,
                    on_event=None) -> GraphResult:
        if isinstance(self.runtime, RuntimeClient):
            for ev in self.runtime.stream(f"graph.{graph_id}", timeout=timeout):
                if on_event:
                    on_event(ev)
                if ev.get("kind") in ("graph.completed", "graph.failed", "graph.cancelled"):
                    return GraphResult(**ev["payload"]["result"])
            raise TimeoutError(f"graph {graph_id} stream ended without terminal event")
        # in-process
        for event in self.runtime.bus.stream(f"graph.{graph_id}", timeout=0.5):
            if on_event:
                on_event({"kind": event.kind, "ts": event.ts,
                          "payload": event.payload, "topic": event.topic})
            if event.kind in ("graph.completed", "graph.failed", "graph.cancelled"):
                return GraphResult(**event.payload["result"])
        raise TimeoutError(f"graph {graph_id} stream ended without terminal event")

    def _plan(self, goal: str, constraints: str = "") -> GraphSpec:
        """Ask the runtime to produce a GraphSpec for the goal."""
        spec = TaskSpec(kind="plan", input={"goal": goal, "constraints": constraints},
                        role="planner")
        if isinstance(self.runtime, RuntimeClient):
            tid = self.runtime.submit_task(spec)
            # Poll for completion.
            for _ in range(600):
                t = self.runtime.task(tid)
                if t["status"] in ("succeeded", "failed", "cancelled"):
                    if t["status"] != "succeeded":
                        raise RuntimeError(f"plan task failed: {t.get('error')}")
                    return GraphSpec.from_dict(t["result"]["graph"])
                time.sleep(0.5)
            raise TimeoutError("plan task did not complete in time")
        else:
            task = self.runtime.submit_task(spec)
            while not task.status.terminal:
                time.sleep(0.1)
                task = self.runtime.store.get(task.id)
            if task.status.value != "succeeded":
                raise RuntimeError(f"plan task failed: {task.error}")
            return GraphSpec.from_dict(task.result["graph"])

    # ---- public API ----
    def run(self, goal: str, *, constraints: str = "",
            on_event=None) -> CoordinatorReport:
        t0 = time.time()
        events_seen: list[dict] = []
        def _on(ev):
            events_seen.append(ev)
            if on_event:
                on_event(ev)

        last_graph_id: str | None = None
        last_result: GraphResult | None = None
        iterations = 0
        revised_goal = goal
        revised_constraints = constraints
        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration
            graph = self._plan(revised_goal, revised_constraints)
            # Optionally append a final critique node if the planner didn't.
            if self.verify and not any(n.kind == "critique" for n in graph.nodes):
                # Find a terminal node (no node depends on it) to verify.
                produced = {n.id for n in graph.nodes}
                consumed = {d for n in graph.nodes for d in n.depends_on}
                terminal_ids = list(produced - consumed) or [graph.nodes[-1].id]
                # Verify the last terminal node's text output.
                term = terminal_ids[-1]
                graph.nodes.append(NodeSpec(
                    id="_critic_gate",
                    kind="critique",
                    input={"prompt": goal, "response": "${" + term + ".text}"},
                    depends_on=[term],
                    on_failure="skip",
                ))
            gid = self._submit_graph(graph)
            last_graph_id = gid
            last_result = self._wait_graph(gid, on_event=_on)
            critic_score: float | None = None
            critic_out = last_result.outputs.get("_critic_gate")
            if isinstance(critic_out, dict) and "score" in critic_out:
                critic_score = float(critic_out["score"])
            # Accept if no critic gate or score >= threshold.
            ok = (last_result.status == "succeeded"
                  and (critic_score is None or critic_score >= self.verify_threshold))
            if ok:
                final_text = _extract_terminal_text(last_result)
                return CoordinatorReport(
                    goal=goal, final_text=final_text, graph_id=gid,
                    iterations=iteration, status="succeeded",
                    total_cost_usd=last_result.total_cost_usd,
                    total_tokens=last_result.total_tokens,
                    elapsed=time.time() - t0,
                    critic_score=critic_score,
                    events=events_seen,
                )
            # Revise: hand the critic's explanation back as a constraint.
            if isinstance(critic_out, dict) and critic_out.get("explanation"):
                revised_constraints = (
                    constraints + "\nPrevious attempt was judged insufficient: "
                    + str(critic_out["explanation"])
                )
        # Out of iterations.
        final_text = _extract_terminal_text(last_result) if last_result else ""
        return CoordinatorReport(
            goal=goal, final_text=final_text, graph_id=last_graph_id,
            iterations=iterations, status="failed_after_revisions",
            total_cost_usd=last_result.total_cost_usd if last_result else 0.0,
            total_tokens=last_result.total_tokens if last_result else 0,
            elapsed=time.time() - t0,
            events=events_seen,
        )


def _extract_terminal_text(result: GraphResult | None) -> str:
    if result is None:
        return ""
    # Pick the most recent text-bearing terminal output.
    for nid, out in reversed(list(result.outputs.items())):
        if nid == "_critic_gate":
            continue
        if isinstance(out, dict) and isinstance(out.get("text"), str):
            return out["text"]
    return ""
