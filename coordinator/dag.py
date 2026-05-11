"""Tiny DAG executor that drives an agi.Runtime.

A `Node` represents one atomic agent task. `DAG` is a set of nodes with
declared dependencies. `run(dag, runtime)` executes the DAG in
topological order, opening a fresh session per node, threading parent
outputs into child prompts via `{deps}` substitution.

This is intentionally minimal — it's a *demonstration* of the contract,
not a production scheduler. A real coordination engine would add:
parallel layers, retries with exponential backoff, distributed locking,
deadlines, persistence, and so on. The contract it speaks to the runtime
(open_session → chat → close_session, plus events) is the stable thing.

Example
-------
    from agi import Runtime, Budget
    from coordinator import DAG, Node, run

    rt = Runtime()
    dag = DAG([
        Node("plan",   role="planner",  prompt="How would I {ask}?"),
        Node("do",     role="executor", prompt="Carry out: {plan}",  deps=["plan"]),
        Node("grade",  role="critic",   prompt="Did this satisfy {ask}? {do}",
             deps=["plan", "do"]),
    ])
    results = run(dag, rt, inputs={"ask": "summarize README.md in 3 bullets"},
                  budget=Budget(max_usd=0.20))
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from agi.budget import Budget
from agi.runtime import Runtime


@dataclass
class Node:
    name: str
    role: str = "general"
    prompt: str = ""
    deps: list[str] = field(default_factory=list)
    model: str | None = None
    tags: list[str] = field(default_factory=list)


class DAG:
    def __init__(self, nodes: list[Node]) -> None:
        seen: set[str] = set()
        for n in nodes:
            if n.name in seen:
                raise ValueError(f"duplicate node name: {n.name}")
            seen.add(n.name)
        self.nodes: dict[str, Node] = {n.name: n for n in nodes}
        # Validate edges.
        for n in nodes:
            for dep in n.deps:
                if dep not in self.nodes:
                    raise ValueError(f"node {n.name!r} depends on unknown {dep!r}")

    def topo_order(self) -> list[Node]:
        indeg: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            indeg[n.name]  # touch
            for d in n.deps:
                indeg[n.name] += 1
        ready = deque(n for n, d in indeg.items() if d == 0)
        children: dict[str, list[str]] = defaultdict(list)
        for n in self.nodes.values():
            for d in n.deps:
                children[d].append(n.name)
        order: list[Node] = []
        while ready:
            name = ready.popleft()
            order.append(self.nodes[name])
            for c in children[name]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("cycle detected in DAG")
        return order


def _format_prompt(template: str, inputs: dict[str, Any], outputs: dict[str, str]) -> str:
    """{name} substitution against inputs ∪ outputs. Missing → kept literal.

    Deliberately cheap — no jinja, no escapes — because the DAG is meant
    to be glue, not a templating runtime.
    """
    bag = {**inputs, **outputs}
    out = template
    for k, v in bag.items():
        out = out.replace("{" + k + "}", str(v))
    return out


@dataclass
class NodeResult:
    name: str
    session_id: str
    text: str
    cost_usd: float
    stop_reason: str
    critic_score: float | None


def run(
    dag: DAG,
    runtime: Runtime,
    inputs: dict[str, Any] | None = None,
    budget: Budget | None = None,
    on_node_start=None,
    on_node_finish=None,
) -> list[NodeResult]:
    """Run a DAG against a runtime. Returns results in topological order.

    `budget` (if set) is applied to *each* node's session, not the whole
    DAG. Production coordinators usually want both per-node and
    per-DAG ceilings; we keep just the per-node one for now.
    """
    inputs = inputs or {}
    outputs: dict[str, str] = {}
    results: list[NodeResult] = []
    for node in dag.topo_order():
        prompt = _format_prompt(node.prompt, inputs, outputs)
        if on_node_start:
            on_node_start(node, prompt)
        session = runtime.open_session(
            role=node.role,
            model=node.model,
            budget=budget,
            tags=list(node.tags) + [f"dag:{node.name}"],
        )
        try:
            chat_result = session.chat(prompt)
        finally:
            runtime.close_session(session.id)
        outputs[node.name] = chat_result["text"]
        nr = NodeResult(
            name=node.name,
            session_id=chat_result["session_id"],
            text=chat_result["text"],
            cost_usd=chat_result["usage"]["cost_usd"],
            stop_reason=chat_result["stop_reason"],
            critic_score=chat_result.get("critic_score"),
        )
        results.append(nr)
        if on_node_finish:
            on_node_finish(node, nr)
    return results
