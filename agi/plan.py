"""Plan / Subgoal abstraction.

A `Plan` is a DAG of subgoals. Each subgoal becomes a Runtime session when
executed. Dependencies expose upstream final_text into the downstream
prompt (substituted as `{{ name }}` placeholders or appended as a
"Context:" block).

This is a small, opinion-light primitive: it lets a coordinator say
"decompose, fan out, gather" without the runtime growing planner
opinions of its own. Coordinators that prefer a different decomposition
strategy can build it on top of `Runtime.submit` directly.

Cycle detection is naive (DFS); plans should be small.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from agi.budget import Budget
from agi.runtime import Runtime


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][\w]*)\s*\}\}")


def plan_from_dict(d: dict) -> "Plan":
    """Build a Plan from a JSON-friendly dict.

    Schema:
      { "name": "...", "subgoals": [
          { "name": "...", "prompt": "...",
            "depends_on": [...], "max_iterations": 25,
            "budget": { "max_cost_usd": 0.5, ... },
            "tags": {...} },
          ...
      ] }
    """
    from agi.budget import Budget  # local to avoid circular at module load

    def budget_from(b):
        if not b:
            return None
        return Budget(
            max_cost_usd=b.get("max_cost_usd"),
            max_tokens=b.get("max_tokens"),
            max_iterations=b.get("max_iterations"),
            max_wall_seconds=b.get("max_wall_seconds"),
        )

    subgoals = [
        Subgoal(
            name=sg["name"],
            prompt=sg["prompt"],
            depends_on=list(sg.get("depends_on", [])),
            budget=budget_from(sg.get("budget")),
            max_iterations=int(sg.get("max_iterations", 25)),
            tags=dict(sg.get("tags", {})),
        )
        for sg in d.get("subgoals", [])
    ]
    return Plan(name=d["name"], subgoals=subgoals)


@dataclass
class Subgoal:
    name: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    budget: Optional[Budget] = None
    max_iterations: int = 25
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SubgoalResult:
    name: str
    session_id: str
    status: str
    final_text: str
    cost_usd: float
    elapsed_s: float
    error: str = ""


@dataclass
class Plan:
    name: str
    subgoals: list[Subgoal]

    def __post_init__(self) -> None:
        names = [g.name for g in self.subgoals]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate subgoal name in plan {self.name!r}")
        name_set = set(names)
        for g in self.subgoals:
            for dep in g.depends_on:
                if dep not in name_set:
                    raise ValueError(
                        f"subgoal {g.name!r} depends on unknown {dep!r}"
                    )
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        graph = {g.name: list(g.depends_on) for g in self.subgoals}
        visiting: set[str] = set()
        done: set[str] = set()

        def dfs(node: str) -> None:
            if node in done:
                return
            if node in visiting:
                raise ValueError(f"cycle in plan {self.name!r} at {node!r}")
            visiting.add(node)
            for dep in graph[node]:
                dfs(dep)
            visiting.discard(node)
            done.add(node)

        for n in graph:
            dfs(n)


def execute_plan(
    runtime: Runtime,
    plan: Plan,
    *,
    timeout: Optional[float] = None,
    parent_id: Optional[str] = None,
    stop_on_error: bool = False,
) -> dict[str, SubgoalResult]:
    """Execute a plan against a runtime. Returns name -> result.

    Independent subgoals fan out concurrently (bounded by the runtime's
    max_concurrent). Dependent subgoals start when all upstreams finish.
    """
    results: dict[str, SubgoalResult] = {}
    by_name = {g.name: g for g in plan.subgoals}
    pending = set(by_name)
    in_flight: dict[str, str] = {}  # name -> session_id
    started_at: dict[str, float] = {}

    while pending or in_flight:
        # Launch any subgoals whose deps are satisfied.
        for name in list(pending):
            g = by_name[name]
            if any(d not in results for d in g.depends_on):
                continue
            if stop_on_error and any(
                results[d].status not in ("ok",) for d in g.depends_on
            ):
                # Upstream failed; mark downstream as skipped.
                results[name] = SubgoalResult(
                    name=name,
                    session_id="",
                    status="skipped",
                    final_text="",
                    cost_usd=0.0,
                    elapsed_s=0.0,
                    error="upstream failed",
                )
                pending.discard(name)
                continue
            prompt = _substitute(g.prompt, results)
            sid = runtime.submit(
                prompt,
                budget=g.budget,
                max_iterations=g.max_iterations,
                tags={"plan": plan.name, "subgoal": name, **g.tags},
                parent_id=parent_id,
            )
            in_flight[name] = sid
            started_at[name] = time.time()
            pending.discard(name)

        if not in_flight:
            break

        # Wait for at least one in-flight to finish.
        # We poll with a short timeout to keep the loop responsive.
        deadline = (time.time() + timeout) if timeout else None
        finished_any = False
        while not finished_any:
            for name, sid in list(in_flight.items()):
                rec = runtime.status(sid)
                if rec["status"] in ("pending", "running"):
                    continue
                results[name] = SubgoalResult(
                    name=name,
                    session_id=sid,
                    status=rec["status"],
                    final_text=rec["final_text"],
                    cost_usd=rec["total_cost_usd"],
                    elapsed_s=time.time() - started_at[name],
                    error=rec.get("error", ""),
                )
                in_flight.pop(name)
                finished_any = True
            if finished_any:
                break
            if deadline is not None and time.time() > deadline:
                # Timeout: mark remaining as timed-out and bail.
                for name, sid in in_flight.items():
                    runtime.cancel(sid)
                    results[name] = SubgoalResult(
                        name=name,
                        session_id=sid,
                        status="timeout",
                        final_text="",
                        cost_usd=0.0,
                        elapsed_s=time.time() - started_at[name],
                        error="plan timeout",
                    )
                return results
            time.sleep(0.05)

    return results


def _substitute(prompt: str, results: dict[str, SubgoalResult]) -> str:
    """Replace `{{ name }}` with the corresponding subgoal's final_text.

    Unknown placeholders are left in place — the model can complain about
    them, which is more debuggable than silently dropping context.
    """
    def repl(m: re.Match) -> str:
        name = m.group(1)
        r = results.get(name)
        if r is None:
            return m.group(0)
        return r.final_text

    return _PLACEHOLDER.sub(repl, prompt)
