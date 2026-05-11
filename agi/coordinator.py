"""Example coordination engine.

What a coordination engine actually does:

  1. Take a goal.
  2. Decompose it into independent or dependent subtasks.
  3. Dispatch each subtask to a runtime (this one, or some pool of
     specialized runtimes).
  4. Observe runtime events; react (retry, cancel, reassign).
  5. Aggregate sub-results into a final answer.

This module is a deliberately small demonstration of how the Runtime
exposes the right shape for a higher-level orchestrator to live on top
of it — not a production-grade planner. The interesting code is the
runtime; the coordinator is an existence proof.

Two coordinators ship here:

  RuleBasedCoordinator: decomposes by simple heuristics (bullet list,
  numbered steps, parts separated by 'and then', etc.). Useful in tests
  because it doesn't need the API.

  LLMCoordinator: uses one extra Runtime session as a "planner" that
  produces a JSON plan; subsequent sessions execute each step. Real use
  case; gated on having an API key.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from agi.runtime import RunResult, Runtime


@dataclass
class SubTask:
    id: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class CoordinationResult:
    goal: str
    plan: list[SubTask]
    results: dict[str, RunResult]
    total_cost_usd: float
    elapsed_s: float
    summary: str


class RuleBasedCoordinator:
    """Decompose by simple parsing. No LLM calls in the planner.

    Used as a test fixture and as the deterministic backbone of demos.
    """

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def plan(self, goal: str) -> list[SubTask]:
        # Recognise three patterns in order of specificity:
        # 1. Explicit numbered list "1. ..." / "1) ..."
        # 2. Bulleted list "- ..." / "* ..."
        # 3. ' and then ' separators
        # If none match, treat the goal as a single sub-task.
        numbered = re.findall(r"^\s*\d+[.)]\s+(.+)$", goal, re.MULTILINE)
        if numbered:
            return [SubTask(id=f"s{i}", prompt=line.strip()) for i, line in enumerate(numbered)]
        bulleted = re.findall(r"^\s*[-*]\s+(.+)$", goal, re.MULTILINE)
        if bulleted:
            return [SubTask(id=f"s{i}", prompt=line.strip()) for i, line in enumerate(bulleted)]
        if " and then " in goal.lower():
            parts = re.split(r"\s+and then\s+", goal, flags=re.IGNORECASE)
            return [SubTask(id=f"s{i}", prompt=p.strip()) for i, p in enumerate(parts) if p.strip()]
        return [SubTask(id="s0", prompt=goal.strip())]

    def execute(
        self,
        goal: str,
        *,
        parallel: bool = False,
        session_metadata: dict[str, Any] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
    ) -> CoordinationResult:
        plan = self.plan(goal)
        return self._run_plan(goal, plan, session_metadata=session_metadata, on_event=on_event)

    # Shared by both coordinator flavors.
    def _run_plan(
        self,
        goal: str,
        plan: list[SubTask],
        *,
        session_metadata: dict[str, Any] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
    ) -> CoordinationResult:
        t0 = time.time()
        results: dict[str, RunResult] = {}
        # One session per sub-task — fresh context, fresh memory.
        for sub in plan:
            sid = self.runtime.create_session(metadata={"goal": goal, "subtask": sub.id, **(session_metadata or {})})
            prompt = sub.prompt
            if sub.depends_on:
                deps = "\n\n".join(
                    f"### Result from {dep}\n{results[dep].output_text}"
                    for dep in sub.depends_on
                    if dep in results
                )
                prompt = f"{prompt}\n\nContext from previous steps:\n{deps}"
            if on_event:
                on_event("subtask.started", {"subtask": sub.id, "prompt": prompt[:200]})
            try:
                result = self.runtime.run(
                    sid,
                    prompt,
                    idempotency_key=f"{goal[:32]}::{sub.id}",
                )
                results[sub.id] = result
                if on_event:
                    on_event("subtask.finished", {"subtask": sub.id, "cost_usd": result.cost_usd})
            finally:
                self.runtime.destroy_session(sid)

        total_cost = sum(r.cost_usd for r in results.values())
        elapsed = time.time() - t0
        summary = "\n\n".join(
            f"### {sub.id}: {sub.prompt[:80]}\n{results[sub.id].output_text}"
            for sub in plan
            if sub.id in results
        )
        return CoordinationResult(
            goal=goal,
            plan=plan,
            results=results,
            total_cost_usd=total_cost,
            elapsed_s=elapsed,
            summary=summary,
        )


_PLANNER_PROMPT = """\
You are a planner for an agent runtime. Decompose the user's goal into a
short list of concrete, independently-runnable subtasks. Output ONLY a
JSON object with this shape:

{
  "subtasks": [
    {"id": "s0", "prompt": "...", "depends_on": []},
    {"id": "s1", "prompt": "...", "depends_on": ["s0"]}
  ]
}

Rules:
- Keep the plan small (1-5 subtasks). Trivial goals stay as one subtask.
- Each subtask prompt must be self-contained enough for a fresh agent to
  execute given results from its declared dependencies.
- Order is significant; downstream tasks reference upstream ids in
  depends_on.
- Output JSON only — no prose, no markdown fences.
"""


class LLMCoordinator(RuleBasedCoordinator):
    """A planner that uses the same runtime to plan, then executes.

    The planner session is created with verbose=False and gets a focused
    system prompt overlay through the first user message. Cost rolls up
    into the result like any other subtask.
    """

    def execute(
        self,
        goal: str,
        *,
        parallel: bool = False,
        session_metadata: dict[str, Any] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
    ) -> CoordinationResult:
        planner_sid = self.runtime.create_session(metadata={"role": "planner", "goal": goal})
        try:
            planner_prompt = f"{_PLANNER_PROMPT}\n\nGoal:\n{goal}"
            planner_result = self.runtime.run(
                planner_sid,
                planner_prompt,
                idempotency_key=f"plan::{goal[:64]}",
            )
        finally:
            self.runtime.destroy_session(planner_sid)

        plan = _parse_plan(planner_result.output_text) or self.plan(goal)
        if on_event:
            on_event("plan.ready", {"plan": [{"id": s.id, "prompt": s.prompt[:120]} for s in plan]})
        return self._run_plan(goal, plan, session_metadata=session_metadata, on_event=on_event)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_plan(text: str) -> list[SubTask] | None:
    if not text:
        return None
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    raw = payload.get("subtasks")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[SubTask] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "prompt" not in item:
            continue
        out.append(
            SubTask(
                id=str(item.get("id") or f"s{i}"),
                prompt=str(item["prompt"]),
                depends_on=[str(d) for d in (item.get("depends_on") or [])],
            )
        )
    return out or None
