"""Coordinator: declarative task graphs over the Runtime.

A coordination engine — whether it's our own `Coordinator` or an external
orchestrator wrapping the HTTP API — needs three things from the runtime:

1. Spawn an addressable session with a budget and a role.
2. Drive it through one or more turns and observe its event stream.
3. Tear it down, with cost accounting that rolls up to the parent.

`Coordinator` is a *reference implementation* of that pattern. It takes a
high-level `Task`, optionally decomposes it into sub-`Task`s (each gets its own
session with its own budget), executes them, and aggregates the results. The
parent task's budget is the cap on total spend across children — if a child
overruns, the coordinator surfaces an error rather than silently absorbing it.

Two execution strategies ship in this file:

- `run_one(task)`: single-session execution. Spin up a session, send the
  prompt, return the final text + cost + trace. The common case.
- `run_parallel(tasks)`: fan out to N sessions concurrently, gather results.
  This is what a coordination engine wants when it has independent subtasks.

The `decompose(task, max_subtasks)` helper uses the planner agent to break a
task into a JSON list of subtasks. It's optional — the coordinator works fine
with hand-authored subtask lists too.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agi.runtime import Runtime, BudgetExceededError
from agi.events import TurnEnd


@dataclass
class Task:
    """Declarative description of work for an agent session.

    `prompt` is the user message sent into the session. `role` is a short
    descriptor that gets appended to the system prompt — "planner", "executor",
    "researcher" — so different sessions specialize without code changes.
    `budget_usd` caps total spend for this task.
    """
    prompt: str
    role: str = "executor"
    budget_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional: hand-authored subtasks. If set, `Coordinator.run_one` will
    # execute them in parallel and feed the joined results back to a final
    # synthesis turn.
    subtasks: list["Task"] = field(default_factory=list)


@dataclass
class TaskResult:
    """What a coordinator returns for a single task. `children` lets a caller
    walk the full execution tree, including each subtask's cost."""
    task: Task
    session_id: str
    final_text: str
    cost_usd: float
    elapsed_s: float
    error: str | None = None
    children: list["TaskResult"] = field(default_factory=list)

    def total_cost_usd(self) -> float:
        return self.cost_usd + sum(c.total_cost_usd() for c in self.children)


ROLE_PROMPTS: dict[str, str] = {
    "planner": (
        "You are the PLANNER role in a multi-agent system. Break the user's "
        "request into 2-5 independent, well-scoped subtasks. Return ONLY a "
        "JSON array of objects with keys `prompt` and `role`. Pick role from "
        "{researcher, executor, writer, critic}. No prose outside the JSON."
    ),
    "researcher": (
        "You are the RESEARCHER role. Gather facts using web_search and "
        "web_fetch; cite sources. Return a tight, bulleted findings list."
    ),
    "executor": (
        "You are the EXECUTOR role. Carry out the requested task end-to-end "
        "using your tools. Verify before claiming success."
    ),
    "writer": (
        "You are the WRITER role. Produce the requested artifact. No filler."
    ),
    "critic": (
        "You are the CRITIC role. Review the prior output for correctness, "
        "completeness, and risks. Return a JSON object with keys `verdict` "
        "in {pass, revise, fail} and `notes`."
    ),
    "synthesizer": (
        "You are the SYNTHESIZER role. Subtasks have already produced their "
        "outputs and they are in your context. Produce a single coherent "
        "final answer for the original user request."
    ),
}


class Coordinator:
    """Reference coordination engine that drives the Runtime.

    External coordination engines should use the HTTP API (see `agi.server`).
    This class is for in-process orchestration and serves as the canonical
    example of how the runtime contract is supposed to be used.
    """

    def __init__(
        self,
        runtime: Runtime,
        max_parallel: int = 4,
        default_subtask_budget_usd: float | None = None,
    ) -> None:
        self.runtime = runtime
        self.max_parallel = max_parallel
        self.default_subtask_budget_usd = default_subtask_budget_usd

    def run_one(self, task: Task) -> TaskResult:
        """Execute a single task. If the task has `subtasks`, runs them in
        parallel first, then a synthesizer turn over their outputs."""
        if task.subtasks:
            return self._run_with_subtasks(task)
        return self._run_leaf(task)

    def run_parallel(self, tasks: list[Task]) -> list[TaskResult]:
        """Run `tasks` concurrently and return results in the same order."""
        results: list[TaskResult | None] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as ex:
            futures = {ex.submit(self.run_one, t): i for i, t in enumerate(tasks)}
            for fut in list(futures.keys()):
                i = futures[fut]
                results[i] = fut.result()
        return [r for r in results if r is not None]

    def decompose(self, task: Task, max_subtasks: int = 5) -> list[Task]:
        """Ask a planner session to break a task into subtasks. The agent
        returns JSON; we parse it. Failures fall back to a single-task list
        (no decomposition) — never crash the coordinator on bad planner output.
        """
        planner = Task(
            prompt=(
                f"User request: {task.prompt}\n\n"
                f"Break this into at most {max_subtasks} subtasks. "
                "Return ONLY the JSON array."
            ),
            role="planner",
            budget_usd=task.budget_usd,
        )
        result = self._run_leaf(planner)
        subtasks = _parse_subtasks_json(result.final_text)
        return subtasks or [task]

    # -------- internal --------

    def _run_leaf(self, task: Task) -> TaskResult:
        t0 = time.time()
        extra = ROLE_PROMPTS.get(task.role, "")
        budget = task.budget_usd if task.budget_usd is not None else self.default_subtask_budget_usd
        sid = self.runtime.open(
            budget_usd=budget,
            system_prompt_extra=extra,
            metadata={"role": task.role, **task.metadata},
        )
        try:
            self.runtime.send(sid, task.prompt)
            end = self.runtime.wait_for_turn_end(sid)
            elapsed = time.time() - t0
            if end is None:
                status = self.runtime.status(sid)
                return TaskResult(
                    task=task,
                    session_id=sid,
                    final_text="",
                    cost_usd=status.cost_usd,
                    elapsed_s=elapsed,
                    error="no turn-end event (stream closed or timed out)",
                )
            return TaskResult(
                task=task,
                session_id=sid,
                final_text=end.final_text,
                cost_usd=end.cost_usd,
                elapsed_s=elapsed,
            )
        except BudgetExceededError as e:
            return TaskResult(
                task=task,
                session_id=sid,
                final_text="",
                cost_usd=self.runtime.status(sid).cost_usd,
                elapsed_s=time.time() - t0,
                error=str(e),
            )
        finally:
            self.runtime.close(sid)

    def _run_with_subtasks(self, task: Task) -> TaskResult:
        t0 = time.time()
        # Apply default budget to children without one.
        children_specs: list[Task] = []
        for st in task.subtasks:
            if st.budget_usd is None and self.default_subtask_budget_usd is not None:
                st = Task(
                    prompt=st.prompt,
                    role=st.role,
                    budget_usd=self.default_subtask_budget_usd,
                    metadata=dict(st.metadata),
                    subtasks=list(st.subtasks),
                )
            children_specs.append(st)

        child_results = self.run_parallel(children_specs)

        # Synthesis: feed the original prompt + each child's output into one
        # final agent turn. This is where a real coordination engine would
        # usually plug in its own merge logic; we ship a reasonable default.
        synth_prompt = _build_synthesis_prompt(task, child_results)
        synth_task = Task(
            prompt=synth_prompt,
            role="synthesizer",
            budget_usd=task.budget_usd,
            metadata={"phase": "synthesis"},
        )
        synth_result = self._run_leaf(synth_task)

        return TaskResult(
            task=task,
            session_id=synth_result.session_id,
            final_text=synth_result.final_text,
            cost_usd=synth_result.cost_usd,
            elapsed_s=time.time() - t0,
            error=synth_result.error,
            children=child_results,
        )


# ---- helpers ----


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_subtasks_json(text: str) -> list[Task]:
    """Pull the first JSON array out of the model's response. Tolerate the
    common failure modes: ```json fences, leading prose, trailing prose."""
    if not text:
        return []
    # Strip code fences if present: ```json\n...\n``` or ```\n...\n```
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        # parts = ["", "json\n...\n", ""]  → take the middle
        if len(parts) >= 2:
            cleaned = parts[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    m = _JSON_ARRAY_RE.search(cleaned)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[Task] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        prompt = it.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        role = it.get("role", "executor")
        if not isinstance(role, str):
            role = "executor"
        out.append(Task(prompt=prompt, role=role))
    return out


def _build_synthesis_prompt(parent: Task, children: list[TaskResult]) -> str:
    parts = [
        f"Original request: {parent.prompt}",
        "",
        "Subtask outputs:",
    ]
    for i, c in enumerate(children, start=1):
        parts.append("")
        parts.append(f"--- subtask {i} ({c.task.role}) ---")
        parts.append(f"prompt: {c.task.prompt}")
        if c.error:
            parts.append(f"error: {c.error}")
        parts.append("output:")
        parts.append(c.final_text or "(empty)")
    parts.append("")
    parts.append("Produce the final coherent answer for the original request.")
    return "\n".join(parts)
