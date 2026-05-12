"""Goal compiler — turn a Goal into a Plan automatically.

The reference Coordinator takes a pluggable `decomposer: Goal → Plan`.
`single_step_decomposer` is fine for tests but useless in production: it
produces a one-step plan no matter how complex the intent.

This module ships two decomposers that are actually useful:

  - `heuristic_decomposer`: rule-based pattern matcher. No LLM call.
    Recognises common shapes (analyze X, build Y, find Z then summarize)
    and emits a multi-step DAG. Deterministic, fast, free.

  - `llm_decomposer`: asks the Runtime itself (through a planner-role
    session) to write a Plan. Reads the runtime's `capabilities()` so
    the planner knows which skills exist before slicing the work.

Both produce the same `Plan` shape so a Coordinator can swap them or
chain them: heuristic first, fall back to LLM if the heuristic returns
a trivial plan.

Investors care because this is the difference between "you write your
own planner" and "the runtime plans itself when you don't." A
coordination engine drives the runtime via Goals; the goal compiler
makes Goals actually executable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from agi.coordinator import Goal, Plan, PlanStep
from agi.runtime import Runtime, SessionConfig


# --- Heuristic patterns -----------------------------------------------------

# Each pattern is (compiled_re, builder). The builder receives the goal
# and the regex match and returns a list of PlanSteps.

PatternBuilder = Callable[[Goal, "re.Match[str]"], list[PlanStep]]


def _build_analyze_pattern(goal: Goal, m: "re.Match[str]") -> list[PlanStep]:
    target = m.group("target").strip()
    return [
        PlanStep(id="gather",  role="researcher",
                 prompt=f"Gather facts about: {target}"),
        PlanStep(id="analyze", role="planner",
                 prompt=f"Analyze the gathered facts about {target} and surface the most "
                        f"important findings.",
                 depends_on=["gather"]),
        PlanStep(id="summary", role="writer",
                 prompt=f"Write a concise summary of the analysis of {target}.",
                 depends_on=["analyze"]),
    ]


def _build_build_pattern(goal: Goal, m: "re.Match[str]") -> list[PlanStep]:
    target = m.group("target").strip()
    return [
        PlanStep(id="design",   role="planner",
                 prompt=f"Design the architecture for: {target}"),
        PlanStep(id="implement", role="executor",
                 prompt=f"Implement: {target}", depends_on=["design"]),
        PlanStep(id="test",      role="executor",
                 prompt=f"Write and run tests for: {target}", depends_on=["implement"]),
    ]


def _build_compare_pattern(goal: Goal, m: "re.Match[str]") -> list[PlanStep]:
    a = m.group("a").strip()
    b = m.group("b").strip()
    return [
        PlanStep(id="probe_a", role="researcher", prompt=f"Gather facts about: {a}"),
        PlanStep(id="probe_b", role="researcher", prompt=f"Gather facts about: {b}"),
        PlanStep(id="compare", role="writer",
                 prompt=f"Compare {a} and {b} using the facts gathered. Highlight tradeoffs.",
                 depends_on=["probe_a", "probe_b"]),
    ]


def _build_find_summarize_pattern(goal: Goal, m: "re.Match[str]") -> list[PlanStep]:
    target = m.group("target").strip()
    return [
        PlanStep(id="find",      role="researcher",
                 prompt=f"Find: {target}"),
        PlanStep(id="summarize", role="writer",
                 prompt=f"Summarize the findings about: {target}",
                 depends_on=["find"]),
    ]


HEURISTIC_PATTERNS: list[tuple[re.Pattern[str], PatternBuilder]] = [
    (re.compile(r"^\s*compare\s+(?P<a>[^\s].*?)\s+(?:and|vs\.?|versus)\s+(?P<b>.+?)\s*$",
                re.IGNORECASE), _build_compare_pattern),
    (re.compile(r"^\s*(?:analy[sz]e|study|investigate)\s+(?P<target>.+?)\s*$",
                re.IGNORECASE), _build_analyze_pattern),
    (re.compile(r"^\s*(?:build|create|implement|design)\s+(?P<target>.+?)\s*$",
                re.IGNORECASE), _build_build_pattern),
    (re.compile(r"^\s*find\s+(?P<target>.+?)(?:\s+(?:and|then)\s+(?:summari[sz]e|describe).*)?\s*$",
                re.IGNORECASE), _build_find_summarize_pattern),
]


def heuristic_decomposer(goal: Goal) -> Plan:
    """Rule-based decomposition. Falls back to a single-step plan when
    no pattern matches.

    Deterministic and free; coordinators use it as a fast first pass.
    """
    intent = goal.intent.strip().splitlines()[0] if goal.intent else ""
    for rx, build in HEURISTIC_PATTERNS:
        m = rx.search(intent)
        if m:
            steps = build(goal, m)
            return Plan(
                steps=steps,
                rationale=f"heuristic: matched pattern {rx.pattern!r}",
            )
    return Plan(
        steps=[PlanStep(id="root", role="executor", prompt=goal.intent)],
        rationale="heuristic: no pattern matched, single-step fallback",
    )


# --- LLM-based decomposer ---------------------------------------------------


_PLAN_SCHEMA = """\
Respond with a JSON object only — no prose, no markdown fences. Schema:

  {
    "rationale": "one sentence justifying the decomposition",
    "steps": [
      {
        "id": "snake_case_unique",
        "role": "planner" | "researcher" | "executor" | "writer" | "reviewer",
        "prompt": "self-contained instruction for this step",
        "depends_on": ["earlier_step_id", ...]
      },
      ...
    ]
  }

Rules:
  - 1-5 steps. Each step should be independently executable.
  - Earlier steps' results are NOT visible to later steps automatically;
    if a step depends on context from earlier work, say so in its prompt.
  - depends_on may be empty.
  - Steps with no dependencies run in parallel.
"""


@dataclass
class LlmDecomposerConfig:
    role: str = "planner"
    model: str = "claude-opus-4-7"
    effort: str = "medium"
    max_steps: int = 5
    fallback_on_error: bool = True


def llm_decomposer(
    runtime: Runtime,
    *,
    config: LlmDecomposerConfig | None = None,
) -> Callable[[Goal], Plan]:
    """Returns a `Goal → Plan` decomposer that asks the Runtime itself.

    The planner session is a regular session with skills disabled (so
    skill-loading doesn't bias the plan) and a planner role. It's
    asked to consider the runtime's current capabilities() before
    deciding.

    On any parsing / API error, falls back to the heuristic decomposer
    so a planning failure can't kill an autonomous loop.
    """
    cfg = config or LlmDecomposerConfig()

    def _decompose(goal: Goal) -> Plan:
        caps = runtime.capabilities()
        skill_lines = "\n".join(
            f"  - {s['name']}: {s['description']}" for s in caps.get("skills", [])[:10]
        ) or "  (none yet)"
        tool_lines = "\n".join(
            f"  - {t['name']}: {t['description']}" for t in caps.get("synthesized_tools", [])[:10]
        ) or "  (none yet)"
        planner_prompt = (
            f"You are decomposing a goal into a parallel/sequential plan.\n\n"
            f"# Goal\n{goal.intent}\n\n"
            f"# Available skills\n{skill_lines}\n\n"
            f"# Available synthesized tools\n{tool_lines}\n\n"
            f"# Output format\n{_PLAN_SCHEMA}\n"
        )
        session_cfg = SessionConfig(
            model=cfg.model,
            effort=cfg.effort,
            use_skills=False,
            role=cfg.role,
            system_prompt_extra=(
                "You are a planning specialist. Decompose the user's goal into "
                "ordered steps. Return ONLY valid JSON in the schema given."
            ),
            max_iterations=2,
        )
        sid = runtime.create_session(session_cfg)
        try:
            raw = runtime.chat(sid, planner_prompt)
        except Exception:
            if cfg.fallback_on_error:
                return heuristic_decomposer(goal)
            raise
        finally:
            try:
                runtime.end_session(sid)
            except KeyError:
                pass
        plan = parse_plan_json(raw, max_steps=cfg.max_steps)
        if plan is None:
            if cfg.fallback_on_error:
                return heuristic_decomposer(goal)
            raise ValueError(f"could not parse plan from planner output: {raw[:200]!r}")
        return plan

    return _decompose


def parse_plan_json(raw: str, *, max_steps: int = 5) -> Plan | None:
    """Robustly extract a Plan from possibly noisy LLM output.

    Looks for a JSON object in the text (greedy first-brace to
    last-brace) and validates the schema. Returns None on any failure
    so callers can fall back.
    """
    if not raw:
        return None
    # Strip code fences if present.
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    # Find the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    rationale = str(data.get("rationale", ""))
    steps_raw = data.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        return None
    steps: list[PlanStep] = []
    seen: set[str] = set()
    for s in steps_raw[:max_steps]:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", "")).strip()
        prompt = str(s.get("prompt", "")).strip()
        if not sid or not prompt or sid in seen:
            continue
        role = str(s.get("role", "executor")).strip() or "executor"
        deps_raw = s.get("depends_on", [])
        deps: list[str] = []
        if isinstance(deps_raw, list):
            for d in deps_raw:
                if isinstance(d, str) and d in seen:
                    deps.append(d)
        seen.add(sid)
        steps.append(PlanStep(id=sid, role=role, prompt=prompt, depends_on=deps))
    if not steps:
        return None
    return Plan(steps=steps, rationale=rationale)


def chained_decomposer(*decomposers: Callable[[Goal], Plan],
                       min_steps: int = 2) -> Callable[[Goal], Plan]:
    """Try each decomposer until one returns a Plan with ≥ `min_steps` steps.

    Falls back to the last decomposer's plan if all return trivial.
    Useful when wiring `heuristic_decomposer → llm_decomposer(runtime)`
    so the fast/free path runs first and the LLM only fires on novel
    goals.
    """
    def _run(goal: Goal) -> Plan:
        last: Plan | None = None
        for dec in decomposers:
            try:
                plan = dec(goal)
            except Exception:
                continue
            last = plan
            if len(plan.steps) >= min_steps:
                return plan
        return last or Plan(
            steps=[PlanStep(id="root", role="executor", prompt=goal.intent)],
            rationale="all decomposers failed; trivial fallback",
        )
    return _run
