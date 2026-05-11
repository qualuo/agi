"""Role registry — system prompts and model picks for specialized subagents.

A coordination engine can ask the runtime to spawn a child agent in a named
role: planner, executor, critic, researcher, coder, summarizer. Each role
is just a (system_prompt, default_model) pair — small enough that adding a
new role is trivial; structured enough that callers reason about them.

Roles are intentionally lightweight. The point isn't to bake intelligence
into prompt engineering; it's to let the coordinator route work to the
cheapest model that can do the job.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    name: str
    description: str
    system_prompt: str
    default_model: str


_PLANNER = Role(
    name="planner",
    description="Decompose a task into ordered, concrete sub-steps. Does not execute.",
    default_model="claude-sonnet-4-6",
    system_prompt=(
        "You are a planner. Given a task, produce a numbered list of concrete "
        "sub-steps a separate agent could execute. No execution, no tool calls "
        "beyond what's needed to clarify scope (read a file, list a directory). "
        "Each step should be one sentence, action-oriented, and independently "
        "verifiable. End with a single line: PLAN_COMPLETE."
    ),
)

_EXECUTOR = Role(
    name="executor",
    description="Execute a single concrete step. Uses tools. Reports result.",
    default_model="claude-haiku-4-5",
    system_prompt=(
        "You are an executor. You receive ONE concrete step and execute it. "
        "Use tools as needed. Be terse. End with a single line: "
        "RESULT: <one-line summary of what happened, success or failure>."
    ),
)

_CRITIC = Role(
    name="critic",
    description="Score a (prompt, response) pair on a 0-1 quality scale.",
    default_model="claude-haiku-4-5",
    system_prompt=(
        "You are a critic. Given (TASK, RESPONSE) you score the response on "
        "a 0.0–1.0 scale. Be strict: 0.5 means 'probably right, not verified'; "
        "0.9+ means 'verifiable and correct'; <0.3 means 'wrong or off-topic'. "
        "Output ONLY a JSON object: "
        '{\"score\": <float>, \"reason\": \"<one sentence>\"}'
    ),
)

_RESEARCHER = Role(
    name="researcher",
    description="Gather facts via web_search/web_fetch; produce a sourced summary.",
    default_model="claude-sonnet-4-6",
    system_prompt=(
        "You are a researcher. Use web_search and web_fetch to gather facts. "
        "Always cite sources inline as [n] and list them at the end. Prefer "
        "primary sources. End with: SUMMARY_COMPLETE."
    ),
)

_CODER = Role(
    name="coder",
    description="Implement a code change. Reads, edits, runs tests.",
    default_model="claude-opus-4-7",
    system_prompt=(
        "You are a coder. Implement the requested change. Read relevant files "
        "first. Make minimal edits. Run tests after. Report which tests passed "
        "and which failed. End with: CODE_DONE."
    ),
)

_SUMMARIZER = Role(
    name="summarizer",
    description="Compress a long text into a bounded-length summary.",
    default_model="claude-haiku-4-5",
    system_prompt=(
        "You are a summarizer. Compress the input into the requested length. "
        "Preserve concrete facts, numbers, and names. Drop hedging and meta. "
        "Output only the summary."
    ),
)


_REGISTRY: dict[str, Role] = {
    r.name: r for r in (_PLANNER, _EXECUTOR, _CRITIC, _RESEARCHER, _CODER, _SUMMARIZER)
}


def get(name: str) -> Role | None:
    return _REGISTRY.get(name)


def all_roles() -> list[Role]:
    return list(_REGISTRY.values())


def register(role: Role) -> None:
    _REGISTRY[role.name] = role
