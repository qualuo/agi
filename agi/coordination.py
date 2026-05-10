"""Coordination primitives: subagent delegation and reflection.

Two related ideas live here:

1. **Delegate** — a tool that lets an agent spawn a role-specialized
   subagent for a focused subtask, run it to completion, return the result.
   Subagent token usage rolls up into the parent's running total so the
   caller sees a true total cost.

2. **Reflect** — after a task, write a one-paragraph "what worked / what
   didn't" entry to long-term memory tagged `lesson`. This is the unit of
   per-task learning the architecture calls for.

Both are intentionally thin wrappers — they use the existing Agent and
Memory primitives, no new infrastructure.
"""
from __future__ import annotations

from typing import Any, Callable

from agi.costs import Usage
from agi.memory import Memory


# Pre-canned role primers. Coordinators (and the parent agent) can override
# by supplying a custom system prompt, but these cover common patterns.
ROLE_PROMPTS: dict[str, str] = {
    "planner": (
        "You are a planning subagent. Decompose the task into a numbered list of "
        "concrete, verifiable steps. Do not execute — produce the plan only."
    ),
    "executor": (
        "You are an execution subagent. Carry out the task using the tools "
        "available. Verify your work. Report back what you did and the outcome."
    ),
    "critic": (
        "You are a critic subagent. Read the work and identify concrete "
        "problems: errors, missing steps, weak assumptions. Be specific. "
        "Do not rewrite — diagnose."
    ),
    "researcher": (
        "You are a research subagent. Use web search and web fetch to gather "
        "evidence on the task. Cite sources. Distinguish claim from speculation."
    ),
    "summarizer": (
        "You are a summarization subagent. Produce a tight summary at the "
        "level of detail requested. Do not editorialize."
    ),
}


def make_delegate_tool(
    *,
    parent_usage: Usage,
    parent_memory: Memory,
    parent_model: str,
    parent_extra_system: str | None = None,
    max_depth: int = 2,
    current_depth: int = 0,
) -> tuple[list[dict], dict]:
    """Returns (schemas, handlers) for a `delegate` tool that the parent
    agent can call. The parent's Usage object is mutated to reflect the
    child's spend — this is the "honest accounting" the plan calls for.

    `max_depth` caps recursion so a runaway agent doesn't fan out into a
    forest of subagents."""
    if current_depth >= max_depth:
        return [], {}

    def delegate(role: str, task: str, model: str | None = None, max_iterations: int = 12) -> str:
        # Local import to avoid a circular dependency at module-load time.
        from agi.agent import Agent

        primer = ROLE_PROMPTS.get(role, f"You are a {role} subagent. Carry out the task as instructed.")
        extra_system = primer
        if parent_extra_system:
            extra_system = parent_extra_system + "\n\n" + primer

        # Children get their own delegate tool with depth-1 budget so a
        # subtree can fan out further if needed but not infinitely.
        child_delegate_schemas, child_delegate_handlers = make_delegate_tool(
            parent_usage=parent_usage,
            parent_memory=parent_memory,
            parent_model=model or parent_model,
            parent_extra_system=parent_extra_system,
            max_depth=max_depth,
            current_depth=current_depth + 1,
        )

        child = Agent(
            memory=parent_memory,
            model=model or parent_model,
            verbose=False,
            extra_system=extra_system,
            extra_tool_schemas=child_delegate_schemas,
            extra_tool_handlers=child_delegate_handlers,
        )
        try:
            text = child.chat(task, max_iterations=max_iterations)
        except Exception as e:
            return f"[delegate {role!r} errored: {type(e).__name__}: {e}]"

        # Roll up child usage into parent's totals.
        parent_usage.input_tokens += child.usage.input_tokens
        parent_usage.output_tokens += child.usage.output_tokens
        parent_usage.cache_creation_input_tokens += child.usage.cache_creation_input_tokens
        parent_usage.cache_read_input_tokens += child.usage.cache_read_input_tokens

        cost = child.usage.cost_usd(child.model)
        header = f"[delegated to {role!r} on {child.model} — ${cost:.4f}, {child.usage.input_tokens + child.usage.output_tokens} tokens]\n"
        return header + text

    schemas = [
        {
            "name": "delegate",
            "description": (
                "Spawn a role-specialized subagent for a focused subtask. The "
                "subagent has the same tool surface and shares long-term memory "
                "with you. Token usage rolls up to the parent. Roles: "
                + ", ".join(sorted(ROLE_PROMPTS.keys()))
                + ". Use sparingly — only when decomposition is clearly cheaper "
                "or higher-quality than handling the subtask yourself."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "One of: planner, executor, critic, researcher, summarizer."},
                    "task": {"type": "string", "description": "The subtask, framed as a complete instruction to a fresh agent."},
                    "model": {"type": "string", "description": "Optional model override (default: parent's model)."},
                    "max_iterations": {"type": "integer", "default": 12, "description": "Cap on the subagent's tool-loop iterations."},
                },
                "required": ["role", "task"],
            },
        }
    ]
    handlers: dict[str, Callable[..., str]] = {"delegate": delegate}
    return schemas, handlers


def make_reflection_tool(memory: Memory) -> tuple[list[dict], dict]:
    """Tool the agent calls at the end of a task to durably record what it
    learned. Tagged `lesson` for retrieval at the start of related tasks."""

    def reflect(task: str, what_worked: str, what_failed: str = "", lesson: str = "") -> str:
        # One memory entry, structured so it's both human-readable and
        # easy to filter/aggregate later.
        text = f"task: {task}\nworked: {what_worked}"
        if what_failed:
            text += f"\nfailed: {what_failed}"
        if lesson:
            text += f"\nlesson: {lesson}"
        note = memory.save(text, tags=["lesson"])
        return f"recorded lesson {note.id}"

    schemas = [
        {
            "name": "reflect",
            "description": (
                "Record a one-paragraph reflection on the task you just completed. "
                "Saved to long-term memory tagged 'lesson' so future related tasks "
                "can retrieve it. Use this at the end of non-trivial tasks."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Short description of the task."},
                    "what_worked": {"type": "string", "description": "What approach succeeded."},
                    "what_failed": {"type": "string", "description": "What didn't work, if anything."},
                    "lesson": {"type": "string", "description": "The takeaway: how you'd approach this again."},
                },
                "required": ["task", "what_worked"],
            },
        }
    ]
    handlers = {"reflect": reflect}
    return schemas, handlers


def make_coordination_tools(
    *,
    parent_usage: Usage,
    parent_memory: Memory,
    parent_model: str,
    parent_extra_system: str | None = None,
    enable_delegate: bool = True,
    enable_reflect: bool = True,
) -> tuple[list[dict], dict[str, Callable[..., str]]]:
    schemas: list[dict] = []
    handlers: dict[str, Callable[..., str]] = {}
    if enable_delegate:
        ds, dh = make_delegate_tool(
            parent_usage=parent_usage,
            parent_memory=parent_memory,
            parent_model=parent_model,
            parent_extra_system=parent_extra_system,
        )
        schemas.extend(ds)
        handlers.update(dh)
    if enable_reflect:
        rs, rh = make_reflection_tool(parent_memory)
        schemas.extend(rs)
        handlers.update(rh)
    return schemas, handlers
