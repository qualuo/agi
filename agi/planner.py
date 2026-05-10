"""Planner: turn a goal into a typed task DAG.

The planner is just an LLM call constrained to produce a `GraphSpec`-shaped
JSON object. We pass the `decompose-goal` skill into the system prompt and
parse the result. If parsing fails, we degrade to a single-node graph (the
goal as one chat task) — the coordination engine can then re-plan.

This is the bridge between the reasoning core and the coordination engine:
the engine asks the runtime to plan, gets a graph back, and executes it.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from agi.runtime.graph import GraphSpec, NodeSpec


_PLANNER_SYSTEM = """\
You are a planner. Given a goal, output a typed task DAG as JSON. The DAG
will be executed by a coordination engine; your output is the contract.

Schema (GraphSpec):
{
  "name": "<slug>",
  "nodes": [
    {
      "id": "<unique-id>",
      "kind": "chat" | "plan" | "critique" | "skill.invoke" | "tool" | "noop",
      "role": "planner" | "executor" | "critic" | null,
      "input": { ... kind-specific fields ... },
      "depends_on": ["<other-id>", ...],
      "on_failure": "fail_graph" | "skip" | "retry:N"
    }
  ]
}

Rules:
- Reference upstream outputs with "${node_id.field}" placeholders inside string
  values. The executor substitutes them before dispatch.
- Each chat node takes {"message": "..."}.
- Each critique node takes {"prompt": "...", "response": "${node_id.text}"}.
- Each skill.invoke takes {"skill": "<name>", "args": {...}}.
- Each tool node takes {"name": "<tool-name>", "args": {...}}.
- Add a critique node before any irreversible / externally-visible action.

Reply with the JSON object, nothing else. No markdown fences.
"""


_FALLBACK_GRAPH_NAME_RE = re.compile(r"[^a-z0-9-]+")


def _slug(text: str) -> str:
    s = text.strip().lower().replace(" ", "-")
    s = _FALLBACK_GRAPH_NAME_RE.sub("", s)[:40]
    return s or "task"


def _extract_json(text: str) -> dict | None:
    # Strip Markdown fences if the LLM adds them despite instructions.
    text = text.strip()
    if text.startswith("```"):
        # remove first and last fence
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def propose_graph(
    *,
    goal: str,
    constraints: str = "",
    agent_factory: Callable[[str | None], Any] | None = None,
) -> dict[str, Any]:
    """Ask the LLM to produce a GraphSpec for `goal`.

    Returns a dict (the serialized GraphSpec). If the LLM output can't be
    parsed, returns a degenerate one-node graph so the coordinator can still
    make progress.
    """
    if agent_factory is None:
        from agi.agent import Agent
        agent_factory = lambda role: Agent(verbose=False, role=role,
                                            extra_system_prompt=_PLANNER_SYSTEM)
    agent = agent_factory("planner")
    # If the agent doesn't accept extra_system_prompt, append as a user note.
    if not hasattr(agent, "_planner_primed"):
        agent.messages.append({"role": "user",
                               "content": "[planner-mode]\n" + _PLANNER_SYSTEM})
        agent._planner_primed = True  # type: ignore[attr-defined]

    prompt = f"GOAL:\n{goal}\n"
    if constraints:
        prompt += f"\nCONSTRAINTS:\n{constraints}\n"
    prompt += "\nReturn the GraphSpec JSON now."

    text = agent.chat(prompt)
    parsed = _extract_json(text)
    if parsed is None or "nodes" not in parsed:
        return GraphSpec(
            name=_slug(goal),
            nodes=[NodeSpec(id="root", kind="chat", role="executor",
                            input={"message": goal})],
        ).to_dict()
    # Normalize: ensure required fields exist.
    nodes_in = parsed.get("nodes", [])
    nodes: list[NodeSpec] = []
    for n in nodes_in:
        if not isinstance(n, dict) or "id" not in n or "kind" not in n:
            continue
        nodes.append(NodeSpec(
            id=str(n["id"]),
            kind=str(n["kind"]),
            input=n.get("input") or {},
            depends_on=[str(d) for d in n.get("depends_on", [])],
            on_failure=str(n.get("on_failure", "fail_graph")),
            role=n.get("role"),
            budget_tokens=n.get("budget_tokens"),
            budget_seconds=n.get("budget_seconds"),
        ))
    if not nodes:
        return GraphSpec(
            name=_slug(goal),
            nodes=[NodeSpec(id="root", kind="chat", role="executor",
                            input={"message": goal})],
        ).to_dict()
    return GraphSpec(name=parsed.get("name") or _slug(goal), nodes=nodes).to_dict()
