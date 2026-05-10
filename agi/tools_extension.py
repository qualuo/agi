"""Extended tools: subagent delegation, tool synthesis, plan-as-graph.

These attach on top of the base tool set in `agi.tools`. They're separated
because they require the Agent class (delegation) and a sandbox primitive
(tool synthesis), which the base tools don't.
"""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
from typing import Any, Callable

from agi.memory import Memory
from agi.world_model import WorldModel


# Names a synthesized tool may not bind to (avoid overriding builtin tools).
_PROTECTED_TOOL_NAMES = {
    "read_file", "write_file", "list_dir", "run_bash",
    "save_memory", "search_memory", "recent_memory",
    "delegate", "make_tool", "plan_graph", "list_skills", "invoke_skill",
    "remember_observation",
}


def _safe_compile(code: str, name: str) -> Callable[..., Any]:
    """Compile a Python function body into a callable.

    Restricts the AST to a conservative subset: no imports, no exec/eval,
    no attribute access on `__` dunders, no globals/locals/open of arbitrary
    files. This is *not* a security boundary — it's a guardrail. Treat the
    runtime host as untrusted-by-default and run in a sandbox in production.
    """
    tree = ast.parse(code, mode="exec")

    class Guard(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            raise ValueError("imports not allowed in synthesized tools")

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            raise ValueError("imports not allowed in synthesized tools")

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr.startswith("__"):
                raise ValueError("dunder access not allowed")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            f = node.func
            if isinstance(f, ast.Name) and f.id in {"eval", "exec", "compile",
                                                    "open", "__import__", "globals",
                                                    "locals", "vars"}:
                raise ValueError(f"disallowed call: {f.id}")
            self.generic_visit(node)

    Guard().visit(tree)

    # Allow only a function definition at the top level matching `name`.
    if not (len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)):
        raise ValueError("synthesized tool must be a single top-level function")
    fdef: ast.FunctionDef = tree.body[0]  # type: ignore[assignment]
    if fdef.name != name:
        raise ValueError(f"function name must be {name!r}, got {fdef.name!r}")

    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "sorted": sorted, "abs": abs, "round": round, "any": any, "all": all,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "list": list, "dict": dict, "tuple": tuple, "set": set, "str": str,
        "int": int, "float": float, "bool": bool, "isinstance": isinstance,
        "True": True, "False": False, "None": None,
        "print": print,
    }
    env: dict[str, Any] = {"__builtins__": safe_builtins}
    exec(compile(tree, f"<tool:{name}>", "exec"), env)  # noqa: S102
    fn = env[name]
    if not callable(fn):
        raise ValueError("compiled object is not callable")
    return fn


def make_extension_tools(
    *,
    agent_factory: Callable[..., Any] | None = None,
    world_model: WorldModel | None = None,
    persistent_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., str]], dict[str, Any]]:
    """Return (schemas, handlers, registry) for the extended tool set.

    `registry` is a mutable dict the synthesized tools live in; the Agent
    surfaces them after a successful `make_tool` call.
    """
    wm = world_model or WorldModel()
    synth_dir = persistent_dir or (Path.home() / ".agi" / "tools_synth")
    synth_dir.mkdir(parents=True, exist_ok=True)
    synthesized: dict[str, Any] = {}  # name -> {"code": str, "schema": dict, "fn": callable}

    def delegate(task: str, role: str = "executor") -> str:
        """Spawn a subagent with a focused task; return its final text."""
        if agent_factory is None:
            from agi.agent import Agent
            sub = Agent(verbose=False, role=role)
        else:
            sub = agent_factory(role)
        text = sub.chat(task)
        # Roll up token usage into the world model so we have provenance.
        wm.observe(kind="entity", id=f"subagent:{role}", action="delegate",
                   outcome="success",
                   detail={"task_head": task[:120], "tokens":
                           sub.usage.input_tokens + sub.usage.output_tokens})
        return text

    def make_tool(name: str, description: str, code: str,
                  args_schema: dict | None = None) -> str:
        """Define a new Python tool from source. Survives the session if
        the user explicitly promotes it (`/promote-tool`).

        The function must be a single top-level def matching `name`,
        with no imports and no eval/exec/open. This is a guardrail, not
        a sandbox; do not enable this tool in untrusted contexts.
        """
        if name in _PROTECTED_TOOL_NAMES:
            return f"error: '{name}' is a protected tool name"
        try:
            fn = _safe_compile(textwrap.dedent(code), name)
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"
        schema = {
            "name": name,
            "description": description,
            "input_schema": args_schema or {"type": "object", "properties": {}},
        }
        synthesized[name] = {"code": code, "schema": schema, "fn": fn}
        # Persist a draft (not promoted) so the next session can see it.
        (synth_dir / f"{name}.py").write_text(code)
        (synth_dir / f"{name}.json").write_text(json.dumps(schema, indent=2))
        return f"tool '{name}' defined ({len(code.splitlines())} lines). Use it now."

    def plan_graph(goal: str, constraints: str = "") -> str:
        from agi.planner import propose_graph
        graph = propose_graph(goal=goal, constraints=constraints,
                              agent_factory=agent_factory)
        return json.dumps(graph, indent=2)

    def list_skills(query: str = "") -> str:
        from agi.skills.library import SkillLibrary
        lib = SkillLibrary()
        skills = lib.retrieve(query, k=10) if query else lib.all()
        if not skills:
            return "(no skills found)"
        return "\n".join(f"- {s.name}: {s.description}" for s in skills)

    def invoke_skill(skill: str, args_json: str = "{}") -> str:
        from agi.skills.library import SkillLibrary
        lib = SkillLibrary()
        s = lib.get(skill)
        if s is None:
            return f"error: unknown skill '{skill}'"
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError as e:
            return f"error: args_json is not valid JSON: {e}"
        return s.render(args)

    def remember_observation(kind: str, id: str, action: str,
                             outcome: str = "success", detail_json: str = "{}") -> str:
        try:
            detail = json.loads(detail_json) if detail_json else {}
        except json.JSONDecodeError:
            detail = {"raw": detail_json}
        wm.observe(kind=kind, id=id, action=action, outcome=outcome, detail=detail)
        return f"observed {kind}:{id} action={action} outcome={outcome}"

    schemas: list[dict[str, Any]] = [
        {
            "name": "delegate",
            "description": "Spawn a focused subagent with a specific task. Returns the subagent's final answer. Use for parallel decomposition or for cheaper specialist work.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task the subagent should accomplish."},
                    "role": {"type": "string", "description": "Role hint: executor | critic | planner.", "default": "executor"},
                },
                "required": ["task"],
            },
        },
        {
            "name": "make_tool",
            "description": "Define a new tool from Python source. Use when no existing tool fits and the procedure is reusable. Must be a single function with no imports and no eval/exec/open.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "code": {"type": "string", "description": "Python source: single `def <name>(...)`."},
                    "args_schema": {"type": "object", "description": "JSON schema for the tool's input."},
                },
                "required": ["name", "description", "code"],
            },
        },
        {
            "name": "plan_graph",
            "description": "Decompose a goal into a typed task DAG (GraphSpec JSON) the coordination engine can execute. Use for multi-step goals where parallelism or verification gates help.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "constraints": {"type": "string", "default": ""},
                },
                "required": ["goal"],
            },
        },
        {
            "name": "list_skills",
            "description": "List skills available in the skill library, optionally filtered by a keyword query.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "default": ""},
                },
            },
        },
        {
            "name": "invoke_skill",
            "description": "Render a named skill's SOP with the given args. Returns the rendered SOP text; reading it tells you the procedure to follow.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "args_json": {"type": "string", "default": "{}",
                                  "description": "JSON object of args, e.g. '{\"question\": \"...\"}'"},
                },
                "required": ["skill"],
            },
        },
        {
            "name": "remember_observation",
            "description": "Record a structured observation (entity, action, outcome) in the world model. Use for facts that aren't natural prose: 'I read file X, succeeded'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "description": "file | url | command | entity"},
                    "id": {"type": "string", "description": "Canonical id of the thing observed."},
                    "action": {"type": "string"},
                    "outcome": {"type": "string", "default": "success"},
                    "detail_json": {"type": "string", "default": "{}"},
                },
                "required": ["kind", "id", "action"],
            },
        },
    ]
    handlers: dict[str, Callable[..., str]] = {
        "delegate": delegate,
        "make_tool": make_tool,
        "plan_graph": plan_graph,
        "list_skills": list_skills,
        "invoke_skill": invoke_skill,
        "remember_observation": remember_observation,
    }
    return schemas, handlers, synthesized
