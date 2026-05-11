"""Tools the agent can call.

`make_tools(memory, ...)` returns a `(schemas, handlers)` pair: the JSON
schemas go in the API request, the handlers are dispatched on `tool_use`
blocks. File and shell tools touch the host directly — run this in a
sandbox if you don't trust the model's choices.

The optional `runtime`, `current_agent`, and `skill_library` parameters
unlock three higher-leverage tools:

  - `delegate`        — spawn a subagent run; cost rolls up to the parent.
  - `save_skill` /    — read/write the persistent skill library so the
    `search_skills` /   next run on the same task family is cheaper.
    `load_skill`
  - `make_tool`       — register a new Python tool for the rest of this
                        session (sandboxed exec, no I/O at registration).
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any, Callable

from agi.memory import Memory


def make_tools(
    memory: Memory,
    *,
    runtime=None,
    current_agent=None,
    skill_library=None,
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., str]]]:
    def read_file(path: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"error: {p} does not exist"
        if not p.is_file():
            return f"error: {p} is not a file"
        try:
            return p.read_text()
        except UnicodeDecodeError:
            return f"error: {p} is not a UTF-8 text file"

    def write_file(path: str, content: str) -> str:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content.encode())} bytes to {p}"

    def list_dir(path: str = ".") -> str:
        p = Path(path).expanduser()
        if not p.is_dir():
            return f"error: {p} is not a directory"
        entries = []
        for entry in sorted(p.iterdir()):
            kind = "d" if entry.is_dir() else "f"
            entries.append(f"{kind} {entry.name}")
        return "\n".join(entries) or "(empty)"

    def run_bash(command: str, timeout_seconds: int = 30) -> str:
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout_seconds}s"
        out = result.stdout
        if result.stderr:
            out += ("\n" if out else "") + f"[stderr]\n{result.stderr}"
        out += f"\n[exit {result.returncode}]"
        return out

    def save_memory(text: str, tags: list[str] | None = None) -> str:
        note = memory.save(text, tags or [])
        return f"saved note {note.id}"

    def search_memory(query: str, k: int = 5) -> str:
        results = memory.search(query, k)
        if not results:
            return "no matches"
        lines = []
        for n in results:
            tag_str = f" [{', '.join(n.tags)}]" if n.tags else ""
            lines.append(f"- ({n.id}){tag_str} {n.text}")
        return "\n".join(lines)

    def recent_memory(k: int = 10) -> str:
        results = memory.recent(k)
        if not results:
            return "(memory is empty)"
        lines = []
        for n in results:
            tag_str = f" [{', '.join(n.tags)}]" if n.tags else ""
            lines.append(f"- ({n.id}){tag_str} {n.text}")
        return "\n".join(lines)

    # ---- Skill library tools -----------------------------------------------

    def save_skill(name: str, description: str, body: str, tags: list[str] | None = None) -> str:
        if skill_library is None:
            return "error: no skill library configured"
        s = skill_library.save(name=name, description=description, body=body, tags=tags or [])
        return f"saved skill '{s.name}' ({len(s.body)} chars)"

    def search_skills(query: str, k: int = 3) -> str:
        if skill_library is None:
            return "error: no skill library configured"
        hits = skill_library.search(query, k)
        if not hits:
            return "no skills match"
        return "\n\n".join(s.render() for s in hits)

    def load_skill(name: str) -> str:
        if skill_library is None:
            return "error: no skill library configured"
        s = skill_library.load(name)
        if s is None:
            return f"error: no skill named {name!r}"
        return s.render()

    # ---- Subagent delegation -----------------------------------------------

    def delegate(
        prompt: str,
        role: str = "executor",
        max_usd: float | None = 0.50,
        max_turns: int | None = 8,
    ) -> str:
        """Spawn a subagent run, wait for it, return its final answer.

        Cost rolls up to the parent run; child events are forwarded to the
        parent's event stream as child_run_started / child_run_completed.
        """
        if runtime is None or current_agent is None:
            return "error: delegation requires a Runtime"
        from agi.runtime import RunSpec, Budget  # local import — avoid cycles

        if current_agent.depth + 1 > runtime.max_subagent_depth:
            return f"error: subagent depth limit reached ({runtime.max_subagent_depth})"

        budget = Budget(max_usd=max_usd, max_turns=max_turns)
        spec = RunSpec(
            prompt=prompt,
            role=role,
            model=current_agent.model,
            effort=current_agent.effort,
            budget=budget,
            parent_run_id=current_agent._rid(),
        )
        # Notify the parent's event sink that a child is starting, with the
        # prompt and role for downstream observability.
        child = runtime.submit(spec, depth=current_agent.depth + 1)
        if current_agent.event_sink is not None:
            from agi import events as _ev
            current_agent.event_sink.emit(_ev.child_run_started(
                current_agent._rid(), child.id, role, prompt,
            ))
            # Track child id on the parent run for the manifest.
            if hasattr(current_agent.event_sink, "child_run_ids"):
                current_agent.event_sink.child_run_ids.append(child.id)

        child.wait()
        if current_agent.event_sink is not None and child.result is not None:
            from agi import events as _ev
            current_agent.event_sink.emit(_ev.child_run_completed(
                current_agent._rid(),
                child.id,
                child.result.text,
                child.result.usage,
                child.result.cost_usd,
            ))

        # Roll cost into parent for accurate parent-run accounting.
        current_agent.usage.input_tokens += child.usage.input_tokens
        current_agent.usage.output_tokens += child.usage.output_tokens
        current_agent.usage.cache_creation_input_tokens += child.usage.cache_creation_input_tokens
        current_agent.usage.cache_read_input_tokens += child.usage.cache_read_input_tokens

        if child.status == "completed" and child.result is not None:
            return (
                f"[subagent: {role} / run {child.id}] "
                f"${child.result.cost_usd:.4f} / {child.usage.turns} turns\n\n"
                f"{child.result.text}"
            )
        return f"[subagent failed: {child.status}] {child.error or '(no message)'}"

    # ---- Tool synthesis ----------------------------------------------------

    def make_tool(
        name: str,
        description: str,
        code: str,
        input_schema: dict | None = None,
    ) -> str:
        """Compile a Python function and register it as a tool for this session.

        The `code` must define exactly one top-level function named `name`.
        It is exec'd with full stdlib access (no greater attack surface than
        `run_bash`), and its callable is added to the live agent's tool
        handler map.
        """
        if current_agent is None:
            return "error: make_tool requires an active agent"
        if not name.isidentifier():
            return f"error: {name!r} is not a valid identifier"
        if name in current_agent.handlers:
            return f"error: tool {name!r} already exists"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"error: syntax error in code: {e}"
        func_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(func_defs) != 1 or func_defs[0].name != name:
            return f"error: code must define exactly one function named {name!r}"

        ns: dict[str, Any] = {}  # exec auto-populates __builtins__
        try:
            exec(compile(tree, f"<make_tool:{name}>", "exec"), ns)
        except Exception as e:
            return f"error: failed to compile {name!r}: {type(e).__name__}: {e}"
        fn = ns.get(name)
        if not callable(fn):
            return f"error: {name!r} did not produce a callable"

        schema = {
            "name": name,
            "description": description,
            "input_schema": input_schema or {
                "type": "object",
                "properties": {},
            },
        }
        current_agent.handlers[name] = fn  # type: ignore[assignment]
        current_agent.tool_schemas.append(schema)
        return f"registered tool {name!r} ({len(current_agent.handlers)} tools available)"

    schemas: list[dict[str, Any]] = [
        {
            "name": "read_file",
            "description": "Read a UTF-8 text file and return its contents.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file."}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file, overwriting if it exists. Creates parent directories.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write to."},
                    "content": {"type": "string", "description": "UTF-8 content to write."},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "list_dir",
            "description": "List entries in a directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Defaults to '.'.", "default": "."},
                },
            },
        },
        {
            "name": "run_bash",
            "description": "Run a bash command and return combined stdout/stderr plus the exit code.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                    "timeout_seconds": {"type": "integer", "description": "Kill after this many seconds (default 30).", "default": 30},
                },
                "required": ["command"],
            },
        },
        {
            "name": "save_memory",
            "description": (
                "Save a note to long-term memory (persists across sessions). "
                "Use this for facts the user tells you, useful patterns you "
                "discover, and intermediate results worth recalling later."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The note to remember."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for retrieval.",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "search_memory",
            "description": "Search long-term memory by keyword across note text and tags.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "k": {"type": "integer", "description": "Max results (default 5).", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "recent_memory",
            "description": "Return the k most recent notes from long-term memory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "k": {"type": "integer", "description": "Number of recent notes (default 10).", "default": 10},
                },
            },
        },
    ]

    handlers: dict[str, Callable[..., str]] = {
        "read_file": read_file,
        "write_file": write_file,
        "list_dir": list_dir,
        "run_bash": run_bash,
        "save_memory": save_memory,
        "search_memory": search_memory,
        "recent_memory": recent_memory,
    }

    if skill_library is not None:
        schemas.extend([
            {
                "name": "save_skill",
                "description": (
                    "Persist a reusable procedure to the skill library. Use when you"
                    " discover a generalisable pattern worth re-running on similar tasks."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short identifier."},
                        "description": {"type": "string", "description": "When to use this skill."},
                        "body": {"type": "string", "description": "The procedure, as a numbered list or prose."},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
                    },
                    "required": ["name", "description", "body"],
                },
            },
            {
                "name": "search_skills",
                "description": "Find skills relevant to a query string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "load_skill",
                "description": "Load a named skill by exact name.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        ])
        handlers.update({
            "save_skill": save_skill,
            "search_skills": search_skills,
            "load_skill": load_skill,
        })

    if runtime is not None and current_agent is not None:
        schemas.append({
            "name": "delegate",
            "description": (
                "Spawn a subagent run for a self-contained sub-task. Roles:"
                " 'planner' produces a step list, 'executor' runs a single step,"
                " 'critic' reviews a candidate answer, 'researcher' gathers"
                " web evidence, 'general' is a fresh general-purpose agent."
                " Cost rolls up to this run."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The task the subagent must accomplish."},
                    "role": {
                        "type": "string",
                        "enum": ["planner", "executor", "critic", "researcher", "general"],
                        "default": "executor",
                    },
                    "max_usd": {"type": "number", "default": 0.5, "description": "Cost ceiling for the subagent in USD."},
                    "max_turns": {"type": "integer", "default": 8, "description": "Max model turns for the subagent."},
                },
                "required": ["prompt"],
            },
        })
        handlers["delegate"] = delegate

    if current_agent is not None:
        schemas.append({
            "name": "make_tool",
            "description": (
                "Register a new Python tool for the remainder of this session."
                " Provide a function definition; it must define exactly one"
                " top-level function with the given name. Use for genuinely"
                " novel sub-capabilities; don't redefine existing tools."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool identifier (must be a valid Python identifier)."},
                    "description": {"type": "string", "description": "What the tool does and when to use it."},
                    "code": {"type": "string", "description": "Python source defining exactly one top-level function named `name`."},
                    "input_schema": {"type": "object", "description": "JSON schema for the tool's arguments. Optional."},
                },
                "required": ["name", "description", "code"],
            },
        })
        handlers["make_tool"] = make_tool

    return schemas, handlers
