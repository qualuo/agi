"""Tools the agent can call.

`make_tools(memory, ...)` returns a `(schemas, handlers)` pair: the JSON schemas
go in the API request, the handlers are dispatched on `tool_use` blocks.
File and shell tools touch the host directly — run this in a sandbox if you
don't trust the model's choices.

Optional components, when supplied, expose extra tools:
- `skills` → list_skills, add_skill
- `synth` → define_tool, list_synth_tools, promote_tool
- `delegate_fn` → delegate (spawn a subagent in a named role)
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from agi.memory import Memory
from agi.sandbox import SandboxError, stringify as _sandbox_stringify


def make_tools(
    memory: Memory,
    *,
    skills=None,            # agi.skills.SkillLibrary | None
    synth=None,             # agi.synth_registry.SynthToolRegistry | None
    delegate_fn=None,       # Callable[[str, str, dict|None], str] | None
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

    schemas: list[dict[str, Any]] = [
        {
            "name": "read_file",
            "description": "Read a UTF-8 text file and return its contents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                },
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

    # --- Skill library tools (optional) --------------------------------------
    if skills is not None:
        def list_skills() -> str:
            items = skills.all()
            if not items:
                return "(no skills registered)"
            return "\n".join(f"- {s.name} (uses={s.usage_count}): {s.when_to_use}" for s in items)

        def add_skill(name: str, when_to_use: str, procedure: str, failure_modes: str = "") -> str:
            skill = skills.add(name, when_to_use, procedure, failure_modes)
            return f"saved skill '{skill.name}' to {skill.path}"

        schemas.extend([
            {
                "name": "list_skills",
                "description": "List all skills in the procedural-memory library.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "add_skill",
                "description": (
                    "Save a new procedural skill (a named procedure for a class "
                    "of task). Use sparingly — only after you've solved a "
                    "non-trivial task and want the next instance to be cheaper. "
                    "The 'procedure' field should be a numbered list of concrete "
                    "steps. The 'failure_modes' field is optional but valuable."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short, distinctive name."},
                        "when_to_use": {"type": "string", "description": "One-line trigger condition."},
                        "procedure": {"type": "string", "description": "Numbered list of steps."},
                        "failure_modes": {"type": "string", "description": "Known failure modes + recovery."},
                    },
                    "required": ["name", "when_to_use", "procedure"],
                },
            },
        ])
        handlers["list_skills"] = list_skills
        handlers["add_skill"] = add_skill

    # --- Synthesized-tool registry (optional) --------------------------------
    if synth is not None:
        def define_tool(name: str, description: str, source: str, input_schema: dict | None = None) -> str:
            try:
                tool = synth.define(
                    name=name,
                    description=description,
                    input_schema=input_schema or {"type": "object", "properties": {}},
                    source=source,
                )
            except SandboxError as e:
                return f"error: {e}"
            return f"defined tool '{tool.name}' (session-scoped). Call promote_tool('{tool.name}') to persist."

        def list_synth_tools() -> str:
            items = synth.all()
            if not items:
                return "(no synthesized tools)"
            return "\n".join(f"- {n}: {t.description}" for n, t in items.items())

        def promote_tool(name: str) -> str:
            ok = synth.promote(name)
            return f"promoted '{name}' to persistent storage" if ok else f"no such tool: {name}"

        def call_synth(name: str, args: dict | None = None) -> str:
            tool = synth.all().get(name)
            if tool is None:
                return f"error: no synthesized tool named {name!r}"
            try:
                result = tool.func(**(args or {}))
            except SandboxError as e:
                return f"error: {e}"
            return _sandbox_stringify(result)

        schemas.extend([
            {
                "name": "define_tool",
                "description": (
                    "Define a new tool at runtime as a pure-Python function. "
                    "The 'source' must contain exactly one top-level function "
                    "whose name matches 'name'. Imports are not allowed; the "
                    "modules math, statistics, re, json, datetime, itertools, "
                    "functools, collections, string, hashlib, base64, textwrap, "
                    "and urlparse are pre-loaded. The function runs in a "
                    "best-effort sandbox with a 5s timeout. Tools defined "
                    "this way are session-scoped until promote_tool is called."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Python identifier; matches the function name."},
                        "description": {"type": "string", "description": "What the tool does."},
                        "source": {"type": "string", "description": "Python source defining one function."},
                        "input_schema": {"type": "object", "description": "JSON schema for the function's kwargs.", "additionalProperties": True},
                    },
                    "required": ["name", "description", "source"],
                },
            },
            {
                "name": "list_synth_tools",
                "description": "List all currently-defined synthesized tools.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "promote_tool",
                "description": "Promote a session-scoped synthesized tool to persistent disk storage.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "call_synth",
                "description": (
                    "Invoke a previously-defined synthesized tool by name. "
                    "Use this when the tool isn't yet visible as a first-class "
                    "tool in the current session (synthesized tools register "
                    "with the API on the next turn)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "args": {"type": "object", "description": "kwargs passed to the function.", "additionalProperties": True},
                    },
                    "required": ["name"],
                },
            },
        ])
        handlers["define_tool"] = define_tool
        handlers["list_synth_tools"] = list_synth_tools
        handlers["promote_tool"] = promote_tool
        handlers["call_synth"] = call_synth

    # --- Subagent delegation (optional) --------------------------------------
    if delegate_fn is not None:
        def delegate(role: str, task: str, max_usd: float | None = None) -> str:
            try:
                return delegate_fn(role, task, {"max_usd": max_usd} if max_usd else None)
            except Exception as e:
                return f"error: {type(e).__name__}: {e}"

        schemas.append({
            "name": "delegate",
            "description": (
                "Spawn a subagent in a named role and wait for its result. "
                "Available roles: planner, executor, critic, researcher, "
                "coder, summarizer. Use delegation for clearly-decomposable "
                "subtasks or when a cheaper model can handle a sub-step. "
                "Subagent token usage rolls up to this task."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "One of: planner, executor, critic, researcher, coder, summarizer."},
                    "task": {"type": "string", "description": "The subtask to delegate."},
                    "max_usd": {"type": "number", "description": "Optional USD budget for the subagent."},
                },
                "required": ["role", "task"],
            },
        })
        handlers["delegate"] = delegate

    return schemas, handlers
