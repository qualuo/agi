"""Tools the agent can call.

`make_tools(memory, ...)` returns a `ToolRegistry` exposing schemas for
the API request and handlers for dispatch on `tool_use` blocks. The
registry is dynamic — synthesized tools added during a session are
visible to the next API call.

File and shell tools touch the host directly — run this in a sandbox if
you don't trust the model's choices.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agi.memory import Memory
from agi.sandbox import SandboxError, SynthesizedTool, call_tool, compile_tool
from agi.skills import SkillLibrary


@dataclass
class ToolRegistry:
    """Mutable registry of tools available to the agent.

    Holds schemas (sent to the API), handlers (dispatched on tool_use),
    and a small audit log of which tools were called this session.
    Synthesized tools register here at runtime via `register_synth`.
    """

    schemas: list[dict[str, Any]] = field(default_factory=list)
    handlers: dict[str, Callable[..., str]] = field(default_factory=dict)
    synth: dict[str, SynthesizedTool] = field(default_factory=dict)
    call_log: list[str] = field(default_factory=list)

    # ---- registration ----

    def register(self, schema: dict[str, Any], handler: Callable[..., str]) -> None:
        if schema["name"] in self.handlers:
            raise ValueError(f"tool already registered: {schema['name']}")
        self.schemas.append(schema)
        self.handlers[schema["name"]] = handler

    def register_synth(self, tool: SynthesizedTool, *, timeout: int = 5) -> None:
        if tool.name in self.handlers:
            raise ValueError(f"tool already registered: {tool.name}")
        # Wrap to enforce timeout, stringify result, surface errors.
        def _handler(**kwargs) -> str:
            try:
                result = call_tool(tool, kwargs, timeout_seconds=timeout)
            except Exception as e:  # noqa: BLE001
                return f"error in synth tool {tool.name}: {type(e).__name__}: {e}"
            return _stringify(result)

        # Synthesized tools accept arbitrary kwargs — the schema declares an
        # open object. The model is told to pass the same keyword names the
        # function declares.
        schema = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        }
        self.schemas.append(schema)
        self.handlers[tool.name] = _handler
        self.synth[tool.name] = tool

    def remove(self, name: str) -> bool:
        if name not in self.handlers:
            return False
        self.schemas = [s for s in self.schemas if s.get("name") != name]
        self.handlers.pop(name, None)
        self.synth.pop(name, None)
        return True

    # ---- introspection ----

    def names(self) -> list[str]:
        return list(self.handlers)

    def descriptors(self) -> list[dict[str, Any]]:
        """Lightweight summaries (name + description) for capability listings."""
        return [
            {"name": s["name"], "description": s.get("description", "")}
            for s in self.schemas
        ]

    # ---- dispatch ----

    def dispatch(self, name: str, kwargs: dict[str, Any]) -> tuple[str, bool]:
        handler = self.handlers.get(name)
        if handler is None:
            return f"error: unknown tool {name}", True
        self.call_log.append(name)
        try:
            return handler(**(kwargs or {})), False
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}", True


def _stringify(result) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


def make_tools(
    memory: Memory,
    *,
    skills: Optional[SkillLibrary] = None,
    delegate_fn: Optional[Callable[..., str]] = None,
    enable_synth: bool = True,
) -> ToolRegistry:
    """Build the default tool registry.

    `delegate_fn(role, task)` is optional — when provided, the `delegate`
    tool spawns a sub-agent with that role on the given task. The runtime
    is the natural place to wire this; we don't import Agent here to
    avoid a cycle.
    """
    registry = ToolRegistry()

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

    base_schemas: list[dict[str, Any]] = [
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
    base_handlers: dict[str, Callable[..., str]] = {
        "read_file": read_file,
        "write_file": write_file,
        "list_dir": list_dir,
        "run_bash": run_bash,
        "save_memory": save_memory,
        "search_memory": search_memory,
        "recent_memory": recent_memory,
    }
    for s in base_schemas:
        registry.register(s, base_handlers[s["name"]])

    # ---- skills ----

    if skills is not None:
        def list_skills(query: str = "") -> str:
            entries = skills.search(query, k=10) if query else skills.all()
            if not entries:
                return "(no skills)"
            return "\n".join(
                f"- ({s.id}) {s.title}" + (f" [triggers: {', '.join(s.triggers)}]" if s.triggers else "")
                for s in entries
            )

        def read_skill(skill_id: str) -> str:
            s = skills.get(skill_id)
            if s is None:
                return f"error: no skill with id {skill_id!r}"
            return s.to_prompt_block()

        def add_skill(
            title: str,
            when: str,
            procedure: str,
            failure_modes: str = "(none recorded)",
            triggers: list[str] | None = None,
        ) -> str:
            s = skills.add(
                title=title,
                when=when,
                procedure=procedure,
                failure_modes=failure_modes,
                triggers=triggers or [],
            )
            return f"added skill {s.id}: {s.title}"

        registry.register(
            {
                "name": "list_skills",
                "description": "List skills, optionally filtered by a search query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Optional search query.", "default": ""},
                    },
                },
            },
            list_skills,
        )
        registry.register(
            {
                "name": "read_skill",
                "description": "Read a skill by id. Returns when-to-use, procedure, failure modes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string", "description": "Skill id from list_skills."},
                    },
                    "required": ["skill_id"],
                },
            },
            read_skill,
        )
        registry.register(
            {
                "name": "add_skill",
                "description": (
                    "Save a new procedural skill to the library. Use this when "
                    "you discover a reusable procedure that future tasks could "
                    "apply directly. Triggers should be keywords that retrieve it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short skill title."},
                        "when": {"type": "string", "description": "When this skill applies."},
                        "procedure": {"type": "string", "description": "Step-by-step procedure."},
                        "failure_modes": {"type": "string", "description": "Known failure modes.", "default": "(none recorded)"},
                        "triggers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Trigger keywords for retrieval.",
                        },
                    },
                    "required": ["title", "when", "procedure"],
                },
            },
            add_skill,
        )

    # ---- self-extension: synthesize a tool at runtime ----

    if enable_synth:
        def make_tool(name: str, description: str, source: str) -> str:
            try:
                tool = compile_tool(name, description, source)
                registry.register_synth(tool)
            except SandboxError as e:
                return f"error: {e}"
            except ValueError as e:
                return f"error: {e}"
            return (
                f"registered tool {name}: {description}. "
                f"It is now callable in subsequent turns."
            )

        registry.register(
            {
                "name": "make_tool",
                "description": (
                    "Create a new Python tool you can call on subsequent turns. "
                    "The source must define a top-level function with the given "
                    "name; only stdlib modules math/statistics/re/json/datetime/"
                    "itertools/collections/functools/string/hashlib/base64/textwrap "
                    "may be imported. Use this when a sub-task is a small "
                    "well-defined function you'll re-use this session."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "snake_case function name."},
                        "description": {"type": "string", "description": "What the tool does (shown to you on later calls)."},
                        "source": {"type": "string", "description": "Python source defining the function."},
                    },
                    "required": ["name", "description", "source"],
                },
            },
            make_tool,
        )

    # ---- delegation: spawn a sub-agent with a role ----

    if delegate_fn is not None:
        def delegate(role: str, task: str) -> str:
            try:
                return delegate_fn(role=role, task=task)
            except Exception as e:  # noqa: BLE001
                return f"error in delegate({role}): {type(e).__name__}: {e}"

        registry.register(
            {
                "name": "delegate",
                "description": (
                    "Spawn a sub-agent with a specialized role and a focused "
                    "sub-task. Returns the sub-agent's final answer. Roles: "
                    "'planner' (decompose the task), 'executor' (carry out a "
                    "focused step), 'critic' (review work for issues). Use this "
                    "when a sub-task is genuinely independent of the main "
                    "thread; coordination overhead means flat is often cheaper."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["planner", "executor", "critic"]},
                        "task": {"type": "string", "description": "The focused sub-task."},
                    },
                    "required": ["role", "task"],
                },
            },
            delegate,
        )

    return registry
