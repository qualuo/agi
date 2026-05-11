"""Tools the agent can call.

`make_tools(memory, skills=None)` returns a `(schemas, handlers)` pair: the
JSON schemas go in the API request, the handlers are dispatched on
`tool_use` blocks. File and shell tools touch the host directly — run this
in a sandbox if you don't trust the model's choices.

Skill tools (`save_skill`, `list_skills`) only register if a SkillLibrary
is provided. The reflection tool always registers and writes a `lesson`
tag to long-term memory.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from agi.memory import Memory
from agi.skills import SkillLibrary


def make_tools(
    memory: Memory,
    skills: SkillLibrary | None = None,
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

    def reflect(lesson: str, tags: list[str] | None = None) -> str:
        """Write a durable lesson to memory with a 'lesson' tag.

        Used at the end of a task so future runs can find what worked
        and what didn't. This is the per-task learning channel.
        """
        all_tags = list({*(tags or []), "lesson"})
        note = memory.save(lesson, all_tags)
        return f"saved lesson {note.id}"

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
        {
            "name": "reflect",
            "description": (
                "Write a durable lesson to long-term memory tagged 'lesson'. "
                "Use at end of a task to capture what worked / what didn't, "
                "so future runs of related tasks can find it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "lesson": {"type": "string", "description": "The lesson, in 1-3 sentences."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional extra tags (besides 'lesson').",
                    },
                },
                "required": ["lesson"],
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
        "reflect": reflect,
    }

    if skills is not None:
        def save_skill(
            name: str,
            description: str,
            body: str,
            tags: list[str] | None = None,
        ) -> str:
            s = skills.save(name=name, description=description, body=body, tags=tags or [])
            return f"saved skill '{s.name}' to {s.path}"

        def list_skills(query: str | None = None, k: int = 5) -> str:
            results = (
                skills.search(query, k=k) if query else skills.all()[:k]
            )
            if not results:
                return "(no skills)"
            lines = []
            for s in results:
                tag_str = f" [{', '.join(s.tags)}]" if s.tags else ""
                lines.append(f"- {s.name}{tag_str}: {s.description}")
            return "\n".join(lines)

        schemas.extend(
            [
                {
                    "name": "save_skill",
                    "description": (
                        "Save a procedural skill (markdown SOP) for future tasks. "
                        "Use this after solving a class of problem so the next "
                        "instance is cheaper. Body is markdown: include when to "
                        "use it, the procedure, and known failure modes."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Short name (used as filename slug)."},
                            "description": {"type": "string", "description": "One-line trigger description used for retrieval."},
                            "body": {"type": "string", "description": "Markdown body: when/how/gotchas."},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags.",
                            },
                        },
                        "required": ["name", "description", "body"],
                    },
                },
                {
                    "name": "list_skills",
                    "description": "List skills currently in the library; optionally filter by query.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Optional keyword filter."},
                            "k": {"type": "integer", "description": "Max results (default 5).", "default": 5},
                        },
                    },
                },
            ]
        )
        handlers["save_skill"] = save_skill
        handlers["list_skills"] = list_skills

    return schemas, handlers


def make_delegate_tool(runtime, parent_id: str) -> tuple[dict, Callable[..., str]]:
    """Build a `delegate` tool bound to a specific parent session.

    The Session attaches this after construction (it can't go in
    `make_tools` because Memory/SkillLibrary don't carry a Runtime
    reference).
    """
    def delegate(task: str, role: str = "executor", max_usd: float | None = 0.50) -> str:
        from agi.budget import Budget
        budget = Budget(max_usd=max_usd, max_turns=20, model=runtime.model) if max_usd else None
        result = runtime.delegate(
            parent_id=parent_id,
            task=task,
            role=role,
            budget=budget,
        )
        return (
            f"[child {result['session_id']} {result['stop_reason']} "
            f"${result['usage']['cost_usd']:.4f}]\n{result['text']}"
        )

    schema = {
        "name": "delegate",
        "description": (
            "Spawn a child agent to handle one well-scoped subtask. The child "
            "runs to completion with its own budget, then returns its final "
            "text. Use for: parallelizable subtasks, role-specialized work "
            "(e.g. researcher), or to keep the parent's context clean. "
            "Available roles: planner, executor, critic, researcher, general."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The full subtask description."},
                "role": {"type": "string", "description": "Role for the child (default 'executor')."},
                "max_usd": {
                    "type": "number",
                    "description": "Cost ceiling for the child (default 0.50).",
                },
            },
            "required": ["task"],
        },
    }
    return schema, delegate
