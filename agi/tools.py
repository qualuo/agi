"""Tools the agent can call.

`make_tools(memory)` returns a `(schemas, handlers)` pair: the JSON schemas
go in the API request, the handlers are dispatched on `tool_use` blocks.
File and shell tools touch the host directly — run this in a sandbox if you
don't trust the model's choices.

When invoked through agi.runtime.Runtime, an additional `delegate` tool is
registered: the agent can spawn a child session/job and synchronously wait
for its result. This is how the Agent itself becomes a coordinator.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from agi.memory import Memory


def make_tools(
    memory: Memory,
    *,
    runtime=None,
    parent_agent=None,
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

    # ---- delegate (only available when running under a Runtime) ----
    if runtime is not None:
        def delegate(
            prompt: str,
            role: str | None = None,
            budget_usd: float = 0.50,
            timeout_seconds: int = 300,
            max_iterations: int = 15,
        ) -> str:
            """Spawn a child agent through the Runtime and wait for the result.

            This is the agent's own subagent primitive. Cost is rolled up into
            the runtime's metrics; the parent's session id is recorded so an
            external coordinator can trace the call graph.
            """
            parent_session_id = getattr(parent_agent, "session_id", None) if parent_agent else None
            session = runtime.create_session(role=role, metadata={"parent_session_id": parent_session_id})
            parent_job_id = None  # parent job id isn't surfaced to the agent in v1
            job = runtime.submit(
                session.id,
                prompt,
                budget_usd=budget_usd,
                max_iterations=max_iterations,
                parent_job_id=parent_job_id,
                metadata={"parent_session_id": parent_session_id},
            )
            try:
                result = runtime.await_job(job.id, timeout=timeout_seconds)
            except TimeoutError:
                runtime.cancel(job.id)
                return f"error: subagent timed out after {timeout_seconds}s"
            if result.status == "succeeded":
                return result.output
            return f"error: subagent {result.status}: {result.error or '(no detail)'}"

        schemas.append({
            "name": "delegate",
            "description": (
                "Spawn a child agent in a fresh session through the runtime "
                "and wait for its final answer. Use to parallelize subtasks, "
                "isolate context, or run an independent role (e.g. 'critic'). "
                "Subagent cost counts against the runtime's total budget."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Task for the subagent."},
                    "role": {"type": "string", "description": "Optional role label for observability."},
                    "budget_usd": {"type": "number", "description": "Max $ for the subagent (default 0.50).", "default": 0.50},
                    "timeout_seconds": {"type": "integer", "description": "Wall-clock timeout (default 300).", "default": 300},
                    "max_iterations": {"type": "integer", "description": "Max agent loop iterations (default 15).", "default": 15},
                },
                "required": ["prompt"],
            },
        })
        handlers["delegate"] = delegate

    return schemas, handlers
