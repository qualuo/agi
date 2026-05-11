"""Tools the agent can call.

`make_tools(memory)` returns a `(schemas, handlers)` pair: the JSON schemas
go in the API request, the handlers are dispatched on `tool_use` blocks.
File and shell tools touch the host directly — run this in a sandbox if you
don't trust the model's choices.

Pass an optional `world_model` to auto-record file/shell interactions into
a WorldModel so a coordinator (or the agent itself) can answer "have I
touched this before, and how did it go?" without re-deriving from raw
conversation history. Two extra tools are exposed when set: `world_summary`
and `world_known`. Auto-recording is opt-in — `make_tools(memory)` without
a world_model behaves exactly as before.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from agi.memory import Memory
from agi.world_model import WorldModel


def make_tools(
    memory: Memory,
    world_model: WorldModel | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., str]]]:
    def _record(kind: str, id: str, action: str, outcome: str = "success", **detail) -> None:
        if world_model is None:
            return
        try:
            world_model.observe(kind=kind, id=id, action=action, outcome=outcome, detail=detail)
        except Exception:
            # World model failures must never break a tool call.
            pass

    def read_file(path: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            _record("file", str(p), "read", "failure", reason="missing")
            return f"error: {p} does not exist"
        if not p.is_file():
            _record("file", str(p), "read", "failure", reason="not_a_file")
            return f"error: {p} is not a file"
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            _record("file", str(p), "read", "failure", reason="not_utf8")
            return f"error: {p} is not a UTF-8 text file"
        _record("file", str(p), "read", "success", bytes=len(text.encode()))
        return text

    def write_file(path: str, content: str) -> str:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        _record("file", str(p), "write", "success", bytes=len(content.encode()))
        return f"wrote {len(content.encode())} bytes to {p}"

    def list_dir(path: str = ".") -> str:
        p = Path(path).expanduser()
        if not p.is_dir():
            _record("file", str(p), "read", "failure", reason="not_a_directory")
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
            _record("command", command, "run", "failure", reason="timeout")
            return f"error: command timed out after {timeout_seconds}s"
        out = result.stdout
        if result.stderr:
            out += ("\n" if out else "") + f"[stderr]\n{result.stderr}"
        out += f"\n[exit {result.returncode}]"
        _record(
            "command",
            command,
            "run",
            "success" if result.returncode == 0 else "failure",
            exit_code=result.returncode,
        )
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

    if world_model is not None:
        def world_summary() -> str:
            return json.dumps(world_model.summary(), default=str)

        def world_known(kind: str) -> str:
            obs = world_model.known(kind)
            if not obs:
                return f"(no {kind} observations recorded)"
            lines = [
                f"- {o.entity_id} [{o.action}/{o.outcome}]"
                for o in sorted(obs, key=lambda o: o.ts, reverse=True)[:50]
            ]
            return "\n".join(lines)

        schemas.extend([
            {
                "name": "world_summary",
                "description": (
                    "Summarize what the agent has interacted with this run "
                    "(file/url/command counts and recent failures). Use to "
                    "decide whether to retry, skip duplicate work, or warn "
                    "about a known-bad path."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "world_known",
                "description": (
                    "List entities the agent has observed of a given kind "
                    "(file, url, command, entity), most recent first."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["file", "url", "command", "entity"]},
                    },
                    "required": ["kind"],
                },
            },
        ])
        handlers["world_summary"] = world_summary
        handlers["world_known"] = world_known

    return schemas, handlers
