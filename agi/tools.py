"""Tools the agent can call.

`make_tools(memory, skills=None, runtime=None)` returns a `(schemas, handlers)`
pair: the JSON schemas go in the API request, the handlers are dispatched on
`tool_use` blocks. File and shell tools touch the host directly — run this in
a sandbox if you don't trust the model's choices.

Extension hooks:
- `skills`: a `learner.SkillLibrary` instance. When passed, the agent gets
  `recall_skill`, `save_skill`, and `list_skills` tools.
- `runtime`: a callable `(task: str) -> str` for spawning a subagent
  (the runtime supplies one). When passed, the agent gets `delegate`.
- The `make_tool` capability registers a new tool at runtime via the
  returned `register_dynamic_tool` callable. The agent loop has to plumb
  that through — see `agi.Agent`.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any, Callable

from agi.memory import Memory


# Names that are safe to expose to user-synthesized tools. Everything else
# resolves to NameError at exec time. This is a deny-by-default sandbox, not
# a security boundary — a determined adversary can escape Python sandboxes.
# We keep the surface small to reduce accidental damage, not malicious damage.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "isinstance", "len", "list", "map", "max", "min", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "True", "False", "None",
    )
}


def _validate_tool_source(source: str, fn_name: str) -> None:
    """Reject obviously-dangerous syntax before exec.

    We're not trying to be airtight — Python sandboxing fundamentally
    isn't. We reject imports, attribute access on dunder names, and exec/eval
    calls. Combined with a stripped builtins dict, this catches casual
    mistakes and keeps trivially-malicious payloads from running.
    """
    tree = ast.parse(source)
    has_target = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("dynamic tools may not import modules")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id in ("exec", "eval", "compile", "open"):
            raise ValueError(f"reference to {node.id!r} is not allowed")
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            has_target = True
    if not has_target:
        raise ValueError(f"source must define a function named {fn_name!r}")


def _compile_dynamic_tool(name: str, source: str) -> Callable[..., Any]:
    """Compile a user-supplied Python function in a restricted namespace.

    Returns the function object. Raises ValueError if the source is rejected
    or the function fails to compile.
    """
    _validate_tool_source(source, name)
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    exec(compile(source, f"<dynamic_tool:{name}>", "exec"), namespace)  # noqa: S102
    fn = namespace.get(name)
    if not callable(fn):
        raise ValueError(f"compiled source did not produce a callable named {name!r}")
    return fn


def make_tools(
    memory: Memory,
    *,
    skills=None,
    runtime: Callable[[str], str] | None = None,
    on_register_tool: Callable[[dict, Callable[..., str]], None] | None = None,
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

    # Skill library tools — only registered if the library is provided.
    if skills is not None:
        from learner.skills import Skill

        def recall_skill(query: str, k: int = 3) -> str:
            found = skills.retrieve(query, k=k)
            if not found:
                return "no matching skills"
            return "\n\n".join(f"# {s.name}\n{s.body}" for s in found)

        def save_skill(name: str, body: str, triggers: list[str] | None = None) -> str:
            skill = Skill(name=name, triggers=triggers or [], body=body)
            target = skills.write(skill)
            return f"saved skill {skill.name!r} to {target}"

        def list_skills() -> str:
            items = skills.all()
            if not items:
                return "(no skills)"
            return "\n".join(f"- {s.name}: {', '.join(s.triggers) or '(no triggers)'}" for s in items)

        schemas.extend([
            {
                "name": "recall_skill",
                "description": (
                    "Retrieve relevant skills (named SOPs) from the skill library "
                    "given a task description. Use at the start of unfamiliar tasks "
                    "to check whether a known procedure applies."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Task description / keywords."},
                        "k": {"type": "integer", "description": "Max skills to return (default 3).", "default": 3},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "save_skill",
                "description": (
                    "Save a reusable procedure to the skill library. Use when "
                    "you've just solved a task whose procedure will likely apply "
                    "again. Body should be a short markdown SOP: when to use, "
                    "steps, known pitfalls."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short slug for the skill."},
                        "body": {"type": "string", "description": "Markdown body of the procedure."},
                        "triggers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords that should retrieve this skill.",
                        },
                    },
                    "required": ["name", "body"],
                },
            },
            {
                "name": "list_skills",
                "description": "List all known skills with their trigger keywords.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ])
        handlers["recall_skill"] = recall_skill
        handlers["save_skill"] = save_skill
        handlers["list_skills"] = list_skills

    # Delegation: spawn a subagent for a bounded sub-task.
    if runtime is not None:
        def delegate(task: str, role: str | None = None) -> str:
            prefix = f"[role: {role}] " if role else ""
            return runtime(prefix + task)

        schemas.append({
            "name": "delegate",
            "description": (
                "Spawn a subagent to handle a bounded sub-task and return its "
                "final answer. Use for decomposable work where parallel or "
                "specialized handling helps. Subagent token usage rolls up to "
                "the parent. Avoid trivial uses — overhead matters."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The sub-task to delegate."},
                    "role": {
                        "type": "string",
                        "description": "Optional role hint (e.g. 'planner', 'researcher', 'critic').",
                    },
                },
                "required": ["task"],
            },
        })
        handlers["delegate"] = delegate

    # Tool synthesis: register a new tool at runtime if the agent supports it.
    if on_register_tool is not None:
        def make_tool(name: str, description: str, source: str) -> str:
            """Compile and register a Python tool of one argument named `text`.

            Convention to keep schemas simple: the synthesized function takes
            a single string `text` and returns a string. The model can
            structure inputs inside the string however it likes. Future:
            accept a JSONSchema and infer call signature.
            """
            try:
                fn = _compile_dynamic_tool(name, source)
            except (SyntaxError, ValueError) as e:
                return f"error: {type(e).__name__}: {e}"

            def handler(text: str = "") -> str:
                try:
                    result = fn(text)
                except Exception as e:  # tool errors return as text, never crash the agent
                    return f"error: {type(e).__name__}: {e}"
                return str(result)

            schema = {
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Single input string (the tool author decides the format)."},
                    },
                    "required": ["text"],
                },
            }
            on_register_tool(schema, handler)
            return f"registered tool {name!r}"

        schemas.append({
            "name": "make_tool",
            "description": (
                "Compile a Python function and register it as a callable tool "
                "for the rest of this session. The function must be named "
                "exactly as `name`, take a single string argument `text`, and "
                "return a string. No imports allowed. Use this when an "
                "ergonomic helper would beat repeated multi-step tool calls."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool name (also the function name in source)."},
                    "description": {"type": "string", "description": "What the tool does, shown to the model."},
                    "source": {"type": "string", "description": "Python source defining `def <name>(text: str) -> str`."},
                },
                "required": ["name", "description", "source"],
            },
        })
        handlers["make_tool"] = make_tool

    return schemas, handlers
