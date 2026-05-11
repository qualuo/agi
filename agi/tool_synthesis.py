"""Runtime tool synthesis.

The agent writes a Python function and registers it as a callable tool
for the rest of the session. This is the "self-extension" primitive
described in PLAN.md Stage 2: novel-task pass rate ought to climb when
the agent can mint exactly the tool it needs.

Safety: we exec arbitrary Python the model writes. This is not safe in
the abstract — only safe in contexts where the operator already trusts
the model with shell access (which the harness gives by default via
`run_bash`). The synthesis layer adds:

- A restricted import allowlist. The exec'd code can import only
  standard-library modules we've vetted as side-effect-light. No
  `socket`, no `subprocess`, no `os` (use the `run_bash` tool for
  shell). To enable broader imports, run with `allow_imports=...`.
- A wall-clock timeout enforced via signal (POSIX) or thread (Windows
  fallback).
- A handler that returns the *function's return value* stringified —
  so synthesized tools can't smuggle non-JSON-able objects into the
  Messages API.

This is intentionally minimal. Sandboxing arbitrary Python in-process is
not solvable; the right long-term answer is running synthesized tools in
a subprocess or container. v1 keeps it in-process and trusts the same
trust boundary as `run_bash`.
"""
from __future__ import annotations

import ast
import json
import signal
import threading
from typing import Any, Callable


_DEFAULT_ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "json",
        "re",
        "random",
        "statistics",
        "itertools",
        "collections",
        "functools",
        "datetime",
        "time",
        "string",
        "textwrap",
        "hashlib",
        "base64",
        "urllib.parse",
        "uuid",
    }
)


class _ToolRegistry:
    """The mutable handler+schema state the agent edits via make_tool.

    Wrap the Agent's dicts so changes inside the handler closure mutate
    the live tool surface for the next turn.
    """

    def __init__(
        self,
        schemas: list[dict],
        handlers: dict[str, Callable[..., str]],
        protected: set[str],
    ) -> None:
        self.schemas = schemas
        self.handlers = handlers
        self.protected = protected

    def register(self, name: str, schema: dict, handler: Callable[..., str]) -> None:
        if name in self.protected:
            raise ValueError(f"tool name '{name}' is reserved and cannot be overwritten")
        # Replace existing schema with same name if present (allow refinement).
        self.schemas[:] = [s for s in self.schemas if s.get("name") != name]
        self.schemas.append(schema)
        self.handlers[name] = handler


def _validate_imports(code: str, allowed: frozenset[str]) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if alias.name not in allowed and root not in allowed:
                    raise ValueError(f"import '{alias.name}' not in allowlist")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if mod not in allowed and root not in allowed:
                raise ValueError(f"import 'from {mod}' not in allowlist")


def _compile_function(code: str, fn_name: str, allowed: frozenset[str]) -> Callable[..., Any]:
    _validate_imports(code, allowed)
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    exec(compile(code, f"<synthesized:{fn_name}>", "exec"), namespace)
    fn = namespace.get(fn_name)
    if not callable(fn):
        raise ValueError(f"code did not define a callable named '{fn_name}'")
    return fn


def _run_with_timeout(fn: Callable[..., Any], kwargs: dict[str, Any], timeout_s: float) -> Any:
    """POSIX: SIGALRM. Fallback: thread + join with timeout (no kill, but
    we surface the timeout to the caller)."""
    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        def _handler(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"synthesized tool exceeded {timeout_s}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return fn(**kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _target():
        try:
            result["v"] = fn(**kwargs)
        except BaseException as e:  # noqa: BLE001
            error["e"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        # We can't safely kill the thread; surface the timeout.
        raise TimeoutError(f"synthesized tool exceeded {timeout_s}s (thread still running)")
    if "e" in error:
        raise error["e"]
    return result.get("v")


def make_tool_synthesis(
    schemas: list[dict],
    handlers: dict[str, Callable[..., str]],
    *,
    allowed_imports: frozenset[str] | None = None,
    default_timeout_s: float = 5.0,
) -> tuple[dict, Callable[..., str]]:
    """Build the `make_tool` tool, bound to the agent's live tool surface.

    Existing tool names are protected — synthesized tools cannot shadow
    core tools like `run_bash` or `delegate`.
    """
    allowed = allowed_imports or _DEFAULT_ALLOWED_IMPORTS
    protected = {s["name"] for s in schemas if "name" in s}
    registry = _ToolRegistry(schemas, handlers, protected)

    schema = {
        "name": "make_tool",
        "description": (
            "Define a new callable tool for the rest of this session. Provide:\n"
            "- name: identifier (snake_case)\n"
            "- description: when to use it\n"
            "- input_schema: JSON Schema (object) describing parameters\n"
            "- code: Python source defining a function with the same name; "
            "imports limited to: " + ", ".join(sorted(allowed)) + ".\n"
            "Returns 'registered <name>' on success. The tool is callable "
            "from the next assistant turn. Use this when the existing "
            "toolset is missing exactly one thing — not as a substitute "
            "for run_bash."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "input_schema": {"type": "object"},
                "code": {"type": "string"},
            },
            "required": ["name", "description", "input_schema", "code"],
        },
    }

    def handler(name: str, description: str, input_schema: dict, code: str) -> str:
        if not name.isidentifier():
            return f"error: '{name}' is not a valid Python identifier"
        try:
            fn = _compile_function(code, name, allowed)
        except SyntaxError as e:
            return f"error: SyntaxError: {e}"
        except ValueError as e:
            return f"error: {e}"

        def synthesized(**kwargs) -> str:
            try:
                result = _run_with_timeout(fn, kwargs, default_timeout_s)
            except Exception as e:  # noqa: BLE001 — surface every failure
                return f"error: {type(e).__name__}: {e}"
            if isinstance(result, str):
                return result
            try:
                return json.dumps(result, default=str)
            except (TypeError, ValueError):
                return repr(result)

        try:
            registry.register(
                name,
                {"name": name, "description": description, "input_schema": input_schema},
                synthesized,
            )
        except ValueError as e:
            return f"error: {e}"
        return f"registered {name}"

    return schema, handler
