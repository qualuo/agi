"""Restricted exec for agent-authored tools.

The agent can synthesize new tools at runtime via the `make_tool` action.
This module compiles the proposed Python source in a restricted namespace
and returns a callable. The sandbox is best-effort: it blocks imports of
known-dangerous modules, denies writes to the global builtins, and runs
the function body under a wall-clock timeout. It is NOT a hostile-input
sandbox — assume the agent author is aligned and fix anything that
breaks that assumption with process-level isolation.

Why this exists in v1: tool synthesis is the cheapest test of
"self-extension." If the agent can author a function that solves a
sub-problem and call it on later turns, $/passed-task should fall on
repeat workloads. We measure that, not feel about it.
"""
from __future__ import annotations

import ast
import builtins
import math
import re
import signal
import statistics
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass


# Module-level imports the agent is allowed to use inside a synthesized tool.
# Permissive enough to be useful, restrictive enough that we don't accidentally
# enable arbitrary network/process control via a stray import.
ALLOWED_IMPORTS: dict[str, object] = {
    "math": math,
    "statistics": statistics,
    "re": re,
    "json": __import__("json"),
    "datetime": __import__("datetime"),
    "itertools": __import__("itertools"),
    "collections": __import__("collections"),
    "functools": __import__("functools"),
    "string": __import__("string"),
    "hashlib": __import__("hashlib"),
    "base64": __import__("base64"),
    "textwrap": textwrap,
}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Replacement __import__ that only allows whitelisted top-level modules.

    The static AST check is the first gate (rejects bad source); this
    runtime gate exists because the AST check can't always reason about
    import-from-package and similar edge cases.
    """
    root = (name or "").split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(
            f"sandbox forbids import of {name!r}; permitted: {sorted(ALLOWED_IMPORTS)}"
        )
    return ALLOWED_IMPORTS[root]


# Builtins the synthesized code can call. Excluded: exec, eval,
# open, compile, input, exit, quit, breakpoint, globals, locals, vars.
# `__import__` is replaced with a restricted version above.
SAFE_BUILTINS: dict[str, object] = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
        "format", "frozenset", "hash", "hex", "id", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min", "next",
        "object", "oct", "ord", "pow", "print", "property", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "type", "zip", "True", "False", "None", "Exception", "ValueError",
        "TypeError", "KeyError", "IndexError", "ZeroDivisionError",
        "ArithmeticError", "RuntimeError", "StopIteration",
    )
    if hasattr(builtins, name)
}
SAFE_BUILTINS["__import__"] = _restricted_import


class SandboxError(Exception):
    pass


@dataclass
class SynthesizedTool:
    name: str
    description: str
    fn: object  # callable
    source: str


_BANNED_NAMES = {
    "__import__", "exec", "eval", "open", "compile", "input",
    "globals", "locals", "vars", "breakpoint", "exit", "quit",
}


def _check_ast_safe(source: str) -> None:
    """Static check: reject source that uses banned names or unauthorized imports."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (
                node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            )
            root = (mod or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise SandboxError(
                    f"import of {mod!r} is not allowed; permitted: "
                    f"{sorted(ALLOWED_IMPORTS)}"
                )
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise SandboxError(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            # Block dunder-attribute access ("escape via __class__ etc.")
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in {
                "__name__", "__doc__"
            }:
                raise SandboxError(f"access to attribute {node.attr!r} is not allowed")


@contextmanager
def _wallclock(seconds: int):
    """SIGALRM-based wall-clock timeout. Unix only; falls open on Windows."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    def _handler(signum, frame):
        raise TimeoutError(f"sandbox timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def compile_tool(name: str, description: str, source: str) -> SynthesizedTool:
    """Compile agent-authored Python source into a callable tool.

    `source` must define a top-level function whose name matches `name`.
    The function is executed with restricted builtins; only the modules
    in ALLOWED_IMPORTS may be imported.
    """
    if not re.match(r"^[a-z_][a-z0-9_]*$", name):
        raise SandboxError(f"invalid tool name {name!r}")
    _check_ast_safe(source)

    # Restricted namespace: safe builtins and pre-imported modules.
    sandbox_globals: dict = {
        "__builtins__": SAFE_BUILTINS,
        **ALLOWED_IMPORTS,
    }
    try:
        exec(compile(source, f"<synth:{name}>", "exec"), sandbox_globals)
    except Exception as e:  # noqa: BLE001
        raise SandboxError(f"compile error: {type(e).__name__}: {e}") from e

    fn = sandbox_globals.get(name)
    if not callable(fn):
        raise SandboxError(f"source did not define a callable named {name!r}")
    return SynthesizedTool(name=name, description=description, fn=fn, source=source)


def call_tool(tool: SynthesizedTool, kwargs: dict, *, timeout_seconds: int = 5):
    """Invoke a synthesized tool under a wall-clock timeout."""
    with _wallclock(timeout_seconds):
        return tool.fn(**kwargs)
