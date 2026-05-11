"""Sandboxed Python execution for tool synthesis.

The agent can `define_tool(name, description, code)` at runtime. The body
is a Python function the agent writes. To register it, we:

1. Lint the source AST for disallowed nodes (no imports of os/socket/subprocess,
   no `__import__`, no `exec`/`eval`, no attribute access on `__builtins__`).
2. Compile it in a restricted globals dict — a small allow-list of safe
   builtins and a fixed set of stdlib modules pre-imported as locals.
3. Wrap the resulting callable in a process-level timeout via a watchdog
   thread (best-effort; CPython can't actually interrupt arbitrary native
   code, so this is a soft guarantee).

This is **not a security sandbox** in the cryptographic sense — a determined
adversary can still escape via gadgets in pure Python. It's a guardrail
against the model accidentally calling `os.system('rm -rf /')` because it
hallucinated a useful tool. For untrusted models, run the whole runtime
inside a container or VM.
"""
from __future__ import annotations

import ast
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "bytes": bytes,
    "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
    "hash": hash, "hex": hex, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "oct": oct,
    "ord": ord, "pow": pow, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip, "True": True,
    "False": False, "None": None, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "ArithmeticError": ArithmeticError,
    "Exception": Exception, "print": print,
}

# Modules safe to expose to synthesized tools — pure-computation only.
import math as _math
import statistics as _statistics
import re as _re
import json as _json
import datetime as _datetime
import itertools as _itertools
import functools as _functools
import collections as _collections
import string as _string
import hashlib as _hashlib
import base64 as _base64
import textwrap as _textwrap
import urllib.parse as _urlparse

SAFE_MODULES = {
    "math": _math,
    "statistics": _statistics,
    "re": _re,
    "json": _json,
    "datetime": _datetime,
    "itertools": _itertools,
    "functools": _functools,
    "collections": _collections,
    "string": _string,
    "hashlib": _hashlib,
    "base64": _base64,
    "textwrap": _textwrap,
    "urlparse": _urlparse,
}

DISALLOWED_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input", "globals",
    "locals", "vars", "dir", "getattr", "setattr", "delattr", "hasattr",
    "memoryview", "exit", "quit", "help", "breakpoint",
}

DISALLOWED_ATTRS = {
    "__builtins__", "__globals__", "__class__", "__bases__", "__subclasses__",
    "__mro__", "__dict__", "__getattr__", "__getattribute__", "__import__",
    "__loader__", "__spec__", "__code__", "__closure__", "__func__",
    "f_back", "f_globals", "f_locals", "f_builtins",
}


class SandboxError(Exception):
    pass


def _lint(source: str) -> ast.FunctionDef:
    """Parse the source, ensure it defines exactly one function, and walk
    the AST rejecting any disallowed nodes. Returns the function definition
    node so callers can read its name."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}") from e

    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1:
        raise SandboxError(f"expected exactly one top-level function, got {len(fns)}")
    if len(tree.body) != len(fns):
        raise SandboxError("only a single function definition is permitted at module scope")
    fn = fns[0]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("imports are not allowed; required modules are preloaded (math, json, re, ...)")
        if isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:
            raise SandboxError(f"use of disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in DISALLOWED_ATTRS:
            raise SandboxError(f"access to disallowed attribute: .{node.attr}")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SandboxError("global/nonlocal statements are not allowed")
        if isinstance(node, ast.AsyncFunctionDef):
            raise SandboxError("async functions are not allowed")
        if isinstance(node, (ast.Lambda,)):
            pass  # lambdas are fine
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Cheap defense against dunder-string evasion
            if "__" in node.value and any(d in node.value for d in DISALLOWED_ATTRS):
                raise SandboxError(f"string literal contains disallowed dunder: {node.value!r}")
    return fn


@dataclass
class SynthTool:
    name: str
    description: str
    input_schema: dict
    source: str
    func: Callable[..., Any]
    created_ts: float = field(default_factory=time.time)


def synthesize_tool(
    *, name: str, description: str, input_schema: dict, source: str, timeout_sec: float = 5.0
) -> SynthTool:
    """Compile a synthesized tool from source.

    The source must define exactly one top-level function named `name`.
    Returns a SynthTool whose `func` is the wrapped, time-limited callable.
    """
    if not name.isidentifier():
        raise SandboxError(f"tool name must be a Python identifier: {name!r}")

    fn_node = _lint(source)
    if fn_node.name != name:
        raise SandboxError(f"function name {fn_node.name!r} does not match tool name {name!r}")

    g: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, **SAFE_MODULES}
    try:
        code = compile(source, f"<synth:{name}>", "exec")
        exec(code, g)
    except Exception as e:
        raise SandboxError(f"compile/exec failed: {type(e).__name__}: {e}") from e

    raw = g.get(name)
    if not callable(raw):
        raise SandboxError(f"compiled object {name!r} is not callable")

    def _wrapped(**kwargs: Any) -> Any:
        result_box: dict[str, Any] = {}

        def _run() -> None:
            try:
                result_box["value"] = raw(**kwargs)
            except Exception as e:  # bubble up as SandboxError so caller sees it
                result_box["error"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            raise SandboxError(f"synthesized tool {name!r} timed out after {timeout_sec}s")
        if "error" in result_box:
            raise SandboxError(result_box["error"])
        return result_box.get("value")

    return SynthTool(
        name=name,
        description=description,
        input_schema=input_schema,
        source=source,
        func=_wrapped,
    )


def stringify(value: Any) -> str:
    """Convert a synthesized-tool return value into a string suitable for
    handing back to the model as a tool_result."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except TypeError:
        return repr(value)
