"""Runtime tool synthesis.

The agent can author its own tools. `ToolForge.compile(name, description, code,
input_schema)` evaluates a Python function in a constrained namespace, smoke-
tests it with the provided example inputs, and on success returns a
`(schema, handler)` pair that the Agent can plug into its tool surface for
the rest of the session.

Constraints v1:
- No file imports outside an allow-list (stdlib subset).
- A single top-level function definition matching the declared `name`.
- A wall-clock timeout per call.

This is deliberately conservative. The point of the feature is durable
self-extension, not arbitrary remote code execution; the agent gets a sharper
tool surface without giving the host process more capability than it already
has via `run_bash`.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Callable


# Stdlib modules tools may use. Conservative on purpose; expand on demand.
_ALLOWED_IMPORTS = {
    "math", "statistics", "json", "re", "itertools", "functools", "collections",
    "datetime", "decimal", "fractions", "random", "string", "textwrap",
    "base64", "hashlib", "urllib.parse", "csv", "io", "uuid",
}


class ToolSynthesisError(Exception):
    pass


@dataclass
class SynthesizedTool:
    name: str
    description: str
    code: str
    input_schema: dict
    handler: Callable[..., Any]

    def as_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _static_check(code: str, name: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ToolSynthesisError(f"syntax error: {e}") from e

    found_fn = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == name:
                found_fn = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod_names: list[str] = []
            if isinstance(node, ast.Import):
                mod_names = [alias.name for alias in node.names]
            else:
                mod_names = [node.module or ""]
            for m in mod_names:
                root = m.split(".")[0]
                if m not in _ALLOWED_IMPORTS and root not in _ALLOWED_IMPORTS:
                    raise ToolSynthesisError(
                        f"import {m!r} is not in the allow-list"
                    )
        elif isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Expr)):
            # Top-level expressions / assignments / class / async are fine.
            continue
        else:
            raise ToolSynthesisError(
                f"unsupported top-level node: {type(node).__name__}"
            )

    if not found_fn:
        raise ToolSynthesisError(
            f"code does not define a top-level function named {name!r}"
        )


def _exec_in_subprocess(code: str, name: str, kwargs: dict, q: "mp.Queue") -> None:
    try:
        ns: dict = {}
        exec(compile(code, f"<tool:{name}>", "exec"), ns, ns)
        fn = ns.get(name)
        if fn is None:
            q.put(("error", f"function {name!r} not found after exec"))
            return
        result = fn(**kwargs)
        q.put(("ok", result))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))


def _run_with_timeout(code: str, name: str, kwargs: dict, timeout: float) -> Any:
    # We re-exec the source in the worker to keep the parent free of side
    # effects from the user-authored module.
    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_exec_in_subprocess, args=(code, name, kwargs, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1)
        raise ToolSynthesisError(f"tool exceeded {timeout:.1f}s timeout")
    try:
        kind, payload = q.get_nowait()
    except Exception as e:
        raise ToolSynthesisError(f"tool produced no result: {e}") from e
    if kind != "ok":
        raise ToolSynthesisError(payload)
    return payload


class ToolForge:
    """Build new tools at runtime, smoke-test them, return runnable handlers."""

    def __init__(self, call_timeout_seconds: float = 5.0) -> None:
        self.call_timeout = call_timeout_seconds

    def compile(
        self,
        name: str,
        description: str,
        code: str,
        input_schema: dict,
        smoke_test: dict | None = None,
        expected: Any | None = None,
    ) -> SynthesizedTool:
        if not name.isidentifier():
            raise ToolSynthesisError(f"{name!r} is not a valid identifier")
        _static_check(code, name)

        timeout = self.call_timeout

        def handler(**kwargs: Any) -> Any:
            t0 = time.time()
            result = _run_with_timeout(code, name, kwargs, timeout)
            elapsed = time.time() - t0
            return {"result": result, "elapsed_seconds": round(elapsed, 4)}

        if smoke_test is not None:
            out = handler(**smoke_test)
            if expected is not None and out.get("result") != expected:
                raise ToolSynthesisError(
                    f"smoke test failed: got {out['result']!r}, expected {expected!r}"
                )

        return SynthesizedTool(
            name=name,
            description=description,
            code=code,
            input_schema=input_schema,
            handler=handler,
        )
