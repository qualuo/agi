"""Tool synthesis — the agent extends its own capabilities.

The agent calls `make_tool(name, description, code, input_schema)`. The code
is a Python source string that must define a `run(**kwargs) -> str` function.
We validate via AST, run a quick smoke test in a subprocess sandbox, and on
success register a handler that invokes the code in a fresh sandboxed
subprocess on every call.

Sandbox model:
- Each invocation is a fresh `python -I` subprocess (isolated mode: no
  site-packages auto-import, no PYTHON* env vars).
- Working directory is a temp dir scoped to the call.
- Input is passed as a JSON blob on stdin; output is read as JSON from
  stdout. The subprocess can't reach the parent's memory.
- Wall-clock timeout enforced by `subprocess.run(timeout=…)`.
- An AST scan rejects obvious foot-guns at registration time (imports of
  `os.system`, `subprocess`, `socket`; calls to `eval`/`exec`/`open` —
  unless explicitly allowed at registration).

This isn't a hard security boundary against an adversarial code generator —
that needs OS-level sandboxing (firejail, container, seccomp). It's a
practical "the model occasionally writes broken code" guard. Deployments
that care should run the whole runtime under a real sandbox.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


DEFAULT_BANNED_IMPORTS = {"os", "subprocess", "socket", "shutil", "ctypes", "multiprocessing"}
DEFAULT_BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open"}


class ToolSynthError(Exception):
    pass


@dataclass
class SynthesizedTool:
    name: str
    description: str
    code: str
    input_schema: dict[str, Any]
    created_ts: float = field(default_factory=time.time)
    invocation_count: int = 0
    error_count: int = 0


def _scan_code(
    code: str,
    *,
    banned_imports: set[str] | None = None,
    banned_calls: set[str] | None = None,
) -> None:
    banned_imports = banned_imports if banned_imports is not None else DEFAULT_BANNED_IMPORTS
    banned_calls = banned_calls if banned_calls is not None else DEFAULT_BANNED_CALLS

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ToolSynthError(f"syntax error: {e}") from e

    has_run = any(
        isinstance(node, ast.FunctionDef) and node.name == "run"
        for node in tree.body
    )
    if not has_run:
        raise ToolSynthError("code must define a `run(**kwargs) -> str` function at module top level")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in banned_imports:
                    raise ToolSynthError(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in banned_imports:
                raise ToolSynthError(f"banned import: {node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in banned_calls:
                raise ToolSynthError(f"banned call: {func.id}")


_RUNNER = """\
import json, sys, traceback
_USER_CODE = {code!r}
_NS = {{}}
try:
    exec(compile(_USER_CODE, "<synthesized>", "exec"), _NS, _NS)
    if "run" not in _NS:
        print(json.dumps({{"ok": False, "error": "no run() defined"}}))
        sys.exit(0)
    kwargs = json.loads(sys.stdin.read() or "{{}}")
    result = _NS["run"](**kwargs)
    if not isinstance(result, str):
        result = json.dumps(result, default=str)
    print(json.dumps({{"ok": True, "result": result}}))
except SystemExit:
    raise
except BaseException as e:
    tb = "".join(traceback.format_exception_only(type(e), e)).strip()
    print(json.dumps({{"ok": False, "error": tb}}))
"""


def _invoke_subprocess(
    code: str,
    kwargs: dict[str, Any],
    *,
    timeout_seconds: float = 5.0,
) -> str:
    """Run the synthesized tool's `run(**kwargs)` in a sandboxed subprocess.

    Returns the str result (or stringified result). Raises on failure.
    """
    runner = _RUNNER.format(code=code)
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                input=json.dumps(kwargs).encode("utf-8"),
                capture_output=True,
                timeout=timeout_seconds,
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin"},
            )
        except subprocess.TimeoutExpired as e:
            raise ToolSynthError(f"timeout after {timeout_seconds}s") from e

    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    if not stdout:
        raise ToolSynthError(f"empty output. stderr={proc.stderr.decode('utf-8', errors='replace')[:400]}")
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as e:
        raise ToolSynthError(f"non-JSON output: {stdout[:400]}") from e
    if not payload.get("ok"):
        raise ToolSynthError(payload.get("error", "unknown error"))
    return payload["result"]


class ToolSynthRegistry:
    """Holds synthesized tools and produces schemas + handlers for them.

    The Agent re-reads `schemas()` and `handlers()` between turns so newly
    synthesized tools become available immediately on the next turn.
    """

    def __init__(
        self,
        *,
        banned_imports: set[str] | None = None,
        banned_calls: set[str] | None = None,
        invoke_timeout: float = 5.0,
        smoke_test_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.banned_imports = banned_imports
        self.banned_calls = banned_calls
        self.invoke_timeout = invoke_timeout
        self.smoke_test_kwargs = smoke_test_kwargs or {}
        self._tools: dict[str, SynthesizedTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        code: str,
        input_schema: dict[str, Any] | None = None,
        smoke_test_kwargs: dict[str, Any] | None = None,
    ) -> SynthesizedTool:
        if not name.isidentifier() or name.startswith("_"):
            raise ToolSynthError(f"invalid tool name: {name!r}")
        if name in self._tools:
            raise ToolSynthError(f"tool already registered: {name}")
        _scan_code(code, banned_imports=self.banned_imports, banned_calls=self.banned_calls)
        # Smoke test: run with provided sample kwargs (or empty). If the
        # tool actually requires kwargs the smoke test will fail, which is
        # fine — we surface it to the model so it retries with the right
        # signature.
        sample = smoke_test_kwargs or self.smoke_test_kwargs or {}
        try:
            _invoke_subprocess(code, sample, timeout_seconds=self.invoke_timeout)
        except ToolSynthError as e:
            raise ToolSynthError(f"smoke test failed: {e}") from e

        tool = SynthesizedTool(
            name=name,
            description=description,
            code=code,
            input_schema=input_schema or {"type": "object", "properties": {}},
        )
        self._tools[name] = tool
        return tool

    def list(self) -> list[SynthesizedTool]:
        return list(self._tools.values())

    def get(self, name: str) -> SynthesizedTool | None:
        return self._tools.get(name)

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def handlers(self) -> dict[str, Callable[..., str]]:
        out: dict[str, Callable[..., str]] = {}
        for tool in self._tools.values():
            out[tool.name] = self._make_handler(tool)
        return out

    def _make_handler(self, tool: SynthesizedTool) -> Callable[..., str]:
        def handler(**kwargs):
            tool.invocation_count += 1
            try:
                return _invoke_subprocess(tool.code, kwargs, timeout_seconds=self.invoke_timeout)
            except ToolSynthError as e:
                tool.error_count += 1
                return f"error: {e}"
        handler.__name__ = f"synthesized_{tool.name}"
        return handler
