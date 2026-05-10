"""Tests for the world model and the synthesized-tool guardrails."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.tools_extension import _safe_compile, make_extension_tools
from agi.world_model import WorldModel


def test_world_model_records_and_indexes(tmp_path):
    wm = WorldModel(path=tmp_path / "world.jsonl")
    wm.observe(kind="file", id="/a", action="read", outcome="success")
    wm.observe(kind="file", id="/a", action="write", outcome="failure",
               detail={"errno": 13})
    latest = wm.latest("file", "/a")
    assert latest is not None
    assert latest.action == "write"
    assert latest.outcome == "failure"
    s = wm.summary()
    assert s["entity_counts"]["file"] == 1
    assert s["recent_failures"]


def test_world_model_survives_reload(tmp_path):
    p = tmp_path / "w.jsonl"
    wm = WorldModel(path=p)
    wm.observe(kind="url", id="https://x.test/", action="fetch")
    wm2 = WorldModel(path=p)
    assert wm2.latest("url", "https://x.test/") is not None


def test_safe_compile_allows_basic_function():
    fn = _safe_compile("def add(a, b):\n    return a + b\n", "add")
    assert fn(2, 3) == 5


def test_safe_compile_rejects_imports():
    with pytest.raises(ValueError):
        _safe_compile("def f():\n    import os\n    return os\n", "f")


def test_safe_compile_rejects_eval_and_open():
    with pytest.raises(ValueError):
        _safe_compile("def f():\n    return eval('1+1')\n", "f")
    with pytest.raises(ValueError):
        _safe_compile("def f():\n    return open('/etc/passwd').read()\n", "f")


def test_safe_compile_rejects_dunder_access():
    with pytest.raises(ValueError):
        _safe_compile("def f(x):\n    return x.__class__\n", "f")


def test_make_tool_registers_and_runs(tmp_path):
    schemas, handlers, registry = make_extension_tools(persistent_dir=tmp_path)
    out = handlers["make_tool"](
        name="square",
        description="square a number",
        code="def square(x):\n    return x * x\n",
        args_schema={"type": "object", "properties": {"x": {"type": "number"}},
                     "required": ["x"]},
    )
    assert "defined" in out
    assert "square" in registry
    assert registry["square"]["fn"](5) == 25


def test_make_tool_rejects_protected_names(tmp_path):
    _, handlers, _ = make_extension_tools(persistent_dir=tmp_path)
    out = handlers["make_tool"](name="read_file", description="x",
                                code="def read_file():\n    return 1\n")
    assert out.startswith("error:")
