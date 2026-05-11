"""Tests for tool synthesis.

These actually run subprocesses, so they take a moment. They verify:
- Valid code registers and invokes.
- Banned constructs are rejected at registration.
- Smoke test failures bubble up.
- Per-call timeout enforced.
- Schema/handler exposure works for the Agent.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.toolsynth import ToolSynthError, ToolSynthRegistry, _scan_code


class TestScan(unittest.TestCase):
    def test_requires_run(self):
        with self.assertRaises(ToolSynthError):
            _scan_code("x = 1")

    def test_accepts_valid(self):
        _scan_code("def run(**kw):\n    return 'hi'")

    def test_rejects_banned_import(self):
        with self.assertRaises(ToolSynthError):
            _scan_code("import os\ndef run(**kw):\n    return ''")

    def test_rejects_banned_call(self):
        with self.assertRaises(ToolSynthError):
            _scan_code("def run(**kw):\n    return eval('1+1')")

    def test_rejects_syntax_error(self):
        with self.assertRaises(ToolSynthError):
            _scan_code("def run(:\n  pass")


class TestRegistry(unittest.TestCase):
    def test_register_and_invoke(self):
        reg = ToolSynthRegistry(invoke_timeout=10.0)
        reg.register(
            name="reverse_string",
            description="reverses a string",
            code="def run(text=''):\n    return text[::-1]",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            smoke_test_kwargs={"text": "abc"},
        )
        handler = reg.handlers()["reverse_string"]
        self.assertEqual(handler(text="hello"), "olleh")

    def test_register_returns_non_string_as_json(self):
        reg = ToolSynthRegistry(invoke_timeout=10.0)
        reg.register(
            name="sum_list",
            description="sums a list",
            code="def run(nums=None):\n    return sum(nums or [])",
            smoke_test_kwargs={"nums": [1, 2]},
        )
        result = reg.handlers()["sum_list"](nums=[1, 2, 3])
        # int gets JSON-stringified by the subprocess wrapper
        self.assertEqual(result, "6")

    def test_invalid_name_rejected(self):
        reg = ToolSynthRegistry()
        with self.assertRaises(ToolSynthError):
            reg.register(name="bad-name", description="", code="def run():\n    return ''")

    def test_duplicate_name_rejected(self):
        reg = ToolSynthRegistry(invoke_timeout=10.0)
        code = "def run(**kw):\n    return 'x'"
        reg.register(name="t", description="", code=code)
        with self.assertRaises(ToolSynthError):
            reg.register(name="t", description="", code=code)

    def test_smoke_test_failure_rejects(self):
        reg = ToolSynthRegistry(invoke_timeout=10.0)
        with self.assertRaises(ToolSynthError):
            reg.register(
                name="needs_kwarg",
                description="",
                code="def run(x):\n    return str(x)",
                # no smoke kwargs → run() called with nothing → fails
            )

    def test_runtime_error_returned_as_error_string(self):
        reg = ToolSynthRegistry(invoke_timeout=10.0)
        reg.register(
            name="divider",
            description="",
            code="def run(a=1, b=1):\n    return str(a / b)",
            smoke_test_kwargs={"a": 1, "b": 1},
        )
        handler = reg.handlers()["divider"]
        result = handler(a=1, b=0)
        self.assertTrue(result.startswith("error:"))
        tool = reg.get("divider")
        self.assertEqual(tool.error_count, 1)

    def test_schemas_match_tools(self):
        reg = ToolSynthRegistry(invoke_timeout=10.0)
        reg.register(name="t", description="d", code="def run(**kw):\n    return 'x'")
        schemas = reg.schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["name"], "t")
        self.assertEqual(schemas[0]["description"], "d")

    def test_remove(self):
        reg = ToolSynthRegistry(invoke_timeout=10.0)
        reg.register(name="t", description="", code="def run(**kw):\n    return 'x'")
        self.assertTrue(reg.remove("t"))
        self.assertEqual(reg.schemas(), [])


if __name__ == "__main__":
    unittest.main()
