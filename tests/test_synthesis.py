"""Tool synthesis tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.synthesis import ToolForge, ToolSynthesisError


class TestToolForge(unittest.TestCase):
    def setUp(self):
        self.forge = ToolForge(call_timeout_seconds=3.0)

    def test_compile_and_call(self):
        tool = self.forge.compile(
            name="reverse",
            description="Reverse a string.",
            code="def reverse(s):\n    return s[::-1]\n",
            input_schema={"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]},
            smoke_test={"s": "abc"},
            expected="cba",
        )
        out = tool.handler(s="hello")
        self.assertEqual(out["result"], "olleh")

    def test_rejects_bad_name(self):
        with self.assertRaises(ToolSynthesisError):
            self.forge.compile("not-a-name", "x", "def x(): pass\n", {})

    def test_rejects_disallowed_import(self):
        code = "import socket\ndef bad():\n    return 0\n"
        with self.assertRaises(ToolSynthesisError):
            self.forge.compile("bad", "x", code, {})

    def test_rejects_syntax_error(self):
        with self.assertRaises(ToolSynthesisError):
            self.forge.compile("ok", "x", "def ok(:\n    pass\n", {})

    def test_rejects_missing_function(self):
        with self.assertRaises(ToolSynthesisError):
            self.forge.compile("missing", "x", "x = 1\n", {})

    def test_failed_smoke_test(self):
        code = "def add(a, b):\n    return a + b\n"
        with self.assertRaises(ToolSynthesisError):
            self.forge.compile(
                name="add",
                description="x",
                code=code,
                input_schema={},
                smoke_test={"a": 1, "b": 1},
                expected=3,
            )

    def test_allows_safe_stdlib_import(self):
        code = "import math\ndef circle_area(r):\n    return math.pi * r * r\n"
        tool = self.forge.compile(
            name="circle_area",
            description="x",
            code=code,
            input_schema={"type": "object", "properties": {"r": {"type": "number"}}, "required": ["r"]},
        )
        self.assertAlmostEqual(tool.handler(r=1.0)["result"], 3.141592, places=4)


if __name__ == "__main__":
    unittest.main()
