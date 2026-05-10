"""Tests for the tool-synthesis sandbox.

Covers compile-time AST checks (banned names, blocked imports, dunder
attribute access) and runtime behavior (callable returned, kwargs work,
restricted builtins).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.sandbox import SandboxError, call_tool, compile_tool


class TestCompileTool(unittest.TestCase):
    def test_basic_compiles_and_runs(self):
        src = "def add(a, b): return a + b"
        tool = compile_tool("add", "addition", src)
        self.assertEqual(tool.fn(a=2, b=3), 5)

    def test_invalid_name_rejected(self):
        with self.assertRaises(SandboxError):
            compile_tool("Bad-Name", "x", "def Bad_Name(): pass")
        with self.assertRaises(SandboxError):
            compile_tool("123abc", "x", "def x(): pass")

    def test_missing_function_rejected(self):
        with self.assertRaises(SandboxError):
            compile_tool("foo", "x", "x = 1")

    def test_blocked_import(self):
        with self.assertRaises(SandboxError):
            compile_tool("f", "x", "import os\ndef f(): return os.getcwd()")
        with self.assertRaises(SandboxError):
            compile_tool("f", "x", "import subprocess\ndef f(): return 1")

    def test_blocked_eval(self):
        with self.assertRaises(SandboxError):
            compile_tool("f", "x", "def f(s): return eval(s)")

    def test_blocked_open(self):
        with self.assertRaises(SandboxError):
            compile_tool("f", "x", "def f(p): return open(p).read()")

    def test_blocked_dunder_attribute(self):
        with self.assertRaises(SandboxError):
            compile_tool(
                "f", "x",
                "def f(): return ().__class__.__bases__"
            )

    def test_allowed_imports_work(self):
        src = "import math\ndef sq(x): return math.sqrt(x)"
        tool = compile_tool("sq", "sqrt", src)
        self.assertAlmostEqual(tool.fn(x=9), 3.0)

    def test_allowed_import_re(self):
        src = "import re\ndef m(s, p): return bool(re.search(p, s))"
        tool = compile_tool("m", "regex match", src)
        self.assertTrue(tool.fn(s="hello world", p=r"world"))


class TestCallTool(unittest.TestCase):
    def test_call_passes_kwargs(self):
        tool = compile_tool("rev", "reverse", "def rev(s): return s[::-1]")
        self.assertEqual(call_tool(tool, {"s": "abc"}), "cba")

    def test_call_propagates_exception(self):
        tool = compile_tool("div", "div", "def div(a, b): return a // b")
        with self.assertRaises(ZeroDivisionError):
            call_tool(tool, {"a": 1, "b": 0})


if __name__ == "__main__":
    unittest.main()
