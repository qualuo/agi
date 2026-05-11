"""Tests for agi.sandbox tool synthesis."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.sandbox import SandboxError, stringify, synthesize_tool


class TestSandbox(unittest.TestCase):
    def test_simple_pure_function_works(self) -> None:
        tool = synthesize_tool(
            name="add_two",
            description="add two numbers",
            input_schema={"type": "object", "properties": {"a": {}, "b": {}}},
            source="def add_two(a, b):\n    return a + b\n",
        )
        self.assertEqual(tool.func(a=2, b=3), 5)

    def test_preloaded_modules_available(self) -> None:
        tool = synthesize_tool(
            name="hash_md5",
            description="md5",
            input_schema={},
            source="def hash_md5(s):\n    return hashlib.md5(s.encode()).hexdigest()\n",
        )
        self.assertEqual(len(tool.func(s="hi")), 32)

    def test_json_module_available(self) -> None:
        tool = synthesize_tool(
            name="parse_int_list",
            description="parse a json list",
            input_schema={},
            source="def parse_int_list(s):\n    return sum(json.loads(s))\n",
        )
        self.assertEqual(tool.func(s="[1, 2, 3]"), 6)

    def test_import_rejected(self) -> None:
        # Import inside the function body — passes the "single top-level
        # function" check, then the AST walker rejects the import.
        with self.assertRaises(SandboxError) as ctx:
            synthesize_tool(
                name="bad",
                description="bad",
                input_schema={},
                source="def bad():\n    import os\n    return os.listdir('.')\n",
            )
        self.assertIn("import", str(ctx.exception).lower())

    def test_module_level_import_rejected(self) -> None:
        with self.assertRaises(SandboxError):
            synthesize_tool(
                name="bad",
                description="bad",
                input_schema={},
                source="import os\ndef bad():\n    return 1\n",
            )

    def test_eval_rejected(self) -> None:
        with self.assertRaises(SandboxError):
            synthesize_tool(
                name="bad",
                description="bad",
                input_schema={},
                source="def bad(s):\n    return eval(s)\n",
            )

    def test_dunder_attr_rejected(self) -> None:
        with self.assertRaises(SandboxError):
            synthesize_tool(
                name="bad",
                description="bad",
                input_schema={},
                source="def bad(x):\n    return x.__class__\n",
            )

    def test_name_mismatch_rejected(self) -> None:
        with self.assertRaises(SandboxError):
            synthesize_tool(
                name="declared",
                description="",
                input_schema={},
                source="def actual():\n    return 1\n",
            )

    def test_multiple_top_level_functions_rejected(self) -> None:
        with self.assertRaises(SandboxError):
            synthesize_tool(
                name="a",
                description="",
                input_schema={},
                source="def a():\n    return 1\ndef b():\n    return 2\n",
            )

    def test_runtime_error_surfaced(self) -> None:
        tool = synthesize_tool(
            name="divz",
            description="",
            input_schema={},
            source="def divz():\n    return 1 / 0\n",
        )
        with self.assertRaises(SandboxError) as ctx:
            tool.func()
        self.assertIn("ZeroDivisionError", str(ctx.exception))

    def test_stringify_json_friendly(self) -> None:
        self.assertEqual(stringify("hi"), "hi")
        self.assertEqual(stringify([1, 2, 3]), "[1, 2, 3]")
        self.assertEqual(stringify({"a": 1}), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()
