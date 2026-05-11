"""Tool synthesis tests — pure Python, no API."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.tool_synthesis import make_tool_synthesis


class TestToolSynthesis(unittest.TestCase):
    def setUp(self):
        # Seed with a "core" tool name that synthesis cannot overwrite.
        self.schemas = [{"name": "run_bash", "description": "...", "input_schema": {}}]
        self.handlers = {"run_bash": lambda **kw: "fake bash"}
        self.schema, self.handler = make_tool_synthesis(self.schemas, self.handlers)

    def test_registers_new_tool(self):
        code = "def add(a, b):\n    return a + b\n"
        out = self.handler(
            name="add",
            description="add two ints",
            input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            code=code,
        )
        self.assertEqual(out, "registered add")
        self.assertIn("add", self.handlers)
        result = self.handlers["add"](a=2, b=3)
        self.assertEqual(result, "5")

    def test_returns_json_for_dict_result(self):
        code = "def kv():\n    return {'k': 'v'}\n"
        self.handler(name="kv", description="", input_schema={"type": "object"}, code=code)
        result = self.handlers["kv"]()
        self.assertIn('"k"', result)
        self.assertIn('"v"', result)

    def test_rejects_protected_name(self):
        code = "def run_bash():\n    return 'pwned'\n"
        out = self.handler(name="run_bash", description="", input_schema={}, code=code)
        self.assertIn("error", out)
        self.assertIn("reserved", out)
        # The original handler must still be intact.
        self.assertEqual(self.handlers["run_bash"](), "fake bash")

    def test_rejects_invalid_identifier(self):
        out = self.handler(name="not a name", description="", input_schema={}, code="x = 1")
        self.assertIn("error", out)

    def test_rejects_disallowed_import(self):
        code = "import subprocess\n\ndef pwn():\n    return subprocess.check_output(['id'])\n"
        out = self.handler(name="pwn", description="", input_schema={}, code=code)
        self.assertIn("error", out)
        self.assertIn("subprocess", out)
        self.assertNotIn("pwn", self.handlers)

    def test_rejects_disallowed_from_import(self):
        code = "from os import system\n\ndef pwn():\n    return system('echo hi')\n"
        out = self.handler(name="pwn", description="", input_schema={}, code=code)
        self.assertIn("error", out)
        self.assertNotIn("pwn", self.handlers)

    def test_allows_whitelisted_import(self):
        code = "import math\n\ndef sqrt(x):\n    return math.sqrt(x)\n"
        out = self.handler(name="sqrt", description="", input_schema={}, code=code)
        self.assertEqual(out, "registered sqrt")
        self.assertEqual(self.handlers["sqrt"](x=4), "2.0")

    def test_syntax_error_surfaced(self):
        out = self.handler(name="bad", description="", input_schema={}, code="def bad(:\n  ...")
        self.assertIn("error", out)
        self.assertIn("Syntax", out)
        self.assertNotIn("bad", self.handlers)

    def test_handler_catches_runtime_error(self):
        code = "def divz():\n    return 1/0\n"
        self.handler(name="divz", description="", input_schema={}, code=code)
        result = self.handlers["divz"]()
        self.assertIn("error", result)
        self.assertIn("ZeroDivision", result)

    def test_redefinition_replaces_schema(self):
        code1 = "def f():\n    return 1\n"
        code2 = "def f():\n    return 2\n"
        self.handler(name="f", description="v1", input_schema={}, code=code1)
        self.handler(name="f", description="v2", input_schema={}, code=code2)
        f_schemas = [s for s in self.schemas if s.get("name") == "f"]
        self.assertEqual(len(f_schemas), 1)
        self.assertEqual(f_schemas[0]["description"], "v2")
        self.assertEqual(self.handlers["f"](), "2")


if __name__ == "__main__":
    unittest.main()
