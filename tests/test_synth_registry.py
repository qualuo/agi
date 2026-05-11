"""Tests for agi.synth_registry persistence."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.synth_registry import SynthToolRegistry


SOURCE = "def double(x):\n    return x * 2\n"


class TestSynthRegistry(unittest.TestCase):
    def test_define_then_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = SynthToolRegistry(root=tmp)
            tool = reg.define(
                name="double", description="x*2",
                input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
                source=SOURCE,
            )
            self.assertEqual(tool.func(x=4), 8)
            self.assertIn("double", reg.all())

    def test_promote_persists_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = SynthToolRegistry(root=tmp)
            reg.define(name="double", description="x*2", input_schema={}, source=SOURCE)
            self.assertTrue(reg.promote("double"))
            self.assertTrue((Path(tmp) / "double.json").exists())

    def test_persistent_loads_on_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = SynthToolRegistry(root=tmp)
            reg.define(name="double", description="x*2", input_schema={}, source=SOURCE)
            reg.promote("double")

            reg2 = SynthToolRegistry(root=tmp)
            self.assertIn("double", reg2.all())
            self.assertEqual(reg2.all()["double"].func(x=10), 20)

    def test_remove_clears_both_session_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = SynthToolRegistry(root=tmp)
            reg.define(name="double", description="", input_schema={}, source=SOURCE)
            reg.promote("double")
            self.assertTrue(reg.remove("double"))
            self.assertNotIn("double", reg.all())
            self.assertFalse((Path(tmp) / "double.json").exists())


if __name__ == "__main__":
    unittest.main()
