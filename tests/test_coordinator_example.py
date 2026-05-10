"""Tests for examples/coordinator.py.

Imports the example as a module and drives it with a Runtime backed by
FakeAgent so no API calls happen. Verifies the coordinator consumes the
runtime contract correctly: parallel sessions, rollup cost, metrics.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.runtime import Runtime
from tests.test_runtime import FakeAgent


def _load_coordinator_module():
    name = "coordinator_example"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / "coordinator.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module via sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCoordinatorExample(unittest.TestCase):
    def setUp(self):
        # The coordinator's observer prints to stdout for live tracing in
        # demo runs; capture it so the test output stays clean.
        self._stdout_ctx = contextlib.redirect_stdout(io.StringIO())
        self._stdout_ctx.__enter__()

    def tearDown(self):
        self._stdout_ctx.__exit__(None, None, None)

    def test_run_one_returns_record(self):
        mod = _load_coordinator_module()
        rt = Runtime(agent_factory=FakeAgent)
        record = mod.run_one(rt, mod.WorkItem(role="executor", prompt="hi"))
        self.assertEqual(record.work.role, "executor")
        self.assertEqual(record.text, "ok")
        self.assertIsNone(record.error)
        self.assertGreaterEqual(record.duration_s, 0.0)

    def test_coordinate_fans_out_to_three_sessions(self):
        mod = _load_coordinator_module()
        rt = Runtime(agent_factory=FakeAgent)
        summary = mod.coordinate("test prompt", rt, max_workers=3)
        self.assertEqual(summary["fanout"], 3)
        roles = {r["role"] for r in summary["results"]}
        self.assertEqual(roles, {"planner", "researcher", "executor"})
        # Each result has a non-error preview
        for r in summary["results"]:
            self.assertIsNone(r["error"])
            self.assertEqual(r["text_preview"], "ok")
        # Metrics rolled up
        m = summary["metrics"]
        self.assertEqual(m["sessions"]["created"], 3)
        self.assertEqual(m["turns"]["completed"], 3)
        self.assertSetEqual(set(m["by_role"]), {"planner", "researcher", "executor"})


if __name__ == "__main__":
    unittest.main()
