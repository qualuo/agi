"""Tests for capability introspection."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.capabilities import Capabilities, ToolSpec, describe_runtime
from agi.costs import Usage


class FakeAgentForCaps:
    def __init__(self):
        self.model = "claude-opus-4-7"
        self.usage = Usage()
        self.messages = []
        self.last_critic_score = None
        self.critic = None
        self.tracer = None
        self.tool_schemas = [
            {"name": "read_file", "description": "Read a file.", "input_schema": {"type": "object"}},
            {"name": "web_search", "type": "web_search_20260209"},
        ]
        self.handlers = {"read_file": lambda **kw: ""}


class TestCapabilities(unittest.TestCase):
    def test_describe_runtime_basic(self):
        caps = describe_runtime(lambda: FakeAgentForCaps())
        self.assertEqual(caps.model, "claude-opus-4-7")
        self.assertIn("input_per_mtok", caps.pricing)
        self.assertGreater(caps.pricing["input_per_mtok"], 0)

    def test_tools_classified(self):
        caps = describe_runtime(lambda: FakeAgentForCaps())
        names = {t.name: t for t in caps.tools}
        self.assertIn("read_file", names)
        self.assertEqual(names["read_file"].kind, "client")
        self.assertIn("web_search", names)
        self.assertEqual(names["web_search"].kind, "server")

    def test_features_reported(self):
        caps = describe_runtime(lambda: FakeAgentForCaps())
        self.assertTrue(caps.features["streaming"])
        self.assertFalse(caps.features["critic_gate"])
        self.assertFalse(caps.features["trace_logging"])

    def test_to_dict_is_json_safe(self):
        import json
        caps = describe_runtime(lambda: FakeAgentForCaps())
        encoded = json.dumps(caps.to_dict())
        self.assertIn("read_file", encoded)
        self.assertIn("claude-opus-4-7", encoded)

    def test_default_budget_present(self):
        caps = describe_runtime(lambda: FakeAgentForCaps())
        for key in ("max_iterations", "max_tokens", "max_cost_usd", "deadline_s"):
            self.assertIn(key, caps.default_budget)


if __name__ == "__main__":
    unittest.main()
