"""Tests for Agent snapshot/restore (no API calls)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import Agent
from agi.costs import Usage


def _bare_agent() -> Agent:
    a = Agent.__new__(Agent)
    a.model = "claude-opus-4-7"
    a.messages = []
    a.usage = Usage()
    a.extra_system = None
    return a


class TestAgentSnapshot(unittest.TestCase):
    def test_snapshot_round_trip_preserves_usage(self):
        a = _bare_agent()
        a.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        a.usage = Usage(input_tokens=120, output_tokens=80, turns=1)
        a.extra_system = "use git carefully"

        snap = a.snapshot()
        b = _bare_agent()
        b.restore(snap)

        self.assertEqual(b.messages, a.messages)
        self.assertEqual(b.usage.input_tokens, 120)
        self.assertEqual(b.usage.output_tokens, 80)
        self.assertEqual(b.usage.turns, 1)
        self.assertEqual(b.extra_system, "use git carefully")

    def test_restore_with_empty_snapshot_is_safe(self):
        b = _bare_agent()
        b.restore({})
        self.assertEqual(b.messages, [])
        self.assertEqual(b.usage.turns, 0)


if __name__ == "__main__":
    unittest.main()
