"""Tests for agi.budget."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.budget import Budget
from agi.costs import Usage


class TestBudget(unittest.TestCase):
    def test_within_budget_returns_none(self) -> None:
        b = Budget(max_usd=1.0, max_tokens=1_000_000, max_wall_seconds=60)
        u = Usage(input_tokens=100, output_tokens=100)
        self.assertIsNone(b.check(u, "claude-opus-4-7"))

    def test_token_ceiling_trips(self) -> None:
        b = Budget(max_tokens=100)
        u = Usage(input_tokens=80, output_tokens=80)
        self.assertIsNotNone(b.check(u, "claude-opus-4-7"))

    def test_usd_ceiling_trips(self) -> None:
        b = Budget(max_usd=0.0001)
        u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        reason = b.check(u, "claude-opus-4-7")
        self.assertIsNotNone(reason)
        self.assertIn("usd", reason)

    def test_wall_ceiling_trips(self) -> None:
        b = Budget(max_wall_seconds=0.05)
        time.sleep(0.1)
        self.assertIsNotNone(b.check(Usage(), "claude-opus-4-7"))

    def test_no_limits_set_returns_none(self) -> None:
        self.assertIsNone(Budget().check(Usage(input_tokens=10**9), "claude-opus-4-7"))

    def test_reset_clock(self) -> None:
        b = Budget(max_wall_seconds=0.05)
        time.sleep(0.1)
        self.assertIsNotNone(b.check(Usage(), "claude-opus-4-7"))
        b.reset_clock()
        self.assertIsNone(b.check(Usage(), "claude-opus-4-7"))


if __name__ == "__main__":
    unittest.main()
