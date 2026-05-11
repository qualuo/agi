"""Budget gate tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.budget import Budget, BudgetExceeded
from agi.costs import Usage


class TestBudget(unittest.TestCase):
    def test_unbounded_never_raises(self):
        b = Budget()
        b.check(Usage(input_tokens=10**9, output_tokens=10**9))  # no raise

    def test_max_turns_enforced(self):
        b = Budget(max_turns=3)
        u = Usage()
        u.turns = 2
        b.check(u)  # ok
        u.turns = 3
        with self.assertRaises(BudgetExceeded):
            b.check(u)

    def test_max_usd_enforced(self):
        # 1M input on opus-4-7 = $5 → tip over a $4 cap
        b = Budget(max_usd=4.0, model="claude-opus-4-7")
        u = Usage(input_tokens=1_000_000)
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check(u)
        self.assertIn("cap", ctx.exception.reason)

    def test_remaining_usd(self):
        b = Budget(max_usd=10.0, model="claude-opus-4-7")
        u = Usage(input_tokens=1_000_000)  # $5
        self.assertAlmostEqual(b.remaining_usd(u), 5.0)
        # exhausted
        u2 = Usage(input_tokens=2_000_000)  # $10
        self.assertEqual(b.remaining_usd(u2), 0.0)

    def test_remaining_turns(self):
        b = Budget(max_turns=5)
        u = Usage()
        u.turns = 2
        self.assertEqual(b.remaining_turns(u), 3)


if __name__ == "__main__":
    unittest.main()
