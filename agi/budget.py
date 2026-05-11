"""Per-session budget gate.

A `Budget` enforces hard ceilings on cost (USD) and turns. The Agent
checks `Budget.check()` before each model call; if the ceiling is
breached, `BudgetExceeded` is raised and the agent loop stops cleanly.

Budgets are first-class because:
- A coordination engine has to bound spend per task to keep unit
  economics legible.
- Runaway loops (model insists on calling tools forever) are the
  default-bad outcome of agent harnesses; a budget makes them safe.
"""
from __future__ import annotations

from dataclasses import dataclass

from agi.costs import Usage


class BudgetExceeded(Exception):
    """Raised when a Budget ceiling is hit. Carries the reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class Budget:
    max_usd: float | None = None        # cumulative USD ceiling, None = unbounded
    max_turns: int | None = None        # cumulative turn ceiling
    model: str = "claude-opus-4-7"      # which pricing table to charge against

    def check(self, usage: Usage) -> None:
        """Raise BudgetExceeded if any ceiling has been crossed."""
        if self.max_usd is not None:
            spent = usage.cost_usd(self.model)
            if spent >= self.max_usd:
                raise BudgetExceeded(
                    f"cost ${spent:.4f} >= cap ${self.max_usd:.4f}"
                )
        if self.max_turns is not None and usage.turns >= self.max_turns:
            raise BudgetExceeded(
                f"turns {usage.turns} >= cap {self.max_turns}"
            )

    def remaining_usd(self, usage: Usage) -> float | None:
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - usage.cost_usd(self.model))

    def remaining_turns(self, usage: Usage) -> int | None:
        if self.max_turns is None:
            return None
        return max(0, self.max_turns - usage.turns)
