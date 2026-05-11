"""Budget enforcement for an agent session.

A Budget caps any combination of:
  - total cost (USD)
  - input + output tokens
  - tool-use iterations
  - wall-clock seconds

The session checks `Budget.check(...)` between turns. The first exhausted
limit short-circuits the session with a `BudgetExceeded` event and a
`budget_exceeded` final status. This is the substrate for unit economics:
a coordinator can submit "spend at most $0.50 on this subgoal" and get a
hard guarantee, not a hope.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Budget:
    max_cost_usd: Optional[float] = None
    max_tokens: Optional[int] = None         # input + output combined
    max_iterations: Optional[int] = None
    max_wall_seconds: Optional[float] = None

    started_at: float = field(default_factory=time.time)

    def reset_clock(self) -> None:
        self.started_at = time.time()

    def check(
        self,
        *,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        iterations: int,
    ) -> Optional[tuple[str, float, float]]:
        """Return (reason, limit, actual) if a limit is exceeded, else None."""
        if self.max_cost_usd is not None and cost_usd >= self.max_cost_usd:
            return ("cost", self.max_cost_usd, cost_usd)
        if self.max_tokens is not None:
            total = input_tokens + output_tokens
            if total >= self.max_tokens:
                return ("tokens", float(self.max_tokens), float(total))
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return ("iterations", float(self.max_iterations), float(iterations))
        if self.max_wall_seconds is not None:
            elapsed = time.time() - self.started_at
            if elapsed >= self.max_wall_seconds:
                return ("wall_time", self.max_wall_seconds, elapsed)
        return None
