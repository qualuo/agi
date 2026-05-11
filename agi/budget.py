"""Budgets: token / dollar / time / iteration ceilings.

A coordination engine that drives the runtime needs to bound any single task
so a misbehaving agent can't burn the bank or stall the queue. Budgets are
checked inside the agent loop between turns; when one is exceeded the loop
stops and the partial result is returned with `status: 'over_budget'`.

Budgets compose: the strictest active limit fires first. None disables a limit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from agi.costs import Usage


@dataclass
class Budget:
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_usd: float | None = None
    max_seconds: float | None = None
    max_iterations: int | None = None

    def merged_with(self, other: "Budget | None") -> "Budget":
        """Return a Budget tighter than either (None loses)."""
        if other is None:
            return self
        def tighter(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)
        return Budget(
            max_input_tokens=tighter(self.max_input_tokens, other.max_input_tokens),
            max_output_tokens=tighter(self.max_output_tokens, other.max_output_tokens),
            max_usd=tighter(self.max_usd, other.max_usd),
            max_seconds=tighter(self.max_seconds, other.max_seconds),
            max_iterations=tighter(self.max_iterations, other.max_iterations),
        )

    def check(self, *, usage: Usage, model: str, started_at: float, iterations: int) -> str | None:
        """Return None if within budget; otherwise a short reason string."""
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return f"iterations >= {self.max_iterations}"
        if self.max_seconds is not None and (time.time() - started_at) >= self.max_seconds:
            return f"elapsed >= {self.max_seconds}s"
        if self.max_input_tokens is not None and usage.input_tokens >= self.max_input_tokens:
            return f"input_tokens >= {self.max_input_tokens}"
        if self.max_output_tokens is not None and usage.output_tokens >= self.max_output_tokens:
            return f"output_tokens >= {self.max_output_tokens}"
        if self.max_usd is not None and usage.cost_usd(model) >= self.max_usd:
            return f"cost_usd >= ${self.max_usd}"
        return None

    def to_dict(self) -> dict:
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_usd": self.max_usd,
            "max_seconds": self.max_seconds,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Budget | None":
        if not d:
            return None
        return cls(**{k: d.get(k) for k in (
            "max_input_tokens",
            "max_output_tokens",
            "max_usd",
            "max_seconds",
            "max_iterations",
        )})
