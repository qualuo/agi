"""Budget enforcement.

A Budget is a soft contract between a coordination engine and the runtime: it
caps tokens, dollars, and tool-loop iterations before a session can run away.
The runtime checks the budget *before* each new turn and after the SDK reports
usage; if either limit is exceeded the next call raises `BudgetError`.

Budgets are advisory in the sense that an already-streaming response is not
truncated mid-stream — that would corrupt tool-result accounting. They gate
the *next* turn, which is the cheap and correct place to enforce.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from agi.costs import Usage


class BudgetError(RuntimeError):
    """Raised when a session attempts work that would exceed its budget."""


@dataclass
class Budget:
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_usd: float | None = None
    max_turns: int | None = None
    max_jobs: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Budget":
        if not d:
            return cls()
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def check(self, usage: Usage, model: str, jobs_run: int) -> None:
        """Raise BudgetError if usage already exceeds the cap. Called before
        each new turn — the over-by-one-turn slop is intentional and small."""
        if self.max_input_tokens is not None and usage.input_tokens >= self.max_input_tokens:
            raise BudgetError(f"input tokens {usage.input_tokens} >= cap {self.max_input_tokens}")
        if self.max_output_tokens is not None and usage.output_tokens >= self.max_output_tokens:
            raise BudgetError(f"output tokens {usage.output_tokens} >= cap {self.max_output_tokens}")
        if self.max_total_tokens is not None:
            total = usage.input_tokens + usage.output_tokens
            if total >= self.max_total_tokens:
                raise BudgetError(f"total tokens {total} >= cap {self.max_total_tokens}")
        if self.max_usd is not None and usage.cost_usd(model) >= self.max_usd:
            raise BudgetError(f"cost ${usage.cost_usd(model):.4f} >= cap ${self.max_usd:.4f}")
        if self.max_turns is not None and usage.turns >= self.max_turns:
            raise BudgetError(f"turns {usage.turns} >= cap {self.max_turns}")
        if self.max_jobs is not None and jobs_run >= self.max_jobs:
            raise BudgetError(f"jobs {jobs_run} >= cap {self.max_jobs}")
