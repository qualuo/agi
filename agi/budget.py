"""Per-task budget tracking and enforcement.

A coordination engine submitting tasks to the runtime needs to bound cost
and latency per task. The Budget object is consulted between agent turns
(after each `_stream_one`) — if the limit is exceeded, the runtime emits
`task.budget_exceeded` and the agent stops looping.

Three independent ceilings:
- USD: cumulative dollars at current pricing
- tokens: cumulative input + output tokens
- wall_seconds: real time since task start

Any one tripping ends the task. None set = no limit on that dimension.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agi.costs import Usage


@dataclass
class Budget:
    max_usd: float | None = None
    max_tokens: int | None = None
    max_wall_seconds: float | None = None
    _start_ts: float = field(default_factory=time.time)

    def reset_clock(self) -> None:
        self._start_ts = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start_ts

    def check(self, usage: Usage, model: str) -> str | None:
        """Return None if within budget, else a short reason string."""
        if self.max_wall_seconds is not None and self.elapsed() > self.max_wall_seconds:
            return f"wall_seconds > {self.max_wall_seconds}s"
        if self.max_tokens is not None:
            total = usage.input_tokens + usage.output_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
            if total > self.max_tokens:
                return f"tokens > {self.max_tokens}"
        if self.max_usd is not None and usage.cost_usd(model) > self.max_usd:
            return f"usd > ${self.max_usd:.4f}"
        return None
