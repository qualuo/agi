"""Token usage and cost accounting.

Pricing is per-1M-tokens, sourced from the Claude API skill (current as of
the skill's cache date — 2026-04-29). Cache writes cost 1.25x base for the
default 5-minute TTL; cache reads cost ~0.1x. Update the table when
Anthropic updates their pricing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# (input $/1M, output $/1M) per model.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL (default)
CACHE_READ_MULTIPLIER = 0.10


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    turns: int = 0

    def add(self, response_usage) -> None:
        self.input_tokens += getattr(response_usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(response_usage, "output_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(response_usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(response_usage, "cache_read_input_tokens", 0) or 0
        self.turns += 1

    def cost_usd(self, model: str) -> float:
        if model not in PRICING:
            return 0.0
        in_rate, out_rate = PRICING[model]
        return (
            self.input_tokens * in_rate
            + self.cache_creation_input_tokens * in_rate * CACHE_WRITE_MULTIPLIER
            + self.cache_read_input_tokens * in_rate * CACHE_READ_MULTIPLIER
            + self.output_tokens * out_rate
        ) / 1_000_000

    def format(self, model: str) -> str:
        return (
            f"{self.input_tokens:,} in / {self.output_tokens:,} out / "
            f"{self.cache_creation_input_tokens:,} cache_w / "
            f"{self.cache_read_input_tokens:,} cache_r "
            f"— ${self.cost_usd(model):.4f}"
        )
