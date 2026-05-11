"""Structured events emitted by AgentRuntime.

A coordination engine drives a runtime by sending prompts and consuming an
event stream. Events are plain dataclasses (JSON-serializable via `to_dict`)
so they can cross process or language boundaries.

Design notes:
- Every event carries `session_id` and a monotonic `seq` so consumers can
  reorder/dedup.
- `kind` is a string discriminator for wire formats that don't preserve types.
- Token deltas (`TextDelta`, `ThinkingDelta`) are fine-grained; consumers can
  buffer them or only act on `TurnCompleted`.
- `ToolUseRequested` lets a coordinator intercept before execution (approval,
  policy enforcement). `ToolResult` reports the outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class Event:
    session_id: str
    seq: int
    kind: str = field(init=False, default="event")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind
        return d


@dataclass
class SessionStarted(Event):
    model: str = ""
    kind: str = field(init=False, default="session_started")


@dataclass
class TextDelta(Event):
    text: str = ""
    kind: str = field(init=False, default="text_delta")


@dataclass
class ThinkingDelta(Event):
    text: str = ""
    kind: str = field(init=False, default="thinking_delta")


@dataclass
class ToolUseRequested(Event):
    tool_id: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    kind: str = field(init=False, default="tool_use_requested")


@dataclass
class ToolResult(Event):
    tool_id: str = ""
    tool_name: str = ""
    output: str = ""
    is_error: bool = False
    intercepted: bool = False  # True if the coordinator replaced the result
    kind: str = field(init=False, default="tool_result")


@dataclass
class TurnCompleted(Event):
    text: str = ""
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    critic_score: float | None = None
    kind: str = field(init=False, default="turn_completed")


@dataclass
class BudgetExceeded(Event):
    limit_kind: str = ""  # "cost_usd" or "max_iterations"
    limit_value: float = 0.0
    actual: float = 0.0
    kind: str = field(init=False, default="budget_exceeded")


@dataclass
class RuntimeError_(Event):
    """Surfaces an exception during the run without aborting the consumer's
    event loop. Named with trailing underscore to avoid shadowing builtin."""
    error_type: str = ""
    message: str = ""
    kind: str = field(init=False, default="runtime_error")
