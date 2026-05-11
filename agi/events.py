"""Structured events emitted by an Agent during execution.

Decouples the agent's inner loop from how its activity is observed. A
single `on_event` callback on the Agent is invoked for every event; the
default REPL renders some, the Runtime forwards them to session
subscribers, and tests assert on the sequence.

Events are dataclasses with `kind` and `ts` plus kind-specific fields.
They serialize cleanly to JSON for SSE / wire transport.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Event:
    kind: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class SessionStarted(Event):
    kind: str = "session_started"
    session_id: str = ""
    goal: str = ""
    model: str = ""


@dataclass
class TurnStarted(Event):
    kind: str = "turn_started"
    iteration: int = 0


@dataclass
class ThinkingDelta(Event):
    kind: str = "thinking_delta"
    text: str = ""


@dataclass
class TextDelta(Event):
    kind: str = "text_delta"
    text: str = ""


@dataclass
class ToolUse(Event):
    kind: str = "tool_use"
    name: str = ""
    server_side: bool = False
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult(Event):
    kind: str = "tool_result"
    name: str = ""
    output: str = ""
    is_error: bool = False
    elapsed_ms: int = 0


@dataclass
class TurnFinished(Event):
    kind: str = "turn_finished"
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class BudgetExceeded(Event):
    kind: str = "budget_exceeded"
    reason: str = ""  # "cost" | "tokens" | "iterations" | "wall_time"
    limit: float = 0.0
    actual: float = 0.0


@dataclass
class SessionFinished(Event):
    kind: str = "session_finished"
    session_id: str = ""
    status: str = ""  # "ok" | "budget_exceeded" | "error" | "cancelled"
    final_text: str = ""
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turns: int = 0
    error: str = ""
