"""Structured events emitted by an Agent during a turn.

The agent has historically printed directly to stdout. That's fine for a CLI
but useless to anything trying to drive the agent programmatically: a
coordination engine, a UI, another agent. Events give us a stable, typed
surface for everything the agent does within a turn — text deltas, thinking,
tool calls, tool results, usage updates, end-of-turn — which downstream
consumers can stream, log, render, meter, or gate on.

Events are plain dataclasses. They serialize to JSON via `asdict(ev) | {"type": ev.type}`
and back via `Event.from_dict`. Keeping them dataclasses (not pydantic) means
zero new dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Event:
    """Base class. Subclasses set `type` as a class var via ClassVar pattern."""
    session_id: str
    seq: int = 0  # monotonic per-session sequence number, assigned by the runtime

    @property
    def type(self) -> str:
        return self.__class__.__name__

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type
        return d


@dataclass
class TurnStart(Event):
    """Marks the start of an agent turn (a user message coming in)."""
    user_input: str = ""


@dataclass
class TextDelta(Event):
    """A chunk of streamed text from the assistant."""
    text: str = ""


@dataclass
class ThinkingDelta(Event):
    """A chunk of streamed extended-thinking summary."""
    text: str = ""


@dataclass
class ToolUseStart(Event):
    """The model has requested a client-side tool call."""
    tool_use_id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolUseResult(Event):
    """The dispatched tool returned (or errored)."""
    tool_use_id: str = ""
    name: str = ""
    output: str = ""
    is_error: bool = False


@dataclass
class ServerToolUse(Event):
    """The model used a server-side tool (web_search / web_fetch)."""
    name: str = ""


@dataclass
class UsageDelta(Event):
    """Token usage for the turn-so-far, emitted after each LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class TurnEnd(Event):
    """End of turn. Carries the final text and any critic verdict."""
    final_text: str = ""
    stop_reason: str = "end_turn"
    critic_score: float | None = None
    cost_usd: float = 0.0


@dataclass
class ErrorEvent(Event):
    """An exception escaped the agent loop."""
    message: str = ""
    exc_type: str = ""


EVENT_TYPES: dict[str, type[Event]] = {
    cls.__name__: cls
    for cls in (
        TurnStart,
        TextDelta,
        ThinkingDelta,
        ToolUseStart,
        ToolUseResult,
        ServerToolUse,
        UsageDelta,
        TurnEnd,
        ErrorEvent,
    )
}


def from_dict(d: dict[str, Any]) -> Event:
    """Inverse of Event.to_dict — useful for clients consuming the SSE stream."""
    ev_type = d.get("type")
    cls = EVENT_TYPES.get(ev_type or "")
    if cls is None:
        raise ValueError(f"unknown event type: {ev_type!r}")
    payload = {k: v for k, v in d.items() if k != "type"}
    return cls(**payload)
