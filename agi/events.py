"""Event types emitted by a Run.

A coordination engine consumes these events to follow what the agent is doing
in real time: thinking deltas, tool calls and their results, partial text,
costs, critic scores, sub-runs, and terminal status.

Events are plain dataclasses with `.to_dict()` so they serialize trivially
over HTTP / SSE / message bus, and round-trip with `Event.from_dict(...)`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Event:
    type: str
    run_id: str
    ts: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        return cls(
            type=d["type"],
            run_id=d["run_id"],
            ts=float(d.get("ts", time.time())),
            data=dict(d.get("data", {})),
        )


# Canonical event types. Keep the wire format stable; downstream code keys on these.
RUN_STARTED = "run_started"
THINKING = "thinking"
TEXT_DELTA = "text_delta"
TEXT = "text"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
SUBRUN_STARTED = "subrun_started"
SUBRUN_COMPLETED = "subrun_completed"
SKILLS_LOADED = "skills_loaded"
REFLECTION = "reflection"
CRITIC_SCORE = "critic_score"
USAGE = "usage"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"


def make(type_: str, run_id: str, **data: Any) -> Event:
    return Event(type=type_, run_id=run_id, data=data)
