"""Event bus for the runtime.

The Agent emits lifecycle events to an optional EventBus. A coordination
engine subscribes (in-process or via the HTTP/SSE server) to observe what
the agent is doing, stream partial output, enforce policy, and trigger
side-effects. Without a bus attached the agent stays silent — events are
purely opt-in and have no effect on chat semantics.

Event types are stable strings; payloads are plain JSON-serializable dicts.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

EventHandler = Callable[["Event"], None]


@dataclass
class Event:
    seq: int
    ts: float
    session_id: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "session_id": self.session_id,
            "type": self.type,
            "data": self.data,
        }


# Stable event type names. A coordinator can switch on these.
SESSION_CREATED = "session.created"
SESSION_CLOSED = "session.closed"
TURN_STARTED = "turn.started"
TURN_COMPLETED = "turn.completed"
TURN_ERRORED = "turn.errored"
TEXT_DELTA = "text.delta"
THINKING_DELTA = "thinking.delta"
TOOL_INVOKED = "tool.invoked"
TOOL_COMPLETED = "tool.completed"
TOOL_ERRORED = "tool.errored"
BUDGET_EXCEEDED = "budget.exceeded"
CRITIC_SCORED = "critic.scored"


class EventBus:
    """Thread-safe pub/sub. Buffers a tail of events for late subscribers
    and supports synchronous handlers."""

    def __init__(self, *, buffer_size: int = 1024) -> None:
        self._lock = threading.Lock()
        self._handlers: list[EventHandler] = []
        self._buffer: list[Event] = []
        self._buffer_size = buffer_size
        self._seq = 0
        self.session_id = ""  # set by Session

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        with self._lock:
            self._handlers.append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return unsubscribe

    def emit(self, type: str, data: dict[str, Any] | None = None) -> Event:
        with self._lock:
            self._seq += 1
            event = Event(seq=self._seq, ts=time.time(), session_id=self.session_id, type=type, data=data or {})
            self._buffer.append(event)
            if len(self._buffer) > self._buffer_size:
                self._buffer = self._buffer[-self._buffer_size :]
            handlers = list(self._handlers)
        for h in handlers:
            try:
                h(event)
            except Exception:
                # A bad subscriber should not break the agent loop.
                pass
        return event

    def replay(self, since_seq: int = 0) -> list[Event]:
        with self._lock:
            return [e for e in self._buffer if e.seq > since_seq]


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]
