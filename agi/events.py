"""Pub-sub event bus.

A `Runtime` and `Session` publish typed events as the agent runs:
turn_started, thinking, tool_call, tool_result, text, turn_finished,
budget_warning, error. Subscribers — a coordination engine, an SSE HTTP
endpoint, a test harness — register a callback and consume.

Events are plain dicts so they serialize trivially to JSON. Bus is
synchronous-publish, multi-subscriber, no buffering. Subscribers that
raise are unsubscribed (one bad listener can't take the whole bus
down).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Iterable

EventCallback = Callable[[dict], None]


# Canonical event types. Listed here so consumers can rely on a stable set.
EVENT_TYPES = (
    "session_opened",
    "session_closed",
    "turn_started",
    "turn_finished",
    "thinking",
    "text",
    "tool_call",
    "tool_result",
    "server_tool",
    "budget_warning",
    "budget_exceeded",
    "critic_score",
    "skill_loaded",
    "delegate_spawn",
    "delegate_return",
    "error",
)


def make_event(type_: str, **fields: Any) -> dict:
    return {"type": type_, "ts": time.time(), **fields}


class EventBus:
    def __init__(self, history: int = 0) -> None:
        """history > 0 keeps a ring buffer of recent events (useful for
        late subscribers — e.g. an HTTP client that connects mid-task and
        wants to catch up)."""
        self._subs: list[EventCallback] = []
        self._lock = threading.Lock()
        self._history: deque[dict] = deque(maxlen=history) if history else deque(maxlen=0)
        self._history_enabled = history > 0

    def subscribe(self, cb: EventCallback) -> Callable[[], None]:
        with self._lock:
            self._subs.append(cb)

        def unsubscribe() -> None:
            with self._lock:
                if cb in self._subs:
                    self._subs.remove(cb)

        return unsubscribe

    def publish(self, event: dict) -> None:
        if self._history_enabled:
            self._history.append(event)
        # Snapshot so callbacks can mutate the subscriber list safely.
        with self._lock:
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                # bad subscriber — drop it, keep the bus alive
                with self._lock:
                    if cb in self._subs:
                        self._subs.remove(cb)

    def emit(self, type_: str, **fields: Any) -> dict:
        evt = make_event(type_, **fields)
        self.publish(evt)
        return evt

    def history(self) -> list[dict]:
        return list(self._history)

    def __len__(self) -> int:
        with self._lock:
            return len(self._subs)


def collect(bus: EventBus) -> tuple[list[dict], Callable[[], None]]:
    """Convenience: subscribe a list-collector to the bus.

    Returns (events_list, unsubscribe). Useful for tests and for
    debugging coordination flows.
    """
    events: list[dict] = []
    unsub = bus.subscribe(events.append)
    return events, unsub


def filter_types(events: Iterable[dict], *types: str) -> list[dict]:
    return [e for e in events if e.get("type") in types]
