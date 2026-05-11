"""Structured runtime events.

This is the wire format a coordination engine subscribes to. Every meaningful
state change inside the runtime emits one of these — task lifecycle, plan
formation, tool calls, subagent spawns, token usage, critic gating.

Two design constraints:

1. **Stable schema.** A coordination engine integrates against these events;
   they can't quietly change shape. Each event is a frozen dataclass with a
   `kind` literal and JSON-serializable fields.
2. **No back-pressure on the agent.** Subscribers run async via a thread-safe
   queue; a slow subscriber drops events rather than stalling the agent loop.

Subscribers can be in-process callbacks (Python) or remote (the JSON-RPC
server fans events out over server-sent events / websocket frames).
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal


EventKind = Literal[
    "task.submitted",
    "task.started",
    "task.plan",
    "task.tool_call",
    "task.tool_result",
    "task.text",
    "task.thinking",
    "task.subagent_spawned",
    "task.subagent_finished",
    "task.skill_loaded",
    "task.tool_synthesized",
    "task.reflection",
    "task.critic_score",
    "task.usage",
    "task.budget_exceeded",
    "task.cancelled",
    "task.failed",
    "task.completed",
]


@dataclass
class Event:
    """One runtime event. JSON-serializable via asdict()."""

    kind: EventKind
    task_id: str
    ts: float = field(default_factory=time.time)
    seq: int = 0
    parent_task_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Subscriber = Callable[[Event], None]


class EventBus:
    """In-process pub/sub for runtime events.

    Subscribers are invoked from a single background dispatch thread, so they
    don't block the agent loop. If a subscriber raises, the bus catches and
    keeps dispatching to the others — one bad listener shouldn't break the
    others.

    For remote consumers, the JSON-RPC server attaches its own subscriber that
    pushes events onto an HTTP stream.
    """

    def __init__(self, max_queue: int = 10000) -> None:
        self._subs: list[Subscriber] = []
        self._lock = threading.Lock()
        self._q: queue.Queue[Event | None] = queue.Queue(maxsize=max_queue)
        self._seq = 0
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name="agi-eventbus", daemon=True)
        self._thread.start()

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe callable."""
        with self._lock:
            self._subs.append(fn)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._subs.remove(fn)
                except ValueError:
                    pass

        return _unsub

    def emit(
        self,
        kind: EventKind,
        task_id: str,
        *,
        parent_task_id: str | None = None,
        **data: Any,
    ) -> Event:
        with self._lock:
            self._seq += 1
            seq = self._seq
        ev = Event(kind=kind, task_id=task_id, seq=seq, parent_task_id=parent_task_id, data=data)
        try:
            self._q.put_nowait(ev)
        except queue.Full:
            # Drop on overflow rather than blocking the agent. The seq gap
            # tells subscribers they missed something.
            pass
        return ev

    def close(self, timeout: float = 1.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while True:
            ev = self._q.get()
            if ev is None:
                return
            with self._lock:
                subs = list(self._subs)
            for fn in subs:
                try:
                    fn(ev)
                except Exception:
                    # A subscriber crash must not take down the bus.
                    pass


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]
