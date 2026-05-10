"""In-process pub/sub event bus.

The runtime emits structured events as tasks progress. A coordination engine
subscribes to receive them as they happen. Events are also persisted in a
bounded ring buffer per topic so a late subscriber can replay recent history.

Topics are dotted strings: `task.{id}`, `graph.{id}`, `runtime`. A subscriber
can match a prefix to listen to a whole sub-tree.

Thread-safe; subscribers receive events on the thread that called `publish`,
so handlers should be fast (typically: enqueue to a per-subscriber queue).
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable


@dataclass
class Event:
    id: str
    ts: float
    topic: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


Handler = Callable[[Event], None]


class _Subscription:
    __slots__ = ("prefix", "handler", "queue", "active")

    def __init__(self, prefix: str, handler: Handler | None = None) -> None:
        self.prefix = prefix
        self.handler = handler
        self.queue: queue.Queue[Event] | None = None if handler else queue.Queue()
        self.active = True


class EventBus:
    """Pub/sub with bounded per-topic history for replay."""

    def __init__(self, history_per_topic: int = 256) -> None:
        self._history_per_topic = history_per_topic
        self._lock = threading.RLock()
        self._subs: list[_Subscription] = []
        self._history: dict[str, list[Event]] = {}

    def publish(self, topic: str, kind: str, payload: dict[str, Any] | None = None) -> Event:
        event = Event(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            topic=topic,
            kind=kind,
            payload=payload or {},
        )
        with self._lock:
            hist = self._history.setdefault(topic, [])
            hist.append(event)
            if len(hist) > self._history_per_topic:
                del hist[: len(hist) - self._history_per_topic]
            targets = [s for s in self._subs if s.active and topic.startswith(s.prefix)]
        for sub in targets:
            try:
                if sub.handler is not None:
                    sub.handler(event)
                elif sub.queue is not None:
                    sub.queue.put_nowait(event)
            except Exception:
                # Handler failures must never crash publishing.
                pass
        return event

    def subscribe(self, prefix: str = "", handler: Handler | None = None) -> _Subscription:
        sub = _Subscription(prefix=prefix, handler=handler)
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: _Subscription) -> None:
        sub.active = False
        with self._lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                pass

    def history(self, prefix: str = "") -> list[Event]:
        with self._lock:
            out: list[Event] = []
            for topic, events in self._history.items():
                if topic.startswith(prefix):
                    out.extend(events)
        out.sort(key=lambda e: e.ts)
        return out

    def stream(
        self,
        prefix: str = "",
        *,
        include_history: bool = True,
        timeout: float | None = None,
    ) -> Iterable[Event]:
        """Yield events matching prefix. Optionally replay history first.

        Returns when timeout elapses with no new event, or when caller breaks.
        """
        if include_history:
            for e in self.history(prefix):
                yield e
        sub = self.subscribe(prefix=prefix)
        try:
            while True:
                assert sub.queue is not None
                try:
                    yield sub.queue.get(timeout=timeout)
                except queue.Empty:
                    return
        finally:
            self.unsubscribe(sub)
