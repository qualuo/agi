"""Event bus for the runtime.

Every meaningful state transition in a session emits a typed Event. A
coordination engine subscribes to this stream to observe progress, route
work, enforce budgets, or react to specific tool calls. The Agent itself
emits; the Runtime fans out to subscribers.

The bus is in-process and thread-safe. For multi-process coordination, an
adapter would forward events over a transport (HTTP/SSE, websocket, queue).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


# Event kinds. Strings instead of an enum so external coordinators can
# pattern-match without importing this module.
SESSION_CREATED = "session.created"
SESSION_ENDED = "session.ended"
CHAT_STARTED = "chat.started"
CHAT_COMPLETED = "chat.completed"
THINKING_DELTA = "thinking.delta"
TEXT_DELTA = "text.delta"
TOOL_CALLED = "tool.called"
TOOL_RESULT = "tool.result"
USAGE_UPDATED = "usage.updated"
SKILL_LOADED = "skill.loaded"
SUBAGENT_STARTED = "subagent.started"
SUBAGENT_COMPLETED = "subagent.completed"
TOOL_SYNTHESIZED = "tool.synthesized"
CRITIC_SCORED = "critic.scored"
ERROR = "error"


@dataclass
class Event:
    kind: str
    session_id: str | None = None
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Subscriber = Callable[[Event], None]


class EventBus:
    """Thread-safe pub-sub. Subscribers can filter by session id and/or kind.

    Subscribers run inline on `publish()`; if a subscriber raises, the
    exception is swallowed (with the option to log) so a buggy listener
    can't poison the bus. A failing listener should not block the agent.
    """

    def __init__(self, history_limit: int = 1000) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[tuple[int, Subscriber, str | None, str | None]] = []
        self._next_id = 0
        self._history: list[Event] = []
        self._history_limit = history_limit

    def publish(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]
            subs = list(self._subscribers)
        for _, cb, sid_filter, kind_filter in subs:
            if sid_filter is not None and event.session_id != sid_filter:
                continue
            if kind_filter is not None and event.kind != kind_filter:
                continue
            try:
                cb(event)
            except Exception:
                # Buggy subscriber must not break the agent. Production
                # deployments should wrap subscribers with their own logger.
                pass

    def subscribe(
        self,
        callback: Subscriber,
        *,
        session_id: str | None = None,
        kind: str | None = None,
    ) -> int:
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers.append((sub_id, callback, session_id, kind))
            return sub_id

    def unsubscribe(self, sub_id: int) -> bool:
        with self._lock:
            before = len(self._subscribers)
            self._subscribers = [s for s in self._subscribers if s[0] != sub_id]
            return len(self._subscribers) < before

    def history(
        self,
        *,
        session_id: str | None = None,
        kind: str | None = None,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        with self._lock:
            events = list(self._history)
        out = []
        for e in events:
            if session_id is not None and e.session_id != session_id:
                continue
            if kind is not None and e.kind != kind:
                continue
            if since_ts is not None and e.ts <= since_ts:
                continue
            out.append(e)
        if limit is not None:
            out = out[-limit:]
        return out

    def clear_history(self) -> None:
        with self._lock:
            self._history = []
