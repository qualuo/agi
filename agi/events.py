"""Structured agent events.

The agent loop used to print directly to stdout. A coordination engine
calling this runtime wants to *react* to thinking, tool calls, and
streamed text — not parse a terminal log. So the agent emits typed
`Event`s through an `EventBus`. Subscribers translate to whatever sink
they need: stdout, SSE, a queue, a websocket, a trace file.

The bus is intentionally synchronous and unbounded. Subscribers that
need backpressure or async should wrap their handler with a queue.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


EventHandler = Callable[["Event"], None]


@dataclass
class Event:
    """A single structured event emitted during a run.

    `kind` is one of:
      run.started, run.finished, turn.started, turn.finished,
      thinking.delta, text.delta, tool.requested, tool.result,
      server_tool.requested, usage.updated, critic.scored, error,
      cancelled.

    `data` carries kind-specific fields. Keep it JSON-serializable so
    consumers can forward unmodified over HTTP/SSE.
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    session_id: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    """Fan-out publisher. Synchronous; subscribers fire in registration order."""

    def __init__(self) -> None:
        self._subs: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        self._subs.append(handler)

        def unsubscribe() -> None:
            try:
                self._subs.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def publish(self, event: Event) -> None:
        for sub in list(self._subs):
            try:
                sub(event)
            except Exception:
                # Subscribers must not break the agent loop. Drop and move on.
                # In production we'd log; v1 stays silent to avoid feedback loops.
                pass

    def emit(self, kind: str, **data: Any) -> Event:
        evt = Event(kind=kind, data=data)
        self.publish(evt)
        return evt


def stdout_printer(event: Event) -> None:
    """Drop-in subscriber that mimics the old print-to-stdout behavior."""
    k = event.kind
    d = event.data
    if k == "thinking.delta":
        print(d.get("text", ""), end="", flush=True)
    elif k == "text.delta":
        print(d.get("text", ""), end="", flush=True)
    elif k == "thinking.started":
        print("\n[thinking] ", end="", flush=True)
    elif k == "text.started":
        print()
    elif k == "tool.requested":
        print(f"\n[tool: {d.get('name')}]", end="", flush=True)
    elif k == "server_tool.requested":
        print(f"\n[server: {d.get('name')}]", end="", flush=True)
    elif k == "turn.finished":
        usage = d.get("usage_formatted")
        if usage:
            print(f"\n[{usage}]", flush=True)
    elif k == "error":
        print(f"\n[error] {d.get('message', '')}", flush=True)
    elif k == "critic.scored":
        score = d.get("score")
        thr = d.get("threshold")
        if score is not None and thr is not None and score < thr:
            print(
                f"\n[critic confidence: {score:.2f} (< {thr}) — response may be unreliable]",
                flush=True,
            )
