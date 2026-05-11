"""Structured events emitted by a Run.

A coordination engine (or any external orchestrator) subscribes to these
events to follow a run's progress, attribute cost, and decide when to
intervene. Every event is JSON-serializable. Field names are stable.

Event flow for a typical run:

    run_started
      task_started        # agent receives the user prompt
      thinking_delta*     # streamed thinking summary
      text_delta*         # streamed assistant text
      tool_call           # client-side or server-side tool invoked
      tool_result         # paired with the matching tool_call
      ...                 # may loop on (thinking, text, tool)
      turn_completed      # one model turn finished
      task_completed      # final text produced
    run_completed | run_failed | run_cancelled
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    """Base event.

    Subclasses set their own `type`; `seq` is filled in by the emitter so
    consumers can reassemble a stream and detect drops.
    """
    type: str
    run_id: str
    ts: float = field(default_factory=time.time)
    seq: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_started(run_id: str, prompt: str, model: str, budget: dict | None = None) -> Event:
    return Event(
        type="run_started",
        run_id=run_id,
        data={"prompt": prompt, "model": model, "budget": budget or {}},
    )


def task_started(run_id: str, prompt: str) -> Event:
    return Event(type="task_started", run_id=run_id, data={"prompt": prompt})


def thinking_delta(run_id: str, text: str) -> Event:
    return Event(type="thinking_delta", run_id=run_id, data={"text": text})


def text_delta(run_id: str, text: str) -> Event:
    return Event(type="text_delta", run_id=run_id, data={"text": text})


def tool_call(run_id: str, name: str, tool_use_id: str, input: dict, server_side: bool = False) -> Event:
    return Event(
        type="tool_call",
        run_id=run_id,
        data={
            "name": name,
            "tool_use_id": tool_use_id,
            "input": input,
            "server_side": server_side,
        },
    )


def tool_result(run_id: str, tool_use_id: str, content: str, is_error: bool = False) -> Event:
    return Event(
        type="tool_result",
        run_id=run_id,
        data={"tool_use_id": tool_use_id, "content": content, "is_error": is_error},
    )


def turn_completed(run_id: str, usage: dict, stop_reason: str | None) -> Event:
    return Event(
        type="turn_completed",
        run_id=run_id,
        data={"usage": usage, "stop_reason": stop_reason},
    )


def task_completed(run_id: str, text: str, critic_score: float | None) -> Event:
    return Event(
        type="task_completed",
        run_id=run_id,
        data={"text": text, "critic_score": critic_score},
    )


def run_completed(run_id: str, text: str, usage: dict, cost_usd: float) -> Event:
    return Event(
        type="run_completed",
        run_id=run_id,
        data={"text": text, "usage": usage, "cost_usd": cost_usd},
    )


def run_failed(run_id: str, error: str, usage: dict | None = None) -> Event:
    return Event(
        type="run_failed",
        run_id=run_id,
        data={"error": error, "usage": usage or {}},
    )


def run_cancelled(run_id: str, reason: str, usage: dict | None = None) -> Event:
    return Event(
        type="run_cancelled",
        run_id=run_id,
        data={"reason": reason, "usage": usage or {}},
    )


def budget_exceeded(run_id: str, kind: str, limit: float, actual: float) -> Event:
    return Event(
        type="budget_exceeded",
        run_id=run_id,
        data={"kind": kind, "limit": limit, "actual": actual},
    )


def child_run_started(run_id: str, child_run_id: str, role: str, prompt: str) -> Event:
    """A subagent run was spawned by this run."""
    return Event(
        type="child_run_started",
        run_id=run_id,
        data={"child_run_id": child_run_id, "role": role, "prompt": prompt},
    )


def child_run_completed(run_id: str, child_run_id: str, text: str, usage: dict, cost_usd: float) -> Event:
    return Event(
        type="child_run_completed",
        run_id=run_id,
        data={
            "child_run_id": child_run_id,
            "text": text,
            "usage": usage,
            "cost_usd": cost_usd,
        },
    )
