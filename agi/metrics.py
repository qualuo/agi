"""Runtime telemetry.

The coordination engine wants to query: how much have we spent overall?
how many turns ran? what's the failure rate by role? which tools are hot?
Without this a coordinator can't make routing decisions or enforce
fleet-wide budgets.

`RuntimeMetrics` subscribes to a Session's EventBus and aggregates as
events arrive. It is process-local; for cross-process aggregation a
coordinator polls each runtime's /metrics endpoint and rolls up itself.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from agi.events import (
    BUDGET_EXCEEDED,
    CRITIC_SCORED,
    Event,
    EventBus,
    SESSION_CLOSED,
    SESSION_CREATED,
    TOOL_COMPLETED,
    TOOL_ERRORED,
    TOOL_INVOKED,
    TURN_COMPLETED,
    TURN_ERRORED,
    TURN_STARTED,
)


@dataclass
class RoleMetrics:
    sessions_created: int = 0
    sessions_active: int = 0
    turns_completed: int = 0
    turns_errored: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class RuntimeMetrics:
    started_at: float = field(default_factory=time.time)

    sessions_created: int = 0
    sessions_active: int = 0
    sessions_closed: int = 0

    turns_started: int = 0
    turns_completed: int = 0
    turns_errored: int = 0

    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    budgets_exceeded: int = 0

    tool_invocations: Counter = field(default_factory=Counter)
    tool_errors: Counter = field(default_factory=Counter)
    role_metrics: dict[str, RoleMetrics] = field(default_factory=dict)

    critic_scores_total: float = 0.0
    critic_scores_n: int = 0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def attach(self, bus: EventBus, *, role: Optional[str] = None) -> None:
        """Subscribe this metrics aggregator to a Session's bus.

        `role` is captured in a closure so events arriving on this bus are
        attributed to the right role bucket (events themselves don't carry
        role today)."""

        def handler(ev: Event) -> None:
            self._on_event(ev, role)

        bus.subscribe(handler)

    def _role(self, role: Optional[str]) -> RoleMetrics:
        key = role or "_default"
        rm = self.role_metrics.get(key)
        if rm is None:
            rm = RoleMetrics()
            self.role_metrics[key] = rm
        return rm

    def _on_event(self, ev: Event, role: Optional[str]) -> None:
        with self._lock:
            rm = self._role(role)
            t = ev.type
            d = ev.data or {}
            if t == SESSION_CREATED:
                self.sessions_created += 1
                self.sessions_active += 1
                rm.sessions_created += 1
                rm.sessions_active += 1
            elif t == SESSION_CLOSED:
                self.sessions_active = max(0, self.sessions_active - 1)
                self.sessions_closed += 1
                rm.sessions_active = max(0, rm.sessions_active - 1)
            elif t == TURN_STARTED:
                self.turns_started += 1
            elif t == TURN_COMPLETED:
                self.turns_completed += 1
                rm.turns_completed += 1
                self.cost_usd += float(d.get("cost_usd") or 0)
                self.input_tokens += int(d.get("input_tokens") or 0)
                self.output_tokens += int(d.get("output_tokens") or 0)
                self.cache_read_input_tokens += int(d.get("cache_read_input_tokens") or 0)
                self.cache_creation_input_tokens += int(d.get("cache_creation_input_tokens") or 0)
                rm.cost_usd += float(d.get("cost_usd") or 0)
                rm.input_tokens += int(d.get("input_tokens") or 0)
                rm.output_tokens += int(d.get("output_tokens") or 0)
            elif t == TURN_ERRORED:
                self.turns_errored += 1
                rm.turns_errored += 1
            elif t == TOOL_INVOKED:
                name = str(d.get("name") or "unknown")
                self.tool_invocations[name] += 1
            elif t == TOOL_ERRORED:
                name = str(d.get("name") or "unknown")
                self.tool_errors[name] += 1
            elif t == BUDGET_EXCEEDED:
                self.budgets_exceeded += 1
            elif t == CRITIC_SCORED:
                score = d.get("score")
                if isinstance(score, (int, float)):
                    self.critic_scores_total += float(score)
                    self.critic_scores_n += 1

    def to_dict(self) -> dict:
        with self._lock:
            avg_critic = (
                self.critic_scores_total / self.critic_scores_n
                if self.critic_scores_n
                else None
            )
            return {
                "started_at": self.started_at,
                "uptime_seconds": time.time() - self.started_at,
                "sessions": {
                    "created": self.sessions_created,
                    "active": self.sessions_active,
                    "closed": self.sessions_closed,
                },
                "turns": {
                    "started": self.turns_started,
                    "completed": self.turns_completed,
                    "errored": self.turns_errored,
                    "success_rate": (
                        self.turns_completed / max(self.turns_started, 1)
                    ),
                },
                "cost": {
                    "usd": round(self.cost_usd, 6),
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "cache_read_input_tokens": self.cache_read_input_tokens,
                    "cache_creation_input_tokens": self.cache_creation_input_tokens,
                },
                "budgets_exceeded": self.budgets_exceeded,
                "tool_invocations": dict(self.tool_invocations),
                "tool_errors": dict(self.tool_errors),
                "critic": {
                    "n": self.critic_scores_n,
                    "average_score": avg_critic,
                },
                "by_role": {
                    role: {
                        "sessions_created": rm.sessions_created,
                        "sessions_active": rm.sessions_active,
                        "turns_completed": rm.turns_completed,
                        "turns_errored": rm.turns_errored,
                        "cost_usd": round(rm.cost_usd, 6),
                        "input_tokens": rm.input_tokens,
                        "output_tokens": rm.output_tokens,
                    }
                    for role, rm in self.role_metrics.items()
                },
            }
