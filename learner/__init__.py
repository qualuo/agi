"""Learner package.

Lazy imports: the critic + LoRA pieces depend on torch (optional). We only
import them on first access so the always-available pieces (`Trace`,
`TraceLogger`, filters, goals) work even when torch isn't installed.
"""
from __future__ import annotations

from learner.filter import filter_traces, eval_passing, min_quality
from learner.goals import Addition, Example, Goal
from learner.traces import Trace, TraceLogger

__all__ = [
    "Trace",
    "TraceLogger",
    "filter_traces",
    "eval_passing",
    "min_quality",
    "Goal",
    "Example",
    "Addition",
    "Critic",
    "CriticConfig",
]


def __getattr__(name: str):
    """Lazy import for torch-dependent symbols."""
    if name in ("Critic", "CriticConfig"):
        from learner.critic import Critic, CriticConfig  # noqa: F401
        return {"Critic": Critic, "CriticConfig": CriticConfig}[name]
    raise AttributeError(f"module 'learner' has no attribute {name!r}")
