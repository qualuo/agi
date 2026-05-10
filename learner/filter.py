"""Quality gates for training data.

The data-quality problem is THE problem in online learning. Garbage in →
the adapter learns to be wrong, durably. Every trace fed to training has
to pass a filter — and the filter is the most important knob in the
system.

Conservative defaults: require an explicit positive signal (eval pass,
user thumbs-up, or a quality score above threshold). Don't train on
ambiguous traces.
"""
from __future__ import annotations

from typing import Callable

from learner.traces import Trace


def eval_passing(trace: Trace) -> bool:
    """Trace from an eval task that was marked passed=True in metadata."""
    return trace.metadata.get("eval_passed") is True


def min_quality(threshold: float = 0.7) -> Callable[[Trace], bool]:
    """Trace has metadata.quality_score >= threshold."""
    def predicate(trace: Trace) -> bool:
        score = trace.metadata.get("quality_score")
        return isinstance(score, (int, float)) and score >= threshold
    return predicate


def user_thumbs_up(trace: Trace) -> bool:
    return trace.metadata.get("user_rating") == "up"


def filter_traces(traces: list[Trace], predicate: Callable[[Trace], bool]) -> list[Trace]:
    return [t for t in traces if predicate(t)]
