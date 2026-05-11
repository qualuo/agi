from learner.filter import filter_traces, eval_passing, min_quality
from learner.goals import Addition, Example, Goal
from learner.skills import Skill, SkillLibrary
from learner.traces import Trace, TraceLogger

# Critic depends on torch, which is an optional dep (install with [learner]).
# Skip it cleanly so the rest of the package is importable on CPU-only boxes
# and in CI without GPU-stack installs.
try:
    from learner.critic import Critic, CriticConfig  # noqa: F401
except ImportError:  # torch not installed
    Critic = None  # type: ignore
    CriticConfig = None  # type: ignore

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
    "Skill",
    "SkillLibrary",
]
