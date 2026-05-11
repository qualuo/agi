"""Coordination engine — drives one or more Runtimes.

The agent harness is the *executor*. The Runtime is the *contract*. The
Coordinator is the *strategy*: which runtime to send work to, when to fan
out and pick the best, when to retry, how to enforce a global budget.

Keep this module thin and free of model-specific knowledge. The coordinator
talks to runtimes only through `agi.protocol` and `agi.runtime.Runtime`. A
runtime backed by Opus, by a small local LoRA-tuned model, or by a mock for
tests, looks identical from here.
"""
from coord.coordinator import Coordinator, RoutingPolicy

__all__ = ["Coordinator", "RoutingPolicy"]
