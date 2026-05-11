"""Runtime engine.

The `agi` package gives you an `Agent` that does one task at a time on the
caller's thread. The `runtime` package wraps the agent in a structured
*task* with lifecycle, budgets, structured events, cancellation, and
delegation, and exposes an `Engine` that runs many tasks concurrently.

A coordination engine (external system that decides *what* to run) drives
this runtime (the layer that knows *how* to run it). Use the Python API
(`Engine.submit`) in-process, or stand up the HTTP server
(`python -m runtime`) and drive it over the wire.
"""
from runtime.task import Task, TaskStatus, TaskEvent, Budget, BudgetExceeded
from runtime.engine import Engine
from runtime.backend import Backend, AnthropicBackend, MockBackend

__all__ = [
    "Task",
    "TaskStatus",
    "TaskEvent",
    "Budget",
    "BudgetExceeded",
    "Engine",
    "Backend",
    "AnthropicBackend",
    "MockBackend",
]
