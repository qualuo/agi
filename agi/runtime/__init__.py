"""Coordination-engine runtime.

The runtime exposes the agent harness as a long-running execution engine that
an external coordination engine (or another agent) can drive over a stable
JSON protocol. Components:

- `tasks`: task lifecycle, state machine, idempotent submission
- `events`: in-process pub/sub event bus with replay
- `graph`: typed DAG executor — submit a task graph, get streamed results
- `worker`: agent workers that execute leaf tasks
- `server`: stdlib HTTP server (POST /tasks, GET /tasks/{id}/stream, ...)
- `capabilities`: machine-readable description of what this runtime can do

The runtime is intentionally small and stdlib-only so it can be embedded
inside other Python processes, run as a sidecar, or invoked directly.
"""
from agi.runtime.tasks import Task, TaskStatus, TaskStore, TaskSpec
from agi.runtime.events import Event, EventBus
from agi.runtime.graph import GraphSpec, NodeSpec, GraphExecutor, GraphResult
from agi.runtime.capabilities import RUNTIME_CAPABILITIES
from agi.runtime.server import Runtime, serve
from agi.runtime.worker import Worker, make_default_registry

__all__ = [
    "Task",
    "TaskStatus",
    "TaskStore",
    "TaskSpec",
    "Event",
    "EventBus",
    "GraphSpec",
    "NodeSpec",
    "GraphExecutor",
    "GraphResult",
    "RUNTIME_CAPABILITIES",
    "Runtime",
    "serve",
    "Worker",
    "make_default_registry",
]
