"""Reference coordination engine.

This package is an example of how an *external* system can drive the
agi.Runtime. It is deliberately separate from `agi/` so the runtime can
evolve without coupling to one orchestration style.

`coordinator.dag` exposes `DAG`, `Node`, and `run` — a small DAG executor
that runs each node as a one-shot session against a runtime and threads
parent outputs into child inputs.
"""
from coordinator.dag import DAG, Node, run

__all__ = ["DAG", "Node", "run"]
