"""Subagent delegation tool.

A `delegate` tool lets a running agent spawn a *child* run on the same
Runtime. The child runs in its own thread with its own context, returns
its final text, and rolls token usage up to the parent.

This is the smallest credible "multi-agent" primitive: it shares the
Runtime's lifecycle, cancellation, and budget machinery rather than
inventing parallel infrastructure. Other patterns (planner / executor /
critic, fan-out search, debate) compose on top of it.

The parent task only gets the child's final text back — not its
intermediate steps. If the coordinator wants the child's full event
stream, it can subscribe via `runtime.get(child_id).stream()` once it
observes the `delegate.spawned` event on the parent.

Watch out for:
- Unbounded depth. Default `max_depth=3`; raise `RuntimeError` past that.
- Coordination cost. Decomposition only wins when subtasks are
  genuinely separable; otherwise it just multiplies tokens.
"""
from __future__ import annotations

from typing import Any, Callable


_MAX_DEPTH = 3
_DEFAULT_TIMEOUT_S = 600
_DEFAULT_BUDGET_USD = 1.0


def make_delegate_tool(
    runtime: Any,
    parent_run_id: str | None,
    *,
    max_depth: int = _MAX_DEPTH,
    default_timeout_s: float = _DEFAULT_TIMEOUT_S,
    default_budget_usd: float = _DEFAULT_BUDGET_USD,
) -> tuple[dict, Callable[..., str]]:
    """Return a `(schema, handler)` pair the Agent can register.

    Closes over the Runtime + parent run id so the handler can submit
    new runs on the same registry. The parent's depth is read from the
    parent run's metadata at submit-time.
    """
    schema = {
        "name": "delegate",
        "description": (
            "Spawn a child agent run for a clearly-separable subtask. "
            "Returns the child's final text. The child runs in its own "
            "context (no shared chat history) so use this when a fresh "
            "context would help: research, parallel exploration, narrow "
            "verification. The child inherits a conservative cost ceiling; "
            "raise it explicitly for harder subtasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The subtask prompt the child agent will receive.",
                },
                "role": {
                    "type": "string",
                    "description": "Short role label for observability (e.g. 'researcher', 'critic'). Optional.",
                },
                "cost_ceiling_usd": {
                    "type": "number",
                    "description": f"Max $ for the child run (default {default_budget_usd}). The child hard-stops past this.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": f"Max wall time for the child run (default {default_timeout_s}s).",
                },
            },
            "required": ["task"],
        },
    }

    def handler(
        task: str,
        role: str | None = None,
        cost_ceiling_usd: float | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        parent = runtime.get(parent_run_id) if parent_run_id else None
        parent_depth = (parent.metadata.get("depth", 0) if parent else 0)
        if parent_depth >= max_depth:
            return (
                f"error: delegation depth {parent_depth} >= max {max_depth}. "
                "Solve this subtask in the current agent instead of nesting further."
            )

        child = runtime.submit(
            task=task,
            cost_ceiling_usd=cost_ceiling_usd if cost_ceiling_usd is not None else default_budget_usd,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else default_timeout_s,
            parent_id=parent_run_id,
            metadata={"depth": parent_depth + 1, "role": role or "child"},
        )

        # Emit a "spawned" event on the parent's bus so observers (the
        # coordination engine) can subscribe to the child's stream
        # without polling.
        if parent is not None:
            from agi.runtime import Event
            import time

            parent._bus.emit(
                Event(
                    ts=time.time(),
                    run_id=parent.id,
                    type="delegate.spawned",
                    payload={"child_id": child.id, "role": role or "child", "task": task},
                )
            )

        child.wait()

        if child.status.value == "succeeded":
            return child.result
        if child.error:
            return f"[child {child.id} {child.status.value}: {child.error}]"
        return f"[child {child.id} {child.status.value}]"

    return schema, handler
