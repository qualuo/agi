"""Machine-readable capability descriptor.

A coordination engine asks `GET /capabilities` to learn what this runtime
can do. The shape is intentionally close to OpenRPC: a list of `methods`
(task kinds) with input/output schemas, plus a `roles` enumeration and
a `models` list. The descriptor is the source of truth for the protocol —
the HTTP server and the in-process client both honor it.
"""
from __future__ import annotations


RUNTIME_CAPABILITIES: dict = {
    "name": "agi-runtime",
    "version": "1.0.0",
    "protocol": "agi.runtime/1",
    "description": (
        "Agent-harness runtime. Executes goal-directed tasks via a Claude "
        "Opus 4.7 reasoning core with tool use, persistent memory, skill "
        "retrieval, multi-agent delegation, and trace logging."
    ),
    "models": [
        {"id": "claude-opus-4-7", "context": 1_000_000, "default": True},
        {"id": "claude-sonnet-4-6", "context": 1_000_000},
        {"id": "claude-haiku-4-5-20251001", "context": 200_000},
    ],
    "roles": [
        {"id": "executor", "description": "Default: plan + act + verify."},
        {"id": "planner", "description": "Decompose into a typed task graph; do not execute."},
        {"id": "critic", "description": "Score a candidate response against the prompt."},
    ],
    "methods": [
        {
            "name": "chat",
            "summary": "Run an agent turn on a user message; returns final text.",
            "input": {
                "message": "string (required)",
                "session_id": "string (optional; reuses conversation state)",
                "effort": "low|medium|high (optional)",
                "skills": "list[str] (optional; force-load named skills)",
            },
            "output": {"text": "string", "critic_score": "float|null"},
        },
        {
            "name": "plan",
            "summary": "Decompose a goal into a typed task graph the coordinator can dispatch.",
            "input": {"goal": "string (required)", "constraints": "string (optional)"},
            "output": {"graph": "GraphSpec"},
        },
        {
            "name": "critique",
            "summary": "Score a candidate response for trustworthiness on a goal.",
            "input": {"prompt": "string", "response": "string"},
            "output": {"score": "float in [0,1]", "explanation": "string"},
        },
        {
            "name": "skill.invoke",
            "summary": "Run a named skill from the skill library.",
            "input": {"skill": "string (required)", "args": "dict (optional)"},
            "output": {"text": "string"},
        },
        {
            "name": "graph.submit",
            "summary": "Submit a typed task DAG; runtime executes nodes respecting deps.",
            "input": {"graph": "GraphSpec"},
            "output": {"graph_id": "string", "results": "dict[node_id, any] (when complete)"},
        },
    ],
    "events": [
        {"kind": "task.queued"},
        {"kind": "task.started"},
        {"kind": "task.progress"},
        {"kind": "task.tool_use"},
        {"kind": "task.succeeded"},
        {"kind": "task.failed"},
        {"kind": "task.cancelled"},
        {"kind": "graph.node_ready"},
        {"kind": "graph.completed"},
        {"kind": "runtime.startup"},
    ],
    "features": {
        "streaming": True,
        "idempotency": True,
        "cancellation": True,
        "budgets": ["tokens", "wall_seconds"],
        "persistence": ["memory.jsonl", "traces.jsonl", "skills/"],
        "multi_agent": ["delegate_subagent"],
        "self_extension": ["make_tool"],
        "learning": ["trace_logging", "critic_gate", "skill_compilation", "lora_adapter (offline)"],
    },
}
