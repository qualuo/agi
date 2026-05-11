"""Capability manifest.

Coordination engines route tasks based on capability, not on which runtime
they happen to be holding. The manifest is a machine-readable description of
what this runtime can do: available tools (with schemas), supported models
(with pricing), feature flags, and protocol version. A coordinator can diff
two manifests to decide which runtime to dispatch to.
"""
from __future__ import annotations

from typing import Any

from agi.costs import PRICING
from agi.memory import Memory
from agi.tools import make_tools


PROTOCOL_VERSION = "1.0"
RUNTIME_NAME = "agi-runtime"


def build_manifest(*, include_web: bool = True) -> dict[str, Any]:
    """Return the manifest a coordinator should read on startup."""
    schemas, _ = make_tools(Memory(path="/tmp/_manifest_memory.jsonl"))
    tools: list[dict[str, Any]] = list(schemas)
    if include_web:
        tools.append(
            {
                "name": "web_search",
                "kind": "server",
                "description": "Server-side web search executed by the Anthropic API.",
            }
        )
        tools.append(
            {
                "name": "web_fetch",
                "kind": "server",
                "description": "Server-side URL fetch executed by the Anthropic API.",
            }
        )

    models = [
        {
            "id": model_id,
            "input_usd_per_mtok": rates[0],
            "output_usd_per_mtok": rates[1],
        }
        for model_id, rates in PRICING.items()
    ]

    return {
        "runtime": RUNTIME_NAME,
        "protocol": PROTOCOL_VERSION,
        "tools": tools,
        "models": models,
        "features": {
            "streaming": True,
            "sessions": True,
            "async_jobs": True,
            "budgets": True,
            "memory": True,
            "traces": True,
            "critic_gate": True,
        },
        "endpoints": [
            "GET  /v1/health",
            "GET  /v1/capabilities",
            "GET  /v1/metrics",
            "POST /v1/sessions",
            "GET  /v1/sessions",
            "GET  /v1/sessions/{sid}",
            "DELETE /v1/sessions/{sid}",
            "POST /v1/sessions/{sid}/messages",
            "POST /v1/sessions/{sid}/jobs",
            "GET  /v1/jobs/{jid}",
            "GET  /v1/jobs/{jid}/stream",
            "POST /v1/jobs/{jid}/cancel",
            "GET  /v1/sessions/{sid}/memory",
            "POST /v1/sessions/{sid}/memory",
        ],
    }
