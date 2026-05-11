"""Capability introspection.

A coordination engine that dispatches work to multiple runtimes needs to
know what each runtime can do: which model, which tools, what budgets are
sane, what skills are loaded. `Capabilities` is the contract — a single
JSON-serializable structure the engine can fetch once and route against.

This is deliberately small. Anything richer (e.g. per-tool latency
histograms) belongs in observability, not capability advertisement.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from agi.costs import PRICING


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    kind: str = "client"  # "client" (we dispatch) | "server" (Anthropic dispatches)


@dataclass
class Capabilities:
    runtime_name: str
    runtime_version: str
    model: str
    pricing: dict[str, float] = field(default_factory=dict)   # {"input_per_mtok": ..., "output_per_mtok": ...}
    tools: list[ToolSpec] = field(default_factory=list)
    skills: list[dict[str, str]] = field(default_factory=list)  # [{"name":..., "description":...}]
    features: dict[str, bool] = field(default_factory=dict)
    default_budget: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tools"] = [asdict(t) for t in self.tools]
        return d


RUNTIME_NAME = "agi-runtime"
RUNTIME_VERSION = "0.2.0"


def describe_runtime(agent_factory: Callable[[], Any]) -> Capabilities:
    """Build a Capabilities object by spinning up an agent and reading
    its surface. The factory is called exactly once; the agent is then
    discarded. Side-effect-free for memory because the factory should
    return an isolated agent."""
    agent = agent_factory()
    model = getattr(agent, "model", "unknown")
    tool_schemas = list(getattr(agent, "tool_schemas", []))
    handlers = getattr(agent, "handlers", {})

    tools: list[ToolSpec] = []
    for schema in tool_schemas:
        name = schema.get("name", "")
        if not name:
            continue
        if "type" in schema and schema["type"].startswith(("web_search", "web_fetch")):
            kind = "server"
            tools.append(ToolSpec(name=name, description="Anthropic server-side tool", kind=kind))
            continue
        tools.append(ToolSpec(
            name=name,
            description=schema.get("description", ""),
            input_schema=schema.get("input_schema", {}),
            kind="client" if name in handlers else "server",
        ))

    # Skills (optional — skill library may not exist at runtime build time)
    skills: list[dict[str, str]] = []
    try:
        from agi.skills import SkillLibrary
        for s in SkillLibrary().list():
            skills.append({"name": s.name, "description": s.description})
    except Exception:
        pass

    pricing: dict[str, float] = {}
    if model in PRICING:
        in_rate, out_rate = PRICING[model]
        pricing = {"input_per_mtok": in_rate, "output_per_mtok": out_rate}

    return Capabilities(
        runtime_name=RUNTIME_NAME,
        runtime_version=RUNTIME_VERSION,
        model=model,
        pricing=pricing,
        tools=tools,
        skills=skills,
        features={
            "streaming": True,
            "thinking": True,
            "memory": True,
            "critic_gate": getattr(agent, "critic", None) is not None,
            "skill_library": bool(skills),
            "trace_logging": getattr(agent, "tracer", None) is not None,
        },
        default_budget={
            "max_iterations": 25,
            "max_tokens": 200_000,
            "max_cost_usd": 5.0,
            "deadline_s": 600.0,
        },
    )
