"""Capability manifest.

A coordination engine asks the runtime "what can you do?" before routing
work to it. The manifest is the answer: a JSON-serializable description of
tools (built-in + synthesized), skills, sub-roles, models, and runtime
limits. Stable schema; additive evolution only.

The coordinator uses this to:
- Decide whether a runtime instance can handle a task (capability match).
- Discover newly-synthesized tools across runtime versions.
- Set sensible budgets (the manifest reports typical $/task).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolDescriptor:
    name: str
    description: str
    input_schema: dict
    origin: str  # "builtin" | "synthesized" | "server"


@dataclass
class SkillDescriptor:
    name: str
    when_to_use: str
    usage_count: int


@dataclass
class RoleDescriptor:
    name: str
    description: str
    system_prompt: str
    model: str


@dataclass
class CapabilityManifest:
    runtime_version: str
    models: list[str]
    tools: list[ToolDescriptor] = field(default_factory=list)
    skills: list[SkillDescriptor] = field(default_factory=list)
    roles: list[RoleDescriptor] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)
    features: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
