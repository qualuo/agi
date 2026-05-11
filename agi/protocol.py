"""Wire protocol between a Runtime and a coordination engine.

The Runtime (this package) executes work. A coordination engine — written by
us, by a customer, or by a third party — routes work to runtimes, enforces
budgets, branches on uncertainty, and aggregates results.

Everything here is plain dataclasses with a `to_dict()` / `from_dict()` round
trip, so the protocol survives a JSON or a process boundary. A coordinator
implementer does not need to install `anthropic` to depend on this module.

Design constraints:
- Forward-compatible: unknown fields on the wire are dropped, not errored.
- Idempotent: a coordinator may resubmit the same `job_id`; the runtime is
  free to return the prior result.
- Budget-first: every job carries `max_cost_usd`; runtimes must enforce.
- Auditable: every result references the trace id that produced it.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A unit of work submitted to a Runtime by a coordinator.

    `prompt` is the user-message contents. `output_contract` is an optional
    free-text description of what the coordinator expects back (e.g. "a
    JSON object with keys `summary` and `confidence`"); the runtime adds it
    to the prompt so the agent knows the shape it must return. Coordinators
    that need machine-checkable shapes should validate `JobResult.output`
    themselves — runtimes do not parse it.
    """

    prompt: str
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    output_contract: str | None = None
    max_cost_usd: float = 1.00
    max_iterations: int = 25
    session_id: str | None = None  # if set, runtime resumes that session
    metadata: dict[str, Any] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Job":
        return cls(**_only_known(cls, d))


@dataclass
class JobResult:
    """What a runtime returns for a job.

    `output` is the agent's final text. Coordinators that asked for a
    structured `output_contract` are responsible for parsing it.

    `cost_usd` is the actual spend; `budget_remaining_usd` lets a coordinator
    keep a running tab without rescanning history.

    `trace_id` references the durable trace logger entry — the audit trail.
    """

    job_id: str
    status: JobStatus
    output: str = ""
    error: str | None = None
    cost_usd: float = 0.0
    budget_remaining_usd: float = 0.0
    elapsed_seconds: float = 0.0
    iterations: int = 0
    trace_id: str | None = None
    critic_score: float | None = None
    session_id: str | None = None  # set so the coordinator can fork/resume
    finished_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobResult":
        d = dict(d)
        if "status" in d and not isinstance(d["status"], JobStatus):
            d["status"] = JobStatus(d["status"])
        return cls(**_only_known(cls, d))

    @property
    def succeeded(self) -> bool:
        return self.status == JobStatus.SUCCEEDED


@dataclass
class ToolDescriptor:
    """How the runtime advertises a single tool to a coordinator."""

    name: str
    description: str
    server_side: bool = False  # True for web_search / web_fetch etc.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolDescriptor":
        return cls(**_only_known(cls, d))


@dataclass
class RuntimeCapabilities:
    """What a runtime can do. A coordinator reads this to route work.

    `model` and `cost_per_1m_input/output` let the coordinator pick the
    cheapest runtime that can do the job. `tools` advertises action
    surface — a coordinator that needs `web_search` but is talking to a
    runtime without it will route elsewhere.

    `tags` is freeform — runtimes can self-describe ("expert:legal",
    "specialist:critic", "fast", "cheap") and coordinators can match.
    """

    runtime_id: str
    model: str
    cost_per_1m_input_usd: float
    cost_per_1m_output_usd: float
    tools: list[ToolDescriptor]
    has_critic: bool = False
    has_memory: bool = True
    tags: list[str] = field(default_factory=list)
    version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "tools"},
            "tools": [t.to_dict() if isinstance(t, ToolDescriptor) else t for t in self.tools],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeCapabilities":
        d = dict(d)
        if "tools" in d:
            d["tools"] = [
                t if isinstance(t, ToolDescriptor) else ToolDescriptor.from_dict(t)
                for t in d["tools"]
            ]
        return cls(**_only_known(cls, d))


@dataclass
class ProgressEvent:
    """Streamed mid-job event a coordinator can subscribe to.

    Kept small and string-typed so it survives JSON. `kind` is one of:
    "turn_started", "tool_call", "text_delta", "budget_check", "turn_ended".
    """

    job_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProgressEvent":
        return cls(**_only_known(cls, d))


def _only_known(cls, d: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys so older runtimes can read newer payloads."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in known}
