"""Coordinator — routes Jobs to Runtimes by policy.

Three primitives every coordination engine needs:

1. **Route**: given a job, pick the best runtime for it.
2. **Race**: send the same job to N runtimes, take the first satisfactory
   answer; cancel the rest. Useful for latency-sensitive paths and for
   exploring uncertain plans cheaply.
3. **Budget**: enforce a global spend ceiling across many jobs, not just
   per-job. A single job's `max_cost_usd` cannot, by itself, prevent a
   runaway loop of cheap jobs.

Everything else is composition on top of these.

The current implementation is in-process and synchronous. The interfaces
are deliberately the same shape we'd want for a multi-process / multi-host
deployment — a future RPC layer slots in by replacing `runtime.submit` with
a network call. We are not building that yet; we are not pretending we are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from agi.protocol import Job, JobResult, JobStatus, RuntimeCapabilities
from agi.runtime import Runtime


class RoutingPolicy(str, Enum):
    CHEAPEST = "cheapest"          # lowest input+output rate
    TAGGED = "tagged"              # require all `required_tags` on runtime
    ROUND_ROBIN = "round_robin"    # spread load
    FIRST_AVAILABLE = "first"      # in registration order


@dataclass
class CoordinatorBudget:
    """Global ceiling across all jobs the coordinator submits."""

    max_total_usd: float
    spent_usd: float = 0.0

    def remaining(self) -> float:
        return max(0.0, self.max_total_usd - self.spent_usd)

    def can_afford(self, job: Job) -> bool:
        return self.remaining() >= job.max_cost_usd


@dataclass
class CoordinatorStats:
    submitted: int = 0
    succeeded: int = 0
    failed: int = 0
    rejected_no_budget: int = 0
    rejected_no_route: int = 0
    by_runtime: dict[str, int] = field(default_factory=dict)


class Coordinator:
    """A minimal coordination engine over Runtimes.

    Usage:
        c = Coordinator(budget=CoordinatorBudget(max_total_usd=5.00))
        c.register(opus_runtime(tags=["frontier"]))
        c.register(haiku_runtime(tags=["fast", "cheap"]))
        result = c.run(Job(prompt="..."), policy=RoutingPolicy.CHEAPEST)
        winner = c.race(Job(prompt="..."),
                        runtime_ids=["rt-abc", "rt-def"],
                        accept=lambda r: r.succeeded and len(r.output) > 100)
    """

    def __init__(self, budget: CoordinatorBudget | None = None) -> None:
        self.budget = budget or CoordinatorBudget(max_total_usd=10.0)
        self.runtimes: dict[str, Runtime] = {}
        self._caps_cache: dict[str, RuntimeCapabilities] = {}
        self._rr_index: int = 0
        self.stats = CoordinatorStats()

    # ----- registration -------------------------------------------------------

    def register(self, runtime: Runtime) -> str:
        rid = runtime.runtime_id
        self.runtimes[rid] = runtime
        self._caps_cache[rid] = runtime.capabilities()
        return rid

    def deregister(self, runtime_id: str) -> None:
        self.runtimes.pop(runtime_id, None)
        self._caps_cache.pop(runtime_id, None)

    def capabilities(self) -> list[RuntimeCapabilities]:
        return list(self._caps_cache.values())

    # ----- routing ------------------------------------------------------------

    def route(
        self,
        job: Job,
        policy: RoutingPolicy = RoutingPolicy.CHEAPEST,
        required_tags: list[str] | None = None,
    ) -> Runtime | None:
        candidates = list(self.runtimes.values())
        if required_tags:
            tagset = set(required_tags)
            candidates = [
                r for r in candidates
                if tagset.issubset(set(self._caps_cache[r.runtime_id].tags))
            ]
        if not candidates:
            return None

        if policy == RoutingPolicy.CHEAPEST:
            return min(
                candidates,
                key=lambda r: self._caps_cache[r.runtime_id].cost_per_1m_output_usd,
            )
        if policy == RoutingPolicy.ROUND_ROBIN:
            r = candidates[self._rr_index % len(candidates)]
            self._rr_index += 1
            return r
        # FIRST_AVAILABLE / TAGGED both fall through to first candidate
        return candidates[0]

    # ----- execution ----------------------------------------------------------

    def run(
        self,
        job: Job,
        policy: RoutingPolicy = RoutingPolicy.CHEAPEST,
        required_tags: list[str] | None = None,
    ) -> JobResult:
        """Route the job, enforce global budget, submit, account."""
        self.stats.submitted += 1
        if not self.budget.can_afford(job):
            self.stats.rejected_no_budget += 1
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.BUDGET_EXCEEDED,
                error=f"global budget ${self.budget.remaining():.4f} < job ${job.max_cost_usd:.4f}",
            )
        rt = self.route(job, policy=policy, required_tags=required_tags)
        if rt is None:
            self.stats.rejected_no_route += 1
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error="no runtime matches policy/tags",
            )
        # Cap the job's spend at whatever's left globally.
        capped = Job(**{**job.to_dict(),
                        "max_cost_usd": min(job.max_cost_usd, self.budget.remaining())})
        result = rt.submit(capped)
        self.budget.spent_usd += result.cost_usd
        self.stats.by_runtime[rt.runtime_id] = self.stats.by_runtime.get(rt.runtime_id, 0) + 1
        if result.succeeded:
            self.stats.succeeded += 1
        else:
            self.stats.failed += 1
        return result

    def race(
        self,
        job: Job,
        runtime_ids: list[str],
        accept: Callable[[JobResult], bool] | None = None,
    ) -> JobResult:
        """Submit to multiple runtimes; return the first that meets `accept`.

        Synchronous in v1: walks runtime_ids in order. The shape is here so
        an async/threaded implementation can swap in without changing callers.
        """
        accept = accept or (lambda r: r.succeeded)
        last: JobResult | None = None
        for rid in runtime_ids:
            rt = self.runtimes.get(rid)
            if rt is None:
                continue
            remaining = self.budget.remaining()
            if remaining <= 0:
                continue
            sub = Job(**{
                **job.to_dict(),
                "job_id": f"{job.job_id}.{rid}",
                "max_cost_usd": min(job.max_cost_usd, remaining),
            })
            result = rt.submit(sub)
            self.budget.spent_usd += result.cost_usd
            self.stats.by_runtime[rid] = self.stats.by_runtime.get(rid, 0) + 1
            self.stats.submitted += 1
            last = result
            if accept(result):
                self.stats.succeeded += 1
                return result
            self.stats.failed += 1
        return last or JobResult(
            job_id=job.job_id,
            status=JobStatus.FAILED,
            error="no runtime produced an acceptable result",
        )
