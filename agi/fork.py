"""Session fork — race N variants of a task, return the best.

A coordination engine often wants to *hedge*: ask the same question with
different configs (different roles, different effort, different system
prompts), run them concurrently, and take the winner by some judge.

`SessionFork.race(prompt, variants)` submits one Task per variant to a
TaskRunner pool, blocks on completion, and selects via a pluggable judge.
Default judge prefers:
    1. higher critic_score (when available),
    2. else success status,
    3. else lower cost.

This is one of the cheapest ways to *measurably* lift pass rate when
individual attempts are noisy. It also stress-tests the runtime: many
sessions in flight, shared event bus, separate cost accounting.

It deliberately leans on the existing TaskQueue/TaskRunner so the same
bookkeeping (events, costs, cancellation) flows uniformly. The fork
itself is dumb glue.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable

from agi.events import Event
from agi.runtime import Runtime, SessionConfig
from agi.tasks import (
    Task,
    TaskQueue,
    TaskRunner,
    submit_task,
)


@dataclass
class ForkVariant:
    """A named SessionConfig variant for racing."""
    name: str
    config: SessionConfig
    priority: int = 0


@dataclass
class ForkOutcome:
    """The result of one variant's run within a race."""
    variant: ForkVariant
    task_id: str
    session_id: str | None
    status: str
    result: str | None
    error: str | None
    cost_usd: float
    duration_seconds: float
    critic_score: float | None = None
    judge_score: float | None = None


@dataclass
class RaceResult:
    """The aggregate of a single race."""
    prompt: str
    outcomes: list[ForkOutcome]
    winner: ForkOutcome | None
    total_cost_usd: float
    total_duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.winner is not None and self.winner.status == "done"


Judge = Callable[[ForkOutcome], float]


def default_judge(o: ForkOutcome) -> float:
    """Higher is better. Order: critic_score, then done-success, then
    -cost (cheaper is better), then -duration."""
    if o.status != "done" or not (o.result or "").strip():
        return -1e9
    score = 0.0
    if o.critic_score is not None:
        # critic_score is in [0,1]; weight heavily.
        score += 100.0 * o.critic_score
    else:
        score += 50.0  # baseline for "succeeded"
    # Penalize cost / latency softly so ties break on cheapest fastest
    score -= o.cost_usd * 10.0
    score -= o.duration_seconds * 0.1
    return score


class SessionFork:
    """Runs N variants in parallel against a Runtime and picks a winner.

    `max_workers` bounds in-flight tasks. Each variant gets its own
    Session (so memory, costs, and events don't cross). Cancellation is
    cooperative: when a clear winner emerges and `cancel_losers` is
    True, sibling sessions are asked to cancel — they will halt at the
    next chat boundary (Sessions don't kill streams mid-token).
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        queue: TaskQueue | None = None,
        max_workers: int = 4,
        judge: Judge = default_judge,
    ) -> None:
        self.runtime = runtime
        self.queue = queue or TaskQueue()
        self.max_workers = max_workers
        self.judge = judge

    def race(
        self,
        prompt: str,
        variants: list[ForkVariant],
        *,
        namespace: str | None = None,
        tag: str | None = None,
        deadline_ts: float | None = None,
        cancel_losers: bool = False,
        first_success_wins: bool = False,
    ) -> RaceResult:
        if not variants:
            raise ValueError("race needs at least one variant")
        start = time.time()
        self.runtime.bus.publish(
            Event(
                kind="fork.race_started",
                data={
                    "prompt": prompt[:200],
                    "variant_count": len(variants),
                    "variants": [v.name for v in variants],
                },
            )
        )

        # One TaskRunner per worker; they all drain the same queue, but
        # each worker is single-threaded internally (matches Coordinator
        # semantics: parallelism = N runners).
        runners = [TaskRunner(self.runtime, self.queue) for _ in range(min(self.max_workers, len(variants)))]
        task_ids: list[str] = []
        variant_by_task: dict[str, ForkVariant] = {}
        for v in variants:
            tid = submit_task(
                self.queue,
                prompt=prompt,
                session_config=v.config,
                priority=v.priority,
                deadline_ts=deadline_ts,
                namespace=namespace,
                tag=tag,
            )
            task_ids.append(tid)
            variant_by_task[tid] = v

        outcomes: list[ForkOutcome] = []
        outcomes_lock = threading.Lock()
        early_winner = threading.Event()

        def drain(runner: TaskRunner) -> None:
            while True:
                if first_success_wins and early_winner.is_set():
                    return
                task = runner.tick()
                if task is None:
                    return
                outcome = self._task_to_outcome(task, variant_by_task[task.id])
                with outcomes_lock:
                    outcomes.append(outcome)
                if first_success_wins and outcome.status == "done" and (outcome.result or "").strip():
                    early_winner.set()

        with ThreadPoolExecutor(max_workers=len(runners)) as ex:
            futures: list[Future[None]] = [ex.submit(drain, r) for r in runners]
            for f in futures:
                f.result()

        # If first_success_wins triggered, cancel any tasks still queued
        # (they would otherwise run after our return).
        if first_success_wins and early_winner.is_set():
            for tid in task_ids:
                self.queue.cancel(tid)

        # Score and pick winner
        for o in outcomes:
            try:
                o.judge_score = self.judge(o)
            except Exception:
                o.judge_score = None
        ranked = sorted(
            outcomes,
            key=lambda o: -(o.judge_score if o.judge_score is not None else -1e9),
        )
        winner = ranked[0] if ranked and ranked[0].judge_score is not None and ranked[0].judge_score > -1e9 else None
        if winner is not None and cancel_losers:
            for o in outcomes:
                if o is winner or not o.session_id:
                    continue
                try:
                    self.runtime.cancel(o.session_id)
                except KeyError:
                    pass

        total_cost = sum(o.cost_usd for o in outcomes)
        self.runtime.bus.publish(
            Event(
                kind="fork.race_completed",
                data={
                    "prompt": prompt[:200],
                    "winner": winner.variant.name if winner else None,
                    "total_cost_usd": total_cost,
                    "variant_count": len(variants),
                },
            )
        )
        return RaceResult(
            prompt=prompt,
            outcomes=outcomes,
            winner=winner,
            total_cost_usd=total_cost,
            total_duration_seconds=time.time() - start,
        )

    def _task_to_outcome(self, task: Task, variant: ForkVariant) -> ForkOutcome:
        critic_score: float | None = None
        if task.session_id:
            try:
                sess = self.runtime.get_session(task.session_id)
                critic_score = sess.state.last_critic_score
            except KeyError:
                pass
        cost = 0.0
        if task.session_id:
            try:
                sess = self.runtime.get_session(task.session_id)
                cost = sess.state.total_cost_usd
            except KeyError:
                pass
        duration = (
            (task.completed_ts or time.time()) - (task.started_ts or task.created_ts)
        )
        return ForkOutcome(
            variant=variant,
            task_id=task.id,
            session_id=task.session_id,
            status=task.status,
            result=task.result,
            error=task.error,
            cost_usd=cost,
            duration_seconds=duration,
            critic_score=critic_score,
        )
