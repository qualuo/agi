"""Deliberator demo — adaptive sequential sampling with anytime-valid stopping.

This is the runtime kernel a coordination engine uses to decide *how much
compute* to spend per query. It draws samples one at a time from a caller-
supplied sampler (an LLM call, a tool invocation, anything stochastic),
clusters them by a caller-defined `cluster_key`, and decides when to stop:

    STOP_EVIDENCE     anytime-valid lower bound on P(top cluster) clears the
                      commit threshold — commit the answer, statistically
                      defensible under sequential testing.
    STOP_CONVERGENCE  posterior has stabilised but no cluster dominates —
                      ambiguous query, escalate to a stronger model.
    STOP_BUDGET       max_samples / max_cost reached — defer.

The investor pitch:

    "Naive self-consistency draws K samples regardless of difficulty,
     spending 10x compute on a query the model would have nailed in 2
     samples. Deliberator gives the runtime one dial — a quality level α —
     and the runtime figures out how many samples each query actually
     deserves. Cheap queries finish fast; hard ones get escalated before
     they burn budget. The early stopping is mathematically honest because
     the lower bound is anytime-valid (Waudby-Smith & Ramdas, JMLR 2024)."

This script runs three workloads and shows what the kernel does:

    1. EASY: the underlying answer distribution is sharply peaked. The
       deliberator should commit in 5-10 samples.

    2. AMBIGUOUS: the distribution is genuinely tied across 3 clusters.
       The deliberator should detect convergence and escalate.

    3. HARD: the distribution narrowly favours one cluster (60/40). The
       deliberator should either exhaust budget or commit only after
       many samples — exactly the case where escalation pays off.
"""
from __future__ import annotations

import random
import statistics

from agi.deliberator import (
    Deliberator,
    Sample,
    STOP_BUDGET,
    STOP_CONVERGENCE,
    STOP_EVIDENCE,
)
from agi.events import Event, EventBus


def make_sampler(rng: random.Random, weights: dict[str, float], cost_per_sample: float = 0.01):
    keys = list(weights)
    probs = [weights[k] for k in keys]
    z = sum(probs)
    probs = [p / z for p in probs]

    def _draw() -> Sample:
        u = rng.random()
        c = 0.0
        for k, p in zip(keys, probs):
            c += p
            if u <= c:
                return Sample(answer=k, cluster_key=k, cost=cost_per_sample)
        return Sample(answer=keys[-1], cluster_key=keys[-1], cost=cost_per_sample)

    return _draw


def run_workload(label: str, weights: dict[str, float], *, trials: int = 30) -> None:
    print()
    print(f"=== {label} ===")
    print(f"true cluster mass: {weights}")
    rng = random.Random(7)
    d = Deliberator()
    n_by_reason: dict[str, int] = {}
    samples_by_reason: dict[str, list[int]] = {}
    cost_by_reason: dict[str, list[float]] = {}
    correct_evidence_commits = 0
    total_evidence = 0
    top_truth = max(weights, key=weights.get)

    for _ in range(trials):
        sampler = make_sampler(rng, weights, cost_per_sample=0.012)
        delib = d.deliberate(
            sampler,
            max_samples=24,
            max_cost=10.0,
            alpha=0.05,
            commit_threshold=0.5,
            eig_floor=0.005,
        )
        n_by_reason[delib.stop_reason] = n_by_reason.get(delib.stop_reason, 0) + 1
        samples_by_reason.setdefault(delib.stop_reason, []).append(delib.n_samples)
        cost_by_reason.setdefault(delib.stop_reason, []).append(delib.cost)
        if delib.stop_reason == STOP_EVIDENCE:
            total_evidence += 1
            ok = delib.cluster_key == top_truth
            d.observe(delib, success=ok)
            if ok:
                correct_evidence_commits += 1
        else:
            d.observe(delib, success=None)

    for reason, n in n_by_reason.items():
        ns = samples_by_reason[reason]
        cs = cost_by_reason[reason]
        print(
            f"  {reason:12s} n={n:3d}  "
            f"samples median={statistics.median(ns):4.1f} mean={statistics.fmean(ns):4.1f}  "
            f"cost mean=${statistics.fmean(cs):.4f}"
        )
    if total_evidence > 0:
        rate = correct_evidence_commits / total_evidence
        print(
            f"  commits that picked the true top cluster: "
            f"{correct_evidence_commits}/{total_evidence} = {rate:.0%}"
        )
    cov = d.coverage_report()
    print(
        f"  coverage report: realised={cov.realised_success_rate:.2%} "
        f"target={cov.target_success_rate:.2%}  "
        f"miscoverage={cov.miscoverage:.2%}"
    )


def demo_event_stream() -> None:
    """Show that a coordination engine can subscribe to the deliberator's
    event stream and react to per-sample telemetry in real time."""
    print()
    print("=== event-stream telemetry ===")
    bus = EventBus()
    log: list[Event] = []
    bus.subscribe(lambda e: log.append(e))
    d = Deliberator(bus=bus)
    rng = random.Random(0)
    sampler = make_sampler(rng, {"yes": 0.9, "no": 0.1})
    delib = d.deliberate(sampler, max_samples=12, alpha=0.05, commit_threshold=0.5)
    print(f"  emitted {len(log)} events, terminal kind = {log[-1].kind!r}")
    print(f"  decision: cluster={delib.cluster_key!r}  "
          f"posterior_mean={delib.posterior_mean:.3f}  "
          f"posterior_lower={delib.posterior_lower:.3f}  "
          f"stop_reason={delib.stop_reason}")
    print(f"  receipt hash: {delib.receipt_hash}")
    print(f"  rationale: {delib.rationale}")


def demo_savings_vs_fixed_n() -> None:
    """Show concrete compute savings versus a naive fixed-K self-consistency."""
    print()
    print("=== compute savings vs. fixed-K self-consistency ===")
    rng = random.Random(2024)
    d = Deliberator()
    fixed_k = 16
    weights_seq = [
        ("trivial (95/5)", {"yes": 0.95, "no": 0.05}),
        ("easy (85/15)",   {"yes": 0.85, "no": 0.15}),
        ("medium (70/30)", {"yes": 0.70, "no": 0.30}),
        ("hard (55/45)",   {"yes": 0.55, "no": 0.45}),
    ]
    print(f"  baseline: fixed K = {fixed_k} samples per query")
    print(f"  {'workload':<18s}  {'mean N':>8s}  {'savings':>10s}  {'stop reasons'}")
    for label, w in weights_seq:
        rng2 = random.Random(2024)
        sampler = make_sampler(rng2, w, cost_per_sample=0.01)
        ns: list[int] = []
        reasons: dict[str, int] = {}
        for _ in range(40):
            delib = d.deliberate(
                sampler,
                max_samples=fixed_k,
                alpha=0.05,
                commit_threshold=0.5,
                eig_floor=0.005,
            )
            ns.append(delib.n_samples)
            reasons[delib.stop_reason] = reasons.get(delib.stop_reason, 0) + 1
        mean_n = statistics.fmean(ns)
        savings = 1.0 - mean_n / fixed_k
        breakdown = ", ".join(f"{k}={v}" for k, v in reasons.items())
        print(f"  {label:<18s}  {mean_n:>8.2f}  {savings:>9.0%}  {breakdown}")


def main() -> None:
    print("Deliberator — adaptive sequential sampling kernel")
    print("=" * 60)
    run_workload("EASY: one cluster dominates (90/10)",
                 {"yes": 0.9, "no": 0.1})
    run_workload("AMBIGUOUS: three-way tie",
                 {"a": 1.0, "b": 1.0, "c": 1.0})
    run_workload("HARD: narrow 60/40 lead",
                 {"yes": 0.6, "no": 0.4})
    demo_event_stream()
    demo_savings_vs_fixed_n()
    print()
    print("[Done] Deliberator is the adaptive-compute kernel of the runtime:")
    print("       confident queries commit fast, ambiguous queries escalate,")
    print("       and the early stopping is anytime-valid.")


if __name__ == "__main__":
    main()
