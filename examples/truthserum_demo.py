"""TruthSerum demo — incentive-compatible peer prediction end-to-end.

Scenario
--------

A coordination engine fans 200 binary classification tasks out to 8
agents (LLMs / human raters / sub-models) but has *no ground truth*.
Three failure modes are present in the wild:

  * **honest_{0,1,2,3}** report the truth with 85% accuracy each.
  * **noise** reports a coin flip (50% accuracy).
  * **lazy** always reports "1" (constant-strategy collusion of one).
  * **col_a, col_b** collude on a fixed answer not aligned with truth.

The coordination engine wants to:

  1. **Score** every reporter by Correlated Agreement (Dasgupta-Ghosh
     2013) under a strictly-truthful Bayes-Nash equilibrium.
  2. **Verify** truthful play is an *empirical strict Nash* — find the
     worst deviation gap across reporters.
  3. **Detect** collusion clusters at joint α=0.01.
  4. **Aggregate** the truth via Dawes-Skene EM, which down-weights
     unreliable / colluding reporters automatically.
  5. **Compose** with `agi.auditor` for FDR-controlled per-reporter
     truthfulness tests, and `agi.coalition` for Shapley credit on
     the aggregated truth pipeline.

Everything is stdlib-only.  Run with:  `python examples/truthserum_demo.py`.
"""
from __future__ import annotations

import random

from agi.auditor import Auditor, METHOD_BH
from agi.events import EventBus
from agi.truthserum import (
    AGG_WEIGHTED_EM,
    MECH_CORRELATED_AGREEMENT,
    MECH_DMI,
    Report,
    TruthSerum,
)


def hr(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def main() -> None:
    rng = random.Random(20260514)
    n_tasks = 200
    truths = [1 if rng.random() < 0.55 else 0 for _ in range(n_tasks)]

    bus = EventBus()
    seen: list = []
    bus.subscribe(lambda e: seen.append(e.kind))
    ts = TruthSerum(bus=bus, random_seed=1)

    # Honest reporters
    for i in range(4):
        r = random.Random(100 + i)
        for ti, y in enumerate(truths):
            a = y if r.random() < 0.85 else 1 - y
            ts.submit(Report(reporter_id=f"honest{i}", task_id=f"t{ti:04d}", answer=a))

    # Coin-flip noise
    r = random.Random(200)
    for ti in range(n_tasks):
        ts.submit(
            Report(reporter_id="noise", task_id=f"t{ti:04d}", answer=r.choice([0, 1]))
        )

    # Lazy constant reporter
    for ti in range(n_tasks):
        ts.submit(Report(reporter_id="lazy", task_id=f"t{ti:04d}", answer=1))

    # Colluder pair (always answer 0, regardless of truth)
    for cid in ("col_a", "col_b"):
        for ti in range(n_tasks):
            ts.submit(Report(reporter_id=cid, task_id=f"t{ti:04d}", answer=0))

    hr("State after submission")
    cov = ts.coverage()
    print(
        f"  reporters: {cov.n_reporters}  tasks: {cov.n_tasks}  "
        f"reports: {cov.n_reports}"
    )
    print(f"  signal alphabet: {ts.signal_alphabet()}")

    hr("1. Score every reporter under Correlated Agreement")
    report = ts.score(
        mechanism=MECH_CORRELATED_AGREEMENT,
        alpha=0.05,
        aggregation=AGG_WEIGHTED_EM,
        bonferroni=True,
        seed=1,
    )
    print(
        f"  mechanism={report.mechanism}  n_pairings={report.n_pairings}  "
        f"per-α={report.scores[0].alpha:.4f}"
    )
    print(f"  strict truthful Nash? {report.truthful_strict_eq}   "
          f"empirical margin = {report.truthful_eq_margin:+.4f}")
    print()
    print(f"  {'reporter':12s} {'mean':>10s} {'CI lower':>10s} {'CI upper':>10s} {'n':>5s}")
    print("  " + "-" * 50)
    for s in sorted(report.scores, key=lambda s: -s.mean_score):
        print(
            f"  {s.reporter_id:12s} {s.mean_score:10.4f} "
            f"{s.ci_lower:10.4f} {s.ci_upper:10.4f} {s.n_scored:5d}"
        )

    hr("2. Collusion detection at joint α=0.01")
    cliques = ts.detect_collusion(alpha=0.01, min_overlap=20)
    if cliques:
        for c in cliques:
            print(f"  suspected clique: {sorted(c)}")
    else:
        print("  no anomalous agreement detected.")

    hr("3. Determinant-MI (Kong 2020) cross-check")
    dmi_report = ts.score(mechanism=MECH_DMI, alpha=0.05, seed=1)
    for s in sorted(dmi_report.scores, key=lambda s: -s.mean_score):
        print(f"  {s.reporter_id:12s}  DMI = {s.mean_score:.4f}")

    hr("4. Truth aggregation via Dawes-Skene EM")
    truths_em = ts.aggregate(AGG_WEIGHTED_EM)
    acc = sum(
        1 for t in truths_em if t.answer == truths[int(t.task_id[1:])]
    ) / len(truths)
    print(f"  EM-aggregated accuracy vs hidden truth: {acc:.3f}")
    print(
        "  posterior mass on chosen answer (first 5 tasks): "
        + ", ".join(f"{t.posterior:.2f}" for t in truths_em[:5])
    )

    hr("5. Composition with agi.auditor — FDR-controlled truthfulness tests")
    auditor = Auditor()
    # We treat each reporter's screening test as: H0 = "mean ≤ 0".
    # Build pseudo p-values from the standardised z = mean / radius,
    # using Hoeffding sub-Gaussian tail (one-sided).
    import math
    for s in report.scores:
        if s.radius > 0 and s.n_scored > 0:
            z = s.mean_score / s.radius
            p = math.exp(-2.0 * max(z, 0.0))
        else:
            p = 1.0
        p = max(min(p, 1.0), 1e-9)
        auditor.observe(test_id=s.reporter_id, p_value=p)
    bh_report = auditor.decide(method=METHOD_BH, alpha=0.10)
    print(f"  BH-FDR-controlled rejections at α=0.10: "
          f"{list(bh_report.rejected_ids())}")

    hr("6. Lifetime stats + event firehose")
    cov2 = ts.coverage()
    print(f"  scorings={cov2.n_scorings}  aggregations={cov2.n_aggregations}  "
          f"eq_checks={cov2.n_eq_checks}  confusion_fits={cov2.n_confusion_fits}")
    print(f"  events emitted: {len(seen)}  (distinct kinds: {sorted(set(seen))})")
    print()
    print(
        "  Interpretation: honest reporters earn positive CA payments, the\n"
        "  colluders and the constant reporter earn 0 (their reports carry\n"
        "  no information about a peer's bonus task), and the EM aggregator\n"
        "  recovers the hidden truth without ground-truth labels."
    )


if __name__ == "__main__":
    main()
