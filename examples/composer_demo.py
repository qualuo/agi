"""Composer — runtime demo.

Walks through eight scenarios that match the docstring of
``agi.composer.Composer``:

  1. A three-step RAG pipeline planned by A*
  2. The same goal solved by Dijkstra, IDA* and STRIPS regression
  3. Bonferroni-composed PAC certificate at α = 0.05
  4. Worst-case-vs-independent composition regimes
  5. Closed-loop reliability update (observe successes / failures)
  6. Infeasible-goal diagnosis
  7. SCC report — detecting a cyclic operator registration
  8. Tamper-evident fingerprint replay determinism

Run with::

    python examples/composer_demo.py

Every line printed below is reproducible (the demo seeds the clock and
uses no randomness).
"""
from __future__ import annotations

import json

from agi.composer import (
    ASTAR,
    DIJKSTRA,
    IDA_STAR,
    INDEPENDENT,
    REGRESSION,
    WORST_CASE,
    Composer,
)


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def fmt_plan(plan) -> str:
    lines = [
        f"  verdict       : {plan.verdict}",
        f"  length        : {plan.length}",
        f"  cost          : {plan.cost:.4f}",
        f"  rel mean (Π)  : {plan.reliability_mean:.4f}",
        f"  fingerprint   : {plan.fingerprint[:32]}…",
    ]
    if plan.goal_bindings:
        lines.append(f"  goal bindings : {dict(plan.goal_bindings)}")
    for i, step in enumerate(plan.steps):
        binds = dict(step.bindings)
        lines.append(
            f"    [{i}] {step.op_name:<12s}  {binds}    "
            f"cost={step.cost:.4f}  rel={step.reliability_mean:.3f}"
        )
    return "\n".join(lines)


def fmt_cert(cert) -> str:
    return (
        f"  alpha (1−conf)  : {cert.alpha}\n"
        f"  regime          : {cert.regime}\n"
        f"  bound method    : {cert.bound_method}\n"
        f"  per-step lower  : {[(n, round(v, 4)) for n, v in cert.per_step_lower]}\n"
        f"  per-step upper  : {[(n, round(v, 4)) for n, v in cert.per_step_upper]}\n"
        f"  E2E reliability : [{cert.reliability_lower:.4f}, "
        f"{cert.reliability_upper:.4f}]\n"
        f"  expected cost   : {cert.expected_cost:.4f}\n"
        f"  fingerprint     : {cert.fingerprint[:32]}…"
    )


def build_pipeline(c: Composer) -> None:
    c.register_operator(
        "retrieve",
        params=[("d", "DocId")],
        pre=["indexed(?d)"],
        add=["retrieved(?d)"],
        cost=0.002,
        reliability=0.995,
        prior_strength=200.0,
    )
    c.register_operator(
        "embed",
        params=[("d", "DocId")],
        pre=["retrieved(?d)"],
        add=["embedded(?d)"],
        cost=0.020,
        reliability=0.998,
        prior_strength=200.0,
    )
    c.register_operator(
        "summarise",
        params=[("d", "DocId")],
        pre=["embedded(?d)"],
        add=["summarised(?d)"],
        cost=0.050,
        reliability=0.96,
        prior_strength=20.0,
    )


def main() -> None:
    # ----------------------------------------------------------------
    # 1) Three-step pipeline, A* planning
    # ----------------------------------------------------------------
    banner("1) Three-step RAG pipeline — A* planning")
    c = Composer(clock=lambda: 0.0)
    build_pipeline(c)
    c.add_axiom("indexed(d1)")
    plan = c.synthesize(initial=[], post=["summarised(?d)"], algorithm=ASTAR)
    print(fmt_plan(plan))

    # ----------------------------------------------------------------
    # 2) Same goal solved by alternative algorithms
    # ----------------------------------------------------------------
    banner("2) Same goal solved by Dijkstra / IDA* / STRIPS regression")
    for algo in (DIJKSTRA, IDA_STAR, REGRESSION):
        p = c.synthesize(initial=[], post=["summarised(?d)"], algorithm=algo)
        print(f"  {algo:<11s} -> len={p.length}  cost={p.cost:.4f}  "
              f"rel={p.reliability_mean:.4f}")

    # ----------------------------------------------------------------
    # 3) PAC certificate with Bonferroni-composed step alphas
    # ----------------------------------------------------------------
    banner("3) PAC certificate (α=0.05, Bonferroni-composed)")
    cert = c.verify(plan, alpha=0.05, regime=INDEPENDENT)
    print(fmt_cert(cert))

    # ----------------------------------------------------------------
    # 4) Worst-case vs independent
    # ----------------------------------------------------------------
    banner("4) WORST_CASE vs INDEPENDENT composition")
    cert_wc = c.verify(plan, alpha=0.05, regime=WORST_CASE)
    print(f"  INDEPENDENT  reliability lower bound: {cert.reliability_lower:.4f}")
    print(f"  WORST_CASE   reliability lower bound: {cert_wc.reliability_lower:.4f}")
    print("  (Worst-case union bound is conservative; independent")
    print("  composition is tighter whenever operator failures are")
    print("  uncorrelated — which they typically are when each operator")
    print("  is a separate primitive call from the runtime.)")

    # ----------------------------------------------------------------
    # 5) Closed-loop reliability update
    # ----------------------------------------------------------------
    banner("5) Closed-loop reliability update — observe successes")
    for _ in range(500):
        c.observe("retrieve", True)
        c.observe("embed", True)
        c.observe("summarise", True)
    cert_after = c.verify(plan, alpha=0.05, regime=INDEPENDENT)
    print(f"  Before observation: rel_lower={cert.reliability_lower:.4f}")
    print(f"  After 500 obs/op : rel_lower={cert_after.reliability_lower:.4f}")
    rep = c.report(alpha=0.05)
    for s in rep.operator_stats:
        print(f"    op={s['name']:<11s}  n={s['observations']:<4d} "
              f"k={s['successes']:<4d}  mean={s['mean']:.4f}  "
              f"CI=[{s['lower']:.4f}, {s['upper']:.4f}]")

    # ----------------------------------------------------------------
    # 6) Infeasible-goal diagnosis
    # ----------------------------------------------------------------
    banner("6) Infeasible-goal diagnosis")
    c2 = Composer(clock=lambda: 0.0)
    build_pipeline(c2)
    # No 'indexed' axiom → 'retrieved' is never derivable
    bad = c2.synthesize(initial=[], post=["summarised(?d)"])
    print(f"  verdict       : {bad.verdict}")
    print(f"  length        : {bad.length}")
    cert_bad = c2.verify(bad)
    print(f"  rel_lower     : {cert_bad.reliability_lower}  (zeroed for infeasible)")

    # ----------------------------------------------------------------
    # 7) SCC diagnostic — detect a cyclic operator registration
    # ----------------------------------------------------------------
    banner("7) Cyclic operator graph — SCC diagnostic in the report")
    c3 = Composer(clock=lambda: 0.0)
    c3.register_operator("a", params=[("x", "T")], pre=["p(?x)"], add=["q(?x)"])
    c3.register_operator("b", params=[("x", "T")], pre=["q(?x)"], add=["p(?x)"])
    rep3 = c3.report()
    print(f"  SCCs   : {rep3.sccs}")
    print(f"  Cycles : {rep3.cycles}  (non-empty ⇒ ill-typed registry)")

    # ----------------------------------------------------------------
    # 8) Tamper-evident fingerprint replay determinism
    # ----------------------------------------------------------------
    banner("8) Tamper-evident replay — fingerprint determinism")

    def run() -> str:
        c4 = Composer(clock=lambda: 0.0)
        build_pipeline(c4)
        c4.add_axiom("indexed(d1)")
        plan = c4.synthesize(initial=[], post=["summarised(?d)"])
        c4.verify(plan, alpha=0.05)
        for _ in range(5):
            c4.observe("retrieve", True)
            c4.observe("summarise", False)
        return c4.fingerprint

    a = run()
    b = run()
    print(f"  fingerprint(run1): {a[:32]}…")
    print(f"  fingerprint(run2): {b[:32]}…")
    print(f"  identical: {a == b}")
    print()
    print("  Feed the fingerprint directly into AttestationLedger.append() and")
    print("  any external auditor can re-derive it from the recorded events.")

    # ----------------------------------------------------------------
    # JSON-able output for a coordinator
    # ----------------------------------------------------------------
    banner("Bonus — coordinator-facing JSON")
    j = {
        "plan": plan.to_jsonable(),
        "certificate": cert_after.to_jsonable(),
    }
    print(json.dumps(j, indent=2, sort_keys=False)[:1200] + "...")

    print()
    print("All Composer scenarios completed.  Every fingerprint above")
    print("is reproducible byte-for-byte; the coordination engine can")
    print("hand each certificate to AttestationLedger and have an")
    print("external auditor reconstruct it from the operator-observation log.")


if __name__ == "__main__":
    main()
