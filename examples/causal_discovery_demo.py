"""CausalDiscoverer demo — learn the DAG, plan the next intervention.

A coordination engine has logged 1,200 (context, action, reward) tuples
across two model variants. Without a causal graph, the engine can only
chase whatever feature *correlates* with success this week. With
CausalDiscoverer it learns which features *cause* success and which are
spurious — then asks "which variable should I intervene on next sprint
to maximally disambiguate what's left ambiguous?"

This demo simulates that loop end-to-end:

  1. Generate observational data from a known DAG with two true causes,
     one confounder, and one spurious correlate.
  2. Run PC to recover the CPDAG; verify ground-truth recovery.
  3. Run bootstrap-PC to get edge-stability confidence intervals.
  4. Score the CPDAG by BIC and compute the Markov blanket of the
     outcome variable — the minimal sufficient feature set for routing.
  5. Use active intervention selection to pick the next experiment.

Stdlib only; runs in a couple of seconds.
"""
from __future__ import annotations

import random

from agi import (
    CausalDiscoverer,
    DiscoveryRequest,
    intervention_targets,
    run_pc,
)


def synthetic_routing_dag(n: int, seed: int = 42) -> tuple[list[list[float]], list[str]]:
    """Ground truth:

        difficulty ──┐
                     ├──> success
        prompt_len ──┘                ^
                                      │
        tenant_tier ────┐             │
                        ├──> latency ─┘  (latency is a confounder)
        model_variant ──┘

        spurious ⫫ everything (pure noise that should not be in the
                                Markov blanket of success).

    `success` (the outcome) depends causally on `difficulty`,
    `prompt_len`, and `latency`. The runtime's coordinator should NOT
    condition on `spurious` (no causal link) but SHOULD condition on
    `latency` (a parent).
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        difficulty = rng.gauss(0, 1)
        prompt_len = rng.gauss(0, 1)
        tenant_tier = rng.gauss(0, 1)
        model_variant = rng.gauss(0, 1)
        latency = 0.6 * tenant_tier + 0.6 * model_variant + rng.gauss(0, 0.4)
        success = (
            -0.5 * difficulty - 0.3 * prompt_len - 0.4 * latency + rng.gauss(0, 0.4)
        )
        spurious = rng.gauss(0, 1)
        rows.append(
            [
                difficulty,
                prompt_len,
                tenant_tier,
                model_variant,
                latency,
                success,
                spurious,
            ]
        )
    variables = [
        "difficulty",
        "prompt_len",
        "tenant_tier",
        "model_variant",
        "latency",
        "success",
        "spurious",
    ]
    return rows, variables


def main() -> None:
    rows, variables = synthetic_routing_dag(n=1200)

    print("=== CausalDiscoverer demo ===")
    print(f"  n_samples = {len(rows)}, n_variables = {len(variables)}")
    print()

    # ---- 1. Single-shot PC ----
    discoverer = CausalDiscoverer()
    report = discoverer.discover(
        rows, variables, request=DiscoveryRequest(method="pc", alpha=0.05)
    )
    print(f"-- PC ({report.elapsed_seconds * 1000:.1f} ms) --")
    print(f"  directed:   {sorted(report.graph.directed)}")
    print(
        f"  undirected: {sorted([sorted(list(e)) for e in report.graph.undirected])}"
    )
    print(f"  BIC:        {report.bic_score:.2f}")
    print()

    # ---- 2. Bootstrap stability ----
    report_b = discoverer.discover(
        rows,
        variables,
        request=DiscoveryRequest(
            method="bootstrap_pc",
            n_bootstrap=30,
            edge_threshold=0.7,
            seed=42,
            alpha=0.05,
        ),
    )
    print(f"-- Bootstrap-PC, 30 resamples, edge threshold 0.7 --")
    for a, b, kind, conf in report_b.graph.edge_summary():
        print(f"  {a:>14} {kind} {b:<14}  conf = {conf:.2f}")
    print()

    # ---- 3. Markov blanket of `success` ----
    mb = report.graph.markov_blanket("success")
    print(f"-- Markov blanket of outcome --")
    print(f"  MB(success) = {sorted(mb)}")
    print(
        "  (Minimal sufficient feature set for routing — conditioning on this\n"
        "   makes the outcome independent of every other observed variable.\n"
        "   Note 'spurious' is correctly excluded.)"
    )
    print()

    # ---- 4. Active intervention selection (canonical ambiguous case) ----
    # A 3-variable chain X — Y — Z is the simplest CPDAG where
    # observation cannot identify direction (chain, fork, and reverse
    # chain are Markov equivalent). The runtime must intervene to
    # resolve it. Intervening on Y orients both Y — X and Y — Z; on
    # X or Z only orients one edge.
    chain_rows = []
    rng = random.Random(7)
    for _ in range(500):
        x = rng.gauss(0, 1)
        y = 0.8 * x + rng.gauss(0, 0.5)
        z = 0.8 * y + rng.gauss(0, 0.5)
        chain_rows.append([x, y, z])
    chain_report = discoverer.discover(
        chain_rows,
        ["X", "Y", "Z"],
        request=DiscoveryRequest(method="pc", alpha=0.05),
    )
    targets = intervention_targets(chain_report.graph, budget=3)
    print(f"-- Active intervention selection on Markov-equivalent chain --")
    print(
        f"  Current CPDAG: directed={len(chain_report.graph.directed)}, "
        f"undirected={len(chain_report.graph.undirected)}"
    )
    print(f"  (chain & fork are observationally indistinguishable — need to intervene)")
    for t in targets:
        print(
            f"  intervene on {t.variable!r}: "
            f"expected_orientations = {t.expected_orientations:.1f}  "
            f"({t.rationale})"
        )
    print()

    # ---- 5. Coordination-engine framing ----
    print("-- Coordination-engine framing --")
    print(
        "  PolicyLab tells you the average lift of a new routing policy.\n"
        "  CausalLab tells you per-context lift under a *given* DAG.\n"
        "  CausalDiscoverer tells you *which DAG* the data actually came from.\n"
        "  Together: route by what causes outcomes, not what merely correlates."
    )


if __name__ == "__main__":
    main()
