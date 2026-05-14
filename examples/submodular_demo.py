"""Submodular demo — subset selection with provable approximation bounds.

Scenario
--------

A coordination engine has to make a constant stream of *subset*
decisions:

  1. Pick the K best demonstrations to include in a prompt under a
     token budget.
  2. Pick the K most diverse-and-skillful sensors / probes to wake
     under a power budget.
  3. Pick the K most informative experiments to run from a backlog
     of N candidates.
  4. Pick the K agents to dispatch from a pool that maximise
     marginal coverage of a customer's intent space.

Every one of these is **NP-hard** as a generic combinatorial search
— but every one of them has a submodular utility (diminishing
returns) and is exactly the regime where greedy algorithms have
*tight* approximation guarantees.

This demo walks an investor through five realistic uses of the
``agi.submodular.Submodular`` primitive end-to-end:

  1. **Diverse demonstration selection** (facility-location utility) —
     pick the K most representative few-shot exemplars.
  2. **Sensor / tool placement** (weighted set-cover) — cover the
     query space with the fewest tools.
  3. **DPP-style diverse sampling** (log-determinant) — diversity
     + quality on a kernel of skill embeddings.
  4. **Non-monotone max-cut summarisation** (unconstrained,
     random and deterministic double greedy).
  5. **Knapsack-constrained portfolio** (Sviridenko 2004) — select
     under a hard $ budget with per-element costs.

Plus:

  * **Curvature** computation and the matching Conforti-Cornuéjols
    bound, showing the bound *tightens* on near-modular utilities.
  * **Submodularity certificate** with anytime PAC Hoeffding bound
    on the diminishing-returns violation rate.

Run:
    python -m examples.submodular_demo

Honest about limits
-------------------

The ``(1 - 1/e) ≈ 0.632`` bound is over the *combinatorial optimum*
of an oracle-defined submodular function.  Always run
``certify_submodular(...)`` first if your utility was hand-built —
the bound vanishes for non-submodular inputs.

For non-monotone problems (max-cut, max-bisection), drive with
``METHOD_DOUBLE_GREEDY_RANDOM`` (½) or
``METHOD_DOUBLE_GREEDY_DETERMINISTIC`` (⅓).

For streaming inputs that don't fit in memory, drive with
``stream(...)`` (Sieve-Streaming, ``½ - ε`` in one pass).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

# Allow running from the repo root without `pip install -e .` having taken effect.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.events import Event, EventBus
from agi.submodular import (
    ConcaveOverModular,
    FacilityLocation,
    LogDeterminant,
    METHOD_DOUBLE_GREEDY_DETERMINISTIC,
    METHOD_DOUBLE_GREEDY_RANDOM,
    METHOD_LAZY_GREEDY,
    METHOD_SIEVE_STREAMING,
    METHOD_STOCHASTIC_GREEDY,
    METHOD_SVIRIDENKO_KNAPSACK,
    MaxCut,
    MonotoneSetCover,
    Submodular,
    WeightedCoverage,
)


ONE_MINUS_INV_E = 1.0 - math.exp(-1.0)


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def step1_demonstration_selection() -> None:
    """Pick the K most representative demonstrations from a pool of 30.

    Facility-location utility: f(S) = Σ_demos max_{ex ∈ S} sim(demo, ex).
    """
    banner("1. Diverse demonstration selection (facility location, K=4)")

    rng = random.Random(0)
    n_demos = 30
    # 5 latent "skills"; each demo lies near one of them.
    skills = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(5)]
    points = []
    for _ in range(n_demos):
        cx, cy = rng.choice(skills)
        points.append((cx + rng.gauss(0, 0.3), cy + rng.gauss(0, 0.3)))

    def sim(p, q):
        return math.exp(-((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2))

    # f(S) maximises *coverage of the query space by the chosen demos*.
    # Queries are the same pool: which K demos best represent the others?
    n = n_demos
    W = [[sim(points[i], points[j]) for j in range(n)] for i in range(n)]
    facility = FacilityLocation(W)
    sm = Submodular(random_seed=0)
    rep = sm.maximize(facility, facility.ground_set(), k=4, method=METHOD_LAZY_GREEDY)
    print(f"  picked: {rep.selected}")
    print(f"  f(S)            = {rep.value:.4f}")
    print(f"  upper bound     = {rep.upper_bound:.4f}  (sum of top-k singletons)")
    print(f"  realised ratio  = {rep.value / rep.upper_bound:.4f}")
    print(f"  worst-case bound = 1 - 1/e = {ONE_MINUS_INV_E:.4f}")
    print(f"  oracle calls    = {rep.n_oracle_calls}")

    # Stochastic greedy — drastically cheaper on huge ground sets,
    # 50 runs for an empirical-Bernstein band.
    vals = [
        sm.maximize(
            facility,
            facility.ground_set(),
            k=4,
            method=METHOD_STOCHASTIC_GREEDY,
            epsilon=0.1,
            seed=s,
        ).value
        for s in range(50)
    ]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    print(
        f"  stochastic_greedy  : mean = {mean:.4f}  ± {math.sqrt(var):.4f}  (50 seeds)"
    )


def step2_tool_selection() -> None:
    """Cover the query space with the fewest tools (set-cover).

    Submodular cover: minimise |S| s.t. f(S) ≥ Q.
    """
    banner("2. Tool / sensor selection (weighted set-cover)")

    # Each "tool" handles a subset of intents.
    tool_capabilities = [
        {"summarise", "extract"},
        {"summarise", "translate"},
        {"code", "debug"},
        {"code", "test"},
        {"search", "extract"},
        {"plan", "schedule"},
        {"math", "stats"},
        {"draft", "edit"},
        {"draft", "code"},
        {"search", "summarise"},
    ]
    f = MonotoneSetCover(tool_capabilities)
    sm = Submodular(random_seed=0)
    # Cover at least 9 distinct intents (out of the 13-element universe).
    rep = sm.cover(f, f.ground_set(), quota=9.0)
    print(f"  tools picked: {rep.selected}")
    print(f"  intents covered: {rep.value:.0f} / {len(f._inner.universe)}")
    print(
        f"  greedy-cover bound on |Ŝ| / |S*|: {rep.approx_ratio:.4f}  (Wolsey 1982)"
    )


def step3_dpp_diverse_quality() -> None:
    """Pick K agents that are both *good* and *complementary*.

    Each agent has a per-intent skill vector; the kernel ``K`` mixes
    quality (diagonal) with diversity (off-diagonal).
    """
    banner("3. DPP-style diverse-and-skillful pick (log-determinant, K=3)")

    rng = random.Random(7)
    agents = ["A", "B", "C", "D", "E", "F"]
    n = len(agents)
    # Quality per agent.
    q = [0.9, 0.8, 0.7, 0.85, 0.6, 0.95]
    # Similarity between agents.
    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        S[i][i] = 1.0
        for j in range(i + 1, n):
            S[i][j] = S[j][i] = 0.2 + 0.5 * rng.random()
    # Quality-similarity kernel K_ij = q_i q_j S_ij  (Kulesza-Taskar).
    K = [[q[i] * q[j] * S[i][j] for j in range(n)] for i in range(n)]
    f = LogDeterminant(K, alpha=1.0)  # large alpha → monotone regime
    sm = Submodular(random_seed=0)
    rep = sm.maximize(f, f.ground_set(), k=3, method=METHOD_LAZY_GREEDY)
    picked_names = [agents[i] for i in rep.selected]
    print(f"  picked agents: {picked_names}  (qualities: {[q[i] for i in rep.selected]})")
    print(f"  log-det = {rep.value:.4f}")
    print(f"  oracle calls = {rep.n_oracle_calls}  (lazy-greedy short-circuit)")


def step4_non_monotone_summarisation() -> None:
    """Pick a max-cut subset of a similarity graph.

    Non-monotone; unconstrained.  Drive with double-greedy.
    """
    banner("4. Non-monotone summarisation (max-cut, double greedy)")

    rng = random.Random(0)
    n = 12
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            W[i][j] = W[j][i] = max(0.0, rng.gauss(0.5, 0.2))
    f = MaxCut(W)
    sm = Submodular(random_seed=0)
    rep_r = sm.maximize(
        f,
        f.ground_set(),
        method=METHOD_DOUBLE_GREEDY_RANDOM,
        monotone=False,
        seed=0,
    )
    rep_d = sm.maximize(
        f, f.ground_set(), method=METHOD_DOUBLE_GREEDY_DETERMINISTIC, monotone=False
    )
    print(f"  randomised double greedy: |S| = {len(rep_r.selected)}, f = {rep_r.value:.4f}")
    print(
        f"     worst-case bound = ½  → guaranteed f ≥ ½ · OPT (Buchbinder et al. 2015)"
    )
    print(
        f"  deterministic double greedy: |S| = {len(rep_d.selected)}, f = {rep_d.value:.4f}"
    )
    print(f"     worst-case bound = ⅓")


def step5_knapsack_portfolio() -> None:
    """Pick subsets of experiments under a hard $ budget.

    Sviridenko 2004: partial enumeration + cost-benefit greedy.
    """
    banner("5. Budget-constrained experiment portfolio (Sviridenko 2004 knapsack)")

    # 8 candidate experiments; utility = expected information gain (mocked
    # as a submodular set-cover utility), cost = $ to run.
    coverage = [
        {"a", "b"},
        {"b", "c"},
        {"a", "c", "d"},
        {"e", "f"},
        {"d", "e"},
        {"f", "g", "h"},
        {"a", "h"},
        {"c", "e", "g"},
    ]
    costs = [1.0, 0.5, 1.5, 0.8, 0.6, 1.2, 0.4, 1.1]
    f = WeightedCoverage(coverage)
    sm = Submodular(random_seed=0)
    budget = 2.5
    rep = sm.maximize(
        f,
        f.ground_set(),
        method=METHOD_SVIRIDENKO_KNAPSACK,
        budget=budget,
        costs=costs,
        enum_size=2,
    )
    spent = sum(costs[i] for i in rep.selected)
    print(f"  picked experiments: {rep.selected}")
    print(f"  cost  spent       : ${spent:.2f}  (budget ${budget:.2f})")
    print(f"  intents covered   : {rep.value:.0f}")
    print(f"  approx bound      : (1 - 1/e) = {ONE_MINUS_INV_E:.4f}  (enum_size=3 needed)")
    print(f"  oracle calls      : {rep.n_oracle_calls}")


def step6_curvature_and_certificate() -> None:
    """Curvature-aware bound + submodularity certificate."""
    banner("6. Curvature bound + submodularity certificate")

    # Near-modular: each set contributes mostly disjoint mass → curvature small.
    near_modular = [
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12},
        {13, 14, 15},
    ]
    # Hard set-cover: heavy overlap → curvature near 1.
    hard = [
        {1, 2, 3, 4, 5},
        {1, 2, 6},
        {3, 7},
        {4, 8},
        {5, 9},
    ]
    sm = Submodular(random_seed=0)
    for name, sets in [("near-modular", near_modular), ("hard overlap", hard)]:
        f = WeightedCoverage(sets)
        c = sm.curvature(f, f.ground_set())
        b = sm.curvature_bound(c, k=3)
        print(f"  {name:13s} : curvature = {c:.3f}  →  bound at k=3 = {b:.4f}")

    # Certificate of submodularity on the WeightedCoverage utility.
    f = WeightedCoverage(near_modular)
    cert = sm.certify_submodular(f, f.ground_set(), n_samples=300, alpha=0.05)
    print(
        f"\n  submodularity check (300 samples, α=0.05):"
        f"\n    empirical violation rate = {cert.violation_rate:.3f}"
        f"\n    Hoeffding 95% upper      = {cert.hoeffding_upper:.4f}"
        f"\n    empirical-Bernstein 95% upper = {cert.bernstein_upper:.4f}"
    )

    # A non-submodular function for contrast.
    def supermod(S):
        return float(len(set(S))) ** 2

    cert_bad = sm.certify_submodular(supermod, [0, 1, 2, 3, 4], n_samples=200, alpha=0.05)
    print(
        f"\n  contrast on supermodular f(S) = |S|²:"
        f"\n    empirical violation rate = {cert_bad.violation_rate:.3f}"
        f"\n    Hoeffding 95% upper      = {cert_bad.hoeffding_upper:.4f}  ← rejects"
    )


def step7_event_bus() -> None:
    """Show the runtime emitting submodular events for a coordination engine."""
    banner("7. Coordination-engine integration via EventBus")

    bus = EventBus()
    captured: list[Event] = []
    bus.subscribe(captured.append, kind="submodular.solved")
    sm = Submodular(bus=bus)
    sets = [{1, 2, 3}, {3, 4}, {1, 4, 5}, {2, 5, 6}]
    f = WeightedCoverage(sets)
    sm.maximize(f, f.ground_set(), k=2, method=METHOD_LAZY_GREEDY)
    sm.maximize(f, f.ground_set(), k=3, method=METHOD_LAZY_GREEDY)
    print(f"  bus received {len(captured)} 'submodular.solved' events")
    for e in captured:
        print(
            f"    digest={e.data['digest'][:12]}  value={e.data['value']:.2f}  "
            f"selected={e.data['selected']}"
        )


def main() -> None:
    step1_demonstration_selection()
    step2_tool_selection()
    step3_dpp_diverse_quality()
    step4_non_monotone_summarisation()
    step5_knapsack_portfolio()
    step6_curvature_and_certificate()
    step7_event_bus()

    banner("Summary")
    print(
        """
The Submodular primitive ships:

  * lazy_greedy / naive_greedy / celf / threshold_greedy   — (1 - 1/e)
  * stochastic_greedy                                       — (1 - 1/e - ε)
  * cost_greedy / sviridenko_knapsack                       — (1 - 1/e) under knapsack
  * double_greedy_random / double_greedy_deterministic      — ½ / ⅓ unconstrained
  * distorted_greedy                                        — γ · (1 - 1/e^γ) weakly-submod
  * sieve_streaming                                         — (½ - ε) one pass
  * submodular_cover (Wolsey)                               — (1 + ln(Q/η)) on |Ŝ|

…with a content-hashed receipt per solve, an EventBus stream, and a
PAC certificate of submodularity on demand.

Composes with: Cartographer (next-K frontier tasks), ExperimentDesigner
(batch BOED), Coalition (Shapley-baseline credit), Negotiator (subset
allocation), Auditor (top-K significant findings), PolicyLab /
PolicyImprover (diverse counterfactual policy bank), Strategist
(portfolio EV), Forecaster (ensemble pick), Skills (top-K retrieval),
AttestationLedger (replayable receipts).
"""
    )


if __name__ == "__main__":
    main()
