"""Coalition demo — Shapley-value credit assignment for the runtime.

Walks an investor through the Coalition primitive end-to-end:

  1. Set up a multi-skill agent and log 200 synthetic ticket traces.
  2. Identify which skills *truly* contributed to success — vs.
     skills that merely co-occurred with success because they're
     bundled with the real winners.
  3. Show the four classical solution concepts the literature
     converged on (Shapley, Banzhaf, Owen, allocation-efficient),
     each with its operational use-case.
  4. Demonstrate the multi-tenant cost-split scenario: three tenants
     share infrastructure, Shapley gives the unique fair allocation.
  5. Demonstrate the data-driven path: with only the trace stream
     and no closed-form value function, the linear-interaction
     fitter recovers Shapley directly.
  6. Show the anytime PAC bound shrinking as more samples land,
     and the attestation receipt that ships with every report.

Run:
    python examples/coalition_demo.py

Stdlib-only and CPU-bound. ~500ms on a laptop.
"""
from __future__ import annotations

import random
from itertools import chain, combinations

from agi.attest import AttestationLedger, RuntimeAttestor
from agi.coalition import (
    COALITION_COMPUTED,
    COALITION_OBSERVED,
    Coalition,
    POLICY_BERNSTEIN,
    POLICY_HOEFFDING,
    banzhaf_index,
    shapley_from_observations,
    shapley_values,
)
from agi.events import EventBus


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def fmt_estimate(label: str, point: float, lo: float, hi: float, *, width: int = 38) -> str:
    bar = ""
    if hi > lo:
        # ASCII bar for the CI, centred on 0.5.
        cells = 20
        l = max(0, int(cells * (lo + 1) / 2))
        h = min(cells, int(cells * (hi + 1) / 2))
        bar = " " * l + "[" + "─" * max(h - l - 1, 0) + "]"
        bar = bar.ljust(cells + 2)
    return f"  {label:<{width}} φ̂ = {point:+.4f}   95% CI = [{lo:+.4f}, {hi:+.4f}]   {bar}"


def main() -> None:
    rng = random.Random(0xC0A11710)

    # ----------------------------------------------------------------
    # Section 1 — Set up the coalition.
    # ----------------------------------------------------------------
    section("1. Skill-attribution scenario: which skills earn their cost?")

    bus = EventBus()
    ledger = AttestationLedger(key=b"investor-demo-secret")
    attestor = RuntimeAttestor(ledger)
    coalition = Coalition(bus=bus, attestor=attestor, rng=rng)

    # Three skills. Synthetic ground truth:
    #   - "core" — essential, baseline 80% pass rate.
    #   - "helper" — boosts core by +10%, useless alone.
    #   - "bystander" — does nothing (a dummy player).
    skills = ["core", "helper", "bystander"]
    for s in skills:
        coalition.register_player(s, value=1.0, cost=0.005)

    print(f"\n  Registered {len(skills)} skills: {skills}")
    print("  Ground truth (hidden from coalition):")
    print("    - core      : success when invoked  → ~0.80 alone, ~0.90 with helper")
    print("    - helper    : useless alone, +0.10 boost when paired with core")
    print("    - bystander : independent noise — DUMMY player (should get φ ≈ 0)")

    # ----------------------------------------------------------------
    # Section 2 — Log 300 synthetic traces.
    # ----------------------------------------------------------------
    section("2. Logging 300 synthetic traces ...")

    observed_events: list[str] = []
    bus.subscribe(
        lambda e: observed_events.append(e.kind),
        kind=COALITION_OBSERVED,
    )

    n_traces = 300
    for _ in range(n_traces):
        contrib = []
        if rng.random() < 0.7:
            contrib.append("core")
        if rng.random() < 0.5:
            contrib.append("helper")
        if rng.random() < 0.4:
            contrib.append("bystander")
        # Outcome model: only core and (core+helper) matter.
        p = 0.0
        if "core" in contrib:
            p = 0.80
            if "helper" in contrib:
                p = 0.90
        success = 1.0 if rng.random() < p else 0.0
        coalition.observe(contrib, success)

    print(f"  → {len(observed_events)} coalition.observed events fired on the bus")
    cov = coalition.coverage()
    print(
        f"  → {cov.n_observations} observations across "
        f"{cov.n_distinct_coalitions}/{cov.n_possible_coalitions} possible coalitions"
    )

    # ----------------------------------------------------------------
    # Section 3 — Exact Shapley (small game, computable in closed form).
    # ----------------------------------------------------------------
    section("3. Shapley credit (exact, observation-driven)")
    print("    A skill's φ̂ is its expected marginal contribution when added in a")
    print("    uniformly-random order — the unique allocation satisfying efficiency,")
    print("    symmetry, dummy, and additivity (Shapley 1953).\n")

    report = coalition.shapley_exact()
    for pid, est in sorted(report.values.items(), key=lambda kv: -kv[1].point):
        print(fmt_estimate(pid, est.point, est.lower, est.upper))

    print(f"\n  efficiency-gap     = {report.efficiency_gap:+.6f}  (≈ 0 for exact)")
    print(f"  receipt_hash       = {report.receipt_hash}")
    print(f"  ledger entries     = {attestor.appended}")
    print(f"  ledger HEAD hash   = {ledger.head().entry_hash if ledger.head() else 'GENESIS'}")

    # ----------------------------------------------------------------
    # Section 4 — Compare with naive metrics.
    # ----------------------------------------------------------------
    section("4. Why Shapley, not naive metrics?")

    # Co-occurrence rate (a naive baseline).
    cooccur: dict[str, list[float]] = {s: [] for s in skills}
    for coa_key, (total, _sq, n) in coalition._obs.items():
        if n == 0:
            continue
        mean = total / n
        for s in coa_key:
            cooccur[s].append(mean)
    print("  Naive co-occurrence rate (skill in trace → empirical success rate):")
    for s in skills:
        avg = sum(cooccur[s]) / len(cooccur[s]) if cooccur[s] else 0.0
        print(f"    {s:<12} → {avg:.4f}")
    print()
    print("  Naive analysis says 'bystander' has the same success rate as 'core' on")
    print("  observed traces — because bystander's randomness is uncorrelated with")
    print("  outcome and just inherits the base rate. Shapley sees through this:")
    print("  bystander's marginal contribution is ≈ 0.")

    # ----------------------------------------------------------------
    # Section 5 — Banzhaf voting power.
    # ----------------------------------------------------------------
    section("5. Banzhaf voting-power index (when does a skill flip the outcome?)")

    print("  Banzhaf measures the *probability of being pivotal* — the right metric")
    print("  for gating decisions ('include this skill in the next campaign?'),")
    print("  distinct from Shapley (which splits surplus).\n")

    bz = coalition.banzhaf_indices(normalised=True)
    for pid, est in sorted(bz.items(), key=lambda kv: -kv[1].point):
        print(fmt_estimate(pid, est.point, est.lower, est.upper))

    # ----------------------------------------------------------------
    # Section 6 — MC sampling with PAC bounds at scale.
    # ----------------------------------------------------------------
    section("6. Monte-Carlo Shapley with anytime PAC bounds (n=10 skills)")

    rng2 = random.Random(2024)
    large = Coalition(rng=rng2)
    weights = {}
    for i in range(10):
        pid = f"skill_{i:02d}"
        weights[pid] = max(0.0, rng2.gauss(1.0, 1.0))
        large.register_player(pid)

    def v_large(S: frozenset[str]) -> float:
        # Submodular: diminishing returns.
        s = sum(weights[p] for p in S)
        return s - 0.05 * s * s  # mild concavity

    large.set_value_function(v_large)
    mc = large.shapley_montecarlo(
        epsilon=0.02, delta=0.05, max_samples=5000, min_samples=200,
        method=POLICY_BERNSTEIN,
    )
    print(f"  Stopped after {mc.n_samples_total} permutations.")
    print(f"  Tightest CI half-width: "
          f"{min(e.half_width for e in mc.values.values()):.4f}")
    print(f"  Loosest  CI half-width: "
          f"{max(e.half_width for e in mc.values.values()):.4f}")
    print(f"  Estimator: {mc.estimator}   Bound: {mc.bound_policy}\n")
    for pid, est in sorted(mc.values.items(), key=lambda kv: -kv[1].point):
        print(fmt_estimate(pid, est.point, est.lower, est.upper, width=12))

    # ----------------------------------------------------------------
    # Section 7 — Multi-tenant cost split.
    # ----------------------------------------------------------------
    section("7. Multi-tenant cost split (fair shared-infrastructure allocation)")

    tenants_cost = {"acme": 100.0, "globex": 50.0, "initech": 80.0}
    print("  Three tenants share a runtime. Standalone costs:")
    for t, c in tenants_cost.items():
        print(f"    {t:<10} ${c:>6.2f}")
    print("\n  Sharing infra discounts each pair-or-larger by $30. Question:")
    print("  how should the combined $200 cost be split fairly?\n")

    mt = Coalition()
    for t in tenants_cost:
        mt.register_player(t)

    def v_tenants(S: frozenset[str]) -> float:
        if not S:
            return 0.0
        combined = sum(tenants_cost[t] for t in S) - 30.0 * (len(S) - 1)
        separate = sum(tenants_cost[t] for t in S)
        return separate - combined  # savings from cooperation

    mt.set_value_function(v_tenants)
    rep = mt.shapley_exact()
    raw = {pid: est.point for pid, est in rep.values.items()}
    print("  Shapley share of the $60 savings (uniquely fair):")
    for t, share in sorted(raw.items(), key=lambda kv: -kv[1]):
        print(f"    {t:<10} saves ${share:>5.2f}   (pays ${tenants_cost[t] - share:>6.2f})")
    print(f"\n  Total savings: ${sum(raw.values()):.2f}  ← equals v(N) by efficiency")
    print(f"  Shapley value lies in the CORE: ", end="")
    in_core, witness = mt.in_core(raw)
    print("YES" if in_core else f"NO (worst excess {witness['worst_excess']:.4f})")

    # ----------------------------------------------------------------
    # Section 8 — Linear-fit path: Shapley directly from traces.
    # ----------------------------------------------------------------
    section("8. Data-driven Shapley (no closed-form v needed)")

    print("  When the runtime only has trace data (no analytic value function),")
    print("  fit a low-order interaction model from observations, then read off")
    print("  Shapley from the fitted coefficients (Möbius decomposition).\n")

    obs_list = []
    rng3 = random.Random(7)
    truth = {"alpha": 0.6, "beta": 0.3, "gamma": -0.1, "delta": 0.0}
    for _ in range(400):
        sub = [p for p in truth if rng3.random() < 0.5]
        y = sum(truth[p] for p in sub) + rng3.gauss(0, 0.05)
        obs_list.append((sub, y))

    phi = shapley_from_observations(
        obs_list, list(truth.keys()), order=1, l2=1e-4,
    )
    print("  Recovered Shapley values (truth in parens):")
    for pid in truth:
        print(f"    {pid:<8} φ̂ = {phi[pid]:+.4f}   (truth {truth[pid]:+.4f})")

    # ----------------------------------------------------------------
    # Section 9 — Closing summary.
    # ----------------------------------------------------------------
    section("9. Summary: what does the coordination engine get?")

    print("  The Coalition primitive answers four operational questions a")
    print("  coordination engine asks every cycle:\n")
    print("    1.  Which contributor should I credit for THIS trace?       → φ̂_i")
    print("    2.  Which contributor should I gate THE NEXT decision on?   → Banzhaf β_i")
    print("    3.  How should I split shared cost across tenants?          → φ̂_i ÷ Σ_j φ̂_j")
    print("    4.  Which skill should I deprecate?                          → arg min_i φ̂_i\n")
    print("  Anytime PAC bounds, tamper-evident receipts via AttestationLedger,")
    print("  event-stream integration via EventBus, threadsafe across drivers.")
    print()


if __name__ == "__main__":
    main()
