"""Negotiator demo — multi-party allocation as a runtime primitive.

Walks an investor through the seven solution concepts a real
coordination engine has to be able to switch between on demand:

  1. **Utilitarian** — max-welfare; gives everything to the highest
     marginal-utility party. Picks "the most efficient" but not
     necessarily "the fairest".
  2. **Egalitarian (max-min)** — Rawlsian; equalises the floor across
     parties. The SLA-critical default for multi-tenant runtimes.
  3. **Leximin** — strict Sen-Hammond equity refinement of max-min;
     uses the remaining slack after the bottleneck.
  4. **Nash bargaining** — the unique split that is Pareto-optimal,
     symmetric, IIA-respecting, and affine-invariant. The classical
     "cooperative split."
  5. **Kalai-Smorodinsky** — monotonic; equal proportional progress
     toward each party's ideal. The right pick when ideal points are
     known and monotonicity matters more than IIA.
  6. **Proportional fair** — max Σ log u; the TCP-style answer.
     Coincides with Nash at zero disagreement, gives uniform shares
     for linear utilities.
  7. **VCG** — sealed-bid truthful auction for indivisible items.

For each concept, the demo prints the assignment, the realised
utilities, the welfare, the min-utility, the envy-freeness flag,
and the axiom certificate. The audit trail flows through an
AttestationLedger.

Run:
    python examples/negotiator_demo.py

Stdlib-only and CPU-bound. ~200ms on a laptop.
"""
from __future__ import annotations

from agi.attest import AttestationLedger, RuntimeAttestor
from agi.events import Event, EventBus
from agi.negotiator import (
    AXIOM_AFFINE_INVARIANCE,
    AXIOM_ENVY_FREE,
    AXIOM_IIA,
    AXIOM_MONOTONICITY,
    LinearUtility,
    Negotiator,
    NegotiationInfeasible,
    PiecewiseLinearUtility,
    QuadraticUtility,
)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_alloc(name: str, report) -> None:
    parts = ", ".join(
        f"{pid}={x:.3f}"
        for pid, x in sorted(report.allocation.assignments.items())
    )
    utils = ", ".join(
        f"u({pid})={u:.3f}"
        for pid, u in sorted(report.allocation.utilities.items())
    )
    cert = ", ".join(report.certificate)
    print(f"  {name:>22}  {parts}")
    print(f"  {'utilities':>22}  {utils}")
    print(f"  {'welfare / min_u':>22}  W={report.welfare:.3f}  "
          f"min_u={report.min_utility:.3f}  "
          f"nash_product={report.nash_product:.3f}")
    print(f"  {'envy-free?':>22}  {report.envy.envy_free}  "
          f"(max_envy={report.envy.max_envy:.3f})")
    print(f"  {'axioms':>22}  {cert}")
    if report.receipt_hash:
        print(f"  {'receipt':>22}  {report.receipt_hash[:24]}…")
    print()


def main() -> None:
    section("Setup: three tenants competing for 12 units of capacity")
    print("""
  Premium  : LinearUtility(slope=3.0, cap=10.0)
             High-paying tenant; high marginal value, big cap.

  Standard : LinearUtility(slope=1.5, cap=10.0)
             Standard SLA; medium marginal value.

  Economy  : LinearUtility(slope=0.5, cap=10.0)
             Cost-sensitive tier; low marginal value.

  Total budget: 12 units of capacity (e.g. GPU-seconds, requests/s, tokens/min)
""")

    bus = EventBus()
    bus.subscribe(lambda e: None)  # silent subscription for demo timing
    ledger = AttestationLedger()
    attestor = RuntimeAttestor(ledger=ledger)
    neg = Negotiator(bus=bus, attestor=attestor)
    neg.register_party("premium", LinearUtility(slope=3.0, cap=10.0))
    neg.register_party("standard", LinearUtility(slope=1.5, cap=10.0))
    neg.register_party("economy", LinearUtility(slope=0.5, cap=10.0))
    neg.set_budget(12.0)

    section("Concept 1 — Utilitarian (max-welfare)")
    print("\n  Maximises Σ u_i. Pareto-optimal. Not envy-free in general.\n")
    r_util = neg.allocate_utilitarian()
    print_alloc("utilitarian", r_util)
    print("  Interpretation: premium fills its cap (highest slope), standard")
    print("  takes the rest, economy gets zero. Most efficient — least fair.")

    section("Concept 2 — Egalitarian (max-min)")
    print("\n  Maximises min_i u_i. SLA floor for every tenant.\n")
    r_egal = neg.allocate_egalitarian()
    print_alloc("egalitarian", r_egal)
    print("  Interpretation: every tier gets the same realised utility.")
    print("  Economy gets the largest share because its slope is lowest.")

    section("Concept 3 — Leximin")
    print("\n  Iterates max-min; uses slack after the bottleneck.\n")
    r_lex = neg.allocate_leximin()
    print_alloc("leximin", r_lex)

    section("Concept 4 — Nash bargaining")
    print("\n  Pareto + symmetry + IIA + affine-invariance.\n")
    r_nash = neg.allocate_nash()
    print_alloc("nash", r_nash)
    print(f"  IIA in certificate: {AXIOM_IIA in r_nash.certificate}")
    print(f"  Affine-invariance in certificate: "
          f"{AXIOM_AFFINE_INVARIANCE in r_nash.certificate}")

    section("Concept 5 — Kalai-Smorodinsky")
    print("\n  Equal proportional progress toward ideal.\n")
    r_ks = neg.allocate_kalai_smorodinsky()
    print_alloc("kalai_smorodinsky", r_ks)
    print(f"  Monotonicity in certificate: "
          f"{AXIOM_MONOTONICITY in r_ks.certificate}")

    section("Concept 6 — Proportional fair")
    print("\n  Max Σ log u — the TCP-style network share.\n")
    r_pf = neg.allocate_proportional_fair()
    print_alloc("proportional_fair", r_pf)

    section("Concept 7 — VCG auction for indivisible GPU rentals")
    print("""
  Three tenants bid on two exclusive GPU rentals (truthfully).
  VCG charges each winner the externality they impose on the others.
""")
    vcg = neg.vcg_auction(
        items=("GPU-A100-east", "GPU-A100-west"),
        bids={
            "premium":  {"GPU-A100-east": 15.0, "GPU-A100-west": 12.0},
            "standard": {"GPU-A100-east":  8.0, "GPU-A100-west": 10.0},
            "economy":  {"GPU-A100-east":  3.0, "GPU-A100-west":  4.0},
        },
    )
    print(f"  Winners:    {vcg.winners}")
    print(f"  Bundles:    "
          + ", ".join(f"{w}: {list(items)}" for w, items in vcg.bundle.items()))
    print(f"  Payments:   {vcg.payments}")
    print(f"  Welfare:    {vcg.welfare:.3f}")
    if vcg.receipt_hash:
        print(f"  Receipt:    {vcg.receipt_hash[:24]}…")
    print("""
  Interpretation: premium wins both GPUs (highest valuations); each
  payment equals the welfare the rest of the world *would have* won
  had premium not bid — the dominant-strategy-truthful price.""")

    section("Scenario A — refund-pool split under Nash bargaining")
    print("""
  Three tenants each had a SLO breach this billing period. The
  economist has a $5,000 refund pool to distribute. Each tenant's
  disagreement point is its current SLA-floor refund; the Negotiator
  splits the pool fairly above that floor.
""")
    refund = Negotiator()
    # Utilities here represent "value of the refund" — tenants with
    # higher SLA stakes have steeper utility curves.
    refund.register_party(
        "tenant-a", LinearUtility(slope=2.0, cap=4000.0),
        disagreement=500.0,
    )
    refund.register_party(
        "tenant-b", LinearUtility(slope=1.0, cap=4000.0),
        disagreement=300.0,
    )
    refund.register_party(
        "tenant-c", LinearUtility(slope=0.8, cap=4000.0),
        disagreement=100.0,
    )
    refund.set_budget(5000.0)
    r_refund = refund.allocate_nash()
    print_alloc("nash refund-split", r_refund)

    section("Scenario B — premium-floor SLA via leximin with weights")
    print("""
  Same capacity problem, but premium tenants are weighted 2× —
  the leximin floor is u/w-equal, so premium gets 2× the realised
  utility of the floor tier.
""")
    weighted = Negotiator()
    weighted.register_party(
        "premium", LinearUtility(slope=1.0, cap=10.0), weight=2.0,
    )
    weighted.register_party(
        "standard", LinearUtility(slope=1.0, cap=10.0), weight=1.0,
    )
    weighted.register_party(
        "economy", LinearUtility(slope=1.0, cap=10.0), weight=1.0,
    )
    weighted.set_budget(12.0)
    r_weighted = weighted.allocate_leximin()
    print_alloc("weighted leximin", r_weighted)

    section("Scenario C — quadratic utilities (saturation, decreasing returns)")
    print("""
  Most real workloads have decreasing marginal value past a knee
  — beyond that, extra capacity yields diminishing utility. The
  Negotiator handles QuadraticUtility natively via the KKT
  water-filling solver.
""")
    qneg = Negotiator()
    qneg.register_party("a", QuadraticUtility(a=10.0, b=2.0))  # cap=5
    qneg.register_party("b", QuadraticUtility(a=8.0, b=1.0))   # cap=8
    qneg.register_party("c", QuadraticUtility(a=6.0, b=0.5))   # cap=12
    qneg.set_budget(15.0)
    r_q = qneg.allocate_utilitarian()
    print_alloc("utilitarian (quadratic)", r_q)
    print("  Each party's allocation x satisfies u_i'(x) = λ at the optimum.")

    section("Scenario D — piecewise-linear utility (tiered SaaS pricing)")
    print("""
  Many real utilities are piecewise-linear: cheap up to a
  contracted limit, expensive after. The Negotiator handles this
  natively via PiecewiseLinearUtility.
""")
    pneg = Negotiator()
    pneg.register_party(
        "saas",
        PiecewiseLinearUtility(
            breakpoints=((0.0, 0.0), (5.0, 10.0), (10.0, 13.0)),
        ),
    )
    pneg.register_party(
        "saas-2",
        PiecewiseLinearUtility(
            breakpoints=((0.0, 0.0), (3.0, 6.0), (8.0, 9.0)),
        ),
    )
    pneg.set_budget(8.0)
    r_pl = pneg.allocate_utilitarian()
    print_alloc("utilitarian (piecewise)", r_pl)

    section("Wrap-up")
    print("""
  Every concept above ships:
    • the assignment ‹party → share›
    • the realised utilities ‹party → u_i(x_i)›
    • the axiom certificate (which axioms the allocation provably
      satisfies)
    • a pairwise envy diagnostic
    • a Pareto-dominance probe
    • a tamper-evident SHA-256 receipt via AttestationLedger

  The coordination engine wires Negotiator to:
    – TicketMarket  → which tickets dispatch when capacity tightens
    – TicketEconomist → fair refund-pool split
    – PortfolioOptimizer → pick the operating point on a Pareto curve
    – Coalition → use Shapley values as disagreement points
    – Auditor / RiskController → allocate α-budget across tests
""")
    # Show that the ledger captured every report.
    print(f"  Attestation ledger entries: {len(ledger)}")
    print()


if __name__ == "__main__":
    main()
