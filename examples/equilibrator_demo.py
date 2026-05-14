"""Equilibrator demo — non-cooperative game-theoretic equilibria as a runtime primitive.

Walks an investor through the six solution concepts a strategically-
aware coordination engine has to be able to switch between on demand:

  1. **Pure Nash equilibrium** — enumerated joint actions where no
     player has a strictly profitable unilateral deviation. The
     dominance-solvable default; missing in generic games.

  2. **Mixed Nash equilibrium** — Nash 1951 existence theorem says
     every finite game has one. We compute it by support
     enumeration (exact, small games) or multiplicative weights
     self-play (approximate, scalable; converges to ε-Nash in
     zero-sum after O(log K / ε²) rounds).

  3. **Minimax / zero-sum value** — von Neumann 1928. The unique
     game value when payoffs are pure conflict. Two-player zero-sum
     LP solved exactly via simplex.

  4. **Correlated equilibrium** — Aumann 1974. A distribution over
     joint actions such that no player can profit by conditional
     deviation. Convex superset of Nash; LP-solvable.

  5. **Coarse correlated equilibrium** — Hannan 1957; what
     independent no-regret learners produce in self-play. Looser
     than CE (only protects against unconditional deviation), but
     reaches `√(log K / T)` exploitability with no LP needed.

  6. **Evolutionarily stable strategy (ESS)** — Maynard Smith &
     Price 1973. A symmetric Nash that is locally asymptotically
     stable under replicator dynamics. The biological / population
     primitive — when self-interested agents drift, ESS profiles
     attract.

The demo runs each concept on a classic game and prints the profile,
exploitability, expected payoffs, and certificate.

Run:
    python examples/equilibrator_demo.py

Stdlib-only and CPU-bound. <500ms on a laptop.
"""
from __future__ import annotations

from agi.attest import AttestationLedger, RuntimeAttestor
from agi.events import Event, EventBus
from agi.equilibrator import (
    CONCEPT_COARSE_CORRELATED,
    CONCEPT_CORRELATED,
    CONCEPT_ESS,
    CONCEPT_MINIMAX,
    CONCEPT_NASH,
    CONCEPT_PURE_NASH,
    Equilibrator,
    METHOD_LINEAR_PROGRAM,
    METHOD_MULTIPLICATIVE_WEIGHTS,
    METHOD_REPLICATOR,
    METHOD_SUPPORT_ENUMERATION,
    Profile,
    Strategy,
    coarse_correlated_equilibrium,
    correlated_equilibrium_lp,
    exploitability,
    multiplicative_weights,
    pure_nash_equilibria,
    support_enumeration_bimatrix,
    zero_sum_value,
)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def fmt_profile(profile: Profile) -> str:
    parts = []
    for p in range(profile.n_players):
        probs = profile[p].probabilities
        parts.append("[" + ", ".join(f"{x:.3f}" for x in probs) + "]")
    return ", ".join(parts)


def fmt_distribution(dist) -> str:
    return ", ".join(
        f"{tuple(j)}={p:.3f}" for j, p in dist if p > 1e-3
    )


def print_report(name: str, report) -> None:
    print(f"\n  {name}:")
    if report.profile is not None:
        print(f"    profile         {fmt_profile(report.profile)}")
    if report.distribution is not None:
        print(f"    distribution    {fmt_distribution(report.distribution)}")
    print(f"    payoffs         ({', '.join(f'{p:+.3f}' for p in report.expected_payoff)})")
    print(f"    exploitability  {report.exploitability:.6f}")
    print(f"    epsilon         {report.epsilon:.6f}")
    print(f"    method/iters    {report.method} / {report.iterations}")
    print(f"    converged       {report.converged}")
    axioms = report.certificate.get("axioms", [])
    if axioms:
        print(f"    axioms          {axioms}")
    if report.value is not None:
        print(f"    game value      {report.value:.6f}")


def main() -> None:
    bus = EventBus()
    ledger = AttestationLedger()
    attestor = RuntimeAttestor(ledger=ledger)
    events = []
    bus.subscribe(lambda e: events.append(e))

    eq = Equilibrator(bus=bus, attestor=attestor, random_seed=0)

    # ---------------------------------------------------------------
    section("1. Prisoners' Dilemma — dominance-solvable pure Nash")
    # ---------------------------------------------------------------
    # Row payoffs: C=cooperate, D=defect
    # P1\P2:  C    D
    #  C  [  3 ,  0 ]
    #  D  [  5 ,  1 ]   ← (D, D) is the unique pure Nash but not Pareto-optimal
    eq.register_game(
        "pd",
        [
            [[3, 0], [5, 1]],
            [[3, 5], [0, 1]],
        ],
        action_names=[("C", "D"), ("C", "D")],
    )
    pure = eq.solve("pd", concept=CONCEPT_PURE_NASH)
    print_report("pure Nash", pure)
    # The (C, C) outcome is socially superior (payoffs 3, 3) but
    # not strategically stable. The Equilibrator certifies this.

    # ---------------------------------------------------------------
    section("2. Matching Pennies — uniqueness of mixed Nash")
    # ---------------------------------------------------------------
    # Zero-sum: P1 wins iff coins match. No pure Nash; unique mixed
    # Nash at (1/2, 1/2) for both players, with game value = 0.
    eq.register_game(
        "mp",
        [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ],
        action_names=[("H", "T"), ("H", "T")],
    )
    se = eq.solve("mp", concept=CONCEPT_NASH, method=METHOD_SUPPORT_ENUMERATION)
    print_report("Nash (support enumeration, exact)", se)

    mw = eq.solve("mp", concept=CONCEPT_NASH,
                  method=METHOD_MULTIPLICATIVE_WEIGHTS,
                  iterations=5_000)
    print_report("Nash (multiplicative weights, 5k iters)", mw)

    minimax = eq.solve("mp", concept=CONCEPT_MINIMAX)
    print_report("Minimax LP", minimax)

    # ---------------------------------------------------------------
    section("3. Rock-Paper-Scissors — 3×3 zero-sum")
    # ---------------------------------------------------------------
    eq.register_game(
        "rps",
        [
            [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
            [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        ],
        action_names=[("R", "P", "S"), ("R", "P", "S")],
    )
    rps_mw = eq.solve("rps", concept=CONCEPT_NASH,
                      method=METHOD_MULTIPLICATIVE_WEIGHTS,
                      iterations=5_000)
    print_report("Nash (MW)", rps_mw)

    rps_minimax = eq.solve("rps", concept=CONCEPT_MINIMAX)
    print_report("Minimax LP", rps_minimax)

    # ---------------------------------------------------------------
    section("4. Battle of the Sexes — multiple Nash, mixed + correlated")
    # ---------------------------------------------------------------
    # Two pure Nash (O,O) and (F,F) plus one mixed Nash. The mixed
    # Nash gives both players strictly less expected payoff than
    # either pure Nash — a coordination failure that correlated
    # equilibrium can resolve.
    eq.register_game(
        "bos",
        [
            [[2, 0], [0, 1]],   # P1: opera, football
            [[1, 0], [0, 2]],
        ],
        action_names=[("O", "F"), ("O", "F")],
    )
    se_bos = eq.solve("bos", concept=CONCEPT_NASH,
                       method=METHOD_SUPPORT_ENUMERATION)
    print_report("Nash (support enumeration, max-welfare)", se_bos)
    # Internal Note: 3 equilibria found — two pure, one mixed.

    cce_bos = eq.solve("bos", concept=CONCEPT_COARSE_CORRELATED,
                       iterations=5_000)
    print_report("Coarse correlated (no-regret self-play)", cce_bos)

    # ---------------------------------------------------------------
    section("5. Chicken — correlated equilibrium beats Nash")
    # ---------------------------------------------------------------
    # Two pure Nash (Hawk, Dove) and (Dove, Hawk) — coordinated
    # asymmetric. The correlated equilibrium spreads mass and
    # achieves higher total welfare.
    eq.register_game(
        "chicken",
        [
            [[0, -1], [1, -10]],
            [[0, 1], [-1, -10]],
        ],
        action_names=[("Dove", "Hawk"), ("Dove", "Hawk")],
    )
    se_ch = eq.solve("chicken", concept=CONCEPT_NASH,
                     method=METHOD_SUPPORT_ENUMERATION)
    print_report("Nash (support enumeration)", se_ch)

    ce_ch = eq.solve("chicken", concept=CONCEPT_CORRELATED,
                     method=METHOD_LINEAR_PROGRAM)
    print_report("Correlated equilibrium (LP, uniform-bias)", ce_ch)

    # ---------------------------------------------------------------
    section("6. Hawk-Dove — symmetric mixed ESS")
    # ---------------------------------------------------------------
    # Classic biology: v=2 (value of resource), c=6 (cost of fight).
    # Mixed ESS at p_hawk = v/c = 1/3.
    V, C = 2.0, 6.0
    eq.register_game(
        "hd",
        [
            [[(V - C) / 2.0, V], [0.0, V / 2.0]],
            [[(V - C) / 2.0, 0.0], [V, V / 2.0]],
        ],
        action_names=[("H", "D"), ("H", "D")],
    )
    ess = eq.solve("hd", concept=CONCEPT_ESS, iterations=3_000)
    print_report("ESS (replicator dynamics)", ess)
    print(f"    theory          p_hawk = V/C = {V/C:.3f}, p_dove = {1-V/C:.3f}")

    # ---------------------------------------------------------------
    section("7. Composability: minimax as Coalition threat-point")
    # ---------------------------------------------------------------
    # Coalition's characteristic function v(S) is "what S guarantees
    # against the rest". For a 2-player zero-sum subgame, that's
    # exactly the Equilibrator's minimax value.
    threat = zero_sum_value(
        [[3, -2], [-1, 4]],
        method=METHOD_LINEAR_PROGRAM,
    )
    print(f"\n  Threat-point computation (2-player zero-sum sub-coalition):")
    print(f"    payoff matrix    [[3, -2], [-1, 4]]")
    print(f"    value            {threat['value']:.6f}")
    print(f"    row optimal      {threat['row_strategy'].probabilities}")
    print(f"    col optimal      {threat['col_strategy'].probabilities}")
    # This value can be fed into Negotiator as a disagreement point
    # for the cooperative bargaining problem on the same parties.

    # ---------------------------------------------------------------
    section("Receipts and the attestation chain")
    # ---------------------------------------------------------------
    print(f"\n  AttestationLedger size:    {len(ledger)}")
    print(f"  Solve events on bus:       {sum(1 for e in events if e.kind == 'equilibrator.solved')}")
    print(f"  Game-register events:      {sum(1 for e in events if e.kind == 'equilibrator.game_registered')}")
    cov = eq.coverage()
    print(f"  Equilibrator coverage:     {cov.n_games} games, {cov.n_solved} solves")
    print(f"  Registered game_ids:       {list(cov.games)}")
    print()


if __name__ == "__main__":
    main()
