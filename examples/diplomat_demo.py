"""Diplomat — end-to-end demo of CFR-family extensive-form game solvers.

Runs every solver in the Diplomat module on three canonical benchmark
games and reports exploitability, root value, and wall time.  No API
key required; everything runs in-process.
"""
from __future__ import annotations

import time

from agi.diplomat import (
    CFRConfig,
    Diplomat,
    KIND_CFR,
    KIND_CFR_PLUS,
    KIND_CHANCE_SAMPLING,
    KIND_DISCOUNTED_CFR,
    KIND_EXTERNAL_SAMPLING,
    KIND_LINEAR_CFR,
    KIND_PREDICTIVE_CFR_PLUS,
    KIND_SEQUENCE_FORM_LP,
    coin_match_with_signal,
    exploitability,
    kuhn_poker,
    matching_pennies_simultaneous,
    rock_paper_scissors,
)


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print("  " + title)
    print("=" * 78)


def show(rep, label: str) -> None:
    print(
        f"  {label:30s}  "
        f"iter={rep.iterations:>5}  "
        f"value(P0)={rep.root_value[0]:+.5f}  "
        f"exploitability={rep.exploitability:.6f}  "
        f"t={rep.wall_seconds:.2f}s"
    )


def main() -> None:
    d = Diplomat()

    banner("Matching pennies (P1 doesn't observe) — Nash value 0, σ = uniform")
    g = matching_pennies_simultaneous()
    show(d.solve(g, CFRConfig(kind=KIND_CFR, iterations=2000)), "CFR (vanilla)")
    g = matching_pennies_simultaneous()
    show(d.solve(g, CFRConfig(kind=KIND_CFR_PLUS, iterations=2000)), "CFR+")
    g = matching_pennies_simultaneous()
    show(d.solve(g, CFRConfig(kind=KIND_PREDICTIVE_CFR_PLUS, iterations=500)),
         "Predictive CFR+ (O(1/T))")
    g = matching_pennies_simultaneous()
    rep = d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))
    show(rep, "Sequence-form LP (exact)")
    print(f"    avg σ P0 = {[round(x,4) for x in rep.average_strategy['P0']]}")
    print(f"    avg σ P1 = {[round(x,4) for x in rep.average_strategy['P1']]}")

    banner("Rock-Paper-Scissors — Nash σ = (1/3, 1/3, 1/3), value 0")
    g = rock_paper_scissors()
    show(d.solve(g, CFRConfig(kind=KIND_CFR_PLUS, iterations=2000)), "CFR+")
    g = rock_paper_scissors()
    rep = d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))
    show(rep, "Sequence-form LP (exact)")
    print(f"    avg σ P0 = {[round(x,4) for x in rep.average_strategy['P0']]}")

    banner("Kuhn poker (Kuhn 1953) — Nash value to dealer = -1/18 ≈ -0.0556")
    print("  Canonical regression test for imperfect-information solvers.")
    print()
    for kind, iters in [
        (KIND_CFR, 5000),
        (KIND_CFR_PLUS, 5000),
        (KIND_LINEAR_CFR, 5000),
        (KIND_DISCOUNTED_CFR, 5000),
        (KIND_PREDICTIVE_CFR_PLUS, 3000),
        (KIND_EXTERNAL_SAMPLING, 5000),
        (KIND_CHANCE_SAMPLING, 5000),
        (KIND_SEQUENCE_FORM_LP, 1),
    ]:
        g = kuhn_poker()
        rep = d.solve(g, CFRConfig(kind=kind, iterations=iters, seed=42))
        show(rep, kind)

    banner("Best-response surgery on Kuhn poker")
    g = kuhn_poker()
    rep = d.solve(g, CFRConfig(kind=KIND_CFR_PLUS, iterations=3000, seed=0))
    print(f"  CFR+ exploitability:    {rep.exploitability:.6f}")
    print(f"  Uniform exploitability: {exploitability(g, d.uniform_strategy(g)):.6f}")
    print()
    print("  Player 1 best-responds to CFR+ player 0 strategy:")
    br = d.best_response(g, rep.average_strategy, player=1)
    print(f"    BR value = {br.value:+.6f}, ΔBR = {br.delta:+.6f}")
    print(f"    Pure BR action at P1|Q|bet: {br.response['P1|Q|bet']}")

    banner("Coin-match with private signal (chance + asymmetric info)")
    print("  Chance flips a coin; only P0 observes it.  P0 chooses an action,")
    print("  P1 chooses without observing.  Solver finds the equilibrium.")
    print()
    g = coin_match_with_signal(p_heads=0.6)
    rep = d.solve(g, CFRConfig(kind=KIND_SEQUENCE_FORM_LP))
    show(rep, "Sequence-form LP")
    print(f"  P0|H = {[round(x,4) for x in rep.average_strategy['P0|H']]}")
    print(f"  P0|T = {[round(x,4) for x in rep.average_strategy['P0|T']]}")
    print(f"  P1   = {[round(x,4) for x in rep.average_strategy['P1']]}")
    print(f"  Certificate fingerprint: {rep.fingerprint()[:16]}...")

    print()
    print("Done.  All solvers shipped: vanilla CFR, CFR+, Linear CFR,")
    print("Discounted CFR, Predictive CFR+, external-sampling MCCFR,")
    print("chance-sampling CFR, and an exact sequence-form LP.")


if __name__ == "__main__":
    main()
