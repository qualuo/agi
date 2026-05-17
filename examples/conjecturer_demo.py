"""Conjecturer demo — automated mathematical conjecture generation.

Each scenario shows a different way a coordination engine can use the
``Conjecturer`` primitive to lift raw numerical evidence into a closed-
form integer identity, then verify that identity at higher precision.

Run::

    python -m examples.conjecturer_demo
"""
from __future__ import annotations

import math

from agi.conjecturer import Conjecturer, best_rational, continued_fraction


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def scenario_golden_ratio() -> None:
    banner("Scenario 1 — golden ratio: φ² = φ + 1")
    cj = Conjecturer.create(precision_digits=14, seed=0)
    phi = (1 + math.sqrt(5)) / 2
    cj.observe("phi", phi)
    cj.observe("phi2", phi * phi)
    cj.with_constants(("phi2", "phi", "one"))
    out = cj.propose(max_coeff=3)
    print(f"Observed φ ≈ {phi:.16f}")
    print(f"          φ² ≈ {phi*phi:.16f}")
    print("Top conjectures (by residual):")
    for c in out[:3]:
        print(f"  {c.signature!r:32s}   residual ≤ {float(c.residual):.2e}"
              f"   FDR ≤ {c.fdr_bound:.2e}")
    target = next(c for c in out if c.coeffs == (1, -1, -1)
                  or c.coeffs == (-1, 1, 1))
    v = cj.verify(target, factor=2)
    print(f"  → re-verified at {v.verified_at_digits} digits: "
          f"verified={v.verified}, residual={float(v.residual):.2e}")


def scenario_machin() -> None:
    banner("Scenario 2 — Machin's formula:  π = 16·arctan(1/5) − 4·arctan(1/239)")
    cj = Conjecturer.create(precision_digits=14, seed=0)
    cj.observe("a", math.atan(1 / 5))
    cj.observe("b", math.atan(1 / 239))
    cj.with_constants(("pi", "a", "b"))
    out = cj.propose(max_coeff=20)
    for c in out[:3]:
        print(f"  {c.signature!r:32s}   residual ≤ {float(c.residual):.2e}"
              f"   FDR ≤ {c.fdr_bound:.2e}")


def scenario_zeta() -> None:
    banner("Scenario 3 — Euler's identity:  ζ(2) = π²/6")
    cj = Conjecturer.create(precision_digits=14, seed=0)
    cj.observe("z", math.pi ** 2 / 6)
    cj.with_constants(("z", "zeta2", "one"))
    out = cj.propose(max_coeff=3)
    for c in out[:3]:
        print(f"  {c.signature!r:32s}   residual ≤ {float(c.residual):.2e}")


def scenario_recognize_phi() -> None:
    banner("Scenario 4 — closed-form recognition: φ = (1 + √5)/2")
    cj = Conjecturer.create(precision_digits=14, seed=0)
    phi = (1 + math.sqrt(5)) / 2
    recs = cj.recognize_constant(phi, basis=("one", "sqrt5"))
    for r in recs[:3]:
        print(f"  {r.expression!r:32s}   kind={r.kind!s:10s}"
              f"   residual ≤ {float(r.residual):.2e}")


def scenario_continued_fraction() -> None:
    banner("Scenario 5 — continued fractions: best rational approximations to π")
    cf = continued_fraction(math.pi, max_depth=10, huge_quotient=10 ** 8)
    print(f"  π ≈ [{cf.coefficients[0]}; "
          + ", ".join(str(a) for a in cf.coefficients[1:]) + "]")
    print("  Convergents (best rationals):")
    for p, q in cf.convergents():
        err = abs(math.pi - p / q)
        print(f"    {p}/{q}    error ≈ {err:.2e}")


def scenario_best_rational() -> None:
    banner("Scenario 6 — best rational under denominator budget")
    for D in (10, 100, 1000, 1_000_000):
        r = best_rational(math.pi, D)
        err = abs(math.pi - r)
        print(f"  D ≤ {D:>9d}   best rational = {r.numerator}/{r.denominator}"
              f"   error ≈ {float(err):.2e}")


def scenario_pythagoras_id() -> None:
    banner("Scenario 7 — Pythagorean identity: sin²(θ) + cos²(θ) = 1")
    # Treat θ as a generic angle that the coordinator measured.
    theta = 0.7
    cj = Conjecturer.create(precision_digits=14, seed=0)
    cj.observe("s2", math.sin(theta) ** 2)
    cj.observe("c2", math.cos(theta) ** 2)
    cj.with_constants(("s2", "c2", "one"))
    out = cj.propose(max_coeff=3)
    for c in out[:3]:
        print(f"  {c.signature!r:32s}   residual ≤ {float(c.residual):.2e}")


def scenario_audit_chain() -> None:
    banner("Scenario 8 — audit chain (SHA-256 hash of every step)")
    cj = Conjecturer.create(precision_digits=12, seed=0)
    print(f"  genesis           {cj.head()}")
    cj.observe("x", 1.0)
    print(f"  after observe     {cj.head()}")
    cj.observe("y", 2.0)
    print(f"  after observe     {cj.head()}")
    cj.with_constants(("x", "y", "one"))
    cj.propose(max_coeff=2)
    print(f"  after propose     {cj.head()}")
    rep = cj.report()
    print(f"  report head       {rep.head}")
    print(f"  proposed: {rep.n_proposed}  verified: {rep.n_verified}"
          f"  rejected: {rep.n_rejected}")


def main() -> None:
    scenario_golden_ratio()
    scenario_machin()
    scenario_zeta()
    scenario_recognize_phi()
    scenario_continued_fraction()
    scenario_best_rational()
    scenario_pythagoras_id()
    scenario_audit_chain()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
