"""Solver demo — CDCL SAT as a runtime primitive.

Run with ``python -m examples.solver_demo``.  Demonstrates the four
composable layers exposed by ``agi.solver``:

  1. Bare CNF.
  2. Cardinality constraints (Sinz sequential counter).
  3. The :class:`Formula` DSL with Tseitin encoding.
  4. UNSAT-core MUS extraction and MaxSAT optimisation.
"""

from __future__ import annotations

from agi.solver import (
    Solver,
    STATUS_SAT,
    STATUS_UNSAT,
    at_most,
    exactly,
    land,
    lnot,
    lor,
    var,
    ximp,
)


def demo_cnf() -> None:
    print("--- 1. Bare CNF ---")
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2, -3])
    sv.add_clause([-1, 3])
    sv.add_clause([-2, -3])
    res = sv.solve()
    print(f"  status={res.status}  model={dict(res.model)}")
    print(f"  decisions={res.stats['decisions']} propagations={res.stats['propagations']}")


def demo_unsat_with_proof() -> None:
    print("\n--- 2. UNSAT with DRAT-style proof ---")
    sv = Solver.create(seed=0)
    sv.reserve_vars(20)

    def x(i: int, j: int) -> int:
        return (i - 1) * 4 + j

    # Pigeonhole: 5 pigeons, 4 holes.
    for i in range(1, 6):
        sv.add_clause([x(i, j) for j in range(1, 5)])
    for j in range(1, 5):
        for i1 in range(1, 6):
            for i2 in range(i1 + 1, 6):
                sv.add_clause([-x(i1, j), -x(i2, j)])
    res = sv.solve()
    print(f"  status={res.status}  conflicts={res.stats['conflicts']}  proof_steps={len(res.proof)}")
    print(f"  proof self-check (RUP): {sv.check_proof()}")


def demo_sudoku() -> None:
    print("\n--- 3. 4x4 Sudoku via exactly-1 cardinality ---")
    sv = Solver.create(seed=0)
    sv.reserve_vars(64)

    def x(r: int, c: int, v: int) -> int:
        return ((r - 1) * 4 + (c - 1)) * 4 + v

    for r in (1, 2, 3, 4):
        for c in (1, 2, 3, 4):
            sv.add_exactly([x(r, c, v) for v in (1, 2, 3, 4)], 1)
    for r in (1, 2, 3, 4):
        for v in (1, 2, 3, 4):
            sv.add_exactly([x(r, c, v) for c in (1, 2, 3, 4)], 1)
    for c in (1, 2, 3, 4):
        for v in (1, 2, 3, 4):
            sv.add_exactly([x(r, c, v) for r in (1, 2, 3, 4)], 1)
    for br in (1, 3):
        for bc in (1, 3):
            for v in (1, 2, 3, 4):
                sv.add_exactly(
                    [x(r, c, v) for r in (br, br + 1) for c in (bc, bc + 1)], 1
                )
    sv.assume(x(1, 1, 1))
    sv.assume(x(2, 3, 3))
    res = sv.solve()
    print(f"  status={res.status}")
    grid = [[0] * 4 for _ in range(4)]
    for rr in (1, 2, 3, 4):
        for cc in (1, 2, 3, 4):
            for vv in (1, 2, 3, 4):
                if res.model[x(rr, cc, vv)]:
                    grid[rr - 1][cc - 1] = vv
    for row in grid:
        print("   ", row)


def demo_dsl() -> None:
    print("\n--- 4. Boolean DSL with Tseitin CNF compilation ---")
    sv = Solver.create(seed=0)
    a, b, c = var(1), var(2), var(3)
    # "If a is true and a implies b, then b" — a tautology.  Refute its
    # negation.
    phi = land(a, ximp(a, b), lnot(b))
    sv.add_formula(phi)
    res = sv.solve()
    print(f"  ((a ∧ (a→b)) → b) refuted? status={res.status}")
    # Exactly-2-of-3 mixed with bare clauses.
    sv = Solver.create(seed=0)
    sv.add_formula(exactly(2, [a, b, c]))
    sv.add_formula(lor(lnot(a), b))  # ¬a ∨ b
    res = sv.solve()
    print(f"  exactly-2 ∧ (¬a ∨ b) → model={ {v: res.model[v] for v in (1,2,3)} }")


def demo_mus() -> None:
    print("\n--- 5. Minimal Unsatisfiable Subset over assumptions ---")
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2])
    sv.add_clause([-1, 3])
    sv.add_clause([-2, 3])
    # Assumptions: -3 alone makes the formula UNSAT.  +1 and +2 are
    # extra (the deletion-based MUS algorithm drops them).
    sv.assume(-3)
    sv.assume(1)
    sv.assume(2)
    res = sv.solve()
    print(f"  status={res.status}  initial core={res.core}")
    mus = sv.extract_mus()
    print(f"  MUS = {mus}")


def demo_maxsat() -> None:
    print("\n--- 6. Weighted MaxSAT ---")
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2, 3])
    # Soft clauses prefer "all variables false"; weights bias the cost.
    soft = [[-1], [-2], [-3]]
    weights = [10, 1, 1]
    cost, model, violated = sv.solve_max_sat(soft, weights=weights)
    print(f"  cost={cost}  model={ {v: model[v] for v in (1,2,3)} }  violated_soft={violated}")


def demo_attestation() -> None:
    print("\n--- 7. Attestation ledger (SHA-256 chain) ---")
    sv = Solver.create(seed=0)
    sv.reserve_vars(2)
    sv.add_clause([1, 2])
    sv.assume(1)
    sv.solve()
    rep = sv.report()
    print(f"  num_clauses={rep.num_clauses}  conflicts={rep.conflicts}")
    print(f"  ledger head = {rep.ledger_head}")
    print(f"  ledger length = {len(sv.ledger())} events")


def main() -> None:
    demo_cnf()
    demo_unsat_with_proof()
    demo_sudoku()
    demo_dsl()
    demo_mus()
    demo_maxsat()
    demo_attestation()


if __name__ == "__main__":
    main()
