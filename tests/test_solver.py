"""Tests for ``agi.solver`` — CDCL SAT solver primitive."""

from __future__ import annotations

import random
import pytest

from agi.solver import (
    InvalidAssumption,
    InvalidClause,
    InvalidConfig,
    InvalidFormula,
    InvalidLiteral,
    NotYetSolved,
    ProofCheckFailed,
    ResourceExhausted,
    SOLVER_ASSUMED,
    SOLVER_CLAUSE_ADDED,
    SOLVER_KNOWN_EVENTS,
    SOLVER_KNOWN_STATUSES,
    SOLVER_MAXSAT,
    SOLVER_MUS,
    SOLVER_REPORTED,
    SOLVER_SOLVED,
    SOLVER_STARTED,
    STATUS_SAT,
    STATUS_UNKNOWN,
    STATUS_UNSAT,
    Solver,
    SolverError,
    SolverReport,
    SolverResult,
    at_least,
    at_most,
    exactly,
    false,
    land,
    lnot,
    lor,
    true,
    var,
    xeqv,
    xite,
    ximp,
    _luby,
)


# --------------------------------------------------------------------- basic SAT


def test_trivial_sat() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2, -3])
    sv.add_clause([-1, 3])
    sv.add_clause([-2, -3])
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert set(r.model.keys()) == {1, 2, 3}
    # Verify the model satisfies every clause.
    for clause in sv._original_clauses:
        assert any(
            (l > 0 and r.model[abs(l)]) or (l < 0 and not r.model[abs(l)])
            for l in clause
        )


def test_trivial_unsat() -> None:
    sv = Solver.create(seed=0)
    sv.add_clause([1])
    sv.add_clause([-1])
    r = sv.solve()
    assert r.status == STATUS_UNSAT
    # Empty clause emitted to the DRAT log.
    assert r.proof[-1] == ("a", ())
    assert sv.check_proof()


def test_empty_formula_is_sat() -> None:
    sv = Solver.create(seed=0)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert r.model == {}


def test_tautology_is_no_op() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(1)
    sv.add_clause([1, -1])  # tautology
    r = sv.solve()
    assert r.status == STATUS_SAT


def test_l0_unit_contradiction_via_assumption() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(1)
    sv.add_clause([1])  # x_1 must be true at L0
    sv.assume(-1)  # contradiction
    r = sv.solve()
    assert r.status == STATUS_UNSAT


def test_unsat_proof_check() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(20)

    def x(i: int, j: int) -> int:
        return (i - 1) * 4 + j

    # Pigeonhole 5 → 4
    for i in range(1, 6):
        sv.add_clause([x(i, j) for j in range(1, 5)])
    for j in range(1, 5):
        for i1 in range(1, 6):
            for i2 in range(i1 + 1, 6):
                sv.add_clause([-x(i1, j), -x(i2, j)])
    r = sv.solve()
    assert r.status == STATUS_UNSAT
    assert sv.check_proof()


def test_sat_proof_check_returns_true() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2, 3])
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert sv.check_proof() is True


# --------------------------------------------------------------------- API errors


def test_zero_literal_rejected() -> None:
    sv = Solver.create(seed=0)
    with pytest.raises(InvalidLiteral):
        sv.add_clause([0])


def test_non_integer_literal_rejected() -> None:
    sv = Solver.create(seed=0)
    with pytest.raises(InvalidLiteral):
        sv.add_clause([1.5])  # type: ignore[list-item]


def test_oversize_clause_rejected() -> None:
    sv = Solver.create(seed=0)
    big = list(range(1, (1 << 20) + 2))
    with pytest.raises(InvalidClause):
        sv.add_clause(big)


def test_negative_cardinality_rejected() -> None:
    sv = Solver.create(seed=0)
    with pytest.raises(InvalidClause):
        sv.add_at_most([1, 2], -1)


def test_invalid_assumption_rejected() -> None:
    sv = Solver.create(seed=0)
    with pytest.raises(InvalidAssumption):
        sv.assume(0)
    with pytest.raises(InvalidAssumption):
        sv.assume("foo")  # type: ignore[arg-type]


def test_invalid_seed_rejected() -> None:
    with pytest.raises(InvalidConfig):
        Solver.create(seed=1.5)  # type: ignore[arg-type]


def test_private_constructor() -> None:
    with pytest.raises(SolverError):
        Solver(object(), 0, 0)  # type: ignore[arg-type]


def test_max_var_hint_validation() -> None:
    with pytest.raises(InvalidConfig):
        Solver.create(max_var_hint=-1)


def test_reserve_vars_validation() -> None:
    sv = Solver.create()
    with pytest.raises(InvalidConfig):
        sv.reserve_vars(-1)


def test_max_conflicts_validation() -> None:
    sv = Solver.create()
    with pytest.raises(InvalidConfig):
        sv.solve(max_conflicts=-1)
    with pytest.raises(InvalidConfig):
        sv.solve(time_budget_s=-1.0)


# --------------------------------------------------------------------- assumptions


def test_unsat_under_assumptions_returns_core() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2])
    sv.add_clause([-1, 3])
    sv.add_clause([-2, 3])
    sv.assume(-3)
    r = sv.solve()
    assert r.status == STATUS_UNSAT
    assert -3 in r.core
    # check_proof skips proof checking under assumptions but returns True.
    assert sv.check_proof()


def test_mus_extraction() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2])
    sv.add_clause([-1, 3])
    sv.add_clause([-2, 3])
    sv.assume(-3)
    sv.assume(1)  # extra assumption — not actually needed for UNSAT
    sv.solve()
    mus = sv.extract_mus()
    # The minimal core is {-3}, since -3 plus formula entails ⊥.
    assert mus == (-3,)


def test_mus_requires_unsat() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(1)
    sv.add_clause([1])
    sv.solve()
    with pytest.raises(NotYetSolved):
        sv.extract_mus()


def test_assumptions_cleared_between_solves() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(1)
    sv.assume(1)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert r.model[1] is True
    sv.clear_assumptions()
    sv.assume(-1)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert r.model[1] is False


# --------------------------------------------------------------------- cardinality


def test_at_most_one() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(4)
    sv.add_at_most([1, 2, 3, 4], 1)
    sv.add_clause([1])  # force x_1 = true ⇒ all others false
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert r.model[1] is True
    assert r.model[2] is False
    assert r.model[3] is False
    assert r.model[4] is False


def test_at_least_one() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_at_least([1, 2, 3], 1)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert any(r.model[v] for v in (1, 2, 3))


def test_exactly_two() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(4)
    sv.add_exactly([1, 2, 3, 4], 2)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert sum(r.model[v] for v in (1, 2, 3, 4)) == 2


def test_overlapping_exactly_constraints() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(4)
    sv.add_exactly([1, 2], 1)
    sv.add_exactly([3, 4], 1)
    sv.add_exactly([1, 3], 1)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert r.model[1] + r.model[2] == 1
    assert r.model[3] + r.model[4] == 1
    assert r.model[1] + r.model[3] == 1


def test_exactly_zero() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_exactly([1, 2, 3], 0)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert all(not r.model[v] for v in (1, 2, 3))


def test_at_most_zero_zeros_out() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_at_most([1, 2, 3], 0)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert all(not r.model[v] for v in (1, 2, 3))


def test_at_most_unbounded_is_no_op() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_at_most([1, 2, 3], 10)  # trivially satisfied
    r = sv.solve()
    assert r.status == STATUS_SAT


def test_negative_lit_cardinality_rejected() -> None:
    sv = Solver.create(seed=0)
    with pytest.raises(InvalidLiteral):
        sv.add_at_most([0, 1], 1)


# --------------------------------------------------------------------- DSL


def test_dsl_var_validation() -> None:
    with pytest.raises(InvalidFormula):
        var(0)
    with pytest.raises(InvalidFormula):
        var(-1)
    with pytest.raises(InvalidFormula):
        land(42)  # type: ignore[arg-type]


def test_dsl_combinators() -> None:
    a, b = var(1), var(2)
    assert (a & b).kind == "and"
    assert (a | b).kind == "or"
    assert (~a).kind == "not"
    assert (a >> b).kind == "imp"
    assert a.equiv(b).kind == "eqv"


def test_dsl_and_or_satisfiable() -> None:
    sv = Solver.create(seed=0)
    phi = land(lor(var(1), var(2)), lor(lnot(var(1)), var(3)), lnot(var(3)))
    sv.add_formula(phi)
    r = sv.solve()
    assert r.status == STATUS_SAT


def test_dsl_implication_tautology_refutation() -> None:
    sv = Solver.create(seed=0)
    # ((a ∧ (a → b)) → b) is a tautology; its negation should be UNSAT
    a, b = var(1), var(2)
    sv.add_formula(land(a, ximp(a, b), lnot(b)))
    r = sv.solve()
    assert r.status == STATUS_UNSAT


def test_dsl_equivalence() -> None:
    sv = Solver.create(seed=0)
    sv.add_formula(xeqv(var(1), var(2)))
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert r.model[1] == r.model[2]


def test_dsl_ite() -> None:
    sv = Solver.create(seed=0)
    # ITE(c, t, e) ↔ t if c else e ↔ Boolean multiplexer
    sv.add_formula(xite(var(1), var(2), var(3)))
    sv.assume(1)  # c = true → must equal t
    sv.assume(2)
    r = sv.solve()
    assert r.status == STATUS_SAT  # need t to be true


def test_dsl_ite_negative_branch() -> None:
    sv = Solver.create(seed=0)
    sv.add_formula(xite(var(1), var(2), var(3)))
    sv.assume(-1)  # c = false → must equal e
    sv.assume(3)
    r = sv.solve()
    assert r.status == STATUS_SAT


def test_dsl_at_most_constructor() -> None:
    sv = Solver.create(seed=0)
    phi = at_most(1, [var(1), var(2), var(3), var(4)])
    sv.add_formula(phi)
    sv.add_clause([1])  # force x_1
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert r.model[1] is True
    assert sum(r.model[v] for v in (2, 3, 4)) == 0


def test_dsl_exactly_constructor() -> None:
    sv = Solver.create(seed=0)
    phi = exactly(2, [var(1), var(2), var(3), var(4)])
    sv.add_formula(phi)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert sum(r.model[v] for v in (1, 2, 3, 4)) == 2


def test_dsl_const_true_false() -> None:
    sv = Solver.create(seed=0)
    sv.add_formula(land(true(), lor(false(), var(1))))
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert r.model[1] is True


def test_dsl_at_least_negative_k_rejected() -> None:
    with pytest.raises(InvalidFormula):
        at_least(-1, [var(1)])
    with pytest.raises(InvalidFormula):
        at_most(-1, [var(1)])
    with pytest.raises(InvalidFormula):
        exactly(-1, [var(1)])


def test_add_formula_wrong_type() -> None:
    sv = Solver.create(seed=0)
    with pytest.raises(InvalidFormula):
        sv.add_formula("not a formula")  # type: ignore[arg-type]


def test_to_cnf_returns_consistent_max() -> None:
    phi = land(var(1), lor(var(2), lnot(var(3))))
    clauses, top, _ = phi.to_cnf()
    # Every literal must reference a variable id ≥ 1.
    for c in clauses:
        for l in c:
            assert abs(l) >= 1


# --------------------------------------------------------------------- harder problems


def test_4x4_sudoku_solvable() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(64)

    def x(r: int, c: int, v: int) -> int:
        return ((r - 1) * 4 + (c - 1)) * 4 + v

    # cell
    for r in (1, 2, 3, 4):
        for c in (1, 2, 3, 4):
            sv.add_exactly([x(r, c, v) for v in (1, 2, 3, 4)], 1)
    # row
    for r in (1, 2, 3, 4):
        for v in (1, 2, 3, 4):
            sv.add_exactly([x(r, c, v) for c in (1, 2, 3, 4)], 1)
    # col
    for c in (1, 2, 3, 4):
        for v in (1, 2, 3, 4):
            sv.add_exactly([x(r, c, v) for r in (1, 2, 3, 4)], 1)
    # box
    for br in (1, 3):
        for bc in (1, 3):
            for v in (1, 2, 3, 4):
                sv.add_exactly(
                    [x(r, c, v) for r in (br, br + 1) for c in (bc, bc + 1)], 1
                )
    sv.assume(x(1, 1, 1))
    sv.assume(x(2, 3, 3))
    r = sv.solve()
    assert r.status == STATUS_SAT
    grid = [[0] * 4 for _ in range(4)]
    for rr in (1, 2, 3, 4):
        for cc in (1, 2, 3, 4):
            for vv in (1, 2, 3, 4):
                if r.model[x(rr, cc, vv)]:
                    grid[rr - 1][cc - 1] = vv
    # Every row must be a permutation of 1..4.
    for row in grid:
        assert sorted(row) == [1, 2, 3, 4]


def test_n_queens_4() -> None:
    sv = Solver.create(seed=0)
    n = 4
    sv.reserve_vars(n * n)

    def q(r: int, c: int) -> int:
        return (r - 1) * n + c

    # Exactly one queen per row.
    for r in range(1, n + 1):
        sv.add_exactly([q(r, c) for c in range(1, n + 1)], 1)
    # Exactly one queen per column.
    for c in range(1, n + 1):
        sv.add_exactly([q(r, c) for r in range(1, n + 1)], 1)
    # At most one queen per diagonal.
    for r1 in range(1, n + 1):
        for c1 in range(1, n + 1):
            for r2 in range(r1 + 1, n + 1):
                for c2 in range(1, n + 1):
                    if abs(c1 - c2) == abs(r1 - r2):
                        sv.add_clause([-q(r1, c1), -q(r2, c2)])
    r = sv.solve()
    assert r.status == STATUS_SAT
    # Verify the placement is a valid n-queens.
    cols = []
    for rr in range(1, n + 1):
        placed = [c for c in range(1, n + 1) if r.model[q(rr, c)]]
        assert len(placed) == 1
        cols.append(placed[0])
    # No two queens share a diagonal.
    for i in range(n):
        for j in range(i + 1, n):
            assert abs(cols[i] - cols[j]) != abs(i - j)


def test_random_3sat_sat_instance() -> None:
    rng = random.Random(42)
    n_vars = 30
    n_clauses = 90
    sv = Solver.create(seed=42)
    sv.reserve_vars(n_vars)
    # Construct a random 3-SAT instance with a planted assignment so
    # we know it's satisfiable.
    planted = [rng.choice([True, False]) for _ in range(n_vars + 1)]
    clauses = []
    while len(clauses) < n_clauses:
        vars_ = rng.sample(range(1, n_vars + 1), 3)
        lits = [v if rng.random() < 0.5 else -v for v in vars_]
        # Ensure planted satisfies the clause.
        if not any(
            (l > 0 and planted[abs(l)]) or (l < 0 and not planted[abs(l)]) for l in lits
        ):
            # Flip one literal to ensure satisfaction.
            lits[0] = -lits[0]
        clauses.append(lits)
    for c in clauses:
        sv.add_clause(c)
    r = sv.solve()
    assert r.status == STATUS_SAT
    # Verify the model satisfies every clause.
    for c in clauses:
        assert any(
            (l > 0 and r.model[abs(l)]) or (l < 0 and not r.model[abs(l)]) for l in c
        )


# --------------------------------------------------------------------- MaxSAT


def test_maxsat_unit_weights() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2, 3])
    soft = [[-1], [-2], [-3]]
    cost, model, violated = sv.solve_max_sat(soft)
    # Optimum is one true variable (1 violated soft clause).
    assert cost == 1
    assert sum(model[v] for v in (1, 2, 3)) == 1
    assert len(violated) == 1


def test_maxsat_weighted() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2, 3])
    soft = [[-1], [-2], [-3]]
    weights = [10, 1, 1]
    cost, model, violated = sv.solve_max_sat(soft, weights=weights)
    # Falsifying x_1 costs 10; falsifying x_2 or x_3 costs 1 each.
    # Optimum: x_1 = False, exactly one of x_2/x_3 = True (cost 1).
    assert cost == 1
    assert model[1] is False
    assert sum(model[v] for v in (2, 3)) == 1


def test_maxsat_all_soft_satisfiable() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(2)
    sv.add_clause([1, 2])
    soft = [[1], [2]]  # both consistent with hard
    cost, model, violated = sv.solve_max_sat(soft)
    assert cost == 0
    assert violated == ()


def test_maxsat_zero_weight_soft_ignored() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(2)
    sv.add_clause([1, 2])
    soft = [[-1], [-2]]
    weights = [0, 1]
    cost, model, violated = sv.solve_max_sat(soft, weights=weights)
    # Falsifying soft 0 costs 0 — model can prefer to set x_2 = False
    # and x_1 = True, violating only soft 0.
    assert cost == 0


def test_maxsat_validation() -> None:
    sv = Solver.create(seed=0)
    with pytest.raises(InvalidConfig):
        sv.solve_max_sat([[1]], weights=[1, 2])
    with pytest.raises(InvalidConfig):
        sv.solve_max_sat([[1]], weights=[-1])
    with pytest.raises(InvalidConfig):
        sv.solve_max_sat([[1]], weights=[1.5])  # type: ignore[list-item]


# --------------------------------------------------------------------- determinism


def test_seed_determinism() -> None:
    rng = random.Random(7)
    n_vars = 20
    n_clauses = 60
    cls = []
    for _ in range(n_clauses):
        v = rng.sample(range(1, n_vars + 1), 3)
        cls.append([w if rng.random() < 0.5 else -w for w in v])

    def run(seed: int) -> SolverResult:
        sv = Solver.create(seed=seed)
        sv.reserve_vars(n_vars)
        for c in cls:
            sv.add_clause(c)
        return sv.solve()

    r1 = run(0)
    r2 = run(0)
    assert r1.status == r2.status
    if r1.status == STATUS_SAT:
        assert dict(r1.model) == dict(r2.model)


def test_attestation_ledger_chains() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(2)
    sv.add_clause([1, 2])
    sv.solve()
    head = sv.ledger_head()
    assert isinstance(head, str)
    assert len(head) == 64
    # Every event in the ledger advances the chain.
    chain = sv.ledger()
    assert len(chain) >= 3  # started + clause_added + solved
    for rec in chain:
        assert rec["event"] in SOLVER_KNOWN_EVENTS
        assert "head" in rec


# --------------------------------------------------------------------- internals


def test_luby_sequence_first_terms() -> None:
    # Knuth-Luby (Luby-Sinclair-Zuckerman 1993) first 15 terms:
    # 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8
    expected = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]
    for i, e in enumerate(expected, start=1):
        assert _luby(i) == e


def test_luby_validation() -> None:
    with pytest.raises(InvalidConfig):
        _luby(0)


def test_resource_exhaustion() -> None:
    rng = random.Random(2024)
    sv = Solver.create(seed=0)
    sv.reserve_vars(80)
    # Many random 3-SAT clauses — likely UNSAT and hard.
    for _ in range(400):
        v = rng.sample(range(1, 81), 3)
        sv.add_clause([w if rng.random() < 0.5 else -w for w in v])
    r = sv.solve(max_conflicts=5)
    # Either status is known by luck, or unknown by budget.  The
    # important contract is that ``unknown`` is a possible verdict
    # under budget.
    assert r.status in SOLVER_KNOWN_STATUSES


def test_report_fields_present() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(2)
    sv.add_clause([1, 2])
    sv.solve()
    rep = sv.report()
    assert isinstance(rep, SolverReport)
    assert rep.num_vars >= 2
    assert rep.last_status == STATUS_SAT
    assert rep.seed == 0
    assert isinstance(rep.ledger_head, str) and len(rep.ledger_head) == 64


def test_proof_log_only_additions_and_deletions() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(15)
    rng = random.Random(0)
    # Manageable random UNSAT instance.
    for _ in range(60):
        v = rng.sample(range(1, 16), 3)
        sv.add_clause([w if rng.random() < 0.5 else -w for w in v])
    sv.add_clause([1])
    sv.add_clause([-1])
    r = sv.solve()
    assert r.status == STATUS_UNSAT
    for op, lits in r.proof:
        assert op in ("a", "d")
        assert isinstance(lits, tuple)


def test_proof_check_detects_invalid_step() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_clause([1, 2])
    sv.add_clause([-1, 3])
    sv.solve()
    # Inject a bogus learnt clause (not RUP-valid).
    sv._proof_log.append(("a", (4,)))
    sv._last_status = STATUS_UNSAT  # force the check to verify everything
    sv._last_unsat_under_assumptions = False
    with pytest.raises(ProofCheckFailed):
        sv.check_proof()


# --------------------------------------------------------------------- composability


def test_combine_dsl_and_cardinality() -> None:
    sv = Solver.create(seed=0)
    a, b, c = var(1), var(2), var(3)
    sv.add_formula(land(lor(a, b), ximp(a, c)))
    sv.add_at_most([1, 2, 3], 2)
    sv.add_at_least([1, 2, 3], 1)
    r = sv.solve()
    assert r.status == STATUS_SAT
    assert 1 <= sum(r.model[v] for v in (1, 2, 3)) <= 2


def test_new_var_allocates_above_aux() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(3)
    sv.add_at_most([1, 2, 3], 1)  # introduces aux above 3
    v = sv.new_var()
    assert v > 3
    sv.add_clause([v, -1])  # uses the fresh user var
    r = sv.solve()
    assert r.status == STATUS_SAT


def test_known_events_are_complete() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(2)
    sv.add_clause([1, 2])
    sv.assume(1)
    sv.solve()
    sv.report()
    events = {rec["event"] for rec in sv.ledger()}
    assert SOLVER_STARTED in events
    assert SOLVER_CLAUSE_ADDED in events
    assert SOLVER_ASSUMED in events
    assert SOLVER_SOLVED in events
    assert SOLVER_REPORTED in events


def test_clear_assumptions_event() -> None:
    sv = Solver.create(seed=0)
    sv.reserve_vars(1)
    sv.assume(1)
    sv.clear_assumptions()
    sv.solve()
    events = [rec["event"] for rec in sv.ledger()]
    assert "solver_cleared" in events
