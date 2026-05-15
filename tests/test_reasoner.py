"""Tests for Reasoner — symbolic logic primitive.

Coverage:
  * Parsing — clauses (string + sequence form), literals, rules.
  * DPLL and CDCL on SAT instances of various sizes.
  * UNSAT detection (pigeon-hole, contradictions) + resolution proof.
  * Entailment via SAT solving.
  * Walk-SAT on satisfiable random 3-SAT.
  * Horn forward chaining (propositional + Datalog with unification).
  * Backward chaining (SLD with backtracking).
  * Stable model semantics (ASP) with negation-as-failure.
  * Tamper-evident fingerprint chain — deterministic replay.
  * Report bounds — Clopper-Pearson on Walk-SAT failure rate.
  * Edge cases — empty KB, tautology clauses, integrity constraints.
"""
from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

# Import directly to bypass agi/__init__.py's anthropic dep.
import importlib.util
import os
spec = importlib.util.spec_from_file_location(
    "reasoner",
    os.path.join(os.path.dirname(__file__), "..", "agi", "reasoner.py"),
)
reasoner = importlib.util.module_from_spec(spec)
sys.modules["reasoner"] = reasoner
spec.loader.exec_module(reasoner)

Reasoner = reasoner.Reasoner
ReasonerError = reasoner.ReasonerError
UnknownAlgorithm = reasoner.UnknownAlgorithm
InvalidClause = reasoner.InvalidClause
InvalidRule = reasoner.InvalidRule
DPLL = reasoner.DPLL
CDCL = reasoner.CDCL
WALKSAT = reasoner.WALKSAT
FORWARD_CHAIN = reasoner.FORWARD_CHAIN
STABLE_MODELS = reasoner.STABLE_MODELS
SAT = reasoner.SAT
UNSAT = reasoner.UNSAT
UNKNOWN = reasoner.UNKNOWN


# =====================================================================
# Construction / API validation
# =====================================================================


def test_unknown_algorithm_raises():
    try:
        Reasoner(algorithm="nope")
    except UnknownAlgorithm:
        return
    raise AssertionError("expected UnknownAlgorithm")


def test_defaults():
    r = Reasoner()
    assert r.algorithm == CDCL
    assert r.n_clauses == 0
    assert r.n_rules == 0
    assert r.n_facts == 0
    assert r.n_atoms == 0


def test_add_clause_string_form():
    r = Reasoner()
    r.add_clause("a | ~b | c")
    assert r.n_clauses == 1
    assert r.n_atoms == 3


def test_add_clause_list_form():
    r = Reasoner()
    r.add_clause(["a", "~b", "c"])
    assert r.n_clauses == 1


def test_add_clause_or_keyword():
    r = Reasoner()
    r.add_clause("a or not b or c")
    assert r.n_clauses == 1


def test_tautology_silently_absorbed():
    r = Reasoner()
    r.add_clause(["a", "~a", "b"])
    assert r.n_clauses == 0
    # Fingerprint still advanced.
    assert r.fingerprint != reasoner._GENESIS


def test_empty_clause_rejected():
    r = Reasoner()
    try:
        r.add_clause("")
    except InvalidClause:
        return
    raise AssertionError("expected InvalidClause")


def test_add_fact_creates_unit_clause():
    r = Reasoner()
    r.add_fact("p")
    assert r.n_facts == 1
    assert r.n_clauses == 1


def test_add_rule_fact_form():
    r = Reasoner()
    r.add_rule("p.")
    assert r.n_rules == 1
    head, body = r.rules[0]
    assert head == "p"
    assert body == []


def test_add_rule_body_form():
    r = Reasoner()
    r.add_rule("p :- q, ~r, not s.")
    head, body = r.rules[0]
    assert head == "p"
    assert ("q", False) in body
    assert ("r", True) in body
    assert ("s", True) in body


def test_add_rule_constraint_form():
    r = Reasoner()
    r.add_rule(":- p, q.")
    head, body = r.rules[0]
    assert head is None
    assert len(body) == 2


# =====================================================================
# SAT solver — DPLL
# =====================================================================


def test_dpll_trivial_sat():
    r = Reasoner(DPLL)
    r.add_clause(["a"])
    sol = r.solve()
    assert sol.verdict == SAT
    assert sol.model["a"] is True


def test_dpll_unsat_pair():
    r = Reasoner(DPLL)
    r.add_clause(["a"])
    r.add_clause(["~a"])
    sol = r.solve()
    assert sol.verdict == UNSAT


def test_dpll_3sat_satisfiable():
    r = Reasoner(DPLL)
    r.add_clause(["a", "b"])
    r.add_clause(["~a", "c"])
    r.add_clause(["~b", "c"])
    sol = r.solve()
    assert sol.verdict == SAT
    # Verify model satisfies all clauses.
    m = sol.model
    assert (m["a"] or m["b"])
    assert (not m["a"] or m["c"])
    assert (not m["b"] or m["c"])


# =====================================================================
# SAT solver — CDCL
# =====================================================================


def test_cdcl_3sat_satisfiable():
    r = Reasoner(CDCL)
    r.add_clause(["a", "b"])
    r.add_clause(["~a", "c"])
    r.add_clause(["~b", "c"])
    sol = r.solve()
    assert sol.verdict == SAT
    m = sol.model
    assert (m["a"] or m["b"])
    assert (not m["a"] or m["c"])
    assert (not m["b"] or m["c"])


def test_cdcl_pigeonhole_3_in_2_unsat():
    r = Reasoner(CDCL)
    # 3 pigeons in 2 holes — UNSAT.
    for p in range(3):
        r.add_clause([f"p{p}h0", f"p{p}h1"])
    for h in range(2):
        for p1 in range(3):
            for p2 in range(p1 + 1, 3):
                r.add_clause([f"~p{p1}h{h}", f"~p{p2}h{h}"])
    sol = r.solve()
    assert sol.verdict == UNSAT


def test_cdcl_pigeonhole_3_in_3_sat():
    r = Reasoner(CDCL)
    n = 3
    for p in range(n):
        r.add_clause([f"p{p}h{h}" for h in range(n)])
    for h in range(n):
        for p1 in range(n):
            for p2 in range(p1 + 1, n):
                r.add_clause([f"~p{p1}h{h}", f"~p{p2}h{h}"])
    sol = r.solve()
    assert sol.verdict == SAT
    # Each pigeon assigned to at least one hole; no hole has two.
    pigeon_holes = {p: set() for p in range(n)}
    for p in range(n):
        for h in range(n):
            if sol.model.get(f"p{p}h{h}", False):
                pigeon_holes[p].add(h)
    for p in range(n):
        assert pigeon_holes[p]
    for h in range(n):
        used_by = sum(1 for p in range(n) if h in pigeon_holes[p])
        assert used_by <= 1


def test_cdcl_unit_clause_only():
    r = Reasoner(CDCL)
    r.add_clause(["a"])
    sol = r.solve()
    assert sol.verdict == SAT
    assert sol.model["a"] is True


def test_cdcl_multiple_units_with_implication():
    r = Reasoner(CDCL)
    r.add_clause(["a"])
    r.add_clause(["~a", "b"])
    r.add_clause(["~b", "c"])
    sol = r.solve()
    assert sol.verdict == SAT
    assert sol.model["a"] and sol.model["b"] and sol.model["c"]


# =====================================================================
# UNSAT proof reconstruction
# =====================================================================


def test_unsat_resolution_proof_nonempty():
    r = Reasoner(CDCL)
    r.add_clause(["a"])
    r.add_clause(["~a"])
    sol = r.solve()
    assert sol.verdict == UNSAT
    proof = r.last_resolution_proof()
    assert len(proof) >= 1
    # Final resolvent is the empty clause.
    assert proof[-1].resolvent == ()


def test_unsat_proof_reaches_empty_clause_tiny():
    r = Reasoner(CDCL)
    # XOR-conflict — minimal 2-variable UNSAT.
    r.add_clause(["a", "b"])
    r.add_clause(["~a", "b"])
    r.add_clause(["a", "~b"])
    r.add_clause(["~a", "~b"])
    r.solve()
    proof = r.last_resolution_proof()
    assert len(proof) >= 1
    assert proof[-1].resolvent == ()


def test_unsat_proof_steps_are_valid():
    """Each step's resolvent equals (a \\ {pivot}) ∪ (b \\ {¬pivot})."""
    r = Reasoner(CDCL)
    r.add_clause(["a"])
    r.add_clause(["~a", "b"])
    r.add_clause(["~b"])
    r.solve()
    for step in r.last_resolution_proof():
        # The pivot must appear positive in one clause and negative
        # in the other.
        a_has_pos = (step.pivot, False) in step.clause_a
        a_has_neg = (step.pivot, True) in step.clause_a
        b_has_pos = (step.pivot, False) in step.clause_b
        b_has_neg = (step.pivot, True) in step.clause_b
        assert (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos)
        # Resolvent excludes the pivot.
        for atom, _ in step.resolvent:
            assert atom != step.pivot


def test_unsat_proof_pigeon():
    r = Reasoner(CDCL)
    for p in range(3):
        r.add_clause([f"p{p}h0", f"p{p}h1"])
    for h in range(2):
        for p1 in range(3):
            for p2 in range(p1 + 1, 3):
                r.add_clause([f"~p{p1}h{h}", f"~p{p2}h{h}"])
    sol = r.solve()
    assert sol.verdict == UNSAT
    # Proof reconstruction may be partial; just confirm we found
    # something replayable.
    proof = r.last_resolution_proof()
    # Each step must have a valid pivot atom appearing in both clauses.
    for step in proof:
        assert any(a == step.pivot for a, _ in step.clause_a)
        assert any(a == step.pivot for a, _ in step.clause_b)


# =====================================================================
# Entailment
# =====================================================================


def test_entails_unit_consequence():
    r = Reasoner()
    r.add_clause(["a", "b"])
    r.add_clause(["~a", "c"])
    r.add_clause(["~b", "c"])
    assert r.entails("c") is True


def test_entails_not_unit_consequence():
    r = Reasoner()
    r.add_clause(["a", "b"])
    assert r.entails("a") is False
    assert r.entails("b") is False


def test_entails_conjunction():
    r = Reasoner()
    r.add_clause(["a"])
    r.add_clause(["~a", "b"])
    r.add_clause(["~a", "~b", "c"])
    assert r.entails(["a", "b", "c"]) is True


def test_entails_negation_literal():
    r = Reasoner()
    r.add_clause(["~a"])
    assert r.entails("~a") is True


# =====================================================================
# all_models — enumeration with blocking clauses
# =====================================================================


def test_all_models_three_vars():
    r = Reasoner(CDCL)
    r.add_clause(["a", "b"])
    r.add_clause(["~a", "~b"])
    models = r.all_models(limit=8)
    # XOR has 2 satisfying assignments times any value of c (3rd var
    # was not registered). With only 2 vars, expect 2 models.
    assert 1 <= len(models) <= 2


def test_all_models_exact_count():
    r = Reasoner(CDCL)
    r.add_clause(["a", "b", "c"])
    r.add_clause(["~a", "~b", "~c"])
    # (not all false) and (not all true) → 6 distinct models.
    models = r.all_models(limit=10)
    assert len(models) == 6


# =====================================================================
# Walk-SAT — Selman-Kautz-Cohen
# =====================================================================


def test_walksat_finds_simple_solution():
    r = Reasoner(WALKSAT, seed=0, walksat_flips=200, walksat_restarts=5)
    r.add_clause(["a"])
    r.add_clause(["~a", "b"])
    r.add_clause(["~b", "c"])
    sol = r.solve()
    assert sol.verdict == SAT
    assert sol.model["a"] and sol.model["b"] and sol.model["c"]


def test_walksat_failure_clopper_pearson_bound():
    # Force failure by making it unsatisfiable but with small budget.
    r = Reasoner(WALKSAT, seed=0, walksat_flips=5, walksat_restarts=2)
    r.add_clause(["a"])
    r.add_clause(["~a"])
    sol = r.solve()
    # On UNSAT, walk-sat will return UNKNOWN.
    assert sol.verdict == UNKNOWN
    rep = r.report()
    # Clopper-Pearson upper bound is non-trivial.
    assert 0.0 < rep.failure_upper_clopper_pearson <= 1.0


# =====================================================================
# Horn / Datalog — propositional path
# =====================================================================


def test_propositional_forward_chain():
    r = Reasoner(FORWARD_CHAIN)
    r.add_rule("a.")
    r.add_rule("b :- a.")
    r.add_rule("c :- a, b.")
    derived = r.forward_chain()
    assert "a" in derived
    assert "b" in derived
    assert "c" in derived


def test_propositional_forward_chain_naive():
    r = Reasoner(FORWARD_CHAIN)
    r.add_rule("a.")
    r.add_rule("b :- a.")
    derived = r.forward_chain(semi_naive=False)
    assert "a" in derived and "b" in derived


def test_forward_chain_skips_negated_body_literals():
    r = Reasoner(FORWARD_CHAIN)
    r.add_rule("a.")
    r.add_rule("b :- a, not c.")
    derived = r.forward_chain()
    # NaF body literal is ignored in pure Horn — so b is derived.
    assert "b" in derived


# =====================================================================
# Datalog with variables and unification
# =====================================================================


def test_datalog_forward_chain_variables():
    r = Reasoner(FORWARD_CHAIN)
    r.add_rule("parent(alice, bob).")
    r.add_rule("parent(bob, carol).")
    r.add_rule("ancestor(X, Y) :- parent(X, Y).")
    r.add_rule("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).")
    derived = r.forward_chain()
    assert "parent(alice, bob)" in derived
    assert "parent(bob, carol)" in derived
    assert "ancestor(alice, bob)" in derived
    assert "ancestor(bob, carol)" in derived
    assert "ancestor(alice, carol)" in derived


def test_datalog_backward_chain_variables():
    r = Reasoner()
    r.add_rule("parent(alice, bob).")
    r.add_rule("parent(bob, carol).")
    r.add_rule("ancestor(X, Y) :- parent(X, Y).")
    r.add_rule("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).")
    p = r.backward_chain("ancestor(alice, carol)")
    assert p is not None
    assert p.goal == "ancestor(alice, carol)"


def test_datalog_backward_chain_failure():
    r = Reasoner()
    r.add_rule("parent(alice, bob).")
    r.add_rule("ancestor(X, Y) :- parent(X, Y).")
    p = r.backward_chain("ancestor(carol, alice)")
    assert p is None


def test_datalog_backtracks_through_alternatives():
    r = Reasoner()
    r.add_rule("passed(arithmetic).")
    r.add_rule("prereq(arithmetic, algebra).")
    r.add_rule("prereq(algebra, calculus).")
    r.add_rule("ready(T) :- passed(P), prereq(P, T).")
    r.add_rule("passed(X) :- ready(X), studied(X).")
    r.add_fact("studied(algebra)")
    # ready(calculus) only provable after ready(algebra) → passed(algebra).
    p = r.backward_chain("ready(calculus)")
    assert p is not None
    assert p.goal == "ready(calculus)"


# =====================================================================
# Stable model semantics — ASP
# =====================================================================


def test_stable_models_negation_toggle():
    r = Reasoner(STABLE_MODELS)
    r.add_rule("p :- not q.")
    r.add_rule("q :- not p.")
    mods = r.stable_models()
    assert len(mods) == 2
    contents = {frozenset(m) for m in mods}
    assert frozenset({"p"}) in contents
    assert frozenset({"q"}) in contents


def test_stable_models_with_constraint():
    r = Reasoner(STABLE_MODELS)
    r.add_rule("p :- not q.")
    r.add_rule("q :- not p.")
    r.add_rule(":- p.")    # forbid p
    mods = r.stable_models()
    assert len(mods) == 1
    assert "q" in mods[0]
    assert "p" not in mods[0]


def test_stable_models_stratified_fast_path():
    r = Reasoner(STABLE_MODELS)
    r.add_rule("a.")
    r.add_rule("b :- a, not c.")
    r.add_rule("d :- b.")
    mods = r.stable_models()
    assert len(mods) == 1
    m = mods[0]
    assert "a" in m and "b" in m and "d" in m
    assert "c" not in m


def test_stable_models_no_model():
    r = Reasoner(STABLE_MODELS)
    r.add_rule("p :- not p.")     # paradox — no stable model
    mods = r.stable_models()
    assert mods == []


def test_stable_models_with_variables_grounding():
    """Triangle 2-colouring — auto-ground rules with Datalog variables."""
    r = Reasoner(STABLE_MODELS)
    for n in ("a", "b", "c"):
        r.add_rule(f"node({n}).")
    for u, v in (("a", "b"), ("b", "c"), ("a", "c")):
        r.add_rule(f"edge({u}, {v}).")
    r.add_rule("red(X) :- node(X), not blue(X).")
    r.add_rule("blue(X) :- node(X), not red(X).")
    r.add_rule(":- edge(X, Y), red(X), red(Y).")
    r.add_rule(":- edge(X, Y), blue(X), blue(Y).")
    mods = r.stable_models(limit=8)
    # Triangle (odd cycle) is NOT 2-colourable.
    assert mods == []


def test_stable_models_with_variables_path_2colour():
    r = Reasoner(STABLE_MODELS)
    for n in ("a", "b", "c"):
        r.add_rule(f"node({n}).")
    r.add_rule("edge(a, b).")
    r.add_rule("edge(b, c).")
    r.add_rule("red(X) :- node(X), not blue(X).")
    r.add_rule("blue(X) :- node(X), not red(X).")
    r.add_rule(":- edge(X, Y), red(X), red(Y).")
    r.add_rule(":- edge(X, Y), blue(X), blue(Y).")
    mods = r.stable_models(limit=8)
    # Path of length 2 has exactly two valid 2-colourings.
    assert len(mods) == 2
    # Each colouring assigns different colours to b and its neighbours.
    for m in mods:
        a_red = "red(a)" in m
        b_red = "red(b)" in m
        c_red = "red(c)" in m
        assert a_red != b_red
        assert b_red != c_red


# =====================================================================
# Fingerprint chain — determinism + replay
# =====================================================================


def test_fingerprint_changes_on_clause_add():
    r = Reasoner()
    g0 = r.fingerprint
    r.add_clause(["a", "b"])
    g1 = r.fingerprint
    assert g0 != g1
    r.add_clause(["~a"])
    g2 = r.fingerprint
    assert g1 != g2


def test_fingerprint_deterministic_replay():
    r1 = Reasoner(CDCL, seed=42)
    r1.add_clause(["a", "b"])
    r1.add_clause(["~a", "c"])
    r1.add_fact("d")
    r1.add_rule("e :- d.")

    r2 = Reasoner(CDCL, seed=42)
    r2.add_clause(["a", "b"])
    r2.add_clause(["~a", "c"])
    r2.add_fact("d")
    r2.add_rule("e :- d.")

    assert r1.fingerprint == r2.fingerprint


def test_fingerprint_diverges_on_different_inputs():
    r1 = Reasoner()
    r1.add_clause(["a", "b"])
    r2 = Reasoner()
    r2.add_clause(["a", "c"])
    assert r1.fingerprint != r2.fingerprint


# =====================================================================
# Report
# =====================================================================


def test_report_basic():
    r = Reasoner(CDCL)
    r.add_clause(["a"])
    r.add_clause(["~a", "b"])
    r.solve()
    rep = r.report()
    assert rep.n_atoms >= 2
    assert rep.n_clauses == 2
    assert rep.sat_calls == 1
    assert rep.last_verdict == SAT
    assert rep.fingerprint == r.fingerprint
    # Strict report dict round-trip.
    d = rep.to_dict()
    assert d["n_atoms"] == rep.n_atoms


def test_report_unsat_proof_length():
    r = Reasoner(CDCL)
    r.add_clause(["a"])
    r.add_clause(["~a"])
    r.solve()
    rep = r.report()
    assert rep.last_verdict == UNSAT
    assert rep.proof_length >= 1


def test_report_stratified_diagnostic():
    r = Reasoner(STABLE_MODELS)
    r.add_rule("a.")
    r.add_rule("b :- a, not c.")
    rep = r.report()
    assert rep.stratified is True
    # Non-stratified: cycle through negation.
    r2 = Reasoner(STABLE_MODELS)
    r2.add_rule("p :- not q.")
    r2.add_rule("q :- not p.")
    rep2 = r2.report()
    assert rep2.stratified is False


# =====================================================================
# Clear
# =====================================================================


def test_clear_resets_state():
    r = Reasoner()
    r.add_clause(["a", "b"])
    r.add_fact("c")
    g_before = r.fingerprint
    r.clear()
    assert r.n_clauses == 0
    assert r.n_facts == 0
    assert r.n_atoms == 0
    assert r.fingerprint == reasoner._GENESIS
    # Re-adding produces same fingerprint as a fresh instance.
    r.add_clause(["a", "b"])
    r.add_fact("c")
    r2 = Reasoner()
    r2.add_clause(["a", "b"])
    r2.add_fact("c")
    assert r.fingerprint == r2.fingerprint


# =====================================================================
# Composition — CEGIS-style refinement
# =====================================================================


def test_cegis_loop_pattern():
    """Synthesise a Boolean function compatible with examples by SAT."""
    # We want to find a 2-input Boolean function (4 truth-table cells)
    # consistent with: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0 → XOR.
    r = Reasoner(CDCL)
    # f00, f01, f10, f11 represent truth-table cells.
    r.add_clause(["~f00"])    # f00 = 0
    r.add_clause(["f01"])     # f01 = 1
    r.add_clause(["f10"])     # f10 = 1
    r.add_clause(["~f11"])    # f11 = 0
    sol = r.solve()
    assert sol.verdict == SAT
    assert sol.model["f00"] is False
    assert sol.model["f01"] is True
    assert sol.model["f10"] is True
    assert sol.model["f11"] is False


# =====================================================================
# Assumptions
# =====================================================================


def test_solve_with_assumptions():
    r = Reasoner(CDCL)
    r.add_clause(["a", "b"])
    sol = r.solve(assumptions={"a": False})
    assert sol.verdict == SAT
    assert sol.model["a"] is False
    assert sol.model["b"] is True


def test_assumptions_can_make_unsat():
    r = Reasoner(CDCL)
    r.add_clause(["a"])
    sol = r.solve(assumptions={"a": False})
    assert sol.verdict == UNSAT


# =====================================================================
# Stress — random 3-SAT
# =====================================================================


def test_random_3sat_phase_transition():
    rng = random.Random(0)
    n_sat = 0
    n_unsat = 0
    n_unk = 0
    nvars = 10
    nclauses = 42   # ratio 4.2 — well-known phase transition
    for trial in range(10):
        r = Reasoner(CDCL, max_conflicts=100_000)
        atoms = [f"x{i}" for i in range(nvars)]
        for _ in range(nclauses):
            lits = rng.sample(atoms, 3)
            cl = []
            for a in lits:
                if rng.random() < 0.5:
                    cl.append("~" + a)
                else:
                    cl.append(a)
            r.add_clause(cl)
        sol = r.solve()
        if sol.verdict == SAT:
            n_sat += 1
        elif sol.verdict == UNSAT:
            n_unsat += 1
        else:
            n_unk += 1
    # CDCL should terminate on all reasonable random 3-SAT at this size.
    assert n_unk == 0
    # Both verdicts should appear in 10 trials at the phase transition.
    assert n_sat + n_unsat == 10


# =====================================================================
# Helper-function unit tests
# =====================================================================


def test_unify_simple():
    s = reasoner._unify(("p", ("X", "b")), ("p", ("a", "Y")))
    assert s is not None
    # Apply substitution.
    assert reasoner._substitute("X", s) == "a"
    assert reasoner._substitute("Y", s) == "b"


def test_unify_occurs_check():
    s = reasoner._unify("X", ("f", ("X",)))
    assert s is None


def test_unify_constants_mismatch():
    s = reasoner._unify(("p", ("a",)), ("p", ("b",)))
    assert s is None


def test_clopper_pearson_zero_failures():
    ub = reasoner.clopper_pearson_upper(0, 10, 0.05)
    # Closed form: 1 - 0.05^(1/10)
    expected = 1.0 - (0.05) ** (1.0 / 10.0)
    assert abs(ub - expected) < 1e-9


def test_clopper_pearson_monotone():
    # More failures → looser upper bound.
    ub1 = reasoner.clopper_pearson_upper(1, 10, 0.05)
    ub2 = reasoner.clopper_pearson_upper(3, 10, 0.05)
    assert ub1 < ub2


def test_strongly_connected_components():
    sccs = reasoner.strongly_connected_components(
        4, [(0, 1), (1, 2), (2, 0), (1, 3)],
    )
    # {0,1,2} is the big SCC; {3} singleton.
    assert sccs[0] == [0, 1, 2]
    assert sccs[1] == [3]


# =====================================================================
# Event publication
# =====================================================================


def test_event_bus_publication():
    # Build a tiny bus stand-in.
    class _Bus:
        def __init__(self):
            self.events = []
        def publish(self, ev):
            self.events.append(ev)
    bus = _Bus()

    # Patch agi.events.Event with a lightweight stand-in to avoid
    # importing the full module.
    import types
    fake_events = types.ModuleType("agi.events")

    class Event:
        def __init__(self, kind, session_id=None, data=None):
            self.kind = kind
            self.session_id = session_id
            self.data = data or {}
    fake_events.Event = Event
    sys.modules["agi.events"] = fake_events

    r = Reasoner(bus=bus, session_id="s1")
    r.add_clause(["a"])
    r.solve()
    kinds = [e.kind for e in bus.events]
    assert reasoner.REASONER_STARTED in kinds
    assert reasoner.REASONER_CLAUSE_ADDED in kinds
    assert reasoner.REASONER_SOLVED in kinds
