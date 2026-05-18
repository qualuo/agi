"""Tests for the Verifier runtime primitive — LCF-style proof kernel."""
from __future__ import annotations

import pytest

from agi.verifier import (
    CNFFormula,
    EquationalAxiom,
    EquationalProof,
    FAILED,
    InvalidConfig,
    InvalidFormula,
    InvalidProof,
    KIND_EQUATIONAL,
    KIND_NATURAL_DEDUCTION,
    KIND_RESOLUTION,
    Kernel,
    KernelViolation,
    MALFORMED,
    NaturalDeductionProof,
    NaturalDeductionStep,
    REWRITE_BACKWARD,
    REWRITE_FORWARD,
    ResolutionProof,
    ResolutionStep,
    RewriteStep,
    RULE_AND_ELIM_L,
    RULE_AND_ELIM_R,
    RULE_AND_INTRO,
    RULE_ASSUMPTION,
    RULE_BOT_ELIM,
    RULE_DNE,
    RULE_IFF_ELIM_L,
    RULE_IFF_ELIM_R,
    RULE_IFF_INTRO,
    RULE_IMP_ELIM,
    RULE_IMP_INTRO,
    RULE_LEM,
    RULE_NOT_ELIM,
    RULE_NOT_INTRO,
    RULE_OR_ELIM,
    RULE_OR_INTRO_L,
    RULE_OR_INTRO_R,
    RULE_PREMISE,
    RULE_REPEAT,
    RULE_TOP_INTRO,
    Term,
    VERIFIED,
    VERIFIER_KNOWN_EVENTS,
    VERIFIER_KNOWN_KINDS,
    VERIFIER_KNOWN_ND_RULES,
    VERIFIER_KNOWN_STATUSES,
    Verifier,
    VerifierConfig,
    VerifierReport,
    kernel_rule_count,
    parse_term,
    tcb_summary,
    verify_equational,
    verify_natural_deduction,
    verify_resolution,
)


# ---------------------------------------------------------------------------
# Term construction + parsing
# ---------------------------------------------------------------------------


class TestTerm:
    def test_atom(self):
        t = Term.atom("p")
        assert t.is_atom()
        assert str(t) == "p"
        assert t.depth() == 1
        assert t.size() == 1

    def test_not(self):
        t = Term.neg(Term.atom("p"))
        assert t.is_not()
        assert str(t) == "¬p"
        assert t.depth() == 2

    def test_binary_connectives(self):
        p, q = Term.atom("p"), Term.atom("q")
        assert Term.conj(p, q).is_and()
        assert Term.disj(p, q).is_or()
        assert Term.imp(p, q).is_imp()
        assert Term.iff(p, q).is_iff()

    def test_bot_top(self):
        assert Term.bot().is_bot()
        assert Term.top().is_top()
        assert Term.bot() == Term.bot()
        assert Term.top() == Term.top()

    def test_value_equality(self):
        p1 = Term.atom("p")
        p2 = Term.atom("p")
        assert p1 == p2
        assert hash(p1) == hash(p2)
        # Used as dict keys.
        d = {Term.conj(p1, p1): "x"}
        assert d[Term.conj(p2, p2)] == "x"

    def test_invalid_atom_name(self):
        with pytest.raises(InvalidFormula):
            Term.atom("")
        with pytest.raises(InvalidFormula):
            Term.atom(None)  # type: ignore[arg-type]

    def test_require_term(self):
        with pytest.raises(InvalidFormula):
            Term.neg("not a term")  # type: ignore[arg-type]


class TestParser:
    def test_atom(self):
        assert parse_term("p") == Term.atom("p")

    def test_negation(self):
        assert parse_term("~p") == Term.neg(Term.atom("p"))
        # Double negation.
        assert parse_term("~~p") == Term.neg(Term.neg(Term.atom("p")))

    def test_imp_right_assoc(self):
        # p -> q -> r  ==  p -> (q -> r)
        t = parse_term("p -> q -> r")
        p, q, r = Term.atom("p"), Term.atom("q"), Term.atom("r")
        assert t == Term.imp(p, Term.imp(q, r))

    def test_iff(self):
        assert parse_term("p <-> q") == Term.iff(Term.atom("p"), Term.atom("q"))

    def test_and_or_precedence(self):
        # & binds tighter than | which binds tighter than ->
        t = parse_term("p & q | r")
        p, q, r = Term.atom("p"), Term.atom("q"), Term.atom("r")
        assert t == Term.disj(Term.conj(p, q), r)

    def test_parens(self):
        t = parse_term("(p | q) & r")
        p, q, r = Term.atom("p"), Term.atom("q"), Term.atom("r")
        assert t == Term.conj(Term.disj(p, q), r)

    def test_top_bot(self):
        assert parse_term("T") == Term.top()
        assert parse_term("F") == Term.bot()
        assert parse_term("Top") == Term.top()
        assert parse_term("Bot") == Term.bot()

    def test_function_application_is_opaque(self):
        # f(x,y) is parsed as a single atom with that literal name.
        t = parse_term("p(x, y)")
        assert t.is_atom()
        assert t.payload == "p(x, y)"

    def test_trailing_garbage_rejected(self):
        with pytest.raises(InvalidFormula):
            parse_term("p garbage")

    def test_missing_paren(self):
        with pytest.raises(InvalidFormula):
            parse_term("(p & q")

    def test_empty_string(self):
        with pytest.raises(InvalidFormula):
            parse_term("")

    def test_round_trip(self):
        for s in ["p", "p & q", "p | q", "p -> q", "(p & q) -> r", "~p -> q"]:
            t = parse_term(s)
            t2 = parse_term(str(t).replace("¬", "~").replace("∧", "&").replace("∨", "|").replace("→", "->").replace("↔", "<->"))
            assert t == t2


# ---------------------------------------------------------------------------
# CNF / Resolution data structures
# ---------------------------------------------------------------------------


class TestCNFFormula:
    def test_parse_basic(self):
        f = CNFFormula.parse("1 2\n-1 3\n-2 -3")
        assert len(f) == 3
        # canonical order: sort by (abs(lit), lit < 0)
        assert f.clauses[0] == (1, 2)
        assert f.clauses[1] == (-1, 3)
        assert f.clauses[2] == (-2, -3)

    def test_parse_skips_blanks_and_comments(self):
        f = CNFFormula.parse("# header\n1 -2\n\n# mid\n3\n")
        assert len(f) == 2

    def test_canonical_clause_dedups(self):
        f = CNFFormula.of([[1, 2, 1]])
        assert f.clauses[0] == (1, 2)

    def test_tautology_rejected(self):
        with pytest.raises(InvalidFormula):
            CNFFormula.of([[1, -1, 2]])

    def test_literal_zero_rejected(self):
        with pytest.raises(InvalidFormula):
            CNFFormula.of([[1, 0, 2]])

    def test_variables(self):
        f = CNFFormula.of([[1, -2], [3, -1]])
        assert f.variables() == (1, 2, 3)


class TestResolutionStep:
    def test_valid(self):
        s = ResolutionStep(parents=(0, 1), pivot=1, resolvent=(2,))
        assert s.pivot == 1

    def test_bad_parents(self):
        with pytest.raises(InvalidProof):
            ResolutionStep(parents=(0,), pivot=1, resolvent=())  # type: ignore[arg-type]

    def test_negative_parent(self):
        with pytest.raises(InvalidProof):
            ResolutionStep(parents=(-1, 0), pivot=1, resolvent=())

    def test_nonpositive_pivot(self):
        with pytest.raises(InvalidProof):
            ResolutionStep(parents=(0, 1), pivot=0, resolvent=())
        with pytest.raises(InvalidProof):
            ResolutionStep(parents=(0, 1), pivot=-1, resolvent=())


# ---------------------------------------------------------------------------
# Resolution kernel
# ---------------------------------------------------------------------------


class TestResolutionKernel:
    def test_basic_resolution(self):
        # (p ∨ q), (¬p ∨ r) → (q ∨ r)
        assert Kernel.resolve((1, 2), (-1, 3), 1) == (2, 3)

    def test_resolution_to_empty(self):
        assert Kernel.resolve((1,), (-1,), 1) == ()

    def test_resolution_dedup(self):
        # (p ∨ q), (¬p ∨ q) on p → (q,)
        assert Kernel.resolve((1, 2), (-1, 2), 1) == (2,)

    def test_resolution_tautology_rejected(self):
        # (p ∨ q), (¬p ∨ ¬q) on p → would be (q ∨ ¬q), a tautology
        with pytest.raises(KernelViolation):
            Kernel.resolve((1, 2), (-1, -2), 1)

    def test_pivot_not_in_clauses(self):
        with pytest.raises(KernelViolation):
            Kernel.resolve((2, 3), (-2, -3), 1)

    def test_same_sign_rejected(self):
        # both have positive p
        with pytest.raises(KernelViolation):
            Kernel.resolve((1, 2), (1, 3), 1)

    def test_pivot_must_be_positive(self):
        with pytest.raises(KernelViolation):
            Kernel.resolve((1,), (-1,), -1)


# ---------------------------------------------------------------------------
# verify_resolution
# ---------------------------------------------------------------------------


class TestVerifyResolution:
    def test_simple_unsat(self):
        # (p), (¬p) → ⊥
        f = CNFFormula.of([[1], [-1]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
        ))
        rep = verify_resolution(f, proof)
        assert rep.status == VERIFIED
        assert rep.kind == KIND_RESOLUTION
        assert rep.failed_step is None
        assert rep.n_steps == 1
        assert rep.kernel_calls == 1
        assert rep.certificate

    def test_chain_to_empty(self):
        # (p ∨ q) ∧ (¬p) ∧ (¬q) → step 0: resolve (p∨q) with (¬p) on p → (q)
        # step 1: resolve new (q) with (¬q) on q → ⊥
        f = CNFFormula.of([[1, 2], [-1], [-2]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=(2,)),
            ResolutionStep(parents=(3, 2), pivot=2, resolvent=()),
        ))
        rep = verify_resolution(f, proof)
        assert rep.status == VERIFIED
        assert rep.kernel_calls == 2

    def test_wrong_pivot(self):
        f = CNFFormula.of([[1, 2], [-1, 3]])
        # Step claims pivot 2, but 2 is in only one clause.
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=2, resolvent=()),
        ))
        rep = verify_resolution(f, proof)
        assert rep.status == FAILED
        assert rep.failed_step == 0
        assert "pivot" in rep.failure_reason

    def test_resolvent_mismatch(self):
        f = CNFFormula.of([[1], [-1]])
        # Claim resolvent is (5,) instead of empty.
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=(5,)),
        ))
        rep = verify_resolution(f, proof)
        assert rep.status == FAILED
        assert rep.failed_step == 0
        assert "mismatch" in rep.failure_reason

    def test_no_empty_at_end(self):
        # Valid resolution but never reaches ⊥.
        f = CNFFormula.of([[1, 2], [-1, 3]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=(2, 3)),
        ))
        rep = verify_resolution(f, proof)
        assert rep.status == FAILED
        assert "empty clause" in rep.failure_reason

    def test_out_of_range_parent(self):
        f = CNFFormula.of([[1], [-1]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 5), pivot=1, resolvent=()),
        ))
        rep = verify_resolution(f, proof)
        assert rep.status == FAILED
        assert "out of range" in rep.failure_reason

    def test_certificate_deterministic(self):
        f = CNFFormula.of([[1], [-1]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
        ))
        r1 = verify_resolution(f, proof, hmac_key=b"k")
        r2 = verify_resolution(f, proof, hmac_key=b"k")
        assert r1.certificate == r2.certificate
        r3 = verify_resolution(f, proof, hmac_key=b"other")
        assert r3.certificate != r1.certificate

    def test_malformed(self):
        rep = verify_resolution(None, ResolutionProof())  # type: ignore[arg-type]
        assert rep.status == MALFORMED
        rep = verify_resolution(CNFFormula(), "not a proof")  # type: ignore[arg-type]
        assert rep.status == MALFORMED

    def test_max_proof_length(self):
        f = CNFFormula.of([[1], [-1]])
        # Build a 3-step proof; cap at 1 to trigger MALFORMED.
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
        ))
        rep = verify_resolution(f, proof, max_proof_length=1)
        assert rep.status == MALFORMED


# ---------------------------------------------------------------------------
# Natural-deduction kernel
# ---------------------------------------------------------------------------


class TestNDKernel:
    def test_assume(self):
        p = Term.atom("p")
        s = Kernel.nd_assume(p)
        assert s.phi == p
        assert s.gamma == frozenset({p})

    def test_and_intro_elim(self):
        p, q = Term.atom("p"), Term.atom("q")
        s_p = Kernel.nd_assume(p)
        s_q = Kernel.nd_assume(q)
        s_and = Kernel.nd_and_intro(s_p, s_q)
        assert s_and.phi == Term.conj(p, q)
        assert s_and.gamma == frozenset({p, q})
        assert Kernel.nd_and_elim_l(s_and).phi == p
        assert Kernel.nd_and_elim_r(s_and).phi == q

    def test_and_elim_requires_and(self):
        with pytest.raises(KernelViolation):
            Kernel.nd_and_elim_l(Kernel.nd_assume(Term.atom("p")))

    def test_imp_intro_discharges(self):
        p = Term.atom("p")
        s = Kernel.nd_assume(p)
        s_imp = Kernel.nd_imp_intro(s, p)
        assert s_imp.phi == Term.imp(p, p)
        assert s_imp.gamma == frozenset()

    def test_imp_intro_undischarged_premise(self):
        p, q = Term.atom("p"), Term.atom("q")
        s = Kernel.nd_assume(p)
        # Trying to discharge q (not in gamma).
        with pytest.raises(KernelViolation):
            Kernel.nd_imp_intro(s, q)

    def test_modus_ponens(self):
        p, q = Term.atom("p"), Term.atom("q")
        s_imp = Kernel.nd_assume(Term.imp(p, q))
        s_p = Kernel.nd_assume(p)
        s_q = Kernel.nd_imp_elim(s_imp, s_p)
        assert s_q.phi == q
        assert s_q.gamma == frozenset({Term.imp(p, q), p})

    def test_not_intro_elim(self):
        p = Term.atom("p")
        # Assume p, assume ¬p; derive ⊥ via not_elim.
        s_p = Kernel.nd_assume(p)
        s_np = Kernel.nd_assume(Term.neg(p))
        s_bot = Kernel.nd_not_elim(s_p, s_np)
        assert s_bot.phi == Term.bot()
        # Then not_intro to discharge p.
        s_not_p = Kernel.nd_not_intro(s_bot, p)
        # We discharge p; ¬p remains in gamma.
        assert s_not_p.phi == Term.neg(p)
        assert s_not_p.gamma == frozenset({Term.neg(p)})

    def test_bot_elim(self):
        s = Kernel.nd_assume(Term.bot())
        s2 = Kernel.nd_bot_elim(s, Term.atom("anything"))
        assert s2.phi == Term.atom("anything")

    def test_lem(self):
        p = Term.atom("p")
        s = Kernel.nd_lem(Term.disj(p, Term.neg(p)))
        assert s.gamma == frozenset()
        assert s.phi == Term.disj(p, Term.neg(p))

    def test_lem_only_for_phi_or_not_phi(self):
        p, q = Term.atom("p"), Term.atom("q")
        with pytest.raises(KernelViolation):
            Kernel.nd_lem(Term.disj(p, q))

    def test_dne(self):
        p = Term.atom("p")
        s = Kernel.nd_assume(Term.neg(Term.neg(p)))
        assert Kernel.nd_dne(s).phi == p

    def test_top_intro(self):
        s = Kernel.nd_top_intro()
        assert s.phi == Term.top()
        assert s.gamma == frozenset()

    def test_iff_intro_elim(self):
        p, q = Term.atom("p"), Term.atom("q")
        s_fwd = Kernel.nd_assume(Term.imp(p, q))
        s_bwd = Kernel.nd_assume(Term.imp(q, p))
        s_iff = Kernel.nd_iff_intro(s_fwd, s_bwd)
        assert s_iff.phi == Term.iff(p, q)
        assert Kernel.nd_iff_elim_l(s_iff).phi == Term.imp(p, q)
        assert Kernel.nd_iff_elim_r(s_iff).phi == Term.imp(q, p)

    def test_or_intro(self):
        p, q = Term.atom("p"), Term.atom("q")
        s_p = Kernel.nd_assume(p)
        target = Term.disj(p, q)
        s = Kernel.nd_or_intro_l(s_p, target)
        assert s.phi == target
        # Wrong-side target rejected.
        with pytest.raises(KernelViolation):
            Kernel.nd_or_intro_l(s_p, Term.disj(q, p))

    def test_or_elim(self):
        p, q, r = Term.atom("p"), Term.atom("q"), Term.atom("r")
        s_or = Kernel.nd_assume(Term.disj(p, q))
        # Left branch: assume p, conclude r (cheat: assume r in context)
        s_pr = Kernel.nd_and_elim_l(
            Kernel.nd_assume(Term.conj(r, p))
        )
        # Left branch gamma must contain p — construct it differently.
        # We instead build: from (p), get (p ∧ p) via and_intro, then any conclusion
        # that's an assumption. Simpler: assume p, derive p (so r := p).
        s_left = Kernel.nd_assume(p)   # {p} ⊢ p
        s_right_base = Kernel.nd_assume(q)
        # Need both branches to conclude the same formula. Choose r = ⊤.
        s_left2 = Kernel.nd_top_intro()
        # But s_left2 has empty gamma, so {phi} doesn't contain p → kernel rejects.
        # So we use a richer construction with both branches concluding (p ∨ q):
        s_left_disj = Kernel.nd_or_intro_l(s_left, Term.disj(p, q))
        # Right branch: assume q, derive (p ∨ q).
        s_right_disj = Kernel.nd_or_intro_r(s_right_base, Term.disj(p, q))
        s_elim = Kernel.nd_or_elim(s_or, s_left_disj, s_right_disj)
        # Final conclusion: p ∨ q from {p ∨ q} alone (discharge p and q).
        assert s_elim.phi == Term.disj(p, q)
        assert s_elim.gamma == frozenset({Term.disj(p, q)})

    def test_repeat(self):
        p = Term.atom("p")
        s = Kernel.nd_assume(p)
        s2 = Kernel.nd_repeat(s)
        assert s2.phi == s.phi
        assert s2.gamma == s.gamma


# ---------------------------------------------------------------------------
# verify_natural_deduction — full proofs
# ---------------------------------------------------------------------------


class TestVerifyNaturalDeduction:
    def test_identity_p_implies_p(self):
        # Goal: p → p
        # Step 0: assume p; {p} ⊢ p
        # Step 1: imp_intro discharging p; ∅ ⊢ p → p
        p = Term.atom("p")
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_ASSUMPTION, conclusion=p),
            NaturalDeductionStep(
                rule=RULE_IMP_INTRO, premises=(0,), discharge=p,
                conclusion=Term.imp(p, p),
            ),
        ))
        rep = verify_natural_deduction([], Term.imp(p, p), proof)
        assert rep.status == VERIFIED
        assert rep.kernel_calls == 2

    def test_modus_ponens_proof(self):
        # Premises: p, p → q.  Goal: q.
        p, q = Term.atom("p"), Term.atom("q")
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=Term.imp(p, q)),
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=p),
            NaturalDeductionStep(
                rule=RULE_IMP_ELIM, premises=(0, 1), conclusion=q,
            ),
        ))
        rep = verify_natural_deduction([p, Term.imp(p, q)], q, proof)
        assert rep.status == VERIFIED

    def test_and_intro_then_elim(self):
        # Premises: p, q.  Goal: q ∧ p.
        p, q = Term.atom("p"), Term.atom("q")
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=q),
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=p),
            NaturalDeductionStep(
                rule=RULE_AND_INTRO, premises=(0, 1),
                conclusion=Term.conj(q, p),
            ),
        ))
        rep = verify_natural_deduction([p, q], Term.conj(q, p), proof)
        assert rep.status == VERIFIED

    def test_classical_lem(self):
        p = Term.atom("p")
        goal = Term.disj(p, Term.neg(p))
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_LEM, target=goal, conclusion=goal),
        ))
        rep = verify_natural_deduction([], goal, proof)
        assert rep.status == VERIFIED

    def test_intuitionistic_rejects_lem(self):
        p = Term.atom("p")
        goal = Term.disj(p, Term.neg(p))
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_LEM, target=goal, conclusion=goal),
        ))
        rep = verify_natural_deduction([], goal, proof, enforce_intuitionistic=True)
        assert rep.status == FAILED
        assert "classical" in rep.failure_reason

    def test_undischarged_assumption(self):
        # Try to prove p → p but forget to discharge.
        p = Term.atom("p")
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_ASSUMPTION, conclusion=p),
        ))
        rep = verify_natural_deduction([], p, proof)
        assert rep.status == FAILED
        # The kernel did derive {p} ⊢ p (matching goal p), but {p} is not a
        # premise so the leftover assumption is detected.
        assert "undischarged" in rep.failure_reason or "leftover" in rep.failure_reason.lower()

    def test_premise_not_in_set_rejected(self):
        # Cite p as a premise but only q is provided.
        p, q = Term.atom("p"), Term.atom("q")
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=p),
        ))
        rep = verify_natural_deduction([q], p, proof)
        assert rep.status == FAILED

    def test_wrong_goal(self):
        p, q = Term.atom("p"), Term.atom("q")
        # Prove p but the goal is q.
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=p),
        ))
        rep = verify_natural_deduction([p], q, proof)
        assert rep.status == FAILED
        assert "goal" in rep.failure_reason

    def test_modus_tollens_classical(self):
        # Classic proof: from (p → q) and ¬q, derive ¬p.
        # ND proof:
        #  0: premise (p → q)
        #  1: premise ¬q
        #  2: assume p
        #  3: imp_elim(0, 2) → q
        #  4: not_elim(3, 1) → ⊥
        #  5: not_intro(4, p) → ¬p
        p, q = Term.atom("p"), Term.atom("q")
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=Term.imp(p, q)),
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=Term.neg(q)),
            NaturalDeductionStep(rule=RULE_ASSUMPTION, conclusion=p),
            NaturalDeductionStep(rule=RULE_IMP_ELIM, premises=(0, 2), conclusion=q),
            NaturalDeductionStep(rule=RULE_NOT_ELIM, premises=(3, 1), conclusion=Term.bot()),
            NaturalDeductionStep(
                rule=RULE_NOT_INTRO, premises=(4,), discharge=p,
                conclusion=Term.neg(p),
            ),
        ))
        rep = verify_natural_deduction(
            [Term.imp(p, q), Term.neg(q)], Term.neg(p), proof,
        )
        assert rep.status == VERIFIED

    def test_empty_proof(self):
        p = Term.atom("p")
        rep = verify_natural_deduction([p], p, NaturalDeductionProof())
        assert rep.status == FAILED

    def test_iff_intro_then_use(self):
        p, q = Term.atom("p"), Term.atom("q")
        # Premises: p → q, q → p.  Goal: p ↔ q.
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=Term.imp(p, q)),
            NaturalDeductionStep(rule=RULE_PREMISE, conclusion=Term.imp(q, p)),
            NaturalDeductionStep(
                rule=RULE_IFF_INTRO, premises=(0, 1),
                conclusion=Term.iff(p, q),
            ),
        ))
        rep = verify_natural_deduction(
            [Term.imp(p, q), Term.imp(q, p)], Term.iff(p, q), proof,
        )
        assert rep.status == VERIFIED

    def test_kernel_call_count(self):
        p = Term.atom("p")
        proof = NaturalDeductionProof((
            NaturalDeductionStep(rule=RULE_ASSUMPTION, conclusion=p),
            NaturalDeductionStep(
                rule=RULE_IMP_INTRO, premises=(0,), discharge=p,
                conclusion=Term.imp(p, p),
            ),
        ))
        rep = verify_natural_deduction([], Term.imp(p, p), proof)
        assert rep.kernel_calls == len(proof)


# ---------------------------------------------------------------------------
# Equational rewriting
# ---------------------------------------------------------------------------


class TestEquational:
    def test_simple_rewrite(self):
        # Axiom: f(x) = g(x).  Prove f(a) = g(a).
        # Both sides are opaque atoms — no variables, no substitution.
        f_a = Term.atom("f(a)")
        g_a = Term.atom("g(a)")
        ax = EquationalAxiom(name="A1", lhs=f_a, rhs=g_a)
        proof = EquationalProof((
            RewriteStep(axiom_index=0, position=(), direction=REWRITE_FORWARD),
        ))
        rep = verify_equational([ax], f_a, g_a, proof)
        assert rep.status == VERIFIED
        assert rep.kernel_calls == 1

    def test_rewrite_with_variable(self):
        # Axiom: x ∧ x = x with variable x.
        # Prove: (p ∨ q) ∧ (p ∨ q) = (p ∨ q).
        x = Term.atom("x")
        ax = EquationalAxiom(
            name="idemp", lhs=Term.conj(x, x), rhs=x, variables=("x",),
        )
        pq = Term.disj(Term.atom("p"), Term.atom("q"))
        proof = EquationalProof((
            RewriteStep(
                axiom_index=0, position=(), direction=REWRITE_FORWARD,
                substitution=(("x", pq),),
            ),
        ))
        rep = verify_equational([ax], Term.conj(pq, pq), pq, proof)
        assert rep.status == VERIFIED

    def test_backward_rewrite(self):
        # Same axiom, prove the other direction: x = x ∧ x.
        x = Term.atom("x")
        ax = EquationalAxiom(
            name="idemp", lhs=Term.conj(x, x), rhs=x, variables=("x",),
        )
        p = Term.atom("p")
        proof = EquationalProof((
            RewriteStep(
                axiom_index=0, position=(), direction=REWRITE_BACKWARD,
                substitution=(("x", p),),
            ),
        ))
        rep = verify_equational([ax], p, Term.conj(p, p), proof)
        assert rep.status == VERIFIED

    def test_rewrite_subterm(self):
        # Axiom: a = b.  Prove: a ∨ c = b ∨ c by rewriting position (0,).
        a = Term.atom("a"); b = Term.atom("b"); c = Term.atom("c")
        ax = EquationalAxiom(name="A", lhs=a, rhs=b)
        proof = EquationalProof((
            RewriteStep(axiom_index=0, position=(0,), direction=REWRITE_FORWARD),
        ))
        rep = verify_equational([ax], Term.disj(a, c), Term.disj(b, c), proof)
        assert rep.status == VERIFIED

    def test_position_invalid(self):
        a = Term.atom("a"); b = Term.atom("b")
        ax = EquationalAxiom(name="A", lhs=a, rhs=b)
        # position 0 in an atom is invalid.
        proof = EquationalProof((
            RewriteStep(axiom_index=0, position=(0,), direction=REWRITE_FORWARD),
        ))
        rep = verify_equational([ax], a, b, proof)
        assert rep.status == FAILED

    def test_redex_mismatch(self):
        # Axiom claims to rewrite a → b but the current term has c at the position.
        a = Term.atom("a"); b = Term.atom("b"); c = Term.atom("c")
        ax = EquationalAxiom(name="A", lhs=a, rhs=b)
        proof = EquationalProof((
            RewriteStep(axiom_index=0, position=(), direction=REWRITE_FORWARD),
        ))
        rep = verify_equational([ax], c, b, proof)
        assert rep.status == FAILED

    def test_axiom_index_out_of_range(self):
        a = Term.atom("a")
        proof = EquationalProof((
            RewriteStep(axiom_index=5, position=(), direction=REWRITE_FORWARD),
        ))
        rep = verify_equational([], a, a, proof)
        assert rep.status == FAILED

    def test_undeclared_variable_in_substitution(self):
        ax = EquationalAxiom(name="A", lhs=Term.atom("a"), rhs=Term.atom("b"))
        # Axiom has no declared variables, but we bind "x".
        proof = EquationalProof((
            RewriteStep(
                axiom_index=0, position=(), direction=REWRITE_FORWARD,
                substitution=(("x", Term.atom("c")),),
            ),
        ))
        rep = verify_equational([ax], Term.atom("a"), Term.atom("b"), proof)
        assert rep.status == FAILED

    def test_multi_step_chain(self):
        # a = b, b = c → derive a = c.
        a = Term.atom("a"); b = Term.atom("b"); c = Term.atom("c")
        ax1 = EquationalAxiom(name="A1", lhs=a, rhs=b)
        ax2 = EquationalAxiom(name="A2", lhs=b, rhs=c)
        proof = EquationalProof((
            RewriteStep(axiom_index=0, position=(), direction=REWRITE_FORWARD),
            RewriteStep(axiom_index=1, position=(), direction=REWRITE_FORWARD),
        ))
        rep = verify_equational([ax1, ax2], a, c, proof)
        assert rep.status == VERIFIED


# ---------------------------------------------------------------------------
# VerifierConfig and report shape
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default(self):
        c = VerifierConfig()
        assert c.max_proof_length > 0
        assert c.fail_fast is True

    def test_bad_hmac(self):
        with pytest.raises(InvalidConfig):
            VerifierConfig(hmac_key="not bytes")  # type: ignore[arg-type]

    def test_bad_max_proof_length(self):
        with pytest.raises(InvalidConfig):
            VerifierConfig(max_proof_length=0)
        with pytest.raises(InvalidConfig):
            VerifierConfig(max_proof_length=-1)

    def test_bad_max_term_depth(self):
        with pytest.raises(InvalidConfig):
            VerifierConfig(max_term_depth=0)


class TestReportShape:
    def test_verified_is_true(self):
        f = CNFFormula.of([[1], [-1]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
        ))
        rep = verify_resolution(f, proof)
        assert rep.verified is True

    def test_as_dict(self):
        f = CNFFormula.of([[1], [-1]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
        ))
        rep = verify_resolution(f, proof)
        d = rep.as_dict()
        assert d["status"] == VERIFIED
        assert d["kind"] == KIND_RESOLUTION
        assert isinstance(d["certificate"], str)

    def test_trace_recorded(self):
        f = CNFFormula.of([[1], [-1]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
        ))
        rep = verify_resolution(f, proof, record_trace=True)
        assert len(rep.trace) == 1

    def test_trace_disabled(self):
        f = CNFFormula.of([[1], [-1]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=()),
        ))
        rep = verify_resolution(f, proof, record_trace=False)
        assert rep.trace == ()


# ---------------------------------------------------------------------------
# Trusted-base introspection
# ---------------------------------------------------------------------------


class TestTCB:
    def test_kernel_rule_count(self):
        # 1 resolution + len(ND rules) + 1 rewrite
        assert kernel_rule_count() == 1 + len(VERIFIER_KNOWN_ND_RULES) + 1

    def test_tcb_summary(self):
        s = tcb_summary()
        assert s["kernel_rule_count"] == kernel_rule_count()
        assert "nd_rules" in s
        assert RULE_IMP_ELIM in s["nd_rules"]

    def test_known_constants(self):
        assert VERIFIED in VERIFIER_KNOWN_STATUSES
        assert FAILED in VERIFIER_KNOWN_STATUSES
        assert MALFORMED in VERIFIER_KNOWN_STATUSES
        assert KIND_RESOLUTION in VERIFIER_KNOWN_KINDS
        assert KIND_NATURAL_DEDUCTION in VERIFIER_KNOWN_KINDS
        assert KIND_EQUATIONAL in VERIFIER_KNOWN_KINDS
        assert len(VERIFIER_KNOWN_EVENTS) >= 5


# ---------------------------------------------------------------------------
# Integration: Reasoner-style proof verified by Verifier
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_reasoner_emitted_resolution_proof_verified(self):
        """A coordination engine could pipe Reasoner's
        last_resolution_proof() into Verifier; this test exercises
        the contract that Reasoner promises a tuple-of-(parents, pivot,
        resolvent) and Verifier consumes exactly that shape.
        """
        # Build a tiny unsat instance: (a ∨ b), (¬a ∨ c), (¬b ∨ c), (¬c).
        # Resolution proof:
        #   (a ∨ b), (¬a ∨ c) on a → (b ∨ c)            (index 4)
        #   (b ∨ c), (¬b ∨ c)  on b → (c)               (index 5)
        #   (c), (¬c)          on c → ⊥                 (index 6)
        f = CNFFormula.of([[1, 2], [-1, 3], [-2, 3], [-3]])
        proof = ResolutionProof((
            ResolutionStep(parents=(0, 1), pivot=1, resolvent=(2, 3)),
            ResolutionStep(parents=(4, 2), pivot=2, resolvent=(3,)),
            ResolutionStep(parents=(5, 3), pivot=3, resolvent=()),
        ))
        rep = verify_resolution(f, proof, hmac_key=b"runtime")
        assert rep.status == VERIFIED
        assert rep.n_steps == 3
        assert rep.certificate

    def test_intuitionistic_proof_constructive(self):
        """The deduction theorem variant ((p → q) ∧ p) → q is
        constructively provable; Verifier accepts it under
        intuitionistic mode.
        """
        p, q = Term.atom("p"), Term.atom("q")
        goal = Term.imp(Term.conj(Term.imp(p, q), p), q)
        proof = NaturalDeductionProof((
            NaturalDeductionStep(
                rule=RULE_ASSUMPTION,
                conclusion=Term.conj(Term.imp(p, q), p),
            ),
            NaturalDeductionStep(
                rule=RULE_AND_ELIM_L, premises=(0,),
                conclusion=Term.imp(p, q),
            ),
            NaturalDeductionStep(
                rule=RULE_AND_ELIM_R, premises=(0,),
                conclusion=p,
            ),
            NaturalDeductionStep(
                rule=RULE_IMP_ELIM, premises=(1, 2), conclusion=q,
            ),
            NaturalDeductionStep(
                rule=RULE_IMP_INTRO, premises=(3,),
                discharge=Term.conj(Term.imp(p, q), p),
                conclusion=goal,
            ),
        ))
        rep = verify_natural_deduction(
            [], goal, proof, enforce_intuitionistic=True,
        )
        assert rep.status == VERIFIED
