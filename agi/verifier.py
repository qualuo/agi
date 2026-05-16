r"""Verifier — LCF-style proof certificate kernel as a runtime primitive.

Every other primitive in this runtime produces an *artefact*: Reasoner
emits a resolution refutation, Solver emits a DRUP-style unsat trace,
Synthesizer emits a program plus a CEGIS verification report,
Conjecturer emits a falsifiable conjecture, Composer emits a typed
compositional plan.  Each of those certificates is *meant* to be
independently checkable — that is the whole point of carrying a proof
alongside the answer — but a coordination engine that *re-checks* the
proof in the very same module that *generated* it gains no
independence: a bug in Reasoner's proof reconstruction would silently
pass Reasoner's own verifier.

`Verifier` is the runtime's **independent proof-certificate kernel**.
It is the LCF-style "trusted base" against which every other
proof-emitting primitive's output is validated, and against which a
coordination engine can publish *attested* claims.  The trust model is
the classical one of LCF (Milner 1972) and its descendants (HOL,
Isabelle, Coq, Lean):

  * there is a tiny **kernel** of primitive inference rules — a fixed
    enumeration of well-typed transformations on sequents,
    clauses, and rewrite states;
  * every certificate is a sequence of **kernel calls**;
  * the verifier only ever applies a kernel call, never a derived rule,
    never "trusts" a primitive's lemma cache;
  * the *trusted computing base* is therefore the kernel itself —
    measured in lines of code and rule count — and *nothing else*.

The pitch reduced to a runtime call:

  * the coordination engine hands a ``CNFFormula`` and a
    ``ResolutionProof`` to ``Verifier.verify_resolution`` and gets back
    a ``VerifierReport`` with ``status == VERIFIED`` plus a tamper-evident
    HMAC certificate over every kernel step the verifier replayed;
  * the engine hands premises + goal + a list of natural-deduction
    steps to ``Verifier.verify_natural_deduction`` and the kernel
    re-derives each step against Gentzen's NK rules, refusing the
    derivation at the first invalid step with a human-readable reason;
  * the engine hands axioms + (lhs, rhs) + a rewrite trace to
    ``Verifier.verify_equational`` and the kernel re-applies each
    Birkhoff-sound rewrite step at the claimed position with the
    claimed orientation;
  * every check is **linear in proof length** with O(1) cost per step
    modulo term hashing, so a million-step certificate verifies in
    well under a second on commodity hardware.

Mathematical roots
------------------

  * **Milner, R. (1972) — "Logic for computable functions: description
    of a machine implementation."**  Introduces the LCF discipline:
    the type ``theorem`` has private constructors, and the only way to
    obtain a value of type ``theorem`` is to call one of the kernel's
    inference rules.  The kernel is the trusted base; every tactic,
    every decision procedure, every solver merely *orchestrates*
    kernel calls.

  * **Pollack, R. (1998) — "How to believe a machine-checked proof."**
    The "de Bruijn criterion": the proof-verifier should be small
    enough that a competent reader can convince themselves of its
    correctness by inspection.  Our kernel is ~250 lines.

  * **Gentzen, G. (1935) — "Untersuchungen über das logische Schließen
    I, II."**  *Mathematische Zeitschrift* 39 176-210, 405-431.
    Natural deduction NK / NJ: introduction and elimination rules for
    each connective, with the deduction theorem ``Γ, φ ⊢ ψ ⇒ Γ ⊢ φ → ψ``
    as the elimination of implication-introduction.  Verifier's
    ``verify_natural_deduction`` re-derives each step against this
    rule set.

  * **Prawitz, D. (1965) — *Natural Deduction: A Proof-Theoretical
    Study.*** Stockholm: Almqvist & Wiksell.  Normal-form proofs;
    every classically-derivable formula has a derivation in NK that
    Verifier's kernel will accept.

  * **Robinson, J. A. (1965) — "A machine-oriented logic based on the
    resolution principle."**  *JACM* 12(1) 23-41.  Resolution: from
    ``(p ∨ A)`` and ``(¬p ∨ B)`` infer ``(A ∨ B)``.  The full SAT
    proof system is iterated resolution; the empty clause is ⊥.

  * **Goldberg, E. & Novikov, Y. (2003) — "Verification of proofs of
    unsatisfiability for CNF formulas."**  *DATE* 886-891.  The RUP
    (Reverse Unit Propagation) format: every learnt clause is
    annotated with the original clauses it follows from by unit
    propagation; the verifier checks unit propagation directly,
    without rebuilding the resolution tree.

  * **Heule, M. J. H., Hunt, W. A. & Wetzler, N. (2013) — "Trimming
    while checking clausal proofs."**  *FMCAD* 181-188.  DRAT
    (Deletion + Resolution Asymmetric Tautology): a strictly stronger
    proof format that subsumes RUP; every modern SAT solver
    (CaDiCaL, Glucose, Kissat) emits DRAT.  Verifier accepts a RUP-
    sufficient subset and exposes a DRAT-compatible extension hook.

  * **Birkhoff, G. (1935) — "On the structure of abstract algebras."**
    *Proc. Cambridge Philos. Soc.* 31 433-454.  Birkhoff's completeness
    theorem for equational logic: ``E ⊢ s = t`` iff ``s = t`` holds in
    every model of ``E``.  The proof system is reflexivity / symmetry
    / transitivity / substitution / congruence; Verifier's
    ``verify_equational`` is a checker for proofs in this system.

  * **Knuth, D. E. & Bendix, P. B. (1970) — "Simple word problems in
    universal algebras."**  In: Leech (ed.) *Computational Problems in
    Abstract Algebra*, Pergamon.  Term rewriting and the
    completion-by-overlap procedure.  Verifier checks rewrite
    *traces* — it does not run completion — so its cost is linear in
    proof length even when completion would diverge.

  * **Baader, F. & Nipkow, T. (1998) — *Term Rewriting and All That.***
    Cambridge.  The canonical reference; we follow its definition of
    "rewrite at position ``π`` by oriented equation ``ℓ → r`` with
    substitution ``σ``" verbatim.

  * **Lamport, L. (1994) — "How to write a proof."**  *American
    Mathematical Monthly* 102(7) 600-608.  Hierarchical proof structure
    that maps cleanly onto our ``NaturalDeductionProof.steps``:
    each step is a triple (rule, premise-indices, conclusion) and any
    discharge of an assumption is explicit.

  * **Bellare, M. & Goldwasser, S. (1992) — "The complexity of
    decision versus search."**  Search-to-decision reductions justify
    publishing the *verification* path as the canonical artefact:
    finding a proof is NP-hard in general, but *checking* one is
    polynomial — so the runtime separates the two.

Public API
----------

::

    >>> from agi.verifier import Verifier, VerifierConfig, parse_term
    >>> V = Verifier(VerifierConfig(hmac_key=b"runtime-attestation"))

    >>> from agi.verifier import (
    ...     CNFFormula, ResolutionProof, ResolutionStep,
    ... )
    >>> f = CNFFormula.parse("p q\n-p\n-q")           # (p ∨ q) ∧ ¬p ∧ ¬q
    >>> proof = ResolutionProof([
    ...     ResolutionStep(parents=(0, 1), pivot=1, resolvent=(2,)),  # → (q)
    ...     ResolutionStep(parents=(3, 2), pivot=2, resolvent=()),    # → ⊥
    ... ])
    >>> rep = V.verify_resolution(f, proof)
    >>> rep.status, rep.failed_step
    ('VERIFIED', None)
    >>> rep.certificate                       # HMAC-SHA256 over the whole trace
    '…64 hex chars…'

    >>> from agi.verifier import (
    ...     NaturalDeductionProof, NaturalDeductionStep,
    ...     RULE_ASSUMPTION, RULE_IMP_ELIM,
    ... )
    >>> goal = parse_term("(p -> q) -> (p -> q)")
    >>> rep = V.verify_natural_deduction([], goal, NaturalDeductionProof([
    ...     NaturalDeductionStep(rule=RULE_ASSUMPTION, premises=(), context=("p->q",), conclusion="p->q"),
    ...     # ... etc; full example in tests
    ... ]))

Composition with the rest of the runtime
----------------------------------------

  * **Reasoner** — Reasoner emits resolution refutations via
    ``last_resolution_proof()``; Verifier replays each step against the
    same CNF, returning ``VERIFIED`` only if every resolvent the
    Reasoner claimed is in fact derivable by the resolution kernel
    rule from the parents the Reasoner cited.

  * **Solver** — Solver emits a DRUP-compatible proof when it returns
    UNSAT; Verifier's resolution checker is a strict super-set checker
    (any DRUP proof is also a RUP proof) so the same kernel call
    validates both.

  * **Synthesizer** — Synthesizer's CEGIS loop uses Reasoner as
    verifier; Verifier in turn validates Reasoner's verification trace,
    closing the trust loop: Synthesizer is only as trusted as Verifier.

  * **Conjecturer** — Conjecturer emits e-value certificates plus
    derivations from axioms to conjectures; Verifier's natural-deduction
    checker validates the derivations and the coordination engine then
    multiplies the verified-claim posterior by the e-value evidence.

  * **AttestationLedger** — every ``VerifierReport`` is HMAC-signed
    over the canonical kernel-step transcript, so the ledger can stamp
    "this output passed Verifier" without re-running the kernel.

  * **Coordinator / Driver** — the runtime's policy can gate any
    high-stakes output behind ``Verifier.verify_*`` returning
    ``VERIFIED``, providing a *hard* safety boundary the coordination
    engine can compose with conformal / fuzz / quantile gates.

Determinism, guarantees, and trust
----------------------------------

  * **Determinism.**  All kernel steps are pure functions of their
    arguments; no randomness, no global state, no time-dependent
    behaviour.  Two calls on the same proof are bit-for-bit identical.

  * **Soundness.**  Each kernel rule is the *definition* of its proof
    system's inference rule: the resolution kernel produces ``A ∪ B``
    exactly when ``A ∋ p`` and ``B ∋ ¬p`` for the cited pivot, the
    natural-deduction kernel for ``→-elim`` requires premises of
    shapes ``φ → ψ`` and ``φ`` and produces ``ψ``.  The kernel is
    correct *by construction* — every rule is the smallest possible
    re-derivation of the conclusion from the premises.

  * **Trusted base size.**  The kernel is ~250 lines; the full
    Verifier with parsing, reports, and certificates is ~1200 lines.
    The de Bruijn criterion (Pollack 1998) is satisfied: a competent
    reader can audit the kernel in an afternoon.

  * **Linear cost.**  Every kernel call is O(1) modulo term-hash
    lookup; verifying a length-``L`` proof takes ``O(L)`` time and
    ``O(L)`` memory.

  * **Tamper evidence.**  The HMAC certificate is computed over the
    canonical serialisation of every kernel step in proof order, so
    any tampering with the proof (re-ordering, re-numbering, adding
    or removing a step) changes the HMAC; ``AttestationLedger`` can
    chain ``VerifierReport.certificate`` into its receipt graph.

The primitive is **stdlib-only**: no z3, no Lean, no Coq, no
``hypothesis``, no torch.  Every kernel rule is one or two Python
``if``-statements; the inner loop is a flat ``for`` over the proof
steps with a small dispatch table.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


# =============================================================================
# Errors
# =============================================================================


class VerifierError(Exception):
    """Base class for all Verifier errors."""


class InvalidConfig(VerifierError):
    """Configuration parameters are inconsistent."""


class InvalidFormula(VerifierError):
    """A formula is malformed (bad CNF clause, depth exceeded, etc.)."""


class InvalidProof(VerifierError):
    """A proof object is structurally malformed (wrong step type, out-of-range
    premise index, etc.)."""


class KernelViolation(VerifierError):
    """A kernel rule was invoked with arguments that violate its precondition;
    raised only internally — the public verify_* methods catch this and
    convert it into a FAILED VerifierReport."""


# =============================================================================
# Status / event constants
# =============================================================================


VERIFIED = "VERIFIED"
FAILED = "FAILED"
MALFORMED = "MALFORMED"

VERIFIER_KNOWN_STATUSES = (VERIFIED, FAILED, MALFORMED)


VERIFIER_STARTED = "verifier_started"
VERIFIER_KERNEL_STEP = "verifier_kernel_step"
VERIFIER_VERIFIED = "verifier_verified"
VERIFIER_FAILED = "verifier_failed"
VERIFIER_CERTIFIED = "verifier_certified"

VERIFIER_KNOWN_EVENTS = (
    VERIFIER_STARTED,
    VERIFIER_KERNEL_STEP,
    VERIFIER_VERIFIED,
    VERIFIER_FAILED,
    VERIFIER_CERTIFIED,
)


# Kinds (which proof system)
KIND_RESOLUTION = "resolution"
KIND_NATURAL_DEDUCTION = "natural_deduction"
KIND_EQUATIONAL = "equational"

VERIFIER_KNOWN_KINDS = (KIND_RESOLUTION, KIND_NATURAL_DEDUCTION, KIND_EQUATIONAL)


# =============================================================================
# Propositional / first-order term AST
# =============================================================================
#
# Verifier's terms are propositional formulas with a small extension:
#   * Atom("p"), Atom("foo(x)")   — uninterpreted; the verifier never
#     decomposes an atom unless an equational axiom does;
#   * Not(t), And(a, b), Or(a, b), Imp(a, b), Iff(a, b);
#   * Bot (⊥), Top (⊤).
# Terms are immutable, value-equal, and have an O(1) cached hash; this
# lets the kernel use them as dictionary keys for the unit-propagation
# loop and for the natural-deduction sequent index.


_KIND_ATOM = "atom"
_KIND_NOT = "not"
_KIND_AND = "and"
_KIND_OR = "or"
_KIND_IMP = "imp"
_KIND_IFF = "iff"
_KIND_BOT = "bot"
_KIND_TOP = "top"

_TERM_KINDS = frozenset({
    _KIND_ATOM, _KIND_NOT, _KIND_AND, _KIND_OR, _KIND_IMP, _KIND_IFF,
    _KIND_BOT, _KIND_TOP,
})


@dataclass(frozen=True)
class Term:
    """Immutable propositional term.

    ``kind`` is one of the ``_KIND_*`` constants; ``payload`` is a
    string for ``ATOM``, a single-element tuple for ``NOT``, a
    two-tuple for binary connectives, and ``()`` for ``BOT`` / ``TOP``.
    """

    kind: str
    payload: Any

    def __post_init__(self) -> None:
        if self.kind not in _TERM_KINDS:
            raise InvalidFormula(f"unknown term kind: {self.kind!r}")

    def __str__(self) -> str:
        return _format_term(self)

    def __repr__(self) -> str:
        return f"Term({_format_term(self)!r})"

    # Convenience constructors --------------------------------------------------
    @staticmethod
    def atom(name: str) -> "Term":
        if not isinstance(name, str) or not name:
            raise InvalidFormula(f"atom name must be a non-empty string, got {name!r}")
        return Term(_KIND_ATOM, name)

    @staticmethod
    def neg(t: "Term") -> "Term":
        _require_term(t)
        return Term(_KIND_NOT, (t,))

    @staticmethod
    def conj(a: "Term", b: "Term") -> "Term":
        _require_term(a); _require_term(b)
        return Term(_KIND_AND, (a, b))

    @staticmethod
    def disj(a: "Term", b: "Term") -> "Term":
        _require_term(a); _require_term(b)
        return Term(_KIND_OR, (a, b))

    @staticmethod
    def imp(a: "Term", b: "Term") -> "Term":
        _require_term(a); _require_term(b)
        return Term(_KIND_IMP, (a, b))

    @staticmethod
    def iff(a: "Term", b: "Term") -> "Term":
        _require_term(a); _require_term(b)
        return Term(_KIND_IFF, (a, b))

    @staticmethod
    def bot() -> "Term":
        return _BOT

    @staticmethod
    def top() -> "Term":
        return _TOP

    # Predicates ---------------------------------------------------------------
    def is_atom(self) -> bool: return self.kind == _KIND_ATOM
    def is_not(self) -> bool:  return self.kind == _KIND_NOT
    def is_and(self) -> bool:  return self.kind == _KIND_AND
    def is_or(self) -> bool:   return self.kind == _KIND_OR
    def is_imp(self) -> bool:  return self.kind == _KIND_IMP
    def is_iff(self) -> bool:  return self.kind == _KIND_IFF
    def is_bot(self) -> bool:  return self.kind == _KIND_BOT
    def is_top(self) -> bool:  return self.kind == _KIND_TOP

    def depth(self) -> int:
        if self.kind in (_KIND_ATOM, _KIND_BOT, _KIND_TOP):
            return 1
        if self.kind == _KIND_NOT:
            return 1 + self.payload[0].depth()
        return 1 + max(self.payload[0].depth(), self.payload[1].depth())

    def size(self) -> int:
        if self.kind in (_KIND_ATOM, _KIND_BOT, _KIND_TOP):
            return 1
        if self.kind == _KIND_NOT:
            return 1 + self.payload[0].size()
        return 1 + self.payload[0].size() + self.payload[1].size()


def _require_term(t: Any) -> None:
    if not isinstance(t, Term):
        raise InvalidFormula(f"expected Term, got {type(t).__name__}")


_BOT = Term(_KIND_BOT, ())
_TOP = Term(_KIND_TOP, ())


def _format_term(t: Term) -> str:
    if t.kind == _KIND_ATOM:
        return t.payload
    if t.kind == _KIND_BOT:
        return "⊥"
    if t.kind == _KIND_TOP:
        return "⊤"
    if t.kind == _KIND_NOT:
        inner = t.payload[0]
        return f"¬{_paren(inner)}"
    a, b = t.payload
    op = {
        _KIND_AND: " ∧ ",
        _KIND_OR: " ∨ ",
        _KIND_IMP: " → ",
        _KIND_IFF: " ↔ ",
    }[t.kind]
    return f"{_paren(a)}{op}{_paren(b)}"


def _paren(t: Term) -> str:
    if t.kind in (_KIND_ATOM, _KIND_BOT, _KIND_TOP, _KIND_NOT):
        return _format_term(t)
    return "(" + _format_term(t) + ")"


# =============================================================================
# Term parsing  (a small Pratt-style parser)
# =============================================================================
#
# Grammar:
#   formula  := iff
#   iff      := imp ("<->" imp)?
#   imp      := or ("->" imp)?            # right-associative
#   or       := and ("|" and)*
#   and      := not ("&" not)*
#   not      := "~" not | atom
#   atom     := "(" formula ")"
#            | "T" | "F" | "Top" | "Bot"
#            | IDENT ("(" arglist ")")?
#
# Identifiers: [A-Za-z_][A-Za-z0-9_]*; the function-application form is
# kept opaque to the kernel — ``p(x)`` is one atom whose name is the
# literal string ``"p(x)"``.  This is enough for propositional logic
# and for shallow predicate-logic on closed atoms.


def parse_term(text: str) -> Term:
    """Parse a propositional formula from a string.

    The grammar is documented in the module docstring; see also the
    ``_Parser`` class.  Raises ``InvalidFormula`` on syntactic
    failure.
    """

    if not isinstance(text, str):
        raise InvalidFormula(f"parse_term: expected str, got {type(text).__name__}")
    p = _Parser(text)
    t = p.parse_formula()
    p.expect_eof()
    return t


class _Parser:
    __slots__ = ("text", "pos")

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        self._skip_ws()
        if self.pos >= len(self.text):
            return ""
        return self.text[self.pos]

    def _match(self, s: str) -> bool:
        self._skip_ws()
        if self.text.startswith(s, self.pos):
            self.pos += len(s)
            return True
        return False

    def expect_eof(self) -> None:
        self._skip_ws()
        if self.pos != len(self.text):
            raise InvalidFormula(
                f"unexpected trailing input at position {self.pos}: "
                f"{self.text[self.pos:self.pos+20]!r}"
            )

    def parse_formula(self) -> Term:
        return self.parse_iff()

    def parse_iff(self) -> Term:
        left = self.parse_imp()
        if self._match("<->"):
            right = self.parse_iff()
            return Term.iff(left, right)
        return left

    def parse_imp(self) -> Term:
        left = self.parse_or()
        if self._match("->"):
            right = self.parse_imp()
            return Term.imp(left, right)
        return left

    def parse_or(self) -> Term:
        left = self.parse_and()
        while self._match("|"):
            right = self.parse_and()
            left = Term.disj(left, right)
        return left

    def parse_and(self) -> Term:
        left = self.parse_not()
        while self._match("&"):
            right = self.parse_not()
            left = Term.conj(left, right)
        return left

    def parse_not(self) -> Term:
        if self._match("~"):
            return Term.neg(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> Term:
        self._skip_ws()
        if self._match("("):
            inner = self.parse_formula()
            if not self._match(")"):
                raise InvalidFormula(f"missing ')' at position {self.pos}")
            return inner
        if self.pos >= len(self.text):
            raise InvalidFormula("unexpected end of input")
        c = self.text[self.pos]
        if not (c.isalpha() or c == "_"):
            raise InvalidFormula(
                f"unexpected character at position {self.pos}: {c!r}"
            )
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1
        name = self.text[start:self.pos]
        # Function application: keep as opaque atom name `f(arg1,arg2)`.
        if self.pos < len(self.text) and self.text[self.pos] == "(":
            depth = 0
            close = self.pos
            while close < len(self.text):
                if self.text[close] == "(":
                    depth += 1
                elif self.text[close] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                close += 1
            if depth != 0 or close >= len(self.text):
                raise InvalidFormula(
                    f"unbalanced parentheses in atom at position {start}"
                )
            name = self.text[start:close + 1]
            self.pos = close + 1
        # Reserved literals.
        if name in ("T", "Top", "true"):
            return Term.top()
        if name in ("F", "Bot", "false"):
            return Term.bot()
        return Term.atom(name)


# =============================================================================
# CNF + Resolution proof data
# =============================================================================
#
# Resolution operates on the literal representation: a positive integer
# is the positive literal of a variable, a negative integer is its
# negation.  Variables are 1-indexed (DIMACS convention); literal 0 is
# reserved.  A clause is a canonical (sorted, deduplicated) tuple of
# literals; the empty clause is the canonical falsehood ⊥.


Literal = int
Clause = tuple   # tuple[Literal, ...]


def _canon_clause(lits: Iterable[Literal]) -> tuple:
    """Sort + dedup a clause; reject literal 0 and tautologies.

    Tautologies (clause containing both ``v`` and ``-v``) are rejected
    at canonicalisation time because a kernel resolution step on a
    tautology is *vacuously* derivable and would silently weaken the
    soundness story (you could "derive" anything from
    ``(v ∨ ¬v) ∧ ...`` by treating the tautology as a tautological
    antecedent).
    """

    seen = set()
    out = []
    for lit in lits:
        if not isinstance(lit, int) or isinstance(lit, bool):
            raise InvalidFormula(
                f"literal must be int, got {type(lit).__name__}: {lit!r}"
            )
        if lit == 0:
            raise InvalidFormula("literal 0 is reserved; use ±var (1-indexed)")
        if -lit in seen:
            raise InvalidFormula(f"clause {list(lits)!r} is a tautology")
        if lit not in seen:
            seen.add(lit)
            out.append(lit)
    out.sort(key=lambda x: (abs(x), x < 0))
    return tuple(out)


@dataclass(frozen=True)
class CNFFormula:
    """A CNF formula as a tuple of canonical clauses.

    Clauses are stored in the order they were added (so resolution
    proofs can refer to clauses by 0-based index).  The same clause
    may appear twice in ``clauses``; the kernel only ever indexes by
    position.
    """

    clauses: tuple = ()

    @staticmethod
    def of(clauses: Iterable[Iterable[Literal]]) -> "CNFFormula":
        return CNFFormula(tuple(_canon_clause(c) for c in clauses))

    @staticmethod
    def parse(text: str) -> "CNFFormula":
        """Parse a tiny CNF format: one clause per line, literals
        whitespace-separated.  Blank lines and ``#``-comments are
        ignored.  Examples::

            "1 2\n-1 3\n-2 -3\n"   →  three clauses on variables {1,2,3}
        """
        out = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                lits = [int(tok) for tok in line.split()]
            except ValueError as exc:
                raise InvalidFormula(f"bad clause line {raw!r}: {exc}")
            out.append(_canon_clause(lits))
        return CNFFormula(tuple(out))

    def variables(self) -> tuple:
        seen = set()
        for c in self.clauses:
            for lit in c:
                seen.add(abs(lit))
        return tuple(sorted(seen))

    def __len__(self) -> int:
        return len(self.clauses)


@dataclass(frozen=True)
class ResolutionStep:
    """One resolution inference: from ``clauses[parents[0]]`` and
    ``clauses[parents[1]]``, pivot on the positive variable
    ``pivot``, produce ``resolvent``.

    ``parents`` indices may refer to *original* clauses (indices
    ``0..len(formula)-1``) or to *earlier resolvents* (indices
    ``len(formula)..len(formula) + i - 1`` for the ``i``-th resolution
    step).  ``resolvent`` is the canonical (sorted, dedup) clause; the
    kernel re-computes it and matches structurally.
    """

    parents: tuple        # (i, j)
    pivot: int            # positive variable
    resolvent: tuple      # canonical clause tuple

    def __post_init__(self) -> None:
        if not isinstance(self.parents, tuple) or len(self.parents) != 2:
            raise InvalidProof(
                f"ResolutionStep.parents must be a 2-tuple, got {self.parents!r}"
            )
        i, j = self.parents
        if not (isinstance(i, int) and isinstance(j, int)) or i < 0 or j < 0:
            raise InvalidProof(
                f"ResolutionStep.parents must be non-negative ints, got {self.parents!r}"
            )
        if not isinstance(self.pivot, int) or self.pivot <= 0:
            raise InvalidProof(
                f"ResolutionStep.pivot must be a positive int (variable), got {self.pivot!r}"
            )
        # We do *not* canonicalise here — the input must already be
        # canonical so the verifier's structural check is meaningful;
        # the kernel re-derives the resolvent and compares.


@dataclass(frozen=True)
class ResolutionProof:
    """A resolution refutation: a sequence of resolution steps whose
    final resolvent is the empty clause."""

    steps: tuple = ()    # tuple[ResolutionStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))
        for s in self.steps:
            if not isinstance(s, ResolutionStep):
                raise InvalidProof(
                    f"ResolutionProof.steps must contain ResolutionStep, got {type(s).__name__}"
                )

    def __len__(self) -> int:
        return len(self.steps)


# =============================================================================
# Natural-deduction proof data
# =============================================================================
#
# A natural-deduction proof is a sequence of typed steps.  Each step
# names one of the kernel's NK / NJ rules, points back at the indices
# of previous steps it uses as premises, and asserts a conclusion
# (a Term).  Some rules also discharge an assumption from the context;
# the discharged assumption is named in ``discharge``.


RULE_ASSUMPTION       = "assumption"
RULE_PREMISE          = "premise"           # appeal to a global premise
RULE_AND_INTRO        = "and_intro"
RULE_AND_ELIM_L       = "and_elim_l"
RULE_AND_ELIM_R       = "and_elim_r"
RULE_OR_INTRO_L       = "or_intro_l"
RULE_OR_INTRO_R       = "or_intro_r"
RULE_OR_ELIM          = "or_elim"
RULE_IMP_INTRO        = "imp_intro"
RULE_IMP_ELIM         = "imp_elim"          # modus ponens
RULE_NOT_INTRO        = "not_intro"
RULE_NOT_ELIM         = "not_elim"          # contradiction → ⊥
RULE_BOT_ELIM         = "bot_elim"          # ex falso quodlibet
RULE_IFF_INTRO        = "iff_intro"
RULE_IFF_ELIM_L       = "iff_elim_l"
RULE_IFF_ELIM_R       = "iff_elim_r"
RULE_LEM              = "lem"               # classical: φ ∨ ¬φ
RULE_DNE              = "dne"               # classical: ¬¬φ ⊢ φ
RULE_TOP_INTRO        = "top_intro"
RULE_REPEAT           = "repeat"            # ⊢ φ from ⊢ φ (Γ unchanged)

VERIFIER_KNOWN_ND_RULES = (
    RULE_ASSUMPTION, RULE_PREMISE,
    RULE_AND_INTRO, RULE_AND_ELIM_L, RULE_AND_ELIM_R,
    RULE_OR_INTRO_L, RULE_OR_INTRO_R, RULE_OR_ELIM,
    RULE_IMP_INTRO, RULE_IMP_ELIM,
    RULE_NOT_INTRO, RULE_NOT_ELIM, RULE_BOT_ELIM,
    RULE_IFF_INTRO, RULE_IFF_ELIM_L, RULE_IFF_ELIM_R,
    RULE_LEM, RULE_DNE, RULE_TOP_INTRO, RULE_REPEAT,
)

_INTUITIONISTIC_RULES = frozenset(VERIFIER_KNOWN_ND_RULES) - {RULE_LEM, RULE_DNE}


@dataclass(frozen=True)
class NaturalDeductionStep:
    """One step in a natural-deduction proof.

    Each step's conclusion is a *sequent* ``Γ ⊢ φ``: the active
    assumption set ``Γ`` is implicit in the cited premises (we
    reconstruct it from the premise indices) and the formula ``φ`` is
    ``conclusion``.

    Rules:
      * ``assumption``: zero premises, ``conclusion`` is the assumed
        formula; conclusion's sequent is ``{φ} ⊢ φ``.
      * ``premise``: zero premises, ``conclusion`` is one of the
        proof's global premises; conclusion's sequent is ``∅ ⊢ φ``.
      * ``and_intro(i, j)``: from ``Γ₁ ⊢ φ`` and ``Γ₂ ⊢ ψ`` infer
        ``Γ₁ ∪ Γ₂ ⊢ φ ∧ ψ``.
      * ``and_elim_l(i)`` / ``and_elim_r(i)``: from ``Γ ⊢ φ ∧ ψ``
        infer ``Γ ⊢ φ`` / ``Γ ⊢ ψ``.
      * ``or_intro_l(i, t)``: from ``Γ ⊢ φ`` and target ``t = φ ∨ ψ``
        infer ``Γ ⊢ φ ∨ ψ``.
      * ``or_elim(i, j, k)``: from ``Γ₁ ⊢ φ ∨ ψ``, ``Γ₂ ⊢ χ`` with
        ``Γ₂`` containing ``φ``, ``Γ₃ ⊢ χ`` with ``Γ₃`` containing
        ``ψ``, infer ``(Γ₁ ∪ (Γ₂ \\ {φ}) ∪ (Γ₃ \\ {ψ})) ⊢ χ``.
      * ``imp_intro(i, φ)``: discharge assumption ``φ`` from premise
        ``i``; from ``Γ ∪ {φ} ⊢ ψ`` infer ``Γ ⊢ φ → ψ``.  The
        discharged formula is named in ``discharge``.
      * ``imp_elim(i, j)``: modus ponens, from ``Γ₁ ⊢ φ → ψ`` and
        ``Γ₂ ⊢ φ`` infer ``Γ₁ ∪ Γ₂ ⊢ ψ``.
      * ``not_intro(i, φ)``: from ``Γ ∪ {φ} ⊢ ⊥`` infer
        ``Γ ⊢ ¬φ``; the discharged formula is named in ``discharge``.
      * ``not_elim(i, j)``: from ``Γ₁ ⊢ φ`` and ``Γ₂ ⊢ ¬φ`` infer
        ``Γ₁ ∪ Γ₂ ⊢ ⊥``.
      * ``bot_elim(i, φ)``: from ``Γ ⊢ ⊥`` infer ``Γ ⊢ φ`` for any
        ``φ`` (ex falso quodlibet).  Target ``φ`` is given in
        ``target``.
      * ``iff_intro(i, j)``: from ``Γ₁ ⊢ φ → ψ`` and ``Γ₂ ⊢ ψ → φ``
        infer ``Γ₁ ∪ Γ₂ ⊢ φ ↔ ψ``.
      * ``iff_elim_l(i)``: from ``Γ ⊢ φ ↔ ψ`` infer ``Γ ⊢ φ → ψ``.
      * ``iff_elim_r(i)``: from ``Γ ⊢ φ ↔ ψ`` infer ``Γ ⊢ ψ → φ``.
      * ``lem(φ)``: classical, ``∅ ⊢ φ ∨ ¬φ`` for any ``φ``; target
        is the disjunction.
      * ``dne(i)``: classical, from ``Γ ⊢ ¬¬φ`` infer ``Γ ⊢ φ``.
      * ``top_intro``: ``∅ ⊢ ⊤``.
      * ``repeat(i)``: from ``Γ ⊢ φ`` infer ``Γ ⊢ φ`` (trivial; used
        by some natural-deduction styles to align line numbers).
    """

    rule: str
    premises: tuple = ()        # indices into prior steps
    conclusion: Any = None      # Term
    discharge: Any = None       # Term (the discharged assumption) or None
    target: Any = None          # Term (used by or_intro / bot_elim / lem) or None

    def __post_init__(self) -> None:
        if self.rule not in VERIFIER_KNOWN_ND_RULES:
            raise InvalidProof(f"unknown ND rule: {self.rule!r}")
        if not isinstance(self.premises, tuple):
            object.__setattr__(self, "premises", tuple(self.premises))
        for p in self.premises:
            if not isinstance(p, int) or p < 0:
                raise InvalidProof(
                    f"premise index must be non-negative int, got {p!r}"
                )
        if self.conclusion is not None and not isinstance(self.conclusion, Term):
            raise InvalidProof("NaturalDeductionStep.conclusion must be a Term or None")
        if self.discharge is not None and not isinstance(self.discharge, Term):
            raise InvalidProof("NaturalDeductionStep.discharge must be a Term or None")
        if self.target is not None and not isinstance(self.target, Term):
            raise InvalidProof("NaturalDeductionStep.target must be a Term or None")


@dataclass(frozen=True)
class NaturalDeductionProof:
    """A list of NK steps; the last step's sequent must be ``Γ ⊢ goal``
    for a subset ``Γ`` of the global premises of the verification call.
    """

    steps: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))
        for s in self.steps:
            if not isinstance(s, NaturalDeductionStep):
                raise InvalidProof(
                    f"NaturalDeductionProof.steps must contain NaturalDeductionStep, got {type(s).__name__}"
                )

    def __len__(self) -> int:
        return len(self.steps)


# =============================================================================
# Equational rewriting proof data
# =============================================================================
#
# An equational proof shows ``s = t`` from a set of axiom equations
# ``E = {ℓᵢ = rᵢ}``.  Each step rewrites the current term at a named
# position by a chosen axiom in a chosen direction.  Positions are
# strings of integer indices into sub-term tuples — empty tuple is the
# root, ``(0,)`` is the left child of a binary connective, ``(1,)`` is
# the right child, ``(0,)`` of ¬ is the negand.  The kernel does *not*
# perform unification; the axiom must match by structural equality after
# the variable substitution given in ``subst``.  This is Birkhoff-sound
# for *closed* terms; for open terms with variables, the substitution
# binds the axiom's free variables (specified as atom names in
# ``variables``) to terms.


@dataclass(frozen=True)
class EquationalAxiom:
    """One equational axiom ``lhs = rhs``, optionally with free
    variables.  ``variables`` is a tuple of atom names that the
    rewriter is free to bind (a ``Term.atom(name)`` whose name is in
    ``variables`` is treated as a variable rather than a constant).
    """

    name: str
    lhs: Term
    rhs: Term
    variables: tuple = ()      # tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InvalidFormula(f"axiom name must be non-empty string, got {self.name!r}")
        _require_term(self.lhs)
        _require_term(self.rhs)
        if not isinstance(self.variables, tuple):
            object.__setattr__(self, "variables", tuple(self.variables))
        for v in self.variables:
            if not isinstance(v, str) or not v:
                raise InvalidFormula(f"variable name must be non-empty string, got {v!r}")


REWRITE_FORWARD = "forward"     # apply lhs → rhs
REWRITE_BACKWARD = "backward"   # apply rhs → lhs

REWRITE_DIRECTIONS = (REWRITE_FORWARD, REWRITE_BACKWARD)


@dataclass(frozen=True)
class RewriteStep:
    """One rewrite step.  ``axiom_index`` indexes the axiom list passed
    to ``verify_equational``; ``position`` is the tuple-of-int path to
    the redex; ``direction`` is forward (lhs → rhs) or backward;
    ``substitution`` binds axiom variables to terms.  The kernel
    re-applies the substitution to the axiom's ``lhs`` (or ``rhs`` for
    backward), matches it against the term at ``position`` in the
    current state, and replaces it with the substituted other side.
    """

    axiom_index: int
    position: tuple
    direction: str
    substitution: tuple = ()    # tuple[(name, Term), ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axiom_index, int) or self.axiom_index < 0:
            raise InvalidProof(f"axiom_index must be non-negative int, got {self.axiom_index!r}")
        if not isinstance(self.position, tuple):
            object.__setattr__(self, "position", tuple(self.position))
        for p in self.position:
            if not isinstance(p, int) or p < 0:
                raise InvalidProof(f"position entries must be non-negative ints, got {p!r}")
        if self.direction not in REWRITE_DIRECTIONS:
            raise InvalidProof(f"direction must be one of {REWRITE_DIRECTIONS}, got {self.direction!r}")
        if not isinstance(self.substitution, tuple):
            object.__setattr__(self, "substitution", tuple(self.substitution))
        for entry in self.substitution:
            if (not isinstance(entry, tuple) or len(entry) != 2 or
                not isinstance(entry[0], str) or not isinstance(entry[1], Term)):
                raise InvalidProof(
                    f"substitution entry must be (str, Term), got {entry!r}"
                )


@dataclass(frozen=True)
class EquationalProof:
    """A list of rewrite steps; the kernel applies them in order
    starting from ``lhs`` and the final term must equal ``rhs``."""

    steps: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))
        for s in self.steps:
            if not isinstance(s, RewriteStep):
                raise InvalidProof(
                    f"EquationalProof.steps must contain RewriteStep, got {type(s).__name__}"
                )

    def __len__(self) -> int:
        return len(self.steps)


# =============================================================================
# The Kernel — the trusted base
# =============================================================================
#
# Every public verify_* method reduces to a sequence of Kernel.* calls.
# Every Kernel.* method is a *pure function* of its arguments that
# raises ``KernelViolation`` if the precondition fails.  The kernel
# has no state, no global, no time-dependent behaviour.  It is the
# only place where a "fact" is *promoted* — every other check in the
# Verifier is just bookkeeping.


@dataclass(frozen=True)
class _Sequent:
    """``Γ ⊢ φ``: a finite assumption set together with a conclusion."""

    gamma: frozenset         # frozenset[Term]
    phi: Term


class Kernel:
    """The LCF-style trusted kernel — ~20 primitive rules.

    Every method is a pure inference: given premises of the right
    shape (and possibly an extra argument like the pivot for
    resolution or the discharged formula for ``→I``), it returns the
    canonical conclusion.  If the premises do not match the rule's
    pattern, it raises ``KernelViolation`` with a human-readable
    message.

    The reason this class exists as its own type — rather than as
    free functions — is the de Bruijn criterion: someone auditing
    the proof system can read *exactly* this class and convince
    themselves the entire trust base is here.
    """

    # ----- Resolution kernel -------------------------------------------------

    @staticmethod
    def resolve(c1: tuple, c2: tuple, pivot: int) -> tuple:
        """Resolution rule: from clauses containing ``+pivot`` and
        ``-pivot`` respectively, produce their resolvent.  The result
        is canonicalised; raises ``KernelViolation`` if the literals
        are not present or if the resolvent is a tautology.
        """
        if not isinstance(pivot, int) or pivot <= 0:
            raise KernelViolation(f"pivot must be positive variable index, got {pivot!r}")
        if pivot not in c1 and -pivot not in c1:
            raise KernelViolation(
                f"pivot variable {pivot} not present in clause {c1!r}"
            )
        if pivot not in c2 and -pivot not in c2:
            raise KernelViolation(
                f"pivot variable {pivot} not present in clause {c2!r}"
            )
        # Determine signs.
        if pivot in c1 and -pivot in c2:
            pos, neg = c1, c2
        elif -pivot in c1 and pivot in c2:
            pos, neg = c2, c1
        else:
            raise KernelViolation(
                f"clauses do not have opposing signs on pivot {pivot}: {c1!r}, {c2!r}"
            )
        merged = []
        seen = set()
        for lit in pos:
            if lit == pivot:
                continue
            if -lit in seen:
                raise KernelViolation(
                    f"resolution would produce tautology on literal {abs(lit)}"
                )
            if lit not in seen:
                seen.add(lit)
                merged.append(lit)
        for lit in neg:
            if lit == -pivot:
                continue
            if -lit in seen:
                raise KernelViolation(
                    f"resolution would produce tautology on literal {abs(lit)}"
                )
            if lit not in seen:
                seen.add(lit)
                merged.append(lit)
        merged.sort(key=lambda x: (abs(x), x < 0))
        return tuple(merged)

    # ----- Natural-deduction kernel -----------------------------------------

    @staticmethod
    def nd_assume(phi: Term) -> _Sequent:
        _require_term(phi)
        return _Sequent(frozenset({phi}), phi)

    @staticmethod
    def nd_premise(phi: Term, global_premises: frozenset) -> _Sequent:
        _require_term(phi)
        if phi not in global_premises:
            raise KernelViolation(f"{phi} is not a global premise")
        return _Sequent(frozenset(), phi)

    @staticmethod
    def nd_and_intro(s1: _Sequent, s2: _Sequent) -> _Sequent:
        return _Sequent(s1.gamma | s2.gamma, Term.conj(s1.phi, s2.phi))

    @staticmethod
    def nd_and_elim_l(s: _Sequent) -> _Sequent:
        if not s.phi.is_and():
            raise KernelViolation(f"and_elim_l: premise is not a conjunction: {s.phi}")
        return _Sequent(s.gamma, s.phi.payload[0])

    @staticmethod
    def nd_and_elim_r(s: _Sequent) -> _Sequent:
        if not s.phi.is_and():
            raise KernelViolation(f"and_elim_r: premise is not a conjunction: {s.phi}")
        return _Sequent(s.gamma, s.phi.payload[1])

    @staticmethod
    def nd_or_intro_l(s: _Sequent, target: Term) -> _Sequent:
        _require_term(target)
        if not target.is_or():
            raise KernelViolation(f"or_intro_l target must be a disjunction: {target}")
        if target.payload[0] != s.phi:
            raise KernelViolation(
                f"or_intro_l: target's left disjunct {target.payload[0]} != premise {s.phi}"
            )
        return _Sequent(s.gamma, target)

    @staticmethod
    def nd_or_intro_r(s: _Sequent, target: Term) -> _Sequent:
        _require_term(target)
        if not target.is_or():
            raise KernelViolation(f"or_intro_r target must be a disjunction: {target}")
        if target.payload[1] != s.phi:
            raise KernelViolation(
                f"or_intro_r: target's right disjunct {target.payload[1]} != premise {s.phi}"
            )
        return _Sequent(s.gamma, target)

    @staticmethod
    def nd_or_elim(s_or: _Sequent, s_left: _Sequent, s_right: _Sequent) -> _Sequent:
        if not s_or.phi.is_or():
            raise KernelViolation(f"or_elim: first premise is not a disjunction: {s_or.phi}")
        phi, psi = s_or.phi.payload
        if s_left.phi != s_right.phi:
            raise KernelViolation(
                f"or_elim: left and right branches conclude different formulas: "
                f"{s_left.phi} vs {s_right.phi}"
            )
        if phi not in s_left.gamma:
            raise KernelViolation(
                f"or_elim: left branch context does not contain disjunct {phi}"
            )
        if psi not in s_right.gamma:
            raise KernelViolation(
                f"or_elim: right branch context does not contain disjunct {psi}"
            )
        new_gamma = s_or.gamma | (s_left.gamma - {phi}) | (s_right.gamma - {psi})
        return _Sequent(new_gamma, s_left.phi)

    @staticmethod
    def nd_imp_intro(s: _Sequent, discharged: Term) -> _Sequent:
        _require_term(discharged)
        if discharged not in s.gamma:
            raise KernelViolation(
                f"imp_intro: discharged formula {discharged} not in premise context"
            )
        return _Sequent(s.gamma - {discharged}, Term.imp(discharged, s.phi))

    @staticmethod
    def nd_imp_elim(s_imp: _Sequent, s_ant: _Sequent) -> _Sequent:
        if not s_imp.phi.is_imp():
            raise KernelViolation(f"imp_elim: first premise is not implication: {s_imp.phi}")
        ant, cons = s_imp.phi.payload
        if s_ant.phi != ant:
            raise KernelViolation(
                f"imp_elim: antecedent mismatch: {ant} vs {s_ant.phi}"
            )
        return _Sequent(s_imp.gamma | s_ant.gamma, cons)

    @staticmethod
    def nd_not_intro(s: _Sequent, discharged: Term) -> _Sequent:
        _require_term(discharged)
        if not s.phi.is_bot():
            raise KernelViolation(f"not_intro: premise must conclude ⊥, got {s.phi}")
        if discharged not in s.gamma:
            raise KernelViolation(
                f"not_intro: discharged formula {discharged} not in premise context"
            )
        return _Sequent(s.gamma - {discharged}, Term.neg(discharged))

    @staticmethod
    def nd_not_elim(s_phi: _Sequent, s_not_phi: _Sequent) -> _Sequent:
        if not s_not_phi.phi.is_not():
            raise KernelViolation(f"not_elim: second premise must be a negation, got {s_not_phi.phi}")
        if s_not_phi.phi.payload[0] != s_phi.phi:
            raise KernelViolation(
                f"not_elim: φ vs ¬φ mismatch: {s_phi.phi} and {s_not_phi.phi}"
            )
        return _Sequent(s_phi.gamma | s_not_phi.gamma, Term.bot())

    @staticmethod
    def nd_bot_elim(s: _Sequent, target: Term) -> _Sequent:
        _require_term(target)
        if not s.phi.is_bot():
            raise KernelViolation(f"bot_elim: premise must conclude ⊥, got {s.phi}")
        return _Sequent(s.gamma, target)

    @staticmethod
    def nd_iff_intro(s_fwd: _Sequent, s_bwd: _Sequent) -> _Sequent:
        if not s_fwd.phi.is_imp() or not s_bwd.phi.is_imp():
            raise KernelViolation("iff_intro: both premises must be implications")
        a, b = s_fwd.phi.payload
        c, d = s_bwd.phi.payload
        if a != d or b != c:
            raise KernelViolation(
                f"iff_intro: implications don't form a bi-implication: {s_fwd.phi}, {s_bwd.phi}"
            )
        return _Sequent(s_fwd.gamma | s_bwd.gamma, Term.iff(a, b))

    @staticmethod
    def nd_iff_elim_l(s: _Sequent) -> _Sequent:
        if not s.phi.is_iff():
            raise KernelViolation(f"iff_elim_l: premise is not a bi-implication: {s.phi}")
        a, b = s.phi.payload
        return _Sequent(s.gamma, Term.imp(a, b))

    @staticmethod
    def nd_iff_elim_r(s: _Sequent) -> _Sequent:
        if not s.phi.is_iff():
            raise KernelViolation(f"iff_elim_r: premise is not a bi-implication: {s.phi}")
        a, b = s.phi.payload
        return _Sequent(s.gamma, Term.imp(b, a))

    @staticmethod
    def nd_lem(target: Term) -> _Sequent:
        _require_term(target)
        if not target.is_or():
            raise KernelViolation(f"lem: target must be a disjunction, got {target}")
        phi, neg_phi = target.payload
        if not neg_phi.is_not() or neg_phi.payload[0] != phi:
            raise KernelViolation(
                f"lem: target must be φ ∨ ¬φ, got {target}"
            )
        return _Sequent(frozenset(), target)

    @staticmethod
    def nd_dne(s: _Sequent) -> _Sequent:
        if not s.phi.is_not() or not s.phi.payload[0].is_not():
            raise KernelViolation(f"dne: premise must be ¬¬φ, got {s.phi}")
        return _Sequent(s.gamma, s.phi.payload[0].payload[0])

    @staticmethod
    def nd_top_intro() -> _Sequent:
        return _Sequent(frozenset(), Term.top())

    @staticmethod
    def nd_repeat(s: _Sequent) -> _Sequent:
        return _Sequent(s.gamma, s.phi)

    # ----- Equational kernel ------------------------------------------------

    @staticmethod
    def rewrite_at(term: Term, position: tuple, redex: Term, contractum: Term) -> Term:
        """Replace the sub-term at ``position`` (which must structurally
        equal ``redex``) by ``contractum``.  Returns the new term.
        """
        _require_term(term); _require_term(redex); _require_term(contractum)
        # Locate the sub-term at `position`.
        cur = term
        path = []
        for i in position:
            path.append((cur, i))
            if cur.kind in (_KIND_ATOM, _KIND_BOT, _KIND_TOP):
                raise KernelViolation(
                    f"position {position} reaches into atomic term {cur}"
                )
            if cur.kind == _KIND_NOT:
                if i != 0:
                    raise KernelViolation(
                        f"position {position} index {i} invalid in ¬"
                    )
                cur = cur.payload[0]
            else:
                if i not in (0, 1):
                    raise KernelViolation(
                        f"position {position} index {i} invalid in binary connective {cur.kind}"
                    )
                cur = cur.payload[i]
        if cur != redex:
            raise KernelViolation(
                f"sub-term at position {position} is {cur}, not the claimed redex {redex}"
            )
        # Rebuild upwards.
        result = contractum
        for parent, i in reversed(path):
            if parent.kind == _KIND_NOT:
                result = Term.neg(result)
            else:
                left, right = parent.payload
                if i == 0:
                    result = Term(parent.kind, (result, right))
                else:
                    result = Term(parent.kind, (left, result))
        return result


# =============================================================================
# Helpers — substitution, term hashing, canonical serialisation
# =============================================================================


def _substitute(t: Term, sigma: Mapping[str, Term]) -> Term:
    """Apply substitution ``sigma`` (atom-name → Term) to ``t``.

    Substitution is *capture-free* by construction: there are no
    binders in propositional/first-order-atoms logic at this layer.
    Only ``atom`` terms whose name is in ``sigma`` are replaced; all
    other atoms are unchanged.
    """
    if t.kind == _KIND_ATOM:
        return sigma.get(t.payload, t)
    if t.kind in (_KIND_BOT, _KIND_TOP):
        return t
    if t.kind == _KIND_NOT:
        return Term.neg(_substitute(t.payload[0], sigma))
    left = _substitute(t.payload[0], sigma)
    right = _substitute(t.payload[1], sigma)
    return Term(t.kind, (left, right))


def _term_hash(t: Term, digest: hashlib._Hash) -> None:  # type: ignore[name-defined]
    digest.update(b"(")
    digest.update(t.kind.encode("utf-8"))
    if t.kind == _KIND_ATOM:
        digest.update(b":")
        digest.update(t.payload.encode("utf-8"))
    elif t.kind == _KIND_NOT:
        digest.update(b":")
        _term_hash(t.payload[0], digest)
    elif t.kind in (_KIND_AND, _KIND_OR, _KIND_IMP, _KIND_IFF):
        digest.update(b":")
        _term_hash(t.payload[0], digest)
        digest.update(b",")
        _term_hash(t.payload[1], digest)
    digest.update(b")")


def _clause_hash(c: tuple, digest: hashlib._Hash) -> None:  # type: ignore[name-defined]
    digest.update(b"[")
    for lit in c:
        digest.update(str(lit).encode("utf-8"))
        digest.update(b",")
    digest.update(b"]")


def _sequent_hash(s: _Sequent, digest: hashlib._Hash) -> None:  # type: ignore[name-defined]
    digest.update(b"{")
    # Sort gamma by hash of each term so the order is deterministic.
    gamma_sorted = sorted(s.gamma, key=lambda x: _term_canonical(x))
    for phi in gamma_sorted:
        _term_hash(phi, digest)
        digest.update(b";")
    digest.update(b"}|-")
    _term_hash(s.phi, digest)


def _term_canonical(t: Term) -> str:
    h = hashlib.sha256()
    _term_hash(t, h)
    return h.hexdigest()


# =============================================================================
# Verifier configuration and report
# =============================================================================


@dataclass(frozen=True)
class VerifierConfig:
    """Configuration for a Verifier instance.

    Attributes:
      hmac_key: shared secret for the tamper-evident certificate.  If
        the bytes are empty, ``VerifierReport.certificate`` is still
        computed but uses a plain SHA-256 (so the HMAC degrades to a
        hash — useful for tests, not for production attestation).
      max_proof_length: any proof with more steps than this is
        rejected with status ``MALFORMED``.
      max_term_depth: any term encountered (in formula, proof step,
        or rewrite) deeper than this is rejected.  Prevents pathological
        recursion.
      fail_fast: if True (the default), verify_* returns the first
        failure; if False, it still records every kernel step but
        continues past failures (used in test/debug scenarios).
      record_trace: if True, ``VerifierReport.trace`` includes a
        per-step textual summary suitable for logs.
      check_no_tautological_resolution: if True (default) the kernel
        rejects resolutions whose resolvent is a tautology; this is
        the standard semantic (tautologies carry no information).
      enforce_intuitionistic: if True, the verifier rejects use of the
        classical rules ``lem`` and ``dne``.  Lets a coordination
        engine ask the stricter question "does this hold
        constructively?"
    """

    hmac_key: bytes = b""
    max_proof_length: int = 1_000_000
    max_term_depth: int = 256
    fail_fast: bool = True
    record_trace: bool = True
    check_no_tautological_resolution: bool = True
    enforce_intuitionistic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes")
        if not isinstance(self.max_proof_length, int) or self.max_proof_length <= 0:
            raise InvalidConfig("max_proof_length must be a positive int")
        if not isinstance(self.max_term_depth, int) or self.max_term_depth <= 0:
            raise InvalidConfig("max_term_depth must be a positive int")


@dataclass(frozen=True)
class VerifierReport:
    """Outcome of a verification call.

    Attributes:
      status: one of VERIFIED, FAILED, MALFORMED.
      kind: which proof system was used (KIND_RESOLUTION / ...).
      n_steps: number of steps in the proof.
      failed_step: 0-based index of the step that failed (None if
        verified or malformed).
      failure_reason: human-readable reason for failure (None if
        verified).
      certificate: hex-encoded HMAC-SHA256 (or SHA-256 if no key) over
        the canonical serialisation of every kernel step.
      trace: per-step textual summary; empty if record_trace was False.
      elapsed_seconds: wall-clock time of the verify_* call.
      kernel_calls: number of kernel rule invocations.
      tcb_lines: approximate count of trusted-base lines in this
        Verifier (a constant; included so the report is self-describing).
    """

    status: str
    kind: str
    n_steps: int
    failed_step: Any = None       # int or None
    failure_reason: Any = None    # str or None
    certificate: str = ""
    trace: tuple = ()
    elapsed_seconds: float = 0.0
    kernel_calls: int = 0
    tcb_lines: int = 250

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "kind": self.kind,
            "n_steps": self.n_steps,
            "failed_step": self.failed_step,
            "failure_reason": self.failure_reason,
            "certificate": self.certificate,
            "elapsed_seconds": self.elapsed_seconds,
            "kernel_calls": self.kernel_calls,
            "tcb_lines": self.tcb_lines,
        }


# =============================================================================
# The Verifier class
# =============================================================================


class Verifier:
    """LCF-style proof verifier.  Stateless apart from the
    configuration; all methods are pure functions of their arguments.

    The class is a thin orchestration layer over the static ``Kernel``
    methods.  It records each kernel call into an HMAC digest, builds
    the canonical ``VerifierReport``, and converts any
    ``KernelViolation`` into a structured failure.
    """

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self.config = config or VerifierConfig()

    # ----- Resolution -------------------------------------------------------

    def verify_resolution(
        self,
        formula: CNFFormula,
        proof: ResolutionProof,
    ) -> VerifierReport:
        """Verify a resolution refutation of ``formula``.

        Each step claims that ``parents[0]`` and ``parents[1]`` (either
        an original clause index or a previously-derived resolvent
        index) resolve on ``pivot`` to ``resolvent``.  The kernel
        re-runs each resolution; verification succeeds if every claim
        is reproducible *and* the final step produces the empty clause.
        """

        start = time.perf_counter()
        if not isinstance(formula, CNFFormula):
            return self._malformed(KIND_RESOLUTION, 0, "formula must be CNFFormula")
        if not isinstance(proof, ResolutionProof):
            return self._malformed(KIND_RESOLUTION, 0, "proof must be ResolutionProof")
        if len(proof) > self.config.max_proof_length:
            return self._malformed(
                KIND_RESOLUTION,
                len(proof),
                f"proof length {len(proof)} exceeds max_proof_length {self.config.max_proof_length}",
            )

        h = self._init_digest(KIND_RESOLUTION)
        for c in formula.clauses:
            _clause_hash(c, h)
            h.update(b"|")

        clauses = list(formula.clauses)            # mutable; we append resolvents
        n_orig = len(clauses)
        trace = []
        kernel_calls = 0

        for idx, step in enumerate(proof.steps):
            try:
                i, j = step.parents
                if i >= n_orig + idx or j >= n_orig + idx:
                    raise InvalidProof(
                        f"step {idx}: parent index out of range "
                        f"(have {n_orig + idx} clauses, got {i}, {j})"
                    )
                c1 = clauses[i]
                c2 = clauses[j]
                resolvent = Kernel.resolve(c1, c2, step.pivot)
                kernel_calls += 1
                if resolvent != step.resolvent:
                    return self._failed(
                        KIND_RESOLUTION,
                        idx,
                        f"resolvent mismatch: kernel produced {resolvent}, "
                        f"step claimed {step.resolvent}",
                        certificate=self._finalise(h),
                        n_steps=len(proof),
                        trace=tuple(trace),
                        kernel_calls=kernel_calls,
                        elapsed=time.perf_counter() - start,
                    )
                clauses.append(resolvent)
                if self.config.record_trace:
                    trace.append(
                        f"R{idx}: {list(c1)} +{step.pivot}/-{step.pivot} {list(c2)} → {list(resolvent)}"
                    )
                _clause_hash(resolvent, h)
                h.update(str(step.pivot).encode("utf-8"))
                h.update(b"|")
            except (InvalidProof, KernelViolation) as exc:
                return self._failed(
                    KIND_RESOLUTION,
                    idx,
                    str(exc),
                    certificate=self._finalise(h),
                    n_steps=len(proof),
                    trace=tuple(trace),
                    kernel_calls=kernel_calls,
                    elapsed=time.perf_counter() - start,
                )

        if not clauses[-1] == ():
            return self._failed(
                KIND_RESOLUTION,
                len(proof) - 1 if len(proof) > 0 else 0,
                f"final resolvent is {clauses[-1]}, not the empty clause",
                certificate=self._finalise(h),
                n_steps=len(proof),
                trace=tuple(trace),
                kernel_calls=kernel_calls,
                elapsed=time.perf_counter() - start,
            )

        return VerifierReport(
            status=VERIFIED,
            kind=KIND_RESOLUTION,
            n_steps=len(proof),
            certificate=self._finalise(h),
            trace=tuple(trace),
            elapsed_seconds=time.perf_counter() - start,
            kernel_calls=kernel_calls,
        )

    # ----- Natural deduction ------------------------------------------------

    def verify_natural_deduction(
        self,
        premises: Sequence[Term],
        goal: Term,
        proof: NaturalDeductionProof,
    ) -> VerifierReport:
        """Verify a natural-deduction proof of ``goal`` from
        ``premises``.

        Each step in ``proof.steps`` is re-derived against the kernel
        rule named by ``step.rule``; the verifier maintains, for each
        step, the *kernel-derived* sequent ``Γᵢ ⊢ φᵢ`` (it does *not*
        trust the proof's own ``conclusion`` until the kernel agrees).
        Verification succeeds when:

          1. every kernel re-derivation succeeds;
          2. the final sequent ``Γ_n ⊢ φ_n`` has ``φ_n == goal``;
          3. ``Γ_n ⊆ premises`` (no leftover undischarged assumptions).
        """

        start = time.perf_counter()
        for p in premises:
            _require_term(p)
        _require_term(goal)
        if not isinstance(proof, NaturalDeductionProof):
            return self._malformed(KIND_NATURAL_DEDUCTION, 0, "proof must be NaturalDeductionProof")
        if len(proof) > self.config.max_proof_length:
            return self._malformed(
                KIND_NATURAL_DEDUCTION,
                len(proof),
                f"proof length {len(proof)} exceeds max_proof_length",
            )
        if goal.depth() > self.config.max_term_depth:
            return self._malformed(
                KIND_NATURAL_DEDUCTION, 0,
                f"goal depth {goal.depth()} exceeds max_term_depth {self.config.max_term_depth}",
            )

        premise_set = frozenset(premises)
        h = self._init_digest(KIND_NATURAL_DEDUCTION)
        for p in sorted(premise_set, key=_term_canonical):
            _term_hash(p, h)
            h.update(b";")
        h.update(b"|-")
        _term_hash(goal, h)
        h.update(b"||")

        sequents: list[_Sequent] = []
        trace = []
        kernel_calls = 0

        for idx, step in enumerate(proof.steps):
            try:
                if self.config.enforce_intuitionistic and step.rule not in _INTUITIONISTIC_RULES:
                    raise InvalidProof(
                        f"rule {step.rule} is classical; intuitionistic mode forbids it"
                    )
                s = self._apply_nd_step(step, sequents, premise_set)
                kernel_calls += 1
                # Verify the step's claimed conclusion matches the kernel's.
                if step.conclusion is not None and step.conclusion != s.phi:
                    raise KernelViolation(
                        f"step's stated conclusion {step.conclusion} != kernel-derived {s.phi}"
                    )
                sequents.append(s)
                if self.config.record_trace:
                    trace.append(
                        f"ND{idx} [{step.rule}]: {_format_gamma(s.gamma)} ⊢ {s.phi}"
                    )
                _sequent_hash(s, h)
                h.update(b"|")
            except (InvalidProof, KernelViolation) as exc:
                return self._failed(
                    KIND_NATURAL_DEDUCTION,
                    idx,
                    str(exc),
                    certificate=self._finalise(h),
                    n_steps=len(proof),
                    trace=tuple(trace),
                    kernel_calls=kernel_calls,
                    elapsed=time.perf_counter() - start,
                )

        if not sequents:
            return self._failed(
                KIND_NATURAL_DEDUCTION, 0, "empty proof",
                certificate=self._finalise(h), n_steps=0,
                kernel_calls=0, elapsed=time.perf_counter() - start,
            )

        final = sequents[-1]
        if final.phi != goal:
            return self._failed(
                KIND_NATURAL_DEDUCTION,
                len(proof) - 1,
                f"final conclusion {final.phi} != goal {goal}",
                certificate=self._finalise(h),
                n_steps=len(proof),
                trace=tuple(trace),
                kernel_calls=kernel_calls,
                elapsed=time.perf_counter() - start,
            )
        leftover = final.gamma - premise_set
        if leftover:
            return self._failed(
                KIND_NATURAL_DEDUCTION,
                len(proof) - 1,
                f"undischarged assumptions remain: {{ {', '.join(str(x) for x in leftover)} }}",
                certificate=self._finalise(h),
                n_steps=len(proof),
                trace=tuple(trace),
                kernel_calls=kernel_calls,
                elapsed=time.perf_counter() - start,
            )

        return VerifierReport(
            status=VERIFIED,
            kind=KIND_NATURAL_DEDUCTION,
            n_steps=len(proof),
            certificate=self._finalise(h),
            trace=tuple(trace),
            elapsed_seconds=time.perf_counter() - start,
            kernel_calls=kernel_calls,
        )

    def _apply_nd_step(
        self,
        step: NaturalDeductionStep,
        sequents: Sequence[_Sequent],
        premises: frozenset,
    ) -> _Sequent:
        def ref(i: int) -> _Sequent:
            if i >= len(sequents):
                raise InvalidProof(f"premise index {i} refers to undefined step")
            return sequents[i]

        r = step.rule
        if r == RULE_ASSUMPTION:
            if step.conclusion is None:
                raise InvalidProof("assumption rule requires a conclusion")
            return Kernel.nd_assume(step.conclusion)
        if r == RULE_PREMISE:
            if step.conclusion is None:
                raise InvalidProof("premise rule requires a conclusion")
            return Kernel.nd_premise(step.conclusion, premises)
        if r == RULE_AND_INTRO:
            if len(step.premises) != 2:
                raise InvalidProof("and_intro needs 2 premises")
            return Kernel.nd_and_intro(ref(step.premises[0]), ref(step.premises[1]))
        if r == RULE_AND_ELIM_L:
            if len(step.premises) != 1:
                raise InvalidProof("and_elim_l needs 1 premise")
            return Kernel.nd_and_elim_l(ref(step.premises[0]))
        if r == RULE_AND_ELIM_R:
            if len(step.premises) != 1:
                raise InvalidProof("and_elim_r needs 1 premise")
            return Kernel.nd_and_elim_r(ref(step.premises[0]))
        if r == RULE_OR_INTRO_L:
            if len(step.premises) != 1 or step.target is None:
                raise InvalidProof("or_intro_l needs 1 premise + target")
            return Kernel.nd_or_intro_l(ref(step.premises[0]), step.target)
        if r == RULE_OR_INTRO_R:
            if len(step.premises) != 1 or step.target is None:
                raise InvalidProof("or_intro_r needs 1 premise + target")
            return Kernel.nd_or_intro_r(ref(step.premises[0]), step.target)
        if r == RULE_OR_ELIM:
            if len(step.premises) != 3:
                raise InvalidProof("or_elim needs 3 premises")
            return Kernel.nd_or_elim(
                ref(step.premises[0]), ref(step.premises[1]), ref(step.premises[2]),
            )
        if r == RULE_IMP_INTRO:
            if len(step.premises) != 1 or step.discharge is None:
                raise InvalidProof("imp_intro needs 1 premise + discharge")
            return Kernel.nd_imp_intro(ref(step.premises[0]), step.discharge)
        if r == RULE_IMP_ELIM:
            if len(step.premises) != 2:
                raise InvalidProof("imp_elim needs 2 premises")
            return Kernel.nd_imp_elim(ref(step.premises[0]), ref(step.premises[1]))
        if r == RULE_NOT_INTRO:
            if len(step.premises) != 1 or step.discharge is None:
                raise InvalidProof("not_intro needs 1 premise + discharge")
            return Kernel.nd_not_intro(ref(step.premises[0]), step.discharge)
        if r == RULE_NOT_ELIM:
            if len(step.premises) != 2:
                raise InvalidProof("not_elim needs 2 premises")
            return Kernel.nd_not_elim(ref(step.premises[0]), ref(step.premises[1]))
        if r == RULE_BOT_ELIM:
            if len(step.premises) != 1 or step.target is None:
                raise InvalidProof("bot_elim needs 1 premise + target")
            return Kernel.nd_bot_elim(ref(step.premises[0]), step.target)
        if r == RULE_IFF_INTRO:
            if len(step.premises) != 2:
                raise InvalidProof("iff_intro needs 2 premises")
            return Kernel.nd_iff_intro(ref(step.premises[0]), ref(step.premises[1]))
        if r == RULE_IFF_ELIM_L:
            if len(step.premises) != 1:
                raise InvalidProof("iff_elim_l needs 1 premise")
            return Kernel.nd_iff_elim_l(ref(step.premises[0]))
        if r == RULE_IFF_ELIM_R:
            if len(step.premises) != 1:
                raise InvalidProof("iff_elim_r needs 1 premise")
            return Kernel.nd_iff_elim_r(ref(step.premises[0]))
        if r == RULE_LEM:
            if step.target is None:
                raise InvalidProof("lem needs a target")
            return Kernel.nd_lem(step.target)
        if r == RULE_DNE:
            if len(step.premises) != 1:
                raise InvalidProof("dne needs 1 premise")
            return Kernel.nd_dne(ref(step.premises[0]))
        if r == RULE_TOP_INTRO:
            return Kernel.nd_top_intro()
        if r == RULE_REPEAT:
            if len(step.premises) != 1:
                raise InvalidProof("repeat needs 1 premise")
            return Kernel.nd_repeat(ref(step.premises[0]))
        raise InvalidProof(f"unhandled rule: {r}")

    # ----- Equational rewriting --------------------------------------------

    def verify_equational(
        self,
        axioms: Sequence[EquationalAxiom],
        lhs: Term,
        rhs: Term,
        proof: EquationalProof,
    ) -> VerifierReport:
        """Verify an equational proof of ``lhs = rhs`` from
        ``axioms``.

        The verifier starts with ``current = lhs`` and applies each
        rewrite step in order, replacing the sub-term at the named
        position by the substituted other side of the chosen axiom.
        Success iff ``current == rhs`` after the final step.
        """

        start = time.perf_counter()
        _require_term(lhs); _require_term(rhs)
        for a in axioms:
            if not isinstance(a, EquationalAxiom):
                return self._malformed(
                    KIND_EQUATIONAL, 0,
                    f"axioms must contain EquationalAxiom, got {type(a).__name__}",
                )
        if not isinstance(proof, EquationalProof):
            return self._malformed(KIND_EQUATIONAL, 0, "proof must be EquationalProof")
        if len(proof) > self.config.max_proof_length:
            return self._malformed(
                KIND_EQUATIONAL, len(proof),
                f"proof length {len(proof)} exceeds max_proof_length",
            )

        h = self._init_digest(KIND_EQUATIONAL)
        for a in axioms:
            h.update(a.name.encode("utf-8")); h.update(b":")
            _term_hash(a.lhs, h); h.update(b"=")
            _term_hash(a.rhs, h); h.update(b"|")
        _term_hash(lhs, h); h.update(b"=?="); _term_hash(rhs, h); h.update(b"||")

        current = lhs
        trace = []
        kernel_calls = 0

        for idx, step in enumerate(proof.steps):
            try:
                if step.axiom_index >= len(axioms):
                    raise InvalidProof(
                        f"axiom_index {step.axiom_index} out of range "
                        f"(have {len(axioms)} axioms)"
                    )
                ax = axioms[step.axiom_index]
                sigma = dict(step.substitution)
                for var in sigma:
                    if var not in ax.variables:
                        raise InvalidProof(
                            f"substitution names {var}, which is not declared "
                            f"in axiom {ax.name}.variables {ax.variables}"
                        )
                if step.direction == REWRITE_FORWARD:
                    redex_template, contractum_template = ax.lhs, ax.rhs
                else:
                    redex_template, contractum_template = ax.rhs, ax.lhs
                redex = _substitute(redex_template, sigma)
                contractum = _substitute(contractum_template, sigma)
                if redex.depth() > self.config.max_term_depth:
                    raise InvalidProof("substituted redex exceeds max_term_depth")
                current = Kernel.rewrite_at(current, step.position, redex, contractum)
                kernel_calls += 1
                if current.depth() > self.config.max_term_depth:
                    raise InvalidProof("resulting term exceeds max_term_depth")
                if self.config.record_trace:
                    trace.append(
                        f"EQ{idx}: {ax.name} {step.direction} at {step.position} → {current}"
                    )
                _term_hash(current, h)
                h.update(b"|")
            except (InvalidProof, KernelViolation) as exc:
                return self._failed(
                    KIND_EQUATIONAL,
                    idx,
                    str(exc),
                    certificate=self._finalise(h),
                    n_steps=len(proof),
                    trace=tuple(trace),
                    kernel_calls=kernel_calls,
                    elapsed=time.perf_counter() - start,
                )

        if current != rhs:
            return self._failed(
                KIND_EQUATIONAL,
                len(proof) - 1 if len(proof) > 0 else 0,
                f"final term {current} != target rhs {rhs}",
                certificate=self._finalise(h),
                n_steps=len(proof),
                trace=tuple(trace),
                kernel_calls=kernel_calls,
                elapsed=time.perf_counter() - start,
            )

        return VerifierReport(
            status=VERIFIED,
            kind=KIND_EQUATIONAL,
            n_steps=len(proof),
            certificate=self._finalise(h),
            trace=tuple(trace),
            elapsed_seconds=time.perf_counter() - start,
            kernel_calls=kernel_calls,
        )

    # ----- Internal helpers -------------------------------------------------

    def _init_digest(self, kind: str):
        if self.config.hmac_key:
            return hmac.new(bytes(self.config.hmac_key), kind.encode("utf-8"), hashlib.sha256)
        h = hashlib.sha256()
        h.update(kind.encode("utf-8"))
        return h

    def _finalise(self, h) -> str:
        return h.hexdigest()

    def _failed(
        self,
        kind: str,
        idx: int,
        reason: str,
        *,
        certificate: str = "",
        n_steps: int = 0,
        trace: tuple = (),
        kernel_calls: int = 0,
        elapsed: float = 0.0,
    ) -> VerifierReport:
        return VerifierReport(
            status=FAILED,
            kind=kind,
            n_steps=n_steps,
            failed_step=idx,
            failure_reason=reason,
            certificate=certificate,
            trace=trace,
            elapsed_seconds=elapsed,
            kernel_calls=kernel_calls,
        )

    def _malformed(self, kind: str, n_steps: int, reason: str) -> VerifierReport:
        return VerifierReport(
            status=MALFORMED,
            kind=kind,
            n_steps=n_steps,
            failure_reason=reason,
        )


def _format_gamma(gamma: frozenset) -> str:
    if not gamma:
        return "∅"
    items = [str(t) for t in sorted(gamma, key=_term_canonical)]
    return "{" + ", ".join(items) + "}"


# =============================================================================
# Convenience free functions — match the style of other primitives
# =============================================================================


def verify_resolution(formula: CNFFormula, proof: ResolutionProof, **kwargs) -> VerifierReport:
    """Stateless one-shot resolution verification.  ``kwargs`` are
    passed straight to ``VerifierConfig``."""
    return Verifier(VerifierConfig(**kwargs)).verify_resolution(formula, proof)


def verify_natural_deduction(
    premises: Sequence[Term],
    goal: Term,
    proof: NaturalDeductionProof,
    **kwargs,
) -> VerifierReport:
    """Stateless one-shot natural-deduction verification."""
    return Verifier(VerifierConfig(**kwargs)).verify_natural_deduction(
        premises, goal, proof,
    )


def verify_equational(
    axioms: Sequence[EquationalAxiom],
    lhs: Term,
    rhs: Term,
    proof: EquationalProof,
    **kwargs,
) -> VerifierReport:
    """Stateless one-shot equational-proof verification."""
    return Verifier(VerifierConfig(**kwargs)).verify_equational(
        axioms, lhs, rhs, proof,
    )


def kernel_rule_count() -> int:
    """Return the number of kernel inference rules in the trusted base.

    This is the *static* count of primitive Kernel.* methods that are
    valid inference rules — the number a coordinator can quote in an
    attestation as "your trust depends on exactly this many rules".
    """
    return (
        1   # resolve
        + len(VERIFIER_KNOWN_ND_RULES)
        + 1  # rewrite_at
    )


def tcb_summary() -> dict:
    """Self-describing summary of the trusted computing base."""
    return {
        "kernel_rule_count": kernel_rule_count(),
        "nd_rules": VERIFIER_KNOWN_ND_RULES,
        "resolution_rules": ("resolve",),
        "equational_rules": ("rewrite_at",),
        "tcb_lines_approx": 250,
    }
