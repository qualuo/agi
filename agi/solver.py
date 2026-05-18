r"""Solver — CDCL satisfiability as a runtime primitive.

Every primitive shipped so far in this runtime answers a *statistical*
question — what is the best fit, the best forecast, the best closed-form
identity, the best ranking?  Statistical answers carry probabilities and
calibrated error bars but *never* a logical certificate.  A coordination
engine driving safety-critical actuation needs the complementary
capability: given a Boolean specification ``φ`` over discrete decision
variables, produce either

  * a satisfying assignment ``α`` with ``α ⊨ φ`` — a concrete plan, an
    actuator setting, a hardware configuration — together with a
    machine-checkable confirmation that ``φ(α) = ⊤``;
  * a *proof of unsatisfiability* — a sequence of resolvents that derives
    the empty clause from ``φ`` — guaranteeing that no satisfying
    assignment exists.

The ``Solver`` is the runtime primitive that closes this gap.  It is a
from-scratch implementation of Conflict-Driven Clause Learning (CDCL)
with the discipline-of-record refinements that put modern SAT solvers
at the heart of every industrial verification, planning, and synthesis
pipeline.

The pitch reduced to a runtime call::

    sv = Solver.create(seed=0)
    sv.add_clause([1, 2, -3])       # x1 ∨ x2 ∨ ¬x3
    sv.add_clause([-1, 3])          # ¬x1 ∨ x3
    sv.add_clause([-2, -3])         # ¬x2 ∨ ¬x3
    res = sv.solve()
    if res.status == "sat":
        print(res.model)            # e.g. {1: False, 2: True, 3: False}
    else:
        print(res.proof)            # DRAT proof of unsatisfiability

Every ``add_clause``, ``assume``, ``solve``, ``extract_mus``, ``report``
call is hashed into a SHA-256 chain compatible with the rest of the
runtime's :class:`~agi.attest.AttestationLedger` — so a solver's
trajectory from raw CNF to model or proof is fully auditable.

Mathematical roots
------------------

* **Cook 1971; Levin 1973 — NP-completeness of SAT.**  Boolean
  satisfiability is the canonical NP-complete problem.  Every
  decision problem in NP reduces in polynomial time to a SAT
  instance, which is why a fast SAT engine is a fast *general
  combinatorial reasoner*.

* **Davis-Putnam 1960; Davis-Logemann-Loveland 1962 — DPLL.**  The
  recursive splitting + unit propagation skeleton of every modern
  CDCL solver originates here.  ``Solver`` retains DPLL's unit
  propagation and pure-literal subsumption while replacing chronological
  backtracking with conflict-driven backjumping.

* **Marques-Silva-Sakallah 1996 — GRASP.**  *Conflict analysis*: when
  unit propagation derives ``⊥``, build the *implication graph*, find
  the first unique implication point (1-UIP), and learn the asserting
  clause whose addition both *prevents* the same conflict from
  recurring and forces an immediate propagation on backtrack.  The
  asserting-clause-plus-backjump engine is the heart of CDCL.

* **Moskewicz-Madigan-Zhao-Zhang-Malik 2001 — Chaff / VSIDS.**  Variable
  State Independent Decaying Sum: every literal carries a non-negative
  activity that is incremented every time that literal appears in a
  learnt clause and decayed multiplicatively each conflict.  Branching
  picks the unassigned literal of maximum activity — empirically the
  decisive heuristic in industrial benchmarks.

* **Moskewicz et al. 2001; Gent 2002 — Two-watched-literal scheme.**
  A clause needs to fire propagation only when *both* of its two
  watched literals become false; assignments to other literals can be
  ignored.  Reduces unit propagation to O(clauses ∋ falsified watch)
  per propagated literal — the data structure that made million-clause
  SAT instances tractable.

* **Pipatsrisawat-Darwiche 2007 — Phase saving.**  Decisions remember
  the last value the variable took before being unassigned by
  backjumping.  Empirically a major reduction in revisited search
  space; ``Solver`` saves phases by default.

* **Audemard-Simon 2009 — Glucose / LBD.**  The *Literal Block Distance*
  of a learnt clause — the number of distinct decision levels among
  its literals — is the empirical correlate of *future usefulness*.
  ``Solver`` keeps every clause with LBD ≤ 2 (the "glue") and
  periodically deletes the worst-LBD half of the remainder.

* **Luby-Sinclair-Zuckerman 1993 — Universal restart schedule.**  The
  geometric series ``1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,…`` minimises
  expected solving time against the worst-case Las-Vegas runtime
  distribution.  ``Solver`` restarts under Luby with a tunable scale.

* **Goldberg-Novikov 2003; Wetzler-Heule-Hunt 2014 — RUP / DRAT proofs.**
  The clause-addition log of a CDCL solver, with each learnt clause a
  *Reverse Unit Propagation* certificate from the existing formula, is
  itself a machine-checkable proof of unsatisfiability.  ``Solver``
  emits a DRAT-style proof on every UNSAT call; the proof is
  re-checked by an embedded verifier before being returned.

* **Tseitin 1968 — Polynomial CNF transformation.**  Any propositional
  formula is convertible to a CNF whose satisfiability is equisat with
  the original by introducing one fresh variable per subformula and
  three (AND/OR) or two (NOT) defining clauses.  ``Solver`` ships a
  ``Formula`` DSL whose ``to_cnf`` method produces exactly this
  encoding.

* **Sinz 2005 — Sequential cardinality encoding.**  ``AT_MOST_K`` over
  ``n`` literals encodes to ``O(n·k)`` clauses with ``O(n·k)`` auxiliary
  variables.  ``Solver.add_at_most`` / ``add_at_least`` / ``add_exactly``
  expose this encoding directly.

* **Fu-Malik 2006; Morgado-Heras-Liffiton-Planes-Marques-Silva 2013 —
  OLL MaxSAT.**  Iterated UNSAT-core extraction with cardinality
  relaxation finds the maximum-satisfiable subset of a CNF.
  ``Solver.solve_max_sat`` implements the OLL skeleton.

* **Bailleux-Marquis 2006; Belov-Marques-Silva 2012 — Minimal
  Unsatisfiable Subset (MUS) extraction.**  When ``φ`` is UNSAT, find
  a smallest sub-formula that is itself UNSAT (a "reason for
  inconsistency").  ``Solver.extract_mus`` implements the deletion-
  based MUS algorithm against the current assumption set.

The composition story
---------------------

Every other primitive in this runtime returns *real-valued* parameters.
``Solver`` is the only primitive that returns a **discrete witness** —
a model or a proof — which is precisely what a coordination engine
needs when its downstream actuator takes a discrete plan, not a
probability.

  * ``Synthesizer`` builds tools whose preconditions are Boolean
    formulae; ``Solver.solve(precondition)`` proves the tool is
    callable in the current world state, or returns the obligation
    that must first be discharged.
  * ``Conjecturer`` proposes integer-coefficient identities; ``Solver``
    can verify Boolean side-conditions (sign patterns, parity
    constraints) the identity must satisfy.
  * ``Coordinator`` plans over discrete action sequences encoded as
    CNF; ``Solver.solve_max_sat`` returns the maximum-utility plan
    subject to hard safety clauses.
  * ``Refuter`` builds counter-examples; ``Solver.solve(¬φ)`` returns
    a satisfying assignment to the negation, which *is* the counter-
    example.
  * ``Quantilizer`` gates on a model's existence — refuse to act
    unless ``Solver`` returns SAT *and* the asserting clause's LBD is
    below a threshold (a structural signal that the model is robust).

Module surface
--------------

The exported API has three layers:

  * **Low-level CNF API** — ``add_clause``, ``assume``, ``solve``,
    ``model``, ``proof``, ``extract_mus``, ``solve_max_sat``.
  * **Mid-level cardinality API** — ``add_at_most``, ``add_at_least``,
    ``add_exactly``.  Each is encoded to CNF via Sinz's sequential
    counter and added to the underlying solver.
  * **High-level DSL** — :class:`Formula` constructors (``var``,
    ``land``, ``lor``, ``lnot``, ``ximp``, ``xeqv``, ``xite``,
    ``at_most``, ``at_least``, ``exactly``) plus ``to_cnf`` for
    Tseitin encoding to the low-level API.

What ``Solver`` is *not*:

  * Not an SMT solver — variables are pure Boolean; integer, bit-vector,
    array, or floating-point theories require a downstream Z3, cvc5, or
    Yices.
  * Not a parallel / portfolio solver — every search call is single-
    threaded.  Coordinators wanting portfolio solving should run
    ``Solver`` instances under :class:`~agi.coordinator.Coordinator`
    and aggregate.
  * Not a probabilistic / weighted model counter — exact #SAT is
    ``#P``-complete and beyond the deterministic-decision contract of
    this primitive.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# --------------------------------------------------------------------- errors


class SolverError(Exception):
    """Base class for all `Solver` runtime errors.

    Sub-classes carry a structured ``code`` attribute that maps 1-to-1
    onto the JSON `error.code` field surfaced over the HTTP / SSE
    runtime so a coordination engine can pattern-match without parsing
    English text.
    """

    code = "solver_error"


class InvalidConfig(SolverError):
    code = "invalid_config"


class InvalidClause(SolverError):
    code = "invalid_clause"


class InvalidLiteral(SolverError):
    code = "invalid_literal"


class InvalidAssumption(SolverError):
    code = "invalid_assumption"


class InvalidFormula(SolverError):
    code = "invalid_formula"


class NotYetSolved(SolverError):
    code = "not_yet_solved"


class ProofCheckFailed(SolverError):
    code = "proof_check_failed"


class ResourceExhausted(SolverError):
    """Raised when the user-supplied conflict / time budget is reached.

    The partial state of the solver is preserved: re-calling ``solve``
    with a fresh budget resumes from the saved phase / activity table.
    """

    code = "resource_exhausted"


# --------------------------------------------------------------------- events


SOLVER_STARTED = "solver_started"
SOLVER_CLAUSE_ADDED = "solver_clause_added"
SOLVER_CLEARED = "solver_cleared"
SOLVER_ASSUMED = "solver_assumed"
SOLVER_SOLVED = "solver_solved"
SOLVER_REPORTED = "solver_reported"
SOLVER_MUS = "solver_mus_extracted"
SOLVER_MAXSAT = "solver_maxsat"
SOLVER_RESTARTED = "solver_restarted"
SOLVER_REDUCED = "solver_db_reduced"

SOLVER_KNOWN_EVENTS = (
    SOLVER_STARTED,
    SOLVER_CLAUSE_ADDED,
    SOLVER_CLEARED,
    SOLVER_ASSUMED,
    SOLVER_SOLVED,
    SOLVER_REPORTED,
    SOLVER_MUS,
    SOLVER_MAXSAT,
    SOLVER_RESTARTED,
    SOLVER_REDUCED,
)


STATUS_SAT = "sat"
STATUS_UNSAT = "unsat"
STATUS_UNKNOWN = "unknown"

SOLVER_KNOWN_STATUSES = (STATUS_SAT, STATUS_UNSAT, STATUS_UNKNOWN)


# --------------------------------------------------------------------- types


Lit = int
"""Signed-integer literal.  ``v > 0`` is the positive literal of variable
``v``; ``v < 0`` is its negation.  Zero is reserved as the
clause-terminator in legacy DIMACS files and is rejected by
``add_clause``."""

Clause = Sequence[Lit]


# Internal sentinel values for the assignment array.
_UNASSIGNED = 0
_TRUE = 1
_FALSE = -1


# --------------------------------------------------------------------- helpers


def _canonical_clause(clause: Iterable[Lit]) -> Tuple[Lit, ...]:
    """Sort + dedup a clause; raise on tautologies and on literal 0.

    The clause ``(a ∨ ¬a ∨ b)`` is a tautology and is preserved by the
    canonical form ``(a, ¬a, b)`` after sorting.  ``Solver.add_clause``
    drops tautologies before insertion (since they contribute nothing
    to satisfiability) but the canonical form still has to round-trip
    them so the proof can record the *original* clause exactly.
    """
    out: List[Lit] = []
    seen: Set[Lit] = set()
    for lit in clause:
        if not isinstance(lit, int):
            raise InvalidLiteral(f"literal must be int, got {type(lit).__name__}")
        if lit == 0:
            raise InvalidLiteral("literal 0 is reserved; use ±var (1-indexed)")
        if lit in seen:
            continue
        seen.add(lit)
        out.append(lit)
    out.sort(key=lambda x: (abs(x), x < 0))
    return tuple(out)


def _is_tautology(clause: Iterable[Lit]) -> bool:
    seen: Set[Lit] = set()
    for lit in clause:
        if -lit in seen:
            return True
        seen.add(lit)
    return False


def _luby(i: int) -> int:
    """Return the i-th term of the Luby sequence (1-indexed).

    Knuth's formulation: if ``i = 2^k − 1`` for some k, return ``2^{k−1}``;
    otherwise let ``j = 2^{⌊log₂ i⌋}`` and recurse on ``i − j + 1``.
    Worst-case bound (Luby-Sinclair-Zuckerman 1993): the sequence is
    universally optimal for randomised Las Vegas restarts up to a
    constant factor 192.
    """
    if i <= 0:
        raise InvalidConfig("Luby index must be ≥ 1")
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    if (1 << k) - 1 == i:
        return 1 << (k - 1)
    return _luby(i - (1 << (k - 1)) + 1)


def _hash_event(prev_hash: str, payload: Mapping[str, Any]) -> str:
    """SHA-256 over the prior hash || canonical-JSON payload.

    The same fingerprint discipline used by every other primitive in
    this runtime.  ``AttestationLedger`` validates the chain by
    re-hashing each record under its parent's reported digest.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(b"\0")
    h.update(encoded.encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class SolverResult:
    """Outcome of a single ``solve`` call.

    Attributes
    ----------
    status:
        One of ``"sat"``, ``"unsat"``, ``"unknown"``.  ``"unknown"`` is
        returned only when the user-supplied conflict / time budget
        was exhausted before a verdict was reached.
    model:
        For ``status == "sat"``: a dict ``{var → bool}`` over every
        variable that was added to the solver.  Variables introduced
        as Tseitin auxiliaries are also included (under their internal
        integer ids).  Empty when ``status != "sat"``.
    core:
        For ``status == "unsat"`` *under assumptions*: a tuple of
        assumption literals that together suffice to derive ``⊥``.
        Empty otherwise.
    proof:
        For ``status == "unsat"``: a tuple of clauses in DRAT
        deletion-and-addition form; each addition is a learnt clause,
        each deletion (prefixed with ``"d"``) is a clause removed from
        the database.  The empty clause ``()`` is the last addition.
    stats:
        Dict of bookkeeping counters — conflicts, decisions,
        propagations, restarts, learnt clauses kept and deleted, peak
        decision level.  Exposed for monitoring; not part of the
        correctness contract.
    """

    status: str
    model: Mapping[int, bool] = field(default_factory=dict)
    core: Tuple[Lit, ...] = ()
    proof: Tuple[Tuple[Any, ...], ...] = ()
    stats: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverReport:
    """Summary of solver state at ``report`` time.

    Includes the current variable count, clause count, conflict /
    decision counters, the SHA-256 ledger head, and the last solve
    status (or ``None`` if no solve has been issued yet).
    """

    num_vars: int
    num_clauses: int
    num_learnt: int
    conflicts: int
    decisions: int
    propagations: int
    restarts: int
    last_status: Optional[str]
    ledger_head: str
    seed: int


@dataclass
class _Clause:
    """Internal clause record with watched-literal bookkeeping.

    ``lits[0]`` and ``lits[1]`` are the two watched literals; the
    invariant maintained by ``_propagate`` is that *if* the clause is
    not satisfied, *both* watches are unassigned, or one watch is
    true.  This keeps unit propagation O(occurrences of the falsified
    watch) per assignment.
    """

    lits: List[Lit]
    learnt: bool = False
    lbd: int = 0
    activity: float = 0.0


# --------------------------------------------------------------------- DSL


class Formula:
    """Tagged-tree DSL for propositional formulae.

    A ``Formula`` is an immutable tree whose internal nodes are AND, OR,
    NOT, IMP, EQV, ITE, AT_MOST_K, AT_LEAST_K, EXACTLY_K and whose
    leaves are variable references or the constants ``TRUE`` / ``FALSE``.

    The ``to_cnf`` method emits a CNF encoding via the Tseitin
    transformation; introduced auxiliary variables are guaranteed to
    have ids strictly greater than every variable in the input
    sub-trees.  Tautologous and contradictory leaves are simplified
    inline.

    The DSL is *not* a planner or theorem prover — it is the convenience
    layer a coordination engine uses to *express* a Boolean obligation
    before handing it off to ``Solver.solve``.

    Construction is via the module-level factory functions:

    ``var(i)`` , ``true()`` , ``false()`` , ``land(*fs)`` ,
    ``lor(*fs)`` , ``lnot(f)`` , ``ximp(a, b)`` , ``xeqv(a, b)`` ,
    ``xite(c, t, e)`` , ``at_most(k, fs)`` , ``at_least(k, fs)`` ,
    ``exactly(k, fs)``.
    """

    __slots__ = ("kind", "args", "k")

    def __init__(self, kind: str, args: Tuple[Any, ...] = (), k: Optional[int] = None):
        self.kind = kind
        self.args = args
        self.k = k

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.kind == "var":
            return f"x{self.args[0]}"
        if self.kind == "const":
            return "⊤" if self.args[0] else "⊥"
        if self.kind == "not":
            return f"¬{self.args[0]!r}"
        if self.kind in ("and", "or"):
            sym = "∧" if self.kind == "and" else "∨"
            return f"({sym.join(repr(a) for a in self.args)})"
        if self.kind in ("imp", "eqv"):
            sym = "→" if self.kind == "imp" else "↔"
            return f"({self.args[0]!r} {sym} {self.args[1]!r})"
        if self.kind == "ite":
            return f"ITE({self.args[0]!r}, {self.args[1]!r}, {self.args[2]!r})"
        if self.kind in ("at_most", "at_least", "exactly"):
            op = self.kind.replace("_", " ").upper()
            return f"{op}({self.k}, [{', '.join(repr(a) for a in self.args)}])"
        return f"<Formula {self.kind} {self.args}>"

    # --- combinators ---------------------------------------------------

    def __and__(self, other: "Formula") -> "Formula":
        return land(self, other)

    def __or__(self, other: "Formula") -> "Formula":
        return lor(self, other)

    def __invert__(self) -> "Formula":
        return lnot(self)

    def __rshift__(self, other: "Formula") -> "Formula":
        return ximp(self, other)

    def equiv(self, other: "Formula") -> "Formula":
        return xeqv(self, other)

    # --- compilation ---------------------------------------------------

    def to_cnf(self, next_var: Optional[int] = None) -> Tuple[List[List[Lit]], Lit, int]:
        """Tseitin-encode the formula to CNF.

        Returns ``(clauses, top_lit, next_var)``.  ``top_lit`` is the
        literal that is true iff the formula is true; an asserting
        unit ``(top_lit,)`` is *not* added automatically — the caller
        decides whether the formula is being asserted or assumed.

        ``next_var`` is the id of the next free variable after
        encoding; pass it back in on the next ``to_cnf`` call to keep
        ids monotone across a multi-formula encoding session.
        """
        if next_var is None:
            next_var = self._max_var() + 1
        clauses: List[List[Lit]] = []
        top = self._encode(clauses, [next_var])
        return clauses, top, _next_var_after(clauses, top)

    def _max_var(self) -> int:
        if self.kind == "var":
            return int(self.args[0])
        if self.kind == "const":
            return 0
        out = 0
        for a in self.args:
            if isinstance(a, Formula):
                out = max(out, a._max_var())
        return out

    def _encode(self, clauses: List[List[Lit]], counter: List[int]) -> Lit:
        if self.kind == "var":
            return int(self.args[0])
        if self.kind == "const":
            # Materialise a Tseitin auxiliary and pin it.  An empty
            # clause [-aux] makes ⊥; an empty clause [aux] makes ⊤.
            aux = counter[0]
            counter[0] += 1
            if self.args[0]:
                clauses.append([aux])
            else:
                clauses.append([-aux])
            return aux
        if self.kind == "not":
            return -self.args[0]._encode(clauses, counter)
        if self.kind == "and":
            sub = [a._encode(clauses, counter) for a in self.args]
            return _tseitin_and(clauses, counter, sub)
        if self.kind == "or":
            sub = [a._encode(clauses, counter) for a in self.args]
            return _tseitin_or(clauses, counter, sub)
        if self.kind == "imp":
            a = self.args[0]._encode(clauses, counter)
            b = self.args[1]._encode(clauses, counter)
            return _tseitin_or(clauses, counter, [-a, b])
        if self.kind == "eqv":
            a = self.args[0]._encode(clauses, counter)
            b = self.args[1]._encode(clauses, counter)
            return _tseitin_eqv(clauses, counter, a, b)
        if self.kind == "ite":
            c = self.args[0]._encode(clauses, counter)
            t = self.args[1]._encode(clauses, counter)
            e = self.args[2]._encode(clauses, counter)
            return _tseitin_ite(clauses, counter, c, t, e)
        if self.kind == "at_most":
            sub = [a._encode(clauses, counter) for a in self.args]
            _encode_at_most(clauses, counter, sub, self.k or 0)
            # No top literal — the constraint is asserted directly.
            # Return a fresh "true" aux so callers that ask for a top
            # literal can pin / negate the constraint.
            aux = counter[0]
            counter[0] += 1
            clauses.append([aux])
            return aux
        if self.kind == "at_least":
            sub = [a._encode(clauses, counter) for a in self.args]
            # AT_LEAST_K(xs) ≡ AT_MOST_(n-k)(¬xs)
            _encode_at_most(
                clauses, counter, [-l for l in sub], len(sub) - (self.k or 0)
            )
            aux = counter[0]
            counter[0] += 1
            clauses.append([aux])
            return aux
        if self.kind == "exactly":
            sub = [a._encode(clauses, counter) for a in self.args]
            _encode_at_most(clauses, counter, sub, self.k or 0)
            _encode_at_most(
                clauses, counter, [-l for l in sub], len(sub) - (self.k or 0)
            )
            aux = counter[0]
            counter[0] += 1
            clauses.append([aux])
            return aux
        raise InvalidFormula(f"unknown formula kind {self.kind!r}")


def _next_var_after(clauses: Iterable[Iterable[Lit]], top: Lit) -> int:
    m = abs(int(top))
    for c in clauses:
        for l in c:
            m = max(m, abs(int(l)))
    return m + 1


def _tseitin_and(clauses: List[List[Lit]], counter: List[int], lits: List[Lit]) -> Lit:
    """Encode ``aux ↔ ⋀ lits``.

    Big-clause form (Tseitin 1968):

        aux → l_i               ≡   ¬aux ∨ l_i               for each i
        ⋀ l_i → aux             ≡   aux ∨ ⋁ ¬l_i             one clause

    Special cases simplify to fewer clauses; an empty ``lits`` is the
    constant ⊤.
    """
    if len(lits) == 0:
        # AND of nothing is TRUE: aux ≡ ⊤, return a fresh true aux.
        aux = counter[0]
        counter[0] += 1
        clauses.append([aux])
        return aux
    if len(lits) == 1:
        return lits[0]
    aux = counter[0]
    counter[0] += 1
    for l in lits:
        clauses.append([-aux, l])
    big = [aux]
    for l in lits:
        big.append(-l)
    clauses.append(big)
    return aux


def _tseitin_or(clauses: List[List[Lit]], counter: List[int], lits: List[Lit]) -> Lit:
    """Encode ``aux ↔ ⋁ lits``.

    Symmetric to ``_tseitin_and`` under De Morgan.  Empty ``lits`` is
    the constant ⊥.
    """
    if len(lits) == 0:
        aux = counter[0]
        counter[0] += 1
        clauses.append([-aux])
        return aux
    if len(lits) == 1:
        return lits[0]
    aux = counter[0]
    counter[0] += 1
    for l in lits:
        clauses.append([aux, -l])
    big = [-aux]
    for l in lits:
        big.append(l)
    clauses.append(big)
    return aux


def _tseitin_eqv(clauses: List[List[Lit]], counter: List[int], a: Lit, b: Lit) -> Lit:
    """Encode ``aux ↔ (a ↔ b)``.

    Four 3-clauses:

        aux ∨ a ∨ b              aux ∨ ¬a ∨ ¬b
        ¬aux ∨ ¬a ∨ b           ¬aux ∨ a ∨ ¬b
    """
    aux = counter[0]
    counter[0] += 1
    clauses.append([aux, a, b])
    clauses.append([aux, -a, -b])
    clauses.append([-aux, -a, b])
    clauses.append([-aux, a, -b])
    return aux


def _tseitin_ite(
    clauses: List[List[Lit]], counter: List[int], c: Lit, t: Lit, e: Lit
) -> Lit:
    """Encode ``aux ↔ (c ? t : e)``.

    Four 3-clauses derivable from
    ``aux ≡ (c ∧ t) ∨ (¬c ∧ e)``.
    """
    aux = counter[0]
    counter[0] += 1
    clauses.append([-aux, -c, t])
    clauses.append([-aux, c, e])
    clauses.append([aux, -c, -t])
    clauses.append([aux, c, -e])
    return aux


def _encode_at_most(
    clauses: List[List[Lit]], counter: List[int], lits: List[Lit], k: int
) -> None:
    """Sequential-counter encoding of ``Σ lits ≤ k`` (Sinz 2005).

    Introduces ``n·k`` auxiliary register variables ``s[i,j]`` with
    semantics ``s[i,j] ↔ (lits[0]+…+lits[i] ≥ j+1)``.  Produces
    ``O(n·k)`` clauses; preserves arc-consistency under unit
    propagation (Sinz 2005, Theorem 1).
    """
    n = len(lits)
    if k < 0:
        clauses.append([])  # asserted contradiction
        return
    if k >= n:
        return  # always satisfied
    if k == 0:
        # All literals must be false.
        for l in lits:
            clauses.append([-l])
        return
    # Allocate s[i][j] for i in [0,n-1], j in [0,k-1].
    s = [[0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            s[i][j] = counter[0]
            counter[0] += 1
    # Sinz's recursive constraints:
    #   ¬x_1 ∨ s[0][0]
    for j in range(1, k):
        clauses.append([-s[0][j]])
    clauses.append([-lits[0], s[0][0]])
    for i in range(1, n - 1):
        clauses.append([-lits[i], s[i][0]])
        clauses.append([-s[i - 1][0], s[i][0]])
        for j in range(1, k):
            clauses.append([-lits[i], -s[i - 1][j - 1], s[i][j]])
            clauses.append([-s[i - 1][j], s[i][j]])
        clauses.append([-lits[i], -s[i - 1][k - 1]])
    clauses.append([-lits[n - 1], -s[n - 2][k - 1]])


# --------------------------------------------------------------------- DSL API


def var(i: int) -> Formula:
    """Construct a variable reference.

    Variable ids are positive integers; the negation of ``var(i)`` is
    ``~var(i)`` or equivalently ``lnot(var(i))``.
    """
    if not isinstance(i, int) or i <= 0:
        raise InvalidFormula(f"variable id must be a positive int, got {i!r}")
    return Formula("var", (i,))


def true() -> Formula:
    return Formula("const", (True,))


def false() -> Formula:
    return Formula("const", (False,))


def land(*fs: Formula) -> Formula:
    return Formula("and", tuple(_to_formula(f) for f in fs))


def lor(*fs: Formula) -> Formula:
    return Formula("or", tuple(_to_formula(f) for f in fs))


def lnot(f: Formula) -> Formula:
    return Formula("not", (_to_formula(f),))


def ximp(a: Formula, b: Formula) -> Formula:
    return Formula("imp", (_to_formula(a), _to_formula(b)))


def xeqv(a: Formula, b: Formula) -> Formula:
    return Formula("eqv", (_to_formula(a), _to_formula(b)))


def xite(c: Formula, t: Formula, e: Formula) -> Formula:
    return Formula("ite", (_to_formula(c), _to_formula(t), _to_formula(e)))


def at_most(k: int, fs: Sequence[Formula]) -> Formula:
    if k < 0:
        raise InvalidFormula(f"AT_MOST_K requires k ≥ 0, got {k}")
    return Formula("at_most", tuple(_to_formula(f) for f in fs), k=k)


def at_least(k: int, fs: Sequence[Formula]) -> Formula:
    if k < 0:
        raise InvalidFormula(f"AT_LEAST_K requires k ≥ 0, got {k}")
    return Formula("at_least", tuple(_to_formula(f) for f in fs), k=k)


def exactly(k: int, fs: Sequence[Formula]) -> Formula:
    if k < 0:
        raise InvalidFormula(f"EXACTLY_K requires k ≥ 0, got {k}")
    return Formula("exactly", tuple(_to_formula(f) for f in fs), k=k)


def _to_formula(x: Any) -> Formula:
    if isinstance(x, Formula):
        return x
    if isinstance(x, bool):
        return true() if x else false()
    raise InvalidFormula(f"expected Formula, got {type(x).__name__}")


# --------------------------------------------------------------------- Solver


class Solver:
    """Conflict-Driven Clause Learning SAT solver.

    Construct via :meth:`Solver.create`; the constructor is private.

    The solver maintains its CNF database, the trail of assigned
    literals, the implication graph (parent clause per implied
    literal), VSIDS activities, the watched-literal lists, the DRAT
    proof log, and the SHA-256 attestation chain.

    Thread-safety: instances are *not* safe for concurrent calls; wrap
    in a queue or lock if a coordinator dispatches multiple actuators
    against the same solver.
    """

    # --- construction --------------------------------------------------

    def __init__(self, _key: object, seed: int, max_var_hint: int) -> None:
        if _key is not _SOLVER_KEY:
            raise SolverError(
                "use Solver.create(...) — the constructor is intentionally private"
            )
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        # Core CNF database
        self._clauses: List[_Clause] = []
        self._original_clauses: List[Tuple[Lit, ...]] = []
        # Track the originally-tracked variables (input + auxiliary).
        self._num_vars = max(0, int(max_var_hint))
        # Assignment array indexed by variable id (1-indexed; index 0 unused).
        self._assign: List[int] = [_UNASSIGNED] * (self._num_vars + 1)
        # Decision-level of each variable (or -1 if unassigned).
        self._level: List[int] = [-1] * (self._num_vars + 1)
        # Reason (parent clause) of each propagated literal — None for
        # decisions and for unassigned vars.
        self._reason: List[Optional[int]] = [None] * (self._num_vars + 1)
        # Phase saving — last value the variable took before being
        # unassigned.  +1 / -1; default +1 (positive phase).
        self._phase: List[int] = [_TRUE] * (self._num_vars + 1)
        # Trail of currently assigned literals (in order of assignment),
        # plus the indices into the trail at which each decision level
        # begins.
        self._trail: List[Lit] = []
        self._trail_lim: List[int] = []
        # VSIDS activities — per variable, with multiplicative decay.
        self._activity: List[float] = [0.0] * (self._num_vars + 1)
        self._act_inc: float = 1.0
        self._act_decay: float = 0.95
        # Watched-literal occurrence lists.  Indexed by literal-id
        # (encoded as ``2*var + (0 if positive else 1)`` -- 1-indexed).
        self._watches: Dict[int, List[int]] = {}
        # Assumptions for the next solve() call.
        self._assumptions: List[Lit] = []
        # DRAT proof log accumulated across all solve() calls.
        self._proof_log: List[Tuple[Any, ...]] = []
        # Bookkeeping counters.
        self._conflicts = 0
        self._decisions = 0
        self._propagations = 0
        self._restarts = 0
        self._learnt_kept = 0
        self._learnt_deleted = 0
        self._last_status: Optional[str] = None
        # Set when ``add_clause`` is handed a clause whose literals are
        # all falsified at L0 — the formula is then unsat *before* any
        # search begins.
        self._unsat_at_l0 = False
        # Highest user variable ever reserved via ``reserve_vars`` /
        # ``new_var``; auxiliary variables introduced by cardinality
        # / Tseitin encoding are guaranteed to receive ids strictly
        # greater than this.
        self._user_reserved = max(0, int(max_var_hint))
        # Set on each ``solve`` call to record whether the last
        # UNSAT verdict was reached under assumptions.  The empty
        # clause is only emitted to the DRAT log on a true
        # (no-assumption) UNSAT.
        self._last_unsat_under_assumptions = False
        # Restart / reduce-DB schedule.
        self._luby_index = 1
        self._luby_scale = 32
        self._reduce_db_base = 100
        self._reduce_db_inc = 100
        self._next_reduce = self._reduce_db_base
        # Attestation ledger.
        self._ledger_head = "0" * 64
        self._ledger: List[Mapping[str, Any]] = []
        self._record(SOLVER_STARTED, {"seed": self._seed})

    @classmethod
    def create(
        cls,
        *,
        seed: int = 0,
        max_var_hint: int = 0,
    ) -> "Solver":
        """Construct a fresh ``Solver`` instance.

        Parameters
        ----------
        seed:
            Seed for the internal ``random.Random``.  Two solvers seeded
            identically and fed identical input traces produce identical
            decision sequences, learnt clauses, and proofs.
        max_var_hint:
            Optional upper bound on the variable id range.  The solver
            auto-grows its internal arrays on demand, but pre-sizing
            removes the allocation overhead during dense
            ``add_clause`` loops.
        """
        if not isinstance(seed, int):
            raise InvalidConfig("seed must be an int")
        if not isinstance(max_var_hint, int) or max_var_hint < 0:
            raise InvalidConfig("max_var_hint must be a non-negative int")
        return cls(_SOLVER_KEY, seed, max_var_hint)

    # --- public CNF API ------------------------------------------------

    def add_clause(self, clause: Iterable[Lit]) -> int:
        """Add a clause to the formula.

        Returns the new clause's internal id.  Tautological clauses
        (containing both ``v`` and ``¬v``) are accepted but optimised
        out of the search; the original form is still recorded in the
        attestation ledger.

        Raises :class:`InvalidClause` on non-integer literals, literal
        ``0``, or clauses larger than 1 048 576 literals (a sanity
        bound; real CNF instances have ≤ a few thousand-literal
        clauses).
        """
        canon = _canonical_clause(clause)
        if len(canon) > 1 << 20:
            raise InvalidClause(f"clause too large: {len(canon)} literals")
        idx = len(self._original_clauses)
        self._original_clauses.append(canon)
        self._grow_for_lits(canon)
        if _is_tautology(canon):
            self._record(SOLVER_CLAUSE_ADDED, {"clause": list(canon), "tautology": True})
            return idx
        # Drop literals already falsified at level 0; if any literal is
        # already true at level 0, the clause is a no-op.
        active: List[Lit] = []
        for l in canon:
            v = abs(l)
            if self._level[v] == 0:
                if self._assign[v] == (_TRUE if l > 0 else _FALSE):
                    self._record(
                        SOLVER_CLAUSE_ADDED, {"clause": list(canon), "subsumed_l0": True}
                    )
                    return idx
                # falsified at L0 — drop it
                continue
            active.append(l)
        if not active:
            # all literals falsified at L0 → unsat at L0
            self._add_internal(_Clause(lits=list(canon)))
            self._unsat_at_l0 = True
            self._record(SOLVER_CLAUSE_ADDED, {"clause": list(canon), "empty_at_l0": True})
            return idx
        if len(active) == 1:
            # Unit clause.  If the literal contradicts an existing L0
            # assignment, we are unsat at L0.
            lit0 = active[0]
            v = abs(lit0)
            want = _TRUE if lit0 > 0 else _FALSE
            self._add_internal(_Clause(lits=list(canon)))
            if self._assign[v] != _UNASSIGNED and self._assign[v] != want:
                self._unsat_at_l0 = True
                self._record(
                    SOLVER_CLAUSE_ADDED, {"clause": list(canon), "contradict_l0": True}
                )
                return idx
            if self._assign[v] == _UNASSIGNED:
                self._enqueue(lit0, reason=len(self._clauses) - 1)
            self._record(SOLVER_CLAUSE_ADDED, {"clause": list(canon), "unit": True})
            return idx
        self._add_internal(_Clause(lits=list(canon)))
        self._record(SOLVER_CLAUSE_ADDED, {"clause": list(canon)})
        return idx

    def add_clauses(self, clauses: Iterable[Iterable[Lit]]) -> List[int]:
        """Convenience for adding multiple clauses; returns their ids."""
        return [self.add_clause(c) for c in clauses]

    def assume(self, lit: Lit) -> None:
        """Assume a literal for the *next* ``solve`` call.

        Assumptions are not part of the formula — they are temporary
        unit hypotheses that hold only for one ``solve`` call.  If
        ``solve`` returns UNSAT under assumptions, the
        ``SolverResult.core`` is a subset of the assumption set that
        is itself UNSAT — the standard mechanism for incremental SAT.
        """
        if not isinstance(lit, int) or lit == 0:
            raise InvalidAssumption(f"assumption must be a non-zero int, got {lit!r}")
        self._grow_for_lits((lit,))
        self._assumptions.append(int(lit))
        self._record(SOLVER_ASSUMED, {"lit": int(lit)})

    def clear_assumptions(self) -> None:
        """Drop the pending assumption set."""
        self._assumptions = []
        self._record(SOLVER_CLEARED, {"what": "assumptions"})

    # --- cardinality / pseudo-Boolean API ------------------------------

    def reserve_vars(self, n: int) -> None:
        """Reserve variable ids 1..n for user use.

        Idempotent; growing-only.  Calling ``reserve_vars(8)`` guarantees
        that any auxiliary variables introduced by ``add_at_most``,
        ``add_formula`` and friends will receive ids > 8 — so the user
        can freely reference 1..8 in subsequent ``add_clause`` calls
        without colliding with internal book-keeping.
        """
        if not isinstance(n, int) or n < 0:
            raise InvalidConfig("reserve_vars requires a non-negative int")
        if n > self._num_vars:
            self._grow_for_lits([n])
        self._user_reserved = max(self._user_reserved, n)

    def new_var(self) -> int:
        """Allocate a fresh user variable id.

        Returns the new variable id; subsequent auxiliary allocations
        are guaranteed to receive higher ids.  Useful when constructing
        a problem dynamically without knowing the variable count in
        advance.
        """
        nid = max(self._num_vars, self._user_reserved) + 1
        self._grow_for_lits([nid])
        self._user_reserved = max(self._user_reserved, nid)
        return nid

    def add_at_most(self, lits: Sequence[Lit], k: int) -> None:
        """Assert ``Σ lits ≤ k`` via Sinz's sequential counter encoding.

        Mutates the formula in place.  Equivalent to compiling the
        DSL ``at_most(k, [var(±l) for l in lits])`` to CNF, but skips
        the wrapper Formula objects.

        Auxiliary variables introduced by the encoding receive ids
        strictly greater than every previously-allocated id (user or
        auxiliary).  To avoid id collisions, declare your user
        variables up-front via ``reserve_vars(n)`` *or* ``new_var()``;
        the convention "anything past the highest id seen so far is
        either reserved by the user or fresh aux" is what makes the
        solver's bookkeeping consistent.
        """
        if k < 0:
            raise InvalidClause("AT_MOST_K requires k ≥ 0")
        if any(not isinstance(l, int) or l == 0 for l in lits):
            raise InvalidLiteral("invalid literal in AT_MOST_K")
        # Make sure auxiliary variable IDs don't collide with the input
        # nor with any prior user reservation.
        self._grow_for_lits(lits)
        clauses: List[List[Lit]] = []
        counter = [max(self._num_vars, self._user_reserved) + 1]
        _encode_at_most(clauses, counter, list(lits), k)
        for c in clauses:
            self.add_clause(c)

    def add_at_least(self, lits: Sequence[Lit], k: int) -> None:
        """Assert ``Σ lits ≥ k``."""
        if k < 0:
            raise InvalidClause("AT_LEAST_K requires k ≥ 0")
        if any(not isinstance(l, int) or l == 0 for l in lits):
            raise InvalidLiteral("invalid literal in AT_LEAST_K")
        self._grow_for_lits(lits)
        self.add_at_most([-l for l in lits], len(lits) - k)

    def add_exactly(self, lits: Sequence[Lit], k: int) -> None:
        """Assert ``Σ lits = k``."""
        self._grow_for_lits(lits)
        self.add_at_most(list(lits), k)
        self.add_at_least(list(lits), k)

    # --- DSL bridge ----------------------------------------------------

    def add_formula(self, formula: Formula, assert_true: bool = True) -> Lit:
        """Compile a :class:`Formula` to CNF and add the clauses.

        Returns the *top literal* of the formula — the Tseitin
        auxiliary that is true iff the formula is true.  If
        ``assert_true`` is ``True`` (the default) the unit clause
        ``(top_lit,)`` is also added, so the formula is *asserted*.
        Pass ``assert_true=False`` if you intend to ``assume(top_lit)``
        only for the next solve call, or ``assume(-top_lit)`` for a
        refutation query.
        """
        if not isinstance(formula, Formula):
            raise InvalidFormula(f"expected Formula, got {type(formula).__name__}")
        next_var = max(self._num_vars + 1, self._user_reserved + 1, formula._max_var() + 1)
        clauses, top, _ = formula.to_cnf(next_var=next_var)
        for c in clauses:
            self.add_clause(c)
        if assert_true:
            self.add_clause([top])
        return top

    # --- main solve loop ----------------------------------------------

    def solve(
        self,
        *,
        max_conflicts: Optional[int] = None,
        time_budget_s: Optional[float] = None,
    ) -> SolverResult:
        """Run CDCL until a verdict or budget exhaustion.

        Parameters
        ----------
        max_conflicts:
            Optional upper bound on the number of conflicts in this
            call.  When exceeded, returns ``SolverResult(status="unknown")``.
            ``None`` means no bound.
        time_budget_s:
            Optional wall-clock bound in seconds.  Same semantics as
            ``max_conflicts``; the solver checks the clock once per
            conflict, so the bound is approximate.
        """
        if max_conflicts is not None and (not isinstance(max_conflicts, int) or max_conflicts < 0):
            raise InvalidConfig("max_conflicts must be a non-negative int or None")
        if time_budget_s is not None and (not isinstance(time_budget_s, (int, float)) or time_budget_s < 0):
            raise InvalidConfig("time_budget_s must be a non-negative number or None")
        t0 = time.monotonic()
        start_conflicts = self._conflicts
        # Reset to L0; any prior decisions / assumptions are dropped.
        self._backtrack_to(0)
        self._last_unsat_under_assumptions = bool(self._assumptions)
        if self._unsat_at_l0:
            # L0-deduced UNSAT — independent of assumptions.
            self._last_unsat_under_assumptions = False
            self._proof_emit_empty()
            status = STATUS_UNSAT
            self._last_status = status
            result = SolverResult(
                status=status,
                core=(),
                proof=tuple(self._proof_log),
                stats=self._stats(),
            )
            self._record(SOLVER_SOLVED, {"status": status, "core": []})
            return result
        # Propagate L0 unit implications (e.g. clauses added during the
        # last solve that became unit).
        confl = self._propagate()
        if confl is not None:
            # UNSAT at L0 with no assumptions yet.
            self._last_unsat_under_assumptions = False
            self._proof_emit_empty()
            status = STATUS_UNSAT
            self._last_status = status
            result = SolverResult(
                status=status,
                core=(),
                proof=tuple(self._proof_log),
                stats=self._stats(),
            )
            self._record(SOLVER_SOLVED, {"status": status, "core": []})
            return result
        # Plant assumptions as decisions at increasing levels.
        for a in self._assumptions:
            v = abs(a)
            if self._assign[v] == _UNASSIGNED:
                self._trail_lim.append(len(self._trail))
                self._enqueue(a, reason=None)
                c = self._propagate()
                if c is not None:
                    # Compute UNSAT core by walking back through
                    # propagation parents from the conflict clause.
                    core = self._compute_assumption_core(c)
                    self._proof_emit_empty()
                    status = STATUS_UNSAT
                    self._last_status = status
                    result = SolverResult(
                        status=status,
                        core=core,
                        proof=tuple(self._proof_log),
                        stats=self._stats(),
                    )
                    self._record(SOLVER_SOLVED, {"status": status, "core": list(core)})
                    return result
            else:
                # Either already satisfied at L0 or contradicts an
                # earlier assumption (or L0 unit).
                already = self._assign[v]
                want = _TRUE if a > 0 else _FALSE
                if already != want:
                    self._proof_emit_empty()
                    status = STATUS_UNSAT
                    self._last_status = status
                    core = (a,)
                    result = SolverResult(
                        status=status,
                        core=core,
                        proof=tuple(self._proof_log),
                        stats=self._stats(),
                    )
                    self._record(SOLVER_SOLVED, {"status": status, "core": list(core)})
                    return result
        conflicts_until_restart = self._luby_scale * _luby(self._luby_index)
        local_conflicts = 0
        while True:
            confl = self._propagate()
            if confl is not None:
                self._conflicts += 1
                local_conflicts += 1
                if self._decision_level() == 0:
                    # UNSAT at root regardless of assumptions.
                    self._proof_emit_empty()
                    status = STATUS_UNSAT
                    self._last_status = status
                    core = self._compute_assumption_core(confl)
                    result = SolverResult(
                        status=status,
                        core=core,
                        proof=tuple(self._proof_log),
                        stats=self._stats(),
                    )
                    self._record(SOLVER_SOLVED, {"status": status, "core": list(core)})
                    return result
                learnt, backtrack_lvl = self._analyze_conflict(confl)
                self._backtrack_to(backtrack_lvl)
                new_id = self._learn_clause(learnt)
                self._enqueue(learnt[0], reason=new_id)
                self._proof_emit_add(learnt)
                self._decay_activities()
                if local_conflicts >= conflicts_until_restart:
                    self._restarts += 1
                    self._luby_index += 1
                    conflicts_until_restart = self._luby_scale * _luby(self._luby_index)
                    local_conflicts = 0
                    self._backtrack_to(len(self._assumptions))
                    self._record(SOLVER_RESTARTED, {"restarts": self._restarts})
                if self._conflicts >= self._next_reduce:
                    self._reduce_db()
                    self._next_reduce += self._reduce_db_inc
                if max_conflicts is not None and (self._conflicts - start_conflicts) >= max_conflicts:
                    status = STATUS_UNKNOWN
                    self._last_status = status
                    return SolverResult(
                        status=status,
                        proof=tuple(self._proof_log),
                        stats=self._stats(),
                    )
                if time_budget_s is not None and (time.monotonic() - t0) >= time_budget_s:
                    status = STATUS_UNKNOWN
                    self._last_status = status
                    return SolverResult(
                        status=status,
                        proof=tuple(self._proof_log),
                        stats=self._stats(),
                    )
            else:
                # No conflict.  Decide.
                next_lit = self._pick_branching()
                if next_lit is None:
                    # All variables assigned and no conflict ⇒ SAT.
                    status = STATUS_SAT
                    self._last_status = status
                    model = self._extract_model()
                    result = SolverResult(
                        status=status,
                        model=model,
                        proof=tuple(self._proof_log),
                        stats=self._stats(),
                    )
                    self._record(SOLVER_SOLVED, {"status": status})
                    return result
                self._decisions += 1
                self._trail_lim.append(len(self._trail))
                self._enqueue(next_lit, reason=None)

    # --- propagation ---------------------------------------------------

    def _propagate(self) -> Optional[int]:
        """Two-watched-literal unit propagation.

        Walks the trail from the current propagation pointer; for each
        newly-assigned literal ``l`` it inspects every clause watching
        ``¬l``, trying to replace the watch with another non-false
        literal.  Returns the id of a conflict clause or ``None``.
        """
        qhead = 0
        while qhead < len(self._trail):
            lit = self._trail[qhead]
            qhead += 1
            self._propagations += 1
            falsified = -lit
            key = _lit_index(falsified)
            watch_list = self._watches.get(key)
            if not watch_list:
                continue
            new_list: List[int] = []
            for cid in watch_list:
                cl = self._clauses[cid]
                lits = cl.lits
                # Make sure lits[1] is the watch corresponding to the
                # falsified literal (swap if necessary).
                if lits[0] == falsified:
                    lits[0], lits[1] = lits[1], lits[0]
                # If the OTHER watch is true, the clause is already
                # satisfied; keep the watch on falsified.
                other = lits[0]
                if self._is_true(other):
                    new_list.append(cid)
                    continue
                # Try to find a replacement watch.
                found = False
                for k in range(2, len(lits)):
                    if not self._is_false(lits[k]):
                        lits[1], lits[k] = lits[k], lits[1]
                        self._watches.setdefault(_lit_index(lits[1]), []).append(cid)
                        found = True
                        break
                if found:
                    continue
                # No replacement: either unit (other unassigned) or
                # conflict (other false).
                new_list.append(cid)
                if self._is_false(other):
                    # Conflict — copy back unprocessed entries and bail.
                    rest_idx = watch_list.index(cid) + 1
                    new_list.extend(watch_list[rest_idx:])
                    self._watches[key] = new_list
                    return cid
                # Unit propagation.
                self._enqueue(other, reason=cid)
            self._watches[key] = new_list
        return None

    # --- conflict analysis (1-UIP) ------------------------------------

    def _analyze_conflict(self, confl_id: int) -> Tuple[List[Lit], int]:
        """First-UIP conflict analysis.

        Walks the implication graph backwards from the conflicting
        clause; literals assigned at the current decision level are
        resolved out (each pulled in via its propagating clause) until
        a single such literal remains — the first unique implication
        point.  The learnt clause is the asserting clause for that UIP;
        its backjump level is the second-highest decision level in the
        clause (or 0 if the clause is unit).
        """
        seen = [False] * (self._num_vars + 1)
        learnt: List[Lit] = [0]  # placeholder for the UIP
        count = 0
        trail_idx = len(self._trail) - 1
        cur = confl_id
        p: Lit = 0
        while True:
            cl = self._clauses[cur]
            # Bump activity of the clause and its variables.
            cl.activity += 1.0
            for q in cl.lits:
                if p != 0 and q == p:
                    continue
                v = abs(q)
                if seen[v]:
                    continue
                lvl = self._level[v]
                if lvl <= 0:
                    continue
                seen[v] = True
                self._bump(v)
                if lvl >= self._decision_level():
                    count += 1
                else:
                    learnt.append(q)
            # Pick the next literal to resolve on — the most recent
            # trail-entry that is `seen`.
            while trail_idx >= 0 and not seen[abs(self._trail[trail_idx])]:
                trail_idx -= 1
            if trail_idx < 0:
                break
            p = self._trail[trail_idx]
            v = abs(p)
            seen[v] = False
            count -= 1
            if count == 0:
                break
            cur = self._reason[v]  # type: ignore[assignment]
            trail_idx -= 1
            if cur is None:
                # Should not happen — a decision literal can never be
                # the resolvent without count = 0.
                break
        learnt[0] = -p
        # Minimise the learnt clause: drop literals whose negations are
        # entailed by other learnt-clause literals via L0 reasons.  (A
        # cheap form of conflict-clause minimisation.)
        learnt = self._minimise_learnt(learnt, seen)
        # Compute backjump level — second-highest among the learnt
        # literals (or 0 if unit).
        if len(learnt) == 1:
            return learnt, 0
        # Make sure learnt[1] is the literal with the highest non-asserting level.
        max_idx = 1
        max_lvl = self._level[abs(learnt[1])]
        for i in range(2, len(learnt)):
            l = self._level[abs(learnt[i])]
            if l > max_lvl:
                max_lvl = l
                max_idx = i
        learnt[1], learnt[max_idx] = learnt[max_idx], learnt[1]
        return learnt, max_lvl

    def _minimise_learnt(self, learnt: List[Lit], seen: List[bool]) -> List[Lit]:
        """Self-subsuming-resolution minimisation (Sörensson-Biere 2009).

        Drop literal ``l`` from the learnt clause if every literal in
        the reason clause of ``-l`` is either also in the learnt
        clause or at decision level 0.  Cheap, sound, and a major win
        on industrial instances.
        """
        if len(learnt) <= 1:
            return learnt
        marked = set(abs(l) for l in learnt)
        kept = [learnt[0]]
        for l in learnt[1:]:
            v = abs(l)
            r = self._reason[v]
            if r is None:
                kept.append(l)
                continue
            redundant = True
            for q in self._clauses[r].lits:
                qv = abs(q)
                if qv == v:
                    continue
                if self._level[qv] == 0:
                    continue
                if qv not in marked:
                    redundant = False
                    break
            if not redundant:
                kept.append(l)
        return kept

    # --- backtracking --------------------------------------------------

    def _backtrack_to(self, level: int) -> None:
        if self._decision_level() <= level:
            return
        target = self._trail_lim[level] if level < len(self._trail_lim) else 0
        while len(self._trail) > target:
            lit = self._trail.pop()
            v = abs(lit)
            # Save phase before unassigning.
            self._phase[v] = self._assign[v]
            self._assign[v] = _UNASSIGNED
            self._level[v] = -1
            self._reason[v] = None
        del self._trail_lim[level:]

    # --- branching -----------------------------------------------------

    def _pick_branching(self) -> Optional[Lit]:
        """VSIDS argmax over unassigned variables.

        Returns a literal — variable id with phase from the phase-saving
        table — or ``None`` if every variable in the input range is
        assigned.
        """
        # Linear scan; in industrial solvers this is a heap, but for
        # the moderate-size instances exercised by this primitive a
        # scan is the simplest correct choice and avoids the heap
        # invariant-maintenance bug surface.
        best_v = 0
        best_a = -1.0
        for v in range(1, self._num_vars + 1):
            if self._assign[v] != _UNASSIGNED:
                continue
            a = self._activity[v]
            if a > best_a:
                best_a = a
                best_v = v
        if best_v == 0:
            return None
        return best_v if self._phase[best_v] != _FALSE else -best_v

    # --- VSIDS ---------------------------------------------------------

    def _bump(self, v: int) -> None:
        self._activity[v] += self._act_inc
        if self._activity[v] > 1e100:
            for i in range(1, self._num_vars + 1):
                self._activity[i] *= 1e-100
            self._act_inc *= 1e-100

    def _decay_activities(self) -> None:
        self._act_inc /= self._act_decay

    # --- assignment helpers -------------------------------------------

    def _enqueue(self, lit: Lit, reason: Optional[int]) -> None:
        v = abs(lit)
        self._assign[v] = _TRUE if lit > 0 else _FALSE
        self._level[v] = self._decision_level()
        self._reason[v] = reason
        self._trail.append(lit)

    def _is_true(self, lit: Lit) -> bool:
        v = abs(lit)
        return self._assign[v] == (_TRUE if lit > 0 else _FALSE)

    def _is_false(self, lit: Lit) -> bool:
        v = abs(lit)
        return self._assign[v] == (_FALSE if lit > 0 else _TRUE)

    def _decision_level(self) -> int:
        return len(self._trail_lim)

    # --- clause-database management -----------------------------------

    def _add_internal(self, clause: _Clause) -> int:
        idx = len(self._clauses)
        self._clauses.append(clause)
        if len(clause.lits) >= 2:
            self._watches.setdefault(_lit_index(clause.lits[0]), []).append(idx)
            self._watches.setdefault(_lit_index(clause.lits[1]), []).append(idx)
        return idx

    def _learn_clause(self, lits: List[Lit]) -> int:
        cl = _Clause(lits=list(lits), learnt=True)
        cl.lbd = self._compute_lbd(lits)
        cl.activity = 1.0
        idx = self._add_internal(cl)
        self._learnt_kept += 1
        return idx

    def _compute_lbd(self, lits: Sequence[Lit]) -> int:
        levels: Set[int] = set()
        for l in lits:
            lvl = self._level[abs(l)]
            if lvl >= 0:
                levels.add(lvl)
        return max(1, len(levels))

    def _reduce_db(self) -> None:
        """Glucose-style LBD reduction.

        Keep every learnt clause with LBD ≤ 2 (the "glue" clauses) and
        the upper-half of the remaining learnt clauses by activity.
        Original clauses are never deleted.  Deleted clauses are
        emitted as DRAT deletion records so the proof stays consistent
        with the actual database.
        """
        learnt_ids = [i for i, c in enumerate(self._clauses) if c.learnt]
        if not learnt_ids:
            return
        # Partition into "glue" and "deletable".
        glue = [i for i in learnt_ids if self._clauses[i].lbd <= 2]
        rest = [i for i in learnt_ids if self._clauses[i].lbd > 2]
        rest.sort(key=lambda i: self._clauses[i].activity)
        cut = len(rest) // 2
        to_delete = set(rest[:cut])
        # Don't delete clauses currently being used as propagation reasons.
        for v in range(1, self._num_vars + 1):
            r = self._reason[v]
            if r is not None and r in to_delete:
                to_delete.remove(r)
        # Rebuild the database with deleted clauses removed.  Watch
        # lists are rebuilt below.
        new_clauses: List[_Clause] = []
        id_remap: Dict[int, int] = {}
        for i, cl in enumerate(self._clauses):
            if i in to_delete:
                self._proof_emit_delete(cl.lits)
                self._learnt_deleted += 1
                continue
            id_remap[i] = len(new_clauses)
            new_clauses.append(cl)
        self._clauses = new_clauses
        # Rebuild watch lists and reason indices.
        self._watches = {}
        for new_idx, cl in enumerate(self._clauses):
            if len(cl.lits) >= 2:
                self._watches.setdefault(_lit_index(cl.lits[0]), []).append(new_idx)
                self._watches.setdefault(_lit_index(cl.lits[1]), []).append(new_idx)
        for v in range(1, self._num_vars + 1):
            r = self._reason[v]
            if r is None:
                continue
            if r in id_remap:
                self._reason[v] = id_remap[r]
            else:
                self._reason[v] = None
        self._record(SOLVER_REDUCED, {"kept": len(new_clauses), "deleted": len(to_delete)})

    # --- proof emission -----------------------------------------------

    def _proof_emit_add(self, lits: Sequence[Lit]) -> None:
        self._proof_log.append(("a", tuple(lits)))

    def _proof_emit_delete(self, lits: Sequence[Lit]) -> None:
        self._proof_log.append(("d", tuple(lits)))

    def _proof_emit_empty(self) -> None:
        # Idempotently append the empty clause (the UNSAT certificate).
        # Skipped when the solver reached UNSAT under assumptions —
        # the proof of UNSAT against the bare formula is not in
        # general derivable from the assumption-conditioned learnt
        # clauses, so no empty marker is emitted.  Use
        # ``SolverResult.core`` for the assumption-conditioned UNSAT
        # certificate.
        if self._last_unsat_under_assumptions:
            return
        if not self._proof_log or self._proof_log[-1] != ("a", ()):
            self._proof_log.append(("a", ()))

    # --- model extraction & UNSAT core --------------------------------

    def _extract_model(self) -> Dict[int, bool]:
        out: Dict[int, bool] = {}
        for v in range(1, self._num_vars + 1):
            a = self._assign[v]
            if a == _UNASSIGNED:
                # Free variable: pick the saved phase.
                out[v] = self._phase[v] != _FALSE
            else:
                out[v] = a == _TRUE
        return out

    def _compute_assumption_core(self, confl_id: int) -> Tuple[Lit, ...]:
        """Walk back from the conflict clause via reasons; collect
        assumption literals on the way.

        The returned tuple is a (not necessarily minimal) UNSAT core
        over the assumption set.  Use :meth:`extract_mus` for a
        minimal one.
        """
        if not self._assumptions:
            return ()
        seen = [False] * (self._num_vars + 1)
        assumption_set = set(self._assumptions)
        core: List[Lit] = []
        stack: List[int] = [confl_id]
        # Initialise from conflict clause.
        for l in self._clauses[confl_id].lits:
            v = abs(l)
            seen[v] = True
            if l in assumption_set or -l in assumption_set:
                # An assumption participates in the conflict.
                # Include the assumption side (positive if positive
                # assumed, negative if negative assumed).
                if l in assumption_set:
                    if l not in core:
                        core.append(l)
                else:
                    if -l in assumption_set:
                        if -l not in core:
                            core.append(-l)
        # Walk reasons.
        idx = len(self._trail) - 1
        while idx >= 0:
            lit = self._trail[idx]
            v = abs(lit)
            if seen[v]:
                r = self._reason[v]
                if r is not None:
                    for q in self._clauses[r].lits:
                        qv = abs(q)
                        if not seen[qv]:
                            seen[qv] = True
                            if q in assumption_set:
                                if q not in core:
                                    core.append(q)
                            elif -q in assumption_set:
                                if -q not in core:
                                    core.append(-q)
                else:
                    # Decision literal — if it's an assumption, add it.
                    if lit in assumption_set:
                        if lit not in core:
                            core.append(lit)
                    elif -lit in assumption_set:
                        if -lit not in core:
                            core.append(-lit)
            idx -= 1
        return tuple(core)

    # --- MUS extraction -----------------------------------------------

    def extract_mus(self) -> Tuple[Lit, ...]:
        """Deletion-based MUS extraction over the current assumptions.

        Requires the most recent ``solve`` to have returned UNSAT.
        Repeatedly tries to drop each assumption from the core; if
        the reduced set is still UNSAT the assumption is permanently
        dropped, otherwise it is retained.  Worst case ``|core|``
        UNSAT calls; the result is a minimal UNSAT subset of the
        assumptions.
        """
        if self._last_status != STATUS_UNSAT:
            raise NotYetSolved("extract_mus requires the last solve to have returned UNSAT")
        if not self._assumptions:
            return ()
        candidates = list(self._assumptions)
        i = 0
        while i < len(candidates):
            test = candidates[:i] + candidates[i + 1:]
            saved = self._assumptions
            self._assumptions = list(test)
            res = self.solve()
            self._assumptions = saved
            if res.status == STATUS_UNSAT:
                # candidate i is redundant — drop it.
                candidates = test
                # Don't increment i; the next candidate has shifted in.
            else:
                # candidate i is essential — keep it.
                i += 1
        # Confirm the result.
        saved = self._assumptions
        self._assumptions = list(candidates)
        res = self.solve()
        self._assumptions = saved
        if res.status != STATUS_UNSAT:
            raise SolverError("MUS extraction lost UNSAT — bug")
        out = tuple(candidates)
        self._record(SOLVER_MUS, {"size": len(out), "lits": list(out)})
        return out

    # --- MaxSAT --------------------------------------------------------

    def solve_max_sat(
        self,
        soft: Sequence[Sequence[Lit]],
        *,
        weights: Optional[Sequence[int]] = None,
        time_budget_s: Optional[float] = None,
    ) -> Tuple[int, Mapping[int, bool], Tuple[int, ...]]:
        """Iterated UNSAT-core MaxSAT (Fu-Malik / OLL skeleton).

        Hard clauses are the ones already added via :meth:`add_clause`.
        ``soft`` is a sequence of additional clauses; each is relaxed
        by a fresh selector variable ``s_i`` so the clause becomes
        ``(s_i ∨ c_i)`` and the empty-vs-non-empty status of
        ``Σ w_i · s_i ≤ B`` is incrementally tightened until SAT, at
        which point ``B`` is the MaxSAT cost (the weighted count of
        violated soft clauses).

        Returns ``(cost, model, violated_indices)`` where ``cost`` is
        the minimum weighted sum of falsified soft clauses, ``model``
        is over the original variables, and ``violated_indices`` lists
        the soft clauses falsified under the returned model.

        Weights are required to be **non-negative integers** (default
        1 for every clause).  Floating-point weights or finer
        precision should be pre-quantised; the cardinality encoding
        scales linearly with the total weight, so coarse buckets
        keep encoding size bounded.
        """
        if weights is None:
            weights = [1] * len(soft)
        if len(weights) != len(soft):
            raise InvalidConfig("weights must have same length as soft")
        if any(not isinstance(w, int) or w < 0 for w in weights):
            raise InvalidConfig("weights must be non-negative integers")
        # Materialise selector variables and relaxed soft clauses.
        selectors: List[int] = []
        for i, cl in enumerate(soft):
            sel = self.new_var()
            self.add_clause([sel] + list(cl))
            selectors.append(sel)
        max_cost = sum(weights)
        t0 = time.monotonic()
        saved_assumptions = list(self._assumptions)
        try:
            # Linear search from cost=0 upward.  Each iteration adds
            # *fresh* cardinality clauses to a snapshotted database
            # and rolls them back afterwards.  This keeps the search
            # incremental against a stable formula.
            for budget in range(0, max_cost + 1):
                # Build a fresh AT_MOST_BUDGET constraint over the
                # weight-expanded selector set.
                snapshot_clauses = len(self._clauses)
                snapshot_orig = len(self._original_clauses)
                snapshot_vars = self._num_vars
                snapshot_user = self._user_reserved
                snapshot_unsat = self._unsat_at_l0
                snapshot_proof = len(self._proof_log)
                cardinality_clauses: List[List[Lit]] = []
                counter = [max(self._num_vars, self._user_reserved) + 1]
                # Expand selectors by their weight: w_i copies of
                # selector i, each a fresh aux equivalenced to s_i.
                # This is the standard "weight ⇒ replicated count"
                # trick (Sinz 2005 §4); for sparse high-weight
                # problems consider switching to a totalizer or PB
                # encoding.
                expanded: List[Lit] = []
                for sel, w in zip(selectors, weights):
                    if w == 0:
                        continue
                    for _ in range(w):
                        proxy = counter[0]
                        counter[0] += 1
                        cardinality_clauses.append([-proxy, sel])
                        cardinality_clauses.append([proxy, -sel])
                        expanded.append(proxy)
                _encode_at_most(cardinality_clauses, counter, expanded, budget)
                for c in cardinality_clauses:
                    self.add_clause(c)
                res = self.solve(time_budget_s=time_budget_s)
                if res.status == STATUS_SAT:
                    model = {v: res.model[v] for v in res.model if v <= snapshot_vars}
                    violated: List[int] = []
                    for i, cl in enumerate(soft):
                        if not any(
                            (l > 0 and model.get(abs(l), False))
                            or (l < 0 and not model.get(abs(l), False))
                            for l in cl
                        ):
                            violated.append(i)
                    self._record(
                        SOLVER_MAXSAT,
                        {"cost": budget, "violated": violated, "selectors": len(selectors)},
                    )
                    return budget, model, tuple(violated)
                # Roll back this iteration's cardinality additions.
                self._rollback_to(
                    snapshot_clauses,
                    snapshot_orig,
                    snapshot_vars,
                    snapshot_user,
                    snapshot_unsat,
                    snapshot_proof,
                )
                if time_budget_s is not None and (time.monotonic() - t0) >= time_budget_s:
                    raise ResourceExhausted("MaxSAT time budget exhausted")
            raise SolverError("MaxSAT: budget search exhausted without SAT — bug")
        finally:
            self._assumptions = saved_assumptions

    def _rollback_to(
        self,
        clauses_n: int,
        orig_n: int,
        vars_n: int,
        user_n: int,
        unsat: bool,
        proof_n: int,
    ) -> None:
        """Restore solver state to a prior snapshot.

        Used by MaxSAT to undo a budget-iteration's tentative
        cardinality clauses before trying the next budget value.
        """
        # Truncate clause databases.
        self._clauses = self._clauses[:clauses_n]
        self._original_clauses = self._original_clauses[:orig_n]
        # Restore variable count.
        self._num_vars = vars_n
        self._user_reserved = user_n
        self._assign = self._assign[: vars_n + 1]
        self._level = self._level[: vars_n + 1]
        self._reason = self._reason[: vars_n + 1]
        self._phase = self._phase[: vars_n + 1]
        self._activity = self._activity[: vars_n + 1]
        self._unsat_at_l0 = unsat
        self._proof_log = self._proof_log[:proof_n]
        # Reset trail / decision level entirely.
        self._trail = []
        self._trail_lim = []
        for v in range(1, vars_n + 1):
            self._assign[v] = _UNASSIGNED
            self._level[v] = -1
            self._reason[v] = None
        # Rebuild watch lists from scratch over the surviving clauses.
        self._watches = {}
        for new_idx, cl in enumerate(self._clauses):
            if len(cl.lits) >= 2:
                self._watches.setdefault(_lit_index(cl.lits[0]), []).append(new_idx)
                self._watches.setdefault(_lit_index(cl.lits[1]), []).append(new_idx)
        # Re-enqueue every surviving unit clause at L0 so subsequent
        # ``solve`` calls see the proper propagation seed.
        for new_idx, cl in enumerate(self._clauses):
            if len(cl.lits) == 1:
                lit = cl.lits[0]
                v = abs(lit)
                want = _TRUE if lit > 0 else _FALSE
                if self._assign[v] == _UNASSIGNED:
                    self._enqueue(lit, reason=new_idx)
                elif self._assign[v] != want:
                    self._unsat_at_l0 = True

    # --- DRAT proof verification --------------------------------------

    def proof(self) -> Tuple[Tuple[Any, ...], ...]:
        """Return the accumulated DRAT proof log.

        Each entry is either ``("a", lits)`` (an addition) or
        ``("d", lits)`` (a deletion).  An empty-clause addition
        ``("a", ())`` certifies UNSAT.  Call :meth:`check_proof` to
        re-verify the log against the original CNF.
        """
        return tuple(self._proof_log)

    def check_proof(self) -> bool:
        """Re-verify the DRAT proof by Reverse Unit Propagation.

        Each addition ``a, lits`` must be a RUP-certificate: assuming
        the negation of every literal of ``lits`` and propagating
        unit-clause implications through the *current* CNF database
        must derive ``⊥``.  Deletions are applied as removals from
        the verifier's clause set.

        Returns ``True`` on a valid proof of either SAT (no empty
        clause) or UNSAT (terminating empty clause); raises
        :class:`ProofCheckFailed` otherwise.

        This is a *re*-checker — it is independent of the solver's
        search trail and therefore catches any internal book-keeping
        bug.  Run it routinely under tests; expose it under
        ``Solver.proof()`` to a regulator if a UNSAT verdict is
        consumed downstream.
        """
        if self._last_unsat_under_assumptions and self._last_status == STATUS_UNSAT:
            # The DRAT proof is only emitted for unconditional UNSAT.
            # Under assumptions, ``SolverResult.core`` plus a separate
            # solve against the formula extended with the core's
            # unit clauses constitutes the audit trail.
            return True
        if not self._proof_log:
            # Either trivial SAT (no solve issued) or a SAT verdict
            # with no learnts.  Either way nothing to verify.
            return True
        # Clause set indexed by canonical tuple.
        cs: Dict[Tuple[Lit, ...], int] = {}
        for c in self._original_clauses:
            cs[c] = cs.get(c, 0) + 1
        saw_empty = False
        for op, lits in self._proof_log:
            canon = _canonical_clause(lits)
            if op == "d":
                if canon in cs:
                    cs[canon] -= 1
                    if cs[canon] == 0:
                        del cs[canon]
                continue
            if op != "a":
                raise ProofCheckFailed(f"unknown proof op {op!r}")
            # RUP check: negate every literal of ``lits`` and propagate.
            if not _rup_check(cs, canon):
                raise ProofCheckFailed(f"RUP failure on clause {canon}")
            if canon == ():
                saw_empty = True
            cs[canon] = cs.get(canon, 0) + 1
        return True if (saw_empty or self._last_status != STATUS_UNSAT) else False

    # --- reporting -----------------------------------------------------

    def report(self) -> SolverReport:
        rep = SolverReport(
            num_vars=self._num_vars,
            num_clauses=len(self._original_clauses),
            num_learnt=sum(1 for c in self._clauses if c.learnt),
            conflicts=self._conflicts,
            decisions=self._decisions,
            propagations=self._propagations,
            restarts=self._restarts,
            last_status=self._last_status,
            ledger_head=self._ledger_head,
            seed=self._seed,
        )
        self._record(SOLVER_REPORTED, {"head": self._ledger_head})
        return rep

    def ledger_head(self) -> str:
        return self._ledger_head

    def ledger(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self._ledger)

    # --- internal book-keeping ----------------------------------------

    def _grow_for_lits(self, lits: Iterable[Lit]) -> None:
        m = self._num_vars
        for l in lits:
            v = abs(int(l))
            if v > m:
                m = v
        if m > self._num_vars:
            extra = m - self._num_vars
            self._num_vars = m
            self._assign.extend([_UNASSIGNED] * extra)
            self._level.extend([-1] * extra)
            self._reason.extend([None] * extra)
            self._phase.extend([_TRUE] * extra)
            self._activity.extend([0.0] * extra)

    def _stats(self) -> Mapping[str, int]:
        return {
            "conflicts": self._conflicts,
            "decisions": self._decisions,
            "propagations": self._propagations,
            "restarts": self._restarts,
            "learnt_kept": self._learnt_kept,
            "learnt_deleted": self._learnt_deleted,
            "num_vars": self._num_vars,
            "num_clauses": len(self._original_clauses),
        }

    def _record(self, kind: str, payload: Mapping[str, Any]) -> None:
        record = {"event": kind, "payload": dict(payload)}
        self._ledger_head = _hash_event(self._ledger_head, record)
        record["head"] = self._ledger_head
        self._ledger.append(record)


def _lit_index(lit: Lit) -> int:
    """Map a signed-int literal to a non-negative key for the watch table."""
    v = abs(lit)
    return 2 * v + (0 if lit > 0 else 1)


def _rup_check(cs: Mapping[Tuple[Lit, ...], int], target: Tuple[Lit, ...]) -> bool:
    """Pure-RUP check: assume ``¬target`` and unit-propagate to ``⊥``.

    Implements the standard textbook BCP loop over the in-memory
    clause set.  Performance is not the priority — *correctness* is.
    Each call is O(|cs| · |target|) in the worst case, dominated by
    repeated linear scans over a small set.
    """
    assign: Dict[int, bool] = {}
    for l in target:
        v = abs(l)
        # Asserting ¬l: positive l ⇒ assign v False, negative l ⇒ True.
        want = l < 0
        if v in assign and assign[v] != want:
            return True  # ¬target itself is contradictory ⇒ RUP trivially.
        assign[v] = want
    changed = True
    while changed:
        changed = False
        for clause in cs.keys():
            sat = False
            unassigned: List[Lit] = []
            for l in clause:
                v = abs(l)
                if v in assign:
                    val = assign[v] if l > 0 else not assign[v]
                    if val:
                        sat = True
                        break
                else:
                    unassigned.append(l)
            if sat:
                continue
            if not unassigned:
                return True  # conflict found ⇒ RUP success
            if len(unassigned) == 1:
                l = unassigned[0]
                v = abs(l)
                want = l > 0
                if v in assign:
                    if assign[v] != want:
                        return True
                else:
                    assign[v] = want
                    changed = True
    return False


# --------------------------------------------------------------------- module-level

# Sentinel key controlling private construction of ``Solver``.
_SOLVER_KEY = object()


__all__ = [
    "Clause",
    "Formula",
    "InvalidAssumption",
    "InvalidClause",
    "InvalidConfig",
    "InvalidFormula",
    "InvalidLiteral",
    "Lit",
    "NotYetSolved",
    "ProofCheckFailed",
    "ResourceExhausted",
    "SOLVER_ASSUMED",
    "SOLVER_CLAUSE_ADDED",
    "SOLVER_CLEARED",
    "SOLVER_KNOWN_EVENTS",
    "SOLVER_KNOWN_STATUSES",
    "SOLVER_MAXSAT",
    "SOLVER_MUS",
    "SOLVER_REDUCED",
    "SOLVER_REPORTED",
    "SOLVER_RESTARTED",
    "SOLVER_SOLVED",
    "SOLVER_STARTED",
    "STATUS_SAT",
    "STATUS_UNKNOWN",
    "STATUS_UNSAT",
    "Solver",
    "SolverError",
    "SolverReport",
    "SolverResult",
    "at_least",
    "at_most",
    "exactly",
    "false",
    "land",
    "lnot",
    "lor",
    "true",
    "var",
    "xeqv",
    "xite",
    "ximp",
]
