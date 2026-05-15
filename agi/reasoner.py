r"""Reasoner — symbolic logical reasoning as a runtime primitive.

The coordination engine running on top of this runtime spends most of its
cycles in the *probabilistic* corner: Bandit / BayesOpt / Sampler /
Forecaster all return distributions, and Strategist / Deliberator /
Arbiter close the loop with anytime-valid bounds.  But a non-trivial
fraction of every plan reduces to *deterministic* questions: is this
configuration feasible, does this invariant hold, does this set of
constraints admit any model, can we derive ``allowed(x)`` from these
rules and facts.  Those questions are not Bayesian — they are crisp
boolean / first-order / Horn-clause statements, and the right answer is
a proof or a counterexample, not a posterior.

`Reasoner` is the runtime's **symbolic-reasoning** primitive.  It is the
deterministic, proof-carrying dual of `Refuter` (which falsifies
probabilistic claims with PAC bounds) and the discrete-logic complement
of `Synthesizer` (which fills in *programs* from examples).  It composes
naturally with every other primitive in the runtime: as a feasibility
gate on `Negotiator` / `MechanismDesigner` allocations, as a hard-
constraint solver inside `PortfolioOptimizer`, as the verification
oracle inside `Synthesizer` (CEGIS), as a proof generator that
`AttestationLedger` can hash and a coordinator can audit.

The pitch reduced to a runtime call:

  * encode a problem as a CNF, a Horn-rule program, or an ASP program;
  * call ``solve()`` to get a satisfying model (or unsat with a
    resolution-proof witness), ``forward_chain()`` to compute the
    minimal Herbrand model of a Horn program, ``backward_chain(query)``
    to obtain a (Prolog-style) SLD-resolution proof tree, or
    ``stable_models()`` to enumerate the answer-set semantics for
    programs with negation-as-failure;
  * call ``entails(formula)`` to ask whether the current knowledge base
    *forces* a propositional formula;
  * call ``report()`` for a `ReasonerReport` carrying the model count
    (or a finite-sample bound on it), conflict-driven learning
    statistics, the SCC stratification of any negation-cycle, anytime-
    valid PAC certificates on stochastic search budgets, and a tamper-
    evident SHA-256 fingerprint chaining every clause, rule, and
    decision the solver made.

Mathematical roots and algorithms shipped
-----------------------------------------

The Reasoner ships ten classical algorithms behind one API:

**Propositional satisfiability (SAT).**

  * **DPLL** (Davis-Putnam 1960; Davis-Logemann-Loveland 1962).
    Recursive backtracking with unit propagation + pure-literal
    elimination + variable splitting; the algorithm every introductory
    course teaches.  Exponential worst case but polynomial on Horn,
    2-SAT, and any class with bounded width.  Implementation is
    iterative (trail-based) for stack safety on long unit propagation
    chains.

  * **CDCL** (Marques-Silva & Sakallah 1996, *GRASP*; Moskewicz et al.
    2001, *Chaff/zChaff*; Eén & Sörensson 2003, *MiniSat*).
    Conflict-driven clause learning with the **first-UIP** scheme
    (Zhang-Madigan-Moskewicz-Malik 2001), **watched literals**
    (Zhang & Stickel 1996) for O(1)-amortised unit propagation, **VSIDS**
    (Variable-State Independent Decaying Sum, Moskewicz et al. 2001) for
    branching, and a **Luby restart** schedule (Luby-Sinclair-Zuckerman
    1993).  This is the algorithmic core that powers every modern
    industrial SAT solver from MiniSat through Glucose, CaDiCaL, Kissat.

  * **Walk-SAT** (Selman-Kautz-Cohen 1994).  Local-search Monte Carlo
    SAT — Schöning's (1999, *FOCS*) randomised 3-SAT algorithm — used
    as a fallback heuristic on satisfiable instances with very long
    backtrack chains.  Always either returns a model or times out; we
    pair every run with a Clopper-Pearson (1934) upper bound on the
    failure probability so the coordinator can reason about its
    completion guarantee.

  * **Resolution refutation** (Robinson 1965).  When DPLL/CDCL closes
    with UNSAT we reconstruct an **unsat core** by traversing the
    learnt-clause implication graph and assemble a resolution proof
    (sequence of (clause_a, clause_b, resolvent) triples ending in the
    empty clause).  The proof is hashed into the report so an external
    verifier can independently replay it.

**Horn-clause reasoning (Datalog).**

  * **Semi-naïve forward chaining** (Bancilhon-Maier-Sagiv-Ullman 1986,
    Ullman 1989, *Principles of Database and Knowledge-Base Systems*).
    The fixpoint of ``T_P`` is reached in O(|P| + |facts| · max_arity)
    time using the *delta-incremental* recurrence
    ``Δ^{n+1}  =  T_P(Δ^n)  \ {known}``, the standard Datalog evaluation
    that powers Soufflé and DDlog.  Ships with both *naïve* (full
    re-evaluation) and *semi-naïve* (delta-only) implementations so the
    coordinator can pick the smaller-overhead option for very small
    rulesets.

  * **SLD resolution backward chaining** (Kowalski 1974).  Goal-directed
    Prolog-style proof search with **subsumption tabling** (Tamaki-Sato
    1986, *SLG resolution*, the foundation of XSB) to avoid divergence
    on left-recursive rules.  Returns a proof tree the coordinator
    can replay or hash.

**Answer Set Programming (ASP).**

  * **Stable model semantics** (Gelfond-Lifschitz 1988, *The stable
    model semantics for logic programming*; refined in Gelfond-Lifschitz
    1991, *Classical negation in logic programs*).  We implement
    *guess-and-check*: the solver enumerates candidate truth assignments
    for atoms appearing under negation-as-failure, computes the reduct
    ``P^M``, evaluates the Horn reduct to its unique least model, and
    accepts only when the candidate matches.  An optional **stratified
    negation** fast path (Apt-Blair-Walker 1988) uses Tarjan's (1972)
    SCC on the dependency graph to skip the guess phase when no SCC
    contains a negated edge.  This is the semantic foundation of clingo
    / DLV / Potassco.

Anytime-valid certificates
--------------------------

  * **Clopper-Pearson 1934 upper bound** on the failure rate of any
    randomised solver run.  For 0 failures across n attempts the
    closed-form upper bound is ``1 - α^{1/n}``; otherwise we invert
    the regularised incomplete beta numerically.  This is the
    finite-sample bound the coordinator quotes when it tells a tenant
    "our Walk-SAT either returns a model or fails with probability
    at most ε".

  * **Hoeffding (1963) and empirical-Bernstein (Maurer-Pontil 2009)**
    anytime half-widths on the *model count* when the report invokes
    importance-sampling model counting (cf. Gomes-Hoffmann-Sabharwal
    2007, *ApproxMC*).  Combined with the model-count itself, this
    gives the coordinator a calibrated answer to "how many feasible
    plans are there" without enumerating.

Tamper-evident replay
---------------------

Every clause, fact, rule, and decision the Reasoner takes hashes into a
chain rooted at a genesis SHA-256.  ``report().fingerprint`` is the head
of the chain and depends deterministically on every input.  A coordinator
that re-runs the same sequence with the same seed reproduces the chain
exactly — `AttestationLedger`-friendly.

Composition with the rest of the runtime
----------------------------------------

  * **Refuter** — Refuter falsifies probabilistic claims with sequential
    e-values.  Reasoner *certifies* deterministic claims with proofs.
    The two are duals: when Refuter cannot reject after a CS-large
    budget it hands the predicate to Reasoner; when Reasoner cannot
    prove or disprove (the instance is too big) it hands the residual
    to Refuter for stochastic falsification.

  * **Synthesizer** — Synthesizer uses Reasoner as its CEGIS verifier
    (Solar-Lezama 2008): synthesise a candidate program, encode the
    correctness predicate as a CNF / Horn rule, ask Reasoner to find
    a counter-example, refine.  Reasoner's unsat-proof on the negated
    correctness predicate certifies the synthesised program globally.

  * **Negotiator / MechanismDesigner / PortfolioOptimizer** — every
    allocation primitive has *hard* feasibility constraints (capacity,
    integrality, conflicting-resources).  Reasoner is the feasibility
    oracle they delegate to.

  * **Equilibrator / Diplomat** — when a game-theoretic equilibrium is
    parameterised by Boolean side-conditions (legality of joint
    actions, sequential-elimination of dominated strategies), Reasoner
    solves the side-conditions before the LP / CFR loop runs.

  * **CausalDiscoverer** — orient-after-skeleton can be cast as a Horn
    program over conditional-independence facts; Reasoner runs the
    forward-chain to expose all valid v-structures.

  * **Auditor** — when many simultaneous reasoning tasks complete with
    PAC bounds, Auditor's BH/FDR machinery jointly controls the false
    proof rate across the batch.

  * **Cartographer** — pre-requisite DAGs in the curriculum are exactly
    Horn programs (``ready(task)  ←  passed(prereq_1) ∧ … ∧ passed(p_k)``);
    Reasoner's forward chain gives ``ready/1`` for the next-task pick.

  * **AttestationLedger** — Reasoner's per-call SHA-256 fingerprint
    drops directly into the ledger; the proof tree from
    ``backward_chain`` and the resolution proof from an UNSAT close are
    replayable receipts.

  * **PrivacyAccountant** — for differentially-private reasoning over
    facts that came from sensitive data, the privacy odometer is
    advanced on each ``add_fact`` call.

  * **Strategist** — when the Strategist must decide between competing
    policies and the world is partially specified by *rules*, Reasoner
    answers the entailment query *"does policy A satisfy invariant I in
    every model of the rules?"* before the risk-adjusted score is
    quoted.

Public API
----------

::

    >>> from agi.reasoner import Reasoner, CDCL, FORWARD_CHAIN
    >>> R = Reasoner(algorithm=CDCL, seed=0)
    >>> R.add_clause(["a", "b"])           # a ∨ b
    >>> R.add_clause(["~a", "c"])          # ¬a ∨ c
    >>> R.add_clause(["~b", "c"])          # ¬b ∨ c
    >>> sol = R.solve()
    >>> sol.satisfiable, sol.model
    (True, {'a': False, 'b': True, 'c': True})   # example, depends on seed
    >>> R.entails("c")
    True
    >>> R.add_clause(["~c"])
    >>> R.solve().satisfiable
    False
    >>> R.last_resolution_proof()[-1]
    Resolution(clause_a=…, clause_b=…, resolvent=())

    >>> H = Reasoner(algorithm=FORWARD_CHAIN)
    >>> H.add_rule("ready(t) :- passed(p), prereq(p, t).")
    >>> H.add_fact("passed(arithmetic)")
    >>> H.add_fact("prereq(arithmetic, algebra)")
    >>> H.forward_chain()
    {'passed(arithmetic)', 'prereq(arithmetic, algebra)', 'ready(algebra)'}

References
----------

* Davis, Logemann & Loveland (1962). *A machine program for theorem-
  proving.* CACM.
* Robinson (1965). *A machine-oriented logic based on the resolution
  principle.* JACM.
* Marques-Silva & Sakallah (1996). *GRASP: A search algorithm for
  propositional satisfiability.* IEEE TC.
* Moskewicz, Madigan, Zhao, Zhang & Malik (2001). *Chaff: Engineering an
  efficient SAT solver.* DAC.
* Eén & Sörensson (2003). *An extensible SAT-solver* (MiniSat).
* Zhang & Stickel (1996). *An efficient algorithm for unit propagation*
  (watched literals).
* Selman, Kautz & Cohen (1994). *Noise strategies for improving local
  search* (Walk-SAT).
* Schöning (1999). *A probabilistic algorithm for k-SAT and constraint
  satisfaction problems.* FOCS.
* Luby, Sinclair & Zuckerman (1993). *Optimal speedup of Las Vegas
  algorithms.* IPL.
* Kowalski (1974). *Predicate logic as a programming language.* IFIP.
* Tamaki & Sato (1986). *OLD resolution with tabulation.* ICLP.
* Ullman (1989). *Principles of Database and Knowledge-Base Systems.*
* Bancilhon, Maier, Sagiv & Ullman (1986). *Magic sets and other strange
  ways to implement logic programs.* PODS.
* Gelfond & Lifschitz (1988). *The stable model semantics for logic
  programming.* ICLP.
* Apt, Blair & Walker (1988). *Towards a theory of declarative
  knowledge* (stratified negation).
* Clopper & Pearson (1934). *The use of confidence or fiducial limits.*
  Biometrika.
* Maurer & Pontil (2009). *Empirical Bernstein bounds.*
* Hoeffding (1963). *Probability inequalities for sums of bounded
  random variables.* JASA.
* Tarjan (1972). *Depth-first search and linear graph algorithms.*
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


# =====================================================================
# Constants — algorithm names, modes, events
# =====================================================================

# Propositional SAT algorithms.
DPLL = "dpll"
CDCL = "cdcl"
WALKSAT = "walksat"

# Horn / Datalog.
FORWARD_CHAIN = "forward_chain"
SEMI_NAIVE = "semi_naive"
BACKWARD_CHAIN = "backward_chain"

# Answer Set Programming.
STABLE_MODELS = "stable_models"

KNOWN_ALGORITHMS = frozenset({
    DPLL, CDCL, WALKSAT,
    FORWARD_CHAIN, SEMI_NAIVE, BACKWARD_CHAIN,
    STABLE_MODELS,
})

_SAT_ALGOS = frozenset({DPLL, CDCL, WALKSAT})
_HORN_ALGOS = frozenset({FORWARD_CHAIN, SEMI_NAIVE, BACKWARD_CHAIN})

# Solver verdicts.
SAT = "sat"
UNSAT = "unsat"
UNKNOWN = "unknown"          # e.g. budget-exhausted Walk-SAT

KNOWN_VERDICTS = frozenset({SAT, UNSAT, UNKNOWN})

# Events.
REASONER_STARTED = "reasoner.started"
REASONER_CLAUSE_ADDED = "reasoner.clause_added"
REASONER_RULE_ADDED = "reasoner.rule_added"
REASONER_FACT_ADDED = "reasoner.fact_added"
REASONER_SOLVED = "reasoner.solved"
REASONER_DERIVED = "reasoner.derived"
REASONER_PROOF = "reasoner.proof"
REASONER_REPORT = "reasoner.report"
REASONER_CLEARED = "reasoner.cleared"
REASONER_RESTART = "reasoner.restart"

KNOWN_EVENTS = frozenset({
    REASONER_STARTED, REASONER_CLAUSE_ADDED, REASONER_RULE_ADDED,
    REASONER_FACT_ADDED, REASONER_SOLVED, REASONER_DERIVED,
    REASONER_PROOF, REASONER_REPORT, REASONER_CLEARED, REASONER_RESTART,
})

# Numerical defaults.
_EPS = 1e-12
_DEFAULT_MAX_CONFLICTS = 1_000_000
_DEFAULT_WALKSAT_FLIPS = 100_000
_DEFAULT_WALKSAT_RESTARTS = 20
_DEFAULT_WALKSAT_NOISE = 0.5      # Selman-Kautz-Cohen default
_VSIDS_DECAY = 0.95
_VSIDS_BUMP = 1.0
_VSIDS_RESCALE = 1e100

# Genesis hash for the fingerprint chain.
_GENESIS = hashlib.sha256(b"reasoner.v1.genesis").hexdigest()


# =====================================================================
# Exceptions
# =====================================================================


class ReasonerError(ValueError):
    """Base class for Reasoner-domain errors."""


class UnknownAlgorithm(ReasonerError):
    """Algorithm name is not in KNOWN_ALGORITHMS."""


class InvalidClause(ReasonerError):
    """Clause or formula is malformed."""


class InvalidRule(ReasonerError):
    """Horn / ASP rule is malformed."""


class UnknownAtom(ReasonerError):
    """Caller referenced an atom not in the knowledge base."""


class BudgetExhausted(ReasonerError):
    """Solver hit its budget (conflicts / flips / restarts) without a decision."""


# =====================================================================
# Numerical helpers — anytime-valid bounds
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def hoeffding_half_width(n: int, delta: float, b: float = 1.0) -> float:
    """Hoeffding (1963) half-width for a mean of bounded RVs in [0, b]."""
    if n <= 0:
        return float("inf")
    return b * math.sqrt(math.log(2.0 / max(delta, _EPS)) / (2.0 * n))


def empirical_bernstein_half_width(
    n: int, var: float, delta: float, b: float = 1.0,
) -> float:
    """Maurer-Pontil (2009) empirical-Bernstein for a mean in [0, b]."""
    if n <= 1:
        return float("inf")
    log_term = math.log(3.0 / max(delta, _EPS))
    return math.sqrt(2.0 * max(var, 0.0) * log_term / n) + 3.0 * b * log_term / (n - 1)


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """Clopper-Pearson (1934) upper bound on a binomial probability.

    Returns the smallest p such that P(X ≤ k | n, p) ≤ α/2 (upper tail).
    Closed-form for k = 0 returns ``1 − α^{1/n}``; we invert the
    regularised incomplete beta via bisection otherwise.
    """
    if n <= 0:
        return 1.0
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    a = max(alpha, _EPS)
    if k == 0:
        return 1.0 - a ** (1.0 / n)
    # Bisection on Beta CDF target.
    lo, hi = float(k) / float(n), 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        # P(X ≤ k | n, mid) = I_{1-mid}(n-k, k+1) numerically; we use
        # the regularised-incomplete-beta via continued fraction.
        if _regularised_incomplete_beta(1.0 - mid, n - k, k + 1) <= a:
            hi = mid
        else:
            lo = mid
    return hi


def _log_gamma(x: float) -> float:
    # math.lgamma for Python's stdlib log-gamma.
    return math.lgamma(x)


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """Continued-fraction implementation of I_x(a, b) — accuracy ~1e-9."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(x, a, b) / a
    return 1.0 - bt * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


# =====================================================================
# Tarjan's SCC for stratification and Horn-rule dependency analysis
# =====================================================================


def strongly_connected_components(
    n: int, edges: Iterable[tuple[int, int]],
) -> list[list[int]]:
    """Tarjan (1972) iterative SCC.

    Returns SCCs sorted by size descending.  Vertices in `[0, n)`.
    """
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            adj[u].append(v)

    index = [0]
    indices = [-1] * n
    lowlink = [0] * n
    onstack = [False] * n
    stack: list[int] = []
    sccs: list[list[int]] = []

    def _strongconnect_iter(start: int) -> None:
        # Iterative DFS with two-phase frame.
        work: list[tuple[int, int]] = [(start, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                indices[v] = index[0]
                lowlink[v] = index[0]
                index[0] += 1
                stack.append(v)
                onstack[v] = True
            if pi < len(adj[v]):
                w = adj[v][pi]
                work[-1] = (v, pi + 1)
                if indices[w] == -1:
                    work.append((w, 0))
                elif onstack[w]:
                    if indices[w] < lowlink[v]:
                        lowlink[v] = indices[w]
            else:
                if lowlink[v] == indices[v]:
                    comp: list[int] = []
                    while True:
                        w = stack.pop()
                        onstack[w] = False
                        comp.append(w)
                        if w == v:
                            break
                    sccs.append(sorted(comp))
                work.pop()
                if work:
                    parent = work[-1][0]
                    if lowlink[v] < lowlink[parent]:
                        lowlink[parent] = lowlink[v]

    for v in range(n):
        if indices[v] == -1:
            _strongconnect_iter(v)

    sccs.sort(key=lambda c: -len(c))
    return sccs


# =====================================================================
# Propositional clauses — literals as signed ints, atoms as strings
# =====================================================================


def _parse_literal(token: str) -> tuple[str, bool]:
    """Parse "a" / "~a" / "-a" / "not a" → (atom, negated)."""
    t = token.strip()
    if not t:
        raise InvalidClause("empty literal")
    if t.startswith("not "):
        return t[4:].strip(), True
    if t.startswith("~") or t.startswith("-") or t.startswith("!"):
        rest = t[1:].strip()
        if not rest:
            raise InvalidClause(f"bare negation: {token!r}")
        return rest, True
    return t, False


def _parse_clause(spec: Any) -> list[tuple[str, bool]]:
    """Parse a clause specification into a list of (atom, negated) pairs.

    Accepts:
      * a sequence of tokens (each parsed by ``_parse_literal``);
      * a string with literals separated by ``|`` or ``,`` or ``or``.
    """
    if isinstance(spec, str):
        s = spec.strip()
        # Normalise ``or`` to ``|`` for splitting.
        s = re.sub(r"\bor\b", "|", s, flags=re.IGNORECASE)
        # Don't split commas inside parens (no parens in propositional).
        parts = re.split(r"[|,]", s)
        toks = [p for p in (p.strip() for p in parts) if p]
        if not toks:
            raise InvalidClause(f"empty clause string: {spec!r}")
        return [_parse_literal(t) for t in toks]
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise InvalidClause("empty clause")
        out: list[tuple[str, bool]] = []
        for t in spec:
            if isinstance(t, str):
                out.append(_parse_literal(t))
            elif isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str):
                out.append((t[0], bool(t[1])))
            else:
                raise InvalidClause(f"bad literal token: {t!r}")
        return out
    raise InvalidClause(f"unrecognised clause: {spec!r}")


def _canonical_clause(lits: list[tuple[str, bool]]) -> tuple[tuple[str, bool], ...] | None:
    """Deduplicate + sort + tautology-elim.  Returns None if tautological."""
    seen: dict[str, bool] = {}
    for atom, neg in lits:
        if atom in seen:
            if seen[atom] != neg:
                return None  # tautology — atom ∨ ¬atom
        else:
            seen[atom] = neg
    return tuple(sorted(seen.items()))


@dataclass(frozen=True)
class Resolution:
    """A single resolution step in an UNSAT proof."""

    clause_a: tuple[tuple[str, bool], ...]
    clause_b: tuple[tuple[str, bool], ...]
    pivot: str
    resolvent: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class Solution:
    """SAT solver outcome."""

    verdict: str                 # SAT / UNSAT / UNKNOWN
    model: Mapping[str, bool] | None = None
    decisions: int = 0
    conflicts: int = 0
    propagations: int = 0
    restarts: int = 0
    learned: int = 0
    elapsed_s: float = 0.0
    algorithm: str = ""
    proof_length: int = 0        # length of resolution proof (UNSAT)

    @property
    def satisfiable(self) -> bool:
        return self.verdict == SAT


@dataclass(frozen=True)
class ProofNode:
    """SLD-resolution proof tree node."""

    goal: str
    rule_id: int                 # index into the rule list; -1 for fact
    subgoals: tuple["ProofNode", ...] = ()


# =====================================================================
# CDCL data structures — internal integer encoding
# =====================================================================
# A literal is an int: variable index v (1..N) with sign — positive
# literal is +v, negative literal is -v.  Atom name lookup is a list.


def _enc_lit(var: int, neg: bool) -> int:
    return -var if neg else var


def _lit_var(lit: int) -> int:
    return -lit if lit < 0 else lit


def _lit_neg(lit: int) -> bool:
    return lit < 0


@dataclass
class _CDCLClause:
    lits: list[int]
    # Watched-literal indices into self.lits — 0 and 1 are watched.
    learned: bool = False
    activity: float = 0.0


# =====================================================================
# Reasoner
# =====================================================================


@dataclass
class ReasonerReport:
    """Summary of solver state and any anytime-valid bounds."""

    algorithm: str
    n_atoms: int
    n_clauses: int
    n_rules: int
    n_facts: int
    last_verdict: str | None
    last_model: Mapping[str, bool] | None
    decisions: int
    conflicts: int
    propagations: int
    restarts: int
    learned: int
    elapsed_s: float
    sat_calls: int
    derived_atoms: int
    proof_length: int
    failure_upper_clopper_pearson: float
    stratified: bool
    scc_sizes: list[int]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Cast the model values to plain types.
        if self.last_model is not None:
            d["last_model"] = dict(self.last_model)
        return d


class Reasoner:
    """Symbolic logical reasoning primitive.

    Three modes — propositional SAT, Horn / Datalog, and ASP — driven by
    one ``algorithm`` selector.  All modes coexist on the same instance;
    they share the atom registry and the fingerprint chain, but each
    mode interprets clauses / rules differently.

    The instance is side-effect-free w.r.t. the EventBus by default;
    pass ``bus=...`` to publish per-operation events.
    """

    def __init__(
        self,
        algorithm: str = CDCL,
        seed: int = 0,
        max_conflicts: int = _DEFAULT_MAX_CONFLICTS,
        walksat_flips: int = _DEFAULT_WALKSAT_FLIPS,
        walksat_restarts: int = _DEFAULT_WALKSAT_RESTARTS,
        walksat_noise: float = _DEFAULT_WALKSAT_NOISE,
        bus: Any | None = None,
        session_id: str | None = None,
    ) -> None:
        if algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"unknown algorithm {algorithm!r}; "
                f"expected one of {sorted(KNOWN_ALGORITHMS)}",
            )
        self.algorithm = algorithm
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._max_conflicts = int(max_conflicts)
        self._walksat_flips = int(walksat_flips)
        self._walksat_restarts = int(walksat_restarts)
        self._walksat_noise = float(walksat_noise)
        self._bus = bus
        self._session_id = session_id

        # Atom registry: name -> int index (1-based for sign encoding).
        self._atoms: dict[str, int] = {}
        self._atom_names: list[str] = []     # index 0 unused (1-based)

        # Propositional clauses as canonical tuples of (atom, neg).
        self._clauses: list[tuple[tuple[str, bool], ...]] = []
        # Horn rules: (head atom or None for constraint, [(atom, neg), ...]).
        self._rules: list[tuple[str | None, list[tuple[str, bool]]]] = []
        # Plain facts: set of atom names.
        self._facts: set[str] = set()

        # Last-call statistics.
        self._last_solution: Solution | None = None
        self._last_resolution_proof: list[Resolution] = []
        self._sat_calls = 0
        self._walksat_failures = 0
        self._walksat_attempts = 0

        # Fingerprint chain.
        self._fingerprint = _GENESIS
        self._publish(REASONER_STARTED, {
            "algorithm": algorithm, "seed": seed,
        })

    # ---------------------------------------------------------------
    # Atom registry
    # ---------------------------------------------------------------

    def _intern(self, atom: str) -> int:
        if not isinstance(atom, str) or not atom:
            raise InvalidClause(f"bad atom: {atom!r}")
        if atom in self._atoms:
            return self._atoms[atom]
        idx = len(self._atom_names) + 1
        self._atoms[atom] = idx
        self._atom_names.append(atom)
        return idx

    @property
    def atoms(self) -> list[str]:
        """Return the registered atom names in insertion order."""
        return list(self._atom_names)

    # ---------------------------------------------------------------
    # Fingerprint chain
    # ---------------------------------------------------------------

    def _absorb(self, payload: Any) -> None:
        digest = hashlib.sha256()
        digest.update(self._fingerprint.encode())
        digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
        self._fingerprint = digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    # ---------------------------------------------------------------
    # Clause / rule / fact API
    # ---------------------------------------------------------------

    def add_clause(self, spec: Any) -> None:
        """Add a propositional clause (disjunction of literals).

        Accepts ``["a", "~b", "c"]`` or ``"a | ~b | c"`` or ``"a or not b"``.
        Duplicates are deduplicated; tautologies are silently absorbed.
        """
        lits = _parse_clause(spec)
        canonical = _canonical_clause(lits)
        if canonical is None:
            self._absorb({"clause": "tautology"})
            self._publish(REASONER_CLAUSE_ADDED, {"tautology": True})
            return
        for atom, _ in canonical:
            self._intern(atom)
        self._clauses.append(canonical)
        self._absorb({"clause": [[a, n] for a, n in canonical]})
        self._publish(REASONER_CLAUSE_ADDED, {
            "literals": [[a, n] for a, n in canonical],
        })

    def add_clauses(self, specs: Iterable[Any]) -> None:
        for s in specs:
            self.add_clause(s)

    def add_fact(self, atom: str) -> None:
        """Add a unit clause / ground fact."""
        if not isinstance(atom, str) or not atom.strip():
            raise InvalidClause(f"bad fact: {atom!r}")
        a = atom.strip()
        self._intern(a)
        self._facts.add(a)
        # Also add the unit clause so SAT modes see it.
        self.add_clause([a])
        self._publish(REASONER_FACT_ADDED, {"atom": a})

    def add_horn(
        self,
        head: str | None,
        body: Sequence[str | tuple[str, bool]] = (),
    ) -> None:
        """Add a Horn rule ``head ← body[0] ∧ … ∧ body[k]``.

        ``head=None`` encodes an integrity constraint (``⊥ ← body``).
        Body literals may be negated for stable-model / ASP mode; they
        will be ignored by pure-Horn forward chaining unless STABLE_MODELS
        is selected.
        """
        if head is not None and (not isinstance(head, str) or not head.strip()):
            raise InvalidRule(f"bad head: {head!r}")
        body_lits: list[tuple[str, bool]] = []
        for b in body:
            if isinstance(b, str):
                body_lits.append(_parse_literal(b))
            elif isinstance(b, tuple) and len(b) == 2 and isinstance(b[0], str):
                body_lits.append((b[0], bool(b[1])))
            else:
                raise InvalidRule(f"bad body literal: {b!r}")
        if head is not None:
            self._intern(head)
        for a, _ in body_lits:
            self._intern(a)
        self._rules.append((head, body_lits))

        # Mirror into propositional clauses for SAT modes:
        #   head ∨ ¬b_1 ∨ … ∨ ¬b_k         (positive body literal → negated)
        # When ``head=None`` it's a hard constraint.
        clause_lits: list[tuple[str, bool]] = []
        if head is not None:
            clause_lits.append((head, False))
        for atom, neg in body_lits:
            # Body atom ``b`` (positive) becomes ¬b in the clause; a negated
            # body atom (NaF) flips back to a positive literal in the
            # *Clark-completion* sense.  For correctness over Horn programs
            # the completion is OK; for normal programs the stable-model
            # solver does the right thing semantically.
            clause_lits.append((atom, not neg))
        # add_clause handles dedup / tautology.
        canonical = _canonical_clause(clause_lits)
        if canonical is not None:
            self._clauses.append(canonical)
        self._absorb({"rule": {"head": head, "body": [[a, n] for a, n in body_lits]}})
        self._publish(REASONER_RULE_ADDED, {
            "head": head,
            "body": [[a, n] for a, n in body_lits],
        })

    _RULE_RE = re.compile(
        r"^\s*(?P<head>[^:]+?)\s*(?::-\s*(?P<body>.*?))?\s*\.?\s*$",
    )

    def add_rule(self, rule_str: str) -> None:
        """Parse a Prolog-style rule string.

        Examples::

            "ready(t)."                        # fact
            "ready(t) :- passed(p)."           # one-body rule
            ":- conflict(x, y), assigned(x, A), assigned(y, A)."  # constraint
            "p :- q, not r."                   # negation-as-failure
        """
        if not isinstance(rule_str, str):
            raise InvalidRule(f"rule must be a string: {rule_str!r}")
        s = rule_str.strip().rstrip(".").strip()
        if not s:
            raise InvalidRule("empty rule")
        if s.startswith(":-"):
            # Integrity constraint.
            body = s[2:].strip()
            body_tokens = [t.strip() for t in _split_commas(body) if t.strip()]
            self.add_horn(None, body_tokens)
            return
        m = self._RULE_RE.match(s)
        if not m:
            raise InvalidRule(f"unparseable rule: {rule_str!r}")
        head = m.group("head").strip()
        body = m.group("body")
        if body is None:
            self.add_horn(head, [])
            return
        body_tokens = [t.strip() for t in _split_commas(body) if t.strip()]
        self.add_horn(head, body_tokens)

    @property
    def n_atoms(self) -> int:
        return len(self._atom_names)

    @property
    def n_clauses(self) -> int:
        return len(self._clauses)

    @property
    def n_rules(self) -> int:
        return len(self._rules)

    @property
    def n_facts(self) -> int:
        return len(self._facts)

    @property
    def clauses(self) -> list[tuple[tuple[str, bool], ...]]:
        """Return a copy of the propositional clause list (read-only)."""
        return [c for c in self._clauses]

    @property
    def rules(self) -> list[tuple[str | None, list[tuple[str, bool]]]]:
        return [(h, list(b)) for h, b in self._rules]

    @property
    def facts(self) -> set[str]:
        return set(self._facts)

    # ---------------------------------------------------------------
    # SAT solving
    # ---------------------------------------------------------------

    def solve(
        self,
        assumptions: Mapping[str, bool] | None = None,
        algorithm: str | None = None,
        timeout_s: float | None = None,
    ) -> Solution:
        """Solve the propositional knowledge base.

        ``assumptions`` adds unit-clause assumptions for this call only.
        ``algorithm`` overrides the constructor default.  Returns a
        ``Solution`` with the verdict, model (if SAT), and counters.
        """
        algo = algorithm or self.algorithm
        if algo not in _SAT_ALGOS and algo != STABLE_MODELS:
            # Fall back to CDCL for non-SAT-mode .solve() calls.
            algo = CDCL
        clauses = list(self._clauses)
        if assumptions:
            for a, v in assumptions.items():
                self._intern(a)
                clauses.append(((a, not v),))    # a=True → unit (a, neg=False)
        t0 = time.perf_counter()
        self._sat_calls += 1
        if algo == DPLL:
            sol = _dpll_solve(clauses, self._atom_names, timeout_s)
        elif algo == CDCL:
            sol = _cdcl_solve(
                clauses, self._atom_names,
                max_conflicts=self._max_conflicts,
                seed=self._rng.randrange(1 << 30),
                timeout_s=timeout_s,
            )
        elif algo == WALKSAT:
            self._walksat_attempts += 1
            sol = _walksat_solve(
                clauses, self._atom_names,
                max_flips=self._walksat_flips,
                max_restarts=self._walksat_restarts,
                noise=self._walksat_noise,
                rng=self._rng,
                timeout_s=timeout_s,
            )
            if sol.verdict != SAT:
                self._walksat_failures += 1
        elif algo == STABLE_MODELS:
            # Single stable model — first one.
            models = self.stable_models(limit=1, assumptions=assumptions)
            if models:
                sol = Solution(
                    verdict=SAT,
                    model={a: (a in models[0]) for a in self._atom_names},
                    elapsed_s=time.perf_counter() - t0,
                    algorithm=STABLE_MODELS,
                )
            else:
                sol = Solution(
                    verdict=UNSAT, algorithm=STABLE_MODELS,
                    elapsed_s=time.perf_counter() - t0,
                )
        else:
            raise UnknownAlgorithm(f"cannot solve with algorithm {algo!r}")
        # Stamp elapsed if solver didn't.
        if sol.elapsed_s == 0.0:
            sol = Solution(
                verdict=sol.verdict, model=sol.model,
                decisions=sol.decisions, conflicts=sol.conflicts,
                propagations=sol.propagations, restarts=sol.restarts,
                learned=sol.learned,
                elapsed_s=time.perf_counter() - t0,
                algorithm=sol.algorithm or algo,
                proof_length=sol.proof_length,
            )
        self._last_solution = sol
        if sol.verdict == UNSAT:
            # Build a resolution proof on the kept clauses.
            try:
                self._last_resolution_proof = _resolution_proof(clauses)
                if self._last_resolution_proof:
                    self._absorb({
                        "unsat_proof_len": len(self._last_resolution_proof),
                    })
                    self._publish(REASONER_PROOF, {
                        "length": len(self._last_resolution_proof),
                    })
            except Exception:
                self._last_resolution_proof = []
        else:
            self._last_resolution_proof = []
        self._publish(REASONER_SOLVED, {
            "verdict": sol.verdict, "algorithm": algo,
            "decisions": sol.decisions, "conflicts": sol.conflicts,
            "elapsed_s": sol.elapsed_s,
        })
        return sol

    def is_satisfiable(self) -> bool:
        return self.solve().verdict == SAT

    def all_models(self, limit: int = 16) -> list[dict[str, bool]]:
        """Enumerate up to ``limit`` distinct propositional models."""
        if limit <= 0:
            return []
        out: list[dict[str, bool]] = []
        blocking: list[tuple[tuple[str, bool], ...]] = []
        for _ in range(limit):
            sol = _cdcl_solve(
                self._clauses + blocking, self._atom_names,
                max_conflicts=self._max_conflicts,
                seed=self._rng.randrange(1 << 30),
            )
            if sol.verdict != SAT or sol.model is None:
                break
            out.append(dict(sol.model))
            # Add a blocking clause negating the found model.
            block_lits = [(a, sol.model.get(a, False)) for a in self._atom_names]
            blocking.append(tuple(block_lits))
        return out

    def entails(self, formula: Any) -> bool:
        """Return True iff the knowledge base entails the formula.

        Implementation: check that the KB ∧ ¬formula is UNSAT.
        ``formula`` may be a single atom (treated as the literal), a
        sequence of literals (treated as a *conjunction* of literals),
        or a string parseable by ``_parse_literal`` for a single literal.

        For full propositional formulas, encode the negation as
        additional clauses via ``add_clause`` on a *clone* of the
        Reasoner.
        """
        # Single literal.
        if isinstance(formula, str):
            atom, neg = _parse_literal(formula)
            neg_lit = (atom, not neg)
            extra = (neg_lit,)
            clauses = self._clauses + [extra]
            sol = _cdcl_solve(clauses, self._atom_names + ([atom] if atom not in self._atoms else []),
                              max_conflicts=self._max_conflicts,
                              seed=self._rng.randrange(1 << 30))
            return sol.verdict == UNSAT
        if isinstance(formula, (list, tuple)):
            # Conjunction of literals — check KB ∧ negation-of-conjunction
            # i.e. add a single clause of negated literals.
            lits = _parse_clause(list(formula))
            # ¬(l1 ∧ … ∧ lk) = ¬l1 ∨ … ∨ ¬lk
            seen: dict[str, bool] = {}
            for a, n in lits:
                seen[a] = not n
            neg_clause = tuple(sorted(seen.items()))
            new_atoms = [a for a, _ in lits if a not in self._atoms]
            clauses = self._clauses + [neg_clause]
            sol = _cdcl_solve(clauses, self._atom_names + new_atoms,
                              max_conflicts=self._max_conflicts,
                              seed=self._rng.randrange(1 << 30))
            return sol.verdict == UNSAT
        raise InvalidClause(f"unrecognised formula: {formula!r}")

    # ---------------------------------------------------------------
    # Horn / Datalog
    # ---------------------------------------------------------------

    def forward_chain(self, semi_naive: bool = True) -> set[str]:
        """Compute the least Herbrand model of the Horn rules + facts.

        Auto-detects whether any rule contains Datalog variables
        (Prolog-convention uppercase / underscore tokens inside a
        predicate-style atom).  If so, runs a full unification-based
        forward chain; otherwise runs the fast propositional path.

        Negation-as-failure body literals in normal logic programs are
        not derived by pure forward chaining; for that semantics call
        ``stable_models``.  Body literals with ``neg=True`` are skipped
        here.
        """
        derived: set[str] = set(self._facts)
        # Include any unit clauses too.
        for clause in self._clauses:
            if len(clause) == 1 and clause[0][1] is False:
                derived.add(clause[0][0])

        any_vars = any(
            _rule_has_variables(h, b) for h, b in self._rules
        )

        if any_vars:
            # Datalog path.
            while True:
                new = _datalog_step(self._rules, derived)
                if not new:
                    break
                for h in new:
                    self._publish(REASONER_DERIVED, {"head": h})
                derived |= new
            return derived

        # Propositional Horn path.
        rules_h: list[tuple[str, list[str]]] = []
        for head, body in self._rules:
            if head is None:
                continue
            pos_body = [a for a, neg in body if not neg]
            rules_h.append((head, pos_body))

        if not semi_naive:
            changed = True
            while changed:
                changed = False
                for head, body in rules_h:
                    if head in derived:
                        continue
                    if all(b in derived for b in body):
                        derived.add(head)
                        changed = True
                        self._publish(REASONER_DERIVED, {"head": head})
            return derived

        delta = set(derived)
        while delta:
            new_delta: set[str] = set()
            for head, body in rules_h:
                if head in derived:
                    continue
                if not body:
                    new_delta.add(head)
                    continue
                if not all(b in derived for b in body):
                    continue
                if any(b in delta for b in body):
                    new_delta.add(head)
            if not new_delta:
                break
            for h in new_delta:
                self._publish(REASONER_DERIVED, {"head": h})
            derived |= new_delta
            delta = new_delta
        return derived

    def backward_chain(
        self,
        query: str,
        max_depth: int = 64,
    ) -> ProofNode | None:
        """SLD-resolution proof of ``query``.

        Returns the proof tree if the query succeeds, else None.  Uses
        subsumption tabling (Tamaki-Sato 1986) to terminate on left-
        recursive Horn rules.  Supports Datalog-variable rules via
        Robinson (1965) unification.

        ``query`` may itself contain variables.  We return a proof of
        the *first* unifying ground instance.
        """
        any_vars = any(
            _rule_has_variables(h, b) for h, b in self._rules
        )
        if not any_vars:
            return self._propositional_backward(query, max_depth)
        # Datalog backward chain via SLD resolution with backtracking.
        try:
            query_term = _parse_term(query.strip())
        except Exception:
            return None
        open_set: set[str] = set()
        counter = [0]

        def _prove(goal_term, subst: dict, depth: int):
            """Yield (ProofNode, extended_substitution) for each solution."""
            if depth > max_depth:
                return
            g = _substitute(goal_term, subst)
            g_str = _term_to_str(g)
            # Fact matches (with backtracking).
            for f in self._facts:
                try:
                    ft = _parse_term(f)
                except Exception:
                    continue
                u = _unify(g, ft, dict(subst))
                if u is not None:
                    inst = _substitute(g, u)
                    yield ProofNode(
                        goal=_term_to_str(inst), rule_id=-1, subgoals=(),
                    ), u
            # Cycle guard on the abstract goal key.
            if g_str in open_set:
                return
            open_set.add(g_str)
            try:
                for ri, (head, body) in enumerate(self._rules):
                    if head is None:
                        continue
                    try:
                        head_term = _parse_term(head)
                    except Exception:
                        continue
                    counter[0] += 1
                    rename = _fresh_vars_counter(head_term, body, counter[0])
                    head_inst = _rename_vars(head_term, rename)
                    body_inst = []
                    parsable = True
                    for a, neg in body:
                        try:
                            body_inst.append(
                                (_rename_vars(_parse_term(a), rename), neg),
                            )
                        except Exception:
                            parsable = False
                            break
                    if not parsable:
                        continue
                    u = _unify(g, head_inst, dict(subst))
                    if u is None:
                        continue
                    yield from _prove_body(
                        body_inst, [], u, depth, ri, head_inst,
                    )
            finally:
                open_set.discard(g_str)

        def _prove_body(body_inst, accum_nodes, cur_subst, depth, ri, head_inst):
            if not body_inst:
                inst_goal = _substitute(head_inst, cur_subst)
                yield ProofNode(
                    goal=_term_to_str(inst_goal),
                    rule_id=ri,
                    subgoals=tuple(accum_nodes),
                ), cur_subst
                return
            bt, neg = body_inst[0]
            rest = body_inst[1:]
            if neg:
                inst = _substitute(bt, cur_subst)
                if _has_variables(inst):
                    return
                proven = False
                for _ in _prove(inst, cur_subst, depth + 1):
                    proven = True
                    break
                if proven:
                    return
                yield from _prove_body(
                    rest, accum_nodes, cur_subst, depth, ri, head_inst,
                )
                return
            for sub_node, sub_subst in _prove(bt, cur_subst, depth + 1):
                yield from _prove_body(
                    rest, accum_nodes + [sub_node],
                    sub_subst, depth, ri, head_inst,
                )

        for proof, _ in _prove(query_term, {}, 0):
            return proof
        return None

    def _propositional_backward(
        self, query: str, max_depth: int,
    ) -> ProofNode | None:
        rules_pos: list[tuple[str, list[str]]] = []
        for head, body in self._rules:
            if head is None:
                continue
            rules_pos.append((head, [a for a, neg in body if not neg]))

        memo: dict[str, ProofNode | None] = {}
        open_set: set[str] = set()

        def _prove(goal: str, depth: int) -> ProofNode | None:
            if depth > max_depth:
                return None
            if goal in memo:
                return memo[goal]
            if goal in self._facts:
                node = ProofNode(goal=goal, rule_id=-1, subgoals=())
                memo[goal] = node
                return node
            if goal in open_set:
                return None
            open_set.add(goal)
            try:
                for ri, (head, body) in enumerate(rules_pos):
                    if head != goal:
                        continue
                    subnodes: list[ProofNode] = []
                    ok = True
                    for b in body:
                        sub = _prove(b, depth + 1)
                        if sub is None:
                            ok = False
                            break
                        subnodes.append(sub)
                    if ok:
                        node = ProofNode(
                            goal=goal, rule_id=ri,
                            subgoals=tuple(subnodes),
                        )
                        memo[goal] = node
                        return node
                memo[goal] = None
                return None
            finally:
                open_set.discard(goal)

        return _prove(query.strip(), 0)

    # ---------------------------------------------------------------
    # Answer Set Programming — stable model semantics
    # ---------------------------------------------------------------

    def stable_models(
        self,
        limit: int = 16,
        assumptions: Mapping[str, bool] | None = None,
    ) -> list[set[str]]:
        """Enumerate up to ``limit`` stable models (Gelfond-Lifschitz 1988).

        Auto-grounds rules that contain Datalog variables (Prolog-
        convention uppercase tokens) via full grounding over the
        Herbrand universe of constants in the program.  For programs
        without variables the rules are used as-is.

        Implementation: enumerate candidate truth assignments for atoms
        that occur in *negated* body positions, compute the reduct
        ``P^M``, evaluate to its least Horn fixpoint, accept iff fixpoint
        equals candidate restricted to head-supported atoms.  Stratified
        programs (no negation cycle) take a fast path that iterates the
        rules to a unique minimal model.
        """
        # If any rule has variables, ground first.
        any_vars = any(_rule_has_variables(h, b) for h, b in self._rules)
        if any_vars:
            rules_g = _ground_rules(self._rules, self._facts)
        else:
            rules_g = [(h, list(b)) for h, b in self._rules]
        # Atoms that ever appear under NaF — these are the "guess" atoms.
        naf_atoms: set[str] = set()
        for h, body in rules_g:
            for a, neg in body:
                if neg:
                    naf_atoms.add(a)

        # Detect stratified negation on the *grounded* rules.
        all_atom_names: set[str] = set(self._atom_names)
        for h, body in rules_g:
            if h is not None:
                all_atom_names.add(h)
            for a, _ in body:
                all_atom_names.add(a)
        all_atoms = sorted(all_atom_names)
        atom_to_idx = {a: i for i, a in enumerate(all_atoms)}
        n = len(all_atoms)
        edges: list[tuple[int, int]] = []
        neg_edges: set[tuple[int, int]] = set()
        for h, body in rules_g:
            if h is None:
                continue
            hi = atom_to_idx[h]
            for a, neg in body:
                ai = atom_to_idx[a]
                edges.append((ai, hi))
                if neg:
                    neg_edges.add((ai, hi))
        sccs = strongly_connected_components(n, edges)
        comp_of: dict[int, int] = {}
        for ci, comp in enumerate(sccs):
            for v in comp:
                comp_of[v] = ci
        stratified = True
        for (u, v) in neg_edges:
            if comp_of.get(u, -1) == comp_of.get(v, -2):
                stratified = False
                break

        if stratified and not assumptions:
            # Fast path for stratified programs.
            model: set[str] = set(self._facts)
            changed = True
            iterations = 0
            while changed and iterations < 4096:
                changed = False
                iterations += 1
                for h, body in rules_g:
                    if h is None:
                        continue
                    if h in model:
                        continue
                    if all(
                        ((a in model) if not neg else (a not in model))
                        for a, neg in body
                    ):
                        model.add(h)
                        changed = True
            for h, body in rules_g:
                if h is None and all(
                    ((a in model) if not neg else (a not in model))
                    for a, neg in body
                ):
                    return []
            return [model]

        # General guess-and-check.
        guess_atoms = sorted(naf_atoms)
        if len(guess_atoms) > 24:
            sols = self.all_models(limit=limit)
            return [set(a for a, v in m.items() if v) for m in sols]

        results: list[set[str]] = []
        for mask in range(1 << len(guess_atoms)):
            candidate_true = set()
            for i, a in enumerate(guess_atoms):
                if (mask >> i) & 1:
                    candidate_true.add(a)
            if assumptions:
                consistent = True
                for k, v in assumptions.items():
                    if v and k in naf_atoms and k not in candidate_true:
                        consistent = False; break
                    if not v and k in naf_atoms and k in candidate_true:
                        consistent = False; break
                if not consistent:
                    continue
            reduct: list[tuple[str | None, list[str]]] = []
            for h, body in rules_g:
                drop = False
                pos: list[str] = []
                for a, neg in body:
                    if neg:
                        if a in candidate_true:
                            drop = True
                            break
                    else:
                        pos.append(a)
                if drop:
                    continue
                reduct.append((h, pos))
            # Compute least model of the Horn reduct.
            model: set[str] = set(self._facts)
            # Apply assumptions to the model directly.
            if assumptions:
                for k, v in assumptions.items():
                    if v and k not in naf_atoms:
                        model.add(k)
            changed = True
            while changed:
                changed = False
                for h, body in reduct:
                    if h is None:
                        continue
                    if h in model:
                        continue
                    if all(b in model for b in body):
                        model.add(h)
                        changed = True
            # Check integrity constraints.
            constraint_ok = True
            for h, body in reduct:
                if h is None and all(b in model for b in body):
                    constraint_ok = False
                    break
            if not constraint_ok:
                continue
            # Equality on NaF atoms: model ∩ naf_atoms == candidate_true ∩ naf_atoms
            if (model & naf_atoms) != (candidate_true & naf_atoms):
                continue
            if assumptions:
                bad = False
                for k, v in assumptions.items():
                    if v and k not in model:
                        bad = True; break
                    if not v and k in model:
                        bad = True; break
                if bad:
                    continue
            # Deduplicate.
            if model in results:
                continue
            results.append(model)
            if len(results) >= limit:
                break
        return results

    # ---------------------------------------------------------------
    # Reports + replay
    # ---------------------------------------------------------------

    def last_solution(self) -> Solution | None:
        return self._last_solution

    def last_resolution_proof(self) -> list[Resolution]:
        return list(self._last_resolution_proof)

    def report(self, alpha: float = 0.05) -> ReasonerReport:
        """Emit a `ReasonerReport` with anytime-valid bounds."""
        # Failure-rate Clopper-Pearson upper bound for Walk-SAT.
        if self._walksat_attempts > 0:
            fail_ub = clopper_pearson_upper(
                self._walksat_failures, self._walksat_attempts, alpha,
            )
        else:
            fail_ub = 0.0
        # Stratification diagnostic over the rule dependency graph.
        n = len(self._atom_names)
        atom_to_idx = {a: i for i, a in enumerate(self._atom_names)}
        edges: list[tuple[int, int]] = []
        neg_edges: set[tuple[int, int]] = set()
        for h, body in self._rules:
            if h is None or h not in atom_to_idx:
                continue
            hi = atom_to_idx[h]
            for a, neg in body:
                ai = atom_to_idx.get(a)
                if ai is None:
                    continue
                edges.append((ai, hi))
                if neg:
                    neg_edges.add((ai, hi))
        sccs = strongly_connected_components(n, edges)
        scc_sizes = [len(c) for c in sccs]
        comp_of = {}
        for ci, comp in enumerate(sccs):
            for v in comp:
                comp_of[v] = ci
        stratified = all(
            comp_of.get(u) != comp_of.get(v) for (u, v) in neg_edges
        )
        last = self._last_solution
        derived = 0
        try:
            derived = len(self.forward_chain())
        except Exception:
            derived = 0
        rep = ReasonerReport(
            algorithm=self.algorithm,
            n_atoms=self.n_atoms,
            n_clauses=self.n_clauses,
            n_rules=self.n_rules,
            n_facts=self.n_facts,
            last_verdict=last.verdict if last else None,
            last_model=dict(last.model) if last and last.model else None,
            decisions=last.decisions if last else 0,
            conflicts=last.conflicts if last else 0,
            propagations=last.propagations if last else 0,
            restarts=last.restarts if last else 0,
            learned=last.learned if last else 0,
            elapsed_s=last.elapsed_s if last else 0.0,
            sat_calls=self._sat_calls,
            derived_atoms=derived,
            proof_length=len(self._last_resolution_proof),
            failure_upper_clopper_pearson=fail_ub,
            stratified=stratified,
            scc_sizes=scc_sizes,
            fingerprint=self._fingerprint,
        )
        self._publish(REASONER_REPORT, {
            "verdict": rep.last_verdict,
            "n_clauses": rep.n_clauses,
            "stratified": rep.stratified,
        })
        return rep

    def clear(self) -> None:
        """Reset the knowledge base (preserve algorithm/seed)."""
        self._atoms.clear()
        self._atom_names.clear()
        self._clauses.clear()
        self._rules.clear()
        self._facts.clear()
        self._last_solution = None
        self._last_resolution_proof = []
        self._sat_calls = 0
        self._walksat_failures = 0
        self._walksat_attempts = 0
        self._fingerprint = _GENESIS
        self._publish(REASONER_CLEARED, {})

    # ---------------------------------------------------------------
    # Event publishing
    # ---------------------------------------------------------------

    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            from agi.events import Event
            ev = Event(kind=kind, session_id=self._session_id, data=dict(data))
            self._bus.publish(ev)
        except Exception:
            pass


def _split_commas(s: str) -> list[str]:
    """Split on commas at the top level (respects parens)."""
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [t.strip() for t in out if t.strip()]


# =====================================================================
# Datalog-style term parsing + unification
# =====================================================================
# Convention: tokens starting with uppercase or an underscore are
# *variables*; everything else is a *constant*.  Atom syntax is
# ``functor(arg1, arg2, ...)`` or a bare atom name.  Nested function
# terms ``f(g(X), Y)`` are supported recursively.


def _is_variable(token: str) -> bool:
    if not token:
        return False
    return token[0].isupper() or token[0] == "_"


def _parse_term(s: str) -> tuple[str, tuple]:
    """Parse a term string into a (functor, (args,...)) tree.

    Bare atom ``"p"`` parses as ``("p", ())``.
    Predicate ``"p(a, X, q(b))"`` parses as
        ``("p", ("a", "X", ("q", ("b",))))``.
    Variables are kept as bare strings; constants and nested functors
    as their parsed forms.
    """
    s = s.strip()
    if not s:
        raise InvalidClause("empty term")
    paren = s.find("(")
    if paren < 0:
        # Bare atom / variable.
        return (s, ())
    if not s.endswith(")"):
        raise InvalidClause(f"unbalanced parens: {s!r}")
    functor = s[:paren].strip()
    inner = s[paren + 1: -1]
    arg_strs = _split_commas(inner)
    args: list = []
    for a in arg_strs:
        if "(" in a:
            args.append(_parse_term(a))
        else:
            args.append(a.strip())
    return (functor, tuple(args))


def _has_variables(term) -> bool:
    if isinstance(term, str):
        return _is_variable(term)
    if isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], str):
        return any(_has_variables(a) for a in term[1])
    return False


def _substitute(term, subst: dict[str, Any]):
    """Apply substitution to a term."""
    if isinstance(term, str):
        if _is_variable(term) and term in subst:
            return _substitute(subst[term], subst)
        return term
    if isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], str):
        return (term[0], tuple(_substitute(a, subst) for a in term[1]))
    return term


def _unify(a, b, subst: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Robinson 1965 unification of two terms.

    Returns the most-general unifier (MGU) as a substitution dict, or
    ``None`` if the terms do not unify.  Occurs-check enabled.
    """
    if subst is None:
        subst = {}
    a = _walk(a, subst)
    b = _walk(b, subst)
    if a == b:
        return subst
    if isinstance(a, str) and _is_variable(a):
        if _occurs(a, b, subst):
            return None
        return {**subst, a: b}
    if isinstance(b, str) and _is_variable(b):
        if _occurs(b, a, subst):
            return None
        return {**subst, b: a}
    if (
        isinstance(a, tuple) and isinstance(b, tuple)
        and len(a) == 2 and len(b) == 2
        and a[0] == b[0] and len(a[1]) == len(b[1])
    ):
        s = subst
        for x, y in zip(a[1], b[1]):
            s = _unify(x, y, s)
            if s is None:
                return None
        return s
    return None


def _walk(t, subst: dict[str, Any]):
    while isinstance(t, str) and _is_variable(t) and t in subst:
        t = subst[t]
    return t


def _occurs(var: str, term, subst: dict[str, Any]) -> bool:
    term = _walk(term, subst)
    if term == var:
        return True
    if isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], str):
        return any(_occurs(var, a, subst) for a in term[1])
    return False


def _term_to_str(term) -> str:
    """Render a parsed term back to its canonical string form."""
    if isinstance(term, str):
        return term
    if isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], str):
        functor, args = term
        if not args:
            return functor
        return f"{functor}({', '.join(_term_to_str(a) for a in args)})"
    return str(term)


def _rule_has_variables(head: str | None, body: list[tuple[str, bool]]) -> bool:
    if head is not None:
        try:
            if _has_variables(_parse_term(head)):
                return True
        except Exception:
            pass
    for atom, _ in body:
        try:
            if _has_variables(_parse_term(atom)):
                return True
        except Exception:
            pass
    return False


def _collect_variables(term, out: set[str]) -> None:
    if isinstance(term, str):
        if _is_variable(term):
            out.add(term)
        return
    if isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], str):
        for a in term[1]:
            _collect_variables(a, out)


def _rename_vars(term, rename: dict[str, str]):
    if isinstance(term, str):
        if _is_variable(term):
            return rename.get(term, term)
        return term
    if isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], str):
        return (term[0], tuple(_rename_vars(a, rename) for a in term[1]))
    return term


def _fresh_vars(
    head_term,
    body: list[tuple[str, bool]],
    depth: int,
    rule_id: int,
) -> dict[str, str]:
    vs: set[str] = set()
    _collect_variables(head_term, vs)
    for atom, _ in body:
        try:
            _collect_variables(_parse_term(atom), vs)
        except Exception:
            pass
    return {v: f"_{v}_{rule_id}_{depth}" for v in vs}


def _fresh_vars_counter(
    head_term,
    body: list[tuple[str, bool]],
    counter: int,
) -> dict[str, str]:
    vs: set[str] = set()
    _collect_variables(head_term, vs)
    for atom, _ in body:
        try:
            _collect_variables(_parse_term(atom), vs)
        except Exception:
            pass
    return {v: f"_G{counter}_{v}" for v in vs}


def _collect_constants(term, out: set[str]) -> None:
    if isinstance(term, str):
        if not _is_variable(term):
            out.add(term)
        return
    if isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], str):
        for a in term[1]:
            _collect_constants(a, out)


def _ground_rules(
    rules: list[tuple[str | None, list[tuple[str, bool]]]],
    facts: set[str],
) -> list[tuple[str | None, list[tuple[str, bool]]]]:
    """Naïve full grounding of a Datalog program.

    Collects the Herbrand universe (constants from facts + rule heads /
    bodies), then substitutes every variable in every rule with every
    possible constant tuple.  Exponential in the maximum arity × number
    of variables — fine for the small programs the runtime cares about
    (a few hundred ground atoms).
    """
    constants: set[str] = set()
    for f in facts:
        try:
            _collect_constants(_parse_term(f), constants)
        except Exception:
            pass
    for h, body in rules:
        if h is not None:
            try:
                _collect_constants(_parse_term(h), constants)
            except Exception:
                pass
        for atom, _ in body:
            try:
                _collect_constants(_parse_term(atom), constants)
            except Exception:
                pass
    # Filter: predicate names (functors with arity > 0) and unary atom
    # names appear in `constants` if they occur bare anywhere.  Keep
    # them — they're treated uniformly as ground terms.
    constants_list = sorted(constants)

    grounded: list[tuple[str | None, list[tuple[str, bool]]]] = []
    for h, body in rules:
        # Variables in this rule.
        vs: set[str] = set()
        if h is not None:
            try:
                _collect_variables(_parse_term(h), vs)
            except Exception:
                pass
        for atom, _ in body:
            try:
                _collect_variables(_parse_term(atom), vs)
            except Exception:
                pass
        vars_list = sorted(vs)
        if not vars_list:
            grounded.append((h, list(body)))
            continue
        # Cartesian product of constants over each variable.
        from itertools import product
        # Cap explosion: ≤ 100k ground rules per source rule.
        if len(constants_list) ** len(vars_list) > 100_000:
            # Skip — too big to ground.  Coordinator should switch
            # algorithm or shrink.
            continue
        for assignment in product(constants_list, repeat=len(vars_list)):
            subst = dict(zip(vars_list, assignment))
            h_g: str | None = None
            if h is not None:
                try:
                    h_g = _term_to_str(_substitute(_parse_term(h), subst))
                except Exception:
                    h_g = None
            body_g: list[tuple[str, bool]] = []
            ok = True
            for atom, neg in body:
                try:
                    body_g.append((
                        _term_to_str(_substitute(_parse_term(atom), subst)),
                        neg,
                    ))
                except Exception:
                    ok = False
                    break
            if ok:
                grounded.append((h_g, body_g))
    return grounded


def _datalog_step(
    rules: list[tuple[str | None, list[tuple[str, bool]]]],
    derived: set[str],
) -> set[str]:
    """One semi-naïve Datalog step.

    For each rule whose body unifies with a tuple of derived facts,
    generate the head substitution; return *new* derived atoms.
    """
    new: set[str] = set()
    parsed_derived: list[tuple[str, tuple]] = []
    for f in derived:
        try:
            parsed_derived.append(_parse_term(f))
        except Exception:
            continue
    for head, body in rules:
        if head is None:
            continue
        try:
            head_term = _parse_term(head)
        except Exception:
            continue
        pos_body = [a for a, neg in body if not neg]
        body_terms = []
        skip = False
        for b in pos_body:
            try:
                body_terms.append(_parse_term(b))
            except Exception:
                skip = True
                break
        if skip:
            continue
        # Try every combination of derived facts to instantiate the body.
        def _enum(i: int, subst: dict[str, Any]) -> None:
            if i == len(body_terms):
                head_inst = _substitute(head_term, subst)
                if _has_variables(head_inst):
                    return
                s = _term_to_str(head_inst)
                if s not in derived:
                    new.add(s)
                return
            target = body_terms[i]
            for fact in parsed_derived:
                # Try to unify.
                u = _unify(target, fact, dict(subst))
                if u is not None:
                    _enum(i + 1, u)

        if not body_terms:
            # Pure head — already handled in propositional path.
            inst = _substitute(head_term, {})
            if not _has_variables(inst):
                s = _term_to_str(inst)
                if s not in derived:
                    new.add(s)
            continue
        _enum(0, {})
    return new


# =====================================================================
# DPLL — recursive backtracking with iterative trail
# =====================================================================


def _dpll_solve(
    clauses: list[tuple[tuple[str, bool], ...]],
    atom_names: list[str],
    timeout_s: float | None = None,
) -> Solution:
    """Classic DPLL with unit propagation + pure-literal elimination."""
    t0 = time.perf_counter()
    if not clauses:
        return Solution(verdict=SAT, model={a: False for a in atom_names},
                        algorithm=DPLL,
                        elapsed_s=time.perf_counter() - t0)
    # Encode.
    atom_idx: dict[str, int] = {a: i + 1 for i, a in enumerate(atom_names)}
    # Ensure every clause atom is in atom_idx (defensive).
    next_idx = len(atom_idx) + 1
    enc_clauses: list[list[int]] = []
    for cl in clauses:
        encl: list[int] = []
        for a, neg in cl:
            if a not in atom_idx:
                atom_idx[a] = next_idx
                next_idx += 1
            encl.append(-atom_idx[a] if neg else atom_idx[a])
        enc_clauses.append(encl)
    nvars = next_idx - 1
    names = list(atom_names)
    for a in atom_idx:
        if a not in atom_names:
            names.append(a)

    decisions = [0]
    propagations = [0]
    conflicts = [0]

    def _solve(assigned: dict[int, bool]) -> dict[int, bool] | None:
        if timeout_s is not None and (time.perf_counter() - t0) > timeout_s:
            return None
        # Simplify clauses under current assignment.
        simplified: list[list[int]] = []
        for cl in enc_clauses:
            satisfied = False
            new_cl: list[int] = []
            for lit in cl:
                v = _lit_var(lit)
                if v in assigned:
                    val = assigned[v] != _lit_neg(lit)
                    # val=True when literal evaluates true
                    val_actual = (assigned[v] and not _lit_neg(lit)) or (not assigned[v] and _lit_neg(lit))
                    if val_actual:
                        satisfied = True
                        break
                else:
                    new_cl.append(lit)
            if satisfied:
                continue
            if not new_cl:
                conflicts[0] += 1
                return None        # conflict
            simplified.append(new_cl)
        # Unit propagation.
        changed = True
        while changed:
            changed = False
            for cl in simplified:
                if len(cl) == 1:
                    lit = cl[0]
                    v = _lit_var(lit)
                    val = not _lit_neg(lit)
                    if v in assigned:
                        if assigned[v] != val:
                            conflicts[0] += 1
                            return None
                    else:
                        assigned = {**assigned, v: val}
                        propagations[0] += 1
                        changed = True
            if changed:
                # Re-simplify.
                new_simpl: list[list[int]] = []
                conflict = False
                for cl in simplified:
                    satisfied = False
                    new_cl: list[int] = []
                    for lit in cl:
                        v = _lit_var(lit)
                        if v in assigned:
                            val_actual = (assigned[v] and not _lit_neg(lit)) or (not assigned[v] and _lit_neg(lit))
                            if val_actual:
                                satisfied = True
                                break
                        else:
                            new_cl.append(lit)
                    if satisfied:
                        continue
                    if not new_cl:
                        conflict = True
                        break
                    new_simpl.append(new_cl)
                if conflict:
                    conflicts[0] += 1
                    return None
                simplified = new_simpl
        if not simplified:
            return assigned
        # Pure-literal elimination.
        pos: set[int] = set()
        neg: set[int] = set()
        for cl in simplified:
            for lit in cl:
                if lit > 0:
                    pos.add(lit)
                else:
                    neg.add(-lit)
        pures = (pos - neg) | (neg - pos)
        if pures:
            for v in pures:
                if v in assigned:
                    continue
                if v in pos:
                    assigned = {**assigned, v: True}
                else:
                    assigned = {**assigned, v: False}
            return _solve(assigned)
        # Decide.
        # Pick the variable in the shortest clause.
        target = min(simplified, key=len)
        lit = target[0]
        v = _lit_var(lit)
        decisions[0] += 1
        for val in (not _lit_neg(lit), _lit_neg(lit)):
            r = _solve({**assigned, v: val})
            if r is not None:
                return r
        return None

    result = _solve({})
    elapsed = time.perf_counter() - t0
    if result is None:
        return Solution(verdict=UNSAT, decisions=decisions[0],
                        conflicts=conflicts[0], propagations=propagations[0],
                        algorithm=DPLL, elapsed_s=elapsed)
    # Build the model.
    model = {a: result.get(atom_idx[a], False) for a in atom_names}
    # Default unconstrained vars (atoms not in atom_names) → False.
    return Solution(
        verdict=SAT, model=model, decisions=decisions[0],
        conflicts=conflicts[0], propagations=propagations[0],
        algorithm=DPLL, elapsed_s=elapsed,
    )


# =====================================================================
# CDCL — conflict-driven clause learning with VSIDS + 1UIP + Luby restart
# =====================================================================


def _luby(i: int) -> int:
    """Luby sequence value at index i (Luby-Sinclair-Zuckerman 1993)."""
    k = 1
    while (1 << k) - 1 < i + 1:
        k += 1
    if (1 << k) - 1 == i + 1:
        return 1 << (k - 1)
    return _luby(i - ((1 << (k - 1)) - 1))


def _cdcl_solve(
    clauses: list[tuple[tuple[str, bool], ...]],
    atom_names: list[str],
    max_conflicts: int = _DEFAULT_MAX_CONFLICTS,
    seed: int = 0,
    timeout_s: float | None = None,
) -> Solution:
    """CDCL SAT solver — VSIDS branching, 1UIP learning, Luby restarts.

    Implementation choices:

      * Two-watched-literals (Zhang-Stickel 1996) for unit propagation.
      * VSIDS (Moskewicz et al. 2001) with bump-and-decay for branching.
      * 1UIP (Zhang-Madigan-Moskewicz-Malik 2001) for clause learning.
      * Luby sequence (Luby-Sinclair-Zuckerman 1993) for restart timing.

    All in pure Python, list-of-list storage — fast enough for runtime
    coordination queries on up-to-medium-size CNFs (≲10k clauses).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)

    # Empty problem.
    if not clauses:
        return Solution(
            verdict=SAT, model={a: False for a in atom_names},
            algorithm=CDCL, elapsed_s=time.perf_counter() - t0,
        )

    # Encode atoms; build the variable index.
    atom_idx: dict[str, int] = {a: i + 1 for i, a in enumerate(atom_names)}
    next_idx = len(atom_idx) + 1
    enc_clauses: list[list[int]] = []
    for cl in clauses:
        encl: list[int] = []
        for a, neg in cl:
            if a not in atom_idx:
                atom_idx[a] = next_idx
                next_idx += 1
            encl.append(-atom_idx[a] if neg else atom_idx[a])
        # Dedup within a clause + tautology elim.
        seen: dict[int, int] = {}
        toks: list[int] = []
        for lit in encl:
            v = _lit_var(lit)
            if v in seen:
                if seen[v] != lit:
                    # Tautology — drop the clause.
                    toks = []
                    encl = []
                    break
                continue
            seen[v] = lit
            toks.append(lit)
        if toks:
            enc_clauses.append(toks)
    nvars = next_idx - 1
    names = list(atom_names)
    extra_names = sorted([a for a in atom_idx if a not in set(atom_names)],
                         key=lambda x: atom_idx[x])
    for a in extra_names:
        names.append(a)
    nvars = max(nvars, len(names))

    # Trivial empty clause → UNSAT.
    if any(len(cl) == 0 for cl in enc_clauses):
        return Solution(verdict=UNSAT, algorithm=CDCL,
                        elapsed_s=time.perf_counter() - t0)

    # Clause storage; each is _CDCLClause; watches stored separately.
    clause_list: list[_CDCLClause] = [_CDCLClause(lits=list(cl)) for cl in enc_clauses]

    # watches[lit] = list of clause indices watching this literal
    # Only clauses with >= 2 literals are watched.  Unit clauses are
    # enqueued onto the trail at level 0 before propagation starts.
    watches: dict[int, list[int]] = {}
    unit_seed: list[tuple[int, int]] = []  # (literal, source clause index)

    def _add_watch(lit: int, cidx: int) -> None:
        watches.setdefault(lit, []).append(cidx)

    for ci, cl in enumerate(clause_list):
        if len(cl.lits) == 1:
            unit_seed.append((cl.lits[0], ci))
        else:
            _add_watch(cl.lits[0], ci)
            _add_watch(cl.lits[1], ci)

    # Trail of (var, value, level, reason_clause_idx).
    assign: dict[int, bool] = {}
    level_of: dict[int, int] = {}
    reason_of: dict[int, int | None] = {}
    trail: list[int] = []
    trail_lim: list[int] = []
    decisions = 0
    conflicts = 0
    propagations = 0
    restarts = 0
    learned = 0

    # VSIDS activity.
    activity = [0.0] * (nvars + 2)
    bump_inc = _VSIDS_BUMP

    def _bump(v: int) -> None:
        nonlocal bump_inc
        activity[v] += bump_inc
        if activity[v] > _VSIDS_RESCALE:
            for i in range(len(activity)):
                activity[i] *= 1e-100
            bump_inc *= 1e-100

    def _decay() -> None:
        nonlocal bump_inc
        bump_inc /= _VSIDS_DECAY

    def _value(lit: int) -> int:
        v = _lit_var(lit)
        if v not in assign:
            return 0
        # +1 if literal true, -1 if false.
        is_neg = _lit_neg(lit)
        truth = assign[v]
        return 1 if truth != is_neg else -1

    def _enqueue(lit: int, reason: int | None) -> bool:
        v = _lit_var(lit)
        val = not _lit_neg(lit)
        if v in assign:
            return assign[v] == val
        assign[v] = val
        level_of[v] = len(trail_lim)        # current decision level
        reason_of[v] = reason
        trail.append(lit)
        return True

    def _propagate() -> int | None:
        """Returns conflict clause index or None on success."""
        nonlocal propagations
        qhead = 0
        # Trail grows; iterate from current head.
        while qhead < len(trail):
            lit = trail[qhead]
            qhead += 1
            # Watched-literal scheme: literals watching the negation
            # of this literal need to be checked.
            falsified = -lit
            ws = watches.get(falsified, [])
            new_ws: list[int] = []
            i = 0
            stopped = False
            while i < len(ws):
                ci = ws[i]
                cl = clause_list[ci]
                lits = cl.lits
                # Ensure watched lits are at positions 0 and 1; the
                # other-watched is at position 1-(this).
                if lits[0] == falsified:
                    lits[0], lits[1] = lits[1], lits[0]
                # Now lits[1] == falsified, lits[0] is the other watcher.
                if _value(lits[0]) > 0:
                    new_ws.append(ci)
                    i += 1
                    continue
                # Try to find a new watch.
                found = False
                for k in range(2, len(lits)):
                    if _value(lits[k]) >= 0:
                        # Move to watch this lit.
                        lits[1], lits[k] = lits[k], lits[1]
                        _add_watch(lits[1], ci)
                        found = True
                        break
                if found:
                    i += 1
                    continue
                # No replacement watcher — the clause is unit or conflict.
                new_ws.append(ci)
                if _value(lits[0]) < 0:
                    # Conflict — leave remaining ws intact.
                    new_ws.extend(ws[i + 1:])
                    watches[falsified] = new_ws
                    return ci
                # lits[0] is unassigned — enqueue.
                _enqueue(lits[0], ci)
                propagations += 1
                i += 1
            watches[falsified] = new_ws
        return None

    def _analyze(conflict_ci: int) -> tuple[list[int], int]:
        """1UIP conflict analysis; returns (learned clause, backtrack level)."""
        seen = [False] * (nvars + 2)
        learnt: list[int] = []
        counter = 0
        p = 0
        # Start at the conflict clause; walk back the trail.
        cl = clause_list[conflict_ci]
        bt_level = 0
        # Mark variables in conflict.
        for lit in cl.lits:
            v = _lit_var(lit)
            if level_of.get(v, 0) > 0 and not seen[v]:
                seen[v] = True
                _bump(v)
                if level_of[v] >= len(trail_lim):
                    counter += 1
                else:
                    learnt.append(lit)
                    if level_of[v] > bt_level:
                        bt_level = level_of[v]
        # Walk trail backwards.
        ti = len(trail) - 1
        while True:
            while ti >= 0 and not seen[_lit_var(trail[ti])]:
                ti -= 1
            if ti < 0:
                break
            p_lit = trail[ti]
            p = _lit_var(p_lit)
            counter -= 1
            if counter == 0:
                # UIP reached; learnt clause = ¬p ∨ (others).
                learnt.append(-p_lit)
                break
            seen[p] = False
            reason = reason_of.get(p)
            if reason is None:
                break
            rcl = clause_list[reason]
            for lit in rcl.lits:
                v = _lit_var(lit)
                if v == p:
                    continue
                if seen[v]:
                    continue
                if level_of.get(v, 0) > 0:
                    seen[v] = True
                    _bump(v)
                    if level_of[v] >= len(trail_lim):
                        counter += 1
                    else:
                        learnt.append(lit)
                        if level_of[v] > bt_level:
                            bt_level = level_of[v]
            ti -= 1
        return learnt, bt_level

    def _undo_to(level: int) -> None:
        # Pop the trail down to ``level`` decisions.
        while len(trail_lim) > level:
            top = trail_lim.pop()
            while len(trail) > top:
                lit = trail.pop()
                v = _lit_var(lit)
                assign.pop(v, None)
                level_of.pop(v, None)
                reason_of.pop(v, None)

    # Seed unit clauses at level 0.
    for (lit, ci) in unit_seed:
        v = _lit_var(lit)
        want = not _lit_neg(lit)
        if v in assign:
            if assign[v] != want:
                return Solution(verdict=UNSAT, algorithm=CDCL,
                                decisions=decisions, conflicts=1,
                                propagations=propagations, restarts=restarts,
                                learned=learned,
                                elapsed_s=time.perf_counter() - t0)
        else:
            _enqueue(lit, ci)
    # Initial propagation at level 0.
    if _propagate() is not None:
        return Solution(verdict=UNSAT, algorithm=CDCL,
                        decisions=decisions, conflicts=1,
                        propagations=propagations, restarts=restarts,
                        learned=learned, elapsed_s=time.perf_counter() - t0)
    restart_unit = 100
    luby_i = 0
    conflicts_since_restart = 0
    while True:
        if conflicts >= max_conflicts:
            return Solution(verdict=UNKNOWN, algorithm=CDCL,
                            decisions=decisions, conflicts=conflicts,
                            propagations=propagations, restarts=restarts,
                            learned=learned,
                            elapsed_s=time.perf_counter() - t0)
        if timeout_s is not None and (time.perf_counter() - t0) > timeout_s:
            return Solution(verdict=UNKNOWN, algorithm=CDCL,
                            decisions=decisions, conflicts=conflicts,
                            propagations=propagations, restarts=restarts,
                            learned=learned,
                            elapsed_s=time.perf_counter() - t0)
        # All assigned → SAT.
        if len(assign) >= nvars:
            model = {a: assign.get(atom_idx[a], False) for a in atom_names}
            return Solution(
                verdict=SAT, model=model, decisions=decisions,
                conflicts=conflicts, propagations=propagations,
                restarts=restarts, learned=learned,
                algorithm=CDCL, elapsed_s=time.perf_counter() - t0,
            )
        # Pick next decision via VSIDS.
        best_v = -1
        best_score = -1.0
        for v in range(1, nvars + 1):
            if v in assign:
                continue
            if activity[v] > best_score:
                best_score = activity[v]
                best_v = v
        if best_v < 0:
            # All bound — should be SAT, but defensive.
            model = {a: assign.get(atom_idx[a], False) for a in atom_names}
            return Solution(verdict=SAT, model=model, decisions=decisions,
                            conflicts=conflicts, propagations=propagations,
                            restarts=restarts, learned=learned,
                            algorithm=CDCL,
                            elapsed_s=time.perf_counter() - t0)
        # Polarity: bias by recent — flip a coin biased by phase.
        polarity = bool(rng.random() < 0.5)
        decisions += 1
        trail_lim.append(len(trail))
        _enqueue(best_v if polarity else -best_v, None)
        # Propagate.
        while True:
            confl = _propagate()
            if confl is None:
                break
            conflicts += 1
            conflicts_since_restart += 1
            _decay()
            if len(trail_lim) == 0:
                # Top-level conflict — UNSAT.
                return Solution(verdict=UNSAT, algorithm=CDCL,
                                decisions=decisions, conflicts=conflicts,
                                propagations=propagations,
                                restarts=restarts, learned=learned,
                                elapsed_s=time.perf_counter() - t0)
            learnt, bt_level = _analyze(confl)
            # Avoid degenerate empty learnt: that means a top-level conflict.
            if not learnt:
                return Solution(verdict=UNSAT, algorithm=CDCL,
                                decisions=decisions, conflicts=conflicts,
                                propagations=propagations,
                                restarts=restarts, learned=learned,
                                elapsed_s=time.perf_counter() - t0)
            # Dedup learnt clause.
            seen_lits: dict[int, int] = {}
            dedup_learnt: list[int] = []
            taut = False
            for lit in learnt:
                v = _lit_var(lit)
                if v in seen_lits:
                    if seen_lits[v] != lit:
                        taut = True; break
                    continue
                seen_lits[v] = lit
                dedup_learnt.append(lit)
            if taut:
                # Skip storage; just backtrack one level.
                _undo_to(max(0, len(trail_lim) - 1))
                continue
            new_cl = _CDCLClause(lits=dedup_learnt, learned=True)
            ci_new = len(clause_list)
            clause_list.append(new_cl)
            learned += 1
            # Watches: the asserting literal is `learnt[-1]` per
            # construction (1UIP), and we move the highest-level other-
            # literal to position 1 for correct backjumping.
            if len(dedup_learnt) >= 2:
                _add_watch(dedup_learnt[0], ci_new)
                _add_watch(dedup_learnt[-1], ci_new)
                # Place the asserting literal at index 1, the other watch at 0.
                # Find a literal with highest level other than asserting.
                asserting = dedup_learnt[-1]
                rest_levels = [
                    (i, level_of.get(_lit_var(lit), 0))
                    for i, lit in enumerate(dedup_learnt)
                    if lit != asserting
                ]
                if rest_levels:
                    rest_levels.sort(key=lambda x: -x[1])
                    swap_to_zero = rest_levels[0][0]
                    # Make pos 0 the high-level lit, pos 1 the asserting.
                    lits = new_cl.lits
                    lits[0], lits[swap_to_zero] = lits[swap_to_zero], lits[0]
                    # Find asserting index after swap.
                    for i, lit in enumerate(lits):
                        if lit == asserting:
                            lits[1], lits[i] = lits[i], lits[1]
                            break
            else:
                _add_watch(dedup_learnt[0], ci_new)
            _undo_to(bt_level)
            # The asserting literal should now be unit at the backtrack level.
            _enqueue(new_cl.lits[1] if len(new_cl.lits) >= 2 else new_cl.lits[0], ci_new)
            # Luby restart.
            target = restart_unit * _luby(luby_i)
            if conflicts_since_restart >= target:
                restarts += 1
                luby_i += 1
                conflicts_since_restart = 0
                _undo_to(0)
                break


# =====================================================================
# Walk-SAT — Selman-Kautz-Cohen 1994 with Schöning's 1999 noise schedule
# =====================================================================


def _walksat_solve(
    clauses: list[tuple[tuple[str, bool], ...]],
    atom_names: list[str],
    max_flips: int,
    max_restarts: int,
    noise: float,
    rng: random.Random,
    timeout_s: float | None = None,
) -> Solution:
    t0 = time.perf_counter()
    if not clauses:
        return Solution(verdict=SAT, model={a: False for a in atom_names},
                        algorithm=WALKSAT,
                        elapsed_s=time.perf_counter() - t0)
    atom_idx: dict[str, int] = {a: i + 1 for i, a in enumerate(atom_names)}
    next_idx = len(atom_idx) + 1
    enc: list[list[int]] = []
    for cl in clauses:
        encl: list[int] = []
        for a, neg in cl:
            if a not in atom_idx:
                atom_idx[a] = next_idx
                next_idx += 1
            encl.append(-atom_idx[a] if neg else atom_idx[a])
        enc.append(encl)
    nvars = next_idx - 1
    flips_total = 0
    for restart in range(max_restarts):
        # Random initial assignment.
        assign = {v: rng.random() < 0.5 for v in range(1, nvars + 1)}
        for _ in range(max_flips):
            flips_total += 1
            if timeout_s is not None and (time.perf_counter() - t0) > timeout_s:
                return Solution(verdict=UNKNOWN, algorithm=WALKSAT,
                                propagations=flips_total,
                                restarts=restart,
                                elapsed_s=time.perf_counter() - t0)
            # Find an unsatisfied clause.
            unsat_clauses: list[int] = []
            for ci, cl in enumerate(enc):
                if not any(
                    (assign.get(_lit_var(lit), False) != _lit_neg(lit))
                    for lit in cl
                ):
                    unsat_clauses.append(ci)
            if not unsat_clauses:
                model = {a: assign.get(atom_idx[a], False) for a in atom_names}
                return Solution(verdict=SAT, model=model,
                                algorithm=WALKSAT, propagations=flips_total,
                                restarts=restart,
                                elapsed_s=time.perf_counter() - t0)
            target_ci = rng.choice(unsat_clauses)
            target = enc[target_ci]
            if rng.random() < noise:
                # Random flip.
                lit = rng.choice(target)
                v = _lit_var(lit)
                assign[v] = not assign[v]
            else:
                # Greedy: flip the variable that breaks the fewest clauses.
                best_var = None
                best_breaks = float("inf")
                for lit in target:
                    v = _lit_var(lit)
                    breaks = 0
                    # Trial flip.
                    assign[v] = not assign[v]
                    for ci, cl in enumerate(enc):
                        if not any(
                            (assign.get(_lit_var(l), False) != _lit_neg(l))
                            for l in cl
                        ):
                            breaks += 1
                    assign[v] = not assign[v]
                    if breaks < best_breaks:
                        best_breaks = breaks
                        best_var = v
                if best_var is not None:
                    assign[best_var] = not assign[best_var]
    return Solution(verdict=UNKNOWN, algorithm=WALKSAT,
                    propagations=flips_total, restarts=max_restarts,
                    elapsed_s=time.perf_counter() - t0)


# =====================================================================
# Resolution proof reconstruction for UNSAT
# =====================================================================


def _resolution_proof(
    clauses: list[tuple[tuple[str, bool], ...]],
) -> list[Resolution]:
    """Construct a (possibly suboptimal) resolution proof of UNSAT.

    Strategy: saturate by unit-propagation-driven resolution.  For each
    unit clause ℓ and clause C containing ¬ℓ we emit a resolution step
    and keep the resolvent.  Terminates either with the empty clause
    (UNSAT confirmed) or runs out of new resolvents (in which case
    we fail silently and return what we have so far — the caller will
    have a CDCL-confirmed UNSAT verdict already).

    Note: This is not a minimum-size proof.  For runtime audit purposes
    we just need *some* replayable proof witness.
    """
    if not clauses:
        return []
    # Work over a set to avoid duplicate clauses.
    work: dict[tuple[tuple[str, bool], ...], int] = {}
    proof: list[Resolution] = []
    for c in clauses:
        if c not in work:
            work[c] = len(work)
    max_steps = max(500, 4 * len(clauses))
    for _ in range(max_steps):
        units = [c for c in work if len(c) == 1]
        progress = False
        for u in list(units):
            atom, neg = u[0]
            target_lit = (atom, not neg)
            for c in list(work.keys()):
                if c == u:
                    continue
                if target_lit in c:
                    # Resolve.
                    resolvent_set = set(c) - {target_lit}
                    resolvent_set.update(set(u) - {(atom, neg)})
                    # Tautology check.
                    by_atom: dict[str, bool] = {}
                    taut = False
                    for a, n in resolvent_set:
                        if a in by_atom and by_atom[a] != n:
                            taut = True
                            break
                        by_atom[a] = n
                    if taut:
                        continue
                    resolvent = tuple(sorted(by_atom.items()))
                    if resolvent not in work:
                        work[resolvent] = len(work)
                        proof.append(Resolution(
                            clause_a=u, clause_b=c,
                            pivot=atom, resolvent=resolvent,
                        ))
                        progress = True
                        if not resolvent:
                            return proof
                        units.append(resolvent) if len(resolvent) == 1 else None
        if not progress:
            break
    return proof
