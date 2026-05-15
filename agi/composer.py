r"""Composer — typed, certified compositional planning as a runtime primitive.

Every other primitive in this runtime answers one question.  ``Bandit``
trades off explore-vs-exploit on a single decision.  ``BayesOpt`` finds
the optimum of a single black-box.  ``Sampler`` draws from a posterior.
``Reasoner`` solves a single SAT / Horn / ASP instance.  ``Refuter``
falsifies a single claim.  ``Causal`` estimates one ATE.  ``Synthesizer``
fills in one program from examples.  ``Ranker`` rates a fixed K of
items.  Each is, by design, a thin, well-bounded specialist.

The coordination engine running on top of this runtime, by contrast,
solves **plans** — *sequences* (or DAGs) of primitive calls in which
each call's output feeds the next call's input, the *types* line up
(rates flow into bandit, rankings flow into experiment design,
forecasts flow into portfolio allocation), the *budgets* compose
(privacy ε adds, statistical α composes by union bound, monetary cost
sums), and the *failure modes* compose into a single end-to-end
reliability we can certify and bill against.

``Composer`` is the primitive that performs that composition.  It is
the runtime's **planning** primitive: take a typed registry of
operators (each operator being any function the coordinator can call —
typically another runtime primitive — annotated with its input types,
output types, preconditions, postconditions, cost, and a Bayesian
prior over its per-call success probability), take a ``Goal`` (an
initial state + a conjunctive postcondition the plan must satisfy),
and synthesise a **certified plan**: an ordered list of operator
applications, a typed dataflow witness, and a PAC lower bound on the
plan's end-to-end success probability.

The pitch reduced to a runtime call:

  * ``register_operator(name, params, pre, add, delete, cost,
    reliability)`` for every callable in the coordinator's toolkit
    (Bandit.recommend, Reasoner.solve, Refuter.refute, an external
    HTTP call, an LLM tool, …);
  * call ``synthesize(initial_state, goal_postcondition, budget=…)`` to
    get a ``Plan`` and a ``Certificate`` carrying:

      - the ordered sequence of ``(op, bindings)`` calls,
      - the Hindley-Milner type substitution that witnesses dataflow,
      - a PAC lower bound on end-to-end reliability with confidence
        ``1 − α`` (Clopper-Pearson per-operator, composed by product
        for independent operators or by union-bound for the worst-case
        dependent regime),
      - a tamper-evident SHA-256 fingerprint chaining every plan
        decision the planner made;
  * call ``execute(plan, executor)`` to run the plan against a caller-
    supplied ``executor(op_name, **bindings) -> (success, output)``;
  * call ``observe(op_name, success)`` after every primitive call —
    inside an execution or outside — to update the Beta-Bernoulli
    posterior on that operator's reliability;
  * call ``report()`` for a ``ComposerReport`` carrying the operator
    posteriors, the empirical PAC lower bounds, the plan-quality
    statistics, the strongly-connected-components of the operator
    dataflow graph (a planning-failure diagnostic), and the
    fingerprint.

Mathematical roots and algorithms shipped
-----------------------------------------

The planning core is classical STRIPS / ADL with conjunctive
preconditions and add/delete-list effects (Fikes-Nilsson 1971,
Pednault 1989).  The search is A* with consistent admissible
heuristics over a state-space graph whose nodes are sets of grounded
atoms; the reliability calculus is Bayesian (Beta-Bernoulli) with
Clopper-Pearson and PAC-Bayes confidence; the typing discipline is
monomorphic Hindley-Milner with first-order unification.

**Planning.**

  * **A\* search** (Hart-Nilsson-Raphael 1968).  Best-first search of
    the state space ordered by ``f(s) = g(s) + h(s)`` where ``g`` is
    the accumulated negative-log reliability (so additive product of
    reliabilities) plus monetary cost, and ``h`` is an admissible
    underestimate of the remaining negative-log reliability.  Two
    heuristics ship: ``h_zero`` (always 0; reduces A* to Dijkstra) and
    ``h_landmark`` (admissible HSP-style landmark heuristic in the
    sense of Helmert-Domshlak 2009: count the unachieved goal atoms
    weighted by the *cheapest* operator that adds each atom).

  * **IDA\* search** (Korf 1985).  Iterative-deepening A* with the
    same heuristic.  Constant memory; preferred for very deep plans
    (depth ≫ 10⁴) where A*'s open list would blow up.  Returns the
    same optimal plan as A* under a consistent heuristic.

  * **STRIPS regression** (Fikes-Nilsson 1971; Bonet-Geffner 2001).
    Backward planning — start from the goal, regress through operators
    that *delete-add* a goal atom, search to the initial state.  Useful
    when the operator set is dense and the goal is small.

  * **Topological sort with Tarjan SCC** (Tarjan 1972).  Every plan is
    a totally ordered list of steps; the underlying *dataflow DAG*
    (operator A produced an object that operator B consumed) is
    extracted and topologically sorted for the executor.  Tarjan SCC
    on the operator-dependency graph diagnoses planning failure
    (cycles indicate ill-typed registration).

**Type system.**

  * **Hindley-Milner unification** (Robinson 1965, Milner 1978).  Each
    operator's parameter types are first-order monomorphic type terms
    (e.g. ``"Hypothesis"``, ``"Distribution<Float>"``,
    ``"Posterior<a>"`` with type variable ``a``).  The planner solves
    a sequence of unification problems; failure yields an
    ``UnificationError`` that lists the conflicting positions.  Occurs-
    check is enabled.  No higher-rank polymorphism.

  * **Subtype lattice** (optional).  Operators can declare ``a <: b``
    facts; the unifier checks subtyping with depth-first transitive
    closure.  A type variable unifies upward to its least supertype.

**Reliability calculus.**

  * **Beta-Bernoulli posterior** (Bayes 1763 / Laplace 1814).  Every
    operator maintains a posterior ``Beta(α + s, β + f)`` over its
    success probability, updated by ``observe(name, success)``.  The
    default prior is ``Beta(1, 1)`` (Laplace's rule of succession);
    the runtime exposes ``Beta(α₀, β₀)`` per-operator at register
    time so an operator with strong prior knowledge ("Reasoner is
    sound on SAT instances within its budget — reliability 0.99 with
    prior strength 100") can be wired in without burning observations.

  * **Clopper-Pearson exact intervals** (Clopper-Pearson 1934).  The
    finite-sample (1 − α) lower bound on each operator's reliability
    used in the certificate.  ``CP_lo(k, n, α)`` is the largest p such
    that ``P(X ≥ k | n, p) ≤ α/2``; closed form for k = n returns
    ``α^{1/n}``, bisection of the regularised incomplete beta otherwise.
    Exact under the Binomial model — not asymptotic.

  * **Empirical Bernstein** (Maurer-Pontil 2009).  Tighter than
    Hoeffding when the empirical variance is small; the certificate
    includes both, the coordinator picks the tighter one.

  * **PAC-Bayes Catoni** (Catoni 2007).  A bound on the *average*
    reliability of a *posterior over operators* (i.e. on a randomised
    operator-choice policy) of the form

      ``E_{Q}[reliability] ≥ 1 − (KL(Q‖P) + log(2√n / δ)) / n``.

    Used when the planner has multiple alternatives at a single
    step (e.g. ``Bandit.recommend_thompson`` vs
    ``Bandit.recommend_ucb``) and the executor draws one stochastically.

  * **Composition theorems for end-to-end reliability.**

    - *Independent product.*  If operators ``o₁, …, o_n`` along a
      plan path are independent, ``P(plan succeeds) = ∏ p_i``.  The
      certificate's PAC lower bound replaces each ``p_i`` with its
      Clopper-Pearson lower bound and re-multiplies; the result is a
      valid (1 − α) lower bound on the *product* (because the
      product of lower bounds underestimates the product of true
      probabilities; the per-step α budget is set to ``α / n`` by
      Bonferroni).

    - *Worst-case dependent union bound.*  ``P(plan fails) ≤ ∑ P(o_i
      fails)``.  Always valid, never tight; used when the executor
      cannot guarantee operator-call independence (shared state,
      shared random seed).

    The coordinator picks which regime applies via the
    ``independence`` argument at synthesis time.

**Tamper-evident replay.**

Every register / synthesize / observe / report call emits an event
into a SHA-256 hash chain (genesis hash ``composer.v1.genesis``).
Replaying the same byte sequence reproduces the same fingerprint.  The
fingerprint is hash-friendly for direct embedding in
``AttestationLedger``: the chain is collision-resistant and any
divergence in operator registry, observation sequence, or plan
synthesis is detectable.

What it composes with
---------------------

``Composer`` is built to be driven by — and to drive — every other
primitive in the runtime:

  * **Reasoner.**  Register ``Reasoner.solve`` as an operator with
    pre = "SAT-instance available" and post = "model or unsat-proof
    derived".  Composer's planner then automatically routes through
    Reasoner when the goal includes a SAT discharge step.

  * **Refuter.**  Register ``Refuter.refute`` as an operator whose
    post is "claim disproven or PAC-certified within ε".  Composer
    schedules it as a *gate* before any operator that consumes the
    claim.

  * **Bandit / BayesOpt / Arbiter.**  Each registered with its
    Bayesian-decision-theoretic surface; Composer's reliability
    update is fed by the bandit's per-pull observations.

  * **Synthesizer.**  Composer is *not* Synthesizer — Synthesizer
    fabricates a *program* from input/output examples, Composer
    fabricates a *plan* from a *typed library of operators*.  But
    Composer can use Synthesizer as one of its registered operators
    (post-condition: "program for relation X exists"), and
    Synthesizer can use Composer as a sub-planner over its DSL.

  * **PrivacyAccountant.**  The plan's privacy cost composes by
    advanced composition over the per-operator ε contributions;
    Composer surfaces the composed privacy budget alongside the
    reliability bound.

  * **AttestationLedger.**  The plan fingerprint, certificate, and
    every operator observation are append-only and hash-chained, so
    an external auditor can reconstruct the planning decision and
    the empirical-reliability evidence that supported it.

Investor framing
----------------

The pitch a coordinator's UI can surface, automatically, for every
high-level goal the user submits:

    "Achieving goal *G* requires the following plan:
        1. ``Reasoner.solve(...)``  — proven reliability ≥ 0.998
        2. ``BayesOpt.suggest(...)`` — Clopper-Pearson lower bound 0.94
        3. ``Forecaster.update(...)`` — observed 1.00 on 24 calls
     End-to-end reliability ≥ 0.92 with 95 % confidence.
     Expected monetary cost: $0.087.
     Expected wall-clock: 12.4 s.
     Replay fingerprint: 4a8c2f… (verifiable via AttestationLedger)."

Every claim here is grounded in published, citable mathematics; every
number is reproducible from the operator observation log.
"""
from __future__ import annotations

import bisect
import hashlib
import heapq
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


# =====================================================================
# Public constants
# =====================================================================

# Search algorithms.
ASTAR = "astar"
IDA_STAR = "ida_star"
DIJKSTRA = "dijkstra"
REGRESSION = "regression"

# Heuristics.
H_ZERO = "zero"
H_LANDMARK = "landmark"
H_GOAL_COUNT = "goal_count"

# Composition regimes for end-to-end reliability.
INDEPENDENT = "independent"
WORST_CASE = "worst_case"

# Verdicts.
SOLVED = "solved"
INFEASIBLE = "infeasible"
BUDGET_EXHAUSTED = "budget_exhausted"
ILL_TYPED = "ill_typed"

KNOWN_ALGORITHMS = frozenset({ASTAR, IDA_STAR, DIJKSTRA, REGRESSION})
KNOWN_HEURISTICS = frozenset({H_ZERO, H_LANDMARK, H_GOAL_COUNT})
KNOWN_REGIMES = frozenset({INDEPENDENT, WORST_CASE})
KNOWN_VERDICTS = frozenset({SOLVED, INFEASIBLE, BUDGET_EXHAUSTED, ILL_TYPED})

# Events.
COMPOSER_STARTED = "composer.started"
COMPOSER_OPERATOR_REGISTERED = "composer.operator_registered"
COMPOSER_AXIOM_ADDED = "composer.axiom_added"
COMPOSER_PLANNED = "composer.planned"
COMPOSER_VERIFIED = "composer.verified"
COMPOSER_EXECUTED = "composer.executed"
COMPOSER_STEP_OK = "composer.step_ok"
COMPOSER_STEP_FAILED = "composer.step_failed"
COMPOSER_OBSERVED = "composer.observed"
COMPOSER_REPORT = "composer.report"
COMPOSER_CLEARED = "composer.cleared"

KNOWN_EVENTS = frozenset({
    COMPOSER_STARTED,
    COMPOSER_OPERATOR_REGISTERED,
    COMPOSER_AXIOM_ADDED,
    COMPOSER_PLANNED,
    COMPOSER_VERIFIED,
    COMPOSER_EXECUTED,
    COMPOSER_STEP_OK,
    COMPOSER_STEP_FAILED,
    COMPOSER_OBSERVED,
    COMPOSER_REPORT,
    COMPOSER_CLEARED,
})

# Numerical defaults.
_EPS = 1e-12
_INF = float("inf")
_LOG_EPS = math.log(_EPS)
_DEFAULT_PRIOR_ALPHA = 1.0
_DEFAULT_PRIOR_BETA = 1.0
_DEFAULT_ALPHA = 0.05
_DEFAULT_MAX_EXPANSIONS = 200_000
_DEFAULT_MAX_DEPTH = 64
_GENESIS = hashlib.sha256(b"composer.v1.genesis").hexdigest()


# =====================================================================
# Exceptions
# =====================================================================


class ComposerError(ValueError):
    """Base class for composer-domain errors."""


class UnknownAlgorithm(ComposerError):
    """Algorithm name is not in KNOWN_ALGORITHMS."""


class UnknownHeuristic(ComposerError):
    """Heuristic name is not in KNOWN_HEURISTICS."""


class UnknownRegime(ComposerError):
    """Composition regime is not in KNOWN_REGIMES."""


class UnificationError(ComposerError):
    """Two types cannot be unified."""


class TypeError_(ComposerError):
    """An operator's type schema is malformed."""


class InvalidPredicate(ComposerError):
    """A predicate / atom / effect is malformed."""


class InvalidOperator(ComposerError):
    """An operator definition is internally inconsistent."""


class UnknownOperator(ComposerError):
    """An operator name was used that was never registered."""


class InvalidGoal(ComposerError):
    """A goal's pre/post is malformed."""


class PlanningFailure(ComposerError):
    """The planner exhausted its budget or proved the goal infeasible."""


class ExecutionFailure(ComposerError):
    """An operator returned failure during plan execution."""


# =====================================================================
# Numerical helpers
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_log(x: float) -> float:
    return math.log(max(x, _EPS))


def hoeffding_lower(k: int, n: int, alpha: float) -> float:
    r"""Hoeffding (1963) lower bound on a binomial success probability.

    Returns the largest ``p`` such that ``P(X ≤ k | n, p) ≥ α/2``,
    which simplifies to ``p_hat - sqrt(log(2/α) / (2n))`` clipped to
    [0, 1].  Always valid, asymptotically loose.
    """
    if n <= 0:
        return 0.0
    p_hat = k / n
    half = math.sqrt(math.log(2.0 / max(alpha, _EPS)) / (2.0 * n))
    return _clip(p_hat - half, 0.0, 1.0)


def hoeffding_upper(k: int, n: int, alpha: float) -> float:
    """Hoeffding upper bound on a binomial success probability."""
    if n <= 0:
        return 1.0
    p_hat = k / n
    half = math.sqrt(math.log(2.0 / max(alpha, _EPS)) / (2.0 * n))
    return _clip(p_hat + half, 0.0, 1.0)


def empirical_bernstein_lower(
    k: int, n: int, alpha: float
) -> float:
    """Maurer-Pontil (2009) empirical-Bernstein lower bound for Bernoulli.

    The variance of a Bernoulli with empirical mean ``p_hat`` is
    bounded by ``p_hat * (1 - p_hat)``; substituting gives a closed
    form that is always finite for ``n >= 2``.
    """
    if n <= 1:
        return 0.0
    p_hat = k / n
    var_hat = p_hat * (1.0 - p_hat)
    log_term = math.log(3.0 / max(alpha, _EPS))
    half = math.sqrt(2.0 * var_hat * log_term / n) + 3.0 * log_term / (n - 1)
    return _clip(p_hat - half, 0.0, 1.0)


def _log_gamma(x: float) -> float:
    """math.lgamma but safe."""
    return math.lgamma(max(x, _EPS))


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b) — regularised incomplete Beta via continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    """Lentz's algorithm continued-fraction expansion for the incomplete beta."""
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-7:
            break
    return h


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    r"""Clopper-Pearson (1934) lower bound on a binomial probability.

    Returns the largest ``p`` such that ``P(X ≥ k | n, p) ≤ α/2``
    (lower tail).  Closed form for ``k = 0`` returns 0; otherwise we
    invert ``I_p(k, n-k+1) = α/2`` by bisection.
    """
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    if k >= n:
        # all successes: p_lo = (α/2)^{1/n}
        return math.pow(max(alpha / 2.0, _EPS), 1.0 / n)
    a, b = float(k), float(n - k + 1)
    target = alpha / 2.0
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _regularised_incomplete_beta(mid, a, b) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    return _clip(0.5 * (lo + hi), 0.0, 1.0)


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    r"""Clopper-Pearson upper bound on a binomial probability.

    Returns the smallest ``p`` such that ``P(X ≤ k | n, p) ≤ α/2``.
    Closed form for ``k = n`` returns 1; bisection otherwise.
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k <= 0:
        # no successes: p_hi = 1 − (α/2)^{1/n}
        return 1.0 - math.pow(max(alpha / 2.0, _EPS), 1.0 / n)
    a, b = float(k + 1), float(n - k)
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _regularised_incomplete_beta(mid, a, b) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return _clip(0.5 * (lo + hi), 0.0, 1.0)


def pac_bayes_catoni(kl_div: float, n: int, alpha: float) -> float:
    r"""Catoni (2007) PAC-Bayes lower bound on an expected reliability.

    Returns a (1 − α) lower bound on ``E_{Q}[reliability]`` when the
    empirical mean of i.i.d. Bernoulli draws under ``Q`` is achieved
    at the limit ``1 - p_hat = 0`` (we use the standard form
    ``mean_hat - (KL + log(2√n / δ)) / n`` clipped to [0, 1]).
    """
    if n <= 0:
        return 0.0
    penalty = (kl_div + math.log(2.0 * math.sqrt(n) / max(alpha, _EPS))) / n
    return _clip(1.0 - penalty, 0.0, 1.0)


def kl_bernoulli(p: float, q: float) -> float:
    """KL divergence of two Bernoullis ``B(p) || B(q)``."""
    p = _clip(p, _EPS, 1.0 - _EPS)
    q = _clip(q, _EPS, 1.0 - _EPS)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_bernoulli_upper_inverse(p_hat: float, n: int, alpha: float) -> float:
    r"""KL-inverse upper confidence bound for a Bernoulli mean (Garivier-Cappé 2011).

    Returns the largest ``q`` such that ``n · KL(p_hat || q) ≤ log(1/α)``.
    Tighter than Hoeffding and almost matches Clopper-Pearson; cheaper
    than CP at very large ``n``.
    """
    if n <= 0:
        return 1.0
    if p_hat >= 1.0:
        return 1.0
    target = math.log(1.0 / max(alpha, _EPS)) / n
    lo, hi = p_hat, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(p_hat, mid) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10:
            break
    return _clip(0.5 * (lo + hi), 0.0, 1.0)


def kl_bernoulli_lower_inverse(p_hat: float, n: int, alpha: float) -> float:
    """KL-inverse lower confidence bound (mirror of the upper inverse)."""
    if n <= 0:
        return 0.0
    if p_hat <= 0.0:
        return 0.0
    target = math.log(1.0 / max(alpha, _EPS)) / n
    lo, hi = 0.0, p_hat
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(p_hat, mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return _clip(0.5 * (lo + hi), 0.0, 1.0)


# =====================================================================
# Type system — monomorphic Hindley-Milner with first-order unification
# =====================================================================


@dataclass(frozen=True)
class TypeVar:
    """A type variable, written ``?a``, ``?b``, …"""
    name: str

    def __str__(self) -> str:
        return f"?{self.name}"


@dataclass(frozen=True)
class TypeCon:
    r"""A type constructor applied to zero or more argument types.

    ``TypeCon("Int")`` is a base type; ``TypeCon("List", (TypeCon("Int"),))``
    is the parameterised ``List<Int>``.
    """
    name: str
    args: tuple = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}<{', '.join(str(a) for a in self.args)}>"


def parse_type(spec: Any) -> Any:
    r"""Parse a textual or compositional type spec into TypeVar/TypeCon.

    Accepts:
      * ``"Int"`` → ``TypeCon("Int")``
      * ``"List<Int>"`` → ``TypeCon("List", (TypeCon("Int"),))``
      * ``"List<Map<Str, Int>>"`` etc, arbitrary nesting
      * ``"?a"`` → ``TypeVar("a")``
      * A TypeVar / TypeCon directly → identity
      * ``("List", "Int")`` → ``TypeCon("List", (TypeCon("Int"),))``
    """
    if isinstance(spec, (TypeVar, TypeCon)):
        return spec
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            raise TypeError_("empty type spec")
        return _parse_type_str(s)
    if isinstance(spec, tuple) and spec:
        head = spec[0]
        if isinstance(head, str) and head.startswith("?"):
            raise TypeError_("tuple-form types cannot have variable heads")
        args = tuple(parse_type(a) for a in spec[1:])
        return TypeCon(str(head), args)
    raise TypeError_(f"cannot parse type from {spec!r}")


def _parse_type_str(s: str) -> Any:
    s = s.strip()
    if s.startswith("?"):
        rest = s[1:]
        if not rest or not all(c.isalnum() or c == "_" for c in rest):
            raise TypeError_(f"malformed type variable {s!r}")
        return TypeVar(rest)
    # Find a top-level "<"
    depth = 0
    open_idx = -1
    for i, ch in enumerate(s):
        if ch == "<":
            if depth == 0 and open_idx == -1:
                open_idx = i
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth < 0:
                raise TypeError_(f"unmatched '>' in {s!r}")
    if depth != 0:
        raise TypeError_(f"unbalanced angle brackets in {s!r}")
    if open_idx < 0:
        # bare constructor
        if not s or not all(c.isalnum() or c == "_" for c in s):
            raise TypeError_(f"malformed type constructor {s!r}")
        return TypeCon(s)
    if not s.endswith(">"):
        raise TypeError_(f"malformed parameterised type {s!r}")
    head = s[:open_idx].strip()
    inner = s[open_idx + 1 : -1]
    args = _split_top_level_commas(inner)
    return TypeCon(head, tuple(_parse_type_str(a) for a in args))


def _split_top_level_commas(s: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "<":
            depth += 1
            cur.append(ch)
        elif ch == ">":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def free_vars(t: Any) -> set[str]:
    """Free type-variable names in a type."""
    if isinstance(t, TypeVar):
        return {t.name}
    if isinstance(t, TypeCon):
        out: set[str] = set()
        for a in t.args:
            out |= free_vars(a)
        return out
    raise TypeError_(f"not a type: {t!r}")


def apply_subst(t: Any, subst: Mapping[str, Any]) -> Any:
    """Apply a type-variable substitution to ``t``."""
    if isinstance(t, TypeVar):
        if t.name in subst:
            return apply_subst(subst[t.name], subst)
        return t
    if isinstance(t, TypeCon):
        if not t.args:
            return t
        new_args = tuple(apply_subst(a, subst) for a in t.args)
        return TypeCon(t.name, new_args)
    raise TypeError_(f"not a type: {t!r}")


def unify(a: Any, b: Any, subst: dict[str, Any] | None = None) -> dict[str, Any]:
    r"""First-order Hindley-Milner unification (Robinson 1965).

    Returns a substitution ``σ`` such that ``σ(a) == σ(b)``.  Raises
    ``UnificationError`` if no such substitution exists (mismatched
    constructor heads or an occurs-check failure).
    """
    s: dict[str, Any] = dict(subst or {})
    a = apply_subst(a, s)
    b = apply_subst(b, s)
    if a == b:
        return s
    if isinstance(a, TypeVar):
        if a.name in free_vars(b):
            raise UnificationError(f"occurs check: {a} in {b}")
        s[a.name] = b
        return s
    if isinstance(b, TypeVar):
        if b.name in free_vars(a):
            raise UnificationError(f"occurs check: {b} in {a}")
        s[b.name] = a
        return s
    if isinstance(a, TypeCon) and isinstance(b, TypeCon):
        if a.name != b.name or len(a.args) != len(b.args):
            raise UnificationError(f"cannot unify {a} with {b}")
        for x, y in zip(a.args, b.args):
            s = unify(x, y, s)
        return s
    raise UnificationError(f"cannot unify {a!r} with {b!r}")


def fresh_renaming(t: Any, counter: list[int]) -> Any:
    """Rename every type variable in ``t`` to a fresh name."""
    mapping: dict[str, Any] = {}
    return _rename(t, mapping, counter)


def _rename(t: Any, mapping: dict[str, Any], counter: list[int]) -> Any:
    if isinstance(t, TypeVar):
        if t.name not in mapping:
            counter[0] += 1
            mapping[t.name] = TypeVar(f"g{counter[0]}")
        return mapping[t.name]
    if isinstance(t, TypeCon):
        return TypeCon(t.name, tuple(_rename(a, mapping, counter) for a in t.args))
    raise TypeError_(f"not a type: {t!r}")


# =====================================================================
# Predicates, states, and operators
# =====================================================================


def _is_variable(token: str) -> bool:
    """A predicate argument is a variable iff it starts with '?'."""
    return isinstance(token, str) and token.startswith("?") and len(token) > 1


@dataclass(frozen=True)
class Predicate:
    r"""An atomic predicate over typed arguments.

    A predicate is a tuple ``(name, args)``.  Each argument is either a
    *constant* (a plain Python value — string, int, float, tuple of
    same) or a *variable* (a string starting with ``?``).  The same
    variable name in two argument positions binds them together.
    """
    name: str
    args: tuple = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(_arg_repr(a) for a in self.args)})"

    @property
    def is_ground(self) -> bool:
        return not any(_is_variable(a) for a in self.args)

    def variables(self) -> list[str]:
        return [a[1:] for a in self.args if _is_variable(a)]

    def substitute(self, bindings: Mapping[str, Any]) -> "Predicate":
        new_args = tuple(
            bindings[a[1:]] if _is_variable(a) and a[1:] in bindings else a
            for a in self.args
        )
        return Predicate(self.name, new_args)


def _arg_repr(a: Any) -> str:
    if _is_variable(a):
        return str(a)
    if isinstance(a, str):
        return repr(a)
    return repr(a)


def parse_predicate(spec: Any) -> Predicate:
    r"""Parse a predicate from a string or tuple.

    Accepts:
      * ``"on(a, b)"`` → ``Predicate("on", ("a", "b"))``
      * ``"on(?x, b)"`` → ``Predicate("on", ("?x", "b"))``
      * ``("on", "a", "b")`` → identity-ish
      * A ``Predicate`` → identity
    """
    if isinstance(spec, Predicate):
        return spec
    if isinstance(spec, str):
        s = spec.strip()
        if "(" not in s:
            return Predicate(s, ())
        if not s.endswith(")"):
            raise InvalidPredicate(f"malformed predicate {s!r}")
        head, body = s.split("(", 1)
        body = body[:-1]
        parts = [p.strip() for p in body.split(",") if p.strip()]
        args = tuple(_coerce_arg(p) for p in parts)
        return Predicate(head.strip(), args)
    if isinstance(spec, (tuple, list)) and spec:
        return Predicate(str(spec[0]), tuple(spec[1:]))
    raise InvalidPredicate(f"cannot parse predicate from {spec!r}")


def _coerce_arg(s: str) -> Any:
    if not s:
        raise InvalidPredicate("empty predicate argument")
    if _is_variable(s):
        return s
    # Try int / float; otherwise treat as a bare-string constant.
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            # strip quotes if present
            if (s.startswith('"') and s.endswith('"')) or (
                s.startswith("'") and s.endswith("'")
            ):
                return s[1:-1]
            return s


def _norm_predicates(preds: Iterable[Any]) -> tuple[Predicate, ...]:
    return tuple(parse_predicate(p) for p in preds)


def _state_freeze(atoms: Iterable[Predicate]) -> frozenset[Predicate]:
    return frozenset(atoms)


@dataclass(frozen=True)
class Operator:
    r"""A typed STRIPS operator (Fikes-Nilsson 1971).

    Fields:

      * ``name`` — unique identifier.
      * ``params`` — ordered ``(var_name, type)`` pairs.  ``var_name``
        is the bare name (no ``?``).  Used both for typed dataflow and
        for variable references in pre/add/delete.
      * ``pre`` — list of ``Predicate`` that must hold (after binding
        the parameters) before the operator is applicable.
      * ``add`` — list of ``Predicate`` added to the state on success.
      * ``delete`` — list of ``Predicate`` deleted from the state on
        success.
      * ``cost`` — additive monetary / latency cost the planner
        accumulates in ``g(s)``.
      * ``alpha`` / ``beta`` — Beta-Bernoulli posterior on the
        operator's reliability.  ``alpha`` = prior_alpha + observed
        successes; ``beta`` = prior_beta + observed failures.
      * ``prior_alpha`` / ``prior_beta`` — the prior set at register
        time; subtracting these from ``alpha`` / ``beta`` recovers
        the *actual* observed counts (vs prior-induced effective n).
      * ``meta`` — opaque payload (e.g. a Python callable, an LLM
        prompt, an HTTP endpoint).
    """
    name: str
    params: tuple = ()
    pre: tuple = ()
    add: tuple = ()
    delete: tuple = ()
    cost: float = 0.0
    alpha: float = _DEFAULT_PRIOR_ALPHA
    beta: float = _DEFAULT_PRIOR_BETA
    prior_alpha: float = _DEFAULT_PRIOR_ALPHA
    prior_beta: float = _DEFAULT_PRIOR_BETA
    meta: Any = None

    def parameter_names(self) -> list[str]:
        return [p[0] for p in self.params]

    def parameter_types(self) -> list[Any]:
        return [p[1] for p in self.params]

    def variables(self) -> set[str]:
        out: set[str] = set()
        for preds in (self.pre, self.add, self.delete):
            for p in preds:
                out.update(p.variables())
        return out

    def reliability_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def reliability_observed_n(self) -> int:
        return int(round((self.alpha - self.prior_alpha) + (self.beta - self.prior_beta)))

    def reliability_observed_k(self) -> int:
        return int(round(self.alpha - self.prior_alpha))


def _make_operator(
    name: str,
    *,
    params: Iterable[Any] = (),
    pre: Iterable[Any] = (),
    add: Iterable[Any] = (),
    delete: Iterable[Any] = (),
    cost: float = 0.0,
    alpha: float = _DEFAULT_PRIOR_ALPHA,
    beta: float = _DEFAULT_PRIOR_BETA,
    meta: Any = None,
) -> Operator:
    if not name or not isinstance(name, str):
        raise InvalidOperator("operator name must be a non-empty string")
    norm_params: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for p in params:
        if isinstance(p, str):
            # parse as "name:Type"
            if ":" not in p:
                raise InvalidOperator(f"parameter spec {p!r} missing ':type'")
            n, t = p.split(":", 1)
            n = n.strip()
            t = parse_type(t.strip())
        elif isinstance(p, (tuple, list)) and len(p) == 2:
            n, t = p[0], parse_type(p[1])
        else:
            raise InvalidOperator(f"cannot parse parameter {p!r}")
        if not n or n.startswith("?"):
            raise InvalidOperator(f"parameter name {n!r} must be bare (no '?')")
        if n in seen:
            raise InvalidOperator(f"duplicate parameter {n!r} in operator {name!r}")
        seen.add(n)
        norm_params.append((n, t))
    pre_n = _norm_predicates(pre)
    add_n = _norm_predicates(add)
    del_n = _norm_predicates(delete)
    cost = float(cost)
    if cost < 0:
        raise InvalidOperator(f"operator {name!r} cost {cost} is negative")
    if alpha <= 0 or beta <= 0:
        raise InvalidOperator(f"Beta prior must be positive: ({alpha}, {beta})")
    op = Operator(
        name=name,
        params=tuple(norm_params),
        pre=pre_n,
        add=add_n,
        delete=del_n,
        cost=cost,
        alpha=float(alpha),
        beta=float(beta),
        prior_alpha=float(alpha),
        prior_beta=float(beta),
        meta=meta,
    )
    # Sanity: every variable in pre/add/delete must be a registered parameter.
    declared = {n for n, _ in norm_params}
    used = op.variables()
    unknown = used - declared
    if unknown:
        raise InvalidOperator(
            f"operator {name!r} references undeclared variables: {sorted(unknown)}"
        )
    return op


# =====================================================================
# Goals, plans, certificates
# =====================================================================


@dataclass(frozen=True)
class Goal:
    r"""A planning goal — initial state + conjunctive postcondition.

    ``initial`` is a set of ground ``Predicate`` (no variables).
    ``post`` is a list of (possibly variable-containing) predicates
    that must all hold in the final state; the planner reports the
    binding of any goal variables.
    """
    initial: frozenset
    post: tuple
    name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlanStep:
    r"""One concrete operator application in a plan.

    ``op_name`` is the registered operator name; ``bindings`` is a
    map from operator-parameter-name to bound value (constant or, in
    pure planning, intermediate symbolic value).  ``produced_atoms``
    and ``consumed_atoms`` carry the dataflow witness used by the
    typed checker.
    """
    op_name: str
    bindings: tuple   # tuple of (param_name, value)
    cost: float = 0.0
    reliability_mean: float = 1.0

    @property
    def binding_map(self) -> dict[str, Any]:
        return dict(self.bindings)


@dataclass(frozen=True)
class Plan:
    r"""A totally ordered sequence of typed operator applications.

    Fields:

      * ``steps`` — ordered ``PlanStep``s.
      * ``initial`` / ``final`` — frozensets of ``Predicate``.
      * ``cost`` — sum of step costs.
      * ``reliability_mean`` — product of per-step posterior means.
      * ``type_subst`` — Hindley-Milner substitution that types the
        dataflow.
      * ``goal_bindings`` — final binding of goal variables.
      * ``fingerprint`` — SHA-256 of the planner trace.
    """
    steps: tuple
    initial: frozenset
    final: frozenset
    cost: float
    reliability_mean: float
    type_subst: dict
    goal_bindings: dict
    fingerprint: str
    verdict: str = SOLVED

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_jsonable(self) -> dict:
        return {
            "verdict": self.verdict,
            "length": self.length,
            "cost": self.cost,
            "reliability_mean": self.reliability_mean,
            "steps": [
                {
                    "op": s.op_name,
                    "bindings": list(s.bindings),
                    "cost": s.cost,
                    "reliability_mean": s.reliability_mean,
                }
                for s in self.steps
            ],
            "initial": [str(p) for p in sorted(self.initial, key=str)],
            "final": [str(p) for p in sorted(self.final, key=str)],
            "goal_bindings": dict(self.goal_bindings),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class Certificate:
    r"""PAC certificate of a plan's end-to-end reliability.

    Fields:

      * ``plan`` — the certified ``Plan``.
      * ``alpha`` — overall PAC confidence level (1 − α).
      * ``regime`` — INDEPENDENT or WORST_CASE.
      * ``per_step_lower`` — list of (op_name, lower-bound) pairs.
      * ``per_step_upper`` — list of (op_name, upper-bound) pairs.
      * ``reliability_lower`` — overall lower bound at 1 − α.
      * ``reliability_upper`` — overall upper bound at 1 − α.
      * ``expected_cost`` — expected monetary cost.
      * ``bound_method`` — "clopper_pearson" | "kl_inv" |
        "empirical_bernstein" | "hoeffding".
      * ``fingerprint`` — SHA-256 hash chain reference.
    """
    plan: Plan
    alpha: float
    regime: str
    per_step_lower: tuple
    per_step_upper: tuple
    reliability_lower: float
    reliability_upper: float
    expected_cost: float
    bound_method: str
    fingerprint: str

    def to_jsonable(self) -> dict:
        return {
            "alpha": self.alpha,
            "regime": self.regime,
            "reliability_lower": self.reliability_lower,
            "reliability_upper": self.reliability_upper,
            "expected_cost": self.expected_cost,
            "bound_method": self.bound_method,
            "per_step_lower": [list(t) for t in self.per_step_lower],
            "per_step_upper": [list(t) for t in self.per_step_upper],
            "fingerprint": self.fingerprint,
            "plan": self.plan.to_jsonable(),
        }


@dataclass(frozen=True)
class Outcome:
    r"""Result of executing a plan against a caller-supplied executor."""
    plan: Plan
    succeeded: bool
    steps_run: int
    last_op: str
    error: str
    outputs: tuple
    fingerprint: str

    def to_jsonable(self) -> dict:
        return {
            "succeeded": self.succeeded,
            "steps_run": self.steps_run,
            "last_op": self.last_op,
            "error": self.error,
            "outputs": list(self.outputs),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ComposerReport:
    r"""Aggregate snapshot of a Composer instance."""
    operator_count: int
    axiom_count: int
    plan_count: int
    operator_stats: tuple
    sccs: tuple
    cycles: tuple
    type_classes: tuple
    fingerprint: str
    timestamp: float

    def to_jsonable(self) -> dict:
        return {
            "operator_count": self.operator_count,
            "axiom_count": self.axiom_count,
            "plan_count": self.plan_count,
            "operator_stats": [dict(s) for s in self.operator_stats],
            "sccs": [list(c) for c in self.sccs],
            "cycles": [list(c) for c in self.cycles],
            "type_classes": [list(c) for c in self.type_classes],
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
        }


# =====================================================================
# Graph utilities — SCC, topological sort
# =====================================================================


def strongly_connected_components(
    nodes: Iterable[Any], edges: Iterable[tuple],
) -> list[list]:
    r"""Tarjan's strongly-connected-components algorithm (1972).

    Returns the SCCs in *reverse topological order* (sinks first).
    Each SCC is a list of node identifiers.
    """
    node_list = list(nodes)
    index = {n: i for i, n in enumerate(node_list)}
    adj: dict[Any, list] = {n: [] for n in node_list}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].append(v)

    idx_counter = [0]
    stack: list = []
    on_stack: set = set()
    indices: dict = {}
    lowlinks: dict = {}
    sccs: list[list] = []

    def strongconnect(v):
        # Iterative Tarjan to avoid stack overflow on deep graphs.
        work = [(v, iter(adj[v]), False)]
        while work:
            node, it, started = work[-1]
            if not started:
                indices[node] = idx_counter[0]
                lowlinks[node] = idx_counter[0]
                idx_counter[0] += 1
                stack.append(node)
                on_stack.add(node)
                work[-1] = (node, it, True)
            try:
                w = next(it)
            except StopIteration:
                work.pop()
                if lowlinks[node] == indices[node]:
                    component: list = []
                    while True:
                        w2 = stack.pop()
                        on_stack.discard(w2)
                        component.append(w2)
                        if w2 == node:
                            break
                    sccs.append(component)
                if work:
                    parent = work[-1][0]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[node])
                continue
            if w not in indices:
                work.append((w, iter(adj[w]), False))
            elif w in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[w])
    for v in node_list:
        if v not in indices:
            strongconnect(v)
    return sccs


def topological_sort(nodes: Iterable[Any], edges: Iterable[tuple]) -> list:
    """Kahn's algorithm (1962).  Raises if the graph contains a cycle."""
    node_list = list(nodes)
    in_deg: dict = {n: 0 for n in node_list}
    adj: dict[Any, list] = {n: [] for n in node_list}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].append(v)
            in_deg[v] += 1
    ready = sorted([n for n in node_list if in_deg[n] == 0], key=str)
    out: list = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for w in adj[n]:
            in_deg[w] -= 1
            if in_deg[w] == 0:
                bisect.insort(ready, w, key=str)
    if len(out) != len(node_list):
        raise ComposerError("graph has a cycle; topological sort impossible")
    return out


# =====================================================================
# Composer
# =====================================================================


class Composer:
    r"""Typed, certified compositional planner.

    Thread-safe.  All public methods acquire a single lock so the
    operator registry and the fingerprint chain stay consistent.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        max_expansions: int = _DEFAULT_MAX_EXPANSIONS,
        max_depth: int = _DEFAULT_MAX_DEPTH,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock or time.time
        self._max_expansions = int(max_expansions)
        self._max_depth = int(max_depth)
        self._operators: dict[str, Operator] = {}
        self._axioms: set[Predicate] = set()
        self._plans: list[Plan] = []
        self._fingerprint: str = _GENESIS
        self._events: list[dict] = []
        # operator-pair graph for SCC analysis
        self._produces: dict[str, set] = {}      # op -> set[pred-name-arity]
        self._consumes: dict[str, set] = {}      # op -> set[pred-name-arity]
        # cached counter for fresh type variables
        self._tv_counter = [0]
        self._emit(COMPOSER_STARTED, {"genesis": _GENESIS})

    # ------------------------------------------------------------------
    # Fingerprint + event log
    # ------------------------------------------------------------------

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if kind not in KNOWN_EVENTS:
            raise ComposerError(f"unknown event {kind!r}")
        canonical = json.dumps(
            {"kind": kind, "payload": payload},
            sort_keys=True,
            default=_jsonable,
            separators=(",", ":"),
        )
        h = hashlib.sha256()
        h.update(self._fingerprint.encode())
        h.update(canonical.encode())
        self._fingerprint = h.hexdigest()
        self._events.append(
            {
                "kind": kind,
                "ts": self._clock(),
                "payload": dict(payload),
                "fingerprint": self._fingerprint,
            }
        )

    @property
    def fingerprint(self) -> str:
        with self._lock:
            return self._fingerprint

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_operator(
        self,
        name: str,
        *,
        params: Iterable[Any] = (),
        pre: Iterable[Any] = (),
        add: Iterable[Any] = (),
        delete: Iterable[Any] = (),
        cost: float = 0.0,
        reliability: float | None = None,
        prior_strength: float = 2.0,
        alpha: float | None = None,
        beta: float | None = None,
        meta: Any = None,
    ) -> Operator:
        r"""Register a new typed operator.

        ``reliability`` and ``prior_strength`` are a convenient
        parameterisation: ``Beta(reliability·N, (1-reliability)·N)``
        with ``N = prior_strength`` gives a prior mean equal to
        ``reliability`` and effective sample size ``N``.  Alternatively
        pass ``alpha`` and ``beta`` directly.
        """
        with self._lock:
            if name in self._operators:
                raise InvalidOperator(f"operator {name!r} is already registered")
            if reliability is not None or prior_strength != 2.0:
                if alpha is not None or beta is not None:
                    raise InvalidOperator(
                        "specify (reliability, prior_strength) OR (alpha, beta), not both"
                    )
                if reliability is None:
                    reliability = 0.9
                if not (0.0 < reliability < 1.0):
                    raise InvalidOperator(
                        f"reliability must be in (0, 1); got {reliability}"
                    )
                if prior_strength <= 0:
                    raise InvalidOperator(
                        f"prior_strength must be positive; got {prior_strength}"
                    )
                a = max(reliability * prior_strength, _EPS)
                b = max((1.0 - reliability) * prior_strength, _EPS)
            else:
                a = _DEFAULT_PRIOR_ALPHA if alpha is None else float(alpha)
                b = _DEFAULT_PRIOR_BETA if beta is None else float(beta)
            op = _make_operator(
                name,
                params=params,
                pre=pre,
                add=add,
                delete=delete,
                cost=cost,
                alpha=a,
                beta=b,
                meta=meta,
            )
            self._operators[name] = op
            # update graph book-keeping
            self._produces[name] = {(p.name, len(p.args)) for p in op.add}
            self._consumes[name] = {(p.name, len(p.args)) for p in op.pre}
            self._emit(
                COMPOSER_OPERATOR_REGISTERED,
                {
                    "name": name,
                    "params": [(n, str(t)) for n, t in op.params],
                    "pre": [str(p) for p in op.pre],
                    "add": [str(p) for p in op.add],
                    "delete": [str(p) for p in op.delete],
                    "cost": op.cost,
                    "alpha": op.alpha,
                    "beta": op.beta,
                },
            )
            return op

    def add_axiom(self, predicate: Any) -> Predicate:
        """Add a ground predicate to the global axiom store.

        Axioms are facts every plan starts with, in addition to any
        ``Goal.initial`` provided at synthesis.
        """
        with self._lock:
            p = parse_predicate(predicate)
            if not p.is_ground:
                raise InvalidPredicate(
                    f"axiom must be ground; {p} contains a variable"
                )
            self._axioms.add(p)
            self._emit(COMPOSER_AXIOM_ADDED, {"predicate": str(p)})
            return p

    def operators(self) -> dict[str, Operator]:
        with self._lock:
            return dict(self._operators)

    def axioms(self) -> frozenset:
        with self._lock:
            return frozenset(self._axioms)

    def clear(self) -> None:
        with self._lock:
            self._operators.clear()
            self._axioms.clear()
            self._plans.clear()
            self._produces.clear()
            self._consumes.clear()
            self._emit(COMPOSER_CLEARED, {})

    # ------------------------------------------------------------------
    # Synthesis — A* planning
    # ------------------------------------------------------------------

    def synthesize(
        self,
        initial: Iterable[Any],
        post: Iterable[Any],
        *,
        algorithm: str = ASTAR,
        heuristic: str = H_LANDMARK,
        budget: int | None = None,
        max_depth: int | None = None,
        weight: float = 1.0,
    ) -> Plan:
        r"""Synthesise a plan from ``initial`` to ``post``.

        ``algorithm``: one of KNOWN_ALGORITHMS.
        ``heuristic``: one of KNOWN_HEURISTICS.
        ``budget``: maximum number of A*-node expansions; ``None``
        uses the Composer-level default.
        ``max_depth``: maximum plan length.
        ``weight``: weighted-A* multiplier on the heuristic (1.0 =
        ordinary A*; > 1.0 trades optimality for speed).
        """
        with self._lock:
            if algorithm not in KNOWN_ALGORITHMS:
                raise UnknownAlgorithm(
                    f"{algorithm!r} not in KNOWN_ALGORITHMS"
                )
            if heuristic not in KNOWN_HEURISTICS:
                raise UnknownHeuristic(
                    f"{heuristic!r} not in KNOWN_HEURISTICS"
                )
            if not self._operators:
                raise PlanningFailure("no operators registered")
            init_state = _state_freeze(
                set(self._axioms) | {parse_predicate(p) for p in initial}
            )
            for atom in init_state:
                if not atom.is_ground:
                    raise InvalidGoal(
                        f"initial state must be ground; {atom} has a variable"
                    )
            goal_atoms = _norm_predicates(post)
            budget = budget if budget is not None else self._max_expansions
            max_depth = max_depth if max_depth is not None else self._max_depth
            if algorithm in (ASTAR, DIJKSTRA):
                plan = self._astar(
                    init_state,
                    goal_atoms,
                    heuristic if algorithm == ASTAR else H_ZERO,
                    budget,
                    max_depth,
                    weight,
                )
            elif algorithm == IDA_STAR:
                plan = self._ida_star(
                    init_state, goal_atoms, heuristic, budget, max_depth
                )
            elif algorithm == REGRESSION:
                plan = self._regression(
                    init_state, goal_atoms, budget, max_depth
                )
            else:
                raise UnknownAlgorithm(algorithm)
            self._plans.append(plan)
            self._emit(
                COMPOSER_PLANNED,
                {
                    "algorithm": algorithm,
                    "heuristic": heuristic,
                    "verdict": plan.verdict,
                    "length": plan.length,
                    "cost": plan.cost,
                    "reliability_mean": plan.reliability_mean,
                    "fingerprint_in": self._fingerprint,
                },
            )
            return plan

    # ------------------------------------------------------------------
    # Ground-operator enumeration
    # ------------------------------------------------------------------

    def _enumerate_groundings(
        self, op: Operator, state: frozenset,
    ) -> Iterator[dict[str, Any]]:
        """Generate all ground bindings of ``op`` whose preconditions hold."""
        param_names = op.parameter_names()
        # First, partial-bind from any variables that occur in ``pre``.
        # Strategy: index state by predicate name; for each precondition,
        # iterate state atoms with the same name; merge bindings.
        state_by_name: dict[str, list[Predicate]] = {}
        for atom in state:
            state_by_name.setdefault(atom.name, []).append(atom)
        # Recursive backtracking over preconditions.
        pre_list = list(op.pre)
        if not pre_list:
            # No preconditions — bind all params to a sentinel "_".
            # We only generate this if every parameter is unused; otherwise
            # there is no way to choose it without input examples, so we
            # yield a single canonical binding.
            yield {n: f"_{op.name}_{n}" for n in param_names}
            return

        def backtrack(i: int, bindings: dict[str, Any]):
            if i == len(pre_list):
                # Fill any unbound params with sentinels (objects introduced
                # by the operator only).
                for n in param_names:
                    if n not in bindings:
                        bindings = {**bindings, n: f"_{op.name}_{n}"}
                yield bindings
                return
            cur = pre_list[i].substitute(bindings)
            candidates = state_by_name.get(cur.name, [])
            for atom in candidates:
                if len(atom.args) != len(cur.args):
                    continue
                # Unify position-wise: variables in ``cur`` must equal the
                # corresponding constant in ``atom``; existing bindings
                # must match.
                new_b = dict(bindings)
                ok = True
                for a, c in zip(atom.args, cur.args):
                    if _is_variable(c):
                        v = c[1:]
                        if v in new_b:
                            if new_b[v] != a:
                                ok = False
                                break
                        else:
                            new_b[v] = a
                    else:
                        if a != c:
                            ok = False
                            break
                if ok:
                    yield from backtrack(i + 1, new_b)

        yield from backtrack(0, {})

    def _apply_operator(
        self, op: Operator, bindings: dict[str, Any], state: frozenset,
    ) -> frozenset:
        """Apply the (already-validated) operator and return the new state."""
        new_state = set(state)
        for p in op.delete:
            new_state.discard(p.substitute(bindings))
        for p in op.add:
            new_state.add(p.substitute(bindings))
        return frozenset(new_state)

    def _goal_satisfied(
        self, goal: tuple, state: frozenset,
    ) -> dict[str, Any] | None:
        """Return a binding under which every goal atom holds in ``state``.

        ``None`` if no such binding exists.
        """
        state_by_name: dict[str, list[Predicate]] = {}
        for atom in state:
            state_by_name.setdefault(atom.name, []).append(atom)
        result: dict[str, Any] = {}

        def backtrack(i: int, bindings: dict[str, Any]):
            if i == len(goal):
                return bindings
            cur = goal[i].substitute(bindings)
            for atom in state_by_name.get(cur.name, []):
                if len(atom.args) != len(cur.args):
                    continue
                new_b = dict(bindings)
                ok = True
                for a, c in zip(atom.args, cur.args):
                    if _is_variable(c):
                        v = c[1:]
                        if v in new_b:
                            if new_b[v] != a:
                                ok = False
                                break
                        else:
                            new_b[v] = a
                    else:
                        if a != c:
                            ok = False
                            break
                if ok:
                    out = backtrack(i + 1, new_b)
                    if out is not None:
                        return out
            return None

        return backtrack(0, result)

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def _h_zero(self, _state: frozenset, _goal: tuple) -> float:
        return 0.0

    def _h_goal_count(self, state: frozenset, goal: tuple) -> float:
        """How many goal atoms aren't yet (consistently) satisfied."""
        if self._goal_satisfied(goal, state) is not None:
            return 0.0
        sat = 0
        for atom in goal:
            if atom.is_ground:
                if atom in state:
                    sat += 1
            else:
                # any state atom with the right name & arity counts as
                # potentially satisfiable.
                for a in state:
                    if a.name == atom.name and len(a.args) == len(atom.args):
                        sat += 1
                        break
        return float(len(goal) - sat)

    def _h_landmark(self, state: frozenset, goal: tuple) -> float:
        r"""Admissible HSP-style landmark heuristic (Helmert-Domshlak 2009).

        For every unsatisfied goal atom, the heuristic adds the
        *cheapest* operator that has that atom in its add list.  This
        underestimates the true cost of reaching the goal (because the
        cheapest single-step achiever is a lower bound) and is
        therefore admissible.
        """
        if self._goal_satisfied(goal, state) is not None:
            return 0.0
        # Index: which goal-atom names are unmet (loosely, name+arity).
        unmet: list[tuple[str, int]] = []
        for atom in goal:
            if atom.is_ground and atom in state:
                continue
            unmet.append((atom.name, len(atom.args)))
        if not unmet:
            return 0.0
        # Cost-to-achieve for each (name, arity).
        # We use the negative-log mean reliability of the cheapest adder
        # plus its monetary cost; choose the smallest sum.
        cost = 0.0
        for sig in unmet:
            best = _INF
            for op in self._operators.values():
                for a in op.add:
                    if (a.name, len(a.args)) == sig:
                        c = op.cost - _safe_log(op.reliability_mean())
                        if c < best:
                            best = c
                        break
            if best < _INF:
                cost += best
        return cost

    def _heuristic(self, name: str) -> Callable[[frozenset, tuple], float]:
        if name == H_ZERO:
            return self._h_zero
        if name == H_GOAL_COUNT:
            return self._h_goal_count
        if name == H_LANDMARK:
            return self._h_landmark
        raise UnknownHeuristic(name)

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------

    def _astar(
        self,
        initial: frozenset,
        goal: tuple,
        heuristic: str,
        budget: int,
        max_depth: int,
        weight: float,
    ) -> Plan:
        h_fn = self._heuristic(heuristic)
        start_g = 0.0
        start_h = h_fn(initial, goal)
        # priority queue: (f, tiebreak, state, g, path)
        # path is a tuple of (op_name, bindings_tuple, cost, rel_mean).
        tiebreak = 0
        open_heap: list = []
        heapq.heappush(
            open_heap, (start_g + weight * start_h, tiebreak, initial, start_g, ())
        )
        closed: dict[frozenset, float] = {}
        expansions = 0
        while open_heap:
            f, _tb, state, g, path = heapq.heappop(open_heap)
            if state in closed and closed[state] <= g:
                continue
            closed[state] = g
            sat = self._goal_satisfied(goal, state)
            if sat is not None:
                return self._build_plan(initial, state, path, sat, SOLVED)
            if len(path) >= max_depth:
                continue
            expansions += 1
            if expansions > budget:
                return self._build_failure(initial, BUDGET_EXHAUSTED)
            # Expand: enumerate (op, binding) pairs.
            for op in self._operators.values():
                for bindings in self._enumerate_groundings(op, state):
                    next_state = self._apply_operator(op, bindings, state)
                    if next_state == state:
                        continue  # no progress
                    step_g = (
                        op.cost - _safe_log(op.reliability_mean())
                    )
                    next_g = g + step_g
                    if (
                        next_state in closed
                        and closed[next_state] <= next_g
                    ):
                        continue
                    next_h = h_fn(next_state, goal)
                    tiebreak += 1
                    next_path = path + (
                        (
                            op.name,
                            tuple(sorted(bindings.items())),
                            step_g,
                            op.reliability_mean(),
                        ),
                    )
                    heapq.heappush(
                        open_heap,
                        (
                            next_g + weight * next_h,
                            tiebreak,
                            next_state,
                            next_g,
                            next_path,
                        ),
                    )
        return self._build_failure(initial, INFEASIBLE)

    # ------------------------------------------------------------------
    # IDA*
    # ------------------------------------------------------------------

    def _ida_star(
        self,
        initial: frozenset,
        goal: tuple,
        heuristic: str,
        budget: int,
        max_depth: int,
    ) -> Plan:
        h_fn = self._heuristic(heuristic)
        threshold = h_fn(initial, goal)
        expansions = [0]

        def dfs(
            state: frozenset,
            g: float,
            path: tuple,
            seen: frozenset,
            thr: float,
        ) -> tuple[Any, float]:
            f = g + h_fn(state, goal)
            if f > thr:
                return (None, f)
            sat = self._goal_satisfied(goal, state)
            if sat is not None:
                return ((state, path, sat), thr)
            if len(path) >= max_depth:
                return (None, _INF)
            expansions[0] += 1
            if expansions[0] > budget:
                return (None, _INF)
            min_next = _INF
            for op in self._operators.values():
                for bindings in self._enumerate_groundings(op, state):
                    next_state = self._apply_operator(op, bindings, state)
                    if next_state == state or next_state in seen:
                        continue
                    step_g = op.cost - _safe_log(op.reliability_mean())
                    new_seen = seen | {next_state}
                    new_path = path + (
                        (
                            op.name,
                            tuple(sorted(bindings.items())),
                            step_g,
                            op.reliability_mean(),
                        ),
                    )
                    result, nt = dfs(
                        next_state, g + step_g, new_path, new_seen, thr
                    )
                    if result is not None:
                        return (result, thr)
                    if nt < min_next:
                        min_next = nt
            return (None, min_next)

        seen0: frozenset = frozenset({initial})
        # Iterate, increasing the threshold each time.
        for _ in range(256):
            result, next_thr = dfs(initial, 0.0, (), seen0, threshold)
            if result is not None:
                state_f, path_f, sat = result
                return self._build_plan(initial, state_f, path_f, sat, SOLVED)
            if next_thr == _INF or expansions[0] > budget:
                if expansions[0] > budget:
                    return self._build_failure(initial, BUDGET_EXHAUSTED)
                return self._build_failure(initial, INFEASIBLE)
            if next_thr <= threshold:
                # ill-formed monotone iteration; bail out
                return self._build_failure(initial, INFEASIBLE)
            threshold = next_thr
        return self._build_failure(initial, BUDGET_EXHAUSTED)

    # ------------------------------------------------------------------
    # Regression planning (backward chaining)
    # ------------------------------------------------------------------

    def _regression(
        self,
        initial: frozenset,
        goal: tuple,
        budget: int,
        max_depth: int,
    ) -> Plan:
        r"""STRIPS regression (Fikes-Nilsson 1971).

        Start from the goal as a set of *abstract* sub-goals; at each
        step, pick a sub-goal, find an operator whose ``add`` list
        contains it, replace the sub-goal with the operator's ``pre``
        list.  Search succeeds when the regressed sub-goal set is
        contained in ``initial``.
        """
        # Represent the open set as (regressed_atoms_frozenset, path).
        # The path is reverse-applied: it is the *plan suffix* from
        # the current sub-goal set back to the initial state.
        start = frozenset(goal)
        if self._goal_satisfied(goal, initial) is not None:
            return self._build_plan(initial, initial, (), {}, SOLVED)
        queue: list = [(0.0, 0, start, ())]
        seen: dict[frozenset, float] = {start: 0.0}
        tiebreak = 0
        expansions = 0
        while queue:
            cost, _tb, subgoals, suffix = heapq.heappop(queue)
            if seen.get(subgoals, _INF) < cost:
                continue
            # Check if ``initial`` already entails ``subgoals``.
            if all(self._regression_entailed(sg, initial) for sg in subgoals):
                # We have to assemble the path: ``suffix`` is the
                # reverse application — re-order forwards.
                final, path = self._regression_replay(initial, suffix)
                sat = self._goal_satisfied(goal, final) or {}
                return self._build_plan(initial, final, path, sat, SOLVED)
            if len(suffix) >= max_depth:
                continue
            expansions += 1
            if expansions > budget:
                return self._build_failure(initial, BUDGET_EXHAUSTED)
            # Expand: for each sub-goal, find every operator that adds it.
            for sg in list(subgoals):
                for op in self._operators.values():
                    for add_atom in op.add:
                        if add_atom.name != sg.name:
                            continue
                        if len(add_atom.args) != len(sg.args):
                            continue
                        # Unify the add atom with the sub-goal.
                        binding: dict[str, Any] = {}
                        ok = True
                        for a, b in zip(add_atom.args, sg.args):
                            if _is_variable(a):
                                v = a[1:]
                                if v in binding:
                                    if binding[v] != b:
                                        ok = False
                                        break
                                else:
                                    binding[v] = b
                            elif _is_variable(b):
                                # sub-goal variable; bind it to the add
                                # constant.
                                ok = ok  # keep as-is; we'll let it stay
                                # We treat sub-goal variables as
                                # existentially quantified: bind them.
                                bv = b[1:]
                                if bv in binding:
                                    if binding[bv] != a:
                                        ok = False
                                        break
                                else:
                                    binding[bv] = a
                            else:
                                if a != b:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        # Replace ``sg`` with the operator's pre-list
                        # (after applying ``binding``).
                        new_subgoals = set(subgoals)
                        new_subgoals.discard(sg)
                        for pre_atom in op.pre:
                            new_subgoals.add(pre_atom.substitute(binding))
                        # Also subtract anything the operator deletes
                        # from the existing sub-goals would be a no-op
                        # (we don't track conflict yet — pure STRIPS).
                        next_set = frozenset(new_subgoals)
                        step_g = op.cost - _safe_log(op.reliability_mean())
                        next_cost = cost + step_g
                        if seen.get(next_set, _INF) <= next_cost:
                            continue
                        seen[next_set] = next_cost
                        tiebreak += 1
                        new_suffix = (
                            (
                                op.name,
                                tuple(sorted(binding.items())),
                                step_g,
                                op.reliability_mean(),
                            ),
                        ) + suffix
                        heapq.heappush(
                            queue,
                            (next_cost, tiebreak, next_set, new_suffix),
                        )
        return self._build_failure(initial, INFEASIBLE)

    def _regression_entailed(
        self, sg: Predicate, state: frozenset,
    ) -> bool:
        if sg.is_ground:
            return sg in state
        # existential: try to find any atom in ``state`` matching
        for atom in state:
            if atom.name != sg.name:
                continue
            if len(atom.args) != len(sg.args):
                continue
            ok = True
            binding: dict[str, Any] = {}
            for a, b in zip(atom.args, sg.args):
                if _is_variable(b):
                    v = b[1:]
                    if v in binding:
                        if binding[v] != a:
                            ok = False
                            break
                    else:
                        binding[v] = a
                else:
                    if a != b:
                        ok = False
                        break
            if ok:
                return True
        return False

    def _regression_replay(
        self, initial: frozenset, suffix: tuple,
    ) -> tuple[frozenset, tuple]:
        """Replay the regression suffix forward against the initial state."""
        state = initial
        path: list = []
        for step in suffix:
            op_name, bindings_tuple, step_g, rel = step
            op = self._operators[op_name]
            bindings = dict(bindings_tuple)
            # If binding doesn't cover the operator's preconditions
            # against the current state, we re-bind using
            # _enumerate_groundings filtered to the partial bindings
            # we already have.
            applied = False
            for cand in self._enumerate_groundings(op, state):
                # cand must agree with bindings on shared keys; symbolic
                # ``?``-prefixed values from the regression search are
                # placeholders to be concretised by the candidate.
                ok = True
                for k, v in bindings.items():
                    if isinstance(v, str) and v.startswith("?"):
                        continue
                    if k in cand and cand[k] != v:
                        ok = False
                        break
                if ok:
                    state = self._apply_operator(op, cand, state)
                    path.append(
                        (op.name, tuple(sorted(cand.items())), step_g, rel)
                    )
                    applied = True
                    break
            if not applied:
                # Could not concretise; abandon the plan as ill-grounded.
                raise PlanningFailure(
                    f"regression plan step {op_name!r} could not ground"
                )
        return state, tuple(path)

    # ------------------------------------------------------------------
    # Build a Plan from a search result
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        initial: frozenset,
        final: frozenset,
        path: tuple,
        goal_bindings: dict,
        verdict: str,
    ) -> Plan:
        steps: list[PlanStep] = []
        total_cost = 0.0
        log_rel = 0.0
        # Build a type witness: for every parameter binding, the
        # declared parameter type is captured so a downstream typed
        # executor or auditor can read what the planner intended.
        # We do *not* unify across operator calls here — the STRIPS
        # predicates already encode dataflow; types are a documentation
        # / executor-side discipline.  Use ``typecheck_plan`` for the
        # stricter cross-operator unification.
        subst: dict[str, Any] = {}
        for op_name, bindings_tuple, step_g, rel in path:
            op = self._operators[op_name]
            for n, t in op.params:
                subst.setdefault(f"{op_name}:{n}", t)
            mono = float(op.cost)
            steps.append(
                PlanStep(
                    op_name=op_name,
                    bindings=tuple(bindings_tuple),
                    cost=mono,
                    reliability_mean=float(rel),
                )
            )
            total_cost += mono
            log_rel += _safe_log(float(rel))
        rel_mean = math.exp(log_rel)
        plan = Plan(
            steps=tuple(steps),
            initial=initial,
            final=final,
            cost=total_cost,
            reliability_mean=rel_mean,
            type_subst={k: str(v) for k, v in subst.items()},
            goal_bindings={k: v for k, v in goal_bindings.items()},
            fingerprint=self._fingerprint,
            verdict=verdict,
        )
        return plan

    def _build_failure(self, initial: frozenset, verdict: str) -> Plan:
        return Plan(
            steps=(),
            initial=initial,
            final=initial,
            cost=0.0,
            reliability_mean=0.0,
            type_subst={},
            goal_bindings={},
            fingerprint=self._fingerprint,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Optional Hindley-Milner type checking of a synthesised plan
    # ------------------------------------------------------------------

    def typecheck_plan(self, plan: Plan) -> dict:
        r"""Run full HM unification across a synthesised plan's bindings.

        Two parameters are forced to unify when the *same constant
        value* is bound to both of them (i.e. the same Python object
        flows through both operator calls).  Returns a substitution
        on success.  Raises ``UnificationError`` on conflict — the
        caller can choose to reject the plan, replan with stricter
        predicates, or proceed with a typed wrapper.
        """
        with self._lock:
            counter = [self._tv_counter[0]]
            subst: dict[str, Any] = {}
            type_by_value: dict[Any, Any] = {}
            for step in plan.steps:
                op = self._operators.get(step.op_name)
                if op is None:
                    raise UnknownOperator(step.op_name)
                renamed: dict[str, Any] = {}
                for n, t in op.params:
                    renamed[n] = fresh_renaming(t, counter)
                for name, value in step.bindings:
                    pt = renamed.get(name)
                    if pt is None:
                        continue
                    if value in type_by_value:
                        subst = unify(pt, type_by_value[value], subst)
                    else:
                        type_by_value[value] = pt
            self._tv_counter[0] = counter[0]
            return {k: str(apply_subst(v, subst)) for k, v in subst.items()}

    # ------------------------------------------------------------------
    # Certification
    # ------------------------------------------------------------------

    def verify(
        self,
        plan: Plan,
        *,
        alpha: float = _DEFAULT_ALPHA,
        regime: str = INDEPENDENT,
        bound: str = "clopper_pearson",
    ) -> Certificate:
        r"""Return a PAC certificate for ``plan``'s end-to-end reliability.

        ``alpha``: overall confidence level (1 − α).
        ``regime``: INDEPENDENT or WORST_CASE.
        ``bound``: "clopper_pearson" | "kl_inv" |
        "empirical_bernstein" | "hoeffding".
        """
        if regime not in KNOWN_REGIMES:
            raise UnknownRegime(regime)
        if bound not in (
            "clopper_pearson",
            "kl_inv",
            "empirical_bernstein",
            "hoeffding",
        ):
            raise ComposerError(f"unknown bound method {bound!r}")
        if plan.verdict in (INFEASIBLE, BUDGET_EXHAUSTED):
            cert = Certificate(
                plan=plan,
                alpha=alpha,
                regime=regime,
                per_step_lower=(),
                per_step_upper=(),
                reliability_lower=0.0,
                reliability_upper=0.0,
                expected_cost=plan.cost,
                bound_method=bound,
                fingerprint=self._fingerprint,
            )
            return cert
        with self._lock:
            n_steps = max(plan.length, 1)
            # Bonferroni-divide α across steps.
            step_alpha = alpha / n_steps
            per_lo: list[tuple[str, float]] = []
            per_hi: list[tuple[str, float]] = []
            for step in plan.steps:
                op = self._operators.get(step.op_name)
                if op is None:
                    raise UnknownOperator(step.op_name)
                # The bound treats the prior as pseudo-data (Bayesian
                # credible interval = frequentist Clopper-Pearson under
                # the operational interpretation that ``prior_strength``
                # encodes effective pseudo-observations).  Callers who
                # want a pure frequentist bound on actual draws can use
                # ``op.reliability_observed_{k,n}()`` directly.
                n = max(int(round(op.alpha + op.beta - 2.0)), 0)
                k = max(int(round(op.alpha - 1.0)), 0)
                if n <= 0:
                    # Fall back to using the prior mean as a soft estimate;
                    # report a wide [0, 1] interval to signal no evidence.
                    lo = 0.0
                    hi = 1.0
                else:
                    if bound == "clopper_pearson":
                        lo = clopper_pearson_lower(k, n, step_alpha)
                        hi = clopper_pearson_upper(k, n, step_alpha)
                    elif bound == "kl_inv":
                        p_hat = k / n
                        lo = kl_bernoulli_lower_inverse(p_hat, n, step_alpha)
                        hi = kl_bernoulli_upper_inverse(p_hat, n, step_alpha)
                    elif bound == "empirical_bernstein":
                        lo = empirical_bernstein_lower(k, n, step_alpha)
                        hi = 1.0 - empirical_bernstein_lower(n - k, n, step_alpha)
                    elif bound == "hoeffding":
                        lo = hoeffding_lower(k, n, step_alpha)
                        hi = hoeffding_upper(k, n, step_alpha)
                    else:
                        lo = 0.0
                        hi = 1.0
                per_lo.append((step.op_name, lo))
                per_hi.append((step.op_name, hi))
            if regime == INDEPENDENT:
                # P(all succeed) = ∏ p_i ≥ ∏ lo_i
                rel_lo = 1.0
                rel_hi = 1.0
                for (_, lo), (_, hi) in zip(per_lo, per_hi):
                    rel_lo *= lo
                    rel_hi *= hi
            else:
                # P(plan fails) ≤ Σ P(o_i fails) ⇒ P(plan succeeds) ≥
                #     1 − Σ (1 − lo_i)
                fail_lo = sum(1.0 - lo for _, lo in per_lo)
                fail_hi = sum(1.0 - hi for _, hi in per_hi)
                rel_lo = _clip(1.0 - fail_lo, 0.0, 1.0)
                rel_hi = _clip(1.0 - fail_hi, 0.0, 1.0)
            cert = Certificate(
                plan=plan,
                alpha=alpha,
                regime=regime,
                per_step_lower=tuple(per_lo),
                per_step_upper=tuple(per_hi),
                reliability_lower=rel_lo,
                reliability_upper=rel_hi,
                expected_cost=plan.cost,
                bound_method=bound,
                fingerprint=self._fingerprint,
            )
            self._emit(
                COMPOSER_VERIFIED,
                {
                    "alpha": alpha,
                    "regime": regime,
                    "bound": bound,
                    "reliability_lower": rel_lo,
                    "reliability_upper": rel_hi,
                    "length": plan.length,
                    "cost": plan.cost,
                },
            )
            return cert

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: Plan,
        executor: Callable[[str, Mapping[str, Any]], tuple],
        *,
        observe: bool = True,
        stop_on_failure: bool = True,
    ) -> Outcome:
        r"""Run a plan against a caller-supplied ``executor``.

        ``executor`` is called for every step as
        ``executor(op_name, bindings_map)`` and must return a tuple
        ``(success: bool, output: Any)``.  On success, the binding
        of any newly-introduced operator output (which is *not*
        decomposed by this primitive — the executor is responsible
        for connecting outputs to subsequent step bindings) is
        recorded.

        If ``observe`` is True, the Beta-Bernoulli posterior of each
        operator is updated automatically with the executor's
        success bit.

        If ``stop_on_failure`` is True, the first failing step aborts
        the run; otherwise execution continues and the outcome reports
        the *fraction* of steps that succeeded.
        """
        with self._lock:
            if plan.verdict in (INFEASIBLE, BUDGET_EXHAUSTED):
                raise ExecutionFailure(
                    f"cannot execute plan with verdict {plan.verdict}"
                )
            outputs: list = []
            last_op = ""
            error = ""
            steps_run = 0
            any_failed = False
            for i, step in enumerate(plan.steps):
                last_op = step.op_name
                steps_run = i + 1
                try:
                    success, output = executor(step.op_name, step.binding_map)
                except Exception as exc:  # pragma: no cover - executor-defined
                    success, output = False, repr(exc)
                outputs.append({"op": step.op_name, "output": output})
                if observe:
                    self._observe_unlocked(step.op_name, bool(success))
                if not success:
                    any_failed = True
                    if not error:
                        error = f"step {i} ({step.op_name}) failed: {output!r}"
                    self._emit(
                        COMPOSER_STEP_FAILED,
                        {"index": i, "op": step.op_name, "output": _short(output)},
                    )
                    if stop_on_failure:
                        break
                else:
                    self._emit(
                        COMPOSER_STEP_OK,
                        {"index": i, "op": step.op_name, "output": _short(output)},
                    )
            ok = (not any_failed) and steps_run == plan.length
            out = Outcome(
                plan=plan,
                succeeded=ok,
                steps_run=steps_run,
                last_op=last_op,
                error=error,
                outputs=tuple(outputs),
                fingerprint=self._fingerprint,
            )
            self._emit(
                COMPOSER_EXECUTED,
                {
                    "succeeded": ok,
                    "steps_run": steps_run,
                    "last_op": last_op,
                    "fingerprint_in": self._fingerprint,
                },
            )
            return out

    # ------------------------------------------------------------------
    # Closed-loop observation
    # ------------------------------------------------------------------

    def observe(self, op_name: str, success: bool) -> Operator:
        """Record one Bernoulli observation of ``op_name``'s reliability."""
        with self._lock:
            return self._observe_unlocked(op_name, success)

    def _observe_unlocked(self, op_name: str, success: bool) -> Operator:
        if op_name not in self._operators:
            raise UnknownOperator(op_name)
        op = self._operators[op_name]
        if success:
            new = Operator(
                name=op.name,
                params=op.params,
                pre=op.pre,
                add=op.add,
                delete=op.delete,
                cost=op.cost,
                alpha=op.alpha + 1.0,
                beta=op.beta,
                prior_alpha=op.prior_alpha,
                prior_beta=op.prior_beta,
                meta=op.meta,
            )
        else:
            new = Operator(
                name=op.name,
                params=op.params,
                pre=op.pre,
                add=op.add,
                delete=op.delete,
                cost=op.cost,
                alpha=op.alpha,
                beta=op.beta + 1.0,
                prior_alpha=op.prior_alpha,
                prior_beta=op.prior_beta,
                meta=op.meta,
            )
        self._operators[op_name] = new
        self._emit(
            COMPOSER_OBSERVED,
            {"op": op_name, "success": bool(success), "alpha": new.alpha, "beta": new.beta},
        )
        return new

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, *, alpha: float = _DEFAULT_ALPHA) -> ComposerReport:
        with self._lock:
            op_stats: list[dict] = []
            for op in self._operators.values():
                k = max(op.reliability_observed_k(), 0)
                n = max(op.reliability_observed_n(), 0)
                if n > 0:
                    lo = clopper_pearson_lower(k, n, alpha)
                    hi = clopper_pearson_upper(k, n, alpha)
                else:
                    lo = 0.0
                    hi = 1.0
                op_stats.append(
                    {
                        "name": op.name,
                        "alpha": op.alpha,
                        "beta": op.beta,
                        "mean": op.reliability_mean(),
                        "observations": n,
                        "successes": k,
                        "lower": lo,
                        "upper": hi,
                        "cost": op.cost,
                    }
                )
            # SCCs of the operator-flow graph: edge u→v if u produces a
            # predicate that v consumes.
            nodes = list(self._operators.keys())
            edges: list[tuple] = []
            for u in nodes:
                produced = self._produces.get(u, set())
                for v in nodes:
                    if u == v:
                        continue
                    if produced & self._consumes.get(v, set()):
                        edges.append((u, v))
            sccs = strongly_connected_components(nodes, edges)
            cycles = [c for c in sccs if len(c) > 1 or self._self_loop(c[0], edges)]
            # Type equivalence classes: types appearing in operator
            # parameter or predicate-implicit signatures, partitioned
            # by their constructor name.
            tc: dict[str, list] = {}
            for op in self._operators.values():
                for _, t in op.params:
                    key = _type_class_key(t)
                    tc.setdefault(key, []).append(op.name)
            type_classes = tuple(
                tuple([k] + sorted(set(v))) for k, v in sorted(tc.items())
            )
            rep = ComposerReport(
                operator_count=len(self._operators),
                axiom_count=len(self._axioms),
                plan_count=len(self._plans),
                operator_stats=tuple(op_stats),
                sccs=tuple(tuple(c) for c in sccs),
                cycles=tuple(tuple(c) for c in cycles),
                type_classes=type_classes,
                fingerprint=self._fingerprint,
                timestamp=self._clock(),
            )
            self._emit(
                COMPOSER_REPORT,
                {
                    "operator_count": rep.operator_count,
                    "axiom_count": rep.axiom_count,
                    "plan_count": rep.plan_count,
                    "scc_count": len(sccs),
                    "cycle_count": len(cycles),
                },
            )
            return rep

    def _self_loop(self, node: Any, edges: list) -> bool:
        return any(u == node and v == node for u, v in edges)


# =====================================================================
# Small helpers
# =====================================================================


def _value_type(value: Any) -> Any:
    """Project a Python value to a TypeCon for unification."""
    if isinstance(value, bool):
        return TypeCon("Bool")
    if isinstance(value, int):
        return TypeCon("Int")
    if isinstance(value, float):
        return TypeCon("Float")
    if isinstance(value, str):
        return TypeCon("Str")
    if isinstance(value, (tuple, list)):
        if not value:
            return TypeCon("List", (TypeVar("a"),))
        inner = _value_type(value[0])
        return TypeCon("List", (inner,))
    if isinstance(value, dict):
        return TypeCon("Dict")
    return TypeCon("Object")


def _type_class_key(t: Any) -> str:
    if isinstance(t, TypeVar):
        return "?"
    if isinstance(t, TypeCon):
        return t.name
    return str(t)


def _short(x: Any, n: int = 80) -> str:
    s = repr(x)
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, frozenset):
        return sorted(obj, key=str)
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if isinstance(obj, (TypeVar, TypeCon, Predicate)):
        return str(obj)
    if isinstance(obj, Operator):
        return {
            "name": obj.name,
            "alpha": obj.alpha,
            "beta": obj.beta,
            "cost": obj.cost,
        }
    if hasattr(obj, "to_jsonable"):
        return obj.to_jsonable()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# =====================================================================
# Convenience: a minimal "registry from a list of dicts" constructor
# =====================================================================


def composer_from_spec(spec: Sequence[Mapping[str, Any]]) -> Composer:
    r"""Build a Composer from a JSON-style list of operator specs.

    Each spec is::

        {"name": "solve",
         "params": [("instance", "SATInstance")],
         "pre": ["available(?instance)"],
         "add": ["model(?instance)"],
         "cost": 0.001,
         "reliability": 0.99}
    """
    c = Composer()
    for entry in spec:
        kwargs = {
            "params": entry.get("params", ()),
            "pre": entry.get("pre", ()),
            "add": entry.get("add", ()),
            "delete": entry.get("delete", ()),
            "cost": float(entry.get("cost", 0.0)),
        }
        if "reliability" in entry:
            kwargs["reliability"] = float(entry["reliability"])
            kwargs["prior_strength"] = float(entry.get("prior_strength", 2.0))
        if "alpha" in entry or "beta" in entry:
            kwargs["alpha"] = float(entry.get("alpha", _DEFAULT_PRIOR_ALPHA))
            kwargs["beta"] = float(entry.get("beta", _DEFAULT_PRIOR_BETA))
        c.register_operator(entry["name"], **kwargs)
    return c


__all__ = [
    # Algorithms / heuristics
    "ASTAR", "IDA_STAR", "DIJKSTRA", "REGRESSION",
    "H_ZERO", "H_LANDMARK", "H_GOAL_COUNT",
    "INDEPENDENT", "WORST_CASE",
    "SOLVED", "INFEASIBLE", "BUDGET_EXHAUSTED", "ILL_TYPED",
    "KNOWN_ALGORITHMS", "KNOWN_HEURISTICS", "KNOWN_REGIMES",
    "KNOWN_VERDICTS",
    # Events
    "COMPOSER_STARTED", "COMPOSER_OPERATOR_REGISTERED",
    "COMPOSER_AXIOM_ADDED", "COMPOSER_PLANNED", "COMPOSER_VERIFIED",
    "COMPOSER_EXECUTED", "COMPOSER_STEP_OK", "COMPOSER_STEP_FAILED",
    "COMPOSER_OBSERVED", "COMPOSER_REPORT", "COMPOSER_CLEARED",
    "KNOWN_EVENTS",
    # Exceptions
    "ComposerError", "UnknownAlgorithm", "UnknownHeuristic",
    "UnknownRegime", "UnificationError", "TypeError_",
    "InvalidPredicate", "InvalidOperator", "UnknownOperator",
    "InvalidGoal", "PlanningFailure", "ExecutionFailure",
    # Numerical helpers
    "hoeffding_lower", "hoeffding_upper",
    "clopper_pearson_lower", "clopper_pearson_upper",
    "empirical_bernstein_lower",
    "kl_bernoulli", "kl_bernoulli_lower_inverse",
    "kl_bernoulli_upper_inverse", "pac_bayes_catoni",
    # Types
    "TypeVar", "TypeCon", "parse_type", "free_vars",
    "apply_subst", "unify", "fresh_renaming",
    # Predicates / operators / plans
    "Predicate", "parse_predicate",
    "Operator", "Goal", "PlanStep", "Plan", "Certificate",
    "Outcome", "ComposerReport",
    # Graph
    "strongly_connected_components", "topological_sort",
    # Composer
    "Composer", "composer_from_spec",
]
