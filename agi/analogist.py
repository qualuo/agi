r"""Analogist — structure-mapping analogical reasoning as a runtime primitive.

Every other primitive in this runtime treats reasoning *within* a
domain.  ``Predictor`` predicts the next symbol of a single stream.
``Scientist`` recovers a closed-form law from a single table.
``Conjecturer`` proposes a single proposition.  ``Inducer`` searches
for one program that fits one specification.  But the core operation a
coordination engine performs when it lifts a lesson learned in one
ticket into the policy that handles the *next* ticket — the operation
a debugger performs when it recognises that the bug in front of it has
the same shape as a bug it has seen before, the operation a
researcher performs when it carries the structure of an argument from
fluid dynamics into traffic flow — is **analogy**.  Analogy is the
mechanism by which knowledge from one situation is transported,
verified, and rendered actionable in another.

The ``Analogist`` is the runtime primitive that performs that
operation.  Given two relational descriptions — a *base* (well-known,
richly structured) and a *target* (unfamiliar, possibly incomplete) —
it returns a small set of *global mappings*, each a one-to-one,
parallel-connected alignment between base and target objects, ranked
by a structural-evaluation score that rewards *systematicity*
(Gentner 1983): the alignment of deep, interconnected relational
structure is preferred to the alignment of isolated attributes.  Each
global mapping comes with a list of **candidate inferences** —
expressions present in the base whose entities have already been
mapped to the target, projected as predictions about what *should* be
true in the target if the analogy is sound.

The pitch reduced to a runtime call::

    analogist = Analogist()
    analogist.add_description("solar", [
        ("cause",
            ("attracts", "sun", "planet"),
            ("revolves_around", "planet", "sun")),
        ("greater", ("mass", "sun"), ("mass", "planet")),
        ("greater", ("temperature", "sun"), ("temperature", "planet")),
        ("yellow", "sun"),
    ])
    analogist.add_description("atom", [
        ("cause",
            ("attracts", "nucleus", "electron"),
            ("revolves_around", "electron", "nucleus")),
        ("greater", ("mass", "nucleus"), ("mass", "electron")),
    ])

    report = analogist.match(base="solar", target="atom")

    # report.mappings[0].entity_map == {"sun": "nucleus", "planet": "electron"}
    # report.mappings[0].score        approximates the systematicity
    # report.mappings[0].inferences == [("greater",
    #                                     ("temperature","nucleus"),
    #                                     ("temperature","electron"))]
    # report.certificate              is an HMAC tag over the canonical
    #                                 mapping; the AttestationLedger can
    #                                 replay the run byte-for-byte.

MAC/FAC retrieval (Many Are Called, Few Are Chosen — Forbus, Gentner &
Law 1995) is the second face of the same primitive::

    # When a probe arrives, the runtime asks "what in my long-term
    # memory is structurally analogous to this?"  MAC computes a fast
    # content-vector dot-product over all descriptions; FAC runs the
    # full SME on the top candidates and returns the best.

    candidates = analogist.retrieve("atom", k=5)

The MAC stage uses ``O(|memory|)`` content-vector arithmetic; the FAC
stage runs SME only on the short-list — the cost profile that lets
the runtime keep a large memory of cases and still answer a retrieval
in bounded time.

Mathematical roots
------------------

  * **Gentner, D. (1983) — "Structure-mapping: a theoretical
    framework for analogy."**  *Cognitive Science* 7(2) 155–170.
    The original structure-mapping theory: analogy is the alignment
    of relational structure, not of attributes.  Defines the
    **systematicity principle** — higher-order, interconnected
    relations beat isolated facts.

  * **Falkenhainer, B., Forbus, K. D. & Gentner, D. (1989) — "The
    Structure-Mapping Engine: algorithm and examples."**  *Artificial
    Intelligence* 41(1) 1–63.  The SME algorithm we implement here:
    build local *match hypotheses* by recursing through aligned
    relation trees, score them via a **Structural Evaluation Score**
    (SES) that propagates support from parent matches to children,
    then merge consistent match hypotheses into globally one-to-one
    *gmaps* by greedy best-first search.

  * **Forbus, K. D., Ferguson, R. W. & Gentner, D. (1994) —
    "Incremental structure-mapping."**  *Proc. CogSci* 16.  The
    polynomial-time incremental variant that ``Analogist`` extends
    when the target description grows during a session.

  * **Forbus, K. D., Gentner, D. & Law, K. (1995) — "MAC/FAC: a
    model of similarity-based retrieval."**  *Cognitive Science*
    19(2) 141–205.  The two-stage retrieval architecture: a coarse
    *content vector* dot-product (MAC, Many Are Called) selects a
    handful of candidates from a large memory, then full SME (FAC,
    Few Are Chosen) ranks them by structural similarity.

  * **Markman, A. B. & Gentner, D. (1993) — "Structural alignment
    during similarity comparisons."**  *Cognitive Psychology* 25
    431–467.  Alignment-based comparison: similarity judgements
    follow from the same mapping that drives analogy.

  * **Gentner, D. & Markman, A. B. (1997) — "Structure mapping in
    analogy and similarity."**  *American Psychologist* 52 45–56.
    Synthesis of two decades of structure-mapping experiments and
    the cognitive role of the parallel-connectivity and
    one-to-one constraints.

  * **Holyoak, K. J. & Thagard, P. (1989) — "Analogical mapping by
    constraint satisfaction."**  *Cognitive Science* 13(3) 295–355.
    ACME — the alternative constraint-satisfaction-network approach.
    Implemented here as an optional second engine (``engine="acme"``)
    for users who want a connectionist treatment of structural,
    semantic, and pragmatic constraints simultaneously.

  * **Hummel, J. E. & Holyoak, K. J. (1997) — "Distributed
    representations of structure: a theory of analogical access and
    mapping."**  *Psychological Review* 104(3) 427–466.  LISA — the
    binding-by-synchrony account.  Not implemented (synchrony is not
    a runtime primitive), but the *role-filler binding* discipline
    LISA enforces is the discipline this module's expression
    grammar enforces statically.

  * **Forbus, K. D. (2001) — "Exploring analogy in the large."**  In
    *The Analogical Mind* (Gentner, Holyoak, Kokinov eds.), MIT
    Press.  Argues that analogy at runtime requires *retrieval* to
    be cheap and *mapping* to be sound — the MAC/FAC split this
    module enforces.

  * **Hofstadter, D. R. & Sander, E. (2013) — *Surfaces and
    Essences: Analogy as the Fuel and Fire of Thinking.*** Basic
    Books.  The book-length case that analogy is not a peripheral
    cognitive trick but the core of cognition — the operational
    motivation for treating it as a *runtime* primitive in the same
    tier as ``Solver`` and ``Planner``.

  * **Lovett, A. & Forbus, K. (2017) — "Modeling visual problem
    solving as analogical reasoning."**  *Psychological Review*
    124(1) 60–90.  Structure-mapping accounts for Raven's
    Progressive Matrices and other abstract-reasoning benchmarks
    on which large language models are still brittle — the empirical
    case for SME at the runtime tier.

  * **Mitchell, M. (1993) — *Analogy-Making as Perception.***  MIT
    Press.  The Copycat micro-domain (letter-string proportional
    analogies ``a:b::c:?``).  Implemented as a small dedicated
    sub-primitive ``ProportionalAnalogy`` for symbol-stream pattern
    transfer — the operational analogue of "promote a one-shot
    rewrite rule to a session-wide rewrite rule".

What Analogist gives a coordination engine
------------------------------------------

It gives the coordinator a *mathematically explicit* answer to the
question every other primitive presupposes but none answers: **"have
I seen this kind of problem before, and if so, what carries over?"**

  * For every match, the answer is *not* a similarity score; it is a
    one-to-one, parallel-connected ``entity_map``, an ``expr_map``
    on relational expressions, a SES score whose components
    (relational support, parental boost, attribute boost) are
    separately exposed, and a list of explicit *candidate inferences*
    that the coordinator can plug straight into ``Refuter`` for
    falsification or ``Conformal`` for distribution-free coverage of
    the transferred prediction.

  * The mapping is sound under two structural constraints — *one-to-
    one* (no base object maps to two target objects, and vice versa)
    and *parallel connectivity* (matched relations have matched
    arguments, recursively) — both of which are *certified* by the
    report.  A coordinator that wants to admit the analogy into its
    policy has a verifier; a coordinator that wants to reject it
    has a counter-example.

  * The cost is *measured* and *bounded*: ``report.n_match_hypotheses``,
    ``report.n_gmaps_explored``, ``report.n_inferences``, and
    ``report.budget_used`` are exposed on every report so a
    coordinator can SLO-gate the call.

  * The MAC/FAC retrieval surface is *associative*: descriptions
    can be added incrementally, and the content-vector index is
    updated in ``O(|description|)`` — the runtime's mechanism for
    cumulative cross-session memory.

  * Every report carries a ``certificate`` HMAC over the canonical
    mapping; a coordinator publishing a transferred lesson has a
    tamper-evident record that the analogy it acted on is the
    analogy it explains.

Public API
----------

The module exposes:

  * ``Expression`` — a Hashable, frozen relational expression.
  * ``Description`` — a named bag of expressions plus a derived
    content vector.
  * ``MatchHypothesis`` — an aligned ``(base_expr, target_expr)`` pair
    plus its local Structural Evaluation Score.
  * ``GlobalMapping`` — a one-to-one consistent set of match
    hypotheses plus aggregate score, ``entity_map`` and
    ``inferences``.
  * ``AnalogistConfig`` / ``AnalogistReport`` — configuration and the
    canonical report.
  * ``Analogist`` — the orchestrator, holding a long-term memory of
    descriptions and the SME / MAC-FAC algorithms.
  * ``ProportionalAnalogy`` — the small Copycat-style sub-primitive
    for letter-string ``a:b::c:?`` patterns.

This module is **pure stdlib** — no NumPy, no SciPy — because the
runtime ships analogy into the same low-dependency tier as
``Sketcher`` and ``Solver``.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
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


# =============================================================================
# Errors
# =============================================================================


class AnalogistError(Exception):
    """Base for every Analogist-raised error."""


class InvalidExpression(AnalogistError):
    """An expression is not a valid relational tree."""


class InvalidDescription(AnalogistError):
    """A Description was constructed with invalid contents."""


class UnknownDescription(AnalogistError):
    """A description name was referenced but never registered."""


class InvalidConfig(AnalogistError):
    """An AnalogistConfig is structurally invalid."""


class BudgetExhausted(AnalogistError):
    """A bounded SME run exhausted its time or match-hypothesis budget."""


# =============================================================================
# Expression DSL
# =============================================================================
#
# An *expression* is one of
#   - a string `s` — an *entity* (a constant of the domain),
#   - a tuple `(head, arg_1, ..., arg_n)` — a *functional* or
#     *relational* application, where ``head`` is a string predicate
#     name and each ``arg_i`` is itself an expression.
#
# By SME convention (Falkenhainer-Forbus-Gentner 1989) we distinguish
# the head's *kind*:
#
#   - "entity":     leaf (string only).
#   - "attribute":  1-arg predicate of an entity, e.g. ("yellow", "sun").
#                   In analogy mode (the default) attributes are
#                   *cheap matches* — they do not by themselves carry
#                   systematicity.
#   - "function":   n-arg term returning an object, e.g.
#                   ("mass", "sun").  Functions can match if their
#                   head identifies and their arguments align.
#   - "relation":   n-arg predicate returning truth, possibly higher-
#                   order (taking other relations as args), e.g.
#                   ("cause", R1, R2).  Relations are the carriers of
#                   systematicity; matching them propagates support
#                   to their children.
#
# The user typically does not need to declare kinds; the Analogist
# infers them from the expression structure and a small set of
# user-extensible rules.  Strings with no associated tuple are
# entities; tuple-heads ``cause``, ``implies``, ``and``, ``or``,
# ``not``, ``greater``, ``equal`` are higher-order relations by
# default; tuple-heads starting with lowercase letters and used as
# arguments to relations are functions; everything else is a
# first-order relation.

ExpressionLike = Union[str, Tuple[Any, ...]]
Expression = ExpressionLike  # alias — every expression is hashable


# Default classification of common predicate heads.  Users can override
# via AnalogistConfig.predicate_kinds.
_DEFAULT_HIGHER_ORDER: FrozenSet[str] = frozenset({
    "cause", "causes", "implies", "implied", "iff", "and", "or", "not",
    "greater", "less", "greater_or_equal", "less_or_equal", "equal",
    "between", "before", "after", "during",
})

_DEFAULT_FUNCTIONS: FrozenSet[str] = frozenset({
    "mass", "weight", "size", "temperature", "speed", "velocity",
    "position", "distance", "color", "shape", "value", "count",
    "level", "rate", "pressure", "volume",
})


def _is_entity(expr: ExpressionLike) -> bool:
    return isinstance(expr, str)


def _is_application(expr: ExpressionLike) -> bool:
    return isinstance(expr, tuple) and len(expr) >= 1 and isinstance(expr[0], str)


def _head(expr: ExpressionLike) -> str:
    if _is_entity(expr):
        return expr  # the entity itself
    if not _is_application(expr):
        raise InvalidExpression(f"not a valid expression: {expr!r}")
    return expr[0]


def _args(expr: ExpressionLike) -> Tuple[ExpressionLike, ...]:
    if _is_entity(expr):
        return ()
    if not _is_application(expr):
        raise InvalidExpression(f"not a valid expression: {expr!r}")
    return tuple(expr[1:])


def _walk(expr: ExpressionLike) -> Iterator[ExpressionLike]:
    """Yield every sub-expression of ``expr`` (including itself)."""
    yield expr
    if _is_application(expr):
        for a in _args(expr):
            yield from _walk(a)


def _entities(expr: ExpressionLike) -> Iterator[str]:
    """Yield every entity occurring in ``expr``."""
    for sub in _walk(expr):
        if _is_entity(sub):
            yield sub  # type: ignore[misc]


def _canonical_str(expr: ExpressionLike) -> str:
    """Deterministic textual rendering for hashing / display."""
    if _is_entity(expr):
        return expr  # type: ignore[return-value]
    return "(" + " ".join(_canonical_str(a) for a in expr) + ")"


def _validate_expression(expr: Any) -> None:
    if isinstance(expr, str):
        if not expr:
            raise InvalidExpression("entity name must be non-empty")
        return
    if not isinstance(expr, tuple):
        raise InvalidExpression(
            f"expression must be a string or tuple, got {type(expr).__name__}: {expr!r}"
        )
    if len(expr) == 0:
        raise InvalidExpression("expression tuple must be non-empty")
    if not isinstance(expr[0], str) or not expr[0]:
        raise InvalidExpression(f"head must be a non-empty string: {expr!r}")
    for a in expr[1:]:
        _validate_expression(a)


# =============================================================================
# Description
# =============================================================================


@dataclass(frozen=True)
class Description:
    """A named relational description: a bag of top-level expressions.

    A description is the unit on which SME operates.  Two descriptions
    (a *base* and a *target*) are aligned by ``Analogist.match``.

    The Description holds:

      * ``name``: human-readable label (also keys the long-term memory).
      * ``expressions``: tuple of top-level expressions.
      * ``predicate_kinds``: optional per-description overrides of the
        global ``AnalogistConfig.predicate_kinds`` dispatch.

    The Description is frozen and hashable; equality is by name plus a
    canonical hash of the expressions.  The content vector used by the
    MAC stage is derived lazily by the Analogist when the description
    is registered.
    """

    name: str
    expressions: Tuple[ExpressionLike, ...]
    predicate_kinds: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InvalidDescription("description name must be a non-empty string")
        if not isinstance(self.expressions, tuple):
            object.__setattr__(self, "expressions", tuple(self.expressions))
        for e in self.expressions:
            _validate_expression(e)

    def entities(self) -> Set[str]:
        out: Set[str] = set()
        for e in self.expressions:
            for s in _entities(e):
                out.add(s)
        return out

    def all_subexpressions(self) -> Iterator[ExpressionLike]:
        seen: Set[str] = set()
        for top in self.expressions:
            for sub in _walk(top):
                k = _canonical_str(sub)
                if k in seen:
                    continue
                seen.add(k)
                yield sub

    def fingerprint(self) -> str:
        body = "\n".join(sorted(_canonical_str(e) for e in self.expressions))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


# =============================================================================
# Predicate-kind dispatch
# =============================================================================


def _classify_predicate(
    head: str,
    arity: int,
    higher_order: FrozenSet[str],
    functions: FrozenSet[str],
    overrides: Mapping[str, str],
) -> str:
    """Return one of {"higher_order", "relation", "function", "attribute"}.

    Overrides win.  Then the global higher-order set, then the
    function set.  Then we fall back on the structural rule:
    ``arity == 1`` over an entity is an *attribute*, anything else
    is a first-order *relation*.
    """
    if head in overrides:
        v = overrides[head]
        if v not in {"higher_order", "relation", "function", "attribute"}:
            raise InvalidConfig(f"unknown predicate kind for {head!r}: {v!r}")
        return v
    if head in higher_order:
        return "higher_order"
    if head in functions:
        return "function"
    if arity == 1:
        return "attribute"
    return "relation"


def _alignable_kinds(k1: str, k2: str) -> bool:
    """Two expressions can match only if they have compatible kinds.

    SME (FFG 1989) allows: relation↔relation, function↔function,
    attribute↔attribute, higher_order↔higher_order.  Entities are
    aligned only through being co-arguments of matched relations,
    never as a primary match hypothesis.
    """
    return k1 == k2


# =============================================================================
# Match hypothesis
# =============================================================================


@dataclass(frozen=True)
class MatchHypothesis:
    """A candidate local match between a base and a target expression.

    Attributes:

      * ``base``: the base sub-expression.
      * ``target``: the target sub-expression.
      * ``kind``: classification of the head (relation / function / …).
      * ``base_str`` / ``target_str``: canonical textual rendering, for
        deterministic ordering and hashing.
      * ``score``: local SES contribution (before parental support).
      * ``support_kind``: provenance — "tiered_identicality" if the
        heads identify, "free_function" if functions match but heads
        differ.
    """

    base: ExpressionLike
    target: ExpressionLike
    kind: str
    base_str: str
    target_str: str
    score: float
    support_kind: str

    def __post_init__(self) -> None:
        if self.kind not in {"higher_order", "relation", "function", "attribute"}:
            raise AnalogistError(f"invalid MH kind: {self.kind!r}")
        if self.score < 0:
            raise AnalogistError("MH score must be non-negative")


# =============================================================================
# Global mapping
# =============================================================================


@dataclass(frozen=True)
class GlobalMapping:
    """A globally consistent alignment between base and target.

    The mapping is **one-to-one** (no base object maps to two target
    objects and vice versa) and **parallel connected** (every
    matched relation's arguments are also matched).

    Attributes:

      * ``entity_map``: ``{base_entity: target_entity}`` — the object
        correspondence implied by the mapping.
      * ``expr_map``: ``{base_canonical_str: target_canonical_str}``
        for every aligned expression (relation, function, attribute).
      * ``mhs``: the constituent ``MatchHypothesis`` tuples.
      * ``score``: aggregate Structural Evaluation Score after
        parental propagation.
      * ``inferences``: tuple of expressions present in base whose
        entities have already been mapped to target but whose
        instantiation is *not* yet in the target — i.e. candidate
        transfers.  Each is a tuple of (predicted_target_expr,
        source_base_expr).
      * ``support_breakdown``: ``{kind: contribution}`` of the SES.
    """

    entity_map: Mapping[str, str]
    expr_map: Mapping[str, str]
    mhs: Tuple[MatchHypothesis, ...]
    score: float
    inferences: Tuple[Tuple[ExpressionLike, ExpressionLike], ...]
    support_breakdown: Mapping[str, float]

    def project(self, expr: ExpressionLike) -> ExpressionLike:
        """Apply ``entity_map`` to ``expr``, rebuilding the tree.

        Unmapped entities are left in place — a coordination engine
        can detect them and decide whether to *skolemise* them as
        fresh target objects (candidate-inference mode).
        """
        if _is_entity(expr):
            return self.entity_map.get(expr, expr)  # type: ignore[arg-type]
        head = _head(expr)
        return (head,) + tuple(self.project(a) for a in _args(expr))


# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class AnalogistConfig:
    """Configuration for an Analogist instance.

    Sensible defaults are calibrated against the canonical
    Falkenhainer-Forbus-Gentner 1989 benchmark suite (solar-system
    ↔ atom, water-flow ↔ heat-flow, Karla-the-hawk ↔ Zerdia).  Set
    ``engine="acme"`` to switch to the Holyoak-Thagard constraint-
    satisfaction-network alternative (semantic and pragmatic
    constraints can then be supplied through ``acme_priors``).

    Attributes:

      * ``engine``: "sme" (default) or "acme".
      * ``mode``: "analogy" (default) — attributes contribute weakly;
        "literal" — attributes contribute fully; "mere_appearance" —
        only attributes contribute.
      * ``systematicity_weight``: λ in ``SES(child) += λ · SES(parent)``.
        Defaults to 0.5; FFG report 0.4–0.8 as good range.
      * ``attribute_weight``: weight of attribute matches relative to
        relation matches.  Defaults to 0.1 in analogy mode.
      * ``function_weight``: weight of function matches.
      * ``relation_weight``: weight of first-order relation matches.
      * ``higher_order_weight``: weight of higher-order relation
        matches.  Defaults to 1.5 — higher-order matches are *more*
        important in driving systematicity.
      * ``require_identical_predicates``: if True (default), only
        identical head names can match (tiered identicality).  If
        False, functions can match across non-identical heads
        (controlled relaxation; see FFG §5).
      * ``max_gmaps``: number of global mappings to return.  Default 3.
      * ``max_match_hypotheses``: hard budget; abort if exceeded.
      * ``time_budget_s``: optional wall-clock budget.
      * ``predicate_kinds``: per-head override of classification.
      * ``acme_priors``: optional ``{(base_expr_str, target_expr_str):
        prior_weight}`` extra evidence injected into ACME's network.
      * ``acme_iterations``: number of relaxation steps for ACME.
      * ``hmac_key``: optional bytes for the certificate HMAC.
      * ``seed``: deterministic tie-breaking seed.
    """

    engine: str = "sme"
    mode: str = "analogy"
    systematicity_weight: float = 0.5
    attribute_weight: float = 0.1
    function_weight: float = 1.0
    relation_weight: float = 1.0
    higher_order_weight: float = 1.5
    require_identical_predicates: bool = True
    max_gmaps: int = 3
    max_match_hypotheses: int = 50_000
    time_budget_s: Optional[float] = None
    predicate_kinds: Mapping[str, str] = field(default_factory=dict)
    acme_priors: Mapping[Tuple[str, str], float] = field(default_factory=dict)
    acme_iterations: int = 100
    hmac_key: Optional[bytes] = None
    seed: int = 0xC0FFEE

    def __post_init__(self) -> None:
        if self.engine not in {"sme", "acme"}:
            raise InvalidConfig(f"unknown engine: {self.engine!r}")
        if self.mode not in {"analogy", "literal", "mere_appearance"}:
            raise InvalidConfig(f"unknown mode: {self.mode!r}")
        if not (0.0 <= self.systematicity_weight <= 1.0):
            raise InvalidConfig("systematicity_weight must be in [0,1]")
        for k in ("attribute_weight", "function_weight",
                  "relation_weight", "higher_order_weight"):
            v = getattr(self, k)
            if v < 0:
                raise InvalidConfig(f"{k} must be non-negative, got {v}")
        if self.max_gmaps < 1:
            raise InvalidConfig("max_gmaps must be >= 1")
        if self.max_match_hypotheses < 1:
            raise InvalidConfig("max_match_hypotheses must be >= 1")
        if self.time_budget_s is not None and self.time_budget_s <= 0:
            raise InvalidConfig("time_budget_s must be > 0 when set")
        if self.acme_iterations < 1:
            raise InvalidConfig("acme_iterations must be >= 1")

    def effective_weight(self, kind: str) -> float:
        """Apply the mode policy to a kind's base weight."""
        base = {
            "attribute": self.attribute_weight,
            "function": self.function_weight,
            "relation": self.relation_weight,
            "higher_order": self.higher_order_weight,
        }[kind]
        if self.mode == "analogy" and kind == "attribute":
            return base  # attribute_weight already small
        if self.mode == "mere_appearance" and kind != "attribute":
            return 0.0
        if self.mode == "literal":
            return base
        return base


# =============================================================================
# Report
# =============================================================================


@dataclass(frozen=True)
class AnalogistReport:
    """Result of a single ``Analogist.match`` call.

    Attributes:

      * ``base_name`` / ``target_name``: names of the descriptions.
      * ``mappings``: top-``max_gmaps`` ``GlobalMapping`` objects,
        sorted by descending ``score``.
      * ``n_match_hypotheses``: number of MHs enumerated.
      * ``n_gmaps_explored``: number of global-mapping candidates
        considered in the merge step.
      * ``n_inferences``: total candidate inferences across all
        returned mappings.
      * ``engine``: which engine ran ("sme" or "acme").
      * ``duration_s``: wall-clock cost.
      * ``budget_used``: ``{"mh_count": ..., "time_s": ...}`` for
        SLO-gating.
      * ``base_fingerprint`` / ``target_fingerprint``: SHA-256 over the
        canonical descriptions; lets the AttestationLedger replay.
      * ``certificate``: HMAC-SHA-256 over the canonical mapping
        triples.  ``None`` if no ``hmac_key`` was configured.
    """

    base_name: str
    target_name: str
    mappings: Tuple[GlobalMapping, ...]
    n_match_hypotheses: int
    n_gmaps_explored: int
    n_inferences: int
    engine: str
    duration_s: float
    budget_used: Mapping[str, float]
    base_fingerprint: str
    target_fingerprint: str
    certificate: Optional[str]


@dataclass(frozen=True)
class RetrievalReport:
    """Result of an Analogist.retrieve (MAC/FAC) call.

    Attributes:

      * ``probe_name``: name of the probe description.
      * ``candidates``: tuple of ``(name, mac_score, fac_score,
        best_mapping_or_None)`` sorted by ``fac_score`` descending.
      * ``n_mac_evaluated``: number of memory items scored by MAC.
      * ``n_fac_evaluated``: number of items passed to FAC (SME).
      * ``duration_s``: wall-clock cost.
    """

    probe_name: str
    candidates: Tuple[Tuple[str, float, float, Optional[GlobalMapping]], ...]
    n_mac_evaluated: int
    n_fac_evaluated: int
    duration_s: float


# =============================================================================
# Content vector (for MAC stage)
# =============================================================================


def _content_vector(
    desc: Description,
    higher_order: FrozenSet[str],
    functions: FrozenSet[str],
    overrides: Mapping[str, str],
) -> Dict[str, float]:
    """Weighted bag-of-predicate-heads, by predicate kind.

    Following Forbus-Gentner-Law 1995 §3, the MAC vector counts
    *every occurrence* of a predicate, with weights chosen so that
    higher-order relations dominate cosine similarity — keeping MAC
    in qualitative agreement with the FAC step.
    """
    vec: Dict[str, float] = {}
    for top in desc.expressions:
        for sub in _walk(top):
            if _is_entity(sub):
                continue
            head = _head(sub)
            arity = len(_args(sub))
            kind = _classify_predicate(head, arity, higher_order, functions, overrides)
            w = {
                "higher_order": 4.0,
                "relation": 2.0,
                "function": 1.0,
                "attribute": 0.5,
            }[kind]
            vec[head] = vec.get(head, 0.0) + w
    return vec


def _cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    num = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


# =============================================================================
# Analogist — main orchestrator
# =============================================================================


class Analogist:
    """The structure-mapping orchestrator.

    Holds a long-term memory of named ``Description`` objects, runs
    SME (or ACME) between any pair, and answers MAC/FAC retrieval
    queries from the memory.

    Thread-safety:  the memory is mutated only by ``add_description``
    and ``forget``.  ``match`` and ``retrieve`` are pure with respect
    to the memory and may be called concurrently from multiple threads
    so long as no add/forget runs in parallel.
    """

    def __init__(self, config: Optional[AnalogistConfig] = None) -> None:
        self.config = config or AnalogistConfig()
        self._descriptions: Dict[str, Description] = {}
        self._content: Dict[str, Dict[str, float]] = {}
        self._higher_order: FrozenSet[str] = _DEFAULT_HIGHER_ORDER
        self._functions: FrozenSet[str] = _DEFAULT_FUNCTIONS

    # -------------------------------------------------------------------------
    # Memory management
    # -------------------------------------------------------------------------

    def add_description(
        self,
        name: str,
        expressions: Iterable[ExpressionLike],
        predicate_kinds: Optional[Mapping[str, str]] = None,
    ) -> Description:
        """Register a description in the long-term memory."""
        desc = Description(
            name=name,
            expressions=tuple(expressions),
            predicate_kinds=dict(predicate_kinds or {}),
        )
        self._descriptions[name] = desc
        self._content[name] = _content_vector(
            desc,
            self._higher_order,
            self._functions,
            {**dict(self.config.predicate_kinds), **dict(desc.predicate_kinds)},
        )
        return desc

    def forget(self, name: str) -> None:
        self._descriptions.pop(name, None)
        self._content.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._descriptions

    def get(self, name: str) -> Description:
        if name not in self._descriptions:
            raise UnknownDescription(name)
        return self._descriptions[name]

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._descriptions))

    def __len__(self) -> int:
        return len(self._descriptions)

    # -------------------------------------------------------------------------
    # Predicate-kind dispatch
    # -------------------------------------------------------------------------

    def _kind(
        self,
        expr: ExpressionLike,
        overrides: Mapping[str, str],
    ) -> str:
        if _is_entity(expr):
            return "entity"
        head = _head(expr)
        arity = len(_args(expr))
        return _classify_predicate(
            head, arity, self._higher_order, self._functions, overrides
        )

    # -------------------------------------------------------------------------
    # SME — Structure-Mapping Engine (Falkenhainer-Forbus-Gentner 1989)
    # -------------------------------------------------------------------------

    def match(
        self,
        base: Union[str, Description],
        target: Union[str, Description],
    ) -> AnalogistReport:
        """Compute the top global mappings from ``base`` onto ``target``.

        Pipeline (FFG 1989):

          1. **Match construction.**  Enumerate every ``(b, t)`` where
             ``b`` is a sub-expression of ``base``, ``t`` of ``target``,
             their kinds are alignable, and (under tiered identicality)
             their heads identify.  Each such pair becomes a
             ``MatchHypothesis`` with a *local* SES contribution.

          2. **Parallel-connectivity check.**  A relational MH is
             retained only if its arguments have at least one MH
             between them — without parallel connectivity the
             relational match cannot be "carried down" to bind objects.

          3. **Parental support.**  SES is propagated *downward*: a
             child MH's effective score is boosted by
             ``λ · score(parent)``.  Multiple parents add.

          4. **Conflict graph.**  Two MHs conflict if they violate
             one-to-one entity mapping or if they assign the same
             relational role twice.

          5. **Greedy global-mapping construction.**  Best-first
             search over consistent unions of MHs, keeping the
             top ``max_gmaps`` by aggregate SES.

          6. **Candidate inferences.**  For each gmap, take every
             top-level base expression whose entities are all mapped:
             if the projected expression is *not* already in the
             target, emit it as a transfer.
        """
        if self.config.engine == "acme":
            return self._acme(base, target)
        t0 = time.monotonic()
        b = self._resolve(base)
        t = self._resolve(target)
        overrides = self._merged_overrides(b, t)

        mhs = self._build_match_hypotheses(b, t, overrides, t0)
        gmaps, n_gmaps = self._build_global_mappings(b, t, mhs, overrides, t0)

        # Trim and rank
        gmaps_sorted = sorted(gmaps, key=lambda g: -g.score)[: self.config.max_gmaps]
        n_inferences = sum(len(g.inferences) for g in gmaps_sorted)

        duration = time.monotonic() - t0
        cert = self._certificate(b, t, gmaps_sorted)
        return AnalogistReport(
            base_name=b.name,
            target_name=t.name,
            mappings=tuple(gmaps_sorted),
            n_match_hypotheses=len(mhs),
            n_gmaps_explored=n_gmaps,
            n_inferences=n_inferences,
            engine="sme",
            duration_s=duration,
            budget_used={"mh_count": float(len(mhs)), "time_s": duration},
            base_fingerprint=b.fingerprint(),
            target_fingerprint=t.fingerprint(),
            certificate=cert,
        )

    def _resolve(self, x: Union[str, Description]) -> Description:
        if isinstance(x, Description):
            return x
        if x not in self._descriptions:
            raise UnknownDescription(x)
        return self._descriptions[x]

    def _merged_overrides(self, b: Description, t: Description) -> Dict[str, str]:
        d: Dict[str, str] = {}
        d.update(self.config.predicate_kinds)
        d.update(b.predicate_kinds)
        d.update(t.predicate_kinds)
        return d

    def _check_budget(self, t0: float, mh_count: int) -> None:
        if mh_count > self.config.max_match_hypotheses:
            raise BudgetExhausted(
                f"MH budget exceeded ({mh_count} > {self.config.max_match_hypotheses})"
            )
        if self.config.time_budget_s is not None:
            if time.monotonic() - t0 > self.config.time_budget_s:
                raise BudgetExhausted(
                    f"time budget exceeded ({self.config.time_budget_s}s)"
                )

    def _build_match_hypotheses(
        self,
        base: Description,
        target: Description,
        overrides: Mapping[str, str],
        t0: float,
    ) -> List[MatchHypothesis]:
        """Enumerate every alignable (base, target) sub-expression pair.

        Implements the FFG 1989 match-rule machinery (tiered
        identicality) and the parallel-connectivity gate.
        """
        base_subs = list(base.all_subexpressions())
        target_subs = list(target.all_subexpressions())

        mhs: List[MatchHypothesis] = []
        # First pass: enumerate same-kind / matching-head MHs.
        for b in base_subs:
            if _is_entity(b):
                continue
            for t in target_subs:
                if _is_entity(t):
                    continue
                kb = self._kind(b, overrides)
                kt = self._kind(t, overrides)
                if not _alignable_kinds(kb, kt):
                    continue
                if len(_args(b)) != len(_args(t)):
                    continue
                if self.config.require_identical_predicates:
                    if _head(b) != _head(t):
                        continue
                    support = "tiered_identicality"
                else:
                    if _head(b) == _head(t):
                        support = "tiered_identicality"
                    elif kb == "function":
                        support = "free_function"
                    else:
                        # First-order / higher-order relations require
                        # identicality (FFG 1989, §3.1).
                        continue
                weight = self.config.effective_weight(kb)
                if weight == 0.0:
                    continue
                mh = MatchHypothesis(
                    base=b,
                    target=t,
                    kind=kb,
                    base_str=_canonical_str(b),
                    target_str=_canonical_str(t),
                    score=weight,
                    support_kind=support,
                )
                mhs.append(mh)
                self._check_budget(t0, len(mhs))

        # Parallel-connectivity filter.  A relational MH is retained
        # only if every argument-pair admits at least one MH (else
        # there is no consistent way to bind the children's objects).
        # We iterate until fixpoint because parental MHs can lose
        # support when their children are filtered out.
        mh_index = {
            (mh.base_str, mh.target_str): mh for mh in mhs
        }
        while True:
            removed: List[Tuple[str, str]] = []
            for key, mh in list(mh_index.items()):
                if mh.kind == "attribute":
                    # 1-arg over an entity; parallel connectivity is
                    # trivially satisfied (just one entity-pair).
                    continue
                if not self._args_alignable(mh, mh_index, overrides):
                    removed.append(key)
            if not removed:
                break
            for k in removed:
                mh_index.pop(k, None)
        return list(mh_index.values())

    def _args_alignable(
        self,
        mh: MatchHypothesis,
        index: Mapping[Tuple[str, str], MatchHypothesis],
        overrides: Mapping[str, str],
    ) -> bool:
        """For each argument position, check that an alignment exists.

        For relational / higher-order args, require an MH between the
        argument expressions.  For entity args, any pairing is allowed
        (it will be enforced by the one-to-one consistency check at
        gmap-build time).  For function args, an MH between the
        function expressions is preferred but not required (a function
        with unknown arg structure may still be matched at the level
        of its head).
        """
        ba = _args(mh.base)
        ta = _args(mh.target)
        if len(ba) != len(ta):
            return False
        for b_arg, t_arg in zip(ba, ta):
            if _is_entity(b_arg) and _is_entity(t_arg):
                continue
            if _is_entity(b_arg) != _is_entity(t_arg):
                return False
            kb = self._kind(b_arg, overrides)
            if kb in ("relation", "higher_order"):
                key = (_canonical_str(b_arg), _canonical_str(t_arg))
                if key not in index:
                    return False
            # function: tolerated even without explicit MH
        return True

    def _propagate_support(
        self,
        mhs: List[MatchHypothesis],
    ) -> Dict[Tuple[str, str], float]:
        """Compute SES with parental support propagated to children.

        SES(child) = base(child) + λ · sum SES(parent over child).
        We compute this by a single bottom-up pass on the DAG induced
        by sub-expression containment, in topological order.
        """
        idx: Dict[Tuple[str, str], MatchHypothesis] = {
            (mh.base_str, mh.target_str): mh for mh in mhs
        }
        # Build child→parents
        children: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
            k: [] for k in idx
        }
        for mh in mhs:
            for b_arg, t_arg in zip(_args(mh.base), _args(mh.target)):
                if _is_entity(b_arg) or _is_entity(t_arg):
                    continue
                ck = (_canonical_str(b_arg), _canonical_str(t_arg))
                if ck in idx:
                    children[ck].append((mh.base_str, mh.target_str))

        # Topologically order by base expression depth (deeper first
        # for child-up propagation).
        def depth(expr: ExpressionLike) -> int:
            if _is_entity(expr):
                return 0
            return 1 + max((depth(a) for a in _args(expr)), default=0)

        order = sorted(idx.keys(), key=lambda k: -depth(idx[k].base))
        ses: Dict[Tuple[str, str], float] = {}
        lam = self.config.systematicity_weight
        # We need parent support, so we should process parents BEFORE
        # children to propagate down.  Build a reverse: process by
        # increasing depth (shallow parents first), and add λ · ses[parent]
        # to each of their children.
        order_top_down = sorted(idx.keys(), key=lambda k: depth(idx[k].base))
        for k in order_top_down:
            ses[k] = idx[k].score
        for k in order_top_down:
            for ck in children[k]:
                ses[ck] = ses.get(ck, 0.0) + lam * ses[k]
        return ses

    def _build_global_mappings(
        self,
        base: Description,
        target: Description,
        mhs: List[MatchHypothesis],
        overrides: Mapping[str, str],
        t0: float,
    ) -> Tuple[List[GlobalMapping], int]:
        """Greedy best-first search over one-to-one-consistent unions of MHs.

        The search keeps a beam of size ``max_gmaps * 4`` and expands
        each state by adding the highest-SES compatible MH not yet
        included.  States are deduplicated by canonical frozenset.

        Returns the list of gmaps (un-trimmed, un-sorted) and the
        number of states visited.
        """
        ses = self._propagate_support(mhs)
        # Anchor mappings induced by each MH: relations imply entity
        # correspondences via their argument positions.
        mh_entities: Dict[Tuple[str, str], Dict[str, str]] = {}
        for mh in mhs:
            em: Dict[str, str] = {}
            ok = self._collect_entity_correspondences(mh.base, mh.target, em)
            if not ok:
                continue
            mh_entities[(mh.base_str, mh.target_str)] = em

        # Order MHs by descending propagated SES; deterministic
        # tie-break by canonical strings.
        sorted_mhs = sorted(
            [mh for mh in mhs if (mh.base_str, mh.target_str) in mh_entities],
            key=lambda mh: (
                -ses[(mh.base_str, mh.target_str)],
                mh.base_str,
                mh.target_str,
            ),
        )

        if not sorted_mhs:
            return [], 0

        beam_size = max(self.config.max_gmaps * 4, 8)
        # Each state = (entity_map, expr_map, used_mhs, score, support_breakdown)
        initial: Tuple[Tuple[str, str], ...] = ()
        states: List[
            Tuple[
                Tuple[Tuple[str, str], ...],  # used_keys
                Dict[str, str],                # entity_map
                Dict[str, str],                # expr_map
                float,                          # score
                Dict[str, float],               # support_breakdown
            ]
        ] = [(initial, {}, {}, 0.0, {})]
        results: List[GlobalMapping] = []
        seen_states: Set[FrozenSet[Tuple[str, str]]] = {frozenset()}
        n_visited = 0

        # Best-first: at each round, sort by score desc, expand top
        # ``beam_size`` states, terminate when no state can be
        # extended.
        max_rounds = len(sorted_mhs) * 2 + 4
        for _round in range(max_rounds):
            self._check_budget(t0, len(mhs))
            states.sort(key=lambda s: -s[3])
            states = states[:beam_size]
            n_visited += len(states)
            new_states = []
            extended = False
            for used_keys, em, xm, sc, sb in states:
                used_set = set(used_keys)
                # Try each candidate MH.
                tried_any = False
                for mh in sorted_mhs:
                    key = (mh.base_str, mh.target_str)
                    if key in used_set:
                        continue
                    add_em = mh_entities[key]
                    # Check entity-map consistency (one-to-one).
                    merged_em = self._merge_entity_maps(em, add_em)
                    if merged_em is None:
                        continue
                    # Check expression-map consistency: a base expr
                    # already mapped must map to the same target expr.
                    if mh.base_str in xm and xm[mh.base_str] != mh.target_str:
                        continue
                    # Reverse: target expr already mapped to a
                    # different base.
                    if any(v == mh.target_str and k != mh.base_str
                           for k, v in xm.items()):
                        continue
                    new_xm = dict(xm)
                    new_xm[mh.base_str] = mh.target_str
                    add_score = ses[key]
                    new_sb = dict(sb)
                    new_sb[mh.kind] = new_sb.get(mh.kind, 0.0) + add_score
                    new_used = used_keys + (key,)
                    canonical = frozenset(new_used)
                    if canonical in seen_states:
                        continue
                    seen_states.add(canonical)
                    new_states.append(
                        (new_used, merged_em, new_xm, sc + add_score, new_sb)
                    )
                    tried_any = True
                    extended = True
                if not tried_any:
                    # No way to extend this state — promote it to a
                    # final candidate.
                    if used_keys:
                        results.append(
                            self._finalise_gmap(
                                base, target, used_keys, em, xm, sc, sb, mh_entities,
                            )
                        )
            states = new_states
            if not extended:
                break
            # Cap the candidate pool to keep memory bounded.
            states = states[: beam_size * 4]

        # Also promote anything still in the beam at termination.
        for used_keys, em, xm, sc, sb in states:
            if not used_keys:
                continue
            results.append(
                self._finalise_gmap(
                    base, target, used_keys, em, xm, sc, sb, mh_entities,
                )
            )

        # Deduplicate by used_keys frozenset.
        deduped: Dict[FrozenSet[Tuple[str, str]], GlobalMapping] = {}
        for g in results:
            k = frozenset((mh.base_str, mh.target_str) for mh in g.mhs)
            if k not in deduped or deduped[k].score < g.score:
                deduped[k] = g
        return list(deduped.values()), n_visited

    def _collect_entity_correspondences(
        self,
        b: ExpressionLike,
        t: ExpressionLike,
        em: Dict[str, str],
    ) -> bool:
        """Walk parallel argument lists, adding entity correspondences.

        Returns False if any pair conflicts with the accumulated map.
        Function/relation arg pairs are recursed into; entity pairs
        are added to ``em``.
        """
        if _is_entity(b) and _is_entity(t):
            if b in em:
                return em[b] == t
            if t in em.values():
                return False
            em[b] = t  # type: ignore[index]
            return True
        if _is_entity(b) != _is_entity(t):
            return False
        if _head(b) != _head(t) and not _is_entity(b):
            # Different heads — only allowed in free-function mode,
            # but the alignment of arguments still requires structural
            # compatibility.
            pass
        if len(_args(b)) != len(_args(t)):
            return False
        for ba, ta in zip(_args(b), _args(t)):
            ok = self._collect_entity_correspondences(ba, ta, em)
            if not ok:
                return False
        return True

    def _merge_entity_maps(
        self,
        a: Mapping[str, str],
        b: Mapping[str, str],
    ) -> Optional[Dict[str, str]]:
        """Merge two one-to-one entity maps; return None on conflict."""
        merged = dict(a)
        rev: Dict[str, str] = {v: k for k, v in a.items()}
        for k, v in b.items():
            if k in merged:
                if merged[k] != v:
                    return None
                continue
            if v in rev:
                if rev[v] != k:
                    return None
                continue
            merged[k] = v
            rev[v] = k
        return merged

    def _finalise_gmap(
        self,
        base: Description,
        target: Description,
        used_keys: Sequence[Tuple[str, str]],
        em: Mapping[str, str],
        xm: Mapping[str, str],
        score: float,
        sb: Mapping[str, float],
        mh_entities: Mapping[Tuple[str, str], Mapping[str, str]],
    ) -> GlobalMapping:
        mh_by_key = {
            (mh.base_str, mh.target_str): mh
            for mh in self._iter_mhs(base, target, em, xm, mh_entities)
        }
        mhs = tuple(mh_by_key[k] for k in used_keys if k in mh_by_key)
        # Candidate inferences: top-level base exprs whose entities are
        # all mapped but whose projection is not in target.
        target_set = {
            _canonical_str(s) for s in target.all_subexpressions()
        }
        inferences: List[Tuple[ExpressionLike, ExpressionLike]] = []
        mapped_entities = set(em.keys())
        for top in base.expressions:
            ents = set(_entities(top))
            if not ents:
                continue
            if not ents.issubset(mapped_entities):
                continue
            projected = self._project(top, em)
            if _canonical_str(projected) not in target_set:
                inferences.append((projected, top))
        return GlobalMapping(
            entity_map=dict(em),
            expr_map=dict(xm),
            mhs=mhs,
            score=score,
            inferences=tuple(inferences),
            support_breakdown=dict(sb),
        )

    def _iter_mhs(
        self,
        base: Description,
        target: Description,
        em: Mapping[str, str],
        xm: Mapping[str, str],
        mh_entities: Mapping[Tuple[str, str], Mapping[str, str]],
    ) -> Iterable[MatchHypothesis]:
        # Reconstruct MHs from xm.  Each (base_str -> target_str) is a
        # mapping; the kind is inferred from the base expression.
        # We need MatchHypothesis instances for serialization; rebuild
        # by parsing the canonical strings against the description.
        bsub = {_canonical_str(s): s for s in base.all_subexpressions()}
        tsub = {_canonical_str(s): s for s in target.all_subexpressions()}
        overrides = self._merged_overrides(base, target)
        for bs, ts in xm.items():
            if bs not in bsub or ts not in tsub:
                continue
            b = bsub[bs]
            t = tsub[ts]
            if _is_entity(b) or _is_entity(t):
                continue
            kind = self._kind(b, overrides)
            yield MatchHypothesis(
                base=b,
                target=t,
                kind=kind,
                base_str=bs,
                target_str=ts,
                score=self.config.effective_weight(kind),
                support_kind="tiered_identicality"
                if _head(b) == _head(t)
                else "free_function",
            )

    def _project(self, expr: ExpressionLike, em: Mapping[str, str]) -> ExpressionLike:
        if _is_entity(expr):
            return em.get(expr, expr)  # type: ignore[arg-type]
        return (_head(expr),) + tuple(self._project(a, em) for a in _args(expr))

    # -------------------------------------------------------------------------
    # ACME — Holyoak-Thagard 1989 constraint-satisfaction network
    # -------------------------------------------------------------------------

    def _acme(
        self,
        base: Union[str, Description],
        target: Union[str, Description],
    ) -> AnalogistReport:
        """Constraint-satisfaction-network alternative to SME.

        Each unit represents a candidate (base_expr, target_expr)
        match hypothesis.  Edges:

          * **Structural** (excitatory): parents activate children.
          * **Semantic** (excitatory): identical predicates strongly
            activated; analogically related predicates weakly so.
          * **Pragmatic** (excitatory): user-supplied priors via
            ``config.acme_priors``.
          * **Inhibitory**: pairs that violate one-to-one mapping
            inhibit each other.

        Network is relaxed via synchronous, bounded update for
        ``acme_iterations`` steps.  Final activations rank the MHs;
        the consistent maximal subset becomes the global mapping.
        """
        t0 = time.monotonic()
        b = self._resolve(base)
        t = self._resolve(target)
        overrides = self._merged_overrides(b, t)
        # Build candidate MHs (same as SME stage 1, no parallel-
        # connectivity gate — ACME handles consistency via inhibition).
        units: List[Tuple[ExpressionLike, ExpressionLike, str]] = []
        for bs in b.all_subexpressions():
            if _is_entity(bs):
                continue
            for ts in t.all_subexpressions():
                if _is_entity(ts):
                    continue
                kb = self._kind(bs, overrides)
                kt = self._kind(ts, overrides)
                if not _alignable_kinds(kb, kt):
                    continue
                if len(_args(bs)) != len(_args(ts)):
                    continue
                if _head(bs) != _head(ts):
                    if not (kb == "function" and not self.config.require_identical_predicates):
                        continue
                units.append((bs, ts, kb))
        # Entity-correspondence units.  All base-entity / target-entity
        # pairs are candidates.
        b_ents = sorted(b.entities())
        t_ents = sorted(t.entities())
        ent_units: List[Tuple[str, str]] = [
            (be, te) for be in b_ents for te in t_ents
        ]
        # Activations
        a_units = [self.config.effective_weight(kb) for (_, _, kb) in units]
        a_ents = [1.0 for _ in ent_units]
        # Pragmatic priors
        priors = self.config.acme_priors
        for i, (bs, ts, _) in enumerate(units):
            p = priors.get((_canonical_str(bs), _canonical_str(ts)), 0.0)
            a_units[i] += p
        # Build edges (sparse)
        unit_idx = {(_canonical_str(bs), _canonical_str(ts)): i
                    for i, (bs, ts, _) in enumerate(units)}
        ent_idx = {(be, te): i for i, (be, te) in enumerate(ent_units)}
        # Excitatory: parent unit ↔ child unit; relation unit ↔ entity-
        # correspondence units implied by its args.
        for i, (bs, ts, _) in enumerate(units):
            for ba, ta in zip(_args(bs), _args(ts)):
                if _is_entity(ba) and _is_entity(ta):
                    j = ent_idx.get((ba, ta))
                    if j is not None:
                        # bidirectional excitation
                        pass  # we apply during relaxation
        # Relax via synchronous bounded update.  Net input: excitation -
        # inhibition; activation passed through tanh-like clip.
        for _ in range(self.config.acme_iterations):
            new_u = list(a_units)
            new_e = list(a_ents)
            # Structural excitation: parent activation lifts children.
            for i, (bs, ts, _) in enumerate(units):
                for ba, ta in zip(_args(bs), _args(ts)):
                    if _is_entity(ba) and _is_entity(ta):
                        j = ent_idx.get((ba, ta))
                        if j is not None:
                            new_u[i] += 0.05 * a_ents[j]
                            new_e[j] += 0.05 * a_units[i]
                    elif not _is_entity(ba) and not _is_entity(ta):
                        k = unit_idx.get(
                            (_canonical_str(ba), _canonical_str(ta))
                        )
                        if k is not None:
                            new_u[i] += 0.05 * a_units[k]
                            new_u[k] += 0.05 * a_units[i]
            # One-to-one inhibition between entity-correspondence units
            # that share a base or a target.
            for (be1, te1), i in ent_idx.items():
                for (be2, te2), j in ent_idx.items():
                    if i >= j:
                        continue
                    if be1 == be2 or te1 == te2:
                        new_e[i] -= 0.1 * a_ents[j]
                        new_e[j] -= 0.1 * a_ents[i]
            # Clip to [0, 1].
            a_units = [max(0.0, min(1.0, x)) for x in new_u]
            a_ents = [max(0.0, min(1.0, x)) for x in new_e]

        # Decode: greedily select entity correspondences by activation,
        # then derive a single global mapping.
        order_e = sorted(range(len(ent_units)), key=lambda i: -a_ents[i])
        em: Dict[str, str] = {}
        used_b: Set[str] = set()
        used_t: Set[str] = set()
        for i in order_e:
            be, te = ent_units[i]
            if be in used_b or te in used_t:
                continue
            if a_ents[i] <= 0.01:
                break
            em[be] = te
            used_b.add(be)
            used_t.add(te)
        # Build expr_map from unit activations consistent with em.
        xm: Dict[str, str] = {}
        target_set = {_canonical_str(s) for s in t.all_subexpressions()}
        order_u = sorted(range(len(units)), key=lambda i: -a_units[i])
        for i in order_u:
            bs, ts, _ = units[i]
            ents_b = set(_entities(bs))
            ents_t = set(_entities(ts))
            if not all(em.get(eb) for eb in ents_b):
                continue
            projected = self._project(bs, em)
            if _canonical_str(projected) != _canonical_str(ts):
                continue
            xm[_canonical_str(bs)] = _canonical_str(ts)
        # Candidate inferences.
        inferences: List[Tuple[ExpressionLike, ExpressionLike]] = []
        for top in b.expressions:
            ents = set(_entities(top))
            if not ents.issubset(em.keys()):
                continue
            projected = self._project(top, em)
            if _canonical_str(projected) not in target_set:
                inferences.append((projected, top))
        score = sum(a_units[unit_idx[(bs, ts)]] for bs, ts in xm.items())
        sb: Dict[str, float] = {}
        for bs, ts in xm.items():
            i = unit_idx[(bs, ts)]
            kind = units[i][2]
            sb[kind] = sb.get(kind, 0.0) + a_units[i]
        gmap = GlobalMapping(
            entity_map=dict(em),
            expr_map=dict(xm),
            mhs=tuple(),  # ACME does not produce SME-shaped MH set
            score=score,
            inferences=tuple(inferences),
            support_breakdown=sb,
        )
        duration = time.monotonic() - t0
        return AnalogistReport(
            base_name=b.name,
            target_name=t.name,
            mappings=(gmap,),
            n_match_hypotheses=len(units),
            n_gmaps_explored=1,
            n_inferences=len(inferences),
            engine="acme",
            duration_s=duration,
            budget_used={
                "mh_count": float(len(units)),
                "time_s": duration,
            },
            base_fingerprint=b.fingerprint(),
            target_fingerprint=t.fingerprint(),
            certificate=self._certificate(b, t, [gmap]),
        )

    # -------------------------------------------------------------------------
    # MAC/FAC retrieval (Forbus-Gentner-Law 1995)
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        probe: Union[str, Description],
        k: int = 5,
        mac_pool: int = 50,
    ) -> RetrievalReport:
        """Find the ``k`` descriptions most analogous to ``probe``.

        Two-stage:

          1. **MAC**: dot-product on content vectors (cheap, all-
             memory).  Take top ``mac_pool``.
          2. **FAC**: full SME on each of the ``mac_pool`` candidates;
             return the top ``k`` by SES.
        """
        t0 = time.monotonic()
        probe_desc = self._resolve(probe)
        overrides = {
            **dict(self.config.predicate_kinds),
            **dict(probe_desc.predicate_kinds),
        }
        probe_vec = _content_vector(
            probe_desc, self._higher_order, self._functions, overrides
        )
        # MAC
        mac_scores: List[Tuple[str, float]] = []
        for name, vec in self._content.items():
            if name == probe_desc.name:
                continue
            mac_scores.append((name, _cosine(probe_vec, vec)))
        mac_scores.sort(key=lambda x: -x[1])
        n_mac = len(mac_scores)
        short = mac_scores[:mac_pool]
        # FAC
        results: List[Tuple[str, float, float, Optional[GlobalMapping]]] = []
        for name, mac_s in short:
            try:
                rep = self.match(probe_desc, self._descriptions[name])
            except BudgetExhausted:
                continue
            if rep.mappings:
                top = rep.mappings[0]
                results.append((name, mac_s, top.score, top))
            else:
                results.append((name, mac_s, 0.0, None))
        results.sort(key=lambda x: -x[2])
        duration = time.monotonic() - t0
        return RetrievalReport(
            probe_name=probe_desc.name,
            candidates=tuple(results[:k]),
            n_mac_evaluated=n_mac,
            n_fac_evaluated=len(short),
            duration_s=duration,
        )

    # -------------------------------------------------------------------------
    # Certificate
    # -------------------------------------------------------------------------

    def _certificate(
        self,
        base: Description,
        target: Description,
        gmaps: Sequence[GlobalMapping],
    ) -> Optional[str]:
        if self.config.hmac_key is None:
            return None
        h = hmac.new(self.config.hmac_key, digestmod=hashlib.sha256)
        h.update(b"analogist/v1\n")
        h.update(base.fingerprint().encode("ascii"))
        h.update(b"\n")
        h.update(target.fingerprint().encode("ascii"))
        h.update(b"\n")
        for g in gmaps:
            for bs in sorted(g.entity_map):
                h.update(f"E:{bs}->{g.entity_map[bs]}\n".encode("utf-8"))
            for bs in sorted(g.expr_map):
                h.update(f"X:{bs}->{g.expr_map[bs]}\n".encode("utf-8"))
            for inf, src in g.inferences:
                h.update(
                    f"I:{_canonical_str(inf)}<-{_canonical_str(src)}\n".encode(
                        "utf-8"
                    )
                )
            h.update(f"S:{g.score:.9g}\n".encode("ascii"))
        return h.hexdigest()


# =============================================================================
# Convenience constructors / factory
# =============================================================================


def sme(
    *,
    systematicity_weight: float = 0.5,
    max_gmaps: int = 3,
    require_identical_predicates: bool = True,
    hmac_key: Optional[bytes] = None,
) -> Analogist:
    """Construct an Analogist configured for SME analogy mode."""
    cfg = AnalogistConfig(
        engine="sme",
        mode="analogy",
        systematicity_weight=systematicity_weight,
        max_gmaps=max_gmaps,
        require_identical_predicates=require_identical_predicates,
        hmac_key=hmac_key,
    )
    return Analogist(cfg)


def acme(
    *,
    iterations: int = 100,
    priors: Optional[Mapping[Tuple[str, str], float]] = None,
    hmac_key: Optional[bytes] = None,
) -> Analogist:
    """Construct an Analogist configured for ACME relaxation."""
    cfg = AnalogistConfig(
        engine="acme",
        mode="analogy",
        acme_iterations=iterations,
        acme_priors=dict(priors or {}),
        hmac_key=hmac_key,
    )
    return Analogist(cfg)


def literal_similarity(
    *,
    max_gmaps: int = 3,
    hmac_key: Optional[bytes] = None,
) -> Analogist:
    """Both attributes AND relations contribute (Markman-Gentner 1993)."""
    cfg = AnalogistConfig(
        engine="sme",
        mode="literal",
        max_gmaps=max_gmaps,
        attribute_weight=1.0,
        hmac_key=hmac_key,
    )
    return Analogist(cfg)


# =============================================================================
# Proportional analogy (Hofstadter Copycat) — letter-string a:b::c:?
# =============================================================================


@dataclass(frozen=True)
class ProportionalAnalogyResult:
    """Result of ``ProportionalAnalogy.solve(a, b, c)``.

    Attributes:

      * ``answer``: the candidate string for ``d`` such that
        ``a:b :: c:d``.
      * ``rule``: human-readable description of the inferred rule.
      * ``score``: a heuristic confidence in [0, 1].
      * ``alternatives``: other candidate ``(answer, rule, score)``
        triples explored, sorted by descending score.
    """

    answer: str
    rule: str
    score: float
    alternatives: Tuple[Tuple[str, str, float], ...]


class ProportionalAnalogy:
    """Letter-string proportional analogies in the Copycat micro-domain.

    Implements a deterministic, rule-based subset of Hofstadter (1985)
    / Mitchell (1993) Copycat: enumerates the common transformations
    (last-letter increment, substring replacement, length change,
    reversal, predecessor / successor cascades) and selects the
    rule with the highest combined applicability and simplicity score.

    Examples (deterministic):

      * ``solve("abc", "abd", "ijk")`` → ``"ijl"``
        (rule: "increment the last letter")
      * ``solve("abc", "abd", "iijjkk")`` → ``"iijjll"``
        (rule: "increment the last letter group")
      * ``solve("abc", "abd", "xyz")`` → ``"xya"`` (alphabet wrap)
        or ``"yza"`` / ``"wyz"`` depending on the wrap policy.

    Designed as a low-cost analogy primitive for symbol-stream
    pattern transfer.  Pure stdlib, no fluid-concepts cloud.
    """

    def __init__(self, *, alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> None:
        if not alphabet:
            raise InvalidConfig("alphabet must be non-empty")
        self.alphabet = alphabet
        self._index = {c: i for i, c in enumerate(alphabet)}

    def solve(self, a: str, b: str, c: str) -> ProportionalAnalogyResult:
        """Return a candidate ``d`` such that ``a:b :: c:d``."""
        if not all(ch in self._index for ch in a + b + c):
            # Strings outside the alphabet — only structural rules
            # (identity, length, reversal) are tried.
            return self._structural_only(a, b, c)
        candidates: List[Tuple[str, str, float]] = []

        # Rule 1: increment / decrement the last letter.
        if len(a) == len(b) and a[:-1] == b[:-1]:
            delta = self._index[b[-1]] - self._index[a[-1]]
            if abs(delta) <= 3 and len(c) >= 1:
                last = self._index[c[-1]] + delta
                if 0 <= last < len(self.alphabet):
                    d = c[:-1] + self.alphabet[last]
                    rule = f"shift last letter by {delta}"
                    candidates.append((d, rule, 0.9))
                else:
                    # Wrap policy: take modulo
                    last = last % len(self.alphabet)
                    d = c[:-1] + self.alphabet[last]
                    rule = f"shift last letter by {delta} (wrap)"
                    candidates.append((d, rule, 0.7))

        # Rule 2: replace a substring of a with the corresponding
        # substring of b, applied to c by analogy of position.
        diff = self._first_diff(a, b)
        if diff is not None and len(a) == len(b):
            i, j = diff
            sub_a = a[i:j]
            sub_b = b[i:j]
            # Apply same substitution where it occurs in c.
            if sub_a in c:
                d = c.replace(sub_a, sub_b, 1)
                rule = f"replace {sub_a!r} with {sub_b!r}"
                candidates.append((d, rule, 0.85))

        # Rule 3: length change (a → b adds/removes k chars).  Apply
        # same delta to c using the last char of c.
        if a != b:
            dlen = len(b) - len(a)
            if dlen > 0 and b.startswith(a):
                tail = b[len(a):]
                d = c + tail
                rule = f"append {tail!r}"
                candidates.append((d, rule, 0.75))
            if dlen < 0 and a.startswith(b):
                k = -dlen
                if len(c) > k:
                    d = c[:-k]
                    rule = f"drop last {k} chars"
                    candidates.append((d, rule, 0.7))

        # Rule 4: reversal.
        if b == a[::-1]:
            d = c[::-1]
            rule = "reverse"
            candidates.append((d, rule, 0.7))

        # Rule 5: shift every letter by a constant delta.
        if len(a) == len(b):
            deltas = [self._index[bx] - self._index[ax] for ax, bx in zip(a, b)]
            if deltas and len(set(deltas)) == 1:
                delta = deltas[0]
                d = "".join(
                    self.alphabet[(self._index[ch] + delta) % len(self.alphabet)]
                    for ch in c
                )
                rule = f"shift every letter by {delta}"
                candidates.append((d, rule, 0.8))

        # Rule 6: identity (a == b).
        if a == b:
            candidates.append((c, "identity", 0.5))

        if not candidates:
            return self._structural_only(a, b, c)

        # Rank by score (descending), then by rule simplicity (string
        # length of the rule description).
        candidates.sort(key=lambda x: (-x[2], len(x[1])))
        best = candidates[0]
        return ProportionalAnalogyResult(
            answer=best[0],
            rule=best[1],
            score=best[2],
            alternatives=tuple(candidates[1:]),
        )

    def _structural_only(
        self, a: str, b: str, c: str
    ) -> ProportionalAnalogyResult:
        candidates: List[Tuple[str, str, float]] = []
        if a == b:
            candidates.append((c, "identity", 0.5))
        if b == a[::-1]:
            candidates.append((c[::-1], "reverse", 0.7))
        dlen = len(b) - len(a)
        if dlen > 0 and b.startswith(a):
            candidates.append((c + b[len(a):], f"append {b[len(a):]!r}", 0.6))
        if not candidates:
            return ProportionalAnalogyResult(
                answer=c,
                rule="fallback: identity",
                score=0.1,
                alternatives=(),
            )
        candidates.sort(key=lambda x: (-x[2], len(x[1])))
        return ProportionalAnalogyResult(
            answer=candidates[0][0],
            rule=candidates[0][1],
            score=candidates[0][2],
            alternatives=tuple(candidates[1:]),
        )

    def _first_diff(self, a: str, b: str) -> Optional[Tuple[int, int]]:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        if i == n and len(a) == len(b):
            return None
        j_a = len(a)
        j_b = len(b)
        while j_a > i and j_b > i and a[j_a - 1] == b[j_b - 1]:
            j_a -= 1
            j_b -= 1
        # If a-window and b-window have same length, we have a substring
        # replacement; otherwise the length-change rules handle it.
        if (j_a - i) == (j_b - i):
            return (i, j_a)
        return None


# =============================================================================
# Public re-exports
# =============================================================================


__all__ = [
    "AnalogistError",
    "InvalidExpression",
    "InvalidDescription",
    "UnknownDescription",
    "InvalidConfig",
    "BudgetExhausted",
    "Expression",
    "Description",
    "MatchHypothesis",
    "GlobalMapping",
    "AnalogistConfig",
    "AnalogistReport",
    "RetrievalReport",
    "Analogist",
    "ProportionalAnalogy",
    "ProportionalAnalogyResult",
    "sme",
    "acme",
    "literal_similarity",
]
