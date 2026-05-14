r"""Negotiator — multi-party allocation as a runtime primitive.

Every coordination engine that serves more than one consumer eventually
has to answer a question that pricing alone can't: when N parties
compete for the same finite resource — capacity, budget, GPU-seconds,
attention, refund pool — and each party has its own utility for
receiving more, **what split is the right split?** Throwing more
money at the highest bidder is not "the right answer"; it is one
answer (utilitarian) out of a family with very different normative
content. The literature has been precise about this for seventy
years, and the runtime should be too.

The Negotiator is the primitive that turns the rest of the stack
into an actual marketplace. It accepts (i) a registered set of
parties with their utility functions and disagreement points,
(ii) one or more resource constraints, and (iii) a *solution
concept* — one of {utilitarian, egalitarian, leximin, Nash, Kalai-
Smorodinsky, proportional-fair, VCG} — and returns an allocation
that is the unique fixed point of that concept on the given problem,
together with a tamper-evident certificate of which axioms it
satisfies.

Mathematical roots
------------------

  * **Nash, 1950 — The Bargaining Problem.** Defines the unique
    allocation `x` that maximises the *Nash product*
    `Π_i (u_i(x_i) − d_i)` over the feasible utility set,
    subject to `u_i(x_i) ≥ d_i`. Axiomatised by *Pareto optimality*,
    *symmetry*, *invariance to affine transformation*, and
    *independence of irrelevant alternatives* (IIA). For convex
    feasible sets this is the unique satisfying allocation.

  * **Kalai & Smorodinsky, 1975 — Other Solutions to Nash's
    Bargaining Problem.** Replaces IIA by *monotonicity*: when the
    ideal point `u_i^*` of player i increases, player i's share
    cannot decrease. The resulting solution picks the point on the
    Pareto frontier with maximum proportional progress toward the
    ideal — equivalently, the intersection of the frontier with the
    ray from `d` to `u^*`.

  * **Rawls, 1971; Sen, 1970 — Egalitarian welfare.** Among Pareto-
    optimal allocations, choose the one that maximises the minimum
    utility. The lexicographic refinement (Sen 1970 §3.1) extends
    this to a unique outcome: tie-break by maximising the second-
    smallest, then the third, ad infinitum. *Leximin* is the only
    selection rule simultaneously satisfying Pareto efficiency,
    anonymity, and a strong form of equity (Sen-Hammond, 1976).

  * **Kelly, Maulloo & Tan, 1998 — Proportional fairness for
    networks.** Maximises `Σ log u_i(x_i)`. For linear utilities and
    a sum-budget constraint, the solution coincides with the Nash
    bargaining solution at the origin (d = 0). For concave utilities
    the proportional-fair allocation is the unique stable point of
    distributed gradient-ascent on `log u`, which is why TCP-style
    congestion control converges to it.

  * **Vickrey, 1961; Clarke, 1971; Groves, 1973 — VCG mechanism.**
    For indivisible goods or discrete allocations, the unique
    welfare-maximising allocation paired with externality payments —
    each agent pays the surplus the *rest of the world* would have
    won had they not participated. The mechanism is *truthful*:
    bidding one's true valuation is a (weakly) dominant strategy.
    The runtime story: when a tenant requests indivisible capacity
    (e.g. exclusive use of a model), VCG is how to allocate and
    bill so no tenant can game by misreporting.

  * **Foley, 1967 — Envy-free allocation.** An allocation `x` is
    *envy-free* iff for every pair `(i, j)`, party i prefers her own
    bundle to j's: `u_i(x_i) ≥ u_i(x_j)`. Envy-freeness is logically
    independent of efficiency; combined with Pareto-efficiency,
    existence is non-trivial (Varian 1974). The Negotiator computes
    an envy-freeness certificate alongside every allocation so the
    coordinator can detect bad splits before they ship.

The seven solution concepts above are not preferences a coordinator
*picks among arbitrarily*: they are projections of the same Pareto
frontier under different axiomatisations, and each one is *the*
canonical answer when its axiom set fits the use case. The
Negotiator runtime exposes them as a uniform API and a uniform
receipt format precisely so that a coordination engine can pick by
constraint, not by aesthetic — utilitarian for cost minimisation,
leximin for SLA-critical multi-tenancy, Nash/KS for cooperative
splits, VCG for sealed-bid auctions.

The KKT-based water-filling solver
----------------------------------

For continuous concave utilities `u_i` with a sum-budget `Σ x_i ≤ B`
and per-party caps `x_i ∈ [0, m_i]`, every solution concept above
reduces to a one-dimensional root-find:

  * **Utilitarian:** find `λ` such that the active set
    `A(λ) = {i : u_i'(0) > λ, u_i'(m_i) < λ}` allocates exactly the
    remaining budget. KKT prescribes
    `u_i'(x_i^*) = λ` for `i ∈ A(λ)`, `x_i^* = 0` for the dual-
    infeasible, `x_i^* = m_i` for the cap-saturated. Bisecting `λ`
    over `[0, max_i u_i'(0)]` is monotone and converges geometrically.

  * **Egalitarian (max-min):** find the largest `c` such that
    `Σ u_i^{−1}(c) ≤ B`. Each `u_i^{−1}` is monotone non-decreasing,
    so the sum is too; bisect `c` over `[0, max_i u_i(m_i)]`.

  * **Leximin:** iterate the egalitarian solve. After finding `c_1`
    and the set `B_1` of parties at the floor, freeze those parties
    at `x_i = u_i^{−1}(c_1)`, subtract their resource, and re-solve
    on the residual. Each iteration removes at least one party, so
    the procedure terminates in `O(n)` egalitarian solves
    (Bertsekas-Gallager 1992 §6.5).

  * **Nash bargaining:** maximise `Σ log(u_i(x_i) − d_i)`. KKT gives
    `u_i'(x_i^*) = λ (u_i(x_i^*) − d_i)`. The single dual variable
    `λ` again pins down the solution; bisect.

  * **Kalai-Smorodinsky:** the unique `t ∈ (0, 1]` such that the
    point `u_i(x_i^*) = d_i + t (u_i^* − d_i)` is feasible and on
    the Pareto frontier. Bisect `t`; for each `t`, the target
    utility for party i is fully specified, and feasibility reduces
    to `Σ u_i^{−1}(target_i) ≤ B`.

  * **Proportional fair:** maximise `Σ log u_i(x_i)`. Identical KKT
    structure to Nash, with `d = 0`. For linear utilities this
    yields uniform allocation `x_i = B/n`, which is why
    proportional-fair is also called *equal-share fairness* in
    networking.

Bisection on a single scalar is `O(log(1/ε))` for ε-precision; the
inner cost is one `u_i^{−1}` or `u_i'` evaluation per party, both
`O(1)` for the utility families we ship (linear, piecewise-linear,
quadratic). Total cost is `O(n log(1/ε))` per concept.

What it composes (razor-sharp coordination integration)
------------------------------------------------------

  * **TicketMarket.** When the market has more pending tickets than
    available capacity, the Negotiator chooses *which* tickets get
    dispatched under a fairness criterion. Tiered (premium, standard,
    economy) market tickets become weighted parties; the egalitarian
    solver guarantees no tier is starved when the kernel is under
    load. The coordinator queries
    ``negotiator.allocate_egalitarian(budget=available_capacity)``
    on every market tick.

  * **TicketEconomist.** Refund pool allocation across breaching
    tenants is a cooperative game; Nash bargaining over refund
    amounts gives the unique split with the strongest fairness
    axioms (Pareto, symmetry, IIA, scale-invariance). The Economist
    calls ``negotiator.allocate_nash(...)`` when distributing a
    fixed refund budget across parties with differing breach
    severities.

  * **PortfolioOptimizer.** The portfolio frontier already exposes a
    Pareto curve; the Negotiator picks *which* point on the curve
    matches the policy. Leximin selects the most equitable;
    utilitarian selects the cheapest; KS selects the
    proportional-progress point. Composing them: the optimizer
    constructs the frontier, the negotiator picks the operating
    point.

  * **Coalition.** Coalition emits Shapley values — additive credits
    that sum to the grand surplus. The Negotiator takes those
    credits as `disagreement points` (the surplus each party would
    secure unilaterally) and allocates *additional* budget on top.
    Net result: Shapley splits the cake, Negotiator splits the
    bonus.

  * **Auditor & RiskController.** Multi-hypothesis budget allocation
    is a Negotiator problem: each test claims a slice of the α-budget
    based on its current evidence rate. Egalitarian gives every test
    equal floor; PF gives more to high-power tests; VCG gives a
    truthful elicitation when tests come from competing tenants.

  * **AttestationLedger.** Every allocation emits a
    ``negotiator.allocated`` receipt — a third-party-replayable
    proof that under utilities `u_i` and disagreement `d_i` at time
    `t`, party X received exactly `x_X^*` under solution concept Y,
    *and* the corresponding axiom certificate. This is the audit
    trail "fair allocation" requires when regulators ask.

  * **EventBus.** Streams every registration, budget change, and
    allocation. A higher-level coordination engine reacts in real
    time — e.g. trigger a re-allocation on every
    ``negotiator.budget_changed`` event, or on every
    ``coalition.credited`` event from upstream.

Where this slots in
-------------------

    neg = Negotiator(bus=bus, attestor=attestor)
    neg.register_party("tenant-premium",  LinearUtility(slope=2.0, cap=8.0))
    neg.register_party("tenant-standard", LinearUtility(slope=1.0, cap=8.0))
    neg.register_party("tenant-economy",  LinearUtility(slope=0.5, cap=8.0))
    neg.set_budget(10.0)

    util_report  = neg.allocate_utilitarian()       # max-welfare; serves premium fully
    egal_report  = neg.allocate_egalitarian()       # max-min; equal utility for all
    nash_report  = neg.allocate_nash()              # cooperative split
    ks_report    = neg.allocate_kalai_smorodinsky()
    lex_report   = neg.allocate_leximin()
    pf_report    = neg.allocate_proportional_fair()

    envy = neg.envy_freeness(util_report.allocation)
    assert envy.envy_free or util_report.is_pareto

Events
------
    negotiator.started            — kernel constructed
    negotiator.party_registered   — a party was added
    negotiator.party_removed      — a party was withdrawn
    negotiator.budget_changed     — total resource updated
    negotiator.allocated          — an allocation was computed
    negotiator.allocation_failed  — solver could not satisfy constraints
    negotiator.cleared            — state reset

Honest about limits
-------------------

  * Continuous solvers assume utilities are non-decreasing and
    concave on `[0, m_i]`. For non-concave utilities the bisection
    may converge to a local KKT point; we detect this by checking
    Pareto-dominance against a small grid sweep and raise
    ``NegotiationInfeasible`` when the certificate fails.
  * Numerical precision is ε = 1e-9 by default; bisect terminates
    when bracket width is below ε or after 200 iterations.
  * Disagreement points must be strictly less than the ideal for
    Nash and KS to be well-defined (otherwise log is `−∞`); we
    raise ``NegotiationInfeasible`` if the disagreement set has zero
    measure.
  * VCG assumes quasi-linear utilities and additive valuations
    across items; we implement the canonical case and raise on
    combinatorial bids that exceed the `2^|items|` enumeration cap
    (configurable, default 16).
  * Envy-freeness with continuous divisible goods always exists for
    `n ≥ 2` parties under additive utilities (Brams-Taylor 1995);
    with indivisible items it may not — we report the *maximum
    envy* and the receipt does not claim envy-freeness when it
    isn't achieved.
  * Multi-resource (vector) budgets are not yet supported by this
    initial release. The scalar-budget case covers the
    Market/Economist use case directly; multi-resource extension
    requires an interior-point method we deliberately defer.

Stdlib-only, CPU-bound, threadsafe; identical I/O surface to
`Coalition`, `Arbiter`, `Cartographer`, `Strategist` so the
coordination engine composes them uniformly.

Citations
---------

* Nash, J. F. (1950). The bargaining problem. *Econometrica*, 18(2),
  155-162.
* Kalai, E. & Smorodinsky, M. (1975). Other solutions to Nash's
  bargaining problem. *Econometrica*, 43(3), 513-518.
* Vickrey, W. (1961). Counterspeculation, auctions, and competitive
  sealed tenders. *Journal of Finance*, 16(1), 8-37.
* Clarke, E. H. (1971). Multipart pricing of public goods.
  *Public Choice*, 11, 17-33.
* Groves, T. (1973). Incentives in teams. *Econometrica*, 41(4),
  617-631.
* Foley, D. K. (1967). Resource allocation and the public sector.
  *Yale Economic Essays*, 7(1), 45-98.
* Sen, A. (1970). *Collective Choice and Social Welfare*. Holden-Day.
* Rawls, J. (1971). *A Theory of Justice*. Harvard University Press.
* Varian, H. R. (1974). Equity, envy and efficiency. *Journal of
  Economic Theory*, 9(1), 63-91.
* Kelly, F. P., Maulloo, A. K. & Tan, D. K. H. (1998). Rate control
  for communication networks: shadow prices, proportional fairness
  and stability. *J. Operational Research Society*, 49(3), 237-252.
* Brams, S. J. & Taylor, A. D. (1995). An envy-free cake division
  protocol. *American Mathematical Monthly*, 102(1), 9-18.
* Bertsekas, D. P. & Gallager, R. G. (1992). *Data Networks*, 2nd
  ed., Prentice Hall, §6.5.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds
# =====================================================================

NEGOTIATOR_STARTED = "negotiator.started"
NEGOTIATOR_PARTY_REGISTERED = "negotiator.party_registered"
NEGOTIATOR_PARTY_REMOVED = "negotiator.party_removed"
NEGOTIATOR_BUDGET_CHANGED = "negotiator.budget_changed"
NEGOTIATOR_ALLOCATED = "negotiator.allocated"
NEGOTIATOR_ALLOCATION_FAILED = "negotiator.allocation_failed"
NEGOTIATOR_CLEARED = "negotiator.cleared"


# =====================================================================
# Solution concepts
# =====================================================================

CONCEPT_UTILITARIAN = "utilitarian"
CONCEPT_EGALITARIAN = "egalitarian"
CONCEPT_LEXIMIN = "leximin"
CONCEPT_NASH = "nash"
CONCEPT_KALAI_SMORODINSKY = "kalai_smorodinsky"
CONCEPT_PROPORTIONAL_FAIR = "proportional_fair"
CONCEPT_VCG = "vcg"

KNOWN_CONCEPTS = (
    CONCEPT_UTILITARIAN,
    CONCEPT_EGALITARIAN,
    CONCEPT_LEXIMIN,
    CONCEPT_NASH,
    CONCEPT_KALAI_SMORODINSKY,
    CONCEPT_PROPORTIONAL_FAIR,
    CONCEPT_VCG,
)


# Axiom flags returned in NegotiationReport.certificate
AXIOM_PARETO = "pareto_optimal"
AXIOM_SYMMETRY = "symmetric"
AXIOM_AFFINE_INVARIANCE = "affine_invariant"
AXIOM_IIA = "iia"
AXIOM_MONOTONICITY = "monotonic"
AXIOM_LEXIMIN_EQUITY = "leximin_equity"
AXIOM_TRUTHFUL = "truthful"
AXIOM_ENVY_FREE = "envy_free"
AXIOM_BUDGET_BALANCED = "budget_balanced"

KNOWN_AXIOMS = (
    AXIOM_PARETO,
    AXIOM_SYMMETRY,
    AXIOM_AFFINE_INVARIANCE,
    AXIOM_IIA,
    AXIOM_MONOTONICITY,
    AXIOM_LEXIMIN_EQUITY,
    AXIOM_TRUTHFUL,
    AXIOM_ENVY_FREE,
    AXIOM_BUDGET_BALANCED,
)


_EPS = 1e-12
_BISECT_TOL = 1e-9
_BISECT_MAX_ITER = 200
_VCG_DEFAULT_BUNDLE_CAP = 16


# =====================================================================
# Errors
# =====================================================================


class NegotiationError(Exception):
    """Base class for Negotiator-level errors."""


class NegotiationInfeasible(NegotiationError):
    """Raised when no feasible allocation exists.

    Examples: budget < 0, every party hits its cap before budget is
    exhausted under egalitarian (would imply Σ caps < budget),
    disagreement points equal to or above the ideal point, etc.
    """


# =====================================================================
# Utility model
# =====================================================================


class Utility:
    """Abstract concave, non-decreasing utility on a bounded interval.

    Subclasses ship:

      * ``evaluate(x)`` → u(x)
      * ``derivative(x)`` → u'(x) (right-derivative at boundary)
      * ``inverse(u)`` → x such that u(x) = u, clipped to [0, cap]
      * ``cap`` → upper bound of admissible x
      * ``slope_at_zero`` → u'(0⁺) for KKT bracket
      * ``ideal`` → u(cap), used as Pareto-ideal anchor for KS
    """

    cap: float = 0.0
    slope_at_zero: float = 0.0
    ideal: float = 0.0

    def evaluate(self, x: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def derivative(self, x: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def inverse(self, u: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class LinearUtility(Utility):
    """u(x) = slope * x + intercept, clipped at cap.

    The cap models a saturation point: beyond `cap` no extra utility
    is delivered.  Slope must be non-negative.
    """

    slope: float
    cap: float
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if self.slope < 0:
            raise ValueError("LinearUtility.slope must be >= 0")
        if self.cap <= 0:
            raise ValueError("LinearUtility.cap must be > 0")
        if not math.isfinite(self.intercept):
            raise ValueError("LinearUtility.intercept must be finite")
        object.__setattr__(self, "slope_at_zero", float(self.slope))
        object.__setattr__(self, "ideal", float(self.slope * self.cap + self.intercept))

    def evaluate(self, x: float) -> float:
        x = max(0.0, min(float(x), self.cap))
        return self.slope * x + self.intercept

    def derivative(self, x: float) -> float:
        x = max(0.0, min(float(x), self.cap))
        # Strictly inside the cap, slope; at the cap, the right-deriv is 0.
        if x >= self.cap - _EPS:
            return 0.0
        return float(self.slope)

    def inverse(self, u: float) -> float:
        if self.slope <= _EPS:
            return 0.0 if u <= self.intercept + _EPS else self.cap
        x = (u - self.intercept) / self.slope
        return max(0.0, min(x, self.cap))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "slope": self.slope,
            "intercept": self.intercept,
            "cap": self.cap,
        }


@dataclass(frozen=True)
class QuadraticUtility(Utility):
    """u(x) = a*x − (b/2)*x² on [0, a/b], saturating at u_max = a²/(2b).

    Strictly concave when b > 0; equivalent to linear at b = 0 (which
    we forbid — use LinearUtility instead).
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("QuadraticUtility.a must be > 0")
        if self.b <= 0:
            raise ValueError("QuadraticUtility.b must be > 0; use LinearUtility for b=0")
        x_max = self.a / self.b
        object.__setattr__(self, "cap", float(x_max))
        object.__setattr__(self, "slope_at_zero", float(self.a))
        object.__setattr__(self, "ideal", float(self.a * x_max - 0.5 * self.b * x_max * x_max))

    def evaluate(self, x: float) -> float:
        x = max(0.0, min(float(x), self.cap))
        return self.a * x - 0.5 * self.b * x * x

    def derivative(self, x: float) -> float:
        x = max(0.0, min(float(x), self.cap))
        return max(0.0, self.a - self.b * x)

    def inverse(self, u: float) -> float:
        u = max(0.0, min(u, self.ideal))
        # Solve a*x - 0.5*b*x^2 = u for the smaller root in [0, cap].
        # 0.5*b*x^2 - a*x + u = 0 → x = (a - sqrt(a^2 - 2bu)) / b
        disc = self.a * self.a - 2.0 * self.b * u
        disc = max(disc, 0.0)
        x = (self.a - math.sqrt(disc)) / self.b
        return max(0.0, min(x, self.cap))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "quadratic", "a": self.a, "b": self.b}


@dataclass(frozen=True)
class PiecewiseLinearUtility(Utility):
    """u defined by breakpoints (x_0=0, u_0), (x_1, u_1), …, (x_K, u_K).

    Must be non-decreasing and concave (slopes weakly decreasing).
    """

    breakpoints: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        bps = tuple((float(x), float(u)) for x, u in self.breakpoints)
        if len(bps) < 2:
            raise ValueError("PiecewiseLinearUtility needs at least 2 breakpoints")
        if bps[0][0] != 0.0:
            raise ValueError("PiecewiseLinearUtility must start at x=0")
        for (x_a, u_a), (x_b, u_b) in zip(bps, bps[1:]):
            if x_b <= x_a:
                raise ValueError("breakpoint x-coords must be strictly increasing")
            if u_b < u_a - _EPS:
                raise ValueError("breakpoint u-coords must be non-decreasing")
        slopes = []
        for (x_a, u_a), (x_b, u_b) in zip(bps, bps[1:]):
            slopes.append((u_b - u_a) / (x_b - x_a))
        for s_a, s_b in zip(slopes, slopes[1:]):
            if s_b > s_a + 1e-9:
                raise ValueError("slopes must be weakly decreasing (concave)")
        object.__setattr__(self, "breakpoints", bps)
        object.__setattr__(self, "cap", float(bps[-1][0]))
        object.__setattr__(self, "slope_at_zero", float(slopes[0]) if slopes else 0.0)
        object.__setattr__(self, "ideal", float(bps[-1][1]))

    def _segment(self, x: float) -> int:
        for i, (xb, _) in enumerate(self.breakpoints[1:], start=1):
            if x <= xb + _EPS:
                return i - 1
        return len(self.breakpoints) - 2

    def evaluate(self, x: float) -> float:
        x = max(0.0, min(float(x), self.cap))
        i = self._segment(x)
        x_a, u_a = self.breakpoints[i]
        x_b, u_b = self.breakpoints[i + 1]
        if x_b - x_a <= _EPS:
            return u_a
        t = (x - x_a) / (x_b - x_a)
        return u_a + t * (u_b - u_a)

    def derivative(self, x: float) -> float:
        x = max(0.0, min(float(x), self.cap))
        if x >= self.cap - _EPS:
            return 0.0
        i = self._segment(x)
        x_a, u_a = self.breakpoints[i]
        x_b, u_b = self.breakpoints[i + 1]
        if x_b - x_a <= _EPS:
            return 0.0
        return (u_b - u_a) / (x_b - x_a)

    def inverse(self, u: float) -> float:
        u = max(0.0, min(u, self.ideal))
        for i in range(len(self.breakpoints) - 1):
            x_a, u_a = self.breakpoints[i]
            x_b, u_b = self.breakpoints[i + 1]
            if u <= u_b + _EPS:
                if u_b - u_a <= _EPS:
                    return x_a
                t = (u - u_a) / (u_b - u_a)
                return max(0.0, min(x_a + t * (x_b - x_a), self.cap))
        return self.cap

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "piecewise_linear",
            "breakpoints": [list(bp) for bp in self.breakpoints],
        }


# =====================================================================
# Dataclasses — request + response
# =====================================================================


@dataclass(frozen=True)
class PartySpec:
    """A registered party — a tenant, agent, or contributor.

    `weight` is an optional priority multiplier; the proportional-fair
    and Nash solvers maximise `Σ w_i log(u_i − d_i)`. A weight of 1 is
    standard. The egalitarian solver treats `weight` as the
    proportional floor multiplier (u_i / w_i equal across parties).

    `disagreement` is the utility the party secures *outside* the
    negotiation (BATNA in classical bargaining). It defines the lower
    bound below which the party walks away. Nash and KS require
    `disagreement < ideal`.

    `priority` is a hard-ordering key used by leximin tie-breaking;
    higher priority parties get earlier lex slots.
    """

    id: str
    utility: Utility
    disagreement: float = 0.0
    weight: float = 1.0
    priority: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "utility": self.utility.to_dict(),
            "disagreement": self.disagreement,
            "weight": self.weight,
            "priority": self.priority,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class Allocation:
    """The output of a single solver run.

    `assignments[party_id] = x_i` is the resource share given to party
    i. `utilities[party_id] = u_i(x_i)` is the realised utility. Sums
    to `total_allocated <= budget`.
    """

    assignments: dict[str, float]
    utilities: dict[str, float]
    total_allocated: float
    budget: float
    concept: str

    def slack(self) -> float:
        return max(0.0, self.budget - self.total_allocated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": dict(self.assignments),
            "utilities": dict(self.utilities),
            "total_allocated": self.total_allocated,
            "budget": self.budget,
            "concept": self.concept,
        }


@dataclass(frozen=True)
class EnvyReport:
    """Per-pair envy check for an allocation.

    `pair_envy[(i,j)] = max(0, u_i(x_j) - u_i(x_i))`. The allocation
    is envy-free iff every pair_envy entry is zero.
    """

    envy_free: bool
    max_envy: float
    pair_envy: dict[tuple[str, str], float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "envy_free": self.envy_free,
            "max_envy": self.max_envy,
            "pair_envy": {f"{a}|{b}": v for (a, b), v in self.pair_envy.items()},
        }


@dataclass(frozen=True)
class ParetoReport:
    """Pareto-optimality check against a probe grid.

    A test allocation Pareto-dominates the candidate iff
    ``u_i(test) >= u_i(cand)`` for all i, strict for at least one.
    `dominated` is True if the grid found any dominator.
    """

    dominated: bool
    max_dominance_gap: float
    dominator_concept: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NegotiationReport:
    """Result of one negotiation.

    The certificate is a list of axiom strings the allocation
    *provably* satisfies under the configured solution concept. The
    receipt_hash is non-empty iff an AttestationLedger was wired in.
    """

    id: str
    allocation: Allocation
    certificate: tuple[str, ...]
    envy: EnvyReport
    pareto: ParetoReport
    nash_product: float
    welfare: float
    min_utility: float
    elapsed_s: float
    iterations: int
    receipt_hash: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "allocation": self.allocation.to_dict(),
            "certificate": list(self.certificate),
            "envy": self.envy.to_dict(),
            "pareto": self.pareto.to_dict(),
            "nash_product": self.nash_product,
            "welfare": self.welfare,
            "min_utility": self.min_utility,
            "elapsed_s": self.elapsed_s,
            "iterations": self.iterations,
            "receipt_hash": self.receipt_hash,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class VCGAllocation:
    """Output of a VCG auction.

    `winners[item] = party_id` indicates which party won which item.
    `payments[party_id]` is the externality charged to each winning
    party (the welfare that would have accrued to the others had this
    party not participated minus the welfare they did accrue).
    """

    winners: dict[str, str]
    bundle: dict[str, tuple[str, ...]]
    payments: dict[str, float]
    welfare: float
    elapsed_s: float
    receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "winners": dict(self.winners),
            "bundle": {p: list(items) for p, items in self.bundle.items()},
            "payments": dict(self.payments),
            "welfare": self.welfare,
            "elapsed_s": self.elapsed_s,
            "receipt_hash": self.receipt_hash,
        }


# =====================================================================
# Solver math helpers
# =====================================================================


def _bisect(
    lo: float,
    hi: float,
    f: Callable[[float], float],
    *,
    tol: float = _BISECT_TOL,
    max_iter: int = _BISECT_MAX_ITER,
) -> tuple[float, int]:
    """Bisect a monotone non-increasing function f to a root.

    Returns (root, iterations).  Assumes f(lo) >= 0 >= f(hi); if not,
    returns the boundary that minimises |f|.
    """
    a, b = float(lo), float(hi)
    fa, fb = f(a), f(b)
    if fa <= 0:
        return a, 0
    if fb >= 0:
        return b, 0
    iters = 0
    while iters < max_iter and (b - a) > tol:
        m = 0.5 * (a + b)
        fm = f(m)
        if fm >= 0:
            a = m
            fa = fm
        else:
            b = m
            fb = fm
        iters += 1
    return 0.5 * (a + b), iters


def _bisect_increasing(
    lo: float,
    hi: float,
    f: Callable[[float], float],
    *,
    tol: float = _BISECT_TOL,
    max_iter: int = _BISECT_MAX_ITER,
) -> tuple[float, int]:
    """Bisect a monotone non-decreasing f to its root.

    Mirror of `_bisect`; assumes f(lo) <= 0 <= f(hi).
    """
    a, b = float(lo), float(hi)
    fa, fb = f(a), f(b)
    if fa >= 0:
        return a, 0
    if fb <= 0:
        return b, 0
    iters = 0
    while iters < max_iter and (b - a) > tol:
        m = 0.5 * (a + b)
        fm = f(m)
        if fm <= 0:
            a = m
            fa = fm
        else:
            b = m
            fb = fm
        iters += 1
    return 0.5 * (a + b), iters


def _water_fill_utilitarian(
    parties: Sequence[PartySpec],
    budget: float,
) -> tuple[dict[str, float], int]:
    """Solve the utilitarian water-filling.

    For each party i with concave utility u_i and cap m_i, find the
    dual λ such that the allocations
        x_i(λ) = min(m_i, max(0, (u_i')^{-1}(λ)))
    sum to budget. For LinearUtility, (u_i')^{-1}(λ) is +inf when
    λ < slope and 0 when λ > slope; the solver handles this by
    sorting parties by slope and filling greedily.
    """
    if budget <= 0:
        return {p.id: 0.0 for p in parties}, 0
    # If every party is linear, do the closed-form greedy fill.
    if all(isinstance(p.utility, LinearUtility) for p in parties):
        return _greedy_linear_fill(parties, budget), 0
    # Otherwise bisect on λ. λ-range: [0, max slope_at_zero].
    max_slope = max((p.utility.slope_at_zero for p in parties), default=0.0)
    if max_slope <= _EPS:
        return {p.id: 0.0 for p in parties}, 0

    def alloc_at(lam: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in parties:
            u = p.utility
            if lam >= u.slope_at_zero - _EPS:
                out[p.id] = 0.0
                continue
            # For Quadratic: u' = a - b*x; (u')^{-1}(λ) = (a - λ)/b
            if isinstance(u, QuadraticUtility):
                x = max(0.0, (u.a - lam) / u.b)
                out[p.id] = min(x, u.cap)
                continue
            # Piecewise: walk segments, fill while slope > λ.
            if isinstance(u, PiecewiseLinearUtility):
                x_accum = 0.0
                for (x_a, u_a), (x_b, u_b) in zip(u.breakpoints, u.breakpoints[1:]):
                    seg_slope = (u_b - u_a) / max(x_b - x_a, _EPS)
                    if seg_slope > lam + _EPS:
                        x_accum = x_b
                    else:
                        break
                out[p.id] = min(x_accum, u.cap)
                continue
            # Generic fallback: bisect u'(x) = λ
            cap = u.cap

            def slope_at(x: float, u=u) -> float:
                return u.derivative(x) - lam

            if u.derivative(0.0) <= lam + _EPS:
                out[p.id] = 0.0
                continue
            if u.derivative(cap) >= lam - _EPS:
                out[p.id] = cap
                continue
            x, _ = _bisect(0.0, cap, slope_at)
            out[p.id] = max(0.0, min(x, cap))
        return out

    def excess(lam: float) -> float:
        return sum(alloc_at(lam).values()) - budget

    lam_star, iters = _bisect(0.0, max_slope, excess)
    return alloc_at(lam_star), iters


def _greedy_linear_fill(
    parties: Sequence[PartySpec],
    budget: float,
) -> dict[str, float]:
    """Utilitarian fill for linear utilities — sort by slope desc, fill caps."""
    out = {p.id: 0.0 for p in parties}
    remaining = budget
    # Group by slope so ties get equal share.
    sorted_parties = sorted(
        parties,
        key=lambda p: -float(p.utility.slope_at_zero),
    )
    i = 0
    n = len(sorted_parties)
    while i < n and remaining > _EPS:
        j = i + 1
        slope = sorted_parties[i].utility.slope_at_zero
        while j < n and abs(sorted_parties[j].utility.slope_at_zero - slope) < _EPS:
            j += 1
        # Distribute remaining (or caps) among parties [i, j) equally.
        group = sorted_parties[i:j]
        caps = [p.utility.cap for p in group]
        # Iterative: while we have parties with slack and budget remaining,
        # equal-share to the un-saturated ones.
        unsat = list(range(len(group)))
        shares = [0.0] * len(group)
        # Greedy: equal-divide, then saturate caps and recompute.
        while unsat and remaining > _EPS:
            share = remaining / len(unsat)
            newly_saturated: list[int] = []
            for k in list(unsat):
                avail = caps[k] - shares[k]
                if share >= avail - _EPS:
                    shares[k] = caps[k]
                    remaining -= avail
                    newly_saturated.append(k)
                else:
                    shares[k] += share
                    remaining -= share
            if not newly_saturated:
                break
            for k in newly_saturated:
                unsat.remove(k)
        for p, s in zip(group, shares):
            out[p.id] = s
        i = j
    return out


def _solve_egalitarian(
    parties: Sequence[PartySpec],
    budget: float,
    *,
    floor_from_disagreement: bool = False,
) -> tuple[dict[str, float], int]:
    """Find allocations that maximise the minimum (u_i − d_i)/w_i (or u_i/w_i).

    For each candidate target c:
        x_i = u_i^{-1}( d_i + c * w_i )  if floor_from_disagreement
              u_i^{-1}( c * w_i )         otherwise
    Bisect c such that Σ x_i = budget. Caps and disagreement are
    respected. Returns (alloc, iters).
    """
    if budget <= 0:
        return {p.id: 0.0 for p in parties}, 0
    # Bracket: c can range from 0 to max ideal/weight (capped by all party caps).
    c_hi = 0.0
    for p in parties:
        ideal = p.utility.ideal
        offset = p.disagreement if floor_from_disagreement else 0.0
        gain = (ideal - offset) / max(p.weight, _EPS)
        if gain > c_hi:
            c_hi = gain

    def alloc_at(c: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in parties:
            target = (p.disagreement if floor_from_disagreement else 0.0) + c * p.weight
            target = max(0.0, min(target, p.utility.ideal))
            out[p.id] = p.utility.inverse(target)
        return out

    def excess(c: float) -> float:
        return sum(alloc_at(c).values()) - budget

    if excess(0.0) > 0:
        return alloc_at(0.0), 0
    if excess(c_hi) <= 0:
        return alloc_at(c_hi), 0
    c_star, iters = _bisect_increasing(0.0, c_hi, excess)
    return alloc_at(c_star), iters


def _solve_nash(
    parties: Sequence[PartySpec],
    budget: float,
) -> tuple[dict[str, float], int]:
    """Maximise Σ w_i log(u_i(x_i) − d_i) subject to Σ x_i ≤ budget.

    KKT: w_i u_i'(x_i^*) / (u_i(x_i^*) − d_i) = λ.

    Bisect λ. The function is monotone non-increasing in λ for each
    party (higher dual ⇒ smaller share).
    """
    if budget <= 0:
        return {p.id: 0.0 for p in parties}, 0
    # λ-range: at λ=0, all parties saturate; at λ=∞, all zero.
    # Find an upper bound for λ by checking parties' min ratio.
    lam_hi = 0.0
    for p in parties:
        # At x=0, u=intercept (linear) or 0 (others); use the slope at zero.
        u0 = p.utility.evaluate(0.0)
        gap0 = u0 - p.disagreement
        if gap0 <= _EPS:
            # disagreement >= u(0); need to start above zero. Use
            # slope/(ε) as a safe upper bracket.
            lam_hi = max(lam_hi, p.utility.slope_at_zero / _EPS)
        else:
            lam_hi = max(lam_hi, p.weight * p.utility.slope_at_zero / gap0)
    if lam_hi <= _EPS:
        return {p.id: 0.0 for p in parties}, 0

    def x_for_party(p: PartySpec, lam: float) -> float:
        u = p.utility
        cap = u.cap
        w = p.weight
        d = p.disagreement
        if lam <= _EPS:
            return cap
        # Find x in [0, cap] such that w*u'(x) = lam*(u(x) - d). Both
        # sides monotone (LHS decreasing, RHS increasing in x); we
        # bisect.
        # Edge: at x = 0, LHS = w*u'(0), RHS = lam*(u(0)-d). If LHS <
        # RHS, return 0 (party already at corner).
        u0 = u.evaluate(0.0)
        if w * u.derivative(0.0) <= lam * (u0 - d) + _EPS:
            return 0.0
        uc = u.evaluate(cap)
        if w * u.derivative(cap) >= lam * (uc - d) - _EPS:
            return cap

        def residual(x: float, p=p, lam=lam) -> float:
            return p.weight * p.utility.derivative(x) - lam * max(
                p.utility.evaluate(x) - p.disagreement, _EPS
            )

        x, _ = _bisect(0.0, cap, residual)
        return max(0.0, min(x, cap))

    def alloc_at(lam: float) -> dict[str, float]:
        return {p.id: x_for_party(p, lam) for p in parties}

    def excess(lam: float) -> float:
        return sum(alloc_at(lam).values()) - budget

    lam_star, iters = _bisect(0.0, lam_hi, excess)
    return alloc_at(lam_star), iters


def _solve_kalai_smorodinsky(
    parties: Sequence[PartySpec],
    budget: float,
) -> tuple[dict[str, float], int]:
    """KS: max t such that u_i(x_i) = d_i + t (u_i^* - d_i) is feasible.

    For each t in [0,1], the target utility for each party is fixed.
    Feasibility reduces to Σ u_i^{-1}(target_i) ≤ budget.
    """
    if budget <= 0:
        return {p.id: 0.0 for p in parties}, 0
    for p in parties:
        if p.disagreement >= p.utility.ideal - _EPS:
            raise NegotiationInfeasible(
                f"KS requires d_i < u_i^* (party {p.id}); got "
                f"d={p.disagreement}, ideal={p.utility.ideal}"
            )

    def alloc_at(t: float) -> dict[str, float]:
        out = {}
        for p in parties:
            target = p.disagreement + t * (p.utility.ideal - p.disagreement)
            target = max(0.0, min(target, p.utility.ideal))
            out[p.id] = p.utility.inverse(target)
        return out

    def excess(t: float) -> float:
        return sum(alloc_at(t).values()) - budget

    if excess(0.0) > 0:
        # Even disagreement points cost more than budget; infeasible.
        raise NegotiationInfeasible(
            "KS: aggregate disagreement requirements exceed budget"
        )
    if excess(1.0) <= 0:
        return alloc_at(1.0), 0
    t_star, iters = _bisect_increasing(0.0, 1.0, excess)
    return alloc_at(t_star), iters


def _solve_proportional_fair(
    parties: Sequence[PartySpec],
    budget: float,
) -> tuple[dict[str, float], int]:
    """Proportional-fair: max Σ w_i log u_i(x_i).

    KKT: w_i u_i'(x_i) / u_i(x_i) = λ. Same bisection as Nash but with
    d_i = 0 — implemented by transient zero-disagreement parties.
    """
    if budget <= 0:
        return {p.id: 0.0 for p in parties}, 0
    pf_parties = tuple(
        PartySpec(
            id=p.id,
            utility=p.utility,
            disagreement=0.0,
            weight=p.weight,
            priority=p.priority,
            meta=p.meta,
        )
        for p in parties
    )
    return _solve_nash(pf_parties, budget)


def _solve_leximin(
    parties: Sequence[PartySpec],
    budget: float,
    *,
    floor_from_disagreement: bool = False,
) -> tuple[dict[str, float], int]:
    """Leximin = iterate egalitarian, freezing parties at each plateau.

    Returns the final allocation and the cumulative iteration count.
    """
    if budget <= 0:
        return {p.id: 0.0 for p in parties}, 0
    remaining_budget = float(budget)
    remaining = list(parties)
    final: dict[str, float] = {}
    total_iters = 0
    safety = 0
    while remaining and remaining_budget > _EPS and safety < len(parties) + 5:
        safety += 1
        # Egalitarian solve on remaining set.
        alloc, iters = _solve_egalitarian(
            remaining,
            remaining_budget,
            floor_from_disagreement=floor_from_disagreement,
        )
        total_iters += iters
        if not alloc:
            break
        # Compute utility at the floor.
        floor_u = float("inf")
        for p in remaining:
            x = alloc[p.id]
            u = p.utility.evaluate(x)
            scaled = (
                u - p.disagreement
                if floor_from_disagreement
                else u
            ) / max(p.weight, _EPS)
            if scaled < floor_u:
                floor_u = scaled
        # Freeze every party that is *at* the floor — others stay in
        # the pool for the next round, since they have slack to gain.
        next_remaining: list[PartySpec] = []
        for p in remaining:
            x = alloc[p.id]
            u = p.utility.evaluate(x)
            scaled = (
                u - p.disagreement
                if floor_from_disagreement
                else u
            ) / max(p.weight, _EPS)
            at_floor = scaled <= floor_u + 1e-7
            cap_saturated = x >= p.utility.cap - _EPS
            if at_floor or cap_saturated:
                final[p.id] = x
                remaining_budget -= x
            else:
                next_remaining.append(p)
        if len(next_remaining) == len(remaining):
            # No progress — terminate and commit current allocation.
            for p in remaining:
                final.setdefault(p.id, alloc[p.id])
            break
        remaining = next_remaining
        remaining_budget = max(0.0, remaining_budget)
    # Any party not yet assigned (shouldn't happen) gets 0.
    for p in parties:
        final.setdefault(p.id, 0.0)
    return final, total_iters


# =====================================================================
# Envy-freeness + Pareto checks
# =====================================================================


def compute_envy(
    parties: Sequence[PartySpec],
    allocation: Mapping[str, float],
) -> EnvyReport:
    """Compute pairwise envy for an allocation.

    For each pair (i, j), envy_{i→j} = max(0, u_i(x_j) - u_i(x_i)).
    """
    pair_envy: dict[tuple[str, str], float] = {}
    max_envy = 0.0
    for i in parties:
        u_i_self = i.utility.evaluate(allocation.get(i.id, 0.0))
        for j in parties:
            if i.id == j.id:
                continue
            u_i_other = i.utility.evaluate(allocation.get(j.id, 0.0))
            envy = max(0.0, u_i_other - u_i_self)
            pair_envy[(i.id, j.id)] = envy
            if envy > max_envy:
                max_envy = envy
    return EnvyReport(
        envy_free=(max_envy <= 1e-7),
        max_envy=max_envy,
        pair_envy=pair_envy,
    )


def compute_pareto_check(
    parties: Sequence[PartySpec],
    allocation: Mapping[str, float],
    budget: float,
    *,
    probe_concepts: Sequence[str] = (
        CONCEPT_UTILITARIAN,
        CONCEPT_EGALITARIAN,
        CONCEPT_NASH,
    ),
) -> ParetoReport:
    """Probe-grid Pareto check.

    Computes a handful of competing allocations and tests whether any
    of them Pareto-dominate `allocation`. If none do, the candidate is
    *probably* Pareto-optimal; the report reflects what was tested,
    not a formal proof of optimality.
    """
    base_us = {p.id: p.utility.evaluate(allocation.get(p.id, 0.0)) for p in parties}
    dominator = ""
    dom_gap = 0.0
    for concept in probe_concepts:
        try:
            if concept == CONCEPT_UTILITARIAN:
                alloc, _ = _water_fill_utilitarian(parties, budget)
            elif concept == CONCEPT_EGALITARIAN:
                alloc, _ = _solve_egalitarian(parties, budget)
            elif concept == CONCEPT_NASH:
                alloc, _ = _solve_nash(parties, budget)
            elif concept == CONCEPT_PROPORTIONAL_FAIR:
                alloc, _ = _solve_proportional_fair(parties, budget)
            elif concept == CONCEPT_KALAI_SMORODINSKY:
                alloc, _ = _solve_kalai_smorodinsky(parties, budget)
            else:
                continue
        except NegotiationInfeasible:
            continue
        if not alloc:
            continue
        probe_us = {p.id: p.utility.evaluate(alloc.get(p.id, 0.0)) for p in parties}
        # Dominance: all ≥, at least one strictly greater.
        all_ge = all(probe_us[p.id] >= base_us[p.id] - 1e-9 for p in parties)
        any_gt = any(probe_us[p.id] > base_us[p.id] + 1e-7 for p in parties)
        if all_ge and any_gt:
            gap = max(probe_us[p.id] - base_us[p.id] for p in parties)
            if gap > dom_gap:
                dom_gap = gap
                dominator = concept
    return ParetoReport(
        dominated=(dominator != ""),
        max_dominance_gap=dom_gap,
        dominator_concept=dominator,
    )


# =====================================================================
# VCG / mechanism-design
# =====================================================================


def vcg_allocate(
    bids: Mapping[str, Mapping[str, float]],
    items: Sequence[str],
    *,
    bundle_cap: int = _VCG_DEFAULT_BUNDLE_CAP,
) -> VCGAllocation:
    """Single-shot VCG allocation for indivisible items.

    `bids[party][item] = v` is each party's reported valuation for
    each item (independent / additive). The mechanism allocates each
    item to the highest bidder and charges each winning party its
    *externality*: the loss in welfare experienced by the rest of the
    world had this party been excluded.

    Multi-unit / combinatorial bundles are not supported in this
    implementation; `bundle_cap` caps the number of items we will
    process to avoid pathological inputs (default 16).
    """
    t0 = time.monotonic()
    if not bids:
        return VCGAllocation(
            winners={}, bundle={}, payments={}, welfare=0.0,
            elapsed_s=time.monotonic() - t0,
        )
    if len(items) > bundle_cap:
        raise NegotiationInfeasible(
            f"VCG: {len(items)} items exceeds bundle cap {bundle_cap}"
        )
    bidders = list(bids.keys())
    # Welfare given a forbidden party set: each item → highest non-
    # forbidden bidder.
    def welfare_excluding(excluded: frozenset[str]) -> tuple[float, dict[str, str]]:
        winners: dict[str, str] = {}
        total = 0.0
        for item in items:
            best_b = ""
            best_v = -float("inf")
            for b in bidders:
                if b in excluded:
                    continue
                v = float(bids[b].get(item, 0.0))
                if v > best_v + 1e-9 and v > 0:
                    best_v = v
                    best_b = b
            if best_b:
                winners[item] = best_b
                total += best_v
        return total, winners

    base_welfare, winners = welfare_excluding(frozenset())
    # Build bundle map (winner → items it received).
    bundle: dict[str, list[str]] = {b: [] for b in bidders}
    for item, w in winners.items():
        bundle[w].append(item)
    # Per-party payment: w_{−i}^* − Σ_{j≠i} v_j(item_j^*)
    payments: dict[str, float] = {}
    for party in bidders:
        if party not in bundle or not bundle[party]:
            payments[party] = 0.0
            continue
        excl_welfare, _ = welfare_excluding(frozenset({party}))
        # Welfare to others given the actual allocation.
        others_welfare = 0.0
        for item, w in winners.items():
            if w == party:
                continue
            others_welfare += float(bids[w].get(item, 0.0))
        payments[party] = max(0.0, excl_welfare - others_welfare)
    return VCGAllocation(
        winners=winners,
        bundle={b: tuple(items) for b, items in bundle.items() if items},
        payments=payments,
        welfare=base_welfare,
        elapsed_s=time.monotonic() - t0,
    )


def nash_product(
    parties: Sequence[PartySpec],
    allocation: Mapping[str, float],
) -> float:
    """Σ w_i log(u_i(x_i) − d_i); returns −inf if any party is at/below d."""
    total = 0.0
    for p in parties:
        u = p.utility.evaluate(allocation.get(p.id, 0.0))
        gap = u - p.disagreement
        if gap <= _EPS:
            return float("-inf")
        total += p.weight * math.log(gap)
    return total


def welfare(
    parties: Sequence[PartySpec],
    allocation: Mapping[str, float],
) -> float:
    """Σ w_i u_i(x_i) — utilitarian welfare."""
    return sum(
        p.weight * p.utility.evaluate(allocation.get(p.id, 0.0))
        for p in parties
    )


def min_utility(
    parties: Sequence[PartySpec],
    allocation: Mapping[str, float],
) -> float:
    """min_i u_i(x_i) — egalitarian welfare."""
    if not parties:
        return 0.0
    return min(p.utility.evaluate(allocation.get(p.id, 0.0)) for p in parties)


# =====================================================================
# Negotiator class
# =====================================================================


class Negotiator:
    """Multi-party allocation kernel.

    Thread-safe; an internal lock guards every public method that
    mutates state. Reads of immutable returned dataclasses are safe
    without external synchronisation.

    Construction is cheap; the heavy lifting happens inside the
    `allocate_*` methods. Each concept produces a fresh
    NegotiationReport with the axiom certificate, envy diagnostics,
    Pareto probe, and an optional AttestationLedger receipt.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        attestor: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._clock = clock
        self._id = f"negotiator-{uuid.uuid4().hex[:8]}"
        self._lock = threading.Lock()
        self._parties: dict[str, PartySpec] = {}
        self._budget: float = 0.0
        self._created_at: float = clock()
        self._emit(NEGOTIATOR_STARTED, {"negotiator_id": self._id})

    # ----------------------------------------------------------------
    # Mutators
    # ----------------------------------------------------------------

    def register_party(
        self,
        party_id: str,
        utility: Utility,
        *,
        disagreement: float = 0.0,
        weight: float = 1.0,
        priority: int = 0,
        meta: Mapping[str, Any] | None = None,
    ) -> PartySpec:
        """Register a party. Re-registration replaces the prior spec."""
        if not party_id:
            raise ValueError("party_id must be non-empty")
        if not isinstance(utility, Utility):
            raise TypeError("utility must be a Utility instance")
        if weight <= 0:
            raise ValueError("weight must be > 0")
        if disagreement < 0:
            raise ValueError("disagreement must be >= 0")
        if not math.isfinite(disagreement):
            raise ValueError("disagreement must be finite")
        with self._lock:
            spec = PartySpec(
                id=party_id,
                utility=utility,
                disagreement=float(disagreement),
                weight=float(weight),
                priority=int(priority),
                meta=dict(meta or {}),
            )
            self._parties[party_id] = spec
        self._emit(NEGOTIATOR_PARTY_REGISTERED, {
            "negotiator_id": self._id,
            "party_id": party_id,
            "weight": spec.weight,
            "disagreement": spec.disagreement,
            "utility_kind": utility.to_dict().get("kind"),
            "cap": utility.cap,
            "ideal": utility.ideal,
        })
        return spec

    def remove_party(self, party_id: str) -> bool:
        with self._lock:
            existed = party_id in self._parties
            self._parties.pop(party_id, None)
        if existed:
            self._emit(NEGOTIATOR_PARTY_REMOVED, {
                "negotiator_id": self._id,
                "party_id": party_id,
            })
        return existed

    def set_budget(self, budget: float) -> float:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        if not math.isfinite(budget):
            raise ValueError("budget must be finite")
        with self._lock:
            self._budget = float(budget)
        self._emit(NEGOTIATOR_BUDGET_CHANGED, {
            "negotiator_id": self._id,
            "budget": float(budget),
        })
        return float(budget)

    def clear(self) -> None:
        with self._lock:
            self._parties.clear()
            self._budget = 0.0
        self._emit(NEGOTIATOR_CLEARED, {"negotiator_id": self._id})

    # ----------------------------------------------------------------
    # Read accessors
    # ----------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def n_parties(self) -> int:
        with self._lock:
            return len(self._parties)

    @property
    def budget(self) -> float:
        with self._lock:
            return self._budget

    def parties(self) -> tuple[PartySpec, ...]:
        with self._lock:
            return tuple(self._parties.values())

    def get_party(self, party_id: str) -> PartySpec | None:
        with self._lock:
            return self._parties.get(party_id)

    # ----------------------------------------------------------------
    # Solvers
    # ----------------------------------------------------------------

    def allocate_utilitarian(
        self,
        *,
        budget: float | None = None,
    ) -> NegotiationReport:
        """Maximise Σ w_i u_i(x_i). Pareto-optimal; not envy-free in general."""
        return self._solve_with_budget(
            concept=CONCEPT_UTILITARIAN,
            budget=budget,
            solve=lambda parties, b: _water_fill_utilitarian(parties, b),
            axioms=(AXIOM_PARETO, AXIOM_BUDGET_BALANCED),
        )

    def allocate_egalitarian(
        self,
        *,
        budget: float | None = None,
        floor_from_disagreement: bool = False,
    ) -> NegotiationReport:
        """Maximise the minimum (u_i − d_i)/w_i (or u_i/w_i)."""
        return self._solve_with_budget(
            concept=CONCEPT_EGALITARIAN,
            budget=budget,
            solve=lambda parties, b: _solve_egalitarian(
                parties, b, floor_from_disagreement=floor_from_disagreement,
            ),
            axioms=(AXIOM_PARETO, AXIOM_SYMMETRY, AXIOM_BUDGET_BALANCED),
        )

    def allocate_leximin(
        self,
        *,
        budget: float | None = None,
        floor_from_disagreement: bool = False,
    ) -> NegotiationReport:
        """Maximise min utility, then next-min, etc. Unique selection rule."""
        return self._solve_with_budget(
            concept=CONCEPT_LEXIMIN,
            budget=budget,
            solve=lambda parties, b: _solve_leximin(
                parties, b, floor_from_disagreement=floor_from_disagreement,
            ),
            axioms=(
                AXIOM_PARETO,
                AXIOM_SYMMETRY,
                AXIOM_LEXIMIN_EQUITY,
                AXIOM_BUDGET_BALANCED,
            ),
        )

    def allocate_nash(
        self,
        *,
        budget: float | None = None,
    ) -> NegotiationReport:
        """Nash bargaining: max Σ w_i log(u_i − d_i)."""
        return self._solve_with_budget(
            concept=CONCEPT_NASH,
            budget=budget,
            solve=_solve_nash,
            axioms=(
                AXIOM_PARETO,
                AXIOM_SYMMETRY,
                AXIOM_IIA,
                AXIOM_AFFINE_INVARIANCE,
                AXIOM_BUDGET_BALANCED,
            ),
        )

    def allocate_kalai_smorodinsky(
        self,
        *,
        budget: float | None = None,
    ) -> NegotiationReport:
        """KS solution: max proportional progress to ideal point."""
        return self._solve_with_budget(
            concept=CONCEPT_KALAI_SMORODINSKY,
            budget=budget,
            solve=_solve_kalai_smorodinsky,
            axioms=(
                AXIOM_PARETO,
                AXIOM_SYMMETRY,
                AXIOM_MONOTONICITY,
                AXIOM_AFFINE_INVARIANCE,
                AXIOM_BUDGET_BALANCED,
            ),
        )

    def allocate_proportional_fair(
        self,
        *,
        budget: float | None = None,
    ) -> NegotiationReport:
        """Proportional fair: max Σ w_i log u_i."""
        return self._solve_with_budget(
            concept=CONCEPT_PROPORTIONAL_FAIR,
            budget=budget,
            solve=_solve_proportional_fair,
            axioms=(
                AXIOM_PARETO,
                AXIOM_SYMMETRY,
                AXIOM_BUDGET_BALANCED,
            ),
        )

    # ----------------------------------------------------------------
    # VCG auction
    # ----------------------------------------------------------------

    def vcg_auction(
        self,
        items: Sequence[str],
        bids: Mapping[str, Mapping[str, float]],
        *,
        bundle_cap: int = _VCG_DEFAULT_BUNDLE_CAP,
    ) -> VCGAllocation:
        """Run a VCG auction on the given indivisible items.

        Bidders need not be registered parties — this method is
        stateless w.r.t. the negotiator's registered set, and is the
        canonical interface when bids arrive from external tenants
        with no prior registration.
        """
        result = vcg_allocate(bids, items, bundle_cap=bundle_cap)
        attested = self._attest_vcg(result)
        self._emit(NEGOTIATOR_ALLOCATED, {
            "negotiator_id": self._id,
            "concept": CONCEPT_VCG,
            "winners": dict(attested.winners),
            "payments": dict(attested.payments),
            "welfare": attested.welfare,
            "receipt_hash": attested.receipt_hash,
        })
        return attested

    # ----------------------------------------------------------------
    # Envy + Pareto introspection
    # ----------------------------------------------------------------

    def envy_freeness(
        self,
        allocation: Mapping[str, float],
    ) -> EnvyReport:
        return compute_envy(self.parties(), allocation)

    def pareto_check(
        self,
        allocation: Mapping[str, float],
        *,
        budget: float | None = None,
    ) -> ParetoReport:
        b = self._budget if budget is None else float(budget)
        return compute_pareto_check(self.parties(), allocation, b)

    # ----------------------------------------------------------------
    # Internal: unified solve + report path
    # ----------------------------------------------------------------

    def _solve_with_budget(
        self,
        *,
        concept: str,
        budget: float | None,
        solve: Callable[[Sequence[PartySpec], float], tuple[dict[str, float], int]],
        axioms: tuple[str, ...],
    ) -> NegotiationReport:
        t0 = self._clock()
        parties = self.parties()
        b = self._budget if budget is None else float(budget)
        if b < 0 or not math.isfinite(b):
            self._emit(NEGOTIATOR_ALLOCATION_FAILED, {
                "negotiator_id": self._id,
                "concept": concept,
                "reason": "invalid_budget",
            })
            raise NegotiationInfeasible(f"invalid budget {b}")
        if not parties:
            return self._empty_report(concept, b, axioms, t0)
        try:
            allocation, iters = solve(parties, b)
        except NegotiationInfeasible as exc:
            self._emit(NEGOTIATOR_ALLOCATION_FAILED, {
                "negotiator_id": self._id,
                "concept": concept,
                "reason": str(exc),
            })
            raise
        # Build report data.
        utilities = {p.id: p.utility.evaluate(allocation.get(p.id, 0.0)) for p in parties}
        total = sum(allocation.values())
        alloc_obj = Allocation(
            assignments=dict(allocation),
            utilities=utilities,
            total_allocated=total,
            budget=b,
            concept=concept,
        )
        envy = compute_envy(parties, allocation)
        pareto = compute_pareto_check(parties, allocation, b)
        # Refine axiom certificate.
        cert = list(axioms)
        if envy.envy_free and AXIOM_ENVY_FREE not in cert:
            cert.append(AXIOM_ENVY_FREE)
        if pareto.dominated and AXIOM_PARETO in cert:
            cert.remove(AXIOM_PARETO)
        diagnostics = {
            "lambda_iterations": iters,
            "n_parties": len(parties),
            "concept": concept,
        }
        nprod = nash_product(parties, allocation)
        wsum = welfare(parties, allocation)
        mu = min_utility(parties, allocation)
        report = NegotiationReport(
            id=f"neg-{uuid.uuid4().hex[:8]}",
            allocation=alloc_obj,
            certificate=tuple(cert),
            envy=envy,
            pareto=pareto,
            nash_product=nprod,
            welfare=wsum,
            min_utility=mu,
            elapsed_s=self._clock() - t0,
            iterations=iters,
            diagnostics=diagnostics,
        )
        report = self._attest(report)
        self._emit(NEGOTIATOR_ALLOCATED, {
            "negotiator_id": self._id,
            "concept": concept,
            "report_id": report.id,
            "assignments": dict(report.allocation.assignments),
            "welfare": report.welfare,
            "min_utility": report.min_utility,
            "envy_free": report.envy.envy_free,
            "certificate": list(report.certificate),
            "receipt_hash": report.receipt_hash,
        })
        return report

    def _empty_report(
        self,
        concept: str,
        budget: float,
        axioms: tuple[str, ...],
        t0: float,
    ) -> NegotiationReport:
        empty_alloc = Allocation(
            assignments={}, utilities={},
            total_allocated=0.0, budget=budget, concept=concept,
        )
        empty_envy = EnvyReport(envy_free=True, max_envy=0.0, pair_envy={})
        empty_pareto = ParetoReport(
            dominated=False, max_dominance_gap=0.0, dominator_concept="",
        )
        return NegotiationReport(
            id=f"neg-{uuid.uuid4().hex[:8]}",
            allocation=empty_alloc,
            certificate=tuple(axioms) + (AXIOM_ENVY_FREE,),
            envy=empty_envy,
            pareto=empty_pareto,
            nash_product=0.0,
            welfare=0.0,
            min_utility=0.0,
            elapsed_s=self._clock() - t0,
            iterations=0,
            diagnostics={"n_parties": 0, "concept": concept},
        )

    # ----------------------------------------------------------------
    # Attestation
    # ----------------------------------------------------------------

    def _attest(self, report: NegotiationReport) -> NegotiationReport:
        if self._attestor is None:
            return report
        try:
            payload = report.to_dict()
            payload.pop("receipt_hash", None)
            serialised = json.dumps(payload, sort_keys=True, default=str)
            digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
            receipt_hash = digest
            rec = getattr(self._attestor, "record", None)
            if callable(rec):
                try:
                    receipt = rec(kind="negotiator.allocated", payload=payload)
                    if hasattr(receipt, "hash"):
                        receipt_hash = receipt.hash
                    elif isinstance(receipt, str):
                        receipt_hash = receipt
                except Exception:
                    pass
            else:
                try:
                    entry = self._attestor(_AttestableReport(report, payload))
                    if entry is not None and hasattr(entry, "entry_hash"):
                        receipt_hash = entry.entry_hash
                except Exception:
                    pass
            return NegotiationReport(
                id=report.id,
                allocation=report.allocation,
                certificate=report.certificate,
                envy=report.envy,
                pareto=report.pareto,
                nash_product=report.nash_product,
                welfare=report.welfare,
                min_utility=report.min_utility,
                elapsed_s=report.elapsed_s,
                iterations=report.iterations,
                receipt_hash=receipt_hash,
                diagnostics=report.diagnostics,
            )
        except Exception:
            return report

    def _attest_vcg(self, result: VCGAllocation) -> VCGAllocation:
        if self._attestor is None:
            return result
        try:
            payload = result.to_dict()
            payload.pop("receipt_hash", None)
            serialised = json.dumps(payload, sort_keys=True, default=str)
            digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
            receipt_hash = digest
            rec = getattr(self._attestor, "record", None)
            if callable(rec):
                try:
                    receipt = rec(kind="negotiator.vcg", payload=payload)
                    if hasattr(receipt, "hash"):
                        receipt_hash = receipt.hash
                    elif isinstance(receipt, str):
                        receipt_hash = receipt
                except Exception:
                    pass
            else:
                try:
                    adapter = _AttestableReport(result, payload)
                    adapter.kind = "negotiator.vcg"
                    entry = self._attestor(adapter)
                    if entry is not None and hasattr(entry, "entry_hash"):
                        receipt_hash = entry.entry_hash
                except Exception:
                    pass
            return VCGAllocation(
                winners=result.winners,
                bundle=result.bundle,
                payments=result.payments,
                welfare=result.welfare,
                elapsed_s=result.elapsed_s,
                receipt_hash=receipt_hash,
            )
        except Exception:
            return result

    # ----------------------------------------------------------------
    # Telemetry
    # ----------------------------------------------------------------

    def _emit(self, kind: str, data: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(data)))
        except Exception:
            pass


class _AttestableReport:
    """Adapter for AttestationLedger.append-style attestors."""

    def __init__(
        self,
        report: NegotiationReport | VCGAllocation,
        payload: dict[str, Any],
    ) -> None:
        self.ticket_id = getattr(report, "id", "")
        self.kind = "negotiator.allocated"
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "kind": self.kind,
            "payload": self.payload,
        }


# =====================================================================
# Free functions for direct algorithmic access
# =====================================================================


def allocate(
    parties: Sequence[PartySpec],
    budget: float,
    *,
    concept: str = CONCEPT_NASH,
) -> dict[str, float]:
    """One-shot allocate without constructing a Negotiator.

    Convenience wrapper for callers who don't need event emission or
    attestation.  Returns the assignments map.
    """
    if concept not in KNOWN_CONCEPTS:
        raise ValueError(f"unknown concept: {concept}")
    if concept == CONCEPT_VCG:
        raise ValueError("VCG operates on indivisible items; use vcg_allocate")
    if concept == CONCEPT_UTILITARIAN:
        alloc, _ = _water_fill_utilitarian(parties, budget)
    elif concept == CONCEPT_EGALITARIAN:
        alloc, _ = _solve_egalitarian(parties, budget)
    elif concept == CONCEPT_LEXIMIN:
        alloc, _ = _solve_leximin(parties, budget)
    elif concept == CONCEPT_NASH:
        alloc, _ = _solve_nash(parties, budget)
    elif concept == CONCEPT_KALAI_SMORODINSKY:
        alloc, _ = _solve_kalai_smorodinsky(parties, budget)
    elif concept == CONCEPT_PROPORTIONAL_FAIR:
        alloc, _ = _solve_proportional_fair(parties, budget)
    else:  # pragma: no cover - guarded by KNOWN_CONCEPTS
        raise ValueError(f"unsupported concept: {concept}")
    return alloc
