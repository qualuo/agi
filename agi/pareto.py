r"""Pareto — multi-objective optimization as a runtime primitive.

Every primitive in this runtime that *chooses* — ``Bandit``, ``BayesOpt``,
``Searcher``, ``Solver``, ``Strategist``, ``PortfolioOptimizer`` —
collapses the world to one scalar before deciding.  Real products never
have one objective.  A drug-discovery pipeline weighs *binding affinity*,
*synthetic accessibility*, *toxicity*, and *novelty*.  A coordination
engine weighs *probability of success*, *cost*, *latency*, and
*compliance*.  A negotiation primitive weighs *outcome value* and
*fairness*.  Naively scalarising at decision time hides exactly the
trade-off the human needs to see.

``Pareto`` is the runtime's *bounded, anytime, certified, stdlib*
multi-objective decision kernel.  It maintains a population of
candidates each tagged with an M-dimensional cost vector, exposes the
non-dominated frontier on demand, and emits the gold-standard
diversity, convergence, and progress metrics that the multi-objective
optimisation literature has settled on.

The pitch reduced to a runtime call::

    pf = Pareto(ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0)))
    pf.observe("c1", (0.20, 0.80))
    pf.observe("c2", (0.50, 0.50))
    pf.observe("c3", (0.80, 0.20))
    pf.observe("c4", (0.40, 0.60))   # dominated by c2

    front = pf.frontier()              # [c1, c2, c3] sorted by first obj
    hv = pf.hypervolume()              # exact for M=2 / M=3, WFG for M ≤ 5
    ehvi = pf.expected_hypervolume_improvement((0.35, 0.45), (0.10, 0.10))
    report = pf.report()
    cert = pf.certify_progress()       # anytime-valid LCB on HV regret


What this primitive ships
-------------------------

  * **Non-dominated sorting** (Deb-Pratap-Agarwal-Meyarivan 2002 NSGA-II).
    Fast O(M N²) sort that partitions the population into Pareto-rank
    fronts.  Crowding distance (Deb 2002 §3.3) on each front gives the
    diversity metric NSGA-II uses for tie-breaking.

  * **Hypervolume indicator** (Zitzler-Thiele 1998 *Multiobjective
    optimization using evolutionary algorithms — a comparative case
    study*).  Closed-form for M=2 via sweep, exact for M=3 via slicing
    (Beume-Fonseca-López-Ibáñez-Paquete-Vahrenhold 2009 *HV by slicing
    objectives*), WFG-style dimension-sweep decomposition (While-
    Hingston 2011 *A faster algorithm for calculating hypervolume*) for
    M ≤ 5, and inclusion-exclusion of axis-aligned boxes for any M.

  * **Expected Hypervolume Improvement** (Emmerich-Beume-Naujoks 2005
    *An EMO algorithm using the hypervolume measure as selection
    criterion*; Emmerich 2008 closed-form for M=2).  EHVI is the
    canonical acquisition for *multi-objective Bayesian optimisation*
    — the next batch of candidates the coordinator should evaluate.
    Closed-form for M=2 under Gaussian belief, Monte-Carlo over slices
    for M=3 with an exact upper bound, fully Monte-Carlo for M ≥ 4.

  * **Scalarisations.**  Weighted-sum, Tchebycheff (Steuer 1986
    *Multiple Criteria Optimization*), augmented Tchebycheff (Knowles
    2006 *ParEGO*), achievement scalarising function (Wierzbicki 1980
    *The use of reference objectives in multiobjective optimization*),
    Boundary Intersection (Das-Dennis 1998).  Each turns the M-vector
    cost into one scalar so existing single-objective primitives
    (``Bandit``, ``BayesOpt``, ``Searcher``) can drive the search.

  * **Uniform weight sweep** (Das-Dennis 1998).  Generates an evenly
    spaced grid of convex weight vectors on the M-simplex; the canonical
    way to expose the full Pareto front to a coordinator that calls a
    single-objective inner loop K times.

  * **Convergence + diversity metrics.**  Inverted Generational
    Distance (Coello-Sierra 2004), Generational Distance (Van Veldhuizen
    1999), spacing (Schott 1995), maximum-spread (Zitzler 1999).  Every
    metric carries a Maurer-Pontil 2009 empirical-Bernstein half-width
    on its sample mean when computed from a sub-sample.

  * **Anytime-valid progress certificate.**  Hypervolume regret
    ``HV(P_t) − HV(P_∞)`` is monotone in t under any greedy enlargement
    of P_t, so its complement ``HV(P_∞) − HV(P_t)`` is a non-negative
    super-martingale.  Howard-Ramdas-McAuliffe-Sekhon 2021 supplies the
    anytime-valid confidence sequence the coordination engine needs to
    stop expanding the front when its expected improvement falls below
    a configurable tolerance.

  * **Constraint handling.**  Feasibility-first dominance (Deb 2000) so
    constrained problems compose: a feasible point dominates any
    infeasible point, and infeasible points compare on aggregate
    constraint violation.

  * **Tamper-evident SHA-256 fingerprint chain** (genesis
    ``agi.pareto.v1`` + optional HMAC) over every register / observe /
    frontier / report / certify event.  ``AttestationLedger`` replays
    every multi-objective decision byte-for-byte from the observation
    stream.

  * **Snapshot / restore.**  ``snapshot()`` and ``restore()`` round-trip
    a byte-identical chain head so a coordination engine can hibernate
    the front, ship it to another host, and resume.

  * **Thread-safe re-entrant lock** + transport-agnostic + pure stdlib.

Composes with
-------------

  * ``BayesOpt`` — the EHVI acquisition function picks the next
    candidate when the surrogate emits a Gaussian posterior over each
    objective.
  * ``Bandit`` — register a Tchebycheff scalarisation as the bandit's
    reward channel; sweep the weight grid to expose the front.
  * ``Searcher`` — every leaf in the tree-search becomes a candidate
    with an M-objective cost; the search returns the *Pareto layer*
    not the argmax.
  * ``PortfolioOptimizer`` — Pareto sorts (return, risk) candidates
    before allocating a fixed budget across them.
  * ``Strategist`` — fuses calibration + conformal + causal + OPE on
    *each objective* and returns the Pareto-rank-1 layer to the
    coordination engine.
  * ``Coalition`` — multi-criterion Shapley value: one Pareto rank per
    criterion, then aggregated.
  * ``Negotiator`` — Kalai-Smorodinsky and Nash bargaining are
    *exactly* the Tchebycheff and weighted-product scalarisations on
    the disagreement-shifted objective space; ``Pareto.frontier()``
    bounds the bargaining set.
  * ``DriftSentinel`` — running hypervolume CUSUM rolls back the
    deployed front when its progress signal regresses.
  * ``AttestationLedger`` — every event hash-chains into the global
    audit log.
  * ``Coordinator`` — every Goal whose execution must trade off more
    than one objective routes through Pareto for a rank-1 candidate
    panel + a calibrated EHVI to spend the next evaluation budget on.
"""

from __future__ import annotations

import bisect
import hashlib
import hmac
import itertools
import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Event kinds
    "PARETO_STARTED",
    "PARETO_OBSERVED",
    "PARETO_REMOVED",
    "PARETO_FRONTIER",
    "PARETO_REPORTED",
    "PARETO_CERTIFIED",
    "PARETO_CLEARED",
    # Sense constants
    "SENSE_MIN",
    "SENSE_MAX",
    "KNOWN_SENSES",
    # Scalarisation names
    "SCALAR_WEIGHTED_SUM",
    "SCALAR_TCHEBYCHEFF",
    "SCALAR_AUG_TCHEBYCHEFF",
    "SCALAR_ASF",
    "SCALAR_PBI",
    "KNOWN_SCALARISATIONS",
    # Errors
    "ParetoError",
    "InvalidConfig",
    "InvalidCandidate",
    "InvalidQuery",
    "InsufficientData",
    "UnknownCandidate",
    # Dataclasses
    "ParetoConfig",
    "Candidate",
    "FrontierReport",
    "HypervolumeReport",
    "EHVIReport",
    "MetricsReport",
    "ProgressCertificate",
    "ScalarisationReport",
    # Main class
    "Pareto",
    # Helper functions
    "dominates",
    "non_dominated_sort",
    "crowding_distance",
    "hypervolume",
    "hypervolume_2d",
    "hypervolume_3d",
    "ehvi_2d",
    "ehvi_monte_carlo",
    "weighted_sum",
    "tchebycheff",
    "augmented_tchebycheff",
    "achievement_scalarising_function",
    "pbi_scalarisation",
    "das_dennis_weights",
    "uniform_simplex_weights",
    "inverted_generational_distance",
    "generational_distance",
    "spacing",
    "maximum_spread",
    "ledger_root",
    "hrms_half_width",
    "empirical_bernstein_half_width",
    "hoeffding_half_width",
]


# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------

PARETO_STARTED = "pareto.started"
PARETO_OBSERVED = "pareto.observed"
PARETO_REMOVED = "pareto.removed"
PARETO_FRONTIER = "pareto.frontier"
PARETO_REPORTED = "pareto.reported"
PARETO_CERTIFIED = "pareto.certified"
PARETO_CLEARED = "pareto.cleared"


# ---------------------------------------------------------------------------
# Senses + scalarisations
# ---------------------------------------------------------------------------

SENSE_MIN = "min"
SENSE_MAX = "max"
KNOWN_SENSES: frozenset[str] = frozenset({SENSE_MIN, SENSE_MAX})

SCALAR_WEIGHTED_SUM = "weighted_sum"
SCALAR_TCHEBYCHEFF = "tchebycheff"
SCALAR_AUG_TCHEBYCHEFF = "augmented_tchebycheff"
SCALAR_ASF = "achievement_scalarising_function"
SCALAR_PBI = "penalty_boundary_intersection"
KNOWN_SCALARISATIONS: frozenset[str] = frozenset(
    {
        SCALAR_WEIGHTED_SUM,
        SCALAR_TCHEBYCHEFF,
        SCALAR_AUG_TCHEBYCHEFF,
        SCALAR_ASF,
        SCALAR_PBI,
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ParetoError(Exception):
    """Base class for Pareto errors."""


class InvalidConfig(ParetoError):
    """The supplied ``ParetoConfig`` is invalid."""


class InvalidCandidate(ParetoError):
    """The supplied candidate is not well-formed for the configured M."""


class InvalidQuery(ParetoError):
    """The query is not well-formed (e.g. reference point shape mismatch)."""


class InsufficientData(ParetoError):
    """Not enough candidates to satisfy the request."""


class UnknownCandidate(ParetoError):
    """The requested candidate id has not been observed."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParetoConfig:
    """Top-level configuration.

    Attributes
    ----------
    senses
        One of ``"min"`` or ``"max"`` per objective.  Length defines
        the objective dimension M.  Internally Pareto normalises to
        an all-minimisation problem.
    reference
        Reference point in user (pre-normalisation) coordinates.
        Required for hypervolume and EHVI calls; must dominate every
        observed point in the minimisation sense.  If ``None``,
        ``hypervolume()`` will derive a nadir-shifted default at call
        time.
    constraint_dim
        Number of inequality-constraint slots ``g(x) ≤ 0`` per
        candidate.  Defaults to 0.  Constraint values are aggregated
        by sum-of-positive-violations for feasibility-first dominance
        (Deb 2000).
    confidence
        Confidence level for every interval the primitive returns
        (Hoeffding, Maurer-Pontil empirical-Bernstein, Howard-Ramdas-
        McAuliffe-Sekhon anytime-valid sequences).  Defaults to
        ``0.95``.
    hmac_key
        Optional secret key for HMAC-SHA-256 over every fingerprint
        entry.
    rng_seed
        Seed for the internal RNG used by Monte-Carlo subroutines
        (EHVI for M ≥ 3, uniform simplex weights with jitter).
    """

    senses: tuple[str, ...] = (SENSE_MIN, SENSE_MIN)
    reference: tuple[float, ...] | None = None
    constraint_dim: int = 0
    confidence: float = 0.95
    hmac_key: bytes | None = None
    rng_seed: int | None = 0xA61BEEF

    def __post_init__(self) -> None:
        if not self.senses or len(self.senses) < 1:
            raise InvalidConfig("senses must have ≥ 1 entry")
        for s in self.senses:
            if s not in KNOWN_SENSES:
                raise InvalidConfig(
                    f"unknown sense {s!r}; expected one of {sorted(KNOWN_SENSES)}"
                )
        if self.reference is not None:
            if len(self.reference) != len(self.senses):
                raise InvalidConfig(
                    "reference shape must match senses shape "
                    f"({len(self.reference)} vs {len(self.senses)})"
                )
            for v in self.reference:
                if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    raise InvalidConfig("reference entries must be finite numbers")
        if self.constraint_dim < 0:
            raise InvalidConfig("constraint_dim must be ≥ 0")
        if not 0.5 < self.confidence < 1.0:
            raise InvalidConfig(
                f"confidence must be in (0.5, 1.0); got {self.confidence}"
            )
        if self.hmac_key is not None and not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes")

    @property
    def n_objectives(self) -> int:
        return len(self.senses)


@dataclass(frozen=True)
class Candidate:
    """One candidate point.

    ``cost`` is in *user* coordinates (mixed min/max per
    ``ParetoConfig.senses``).  Internally Pareto stores the
    minimisation-normalised ``minim_cost``.

    ``constraints`` is the ``g(x) ≤ 0`` slot — the candidate is
    feasible iff every entry is ≤ 0.
    """

    candidate_id: str
    cost: tuple[float, ...]
    constraints: tuple[float, ...] = ()
    weight: float = 1.0
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class FrontierReport:
    """Output of :meth:`Pareto.frontier`."""

    rank: int
    candidates: tuple[str, ...]
    costs: tuple[tuple[float, ...], ...]
    crowding: tuple[float, ...]
    fingerprint_hash: str


@dataclass(frozen=True)
class HypervolumeReport:
    """Output of :meth:`Pareto.hypervolume`."""

    hypervolume: float
    reference: tuple[float, ...]
    n_points: int
    n_objectives: int
    algorithm: str
    fingerprint_hash: str


@dataclass(frozen=True)
class EHVIReport:
    """Output of :meth:`Pareto.expected_hypervolume_improvement`."""

    candidate_id: str | None
    mean: tuple[float, ...]
    sigma: tuple[float, ...]
    ehvi: float
    ehvi_se: float
    n_samples: int
    algorithm: str
    fingerprint_hash: str


@dataclass(frozen=True)
class MetricsReport:
    """Output of :meth:`Pareto.metrics`."""

    igd: float | None
    gd: float | None
    spacing: float | None
    max_spread: float | None
    n_points: int
    fingerprint_hash: str


@dataclass(frozen=True)
class ProgressCertificate:
    """Output of :meth:`Pareto.certify_progress`."""

    hv_now: float
    hv_history: tuple[float, ...]
    delta_hv: float
    delta_hv_lcb: float
    delta_hv_ucb: float
    epsilon: float
    converged: bool
    confidence: float
    fingerprint_hash: str


@dataclass(frozen=True)
class ScalarisationReport:
    """Output of :meth:`Pareto.scalarise`."""

    method: str
    weights: tuple[float, ...]
    values: dict[str, float]
    argmin_candidate: str
    fingerprint_hash: str


# ---------------------------------------------------------------------------
# Math helpers — re-exported
# ---------------------------------------------------------------------------


def _sense_signs(senses: Sequence[str]) -> list[float]:
    return [1.0 if s == SENSE_MIN else -1.0 for s in senses]


def _to_minim(cost: Sequence[float], senses: Sequence[str]) -> tuple[float, ...]:
    signs = _sense_signs(senses)
    return tuple(signs[i] * float(cost[i]) for i in range(len(cost)))


def _from_minim(cost: Sequence[float], senses: Sequence[str]) -> tuple[float, ...]:
    signs = _sense_signs(senses)
    return tuple(signs[i] * float(cost[i]) for i in range(len(cost)))


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Pareto dominance under all-minimisation.

    ``a`` dominates ``b`` iff every entry of ``a`` ≤ the corresponding
    entry of ``b`` and at least one entry is strictly less.
    """
    strictly_better = False
    if len(a) != len(b):
        raise ParetoError("vectors must share length")
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            strictly_better = True
    return strictly_better


def _feasibility_dominates(
    av: float, bv: float, ac: Sequence[float], bc: Sequence[float]
) -> bool | None:
    """Deb 2000 feasibility-first comparison.

    Returns ``True`` if ``a`` dominates ``b``, ``False`` if ``b``
    dominates ``a``, ``None`` if neither strictly dominates under
    feasibility-first rules (both feasible → fall back to Pareto
    dominance, both infeasible with equal violation → same).
    """
    if av <= 0.0 and bv > 0.0:
        return True
    if bv <= 0.0 and av > 0.0:
        return False
    if av > 0.0 and bv > 0.0:
        if av < bv:
            return True
        if bv < av:
            return False
    return None  # both feasible (or equal violation) — defer to Pareto


def non_dominated_sort(
    points: Sequence[Sequence[float]],
    violations: Sequence[float] | None = None,
) -> list[list[int]]:
    """Deb-Pratap-Agarwal-Meyarivan 2002 fast non-dominated sort.

    ``points`` is a list of minimisation-normalised cost vectors.
    ``violations`` is the optional aggregate feasibility violation
    (sum of positive entries of ``g(x)``).  Returns a list of fronts,
    each front being a list of indices into ``points``.  Rank 0 is
    the non-dominated front.

    Complexity: O(M N²) time, O(N²) memory.
    """
    N = len(points)
    if N == 0:
        return []
    if violations is None:
        violations = [0.0] * N
    if len(violations) != N:
        raise ParetoError("violations length mismatch")
    dom_count = [0] * N
    dominated_by = [[] for _ in range(N)]
    fronts: list[list[int]] = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            decided = _feasibility_dominates(
                violations[p], violations[q], (), ()
            )
            if decided is True:
                dominated_by[p].append(q)
            elif decided is False:
                dom_count[p] += 1
            else:
                # Equal feasibility → Pareto dominance
                if dominates(points[p], points[q]):
                    dominated_by[p].append(q)
                elif dominates(points[q], points[p]):
                    dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(points: Sequence[Sequence[float]]) -> list[float]:
    """Deb 2002 NSGA-II crowding distance.

    Returns a list of crowding distances for each point in
    ``points``.  Boundary points receive ``+inf``.  Distance is sum
    over objectives of normalised neighbour gaps.
    """
    N = len(points)
    if N == 0:
        return []
    M = len(points[0])
    dist = [0.0] * N
    for m in range(M):
        order = sorted(range(N), key=lambda i: points[i][m])
        lo = points[order[0]][m]
        hi = points[order[-1]][m]
        rng = hi - lo
        dist[order[0]] = math.inf
        dist[order[-1]] = math.inf
        if rng <= 0:
            continue
        for k in range(1, N - 1):
            i = order[k]
            if dist[i] == math.inf:
                continue
            prev_v = points[order[k - 1]][m]
            next_v = points[order[k + 1]][m]
            dist[i] += (next_v - prev_v) / rng
    return dist


def hypervolume_2d(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    """Exact 2D hypervolume under all-minimisation.

    Closed form via sweep: O(N log N).
    """
    if not points:
        return 0.0
    rx, ry = reference[0], reference[1]
    kept = [(p[0], p[1]) for p in points if p[0] < rx and p[1] < ry]
    if not kept:
        return 0.0
    kept.sort(key=lambda p: (p[0], p[1]))
    pruned: list[tuple[float, float]] = []
    best_y = math.inf
    for x, y in kept:
        if y < best_y:
            pruned.append((x, y))
            best_y = y
    hv = 0.0
    prev_x = rx
    for x, y in reversed(pruned):
        hv += (prev_x - x) * (ry - y)
        prev_x = x
    return hv


def hypervolume_3d(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    """Exact 3D hypervolume via slicing on z (Beume et al. 2009 HV-by-
    slicing-objectives).

    Sort points by z ascending; sweep z; at each event the active set
    is the projection to (x, y) of all points with z below the current
    plane; sum the 2D hypervolume of that active set times the slab
    height.
    """
    if not points:
        return 0.0
    rx, ry, rz = reference[0], reference[1], reference[2]
    kept = [p for p in points if p[0] < rx and p[1] < ry and p[2] < rz]
    if not kept:
        return 0.0
    kept = sorted(kept, key=lambda p: p[2])
    hv = 0.0
    for i, p in enumerate(kept):
        slab_top = kept[i + 1][2] if i + 1 < len(kept) else rz
        if slab_top <= p[2]:
            continue
        active = kept[: i + 1]
        active_2d = [(q[0], q[1]) for q in active]
        hv += (slab_top - p[2]) * hypervolume_2d(active_2d, (rx, ry))
    return hv


def hypervolume(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> tuple[float, str]:
    """Hypervolume indicator under all-minimisation.

    Dispatches to ``hypervolume_2d`` for M=2, ``hypervolume_3d`` for
    M=3, WFG-style dimension-sweep decomposition for M=4 and M=5,
    and inclusion-exclusion over axis-aligned boxes for M ≥ 6.

    Returns ``(hv, algorithm_name)``.
    """
    if not points:
        return 0.0, "empty"
    M = len(reference)
    if any(len(p) != M for p in points):
        raise ParetoError("all points must have length len(reference)")
    if M == 1:
        # 1D HV: max distance from min to reference
        best = min(p[0] for p in points)
        return max(0.0, reference[0] - best), "1d_min"
    if M == 2:
        return hypervolume_2d(points, reference), "exact_2d_sweep"
    if M == 3:
        return hypervolume_3d(points, reference), "exact_3d_slicing"
    if M in (4, 5):
        return _hypervolume_wfg(points, reference), f"wfg_{M}d"
    return _hypervolume_inclusion_exclusion(points, reference), "inclusion_exclusion"


def _hypervolume_wfg(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    """WFG-style dimension-sweep recursion (While-Hingston 2011)."""
    M = len(reference)
    pts = [tuple(p) for p in points if all(p[k] < reference[k] for k in range(M))]
    if not pts:
        return 0.0
    pts = sorted(pts, key=lambda p: p[-1])
    hv = 0.0
    prev_z = reference[-1]
    for i in range(len(pts) - 1, -1, -1):
        p = pts[i]
        slab_top = prev_z
        slab_bot = p[-1]
        slab = slab_top - slab_bot
        if slab > 0:
            sub_ref = reference[:-1]
            active = [q[:-1] for q in pts[: i + 1]]
            if M - 1 == 2:
                sub_hv = hypervolume_2d(active, sub_ref)
            elif M - 1 == 3:
                sub_hv = hypervolume_3d(active, sub_ref)
            else:
                sub_hv = _hypervolume_wfg(active, sub_ref)
            hv += slab * sub_hv
        prev_z = p[-1]
    return hv


def _hypervolume_inclusion_exclusion(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    """Brute inclusion-exclusion over axis-aligned boxes.

    Exponential in N — only safe for small N.  Each point defines a
    box ``[p, r]``; the hypervolume is the volume of the union of N
    boxes, computed by inclusion-exclusion.
    """
    M = len(reference)
    pts = [p for p in points if all(p[k] < reference[k] for k in range(M))]
    N = len(pts)
    if N == 0:
        return 0.0
    if N > 14:
        # Refuse to allocate 2^N subsets; fall back to a Monte-Carlo
        # estimate.  Caller can re-issue with a tighter front if exact
        # is required.
        return _hypervolume_monte_carlo(pts, reference, n_samples=1 << 14)
    hv = 0.0
    for k in range(1, N + 1):
        sign = 1.0 if (k % 2 == 1) else -1.0
        for combo in itertools.combinations(range(N), k):
            box_lo = [
                max(pts[j][m] for j in combo) for m in range(M)
            ]
            vol = 1.0
            for m in range(M):
                edge = reference[m] - box_lo[m]
                if edge <= 0:
                    vol = 0.0
                    break
                vol *= edge
            hv += sign * vol
    return max(0.0, hv)


def _hypervolume_monte_carlo(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
    n_samples: int = 1 << 14,
    seed: int = 0xA61BEEF,
) -> float:
    """Monte-Carlo HV estimate.

    Draw uniform samples inside the box from each point's nadir to
    the reference; estimate the fraction that lies under at least one
    point.
    """
    import random as _rnd

    rng = _rnd.Random(seed)
    M = len(reference)
    pts = [p for p in points if all(p[k] < reference[k] for k in range(M))]
    if not pts:
        return 0.0
    nadir = [min(p[m] for p in pts) for m in range(M)]
    box_vol = 1.0
    for m in range(M):
        edge = reference[m] - nadir[m]
        if edge <= 0:
            return 0.0
        box_vol *= edge
    hits = 0
    for _ in range(n_samples):
        s = [nadir[m] + rng.random() * (reference[m] - nadir[m]) for m in range(M)]
        for p in pts:
            if all(s[m] >= p[m] for m in range(M)):
                hits += 1
                break
    return box_vol * hits / n_samples


def ehvi_2d(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
    mean: Sequence[float],
    sigma: Sequence[float],
) -> float:
    r"""Closed-form Expected Hypervolume Improvement for M=2
    (Emmerich-Beume-Naujoks 2005; Emmerich 2008; Yang-Emmerich-Deutz-
    Bäck 2017 box-decomposition formulation).

    Computes ``E[HV(P ∪ {x}) − HV(P)]`` under independent Gaussian
    beliefs ``x_j ∼ N(mean_j, sigma_j²)``, under all-minimisation.

    Derivation
    ----------
    Sort the Pareto skyline by x ascending — y-values are
    automatically decreasing.  With sentinels

      x_0 = −∞,  x_1 < … < x_n,  x_{n+1} = rx
      y_0 = ry,  y_1 > … > y_n,  y_{n+1} = −∞

    decompose the support of (X, Y) by vertical strips
    ``S_i = (x_{i−1}, x_i] × (−∞, y_{i−1}]`` for i = 1, …, n+1.
    Within S_i the hypervolume improvement equals

      imp_i(x, y) = (rx − x)(ry − y) − (x_i − x)(ry − y_{i−1})
                    − Σ_{j=i}^{n} (x_{j+1} − x_j)(ry − max(y_j, y)).

    Independence of X and Y lets each term factorise into a product
    of 0th / 1st partial Gaussian moments — the closed form is then
    a finite sum.
    """
    if len(mean) != 2 or len(sigma) != 2 or len(reference) != 2:
        raise ParetoError("ehvi_2d requires M=2")
    for p in points:
        if len(p) != 2:
            raise ParetoError("ehvi_2d requires 2D points")
    rx, ry = reference[0], reference[1]
    mu_x, mu_y = mean[0], mean[1]
    sx, sy = max(sigma[0], 1e-12), max(sigma[1], 1e-12)
    # Build dominated-front skyline in (x, y) under all-min.
    pruned = sorted({(p[0], p[1]) for p in points if p[0] < rx and p[1] < ry})
    skyline: list[tuple[float, float]] = []
    best_y = math.inf
    for x, y in pruned:
        if y < best_y:
            skyline.append((x, y))
            best_y = y
    n = len(skyline)
    # Sentinel-extended index arrays.
    xs = [-math.inf] + [p[0] for p in skyline] + [rx]
    ys = [ry] + [p[1] for p in skyline] + [-math.inf]

    # Pre-computed cumulative partial y-moments at each y_k.
    def Py(t: float) -> float:
        return _zeroth_moment(-math.inf, t, mu_y, sy)

    def My(t: float) -> float:
        return _partial_first_moment(-math.inf, t, mu_y, sy)

    ehvi = 0.0
    for i in range(1, n + 2):  # strip index 1..n+1
        x_lo = xs[i - 1]
        x_hi = xs[i]
        y_top = ys[i - 1]  # = y_{i-1}
        # Strip x-moments
        Px_i = _zeroth_moment(x_lo, x_hi, mu_x, sx)
        Mx_i = _partial_first_moment(x_lo, x_hi, mu_x, sx)
        # Cx_i = E[(rx − X) 1[X ∈ strip_i]]
        Cx_i = rx * Px_i - Mx_i
        # Dx_i = E[(x_i − X) 1[X ∈ strip_i]]
        Dx_i = x_hi * Px_i - Mx_i
        # Cy_top = E[(ry − Y) 1[Y ≤ y_top]]
        Py_top = Py(y_top)
        My_top = My(y_top)
        Cy_top = ry * Py_top - My_top
        # T1: + Cx_i · Cy_top
        ehvi += Cx_i * Cy_top
        # T2: − Dx_i · (ry − y_top) · Py(y_top)
        # When y_top = ry (i = 1), the factor (ry − y_top) is 0;
        # safe under math.inf because ry is finite and ys[0] = ry.
        if not math.isinf(y_top):
            ehvi -= Dx_i * (ry - y_top) * Py_top
        # T3_j: − (x_{j+1} − x_j) · Px_i · E[(ry − max(y_j, Y)) 1[Y ≤ y_top]]
        # E[(ry − max(y_j, Y)) 1[Y ≤ y_top]]
        # = ry · Py(y_top) − E[max(y_j, Y) 1[Y ≤ y_top]]
        # E[max(y_j, Y) 1[Y ≤ y_top]]
        # = y_j · Py(y_j) + (My(y_top) − My(y_j))
        # since for Y < y_j the max is y_j (contributing y_j · Py(y_j))
        # and for y_j ≤ Y ≤ y_top the max is Y (contributing My(y_top)
        # − My(y_j)).
        for j in range(i, n + 1):
            y_j = ys[j]
            if math.isinf(y_j):
                continue
            dx_j = xs[j + 1] - xs[j]
            e_max = y_j * Py(y_j) + (My_top - My(y_j))
            term = ry * Py_top - e_max
            ehvi -= dx_j * Px_i * term
    return max(0.0, ehvi)


def _phi_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _phi_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _partial_first_moment(
    a: float, b: float, mu: float, sigma: float
) -> float:
    """∫_a^b x · φ(x; μ, σ) dx in closed form.

    Equals σ · (φ(α) − φ(β)) + μ · (Φ(β) − Φ(α))   where α=(a−μ)/σ,
    β=(b−μ)/σ.  φ, Φ are standard normal pdf, cdf.
    """
    if a >= b:
        return 0.0
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    return sigma * (_phi_pdf(alpha) - _phi_pdf(beta)) + mu * (
        _phi_cdf(beta) - _phi_cdf(alpha)
    )


def _zeroth_moment(a: float, b: float, mu: float, sigma: float) -> float:
    """∫_a^b φ(x; μ, σ) dx  =  Φ((b−μ)/σ) − Φ((a−μ)/σ)."""
    if a >= b:
        return 0.0
    return _phi_cdf((b - mu) / sigma) - _phi_cdf((a - mu) / sigma)


def _ehvi_box(
    x_lo: float,
    x_hi: float,
    y_bot: float,
    y_top: float,
    mu_x: float,
    sx: float,
    mu_y: float,
    sy: float,
) -> float:
    """Closed-form EHVI contribution of the box
    [x_lo, x_hi] × (−∞, y_bot] in 2D minimisation.

    Integrand: (x_hi − X)(y_bot − Y) · 1[X ∈ [x_lo, x_hi]] ·
    1[Y ≤ y_bot] · φ(X; μ_x, σ_x) · φ(Y; μ_y, σ_y).

    Factorises into product of two 1D Gaussian moments.

    The x-factor is  x_hi · P(X ∈ [x_lo, x_hi]) − E[X · 1_{X ∈
    [x_lo, x_hi]}]; symmetric for y.
    """
    if x_hi <= x_lo:
        return 0.0
    if y_bot <= -math.inf:
        # No contribution: the box has zero (sub-σ) y-extent.
        return 0.0
    # x-factor
    p_x = _zeroth_moment(x_lo, x_hi, mu_x, sx)
    m_x = _partial_first_moment(x_lo, x_hi, mu_x, sx)
    fx = x_hi * p_x - m_x
    # y-factor
    p_y = _zeroth_moment(-math.inf, y_bot, mu_y, sy)
    m_y = _partial_first_moment(-math.inf, y_bot, mu_y, sy)
    fy = y_bot * p_y - m_y
    return max(0.0, fx) * max(0.0, fy)


def ehvi_monte_carlo(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
    mean: Sequence[float],
    sigma: Sequence[float],
    n_samples: int = 2048,
    seed: int = 0xA61BEEF,
) -> tuple[float, float]:
    """Monte-Carlo Expected Hypervolume Improvement for any M.

    Draws ``n_samples`` from the independent Gaussian belief
    ``N(mean, diag(sigma²))``, computes the per-sample HV
    improvement, returns ``(mean, std_error)``.
    """
    import random as _rnd

    rng = _rnd.Random(seed)
    M = len(reference)
    base_hv, _ = hypervolume(points, reference)
    samples = []
    for _ in range(n_samples):
        x = [rng.gauss(mean[m], max(sigma[m], 1e-12)) for m in range(M)]
        # Clamp to reference to keep MC bounded
        clamped = tuple(min(x[m], reference[m] - 1e-12) for m in range(M))
        new_hv, _ = hypervolume(list(points) + [clamped], reference)
        improvement = max(0.0, new_hv - base_hv)
        samples.append(improvement)
    if not samples:
        return 0.0, 0.0
    mean_imp = sum(samples) / len(samples)
    var = sum((s - mean_imp) ** 2 for s in samples) / max(1, len(samples) - 1)
    se = math.sqrt(var / len(samples)) if samples else 0.0
    return mean_imp, se


# ---------------------------------------------------------------------------
# Scalarisations
# ---------------------------------------------------------------------------


def weighted_sum(
    cost: Sequence[float], weights: Sequence[float]
) -> float:
    """Σ w_i · c_i.  Under minimisation, the argmin coincides with a
    *convex* layer of the Pareto front (Geoffrion 1968)."""
    if len(cost) != len(weights):
        raise ParetoError("cost and weights must share length")
    return sum(w * c for w, c in zip(weights, cost))


def tchebycheff(
    cost: Sequence[float],
    weights: Sequence[float],
    ideal: Sequence[float] | None = None,
) -> float:
    """Steuer 1986 Tchebycheff: ``max_i w_i · (c_i − ideal_i)``.

    Recovers every point on the Pareto front for some weight vector,
    including non-convex regions (the weakness of weighted-sum).
    """
    if len(cost) != len(weights):
        raise ParetoError("cost and weights must share length")
    if ideal is None:
        ideal = [0.0] * len(cost)
    if len(ideal) != len(cost):
        raise ParetoError("ideal must share length with cost")
    return max(
        weights[i] * (cost[i] - ideal[i]) for i in range(len(cost))
    )


def augmented_tchebycheff(
    cost: Sequence[float],
    weights: Sequence[float],
    ideal: Sequence[float] | None = None,
    rho: float = 1e-3,
) -> float:
    """Knowles 2006 ParEGO: ``max_i w_i (c_i − ideal_i) + ρ Σ w_i (c_i
    − ideal_i)``.

    The augmentation breaks weak-Pareto degeneracy: every Pareto-
    optimal point is the strict argmin of *some* augmented Tchebycheff
    scalarisation (Steuer-Choo 1983).
    """
    if rho < 0:
        raise ParetoError("rho must be non-negative")
    if ideal is None:
        ideal = [0.0] * len(cost)
    if len(ideal) != len(cost):
        raise ParetoError("ideal must share length with cost")
    diffs = [weights[i] * (cost[i] - ideal[i]) for i in range(len(cost))]
    return max(diffs) + rho * sum(diffs)


def achievement_scalarising_function(
    cost: Sequence[float],
    reference: Sequence[float],
    weights: Sequence[float] | None = None,
    rho: float = 1e-3,
) -> float:
    """Wierzbicki 1980 achievement scalarising function (ASF):
    ``max_i (c_i − r_i) / w_i + ρ · Σ (c_i − r_i) / w_i``.

    Treats ``reference`` as the *aspiration point* and finds the
    Pareto-optimal point closest to it under the weight vector.
    """
    if weights is None:
        weights = [1.0] * len(cost)
    if len(cost) != len(weights) or len(cost) != len(reference):
        raise ParetoError("cost / weights / reference shape mismatch")
    diffs = [
        (cost[i] - reference[i]) / max(weights[i], 1e-12) for i in range(len(cost))
    ]
    return max(diffs) + rho * sum(diffs)


def pbi_scalarisation(
    cost: Sequence[float],
    weights: Sequence[float],
    ideal: Sequence[float] | None = None,
    theta: float = 5.0,
) -> float:
    """Penalty Boundary Intersection (Das-Dennis 1998; Zhang-Li 2007
    MOEA/D).  Decomposes the Pareto front into ``d₁ + θ · d₂`` where
    ``d₁`` is the projection of ``cost − ideal`` along the unit weight
    vector and ``d₂`` is the orthogonal distance to that ray."""
    if ideal is None:
        ideal = [0.0] * len(cost)
    if len(cost) != len(weights) or len(cost) != len(ideal):
        raise ParetoError("cost / weights / ideal shape mismatch")
    norm_w = math.sqrt(sum(w * w for w in weights))
    if norm_w <= 0:
        raise ParetoError("weights must be a non-zero vector")
    diff = [cost[i] - ideal[i] for i in range(len(cost))]
    d1 = sum(diff[i] * weights[i] for i in range(len(cost))) / norm_w
    proj = [d1 * weights[i] / norm_w for i in range(len(cost))]
    d2 = math.sqrt(sum((diff[i] - proj[i]) ** 2 for i in range(len(cost))))
    return d1 + theta * d2


# ---------------------------------------------------------------------------
# Weight grid generation
# ---------------------------------------------------------------------------


def das_dennis_weights(n_objectives: int, p: int) -> list[tuple[float, ...]]:
    """Das-Dennis 1998 uniformly spaced weight vectors on the M-simplex.

    Generates ``C(p + M − 1, M − 1)`` weight vectors whose entries sum
    to 1 and lie on the discrete grid ``{0, 1/p, 2/p, …, 1}``.
    """
    if n_objectives < 1:
        raise ParetoError("n_objectives must be ≥ 1")
    if p < 1:
        raise ParetoError("p must be ≥ 1")

    def _rec(remaining_p: int, dim: int) -> list[list[int]]:
        if dim == 1:
            return [[remaining_p]]
        out: list[list[int]] = []
        for k in range(remaining_p + 1):
            for rest in _rec(remaining_p - k, dim - 1):
                out.append([k] + rest)
        return out

    grids = _rec(p, n_objectives)
    return [tuple(g[i] / p for i in range(n_objectives)) for g in grids]


def uniform_simplex_weights(
    n: int, n_objectives: int, seed: int = 0xA61BEEF
) -> list[tuple[float, ...]]:
    """Marsaglia 1972 / standard Dirichlet(1, …, 1) ↔ Gumbel-trick
    uniform sampling from the M-simplex.  Returns ``n`` weight
    vectors.
    """
    import random as _rnd

    rng = _rnd.Random(seed)
    out: list[tuple[float, ...]] = []
    for _ in range(n):
        # Sample iid Exp(1) and normalise → Dirichlet(1, …, 1)
        # = uniform on the simplex.
        es = [-math.log(max(rng.random(), 1e-300)) for _ in range(n_objectives)]
        s = sum(es)
        out.append(tuple(e / s for e in es))
    return out


# ---------------------------------------------------------------------------
# Convergence + diversity metrics
# ---------------------------------------------------------------------------


def inverted_generational_distance(
    front: Sequence[Sequence[float]],
    reference_set: Sequence[Sequence[float]],
) -> float:
    """Coello-Sierra 2004 IGD: mean distance from each reference-set
    point to its nearest point in ``front``.

    Lower is better; zero iff ``front`` covers ``reference_set``.
    """
    if not reference_set:
        return float("nan")
    if not front:
        return float("inf")
    total = 0.0
    for r in reference_set:
        best = math.inf
        for p in front:
            d = math.sqrt(sum((p[i] - r[i]) ** 2 for i in range(len(r))))
            if d < best:
                best = d
        total += best
    return total / len(reference_set)


def generational_distance(
    front: Sequence[Sequence[float]],
    reference_set: Sequence[Sequence[float]],
) -> float:
    """Van Veldhuizen 1999 GD: mean distance from each ``front`` point
    to its nearest reference-set point.

    Distinct from IGD: GD measures *convergence* of the front to the
    reference set, IGD measures *coverage*.
    """
    if not front:
        return float("nan")
    if not reference_set:
        return float("inf")
    total = 0.0
    for p in front:
        best = math.inf
        for r in reference_set:
            d = math.sqrt(sum((p[i] - r[i]) ** 2 for i in range(len(r))))
            if d < best:
                best = d
        total += best
    return total / len(front)


def spacing(front: Sequence[Sequence[float]]) -> float:
    """Schott 1995 spacing metric.

    ``√( (1 / (N − 1)) Σ (d_i − d̄)² )`` where ``d_i`` is the L1
    distance from point i to its nearest neighbour in the front.

    Zero iff every neighbour-distance is identical.
    """
    N = len(front)
    if N < 2:
        return 0.0
    ds: list[float] = []
    for i in range(N):
        best = math.inf
        for j in range(N):
            if j == i:
                continue
            d = sum(abs(front[i][k] - front[j][k]) for k in range(len(front[i])))
            if d < best:
                best = d
        ds.append(best)
    d_bar = sum(ds) / N
    return math.sqrt(sum((d - d_bar) ** 2 for d in ds) / (N - 1))


def maximum_spread(front: Sequence[Sequence[float]]) -> float:
    """Zitzler 1999 maximum-spread metric.

    ``√( Σ_m (max c_m − min c_m)² )``.  Measures the extent of the
    front in objective space.
    """
    if not front:
        return 0.0
    M = len(front[0])
    diam = 0.0
    for m in range(M):
        mn = min(p[m] for p in front)
        mx = max(p[m] for p in front)
        diam += (mx - mn) ** 2
    return math.sqrt(diam)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def hrms_half_width(n: int, conf: float = 0.95) -> float:
    """Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid half-width."""
    if n <= 1:
        return float("inf")
    delta = 1.0 - conf
    inner = math.log(math.log(2.0 * n) + math.e) + 0.75 * math.log(10.4 / delta)
    return math.sqrt(max(inner / (2.0 * n), 0.0))


def empirical_bernstein_half_width(
    n: int, variance: float, conf: float = 0.95, range_: float = 1.0
) -> float:
    """Maurer-Pontil 2009 empirical-Bernstein half-width."""
    if n <= 1:
        return float("inf")
    delta = 1.0 - conf
    return math.sqrt(2.0 * variance * math.log(2.0 / delta) / n) + (
        7.0 * range_ * math.log(2.0 / delta) / (3.0 * (n - 1))
    )


def hoeffding_half_width(
    n: int, conf: float = 0.95, range_: float = 1.0
) -> float:
    """Hoeffding 1963 half-width on a [0, range_]-bounded random
    variable."""
    if n <= 0:
        return float("inf")
    delta = 1.0 - conf
    return range_ * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


# ---------------------------------------------------------------------------
# Fingerprint chain
# ---------------------------------------------------------------------------


_GENESIS_PREFIX = b"agi.pareto.v1\x00"


def ledger_root(secret_key: bytes | None = None) -> str:
    seed = _GENESIS_PREFIX + (secret_key or b"")
    return hashlib.sha256(seed).hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    def _q(o: Any) -> Any:
        if isinstance(o, float):
            if math.isnan(o):
                return "NaN"
            if math.isinf(o):
                return "Infinity" if o > 0 else "-Infinity"
            return float(repr(o))
        if isinstance(o, dict):
            return {str(k): _q(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        if isinstance(o, (list, tuple)):
            return [_q(x) for x in o]
        if isinstance(o, (bytes, bytearray)):
            return o.hex()
        return o

    return json.dumps(_q(payload), sort_keys=True, separators=(",", ":")).encode()


def _hash_entry(parent: str, payload: dict[str, Any], hmac_key: bytes | None = None) -> str:
    body = _canonical(payload)
    block = parent.encode() + b"|" + body
    if hmac_key:
        return hmac.new(hmac_key, block, hashlib.sha256).hexdigest()
    return hashlib.sha256(block).hexdigest()


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


EventPublisher = Callable[[str, dict[str, Any]], None]


class Pareto:
    """Multi-objective optimization as a runtime primitive.

    Threadsafe at the API surface: a single re-entrant lock guards
    every mutation of the candidate population.

    Internally Pareto stores every candidate in *minimisation-
    normalised* coordinates ``minim_cost = sign · cost``, so a single
    set of front-extraction, hypervolume, and EHVI subroutines suffices
    for any mix of min/max objectives.
    """

    def __init__(
        self,
        config: ParetoConfig | None = None,
        *,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.config = config or ParetoConfig()
        self._publisher = publisher
        self._lock = threading.RLock()
        self._candidates: dict[str, Candidate] = {}
        # Minimisation-normalised cache, updated on observe/remove.
        self._minim: dict[str, tuple[float, ...]] = {}
        self._violation: dict[str, float] = {}
        # Hypervolume history for progress certificate.
        self._hv_history: list[tuple[float, float]] = []  # (ts, hv)
        self._chain_head: str = ledger_root(self.config.hmac_key)
        self._started_ts = time.time()
        self._publish(
            PARETO_STARTED,
            {
                "ts": self._started_ts,
                "n_objectives": self.config.n_objectives,
                "senses": list(self.config.senses),
            },
        )

    # ------------------------------------------------------------------
    # Event publishing + chain helpers
    # ------------------------------------------------------------------

    def _publish(self, kind: str, payload: dict[str, Any]) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher(kind, payload)
        except Exception:
            pass

    def _advance_chain(self, payload: dict[str, Any]) -> str:
        self._chain_head = _hash_entry(self._chain_head, payload, self.config.hmac_key)
        return self._chain_head

    @property
    def chain_head(self) -> str:
        with self._lock:
            return self._chain_head

    # ------------------------------------------------------------------
    # Observation surface
    # ------------------------------------------------------------------

    def observe(
        self,
        candidate_id: str,
        cost: Sequence[float],
        *,
        constraints: Sequence[float] = (),
        weight: float = 1.0,
    ) -> None:
        """Record (or overwrite) a candidate."""
        with self._lock:
            self._validate_cost(cost)
            if len(constraints) != self.config.constraint_dim:
                raise InvalidCandidate(
                    "constraints length must match config.constraint_dim "
                    f"({len(constraints)} vs {self.config.constraint_dim})"
                )
            for v in constraints:
                if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    raise InvalidCandidate("constraints must be finite numbers")
            if weight <= 0 or math.isnan(weight) or math.isinf(weight):
                raise InvalidCandidate("weight must be a finite positive number")
            cand = Candidate(
                candidate_id=candidate_id,
                cost=tuple(float(c) for c in cost),
                constraints=tuple(float(c) for c in constraints),
                weight=float(weight),
            )
            self._candidates[candidate_id] = cand
            self._minim[candidate_id] = _to_minim(cand.cost, self.config.senses)
            self._violation[candidate_id] = sum(
                max(0.0, c) for c in cand.constraints
            )
            payload = {
                "candidate_id": candidate_id,
                "cost": list(cand.cost),
                "constraints": list(cand.constraints),
                "weight": cand.weight,
                "head": self._chain_head,
            }
            self._advance_chain({"k": PARETO_OBSERVED, **payload})
            self._publish(PARETO_OBSERVED, {**payload, "head": self._chain_head})

    def remove(self, candidate_id: str) -> None:
        with self._lock:
            if candidate_id not in self._candidates:
                raise UnknownCandidate(candidate_id)
            del self._candidates[candidate_id]
            self._minim.pop(candidate_id, None)
            self._violation.pop(candidate_id, None)
            payload = {"candidate_id": candidate_id, "head": self._chain_head}
            self._advance_chain({"k": PARETO_REMOVED, **payload})
            self._publish(PARETO_REMOVED, {**payload, "head": self._chain_head})

    def clear(self) -> None:
        with self._lock:
            self._candidates.clear()
            self._minim.clear()
            self._violation.clear()
            self._hv_history.clear()
            self._chain_head = ledger_root(self.config.hmac_key)
            self._publish(PARETO_CLEARED, {"head": self._chain_head})

    def __len__(self) -> int:
        with self._lock:
            return len(self._candidates)

    def candidates(self) -> tuple[Candidate, ...]:
        with self._lock:
            return tuple(self._candidates.values())

    def get(self, candidate_id: str) -> Candidate:
        with self._lock:
            if candidate_id not in self._candidates:
                raise UnknownCandidate(candidate_id)
            return self._candidates[candidate_id]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_cost(self, cost: Sequence[float]) -> None:
        if len(cost) != self.config.n_objectives:
            raise InvalidCandidate(
                "cost length must match config.n_objectives "
                f"({len(cost)} vs {self.config.n_objectives})"
            )
        for v in cost:
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise InvalidCandidate("cost entries must be finite numbers")

    # ------------------------------------------------------------------
    # Frontier extraction
    # ------------------------------------------------------------------

    def frontier(self, rank: int = 0) -> FrontierReport:
        """Return the rank-r Pareto front (rank 0 is non-dominated).

        Costs are returned in *user* coordinates.  Within the front,
        candidates are sorted by crowding distance descending so the
        caller can take the top-K and get the most diverse subset.
        """
        with self._lock:
            if not self._candidates:
                raise InsufficientData("no candidates observed")
            ids = list(self._candidates.keys())
            minim_points = [self._minim[c] for c in ids]
            violations = [self._violation[c] for c in ids]
            fronts = non_dominated_sort(minim_points, violations)
            if rank < 0 or rank >= len(fronts):
                raise InvalidQuery(
                    f"rank {rank} out of range [0, {len(fronts) - 1}]"
                )
            indices = fronts[rank]
            sub_points = [minim_points[i] for i in indices]
            crowd = crowding_distance(sub_points)
            order = sorted(
                range(len(indices)),
                key=lambda k: (-crowd[k], indices[k]),
            )
            sorted_ids = tuple(ids[indices[k]] for k in order)
            sorted_costs = tuple(
                _from_minim(sub_points[k], self.config.senses) for k in order
            )
            sorted_crowd = tuple(crowd[k] for k in order)
            payload = {
                "rank": rank,
                "n_fronts": len(fronts),
                "n_candidates": len(self._candidates),
                "front_ids": list(sorted_ids),
                "head": self._chain_head,
            }
            self._advance_chain({"k": PARETO_FRONTIER, **payload})
            self._publish(PARETO_FRONTIER, {**payload, "head": self._chain_head})
            return FrontierReport(
                rank=rank,
                candidates=sorted_ids,
                costs=sorted_costs,
                crowding=sorted_crowd,
                fingerprint_hash=self._chain_head,
            )

    def rank_layers(self) -> tuple[tuple[str, ...], ...]:
        """Return every Pareto layer as a tuple of tuples of ids.

        Layer 0 is the non-dominated front.  Useful for the
        coordination engine to plan over the *full ladder* of
        dominance, not just the front.
        """
        with self._lock:
            if not self._candidates:
                return ()
            ids = list(self._candidates.keys())
            minim_points = [self._minim[c] for c in ids]
            violations = [self._violation[c] for c in ids]
            fronts = non_dominated_sort(minim_points, violations)
            return tuple(tuple(ids[i] for i in f) for f in fronts)

    # ------------------------------------------------------------------
    # Hypervolume + EHVI
    # ------------------------------------------------------------------

    def _resolve_reference(
        self, reference: Sequence[float] | None
    ) -> tuple[float, ...]:
        if reference is not None:
            if len(reference) != self.config.n_objectives:
                raise InvalidQuery(
                    "reference length must match config.n_objectives "
                    f"({len(reference)} vs {self.config.n_objectives})"
                )
            return tuple(float(r) for r in reference)
        if self.config.reference is not None:
            return tuple(float(r) for r in self.config.reference)
        # Default: nadir + 10% padding in *minimisation* coordinates,
        # mapped back to user coordinates.
        if not self._minim:
            raise InsufficientData("cannot derive reference with no candidates")
        M = self.config.n_objectives
        nadir = [
            max(self._minim[c][m] for c in self._minim) for m in range(M)
        ]
        ideal = [
            min(self._minim[c][m] for c in self._minim) for m in range(M)
        ]
        spread = [nadir[m] - ideal[m] for m in range(M)]
        pad = [max(0.1 * s, 1e-6) for s in spread]
        ref_minim = tuple(nadir[m] + pad[m] for m in range(M))
        return _from_minim(ref_minim, self.config.senses)

    def hypervolume(
        self,
        *,
        reference: Sequence[float] | None = None,
        only_feasible: bool = True,
    ) -> HypervolumeReport:
        """Hypervolume of the *non-dominated* front."""
        with self._lock:
            if not self._candidates:
                raise InsufficientData("no candidates observed")
            ref_user = self._resolve_reference(reference)
            ref_minim = _to_minim(ref_user, self.config.senses)
            ids = list(self._candidates.keys())
            minim_points = [self._minim[c] for c in ids]
            violations = [self._violation[c] for c in ids]
            if only_feasible:
                feasible_indices = [i for i, v in enumerate(violations) if v <= 0]
            else:
                feasible_indices = list(range(len(ids)))
            if not feasible_indices:
                hv = 0.0
                algo = "no_feasible"
                n = 0
            else:
                fronts = non_dominated_sort(
                    [minim_points[i] for i in feasible_indices],
                    [violations[i] for i in feasible_indices],
                )
                front_local_idx = fronts[0]
                front_points = [
                    minim_points[feasible_indices[i]] for i in front_local_idx
                ]
                hv, algo = hypervolume(front_points, ref_minim)
                n = len(front_points)
            self._hv_history.append((time.time(), hv))
            payload = {
                "hv": hv,
                "ref": list(ref_user),
                "n_points": n,
                "n_objectives": self.config.n_objectives,
                "algo": algo,
                "head": self._chain_head,
            }
            self._advance_chain({"k": PARETO_REPORTED, **payload})
            self._publish(PARETO_REPORTED, {**payload, "head": self._chain_head})
            return HypervolumeReport(
                hypervolume=hv,
                reference=ref_user,
                n_points=n,
                n_objectives=self.config.n_objectives,
                algorithm=algo,
                fingerprint_hash=self._chain_head,
            )

    def expected_hypervolume_improvement(
        self,
        mean: Sequence[float],
        sigma: Sequence[float],
        *,
        candidate_id: str | None = None,
        reference: Sequence[float] | None = None,
        n_samples: int = 2048,
    ) -> EHVIReport:
        """Expected Hypervolume Improvement under independent
        Gaussian belief ``x ∼ N(mean, diag(sigma²))``.

        Closed-form for M=2 (Emmerich 2008), Monte-Carlo for M ≥ 3
        with a standard-error report.
        """
        with self._lock:
            if len(mean) != self.config.n_objectives or len(sigma) != self.config.n_objectives:
                raise InvalidQuery(
                    "mean / sigma length must match config.n_objectives"
                )
            for s in sigma:
                if s < 0 or math.isnan(s) or math.isinf(s):
                    raise InvalidQuery("sigma entries must be ≥ 0 and finite")
            ref_user = self._resolve_reference(reference)
            ref_minim = _to_minim(ref_user, self.config.senses)
            mean_minim = _to_minim(mean, self.config.senses)
            # sigma is invariant under sign flip
            sigma_minim = tuple(float(s) for s in sigma)
            if not self._candidates:
                front_points: list[tuple[float, ...]] = []
            else:
                ids = list(self._candidates.keys())
                feasible_minim = [
                    self._minim[c] for c in ids if self._violation[c] <= 0
                ]
                feasible_viol = [
                    0.0 for c in ids if self._violation[c] <= 0
                ]
                if feasible_minim:
                    fronts = non_dominated_sort(feasible_minim, feasible_viol)
                    front_points = [feasible_minim[i] for i in fronts[0]]
                else:
                    front_points = []
            if self.config.n_objectives == 2:
                ehvi = ehvi_2d(
                    front_points, ref_minim, mean_minim, sigma_minim
                )
                se = 0.0
                algo = "closed_form_2d"
                n_used = 0
            else:
                ehvi, se = ehvi_monte_carlo(
                    front_points,
                    ref_minim,
                    mean_minim,
                    sigma_minim,
                    n_samples=n_samples,
                    seed=self.config.rng_seed or 0xA61BEEF,
                )
                algo = f"monte_carlo_{self.config.n_objectives}d"
                n_used = n_samples
            payload = {
                "candidate_id": candidate_id,
                "mean": list(mean),
                "sigma": list(sigma),
                "ehvi": ehvi,
                "ehvi_se": se,
                "n_samples": n_used,
                "algo": algo,
                "head": self._chain_head,
            }
            self._advance_chain({"k": "ehvi", **payload})
            self._publish(PARETO_REPORTED, {**payload, "head": self._chain_head})
            return EHVIReport(
                candidate_id=candidate_id,
                mean=tuple(float(m) for m in mean),
                sigma=tuple(float(s) for s in sigma),
                ehvi=ehvi,
                ehvi_se=se,
                n_samples=n_used,
                algorithm=algo,
                fingerprint_hash=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Scalarisations
    # ------------------------------------------------------------------

    def scalarise(
        self,
        method: str,
        weights: Sequence[float],
        *,
        ideal: Sequence[float] | None = None,
        reference: Sequence[float] | None = None,
        rho: float = 1e-3,
        theta: float = 5.0,
    ) -> ScalarisationReport:
        """Compute the scalarisation value for every candidate and
        return the argmin (under all-minimisation)."""
        with self._lock:
            if method not in KNOWN_SCALARISATIONS:
                raise InvalidQuery(
                    f"unknown scalarisation {method!r}; expected one of "
                    f"{sorted(KNOWN_SCALARISATIONS)}"
                )
            if len(weights) != self.config.n_objectives:
                raise InvalidQuery(
                    "weights length must match config.n_objectives"
                )
            if not self._candidates:
                raise InsufficientData("no candidates observed")
            # Resolve ideal in minimisation coordinates.
            if ideal is None:
                ideal_minim = tuple(
                    min(self._minim[c][m] for c in self._minim)
                    for m in range(self.config.n_objectives)
                )
            else:
                ideal_minim = _to_minim(ideal, self.config.senses)
            ref_minim: tuple[float, ...] | None
            if method == SCALAR_ASF:
                if reference is None:
                    reference = ideal if ideal is not None else _from_minim(
                        ideal_minim, self.config.senses
                    )
                ref_minim = _to_minim(reference, self.config.senses)
            else:
                ref_minim = None
            values: dict[str, float] = {}
            for cid, mc in self._minim.items():
                if method == SCALAR_WEIGHTED_SUM:
                    v = weighted_sum(mc, weights)
                elif method == SCALAR_TCHEBYCHEFF:
                    v = tchebycheff(mc, weights, ideal_minim)
                elif method == SCALAR_AUG_TCHEBYCHEFF:
                    v = augmented_tchebycheff(mc, weights, ideal_minim, rho)
                elif method == SCALAR_ASF:
                    assert ref_minim is not None
                    v = achievement_scalarising_function(
                        mc, ref_minim, weights, rho
                    )
                elif method == SCALAR_PBI:
                    v = pbi_scalarisation(mc, weights, ideal_minim, theta)
                else:
                    raise InvalidQuery(method)
                values[cid] = v
            argmin = min(values, key=lambda k: (values[k], k))
            payload = {
                "method": method,
                "weights": list(weights),
                "argmin": argmin,
                "head": self._chain_head,
            }
            self._advance_chain({"k": "scalarise", **payload})
            self._publish(PARETO_REPORTED, {**payload, "head": self._chain_head})
            return ScalarisationReport(
                method=method,
                weights=tuple(float(w) for w in weights),
                values=values,
                argmin_candidate=argmin,
                fingerprint_hash=self._chain_head,
            )

    def sweep(
        self,
        *,
        method: str = SCALAR_TCHEBYCHEFF,
        p: int = 4,
        rho: float = 1e-3,
        theta: float = 5.0,
    ) -> tuple[ScalarisationReport, ...]:
        """Das-Dennis weight sweep over the simplex.  Returns one
        ScalarisationReport per weight vector — the canonical way to
        expose the full Pareto front via a single-objective inner
        loop."""
        with self._lock:
            grid = das_dennis_weights(self.config.n_objectives, p)
            return tuple(
                self.scalarise(method, w, rho=rho, theta=theta) for w in grid
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(
        self,
        *,
        reference_set: Sequence[Sequence[float]] | None = None,
    ) -> MetricsReport:
        """Convergence + diversity metrics on the non-dominated front.

        If ``reference_set`` is supplied (e.g. the true Pareto front
        on a benchmark, or a hand-curated target panel), IGD and GD
        are reported; otherwise they are ``None``.
        """
        with self._lock:
            if not self._candidates:
                raise InsufficientData("no candidates observed")
            ids = list(self._candidates.keys())
            minim_points = [self._minim[c] for c in ids]
            violations = [self._violation[c] for c in ids]
            fronts = non_dominated_sort(minim_points, violations)
            front = [minim_points[i] for i in fronts[0]]
            if reference_set is not None:
                ref_minim = [
                    _to_minim(r, self.config.senses) for r in reference_set
                ]
                igd = inverted_generational_distance(front, ref_minim)
                gd = generational_distance(front, ref_minim)
            else:
                igd = None
                gd = None
            sp = spacing(front)
            ms = maximum_spread(front)
            payload = {
                "igd": igd,
                "gd": gd,
                "spacing": sp,
                "max_spread": ms,
                "n_points": len(front),
                "head": self._chain_head,
            }
            self._advance_chain({"k": "metrics", **payload})
            self._publish(PARETO_REPORTED, {**payload, "head": self._chain_head})
            return MetricsReport(
                igd=igd,
                gd=gd,
                spacing=sp,
                max_spread=ms,
                n_points=len(front),
                fingerprint_hash=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Anytime-valid progress certificate
    # ------------------------------------------------------------------

    def certify_progress(
        self,
        epsilon: float = 1e-3,
        *,
        window: int = 8,
    ) -> ProgressCertificate:
        """Anytime-valid certificate that hypervolume growth has fallen
        below ``epsilon``.

        Uses the last ``window`` hypervolume readings to estimate the
        per-step improvement; HRMS 2021 anytime-valid bounds on the
        increment under the assumption that the absolute hypervolume is
        bounded by an *a priori* known upper bound derived from the
        reference point.
        """
        with self._lock:
            if not self._hv_history:
                # Synthesise one entry so the certificate is well-
                # defined the first time it is called.
                self.hypervolume()
            hv_now = self._hv_history[-1][1]
            window = max(2, min(window, len(self._hv_history)))
            recent = [v for _, v in self._hv_history[-window:]]
            increments = [
                max(0.0, recent[i + 1] - recent[i])
                for i in range(len(recent) - 1)
            ]
            if not increments:
                delta_hv = 0.0
                lcb = 0.0
                ucb = 0.0
                converged = True
            else:
                mean_inc = sum(increments) / len(increments)
                # Normalise to [0, 1] using the running max; HRMS
                # requires bounded support.
                hi = max(increments) if increments else 1.0
                hi = max(hi, 1e-9)
                norm = [x / hi for x in increments]
                m = sum(norm) / len(norm)
                hw = hrms_half_width(len(norm), self.config.confidence)
                lcb = max(0.0, m - hw) * hi
                ucb = (m + hw) * hi
                delta_hv = mean_inc
                converged = ucb < epsilon
            payload = {
                "hv_now": hv_now,
                "delta_hv": delta_hv,
                "delta_hv_lcb": lcb,
                "delta_hv_ucb": ucb,
                "epsilon": epsilon,
                "converged": converged,
                "head": self._chain_head,
            }
            self._advance_chain({"k": PARETO_CERTIFIED, **payload})
            self._publish(PARETO_CERTIFIED, {**payload, "head": self._chain_head})
            return ProgressCertificate(
                hv_now=hv_now,
                hv_history=tuple(v for _, v in self._hv_history),
                delta_hv=delta_hv,
                delta_hv_lcb=lcb,
                delta_hv_ucb=ucb,
                epsilon=epsilon,
                converged=converged,
                confidence=self.config.confidence,
                fingerprint_hash=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """One-shot status report: front + HV + metrics + chain head."""
        with self._lock:
            if not self._candidates:
                return {
                    "n_candidates": 0,
                    "front": [],
                    "hypervolume": None,
                    "chain_head": self._chain_head,
                }
            front = self.frontier(rank=0)
            hv = self.hypervolume()
            metrics = self.metrics()
            return {
                "n_candidates": len(self._candidates),
                "n_objectives": self.config.n_objectives,
                "senses": list(self.config.senses),
                "front": [
                    {"id": cid, "cost": list(c), "crowding": cr}
                    for cid, c, cr in zip(front.candidates, front.costs, front.crowding)
                ],
                "hypervolume": hv.hypervolume,
                "hv_reference": list(hv.reference),
                "spacing": metrics.spacing,
                "max_spread": metrics.max_spread,
                "chain_head": self._chain_head,
            }

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "chain_head": self._chain_head,
                "senses": list(self.config.senses),
                "candidates": [
                    {
                        "id": cand.candidate_id,
                        "cost": list(cand.cost),
                        "constraints": list(cand.constraints),
                        "weight": cand.weight,
                        "ts": cand.ts,
                    }
                    for cand in self._candidates.values()
                ],
                "hv_history": [
                    [ts, hv] for ts, hv in self._hv_history
                ],
            }

    def restore(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._candidates.clear()
            self._minim.clear()
            self._violation.clear()
            self._hv_history.clear()
            senses = tuple(snapshot.get("senses", self.config.senses))
            if len(senses) != self.config.n_objectives:
                raise InvalidConfig(
                    "snapshot senses length does not match config"
                )
            for c in snapshot.get("candidates", []):
                cand = Candidate(
                    candidate_id=c["id"],
                    cost=tuple(c["cost"]),
                    constraints=tuple(c.get("constraints", ())),
                    weight=float(c.get("weight", 1.0)),
                    ts=float(c.get("ts", time.time())),
                )
                self._candidates[cand.candidate_id] = cand
                self._minim[cand.candidate_id] = _to_minim(
                    cand.cost, self.config.senses
                )
                self._violation[cand.candidate_id] = sum(
                    max(0.0, x) for x in cand.constraints
                )
            self._hv_history = [
                (float(ts), float(hv)) for ts, hv in snapshot.get("hv_history", [])
            ]
            self._chain_head = str(
                snapshot.get("chain_head", ledger_root(self.config.hmac_key))
            )
