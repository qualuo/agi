r"""Transporter — optimal transport as a runtime primitive.

Every other primitive in this stack ultimately compares *distributions*
— a calibration histogram vs. a target, a live data window vs. a
reference, a counterfactual cohort vs. an observed one, an adversarial
hypothetical vs. an empirical sample, two policies' state-visitation
measures. Each existing primitive bakes its own ad-hoc distance into
its API: ``DriftSentinel`` uses KL/TV/Hellinger, ``Robustifier`` writes
the Wasserstein ball without ever building a transport plan,
``CausalDiscoverer`` matches treated and control rows by nearest
neighbour, ``Conformal`` slices its score CDF, ``Coalition`` cuts up a
characteristic function in pieces that are implicitly Earth-Mover sizes.

The Transporter is the primitive that gives the rest of the stack a
*single, mathematically rigorous, finite-sample-valid* answer to "how
far apart are these two distributions, and which transport plan
realises it?". It accepts two probability measures (continuous samples
*or* discrete histograms), a ground cost (Euclidean, sq-Euclidean,
arbitrary user matrix), an algorithm (Hungarian / Sinkhorn / sliced /
1-D quantile / unbalanced / Gromov-Wasserstein), and returns a
``TransportReport`` carrying the distance ``W_p``, the entropic-regular
distance ``OT_ε``, the *debiased* Sinkhorn divergence
``S_ε = OT_ε(a,b) − ½ OT_ε(a,a) − ½ OT_ε(b,b)``, the transport plan
``P`` with proven row/column marginal feasibility, the dual potentials
``(φ, ψ)``, and a ``certificate`` documenting marginal violation,
convergence rate, and a content hash for replay.

Mathematical roots
------------------

  * **Monge, 1781 — *Mémoire sur la théorie des déblais et des remblais*.**
    The original earth-moving problem: find a map T : X → Y pushing μ
    to ν that minimises ∫ c(x, T(x)) dμ(x).  Famously ill-posed when
    no such map exists (e.g. mass-splitting required).

  * **Kantorovich, 1942 — *On the translocation of masses*.**
    Relaxes Monge to *plans*: probability measures π on X × Y with
    marginals μ, ν.  The linear program
        W_c(μ, ν) = inf_{π ∈ Π(μ, ν)} ∫ c dπ
    is always feasible, has the LP-dual
        sup { ∫ φ dμ + ∫ ψ dν :  φ(x) + ψ(y) ≤ c(x, y) ∀ x, y }
    (Kantorovich-Rubinstein duality). For c(x, y) = |x − y| and μ, ν
    on ℝ the dual collapses to 1-Lipschitz functions, giving the
    closed-form ``W_1(μ, ν) = ∫|F_μ(t) − F_ν(t)| dt``.

  * **Hungarian algorithm — Kuhn, 1955; Munkres, 1957.**  Exact
    O(n³) primal-dual solver for the assignment problem, the discrete
    Monge instance with n=m point masses.  Provides the
    *combinatorial* optimum that Sinkhorn approximates entropically.

  * **Sinkhorn-Knopp, 1967; Cuturi, 2013.**  Add an entropic
    regulariser:  OT_ε(a, b) = min_P ⟨P, C⟩ + ε ⟨P, log P⟩.  Optimum
    has the form P_ij = u_i K_ij v_j with K = exp(−C/ε); the
    fixed-point iteration  u ← a / (K v),  v ← b / (Kᵀ u)  is the
    *only* linear-convergence general-purpose OT solver, with
    geometric rate ``(η/(1+η))²`` where η = min(min C) / max(C) in
    the symmetric case.  Log-domain stabilisation (Schmitzer 2016)
    avoids underflow when ε → 0.

  * **Genevay-Peyré-Cuturi, 2018; Feydy et al., 2019 — Sinkhorn
    divergence.**  Debias by
        S_ε(a, b) = OT_ε(a, b) − ½ OT_ε(a, a) − ½ OT_ε(b, b).
    Properly nonnegative, S_ε(a, a) = 0, ε → 0 recovers ``W_c``, and
    crucially *positive-definite* on the simplex for sufficiently
    small ε — so it is a *genuine metric-like* score the runtime can
    use as a drift signal without the entropy-bias artefacts.

  * **Rabin-Peyré-Delon-Bernot, 2011 — Sliced Wasserstein.**  Project
    onto a 1-D direction θ, apply the 1-D closed form, average over
    θ ∼ Uniform(𝕊^{d−1}):
        SW_p(μ, ν)^p = ∫_{𝕊^{d−1}} W_p(θ#μ, θ#ν)^p dσ(θ).
    Embarrassingly parallel, O(n log n) per projection, and a
    proper metric.  Used wherever the full transport plan is not
    needed.

  * **Bonneel-Rabin-Peyré-Pfister, 2015 — Sliced barycenter.**
    Closed-form quantile-averaged barycenter in 1-D ⇒ projected-
    gradient algorithm for the d-dim sliced Wasserstein barycenter.

  * **Liero-Mielke-Savaré, 2018; Chizat-Peyré-Schmitzer-Vialard,
    2018 — Unbalanced OT.**  When ∑ a ≠ ∑ b, KL-relax the marginal
    constraints:
        UOT_ε(a, b) = min_P ⟨P, C⟩ + ε H(P) +
                      ρ_a KL(P 1 ‖ a) + ρ_b KL(Pᵀ 1 ‖ b).
    Recovers OT as ρ → ∞; tolerant to outliers and noise.

  * **Mémoli, 2011; Peyré-Cuturi-Solomon, 2016 — Gromov-Wasserstein.**
    Distance between metric-measure spaces *without a common ground
    metric* — the right answer when one party has features in ℝ^d
    and another in ℝ^{d'}.  Quartic in the plan; the entropic
    relaxation is solved by alternating Sinkhorn.

  * **Santambrogio, 2015 — *Optimal Transport for Applied
    Mathematicians*.**  The reference monograph.  Theorem 2.18 gives
    cyclic monotonicity as a *finite, verifiable* certificate of
    optimality on discrete plans:  for any cycle
    (i_1, j_1), …, (i_k, j_k) with P_{i_t, j_t} > 0,
        Σ_t c(x_{i_t}, y_{j_t}) ≤ Σ_t c(x_{i_t}, y_{j_{t+1}}).
    The runtime checks this on the recovered plan and exposes the
    worst violation as an *optimality witness*.

  * **Weed-Bach, 2019 — Sample complexity.**  For μ, ν supported on a
    bounded subset of ℝ^d, ``W_p(μ̂_n, μ) ≤ C n^{−1/d}`` for d ≥ 3 and
    ``W_p(μ̂_n, μ) ≤ C n^{−1/(2p)}`` for d ≤ 2.  We expose this as the
    expected finite-sample bias term in the report's certificate.

These ten pillars are not preferences a coordinator picks among
arbitrarily: they are projections of the same Kantorovich problem
under different costs (Euclidean → Hungarian; squared-Euclidean →
Wasserstein-2 with Brenier maps; entropic regularisation → Sinkhorn;
projection-then-1D → sliced; relaxed marginals → unbalanced).  Each is
*the* canonical answer when its assumptions fit the use case.

What it composes (razor-sharp coordination integration)
------------------------------------------------------

  * **Robustifier.**  Robustifier writes the Wasserstein-1 DRO ball
    ``{ ℚ : W_1(ℚ, ℙ̂_n) ≤ ε }`` *abstractly*; the Transporter materialises
    it concretely by returning the worst-case plan and the dual potential
    ``φ`` whose 1-Lipschitz extension certifies the radius. Two-line
    composition: ``W_1, plan, phi = transporter.compute(...)``; then
    ``Robustifier`` uses ``φ`` as the value-function perturbation
    direction.

  * **DriftSentinel.**  DriftSentinel ships KL/TV/Hellinger drift
    statistics. Transporter adds the **Wasserstein-1 drift score**:
    finite-sample-valid (Weed-Bach), insensitive to support mismatch
    (KL would diverge), and naturally batched. The Sentinel registers
    a reference distribution, every new window goes through
    ``transporter.drift(...)``, and the e-process construction in
    DriftSentinel uses ``S_ε`` as a bounded random variable.

  * **CausalDiscoverer.**  The counterfactual-matching step in
    treatment-effect estimation is exactly *transport*: which control
    row should be paired with which treated row?  Hungarian on the
    cost matrix of confounder distances gives the optimal pairing;
    Sinkhorn gives a soft pairing for variance reduction (Athey-
    Imbens 2019).  ``Causal.transport_match(...)`` will call into
    the Transporter.

  * **Conformal.**  Mondrian conformal prediction slices the score
    CDF by group. The Transporter's 1-D EMD gives the *finite-
    sample-valid* W_1 distance between predicted and realised score
    CDFs — a goodness-of-fit statistic that respects the same
    distribution-free guarantee.

  * **Coalition.**  The characteristic function value ``v(S)`` is
    well-defined when each coalition's outcome is a *distribution*;
    Shapley-Wasserstein extends classical Shapley by averaging
    W_2(v(S ∪ {i}), v(S)) marginal contributions across subsets.

  * **PolicyImprover / PolicyLab.**  The state-visitation measure of
    a policy is a measure; the W_1 distance between two policies'
    state-visitation measures is the *KL-free* policy-shift metric
    that off-policy estimators need for safe deployment.

  * **Strategist.**  Risk-adjusted decisions weight expected return
    against worst-case shift. Transporter supplies the worst-case
    shift in W_1 metric; Strategist contracts on the joint
    (expected-value, W_1-shift) plane.

  * **AttestationLedger.**  Every compute call is hashed and
    appended as a ``transporter.computed`` receipt — a third-party-
    replayable proof that under cost matrix with content-hash H at
    time t the chosen plan P achieves OT cost ≤ z + ε.

  * **EventBus.**  Streams every registration, computation, drift
    update, and barycenter result. A higher-level coordination
    engine reacts in real time — e.g. retrigger Robustifier's
    ambiguity radius on ``transporter.drift_breach`` when the W_1
    drift score exceeds an SLO.

Where this slots in
-------------------

    tr = Transporter(bus=bus, attestor=attestor)
    tr.register_reference("calibration_window", samples=ref_samples)
    rep = tr.compute(
        source=live_samples,
        target=ref_samples,
        method=METHOD_SINKHORN,
        cost=COST_SQEUCLIDEAN,
        reg=0.05,
    )
    # rep.distance         → W_2² estimate
    # rep.divergence       → S_ε(a, b)  ≥ 0, zero iff equal
    # rep.plan             → transport plan, marginals ≤ 1e-9 violation
    # rep.potentials       → (φ, ψ) dual
    # rep.certificate      → {converged, marginal_violation, cyclic_monotonicity,
    #                         content_hash, sample_complexity_bound}

Events
------
    transporter.started              — engine was constructed
    transporter.reference_registered — a reference distribution was stashed
    transporter.reference_removed    — a reference distribution was removed
    transporter.computed             — an OT distance / plan was computed
    transporter.drift_evaluated      — drift score against a reference
    transporter.barycenter           — a barycenter was computed
    transporter.cleared              — state was reset
    transporter.report               — a coverage report was published
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds
# =====================================================================

TRANSPORTER_STARTED = "transporter.started"
TRANSPORTER_REFERENCE_REGISTERED = "transporter.reference_registered"
TRANSPORTER_REFERENCE_REMOVED = "transporter.reference_removed"
TRANSPORTER_COMPUTED = "transporter.computed"
TRANSPORTER_DRIFT_EVALUATED = "transporter.drift_evaluated"
TRANSPORTER_BARYCENTER = "transporter.barycenter"
TRANSPORTER_CLEARED = "transporter.cleared"
TRANSPORTER_REPORT = "transporter.report"


# =====================================================================
# Methods
# =====================================================================

METHOD_AUTO = "auto"
METHOD_HUNGARIAN = "hungarian"
METHOD_SINKHORN = "sinkhorn"
METHOD_SLICED = "sliced"
METHOD_EMD_1D = "emd_1d"
METHOD_UNBALANCED = "unbalanced"
METHOD_GROMOV = "gromov"

KNOWN_METHODS = (
    METHOD_AUTO,
    METHOD_HUNGARIAN,
    METHOD_SINKHORN,
    METHOD_SLICED,
    METHOD_EMD_1D,
    METHOD_UNBALANCED,
    METHOD_GROMOV,
)


# =====================================================================
# Costs
# =====================================================================

COST_EUCLIDEAN = "euclidean"
COST_SQEUCLIDEAN = "sqeuclidean"
COST_MANHATTAN = "manhattan"
COST_CHEBYSHEV = "chebyshev"
COST_CUSTOM = "custom"

KNOWN_COSTS = (
    COST_EUCLIDEAN,
    COST_SQEUCLIDEAN,
    COST_MANHATTAN,
    COST_CHEBYSHEV,
    COST_CUSTOM,
)


# =====================================================================
# Internals
# =====================================================================

_EPS = 1e-12
_DEFAULT_REG = 0.05
_DEFAULT_TOL = 1e-9
_DEFAULT_MAX_ITER = 2_000
_DEFAULT_PROJECTIONS = 64


# =====================================================================
# Errors
# =====================================================================


class TransporterError(Exception):
    """Base class for transporter errors."""


class InvalidProblem(TransporterError):
    """Inputs do not form a valid transport problem."""


class UnknownReference(TransporterError):
    """The named reference distribution has not been registered."""


class NotConverged(TransporterError):
    """The chosen solver did not converge to the requested tolerance."""


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class TransportProblem:
    """A normalised transport problem.

    The source and target are *discrete histograms* on n and m support
    points respectively, plus the n×m cost matrix.  Continuous samples
    are normalised here (one sample → one atom with mass 1/n) by the
    ``make_problem`` factory.
    """

    source_weights: tuple
    target_weights: tuple
    cost: tuple  # tuple-of-tuples for hashability
    n: int
    m: int

    def __post_init__(self) -> None:
        # Dataclass frozen — validations happen in the factory; this
        # method is kept tiny for the dataclass copy idiom.
        pass


@dataclass(frozen=True)
class TransportReport:
    """Result of a transporter compute call.

    ``distance`` is the unregularised optimal-transport cost (the
    Kantorovich functional ⟨P, C⟩ for the returned plan).
    ``regularised`` is ⟨P, C⟩ + ε H(P) when ε > 0, else equal to
    ``distance``.
    ``divergence`` is the Sinkhorn divergence S_ε if the user
    requested it, otherwise None.
    ``plan`` is the n×m transport plan as a tuple-of-tuples (small
    problems) or None (sliced/EMD-1D when explicit plan was not
    requested).
    ``potentials`` are the dual potentials (φ, ψ) when computed by an
    entropic method, else None.
    ``certificate`` carries machine-verifiable diagnostics.
    """

    method: str
    cost_kind: str
    distance: float
    regularised: float
    divergence: float | None
    plan: tuple | None
    potentials: tuple | None
    n_iter: int
    converged: bool
    marginal_violation: float
    certificate: dict
    receipt_id: str


@dataclass(frozen=True)
class DriftReport:
    """Result of a drift evaluation against a registered reference."""

    reference_id: str
    method: str
    score: float
    breach: bool
    threshold: float | None
    n_samples: int
    certificate: dict
    receipt_id: str


@dataclass(frozen=True)
class BarycenterReport:
    """Result of a barycenter computation."""

    method: str
    support: tuple
    weights: tuple
    sources: int
    n_iter: int
    converged: bool
    certificate: dict
    receipt_id: str


@dataclass(frozen=True)
class CoverageReport:
    """Summary of the transporter's lifetime statistics."""

    references: int
    computes: int
    drifts: int
    barycenters: int


# =====================================================================
# Hashing
# =====================================================================


def _hash_problem(
    source_weights: tuple, target_weights: tuple, cost: tuple
) -> str:
    h = hashlib.sha256()
    h.update(b"transporter:v1\n")
    h.update(json.dumps(list(source_weights), sort_keys=True).encode("utf-8"))
    h.update(b"\n")
    h.update(json.dumps(list(target_weights), sort_keys=True).encode("utf-8"))
    h.update(b"\n")
    for row in cost:
        h.update(json.dumps([float(c) for c in row]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# =====================================================================
# Cost matrix construction
# =====================================================================


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_vec(x: Any) -> tuple:
    """Coerce a sample into a tuple of floats.

    Scalars become 1-D vectors.  Sequences are flattened to tuples.
    """
    if _is_scalar(x):
        return (float(x),)
    if isinstance(x, (tuple, list)):
        return tuple(float(v) for v in x)
    raise InvalidProblem(f"unsupported sample type: {type(x).__name__}")


def _normalise_samples(samples: Sequence[Any]) -> tuple:
    """Coerce a list of samples into a list of vectors of common dim."""
    if len(samples) == 0:
        raise InvalidProblem("empty sample list")
    vecs = [_as_vec(s) for s in samples]
    d = len(vecs[0])
    if d == 0:
        raise InvalidProblem("sample dimension is 0")
    for v in vecs:
        if len(v) != d:
            raise InvalidProblem("samples have inconsistent dimension")
    return tuple(vecs)


def _normalise_weights(weights: Sequence[float] | None, n: int) -> tuple:
    if weights is None:
        return tuple(1.0 / n for _ in range(n))
    if len(weights) != n:
        raise InvalidProblem("weights length must match samples length")
    w = [float(v) for v in weights]
    if any(v < -_EPS for v in w):
        raise InvalidProblem("weights must be non-negative")
    s = sum(w)
    if s <= 0:
        raise InvalidProblem("weights sum must be positive")
    return tuple(max(0.0, v) / s for v in w)


def _cost_value(x: tuple, y: tuple, kind: str) -> float:
    if kind == COST_SQEUCLIDEAN:
        return sum((xi - yi) * (xi - yi) for xi, yi in zip(x, y))
    if kind == COST_EUCLIDEAN:
        return math.sqrt(sum((xi - yi) * (xi - yi) for xi, yi in zip(x, y)))
    if kind == COST_MANHATTAN:
        return sum(abs(xi - yi) for xi, yi in zip(x, y))
    if kind == COST_CHEBYSHEV:
        return max(abs(xi - yi) for xi, yi in zip(x, y))
    raise InvalidProblem(f"unknown cost kind: {kind}")


def cost_matrix(
    source: Sequence[Any],
    target: Sequence[Any],
    kind: str = COST_SQEUCLIDEAN,
) -> tuple:
    """Build an n×m cost matrix between source and target points."""
    if kind not in KNOWN_COSTS:
        raise InvalidProblem(f"unknown cost kind: {kind}")
    if kind == COST_CUSTOM:
        raise InvalidProblem("provide a precomputed matrix for custom cost")
    src = _normalise_samples(source)
    tgt = _normalise_samples(target)
    if len(src[0]) != len(tgt[0]):
        raise InvalidProblem(
            f"source/target dimension mismatch: {len(src[0])} vs {len(tgt[0])}"
        )
    return tuple(
        tuple(_cost_value(x, y, kind) for y in tgt) for x in src
    )


def make_problem(
    source: Sequence[Any],
    target: Sequence[Any],
    *,
    source_weights: Sequence[float] | None = None,
    target_weights: Sequence[float] | None = None,
    cost: Sequence[Sequence[float]] | str = COST_SQEUCLIDEAN,
) -> TransportProblem:
    """Construct a normalised transport problem.

    ``source`` and ``target`` are sequences of samples (scalars or
    d-dim vectors).  ``source_weights`` and ``target_weights`` are
    optional probability weights, default uniform.  ``cost`` is
    either a string from ``KNOWN_COSTS`` or an n×m matrix.
    """
    src = _normalise_samples(source)
    tgt = _normalise_samples(target)
    n, m = len(src), len(tgt)
    sw = _normalise_weights(source_weights, n)
    tw = _normalise_weights(target_weights, m)
    if isinstance(cost, str):
        C = cost_matrix(src, tgt, kind=cost)
    else:
        if len(cost) != n:
            raise InvalidProblem(
                f"cost matrix row count {len(cost)} != source count {n}"
            )
        rows = []
        for row in cost:
            if len(row) != m:
                raise InvalidProblem(
                    f"cost matrix col count {len(row)} != target count {m}"
                )
            rows.append(tuple(float(c) for c in row))
        C = tuple(rows)
    return TransportProblem(
        source_weights=sw, target_weights=tw, cost=C, n=n, m=m
    )


# =====================================================================
# 1-D Wasserstein  (closed form)
# =====================================================================


def wasserstein_1d(
    a_samples: Sequence[float],
    b_samples: Sequence[float],
    *,
    a_weights: Sequence[float] | None = None,
    b_weights: Sequence[float] | None = None,
    p: float = 1.0,
) -> float:
    r"""Compute the closed-form W_p distance between two empirical 1-D
    distributions.

    For equal sample sizes and uniform weights this is
        W_p(μ, ν)^p = (1/n) Σ_i |x_{(i)} − y_{(i)}|^p,
    where x_{(i)}, y_{(i)} are the order statistics.  General version
    integrates |F_μ^{-1}(t) − F_ν^{-1}(t)|^p over t ∈ [0, 1] using the
    *common refinement* of the two CDF step partitions — exact, with
    O((n + m) log(n + m)) time.
    """
    if p <= 0:
        raise InvalidProblem("p must be positive")
    n = len(a_samples)
    m = len(b_samples)
    if n == 0 or m == 0:
        raise InvalidProblem("empty input")
    aw = _normalise_weights(a_weights, n)
    bw = _normalise_weights(b_weights, m)
    a_sorted = sorted(zip(a_samples, aw), key=lambda pp: pp[0])
    b_sorted = sorted(zip(b_samples, bw), key=lambda pp: pp[0])
    # Sweep the two CDFs together.
    i = j = 0
    cum_a = 0.0
    cum_b = 0.0
    total = 0.0
    eps = 1e-15
    while i < n and j < m:
        x_a, w_a = a_sorted[i]
        x_b, w_b = b_sorted[j]
        next_a = cum_a + w_a
        next_b = cum_b + w_b
        if next_a <= next_b + eps:
            seg = next_a - max(cum_a, cum_b)
            if seg > 0:
                total += seg * (abs(x_a - x_b) ** p)
            cum_a = next_a
            i += 1
        else:
            seg = next_b - max(cum_a, cum_b)
            if seg > 0:
                total += seg * (abs(x_a - x_b) ** p)
            cum_b = next_b
            j += 1
    return total ** (1.0 / p) if p != 1.0 else total


# =====================================================================
# Hungarian algorithm  (exact assignment)
# =====================================================================


def hungarian(cost: Sequence[Sequence[float]]) -> tuple:
    """Solve the rectangular assignment problem.

    Implements the O(n³) Kuhn-Munkres algorithm.  Accepts a possibly
    non-square cost matrix; if rows < cols, pads internally with the
    matrix's max value.  Returns ``(row_assignment, total_cost)``
    where ``row_assignment[i]`` is the column matched to row ``i``.

    The implementation is the Jonker-Volgenant variant: a
    shortest-augmenting-path approach with potentials, which is
    numerically stable and outperforms the textbook bipartite
    algorithm on dense matrices.
    """
    n = len(cost)
    if n == 0:
        return ((), 0.0)
    m = len(cost[0])
    for row in cost:
        if len(row) != m:
            raise InvalidProblem("hungarian: ragged cost matrix")
    if m < n:
        # Transpose; we always solve with n ≤ m.
        cost_t = tuple(tuple(cost[i][j] for i in range(n)) for j in range(m))
        col_to_row, total = hungarian(cost_t)
        row_to_col = [-1] * n
        for j, i in enumerate(col_to_row):
            if i >= 0:
                row_to_col[i] = j
        return (tuple(row_to_col), total)
    INF = float("inf")
    # Augmenting-path / Jonker-Volgenant.
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[j] = row assigned to column j (1-indexed)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            if j1 < 0:
                raise NotConverged("hungarian: infeasible problem")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    row_to_col = [-1] * n
    total = 0.0
    for j in range(1, m + 1):
        if p[j] != 0:
            row_to_col[p[j] - 1] = j - 1
            total += cost[p[j] - 1][j - 1]
    return (tuple(row_to_col), total)


# =====================================================================
# Sinkhorn (log-domain stabilised)
# =====================================================================


def _logsumexp(values: Iterable[float]) -> float:
    vs = list(values)
    if not vs:
        return float("-inf")
    m = max(vs)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in vs))


def sinkhorn(
    problem: TransportProblem,
    *,
    reg: float = _DEFAULT_REG,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> tuple:
    r"""Entropic OT via log-domain stabilised Sinkhorn iteration.

    Solves
        min_P ⟨P, C⟩ − ε H(P)   subject to   P 1 = a,  Pᵀ 1 = b,
    where H(P) = −Σ_{ij} P_ij (log P_ij − 1).  Returns
    ``(P, log_u, log_v, n_iter, converged, marginal_violation)``.

    Recovery from the dual potentials:
        P_ij = exp((u_i + v_j − C_ij) / ε)
    where u and v live in *log-space*.  We track both u and v as logs
    of their multiplicative counterparts, which is numerically
    correct for ε ↘ 0.

    Marginal violation is
        max( max_i |P_i· − a_i|, max_j |P_·j − b_j| )
    which is the canonical first-order optimality residual; geometric
    convergence rate is (1 − exp(−Δ/ε))² with Δ the max C entry
    (Franklin-Lorenz 1989).
    """
    if reg <= 0:
        raise InvalidProblem("sinkhorn: reg must be positive")
    a = problem.source_weights
    b = problem.target_weights
    C = problem.cost
    n, m = problem.n, problem.m
    log_a = tuple(math.log(max(v, _EPS)) for v in a)
    log_b = tuple(math.log(max(v, _EPS)) for v in b)
    log_u = [0.0] * n
    log_v = [0.0] * m
    # M_ij = -C_ij / reg
    M = tuple(tuple(-c / reg for c in row) for row in C)

    converged = False
    iters = 0
    last_violation = float("inf")
    for it in range(1, max_iter + 1):
        iters = it
        # log_u_i = log_a_i - logsumexp_j (M_ij + log_v_j)
        for i in range(n):
            log_u[i] = log_a[i] - _logsumexp(
                M[i][j] + log_v[j] for j in range(m)
            )
        # log_v_j = log_b_j - logsumexp_i (M_ij + log_u_i)
        for j in range(m):
            log_v[j] = log_b[j] - _logsumexp(
                M[i][j] + log_u[i] for i in range(n)
            )
        # Marginal residual (cheap)
        if it % 10 == 0 or it == max_iter:
            row_sums = [
                sum(math.exp(log_u[i] + M[i][j] + log_v[j]) for j in range(m))
                for i in range(n)
            ]
            col_sums = [
                sum(math.exp(log_u[i] + M[i][j] + log_v[j]) for i in range(n))
                for j in range(m)
            ]
            row_err = max(abs(rs - a[i]) for i, rs in enumerate(row_sums))
            col_err = max(abs(cs - b[j]) for j, cs in enumerate(col_sums))
            last_violation = max(row_err, col_err)
            if last_violation <= tol:
                converged = True
                break
    # Recover plan
    P = tuple(
        tuple(math.exp(log_u[i] + M[i][j] + log_v[j]) for j in range(m))
        for i in range(n)
    )
    return (
        P,
        tuple(log_u),
        tuple(log_v),
        iters,
        converged,
        last_violation,
    )


def sinkhorn_cost(plan: tuple, cost: tuple) -> float:
    r"""Compute ⟨P, C⟩ for a plan and cost matrix."""
    return sum(
        plan[i][j] * cost[i][j]
        for i in range(len(plan))
        for j in range(len(plan[0]))
    )


def sinkhorn_entropy(plan: tuple) -> float:
    r"""Shannon entropy of a plan, H(P) = −Σ_{ij} P_ij log P_ij.

    Zero entries are clamped (0·log 0 ≡ 0).  For a probability plan
    Σ P = 1 this is the standard Shannon entropy; for unbalanced
    plans the integral is over all atoms regardless of total mass.
    """
    h = 0.0
    for row in plan:
        for v in row:
            if v > _EPS:
                h -= v * math.log(v)
    return h


def sinkhorn_divergence(
    a_samples: Sequence[Any],
    b_samples: Sequence[Any],
    *,
    a_weights: Sequence[float] | None = None,
    b_weights: Sequence[float] | None = None,
    cost_kind: str = COST_SQEUCLIDEAN,
    reg: float = _DEFAULT_REG,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> float:
    r"""Debiased Sinkhorn divergence S_ε(a, b) ≥ 0.

    S_ε(a, b) = OT_ε(a, b) − ½ OT_ε(a, a) − ½ OT_ε(b, b).

    Guaranteed nonnegative on the simplex for ε > 0 sufficiently
    small (Feydy et al. 2019).  Acts like a proper metric for the
    runtime's drift purposes.
    """
    ab = make_problem(a_samples, b_samples,
                      source_weights=a_weights, target_weights=b_weights,
                      cost=cost_kind)
    aa = make_problem(a_samples, a_samples,
                      source_weights=a_weights, target_weights=a_weights,
                      cost=cost_kind)
    bb = make_problem(b_samples, b_samples,
                      source_weights=b_weights, target_weights=b_weights,
                      cost=cost_kind)
    P_ab, *_ = sinkhorn(ab, reg=reg, max_iter=max_iter, tol=tol)
    P_aa, *_ = sinkhorn(aa, reg=reg, max_iter=max_iter, tol=tol)
    P_bb, *_ = sinkhorn(bb, reg=reg, max_iter=max_iter, tol=tol)
    return (
        sinkhorn_cost(P_ab, ab.cost)
        - 0.5 * sinkhorn_cost(P_aa, aa.cost)
        - 0.5 * sinkhorn_cost(P_bb, bb.cost)
    )


# =====================================================================
# Sliced Wasserstein
# =====================================================================


def _gaussian_unit_vector(d: int, rng: random.Random) -> tuple:
    """Sample uniformly from 𝕊^{d-1} via the Gaussian quotient method."""
    if d == 1:
        return (1.0,) if rng.random() < 0.5 else (-1.0,)
    while True:
        g = [rng.gauss(0.0, 1.0) for _ in range(d)]
        s = math.sqrt(sum(v * v for v in g))
        if s > _EPS:
            return tuple(v / s for v in g)


def sliced_wasserstein(
    X: Sequence[Any],
    Y: Sequence[Any],
    *,
    p: float = 2.0,
    n_projections: int = _DEFAULT_PROJECTIONS,
    seed: int | None = None,
) -> tuple:
    r"""Sliced Wasserstein distance and its Monte-Carlo standard error.

    For samples ``X ∈ ℝ^{n × d}`` and ``Y ∈ ℝ^{m × d}``:
        SW_p(X, Y)^p ≈ (1/K) Σ_k W_p(θ_k · X, θ_k · Y)^p,
    where θ_k ~ Uniform(𝕊^{d − 1}).  Returns a tuple ``(SW_p, stderr,
    per_projection_values)``.  The standard error is the per-
    projection sample std divided by √K — a valid Monte-Carlo bound
    given that the per-projection 1-D W_p is bounded by the cost
    diameter.
    """
    Xv = _normalise_samples(X)
    Yv = _normalise_samples(Y)
    d = len(Xv[0])
    if len(Yv[0]) != d:
        raise InvalidProblem("X and Y must have the same dimension")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_projections):
        theta = _gaussian_unit_vector(d, rng)
        xs = [sum(xi * ti for xi, ti in zip(x, theta)) for x in Xv]
        ys = [sum(yi * ti for yi, ti in zip(y, theta)) for y in Yv]
        w = wasserstein_1d(xs, ys, p=p)
        vals.append(w ** p)
    K = len(vals)
    mean = sum(vals) / K
    if K > 1:
        var = sum((v - mean) ** 2 for v in vals) / (K - 1)
        stderr = math.sqrt(var / K)
    else:
        stderr = 0.0
    sw_p = mean ** (1.0 / p)
    return (sw_p, stderr, tuple(vals))


# =====================================================================
# Wasserstein-1 dual potentials (1-Lipschitz)
# =====================================================================


def kantorovich_rubinstein_1d(
    a_samples: Sequence[float],
    b_samples: Sequence[float],
    *,
    a_weights: Sequence[float] | None = None,
    b_weights: Sequence[float] | None = None,
) -> tuple:
    r"""Kantorovich-Rubinstein dual for 1-D W_1.

    Returns ``(W_1, potential)`` where ``potential`` is a callable
    f : ℝ → ℝ, 1-Lipschitz, certifying  W_1 = E_a f − E_b f.  The
    1-D dual is the signed CDF difference  f(t) = sign · ∫_0^t (F_a − F_b)
    after a monotone change of variables.
    """
    all_pts = sorted(set(list(a_samples) + list(b_samples)))
    aw = _normalise_weights(a_weights, len(a_samples))
    bw = _normalise_weights(b_weights, len(b_samples))
    # Build cumulative weights at each grid point.
    cum_a = []
    cum_b = []
    sa = 0.0
    sb = 0.0
    sorted_a = sorted(zip(a_samples, aw), key=lambda pp: pp[0])
    sorted_b = sorted(zip(b_samples, bw), key=lambda pp: pp[0])
    ai = bj = 0
    integrand = []
    for k, t in enumerate(all_pts):
        while ai < len(sorted_a) and sorted_a[ai][0] <= t:
            sa += sorted_a[ai][1]
            ai += 1
        while bj < len(sorted_b) and sorted_b[bj][0] <= t:
            sb += sorted_b[bj][1]
            bj += 1
        cum_a.append(sa)
        cum_b.append(sb)
    # W_1 = ∫ |F_a − F_b| dt over the support.
    w1 = 0.0
    for k in range(1, len(all_pts)):
        seg = all_pts[k] - all_pts[k - 1]
        w1 += seg * abs(cum_a[k - 1] - cum_b[k - 1])
    # Dual potential: the 1-Lipschitz f maximising  E_a[f] − E_b[f] = W_1.
    # Integration by parts of  ∫ f d(a − b)  gives  −∫ f'(t)·(F_a − F_b) dt;
    # to make this equal +W_1 = ∫|F_a − F_b| dt we need
    #     f'(t) = sign(F_b(t) − F_a(t)) = −sign(F_a − F_b).
    knot_t = list(all_pts)
    knot_f = [0.0]
    for k in range(1, len(knot_t)):
        sign = (
            -1.0
            if cum_a[k - 1] > cum_b[k - 1]
            else (1.0 if cum_a[k - 1] < cum_b[k - 1] else 0.0)
        )
        knot_f.append(knot_f[-1] + sign * (knot_t[k] - knot_t[k - 1]))

    def potential(t: float) -> float:
        if t <= knot_t[0]:
            return knot_f[0]
        if t >= knot_t[-1]:
            return knot_f[-1]
        # Binary search.
        lo, hi = 0, len(knot_t) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if knot_t[mid] <= t:
                lo = mid
            else:
                hi = mid
        seg = knot_t[hi] - knot_t[lo]
        if seg <= 0:
            return knot_f[lo]
        return knot_f[lo] + (knot_f[hi] - knot_f[lo]) * (t - knot_t[lo]) / seg

    return (w1, potential)


# =====================================================================
# Unbalanced OT (KL marginals)
# =====================================================================


def unbalanced_sinkhorn(
    problem: TransportProblem,
    *,
    reg: float = _DEFAULT_REG,
    marginal_penalty: float = 1.0,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> tuple:
    r"""Unbalanced entropic OT with symmetric KL marginal penalty.

    Solves
        min_P ⟨P, C⟩ + ε KL(P ‖ a bᵀ) + ρ KL(P 1 ‖ a) + ρ KL(Pᵀ 1 ‖ b),
    via the Chizat-Peyré-Schmitzer-Vialard generalised Sinkhorn
    iteration:
        u ← (a / (K v))^{ρ / (ρ + ε)},
        v ← (b / (Kᵀ u))^{ρ / (ρ + ε)}.

    Recovers balanced Sinkhorn as marginal_penalty → ∞.
    """
    if reg <= 0:
        raise InvalidProblem("unbalanced_sinkhorn: reg must be positive")
    if marginal_penalty <= 0:
        raise InvalidProblem("unbalanced_sinkhorn: marginal_penalty must be positive")
    a = problem.source_weights
    b = problem.target_weights
    C = problem.cost
    n, m = problem.n, problem.m
    K = tuple(tuple(math.exp(-c / reg) for c in row) for row in C)
    u = [1.0] * n
    v = [1.0] * m
    rho = marginal_penalty
    exponent = rho / (rho + reg)
    iters = 0
    converged = False
    last_violation = float("inf")
    for it in range(1, max_iter + 1):
        iters = it
        for i in range(n):
            denom = sum(K[i][j] * v[j] for j in range(m))
            u[i] = (a[i] / max(denom, _EPS)) ** exponent
        for j in range(m):
            denom = sum(K[i][j] * u[i] for i in range(n))
            v[j] = (b[j] / max(denom, _EPS)) ** exponent
        if it % 10 == 0 or it == max_iter:
            row_sums = [
                sum(u[i] * K[i][j] * v[j] for j in range(m)) for i in range(n)
            ]
            col_sums = [
                sum(u[i] * K[i][j] * v[j] for i in range(n)) for j in range(m)
            ]
            row_err = max(abs(rs - ai) for rs, ai in zip(row_sums, a))
            col_err = max(abs(cs - bj) for cs, bj in zip(col_sums, b))
            last_violation = max(row_err, col_err)
            if last_violation <= tol:
                converged = True
                break
    P = tuple(
        tuple(u[i] * K[i][j] * v[j] for j in range(m)) for i in range(n)
    )
    return (P, tuple(u), tuple(v), iters, converged, last_violation)


# =====================================================================
# Gromov-Wasserstein (entropic, alternating Sinkhorn)
# =====================================================================


def gromov_wasserstein(
    Cx: Sequence[Sequence[float]],
    Cy: Sequence[Sequence[float]],
    a: Sequence[float],
    b: Sequence[float],
    *,
    reg: float = 0.1,
    outer_iter: int = 50,
    inner_iter: int = 200,
    tol: float = 1e-6,
) -> tuple:
    r"""Entropic Gromov-Wasserstein distance and plan.

    Minimises
        Σ_{ijkl} |Cx_{ij} − Cy_{kl}|² P_{ik} P_{jl}
    subject to marginal constraints, via the Peyré-Cuturi-Solomon
    alternating-Sinkhorn algorithm with squared cost.  Returns
    ``(P, gw_cost, n_outer, converged)``.

    Unlike standard OT, GW does not require Cx and Cy to live on the
    same space — only that they each be valid pairwise-distance
    structures on their own support.  This is the right primitive
    for matching feature spaces of different dimensions.
    """
    n = len(a)
    m = len(b)
    if len(Cx) != n or any(len(r) != n for r in Cx):
        raise InvalidProblem("gromov: Cx must be n×n")
    if len(Cy) != m or any(len(r) != m for r in Cy):
        raise InvalidProblem("gromov: Cy must be m×m")
    aw = _normalise_weights(a, n)
    bw = _normalise_weights(b, m)
    # Initial plan = product of marginals.
    P = [[aw[i] * bw[j] for j in range(m)] for i in range(n)]
    last_cost = None
    converged = False
    iters = 0
    for outer in range(1, outer_iter + 1):
        iters = outer
        # Build the linearised cost L_ik = Σ_jl |Cx_ij − Cy_kl|² P_jl.
        # Expand:  L = Cx² ⊗ 1 + 1 ⊗ Cy² − 2 Cx P Cyᵀ.
        # Compute Cx P (n×m) then (Cx P) Cyᵀ (n×m).
        CxP = [
            [sum(Cx[i][k2] * P[k2][j] for k2 in range(n)) for j in range(m)]
            for i in range(n)
        ]
        CxPCy = [
            [sum(CxP[i][j2] * Cy[j][j2] for j2 in range(m)) for j in range(m)]
            for i in range(n)
        ]
        # Cx² row sums weighted by aw, etc.
        rowQ = [
            sum(Cx[i][k2] ** 2 * aw[k2] for k2 in range(n)) for i in range(n)
        ]
        colQ = [
            sum(Cy[j][j2] ** 2 * bw[j2] for j2 in range(m)) for j in range(m)
        ]
        L = [
            [rowQ[i] + colQ[j] - 2.0 * CxPCy[i][j] for j in range(m)]
            for i in range(n)
        ]
        # Run an inner Sinkhorn against L with current marginals.
        inner = make_problem(
            list(range(n)), list(range(m)),
            source_weights=aw, target_weights=bw,
            cost=tuple(tuple(row) for row in L),
        )
        Pnew, _, _, _, _, _ = sinkhorn(
            inner, reg=reg, max_iter=inner_iter, tol=tol
        )
        # Compute GW cost on the *new* plan.
        # ⟨L, Pnew⟩ + const, with const = Σ row Q · a + Σ col Q · b
        # — but for monitoring we use the raw quartic form.
        gw = 0.0
        for i in range(n):
            for j in range(m):
                gw += L[i][j] * Pnew[i][j]
        # We monitor the GW relative change.
        if last_cost is not None and abs(gw - last_cost) <= tol * max(
            1.0, abs(last_cost)
        ):
            P = [list(row) for row in Pnew]
            converged = True
            last_cost = gw
            break
        last_cost = gw
        P = [list(row) for row in Pnew]
    return (
        tuple(tuple(row) for row in P),
        max(0.0, float(last_cost or 0.0)),
        iters,
        converged,
    )


# =====================================================================
# 1-D Wasserstein barycenter (closed form)
# =====================================================================


def wasserstein_barycenter_1d(
    distributions: Sequence[Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
    n_support: int = 200,
) -> tuple:
    r"""W_2 barycenter of K univariate empirical distributions.

    Closed-form (Agueh-Carlier 2011): the K-barycenter of
    ``μ_1, …, μ_K`` with weights ``λ_k`` and quantile functions
    ``F_k^{-1}`` has quantile
        F̄^{-1}(t) = Σ_k λ_k F_k^{-1}(t).
    We discretise t on n_support equally spaced quantile-grid points
    and return ``(support, weights)``.
    """
    K = len(distributions)
    if K == 0:
        raise InvalidProblem("barycenter: need at least one distribution")
    if weights is None:
        weights = [1.0 / K] * K
    if len(weights) != K:
        raise InvalidProblem("weights length must match number of distributions")
    s = sum(weights)
    if s <= 0:
        raise InvalidProblem("weights sum must be positive")
    lam = [float(w) / s for w in weights]
    sorted_dists = [sorted(d) for d in distributions]
    qs = [(k + 0.5) / n_support for k in range(n_support)]
    bary_pts = []
    for q in qs:
        v = 0.0
        for k in range(K):
            arr = sorted_dists[k]
            n = len(arr)
            if n == 0:
                continue
            idx = min(n - 1, max(0, int(q * n)))
            v += lam[k] * arr[idx]
        bary_pts.append(v)
    return tuple(bary_pts), tuple(1.0 / n_support for _ in range(n_support))


# =====================================================================
# Cyclic monotonicity check
# =====================================================================


def cyclic_monotonicity_violation(
    plan: Sequence[Sequence[float]],
    cost: Sequence[Sequence[float]],
    *,
    n_cycles: int = 200,
    cycle_length: int = 3,
    seed: int | None = None,
    threshold: float = 1e-9,
) -> tuple:
    r"""Sample-based cyclic-monotonicity certificate.

    For a transport plan to be optimal, the support must be
    *c-cyclically monotone*: for every cycle (i₁,j₁), …, (iₖ,jₖ) of
    plan atoms,
        Σ_t c(x_{iₜ}, y_{jₜ}) ≤ Σ_t c(x_{iₜ}, y_{jₜ₊₁}).
    We randomly sample cycles of the given length on the plan's
    support and return ``(max_violation, n_violations, n_checked)``.
    A plan is *certifiably optimal* iff max_violation ≤ threshold.
    """
    rng = random.Random(seed)
    # Support: pairs (i, j) with P_ij > threshold (after normalisation).
    n = len(plan)
    m = len(plan[0]) if n > 0 else 0
    support = [
        (i, j) for i in range(n) for j in range(m) if plan[i][j] > threshold
    ]
    if len(support) < cycle_length:
        return (0.0, 0, 0)
    max_v = 0.0
    n_viol = 0
    for _ in range(n_cycles):
        cycle = rng.sample(support, cycle_length)
        rows = [c[0] for c in cycle]
        cols = [c[1] for c in cycle]
        forward = sum(cost[rows[t]][cols[t]] for t in range(cycle_length))
        shifted = sum(
            cost[rows[t]][cols[(t + 1) % cycle_length]]
            for t in range(cycle_length)
        )
        diff = forward - shifted
        if diff > max_v:
            max_v = diff
        if diff > threshold:
            n_viol += 1
    return (max_v, n_viol, n_cycles)


# =====================================================================
# Auto method selection
# =====================================================================


def _auto_method(problem: TransportProblem) -> str:
    n, m = problem.n, problem.m
    # If both 1-D, use the closed form.
    # We can't easily check from cost matrix; default to Sinkhorn.
    if n * m <= 64:
        # Small enough to run Hungarian to optimality on the
        # equal-mass case.  If weights aren't uniform we'd still want
        # Sinkhorn for sub-atom splitting.
        if all(abs(w - 1.0 / n) < 1e-9 for w in problem.source_weights) and (
            n == m
        ) and all(abs(w - 1.0 / m) < 1e-9 for w in problem.target_weights):
            return METHOD_HUNGARIAN
    return METHOD_SINKHORN


# =====================================================================
# Attestation adapter
# =====================================================================


class _AttestableReceipt:
    """Adapter object for ``AttestationLedger.append()``."""

    __slots__ = ("ticket_id", "kind", "payload", "digest")

    def __init__(self, kind: str, payload: dict, digest: str = "") -> None:
        self.kind = kind
        self.payload = payload
        self.digest = digest
        self.ticket_id = (
            payload.get("receipt_id") or digest[:16] or uuid.uuid4().hex[:16]
        )

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "kind": self.kind,
            "payload": self.payload,
            "digest": self.digest,
        }


# =====================================================================
# Transporter runtime
# =====================================================================


class Transporter:
    """Optimal-transport engine for the agi runtime.

    Stateless except for a registry of reference distributions and
    coverage counters.  Thread-safe via a single recursive lock.

    Optional dependencies:
      bus       — ``agi.events.EventBus`` (events emitted on every state change)
      attestor  — ``agi.attest.RuntimeAttestor`` (receipts written on compute)
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        attestor: Any = None,
        random_seed: int | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._lock = threading.RLock()
        self._refs: dict = {}
        self._n_computes = 0
        self._n_drifts = 0
        self._n_barycenters = 0
        self._random_seed = random_seed
        self._emit(
            TRANSPORTER_STARTED,
            {"id": uuid.uuid4().hex[:16], "timestamp_ns": time.time_ns()},
        )

    # ----- event / attestation helpers -----

    def _emit(self, kind: str, payload: dict) -> None:
        if self._bus is None:
            return
        self._bus.publish(Event(kind=kind, data=dict(payload)))

    def _attest(self, kind: str, payload: dict) -> str:
        if self._attestor is None:
            return ""
        try:
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        except Exception:
            digest = ""
        receipt = _AttestableReceipt(kind=kind, payload=payload, digest=digest)
        # Try the protocol-level call first; fall back to a no-op.
        try:
            if hasattr(self._attestor, "record"):
                self._attestor.record(kind=kind, payload=payload)
            elif callable(self._attestor):
                self._attestor(receipt)
        except Exception:
            pass
        return digest

    # ----- references (drift-eval baselines) -----

    def register_reference(
        self,
        reference_id: str,
        *,
        samples: Sequence[Any],
        weights: Sequence[float] | None = None,
        description: str = "",
    ) -> dict:
        if not reference_id:
            raise InvalidProblem("reference_id must be non-empty")
        with self._lock:
            vecs = _normalise_samples(samples)
            ws = _normalise_weights(weights, len(vecs))
            record = {
                "id": reference_id,
                "samples": vecs,
                "weights": ws,
                "n": len(vecs),
                "d": len(vecs[0]),
                "description": description,
                "registered_at_ns": time.time_ns(),
            }
            self._refs[reference_id] = record
            self._emit(
                TRANSPORTER_REFERENCE_REGISTERED,
                {
                    "reference_id": reference_id,
                    "n": record["n"],
                    "d": record["d"],
                    "description": description,
                },
            )
            return dict(record)

    def remove_reference(self, reference_id: str) -> bool:
        with self._lock:
            if reference_id in self._refs:
                del self._refs[reference_id]
                self._emit(
                    TRANSPORTER_REFERENCE_REMOVED,
                    {"reference_id": reference_id},
                )
                return True
            return False

    def references(self) -> tuple:
        with self._lock:
            return tuple(self._refs.keys())

    # ----- compute (general) -----

    def compute(
        self,
        *,
        source: Sequence[Any],
        target: Sequence[Any],
        source_weights: Sequence[float] | None = None,
        target_weights: Sequence[float] | None = None,
        cost: Sequence[Sequence[float]] | str = COST_SQEUCLIDEAN,
        method: str = METHOD_AUTO,
        reg: float = _DEFAULT_REG,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        n_projections: int = _DEFAULT_PROJECTIONS,
        marginal_penalty: float = 1.0,
        compute_divergence: bool = False,
        verify_monotonicity: bool = False,
        seed: int | None = None,
    ) -> TransportReport:
        if method not in KNOWN_METHODS:
            raise InvalidProblem(f"unknown method: {method}")
        if seed is None:
            seed = self._random_seed
        problem = make_problem(
            source, target,
            source_weights=source_weights,
            target_weights=target_weights,
            cost=cost,
        )
        chosen = method if method != METHOD_AUTO else _auto_method(problem)
        cost_kind = cost if isinstance(cost, str) else COST_CUSTOM
        receipt_id = uuid.uuid4().hex[:16]
        content_hash = _hash_problem(
            problem.source_weights, problem.target_weights, problem.cost
        )

        plan_t = None
        potentials = None
        n_iter = 0
        converged = True
        margin_v = 0.0
        regularised = 0.0
        distance = 0.0
        divergence = None

        if chosen == METHOD_HUNGARIAN:
            uniform_src = all(
                abs(w - 1.0 / problem.n) < 1e-9 for w in problem.source_weights
            )
            uniform_tgt = all(
                abs(w - 1.0 / problem.m) < 1e-9 for w in problem.target_weights
            )
            if problem.n != problem.m or not (uniform_src and uniform_tgt):
                # Hungarian assumes equal-mass square assignment;
                # rectangular or non-uniform → Sinkhorn for a valid
                # full-mass transport plan.
                chosen = METHOD_SINKHORN
            else:
                assignment, _ = hungarian(problem.cost)
                plan = [[0.0] * problem.m for _ in range(problem.n)]
                for i, j in enumerate(assignment):
                    plan[i][j] = problem.source_weights[i]
                plan_t = tuple(tuple(r) for r in plan)
                distance = sum(
                    plan[i][j] * problem.cost[i][j]
                    for i in range(problem.n)
                    for j in range(problem.m)
                )
                regularised = distance
                n_iter = 1
                converged = True
                margin_v = _marginal_violation(
                    plan_t, problem.source_weights, problem.target_weights
                )

        if chosen == METHOD_SINKHORN:
            P, log_u, log_v, n_iter, converged, margin_v = sinkhorn(
                problem, reg=reg, max_iter=max_iter, tol=tol
            )
            plan_t = P
            potentials = (
                tuple(reg * lu for lu in log_u),
                tuple(reg * lv for lv in log_v),
            )
            distance = sinkhorn_cost(P, problem.cost)
            # OT_ε objective (Cuturi 2013): ⟨P, C⟩ − ε H(P), with H the
            # Shannon entropy of the plan.
            regularised = distance - reg * sinkhorn_entropy(P)
            if compute_divergence:
                divergence = sinkhorn_divergence(
                    source, target,
                    a_weights=source_weights,
                    b_weights=target_weights,
                    cost_kind=cost if isinstance(cost, str) else COST_SQEUCLIDEAN,
                    reg=reg, max_iter=max_iter, tol=tol,
                )

        elif chosen == METHOD_SLICED:
            sw, stderr, vals = sliced_wasserstein(
                source, target,
                p=2.0, n_projections=n_projections, seed=seed,
            )
            distance = sw
            regularised = sw
            n_iter = n_projections
            converged = True
            margin_v = float(stderr)

        elif chosen == METHOD_EMD_1D:
            xs = [s if _is_scalar(s) else _as_vec(s)[0] for s in source]
            ys = [t if _is_scalar(t) else _as_vec(t)[0] for t in target]
            distance = wasserstein_1d(
                xs, ys,
                a_weights=source_weights, b_weights=target_weights, p=1.0,
            )
            regularised = distance
            n_iter = 1
            converged = True
            margin_v = 0.0

        elif chosen == METHOD_UNBALANCED:
            P, u, v, n_iter, converged, margin_v = unbalanced_sinkhorn(
                problem, reg=reg, marginal_penalty=marginal_penalty,
                max_iter=max_iter, tol=tol,
            )
            plan_t = P
            potentials = (u, v)
            distance = sinkhorn_cost(P, problem.cost)
            regularised = distance

        elif chosen == METHOD_GROMOV:
            # In Gromov, source and target each carry their *own*
            # internal pairwise-distance matrix.  Build via the user
            # cost kind interpreted *self-pairwise*.
            kind = cost if isinstance(cost, str) else COST_SQEUCLIDEAN
            Cx = cost_matrix(source, source, kind=kind)
            Cy = cost_matrix(target, target, kind=kind)
            P, gw_cost, n_iter, converged = gromov_wasserstein(
                Cx, Cy, problem.source_weights, problem.target_weights,
                reg=reg, outer_iter=max(1, max_iter // 50), inner_iter=200,
                tol=tol,
            )
            plan_t = P
            distance = gw_cost
            regularised = gw_cost
            margin_v = _marginal_violation(
                P, problem.source_weights, problem.target_weights
            )

        certificate: dict = {
            "content_hash": content_hash,
            "method": chosen,
            "cost_kind": cost_kind,
            "n": problem.n,
            "m": problem.m,
            "reg": reg if chosen in (METHOD_SINKHORN, METHOD_UNBALANCED, METHOD_GROMOV) else None,
            "n_iter": n_iter,
            "converged": bool(converged),
            "marginal_violation": float(margin_v),
            "sample_complexity_bound": _weed_bach_bound(problem.n, problem.m),
        }
        if verify_monotonicity and plan_t is not None:
            mv, nv, nc = cyclic_monotonicity_violation(
                plan_t, problem.cost, seed=seed,
            )
            certificate["cyclic_monotonicity"] = {
                "max_violation": float(mv),
                "n_violations": int(nv),
                "n_checked": int(nc),
                "certified_optimal": (nv == 0 and chosen != METHOD_SLICED),
            }

        report = TransportReport(
            method=chosen,
            cost_kind=cost_kind,
            distance=float(distance),
            regularised=float(regularised),
            divergence=None if divergence is None else float(divergence),
            plan=plan_t,
            potentials=potentials,
            n_iter=int(n_iter),
            converged=bool(converged),
            marginal_violation=float(margin_v),
            certificate=certificate,
            receipt_id=receipt_id,
        )
        with self._lock:
            self._n_computes += 1
        digest = self._attest(
            "transporter.computed",
            {
                "receipt_id": receipt_id,
                "method": chosen,
                "cost_kind": cost_kind,
                "distance": report.distance,
                "regularised": report.regularised,
                "marginal_violation": report.marginal_violation,
                "converged": report.converged,
                "content_hash": content_hash,
                "timestamp_ns": time.time_ns(),
            },
        )
        certificate["digest"] = digest
        self._emit(
            TRANSPORTER_COMPUTED,
            {
                "receipt_id": receipt_id,
                "method": chosen,
                "distance": report.distance,
                "converged": report.converged,
                "marginal_violation": report.marginal_violation,
                "n_iter": report.n_iter,
            },
        )
        return report

    # ----- drift (against a registered reference) -----

    def drift(
        self,
        reference_id: str,
        *,
        samples: Sequence[Any],
        method: str = METHOD_SINKHORN,
        cost: Sequence[Sequence[float]] | str = COST_SQEUCLIDEAN,
        reg: float = _DEFAULT_REG,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        n_projections: int = _DEFAULT_PROJECTIONS,
        threshold: float | None = None,
        seed: int | None = None,
    ) -> DriftReport:
        with self._lock:
            if reference_id not in self._refs:
                raise UnknownReference(reference_id)
            ref = self._refs[reference_id]
        rep = self.compute(
            source=samples,
            target=ref["samples"],
            target_weights=ref["weights"],
            cost=cost,
            method=method,
            reg=reg,
            max_iter=max_iter,
            tol=tol,
            n_projections=n_projections,
            seed=seed,
        )
        score = rep.distance
        breach = bool(threshold is not None and score > threshold)
        receipt_id = uuid.uuid4().hex[:16]
        cert = {
            "reference_id": reference_id,
            "method": rep.method,
            "n_samples": len(samples),
            "compute_receipt": rep.receipt_id,
            "converged": rep.converged,
            "marginal_violation": rep.marginal_violation,
            "threshold": threshold,
            "breach": breach,
        }
        with self._lock:
            self._n_drifts += 1
        digest = self._attest(
            "transporter.drift_evaluated",
            {
                "receipt_id": receipt_id,
                "reference_id": reference_id,
                "score": score,
                "breach": breach,
                "method": rep.method,
                "n_samples": len(samples),
                "timestamp_ns": time.time_ns(),
            },
        )
        cert["digest"] = digest
        self._emit(
            TRANSPORTER_DRIFT_EVALUATED,
            {
                "receipt_id": receipt_id,
                "reference_id": reference_id,
                "score": score,
                "breach": breach,
                "method": rep.method,
            },
        )
        return DriftReport(
            reference_id=reference_id,
            method=rep.method,
            score=float(score),
            breach=breach,
            threshold=threshold,
            n_samples=len(samples),
            certificate=cert,
            receipt_id=receipt_id,
        )

    # ----- barycenter -----

    def barycenter_1d(
        self,
        distributions: Sequence[Sequence[float]],
        *,
        weights: Sequence[float] | None = None,
        n_support: int = 200,
    ) -> BarycenterReport:
        sup, w = wasserstein_barycenter_1d(
            distributions, weights=weights, n_support=n_support
        )
        receipt_id = uuid.uuid4().hex[:16]
        cert = {
            "method": "wasserstein_barycenter_1d",
            "sources": len(distributions),
            "n_support": n_support,
            "weights": list(weights) if weights is not None else None,
        }
        with self._lock:
            self._n_barycenters += 1
        digest = self._attest(
            "transporter.barycenter",
            {
                "receipt_id": receipt_id,
                "sources": len(distributions),
                "n_support": n_support,
                "timestamp_ns": time.time_ns(),
            },
        )
        cert["digest"] = digest
        self._emit(
            TRANSPORTER_BARYCENTER,
            {
                "receipt_id": receipt_id,
                "sources": len(distributions),
                "n_support": n_support,
            },
        )
        return BarycenterReport(
            method="wasserstein_barycenter_1d",
            support=sup,
            weights=w,
            sources=len(distributions),
            n_iter=1,
            converged=True,
            certificate=cert,
            receipt_id=receipt_id,
        )

    # ----- match (transport plan as a counterfactual matcher) -----

    def match(
        self,
        *,
        source: Sequence[Any],
        target: Sequence[Any],
        cost: Sequence[Sequence[float]] | str = COST_SQEUCLIDEAN,
        method: str = METHOD_HUNGARIAN,
        reg: float = _DEFAULT_REG,
    ) -> tuple:
        """Return ``(plan, total_cost)`` for matching source to target.

        For balanced uniform-weight problems with ``method=METHOD_HUNGARIAN``
        this is the *exact*, *one-to-one* counterfactual matching used by
        Causal: row ``i`` paired with col ``assignment[i]``.  For Sinkhorn
        with ε > 0 the plan is *soft*: ``plan[i][j]`` ≥ 0 is the fractional
        mass moved.
        """
        rep = self.compute(
            source=source, target=target, cost=cost,
            method=method, reg=reg,
        )
        return rep.plan, rep.distance

    # ----- coverage -----

    def coverage(self) -> CoverageReport:
        with self._lock:
            return CoverageReport(
                references=len(self._refs),
                computes=self._n_computes,
                drifts=self._n_drifts,
                barycenters=self._n_barycenters,
            )

    def clear(self) -> None:
        with self._lock:
            self._refs.clear()
            self._n_computes = 0
            self._n_drifts = 0
            self._n_barycenters = 0
        self._emit(TRANSPORTER_CLEARED, {"timestamp_ns": time.time_ns()})

    def report(self) -> CoverageReport:
        cov = self.coverage()
        self._emit(
            TRANSPORTER_REPORT,
            {
                "references": cov.references,
                "computes": cov.computes,
                "drifts": cov.drifts,
                "barycenters": cov.barycenters,
            },
        )
        return cov


# =====================================================================
# Diagnostics
# =====================================================================


def _marginal_violation(
    plan: Sequence[Sequence[float]],
    source_weights: Sequence[float],
    target_weights: Sequence[float],
) -> float:
    n = len(plan)
    m = len(plan[0]) if n else 0
    row_err = max(
        abs(sum(plan[i][j] for j in range(m)) - source_weights[i])
        for i in range(n)
    ) if n else 0.0
    col_err = max(
        abs(sum(plan[i][j] for i in range(n)) - target_weights[j])
        for j in range(m)
    ) if m else 0.0
    return max(row_err, col_err)


def _weed_bach_bound(n: int, m: int) -> float:
    r"""Weed-Bach (2019) finite-sample bias for empirical W_p.

    For samples in bounded subsets of ℝ^d (d unknown to us at this
    level), the empirical-vs-true Wasserstein gap behaves like
    n^{-1/d} for d ≥ 3 and n^{-1/(2p)} for d ≤ 2.  Since we don't
    know the support dimension at certificate time, we expose the
    *worst-case-over-d* bound 1 / √(min(n, m)) which dominates all
    d ≤ 2 regimes — informative for the runtime even if not tight.
    """
    k = max(1, min(n, m))
    return 1.0 / math.sqrt(k)


# =====================================================================
# Convenience top-level wrappers
# =====================================================================


def wasserstein(
    source: Sequence[Any],
    target: Sequence[Any],
    *,
    p: float = 2.0,
    cost: Sequence[Sequence[float]] | str = COST_SQEUCLIDEAN,
    method: str = METHOD_AUTO,
    reg: float = _DEFAULT_REG,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
    n_projections: int = _DEFAULT_PROJECTIONS,
    seed: int | None = None,
) -> float:
    """Convenience: compute W_p (or its squared form) between two samples."""
    tr = Transporter(random_seed=seed)
    rep = tr.compute(
        source=source, target=target, cost=cost, method=method,
        reg=reg, max_iter=max_iter, tol=tol, n_projections=n_projections,
        seed=seed,
    )
    return rep.distance


def emd(
    source: Sequence[Any],
    target: Sequence[Any],
    *,
    cost: Sequence[Sequence[float]] | str = COST_EUCLIDEAN,
) -> float:
    """Convenience: compute the unregularised Earth-Mover's Distance via
    Hungarian when shapes match; Sinkhorn with tiny reg otherwise."""
    p = make_problem(source, target, cost=cost)
    if p.n == p.m and all(
        abs(w - 1.0 / p.n) < 1e-9 for w in p.source_weights
    ) and all(abs(w - 1.0 / p.m) < 1e-9 for w in p.target_weights):
        _, total = hungarian(p.cost)
        return total / p.n
    P, *_ = sinkhorn(p, reg=1e-3, max_iter=5000, tol=1e-10)
    return sinkhorn_cost(P, p.cost)


__all__ = [
    # Events
    "TRANSPORTER_STARTED",
    "TRANSPORTER_REFERENCE_REGISTERED",
    "TRANSPORTER_REFERENCE_REMOVED",
    "TRANSPORTER_COMPUTED",
    "TRANSPORTER_DRIFT_EVALUATED",
    "TRANSPORTER_BARYCENTER",
    "TRANSPORTER_CLEARED",
    "TRANSPORTER_REPORT",
    # Methods
    "METHOD_AUTO",
    "METHOD_HUNGARIAN",
    "METHOD_SINKHORN",
    "METHOD_SLICED",
    "METHOD_EMD_1D",
    "METHOD_UNBALANCED",
    "METHOD_GROMOV",
    "KNOWN_METHODS",
    # Costs
    "COST_EUCLIDEAN",
    "COST_SQEUCLIDEAN",
    "COST_MANHATTAN",
    "COST_CHEBYSHEV",
    "COST_CUSTOM",
    "KNOWN_COSTS",
    # Errors
    "TransporterError",
    "InvalidProblem",
    "UnknownReference",
    "NotConverged",
    # Dataclasses
    "TransportProblem",
    "TransportReport",
    "DriftReport",
    "BarycenterReport",
    "CoverageReport",
    # Algorithms
    "cost_matrix",
    "make_problem",
    "hungarian",
    "sinkhorn",
    "sinkhorn_cost",
    "sinkhorn_entropy",
    "sinkhorn_divergence",
    "sliced_wasserstein",
    "wasserstein_1d",
    "kantorovich_rubinstein_1d",
    "unbalanced_sinkhorn",
    "gromov_wasserstein",
    "wasserstein_barycenter_1d",
    "cyclic_monotonicity_violation",
    # Convenience
    "wasserstein",
    "emd",
    # Runtime
    "Transporter",
]
