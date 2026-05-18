r"""Topologist — topological data analysis as a runtime primitive.

Every primitive in this runtime that "looks at the shape of data" does
so through a statistical lens — moments, quantiles, kernel densities,
mixture parameters, calibration histograms.  Statistics is a contraction
that throws away geometric information that is sometimes precisely the
information a coordination engine needs.  A point cloud that has three
well-separated clusters, a point cloud that lies on a circle, a point
cloud that lies on a sphere with a hole punched in it, and a uniform
point cloud over a square *can all share the same mean and covariance*
while having radically different latent structure.  No primitive in
this runtime currently distinguishes them.

The ``Topologist`` is the runtime primitive that closes that gap.
It implements **Persistent Homology** (Edelsbrunner-Letscher-Zomorodian
2002 *Topological persistence and simplification*; Carlsson 2009
*Topology and data*) on a filtered Vietoris-Rips complex
(Vietoris 1927; Rips 1981) and returns, for every requested homological
dimension, the multiset of (birth, death) pairs that summarise the
*topological invariants* of the data at every scale.  Dimension 0
counts connected components (clusters); dimension 1 counts independent
loops (cycles); dimension 2 counts independent voids (cavities).  Each
pair has a *persistence* ``death − birth`` measuring how robust the
feature is to scale perturbation, and the diagram as a whole comes with
a **stability certificate**:

    For any two finite metric spaces ``X``, ``Y`` and any homological
    dimension ``k``,

        ``d_B(D_k(X), D_k(Y)) ≤ d_H(X, Y)``

    where ``d_B`` is the bottleneck distance between persistence
    diagrams and ``d_H`` is the Hausdorff distance between point
    clouds (Cohen-Steiner-Edelsbrunner-Harer 2007 *Stability of
    persistence diagrams*).  Stability holds for *any* underlying
    distribution — no smoothness, ergodicity, or i.i.d. assumption is
    required.

The pitch reduced to a runtime call::

    top = Topologist.create(max_dim=1, max_scale=2.5, seed=0)
    for p in points:
        top.observe(p)
    diag = top.compute()
    barcode = diag.barcode(dim=0)             # cluster stability ranking
    loops = diag.diagram(dim=1)               # circular structure
    bd = diag.bottleneck_distance(reference)  # drift in topology
    ls = diag.landscape(dim=1, num_levels=3)  # vectorised feature
    band = top.bootstrap_band(n_resamples=50, alpha=0.05)
    sig = diag.significant_features(dim=1, threshold=band.dim(1))
    report = top.report()

Every ``observe``, ``compute``, ``bootstrap_band`` and ``report``
is hashed into a SHA-256 fingerprint chain compatible with
``AttestationLedger``.

Mathematical roots
------------------

* **Vietoris 1927; Rips 1981 — Vietoris-Rips complex.**  Given a
  finite metric space ``(X, d)`` and a scale ``r ≥ 0``, the
  Vietoris-Rips complex ``VR(X, r)`` is the abstract simplicial
  complex whose ``k``-simplices are the (``k+1``)-element subsets
  ``σ ⊆ X`` with ``diam(σ) ≤ r``.  As ``r`` grows from ``0`` to
  ``∞`` the complexes nest: ``r ≤ r' ⇒ VR(X, r) ⊆ VR(X, r')``.

* **Edelsbrunner-Letscher-Zomorodian 2002 — Persistent homology.**
  The filtration ``{VR(X, r)}_{r ≥ 0}`` induces a sequence of
  inclusion-induced maps on simplicial homology groups
  ``H_k(VR(X, r); 𝔽_2)``.  The persistent ``k``-th homology classes
  are the equivalence classes of cycles that are born at some scale
  ``r_b`` (when the cycle first appears) and die at some scale
  ``r_d ≥ r_b`` (when the cycle bounds a higher-dimensional simplex).
  The collection of points ``(r_b, r_d)`` is the *persistence
  diagram* ``D_k(X)``; points on the diagonal are the diagonal
  multiset ``Δ``.

* **Elder rule (dim-0).**  Connected components admit a closed-form
  persistence algorithm: process edges in nondecreasing scale order,
  use a union-find to track components, and when an edge merges two
  components, kill the *younger* (later-born) of the two.  The
  ``Topologist`` uses Tarjan-style union-find with path compression
  and union-by-rank.

* **Standard matrix reduction (dim ≥ 1).**  Edelsbrunner-Letscher-
  Zomorodian's algorithm orders all simplices by filtration value
  (breaking ties by dimension, then index), builds the boundary
  matrix ``∂`` over 𝔽_2 = {0, 1}, and reduces it column-by-column:
  for each column ``j`` with lowest non-zero row ``low(j)``, while
  some earlier column ``j' < j`` has the same ``low(j') = low(j)``,
  XOR column ``j'`` into column ``j``.  After reduction, unpaired
  columns of dimension ``k`` are births of ``k``-features, paired
  columns are death-of-(``k-1``) / birth-of-(``k``).

* **Cohen-Steiner-Edelsbrunner-Harer 2007 — Stability.**
  The bottleneck distance between two persistence diagrams,

      ``d_B(D, D') = inf_{φ : D ∪ Δ → D' ∪ Δ bijection}
                       sup_{x ∈ D ∪ Δ} ‖x − φ(x)‖_∞``,

  is **1-Lipschitz** in the Hausdorff distance between underlying
  point clouds.  Equivalently: a perturbation of the data of size
  ``ε`` moves every persistence-diagram point by at most ``ε``
  in ``ℓ_∞``.  This is the runtime-actionable certificate: small
  data error ⇒ small diagram error.

* **Bubenik 2015 — Persistence landscapes.**  The persistence
  landscape ``λ_k : ℝ → ℝ`` is

      ``λ_k(t) = k-th max of { tent_{(b,d)}(t) : (b, d) ∈ D }``

  where ``tent_{(b,d)}(t) = max(0, min(t − b, d − t))``.  The
  landscape is a function in ``L^p(ℝ)`` and is **1-Lipschitz** in
  bottleneck distance; it lets a coordination engine vectorise a
  persistence diagram for downstream estimators while inheriting
  the stability guarantee.

* **Fasy-Lecci-Rinaldo-Wasserman-Balakrishnan-Singh 2014 — Subsampled
  bootstrap.**  For an i.i.d. sample ``X_1, …, X_n`` from ``μ``,
  let ``W_m^*`` be the bottleneck distance between the diagram of a
  random size-``m`` subsample (with replacement) and the diagram of
  the full sample.  Under a uniform Hausdorff stability of ``μ``,
  the empirical ``1 − α`` quantile of ``W_m^*`` is an asymptotic
  ``1 − α`` confidence band for the *population* diagram of ``μ``;
  features above ``2 · quantile`` from the diagonal are statistically
  significant at level ``α``.  The ``Topologist`` implements the
  bootstrap with reproducible seeded subsamples.

What this primitive *is* and *is not*
-------------------------------------

What it **is**:

  * The single rigorous answer to "how many clusters / loops /
    voids are in this point cloud, with finite-sample certificate?".
  * Distribution-free: stability holds for *any* underlying
    distribution, including non-i.i.d. and adversarial.
  * Composable: returns a structured ``PersistenceDiagram`` that
    other primitives (``Cartographer`` for curricula; ``Drift`` for
    drift detection; ``Robustifier`` for distributional robustness)
    can consume directly.
  * Auditable: every ``observe``, ``compute`` and ``bootstrap_band``
    is hashed into a fingerprint chain.

What it **is not**:

  * A full GUDHI/Dionysus replacement.  The runtime is pure-Python
    and built for the *coordination* use case (small to medium
    clouds, fast, deterministic, certified).  The 2-skeleton
    reduction is ``O((|X|^2)^ω)`` in the worst case; in practice the
    user caps ``max_scale``, ``max_points`` or ``max_dim`` so the
    complex is small.  For very large clouds (≥ a few thousand
    points and dim ≥ 1), an external library is the right tool.
  * A statistical *test* of "this data has a loop".  The
    bootstrap band is a confidence statement on the population
    diagram; the user still has to decide what "significant
    persistence" means for their application.

Investor framing
----------------

Existing primitives commit to a parametric *model class* before
they see the data (mixture of Gaussians, tree source, GP,
Markov chain).  The ``Topologist`` is the only primitive in the
runtime that supplies a *model-free, geometry-only* answer to the
shape question.  When a coordination engine has to decide

  * "Is this batch of embeddings still on the manifold the policy
    was trained on?"  (bottleneck distance to a reference)
  * "How many distinct modes do these LLM rollouts cluster into?"
    (dim-0 diagram + bootstrap band)
  * "Did the world model close a loop in latent space?"  (dim-1
    diagram, persistence above noise)
  * "Are these calibration buckets actually a 1-D curve, or has a
    new failure mode opened a hole?"  (dim-1 persistence)

— the ``Topologist`` is the primitive that answers without
committing to a model.  The output is a structured report with a
stability certificate; the coordination engine routes the decision
through the same audit ledger every other primitive emits.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from agi.events import Event, EventBus


# ---------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------

TOPOLOGIST_STARTED = "topologist.started"
TOPOLOGIST_OBSERVED = "topologist.observed"
TOPOLOGIST_COMPUTED = "topologist.computed"
TOPOLOGIST_BOOTSTRAPPED = "topologist.bootstrapped"
TOPOLOGIST_COMPARED = "topologist.compared"
TOPOLOGIST_REPORTED = "topologist.reported"
TOPOLOGIST_CLEARED = "topologist.cleared"

TOPOLOGIST_KNOWN_EVENTS = frozenset(
    {
        TOPOLOGIST_STARTED,
        TOPOLOGIST_OBSERVED,
        TOPOLOGIST_COMPUTED,
        TOPOLOGIST_BOOTSTRAPPED,
        TOPOLOGIST_COMPARED,
        TOPOLOGIST_REPORTED,
        TOPOLOGIST_CLEARED,
    }
)

# Built-in metrics
METRIC_EUCLIDEAN = "euclidean"
METRIC_SQEUCLIDEAN = "sqeuclidean"
METRIC_MANHATTAN = "manhattan"
METRIC_CHEBYSHEV = "chebyshev"
METRIC_COSINE = "cosine"
METRIC_HAMMING = "hamming"
METRIC_PRECOMPUTED = "precomputed"

TOPOLOGIST_KNOWN_METRICS = frozenset(
    {
        METRIC_EUCLIDEAN,
        METRIC_SQEUCLIDEAN,
        METRIC_MANHATTAN,
        METRIC_CHEBYSHEV,
        METRIC_COSINE,
        METRIC_HAMMING,
        METRIC_PRECOMPUTED,
    }
)


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class TopologistError(Exception):
    """Base error for the Topologist primitive."""


class InvalidConfig(TopologistError):
    """Configuration values out of range."""


class InvalidPoint(TopologistError):
    """Point coordinates rejected (wrong shape, non-numeric, …)."""


class InvalidMetric(TopologistError):
    """Metric name not in TOPOLOGIST_KNOWN_METRICS or callable rejected."""


class InsufficientData(TopologistError):
    """Operation requires at least one observation."""


class ComplexTooLarge(TopologistError):
    """The requested Vietoris-Rips complex exceeds ``max_simplices``.

    Raised before any reduction begins so the user can lower
    ``max_scale``, raise ``max_simplices``, or reduce ``max_dim``.
    """


class DimensionMismatch(TopologistError):
    """Point added with a coordinate dimension different from earlier points."""


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

_GENESIS = "0" * 64
_INF = float("inf")
_EPS = 1e-12


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _hash_link(prev: str, payload: str) -> str:
    """SHA-256 hash chain: ``H(prev || 0x1f || payload)``."""
    h = hashlib.sha256()
    h.update(prev.encode("ascii"))
    h.update(b"\x1f")
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def _payload_repr(obj: Any) -> str:
    """Canonical deterministic string repr for hashing."""
    if isinstance(obj, dict):
        keys = sorted(obj.keys())
        parts = [f"{k}={_payload_repr(obj[k])}" for k in keys]
        return "{" + ",".join(parts) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_payload_repr(x) for x in obj) + "]"
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return f"{obj:.17g}"
    return repr(obj)


def _coerce_point(p: Any, dim: int | None) -> tuple[float, ...]:
    """Validate and normalise a point into a tuple of floats."""
    try:
        seq = tuple(float(x) for x in p)
    except Exception as exc:
        raise InvalidPoint(f"point not numeric or iterable: {p!r}") from exc
    if not seq:
        raise InvalidPoint("point has zero dimension")
    for v in seq:
        if math.isnan(v) or math.isinf(v):
            raise InvalidPoint(f"point contains nan/inf: {seq}")
    if dim is not None and len(seq) != dim:
        raise DimensionMismatch(
            f"point dimension {len(seq)} does not match earlier {dim}"
        )
    return seq


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def sqeuclidean(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += abs(x - y)
    return s


def chebyshev(a: Sequence[float], b: Sequence[float]) -> float:
    m = 0.0
    for x, y in zip(a, b):
        d = abs(x - y)
        if d > m:
            m = d
    return m


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cos(a, b), clamped to [0, 2]; 0 for zero vectors paired with zero."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= _EPS and nb <= _EPS:
        return 0.0
    if na <= _EPS or nb <= _EPS:
        return 1.0
    c = dot / math.sqrt(na * nb)
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    return 1.0 - c


def hamming_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of mismatching coordinates."""
    if not a:
        return 0.0
    n = len(a)
    miss = 0
    for x, y in zip(a, b):
        if x != y:
            miss += 1
    return miss / n


_METRIC_FNS: dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {
    METRIC_EUCLIDEAN: euclidean,
    METRIC_SQEUCLIDEAN: sqeuclidean,
    METRIC_MANHATTAN: manhattan,
    METRIC_CHEBYSHEV: chebyshev,
    METRIC_COSINE: cosine_distance,
    METRIC_HAMMING: hamming_distance,
}


# ---------------------------------------------------------------------
# Union-find for dim-0 PH
# ---------------------------------------------------------------------


class _UnionFind:
    """Tarjan union-find with path compression and union-by-birth.

    ``birth[i]`` is the filtration value at which component ``i`` was
    born (always ``0`` for the dim-0 Rips filtration).  When two
    components merge at scale ``r``, the *younger* (later-born) of
    the two dies at ``r``.  Ties are broken by the smaller root index
    surviving — this keeps the implementation deterministic.
    """

    __slots__ = ("parent", "birth", "rank")

    def __init__(self, n: int, births: Sequence[float]) -> None:
        if len(births) != n:
            raise InvalidConfig("births length must match n")
        self.parent = list(range(n))
        self.birth = list(births)
        self.rank = [0] * n

    def find(self, i: int) -> int:
        # Iterative path compression to avoid Python recursion limits.
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            nxt = self.parent[i]
            self.parent[i] = root
            i = nxt
        return root

    def union(self, i: int, j: int, r: float) -> tuple[int, float] | None:
        """Merge the components of ``i`` and ``j`` at scale ``r``.

        Returns ``(killed_root, birth_of_killed)`` if a component dies,
        ``None`` if the two were already merged.  The elder rule says
        the younger (later-born) component dies; ties go to the larger
        root index dying, which keeps the surviving root canonical.
        """
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return None
        bi, bj = self.birth[ri], self.birth[rj]
        # Elder rule: younger dies. Tie-breaking by index keeps it
        # deterministic and lets the test suite reason about which
        # root survives.
        if bi < bj or (bi == bj and ri < rj):
            killer, dying = ri, rj
        else:
            killer, dying = rj, ri
        self.parent[dying] = killer
        # birth of killer stays; rank update for balance
        if self.rank[killer] == self.rank[dying]:
            self.rank[killer] += 1
        return dying, self.birth[dying]


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PersistencePair:
    """One persistence pair ``(b, d)`` of homological dimension ``k``.

    ``birth`` is the filtration scale at which the homology class
    appeared; ``death`` is the scale at which it became trivial
    (``+inf`` if it persists to the end of the filtration).
    """

    dim: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        if math.isinf(self.death):
            return _INF
        return self.death - self.birth

    @property
    def is_infinite(self) -> bool:
        return math.isinf(self.death)

    def midlife(self) -> float:
        """Average of birth and death (``inf`` if death is ``inf``)."""
        if math.isinf(self.death):
            return _INF
        return 0.5 * (self.birth + self.death)


@dataclass(frozen=True)
class PersistenceDiagram:
    """A persistence diagram: the multiset of pairs of every dimension.

    The diagram is the canonical topological summary of the data.  It
    is invariant under isometries of the point cloud and stable in the
    bottleneck distance under perturbations of the metric
    (Cohen-Steiner-Edelsbrunner-Harer 2007).
    """

    pairs: tuple[PersistencePair, ...]
    max_dim: int
    n_points: int
    max_scale: float
    metric: str

    def diagram(self, dim: int) -> tuple[PersistencePair, ...]:
        """All pairs of the given homological dimension."""
        return tuple(p for p in self.pairs if p.dim == dim)

    def barcode(self, dim: int) -> tuple[tuple[float, float], ...]:
        """Barcode: sorted (birth, death) pairs of dimension ``dim``."""
        bars = [(p.birth, p.death) for p in self.diagram(dim)]
        # Sort by birth, then death, with infinite-death bars last.
        bars.sort(key=lambda b: (b[0], _INF if math.isinf(b[1]) else b[1]))
        return tuple(bars)

    def betti(self, scale: float) -> dict[int, int]:
        """Betti numbers at scale ``scale``.

        ``β_k(r) = #{(b, d) ∈ D_k : b ≤ r < d}``.
        """
        counts: dict[int, int] = {}
        for p in self.pairs:
            if p.birth <= scale and scale < p.death:
                counts[p.dim] = counts.get(p.dim, 0) + 1
        # Always report 0 for known dimensions if absent.
        for k in range(self.max_dim + 1):
            counts.setdefault(k, 0)
        return counts

    def total_persistence(self, dim: int, p_norm: float = 1.0) -> float:
        r"""``\sum (death - birth)^p`` over finite pairs of dim ``dim``."""
        s = 0.0
        for pair in self.diagram(dim):
            if math.isinf(pair.death):
                continue
            s += (pair.death - pair.birth) ** p_norm
        return s

    def k_most_persistent(self, dim: int, k: int) -> tuple[PersistencePair, ...]:
        """Top-``k`` pairs of dim ``dim`` by persistence (``death-birth``)."""
        pairs = list(self.diagram(dim))

        def key(p: PersistencePair) -> float:
            return _INF if math.isinf(p.death) else (p.death - p.birth)

        pairs.sort(key=key, reverse=True)
        return tuple(pairs[: max(0, int(k))])

    def significant_features(
        self, dim: int, threshold: float
    ) -> tuple[PersistencePair, ...]:
        """Pairs of dim ``dim`` with ``persistence > 2 · threshold``.

        Per Fasy et al. 2014, features whose distance to the diagonal
        is more than twice the bootstrap quantile are confidence-band-
        significant at the bootstrap level.
        """
        cut = 2.0 * float(threshold)
        out = []
        for p in self.diagram(dim):
            if math.isinf(p.death):
                out.append(p)
                continue
            if (p.death - p.birth) > cut:
                out.append(p)
        return tuple(out)

    def landscape(
        self, dim: int, num_levels: int = 3, grid: int = 64
    ) -> "PersistenceLandscape":
        """Bubenik 2015 persistence landscape on a uniform grid.

        Returns the first ``num_levels`` landscape functions evaluated
        on a uniform grid over ``[t_min, t_max]`` where ``t_min`` is the
        smallest finite birth and ``t_max`` is the largest finite death.
        Infinite-death bars are clipped to ``t_max``.
        """
        if num_levels < 1:
            raise InvalidConfig("num_levels must be >= 1")
        if grid < 2:
            raise InvalidConfig("grid must be >= 2")
        pairs = self.diagram(dim)
        # Determine grid range
        births = [p.birth for p in pairs]
        finite_deaths = [p.death for p in pairs if not math.isinf(p.death)]
        if not births:
            return PersistenceLandscape(
                dim=dim,
                num_levels=num_levels,
                grid=(),
                levels=tuple(() for _ in range(num_levels)),
            )
        t_min = min(births)
        if finite_deaths:
            t_max = max(finite_deaths)
        else:
            # All infinite — use max(birth) + 1 as a placeholder
            t_max = max(births) + 1.0
        if t_max <= t_min:
            t_max = t_min + 1.0
        ts = tuple(
            t_min + (t_max - t_min) * (i / (grid - 1)) for i in range(grid)
        )
        levels: list[tuple[float, ...]] = []
        for _ in range(num_levels):
            levels.append(tuple(0.0 for _ in ts))
        for li in range(num_levels):
            row = []
            for t in ts:
                values: list[float] = []
                for p in pairs:
                    d = t_max if math.isinf(p.death) else p.death
                    tent = max(0.0, min(t - p.birth, d - t))
                    if tent > 0.0:
                        values.append(tent)
                values.sort(reverse=True)
                if li < len(values):
                    row.append(values[li])
                else:
                    row.append(0.0)
            levels[li] = tuple(row)
        return PersistenceLandscape(
            dim=dim,
            num_levels=num_levels,
            grid=ts,
            levels=tuple(levels),
        )

    def bottleneck_distance(self, other: "PersistenceDiagram", dim: int) -> float:
        """Bottleneck distance ``d_B`` between this and ``other`` at dim.

        Implementation: binary-search the threshold ``ε`` and check for a
        perfect bipartite matching between off-diagonal points and
        diagonal projections under the ``ℓ_∞`` ball of radius ``ε``.
        Returns a value provably in ``[d_B − tol, d_B + tol]`` where
        ``tol = 1e-9 + 1e-6 · diameter``.

        ``inf`` deaths in *only one* diagram contribute an unbounded
        cost; the function returns ``inf`` in that case.
        """
        return _bottleneck_distance(self.diagram(dim), other.diagram(dim))


@dataclass(frozen=True)
class PersistenceLandscape:
    """Bubenik 2015 persistence landscape on a uniform grid.

    ``levels[k]`` is the ``k``-th landscape ``λ_{k+1}`` (1-indexed
    in the paper, 0-indexed here) evaluated at ``grid`` points.
    """

    dim: int
    num_levels: int
    grid: tuple[float, ...]
    levels: tuple[tuple[float, ...], ...]

    def norm(self, p: float = 2.0) -> float:
        r"""``\big(\sum_k \int \lambda_k(t)^p dt\big)^{1/p}``, trapezoidal."""
        if len(self.grid) < 2:
            return 0.0
        dx = self.grid[1] - self.grid[0]
        s = 0.0
        for row in self.levels:
            for i in range(len(row) - 1):
                a = row[i] ** p
                b = row[i + 1] ** p
                s += 0.5 * (a + b) * dx
        return s ** (1.0 / p)

    def vector(self) -> tuple[float, ...]:
        """Concatenated levels for direct downstream use."""
        out: list[float] = []
        for row in self.levels:
            out.extend(row)
        return tuple(out)


@dataclass(frozen=True)
class BootstrapBand:
    """Confidence band on the persistence diagram (Fasy et al. 2014).

    The band gives, for each dimension, a ``1 − α`` quantile of the
    bottleneck distance between subsample diagrams and the full-sample
    diagram.  Features whose persistence exceeds ``2 · band[k]`` are
    confidence-band-significant.
    """

    alpha: float
    n_resamples: int
    subsample_size: int
    quantiles: dict[int, float]

    def dim(self, k: int) -> float:
        """Quantile at dimension ``k`` (defaults to ``0`` if absent)."""
        return float(self.quantiles.get(int(k), 0.0))


@dataclass(frozen=True)
class StabilityCertificate:
    """Quantitative restatement of the Cohen-Steiner et al. bound.

    ``hausdorff_perturbation`` is an upper bound on ``d_H(X, X')`` that
    the user is willing to tolerate; the certificate states that the
    bottleneck distance between persistence diagrams is bounded above
    by the same number.  The certificate is a *deductive* statement —
    no data-dependence — and is included in every report so the
    coordination engine can route on it without recomputing.
    """

    hausdorff_perturbation: float
    bottleneck_bound_bits: float
    statement: str


@dataclass(frozen=True)
class TopologistReport:
    """Comprehensive structured report for the coordination engine."""

    n_points: int
    dim_input: int | None
    max_dim: int
    max_scale: float
    metric: str
    n_pairs: dict[int, int]
    betti_at_max: dict[int, int]
    top_persistence: dict[int, list[tuple[float, float]]]
    fingerprint: str
    truncated: bool
    n_simplices: int
    stability: StabilityCertificate
    diagram: PersistenceDiagram | None


# ---------------------------------------------------------------------
# Bottleneck distance (bipartite matching with binary search)
# ---------------------------------------------------------------------


def _bottleneck_distance(
    da: Sequence[PersistencePair], db: Sequence[PersistencePair]
) -> float:
    """Exact (up to numerical tol) bottleneck distance between diagrams.

    Treats infinite-death pairs as a separate matching pool: an infinite
    pair must match an infinite pair, with cost ``|b - b'|``.  If the
    two diagrams have a different count of infinite pairs, returns
    ``inf``.

    For the finite remainder we use the standard "augmented" bipartite
    matching: a point ``x`` in diagram ``A`` may match either a finite
    point ``y`` in diagram ``B`` with cost ``||x - y||_∞`` or its own
    diagonal projection with cost ``persistence(x) / 2``.  Diagonal
    points are added on each side as needed so the matching is a
    bijection.

    We binary-search the threshold ``ε`` and test feasibility via the
    Hopcroft-Karp algorithm; complexity ``O(n^{2.5} log(diameter / tol))``.
    """
    a_inf = [p for p in da if math.isinf(p.death)]
    b_inf = [p for p in db if math.isinf(p.death)]
    a_fin = [p for p in da if not math.isinf(p.death)]
    b_fin = [p for p in db if not math.isinf(p.death)]

    if len(a_inf) != len(b_inf):
        return _INF

    # Cost of matching infinite pairs: solve a small min-max assignment.
    inf_cost = 0.0
    if a_inf:
        # Each infinite pair has only a birth coordinate that matters.
        ax = sorted(p.birth for p in a_inf)
        bx = sorted(p.birth for p in b_inf)
        # The min-max bipartite matching cost when both lists are 1-D
        # is achieved by sorting both and pairing by rank (sorted-rank
        # matching minimises the max gap).
        inf_cost = max(abs(x - y) for x, y in zip(ax, bx)) if ax else 0.0

    # Build finite candidate edge weights.
    A = list(a_fin)
    B = list(b_fin)
    nA = len(A)
    nB = len(B)
    # Each side gets nB / nA "diagonal" placeholders so the matching is
    # always feasible. The total node count is nA + nB.
    N = nA + nB
    if N == 0:
        return inf_cost

    # Precompute edge weights:
    #   w[i][j] for i in [0, nA), j in [0, nB):  ||A[i] - B[j]||_∞
    #   diagA[i]: A[i] to its diagonal projection: persistence(A[i]) / 2
    #   diagB[j]: B[j] to its diagonal projection: persistence(B[j]) / 2
    def linf(p: PersistencePair, q: PersistencePair) -> float:
        return max(abs(p.birth - q.birth), abs(p.death - q.death))

    def diag(p: PersistencePair) -> float:
        return 0.5 * (p.death - p.birth)

    real_edges = [[linf(A[i], B[j]) for j in range(nB)] for i in range(nA)]
    diagA = [diag(A[i]) for i in range(nA)]
    diagB = [diag(B[j]) for j in range(nB)]

    # Build edge cost matrix on the augmented graph:
    #   Left side L: A[0..nA-1]  ++  "diagonal-of-B[j]" sinks, j in 0..nB-1
    #   Right side R: B[0..nB-1] ++  "diagonal-of-A[i]" sinks, i in 0..nA-1
    # All diagonal-of-X sinks pair with diagonal-of-X sources at cost 0.
    # An A[i] paired with a "diagonal-of-A[i]" sink costs diagA[i].
    # A "diagonal-of-B[j]" source paired with B[j] costs diagB[j].
    # Two diagonal sources cross-pair at cost 0.
    L = N  # nA + nB
    R = N

    # Unique candidate ε values: real_edges entries, diagA entries,
    # diagB entries, plus 0. Binary-searching over the sorted list gives
    # an exact distance for the augmented matching.
    cand = set()
    cand.add(0.0)
    for row in real_edges:
        for v in row:
            cand.add(v)
    for v in diagA:
        cand.add(v)
    for v in diagB:
        cand.add(v)
    candidates = sorted(cand)

    def feasible(eps: float) -> bool:
        # Build bipartite adjacency.
        adj: list[list[int]] = [[] for _ in range(L)]
        for i in range(nA):
            # A[i] vs real B[j]
            for j in range(nB):
                if real_edges[i][j] <= eps + 1e-12:
                    adj[i].append(j)
            # A[i] vs diagonal-of-A[i] (right node nB + i)
            if diagA[i] <= eps + 1e-12:
                adj[i].append(nB + i)
        for j in range(nB):
            # diagonal-of-B[j] (left node nA + j) vs B[j]
            if diagB[j] <= eps + 1e-12:
                adj[nA + j].append(j)
            # diagonal-of-B[j] vs all diagonal-of-A[i] (cross-pairs at 0)
            for i in range(nA):
                # cost 0, always feasible for any eps >= 0
                adj[nA + j].append(nB + i)
        return _hopcroft_karp(adj, L, R) == L

    # Binary search on the sorted candidates.
    lo, hi = 0, len(candidates) - 1
    # Quick check: hi must be feasible.
    if not feasible(candidates[hi]):
        return _INF
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(candidates[mid]):
            hi = mid
        else:
            lo = mid + 1
    return max(inf_cost, candidates[lo])


def _hopcroft_karp(adj: list[list[int]], L: int, R: int) -> int:
    """Hopcroft-Karp maximum bipartite matching; returns matched count."""
    NIL = -1
    pair_u = [NIL] * L
    pair_v = [NIL] * R
    dist = [0] * L
    INF_LOCAL = float("inf")

    def bfs() -> bool:
        from collections import deque

        q: "deque[int]" = deque()
        found = False
        for u in range(L):
            if pair_u[u] == NIL:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF_LOCAL  # type: ignore[assignment]
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == NIL:
                    found = True
                elif dist[pu] == INF_LOCAL:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            pu = pair_v[v]
            if pu == NIL or (
                dist[pu] == dist[u] + 1 and dfs(pu)
            ):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF_LOCAL  # type: ignore[assignment]
        return False

    matched = 0
    while bfs():
        for u in range(L):
            if pair_u[u] == NIL:
                if dfs(u):
                    matched += 1
    return matched


# ---------------------------------------------------------------------
# Vietoris-Rips filtration and persistent homology
# ---------------------------------------------------------------------


def _distance_matrix(
    points: Sequence[Sequence[float]],
    metric: str | Callable[[Sequence[float], Sequence[float]], float],
    precomputed: Sequence[Sequence[float]] | None,
) -> list[list[float]]:
    """Pairwise distance matrix; ``points`` is ignored when ``precomputed``."""
    if precomputed is not None:
        n = len(precomputed)
        m = [list(row) for row in precomputed]
        if any(len(r) != n for r in m):
            raise InvalidConfig("precomputed distance matrix not square")
        for i in range(n):
            if m[i][i] != 0.0:
                m[i][i] = 0.0
            for j in range(i + 1, n):
                if abs(m[i][j] - m[j][i]) > _EPS:
                    raise InvalidConfig("precomputed matrix not symmetric")
        return m
    n = len(points)
    if isinstance(metric, str):
        fn = _METRIC_FNS.get(metric)
        if fn is None:
            raise InvalidMetric(f"unknown metric: {metric!r}")
    elif callable(metric):
        fn = metric
    else:
        raise InvalidMetric("metric must be a known name or callable")
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(fn(points[i], points[j]))
            if math.isnan(d):
                raise InvalidPoint(f"metric returned NaN for points {i}, {j}")
            if d < 0.0:
                d = 0.0
            m[i][j] = d
            m[j][i] = d
    return m


def _enumerate_simplices(
    dist: list[list[float]], max_dim: int, max_scale: float, max_simplices: int
) -> list[tuple[tuple[int, ...], float, int]]:
    """All simplices of the Vietoris-Rips complex up to ``max_dim``.

    A simplex ``σ = (i_0, …, i_k)`` is included iff every pairwise
    distance ``d(i_a, i_b) ≤ max_scale``.  Filtration value is the
    diameter of ``σ``.

    Returns a list of ``(vertices, filt_value, dim)`` triples.  Raises
    ``ComplexTooLarge`` if the count exceeds ``max_simplices``.
    """
    n = len(dist)
    out: list[tuple[tuple[int, ...], float, int]] = []
    # 0-simplices
    for i in range(n):
        out.append(((i,), 0.0, 0))
    # 1-simplices
    edges: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i][j]
            if d <= max_scale:
                edges.append((i, j, d))
    edges.sort(key=lambda e: (e[2], e[0], e[1]))
    for i, j, d in edges:
        out.append(((i, j), d, 1))
        if len(out) > max_simplices:
            raise ComplexTooLarge(
                f"Vietoris-Rips complex exceeds max_simplices={max_simplices}"
            )
    if max_dim < 2:
        return out
    # Higher-dim: inductive coface enumeration via "lower neighbours".
    # neighbours[i] = sorted list of j > i such that dist[i][j] <= max_scale.
    neighbours = [[] for _ in range(n)]
    for i, j, _ in edges:
        neighbours[i].append(j)
    # Already sorted by edge order? Re-sort by index for set intersection.
    for lst in neighbours:
        lst.sort()
    # All k-simplices for k = 2 .. max_dim
    current: list[tuple[int, ...]] = [(i, j) for i, j, _ in edges]
    for k in range(2, max_dim + 1):
        nxt: list[tuple[int, ...]] = []
        for s in current:
            # Common lower-neighbours of all vertices in s above s[-1]
            common: list[int] = list(neighbours[s[0]])
            for v in s[1:]:
                # intersect with neighbours[v]
                lst = neighbours[v]
                # Both sorted; classic merge intersection
                ci = 0
                cj = 0
                new_common: list[int] = []
                while ci < len(common) and cj < len(lst):
                    if common[ci] == lst[cj]:
                        new_common.append(common[ci])
                        ci += 1
                        cj += 1
                    elif common[ci] < lst[cj]:
                        ci += 1
                    else:
                        cj += 1
                common = new_common
                if not common:
                    break
            for v in common:
                if v <= s[-1]:
                    continue
                t = s + (v,)
                # filtration value = max pairwise distance
                fv = 0.0
                tn = len(t)
                for a in range(tn):
                    for b in range(a + 1, tn):
                        dd = dist[t[a]][t[b]]
                        if dd > fv:
                            fv = dd
                nxt.append(t)
                out.append((t, fv, k))
                if len(out) > max_simplices:
                    raise ComplexTooLarge(
                        f"Vietoris-Rips complex exceeds max_simplices={max_simplices}"
                    )
        current = nxt
        if not current:
            break
    return out


def _order_simplices(
    simplices: list[tuple[tuple[int, ...], float, int]]
) -> list[tuple[tuple[int, ...], float, int]]:
    """Stable order: filt_value asc, dim asc, vertices asc.

    The dimension tiebreaker is required for correctness: a coface
    must appear after all its faces even when they share a filtration
    value (which is the generic case in Vietoris-Rips).
    """
    indexed = sorted(simplices, key=lambda s: (s[1], s[2], s[0]))
    return indexed


def _boundary(
    simplex: tuple[int, ...], index_of: dict[tuple[int, ...], int]
) -> list[int]:
    """Indices of the faces of ``simplex`` in the global ordering."""
    if len(simplex) <= 1:
        return []
    faces: list[int] = []
    for k in range(len(simplex)):
        face = simplex[:k] + simplex[k + 1 :]
        idx = index_of.get(face)
        if idx is not None:
            faces.append(idx)
    return faces


def _persistence_via_reduction(
    simplices: list[tuple[tuple[int, ...], float, int]]
) -> list[PersistencePair]:
    """Standard Z_2 boundary-matrix reduction.

    Returns the list of persistence pairs.  Unpaired columns of dim
    ``k`` correspond to essential (infinite-death) homology classes
    of dim ``k``.
    """
    n = len(simplices)
    index_of: dict[tuple[int, ...], int] = {s[0]: i for i, s in enumerate(simplices)}
    # cols[j]: set of row indices of non-zero entries in column j (Z_2).
    cols: list[set[int]] = [set() for _ in range(n)]
    for j, (sigma, _fv, dim) in enumerate(simplices):
        if dim == 0:
            continue
        for face_idx in _boundary(sigma, index_of):
            cols[j].add(face_idx)
    # low[j] = max row index of non-zero entry in column j; None if empty.
    low: list[int | None] = [None] * n
    # low_to_col: row -> column index whose pivot is at that row.
    low_to_col: dict[int, int] = {}
    paired_birth: dict[int, int] = {}  # row of dying column -> col that killed it
    for j in range(n):
        while cols[j]:
            r = max(cols[j])
            other = low_to_col.get(r)
            if other is None:
                low[j] = r
                low_to_col[r] = j
                break
            # XOR column 'other' into cols[j]
            cols[j].symmetric_difference_update(cols[other])
        if low[j] is None:
            # column reduced to empty; this simplex *creates* a class.
            pass
        else:
            paired_birth[low[j]] = j
    pairs: list[PersistencePair] = []
    paired_set: set[int] = set()
    for birth_idx, death_idx in paired_birth.items():
        b_sigma, b_fv, b_dim = simplices[birth_idx]
        d_sigma, d_fv, d_dim = simplices[death_idx]
        if d_fv <= b_fv + 1e-12:
            # Zero-persistence pair; skip (the diagonal absorbs it).
            paired_set.add(birth_idx)
            paired_set.add(death_idx)
            continue
        # The dimension is the dim of the *birth* simplex.
        pairs.append(PersistencePair(dim=b_dim, birth=b_fv, death=d_fv))
        paired_set.add(birth_idx)
        paired_set.add(death_idx)
    # Unpaired creators are infinite-death pairs.
    for j, (_sigma, fv, dim) in enumerate(simplices):
        if j in paired_set:
            continue
        if low[j] is not None:
            # Column was a destroyer for a non-existent birth; should not happen.
            continue
        # Column reduced to empty AND not paired => creator with no death.
        pairs.append(PersistencePair(dim=dim, birth=fv, death=_INF))
    return pairs


def _dim0_persistence_fast(
    dist: list[list[float]], max_scale: float
) -> tuple[list[PersistencePair], int]:
    """Closed-form dim-0 persistence via Kruskal-style union-find.

    Returns ``(pairs, n_edges_used)``.  All points are born at scale
    ``0``; the one surviving root is the essential infinite-death
    component.  Equivalent to running the standard reduction on the
    1-skeleton, in ``O(|E| α(n))`` time.
    """
    n = len(dist)
    uf = _UnionFind(n, [0.0] * n)
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i][j]
            if d <= max_scale:
                edges.append((d, i, j))
    edges.sort(key=lambda e: (e[0], e[1], e[2]))
    pairs: list[PersistencePair] = []
    for d, i, j in edges:
        merged = uf.union(i, j, d)
        if merged is not None:
            _, birth = merged
            if d > birth + 1e-12:
                pairs.append(PersistencePair(dim=0, birth=birth, death=d))
    # The remaining "alive" roots are essential dim-0 features.
    seen: set[int] = set()
    for i in range(n):
        r = uf.find(i)
        if r not in seen:
            seen.add(r)
            pairs.append(PersistencePair(dim=0, birth=0.0, death=_INF))
    return pairs, len(edges)


# ---------------------------------------------------------------------
# Topologist
# ---------------------------------------------------------------------


class Topologist:
    """Topological data analysis as a runtime primitive.

    Thread-safe.  Use ``Topologist.create(...)`` to construct.
    """

    def __init__(
        self,
        *,
        max_dim: int = 1,
        max_scale: float = float("inf"),
        metric: str | Callable[[Sequence[float], Sequence[float]], float] = METRIC_EUCLIDEAN,
        max_simplices: int = 200_000,
        max_points: int = 256,
        seed: int = 0,
        session_id: str | None = None,
        bus: EventBus | None = None,
    ) -> None:
        if max_dim < 0:
            raise InvalidConfig("max_dim must be >= 0")
        if max_dim > 5:
            raise InvalidConfig("max_dim > 5 is not supported")
        if max_scale <= 0.0 and not math.isinf(max_scale):
            raise InvalidConfig("max_scale must be > 0 or +inf")
        if max_simplices < 1:
            raise InvalidConfig("max_simplices must be >= 1")
        if max_points < 1:
            raise InvalidConfig("max_points must be >= 1")
        if isinstance(metric, str) and metric not in TOPOLOGIST_KNOWN_METRICS:
            raise InvalidMetric(f"unknown metric: {metric!r}")
        if not isinstance(metric, str) and not callable(metric):
            raise InvalidMetric("metric must be a known name or callable")
        self._max_dim = int(max_dim)
        self._max_scale = float(max_scale)
        self._metric = metric
        self._max_simplices = int(max_simplices)
        self._max_points = int(max_points)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._session_id = session_id or f"top-{uuid.uuid4().hex[:8]}"
        self._bus = bus
        self._lock = threading.RLock()
        self._points: list[tuple[float, ...]] = []
        self._point_ids: list[str] = []
        self._dim_input: int | None = None
        self._precomputed: list[list[float]] | None = None
        self._fingerprint: str = _GENESIS
        self._diagram: PersistenceDiagram | None = None
        self._last_n_simplices: int = 0
        self._last_truncated: bool = False
        self._emit(TOPOLOGIST_STARTED, {"max_dim": self._max_dim, "metric": self._metric_name()})

    # ----- construction -----

    @classmethod
    def create(
        cls,
        *,
        max_dim: int = 1,
        max_scale: float = float("inf"),
        metric: str | Callable[[Sequence[float], Sequence[float]], float] = METRIC_EUCLIDEAN,
        max_simplices: int = 200_000,
        max_points: int = 256,
        seed: int = 0,
        session_id: str | None = None,
        bus: EventBus | None = None,
    ) -> "Topologist":
        return cls(
            max_dim=max_dim,
            max_scale=max_scale,
            metric=metric,
            max_simplices=max_simplices,
            max_points=max_points,
            seed=seed,
            session_id=session_id,
            bus=bus,
        )

    def _metric_name(self) -> str:
        if isinstance(self._metric, str):
            return self._metric
        return getattr(self._metric, "__name__", "custom_callable")

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            Event(kind=kind, session_id=self._session_id, data=dict(data))
        )

    def _chain(self, payload: Any) -> str:
        self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
        return self._fingerprint

    # ----- ingestion -----

    def observe(
        self,
        point: Sequence[float],
        *,
        point_id: str | None = None,
    ) -> str:
        """Add a single point to the cloud and invalidate the diagram."""
        with self._lock:
            if self._precomputed is not None:
                raise InvalidConfig(
                    "cannot observe coordinates after observe_distance_matrix"
                )
            coords = _coerce_point(point, self._dim_input)
            if self._dim_input is None:
                self._dim_input = len(coords)
            if len(self._points) >= self._max_points:
                raise InvalidConfig(
                    f"point cloud already at max_points={self._max_points}"
                )
            self._points.append(coords)
            pid = point_id if point_id is not None else f"p{len(self._points) - 1}"
            self._point_ids.append(pid)
            self._diagram = None
            fp = self._chain(("observe", pid, list(coords)))
            self._emit(
                TOPOLOGIST_OBSERVED,
                {"point_id": pid, "n_points": len(self._points), "fingerprint": fp},
            )
            return pid

    def observe_batch(self, points: Iterable[Sequence[float]]) -> list[str]:
        ids: list[str] = []
        for p in points:
            ids.append(self.observe(p))
        return ids

    def observe_distance_matrix(self, dm: Sequence[Sequence[float]]) -> None:
        """Skip coordinates: provide a precomputed distance matrix.

        After this call ``observe`` is disabled — the cloud is fixed.
        """
        with self._lock:
            if self._points:
                raise InvalidConfig(
                    "cannot observe_distance_matrix after coordinate observations"
                )
            n = len(dm)
            if n < 1:
                raise InvalidConfig("distance matrix must have at least 1 row")
            if n > self._max_points:
                raise InvalidConfig(
                    f"matrix size {n} exceeds max_points={self._max_points}"
                )
            m = [list(map(float, row)) for row in dm]
            if any(len(r) != n for r in m):
                raise InvalidConfig("distance matrix not square")
            for i in range(n):
                for j in range(n):
                    v = m[i][j]
                    if math.isnan(v) or math.isinf(v):
                        raise InvalidConfig("distance matrix has nan/inf")
                    if v < 0:
                        raise InvalidConfig("distance matrix has negative entry")
                if m[i][i] != 0.0:
                    raise InvalidConfig("distance matrix diagonal must be 0")
                for j in range(i + 1, n):
                    if abs(m[i][j] - m[j][i]) > 1e-9:
                        raise InvalidConfig("distance matrix not symmetric")
            self._precomputed = m
            self._point_ids = [f"p{i}" for i in range(n)]
            self._dim_input = None
            self._diagram = None
            fp = self._chain(("observe_dm", n))
            self._emit(
                TOPOLOGIST_OBSERVED,
                {"matrix_size": n, "fingerprint": fp},
            )

    def n_points(self) -> int:
        with self._lock:
            if self._precomputed is not None:
                return len(self._precomputed)
            return len(self._points)

    # ----- core computation -----

    def compute(
        self,
        *,
        max_scale: float | None = None,
        max_dim: int | None = None,
    ) -> PersistenceDiagram:
        """Build the Vietoris-Rips filtration and return the diagram.

        ``max_scale`` and ``max_dim`` override the constructor defaults
        for this call only.  Subsequent ``compute()`` calls revert.
        """
        with self._lock:
            n = self.n_points()
            if n < 1:
                raise InsufficientData("compute() requires at least 1 observed point")
            md = self._max_dim if max_dim is None else int(max_dim)
            if md < 0 or md > 5:
                raise InvalidConfig("max_dim out of range")
            ms = self._max_scale if max_scale is None else float(max_scale)
            if ms <= 0.0 and not math.isinf(ms):
                raise InvalidConfig("max_scale must be > 0 or +inf")
            # Build distance matrix once
            if self._precomputed is not None:
                dist = self._precomputed
            else:
                dist = _distance_matrix(self._points, self._metric, None)
            # If the user hasn't capped max_scale, use the enclosing
            # radius — the largest finite pairwise distance.
            effective_ms = ms
            if math.isinf(ms):
                m = 0.0
                for i in range(n):
                    for j in range(i + 1, n):
                        if dist[i][j] > m:
                            m = dist[i][j]
                effective_ms = m if m > 0.0 else 1.0
            truncated = False
            if md == 0:
                pairs, n_edges = _dim0_persistence_fast(dist, effective_ms)
                n_simplices = n + n_edges
            else:
                # To compute H_k we need (k+1)-simplices to kill k-cycles.
                # Build up to the (max_dim + 1)-skeleton.
                simplex_dim = md + 1
                try:
                    simplices = _enumerate_simplices(
                        dist, simplex_dim, effective_ms, self._max_simplices
                    )
                except ComplexTooLarge:
                    # Auto-degrade: cap scale to whatever yields at most
                    # max_simplices simplices and flag truncated=True.
                    truncated = True
                    simplices = _enumerate_simplices_capped(
                        dist, simplex_dim, effective_ms, self._max_simplices
                    )
                ordered = _order_simplices(simplices)
                pairs = _persistence_via_reduction(ordered)
                # Drop pairs whose birth dimension exceeds max_dim — those
                # would only die via (max_dim+2)-simplices, which we never
                # built, so we cannot trust their infinite-death status.
                pairs = [p for p in pairs if p.dim <= md]
                n_simplices = len(ordered)
            diag = PersistenceDiagram(
                pairs=tuple(pairs),
                max_dim=md,
                n_points=n,
                max_scale=effective_ms,
                metric=self._metric_name(),
            )
            self._diagram = diag
            self._last_n_simplices = n_simplices
            self._last_truncated = truncated
            fp = self._chain(("compute", md, effective_ms, len(pairs)))
            self._emit(
                TOPOLOGIST_COMPUTED,
                {
                    "max_dim": md,
                    "max_scale": effective_ms,
                    "n_simplices": n_simplices,
                    "n_pairs": len(pairs),
                    "truncated": truncated,
                    "fingerprint": fp,
                },
            )
            return diag

    def diagram(self) -> PersistenceDiagram:
        """Return the most recently computed diagram, computing if needed."""
        with self._lock:
            if self._diagram is None:
                return self.compute()
            return self._diagram

    # ----- statistical inference: bootstrap confidence band -----

    def bootstrap_band(
        self,
        *,
        n_resamples: int = 50,
        subsample_size: int | None = None,
        alpha: float = 0.05,
        max_dim: int | None = None,
        max_scale: float | None = None,
    ) -> BootstrapBand:
        """Fasy et al. 2014 subsampled bootstrap confidence band.

        For each resample, draws ``subsample_size`` points with
        replacement, computes the diagram, and records the bottleneck
        distance to the full-sample diagram.  Returns the empirical
        ``1 − α`` quantile per dimension.
        """
        if n_resamples < 1:
            raise InvalidConfig("n_resamples must be >= 1")
        if not (0.0 < alpha < 1.0):
            raise InvalidConfig("alpha must be in (0, 1)")
        with self._lock:
            n = self.n_points()
            if n < 2:
                raise InsufficientData("bootstrap_band requires at least 2 points")
            md = self._max_dim if max_dim is None else int(max_dim)
            ms = self._max_scale if max_scale is None else float(max_scale)
            ss = subsample_size if subsample_size is not None else max(2, n // 2)
            if ss < 2:
                raise InvalidConfig("subsample_size must be >= 2")
            # Get the full diagram (with override)
            full = self.compute(max_dim=md, max_scale=ms)
            # Resample
            per_dim_distances: dict[int, list[float]] = {k: [] for k in range(md + 1)}
            for _ in range(n_resamples):
                if self._precomputed is not None:
                    idx = [self._rng.randrange(n) for _ in range(ss)]
                    sub = [
                        [self._precomputed[i][j] for j in idx] for i in idx
                    ]
                    sub_top = Topologist(
                        max_dim=md,
                        max_scale=self._max_scale,
                        max_simplices=self._max_simplices,
                        max_points=max(self._max_points, ss),
                        seed=0,
                    )
                    sub_top.observe_distance_matrix(sub)
                else:
                    idx = [self._rng.randrange(n) for _ in range(ss)]
                    sub_points = [self._points[i] for i in idx]
                    sub_top = Topologist(
                        max_dim=md,
                        max_scale=self._max_scale,
                        metric=self._metric,
                        max_simplices=self._max_simplices,
                        max_points=max(self._max_points, ss),
                        seed=0,
                    )
                    for p in sub_points:
                        sub_top.observe(p)
                sub_diag = sub_top.compute(max_dim=md, max_scale=ms)
                for k in range(md + 1):
                    d = full.bottleneck_distance(sub_diag, k)
                    if math.isinf(d):
                        d = max(self._diag_max_finite(full, k), 0.0)
                    per_dim_distances[k].append(d)
            quantiles: dict[int, float] = {}
            for k, ds in per_dim_distances.items():
                quantiles[k] = _quantile(ds, 1.0 - alpha)
            band = BootstrapBand(
                alpha=alpha,
                n_resamples=n_resamples,
                subsample_size=ss,
                quantiles=quantiles,
            )
            fp = self._chain(
                (
                    "bootstrap",
                    n_resamples,
                    ss,
                    alpha,
                    [(k, round(v, 9)) for k, v in sorted(quantiles.items())],
                )
            )
            self._emit(
                TOPOLOGIST_BOOTSTRAPPED,
                {
                    "alpha": alpha,
                    "n_resamples": n_resamples,
                    "subsample_size": ss,
                    "quantiles": {str(k): v for k, v in quantiles.items()},
                    "fingerprint": fp,
                },
            )
            return band

    @staticmethod
    def _diag_max_finite(d: PersistenceDiagram, dim: int) -> float:
        m = 0.0
        for p in d.diagram(dim):
            if math.isinf(p.death):
                continue
            v = p.death - p.birth
            if v > m:
                m = v
        return m

    # ----- comparison / drift detection -----

    def bottleneck_to(
        self, other: PersistenceDiagram, dim: int
    ) -> float:
        """Bottleneck distance from this diagram (computed if needed) to other."""
        with self._lock:
            mine = self.diagram()
            d = mine.bottleneck_distance(other, dim)
            fp = self._chain(("bottleneck_to", dim, round(d, 12) if not math.isinf(d) else "inf"))
            self._emit(
                TOPOLOGIST_COMPARED,
                {"dim": dim, "bottleneck_distance": d, "fingerprint": fp},
            )
            return d

    # ----- audit & state management -----

    def fingerprint(self) -> str:
        with self._lock:
            return self._fingerprint

    def clear(self) -> None:
        with self._lock:
            self._points = []
            self._point_ids = []
            self._dim_input = None
            self._precomputed = None
            self._diagram = None
            self._last_n_simplices = 0
            self._last_truncated = False
            fp = self._chain(("clear",))
            self._emit(TOPOLOGIST_CLEARED, {"fingerprint": fp})

    def stability_certificate(
        self, hausdorff_perturbation: float
    ) -> StabilityCertificate:
        """Deductive bottleneck-stability statement for the user's ε."""
        if hausdorff_perturbation < 0:
            raise InvalidConfig("hausdorff_perturbation must be >= 0")
        return StabilityCertificate(
            hausdorff_perturbation=float(hausdorff_perturbation),
            bottleneck_bound_bits=float(hausdorff_perturbation),
            statement=(
                f"For any X' with d_H(X, X') ≤ {hausdorff_perturbation:.6g}, "
                "for every homological dimension k, "
                f"d_B(D_k(X), D_k(X')) ≤ {hausdorff_perturbation:.6g} "
                "(Cohen-Steiner-Edelsbrunner-Harer 2007)."
            ),
        )

    def report(
        self,
        *,
        top_k: int = 5,
        eval_scale: float | None = None,
        hausdorff_perturbation: float | None = None,
    ) -> TopologistReport:
        """Comprehensive report for the coordination engine."""
        with self._lock:
            diag = self.diagram()
            scale = (
                eval_scale
                if eval_scale is not None
                else (diag.max_scale if not math.isinf(diag.max_scale) else 0.0)
            )
            n_pairs: dict[int, int] = {k: 0 for k in range(diag.max_dim + 1)}
            for p in diag.pairs:
                n_pairs[p.dim] = n_pairs.get(p.dim, 0) + 1
            betti = diag.betti(scale)
            top: dict[int, list[tuple[float, float]]] = {}
            for k in range(diag.max_dim + 1):
                top[k] = [
                    (pp.birth, pp.death)
                    for pp in diag.k_most_persistent(k, top_k)
                ]
            haus = (
                hausdorff_perturbation
                if hausdorff_perturbation is not None
                else 0.0
            )
            cert = self.stability_certificate(haus)
            rep = TopologistReport(
                n_points=self.n_points(),
                dim_input=self._dim_input,
                max_dim=diag.max_dim,
                max_scale=diag.max_scale,
                metric=diag.metric,
                n_pairs=dict(n_pairs),
                betti_at_max=betti,
                top_persistence=top,
                fingerprint=self._fingerprint,
                truncated=self._last_truncated,
                n_simplices=self._last_n_simplices,
                stability=cert,
                diagram=diag,
            )
            fp = self._chain(
                (
                    "report",
                    {
                        "n_pairs": [(k, v) for k, v in sorted(n_pairs.items())],
                        "betti": [(k, v) for k, v in sorted(betti.items())],
                    },
                )
            )
            self._emit(
                TOPOLOGIST_REPORTED,
                {
                    "n_pairs": {str(k): v for k, v in n_pairs.items()},
                    "betti": {str(k): v for k, v in betti.items()},
                    "fingerprint": fp,
                },
            )
            return rep

    # ----- pure-function exports -----

    @staticmethod
    def from_points(
        points: Sequence[Sequence[float]],
        *,
        max_dim: int = 1,
        max_scale: float = float("inf"),
        metric: str = METRIC_EUCLIDEAN,
        max_simplices: int = 200_000,
    ) -> PersistenceDiagram:
        """One-shot: compute a diagram from a list of points."""
        n = len(points)
        if n < 1:
            raise InsufficientData("from_points requires at least 1 point")
        top = Topologist(
            max_dim=max_dim,
            max_scale=max_scale,
            metric=metric,
            max_simplices=max_simplices,
            max_points=max(n, 1),
        )
        for p in points:
            top.observe(p)
        return top.compute()


# ---------------------------------------------------------------------
# Auto-truncation helper
# ---------------------------------------------------------------------


def _enumerate_simplices_capped(
    dist: list[list[float]], max_dim: int, max_scale: float, max_simplices: int
) -> list[tuple[tuple[int, ...], float, int]]:
    """Like ``_enumerate_simplices`` but binary-searches a scale cap so
    the complex fits in ``max_simplices``.

    Used when the user-requested scale is too aggressive.  The returned
    complex is the largest depth-``max_dim`` Vietoris-Rips complex that
    fits, with scale ≤ ``max_scale``.
    """
    # Collect all candidate scales (the filtration values of all
    # pairwise distances ≤ max_scale, plus 0).
    n = len(dist)
    scales: set[float] = {0.0}
    for i in range(n):
        for j in range(i + 1, n):
            v = dist[i][j]
            if v <= max_scale:
                scales.add(v)
    sorted_scales = sorted(scales)
    if not sorted_scales:
        return [((i,), 0.0, 0) for i in range(n)]
    lo, hi = 0, len(sorted_scales) - 1
    best: list[tuple[tuple[int, ...], float, int]] = [
        ((i,), 0.0, 0) for i in range(n)
    ]
    while lo <= hi:
        mid = (lo + hi) // 2
        s = sorted_scales[mid]
        try:
            built = _enumerate_simplices(dist, max_dim, s, max_simplices)
            best = built
            lo = mid + 1
        except ComplexTooLarge:
            hi = mid - 1
    return best


# ---------------------------------------------------------------------
# Quantile helper
# ---------------------------------------------------------------------


def _quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of a non-empty sequence."""
    if not xs:
        return 0.0
    if q <= 0.0:
        return min(xs)
    if q >= 1.0:
        return max(xs)
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


__all__ = [
    "Topologist",
    "TopologistError",
    "InvalidConfig",
    "InvalidPoint",
    "InvalidMetric",
    "InsufficientData",
    "ComplexTooLarge",
    "DimensionMismatch",
    "PersistencePair",
    "PersistenceDiagram",
    "PersistenceLandscape",
    "BootstrapBand",
    "StabilityCertificate",
    "TopologistReport",
    "TOPOLOGIST_STARTED",
    "TOPOLOGIST_OBSERVED",
    "TOPOLOGIST_COMPUTED",
    "TOPOLOGIST_BOOTSTRAPPED",
    "TOPOLOGIST_COMPARED",
    "TOPOLOGIST_REPORTED",
    "TOPOLOGIST_CLEARED",
    "TOPOLOGIST_KNOWN_EVENTS",
    "TOPOLOGIST_KNOWN_METRICS",
    "METRIC_EUCLIDEAN",
    "METRIC_SQEUCLIDEAN",
    "METRIC_MANHATTAN",
    "METRIC_CHEBYSHEV",
    "METRIC_COSINE",
    "METRIC_HAMMING",
    "METRIC_PRECOMPUTED",
    "euclidean",
    "sqeuclidean",
    "manhattan",
    "chebyshev",
    "cosine_distance",
    "hamming_distance",
]
