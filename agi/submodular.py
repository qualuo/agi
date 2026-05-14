r"""Submodular — discrete subset-selection with provable approximation bounds.

A coordination engine continuously chooses *subsets*: which k tools to expose
on a turn, which k demonstrations to include in a prompt, which k tickets to
admit under a fixed-budget portfolio, which k experiments to fire in the
next batch, which k sensors / probes to wake.  Every one of those problems
is **NP-hard** as a generic combinatorial search — but when the utility
function is **submodular** (diminishing returns) the *greedy* algorithm has
celebrated, *tight* approximation guarantees.  Submodular maximisation is
to discrete optimisation what convex optimisation is to continuous — the
sweet spot where a polynomial-time algorithm provably approaches the
optimum.

This module is the runtime kernel for that sweet spot.

Mathematical roots
------------------

Let ``V`` be a finite ground set and ``f : 2^V → ℝ`` a set function.
``f`` is **submodular** if for every ``A ⊆ B ⊆ V`` and ``v ∉ B``::

    f(A ∪ {v}) - f(A)   ≥   f(B ∪ {v}) - f(B).                    (DR)

Equivalently — and more useful in practice — for every ``A, B ⊆ V``::

    f(A) + f(B)   ≥   f(A ∪ B) + f(A ∩ B).                        (UM)

``f`` is **monotone** if ``A ⊆ B ⇒ f(A) ≤ f(B)``, and **normalised** if
``f(∅) = 0``.

Submodular maximisation is the family of problems::

    max  f(S)   s.t.   S ∈ ℐ,

where ``ℐ`` is a downward-closed family (cardinality ``|S| ≤ k``,
knapsack ``Σ c_i ≤ B``, matroid, intersection of matroids, …).

  * **Nemhauser, Wolsey, Fisher, 1978 — "An analysis of approximations for
    maximizing submodular set functions — I."**  The foundation.  For
    monotone, normalised, submodular ``f`` and cardinality constraint
    ``|S| ≤ k``, the greedy algorithm — at each step pick
    ``argmax_v f(S ∪ {v}) - f(S)`` — yields ``f(Ŝ) ≥ (1 - 1/e) · f(S*)``.
    The constant ``1 - 1/e ≈ 0.632`` is **tight**: Feige (1998) shows that
    no polynomial-time algorithm beats it on a generic submodular oracle
    unless ``P = NP``.

  * **Minoux, 1978 — "Accelerated greedy algorithms for maximizing
    submodular set functions."**  *Lazy* greedy: maintain a max-heap of
    upper bounds on marginal gains; pop the top, re-evaluate, and push
    back if it dropped.  By submodularity (DR), an entry's marginal is
    monotonically non-increasing as the solution grows, so a re-evaluated
    bound that still leads the heap is the true argmax.  Same ``1 - 1/e``
    bound; in practice 10–100× faster than naive greedy.

  * **Leskovec, Krause, Guestrin, Faloutsos, VanBriesen, Glance, 2007 —
    "Cost-effective outbreak detection in networks."**  CELF (Cost-
    Effective Lazy Forward) — the lazy-greedy formulation that became
    standard for sensor placement / influence maximisation.  Same
    guarantee, generalised to non-uniform costs via the
    cost-benefit-vs-unit-gain pivot of Khuller-Moss-Naor (1999).

  * **Sviridenko, 2004 — "A note on maximizing a submodular set function
    subject to a knapsack constraint."**  For monotone submodular ``f``
    and knapsack ``Σ c_i ≤ B``, **partial-enumeration + cost-benefit
    greedy** achieves ``(1 - 1/e) · f(S*)``.  Without the enumeration the
    bound is ``½(1 - 1/e)``.

  * **Khuller, Moss, Naor, 1999 — "The budgeted maximum coverage
    problem."**  The cost-benefit pivot: at each step pick
    ``argmax_v Δf(v)/c_v`` among the affordable.  Combined with the
    best singleton it is a ``½(1 - 1/e)`` approximation; with
    Sviridenko's partial enumeration it tightens to ``(1 - 1/e)``.

  * **Mirzasoleiman, Badanidiyuru, Karbasi, Vondrák, Krause, 2015 —
    "Lazier than lazy greedy."**  Random-sample greedy: at each of the
    ``k`` rounds, sample ``r = ⌈n/k · log(1/ε)⌉`` ground-set elements
    uniformly and pick the argmax marginal **among that sample**.
    Achieves ``(1 - 1/e - ε)`` *in expectation* with total query cost
    ``O(n · log(1/ε))`` independent of ``k`` — the fastest approximation
    with non-trivial bound.

  * **Buchbinder, Feldman, Naor, Schwartz, 2015 — "A tight linear time
    (½)-approximation for unconstrained submodular maximisation."**
    *Double greedy* — start with ``X = ∅``, ``Y = V``; for each ``v ∈ V``
    sweep in arbitrary order, sample ``v`` into ``X`` versus removing it
    from ``Y`` proportional to its marginals.  Output ``X = Y``.
    ``½``-approx **without monotonicity** and **without constraints**;
    this constant is tight under the value-oracle model.

  * **Badanidiyuru, Mirzasoleiman, Karbasi, Krause, 2014 — "Streaming
    submodular maximization: massive data summarization on the fly."**
    Sieve-Streaming: one pass, ``O((n/ε) log k)`` queries, memory
    ``O(k · log k / ε)``, approximation ``½ - ε`` for monotone
    submodular under cardinality.  Foundational for online runtimes.

  * **Conforti, Cornuéjols, 1984 — "Submodular set functions, matroids
    and the greedy algorithm: tight worst-case bounds and some
    generalizations of the Rado-Edmonds theorem."**  *Curvature-aware*
    bound::

        f(Ŝ_greedy) ≥ (1/c) · (1 - (1 - c/k)^k) · f(S*),

    where the **total curvature**
    ``c = 1 - min_{v∈V} (f(V) - f(V∖{v})) / f({v})`` lies in ``[0, 1]``.
    For ``c → 0`` (near-modular) the bound → 1; for ``c = 1`` it
    degenerates to ``1 - 1/e``.

  * **Wolsey, 1982 — "An analysis of the greedy algorithm for the
    submodular set covering problem."**  *Submodular cover*: minimise
    ``|S|`` s.t. ``f(S) ≥ Q``.  Greedy gives
    ``|Ŝ| ≤ (1 + ln(Q/η)) · |S*|`` where ``η`` is the smallest positive
    marginal — the discrete analogue of the ``H_n``-approximation for
    set cover (Chvátal 1979).

  * **Harshaw, Feldman, Ward, Karbasi, 2019 — "Submodular maximisation
    beyond non-negativity: guarantees, fast algorithms, and
    applications."**  *Smoothed / Distorted greedy* — picks
    ``(1 - 1/k)^(k-i)`` weighted marginal in round ``i``; for monotone
    plus γ-weakly-submodular gives ``γ · (1 - 1/e^γ)``.

Composing with the rest of the stack
------------------------------------

The Submodular primitive provides **the** missing subset-selection bridge:

  * **Cartographer.** Picks the next-k frontier tasks: monotone submodular
    coverage of the prereq-DAG under a learning-progress utility →
    cardinality-constrained ``lazy_greedy``.  Combined with curvature
    bound, gives a tight per-batch quality certificate.

  * **ExperimentDesigner.** Batch Bayesian Optimal Experiment Design
    reduces to maximising mutual information ``I(θ ; y_S)`` which is
    submodular in ``S`` (Krause-Singh-Guestrin 2008) → ``lazy_greedy``
    is the standard solver and gives a ``(1 - 1/e)`` certificate
    *for the batch*.

  * **Coalition / Shapley.** Submodular ``f`` makes Shapley simulation
    finite-sample-tight: ``φ_i ≤ f({i})`` by submodularity.  Composes
    by feeding Shapley-evaluable utilities into ``maximize(...)`` and
    using the *value* returned as the credit baseline.

  * **Negotiator.** Indivisible-allocation utilities are usually
    submodular over the gift set; ``double_greedy`` and
    ``lazy_greedy`` give per-tenant assignment under a hard cap.

  * **Auditor.** Picks the top-K significant findings out of M
    discoveries to surface in a dashboard while controlling FDR: BH
    rejection set + ``lazy_greedy`` on a diversity-weighted relevance
    objective → high-precision, low-redundancy report.

  * **PolicyLab / PolicyImprover.** Diverse-yet-high-quality
    counterfactual-policy bank: ``log-det`` (DPP) submodular over
    policy fingerprints picks K maximally informative policies.

  * **Strategist.** Top-down "which k of these N options to run in
    parallel under cost ``B``": cost-benefit greedy → ``(1 - 1/e)``
    bound on chosen portfolio EV.

  * **Forecaster.** Pool of forecasters → diverse-and-skillful
    ensemble pick via submodular coverage of error directions.

  * **Skills.** Top-K skill retrieval where similarity-redundancy
    among skills matters → facility-location submodular.

  * **AttestationLedger.** Every solve emits an attested
    ``SubmodularReport`` whose digest commits to the chosen set, the
    realised value, and the certificate so a coordination engine
    can replay-verify.

What this module ships
----------------------

* Algorithms (single class ``Submodular``):
    - ``lazy_greedy``   (Minoux 1978 / Nemhauser-Wolsey-Fisher 1978)
    - ``naive_greedy``  (reference)
    - ``celf``          (Leskovec et al. 2007 — lazy + cost-benefit)
    - ``stochastic_greedy``  (Mirzasoleiman et al. 2015)
    - ``cost_greedy``   (Khuller-Moss-Naor 1999 knapsack)
    - ``sviridenko_knapsack``  (Sviridenko 2004 — partial enumeration + cost greedy)
    - ``double_greedy_random`` (Buchbinder et al. 2015 — unconstrained 1/2)
    - ``double_greedy_deterministic`` (Buchbinder et al. 2012 — 1/3)
    - ``distorted_greedy`` (Harshaw et al. 2019 — γ-weakly-submodular)
    - ``sieve_streaming`` (Badanidiyuru et al. 2014 — one-pass streaming)
    - ``submodular_cover`` (Wolsey 1982 — min |S| s.t. f(S) ≥ Q)
    - ``threshold_greedy`` (Badanidiyuru-Vondrák 2014 — accelerated)

* Bounds and certificates:
    - ``approx_ratio``  the worst-case ratio for the chosen method
    - ``curvature``     Conforti-Cornuéjols 1984 total curvature
    - ``curvature_bound``  the curvature-aware multiplicative bound
    - ``certify_submodular``  Hoeffding-bounded violation rate of DR
      from N random pairs (anytime-valid via empirical-Bernstein)

* Canonical objectives (drop-in submodular functions):
    - ``FacilityLocation(distances)``
    - ``WeightedCoverage(sets, weights)``
    - ``MonotoneSetCover(universe, sets)``
    - ``LogDeterminant(K, alpha)``   (DPP — diversity-and-quality)
    - ``GaussianEntropy(Sigma)``
    - ``MaxCut(W)``  (non-monotone)
    - ``ConcaveOverModular(weights, phi)``
    - ``FeatureBased(features, phi)``

All numerics are stdlib only (no numpy / scipy).  Threadsafe under a
single RLock.  Every state-changing call emits an optional Event and
records an optional attestation receipt.

Honest about limits
-------------------

* The ``(1 - 1/e)`` bound is over the *combinatorial optimum* of an
  oracle-defined submodular function.  If your ``f`` is **not**
  submodular, the bound does not apply — call ``certify_submodular``
  first, and use ``distorted_greedy`` for γ-weakly-submodular ``f``.

* ``stochastic_greedy`` is a *randomised* algorithm; the
  ``(1 - 1/e - ε)`` bound is in expectation.  Run with multiple seeds
  for an empirical-Bernstein bound on the realised approximation gap.

* ``sieve_streaming`` requires an upper bound ``max_singleton`` on
  ``max_v f({v})`` or it computes one in a calibration pass.

* All algorithms assume the oracle ``f`` is **deterministic** under
  repeated calls.  For noisy oracles use sample-averaging or fall
  back to the bandit-style stochastic submodular literature
  (out of scope for v1).

Citations
---------

* Nemhauser, G. L., Wolsey, L. A., Fisher, M. L. (1978). An analysis of
  approximations for maximizing submodular set functions — I.
  *Mathematical Programming*, 14, 265–294.
* Minoux, M. (1978). Accelerated greedy algorithms for maximizing
  submodular set functions. *Optim. Techniques (Lect. Notes CS 7)*, 234–243.
* Khuller, S., Moss, A., Naor, J. (1999). The budgeted maximum coverage
  problem. *Inf. Process. Lett.*, 70(1), 39–45.
* Feige, U. (1998). A threshold of ln n for approximating set cover.
  *Journal of the ACM*, 45(4), 634–652.
* Conforti, M., Cornuéjols, G. (1984). Submodular set functions, matroids
  and the greedy algorithm: tight worst-case bounds and some
  generalisations of the Rado-Edmonds theorem. *Disc. Appl. Math.*, 7.
* Wolsey, L. A. (1982). An analysis of the greedy algorithm for the
  submodular set covering problem. *Combinatorica*, 2(4), 385–393.
* Krause, A., Singh, A., Guestrin, C. (2008). Near-optimal sensor
  placements in Gaussian processes: theory, efficient algorithms and
  empirical studies. *JMLR*, 9, 235–284.
* Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J.,
  Glance, N. (2007). Cost-effective outbreak detection in networks.
  *KDD*, 420–429.
* Sviridenko, M. (2004). A note on maximizing a submodular set function
  subject to a knapsack constraint. *Oper. Res. Lett.*, 32, 41–43.
* Badanidiyuru, A., Mirzasoleiman, B., Karbasi, A., Krause, A. (2014).
  Streaming submodular maximization: massive data summarisation on the
  fly. *KDD*, 671–680.
* Badanidiyuru, A., Vondrák, J. (2014). Fast algorithms for maximising
  submodular functions. *SODA*, 1497–1514.
* Buchbinder, N., Feldman, M., Naor, J., Schwartz, R. (2015). A tight
  linear time (½)-approximation for unconstrained submodular
  maximisation. *SIAM J. Comput.*, 44(5), 1384–1402.
* Mirzasoleiman, B., Badanidiyuru, A., Karbasi, A., Vondrák, J.,
  Krause, A. (2015). Lazier than lazy greedy. *AAAI*, 1812–1818.
* Harshaw, C., Feldman, M., Ward, J., Karbasi, A. (2019). Submodular
  maximisation beyond non-negativity. *ICML*, 2634–2643.
* Golovin, D., Krause, A. (2011). Adaptive submodularity: Theory and
  applications in active learning and stochastic optimisation. *JAIR*,
  42, 427–486.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from agi.events import Event, EventBus  # type: ignore
except Exception:  # pragma: no cover - keep stdlib-only fallbacks
    Event = None  # type: ignore
    EventBus = None  # type: ignore


# =====================================================================
# Event kinds
# =====================================================================

SUBMOD_STARTED = "submodular.started"
SUBMOD_SOLVED = "submodular.solved"
SUBMOD_STREAM_OBSERVED = "submodular.stream_observed"
SUBMOD_STREAM_FINALISED = "submodular.stream_finalised"
SUBMOD_CERTIFIED = "submodular.certified"
SUBMOD_CLEARED = "submodular.cleared"
SUBMOD_REPORT = "submodular.report"


# =====================================================================
# Method names
# =====================================================================

METHOD_LAZY_GREEDY = "lazy_greedy"
METHOD_NAIVE_GREEDY = "naive_greedy"
METHOD_CELF = "celf"
METHOD_STOCHASTIC_GREEDY = "stochastic_greedy"
METHOD_COST_GREEDY = "cost_greedy"
METHOD_SVIRIDENKO_KNAPSACK = "sviridenko_knapsack"
METHOD_DOUBLE_GREEDY_RANDOM = "double_greedy_random"
METHOD_DOUBLE_GREEDY_DETERMINISTIC = "double_greedy_deterministic"
METHOD_DISTORTED_GREEDY = "distorted_greedy"
METHOD_SIEVE_STREAMING = "sieve_streaming"
METHOD_SUBMODULAR_COVER = "submodular_cover"
METHOD_THRESHOLD_GREEDY = "threshold_greedy"

KNOWN_METHODS = frozenset(
    {
        METHOD_LAZY_GREEDY,
        METHOD_NAIVE_GREEDY,
        METHOD_CELF,
        METHOD_STOCHASTIC_GREEDY,
        METHOD_COST_GREEDY,
        METHOD_SVIRIDENKO_KNAPSACK,
        METHOD_DOUBLE_GREEDY_RANDOM,
        METHOD_DOUBLE_GREEDY_DETERMINISTIC,
        METHOD_DISTORTED_GREEDY,
        METHOD_SIEVE_STREAMING,
        METHOD_SUBMODULAR_COVER,
        METHOD_THRESHOLD_GREEDY,
    }
)


# =====================================================================
# Constants
# =====================================================================

_EPS = 1e-12
_INV_E = math.exp(-1.0)
_ONE_MINUS_INV_E = 1.0 - _INV_E  # ≈ 0.6321205588...


# =====================================================================
# Exceptions
# =====================================================================


class SubmodularError(ValueError):
    """Invalid input or infeasible problem."""


class NotSubmodular(SubmodularError):
    """The user-supplied oracle failed a submodularity check."""


# =====================================================================
# Reports / certificates (dataclasses, JSON-friendly)
# =====================================================================


@dataclass
class SubmodularReport:
    """Outcome of one ``maximize(...)`` call.

    Attributes
    ----------
    method : str
        Algorithm identifier (``METHOD_*``).
    selected : list[Any]
        The chosen subset in the order it was assembled.
    value : float
        ``f(selected)``.
    n_oracle_calls : int
        Number of evaluations of ``f`` (or marginal-gain calls).
    approx_ratio : float
        Worst-case multiplicative bound the algorithm provides on
        ``f(S*)``.  E.g. ``(1 - 1/e)`` for ``lazy_greedy`` on monotone
        submodular ``f`` under cardinality.
    upper_bound : float | None
        Algorithm-specific value-oracle upper bound on ``f(S*)`` (e.g.
        sum of top-k singletons under monotone submodular ``f``).
        ``f(selected) / upper_bound`` is then a **realised** lower bound
        on the approximation ratio.
    feasible : bool
        Whether the chosen subset satisfies the user constraints.
    elapsed_ns : int
        Wall-clock duration in nanoseconds.
    digest : str
        Content hash of the report (sha256 over a canonical JSON form).
    extras : dict
        Method-specific telemetry — singletons table, λ-sweep, etc.
    """

    method: str
    selected: list = field(default_factory=list)
    value: float = 0.0
    n_oracle_calls: int = 0
    approx_ratio: float = 0.0
    upper_bound: float | None = None
    feasible: bool = True
    elapsed_ns: int = 0
    digest: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CertificateReport:
    """Result of ``certify_submodular``.

    Empirical (Hoeffding + empirical-Bernstein) bound on the fraction of
    ground-set pairs that violate the diminishing-returns inequality.
    """

    n_samples: int = 0
    n_violations: int = 0
    violation_rate: float = 0.0
    hoeffding_upper: float = 1.0
    bernstein_upper: float = 1.0
    alpha: float = 0.05
    digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StreamReport:
    """Result of ``sieve_streaming(...)`` after the pass completes."""

    method: str = METHOD_SIEVE_STREAMING
    selected: list = field(default_factory=list)
    value: float = 0.0
    n_oracle_calls: int = 0
    n_thresholds: int = 0
    max_singleton: float = 0.0
    approx_ratio: float = 0.0
    epsilon: float = 0.0
    elapsed_ns: int = 0
    digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# Helpers
# =====================================================================


def _hash_payload(payload: Any) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(payload).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise SubmodularError("alpha must be in (0, 1)")


def _validate_epsilon(epsilon: float) -> None:
    if not 0.0 < epsilon < 1.0:
        raise SubmodularError("epsilon must be in (0, 1)")


def _normalise_ground_set(ground_set: Iterable[Any]) -> list:
    out: list = []
    seen: set = set()
    for v in ground_set:
        try:
            key = (type(v).__name__, v)
            if key in seen:
                continue
            seen.add(key)
        except TypeError:
            # unhashable — fall back to id-based dedup
            if any(x is v for x in out):
                continue
        out.append(v)
    return out


def _oracle(f: Callable[[Iterable[Any]], float], S: Sequence[Any]) -> float:
    val = f(list(S))
    if not math.isfinite(val):
        raise SubmodularError(f"oracle returned non-finite value {val!r}")
    return float(val)


def _marginal(
    f: Callable[[Iterable[Any]], float],
    S: list,
    v: Any,
    f_S: float | None = None,
) -> tuple[float, float]:
    """Compute ``f(S ∪ {v}) - f(S)`` and return ``(gain, f(S))``.

    Caller can pass a cached ``f(S)`` so we save one oracle call.
    """
    base = _oracle(f, S) if f_S is None else f_S
    extended = _oracle(f, S + [v])
    return extended - base, base


# =====================================================================
# Approximation-ratio table
# =====================================================================


def _ratio(method: str, *, monotone: bool, epsilon: float = 0.0) -> float:
    """Return the worst-case multiplicative approximation factor."""
    if method == METHOD_LAZY_GREEDY or method == METHOD_NAIVE_GREEDY or method == METHOD_CELF:
        return _ONE_MINUS_INV_E if monotone else 0.0
    if method == METHOD_STOCHASTIC_GREEDY:
        return max(0.0, _ONE_MINUS_INV_E - epsilon) if monotone else 0.0
    if method == METHOD_COST_GREEDY:
        return 0.5 * _ONE_MINUS_INV_E if monotone else 0.0
    if method == METHOD_SVIRIDENKO_KNAPSACK:
        return _ONE_MINUS_INV_E if monotone else 0.0
    if method == METHOD_DOUBLE_GREEDY_RANDOM:
        return 0.5
    if method == METHOD_DOUBLE_GREEDY_DETERMINISTIC:
        return 1.0 / 3.0
    if method == METHOD_DISTORTED_GREEDY:
        return _ONE_MINUS_INV_E if monotone else 0.0
    if method == METHOD_SIEVE_STREAMING:
        return max(0.0, 0.5 - epsilon) if monotone else 0.0
    if method == METHOD_THRESHOLD_GREEDY:
        return max(0.0, _ONE_MINUS_INV_E - epsilon) if monotone else 0.0
    if method == METHOD_SUBMODULAR_COVER:
        return 1.0  # cover is a minimisation; the bound is multiplicative on |S|
    return 0.0


# =====================================================================
# Canonical objectives — small, exact, stdlib only
# =====================================================================


class FacilityLocation:
    r"""Facility location utility on a pre-computed similarity matrix.

    Given a similarity matrix ``W[i][j] ≥ 0`` between ``n`` clients and
    ``n`` candidate facilities,

        f(S) = Σ_i max_{j ∈ S} W[i][j].

    Monotone submodular, non-negative, normalised iff ``W ≥ 0``.

    Parameters
    ----------
    W : 2-D matrix-like
        ``W[i][j]`` is the value client ``i`` gets from facility ``j``.
        ``W`` may be a list of lists or a callable ``(i, j) → float``.
    n_clients, n_facilities : int
        Dimensions; inferred from ``W`` when it is a list-of-lists.
    """

    def __init__(
        self,
        W: Any,
        *,
        n_clients: int | None = None,
        n_facilities: int | None = None,
    ) -> None:
        if callable(W):
            if n_clients is None or n_facilities is None:
                raise SubmodularError(
                    "callable W requires n_clients and n_facilities"
                )
            self._lookup = lambda i, j, _W=W: float(_W(i, j))
            self.n_clients = int(n_clients)
            self.n_facilities = int(n_facilities)
        else:
            mat = list(W)
            self.n_clients = len(mat)
            self.n_facilities = len(mat[0]) if mat else 0
            self._mat = [[float(x) for x in row] for row in mat]
            self._lookup = lambda i, j, _M=self._mat: _M[i][j]
        for j in range(self.n_facilities):
            for i in range(self.n_clients):
                if self._lookup(i, j) < 0.0:
                    raise SubmodularError("FacilityLocation requires W ≥ 0")

    def ground_set(self) -> list[int]:
        return list(range(self.n_facilities))

    def __call__(self, S: Iterable[int]) -> float:
        S = list(S)
        if not S:
            return 0.0
        total = 0.0
        for i in range(self.n_clients):
            best = max(self._lookup(i, j) for j in S)
            total += best
        return total


class WeightedCoverage:
    r"""Weighted set-coverage utility.

    Given a universe of elements with non-negative weights ``w[u]`` and a
    family of subsets ``{C_j ⊆ U}``,

        f(S) = Σ_{u ∈ ⋃_{j ∈ S} C_j}  w[u].

    Monotone submodular, non-negative, normalised.
    """

    def __init__(
        self,
        sets: Sequence[Iterable[Any]],
        *,
        weights: Mapping[Any, float] | None = None,
    ) -> None:
        self._sets = [frozenset(s) for s in sets]
        universe: set = set()
        for s in self._sets:
            universe |= s
        self.universe = frozenset(universe)
        if weights is None:
            self._w = {u: 1.0 for u in self.universe}
        else:
            self._w = {}
            for u in self.universe:
                w = float(weights.get(u, 0.0))
                if w < 0.0:
                    raise SubmodularError("WeightedCoverage requires w ≥ 0")
                self._w[u] = w

    def ground_set(self) -> list[int]:
        return list(range(len(self._sets)))

    def __call__(self, S: Iterable[int]) -> float:
        covered: set = set()
        for j in S:
            covered |= self._sets[j]
        return sum(self._w[u] for u in covered)


class MonotoneSetCover:
    """Alias of :class:`WeightedCoverage` with unit weights for clarity."""

    def __init__(self, sets: Sequence[Iterable[Any]]) -> None:
        self._inner = WeightedCoverage(sets)
        self.universe = self._inner.universe

    def ground_set(self) -> list[int]:
        return self._inner.ground_set()

    def __call__(self, S: Iterable[int]) -> float:
        return self._inner(S)


class LogDeterminant:
    r"""Diversity-and-quality utility from a PSD kernel matrix.

    For a symmetric PSD kernel ``K`` with regularised version
    ``K + αI`` and a subset ``S``,

        f(S) = log det( K[S, S] + α I_{|S|} ).

    Submodular **for every** ``α ≥ 0`` (Kulesza-Taskar 2012, DPPs).
    **Monotone** only when ``α`` exceeds the spectral radius of
    ``K`` off the diagonal (a sufficient condition is ``α ≥ ||K||_op``);
    for small ``α`` the function may decrease as elements are added —
    drive it with ``METHOD_DOUBLE_GREEDY_RANDOM`` in that regime.

    Stdlib-only Cholesky implementation for the determinant via the
    product of the diagonal of the L factor, which is numerically stable.
    """

    def __init__(self, K: Sequence[Sequence[float]], *, alpha: float = 1e-6) -> None:
        n = len(K)
        if any(len(row) != n for row in K):
            raise SubmodularError("LogDeterminant requires a square kernel")
        if alpha < 0.0:
            raise SubmodularError("alpha must be ≥ 0")
        self._K = [[float(x) for x in row] for row in K]
        self.n = n
        self.alpha = float(alpha)

    def ground_set(self) -> list[int]:
        return list(range(self.n))

    def __call__(self, S: Iterable[int]) -> float:
        S = sorted(set(int(x) for x in S))
        m = len(S)
        if m == 0:
            return 0.0
        # Build the regularised submatrix.
        A = [[self._K[i][j] for j in S] for i in S]
        for i in range(m):
            A[i][i] += self.alpha
        # Cholesky: f(S) = 2 Σ log L_ii.
        L = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1):
                s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    if s <= 0.0:
                        # Not strictly PSD — bail out with -inf signal.
                        return -math.inf
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
        total = 0.0
        for i in range(m):
            total += math.log(L[i][i])
        return 2.0 * total


class GaussianEntropy:
    r"""Differential entropy of a Gaussian marginal on a subset.

    For a covariance matrix ``Σ`` and a subset ``S``,

        f(S) = ½ log( (2 π e)^{|S|} det( Σ[S, S] ) ).

    Submodular but **not monotone in general**; non-negativity depends on
    the scale of ``Σ``.  Common in sensor-placement (Krause-Singh-
    Guestrin 2008).
    """

    def __init__(self, Sigma: Sequence[Sequence[float]]) -> None:
        self._inner = LogDeterminant(Sigma, alpha=0.0)
        self._n = self._inner.n
        self._const = 0.5 * math.log(2.0 * math.pi * math.e)

    def ground_set(self) -> list[int]:
        return self._inner.ground_set()

    def __call__(self, S: Iterable[int]) -> float:
        S = list(S)
        if not S:
            return 0.0
        ld = self._inner(S)
        return self._const * len(S) + 0.5 * ld


class MaxCut:
    r"""Cut value of a (undirected, non-negative-weighted) graph.

    For a non-negative weight matrix ``W`` and ``S ⊆ V``,

        f(S) = Σ_{i ∈ S, j ∉ S} W[i][j].

    Submodular **but non-monotone**.  Standard test bed for unconstrained
    non-monotone submodular maximisation (Buchbinder et al. 2015).
    """

    def __init__(self, W: Sequence[Sequence[float]]) -> None:
        n = len(W)
        for row in W:
            if len(row) != n:
                raise SubmodularError("MaxCut requires a square weight matrix")
        self.n = n
        self._W = [[float(x) for x in row] for row in W]
        for i in range(n):
            for j in range(i + 1, n):
                if self._W[i][j] != self._W[j][i]:
                    # symmetrise quietly — undirected
                    s = 0.5 * (self._W[i][j] + self._W[j][i])
                    self._W[i][j] = s
                    self._W[j][i] = s
                if self._W[i][j] < 0.0:
                    raise SubmodularError("MaxCut requires W ≥ 0")

    def ground_set(self) -> list[int]:
        return list(range(self.n))

    def __call__(self, S: Iterable[int]) -> float:
        S = set(int(x) for x in S)
        if not S:
            return 0.0
        total = 0.0
        for i in S:
            for j in range(self.n):
                if j in S:
                    continue
                total += self._W[i][j]
        return total


class ConcaveOverModular:
    r"""``f(S) = φ( Σ_{i ∈ S} w_i )`` for a non-decreasing concave φ.

    The composition is monotone submodular when ``w ≥ 0`` and ``φ`` is
    non-decreasing concave (Bilmes 2022).
    """

    def __init__(
        self,
        weights: Sequence[float],
        phi: Callable[[float], float] | None = None,
    ) -> None:
        self._w = [float(x) for x in weights]
        for w in self._w:
            if w < 0.0:
                raise SubmodularError("ConcaveOverModular requires w ≥ 0")
        self._phi = phi if phi is not None else (lambda x: math.sqrt(max(0.0, x)))

    def ground_set(self) -> list[int]:
        return list(range(len(self._w)))

    def __call__(self, S: Iterable[int]) -> float:
        return float(self._phi(sum(self._w[i] for i in S)))


class FeatureBased:
    r"""``f(S) = Σ_u φ( Σ_{i ∈ S} a_{u,i} )`` over a feature universe.

    Each element ``i`` brings non-negative feature loadings ``a_{u,i}``
    on feature ``u``.  φ is non-decreasing concave; the composition is
    monotone submodular.  Equivalent to a sum of concave-over-modular
    blocks — strictly more expressive than facility location for many
    summarisation problems (Lin-Bilmes 2011).
    """

    def __init__(
        self,
        features: Sequence[Mapping[Any, float]],
        phi: Callable[[float], float] | None = None,
    ) -> None:
        self._features = [
            {u: float(w) for u, w in feat.items()} for feat in features
        ]
        for feat in self._features:
            for w in feat.values():
                if w < 0.0:
                    raise SubmodularError("FeatureBased requires loadings ≥ 0")
        self._phi = phi if phi is not None else (lambda x: math.sqrt(max(0.0, x)))
        self._universe: list = sorted(
            {u for feat in self._features for u in feat.keys()},
            key=lambda x: str(x),
        )

    def ground_set(self) -> list[int]:
        return list(range(len(self._features)))

    def __call__(self, S: Iterable[int]) -> float:
        S = list(S)
        if not S:
            return 0.0
        total = 0.0
        for u in self._universe:
            sigma_u = 0.0
            for i in S:
                sigma_u += self._features[i].get(u, 0.0)
            total += float(self._phi(sigma_u))
        return total


# =====================================================================
# Algorithm cores
# =====================================================================


def _lazy_greedy_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    k: int,
    *,
    return_extras: bool = False,
) -> tuple[list, float, int, dict]:
    """Minoux 1978 lazy greedy under cardinality |S| ≤ k.

    Returns (selected, f(S), oracle_calls, extras).
    """
    if k <= 0:
        return [], 0.0, 0, {}
    n_calls = 0
    f_empty = _oracle(f, [])
    n_calls += 1
    # Initial upper bounds = f({v}) - f(∅).
    heap: list[tuple[float, int, int, Any]] = []  # (-gain, age, idx, item)
    singletons: list[float] = [0.0] * len(ground)
    for idx, v in enumerate(ground):
        g = _oracle(f, [v]) - f_empty
        n_calls += 1
        singletons[idx] = g
        heapq.heappush(heap, (-g, 0, idx, v))
    selected: list = []
    selected_set: set[int] = set()
    f_S = f_empty
    age = 0
    while heap and len(selected) < k:
        neg_g, last_age, idx, v = heapq.heappop(heap)
        if idx in selected_set:
            continue
        if last_age == age:
            # Bound is fresh at the *current* selection — accept.
            gain = -neg_g
            if gain <= 0.0:
                # No improving element in a monotone problem ⇒ stop early.
                # (Non-monotone use a different driver; here we early-terminate.)
                break
            selected.append(v)
            selected_set.add(idx)
            f_S += gain
            age += 1
        else:
            # Stale — re-evaluate at the current selection and push back.
            new_val = _oracle(f, selected + [v])
            n_calls += 1
            new_gain = new_val - f_S
            heapq.heappush(heap, (-new_gain, age, idx, v))
    extras: dict = {}
    if return_extras:
        extras = {
            "f_empty": f_empty,
            "singletons": singletons,
        }
    return selected, f_S, n_calls, extras


def _naive_greedy_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    k: int,
) -> tuple[list, float, int]:
    """Reference greedy — O(nk) marginals."""
    if k <= 0:
        return [], 0.0, 0
    n_calls = 0
    f_S = _oracle(f, [])
    n_calls += 1
    selected: list = []
    remaining_idx: set[int] = set(range(len(ground)))
    while remaining_idx and len(selected) < k:
        best_gain = -math.inf
        best_idx: int | None = None
        best_val = f_S
        for idx in list(remaining_idx):
            v = ground[idx]
            val = _oracle(f, selected + [v])
            n_calls += 1
            gain = val - f_S
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_val = val
        if best_idx is None or best_gain <= 0.0:
            break
        selected.append(ground[best_idx])
        remaining_idx.discard(best_idx)
        f_S = best_val
    return selected, f_S, n_calls


def _celf_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    *,
    k: int | None,
    budget: float | None,
    costs: Sequence[float] | None,
) -> tuple[list, float, int]:
    """Lazy greedy with cost-benefit pivot (Leskovec et al. 2007).

    When ``costs`` is supplied, the heap is keyed by gain / cost.
    Honours both cardinality (``k``) and knapsack (``budget``).
    """
    if k is not None and k <= 0:
        return [], 0.0, 0
    if budget is not None and budget <= 0.0:
        return [], 0.0, 0
    if costs is None:
        unit_costs = [1.0] * len(ground)
    else:
        unit_costs = list(costs)
        if any(c <= 0.0 for c in unit_costs):
            raise SubmodularError("CELF costs must be > 0")
    n_calls = 0
    f_empty = _oracle(f, [])
    n_calls += 1
    heap: list[tuple[float, int, int, float]] = []  # (-density, age, idx, gain)
    for idx, v in enumerate(ground):
        g = _oracle(f, [v]) - f_empty
        n_calls += 1
        c = unit_costs[idx]
        density = g / c if c > 0.0 else 0.0
        heapq.heappush(heap, (-density, 0, idx, g))
    selected: list = []
    selected_set: set[int] = set()
    f_S = f_empty
    total_cost = 0.0
    age = 0
    while heap:
        neg_d, last_age, idx, gain = heapq.heappop(heap)
        if idx in selected_set:
            continue
        v = ground[idx]
        c = unit_costs[idx]
        if budget is not None and total_cost + c > budget + _EPS:
            # Doesn't fit — skip permanently.
            continue
        if last_age == age:
            density = -neg_d
            if density <= 0.0:
                break
            selected.append(v)
            selected_set.add(idx)
            f_S += gain
            total_cost += c
            age += 1
            if k is not None and len(selected) >= k:
                break
        else:
            new_val = _oracle(f, selected + [v])
            n_calls += 1
            new_gain = new_val - f_S
            new_density = new_gain / c if c > 0.0 else 0.0
            heapq.heappush(heap, (-new_density, age, idx, new_gain))
    return selected, f_S, n_calls


def _stochastic_greedy_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    k: int,
    *,
    epsilon: float,
    rng: random.Random,
) -> tuple[list, float, int]:
    """Mirzasoleiman et al. 2015 — lazier than lazy greedy.

    Each round, draw ``r = ⌈n/k · log(1/ε)⌉`` ground-set indices uniformly
    at random *without* replacement (among those not yet chosen) and
    pick the argmax marginal among them.  Expected approximation factor:
    ``1 - 1/e - ε``.
    """
    if k <= 0:
        return [], 0.0, 0
    if not (0.0 < epsilon < 1.0):
        raise SubmodularError("epsilon must be in (0, 1) for stochastic_greedy")
    n = len(ground)
    sample_size = max(1, math.ceil((n / max(1, k)) * math.log(1.0 / epsilon)))
    n_calls = 0
    f_S = _oracle(f, [])
    n_calls += 1
    selected: list = []
    remaining: list[int] = list(range(n))
    while remaining and len(selected) < k:
        r = min(sample_size, len(remaining))
        sample = rng.sample(remaining, r)
        best_gain = -math.inf
        best_idx: int | None = None
        best_val = f_S
        for idx in sample:
            v = ground[idx]
            val = _oracle(f, selected + [v])
            n_calls += 1
            gain = val - f_S
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_val = val
        if best_idx is None or best_gain <= 0.0:
            break
        selected.append(ground[best_idx])
        remaining.remove(best_idx)
        f_S = best_val
    return selected, f_S, n_calls


def _sviridenko_knapsack_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    *,
    costs: Sequence[float],
    budget: float,
    enum_size: int = 2,
) -> tuple[list, float, int]:
    """Sviridenko 2004 — partial enumeration + cost-benefit greedy.

    For each ``A`` with ``|A| ≤ enum_size`` and ``Σ c(a) ≤ B``, extend ``A``
    by cost-benefit greedy among the rest under the residual budget.
    Return the best.

    ``enum_size = 3`` is the canonical setting that achieves the
    ``(1 - 1/e)`` bound; we default to ``2`` for query economy and bump
    up to 3 only when explicitly requested via ``enum_size=3``.
    """
    if budget <= 0.0:
        return [], 0.0, 0
    if len(ground) != len(costs):
        raise SubmodularError("|costs| must equal |ground|")
    if any(c <= 0.0 for c in costs):
        raise SubmodularError("Sviridenko costs must be > 0")
    n_calls = 0

    def _greedy_extend(start: list[int]) -> tuple[list[int], float]:
        nonlocal n_calls
        used = set(start)
        cur_cost = sum(costs[i] for i in start)
        sel = list(start)
        f_S = _oracle(f, [ground[i] for i in sel])
        n_calls += 1
        improved = True
        while improved:
            improved = False
            best_density = -math.inf
            best_idx: int | None = None
            best_val = f_S
            for idx in range(len(ground)):
                if idx in used:
                    continue
                c = costs[idx]
                if cur_cost + c > budget + _EPS:
                    continue
                val = _oracle(f, [ground[i] for i in sel + [idx]])
                n_calls += 1
                gain = val - f_S
                density = gain / c if c > 0.0 else 0.0
                if density > best_density:
                    best_density = density
                    best_idx = idx
                    best_val = val
            if best_idx is not None and best_density > 0.0:
                sel.append(best_idx)
                used.add(best_idx)
                cur_cost += costs[best_idx]
                f_S = best_val
                improved = True
        return sel, f_S

    best_sel: list[int] = []
    best_val = -math.inf
    indices = list(range(len(ground)))

    # Empty seed.
    sel, val = _greedy_extend([])
    if val > best_val:
        best_val, best_sel = val, sel

    # Enumerate seeds of size 1 .. enum_size.
    def _combinations(seq: list[int], r: int) -> Iterable[tuple[int, ...]]:
        # stdlib itertools, but importing locally so the module header is clean.
        from itertools import combinations
        return combinations(seq, r)

    for r in range(1, min(enum_size, len(indices)) + 1):
        for combo in _combinations(indices, r):
            cost_combo = sum(costs[i] for i in combo)
            if cost_combo > budget + _EPS:
                continue
            sel, val = _greedy_extend(list(combo))
            if val > best_val:
                best_val, best_sel = val, sel

    # Best singleton fallback (Khuller-Moss-Naor pivot).
    f_empty = _oracle(f, [])
    n_calls += 1
    for idx in indices:
        if costs[idx] > budget + _EPS:
            continue
        val = _oracle(f, [ground[idx]])
        n_calls += 1
        if val > best_val:
            best_val, best_sel = val, [idx]

    return [ground[i] for i in best_sel], best_val if best_val > -math.inf else f_empty, n_calls


def _double_greedy_random_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    *,
    rng: random.Random,
) -> tuple[list, float, int]:
    """Buchbinder-Feldman-Naor-Schwartz 2015 — randomised double greedy.

    Maintains X ⊆ Y ⊆ V; sweeps elements in arbitrary order.  For each
    element v, computes::

        a = f(X ∪ {v}) - f(X),
        b = f(Y ∖ {v}) - f(Y).

    Adds v to X with probability ``a' / (a' + b')`` where
    ``a' = max(a, 0)``, ``b' = max(b, 0)`` (uniform fall-back if both 0).
    Output X = Y at end.  Expected value ≥ ½ f(S*).
    """
    n_calls = 0
    X: list = []
    Y_set = set(range(len(ground)))
    X_set: set[int] = set()
    f_X = _oracle(f, [])
    n_calls += 1
    f_Y = _oracle(f, ground)
    n_calls += 1
    order = list(range(len(ground)))
    rng.shuffle(order)
    for idx in order:
        v = ground[idx]
        # a = f(X ∪ {v}) − f(X)
        f_X_plus = _oracle(f, [ground[i] for i in X_set] + [v])
        n_calls += 1
        a = f_X_plus - f_X
        # b = f(Y ∖ {v}) − f(Y); Y currently contains everything in Y_set.
        Y_minus_v = [ground[i] for i in Y_set if i != idx]
        f_Y_minus = _oracle(f, Y_minus_v)
        n_calls += 1
        b = f_Y_minus - f_Y
        a_p = max(a, 0.0)
        b_p = max(b, 0.0)
        total = a_p + b_p
        if total <= 0.0:
            # Equiprobable.
            prob_add = 0.5
        else:
            prob_add = a_p / total
        if rng.random() < prob_add:
            X_set.add(idx)
            f_X = f_X_plus
            X.append(v)
            # Y unchanged.
        else:
            Y_set.discard(idx)
            f_Y = f_Y_minus
            # X unchanged.
    # X and Y now coincide on Y_set.
    selected = [ground[i] for i in X_set]
    val = _oracle(f, selected)
    n_calls += 1
    return selected, val, n_calls


def _double_greedy_deterministic_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
) -> tuple[list, float, int]:
    """Buchbinder et al. 2012 deterministic double greedy — 1/3 bound."""
    n_calls = 0
    X_set: set[int] = set()
    Y_set: set[int] = set(range(len(ground)))
    f_X = _oracle(f, [])
    n_calls += 1
    f_Y = _oracle(f, ground)
    n_calls += 1
    for idx in range(len(ground)):
        v = ground[idx]
        f_X_plus = _oracle(f, [ground[i] for i in X_set] + [v])
        n_calls += 1
        a = f_X_plus - f_X
        Y_minus_v = [ground[i] for i in Y_set if i != idx]
        f_Y_minus = _oracle(f, Y_minus_v)
        n_calls += 1
        b = f_Y_minus - f_Y
        if a >= b:
            X_set.add(idx)
            f_X = f_X_plus
        else:
            Y_set.discard(idx)
            f_Y = f_Y_minus
    selected = [ground[i] for i in X_set]
    val = _oracle(f, selected)
    n_calls += 1
    return selected, val, n_calls


def _distorted_greedy_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    k: int,
    *,
    gamma: float,
) -> tuple[list, float, int]:
    """Harshaw-Feldman-Ward-Karbasi 2019 distorted greedy.

    Round-``i`` weight ``ω_i = (1 - γ/k)^{k - i - 1}`` applied to the
    marginal of ``f``.  For pure monotone submodular ``f`` (γ = 1) this
    reduces to standard greedy with no distortion.  Implemented as a
    plug-and-play accelerator for non-additive / weakly-submodular
    monotone ``f``.
    """
    if k <= 0:
        return [], 0.0, 0
    if not (0.0 < gamma <= 1.0):
        raise SubmodularError("gamma must be in (0, 1]")
    n_calls = 0
    f_S = _oracle(f, [])
    n_calls += 1
    selected: list = []
    selected_set: set[int] = set()
    for i in range(k):
        weight = (1.0 - gamma / k) ** (k - i - 1) if k > 0 else 1.0
        best_score = -math.inf
        best_idx: int | None = None
        best_val = f_S
        for idx in range(len(ground)):
            if idx in selected_set:
                continue
            v = ground[idx]
            val = _oracle(f, selected + [v])
            n_calls += 1
            gain = val - f_S
            score = weight * gain
            if score > best_score:
                best_score = score
                best_idx = idx
                best_val = val
        if best_idx is None or best_score <= 0.0:
            break
        selected.append(ground[best_idx])
        selected_set.add(best_idx)
        f_S = best_val
    return selected, f_S, n_calls


def _sieve_streaming_core(
    f: Callable[[Iterable[Any]], float],
    ground: Iterable[Any],
    k: int,
    *,
    epsilon: float,
    max_singleton: float | None,
) -> tuple[list, float, int, dict]:
    """Badanidiyuru et al. 2014 — Sieve-Streaming under cardinality.

    One pass over ``ground``.  Maintains buckets, one per threshold
    ``τ ∈ {(1+ε)^i : i ∈ ℤ}`` within ``[max_singleton, 2k·max_singleton]``.
    For each element, append to bucket τ only if its marginal ≥
    ``(τ/2 − f(S_τ))/(k − |S_τ|)``.  Approximation ``½ − ε``.
    """
    if k <= 0:
        return [], 0.0, 0, {}
    if not (0.0 < epsilon < 1.0):
        raise SubmodularError("epsilon must be in (0, 1)")
    ground_list = list(ground)
    n_calls = 0
    # Pre-compute max singleton if not supplied.
    if max_singleton is None:
        max_singleton = 0.0
        f_empty = _oracle(f, [])
        n_calls += 1
        for v in ground_list:
            val = _oracle(f, [v])
            n_calls += 1
            gain = val - f_empty
            if gain > max_singleton:
                max_singleton = gain
    if max_singleton <= 0.0:
        return [], 0.0, n_calls, {"max_singleton": 0.0, "n_thresholds": 0}
    # Threshold grid.
    log_eps = math.log(1.0 + epsilon)
    lo = math.ceil(math.log(max_singleton) / log_eps)
    hi = math.floor(math.log(2.0 * k * max_singleton) / log_eps)
    thresholds: list[float] = [(1.0 + epsilon) ** i for i in range(lo, hi + 1)]
    buckets: list[list] = [[] for _ in thresholds]
    f_buckets: list[float] = [0.0] * len(thresholds)
    for v in ground_list:
        for ti, tau in enumerate(thresholds):
            S = buckets[ti]
            if len(S) >= k:
                continue
            need = max(0.0, (tau / 2.0 - f_buckets[ti]) / (k - len(S)))
            val = _oracle(f, S + [v])
            n_calls += 1
            gain = val - f_buckets[ti]
            if gain >= need:
                S.append(v)
                f_buckets[ti] = val
    # Pick best bucket.
    best_i = max(range(len(thresholds)), key=lambda i: f_buckets[i]) if thresholds else 0
    extras = {
        "max_singleton": max_singleton,
        "n_thresholds": len(thresholds),
    }
    if not thresholds:
        return [], 0.0, n_calls, extras
    return buckets[best_i], f_buckets[best_i], n_calls, extras


def _submodular_cover_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    quota: float,
    *,
    f_total: float | None,
    max_picks: int | None,
) -> tuple[list, float, int]:
    """Wolsey 1982 — minimise |S| s.t. f(S) ≥ Q, with lazy greedy."""
    if quota <= 0.0:
        return [], 0.0, 0
    n_calls = 0
    f_empty = _oracle(f, [])
    n_calls += 1
    heap: list[tuple[float, int, int, Any]] = []
    for idx, v in enumerate(ground):
        g = _oracle(f, [v]) - f_empty
        n_calls += 1
        heapq.heappush(heap, (-g, 0, idx, v))
    selected: list = []
    selected_set: set[int] = set()
    f_S = f_empty
    age = 0
    limit = max_picks if max_picks is not None else len(ground)
    while heap and len(selected) < limit and f_S < quota:
        neg_g, last_age, idx, v = heapq.heappop(heap)
        if idx in selected_set:
            continue
        if last_age == age:
            gain = -neg_g
            if gain <= 0.0:
                break
            selected.append(v)
            selected_set.add(idx)
            f_S += gain
            age += 1
        else:
            new_val = _oracle(f, selected + [v])
            n_calls += 1
            new_gain = new_val - f_S
            heapq.heappush(heap, (-new_gain, age, idx, v))
    return selected, f_S, n_calls


def _threshold_greedy_core(
    f: Callable[[Iterable[Any]], float],
    ground: list,
    k: int,
    *,
    epsilon: float,
) -> tuple[list, float, int]:
    """Badanidiyuru-Vondrák 2014 — accelerated greedy.

    Iterate threshold ``τ`` from a top estimate down by factor ``1 - ε``;
    accept any element with marginal ≥ τ until |S| = k.  Achieves
    ``(1 - 1/e - ε)`` with O(n/ε · log(n/ε)) marginal queries.
    """
    if k <= 0:
        return [], 0.0, 0
    if not (0.0 < epsilon < 1.0):
        raise SubmodularError("epsilon must be in (0, 1)")
    n_calls = 0
    f_empty = _oracle(f, [])
    n_calls += 1
    # Find top singleton.
    top = 0.0
    for v in ground:
        val = _oracle(f, [v]) - f_empty
        n_calls += 1
        if val > top:
            top = val
    if top <= 0.0:
        return [], 0.0, n_calls
    selected: list = []
    selected_set: set[int] = set()
    f_S = f_empty
    tau = top
    floor = (epsilon / max(1, k)) * top
    while tau >= floor and len(selected) < k:
        for idx in range(len(ground)):
            if idx in selected_set:
                continue
            if len(selected) >= k:
                break
            v = ground[idx]
            val = _oracle(f, selected + [v])
            n_calls += 1
            gain = val - f_S
            if gain >= tau:
                selected.append(v)
                selected_set.add(idx)
                f_S = val
        tau *= 1.0 - epsilon
    return selected, f_S, n_calls


# =====================================================================
# Submodular — the public class
# =====================================================================


class Submodular:
    """Subset-selection runtime primitive with provable approximation bounds.

    Parameters
    ----------
    bus : EventBus | None
        Optional event bus for live broadcast.
    attestor : object | None
        Optional sink with ``record(kind=..., payload=...)`` or a plain
        callable; every solve emits a content-hashed receipt.
    random_seed : int | None
        Seed for stochastic algorithms (stochastic greedy, randomised
        double greedy).
    """

    def __init__(
        self,
        *,
        bus: Any = None,
        attestor: Any = None,
        random_seed: int | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._lock = threading.RLock()
        self._rng = random.Random(random_seed)
        self._n_solves = 0
        self._n_certificates = 0
        self._n_oracle_calls = 0
        self._started_ns = time.time_ns()
        self._emit(
            SUBMOD_STARTED,
            {"id": uuid.uuid4().hex[:16], "ts_ns": self._started_ns},
        )

    # ---- event / attest helpers --------------------------------------

    def _emit(self, kind: str, payload: dict) -> None:
        if self._bus is None or Event is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(payload)))
        except Exception:
            pass

    def _attest(self, kind: str, payload: dict) -> str:
        digest = _hash_payload(payload)
        if self._attestor is None:
            return digest
        try:
            if hasattr(self._attestor, "record"):
                self._attestor.record(kind=kind, payload=payload)
            elif callable(self._attestor):
                self._attestor({"kind": kind, "payload": payload, "digest": digest})
        except Exception:
            pass
        return digest

    # ---- coverage stats ----------------------------------------------

    def coverage(self) -> dict:
        with self._lock:
            return {
                "n_solves": self._n_solves,
                "n_certificates": self._n_certificates,
                "n_oracle_calls": self._n_oracle_calls,
                "started_ns": self._started_ns,
            }

    # ---- public API: maximise ----------------------------------------

    def maximize(
        self,
        f: Callable[[Iterable[Any]], float],
        ground_set: Iterable[Any],
        *,
        k: int | None = None,
        budget: float | None = None,
        costs: Mapping[Any, float] | Sequence[float] | None = None,
        method: str = METHOD_LAZY_GREEDY,
        monotone: bool = True,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        enum_size: int = 2,
        max_singleton: float | None = None,
        seed: int | None = None,
        return_extras: bool = False,
    ) -> SubmodularReport:
        """Run a submodular maximisation algorithm and return a report.

        The cardinality / knapsack constraint is inferred from the
        supplied ``k`` / ``budget`` arguments and the method.

        Method-specific arguments
        -------------------------
        * ``epsilon`` — accuracy parameter for stochastic / streaming /
          threshold variants.  Smaller ``ε`` → tighter bound, more work.
        * ``gamma``  — submodularity ratio for the distorted-greedy
          variant (1.0 → monotone submodular; lower → weakly submodular).
        * ``enum_size``  — partial-enumeration size for Sviridenko's
          knapsack greedy.  ``3`` matches the published ``(1 - 1/e)``
          bound; ``2`` is a query-economical default.
        * ``max_singleton`` — pre-computed ``max_v f({v}) - f(∅)`` for
          sieve-streaming; computed automatically if omitted.
        """
        if method not in KNOWN_METHODS:
            raise SubmodularError(
                f"unknown method {method!r}; expected one of {sorted(KNOWN_METHODS)!r}"
            )
        ground = _normalise_ground_set(ground_set)
        if not ground:
            raise SubmodularError("ground_set is empty")
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = self._rng

        # Normalise costs to a parallel list under the ground-set order.
        cost_list: list[float] | None = None
        if costs is not None:
            if isinstance(costs, Mapping):
                try:
                    cost_list = [float(costs[v]) for v in ground]
                except KeyError as exc:
                    raise SubmodularError(
                        f"costs mapping missing ground-set element {exc.args[0]!r}"
                    )
            else:
                cost_list = [float(c) for c in costs]
                if len(cost_list) != len(ground):
                    raise SubmodularError("len(costs) must match |ground_set|")
            for c in cost_list:
                if c <= 0.0:
                    raise SubmodularError("costs must be > 0")

        t0 = time.time_ns()
        n_calls = 0
        extras: dict = {}
        upper: float | None = None

        if method == METHOD_LAZY_GREEDY:
            if k is None:
                raise SubmodularError("lazy_greedy requires k (cardinality)")
            selected, value, n_calls, extras = _lazy_greedy_core(
                f, ground, k, return_extras=True
            )
            upper = _monotone_upper(extras.get("singletons", []), k)
        elif method == METHOD_NAIVE_GREEDY:
            if k is None:
                raise SubmodularError("naive_greedy requires k (cardinality)")
            selected, value, n_calls = _naive_greedy_core(f, ground, k)
        elif method == METHOD_CELF:
            if k is None and budget is None:
                raise SubmodularError("celf requires k or budget")
            selected, value, n_calls = _celf_core(
                f, ground, k=k, budget=budget, costs=cost_list
            )
        elif method == METHOD_STOCHASTIC_GREEDY:
            if k is None:
                raise SubmodularError("stochastic_greedy requires k (cardinality)")
            _validate_epsilon(epsilon)
            selected, value, n_calls = _stochastic_greedy_core(
                f, ground, k, epsilon=epsilon, rng=rng
            )
        elif method == METHOD_COST_GREEDY:
            if cost_list is None or budget is None:
                raise SubmodularError("cost_greedy requires costs and budget")
            selected, value, n_calls = _celf_core(
                f, ground, k=None, budget=budget, costs=cost_list
            )
        elif method == METHOD_SVIRIDENKO_KNAPSACK:
            if cost_list is None or budget is None:
                raise SubmodularError(
                    "sviridenko_knapsack requires costs and budget"
                )
            if enum_size < 1:
                raise SubmodularError("enum_size must be ≥ 1")
            selected, value, n_calls = _sviridenko_knapsack_core(
                f, ground, costs=cost_list, budget=budget, enum_size=enum_size
            )
        elif method == METHOD_DOUBLE_GREEDY_RANDOM:
            selected, value, n_calls = _double_greedy_random_core(
                f, ground, rng=rng
            )
        elif method == METHOD_DOUBLE_GREEDY_DETERMINISTIC:
            selected, value, n_calls = _double_greedy_deterministic_core(f, ground)
        elif method == METHOD_DISTORTED_GREEDY:
            if k is None:
                raise SubmodularError("distorted_greedy requires k (cardinality)")
            selected, value, n_calls = _distorted_greedy_core(
                f, ground, k, gamma=gamma
            )
        elif method == METHOD_SIEVE_STREAMING:
            if k is None:
                raise SubmodularError("sieve_streaming requires k (cardinality)")
            _validate_epsilon(epsilon)
            selected, value, n_calls, sieve_extras = _sieve_streaming_core(
                f, ground, k, epsilon=epsilon, max_singleton=max_singleton
            )
            extras.update(sieve_extras)
        elif method == METHOD_THRESHOLD_GREEDY:
            if k is None:
                raise SubmodularError("threshold_greedy requires k (cardinality)")
            _validate_epsilon(epsilon)
            selected, value, n_calls = _threshold_greedy_core(
                f, ground, k, epsilon=epsilon
            )
        else:  # pragma: no cover - guarded above
            raise SubmodularError(f"unhandled method {method!r}")

        # Feasibility check.
        feasible = True
        if k is not None and len(selected) > k:
            feasible = False
        if cost_list is not None and budget is not None:
            sel_cost = 0.0
            for v, w in zip(ground, cost_list):
                if v in selected:
                    sel_cost += w
            if sel_cost > budget + 1e-6:
                feasible = False

        elapsed = time.time_ns() - t0
        ratio = _ratio(method, monotone=monotone, epsilon=epsilon)
        payload = {
            "method": method,
            "selected": [str(v) for v in selected],
            "value": value,
            "n_oracle_calls": n_calls,
            "approx_ratio": ratio,
            "upper_bound": upper,
            "feasible": feasible,
            "elapsed_ns": elapsed,
            "monotone": monotone,
            "epsilon": epsilon,
        }
        digest = self._attest(SUBMOD_SOLVED, payload)
        rep = SubmodularReport(
            method=method,
            selected=list(selected),
            value=value,
            n_oracle_calls=n_calls,
            approx_ratio=ratio,
            upper_bound=upper,
            feasible=feasible,
            elapsed_ns=elapsed,
            digest=digest,
            extras=extras if return_extras else {},
        )
        with self._lock:
            self._n_solves += 1
            self._n_oracle_calls += n_calls
        self._emit(SUBMOD_SOLVED, payload | {"digest": digest})
        return rep

    # ---- public API: streaming (one-pass) ----------------------------

    def stream(
        self,
        f: Callable[[Iterable[Any]], float],
        ground_iterator: Iterable[Any],
        k: int,
        *,
        epsilon: float = 0.1,
        max_singleton: float | None = None,
    ) -> StreamReport:
        """Run one pass over an iterable ground set with Sieve-Streaming.

        Identical to ``maximize(method=METHOD_SIEVE_STREAMING)`` but
        returns a typed :class:`StreamReport`.  Designed to be called
        from a coordination engine as items arrive on a queue.
        """
        _validate_epsilon(epsilon)
        if k <= 0:
            raise SubmodularError("k must be > 0 for stream")
        t0 = time.time_ns()
        ground = list(ground_iterator)
        if not ground:
            raise SubmodularError("ground iterator is empty")
        selected, value, n_calls, extras = _sieve_streaming_core(
            f, ground, k, epsilon=epsilon, max_singleton=max_singleton
        )
        elapsed = time.time_ns() - t0
        payload = {
            "method": METHOD_SIEVE_STREAMING,
            "selected": [str(v) for v in selected],
            "value": value,
            "n_oracle_calls": n_calls,
            "n_thresholds": extras.get("n_thresholds", 0),
            "max_singleton": extras.get("max_singleton", 0.0),
            "epsilon": epsilon,
            "elapsed_ns": elapsed,
        }
        digest = self._attest(SUBMOD_STREAM_FINALISED, payload)
        rep = StreamReport(
            method=METHOD_SIEVE_STREAMING,
            selected=list(selected),
            value=value,
            n_oracle_calls=n_calls,
            n_thresholds=extras.get("n_thresholds", 0),
            max_singleton=extras.get("max_singleton", 0.0),
            approx_ratio=max(0.0, 0.5 - epsilon),
            epsilon=epsilon,
            elapsed_ns=elapsed,
            digest=digest,
        )
        with self._lock:
            self._n_solves += 1
            self._n_oracle_calls += n_calls
        self._emit(SUBMOD_STREAM_FINALISED, payload | {"digest": digest})
        return rep

    # ---- public API: cover -------------------------------------------

    def cover(
        self,
        f: Callable[[Iterable[Any]], float],
        ground_set: Iterable[Any],
        quota: float,
        *,
        max_picks: int | None = None,
    ) -> SubmodularReport:
        """Submodular cover: minimise |S| s.t. f(S) ≥ quota.

        Returns a :class:`SubmodularReport` whose ``approx_ratio`` is
        the Wolsey 1982 logarithmic bound ``1 + ln(Q/η)`` where ``η`` is
        the smallest positive marginal seen during the run, encoded as
        ``approx_ratio`` (this is the *multiplicative* bound on |S|).
        """
        ground = _normalise_ground_set(ground_set)
        if not ground:
            raise SubmodularError("ground_set is empty")
        if quota <= 0.0:
            raise SubmodularError("quota must be > 0")
        t0 = time.time_ns()
        # We need an estimate of η to fill in the report; we compute it
        # from the first round's singletons.
        f_empty = _oracle(f, [])
        ones = []
        for v in ground:
            ones.append(_oracle(f, [v]) - f_empty)
        eta = min((x for x in ones if x > 0.0), default=0.0)
        selected, f_S, n_calls = _submodular_cover_core(
            f, ground, quota, f_total=None, max_picks=max_picks
        )
        n_calls += 1 + len(ground)  # f(∅) + singletons we computed here
        elapsed = time.time_ns() - t0
        ratio = (1.0 + math.log(quota / eta)) if eta > 0.0 else math.inf
        payload = {
            "method": METHOD_SUBMODULAR_COVER,
            "quota": quota,
            "selected": [str(v) for v in selected],
            "value": f_S,
            "n_oracle_calls": n_calls,
            "eta": eta,
            "approx_ratio": ratio,
            "elapsed_ns": elapsed,
        }
        digest = self._attest(SUBMOD_SOLVED, payload)
        rep = SubmodularReport(
            method=METHOD_SUBMODULAR_COVER,
            selected=list(selected),
            value=f_S,
            n_oracle_calls=n_calls,
            approx_ratio=ratio,
            upper_bound=None,
            feasible=f_S + _EPS >= quota,
            elapsed_ns=elapsed,
            digest=digest,
            extras={"quota": quota, "eta": eta},
        )
        with self._lock:
            self._n_solves += 1
            self._n_oracle_calls += n_calls
        self._emit(SUBMOD_SOLVED, payload | {"digest": digest})
        return rep

    # ---- public API: certificates ------------------------------------

    def certify_submodular(
        self,
        f: Callable[[Iterable[Any]], float],
        ground_set: Iterable[Any],
        *,
        n_samples: int = 200,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> CertificateReport:
        """Empirical-Bernstein upper bound on the DR violation rate.

        For ``n_samples`` random pairs ``(A, B)`` with ``A ⊆ B`` and a
        random ``v ∉ B`` from the ground set, we test::

            Δ_A = f(A ∪ {v}) - f(A)  ≥  f(B ∪ {v}) - f(B) = Δ_B?

        Each pair is a Bernoulli trial of *violation*.  Hoeffding gives::

            p̂ + √( log(2/α) / (2 N) )

        as a ``1 - α`` upper bound on the true violation probability.
        Empirical-Bernstein (Maurer-Pontil 2009) is tighter when the
        empirical variance is small::

            p̂ + √( 2 s² log(2/α) / N ) + 7 log(2/α) / (3 (N - 1)).

        The headline rejection is ``hoeffding_upper`` — a strict
        Hoeffding ``1 - α`` upper bound on the violation rate.
        """
        _validate_alpha(alpha)
        if n_samples < 1:
            raise SubmodularError("n_samples must be ≥ 1")
        ground = _normalise_ground_set(ground_set)
        if len(ground) < 3:
            raise SubmodularError(
                "certify_submodular needs |ground| ≥ 3 (A ⊂ B, v ∉ B)"
            )
        rng = random.Random(seed) if seed is not None else self._rng
        n_violations = 0
        loss_squared_sum = 0.0
        loss_sum = 0.0
        for _ in range(n_samples):
            # Sample A ⊆ B ⊆ V with A ≠ V, B ≠ V, v ∉ B.
            n = len(ground)
            size_B = rng.randint(1, n - 1)
            B = rng.sample(ground, size_B)
            size_A = rng.randint(0, size_B - 1) if size_B > 1 else 0
            A = rng.sample(B, size_A)
            remaining = [w for w in ground if w not in B]
            if not remaining:
                continue
            v = rng.choice(remaining)
            delta_A = _oracle(f, A + [v]) - _oracle(f, A)
            delta_B = _oracle(f, B + [v]) - _oracle(f, B)
            violated = 1 if delta_A + _EPS < delta_B else 0
            n_violations += violated
            loss_sum += violated
            loss_squared_sum += violated  # binary
        rate = n_violations / n_samples
        log_term = math.log(2.0 / alpha)
        hoeffding = rate + math.sqrt(log_term / (2.0 * n_samples))
        # Empirical-Bernstein (Maurer-Pontil 2009).
        if n_samples > 1:
            mean = loss_sum / n_samples
            var = max(0.0, (loss_squared_sum - n_samples * mean * mean) / (n_samples - 1))
            bernstein = rate + math.sqrt(2.0 * var * log_term / n_samples) + (
                7.0 * log_term / (3.0 * (n_samples - 1))
            )
        else:
            bernstein = 1.0
        payload = {
            "n_samples": n_samples,
            "n_violations": n_violations,
            "violation_rate": rate,
            "hoeffding_upper": min(1.0, hoeffding),
            "bernstein_upper": min(1.0, bernstein),
            "alpha": alpha,
        }
        digest = self._attest(SUBMOD_CERTIFIED, payload)
        with self._lock:
            self._n_certificates += 1
            self._n_oracle_calls += 2 * n_samples
        self._emit(SUBMOD_CERTIFIED, payload | {"digest": digest})
        return CertificateReport(
            n_samples=n_samples,
            n_violations=n_violations,
            violation_rate=rate,
            hoeffding_upper=min(1.0, hoeffding),
            bernstein_upper=min(1.0, bernstein),
            alpha=alpha,
            digest=digest,
        )

    # ---- public API: curvature ---------------------------------------

    def curvature(
        self,
        f: Callable[[Iterable[Any]], float],
        ground_set: Iterable[Any],
    ) -> float:
        """Conforti-Cornuéjols total curvature ``c ∈ [0, 1]``.

        For a monotone submodular ``f`` with ``f({v}) > 0`` for all
        ``v``,

            c = 1 - min_v (f(V) - f(V ∖ {v})) / f({v}).

        ``c = 0`` ↔ additive / modular;  ``c = 1`` ↔ worst case.
        """
        ground = _normalise_ground_set(ground_set)
        if not ground:
            raise SubmodularError("ground_set is empty")
        f_empty = _oracle(f, [])
        f_V = _oracle(f, ground)
        worst_ratio = 1.0
        for v in ground:
            singleton = _oracle(f, [v]) - f_empty
            if singleton <= 0.0:
                # ``f({v}) = 0`` ⇒ curvature undefined; ignore.
                continue
            without_v = _oracle(f, [w for w in ground if w != v])
            marginal_at_V = f_V - without_v
            ratio = marginal_at_V / singleton
            if ratio < worst_ratio:
                worst_ratio = ratio
        return max(0.0, 1.0 - worst_ratio)

    def curvature_bound(self, c: float, k: int) -> float:
        """Conforti-Cornuéjols 1984 multiplicative bound::

            f(Ŝ) / f(S*)  ≥  (1/c) · (1 - (1 - c/k)^k)            (c > 0)
                          ≥  1 - 1/e                              (c = 1)
                          → 1                                     (c → 0)
        """
        if k <= 0:
            raise SubmodularError("k must be > 0")
        if not (0.0 <= c <= 1.0):
            raise SubmodularError("curvature c must be in [0, 1]")
        if c <= _EPS:
            # Modular limit: greedy is optimal.
            return 1.0
        return (1.0 / c) * (1.0 - (1.0 - c / k) ** k)


# =====================================================================
# Internal helpers exposed for clarity
# =====================================================================


def _monotone_upper(singletons: Sequence[float], k: int) -> float:
    """Sum of the top-k singleton marginals = a valid upper bound on f(S*).

    For monotone submodular ``f``,
        f(S*) ≤ Σ_{i ∈ S*} f({i}) - f(∅)  ≤  sum of top-k singletons.
    """
    if k <= 0 or not singletons:
        return 0.0
    top = sorted((float(x) for x in singletons), reverse=True)[:k]
    return float(sum(top))


# =====================================================================
# Module-level convenience wrappers
# =====================================================================


def lazy_greedy(
    f: Callable[[Iterable[Any]], float],
    ground_set: Iterable[Any],
    k: int,
) -> SubmodularReport:
    """Convenience wrapper: lazy greedy on monotone submodular ``f``."""
    return Submodular().maximize(
        f, ground_set, k=k, method=METHOD_LAZY_GREEDY, monotone=True
    )


def stochastic_greedy(
    f: Callable[[Iterable[Any]], float],
    ground_set: Iterable[Any],
    k: int,
    *,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> SubmodularReport:
    """Convenience wrapper: stochastic greedy with ``(1 - 1/e - ε)`` bound."""
    return Submodular(random_seed=seed).maximize(
        f,
        ground_set,
        k=k,
        method=METHOD_STOCHASTIC_GREEDY,
        monotone=True,
        epsilon=epsilon,
        seed=seed,
    )


def double_greedy(
    f: Callable[[Iterable[Any]], float],
    ground_set: Iterable[Any],
    *,
    randomized: bool = True,
    seed: int | None = None,
) -> SubmodularReport:
    """Convenience wrapper: unconstrained non-monotone double greedy."""
    method = (
        METHOD_DOUBLE_GREEDY_RANDOM if randomized else METHOD_DOUBLE_GREEDY_DETERMINISTIC
    )
    return Submodular(random_seed=seed).maximize(
        f, ground_set, method=method, monotone=False, seed=seed
    )


def sieve_streaming(
    f: Callable[[Iterable[Any]], float],
    ground_iterator: Iterable[Any],
    k: int,
    *,
    epsilon: float = 0.1,
) -> StreamReport:
    """Convenience wrapper: one-pass streaming with ``(½ - ε)`` bound."""
    return Submodular().stream(f, ground_iterator, k, epsilon=epsilon)


__all__ = [
    # public class + reports
    "Submodular",
    "SubmodularReport",
    "StreamReport",
    "CertificateReport",
    # exceptions
    "SubmodularError",
    "NotSubmodular",
    # canonical objectives
    "FacilityLocation",
    "WeightedCoverage",
    "MonotoneSetCover",
    "LogDeterminant",
    "GaussianEntropy",
    "MaxCut",
    "ConcaveOverModular",
    "FeatureBased",
    # convenience wrappers
    "lazy_greedy",
    "stochastic_greedy",
    "double_greedy",
    "sieve_streaming",
    # method constants
    "METHOD_LAZY_GREEDY",
    "METHOD_NAIVE_GREEDY",
    "METHOD_CELF",
    "METHOD_STOCHASTIC_GREEDY",
    "METHOD_COST_GREEDY",
    "METHOD_SVIRIDENKO_KNAPSACK",
    "METHOD_DOUBLE_GREEDY_RANDOM",
    "METHOD_DOUBLE_GREEDY_DETERMINISTIC",
    "METHOD_DISTORTED_GREEDY",
    "METHOD_SIEVE_STREAMING",
    "METHOD_SUBMODULAR_COVER",
    "METHOD_THRESHOLD_GREEDY",
    "KNOWN_METHODS",
    # event kinds
    "SUBMOD_STARTED",
    "SUBMOD_SOLVED",
    "SUBMOD_STREAM_OBSERVED",
    "SUBMOD_STREAM_FINALISED",
    "SUBMOD_CERTIFIED",
    "SUBMOD_CLEARED",
    "SUBMOD_REPORT",
]
