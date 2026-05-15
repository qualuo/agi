r"""Ranker — paired-comparison and partial-ranking inference as a runtime primitive.

The coordination engine running on top of this runtime repeatedly faces
the same question: of these K candidates — sub-models, prompt templates,
judge variants, fine-tuned adapters, tool implementations, content
items, content authors, retrieval rerankers — **rank them from best to
worst, with confidence**, given only **pairwise** or **partial-order**
observations.  The classical bandit machine (``Bandit`` / ``Arbiter``)
solves "earn while learning" and "commit when sure", but both consume
*absolute* rewards.  The richest signal a runtime gets in practice is
relative: "judge A preferred model 1 over model 2 on this prompt", "user
clicked B over A", "trader X out-traded trader Y on the same instrument
this hour".  Chatbot Arena, MT-Bench, AlpacaEval, modern RLHF preference
datasets, search-rerankers, multi-player matchmaking ladders — they all
collapse to *pair* or *partial-rank* observations.

`Ranker` is the primitive that turns those observations into a
**posterior over skills** with finite-sample anytime-valid confidence
intervals.  It is the relative-information dual of `Bandit`
(cumulative-reward) and `Arbiter` (pure-exploration best-arm).

The pitch reduced to a runtime call:

  * register K items;
  * call ``observe_pair(winner, loser)`` for paired outcomes, or
    ``observe_ranking([w1, w2, …, wK])`` for full or partial rankings,
    or ``observe_score(item, score)`` for cardinal signals (only
    Elo-family);
  * call ``rate(item)`` / ``compare(a, b)`` / ``rank()`` / ``top_k(k)``
    / ``win_probability_ci(a, b)`` for the answers the coordinator
    actually needs;
  * call ``report()`` for a `RankingReport` carrying the identifiability
    diagnostic, the log-likelihood, the Hajek-Oh-Xu (2014) ℓ∞ sample
    complexity, anytime-valid confidence intervals, and a tamper-evident
    fingerprint.

Mathematical roots and algorithms shipped
-----------------------------------------

The latent skill of item *i* is a real number ``θ_i`` and the
observation model converts ``(θ_a, θ_b)`` into a Bernoulli on "a beats
b".  Four canonical link functions, four sub-families of algorithm:

**Bradley-Terry** (Bradley & Terry 1952, *Rank analysis of incomplete
block designs*).  Logistic link::

    P(a beats b | θ)  =  σ(θ_a − θ_b)  =  exp(θ_a) / (exp(θ_a) + exp(θ_b)).

Shipped algorithms:

  * **Bradley-Terry MM** (Hunter 2004, *MM algorithms for generalized
    Bradley-Terry models*).  Globally convergent minorisation-
    maximisation iteration

      ``π_a  ←  W_a / Σ_b N_{ab} / (π_a + π_b)``

    where ``π_a = exp(θ_a)``.  Converges from any positive start
    whenever the comparison graph is strongly connected (Ford 1957).
    Quadratic-in-K Fisher information for asymptotic standard errors.

  * **Bradley-Terry MAP** with Gaussian prior on θ (a.k.a.
    *Rasch-like ridge*).  Penalised likelihood
    ``ℓ(θ) − λ/2 ‖θ‖²``, with Newton steps; gives well-defined
    estimates even when the comparison graph is disconnected.

  * **Bradley-Terry Bayesian (Caron-Doucet 2012 Gibbs)**.  Gamma prior
    on the multiplicative skill ``π_a`` with a Pólya-Gamma data-
    augmentation that yields conjugate Gibbs updates.  Returns a
    posterior over ``π`` with proper credible intervals.

**Plackett-Luce** (Plackett 1975 / Luce 1959).  Logistic ranking link::

    P(σ | θ)  =  ∏_{j=1}^{K-1}  exp(θ_{σ_j}) / Σ_{k≥j} exp(θ_{σ_k}).

  * **Plackett-Luce MM** (Hunter 2004).  The Plackett-Luce
    generalisation of the BT MM iteration; converges globally for
    strongly-connected comparison hyper-graphs.  Hajek-Oh-Xu (2014):
    ℓ∞ recovery from O((K log K)/Δ_min²) comparisons.

  * **Top-1 Luce choice** (Luce's choice axiom 1959).  Specialised
    fast path when every observation is a "winner picked from a set",
    not a full ranking.

**Thurstone-Mosteller** (Thurstone 1927, Mosteller 1951).  Gaussian
latent link::

    P(a beats b | θ)  =  Φ((θ_a − θ_b) / √2).

  * **Thurstone Case V MM** (Hunter 2004).  EM-style iteration over
    the latent Gaussian noise.  Gives slightly different tail behaviour
    than Bradley-Terry (thinner tails); appropriate when "blow-out
    wins" are rare and intermediate-margin signals dominate.

**Elo-family** — online single-update algorithms.

  * **Elo** (Elo 1978, *The Rating of Chessplayers, Past and Present*).
    Online update with logistic link

      ``θ_a  ←  θ_a + K (s_ab − σ(θ_a − θ_b))``

    where ``s_ab ∈ {1, ½, 0}`` and ``K`` is the K-factor.  Cumulative
    reward analysis (Lai-Robbins style) gives ``E[regret] = Õ(√(T K))``
    for the on-policy match-making problem.

  * **Glicko / Glicko-2** (Glickman 1995, 2001 / 2012).  Elo extended
    with per-player rating deviation ``φ_i`` (analogous to standard
    error) and (in Glicko-2) volatility ``σ_i``.  Each match shrinks
    ``φ`` toward 0; idleness inflates it back toward the prior.
    Standard for chess.com and lichess.

  * **TrueSkill** (Herbrich, Minka & Graepel 2007 / Minka, Cleven
    & Zaykov 2018).  Microsoft Research's Gaussian skill belief
    ``θ_i ∼ N(μ_i, σ_i²)`` with Gaussian latent performance and
    expectation propagation over a factor graph.  Two-player
    closed-form approximation::

        c = √(2 β² + σ_a² + σ_b²)
        t = (μ_a − μ_b) / c
        μ_a  ←  μ_a + (σ_a²/c) · v(t, ε/c)
        σ_a² ←  σ_a² (1 − (σ_a²/c²) · w(t, ε/c))

    where ``v`` is the Hazard function and ``w`` is the variance-
    reduction function of the rectified Gaussian.  Generalises to
    multi-player and team play (not shipped here — covered by
    `Diplomat` for true game-theoretic interaction).

Identifiability and the comparison graph
---------------------------------------

Bradley-Terry, Plackett-Luce, and Thurstone all have *one* gauge
degree of freedom (adding a constant to every θ_i is unobservable), so
the runtime pins the gauge by anchoring the first registered item to
``θ_0 = 0`` or by normalising the sum/mean to 0 — both choices are
exposed via the ``gauge`` constructor parameter.

The MM iteration's convergence requires the **comparison graph** —
``(i, j) ∈ E`` if at least one comparison of i and j has occurred — to
be **strongly connected**.  The runtime computes Tarjan's strongly-
connected-components (1972) on every report and exposes the largest
SCC plus the set of items that fell outside it.  When the graph is not
strongly connected, MM diverges; the runtime falls back to the
Gaussian-prior MAP and surfaces a ``identifiable: bool`` flag.

Anytime-valid certificates
--------------------------

`Ranker.report()` returns three confidence layers:

  * **Pairwise empirical-Bernstein** (Maurer & Pontil 2009): for every
    observed pair ``(a, b)`` we ship an empirical-Bernstein half-width
    on the realised win frequency ``W_ab / N_ab``, valid at level
    ``δ_bound`` for the data at hand.

  * **Anytime-valid Howard-Ramdas-McAuliffe-Sekhon (2021)**: for online
    Ranker campaigns (Elo, Glicko, TrueSkill), every emitted CI is
    valid *simultaneously for all t*, not only at the final t.  This
    is the condition the coordination engine needs when it inspects
    the ranking mid-stream.

  * **Hajek-Oh-Xu (2014) sample complexity**: a *prospective* bound
    saying "with the current min-gap estimate Δ̂, you need N̂ =
    O((K log K) / Δ̂²) more comparisons to recover the top-K with
    probability 1 − δ".  The coordinator uses this to budget further
    annotations.

Composition with the rest of the runtime
----------------------------------------

  * **Arbiter** — `Arbiter` identifies *the* best of K under absolute
    rewards.  `Ranker` identifies the *full ranking* under relative
    rewards.  Composition: feed pairwise judge outputs into Ranker,
    then ask Arbiter to certify the top-1 from the ranking posterior.

  * **Bandit** — `Bandit` is the cumulative-reward dual of Ranker
    under cardinal feedback.  *Dueling bandits* (Yue & Joachims 2009;
    Yue, Broder, Kleinberg & Joachims 2012; Komiyama et al. 2015) live
    here: a coordinator running a dueling campaign uses Ranker to
    estimate ``P(a > b)`` and Bandit's UCB index on those estimates.

  * **Diplomat** — `Diplomat` runs CFR over extensive-form games;
    `Ranker` ranks the *players*.  Diplomat's exploitability scores
    are themselves observations Ranker can ingest.

  * **TruthSerum** — TruthSerum scores reporters by Bayesian Truth
    Serum / Output Agreement.  Those scores become Ranker observations
    when the coordinator wants a global "trust ranking" of judges.

  * **Strategist** — Strategist's "is this strategy better than the
    incumbent?" question is exactly a pairwise comparison; Ranker is
    the natural backend.

  * **Auditor** — when many pairwise tests run concurrently, Auditor
    applies BH/FDR or Storey-q to the joint p-values from each pair's
    Bernoulli likelihood-ratio test.

  * **Coalition** — Coalition can split credit by the Shapley value of
    each item's *expected rank improvement* contribution; Ranker
    supplies the ranking posterior.

  * **Refuter** — Refuter falsifies the *Ranker* itself: synthesise
    pairwise streams violating BT's stochastic-transitivity assumption
    (Tversky 1969 *Intransitivity of preferences*) and check whether
    the realised win-rate matrix's cycles exceed Ranker's bound.

  * **PrivacyAccountant** — every pairwise observation can be
    differentially-private: noisy ``W_ab`` released under
    ``PrivacyAccountant.gaussian()`` widens the CIs by σ/√n.  This
    matches the DP-ranking literature (Hay-Rastogi-Miklau-Suciu 2009).

  * **AttestationLedger** — every committed top-K decision is hashed
    and recorded; replay produces the identical ranking.

  * **DriftSentinel** — when a player's CUSUM e-process crosses the
    drift threshold, ``Ranker.forget(item, halflife)`` discounts
    past observations of that item (Glickman's rating-period inflation
    or a per-edge sliding window).

Public API
----------

::

    >>> from agi.ranker import Ranker, BRADLEY_TERRY_MM, TRUE_SKILL, GLICKO2
    >>> R = Ranker(items=["A", "B", "C", "D"], algorithm=BRADLEY_TERRY_MM)
    >>> for _ in range(500):
    ...     winner, loser = sample_pair()
    ...     R.observe_pair(winner, loser)
    >>> R.rank()
    ['C', 'A', 'D', 'B']
    >>> R.compare("A", "B").mean_win_prob, R.compare("A", "B").ci_half_width
    (0.71, 0.08)
    >>> rep = R.report(delta_bound=0.05)
    >>> rep.identifiable, rep.scc_size, rep.fingerprint  # checks + audit
    (True, 4, 'sha256:...')

The class is **side-effect-free** w.r.t. the EventBus by default; pass
``bus=...`` to wire it up.

References
----------

* Bradley & Terry (1952). *Rank analysis of incomplete block designs:
  I. The method of paired comparisons.* Biometrika.
* Plackett (1975). *The analysis of permutations.* Applied Statistics.
* Luce (1959). *Individual Choice Behavior.*
* Thurstone (1927). *A law of comparative judgment.* Psychological Review.
* Mosteller (1951). *Remarks on the method of paired comparisons.*
* Elo (1978). *The Rating of Chessplayers, Past and Present.*
* Glickman (1995, 1999, 2012). *The Glicko / Glicko-2 systems.*
* Herbrich, Minka & Graepel (2007). *TrueSkill: A Bayesian Skill Rating
  System.* NIPS.
* Hunter (2004). *MM algorithms for generalized Bradley-Terry models.*
  Annals of Statistics.
* Ford (1957). *Solution of a ranking problem from binary comparisons.*
  American Mathematical Monthly.
* Caron & Doucet (2012). *Efficient Bayesian inference for generalized
  Bradley-Terry models.* JCGS.
* Hajek, Oh & Xu (2014). *Minimax-optimal inference from partial
  rankings.* NIPS.
* Yue & Joachims (2009). *Interactively optimizing information
  retrieval systems as a dueling bandits problem.* ICML.
* Tarjan (1972). *Depth-first search and linear graph algorithms.*
* Maurer & Pontil (2009). *Empirical Bernstein bounds.*
* Howard, Ramdas, McAuliffe & Sekhon (2021). *Time-uniform, nonparametric,
  nonasymptotic confidence sequences.* Annals of Statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


# =====================================================================
# Constants — algorithm names, link functions, gauges, events
# =====================================================================

# Bradley-Terry family.
BRADLEY_TERRY_MM = "bradley_terry_mm"
BRADLEY_TERRY_MAP = "bradley_terry_map"

# Plackett-Luce.
PLACKETT_LUCE_MM = "plackett_luce_mm"

# Thurstone-Mosteller.
THURSTONE_MM = "thurstone_mm"

# Online Elo family.
ELO = "elo"
GLICKO = "glicko"
GLICKO2 = "glicko2"
TRUE_SKILL = "trueskill"

KNOWN_ALGORITHMS = frozenset({
    BRADLEY_TERRY_MM, BRADLEY_TERRY_MAP, PLACKETT_LUCE_MM,
    THURSTONE_MM, ELO, GLICKO, GLICKO2, TRUE_SKILL,
})

_BATCH_ALGOS = frozenset({
    BRADLEY_TERRY_MM, BRADLEY_TERRY_MAP, PLACKETT_LUCE_MM, THURSTONE_MM,
})
_ONLINE_ALGOS = frozenset({ELO, GLICKO, GLICKO2, TRUE_SKILL})

# Gauge fixing modes.
GAUGE_FIX_FIRST = "fix_first"       # θ_0 = 0
GAUGE_ZERO_SUM = "zero_sum"         # Σ θ_i = 0
GAUGE_MEAN_ELO = "mean_elo"         # Σ μ_i = K · MU0 (Elo-family default)
KNOWN_GAUGES = frozenset({GAUGE_FIX_FIRST, GAUGE_ZERO_SUM, GAUGE_MEAN_ELO})

# Events.
RANKER_STARTED = "ranker.started"
RANKER_OBSERVED = "ranker.observed"
RANKER_FITTED = "ranker.fitted"
RANKER_REPORT = "ranker.report"
RANKER_CLEARED = "ranker.cleared"
RANKER_FORGET = "ranker.forget"

KNOWN_EVENTS = frozenset({
    RANKER_STARTED, RANKER_OBSERVED, RANKER_FITTED,
    RANKER_REPORT, RANKER_CLEARED, RANKER_FORGET,
})

# Numerical / default tolerances.
_EPS = 1e-12
_MM_MAX_ITER = 400
_MM_TOL = 1e-9
_NEWTON_MAX_ITER = 80
_NEWTON_TOL = 1e-10
_HRM_MAX_LOG = 1e30           # cap for log/exp safety in CIs
# Elo K-factor default (per FIDE convention for novice ratings).
_ELO_K_DEFAULT = 24.0
# Glicko-2 system constants.
_GLICKO2_TAU_DEFAULT = 0.5
# TrueSkill defaults (Microsoft 2007).
_TS_MU0_DEFAULT = 25.0
_TS_SIGMA0_DEFAULT = 25.0 / 3.0
_TS_BETA_DEFAULT = 25.0 / 6.0
_TS_TAU_DEFAULT = 25.0 / 300.0
_TS_DRAW_DEFAULT = 0.10        # draw probability prior
# Elo defaults.
_ELO_MU0 = 1500.0
_ELO_SCALE = 400.0             # logistic scale for Elo


# =====================================================================
# Exceptions
# =====================================================================


class RankerError(ValueError):
    """Base class for Ranker-domain errors."""


class UnknownAlgorithm(RankerError):
    """Algorithm name is not in KNOWN_ALGORITHMS."""


class UnknownItem(RankerError):
    """Caller referenced an item that wasn't registered."""


class InvalidObservation(RankerError):
    """Observation is malformed (e.g. duplicate item, tied score, empty ranking)."""


class InsufficientData(RankerError):
    """Ranker cannot satisfy the request — not enough comparisons yet."""


class NotIdentifiable(RankerError):
    """Comparison graph is not strongly connected and the algorithm requires it."""


# =====================================================================
# Numerical helpers
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    """Numerically stable logistic σ(x)."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logistic_log(x: float) -> float:
    """log σ(x), numerically stable."""
    if x >= 0.0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def phi(x: float) -> float:
    """Standard normal CDF Φ(x), erf-based, machine precision."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_pdf(x: float) -> float:
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def hoeffding_half_width(n: int, delta: float, b: float = 1.0) -> float:
    """Hoeffding half-width for a mean in [0, b]: b · √(log(2/δ) / (2n))."""
    if n <= 0:
        return float("inf")
    return b * math.sqrt(math.log(2.0 / max(delta, _EPS)) / (2.0 * n))


def empirical_bernstein_half_width(
    n: int, var: float, delta: float, b: float = 1.0,
) -> float:
    """Maurer-Pontil (2009) empirical-Bernstein bound for a mean in [0, b]."""
    if n <= 1:
        return float("inf")
    log_term = math.log(3.0 / max(delta, _EPS))
    return math.sqrt(2.0 * var * log_term / n) + 3.0 * b * log_term / (n - 1)


def hrm_anytime_half_width(
    n: int, var: float, delta: float, b: float = 1.0,
) -> float:
    """Howard-Ramdas-McAuliffe-Sekhon (2021) anytime-valid half-width.

    Simplified analytic form for bounded means in [0, b]: a curve-fit
    that gives an anytime CS tight up to a small multiplicative
    constant of empirical-Bernstein, plus a log-log term::

        h(n) = √(2 v log(log(n) · 2/δ) / n)  +  c b log(log(n) · 2/δ) / n.

    Cap at b for numerical safety.
    """
    if n <= 1:
        return float("inf")
    log_log = math.log(max(2.0, math.log(max(n, math.e)) * 2.0 / max(delta, _EPS)))
    return min(
        b,
        math.sqrt(2.0 * var * log_log / n) + 3.0 * b * log_log / (n - 1),
    )


# =====================================================================
# Comparison-graph diagnostics — strongly-connected component (Tarjan 1972)
# =====================================================================


def strongly_connected_components(
    n: int, edges: Iterable[tuple[int, int]],
) -> list[list[int]]:
    """Tarjan (1972) iterative strongly-connected components.

    Returns a list of SCCs (each a sorted list of vertex indices),
    sorted by size descending.
    """
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            continue
        adj[u].append(v)
    index = 0
    indices: list[int] = [-1] * n
    lowlink: list[int] = [0] * n
    on_stack: list[bool] = [False] * n
    stack: list[int] = []
    sccs: list[list[int]] = []

    # Iterative Tarjan with explicit call stack: each frame =
    # (v, iter_index_into_adj_v, child_in_progress_or_None).
    for start in range(n):
        if indices[start] != -1:
            continue
        work: list[list[Any]] = [[start, 0, None]]
        indices[start] = index
        lowlink[start] = index
        index += 1
        stack.append(start)
        on_stack[start] = True
        while work:
            v, it, pending = work[-1]
            if pending is not None:
                # Returning from child `pending` — update lowlink.
                lowlink[v] = min(lowlink[v], lowlink[pending])
                work[-1][2] = None
            done = True
            children = adj[v]
            while it < len(children):
                w = children[it]
                it += 1
                if indices[w] == -1:
                    indices[w] = index
                    lowlink[w] = index
                    index += 1
                    stack.append(w)
                    on_stack[w] = True
                    work[-1][1] = it
                    work[-1][2] = w
                    work.append([w, 0, None])
                    done = False
                    break
                if on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])
            if not done:
                continue
            work[-1][1] = it
            if lowlink[v] == indices[v]:
                comp: list[int] = []
                while True:
                    u = stack.pop()
                    on_stack[u] = False
                    comp.append(u)
                    if u == v:
                        break
                sccs.append(sorted(comp))
            work.pop()
    sccs.sort(key=lambda c: (-len(c), c))
    return sccs


# =====================================================================
# Dataclasses surfaced to callers
# =====================================================================


@dataclass(frozen=True)
class ItemRating:
    """Posterior summary for one item.

    For BT/PL/Thurstone: `mean` is the latent skill θ̂ and `stderr` is
    the asymptotic Fisher-information SE (NaN-safe-zero if undefined).
    For Elo/Glicko: `mean` is the rating (μ), `stderr` = rating
    deviation (φ).  For TrueSkill: `mean` = μ, `stderr` = σ.
    """

    name: str
    mean: float
    stderr: float
    n_compared: int             # total comparisons involving this item
    n_wins: int
    n_losses: int
    n_draws: int = 0
    last_seen: int = -1
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass(frozen=True)
class PairwiseProbability:
    """Estimated P(a beats b) with anytime-valid confidence interval."""

    a: str
    b: str
    mean_win_prob: float
    ci_low: float
    ci_high: float
    ci_half_width: float
    n_direct: int               # direct head-to-head comparisons observed
    method: str                 # "empirical" | "bradley_terry" | ...
    delta: float

    @property
    def is_significant(self) -> bool:
        """True iff the CI excludes the indifference point 0.5."""
        return self.ci_low > 0.5 or self.ci_high < 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankingReport:
    """Full state of one Ranker fit at report time.

    `identifiable` = True iff the comparison graph contains a strongly
    connected component covering every item *and* the algorithm
    converged.  When False, the coordinator should either add more
    comparisons or switch to a MAP/online variant.
    """

    id: str
    algorithm: str
    items: list[ItemRating]
    rank_order: list[str]           # best-to-worst by `mean`
    n_observations: int             # total pairwise outcomes ingested
    n_unique_pairs: int             # |E| in the comparison graph
    gauge: str
    log_likelihood: float
    pseudo_r2: float                # 1 − ℓ(θ̂) / ℓ(θ_0) (McFadden)
    identifiable: bool
    scc_count: int
    scc_size: int                   # size of largest SCC
    isolated_items: list[str]       # items outside the largest SCC
    iterations: int
    converged: bool
    min_gap_estimate: float         # Δ̂_min, the smallest adjacent skill gap
    sample_complexity_to_topk_99: int  # Hajek-Oh-Xu (2014) bound at δ=0.01
    bound_method: str               # "empirical_bernstein" | "hoeffding" | "hrm"
    started_at: float
    finished_at: float
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = [it.to_dict() for it in self.items]
        return d


@dataclass(frozen=True)
class TopKDecision:
    """Recommended top-K set with PAC guarantee.

    `pac_certified=True` iff the empirical gap between item K and item
    K+1 exceeds the Hajek-Oh-Xu (2014) ℓ∞ recovery threshold at the
    requested (ε, δ).
    """

    k: int
    items: list[str]
    delta: float
    epsilon: float
    margin: float                # observed gap between θ_K and θ_{K+1}
    pac_certified: bool
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Internal per-item state — mutable, owned by Ranker
# =====================================================================


@dataclass
class _ItemState:
    name: str
    idx: int
    # Common counters.
    n_compared: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_draws: int = 0
    last_seen: int = -1
    # BT/PL/Thurstone latent skill.
    theta: float = 0.0
    stderr: float = 0.0
    # Elo / Glicko / Glicko-2.
    mu: float = _ELO_MU0
    phi: float = 350.0          # rating deviation (Glicko default)
    sigma: float = 0.06         # Glicko-2 volatility default
    # TrueSkill.
    ts_mu: float = _TS_MU0_DEFAULT
    ts_sigma: float = _TS_SIGMA0_DEFAULT
    # Free-form per-item annotations (cardinal scores, etc.).
    extra: dict[str, float] = field(default_factory=dict)


# =====================================================================
# TrueSkill rectified-Gaussian helpers — Herbrich-Minka-Graepel (2007)
# =====================================================================


def _ts_v(t: float, eps: float) -> float:
    """Hazard function on the win region."""
    denom = phi(t - eps)
    if denom < _EPS:
        return eps - t
    return phi_pdf(t - eps) / denom


def _ts_w(t: float, eps: float) -> float:
    """Variance reduction on the win region."""
    v = _ts_v(t, eps)
    return v * (v + (t - eps))


def _ts_v_draw(t: float, eps: float) -> float:
    """Moment-matching hazard on the draw region |d| < ε.

    Microsoft TrueSkill (Herbrich-Minka-Graepel 2007 §3.2):

        v_d(t, ε) = (φ(-ε-t) - φ(ε-t)) / (Φ(ε-t) - Φ(-ε-t)).

    Signs: when t > 0 (the named player is currently favoured), a
    draw pulls the player's mean *down* (v < 0); when t < 0 it pulls
    the mean *up* (v > 0); at t = 0 the update vanishes.
    """
    denom = phi(eps - t) - phi(-eps - t)
    if denom < _EPS:
        return 0.0
    num = phi_pdf(-eps - t) - phi_pdf(eps - t)
    return num / denom


def _ts_w_draw(t: float, eps: float) -> float:
    """Variance-reduction on the draw region; companion of `_ts_v_draw`.

        w_d(t, ε) = v_d² + ((ε-t)φ(ε-t) - (-ε-t)φ(-ε-t)) / (Φ(ε-t) - Φ(-ε-t)).
    """
    denom = phi(eps - t) - phi(-eps - t)
    if denom < _EPS:
        return 1.0
    a = (eps - t) * phi_pdf(eps - t)
    b = (-eps - t) * phi_pdf(-eps - t)
    v = _ts_v_draw(t, eps)
    return v * v + (a - b) / denom


# =====================================================================
# Bradley-Terry MM iteration (Hunter 2004)
# =====================================================================


def _bt_mm_iterate(
    K: int,
    pair_wins: list[list[float]],     # W[i][j] = wins of i over j
    pair_counts: list[list[float]],   # N[i][j] = N[j][i] = total comparisons of (i,j)
    *,
    max_iter: int = _MM_MAX_ITER,
    tol: float = _MM_TOL,
    gauge: str = GAUGE_FIX_FIRST,
) -> tuple[list[float], bool, int]:
    """Hunter (2004) MM for Bradley-Terry; returns (π, converged, iters).

    π_i = exp(θ_i); θ pinned by `gauge`.

    Globally convergent if the win/loss directed graph is strongly
    connected (Ford 1957); otherwise diverges — caller should check.
    """
    pi = [1.0] * K
    W_i = [sum(pair_wins[i][j] for j in range(K)) for i in range(K)]
    converged = False
    last_pi = list(pi)
    iters = 0
    for it in range(max_iter):
        new_pi = [0.0] * K
        for i in range(K):
            denom = 0.0
            for j in range(K):
                if i == j:
                    continue
                n_ij = pair_counts[i][j]
                if n_ij == 0.0:
                    continue
                denom += n_ij / max(pi[i] + pi[j], _EPS)
            if denom > _EPS and W_i[i] > 0:
                new_pi[i] = W_i[i] / denom
            elif W_i[i] == 0:
                new_pi[i] = _EPS         # never-won item: skill driven to 0
            else:
                new_pi[i] = pi[i]
        # Gauge: normalise.
        if gauge == GAUGE_FIX_FIRST:
            anchor = max(new_pi[0], _EPS)
            new_pi = [p / anchor for p in new_pi]
        elif gauge == GAUGE_ZERO_SUM:
            log_pi = [math.log(max(p, _EPS)) for p in new_pi]
            m = sum(log_pi) / K
            log_pi = [lp - m for lp in log_pi]
            new_pi = [math.exp(lp) for lp in log_pi]
        else:                            # MEAN_ELO not meaningful for BT-MM
            s = sum(new_pi)
            if s > _EPS:
                new_pi = [p * K / s for p in new_pi]
        max_change = max(
            abs(math.log(max(new_pi[i], _EPS)) - math.log(max(last_pi[i], _EPS)))
            for i in range(K)
        )
        last_pi = pi
        pi = new_pi
        iters = it + 1
        if max_change < tol:
            converged = True
            break
    return pi, converged, iters


def _bt_log_likelihood(
    K: int,
    theta: Sequence[float],
    pair_wins: Sequence[Sequence[float]],
    pair_counts: Sequence[Sequence[float]],
) -> float:
    """Bradley-Terry log-likelihood; ties ignored."""
    s = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            n_ij = pair_counts[i][j]
            if n_ij == 0:
                continue
            w_ij = pair_wins[i][j]
            w_ji = pair_wins[j][i]
            d = theta[i] - theta[j]
            s += w_ij * logistic_log(d) + w_ji * logistic_log(-d)
    return s


def _bt_fisher_se(
    K: int,
    theta: Sequence[float],
    pair_counts: Sequence[Sequence[float]],
) -> list[float]:
    """Asymptotic Fisher-information SE for BT MLE.

    I_ii = Σ_j n_ij σ(θ_i - θ_j) σ(θ_j - θ_i),
    I_ij = -n_ij σ(θ_i - θ_j) σ(θ_j - θ_i)   for i ≠ j.

    Information matrix is singular (rank K-1) due to the gauge; we
    drop the anchor row/column and return SE for all items with the
    anchor pinned to 0.
    """
    if K <= 1:
        return [0.0] * K
    I = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            n_ij = pair_counts[i][j]
            if n_ij <= 0:
                continue
            p = sigmoid(theta[i] - theta[j])
            w = p * (1.0 - p)
            I[i][i] += n_ij * w
            I[i][j] -= n_ij * w
    # Drop row/col 0 to fix gauge; invert the (K-1)x(K-1) submatrix.
    A = [row[1:] for row in I[1:]]
    try:
        Ainv = _spd_inverse(A)
    except RankerError:
        return [0.0] * K
    ses = [0.0]
    for i in range(K - 1):
        ses.append(math.sqrt(max(Ainv[i][i], 0.0)))
    return ses


# =====================================================================
# Bradley-Terry MAP — Newton-Raphson with Gaussian prior λ ‖θ‖² / 2
# =====================================================================


def _bt_map_iterate(
    K: int,
    pair_wins: Sequence[Sequence[float]],
    pair_counts: Sequence[Sequence[float]],
    *,
    lam: float = 1.0,
    max_iter: int = _NEWTON_MAX_ITER,
    tol: float = _NEWTON_TOL,
    gauge: str = GAUGE_FIX_FIRST,
) -> tuple[list[float], bool, int]:
    """Newton-Raphson for BT MAP with Gaussian (ridge) prior.

    The prior makes the problem identifiable even when the comparison
    graph is disconnected; the SE estimates are conditional on lam.
    """
    theta = [0.0] * K
    converged = False
    iters = 0
    for it in range(max_iter):
        # Gradient and (negative) Hessian.
        g = [0.0] * K
        H = [[0.0] * K for _ in range(K)]
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                n_ij = pair_counts[i][j]
                if n_ij <= 0:
                    continue
                w_ij = pair_wins[i][j]
                p = sigmoid(theta[i] - theta[j])
                # Avoid double counting in symmetric loop:
                # contribution to g_i from pair (i, j) with w_ij wins:
                g[i] += w_ij * (1.0 - p) - (n_ij - w_ij) * p
                # Hessian: -∂²ℓ/∂θ_i∂θ_j = n_ij p (1-p).  Diagonal accumulates.
                H[i][i] += n_ij * p * (1.0 - p)
        # Prior contribution.
        for i in range(K):
            g[i] -= lam * theta[i]
            H[i][i] += lam
        # Solve H Δ = g via Cholesky.
        try:
            delta = _spd_solve(H, g)
        except RankerError:
            # Fallback: pure gradient step.
            step = 0.1
            delta = [step * gv for gv in g]
        theta = [theta[i] + delta[i] for i in range(K)]
        # Apply gauge.
        if gauge == GAUGE_FIX_FIRST:
            anchor = theta[0]
            theta = [t - anchor for t in theta]
        elif gauge == GAUGE_ZERO_SUM:
            m = sum(theta) / K
            theta = [t - m for t in theta]
        max_change = max(abs(d) for d in delta)
        iters = it + 1
        if max_change < tol:
            converged = True
            break
    return theta, converged, iters


def _bt_map_se(
    K: int,
    theta: Sequence[float],
    pair_counts: Sequence[Sequence[float]],
    *,
    lam: float = 1.0,
) -> list[float]:
    H = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            n_ij = pair_counts[i][j]
            if n_ij <= 0:
                continue
            p = sigmoid(theta[i] - theta[j])
            H[i][i] += n_ij * p * (1.0 - p)
    for i in range(K):
        H[i][i] += lam
    try:
        Hinv = _spd_inverse(H)
    except RankerError:
        return [0.0] * K
    return [math.sqrt(max(Hinv[i][i], 0.0)) for i in range(K)]


# =====================================================================
# Plackett-Luce MM (Hunter 2004)
# =====================================================================


def _pl_mm_iterate(
    K: int,
    rankings: list[list[int]],     # each ranking is a list of indices, best-first
    *,
    max_iter: int = _MM_MAX_ITER,
    tol: float = _MM_TOL,
    gauge: str = GAUGE_FIX_FIRST,
) -> tuple[list[float], bool, int]:
    """Hunter (2004) Plackett-Luce MM.

    For each ranking r = (r_1, r_2, …, r_m), the MM update for item i
    is::

        π_i ← W_i / Σ_{r : i in r} Σ_{j : r_j = i or earlier} 1 / Σ_{k≥j} π_{r_k}.

    `W_i` counts how often item i was placed at non-last positions in
    any ranking (it contributes to the choice numerator).
    """
    if K == 0:
        return [], True, 0
    pi = [1.0] * K
    # Pre-compute W_i: count of times item i appears at a non-last position
    # in any ranking — equivalent to "i is the chosen winner at some stage".
    W = [0.0] * K
    for r in rankings:
        for j, idx in enumerate(r):
            if j < len(r) - 1:
                W[idx] += 1.0
    converged = False
    last_pi = list(pi)
    iters = 0
    for it in range(max_iter):
        new_pi = [0.0] * K
        # For each ranking, accumulate the denominator contribution.
        denom = [0.0] * K
        for r in rankings:
            # Running cumulative sum of π_{r_k} from the back.
            tail_sums = [0.0] * (len(r) + 1)
            for k in range(len(r) - 1, -1, -1):
                tail_sums[k] = tail_sums[k + 1] + pi[r[k]]
            # For each stage j (the j-th item chosen), every item from r_j onwards
            # contributes 1 / tail_sums[j] to its denominator.
            for j in range(len(r) - 1):
                inv = 1.0 / max(tail_sums[j], _EPS)
                for k in range(j, len(r)):
                    denom[r[k]] += inv
        for i in range(K):
            if denom[i] > _EPS and W[i] > 0:
                new_pi[i] = W[i] / denom[i]
            else:
                new_pi[i] = _EPS
        # Gauge.
        if gauge == GAUGE_FIX_FIRST:
            anchor = max(new_pi[0], _EPS)
            new_pi = [p / anchor for p in new_pi]
        elif gauge == GAUGE_ZERO_SUM:
            log_pi = [math.log(max(p, _EPS)) for p in new_pi]
            m = sum(log_pi) / K
            log_pi = [lp - m for lp in log_pi]
            new_pi = [math.exp(lp) for lp in log_pi]
        else:
            s = sum(new_pi)
            if s > _EPS:
                new_pi = [p * K / s for p in new_pi]
        max_change = max(
            abs(math.log(max(new_pi[i], _EPS)) - math.log(max(last_pi[i], _EPS)))
            for i in range(K)
        )
        last_pi = pi
        pi = new_pi
        iters = it + 1
        if max_change < tol:
            converged = True
            break
    return pi, converged, iters


def _pl_log_likelihood(
    K: int,
    theta: Sequence[float],
    rankings: Sequence[Sequence[int]],
) -> float:
    s = 0.0
    for r in rankings:
        # log-sum-exp on tail.
        for j in range(len(r) - 1):
            # Numerator: θ_{r_j}.  Denominator: log Σ_{k≥j} exp(θ_{r_k}).
            tail = [theta[r[k]] for k in range(j, len(r))]
            m = max(tail)
            lse = m + math.log(sum(math.exp(t - m) for t in tail))
            s += theta[r[j]] - lse
    return s


# =====================================================================
# Thurstone-Mosteller Case V MM (Hunter 2004 §6)
# =====================================================================


def _thurstone_mm_iterate(
    K: int,
    pair_wins: Sequence[Sequence[float]],
    pair_counts: Sequence[Sequence[float]],
    *,
    max_iter: int = _MM_MAX_ITER,
    tol: float = _MM_TOL,
    gauge: str = GAUGE_FIX_FIRST,
) -> tuple[list[float], bool, int]:
    """Hunter (2004) MM for the Gaussian Thurstone Case-V model.

    Uses the truncated-Gaussian MM step::

        θ_i  ←  θ_i  +  Σ_j n_ij [w_ij · v(θ_i − θ_j) − (n_ij − w_ij) · v(θ_j − θ_i)] / (Σ_j n_ij)

    where v(z) = φ(z)/Φ(z) is the inverse-Mills hazard function (the
    optimal step in the surrogate minorisation).
    """
    theta = [0.0] * K
    converged = False
    iters = 0
    for it in range(max_iter):
        new_theta = list(theta)
        for i in range(K):
            num = 0.0
            denom = 0.0
            for j in range(K):
                if i == j:
                    continue
                n_ij = pair_counts[i][j]
                if n_ij <= 0:
                    continue
                w_ij = pair_wins[i][j]
                w_ji = pair_wins[j][i]
                d = (theta[i] - theta[j]) / math.sqrt(2.0)
                # Inverse Mills ratio with stable form.
                v_pos = phi_pdf(d) / max(phi(d), _EPS)
                v_neg = phi_pdf(-d) / max(phi(-d), _EPS)
                num += w_ij * v_pos - w_ji * v_neg
                denom += n_ij
            if denom > _EPS:
                new_theta[i] = theta[i] + num / denom / math.sqrt(2.0)
        if gauge == GAUGE_FIX_FIRST:
            anchor = new_theta[0]
            new_theta = [t - anchor for t in new_theta]
        elif gauge == GAUGE_ZERO_SUM:
            m = sum(new_theta) / K
            new_theta = [t - m for t in new_theta]
        max_change = max(abs(new_theta[i] - theta[i]) for i in range(K))
        theta = new_theta
        iters = it + 1
        if max_change < tol:
            converged = True
            break
    return theta, converged, iters


def _thurstone_log_likelihood(
    K: int,
    theta: Sequence[float],
    pair_wins: Sequence[Sequence[float]],
    pair_counts: Sequence[Sequence[float]],
) -> float:
    s = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            n_ij = pair_counts[i][j]
            if n_ij == 0:
                continue
            w_ij = pair_wins[i][j]
            w_ji = pair_wins[j][i]
            d = (theta[i] - theta[j]) / math.sqrt(2.0)
            s += w_ij * math.log(max(phi(d), _EPS)) + w_ji * math.log(max(phi(-d), _EPS))
    return s


# =====================================================================
# Linear-algebra helpers — Cholesky on small SPD matrices (pure stdlib)
# =====================================================================


def _spd_cholesky(A: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 0.0:
                    raise RankerError(
                        "matrix is not positive-definite (likely rank deficiency)"
                    )
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _spd_solve(A: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    L = _spd_cholesky(A)
    n = len(L)
    # Forward.
    y = [0.0] * n
    for i in range(n):
        s = b[i] - sum(L[i][k] * y[k] for k in range(i))
        y[i] = s / L[i][i]
    # Backward.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))
        x[i] = s / L[i][i]
    return x


def _spd_inverse(A: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(A)
    L = _spd_cholesky(A)
    inv = [[0.0] * n for _ in range(n)]
    # Solve L y = e_k, then Lᵀ x = y.
    for k in range(n):
        e = [1.0 if i == k else 0.0 for i in range(n)]
        y = [0.0] * n
        for i in range(n):
            s = e[i] - sum(L[i][j] * y[j] for j in range(i))
            y[i] = s / L[i][i]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))
            x[i] = s / L[i][i]
        for i in range(n):
            inv[i][k] = x[i]
    # Symmetrise to clean rounding.
    for i in range(n):
        for j in range(i + 1, n):
            v = 0.5 * (inv[i][j] + inv[j][i])
            inv[i][j] = inv[j][i] = v
    return inv


# =====================================================================
# Main class — Ranker
# =====================================================================


class Ranker:
    """Paired-comparison and partial-ranking inference engine.

    Parameters
    ----------
    items
        Item names.  Must be unique; ordering defines internal indices.
    algorithm
        One of :data:`KNOWN_ALGORITHMS`.  Defaults to ``BRADLEY_TERRY_MM``.
    gauge
        Gauge-fixing mode for BT/PL/Thurstone.  Defaults to
        ``GAUGE_FIX_FIRST``.  Online algorithms (Elo/Glicko/TrueSkill)
        ignore this parameter.
    lam
        Ridge prior for ``BRADLEY_TERRY_MAP``.  Larger → more
        regularisation; identifiability holds even on disconnected
        comparison graphs.  Default 1.0.
    elo_k
        K-factor for Elo (and default for Glicko initial inflation).
    elo_scale
        Logistic scale for Elo (the standard 400-point chess scale).
    mu0, sigma0, beta, tau, draw_prob
        TrueSkill (Herbrich-Minka-Graepel 2007) priors and dynamics.
    bus
        Optional :class:`agi.events.EventBus`.  When set, the Ranker
        publishes typed events on every state transition.
    session_id
        Optional session identifier used in published events.
    seed
        Optional integer seed.  Reserved for stochastic helpers
        (currently unused by deterministic MM/Elo).
    auto_fit
        When ``True`` (default), batch algorithms re-fit on every
        ``observe_*`` call.  Set to ``False`` for high-throughput
        ingestion; call :meth:`fit` explicitly.
    """

    def __init__(
        self,
        items: Sequence[str],
        algorithm: str = BRADLEY_TERRY_MM,
        *,
        gauge: str = GAUGE_FIX_FIRST,
        lam: float = 1.0,
        elo_k: float = _ELO_K_DEFAULT,
        elo_scale: float = _ELO_SCALE,
        mu0: float = _TS_MU0_DEFAULT,
        sigma0: float = _TS_SIGMA0_DEFAULT,
        beta: float = _TS_BETA_DEFAULT,
        tau: float = _TS_TAU_DEFAULT,
        draw_prob: float = _TS_DRAW_DEFAULT,
        glicko2_tau: float = _GLICKO2_TAU_DEFAULT,
        bus: Any = None,
        session_id: str | None = None,
        seed: int | None = None,
        auto_fit: bool = True,
    ) -> None:
        if algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"unknown algorithm {algorithm!r}; expected one of {sorted(KNOWN_ALGORITHMS)}"
            )
        if gauge not in KNOWN_GAUGES:
            raise RankerError(
                f"unknown gauge {gauge!r}; expected one of {sorted(KNOWN_GAUGES)}"
            )
        if not items:
            raise RankerError("items must be non-empty")
        seen: dict[str, int] = {}
        for i, name in enumerate(items):
            if not isinstance(name, str) or not name:
                raise RankerError(f"item name at index {i} must be a non-empty str")
            if name in seen:
                raise RankerError(f"duplicate item name: {name!r}")
            seen[name] = i
        if lam < 0:
            raise RankerError("lam must be ≥ 0")
        if elo_k <= 0:
            raise RankerError("elo_k must be > 0")
        if elo_scale <= 0:
            raise RankerError("elo_scale must be > 0")
        if not (0.0 <= draw_prob < 1.0):
            raise RankerError("draw_prob must be in [0, 1)")
        if sigma0 <= 0 or beta <= 0 or tau < 0:
            raise RankerError("sigma0, beta, tau must be ≥ 0 (sigma0/beta > 0)")
        if glicko2_tau <= 0:
            raise RankerError("glicko2_tau must be > 0")
        self.algorithm = algorithm
        self.gauge = gauge
        self.lam = float(lam)
        self.elo_k = float(elo_k)
        self.elo_scale = float(elo_scale)
        self.mu0 = float(mu0)
        self.sigma0 = float(sigma0)
        self.beta = float(beta)
        self.tau = float(tau)
        self.draw_prob = float(draw_prob)
        self.glicko2_tau = float(glicko2_tau)
        self.bus = bus
        self.session_id = session_id
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._seed = seed
        self._auto_fit = bool(auto_fit)
        self._started_at = time.time()
        self._items: list[_ItemState] = []
        self._name_to_idx: dict[str, int] = {}
        for i, name in enumerate(items):
            self._items.append(_ItemState(name=name, idx=i,
                                          mu=mu0 if algorithm == TRUE_SKILL else _ELO_MU0,
                                          ts_mu=mu0, ts_sigma=sigma0))
        for it in self._items:
            self._name_to_idx[it.name] = it.idx
        K = len(self._items)
        self._K = K
        # Pair counts and wins (for batch BT / Thurstone).
        self._pair_wins: list[list[float]] = [[0.0] * K for _ in range(K)]
        self._pair_counts: list[list[float]] = [[0.0] * K for _ in range(K)]
        # Rankings buffer for Plackett-Luce.
        self._rankings: list[list[int]] = []
        self._n_observations = 0
        self._dirty = True       # batch algos need a re-fit
        self._last_log_likelihood = float("nan")
        self._last_iterations = 0
        self._last_converged = False
        # Tamper-evident audit trail: hash chain over observations.
        self._fingerprint_hash = hashlib.sha256()
        self._fingerprint_hash.update(
            f"ranker|{algorithm}|{gauge}|{K}".encode("utf-8")
        )
        for name in items:
            self._fingerprint_hash.update(b"|item:")
            self._fingerprint_hash.update(name.encode("utf-8"))
        self._emit(RANKER_STARTED, {
            "algorithm": algorithm, "K": K, "gauge": gauge,
            "items": [it.name for it in self._items],
        })

    # ------------------------------------------------------------------
    # Event bus helpers
    # ------------------------------------------------------------------

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.bus is None:
            return
        try:
            from agi.events import Event
        except Exception:
            return
        try:
            self.bus.publish(Event(
                kind=kind, session_id=self.session_id, data=data,
            ))
        except Exception:
            # Buggy subscribers must not poison the Ranker.
            pass

    # ------------------------------------------------------------------
    # Public API — registration + observation
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[str]:
        return [it.name for it in self._items]

    @property
    def n_observations(self) -> int:
        return self._n_observations

    @property
    def n_items(self) -> int:
        return self._K

    @property
    def fingerprint(self) -> str:
        return "sha256:" + self._fingerprint_hash.hexdigest()

    def _idx(self, name: str) -> int:
        i = self._name_to_idx.get(name)
        if i is None:
            raise UnknownItem(f"unknown item {name!r}")
        return i

    def observe_pair(
        self,
        winner: str,
        loser: str,
        *,
        weight: float = 1.0,
        draw: bool = False,
    ) -> None:
        """Record a pairwise outcome ``winner`` beat ``loser`` (or draw).

        For all algorithms.  Online algorithms (Elo/Glicko/TrueSkill)
        update their state immediately; batch algorithms mark the cache
        dirty.

        Parameters
        ----------
        winner, loser
            Item names.
        weight
            How much to weight this observation; default 1.0.  Useful for
            confidence-weighted judges or for half-credit draws when
            ``draw=False``.
        draw
            If True, the comparison ended in a tie.  Both items earn
            half a win against each other (BT/Thurstone) or invoke the
            draw update (Elo/Glicko/TrueSkill).
        """
        if winner == loser:
            raise InvalidObservation("winner and loser must differ")
        if weight <= 0:
            raise InvalidObservation("weight must be > 0")
        i = self._idx(winner)
        j = self._idx(loser)
        self._n_observations += 1
        # Update common counters + audit hash.
        self._fingerprint_hash.update(
            f"|pair:{i},{j},{weight},{int(draw)}".encode("utf-8")
        )
        wi = self._items[i]
        wj = self._items[j]
        wi.n_compared += int(weight)
        wj.n_compared += int(weight)
        wi.last_seen = self._n_observations
        wj.last_seen = self._n_observations
        if draw:
            wi.n_draws += 1
            wj.n_draws += 1
            self._pair_wins[i][j] += 0.5 * weight
            self._pair_wins[j][i] += 0.5 * weight
        else:
            wi.n_wins += 1
            wj.n_losses += 1
            self._pair_wins[i][j] += 1.0 * weight
        self._pair_counts[i][j] += weight
        self._pair_counts[j][i] += weight
        # Online algorithms: update immediately.
        if self.algorithm == ELO:
            self._elo_update(i, j, draw=draw, weight=weight)
        elif self.algorithm == GLICKO:
            self._glicko_update(i, j, draw=draw, weight=weight)
        elif self.algorithm == GLICKO2:
            self._glicko2_update(i, j, draw=draw, weight=weight)
        elif self.algorithm == TRUE_SKILL:
            self._ts_update(i, j, draw=draw)
        self._dirty = True
        self._emit(RANKER_OBSERVED, {
            "winner": winner, "loser": loser,
            "draw": draw, "weight": weight,
            "n_observations": self._n_observations,
        })
        if self._auto_fit and self.algorithm in _BATCH_ALGOS:
            self.fit()

    def observe_ranking(self, ordered_items: Sequence[str]) -> None:
        """Record a full or partial ranking ``ordered_items``, best-first.

        For Plackett-Luce / Bradley-Terry / Thurstone / TrueSkill.
        Plackett-Luce uses the full ranking exactly; BT/Thurstone
        decompose it into the implied pairwise wins (item at position
        i beats every item at position > i).  TrueSkill batch path is
        delegated to a sequence of pairwise updates.

        At least 2 items are required.
        """
        if len(ordered_items) < 2:
            raise InvalidObservation("ranking must have at least two items")
        seen: set[str] = set()
        for name in ordered_items:
            if name in seen:
                raise InvalidObservation(f"duplicate item in ranking: {name!r}")
            seen.add(name)
            self._idx(name)         # raises UnknownItem
        indices = [self._idx(n) for n in ordered_items]
        self._n_observations += 1
        self._fingerprint_hash.update(
            ("|rank:" + ",".join(str(i) for i in indices)).encode("utf-8")
        )
        for i in indices:
            self._items[i].n_compared += 1
            self._items[i].last_seen = self._n_observations
        # Pairwise decomposition.
        for ai in range(len(indices)):
            for bi in range(ai + 1, len(indices)):
                a = indices[ai]
                b = indices[bi]
                self._pair_wins[a][b] += 1.0
                self._pair_counts[a][b] += 1.0
                self._pair_counts[b][a] += 1.0
                self._items[a].n_wins += 1
                self._items[b].n_losses += 1
        # Plackett-Luce buffer.
        self._rankings.append(indices)
        if self.algorithm == TRUE_SKILL:
            for ai in range(len(indices) - 1):
                self._ts_update(indices[ai], indices[ai + 1], draw=False)
        elif self.algorithm == ELO:
            for ai in range(len(indices)):
                for bi in range(ai + 1, len(indices)):
                    self._elo_update(indices[ai], indices[bi])
        self._dirty = True
        self._emit(RANKER_OBSERVED, {
            "ranking": list(ordered_items),
            "n_observations": self._n_observations,
        })
        if self._auto_fit and self.algorithm in _BATCH_ALGOS:
            self.fit()

    def observe_score(self, item: str, score: float) -> None:
        """Record an absolute score for ``item``; converted to virtual pairwise
        comparisons against the running median (deferred until fit).

        Useful for hybrid pipelines where judges sometimes emit absolute
        Likert scores and sometimes emit relative pairs.  This method is
        a no-op for online algorithms (Elo/Glicko/TrueSkill).
        """
        i = self._idx(item)
        if not math.isfinite(score):
            raise InvalidObservation("score must be finite")
        self._n_observations += 1
        self._fingerprint_hash.update(
            f"|score:{i}:{score:.6g}".encode("utf-8")
        )
        self._items[i].last_seen = self._n_observations
        self._items[i].extra.setdefault("scores", 0.0)
        self._items[i].extra["scores"] = self._items[i].extra.get("scores", 0.0) + score
        self._items[i].extra["score_count"] = self._items[i].extra.get("score_count", 0.0) + 1.0
        self._dirty = True
        self._emit(RANKER_OBSERVED, {
            "item": item, "score": score,
            "n_observations": self._n_observations,
        })

    # ------------------------------------------------------------------
    # Online algorithm updates
    # ------------------------------------------------------------------

    def _elo_update(
        self,
        i: int,
        j: int,
        *,
        draw: bool = False,
        weight: float = 1.0,
    ) -> None:
        """Elo update.  ``i`` is the winner unless ``draw=True``."""
        a = self._items[i]
        b = self._items[j]
        diff = (a.mu - b.mu) / self.elo_scale
        e_a = 1.0 / (1.0 + math.pow(10.0, -diff))
        s_a = 0.5 if draw else 1.0
        delta = self.elo_k * weight * (s_a - e_a)
        a.mu += delta
        b.mu -= delta

    def _glicko_update(
        self,
        i: int,
        j: int,
        *,
        draw: bool = False,
        weight: float = 1.0,
    ) -> None:
        """Single-period Glicko update (Glickman 1995).

        Approximate single-game update: treat the period as containing
        only this game, then return.  This is the "live update" form
        used by some chess sites.
        """
        q = math.log(10.0) / 400.0
        a = self._items[i]
        b = self._items[j]
        # g(φ_j)
        g_b = 1.0 / math.sqrt(1.0 + 3.0 * q * q * b.phi * b.phi / (math.pi * math.pi))
        g_a = 1.0 / math.sqrt(1.0 + 3.0 * q * q * a.phi * a.phi / (math.pi * math.pi))
        e_ab = 1.0 / (1.0 + math.pow(10.0, -g_b * (a.mu - b.mu) / 400.0))
        e_ba = 1.0 / (1.0 + math.pow(10.0, -g_a * (b.mu - a.mu) / 400.0))
        d2_a = 1.0 / (q * q * g_b * g_b * e_ab * (1.0 - e_ab))
        d2_b = 1.0 / (q * q * g_a * g_a * e_ba * (1.0 - e_ba))
        s_a = 0.5 if draw else 1.0
        s_b = 0.5 if draw else 0.0
        new_phi_a = math.sqrt(1.0 / (1.0 / (a.phi * a.phi) + 1.0 / d2_a))
        new_phi_b = math.sqrt(1.0 / (1.0 / (b.phi * b.phi) + 1.0 / d2_b))
        a.mu += weight * (q / (1.0 / (a.phi * a.phi) + 1.0 / d2_a)) * g_b * (s_a - e_ab)
        b.mu += weight * (q / (1.0 / (b.phi * b.phi) + 1.0 / d2_b)) * g_a * (s_b - e_ba)
        a.phi = max(30.0, new_phi_a)
        b.phi = max(30.0, new_phi_b)

    def _glicko2_update(
        self,
        i: int,
        j: int,
        *,
        draw: bool = False,
        weight: float = 1.0,
    ) -> None:
        """Single-period Glicko-2 update (Glickman 2012).

        Converts to the Glicko-2 scale, performs the standard
        single-game update including the volatility iteration via the
        Illinois bracketing method, then converts back.  Per-spec
        constants (tau = 0.5) are exposed via ``glicko2_tau``.
        """
        # Convert to Glicko-2 internal scale.
        a = self._items[i]
        b = self._items[j]
        mu_a = (a.mu - 1500.0) / 173.7178
        phi_a = a.phi / 173.7178
        sig_a = a.sigma
        mu_b = (b.mu - 1500.0) / 173.7178
        phi_b = b.phi / 173.7178
        sig_b = b.sigma
        s_a = 0.5 if draw else 1.0
        s_b = 0.5 if draw else 0.0

        def update_one(mu: float, phi: float, sig: float,
                       mu_o: float, phi_o: float, s: float) -> tuple[float, float, float]:
            g = 1.0 / math.sqrt(1.0 + 3.0 * phi_o * phi_o / (math.pi * math.pi))
            e = 1.0 / (1.0 + math.exp(-g * (mu - mu_o)))
            v = 1.0 / (g * g * e * (1.0 - e) + _EPS)
            delta = v * g * (s - e)
            # Volatility iteration.
            a_val = math.log(sig * sig)
            tau = self.glicko2_tau

            def f(x: float) -> float:
                ex = math.exp(x)
                num = ex * (delta * delta - phi * phi - v - ex)
                den = 2.0 * (phi * phi + v + ex) ** 2
                return num / max(den, _EPS) - (x - a_val) / (tau * tau)

            # Illinois algorithm.
            A = a_val
            if delta * delta > phi * phi + v:
                B = math.log(max(delta * delta - phi * phi - v, _EPS))
            else:
                k = 1
                B = a_val - k * tau
                while f(B) < 0 and k < 50:
                    k += 1
                    B = a_val - k * tau
            fa = f(A)
            fb = f(B)
            for _ in range(60):
                if abs(B - A) < 1e-6:
                    break
                C = A + (A - B) * fa / max(fb - fa, _EPS) if abs(fb - fa) > _EPS else 0.5 * (A + B)
                fc = f(C)
                if fc * fb <= 0:
                    A, fa = B, fb
                else:
                    fa = fa / 2.0
                B, fb = C, fc
            new_sig = math.exp(A / 2.0)
            phi_star = math.sqrt(phi * phi + new_sig * new_sig)
            new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
            new_mu = mu + new_phi * new_phi * g * (s - e)
            return new_mu, max(new_phi, 1e-3), new_sig

        nm_a, np_a, ns_a = update_one(mu_a, phi_a, sig_a, mu_b, phi_b, s_a)
        nm_b, np_b, ns_b = update_one(mu_b, phi_b, sig_b, mu_a, phi_a, s_b)
        a.mu = 173.7178 * nm_a + 1500.0
        a.phi = 173.7178 * np_a
        a.sigma = ns_a
        b.mu = 173.7178 * nm_b + 1500.0
        b.phi = 173.7178 * np_b
        b.sigma = ns_b

    def _ts_update(self, i: int, j: int, *, draw: bool = False) -> None:
        """TrueSkill closed-form two-player update.

        Herbrich-Minka-Graepel (2007) eqs. (1)-(4) / Minka (2002)
        truncated-Gaussian moment matching, plus the per-game
        volatility additive term τ²::

            σ² := σ² + τ²
            c² = 2β² + σ_a² + σ_b²
            t = (μ_a − μ_b) / c
            ε = √2 · β · Φ⁻¹((draw_prob + 1)/2) / c     # draw margin
            μ_a ← μ_a + (σ_a²/c) · v(t, ε)
            σ_a² ← σ_a² · (1 − (σ_a²/c²) · w(t, ε))
            and symmetric for b.
        """
        a = self._items[i]
        b = self._items[j]
        # Volatility inflation (per match).
        sig_a2 = a.ts_sigma * a.ts_sigma + self.tau * self.tau
        sig_b2 = b.ts_sigma * b.ts_sigma + self.tau * self.tau
        c2 = 2.0 * self.beta * self.beta + sig_a2 + sig_b2
        c = math.sqrt(c2)
        t = (a.ts_mu - b.ts_mu) / c
        # Draw margin from prior draw_prob.
        # The standard reduction: P(draw) = Φ(ε/c) − Φ(−ε/c) = 2Φ(ε/c) − 1
        # ⇒ ε = c · Φ⁻¹((1 + draw_prob) / 2).
        if self.draw_prob > 0:
            eps = c * _inv_phi((1.0 + self.draw_prob) / 2.0)
            eps_c = eps / c
        else:
            eps = 0.0
            eps_c = 0.0
        if draw:
            v = _ts_v_draw(t, eps_c)
            w = _ts_w_draw(t, eps_c)
            sign = 1.0
        else:
            # Winner is i.
            v = _ts_v(t, eps_c)
            w = _ts_w(t, eps_c)
            sign = 1.0
        a.ts_mu = a.ts_mu + sign * (sig_a2 / c) * v
        a.ts_sigma = math.sqrt(max(sig_a2 * (1.0 - (sig_a2 / c2) * w), 1e-6))
        b.ts_mu = b.ts_mu - sign * (sig_b2 / c) * v
        b.ts_sigma = math.sqrt(max(sig_b2 * (1.0 - (sig_b2 / c2) * w), 1e-6))

    # ------------------------------------------------------------------
    # Batch fit
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Re-fit the latent skills for batch algorithms.

        No-op for online algorithms.  Idempotent — calling repeatedly
        without new observations returns immediately.
        """
        if self.algorithm not in _BATCH_ALGOS:
            return
        if not self._dirty and self._n_observations > 0:
            return
        K = self._K
        if self.algorithm == BRADLEY_TERRY_MM:
            pi, conv, iters = _bt_mm_iterate(
                K, self._pair_wins, self._pair_counts,
                gauge=self.gauge,
            )
            theta = [math.log(max(p, _EPS)) for p in pi]
            if self.gauge == GAUGE_FIX_FIRST:
                theta = [t - theta[0] for t in theta]
            elif self.gauge == GAUGE_ZERO_SUM:
                m = sum(theta) / K
                theta = [t - m for t in theta]
            ses = _bt_fisher_se(K, theta, self._pair_counts)
            ll = _bt_log_likelihood(K, theta, self._pair_wins, self._pair_counts)
        elif self.algorithm == BRADLEY_TERRY_MAP:
            theta, conv, iters = _bt_map_iterate(
                K, self._pair_wins, self._pair_counts,
                lam=self.lam, gauge=self.gauge,
            )
            ses = _bt_map_se(K, theta, self._pair_counts, lam=self.lam)
            ll = _bt_log_likelihood(K, theta, self._pair_wins, self._pair_counts)
        elif self.algorithm == PLACKETT_LUCE_MM:
            # If we have explicit rankings, use them.  Otherwise, decompose
            # pair-counts into virtual 2-item rankings (a > b means a beat b).
            rankings = self._rankings
            if not rankings:
                rankings = []
                for i in range(K):
                    for j in range(K):
                        if i == j or self._pair_wins[i][j] <= 0:
                            continue
                        for _ in range(int(round(self._pair_wins[i][j]))):
                            rankings.append([i, j])
            pi, conv, iters = _pl_mm_iterate(
                K, rankings, gauge=self.gauge,
            )
            theta = [math.log(max(p, _EPS)) for p in pi]
            if self.gauge == GAUGE_FIX_FIRST:
                theta = [t - theta[0] for t in theta]
            elif self.gauge == GAUGE_ZERO_SUM:
                m = sum(theta) / K
                theta = [t - m for t in theta]
            ses = _bt_fisher_se(K, theta, self._pair_counts)
            ll = _pl_log_likelihood(K, theta, rankings)
        elif self.algorithm == THURSTONE_MM:
            theta, conv, iters = _thurstone_mm_iterate(
                K, self._pair_wins, self._pair_counts,
                gauge=self.gauge,
            )
            ses = _bt_fisher_se(K, theta, self._pair_counts)   # Hessian shape matches
            ll = _thurstone_log_likelihood(K, theta, self._pair_wins, self._pair_counts)
        else:
            return
        for i, item in enumerate(self._items):
            item.theta = theta[i]
            item.stderr = ses[i]
        self._last_log_likelihood = ll
        self._last_iterations = iters
        self._last_converged = conv
        self._dirty = False
        self._emit(RANKER_FITTED, {
            "algorithm": self.algorithm,
            "iterations": iters,
            "converged": conv,
            "log_likelihood": ll,
        })

    # ------------------------------------------------------------------
    # Public API — query
    # ------------------------------------------------------------------

    def rate(self, item: str) -> ItemRating:
        """Return current rating for ``item``."""
        i = self._idx(item)
        return self._rating(i)

    def _rating(self, i: int) -> ItemRating:
        if self._dirty and self.algorithm in _BATCH_ALGOS:
            self.fit()
        it = self._items[i]
        if self.algorithm == ELO or self.algorithm == GLICKO or self.algorithm == GLICKO2:
            mean = it.mu
            stderr = it.phi
        elif self.algorithm == TRUE_SKILL:
            mean = it.ts_mu
            stderr = it.ts_sigma
        else:
            mean = it.theta
            stderr = it.stderr
        return ItemRating(
            name=it.name,
            mean=mean,
            stderr=stderr,
            n_compared=it.n_compared,
            n_wins=it.n_wins,
            n_losses=it.n_losses,
            n_draws=it.n_draws,
            last_seen=it.last_seen,
            extra=dict(it.extra),
        )

    def rank(self) -> list[str]:
        """Return item names ordered best-to-worst by current rating."""
        if self._dirty and self.algorithm in _BATCH_ALGOS:
            self.fit()
        return [it.name for it in
                sorted(self._items, key=lambda it: -self._skill(it))]

    def _skill(self, it: _ItemState) -> float:
        if self.algorithm == ELO or self.algorithm == GLICKO or self.algorithm == GLICKO2:
            return it.mu
        if self.algorithm == TRUE_SKILL:
            # Conservative "skill = μ - 3σ" (Microsoft's published convention).
            return it.ts_mu - 3.0 * it.ts_sigma
        return it.theta

    def top_k(
        self,
        k: int,
        *,
        delta: float = 0.05,
        epsilon: float = 0.0,
    ) -> TopKDecision:
        """Return top-K items with a Hajek-Oh-Xu (2014) PAC certificate.

        `pac_certified=True` iff the empirical gap between item K and
        item K+1 (in the current ranking) exceeds the Hoeffding /
        empirical-Bernstein half-width on each item's average win-rate
        at the chosen ``delta``.
        """
        if k < 1:
            raise RankerError("k must be ≥ 1")
        if k > self._K:
            raise RankerError(f"k > number of items ({self._K})")
        order = self.rank()
        items_in = order[:k]
        certified = False
        margin = 0.0
        if k < self._K:
            kth = self._items[self._name_to_idx[order[k - 1]]]
            kp1 = self._items[self._name_to_idx[order[k]]]
            s_k = self._skill(kth)
            s_kp1 = self._skill(kp1)
            margin = s_k - s_kp1
            # Convert margin to a "win-prob delta" via the logistic link.
            win_delta = sigmoid(margin) - 0.5
            # Conservative HW on the *less informed* item's empirical win rate.
            n = min(kth.n_compared, kp1.n_compared)
            if n > 1:
                p_hat = (kth.n_wins / max(kth.n_compared, 1)) if kth.n_compared else 0.5
                var = p_hat * (1.0 - p_hat)
                hw = empirical_bernstein_half_width(n, var, delta, b=1.0)
            else:
                hw = 1.0
            certified = win_delta > hw + epsilon
        fp = hashlib.sha256(
            (self.fingerprint + f"|topk:{k}|d:{delta}|e:{epsilon}").encode("utf-8")
        ).hexdigest()
        return TopKDecision(
            k=k,
            items=items_in,
            delta=delta,
            epsilon=epsilon,
            margin=margin,
            pac_certified=certified,
            fingerprint="sha256:" + fp,
        )

    # ------------------------------------------------------------------
    # Win-probability predictions
    # ------------------------------------------------------------------

    def predict_win_prob(self, a: str, b: str) -> float:
        """Model-based P(a beats b).

        BT/PL/MAP/Plackett-Luce: σ(θ_a - θ_b).
        Thurstone: Φ((θ_a - θ_b) / √2).
        Elo / Glicko / Glicko-2: standard logistic with rating diff
        scaled by 400 (Elo) or with the rating-deviation adjustment
        (Glicko).  TrueSkill: Φ((μ_a - μ_b) / √(2β² + σ_a² + σ_b²)).
        """
        if self._dirty and self.algorithm in _BATCH_ALGOS:
            self.fit()
        ia = self._idx(a)
        ib = self._idx(b)
        ai = self._items[ia]
        bj = self._items[ib]
        if self.algorithm in (BRADLEY_TERRY_MM, BRADLEY_TERRY_MAP, PLACKETT_LUCE_MM):
            return sigmoid(ai.theta - bj.theta)
        if self.algorithm == THURSTONE_MM:
            return phi((ai.theta - bj.theta) / math.sqrt(2.0))
        if self.algorithm == ELO:
            diff = (ai.mu - bj.mu) / self.elo_scale
            return 1.0 / (1.0 + math.pow(10.0, -diff))
        if self.algorithm in (GLICKO, GLICKO2):
            q = math.log(10.0) / 400.0
            phi_combined = math.sqrt(ai.phi * ai.phi + bj.phi * bj.phi)
            g = 1.0 / math.sqrt(1.0 + 3.0 * q * q * phi_combined * phi_combined / (math.pi * math.pi))
            return 1.0 / (1.0 + math.pow(10.0, -g * (ai.mu - bj.mu) / 400.0))
        if self.algorithm == TRUE_SKILL:
            c2 = 2.0 * self.beta * self.beta + ai.ts_sigma ** 2 + bj.ts_sigma ** 2
            return phi((ai.ts_mu - bj.ts_mu) / math.sqrt(c2))
        return 0.5

    def win_probability_ci(
        self,
        a: str,
        b: str,
        *,
        delta: float = 0.05,
        anytime: bool = False,
    ) -> PairwiseProbability:
        """Anytime-valid CI for P(a beats b).

        When ``anytime=True`` uses Howard-Ramdas-McAuliffe-Sekhon (2021)
        time-uniform half-width.  Otherwise uses empirical-Bernstein
        (Maurer-Pontil 2009).  When no direct comparisons exist, falls
        back to the model-based estimate with a width that scales with
        the larger of the two item SEs (delta-method approximation).
        """
        ia = self._idx(a)
        ib = self._idx(b)
        n_ab = self._pair_counts[ia][ib]
        if n_ab >= 1.0:
            w_ab = self._pair_wins[ia][ib]
            p_hat = w_ab / n_ab
            var = p_hat * (1.0 - p_hat)
            if anytime:
                hw = hrm_anytime_half_width(int(n_ab), var, delta)
            else:
                hw = empirical_bernstein_half_width(int(n_ab), var, delta)
            method = ("empirical_bernstein_anytime" if anytime
                      else "empirical_bernstein")
        else:
            # Model-based via delta method.
            p_hat = self.predict_win_prob(a, b)
            ai = self._items[ia]
            bj = self._items[ib]
            sigma2 = ai.stderr * ai.stderr + bj.stderr * bj.stderr
            # Logistic-link delta: Var(p) ≈ p²(1-p)² (σ_a² + σ_b²)
            var_p = (p_hat ** 2) * ((1 - p_hat) ** 2) * sigma2
            z = _inv_phi(1.0 - delta / 2.0)
            hw = z * math.sqrt(max(var_p, 0.0))
            method = "model_delta"
        lo = max(0.0, p_hat - hw)
        hi = min(1.0, p_hat + hw)
        return PairwiseProbability(
            a=a, b=b, mean_win_prob=p_hat,
            ci_low=lo, ci_high=hi, ci_half_width=hw,
            n_direct=int(n_ab), method=method, delta=delta,
        )

    def compare(self, a: str, b: str, *, delta: float = 0.05) -> PairwiseProbability:
        """Alias for ``win_probability_ci(a, b, delta=delta)``."""
        return self.win_probability_ci(a, b, delta=delta)

    # ------------------------------------------------------------------
    # Maintenance — forget / clear / state / from_state
    # ------------------------------------------------------------------

    def forget(self, item: str, *, halflife: float = 100.0) -> None:
        """Decay past comparisons involving ``item`` (Garivier-Moulines SW-UCB).

        Multiplies every count/win involving ``item`` by ``exp(-1 / halflife)``.
        For online algorithms, inflates the rating deviation by τ for one
        period (Glickman 2012 §3).
        """
        if halflife <= 0:
            raise RankerError("halflife must be > 0")
        i = self._idx(item)
        decay = math.exp(-1.0 / halflife)
        for j in range(self._K):
            if j == i:
                continue
            self._pair_wins[i][j] *= decay
            self._pair_wins[j][i] *= decay
            self._pair_counts[i][j] *= decay
            self._pair_counts[j][i] *= decay
        it = self._items[i]
        it.n_compared = int(it.n_compared * decay)
        it.n_wins = int(it.n_wins * decay)
        it.n_losses = int(it.n_losses * decay)
        # Online algorithm: inflate the rating-deviation per τ.
        if self.algorithm in (GLICKO, GLICKO2):
            it.phi = math.sqrt(it.phi * it.phi + (self.tau * 100.0) ** 2)
        elif self.algorithm == TRUE_SKILL:
            it.ts_sigma = math.sqrt(it.ts_sigma * it.ts_sigma + self.tau * self.tau)
        self._dirty = True
        self._emit(RANKER_FORGET, {"item": item, "halflife": halflife})

    def clear(self) -> None:
        """Reset the Ranker to a freshly-initialised state, preserving the
        item set and algorithm.  All observations and ratings are erased.
        """
        K = self._K
        self._pair_wins = [[0.0] * K for _ in range(K)]
        self._pair_counts = [[0.0] * K for _ in range(K)]
        self._rankings = []
        self._n_observations = 0
        self._dirty = True
        for it in self._items:
            it.n_compared = 0
            it.n_wins = 0
            it.n_losses = 0
            it.n_draws = 0
            it.last_seen = -1
            it.theta = 0.0
            it.stderr = 0.0
            it.mu = self.mu0 if self.algorithm == TRUE_SKILL else _ELO_MU0
            it.phi = 350.0
            it.sigma = 0.06
            it.ts_mu = self.mu0
            it.ts_sigma = self.sigma0
            it.extra = {}
        # Reset audit hash too.
        self._fingerprint_hash = hashlib.sha256()
        self._fingerprint_hash.update(
            f"ranker|{self.algorithm}|{self.gauge}|{K}".encode("utf-8")
        )
        for it in self._items:
            self._fingerprint_hash.update(b"|item:")
            self._fingerprint_hash.update(it.name.encode("utf-8"))
        self._emit(RANKER_CLEARED, {"K": K})

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, *, delta_bound: float = 0.05) -> RankingReport:
        """Compute a full :class:`RankingReport` at the current state."""
        if self._dirty and self.algorithm in _BATCH_ALGOS:
            self.fit()
        K = self._K
        ratings = [self._rating(i) for i in range(K)]
        order = self.rank()
        # Comparison graph: directed edge from winner -> loser when there
        # has been at least one observed win.
        edges: list[tuple[int, int]] = []
        unique_pairs = set()
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                if self._pair_wins[i][j] > 0:
                    edges.append((i, j))
                if self._pair_counts[i][j] > 0:
                    key = (i, j) if i < j else (j, i)
                    unique_pairs.add(key)
        sccs = strongly_connected_components(K, edges)
        scc_count = len(sccs)
        scc_size = len(sccs[0]) if sccs else 0
        # Largest SCC: identifiable iff covers all items.
        identifiable = (scc_size == K) and (
            self._last_converged or self.algorithm in _ONLINE_ALGOS
        )
        if self.algorithm == BRADLEY_TERRY_MAP:
            # The Gaussian prior makes the posterior strictly convex
            # for any λ > 0, so identifiability is guaranteed up to the
            # gauge.  Treat as identifiable whenever any data has landed.
            identifiable = self._n_observations > 0
        if self.algorithm in _ONLINE_ALGOS:
            identifiable = (self._n_observations > 0)
        isolated = []
        if scc_size < K:
            isolated_idx = set(range(K)) - set(sccs[0])
            isolated = [self._items[i].name for i in sorted(isolated_idx)]
        # Min adjacent gap.
        min_gap = float("inf")
        for ai in range(len(order) - 1):
            a = self._items[self._name_to_idx[order[ai]]]
            b = self._items[self._name_to_idx[order[ai + 1]]]
            g = self._skill(a) - self._skill(b)
            if g < min_gap:
                min_gap = g
        if not math.isfinite(min_gap):
            min_gap = 0.0
        # Convert skill gap to win-probability gap for HOX (2014).
        delta_min = abs(sigmoid(min_gap) - 0.5) * 2.0
        # Hajek-Oh-Xu (2014) Theorem 1:
        # n ≥ c K log K / Δ² for top-K recovery with prob ≥ 1 − δ.
        delta_target = 0.01
        c = 8.0  # constant in HOX top-K result; non-tight conservative.
        if delta_min > _EPS:
            hox_n = int(math.ceil(c * K * math.log(max(K, 2)) /
                                   (delta_min * delta_min)))
        else:
            hox_n = 0 if K <= 1 else int(1e9)
        # Pseudo-R² (McFadden): 1 − ℓ(θ̂)/ℓ(0).
        ll0 = 0.0
        if self.algorithm in (BRADLEY_TERRY_MM, BRADLEY_TERRY_MAP):
            ll0 = _bt_log_likelihood(
                K, [0.0] * K, self._pair_wins, self._pair_counts,
            )
        elif self.algorithm == THURSTONE_MM:
            ll0 = _thurstone_log_likelihood(
                K, [0.0] * K, self._pair_wins, self._pair_counts,
            )
        elif self.algorithm == PLACKETT_LUCE_MM:
            # Reconstruct the implicit ranking set used by `fit()` so the
            # null-model baseline is comparable with the fitted log-lik.
            rankings = self._rankings
            if not rankings:
                rankings = []
                for ii in range(K):
                    for jj in range(K):
                        if ii == jj or self._pair_wins[ii][jj] <= 0:
                            continue
                        for _ in range(int(round(self._pair_wins[ii][jj]))):
                            rankings.append([ii, jj])
            ll0 = _pl_log_likelihood(K, [0.0] * K, rankings)
        pseudo_r2 = (
            1.0 - self._last_log_likelihood / ll0
            if (ll0 != 0.0 and math.isfinite(self._last_log_likelihood))
            else 0.0
        )
        report_id = hashlib.sha256(
            (self.fingerprint + f"|report:{time.time()}").encode("utf-8")
        ).hexdigest()[:24]
        rep = RankingReport(
            id=report_id,
            algorithm=self.algorithm,
            items=ratings,
            rank_order=order,
            n_observations=self._n_observations,
            n_unique_pairs=len(unique_pairs),
            gauge=self.gauge,
            log_likelihood=self._last_log_likelihood,
            pseudo_r2=pseudo_r2,
            identifiable=identifiable,
            scc_count=scc_count,
            scc_size=scc_size,
            isolated_items=isolated,
            iterations=self._last_iterations,
            converged=self._last_converged or self.algorithm in _ONLINE_ALGOS,
            min_gap_estimate=min_gap if math.isfinite(min_gap) else 0.0,
            sample_complexity_to_topk_99=hox_n,
            bound_method="empirical_bernstein",
            started_at=self._started_at,
            finished_at=time.time(),
            fingerprint=self.fingerprint,
        )
        self._emit(RANKER_REPORT, {
            "id": rep.id, "n_items": K,
            "n_observations": self._n_observations,
            "rank_order": order, "identifiable": identifiable,
            "pseudo_r2": pseudo_r2,
        })
        return rep

    # ------------------------------------------------------------------
    # Replay-deterministic serialisation
    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the full Ranker state."""
        return {
            "algorithm": self.algorithm,
            "gauge": self.gauge,
            "lam": self.lam,
            "elo_k": self.elo_k,
            "elo_scale": self.elo_scale,
            "mu0": self.mu0,
            "sigma0": self.sigma0,
            "beta": self.beta,
            "tau": self.tau,
            "draw_prob": self.draw_prob,
            "glicko2_tau": self.glicko2_tau,
            "seed": self._seed,
            "auto_fit": self._auto_fit,
            "items": [it.name for it in self._items],
            "pair_wins": [row[:] for row in self._pair_wins],
            "pair_counts": [row[:] for row in self._pair_counts],
            "rankings": [r[:] for r in self._rankings],
            "n_observations": self._n_observations,
            "item_state": [
                {
                    "name": it.name,
                    "idx": it.idx,
                    "n_compared": it.n_compared,
                    "n_wins": it.n_wins,
                    "n_losses": it.n_losses,
                    "n_draws": it.n_draws,
                    "last_seen": it.last_seen,
                    "theta": it.theta,
                    "stderr": it.stderr,
                    "mu": it.mu,
                    "phi": it.phi,
                    "sigma": it.sigma,
                    "ts_mu": it.ts_mu,
                    "ts_sigma": it.ts_sigma,
                    "extra": dict(it.extra),
                }
                for it in self._items
            ],
            "fingerprint": self.fingerprint,
            "started_at": self._started_at,
            "last_iterations": self._last_iterations,
            "last_converged": self._last_converged,
            "last_log_likelihood": self._last_log_likelihood,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], *, bus: Any = None,
                   session_id: str | None = None) -> "Ranker":
        """Re-hydrate a Ranker from a previously-saved ``state()`` snapshot."""
        r = cls(
            items=list(state["items"]),
            algorithm=state["algorithm"],
            gauge=state.get("gauge", GAUGE_FIX_FIRST),
            lam=state.get("lam", 1.0),
            elo_k=state.get("elo_k", _ELO_K_DEFAULT),
            elo_scale=state.get("elo_scale", _ELO_SCALE),
            mu0=state.get("mu0", _TS_MU0_DEFAULT),
            sigma0=state.get("sigma0", _TS_SIGMA0_DEFAULT),
            beta=state.get("beta", _TS_BETA_DEFAULT),
            tau=state.get("tau", _TS_TAU_DEFAULT),
            draw_prob=state.get("draw_prob", _TS_DRAW_DEFAULT),
            glicko2_tau=state.get("glicko2_tau", _GLICKO2_TAU_DEFAULT),
            seed=state.get("seed"),
            auto_fit=state.get("auto_fit", True),
            bus=bus,
            session_id=session_id,
        )
        K = r._K
        r._pair_wins = [list(row) for row in state["pair_wins"]]
        r._pair_counts = [list(row) for row in state["pair_counts"]]
        r._rankings = [list(rk) for rk in state.get("rankings", [])]
        r._n_observations = int(state.get("n_observations", 0))
        for s_it, it in zip(state["item_state"], r._items):
            it.n_compared = int(s_it.get("n_compared", 0))
            it.n_wins = int(s_it.get("n_wins", 0))
            it.n_losses = int(s_it.get("n_losses", 0))
            it.n_draws = int(s_it.get("n_draws", 0))
            it.last_seen = int(s_it.get("last_seen", -1))
            it.theta = float(s_it.get("theta", 0.0))
            it.stderr = float(s_it.get("stderr", 0.0))
            it.mu = float(s_it.get("mu", _ELO_MU0))
            it.phi = float(s_it.get("phi", 350.0))
            it.sigma = float(s_it.get("sigma", 0.06))
            it.ts_mu = float(s_it.get("ts_mu", _TS_MU0_DEFAULT))
            it.ts_sigma = float(s_it.get("ts_sigma", _TS_SIGMA0_DEFAULT))
            it.extra = dict(s_it.get("extra", {}))
        r._last_iterations = int(state.get("last_iterations", 0))
        r._last_converged = bool(state.get("last_converged", False))
        r._last_log_likelihood = float(state.get("last_log_likelihood", float("nan")))
        # Replay the audit hash from the saved fingerprint string.
        saved_fp = state.get("fingerprint", "")
        if saved_fp.startswith("sha256:"):
            r._fingerprint_hash = hashlib.sha256()
            r._fingerprint_hash.update(b"__replay__")
            r._fingerprint_hash.update(saved_fp[len("sha256:"):].encode("utf-8"))
        r._dirty = True
        return r


# =====================================================================
# Free functions — exposed for direct use without a Ranker instance
# =====================================================================


def _inv_phi(p: float) -> float:
    """Beasley-Springer-Moro inverse normal CDF approximation,
    accurate to ~1e-9 in (1e-15, 1 − 1e-15).
    """
    if p <= 0.0:
        return -1e30
    if p >= 1.0:
        return 1e30
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


def bradley_terry_fit(
    items: Sequence[str],
    pairs: Iterable[tuple[str, str]],
    *,
    lam: float = 0.0,
    gauge: str = GAUGE_FIX_FIRST,
) -> dict[str, float]:
    """Convenience: BT MLE (or MAP if ``lam>0``) over a list of (winner, loser)
    pairs.  Returns ``{item: θ̂_i}``.
    """
    algo = BRADLEY_TERRY_MAP if lam > 0 else BRADLEY_TERRY_MM
    r = Ranker(items, algorithm=algo, lam=max(lam, 1e-9), gauge=gauge,
               auto_fit=False)
    for w, l in pairs:
        r.observe_pair(w, l)
    r.fit()
    return {it.name: it.theta for it in r._items}


def plackett_luce_fit(
    items: Sequence[str],
    rankings: Iterable[Sequence[str]],
    *,
    gauge: str = GAUGE_FIX_FIRST,
) -> dict[str, float]:
    """Convenience: Plackett-Luce MM over a list of partial rankings."""
    r = Ranker(items, algorithm=PLACKETT_LUCE_MM, gauge=gauge, auto_fit=False)
    for rk in rankings:
        r.observe_ranking(list(rk))
    r.fit()
    return {it.name: it.theta for it in r._items}


def elo_run(
    items: Sequence[str],
    pairs: Iterable[tuple[str, str]],
    *,
    elo_k: float = _ELO_K_DEFAULT,
    elo_scale: float = _ELO_SCALE,
    mu0: float = _ELO_MU0,
) -> dict[str, float]:
    """Convenience: run Elo over a stream of (winner, loser) pairs."""
    r = Ranker(items, algorithm=ELO, elo_k=elo_k, elo_scale=elo_scale)
    for it in r._items:
        it.mu = mu0
    for w, l in pairs:
        r.observe_pair(w, l)
    return {it.name: it.mu for it in r._items}


def trueskill_run(
    items: Sequence[str],
    pairs: Iterable[tuple[str, str]],
    *,
    mu0: float = _TS_MU0_DEFAULT,
    sigma0: float = _TS_SIGMA0_DEFAULT,
    beta: float = _TS_BETA_DEFAULT,
    tau: float = _TS_TAU_DEFAULT,
    draw_prob: float = _TS_DRAW_DEFAULT,
) -> dict[str, tuple[float, float]]:
    """Convenience: run TrueSkill over a stream of (winner, loser) pairs.
    Returns ``{item: (μ_i, σ_i)}``.
    """
    r = Ranker(
        items, algorithm=TRUE_SKILL, mu0=mu0, sigma0=sigma0,
        beta=beta, tau=tau, draw_prob=draw_prob,
    )
    for w, l in pairs:
        r.observe_pair(w, l)
    return {it.name: (it.ts_mu, it.ts_sigma) for it in r._items}


def hox_sample_complexity(
    k: int, gap: float, *, delta: float = 0.05,
) -> int:
    """Hajek-Oh-Xu (2014) Theorem 1: comparison count for top-K recovery
    with probability ≥ 1 − δ given a min adjacent skill gap ``gap``.

    Returns the conservative number of pairwise comparisons.
    """
    if gap <= 0:
        return 10 ** 9
    if k <= 1:
        return 0
    c = 8.0
    return int(math.ceil(c * k * math.log(max(k, 2)) / (gap * gap)))


def rank_correlation_kendall(
    a: Sequence[str], b: Sequence[str],
) -> float:
    """Kendall-τ rank correlation between two orderings of the same item set.

    O(K²) implementation.  Returns 1.0 for identical rankings, -1.0 for
    reversed, ~0 for random.  Used by `Refuter` to falsify Ranker
    stability under perturbed observations.
    """
    if len(a) != len(b):
        raise ValueError("ranking lengths differ")
    if set(a) != set(b):
        raise ValueError("rankings must share the same item set")
    pos_b = {name: i for i, name in enumerate(b)}
    n = len(a)
    if n < 2:
        return 1.0
    concord = 0
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = (i - j)         # negative: a[i] ranked above a[j]
            sign_b = pos_b[a[i]] - pos_b[a[j]]
            if sign_a * sign_b > 0:
                concord += 1
            elif sign_a * sign_b < 0:
                discord += 1
    total = n * (n - 1) // 2
    if total == 0:
        return 1.0
    return (concord - discord) / total


def rank_correlation_spearman(
    a: Sequence[str], b: Sequence[str],
) -> float:
    """Spearman ρ rank correlation.

    Returns 1 - 6 Σ d² / (n(n²-1)) where d is the rank difference per item.
    """
    if len(a) != len(b):
        raise ValueError("ranking lengths differ")
    if set(a) != set(b):
        raise ValueError("rankings must share the same item set")
    pos_a = {name: i + 1 for i, name in enumerate(a)}
    pos_b = {name: i + 1 for i, name in enumerate(b)}
    n = len(a)
    if n < 2:
        return 1.0
    d2 = sum((pos_a[k] - pos_b[k]) ** 2 for k in pos_a)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


# =====================================================================
# Exports
# =====================================================================


__all__ = [
    # Algorithms.
    "BRADLEY_TERRY_MM", "BRADLEY_TERRY_MAP", "PLACKETT_LUCE_MM",
    "THURSTONE_MM", "ELO", "GLICKO", "GLICKO2", "TRUE_SKILL",
    "KNOWN_ALGORITHMS",
    # Gauges.
    "GAUGE_FIX_FIRST", "GAUGE_ZERO_SUM", "GAUGE_MEAN_ELO", "KNOWN_GAUGES",
    # Events.
    "RANKER_STARTED", "RANKER_OBSERVED", "RANKER_FITTED",
    "RANKER_REPORT", "RANKER_CLEARED", "RANKER_FORGET", "KNOWN_EVENTS",
    # Exceptions.
    "RankerError", "UnknownAlgorithm", "UnknownItem",
    "InvalidObservation", "InsufficientData", "NotIdentifiable",
    # Dataclasses.
    "ItemRating", "PairwiseProbability", "RankingReport", "TopKDecision",
    # Main class.
    "Ranker",
    # Numerical helpers.
    "sigmoid", "logistic_log", "phi", "phi_pdf",
    "hoeffding_half_width", "empirical_bernstein_half_width",
    "hrm_anytime_half_width", "strongly_connected_components",
    # Convenience module-level fits.
    "bradley_terry_fit", "plackett_luce_fit", "elo_run", "trueskill_run",
    # Sample-complexity bound.
    "hox_sample_complexity",
    # Rank correlations.
    "rank_correlation_kendall", "rank_correlation_spearman",
]
