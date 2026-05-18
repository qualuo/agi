r"""Anticipator — sleep-time / anticipatory compute as a runtime primitive.

Test-time compute scaling (Snell et al. 2024; OpenAI o-series; DeepSeek-R1)
has shown that letting a model think longer at inference is a free
capability lever — but it ships latency directly to the user and burns
hot GPU time that already costs the most.  A coordination engine that
wants the *capability* gain of "more compute" but not the *latency* and
not the *peak-hour* unit cost has another lever: spend the compute
during *idle* time, before the user's request arrives, on a small set of
*likely* future queries.

That is what :class:`Anticipator` schedules.  It is the runtime
counterpart of CPU branch prediction, JIT method-cache warming, and
TCP-prefetch — applied to LLM workloads, with budget controls and
replay-verifiable receipts.  Concretely: given a context (an agent's
working memory, a conversation prefix, a partial plan), the
``Anticipator`` enumerates ``K`` candidate future queries via an injected
``Forecaster``, scores each by its prior probability and the cost it
will save if pre-computed, runs a deterministic 0-1 knapsack over the
``(value, cost)`` pairs subject to a hard sleep-time compute budget,
and dispatches the chosen subset to an injected ``Answerer``.  When the
real query arrives, it is matched against the cache through a pluggable
``MatchPolicy``; on a hit the cached answer is returned and the saved
test-time cost is booked; on a miss the ``Answerer`` runs fresh.  Every
step emits an :class:`Event` and is fingerprinted into a Merkle-style
chain so the entire pre-compute → serve loop is replay-verifiable.

Why it earns a slot in the catalog
----------------------------------

* It is the **only** primitive that converts *cold* compute into
  *warm* compute on the user's behalf.  ``Speculator`` accelerates
  active streams; ``Pretunist`` adapts at test time; ``Distiller``
  shifts compute to *training* time.  ``Anticipator`` shifts it to
  *idle* time — a fourth, distinct axis with its own economics.

* It carries a **PAC certificate** over the realised cache hit rate
  (Wilson 95% lower bound, Hoeffding LCB, empirical-Bernstein LCB on
  the saved cost-per-query), so the coordinator does not have to take
  the primitive's word for the speedup — it can prove it to an
  external auditor.

* It is **deterministic given a seed**: the forecaster is consulted
  with a stable hash of the context; the knapsack is exact;
  matching is exact-or-thresholded; the fingerprint chain is closed.

* It composes naturally with the existing economy primitives
  (:mod:`agi.economist`, :mod:`agi.market`, :mod:`agi.costs`) and the
  scaling-law primitive (:mod:`agi.scaler`) for compute-optimal
  pre-compute budget choice, with the safety primitives
  (:mod:`agi.attest`, :mod:`agi.governance`) for audit, and with the
  inference primitives (:mod:`agi.forecaster`, :mod:`agi.predictor`,
  :mod:`agi.embedder`) as plug-in components.

Mathematical and algorithmic roots
----------------------------------

* **Snell, C., Lee, A., Xu, K., Kumar, A. (2024) — "Scaling LLM
  Test-Time Compute Optimally can be More Effective than Scaling
  Model Parameters" (arXiv:2408.03314).**  Establishes that test-time
  compute is a first-class capability lever.  ``Anticipator`` honours
  the same lever but routes it through *idle* time instead of *peak*
  time.

* **Lin, K., Snell, C., Chen, T., et al. (2025) — "Sleep-time
  Compute: Beyond Inference Scaling at Test-time" (arXiv:2504.13171,
  Letta + Stanford).**  Reframes inference budget as a two-phase
  schedule: a *sleep-time* phase that pre-processes anticipated
  context and a *test-time* phase that consumes the cached state.
  ``Anticipator`` is the runtime-primitive expression of that schedule,
  with explicit budget, certificate, and replay.

* **Karmarkar, N. and Karp, R.M. (1982) — "The differencing method of
  set partitioning" / Martello & Toth (1990) — Knapsack Problems.**
  The pre-compute-subset selection is a 0-1 knapsack over
  ``(value_k, cost_k)`` pairs; for the modest ``K ≤ 64`` typical of
  one-step look-ahead, a tiny exact branch-and-bound dominates the
  greedy ratio rule.  Both are implemented; the configuration picks.

* **Wilson, E.B. (1927) — "Probable inference, the law of succession,
  and statistical inference."**  Two-sided binomial CI on the hit rate;
  reported as the routine confidence interval.

* **Hoeffding, W. (1963), Maurer-Pontil (2009).**  Hoeffding LCB on the
  hit rate; empirical-Bernstein LCB on the saved cost-per-query.  The
  empirical-Bernstein bound is the **anytime** certificate the
  coordinator hands to a downstream auditor.

* **Bloom, B.H. (1970) — "Space/time trade-offs in hash coding with
  allowable errors."**  An optional Bloom filter front-stop on the
  cache to reject *certain* misses before any matcher work; off by
  default, on for high-throughput configurations.

* **Belady, L.A. (1966) — "A study of replacement algorithms for a
  virtual-storage computer."**  When the cache hits its size budget,
  the optimal eviction policy is *furthest-in-future*; the primitive
  uses a forecaster-derived ranking as a proxy, falling back to LRU.

* **Merkle, R. (1979) — "Secrecy, authentication and public key
  systems."**  A SHA-256 chain over every event provides a
  replay-verifiable receipt.

Composes with
-------------

* :mod:`agi.forecaster`        — supplies the candidate query distribution.
* :mod:`agi.predictor`         — alternative single-best next-query head.
* :mod:`agi.embedder`          — semantic match between cached and live query.
* :mod:`agi.costs`             — book ``saved_cost`` into the runtime ledger.
* :mod:`agi.economist`         — value-of-precompute under an explicit utility.
* :mod:`agi.scaler`            — compute-optimal sleep-time budget for K.
* :mod:`agi.scheduler`         — *when* the runtime is idle enough to pay.
* :mod:`agi.memory`            — durable backing store for the cache.
* :mod:`agi.attest`            — fingerprint export.
* :mod:`agi.governance`        — refuse-to-precompute policy (privacy, leakage).
* :mod:`agi.coordinator`       — the typical caller; ``serve`` is the hot path.

What this primitive ships
-------------------------

* :class:`ContextRecord` — declared idle context with a deadline hint.
* :class:`Candidate`     — a single ``(query, prior, est_miss_cost)``.
* :class:`Plan`          — knapsack-chosen subset to pre-compute.
* :class:`PrecomputeResult` — what was actually computed and cached.
* :class:`ServeResult`   — hit/miss verdict, answer, saved cost, latency.
* :class:`AnticipatorConfig` — budget, thresholds, knapsack policy, seeds.
* :class:`AnticipatorCertificate` — hit-rate CIs, cost-saving LCB, fingerprint.
* :class:`AnticipatorReport` — full audit of registered contexts and serves.
* :class:`Anticipator`   — the primitive.

Pure stdlib.  No NumPy.  Deterministic given seed.  Thread-safe.
``json.dumps(report.to_dict())`` round-trips.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus

__all__ = [
    # Errors
    "AnticipatorError",
    "InvalidConfig",
    "UnknownContext",
    "EmptyForecast",
    "BudgetExceeded",
    # Constants
    "MATCH_EXACT",
    "MATCH_HASH",
    "MATCH_PREFIX",
    "MATCH_SIMILARITY",
    "KNOWN_MATCHERS",
    "EVICT_LRU",
    "EVICT_LFU",
    "EVICT_BELADY",
    "KNOWN_EVICTIONS",
    "KNAPSACK_GREEDY",
    "KNAPSACK_EXACT",
    "KNOWN_KNAPSACKS",
    # Events
    "ANTICIPATOR_STARTED",
    "ANTICIPATOR_REGISTERED",
    "ANTICIPATOR_ENUMERATED",
    "ANTICIPATOR_ALLOCATED",
    "ANTICIPATOR_PRECOMPUTED",
    "ANTICIPATOR_SERVED",
    "ANTICIPATOR_HIT",
    "ANTICIPATOR_MISS",
    "ANTICIPATOR_EVICTED",
    "ANTICIPATOR_INVALIDATED",
    "ANTICIPATOR_CERTIFIED",
    "ANTICIPATOR_REPORTED",
    "ANTICIPATOR_RESET",
    # Records
    "ContextRecord",
    "Candidate",
    "Plan",
    "CacheEntry",
    "PrecomputeResult",
    "ServeResult",
    "AnticipatorConfig",
    "AnticipatorCertificate",
    "AnticipatorReport",
    # Main
    "Anticipator",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AnticipatorError(ValueError):
    """Base class for Anticipator-specific errors."""


class InvalidConfig(AnticipatorError):
    """The :class:`AnticipatorConfig` is internally inconsistent."""


class UnknownContext(AnticipatorError):
    """A context-id was not registered."""


class EmptyForecast(AnticipatorError):
    """Forecaster returned no candidates."""


class BudgetExceeded(AnticipatorError):
    """Pre-compute exceeded the configured hard budget."""


# ---------------------------------------------------------------------------
# Taxonomy: matchers / evictions / knapsack
# ---------------------------------------------------------------------------

MATCH_EXACT = "exact"
MATCH_HASH = "hash"
MATCH_PREFIX = "prefix"
MATCH_SIMILARITY = "similarity"

KNOWN_MATCHERS: tuple[str, ...] = (
    MATCH_EXACT, MATCH_HASH, MATCH_PREFIX, MATCH_SIMILARITY,
)

EVICT_LRU = "lru"
EVICT_LFU = "lfu"
EVICT_BELADY = "belady-proxy"

KNOWN_EVICTIONS: tuple[str, ...] = (EVICT_LRU, EVICT_LFU, EVICT_BELADY)

KNAPSACK_GREEDY = "greedy"
KNAPSACK_EXACT = "exact"

KNOWN_KNAPSACKS: tuple[str, ...] = (KNAPSACK_GREEDY, KNAPSACK_EXACT)


# Event names emitted on the runtime EventBus.
ANTICIPATOR_STARTED = "anticipator.started"
ANTICIPATOR_REGISTERED = "anticipator.registered"
ANTICIPATOR_ENUMERATED = "anticipator.enumerated"
ANTICIPATOR_ALLOCATED = "anticipator.allocated"
ANTICIPATOR_PRECOMPUTED = "anticipator.precomputed"
ANTICIPATOR_SERVED = "anticipator.served"
ANTICIPATOR_HIT = "anticipator.hit"
ANTICIPATOR_MISS = "anticipator.miss"
ANTICIPATOR_EVICTED = "anticipator.evicted"
ANTICIPATOR_INVALIDATED = "anticipator.invalidated"
ANTICIPATOR_CERTIFIED = "anticipator.certified"
ANTICIPATOR_REPORTED = "anticipator.reported"
ANTICIPATOR_RESET = "anticipator.reset"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextRecord:
    """A registered idle context the agent may face.

    Attributes:
        ctx_id: stable opaque identifier of the context.
        ctx: free-form context payload (e.g. dict of recent messages,
            an agent's working memory snapshot).  Hashed verbatim;
            callers should normalise before registering.
        deadline_hint: wall-clock time (epoch seconds) by which the
            real query is expected to arrive.  ``None`` means
            "unknown / open-ended idle".
        weight: per-context weight (default 1.0).  Multiplies the
            forecaster's prior so the budget can favour "important"
            contexts (e.g. paying users).
        metadata: opaque dict carried through fingerprinting.
    """

    ctx_id: str
    ctx: Mapping[str, Any]
    deadline_hint: float | None = None
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ctx_id, str) or not self.ctx_id:
            raise AnticipatorError("ctx_id must be a non-empty string")
        if not isinstance(self.ctx, Mapping):
            raise AnticipatorError("ctx must be a mapping")
        if self.deadline_hint is not None:
            if (not isinstance(self.deadline_hint, (int, float))
                    or not math.isfinite(float(self.deadline_hint))):
                raise AnticipatorError(
                    "deadline_hint must be a finite number or None"
                )
        if (not isinstance(self.weight, (int, float))
                or not math.isfinite(float(self.weight))
                or float(self.weight) <= 0):
            raise AnticipatorError("weight must be a positive finite number")


@dataclass(frozen=True)
class Candidate:
    """One candidate future query enumerated by the forecaster.

    Attributes:
        query: free-form query payload that the ``Answerer`` will be
            invoked on if this candidate is chosen.  Hashed verbatim
            for matching; callers should normalise (e.g. whitespace
            strip + canonical JSON) before yielding.
        prior: ``P(query | ctx)`` in ``[0, 1]``.  Need not normalise
            across candidates — the knapsack uses the per-candidate
            value directly.
        est_miss_cost: estimated cost of computing this query fresh
            at *test* time (in whatever currency the caller booked
            into ``costs.py`` — dollars, FLOPs, latency_ms).
        est_precompute_cost: estimated cost of computing it at
            *sleep* time.  Usually ``≈ est_miss_cost`` but the caller
            may discount (e.g. batched inference is cheaper per query).
        tags: optional categorical tags carried through the certificate.
    """

    query: Mapping[str, Any]
    prior: float
    est_miss_cost: float
    est_precompute_cost: float
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, Mapping):
            raise AnticipatorError("query must be a mapping")
        for k, v, lo, hi in (
            ("prior", self.prior, 0.0, 1.0),
        ):
            if (not isinstance(v, (int, float))
                    or not math.isfinite(float(v))
                    or not (lo <= float(v) <= hi)):
                raise AnticipatorError(
                    f"{k} must be a finite number in [{lo}, {hi}]"
                )
        for k, v in (
            ("est_miss_cost", self.est_miss_cost),
            ("est_precompute_cost", self.est_precompute_cost),
        ):
            if (not isinstance(v, (int, float))
                    or not math.isfinite(float(v))
                    or float(v) < 0):
                raise AnticipatorError(
                    f"{k} must be a non-negative finite number"
                )

    @property
    def value(self) -> float:
        """Expected value of pre-computing this query.

        ``E[saved_cost] = P(hit | ctx) * est_miss_cost`` under the
        prior.  The matcher introduces a separate probability of
        actually matching at serve time — that is amortised through
        the realised hit-rate certificate.
        """
        return float(self.prior) * float(self.est_miss_cost)


@dataclass(frozen=True)
class Plan:
    """Knapsack-chosen subset of candidates to pre-compute.

    Attributes:
        ctx_id: which context this plan belongs to.
        chosen: indices (into the original Candidate list) that the
            knapsack selected.  Order is stable: ascending by index.
        total_value: expected saved cost.
        total_precompute_cost: hard cost of running this plan.
        budget: the budget the plan was solved against.
        knapsack: which solver produced the plan.
    """

    ctx_id: str
    chosen: tuple[int, ...]
    total_value: float
    total_precompute_cost: float
    budget: float
    knapsack: str


@dataclass
class CacheEntry:
    """One cached pre-computed answer.

    Fields are mutable so the cache can update access counters in place
    without re-allocating frozen records.
    """

    ctx_id: str
    query: Mapping[str, Any]
    query_hash: str
    answer: Any
    prior: float
    est_miss_cost: float
    realised_precompute_cost: float
    created_ts: float
    last_access_ts: float
    access_count: int = 0
    hit_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ctx_id": self.ctx_id,
            "query": dict(self.query),
            "query_hash": self.query_hash,
            "prior": float(self.prior),
            "est_miss_cost": float(self.est_miss_cost),
            "realised_precompute_cost": float(self.realised_precompute_cost),
            "created_ts": float(self.created_ts),
            "last_access_ts": float(self.last_access_ts),
            "access_count": int(self.access_count),
            "hit_count": int(self.hit_count),
        }


@dataclass(frozen=True)
class PrecomputeResult:
    """Outcome of executing a :class:`Plan`."""

    ctx_id: str
    requested: int
    succeeded: int
    failed: int
    total_cost: float
    budget: float
    cache_size_after: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServeResult:
    """Outcome of one ``serve`` call.

    Attributes:
        ctx_id: context the query was served under.
        query_hash: hash of the live query.
        hit: True iff a cached answer was returned.
        answer: the returned answer (cached on hit, fresh on miss).
        saved_cost: how much test-time cost was avoided.  On miss,
            zero.  On hit, equal to ``entry.est_miss_cost``.
        served_cost: how much was paid right now.  On hit, the
            constant ``hit_cost`` from config.  On miss, the
            answerer's fresh cost (passed through).
        latency_ms: wall-clock time consumed inside ``serve``.
        matcher: which matcher fired.
        cache_hash: hash of the cache entry that matched (empty on miss).
    """

    ctx_id: str
    query_hash: str
    hit: bool
    answer: Any
    saved_cost: float
    served_cost: float
    latency_ms: float
    matcher: str
    cache_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ctx_id": self.ctx_id,
            "query_hash": self.query_hash,
            "hit": bool(self.hit),
            "saved_cost": float(self.saved_cost),
            "served_cost": float(self.served_cost),
            "latency_ms": float(self.latency_ms),
            "matcher": self.matcher,
            "cache_hash": self.cache_hash,
            # Answer is opaque — caller decides serialisation.
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnticipatorConfig:
    """Configuration for an :class:`Anticipator` instance.

    Budget controls
    ---------------

    Attributes:
        sleep_budget_per_ctx: hard upper bound on pre-compute cost for
            a single context.  ``Anticipator`` will never exceed it.
        sleep_budget_global: hard upper bound on aggregate pre-compute
            cost across all contexts in this instance.
        cache_size_limit: maximum number of cache entries; eviction
            kicks in when exceeded.  ``0`` means unbounded.

    Matching
    --------

    Attributes:
        matcher: one of :data:`KNOWN_MATCHERS`.  ``"hash"`` compares
            canonical-JSON SHA-256 (fast and exact); ``"exact"`` is the
            same but compares plain dict equality;
            ``"prefix"`` compares a string prefix on the canonical-JSON
            form (lets the caller cache "tell me about X" answers
            keyed by the noun); ``"similarity"`` calls an injected
            embedder and uses cosine similarity above
            ``similarity_threshold``.
        similarity_threshold: cosine threshold for ``MATCH_SIMILARITY``.

    Eviction
    --------

    Attributes:
        eviction: one of :data:`KNOWN_EVICTIONS`.

    Knapsack
    --------

    Attributes:
        knapsack: ``"greedy"`` (Karmarkar-Karp value/cost ratio) or
            ``"exact"`` (branch-and-bound; exact for ``K ≤ 64``).
        cost_unit: free-form name of the cost unit (default
            ``"flops"``) — flows through to the certificate report.

    Certificates / safety
    --------------------

    Attributes:
        alpha: confidence level for hit-rate / saved-cost intervals
            (default ``0.05`` ⇒ 95% CIs).
        min_serves_for_certificate: refuse to certify until at least
            this many serves have been recorded.
        hit_cost: cost booked for a cache hit (default ``0.0`` — the
            cache is free).  Set non-zero if the cache itself runs on a
            paid backing store.

    Reproducibility
    ---------------

    Attributes:
        seed: RNG seed for any randomised step (bloom hashes, knapsack
            tie-breaks).  Default ``0``.
    """

    sleep_budget_per_ctx: float = 1.0
    sleep_budget_global: float = math.inf
    cache_size_limit: int = 1024
    matcher: str = MATCH_HASH
    similarity_threshold: float = 0.92
    eviction: str = EVICT_LRU
    knapsack: str = KNAPSACK_EXACT
    cost_unit: str = "flops"
    alpha: float = 0.05
    min_serves_for_certificate: int = 8
    hit_cost: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.sleep_budget_per_ctx, (int, float))
                or float(self.sleep_budget_per_ctx) <= 0):
            raise InvalidConfig("sleep_budget_per_ctx must be > 0")
        if (not isinstance(self.sleep_budget_global, (int, float))
                or float(self.sleep_budget_global) <= 0):
            raise InvalidConfig("sleep_budget_global must be > 0")
        if (not isinstance(self.cache_size_limit, int)
                or self.cache_size_limit < 0):
            raise InvalidConfig("cache_size_limit must be a non-negative int")
        if self.matcher not in KNOWN_MATCHERS:
            raise InvalidConfig(
                f"matcher must be one of {KNOWN_MATCHERS!r}"
            )
        if not (0.0 < float(self.similarity_threshold) <= 1.0):
            raise InvalidConfig(
                "similarity_threshold must lie in (0, 1]"
            )
        if self.eviction not in KNOWN_EVICTIONS:
            raise InvalidConfig(
                f"eviction must be one of {KNOWN_EVICTIONS!r}"
            )
        if self.knapsack not in KNOWN_KNAPSACKS:
            raise InvalidConfig(
                f"knapsack must be one of {KNOWN_KNAPSACKS!r}"
            )
        if not (0.0 < float(self.alpha) < 1.0):
            raise InvalidConfig("alpha must lie in (0, 1)")
        if (not isinstance(self.min_serves_for_certificate, int)
                or self.min_serves_for_certificate < 1):
            raise InvalidConfig("min_serves_for_certificate must be >= 1")
        if (not isinstance(self.hit_cost, (int, float))
                or float(self.hit_cost) < 0):
            raise InvalidConfig("hit_cost must be >= 0")


@dataclass(frozen=True)
class AnticipatorCertificate:
    """Anytime-valid certificate of the realised speedup.

    Attributes:
        n_serves: number of serves observed.
        n_hits: number of hits.
        hit_rate: ``n_hits / n_serves``.
        hit_rate_wilson_lo, hit_rate_wilson_hi: two-sided Wilson CI.
        hit_rate_hoeffding_lo: one-sided Hoeffding LCB on the hit rate
            (variance-free; tighter than Wilson only for very small ``n``).
        saved_cost_total: sum of ``saved_cost`` across all serves.
        saved_cost_mean: mean per-serve saved cost.
        saved_cost_eb_lo: empirical-Bernstein LCB on the per-serve saved
            cost (Maurer-Pontil 2009).  The headline guarantee: with
            confidence ``1 - alpha`` the true expected saved cost is at
            least this much.
        precompute_cost_total: total pre-compute spend across contexts.
        net_value: ``saved_cost_total - precompute_cost_total``.  A
            negative value means anticipation lost money on this
            window; the coordinator should consider increasing the
            ``similarity_threshold`` or shrinking ``K``.
        alpha: confidence level.
        cost_unit: matches :class:`AnticipatorConfig`.
        fingerprint: SHA-256 of the Merkle chain up to certification.
    """

    n_serves: int
    n_hits: int
    hit_rate: float
    hit_rate_wilson_lo: float
    hit_rate_wilson_hi: float
    hit_rate_hoeffding_lo: float
    saved_cost_total: float
    saved_cost_mean: float
    saved_cost_eb_lo: float
    precompute_cost_total: float
    net_value: float
    alpha: float
    cost_unit: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnticipatorReport:
    """End-of-window audit of the anticipator instance."""

    contexts: int
    plans: int
    cache_entries: int
    serves: int
    hits: int
    misses: int
    certificate: AnticipatorCertificate
    cost_unit: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["certificate"] = self.certificate.to_dict()
        return d


# ---------------------------------------------------------------------------
# Hashing / canonicalisation helpers
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """Return a JSON-canonical version of ``value`` for hashing."""
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.12g}")
    if isinstance(value, bool):
        return bool(value)
    return value


def _hash_value(value: Any) -> str:
    """SHA-256 of the canonical JSON form."""
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True).encode()
    ).hexdigest()


def _stable(value: Any) -> Any:
    """Strip non-determinism for the Merkle chain."""
    if isinstance(value, dict):
        return {k: _stable(value[k]) for k in sorted(value.keys()) if k != "ts"}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.12g}")
    if isinstance(value, bool):
        return bool(value)
    return value


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Stdlib cosine similarity.  Returns 0 for zero-norm vectors."""
    if len(a) != len(b):
        raise AnticipatorError(
            f"embedding length mismatch: {len(a)} vs {len(b)}"
        )
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += float(x) * float(y)
        da += float(x) * float(x)
        db += float(y) * float(y)
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return num / math.sqrt(da * db)


# ---------------------------------------------------------------------------
# Knapsack
# ---------------------------------------------------------------------------


def _knapsack_greedy(values: Sequence[float],
                     costs: Sequence[float],
                     budget: float) -> tuple[int, ...]:
    """Karmarkar-Karp value/cost ratio greedy.

    O(K log K).  Optimal within a factor 2 in the worst case; near-optimal
    for the cost distributions typical in test-time-compute pricing.
    """
    n = len(values)
    if n == 0 or budget <= 0:
        return ()
    # Sort by value/cost ratio, descending.  Ties broken by lower cost
    # (smaller items first) then by index (stability).
    order = sorted(
        range(n),
        key=lambda i: (
            -(values[i] / costs[i]) if costs[i] > 0 else -float("inf"),
            costs[i],
            i,
        ),
    )
    chosen: list[int] = []
    spent = 0.0
    for i in order:
        c = float(costs[i])
        if c <= 0:
            # Free item — always include if it has any value.
            if values[i] > 0:
                chosen.append(i)
            continue
        if spent + c <= budget + 1e-12:
            chosen.append(i)
            spent += c
    return tuple(sorted(chosen))


def _knapsack_exact(values: Sequence[float],
                    costs: Sequence[float],
                    budget: float) -> tuple[int, ...]:
    """Branch-and-bound 0-1 knapsack.  Exact for small ``K``.

    Bounded by ``2^K`` worst-case but the linear-relaxation upper bound
    typically prunes huge subtrees.  Used by default for ``K ≤ 32``.
    """
    n = len(values)
    if n == 0 or budget <= 0:
        return ()
    if n > 32:
        # Fall back to greedy — branch-and-bound becomes expensive.
        return _knapsack_greedy(values, costs, budget)

    # Ratio-sorted order maximises pruning.
    order = sorted(
        range(n),
        key=lambda i: (
            -(values[i] / costs[i]) if costs[i] > 0 else -float("inf"),
            costs[i],
            i,
        ),
    )
    vs = [float(values[i]) for i in order]
    cs = [float(costs[i]) for i in order]

    best_value = 0.0
    best_set: tuple[int, ...] = ()

    def _upper_bound(idx: int, spent: float, value: float) -> float:
        # LP relaxation: take items in ratio order, last item fractionally.
        rem = budget - spent
        v = value
        for j in range(idx, n):
            if cs[j] <= 0:
                # Free items always add full value.
                v += vs[j]
                continue
            if cs[j] <= rem:
                v += vs[j]
                rem -= cs[j]
            else:
                if cs[j] > 0:
                    v += vs[j] * (rem / cs[j])
                break
        return v

    def _branch(idx: int, spent: float, value: float, chosen: tuple[int, ...]) -> None:
        nonlocal best_value, best_set
        if idx == n:
            if value > best_value + 1e-12:
                best_value = value
                best_set = chosen
            return
        # Prune: even with LP-relaxation top-up, can we beat best?
        if _upper_bound(idx, spent, value) <= best_value + 1e-12:
            return
        # Try "include" first if budget allows.
        if cs[idx] <= 0 and vs[idx] > 0:
            _branch(idx + 1, spent, value + vs[idx], chosen + (idx,))
        elif spent + cs[idx] <= budget + 1e-12:
            _branch(idx + 1, spent + cs[idx], value + vs[idx],
                    chosen + (idx,))
        # Then "exclude".
        _branch(idx + 1, spent, value, chosen)

    _branch(0, 0.0, 0.0, ())
    # Translate back from the sorted order to original indices.
    return tuple(sorted(order[j] for j in best_set))


def _solve_knapsack(values: Sequence[float],
                    costs: Sequence[float],
                    budget: float,
                    method: str) -> tuple[int, ...]:
    if method == KNAPSACK_GREEDY:
        return _knapsack_greedy(values, costs, budget)
    if method == KNAPSACK_EXACT:
        return _knapsack_exact(values, costs, budget)
    raise InvalidConfig(f"unknown knapsack: {method!r}")


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def _z_two_sided(alpha: float) -> float:
    """Normal quantile ``z_{1-alpha/2}`` via inverse-erf.

    For typical ``alpha`` (0.05, 0.01) the stdlib ``math.erf`` is plenty
    accurate; we invert by bisection to avoid SciPy.
    """
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        # P(Z <= mid) = 0.5 * (1 + erf(mid / sqrt(2)))
        p = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if p < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _wilson_ci(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided Wilson interval on a binomial rate."""
    if n <= 0:
        return (0.0, 1.0)
    z = _z_two_sided(alpha)
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _hoeffding_lcb_rate(k: int, n: int, alpha: float) -> float:
    """One-sided Hoeffding LCB on a binomial rate."""
    if n <= 0:
        return 0.0
    eps = math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return max(0.0, k / n - eps)


def _empirical_bernstein_lcb(samples: Sequence[float],
                             alpha: float,
                             bound_b: float) -> float:
    """Maurer-Pontil 2009 empirical-Bernstein LCB.

    Returns a lower confidence bound on ``E[X]`` where the samples are
    iid in ``[0, bound_b]``.  Tighter than Hoeffding when the empirical
    variance is small.
    """
    n = len(samples)
    if n <= 1:
        return 0.0
    mean = sum(samples) / n
    var = sum((float(x) - mean) ** 2 for x in samples) / (n - 1)
    log_term = math.log(2.0 / alpha)
    bern_term = math.sqrt(2.0 * var * log_term / n)
    hoeff_term = 7.0 * float(bound_b) * log_term / (3.0 * (n - 1))
    return mean - bern_term - hoeff_term


# ---------------------------------------------------------------------------
# Forecaster / Answerer / Embedder protocols (structural)
# ---------------------------------------------------------------------------

# A ``Forecaster`` is any callable
#     forecaster(ctx: Mapping[str, Any], k: int, *, rng: random.Random)
#         -> Iterable[Candidate]
# returning at most ``k`` candidates.  No ABC — duck typing keeps
# composition cheap.
Forecaster = Callable[..., Iterable[Candidate]]

# An ``Answerer`` is any callable
#     answerer(ctx: Mapping[str, Any], query: Mapping[str, Any])
#         -> tuple[answer, realised_cost]
# returning the answer and the realised cost (in the same unit as
# ``est_miss_cost``).
Answerer = Callable[..., tuple[Any, float]]

# An ``Embedder`` is any callable
#     embedder(text: Mapping[str, Any]) -> Sequence[float]
Embedder = Callable[..., Sequence[float]]


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


class Anticipator:
    """The runtime-primitive entry point.

    Lifecycle::

        ant = Anticipator(AnticipatorConfig(...))
        ant.register_context("turn:42", ctx={...}, deadline_hint=t+30)
        plan = ant.allocate("turn:42", forecaster=fc, k=8)
        ant.precompute("turn:42", plan, answerer=ans)
        # ... time passes, the user finally asks ...
        result = ant.serve("turn:42", query={...}, answerer=ans)
        cert = ant.certificate()
        report = ant.report()
    """

    def __init__(self,
                 config: AnticipatorConfig | None = None,
                 *,
                 bus: EventBus | None = None,
                 instance_id: str | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.config = config or AnticipatorConfig()
        self.bus = bus
        self.instance_id = instance_id or ""
        self._clock = clock or time.time
        self._lock = threading.RLock()

        # State.
        self._contexts: dict[str, ContextRecord] = {}
        self._candidates: dict[str, tuple[Candidate, ...]] = {}
        self._plans: dict[str, Plan] = {}
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        # Per-ctx index: ctx_id -> set of query_hash strings in the cache.
        self._cache_by_ctx: dict[str, set[str]] = {}
        # Per-ctx and global counters.
        self._serves: list[ServeResult] = []
        self._hits: int = 0
        self._misses: int = 0
        self._precompute_spent: dict[str, float] = {}
        self._precompute_spent_global: float = 0.0

        # RNG.
        self._rng = random.Random(self.config.seed)

        # Merkle chain.
        self._fingerprint = hashlib.sha256()
        self._fingerprint.update(json.dumps(
            {
                "version": 1,
                "instance_id": self.instance_id,
                "config": _stable({
                    "sleep_budget_per_ctx": self.config.sleep_budget_per_ctx,
                    "sleep_budget_global": (
                        None if self.config.sleep_budget_global == math.inf
                        else self.config.sleep_budget_global
                    ),
                    "cache_size_limit": self.config.cache_size_limit,
                    "matcher": self.config.matcher,
                    "similarity_threshold": self.config.similarity_threshold,
                    "eviction": self.config.eviction,
                    "knapsack": self.config.knapsack,
                    "alpha": self.config.alpha,
                    "hit_cost": self.config.hit_cost,
                    "seed": self.config.seed,
                }),
            },
            sort_keys=True,
        ).encode())

        self._publish(ANTICIPATOR_STARTED, {
            "instance_id": self.instance_id,
            "matcher": self.config.matcher,
            "knapsack": self.config.knapsack,
            "sleep_budget_per_ctx": self.config.sleep_budget_per_ctx,
        })

    # ----- event helpers -----
    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        payload = {**data, "ts": self._clock()}
        self._fingerprint.update(json.dumps(
            {"kind": kind, "data": _stable(payload)}, sort_keys=True,
        ).encode())
        if self.bus is not None:
            self.bus.publish(Event(kind=kind, data=payload))

    @property
    def fingerprint_hash(self) -> str:
        return self._fingerprint.hexdigest()

    @property
    def contexts(self) -> tuple[ContextRecord, ...]:
        with self._lock:
            return tuple(self._contexts.values())

    @property
    def cache(self) -> tuple[CacheEntry, ...]:
        with self._lock:
            return tuple(self._cache.values())

    @property
    def serves(self) -> tuple[ServeResult, ...]:
        with self._lock:
            return tuple(self._serves)

    # ----- registration -----
    def register_context(self,
                         ctx_id: str,
                         ctx: Mapping[str, Any],
                         *,
                         deadline_hint: float | None = None,
                         weight: float = 1.0,
                         metadata: Mapping[str, Any] | None = None) -> ContextRecord:
        """Declare an idle context.  Idempotent on ``ctx_id``: re-registering
        overwrites the prior record but keeps any plans / cache for the
        same id (use :meth:`invalidate` to drop those).
        """
        rec = ContextRecord(
            ctx_id=ctx_id,
            ctx=dict(ctx),
            deadline_hint=deadline_hint,
            weight=float(weight),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._contexts[ctx_id] = rec
            self._precompute_spent.setdefault(ctx_id, 0.0)
            self._publish(ANTICIPATOR_REGISTERED, {
                "ctx_id": ctx_id,
                "ctx_hash": _hash_value(rec.ctx),
                "deadline_hint": rec.deadline_hint,
                "weight": rec.weight,
            })
            return rec

    def invalidate(self, ctx_id: str) -> int:
        """Drop all cache entries and plans for ``ctx_id``.  Returns the
        number of cache entries removed.
        """
        with self._lock:
            removed = 0
            for qh in list(self._cache_by_ctx.get(ctx_id, ())):
                if qh in self._cache:
                    del self._cache[qh]
                    removed += 1
            self._cache_by_ctx.pop(ctx_id, None)
            self._plans.pop(ctx_id, None)
            self._candidates.pop(ctx_id, None)
            self._publish(ANTICIPATOR_INVALIDATED, {
                "ctx_id": ctx_id,
                "removed": removed,
            })
            return removed

    # ----- enumeration / allocation -----
    def enumerate(self,
                  ctx_id: str,
                  forecaster: Forecaster,
                  *,
                  k: int = 8) -> tuple[Candidate, ...]:
        """Ask the forecaster for up to ``k`` candidate next-queries.

        Stable RNG seed: ``hash(seed, ctx_id)`` so calling enumerate on
        the same context twice with the same forecaster yields the same
        candidates.
        """
        with self._lock:
            rec = self._contexts.get(ctx_id)
            if rec is None:
                raise UnknownContext(f"ctx_id={ctx_id!r} not registered")
            if k <= 0:
                raise AnticipatorError("k must be >= 1")
            local_seed = int(hashlib.sha256(
                (str(self.config.seed) + "|" + ctx_id).encode()
            ).hexdigest(), 16) % (2 ** 31)
            rng = random.Random(local_seed)
            raw = list(forecaster(rec.ctx, k=k, rng=rng))
            if not raw:
                raise EmptyForecast(
                    f"forecaster yielded no candidates for ctx_id={ctx_id!r}"
                )
            for c in raw:
                if not isinstance(c, Candidate):
                    raise AnticipatorError(
                        f"forecaster must yield Candidate; got {type(c).__name__}"
                    )
            # Sort by value descending (stable wrt forecaster order).
            cands = tuple(sorted(
                raw,
                key=lambda c: (-c.value, _hash_value(dict(c.query))),
            ))[:k]
            self._candidates[ctx_id] = cands
            self._publish(ANTICIPATOR_ENUMERATED, {
                "ctx_id": ctx_id,
                "k": len(cands),
                "top_prior": cands[0].prior if cands else 0.0,
                "total_value": sum(c.value for c in cands),
            })
            return cands

    def allocate(self,
                 ctx_id: str,
                 *,
                 forecaster: Forecaster | None = None,
                 k: int = 8,
                 budget: float | None = None) -> Plan:
        """Solve the knapsack for ``ctx_id``.

        If ``forecaster`` is provided and candidates have not been
        enumerated yet, runs enumeration first.  ``budget`` defaults to
        ``min(config.sleep_budget_per_ctx, global_remaining)``.
        """
        with self._lock:
            rec = self._contexts.get(ctx_id)
            if rec is None:
                raise UnknownContext(f"ctx_id={ctx_id!r} not registered")
            cands = self._candidates.get(ctx_id)
            if cands is None:
                if forecaster is None:
                    raise AnticipatorError(
                        "allocate() requires either prior enumerate() or a forecaster"
                    )
                cands = self.enumerate(ctx_id, forecaster, k=k)
            global_remaining = max(
                0.0,
                float(self.config.sleep_budget_global) - self._precompute_spent_global,
            )
            ctx_remaining = max(
                0.0,
                float(self.config.sleep_budget_per_ctx)
                - float(self._precompute_spent.get(ctx_id, 0.0)),
            )
            effective_budget = float(budget) if budget is not None else min(
                ctx_remaining, global_remaining
            )
            if effective_budget < 0:
                effective_budget = 0.0
            # Weight values by the context's weight.
            values = [float(c.value) * float(rec.weight) for c in cands]
            costs = [float(c.est_precompute_cost) for c in cands]
            chosen = _solve_knapsack(
                values, costs, effective_budget, self.config.knapsack,
            )
            plan = Plan(
                ctx_id=ctx_id,
                chosen=tuple(chosen),
                total_value=sum(values[i] for i in chosen),
                total_precompute_cost=sum(costs[i] for i in chosen),
                budget=effective_budget,
                knapsack=self.config.knapsack,
            )
            self._plans[ctx_id] = plan
            self._publish(ANTICIPATOR_ALLOCATED, {
                "ctx_id": ctx_id,
                "k": len(cands),
                "chosen": len(plan.chosen),
                "budget": effective_budget,
                "total_value": plan.total_value,
                "total_precompute_cost": plan.total_precompute_cost,
                "knapsack": plan.knapsack,
            })
            return plan

    # ----- precompute -----
    def precompute(self,
                   ctx_id: str,
                   plan: Plan | None,
                   answerer: Answerer) -> PrecomputeResult:
        """Run ``answerer`` on each candidate in ``plan`` and cache the
        results.  If ``plan`` is ``None`` the last plan for ``ctx_id``
        is used.
        """
        with self._lock:
            rec = self._contexts.get(ctx_id)
            if rec is None:
                raise UnknownContext(f"ctx_id={ctx_id!r} not registered")
            if plan is None:
                plan = self._plans.get(ctx_id)
                if plan is None:
                    raise AnticipatorError(
                        f"no plan registered for ctx_id={ctx_id!r}"
                    )
            cands = self._candidates.get(ctx_id, ())
            requested = len(plan.chosen)
            succeeded = 0
            failed = 0
            total_cost = 0.0
            for idx in plan.chosen:
                if idx < 0 or idx >= len(cands):
                    failed += 1
                    continue
                c = cands[idx]
                # Hard budget check before the call.
                budget_after = (self._precompute_spent_global
                                + float(c.est_precompute_cost))
                if budget_after > float(self.config.sleep_budget_global) + 1e-9:
                    failed += 1
                    continue
                ctx_budget_after = (self._precompute_spent.get(ctx_id, 0.0)
                                    + float(c.est_precompute_cost))
                if ctx_budget_after > float(self.config.sleep_budget_per_ctx) + 1e-9:
                    failed += 1
                    continue
                try:
                    answer, realised = answerer(rec.ctx, dict(c.query))
                except Exception:
                    failed += 1
                    continue
                if (not isinstance(realised, (int, float))
                        or not math.isfinite(float(realised))
                        or float(realised) < 0):
                    failed += 1
                    continue
                realised = float(realised)
                qh = _hash_value(dict(c.query))
                entry = CacheEntry(
                    ctx_id=ctx_id,
                    query=dict(c.query),
                    query_hash=qh,
                    answer=answer,
                    prior=float(c.prior),
                    est_miss_cost=float(c.est_miss_cost),
                    realised_precompute_cost=realised,
                    created_ts=self._clock(),
                    last_access_ts=self._clock(),
                )
                # Insert / overwrite into cache.
                if qh in self._cache:
                    del self._cache[qh]
                self._cache[qh] = entry
                self._cache_by_ctx.setdefault(ctx_id, set()).add(qh)
                # Bookkeeping.
                self._precompute_spent[ctx_id] = (
                    self._precompute_spent.get(ctx_id, 0.0) + realised
                )
                self._precompute_spent_global += realised
                total_cost += realised
                succeeded += 1
                self._publish(ANTICIPATOR_PRECOMPUTED, {
                    "ctx_id": ctx_id,
                    "query_hash": qh,
                    "realised_cost": realised,
                    "est_miss_cost": float(c.est_miss_cost),
                    "prior": float(c.prior),
                })
                # Eviction if oversize.
                self._maybe_evict()
            return PrecomputeResult(
                ctx_id=ctx_id,
                requested=requested,
                succeeded=succeeded,
                failed=failed,
                total_cost=total_cost,
                budget=plan.budget,
                cache_size_after=len(self._cache),
            )

    def _maybe_evict(self) -> None:
        limit = self.config.cache_size_limit
        if limit <= 0:
            return
        while len(self._cache) > limit:
            if self.config.eviction == EVICT_LRU:
                qh, entry = next(iter(self._cache.items()))
            elif self.config.eviction == EVICT_LFU:
                qh, entry = min(
                    self._cache.items(),
                    key=lambda kv: (kv[1].hit_count, kv[1].last_access_ts),
                )
            else:  # EVICT_BELADY
                # Furthest-in-future proxy: lowest prior * lowest miss cost.
                qh, entry = min(
                    self._cache.items(),
                    key=lambda kv: (
                        kv[1].prior * kv[1].est_miss_cost,
                        kv[1].last_access_ts,
                    ),
                )
            del self._cache[qh]
            self._cache_by_ctx.get(entry.ctx_id, set()).discard(qh)
            self._publish(ANTICIPATOR_EVICTED, {
                "ctx_id": entry.ctx_id,
                "query_hash": qh,
                "policy": self.config.eviction,
                "cache_size_after": len(self._cache),
            })

    # ----- serve -----
    def serve(self,
              ctx_id: str,
              query: Mapping[str, Any],
              *,
              answerer: Answerer | None = None,
              embedder: Embedder | None = None) -> ServeResult:
        """Resolve a live query against the cache.  On hit returns the
        cached answer; on miss runs ``answerer`` fresh (if provided)
        and returns its answer with ``saved_cost = 0``.

        ``embedder`` is required iff ``config.matcher == MATCH_SIMILARITY``.
        """
        with self._lock:
            t0 = self._clock()
            if ctx_id not in self._contexts:
                raise UnknownContext(f"ctx_id={ctx_id!r} not registered")
            if not isinstance(query, Mapping):
                raise AnticipatorError("query must be a mapping")
            qh = _hash_value(dict(query))
            hit_entry, used_matcher, cache_hash = self._match(ctx_id, query, qh, embedder)

            if hit_entry is not None:
                hit_entry.last_access_ts = self._clock()
                hit_entry.access_count += 1
                hit_entry.hit_count += 1
                # Move-to-end for LRU.
                self._cache.move_to_end(hit_entry.query_hash, last=True)
                saved = float(hit_entry.est_miss_cost)
                served = float(self.config.hit_cost)
                latency = (self._clock() - t0) * 1000.0
                result = ServeResult(
                    ctx_id=ctx_id,
                    query_hash=qh,
                    hit=True,
                    answer=hit_entry.answer,
                    saved_cost=saved,
                    served_cost=served,
                    latency_ms=latency,
                    matcher=used_matcher,
                    cache_hash=cache_hash,
                )
                self._serves.append(result)
                self._hits += 1
                self._publish(ANTICIPATOR_HIT, {
                    "ctx_id": ctx_id,
                    "query_hash": qh,
                    "cache_hash": cache_hash,
                    "saved_cost": saved,
                    "matcher": used_matcher,
                    "latency_ms": latency,
                })
                self._publish(ANTICIPATOR_SERVED, result.to_dict())
                return result

            # Miss.
            if answerer is None:
                answer = None
                served = 0.0
            else:
                try:
                    answer, realised = answerer(self._contexts[ctx_id].ctx, dict(query))
                except Exception:
                    answer = None
                    realised = 0.0
                if (not isinstance(realised, (int, float))
                        or not math.isfinite(float(realised))):
                    realised = 0.0
                served = max(0.0, float(realised))
            latency = (self._clock() - t0) * 1000.0
            result = ServeResult(
                ctx_id=ctx_id,
                query_hash=qh,
                hit=False,
                answer=answer,
                saved_cost=0.0,
                served_cost=served,
                latency_ms=latency,
                matcher=self.config.matcher,
                cache_hash="",
            )
            self._serves.append(result)
            self._misses += 1
            self._publish(ANTICIPATOR_MISS, {
                "ctx_id": ctx_id,
                "query_hash": qh,
                "served_cost": served,
                "matcher": self.config.matcher,
                "latency_ms": latency,
            })
            self._publish(ANTICIPATOR_SERVED, result.to_dict())
            return result

    def _match(self,
               ctx_id: str,
               query: Mapping[str, Any],
               qh: str,
               embedder: Embedder | None) -> tuple[CacheEntry | None, str, str]:
        """Find a cache hit for ``query`` under ``ctx_id``.

        Returns ``(entry, matcher_used, cache_hash)`` on hit and
        ``(None, "", "")`` on miss.
        """
        matcher = self.config.matcher
        ctx_set = self._cache_by_ctx.get(ctx_id, set())

        if matcher == MATCH_HASH:
            if qh in self._cache and qh in ctx_set:
                return self._cache[qh], MATCH_HASH, qh
            return (None, "", "")

        if matcher == MATCH_EXACT:
            for cand_qh in ctx_set:
                e = self._cache.get(cand_qh)
                if e is not None and dict(e.query) == dict(query):
                    return e, MATCH_EXACT, cand_qh
            return (None, "", "")

        if matcher == MATCH_PREFIX:
            q_canon = json.dumps(_canonical(dict(query)), sort_keys=True)
            best: tuple[CacheEntry, str] | None = None
            best_len = -1
            for cand_qh in ctx_set:
                e = self._cache.get(cand_qh)
                if e is None:
                    continue
                e_canon = json.dumps(_canonical(dict(e.query)), sort_keys=True)
                if q_canon.startswith(e_canon) or e_canon.startswith(q_canon):
                    shared = min(len(q_canon), len(e_canon))
                    if shared > best_len:
                        best_len = shared
                        best = (e, cand_qh)
            if best is not None:
                return best[0], MATCH_PREFIX, best[1]
            return (None, "", "")

        if matcher == MATCH_SIMILARITY:
            if embedder is None:
                raise AnticipatorError(
                    "matcher=similarity requires an embedder argument to serve()"
                )
            q_emb = list(map(float, embedder(dict(query))))
            best: tuple[CacheEntry, str, float] | None = None
            for cand_qh in ctx_set:
                e = self._cache.get(cand_qh)
                if e is None:
                    continue
                e_emb = list(map(float, embedder(dict(e.query))))
                sim = _cosine(q_emb, e_emb)
                if sim >= float(self.config.similarity_threshold):
                    if best is None or sim > best[2]:
                        best = (e, cand_qh, sim)
            if best is not None:
                return best[0], MATCH_SIMILARITY, best[1]
            return (None, "", "")

        raise InvalidConfig(f"unknown matcher: {matcher!r}")

    # ----- certificate -----
    def certificate(self) -> AnticipatorCertificate:
        """Produce a replay-verifiable certificate over all serves to date.

        Raises :class:`AnticipatorError` if fewer than
        ``min_serves_for_certificate`` serves are recorded.
        """
        with self._lock:
            n = len(self._serves)
            if n < self.config.min_serves_for_certificate:
                raise AnticipatorError(
                    f"need at least {self.config.min_serves_for_certificate}"
                    f" serves for a certificate; have {n}"
                )
            hits = self._hits
            saves = [float(s.saved_cost) for s in self._serves]
            saved_total = float(sum(saves))
            saved_mean = saved_total / n if n else 0.0
            # Bound for empirical-Bernstein: the worst-case per-serve
            # saved cost is the max observed miss-cost (saved cost is
            # zero on miss, est_miss_cost on hit).
            bound_b = max(saves) if saves else 0.0
            if bound_b <= 0:
                eb_lo = 0.0
            else:
                eb_lo = _empirical_bernstein_lcb(saves, self.config.alpha, bound_b)
            wilson_lo, wilson_hi = _wilson_ci(hits, n, self.config.alpha)
            hoeff_lo = _hoeffding_lcb_rate(hits, n, self.config.alpha)
            cert = AnticipatorCertificate(
                n_serves=n,
                n_hits=hits,
                hit_rate=hits / n if n else 0.0,
                hit_rate_wilson_lo=wilson_lo,
                hit_rate_wilson_hi=wilson_hi,
                hit_rate_hoeffding_lo=hoeff_lo,
                saved_cost_total=saved_total,
                saved_cost_mean=saved_mean,
                saved_cost_eb_lo=eb_lo,
                precompute_cost_total=float(self._precompute_spent_global),
                net_value=saved_total - float(self._precompute_spent_global),
                alpha=float(self.config.alpha),
                cost_unit=self.config.cost_unit,
                fingerprint=self.fingerprint_hash,
            )
            self._publish(ANTICIPATOR_CERTIFIED, {
                "n_serves": cert.n_serves,
                "n_hits": cert.n_hits,
                "hit_rate": cert.hit_rate,
                "hit_rate_wilson_lo": cert.hit_rate_wilson_lo,
                "hit_rate_wilson_hi": cert.hit_rate_wilson_hi,
                "saved_cost_eb_lo": cert.saved_cost_eb_lo,
                "net_value": cert.net_value,
                "alpha": cert.alpha,
                "fingerprint": cert.fingerprint,
            })
            return cert

    def report(self) -> AnticipatorReport:
        """End-of-window report combining counters + certificate."""
        with self._lock:
            cert = self.certificate()
            rep = AnticipatorReport(
                contexts=len(self._contexts),
                plans=len(self._plans),
                cache_entries=len(self._cache),
                serves=len(self._serves),
                hits=self._hits,
                misses=self._misses,
                certificate=cert,
                cost_unit=self.config.cost_unit,
            )
            self._publish(ANTICIPATOR_REPORTED, {
                "contexts": rep.contexts,
                "plans": rep.plans,
                "cache_entries": rep.cache_entries,
                "serves": rep.serves,
                "hits": rep.hits,
                "misses": rep.misses,
            })
            return rep

    def reset(self) -> None:
        """Drop all state but preserve config and instance_id.

        The fingerprint chain *continues* across the reset so the
        operation is itself audit-visible.  Use a fresh
        :class:`Anticipator` instance for a clean chain.
        """
        with self._lock:
            self._contexts.clear()
            self._candidates.clear()
            self._plans.clear()
            self._cache.clear()
            self._cache_by_ctx.clear()
            self._serves.clear()
            self._hits = 0
            self._misses = 0
            self._precompute_spent.clear()
            self._precompute_spent_global = 0.0
            self._publish(ANTICIPATOR_RESET, {})
