r"""Sketcher — bounded-memory streaming sketches as a runtime primitive.

Every other primitive in this runtime assumes someone, somewhere, can
hold the whole dataset in memory: ``Predictor`` keeps every prefix of
its stream, ``Forecaster`` keeps every prediction-target pair,
``DriftSentinel`` keeps a reference window of arbitrary size,
``Auditor`` keeps every event, ``Calibration`` keeps every score.  At
laboratory scale that assumption is fine.  At runtime scale —
millions of events per second arriving over weeks of autonomous
operation through a coordination engine — it is fatal.  A real
runtime cannot keep everything; it must keep a *sketch* with
provable, finite-sample-valid error bounds.

``Sketcher`` is the runtime's **bounded-memory streaming primitive**.
Given a stream of items and a sketch kind, it produces an answer
together with an explicit (ε, δ) error certificate, a byte count of
the state it consumed, and an HMAC over the sketch state for
tamper-evidence.  The pitch reduced to a runtime call:

  * the coordination engine pipes events through
    ``Sketcher.count_min(...).update(item, w)`` and the sketch answers
    ``query(item)`` with an over-estimate bounded by
    ``ε ‖f‖₁`` with probability ``≥ 1 − δ`` —
    *for every item simultaneously* (Cormode-Muthukrishnan 2005);

  * the engine pipes the same stream through
    ``Sketcher.misra_gries(k)`` and gets, *deterministically*, every
    item whose true frequency exceeds ``N / (k + 1)``, with
    additive error at most ``N / (k + 1)`` on the rest
    (Misra-Gries 1982; Bose-Kranakis-Morin-Tang 2003);

  * the engine asks ``Sketcher.hll(p)`` for the *cardinality* of the
    stream and gets an estimate whose relative standard error is
    ``≈ 1.04 / √(2^p)`` — using only ``2^p`` 5- or 6-bit registers
    (Flajolet-Fusy-Gandouet-Meunier 2007);

  * the engine asks ``Sketcher.kll(k)`` for any *quantile* of the
    stream and gets ε-additive-rank answers with
    ``ε = O(1 / k · √log(1/δ))`` — provably optimal up to
    constants (Karnin-Lang-Liberty 2016);

  * the engine asks ``Sketcher.reservoir(k)`` for a uniform random
    sample of size ``k`` from the unbounded stream (Vitter 1985's
    Algorithm R), or ``weighted_reservoir`` for a weighted-without-
    replacement sample (Efraimidis-Spirakis 2006's A-ExpJ);

  * the engine asks ``Sketcher.bloom(...)`` for *probabilistic set
    membership* with a one-sided false-positive rate controlled by
    a Bloom (1970) filter, or ``exp_histogram`` for *sliding-window*
    counts under a fixed time-budget with relative error ε
    (Datar-Gionis-Indyk-Motwani 2002).

Every sketch is **mergeable** where the underlying algorithm admits a
mergeable summary — Misra-Gries, Count-Min, Count-Sketch, HLL, KLL,
Bloom — meaning a distributed coordination engine can shard a stream
across N workers, sketch independently, and combine the sketches into
one answer of the *same* asymptotic quality as a serial sketch with
the same configuration.  This is the runtime's compositional, scalable
read of unbounded state.

Mathematical roots
------------------

  * **Misra, J. & Gries, D. (1982) — "Finding repeated elements."**
    *Science of Computer Programming* 2(2) 143–152.  The original
    deterministic heavy-hitter algorithm: ``k`` counters, one pass,
    every item with true frequency above ``N / (k + 1)`` survives,
    underestimate at most ``N / (k + 1)`` on every item.

  * **Boyer, R. S. & Moore, J S. (1981) — "MJRTY — A fast majority
    vote algorithm."**  Technical report, U. of Texas.  The
    ``k = 1`` special case: linear-time, constant-space majority
    detection.  Misra-Gries is the natural generalisation.

  * **Bose, P., Kranakis, E., Morin, P. & Tang, Y. (2003) — "Bounds
    for frequency estimation of packet streams."**  *SIROCCO*.
    Proves the additive-error bound and shows the sketch is mergeable
    by pairing-and-decrementing.

  * **Cormode, G. & Muthukrishnan, S. (2005) — "An improved data
    stream summary: the count-min sketch and its applications."**
    *Journal of Algorithms* 55(1) 58–75.  ``d × w`` counter array with
    independent hash functions; choosing ``w = ⌈e/ε⌉``,
    ``d = ⌈ln(1/δ)⌉`` gives ε-additive over-estimate with
    probability ≥ 1 − δ.  The textbook randomised sketch.

  * **Estan, C. & Varghese, G. (2003) — "New directions in traffic
    measurement and accounting."**  *SIGCOMM*.  Introduces the
    *conservative update* heuristic — only increment the *minimum*
    counters on insertion — which strictly improves point queries
    without breaking the upper-bound guarantee.

  * **Charikar, M., Chen, K. & Farach-Colton, M. (2002) — "Finding
    frequent items in data streams."**  *Theor. Comput. Sci.*
    312(1) 3–15.  Count-Sketch: signed hash + median estimator;
    unlike Count-Min the estimate is *unbiased* and the error is
    one-sided in the ℓ₂ rather than ℓ₁ norm.

  * **Alon, N., Matias, Y. & Szegedy, M. (1996) — "The space
    complexity of approximating the frequency moments."**  *STOC*.
    The "tug-of-war" sketch for ``F_2``: ``E[(±1 · f)²] = F_2``;
    median-of-means tightens to ``(ε, δ)``.

  * **Flajolet, P., Fusy, É., Gandouet, O. & Meunier, F. (2007) —
    "HyperLogLog: the analysis of a near-optimal cardinality
    estimation algorithm."**  *AofA*.  Replaces the bias-correction
    pile in Loglog/SuperLogLog by a single harmonic-mean estimator
    whose relative standard error is ``1.04 / √m`` for ``m``
    registers of ``log₂ log₂ N`` bits.

  * **Heule, S., Nunkesser, M. & Hall, A. (2013) — "HyperLogLog in
    practice: algorithmic engineering of a state of the art
    cardinality estimation algorithm."**  *EDBT*.  HLL++ — sparse
    representation, bias-corrected small-range estimator using
    Google's empirical bias table.  ``Sketcher`` uses the *linear
    counting* (Whang et al. 1990) correction at small ``E̅``.

  * **Karnin, Z., Lang, K. & Liberty, E. (2016) — "Optimal quantile
    approximation in streams."**  *FOCS*.  KLL: log-structured
    compactors with random eviction.  Achieves
    ``ε = O(√log(1/δ) / k)`` while every previous mergeable sketch
    spent a factor of ``log N`` more.  Mergeable, deterministic
    space.

  * **Greenwald, M. & Khanna, S. (2001) — "Space-efficient online
    computation of quantile summaries."**  *SIGMOD*.  Deterministic
    quantile sketch with ``O(log(εN) / ε)`` space.  Included as a
    deterministic complement to KLL.

  * **Vitter, J. S. (1985) — "Random sampling with a reservoir."**
    *ACM TOMS* 11(1) 37–57.  Algorithm R: a uniform random sample of
    size ``k`` from a stream of unknown length, in one pass, with
    ``O(k)`` memory.

  * **Efraimidis, P. S. & Spirakis, P. G. (2006) — "Weighted random
    sampling with a reservoir."**  *Inf. Proc. Letters* 97 181–185.
    The A-Res / A-ExpJ algorithms for weighted without-replacement
    sampling: keep the items with the top-``k`` priorities
    ``u_i^{1/w_i}``, with an exponential-jump speed-up.

  * **Bloom, B. H. (1970) — "Space/time trade-offs in hash coding
    with allowable errors."**  *CACM* 13(7) 422–426.  ``m`` bits,
    ``k`` hash functions: the canonical probabilistic set membership
    structure.  False-positive rate ≈ ``(1 − e^{−kn/m})^k`` after
    inserting ``n`` items.

  * **Datar, M., Gionis, A., Indyk, P. & Motwani, R. (2002) —
    "Maintaining stream statistics over sliding windows."**  *SIAM
    J. Comput.* 31(6) 1794–1813.  Exponential histograms: an
    ε-relative-error count over the last ``N`` items / time units
    with ``O((1/ε) log²(εN))`` space, no need to materialise the
    window.

  * **Muthukrishnan, S. (2005) — *Data streams: algorithms and
    applications.*** Foundations and Trends in Theoretical Computer
    Science 1(2).  The standard reference unifying all of the above
    under the streaming model.

  * **Cormode, G., Garofalakis, M., Haas, P. J. & Jermaine, C.
    (2012) — "Synopses for massive data."**  *Foundations and Trends
    in Databases* 4(1–3).  The mergeable / linear-sketch / random-
    sampling taxonomy that ``Sketcher`` follows.

What Sketcher gives a coordination engine
-----------------------------------------

It gives the coordinator a *single*, mathematically rigorous,
finite-sample-valid answer to the basic question every runtime keeps
asking but no other primitive answers: **"You have already seen too
much to keep, and you cannot stop seeing more — what can you still
say, and how wrong might you be?"**

  * For every sketch the answer is *not* a point estimate; it is an
    estimate paired with an explicit ``(ε, δ)`` error certificate,
    computed from the sketch's actual configuration, not assumed.

  * For every mergeable sketch the answer is *associative*: a
    distributed coordination engine can shard the stream across
    workers without changing the guarantee.

  * For every sketch the state is *measured in bytes* and exposed on
    the report.  A coordinator that wants to admit/deny a stream
    based on its memory budget has a number to gate on.

  * Every report carries a ``certificate`` HMAC over the canonical
    state; a coordinator publishing aggregate statistics has a
    tamper-evident record of the sketch it produced.

Public API
----------

The module exposes one configuration object (``SketcherConfig``),
one report (``SketcherReport``), one orchestrator class (``Sketcher``)
holding the active sketch, and one stateless convenience function per
sketch kind (``count_min``, ``misra_gries``, ``hyperloglog``, ``kll``,
``reservoir``, ``weighted_reservoir``, ``bloom``, ``exp_histogram``,
``count_sketch``, ``f2_sketch``) that return a ready-to-use sketch.

Every sketch class shares the same lifecycle:

    sk = Sketcher.count_min(epsilon=1e-3, delta=1e-3)
    for item, w in stream:
        sk.update(item, w)
    report = sk.report()
    # report.estimate, report.epsilon, report.delta,
    # report.n_items, report.n_bytes, report.certificate

The orchestrator is intentionally thin: the algorithms are the
contract, the orchestrator is just a uniform façade for a
coordination engine.

This module is **pure stdlib** — no NumPy, no SciPy — because the
runtime ships into environments where adding a heavy numerics
dependency to count items in a stream would be absurd.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import struct
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# =============================================================================
# Errors
# =============================================================================


class SketcherError(Exception):
    """Base for every Sketcher-raised error."""


class InvalidConfig(SketcherError):
    """A SketcherConfig is structurally invalid."""


class InvalidUpdate(SketcherError):
    """An update was rejected (e.g. negative weight on a Bloom filter)."""


class InvalidQuery(SketcherError):
    """A query was structurally invalid for this sketch."""


class NotMergeable(SketcherError):
    """The sketch kind does not admit a mergeable summary."""


class IncompatibleSketch(SketcherError):
    """Two sketches that should be mergeable have incompatible parameters."""


# =============================================================================
# Sketch kinds — string tags used by SketcherConfig.kind
# =============================================================================


KIND_MISRA_GRIES = "misra_gries"
KIND_COUNT_MIN = "count_min"
KIND_COUNT_SKETCH = "count_sketch"
KIND_HLL = "hll"
KIND_KLL = "kll"
KIND_GK = "gk"
KIND_RESERVOIR = "reservoir"
KIND_WEIGHTED_RESERVOIR = "weighted_reservoir"
KIND_BLOOM = "bloom"
KIND_EXP_HISTOGRAM = "exp_histogram"
KIND_F2_SKETCH = "f2_sketch"

KNOWN_KINDS: Tuple[str, ...] = (
    KIND_MISRA_GRIES,
    KIND_COUNT_MIN,
    KIND_COUNT_SKETCH,
    KIND_HLL,
    KIND_KLL,
    KIND_GK,
    KIND_RESERVOIR,
    KIND_WEIGHTED_RESERVOIR,
    KIND_BLOOM,
    KIND_EXP_HISTOGRAM,
    KIND_F2_SKETCH,
)

# Which kinds admit a mergeable summary
MERGEABLE_KINDS: frozenset = frozenset({
    KIND_MISRA_GRIES,
    KIND_COUNT_MIN,
    KIND_COUNT_SKETCH,
    KIND_HLL,
    KIND_KLL,
    KIND_GK,
    KIND_BLOOM,
    KIND_F2_SKETCH,
})


# =============================================================================
# Hash helpers — pure stdlib, seedable, deterministic
# =============================================================================


_MASK64 = (1 << 64) - 1


def _splitmix64(state: int) -> int:
    """SplitMix64 — Vigna's bit mixer.

    A non-zero state ``z`` is mixed by

        z = (z + 0x9E3779B97F4A7C15) mod 2^64
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 mod 2^64
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB mod 2^64
        z = z ^ (z >> 31)

    Used here purely to scramble an integer seed before handing it
    to xorshift, so that small consecutive seeds produce strongly
    decorrelated streams.
    """
    z = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return z & _MASK64


def _scramble_seed(seed: int, salt: int) -> int:
    """Produce a 64-bit non-zero state by mixing ``seed`` and ``salt``."""
    s = _splitmix64((seed * 0xDEADBEEFCAFEBABE) ^ salt)
    if s == 0:
        s = 0xA5A5A5A5A5A5A5A5
    return s


def _hash_bytes(salt: int, payload: bytes) -> int:
    """Stable 64-bit hash derived from BLAKE2b(salt | payload).

    BLAKE2 is keyed by the 8-byte little-endian encoding of ``salt``,
    so different ``salt`` values give pairwise-independent (in
    practice) hash functions for free.
    """
    h = hashlib.blake2b(digest_size=8, key=struct.pack("<Q", salt & ((1 << 64) - 1)))
    h.update(payload)
    return int.from_bytes(h.digest(), "little", signed=False)


def _canonical_item_bytes(item: Any) -> bytes:
    """Map an arbitrary hashable item to a canonical byte string.

    The point is *not* cryptographic — it is repeatability: two
    runs of the runtime must produce the same hash for the same
    item, even across Python invocations.  ``hash(...)`` is unstable
    across runs, so we go through ``repr`` for primitives and
    explicit byte-strings for everything else.
    """
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, str):
        return item.encode("utf-8")
    if isinstance(item, bool):
        return b"\x01" if item else b"\x00"
    if isinstance(item, int):
        # signed two's-complement, big-endian, length-prefixed
        nbytes = max(1, (item.bit_length() + 8) // 8)
        return b"i" + item.to_bytes(nbytes, "big", signed=True)
    if isinstance(item, float):
        return b"f" + struct.pack(">d", item)
    if isinstance(item, tuple):
        return b"t" + b"|".join(_canonical_item_bytes(e) for e in item)
    if isinstance(item, frozenset):
        return b"s" + b"|".join(sorted(_canonical_item_bytes(e) for e in item))
    return b"r" + repr(item).encode("utf-8")


def _twin_hash(salt: int, item: Any) -> Tuple[int, int]:
    """Return a 64-bit hash and a ±1 sign drawn from the same salt.

    Used by Count-Sketch and the AMS / F₂ tug-of-war sketch.  The
    sign is the high bit of the 64-bit hash.
    """
    h = _hash_bytes(salt, _canonical_item_bytes(item))
    sign = 1 if (h >> 63) & 1 == 0 else -1
    return h, sign


# =============================================================================
# Config and Report
# =============================================================================


@dataclass(frozen=True)
class SketcherConfig:
    """Static configuration for a single Sketcher instance.

    The kind tag selects the underlying sketch; the remaining fields
    are interpreted per-kind.  ``__post_init__`` checks that every
    selected field is structurally valid for the chosen kind.
    """

    kind: str

    # General
    seed: int = 0

    # Heavy-hitters (Misra-Gries)
    capacity: int = 0          # number of counters / sample slots / registers

    # Randomised counting (Count-Min, Count-Sketch)
    epsilon: float = 0.0
    delta: float = 0.0
    conservative_update: bool = False  # Estan-Varghese for Count-Min only

    # HyperLogLog
    precision: int = 0         # number of register bits: m = 2^precision

    # KLL / GK quantiles
    quantile_k: int = 0

    # Bloom filter
    bloom_capacity: int = 0    # n: expected number of insertions
    bloom_fpr: float = 0.0     # target false-positive rate

    # Exponential histogram (sliding window)
    window_size: int = 0       # N
    window_epsilon: float = 0.0  # ε

    # AMS / F2 sketch
    f2_rows: int = 0
    f2_cols: int = 0

    # Reservoir-style sketches use ``capacity``

    def __post_init__(self) -> None:
        if self.kind not in KNOWN_KINDS:
            raise InvalidConfig(
                f"unknown sketch kind {self.kind!r}; "
                f"expected one of {KNOWN_KINDS!r}"
            )
        if self.seed < 0:
            raise InvalidConfig("seed must be non-negative")
        k = self.kind
        if k == KIND_MISRA_GRIES:
            if self.capacity < 1:
                raise InvalidConfig("misra_gries requires capacity >= 1")
        elif k in (KIND_COUNT_MIN, KIND_COUNT_SKETCH):
            if not (0.0 < self.epsilon < 1.0):
                raise InvalidConfig(f"{k} requires 0 < epsilon < 1")
            if not (0.0 < self.delta < 1.0):
                raise InvalidConfig(f"{k} requires 0 < delta < 1")
        elif k == KIND_HLL:
            if not (4 <= self.precision <= 18):
                raise InvalidConfig(
                    "hll requires 4 <= precision <= 18 "
                    "(reasonable engineering range)"
                )
        elif k in (KIND_KLL, KIND_GK):
            if self.quantile_k < 8:
                raise InvalidConfig(f"{k} requires quantile_k >= 8")
        elif k in (KIND_RESERVOIR, KIND_WEIGHTED_RESERVOIR):
            if self.capacity < 1:
                raise InvalidConfig(f"{k} requires capacity >= 1")
        elif k == KIND_BLOOM:
            if self.bloom_capacity < 1:
                raise InvalidConfig("bloom requires bloom_capacity >= 1")
            if not (0.0 < self.bloom_fpr < 1.0):
                raise InvalidConfig("bloom requires 0 < bloom_fpr < 1")
        elif k == KIND_EXP_HISTOGRAM:
            if self.window_size < 2:
                raise InvalidConfig(
                    "exp_histogram requires window_size >= 2"
                )
            if not (0.0 < self.window_epsilon < 1.0):
                raise InvalidConfig(
                    "exp_histogram requires 0 < window_epsilon < 1"
                )
        elif k == KIND_F2_SKETCH:
            if self.f2_rows < 1 or self.f2_cols < 4:
                raise InvalidConfig(
                    "f2_sketch requires f2_rows >= 1, f2_cols >= 4"
                )


@dataclass
class SketcherReport:
    """The standardised answer object returned by every sketch's
    ``report()`` method.

    ``estimate`` is sketch-specific (an int for cardinality / count,
    a tuple for quantiles, a list for samples / heavy-hitters);
    ``epsilon`` and ``delta`` are the *actual* certificate values
    that follow from the sketch's configuration, not the user's
    target.  ``n_bytes`` is the sketch's measured state footprint.
    ``certificate`` is an HMAC over the canonical state for
    tamper-evidence.
    """

    kind: str
    estimate: Any
    epsilon: float = 0.0
    delta: float = 0.0
    n_items: int = 0
    n_bytes: int = 0
    capacity: int = 0
    mergeable: bool = False
    certificate: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "estimate": self.estimate,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "n_items": self.n_items,
            "n_bytes": self.n_bytes,
            "capacity": self.capacity,
            "mergeable": self.mergeable,
            "certificate": self.certificate,
            "extra": dict(self.extra),
        }


# =============================================================================
# Base class
# =============================================================================


class _BaseSketch:
    """Common machinery shared by every concrete sketch.

    The base class owns the configuration, the item counter, the
    HMAC certificate seed and a uniform ``report()`` shape.  Each
    subclass overrides ``update``, ``_canonical_state_bytes``,
    ``_estimate``, and (where applicable) ``merge``.
    """

    KIND: str = ""

    def __init__(self, config: SketcherConfig) -> None:
        if config.kind != self.KIND:
            raise InvalidConfig(
                f"{type(self).__name__} expects kind={self.KIND!r}, "
                f"got {config.kind!r}"
            )
        self.config = config
        self.n_items: int = 0
        # Per-sketch HMAC key derived from the seed; not a security boundary,
        # only a tamper-evidence marker tied to the seed.
        self._hmac_key = struct.pack("<Q", config.seed & ((1 << 64) - 1))

    # -- subclass hooks ----------------------------------------------

    def update(self, item: Any, weight: float = 1.0) -> None:
        raise NotImplementedError

    def _canonical_state_bytes(self) -> bytes:
        raise NotImplementedError

    def _estimate(self) -> Any:
        raise NotImplementedError

    def n_bytes(self) -> int:
        return len(self._canonical_state_bytes())

    @property
    def mergeable(self) -> bool:
        return self.KIND in MERGEABLE_KINDS

    # -- public API --------------------------------------------------

    def update_many(self, items: Iterable[Any]) -> None:
        for it in items:
            self.update(it)

    def certificate(self) -> str:
        mac = hmac.new(self._hmac_key, digestmod=hashlib.blake2b)
        mac.update(self.KIND.encode())
        mac.update(b"|")
        mac.update(_canonical_item_bytes(self.config.seed))
        mac.update(b"|")
        mac.update(self._canonical_state_bytes())
        return mac.hexdigest()[:32]

    def report(self) -> SketcherReport:
        eps, dlt = self.epsilon_delta()
        return SketcherReport(
            kind=self.KIND,
            estimate=self._estimate(),
            epsilon=eps,
            delta=dlt,
            n_items=self.n_items,
            n_bytes=self.n_bytes(),
            capacity=self._capacity(),
            mergeable=self.mergeable,
            certificate=self.certificate(),
            extra=self._extra(),
        )

    def epsilon_delta(self) -> Tuple[float, float]:
        return (self.config.epsilon, self.config.delta)

    def _capacity(self) -> int:
        return self.config.capacity

    def _extra(self) -> Mapping[str, Any]:
        return {}

    # default merge — raise; mergeable kinds override
    def merge(self, other: "_BaseSketch") -> None:
        raise NotMergeable(
            f"sketch kind {self.KIND!r} is not mergeable"
        )

    def _require_compatible(self, other: "_BaseSketch") -> None:
        if other.KIND != self.KIND:
            raise IncompatibleSketch(
                f"cannot merge kind {self.KIND!r} with kind {other.KIND!r}"
            )


# =============================================================================
# 1. Misra-Gries — deterministic heavy hitters
# =============================================================================


class MisraGriesSketch(_BaseSketch):
    """Misra-Gries (1982): deterministic heavy hitters with ``k`` counters.

    Invariants
    ----------

    Let ``f(x)`` be the true count of item ``x`` after ``N`` updates,
    and ``f̂(x)`` be the sketch's reported count.  Then

        0 ≤ f(x) − f̂(x) ≤ N / (k + 1)         for every x.

    In particular every item with ``f(x) > N / (k + 1)`` survives in
    the sketch.  The sketch is **deterministic** — ε is exact, δ = 0.
    """

    KIND = KIND_MISRA_GRIES

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._k: int = config.capacity
        self._counters: dict = {}

    def update(self, item: Any, weight: float = 1.0) -> None:
        if weight <= 0:
            raise InvalidUpdate("misra_gries requires positive weight")
        w = int(weight) if float(weight).is_integer() else float(weight)
        self.n_items += int(weight) if isinstance(w, int) else 1
        if item in self._counters:
            self._counters[item] += w
            return
        if len(self._counters) < self._k:
            self._counters[item] = w
            return
        # Pairing-and-decrement step: subtract w from every counter,
        # dropping any that reach 0.  This is the canonical
        # generalisation to weighted updates while preserving the
        # additive error bound.
        to_drop = []
        for k_ in list(self._counters):
            self._counters[k_] -= w
            if self._counters[k_] <= 0:
                to_drop.append(k_)
        for k_ in to_drop:
            del self._counters[k_]

    # ---- queries ---------------------------------------------------

    def query(self, item: Any) -> float:
        """Return ``f̂(item)`` — guaranteed an underestimate of
        ``f(item)`` by at most ``N / (k + 1)``."""
        return float(self._counters.get(item, 0))

    def heavy_hitters(self, threshold_fraction: float = 0.0) -> List[Tuple[Any, float]]:
        """Return ``(item, f̂(item))`` for every item still in the
        sketch, sorted by counter descending.

        ``threshold_fraction`` (∈ [0, 1]) optionally filters to the
        items whose *underestimate* already exceeds
        ``threshold_fraction * N``.  Any item with true frequency
        above ``threshold_fraction + 1/(k+1)`` is guaranteed
        included; any item with true frequency below
        ``threshold_fraction`` is guaranteed excluded.
        """
        if not (0.0 <= threshold_fraction <= 1.0):
            raise InvalidQuery("threshold_fraction must be in [0, 1]")
        cutoff = threshold_fraction * float(self.n_items)
        out: List[Tuple[Any, float]] = [
            (k, float(v)) for k, v in self._counters.items() if float(v) >= cutoff
        ]
        out.sort(key=lambda kv: (-kv[1], _canonical_item_bytes(kv[0])))
        return out

    # ---- merge / report --------------------------------------------

    def merge(self, other: "_BaseSketch") -> None:
        self._require_compatible(other)
        assert isinstance(other, MisraGriesSketch)
        if other._k != self._k:
            raise IncompatibleSketch(
                f"misra_gries capacity mismatch: {self._k} vs {other._k}"
            )
        for k_, v in other._counters.items():
            self._counters[k_] = self._counters.get(k_, 0) + v
        self.n_items += other.n_items
        # Keep top-k by counter, shifting down by the (k+1)-th largest.
        if len(self._counters) <= self._k:
            return
        ordered = sorted(self._counters.items(), key=lambda kv: -kv[1])
        threshold = ordered[self._k][1]
        new: dict = {}
        for k_, v in ordered[: self._k]:
            shifted = v - threshold
            if shifted > 0:
                new[k_] = shifted
        self._counters = new

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [b"mg", self._k.to_bytes(8, "big")]
        for k_, v in sorted(
            self._counters.items(),
            key=lambda kv: _canonical_item_bytes(kv[0]),
        ):
            parts.append(_canonical_item_bytes(k_))
            if isinstance(v, int):
                parts.append(b"i" + v.to_bytes(8, "big", signed=True))
            else:
                parts.append(b"f" + struct.pack(">d", float(v)))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        return self.heavy_hitters()

    def epsilon_delta(self) -> Tuple[float, float]:
        if self.n_items == 0:
            return (0.0, 0.0)
        return (1.0 / (self._k + 1), 0.0)

    def _capacity(self) -> int:
        return self._k

    def _extra(self) -> Mapping[str, Any]:
        return {
            "additive_error_units": float(self.n_items) / (self._k + 1),
            "n_distinct_tracked": len(self._counters),
        }


# =============================================================================
# 2. Count-Min Sketch
# =============================================================================


class CountMinSketch(_BaseSketch):
    """Cormode-Muthukrishnan (2005) Count-Min sketch.

    A ``d × w`` integer counter array with ``d`` independent hash
    functions.  Update increments the ``d`` cells indexed by the
    item's hash row.  Point query returns the *minimum* of the
    ``d`` cells.  With ``w = ⌈e/ε⌉``, ``d = ⌈ln(1/δ)⌉``,

        P[ f̂(x) ≤ f(x) + ε · ‖f‖₁ ]  ≥  1 − δ.

    ``conservative_update`` (Estan-Varghese 2003): on insertion, only
    increment the cells whose current value equals the row-minimum
    for this item.  Strictly tighter on the point query while
    preserving the upper bound, but it breaks linearity (no longer
    mergeable).
    """

    KIND = KIND_COUNT_MIN

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._w = max(1, int(math.ceil(math.e / config.epsilon)))
        self._d = max(1, int(math.ceil(math.log(1.0 / config.delta))))
        self._table: List[List[float]] = [
            [0.0] * self._w for _ in range(self._d)
        ]
        self._salts: List[int] = [config.seed + 0xC1A1A111 * (i + 1) for i in range(self._d)]
        self._total_weight: float = 0.0
        self._conservative = bool(config.conservative_update)

    # ---- internals -------------------------------------------------

    def _row_cells(self, item: Any) -> List[int]:
        payload = _canonical_item_bytes(item)
        return [_hash_bytes(s, payload) % self._w for s in self._salts]

    # ---- updates / queries -----------------------------------------

    def update(self, item: Any, weight: float = 1.0) -> None:
        if weight <= 0:
            raise InvalidUpdate("count_min requires positive weight")
        cells = self._row_cells(item)
        self.n_items += 1
        self._total_weight += float(weight)
        if self._conservative:
            current = [self._table[i][c] for i, c in enumerate(cells)]
            target = min(current) + float(weight)
            for i, c in enumerate(cells):
                if self._table[i][c] < target:
                    self._table[i][c] = target
        else:
            for i, c in enumerate(cells):
                self._table[i][c] += float(weight)

    def query(self, item: Any) -> float:
        cells = self._row_cells(item)
        return min(self._table[i][c] for i, c in enumerate(cells))

    # ---- merge / report --------------------------------------------

    def merge(self, other: "_BaseSketch") -> None:
        if self._conservative:
            raise NotMergeable(
                "count_min with conservative_update is not mergeable"
            )
        self._require_compatible(other)
        assert isinstance(other, CountMinSketch)
        if other._w != self._w or other._d != self._d:
            raise IncompatibleSketch(
                f"count_min shape mismatch: ({self._d},{self._w}) "
                f"vs ({other._d},{other._w})"
            )
        if other._salts != self._salts:
            raise IncompatibleSketch(
                "count_min salts differ — incompatible hash families"
            )
        for i in range(self._d):
            row_a = self._table[i]
            row_b = other._table[i]
            for j in range(self._w):
                row_a[j] += row_b[j]
        self.n_items += other.n_items
        self._total_weight += other._total_weight

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"cm",
            self._d.to_bytes(4, "big"),
            self._w.to_bytes(4, "big"),
            (1 if self._conservative else 0).to_bytes(1, "big"),
        ]
        for s in self._salts:
            parts.append(s.to_bytes(8, "big", signed=True))
        for row in self._table:
            for v in row:
                parts.append(struct.pack(">d", float(v)))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        return {"total_weight": self._total_weight, "rows": self._d, "cols": self._w}

    def epsilon_delta(self) -> Tuple[float, float]:
        return (math.e / self._w, math.exp(-self._d))

    @property
    def mergeable(self) -> bool:
        return not self._conservative

    def _capacity(self) -> int:
        return self._d * self._w

    def _extra(self) -> Mapping[str, Any]:
        return {
            "rows": self._d,
            "cols": self._w,
            "total_weight": self._total_weight,
            "conservative_update": self._conservative,
        }


# =============================================================================
# 3. Count Sketch (Charikar-Chen-Farach-Colton 2002)
# =============================================================================


class CountSketch(_BaseSketch):
    """Count-Sketch with signed hashes and the median estimator.

    For each of ``d`` independent (hash, sign) pairs, increment the
    cell indexed by the hash by ``sign · weight``.  The unbiased
    per-row estimate ``sign · table[i][hash]`` averages over signs;
    taking the median across rows gives error
    ``ε ‖f‖₂`` with probability ``≥ 1 − δ`` (Charikar et al.).
    """

    KIND = KIND_COUNT_SKETCH

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._w = max(1, int(math.ceil(3.0 / (config.epsilon ** 2))))
        self._d = max(1, int(math.ceil(math.log(1.0 / config.delta))))
        # Force odd d so the median is well-defined.
        if self._d % 2 == 0:
            self._d += 1
        self._table: List[List[float]] = [
            [0.0] * self._w for _ in range(self._d)
        ]
        self._h_salts: List[int] = [config.seed + 0xC0DE0001 * (i + 1) for i in range(self._d)]
        self._s_salts: List[int] = [config.seed + 0xB0BA0007 * (i + 1) for i in range(self._d)]

    def update(self, item: Any, weight: float = 1.0) -> None:
        payload = _canonical_item_bytes(item)
        self.n_items += 1
        for i in range(self._d):
            col = _hash_bytes(self._h_salts[i], payload) % self._w
            sign = 1 if _hash_bytes(self._s_salts[i], payload) & 1 == 0 else -1
            self._table[i][col] += sign * float(weight)

    def query(self, item: Any) -> float:
        payload = _canonical_item_bytes(item)
        estimates: List[float] = []
        for i in range(self._d):
            col = _hash_bytes(self._h_salts[i], payload) % self._w
            sign = 1 if _hash_bytes(self._s_salts[i], payload) & 1 == 0 else -1
            estimates.append(sign * self._table[i][col])
        estimates.sort()
        return estimates[len(estimates) // 2]

    def merge(self, other: "_BaseSketch") -> None:
        self._require_compatible(other)
        assert isinstance(other, CountSketch)
        if other._w != self._w or other._d != self._d:
            raise IncompatibleSketch("count_sketch shape mismatch")
        if other._h_salts != self._h_salts or other._s_salts != self._s_salts:
            raise IncompatibleSketch("count_sketch hash families differ")
        for i in range(self._d):
            row_a = self._table[i]
            row_b = other._table[i]
            for j in range(self._w):
                row_a[j] += row_b[j]
        self.n_items += other.n_items

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"cs",
            self._d.to_bytes(4, "big"),
            self._w.to_bytes(4, "big"),
        ]
        for s in self._h_salts + self._s_salts:
            parts.append(s.to_bytes(8, "big", signed=True))
        for row in self._table:
            for v in row:
                parts.append(struct.pack(">d", float(v)))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        # F_2 estimate (Charikar et al.): for each row, sum of squares
        # is an unbiased estimate of F_2; median across rows.
        f2_per_row = [sum(c * c for c in row) for row in self._table]
        f2_per_row.sort()
        f2 = f2_per_row[len(f2_per_row) // 2]
        return {"f2": f2, "rows": self._d, "cols": self._w}

    def epsilon_delta(self) -> Tuple[float, float]:
        return (math.sqrt(3.0 / self._w), math.exp(-self._d))

    def _capacity(self) -> int:
        return self._d * self._w

    def _extra(self) -> Mapping[str, Any]:
        return {"rows": self._d, "cols": self._w}


# =============================================================================
# 4. HyperLogLog (Flajolet et al. 2007 + HLL++)
# =============================================================================


# Empirical alpha constants from Flajolet et al. for the harmonic
# mean estimator; we use the m≥128 closed form ``α_∞ = 0.7213 /
# (1 + 1.079 / m)`` from the original paper.
def _hll_alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


class HyperLogLogSketch(_BaseSketch):
    """Flajolet et al. (2007) HyperLogLog with HLL++ small-range
    linear-counting correction (Heule et al. 2013).

    For each item, hash to 64 bits, split into a ``p``-bit register
    index and a ``64-p``-bit body; track the maximum leading-zero
    count plus one per register.  The cardinality estimate is the
    harmonic-mean estimator on the registers.  At low estimated
    cardinality (E ≤ 2.5 m) the linear-counting estimator
    ``m ln(m / V)`` is used instead, where ``V`` is the number of
    empty registers.
    """

    KIND = KIND_HLL

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._p = config.precision
        self._m = 1 << self._p
        self._registers: List[int] = [0] * self._m
        self._salt = config.seed + 0xF1A50007

    def update(self, item: Any, weight: float = 1.0) -> None:
        # Weight is treated multiplicatively but only insertion
        # matters for cardinality; we record a single update per call
        # — repeated insertions of the same item do not increase
        # cardinality, which is the whole point.
        h = _hash_bytes(self._salt, _canonical_item_bytes(item))
        # Top p bits → register index.
        idx = h >> (64 - self._p)
        # Remaining 64-p bits → body; rho = position of leftmost 1 + 1
        body = h & ((1 << (64 - self._p)) - 1)
        if body == 0:
            rho = 64 - self._p + 1
        else:
            rho = (body.bit_length() ^ (64 - self._p)) + 1
            # equivalent: leading zeros of (64-p)-bit body
            rho = (64 - self._p) - body.bit_length() + 1
        if rho > self._registers[idx]:
            self._registers[idx] = rho
        self.n_items += 1

    def cardinality(self) -> float:
        m = self._m
        alpha = _hll_alpha(m)
        Z = 0.0
        V = 0
        for r in self._registers:
            Z += 1.0 / (1 << r) if r > 0 else 1.0
            if r == 0:
                V += 1
        E = alpha * m * m / Z
        # Small-range correction: linear counting when E ≤ 5/2 * m
        # and V > 0.
        if E <= 2.5 * m and V > 0:
            return float(m * math.log(m / V))
        # Large-range bias of plain HLL near 2^64 is negligible at
        # any realistic stream size; skip it.
        return float(E)

    def merge(self, other: "_BaseSketch") -> None:
        self._require_compatible(other)
        assert isinstance(other, HyperLogLogSketch)
        if other._p != self._p:
            raise IncompatibleSketch(
                f"hll precision mismatch: {self._p} vs {other._p}"
            )
        if other._salt != self._salt:
            raise IncompatibleSketch("hll salt differs")
        for i in range(self._m):
            if other._registers[i] > self._registers[i]:
                self._registers[i] = other._registers[i]
        self.n_items += other.n_items

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"hll",
            self._p.to_bytes(1, "big"),
            self._salt.to_bytes(8, "big", signed=True),
        ]
        # Pack registers as bytes — they fit in 6 bits at any
        # reasonable precision but 1-byte each keeps things stdlib.
        parts.append(bytes(min(r, 255) for r in self._registers))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        return self.cardinality()

    def epsilon_delta(self) -> Tuple[float, float]:
        # Asymptotic RSE of HLL is 1.04 / sqrt(m); we report it as
        # (epsilon, 0).  The Gaussian tail at one standard deviation
        # gives an approximate δ of ~0.317, but the conventional way
        # to publish the guarantee is as a relative standard error.
        rse = 1.04 / math.sqrt(self._m)
        return (rse, 0.317)

    def _capacity(self) -> int:
        return self._m

    def _extra(self) -> Mapping[str, Any]:
        return {
            "precision": self._p,
            "registers": self._m,
            "nonzero_registers": sum(1 for r in self._registers if r > 0),
        }


# =============================================================================
# 5. KLL — optimal mergeable quantile sketch
# =============================================================================


def _kll_capacity_for(level: int, k: int) -> int:
    """Capacity for level ``level`` in the KLL compactor stack.

    KLL uses geometric-decay capacities ``c_i = ⌈k · c^i⌉`` for a
    decay ratio ``c`` slightly less than 1.  We follow the
    simplified KLL+ recommendation ``c = 2/3`` and floor at 2 — the
    smallest capacity that still permits the pair-and-promote
    compaction step.  A higher floor lets items in very high levels
    (each carrying very high weight ``2^level``) dominate the
    aggregate, biasing tail-quantile estimates.
    """
    return max(2, int(math.ceil(k * (2.0 / 3.0) ** level)))


class KLLSketch(_BaseSketch):
    """Karnin-Lang-Liberty (2016) optimal mergeable quantile sketch.

    Maintains a stack of *compactors*.  Each compactor at level ``i``
    holds at most ``c_i`` items.  When full, it sorts its items,
    picks a random parity (even or odd indexes), forwards those
    items to level ``i+1``, and discards the rest.  The forwarded
    items have effective weight ``2^i``.  Querying a rank or
    quantile aggregates over all compactors weighted by their level.

    Guarantee (KLL): for ``ε ≈ 1/k · √log(1/δ)``,

        P[ |r̂(x) − r(x)| > ε · N ]  ≤  δ
        simultaneously for every x.

    The sketch is **mergeable** and accepts both sided-rank and
    quantile queries.
    """

    KIND = KIND_KLL

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._k = config.quantile_k
        self._levels: List[List[float]] = [[]]
        # Per-instance deterministic PRNG so updates are reproducible.
        self._rng_state = _scramble_seed(config.seed, 0xCA11AB1E)

    # ---- internal PRNG -- a 64-bit xorshift, stdlib-only -----------

    def _rand_bit(self) -> int:
        x = self._rng_state
        x ^= (x << 13) & ((1 << 64) - 1)
        x ^= (x >> 7) & ((1 << 64) - 1)
        x ^= (x << 17) & ((1 << 64) - 1)
        self._rng_state = x & ((1 << 64) - 1)
        return x & 1

    # ---- compaction ------------------------------------------------

    def _compact_level(self, level: int) -> None:
        comp = self._levels[level]
        comp.sort()
        # Standard KLL compaction: pair up items, promote one from
        # each pair (random parity), discard the other.  An odd
        # item, if any, stays at the current level so that weight is
        # preserved exactly.  The orphan must be chosen *randomly*
        # (here from the two endpoints) to avoid a systematic
        # high-bias on sorted streams — always keeping ``comp[-1]``
        # would push the largest value back to level 0 every time.
        n = len(comp)
        if n & 1:
            # Random endpoint orphan: either the smallest or the
            # largest item.  Each choice is unbiased on average.
            if self._rand_bit():
                orphan = [comp[-1]]
                rest = comp[:-1]
            else:
                orphan = [comp[0]]
                rest = comp[1:]
        else:
            orphan = []
            rest = comp
        pairs = len(rest) // 2
        parity = self._rand_bit()
        promoted = [rest[2 * i + parity] for i in range(pairs)]
        self._levels[level] = list(orphan)
        if level + 1 >= len(self._levels):
            self._levels.append([])
        self._levels[level + 1].extend(promoted)

    def _maybe_compact(self) -> None:
        # Repeatedly compact any level whose size exceeds its
        # capacity, including new levels created by promotion.
        changed = True
        while changed:
            changed = False
            level = 0
            while level < len(self._levels):
                cap = _kll_capacity_for(level, self._k)
                if len(self._levels[level]) > cap:
                    self._compact_level(level)
                    changed = True
                level += 1

    def update(self, item: Any, weight: float = 1.0) -> None:
        # Only numeric items are quantile-orderable.
        if not isinstance(item, (int, float)):
            raise InvalidUpdate("kll requires numeric items")
        self._levels[0].append(float(item))
        self.n_items += 1
        self._maybe_compact()

    # ---- queries ---------------------------------------------------

    def _items_with_weights(self) -> Iterator[Tuple[float, int]]:
        for level, comp in enumerate(self._levels):
            w = 1 << level
            for v in comp:
                yield v, w

    def rank(self, x: float) -> float:
        """Estimated rank of ``x``: the number of items ≤ x."""
        total = 0
        for v, w in self._items_with_weights():
            if v <= x:
                total += w
        return float(total)

    def quantile(self, q: float) -> float:
        """Estimated value at quantile ``q ∈ [0, 1]``."""
        if not (0.0 <= q <= 1.0):
            raise InvalidQuery("quantile requires q in [0, 1]")
        if self.n_items == 0:
            raise InvalidQuery("no items in sketch")
        items = sorted(self._items_with_weights())
        target = q * self.n_items
        cum = 0
        for v, w in items:
            cum += w
            if cum >= target:
                return v
        return items[-1][0]

    def cdf(self, x: float) -> float:
        """Estimated CDF: P[X ≤ x]."""
        if self.n_items == 0:
            return 0.0
        return self.rank(x) / self.n_items

    # ---- merge / report --------------------------------------------

    def merge(self, other: "_BaseSketch") -> None:
        self._require_compatible(other)
        assert isinstance(other, KLLSketch)
        if other._k != self._k:
            raise IncompatibleSketch(
                f"kll k mismatch: {self._k} vs {other._k}"
            )
        # Extend levels to match the longer of the two.
        while len(self._levels) < len(other._levels):
            self._levels.append([])
        for i, lvl in enumerate(other._levels):
            self._levels[i].extend(lvl)
        self.n_items += other.n_items
        self._maybe_compact()

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"kll",
            self._k.to_bytes(4, "big"),
            len(self._levels).to_bytes(2, "big"),
        ]
        for lvl in self._levels:
            parts.append(len(lvl).to_bytes(4, "big"))
            for v in sorted(lvl):
                parts.append(struct.pack(">d", v))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        if self.n_items == 0:
            return {"q50": None, "q90": None, "q99": None}
        return {
            "q50": self.quantile(0.50),
            "q90": self.quantile(0.90),
            "q99": self.quantile(0.99),
        }

    def epsilon_delta(self) -> Tuple[float, float]:
        # KLL guarantee.  log(1/δ) factor inside the radical, so the
        # exposed (ε, δ) report fixes δ at e^(-k/12) -- a standard
        # parameterisation -- and reports the resulting ε.
        delta = math.exp(-self._k / 12.0)
        eps = math.sqrt(max(1e-9, math.log(1.0 / max(delta, 1e-12)))) / self._k
        return (eps, delta)

    def _capacity(self) -> int:
        return sum(_kll_capacity_for(i, self._k) for i in range(len(self._levels)))

    def _extra(self) -> Mapping[str, Any]:
        return {
            "levels": len(self._levels),
            "items_retained": sum(len(l) for l in self._levels),
        }


# =============================================================================
# 6. Greenwald-Khanna — deterministic quantiles
# =============================================================================


class GKSketch(_BaseSketch):
    """Greenwald-Khanna (2001) deterministic quantile sketch.

    Each tuple stores ``(value, g, Δ)`` where ``g`` is the rank gap
    to the previous tuple and ``Δ`` is the slack.  The invariant
    ``g_i + Δ_i ≤ 2 ε N`` is maintained by a compress step that
    merges adjacent tuples whose combined slack still satisfies it.

    Guarantee: rank queries are answered within additive error
    ``ε N``, deterministically.
    """

    KIND = KIND_GK

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        # ε is implicit in k: roughly ε = 1 / k.
        self._eps = 1.0 / config.quantile_k
        # ``tuples`` is a sorted list of (value, g, delta).
        self._tuples: List[Tuple[float, int, int]] = []
        self._compress_every = max(1, int(1.0 / (2.0 * self._eps)))
        self._steps_since_compress = 0

    def update(self, item: Any, weight: float = 1.0) -> None:
        if not isinstance(item, (int, float)):
            raise InvalidUpdate("gk requires numeric items")
        v = float(item)
        self.n_items += 1
        # Find insertion point.
        lo, hi = 0, len(self._tuples)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._tuples[mid][0] < v:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx == 0 or idx == len(self._tuples):
            self._tuples.insert(idx, (v, 1, 0))
        else:
            band = max(0, int(math.floor(2.0 * self._eps * self.n_items)) - 1)
            self._tuples.insert(idx, (v, 1, band))
        self._steps_since_compress += 1
        if self._steps_since_compress >= self._compress_every:
            self._compress()
            self._steps_since_compress = 0

    def _compress(self) -> None:
        if len(self._tuples) <= 2:
            return
        limit = int(math.floor(2.0 * self._eps * self.n_items))
        i = len(self._tuples) - 2
        while i >= 1:
            v, g, dlt = self._tuples[i]
            v_next, g_next, dlt_next = self._tuples[i + 1]
            if g + g_next + dlt_next <= limit:
                # Merge i into i+1.
                self._tuples[i + 1] = (v_next, g + g_next, dlt_next)
                del self._tuples[i]
            i -= 1

    def quantile(self, q: float) -> float:
        if not (0.0 <= q <= 1.0):
            raise InvalidQuery("quantile requires q in [0, 1]")
        if not self._tuples:
            raise InvalidQuery("no items in sketch")
        target = q * self.n_items
        r_min = 0
        limit = int(math.floor(self._eps * self.n_items))
        prev = self._tuples[0]
        for i, (v, g, dlt) in enumerate(self._tuples):
            r_min += g
            r_max = r_min + dlt
            if r_max - target > limit and target - (r_min - g) > limit:
                # Skip — neither bound straddles the target with slack.
                prev = (v, g, dlt)
                continue
            if abs(target - r_min) <= limit or (r_min - g) <= target <= r_max:
                return v
            prev = (v, g, dlt)
        return self._tuples[-1][0]

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"gk",
            struct.pack(">d", self._eps),
            len(self._tuples).to_bytes(4, "big"),
        ]
        for v, g, dlt in self._tuples:
            parts.append(struct.pack(">d", v))
            parts.append(g.to_bytes(8, "big", signed=False))
            parts.append(dlt.to_bytes(8, "big", signed=False))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        if self.n_items == 0:
            return {"q50": None, "q90": None, "q99": None}
        return {
            "q50": self.quantile(0.50),
            "q90": self.quantile(0.90),
            "q99": self.quantile(0.99),
        }

    def epsilon_delta(self) -> Tuple[float, float]:
        return (self._eps, 0.0)

    def _capacity(self) -> int:
        return len(self._tuples)

    def _extra(self) -> Mapping[str, Any]:
        return {"tuples": len(self._tuples)}

    def merge(self, other: "_BaseSketch") -> None:
        self._require_compatible(other)
        assert isinstance(other, GKSketch)
        if abs(other._eps - self._eps) > 1e-12:
            raise IncompatibleSketch("gk epsilon mismatch")
        # Simple merge: replay items by sampled value.  For an exact
        # mergeable variant see Wang-Luo-Yi 2013 ("Quantiles over data
        # streams: an experimental study").  We keep the standard one.
        merged = sorted(self._tuples + other._tuples, key=lambda t: t[0])
        self._tuples = merged
        self.n_items += other.n_items
        self._compress()


# =============================================================================
# 7. Reservoir Sampling — Vitter's Algorithm R
# =============================================================================


class ReservoirSampler(_BaseSketch):
    """Vitter (1985) Algorithm R: uniform sample of size ``k`` from a
    stream of unknown length.

    Invariant: after seeing ``n`` items, every item in the stream is
    in the reservoir with probability exactly ``k/n``.
    """

    KIND = KIND_RESERVOIR

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._k = config.capacity
        self._reservoir: List[Any] = []
        self._rng_state = _scramble_seed(config.seed, 0xA10B0C0D)

    def _next_u01(self) -> float:
        x = self._rng_state
        x ^= (x << 13) & ((1 << 64) - 1)
        x ^= (x >> 7) & ((1 << 64) - 1)
        x ^= (x << 17) & ((1 << 64) - 1)
        self._rng_state = x & ((1 << 64) - 1)
        return self._rng_state / float(1 << 64)

    def update(self, item: Any, weight: float = 1.0) -> None:
        self.n_items += 1
        if len(self._reservoir) < self._k:
            self._reservoir.append(item)
            return
        j = int(self._next_u01() * self.n_items)
        if j < self._k:
            self._reservoir[j] = item

    def sample(self) -> List[Any]:
        return list(self._reservoir)

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"res",
            self._k.to_bytes(4, "big"),
            self.n_items.to_bytes(8, "big"),
        ]
        for item in self._reservoir:
            parts.append(_canonical_item_bytes(item))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        return self.sample()

    def epsilon_delta(self) -> Tuple[float, float]:
        if self.n_items == 0:
            return (0.0, 0.0)
        return (1.0 / math.sqrt(self._k), 0.0)

    def _capacity(self) -> int:
        return self._k

    def _extra(self) -> Mapping[str, Any]:
        return {"sample_size": len(self._reservoir)}


# =============================================================================
# 8. Weighted Reservoir — Efraimidis-Spirakis A-Res
# =============================================================================


class WeightedReservoirSampler(_BaseSketch):
    """Efraimidis-Spirakis (2006) Algorithm A-Res: weighted-without-
    replacement sample of size ``k``.

    For each item with weight ``w_i`` draw ``u_i ~ U(0,1)`` and key
    ``key_i = u_i^{1/w_i}``; keep the items with the top-``k`` keys.

    Equivalent (and what the original paper proves): the inclusion
    probability of item ``i`` is exactly the order-statistic of its
    key, which matches sampling without replacement with prob
    proportional to weight.
    """

    KIND = KIND_WEIGHTED_RESERVOIR

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._k = config.capacity
        # Min-heap by key: (key, counter, item)
        self._heap: List[Tuple[float, int, Any]] = []
        self._rng_state = _scramble_seed(config.seed, 0xF00DBABE)
        self._counter = 0

    def _next_u01(self) -> float:
        x = self._rng_state
        x ^= (x << 13) & ((1 << 64) - 1)
        x ^= (x >> 7) & ((1 << 64) - 1)
        x ^= (x << 17) & ((1 << 64) - 1)
        self._rng_state = x & ((1 << 64) - 1)
        return max(1e-300, self._rng_state / float(1 << 64))

    def update(self, item: Any, weight: float = 1.0) -> None:
        if weight <= 0:
            raise InvalidUpdate("weighted_reservoir requires positive weight")
        self.n_items += 1
        self._counter += 1
        u = self._next_u01()
        key = u ** (1.0 / weight)
        if len(self._heap) < self._k:
            self._heap.append((key, self._counter, item))
            self._heap.sort()  # ascending; heap[0] is smallest
            return
        if key > self._heap[0][0]:
            # Replace the smallest.
            self._heap[0] = (key, self._counter, item)
            self._heap.sort()

    def sample(self) -> List[Tuple[Any, float]]:
        # Return (item, key) sorted from largest key (most strongly
        # represented) to smallest.
        return [(item, key) for key, _, item in sorted(self._heap, reverse=True)]

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"wres",
            self._k.to_bytes(4, "big"),
            self.n_items.to_bytes(8, "big"),
        ]
        for key, ctr, item in sorted(self._heap):
            parts.append(struct.pack(">d", key))
            parts.append(ctr.to_bytes(8, "big"))
            parts.append(_canonical_item_bytes(item))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        return [item for item, _ in self.sample()]

    def _capacity(self) -> int:
        return self._k

    def _extra(self) -> Mapping[str, Any]:
        return {"sample_size": len(self._heap)}


# =============================================================================
# 9. Bloom Filter (Bloom 1970)
# =============================================================================


class BloomFilter(_BaseSketch):
    """Standard Bloom filter (Bloom 1970).

    Given target false-positive rate ``p`` and expected insertions
    ``n``, choose

        m = ⌈ −n ln p / (ln 2)² ⌉,   k = ⌈ (m/n) · ln 2 ⌉.

    Insertion sets ``k`` bits; membership test checks ``k`` bits.
    False negatives are impossible; the empirical FPR after
    inserting ``n_items`` items is reported on ``report()``.
    """

    KIND = KIND_BLOOM

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        n = config.bloom_capacity
        p = config.bloom_fpr
        ln2 = math.log(2.0)
        m = int(math.ceil(-n * math.log(p) / (ln2 ** 2)))
        k = int(math.ceil((m / n) * ln2))
        self._m = max(8, m)
        self._k = max(1, k)
        # Pack bits into a bytearray.
        self._bits = bytearray((self._m + 7) // 8)
        self._salts = [config.seed + 0xB100D000 * (i + 1) for i in range(self._k)]
        # Empirical insertion count (distinct from item count
        # because Bloom filters are idempotent on duplicates).
        self._inserts = 0

    def _bit_indices(self, item: Any) -> List[int]:
        payload = _canonical_item_bytes(item)
        return [_hash_bytes(s, payload) % self._m for s in self._salts]

    def update(self, item: Any, weight: float = 1.0) -> None:
        self.n_items += 1
        self._inserts += 1
        for b in self._bit_indices(item):
            self._bits[b >> 3] |= (1 << (b & 7))

    def __contains__(self, item: Any) -> bool:
        for b in self._bit_indices(item):
            if not (self._bits[b >> 3] & (1 << (b & 7))):
                return False
        return True

    def contains(self, item: Any) -> bool:
        return self.__contains__(item)

    def merge(self, other: "_BaseSketch") -> None:
        self._require_compatible(other)
        assert isinstance(other, BloomFilter)
        if other._m != self._m or other._k != self._k:
            raise IncompatibleSketch("bloom shape mismatch")
        if other._salts != self._salts:
            raise IncompatibleSketch("bloom hash families differ")
        for i in range(len(self._bits)):
            self._bits[i] |= other._bits[i]
        self.n_items += other.n_items
        self._inserts += other._inserts

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"bloom",
            self._m.to_bytes(4, "big"),
            self._k.to_bytes(4, "big"),
        ]
        for s in self._salts:
            parts.append(s.to_bytes(8, "big", signed=True))
        parts.append(bytes(self._bits))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        # Estimated population size from bit density: m * ln(1 - X/m) / -k
        # where X is the count of set bits (Swamidass-Baldi 2007).
        set_bits = 0
        for b in self._bits:
            v = b
            while v:
                set_bits += v & 1
                v >>= 1
        if set_bits == 0:
            return 0.0
        if set_bits >= self._m:
            return float("inf")
        density = set_bits / self._m
        # Avoid log(0) at saturation.
        if density >= 1.0 - 1e-12:
            return float("inf")
        return -float(self._m) * math.log(1.0 - density) / self._k

    def epsilon_delta(self) -> Tuple[float, float]:
        if self._inserts == 0:
            return (0.0, 0.0)
        empirical_fpr = (1.0 - math.exp(-self._k * self._inserts / self._m)) ** self._k
        # Bloom is a one-sided false-positive structure: δ = 0 for FN,
        # ε = empirical FPR.
        return (empirical_fpr, 0.0)

    def _capacity(self) -> int:
        return self._m

    def _extra(self) -> Mapping[str, Any]:
        set_bits = 0
        for b in self._bits:
            v = b
            while v:
                set_bits += v & 1
                v >>= 1
        return {
            "bits": self._m,
            "hash_functions": self._k,
            "set_bits": set_bits,
            "fill_ratio": set_bits / self._m if self._m else 0.0,
            "estimated_population": self._estimate(),
        }


# =============================================================================
# 10. Exponential Histogram — sliding window (Datar et al. 2002)
# =============================================================================


class ExponentialHistogram(_BaseSketch):
    """Datar-Gionis-Indyk-Motwani (2002) exponential histogram for
    1-bit (presence) counting over the last ``N`` *logical-time*
    steps.

    Maintains a list of *buckets*.  Each bucket records
    ``(timestamp_of_last_one, size)`` where ``size`` is a power of
    two.  When more than ``⌈1/ε⌉/2 + 1`` buckets of the same size
    exist, the two oldest are merged.  Estimated count =
    sum of all bucket sizes inside the window − size_of_oldest / 2.

    Guarantee: relative error ``≤ ε`` simultaneously over any
    sliding window inside the configured horizon.
    """

    KIND = KIND_EXP_HISTOGRAM

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._N = config.window_size
        self._eps = config.window_epsilon
        # k controls how many same-size buckets may exist.
        self._k = max(1, int(math.ceil(1.0 / self._eps)))
        # buckets: list of (timestamp, size) — oldest first.
        self._buckets: List[Tuple[int, int]] = []
        self._t = 0  # logical time

    def update(self, item: Any, weight: float = 1.0) -> None:
        # We track presence of 1-bits in the stream.  Any truthy item
        # contributes a 1 at the current logical time.
        self._t += 1
        self.n_items += 1
        present = bool(item)
        if present:
            self._buckets.append((self._t, 1))
            self._merge_buckets()
        # Drop buckets older than the window.
        cutoff = self._t - self._N
        while self._buckets and self._buckets[0][0] <= cutoff:
            self._buckets.pop(0)

    def _merge_buckets(self) -> None:
        # If we have too many buckets of the same size, merge the
        # two oldest.  Repeat across sizes in increasing order.
        size = 1
        limit = (self._k // 2) + 2
        while True:
            indices = [i for i, (_, s) in enumerate(self._buckets) if s == size]
            if len(indices) <= limit:
                return
            # Merge the two oldest of this size.
            i0, i1 = indices[0], indices[1]
            t1 = self._buckets[i1][0]
            new_bucket = (t1, size * 2)
            # Remove i0 first (i1 > i0 so its index doesn't shift).
            del self._buckets[i1]
            del self._buckets[i0]
            # Insert merged bucket at the correct timestamp position.
            inserted = False
            for j in range(len(self._buckets)):
                if self._buckets[j][0] > t1:
                    self._buckets.insert(j, new_bucket)
                    inserted = True
                    break
            if not inserted:
                self._buckets.append(new_bucket)
            size *= 2

    def count(self) -> float:
        # Sum bucket sizes; subtract oldest_size / 2 (rounded up to
        # integer when sizes are integer).
        if not self._buckets:
            return 0.0
        total = sum(s for _, s in self._buckets)
        oldest = self._buckets[0][1]
        return float(total - oldest / 2.0)

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"eh",
            self._N.to_bytes(8, "big"),
            struct.pack(">d", self._eps),
            self._t.to_bytes(8, "big"),
        ]
        for ts, sz in self._buckets:
            parts.append(ts.to_bytes(8, "big"))
            parts.append(sz.to_bytes(8, "big"))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        return self.count()

    def epsilon_delta(self) -> Tuple[float, float]:
        return (self._eps, 0.0)

    def _capacity(self) -> int:
        return len(self._buckets)

    def _extra(self) -> Mapping[str, Any]:
        return {
            "window_size": self._N,
            "buckets": len(self._buckets),
            "logical_time": self._t,
        }


# =============================================================================
# 11. F2 / AMS sketch — tug of war
# =============================================================================


class F2Sketch(_BaseSketch):
    """Alon-Matias-Szegedy (1996) F_2 tug-of-war estimator.

    For each row ``i`` keep a counter ``X_i = Σ_x sign_i(x) · f(x)``.
    Each row gives an unbiased estimate ``X_i² ≈ F_2``; medians of
    means tighten to ``(ε, δ)`` with ``rows × cols ≈ 1 / (ε² δ)``.
    """

    KIND = KIND_F2_SKETCH

    def __init__(self, config: SketcherConfig) -> None:
        super().__init__(config)
        self._rows = config.f2_rows
        self._cols = config.f2_cols
        # Single counter per row -- AMS in its simplest form.
        # ``f2_cols`` controls the "number of means before median"
        # — we pack cols independent estimators per row.
        self._counters: List[List[float]] = [
            [0.0] * self._cols for _ in range(self._rows)
        ]
        self._salts: List[List[int]] = [
            [
                (config.seed + 0xA15CA15CA15CA15C * (r * self._cols + c + 1))
                & ((1 << 64) - 1)
                for c in range(self._cols)
            ]
            for r in range(self._rows)
        ]

    def update(self, item: Any, weight: float = 1.0) -> None:
        self.n_items += 1
        payload = _canonical_item_bytes(item)
        for r in range(self._rows):
            for c in range(self._cols):
                h = _hash_bytes(self._salts[r][c], payload)
                sign = 1 if h & 1 == 0 else -1
                self._counters[r][c] += sign * float(weight)

    def f2_estimate(self) -> float:
        # Within each row, the mean of X_c² is an unbiased estimator
        # of F_2.  The median across rows is the canonical AMS
        # tightener.
        row_estimates = []
        for r in range(self._rows):
            means = sum(x * x for x in self._counters[r]) / self._cols
            row_estimates.append(means)
        row_estimates.sort()
        return row_estimates[len(row_estimates) // 2]

    def merge(self, other: "_BaseSketch") -> None:
        self._require_compatible(other)
        assert isinstance(other, F2Sketch)
        if other._rows != self._rows or other._cols != self._cols:
            raise IncompatibleSketch("f2 shape mismatch")
        if other._salts != self._salts:
            raise IncompatibleSketch("f2 salts differ")
        for r in range(self._rows):
            for c in range(self._cols):
                self._counters[r][c] += other._counters[r][c]
        self.n_items += other.n_items

    def _canonical_state_bytes(self) -> bytes:
        parts: List[bytes] = [
            b"f2",
            self._rows.to_bytes(4, "big"),
            self._cols.to_bytes(4, "big"),
        ]
        for row_salts in self._salts:
            for s in row_salts:
                parts.append(s.to_bytes(8, "big"))
        for row in self._counters:
            for v in row:
                parts.append(struct.pack(">d", v))
        return b"|".join(parts)

    def _estimate(self) -> Any:
        return self.f2_estimate()

    def epsilon_delta(self) -> Tuple[float, float]:
        # AMS tail: rows × cols ≈ 1/(ε² δ) gives (ε, δ).
        # We expose the asymptotic values from the actual shape.
        eps = math.sqrt(1.0 / max(self._cols, 1))
        delta = math.exp(-self._rows / 2.0)
        return (eps, delta)

    def _capacity(self) -> int:
        return self._rows * self._cols

    def _extra(self) -> Mapping[str, Any]:
        return {"rows": self._rows, "cols": self._cols}


# =============================================================================
# Sketcher façade — uniform constructor and dispatch
# =============================================================================


_SKETCH_CLASSES: Mapping[str, type] = {
    KIND_MISRA_GRIES: MisraGriesSketch,
    KIND_COUNT_MIN: CountMinSketch,
    KIND_COUNT_SKETCH: CountSketch,
    KIND_HLL: HyperLogLogSketch,
    KIND_KLL: KLLSketch,
    KIND_GK: GKSketch,
    KIND_RESERVOIR: ReservoirSampler,
    KIND_WEIGHTED_RESERVOIR: WeightedReservoirSampler,
    KIND_BLOOM: BloomFilter,
    KIND_EXP_HISTOGRAM: ExponentialHistogram,
    KIND_F2_SKETCH: F2Sketch,
}


class Sketcher:
    """Uniform façade over every sketch kind.

    A coordination engine constructs a Sketcher via one of the
    classmethod constructors (``Sketcher.count_min(...)``, etc.)
    or via the generic ``Sketcher(config)`` path, then drives
    ``update / query / report`` through a single, kind-agnostic
    interface.
    """

    def __init__(self, config: SketcherConfig) -> None:
        cls = _SKETCH_CLASSES.get(config.kind)
        if cls is None:
            raise InvalidConfig(f"unknown sketch kind {config.kind!r}")
        self._sketch: _BaseSketch = cls(config)
        self._created_at = time.time()

    # ---- properties ------------------------------------------------

    @property
    def kind(self) -> str:
        return self._sketch.KIND

    @property
    def config(self) -> SketcherConfig:
        return self._sketch.config

    @property
    def n_items(self) -> int:
        return self._sketch.n_items

    @property
    def mergeable(self) -> bool:
        return self._sketch.mergeable

    # ---- streaming -------------------------------------------------

    def update(self, item: Any, weight: float = 1.0) -> None:
        self._sketch.update(item, weight)

    def update_many(self, items: Iterable[Any]) -> None:
        self._sketch.update_many(items)

    # ---- queries (kind-specific, delegated when present) -----------

    def query(self, item: Any) -> float:
        if hasattr(self._sketch, "query"):
            return self._sketch.query(item)  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no point query")

    def quantile(self, q: float) -> float:
        if hasattr(self._sketch, "quantile"):
            return self._sketch.quantile(q)  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no quantile query")

    def rank(self, x: float) -> float:
        if hasattr(self._sketch, "rank"):
            return self._sketch.rank(x)  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no rank query")

    def cdf(self, x: float) -> float:
        if hasattr(self._sketch, "cdf"):
            return self._sketch.cdf(x)  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no cdf query")

    def cardinality(self) -> float:
        if hasattr(self._sketch, "cardinality"):
            return self._sketch.cardinality()  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no cardinality query")

    def heavy_hitters(self, threshold_fraction: float = 0.0) -> List[Tuple[Any, float]]:
        if hasattr(self._sketch, "heavy_hitters"):
            return self._sketch.heavy_hitters(threshold_fraction)  # type: ignore[attr-defined]
        raise InvalidQuery(
            f"sketch kind {self.kind!r} has no heavy_hitters query"
        )

    def sample(self) -> Any:
        if hasattr(self._sketch, "sample"):
            return self._sketch.sample()  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no sample query")

    def contains(self, item: Any) -> bool:
        if hasattr(self._sketch, "contains"):
            return self._sketch.contains(item)  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no contains query")

    def count(self) -> float:
        if hasattr(self._sketch, "count"):
            return self._sketch.count()  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no count query")

    def f2_estimate(self) -> float:
        if hasattr(self._sketch, "f2_estimate"):
            return self._sketch.f2_estimate()  # type: ignore[attr-defined]
        raise InvalidQuery(f"sketch kind {self.kind!r} has no f2 estimate")

    # ---- merge / report --------------------------------------------

    def merge(self, other: "Sketcher") -> None:
        if not isinstance(other, Sketcher):
            raise IncompatibleSketch("merge target must be a Sketcher")
        if not self._sketch.mergeable:
            raise NotMergeable(
                f"sketch kind {self.kind!r} is not mergeable"
            )
        self._sketch.merge(other._sketch)

    def report(self) -> SketcherReport:
        return self._sketch.report()

    def certificate(self) -> str:
        return self._sketch.certificate()

    def n_bytes(self) -> int:
        return self._sketch.n_bytes()

    # ---- typed constructors ----------------------------------------

    @classmethod
    def misra_gries(cls, k: int = 64, *, seed: int = 0) -> "Sketcher":
        return cls(SketcherConfig(kind=KIND_MISRA_GRIES, capacity=k, seed=seed))

    @classmethod
    def count_min(
        cls,
        *,
        epsilon: float = 1e-3,
        delta: float = 1e-3,
        seed: int = 0,
        conservative_update: bool = False,
    ) -> "Sketcher":
        return cls(
            SketcherConfig(
                kind=KIND_COUNT_MIN,
                epsilon=epsilon,
                delta=delta,
                seed=seed,
                conservative_update=conservative_update,
            )
        )

    @classmethod
    def count_sketch(
        cls,
        *,
        epsilon: float = 1e-2,
        delta: float = 1e-3,
        seed: int = 0,
    ) -> "Sketcher":
        return cls(
            SketcherConfig(
                kind=KIND_COUNT_SKETCH,
                epsilon=epsilon,
                delta=delta,
                seed=seed,
            )
        )

    @classmethod
    def hll(cls, precision: int = 12, *, seed: int = 0) -> "Sketcher":
        return cls(SketcherConfig(kind=KIND_HLL, precision=precision, seed=seed))

    @classmethod
    def kll(cls, k: int = 256, *, seed: int = 0) -> "Sketcher":
        return cls(SketcherConfig(kind=KIND_KLL, quantile_k=k, seed=seed))

    @classmethod
    def gk(cls, k: int = 256, *, seed: int = 0) -> "Sketcher":
        return cls(SketcherConfig(kind=KIND_GK, quantile_k=k, seed=seed))

    @classmethod
    def reservoir(cls, k: int = 256, *, seed: int = 0) -> "Sketcher":
        return cls(SketcherConfig(kind=KIND_RESERVOIR, capacity=k, seed=seed))

    @classmethod
    def weighted_reservoir(cls, k: int = 256, *, seed: int = 0) -> "Sketcher":
        return cls(SketcherConfig(kind=KIND_WEIGHTED_RESERVOIR, capacity=k, seed=seed))

    @classmethod
    def bloom(
        cls,
        *,
        capacity: int = 10_000,
        fpr: float = 1e-3,
        seed: int = 0,
    ) -> "Sketcher":
        return cls(
            SketcherConfig(
                kind=KIND_BLOOM,
                bloom_capacity=capacity,
                bloom_fpr=fpr,
                seed=seed,
            )
        )

    @classmethod
    def exp_histogram(
        cls,
        *,
        window: int = 1024,
        epsilon: float = 0.05,
        seed: int = 0,
    ) -> "Sketcher":
        return cls(
            SketcherConfig(
                kind=KIND_EXP_HISTOGRAM,
                window_size=window,
                window_epsilon=epsilon,
                seed=seed,
            )
        )

    @classmethod
    def f2_sketch(
        cls,
        *,
        rows: int = 5,
        cols: int = 32,
        seed: int = 0,
    ) -> "Sketcher":
        return cls(
            SketcherConfig(
                kind=KIND_F2_SKETCH,
                f2_rows=rows,
                f2_cols=cols,
                seed=seed,
            )
        )


# =============================================================================
# Convenience free functions — match the style of other primitives
# =============================================================================


def misra_gries(k: int = 64, *, seed: int = 0) -> Sketcher:
    """One-shot Misra-Gries sketch constructor."""
    return Sketcher.misra_gries(k, seed=seed)


def count_min(
    *,
    epsilon: float = 1e-3,
    delta: float = 1e-3,
    seed: int = 0,
    conservative_update: bool = False,
) -> Sketcher:
    """One-shot Count-Min sketch constructor."""
    return Sketcher.count_min(
        epsilon=epsilon,
        delta=delta,
        seed=seed,
        conservative_update=conservative_update,
    )


def count_sketch(
    *,
    epsilon: float = 1e-2,
    delta: float = 1e-3,
    seed: int = 0,
) -> Sketcher:
    """One-shot Count-Sketch constructor."""
    return Sketcher.count_sketch(epsilon=epsilon, delta=delta, seed=seed)


def hyperloglog(precision: int = 12, *, seed: int = 0) -> Sketcher:
    """One-shot HyperLogLog constructor (alias of hll)."""
    return Sketcher.hll(precision, seed=seed)


def kll(k: int = 256, *, seed: int = 0) -> Sketcher:
    return Sketcher.kll(k, seed=seed)


def gk(k: int = 256, *, seed: int = 0) -> Sketcher:
    return Sketcher.gk(k, seed=seed)


def reservoir(k: int = 256, *, seed: int = 0) -> Sketcher:
    return Sketcher.reservoir(k, seed=seed)


def weighted_reservoir(k: int = 256, *, seed: int = 0) -> Sketcher:
    return Sketcher.weighted_reservoir(k, seed=seed)


def bloom(*, capacity: int = 10_000, fpr: float = 1e-3, seed: int = 0) -> Sketcher:
    return Sketcher.bloom(capacity=capacity, fpr=fpr, seed=seed)


def exp_histogram(
    *,
    window: int = 1024,
    epsilon: float = 0.05,
    seed: int = 0,
) -> Sketcher:
    return Sketcher.exp_histogram(window=window, epsilon=epsilon, seed=seed)


def f2_sketch(*, rows: int = 5, cols: int = 32, seed: int = 0) -> Sketcher:
    return Sketcher.f2_sketch(rows=rows, cols=cols, seed=seed)


# =============================================================================
# Self-describing trusted-base summary
# =============================================================================


def known_kinds() -> Tuple[str, ...]:
    """Return the sketch kinds exposed by this Sketcher build."""
    return KNOWN_KINDS


def mergeable_kinds() -> Tuple[str, ...]:
    """Return the kinds that support a mergeable summary."""
    return tuple(sorted(MERGEABLE_KINDS))


def sketcher_summary() -> dict:
    """Self-describing summary for a coordination engine's preflight."""
    return {
        "known_kinds": list(KNOWN_KINDS),
        "mergeable_kinds": list(sorted(MERGEABLE_KINDS)),
        "n_kinds": len(KNOWN_KINDS),
        "pure_stdlib": True,
    }


__all__ = [
    # errors
    "SketcherError",
    "InvalidConfig",
    "InvalidUpdate",
    "InvalidQuery",
    "NotMergeable",
    "IncompatibleSketch",
    # kinds
    "KIND_MISRA_GRIES",
    "KIND_COUNT_MIN",
    "KIND_COUNT_SKETCH",
    "KIND_HLL",
    "KIND_KLL",
    "KIND_GK",
    "KIND_RESERVOIR",
    "KIND_WEIGHTED_RESERVOIR",
    "KIND_BLOOM",
    "KIND_EXP_HISTOGRAM",
    "KIND_F2_SKETCH",
    "KNOWN_KINDS",
    "MERGEABLE_KINDS",
    # config / report
    "SketcherConfig",
    "SketcherReport",
    # sketches
    "MisraGriesSketch",
    "CountMinSketch",
    "CountSketch",
    "HyperLogLogSketch",
    "KLLSketch",
    "GKSketch",
    "ReservoirSampler",
    "WeightedReservoirSampler",
    "BloomFilter",
    "ExponentialHistogram",
    "F2Sketch",
    # façade
    "Sketcher",
    # convenience constructors
    "misra_gries",
    "count_min",
    "count_sketch",
    "hyperloglog",
    "kll",
    "gk",
    "reservoir",
    "weighted_reservoir",
    "bloom",
    "exp_histogram",
    "f2_sketch",
    # introspection
    "known_kinds",
    "mergeable_kinds",
    "sketcher_summary",
]
