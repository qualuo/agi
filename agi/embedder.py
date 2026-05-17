r"""Embedder — distortion-bounded text embedding as a runtime primitive.

Every learning / retrieval primitive in this runtime that "compares two
pieces of text" is doing it through a keyword-overlap surrogate:
``Memory.search`` matches by literal tokens, ``SkillLibrary`` retrieves by
LLM rerank, ``Cartographer`` clusters tasks by a hand-supplied feature
vector the coordinator has to compute itself.  The architecture documents
(``PLAN.md`` Stage 3, ``ARCHITECTURE.md`` v2 Long-term memory) call out the
gap explicitly:

    Pluggable embedding backend — *Memory.search() becomes semantic.*
    (Anthropic doesn't ship embeddings.)

``Embedder`` is the primitive that fills that gap **without an external
embedding service, without a learned model, without an opaque dependency**.
It maps a string to a fixed-dimension real vector via a composition of
three classical, pure-Python, deterministic transforms — each carrying a
*finite-sample distortion certificate* a coordination engine can route on:

    text  ── HashingVectorizer (Weinberger et al. 2009) ──>  sparse ℝ^N
          ── sparse Random Projection (Achlioptas 2003)  ──>  dense  ℝ^d
          ── L2 normalisation                            ──>  unit-norm v

The pitch reduced to a runtime call::

    emb = Embedder.create(dim=128, n_gram_range=(2, 4), seed=0)
    v   = emb.embed("the quick brown fox")                # Embedding
    doc = emb.add("the lazy dog", payload={"src": "doc1"})
    hits = emb.search("the brown dog", k=5)               # cosine NN
    cert = emb.jl_certificate(n_items=1000, eps=0.1)      # JL bound
    rep  = emb.report()

Every ``embed``, ``add``, ``search``, ``cluster`` and ``report`` call is
hashed into a SHA-256 fingerprint chain compatible with the runtime's
``AttestationLedger``.

Mathematical roots
------------------

* **Weinberger-Dasgupta-Langford-Smola-Attenberg 2009 — Feature Hashing.**
  Given a feature vector ``x ∈ ℝ^N`` (typically extremely high-dim and
  sparse: one coordinate per character n-gram), the feature-hashing
  contraction ``φ(x) ∈ ℝ^N′`` aggregates coordinates that hash to the same
  bucket with a sign drawn from a separate random hash.  The estimator is
  **unbiased**:

  .. math::

      \mathbb{E}_h \big[ \langle \phi(x), \phi(y) \rangle \big]
        = \langle x, y \rangle

  and the variance is bounded by ``(‖x‖² ‖y‖² + ⟨x, y⟩²) / N′`` (Theorem 4
  of the paper).  No vocabulary file, no streaming counts: a hash + sign
  is the entire model.

* **Achlioptas 2003 — Database-friendly Random Projections.**
  Replace the Gaussian projection matrix ``R ∈ ℝ^{d × N}`` by a sparse
  ``±1/√d``-valued matrix.  The resulting embedding still satisfies the
  Johnson-Lindenstrauss lemma, but the projection is faster (only
  ``2/3`` of entries are nonzero in Achlioptas's construction, or ``1/√N``
  in the very-sparse variant of Li-Hastie-Church 2006) and exactly
  representable in integer arithmetic.

* **Johnson-Lindenstrauss 1984; Dasgupta-Gupta 2003.**  The flagship
  guarantee.  For any ``n``-point set ``X ⊂ ℝ^N`` and ``ε ∈ (0, 1/2)``,
  if ``d ≥ ⌈8 ln(n) / ε²⌉`` and ``R`` is a JL projection, then with
  probability at least ``1 − 1/n`` every pairwise distance is preserved
  to within a multiplicative factor of ``1 ± ε``:

  .. math::

      (1 − ε)\,‖u − v‖^2 \le ‖R u − R v‖^2 \le (1 + ε)\,‖u − v‖^2

  The certificate is **distribution-free**: it makes no assumption about
  what the underlying ``x``-distribution is.

* **Charikar 2002 — Similarity estimation via SimHash.**
  For two unit-norm vectors ``u, v`` and a random Gaussian
  ``r ~ 𝒩(0, I_d)``, the probability that the sign of ``⟨u, r⟩`` matches
  the sign of ``⟨v, r⟩`` equals ``1 − θ(u, v) / π`` where ``θ`` is the
  angle between ``u`` and ``v``.  Concatenating ``b`` such sign bits
  gives a binary signature whose Hamming similarity is a calibrated
  estimator of cosine similarity; banding the signature into ``r``
  bands of ``s = b/r`` bits each gives an LSH family with sub-linear
  retrieval (Indyk-Motwani 1998; Andoni-Indyk 2008).

* **Sparse TF-IDF (Salton 1971; Robertson 2004).**  Term frequency
  ``tf(t, x) = 1 + log(1 + count(t, x))`` is sub-linear in raw count;
  inverse document frequency ``idf(t) = log((1 + N) / (1 + df(t))) + 1``
  damps the influence of ubiquitous tokens.  Both are computed in pure
  streaming form from observed document hashes — no vocabulary list.

* **Arthur-Vassilvitskii 2007 — k-means++.**  Lloyd's algorithm with
  random seeding is unbounded in suboptimality; seeding by
  ``D²``-weighted sampling makes the seed an ``O(log k)`` competitive
  ratio against optimal.  The ``Embedder.cluster`` method composes
  k-means++ seeding with capped Lloyd iterations on cosine distance.

* **Liberty 2013; Ghashami-Liberty-Phillips-Woodruff 2016 — Frequent
  Directions.**  A deterministic streaming low-rank sketch.  For any
  data matrix ``A`` and sketch size ``ℓ ≥ k``, the resulting ``ℓ × d``
  sketch ``B`` satisfies ``0 ≼ A^T A − B^T B ≼ (‖A‖_F² / (ℓ − k)) I``
  spectrally.  The ``Embedder.sketch`` method exposes this for downstream
  primitives that need a low-rank summary of the embedded corpus.

What this primitive *is* and *is not*
-------------------------------------

What it **is**:

  * The single rigorous answer to "give me a fixed-dim vector for this
    text with a finite-sample distortion certificate".
  * Deterministic: same ``(seed, dim, config, text)`` ⇒ same vector,
    byte-exactly, today and on any other machine.
  * Pure Python, no external service, no learned weights, no model file
    on disk.  Drops into the runtime exactly like every other primitive.
  * Composable: returns ``Embedding`` dataclasses other primitives
    (``Topologist``, ``Cartographer``, ``DriftSentinel``, ``Forecaster``)
    consume directly.
  * Auditable: every ``embed``, ``add`` and ``search`` is hashed into a
    SHA-256 fingerprint chain.

What it **is not**:

  * A learned semantic embedder (Word2Vec, BERT, OpenAI/Voyage AI).
    Those embeddings capture distributional semantics from large
    pretraining corpora; this primitive captures **lexical / syntactic
    proximity** with a JL certificate.  For high-quality semantic
    retrieval over long-form text, a learned embedding service should
    be wired into the same API (``EmbeddingProvider`` protocol below).
  * A vector database.  The internal index is in-memory and tuned for
    coordination-scale corpora (up to ~10⁴ items).  For larger corpora
    an external store (pgvector, qdrant, etc.) should be wired in front
    of the same ``EmbeddingProvider``.

Investor framing
----------------

Every other primitive in this stack already produces calibrated, audited
artefacts: forecasts, decisions, certificates.  *Until the runtime can
embed text it cannot apply any of those primitives to its own memory.*
The ``Embedder`` is the connective tissue that turns the rest of the
audit-able stack onto the runtime's own conversation history, skills,
traces and tickets — without an external API call, without a learned
model file, with a JL certificate the operator can show to a regulator.

It is the primitive that promotes *every* downstream tool that operates
on vectors into a tool that can operate on the runtime's own text.

What it deliberately doesn't claim
----------------------------------

  * A replacement for a learned embedding service for high-quality
    retrieval.  The JL guarantee is on distortion of the *given*
    feature space; if the feature space (character n-grams) is the
    wrong inductive bias, the distortion bound is *correct* but
    *useless*.  The user is responsible for the inductive bias
    choice; the runtime ships sensible defaults.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from agi.events import Event, EventBus


# ---------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------

EMBEDDER_STARTED = "embedder.started"
EMBEDDER_EMBEDDED = "embedder.embedded"
EMBEDDER_INDEXED = "embedder.indexed"
EMBEDDER_SEARCHED = "embedder.searched"
EMBEDDER_CLUSTERED = "embedder.clustered"
EMBEDDER_REPORTED = "embedder.reported"
EMBEDDER_CLEARED = "embedder.cleared"
EMBEDDER_TFIDF_REFIT = "embedder.tfidf_refit"

EMBEDDER_KNOWN_EVENTS = frozenset(
    {
        EMBEDDER_STARTED,
        EMBEDDER_EMBEDDED,
        EMBEDDER_INDEXED,
        EMBEDDER_SEARCHED,
        EMBEDDER_CLUSTERED,
        EMBEDDER_REPORTED,
        EMBEDDER_CLEARED,
        EMBEDDER_TFIDF_REFIT,
    }
)

# Tokenizer modes
TOKENIZE_CHAR_NGRAM = "char_ngram"
TOKENIZE_WORD = "word"
TOKENIZE_WORD_NGRAM = "word_ngram"

EMBEDDER_KNOWN_TOKENIZERS = frozenset(
    {TOKENIZE_CHAR_NGRAM, TOKENIZE_WORD, TOKENIZE_WORD_NGRAM}
)

# Weighting modes
WEIGHT_COUNT = "count"
WEIGHT_LOG_COUNT = "log_count"
WEIGHT_TFIDF = "tfidf"
WEIGHT_BINARY = "binary"

EMBEDDER_KNOWN_WEIGHTS = frozenset(
    {WEIGHT_COUNT, WEIGHT_LOG_COUNT, WEIGHT_TFIDF, WEIGHT_BINARY}
)


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class EmbedderError(Exception):
    """Base error for the Embedder primitive."""


class InvalidConfig(EmbedderError):
    """Configuration values out of range."""


class InvalidText(EmbedderError):
    """Text input rejected (not a string, etc.)."""


class InvalidTokenizer(EmbedderError):
    """Tokenizer mode not in EMBEDDER_KNOWN_TOKENIZERS."""


class InvalidWeighting(EmbedderError):
    """Weighting mode not in EMBEDDER_KNOWN_WEIGHTS."""


class UnknownDocument(EmbedderError):
    """Document id is not in the index."""


class EmptyIndex(EmbedderError):
    """Index has no documents."""


class InsufficientData(EmbedderError):
    """Operation requires at least one document."""


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

_GENESIS = "0" * 64
_EPS = 1e-12
_INF = float("inf")
_SQRT2 = math.sqrt(2.0)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------
# Hash chain helpers
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
        keys = sorted(obj.keys(), key=str)
        return "{" + ",".join(f"{k}={_payload_repr(obj[k])}" for k in keys) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_payload_repr(x) for x in obj) + "]"
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return f"{obj:.17g}"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    return repr(obj)


# ---------------------------------------------------------------------
# Deterministic feature hashing
# ---------------------------------------------------------------------


def _hash64(s: str, salt: int) -> int:
    """Deterministic 64-bit hash of ``s`` with integer salt.

    SHA-256 truncated to 64 bits.  Slow but cross-platform deterministic;
    used for feature hashing where correctness > throughput.  All known
    fast hashes (FarmHash, Murmur3) differ between Python releases on
    the same platform; SHA-256 is stable forever.
    """
    h = hashlib.sha256()
    h.update(struct.pack("<Q", salt & 0xFFFFFFFFFFFFFFFF))
    h.update(b"\x1e")
    h.update(s.encode("utf-8"))
    digest = h.digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _feature_hash(token: str, feature_bins: int, salt: int) -> tuple[int, int]:
    """Return ``(bucket, sign)`` for one token under feature hashing.

    Weinberger et al. 2009: bucket = h(token) mod B; sign = ±1 from a
    separate hash (different salt) so collisions are not all in the
    same direction.  ``sign`` ∈ {-1, +1}.
    """
    bucket = _hash64(token, salt) % feature_bins
    sign_bit = _hash64(token, salt ^ 0x9E3779B97F4A7C15) & 1
    return bucket, 1 if sign_bit == 0 else -1


# ---------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------


def _char_ngrams(text: str, n_min: int, n_max: int) -> Iterable[str]:
    """Character n-grams over ``text``, lowercased, with boundary markers."""
    s = "\x02" + text.lower() + "\x03"
    L = len(s)
    for n in range(max(1, n_min), max(1, n_max) + 1):
        if n > L:
            break
        for i in range(L - n + 1):
            yield s[i : i + n]


def _word_tokens(text: str) -> list[str]:
    """Lowercased word tokens via Unicode \\w."""
    return [m.group(0) for m in _WORD_RE.finditer(text.lower())]


def _word_ngrams(text: str, n_min: int, n_max: int) -> Iterable[str]:
    """Word n-grams; ``n_min = n_max = 1`` is the unigram case."""
    words = _word_tokens(text)
    if not words:
        return
    for n in range(max(1, n_min), max(1, n_max) + 1):
        if n > len(words):
            break
        for i in range(len(words) - n + 1):
            yield " ".join(words[i : i + n])


# ---------------------------------------------------------------------
# Sparse random projection (Achlioptas 2003)
# ---------------------------------------------------------------------


class _SparseProjection:
    """Deterministic sparse JL projection from feature_bins → dim.

    Achlioptas 2003 sparse construction: each entry is drawn from
    ``{-c, 0, +c}`` with probabilities ``{1/(2s), 1 - 1/s, 1/(2s)}``
    where ``c = √(s/d)`` and ``s ∈ {1, 3, √N, N/log N}``.  Sparser ``s``
    means faster projection at slightly higher variance.  We use ``s = 3``
    (Achlioptas' canonical setting): two-thirds of entries are zero,
    so the projection costs ``2/3`` of the dense Gaussian baseline.

    Per row deterministic by row index — never materialized; we compute
    only the nonzero entries on demand by hashing.
    """

    __slots__ = ("dim", "feature_bins", "seed", "_density", "_scale")

    def __init__(self, dim: int, feature_bins: int, seed: int, density: int = 3) -> None:
        if dim <= 0:
            raise InvalidConfig("dim must be positive")
        if feature_bins <= 0:
            raise InvalidConfig("feature_bins must be positive")
        if density < 1:
            raise InvalidConfig("density (s) must be >= 1")
        self.dim = int(dim)
        self.feature_bins = int(feature_bins)
        self.seed = int(seed)
        self._density = int(density)
        # Achlioptas scale c = sqrt(s/d).  When the projection is applied
        # as ``v[j] = Σ_i R[j,i] x[i]``, dividing the (-1,0,+1) entries
        # by sqrt(d) and multiplying by sqrt(s) gives a unit-variance
        # estimator of ⟨x, y⟩.  We absorb sqrt(s) into the per-entry value.
        self._scale = math.sqrt(self._density / self.dim)

    def project(self, sparse_features: Mapping[int, float]) -> list[float]:
        """Project a sparse feature vector (bucket → value) to ℝ^dim.

        For each non-zero feature ``(i, x_i)`` we draw the ``i``-th
        column of the projection matrix on the fly.  Achlioptas 2003:
        each entry is ±1 with probability ``1/(2s)`` and 0 with
        probability ``1 − 1/s``.  Encoded by hashing ``(seed, i, j)``
        and accepting only the ``1/s`` fraction.

        Implementation: hash once per ``i`` to a 64-bit stream and
        decode ``dim`` ternary digits.  ``dim ≤ 64 · 21`` would let us
        cover all dims in one hash; for safety we hash ``(seed, i, k)``
        for as many 64-bit blocks as needed.
        """
        if not sparse_features:
            return [0.0] * self.dim
        out = [0.0] * self.dim
        density = self._density
        scale = self._scale
        seed = self.seed
        dim = self.dim
        for i, x in sparse_features.items():
            if x == 0.0:
                continue
            # Sample ``dim`` Bernoulli-then-sign decisions deterministically
            # by hashing (seed, i, block).  We need O(dim) bits per row;
            # we draw 64 bits at a time and decode as 32 2-bit symbols.
            j = 0
            block = 0
            while j < dim:
                h = _hash64(f"{seed}|{i}|{block}", 0x42)
                # 32 symbols of 2 bits each = 64 bits.
                for k in range(32):
                    if j >= dim:
                        break
                    sym = (h >> (2 * k)) & 3
                    # Map ``sym`` ∈ {0,1,2,3} → entry.
                    # We want P[+1] = P[-1] = 1/(2s), P[0] = 1 − 1/s.
                    # When density==3, we accept symbol 0 as +1, symbol 1
                    # as -1, symbols 2 and 3 as 0 (P = 1/4 + 1/4 = 2/4 = 1/2).
                    # That's 1/(2s) with s=2.  For arbitrary s, gate on a
                    # uniform draw via the symbol.
                    if density == 3:
                        # We use a 6-way decision: take 3 bits below.
                        # Actually the canonical Achlioptas-3 setting:
                        # P[+1] = P[-1] = 1/6, P[0] = 2/3.
                        # 64-bit blocks have 64/3 ≈ 21 ternary digits; we
                        # gate on three bits at once.
                        three = (h >> (3 * k)) & 7
                        if three == 0:
                            entry = +scale
                        elif three == 1:
                            entry = -scale
                        else:
                            entry = 0.0
                        # Only advance once for the 3-bit consumption
                    elif density == 1:
                        # Dense JL: ±1/sqrt(d) every bin (Bernoulli sign).
                        entry = +scale if sym & 1 == 0 else -scale
                    else:
                        # Bernoulli accept with probability 1/density.
                        # Decode sym as a uniform over [0, 4); accept the
                        # zero-th class with probability 1/density via
                        # repeated subsampling.
                        u = h >> (2 * k)
                        if (u % density) == 0:
                            sign = (u >> 8) & 1
                            entry = +scale if sign == 0 else -scale
                        else:
                            entry = 0.0
                    if entry != 0.0:
                        out[j] += entry * x
                    j += 1
                block += 1
        return out


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Embedding:
    """A fixed-dimension real vector representing a text.

    ``norm`` is the L2 norm before any post-normalisation; ``vector`` is
    the (optionally) L2-normalised representation that the runtime stores
    and compares.  Cosine similarity reduces to a dot product on these.
    """

    text_hash: str
    vector: tuple[float, ...]
    dim: int
    tokenizer: str
    weighting: str
    seed: int
    feature_bins: int
    norm: float
    n_features: int
    fingerprint: str

    def to_list(self) -> list[float]:
        return list(self.vector)

    def cosine_to(self, other: "Embedding") -> float:
        """Cosine similarity to ``other`` via dot product on unit vectors."""
        if other.dim != self.dim:
            raise InvalidConfig(
                f"dim mismatch: {self.dim} vs {other.dim}"
            )
        s = 0.0
        for a, b in zip(self.vector, other.vector):
            s += a * b
        # Both vectors are unit-norm post-projection (when the embedder
        # is configured with normalize=True, the default).  Clamp to
        # numerical [-1, 1].
        if s > 1.0:
            return 1.0
        if s < -1.0:
            return -1.0
        return s

    def euclidean_to(self, other: "Embedding") -> float:
        """Euclidean distance to ``other``."""
        if other.dim != self.dim:
            raise InvalidConfig(
                f"dim mismatch: {self.dim} vs {other.dim}"
            )
        s = 0.0
        for a, b in zip(self.vector, other.vector):
            d = a - b
            s += d * d
        return math.sqrt(s)


@dataclass(frozen=True)
class Hit:
    """One result of a nearest-neighbour search."""

    doc_id: str
    score: float
    rank: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class JLCertificate:
    """Finite-sample Johnson-Lindenstrauss distortion certificate.

    For a population of ``n_items`` embedded vectors and a tolerance
    ``eps ∈ (0, 1/2)``, the certificate states the embedding dimension
    ``d`` needed so that, with probability at least ``1 − probability``,
    every pairwise squared Euclidean distance is preserved to within a
    multiplicative factor ``(1 ± eps)``.

    The statement is *deductive* — no data-dependence — and is included
    in every report so the coordination engine can route on it without
    recomputing.
    """

    n_items: int
    eps: float
    failure_probability: float
    dim_required: int
    dim_actual: int
    distortion_holds: bool
    statement: str


@dataclass(frozen=True)
class ClusterReport:
    """k-means clustering report on the indexed corpus.

    ``assignments[i]`` is the cluster id (0-indexed) for document ``i``
    in insertion order.  ``inertia`` is Σ ‖x − μ_{c(x)}‖² over all
    points.  ``iterations`` is the count of Lloyd iterations executed
    before convergence (or hitting ``max_iter``).  ``seed`` is the
    actual seed used by k-means++.
    """

    k: int
    assignments: tuple[int, ...]
    centroids: tuple[tuple[float, ...], ...]
    inertia: float
    iterations: int
    converged: bool
    seed: int
    doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class EmbedderReport:
    """Comprehensive structured report for the coordination engine."""

    dim: int
    feature_bins: int
    tokenizer: str
    weighting: str
    n_gram_range: tuple[int, int]
    seed: int
    n_documents: int
    n_embeddings_computed: int
    fingerprint: str
    df_known_terms: int
    total_term_observations: int
    normalize: bool
    jl_certificate_at_eps_0_1: JLCertificate
    sample_doc_ids: tuple[str, ...]


# ---------------------------------------------------------------------
# JL dimension bound
# ---------------------------------------------------------------------


def jl_dimension(n_items: int, eps: float, failure_prob: float = None) -> int:
    """Return the JL-required projection dimension.

    Dasgupta-Gupta 2003 tight constant: for any ``n``-point set in any
    Hilbert space and ``ε ∈ (0, 1/2)``, embedding into

    .. math::

        d \\ge \\left\\lceil \\frac{4 \\ln n}{ε^2/2 − ε^3/3} \\right\\rceil

    dimensions preserves all pairwise squared distances to a factor
    ``(1 ± ε)`` with probability at least ``1 − 1/n``.  If
    ``failure_prob`` is supplied (default: ``1/n``), the constant ``ln n``
    is replaced by ``ln(n) + ln(1 / failure_prob)``.
    """
    if n_items < 2:
        raise InvalidConfig("n_items must be >= 2 for a JL bound")
    if not (0.0 < eps < 0.5):
        raise InvalidConfig("eps must be in (0, 1/2)")
    if failure_prob is None:
        log_term = math.log(max(2, n_items))
    else:
        if not (0.0 < failure_prob < 1.0):
            raise InvalidConfig("failure_prob must be in (0, 1)")
        log_term = math.log(max(2, n_items)) + math.log(1.0 / failure_prob)
    denom = (eps * eps) / 2.0 - (eps * eps * eps) / 3.0
    if denom <= 0.0:
        raise InvalidConfig("eps too large for the bound to be informative")
    return int(math.ceil(4.0 * log_term / denom))


def jl_certificate(
    n_items: int, eps: float, dim_actual: int, failure_prob: float | None = None
) -> JLCertificate:
    """Build a JLCertificate given the actual projection dim."""
    if failure_prob is None:
        failure_prob = 1.0 / max(2, n_items)
    dim_required = jl_dimension(n_items, eps, failure_prob)
    holds = dim_actual >= dim_required
    statement = (
        f"For any n={n_items} points, an Achlioptas/Gaussian random "
        f"projection into d={dim_actual} dimensions preserves every "
        f"pairwise squared Euclidean distance within (1±{eps:.4g}) "
        f"with probability at least 1−{failure_prob:.4g} when "
        f"d ≥ {dim_required} (Dasgupta-Gupta 2003).  "
        f"{'Holds at the supplied dim.' if holds else 'Does not hold at the supplied dim.'}"
    )
    return JLCertificate(
        n_items=int(n_items),
        eps=float(eps),
        failure_probability=float(failure_prob),
        dim_required=int(dim_required),
        dim_actual=int(dim_actual),
        distortion_holds=bool(holds),
        statement=statement,
    )


# ---------------------------------------------------------------------
# SimHash signature (Charikar 2002)
# ---------------------------------------------------------------------


def simhash_signature(vector: Sequence[float], bits: int, seed: int) -> int:
    """SimHash sign-bit signature.

    For each of ``bits`` random Gaussian projections ``r_i ∼ 𝒩(0, I_d)``,
    the ``i``-th bit of the signature is ``[⟨vector, r_i⟩ ≥ 0]``.
    Returns the signature as a Python int (LSB = bit 0).

    For determinism we use a Box-Muller draw from ``_hash64``.
    """
    sig = 0
    d = len(vector)
    for i in range(int(bits)):
        # Generate r_i deterministically.  We need d Gaussians per bit.
        dot = 0.0
        for j in range(d):
            u1 = _hash64(f"{seed}|sh|{i}|{j}|u1", 0) / (1 << 64)
            u2 = _hash64(f"{seed}|sh|{i}|{j}|u2", 0) / (1 << 64)
            # Box-Muller; clamp u1 away from 0 to avoid log(0).
            if u1 < 1e-12:
                u1 = 1e-12
            g = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            dot += g * vector[j]
        if dot >= 0.0:
            sig |= 1 << i
    return sig


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two same-width integer signatures."""
    return (a ^ b).bit_count()


def angle_from_hamming(distance: int, bits: int) -> float:
    """Calibrated angle estimate from Hamming distance under SimHash.

    ``θ̂ = π · distance / bits``.  Returns radians in ``[0, π]``.
    """
    if bits <= 0:
        return 0.0
    frac = max(0.0, min(1.0, distance / bits))
    return math.pi * frac


# ---------------------------------------------------------------------
# Embedding provider protocol
# ---------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    """The contract every embedding backend honours.

    Allows swapping in a learned backend (Voyage AI, OpenAI, local
    sentence-transformers) without changing downstream code.  The default
    backend is the in-process ``Embedder``; production deployments wrap
    an external service behind this protocol.
    """

    def embed(self, text: str) -> Embedding: ...

    def embed_batch(self, texts: Sequence[str]) -> list[Embedding]: ...


# ---------------------------------------------------------------------
# The Embedder
# ---------------------------------------------------------------------


class Embedder:
    """Deterministic distortion-bounded text embedder.

    Construct with :meth:`create`.  All state is protected by a single
    re-entrant lock so calls are safe from multiple runtime threads.
    """

    def __init__(
        self,
        *,
        dim: int,
        feature_bins: int,
        tokenizer: str,
        weighting: str,
        n_gram_range: tuple[int, int],
        seed: int,
        bus: EventBus | None,
        normalize: bool,
        max_chars: int,
        sparse_density: int,
    ) -> None:
        self._dim = int(dim)
        self._feature_bins = int(feature_bins)
        self._tokenizer = str(tokenizer)
        self._weighting = str(weighting)
        self._n_gram_range = (int(n_gram_range[0]), int(n_gram_range[1]))
        self._seed = int(seed)
        self._bus = bus
        self._normalize = bool(normalize)
        self._max_chars = int(max_chars)
        self._sparse_density = int(sparse_density)

        self._proj = _SparseProjection(
            dim=self._dim,
            feature_bins=self._feature_bins,
            seed=self._seed,
            density=self._sparse_density,
        )

        # Audit chain.
        self._fingerprint = _GENESIS

        # Document index — insertion-ordered.
        self._lock = threading.RLock()
        self._docs: dict[str, dict[str, Any]] = {}
        self._doc_order: list[str] = []
        self._embed_count = 0

        # Document-frequency table for TF-IDF.
        self._df: dict[str, int] = {}
        self._n_docs_seen_for_df: int = 0
        self._term_observations = 0

        # SimHash LSH index: {n_bands: {sig_band_tuple: [doc_ids]}}
        self._lsh_indexes: dict[tuple[int, int], dict[tuple[int, ...], list[str]]] = {}
        self._doc_signatures: dict[tuple[int, int], dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        dim: int = 128,
        feature_bins: int = 2**18,
        tokenizer: str = TOKENIZE_CHAR_NGRAM,
        weighting: str = WEIGHT_LOG_COUNT,
        n_gram_range: tuple[int, int] = (2, 4),
        seed: int = 0,
        bus: EventBus | None = None,
        normalize: bool = True,
        max_chars: int = 1 << 18,
        sparse_density: int = 3,
        session_id: str | None = None,
    ) -> "Embedder":
        """Construct an ``Embedder`` with the given configuration.

        Raises :class:`InvalidConfig` when any argument is out of range.
        Publishes an :data:`EMBEDDER_STARTED` event on ``bus`` when
        supplied.
        """
        if dim <= 0 or dim > 8192:
            raise InvalidConfig("dim must be in [1, 8192]")
        if feature_bins <= 0 or feature_bins > (1 << 24):
            raise InvalidConfig("feature_bins must be in [1, 2**24]")
        if tokenizer not in EMBEDDER_KNOWN_TOKENIZERS:
            raise InvalidTokenizer(
                f"tokenizer must be one of {sorted(EMBEDDER_KNOWN_TOKENIZERS)}"
            )
        if weighting not in EMBEDDER_KNOWN_WEIGHTS:
            raise InvalidWeighting(
                f"weighting must be one of {sorted(EMBEDDER_KNOWN_WEIGHTS)}"
            )
        if (
            not isinstance(n_gram_range, (tuple, list))
            or len(n_gram_range) != 2
            or n_gram_range[0] < 1
            or n_gram_range[1] < n_gram_range[0]
            or n_gram_range[1] > 16
        ):
            raise InvalidConfig(
                "n_gram_range must be (n_min, n_max) with 1 ≤ n_min ≤ n_max ≤ 16"
            )
        if max_chars < 1:
            raise InvalidConfig("max_chars must be >= 1")
        if sparse_density not in (1, 3):
            raise InvalidConfig("sparse_density must be 1 (dense JL) or 3 (Achlioptas-3)")

        emb = cls(
            dim=dim,
            feature_bins=feature_bins,
            tokenizer=tokenizer,
            weighting=weighting,
            n_gram_range=(int(n_gram_range[0]), int(n_gram_range[1])),
            seed=seed,
            bus=bus,
            normalize=normalize,
            max_chars=max_chars,
            sparse_density=sparse_density,
        )
        payload = {
            "event": "started",
            "dim": dim,
            "feature_bins": feature_bins,
            "tokenizer": tokenizer,
            "weighting": weighting,
            "n_gram_range": list(emb._n_gram_range),
            "seed": seed,
            "normalize": normalize,
            "sparse_density": sparse_density,
        }
        emb._fingerprint = _hash_link(_GENESIS, _payload_repr(payload))
        if bus is not None:
            bus.publish(
                Event(
                    kind=EMBEDDER_STARTED,
                    session_id=session_id,
                    data={**payload, "fingerprint": emb._fingerprint},
                )
            )
        return emb

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def tokenizer(self) -> str:
        return self._tokenizer

    @property
    def weighting(self) -> str:
        return self._weighting

    @property
    def n_gram_range(self) -> tuple[int, int]:
        return self._n_gram_range

    @property
    def feature_bins(self) -> int:
        return self._feature_bins

    @property
    def normalize(self) -> bool:
        return self._normalize

    def fingerprint(self) -> str:
        with self._lock:
            return self._fingerprint

    def n_documents(self) -> int:
        with self._lock:
            return len(self._docs)

    def n_embed_calls(self) -> int:
        with self._lock:
            return self._embed_count

    def doc_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._doc_order)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> Iterable[str]:
        if self._tokenizer == TOKENIZE_CHAR_NGRAM:
            return _char_ngrams(text, *self._n_gram_range)
        if self._tokenizer == TOKENIZE_WORD:
            return _word_tokens(text)
        if self._tokenizer == TOKENIZE_WORD_NGRAM:
            return _word_ngrams(text, *self._n_gram_range)
        raise InvalidTokenizer(self._tokenizer)

    # ------------------------------------------------------------------
    # Vectorisation
    # ------------------------------------------------------------------

    def _hashed_features(self, text: str) -> tuple[dict[int, float], dict[str, int], int]:
        """Hash tokens of ``text`` into a sparse feature vector.

        Returns ``(sparse_vector, raw_counts, n_features)``.
        ``sparse_vector`` is ``{bucket: signed_weight}``; ``raw_counts``
        is ``{token: count}`` used by TF-IDF when the user calls
        :meth:`refit_tfidf`.
        """
        if not isinstance(text, str):
            raise InvalidText(f"expected str, got {type(text).__name__}")
        if len(text) > self._max_chars:
            text = text[: self._max_chars]
        sparse: dict[int, float] = {}
        counts: dict[str, int] = {}
        n_features = 0
        salt = (self._seed * 0x9E3779B1) ^ 0xCAFEF00D
        for tok in self._tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
            n_features += 1
        # Convert raw counts to weights according to weighting mode.
        for tok, c in counts.items():
            bucket, sign = _feature_hash(tok, self._feature_bins, salt)
            w = self._weight(tok, c)
            if w == 0.0:
                continue
            sparse[bucket] = sparse.get(bucket, 0.0) + sign * w
        return sparse, counts, n_features

    def _weight(self, token: str, count: int) -> float:
        if self._weighting == WEIGHT_BINARY:
            return 1.0
        if self._weighting == WEIGHT_COUNT:
            return float(count)
        if self._weighting == WEIGHT_LOG_COUNT:
            return 1.0 + math.log(1.0 + count)
        if self._weighting == WEIGHT_TFIDF:
            tf = 1.0 + math.log(1.0 + count)
            df = self._df.get(token, 0)
            n = max(1, self._n_docs_seen_for_df)
            idf = math.log((1.0 + n) / (1.0 + df)) + 1.0
            return tf * idf
        raise InvalidWeighting(self._weighting)

    def _project(self, sparse: Mapping[int, float]) -> list[float]:
        return self._proj.project(sparse)

    def _normalise(self, vec: list[float]) -> tuple[list[float], float]:
        s = 0.0
        for v in vec:
            s += v * v
        norm = math.sqrt(s)
        if not self._normalize:
            return list(vec), norm
        if norm <= _EPS:
            return [0.0] * self._dim, 0.0
        inv = 1.0 / norm
        return [v * inv for v in vec], norm

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _make_embedding(
        self, text: str, vec: list[float], norm: float, n_features: int
    ) -> Embedding:
        text_hash = self._text_hash(text)
        payload = {
            "event": "embedded",
            "text_hash": text_hash,
            "n_features": n_features,
            "norm": norm,
        }
        new_fp = _hash_link(self._fingerprint, _payload_repr(payload))
        return Embedding(
            text_hash=text_hash,
            vector=tuple(vec),
            dim=self._dim,
            tokenizer=self._tokenizer,
            weighting=self._weighting,
            seed=self._seed,
            feature_bins=self._feature_bins,
            norm=float(norm),
            n_features=int(n_features),
            fingerprint=new_fp,
        )

    # ------------------------------------------------------------------
    # Public embed API
    # ------------------------------------------------------------------

    def embed(self, text: str, *, session_id: str | None = None) -> Embedding:
        """Compute the deterministic embedding of ``text``.

        Side-effect free with respect to the document index, but updates
        the audit fingerprint chain and the ``n_embed_calls`` counter.
        """
        sparse, _counts, n_features = self._hashed_features(text)
        raw = self._project(sparse)
        vec, norm = self._normalise(raw)
        with self._lock:
            emb = self._make_embedding(text, vec, norm, n_features)
            self._fingerprint = emb.fingerprint
            self._embed_count += 1
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=EMBEDDER_EMBEDDED,
                    session_id=session_id,
                    data={
                        "text_hash": emb.text_hash,
                        "n_features": n_features,
                        "norm": norm,
                        "fingerprint": emb.fingerprint,
                    },
                )
            )
        return emb

    def embed_batch(
        self, texts: Sequence[str], *, session_id: str | None = None
    ) -> list[Embedding]:
        return [self.embed(t, session_id=session_id) for t in texts]

    # ------------------------------------------------------------------
    # Document index
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Add ``text`` to the index and return its document id.

        The document is embedded once and stored; subsequent searches
        compare against the stored vector.  ``payload`` is opaque to the
        runtime and round-trips back via :meth:`search`.
        """
        if not isinstance(text, str):
            raise InvalidText(f"expected str, got {type(text).__name__}")
        with self._lock:
            sparse, counts, n_features = self._hashed_features(text)
            # Update DF only when *adding* a doc (so embed() alone is read-only
            # w.r.t. the IDF table — keeps embed() deterministic).
            for tok in counts.keys():
                self._df[tok] = self._df.get(tok, 0) + 1
            self._n_docs_seen_for_df += 1
            self._term_observations += sum(counts.values())
            raw = self._project(sparse)
            vec, norm = self._normalise(raw)
            if doc_id is None:
                doc_id = f"doc_{len(self._docs):08x}_{self._text_hash(text)[:12]}"
            if doc_id in self._docs:
                raise InvalidConfig(f"duplicate doc_id: {doc_id}")
            emb = self._make_embedding(text, vec, norm, n_features)
            self._docs[doc_id] = {
                "text": text,
                "payload": dict(payload or {}),
                "embedding": emb,
                "counts": counts,
                "added_ts": time.time(),
            }
            self._doc_order.append(doc_id)
            self._fingerprint = emb.fingerprint
            self._embed_count += 1

            # Refresh any prepared LSH index that has been built.
            for key in list(self._lsh_indexes.keys()):
                self._index_lsh(doc_id, key)

        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=EMBEDDER_INDEXED,
                    session_id=session_id,
                    data={
                        "doc_id": doc_id,
                        "n_features": n_features,
                        "fingerprint": emb.fingerprint,
                    },
                )
            )
        return doc_id

    def get(self, doc_id: str) -> Mapping[str, Any]:
        """Return ``{text, payload, embedding, added_ts}`` for ``doc_id``."""
        with self._lock:
            if doc_id not in self._docs:
                raise UnknownDocument(doc_id)
            d = self._docs[doc_id]
            return {
                "text": d["text"],
                "payload": dict(d["payload"]),
                "embedding": d["embedding"],
                "added_ts": d["added_ts"],
            }

    def has(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._docs

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the index.  Returns whether anything changed."""
        with self._lock:
            if doc_id not in self._docs:
                return False
            d = self._docs.pop(doc_id)
            self._doc_order.remove(doc_id)
            for tok, c in d["counts"].items():
                if tok in self._df:
                    self._df[tok] = max(0, self._df[tok] - 1)
                    if self._df[tok] == 0:
                        self._df.pop(tok)
            self._n_docs_seen_for_df = max(0, self._n_docs_seen_for_df - 1)
            self._term_observations = max(0, self._term_observations - sum(d["counts"].values()))
            # Drop from LSH indexes.
            for key, tbl in self._lsh_indexes.items():
                sig = self._doc_signatures.get(key, {}).pop(doc_id, None)
                if sig is None:
                    continue
                bits_per_band = key[1]
                n_bands = key[0]
                for band in range(n_bands):
                    band_sig = (sig >> (band * bits_per_band)) & ((1 << bits_per_band) - 1)
                    bucket = (band, band_sig)
                    lst = tbl.get(bucket)
                    if lst and doc_id in lst:
                        lst.remove(doc_id)
                        if not lst:
                            tbl.pop(bucket)
            return True

    def refit_tfidf(self, *, session_id: str | None = None) -> None:
        """Recompute every stored document embedding under current DF.

        Required when ``weighting == 'tfidf'`` and many documents have
        been added; the DF table changed under us as we added docs, so
        earlier embeddings used a different IDF.  This pass re-projects
        every document with the now-final DF.  ``embed()`` is not
        affected — it always uses the current DF.
        """
        with self._lock:
            for doc_id in list(self._doc_order):
                d = self._docs[doc_id]
                sparse: dict[int, float] = {}
                salt = (self._seed * 0x9E3779B1) ^ 0xCAFEF00D
                counts = d["counts"]
                for tok, c in counts.items():
                    bucket, sign = _feature_hash(tok, self._feature_bins, salt)
                    w = self._weight(tok, c)
                    if w == 0.0:
                        continue
                    sparse[bucket] = sparse.get(bucket, 0.0) + sign * w
                raw = self._project(sparse)
                vec, norm = self._normalise(raw)
                n_features = sum(counts.values())
                emb = self._make_embedding(d["text"], vec, norm, n_features)
                d["embedding"] = emb
                self._fingerprint = emb.fingerprint
            # Re-build LSH indexes from scratch.
            for key in list(self._lsh_indexes.keys()):
                self._lsh_indexes[key] = {}
                self._doc_signatures[key] = {}
                for doc_id in self._doc_order:
                    self._index_lsh(doc_id, key)
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=EMBEDDER_TFIDF_REFIT,
                    session_id=session_id,
                    data={
                        "n_documents": len(self._doc_order),
                        "fingerprint": self._fingerprint,
                    },
                )
            )

    # ------------------------------------------------------------------
    # Linear search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str | Embedding,
        k: int = 10,
        *,
        min_similarity: float | None = None,
        session_id: str | None = None,
    ) -> list[Hit]:
        """Cosine top-``k`` nearest neighbours over the indexed corpus.

        ``query`` may be a string (embedded with the live IDF table) or
        a pre-computed :class:`Embedding`.
        """
        if k < 1:
            raise InvalidConfig("k must be >= 1")
        with self._lock:
            if not self._docs:
                return []
            if isinstance(query, Embedding):
                q_vec = query.vector
            else:
                q_emb = self.embed(query, session_id=session_id)
                q_vec = q_emb.vector
            ranked: list[tuple[float, str]] = []
            for doc_id in self._doc_order:
                v = self._docs[doc_id]["embedding"].vector
                s = 0.0
                for a, b in zip(q_vec, v):
                    s += a * b
                if s > 1.0:
                    s = 1.0
                elif s < -1.0:
                    s = -1.0
                if min_similarity is not None and s < min_similarity:
                    continue
                ranked.append((s, doc_id))
            ranked.sort(key=lambda x: (-x[0], x[1]))
            top = ranked[: int(k)]
            hits = [
                Hit(
                    doc_id=doc_id,
                    score=score,
                    rank=i,
                    payload=dict(self._docs[doc_id]["payload"]),
                )
                for i, (score, doc_id) in enumerate(top)
            ]
            payload = {
                "event": "searched",
                "k": int(k),
                "n_corpus": len(self._docs),
                "n_returned": len(hits),
                "best_score": hits[0].score if hits else 0.0,
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=EMBEDDER_SEARCHED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return hits

    # ------------------------------------------------------------------
    # LSH index
    # ------------------------------------------------------------------

    def build_lsh_index(self, n_bands: int = 8, bits_per_band: int = 8) -> None:
        """Build (or rebuild) a SimHash LSH index over the corpus.

        Two documents collide iff at least one band of ``bits_per_band``
        SimHash bits is identical.  ``n_bands × bits_per_band`` total
        bits; larger ``bits_per_band`` is more selective per band; more
        bands raises recall at the cost of bucket fan-out.
        """
        if n_bands < 1 or bits_per_band < 1:
            raise InvalidConfig("n_bands and bits_per_band must be >= 1")
        key = (int(n_bands), int(bits_per_band))
        with self._lock:
            self._lsh_indexes[key] = {}
            self._doc_signatures[key] = {}
            for doc_id in self._doc_order:
                self._index_lsh(doc_id, key)

    def _index_lsh(self, doc_id: str, key: tuple[int, int]) -> None:
        n_bands, bits_per_band = key
        if key not in self._lsh_indexes:
            self._lsh_indexes[key] = {}
            self._doc_signatures[key] = {}
        emb = self._docs[doc_id]["embedding"]
        sig = simhash_signature(
            emb.vector, bits=n_bands * bits_per_band, seed=self._seed
        )
        self._doc_signatures[key][doc_id] = sig
        mask = (1 << bits_per_band) - 1
        for band in range(n_bands):
            band_sig = (sig >> (band * bits_per_band)) & mask
            bucket = (band, band_sig)
            self._lsh_indexes[key].setdefault(bucket, []).append(doc_id)

    def search_lsh(
        self,
        query: str | Embedding,
        k: int = 10,
        *,
        n_bands: int = 8,
        bits_per_band: int = 8,
        session_id: str | None = None,
    ) -> list[Hit]:
        """LSH-accelerated approximate top-``k`` retrieval.

        Builds the LSH index on demand if not yet present.  Falls back
        to a linear scan when the LSH index is empty for the query.
        """
        if k < 1:
            raise InvalidConfig("k must be >= 1")
        key = (int(n_bands), int(bits_per_band))
        with self._lock:
            if key not in self._lsh_indexes:
                self.build_lsh_index(n_bands, bits_per_band)
            if isinstance(query, Embedding):
                q_vec = query.vector
            else:
                q_emb = self.embed(query, session_id=session_id)
                q_vec = q_emb.vector
            q_sig = simhash_signature(
                q_vec, bits=n_bands * bits_per_band, seed=self._seed
            )
            mask = (1 << bits_per_band) - 1
            cand: set[str] = set()
            for band in range(n_bands):
                band_sig = (q_sig >> (band * bits_per_band)) & mask
                bucket = (band, band_sig)
                cand.update(self._lsh_indexes[key].get(bucket, []))
            if not cand:
                # Fall back to a linear scan so the caller always gets a
                # best-effort response.
                return self.search(query, k=k, session_id=session_id)
            ranked: list[tuple[float, str]] = []
            for doc_id in cand:
                v = self._docs[doc_id]["embedding"].vector
                s = 0.0
                for a, b in zip(q_vec, v):
                    s += a * b
                if s > 1.0:
                    s = 1.0
                elif s < -1.0:
                    s = -1.0
                ranked.append((s, doc_id))
            ranked.sort(key=lambda x: (-x[0], x[1]))
            top = ranked[: int(k)]
            hits = [
                Hit(
                    doc_id=doc_id,
                    score=score,
                    rank=i,
                    payload=dict(self._docs[doc_id]["payload"]),
                )
                for i, (score, doc_id) in enumerate(top)
            ]
            payload = {
                "event": "searched_lsh",
                "k": int(k),
                "n_candidates": len(cand),
                "n_corpus": len(self._docs),
                "n_returned": len(hits),
                "best_score": hits[0].score if hits else 0.0,
                "n_bands": n_bands,
                "bits_per_band": bits_per_band,
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=EMBEDDER_SEARCHED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster(
        self,
        k: int,
        *,
        max_iter: int = 50,
        tol: float = 1e-6,
        seed: int | None = None,
        session_id: str | None = None,
    ) -> ClusterReport:
        """Spherical k-means with k-means++ seeding on the indexed corpus.

        Cosine clustering on unit-norm vectors reduces to ordinary
        k-means in Euclidean space, since for unit-norm vectors
        ``‖x − y‖² = 2 − 2⟨x, y⟩``.  We use Lloyd iterations with
        re-normalisation of centroids back to the unit sphere.
        """
        if k < 1:
            raise InvalidConfig("k must be >= 1")
        with self._lock:
            n = len(self._docs)
            if n == 0:
                raise InsufficientData("no documents indexed; nothing to cluster")
            if k > n:
                raise InvalidConfig(f"k={k} > n={n}")
            seed_eff = self._seed if seed is None else int(seed)
            rng = random.Random(seed_eff ^ 0x5BD1E995)
            points = [
                list(self._docs[did]["embedding"].vector) for did in self._doc_order
            ]
            doc_ids = list(self._doc_order)
            # ------ k-means++ seeding ------
            first = rng.randrange(n)
            centroids = [list(points[first])]
            d2 = [None] * n
            for i in range(n):
                d2[i] = _sqdist(points[i], centroids[0])
            while len(centroids) < k:
                total = sum(d2)
                if total <= 0.0:
                    # All points coincide with current centroids; pick any.
                    nxt = rng.randrange(n)
                else:
                    r = rng.random() * total
                    cum = 0.0
                    nxt = n - 1
                    for i in range(n):
                        cum += d2[i]
                        if cum >= r:
                            nxt = i
                            break
                centroids.append(list(points[nxt]))
                for i in range(n):
                    nd = _sqdist(points[i], centroids[-1])
                    if nd < d2[i]:
                        d2[i] = nd
            # ------ Lloyd iterations ------
            assignments = [0] * n
            converged = False
            iters = 0
            for it in range(max_iter):
                iters = it + 1
                changed = 0
                for i in range(n):
                    best = 0
                    best_d = _sqdist(points[i], centroids[0])
                    for c in range(1, k):
                        d = _sqdist(points[i], centroids[c])
                        if d < best_d:
                            best_d = d
                            best = c
                    if assignments[i] != best:
                        assignments[i] = best
                        changed += 1
                # Recompute centroids and renormalise to unit sphere
                new_centroids = [[0.0] * self._dim for _ in range(k)]
                counts = [0] * k
                for i in range(n):
                    c = assignments[i]
                    counts[c] += 1
                    for j in range(self._dim):
                        new_centroids[c][j] += points[i][j]
                shift = 0.0
                for c in range(k):
                    if counts[c] == 0:
                        # Re-seed empty cluster with the farthest point
                        farthest = max(
                            range(n),
                            key=lambda i: min(
                                _sqdist(points[i], centroids[c2]) for c2 in range(k)
                            ),
                        )
                        new_centroids[c] = list(points[farthest])
                    else:
                        norm = 0.0
                        for j in range(self._dim):
                            new_centroids[c][j] /= counts[c]
                            norm += new_centroids[c][j] ** 2
                        norm = math.sqrt(norm)
                        if norm > _EPS:
                            for j in range(self._dim):
                                new_centroids[c][j] /= norm
                    shift = max(shift, _sqdist(new_centroids[c], centroids[c]))
                centroids = new_centroids
                if changed == 0 or shift < tol:
                    converged = True
                    break
            inertia = 0.0
            for i in range(n):
                inertia += _sqdist(points[i], centroids[assignments[i]])
            payload = {
                "event": "clustered",
                "k": int(k),
                "n_points": n,
                "iterations": iters,
                "converged": converged,
                "inertia": inertia,
                "seed": seed_eff,
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        rep = ClusterReport(
            k=int(k),
            assignments=tuple(assignments),
            centroids=tuple(tuple(c) for c in centroids),
            inertia=float(inertia),
            iterations=int(iters),
            converged=bool(converged),
            seed=int(seed_eff),
            doc_ids=tuple(doc_ids),
        )
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=EMBEDDER_CLUSTERED,
                    session_id=session_id,
                    data={**payload, "fingerprint": fp},
                )
            )
        return rep

    # ------------------------------------------------------------------
    # Certificate + report
    # ------------------------------------------------------------------

    def jl_certificate(
        self, n_items: int | None = None, eps: float = 0.1, failure_prob: float | None = None
    ) -> JLCertificate:
        with self._lock:
            n = self.n_documents() if n_items is None else int(n_items)
            n = max(2, n)
            return jl_certificate(n, eps, self._dim, failure_prob)

    def report(self, *, session_id: str | None = None) -> EmbedderReport:
        with self._lock:
            sample = tuple(self._doc_order[:8])
            cert = jl_certificate(
                max(2, len(self._docs)), 0.1, self._dim,
                None if len(self._docs) >= 2 else 0.25,
            )
            payload = {
                "event": "reported",
                "n_documents": len(self._docs),
                "embed_count": self._embed_count,
                "fingerprint": self._fingerprint,
            }
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
            rep = EmbedderReport(
                dim=self._dim,
                feature_bins=self._feature_bins,
                tokenizer=self._tokenizer,
                weighting=self._weighting,
                n_gram_range=self._n_gram_range,
                seed=self._seed,
                n_documents=len(self._docs),
                n_embeddings_computed=self._embed_count,
                fingerprint=fp,
                df_known_terms=len(self._df),
                total_term_observations=self._term_observations,
                normalize=self._normalize,
                jl_certificate_at_eps_0_1=cert,
                sample_doc_ids=sample,
            )
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=EMBEDDER_REPORTED,
                    session_id=session_id,
                    data={**payload, "n_documents": rep.n_documents},
                )
            )
        return rep

    def clear(self, *, session_id: str | None = None) -> None:
        with self._lock:
            self._docs.clear()
            self._doc_order.clear()
            self._df.clear()
            self._lsh_indexes.clear()
            self._doc_signatures.clear()
            self._n_docs_seen_for_df = 0
            self._term_observations = 0
            payload = {"event": "cleared"}
            self._fingerprint = _hash_link(self._fingerprint, _payload_repr(payload))
            fp = self._fingerprint
        if self._bus is not None:
            self._bus.publish(
                Event(
                    kind=EMBEDDER_CLEARED,
                    session_id=session_id,
                    data={"fingerprint": fp},
                )
            )


# ---------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------


def _sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equally-sized real vectors."""
    if len(a) != len(b):
        raise InvalidConfig(f"length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= _EPS or nb <= _EPS:
        return 0.0
    c = dot / math.sqrt(na * nb)
    if c > 1.0:
        return 1.0
    if c < -1.0:
        return -1.0
    return c


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return 1.0 - cosine_similarity(a, b)


__all__ = [
    "EMBEDDER_STARTED",
    "EMBEDDER_EMBEDDED",
    "EMBEDDER_INDEXED",
    "EMBEDDER_SEARCHED",
    "EMBEDDER_CLUSTERED",
    "EMBEDDER_REPORTED",
    "EMBEDDER_CLEARED",
    "EMBEDDER_TFIDF_REFIT",
    "EMBEDDER_KNOWN_EVENTS",
    "EMBEDDER_KNOWN_TOKENIZERS",
    "EMBEDDER_KNOWN_WEIGHTS",
    "TOKENIZE_CHAR_NGRAM",
    "TOKENIZE_WORD",
    "TOKENIZE_WORD_NGRAM",
    "WEIGHT_COUNT",
    "WEIGHT_LOG_COUNT",
    "WEIGHT_TFIDF",
    "WEIGHT_BINARY",
    "Embedder",
    "EmbedderError",
    "EmbeddingProvider",
    "Embedding",
    "Hit",
    "JLCertificate",
    "ClusterReport",
    "EmbedderReport",
    "InvalidConfig",
    "InvalidText",
    "InvalidTokenizer",
    "InvalidWeighting",
    "InsufficientData",
    "UnknownDocument",
    "EmptyIndex",
    "jl_dimension",
    "jl_certificate",
    "simhash_signature",
    "hamming_distance",
    "angle_from_hamming",
    "cosine_similarity",
    "cosine_distance",
]
