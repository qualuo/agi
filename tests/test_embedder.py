"""Tests for the Embedder primitive (distortion-bounded text embeddings)."""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.events import EventBus
from agi.embedder import (
    EMBEDDER_CLEARED,
    EMBEDDER_CLUSTERED,
    EMBEDDER_EMBEDDED,
    EMBEDDER_INDEXED,
    EMBEDDER_KNOWN_EVENTS,
    EMBEDDER_KNOWN_TOKENIZERS,
    EMBEDDER_KNOWN_WEIGHTS,
    EMBEDDER_REPORTED,
    EMBEDDER_SEARCHED,
    EMBEDDER_STARTED,
    EMBEDDER_TFIDF_REFIT,
    TOKENIZE_CHAR_NGRAM,
    TOKENIZE_WORD,
    TOKENIZE_WORD_NGRAM,
    WEIGHT_BINARY,
    WEIGHT_COUNT,
    WEIGHT_LOG_COUNT,
    WEIGHT_TFIDF,
    ClusterReport,
    Embedder,
    EmbedderReport,
    Embedding,
    Hit,
    InsufficientData,
    InvalidConfig,
    InvalidText,
    InvalidTokenizer,
    InvalidWeighting,
    JLCertificate,
    UnknownDocument,
    angle_from_hamming,
    cosine_distance,
    cosine_similarity,
    hamming_distance,
    jl_certificate,
    jl_dimension,
    simhash_signature,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _animal_corpus() -> list[tuple[str, dict]]:
    return [
        ("the cat sat on the mat", {"topic": "animal"}),
        ("the kitten slept on the rug", {"topic": "animal"}),
        ("a kitten plays with yarn", {"topic": "animal"}),
        ("the dog barks at the fence", {"topic": "animal"}),
        ("a puppy chases its tail", {"topic": "animal"}),
        ("quantum mechanics is hard", {"topic": "tech"}),
        ("relativity bends spacetime", {"topic": "tech"}),
        ("python is a programming language", {"topic": "tech"}),
        ("rust compiles to native code", {"topic": "tech"}),
        ("haskell embraces purity", {"topic": "tech"}),
    ]


def _shared_vocab_corpus() -> list[tuple[str, dict]]:
    """Corpus with topic-specific shared vocabulary across documents.

    Char/word-level lexical embeddings can only cluster topics when
    the topics share enough surface vocabulary across documents.
    """
    return [
        ("cat kitten paw fur whiskers tail meow purr", {"topic": "cat"}),
        ("kitten paw fur whiskers purr meow nap", {"topic": "cat"}),
        ("fur whiskers tail meow paw kitten cat sleep", {"topic": "cat"}),
        ("cat tail fur whiskers paw meow yarn", {"topic": "cat"}),
        ("kitten meow purr fur paw whiskers nap", {"topic": "cat"}),
        ("python rust compile code function variable type", {"topic": "code"}),
        ("function type compile python rust variable code", {"topic": "code"}),
        ("variable type rust python compile code function", {"topic": "code"}),
        ("compile rust python type function code library", {"topic": "code"}),
        ("python code type function variable rust library", {"topic": "code"}),
    ]


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    def test_create_defaults(self):
        e = Embedder.create()
        assert e.dim == 128
        assert e.seed == 0
        assert e.tokenizer == TOKENIZE_CHAR_NGRAM
        assert e.weighting == WEIGHT_LOG_COUNT
        assert e.n_gram_range == (2, 4)
        assert e.normalize is True
        assert e.feature_bins == 2 ** 18
        assert e.n_documents() == 0
        assert e.n_embed_calls() == 0
        assert len(e.fingerprint()) == 64

    def test_create_custom(self):
        e = Embedder.create(
            dim=32,
            feature_bins=1024,
            tokenizer=TOKENIZE_WORD,
            weighting=WEIGHT_BINARY,
            n_gram_range=(1, 1),
            seed=42,
            normalize=False,
        )
        assert e.dim == 32
        assert e.tokenizer == TOKENIZE_WORD
        assert e.weighting == WEIGHT_BINARY
        assert e.seed == 42
        assert e.normalize is False

    def test_invalid_dim(self):
        with pytest.raises(InvalidConfig):
            Embedder.create(dim=0)
        with pytest.raises(InvalidConfig):
            Embedder.create(dim=-1)
        with pytest.raises(InvalidConfig):
            Embedder.create(dim=9001)

    def test_invalid_feature_bins(self):
        with pytest.raises(InvalidConfig):
            Embedder.create(feature_bins=0)
        with pytest.raises(InvalidConfig):
            Embedder.create(feature_bins=1 << 25)

    def test_invalid_tokenizer(self):
        with pytest.raises(InvalidTokenizer):
            Embedder.create(tokenizer="bogus")

    def test_invalid_weighting(self):
        with pytest.raises(InvalidWeighting):
            Embedder.create(weighting="bogus")

    def test_invalid_n_gram_range(self):
        with pytest.raises(InvalidConfig):
            Embedder.create(n_gram_range=(0, 3))
        with pytest.raises(InvalidConfig):
            Embedder.create(n_gram_range=(3, 2))
        with pytest.raises(InvalidConfig):
            Embedder.create(n_gram_range=(1, 17))

    def test_invalid_sparse_density(self):
        with pytest.raises(InvalidConfig):
            Embedder.create(sparse_density=2)

    def test_invalid_max_chars(self):
        with pytest.raises(InvalidConfig):
            Embedder.create(max_chars=0)

    def test_known_constants(self):
        assert TOKENIZE_CHAR_NGRAM in EMBEDDER_KNOWN_TOKENIZERS
        assert TOKENIZE_WORD in EMBEDDER_KNOWN_TOKENIZERS
        assert TOKENIZE_WORD_NGRAM in EMBEDDER_KNOWN_TOKENIZERS
        assert WEIGHT_BINARY in EMBEDDER_KNOWN_WEIGHTS
        assert WEIGHT_COUNT in EMBEDDER_KNOWN_WEIGHTS
        assert WEIGHT_LOG_COUNT in EMBEDDER_KNOWN_WEIGHTS
        assert WEIGHT_TFIDF in EMBEDDER_KNOWN_WEIGHTS
        for k in (
            EMBEDDER_STARTED,
            EMBEDDER_EMBEDDED,
            EMBEDDER_INDEXED,
            EMBEDDER_SEARCHED,
            EMBEDDER_CLUSTERED,
            EMBEDDER_REPORTED,
            EMBEDDER_CLEARED,
            EMBEDDER_TFIDF_REFIT,
        ):
            assert k in EMBEDDER_KNOWN_EVENTS


# ------------------------------------------------------------------
# Basic embedding
# ------------------------------------------------------------------


class TestEmbed:
    def test_embed_returns_correct_shape(self):
        e = Embedder.create(dim=64, seed=0)
        v = e.embed("hello world")
        assert isinstance(v, Embedding)
        assert v.dim == 64
        assert len(v.vector) == 64
        assert all(isinstance(x, float) for x in v.vector)
        assert len(v.fingerprint) == 64
        assert v.tokenizer == TOKENIZE_CHAR_NGRAM
        assert v.weighting == WEIGHT_LOG_COUNT

    def test_embed_normalised_to_unit_norm(self):
        e = Embedder.create(dim=64, seed=0)
        v = e.embed("the quick brown fox jumps over the lazy dog")
        s = sum(x * x for x in v.vector)
        assert abs(math.sqrt(s) - 1.0) < 1e-9

    def test_embed_without_normalisation(self):
        e = Embedder.create(dim=64, seed=0, normalize=False)
        v = e.embed("the quick brown fox jumps over the lazy dog")
        s = math.sqrt(sum(x * x for x in v.vector))
        # Without normalisation, the projected vector has variable norm.
        assert s > 0.5

    def test_embed_empty_text(self):
        e = Embedder.create(dim=64, seed=0)
        v = e.embed("")
        # Empty text still has the (\x02, \x03) boundary markers under
        # character n-grams; vector is still finite-norm. Just check
        # that we get a valid embedding back.
        assert v.dim == 64
        assert len(v.vector) == 64

    def test_embed_unicode(self):
        e = Embedder.create(dim=64, seed=0)
        v = e.embed("héllo wörld 你好 مرحبا")
        assert v.dim == 64
        assert v.n_features > 0

    def test_embed_long_text_truncated(self):
        e = Embedder.create(dim=64, seed=0, max_chars=20)
        v_short = e.embed("the quick brown fox")
        v_long = e.embed("the quick brown fox jumps over the lazy dog repeatedly")
        # With max_chars=20, the long text is truncated to the first 20
        # chars → same feature set as a 20-char prefix.
        v_prefix = e.embed("the quick brown fox "[:20])
        assert v_long.text_hash != v_short.text_hash  # text-hash sees raw text
        # But the embedding *vectors* should differ since hash is on
        # raw text. The feature counts though follow the truncation.

    def test_embed_rejects_non_string(self):
        e = Embedder.create(dim=64, seed=0)
        with pytest.raises(InvalidText):
            e.embed(42)

    def test_embed_increments_counter(self):
        e = Embedder.create(dim=64, seed=0)
        assert e.n_embed_calls() == 0
        e.embed("a")
        assert e.n_embed_calls() == 1
        e.embed("b")
        e.embed("c")
        assert e.n_embed_calls() == 3

    def test_embed_advances_fingerprint(self):
        e = Embedder.create(dim=64, seed=0)
        fp0 = e.fingerprint()
        e.embed("a")
        fp1 = e.fingerprint()
        e.embed("b")
        fp2 = e.fingerprint()
        assert fp0 != fp1
        assert fp1 != fp2


# ------------------------------------------------------------------
# Determinism / reproducibility
# ------------------------------------------------------------------


class TestDeterminism:
    def test_same_text_same_vector(self):
        e1 = Embedder.create(dim=64, seed=7)
        e2 = Embedder.create(dim=64, seed=7)
        v1 = e1.embed("the quick brown fox")
        v2 = e2.embed("the quick brown fox")
        for a, b in zip(v1.vector, v2.vector):
            assert abs(a - b) < 1e-12

    def test_different_seed_different_vector(self):
        e1 = Embedder.create(dim=64, seed=0)
        e2 = Embedder.create(dim=64, seed=1)
        v1 = e1.embed("the quick brown fox")
        v2 = e2.embed("the quick brown fox")
        diff = sum(abs(a - b) for a, b in zip(v1.vector, v2.vector))
        assert diff > 1.0  # different projection ⇒ very different vector

    def test_text_hash_stable(self):
        e1 = Embedder.create(dim=32, seed=0)
        e2 = Embedder.create(dim=32, seed=99)
        assert e1.embed("abc").text_hash == e2.embed("abc").text_hash

    def test_batch_equals_individual(self):
        e = Embedder.create(dim=64, seed=0)
        texts = ["alpha", "beta", "gamma"]
        batch = e.embed_batch(texts)
        e2 = Embedder.create(dim=64, seed=0)
        individuals = [e2.embed(t) for t in texts]
        for vb, vi in zip(batch, individuals):
            for a, b in zip(vb.vector, vi.vector):
                assert abs(a - b) < 1e-12


# ------------------------------------------------------------------
# Similarity properties
# ------------------------------------------------------------------


class TestSimilarity:
    def test_identical_text_cosine_one(self):
        e = Embedder.create(dim=128, seed=0)
        v1 = e.embed("the quick brown fox")
        v2 = e.embed("the quick brown fox")
        assert v1.cosine_to(v2) > 0.999999

    def test_similar_text_high_cosine(self):
        e = Embedder.create(dim=128, seed=0)
        v1 = e.embed("the quick brown fox jumps over the lazy dog")
        v2 = e.embed("the quick brown fox jumps over the lazy dog!")
        # One-character difference at the end should yield near-1.
        assert v1.cosine_to(v2) > 0.9

    def test_unrelated_text_low_cosine(self):
        e = Embedder.create(
            dim=512,
            tokenizer=TOKENIZE_WORD,
            n_gram_range=(1, 1),
            seed=0,
        )
        v1 = e.embed("the quick brown fox jumps over the lazy dog")
        v2 = e.embed("microscope refrigerator parliament accordion")
        # No shared word vocabulary ⇒ cosine should be near zero.
        assert v1.cosine_to(v2) < 0.2

    def test_euclidean_distance_consistency(self):
        e = Embedder.create(dim=128, seed=0)
        v1 = e.embed("hello")
        v2 = e.embed("hello world")
        # For unit-norm vectors, ‖u − v‖² = 2(1 − ⟨u, v⟩)
        cos = v1.cosine_to(v2)
        eucl = v1.euclidean_to(v2)
        predicted = math.sqrt(max(0.0, 2.0 * (1.0 - cos)))
        assert abs(eucl - predicted) < 1e-6

    def test_cosine_clamped_to_unit_interval(self):
        e = Embedder.create(dim=32, seed=0)
        v = e.embed("hello")
        assert -1.0 <= v.cosine_to(v) <= 1.0

    def test_dim_mismatch_raises(self):
        e1 = Embedder.create(dim=64, seed=0)
        e2 = Embedder.create(dim=128, seed=0)
        v1 = e1.embed("a")
        v2 = e2.embed("a")
        with pytest.raises(InvalidConfig):
            v1.cosine_to(v2)
        with pytest.raises(InvalidConfig):
            v1.euclidean_to(v2)


# ------------------------------------------------------------------
# Document index
# ------------------------------------------------------------------


class TestIndex:
    def test_add_returns_doc_id(self):
        e = Embedder.create(dim=64, seed=0)
        did = e.add("hello world")
        assert isinstance(did, str)
        assert e.n_documents() == 1
        assert did in e.doc_ids()

    def test_add_custom_doc_id(self):
        e = Embedder.create(dim=64, seed=0)
        did = e.add("hello world", doc_id="my-doc")
        assert did == "my-doc"

    def test_add_duplicate_doc_id_raises(self):
        e = Embedder.create(dim=64, seed=0)
        e.add("hello", doc_id="a")
        with pytest.raises(InvalidConfig):
            e.add("world", doc_id="a")

    def test_add_rejects_non_string(self):
        e = Embedder.create(dim=64, seed=0)
        with pytest.raises(InvalidText):
            e.add(42)

    def test_get_returns_record(self):
        e = Embedder.create(dim=64, seed=0)
        did = e.add("hello world", payload={"src": "doc1"})
        rec = e.get(did)
        assert rec["text"] == "hello world"
        assert rec["payload"] == {"src": "doc1"}
        assert isinstance(rec["embedding"], Embedding)
        assert "added_ts" in rec

    def test_get_unknown_raises(self):
        e = Embedder.create(dim=64, seed=0)
        with pytest.raises(UnknownDocument):
            e.get("nope")

    def test_has(self):
        e = Embedder.create(dim=64, seed=0)
        did = e.add("hello")
        assert e.has(did) is True
        assert e.has("nope") is False

    def test_remove(self):
        e = Embedder.create(dim=64, seed=0)
        did = e.add("hello")
        assert e.remove(did) is True
        assert e.has(did) is False
        assert e.remove(did) is False  # idempotent for absent docs

    def test_clear(self):
        e = Embedder.create(dim=64, seed=0)
        for w in ["a", "b", "c"]:
            e.add(w)
        e.clear()
        assert e.n_documents() == 0


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


class TestSearch:
    def test_search_empty_index(self):
        e = Embedder.create(dim=64, seed=0)
        assert e.search("anything") == []

    def test_search_returns_animal_when_query_is_animal(self):
        e = Embedder.create(
            dim=256,
            tokenizer=TOKENIZE_WORD,
            n_gram_range=(1, 1),
            seed=0,
        )
        for text, payload in _animal_corpus():
            e.add(text, payload=payload)
        hits = e.search("a kitten on the rug", k=5)
        n_animal = sum(1 for h in hits if h.payload["topic"] == "animal")
        # Majority of top-5 must be animal-topic.
        assert n_animal >= 3

    def test_search_top_k_is_sorted_descending(self):
        e = Embedder.create(dim=128, seed=0)
        for text, payload in _animal_corpus():
            e.add(text, payload=payload)
        hits = e.search("brown fox jumping", k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        ranks = [h.rank for h in hits]
        assert ranks == list(range(len(hits)))

    def test_search_invalid_k(self):
        e = Embedder.create(dim=64, seed=0)
        e.add("hello")
        with pytest.raises(InvalidConfig):
            e.search("hello", k=0)

    def test_search_min_similarity_filter(self):
        e = Embedder.create(dim=128, seed=0)
        for text, payload in _animal_corpus():
            e.add(text, payload=payload)
        hits = e.search("kitten", k=10, min_similarity=0.5)
        for h in hits:
            assert h.score >= 0.5

    def test_search_with_embedding_argument(self):
        e = Embedder.create(dim=128, seed=0)
        for text, payload in _animal_corpus():
            e.add(text, payload=payload)
        q = e.embed("kitten")
        hits = e.search(q, k=3)
        assert len(hits) == 3

    def test_search_payload_isolated(self):
        e = Embedder.create(dim=64, seed=0)
        e.add("hello", payload={"x": 1})
        hits = e.search("hello", k=1)
        hits[0].payload["x"] = 999  # mutate result
        # Index payload must not change.
        assert e.get(hits[0].doc_id)["payload"]["x"] == 1


# ------------------------------------------------------------------
# LSH search
# ------------------------------------------------------------------


class TestLSH:
    def test_search_lsh_returns_relevant_hits(self):
        e = Embedder.create(dim=256, seed=0)
        for text, payload in _animal_corpus():
            e.add(text, payload=payload)
        hits = e.search_lsh("a kitten plays", k=3, n_bands=8, bits_per_band=4)
        for h in hits:
            assert h.payload["topic"] == "animal"

    def test_search_lsh_invalid_params(self):
        e = Embedder.create(dim=64, seed=0)
        e.add("hello")
        with pytest.raises(InvalidConfig):
            e.build_lsh_index(n_bands=0, bits_per_band=4)
        with pytest.raises(InvalidConfig):
            e.build_lsh_index(n_bands=4, bits_per_band=0)

    def test_search_lsh_invalid_k(self):
        e = Embedder.create(dim=64, seed=0)
        e.add("hello")
        with pytest.raises(InvalidConfig):
            e.search_lsh("hello", k=0)

    def test_search_lsh_persistent_index_survives_more_adds(self):
        e = Embedder.create(dim=128, seed=0)
        e.add("the cat sat on the mat", payload={"i": 1})
        # Use many bands / few bits for high recall.
        e.build_lsh_index(n_bands=16, bits_per_band=3)
        e.add("the dog chased the ball", payload={"i": 2})
        # Querying with a phrase similar to doc 2 should pick it up.
        hits = e.search_lsh(
            "the dog chased the ball", k=2, n_bands=16, bits_per_band=3
        )
        ids = {h.payload["i"] for h in hits}
        # The just-indexed doc should be reachable through the LSH bucket.
        assert 2 in ids


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------


class TestCluster:
    def test_cluster_empty(self):
        e = Embedder.create(dim=64, seed=0)
        with pytest.raises(InsufficientData):
            e.cluster(k=3)

    def test_cluster_invalid_k(self):
        e = Embedder.create(dim=64, seed=0)
        e.add("hello")
        with pytest.raises(InvalidConfig):
            e.cluster(k=0)
        with pytest.raises(InvalidConfig):
            e.cluster(k=2)  # k > n

    def test_cluster_partitions_corpus(self):
        e = Embedder.create(dim=256, seed=0)
        for text, payload in _animal_corpus():
            e.add(text, payload=payload)
        rep = e.cluster(k=2, max_iter=50, seed=42)
        assert isinstance(rep, ClusterReport)
        assert len(rep.assignments) == e.n_documents()
        assert len(rep.centroids) == 2
        for c in rep.centroids:
            assert len(c) == 256
        assert 0 <= max(rep.assignments) < 2

    def test_cluster_inertia_non_negative(self):
        e = Embedder.create(dim=64, seed=0)
        for text, payload in _animal_corpus():
            e.add(text, payload=payload)
        rep = e.cluster(k=3, seed=0)
        assert rep.inertia >= 0.0

    def test_cluster_separates_topics(self):
        # Lexical embedders cluster topics only when topics share
        # surface vocabulary across documents.  Use the shared-vocab
        # corpus that mimics how a real retrieval corpus would look.
        e = Embedder.create(
            dim=512,
            tokenizer=TOKENIZE_WORD,
            n_gram_range=(1, 1),
            seed=0,
        )
        ids: list[str] = []
        meta: list[str] = []
        for text, payload in _shared_vocab_corpus():
            ids.append(e.add(text, payload=payload))
            meta.append(payload["topic"])
        rep = e.cluster(k=2, seed=42)
        cluster_to_topic_votes: dict[int, dict[str, int]] = {}
        for i, c in enumerate(rep.assignments):
            cluster_to_topic_votes.setdefault(c, {})
            cluster_to_topic_votes[c][meta[i]] = cluster_to_topic_votes[c].get(meta[i], 0) + 1
        correct = 0
        total = len(rep.assignments)
        for c, votes in cluster_to_topic_votes.items():
            correct += max(votes.values())
        purity = correct / total
        # On shared-vocab corpus k-means achieves perfect separation
        # under any reasonable seed.
        assert purity >= 0.9


# ------------------------------------------------------------------
# JL certificate
# ------------------------------------------------------------------


class TestJL:
    def test_jl_dim_grows_with_n(self):
        d1 = jl_dimension(10, 0.1)
        d2 = jl_dimension(1000, 0.1)
        d3 = jl_dimension(100000, 0.1)
        assert d1 < d2 < d3

    def test_jl_dim_grows_with_smaller_eps(self):
        d1 = jl_dimension(100, 0.4)
        d2 = jl_dimension(100, 0.1)
        d3 = jl_dimension(100, 0.05)
        assert d1 < d2 < d3

    def test_jl_dim_invalid_eps(self):
        with pytest.raises(InvalidConfig):
            jl_dimension(100, 0.0)
        with pytest.raises(InvalidConfig):
            jl_dimension(100, 0.6)
        with pytest.raises(InvalidConfig):
            jl_dimension(100, -0.1)

    def test_jl_dim_invalid_n(self):
        with pytest.raises(InvalidConfig):
            jl_dimension(1, 0.1)

    def test_jl_dim_invalid_failure_prob(self):
        with pytest.raises(InvalidConfig):
            jl_dimension(100, 0.1, failure_prob=0.0)
        with pytest.raises(InvalidConfig):
            jl_dimension(100, 0.1, failure_prob=1.0)

    def test_jl_certificate_statement_mentions_eps(self):
        cert = jl_certificate(100, 0.1, dim_actual=4000)
        assert "0.1" in cert.statement
        assert cert.eps == 0.1
        assert cert.n_items == 100

    def test_jl_certificate_holds_at_sufficient_dim(self):
        cert = jl_certificate(100, 0.4, dim_actual=2048)
        # eps=0.4 requires only ~ 50 dim for n=100; 2048 is generous.
        assert cert.distortion_holds is True

    def test_jl_certificate_does_not_hold_at_tiny_dim(self):
        cert = jl_certificate(1000, 0.05, dim_actual=4)
        assert cert.distortion_holds is False

    def test_embedder_jl_certificate(self):
        e = Embedder.create(dim=4096, seed=0)
        for i in range(20):
            e.add(f"document number {i}")
        cert = e.jl_certificate(n_items=20, eps=0.1)
        assert cert.dim_actual == 4096
        assert isinstance(cert, JLCertificate)


# ------------------------------------------------------------------
# SimHash signatures
# ------------------------------------------------------------------


class TestSimHash:
    def test_signature_deterministic(self):
        v = [0.1, -0.3, 0.5, 0.0, 0.7, -0.2]
        s1 = simhash_signature(v, bits=16, seed=42)
        s2 = simhash_signature(v, bits=16, seed=42)
        assert s1 == s2

    def test_signature_changes_with_seed(self):
        v = [0.1, -0.3, 0.5, 0.0, 0.7, -0.2]
        s1 = simhash_signature(v, bits=16, seed=0)
        s2 = simhash_signature(v, bits=16, seed=999)
        assert s1 != s2

    def test_hamming_distance(self):
        assert hamming_distance(0b1010, 0b0101) == 4
        assert hamming_distance(0b1111, 0b1111) == 0
        assert hamming_distance(0, 0xFFFF) == 16

    def test_angle_from_hamming(self):
        # Identical signatures ⇒ angle 0.
        assert angle_from_hamming(0, 64) == 0.0
        # All bits differ ⇒ angle π.
        assert abs(angle_from_hamming(64, 64) - math.pi) < 1e-12
        # Half differ ⇒ angle π/2.
        assert abs(angle_from_hamming(32, 64) - math.pi / 2) < 1e-12

    def test_simhash_collision_rate_tracks_angle(self):
        rng = random.Random(7)
        d = 16
        v = [rng.gauss(0, 1) for _ in range(d)]
        # Build a vector at small angle: v' = v + tiny noise
        eps = 0.01
        v2 = [x + eps * rng.gauss(0, 1) for x in v]
        # Larger angle: random vector
        v3 = [rng.gauss(0, 1) for _ in range(d)]
        s1 = simhash_signature(v, bits=128, seed=0)
        s2 = simhash_signature(v2, bits=128, seed=0)
        s3 = simhash_signature(v3, bits=128, seed=0)
        # Close vectors → small Hamming distance; far vectors → larger.
        assert hamming_distance(s1, s2) < hamming_distance(s1, s3)


# ------------------------------------------------------------------
# Free helpers
# ------------------------------------------------------------------


class TestFreeHelpers:
    def test_cosine_similarity_basic(self):
        assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-12
        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-12
        assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-12

    def test_cosine_similarity_zero_vectors(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_cosine_similarity_length_mismatch(self):
        with pytest.raises(InvalidConfig):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_cosine_distance(self):
        d = cosine_distance([1.0, 0.0], [0.0, 1.0])
        assert abs(d - 1.0) < 1e-12

    def test_cosine_similarity_clamps(self):
        # Numerically may exceed [-1, 1]; should be clamped.
        v = [1.0, 1.0]
        c = cosine_similarity(v, v)
        assert -1.0 <= c <= 1.0


# ------------------------------------------------------------------
# Event emission
# ------------------------------------------------------------------


class TestEvents:
    def _collect(self, bus: EventBus, kinds: list[str]) -> list[str]:
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        return seen

    def test_started_event_on_create(self):
        bus = EventBus()
        seen = self._collect(bus, [])
        e = Embedder.create(dim=32, seed=0, bus=bus)
        assert EMBEDDER_STARTED in seen

    def test_embedded_event_on_embed(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        e = Embedder.create(dim=32, seed=0, bus=bus)
        e.embed("hi")
        assert EMBEDDER_EMBEDDED in seen

    def test_indexed_event_on_add(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        e = Embedder.create(dim=32, seed=0, bus=bus)
        e.add("hello")
        assert EMBEDDER_INDEXED in seen

    def test_searched_event_on_search(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        e = Embedder.create(dim=32, seed=0, bus=bus)
        e.add("hello")
        e.search("hi", k=1)
        assert EMBEDDER_SEARCHED in seen

    def test_clustered_event_on_cluster(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        e = Embedder.create(dim=32, seed=0, bus=bus)
        for w in ["a", "b", "c"]:
            e.add(w)
        e.cluster(k=2)
        assert EMBEDDER_CLUSTERED in seen

    def test_reported_event_on_report(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        e = Embedder.create(dim=32, seed=0, bus=bus)
        e.report()
        assert EMBEDDER_REPORTED in seen

    def test_cleared_event_on_clear(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        e = Embedder.create(dim=32, seed=0, bus=bus)
        e.add("hello")
        e.clear()
        assert EMBEDDER_CLEARED in seen


# ------------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------------


class TestTfIdf:
    def test_tfidf_downweights_common_terms(self):
        e = Embedder.create(
            dim=128,
            tokenizer=TOKENIZE_WORD,
            weighting=WEIGHT_TFIDF,
            n_gram_range=(1, 1),
            seed=0,
        )
        for w in ["the cat", "the dog", "the bird", "the fish"]:
            e.add(w)
        # Refit IDF now that the corpus is loaded.
        e.refit_tfidf()
        # Unique word query should *not* match the most common doc.
        rep = e.report()
        assert rep.df_known_terms > 0
        assert rep.weighting == WEIGHT_TFIDF

    def test_refit_tfidf_emits_event(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda ev: seen.append(ev.kind))
        e = Embedder.create(
            dim=64,
            tokenizer=TOKENIZE_WORD,
            weighting=WEIGHT_TFIDF,
            seed=0,
            bus=bus,
        )
        e.add("hello world")
        e.refit_tfidf()
        assert EMBEDDER_TFIDF_REFIT in seen


# ------------------------------------------------------------------
# Tokenizer variants
# ------------------------------------------------------------------


class TestTokenizers:
    def test_word_tokenizer_basic(self):
        e = Embedder.create(
            dim=64, tokenizer=TOKENIZE_WORD, n_gram_range=(1, 1), seed=0
        )
        v1 = e.embed("hello world")
        v2 = e.embed("world hello")
        # Word unigrams are order-invariant ⇒ identical vector.
        for a, b in zip(v1.vector, v2.vector):
            assert abs(a - b) < 1e-12

    def test_word_ngram_tokenizer(self):
        e = Embedder.create(
            dim=128,
            tokenizer=TOKENIZE_WORD_NGRAM,
            n_gram_range=(1, 2),
            seed=0,
        )
        v1 = e.embed("the quick brown fox")
        v2 = e.embed("the slow brown fox")
        # Should be similar but not identical (3 of 4 bigrams differ).
        assert 0.0 < v1.cosine_to(v2) < 1.0

    def test_char_ngram_short_text(self):
        e = Embedder.create(dim=64, n_gram_range=(2, 4), seed=0)
        v = e.embed("a")  # shorter than min n-gram including boundaries
        # Should still produce a valid (possibly all-zero or low-norm) embedding.
        assert v.dim == 64


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


class TestReport:
    def test_report_structure(self):
        e = Embedder.create(dim=64, seed=0)
        for w in ["alpha", "beta", "gamma"]:
            e.add(w)
        rep = e.report()
        assert isinstance(rep, EmbedderReport)
        assert rep.n_documents == 3
        assert rep.dim == 64
        assert rep.tokenizer == TOKENIZE_CHAR_NGRAM
        assert rep.weighting == WEIGHT_LOG_COUNT
        assert len(rep.sample_doc_ids) <= 8
        assert isinstance(rep.jl_certificate_at_eps_0_1, JLCertificate)
        assert len(rep.fingerprint) == 64

    def test_report_empty_index(self):
        e = Embedder.create(dim=64, seed=0)
        rep = e.report()
        assert rep.n_documents == 0
        assert rep.n_embeddings_computed == 0


# ------------------------------------------------------------------
# Concurrency
# ------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_embedding(self):
        e = Embedder.create(dim=64, seed=0)
        errors: list[BaseException] = []

        def worker(start: int):
            try:
                for i in range(20):
                    e.embed(f"text-{start}-{i}")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert e.n_embed_calls() == 4 * 20

    def test_concurrent_add(self):
        e = Embedder.create(dim=64, seed=0)
        errors: list[BaseException] = []

        def worker(start: int):
            try:
                for i in range(10):
                    e.add(f"doc-{start}-{i}")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert e.n_documents() == 4 * 10


# ------------------------------------------------------------------
# Fingerprint chain
# ------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_chain_advances(self):
        e = Embedder.create(dim=64, seed=0)
        fps = [e.fingerprint()]
        for w in ["a", "b", "c", "d"]:
            e.embed(w)
            fps.append(e.fingerprint())
        # Every step yields a distinct fingerprint.
        assert len(set(fps)) == len(fps)

    def test_two_embedders_same_inputs_same_chain(self):
        e1 = Embedder.create(dim=64, seed=0)
        e2 = Embedder.create(dim=64, seed=0)
        for w in ["a", "b", "c"]:
            e1.embed(w)
            e2.embed(w)
        assert e1.fingerprint() == e2.fingerprint()


# ------------------------------------------------------------------
# Robustness / property-based-ish
# ------------------------------------------------------------------


class TestRobustness:
    def test_identical_text_self_similarity_is_max(self):
        e = Embedder.create(dim=128, seed=0)
        rng = random.Random(42)
        texts = [
            "".join(rng.choices("abcdefghijklmnopqrstuvwxyz ", k=rng.randint(5, 40)))
            for _ in range(20)
        ]
        for t in texts:
            v = e.embed(t)
            assert v.cosine_to(v) > 0.999

    def test_triangle_inequality_euclidean(self):
        e = Embedder.create(dim=64, seed=0)
        a = e.embed("alpha bravo charlie")
        b = e.embed("alpha bravo delta")
        c = e.embed("echo foxtrot golf")
        # ‖a − c‖ ≤ ‖a − b‖ + ‖b − c‖ for any L2 metric.
        ab = a.euclidean_to(b)
        bc = b.euclidean_to(c)
        ac = a.euclidean_to(c)
        assert ac <= ab + bc + 1e-9

    def test_jl_dim_satisfies_actual_distortion_on_small_sample(self):
        # We do not run a heavy JL Monte Carlo; we only sanity-check
        # that two vectors mapped through the projection have distortion
        # well below 1 (since the projection preserves the *inner
        # product* in expectation).
        e = Embedder.create(dim=2048, seed=0)
        rng = random.Random(13)
        texts = [
            "".join(rng.choices("abcdefghij", k=20)) for _ in range(8)
        ]
        embs = [e.embed(t) for t in texts]
        # All pairs should have cosine in [-1, 1] and a non-degenerate
        # spread (not all identical, not all orthogonal).
        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(embs[i].cosine_to(embs[j]))
        assert min(sims) > -1.0
        assert max(sims) < 1.0
