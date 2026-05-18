"""Embedder demo: distortion-bounded text embeddings, search, and clustering.

What this shows
---------------

The ``Embedder`` primitive maps text to a fixed-dimension real vector via
a composition of three classical, pure-Python, deterministic transforms,
each carrying a *finite-sample distortion certificate*:

    text  ── HashingVectorizer (Weinberger et al. 2009) ──>  sparse ℝ^N
          ── sparse Random Projection (Achlioptas 2003)  ──>  dense  ℝ^d
          ── L2 normalisation                            ──>  unit-norm v

The Johnson-Lindenstrauss lemma (Dasgupta-Gupta 2003) provides a
finite-sample bound on the embedding dimension required to preserve
all pairwise distances within (1 ± ε) with high probability.  No
external embedding service, no learned weights, no opaque dependency.

Run with::

    python -m examples.embedder_demo

What you see
------------

1. Embedding a small corpus and showing pairwise cosine similarity.
2. Top-K nearest neighbour search over a corpus by cosine similarity.
3. LSH (SimHash) accelerated approximate retrieval.
4. Spherical k-means++ clustering with audit fingerprint.
5. JL certificate for a target ε.
"""
from __future__ import annotations

import textwrap

from agi.embedder import (
    Embedder,
    TOKENIZE_WORD,
    WEIGHT_LOG_COUNT,
    jl_certificate,
    jl_dimension,
)


CORPUS = [
    ("python is a programming language with dynamic typing", "tech"),
    ("rust compiles to native code with no garbage collector", "tech"),
    ("haskell embraces purity and lazy evaluation", "tech"),
    ("the lambda calculus is the foundation of functional programming", "tech"),
    ("the cat sat on the mat and watched the rain fall", "animal"),
    ("the kitten slept on the rug as the storm passed", "animal"),
    ("a kitten plays with yarn while the dog watches", "animal"),
    ("the dog barks at the fence post every morning", "animal"),
    ("a puppy chases its tail in the afternoon sunshine", "animal"),
    ("a bird sings from the maple tree at dawn", "animal"),
    ("quantum mechanics is hard but elegant", "science"),
    ("relativity bends spacetime around heavy objects", "science"),
    ("entropy increases over time in closed systems", "science"),
    ("the planck constant sets the scale of quanta", "science"),
]


def banner(s: str) -> None:
    bar = "=" * len(s)
    print(f"\n{bar}\n{s}\n{bar}")


def main() -> None:
    banner("1. Embedder construction with a fingerprint chain")
    emb = Embedder.create(
        dim=256,
        tokenizer=TOKENIZE_WORD,
        weighting=WEIGHT_LOG_COUNT,
        n_gram_range=(1, 2),
        seed=42,
    )
    print(f"  dim         = {emb.dim}")
    print(f"  seed        = {emb.seed}")
    print(f"  tokenizer   = {emb.tokenizer}")
    print(f"  weighting   = {emb.weighting}")
    print(f"  fingerprint = {emb.fingerprint()[:24]}...")

    banner("2. Index the corpus")
    for text, topic in CORPUS:
        emb.add(text, payload={"topic": topic, "text": text})
    rep = emb.report()
    print(f"  n_documents             = {rep.n_documents}")
    print(f"  total term observations = {rep.total_term_observations}")
    print(f"  distinct df-known terms = {rep.df_known_terms}")
    print(f"  fingerprint after add   = {rep.fingerprint[:24]}...")

    banner("3. Top-K cosine search")
    queries = [
        "a small cat playing in the living room",
        "compile a rust binary",
        "the universe and black holes",
    ]
    for q in queries:
        print(f"\n  query: {q!r}")
        for h in emb.search(q, k=3):
            row = textwrap.shorten(h.payload["text"], width=56)
            print(f"    {h.score:+.3f}  [{h.payload['topic']:7s}]  {row}")

    banner("4. LSH (SimHash) accelerated approximate retrieval")
    emb.build_lsh_index(n_bands=8, bits_per_band=6)
    for q in queries:
        print(f"\n  query: {q!r}")
        for h in emb.search_lsh(q, k=3, n_bands=8, bits_per_band=6):
            row = textwrap.shorten(h.payload["text"], width=56)
            print(f"    {h.score:+.3f}  [{h.payload['topic']:7s}]  {row}")

    banner("5. Spherical k-means++ clustering")
    cr = emb.cluster(k=3, max_iter=50, seed=7)
    print(f"  k          = {cr.k}")
    print(f"  iterations = {cr.iterations}  converged = {cr.converged}")
    print(f"  inertia    = {cr.inertia:.4f}")
    for c in range(cr.k):
        member_ids = [
            cr.doc_ids[i] for i, ci in enumerate(cr.assignments) if ci == c
        ]
        topics = [emb.get(did)["payload"]["topic"] for did in member_ids]
        print(f"\n  cluster {c}: {len(member_ids)} members  topics={sorted(set(topics))}")
        for did in member_ids[:4]:
            txt = textwrap.shorten(emb.get(did)["payload"]["text"], width=56)
            print(f"    - {txt}")

    banner("6. Johnson-Lindenstrauss certificate at ε=0.1")
    cert = jl_certificate(n_items=rep.n_documents, eps=0.1, dim_actual=emb.dim)
    print(f"  n_items        = {cert.n_items}")
    print(f"  eps            = {cert.eps}")
    print(f"  dim required   = {cert.dim_required}")
    print(f"  dim actual     = {cert.dim_actual}")
    print(f"  bound holds    = {cert.distortion_holds}")
    print(f"  failure prob   ≤ {cert.failure_probability:.3g}")
    print("\n  Statement:")
    for line in textwrap.wrap(cert.statement, width=70):
        print(f"    {line}")

    banner("7. Required dimension vs corpus size at ε=0.1")
    print(f"  {'n':>10}    {'d_required':>10}")
    for n in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
        d_req = jl_dimension(n, 0.1)
        print(f"  {n:>10}    {d_req:>10}")

    banner("8. Final report and audit fingerprint")
    rep2 = emb.report()
    print(f"  embed_count      = {rep2.n_embeddings_computed}")
    print(f"  n_documents      = {rep2.n_documents}")
    print(f"  final fingerprint = {rep2.fingerprint[:24]}...")


if __name__ == "__main__":
    main()
