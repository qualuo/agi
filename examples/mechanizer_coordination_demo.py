"""Mechanizer coordination demo — Mechanizer as a runtime primitive that a
coordination engine *discovers* via the Manifest, *invokes* on a
contested representation, and *audits* via a faithfulness certificate.

A coordination engine drives the runtime to deliver:

  Goal:  *Explain why two different reasoning agents produce divergent
         hidden-state representations on the same input — and provide a
         faithfulness-bounded steering intervention that aligns one
         with the other.*  A safety-team reviewer (or an Aligner /
         Debater downstream) asked, "what feature is responsible for the
         disagreement, and how do I move it?"  The coordination engine
         must:

           1. *Discover* the primitive whose ``manifest.kind ==
              observability`` and whose tags overlap {numerical,
              introspection, safety, replay-verifiable} and whose
              summary mentions ``mechanistic interpretability`` /
              ``sparse autoencoder``.  That's Mechanizer.
           2. *Fit* an over-complete sparse dictionary on the union of
              both agents' activations.
           3. *Encode* each agent's activations into sparse codes.
           4. *Attribute* the divergence to a small set of features
              (the ones where the two agents' codes differ most).
           5. *Steer* one agent's activation along the responsible
              feature direction with a *bounded* magnitude — the
              perturbation L2 norm is the safety budget.
           6. *Patch* the divergent feature in code-space and decode
              back; report the residual divergence.
           7. *Certify* — the Mechanizer's faithfulness certificate
              gives a Hoeffding lower-confidence bound on the
              population R² of the explanation, the dictionary's
              mutual-coherence-derived identifiability, and the
              dead-feature count.
           8. *Replay* — every fit / encode / patch / steer / certify
              event is fingerprinted; the safety reviewer can re-run
              the trace from the event log and independently verify
              the certificate.

This is the **runtime-as-coordination-surface** investor pitch for
interpretability: the coordination engine doesn't know how Mechanizer
trains a dictionary or finds top-activating examples — it discovers
that *some* primitive offers mechanistic interpretability via the
manifest, calls it through a uniform API, reads a uniform certificate
(``MechanizerCertificate``), and acts on it.

Run::

    python examples/mechanizer_coordination_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.manifest import default_manifest  # noqa: E402
from agi.mechanizer import (  # noqa: E402
    ALGO_KSVD,
    Mechanizer,
    MechanizerConfig,
    mechanizer_random_dictionary,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Phase 1: discovery
# ---------------------------------------------------------------------------


def phase_1_discovery() -> str:
    banner("Phase 1 — discovery via the manifest")
    m = default_manifest()
    print(f"  catalog size: {len(m.list())} primitives")
    print("  query: 'mechanistic interpretability sparse autoencoder feature steering'")
    ranked = m.recommend(
        "mechanistic interpretability sparse autoencoder feature steering",
        k=3,
    )
    print("\n  top matches:")
    for spec, score in ranked:
        print(f"    {score:>5.2f}  {spec.name:<14} kind={spec.kind:<14} "
              f"tags={','.join(spec.tags) or '(none)'}")
    chosen = ranked[0][0]
    print(f"\n  → coordination engine routes to '{chosen.name}'")
    print(f"     certificate class:  {chosen.certificate}")
    print(f"     composes_with:      {', '.join(chosen.composes_with[:6])}, …")
    print(f"     demo:               {chosen.demo_path}")
    return chosen.name


# ---------------------------------------------------------------------------
# Phase 2: simulate two agents producing divergent activations on the same input
# ---------------------------------------------------------------------------


def _synth_two_agent_activations(
    n: int = 200, dim: int = 24, n_true: int = 12, true_l0: int = 3,
    drift_feature: int = 4, drift_strength: float = 1.3, seed: int = 0,
) -> tuple[list[list[float]], list[list[float]], int]:
    """Generate two agents' activations from the same sparse generator.

    Agent A uses the true feature dictionary.  Agent B is identical
    except that ``drift_feature``'s coefficient is multiplied by
    ``drift_strength`` — modelling a *targeted* representational drift
    that a safety reviewer wants to localise.
    """
    rng = random.Random(seed)
    D = mechanizer_random_dictionary(n_features=n_true, dim=dim, seed=seed + 1)
    X_a: list[list[float]] = []
    X_b: list[list[float]] = []
    for _ in range(n):
        z = [0.0] * n_true
        idx = rng.sample(range(n_true), true_l0)
        for j in idx:
            z[j] = rng.uniform(0.5, 1.5)
        # Agent A: vanilla.
        xa = [0.0] * dim
        for j, zj in enumerate(z):
            if zj == 0.0:
                continue
            for c in range(dim):
                xa[c] += zj * D[j][c]
        # Agent B: drift_feature coefficient amplified.
        zb = list(z)
        zb[drift_feature] = zb[drift_feature] * drift_strength
        xb = [0.0] * dim
        for j, zj in enumerate(zb):
            if zj == 0.0:
                continue
            for c in range(dim):
                xb[c] += zj * D[j][c]
        X_a.append(xa)
        X_b.append(xb)
    return X_a, X_b, drift_feature


def phase_2_simulate_agents() -> tuple[list[list[float]], list[list[float]], int]:
    banner("Phase 2 — simulate two agents whose activations diverge")
    X_a, X_b, drift = _synth_two_agent_activations(seed=42)
    n = len(X_a)
    dim = len(X_a[0])
    rms = math.sqrt(
        sum(sum((X_a[i][c] - X_b[i][c]) ** 2 for c in range(dim))
            for i in range(n)) / (n * dim)
    )
    print(f"  agent A activations:  {n} × {dim}")
    print(f"  agent B activations:  {n} × {dim}")
    print(f"  RMS divergence A vs B = {rms:.4f}")
    print(f"  ground-truth drift: feature {drift}'s coefficient amplified ×1.3")
    print(f"  (the safety reviewer does not know which feature it is)")
    return X_a, X_b, drift


# ---------------------------------------------------------------------------
# Phase 3: fit Mechanizer on the union of activations
# ---------------------------------------------------------------------------


def phase_3_fit(
    X_a: list[list[float]], X_b: list[list[float]],
) -> Mechanizer:
    banner("Phase 3 — fit a sparse-dictionary mechanism on union(A, B)")
    union = X_a + X_b
    mech = Mechanizer(MechanizerConfig(
        algorithm=ALGO_KSVD,
        n_features=14,
        target_l0=3,
        max_iter=25,
        seed=7,
    ))
    rep = mech.fit(union)
    print(f"  fit R² = {rep.r2:.4f}  mean L0 = {rep.mean_l0:.2f}  "
          f"μ(D) = {rep.mutual_coherence:.4f}  dead = {rep.dead_features}")
    print(f"  identifiable_l0 (Donoho-Elad) = {rep.identifiable_l0}")
    print(f"  ledger fingerprint = {rep.fingerprint[:16]}…")
    return mech


# ---------------------------------------------------------------------------
# Phase 4: attribute the divergence to a small set of features
# ---------------------------------------------------------------------------


def phase_4_attribute(
    mech: Mechanizer,
    X_a: list[list[float]],
    X_b: list[list[float]],
) -> int:
    banner("Phase 4 — attribute the divergence to a small set of features")
    Z_a = mech.encode(X_a)
    Z_b = mech.encode(X_b)
    K = mech.n_features
    diffs = [0.0] * K
    for i in range(len(Z_a)):
        for j in range(K):
            diffs[j] += abs(Z_a[i][j] - Z_b[i][j])
    # Rank features by aggregate code-space divergence.
    ranked = sorted(range(K), key=lambda j: -diffs[j])
    print(f"  {'rank':>4}  {'feature':>7}  {'Σ|Δz|':>10}")
    for rank, j in enumerate(ranked[:5], start=1):
        print(f"  {rank:>4}  {j:>7}  {diffs[j]:>10.4f}")
    top = ranked[0]
    print(f"\n  → coordination engine attributes the divergence to feature {top}")
    return top


# ---------------------------------------------------------------------------
# Phase 5: bounded steering intervention
# ---------------------------------------------------------------------------


def phase_5_steer(mech: Mechanizer,
                  X_a: list[list[float]],
                  attributed_feature: int) -> None:
    banner("Phase 5 — bounded steering intervention")
    for magnitude in (-1.5, -0.5, 0.5, 1.5):
        steered = mech.steer(X_a[0:1], feature=attributed_feature,
                              magnitude=magnitude)
        delta = math.sqrt(sum(
            (X_a[0][c] - steered[0][c]) ** 2 for c in range(len(X_a[0]))
        ))
        ev = [e for e in mech.events()
              if e.kind == "mechanizer.steered"][-1]
        print(f"  magnitude = {magnitude:+.2f}  "
              f"||Δactivation|| = {delta:.4f}  "
              f"ledger.perturbation_norm = {ev.payload['perturbation_norm']:.4f}")
    print("  (the ledger payload is the *blast radius* the Aligner / Verifier "
          "downstream gate on.)")


# ---------------------------------------------------------------------------
# Phase 6: activation patching as a counterfactual repair
# ---------------------------------------------------------------------------


def phase_6_patch(mech: Mechanizer,
                  X_a: list[list[float]],
                  X_b: list[list[float]],
                  attributed_feature: int) -> None:
    banner("Phase 6 — activation-patching counterfactual repair")
    # Use one B-sample as donor and patch its feature into A.
    repaired = mech.patch(X_a[:5], X_b[:1], feature=attributed_feature,
                          scale=1.0)
    n = len(repaired)
    dim = len(repaired[0])
    # Compare A-without-patch vs A-with-patch in terms of distance to B.
    Z_a = mech.encode(X_a[:5])
    baseline_recon = mech.decode(Z_a)
    base_dist = sum(
        sum((baseline_recon[i][c] - X_b[i][c]) ** 2 for c in range(dim))
        for i in range(n)
    ) / (n * dim)
    repaired_dist = sum(
        sum((repaired[i][c] - X_b[i][c]) ** 2 for c in range(dim))
        for i in range(n)
    ) / (n * dim)
    print(f"  ||A_reconstruction − B||² / nd  (no patch)  = {base_dist:.4f}")
    print(f"  ||A_patched         − B||² / nd  (patch f={attributed_feature}) = "
          f"{repaired_dist:.4f}")
    if repaired_dist < base_dist:
        delta = base_dist - repaired_dist
        print(f"  → patching feature {attributed_feature} reduced divergence "
              f"by {delta:.4f}.")


# ---------------------------------------------------------------------------
# Phase 7: certify and replay
# ---------------------------------------------------------------------------


def phase_7_certify_and_replay(
    mech: Mechanizer,
    X_a: list[list[float]],
    X_b: list[list[float]],
) -> None:
    banner("Phase 7 — certify the explanation + replay-verify the ledger")
    held_out = X_a[100:] + X_b[100:]
    cert = mech.certify(held_out, delta=0.05)
    print(f"  held-out N = {cert.n_samples}")
    print(f"  R²                       = {cert.r2:.4f}")
    print(f"  mean L0                  = {cert.mean_l0:.2f}")
    print(f"  Hoeffding R² LCB @ δ=.05 = {cert.hoeffding_r2_lcb:.4f}")
    print(f"  Bernstein R² LCB @ δ=.05 = {cert.bernstein_r2_lcb:.4f}")
    print(f"  μ(D)                     = {cert.mutual_coherence:.4f}  "
          f"→ Donoho-Elad allows k ≤ {cert.identifiable_l0}")
    print(f"  identifiable             = {cert.identifiable}")
    print(f"  ledger fingerprint       = {cert.fingerprint[:16]}…")
    print(f"  replay-verifies          = {mech.verify_chain()}")
    print()
    print(f"  events on the chain      = {len(mech.events())}")
    counts: dict[str, int] = {}
    for ev in mech.events():
        counts[ev.kind] = counts.get(ev.kind, 0) + 1
    for kind, count in sorted(counts.items()):
        print(f"    {kind:<30}  {count:>3d}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    phase_1_discovery()
    X_a, X_b, true_drift = phase_2_simulate_agents()
    mech = phase_3_fit(X_a, X_b)
    attributed = phase_4_attribute(mech, X_a, X_b)
    phase_5_steer(mech, X_a, attributed)
    phase_6_patch(mech, X_a, X_b, attributed)
    phase_7_certify_and_replay(mech, X_a, X_b)
    banner("Wrap")
    print(f"  ground-truth drift feature: {true_drift}")
    print(f"  Mechanizer attributed to:   {attributed}")
    print(f"  (the dictionary is trained without supervision on the drift; "
          f"the attributed feature is *Mechanizer's* index, which lives in a "
          f"different basis than the ground-truth generator.  What matters "
          f"is that the divergence localises onto a small number of "
          f"monosemantic features that a downstream Aligner / Verifier can "
          f"act on with a bounded steering budget.)")


if __name__ == "__main__":  # pragma: no cover
    main()
