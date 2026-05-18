"""Mechanizer demo — mechanistic interpretability as a runtime primitive.

Five scenarios, each demonstrating a different facet of the primitive:

  1. Sparse-autoencoder fit on synthetic features generated from a
     known sparse model — recover the feature structure and verify a
     Donoho-Elad identifiability certificate.
  2. K-SVD vs Top-K vs L1 vs PCA — head-to-head on the same activation
     matrix; show the trade-off between reconstruction R², L0
     sparsity, and dictionary mutual coherence.
  3. Activation patching in code space — take one input's value of a
     learned feature and inject it into another input; observe the
     counterfactual reconstruction.
  4. Feature steering — bump a chosen feature's atom into the
     reconstructed activation and read off the *blast radius* (L2
     perturbation norm) the ledger records.
  5. Auto-interpretation + circuit graph — surface the top-activating
     examples for each feature and build the feature-feature
     co-activation graph.

Run with::

    python examples/mechanizer_demo.py
"""
from __future__ import annotations

from agi.mechanizer import (
    ALGO_KSVD,
    ALGO_L1_SAE,
    ALGO_PCA,
    ALGO_TOPK_SAE,
    Mechanizer,
    MechanizerConfig,
    mechanizer_synthetic_features,
)


def _print_section(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


# =====================================================================
# 1. Sparse-autoencoder fit and identifiability certificate
# =====================================================================


def scenario_1_sparse_recovery() -> None:
    _print_section(
        "[1] Sparse recovery — Donoho-Elad identifiability gate on synthetic data"
    )
    X = mechanizer_synthetic_features(
        n=200, dim=24, n_true=12, true_l0=3, seed=0,
    )
    mech = Mechanizer(MechanizerConfig(
        algorithm=ALGO_KSVD,
        n_features=12,
        target_l0=3,
        max_iter=25,
        seed=0,
    ))
    rep = mech.fit(X)
    cert = mech.certify(X, delta=0.05)
    print(f"  fit:    R² = {rep.r2:.4f}  mean L0 = {rep.mean_l0:.2f}  "
          f"dead = {rep.dead_features}")
    print(f"  cert:   R² = {cert.r2:.4f}  mean L0 = {cert.mean_l0:.2f}  "
          f"dead = {cert.dead_features}")
    print(f"  Hoeffding R² LCB @ δ=0.05 = {cert.hoeffding_r2_lcb:.4f}")
    print(f"  Bernstein R² LCB @ δ=0.05 = {cert.bernstein_r2_lcb:.4f}")
    print(f"  μ(D) = {cert.mutual_coherence:.4f}  → "
          f"Donoho-Elad allows k ≤ {cert.identifiable_l0}  "
          f"(actual mean L0 = {cert.mean_l0:.2f}) → "
          f"identifiable = {cert.identifiable}")
    print(f"  ledger fingerprint = {cert.fingerprint[:16]}…")


# =====================================================================
# 2. Algorithm head-to-head
# =====================================================================


def scenario_2_algorithm_comparison() -> None:
    print()
    _print_section("[2] Algorithm head-to-head — R² vs L0 vs μ(D)")
    X = mechanizer_synthetic_features(
        n=150, dim=20, n_true=14, true_l0=3, seed=1,
    )
    configs = [
        (ALGO_TOPK_SAE, dict(target_l0=3, learning_rate=5e-2, max_iter=80)),
        (ALGO_L1_SAE,   dict(target_l0=3, l1_coeff=5e-2, learning_rate=2e-2, max_iter=80)),
        (ALGO_KSVD,     dict(target_l0=3, max_iter=20)),
        (ALGO_PCA,      dict(target_l0=3, max_iter=1)),
    ]
    print(f"  {'algorithm':>10}  {'R²':>8}  {'mean L0':>8}  {'μ(D)':>8}  "
          f"{'dead':>6}  {'iters':>6}")
    for algo, kw in configs:
        mech = Mechanizer(MechanizerConfig(
            algorithm=algo, n_features=14, seed=2, **kw,
        ))
        rep = mech.fit(X)
        print(f"  {algo:>10}  {rep.r2:>8.4f}  {rep.mean_l0:>8.2f}  "
              f"{rep.mutual_coherence:>8.4f}  {rep.dead_features:>6d}  "
              f"{rep.iterations:>6d}")


# =====================================================================
# 3. Activation patching
# =====================================================================


def scenario_3_activation_patching() -> None:
    print()
    _print_section(
        "[3] Activation patching — counterfactual feature swap in code space"
    )
    X = mechanizer_synthetic_features(
        n=80, dim=16, n_true=12, true_l0=3, seed=2,
    )
    mech = Mechanizer(MechanizerConfig(
        algorithm=ALGO_KSVD, n_features=12, target_l0=3,
        max_iter=20, seed=3,
    ))
    mech.fit(X)
    target = X[0:1]
    donor = X[1:2]
    # Baseline: encode → decode the target with no patch.
    baseline = mech.decode(mech.encode(target))
    print(f"  baseline ||target − reconstruction|| = "
          f"{_norm_diff(target[0], baseline[0]):.4f}")
    for feature in (0, 3, 7, 11):
        patched = mech.patch(target, donor, feature=feature, scale=1.0)
        delta = _norm_diff(baseline[0], patched[0])
        print(f"  patch feature {feature:2d}  "
              f"→  ||reconstruction shifted by|| = {delta:.4f}")
    # Multi-feature patch with interpolation.
    interp = mech.patch(target, donor, feature=[0, 3, 7, 11], scale=0.5)
    print(f"  4-feature patch @ scale=0.5  "
          f"→  ||shift|| = {_norm_diff(baseline[0], interp[0]):.4f}")


# =====================================================================
# 4. Feature steering
# =====================================================================


def scenario_4_feature_steering() -> None:
    print()
    _print_section(
        "[4] Feature steering — perturbation L2 norm recorded in the ledger"
    )
    X = mechanizer_synthetic_features(
        n=60, dim=16, n_true=12, true_l0=3, seed=4,
    )
    mech = Mechanizer(MechanizerConfig(
        algorithm=ALGO_KSVD, n_features=12, target_l0=3,
        max_iter=15, seed=5,
    ))
    mech.fit(X)
    target = X[0:1]
    for feature, magnitude in [(0, 1.0), (1, 2.0), (5, 0.5), (10, 3.0)]:
        out = mech.steer(target, feature=feature, magnitude=magnitude)
        steer_evs = [e for e in mech.events() if e.kind == "mechanizer.steered"]
        last = steer_evs[-1].payload
        delta = _norm_diff(target[0], out[0])
        print(f"  steer feature {feature:2d}  magnitude = {magnitude:.2f}  "
              f"→  ||Δactivation|| = {delta:.4f}  "
              f"(ledger payload reports {last['perturbation_norm']:.4f})")
    print(f"  ledger verifies = {mech.verify_chain()}")


# =====================================================================
# 5. Auto-interpretation + circuit
# =====================================================================


def scenario_5_auto_interpret_and_circuit() -> None:
    print()
    _print_section(
        "[5] Auto-interpretation + feature-feature circuit"
    )
    X = mechanizer_synthetic_features(
        n=100, dim=16, n_true=10, true_l0=3, seed=6,
    )
    mech = Mechanizer(MechanizerConfig(
        algorithm=ALGO_KSVD, n_features=12, target_l0=3,
        max_iter=20, seed=7,
    ))
    mech.fit(X)
    summaries = mech.auto_interpret(X, top_k=3)
    print(f"  {'feat':>4}  {'density':>8}  {'mean act':>10}  "
          f"{'max act':>10}  {'var.expl':>10}  top-3 inputs")
    for s in summaries[:10]:
        print(f"  {s.feature:>4}  {s.activation_density:>8.3f}  "
              f"{s.mean_activation:>10.4f}  {s.max_activation:>10.4f}  "
              f"{s.variance_explained:>10.4f}  {list(s.top_indices)}")
    g = mech.circuit(X, threshold=0.20)
    print(f"\n  circuit: |V| = {g.n_features}  |E| = {g.edge_count}  "
          f"largest component = {g.largest_component}")
    for j in range(min(5, g.n_features)):
        nbrs = g.neighbours(j)[:3]
        if not nbrs:
            continue
        rendered = ", ".join(f"{nb}({w:+.2f})" for nb, w in nbrs)
        print(f"    feature {j:>2}  ↔  {rendered}")


# =====================================================================
# Driver
# =====================================================================


def _norm_diff(a, b) -> float:
    return (sum((ai - bi) ** 2 for ai, bi in zip(a, b))) ** 0.5


def main() -> None:
    scenario_1_sparse_recovery()
    scenario_2_algorithm_comparison()
    scenario_3_activation_patching()
    scenario_4_feature_steering()
    scenario_5_auto_interpret_and_circuit()


if __name__ == "__main__":  # pragma: no cover
    main()
