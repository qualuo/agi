"""Attributor demo — data attribution / influence functions as a runtime primitive.

Four scenarios, each demonstrating a different facet of the primitive:

  1. Outlier diagnosis — Cook's distance, leverage, PRESS residuals on a
     dataset with one obvious outlier.
  2. Counterfactual prediction — "what would the model say if we'd
     never seen point i?"
  3. Decision-flip certificate — the smallest set of training points
     whose removal flips a discrete decision, with an e-value
     attached.
  4. TracIn trajectory attribution — given a learning trajectory,
     attribute the final answer to the training data that drove it.

Run with::

    python examples/attributor_demo.py
"""
from __future__ import annotations

import math
import random

from agi.attributor import (
    Attributor,
    EXACT_LOO,
    INFLUENCE_FUNCTION,
    LINEAR,
    LOGISTIC,
    QUERY_LOSS,
    QUERY_PREDICTION,
)


# =====================================================================
# 1. Outlier diagnosis
# =====================================================================


def outlier_diagnosis() -> None:
    print("=" * 72)
    print("[1] Outlier diagnosis — leverage, Cook's distance, PRESS residuals")
    print("=" * 72)
    # y = 1 + 0.5x with a single severe outlier at x=10.
    X = [[1.0, float(x)] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    fit = a.fit("price_model", LINEAR, X=X, y=y)
    print(f"  full-data fit: β = ({fit.theta[0]:+.3f}, {fit.theta[1]:+.3f})  "
          f"RSS = {fit.residual_sum_squares:.2f}")
    print(f"  (clean signal is y = 1 + 0.5x — the outlier hijacks the fit.)\n")

    diag = a.linear_diagnostics("price_model")
    print(f"  {'i':>3} {'y':>8} {'leverage':>10} {'press':>10} "
          f"{'studentized':>12} {'cooks_D':>10}")
    for i in range(len(X)):
        print(
            f"  {i:>3} {y[i]:>8.2f} "
            f"{diag.leverage[i]:>10.4f} "
            f"{diag.press_residual[i]:>+10.3f} "
            f"{diag.studentized[i]:>+12.3f} "
            f"{diag.cooks_distance[i]:>10.4f}"
        )
    threshold = 4.0 / len(X)
    print(f"\n  Conventional 4/n flag for Cook's D = {threshold:.3f}")
    for i, d in enumerate(diag.cooks_distance):
        if d > threshold:
            print(f"  → observation #{i} flagged as high-influence "
                  f"(Cook's D = {d:.3f})")


# =====================================================================
# 2. Counterfactual prediction
# =====================================================================


def counterfactual_prediction() -> None:
    print("\n" + "=" * 72)
    print("[2] Counterfactual prediction — top-K removed and refit")
    print("=" * 72)
    X = [[1.0, float(x)] for x in (1, 2, 3, 4, 5, 10)]
    y = [1.5, 2.0, 2.5, 3.0, 3.5, 100.0]
    a = Attributor()
    a.fit("price_model", LINEAR, X=X, y=y)

    test_x = [1.0, 6.0]
    inf = a.influence("price_model", query=QUERY_PREDICTION, test_point=test_x)
    print(f"  baseline prediction at x* = {test_x}: ŷ = "
          f"{inf.q_baseline:+.3f}")
    print("\n  Per-point influence on the prediction (IF approximation):")
    for i, v in enumerate(inf.per_point):
        bar = ">" * max(int(v * 0.5), 0) if v > 0 else "<" * max(int(-v * 0.5), 0)
        print(f"    #{i}: Δŷ ≈ {v:+.4f}   {bar}")

    print("\n  Top-3 most influential training points (by |IF|):")
    for i, v in inf.most_influential(3):
        print(f"    #{i}:  |Δŷ| = {abs(v):.4f}  (x = {X[i][1]:.1f}, y = {y[i]:.1f})")

    cf1 = a.counterfactual_refit(
        "price_model", remove=inf.top_indices(1),
        query=QUERY_PREDICTION, test_point=test_x,
    )
    cf3 = a.counterfactual_refit(
        "price_model", remove=inf.top_indices(3),
        query=QUERY_PREDICTION, test_point=test_x,
    )
    print(f"\n  Remove top-1 ({inf.top_indices(1)}): ŷ = {cf1.q_counterfactual:+.3f}  "
          f"(Δ = {cf1.delta_q:+.3f}); θ_cf = {[round(v, 3) for v in cf1.theta_counterfactual]}")
    print(f"  Remove top-3 ({inf.top_indices(3)}): ŷ = {cf3.q_counterfactual:+.3f}  "
          f"(Δ = {cf3.delta_q:+.3f}); θ_cf = {[round(v, 3) for v in cf3.theta_counterfactual]}")


# =====================================================================
# 3. Decision-flip certificate
# =====================================================================


def decision_flip_certificate() -> None:
    print("\n" + "=" * 72)
    print("[3] Decision-flip certificate — how many points until ACCEPT → REJECT?")
    print("=" * 72)
    rnd = random.Random(0xCAFE)
    # Synthetic credit-score regression.  True relationship: a 1-unit
    # increase in feature_1 should raise the prediction by ~0.5.
    X = []
    y = []
    for i in range(20):
        f1 = rnd.gauss(0, 1)
        f2 = rnd.gauss(0, 1)
        X.append([1.0, f1, f2])
        y.append(0.5 * f1 - 0.3 * f2 + rnd.gauss(0, 0.1))
    # Inject one "anomalous" training example that pushes the slope up.
    X.append([1.0, 0.8, 0.0])
    y.append(20.0)
    a = Attributor()
    a.fit("credit", LINEAR, X=X, y=y)
    print(f"  fit on 21 points: β = {[round(v, 3) for v in a._state('credit').theta]}")

    # Decision: accept iff the prediction at the applicant's features
    # exceeds 1.0.  The anomalous training point pushed β_1 up, which
    # tips the borderline applicant from REJECT to ACCEPT.
    test_features = [1.0, 0.3, 0.0]
    pred_full = sum(test_features[j] * a._state("credit").theta[j] for j in range(3))
    decision_full = "ACCEPT" if pred_full > 1.0 else "REJECT"
    print(f"  applicant features: {test_features[1:]} → "
          f"baseline ŷ = {pred_full:+.3f} → decision = {decision_full!r}")

    flip = a.decision_flip(
        "credit",
        decision_fn=lambda theta: (
            "ACCEPT"
            if sum(test_features[j] * theta[j] for j in range(3)) > 1.0
            else "REJECT"
        ),
        budget_k=3,
        query=QUERY_PREDICTION, test_point=test_features,
    )
    if flip.flipped:
        print(f"\n  → Removing {flip.minimal_set} flips the decision "
              f"to {flip.decision_after!r}")
        print(f"     e-value (counterfactual : full LR) = {flip.e_value:.3g}")
        print(f"     log₁₀ Bayes factor                  = {flip.log10_bayes_factor:+.3f}")
        # The most-influential points should be highlighted.
        print(f"     fingerprint: {flip.fingerprint[:16]}…")
    else:
        print(f"\n  No flipping set ≤ {flip.minimal_set} found inside budget.")


# =====================================================================
# 4. TracIn trajectory attribution
# =====================================================================


def tracin_attribution() -> None:
    print("\n" + "=" * 72)
    print("[4] TracIn ideal — attribute a prediction to a training trajectory")
    print("=" * 72)
    # Simulated logistic-regression training trajectory.
    rnd = random.Random(0xBEEF)
    X = []
    y = []
    for _ in range(15):
        f1 = rnd.gauss(0, 1)
        X.append([1.0, f1])
        y.append(1.0 if f1 > 0 else 0.0)
    a = Attributor()
    a.fit("clf", LOGISTIC, X=X, y=y, ridge=1e-3)
    # Build a trajectory of decreasing-noise snapshots — pretend GD
    # walked from a zero init toward the converged β̂.
    final_beta = list(a._state("clf").theta)
    trajectory = []
    for k, alpha in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        theta_t = [alpha * b for b in final_beta]
        trajectory.append((theta_t, 0.1))
    # Attribute the prediction at a held-out test x*.
    test_x = [1.0, 0.7]
    rep = a.tracin_ideal(
        "clf",
        trajectory=trajectory,
        grad_query=lambda theta: list(test_x),  # ∇(θ·x) = x
    )
    print(f"  trajectory length: {rep.n_checkpoints}")
    print(f"  per-training-point cumulative TracIn at x* = {test_x[1:]}:")
    for i, v in enumerate(rep.per_point):
        marker = "★" if abs(v) > max(abs(x) for x in rep.per_point) * 0.5 else " "
        print(f"    {marker} #{i}: TracIn = {v:+.4f}   (x = {X[i][1]:+.3f}, y = {int(y[i])})")
    top = rep.most_influential(3)
    print(f"\n  Top-3 most-influential training points: {[(i, round(v, 4)) for i, v in top]}")
    print(f"  fingerprint: {rep.fingerprint[:16]}…")


# =====================================================================
# 5. Replay attestation
# =====================================================================


def replay_attestation() -> None:
    print("\n" + "=" * 72)
    print("[5] Replay attestation — every step fingerprinted")
    print("=" * 72)
    a = Attributor()
    a.fit("m", LINEAR, X=[[1, 1], [1, 2], [1, 3]], y=[1.0, 2.0, 3.0])
    a.influence("m", query=QUERY_PREDICTION, test_point=[1.0, 4.0])
    a.cooks_distance("m")
    a.counterfactual_refit("m", remove=[0],
                           query=QUERY_PREDICTION, test_point=[1.0, 4.0])
    print(f"  events recorded: {len(a.events())}")
    for ev in a.events():
        print(f"    seq {ev.seq}: {ev.kind}  (hash = {ev.this_hash[:12]}…)")
    print(f"\n  chain valid?  {a.verify_chain()}")
    print(f"  fingerprint:  {a.fingerprint[:16]}…")


if __name__ == "__main__":
    outlier_diagnosis()
    counterfactual_prediction()
    decision_flip_certificate()
    tracin_attribution()
    replay_attestation()
    print("\n" + "=" * 72)
    print("All Attributor demos completed successfully.")
    print("=" * 72)
