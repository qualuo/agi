"""Stepwiser demo — verifier-guided test-time compute scaling.

Walks through the full lifecycle of a Process Reward Model as a
runtime primitive:

  1. observe labelled reasoning trajectories
  2. fit + calibrate
  3. score candidate trajectories
  4. best-of-N rerank (the o1/o3 inference-time scaling pattern)
  5. stepwise beam search guided by the PRM
  6. reward shaping for downstream RL
  7. drift detection + certificate

Run from repo root::

    python examples/stepwiser_demo.py
"""
from __future__ import annotations

import random

from agi.stepwiser import (
    AGG_MIN,
    CAL_PLATT,
    MODEL_PRM,
    Stepwiser,
    StepwiserConfig,
    TrajectoryRecord,
)


def make_trajectories(rng: random.Random, n: int = 400) -> list[TrajectoryRecord]:
    """Synthetic reasoning trajectories: 'good' steps with the word
    'verify' indicate progress; 'guess' steps regress."""
    out: list[TrajectoryRecord] = []
    for _ in range(n):
        T = rng.randint(3, 6)
        steps: list[str] = []
        labels: list[int] = []
        for _ in range(T):
            if rng.random() < 0.55:
                steps.append("step: verify and confirm hypothesis")
                labels.append(1)
            else:
                steps.append("step: guess and move on")
                labels.append(0)
        outcome = 1 if all(labels) else 0
        out.append(TrajectoryRecord(
            steps=tuple(steps),
            step_labels=tuple(labels),
            outcome=outcome,
        ))
    return out


def main() -> None:
    rng = random.Random(0)
    pool = make_trajectories(rng)

    sw = Stepwiser(StepwiserConfig(
        model=MODEL_PRM,
        aggregator=AGG_MIN,
        calibrator=CAL_PLATT,
        feature_dim=4096,
        ngram_n=2,
        epochs=24,
        k_beam=3,
        branch_factor=2,
        max_depth=4,
        confidence=0.95,
        rng_seed=42,
        max_ece=0.5,
    ))

    print("=" * 72)
    print(" Stepwiser — process reward modelling demo")
    print("=" * 72)

    print(f"observing {len(pool)} synthetic labelled trajectories ...")
    sw.observe_many(pool)
    report = sw.fit()
    print(
        f"  fitted: train_acc = {report.train_accuracy:.3f}, "
        f"holdout_acc = {report.holdout_accuracy:.3f}, "
        f"ECE = {report.holdout_ece:.3f}, calibrated = {report.calibrated}"
    )

    # 1. Score
    good = ["step: verify and confirm hypothesis"] * 3
    bad = ["step: guess and move on"] * 3
    mixed = [
        "step: verify and confirm hypothesis",
        "step: guess and move on",
        "step: verify and confirm hypothesis",
    ]
    print()
    print("trajectory scoring (aggregator = min, calibrated probs):")
    for name, traj in [("good", good), ("bad", bad), ("mixed", mixed)]:
        s = sw.score(traj)
        per_step = ", ".join(f"{ss.calibrated_prob:.2f}" for ss in s.per_step)
        print(f"  {name:>6}: aggregated = {s.aggregated:.3f}  per-step=[{per_step}]")

    # 2. Best-of-N
    candidates = [good, bad, mixed,
                  ["step: verify and confirm hypothesis",
                   "step: verify and confirm hypothesis",
                   "step: guess and move on"],
                  ["step: guess and move on",
                   "step: verify and confirm hypothesis"]]
    sel = sw.best_of_n(candidates)
    print()
    print(f"best-of-N (N={len(candidates)}):")
    print(f"  chosen index = {sel.chosen_index}  "
          f"aggregated = {sel.chosen_score.aggregated:.3f}")
    print(f"  selection gap = {sel.selection_gap:.3f}  "
          f"LCB(gap; δ=0.05) = {sel.selection_gap_lcb:.3f}")
    print(f"  ranking = {[f'{i}:{s:.2f}' for i, s in sel.ranked]}")

    # 3. Beam search
    options = [
        "step: verify and confirm hypothesis",
        "step: guess and move on",
    ]
    bs = sw.beam_search(
        ("step: verify and confirm hypothesis",),
        expand=lambda beam: options,
        terminal=lambda beam: len(beam) >= 4,
    )
    print()
    print(f"beam search (depth = {bs.depth}, expansions = {bs.expansions}):")
    for i, (beam, sc) in enumerate(zip(bs.beams, bs.scores)):
        marker = "*" if i == 0 else " "
        n_good = sum(1 for s in beam if "verify" in s)
        n_bad = sum(1 for s in beam if "guess" in s)
        print(f"  {marker} score={sc:.3f}  good={n_good} bad={n_bad}")

    # 4. Shaping rewards for downstream RL
    shaped = sw.shape(mixed)
    print()
    print(f"potential-based shaping (γ={sw.config.discount}):")
    for i, r in enumerate(shaped):
        print(f"  t={i}  r̃_t = {r:+.3f}")

    # 5. Drift detection
    # In-distribution stream matches training trajectory length distribution.
    in_dist = [t.steps for t in make_trajectories(random.Random(7), n=40)]
    out_dist = [["x" * 200] * 5 for _ in range(40)]
    d_in = sw.drift(in_dist)
    d_out = sw.drift(out_dist)
    print()
    print("drift detection (KS two-sample, α=0.05):")
    print(f"  in-distribution stream: KS = {d_in.ks_statistic:.3f}  "
          f"threshold = {d_in.threshold:.3f}  rejected = {d_in.rejected}")
    print(f"  out-of-dist stream:     KS = {d_out.ks_statistic:.3f}  "
          f"threshold = {d_out.threshold:.3f}  rejected = {d_out.rejected}")

    # 6. Certificate
    cert = sw.certify()
    print()
    print("certificate (replay-verifiable):")
    print(f"  confidence            = {cert.confidence}")
    print(f"  holdout accuracy      = {cert.holdout_accuracy:.3f}")
    print(f"  accuracy LCB          = {cert.accuracy_lcb:.3f}")
    print(f"  Hoeffding half-width  = {cert.hoeffding_half_width:.3f}")
    print(f"  Bernstein half-width  = {cert.bernstein_half_width:.3f}")
    print(f"  selection gap         = {cert.last_selection_gap:.3f}")
    print(f"  selection gap LCB     = {cert.last_selection_gap_lcb:.3f}")
    print(f"  chain head            = {cert.chain_head[:24]}...")
    print(f"  fingerprint           = {cert.fingerprint_hash[:24]}...")

    # 7. Top features (skill mining)
    print()
    print("top-8 features by |weight| (interpretable PRM hooks):")
    for fid, w in sw.top_features(k=8):
        print(f"  feature {fid:>5}  weight = {w:+.3f}")

    print()
    print("done.")


if __name__ == "__main__":
    main()
