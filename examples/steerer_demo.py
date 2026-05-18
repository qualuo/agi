"""Steerer demo: extract a behavioural-axis direction from contrastive activation pairs and certify the realised outcome lift at inference time.

The pitch in one runnable script (no API key, no network, pure stdlib):

  1. A coordination engine ships a frozen model in production but
     wants to push it along a behavioural axis — *more truthful*,
     *more refusing*, *less sycophantic* — without retraining.
  2. For 80 contrastive prompt pairs the engine collects two
     activation vectors at a chosen residual-stream layer: one from
     the *positive* class (truthful answer) and one from the
     *negative* (untruthful answer).  Steerer fits a unit steering
     vector from those pairs.
  3. The engine then runs trials at a grid of steering coefficients
     (- 1.0, 0.0, +1.0, +2.0 of the unit direction) and records
     whether each steered answer satisfied the policy.
  4. ``Steerer.certify`` runs the engaged tests — in-training
     separability AUROC (Mason-Graham CI), dose-response
     monotonicity (Spearman), anytime-valid effect-size lift
     (Waudby-Smith-Ramdas hedged-capital betting e-process) — and
     emits a verdict (PASS / WARN / FAIL / INCONCLUSIVE) plus a
     recommendation (PROMOTE / HOLD / REJECT / FLIP) and the
     coefficient the coordinator should apply at inference.
  5. The four estimators (CAA, LAT, Fisher-LDA, logistic-probe) are
     fit on the same pairs and compared — investors want to see that
     the certificate is robust across estimators.

Run:  python examples/steerer_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.steerer import (
    ALG_CAA,
    ALG_LAT,
    ALG_LDA,
    ALG_PROBE,
    ContrastivePair,
    DoseOutcome,
    Steerer,
    SteererConfig,
    apply_addition,
    apply_orthogonal_ablation,
    compare_steerers,
    synthetic_pairs,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def simulate_steered_outcome(
    coefficient: float,
    direction_strength: float,
    seed: int,
) -> bool:
    """Toy model of the post-intervention behaviour: the realised
    outcome rate is a logistic function of (coefficient · direction
    strength).  Captures the qualitative pattern observed in CAA on
    real models — strong direction + positive coefficient → high
    truthful-answer rate, negative coefficient → low rate."""
    rng = random.Random(seed)
    rate = 1.0 / (1.0 + math.exp(-coefficient * direction_strength))
    return rng.random() < rate


def main() -> None:
    rng = random.Random(0)
    bus = EventBus()
    events_count: dict[str, int] = {}

    def count(e):
        events_count[e.kind] = events_count.get(e.kind, 0) + 1

    bus.subscribe(count)

    banner("1) Coordination engine collects contrastive pairs at layer resid.16")
    pairs = synthetic_pairs(
        n=80,
        dim=8,
        axis=0,
        snr=2.0,
        noise=0.4,
        seed=1,
    )
    # Hold a paraphrase-split of 20 pairs out for the leakage test:
    paraphrase = [
        ContrastivePair(
            task_id=f"p{i}",
            positive=pairs[i].positive,
            negative=pairs[i].negative,
            split="paraphrase",
        )
        for i in range(60, 80)
    ]
    train = pairs[:60]
    print(f"   {len(train)} train pairs + {len(paraphrase)} paraphrase pairs at dim=8.")

    banner("2) Fit four estimators on the same pairs")
    fits: dict[str, object] = {}
    for algo in (ALG_CAA, ALG_LAT, ALG_LDA, ALG_PROBE):
        s = Steerer(SteererConfig(algorithm=algo, dim=8, alpha=0.05,
                                  target_layer="resid.16", seed=0))
        for p in train:
            s.observe_pair(p)
        for p in paraphrase:
            s.observe_pair(p)
        fits[algo] = s
        fit = s.fit()
        print(f"   {algo:5s}  AUROC={fit.separability_auroc:.3f}  "
              f"|d|={abs(fit.direction[0]):.3f}  cohen_d={fit.cohen_d:.2f}  "
              f"leak={fit.leakage_auroc:.3f}")

    banner("3) Coordination engine runs steered trials at a coefficient grid")
    # Pick the CAA fit as the production direction (typically the
    # winner on small samples; LDA wins under high-dim + many samples).
    s = fits[ALG_CAA]
    direction = s.last_fit.direction
    # Synthesise dose-response trials.  The *signal* magnitude is the
    # in-training Cohen's-d divided by ~6 (a generous mapping from
    # standardised effect to outcome probability).
    direction_strength = max(0.5, s.last_fit.cohen_d / 6.0)
    rng2 = random.Random(11)
    n_per_coef = 80
    grid = (-1.0, 0.0, 1.0, 2.0)
    print(f"   strength~{direction_strength:.2f}, {n_per_coef} trials × {len(grid)} coefficients")
    for coef in grid:
        for j in range(n_per_coef):
            outcome = simulate_steered_outcome(coef, direction_strength, rng2.randint(0, 1 << 30))
            s.observe_outcome(DoseOutcome(
                task_id=f"q{j}-{coef}", coefficient=coef, outcome=outcome,
            ))
    recommended = s.recommend_coefficient()
    print(f"   recommended coefficient = {recommended}")

    banner("4) Certify the behavioural shift (anytime-valid)")
    cert = s.certify(delta=0.10, alpha=0.05)
    print(f"   verdict         : {cert.verdict}")
    print(f"   recommendation  : {cert.recommendation}")
    print(f"   recommended coef: {cert.recommended_coefficient}")
    print(f"   outcome lift    : {cert.outcome_lift:+.3f}  "
          f"CI = ({cert.outcome_lift_ci[0]:+.3f}, {cert.outcome_lift_ci[1]:+.3f})")
    print(f"   tests           :")
    for t in cert.tests:
        flag = "PASS" if t.passed else "FAIL"
        stat = f"stat={t.statistic:+.3f}"
        thr = f"thr={t.threshold:+.3f}"
        n = f"n={t.n}"
        p = f"p={t.p_value:.4f}" if t.p_value is not None else "p=  -  "
        print(f"     {t.name:14s} {flag}  {stat}  {thr}  {p}  {n}")

    banner("5) Apply the certified steering vector to an unseen activation")
    # Pretend the coordinator hooks the residual stream at resid.16
    # and receives an activation.  Demonstrate addition, ablation,
    # and a +2σ projection.
    a = tuple(rng.gauss(0.0, 0.5) for _ in range(8))
    a_steered = apply_addition(a, direction, cert.recommended_coefficient or 0.0)
    a_ablated = apply_orthogonal_ablation(a, direction)
    proj_before = sum(a[i] * direction[i] for i in range(8))
    proj_after = sum(a_steered[i] * direction[i] for i in range(8))
    proj_ablated = sum(a_ablated[i] * direction[i] for i in range(8))
    print(f"   activation projection along direction:")
    print(f"     before  = {proj_before:+.3f}")
    print(f"     steered = {proj_after:+.3f}  (shifted by "
          f"{proj_after - proj_before:+.3f})")
    print(f"     ablated = {proj_ablated:+.3f}  (component removed)")

    banner("6) Cross-estimator robustness")
    base = fits[ALG_CAA]
    for algo in (ALG_LAT, ALG_LDA, ALG_PROBE):
        cmp = compare_steerers(base, fits[algo])
        print(f"   {ALG_CAA} vs {algo}: cos={cmp['cosine_similarity']:+.3f}  "
              f"AUROC=({cmp['auroc_a']:.3f}, {cmp['auroc_b']:.3f})  "
              f"d=({cmp['cohen_d_a']:.2f}, {cmp['cohen_d_b']:.2f})")

    banner("7) Replay-verifiable fingerprint chain + event volume")
    print(f"   fingerprint: {s.fingerprint[:24]}…")
    print(f"   n_pairs    : {s.n_pairs}")
    print(f"   n_outcomes : {s.n_outcomes}")
    print(f"   events     :")
    for k in sorted(events_count):
        print(f"     {k:34s} {events_count[k]}")


if __name__ == "__main__":  # pragma: no cover
    main()
