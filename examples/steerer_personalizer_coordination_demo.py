"""Steerer + Personalizer coordination demo: per-user behavioural adaptation.

The runtime-engine story end-to-end in one runnable script:

  1. The coordination engine ships one frozen model into production.
     Two layers of inference-time adaptation sit between the model
     and each user:
        (a) a `Steerer` per behavioural axis (truthfulness, formality,
            refusal-direction) — fits a unit steering vector on
            contrastive activation pairs and certifies the realised
            outcome lift.
        (b) a `Personalizer` per deployment — maintains a per-user
            ridge-regularised adapter on top of a shared global
            preference prior; updates online from each user's
            preference signals.
  2. For each incoming request the coordinator:
        - Asks Personalizer for a per-user candidate ranking.
        - Decides whether to trust the personalised score (CI half-
          width below tolerance and observation count over threshold)
          or fall back to the global policy.
        - Picks the highest-trust candidate; applies the certified
          Steerer direction (if the user's profile calls for more or
          less of a behavioural axis) before generating.
  3. After the user responds (thumbs / accept / explicit pair vote)
     the coordinator updates Personalizer and Steerer in lockstep —
     a Personalizer.observe_pair() + a Steerer.observe_outcome()
     under the user's chosen coefficient.
  4. The audit trail is the joined fingerprint chain of both
     primitives plus the EventBus's session events — replayable from
     disk.

Run:  python examples/steerer_personalizer_coordination_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.manifest import default_manifest
from agi.personalizer import (
    ALG_BTL,
    PairwisePreference,
    Personalizer,
    PersonalizerConfig,
    TRUST_BLEND,
    TRUST_PROMOTE,
)
from agi.steerer import (
    ALG_CAA,
    ContrastivePair,
    DoseOutcome,
    Steerer,
    SteererConfig,
    apply_addition,
    synthetic_pairs,
)


def banner(title: str) -> None:
    print()
    print("=" * (len(title) + 4))
    print(f"  {title}")
    print("=" * (len(title) + 4))


def main() -> None:
    bus = EventBus()
    seen: dict[str, int] = {}
    bus.subscribe(lambda e: seen.update({e.kind: seen.get(e.kind, 0) + 1}))

    banner("1) The coordination engine consults the manifest")
    m = default_manifest()
    s_spec = m.lookup("steerer")
    p_spec = m.lookup("personalizer")
    print(f"   steerer        {s_spec.kind:<14}  cert={s_spec.certificate:<10}  "
          f"composes_with includes Personalizer? "
          f"{'(via aligner→personalizer)' if 'aligner' in s_spec.composes_with else '-'}")
    print(f"   personalizer   {p_spec.kind:<14}  cert={p_spec.certificate:<10}  "
          f"composes_with steerer? {'yes' if 'steerer' in p_spec.composes_with else 'no'}")

    banner("2) Steerer: fit a truthfulness-direction at resid.16")
    steerer = Steerer(SteererConfig(
        algorithm=ALG_CAA, dim=6, alpha=0.05, target_layer="resid.16", seed=0,
    ), bus=bus)
    # The coordinator has paired activations from (truthful, untruthful)
    # completions of 60 contrastive prompts:
    for pair in synthetic_pairs(n=60, dim=6, axis=0, snr=2.0, noise=0.4, seed=1):
        steerer.observe_pair(pair)
    fit = steerer.fit()
    print(f"   fitted direction: auroc={fit.separability_auroc:.3f}  "
          f"d={fit.cohen_d:.2f}  norm={fit.norm:.3f}")
    # Simulate dose-response outcomes (high coefficient → more truthful):
    rng = random.Random(11)
    for coef in (-1.0, 0.0, 1.0, 2.0):
        for j in range(60):
            rate = 1.0 / (1.0 + math.exp(-coef * 1.8))
            steerer.observe_outcome(DoseOutcome(
                task_id=f"q{j}-{coef}", coefficient=coef, outcome=rng.random() < rate,
            ))
    cert = steerer.certify(delta=0.10, alpha=0.05)
    print(f"   certified: verdict={cert.verdict}  rec={cert.recommendation}  "
          f"coef={cert.recommended_coefficient}  lift={cert.outcome_lift:+.3f}")

    banner("3) Personalizer: 4 users with different truthfulness preferences")
    pz = Personalizer(PersonalizerConfig(
        algorithm=ALG_BTL,
        dim=6,
        learning_rate=0.1,
        ridge=0.001,
        min_observations_for_trust=20,
        ci_half_width_promote=0.30,
        seed=0,
    ), bus=bus)
    # Each user has a preferred axis; user u_strict prefers more truthful
    # responses (axis 0); user u_casual prefers a different style (axis 5);
    # u_balanced is indifferent; u_blunt prefers axis 2.
    user_axes = {"u_strict": 0, "u_balanced": 1, "u_blunt": 2, "u_casual": 5}
    rng2 = random.Random(7)
    for uid, axis in user_axes.items():
        for _ in range(40):
            v1 = tuple(rng2.gauss(0, 1) for _ in range(6))
            v2 = tuple(rng2.gauss(0, 1) for _ in range(6))
            if v1[axis] > v2[axis]:
                pz.observe_pair(PairwisePreference(uid, v1, v2))
            else:
                pz.observe_pair(PairwisePreference(uid, v2, v1))
        summ = pz.user_summary(uid)
        top = max(range(6), key=lambda j: summ.theta[j])
        print(f"   {uid:<12}  planted=axis{axis}  recovered=axis{top}  "
              f"trust={pz.trust(uid)}")

    banner("4) Live routing: each user gets candidates scored *for them*")
    # Two candidate response embeddings — say a "more truthful" response
    # has high axis-0 weight, a "more casual" one has high axis-5:
    truthful = (1.0, 0.2, 0.0, 0.0, 0.0, 0.0)
    casual = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    print(f"   user            truthful_score   casual_score    prefers     trust")
    print(f"   -------------   --------------   ------------    --------    --------")
    for uid in user_axes:
        s_truth, s_casual = pz.score(uid, [truthful, casual])
        prefers = "truthful" if s_truth.mean > s_casual.mean else "casual"
        print(f"   {uid:<13}   {s_truth.mean:>6.3f}({s_truth.trust[:5]})   "
              f"{s_casual.mean:>6.3f}        {prefers:<10}  {s_truth.trust}")

    banner("5) For u_strict: apply Steerer's truth coefficient at inference")
    # u_strict prefers truthful — turn the Steerer dial *up*.
    # Pretend the model just emitted an activation at resid.16:
    a = tuple(random.Random(99).gauss(0, 0.5) for _ in range(6))
    coef = cert.recommended_coefficient or 0.0
    a_steered = apply_addition(a, fit.direction, coef)
    proj_before = sum(a[i] * fit.direction[i] for i in range(6))
    proj_after = sum(a_steered[i] * fit.direction[i] for i in range(6))
    print(f"   activation projection along truth-direction:")
    print(f"     before steering (control) = {proj_before:+.3f}")
    print(f"     after  steering @ coef={coef:+.1f} = {proj_after:+.3f}")
    print(f"   coordinator now generates the response under the steered residual.")

    banner("6) The user signals back — both primitives update online")
    # Suppose u_strict accepts the steered response (positive pairwise vs an
    # un-steered alternative).  We record that as a preference *and* as a
    # successful outcome for the steering coefficient.
    pz.observe_pair(PairwisePreference(
        user_id="u_strict",
        features_winner=truthful,
        features_loser=casual,
    ))
    steerer.observe_outcome(DoseOutcome(
        task_id="live-1", coefficient=coef, outcome=True,
    ))
    print(f"   u_strict   personalizer.observe_pair    → n_obs={pz.user_summary('u_strict').n_observations}")
    print(f"   steerer    observe_outcome @ coef={coef:+.1f}  → n_outcomes={steerer.n_outcomes}")

    banner("7) Audit trail: joined fingerprint chains + event volume")
    print(f"   steerer       fingerprint: {steerer.fingerprint[:24]}…")
    print(f"   personalizer  fingerprint: {pz.fingerprint[:24]}…")
    print(f"   bus events    (top 8):")
    for k in sorted(seen, key=lambda x: -seen[x])[:8]:
        print(f"     {k:<38s} {seen[k]}")

    banner("8) GDPR Article 17: erase u_casual entirely from Personalizer")
    erased = pz.remove_user("u_casual")
    print(f"   remove_user('u_casual') = {erased}")
    print(f"   n_users after erase    = {pz.n_users}")
    print(f"   the erasure event itself is recorded on the fingerprint chain,")
    print(f"   so an auditor can prove the erasure happened — not just that")
    print(f"   the user is gone.")


if __name__ == "__main__":  # pragma: no cover
    main()
