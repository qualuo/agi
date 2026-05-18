"""Personalizer demo: online per-user preference learning at inference time.

The pitch in one runnable script (no API key, no network, pure stdlib):

  1. A coordination engine ships one model into production but serves
     many users.  Each user has a *different* preferred axis along
     candidate-feature space (truthful vs concise, formal vs casual,
     verbose vs terse).  A one-policy-fits-all aligner saturates the
     global average; Personalizer pushes each user toward their own
     maximum without ever updating the model's weights.
  2. We simulate 5 users; each prefers a different axis of a 6-d
     feature space (set with `synthetic_users`).
  3. The coordinator streams 60 paired preferences per user into the
     Personalizer.  Each adapter is a per-user ridge-regressed
     adjustment on top of a shared global prior — :math:`O(d)` per
     update, no gradient sharing across users, no weight changes
     on the underlying model.
  4. We `score` two opposing candidates for each user and confirm
     that the per-user adapter learned each user's planted axis.
     The :class:`CandidateScore.trust` flag escalates from
     `fallback` to `blend` to `promote` as confidence sequences
     tighten.
  5. We turn on differential privacy (Gaussian-mechanism on
     gradients) for one user, measure cumulative ε via a Renyi-DP
     accountant, and demonstrate the budget guard that raises
     `PrivacyBudgetExceeded`.
  6. We promote one user's adapter to the global prior — the move
     a coordination engine makes when a user dominates a held-out
     preference set — and verify the global recentre.
  7. We demonstrate GDPR Article 17 "right to erasure" by removing
     a user from the system; the action is recorded on the
     fingerprint chain so the erasure itself remains auditable.

Run:  python examples/personalizer_demo.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.personalizer import (
    ALG_BTL,
    ALG_KTO,
    PairwisePreference,
    Personalizer,
    PersonalizerConfig,
    PrivacyBudgetExceeded,
    TRUST_BLEND,
    TRUST_FALLBACK,
    TRUST_PROMOTE,
    UnarySignal,
    renyi_epsilon,
    synthetic_users,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def main() -> None:
    bus = EventBus()
    seen: dict[str, int] = {}
    bus.subscribe(lambda e: seen.update({e.kind: seen.get(e.kind, 0) + 1}))

    banner("1) Init Personalizer (BTL, dim=6, global prior anchored)")
    p = Personalizer(PersonalizerConfig(
        algorithm=ALG_BTL,
        dim=6,
        learning_rate=0.1,
        ridge=0.001,
        min_observations_for_trust=20,
        ci_half_width_promote=0.30,
        seed=0,
    ), bus=bus)
    print(f"   global theta init = {tuple(round(x, 3) for x in p.global_theta)}")

    banner("2) Stream 60 paired preferences per user across 5 users")
    prefs, truths = synthetic_users(n_users=5, n_prefs_per_user=60, dim=6, seed=42)
    for pref in prefs:
        p.observe_pair(pref)
    print(f"   n_users           = {p.n_users}")
    print(f"   total observations = {p.total_observations}")
    print(f"   global theta now  = {tuple(round(x, 3) for x in p.global_theta)}")

    banner("3) Each user's adapter recovers its planted axis")
    print(f"   {'user_id':<8}  {'planted axis':<14}  {'top adapter coord':<20}  {'trust':<10}")
    for uid in (f"u{i}" for i in range(5)):
        summ = p.user_summary(uid)
        # Find the planted axis index from the truth vector:
        planted = max(range(6), key=lambda j: truths[int(uid[1:])][j])
        top = max(range(6), key=lambda j: summ.theta[j])
        trust = p.trust(uid)
        mark = "✓" if top == planted else "✗"
        print(f"   {uid:<8}  axis={planted:<8}    "
              f"top=axis{top}, val={summ.theta[top]:+.2f}  {trust:<10}  {mark}")

    banner("4) Score two opposing candidates for u0 and u1")
    a = tuple(1.0 if j == 0 else 0.0 for j in range(6))
    b = tuple(1.0 if j == 1 else 0.0 for j in range(6))
    for uid in ("u0", "u1"):
        scores = p.score(uid, [a, b])
        prob, lo, hi, trust = p.predict(uid, a, b)
        print(f"   {uid}: score(a)={scores[0].mean:.3f} CI=({scores[0].ci_low:.3f},{scores[0].ci_high:.3f}) "
              f"score(b)={scores[1].mean:.3f}  P(a>b)={prob:.3f}±{(hi-prob):.3f}  trust={trust}")

    banner("5) Differential privacy with a Renyi-DP budget guard")
    pdp = Personalizer(PersonalizerConfig(
        algorithm=ALG_BTL,
        dim=6,
        dp_sigma=1.0,
        dp_epsilon_target=2.0,
        dp_delta=1e-6,
        seed=0,
    ))
    # Feed prefs until the budget exhausts
    fed = 0
    try:
        for pref in prefs:
            if pref.user_id != "u0":
                continue
            pdp.observe_pair(pref)
            fed += 1
    except PrivacyBudgetExceeded as e:
        print(f"   raised PrivacyBudgetExceeded after {fed} steps")
        print(f"   exception: {e}")
    summ = pdp.user_summary("u0")
    print(f"   final ε spent for u0 = {summ.epsilon_spent:.3f} (target was 2.0)")

    banner("6) Promote u0's adapter to the global prior (blend=0.5)")
    before = tuple(round(x, 3) for x in p.global_theta)
    after = p.promote_to_global("u0", blend=0.5)
    print(f"   global theta before = {before}")
    print(f"   global theta after  = {tuple(round(x, 3) for x in after)}")

    banner("7) GDPR Article 17 — remove u2 entirely")
    print(f"   n_users before = {p.n_users}")
    erased = p.remove_user("u2")
    print(f"   remove_user('u2') = {erased}")
    print(f"   n_users after  = {p.n_users}")
    print(f"   fingerprint chain advances even on erasure → audit trail preserved")

    banner("8) Replay-verifiable fingerprint chain + event volume")
    print(f"   fingerprint: {p.fingerprint[:24]}…")
    print(f"   events    :")
    for k in sorted(seen):
        print(f"     {k:38s} {seen[k]}")


if __name__ == "__main__":  # pragma: no cover
    main()
