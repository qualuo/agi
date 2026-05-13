"""policy_lab_demo.py — backtest three routing policies on synthetic traffic.

Investor framing in one breath: we run last week's traffic through three
candidate routing policies *in silico* and rank them by expected reward,
with calibrated confidence intervals, without spending a single real
dollar. The lab uses doubly-robust off-policy evaluation — the workhorse
estimator of industrial contextual bandits.

What this demo shows:

  1.  Synthesize 5000 logged (context, action, propensity, reward) events.
  2.  Compare a context-blind 'cheap-only' router, a 'smart-only' router,
      and a 'context-aware' router that picks based on task difficulty.
  3.  Print Pareto frontier across (expected_reward, expected_cost).
  4.  Make a ship/kill recommendation against the current production
      baseline.

Run:

    python examples/policy_lab_demo.py
"""
from __future__ import annotations

import random

from agi.policy_lab import (
    LinearRewardModel,
    LoggedEvent,
    PolicyCandidate,
    PolicyLab,
)


def main() -> None:
    rng = random.Random(2026)

    # ------------------------------------------------------------------
    # 1. Synthesize logged production traffic.
    # ------------------------------------------------------------------
    # Logging policy: random 50/50 between two models. Reward depends on
    # task difficulty (a context feature):
    #   - cheap model: reward = 1 - difficulty (great on easy, terrible on hard)
    #   - smart model: reward ~ 0.9 (steady but mediocre)
    lab = PolicyLab(reward_model=LinearRewardModel(ridge=0.1))
    for _ in range(5000):
        diff = rng.random()
        if rng.random() < 0.5:
            a, prop = "smart", 0.5
            r = 0.9 + 0.05 * diff + rng.gauss(0, 0.05)
        else:
            a, prop = "cheap", 0.5
            r = (1.0 - diff) + rng.gauss(0, 0.05)
        lab.record(
            LoggedEvent(
                context={"difficulty": diff},
                action=a,
                propensity=prop,
                reward=r,
            )
        )

    print(f"Logged {len(lab)} events from production.")
    print()

    # ------------------------------------------------------------------
    # 2. Define three candidate routing policies.
    # ------------------------------------------------------------------
    candidates = [
        PolicyCandidate("v0_random_50_50", lambda c: {"cheap": 0.5, "smart": 0.5}),
        PolicyCandidate("v1_cheap_only",   lambda c: {"cheap": 1.0}),
        PolicyCandidate("v2_smart_only",   lambda c: {"smart": 1.0}),
        PolicyCandidate(
            "v3_context_aware",
            lambda c: (
                {"cheap": 1.0}
                if c.get("difficulty", 0.5) < 0.5
                else {"smart": 1.0}
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # 3. Rank with cost-aware Pareto frontier.
    # ------------------------------------------------------------------
    # Suppose the cheap model is 100x cheaper than the smart one.
    cost = {"cheap": 0.001, "smart": 0.10}
    report = lab.recommend(candidates, method="dr", cost_per_action=cost)
    print(report.summary)
    print()

    # ------------------------------------------------------------------
    # 4. Compare the new context-aware router to the production baseline.
    # ------------------------------------------------------------------
    cmp = lab.compare(
        target=candidates[3],   # v3_context_aware
        baseline=candidates[0], # v0_random_50_50
        method="dr",
        confidence=0.95,
    )
    print("=== A/B comparison ===")
    print(f"  target     {cmp.target.policy_name} -> {cmp.target.value:+.4f}")
    print(f"  baseline   {cmp.baseline.policy_name} -> {cmp.baseline.value:+.4f}")
    print(
        f"  lift       {cmp.lift:+.4f}  "
        f"95% CI [{cmp.lift_ci_low:+.4f}, {cmp.lift_ci_high:+.4f}]"
    )
    print(f"  p_better   {cmp.p_better:.4f}")
    print(f"  decision   {cmp.recommend.upper()}")
    print(f"  rationale  {cmp.rationale}")


if __name__ == "__main__":
    main()
