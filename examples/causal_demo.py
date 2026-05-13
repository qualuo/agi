"""CausalLab — heterogeneous treatment effects as a runtime primitive.

Run this from the repo root:

    python examples/causal_demo.py

Scenario
--------

A coordination engine is routing tasks between two models:
  - "cheap"  — fast, cheap, weaker on hard tasks
  - "strong" — slow, expensive, robust everywhere

The current policy fires 50/50, randomised. After a week the operator
has 800 logged decisions. They want to know:

  1. The *average* lift of always-cheap vs always-strong (PolicyLab).
  2. The *per-task* lift — for which tasks does cheap actually win?
     (CausalLab — the new primitive.)
  3. Whether there's any heterogeneity at all worth personalising on.
  4. An interpretable rule a coordinator can ship without an ML stack.

The ground truth (hidden from the lab): cheap wins on easy tasks,
strong wins on hard ones, with task_difficulty ∈ [0, 1] driving the
gap. CATE = -0.6 * (task_difficulty - 0.5).
"""
from __future__ import annotations

import random

from agi.causal import CausalLab, LEARNER_DR, LEARNER_X
from agi.policy_lab import LoggedEvent, PolicyLab, LinearRewardModel


def synth(n: int = 800, seed: int = 17) -> list[LoggedEvent]:
    rng = random.Random(seed)
    events: list[LoggedEvent] = []
    for _ in range(n):
        difficulty = rng.uniform(0.0, 1.0)
        # Logging policy: 50/50 between cheap and strong.
        action = "cheap" if rng.random() < 0.5 else "strong"
        # Realised reward: strong baseline + heterogeneous gap for cheap.
        base = 0.8 - 0.4 * difficulty   # both arms decay with difficulty
        gap = -0.6 * (difficulty - 0.5)  # CATE: cheap wins easy, loses hard
        r = base + (gap if action == "cheap" else 0.0)
        r += rng.gauss(0.0, 0.05)
        events.append(
            LoggedEvent(
                context={"task_difficulty": difficulty},
                action=action,
                propensity=0.5,
                reward=r,
            )
        )
    return events


def main() -> None:
    events = synth(n=800)

    # ----- 1. PolicyLab: the *average* picture. -----
    plab = PolicyLab(reward_model=LinearRewardModel(ridge=0.5))
    plab.record_batch(events)
    avg = plab.evaluate(lambda ctx: {"cheap": 1.0}, name="always-cheap", method="dr")
    print(f"[PolicyLab] always-cheap   value={avg.value:+.4f} ± {avg.se:.4f}  "
          f"CI=[{avg.ci_low:+.4f}, {avg.ci_high:+.4f}]")
    avg_s = plab.evaluate(lambda ctx: {"strong": 1.0}, name="always-strong", method="dr")
    print(f"[PolicyLab] always-strong  value={avg_s.value:+.4f} ± {avg_s.se:.4f}  "
          f"CI=[{avg_s.ci_low:+.4f}, {avg_s.ci_high:+.4f}]")
    print()
    print("Verdict at the *average* level: the arms are roughly tied. PolicyLab")
    print("would call this a coin-flip. But that average hides everything.")
    print()

    # ----- 2. CausalLab: per-context counterfactual lift. -----
    clab = CausalLab(treatment="cheap", control="strong")
    clab.attach_to_policy_lab(plab)

    print("[CausalLab] per-task CATE (cheap − strong) with 95% CI, DR-learner:")
    print(f"  {'difficulty':<12}{'lift':>10}{'ci_low':>10}{'ci_high':>10}{'support':>10}  decision")
    for d in (0.05, 0.20, 0.40, 0.50, 0.60, 0.80, 0.95):
        p = clab.cate({"task_difficulty": d}, learner=LEARNER_DR)
        if p.ci_low > 0:
            decision = "→ ship CHEAP"
        elif p.ci_high < 0:
            decision = "→ keep STRONG"
        else:
            decision = "  inconclusive"
        print(f"  {d:<12.2f}{p.value:>+10.4f}{p.ci_low:>+10.4f}{p.ci_high:>+10.4f}"
              f"{p.support_score:>10.2f}  {decision}")
    print()

    # ----- 3. Heterogeneity test — is personalising even worth it? -----
    print("[CausalLab] heterogeneity permutation test (n_perm=80):")
    het = clab.test_heterogeneity(n_permutations=80, max_eval_contexts=80)
    print(f"  observed Var(τ̂(c)) = {het.statistic:.5f}")
    print(f"  null     Var(τ̂(c)) = {het.null_mean:.5f} ± {het.null_std:.5f}")
    print(f"  p-value             = {het.p_value:.4f}    "
          f"heterogeneous? {het.is_heterogeneous}")
    if het.is_heterogeneous:
        print("  → personalisation is justified; ship the per-context router.")
    else:
        print("  → no statistical evidence for personalisation; stay with the average.")
    print()

    # ----- 4. Uplift curve + Qini coefficient. -----
    report = clab.uplift(learner=LEARNER_X, n_buckets=5)
    print("[CausalLab] uplift quintiles (X-learner sort):")
    print(report.summary)
    print(f"  Qini = {report.qini_coefficient:+.4f}   "
          f"Qini/|ATE| = {report.qini_normalised:+.2f}×")
    print()

    # ----- 5. Best Linear Predictor — the rule a coordinator ships. -----
    blp = clab.best_linear_predictor()
    print("[CausalLab] Best Linear Predictor of CATE (interpretable rule):")
    print(f"  R² = {blp.r_squared:.3f}  n = {blp.n}")
    print(f"  τ̂(c) ≈ {blp.intercept.coef:+.4f}  (intercept, "
          f"CI [{blp.intercept.ci_low:+.4f}, {blp.intercept.ci_high:+.4f}])")
    for c in blp.coefficients:
        print(f"          {c.coef:+.4f} · {c.feature}  "
              f"(CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}], p={c.p_value:.4f})")
    print()

    # ----- 6. Per-request recommendation. -----
    print("[CausalLab] per-request routing (coordinator surface):")
    for d in (0.10, 0.50, 0.90):
        rec = clab.recommend(
            {"task_difficulty": d},
            actions=["cheap", "strong"],
            baseline="strong",
            learner=LEARNER_DR,
        )
        print(f"  difficulty={d:.2f}  best={rec.best_action:<7}"
              f"  lift={rec.lift:+.4f}  CI=[{rec.lift_ci_low:+.4f}, "
              f"{rec.lift_ci_high:+.4f}]")
    print()
    print("This is the runtime engine surface a coordination engine drives:")
    print("  - PolicyLab → 'is the population average lifted?'")
    print("  - CausalLab → 'for *this* request, would the new policy win?'")
    print("Together they replace expensive online A/B tests with offline")
    print("counterfactual reasoning the coordinator can call inline.")


if __name__ == "__main__":
    main()
