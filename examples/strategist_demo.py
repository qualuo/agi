"""Strategist demo — meta-decision API for a coordination engine.

This walks an investor through the five verdicts the Strategist can
emit and shows how *every* forecasting primitive in the runtime
(CalibrationEngine, ConformalPredictor, CausalLab, PolicyLab) feeds a
single, structured `StrategyRecommendation` that a coordination engine
can act on.

Run:
    python examples/strategist_demo.py
"""
from __future__ import annotations

import random
import statistics

from agi.calibration import CalibrationEngine, METHOD_ISOTONIC
from agi.causal import CausalLab
from agi.conformal import ConformalPredictor
from agi.policy_lab import LoggedEvent, PolicyLab
from agi.strategist import (
    Candidate,
    Strategist,
    StrategyConstraints,
    StrategyOutcome,
    STRAT_DEFER,
    STRAT_EXPLORE,
    STRAT_HEDGE,
    STRAT_REJECT,
    STRAT_SINGLE,
)


def _print_rec(label: str, rec):
    print(f"\n--- {label} ---")
    print(f"strategy:      {rec.strategy}")
    print(f"confidence:    {rec.confidence}")
    if rec.primary is not None:
        p = rec.primary
        print(f"primary arm:   {p.candidate.id}  (model={p.candidate.model})")
        print(f"  p_success:   raw={p.raw_p_success:.3f}  "
              f"calibrated={p.calibrated_p_success:.3f}  "
              f"[{p.p_success_lower:.3f}, {p.p_success_upper:.3f}]")
        print(f"  cost USD:    point=${p.cost_mean_usd:.4f}  "
              f"p95=${p.cost_p95_usd:.4f}  width=${p.cost_width_usd:.4f}")
        if p.cate_lift is not None:
            print(f"  CATE vs base: {p.cate_lift:+.3f} "
                  f"[{p.cate_ci_low:+.3f}, {p.cate_ci_high:+.3f}]")
        if p.ope_value is not None:
            print(f"  OPE ({p.ope_method}): {p.ope_value:+.3f} "
                  f"[{p.ope_ci_low:+.3f}, {p.ope_ci_high:+.3f}]")
        print(f"  risk score:  {p.risk_score:.2f}")
    if rec.hedged_arms:
        print("hedged arms:   " + ", ".join(a.candidate.id for a in rec.hedged_arms))
    print(f"EV (mean):     ${rec.expected_value_usd:+.4f}")
    print(f"EV (lower):    ${rec.value_lower_bound_usd:+.4f}")
    print(f"cost p95:      ${rec.cost_p95_usd:.4f}")
    print(f"p_success:     {rec.p_success:.3f}")
    print(f"pareto:        {', '.join(rec.pareto_frontier)}")
    print(f"warnings:      {', '.join(rec.warnings) or '-'}")
    print(f"rationale:     {rec.rationale}")


def scenario_single_winner(strat: Strategist):
    """When one candidate beats the SLO on its own, pick it."""
    rec = strat.recommend(
        candidates=[
            Candidate(id="opus",   raw_p_success=0.92, raw_cost_usd=0.40,
                      samples=80, model="claude-opus-4-7"),
            Candidate(id="sonnet", raw_p_success=0.85, raw_cost_usd=0.18,
                      samples=80, model="claude-sonnet-4-6"),
        ],
        constraints=StrategyConstraints(
            target_p_success=0.85, max_cost_usd=1.0, payoff_usd=2.0,
        ),
        context={"task_difficulty": 0.5},
    )
    _print_rec("Scenario 1: clear single-arm winner", rec)


def scenario_hedge(strat: Strategist):
    """No arm meets the SLO alone but two cheaply combined do."""
    rec = strat.recommend(
        candidates=[
            Candidate(id="haiku",  raw_p_success=0.74, raw_cost_usd=0.05,
                      samples=80, model="claude-haiku-4-5"),
            Candidate(id="sonnet", raw_p_success=0.85, raw_cost_usd=0.18,
                      samples=80, model="claude-sonnet-4-6"),
            Candidate(id="opus",   raw_p_success=0.92, raw_cost_usd=0.40,
                      samples=80, model="claude-opus-4-7"),
        ],
        constraints=StrategyConstraints(
            target_p_success=0.95, max_cost_usd=0.50, payoff_usd=2.0,
        ),
    )
    _print_rec("Scenario 2: HEDGE under SLO", rec)


def scenario_explore(strat: Strategist):
    """Cold-start arm with a fat upper tail: run for data."""
    rec = strat.recommend(
        candidates=[
            Candidate(id="experimental", raw_p_success=0.60, raw_cost_usd=0.05,
                      samples=0, model="qwen-experimental"),
        ],
        constraints=StrategyConstraints(
            target_p_success=0.90, max_cost_usd=1.0, payoff_usd=10.0,
            explore_min_evidence=10,
        ),
    )
    _print_rec("Scenario 3: EXPLORE a novel arm", rec)


def scenario_reject(strat: Strategist):
    """All candidates are over budget — caller must abort."""
    rec = strat.recommend(
        candidates=[
            Candidate(id="expensive", raw_p_success=0.95, raw_cost_usd=2.00,
                      samples=80, model="big-model"),
        ],
        constraints=StrategyConstraints(
            target_p_success=0.90, max_cost_usd=0.20, payoff_usd=2.0,
        ),
    )
    _print_rec("Scenario 4: REJECT (over budget)", rec)


def scenario_defer(strat: Strategist):
    """Mean EV is positive but EV_LB is negative under risk aversion."""
    rec = strat.recommend(
        candidates=[
            Candidate(id="marginal", raw_p_success=0.55, raw_cost_usd=0.12,
                      samples=80, model="claude-haiku-4-5"),
        ],
        constraints=StrategyConstraints(
            target_p_success=0.95, max_cost_usd=1.0,
            payoff_usd=0.30, risk_aversion=3.0,
        ),
    )
    _print_rec("Scenario 5: DEFER under risk aversion", rec)


def main():
    rng = random.Random(0)

    # --- Build calibration: forecaster reports 0.90, truth is 0.70.
    cal = CalibrationEngine(method=METHOD_ISOTONIC,
                            min_samples_to_fit=10, refit_every=10)
    for _ in range(100):
        cal.observe(0.90, rng.random() < 0.70)
        cal.observe(0.40, rng.random() < 0.30)
    cal.fit()

    # --- Build conformal cost predictor: forecaster underestimates by 20%.
    conf = ConformalPredictor(target_coverage=0.95, max_history=500)
    for _ in range(200):
        pred = rng.uniform(0.05, 0.50)
        actual = pred * rng.uniform(1.0, 1.4)
        conf.record(features={"prediction": pred}, prediction=pred, outcome=actual)

    # --- Build a PolicyLab and CausalLab from a small synthetic log.
    lab = PolicyLab()
    causal = CausalLab(treatment="haiku", control="opus")
    for _ in range(120):
        action = rng.choice(("haiku", "opus"))
        ctx = {"task_difficulty": rng.uniform(0, 1)}
        # opus is reliably better
        success_p = 0.92 if action == "opus" else 0.72
        reward = 2.0 if rng.random() < success_p else 0.0
        reward -= 0.40 if action == "opus" else 0.05
        ev = LoggedEvent(context=ctx, action=action,
                         propensity=0.5, reward=reward)
        lab.record(ev)
        causal.record(ev)

    print("=" * 72)
    print("Strategist demo — meta-decision API for the coordination engine")
    print("=" * 72)
    print(f"Calibration:  {cal.report().n} samples observed")
    print(f"Conformal:    {len(conf)} (features, prediction, outcome) records")
    print(f"PolicyLab:    {len(lab.events())} logged events")
    print(f"CausalLab:    treatment=haiku, control=opus")

    strat = Strategist(
        calibration=cal,
        conformal=conf,
        causal=causal,
        policy_lab=lab,
        baseline_action_id="opus",
    )

    scenario_single_winner(strat)
    scenario_hedge(strat)
    scenario_explore(strat)
    scenario_reject(strat)
    scenario_defer(strat)

    # --- Closing the loop: simulate 60 outcomes and report self-eval.
    print("\n" + "=" * 72)
    print("Closing the loop: 60 simulated outcomes through observe()")
    print("=" * 72)
    cands = [
        Candidate(id="haiku",  raw_p_success=0.74, raw_cost_usd=0.05,
                  samples=80, model="claude-haiku-4-5"),
        Candidate(id="sonnet", raw_p_success=0.85, raw_cost_usd=0.18,
                  samples=80, model="claude-sonnet-4-6"),
        Candidate(id="opus",   raw_p_success=0.92, raw_cost_usd=0.40,
                  samples=80, model="claude-opus-4-7"),
    ]
    constraints = StrategyConstraints(
        target_p_success=0.85, max_cost_usd=1.0, payoff_usd=2.0,
    )
    rec_log: list[tuple[str, str]] = []
    for _ in range(60):
        rec = strat.recommend(cands, constraints)
        chosen = rec.primary.candidate.id if rec.primary else \
                 rec.hedged_arms[0].candidate.id
        # truth: actual p_success follows calibrated forecast
        truth = next(c for c in cands if c.id == chosen)
        success = rng.random() < (0.70 if truth.id == "opus" else 0.55 if truth.id == "sonnet" else 0.45)
        actual_cost = truth.raw_cost_usd * rng.uniform(1.0, 1.3)
        strat.observe(rec, StrategyOutcome(
            recommendation_id=rec.id, chosen_arm_id=chosen,
            success=success, cost_usd=actual_cost,
        ))
        rec_log.append((rec.strategy, chosen))

    cov = strat.coverage_report()
    print(f"\nstrategist self-evaluation ({cov.n} observations):")
    print(f"  Brier score          {cov.p_success_brier:.4f}  (lower = better; 0.25 = uninformed)")
    print(f"  log loss             {cov.p_success_log_loss:.4f}")
    print(f"  ECE                  {cov.p_success_ece:.4f}  (calibration gap)")
    print(f"  cost_p95 breach rate {cov.cost_p95_breach_rate:.4f}  "
          "(target ~5% at 95% target_coverage)")
    print(f"  mean realised value  ${cov.mean_realised_value_usd:+.4f}")
    print(f"  mean predicted value ${cov.mean_predicted_value_usd:+.4f}")
    print("\n  per-strategy breakdown:")
    for k, v in cov.per_strategy.items():
        print(f"    {k:8s}  n={int(v['n']):3d}  "
              f"success={v['success_rate']:.2f}  "
              f"mean cost=${v['mean_cost_usd']:.4f}  "
              f"realised EV=${v['mean_realised_value_usd']:+.4f}  "
              f"predicted EV=${v['mean_predicted_value_usd']:+.4f}")

    # Distribution of strategies chosen
    from collections import Counter
    dist = Counter(s for s, _ in rec_log)
    print(f"\n  strategy distribution: {dict(dist)}")
    arm_dist = Counter(a for _, a in rec_log)
    print(f"  arm distribution:      {dict(arm_dist)}")


if __name__ == "__main__":
    main()
