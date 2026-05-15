"""Counterfactor demo — sequential off-policy evaluation as a runtime primitive.

Scenario
--------

A coordination engine routes user sessions through a *three-step* chain:

  1. **Triage**: classify the request as `code`, `chat`, or `analysis`.
  2. **Plan**: pick a tool kit from {`fast`, `accurate`, `cheap`}.
  3. **Finalise**: pick a verifier from {`strict`, `lenient`}.

The runtime currently ships a hand-tuned logging policy that mixes
across all paths.  The coordination engine proposes two candidate
policies and wants to *certify* — before shipping — that a candidate
either dominates or under-performs the live policy.

This demo walks an investor through six realistic uses of
``agi.counterfactor.Counterfactor`` end-to-end:

  1. **Trajectory-IS / PDIS / WPDIS** point estimates with finite-sample
     CIs — the basic off-policy value of each candidate.
  2. **Direct method** with a tabular ``Q̂`` fit on the logs — variance
     trade-off for the same data.
  3. **DR-RL / WDR / MAGIC** doubly-robust estimators — the workhorse
     of industrial OPE.  MAGIC chooses an MSE-optimal blend.
  4. **HCOPE** lower bound (Thomas et al. 2015) — *pessimistic*
     confidence bound for safe deployment.
  5. **Paired comparison** — does candidate A dominate the live
     policy with confidence ≥ 95%?
  6. **Diagnostics** — ESS, overlap-KL, max weight, clip fraction:
     surface support-violation regimes *before* you trust the number.

Run::

    python -m examples.counterfactor_demo

Honest about limits
-------------------

Off-policy evaluation inherits the support of the logging policy.  If
the candidate puts probability mass on a (state, action) pair the
behaviour never sampled, the diagnostic will surface a `low overlap`
warning and the estimator should not be trusted.  HCOPE handles this
explicitly by truncating IS-weighted returns; the lower bound widens
gracefully as overlap degrades.
"""
from __future__ import annotations

import random

from agi.counterfactor import (
    CI_BERNSTEIN,
    CI_HOEFFDING,
    Counterfactor,
    DeterministicPolicy,
    EpsilonGreedyPolicy,
    LoggedStep,
    LoggedTrajectory,
    METHOD_DR_RL,
    METHOD_MAGIC,
    METHOD_PDIS,
    METHOD_TRAJ_IS,
    METHOD_WPDIS,
    SoftmaxPolicy,
    TabularQModel,
    UniformPolicy,
)


# ----------------------- world: 3-step routing -----------------------


STAGES = ("triage", "plan", "finalise")
ACTIONS = {
    "triage": ("code", "chat", "analysis"),
    "plan": ("fast", "accurate", "cheap"),
    "finalise": ("strict", "lenient"),
}


def _true_reward(stage: str, action: str) -> float:
    """Hidden true reward of each (stage, action). Returns roughly $ value."""
    table = {
        ("triage", "code"): 0.6,
        ("triage", "chat"): 0.4,
        ("triage", "analysis"): 0.5,
        ("plan", "fast"): 0.3,
        ("plan", "accurate"): 0.7,
        ("plan", "cheap"): 0.2,
        ("finalise", "strict"): 0.6,
        ("finalise", "lenient"): 0.4,
    }
    return table.get((stage, action), 0.0)


# ----------------------- behaviour: live logging policy -------------


def _behaviour_dist(stage: str) -> dict[str, float]:
    """The runtime's current logging policy at each stage."""
    if stage == "triage":
        return {"code": 0.5, "chat": 0.3, "analysis": 0.2}
    if stage == "plan":
        return {"fast": 0.4, "accurate": 0.3, "cheap": 0.3}
    return {"strict": 0.5, "lenient": 0.5}


def _sample_trajectories(n: int, seed: int = 0) -> list[LoggedTrajectory]:
    r = random.Random(seed)
    trajs: list[LoggedTrajectory] = []
    for _ in range(n):
        steps: list[LoggedStep] = []
        for stage in STAGES:
            dist = _behaviour_dist(stage)
            actions = list(dist.keys())
            probs = list(dist.values())
            a = r.choices(actions, weights=probs)[0]
            # add zero-mean noise so the empirical Q̂ is non-degenerate
            reward = _true_reward(stage, a) + r.gauss(0.0, 0.05)
            steps.append(
                LoggedStep(
                    state=stage,
                    action=a,
                    reward=reward,
                    behavior_prob=dist[a],
                )
            )
        trajs.append(LoggedTrajectory(steps=tuple(steps)))
    return trajs


# ----------------------- candidate target policies ------------------


def _candidate_greedy_policy(state):
    """Deterministic: pick the highest-immediate-reward action at every stage."""
    best = {
        "triage": "code",         # 0.6
        "plan": "accurate",       # 0.7
        "finalise": "strict",     # 0.6
    }
    return {a: (1.0 if a == best[state] else 0.0) for a in ACTIONS[state]}


def _candidate_eps_greedy_policy(state):
    """ε-greedy on the same best-action map, ε=0.1."""
    best = {"triage": "code", "plan": "accurate", "finalise": "strict"}
    eps = 0.1
    n = len(ACTIONS[state])
    out = {a: eps / n for a in ACTIONS[state]}
    out[best[state]] += 1.0 - eps
    return out


def _live_policy(state):
    """The runtime's current behaviour (also passed as a target — sanity check)."""
    return _behaviour_dist(state)


def _true_value(policy) -> float:
    """Closed-form true value of a 3-step policy."""
    v = 0.0
    for stage in STAGES:
        dist = policy(stage)
        for a, p in dist.items():
            v += p * _true_reward(stage, a)
    return v


# ----------------------- demo -----------------------


def main() -> None:
    print("=" * 72)
    print("Counterfactor — Off-policy evaluation of a 3-step coordination policy")
    print("=" * 72)

    trajs = _sample_trajectories(n=1000, seed=42)
    ctr = Counterfactor(reward_range=(0.0, 1.0), weight_cap=50.0)
    for tr in trajs:
        ctr.log_trajectory(tr)
    print(f"\nLogged {ctr.n_trajectories} trajectories under live policy")

    # Truth (closed form)
    truth_live = _true_value(_live_policy)
    truth_greedy = _true_value(_candidate_greedy_policy)
    truth_eps = _true_value(_candidate_eps_greedy_policy)
    print(f"\nGround-truth policy values (closed form):")
    print(f"  live policy        V* = {truth_live:.4f}")
    print(f"  candidate ε-greedy V* = {truth_eps:.4f}")
    print(f"  candidate greedy   V* = {truth_greedy:.4f}")

    # Fit a tabular Q̂ on the logged trajectories
    q = TabularQModel(gamma=1.0)
    q.fit(ctr.trajectories())

    # ---- 1. Point estimates with bounded-Bernstein CIs ----
    print("\n" + "-" * 72)
    print("1. Point estimates with finite-sample CIs")
    print("-" * 72)
    print(f"{'method':10s} {'live':>20s} {'eps-greedy':>20s} {'greedy':>20s}")
    for method in (METHOD_TRAJ_IS, METHOD_PDIS, METHOD_WPDIS):
        row = [method]
        for pol in (_live_policy, _candidate_eps_greedy_policy, _candidate_greedy_policy):
            rep = ctr.evaluate(pol, method=method, ci_method=CI_BERNSTEIN, alpha=0.05)
            row.append(f"{rep.value:.4f} [{rep.ci_lo:.3f}, {rep.ci_hi:.3f}]")
        print(f"{row[0]:10s} {row[1]:>20s} {row[2]:>20s} {row[3]:>20s}")

    print("\n   (CIs are Maurer-Pontil 2009 empirical-Bernstein at α=0.05)")

    # ---- 2. Direct method with the fitted Q̂ ----
    print("\n" + "-" * 72)
    print("2. Direct method — model-based estimate (low variance, biased by Q̂ error)")
    print("-" * 72)
    print(f"{'method':10s} {'live':>20s} {'eps-greedy':>20s} {'greedy':>20s}")
    rep_live = ctr.evaluate(_live_policy, method="dm", q_model=q)
    rep_eps = ctr.evaluate(_candidate_eps_greedy_policy, method="dm", q_model=q)
    rep_greedy = ctr.evaluate(_candidate_greedy_policy, method="dm", q_model=q)
    print(
        f"{'dm':10s} "
        f"{rep_live.value:>10.4f} [{rep_live.ci_lo:.3f}, {rep_live.ci_hi:.3f}]   "
        f"{rep_eps.value:>10.4f} [{rep_eps.ci_lo:.3f}, {rep_eps.ci_hi:.3f}]   "
        f"{rep_greedy.value:>10.4f} [{rep_greedy.ci_lo:.3f}, {rep_greedy.ci_hi:.3f}]"
    )

    # ---- 3. Doubly-robust estimators ----
    print("\n" + "-" * 72)
    print("3. Doubly-robust estimators (Jiang-Li 2016 / Thomas-Brunskill 2016)")
    print("-" * 72)
    for method in (METHOD_DR_RL, "wdr", METHOD_MAGIC):
        print(f"\n  {method.upper()}")
        for name, pol in (
            ("live", _live_policy),
            ("eps-greedy", _candidate_eps_greedy_policy),
            ("greedy", _candidate_greedy_policy),
        ):
            rep = ctr.evaluate(pol, method=method, q_model=q, ci_method=CI_BERNSTEIN)
            print(
                f"    {name:12s}  V̂={rep.value:.4f}  "
                f"CI=[{rep.ci_lo:.3f}, {rep.ci_hi:.3f}]  "
                f"ESS={rep.ess:.1f}  clip={rep.clip_fraction:.2%}"
            )

    # ---- 4. HCOPE — pessimistic lower bound for safe deployment ----
    print("\n" + "-" * 72)
    print("4. HCOPE — high-confidence off-policy *lower bound* (Thomas et al. 2015)")
    print("-" * 72)
    for name, pol in (
        ("live", _live_policy),
        ("eps-greedy", _candidate_eps_greedy_policy),
        ("greedy", _candidate_greedy_policy),
    ):
        hr = ctr.hcope(pol, method=METHOD_PDIS, alpha=0.05)
        print(
            f"  {name:12s}  point={hr.point_value:.4f}  "
            f"HCOPE-lb({1 - hr.alpha:.0%})={hr.lower_bound:.4f}  "
            f"ξ={hr.xi:.2f}"
        )
    print("\n  Decision rule: ship candidate iff its HCOPE-lb > live point estimate")

    # ---- 5. Paired comparison ----
    print("\n" + "-" * 72)
    print("5. Paired off-policy comparison (per-trajectory Student-t)")
    print("-" * 72)
    cmp_eps = ctr.compare(
        _candidate_eps_greedy_policy,
        _live_policy,
        name_a="eps-greedy",
        name_b="live",
        method=METHOD_PDIS,
        alpha=0.05,
    )
    cmp_greedy = ctr.compare(
        _candidate_greedy_policy,
        _live_policy,
        name_a="greedy",
        name_b="live",
        method=METHOD_PDIS,
        alpha=0.05,
    )
    for cmp in (cmp_eps, cmp_greedy):
        print(
            f"  {cmp.name_a:12s} vs {cmp.name_b:10s}  "
            f"Δ={cmp.delta:+.4f}  CI=[{cmp.delta_ci_lo:+.3f}, {cmp.delta_ci_hi:+.3f}]  "
            f"P(A>B)={cmp.p_a_better:.3f}  ship={cmp.a_dominates}"
        )

    # ---- 6. Diagnostics ----
    print("\n" + "-" * 72)
    print("6. Diagnostics — ESS, overlap, max weight (per target)")
    print("-" * 72)
    for name, pol in (
        ("live", _live_policy),
        ("eps-greedy", _candidate_eps_greedy_policy),
        ("greedy", _candidate_greedy_policy),
    ):
        d = ctr.diagnostics(pol)
        print(
            f"  {name:12s}  "
            f"ESS_traj={d.ess_trajectory:7.1f}  ESS_pdis_min={d.ess_pdis_min:7.1f}  "
            f"max_w={d.max_weight:6.2f}  coverage={d.coverage:.3f}  "
            f"warns={len(d.warnings)}"
        )
        for w in d.warnings:
            print(f"      ⚠ {w}")

    print("\n" + "=" * 72)
    print("Coverage:", ctr.coverage())
    print("=" * 72)


if __name__ == "__main__":
    main()
