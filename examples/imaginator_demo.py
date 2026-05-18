"""Imaginator demo — learn a world model, imagine, plan, certify.

Run::

    python examples/imaginator_demo.py

Four scenarios show off the Imaginator runtime primitive:

  1. Supply-chain MDP — learn a 2-state, 2-action MDP from random
     transitions; plan via value iteration; imagine returns with
     calibrated 95% Maurer-Pontil empirical-Bernstein bounds and
     Howard-Ramdas-McAuliffe-Sekhon anytime-valid confidence
     sequences; emit a Kearns-Singh PAC value bound; check
     PIT-calibration of one-step reward predictions.

  2. Drift detection on a 4-state random walk — observe a stationary
     transition, then a perturbed transition, and watch the per-step
     posterior shift.

  3. Linear-Gaussian rocket — register a 2D continuous-state system
     with 1D action, observe random transitions, recover ``A`` and
     ``B`` near-perfectly, then run a closed-form PILCO-style moment
     rollout with growing-variance bands.

  4. Coordination-engine handshake — register a Goal, query the
     imagined value of each candidate primitive's policy, pick the
     primitive with the tightest LCB beating its incumbent UCB.  The
     coordination engine reads only the receipts (chain_head + PAC
     bound) before committing real spend.

This demo runs in stdlib only — no Anthropic API key required, no
NumPy, no Torch.  Total runtime under 5 seconds on a laptop CPU.
"""
from __future__ import annotations

import random

from agi.imaginator import (
    FAMILY_LINEAR_GAUSSIAN,
    SAMPLE_THOMPSON,
    Imaginator,
    ImaginatorConfig,
)


def _hdr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def demo_supply_chain() -> None:
    _hdr("1. Supply-chain MDP — observe, plan, imagine, certify")

    im = Imaginator(ImaginatorConfig(rng_seed=7, discount=0.9))
    im.register_env(
        "supply", states=("ok", "stockout"), actions=("ship", "wait")
    )

    print(
        "Observed dynamics: ship-when-ok stays ok 95%, +1 reward; "
        "wait-when-stockout stays stockout 90%, -2 reward; ..."
    )
    rng = random.Random(0)
    for _ in range(400):
        s = rng.choice(["ok", "stockout"])
        a = rng.choice(["ship", "wait"])
        if a == "ship" and s == "stockout":
            nxt = "ok" if rng.random() < 0.8 else "stockout"
            r = -0.5
        elif a == "ship" and s == "ok":
            nxt = "ok" if rng.random() < 0.95 else "stockout"
            r = 1.0
        elif a == "wait" and s == "ok":
            nxt = "ok" if rng.random() < 0.7 else "stockout"
            r = 0.5
        else:
            nxt = "ok" if rng.random() < 0.1 else "stockout"
            r = -2.0
        im.observe("supply", s, a, nxt, r)
    print(f"  observed 400 random transitions.")

    plan = im.value_iteration("supply", horizon=50, discount=0.9, tol=1e-6)
    print("  optimal policy:", plan.policy)
    print(
        "  V*(ok)  ="
        f" {plan.values['ok']:7.3f}    V*(stockout) ="
        f" {plan.values['stockout']:7.3f}"
    )
    print(f"  value-iteration sweeps: {plan.sweeps}")

    roll = im.imagine(
        "supply",
        state="ok",
        policy=lambda s: plan.policy[s],
        horizon=20,
        samples=128,
        method=SAMPLE_THOMPSON,
    )
    print(
        f"  imagined Σγ^h r over 20 steps: "
        f"{roll.expected_return:7.3f} ± {roll.return_std:.3f}"
    )
    print(
        f"  Maurer-Pontil 95% interval     [{roll.value_lcb:7.3f}, "
        f"{roll.value_ucb:7.3f}]"
    )
    print(
        f"  HRMS 2021 anytime-valid CI     [{roll.hrms_lcb:7.3f}, "
        f"{roll.hrms_ucb:7.3f}]"
    )
    print("  return quantiles:")
    for q, v in roll.return_quantiles.items():
        print(f"    p{int(q*100):2d} → {v:7.3f}")

    pac = im.pac_value_bound(
        "supply", policy=plan.policy, delta=0.05, horizon=20
    )
    print(
        f"  Kearns-Singh simulation-lemma PAC bound:"
        f" |V̂π − V*π| ≤ {pac.epsilon:7.2f}"
    )
    print(
        f"     · transition error: {pac.transition_error:7.4f}"
        f"   · reward error: {pac.reward_error:7.4f}"
        f"   · min(n) = {pac.min_observations}"
    )

    need = im.required_samples_for_pac(
        env_id="supply", epsilon=1.0, delta=0.05
    )
    print(f"  Strehl-Littman-Wiewiora 2009 sample complexity for ε=1.0: {need:,}")

    ident = im.identifiability_report("supply", min_observations=10)
    print(
        f"  identifiability: {ident.n_under_observed}/{ident.n_pairs} "
        f"(s, a) pairs under-observed (<{ident.min_observations})"
    )

    pit = im.pit_calibration("supply")
    print(
        f"  PIT calibration (Massey 1951 KS test): "
        f"D={pit.ks_statistic:.4f}, p={pit.p_value:.4f} on {pit.n_observations} obs"
    )

    print(f"  audit-chain head: {im.chain_head[:24]}...")


def demo_drift_detection() -> None:
    _hdr("2. Drift detection on a 4-state random walk")

    im = Imaginator(ImaginatorConfig(rng_seed=11, discount=0.95))
    states = ("a", "b", "c", "d")
    im.register_env("walk", states=states, actions=("step",))

    rng = random.Random(0)
    # Stationary regime: bias +1.
    for _ in range(200):
        s = rng.choice(states)
        idx = states.index(s)
        nxt = states[min(idx + 1, len(states) - 1)] if rng.random() < 0.8 else s
        im.observe("walk", s, "step", nxt, 0.0)

    pre = im.posterior_mean_transition("walk", "b", "step")
    print(f"  pre-drift  posterior P(·|b,step): {pre}")

    # Drift regime: bias -1.
    for _ in range(200):
        s = rng.choice(states)
        idx = states.index(s)
        nxt = states[max(idx - 1, 0)] if rng.random() < 0.8 else s
        im.observe("walk", s, "step", nxt, 0.0)

    post = im.posterior_mean_transition("walk", "b", "step")
    print(f"  post-drift posterior P(·|b,step): {post}")
    print("  → posterior shifts smoothly — DriftSentinel reads per-step log-loss")


def demo_linear_gaussian() -> None:
    _hdr("3. Linear-Gaussian rocket — PILCO-style moment rollout")

    im = Imaginator(ImaginatorConfig(family=FAMILY_LINEAR_GAUSSIAN, rng_seed=1))
    im.register_env("rocket", state_dim=2, action_dim=1)
    rng = random.Random(2)
    A_true = [[1.0, 0.1], [0.0, 1.0]]
    B_true = [[0.0], [0.1]]
    s = [10.0, 0.0]
    for _ in range(500):
        a = [rng.gauss(0, 1)]
        nxt = [
            A_true[i][0] * s[0]
            + A_true[i][1] * s[1]
            + B_true[i][0] * a[0]
            + rng.gauss(0, 0.01)
            for i in range(2)
        ]
        im.observe("rocket", s, a, nxt, -(s[0] ** 2 + s[1] ** 2))
        s = nxt
        if abs(s[0]) > 100 or abs(s[1]) > 100:
            s = [10.0, 0.0]
    A, B = im.posterior_mean_dynamics("rocket")
    print("  posterior-mean A:")
    for row in A:
        print("   ", "  ".join(f"{v:+.3f}" for v in row))
    print("  posterior-mean B:")
    for row in B:
        print("   ", "  ".join(f"{v:+.3f}" for v in row))

    trace = im.moment_rollout(
        "rocket",
        state=[5.0, 0.0],
        policy=lambda s: [-0.5 * s[0]],
        horizon=10,
    )
    print("  moment rollout from [5, 0] under proportional control u=-0.5*x:")
    for h, (mu, cov) in enumerate(trace):
        print(
            f"   h={h+1:2d}: μ = ({mu[0]:+.3f}, {mu[1]:+.3f})"
            f"   σ² = ({cov[0][0]:.4f}, {cov[1][1]:.4f})"
        )


def demo_coordination_engine() -> None:
    _hdr("4. Coordination-engine handshake — which primitive should I call?")

    im = Imaginator(ImaginatorConfig(rng_seed=42, discount=0.9))
    im.register_env(
        "task", states=("backlog", "wip", "done"), actions=("rush", "review")
    )
    rng = random.Random(0)
    for _ in range(300):
        s = rng.choice(["backlog", "wip", "done"])
        a = rng.choice(["rush", "review"])
        if a == "rush":
            if s == "backlog":
                nxt = "wip" if rng.random() < 0.7 else "backlog"
                r = -0.1
            elif s == "wip":
                nxt = "done" if rng.random() < 0.6 else "wip"
                r = 0.2
            else:
                nxt = "done"
                r = 0.0
        else:
            if s == "backlog":
                nxt = "wip" if rng.random() < 0.3 else "backlog"
                r = -0.05
            elif s == "wip":
                nxt = "done" if rng.random() < 0.9 else "wip"
                r = 1.0
            else:
                nxt = "done"
                r = 0.0
        im.observe("task", s, a, nxt, r)

    incumbent = {"backlog": "rush", "wip": "rush", "done": "rush"}
    candidate = {"backlog": "rush", "wip": "review", "done": "review"}

    inc_roll = im.imagine(
        "task",
        state="backlog",
        policy=lambda s: incumbent[s],
        horizon=20,
        samples=256,
    )
    cand_roll = im.imagine(
        "task",
        state="backlog",
        policy=lambda s: candidate[s],
        horizon=20,
        samples=256,
    )
    print("  Coordination engine queries Imaginator:")
    print(
        f"    incumbent (rush-all)     value 95% CI:"
        f" [{inc_roll.value_lcb:6.3f}, {inc_roll.value_ucb:6.3f}]"
    )
    print(
        f"    candidate (rush+review)  value 95% CI:"
        f" [{cand_roll.value_lcb:6.3f}, {cand_roll.value_ucb:6.3f}]"
    )
    if cand_roll.value_lcb > inc_roll.value_ucb:
        verdict = "DEPLOY candidate — LCB beats incumbent UCB (eval-gated)."
    elif inc_roll.value_lcb > cand_roll.value_ucb:
        verdict = "KEEP incumbent — candidate LCB does not beat incumbent UCB."
    else:
        verdict = "INSUFFICIENT EVIDENCE — Imaginator recommends more rollouts."
    print(f"  → {verdict}")

    # PAC certificate the coordinator forwards to compliance:
    pac = im.pac_value_bound("task", policy=candidate, delta=0.05, horizon=20)
    print(
        f"  PAC certificate (δ=0.05, h=20): |V̂π − V*π| ≤ {pac.epsilon:.3f}"
    )
    print(f"  fingerprint head replayed by AttestationLedger: {pac.fingerprint_hash[:24]}...")


def main() -> None:
    print("=" * 70)
    print("Imaginator — learn a world model, imagine, certify.")
    print(
        "Pure stdlib • Bayesian Dirichlet-multinomial + matrix-normal-inverse-Wishart"
    )
    print(
        "Kearns-Singh 2002 PAC bound • Maurer-Pontil 2009 Bernstein • HRMS 2021"
    )
    print("=" * 70)
    demo_supply_chain()
    demo_drift_detection()
    demo_linear_gaussian()
    demo_coordination_engine()
    print()
    print("Done. Compose Imaginator with:")
    print("  Searcher       — tree search over imagined transitions")
    print("  ActiveInferencer — register as generative model")
    print("  Quantilizer    — gate deployment on imagined-return quantile")
    print("  Distiller      — distil value-iteration policy")
    print("  DriftSentinel  — per-step log-loss CUSUM")
    print("  Coordinator    — `imagine → certify → act` for every Goal")


if __name__ == "__main__":
    main()
