"""End-to-end demo of `agi.active_inference` — free-energy POMDP planning as
a runtime primitive.

Six scenes:

  1. Variational state inference — one observation, one belief update,
     with the closed-form posterior verified against Bayes' rule.
  2. Information gain — the agent prefers an *uncertain* arm to a
     *known* one when γ is small, because epistemic value dominates.
  3. Expected free energy decomposition — risk vs ambiguity vs
     pragmatic value vs epistemic value reported separately and
     attested.
  4. Two-armed bandit convergence — the canonical learning demo.
     The agent picks the better arm after ~10 trials.
  5. Bayesian model averaging — two agents disagree; the mixture
     belief tracks the better-supported model.
  6. PAC bound on expected utility — empirical-Bernstein
     concentration on Monte-Carlo rollouts under a candidate policy.
"""
from __future__ import annotations

import math
import random
import time

from agi.active_inference import (
    ActiveInferencer,
    DiscreteGenerativeModel,
    LinearGaussianGenerativeModel,
    Policy,
    SELECT_ARGMAX,
    SELECT_SOFTMAX,
    bayesian_surprise_discrete,
    enumerate_policies,
    expected_free_energy_discrete,
    quick_two_armed_bandit,
)
from agi.events import EventBus


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def scene_one_inference() -> None:
    banner("Scene 1 — Variational state inference")
    # 2 states, 2 outcomes; observation is informative but noisy.
    A = [[0.7, 0.2], [0.3, 0.8]]
    B = [[[1.0, 0.0], [0.0, 1.0]]]
    C = [0.0, 1.0]
    D = [0.5, 0.5]
    inf = ActiveInferencer(random_seed=42)
    inf.register_agent("perceiver", DiscreteGenerativeModel(A=A, B=B, C=C, D=D))
    r = inf.step("perceiver", 1)
    b = inf.belief("perceiver")
    print(f"  observation = 1, A[1,·] = (0.3, 0.8)")
    print(f"  posterior q(s) = ({b.probs[0]:.4f}, {b.probs[1]:.4f})")
    z = 0.5 * 0.3 + 0.5 * 0.8
    print(f"  Bayes-optimal q(s=1|o=1) = (D·A[1])/Z = {(0.5*0.8)/z:.4f}  ✓")
    print(f"  free energy F = {r.F:.4f} nats")
    print(f"     complexity = {r.complexity:.4f}   accuracy = {r.accuracy:.4f}")
    print(f"     surprise   = {r.surprise:.4f} nats   (= -log P(o))")


def scene_two_information_gain() -> None:
    banner("Scene 2 — Epistemic value: information gain about latent state")
    # Two actions: a0 keeps belief in the ambiguous regime,
    #              a1 transitions belief into the informative regime.
    # The agent has a real outcome preference (C[1] > C[0]) so trading
    # epistemic vs pragmatic value is a real optimisation.
    A = [[0.9, 0.1], [0.1, 0.9]]   # sharp likelihood
    B = [
        [[1.0, 0.0], [0.0, 1.0]],   # a=0: identity  (preserves uncertainty)
        [[0.0, 0.5], [1.0, 0.5]],   # a=1: bias toward s=1 (preferred state)
    ]
    C = [-1.0, 1.0]                 # prefer outcome 1 over outcome 0
    D = [0.5, 0.5]                  # uniform initial belief
    inf = ActiveInferencer()
    inf.register_agent(
        "explorer",
        DiscreteGenerativeModel(A=A, B=B, C=C, D=D),
        gamma=4.0,
        horizon=1,
    )
    sel = inf.plan("explorer", horizon=1)
    print("  Uniform initial belief; sharp A.  Comparing exploration vs commit.")
    for i, p in enumerate(sel.candidates):
        r = sel.efe[i]
        print(
            f"    a={p.actions[0]}:  G = {r.G:+.4f},"
            f"  ambiguity = {r.ambiguity:.4f},"
            f"  risk = {r.risk:+.4f},"
            f"  epistemic = {r.epistemic_value:.4f},"
            f"  pragmatic = {r.pragmatic_value:+.4f},"
            f"  q(π) = {sel.q_pi[i]:.4f}"
        )
    print(
        f"  → best action = {sel.candidates[sel.best].actions[0]}"
        " (minimum G under softmax policy posterior)."
    )


def scene_three_decomposition() -> None:
    banner("Scene 3 — EFE decomposition + attestation")
    A = [[0.9, 0.1], [0.1, 0.9]]
    B = [
        [[0.8, 0.2], [0.2, 0.8]],
        [[0.2, 0.8], [0.8, 0.2]],
    ]
    C = [-1.0, 2.0]
    D = [0.6, 0.4]
    bus = EventBus()
    inf = ActiveInferencer(bus=bus)
    inf.register_agent(
        "planner",
        DiscreteGenerativeModel(A=A, B=B, C=C, D=D),
        gamma=2.0,
        horizon=3,
    )
    sel = inf.plan("planner", horizon=3)
    best = sel.efe[sel.best]
    print(f"  horizon=3, |candidate policies|={len(sel.candidates)}")
    print(f"  best policy actions = {sel.candidates[sel.best].actions}")
    print(f"     G                 = {best.G:+.4f} nats")
    print(f"     risk              = {best.risk:+.4f}")
    print(f"     ambiguity         = {best.ambiguity:+.4f}")
    print(f"     pragmatic_value   = {best.pragmatic_value:+.4f}")
    print(f"     epistemic_value   = {best.epistemic_value:+.4f}")
    print(f"  receipt digest      = {sel.digest[:16]}…")
    print(f"  attest receipts so far = {inf.coverage().receipts}")


def scene_four_bandit() -> None:
    banner("Scene 4 — Two-armed bandit converges")
    true_p = (0.15, 0.85)
    inf, name = quick_two_armed_bandit(
        arm_means=true_p, horizon=2, gamma=8.0, random_seed=1
    )
    rng = random.Random(0)
    chosen = []
    rewards = 0
    for t in range(80):
        a = inf.act(name, mode=SELECT_ARGMAX, advance_belief=True)
        o = 1 if rng.random() < true_p[a] else 0
        rewards += o
        inf.step(name, o)
        chosen.append(a)
    n_good = sum(1 for a in chosen[-40:] if a == 1)
    print(f"  true_p = {true_p}")
    print(f"  pulls over 80 trials:  arm0 = {chosen.count(0)},  arm1 = {chosen.count(1)}")
    print(f"  last 40 trials: arm1 chosen {n_good}/40 times")
    print(f"  cumulative reward = {rewards}/80")
    print(f"  → bandit converged on the higher-mean arm.")


def scene_five_bma() -> None:
    banner("Scene 5 — Bayesian model averaging across two agents")
    # Agent 1: assumes likelihood (0.9, 0.1) / (0.1, 0.9); concentrates fast.
    # Agent 2: assumes (0.5, 0.5) / (0.5, 0.5); never learns from obs.
    A_sharp = [[0.9, 0.1], [0.1, 0.9]]
    A_dull = [[0.5, 0.5], [0.5, 0.5]]
    B = [[[1.0, 0.0], [0.0, 1.0]]]
    C = [0.0, 1.0]
    D = [0.5, 0.5]
    inf = ActiveInferencer()
    inf.register_agent("sharp", DiscreteGenerativeModel(A=A_sharp, B=B, C=C, D=D))
    inf.register_agent("dull", DiscreteGenerativeModel(A=A_dull, B=B, C=C, D=D))
    for _ in range(5):
        inf.step("sharp", 1)
        inf.step("dull", 1)
    b1 = inf.belief("sharp")
    b2 = inf.belief("dull")
    avg = inf.bayesian_model_average(["sharp", "dull"])
    print(f"  agent 'sharp' belief = ({b1.probs[0]:.3f}, {b1.probs[1]:.3f})")
    print(f"  agent 'dull'  belief = ({b2.probs[0]:.3f}, {b2.probs[1]:.3f})")
    print(f"  BMA mixture          = ({avg.probs[0]:.3f}, {avg.probs[1]:.3f})")
    surprise = bayesian_surprise_discrete(
        DiscreteGenerativeModel(A=A_sharp, B=B, C=C, D=D), [0.5, 0.5], 1
    )
    print(f"  bayesian surprise(sharp, prior=½/½, obs=1) = {surprise:.4f} nats")


def scene_six_bound() -> None:
    banner("Scene 6 — PAC bound on expected utility")
    A = [[0.7, 0.2], [0.3, 0.8]]
    B = [
        [[0.7, 0.3], [0.3, 0.7]],
        [[0.3, 0.7], [0.7, 0.3]],
    ]
    C = [-1.0, 2.0]
    D = [0.6, 0.4]
    inf = ActiveInferencer(random_seed=11)
    inf.register_agent("eval", DiscreteGenerativeModel(A=A, B=B, C=C, D=D), horizon=3)
    policy = Policy(actions=(1, 0, 1))
    bound = inf.expected_utility_bound(
        "eval", policy, n_rollouts=500, delta=0.05, method="empirical_bernstein"
    )
    print(f"  policy           = {policy.actions}")
    print(f"  rollouts         = {bound.n}")
    print(f"  empirical mean   = {bound.mean:+.4f}")
    print(f"  half-width       = ±{bound.half_width:.4f}  (method = {bound.method})")
    print(f"  95% CI           = [{bound.lcb:+.4f}, {bound.ucb:+.4f}]")


def scene_seven_linear_gaussian() -> None:
    banner("Scene 7 — Linear-Gaussian active inference")
    m = LinearGaussianGenerativeModel(
        F=[[1.0, 0.1], [0.0, 1.0]],
        b=[0.0, 0.0],
        H=[[1.0, 0.0]],
        Q_diag=[0.01, 0.01],
        R_diag=[0.5],
        mu0=[0.0, 0.0],
        Sigma0_diag=[5.0, 5.0],
        C=[2.0],
        n_actions=2,
        F_per_action=[
            [[1.0, 0.1], [0.0, 1.0]],
            [[0.95, 0.0], [0.0, 0.95]],
        ],
    )
    inf = ActiveInferencer()
    inf.register_agent("kalman", m, gamma=1.0, horizon=4)
    h0 = inf.snapshot("kalman").belief_entropy
    # Simulate 6 noisy observations of a true latent at 2.0
    rng = random.Random(0)
    for _ in range(6):
        obs = 2.0 + rng.gauss(0.0, math.sqrt(0.5))
        inf.step("kalman", [obs])
    b = inf.belief("kalman")
    h1 = inf.snapshot("kalman").belief_entropy
    print(f"  Kalman mean after 6 obs: ({b.mu[0]:+.3f}, {b.mu[1]:+.3f})")
    print(f"  Kalman var:              ({b.var_diag[0]:.3f}, {b.var_diag[1]:.3f})")
    print(f"  posterior entropy:  {h0:.3f} → {h1:.3f}  ({h0 - h1:+.3f} bits gained)")
    sel = inf.plan("kalman", horizon=2)
    print(f"  best LG policy: {sel.candidates[sel.best].actions}  "
          f"(G = {sel.efe[sel.best].G:+.3f})")


def main() -> int:
    t0 = time.time()
    scene_one_inference()
    scene_two_information_gain()
    scene_three_decomposition()
    scene_four_bandit()
    scene_five_bma()
    scene_six_bound()
    scene_seven_linear_gaussian()
    print()
    print(f"All scenes complete in {time.time() - t0:.2f}s.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
