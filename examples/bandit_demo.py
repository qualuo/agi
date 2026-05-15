"""Bandit demo — head-to-head comparison of seven algorithms on a
stationary 4-armed Bernoulli problem, then a switch to a contextual
linear setting.

Showcases:
  * Stochastic + adversarial + contextual algorithms on the same
    interface.
  * Tamper-evident SHA-256 fingerprint per campaign.
  * Anytime regret upper bound that holds at every t.
  * Composition: replay-deterministic from `state()` / `from_state()`.

Run::

    python -m examples.bandit_demo
"""

from __future__ import annotations

import random

from agi.bandit import (
    EXP3,
    EXP3_IX,
    IDS,
    KL_UCB,
    LINUCB,
    REWARD_BERNOULLI,
    REWARD_GAUSSIAN,
    THOMPSON_BETA,
    TSALLIS_INF,
    UCB1,
    Bandit,
)


def stationary_demo() -> None:
    """Run seven stochastic / adversarial algorithms head-to-head on a
    common 4-armed Bernoulli problem with means (0.5, 0.6, 0.7, 0.8).
    """
    means = {"a": 0.5, "b": 0.6, "c": 0.7, "d": 0.8}
    T = 5_000
    algos = [UCB1, KL_UCB, THOMPSON_BETA, IDS, EXP3, EXP3_IX, TSALLIS_INF]
    print(f"{'algorithm':<24} {'cumul':>9} {'regret':>9} {'best':>6} "
          f"{'pulls(d)':>10}  {'fingerprint':>12}")
    print("-" * 84)
    optimal_mean = max(means.values())
    for algo in algos:
        bandit = Bandit(
            arms=list(means.keys()), algorithm=algo, seed=42,
            reward_model=REWARD_BERNOULLI,
        )
        rng = random.Random(7)
        for _ in range(T):
            x = bandit.select_arm()
            r = 1.0 if rng.random() < means[x] else 0.0
            bandit.observe(x, r)
        rep = bandit.report(delta_bound=0.05)
        cumul = bandit.cumulative_reward
        regret = T * optimal_mean - cumul
        d_pulls = next(a.n_pulls for a in rep.arms if a.name == "d")
        fp_short = rep.fingerprint.split(":")[1][:10]
        print(
            f"{algo:<24} {cumul:>9.0f} {regret:>9.1f} "
            f"{rep.best_arm_so_far:>6} {d_pulls:>10}  {fp_short:>12}"
        )


def contextual_demo() -> None:
    """Run LinUCB on a 2-armed linear contextual bandit, d=3."""
    print()
    print("LinUCB on a d=3 contextual problem:")
    print("-" * 60)
    bandit = Bandit(
        arms=["a", "b"], algorithm=LINUCB, d=3, seed=0,
        reward_model=REWARD_GAUSSIAN, sigma=0.1, alpha=1.0, lam=1.0,
    )
    theta = {"a": [1.0, 0.5, -0.3], "b": [-0.5, 0.8, 0.2]}
    rng = random.Random(11)
    T = 2_000
    optimal_reward = 0.0
    realised_reward = 0.0
    for _ in range(T):
        ctx = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        x = bandit.select_arm(context=ctx)
        # True linear reward + noise.
        def lin(arm: str) -> float:
            return sum(theta[arm][i] * ctx[i] for i in range(3))
        chosen_r = lin(x) + rng.gauss(0.0, 0.1)
        bandit.observe(x, chosen_r, context=ctx)
        optimal_reward += max(lin(a) for a in theta)
        realised_reward += lin(x)
    rep = bandit.report()
    regret = optimal_reward - realised_reward
    print(f"pulls           : {rep.n_pulls}")
    print(f"cumul. reward   : {rep.cumulative_reward:.2f}")
    print(f"regret (true)   : {regret:.2f}")
    print(f"regret bound 95%: {rep.regret_upper_bound_95:.2f}")
    print(f"theoretical UB  : {rep.pseudo_regret_upper:.2f}")
    print(f"fingerprint     : {rep.fingerprint[:32]}...")


def replay_demo() -> None:
    """Demonstrate replay determinism via state() / from_state()."""
    print()
    print("Replay determinism demo:")
    print("-" * 60)
    bandit = Bandit(arms=["a", "b", "c"], algorithm=THOMPSON_BETA, seed=42)
    rng = random.Random(7)
    for _ in range(50):
        x = bandit.select_arm()
        r = 1.0 if rng.random() < (0.3 if x == "a" else 0.7) else 0.0
        bandit.observe(x, r)
    fp = bandit.fingerprint()
    snapshot = bandit.state()

    # Recreate and replay.
    replay = Bandit.from_state(snapshot)
    print(f"original fp   : {fp[:32]}...")
    print(f"replay   fp   : {replay.fingerprint()[:32]}...")
    print(f"match         : {fp == replay.fingerprint()}")
    print(f"cumul match   : "
          f"{bandit.cumulative_reward == replay.cumulative_reward}")


if __name__ == "__main__":
    stationary_demo()
    contextual_demo()
    replay_demo()
