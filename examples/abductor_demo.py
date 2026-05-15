"""Abductor demo — Bayesian abductive inference / inference to the best
explanation as a runtime primitive.

Three scenarios, each demonstrating a different facet of the primitive:

  1. Coin diagnosis — point hypotheses vs a conjugate Beta-Bernoulli.
  2. Source identification — three competing Poisson sources for an
     observed count stream, with prior robustness + e-process.
  3. Active abduction — pick the next experiment whose data maximally
     reduces posterior entropy over hypotheses.

Run with::

    python examples/abductor_demo.py
"""
from __future__ import annotations

import math

from agi.abductor import (
    Abductor,
    BERNOULLI_BETA,
    GAUSSIAN_KNOWN_VAR,
    POINT_BERNOULLI,
    POINT_POISSON,
    POISSON_GAMMA,
    jeffreys_label,
)


# =====================================================================
# 1. Coin diagnosis
# =====================================================================


def coin_diagnosis() -> None:
    print("=" * 72)
    print("[1] Coin diagnosis — three competing hypotheses, ten flips")
    print("=" * 72)
    abd = Abductor()
    abd.register("fair", POINT_BERNOULLI, p=0.5)
    abd.register("biased_strong", POINT_BERNOULLI, p=0.9)
    abd.register("biased_beta", BERNOULLI_BETA, alpha=2.0, beta=2.0)
    data = [1] * 9 + [0]  # 9 heads, 1 tail
    print(f"  observed: {data}")
    abd.observe(data)
    post = abd.posterior()
    print("\n  Posterior over hypotheses:")
    for name, prob in sorted(post.posterior_probs().items(),
                             key=lambda kv: -kv[1]):
        ev = abd.log_evidence(name)
        print(f"    {name:>16}: p(H|D) = {prob:.4f}   log p(D|H) = {ev:+.3f}")
    sel = abd.select()
    print(f"\n  MAP pick: {sel.winner!r}  (p(H|D) = {sel.posterior_prob:.3f})")
    print(f"  Bayes factor vs runner-up '{sel.runner_up}':")
    print(f"    log10 BF = {sel.log10_bayes_factor:.2f}  → '{sel.jeffreys_label}'"
          f" on Jeffreys's scale")
    print(f"  fingerprint: {abd.fingerprint[:16]}…")


# =====================================================================
# 2. Source identification with prior robustness + e-process
# =====================================================================


def source_identification() -> None:
    print("\n" + "=" * 72)
    print("[2] Source identification — three Poisson sources")
    print("=" * 72)
    abd = Abductor()
    abd.register("low_rate", POINT_POISSON, lam=2.0)
    abd.register("mid_rate", POINT_POISSON, lam=5.0)
    abd.register("high_rate", POINT_POISSON, lam=12.0)
    abd.register("uncertain", POISSON_GAMMA, alpha=2.0, beta=0.5)
    data = [5, 6, 4, 7, 5, 6, 4, 8, 5, 6]
    print(f"  observed counts: {data}")
    abd.observe(data)
    post = abd.posterior()
    print("\n  Posterior over sources:")
    for name, prob in sorted(post.posterior_probs().items(),
                             key=lambda kv: -kv[1]):
        print(f"    {name:>12}: p(H|D) = {prob:.4f}")

    print("\n  Anytime-valid e-process (mid_rate vs low_rate, α=0.05):")
    e = abd.e_process("mid_rate", "low_rate", delta=0.05)
    print(f"    log-likelihood ratio = {e.log_e:+.2f}")
    print(f"    e-value              = {e.e_value:.3g}")
    print(f"    Ville threshold      = {e.threshold_log:.2f}  → ", end="")
    if e.crossed_at is not None:
        print(f"REJECTED low_rate at n={e.crossed_at} (type-I controlled at α=0.05)")
    else:
        print("not yet crossed — need more data")

    print("\n  Prior robustness:")
    rob = abd.prior_robustness()
    print(f"    current MAP winner = {rob.current_winner!r}")
    print(f"    breaking-point KL  = {rob.max_kl_perturbation:.4f}  "
          f"(perturbing prior toward {rob.breaking_runner_up!r})")

    print("\n  Identifiability:")
    ident = abd.identifiability()
    print(f"    equivalence classes  = {ident.classes}")
    print(f"    min pairwise gap     = {ident.min_pairwise_log_evidence_gap:.4f} nats")


# =====================================================================
# 3. Active abduction — pick the next experiment
# =====================================================================


def active_abduction() -> None:
    print("\n" + "=" * 72)
    print("[3] Active abduction — design the next experiment")
    print("=" * 72)
    abd = Abductor()
    abd.register("sensor_a", POINT_BERNOULLI, p=0.5)  # 50/50
    abd.register("sensor_b", POINT_BERNOULLI, p=0.6)  # mild bias to 1
    abd.register("sensor_c", POINT_BERNOULLI, p=0.9)  # strong bias to 1
    # No observations yet — purely the prior-driven design.
    print("  Three candidate sample spaces:")
    candidates = {
        "single_binary_flip":   [0, 1],
        "fair_coin_oracle":     [1],          # never discriminates
        "tampered_thumbtack":   [0, 0, 1, 1],  # multiple obs in one shot
    }
    for name, space in candidates.items():
        ig = abd.expected_information_gain(space)
        print(f"    {name:<22}: EIG = {ig.expected_gain_nats:+.4f} nats   "
              f"(H_now = {ig.posterior_entropy_now:.3f})")
    best, ig = abd.design_next_experiment(candidates)
    print(f"\n  → best next experiment: {best!r}  (EIG = {ig.expected_gain_nats:.3f} nats)")

    # Show how observation drives posterior collapse.
    print("\n  Streaming five additional observations:")
    for x in [1, 1, 1, 1, 0]:
        abd.observe([x])
        post = abd.posterior()
        probs = post.posterior_probs()
        print(f"    after {abd._n_obs} obs: " +  # type: ignore[attr-defined]
              " ".join(f"{n}={p:.3f}" for n, p in probs.items()))


# =====================================================================
# 4. Counterfactual + contrastive explanation
# =====================================================================


def contrastive_explanation() -> None:
    print("\n" + "=" * 72)
    print("[4] Contrastive explanation — 'why H and not H?'")
    print("=" * 72)
    abd = Abductor()
    abd.register("normal_op", POINT_POISSON, lam=3.0)
    abd.register("anomalous", POINT_POISSON, lam=8.0)
    obs = [4, 5, 3, 7, 6, 4, 9, 5]
    print(f"  observed event counts per second: {obs}")
    abd.observe(obs)
    sel = abd.select()
    print(f"  MAP pick: {sel.winner!r}")
    c = abd.contrastive("anomalous", "normal_op", data=obs)
    print(f"\n  Per-observation log-Bayes-factor (anomalous : normal_op):")
    for i, (x, lbf) in enumerate(zip(obs, c.per_obs_log_bf), start=1):
        bar = ">" * max(int(lbf * 5), 0) if lbf > 0 else "<" * max(int(-lbf * 5), 0)
        print(f"    obs {i:>2} (x={x}): Δ log B = {lbf:+.2f}  {bar}")
    print(f"  Cumulative log B at end: {c.final_log_bf:+.2f}")

    print("\n  Counterfactual: 'what if we had observed all 1s?'")
    cf = abd.counterfactual_posterior([1] * 8)
    cf_probs = {n: math.exp(lp) for n, lp in zip(cf.names, cf.log_posteriors)}
    for name, prob in sorted(cf_probs.items(), key=lambda kv: -kv[1]):
        print(f"    {name:>12}: p(H|D') = {prob:.4f}")


if __name__ == "__main__":
    coin_diagnosis()
    source_identification()
    active_abduction()
    contrastive_explanation()
    print("\n" + "=" * 72)
    print("All Abductor demos completed successfully.")
    print("=" * 72)
