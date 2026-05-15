"""Hedger — universal prediction with experts / online learning as a
runtime primitive.

Run this from the repo root:

    python examples/hedger_demo.py

Scenario
--------

A coordination engine has *three* candidate decision primitives wired
into its dispatch surface:

  * ``bandit``     — a Thompson-sampling Bandit on a discrete action set.
  * ``bayesopt``   — a Bayesian-optimisation surrogate on the same set.
  * ``thompson``   — a vanilla Bernoulli Thompson Sampler.

On any one task it cannot tell *a priori* which primitive is best — the
right choice depends on the unknown reward profile, on whether the
losses are stochastic or adversarial, on the horizon, and on the
particular distribution the universe happens to have rolled.

The Hedger primitive is the runtime mechanism that makes this
*automatic*: register the three primitives as Hedger experts, observe
their realised per-round losses, and the Hedger maintains a weight
distribution over experts whose cumulative loss tracks the *best fixed
expert in hindsight* up to a vanishing per-round regret — with an
anytime-valid closed-form regret bound the coordination engine can
audit before committing to a decision.

We compare three algorithms head-to-head:

  1. **Hedge** (Vovk 1990; Freund-Schapire 1997) with the Vovk-1990
     minimax-optimal learning rate ``η = √(8 log N / T)``.  Bound:
     ``R_T ≤ √(T log N / 2)``.

  2. **AdaHedge** (de Rooij-van Erven-Grünwald-Koolen 2014) —
     parameter-free.  Adapts ``η_t`` to the realised mixability gap.
     Bound: ``R_T ≤ 2 √(V_T log N) + O(log N)``.

  3. **NormalHedge** (Chaudhuri-Freund-Hsu 2009) — anytime,
     parameter-free, no learning rate to set.  Bound: ``R_T(rank=1) ≤
     √(2 T log N)``.

For each algorithm we print

  * the realised cumulative weighted loss;
  * the cumulative loss of the best fixed expert in hindsight;
  * the closed-form first-order regret upper bound;
  * the realised regret (cum_alg − cum_best);
  * the PAC-Bayes regret bound against the uniform prior;
  * the per-expert anytime-valid LCB / UCB on the mean loss.

Each of these is hash-chained into an ``AttestationLedger``-compatible
fingerprint, replay-deterministic given the seed.
"""
from __future__ import annotations

import random

from agi.hedger import (
    ADAHEDGE,
    HEDGE,
    Hedger,
    NORMAL_HEDGE,
    hedge_minimax_eta,
)


def simulate(algorithm: str, T: int = 500, *,
             means: dict[str, float] | None = None,
             seed: int = 0) -> dict:
    """Run a Hedger against IID Bernoulli losses for T rounds."""
    if means is None:
        means = {"bandit": 0.40, "bayesopt": 0.20, "thompson": 0.50}
    experts = list(means.keys())
    h = Hedger.create(
        experts,
        algorithm=algorithm,
        # Hedge needs the horizon for the minimax-optimal η; AdaHedge
        # and NormalHedge are parameter-free.
        horizon=T if algorithm == HEDGE else None,
        seed=seed,
    )
    rng = random.Random(seed + 1)
    for _ in range(T):
        # Per-round per-expert Bernoulli loss draws.
        losses = {
            e: 1.0 if rng.random() < means[e] else 0.0
            for e in experts
        }
        h.observe(losses)
    rep = h.report()
    rc = rep.regret_certificate
    return {
        "algorithm": algorithm,
        "T": rep.T,
        "cum_alg": rep.cumulative_weighted_loss,
        "best_expert": rc.best_expert,
        "cum_best": rc.best_cumulative_loss,
        "first_order_bound": rc.first_order_bound,
        "second_order_bound": rc.second_order_bound,
        "pac_bayes_bound": rc.pac_bayes_bound_uniform,
        "realised_regret": rc.realised_regret_so_far,
        "last_weights": rep.last_prediction.weights if rep.last_prediction else {},
        "last_kl_from_prior": (rep.last_prediction.realised_kl_from_prior
                                if rep.last_prediction else 0.0),
        "fingerprint": rep.fingerprint,
        "h": h,
    }


def fmt(d: dict, key: str, fmt_str: str = "{:.4f}") -> str:
    v = d.get(key)
    if v is None:
        return "—"
    return fmt_str.format(v) if isinstance(v, float) else str(v)


def main() -> None:
    T = 500
    print("=" * 72)
    print(f"Hedger demo — 3 experts, T={T} rounds, IID Bernoulli losses")
    print("  means: bandit=0.40, bayesopt=0.20 (best), thompson=0.50")
    print("=" * 72)

    for algo in [HEDGE, ADAHEDGE, NORMAL_HEDGE]:
        r = simulate(algo, T=T)
        print()
        print(f"-- {algo.upper()} --")
        print(f"  cumulative loss (algorithm): "
              f"{r['cum_alg']:.2f} = {r['cum_alg']/T:.4f} / round")
        print(f"  cumulative loss (best fixed expert {r['best_expert']!r}): "
              f"{r['cum_best']:.2f} = {r['cum_best']/T:.4f} / round")
        print(f"  realised regret: {r['realised_regret']:.3f}")
        print(f"  first-order regret upper bound (Vovk-1990 / paper-specific): "
              f"{r['first_order_bound']:.3f}")
        if r['second_order_bound'] is not None:
            print(f"  second-order bound (paper-specific): "
                  f"{r['second_order_bound']:.3f}")
        print(f"  PAC-Bayes regret bound (uniform prior): "
              f"{r['pac_bayes_bound']:.3f}")
        # Show the final weight distribution.
        ws = r['last_weights']
        ws_str = ", ".join(f"{e}={ws[e]:.4f}" for e in sorted(ws))
        print(f"  final weights: {ws_str}")
        print(f"  realised KL from uniform prior: "
              f"{r['last_kl_from_prior']:.4f} nats")
        print(f"  fingerprint: {r['fingerprint'][:16]}…  "
              "(SHA-256, chain into AttestationLedger)")

        # Anytime-valid LCB / UCB on each expert's mean loss.
        print("  per-expert anytime-valid 95 % confidence intervals "
              "(Howard-Ramdas-McAuliffe-Sekhon 2021):")
        h = r['h']
        for e in sorted(["bandit", "bayesopt", "thompson"]):
            cert = h.per_expert_certificate(e, delta=0.05)
            print(f"    {e:10s} mean={cert.mean_loss:.4f}  "
                  f"anytime CI=[{cert.anytime_lcb:.4f}, {cert.anytime_ucb:.4f}]  "
                  f"bernstein CI=[{cert.bernstein_lcb:.4f}, "
                  f"{cert.bernstein_ucb:.4f}]")

    # -----------------------------------------------------------------
    # Coordination-engine smoke: drift detection via mixability gap
    # -----------------------------------------------------------------
    print()
    print("=" * 72)
    print("Coordination-engine smoke: non-stationary regime change at t=250")
    print("=" * 72)
    h = Hedger.create(["bandit", "bayesopt", "thompson"],
                       algorithm=ADAHEDGE, seed=0)
    rng = random.Random(99)
    pre = {"bandit": 0.4, "bayesopt": 0.2, "thompson": 0.5}    # bayesopt best
    post = {"bandit": 0.2, "bayesopt": 0.6, "thompson": 0.5}   # bandit best
    midpoint = 250
    gaps = []
    for t in range(500):
        means_now = pre if t < midpoint else post
        losses = {e: 1.0 if rng.random() < means_now[e] else 0.0
                   for e in means_now}
        rd = h.observe(losses)
        gaps.append((t, rd.delta_mixability_gap, rd.best_expert,
                      rd.cumulative_mixability_gap))

    pre_avg = sum(g[1] for g in gaps[:midpoint]) / midpoint
    post_avg = sum(g[1] for g in gaps[midpoint:]) / (500 - midpoint)
    print(f"  mean mixability gap before regime change (t<{midpoint}): "
          f"{pre_avg:.4f}")
    print(f"  mean mixability gap after regime change  (t≥{midpoint}): "
          f"{post_avg:.4f}")
    print("  (a sustained increase in δ_t is a drift signal — feed it "
          "into DriftSentinel)")

    rep = h.report()
    print(f"  final cumulative regret bound: "
          f"{rep.regret_certificate.first_order_bound:.3f}")
    print(f"  realised regret: {rep.regret_certificate.realised_regret_so_far:.3f}")

    # -----------------------------------------------------------------
    # Composition with Quantilizer: KL-bounded version of the Hedger
    # -----------------------------------------------------------------
    print()
    print("=" * 72)
    print("Composition with Quantilizer — safety-bounded Hedger weights")
    print("=" * 72)
    try:
        from agi.quantilizer import quantilize_discrete
        # Run a short hedger so the weights are not degenerate.
        h = Hedger.create(["bandit", "bayesopt", "thompson"],
                           algorithm=ADAHEDGE, seed=0)
        rng = random.Random(7)
        for _ in range(20):
            losses = {"bandit": 1.0 if rng.random() < 0.4 else 0.0,
                       "bayesopt": 1.0 if rng.random() < 0.2 else 0.0,
                       "thompson": 1.0 if rng.random() < 0.5 else 0.0}
            h.observe(losses)
        weights = h.predict().weights

        # Treat negative cumulative loss as utility for quantilizing.
        cum = h.cumulative_losses
        utility = {e: -cum[e] for e in weights}
        # Use a base distribution = current Hedger weights, q = 0.5.
        qd = quantilize_discrete(weights, utility, q=0.5)
        print(f"  raw Hedger weights : "
              f"{ {k: round(v, 4) for k, v in weights.items()} }")
        print(f"  q=0.5-quantilized  : "
              f"{ {k: round(v, 4) for k, v in qd.probs.items()} }  "
              f"KL={qd.realised_kl:.4f} ≤ {qd.kl_bound:.4f} (bound log(1/q))")
        print(f"  cost amplification : {qd.cost_amplification:.1f}× "
              "(worst-case hidden cost vs base)")
        print("  → coordination engine gets a runtime knob trading off "
              "regret rate against safety drift")
    except Exception as exc:
        print(f"  composition demo skipped: {exc}")


if __name__ == "__main__":
    main()
