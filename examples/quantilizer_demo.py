"""Quantilizer — safety-bounded optimisation as a runtime primitive.

Run this from the repo root:

    python examples/quantilizer_demo.py

Scenario
--------

A coordination engine has to pick one of 100 candidate plans for a
high-stakes task.  Each plan is rated by an LLM-as-judge proxy utility
``U_proxy``.  The truth is that ``U_proxy`` correlates well with the
true outcome ``U_true`` on most plans — but a single adversarial plan
(``trap``) has the maximum proxy score while having the worst true
outcome.  This is the Goodhart trap (Manheim & Garrabrant 2018).

Three optimisers are compared:

  1. **Argmax**            — pick ``argmax U_proxy``.  Falls into the
                             trap.  Expected true utility: ~0.
  2. **Quantilizer (q=0.1)** — sample from the top-10 % of the base
                             distribution by ``U_proxy``.  KL bound
                             from base: ``log(10) ≈ 2.3 nats``.  Cost
                             amplification: 10×.  Expected true
                             utility: ≈ 0.9.
  3. **Soft quantilizer (KL budget = 1.0 nats)** — Boltzmann shift of
                             the base distribution by the proxy with β
                             chosen so the KL from base lands exactly
                             at 1.0.  A *continuous* dial on the
                             safety / performance trade-off.

The demo prints, for each optimiser:

  * the safety certificate (KL bound, TV bound, cost amplification);
  * the expected proxy utility (what the optimiser thinks);
  * the expected true utility (what actually happens);
  * the realised quantile threshold the optimiser landed on.

Then it runs 200 sequential draws from the q=0.1 quantilizer and asks
the Quantilizer for an anytime-valid LCB on the realised true utility
(Howard-Ramdas-McAuliffe-Sekhon 2021), demonstrating the *replay-
verifiable* receipt chain a coordinator gets out of the runtime.
"""
from __future__ import annotations

import random

from agi.quantilizer import (
    ANYTIME,
    HARD,
    SOFT,
    Quantilizer,
    quantilize_discrete,
    soft_quantilize,
)


def build_scenario(n: int = 100, seed: int = 7):
    """Build a 100-atom base distribution + proxy + true utilities.

    The Goodhart trap is atom 0: highest proxy score, lowest truth.
    Atoms 1..99 are well-aligned (proxy correlated with truth).
    """
    rng = random.Random(seed)
    base = {f"plan_{i}": 1.0 / n for i in range(n)}
    proxy = {}
    truth = {}
    for i in range(n):
        if i == 0:
            proxy[f"plan_{i}"] = 10.0      # the trap: max proxy
            truth[f"plan_{i}"] = 0.0       # but minimum truth
        else:
            # Plans 1..99: proxy and truth correlate (noisy)
            p = rng.uniform(0.0, 1.0)
            proxy[f"plan_{i}"] = p
            truth[f"plan_{i}"] = max(0.0, p + rng.gauss(0.0, 0.05))
    return base, proxy, truth


def main() -> None:
    base, proxy, truth = build_scenario()

    # -----------------------------------------------------------------
    # 1. Argmax baseline (q → 0, the unsafe limit)
    # -----------------------------------------------------------------

    d_argmax = quantilize_discrete(base, proxy, 0.01)
    pick_argmax = next(iter(d_argmax.probs.keys()))
    eu_proxy_arg = sum(p * proxy[a] for a, p in d_argmax.probs.items())
    eu_true_arg = sum(p * truth[a] for a, p in d_argmax.probs.items())

    print("=" * 72)
    print("ARGMAX optimiser (q = 0.01 — effectively pure exploitation):")
    print(f"  picked: {pick_argmax}")
    print(f"  KL from base bound: {d_argmax.kl_bound:.3f} nats "
          f"(= log(100) — maximum)")
    print(f"  cost amplification: {d_argmax.cost_amplification:.1f}×")
    print(f"  TV bound from base: {d_argmax.tv_bound:.3f}")
    print(f"  proxy expected utility: {eu_proxy_arg:.4f}")
    print(f"  TRUE  expected utility: {eu_true_arg:.4f}   <-- Goodhart trap")

    # -----------------------------------------------------------------
    # 2. Hard quantilizer at q = 0.1
    # -----------------------------------------------------------------

    d_q10 = quantilize_discrete(base, proxy, 0.1)
    eu_proxy_q10 = sum(p * proxy[a] for a, p in d_q10.probs.items())
    eu_true_q10 = sum(p * truth[a] for a, p in d_q10.probs.items())

    print("=" * 72)
    print("HARD QUANTILIZER (q = 0.1):")
    print(f"  retained: {d_q10.n_kept} / {d_q10.n_support} atoms")
    print(f"  KL from base bound: {d_q10.kl_bound:.3f} nats (= log(10))")
    print(f"  KL realised:        {d_q10.realised_kl:.3f} nats")
    print(f"  cost amplification: {d_q10.cost_amplification:.1f}× "
          f"(at worst)")
    print(f"  TV bound from base: {d_q10.tv_bound:.3f}")
    print(f"  proxy expected utility: {eu_proxy_q10:.4f}")
    print(f"  TRUE  expected utility: {eu_true_q10:.4f}   <-- Goodhart resisted")

    # -----------------------------------------------------------------
    # 3. Soft (Boltzmann) quantilizer with KL = 1.0 nats
    # -----------------------------------------------------------------

    d_soft = soft_quantilize(base, proxy, kl_budget=1.0)
    eu_proxy_soft = sum(p * proxy[a] for a, p in d_soft.probs.items())
    eu_true_soft = sum(p * truth[a] for a, p in d_soft.probs.items())

    print("=" * 72)
    print("SOFT QUANTILIZER (Boltzmann, KL budget = 1.0 nat):")
    print(f"  KL from base bound: {d_soft.kl_bound:.3f} nats")
    print(f"  KL realised:        {d_soft.realised_kl:.3f} nats")
    print(f"  effective q       : {d_soft.q:.4f}")
    print(f"  effective cost ×  : {d_soft.cost_amplification:.2f}×")
    print(f"  proxy expected utility: {eu_proxy_soft:.4f}")
    print(f"  TRUE  expected utility: {eu_true_soft:.4f}")

    # -----------------------------------------------------------------
    # 4. Replay-verifiable receipt chain with anytime-valid LCB
    # -----------------------------------------------------------------

    print("=" * 72)
    print("REPLAY-VERIFIABLE LEDGER + anytime-valid LCB on true utility:")
    Q = Quantilizer(q=0.1, default_lcb_method=ANYTIME, default_delta=0.05)
    rng = random.Random(31)
    for trial in range(200):
        s = Q.select(base, proxy, seed=trial)
        true_u = truth[s.action] + rng.gauss(0.0, 0.02)
        true_u = max(0.0, min(1.0, true_u))
        Q.observe(s.action, true_u)
    b = Q.expected_utility_lcb()
    r = Q.report()
    print(f"  selections: {r.n_selections}")
    print(f"  observations: {r.n_observations}")
    print(f"  realised KL cumulative: {r.cumulative_kl_bound:.3f} nats "
          f"(≤ {r.n_selections * 2.302585:.1f} bound)")
    print(f"  expected true utility (mean): {b.mean:.4f}")
    print(f"  anytime LCB (95 %): {b.lcb:.4f}")
    print(f"  anytime UCB (95 %): {b.ucb:.4f}")
    print(f"  fingerprint (last):  {r.fingerprint[:32]}…")
    print(f"  genesis fingerprint: {r.genesis[:32]}…")

    # -----------------------------------------------------------------
    # Cost UCB demo
    # -----------------------------------------------------------------

    print("=" * 72)
    print("COST UCB via Taylor-2016 amplification:")
    # Suppose a hidden-cost UCB on the base policy is 0.02 (say a
    # privacy violation rate).  The quantilizer with q=0.1 amplifies
    # this by 10×.
    for q_test in (1.0, 0.5, 0.1, 0.01):
        c = Q.cost_ucb(0.02, q=q_test)
        print(f"  q = {q_test:.2f}:  base UCB = {c.base_cost_ucb:.4f}  → "
              f"quantilizer UCB = {c.quantilizer_cost_ucb:.4f}  "
              f"(× {c.amplification:.1f})")


if __name__ == "__main__":
    main()
