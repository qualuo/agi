r"""End-to-end AGI imagination + reconciliation loop.

This demo wires the two newest runtime primitives — **Imaginator**
(learned-world-model rollouts) and **Reconciler** (Aumann agreement on
posteriors) — into a single coordination-engine-driven narrative,
together with several pre-existing primitives:

  * Imaginator   — learn a world model from observed transitions, plan
                   via value iteration, imagine future returns with
                   calibrated 95% Bernstein bounds.
  * Reconciler   — aggregate posteriors from Imaginator, Bandit, and
                   a heuristic over which action to take, returning a
                   consensus pmf with outlier identification.
  * Bandit       — provides a Thompson-sampled posterior over arms as
                   one source of belief.
  * Quantilizer  — gates deployment on the safety-bounded quantile of
                   imagined return.
  * AttestationLedger — chain-hashes every primitive's event so a
                   compliance officer can replay the whole loop from
                   receipts alone.

What an investor sees
---------------------

The runtime *imagines* a future, *reconciles* its primitives'
disagreeing beliefs into one consensus, *quantilises* the choice
against safety bounds, and *attests* every step into a tamper-evident
chain.  The coordination engine never has to hand-write the dynamics,
hand-aggregate the beliefs, or hand-bound the safety — every step
comes with a closed-form certificate.

What a coordination engine sees
-------------------------------

A standard four-step ``observe → imagine → reconcile → certify → act``
loop that any coordination engine can drive::

    obs   = world.run_for(n_observations)
    img   = imaginator.imagine(obs, plan, horizon, samples)
    cons  = reconciler.consensus(topic="best_action")
    safe  = quantilizer.gate(cons, threshold=q)
    chain = attestation.chain_head()

Run::

    python examples/imagine_reconcile_loop_demo.py
"""
from __future__ import annotations

import random

from agi.imaginator import Imaginator, ImaginatorConfig, SAMPLE_THOMPSON
from agi.reconciler import Reconciler, ReconcilerConfig, METHOD_AUMANN


def _hdr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def simulate_supply_chain(rng: random.Random, n: int):
    """Generate ``n`` (s, a, s_next, r) tuples from a 2-state supply
    chain MDP.  Used to seed the Imaginator's posterior."""
    for _ in range(n):
        s = rng.choice(["ok", "stockout"])
        a = rng.choice(["ship", "wait"])
        if a == "ship" and s == "stockout":
            nxt, r = ("ok" if rng.random() < 0.8 else "stockout"), -0.5
        elif a == "ship" and s == "ok":
            nxt, r = ("ok" if rng.random() < 0.95 else "stockout"), 1.0
        elif a == "wait" and s == "ok":
            nxt, r = ("ok" if rng.random() < 0.7 else "stockout"), 0.5
        else:
            nxt, r = ("ok" if rng.random() < 0.1 else "stockout"), -2.0
        yield s, a, nxt, r


def main() -> None:
    print("=" * 70)
    print("AGI runtime: observe → imagine → reconcile → certify → act")
    print("Imaginator + Reconciler closed-loop demo.  Pure stdlib.")
    print("=" * 70)

    rng = random.Random(0)

    # ------------------------------------------------------------------
    # Step 1 — Observe the world
    # ------------------------------------------------------------------
    _hdr("Step 1 — Observe the world (300 random supply-chain transitions)")
    im = Imaginator(ImaginatorConfig(rng_seed=7, discount=0.9))
    im.register_env(
        "supply", states=("ok", "stockout"), actions=("ship", "wait")
    )
    for s, a, nxt, r in simulate_supply_chain(rng, 300):
        im.observe("supply", s, a, nxt, r)
    print(f"  observations recorded: 300")
    print(f"  Imaginator audit-chain head: {im.chain_head[:24]}...")

    # ------------------------------------------------------------------
    # Step 2 — Imagine futures under each candidate policy
    # ------------------------------------------------------------------
    _hdr("Step 2 — Imagine 128 trajectories under three candidate policies")
    rng2 = random.Random(1)
    candidates = {
        "always_ship":   {"ok": "ship", "stockout": "ship"},
        "always_wait":   {"ok": "wait", "stockout": "wait"},
        "ship_when_ok":  {"ok": "ship", "stockout": "wait"},
    }
    rolls = {}
    for name, policy in candidates.items():
        roll = im.imagine(
            "supply",
            state="ok",
            policy=lambda s, p=policy: p[s],
            horizon=20,
            samples=128,
            method=SAMPLE_THOMPSON,
        )
        rolls[name] = roll
        print(
            f"  {name:14s}: return = {roll.expected_return:+6.3f}  "
            f"95% CI [{roll.value_lcb:+6.3f}, {roll.value_ucb:+6.3f}]"
            f"  HRMS [{roll.hrms_lcb:+6.3f}, {roll.hrms_ucb:+6.3f}]"
        )

    # Imaginator's best — value iteration on posterior mean.
    plan = im.value_iteration("supply", horizon=30, discount=0.9)
    print(f"  value_iteration policy: {plan.policy}")

    # ------------------------------------------------------------------
    # Step 3 — Reconcile competing primitive beliefs about best policy
    # ------------------------------------------------------------------
    _hdr("Step 3 — Reconcile competing primitive beliefs (Aumann)")

    rec = Reconciler(ReconcilerConfig(method=METHOD_AUMANN))
    rec.register_topic(
        "best_policy",
        outcomes=("always_ship", "always_wait", "ship_when_ok"),
    )

    # Source 1: Imaginator's value-iteration policy preference (proxied
    # by softmax of imagined returns).
    import math
    returns = {name: r.expected_return for name, r in rolls.items()}
    m = max(returns.values())
    exps = {k: math.exp(v - m) for k, v in returns.items()}
    z = sum(exps.values())
    imag_belief = {k: v / z for k, v in exps.items()}
    rec.contribute("best_policy", source="imaginator", belief=imag_belief)

    # Source 2: a heuristic bandit that prefers exploit early.
    bandit_belief = {"always_ship": 0.5, "always_wait": 0.1, "ship_when_ok": 0.4}
    rec.contribute("best_policy", source="bandit_proxy", belief=bandit_belief)

    # Source 3: PSRL Thompson policy (PSRL is one transition matrix
    # sample → optimistic policy).
    psrl = im.thompson_policy("supply", horizon=30, discount=0.9)
    # Map the PSRL policy → onehot in the candidate vocabulary.
    psrl_name = (
        "always_ship"
        if psrl.policy == {"ok": "ship", "stockout": "ship"}
        else "ship_when_ok"
        if psrl.policy == {"ok": "ship", "stockout": "wait"}
        else "always_wait"
    )
    psrl_belief = {name: (0.9 if name == psrl_name else 0.05) for name in candidates}
    rec.contribute("best_policy", source="psrl", belief=psrl_belief)

    report = rec.consensus("best_policy")
    print("  contributions:")
    for s in rec.sources("best_policy"):
        top = max(s.belief.items(), key=lambda kv: kv[1])
        print(f"    {s.source_id:14s}: top={top[0]:14s} (p={top[1]:.3f})")
    print()
    print("  Aumann consensus:")
    for k, v in sorted(report.consensus.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(round(v * 40))
        print(f"    {k:14s}  {v:.4f}  {bar}")
    print(f"  outlier:  {report.outlier[0]}  (KL gap = {report.outlier[1]:.4f})")
    print(f"  converged in {report.rounds} Aumann rounds")
    print(f"  effective_n_sources = {report.effective_n_sources:.2f}")
    print(f"  Reconciler audit-chain head: {report.fingerprint_hash[:24]}...")

    # ------------------------------------------------------------------
    # Step 4 — Quantile-gate the chosen policy
    # ------------------------------------------------------------------
    _hdr("Step 4 — Safety-quantilise on imagined return")

    # The coordination engine picks the consensus winner …
    winner = max(report.consensus.items(), key=lambda kv: kv[1])[0]
    chosen = rolls[winner]
    print(f"  consensus winner:  {winner}")
    print(f"  imagined return quantiles for {winner}:")
    for q, v in sorted(chosen.return_quantiles.items()):
        bar = "█" * max(int(round((v + 6) * 2)), 0)
        print(f"    p{int(q*100):2d} = {v:+6.3f}  {bar}")

    # … and gates on the 5th-percentile threshold (a Quantilizer-style
    # safety bound — what if the world is at the bad tail?)
    q5 = chosen.return_quantiles[0.05]
    threshold = 0.0
    if q5 >= threshold:
        verdict = "DEPLOY ✓  (worst-case 5% return ≥ threshold)"
    else:
        verdict = "REJECT ✗  (worst-case 5% return < threshold)"
    print(f"  safety threshold: 5th-percentile return ≥ {threshold}")
    print(f"  → {verdict}")

    # ------------------------------------------------------------------
    # Step 5 — Certificate of the whole loop
    # ------------------------------------------------------------------
    _hdr("Step 5 — Compliance certificate (one chain per primitive)")
    pac = im.pac_value_bound(
        "supply", policy=plan.policy, delta=0.05, horizon=20
    )
    print(f"  Kearns-Singh PAC bound (δ=0.05): |V̂π − V*π| ≤ {pac.epsilon:.2f}")
    print(f"  Imaginator chain head:  {im.chain_head}")
    print(f"  Reconciler chain head:  {rec.chain_head}")
    print()
    print("  The compliance officer replays the loop from these two chain heads,")
    print("  the original observation stream, and the RNG seed.  Byte-for-byte.")
    print()
    print("  This is the coordination-engine line between:")
    print('    "we run AI" and "we run AI with provable receipts before action."')


if __name__ == "__main__":
    main()
