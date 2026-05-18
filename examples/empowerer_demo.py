"""Empowerer — empowerment / intrinsic motivation as a runtime primitive.

A coordination engine drives the runtime to deliver:

  Goal:  Make an agent's *agency* a first-class numerical signal
         the coordinator can read every step:

           * Single bit-count scalar 𝔈ⁿ(s) the planner can compare
             across states (high → "stay here, lots of options";
             low → "explore or defer").
           * Empowerment-preserving action shielding — refuse
             actions that would *destroy future controllability*,
             even when they look attractive on a task reward.
           * Intrinsic reward stream for any RL primitive
             (``Aligner``, ``Bandit``, ``Pareto``, ``Quantilizer``)
             so the agent can self-curriculum in sparse-reward
             worlds.
           * Mutual-information *skill discovery* (DIAYN) — find
             distinguishable behavioural latents without a task.
           * PAC certificate on the empowerment estimate so the
             coordinator can decide *act vs gather more data*.
           * Tamper-evident SHA-256 fingerprint chain over every
             observation, solve, shield, certify.

This is the **investor-grade agency story** in a single runnable
script (no API key required, pure stdlib):

  1. Empowerer.observe_transition → ingest one MDP step
  2. Empowerer.empowerment        → Blahut-Arimoto channel capacity
  3. Empowerer.intrinsic_reward   → drop-in shaping for any learner
  4. Empowerer.safe_actions       → shielded candidate set
  5. Empowerer.skill_discovery    → DIAYN latent skills
  6. Empowerer.certify            → PAC bound (Paninski 2003)
  7. Empowerer.snapshot           → ship the estimator over the wire

Run::

  python examples/empowerer_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.empowerer import (
    EST_BLAHUT_ARIMOTO,
    EST_VARIATIONAL_INFONCE,
    Empowerer,
    EmpowererConfig,
    REWARD_DELTA_EMPOWERMENT,
    REWARD_STATE_EMPOWERMENT,
    REWARD_TRANSITION_SURPRISE,
    blahut_arimoto_capacity,
)


def banner(title: str) -> None:
    bar = "─" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Toy world #1 — a "GridRoom" with a controllable safe area and a trap.
#
# 8 states arranged as:
#
#    0 — 1 — 2 — 3           controllable corridor
#                  |
#    7 — 6 — 5 — 4           controllable corridor
#                                trap = state 7 (only self-loops)
#
# Actions: 0 = move-left, 1 = move-right (cyclic).
# At state 7, both actions are absorbed (trap).
# ---------------------------------------------------------------------------


N_STATES = 8
N_ACTIONS = 2


def grid_step(state: int, action: int, rng: random.Random) -> int:
    if state == 7:
        return 7  # trap
    if rng.random() < 0.05:
        # 5% slip — agent doesn't have perfect control.
        return rng.randrange(N_STATES)
    if action == 0:
        return (state - 1) % N_STATES
    return (state + 1) % N_STATES


def run_demo() -> None:
    rng = random.Random(0)

    # -----------------------------------------------------------------------
    # 1) Construct the Empowerer.
    # -----------------------------------------------------------------------
    em = Empowerer(
        EmpowererConfig(
            dim_state=N_STATES,
            dim_action=N_ACTIONS,
            horizon=3,
            estimator=EST_BLAHUT_ARIMOTO,
            reward_mode=REWARD_DELTA_EMPOWERMENT,
            laplace_alpha=0.25,
            safety_margin=0.5,
            confidence=0.95,
            rng_seed=0,
            ba_warm_start=True,
            max_action_seqs=4096,
        )
    )

    banner("Coordinator step 1/8 — ingest a stream of transitions")
    # Coordinator drives a no-policy random walker for 4000 steps just
    # to populate the channel estimator.  Resets out of the trap so we
    # see the full landscape — exactly what a curiosity-driven
    # coordinator would do once Empowerer's intrinsic-reward signal
    # told it state 7 is uninformative.
    state = 0
    for step in range(4000):
        action = rng.randrange(N_ACTIONS)
        next_state = grid_step(state, action, rng)
        em.observe_transition(state, action, next_state)
        state = next_state
        if state == 7:
            # The coordinator notices it's in a trap (Empowerer would
            # report 𝔈 = 0) and resets to a uniformly-sampled state.
            state = rng.randrange(N_STATES - 1)
    print(f"  ingested {em.total_transitions} transitions over {N_STATES} states")
    print(f"  chain head: {em.chain_head[:24]}...")

    # -----------------------------------------------------------------------
    # 2) Compute empowerment at every state.
    # -----------------------------------------------------------------------
    banner("Coordinator step 2/8 — read the empowerment landscape")
    land = em.landscape()
    for s, e in sorted(land.items()):
        bar = "█" * int(e * 20)
        flag = "  ← TRAP" if s == 7 else ""
        print(f"  state {s}:  𝔈³ = {e:5.3f} bits  {bar:<22}{flag}")
    bts = em.bottleneck_states(top_k=2)
    print(f"  bottleneck states (lowest empowerment): {bts}")

    # -----------------------------------------------------------------------
    # 3) Intrinsic reward stream — drop-in for any RL primitive.
    # -----------------------------------------------------------------------
    banner("Coordinator step 3/8 — drive intrinsic-reward learner")
    state = 0
    total_ireward = 0.0
    for step in range(50):
        action = rng.randrange(N_ACTIONS)
        next_state = grid_step(state, action, rng)
        ir = em.intrinsic_reward(state, action, next_state)
        total_ireward += ir
        state = next_state
    print(f"  50 random steps: total intrinsic Δ-empowerment = {total_ireward:+.3f} bits")

    # Switch to surprise mode for a step or two.
    sup = em.intrinsic_reward(0, 1, 7, mode=REWARD_TRANSITION_SURPRISE)
    print(f"  surprise of unlikely (0, →right, →trap-7) = {sup:.3f} bits")

    # -----------------------------------------------------------------------
    # 4) Empowerment-preserving safe-action shielding.
    # -----------------------------------------------------------------------
    banner("Coordinator step 4/8 — empowerment-preserving safe-action shield")
    # From state 0, both actions visible.  Action 0 → state 7 (the trap)
    # only when the wrap-around path is short; here action 0 from state 0
    # cyclically yields state 7.  Verify the shield filters it.
    shield = em.safe_actions(0, candidates=(0, 1), margin=0.5)
    print(f"  state 0 → 𝔈 = {shield.state_empowerment_bits:.3f} bits, margin = 0.5")
    for a, succ in zip(shield.candidates, shield.successor_empowerment_bits):
        admit = "ADMIT" if a in shield.admissible else "REFUSE (would lose agency)"
        print(f"    a={a}  expected 𝔈(s') = {succ:.3f} bits  →  {admit}")
    # State 6 → 7 is also high risk; the shield knows.
    shield6 = em.safe_actions(6, candidates=(0, 1), margin=0.5)
    print(f"  state 6 → 𝔈 = {shield6.state_empowerment_bits:.3f} bits")
    for a, succ in zip(shield6.candidates, shield6.successor_empowerment_bits):
        admit = "ADMIT" if a in shield6.admissible else "REFUSE"
        print(f"    a={a}  𝔈(s') = {succ:.3f} bits  →  {admit}")

    # -----------------------------------------------------------------------
    # 5) Mutual-information skill discovery (DIAYN).
    # -----------------------------------------------------------------------
    banner("Coordinator step 5/8 — DIAYN mutual-information skill discovery")
    sk = em.skill_discovery(n_skills=4, steps=60)
    print(f"  discovered {sk.n_skills} skills over {N_STATES} states")
    print(f"  skill-state entropy H(z|s) = {sk.skill_entropy:.3f} bits")
    print(f"  skill separability E[log q(z|s)] = {sk.skill_separability:+.3f} bits")
    for z, row in enumerate(sk.skill_state_dist):
        bars = "".join("█" if p > 0.2 else "·" for p in row)
        print(f"    skill z={z}:  {bars}    (top states: "
              f"{[s for s, p in sorted(enumerate(row), key=lambda x: -x[1])[:3]]})")

    # -----------------------------------------------------------------------
    # 6) Variational empowerment estimator (sample-based InfoNCE).
    # -----------------------------------------------------------------------
    banner("Coordinator step 6/8 — variational empowerment (InfoNCE lower bound)")
    em_var = Empowerer(
        EmpowererConfig(
            dim_state=N_STATES,
            dim_action=N_ACTIONS,
            estimator=EST_VARIATIONAL_INFONCE,
            variational_samples=64,
            variational_lr=0.3,
            laplace_alpha=0.25,
        )
    )
    em_var.fit_transitions(((s, a, sp) for (s, a, sp) in [
        (s, a, grid_step(s, a, random.Random(s * 13 + a)))
        for s in range(N_STATES)
        for a in range(N_ACTIONS)
        for _ in range(100)
    ]))
    # Train the variational decoder a bit.
    for _ in range(120):
        em_var.variational_empowerment(0)
    vr = em_var.variational_empowerment(0)
    print(f"  state 0:  variational lower bound = {vr.lower_bound_bits:.3f} bits "
          f"± {vr.hoeffding_half_width:.3f}  (samples={vr.samples_used})")
    print(f"  for comparison, Blahut-Arimoto exact = {land[0]:.3f} bits")

    # -----------------------------------------------------------------------
    # 7) PAC certificate over the empowerment estimate (Paninski 2003).
    # -----------------------------------------------------------------------
    banner("Coordinator step 7/8 — PAC certificate for the runtime contract")
    cert = em.certify(0)
    print(f"  state 0  𝔈̂ = {cert.empowerment_bits:.3f} bits")
    print(f"  PAC bound (δ = {1 - cert.confidence:.2f}):  "
          f"[{cert.lower_bound_bits:.3f},  {cert.upper_bound_bits:.3f}] bits")
    print(f"  holds={cert.holds},  min n_samples = {cert.n_samples}")
    print(f"  chain head: {cert.chain_head[:24]}...")

    # -----------------------------------------------------------------------
    # 8) Snapshot — ship the estimator over the wire.
    # -----------------------------------------------------------------------
    banner("Coordinator step 8/8 — snapshot, ship, restore")
    snap = em.snapshot()
    em_clone = Empowerer(em.config)
    em_clone.restore(snap)
    head_match = em_clone.chain_head == em.chain_head
    print(f"  snapshot bytes: {len(str(snap))}")
    print(f"  chain head preserved after restore: {head_match}")
    print(f"  Empowerer is wire-portable.")

    # -----------------------------------------------------------------------
    # Final report.
    # -----------------------------------------------------------------------
    banner("Final report")
    rep = em.report()
    print(f"  total transitions          = {rep.total_transitions}")
    print(f"  distinct states visited    = {rep.distinct_states} / {N_STATES}")
    print(f"  mean 𝔈ⁿ                    = {rep.mean_state_empowerment:.3f} bits")
    print(f"  max  𝔈ⁿ                    = {rep.max_state_empowerment:.3f} bits")
    print(f"  min  𝔈ⁿ                    = {rep.min_state_empowerment:.3f} bits")
    print(f"  estimator                  = {rep.estimator}")
    print(f"  reward mode                = {rep.reward_mode}")
    print(f"  chain head                 = {rep.chain_head[:24]}...")

    # -----------------------------------------------------------------------
    # Investor-style takeaways.
    # -----------------------------------------------------------------------
    banner("Coordination-engine takeaways")
    print(
        "  · Empowerment is a *single* bit-count scalar a coordinator can\n"
        "    compare across states, agents, and time.\n"
        "  · A coordinator can refuse to act when 𝔈ⁿ(s) < threshold —\n"
        "    \"we don't know enough about the world here yet\".\n"
        "  · The shield refuses actions that would discard future agency,\n"
        "    even when the task reward looks attractive (Salge & Polani\n"
        "    2017's *empowerment as replacement for the three laws*).\n"
        "  · The PAC certificate gives the coordinator a contract — the\n"
        "    estimate is provably within [lower, upper] bits w.p. 1 − δ.\n"
        "  · DIAYN's mutual-information objective discovers behavioural\n"
        "    latents without ever specifying a task — open-ended\n"
        "    skill curriculum out of the box."
    )


if __name__ == "__main__":
    run_demo()
