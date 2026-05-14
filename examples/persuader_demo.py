"""Persuader demo — Bayesian persuasion end-to-end for a coordination engine.

Scenarios
---------

A coordination engine has three concrete persuasion problems in production:

  1. **Routing.** A coordinator drives a fleet of receivers (sub-agents,
     tenants, downstream services) that pick an action based on their
     Bayes posterior. The coordinator does *not* have payments available
     (transfer-free environment) but can choose what information to
     reveal. What signaling policy maximises the coordinator's payoff?

  2. **Online routing.** The receiver's utility drifts over time
     (load-adaptive priorities, A/B-test treatment changes). The
     coordinator must achieve sublinear regret against the best fixed
     scheme in retrospect.

  3. **Robust routing.** The coordinator's prior over the receiver
     side is uncertain — it knows the prior lies in a finite candidate
     set ``U``. The robust scheme maximises the worst-case sender
     payoff over ``U``.

We also demonstrate **multi-receiver private persuasion** — independent
optimal schemes for two independent downstream receivers — and the
**Hoeffding PAC certificate** on simulated sender payoff.

Run:  ``python examples/persuader_demo.py``
"""
from __future__ import annotations

import random

from agi.events import EventBus
from agi.persuader import (
    KIND_LP,
    Persuader,
    PersuasionGame,
    SignalingScheme,
    best_response,
    sender_value_under_full_information,
)


def hr(label: str) -> None:
    print()
    print("=" * 72)
    print(label)
    print("=" * 72)


def main() -> None:
    bus = EventBus()
    p = Persuader(bus=bus, rng=random.Random(0))

    # ------------------------------------------------------------------
    # 1. Canonical Kamenica-Gentzkow prosecutor/judge.
    # ------------------------------------------------------------------
    hr("1. Kamenica-Gentzkow 2011 — prosecutor / judge")
    kg = PersuasionGame(
        states=("guilty", "innocent"),
        actions=("convict", "acquit"),
        prior=(0.3, 0.7),
        # Prosecutor wants conviction (+1) regardless of state.
        sender_utility=((1.0, 1.0), (0.0, 0.0)),
        # Judge's BR: convict iff P(guilty) ≥ 1/2.
        receiver_utility=((1.0, -1.0), (0.0, 0.0)),
    )
    no_info = sender_value_under_full_information(kg, kg.prior)
    out = p.persuade(kg)
    print(f"  Prior P(guilty) = {kg.prior[0]:.2f}")
    print(f"  No-info sender payoff:           {no_info:.3f}")
    print(f"  Optimal sender payoff (KG 0.6):  {out.sender_value:.3f}")
    print(f"  Bayes-plausible: {out.bayes_plausible}   "
          f"Obedience: {out.obedience_ok}")
    print(f"  Signaling scheme π(signal | state):")
    for i, sig in enumerate(out.scheme.signals):
        col = ", ".join(f"{kg.states[j]}={out.scheme.pi[i][j]:.3f}"
                        for j in range(len(kg.states)))
        print(f"    π(s='{sig}' | ·) = ({col})")
    print(f"  Induced posteriors:")
    for i, sig in enumerate(out.scheme.signals):
        post = out.induced_posteriors[i]
        col = ", ".join(f"{kg.states[j]}={post[j]:.3f}"
                        for j in range(len(kg.states)))
        m = out.signal_marginals[i]
        print(f"    s='{sig}' (P={m:.3f}): μ = ({col})  → judge plays "
              f"'{out.recommended_actions[i]}'")

    # ------------------------------------------------------------------
    # 2. 3-state LP — umbrella seller / commuter.
    # ------------------------------------------------------------------
    hr("2. Multi-state LP — umbrella seller persuading a commuter")
    weather = PersuasionGame(
        states=("sun", "cloud", "rain"),
        actions=("jacket", "tshirt", "umbrella"),
        prior=(0.3, 0.4, 0.3),
        sender_utility=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),  # umbrella sale is the sender's only payoff
        ),
        receiver_utility=(
            (-1.0, 1.0, 0.0),
            (1.0, 0.0, -2.0),
            (-0.5, -0.5, 1.0),
        ),
    )
    no_info = sender_value_under_full_information(weather, weather.prior)
    full_info = sum(
        weather.prior[w] * sender_value_under_full_information(weather,
            tuple(1.0 if i == w else 0.0 for i in range(3)))
        for w in range(3)
    )
    out = p.persuade(weather, kind=KIND_LP)
    print(f"  Prior over (sun, cloud, rain): {weather.prior}")
    print(f"  No-info sender payoff:    {no_info:.3f}")
    print(f"  Full-info sender payoff:  {full_info:.3f}  (receiver picks BR")
    print(f"                                              under true state)")
    print(f"  Optimal persuasion value: {out.sender_value:.3f}  ← exceeds both")
    print(f"  Sender exploits commuter indifference: under signal 'umbrella'")
    print(f"  the posterior is (1/3, 1/3, 1/3) and the commuter's expected")
    print(f"  utility for *umbrella* and *jacket* are both 0, so a tiny lean")
    print(f"  toward umbrella sells the umbrella in 90% of futures.")

    # ------------------------------------------------------------------
    # 3. PAC certificate on simulated sender payoff.
    # ------------------------------------------------------------------
    hr("3. Hoeffding anytime PAC certificate on sender payoff")
    cert = p.simulate(kg, out.scheme if False else
                       p.persuade(kg).scheme,
                       T=5_000, delta=0.05, method="empirical_bernstein")
    print(f"  N samples = {cert.n}")
    print(f"  Empirical mean: {cert.empirical_mean:.4f}")
    print(f"  (1 − δ)={1 - cert.delta:.2f} half-width "
          f"({cert.method}): {cert.half_width:.4f}")
    print(f"  LCB ≤ E[u_S] ≤ UCB:  [{cert.lcb:.4f}, {cert.ucb:.4f}]")

    # ------------------------------------------------------------------
    # 4. Online persuasion against a changing receiver type.
    # ------------------------------------------------------------------
    hr("4. Online persuasion — Hedge over an ε-net of schemes")
    rng = random.Random(42)
    # Simulate a receiver whose threshold drifts between 0.4 and 0.6.
    state_seq = [rng.choices(("guilty", "innocent"),
                             weights=kg.prior, k=1)[0] for _ in range(50)]
    threshold_seq = [0.4 + 0.2 * rng.random() for _ in range(50)]

    def feedback(scheme: SignalingScheme) -> float:
        # Round t uses (state_seq[t], threshold_seq[t]); we approximate
        # the time-varying receiver as: judge convicts iff μ(guilty) ≥ τ_t.
        t = feedback.t  # type: ignore[attr-defined]
        st = state_seq[t]
        tau = threshold_seq[t]
        # Sample a signal under realised state.
        w = 0 if st == "guilty" else 1
        weights = [scheme.pi[i][w] for i in range(len(scheme.signals))]
        tot = sum(weights)
        if tot < 1e-12:
            return 0.0
        u = rng.random() * tot
        running = 0.0
        sig_idx = 0
        for i, wt in enumerate(weights):
            running += wt
            if u <= running:
                sig_idx = i
                break
        # Posterior under chosen signal.
        post = scheme.posterior(scheme.signals[sig_idx], kg.prior)
        action = "convict" if post["guilty"] >= tau else "acquit"
        return 1.0 if action == "convict" else 0.0

    feedback.t = 0  # type: ignore[attr-defined]

    def stepped() -> float:
        v = feedback(persuader_scheme)
        feedback.t = (feedback.t + 1) % len(state_seq)  # type: ignore
        return v

    # Note: we wrap feedback because Hedge calls feedback(scheme) K times
    # per round; round time advances once per *call* in this demo.
    # Better demo: explicit round counter.
    feedback.t = 0  # type: ignore[attr-defined]
    persuader_scheme = SignalingScheme(
        signals=("acquit", "convict"),
        states=kg.states,
        pi=((1.0, 4.0/7.0), (0.0, 3.0/7.0)),  # the KG optimal scheme
    )
    out_on = p.online_persuade(kg, feedback, T=30, grid=3,
                                rng=random.Random(0))
    print(f"  T = {out_on.T} rounds, K = exhaustive ε-net over a 3-grid")
    print(f"  Cumulative sender payoff (played):   {out_on.cumulative_sender:.3f}")
    print(f"  Cumulative best-fixed payoff:        {out_on.cumulative_best_fixed:.3f}")
    print(f"  Cumulative regret:                   {out_on.cumulative_regret:.3f}")
    print(f"  Hedge bound √(T ln K / 2)·rng + 1:    {out_on.regret_bound:.3f}")
    print(f"  ▶ regret ≤ bound: {out_on.cumulative_regret <= out_on.regret_bound}")

    # ------------------------------------------------------------------
    # 5. Robust persuasion under prior uncertainty.
    # ------------------------------------------------------------------
    hr("5. Robust persuasion under prior uncertainty (Dworczak-Pavan 2022)")
    prior_set = [(0.20, 0.80), (0.30, 0.70), (0.40, 0.60), (0.50, 0.50)]
    robust = p.robust_persuade(
        states=("guilty", "innocent"),
        actions=("convict", "acquit"),
        sender_utility=((1.0, 1.0), (0.0, 0.0)),
        receiver_utility=((1.0, -1.0), (0.0, 0.0)),
        prior_set=prior_set,
    )
    print(f"  Worst-case sender value over U:  {robust.worst_case_value:.3f}")
    print(f"  Worst-case prior:                {robust.worst_case_prior}")
    print(f"  Same scheme deployed for the entire candidate prior set.")

    # ------------------------------------------------------------------
    # 6. Multi-receiver private persuasion.
    # ------------------------------------------------------------------
    hr("6. Multi-receiver private persuasion (Babichenko-Barman 2017)")
    # Two independent receivers, each with their own beliefs / utilities.
    r1 = PersuasionGame(
        states=("good", "bad"),
        actions=("approve", "deny"),
        prior=(0.25, 0.75),
        sender_utility=((1.0, 1.0), (0.0, 0.0)),
        receiver_utility=((1.0, -1.0), (0.0, 0.0)),
    )
    r2 = PersuasionGame(
        states=("good", "bad"),
        actions=("approve", "deny"),
        prior=(0.45, 0.55),
        sender_utility=((1.0, 1.0), (0.0, 0.0)),
        receiver_utility=((1.0, -1.0), (0.0, 0.0)),
    )
    multi = p.multi_receiver_private([("reviewer_A", r1), ("reviewer_B", r2)])
    print(f"  Joint sender value (additive): {multi.joint_sender_value:.3f}")
    for rid, scheme in multi.per_receiver_schemes:
        print(f"  {rid}: signals={scheme.signals}")

    # ------------------------------------------------------------------
    # 7. Composition with the rest of the runtime.
    # ------------------------------------------------------------------
    hr("7. Composition surface")
    print("  • MechanismDesigner: when payments ARE available, persuasion")
    print("    composes — recommend an action via Persuader, attach a")
    print("    Vickrey payment via MechanismDesigner.")
    print("  • TruthSerum: when receiver utility is unknown, elicit it")
    print("    via incentive-compatible peer prediction first.")
    print("  • Equilibrator: the LP's obedience constraints ARE the BCE")
    print("    constraints; Equilibrator verifies exploitability = 0.")
    print("  • Negotiator: induced recommendation profile feeds leximin")
    print("    / Nash-bargaining for fair-and-truthful refinement.")
    print("  • ActiveInferencer: drop-in solver for u_R when receiver")
    print("    is itself a generative-model belief.")
    print("  • AttestationLedger: every solve mints a tamper-evident")
    print("    receipt the coordinator publishes BEFORE the world realises.")
    print()
    print(f"  Persuader stats: {p.stats()}")

    # Event count snapshot.
    history = bus.history(limit=200)
    kinds: dict[str, int] = {}
    for e in history:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print(f"  Event kinds seen: {dict(sorted(kinds.items()))}")


if __name__ == "__main__":
    main()
