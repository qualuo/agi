"""Flower demo — GFlowNets as the runtime's diversification kernel.

Drug-discovery framing in a few hundred lines (no API key needed).
A tiny binary "molecule" lattice with three high-reward modes and many
filler modes; the Flower learns to sample candidates *proportional to
reward*, gives the coordination engine a diverse Pareto-ranked panel,
and emits a tamper-evident receipt the compliance officer signs.

Run:  python examples/flower_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.flower import (
    Flower,
    FlowerConfig,
    LOSS_TRAJECTORY_BALANCE,
)


def banner(title: str) -> None:
    bar = "─" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Tiny "molecule" DAG: 4-bit strings.  Reward boosts on three target modes,
# baseline 1.0 on the rest.
# ---------------------------------------------------------------------------

REWARDS: dict[str, float] = {
    "0000": 6.0,  # mode A
    "1111": 4.0,  # mode B
    "1010": 3.0,  # mode C
}


def succ(s: str):
    if len(s) >= 4:
        return []
    return [("0", s + "0"), ("1", s + "1")]


def terminal(s: str) -> bool:
    return len(s) == 4


def reward(s: str) -> float:
    return REWARDS.get(s, 1.0)


def main() -> None:
    banner("1. Register the candidate-generation environment")
    events: list[tuple[str, dict]] = []

    def publisher(kind: str, payload: dict) -> None:
        events.append((kind, payload))

    flow = Flower(
        FlowerConfig(
            loss=LOSS_TRAJECTORY_BALANCE,
            learning_rate=0.05,
            epsilon_exploration=0.1,
            rng_seed=12345,
        ),
        publisher=publisher,
    )
    flow.register_env(
        "molecules",
        initial="",
        successors=succ,
        terminal=terminal,
        reward=reward,
    )
    all_terminals = [format(i, "04b") for i in range(16)]
    Z = sum(reward(t) for t in all_terminals)
    print(f"  Registered 'molecules' env  (true Z = Σ R(x) = {Z}; log Z = {__import__('math').log(Z):.3f})")
    print(f"  3 high-reward modes: {REWARDS}    13 baseline modes: R=1.0")

    banner("2. Train the GFlowNet — reward-proportional sampling")
    for step in range(400):
        rep = flow.train_step("molecules", n_trajectories=16)
        if step % 80 == 0:
            print(
                f"  step {step:>4d}: loss={rep.loss_value:.3f}  "
                f"logZ={rep.logZ_estimate:.3f}  "
                f"mean_R={rep.weighted_reward:.2f}"
            )
    print(
        f"  final: logZ={rep.logZ_estimate:.3f}  (target log Z ≈ {__import__('math').log(Z):.3f})"
    )

    banner("3. Sample a diverse panel of 200 candidates")
    batch = flow.sample("molecules", n=200, temperature=1.0, epsilon=0.0)
    counts: dict[str, int] = {}
    for t in batch.terminals:
        counts[t] = counts.get(t, 0) + 1
    print(
        f"  unique terminals: {batch.unique_terminals}/16"
        f"   forward entropy: {batch.forward_entropy:.3f}"
    )
    print(f"  mean reward: {batch.mean_reward:.3f}"
          f"   MP-LCB (95%): {batch.mean_reward_lcb:.3f}"
          f"   HRMS-LCB: {batch.mean_reward_hrms_lcb:.3f}")
    print("  top-5 most-sampled terminals:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
        print(
            f"    {k}  count={v:3d}  R={reward(k):.2f}  "
            f"R/Z={reward(k) / Z:.3f}  (empirical={v / 200:.3f})"
        )

    banner("4. Mode-coverage certificate")
    cov = flow.mode_coverage("molecules", n_samples=500, top_k=3)
    print(f"  TV(empirical, target)        = {cov.tv_to_target:.3f}")
    print(f"  Hoeffding 95% UCB on TV       = {cov.tv_hoeffding_ucb:.3f}")
    print(f"  top-{cov.top_k_recovered[0]} modes recovered     = {cov.top_k_recovered[1]}/{cov.top_k_recovered[0]}")
    print(f"  HRMS 95% LCB on coverage prob = {cov.mode_coverage_lcb:.3f}")
    print(f"  log-Z bracket                 = "
          f"[{cov.log_partition_lcb:.3f}, {cov.log_partition_ucb:.3f}]")

    banner("5. Top-K Pareto ranking — the coordinator-facing deliverable")
    topk = flow.top_k("molecules", k=5)
    for state, r, c in topk:
        print(f"    {state}  R={r:.2f}  observed_count={c}")

    banner("6. Identifiability — Curator hand-off")
    ident = flow.identifiability("molecules", top_k=3)
    print("  3 under-sampled edges:")
    for s, a, c in ident.under_sampled_edges:
        print(f"    ({s!r}, action={a!r})  visits={c}")
    if ident.unreachable_modes:
        print(f"  unreachable modes: {ident.unreachable_modes}")
    else:
        print("  all rewarded modes reached at least once")

    banner("7. PIT calibration — DriftSentinel-ready statistic")
    pit = flow.pit_calibration("molecules")
    print(f"  n={pit.n}   KS D={pit.ks_statistic:.3f}   p={pit.p_value:.3f}")
    print("  (small p is expected on reward-proportional draws against a")
    print("   uniform null; this becomes the DriftSentinel baseline that")
    print("   CUSUM trips when live samples drift away from the trained")
    print("   reward distribution.)")

    banner("8. Attestation chain head — coordination-engine receipt")
    print(f"  events emitted     = {len(events)}")
    print(f"  chain head (sha256) = {flow.chain_head}")
    print(f"  fingerprint on batch = {batch.fingerprint}")
    print(f"  fingerprint on cov   = {cov.fingerprint}")
    print(f"  All payloads are JSON-serialisable; coordinator can replay.")
    sample_payload = next((p for k, p in events if k == 'flower.sampled'), None)
    if sample_payload:
        print(f"  example sample event: {json.dumps(sample_payload, sort_keys=True)[:120]}…")


if __name__ == "__main__":
    main()
