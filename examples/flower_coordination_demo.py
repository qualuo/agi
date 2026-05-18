"""Flower + Quantilizer + AttestationLedger — end-to-end coordination flow.

A coordination engine drives the runtime to deliver:

  Goal:  Generate K diverse high-reward candidates from a combinatorial
         space, retain only those in the top-q quantile of reward, and
         hand back a tamper-evident receipt the compliance officer can
         sign before any candidate is acted on.

This is the **panel-generation product story** in a single runnable
script (no API key required):

  1. Flower.register_env       — declare the candidate space
  2. Flower.train_step × N     — learn reward-proportional sampling
  3. Flower.sample             — draw a diverse panel
  4. Flower.mode_coverage      — certify mode coverage
  5. Quantilizer               — threshold to the top-q quantile
  6. AttestationLedger.append  — tamper-evident receipt chain
  7. AttestationLedger.verify  — compliance-officer signature

Run:  python examples/flower_coordination_demo.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.flower import Flower, FlowerConfig, LOSS_TRAJECTORY_BALANCE
from agi.quantilizer import quantilize_samples
from agi.attest import AttestationLedger


def banner(title: str) -> None:
    bar = "─" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Pretend candidate space: 4-bit "lead molecules".
# Three high-affinity scaffolds + thirteen baseline candidates.
# ---------------------------------------------------------------------------

REWARDS: dict[str, float] = {
    "0000": 9.0,   # lead A
    "1111": 6.0,   # lead B
    "0110": 4.5,   # lead C
}


def succ(s):
    if len(s) >= 4:
        return []
    return [("0", s + "0"), ("1", s + "1")]


def terminal(s):
    return len(s) == 4


def reward(s):
    return REWARDS.get(s, 1.0)


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="flower_coord_"))
    ledger_path = tmpdir / "receipts.jsonl"
    secret = b"shared-compliance-key"

    # The coordination engine maintains a shared, signed receipt chain.
    ledger = AttestationLedger(path=ledger_path, key=secret, namespace="flower-demo")

    # 1. Registration ------------------------------------------------------
    banner("1. Coordinator registers candidate-generation env with Flower")
    flow = Flower(
        FlowerConfig(
            loss=LOSS_TRAJECTORY_BALANCE,
            learning_rate=0.05,
            epsilon_exploration=0.1,
            rng_seed=2026,
        )
    )
    flow.register_env(
        "drug-leads",
        initial="",
        successors=succ,
        terminal=terminal,
        reward=reward,
    )
    Z = sum(reward(format(i, "04b")) for i in range(16))
    print(f"  env: drug-leads   |X|=16   Z=ΣR={Z}   log Z = {math.log(Z):.3f}")

    ledger.append({
        "ticket_id": "register-drug-leads",
        "op": "register",
        "env": "drug-leads",
        "n_terminals": 16,
        "fingerprint": flow.chain_head,
    })

    # 2. Train -------------------------------------------------------------
    banner("2. Coordinator trains the GFlowNet (400 SGD steps)")
    for step in range(400):
        rep = flow.train_step("drug-leads", n_trajectories=16)
    print(
        f"  final loss = {rep.loss_value:.4f}   "
        f"logZ = {rep.logZ_estimate:.3f}   (target log Z = {math.log(Z):.3f})"
    )

    ledger.append({
        "ticket_id": "training-complete",
        "op": "train",
        "env": "drug-leads",
        "final_loss": rep.loss_value,
        "final_logZ": rep.logZ_estimate,
        "n_steps": 400,
        "fingerprint": rep.fingerprint,
    })

    # 3. Sample a diverse panel -------------------------------------------
    banner("3. Coordinator asks for a panel of 200 candidates")
    batch = flow.sample("drug-leads", n=200, temperature=1.0, epsilon=0.0)
    print(f"  unique terminals = {batch.unique_terminals}/16")
    print(f"  forward entropy  = {batch.forward_entropy:.3f}")
    print(f"  mean reward      = {batch.mean_reward:.3f}")
    print(f"  MP 95% LCB       = {batch.mean_reward_lcb:.3f}")
    print(f"  HRMS anytime LCB = {batch.mean_reward_hrms_lcb:.3f}")
    print(f"  fingerprint      = {batch.fingerprint}")

    ledger.append({
        "ticket_id": "sample-batch-200",
        "op": "sample",
        "env": "drug-leads",
        "n": 200,
        "mean_reward": batch.mean_reward,
        "mean_reward_lcb": batch.mean_reward_lcb,
        "unique_terminals": batch.unique_terminals,
        "fingerprint": batch.fingerprint,
    })

    # 4. Certify mode coverage --------------------------------------------
    banner("4. Coordinator certifies mode coverage")
    cov = flow.mode_coverage("drug-leads", n_samples=600, top_k=3)
    print(f"  TV(empirical, target)  = {cov.tv_to_target:.3f}")
    print(f"  Hoeffding 95% UCB      = {cov.tv_hoeffding_ucb:.3f}")
    print(f"  top-3 modes recovered  = {cov.top_k_recovered[1]}/3")
    print(f"  HRMS coverage LCB 95%  = {cov.mode_coverage_lcb:.3f}")
    print(f"  fingerprint            = {cov.fingerprint}")

    ledger.append({
        "ticket_id": "certify-mode-coverage",
        "op": "certify",
        "env": "drug-leads",
        "tv": cov.tv_to_target,
        "tv_ucb": cov.tv_hoeffding_ucb,
        "modes_found": cov.modes_found,
        "coverage_lcb": cov.mode_coverage_lcb,
        "fingerprint": cov.fingerprint,
    })

    # 5. Quantilize: top 10% by reward ------------------------------------
    banner("5. Coordinator thresholds to top-10% by reward (Quantilizer)")
    q_report = quantilize_samples(
        list(batch.terminals), utility=reward, q=0.10, delta=0.05
    )
    kept = list(q_report.kept)
    kept_rewards = list(q_report.kept_utilities)
    print(f"  q-quantile threshold = {q_report.threshold:.3f}")
    print(f"  realised q           = {q_report.realised_q:.3f}")
    print(f"  kept {q_report.n_kept} of {q_report.n_total} candidates")
    print(f"  kept rewards         = {sorted(set(kept_rewards), reverse=True)[:5]}")
    print(f"  DKW band ε (δ=0.05)  = {q_report.dkw_band:.3f}")
    print(f"  quantile LCB         = {q_report.quantile_lcb:.3f}")
    print(f"  quantile UCB         = {q_report.quantile_ucb:.3f}")
    print(f"  fingerprint          = {q_report.fingerprint[:24]}…")

    ledger.append({
        "ticket_id": "quantilize-top-10pct",
        "op": "quantilize",
        "env": "drug-leads",
        "q": 0.10,
        "threshold": q_report.threshold,
        "n_kept": q_report.n_kept,
        "kept_rewards_top5": sorted(set(kept_rewards), reverse=True)[:5],
        "fingerprint": q_report.fingerprint,
    })

    # 6. Top-K final deliverable ------------------------------------------
    banner("6. Coordinator extracts top-3 distinct candidates")
    top = flow.top_k("drug-leads", k=3)
    for state, r, c in top:
        print(f"    {state}   R={r:.2f}   observed_count={c}")

    ledger.append({
        "ticket_id": "deliverable-top-3",
        "op": "deliver",
        "env": "drug-leads",
        "candidates": [
            {"state": s, "reward": r, "count": c} for s, r, c in top
        ],
        "fingerprint": flow.chain_head,
    })

    # 7. Compliance verification ------------------------------------------
    banner("7. Compliance officer verifies the receipt chain")
    ok, why = ledger.verify()
    print(f"  ledger entries        = {len(ledger)}")
    print(f"  ledger head (root)    = {ledger.head_hash()}")
    print(f"  ledger verification   = {'OK' if ok else f'FAILED — {why}'}")
    assert ok, why

    # Tamper test: prove the chain detects modification.
    # We do NOT modify the actual entries (verify() uses in-memory data),
    # but we can show that a re-open of the file picks up corruption if
    # any byte was changed.  Here we just re-verify a freshly loaded
    # ledger and confirm the head is the same.
    second = AttestationLedger(path=ledger_path, key=secret)
    ok2, _ = second.verify()
    print(f"  re-load verification  = {'OK' if ok2 else 'FAILED'}")
    print(f"  re-loaded head matches = {second.head_hash() == ledger.head_hash()}")

    banner("DONE — coordinator delivered panel + receipt chain")
    print(f"  receipts at: {ledger_path}")


if __name__ == "__main__":
    main()
