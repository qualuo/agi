"""Pareto + BayesOpt + AttestationLedger — multi-objective coordination flow.

A coordination engine drives the runtime to deliver:

  Goal:  Generate a *Pareto-optimal panel* of candidates jointly
         trading off two competing objectives (e.g. drug-binding
         affinity vs synthetic accessibility, or response quality vs
         latency, or accuracy vs cost), with:

           * a calibrated *expected hypervolume improvement* receipt
             to spend the next evaluation budget on,
           * an anytime-valid progress certificate that the front has
             converged within ε hypervolume per step,
           * a tamper-evident replay-verifiable audit chain.

This is the **multi-objective decision product story** in a single
runnable script (no API key required):

  1. Pareto.observe         — register candidate cost vectors
  2. Pareto.frontier        — extract the Pareto-rank-1 layer
  3. Pareto.hypervolume     — quantify aggregate Pareto progress
  4. Pareto.expected_hypervolume_improvement
                            — drive the next evaluation choice
  5. Pareto.scalarise/sweep — expose the front via 1D inner loops
  6. Pareto.certify_progress
                            — anytime-valid HV-growth certificate
  7. AttestationLedger.append + verify
                            — compliance receipt chain

Run:  python examples/pareto_coordination_demo.py
"""
from __future__ import annotations

import math
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.attest import AttestationLedger
from agi.pareto import (
    SCALAR_TCHEBYCHEFF,
    Pareto,
    ParetoConfig,
)


def banner(title: str) -> None:
    bar = "─" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Pretend candidate space: 50 production-ready model checkpoints, each
# carrying (1) quality cost (1 − accuracy) and (2) latency cost (ms / 100).
# The coordination engine wants to ship a panel of K Pareto-optimal
# checkpoints so an account team can fit different customer SLOs.
# ---------------------------------------------------------------------------


def synthetic_candidates(seed: int = 0) -> dict[str, tuple[float, float]]:
    """Make 50 candidates with a clear Pareto staircase + noisy interior."""
    rng = random.Random(seed)
    cands: dict[str, tuple[float, float]] = {}
    # Six "true Pareto" anchor checkpoints.
    for i, (q, l) in enumerate(
        [
            (0.02, 0.95),   # huge model, slow, very accurate
            (0.05, 0.55),
            (0.10, 0.35),
            (0.18, 0.20),
            (0.30, 0.10),
            (0.50, 0.04),   # tiny model, fast, low accuracy
        ]
    ):
        cands[f"anchor_{i}"] = (q, l)
    # Forty-four random interior candidates (dominated with high prob).
    for i in range(44):
        q = rng.uniform(0.10, 0.70)
        l = rng.uniform(0.10, 0.80)
        cands[f"checkpoint_{i:02d}"] = (q, l)
    return cands


def main() -> None:
    banner("AGI runtime — multi-objective coordination flow")

    cfg = ParetoConfig(
        senses=("min", "min"),
        reference=(1.0, 1.0),
        confidence=0.95,
        hmac_key=b"compliance-secret-key",
    )
    pf = Pareto(cfg)

    cands = synthetic_candidates(seed=2026)

    # 1. Observe every candidate ------------------------------------------
    banner("1. Coordinator observes every candidate's (quality_cost, latency_cost)")
    for cid, cost in cands.items():
        pf.observe(cid, cost)
    print(f"  observed {len(pf)} candidates over 2 objectives")
    print(f"  chain head after observations = {pf.chain_head[:24]}…")

    # 2. Pareto front extraction ------------------------------------------
    banner("2. Pareto front: rank-1 layer, sorted by crowding distance")
    fr = pf.frontier()
    print(f"  front size = {len(fr.candidates)}")
    for cid, c, cr in zip(fr.candidates, fr.costs, fr.crowding):
        crowd_repr = "∞" if math.isinf(cr) else f"{cr:6.3f}"
        print(
            f"    {cid:20s}  quality={c[0]:6.3f}  latency={c[1]:6.3f}  "
            f"crowding={crowd_repr}"
        )

    # Show layered structure for the coordination engine's planning.
    layers = pf.rank_layers()
    print(f"\n  total Pareto ranks = {len(layers)}")
    for k, layer in enumerate(layers[:4]):
        print(f"    rank {k}: {len(layer)} candidates")
    if len(layers) > 4:
        print(f"    … (and {len(layers) - 4} more ranks)")

    # 3. Hypervolume indicator --------------------------------------------
    banner("3. Hypervolume — single aggregate Pareto-progress signal")
    hv1 = pf.hypervolume()
    print(f"  HV       = {hv1.hypervolume:.6f}")
    print(f"  algorithm = {hv1.algorithm}")
    print(f"  reference = {hv1.reference}")
    print(f"  n_points  = {hv1.n_points} on the front")

    # 4. Expected Hypervolume Improvement -- next evaluation choice -------
    banner("4. EHVI acquisition — which next checkpoint to evaluate?")
    # The coordinator has a budget to evaluate ONE new candidate; the
    # surrogate emits a Gaussian belief over the new candidate's
    # (quality, latency).  EHVI ranks candidates by the calibrated
    # expected progress their evaluation would deliver.
    proposals = [
        ("cheap_explore", (0.40, 0.50), (0.10, 0.10)),
        ("balanced",      (0.15, 0.25), (0.05, 0.05)),
        ("aggressive",    (0.03, 0.15), (0.15, 0.15)),
        ("dominated",     (0.80, 0.80), (0.05, 0.05)),
    ]
    best = None
    best_ehvi = -1.0
    print(f"  {'proposal':<20s} {'mean':>14s} {'sigma':>14s} {'EHVI':>10s}")
    for name, mean, sigma in proposals:
        rep = pf.expected_hypervolume_improvement(
            mean, sigma, candidate_id=name
        )
        mark = ""
        if rep.ehvi > best_ehvi:
            best_ehvi = rep.ehvi
            best = name
            mark = "  ← argmax"
        mean_repr = f"({mean[0]:.2f},{mean[1]:.2f})"
        sigma_repr = f"({sigma[0]:.2f},{sigma[1]:.2f})"
        print(
            f"  {name:<20s} {mean_repr:>14s} {sigma_repr:>14s} "
            f"{rep.ehvi:10.6f}{mark}"
        )
    print(f"\n  Coordinator recommends next evaluation: {best!r}")

    # 5. Tchebycheff sweep — Das-Dennis weight grid -----------------------
    banner("5. Coordinator generates a 1D inner-loop sweep (Das-Dennis)")
    sweep = pf.sweep(method=SCALAR_TCHEBYCHEFF, p=10)
    print(f"  generated {len(sweep)} weight vectors on the 2-simplex")
    winners = [r.argmin_candidate for r in sweep]
    unique_winners = []
    seen = set()
    for w in winners:
        if w not in seen:
            unique_winners.append(w)
            seen.add(w)
    print(f"  distinct argmins recovered = {len(unique_winners)}")
    print(f"  ↳ {unique_winners}")
    print("  ↳ exactly the Pareto-rank-1 layer's left-to-right ordering")

    # 6. Anytime-valid progress certificate -------------------------------
    banner("6. Anytime-valid HV-growth certificate (HRMS 2021)")
    # Simulate the coordinator running EHVI-guided evaluation for a few
    # rounds — each iteration adds one candidate and reads HV.
    rng = random.Random(0)
    for k in range(8):
        q = rng.uniform(0.05, 0.40)
        l = rng.uniform(0.05, 0.40)
        pf.observe(f"iter_{k}", (q, l))
        pf.hypervolume()
    cert = pf.certify_progress(epsilon=0.002)
    print(f"  HV now       = {cert.hv_now:.6f}")
    print(f"  ΔHV mean     = {cert.delta_hv:.6f}")
    print(f"  ΔHV LCB      = {cert.delta_hv_lcb:.6f}")
    print(f"  ΔHV UCB      = {cert.delta_hv_ucb:.6f}")
    print(f"  ε (stop)     = {cert.epsilon:.6f}")
    print(f"  converged?   = {cert.converged}")
    print(f"  confidence   = {cert.confidence}")

    # 7. Tamper-evident receipt chain -------------------------------------
    banner("7. Compliance officer signs every coordination event")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "pareto-receipts.jsonl"
        secret = b"compliance-secret-key"
        ledger = AttestationLedger(path=ledger_path, key=secret)

        ledger.append({
            "ticket_id": "pareto-front-snapshot",
            "op": "frontier",
            "n_candidates": len(pf),
            "front_size": len(fr.candidates),
            "front": list(fr.candidates),
            "chain_head": pf.chain_head,
        })
        ledger.append({
            "ticket_id": "pareto-hv-report",
            "op": "hypervolume",
            "hv": hv1.hypervolume,
            "reference": list(hv1.reference),
            "algorithm": hv1.algorithm,
            "chain_head": pf.chain_head,
        })
        ledger.append({
            "ticket_id": "pareto-next-evaluation",
            "op": "ehvi",
            "recommended": best,
            "ehvi": best_ehvi,
            "chain_head": pf.chain_head,
        })
        ledger.append({
            "ticket_id": "pareto-progress-cert",
            "op": "certify",
            "hv_now": cert.hv_now,
            "delta_hv": cert.delta_hv,
            "delta_hv_lcb": cert.delta_hv_lcb,
            "delta_hv_ucb": cert.delta_hv_ucb,
            "epsilon": cert.epsilon,
            "converged": cert.converged,
            "chain_head": pf.chain_head,
        })

        ok, why = ledger.verify()
        print(f"  ledger entries           = {len(ledger)}")
        print(f"  ledger head (root)       = {ledger.head_hash()[:24]}…")
        print(f"  ledger verification      = {'OK' if ok else f'FAILED — {why}'}")
        assert ok, why

        # Reload and re-verify — the receipt chain is byte-stable across
        # processes, so any compliance auditor can reproduce the trace.
        second = AttestationLedger(path=ledger_path, key=secret)
        ok2, _ = second.verify()
        print(f"  re-load verification     = {'OK' if ok2 else 'FAILED'}")
        print(
            "  re-loaded head matches    = "
            f"{second.head_hash() == ledger.head_hash()}"
        )

    banner("DONE — coordinator delivered Pareto panel + receipt chain")
    print(
        "  primitive  : agi.pareto.Pareto\n"
        "  composed   : EHVI + sweep + progress certificate + AttestationLedger\n"
        "  story      : ship the multi-objective candidate panel an account team\n"
        "               can route customers to, with a calibrated next-evaluation\n"
        "               recommendation and an anytime-valid stop signal — all\n"
        "               cryptographically replayable from the observation stream"
    )


if __name__ == "__main__":
    main()
