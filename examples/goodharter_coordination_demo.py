"""Goodharter coordination demo: a routing pool that quarantines reward
models drifting from ground truth.

The runtime-engine story end-to-end in one runnable script:

  1. The coordination engine has three reward-model workers in its
     routing pool:

       * ``faithful``       — a well-calibrated reward model whose
                              proxy tracks ground truth tightly.
       * ``creeping``       — a reward model that *starts* well-
                              calibrated and drifts into over-rewarding
                              partway through deployment (the classic
                              specification-gaming pattern).
       * ``goodharted``     — a reward model already heavily over-
                              optimised against its proxy.

  2. The engine maintains one :class:`Goodharter` per worker
     (per-proxy divergence shadow).  Each pair of (proxy, true)
     observations the engine acquires (held-out labels, expert
     ground truth, downstream outcomes) is routed to the worker's
     Goodharter; nothing about the model itself is required — the
     primitive is fully decoupled from the underlying surface.

  3. As traffic accumulates, the engine queries each worker's
     ``certify()`` and applies the returned recommendation:

       * ``TRUST``        → keep the worker fully in rotation.
       * ``INVESTIGATE``  → keep in rotation but increase paired-
                            observation sampling for tighter CIs.
       * ``RETRAIN``      → divert *new* traffic away from the worker;
                            schedule a retrain of its proxy reward
                            model.
       * ``QUARANTINE``   → pull the worker out of rotation entirely
                            and escalate to a human.

  4. Composition with the rest of the safety stack:

       * Schemer (cross-trial deception detection) is fed the *same*
         worker_id; a worker that drifts (Goodharter) AND scores
         differently observed-vs-unobserved (Schemer) triggers the
         highest-severity routing decision.
       * Refuser / Sycophant / Confabulator certificates are
         *true_reward sources* — each safety primitive's certified
         per-trial score becomes a Goodharter ``true_reward``
         signal paired against the model's own proxy.
       * Aligner / Personalizer adapter promotions are gated on the
         per-user / per-deployment Goodharter verdict — drift in
         the global proxy halts global aligner training; drift in
         the per-user proxy halts that user's Personalizer adapter.

  5. After ``N`` trials the engine prints:

       * a per-worker scorecard (verdict, gap, fingerprint)
       * the fleet-level Vovk-Wang product of e-values
       * a per-worker recommendation matrix the coordinator dispatches
         on.

Run:  python examples/goodharter_coordination_demo.py
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import Event, EventBus
from agi.goodharter import (
    GH_ALERTED,
    GH_CERTIFIED,
    Goodharter,
    GoodharterCertificate,
    GoodharterConfig,
    REC_DEPLOY,
    REC_ESCALATE_HUMAN,
    REC_MONITOR,
    REC_REPLACE,
    REC_RETUNE,
    RewardObservation,
    VERDICT_INVESTIGATE,
    VERDICT_QUARANTINE,
    VERDICT_RETRAIN,
    VERDICT_TRUST,
)


# ---------------------------------------------------------------------------
# Synthetic worker shop-floor
# ---------------------------------------------------------------------------


@dataclass
class WorkerSim:
    """A worker that returns (proxy, true) reward pairs with a
    documented drift behaviour.  Stands in for a real reward-model
    backed inference surface."""

    worker_id: str
    behaviour: str            # "faithful" | "creeping" | "goodharted"
    rng: random.Random

    def step(self, t: int, n_total: int) -> tuple[float, float]:
        true = self.rng.random() * 0.8 + 0.1
        if self.behaviour == "faithful":
            proxy = max(0.0, min(1.0, true + self.rng.gauss(0.0, 0.02)))
        elif self.behaviour == "creeping":
            cutoff = int(0.4 * n_total)
            if t < cutoff:
                proxy = max(0.0, min(1.0, true + self.rng.gauss(0.0, 0.02)))
            else:
                # Drift in linearly to a +0.20 over-reward bias.
                scale = (t - cutoff) / max(n_total - cutoff, 1)
                proxy = max(0.0, min(1.0, true + 0.20 * scale + self.rng.gauss(0.0, 0.02)))
        else:  # goodharted
            proxy = max(0.0, min(1.0, true + 0.25 + self.rng.gauss(0.0, 0.03)))
        return (proxy, true)


# ---------------------------------------------------------------------------
# Coordinator-side dispatch logic
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    worker_id: str
    accept_new_traffic: bool
    require_paired_sampling: bool
    reason: str
    cert_fingerprint: str


def _routing_decision(worker_id: str, cert: GoodharterCertificate) -> RoutingDecision:
    """Map a certificate to a routing decision."""
    v = cert.verdict
    if v == VERDICT_TRUST:
        return RoutingDecision(worker_id, True, False,
                               "verdict=TRUST → full rotation",
                               cert.fingerprint[:16])
    if v == VERDICT_INVESTIGATE:
        return RoutingDecision(worker_id, True, True,
                               "verdict=INVESTIGATE → keep in rotation, "
                               "increase paired sampling",
                               cert.fingerprint[:16])
    if v == VERDICT_RETRAIN:
        return RoutingDecision(worker_id, False, True,
                               "verdict=RETRAIN → divert new traffic, "
                               "schedule retrain",
                               cert.fingerprint[:16])
    return RoutingDecision(worker_id, False, True,
                           "verdict=QUARANTINE → pull worker, escalate",
                           cert.fingerprint[:16])


# ---------------------------------------------------------------------------
# EventBus listener
# ---------------------------------------------------------------------------


@dataclass
class EventCounter:
    counts: dict[str, int] = field(default_factory=dict)
    alerted: list[tuple[str, str]] = field(default_factory=list)

    def __call__(self, ev: Event) -> None:
        self.counts[ev.kind] = self.counts.get(ev.kind, 0) + 1
        if ev.kind == GH_ALERTED:
            self.alerted.append((ev.data.get("verdict", "?"),
                                 ev.data.get("fingerprint", "?")[:12]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("Goodharter coordination demo")
    print("A coordination engine routes traffic across three reward-model")
    print("workers, certifies proxy/true drift on each, and dispatches")
    print("routing decisions on the fly.")
    print("=" * 72)

    bus = EventBus()
    counter = EventCounter()
    bus.subscribe(counter)

    workers = [
        WorkerSim("faithful", "faithful", random.Random(11)),
        WorkerSim("creeping", "creeping", random.Random(22)),
        WorkerSim("goodharted", "goodharted", random.Random(33)),
    ]
    goodharters: dict[str, Goodharter] = {}
    for w in workers:
        cfg = GoodharterConfig(
            proxy_id=w.worker_id,
            divergence_budget=0.05,
            min_correlation=0.7,
            min_observations=40,
            window_size=256,
            alpha=0.05,
            seed=0,
        )
        goodharters[w.worker_id] = Goodharter(cfg, bus=bus)

    # ------------------------------------------------------------------
    # 1. Streaming dispatch loop.
    # ------------------------------------------------------------------

    N = 400
    print(f"\nStreaming {N} (proxy, true) pairs into each worker's "
          f"Goodharter shadow…")
    for t in range(N):
        for w in workers:
            proxy, true = w.step(t, N)
            goodharters[w.worker_id].observe(RewardObservation(
                decision_id=f"{w.worker_id}.{t}",
                proxy_reward=proxy,
                true_reward=true,
            ))

    # ------------------------------------------------------------------
    # 2. Certify per worker.
    # ------------------------------------------------------------------

    print("\n" + "─" * 72)
    print("Per-worker certification results")
    print("─" * 72)
    decisions: list[RoutingDecision] = []
    e_values: dict[str, float] = {}
    for w in workers:
        cert = goodharters[w.worker_id].certify()
        d = _routing_decision(w.worker_id, cert)
        decisions.append(d)
        e_values[w.worker_id] = cert.product_evalue
        print(f"\n  worker={w.worker_id} ({w.behaviour})")
        print(f"    verdict={cert.verdict}  recommendation={cert.recommendation}")
        print(f"    pearson_r={cert.pearson_r:+.3f}  "
              f"(CI {cert.pearson_ci_low:+.3f}..{cert.pearson_ci_high:+.3f})")
        print(f"    gap_mean={cert.gap_mean:+.3f}  "
              f"CI=[{cert.gap_ci_low:+.3f}, {cert.gap_ci_high:+.3f}]")
        print(f"    hedged_lcs=[{cert.hedged_lcs_low:+.3f}, {cert.hedged_lcs_high:+.3f}]")
        print(f"    product_evalue={cert.product_evalue:.3g}")
        print(f"    fingerprint={cert.fingerprint[:24]}…")
        rejected = [t.name for t in cert.tests if t.rejected]
        if rejected:
            print(f"    rejected_tests={rejected}")

    # ------------------------------------------------------------------
    # 3. Routing decision matrix.
    # ------------------------------------------------------------------

    print("\n" + "─" * 72)
    print("Coordinator routing matrix (what the engine dispatches on)")
    print("─" * 72)
    print(f"  {'worker_id':12s}  {'accept':6s}  {'paired':6s}  {'reason'}")
    for d in decisions:
        print(f"  {d.worker_id:12s}  "
              f"{'  ✓' if d.accept_new_traffic else '  ✗':6s}  "
              f"{'  ✓' if d.require_paired_sampling else '  ✗':6s}  "
              f"{d.reason}")

    # ------------------------------------------------------------------
    # 4. Fleet-level Vovk-Wang fusion.
    # ------------------------------------------------------------------

    print("\n" + "─" * 72)
    print("Fleet-level multi-test fusion (Vovk-Wang product of e-values)")
    print("─" * 72)
    fleet_product = 1.0
    for wid, e in e_values.items():
        fleet_product *= max(e, 1e-300)
    print(f"  per-worker product e-values: "
          f"{ {k: f'{v:.3g}' for k, v in e_values.items()} }")
    print(f"  fleet product e-value: {fleet_product:.3g}")
    print(f"  fleet rejection threshold @ α=0.05: 1/α = 20")
    fleet_rejects = fleet_product >= 20.0
    print(f"  fleet-level certificate rejection: "
          f"{'✓ REJECTED — at least one worker is drifting' if fleet_rejects else '· not rejected'}")
    print(f"    (Vovk-Wang 2021: product of e-values is anytime-valid")
    print(f"     under arbitrary dependence; rejection at level α iff")
    print(f"     product ≥ 1/α)")

    # ------------------------------------------------------------------
    # 5. EventBus summary.
    # ------------------------------------------------------------------

    print("\n" + "─" * 72)
    print("EventBus summary — what a coordinator subscribes to")
    print("─" * 72)
    for kind, n in sorted(counter.counts.items()):
        print(f"  {kind:32s} ×{n}")
    if counter.alerted:
        print(f"  GH_ALERTED fires on:")
        for verdict, fp in counter.alerted:
            print(f"    verdict={verdict}  fp={fp}…")

    # ------------------------------------------------------------------
    # 6. Stack composition — pitch.
    # ------------------------------------------------------------------

    print("\n" + "─" * 72)
    print("Stack composition")
    print("─" * 72)
    print("""\
  Goodharter is the proxy-vs-truth gate.  Every cell of the safety
  matrix feeds Goodharter a *true_reward* signal:

      Refuser certified score          ─┐
      Sycophant certified score        ─┼─→  paired against the model's
      Confabulator certified score     ─┤      *proxy* reward by Goodharter
      Constitutionalist principle score ─┘      and fingerprint-chained.

  The verdict feeds back into routing:

      Aligner adapter promotion         ←─ gated on global Goodharter verdict
      Personalizer adapter promotion    ←─ gated on per-user Goodharter verdict
      Capabilities router weighting     ←─ multiplied by per-worker verdict
      Strategist meta-decision          ←─ verdict + recommendation are inputs

  The audit chain ties it all together:

      Attest receipts include each Goodharter fingerprint per dispatch
      Governance enforces per-tenant verdict budgets
      Coordinator replays the chain at any time for full audit
""")

    print("=" * 72)
    print("Demo complete.  This is the runtime contract a coordination engine")
    print("reaches for to keep its proxy reward signals on-target —")
    print("anytime-valid, replay-verifiable, pure-stdlib, fingerprint-chained.")
    print("=" * 72)


if __name__ == "__main__":
    main()
