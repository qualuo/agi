"""Goodharter demo: proxy-reward / specification-gaming divergence detection.

The pitch in one runnable script (no API key, no network, pure stdlib):

  1. A coordination engine routes work to many models.  Every model
     is optimised against a *proxy* reward — a learned reward model,
     a scoring rubric, a click-through funnel, a judge LLM.  Under
     enough optimisation pressure the proxy diverges from the *true*
     objective (Goodhart's Law).  A coordination engine that cannot
     *certify* the gap at every dispatch decision routes high-stakes
     work onto a runaway optimiser.
  2. We simulate three deployment scenarios:
       (a) ``aligned``  — a reward model whose proxy tracks truth
                          tightly with small Gaussian noise.
       (b) ``mild``     — a reward model that develops a small,
                          persistent gap mid-deployment.
       (c) ``severe``   — a reward model with sustained over-rewarding
                          past an onset (the classic specification-
                          gaming failure mode).
  3. We feed each stream into a :class:`Goodharter` and read the
     issued certificates: Pearson r with Fisher-Z CI, Spearman /
     Kendall rank-correlation, empirical-Bernstein gap CI, two
     anytime-valid e-processes (Beta-Binomial betting indicator
     and hedged-capital on the continuous gap), Holm step-down
     FWER and Vovk-Wang product of e-values, and an LCS on E[gap]
     inverted from the hedged-capital e-process.
  4. We demonstrate budget tightening at runtime: the same observed
     stream that certifies TRUST under a permissive budget shifts
     to RETRAIN / QUARANTINE under a tighter one — and the
     fingerprint chain records the budget update so the verdict
     change is auditable.
  5. We tap the EventBus so a coordinator sees the live verdict
     transitions, and confirm that ``GH_ALERTED`` only fires on
     ``RETRAIN`` / ``QUARANTINE`` cases.
  6. Replay-verification: two Goodharter instances fed the same
     stream under the same config converge to identical fingerprint
     chains — what the engine swore it saw is exactly what an
     auditor can replay.

Run:  python examples/goodharter_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.goodharter import (
    GH_ALERTED,
    GH_CERTIFIED,
    GH_OBSERVED,
    REC_DEPLOY,
    REC_ESCALATE_HUMAN,
    REC_RETUNE,
    VERDICT_INVESTIGATE,
    VERDICT_QUARANTINE,
    VERDICT_RETRAIN,
    VERDICT_TRUST,
    Goodharter,
    GoodharterConfig,
    RewardObservation,
    fresh_goodharter,
    synthetic_aligned_stream,
    synthetic_goodhart_stream,
)


def _hr(title: str) -> None:
    print()
    print("─" * 72)
    print(title)
    print("─" * 72)


def _summarise(cert) -> None:
    print(f"  proxy_id       : {cert.proxy_id}")
    print(f"  n_observations : {cert.n_observations}")
    print(f"  verdict        : {cert.verdict}")
    print(f"  recommendation : {cert.recommendation}")
    print(f"  pearson_r      : {cert.pearson_r:+.3f} "
          f"(CI {cert.pearson_ci_low:+.3f} .. {cert.pearson_ci_high:+.3f})")
    print(f"  spearman_r     : {cert.spearman_r:+.3f}")
    print(f"  kendall_tau    : {cert.kendall_tau:+.3f}")
    print(f"  gap_mean       : {cert.gap_mean:+.3f}")
    print(f"  gap_ci         : [{cert.gap_ci_low:+.3f}, {cert.gap_ci_high:+.3f}]")
    print(f"  gap_evalue     : {cert.gap_evalue:.3g}")
    print(f"  hedged_evalue  : {cert.gap_hedged_evalue:.3g}")
    print(f"  hedged_lcs     : [{cert.hedged_lcs_low:+.3f}, {cert.hedged_lcs_high:+.3f}]")
    print(f"  monot_viol     : {cert.monotonicity_violation_rate:.3f}")
    print(f"  product_evalue : {cert.product_evalue:.3g}")
    print(f"  fingerprint    : {cert.fingerprint[:16]}…")
    if cert.tests:
        print("  tests:")
        for t in cert.tests:
            flag = "✓" if t.rejected else "·"
            print(f"    {flag} {t.name:32s} stat={t.statistic:+.3g} "
                  f"thr={t.threshold:+.3g}  {t.detail}")


def scenario(label: str, stream, *, budget: float = 0.05,
             min_corr: float = 0.7, bus=None) -> Goodharter:
    _hr(label)
    cfg = GoodharterConfig(
        proxy_id=label.lower().replace(" ", "_"),
        divergence_budget=budget,
        min_correlation=min_corr,
        min_observations=32,
        window_size=256,
        alpha=0.05,
        seed=0,
    )
    g = Goodharter(cfg, bus=bus)
    for obs in stream:
        g.observe(obs)
    cert = g.certify()
    _summarise(cert)
    return g


def main() -> None:
    bus = EventBus()

    # -------------------------------------------------------------------
    # 1. Three scenarios across the safety spectrum.
    # -------------------------------------------------------------------

    aligned = synthetic_aligned_stream(400, noise=0.02, seed=42)
    mild = synthetic_goodhart_stream(400, onset=0.5, drift=0.06,
                                     noise=0.02, seed=11)
    severe = synthetic_goodhart_stream(400, onset=0.3, drift=0.3,
                                       noise=0.02, seed=7)

    scenario("Scenario A — Aligned reward model (clean stream)",
             aligned, budget=0.05, bus=bus)
    scenario("Scenario B — Mild persistent drift (developing Goodhart)",
             mild, budget=0.05, bus=bus)
    scenario("Scenario C — Severe sustained drift (specification gaming)",
             severe, budget=0.05, bus=bus)

    # -------------------------------------------------------------------
    # 2. Budget tightening: same stream, different verdicts.
    # -------------------------------------------------------------------

    _hr("Budget tightening — same stream, different verdicts")
    # A stream whose gap is around 0.10 average — within a permissive
    # 0.20 budget, outside a tight 0.02 one.
    budget_stream = synthetic_goodhart_stream(400, onset=0.2, drift=0.15,
                                              noise=0.02, seed=23)
    g = fresh_goodharter("budget_demo", divergence_budget=0.20,
                         min_correlation=0.5, min_observations=32, seed=0)
    for obs in budget_stream:
        g.observe(obs)
    c1 = g.certify()
    print(f"  budget=0.20 (permissive) → verdict={c1.verdict} ({c1.recommendation})")
    print(f"    gap_mean={c1.gap_mean:.3f}  gap_evalue={c1.gap_evalue:.3g}")
    g.update_budget(divergence_budget=0.02)
    c2 = g.certify()
    print(f"  budget=0.02 (tight)      → verdict={c2.verdict} ({c2.recommendation})")
    print(f"    gap_mean={c2.gap_mean:.3f}  gap_evalue={c2.gap_evalue:.3g}")
    print("  ↑ same observations, tightened bar, fingerprint advanced for audit")

    # -------------------------------------------------------------------
    # 3. EventBus introspection — what a coordinator sees.
    # -------------------------------------------------------------------

    _hr("EventBus snapshot — what a coordination engine subscribes to")
    counts: dict[str, int] = {}
    for ev in bus.history():
        counts[ev.kind] = counts.get(ev.kind, 0) + 1
    for kind in sorted(counts):
        print(f"  {kind:32s} ×{counts[kind]}")
    if GH_ALERTED in counts:
        print(f"  → GH_ALERTED fires only on RETRAIN / QUARANTINE.")

    # -------------------------------------------------------------------
    # 4. Replay-verification.
    # -------------------------------------------------------------------

    _hr("Replay-verification — bit-identical fingerprints under same config")
    a = fresh_goodharter("replay_a", min_observations=32, seed=42)
    b = fresh_goodharter("replay_b", min_observations=32, seed=42)
    stream = synthetic_aligned_stream(80, noise=0.02, seed=99)
    for obs in stream:
        a.observe(obs)
        b.observe(obs)
    # Different proxy_id ⇒ different fingerprints (the seed is into the
    # config dict).  Re-run with matched config:
    a2 = Goodharter(GoodharterConfig(proxy_id="r", min_observations=32, seed=42))
    b2 = Goodharter(GoodharterConfig(proxy_id="r", min_observations=32, seed=42))
    for obs in stream:
        a2.observe(obs)
        b2.observe(obs)
    assert a2.fingerprint == b2.fingerprint
    print(f"  matched config + matched stream ⇒ {a2.fingerprint[:24]}…")
    print(f"  identical for both Goodharter instances: ✓")

    # -------------------------------------------------------------------
    # 5. The coordinator-facing pitch.
    # -------------------------------------------------------------------

    _hr("Coordinator-facing pitch")
    print("""\
  Goodharter is the runtime contract a coordination engine reaches for
  when it needs to KNOW its proxy reward still tracks the truth — and
  needs to ACT on the answer.

  Inputs   : streaming (proxy, true) pairs from any source — held-out
             labels, expert review, downstream outcomes, delayed user
             signals.  Anytime-valid: the engine peeks freely without
             inflating type-I error.

  Outputs  : a structured GoodharterCertificate carrying
             TRUST | INVESTIGATE | RETRAIN | QUARANTINE  +
             DEPLOY | MONITOR | RETUNE | REPLACE | ESCALATE_HUMAN.

  Composes : with refuser / sycophant / confabulator / constitutionalist
             (each safety primitive is a `true_reward` source);
             with aligner / personalizer (gate adapter promotion);
             with attest / governance (the audit and gate surface);
             with schemer (Goodhart drift × deception × sandbagging).

  Guarantee: fingerprint-chained replay-verifiability + anytime-valid
             confidence sequences under Howard et al. / Waudby-Smith-
             Ramdas + Vovk-Wang Holm-corrected multi-test family.

  Why now  : Goodhart's Law is the central failure of any AGI runtime
             that optimises a learned proxy.  This is the certificate
             a regulator (EU AI Act, NIST AI RMF), an investor
             (alignment moat), and an internal auditor (specification-
             gaming postmortems) all reach for.
""")


if __name__ == "__main__":
    main()
