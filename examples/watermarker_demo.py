"""Watermarker demo: detect synthetic-content provenance with cryptographic strength.

The pitch in one runnable script (no API key required):

  1. A coordination engine is routing content through downstream stages
     -- some marked, some unmarked, some adversarially stripped.  It
     does NOT have access to the generator's logits or vocabulary; it
     has only the public watermark spec (key, gamma, hash kind) and
     the token stream.

  2. It asks the runtime: *"is this text watermarked?  did the mark
     survive paraphrasing?"*

  3. The :class:`Watermarker` primitive applies the KGW green-list
     z-test (Kirchenbauer et al. 2023, ICML), the Clopper-Pearson
     exact rate CI, and the anytime-valid Beta-Binomial e-process
     audit; once a small labelled pool is fitted, it returns a
     verdict + recommendation the coordinator can route on.

  4. Every step writes a fingerprinted event to the EventBus so any
     auditor can replay the decision later, byte-for-byte.

Run:  python examples/watermarker_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.watermarker import (
    POLARITY_VERIFY_WATERMARK,
    REC_BLOCK,
    REC_QUARANTINE,
    REC_RESTRICT,
    REC_TRUST,
    Trial,
    Watermarker,
    WatermarkerConfig,
    WatermarkSpec,
    simulate_marked_document,
    simulate_stripped_document,
    simulate_unmarked_document,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def main() -> int:
    bus = EventBus()
    events_seen: list[str] = []
    bus.subscribe(lambda ev: events_seen.append(ev.kind))

    # The public spec for the watermarked generator.
    spec = WatermarkSpec(
        name="frontier-runtime-v1",
        key=b"sk_demo_watermarker_2026_05_18_revoke-on-leak",
        gamma=0.5,           # half of vocab on green list
        delta=2.0,           # generator added +2.0 logits to green tokens
        left_context=1,
        selfhash=False,
    )

    wm = Watermarker(
        WatermarkerConfig(
            seed=0,
            alpha=0.01,          # type-I error: 1 %
            bootstrap_b=300,
            confidence=0.99,
        ),
        spec=spec,
        event_bus=bus,
    )

    banner("1. Calibration pool — 40 labelled trials")
    # 20 marked, 20 unmarked.  The simulator is a black-box rejection-
    # sampling oracle that mirrors the marked / unmarked token
    # distributions a real generator would produce.
    for i in range(20):
        wm.submit(wm.simulate_trial(spec, doc_id=f"marked_{i:02d}",
                                    n_tokens=200, marked=True, seed=i))
    for i in range(20):
        wm.submit(wm.simulate_trial(spec, doc_id=f"unmarked_{i:02d}",
                                    n_tokens=200, marked=False, seed=100 + i))
    print(f"   submitted: {wm.n_trials}")
    n_marked = sum(1 for r in wm.reports() if r.truth_value)
    n_unmarked = sum(1 for r in wm.reports() if r.truth_value is False)
    print(f"   marked:    {n_marked:2d}   unmarked: {n_unmarked:2d}")

    banner("2. Fit threshold on the labelled pool (Youden's J)")
    thr = wm.calibrate()
    print(f"   threshold (z):  {thr.threshold:.4f}")
    print(f"   AUROC:          {thr.auroc:.3f}  "
          f"[{thr.auroc_lower:.3f}, {thr.auroc_upper:.3f}] @ {int(100*thr.confidence)}%")
    print(f"   Youden's J:     {thr.youden_j:.3f}")
    print(f"   TPR / FPR @ t:  {thr.tpr_at_threshold:.2f} / {thr.fpr_at_threshold:.2f}")
    print(f"   labelled (+/-): {thr.n_positive} / {thr.n_negative}")

    banner("3. Anytime-valid audit on the per-token green rate")
    audit = wm.audit()
    print(f"   tokens seen:            {audit.n_tokens_seen}")
    print(f"   green / total:          {audit.n_green_seen} / {audit.n_tokens_seen}")
    print(f"   running green rate:     {audit.running_rate:.3f}  (γ = {audit.gamma:.2f})")
    print(f"   Clopper-Pearson 99% CI: "
          f"[{audit.rate_lower_clopper_pearson:.3f}, "
          f"{audit.rate_upper_clopper_pearson:.3f}]")
    print(f"   e-value:                {audit.e_value:.3g}")
    print(f"   reject H_0 (one-sided): {audit.rejected_h0}")

    banner("4. Certificate — what the coordinator sees")
    cert = wm.certify()
    print(f"   verdict:                {cert.verdict}")
    print(f"   recommendation:         {cert.recommendation}")
    print(f"   spec fingerprint:       {cert.spec_fingerprint[:16]}…")
    print(f"   green-rate CI:          "
          f"[{cert.rate_lower_cp:.3f}, {cert.rate_upper_cp:.3f}]")
    print(f"   AUROC:                  {cert.auroc:.3f}  "
          f"[{cert.auroc_lower:.3f}, {cert.auroc_upper:.3f}]")
    print(f"   threshold (z):          {cert.threshold:.4f}")
    print(f"   Holm smallest adj. p:   {cert.holm_smallest_p}")
    print(f"   BH FDR threshold p:     {cert.fdr_threshold_p}")
    print(f"   certificate hash:       {cert.fingerprint_hash[:16]}…")

    banner("5. Live gating — three regimes")
    # Regime A: a freshly-generated marked document.
    doc_a = simulate_marked_document(spec, "live_marked", 250, seed=999)
    rep_a = wm.score_only(Trial(document=doc_a, spec=spec))
    print(f"   A (marked):     n={rep_a.n_scoreable:3d}  "
          f"green={rep_a.green_fraction:.3f}  z={rep_a.z_score:6.2f}  "
          f"p={rep_a.chosen_p_value:.3g}  verdict={rep_a.verdict}")

    # Regime B: an unwatermarked document trying to pass as marked.
    doc_b = simulate_unmarked_document(spec, "live_unmarked", 250, seed=998)
    rep_b = wm.score_only(Trial(document=doc_b, spec=spec))
    print(f"   B (unmarked):   n={rep_b.n_scoreable:3d}  "
          f"green={rep_b.green_fraction:.3f}  z={rep_b.z_score:6.2f}  "
          f"p={rep_b.chosen_p_value:.3g}  verdict={rep_b.verdict}")

    # Regime C: an adversarial paraphrase that stripped the mark.
    doc_c = simulate_stripped_document(spec, "live_stripped", 250, seed=997)
    rep_c = wm.score_only(Trial(document=doc_c, spec=spec))
    print(f"   C (stripped):   n={rep_c.n_scoreable:3d}  "
          f"green={rep_c.green_fraction:.3f}  z={rep_c.z_score:6.2f}  "
          f"p={rep_c.chosen_p_value:.3g}  verdict={rep_c.verdict}")

    banner("6. Verify-mode audit — alert on dewatermarking")
    # A second Watermarker, this one configured to *verify* that the
    # incoming text is still marked.  Submit a stream of stripped
    # documents and watch the audit alert.
    wm_v = Watermarker(
        WatermarkerConfig(
            polarity=POLARITY_VERIFY_WATERMARK,
            alpha=0.01,
            seed=1,
        ),
        spec=spec,
    )
    for i in range(5):
        d = simulate_stripped_document(spec, f"stripped_{i:02d}", 200, seed=i)
        wm_v.submit(Trial(document=d, spec=spec, truth=False))
    cert_v = wm_v.certify()
    print(f"   verdict:                {cert_v.verdict}")
    print(f"   recommendation:         {cert_v.recommendation}")
    print(f"   running green rate:     {cert_v.green_rate:.3f}  (γ = {spec.gamma:.2f})")
    print(f"   reject H_0 (verify):    {cert_v.rejected_h0}")
    if cert_v.recommendation == REC_BLOCK:
        print("   → coordinator BLOCKS the stream: mark has been "
              "removed (paraphrase / key compromise / generator swap).")

    banner("7. Event log — every transition fingerprinted")
    kind_counts: dict[str, int] = {}
    for k in events_seen:
        kind_counts[k] = kind_counts.get(k, 0) + 1
    for kind, count in sorted(kind_counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {kind:28s} {count:4d}")
    print(f"\n   certificate hash: {cert.fingerprint_hash[:16]}…")
    print(f"   verify-mode hash: {cert_v.fingerprint_hash[:16]}…")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
