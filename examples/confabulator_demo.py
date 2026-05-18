"""Confabulator demo: detect hallucinations by semantic entropy.

The pitch in one runnable script (no API key required):

  1. A coordination engine has access to ``K = 5`` independently-sampled
     completions per prompt from an opaque LLM worker.  It does NOT
     have access to model internals, ground truth at serving time, or
     any other signal beyond the four returned strings (and, if the
     transport returns them, the per-completion log-probabilities).

  2. It asks the runtime: *"is this answer a confabulation?"*

  3. The :class:`Confabulator` primitive clusters the samples by
     semantic equivalence (Farquhar et al. 2024, Nature), computes
     semantic + lexical + predictive + SelfCheck entropies, fuses them
     into a calibrated score, and — once a small labelled pool is
     fitted — returns a verdict + recommendation the coordinator can
     route on.

  4. Every step writes a fingerprinted event to the EventBus so any
     auditor can replay the decision later, byte-for-byte.

Run:  python examples/confabulator_demo.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.confabulator import (
    REC_ESCALATE,
    REC_QUARANTINE,
    REC_REGENERATE,
    REC_RESTRICT,
    REC_TRUST,
    Confabulator,
    ConfabulatorConfig,
    Sample,
    synthetic_control_trials,
    synthetic_trials,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def main() -> int:
    bus = EventBus()
    events_seen: list[str] = []
    bus.subscribe(lambda ev: events_seen.append(ev.kind))

    c = Confabulator(
        ConfabulatorConfig(
            seed=0,
            budget_p0=0.10,      # documented hallucination budget: 10 %
            alpha=0.05,          # anytime-valid type-I error
            bootstrap_b=300,
        ),
        bus=bus,
    )

    banner("1. Calibration pool — 50 labelled trials")
    cal = synthetic_trials(n_truthful=25, n_hallucinated=25, k=5, seed=0)
    for t in cal:
        c.submit(t.prompt_id, t.samples, truth=t.truth)
    print(f"   submitted: {c.n_trials}")
    print(f"   truthful: {sum(1 for t in cal if t.truth):2d}   "
          f"hallucinated: {sum(1 for t in cal if not t.truth):2d}")

    banner("2. Fit threshold on the labelled pool (Youden's J)")
    thr = c.fit_threshold()
    print(f"   threshold:     {thr.threshold:.4f}")
    print(f"   AUROC:         {thr.auroc:.3f}  "
          f"[{thr.auroc_lower:.3f}, {thr.auroc_upper:.3f}] @ {int(100*thr.confidence)}%")
    print(f"   Youden's J:    {thr.youden_j:.3f}")
    print(f"   TPR / FPR @ t: {thr.tpr_at_threshold:.2f} / {thr.fpr_at_threshold:.2f}")
    print(f"   labelled (+/-): {thr.n_positive} / {thr.n_negative}")

    banner("3. Anytime-valid audit on the labelled pool")
    audit = c.audit()
    print(f"   running rate:           {audit.running_rate:.3f}")
    print(f"   Clopper-Pearson 95% CI: "
          f"[{audit.rate_lower_clopper_pearson:.3f}, "
          f"{audit.rate_upper_clopper_pearson:.3f}]")
    print(f"   e-value:                {audit.e_value:.2f}  "
          f"(reject H_0: p ≤ {audit.p0:.2f}: {audit.rejected_h0})")

    banner("4. Control pool — irreducible noise floor")
    ctrls = synthetic_control_trials(n=20, k=5, seed=42)
    for t in ctrls:
        c.submit(t.prompt_id, t.samples, control=True)
    floor = c.control_singleton_rate()
    print(f"   control trials: {len(ctrls)}")
    print(f"   singleton rate: {floor:.2f}  "
          f"(fraction that collapsed to one cluster)")
    print(f"   ↳ a higher singleton rate means the equivalence oracle "
          f"is better at handling paraphrases on this domain")

    banner("5. Certificate — what the coordinator sees")
    cert = c.certify()
    print(f"   verdict:                {cert.verdict}")
    print(f"   recommendation:         {cert.recommendation}")
    print(f"   hallucination rate:     {cert.hallucination_rate:.3f}")
    print(f"   rate CI:                "
          f"[{cert.rate_lower_cp:.3f}, {cert.rate_upper_cp:.3f}]")
    print(f"   AUROC:                  {cert.auroc:.3f}  "
          f"[{cert.auroc_lower:.3f}, {cert.auroc_upper:.3f}]")
    print(f"   e-value (reject H_0):   {cert.e_value:.2f}  "
          f"(rejected: {cert.rejected_h0})")
    print(f"   threshold:              {cert.threshold:.4f}")
    print(f"   Holm smallest adj. p:   {cert.holm_smallest_p}")
    print(f"   fingerprint:            {cert.fingerprint_hash[:16]}")

    banner("6. Live gating — three regimes")
    # Regime A: a clean, confident answer.
    samples_a = tuple(
        Sample(text="paris", mean_logprob=-0.05, n_tokens=1)
        for _ in range(5)
    )
    rep_a, rec_a = c.gate("live_clean", samples_a)
    print(f"   A (concentrated):  score={rep_a.combined_score:.3f}  rec={rec_a}")

    # Regime B: scattered, uncertain answer.
    samples_b = tuple(
        Sample(text=f"city_{i}", mean_logprob=-2.0, n_tokens=1)
        for i in range(5)
    )
    rep_b, rec_b = c.gate("live_scattered", samples_b)
    print(f"   B (scattered):     score={rep_b.combined_score:.3f}  rec={rec_b}")

    # Regime C: degenerate — no log-probs and same text (the corner
    # case the primitive routes to quarantine when log-probs are the
    # only configured detector).
    rep_c, rec_c = c.gate(
        "live_corner",
        (Sample(text="alpha") for _ in range(3)),
    )
    print(f"   C (no logprobs):   score={rep_c.combined_score:.3f}  rec={rec_c}")

    banner("7. Event log — every transition fingerprinted")
    kind_counts: dict[str, int] = {}
    for k in events_seen:
        kind_counts[k] = kind_counts.get(k, 0) + 1
    for kind, count in sorted(kind_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {kind:28s} {count:4d}")

    # Sanity: events came out of the bus, and the certificate's
    # fingerprint chains every one of them.
    print(f"\n   fingerprint hash: {c.fingerprint_hash[:16]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
