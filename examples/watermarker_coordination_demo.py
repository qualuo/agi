"""Watermarker × Attest coordination demo: end-to-end provenance routing.

This is the *coordination engine view*: a runtime composing the
Watermarker provenance gate with the Attest tamper-evident ledger to
produce a single auditable answer to the question

    "is this output (a) provably from our model, (b) provably
     unaltered since emission, (c) safe to forward downstream?"

The pipeline:

  1. **Generator emits.**  The runtime simulates 8 production
     documents from a watermarked generator.
  2. **Adversary tampers.**  A subset of documents is paraphrased
     (stripped of the watermark) by a malicious downstream.
  3. **Coordinator inspects.**  Each document is scored against the
     public ``WatermarkSpec``; verdict + recommendation are produced.
  4. **Tamper-evident chain.**  A combined receipt (watermark
     certificate + content digest) is appended to an
     :class:`AttestationLedger` so a downstream consumer can verify
     both the watermark *and* the chain.
  5. **Coordinator routes.**  Documents whose watermark FAILed are
     blocked; PASSed ones are forwarded with the full provenance
     receipt attached.

Run:  python examples/watermarker_coordination_demo.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.attest import AttestationLedger
from agi.events import EventBus
from agi.watermarker import (
    POLARITY_VERIFY_WATERMARK,
    REC_BLOCK,
    REC_QUARANTINE,
    REC_TRUST,
    Trial,
    VERDICT_FAIL,
    VERDICT_PASS,
    Watermarker,
    WatermarkerConfig,
    WatermarkSpec,
    simulate_marked_document,
    simulate_stripped_document,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def doc_digest(doc) -> str:
    """SHA-256 of the document's token-id stream — the content hash."""
    h = hashlib.sha256()
    for t in doc.tokens:
        h.update(t.token_id.to_bytes(8, "little"))
    return h.hexdigest()


def main() -> int:
    bus = EventBus()
    events_seen: list[str] = []
    bus.subscribe(lambda ev: events_seen.append(ev.kind))

    # The runtime's signed watermark spec.  Production: this is loaded
    # from a secrets store and re-keyed periodically.
    spec = WatermarkSpec(
        name="prod-watermark-2026Q2",
        key=b"prod_wm_key_2026Q2_revocation_id=874203",
        gamma=0.5,
        delta=2.5,
    )

    # Verify-mode watermarker: H0 = "still marked".  Used at the gate.
    wm = Watermarker(
        WatermarkerConfig(
            polarity=POLARITY_VERIFY_WATERMARK,
            alpha=0.01,
            seed=0,
        ),
        spec=spec,
        event_bus=bus,
    )

    # Tamper-evident ledger.  Production: persisted to disk.  Here we
    # use the in-memory variant with a session signing key.
    ledger = AttestationLedger(key=b"session_signing_key_2026Q2_8d2f0a")

    banner("1. Generator emits 8 watermarked documents")
    docs = []
    for i in range(8):
        doc = simulate_marked_document(spec, f"doc_{i:02d}", 300, seed=i)
        docs.append(doc)
    print(f"   {len(docs)} marked documents emitted")

    banner("2. Adversary strips the mark on documents 3, 5, 7")
    adversary_targets = {3, 5, 7}
    for i in adversary_targets:
        docs[i] = simulate_stripped_document(
            spec, f"doc_{i:02d}", 300, seed=100 + i,
        )
    print(f"   adversary stripped: {sorted(adversary_targets)}")

    banner("3. Coordinator routes — gate each document, attest, forward")
    forwarded: list[str] = []
    blocked: list[str] = []
    for doc in docs:
        # Score the document.
        trial_report = wm.submit(Trial(document=doc, spec=spec))
        # Mint a certificate for *this* document only.
        # (We reset between docs so each certificate is per-document.)
        cert = wm.certify()
        # Build a combined receipt the consumer can verify end-to-end.
        receipt = {
            "ticket_id": doc.doc_id,
            "kind": "watermarker_provenance",
            "content_sha256": doc_digest(doc),
            "n_tokens": trial_report.n_tokens,
            "n_scoreable": trial_report.n_scoreable,
            "n_green": trial_report.n_green,
            "green_fraction": trial_report.green_fraction,
            "z_score": trial_report.z_score,
            "chosen_p_value": trial_report.chosen_p_value,
            "rate_lower_cp": trial_report.rate_lower_cp,
            "rate_upper_cp": trial_report.rate_upper_cp,
            "verdict": cert.verdict,
            "recommendation": cert.recommendation,
            "spec_name": cert.spec_name,
            "spec_fingerprint": cert.spec_fingerprint,
            "wm_fingerprint": cert.fingerprint_hash,
            "rejected_h0": cert.rejected_h0,
            "alpha": wm.config.alpha,
            "polarity": wm.config.polarity,
        }
        entry = ledger.append(receipt)
        # Routing decision.
        if cert.recommendation in (REC_TRUST,):
            forwarded.append(doc.doc_id)
        else:
            blocked.append(doc.doc_id)
        # Reset per-document so the next certificate is independent.
        wm.reset()
        flag = "✓" if cert.recommendation == REC_TRUST else "✗"
        print(f"   {flag} {doc.doc_id}: rate={trial_report.green_fraction:.3f} "
              f"verdict={cert.verdict:14s} rec={cert.recommendation:10s} "
              f"entry={entry.entry_hash[:10]}")

    banner("4. Verify ledger integrity end-to-end")
    ok = ledger.verify(require_signatures=True, key=ledger._key)
    print(f"   ledger entries:        {len(ledger)}")
    print(f"   head hash:             {ledger.head_hash()[:16]}…")
    print(f"   chain integrity:       {ok}")
    forwarded_consistent = len(forwarded) == 8 - len(adversary_targets)
    blocked_consistent = set(blocked) == {f"doc_{i:02d}" for i in adversary_targets}
    print(f"   forwarded:             {len(forwarded)}/{len(docs)} "
          f"({'expected' if forwarded_consistent else 'UNEXPECTED'})")
    print(f"   blocked:               {len(blocked)}/{len(docs)}")
    if blocked_consistent:
        print(f"   blocked = adversary:   YES (perfect catch)")
    else:
        print(f"   blocked = adversary:   {sorted(blocked)} vs "
              f"{sorted(f'doc_{i:02d}' for i in adversary_targets)}")

    banner("5. Replay-from-ledger — any consumer can verify offline")
    # A downstream consumer that holds the spec, the doc bytes, and the
    # ledger export can re-derive the certificate and confirm chain
    # integrity without contacting the runtime.
    sample = next(iter(ledger))
    print(f"   entry seq:             {sample.seq}")
    print(f"   ticket_id:             {sample.ticket_id}")
    print(f"   prev_hash:             {sample.prev_hash[:16]}…")
    print(f"   entry_hash:            {sample.entry_hash[:16]}…")
    print(f"   signature ok:          {sample.verify_signature(ledger._key)}")
    print(f"   receipt verdict:       {sample.receipt['verdict']}")
    print(f"   receipt recommendation: {sample.receipt['recommendation']}")
    print(f"   watermark p-value:     {sample.receipt['chosen_p_value']:.3g}")

    banner("6. Event log — coordination layer visibility")
    kind_counts: dict[str, int] = {}
    for k in events_seen:
        kind_counts[k] = kind_counts.get(k, 0) + 1
    for kind, count in sorted(kind_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {kind:28s} {count:4d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
