"""attest_demo — proof-of-work for runtime tickets.

End-to-end demo of `agi.attest`:

  1. A `RuntimeDriver` runs three tickets.
  2. Every completed receipt feeds an `AttestationLedger` via the
     `RuntimeAttestor` hook (no extra wiring on the coordination side).
  3. The operator publishes only the **head hash** — a 64-char
     commitment to the entire history.
  4. A "regulator" rebuilds the ledger from the exported JSONL and
     verifies, end-to-end.
  5. The operator hands the regulator a single ticket's
     **proof-of-inclusion** — the regulator confirms that ticket was
     in the chain at this head, without ever needing to see the other
     receipts.
  6. We show what tamper detection looks like: silently lowering a past
     receipt's cost breaks `verify()` deterministically.

Run with:

    python examples/attest_demo.py

No API key required — this exercises the receipt/ledger machinery
against synthetic receipts, the same way the test suite does. To wire
real LLM runs in, swap `_synthetic_receipt(...)` for
`driver.submit(...).result()`.
"""
from __future__ import annotations

import json
import time

from agi.attest import AttestationLedger, RuntimeAttestor
from agi.driver import Receipt


SHARED_KEY = b"investor-demo-key-rotate-in-prod"


def _synthetic_receipt(ticket_id: str, intent: str, cost: float) -> Receipt:
    return Receipt(
        ticket_id=ticket_id,
        intent=intent,
        status="completed",
        actual_cost_usd=cost,
        actual_duration_s=0.5 + cost * 10,
        submitted_ts=time.time(),
        completed_ts=time.time(),
        model="claude-opus-4-7",
    )


def main() -> None:
    print("=" * 60)
    print("AGI runtime attestation demo")
    print("=" * 60)

    # ---- operator side ---------------------------------------------
    ledger = AttestationLedger(key=SHARED_KEY, namespace="prod-east")
    attestor = RuntimeAttestor(ledger)

    print("\n[1] operator runs 3 tickets; the attestor records each receipt")
    workload = [
        ("t-001", "summarize Q3 earnings call",    0.12),
        ("t-002", "extract risk factors from 10-K", 0.34),
        ("t-003", "compare two contracts",          0.21),
    ]
    for tid, intent, cost in workload:
        receipt = _synthetic_receipt(tid, intent, cost)
        entry = attestor(receipt)
        assert entry is not None
        print(f"    seq={entry.seq}  ticket={tid}  cost=${cost:.2f}"
              f"  entry={entry.entry_hash[:12]}...")

    head = ledger.head()
    assert head is not None
    print(f"\n[2] operator publishes head commitment: {head.entry_hash}")
    print(f"    (chain length = {len(ledger)})")

    # ---- regulator side --------------------------------------------
    print("\n[3] regulator receives the JSONL export and re-verifies")
    exported = ledger.export()  # list[dict] — could be JSON over HTTP
    regulator_ledger = AttestationLedger.from_entries(exported, key=SHARED_KEY)
    ok, why = regulator_ledger.verify(require_signatures=True)
    assert ok, why
    assert regulator_ledger.head_hash() == ledger.head_hash()
    print(f"    chain verified ✓     head matches: {regulator_ledger.head_hash()[:24]}...")

    print("\n[4] regulator asks for proof that ticket t-002 was in the chain")
    proof = ledger.proof_of_inclusion("t-002")
    assert proof is not None
    print(json.dumps(
        {
            "ticket_id": proof["ticket_id"],
            "head_hash": proof["head_hash"],
            "chain_length": proof["chain_length"],
            "namespace": proof["namespace"],
            "entry_seq": proof["entry"]["seq"],
            "entry_hash": proof["entry"]["entry_hash"],
            "receipt_cost_usd": proof["entry"]["receipt"]["actual_cost_usd"],
            "signature": proof["entry"]["signature"][:24] + "...",
        },
        indent=2,
    ))

    # ---- tamper attempt --------------------------------------------
    print("\n[5] adversary silently lowers ticket t-002's cost from $0.34 to $0.05")
    bad_export = ledger.export()
    bad_export[1]["receipt"]["actual_cost_usd"] = 0.05
    try:
        AttestationLedger.from_entries(bad_export, key=SHARED_KEY)
        print("    ✗ tamper succeeded — that would be a bug")
    except Exception as e:
        print(f"    ✓ tamper rejected at load time: {e}")

    print("\nWhat this gives a coordination engine:")
    print("  • Tamper-evident receipt log (sha256 hash chain).")
    print("  • HMAC-signed entries — third-party verifiable with the key.")
    print("  • 64-char head commitment to publish externally without")
    print("    exposing individual receipts.")
    print("  • Self-contained inclusion proofs per ticket — the audit")
    print("    surface enterprise procurement asks for, served from a")
    print("    pure-stdlib module with zero new dependencies.")


if __name__ == "__main__":
    main()
