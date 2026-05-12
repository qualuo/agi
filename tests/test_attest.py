"""Tests for `agi.attest` — the AttestationLedger / AttestedReceipt
tamper-evident hash chain and the `RuntimeAttestor` driver hook."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agi.attest import (
    AttestationLedger,
    AttestedReceipt,
    GENESIS_HASH,
    LedgerCorrupted,
    RuntimeAttestor,
    _canonical_json,
    _sha256_hex,
)
from agi.driver import Receipt


# --- canonical encoding ----------------------------------------------


def test_canonical_json_sorts_keys_and_strips_whitespace() -> None:
    a = _canonical_json({"b": 1, "a": 2})
    b = _canonical_json({"a": 2, "b": 1})
    assert a == b == b'{"a":2,"b":1}'


def test_canonical_json_is_deterministic_for_nested() -> None:
    obj1 = {"x": [3, 1, 2], "y": {"q": 1, "p": 2}}
    obj2 = {"y": {"p": 2, "q": 1}, "x": [3, 1, 2]}
    # list order is meaningful (preserved); object key order is not
    assert _canonical_json(obj1) == _canonical_json(obj2)


# --- AttestationLedger basics ---------------------------------------


def _mk_receipt(ticket_id: str = "t1", cost: float = 0.10) -> Receipt:
    return Receipt(
        ticket_id=ticket_id,
        intent=f"test-{ticket_id}",
        status="completed",
        actual_cost_usd=cost,
        actual_duration_s=1.0,
    )


def test_first_append_links_to_genesis() -> None:
    led = AttestationLedger()
    entry = led.append(_mk_receipt())
    assert entry.seq == 0
    assert entry.prev_hash == GENESIS_HASH
    assert len(entry.entry_hash) == 64
    assert entry.signature == ""  # no key supplied
    assert entry.ticket_id == "t1"


def test_chain_links_forward() -> None:
    led = AttestationLedger()
    e1 = led.append(_mk_receipt("a"))
    e2 = led.append(_mk_receipt("b"))
    e3 = led.append(_mk_receipt("c"))
    assert e2.prev_hash == e1.entry_hash
    assert e3.prev_hash == e2.entry_hash
    assert led.head_hash() == e3.entry_hash
    assert len(led) == 3


def test_clean_chain_verifies() -> None:
    led = AttestationLedger()
    for i in range(5):
        led.append(_mk_receipt(f"t{i}"))
    ok, why = led.verify()
    assert ok, why


def test_empty_ledger_verifies() -> None:
    led = AttestationLedger()
    ok, why = led.verify()
    assert ok and why is None
    assert led.head() is None
    assert led.head_hash() == GENESIS_HASH


# --- tamper detection -----------------------------------------------


def test_mutating_receipt_breaks_chain() -> None:
    led = AttestationLedger()
    led.append(_mk_receipt("a", cost=0.10))
    led.append(_mk_receipt("b", cost=0.20))
    led.append(_mk_receipt("c", cost=0.30))

    # Tamper with entry 1's receipt: silently lower the cost.
    led._entries[1].receipt["actual_cost_usd"] = 0.05
    ok, why = led.verify()
    assert not ok
    assert "1" in (why or "")
    assert "receipt_hash" in (why or "")


def test_mutating_prev_hash_breaks_chain() -> None:
    led = AttestationLedger()
    led.append(_mk_receipt("a"))
    e2 = led.append(_mk_receipt("b"))
    # Splice in a fake prev_hash to detach the chain.
    led._entries[1] = AttestedReceipt(
        seq=e2.seq,
        ticket_id=e2.ticket_id,
        receipt_hash=e2.receipt_hash,
        prev_hash="f" * 64,
        entry_hash=e2.entry_hash,
        signature=e2.signature,
        issued_ts=e2.issued_ts,
        receipt=e2.receipt,
    )
    ok, why = led.verify()
    assert not ok
    assert "prev_hash" in (why or "")


def test_reordering_breaks_chain() -> None:
    led = AttestationLedger()
    led.append(_mk_receipt("a"))
    led.append(_mk_receipt("b"))
    led.append(_mk_receipt("c"))
    # Swap entry 1 and 2; seq mismatch alone catches this.
    led._entries[1], led._entries[2] = led._entries[2], led._entries[1]
    ok, why = led.verify()
    assert not ok


# --- HMAC signatures ------------------------------------------------


def test_signed_entries_verify_with_correct_key() -> None:
    led = AttestationLedger(key=b"shared-secret")
    e = led.append(_mk_receipt("a"))
    assert e.signature != ""
    ok, why = led.verify()
    assert ok, why
    ok2, _ = led.verify(require_signatures=True)
    assert ok2


def test_signed_entries_fail_with_wrong_key() -> None:
    led = AttestationLedger(key=b"shared-secret")
    led.append(_mk_receipt("a"))
    ok, why = led.verify(key=b"different-secret")
    assert not ok
    assert "signature" in (why or "")


def test_require_signatures_rejects_unsigned() -> None:
    led = AttestationLedger()  # no key → empty signatures
    led.append(_mk_receipt("a"))
    ok, why = led.verify(require_signatures=True)
    assert not ok
    assert "signature" in (why or "")


def test_attested_receipt_verify_signature_helper() -> None:
    led = AttestationLedger(key=b"k1")
    e = led.append(_mk_receipt("a"))
    assert e.verify_signature(b"k1")
    assert not e.verify_signature(b"wrong")
    assert not e.verify_signature(None)


# --- persistence + reload -------------------------------------------


def test_persists_and_reloads(tmp_path: Path) -> None:
    p = tmp_path / "ledger.jsonl"
    led = AttestationLedger(path=p, key=b"k")
    for i in range(4):
        led.append(_mk_receipt(f"t{i}"))
    expected_head = led.head_hash()

    led2 = AttestationLedger(path=p, key=b"k")
    assert len(led2) == 4
    assert led2.head_hash() == expected_head
    ok, _ = led2.verify()
    assert ok


def test_reload_on_corrupted_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "ledger.jsonl"
    led = AttestationLedger(path=p)
    led.append(_mk_receipt("a"))
    led.append(_mk_receipt("b"))

    # Tamper with the second line: lower the cost without recomputing hashes.
    lines = p.read_text().splitlines()
    obj = json.loads(lines[1])
    obj["receipt"]["actual_cost_usd"] = 999.99
    lines[1] = json.dumps(obj)
    p.write_text("\n".join(lines) + "\n")

    with pytest.raises(LedgerCorrupted):
        AttestationLedger(path=p)


def test_reload_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "ledger.jsonl"
    led = AttestationLedger(path=p)
    led.append(_mk_receipt("a"))
    led.append(_mk_receipt("b"))
    # Append a blank line at the end (common edit-tool artifact).
    with p.open("a") as f:
        f.write("\n")
    led2 = AttestationLedger(path=p)
    assert len(led2) == 2


def test_appends_to_existing_ledger(tmp_path: Path) -> None:
    p = tmp_path / "ledger.jsonl"
    led = AttestationLedger(path=p)
    led.append(_mk_receipt("a"))
    led2 = AttestationLedger(path=p)
    led2.append(_mk_receipt("b"))
    assert len(led2) == 2
    ok, _ = led2.verify()
    assert ok


# --- export / import roundtrip --------------------------------------


def test_export_roundtrip_via_from_entries() -> None:
    led = AttestationLedger(key=b"k")
    for i in range(3):
        led.append(_mk_receipt(f"t{i}"))
    exported = led.export()
    led2 = AttestationLedger.from_entries(exported, key=b"k")
    assert led2.head_hash() == led.head_hash()
    assert len(led2) == 3


def test_from_entries_rejects_tampered_export() -> None:
    led = AttestationLedger()
    led.append(_mk_receipt("a"))
    led.append(_mk_receipt("b"))
    exported = led.export()
    exported[0]["receipt"]["actual_cost_usd"] = 12345.0
    with pytest.raises(LedgerCorrupted):
        AttestationLedger.from_entries(exported)


def test_export_jsonl_is_canonical() -> None:
    led = AttestationLedger()
    led.append(_mk_receipt("a"))
    jsonl = led.export_jsonl()
    parsed = [json.loads(line) for line in jsonl.splitlines()]
    assert parsed[0]["seq"] == 0
    # canonical means re-encoding yields the same bytes
    re_encoded = _canonical_json(parsed[0]).decode("utf-8")
    assert re_encoded == jsonl.splitlines()[0]


# --- lookups + proofs -----------------------------------------------


def test_get_by_ticket_returns_latest() -> None:
    led = AttestationLedger()
    led.append(_mk_receipt("a", cost=0.10))
    led.append(_mk_receipt("b"))
    e3 = led.append(_mk_receipt("a", cost=0.20))  # same ticket id again
    assert led.get_by_ticket("a") is e3
    assert led.get_by_ticket("missing") is None


def test_proof_of_inclusion_carries_head_commitment() -> None:
    led = AttestationLedger(key=b"k", namespace="prod-east")
    led.append(_mk_receipt("a"))
    led.append(_mk_receipt("b"))
    led.append(_mk_receipt("c"))
    proof = led.proof_of_inclusion("b")
    assert proof is not None
    assert proof["ticket_id"] == "b"
    assert proof["chain_length"] == 3
    assert proof["head_hash"] == led.head_hash()
    assert proof["namespace"] == "prod-east"
    # The proof entry's own hashes must self-verify.
    entry = AttestedReceipt.from_dict(proof["entry"])
    assert entry.recompute_receipt_hash() == entry.receipt_hash
    assert entry.recompute_entry_hash() == entry.entry_hash


def test_proof_for_missing_ticket_is_none() -> None:
    led = AttestationLedger()
    led.append(_mk_receipt("a"))
    assert led.proof_of_inclusion("nope") is None


# --- coercion -------------------------------------------------------


def test_accepts_plain_dict_receipt() -> None:
    led = AttestationLedger()
    e = led.append({"ticket_id": "raw", "status": "completed", "cost": 0.01})
    assert e.ticket_id == "raw"
    ok, _ = led.verify()
    assert ok


def test_accepts_receipt_dataclass_to_dict() -> None:
    led = AttestationLedger()
    r = _mk_receipt("dc")
    e = led.append(r)
    # The stored receipt should equal r.to_dict()
    assert e.receipt["ticket_id"] == "dc"
    assert e.receipt["intent"] == "test-dc"


def test_rejects_unserializable() -> None:
    led = AttestationLedger()
    with pytest.raises(TypeError):
        led.append(object())


# --- RuntimeAttestor hook -------------------------------------------


def test_runtime_attestor_appends_on_call() -> None:
    led = AttestationLedger()
    att = RuntimeAttestor(led)
    e = att(_mk_receipt("a"))
    assert e is not None
    assert att.appended == 1
    assert len(led) == 1


def test_runtime_attestor_swallows_errors() -> None:
    led = AttestationLedger()
    att = RuntimeAttestor(led)
    # object() can't be serialized; the attestor must not propagate.
    out = att(object())
    assert out is None
    assert att.last_error is not None
    assert att.appended == 0


# --- driver integration ---------------------------------------------


def test_driver_calls_attestor_on_persisted_receipt() -> None:
    # Use the driver's _persist_receipt directly with a stub-shaped
    # driver instance so we don't have to spin up a Runtime.
    from agi.driver import RuntimeDriver
    from agi.runtime import Runtime

    led = AttestationLedger(key=b"k")
    att = RuntimeAttestor(led)
    drv = RuntimeDriver(runtime=Runtime(), attestor=att)

    drv._persist_receipt(_mk_receipt("integ"))
    assert att.appended == 1
    assert led.get_by_ticket("integ") is not None
    ok, _ = led.verify()
    assert ok


def test_driver_attestor_failure_does_not_block_persist() -> None:
    from agi.driver import RuntimeDriver
    from agi.runtime import Runtime

    class Boom:
        def __call__(self, receipt: object) -> None:
            raise RuntimeError("attestor exploded")

    drv = RuntimeDriver(runtime=Runtime(), attestor=Boom())
    # Should not raise.
    drv._persist_receipt(_mk_receipt("safe"))


# --- determinism / commitment property ------------------------------


def test_same_receipts_same_head_hash() -> None:
    """Two independent ledgers built from the same receipts at the
    same issued_ts produce the same head hash — the chain is a pure
    function of (sequence, content, timestamps)."""
    r1 = _mk_receipt("a")
    r2 = _mk_receipt("b")
    led1 = AttestationLedger()
    e1a = led1.append(r1)
    e1b = led1.append(r2)

    # Reconstruct via from_entries from the exported dicts — same hashes.
    led2 = AttestationLedger.from_entries(led1.export())
    assert led2.head_hash() == led1.head_hash()
    assert led2.entries()[0].entry_hash == e1a.entry_hash
    assert led2.entries()[1].entry_hash == e1b.entry_hash
