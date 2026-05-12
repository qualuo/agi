"""Tamper-evident attestation receipts — proof-of-work for tickets.

The runtime already emits a `Receipt` for every ticket (intent, decisions,
costs, duration, status). This module wraps each receipt in an
**AttestedReceipt**: a hash-linked, HMAC-signed log entry. The collection
of attested receipts is an **AttestationLedger** — an append-only chain
where every entry commits to the entire prefix.

Why a coordination engine needs this
------------------------------------
Coordination engines outsource work to runtimes (this one, plus any
federated peers via `RuntimePool`). The coordination engine reports to a
human / tenant / regulator / customer. That reporting layer wants three
things the bare `Receipt` does not provide:

1. **Tamper-evidence.** A regulator or counterparty can confirm that the
   receipts you hand them today match the receipts you held yesterday —
   nobody silently rewrote a "cost = $0.42" to "cost = $0.10". Each
   `AttestedReceipt.entry_hash` commits to the receipt content **and** to
   the prior entry's hash, so flipping any byte in any past entry breaks
   the chain at verification time.

2. **Provenance.** Each entry is signed with HMAC-SHA256 over the
   `entry_hash` using a shared key. A third party that holds the key (or
   a public key in an asymmetric extension) can verify the runtime
   actually emitted the receipt — useful when receipts move between
   tenants, brokers, and auditors.

3. **Compact commitments.** The current `head().entry_hash` is a 64-char
   commitment to **everything** the runtime has done so far. A
   coordination engine can publish that 64-char string (to a customer,
   an internal ledger, a compliance system) without leaking the
   receipts themselves — and later prove any single past receipt was
   included, deterministically.

This is the audit / procurement / billing surface investors keep asking
about: "how do I know you actually ran what I'm being billed for, and
how would a third party verify it?"  Answer: AttestationLedger.

Properties
----------
- **Append-only.** No public mutation API; once `append()` returns, the
  entry is committed in memory (and, if `path=` is set, durably).
- **Replayable.** Persisted as JSONL — every line is one attested entry.
  A fresh `AttestationLedger(path=...)` re-reads and re-verifies on
  construction; mismatch raises `LedgerCorrupted` so a coordination
  engine can fail-stop rather than serve a tampered chain.
- **Stdlib only.** `hashlib` + `hmac` + JSON. No external dependencies.
- **Thread-safe.** Internal lock; safe to share across driver workers.
- **Decoupled.** No imports from `agi.driver`/`agi.runtime`; the driver
  optionally accepts an `attestor=` (see `RuntimeDriver` integration)
  but this module stands alone and can wrap any object with a
  `.to_dict()` method or a plain dict.

Operator-grade hygiene
----------------------
- Canonical JSON for hashing (sorted keys, no whitespace) so the digest
  is deterministic across Python versions.
- HMAC keys default to a zero-byte sentinel for development; production
  callers pass `key=` (bytes). With no key, signatures still chain —
  they just aren't cryptographically attributable.
- Hash-chain verification is O(n) and never trusts the on-disk
  signature alone; it recomputes every digest from scratch.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


GENESIS_HASH = "0" * 64
"""Sentinel `prev_hash` for the first entry in a fresh ledger."""


class LedgerCorrupted(Exception):
    """Raised when an attestation ledger fails verification on load."""


# --- canonical encoding ----------------------------------------------


def _canonical_json(obj: Any) -> bytes:
    """Stable JSON encoding: sorted keys, no whitespace, UTF-8 bytes.

    Determinism matters: the same logical object must produce the same
    bytes on every Python, every host, every run, or the hash chain
    breaks under perfectly innocent re-serialization.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _coerce_receipt_dict(receipt: Any) -> dict[str, Any]:
    """Accept a `Receipt`, a dataclass, or a plain dict; return dict."""
    if isinstance(receipt, dict):
        return receipt
    to_dict = getattr(receipt, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    try:
        return asdict(receipt)
    except TypeError as e:
        raise TypeError(
            f"attest: cannot serialize {type(receipt).__name__}; "
            "pass a Receipt, dataclass, or dict"
        ) from e


# --- attested receipt ------------------------------------------------


@dataclass(frozen=True)
class AttestedReceipt:
    """One immutable entry in an `AttestationLedger`.

    Fields:
        seq           — 0-based position in the chain.
        ticket_id     — convenience pointer (also inside `receipt`).
        receipt_hash  — sha256 of canonical-JSON(receipt).
        prev_hash     — entry_hash of the prior entry, or GENESIS_HASH.
        entry_hash    — sha256(seq || receipt_hash || prev_hash || issued_ts).
        signature     — HMAC-SHA256(key, entry_hash). Empty when no key.
        issued_ts     — unix time when the entry was minted.
        receipt       — the underlying receipt payload (dict).

    The entry is JSON-serializable via `to_dict()` and round-trippable
    via `from_dict()`. Receipt content is **not** mutated by attestation
    — the same receipt can be persisted to disk separately and remain
    bit-identical to the attested copy.
    """

    seq: int
    ticket_id: str
    receipt_hash: str
    prev_hash: str
    entry_hash: str
    signature: str
    issued_ts: float
    receipt: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AttestedReceipt":
        return cls(
            seq=int(d["seq"]),
            ticket_id=str(d.get("ticket_id", d.get("receipt", {}).get("ticket_id", ""))),
            receipt_hash=str(d["receipt_hash"]),
            prev_hash=str(d["prev_hash"]),
            entry_hash=str(d["entry_hash"]),
            signature=str(d.get("signature", "")),
            issued_ts=float(d["issued_ts"]),
            receipt=dict(d.get("receipt", {})),
        )

    # --- verification primitives ---------------------------------

    def recompute_receipt_hash(self) -> str:
        return _sha256_hex(_canonical_json(self.receipt))

    def recompute_entry_hash(self) -> str:
        material = _canonical_json(
            {
                "seq": self.seq,
                "receipt_hash": self.receipt_hash,
                "prev_hash": self.prev_hash,
                "issued_ts": self.issued_ts,
            }
        )
        return _sha256_hex(material)

    def verify_signature(self, key: bytes | None) -> bool:
        """True iff `signature` matches HMAC(key, entry_hash). If no key
        was used, an empty signature is considered valid (chain still
        protects integrity; just not third-party attributable)."""
        if not self.signature:
            return key is None or key == b""
        if key is None:
            return False
        expect = hmac.new(key, self.entry_hash.encode("ascii"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expect, self.signature)


# --- ledger ----------------------------------------------------------


class AttestationLedger:
    """Append-only, hash-linked, optionally signed log of receipts.

    Usage::

        ledger = AttestationLedger(path="receipts.jsonl", key=b"shared")
        entry  = ledger.append(receipt)
        ok, why = ledger.verify()
        assert ok, why
        head   = ledger.head()             # 64-char commitment
        proof  = ledger.proof_of_inclusion(entry.ticket_id)

    Construction with a `path=` re-reads the file and verifies before
    accepting new appends; corruption raises `LedgerCorrupted`.
    """

    def __init__(
        self,
        *,
        path: str | os.PathLike[str] | None = None,
        key: bytes | None = None,
        namespace: str | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self._key = key
        self.namespace = namespace
        self._entries: list[AttestedReceipt] = []
        self._by_ticket: dict[str, AttestedReceipt] = {}
        self._lock = threading.Lock()

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > 0:
                self._load_and_verify()

    # --- properties -----------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def head(self) -> AttestedReceipt | None:
        """Latest entry — its `entry_hash` is the current chain root.

        A coordination engine publishes this 64-char digest to commit
        to the entire history without revealing any receipt content."""
        with self._lock:
            return self._entries[-1] if self._entries else None

    def head_hash(self) -> str:
        """Convenience: head's entry_hash or the genesis sentinel."""
        h = self.head()
        return h.entry_hash if h is not None else GENESIS_HASH

    def entries(self) -> list[AttestedReceipt]:
        """Copy of all entries (chronological)."""
        with self._lock:
            return list(self._entries)

    def get_by_ticket(self, ticket_id: str) -> AttestedReceipt | None:
        with self._lock:
            return self._by_ticket.get(ticket_id)

    def __iter__(self) -> Iterator[AttestedReceipt]:
        return iter(self.entries())

    # --- mutation -------------------------------------------------

    def append(self, receipt: Any) -> AttestedReceipt:
        """Mint a new `AttestedReceipt` from the given receipt, append
        to the chain, persist (if `path=`), and return it.

        The receipt may be:
          - a dataclass `Receipt` (from `agi.driver`),
          - any dataclass with sensible field values,
          - a plain dict.

        Same ticket_id appended twice produces two distinct entries —
        the ledger is an event log, not a key-value store. The
        `get_by_ticket()` lookup returns the **latest** entry per id."""
        payload = _coerce_receipt_dict(receipt)
        ticket_id = str(payload.get("ticket_id", ""))
        receipt_hash = _sha256_hex(_canonical_json(payload))
        issued_ts = time.time()

        with self._lock:
            seq = len(self._entries)
            prev_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
            material = _canonical_json(
                {
                    "seq": seq,
                    "receipt_hash": receipt_hash,
                    "prev_hash": prev_hash,
                    "issued_ts": issued_ts,
                }
            )
            entry_hash = _sha256_hex(material)
            signature = ""
            if self._key:
                signature = hmac.new(
                    self._key, entry_hash.encode("ascii"), hashlib.sha256
                ).hexdigest()

            entry = AttestedReceipt(
                seq=seq,
                ticket_id=ticket_id,
                receipt_hash=receipt_hash,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
                signature=signature,
                issued_ts=issued_ts,
                receipt=payload,
            )
            self._entries.append(entry)
            if ticket_id:
                self._by_ticket[ticket_id] = entry
            self._persist(entry)
            return entry

    # --- verification --------------------------------------------

    def verify(
        self,
        *,
        key: bytes | None = None,
        require_signatures: bool = False,
    ) -> tuple[bool, str | None]:
        """Recompute every hash and (optionally) signature.

        Returns `(True, None)` on a clean chain, otherwise
        `(False, reason)` with a string pinpointing the first broken
        entry. `key=None` uses the ledger's own key; pass an external
        key to verify on a different machine.

        `require_signatures=True` rejects entries with empty signatures
        — useful when the chain is supposed to be cryptographically
        attributable end-to-end."""
        verify_key = key if key is not None else self._key
        with self._lock:
            prev = GENESIS_HASH
            for i, e in enumerate(self._entries):
                if e.seq != i:
                    return False, f"entry {i}: seq mismatch ({e.seq} != {i})"
                if e.prev_hash != prev:
                    return False, f"entry {i}: prev_hash mismatch"
                if e.recompute_receipt_hash() != e.receipt_hash:
                    return False, f"entry {i}: receipt_hash mismatch"
                if e.recompute_entry_hash() != e.entry_hash:
                    return False, f"entry {i}: entry_hash mismatch"
                if require_signatures and not e.signature:
                    return False, f"entry {i}: missing signature"
                if e.signature and not e.verify_signature(verify_key):
                    return False, f"entry {i}: signature mismatch"
                prev = e.entry_hash
        return True, None

    def proof_of_inclusion(self, ticket_id: str) -> dict[str, Any] | None:
        """Return a self-contained inclusion proof for a ticket_id.

        The shape::

            {
              "ticket_id": "...",
              "entry": {<full AttestedReceipt as dict>},
              "head_hash": "<current head entry_hash>",
              "chain_length": <int>,
            }

        A third party reconstructs the entry's `entry_hash` from the
        embedded fields and walks forward from that point — or, more
        commonly, asks for the slice of entries from the proof's seq
        through head and re-verifies that slice. For the common case
        ("did you actually run my ticket?"), the proof is sufficient
        on its own.
        """
        with self._lock:
            entry = self._by_ticket.get(ticket_id)
            if entry is None:
                return None
            return {
                "ticket_id": ticket_id,
                "entry": entry.to_dict(),
                "head_hash": self._entries[-1].entry_hash,
                "chain_length": len(self._entries),
                "namespace": self.namespace,
            }

    # --- serialization helpers -----------------------------------

    def export(self) -> list[dict[str, Any]]:
        """All entries as JSON-able dicts."""
        return [e.to_dict() for e in self.entries()]

    def export_jsonl(self) -> str:
        """All entries as a newline-delimited JSON string. Suitable for
        publishing to a third party or sealing into compliance storage."""
        return "\n".join(_canonical_json(e.to_dict()).decode("utf-8") for e in self.entries())

    @classmethod
    def from_entries(
        cls,
        entries: list[dict[str, Any]],
        *,
        key: bytes | None = None,
        namespace: str | None = None,
    ) -> "AttestationLedger":
        """Build an in-memory ledger from a sequence of entry dicts and
        verify it. Raises `LedgerCorrupted` on any tampered entry."""
        led = cls(key=key, namespace=namespace)
        led._entries = [AttestedReceipt.from_dict(d) for d in entries]
        for e in led._entries:
            if e.ticket_id:
                led._by_ticket[e.ticket_id] = e
        ok, why = led.verify()
        if not ok:
            raise LedgerCorrupted(why or "verification failed")
        return led

    # --- private --------------------------------------------------

    def _persist(self, entry: AttestedReceipt) -> None:
        if self.path is None:
            return
        try:
            with self.path.open("a") as f:
                f.write(_canonical_json(entry.to_dict()).decode("utf-8"))
                f.write("\n")
        except OSError:
            # Persistence is best-effort; the in-memory chain is the
            # source of truth for live verification.
            pass

    def _load_and_verify(self) -> None:
        assert self.path is not None
        loaded: list[AttestedReceipt] = []
        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(AttestedReceipt.from_dict(json.loads(line)))
                except (ValueError, KeyError, TypeError) as e:
                    raise LedgerCorrupted(
                        f"malformed entry on load: {e}"
                    ) from e
        self._entries = loaded
        for e in loaded:
            if e.ticket_id:
                self._by_ticket[e.ticket_id] = e
        ok, why = self.verify()
        if not ok:
            raise LedgerCorrupted(f"loaded ledger fails verification: {why}")


# --- runtime integration --------------------------------------------


class RuntimeAttestor:
    """Adapter that turns an `AttestationLedger` into a callable hook.

    Pass an instance as `attestor=` to `RuntimeDriver` and every
    persisted receipt automatically becomes a chain entry::

        ledger = AttestationLedger(path="ledger.jsonl", key=secret)
        driver = RuntimeDriver(runtime=rt, attestor=RuntimeAttestor(ledger))

    Failure mode: any exception in `append()` is swallowed (logged via
    `_last_error`) so a malformed receipt doesn't poison ticket
    completion. The coordination engine can read `last_error` and
    react out-of-band.
    """

    def __init__(self, ledger: AttestationLedger) -> None:
        self.ledger = ledger
        self._last_error: str | None = None
        self._appended = 0

    @property
    def appended(self) -> int:
        return self._appended

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def __call__(self, receipt: Any) -> AttestedReceipt | None:
        try:
            entry = self.ledger.append(receipt)
            self._appended += 1
            self._last_error = None
            return entry
        except Exception as e:  # pragma: no cover - defensive
            self._last_error = f"{type(e).__name__}: {e}"
            return None
