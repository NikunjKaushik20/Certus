"""Append-only audit trail.

Every row carries the hash of the previous row, so removing or editing a past entry breaks the
chain and `verify_chain` finds it. This is what makes a screening record defensible: the grade a
patient was given, the model version that produced it and the clinician who overrode it cannot be
quietly rewritten later.

Integrity guarantee
-------------------
Each entry stores ``prev_hash`` (the hash of the previous entry) and ``payload_hash``
(``sha256(prev_hash || json(payload))``).  Changing any row invalidates every row that
follows it, and deleting a row leaves a gap that ``verify_chain`` detects.
"""
import hashlib
import json
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .models import AuditLog

# The genesis hash anchors the chain — the first entry's prev_hash points here.
GENESIS: str = "0" * 64


def _hash(prev: str, payload: dict) -> str:
    """Compute the chain hash: sha256 of the previous hash concatenated with the canonical payload."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + body).encode()).hexdigest()


def _lock_writers(db: Session) -> None:
    """Hold the database write lock before reading the chain head, until the caller commits.

    Without it, two sessions can read the same head and each append to it: the one that commits
    second forks the chain, and verify_chain reports tampering that never happened. That is what
    a request writing while the worker held a long transaction produced. On SQLite a write
    statement takes the lock even when it matches no rows; on Postgres a transaction advisory lock
    does the same job."""
    dialect = db.get_bind().dialect.name
    if dialect == "sqlite":
        db.execute(text("UPDATE audit_log SET seq = seq WHERE 0"))
    elif dialect == "postgresql":
        db.execute(text("SELECT pg_advisory_xact_lock(7426811)"))


def record(
    db: Session,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    payload: Optional[dict] = None,
) -> AuditLog:
    """Append a new entry to the audit chain and return the created row.

    The caller must commit the session — this function only adds the row so it can
    participate in the same transaction as the action being audited.
    """
    payload = payload or {}
    _lock_writers(db)
    # Fetch the most recent entry to continue the chain
    prev = db.execute(select(AuditLog).order_by(AuditLog.seq.desc()).limit(1)).scalar_one_or_none()
    prev_hash = prev.payload_hash if prev else GENESIS
    row = AuditLog(actor=actor, action=action, entity_type=entity_type, entity_id=entity_id,
                   payload=payload, prev_hash=prev_hash, payload_hash=_hash(prev_hash, payload))
    db.add(row)
    return row


def verify_chain(db: Session) -> dict:
    """Re-walk the entire chain from genesis; returns the first sequence number that does not match.

    Returns a dict with ``ok: True`` and the head hash when the chain is intact, or
    ``ok: False`` and ``broken_at_seq`` pointing to the first tampered entry.
    """
    prev_hash = GENESIS
    n = 0
    for row in db.execute(select(AuditLog).order_by(AuditLog.seq)).scalars():
        n += 1
        # Each row must reference the previous row's hash and its own payload must re-hash correctly
        if row.prev_hash != prev_hash or row.payload_hash != _hash(prev_hash, row.payload):
            return {"ok": False, "entries": n, "broken_at_seq": row.seq}
        prev_hash = row.payload_hash
    return {"ok": True, "entries": n, "head": prev_hash}
