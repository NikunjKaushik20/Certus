"""Append-only audit trail.

Every row carries the hash of the previous row, so removing or editing a past entry breaks the
chain and `verify_chain` finds it. This is what makes a screening record defensible: the grade a
patient was given, the model version that produced it and the clinician who overrode it cannot be
quietly rewritten later.
"""
import hashlib
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AuditLog

GENESIS = "0" * 64


def _hash(prev: str, payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + body).encode()).hexdigest()


def record(db: Session, actor: str, action: str, entity_type: str, entity_id: str,
           payload: dict | None = None) -> AuditLog:
    payload = payload or {}
    prev = db.execute(select(AuditLog).order_by(AuditLog.seq.desc()).limit(1)).scalar_one_or_none()
    prev_hash = prev.payload_hash if prev else GENESIS
    row = AuditLog(actor=actor, action=action, entity_type=entity_type, entity_id=entity_id,
                   payload=payload, prev_hash=prev_hash, payload_hash=_hash(prev_hash, payload))
    db.add(row)
    return row


def verify_chain(db: Session) -> dict:
    """Re-walk the chain; returns the first sequence number that does not match, if any."""
    prev_hash = GENESIS
    n = 0
    for row in db.execute(select(AuditLog).order_by(AuditLog.seq)).scalars():
        n += 1
        if row.prev_hash != prev_hash or row.payload_hash != _hash(prev_hash, row.payload):
            return {"ok": False, "entries": n, "broken_at_seq": row.seq}
        prev_hash = row.payload_hash
    return {"ok": True, "entries": n, "head": prev_hash}
