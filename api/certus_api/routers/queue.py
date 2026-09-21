"""Referral queue and ophthalmologist review.

Queue order is the one the capacity model assumes: urgent first, then whoever is closest to
breaching the 48-hour turnaround. That is what makes one reviewer enough for a district.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .. import audit
from ..db import get_db
from ..models import AppUser, Encounter, Image, Prediction, Referral, Review, now
from ..schemas import AssignIn, ReferralOut, ReviewIn, ReviewOut
from ..security import Principal, requires

router = APIRouter(prefix="/v1", tags=["queue"])


def _aware(dt: datetime) -> datetime:
    """SQLite hands back naive datetimes; treat stored times as UTC so SLA arithmetic works."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@router.get("/referrals", response_model=list[ReferralOut])
def list_referrals(state: str | None = Query(None), priority: str | None = Query(None),
                   assigned_to: str | None = Query(None), limit: int = Query(50, le=500),
                   db: Session = Depends(get_db),
                   _: Principal = Depends(requires("technician", "ophthalmologist"))):
    q = select(Referral)
    if state:
        q = q.where(Referral.state == state)
    if priority:
        q = q.where(Referral.priority == priority)
    if assigned_to:
        q = q.where(Referral.assigned_to == assigned_to)
    urgent_first = case((Referral.priority == "urgent", 0), else_=1)
    return db.execute(q.order_by(urgent_first, Referral.sla_due_at).limit(limit)).scalars().all()


@router.get("/referrals/stats")
def queue_stats(db: Session = Depends(get_db),
                _: Principal = Depends(requires("technician", "ophthalmologist"))):
    """Live queue health: what the district dashboard shows and the simulation predicts."""
    rows = db.execute(select(Referral.state, Referral.priority, func.count())
                      .group_by(Referral.state, Referral.priority)).all()
    breached = db.execute(select(func.count()).select_from(Referral)
                          .where(Referral.state != "closed")
                          .where(Referral.sla_due_at < now())).scalar_one()
    closed = db.execute(select(Referral).where(Referral.state == "closed")).scalars().all()
    turnaround = [(r.closed_at - r.created_at).total_seconds() / 3600 for r in closed if r.closed_at]
    within = [t for t in turnaround if t <= 48]
    return {
        "by_state": [{"state": s, "priority": p, "n": n} for s, p, n in rows],
        "open_past_sla": breached,
        "closed": len(closed),
        "median_turnaround_hours": round(sorted(turnaround)[len(turnaround) // 2], 2) if turnaround else None,
        "within_48h_fraction": round(len(within) / len(turnaround), 4) if turnaround else None,
    }


@router.post("/referrals/{referral_id}/assign", response_model=ReferralOut)
def assign(referral_id: str, body: AssignIn, db: Session = Depends(get_db),
           p: Principal = Depends(requires("ophthalmologist"))):
    ref = db.get(Referral, referral_id)
    if ref is None:
        raise HTTPException(404, "referral not found")
    if db.get(AppUser, body.reviewer_id) is None:
        raise HTTPException(404, "reviewer not found")
    ref.assigned_to, ref.state = body.reviewer_id, "assigned"
    audit.record(db, p.actor, "referral.assigned", "referral", ref.id, {"reviewer": body.reviewer_id})
    db.commit()
    return ref


@router.get("/referrals/{referral_id}/context")
def review_context(referral_id: str, db: Session = Depends(get_db),
                   _: Principal = Depends(requires("ophthalmologist"))):
    """Everything a reviewer needs in one call: the model's grade, its evidence, its uncertainty."""
    ref = db.get(Referral, referral_id)
    if ref is None:
        raise HTTPException(404, "referral not found")
    enc = db.get(Encounter, ref.encounter_id)
    images = db.execute(select(Image).where(Image.encounter_id == enc.id)).scalars().all()
    preds = db.execute(select(Prediction).where(Prediction.image_id.in_([i.id for i in images]))
                       ).scalars().all()
    return {
        "referral": {"id": ref.id, "priority": ref.priority, "reason": ref.reason,
                     "state": ref.state, "sla_due_at": ref.sla_due_at, "worst_grade": ref.worst_grade,
                     "hours_remaining": round((_aware(ref.sla_due_at) - datetime.now(timezone.utc)
                                               ).total_seconds() / 3600, 2)},
        "encounter": {"id": enc.id, "patient_id": enc.patient_id, "site_id": enc.site_id},
        "eyes": [{"image_id": i.id, "eye": i.eye, "field": i.field,
                  "prediction": next(({"id": p.id, "dr_grade": p.dr_grade, "decision": p.decision,
                                       "p_referable": p.p_referable, "abstained": p.abstained,
                                       "abstain_reason": p.abstain_reason,
                                       "attention": p.attention, "evidence": p.evidence,
                                       "structures": p.structures,
                                       "findings": [{"id": f.id, "type": f.lesion_type,
                                                     "count": f.lesion_count,
                                                     "threshold": f.threshold,
                                                     "peak_prob": f.peak_prob,
                                                     "quadrants": f.quadrant_counts,
                                                     "mask_url": f"/v1/findings/{f.id}/mask"}
                                                    for f in p.findings]}
                                      for p in preds if p.image_id == i.id), None)}
                 for i in images],
    }


@router.post("/referrals/{referral_id}/review", response_model=ReviewOut)
def submit_review(referral_id: str, body: ReviewIn, reviewer_id: str,
                  db: Session = Depends(get_db), p: Principal = Depends(requires("ophthalmologist"))):
    ref = db.get(Referral, referral_id)
    if ref is None:
        raise HTTPException(404, "referral not found")
    if db.get(AppUser, reviewer_id) is None:
        raise HTTPException(404, "reviewer not found")
    rev = Review(referral_id=referral_id, reviewer_id=reviewer_id,
                 **body.model_dump(exclude={"explanation"}))
    db.add(rev)
    ref.state, ref.closed_at = "closed", now()
    enc = db.get(Encounter, ref.encounter_id)
    enc.status, enc.closed_at = "closed", now()
    audit.record(db, p.actor, "referral.reviewed", "referral", ref.id,
                 {"reviewer": reviewer_id, "agreed": body.agreed, "final_grade": body.final_grade,
                  "model_worst_grade": ref.worst_grade})
    if body.explanation is not None:
        # Kept in the hash-chained audit log rather than a new column: no migration for existing
        # databases, and ratings collected for a reader study are tamper-evident.
        audit.record(db, p.actor, "review.explanation_rated", "referral", ref.id,
                     {"reviewer": reviewer_id, "seconds_spent": body.seconds_spent,
                      **body.explanation.model_dump()})
    db.commit()
    return rev


@router.get("/reviews/agreement")
def agreement(db: Session = Depends(get_db), _: Principal = Depends(requires("ophthalmologist"))):
    """How often the reviewer kept the model's grade. The number to watch after deployment."""
    rows = db.execute(select(Review).where(Review.agreed.isnot(None))).scalars().all()
    if not rows:
        return {"n": 0}
    agreed = sum(1 for r in rows if r.agreed)
    spent = [r.seconds_spent for r in rows if r.seconds_spent]
    return {"n": len(rows), "agreement_rate": round(agreed / len(rows), 4),
            "median_seconds_per_review": sorted(spent)[len(spent) // 2] if spent else None}
