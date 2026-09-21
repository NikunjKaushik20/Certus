"""Health, model provenance, calibration and audit verification."""
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import audit
from ..db import get_db
from ..inference import get_engine
from ..models import (AuditLog, CalibrationVersion, Encounter, Image, InferenceJob, ModelVersion,
                      Prediction, Referral)
from ..schemas import CalibrationOut, ModelVersionOut
from ..security import Principal, current_principal, requires

router = APIRouter(tags=["admin"])


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz(db: Session = Depends(get_db)):
    """Ready means: database reachable and the model actually loaded on its device."""
    db.execute(select(func.count()).select_from(ModelVersion))
    eng = get_engine()
    return {"status": "ready", "device": eng.device, "checkpoint": eng.checkpoint, "step": eng.step}


@router.get("/v1/me")
def me(p: Principal = Depends(current_principal)):
    """Who the caller is, so a client can gate its own UI without guessing from 403s.

    Returns the role only, never the key: the client already holds the key, and the audit log is the
    place identity is recorded, not the response body.
    """
    return {"role": p.role, "actor": p.actor}


@router.get("/v1/model")
def model_info(_: Principal = Depends(requires("technician", "ophthalmologist"))):
    """What is running, how well it scored, and whether its confidence has been calibrated."""
    return get_engine().describe()


@router.get("/v1/model/versions", response_model=list[ModelVersionOut])
def model_versions(db: Session = Depends(get_db), _: Principal = Depends(requires("admin"))):
    return db.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc())).scalars().all()


@router.get("/v1/calibrations", response_model=list[CalibrationOut])
def calibrations(db: Session = Depends(get_db), _: Principal = Depends(requires("admin"))):
    return db.execute(select(CalibrationVersion).order_by(CalibrationVersion.created_at.desc())
                      ).scalars().all()


@router.get("/v1/trust")
def trust(_: Principal = Depends(requires("technician", "ophthalmologist"))):
    """Per-domain reliability. An unseen camera is reported as unverified, not silently trusted."""
    eng = get_engine()
    cal = eng.calibration
    return {"fitted": cal["fitted"], "source": cal["source"],
            "temperature": cal["temperature"], "referral_threshold": cal["referral_threshold"],
            "lesion_thresholds": cal["lesion_thresholds"],
            # the band that actually decides: a page explaining the trust layer that omitted it
            # would be describing machinery the server no longer uses
            "screening_band": cal.get("screening_band") or {},
            # the second reader vetoes automatic calls it does not agree with (torch/second_reader.py)
            "second_reader": cal.get("second_reader"),
            "conformal_q": cal.get("conformal_q") or {},
            "domains": cal.get("domains", {}),
            "validation_metrics": eng.metrics,
            "held_out": _held_out(cal)}


def _held_out(cal: dict) -> dict:
    """Test and Messidor-2 before and after temperature scaling, plus what the screening band
    actually did. val is where the checkpoint was chosen, so it is not the number to quote."""
    splits, measured = cal.get("splits") or {}, (cal.get("screening_band") or {}).get("measured") or {}
    two = (cal.get("second_reader") or {}).get("measured") or {}
    keep = ("n", "auc_referable", "sens_at_85spec", "qwk", "accuracy", "ece_referable")
    out = {}
    for name in ("test", "external_messidor2"):
        s = splits.get(name) or {}
        if not s:
            continue
        out[name] = {"raw": {k: (s.get("raw") or {}).get(k) for k in keep},
                     "calibrated": {k: (s.get("calibrated") or {}).get(k) for k in keep},
                     "band": measured.get(name) or {},
                     "two_readers": two.get(name) or {}}
    return out


@router.get("/v1/audit")
def audit_tail(limit: int = 50, db: Session = Depends(get_db),
               _: Principal = Depends(requires("admin"))):
    rows = db.execute(select(AuditLog).order_by(AuditLog.seq.desc()).limit(limit)).scalars().all()
    return [{"seq": r.seq, "at": r.created_at, "actor": r.actor, "action": r.action,
             "entity": f"{r.entity_type}:{r.entity_id}", "payload": r.payload,
             "hash": r.payload_hash[:12]} for r in rows]


@router.get("/v1/audit/verify")
def audit_verify(db: Session = Depends(get_db), _: Principal = Depends(requires("admin"))):
    """Re-walk the hash chain. A tampered or deleted row shows up here, with its sequence number."""
    return audit.verify_chain(db)


@router.get("/v1/stats")
def stats(db: Session = Depends(get_db), _: Principal = Depends(requires("technician", "ophthalmologist"))):
    count = lambda m: db.execute(select(func.count()).select_from(m)).scalar_one()
    by_grade = db.execute(select(Prediction.dr_grade, func.count()).group_by(Prediction.dr_grade)).all()
    by_decision = db.execute(select(Prediction.decision, func.count()).group_by(Prediction.decision)).all()
    runtimes = [r for (r,) in db.execute(select(InferenceJob.runtime_ms)
                                         .where(InferenceJob.runtime_ms.isnot(None))).all()]
    return {
        "encounters": count(Encounter), "images": count(Image), "predictions": count(Prediction),
        "referrals": count(Referral), "audit_entries": count(AuditLog),
        "jobs": {s: n for s, n in db.execute(select(InferenceJob.state, func.count())
                                             .group_by(InferenceJob.state)).all()},
        "grades": {int(g): n for g, n in by_grade},
        "decisions": {d: n for d, n in by_decision},
        "median_inference_ms": sorted(runtimes)[len(runtimes) // 2] if runtimes else None,
    }
