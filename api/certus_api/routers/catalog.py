"""Registry endpoints: sites, cameras, staff, patients, consent."""
import hashlib

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit
from ..db import get_db
from ..models import AppUser, Consent, Device, Patient, Site
from ..schemas import (ConsentIn, DeviceIn, DeviceOut, PatientIn, PatientOut, SiteIn, SiteOut)
from ..security import Principal, requires

router = APIRouter(prefix="/v1", tags=["registry"])


def _hash(value: str | None) -> str | None:
    """Identifiers are stored as digests: the demo never holds a raw ABHA number or phone."""
    return hashlib.sha256(value.encode()).hexdigest() if value else None


@router.post("/sites", response_model=SiteOut)
def create_site(body: SiteIn, db: Session = Depends(get_db), p: Principal = Depends(requires("admin"))):
    site = Site(**body.model_dump())
    db.add(site)
    audit.record(db, p.actor, "site.created", "site", site.id, body.model_dump())
    db.commit()
    return site


@router.get("/sites", response_model=list[SiteOut])
def list_sites(db: Session = Depends(get_db), _: Principal = Depends(requires("technician", "ophthalmologist"))):
    return db.execute(select(Site).order_by(Site.created_at)).scalars().all()


@router.post("/devices", response_model=DeviceOut)
def create_device(body: DeviceIn, db: Session = Depends(get_db), p: Principal = Depends(requires("admin"))):
    if db.get(Site, body.site_id) is None:
        raise HTTPException(404, "site not found")
    dev = Device(**body.model_dump())
    db.add(dev)
    audit.record(db, p.actor, "device.created", "device", dev.id, body.model_dump())
    db.commit()
    return dev


@router.get("/devices", response_model=list[DeviceOut])
def list_devices(db: Session = Depends(get_db), _: Principal = Depends(requires("technician", "ophthalmologist"))):
    return db.execute(select(Device).order_by(Device.created_at)).scalars().all()


@router.post("/users")
def create_user(name: str, role: str, site_id: str | None = None, db: Session = Depends(get_db),
                p: Principal = Depends(requires("admin"))):
    if role not in ("technician", "ophthalmologist", "admin"):
        raise HTTPException(422, "role must be technician, ophthalmologist or admin")
    user = AppUser(name=name, role=role, site_id=site_id)
    db.add(user)
    audit.record(db, p.actor, "user.created", "app_user", user.id, {"name": name, "role": role})
    db.commit()
    return {"id": user.id, "name": user.name, "role": user.role, "site_id": user.site_id}


@router.get("/users")
def list_users(role: str | None = None, db: Session = Depends(get_db),
               # Review needs this to name a reviewer, so an ophthalmologist must read it, and
               # admin passes every check. A technician has no use for the staff list.
               _: Principal = Depends(requires("ophthalmologist"))):
    q = select(AppUser)
    if role:
        q = q.where(AppUser.role == role)
    return [{"id": u.id, "name": u.name, "role": u.role, "site_id": u.site_id}
            for u in db.execute(q.order_by(AppUser.created_at)).scalars()]


@router.post("/patients", response_model=PatientOut)
def create_patient(body: PatientIn, db: Session = Depends(get_db),
                   p: Principal = Depends(requires("technician"))):
    existing = db.execute(select(Patient).where(Patient.pseudo_id == body.pseudo_id)).scalar_one_or_none()
    if existing:
        return existing                                   # re-registering the same person is idempotent
    patient = Patient(pseudo_id=body.pseudo_id, abha_hash=_hash(body.abha), phone_hash=_hash(body.phone),
                      sex=body.sex, birth_year=body.birth_year, diabetes_years=body.diabetes_years)
    db.add(patient)
    audit.record(db, p.actor, "patient.created", "patient", patient.id, {"pseudo_id": body.pseudo_id})
    db.commit()
    return patient


@router.get("/patients", response_model=list[PatientOut])
def list_patients(db: Session = Depends(get_db),
                  _: Principal = Depends(requires("technician", "ophthalmologist"))):
    return db.execute(select(Patient).order_by(Patient.created_at.desc()).limit(200)).scalars().all()


@router.post("/consents")
def grant_consent(body: ConsentIn, db: Session = Depends(get_db),
                  p: Principal = Depends(requires("technician"))):
    if db.get(Patient, body.patient_id) is None:
        raise HTTPException(404, "patient not found")
    c = Consent(**body.model_dump())
    db.add(c)
    audit.record(db, p.actor, "consent.granted", "consent", c.id, body.model_dump())
    db.commit()
    return {"id": c.id, "patient_id": c.patient_id, "scope": c.scope, "method": c.method,
            "granted_at": c.granted_at}


@router.post("/consents/{consent_id}/revoke")
def revoke_consent(consent_id: str, db: Session = Depends(get_db),
                   p: Principal = Depends(requires("technician"))):
    from ..models import now
    c = db.get(Consent, consent_id)
    if c is None:
        raise HTTPException(404, "consent not found")
    c.revoked_at = now()
    audit.record(db, p.actor, "consent.revoked", "consent", c.id, {"patient_id": c.patient_id})
    db.commit()
    return {"id": c.id, "revoked_at": c.revoked_at}
