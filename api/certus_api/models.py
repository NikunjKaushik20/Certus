"""ORM tables (see SCHEMA.md). Ids are uuid4 hex so records can be created offline at a camp
and synced later without collisions."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def uid() -> str:
    return uuid.uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class Mixin:
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)

    def __init__(self, **kw):
        # assign the id at construction, not at INSERT: the audit trail references rows before flush
        kw.setdefault("id", uid())
        super().__init__(**kw)


# ---------------------------------------------------------------- who and where
class Site(Mixin, Base):
    __tablename__ = "site"
    name: Mapped[str] = mapped_column(String(120))
    district: Mapped[str] = mapped_column(String(80), default="")
    state: Mapped[str] = mapped_column(String(80), default="")
    kind: Mapped[str] = mapped_column(String(32), default="camp")      # camp | phc | hospital


class Device(Mixin, Base):
    __tablename__ = "device"
    site_id: Mapped[str] = mapped_column(ForeignKey("site.id"), index=True)
    make: Mapped[str] = mapped_column(String(80), default="")
    model: Mapped[str] = mapped_column(String(80), default="")
    serial: Mapped[str] = mapped_column(String(80), default="")
    # joins a capture to the calibration evidence for that camera family
    domain_key: Mapped[str] = mapped_column(String(64), index=True, default="unknown")


class AppUser(Mixin, Base):
    __tablename__ = "app_user"
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(32), index=True)          # technician | ophthalmologist | admin
    site_id: Mapped[str | None] = mapped_column(ForeignKey("site.id"), nullable=True)
    api_key: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)


class Patient(Mixin, Base):
    __tablename__ = "patient"
    pseudo_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    abha_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)   # never the raw id
    phone_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sex: Mapped[str] = mapped_column(String(8), default="unknown")
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diabetes_years: Mapped[float | None] = mapped_column(Float, nullable=True)


class Consent(Mixin, Base):
    __tablename__ = "consent"
    patient_id: Mapped[str] = mapped_column(ForeignKey("patient.id"), index=True)
    scope: Mapped[str] = mapped_column(String(64), default="screening")
    method: Mapped[str] = mapped_column(String(32), default="abdm")    # abdm | paper | verbal
    artifact_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------- what happened
class Encounter(Mixin, Base):
    __tablename__ = "encounter"
    patient_id: Mapped[str] = mapped_column(ForeignKey("patient.id"), index=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("site.id"), index=True)
    operator_id: Mapped[str | None] = mapped_column(ForeignKey("app_user.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="open", index=True)   # open|analysed|closed
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    images: Mapped[list["Image"]] = relationship(back_populates="encounter")


class Image(Mixin, Base):
    __tablename__ = "image"
    __table_args__ = (UniqueConstraint("encounter_id", "sha256", name="uq_image_per_encounter"),)
    encounter_id: Mapped[str] = mapped_column(ForeignKey("encounter.id"), index=True)
    device_id: Mapped[str | None] = mapped_column(ForeignKey("device.id"), nullable=True)
    eye: Mapped[str] = mapped_column(String(8))                        # L | R
    field: Mapped[str] = mapped_column(String(24), default="macula")
    sha256: Mapped[str] = mapped_column(String(64), index=True)        # content address; re-upload is idempotent
    storage_uri: Mapped[str] = mapped_column(String(256))              # original bytes as received
    proc_uri: Mapped[str | None] = mapped_column(String(256), nullable=True)   # FOV-normalised canvas
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    canvas: Mapped[int] = mapped_column(Integer, default=0)
    fov_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    encounter: Mapped[Encounter] = relationship(back_populates="images")


class InferenceJob(Mixin, Base):
    __tablename__ = "inference_job"
    image_id: Mapped[str] = mapped_column(ForeignKey("image.id"), index=True)
    state: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|done|failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    runtime_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    executor: Mapped[str | None] = mapped_column(String(16), nullable=True)       # cuda | cpu


# ---------------------------------------------------------------- what the model said
class ModelVersion(Mixin, Base):
    __tablename__ = "model_version"
    name: Mapped[str] = mapped_column(String(120), unique=True)
    checkpoint_uri: Mapped[str] = mapped_column(String(256))
    onnx_dir: Mapped[str | None] = mapped_column(String(256), nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_spec: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class CalibrationVersion(Mixin, Base):
    __tablename__ = "calibration_version"
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_version.id"), index=True)
    temperature: Mapped[float] = mapped_column(Float, default=1.0)
    referral_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    lesion_thresholds: Mapped[dict] = mapped_column(JSON, default=dict)
    conformal_q: Mapped[dict] = mapped_column(JSON, default=dict)      # alpha -> {class -> q}
    fitted_on: Mapped[str] = mapped_column(String(64), default="")
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class TrustDomain(Mixin, Base):
    __tablename__ = "trust_domain"
    calibration_version_id: Mapped[str] = mapped_column(ForeignKey("calibration_version.id"), index=True)
    domain_key: Mapped[str] = mapped_column(String(64), index=True)
    n: Mapped[int] = mapped_column(Integer, default=0)
    auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    ece: Mapped[float | None] = mapped_column(Float, nullable=True)
    abstain_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    miss_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="unverified")   # trusted|caution|unverified


class QualityAssessment(Mixin, Base):
    __tablename__ = "quality_assessment"
    image_id: Mapped[str] = mapped_column(ForeignKey("image.id"), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_version.id"))
    label: Mapped[str] = mapped_column(String(16))                     # good | usable | reject
    probs: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_action: Mapped[str] = mapped_column(String(24))               # graded | recapture_requested


class Prediction(Mixin, Base):
    __tablename__ = "prediction"
    image_id: Mapped[str] = mapped_column(ForeignKey("image.id"), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_version.id"), index=True)
    calibration_version_id: Mapped[str | None] = mapped_column(ForeignKey("calibration_version.id"),
                                                               nullable=True)
    dr_grade: Mapped[int] = mapped_column(Integer, index=True)
    ordinal_probs: Mapped[list] = mapped_column(JSON, default=list)    # P(>0), P(>1), P(>2), P(>3)
    p_referable_raw: Mapped[float] = mapped_column(Float)
    p_referable: Mapped[float] = mapped_column(Float, index=True)      # after temperature scaling
    decision: Mapped[str] = mapped_column(String(24), index=True)      # referable|non-referable|refer-to-human
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    abstain_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    threshold_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    conformal_alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Trust: how far the score sits outside the abstention band, discounted by the conditions the
    # photograph was taken in. Nullable because rows written before the trust layer existed have
    # no value for it and inventing one would be worse than showing nothing.
    trust_score: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)   # 0-100
    quality_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    camera_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    reliability: Mapped[float | None] = mapped_column(Float, nullable=True)
    structures: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # disc + fovea geometry
    evidence: Mapped[list] = mapped_column(JSON, default=list)         # 12-d evidence vector
    attention: Mapped[list] = mapped_column(JSON, default=list)        # per-tile attention weights
    runtime_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    executor: Mapped[str | None] = mapped_column(String(16), nullable=True)
    findings: Mapped[list["LesionFinding"]] = relationship(back_populates="prediction")


class LesionFinding(Mixin, Base):
    __tablename__ = "lesion_finding"
    prediction_id: Mapped[str] = mapped_column(ForeignKey("prediction.id"), index=True)
    lesion_type: Mapped[str] = mapped_column(String(8))                # MA | HE | EX | SE
    threshold: Mapped[float] = mapped_column(Float)
    lesion_count: Mapped[int] = mapped_column(Integer, default=0)
    # How many of those sit within one disc diameter of the estimated fovea. Central vision lives
    # there, so this is the count that changes urgency rather than just severity.
    macular_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pixel_area: Mapped[int] = mapped_column(Integer, default=0)
    area_frac: Mapped[float] = mapped_column(Float, default=0.0)
    peak_prob: Mapped[float] = mapped_column(Float, default=0.0)
    quadrant_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    overlay_uri: Mapped[str | None] = mapped_column(String(256), nullable=True)
    prediction: Mapped[Prediction] = relationship(back_populates="findings")


# ---------------------------------------------------------------- what humans did
class Referral(Mixin, Base):
    __tablename__ = "referral"
    encounter_id: Mapped[str] = mapped_column(ForeignKey("encounter.id"), index=True, unique=True)
    priority: Mapped[str] = mapped_column(String(16), default="routine", index=True)  # urgent | routine
    state: Mapped[str] = mapped_column(String(16), default="pending", index=True)     # pending|assigned|closed
    reason: Mapped[str] = mapped_column(String(64), default="")        # referable | abstention | ungradable
    worst_grade: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assigned_to: Mapped[str | None] = mapped_column(ForeignKey("app_user.id"), nullable=True)
    sla_due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Review(Mixin, Base):
    __tablename__ = "review"
    referral_id: Mapped[str] = mapped_column(ForeignKey("referral.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("app_user.id"), index=True)
    agreed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    final_grade: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    seconds_spent: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditLog(Base):
    """Append-only hash chain: each row commits to the previous one, so a silent edit is detectable."""
    __tablename__ = "audit_log"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(32), default=uid, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    actor: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    entity_id: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    payload_hash: Mapped[str] = mapped_column(String(64))
    prev_hash: Mapped[str] = mapped_column(String(64))
