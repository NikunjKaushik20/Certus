"""Request and response shapes. Response models are deliberately explicit: an API that hands a
clinician a grade must also hand them the evidence, the calibration used and the uncertainty."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

ORM = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------- inputs
class SiteIn(BaseModel):
    name: str
    district: str = ""
    state: str = ""
    kind: str = "camp"


class DeviceIn(BaseModel):
    site_id: str
    make: str = ""
    model: str = ""
    serial: str = ""
    domain_key: str = "unknown"


class PatientIn(BaseModel):
    pseudo_id: str = Field(description="local screening id; never a raw ABHA number")
    abha: str | None = Field(default=None, description="hashed before storage")
    phone: str | None = Field(default=None, description="hashed before storage")
    sex: str = "unknown"
    birth_year: int | None = None
    diabetes_years: float | None = None


class ConsentIn(BaseModel):
    patient_id: str
    scope: str = "screening"
    method: str = "abdm"
    artifact_ref: str | None = None


class EncounterIn(BaseModel):
    patient_id: str
    site_id: str
    operator_id: str | None = None


class ExplanationRating(BaseModel):
    """How useful the explanation was, for the reader study (study/PROTOCOL.md)."""
    gradcam_useful: int | None = Field(default=None, ge=1, le=5)      # 1 = misleading, 5 = decisive
    lesions_useful: int | None = Field(default=None, ge=1, le=5)
    heatmap_on_lesions: str | None = Field(default=None, pattern="^(yes|partly|no)$")


class ReviewIn(BaseModel):
    agreed: bool | None = None
    final_grade: int | None = Field(default=None, ge=0, le=4)
    notes: str = ""
    seconds_spent: int | None = None
    explanation: ExplanationRating | None = None


class AssignIn(BaseModel):
    reviewer_id: str


# ---------------------------------------------------------------- outputs
class SiteOut(BaseModel):
    model_config = ORM
    id: str
    name: str
    district: str
    state: str
    kind: str


class DeviceOut(BaseModel):
    model_config = ORM
    id: str
    site_id: str
    make: str
    model: str
    serial: str
    domain_key: str


class PatientOut(BaseModel):
    model_config = ORM
    id: str
    pseudo_id: str
    sex: str
    birth_year: int | None
    created_at: datetime


class ImageOut(BaseModel):
    model_config = ORM
    id: str
    encounter_id: str
    eye: str
    field: str
    sha256: str
    fov_ok: bool
    canvas: int
    created_at: datetime


class JobOut(BaseModel):
    model_config = ORM
    id: str
    image_id: str
    state: str
    attempts: int
    error: str | None
    runtime_ms: int | None
    executor: str | None


class QualityOut(BaseModel):
    model_config = ORM
    label: str
    probs: dict
    gate_action: str
    adequacy: dict | None = None
    recapture_advice: list[str] | None = None


class FindingOut(BaseModel):
    model_config = ORM
    id: str                                    # the client needs this to fetch /v1/findings/{id}/mask
    lesion_type: str
    threshold: float
    lesion_count: int
    pixel_area: int
    area_frac: float
    peak_prob: float
    quadrant_counts: dict
    overlay_uri: str | None
    macular_count: int | None = None
    subpixel_coords: list[dict] | None = None
    subtypes: dict | None = None


class PredictionOut(BaseModel):
    model_config = ORM
    id: str
    image_id: str
    dr_grade: int
    ordinal_probs: list
    p_referable: float
    p_referable_raw: float
    decision: str
    abstained: bool
    abstain_reason: str | None
    threshold_used: float | None
    conformal_alpha: float | None
    trust_score: int | None = None          # 0-100: reliability x decisiveness
    quality_factor: float | None = None     # what the photograph contributes
    camera_factor: float | None = None      # what the camera calibration contributes
    reliability: float | None = None        # their product; below 0.70 forces human review
    structures: dict | None = None          # optic disc and estimated fovea, in canvas pixels
    evidence: list
    attention: list
    runtime_ms: int | None
    executor: str | None
    model_version_id: str
    calibration_version_id: str | None
    findings: list[FindingOut] = []


class EncounterOut(BaseModel):
    model_config = ORM
    id: str
    patient_id: str
    site_id: str
    status: str
    created_at: datetime


class EncounterReport(BaseModel):
    """What the review console renders: one patient visit, both eyes, evidence and next action."""
    encounter: EncounterOut
    images: list[ImageOut]
    quality: dict[str, QualityOut | None]
    predictions: dict[str, PredictionOut | None]
    worst_grade: int | None
    decision: str
    referral: "ReferralOut | None" = None
    trust: dict | None = None
    superseded: list[str] = []          # earlier photographs of an eye that was retaken


class ReferralOut(BaseModel):
    model_config = ORM
    id: str
    encounter_id: str
    priority: str
    state: str
    reason: str
    worst_grade: int | None
    assigned_to: str | None
    sla_due_at: datetime
    created_at: datetime
    closed_at: datetime | None


class ReviewOut(BaseModel):
    model_config = ORM
    id: str
    referral_id: str
    reviewer_id: str
    agreed: bool | None
    final_grade: int | None
    notes: str
    seconds_spent: int | None
    decided_at: datetime


class ModelVersionOut(BaseModel):
    model_config = ORM
    id: str
    name: str
    checkpoint_uri: str
    step: int | None
    metrics: dict
    input_spec: dict
    active: bool


class CalibrationOut(BaseModel):
    model_config = ORM
    id: str
    model_version_id: str
    temperature: float
    referral_threshold: float
    lesion_thresholds: dict
    conformal_q: dict
    fitted_on: str
    metrics: dict
    active: bool


EncounterReport.model_rebuild()
