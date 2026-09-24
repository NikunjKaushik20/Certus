"""The screening flow: open a visit, upload photos, get them graded, read the report."""
import os
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import audit, deident, storage, worker
from ..db import get_db
from ..inference import get_engine
from ..models import (Consent, Encounter, Image, InferenceJob, LesionFinding, Patient, Prediction,
                      QualityAssessment, Referral, Site)
from ..schemas import EncounterIn, EncounterOut, EncounterReport, ImageOut, JobOut, PredictionOut
from ..security import Principal, requires

ANALYSE_TIMEOUT_S = 180

router = APIRouter(prefix="/v1", tags=["screening"])


@router.post("/encounters", response_model=EncounterOut)
def open_encounter(body: EncounterIn, db: Session = Depends(get_db),
                   p: Principal = Depends(requires("technician"))):
    if db.get(Patient, body.patient_id) is None:
        raise HTTPException(404, "patient not found")
    if db.get(Site, body.site_id) is None:
        raise HTTPException(404, "site not found")
    consent = db.execute(select(Consent).where(Consent.patient_id == body.patient_id)
                         .where(Consent.revoked_at.is_(None))).scalars().first()
    if consent is None:
        raise HTTPException(409, "no active consent for this patient")   # screening without consent is a no
    enc = Encounter(**body.model_dump())
    db.add(enc)
    audit.record(db, p.actor, "encounter.opened", "encounter", enc.id, body.model_dump())
    db.commit()
    return enc


@router.post("/encounters/{encounter_id}/images", response_model=ImageOut)
async def upload_image(encounter_id: str, eye: str = Form(...), field: str = Form("macula"),
                       device_id: str | None = Form(None), file: UploadFile = File(...),
                       db: Session = Depends(get_db), p: Principal = Depends(requires("technician"))):
    enc = db.get(Encounter, encounter_id)
    if enc is None:
        raise HTTPException(404, "encounter not found")
    if eye not in ("L", "R"):
        raise HTTPException(422, "eye must be L or R")
    raw = await file.read()
    if not raw:
        raise HTTPException(422, "empty upload")
    if deident.is_dicom(raw):
        # A DICOM header carries patient name, id and birth date in the file itself. Removing them
        # correctly is PS3.15 Annex E, not a metadata strip, and it is implemented in MATLAB where
        # dicomanon does it properly. Refusing is the honest failure: storing one here would put
        # identifiers into content-addressed blobs that are never rewritten.
        raise HTTPException(415, "DICOM upload: de-identify with matlab/deidentifyDicom.m first "
                                 "(writes a *_clean.dcm), then convert to JPEG/PNG before uploading. "
                                 "Many cameras can export directly as JPEG; alternatively dcm2niix "
                                 "or MATLAB's dicomwrite can convert the cleaned file.")
    raw, scrub = deident.strip_metadata(raw)
    sha, rel = storage.put(raw, ".bin")

    dup = db.execute(select(Image).where(Image.encounter_id == encounter_id)
                     .where(Image.sha256 == sha)).scalar_one_or_none()
    if dup:
        return dup                                        # same bytes twice: one record, one job

    img = Image(encounter_id=encounter_id, device_id=device_id, eye=eye, field=field,
                sha256=sha, storage_uri=rel)
    db.add(img)
    db.flush()
    audit.record(db, p.actor, "image.uploaded", "image", img.id,
                 {"encounter_id": encounter_id, "eye": eye, "sha256": sha,
                  # What was removed is itself auditable: a programme that has to answer a privacy
                  # question later needs to show that identifiers were stripped, not assert it.
                  "deidentified": scrub["stripped"], "metadata_bytes_removed": scrub["bytes_removed"]})
    db.commit()
    return img                    # grading is an explicit act: see POST /encounters/{id}/analyse


def _current_images(db: Session, encounter_id: str):
    """The photographs that are this visit, newest per eye, plus the ids they replaced.

    A retake adds a row rather than replacing one, so an encounter can hold several photographs of
    the same eye. Reporting all of them made the summary disagree with the eye cards: the summary
    took the worst grade across every photograph while the cards showed the first, so a grade-3
    original sat behind a grade-0 retake and the visit read "referable" over two cards both
    reading "non-referable"."""
    history = db.execute(select(Image).where(Image.encounter_id == encounter_id)
                         .order_by(Image.created_at)).scalars().all()
    current = {}
    for img in history:                       # ordered oldest first, so the last write wins
        current[(img.eye, img.field)] = img
    live = sorted(current.values(), key=lambda i: i.created_at)
    live_ids = {i.id for i in live}
    return live, [i.id for i in history if i.id not in live_ids]


@router.get("/encounters/{encounter_id}/jobs", response_model=list[JobOut])
def encounter_jobs(encounter_id: str, db: Session = Depends(get_db),
                   _: Principal = Depends(requires("technician", "ophthalmologist"))):
    ids = [i.id for i in db.execute(select(Image).where(Image.encounter_id == encounter_id)).scalars()]
    return db.execute(select(InferenceJob).where(InferenceJob.image_id.in_(ids))).scalars().all()


@router.post("/encounters/{encounter_id}/analyse")
def analyse_now(encounter_id: str, db: Session = Depends(get_db),
                _: Principal = Depends(requires("technician"))):
    """Grade this visit now and return once it is fully settled, for a live demo where polling
    is awkward. Jobs already claimed by the background worker are waited for, not re-run."""
    images, _ = _current_images(db, encounter_id)
    ids = [i.id for i in images]
    if not ids:
        raise HTTPException(409, "no images uploaded for this encounter")

    # Uploading stores a photograph; it does not grade one. Queue whatever this visit still owes,
    # skipping any eye that was retaken, then run it.
    have = {j.image_id for j in db.execute(select(InferenceJob)
                                           .where(InferenceJob.image_id.in_(ids))).scalars()}
    for image_id in ids:
        if image_id not in have:
            worker.enqueue(db, image_id)

    ran = 0
    for job in db.execute(select(InferenceJob).where(InferenceJob.image_id.in_(ids))
                          .where(InferenceJob.state == "queued")).scalars().all():
        before = job.state
        worker._guarded(db, job)
        db.refresh(job)
        ran += int(before != job.state and job.state == "done")

    deadline = time.monotonic() + ANALYSE_TIMEOUT_S
    while time.monotonic() < deadline:
        busy = db.execute(select(func.count()).select_from(InferenceJob)
                          .where(InferenceJob.image_id.in_(ids))
                          .where(InferenceJob.state.in_(("queued", "running")))).scalar_one()
        if not busy:
            break
        db.commit()                                       # release the read snapshot, then look again
        time.sleep(0.25)
    else:
        raise HTTPException(504, "inference still running; poll /jobs instead")
    return {"ran": ran, "images": len(ids)}


@router.get("/encounters/{encounter_id}/report", response_model=EncounterReport)
def report(encounter_id: str, db: Session = Depends(get_db),
           _: Principal = Depends(requires("technician", "ophthalmologist"))):
    enc = db.get(Encounter, encounter_id)
    if enc is None:
        raise HTTPException(404, "encounter not found")
    images, superseded = _current_images(db, encounter_id)
    ids = [i.id for i in images]
    quals = {q.image_id: q for q in db.execute(select(QualityAssessment)
                                               .where(QualityAssessment.image_id.in_(ids))).scalars()}
    preds = {p.image_id: p for p in db.execute(select(Prediction)
                                               .where(Prediction.image_id.in_(ids))).scalars()}
    ref = db.execute(select(Referral).where(Referral.encounter_id == encounter_id)).scalar_one_or_none()
    grades = [p.dr_grade for p in preds.values()]
    decisions = {p.decision for p in preds.values()}
    decision = ("referable" if "referable" in decisions else
                "refer-to-human" if "refer-to-human" in decisions else
                "non-referable" if decisions else "pending")
    eng = get_engine()
    return EncounterReport(
        encounter=enc, images=images,
        quality={i: quals.get(i) for i in ids},
        predictions={i: preds.get(i) for i in ids},
        worst_grade=max(grades) if grades else None,
        decision=decision, referral=ref, superseded=superseded,
        trust={"calibration_fitted": eng.calibration["fitted"],
               "source": eng.calibration["source"],
               "referral_threshold": eng.calibration["referral_threshold"],
               "validation_metrics": eng.metrics},
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: str, db: Session = Depends(get_db),
                   _: Principal = Depends(requires("technician", "ophthalmologist"))):
    pred = db.get(Prediction, prediction_id)
    if pred is None:
        raise HTTPException(404, "prediction not found")
    return pred


@router.get("/images/{image_id}/file")
def image_file(image_id: str, kind: str = Query("processed", pattern="^(processed|original)$"),
               db: Session = Depends(get_db),
               _: Principal = Depends(requires("technician", "ophthalmologist"))):
    img = db.get(Image, image_id)
    if img is None:
        raise HTTPException(404, "image not found")
    rel = img.proc_uri if kind == "processed" else img.storage_uri
    if not rel:
        raise HTTPException(409, "not preprocessed yet")
    return FileResponse(storage.full_path(rel), media_type="image/jpeg")


@router.get("/findings/{finding_id}/mask")
def finding_mask(finding_id: str, db: Session = Depends(get_db),
                 _: Principal = Depends(requires("technician", "ophthalmologist"))):
    """The lesion mask as a PNG, for the frontend to tint and overlay on the processed image."""
    f = db.get(LesionFinding, finding_id)
    if f is None or not f.overlay_uri:
        raise HTTPException(404, "no mask for this finding")
    return FileResponse(storage.full_path(f.overlay_uri), media_type="image/png")


@router.get("/predictions/{prediction_id}/gradcam")
def prediction_gradcam(
        prediction_id: str,
        target: str = Query("referable", pattern="^(referable|any_dr|moderate|proliferative)$"),
        db: Session = Depends(get_db),
        _: Principal = Depends(requires("technician", "ophthalmologist"))):
    """Grad-CAM heatmap for this prediction, overlaid on the processed fundus image.

    Returns a JPEG (not a JSON payload) — the frontend can assign it directly as an <img> src
    or an overlay. The `target` query parameter selects which ordinal logit drives the map:
    referable (default, P(grade>=2)), any_dr, moderate, or proliferative.

    Grad-CAM runs with gradient tracking on the shared model, tile-by-tile to stay within
    GPU memory (see gradcam.py), and the result is stitched back into canvas coordinates so
    it aligns pixel-for-pixel with the lesion mask overlay.

    For every eye routed to a person the worker has already rendered the referable map
    (worker._precompute_gradcam), so the reviewer gets a stored file, not a 15-30 s CPU wait.
    Other targets, and auto-cleared eyes, are computed on demand.
    """
    import cv2
    import numpy as np

    pred = db.get(Prediction, prediction_id)
    if pred is None:
        raise HTTPException(404, "prediction not found")
    cached = (pred.structures or {}).get("gradcam_uri")
    if target == "referable" and cached and os.path.exists(storage.full_path(cached)):
        return FileResponse(storage.full_path(cached), media_type="image/jpeg")
    img = db.get(Image, pred.image_id)
    if img is None or not img.proc_uri:
        raise HTTPException(409, "processed image not available yet")

    path = storage.full_path(img.proc_uri)
    raw = open(path, "rb").read()
    canvas_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if canvas_bgr is None:
        raise HTTPException(409, "could not decode stored canvas")

    engine = get_engine()
    jpeg_bytes = engine.gradcam(canvas_bgr, target=target)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/encounters/{encounter_id}/export")
def export_encounter_report(encounter_id: str, db: Session = Depends(get_db),
                            _: Principal = Depends(requires("technician", "ophthalmologist"))):
    """Export printable publication-grade clinical screening report for the encounter."""
    enc = db.get(Encounter, encounter_id)
    if enc is None:
        raise HTTPException(404, "encounter not found")
    images, superseded = _current_images(db, encounter_id)
    ids = [i.id for i in images]
    quals = {q.image_id: q for q in db.execute(select(QualityAssessment)
                                               .where(QualityAssessment.image_id.in_(ids))).scalars()}
    preds = {p.image_id: p for p in db.execute(select(Prediction)
                                               .where(Prediction.image_id.in_(ids))).scalars()}
    ref = db.execute(select(Referral).where(Referral.encounter_id == encounter_id)).scalar_one_or_none()

    grades = [p.dr_grade for p in preds.values() if p]
    worst_g = max(grades) if grades else None
    GRADE_TEXT = ["No Retinopathy", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
    # Same wording as web/src/abstain.ts, so the printout and the console say the same thing.
    REASON_TEXT = {
        "ambiguous": "The score sits inside the screening band, too close to call either way.",
        "readers_disagree": "The two readers did not agree. Certus and the independent whole-image grader "
                            "must both make the same call before an eye is referred or cleared automatically.",
        "low_reliability": "The photograph or the camera is not reliable enough to believe any score.",
    }

    eye_html = ""
    for img in images:
        pred = preds.get(img.id)
        q = quals.get(img.id)
        eye_label = "Right Eye (OD)" if img.eye == "R" else "Left Eye (OS)"
        if pred:
            struc = pred.structures or {}
            ves = struc.get("vessels") or {}
            nv = struc.get("neovascularization") or {}
            adeq = struc.get("adequacy") or {}
            enh = struc.get("enhancement") or {}

            g_name = GRADE_TEXT[pred.dr_grade] if pred.dr_grade < len(GRADE_TEXT) else f"Grade {pred.dr_grade}"
            findings_rows = ""
            for f in pred.findings:
                quad = f.quadrant_counts or {}
                subtypes = quad.get("subtypes") or {}
                subtype_str = ""
                if subtypes:
                    subtype_str = f" ({subtypes.get('dot_blot', 0)} dot, {subtypes.get('flame_shaped', 0)} flame, {subtypes.get('preretinal_vitreous', 0)} pre)"
                findings_rows += f"""
                <tr>
                    <td><strong>{f.lesion_type}</strong></td>
                    <td>{f.lesion_count}{subtype_str}</td>
                    <td>{f.macular_count or 0}</td>
                    <td>{f.peak_prob:.3f}</td>
                    <td>SN:{quad.get('superior_nasal',0)} ST:{quad.get('superior_temporal',0)} IN:{quad.get('inferior_nasal',0)} IT:{quad.get('inferior_temporal',0)}</td>
                </tr>"""

            enh_note = "<span class='tag'>Enhanced: CLAHE + Bilateral Denoising</span>" if enh.get("applied") else "Standard"
            second = struc.get("second_reader") or {}
            second_html = (f"<div><strong>Second reader:</strong> p(Referable) {second['p_referable']:.4f} "
                           f"(whole-image grader, no lesion maps)</div>") if second else ""
            reason_html = (f"<div><strong>Why a person:</strong> "
                           f"{REASON_TEXT.get(pred.abstain_reason, pred.abstain_reason)}</div>") if pred.abstained else ""

            eye_html += f"""
            <div class="eye-section">
                <h3>{eye_label} — ICDR Level {pred.dr_grade}: {g_name}</h3>
                <div class="summary-grid">
                    <div><strong>Decision:</strong> <span class="badge {pred.decision}">{pred.decision.upper()}</span></div>
                    <div><strong>p(Referable):</strong> {pred.p_referable:.4f} (Raw: {pred.p_referable_raw:.4f})</div>
                    <div><strong>Trust Score:</strong> {pred.trust_score or 0}% (Reliability: {pred.reliability or 0:.2f})</div>
                    <div><strong>Quality:</strong> {q.label if q else 'N/A'} · {enh_note}</div>
                    {second_html}
                    {reason_html}
                </div>
                <div class="structures-info">
                    <div><strong>Optic Disc:</strong> {'Found (diam ' + str(round(struc.get('disc_diameter_px', 0))) + 'px)' if struc.get('disc_found') else 'Not located'}</div>
                    <div><strong>Fovea:</strong> {'Located' if struc.get('fovea_found') else 'Not located'} (Macular radius: {round(struc.get('macula_radius_px', 0))}px)</div>
                    <div><strong>Vessels:</strong> {ves.get('density_pct', 0)}% retinal area</div>
                    <div><strong>NV Assessment:</strong> <span class="{'bad' if nv.get('nvd_detected') or nv.get('nve_detected') else 'good'}">{nv.get('criteria', 'None')}</span> (Risk: {nv.get('nv_risk_score', 0)})</div>
                </div>
                <table class="findings-table">
                    <thead>
                        <tr><th>Lesion</th><th>Count</th><th>Macular</th><th>Peak Prob</th><th>Quadrant Distribution</th></tr>
                    </thead>
                    <tbody>
                        {findings_rows}
                    </tbody>
                </table>
            </div>"""

    # The recommendation follows the decision, not the grade. A grade-1 eye above the refer cut is
    # referred, and an eye the readers disagreed on is waiting for a person, whatever its grade.
    decisions = {p.decision for p in preds.values() if p}
    if "referable" in decisions:
        action_text = "URGENT REFERRAL: Immediate examination by retina specialist."
    elif "refer-to-human" in decisions:
        action_text = "OPHTHALMOLOGIST REVIEW: Certus did not decide this case on its own. A specialist grades it before any referral."
    else:
        action_text = "Routine rescreening recommended in 12 months."

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Certus Screening Report - Encounter {enc.id[:8]}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem; color: #222; }}
    .header {{ border-bottom: 2px solid #0052cc; padding-bottom: 0.8rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: flex-end; }}
    .header h1 {{ margin: 0; font-size: 1.6rem; color: #0052cc; }}
    .header .meta {{ font-size: 0.85rem; color: #555; text-align: right; }}
    .encounter-meta {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; background: #f4f6fa; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; font-size: 0.9rem; }}
    .eye-section {{ border: 1px solid #ddd; border-radius: 6px; padding: 1.2rem; margin-bottom: 1.5rem; }}
    .eye-section h3 {{ margin-top: 0; color: #333; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-bottom: 1rem; font-size: 0.9rem; }}
    .structures-info {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; background: #fafafa; padding: 0.6rem 0.8rem; border-radius: 4px; font-size: 0.85rem; margin-bottom: 1rem; }}
    .findings-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .findings-table th, .findings-table td {{ border: 1px solid #eee; padding: 0.45rem 0.6rem; text-align: left; }}
    .findings-table th {{ background: #f7f7f7; }}
    .badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: bold; font-size: 0.8rem; color: #fff; }}
    .badge.referable {{ background: #d32f2f; }}
    .badge.non-referable {{ background: #2e7d32; }}
    .badge.refer-to-human {{ background: #ed6c02; }}
    .tag {{ background: #e3f2fd; color: #0d47a1; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.75rem; }}
    .good {{ color: #2e7d32; }}
    .bad {{ color: #d32f2f; font-weight: bold; }}
    .action-box {{ border: 2px solid #0052cc; background: #f0f7ff; padding: 1rem; border-radius: 6px; margin-top: 1.5rem; }}
    .sign-box {{ margin-top: 2.5rem; display: flex; justify-content: space-between; font-size: 0.9rem; color: #444; }}
    .print-btn {{ background: #0052cc; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; font-size: 0.9rem; }}
    @media print {{ .print-btn {{ display: none; }} body {{ margin: 0; }} }}
</style>
</head>
<body>
<div class="header">
    <div>
        <h1>CERTUS CLINICAL SCREENING REPORT</h1>
        <div style="font-size: 0.85rem; color: #666; margin-top: 0.2rem;">SIH Problem Statement 26038 (MathWorks) · AI-Assisted Rural Screening Pipeline</div>
    </div>
    <div class="meta">
        <button class="print-btn" onclick="window.print()">Print Report / Save PDF</button>
        <div style="margin-top: 0.4rem;">Encounter ID: <code>{enc.id[:12]}</code><br>Date: {enc.created_at.strftime('%Y-%m-%d %H:%M UTC')}</div>
    </div>
</div>

<div class="encounter-meta">
    <div><strong>Patient ID:</strong> {enc.patient_id}</div>
    <div><strong>Site:</strong> {enc.site_id}</div>
    <div><strong>Worst Grade:</strong> {worst_g if worst_g is not None else 'N/A'} ({GRADE_TEXT[worst_g] if worst_g is not None and worst_g < len(GRADE_TEXT) else 'N/A'})</div>
    <div><strong>Overall Status:</strong> {enc.status.upper()}</div>
</div>

{eye_html}

<div class="action-box">
    <h4 style="margin: 0 0 0.4rem 0;">RECOMMENDED CLINICAL MANAGEMENT</h4>
    <p style="margin: 0; font-size: 0.95rem;">{action_text}</p>
</div>

<div class="sign-box">
    <div><strong>Primary Health Technician:</strong> Verified upload</div>
    <div><strong>Reviewing Ophthalmologist:</strong> Dr. ___________________________  Reg No: ____________</div>
    <div><strong>Signature & Date:</strong> ___________________________</div>
</div>
</body>
</html>"""
    return Response(content=html, media_type="text/html")

