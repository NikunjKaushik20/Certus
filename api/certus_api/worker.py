"""Background inference worker.

A thread drains `inference_job` rows. Deliberately no Redis/Celery: the demo must run from one
`uvicorn` command on a laptop or a small VM. Swapping this for a real broker means replacing
`Worker.run` only, because the queue state lives in the database, not in memory.
"""
import logging
import threading
import time
import traceback
from datetime import timedelta

import cv2
import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from . import audit, storage
from .config import settings
from .db import SessionLocal
from .inference import GRADE_LOCK, get_engine, retina
from .models import (Device, CalibrationVersion, Encounter, Image, InferenceJob, LesionFinding,
                     ModelVersion, Prediction, QualityAssessment, Referral, now)

log = logging.getLogger("certus.worker")

POLL_SECONDS = 0.5
MAX_ATTEMPTS = 3


# ---------------------------------------------------------------- version bookkeeping
def ensure_versions(db: Session) -> tuple[ModelVersion, CalibrationVersion]:
    """Register the loaded checkpoint and its calibration, so predictions can cite them."""
    eng = get_engine()
    name = f"{eng.checkpoint}@{eng.step}"
    mv = db.execute(select(ModelVersion).where(ModelVersion.name == name)).scalar_one_or_none()
    if mv is None:
        mv = ModelVersion(name=name, checkpoint_uri=eng.checkpoint, step=eng.step,
                          metrics=eng.metrics, active=True,
                          input_spec={"canvas": eng.cfg.canvas, "tile": eng.cfg.tile,
                                      "grid": eng.cfg.grid, "layout": "3x3 tiles + global view"})
        db.add(mv)
        db.flush()
        audit.record(db, "system", "model_version.registered", "model_version", mv.id,
                     {"checkpoint": eng.checkpoint, "step": eng.step})

    cal = eng.calibration
    cv = db.execute(select(CalibrationVersion)
                    .where(CalibrationVersion.model_version_id == mv.id)
                    .where(CalibrationVersion.fitted_on == cal["source"])).scalar_one_or_none()
    if cv is None:
        cv = CalibrationVersion(model_version_id=mv.id, temperature=cal["temperature"],
                                referral_threshold=cal["referral_threshold"],
                                lesion_thresholds=cal["lesion_thresholds"],
                                conformal_q=cal.get("conformal_q") or {},
                                fitted_on=cal["source"], active=True,
                                metrics={"fitted": cal["fitted"]})
        db.add(cv)
        db.flush()
        audit.record(db, "system", "calibration_version.registered", "calibration_version", cv.id,
                     {"fitted": cal["fitted"], "source": cal["source"]})
    db.commit()
    return mv, cv


# ---------------------------------------------------------------- one job
def claim(db: Session, job_id: str) -> bool:
    """Take ownership of a queued job atomically.

    The worker thread and the synchronous /analyse endpoint both drain the queue, so without a
    conditional update the same image gets graded twice and stored twice.
    """
    n = db.execute(update(InferenceJob).where(InferenceJob.id == job_id)
                   .where(InferenceJob.state == "queued")
                   .values(state="running", started_at=now(),
                           attempts=InferenceJob.attempts + 1)).rowcount
    db.commit()
    return bool(n)


def process_job(db: Session, job: InferenceJob) -> None:
    if not claim(db, job.id):
        return                                            # someone else is already grading it
    eng = get_engine()
    mv, cv = ensure_versions(db)
    db.refresh(job)
    img = db.get(Image, job.image_id)

    t0 = time.perf_counter()
    pre = eng.preprocess(storage.read(img.storage_uri))
    if not pre["ok"]:
        img.fov_ok = False
        db.add(QualityAssessment(image_id=img.id, model_version_id=mv.id, label="reject",
                                 probs={"reason": pre.get("error", "preprocess failed")},
                                 gate_action="recapture_requested"))
        audit.record(db, "system", "image.rejected", "image", img.id, {"reason": pre.get("error")})
        _finish(db, job, t0, eng.device)
        _settle_encounter(db, img.encounter_id)
        return

    sha, rel = storage.put(pre["jpeg"], ".jpg")
    img.proc_uri, img.width, img.height = rel, pre["width"], pre["height"]
    img.canvas, img.fov_ok = pre["canvas_size"], pre["fov_ok"]

    res = eng.analyse(pre["canvas"])
    q = res["quality"]
    gate = "recapture_requested" if q["label"] == "reject" else "graded"
    q_probs = dict(q["probs"])
    if "adequacy" in q:
        q_probs["adequacy"] = q["adequacy"]
    if "recapture_advice" in q:
        q_probs["recapture_advice"] = q["recapture_advice"]
    db.add(QualityAssessment(image_id=img.id, model_version_id=mv.id, label=q["label"],
                             probs=q_probs, gate_action=gate))

    if gate == "graded":
        # The camera is part of the decision now, not just a label on the report: an eye from a
        # domain with no fitted calibration is trusted less than the same eye from a fitted one.
        dev = db.get(Device, img.device_id) if img.device_id else None
        second = res.get("second_reader")
        d = eng.decide(res["p_referable"], quality_probs=q["probs"],
                       domain_key=dev.domain_key if dev else None,
                       second_logit=second["logit"] if second else None)
        if second:
            res.setdefault("structures", {})["second_reader"] = second
        # Before the first write: once the prediction is flushed this session holds the database
        # write lock, and on a small CPU server Grad-CAM would keep every other writer waiting.
        if d["decision"] != "non-referable":
            res["structures"] = {**(res.get("structures") or {}),
                                 **_precompute_gradcam(eng, pre["canvas"], res, img.id)}
        pred = Prediction(image_id=img.id, model_version_id=mv.id, calibration_version_id=cv.id,
                          dr_grade=res["dr_grade"], ordinal_probs=res["ordinal_probs"],
                          p_referable=res["p_referable"], p_referable_raw=res["p_referable_raw"],
                          evidence=res["evidence"], attention=res["attention"],
                          runtime_ms=res["runtime_ms"], executor=res["executor"],
                          structures=res.get("structures"), **d)
        db.add(pred)
        db.flush()
        for f in res["findings"]:
            mask = f.pop("mask", None)
            subpixel = f.pop("subpixel_coords", None)
            subtypes = f.pop("subtypes", None)
            if subtypes and "quadrant_counts" in f:
                f["quadrant_counts"]["subtypes"] = subtypes
            if subpixel and "quadrant_counts" in f:
                f["quadrant_counts"]["subpixel_sample"] = subpixel[:10]
            uri = None
            if mask is not None:
                ok, png = cv2.imencode(".png", (mask * 255).astype(np.uint8))
                if ok:
                    _, uri = storage.put(png.tobytes(), ".png")
            db.add(LesionFinding(prediction_id=pred.id, overlay_uri=uri, **f))
        audit.record(db, "system", "prediction.created", "prediction", pred.id,
                     {"image_id": img.id, "grade": pred.dr_grade, "decision": pred.decision,
                      "p_referable": round(pred.p_referable, 6), "model_version": mv.id,
                      "calibration_version": cv.id})

    else:
        audit.record(db, "system", "image.ungradable", "image", img.id, {"quality": q["label"]})

    _finish(db, job, t0, res["executor"])
    _settle_encounter(db, img.encounter_id)


def _precompute_gradcam(eng, canvas, res: dict, image_id: str) -> dict:
    """Render the referable Grad-CAM now, for every eye a person will review.

    On CPU Grad-CAM takes 1-2 s on onnxruntime and 15-30 s on torch. Computed on demand, that wait lands inside the 30-second
    review. Here it runs in the background worker instead, only for eyes routed to an
    ophthalmologist, and the endpoint serves the stored JPEG. Auto-cleared eyes still get
    theirs on demand. A failure here must not lose the prediction, so it is logged and skipped.
    Returns the keys to merge into the prediction's structures (empty on failure).
    """
    try:
        if (res.get("structures") or {}).get("enhancement", {}).get("regraded"):
            canvas, _ = retina.adaptive_enhance(canvas, quality_label="usable")  # the input that was graded
        t0 = time.perf_counter()
        jpeg = eng.gradcam(canvas, target="referable")
        _, uri = storage.put(jpeg, ".jpg")
        return {"gradcam_uri": uri, "gradcam_ms": int((time.perf_counter() - t0) * 1000)}
    except Exception as e:                                # noqa: BLE001
        log.warning("grad-cam precompute failed for image %s: %s", image_id, e)
        return {}


def _finish(db: Session, job: InferenceJob, t0: float, executor: str) -> None:
    job.state, job.finished_at = "done", now()
    job.runtime_ms, job.executor = int((time.perf_counter() - t0) * 1000), executor
    db.commit()


# ---------------------------------------------------------------- encounter level
def _settle_encounter(db: Session, encounter_id: str) -> None:
    """Once every image of a visit is processed, decide the patient-level outcome.

    A patient is referred at the severity of their worst eye; an abstention or an ungradable photo
    also reaches a human, because silently dropping either is how screening programmes lose people.
    """
    enc = db.get(Encounter, encounter_id)
    images = db.execute(select(Image).where(Image.encounter_id == encounter_id)).scalars().all()
    pending = db.execute(select(InferenceJob).where(InferenceJob.image_id.in_([i.id for i in images]))
                         .where(InferenceJob.state.in_(("queued", "running")))).scalars().all()
    if pending:
        return

    preds = db.execute(select(Prediction).where(Prediction.image_id.in_([i.id for i in images]))
                       ).scalars().all()
    quals = db.execute(select(QualityAssessment).where(QualityAssessment.image_id.in_([i.id for i in images]))
                       ).scalars().all()
    enc.status = "analysed"

    ungradable = [q for q in quals if q.gate_action == "recapture_requested"]
    worst = max((p.dr_grade for p in preds), default=None)
    referable = [p for p in preds if p.decision == "referable"]
    abstained = [p for p in preds if p.abstained]

    reason = None
    if referable:
        reason = "referable"
    elif abstained:
        reason = "abstention"
    elif ungradable:
        reason = "ungradable"
    if reason is None:
        db.commit()
        return

    urgent = {int(g) for g in settings.urgent_grades.split(",") if g.strip().isdigit()}
    ref = db.execute(select(Referral).where(Referral.encounter_id == encounter_id)).scalar_one_or_none()
    if ref is None:
        ref = Referral(encounter_id=encounter_id)
        db.add(ref)
    ref.reason = reason
    ref.worst_grade = worst
    ref.priority = "urgent" if (worst in urgent) else "routine"
    ref.sla_due_at = now() + timedelta(hours=settings.sla_hours)
    db.flush()
    audit.record(db, "system", "referral.created", "referral", ref.id,
                 {"encounter_id": encounter_id, "reason": reason, "priority": ref.priority,
                  "worst_grade": worst})
    db.commit()


# ---------------------------------------------------------------- queue plumbing
def enqueue(db: Session, image_id: str) -> InferenceJob:
    job = InferenceJob(image_id=image_id)
    db.add(job)
    db.flush()
    audit.record(db, "system", "job.queued", "inference_job", job.id, {"image_id": image_id})
    db.commit()
    return job


def run_pending(db: Session, limit: int = 8) -> int:
    jobs = db.execute(select(InferenceJob).where(InferenceJob.state == "queued")
                      .order_by(InferenceJob.created_at).limit(limit)).scalars().all()
    for job in jobs:
        _guarded(db, job)
    return len(jobs)


def _guarded(db: Session, job: InferenceJob) -> None:
    try:
        # One eye at a time, whichever thread asked: on a 1 GB server two concurrent eyes push the
        # process into swap and each takes ten times as long.
        with GRADE_LOCK:
            process_job(db, job)
    except Exception:                                     # a bad photo must not kill the queue
        db.rollback()
        job = db.get(InferenceJob, job.id)
        job.error = traceback.format_exc()[-2000:]
        job.state = "failed" if job.attempts >= MAX_ATTEMPTS else "queued"
        job.finished_at = now() if job.state == "failed" else None
        db.commit()


class Worker(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__(name="certus-inference")
        self._stop = threading.Event()

    def run(self) -> None:
        db = SessionLocal()
        try:
            while not self._stop.is_set():
                if run_pending(db) == 0:
                    time.sleep(POLL_SECONDS)
        finally:
            db.close()

    def stop(self) -> None:
        self._stop.set()
