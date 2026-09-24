# Certus backend

Screening API for explainable diabetic-retinopathy grading. One FastAPI process serves the HTTP
API and drains the inference queue; SQLite holds the records; photos are content-addressed on disk.
No Redis, no Celery, no object store and no GPU are required to run it.

## Run

```bash
cd api
python -m uvicorn certus_api.main:app --host 127.0.0.1 --port 8000
```

Interactive docs at <http://127.0.0.1:8000/docs>. Then, in another shell:

```bash
python smoke_test.py                     # full flow end to end, exits non-zero on any failure
```

### Settings (environment, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `CERTUS_DEVICE` | `auto` | `auto` uses CUDA when present, else CPU. `cpu` forces CPU. |
| `CERTUS_CHECKPOINT` | newest `best.pt` | Which trained model to serve. |
| `CERTUS_TRUST_JSON` | `trust.json` beside the checkpoint | Calibration. Absent = uncalibrated, and the API says so. |
| `CERTUS_DB_URL` | `sqlite:///api/certus.db` | Any SQLAlchemy URL; Postgres needs no code change. |
| `CERTUS_STORAGE_DIR` | `api/storage` | Blob root. |
| `CERTUS_SLA_HOURS` | `48` | Referral turnaround target. |
| `CERTUS_NO_WORKER` | unset | `1` disables the in-process worker (run workers separately). |

Demo keys, sent as `X-API-Key`: `demo-tech` (technician), `demo-doc` (ophthalmologist),
`demo-admin` (admin). Admin passes every role check; the others are enforced per route.

## The flow

```
POST /v1/patients            register (pseudonymous; ABHA and phone are hashed)
POST /v1/consents            ABDM consent  ->  screening is refused without it
POST /v1/encounters          open a visit
POST /v1/encounters/{id}/images   upload a fundus photo (eye=L|R)  -> queues a job
POST /v1/encounters/{id}/analyse  grade now and wait (demo convenience)
GET  /v1/encounters/{id}/report   grades, evidence, quality, referral, trust
GET  /v1/referrals                queue: urgent first, then nearest SLA deadline
GET  /v1/referrals/{id}/context   everything a reviewer needs in one call
POST /v1/referrals/{id}/review    verdict; closes the referral and the visit
GET  /v1/audit/verify             re-walk the hash chain
```

## What each prediction carries

Not just a grade: `dr_grade`, the four ordinal probabilities, the calibrated and raw referable
probability, the decision (`referable` / `non-referable` / `refer-to-human` with a reason), the
threshold and conformal level used, the 12-number evidence vector, per-tile attention, and per
lesion type a count, area, peak probability, per-quadrant counts and a mask PNG.

Every prediction names the `model_version` and `calibration_version` that produced it, so
re-calibrating later never silently rewrites an existing record.

## Decisions worth knowing

- **Thresholds are data, not code.** A single 0.5 cut-off is wrong for lesions: microaneurysms
  peak near 0.02 while hard exudates sit near 0.78. Values come from `trust.json`; without it the
  API runs on documented defaults and reports `calibration_fitted: false`.
- **Jobs are claimed atomically.** The worker and the synchronous `/analyse` endpoint both drain the
  queue; a conditional update stops the same eye being graded twice.
- **Identical bytes upload once.** Images are keyed by sha256 within an encounter.
- **Consent gates screening**, and revocation is recorded rather than deleted.
- **The audit log is append-only and hash-chained**, so `GET /v1/audit/verify` names the first
  sequence number that fails to verify.
- **Ungradable and abstained cases still reach a human.** Dropping either is how screening
  programmes lose patients.

## Performance

One eye is 3x3 tiles of 512 px plus a global view. Measured here: ~3.5-4 s per eye on 8 CPU cores,
~0.3 s on the RTX 3050. Both eyes of a patient therefore take under 10 s on CPU, which is well
inside a camp workflow; CUDA is used automatically when free.

## Not built yet

Deliberate gaps, not oversights: no pagination on list endpoints, no rate limiting, API-key auth
rather than OIDC/ABDM, no Alembic migrations (tables are created on startup), and the ONNX path is
exported but the API serves the PyTorch checkpoint.
