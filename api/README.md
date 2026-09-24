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
| `CERTUS_RUNTIME` | `auto` | `torch` or `onnx`. `auto` uses torch on a GPU and onnxruntime otherwise. |
| `CERTUS_BATCH_TILES` | `1` | Tiles per encoder pass on CPU. Only changes peak memory, never the result. |
| `CERTUS_THREADS` | `0` | onnxruntime threads; `0` means one per physical core. |
| `CERTUS_ONNX_DIR` | `onnx/` beside the checkpoint | Where the exported graphs live. |
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

One eye is 3x3 tiles of 512 px plus a global view. Measured on one laptop (Ryzen, 2 threads), per eye:

| Runtime | Grade | Grad-CAM | Idle memory | Peak memory |
|---|---|---|---|---|
| torch, CUDA (RTX 3050) | ~0.3 s | | | |
| torch, CPU | ~4.8 s | ~6.7 s | ~680 MB | ~1.1 GB |
| onnxruntime, CPU | ~2.8 s | ~1.0 s | ~150 MB | ~430 MB |

The onnxruntime path runs the graphs `torch/export_onnx.py` wrote, with no torch in the process,
which is what lets the API run on a 1 GB server. `pip install -r requirements-server.txt` installs
only what it needs. `python check_onnx_parity.py` runs both runtimes on the demo photos and fails
unless grade, decision, quality, lesion counts and Grad-CAM agree. After retraining, re-export
with `python torch/export_onnx.py <best.pt>`: the API refuses to start on graphs exported from a
different checkpoint.

## Not built yet

Deliberate gaps, not oversights: no pagination on list endpoints, no rate limiting, API-key auth
rather than OIDC/ABDM, and no Alembic migrations (tables are created on startup).
