"""Application entry point: `uvicorn certus_api.main:app`.

One process runs the API and the inference worker. The model is loaded lazily on the first job so
the service starts instantly and does not hold GPU memory while idle.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import Base, engine, ensure_columns, SessionLocal
from .seed import seed_if_empty
from .routers import admin, catalog, clinical, queue
from .worker import Worker

DESCRIPTION = """
Explainable diabetic-retinopathy screening backend.

Every prediction carries the model version, the calibration version, the evidence the grade was
built from, and either a decision or an explicit refusal to guess.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.storage_dir, exist_ok=True)
    Base.metadata.create_all(engine)
    added = ensure_columns("prediction", {"trust_score": "INTEGER", "quality_factor": "REAL",
                                          "camera_factor": "REAL", "reliability": "REAL",
                                          "structures": "JSON"})
    added += ensure_columns("lesion_finding", {"macular_count": "INTEGER"})
    if added:
        print(f"added columns: {', '.join(added)}", flush=True)
    with SessionLocal() as db:                    # camps and cameras, on an empty catalogue only
        n = seed_if_empty(db)
        if n:
            print(f"seeded {n} demo camps with one camera each", flush=True)
    app.state.worker = None
    if os.environ.get("CERTUS_NO_WORKER") != "1":
        app.state.worker = Worker()
        app.state.worker.start()
    yield
    if app.state.worker:
        app.state.worker.stop()


app = FastAPI(title="Certus API", version="0.1.0", description=DESCRIPTION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

for r in (admin.router, catalog.router, clinical.router, queue.router):
    app.include_router(r)


@app.get("/", include_in_schema=False)
def root():
    return {"name": "Certus API", "docs": "/docs", "health": "/healthz"}
