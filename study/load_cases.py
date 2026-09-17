"""Load the reader-study cases into a running Certus API, one visit per case.

Each case becomes a patient (pseudo id = case id, so the reviewer never sees the source dataset),
a consent, a visit, and one uploaded photograph, graded through the normal worker. Every case in
cases.csv was chosen because the band routes it to review, so each one lands in the ophthalmologist
queue with its Grad-CAM precomputed. Writes study/loaded.csv mapping case -> encounter.

Usage (API running on :8000):  python study/load_cases.py [base_url]
"""
import json
import os
import sys
import time
import urllib.request
import uuid

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
KEY = os.environ.get("CERTUS_TECH_KEY", "demo-tech")


def call(method, path, body=None, form=None):
    headers = {"X-API-Key": KEY}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if form is not None:
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in form.items():
            if isinstance(v, tuple):
                name, blob = v
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                             f"Content-Type: image/jpeg\r\n\r\n".encode() + blob + b"\r\n")
            else:
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        data = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read() or b"null")


def main():
    C = pd.read_csv(os.path.join(ROOT, "study", "cases.csv"))
    site = call("GET", "/v1/sites")[0]["id"]
    rows = []
    for _, c in C.iterrows():
        pat = call("POST", "/v1/patients", {"pseudo_id": f"STUDY-{c.case}"})
        call("POST", "/v1/consents", {"patient_id": pat["id"], "scope": "screening", "method": "study"})
        enc = call("POST", "/v1/encounters", {"patient_id": pat["id"], "site_id": site})
        blob = open(os.path.join(ROOT, "study", "images", f"{c.case}.jpg"), "rb").read()
        call("POST", f"/v1/encounters/{enc['id']}/images", form={"eye": "R", "field": "macula",
                                                                  "file": (f"{c.case}.jpg", blob)})
        t0 = time.time()
        call("POST", f"/v1/encounters/{enc['id']}/analyse")
        rows.append({"case": c.case, "patient_id": pat["id"], "encounter_id": enc["id"]})
        print(f"{c.case}: graded in {time.time() - t0:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "study", "loaded.csv"), index=False)


if __name__ == "__main__":
    main()
