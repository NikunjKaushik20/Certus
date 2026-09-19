"""End-to-end check: register a camp, screen a patient, read the report, review the referral.

Run against a live server:  python smoke_test.py [base_url] [image_path]
Exits non-zero on the first broken step, so it doubles as a CI gate.
"""
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
IMG = sys.argv[2] if len(sys.argv) > 2 else "D:/Certus/Data/processed/IDRiD/img/IDRiD_A_IDRiD_01.jpg"
IMG2 = sys.argv[3] if len(sys.argv) > 3 else "D:/Certus/Data/processed/IDRiD/img/IDRiD_A_IDRiD_02.jpg"
ADMIN = {"X-API-Key": "demo-admin"}
TECH = {"X-API-Key": "demo-tech"}
DOC = {"X-API-Key": "demo-doc"}


def call(method, path, headers, expect=200, **kw):
    r = requests.request(method, BASE + path, headers=headers, timeout=600, **kw)
    if r.status_code != expect:
        raise SystemExit(f"FAIL {method} {path} -> {r.status_code}: {r.text[:400]}")
    return r.json() if r.content else {}


def main():
    print(f"base {BASE}")
    print("health   ", call("GET", "/healthz", {}))

    # auth must actually be enforced, not decorative
    call("GET", "/v1/sites", {"X-API-Key": "nope"}, expect=401)
    call("POST", "/v1/sites", TECH, expect=403, json={"name": "x"})
    print("auth      401 without a key, 403 for the wrong role")

    site = call("POST", "/v1/sites", ADMIN, json={"name": "Sitapur camp", "district": "Sitapur",
                                               "state": "Uttar Pradesh", "kind": "camp"})
    dev = call("POST", "/v1/devices", ADMIN, json={"site_id": site["id"], "make": "Remidio",
                                               "model": "FOP NM-10", "domain_key": "remidio_fop"})
    doc = call("POST", "/v1/users", ADMIN, params={"name": "Dr Rao", "role": "ophthalmologist"})
    pat = call("POST", "/v1/patients", TECH, json={"pseudo_id": f"SIT-{int(time.time())}", "sex": "F",
                                                "birth_year": 1971, "diabetes_years": 9})

    # screening without consent must be refused
    call("POST", "/v1/encounters", TECH, expect=409,
         json={"patient_id": pat["id"], "site_id": site["id"]})
    call("POST", "/v1/consents", TECH, json={"patient_id": pat["id"], "method": "abdm"})
    print("consent   blocked before grant, allowed after")

    enc = call("POST", "/v1/encounters", TECH, json={"patient_id": pat["id"], "site_id": site["id"]})
    with open(IMG, "rb") as fh:
        blob = fh.read()
    uploaded = {}
    for eye, path in (("R", IMG), ("L", IMG2)):
        with open(path, "rb") as fh:
            uploaded[eye] = call("POST", f"/v1/encounters/{enc['id']}/images", TECH,
                                 files={"file": ("fundus.jpg", fh, "image/jpeg")},
                                 data={"eye": eye, "field": "macula", "device_id": dev["id"]})
    # same bytes again must not create a second record
    with open(IMG, "rb") as fh:
        dup = call("POST", f"/v1/encounters/{enc['id']}/images", TECH,
                   files={"file": ("fundus.jpg", fh, "image/jpeg")},
                   data={"eye": "R", "field": "macula", "device_id": dev["id"]})
    assert dup["id"] == uploaded["R"]["id"], "re-upload of identical bytes created a duplicate"
    print("upload    2 eyes stored, duplicate upload deduplicated")

    t0 = time.perf_counter()
    ran = call("POST", f"/v1/encounters/{enc['id']}/analyse", TECH)
    dt = time.perf_counter() - t0
    print(f"inference ran {ran['ran']} job(s) in {dt:.1f}s")

    rep = call("GET", f"/v1/encounters/{enc['id']}/report", DOC)
    for iid, pred in rep["predictions"].items():
        if not pred:
            print(f"  {iid[:8]}: no prediction (quality gate: {rep['quality'][iid]})")
            continue
        f = {x["lesion_type"]: x["lesion_count"] for x in pred["findings"]}
        print(f"  eye {iid[:8]}: grade {pred['dr_grade']} | p(referable) {pred['p_referable']:.3f}"
              f" | {pred['decision']} | lesions {f} | {pred['runtime_ms']}ms on {pred['executor']}")
    print(f"report    worst grade {rep['worst_grade']}, decision {rep['decision']}, "
          f"referral {'yes' if rep['referral'] else 'no'}")

    if rep["referral"]:
        rid = rep["referral"]["id"]
        ctx = call("GET", f"/v1/referrals/{rid}/context", DOC)
        print(f"queue     priority {ctx['referral']['priority']}, reason {ctx['referral']['reason']}, "
              f"{ctx['referral']['hours_remaining']}h left")
        call("POST", f"/v1/referrals/{rid}/assign", DOC, json={"reviewer_id": doc["id"]})
        call("POST", f"/v1/referrals/{rid}/review", DOC, params={"reviewer_id": doc["id"]},
             json={"agreed": True, "final_grade": rep["worst_grade"], "notes": "confirmed",
                   "seconds_spent": 42})
        print("review    assigned and closed;", call("GET", "/v1/referrals/stats", DOC))

    chain = call("GET", "/v1/audit/verify", ADMIN)
    print("audit     ", chain)
    if not chain["ok"]:
        raise SystemExit("FAIL audit chain broken")
    print("stats     ", call("GET", "/v1/stats", DOC))
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
