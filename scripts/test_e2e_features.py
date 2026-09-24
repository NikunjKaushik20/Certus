"""End-to-end verification of all new clinical requirements for SIH 26038.
Tests:
1. Adequacy evaluation & adaptive enhancement
2. 6-channel structural extraction (disc, fovea, vessel density, NV detection)
3. Sub-pixel microaneurysm refinement & hemorrhage subclassification
4. Grad-CAM overlay generation
5. Exportable clinical screening report endpoint
"""
import os
import sys
import cv2
import numpy as np

# Add repo to sys.path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "api"))

from certus_api.inference import Engine
from scripts.retina import evaluate_adequacy, adaptive_enhance

def test_inference_and_structures():
    print("--- 1. Testing Engine Analysis & Structures ---")
    img_path = os.path.join(REPO, "demo_images", "grade3_severe_IDRiD_006.jpg")
    with open(img_path, "rb") as fh:
        raw_bytes = fh.read()

    engine = Engine()
    prep = engine.preprocess(raw_bytes)
    assert prep["ok"], "FOV check failed"
    canvas_bgr = prep["canvas"]
    fov_mask = prep["fov"]

    # Test adequacy
    adeq = evaluate_adequacy(canvas_bgr, fov_mask)
    print(f"Adequacy: focus={adeq['focus']['score']:.1f}, illum={adeq['illumination']['score']:.1f}, "
          f"fov={adeq['field_of_view']['score']:.1f}, adequate={adeq['overall_adequate']}")
    assert "focus" in adeq
    assert "recapture_advice" in adeq

    # Test adaptive enhance
    enh, meta = adaptive_enhance(canvas_bgr, "usable", force=True)
    print(f"Adaptive Enhancement: applied={meta['applied']}, methods={meta['methods']}")
    assert meta["applied"], "Adaptive enhancement failed to apply"

    # Run full analysis
    res = engine.analyse(canvas_bgr)
    sr = res.get("second_reader")
    dec = engine.decide(res["p_referable"], second_logit=sr["logit"] if sr else None)
    print(f"Decision: {dec['decision']}, p(referable): {res['p_referable']:.4f}")
    assert res["dr_grade"] == 3, f"Expected grade 3, got {res['dr_grade']}"

    # Verify structures
    struc = res["structures"]
    print("Structures found:")
    print(f"  - Optic Disc: found={struc['disc_found']}, diam={struc.get('disc_diameter_px', 0):.1f}px")
    print(f"  - Fovea: found={struc['fovea_found']}, macula_radius={struc.get('macula_radius_px', 0):.1f}px")
    print(f"  - Retinal Vessels: density={struc['vessels']['density_pct']}%, area={struc['vessels'].get('vessel_pixels', 0)}px")
    print(f"  - Neovascularization: NVD={struc['neovascularization']['nvd_detected']}, "
          f"NVE={struc['neovascularization']['nve_detected']}, "
          f"Risk={struc['neovascularization']['nv_risk_score']:.2f}, "
          f"Criteria='{struc['neovascularization']['criteria']}'")
    
    assert struc["disc_found"]
    assert struc["fovea_found"]
    assert struc["vessels"]["density_pct"] > 0
    assert "neovascularization" in struc

    # Verify sub-pixel MAs and Hemorrhage subclassification
    findings = {f["lesion_type"]: f for f in res["findings"]}
    assert "MA" in findings and "HE" in findings
    ma = findings["MA"]
    he = findings["HE"]
    print(f"  - Sub-pixel MAs: {len(ma['subpixel_coords'])} coordinates refined (e.g. {ma['subpixel_coords'][:2]})")
    print(f"  - Hemorrhage Subtypes: {he['subtypes']}")
    assert len(ma["subpixel_coords"]) > 0
    assert "dot_blot" in he["subtypes"]
    assert "flame_shaped" in he["subtypes"]
    assert "preretinal_vitreous" in he["subtypes"]

    # Test Grad-CAM
    cam_jpg = engine.gradcam(canvas_bgr, target="referable")
    print(f"Grad-CAM generated: {len(cam_jpg)} bytes JPEG")
    assert len(cam_jpg) > 1000

    print("PASS: Engine analysis, structures, subpixel MA, NV, and Grad-CAM all verified.\n")


def test_api_report_export():
    print("--- 2. Testing FastAPI TestClient & Clinical Export ---")
    from fastapi.testclient import TestClient
    from certus_api.main import app

    with TestClient(app) as client:
        # Create site, device, patient, encounter
        site = client.post("/v1/sites", headers={"X-API-Key": "demo-admin"},
                           json={"name": "Aligarh Camp", "district": "Aligarh", "state": "Uttar Pradesh", "kind": "camp"}).json()
        dev = client.post("/v1/devices", headers={"X-API-Key": "demo-admin"},
                          json={"site_id": site["id"], "make": "Remidio", "model": "FOP", "domain_key": "remidio_fop"}).json()
        pat = client.post("/v1/patients", headers={"X-API-Key": "demo-tech"},
                          json={"pseudo_id": "PAT-TEST-001", "sex": "M", "birth_year": 1968, "diabetes_years": 12}).json()
        client.post("/v1/consents", headers={"X-API-Key": "demo-tech"}, json={"patient_id": pat["id"], "method": "abdm"})
        enc = client.post("/v1/encounters", headers={"X-API-Key": "demo-tech"}, json={"patient_id": pat["id"], "site_id": site["id"]}).json()

        # Ingest image
        img_path = os.path.join(REPO, "demo_images", "grade3_severe_IDRiD_006.jpg")
        with open(img_path, "rb") as fh:
            uploaded = client.post(f"/v1/encounters/{enc['id']}/images", headers={"X-API-Key": "demo-tech"},
                                   files={"file": ("fundus.jpg", fh, "image/jpeg")},
                                   data={"eye": "R", "field": "macula", "device_id": dev["id"]}).json()
        print(f"Uploaded Image ID: {uploaded['id']}")

        # Run analysis
        ran = client.post(f"/v1/encounters/{enc['id']}/analyse", headers={"X-API-Key": "demo-tech"}).json()
        print(f"Analyse response: ran={ran['ran']}")

        # Test clinical report
        rep = client.get(f"/v1/encounters/{enc['id']}/report", headers={"X-API-Key": "demo-doc"}).json()
        print(f"Screening Report: decision={rep['decision']}, worst_grade={rep['worst_grade']}")
        assert rep["worst_grade"] == 3

        # Test export endpoint (HTML screening report)
        export_resp = client.get(f"/v1/encounters/{enc['id']}/export", headers={"X-API-Key": "demo-doc"})
        assert export_resp.status_code == 200
        assert "text/html" in export_resp.headers["content-type"]
        html_content = export_resp.text
        print(f"Export Report Endpoint: returned {len(html_content)} bytes HTML")
        assert "Certus Screening Report" in html_content
        assert "ICDR Level 3: Severe NPDR" in html_content
        assert "Vessels:" in html_content
        assert "NV Assessment:" in html_content
        assert "Reviewing Ophthalmologist" in html_content
        print("PASS: FastAPI export report endpoint returns fully rendered clinical document.\n")

if __name__ == "__main__":
    test_inference_and_structures()
    test_api_report_export()
    print("ALL TESTS PASSED SUCCESSFULLY!")
