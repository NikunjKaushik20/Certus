"""MATLAB against the API engine on the same photographs. Fails loudly if they drift apart.

The README says the MATLAB path returns the same grade, decision and quality as the console,
p(referable) within a small tolerance, and the same Grad-CAM. That is only true until someone
edits one side, so this checks it instead of remembering it.

    matlab -batch "startup_certus; R = certus_demo(Figure=false, HeatDir='<dir>'); writetable(R, '<dir>/matlab.csv')"
    python torch/agreement.py <dir>

Compares, per image: ICDR grade, referral decision, quality class (must be identical),
p(referable) (|diff| <= P_TOL), the second reader's log-odds (|diff| <= Z_TOL; the decision above
already uses it), lesion counts (reported), and Grad-CAM maps (Pearson r and mean |diff| inside the
FOV, r >= CAM_R_MIN). MATLAB runs every eye as a calibrated camera
(camera factor 1.0), so the engine is asked to decide as a calibrated domain too.
Writes <dir>/agreement.json. Exit code 1 on any failure.
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "api")]
from certus_api import gradcam as _gradcam  # noqa: E402
from certus_api.inference import Engine  # noqa: E402

P_TOL = 0.005
Z_TOL = 0.15          # second-reader log-odds; its cuts are -5.15 and -2.29
CAM_R_MIN = 0.95
CALIBRATED_DOMAIN = "APTOS_mixed_India"


def main():
    d = sys.argv[1]
    M = pd.read_csv(os.path.join(d, "matlab.csv"))
    eng = Engine()
    rows, ok = [], True
    for _, m in M.iterrows():
        stem = os.path.splitext(m.file)[0]
        raw = open(os.path.join(ROOT, "demo_images", m.file), "rb").read()
        prep = eng.preprocess(raw)
        res = eng.analyse(prep["canvas"])
        sr = res.get("second_reader")
        dec = eng.decide(res["p_referable"], quality_probs=res["quality"]["probs"], domain_key=CALIBRATED_DOMAIN,
                         second_logit=sr["logit"] if sr else None)
        counts = {f["lesion_type"]: f["lesion_count"] for f in res["findings"]}

        rgb = cv2.cvtColor(prep["canvas"], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        heat_py = _gradcam.grad_cam(eng.model, rgb, eng.cfg)
        heat_ml = cv2.imread(os.path.join(d, f"{stem}_gradcam.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
        fov = prep["fov"].astype(bool) if prep["fov"] is not None else rgb.max(2) > 0.02
        fov = fov & (rgb.max(2) > 0.02)
        r = float(np.corrcoef(heat_py[fov], heat_ml[fov])[0, 1])
        mad = float(np.abs(heat_py[fov] - heat_ml[fov]).mean())

        row = {"file": m.file,
               "grade": [int(m.grade), res["dr_grade"]],
               "decision": [m.decision, dec["decision"]],
               "quality": [m.quality, res["quality"]["label"]],
               "p_referable": [round(float(m.pReferable), 5), round(res["p_referable"], 5)],
               "second_logit": [round(float(m.get("secondLogit", np.nan)), 4), sr["logit"] if sr else None],
               "lesions_MA_HE_EX_SE": [[int(m[k]) for k in ("MA", "HE", "EX", "SE")],
                                       [int(counts.get(k, 0)) for k in ("MA", "HE", "EX", "SE")]],
               "gradcam_pearson_r": round(r, 4), "gradcam_mean_abs_diff": round(mad, 4)}
        fails = [k for k in ("grade", "decision", "quality") if row[k][0] != row[k][1]]
        if abs(row["p_referable"][0] - row["p_referable"][1]) > P_TOL:
            fails.append("p_referable")
        if sr and not abs(row["second_logit"][0] - row["second_logit"][1]) <= Z_TOL:
            fails.append("second_reader")
        if r < CAM_R_MIN:
            fails.append("gradcam")
        row["failures"] = fails
        ok &= not fails
        rows.append(row)
        print(f"{m.file:36s} grade {row['grade']}  {row['decision'][0]:14s}/{row['decision'][1]:14s} "
              f"p {row['p_referable']}  2nd {row['second_logit']}  lesions {row['lesions_MA_HE_EX_SE']}  cam r={r:.3f} |d|={mad:.3f}"
              f"  {'OK' if not fails else 'FAIL ' + ','.join(fails)}")

    summary = {"images": len(rows), "passed": ok, "p_tolerance": P_TOL, "gradcam_r_min": CAM_R_MIN,
               "max_p_diff": max(abs(r["p_referable"][0] - r["p_referable"][1]) for r in rows),
               "second_tolerance": Z_TOL,
               "max_second_logit_diff": max((abs(r["second_logit"][0] - r["second_logit"][1])
                                             for r in rows if r["second_logit"][1] is not None), default=None),
               "min_gradcam_r": min(r["gradcam_pearson_r"] for r in rows),
               "lesion_counts_identical": sum(r["lesions_MA_HE_EX_SE"][0] == r["lesions_MA_HE_EX_SE"][1] for r in rows),
               "rows": rows}
    json.dump(summary, open(os.path.join(d, "agreement.json"), "w"), indent=1)
    print(f"\n{'PASS' if ok else 'FAIL'}: max |dp| {summary['max_p_diff']:.5f}, "
          f"max |d second logit| {summary['max_second_logit_diff']}, min Grad-CAM r "
          f"{summary['min_gradcam_r']:.4f}, lesion counts identical on {summary['lesion_counts_identical']}/{len(rows)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
