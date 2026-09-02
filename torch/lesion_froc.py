"""Lesion-level FROC and canvas-resolution AUPR on the IDRiD segmentation test images.

structures_eval.py's FROC clamped: even at a 0.98 threshold the model makes more than 8
"false positives" per image under a strict any-overlap rule, so the curve never reached the
1/8..8 FP/image range and every point read the same. Two changes here, both standard for
microaneurysm/lesion detection scoring:

  - thresholds run up to 0.999, so the low-FP end of the curve exists;
  - a predicted component counts as a hit if it lies within TOL canvas px of an annotated lesion
    (the annotation is dilated by TOL), because annotators outline a lesion a pixel or two off
    and a 1-pixel miss is not a false detection.

Also reports canvas-resolution pixel AUPR on the same images, next to the native-resolution AUPR
from structures.json, to separate what the model gets wrong from what resampling costs.

Usage: python torch/lesion_froc.py [tol_px=3]   -> reports/lesion_froc.json
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "api"), os.path.dirname(os.path.abspath(__file__))]
from structures_eval import FROC_FP, Hist, Model  # noqa: E402
from certus_api.inference import LESIONS, MIN_AREA  # noqa: E402

GRID = np.r_[np.linspace(0.02, 0.98, 49), 0.99, 0.995, 0.999]


def main():
    tol = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = Model()
    S = pd.read_csv(os.path.join(ROOT, "Data", "manifests", "segmentation.csv"), low_memory=False)
    S = S[(S.split == "test") & (S.dataset == "IDRiD")]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    acc = {c: {"hit": np.zeros(len(GRID)), "fp": np.zeros(len(GRID)), "n": 0, "imgs": 0} for c in LESIONS}
    hist = {c: Hist() for c in LESIONS}
    for _, r in S.iterrows():
        maps, _, _ = model(cv2.imread(r.proc_path))
        fov = cv2.imread(r.fov_path, cv2.IMREAD_GRAYSCALE) > 0
        for c, name in enumerate(LESIONS):
            mp = r[f"proc_mask_{name}"]
            if not isinstance(mp, str) or not os.path.exists(mp):
                continue
            gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
            hist[name].add(maps[c][fov], gt[fov])
            ng, glab = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
            near = cv2.dilate(gt.astype(np.uint8), ker) > 0
            # which GT lesion each "near" pixel belongs to: dilate the label image the same way
            glab_d = cv2.dilate(glab.astype(np.float32), ker).astype(np.int32)
            a = acc[name]
            a["n"] += ng - 1
            a["imgs"] += 1
            for i, t in enumerate(GRID):
                n, plab, stats, _ = cv2.connectedComponentsWithStats((maps[c] >= t).astype(np.uint8), connectivity=8)
                keep = np.zeros(n, bool)
                keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA[name]
                both = near & keep[plab]
                a["hit"][i] += len(np.unique(glab_d[both & (glab_d > 0)]))
                touched = np.zeros(n, bool)
                touched[np.unique(plab[both])] = True
                a["fp"][i] += int((keep & ~touched).sum())
        print(f"  {r.uid}", flush=True)

    thr = model.thr
    out = {"tolerance_canvas_px": tol, "images": int(len(S)), "classes": {}}
    for name, a in acc.items():
        sens, fppi = a["hit"] / max(a["n"], 1), a["fp"] / max(a["imgs"], 1)
        order = np.argsort(fppi)
        covered = [f for f in FROC_FP if fppi.min() <= f <= fppi.max()]
        pts = {str(f): round(float(np.interp(f, fppi[order], sens[order])), 4) for f in covered}
        k = int(np.argmin(np.abs(GRID - thr[name])))
        out["classes"][name] = {
            "lesions": int(a["n"]),
            "fp_per_image_range": [round(float(fppi.min()), 2), round(float(fppi.max()), 1)],
            "sens_at_fppi": pts,
            "froc_score_over_covered_points": round(float(np.mean(list(pts.values()))), 4) if pts else None,
            "points_covered": f"{len(covered)} of {len(FROC_FP)}",
            "at_deployed_threshold": {"threshold": thr[name], "sensitivity": round(float(sens[k]), 4),
                                      "fp_per_image": round(float(fppi[k]), 2)},
            "canvas_pixel_aupr": round(hist[name].aupr(), 4)}
    nat = json.load(open(os.path.join(ROOT, "reports", "structures.json")))["lesions"]["idrid_native_aupr"]
    for name in LESIONS:
        out["classes"][name]["native_pixel_aupr"] = nat.get(name)
    json.dump(out, open(os.path.join(ROOT, "reports", "lesion_froc.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
