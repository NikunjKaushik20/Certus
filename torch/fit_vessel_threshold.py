"""Choose the vessel threshold on HRF's train split and score it once on HRF test (and DRIVE if present).

At the default 0.5 the vessel map over-segments on HRF test: 13.2% of the FOV predicted as vessel
against 9.4% annotated, F1 0.736. The threshold that maximises pooled F1 on the 30 HRF training
images is chosen here, never on test, and written to trust.json as lesion.thresholds.VES so the
engine and the MATLAB export use it.

Usage: python torch/fit_vessel_threshold.py [--write]   -> reports/vessel_threshold.json
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "api"), os.path.join(ROOT, "scripts"), os.path.dirname(os.path.abspath(__file__))]
import retina  # noqa: E402
from structures_eval import Hist, Model, read_gray, to_native, vessels  # noqa: E402


def pooled(model, items):
    h = Hist()
    for raw, gtp, fovp in items:
        img = cv2.imread(raw)
        canvas, _, tf, _ = retina.normalize_fov(img, model.D)
        _, seg, _ = model(canvas)
        p = to_native(seg[5], tf, *img.shape[:2])
        fov = read_gray(fovp)
        h.add(p[fov], read_gray(gtp)[fov])
    return h


def main():
    model = Model()
    idx = pd.read_csv(os.path.join(ROOT, "Data", "manifests", "index.csv"), low_memory=False)
    V = pd.read_csv(os.path.join(ROOT, "Data", "manifests", "vessels.csv"))
    tr = V[V.split == "train"].merge(idx[["uid", "raw_path", "mask_VES", "mask_FOVMASK"]], on="uid")
    h = pooled(model, list(zip(tr.raw_path, tr.mask_VES, tr.mask_FOVMASK)))
    grid = np.round(np.arange(0.30, 0.96, 0.01), 2)
    f1 = [h.at(t)["f1"] for t in grid]
    thr = float(grid[int(np.argmax(f1))])
    print(f"HRF train: best F1 {max(f1):.4f} at threshold {thr} (F1 at 0.5: {h.at(0.5)['f1']:.4f})")

    te = V[V.split == "test"].merge(idx[["uid", "raw_path", "mask_VES", "mask_FOVMASK"]], on="uid")
    ht = pooled(model, list(zip(te.raw_path, te.mask_VES, te.mask_FOVMASK)))
    res = {"threshold": thr, "chosen_on": "HRF train split (30 images), max pooled F1",
           "hrf_train_f1": round(max(f1), 4),
           "hrf_test": {"auc": round(ht.auc(), 4),
                        "at_0.5": {k: round(v, 4) for k, v in ht.at(0.5).items()},
                        f"at_{thr}": {k: round(v, 4) for k, v in ht.at(thr).items()}}}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, "reports", "vessel_threshold.json"), "w"), indent=1)
    if "--write" in sys.argv:
        from certus_api.config import settings
        path = settings.resolve_trust()
        t = json.load(open(path))
        t.setdefault("lesion", {}).setdefault("thresholds", {})["VES"] = thr
        json.dump(t, open(path, "w"), indent=1)
        print("->", path)


if __name__ == "__main__":
    main()
