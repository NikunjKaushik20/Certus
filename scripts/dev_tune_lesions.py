"""Tune the classical lesion-candidate thresholds against ground-truth masks.

Uses IDRiD-A + DDR lesion-segmentation *train* images (never val/test). For each
setting reports object-level MA recall/precision and the Spearman correlation between
candidate counts and ground-truth lesion counts across images (what the features are for).
"""
import os
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import retina  # noqa: E402

DATA = Path("D:/Certus/Data")
feat = pd.concat([pd.read_csv(f) for f in (DATA / "features" / "sample").glob("*.csv")])
idx = pd.read_csv(DATA / "manifests" / "index.csv", low_memory=False)
seg = feat.merge(idx[["uid", "task", "official_split", "dataset"]], on="uid")
seg = seg[(seg.task == "segmentation") & (seg.official_split == "train") & (seg.status == "ok")]
seg = pd.concat([seg[seg.dataset == "IDRiD"], seg[seg.dataset == "DDR"].sample(60, random_state=0)])
print(f"tuning on {len(seg)} images: {seg.dataset.value_counts().to_dict()}")

DARK = [(se, mode, t) for se in (7, 11, 15) for mode, t in
        [("std", 2.0), ("std", 3.0), ("std", 4.0), ("abs", 6), ("abs", 9), ("abs", 12), ("abs", 16)]]
BRIGHT = [(se, mode, t) for se in (15, 25, 41) for mode, t in
          [("std", 3.0), ("std", 4.0), ("std", 6.0), ("abs", 10), ("abs", 15), ("abs", 20), ("abs", 30)]]
dark_stats = {k: [] for k in DARK}
bright_stats = {k: [] for k in BRIGHT}


def load(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0


for i, r in enumerate(seg.itertuples(), 1):
    img, fov = cv2.imread(r.proc_path), load(r.fov_path).astype(np.uint8)
    ctx = retina.lesion_context(img, fov)
    gt_ma, gt_he, gt_ex = load(r.proc_mask_MA), load(r.proc_mask_HE), load(r.proc_mask_EX)
    red_tol = cv2.dilate((gt_ma | gt_he).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    n_ma, lab_ma = cv2.connectedComponents(gt_ma.astype(np.uint8))
    n_he = cv2.connectedComponents(gt_he.astype(np.uint8))[0] - 1
    for se in (7, 11, 15):
        resp = retina.dark_response(ctx, se)
        for key in [k for k in DARK if k[0] == se]:
            p = {**retina.LESION_PARAMS["dark"], "se": se, "mode": key[1], "t": key[2]}
            ma, he, cen = retina.dark_candidates(ctx, resp, p)
            cand = ma | he
            hit_gt = len(np.setdiff1d(np.unique(lab_ma[cv2.dilate(cand.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0]), [0]))
            nc, lab_c = cv2.connectedComponents(cand.astype(np.uint8))
            tp_c = len(np.setdiff1d(np.unique(lab_c[red_tol]), [0]))
            n_ma_c = cv2.connectedComponents(ma.astype(np.uint8))[0] - 1
            dark_stats[key].append((n_ma - 1, hit_gt, nc - 1, tp_c, n_ma_c, len(cen), n_he))
    for se in (15, 25, 41):
        resp = retina.bright_response(ctx, se)
        for key in [k for k in BRIGHT if k[0] == se]:
            p = {**retina.LESION_PARAMS["bright"], "se": se, "mode": key[1], "t": key[2]}
            ex = retina.bright_candidates(ctx, resp, p)
            inter = (ex & gt_ex).sum()
            bright_stats[key].append((gt_ex.sum(), ex.sum(), inter))
    if i % 20 == 0:
        print(f"  {i}/{len(seg)}", flush=True)

rows = []
for k, v in dark_stats.items():
    a = np.array(v)
    rec, prec = a[:, 1].sum() / max(a[:, 0].sum(), 1), a[:, 3].sum() / max(a[:, 2].sum(), 1)
    rows.append({"se": k[0], "mode": k[1], "t": k[2], "MA_recall": rec, "red_precision": prec,
                 "F1": 2 * rec * prec / max(rec + prec, 1e-9),
                 "rho_MA_count": spearmanr(a[:, 4], a[:, 0])[0], "rho_HE_count": spearmanr(a[:, 5], a[:, 6])[0]})
dark = pd.DataFrame(rows).sort_values("F1", ascending=False)
print("\nDARK (MA/HE) candidates\n" + dark.round(3).head(12).to_string(index=False))

rows = []
for k, v in bright_stats.items():
    a = np.array(v, dtype=float)
    rec, prec = a[:, 2].sum() / a[:, 0].sum(), a[:, 2].sum() / max(a[:, 1].sum(), 1)
    rows.append({"se": k[0], "mode": k[1], "t": k[2], "EX_px_recall": rec, "EX_px_precision": prec,
                 "dice": 2 * a[:, 2].sum() / (a[:, 0].sum() + a[:, 1].sum()),
                 "rho_EX_area": spearmanr(a[:, 1], a[:, 0])[0]})
bright = pd.DataFrame(rows).sort_values("dice", ascending=False)
print("\nBRIGHT (EX) candidates\n" + bright.round(3).head(12).to_string(index=False))
dark.to_csv(DATA / "features" / "tuning_dark.csv", index=False)
bright.to_csv(DATA / "features" / "tuning_bright.csv", index=False)
