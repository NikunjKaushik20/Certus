"""Fit the fovea-from-disc rule on IDRiD's localisation TRAIN split; score it on TEST.

The engine placed the fovea at a fixed 2.5 disc diameters temporal and 0.1 DD inferior of the disc
centre, with the disc diameter taken from the segmented disc area. On the IDRiD localisation test
set that put the fovea a median 350 native px from the annotation (the challenge winner: 64 px).
Two things can be wrong: the constants, and the ruler (a segmented disc diameter is noisy, and it
is inflated by peripapillary atrophy). This fits both choices on train only:

  ruler    disc diameter (DD) or FOV diameter (the canvas side, D)
  offsets  median horizontal and vertical fovea offset in that unit, horizontal signed temporal
  refine   optional: move to the darkest point of the smoothed green channel within r of the
           geometric estimate (the foveal pit is the darkest spot in a healthy macula; the prior
           keeps it from wandering to a haemorrhage elsewhere)

and reports the error of each on test. Writes torch/fovea_rule.json, which the engine reads.

Usage: python torch/fit_fovea.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "api"), os.path.dirname(os.path.abspath(__file__))]
from structures_eval import Model  # noqa: E402


def dark_refine(canvas, fx, fy, r):
    g = cv2.GaussianBlur(canvas[:, :, 1].astype(np.float32), (0, 0), max(2.0, r / 6))
    D = g.shape[0]
    x0, x1 = int(max(0, fx - r)), int(min(D, fx + r + 1))
    y0, y1 = int(max(0, fy - r)), int(min(D, fy + r + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - fx) ** 2 + (yy - fy) ** 2 <= r * r
    patch = np.where(inside, g[y0:y1, x0:x1], np.inf)
    iy, ix = np.unravel_index(np.argmin(patch), patch.shape)
    return float(x0 + ix), float(y0 + iy)


def main():
    model = Model()
    D = model.D
    L = pd.read_csv(os.path.join(ROOT, "Data", "manifests", "localization.csv"))
    rows = []
    for _, r in L[L.split.isin(["train", "test"])].iterrows():
        canvas = cv2.imread(r.proc_path)
        _, seg, st = model(canvas)
        if st.get("disc_x") is None:
            rows.append({"split": r.split, "ok": False, "k": r.native_fov_px / D})
            continue
        ox, oy, dd = st["disc_x"], st["disc_y"], st["disc_diameter_px"]
        temporal = -1.0 if ox >= D / 2 else 1.0
        rows.append({"split": r.split, "ok": True, "k": r.native_fov_px / D, "ox": ox, "oy": oy, "dd": dd,
                     "t": temporal, "fx": r.fovea_cx, "fy": r.fovea_cy, "path": r.proc_path})
    T = pd.DataFrame(rows)
    tr, te = T[(T.split == "train") & T.ok], T[T.split == "test"]

    rules = {}
    for ruler in ("dd", "D"):
        u = tr.dd if ruler == "dd" else D
        a = float(np.median((tr.fx - tr.ox) * tr.t / u))      # temporal distance, in ruler units
        b = float(np.median((tr.fy - tr.oy) / u))             # downward distance
        rules[ruler] = {"ruler": ruler, "temporal": round(a, 4), "inferior": round(b, 4)}

    def predict(row, rule, refine_r=None):
        u = row.dd if rule["ruler"] == "dd" else D
        fx = row.ox + row.t * rule["temporal"] * u
        fy = row.oy + rule["inferior"] * u
        if refine_r:
            fx, fy = dark_refine(cv2.imread(row.path), fx, fy, refine_r * row.dd)
        return fx, fy

    def score(df, rule, refine_r=None):
        err = []
        for _, row in df.iterrows():
            if not row.ok:
                err.append(np.inf)
                continue
            fx, fy = predict(row, rule, refine_r)
            err.append(np.hypot(fx - row.fx, fy - row.fy) * row.k)
        err = np.array(err)
        dd_native = float(np.median(df[df.ok].dd * df[df.ok].k))
        return {"median_native_px": round(float(np.median(err)), 1),
                "mean_native_px_located": round(float(err[np.isfinite(err)].mean()), 1),
                "within_1_disc_radius": round(float((err <= dd_native / 2).mean()), 4),
                "within_1_disc_diameter": round(float((err <= dd_native).mean()), 4)}

    old = {"ruler": "dd", "temporal": 2.5, "inferior": 0.1}
    cands = {"engine rule (2.5 DD, 0.1 DD)": (old, None)}
    for ruler, rule in rules.items():
        cands[f"fitted, ruler={ruler}"] = (rule, None)
    # refine radius chosen on train from a small grid
    best_r, best_med = None, np.inf
    for rr in (0.5, 0.75, 1.0):
        med = score(tr, rules["D"], rr)["median_native_px"]
        if med < best_med:
            best_r, best_med = rr, med
    cands[f"fitted, ruler=D, + darkest point within {best_r} DD"] = (rules["D"], best_r)

    res = {"train_n": int(len(tr)), "test_n": int(len(te)), "fitted": rules, "refine_radius_dd": best_r,
           "train": {k: score(tr, r, rr) for k, (r, rr) in cands.items()},
           "test": {k: score(te, r, rr) for k, (r, rr) in cands.items()},
           "idrid_challenge_best_fovea_native_px": 64.49}
    # choose on TRAIN median, report test
    choice = min(cands, key=lambda k: res["train"][k]["median_native_px"])
    rule, rr = cands[choice]
    res["chosen_on_train"] = choice
    json.dump({**rule, "refine_radius_dd": rr, "fitted_on": "IDRiD localisation train split",
               "note": choice}, open(os.path.join(ROOT, "torch", "fovea_rule.json"), "w"), indent=1)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "reports", "fovea_fit.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
