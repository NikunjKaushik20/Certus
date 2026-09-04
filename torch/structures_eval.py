"""Structure-level validation: optic disc and fovea, vessels, lesions, sub-pixel microaneurysms.

Every number is on held-out images only, scored against the dataset's own annotation at the
annotation's native resolution wherever the ground truth exists at native resolution.

  localisation   IDRiD localisation set, test split (115 eyes): Euclidean error of the optic
                 disc centre and the fovea in native pixels (the IDRiD challenge metric), and the
                 share of eyes within one disc radius / one disc diameter.
  vessels        HRF test split (15) and any annotated DRIVE images (test and/or training): the
                 prediction is mapped back to the original image and scored inside the dataset's
                 FOV mask. AUC, and F1 / sensitivity / specificity at 0.5.
  lesions        IDRiD segmentation test set at native resolution: pixel AUPR per class (the
                 IDRiD challenge metric). IDRiD + DDR test at canvas resolution: lesion-level FROC
                 (sensitivity averaged over 1/8..8 false positives per image, as in the ROC
                 microaneurysm challenge) and sensitivity / FP per image at the deployed threshold.
  sub-pixel MA   For every predicted MA that matches an IDRiD ground-truth MA, the distance from
                 the predicted position to the ground-truth centroid, in native pixels, for four
                 localisation rules: integer argmax, probability-weighted centroid (what the
                 engine used), quadratic peak fit, Gaussian (log-quadratic) peak fit.

Usage: python torch/structures_eval.py            -> reports/structures.json
       python torch/structures_eval.py --only vessels   (re-score vessels, e.g. after adding DRIVE)
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "api"), os.path.join(ROOT, "scripts")]
import retina  # noqa: E402
from certus_api.inference import LESIONS, MIN_AREA, Engine  # noqa: E402

MAN = os.path.join(ROOT, "Data", "manifests")
OUT = os.path.join(ROOT, "reports", "structures.json")
FROC_FP = (0.125, 0.25, 0.5, 1, 2, 4, 8)


# ------------------------------------------------------------------ model
class Model:
    def __init__(self):
        self.eng = Engine()
        self.D = self.eng.cfg.canvas
        self.thr = self.eng.calibration["lesion_thresholds"]

    @torch.no_grad()
    def __call__(self, canvas_bgr):
        rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div_(255).to(self.eng.device)
        if self.eng.device == "cuda":
            with torch.autocast("cuda", dtype=getattr(torch, self.eng.cfg.amp_dtype)):
                out = self.eng.model.forward_eye(x.contiguous(memory_format=torch.channels_last), with_maps=True)
        else:
            out = self.eng.model.forward_eye(x, with_maps=True)
        maps = out["lesion_prob"].float().cpu().numpy()[0]
        seg = torch.sigmoid(out["seg_logits"].float()).cpu().numpy()[0]
        return maps, seg, self.eng._structures(seg, self.D, canvas_bgr)


def to_native(prob, tf, H, W):
    """Inverse of retina.normalize_fov for a canvas-sized probability map."""
    D = prob.shape[0]
    side = round(D / tf["s"])
    big = cv2.resize(prob.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((H, W), np.float32)
    x0, y0 = tf["x0"], tf["y0"]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    w = min(W - dx0, side - sx0)
    h = min(H - dy0, side - sy0)
    out[dy0:dy0 + h, dx0:dx0 + w] = big[sy0:sy0 + h, sx0:sx0 + w]
    return out


def read_gray(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:                                    # .gif and some .tif need PIL
        from PIL import Image
        m = np.array(Image.open(p))
    if m.ndim == 3:
        m = m.max(2)
    return m > 0


# ------------------------------------------------------------------ metrics
class Hist:
    """Pooled pixel ROC/PR from a probability histogram: exact to the bin width, any image count."""
    def __init__(self, bins=2000):
        self.pos = np.zeros(bins); self.neg = np.zeros(bins); self.bins = bins

    def add(self, p, y):
        idx = np.minimum((p * self.bins).astype(np.int64), self.bins - 1)
        self.pos += np.bincount(idx[y], minlength=self.bins)
        self.neg += np.bincount(idx[~y], minlength=self.bins)

    def curves(self):
        tp = np.cumsum(self.pos[::-1]); fp = np.cumsum(self.neg[::-1])      # thresholds high -> low
        P, N = self.pos.sum(), self.neg.sum()
        return tp, fp, P, N

    def auc(self):
        tp, fp, P, N = self.curves()
        return float(np.trapezoid(np.r_[0, tp / P], np.r_[0, fp / N]))

    def aupr(self):
        tp, fp, P, _ = self.curves()
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / P
        return float(np.sum(np.diff(np.r_[0, rec]) * prec))

    def best_f1(self):
        tp, fp, P, _ = self.curves()
        f1 = 2 * tp / np.maximum(tp + fp + P, 1)
        k = int(np.argmax(f1))
        return {"f1": round(float(f1[k]), 4), "threshold": round(1 - (k + 1) / self.bins, 4)}

    def at(self, thr):
        k = int(thr * self.bins)
        tp, fn = self.pos[k:].sum(), self.pos[:k].sum()
        fp, tn = self.neg[k:].sum(), self.neg[:k].sum()
        return {"sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
                "f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "accuracy": float((tp + tn) / (tp + tn + fp + fn))}


def components(mask, min_area=1):
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return lab, keep, cents


def lesion_hits(prob, gt, thr, min_area, glab=None, n_gt=None):
    """(GT lesions hit, GT lesions, predicted components that touch no GT lesion).

    One pass over the label images instead of one per component: at low thresholds there are
    thousands of predicted components."""
    if glab is None:
        n, glab = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
        n_gt = n - 1
    n, plab, stats, _ = cv2.connectedComponentsWithStats((prob >= thr).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    both = (glab > 0) & keep[plab]
    hit = len(np.unique(glab[both]))
    touched = np.zeros(n, bool)
    touched[np.unique(plab[both])] = True
    fp = int((keep & ~touched).sum())
    return hit, n_gt, fp


def peak_positions(prob, cx, cy):
    """Four localisation rules around one detected component (canvas px, x/y)."""
    D = prob.shape[0]
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1, y0, y1 = max(0, ix - 4), min(D, ix + 5), max(0, iy - 4), min(D, iy + 5)
    patch = prob[y0:y1, x0:x1]
    py, px = np.unravel_index(np.argmax(patch), patch.shape)
    ax, ay = x0 + px, y0 + py
    res = {"argmax": (float(ax), float(ay))}
    gy, gx = np.mgrid[y0:y1, x0:x1]
    w = patch.sum()
    res["centroid"] = (float((gx * patch).sum() / w), float((gy * patch).sum() / w)) if w > 1e-9 else res["argmax"]

    def fit(f):
        def off(a, b, c):
            den = a - 2 * b + c
            return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        if 0 < ax < D - 1 and 0 < ay < D - 1:
            return (ax + off(f[ay, ax - 1], f[ay, ax], f[ay, ax + 1]),
                    ay + off(f[ay - 1, ax], f[ay, ax], f[ay + 1, ax]))
        return (float(ax), float(ay))
    res["quadratic"] = fit(prob)
    res["gaussian"] = fit(np.log(np.maximum(prob, 1e-6)))
    return res


# ------------------------------------------------------------------ tasks
def localisation(model, idx):
    L = pd.read_csv(os.path.join(MAN, "localization.csv"))
    L = L[L.split == "test"].reset_index(drop=True)
    rows = []
    for _, r in L.iterrows():
        canvas = cv2.imread(r.proc_path)
        _, _, st = model(canvas)
        k = r.native_fov_px / model.D                    # canvas px -> native px
        row = {"uid": r.uid, "disc_found": bool(st.get("disc_x") is not None),
               "fovea_found": bool(st.get("fovea_found"))}
        if st.get("disc_x") is not None:
            row["od_err_native"] = float(np.hypot(st["disc_x"] - r.od_cx, st["disc_y"] - r.od_cy) * k)
            row["dd_native"] = float(st["disc_diameter_px"] * k)
        if st.get("fovea_found"):
            row["fovea_err_native"] = float(np.hypot(st["fovea_x"] - r.fovea_cx, st["fovea_y"] - r.fovea_cy) * k)
        rows.append(row)
    T = pd.DataFrame(rows)
    dd = T.dd_native.median()
    od, fo = T.od_err_native.dropna(), T.fovea_err_native.dropna()
    return {"n": len(T), "disc_located": int(T.disc_found.sum()), "fovea_located": int(T.fovea_found.sum()),
            "median_disc_diameter_native_px": round(float(dd), 1),
            "od_error_native_px": {"mean": round(float(od.mean()), 1), "median": round(float(od.median()), 1)},
            "od_within_1_disc_radius": round(float((od <= dd / 2).sum() / len(T)), 4),
            "fovea_error_native_px": {"mean": round(float(fo.mean()), 1), "median": round(float(fo.median()), 1)},
            "fovea_within_1_disc_diameter": round(float((fo <= dd).sum() / len(T)), 4),
            "fovea_within_1_disc_radius": round(float((fo <= dd / 2).sum() / len(T)), 4),
            "note": "Rates are over all test eyes; an eye where the disc or fovea was not located counts as a miss."}


def vessels(model, idx):
    V = pd.read_csv(os.path.join(MAN, "vessels.csv"))
    sets = {"HRF": [(r.raw_path, r.mask_VES, r.mask_FOVMASK) for _, r in
                    V[V.split == "test"].merge(idx[["uid", "raw_path", "mask_VES", "mask_FOVMASK"]], on="uid").iterrows()]}
    # Certus never trained on DRIVE, so both of its splits are unseen. Many mirrors withhold the
    # test split's manual annotations; images without one are skipped.
    drive = []
    for split in ("test", "training"):
        d = os.path.join(ROOT, "Data", "raw", "DRIVE", split)
        for im in sorted(glob.glob(os.path.join(d, "images", "*.tif"))):
            n = os.path.basename(im).split("_")[0]
            gt = os.path.join(d, "1st_manual", f"{n}_manual1.gif")
            if os.path.isfile(gt):
                drive.append((im, gt, os.path.join(d, "mask", f"{n}_{split}_mask.gif")))
    if drive:
        sets["DRIVE"] = drive
    res = {}
    for name, items in sets.items():
        h = Hist()
        dens = []
        for raw, gtp, fovp in items:
            img = cv2.imread(raw)
            if img is None:
                from PIL import Image
                img = cv2.cvtColor(np.array(Image.open(raw).convert("RGB")), cv2.COLOR_RGB2BGR)
            canvas, cmask, tf, _ = retina.normalize_fov(img, model.D)
            _, seg, _ = model(canvas)
            H, W = img.shape[:2]
            p = to_native(seg[5], tf, H, W)
            gt, fov = read_gray(gtp), read_gray(fovp)
            h.add(p[fov], gt[fov])
            dens.append((float((p[fov] > 0.5).mean()), float(gt[fov].mean())))
        dens = np.array(dens)
        res[name] = {"n": len(items), "auc": round(h.auc(), 4), "at_0.5": {k: round(v, 4) for k, v in h.at(0.5).items()},
                     "best_f1_oracle": h.best_f1(),              # threshold picked on the same images
                     "predicted_vessel_fraction": round(float(dens[:, 0].mean()), 4),
                     "annotated_vessel_fraction": round(float(dens[:, 1].mean()), 4)}
    if "DRIVE" not in res:
        res["DRIVE"] = "not evaluated: no annotated images under Data/raw/DRIVE/{test,training} (download DRIVE and re-run)"
    return res


def lesions(model, idx):
    S = pd.read_csv(os.path.join(MAN, "segmentation.csv"), low_memory=False)
    S = S[S.split == "test"].reset_index(drop=True)
    thr = model.thr
    grid = np.linspace(0.02, 0.98, 49)
    native = {c: Hist() for c in LESIONS}
    froc = {ds: {c: {"hit": np.zeros(len(grid)), "n": 0, "fp": np.zeros(len(grid)), "imgs": 0}
                 for c in LESIONS} for ds in S.dataset.unique()}
    at_thr = {ds: {c: [0, 0, 0, 0] for c in LESIONS} for ds in S.dataset.unique()}
    sub = {k: [] for k in ("argmax", "centroid", "quadratic", "gaussian")}
    I = idx.set_index("uid")
    t0 = time.time()
    for j, r in S.iterrows():
        canvas = cv2.imread(r.proc_path)
        maps, _, _ = model(canvas)
        for c, name in enumerate(LESIONS):
            mp = r[f"proc_mask_{name}"]
            if not isinstance(mp, str) or not os.path.exists(mp):
                continue
            gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
            f = froc[r.dataset][name]
            f["imgs"] += 1
            ng, glab = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
            for gi, t in enumerate(grid):
                hit, n, fp = lesion_hits(maps[c], gt, t, MIN_AREA[name], glab, ng - 1)
                f["hit"][gi] += hit; f["fp"][gi] += fp
            f["n"] += n
            hit, n, fp = lesion_hits(maps[c], gt, thr[name], MIN_AREA[name])
            a = at_thr[r.dataset][name]; a[0] += hit; a[1] += n; a[2] += fp; a[3] += 1

        if r.dataset != "IDRiD":
            continue
        raw = I.loc[r.uid]
        img = cv2.imread(raw.raw_path)
        _, _, tf, _ = retina.normalize_fov(img, model.D)
        H, W = img.shape[:2]
        fovmask = np.zeros((H, W), bool)
        fovmask[max(0, tf["y0"]):tf["y0"] + round(model.D / tf["s"]), max(0, tf["x0"]):tf["x0"] + round(model.D / tf["s"])] = True
        for c, name in enumerate(LESIONS):
            gp = raw[f"mask_{name}"]
            if not isinstance(gp, str) or not os.path.exists(gp):
                continue
            gt = read_gray(gp)
            p = to_native(maps[c], tf, H, W)
            native[name].add(p[fovmask], gt[fovmask])
            if name == "MA":
                glab, gkeep, gcent = components(gt)
                if not gkeep:
                    continue
                gc = np.array([gcent[g] for g in gkeep])
                plab, pkeep, pcent = components(maps[c] >= thr[name], MIN_AREA[name])
                k = 1 / tf["s"]
                for pi in pkeep:
                    pos = peak_positions(maps[c], *pcent[pi])
                    nx, ny = pos["centroid"][0] * k + tf["x0"], pos["centroid"][1] * k + tf["y0"]
                    d = np.hypot(gc[:, 0] - nx, gc[:, 1] - ny)
                    g = int(np.argmin(d))
                    if d[g] > 10:                                   # not matched to a GT MA
                        continue
                    for rule, (x, y) in pos.items():
                        sub[rule].append(float(np.hypot(gc[g, 0] - (x * k + tf["x0"]), gc[g, 1] - (y * k + tf["y0"]))))
        if j % 20 == 0:
            print(f"  lesions {j}/{len(S)} {time.time() - t0:.0f}s", flush=True)

    out = {"idrid_native_aupr": {c: round(native[c].aupr(), 4) for c in LESIONS}, "froc": {}, "at_deployed_threshold": {}}
    for ds, F in froc.items():
        out["froc"][ds] = {}
        out["at_deployed_threshold"][ds] = {}
        for c, f in F.items():
            if not f["imgs"] or not f["n"]:
                continue
            sens, fppi = f["hit"] / f["n"], f["fp"] / f["imgs"]
            order = np.argsort(fppi)
            pts = np.interp(FROC_FP, fppi[order], sens[order])
            out["froc"][ds][c] = {"score": round(float(pts.mean()), 4),
                                  "sens_at_fppi": dict(zip(map(str, FROC_FP), np.round(pts, 4).tolist())),
                                  "lesions": int(f["n"]), "images": int(f["imgs"])}
            a = at_thr[ds][c]
            out["at_deployed_threshold"][ds][c] = {"threshold": thr[c], "sensitivity": round(a[0] / max(a[1], 1), 4),
                                                   "fp_per_image": round(a[2] / max(a[3], 1), 2)}
    s = {k: np.array(v) for k, v in sub.items()}
    n = len(s["argmax"])
    out["subpixel_MA_idrid"] = {
        "matched_MAs": n,
        "canvas_px_in_native_px": "about 2.3 on IDRiD (4288-px originals on a 1536-px canvas)",
        **{rule: {"median_native_px": round(float(np.median(v)), 3), "mean_native_px": round(float(v.mean()), 3)}
           for rule, v in s.items() if n}}
    return out


def main():
    idx = pd.read_csv(os.path.join(MAN, "index.csv"), low_memory=False)
    model = Model()
    if sys.argv[1:] == ["--only", "vessels"]:        # re-score vessels, keep the rest of the report
        res = json.load(open(OUT))
        res["vessels"] = vessels(model, idx)
        print(json.dumps(res["vessels"], indent=1))
        json.dump(res, open(OUT, "w"), indent=1)
        print("->", OUT)
        return
    res = {}
    print("localisation...", flush=True); res["localisation_idrid"] = localisation(model, idx)
    print(json.dumps(res["localisation_idrid"], indent=1))
    print("vessels...", flush=True); res["vessels"] = vessels(model, idx)
    print(json.dumps(res["vessels"], indent=1))
    print("lesions...", flush=True); res["lesions"] = lesions(model, idx)
    print(json.dumps(res["lesions"], indent=1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
