"""FOV-normalise every indexed image, transform its masks/landmarks, and extract features.

Outputs
  Data/processed/<dataset>/img/<uid>.jpg        canvas image (FOV diameter = canvas px)
  Data/processed/<dataset>/fov/<uid>.png        FOV mask (0/255)
  Data/processed/<dataset>/masks/<LES>/<uid>.png lesion / vessel masks (0/255)
  Data/features/parts/<dataset>.csv             per-image features, hashes, geometry

Usage: python 03_preprocess.py [--limit N] [--datasets DDR IDRiD ...] [--workers 7]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import retina  # noqa: E402

DATA = Path("D:/Certus/Data")
PROC = DATA / "processed"
MASK_COLS = ["mask_MA", "mask_HE", "mask_EX", "mask_SE", "mask_OD", "mask_VES"]


def read_mask(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:  # some TIFF/GIF variants
        from PIL import Image
        m = np.array(Image.open(p))
    if m.ndim == 3:
        m = m.max(axis=2)
    return m


def process(r):
    cv2.setNumThreads(1)
    out = {"uid": r["uid"], "status": "ok"}
    try:
        img = cv2.imread(r["raw_path"], cv2.IMREAD_COLOR)
        if img is None:
            return {**out, "status": "unreadable"}
        D = int(r["canvas"])
        can, fov, tf, info = retina.normalize_fov(img, D)
        out.update(info)
        if can is None or info["fov_area_frac"] < 0.3:
            return {**out, "status": "no_fov"}

        base = PROC / r["dataset"]
        ip, fp = base / "img" / f"{r['uid']}.jpg", base / "fov" / f"{r['uid']}.png"
        ip.parent.mkdir(parents=True, exist_ok=True)
        fp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ip), can, [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                                   cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
        cv2.imwrite(str(fp), fov * 255)
        out["proc_path"], out["fov_path"] = ip.as_posix(), fp.as_posix()

        for c in MASK_COLS:
            v = r.get(c)
            if not isinstance(v, str):
                continue
            les = c.split("_")[1]
            mp = base / "masks" / les / f"{r['uid']}.png"
            mp.parent.mkdir(parents=True, exist_ok=True)
            if v == "absent":  # annotated as lesion-free -> explicit empty mask
                m = np.zeros((D, D), np.uint8)
            elif v == "missing":
                out[f"proc_{c}"] = "unknown"
                continue
            else:
                m = retina.transform_mask(read_mask(v), tf, D) & fov
            cv2.imwrite(str(mp), m * 255)
            out[f"proc_{c}"] = mp.as_posix()
            out[f"px_{les}"] = int(m.sum())

        for k in ["od", "fovea"]:
            x, y = r.get(f"{k}_x"), r.get(f"{k}_y")
            if x is not None and not pd.isna(x):
                out[f"{k}_cx"], out[f"{k}_cy"] = retina.transform_point(x, y, tf)

        out["dhash"], out["phash"] = retina.image_hashes(can)
        out.update({f"q_{k}": v for k, v in retina.quality_features(can, fov).items()})
        if r["dataset"] not in ("EyeQ", "EyePACS"):  # 1024px EyePACS: no lesion-scale features
            out.update({f"l_{k}": v for k, v in retina.lesion_features(can, fov).items()})
    except Exception as e:  # keep going; failures are reported by the cleaning step
        out["status"] = f"error: {type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="per-dataset sample size (smoke test)")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()

    idx = pd.read_csv(DATA / "manifests" / "index.csv", low_memory=False)
    parts = DATA / "features" / ("sample" if a.limit else "parts")
    parts.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets or idx.dataset.unique():
        dst = parts / f"{ds}.csv"
        if dst.exists() and not a.limit:
            print(f"{ds}: done already, skipping")
            continue
        sub = idx[idx.dataset == ds]
        if a.limit:
            sub = sub.groupby("task", group_keys=False).apply(lambda g: g.sample(min(len(g), a.limit), random_state=0))
        rows = [r._asdict() for r in sub.itertuples(index=False)]
        t0, res = time.time(), []
        with Pool(a.workers) as pool:
            for i, o in enumerate(pool.imap_unordered(process, rows, chunksize=4), 1):
                res.append(o)
                if i % 500 == 0 or i == len(rows):
                    print(f"{ds}: {i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(res)
        df.to_csv(dst, index=False)
        bad = (df.status != "ok").sum()
        print(f"{ds}: {len(df)} processed, {bad} failed -> {dst}", flush=True)


if __name__ == "__main__":
    main()
