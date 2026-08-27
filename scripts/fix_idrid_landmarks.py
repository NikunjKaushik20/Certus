"""Repair the IDRiD optic-disc / fovea coordinates in the existing manifests.

02_index.py used to merge IDRiD's training and testing markup files into one dict keyed by image
name. IDRiD numbers both splits from 001, so for images 001-103 the file read second won: train
images got test-set disc centres, test images got train-set fovea centres. Nothing trained on
these coordinates; they are only the ground truth for the localisation evaluation, which is why
the fovea "error" looked like 350 px.

This re-reads the markups by split, maps each point onto the canvas with the same transform
03_preprocess.py used (retina.normalize_fov on the raw image, then retina.transform_point), and
rewrites od_x/od_y/fovea_x/fovea_y in index.csv and od_cx/od_cy/fovea_cx/fovea_cy in master.csv
and localization.csv. Rows it changes are printed. Backups are written next to each file.

Usage: python scripts/fix_idrid_landmarks.py
"""
import re
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import retina  # noqa: E402

RAW = Path("D:/Certus/Data/raw/IDRiD/C_Localization/labels")
MAN = Path("D:/Certus/Data/manifests")


def points(kind):
    out = {}
    for f in RAW.glob(f"*{kind}_Center*"):
        split = "train" if "Training" in f.name else "test"
        df = pd.read_csv(f, encoding="latin1").iloc[:, :3].dropna()
        df.columns = ["name", "x", "y"]
        for r in df.itertuples():
            out[(split, r.name.strip())] = (float(r.x), float(r.y))
    return out


def main():
    od, fv = points("OD"), points("Fovea")
    I = pd.read_csv(MAN / "index.csv", low_memory=False)
    M = pd.read_csv(MAN / "master.csv", low_memory=False)
    L = pd.read_csv(MAN / "localization.csv")
    changed = 0
    canvas = {}
    for i, r in I[I.uid.str.startswith("IDRiD_B_")].iterrows():
        split, name = re.match(r"IDRiD_B_(train|test)_(.+)", r.uid).groups()
        o, f = od.get((split, name)), fv.get((split, name))
        if o is None or f is None:
            continue
        if (abs(r.od_x - o[0]) > 0.5 or abs(r.od_y - o[1]) > 0.5 or
                abs(r.fovea_x - f[0]) > 0.5 or abs(r.fovea_y - f[1]) > 0.5):
            changed += 1
            print(f"  {r.uid}: od ({r.od_x:.0f},{r.od_y:.0f})->({o[0]:.0f},{o[1]:.0f})  "
                  f"fovea ({r.fovea_x:.0f},{r.fovea_y:.0f})->({f[0]:.0f},{f[1]:.0f})")
        I.loc[i, ["od_x", "od_y", "fovea_x", "fovea_y"]] = [*o, *f]
        img = cv2.imread(r.raw_path)
        _, _, tf, _ = retina.normalize_fov(img, int(r.canvas))
        canvas[r.uid] = (*retina.transform_point(*o, tf), *retina.transform_point(*f, tf))
    cols = ["od_cx", "od_cy", "fovea_cx", "fovea_cy"]
    for df, name in ((M, "master.csv"), (L, "localization.csv")):
        shutil.copy(MAN / name, MAN / (name + ".bak"))
        for uid, v in canvas.items():
            df.loc[df.uid == uid, cols] = v
        df.to_csv(MAN / name, index=False)
    shutil.copy(MAN / "index.csv", MAN / "index.csv.bak")
    I.to_csv(MAN / "index.csv", index=False)
    print(f"{changed} IDRiD grading images had wrong landmarks; {len(canvas)} rewritten")


if __name__ == "__main__":
    main()
