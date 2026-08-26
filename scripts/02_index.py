"""Build Data/manifests/index.csv: one row per raw image with harmonised labels.

Columns: uid, dataset, task, raw_path, official_split, patient_id, dr_grade (0-4, NaN
if unknown/ungradable), dme, gradable, quality_label (0 good / 1 usable / 2 reject),
mask_<MA|HE|EX|SE|OD|VES|FOVMASK> raw paths, od_x/od_y/fovea_x/fovea_y raw coords, canvas.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("D:/Certus/Data/raw")
OUT = Path("D:/Certus/Data/manifests")
LESIONS = ["MA", "HE", "EX", "SE", "OD"]


def row(**kw):
    base = dict(dr_grade=np.nan, dme=np.nan, gradable=np.nan, quality_label=np.nan,
                patient_id=None, canvas=1536)
    base.update(kw)
    base["raw_path"] = Path(base["raw_path"]).as_posix()
    return base


def aptos():
    d = RAW / "APTOS2019"
    df = pd.read_csv(d / "train.csv")
    return [row(uid=f"APTOS_{r.id_code}", dataset="APTOS2019", task="grading",
                raw_path=d / "train_images" / f"{r.id_code}.png", official_split="none",
                dr_grade=r.diagnosis, gradable=1) for r in df.itertuples()]


def messidor2():
    d = RAW / "Messidor2"
    lab = pd.read_csv(d / "messidor_data.csv")
    pairs = pd.read_csv(d / "messidor-2_exam_pairs.csv", sep=";")
    exam = {}
    for i, r in enumerate(pairs.itertuples(index=False)):
        for name in r:
            exam[str(name).strip().lower()] = f"M2exam{i:04d}"
    # label CSV says ".jpg", files on disk are ".JPG" -> match case-insensitively
    files = {p.name.lower(): p for p in (d / "images").iterdir()}
    rows = []
    for r in lab.itertuples():
        key = r.image_id.lower()
        p = files.get(key)
        if p is None:
            continue
        g = int(r.adjudicated_gradable)
        rows.append(row(uid=f"M2_{Path(r.image_id).stem}", dataset="Messidor2", task="grading",
                        raw_path=p, official_split="external_test", patient_id=exam.get(key),
                        dr_grade=r.adjudicated_dr_grade if g else np.nan,
                        dme=r.adjudicated_dme if g else np.nan, gradable=g,
                        quality_label=np.nan if g else 2))
    missing = set(files) - set(lab.image_id.str.lower())
    print(f"  Messidor2: {len(rows)} labelled, {len(missing)} images without a label row")
    return rows


def _idrid_points(csv):
    df = pd.read_csv(csv, encoding="latin1").iloc[:, :3].dropna()
    df.columns = ["name", "x", "y"]
    return {r.name.strip(): (float(r.x), float(r.y)) for r in df.itertuples()}


def idrid():
    d = RAW / "IDRiD"
    rows = []
    # B: grading (+ C: OD / fovea centres, same images)
    # Keyed by split: IDRiD numbers train and test from 001 independently, so one dict over both
    # files let the second file overwrite the first for images 001-103. Train images got test-set
    # disc centres and test images got train-set fovea centres (fixed by fix_idrid_landmarks.py).
    loc = d / "C_Localization" / "labels"
    od = {"train": {}, "test": {}}
    fv = {"train": {}, "test": {}}
    for f in loc.glob("*OD_Center*"):
        od["train" if "Training" in f.name else "test"] |= _idrid_points(f)
    for f in loc.glob("*Fovea_Center*"):
        fv["train" if "Training" in f.name else "test"] |= _idrid_points(f)
    for split, csv in [("train", "a. IDRiD_Disease Grading_Training Labels.csv"),
                       ("test", "b. IDRiD_Disease Grading_Testing Labels.csv")]:
        df = pd.read_csv(d / "B_Grading" / "labels" / csv).iloc[:, :3].dropna()
        df.columns = ["name", "grade", "dme"]
        od_s, fv_s = od[split], fv[split]
        for r in df.itertuples():
            n = r.name.strip()
            # IDRiD restarts numbering in the test split, so the split is part of the uid
            rows.append(row(uid=f"IDRiD_B_{split}_{n}", dataset="IDRiD", task="grading",
                            raw_path=d / "B_Grading" / "images" / split / f"{n}.jpg",
                            official_split=split, dr_grade=int(r.grade), dme=int(r.dme), gradable=1,
                            od_x=od_s.get(n, (np.nan,))[0], od_y=od_s.get(n, (np.nan, np.nan))[1],
                            fovea_x=fv_s.get(n, (np.nan,))[0], fovea_y=fv_s.get(n, (np.nan, np.nan))[1]))
    # A: lesion segmentation. A missing mask file means that lesion is absent (IDRiD convention).
    for split in ["train", "test"]:
        for p in sorted((d / "A_Segmentation" / "images" / split).glob("*.jpg")):
            masks = {}
            for les in LESIONS:
                mp = d / "A_Segmentation" / "masks" / split / les / f"{p.stem}_{les}.tif"
                masks[f"mask_{les}"] = mp.as_posix() if mp.exists() else "absent"
            rows.append(row(uid=f"IDRiD_A_{p.stem}", dataset="IDRiD", task="segmentation",
                            raw_path=p, official_split=split, **masks))
    return rows


def ddr():
    d = RAW / "DDR"
    rows = []
    for split in ["train", "valid", "test"]:
        lab = pd.read_csv(d / "DR_grading" / f"{split}.txt", sep=" ", header=None, names=["name", "grade"])
        for r in lab.itertuples():
            ok = r.grade != 5  # 5 = ungradable in DDR
            rows.append(row(uid=f"DDR_G_{Path(r.name).stem}", dataset="DDR", task="grading",
                            raw_path=d / "DR_grading" / split / r.name,
                            official_split="val" if split == "valid" else split,
                            dr_grade=r.grade if ok else np.nan, gradable=int(ok),
                            quality_label=np.nan if ok else 2))
        seg = d / "lesion_segmentation" / split
        label_dir = seg / ("segmentation label" if (seg / "segmentation label").exists() else "label")
        for p in sorted((seg / "image").glob("*.jpg")):
            masks = {}
            for les in ["MA", "HE", "EX", "SE"]:
                mp = label_dir / les / f"{p.stem}.tif"
                masks[f"mask_{les}"] = mp.as_posix() if mp.exists() else "missing"
            rows.append(row(uid=f"DDR_S_{p.stem}", dataset="DDR", task="segmentation", raw_path=p,
                            official_split="val" if split == "valid" else split, **masks))
    return rows


def eyeq():
    d = RAW / "EyeQ"
    rows = []
    for split, f in [("train", "Label_EyeQ_train.csv"), ("test", "Label_EyeQ_test.csv")]:
        for r in pd.read_csv(d / f).itertuples():
            p = d / "images" / r.image
            if not p.exists():
                continue
            rows.append(row(uid=f"EyeQ_{Path(r.image).stem}", dataset="EyeQ", task="quality",
                            raw_path=p, official_split=split, patient_id=f"EP{r.image.split('_')[0]}",
                            quality_label=int(r.quality), dr_grade=int(r.DR_grade),
                            gradable=int(r.quality != 2), canvas=1024))
    return rows


def eyepacs():
    """EyePACS train images without EyeQ quality labels: auxiliary grading pool (noisy labels)."""
    d = RAW / "EyePACS"
    lab = pd.read_csv(d / "trainLabels.csv")
    grade = dict(zip(lab.image, lab.level))
    rows = []
    for p in sorted((d / "images").glob("*.jpeg")):
        if p.stem not in grade:
            continue
        rows.append(row(uid=f"EyePACS_{p.stem}", dataset="EyePACS", task="grading_aux", raw_path=p,
                        official_split="none", patient_id=f"EP{p.stem.split('_')[0]}",
                        dr_grade=int(grade[p.stem]), gradable=1, canvas=1024))
    return rows


def hrf():
    d = RAW / "HRF"
    rows = []
    for p in sorted((d / "segmentation" / "images").glob("*")):
        m = re.match(r"(\d+)_(\w+)", p.stem)
        if not m:
            continue
        num, cls = int(m.group(1)), m.group(2).lower()
        ves = next((d / "segmentation" / "manual1").glob(f"{p.stem}.*"), None)
        fm = next((d / "segmentation" / "mask").glob(f"{p.stem}_mask.*"), None)
        rows.append(row(uid=f"HRF_V_{p.stem}", dataset="HRF", task="vessels", raw_path=p,
                        official_split="train" if num <= 10 else "test", patient_id=f"HRF{p.stem}",
                        mask_VES=ves.as_posix() if ves else "missing",
                        mask_FOVMASK=fm.as_posix() if fm else "missing",
                        dr_grade=np.nan, gradable=1, hrf_class=cls))
    for p in sorted((d / "quality").glob("*")):
        m = re.match(r"(\d+)_(good|bad)", p.stem, re.I)
        if not m:
            continue
        good = m.group(2).lower() == "good"
        rows.append(row(uid=f"HRF_Q_{p.stem}", dataset="HRF", task="quality", raw_path=p,
                        official_split="test", patient_id=f"HRFQ{m.group(1)}",
                        quality_label=0 if good else 2, gradable=int(good)))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fn in [("APTOS2019", aptos), ("Messidor2", messidor2), ("IDRiD", idrid),
                     ("DDR", ddr), ("EyeQ", eyeq), ("HRF", hrf), ("EyePACS", eyepacs)]:
        if not (RAW / name / ".done").exists():
            print(f"  {name}: not extracted yet, skipped")
            continue
        r = fn()
        print(f"  {name}: {len(r)} rows")
        rows += r
    df = pd.DataFrame(rows)
    df["raw_exists"] = df.raw_path.map(lambda p: Path(p).exists())
    assert df.uid.is_unique, df.uid[df.uid.duplicated()].head()
    df.to_csv(OUT / "index.csv", index=False)
    print(df.groupby(["dataset", "task", "official_split"]).size().to_string())
    print("missing raw files:", (~df.raw_exists).sum())
