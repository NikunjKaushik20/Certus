"""Unpack the raw downloads into D:/Certus/Data/raw/<Dataset>/ with a clean layout.

Idempotent: each dataset writes a .done marker and is skipped on re-run.
Downloads are never modified or deleted.
"""
import shutil
import sys
import zipfile
from pathlib import Path

DL = Path("D:/Downloads")
RAW = Path("D:/Certus/Data/raw")
TMP = Path("D:/Certus/Data/_tmp")


def done(name):
    return (RAW / name / ".done").exists()


def mark(name):
    (RAW / name / ".done").write_text("ok")


def extract(zip_path, dest, rename=lambda n: n, keep=lambda n: True):
    """Extract members for which keep(name) is true, writing to dest/rename(name)."""
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not keep(info.filename):
                continue
            rel = rename(info.filename)
            if rel is None:
                continue
            out = dest / rel
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)


def concat(parts, out):
    if out.exists() and out.stat().st_size == sum(p.stat().st_size for p in parts):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as dst:
        for p in parts:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, dst, 16 << 20)
    return out


def idrid():
    name = "IDRiD"
    if done(name):
        return
    d = RAW / name
    # "A. Segmentation/1. Original Images/a. Training Set/x.jpg" -> "A_Segmentation/images/train/x.jpg"
    split = {"a. Training Set": "train", "b. Testing Set": "test"}
    lesion = {"1. Microaneurysms": "MA", "2. Haemorrhages": "HE", "3. Hard Exudates": "EX",
              "4. Soft Exudates": "SE", "5. Optic Disc": "OD"}

    def ren_a(n):
        p = n.split("/")
        if p[1] == "1. Original Images":
            return f"A_Segmentation/images/{split[p[2]]}/{p[-1]}"
        if p[1] == "2. All Segmentation Groundtruths":
            return f"A_Segmentation/masks/{split[p[2]]}/{lesion[p[3]]}/{p[-1]}"
        return f"A_Segmentation/{p[-1]}"

    def ren_bc(prefix):
        def f(n):
            p = n.split("/")
            if p[1] == "1. Original Images":
                return f"{prefix}/images/{split[p[2]]}/{p[-1]}"
            return f"{prefix}/labels/{p[-1]}"
        return f

    extract(DL / "A. Segmentation.zip", d, ren_a)
    extract(DL / "B. Disease Grading.zip", d, ren_bc("B_Grading"))
    # C shares the same 516 images as B; only keep the OD/fovea markups.
    extract(DL / "C. Localization.zip", d, ren_bc("C_Localization"), keep=lambda n: n.endswith(".csv"))
    mark(name)


def aptos():
    name = "APTOS2019"
    if done(name):
        return
    # test_images has no public labels -> skipped (saves ~3.5 GB).
    extract(DL / "aptos2019-blindness-detection.zip", RAW / name,
            keep=lambda n: n.startswith("train_images/") or n == "train.csv")
    mark(name)


def messidor2():
    name = "Messidor2"
    if done(name):
        return
    parts = sorted(DL.glob("IMAGES.zip.0*"))
    assert len(parts) == 4, parts
    z = concat(parts, TMP / "Messidor2_IMAGES.zip")
    extract(z, RAW / name, rename=lambda n: "images/" + n.split("/")[-1])
    extract(DL / "archive.zip", RAW / name)  # adjudicated grades + readme
    shutil.copy(DL / "messidor-2.csv", RAW / name / "messidor-2_exam_pairs.csv")
    z.unlink()
    mark(name)


def eyeq():
    name = "EyeQ"
    if done(name):
        return
    import csv
    wanted = set()
    (RAW / name).mkdir(parents=True, exist_ok=True)
    for f in ["Label_EyeQ_train.csv", "Label_EyeQ_test.csv"]:
        shutil.copy(DL / f, RAW / name / f)
        with open(DL / f) as fh:
            wanted |= {r["image"] for r in csv.DictReader(fh)}
    # Uncropped 1024px EyePACS only, restricted to images that carry an EyeQ label.
    src = "resized_train/resized_train/"
    extract(DL / "archive (1).zip", RAW / name,
            keep=lambda n: (n.startswith(src) and n[len(src):] in wanted) or n == "trainLabels.csv",
            rename=lambda n: "images/" + n[len(src):] if n.startswith(src) else n)
    mark(name)


def eyepacs_rest():
    """EyePACS train images that have DR grades but no EyeQ quality label (auxiliary pool)."""
    name = "EyePACS"
    if done(name):
        return
    import csv
    have = set()
    for f in ["Label_EyeQ_train.csv", "Label_EyeQ_test.csv"]:
        with open(RAW / "EyeQ" / f) as fh:  # copies made by eyeq(); downloads may be deleted
            have |= {r["image"] for r in csv.DictReader(fh)}
    src = "resized_train/resized_train/"
    extract(DL / "archive (1).zip", RAW / name,
            keep=lambda n: (n.startswith(src) and n[len(src):] not in have and n.endswith(".jpeg"))
            or n == "trainLabels.csv",
            rename=lambda n: "images/" + n[len(src):] if n.startswith(src) else n)
    mark(name)


def hrf():
    name = "HRF"
    if done(name):
        return
    extract(DL / "all.zip", RAW / name / "segmentation")
    extract(DL / "allQuality.zip", RAW / name / "quality")
    mark(name)


def ddr():
    name = "DDR"
    if done(name):
        return
    z = TMP / "DDR-dataset.zip"
    if not z.exists():
        print("DDR: combined zip not ready yet, skipping")
        return
    extract(z, RAW / name, rename=lambda n: n.split("/", 1)[1] if n.startswith("DDR-dataset/") else n)
    mark(name)


if __name__ == "__main__":
    steps = {"idrid": idrid, "aptos": aptos, "messidor2": messidor2, "eyeq": eyeq, "hrf": hrf, "ddr": ddr,
             "eyepacs_rest": eyepacs_rest}
    for key in (sys.argv[1:] or steps):
        print("==", key, flush=True)
        steps[key]()
        print("   ok", flush=True)
