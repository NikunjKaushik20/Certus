"""Clean, de-duplicate, harmonise labels and assign leakage-free splits.

Reads  Data/manifests/index.csv + Data/features/parts/*.csv
Writes Data/manifests/{master,grading,segmentation,quality,localization,vessels}.csv,
       Data/features/features.csv, Data/manifests/cleaning_report.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import spearmanr

DATA = Path("D:/Certus/Data")
MAN = DATA / "manifests"
# Camera-domain proxy (public sets rarely carry per-image camera metadata)
DOMAIN = {"APTOS2019": "APTOS_mixed_India", "Messidor2": "Topcon_TRC_NW6", "IDRiD": "Kowa_VX-10a",
          "DDR": "DDR_mixed_China", "EyeQ": "EyePACS_mixed_US", "HRF": "Canon_CR-1"}
PROTECT = {"external_test": 3, "test": 2, "val": 1, "train": 0}

log = []


def say(s=""):
    print(s)
    log.append(s)


class DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def grouped_split(df, fracs, seed=0):
    """Stratified (by dr_grade / quality) group split into train/val/test."""
    y = df.dr_grade.fillna(df.quality_label).fillna(-1).astype(int)
    folds = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    out = pd.Series("train", index=df.index)
    n_val, n_test = round(fracs[1] * 20), round(fracs[2] * 20)
    for k, (_, te) in enumerate(folds.split(df, y, df.group_id)):
        if k < n_test:
            out.iloc[te] = "test"
        elif k < n_test + n_val:
            out.iloc[te] = "val"
    return out


def main():
    idx = pd.read_csv(MAN / "index.csv")
    feat = pd.concat([pd.read_csv(f) for f in sorted((DATA / "features" / "parts").glob("*.csv"))])
    df = idx.merge(feat, on="uid", how="left")
    df["domain"] = df.dataset.map(DOMAIN)
    say(f"# Certus data cleaning report\n\nIndexed images: {len(df)}")

    # ---- 1. integrity
    df["status"] = df.status.fillna("not_processed")
    bad = df.status != "ok"
    say(f"\n## 1. Integrity\nDropped {bad.sum()} images: {df.loc[bad, 'status'].str.split(':').str[0].value_counts().to_dict()}")
    df["exclude_reason"] = np.where(bad, df.status, "")

    # ---- 2. flags (kept, but visible to training / evaluation)
    ok = ~bad
    df["flag_low_res"] = ok & (df.native_fov_px < 600)
    df["flag_clipped_fov"] = ok & (df.fov_clip_frac > 0.35)
    say(f"\n## 2. Flags (kept)\nlow native resolution (<600px FOV): {df.flag_low_res.sum()}, "
        f"heavily clipped FOV (>35%): {df.flag_clipped_fov.sum()}")

    # ---- 3. duplicates & groups
    dsu = DSU(df.uid)
    # EyeQ/EyePACS are grouped by patient id instead. Ungradable photos are nearly featureless,
    # look alike to any texture descriptor and would chain unrelated images, so they are left out.
    cand = set(df.uid[ok & ~df.dataset.isin(["EyeQ", "EyePACS"]) & (df.quality_label != 2)])
    cal = json.loads((DATA / "features" / "dup_calibration.json").read_text())
    P = pd.read_csv(DATA / "features" / "dup_pairs.csv")
    P = P[(P.sim >= cal["threshold"]) & P.a.isin(cand) & P.b.isin(cand)]
    pairs = list(zip(P.a, P.b))
    for a, b in pairs:
        dsu.union(a, b)
    # DDR lesion-seg images share file stems with the grading set
    g_stems = {u[6:]: u for u in df.uid if u.startswith("DDR_G_")}
    name_links = [(u, g_stems[u[6:]]) for u in df.uid if u.startswith("DDR_S_") and u[6:] in g_stems]
    for a, b in name_links:
        dsu.union(a, b)
    df["dup_cluster"] = df.uid.map(dsu.find)
    cross = pd.DataFrame(pairs, columns=["a", "b"])
    cross["da"], cross["db"] = cross.a.map(df.set_index("uid").dataset), cross.b.map(df.set_index("uid").dataset)
    say(f"\n## 3. Duplicates\nSame-photo pairs (retinal-texture cosine >= {cal['threshold']}, calibrated on "
        f"{cal['n_known_dup']} known DDR duplicates; see dup_calibration.json): {len(pairs)}; "
        f"DDR seg<->grading name links: {len(name_links)}")
    say("Pairs by dataset: " + str(cross.groupby(["da", "db"]).size().to_dict()))
    sizes = df.dup_cluster.value_counts()
    say(f"Duplicate clusters with >1 image: {(sizes > 1).sum()}; largest cluster: {sizes.max()} images")

    # propagate grades to segmentation images from their graded twin
    graded = df[df.task == "grading"].dropna(subset=["dr_grade"])
    cl_grade = graded.groupby("dup_cluster").dr_grade.agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
    seg = df.task == "segmentation"
    df.loc[seg, "dr_grade"] = df.loc[seg, "dup_cluster"].map(cl_grade)
    say(f"Segmentation images that inherited a DR grade from their graded twin: {df.loc[seg, 'dr_grade'].notna().sum()}/{seg.sum()}")

    # label conflicts inside a duplicate cluster (same photo, different grade)
    gmask = ok & (df.task == "grading") & df.dr_grade.notna()
    nun = df[gmask].groupby("dup_cluster").dr_grade.nunique()
    conflict = df.dup_cluster.isin(nun[nun > 1].index) & gmask
    df.loc[conflict, "exclude_reason"] = "label_conflict"
    # identical-label duplicates: keep the highest-resolution copy for grading
    rest = gmask & ~conflict
    rank = df[rest].sort_values("native_fov_px", ascending=False).groupby("dup_cluster").cumcount()
    dup = rest & df.index.isin(rank[rank > 0].index)
    df.loc[dup, "exclude_reason"] = "duplicate"
    say(f"Grading images dropped for conflicting duplicate labels: {conflict.sum()}; redundant duplicate copies dropped: {dup.sum()}")

    # patient-level grouping on top of duplicate clusters (left/right eyes, exam pairs)
    for pid, g in df[df.patient_id.notna()].groupby("patient_id"):
        u = g.uid.tolist()
        for x in u[1:]:
            dsu.union(u[0], x)
    df["group_id"] = df.uid.map(dsu.find)

    # ---- 4. splits
    df["split"] = df.official_split.replace({"none": np.nan})
    # EyeQ and the auxiliary EyePACS pool share patients, so they are split together
    for dss, fr in [(["APTOS2019"], (0.70, 0.15, 0.15)), (["EyeQ", "EyePACS"], (0.80, 0.10, 0.10))]:
        m = df.dataset.isin(dss)
        if m.any():
            df.loc[m, "split"] = grouped_split(df[m], fr)
    # IDRiD grading has no val split: carve ~15% of train
    m = (df.dataset == "IDRiD") & (df.task == "grading") & (df.split == "train")
    s = grouped_split(df[m], (0.85, 0.15, 0.0))
    df.loc[m, "split"] = s.replace({"test": "val"})
    # leakage guard: a group lives entirely in its most protected split
    lvl = df.split.map(PROTECT)
    top = lvl.groupby(df.group_id).transform("max")
    moved = (lvl != top).sum()
    df["split"] = top.map({v: k for k, v in PROTECT.items()})
    say(f"\n## 4. Splits\nImages moved to a stricter split to stop cross-split leakage: {moved}")
    use = df.exclude_reason == ""
    say("\n" + pd.crosstab([df[use].dataset, df[use].task], df[use].split).to_markdown())

    # ---- 5. harmonised targets
    df["referable"] = (df.dr_grade >= 2).astype("float").where(df.dr_grade.notna())
    say("\n## 5. DR grade distribution (usable grading images)\n")
    gm = use & (df.task == "grading") & df.dr_grade.notna()
    say(pd.crosstab([df[gm].dataset, df[gm].split], df[gm].dr_grade.astype(int)).to_markdown())

    # ---- 6. feature sanity: do engineered features carry signal?
    fcols = [c for c in df.columns if c.startswith(("q_", "l_"))]
    say("\n## 6. Feature validation\n### Quality features: AUC for EyeQ 'reject' vs 'good' (train split)\n")
    e = df[use & (df.dataset == "EyeQ") & (df.split == "train") & df.quality_label.isin([0, 2])]
    rows = []
    if e.quality_label.nunique() == 2:
        for c in [c for c in fcols if c.startswith("q_")]:
            x = e[c].fillna(e[c].median())
            auc = roc_auc_score(e.quality_label == 2, x)
            rows.append((c, round(max(auc, 1 - auc), 3), "higher=worse" if auc >= 0.5 else "lower=worse"))
        say(pd.DataFrame(rows, columns=["feature", "AUC", "direction"]).sort_values("AUC", ascending=False).to_markdown(index=False))
    else:
        say("(EyeQ not processed yet — skipped)")
    say("\n### Lesion-candidate features: Spearman rho with DR grade (grading train split, excl. EyeQ)\n")
    t = df[gm & (df.split == "train")]
    rows = [(c, round(spearmanr(t[c], t.dr_grade, nan_policy="omit")[0], 3)) for c in fcols if c.startswith("l_")]
    say(pd.DataFrame(rows, columns=["feature", "rho"]).sort_values("rho", ascending=False).to_markdown(index=False))

    # ---- 7. write manifests
    base = ["uid", "dataset", "domain", "task", "split", "group_id", "proc_path", "fov_path",
            "native_fov_px", "canvas", "flag_low_res", "flag_clipped_fov"]
    df.to_csv(MAN / "master.csv", index=False)
    df[use & (df.task == "grading") & df.dr_grade.notna()][base + ["dr_grade", "referable", "dme"]] \
        .to_csv(MAN / "grading.csv", index=False)
    df[use & (df.task == "grading_aux") & df.dr_grade.notna()][base + ["dr_grade", "referable"]] \
        .to_csv(MAN / "grading_aux.csv", index=False)
    mcols = [c for c in df.columns if c.startswith("proc_mask_") or c.startswith("px_")]
    df[use & df.task.isin(["segmentation"])][base + ["dr_grade"] + mcols].to_csv(MAN / "segmentation.csv", index=False)
    df[use & df.task.eq("vessels")][base + mcols].to_csv(MAN / "vessels.csv", index=False)
    qm = use & (df.quality_label.notna() | df.gradable.notna())
    df[qm][base + ["quality_label", "gradable"]].to_csv(MAN / "quality.csv", index=False)
    df[use & df.od_cx.notna()][base + ["od_cx", "od_cy", "fovea_cx", "fovea_cy"]].to_csv(MAN / "localization.csv", index=False)
    df[use][["uid", "dataset", "domain", "split"] + fcols].to_csv(DATA / "features" / "features.csv", index=False)
    say("\n## 7. Output manifests\n")
    for f in ["grading", "segmentation", "vessels", "quality", "localization"]:
        say(f"- `{f}.csv`: {len(pd.read_csv(MAN / f'{f}.csv'))} rows")
    (MAN / "cleaning_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
