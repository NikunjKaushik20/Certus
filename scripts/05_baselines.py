"""Single-technique baselines for the ablation table (classical image processing + ML).

1. Referable DR from the 37 engineered features (quality + lesion candidates):
   logistic regression and LightGBM, trained on the grading train split.
2. Quality gate from the quality features: EyeQ good/usable/reject + HRF good/bad pairs.

Model selection uses val only; test and Messidor-2 (external) are reported once.
Writes Data/baselines/baselines.json and baselines.md.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("D:/Certus/Data")
OUT = DATA / "baselines"


def sens_at_spec(y, s, spec=0.85):
    thr = np.quantile(s[~y], spec)
    return float((s[y] > thr).mean())


def report(name, y, s):
    y = np.asarray(y, bool)
    return {"model": name, "n": int(len(y)), "auc": round(float(roc_auc_score(y, s)), 4),
            "sens_at_85spec": round(sens_at_spec(y, np.asarray(s)), 4)}


def main():
    OUT.mkdir(exist_ok=True)
    F = pd.read_csv(DATA / "features" / "features.csv")
    G = pd.read_csv(DATA / "manifests" / "grading.csv").merge(F.drop(columns=["dataset", "domain", "split"]), on="uid")
    qcols = [c for c in F.columns if c.startswith("q_")]
    lcols = [c for c in F.columns if c.startswith("l_")]
    res = {"referable": [], "quality": []}

    # ---- 1. referable DR
    tr, parts = G[G.split == "train"], {p: G[G.split == s] for p, s in
                                         [("val", "val"), ("test", "test"), ("external_messidor2", "external_test")]}
    y = lambda d: d.dr_grade.to_numpy() >= 2
    for feats, tag in [(lcols, "lesion-candidate features"), (qcols + lcols, "all engineered features")]:
        med = tr[feats].median()
        X = lambda d: d[feats].fillna(med).to_numpy()
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X(tr), y(tr))
        gb = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, class_weight="balanced", verbose=-1)
        gb.fit(X(tr), y(tr), eval_set=[(X(parts["val"]), y(parts["val"]))], eval_metric="auc",
               callbacks=[lgb.early_stopping(100, verbose=False)])
        for mname, mdl in [("logistic", lr), ("lightgbm", gb)]:
            for p, d in parts.items():
                r = report(f"{mname} | {tag}", y(d), mdl.predict_proba(X(d))[:, 1])
                r["split"] = p
                res["referable"].append(r)
                print(r)

    # ---- 2. quality gate
    Q = pd.read_csv(DATA / "manifests" / "quality.csv").merge(F[["uid"] + qcols], on="uid")
    e = Q[(Q.dataset == "EyeQ") & Q.quality_label.notna()]
    etr, eva = e[e.split == "train"], e[e.split.isin(["val", "test"])]
    med = etr[qcols].median()
    X = lambda d: d[qcols].fillna(med).to_numpy()
    gq = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, class_weight="balanced", verbose=-1)
    gq.fit(X(etr), etr.quality_label.astype(int))
    pq = gq.predict_proba(X(eva))
    res["quality"].append({"model": "lightgbm | quality features", "split": "EyeQ val+test", "n": int(len(eva)),
                           "accuracy_3class": round(float(accuracy_score(eva.quality_label.astype(int), pq.argmax(1))), 4),
                           "auc_reject": round(float(roc_auc_score(eva.quality_label == 2, pq[:, 2])), 4)})
    h = Q[(Q.dataset == "HRF") & Q.quality_label.notna()]
    if len(h):
        ph = gq.predict_proba(X(h))[:, 2]
        res["quality"].append({"model": "lightgbm | quality features", "split": "HRF good/bad pairs (external)",
                               "n": int(len(h)), "auc_reject": round(float(roc_auc_score(h.quality_label == 2, ph)), 4)})
    print(res["quality"])

    with open(OUT / "baselines.json", "w") as fh:
        json.dump(res, fh, indent=2)
    md = ["# Single-technique baselines", "", "## Referable DR (grade ≥ 2)", "",
          pd.DataFrame(res["referable"]).pivot_table(index="model", columns="split", values=["auc", "sens_at_85spec"]).round(3).to_markdown(),
          "", "## Quality gate", "", pd.DataFrame(res["quality"]).to_markdown(index=False)]
    (OUT / "baselines.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
