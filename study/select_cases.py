"""Pick the reader-study cases: held-out eyes from Indian cameras that Certus sends to a person.

Only eyes the screening band routes to review (auto-referred or abstained) reach an
ophthalmologist, so those are the eyes the 30-second target and the usefulness ratings are about.
Cases come from the test split of APTOS 2019 (India) and IDRiD (India, Kowa VX-10a), never seen in
training, checkpoint selection or calibration. Half auto-referred, half abstained, drawn at random
with a fixed seed; the reference grade is the dataset's own.

Usage: python study/select_cases.py [n_per_arm=25]  -> study/cases.csv, study/images/<case>.jpg
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN = os.path.join(ROOT, "runs_torch", "train_20260912_211356")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    G = pd.read_csv(os.path.join(ROOT, "Data", "manifests", "grading.csv"), low_memory=False)
    I = pd.read_csv(os.path.join(ROOT, "Data", "manifests", "index.csv"), low_memory=False)[["uid", "raw_path"]]
    T = G[G.split == "test"].reset_index(drop=True)
    Z = np.load(os.path.join(RUN, "predictions.npz"), allow_pickle=True)
    t = json.load(open(os.path.join(RUN, "trust.json")))
    assert len(T) == len(Z["test_y"]) and (T.dr_grade.to_numpy().astype(int) == Z["test_y"]).all()
    p = np.minimum.accumulate(1 / (1 + np.exp(-Z["test_logits"] / t["temperature"])), axis=1)[:, 1]
    band = t["screening_band"]
    T["p_referable"] = p
    T["route"] = np.where(p > band["refer_above"], "auto-refer", np.where(p <= band["clear_below"], "auto-clear", "abstain"))
    T = T[T.dataset.isin(["APTOS2019", "IDRiD"])].merge(I, on="uid")
    rng = np.random.default_rng(20260921)
    picks = []
    for route in ("auto-refer", "abstain"):
        pool = T[T.route == route]
        k = min(n, len(pool))
        if k < n:
            print(f"only {k} {route} eyes available")
        picks.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    C = pd.concat(picks).sample(frac=1, random_state=7).reset_index(drop=True)   # shuffle arms together
    C.insert(0, "case", [f"C{i + 1:03d}" for i in range(len(C))])
    out = os.path.join(ROOT, "study", "images")
    os.makedirs(out, exist_ok=True)
    for _, r in C.iterrows():
        shutil.copy(r.raw_path, os.path.join(out, f"{r.case}.jpg"))
    C[["case", "uid", "dataset", "domain", "dr_grade", "referable", "route", "p_referable"]].to_csv(
        os.path.join(ROOT, "study", "cases.csv"), index=False)
    print(C.groupby(["route", "dataset"]).size().to_string())
    print(f"reference grade >= 2 in {int((C.dr_grade >= 2).sum())} of {len(C)} cases")


if __name__ == "__main__":
    main()
