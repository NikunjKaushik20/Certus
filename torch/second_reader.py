"""Two readers: Certus (explainable) and a plain whole-image grader (accurate), agree-or-human.

Each reader has its own screening band, cut at the same val specificities as Certus's deployed
band (clear 77.4%, refer 91.8%), so nothing new is tuned for the combined rule:

  auto-clear   both readers are below their clear cut
  auto-refer   both readers are above their refer cut
  human        anything else: the readers disagree, or either is unsure

"System" sensitivity / specificity count an eye sent to a human as correctly graded.

Three parts, all from saved predictions (no GPU):
  1. fixed      val-fitted cuts on test and Messidor-2, against each reader alone
  2. frontier   the same rule at other band widths, so the workload trade-off is visible
  3. new camera draw k Messidor-2 patients (both eyes together), re-fit both readers' cuts on
                them at the same val specificities, score the remaining patients; 1,000 draws

Usage: python torch/second_reader.py runs_torch/baseline_wholeimage_<stamp> [--write-trust]
Writes <baseline run>/second_reader.json. With --write-trust it also records the second reader
(checkpoint and cuts) in the deployed trust.json, which is what switches it on in the API.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from new_camera import MANIFEST, REPEATS, SENS_TARGET, SPEC_TARGET, summary  # noqa: E402

RUN = os.path.join(os.path.dirname(__file__), "..", "runs_torch", "train_20260912_211356")
PARTS = ("val", "test", "external_messidor2")
PATIENTS = (25, 50, 100, 200)


def scores(bdir):
    """Referable log-odds per reader: Certus at its temperature, baseline raw."""
    B = np.load(os.path.join(bdir, "predictions.npz"))
    C = np.load(os.path.join(RUN, "predictions.npz"), allow_pickle=True)
    T = json.load(open(os.path.join(RUN, "trust.json")))["temperature"]
    out = {}
    for p in PARTS:
        assert (B[f"{p}_y"] == C[f"{p}_y"]).all(), f"{p}: eyes are not in the same order"
        out[p] = (C[f"{p}_y"] >= 2, C[f"{p}_logits"][:, 1] / T, B[f"{p}_logits"][:, 1])
    return out, T


def cuts(s_neg, spec_clear, spec_refer):
    return float(np.quantile(s_neg, spec_clear)), float(np.quantile(s_neg, spec_refer))


def decide(c, b, cc, cb):
    clear = (c <= cc[0]) & (b <= cb[0])
    refer = (c > cc[1]) & (b > cb[1])
    return clear, refer, ~clear & ~refer


def system(y, clear, refer, human):
    return {"system_sensitivity": float((refer | human)[y].mean()),
            "system_specificity": float((clear | human)[~y].mean()),
            "to_human": float(human.mean()),
            "missed_referable": int((clear & y).sum()), "false_auto_refer": int((refer & ~y).sum())}


def main():
    bdir = sys.argv[1]
    S, T = scores(bdir)
    trust = json.load(open(os.path.join(RUN, "trust.json")))
    band = trust["screening_band"]
    yv, cv, bv = S["val"]
    logit = lambda p: float(np.log(p / (1 - p)))               # noqa: E731
    spec_clear = float((cv[~yv] <= logit(band["clear_below"])).mean())
    spec_refer = float((cv[~yv] <= logit(band["refer_above"])).mean())
    cc, cb = cuts(cv[~yv], spec_clear, spec_refer), cuts(bv[~yv], spec_clear, spec_refer)
    out = {"certus_temperature": T, "val_specificities": {"clear": spec_clear, "refer": spec_refer},
           "cuts_logodds": {"certus": cc, "baseline": cb}, "fixed": {}, "frontier": [], "new_camera_messidor2": {}}

    # ------------------------------------------------------------ 1. fixed
    for p in PARTS[1:]:
        y, c, b = S[p]
        alone = lambda s, k: system(y, s <= k[0], s > k[1], (s > k[0]) & (s <= k[1]))   # noqa: E731
        out["fixed"][p] = {"certus alone": alone(c, cc), "baseline alone": alone(b, cb),
                           "two readers": system(y, *decide(c, b, cc, cb))}

    # ------------------------------------------------------------ 2. frontier
    for sc in (0.70, 0.75, round(spec_clear, 3), 0.80, 0.85):
        for sr in (0.85, 0.88, round(spec_refer, 3), 0.95):
            if sr < sc:
                continue
            kc, kb = cuts(cv[~yv], sc, sr), cuts(bv[~yv], sc, sr)
            row = {"spec_clear": sc, "spec_refer": sr}
            for p in PARTS[1:]:
                y, c, b = S[p]
                row[p] = system(y, *decide(c, b, kc, kb))
            out["frontier"].append(row)

    # ------------------------------------------------------------ 3. new camera
    G = pd.read_csv(MANIFEST, low_memory=False)
    E = G[G.split == "external_test"].reset_index(drop=True)
    y, c, b = S["external_messidor2"]
    assert ((E.dr_grade.to_numpy() >= 2) == y).all(), "manifest order drifted"
    groups = E.group_id.to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(20260921)
    for k in PATIENTS:
        rows = {"two readers": [], "certus alone": []}
        for _ in range(REPEATS):
            pick = np.isin(groups, rng.choice(uniq, k, replace=False))
            neg = pick & ~y
            if neg.sum() < 2:
                continue
            kc, kb = cuts(c[neg], spec_clear, spec_refer), cuts(b[neg], spec_clear, spec_refer)
            r, cr, br = ~pick, c[~pick], b[~pick]
            rows["two readers"].append(system(y[r], *decide(cr, br, kc, kb)))
            rows["certus alone"].append(system(y[r], cr <= kc[0], cr > kc[1], (cr > kc[0]) & (cr <= kc[1])))
        res = {}
        for name, L in rows.items():
            se = [d["system_sensitivity"] for d in L]
            sp = [d["system_specificity"] for d in L]
            res[name] = {"system_sensitivity": summary(se), "system_specificity": summary(sp),
                         "to_human": summary([d["to_human"] for d in L]),
                         "meets_both_targets": round(float(np.mean([a > SENS_TARGET and s > SPEC_TARGET
                                                                    for a, s in zip(se, sp)])), 4)}
        out["new_camera_messidor2"][str(k)] = {"draws": len(rows["two readers"]), **res}

    json.dump(out, open(os.path.join(bdir, "second_reader.json"), "w"), indent=1)
    if "--write-trust" in sys.argv:
        tp = os.path.join(RUN, "trust.json")
        shutil.copy(tp, tp + ".bak")
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        trust["second_reader"] = {
            "checkpoint": os.path.relpath(os.path.join(bdir, "best.pt"), root).replace(os.sep, "/"),
            "arch": "efficientnet_b0", "input_size": 512,
            "score": "raw cumulative logit k=1 (grade >= 2)",
            "clear_below_logit": cb[0], "refer_above_logit": cb[1],
            "val_specificities": out["val_specificities"],
            "rule": "auto-decide only when both readers are outside their bands on the same side",
            "source": "torch/second_reader.py",
            # what the rule did on held-out data, for /v1/trust and the console's model page
            "measured": {**{p: out["fixed"][p]["two readers"] for p in PARTS[1:]},
                         "external_messidor2_refit_100": {
                             k: out["new_camera_messidor2"]["100"]["two readers"][k]["mean"]
                             for k in ("system_sensitivity", "system_specificity", "to_human")}}}
        json.dump(trust, open(tp, "w"), indent=1)
        print("second reader written to", tp)

    print(f"val specificities: clear {spec_clear:.3f}  refer {spec_refer:.3f}")
    for p, r in out["fixed"].items():
        for name, m in r.items():
            print(f"{p:20s} {name:15s} sens {m['system_sensitivity']:.3f} spec {m['system_specificity']:.3f} "
                  f"human {m['to_human']:.3f}  missed {m['missed_referable']:4d}  false refer {m['false_auto_refer']:4d}")
    print("\nnew camera (Messidor-2), both readers re-fitted on k patients:")
    for k, r in out["new_camera_messidor2"].items():
        for name in ("certus alone", "two readers"):
            m = r[name]
            print(f"  {k:>3} {name:13s} sens {m['system_sensitivity']['mean']:.3f} "
                  f"[{m['system_sensitivity']['lo95']:.3f}-{m['system_sensitivity']['hi95']:.3f}]  "
                  f"spec {m['system_specificity']['mean']:.3f} "
                  f"[{m['system_specificity']['lo95']:.3f}-{m['system_specificity']['hi95']:.3f}]  "
                  f"human {m['to_human']['mean']:.3f}  both met {m['meets_both_targets']:.0%}")


if __name__ == "__main__":
    main()
