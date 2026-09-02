"""Operating points fixed in advance, and what a small labelled sample from a new camera buys.

Two questions, both answered from saved predictions (no GPU):

1. Fixed thresholds. Every threshold here is fitted on val and then applied unchanged to test
   and Messidor-2. "Sensitivity at 85% specificity" re-fitted on each evaluation set uses that
   set's own labels to choose the operating point, which a screening programme cannot do.

2. New-camera calibration. Messidor-2 (Topcon TRC NW6) is a camera the model never saw, and at
   the val-fitted threshold its specificity collapses. We draw k patients from Messidor-2,
   re-fit the operating threshold on their eyes, and score the remaining patients. Draws are by
   patient (both eyes go together), repeated R times; we report the mean and a 95% interval
   over draws, plus how often a draw meets both targets (sens > 90%, spec > 85%).

   A single threshold can at best land on the ROC curve, and Messidor-2's passes almost exactly
   through (90%, 85%), so a re-fitted threshold meets the targets about half the time at most.
   The margin comes from the screening band. We re-fit both cuts on the same sample, keeping the
   specificity each cut had on val (clear 77.7%, refer 91.8%), and score the system: the band's
   middle goes to an ophthalmologist, who is assumed to grade it correctly.

Usage: python torch/new_camera.py runs_torch/train_<stamp>
Writes <run>/new_camera.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SENS_TARGET, SPEC_TARGET = 0.90, 0.85
PATIENTS = (10, 25, 50, 100, 200)
REPEATS = 1000
MANIFEST = os.path.join(os.path.dirname(__file__), "..", "Data", "manifests", "grading.csv")


def p_ref(logits, T):
    return np.minimum.accumulate(1 / (1 + np.exp(-logits / T)), axis=1)[:, 1]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def band_system(y, p, clear, refer):
    """System sens/spec when eyes between the cuts go to a human who grades them correctly."""
    auto_neg, auto_pos = p <= clear, p > refer
    mid = ~auto_neg & ~auto_pos
    return (float((auto_pos | mid)[y].mean()), float((auto_neg | mid)[~y].mean()), float(mid.mean()),
            float(auto_pos[y].sum() / max((auto_pos | auto_neg)[y].sum(), 1)))


def sens_spec(y, p, thr):
    return float((p[y] > thr).mean()), float((p[~y] <= thr).mean())


def thr_for_spec(p_neg, spec):
    """Smallest threshold that keeps at least `spec` of these negatives at or below it."""
    return float(np.quantile(p_neg, spec, method="higher"))


def summary(x):
    x = np.asarray(x)
    return {"mean": round(float(x.mean()), 4), "lo95": round(float(np.quantile(x, 0.025)), 4),
            "hi95": round(float(np.quantile(x, 0.975)), 4)}


def ece(y, p, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def platt(y, s, iters=200):
    """Fit p = sigmoid(a*s + b) by Newton's method; s is the val-temperature logit."""
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = sigmoid(a * s + b)
        g = np.array([((p - y) * s).sum(), (p - y).sum()])
        w = p * (1 - p) + 1e-9
        H = np.array([[(w * s * s).sum(), (w * s).sum()], [(w * s).sum(), w.sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return a, b


def main():
    run = sys.argv[1]
    Z = np.load(os.path.join(run, "predictions.npz"), allow_pickle=True)
    trust = json.load(open(os.path.join(run, "trust.json")))
    T = trust["temperature"]
    band = trust["screening_band"]

    pv, yv = p_ref(Z["val_logits"], T), Z["val_y"] >= 2
    thr85 = thr_for_spec(pv[~yv], SPEC_TARGET)
    out = {"temperature": T, "val_threshold_85spec": thr85,
           "screening_band": {"clear_below": band["clear_below"], "refer_above": band["refer_above"]},
           "fixed": {}}

    # ------------------------------------------------------------ 1. fixed operating points
    G = pd.read_csv(MANIFEST, low_memory=False)
    test_df = G[G.split == "test"].reset_index(drop=True)
    for part in ("val", "test", "external_messidor2"):
        p, y = p_ref(Z[f"{part}_logits"], T), Z[f"{part}_y"] >= 2
        rows = {}
        for name, t in [("val_85spec", thr85), ("band_clear_cut", band["clear_below"]),
                        ("band_refer_cut", band["refer_above"])]:
            se, sp = sens_spec(y, p, t)
            rows[name] = {"threshold": round(t, 4), "sensitivity": round(se, 4), "specificity": round(sp, 4),
                          "meets_targets": se > SENS_TARGET and sp > SPEC_TARGET}
        rows["n"], rows["prevalence"] = int(len(y)), round(float(y.mean()), 4)
        out["fixed"][part] = rows
    # per dataset inside test, at the val threshold
    p, y = p_ref(Z["test_logits"], T), Z["test_y"] >= 2
    assert len(test_df) == len(y) and ((test_df.dr_grade.to_numpy() >= 2) == y).all(), "manifest order drifted"
    out["fixed"]["test_by_dataset"] = {
        ds: dict(zip(("sensitivity", "specificity"), map(lambda v: round(v, 4), sens_spec(y[m], p[m], thr85))),
                 n=int(m.sum()), camera=str(test_df.domain[m].iloc[0]))
        for ds in test_df.dataset.unique() for m in [(test_df.dataset == ds).to_numpy()]}

    # ------------------------------------------------------------ 2. new-camera calibration
    E = G[G.split == "external_test"].reset_index(drop=True)
    L = Z["external_messidor2_logits"]
    y = Z["external_messidor2_y"] >= 2
    assert len(E) == len(y) and ((E.dr_grade.to_numpy() >= 2) == y).all(), "manifest order drifted"
    p = p_ref(L, T)
    s = np.log(p / (1 - p))                          # calibrated logit for P(grade >= 2)
    groups = E.group_id.to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(20260921)

    base_se, base_sp = sens_spec(y, p, thr85)
    clear_spec = float((pv[~yv] <= band["clear_below"]).mean())    # 0.777 on val
    refer_spec = float((pv[~yv] <= band["refer_above"]).mean())    # 0.918 on val
    b_se, b_sp, b_ab, _ = band_system(y, p, band["clear_below"], band["refer_above"])
    cal = {"patients_total": int(len(uniq)), "repeats": REPEATS,
           "no_calibration": {"sensitivity": round(base_se, 4), "specificity": round(base_sp, 4),
                              "ece": round(ece(y, p), 4),
                              "band_val_fit": {"system_sensitivity": round(b_se, 4),
                                               "system_specificity": round(b_sp, 4),
                                               "abstain_rate": round(b_ab, 4)}},
           "band_spec_levels_from_val": {"clear": round(clear_spec, 4), "refer": round(refer_spec, 4)},
           "by_patients": {}}
    for k in PATIENTS:
        se_l, sp_l, ece_l, thr_l, eyes_l, met = [], [], [], [], [], 0
        bse, bsp, bab, bmet = [], [], [], 0
        for _ in range(REPEATS):
            pick = np.isin(groups, rng.choice(uniq, k, replace=False))
            yc, pc, sc = y[pick], p[pick], s[pick]
            if (~yc).sum() < 2:
                continue
            t = thr_for_spec(pc[~yc], SPEC_TARGET)
            se, sp = sens_spec(y[~pick], p[~pick], t)
            if yc.any() and (~yc).any():
                a, b = platt(yc.astype(float), sc)
                ece_l.append(ece(y[~pick], sigmoid(a * s[~pick] + b)))
            se_l.append(se); sp_l.append(sp); thr_l.append(t); eyes_l.append(int(pick.sum()))
            c, r = thr_for_spec(pc[~yc], clear_spec), thr_for_spec(pc[~yc], refer_spec)
            a_se, a_sp, a_ab, _ = band_system(y[~pick], p[~pick], c, r)
            bse.append(a_se); bsp.append(a_sp); bab.append(a_ab)
            bmet += a_se > SENS_TARGET and a_sp > SPEC_TARGET
            met += se > SENS_TARGET and sp > SPEC_TARGET
        cal["by_patients"][str(k)] = {
            "eyes_mean": round(float(np.mean(eyes_l)), 1), "draws": len(se_l),
            "sensitivity": summary(se_l), "specificity": summary(sp_l), "threshold": summary(thr_l),
            "ece_platt": summary(ece_l) if ece_l else None,
            "meets_both_targets": round(met / len(se_l), 4),
            "band_refit": {"system_sensitivity": summary(bse), "system_specificity": summary(bsp),
                           "abstain_rate": summary(bab), "meets_both_targets": round(bmet / len(bse), 4)}}
    out["new_camera_messidor2"] = cal

    with open(os.path.join(run, "new_camera.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    # ------------------------------------------------------------ print
    print(f"val threshold for 85% spec: {thr85:.4f}  (T={T:.3f})")
    for part in ("val", "test", "external_messidor2"):
        r = out["fixed"][part]
        print(f"\n{part}: n={r['n']} prevalence={r['prevalence']:.1%}")
        for name in ("val_85spec", "band_clear_cut", "band_refer_cut"):
            v = r[name]
            print(f"  {name:15s} thr {v['threshold']:.4f}  sens {v['sensitivity']:.3f}  "
                  f"spec {v['specificity']:.3f}  {'MEETS' if v['meets_targets'] else ''}")
    print("\ntest by dataset @ val threshold:")
    for ds, v in out["fixed"]["test_by_dataset"].items():
        print(f"  {ds:10s} n={v['n']:5d}  sens {v['sensitivity']:.3f}  spec {v['specificity']:.3f}  ({v['camera']})")
    print(f"\nMessidor-2 new-camera calibration ({REPEATS} draws, remainder scored):")
    print(f"  none: sens {base_se:.3f} spec {base_sp:.3f} ece {cal['no_calibration']['ece']:.3f}"
          f"   | val band: system sens {b_se:.3f} spec {b_sp:.3f} abstain {b_ab:.1%}")
    for k, v in cal["by_patients"].items():
        e = v["ece_platt"]
        print(f"  {k:>3} patients (~{v['eyes_mean']:.0f} eyes): "
              f"sens {v['sensitivity']['mean']:.3f} [{v['sensitivity']['lo95']:.3f}-{v['sensitivity']['hi95']:.3f}]  "
              f"spec {v['specificity']['mean']:.3f} [{v['specificity']['lo95']:.3f}-{v['specificity']['hi95']:.3f}]  "
              f"ece {e['mean'] if e else float('nan'):.3f}  both targets met in {v['meets_both_targets']:.0%}")
        b = v["band_refit"]
        print(f"      band re-fit -> system sens {b['system_sensitivity']['mean']:.3f} "
              f"[{b['system_sensitivity']['lo95']:.3f}-{b['system_sensitivity']['hi95']:.3f}]  "
              f"spec {b['system_specificity']['mean']:.3f} "
              f"[{b['system_specificity']['lo95']:.3f}-{b['system_specificity']['hi95']:.3f}]  "
              f"abstain {b['abstain_rate']['mean']:.1%}  both met in {b['meets_both_targets']:.0%}")


if __name__ == "__main__":
    main()
