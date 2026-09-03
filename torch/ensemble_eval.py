"""Can the plain whole-image grader rescue Certus's accuracy without losing its explanations?

The end-to-end baseline (torch/train_baseline.py) beats Certus on AUC, kappa and, most of all,
Messidor-2 specificity. This scores both on the same eyes and compares three ways to combine them.
Every weight and every cut is fitted on val and applied unchanged to test and Messidor-2.

  certus band      Certus alone, its deployed screening band (the reference)
  baseline band    the baseline alone, with a band fitted at the same val specificities
  average          mean of the two referable log-odds (weight chosen on val AUC), band refitted
  second reader    Certus's band decides; an auto-refer or auto-clear the baseline disagrees with
                   (the baseline's own band puts the eye on the other side) goes to abstain

"System" sensitivity / specificity count an abstained eye as correct (an ophthalmologist reads
it), as everywhere else in this repo.

Usage: python torch/ensemble_eval.py <baseline_run_dir>   -> <baseline_run_dir>/ensemble.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import Config  # noqa: E402
from certus.evaluate import auc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN = os.path.join(ROOT, "runs_torch", "train_20260912_211356")
PARTS = ("val", "test", "external_messidor2")


def baseline_logits(bdir):
    cache = os.path.join(bdir, "predictions.npz")
    if os.path.isfile(cache):
        return dict(np.load(cache))
    from train_baseline import WholeImage, SIZE  # noqa: E402
    from certus.data import gpu_decode  # noqa: E402
    from certus.evaluate import _loader  # noqa: E402
    cfg = Config()
    G = pd.read_csv(os.path.join(cfg.manifests, "grading.csv"), low_memory=False)
    model = WholeImage().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(os.path.join(bdir, "best.pt"), map_location="cuda"))
    model.eval()
    out = {}
    for part, split in zip(PARTS, ("val", "test", "external_test")):
        df = G[G.split == split].reset_index(drop=True)
        L = []
        with torch.no_grad():
            for byte_list, _ in _loader(df.proc_path.tolist(), 8):
                x = gpu_decode(byte_list, SIZE).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    L.append(model(x).float().cpu())
        out[f"{part}_logits"] = torch.cat(L).numpy()
        out[f"{part}_y"] = df.dr_grade.to_numpy()
        print(part, len(df), flush=True)
    np.savez(cache, **out)
    return out


def ref_logodds(logits, T=1.0):
    return logits[:, 1] / T                              # cumulative logit k=1: grade >= 2


def cut_at_spec(s_neg, spec):
    return float(np.quantile(s_neg, spec))


def band_fit(s_val, y_val, spec_clear, spec_refer):
    neg = s_val[~y_val]
    return cut_at_spec(neg, spec_clear), cut_at_spec(neg, spec_refer)


def band_eval(s, y, lo, hi):
    refer, clear = s > hi, s <= lo
    abstain = ~refer & ~clear
    return refer, clear, abstain


def summarise(y, refer, clear, abstain):
    return {"system_sens": float((refer | abstain)[y].mean()), "system_spec": float((clear | abstain)[~y].mean()),
            "abstain": float(abstain.mean()),
            "auto_refer_wrong": int((refer & ~y).sum()), "auto_clear_missed": int((clear & y).sum())}


def main():
    bdir = sys.argv[1]
    B = baseline_logits(bdir)
    C = dict(np.load(os.path.join(RUN, "predictions.npz"), allow_pickle=True))
    trust = json.load(open(os.path.join(RUN, "trust.json")))
    T, band = trust["temperature"], trust["screening_band"]
    for part in PARTS:
        assert (B[f"{part}_y"] == C[f"{part}_y"]).all(), f"{part}: eyes are not in the same order"

    y = {p: C[f"{p}_y"] >= 2 for p in PARTS}
    sc = {p: ref_logodds(C[f"{p}_logits"], T) for p in PARTS}      # Certus, calibrated
    sb = {p: ref_logodds(B[f"{p}_logits"]) for p in PARTS}         # baseline

    # Certus's deployed band, expressed as val specificities, so every method is held to the same rule
    lo_c = np.log(band["clear_below"] / (1 - band["clear_below"]))
    hi_c = np.log(band["refer_above"] / (1 - band["refer_above"]))
    spec_clear = float((sc["val"][~y["val"]] <= lo_c).mean())
    spec_refer = float((sc["val"][~y["val"]] <= hi_c).mean())

    # average: weight on val AUC
    ws = np.linspace(0, 1, 21)
    val_auc = [auc(y["val"], w * sc["val"] + (1 - w) * sb["val"]) for w in ws]
    w = float(ws[int(np.argmax(val_auc))])
    sa = {p: w * sc[p] + (1 - w) * sb[p] for p in PARTS}

    res = {"band_val_specificities": {"clear": spec_clear, "refer": spec_refer},
           "average_weight_on_certus": w, "methods": {}}
    bands = {"certus band": (sc, (lo_c, hi_c)),
             "baseline band": (sb, band_fit(sb["val"], y["val"], spec_clear, spec_refer)),
             "average": (sa, band_fit(sa["val"], y["val"], spec_clear, spec_refer))}
    for name, (s, (lo, hi)) in bands.items():
        res["methods"][name] = {p: {"auc": auc(y[p], s[p]), **summarise(y[p], *band_eval(s[p], y[p], lo, hi))}
                                for p in PARTS}

    lo_b, hi_b = bands["baseline band"][1]
    second = {}
    for p in PARTS:
        rc, cc, ac = band_eval(sc[p], y[p], lo_c, hi_c)
        rb, cb, _ = band_eval(sb[p], y[p], lo_b, hi_b)
        veto = (rc & cb) | (cc & rb)                     # the two sit on opposite sides of the band
        refer, clear = rc & ~veto, cc & ~veto
        second[p] = {"auc": None, "vetoed": int(veto.sum()),
                     **summarise(y[p], refer, clear, ac | veto)}
    # softer veto: the baseline only needs to be outside its own confident zone on the same side
    soft = {}
    for p in PARTS:
        rc, cc, ac = band_eval(sc[p], y[p], lo_c, hi_c)
        rb, cb, _ = band_eval(sb[p], y[p], lo_b, hi_b)
        veto = (rc & ~rb) | (cc & ~cb)
        soft[p] = {"auc": None, "vetoed": int(veto.sum()),
                   **summarise(y[p], rc & ~veto, cc & ~veto, ac | veto)}
    res["methods"]["second reader (veto on opposite side)"] = second
    res["methods"]["second reader (both must agree)"] = soft

    json.dump(res, open(os.path.join(bdir, "ensemble.json"), "w"), indent=1)
    print(f"weight on Certus in average: {w}   band val specificities {spec_clear:.3f} / {spec_refer:.3f}")
    print(f"{'method':42s} {'part':20s} {'AUC':>6s} {'sys sens':>9s} {'sys spec':>9s} {'abstain':>8s} {'FP ref':>7s} {'missed':>7s}")
    for name, r in res["methods"].items():
        for p in PARTS[1:]:
            m = r[p]
            a = f"{m['auc']:.3f}" if m["auc"] is not None else "   -"
            print(f"{name:42s} {p:20s} {a:>6s} {m['system_sens']:9.3f} {m['system_spec']:9.3f} {m['abstain']:8.3f} "
                  f"{m['auto_refer_wrong']:7d} {m['auto_clear_missed']:7d}")


if __name__ == "__main__":
    main()
