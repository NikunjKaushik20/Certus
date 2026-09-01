"""Fit the trust layer on a trained checkpoint and report it on test + Messidor-2.

Val is split by patient group into two halves: A fits the temperature, B calibrates the
conformal thresholds (keeps the conformal guarantee honest).

Calibration is fitted *per camera domain* as well as globally. One temperature cannot serve
every camera: on the epoch-3 model the same correction left calibration error at 0.149 on the
internal test set and 0.031 on Messidor-2. A domain seen during calibration gets its own
temperature and its own conformal band; an unseen camera falls back to the global fit and is
reported as unverified rather than silently trusted.

Per-eye logits are written to predictions.npz so operating points can be re-examined without
re-running inference (~45 minutes).

Usage: python calibrate.py runs_torch/train_<stamp>/best.pt [--no-tta]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus import trust  # noqa: E402
from certus.config import Config  # noqa: E402
from certus.data import load_manifests  # noqa: E402
from certus.evaluate import grading_metrics, predict_eyes, seg_metrics  # noqa: E402
from certus.model import CertusNet  # noqa: E402

MIN_DOMAIN_N = 150            # below this a per-domain fit is noise; fall back to the global one
ALPHAS = (0.05, 0.10)


def load_model(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(ck["cfg"]).items()})
    m = CertusNet(cfg, pretrained=False).cuda().to(memory_format=torch.channels_last)
    m.load_state_dict(ck["ema"])
    return m.eval(), cfg, int(ck.get("step", 0))


def domains_of(df):
    return df.domain.astype(str).to_numpy() if "domain" in df else df.dataset.astype(str).to_numpy()


def main():
    ck_path = sys.argv[1]
    tta = "--no-tta" not in sys.argv
    model, cfg, step = load_model(ck_path)
    M = load_manifests(cfg)
    out_dir = os.path.dirname(ck_path)

    # ---------------------------------------------------------------- predictions (once)
    parts = {"val": M["grade_val"], "test": M["grade_test"], "external_messidor2": M["grade_external_test"]}
    pred, saved = {}, {}
    for name, df in parts.items():
        r = predict_eyes(model, df, cfg, tta=tta)
        y = df.dr_grade.to_numpy().astype(int)
        dom = domains_of(df)
        pred[name] = {"logits": r["logits"], "y": y, "domain": dom, "dataset": df.dataset.to_numpy()}
        saved[f"{name}_logits"] = r["logits"]
        saved[f"{name}_y"] = y
        saved[f"{name}_domain"] = dom
        print(f"predicted {name}: {len(y)} eyes (tta={tta})", flush=True)
    np.savez_compressed(os.path.join(out_dir, "predictions.npz"), **saved)

    # ---------------------------------------------------------------- fit on val only
    V = parts["val"]
    groups = V.group_id.unique()
    rng = np.random.default_rng(0)
    A_groups = set(rng.permutation(groups)[: len(groups) // 2])
    mA = V.group_id.isin(A_groups).to_numpy()
    lv, yv, dv = pred["val"]["logits"], pred["val"]["y"], pred["val"]["domain"]

    T = trust.fit_temperature(lv[mA], yv[mA], cfg.num_grades)
    Pv = trust.calibrated_probs(lv, T)
    out = {"checkpoint": ck_path, "step": step, "tta": tta, "temperature": T, "alphas": {}, "per_domain": {}}
    for alpha in ALPHAS:
        out["alphas"][str(alpha)] = {"q": trust.conformal_fit(Pv[~mA, 1], yv[~mA] >= 2, alpha)}

    # per-domain temperature + conformal band, fitted on the same val halves
    for d in np.unique(dv):
        m = dv == d
        if m.sum() < MIN_DOMAIN_N:
            continue
        Td = trust.fit_temperature(lv[m & mA], yv[m & mA], cfg.num_grades)
        Pd = trust.calibrated_probs(lv[m], Td)
        thr = float(np.quantile(Pd[yv[m] < 2, 1], 0.85))          # the domain's own 85%-specificity point
        entry = {"temperature": Td, "referral_threshold": thr, "n_val": int(m.sum()), "alphas": {}}
        mb = (dv == d) & ~mA
        if mb.sum() >= MIN_DOMAIN_N // 2:
            Pb = trust.calibrated_probs(lv[mb], Td)
            for alpha in ALPHAS:
                entry["alphas"][str(alpha)] = {"q": trust.conformal_fit(Pb[:, 1], yv[mb] >= 2, alpha)}
        out["per_domain"][str(d)] = entry
    print("per-domain temperatures:",
          {d: round(v["temperature"], 3) for d, v in out["per_domain"].items()}, flush=True)

    # ---------------------------------------------------------------- lesion thresholds
    S = seg_metrics(model, M["seg_val"], cfg, tta=tta)
    thr = {c: S[c + "_thr"] for c in cfg.seg_classes[:cfg.n_lesion] if c + "_thr" in S}
    out["lesion"] = {"thresholds": thr, "val": S,
                     "test": seg_metrics(model, M["seg_test"], cfg, thr, tta=tta)}
    print("lesion thresholds", thr, flush=True)
    print("lesion test dice at those thresholds",
          {c: out["lesion"]["test"].get(c + "_at_fixed") for c in thr}, flush=True)

    # ---------------------------------------------------------------- report on held-out data
    for part in ("test", "external_messidor2"):
        p = pred[part]
        y, dom, logits = p["y"], p["domain"], p["logits"]
        P = trust.calibrated_probs(logits, T)
        res = {"raw": grading_metrics(y, trust.calibrated_probs(logits, 1.0), p["dataset"], cfg.num_grades),
               "calibrated": grading_metrics(y, P, p["dataset"], cfg.num_grades)}
        base_thr = res["calibrated"]["thr_at_85"]

        # per-domain calibration applied where the domain was seen during fitting
        Pdom = P.copy()
        used, decision = {}, np.full(len(y), "refer-to-human", dtype=object)
        for d in np.unique(dom):
            m = dom == d
            cal = out["per_domain"].get(str(d))
            Pdom[m] = trust.calibrated_probs(logits[m], cal["temperature"] if cal else T)
            used[str(d)] = "per-domain" if cal else "global (unverified camera)"
            q = (cal or {}).get("alphas", {}).get("0.05") or out["alphas"]["0.05"]
            decision[m] = trust.conformal_predict(Pdom[m, 1], q["q"])
        res["domain_calibrated"] = grading_metrics(y, Pdom, p["dataset"], cfg.num_grades)
        res["domain_calibration_used"] = used
        res["domain_conformal_alpha_0.05"] = trust.decision_metrics(decision, y >= 2)

        for alpha, v in out["alphas"].items():
            dec = trust.conformal_predict(P[:, 1], v["q"])
            res[f"conformal_alpha_{alpha}"] = trust.decision_metrics(dec, y >= 2)
            res[f"domain_table_alpha_{alpha}"] = trust.domain_table(dom, y, P[:, 1], dec, base_thr)
        out[part] = res

        c, dc = res["calibrated"], res["domain_calibrated"]
        print(f"{part}: AUC {c['auc_referable']:.4f} | sens@85spec {c['sens_at_85spec']:.4f} | "
              f"ECE global {c['ece_referable']:.3f} -> per-domain {dc['ece_referable']:.3f} | "
              f"conformal(global) abstain {res['conformal_alpha_0.05']['abstain_rate']:.2%} "
              f"sys-sens {res['conformal_alpha_0.05']['system_sensitivity']:.3f} | "
              f"conformal(per-domain) abstain {res['domain_conformal_alpha_0.05']['abstain_rate']:.2%} "
              f"sys-sens {res['domain_conformal_alpha_0.05']['system_sensitivity']:.3f}", flush=True)

    path = os.path.join(out_dir, "trust.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("->", path)
    print("-> predictions.npz (re-run operating_points.py without touching the GPU)")


if __name__ == "__main__":
    main()
