"""Fit the screening band for a new camera from its first labelled patients.

A camera the model has never seen shifts the scores: on Messidor-2 the val-fitted band kept
sensitivity at 99.6% but specificity fell to 72.4%, and 26% of eyes went to a person.
new_camera.py showed that re-fitting both cuts on about 100 labelled patients from that camera,
keeping the specificity each cut had on val, brings it back to 93.1% / 91.6% with 12.9% abstained.
This is the tool that does that fit for a real camera.

Input: a CSV of photographs from the new camera, graded by an ophthalmologist (in practice, the
camera's first ~100 patients, all sent to full review while the camera is unverified):

    path,grade
    /photos/cam7/0001_R.jpg,0
    /photos/cam7/0001_L.jpg,2

Each photograph is scored by the same engine the API runs. The fitted band is written into
trust.json under per_domain[<domain_key>].screening_band; the API then decides eyes from any device
with that domain_key by it (Engine.decide). When a second reader is deployed, its band is re-fitted
the same way (per_domain[<domain_key>].second_reader_band): second_reader.py showed that re-fitting
both readers on 100 Messidor-2 patients gives system 97.4% / 96.5% with 27% of eyes to a human.
The previous trust.json is kept as trust.json.bak.

Usage: python torch/fit_camera.py <domain_key> <labels.csv> [--dry-run]
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "api"), os.path.dirname(os.path.abspath(__file__))]
from new_camera import p_ref, thr_for_spec  # noqa: E402

MIN_NEG, MIN_POS = 50, 10


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    domain, labels = args[0], args[1]
    L = pd.read_csv(labels)
    from certus_api.config import settings
    from certus_api.inference import Engine

    eng = Engine()
    trust_path = settings.resolve_trust()
    t = json.load(open(trust_path))
    scores, second = [], []
    for path in L.path:
        pre = eng.preprocess(open(path, "rb").read())
        res = eng.analyse(pre["canvas"]) if pre["ok"] else None
        scores.append(res["p_referable"] if res else np.nan)
        second.append(res["second_reader"]["logit"] if res and res.get("second_reader") else np.nan)
    L["p"], L["s2"] = scores, second
    L = L.dropna(subset=["p"])
    y = L.grade.to_numpy() >= 2
    p = L.p.to_numpy()
    if (~y).sum() < MIN_NEG or y.sum() < MIN_POS:
        sys.exit(f"need at least {MIN_NEG} non-referable and {MIN_POS} referable graded photographs; "
                 f"have {(~y).sum()} and {y.sum()}")

    # The specificity each cut had on val, exactly as new_camera.py validated it.
    Z = np.load(os.path.join(os.path.dirname(trust_path), "predictions.npz"), allow_pickle=True)
    pv, yv = p_ref(Z["val_logits"], t["temperature"]), Z["val_y"] >= 2
    gb = t["screening_band"]
    clear_spec = float((pv[~yv] <= gb["clear_below"]).mean())
    refer_spec = float((pv[~yv] <= gb["refer_above"]).mean())
    band = {"clear_below": thr_for_spec(p[~y], clear_spec), "refer_above": thr_for_spec(p[~y], refer_spec),
            "fitted_on": f"{len(L)} graded photographs from {os.path.basename(labels)}",
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "spec_levels_from_val": {"clear": round(clear_spec, 4), "refer": round(refer_spec, 4)}}

    def show(name, c, r):
        auto_neg, auto_pos = p <= c, p > r
        mid = ~auto_neg & ~auto_pos
        print(f"  {name:12s} clear <= {c:.4f}  refer > {r:.4f}   on these photographs: "
              f"auto-cleared referable {int((auto_neg & y).sum())}/{int(y.sum())}, "
              f"abstain {mid.mean():.1%}, auto-referred non-referable {int((auto_pos & ~y).sum())}/{int((~y).sum())}")
    print(f"{domain}: {len(L)} photographs, {int(y.sum())} referable")
    show("global band", gb["clear_below"], gb["refer_above"])
    show("fitted band", band["clear_below"], band["refer_above"])
    print("  (in-sample: the held-out estimate for this procedure is in new_camera.json)")
    band2 = None
    if t.get("second_reader"):
        s2 = L.s2.to_numpy()
        band2 = {"clear_below_logit": thr_for_spec(s2[~y], clear_spec),
                 "refer_above_logit": thr_for_spec(s2[~y], refer_spec),
                 "fitted_on": band["fitted_on"], "fitted_at": band["fitted_at"]}
        clear2 = (p <= band["clear_below"]) & (s2 <= band2["clear_below_logit"])
        refer2 = (p > band["refer_above"]) & (s2 > band2["refer_above_logit"])
        print(f"  two readers  on these photographs: auto-cleared referable {int((clear2 & y).sum())}/{int(y.sum())}, "
              f"to a human {(~clear2 & ~refer2).mean():.1%}, auto-referred non-referable "
              f"{int((refer2 & ~y).sum())}/{int((~y).sum())}")
    if dry:
        return
    shutil.copy(trust_path, trust_path + ".bak")
    t.setdefault("per_domain", {}).setdefault(domain, {})["screening_band"] = band
    if band2:
        t["per_domain"][domain]["second_reader_band"] = band2
    json.dump(t, open(trust_path, "w"), indent=1)
    print(f"-> {trust_path} (previous kept as trust.json.bak). Restart the API to load it.")


if __name__ == "__main__":
    main()
