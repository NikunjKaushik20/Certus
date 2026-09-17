"""Score the reader study from the API database.

Reads study/cases.csv (reference grades), study/loaded.csv (case -> encounter), and from
api/certus.db the reviews and the review.explanation_rated audit events. Optionally reads
study/unassisted.csv (arm A, image only: case, reviewer, seconds, grade) for the comparison.

Reports, per arm: median and 90th-percentile seconds per case, the share under 30 s, referral
accuracy against the reference grade (sensitivity / specificity for grade >= 2), and for the
assisted arm the distribution of Grad-CAM and lesion-evidence usefulness ratings.

Usage: python study/analyse.py [path/to/certus.db]   -> study/results.json
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def summarise(df, grade_col):
    y = df.dr_grade >= 2
    g = df[grade_col] >= 2
    s = df.seconds.astype(float)
    return {"reads": int(len(df)), "median_seconds": float(s.median()), "p90_seconds": float(s.quantile(0.9)),
            "under_30s": round(float((s <= 30).mean()), 4),
            "referral_sensitivity": round(float((g & y).sum() / max(y.sum(), 1)), 4),
            "referral_specificity": round(float((~g & ~y).sum() / max((~y).sum(), 1)), 4),
            "exact_grade_agreement": round(float((df[grade_col] == df.dr_grade).mean()), 4)}


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "api", "certus.db")
    C = pd.read_csv(os.path.join(ROOT, "study", "cases.csv"))
    L = pd.read_csv(os.path.join(ROOT, "study", "loaded.csv"))
    con = sqlite3.connect(db)
    R = pd.read_sql("""select r.id as referral_id, r.encounter_id, r.worst_grade, v.reviewer_id, v.agreed,
                              v.final_grade, v.seconds_spent as seconds
                       from review v join referral r on r.id = v.referral_id""", con)
    A = pd.read_sql("select entity_id as referral_id, payload from audit_log where action = 'review.explanation_rated'", con)
    D = C.merge(L, on="case").merge(R, on="encounter_id")
    D["read_grade"] = np.where(D.agreed.fillna(False).astype(bool), D.worst_grade, D.final_grade)
    out = {"assisted": summarise(D.dropna(subset=["read_grade", "seconds"]), "read_grade")}

    if len(A):
        P = pd.json_normalize(A.payload.map(json.loads))
        out["ratings"] = {
            "n": int(len(P)),
            "gradcam_useful": P.gradcam_useful.value_counts().sort_index().to_dict() if "gradcam_useful" in P else {},
            "gradcam_useful_4_or_5": round(float((P.gradcam_useful >= 4).mean()), 4) if "gradcam_useful" in P else None,
            "lesions_useful": P.lesions_useful.value_counts().sort_index().to_dict() if "lesions_useful" in P else {},
            "heatmap_on_lesions": P.heatmap_on_lesions.value_counts().to_dict() if "heatmap_on_lesions" in P else {}}

    un = os.path.join(ROOT, "study", "unassisted.csv")
    if os.path.exists(un):
        U = pd.read_csv(un).merge(C, on="case")
        out["unassisted"] = summarise(U, "grade")

    json.dump(out, open(os.path.join(ROOT, "study", "results.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
