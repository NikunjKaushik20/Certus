"""One-at-a-time sensitivity of the minimum ophthalmologist count to the assumptions
marked in parameters.md. Writes sim/sweep_results.csv."""
import csv
from dataclasses import replace

from capacity import Params, min_reviewers

SWEEPS = {
    "review_certus_s": [20, 30, 45, 60],
    "review_manual_s": [45, 90, 135, 180],
    "referable_prev": [0.04, 0.08, 0.125],
    "ungradable_raw": [0.04, 0.10, 0.225],
    "abstain": [0.05, 0.10, 0.20],
    "spec": [0.80, 0.85, 0.90],
    "reviewer_hours_per_day": [2, 4, 6],
    "annual_patients": [50_000, 100_000, 200_000],
}


def main():
    base = Params()
    with open("D:/Certus/sim/sweep_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        # "1" is a floor for Certus, so utilisation at the minimum staffing is reported too
        w.writerow(["parameter", "value", "min_manual", "util_manual", "min_certus", "util_certus"])
        for k, vals in SWEEPS.items():
            for v in vals:
                p = replace(base, **{k: v})
                row = [k, v]
                for policy, skip in (("manual", k.startswith(("review_certus", "abstain", "spec"))),
                                     ("certus", k == "review_manual_s")):
                    if skip:
                        row += ["", ""]
                        continue
                    n, table = min_reviewers(p, policy, reps=2)
                    row += [n if n is not None else "unreachable",
                            round(table[n]["reviewer_utilisation"], 3) if n is not None else ""]
                w.writerow(row)
                fh.flush()
                print(*row, flush=True)


if __name__ == "__main__":
    main()
