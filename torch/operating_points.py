"""Answer operating-point questions from saved predictions, without touching the GPU.

Screening programmes are specified by the question "what sensitivity can we promise, and what
does it cost in unnecessary referrals?". That is a property of the score distribution, not of the
model weights, so it should never require a 45-minute re-run.

Usage: python operating_points.py runs_torch/train_<stamp>/predictions.npz [--temperature T]
"""
import json
import os
import sys

import numpy as np

TARGETS = (0.90, 0.925, 0.95)


def cummin_sigmoid(logits, T=1.0):
    return np.minimum.accumulate(1 / (1 + np.exp(-logits / T)), axis=1)


def curve(y, s):
    """Sensitivity and specificity at every distinct threshold, sorted by threshold."""
    order = np.argsort(-s)
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    return s, tp / max(y.sum(), 1), 1 - fp / max((~y).sum(), 1)


def report(name, y_ref, p_ref):
    thr, sens, spec = curve(y_ref, p_ref)
    auc = float(np.trapezoid(np.r_[0, sens], np.r_[0, 1 - spec])) if hasattr(np, "trapezoid") else \
        float(np.trapz(np.r_[0, sens], np.r_[0, 1 - spec]))
    print(f"\n== {name}: n={len(y_ref)}, referable={y_ref.mean():.1%}, AUC={auc:.4f}")
    print(f"{'target sens':>12} | {'threshold':>9} | {'sens':>6} | {'spec':>6} | "
          f"{'referred':>8} | per 1000 screened")
    for t in TARGETS:
        i = int(np.argmax(sens >= t)) if (sens >= t).any() else len(sens) - 1
        referred = float((p_ref >= thr[i]).mean())
        missed = (1 - sens[i]) * y_ref.sum()
        print(f"{t:>12.1%} | {thr[i]:>9.4f} | {sens[i]:>6.3f} | {spec[i]:>6.3f} | "
              f"{referred:>8.1%} | {referred * 1000:5.0f} referred, "
              f"{missed / len(y_ref) * 1000:4.1f} referable eyes missed")
    # the point a screening service usually quotes
    i = int(np.argmax(spec <= 0.85))
    print(f"{'at 85% spec':>12} | {thr[i]:>9.4f} | {sens[i]:>6.3f} | {spec[i]:>6.3f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        runs = sorted(__import__("glob").glob("D:/Certus/runs_torch/train_*/predictions.npz"),
                      key=os.path.getmtime)
        path = runs[-1] if runs else sys.exit("no predictions.npz found; run calibrate.py first")
    T = 1.0
    if "--temperature" in sys.argv:
        T = float(sys.argv[sys.argv.index("--temperature") + 1])
    trust_path = os.path.join(os.path.dirname(path), "trust.json")
    if T == 1.0 and os.path.exists(trust_path):
        T = json.load(open(trust_path)).get("temperature", 1.0)

    d = np.load(path, allow_pickle=True)
    print(f"{path}  (temperature {T:.4f})")
    for part in ("test", "external_messidor2", "val"):
        if f"{part}_logits" not in d:
            continue
        y, logits, dom = d[f"{part}_y"], d[f"{part}_logits"], d[f"{part}_domain"]
        p = cummin_sigmoid(logits, T)[:, 1]
        report(part, y >= 2, p)
        for dm in np.unique(dom):
            m = dom == dm
            if m.sum() >= 200:
                report(f"{part} / {dm}", (y[m] >= 2), p[m])


if __name__ == "__main__":
    main()
