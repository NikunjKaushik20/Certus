"""Threshold-free lesion segmentation report for a checkpoint.

seg_metrics scores Dice at a fixed 0.5 cut-off, which reads 0.0 for microaneurysms
long after the model has started to find them (a few-pixel lesion rarely exceeds 0.5).
Here tp/fp/fn are accumulated per image from 256-bin histograms, so Dice at every
threshold and average precision (threshold-free) come out of one pass.

Usage: python dev_ma_check.py runs_torch/train_<stamp>/last.pt
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import load_model  # noqa: E402
from certus.data import load_manifests  # noqa: E402
from certus.evaluate import predict_eyes  # noqa: E402

B = 256


def main():
    ck = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "seg_val"
    model, cfg = load_model(ck)
    M = load_manifests(cfg)
    df = M[split].rename(columns={"path": "proc_path"})
    L = cfg.n_lesion
    res = predict_eyes(model, df, cfg, bs=1, with_maps=True)

    hp = torch.zeros(L, B, device="cuda")            # histogram of predicted prob over lesion pixels
    ha = torch.zeros(L, B, device="cuda")            # ... over all pixels
    seen = np.zeros(L, int)
    for i, masks in enumerate(df.masks):
        P = res["maps"][i].float().cuda()
        for c in range(L):
            if masks[c] == "":
                continue
            seen[c] += 1
            Y = torch.from_numpy(cv2.imread(masks[c], cv2.IMREAD_GRAYSCALE) > 0).cuda()
            p = P[c]
            ha[c] += torch.histc(p, B, 0, 1)
            if Y.any():
                hp[c] += torch.histc(p[Y], B, 0, 1)

    edges = np.linspace(0, 1, B + 1)[:-1]
    print(f"checkpoint {ck} | {split} ({len(df)} images)\n")
    for c in range(L):
        if seen[c] == 0:
            continue
        pos = hp[c].flip(0).cumsum(0).flip(0).cpu().numpy()          # tp at each threshold
        allp = ha[c].flip(0).cumsum(0).flip(0).cpu().numpy()         # tp + fp
        npos = pos[0]
        if npos == 0:
            print(f"{cfg.seg_classes[c]:4s}: no positive pixels in this split")
            continue
        tp, fp, fn = pos, allp - pos, npos - pos
        dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        prec = tp / np.maximum(allp, 1)
        rec = tp / npos
        ap = float(np.sum(np.diff(np.concatenate([[0.0], rec[::-1]])) * prec[::-1]))
        k = int(np.argmax(dice))
        k50 = B // 2
        print(f"{cfg.seg_classes[c]:4s}: AP {ap:.4f} | best dice {dice[k]:.4f} @ thr {edges[k]:.3f} "
              f"(prec {prec[k]:.3f} rec {rec[k]:.3f}) | dice@0.5 {dice[k50]:.4f} | "
              f"peak prob reached {edges[int(np.max(np.nonzero(allp > 0)))]:.3f} | "
              f"{seen[c]} images, {int(npos)} lesion pixels")


if __name__ == "__main__":
    main()
