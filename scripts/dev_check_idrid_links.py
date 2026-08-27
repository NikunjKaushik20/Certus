"""Link IDRiD segmentation images (A) to their graded twin (B) with the texture descriptors,
and report the link for the review-console sample (IDRiD_A_IDRiD_79)."""
import json
from pathlib import Path

import torch

DATA = Path("D:/Certus/Data")
c = torch.load(DATA / "features" / "dup_desc.pt")
thr = json.loads((DATA / "features" / "dup_calibration.json").read_text())["threshold"]
uids = c["uids"]
D = c["D"].float().flatten(1).cuda()
Dm = c["D"].float().flip(-1).flatten(1).cuda()
ia = [i for i, u in enumerate(uids) if u.startswith("IDRiD_A_")]
ib = [i for i, u in enumerate(uids) if u.startswith("IDRiD_B_")]
S = torch.maximum(D[ia] @ D[ib].T, D[ia] @ Dm[ib].T)
best, k = S.max(1)
second = S.topk(2, dim=1).values[:, 1]
linked = (best >= thr).sum().item()
print(f"threshold {thr}: {linked}/{len(ia)} IDRiD-A images have a graded twin")
print(f"best-match similarity: min {best.min():.3f}, median {best.median():.3f}; runner-up median {second.median():.3f}")
for j, i in enumerate(ia):
    if uids[i] == "IDRiD_A_IDRiD_79":
        print(f"IDRiD_A_IDRiD_79 -> {uids[ib[k[j]]]} (sim {best[j]:.3f}, runner-up {second[j]:.3f}, linked={best[j] >= thr})")
