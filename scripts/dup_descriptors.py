"""Duplicate-photo detection that is actually discriminative for fundus images.

8x8 perceptual hashes collide across fundus photos (all are an orange disc on black),
so duplicates are found from the retinal texture instead:
  green channel of the FOV-normalised canvas -> 96x96 -> subtract a masked Gaussian
  background (keeps vessels / lesions) -> zero-mean, unit-norm inside the FOV.
All pairs are compared on the GPU (cosine similarity, also against the mirrored image).
The threshold is calibrated on known duplicates: DDR lesion-segmentation images that
also appear, under the same file name, in the DDR grading set.

Writes Data/features/dup_pairs.csv (a, b, sim) for sim >= 0.5 and dup_calibration.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, read_file

DATA = Path("D:/Certus/Data")
S = 96
DATASETS = ["APTOS2019", "DDR", "IDRiD", "Messidor2", "HRF"]


def blur(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


@torch.no_grad()
def descriptors(paths, bs=48):
    out = []
    for i in range(0, len(paths), bs):
        torch.cuda.synchronize()          # nvJPEG is not stream-ordered in torchvision 0.21: fence both sides
        ims = decode_jpeg([read_file(p) for p in paths[i:i + bs]], device="cuda")
        torch.cuda.synchronize()
        x = torch.stack([F.interpolate(im[None].float() / 255, size=(S, S), mode="area")[0] for im in ims])
        fov = (x.amax(1, keepdim=True) > 0.04).float()
        fov = -F.max_pool2d(-fov, 5, 1, 2)                       # erode: drop the FOV rim edge
        g = x[:, 1:2]
        bg = blur(g * fov, 4.0) / blur(fov, 4.0).clamp_min(1e-3)
        d = (g - bg) * fov
        n = fov.sum((1, 2, 3), keepdim=True).clamp_min(1)
        d = (d - (d.sum((1, 2, 3), keepdim=True) / n) * fov)
        d = d / d.flatten(1).norm(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        out.append(d.half())
    return torch.cat(out)


def main():
    parts = pd.concat([pd.read_csv(DATA / "features" / "parts" / f"{d}.csv", usecols=["uid", "status", "proc_path"])
                       for d in DATASETS if (DATA / "features" / "parts" / f"{d}.csv").exists()])
    parts = parts[parts.status == "ok"].reset_index(drop=True)
    print(f"{len(parts)} images")
    cache = DATA / "features" / "dup_desc.pt"
    if cache.exists() and torch.load(cache)["uids"] == parts.uid.tolist():
        D = torch.load(cache)["D"].cuda()
    else:
        D = descriptors(parts.proc_path.tolist())
        torch.save({"uids": parts.uid.tolist(), "D": D.cpu()}, cache)
    Dm = D.flip(-1)                                              # mirrored images
    A, Af = D.flatten(1), Dm.flatten(1)
    uids = parts.uid.to_numpy()
    rows = []
    for i in range(0, len(A), 2048):
        s = torch.maximum(A[i:i + 2048] @ A.T, A[i:i + 2048] @ Af.T).float()
        a, b = torch.nonzero(s >= 0.5, as_tuple=True)
        a = a + i
        keep = a < b
        for x, y, v in zip(a[keep].tolist(), b[keep].tolist(), s[a[keep] - i, b[keep]].tolist()):
            rows.append((uids[x], uids[y], v))
    pairs = pd.DataFrame(rows, columns=["a", "b", "sim"])
    pairs.to_csv(DATA / "features" / "dup_pairs.csv", index=False)

    # calibration: known duplicates vs random different photos
    pos = []
    idx = {u: k for k, u in enumerate(uids)}
    for u in uids:
        if u.startswith("DDR_S_") and "DDR_G_" + u[6:] in idx:
            pos.append(float(A[idx[u]] @ A[idx["DDR_G_" + u[6:]]]))
    rng = np.random.default_rng(0)
    ra, rb = rng.integers(len(A), size=200000), rng.integers(len(A), size=200000)
    neg = np.concatenate([(A[ra[k:k + 5000]].float() * A[rb[k:k + 5000]].float()).sum(1).cpu().numpy()
                          for k in range(0, len(ra), 5000)])
    # strongest match of each Messidor-2 image in DDR: these sets share no photos
    m = torch.tensor([u.startswith("M2_") for u in uids], device="cuda")
    ddr = torch.tensor([u.startswith("DDR_") for u in uids], device="cuda")
    cross = (A[m] @ A[ddr].T).float().amax(1).cpu().numpy()
    pos = np.array(pos)
    q = lambda x: {p: round(float(np.quantile(x, p)), 4) for p in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)}
    cal = {"n_known_dup": len(pos), "known_dup_sim_quantiles": q(pos),
           "random_pair_sim_quantiles": q(neg), "messidor_best_ddr_match_quantiles": q(cross)}
    thr = float(min(np.quantile(pos, 0.01), 0.5 * (np.quantile(pos, 0.01) + cross.max())))
    cal["threshold"] = round(max(thr, float(cross.max()) + 0.02), 4)
    cal["pairs_above_threshold"] = int((pairs.sim >= cal["threshold"]).sum())
    json.dump(cal, open(DATA / "features" / "dup_calibration.json", "w"), indent=2)
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
