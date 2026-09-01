"""Per-eye features from the shipped encoder, for the frozen-encoder ablation (ablation_heads.py).

For every graded eye: the nine tile embeddings, the global-view embedding, and the 12-number
lesion evidence vector, exactly as forward_eye computes them. No TTA. Saved in fp16.

Usage: python torch/ablation_features.py runs_torch/train_<stamp>/best.pt
Writes Data/ablation/<split>.npz for train, val, test, external_messidor2.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import load_model  # noqa: E402
from certus.data import gpu_decode  # noqa: E402
from certus.evaluate import _loader  # noqa: E402
from certus.model import evidence, make_tiles, stitch_tiles  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "ablation")


@torch.no_grad()
def features(model, df, cfg, bs=2):
    g, t = cfg.grid, cfg.tile
    ET, EG, EV = [], [], []
    t0 = time.time()
    for k, (byte_list, _) in enumerate(_loader(df.proc_path.tolist(), bs)):
        x = gpu_decode(byte_list, cfg.canvas).contiguous(memory_format=torch.channels_last)
        n = x.shape[0]
        nt = n * g * g
        with torch.autocast("cuda", dtype=getattr(torch, cfg.amp_dtype)):
            tiles = make_tiles(x, g, t)
            glob = F.interpolate(x, size=(cfg.global_size,) * 2, mode="bilinear", antialias=True, align_corners=False)
            feats = model.encode(torch.cat([tiles, glob]))
            emb = model.embed(feats).float()
            seg = model.decode({s: f[:nt] for s, f in feats.items()})
        fov_t = (tiles.amax(1, keepdim=True) > 0.02).float()
        P = torch.sigmoid(seg[:, :cfg.n_lesion].float()) * fov_t
        ev = evidence(stitch_tiles(P, g, t))
        ET.append(emb[:nt].view(n, g * g, -1).half().cpu())
        EG.append(emb[nt:].half().cpu())
        EV.append(ev.float().cpu())
        if k % 500 == 0:
            print(f"  {k * bs}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    return torch.cat(ET).numpy(), torch.cat(EG).numpy(), torch.cat(EV).numpy()


def main():
    model, cfg, step = load_model(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    G = pd.read_csv(os.path.join(cfg.manifests, "grading.csv"), low_memory=False)
    A = pd.read_csv(os.path.join(cfg.manifests, "grading_aux.csv"), low_memory=False)
    parts = {"train": pd.concat([G[G.split == "train"], A[A.split == "train"]], ignore_index=True),
             "val": G[G.split == "val"], "test": G[G.split == "test"],
             "external_messidor2": G[G.split == "external_test"]}
    for name in ("val", "test", "external_messidor2", "train"):
        path = os.path.join(OUT, f"{name}.npz")
        if os.path.exists(path):
            print(f"{name}: exists, skipping")
            continue
        df = parts[name].reset_index(drop=True)
        print(f"{name}: {len(df)} eyes", flush=True)
        et, eg, ev = features(model, df, cfg)
        np.savez(path, emb_tiles=et, emb_global=eg, evidence=ev, y=df.dr_grade.to_numpy().astype(int),
                 dataset=df.dataset.astype(str).to_numpy(), domain=df.domain.astype(str).to_numpy(),
                 group=df.group_id.astype(str).to_numpy(), uid=df.uid.astype(str).to_numpy())
        print(f"{name}: saved", flush=True)


if __name__ == "__main__":
    main()
