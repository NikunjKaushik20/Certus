"""The single-technique deep-learning baseline: one CNN on the whole photograph.

EfficientNet-B0 from ImageNet weights, the whole FOV canvas resized to 512 px, the same ordinal
loss (referable threshold x2), the same grade- and dataset-balanced sampling, the same training
eyes (including the EyePACS auxiliary set) and the same GPU augmentation as Certus. No tiles, no
attention, no lesion segmentation, no evidence vector, no quality head. This is what most
published DR graders are, so it is the comparison the integrated pipeline has to beat.

Checkpoint selection on val AUC only. Test and Messidor-2 scored at the threshold giving 85%
specificity on val.

Usage: python torch/train_baseline.py [epochs]    -> runs_torch/baseline_wholeimage_<stamp>/
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.augment import augment  # noqa: E402
from certus.config import Config  # noqa: E402
from certus.data import BytesDataset, collate_bytes, gpu_decode  # noqa: E402
from certus.evaluate import _loader, auc, qwk  # noqa: E402
from certus.losses import ordinal  # noqa: E402
from certus.model import WholeImageGrader as WholeImage  # noqa: E402

SIZE, BATCH = 512, 16


@torch.no_grad()
def predict(model, df, cfg):
    model.eval()
    P = []
    for byte_list, _ in _loader(df.proc_path.tolist(), 8):
        x = gpu_decode(byte_list, SIZE).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(torch.sigmoid(model(x).float()).cpu())
    return torch.cummin(torch.cat(P), 1).values.numpy()


def score(P, y, thr):
    r, yb = P[:, 1], y >= 2
    return {"auc": auc(yb, r), "qwk": qwk(y, (P > 0.5).sum(1)),
            "sens": float((r[yb] > thr).mean()), "spec": float((r[~yb] <= thr).mean())}


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cfg = Config()
    torch.manual_seed(cfg.seed)
    G = pd.read_csv(os.path.join(cfg.manifests, "grading.csv"), low_memory=False)
    A = pd.read_csv(os.path.join(cfg.manifests, "grading_aux.csv"), low_memory=False)
    tr = pd.concat([G[G.split == "train"], A[A.split == "train"]], ignore_index=True)
    va, te, ex = (G[G.split == s].reset_index(drop=True) for s in ("val", "test", "external_test"))

    w = np.ones(len(tr))
    for col, p in (("dr_grade", cfg.grade_power), ("dataset", cfg.dataset_power)):
        _, inv, cnt = np.unique(tr[col].astype(str), return_inverse=True, return_counts=True)
        w *= cnt[inv] ** -p
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w), len(tr), replacement=True)
    loader = torch.utils.data.DataLoader(BytesDataset(tr.proc_path.tolist(), tr.dr_grade.to_numpy()),
                                         batch_size=BATCH, sampler=sampler, num_workers=3,
                                         collate_fn=collate_bytes, drop_last=True, persistent_workers=True)

    model = WholeImage().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.03)
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed)

    out = os.path.join(cfg.runs, "baseline_wholeimage_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    best, log, t0 = -1.0, [], time.time()
    for ep in range(epochs):
        model.train()
        for k, (byte_list, y) in enumerate(loader):
            x = gpu_decode(byte_list, SIZE)
            y = torch.as_tensor(y).cuda()
            x, _, _ = augment(x, None, cfg, "full", gen)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x)
            loss = ordinal(z, y, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            if k % 200 == 0:
                print(f"ep {ep} step {k}/{len(loader)} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        a = auc(va.dr_grade.to_numpy() >= 2, predict(model, va, cfg)[:, 1])
        log.append({"epoch": ep, "val_auc": a, "sec": time.time() - t0})
        print(f"epoch {ep}: val AUC {a:.4f}", flush=True)
        if a > best:
            best = a
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(out, "best.pt")))
    yv = va.dr_grade.to_numpy()
    pv = predict(model, va, cfg)
    thr = float(np.quantile(pv[yv < 2, 1], 0.85, method="higher"))
    res = {"model": "EfficientNet-B0, whole image 512 px, end to end", "epochs": epochs, "log": log,
           "val_auc": best, "threshold": thr,
           "test": score(predict(model, te, cfg), te.dr_grade.to_numpy(), thr),
           "external_messidor2": score(predict(model, ex, cfg), ex.dr_grade.to_numpy(), thr)}
    json.dump(res, open(os.path.join(out, "baseline.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("val_auc", "threshold", "test", "external_messidor2")}, indent=1))


if __name__ == "__main__":
    main()
