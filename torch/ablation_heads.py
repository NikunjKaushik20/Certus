"""Frozen-encoder ablation: which parts of the grade head earn their place?

Every variant gets the same features (ablation_features.py, from the shipped encoder), the same
ordinal loss (referable threshold weighted x2, as in training), the same grade- and
dataset-balanced sampling, and checkpoint selection on val AUC only. Three seeds each.
Test and Messidor-2 are scored at the threshold that gives 85% specificity on val.

Caveat, stated wherever these numbers are quoted: the encoder was trained with every component
in place, so a variant here is "this head on features learned for the full model", not a model
trained end to end without that component. The end-to-end whole-image baseline is
train_baseline.py.

Usage: python torch/ablation_heads.py            -> Data/ablation/ablation.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.evaluate import auc, qwk  # noqa: E402
from certus.model import GatedAttention, mlp  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "ablation")
SEEDS = (0, 1, 2)
EPOCHS, BATCH, LR, WD = 30, 256, 1e-3, 1e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = {
    "global view only (whole image, 512 px)": dict(glob=True),
    "lesion evidence only (12 numbers)": dict(ev=True),
    "tiles, mean-pooled": dict(tiles="mean"),
    "tiles, gated attention": dict(tiles="attn"),
    "tiles (attention) + global": dict(tiles="attn", glob=True),
    "tiles (attention) + global + evidence  [Certus]": dict(tiles="attn", glob=True, ev=True),
}


class Head(nn.Module):
    def __init__(self, E, tiles=None, glob=False, ev=False):
        super().__init__()
        self.tiles, self.glob, self.ev = tiles, glob, ev
        din = (E if tiles else 0) + (E if glob else 0) + (12 if ev else 0)
        self.attn = GatedAttention(E, 128) if tiles == "attn" else None
        self.head = mlp(din, 256, 4, 0.3)

    def forward(self, et, eg, ev):
        parts = []
        if self.tiles == "mean":
            parts.append(et.mean(1))
        elif self.tiles == "attn":
            a = torch.softmax(self.attn(et), 1)                  # (n, 9)
            parts.append((a.unsqueeze(-1) * et).sum(1))
        if self.glob:
            parts.append(eg)
        if self.ev:
            parts.append(ev)
        return self.head(torch.cat(parts, 1))


def load(name, ev_mu=None, ev_sd=None):
    Z = np.load(os.path.join(D, f"{name}.npz"), allow_pickle=True)
    ev = Z["evidence"].astype(np.float32)
    if ev_mu is None:
        ev_mu, ev_sd = ev.mean(0), ev.std(0) + 1e-6
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(DEV)
    return {"et": t(Z["emb_tiles"]), "eg": t(Z["emb_global"]), "ev": t((ev - ev_mu) / ev_sd),
            "y": Z["y"], "dataset": Z["dataset"], "domain": Z["domain"]}, ev_mu, ev_sd


def predict(model, S):
    model.eval()
    with torch.no_grad():
        out = [model(S["et"][i:i + 2048], S["eg"][i:i + 2048], S["ev"][i:i + 2048]).float()
               for i in range(0, len(S["y"]), 2048)]
    return torch.cummin(torch.sigmoid(torch.cat(out)), 1).values.cpu().numpy()


def sampler_weights(y, ds):
    w = np.ones(len(y))
    for v in (y, ds):
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        w *= cnt[inv] ** -0.5
    return torch.tensor(w / w.sum())


def train(spec, S, V, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    E = S["eg"].shape[1]
    model = Head(E, **spec).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * (len(S["y"]) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
    w = sampler_weights(S["y"], S["dataset"])
    yk = torch.tensor((S["y"][:, None] > np.arange(4)[None]).astype(np.float32), device=DEV)
    kw = torch.tensor([1.0, 2.0, 1.0, 1.0], device=DEV)
    best, best_state = -1, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        model.train()
        idx = torch.multinomial(w, len(S["y"]) // BATCH * BATCH, replacement=True, generator=g).view(-1, BATCH)
        for b in idx:
            b = b.to(DEV)
            logits = model(S["et"][b], S["eg"][b], S["ev"][b])
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, yk[b], reduction="none") * kw).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        a = auc(V["y"] >= 2, predict(model, V)[:, 1])
        if a > best:
            best, best_state = a, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def score(P, y, thr):
    r, yb = P[:, 1], y >= 2
    return {"auc": auc(yb, r), "qwk": qwk(y, (P > 0.5).sum(1)),
            "sens": float((r[yb] > thr).mean()), "spec": float((r[~yb] <= thr).mean())}


def main():
    S, mu, sd = load("train")
    V, _, _ = load("val", mu, sd)
    T, _, _ = load("test", mu, sd)
    X, _, _ = load("external_messidor2", mu, sd)
    results = {}
    for name, spec in VARIANTS.items():
        runs = []
        for seed in SEEDS:
            m = train(spec, S, V, seed)
            pv = predict(m, V)
            thr = float(np.quantile(pv[~(V["y"] >= 2), 1], 0.85, method="higher"))
            runs.append({"val_auc": auc(V["y"] >= 2, pv[:, 1]), "threshold": thr,
                         "test": score(predict(m, T), T["y"], thr),
                         "external_messidor2": score(predict(m, X), X["y"], thr)})
        agg = lambda f: (float(np.mean([f(r) for r in runs])), float(np.std([f(r) for r in runs])))
        results[name] = {"runs": runs, "summary": {
            f"{p}_{k}": agg(lambda r, p=p, k=k: r[p][k])
            for p in ("test", "external_messidor2") for k in ("auc", "qwk", "sens", "spec")}}
        s = results[name]["summary"]
        print(f"{name:50s} test AUC {s['test_auc'][0]:.3f}±{s['test_auc'][1]:.3f} "
              f"sens/spec {s['test_sens'][0]:.3f}/{s['test_spec'][0]:.3f}  | M2 AUC "
              f"{s['external_messidor2_auc'][0]:.3f}±{s['external_messidor2_auc'][1]:.3f} "
              f"sens/spec {s['external_messidor2_sens'][0]:.3f}/{s['external_messidor2_spec'][0]:.3f}", flush=True)
    json.dump({"seeds": SEEDS, "epochs": EPOCHS, "variants": results},
              open(os.path.join(D, "ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
