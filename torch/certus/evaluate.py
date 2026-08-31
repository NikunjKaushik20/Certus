"""Validation / test metrics (mirrors matlab/eval). Inference runs on the GPU under autocast."""
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.utils.data import DataLoader

from .data import BytesDataset, collate_bytes, gpu_decode


# ---------------------------------------------------------------- metrics
def auc(y, s):
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def at_specificity(y, s, target=0.85):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    thr = np.quantile(s[~y], target)
    return float((s[y] > thr).mean()), float((s[~y] <= thr).mean()), float(thr)


def at_threshold(y, s, thr):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    return float((s[y] > thr).mean()), float((s[~y] <= thr).mean())


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((idx == k).mean() * abs(y[idx == k].mean() - p[idx == k].mean())
                     for k in range(bins) if (idx == k).any()))


def qwk(a, b, K=5):
    return float(cohen_kappa_score(a, b, weights="quadratic", labels=list(range(K))))


# ---------------------------------------------------------------- evaluation
def _subsample(df, n):
    return df.sample(n, random_state=0).sort_index() if len(df) > n else df


def _loader(paths, bs, workers=2):
    return DataLoader(BytesDataset(paths, np.zeros(len(paths))), batch_size=bs, num_workers=workers,
                      collate_fn=collate_bytes)


@torch.no_grad()
def predict_eyes(model, df, cfg, bs=2, with_maps=False, tta=False):
    """tta: average the logits of the image and its mirror. A fundus photo is the same eye either
    way round, so this is a free variance reduction; attention and evidence stay from the upright
    view, whose tile order matches what the report displays."""
    model.eval()
    P, Z, Q, A, EV, maps = [], [], [], [], [], []
    for byte_list, _ in _loader(df.proc_path.tolist(), bs):
        x = gpu_decode(byte_list, cfg.canvas).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=getattr(torch, cfg.amp_dtype)):
            out = model.forward_eye(x, with_maps=with_maps)
            if tta:
                flipped = model.forward_eye(x.flip(-1).contiguous(memory_format=torch.channels_last),
                                            with_maps=with_maps)
                out["grade_logits"] = (out["grade_logits"].float() + flipped["grade_logits"].float()) / 2
                out["qual_logits"] = (out["qual_logits"].float() + flipped["qual_logits"].float()) / 2
                if with_maps:
                    out["lesion_prob"] = (out["lesion_prob"].float()
                                          + flipped["lesion_prob"].float().flip(-1)) / 2
        Z.append(out["grade_logits"].float().cpu())
        P.append(torch.sigmoid(out["grade_logits"].float()).cpu())
        Q.append(torch.softmax(out["qual_logits"].float(), 1).cpu())
        A.append(out["attn"].cpu()); EV.append(out["evidence"].cpu())
        if with_maps:
            maps.append(out["lesion_prob"].half().cpu())
    P = torch.cummin(torch.cat(P), 1).values.numpy()     # enforce P(>0) >= P(>1) >= ...
    res = {"P": P, "logits": torch.cat(Z).numpy(), "Q": torch.cat(Q).numpy(), "attn": torch.cat(A).numpy(),
           "evidence": torch.cat(EV).numpy()}
    if with_maps:
        res["maps"] = torch.cat(maps)
    return res


def grading_metrics(y, P, datasets=None, K=5):
    y = np.asarray(y).astype(int)
    ref = P[:, 1]
    pred = (P > 0.5).sum(1)
    R = {"n": len(y), "auc_referable": auc(y >= 2, ref), "qwk": qwk(y, pred, K), "accuracy": float((pred == y).mean()),
         "ece_referable": ece(y >= 2, ref)}
    R["sens_at_85spec"], R["spec_at_85"], R["thr_at_85"] = at_specificity(y >= 2, ref, 0.85)
    R["sens_at_0.5"], R["spec_at_0.5"] = at_threshold(y >= 2, ref, 0.5)
    if datasets is not None:
        datasets = np.asarray(datasets)
        R["per_dataset"] = {d: {"n": int((datasets == d).sum()), "auc": auc(y[datasets == d] >= 2, ref[datasets == d]),
                                "qwk": qwk(y[datasets == d], pred[datasets == d], K)} for d in np.unique(datasets)}
    return R


@torch.no_grad()
def seg_metrics(model, df, cfg, thresholds=None, tta=False):
    L = cfg.n_lesion
    tp, fp, fn = np.zeros(L), np.zeros(L), np.zeros(L)
    B = 256                                          # probability histograms -> every threshold at once:
    hp, ha = np.zeros((L, B)), np.zeros((L, B))       # Dice at a fixed 0.5 reads 0.0 for few-pixel lesions
    res = predict_eyes(model, df.rename(columns={"path": "proc_path"}), cfg, bs=1, with_maps=True, tta=tta)
    for i, masks in enumerate(df.masks):
        Pm = res["maps"][i].float()
        P = Pm.numpy() > 0.5
        for c in range(L):
            if masks[c] == "":
                continue
            Y = cv2.imread(masks[c], cv2.IMREAD_GRAYSCALE) > 0
            tp[c] += (P[c] & Y).sum(); fp[c] += (P[c] & ~Y).sum(); fn[c] += (~P[c] & Y).sum()
            ha[c] += np.histogram(Pm[c].numpy(), B, (0, 1))[0]
            if Y.any():
                hp[c] += np.histogram(Pm[c].numpy()[Y], B, (0, 1))[0]
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    out = dict(zip(cfg.seg_classes[:L], dice.round(4).tolist()))
    edges = np.linspace(0, 1, B + 1)[:-1]
    for c in range(L):
        TP = hp[c][::-1].cumsum()[::-1]
        ALL = ha[c][::-1].cumsum()[::-1]
        if TP[0] == 0:
            continue
        d = 2 * TP / np.maximum(ALL + TP[0], 1)       # ALL = tp+fp, TP[0] = tp+fn
        prec, rec = TP / np.maximum(ALL, 1), TP / TP[0]
        k = int(np.argmax(d))
        out[cfg.seg_classes[c] + "_best"] = round(float(d[k]), 4)
        out[cfg.seg_classes[c] + "_thr"] = round(float(edges[k]), 3)
        out[cfg.seg_classes[c] + "_ap"] = round(float(np.sum(np.diff(np.r_[0.0, rec[::-1]]) * prec[::-1])), 4)
        if thresholds and cfg.seg_classes[c] in thresholds:    # honest transfer: threshold fitted elsewhere
            j = min(int(thresholds[cfg.seg_classes[c]] * B), B - 1)
            out[cfg.seg_classes[c] + "_at_fixed"] = round(float(d[j]), 4)
    return out


@torch.no_grad()
def quality_metrics(model, df, cfg, bs=16):
    model.eval()
    Q = []
    for byte_list, _ in _loader(df.proc_path.tolist(), bs):
        x = gpu_decode(byte_list, cfg.global_size).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=getattr(torch, cfg.amp_dtype)):
            Q.append(torch.softmax(model.forward_quality(x).float(), 1).cpu())
    Q = torch.cat(Q).numpy()
    y = df.quality_label.to_numpy().astype(int)
    return {"accuracy": float((Q.argmax(1) == y).mean()), "auc_reject": auc(y == 2, Q[:, 2])}


def evaluate(model, M, cfg, max_images=10 ** 9, part="val"):
    key = {"val": "grade_val", "test": "grade_test", "external": "grade_external_test"}[part]
    G = _subsample(M[key], max_images)
    R = {"grade": grading_metrics(G.dr_grade, predict_eyes(model, G, cfg)["P"], G.dataset, cfg.num_grades)}
    if part in ("val", "test") and len(M[f"seg_{part}"]):
        S = M[f"seg_{part}"]
        S = S[S.masks.map(lambda m: any(v != "" for v in m[:cfg.n_lesion]))]
        R["seg_dice"] = seg_metrics(model, _subsample(S, max_images), cfg)
    if part == "val" and len(M["qual_val"]):
        R["quality"] = quality_metrics(model, _subsample(M["qual_val"], max_images), cfg)
    return R
