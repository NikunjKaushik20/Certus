"""Manifests, samplers and loaders. Workers only read bytes / decode small PNG masks;
JPEG decoding, cropping, resizing and augmentation run on the GPU (nvJPEG)."""
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import decode_jpeg, read_file


class InfiniteSampler(Sampler):
    """Endless weighted sampling with replacement, drawn in small chunks.
    (WeightedRandomSampler(num_samples=huge) materialises every index up front: GBs of RAM.)"""

    def __init__(self, weights, generator, chunk=4096):
        self.w = torch.as_tensor(np.asarray(weights, dtype=np.float64))
        self.g, self.chunk = generator, chunk

    def __iter__(self):
        while True:
            yield from torch.multinomial(self.w, self.chunk, replacement=True, generator=self.g).tolist()


# ---------------------------------------------------------------- manifests
def load_manifests(cfg):
    rd = lambda f: pd.read_csv(os.path.join(cfg.manifests, f), low_memory=False)
    G = rd("grading.csv")
    M = {f"grade_{s}": G[G.split == s].reset_index(drop=True) for s in ["train", "val", "test", "external_test"]}
    aux = os.path.join(cfg.manifests, "grading_aux.csv")
    if cfg.use_aux_eyepacs and os.path.exists(aux):
        A = rd("grading_aux.csv")
        M["grade_train"] = pd.concat([M["grade_train"], A[A.split == "train"]], ignore_index=True)

    S = pd.concat([_seg_table(rd("segmentation.csv"), cfg), _seg_table(rd("vessels.csv"), cfg)], ignore_index=True)
    for s in ["train", "val", "test"]:
        M[f"seg_{s}"] = S[S.split == s].reset_index(drop=True)

    Q = rd("quality.csv")
    Q.loc[Q.quality_label.isna() & (Q.gradable == 0), "quality_label"] = 2
    Q = Q[Q.quality_label.notna() | (Q.gradable == 1)]
    M["qual_train"] = Q[Q.split == "train"].reset_index(drop=True)
    M["qual_val"] = Q[(Q.split == "val") & Q.quality_label.notna()].reset_index(drop=True)
    return M


def _seg_table(T, cfg):
    cols = []
    for c in cfg.seg_classes:
        col = f"proc_mask_{c}"
        v = T[col].fillna("").astype(str) if col in T else pd.Series([""] * len(T))
        cols.append(v.replace("unknown", "").to_numpy())
    return pd.DataFrame({"uid": T.uid.values, "dataset": T.dataset.values, "split": T.split.values,
                         "path": T.proc_path.values, "fov": T.fov_path.values,
                         "masks": list(np.stack(cols, 1))})


def _balance(x, power):
    _, inv, cnt = np.unique(np.asarray(x), return_inverse=True, return_counts=True)
    return cnt[inv].astype(np.float64) ** (-power)


# ---------------------------------------------------------------- datasets
class BytesDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = list(paths), np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return read_file(self.paths[i]), int(self.labels[i])


def collate_bytes(batch):
    return [b[0] for b in batch], torch.tensor([b[1] for b in batch])


class SegDataset(Dataset):
    """Returns image bytes + a lesion-aware crop box + the cropped masks (decoded here)."""

    def __init__(self, df, cfg):
        self.df, self.cfg = df, cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        cfg, r = self.cfg, self.df.iloc[i]
        T, L = cfg.tile, cfg.n_lesion
        masks = r.masks
        valid = np.array([m != "" for m in masks])
        full = [cv2.imread(m, cv2.IMREAD_GRAYSCALE) > 0 if v else None for m, v in zip(masks, valid)]
        D = next((m.shape[0] for m in full if m is not None), cfg.canvas)
        rng = getattr(self, "rng", None) or np.random.default_rng()   # per-worker stream (see _worker_init)
        centre = None
        if rng.random() < cfg.lesion_centred:
            present = [c for c in range(L) if valid[c] and full[c].any()]
            if present:
                w = np.array([cfg.lesion_priority[c] for c in present], float)
                c = present[rng.choice(len(present), p=w / w.sum())]
                ys, xs = np.nonzero(full[c])
                k = rng.integers(len(ys))
                centre = np.array([ys[k], xs[k]]) + rng.integers(-T // 4, T // 4 + 1, size=2)
        if centre is None:
            ys, xs = np.nonzero(cv2.imread(r.fov, cv2.IMREAD_GRAYSCALE) > 0)
            k = rng.integers(len(ys))
            centre = np.array([ys[k], xs[k]])
        r0 = int(np.clip(centre[0] - T // 2, 0, D - T))
        c0 = int(np.clip(centre[1] - T // 2, 0, D - T))
        Y = np.zeros((len(masks), T, T), np.uint8)
        for c, m in enumerate(full):
            if m is not None:
                Y[c] = m[r0:r0 + T, c0:c0 + T]
        return read_file(r.path), torch.tensor([r0, c0]), torch.from_numpy(Y), torch.from_numpy(valid)


def collate_seg(batch):
    return ([b[0] for b in batch], torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]))


def _worker_init(wid):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)   # reproducible given the loader seed


# ---------------------------------------------------------------- loaders
def _loader(ds, batch, sampler, workers, collate, seed):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers, collate_fn=collate,
                      persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
                      worker_init_fn=_worker_init, generator=g)


def make_loaders(M, cfg, seed=None):
    """Three infinite loaders (grade, seg, quality) with the balancing of makeSampler.m."""
    seed = cfg.seed if seed is None else seed
    G = M["grade_train"]
    wg = _balance(G.dr_grade, cfg.grade_power) * _balance(G.dataset, cfg.dataset_power)
    sg = InfiniteSampler(wg, torch.Generator().manual_seed(seed))
    grade = _loader(BytesDataset(G.proc_path, G.dr_grade), cfg.batch_grade, sg, cfg.workers_grade,
                    collate_bytes, seed)

    S = M["seg_train"]
    ss = InfiniteSampler(np.ones(len(S)), torch.Generator().manual_seed(seed + 1))
    seg = _loader(SegDataset(S, cfg), cfg.batch_seg, ss, cfg.workers_seg, collate_seg, seed + 1)

    Q = M["qual_train"]
    full = Q.quality_label.notna().to_numpy()
    wq = np.zeros(len(Q))
    wf = _balance(Q.quality_label[full], cfg.grade_power)
    wq[full] = (1 - cfg.partial_quality) * wf / wf.sum()
    wq[~full] = cfg.partial_quality / (~full).sum()
    labels = Q.quality_label.fillna(-1).astype(int)             # -1 = "not reject"
    sq = InfiniteSampler(wq, torch.Generator().manual_seed(seed + 2))
    qual = _loader(BytesDataset(Q.proc_path, labels), cfg.batch_quality, sq, cfg.workers_quality,
                   collate_bytes, seed + 2)
    return iter(grade), iter(seg), iter(qual)


# ---------------------------------------------------------------- GPU decode
def _decode_cuda(byte_list):
    """Decode on the CPU, then copy to the GPU.

    nvJPEG cost more than it saved on a 6 GB card: its buffers pushed VRAM to the ceiling (steps
    went from 3.1s to 20s+ as the allocator thrashed), and because it runs outside PyTorch's stream
    ordering it needed a synchronize on both sides -- which serialised every step and, without the
    fences, produced NaN gradients. CPU decode of ~10 canvases costs ~0.2s of a 3.1s step.
    """
    imgs = decode_jpeg(byte_list, device="cpu")
    return [im.to("cuda", non_blocking=True) for im in imgs]


def gpu_decode(byte_list, size=None):
    """nvJPEG batch decode -> float (N,3,H,W) in [0,1]; optional resize to `size` (antialiased)."""
    imgs = _decode_cuda(byte_list)
    out = []
    for im in imgs:
        x = im.unsqueeze(0).float().div_(255)
        if size is not None and x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        out.append(x)
    return torch.cat(out)


def gpu_crop(byte_list, boxes, tile):
    full = _decode_cuda(byte_list)
    return torch.stack([im[:, r:r + tile, c:c + tile] for im, (r, c) in zip(full, boxes.tolist())]).float().div_(255)
