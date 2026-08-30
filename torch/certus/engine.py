"""Training step, optimiser, schedule and EMA (mirrors matlab/train/trainStep.m + optim.m).

One optimiser step = `accum_grade` grade sub-batches + one seg batch + one quality batch,
each backpropagated immediately (memory peaks at the largest sub-batch), then one
clipped AdamW update on the summed multi-task gradient.
"""
import copy
import math

import torch

from . import losses
from .augment import augment
from .data import gpu_crop, gpu_decode
from .model import CertusNet, param_groups


class Trainer:
    def __init__(self, cfg, enc_ratio=None, learn_logvar=True, pretrained=True, model=None):
        self.cfg = cfg
        self.model = (model or CertusNet(cfg, pretrained=pretrained)).cuda().to(memory_format=torch.channels_last)
        self.model.train()
        ratio = cfg.enc_lr_ratio if enc_ratio is None else enc_ratio
        self.opt = torch.optim.AdamW(param_groups(self.model, cfg, ratio), lr=0.0, betas=cfg.betas, fused=True)
        self.learn_logvar = learn_logvar
        self.amp = getattr(torch, cfg.amp_dtype)
        self.gen = torch.Generator(device="cuda").manual_seed(cfg.seed)
        self.ema = None

    def enable_ema(self):
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self._ema_params = list(self.ema.parameters())
        self._params = list(self.model.parameters())

    @torch.no_grad()
    def update_ema(self):
        torch._foreach_lerp_(self._ema_params, self._params, 1 - self.cfg.ema_decay)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = 0.0 if (g["name"] == "logvar" and not self.learn_logvar) else lr * g["lr_mult"]

    # -------------------------------------------------------------- sub-batch losses
    def grade_loss(self, byte_list, y):
        x = gpu_decode(byte_list, self.cfg.canvas)
        x, _, _ = augment(x, None, self.cfg, "full", self.gen)
        with torch.autocast("cuda", dtype=self.amp):
            out = self.model.forward_eye(x.contiguous(memory_format=torch.channels_last))
        L = losses.ordinal(out["grade_logits"], y.cuda(), self.cfg)
        return L, losses.weigh(L, self.model.logvar[0])

    def seg_loss(self, byte_list, boxes, Y, V):
        x = gpu_crop(byte_list, boxes, self.cfg.tile)
        x, Y, fov = augment(x, Y.cuda().float(), self.cfg, "patch", self.gen)
        with torch.autocast("cuda", dtype=self.amp):
            Z = self.model.forward_seg(x.contiguous(memory_format=torch.channels_last))
        L = losses.segmentation(Z, Y, V.cuda(), fov, self.cfg)
        return L, losses.weigh(L, self.model.logvar[1])

    def quality_loss(self, byte_list, y):
        x = gpu_decode(byte_list, self.cfg.global_size)
        x, _, _ = augment(x, None, self.cfg, "quality", self.gen)
        with torch.autocast("cuda", dtype=self.amp):
            z = self.model.forward_quality(x.contiguous(memory_format=torch.channels_last))
        L = losses.partial_ce(z, y.cuda())
        return L, losses.weigh(L, self.model.logvar[2])

    # -------------------------------------------------------------- one optimiser step
    def step(self, batches, lr):
        """batches = (list of accum_grade grade batches, seg batch, quality batch)."""
        grade_batches, seg_b, qual_b = batches
        self.set_lr(lr)
        self.opt.zero_grad(set_to_none=True)
        A = len(grade_batches)
        Lg = 0.0
        for gb in grade_batches:
            L, W = self.grade_loss(*gb)
            (W / A).backward()
            Lg = Lg + L.detach() / A
        Ls, W = self.seg_loss(*seg_b)
        W.backward()
        Lq, W = self.quality_loss(*qual_b)
        W.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        if self.ema is not None:
            self.update_ema()
        vals = torch.stack([Lg, Ls.detach(), Lq.detach(), gnorm.detach().float()]).tolist()  # one sync
        return {"grade": vals[0], "seg": vals[1], "quality": vals[2], "total": sum(vals[:3]),
                "grad_norm": vals[3], "lr": lr, "logvar": self.model.logvar.detach().tolist()}


def fetch(iters, cfg):
    g, s, q = iters
    return [next(g) for _ in range(cfg.accum_grade)], next(s), next(q)


def schedule(it, total, cfg):
    w = max(1, round(cfg.warmup_frac * total))
    if it <= w:
        return cfg.lr * it / w
    p = (it - w) / max(1, total - w)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))
