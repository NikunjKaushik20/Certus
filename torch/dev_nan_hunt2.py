"""Reproduce Trainer.step and locate where NaN first appears: accumulation, clip or update."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import load_config  # noqa: E402
from certus.data import load_manifests, make_loaders  # noqa: E402
from certus.engine import Trainer, fetch  # noqa: E402


def nonfinite(model, what="grad"):
    out = []
    for n, p in model.named_parameters():
        t = p.grad if what == "grad" else p.data
        if t is not None and not torch.isfinite(t).all():
            out.append(n)
    return out


def main():
    cfg = load_config()
    M = load_manifests(cfg)
    iters = make_loaders(M, cfg)
    torch.manual_seed(cfg.seed)
    tr = Trainer(cfg, enc_ratio=0.1)
    m = tr.model
    for step in range(4):
        grade_batches, seg_b, qual_b = fetch(iters, cfg)
        tr.set_lr(1e-4)
        tr.opt.zero_grad(set_to_none=True)
        A = len(grade_batches)
        subs = [(f"grade[{i}]", lambda gb=gb: tr.grade_loss(*gb), 1 / A) for i, gb in enumerate(grade_batches)]
        subs += [("seg", lambda: tr.seg_loss(*seg_b), 1.0), ("quality", lambda: tr.quality_loss(*qual_b), 1.0)]
        for name, fn, w in subs:
            L, W = fn()
            (W * w).backward()
            print(f"step {step} after {name:9s}: non-finite grads {nonfinite(m)[:3]}", flush=True)
        total_sq = sum(float(p.grad.float().pow(2).sum()) for p in m.parameters() if p.grad is not None)
        none = [n for n, p in m.named_parameters() if p.grad is None]
        print(f"  sum of squared grads {total_sq:.4e}; params without grad: {len(none)} {none[:5]}", flush=True)
        g = torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip)
        print(f"  clip_grad_norm_ -> {g.item():.4f}; non-finite grads after clip {nonfinite(m)[:3]}", flush=True)
        tr.opt.step()
        print(f"  after AdamW: non-finite params {nonfinite(m, 'data')[:3]}", flush=True)
        if nonfinite(m, "data"):
            break
    info = tr.step(fetch(iters, cfg), 1e-4)
    print("Trainer.step ->", {k: info[k] for k in ("total", "grad_norm")}, flush=True)


if __name__ == "__main__":
    main()
