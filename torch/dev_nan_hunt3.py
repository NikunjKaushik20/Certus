"""Run Trainer.step itself with anomaly detection to find the op that creates NaN."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import load_config  # noqa: E402
from certus.data import load_manifests, make_loaders  # noqa: E402
from certus.engine import Trainer, fetch  # noqa: E402

orig_clip = torch.nn.utils.clip_grad_norm_


def checked_clip(params, max_norm, **kw):
    params = list(params)
    bad = [i for i, p in enumerate(params) if p.grad is not None and not torch.isfinite(p.grad).all()]
    g = orig_clip(params, max_norm, **kw)
    print(f"    clip: {len(bad)} params with non-finite grads before clip; norm {g.item():.4f}", flush=True)
    return g


def main():
    torch.nn.utils.clip_grad_norm_ = checked_clip
    cfg = load_config()
    M = load_manifests(cfg)
    iters = make_loaders(M, cfg)
    torch.manual_seed(cfg.seed)
    tr = Trainer(cfg, enc_ratio=0.1)
    names = [n for n, _ in tr.model.named_parameters()]
    for i in range(6):
        B = fetch(iters, cfg)
        with torch.autograd.detect_anomaly(check_nan=True):
            try:
                info = tr.step(B, 1e-4)
            except RuntimeError as e:
                print(f"step {i}: ANOMALY -> {str(e)[:400]}", flush=True)
                return
        badp = [n for n, p in tr.model.named_parameters() if not torch.isfinite(p).all()]
        print(f"step {i}: total {info['total']:.4f} grad_norm {info['grad_norm']} non-finite params {len(badp)} {badp[:3]}",
              flush=True)
        if badp:
            return


if __name__ == "__main__":
    main()
