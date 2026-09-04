"""Plain Trainer.step loop (no anomaly mode) with NaN diagnostics after each step."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import load_config  # noqa: E402
from certus.data import load_manifests, make_loaders  # noqa: E402
from certus.engine import Trainer, fetch  # noqa: E402


def main():
    tag = "blocking" if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1" else "async"
    cfg = load_config()
    M = load_manifests(cfg)
    iters = make_loaders(M, cfg)
    torch.manual_seed(cfg.seed)
    tr = Trainer(cfg, enc_ratio=0.1)
    nan_steps = 0
    for i in range(15):
        info = tr.step(fetch(iters, cfg), 1e-4)
        badg = [n for n, p in tr.model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        badp = [n for n, p in tr.model.named_parameters() if not torch.isfinite(p).all()]
        if badg or badp or info["grad_norm"] != info["grad_norm"]:
            nan_steps += 1
            print(f"[{tag}] step {i}: grad_norm {info['grad_norm']} | non-finite grads {len(badg)} {badg[:4]} "
                  f"| non-finite params {len(badp)}", flush=True)
            break
        print(f"[{tag}] step {i}: ok grad_norm {info['grad_norm']:.3f}", flush=True)
    print(f"[{tag}] steps with NaN: {nan_steps}", flush=True)


if __name__ == "__main__":
    main()
