"""Find which sub-batch / parameter produces non-finite gradients in Trainer.step."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import load_config  # noqa: E402
from certus.data import load_manifests, make_loaders  # noqa: E402
from certus.engine import Trainer, fetch  # noqa: E402


def bad_params(model):
    out = []
    for n, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            out.append(n)
    return out


def main():
    cfg = load_config()
    M = load_manifests(cfg)
    iters = make_loaders(M, cfg)
    torch.manual_seed(cfg.seed)
    tr = Trainer(cfg, enc_ratio=0.1)
    for step in range(6):
        grade_batches, seg_b, qual_b = fetch(iters, cfg)
        subs = [(f"grade[{i}]", lambda gb=gb: tr.grade_loss(*gb)) for i, gb in enumerate(grade_batches)]
        subs += [("seg", lambda: tr.seg_loss(*seg_b)), ("quality", lambda: tr.quality_loss(*qual_b))]
        for name, fn in subs:
            tr.opt.zero_grad(set_to_none=True)
            L, W = fn()
            W.backward()
            bad = bad_params(tr.model)
            print(f"step {step} {name:9s} loss {L.item():.4f} finite={torch.isfinite(L).item()} "
                  f"bad grads: {len(bad)} {bad[:4]}", flush=True)
            if bad:
                with torch.autograd.detect_anomaly():
                    tr.opt.zero_grad(set_to_none=True)
                    try:
                        fn()[1].backward()
                    except RuntimeError as e:
                        print("ANOMALY:", str(e).splitlines()[0], flush=True)
                return
    print("no non-finite gradients in 6 steps of individual sub-batches", flush=True)


if __name__ == "__main__":
    main()
