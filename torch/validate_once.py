"""Run one validation pass in its own process and write the result as JSON.

Validation allocates far larger blocks than a training step: whole-eye probability maps over 3,049
eyes, against 512 px tiles. On a 6 GB card, doing that in the training process leaves the caching
allocator fragmented right at the ceiling, and Windows then spills training into system RAM instead
of failing. That is not a theory: after the epoch-1 validation of run train_20260912_211356, steps
went from 2.94 s to 45 s and never came back, with the GPU reporting 100% utilisation at 3% memory
utilisation and 27 W -- a card waiting on PCIe, not computing.

A child process hands the entire pool back to the driver when it exits, which is the one thing
empty_cache() cannot do.

Usage: python validate_once.py --run <run dir> --weights <ema .pt> --out <json>
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import Config  # noqa: E402
from certus.data import load_manifests  # noqa: E402
from certus.evaluate import evaluate  # noqa: E402
from certus.model import CertusNet  # noqa: E402


def read_config(run):
    """Rebuild the run's Config from the JSON it wrote at startup.

    JSON has no tuples, so every tuple field comes back as a list. seg_classes and
    decoder_channels are only ever indexed and iterated, but lesion_priority is fed to the
    sampler, and an unhashable list in the wrong place fails minutes in rather than at once.
    Coerce anything the dataclass declared as a tuple back to one."""
    with open(os.path.join(run, "config.json")) as fh:
        d = json.load(fh)
    ref = Config()
    for k, v in list(d.items()):
        if isinstance(getattr(ref, k, None), tuple) and isinstance(v, list):
            d[k] = tuple(v)
    return Config(**{k: v for k, v in d.items() if hasattr(ref, k)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory; its config.json defines the model")
    ap.add_argument("--weights", required=True, help="checkpoint holding an 'ema' state dict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="val")
    ap.add_argument("--max-images", type=int, default=10 ** 9,
                    help="cap the split for a smoke test; the real pass leaves this alone")
    a = ap.parse_args()

    cfg = read_config(a.run)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    M = load_manifests(cfg)
    ck = torch.load(a.weights, map_location="cuda", weights_only=False)

    # pretrained=False: the weights are about to be overwritten anyway, and a child process should
    # not reach for the network at 3am to fetch an encoder it will immediately discard
    model = CertusNet(cfg, pretrained=False).cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(ck["ema"])

    R = evaluate(model, M, cfg, a.max_images, a.part)

    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(R, fh, default=float)          # numpy scalars would not serialise otherwise
    os.replace(tmp, a.out)
    g = R["grade"]
    print(f"validated {a.part}: AUC {g['auc_referable']:.4f} "
          f"sens@85 {g['sens_at_85spec']:.3f}", flush=True)


if __name__ == "__main__":
    main()
