"""Learning-rate range test (Smith 2017) on the full multi-task objective.

For every encoder/head LR ratio in cfg.lrf_ratios the model restarts from the same
ImageNet weights, sees the same data in the same order, and the LR grows
exponentially from lrf_min to lrf_max. Task weights are frozen (logvar = 0) so the
loss curves are comparable. Suggested LR = min(steepest-descent LR, LR at min / 10).
Writes runs_torch/lr_finder.json (read by load_config) and lr_finder.png.
"""
import json
import os
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import load_config  # noqa: E402
from certus.data import load_manifests, make_loaders  # noqa: E402
from certus.engine import Trainer, fetch  # noqa: E402


def analyse(lrs, raw, sm, ratio):
    n = int(np.max(np.nonzero(np.isfinite(sm))[0])) + 1
    lr, s = lrs[:n], sm[:n]
    i_min = int(np.argmin(s))
    i0 = max(1, round(0.1 * n))                        # skip the smoother's warm-up
    d = np.gradient(s, np.log10(lr))
    i_steep = i0 + int(np.argmin(d[i0:i_min + 1])) if i_min > i0 else i_min
    return {"ratio": ratio, "min_loss": float(s[i_min]), "lr_at_min": float(lr[i_min]),
            "lr_steep": float(lr[i_steep]), "lr_suggest": float(min(lr[i_steep], lr[i_min] / 10)),
            "lrs": lr.tolist(), "smooth": s.tolist(), "raw": raw[:n].tolist()}


def main():
    cfg = load_config()
    assert os.path.exists(f"{cfg.runs}/batch_probe.json"), "run probe.py first (batch sizes must be measured)"
    torch.backends.cudnn.benchmark = True
    M = load_manifests(cfg)
    lrs = cfg.lrf_min * (cfg.lrf_max / cfg.lrf_min) ** (np.arange(cfg.lrf_steps) / (cfg.lrf_steps - 1))
    runs = []
    for ratio in cfg.lrf_ratios:
        torch.manual_seed(cfg.seed)
        tr = Trainer(cfg, enc_ratio=ratio, learn_logvar=False)
        iters = make_loaders(M, cfg, seed=cfg.seed)          # identical data order for every ratio
        raw = np.full(cfg.lrf_steps, np.nan)
        sm = raw.copy()
        avg, best, t0 = 0.0, np.inf, time.perf_counter()
        for i in range(cfg.lrf_steps):
            info = tr.step(fetch(iters, cfg), float(lrs[i]))
            raw[i] = info["total"]
            avg = cfg.lrf_smooth * avg + (1 - cfg.lrf_smooth) * raw[i]
            sm[i] = avg / (1 - cfg.lrf_smooth ** (i + 1))
            best = min(best, sm[i])
            if (i + 1) % 10 == 0:
                print(f"ratio {ratio:.2f} step {i + 1:3d}/{cfg.lrf_steps} lr {lrs[i]:.2e} loss {raw[i]:.4f} "
                      f"smooth {sm[i]:.4f} ({(time.perf_counter() - t0) / (i + 1):.2f}s/step)", flush=True)
            if not np.isfinite(raw[i]) or (i > 10 and sm[i] > cfg.lrf_diverge * best):
                print(f"ratio {ratio:.2f} diverged at lr {lrs[i]:.2e}", flush=True)
                sm[i] = np.nan if not np.isfinite(raw[i]) else sm[i]
                break
        del iters, tr
        torch.cuda.empty_cache()
        runs.append(analyse(lrs, raw, sm, ratio))
        r = runs[-1]
        print(f"ratio {ratio:.2f}: min {r['min_loss']:.4f} @ {r['lr_at_min']:.2e} | steepest {r['lr_steep']:.2e} "
              f"| suggested {r['lr_suggest']:.2e}", flush=True)

    mins = np.array([r["min_loss"] for r in runs])
    ok = [r for r in runs if r["min_loss"] <= 1.01 * mins.min()]
    c = min(ok, key=lambda r: r["ratio"])                    # ties within 1% -> smaller encoder ratio
    out = {"lr": c["lr_suggest"], "enc_lr_ratio": c["ratio"],
           "chosen_because": "lowest min smoothed loss (ties within 1% -> smaller encoder ratio)",
           "rule": "lr = min(steepest-descent lr, lr_at_min / 10)", "steps": cfg.lrf_steps, "runs": runs}
    with open(f"{cfg.runs}/lr_finder.json", "w") as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in runs:
        ax.semilogx(r["lrs"], r["smooth"], lw=1.6, label=f"encoder ratio {r['ratio']:.2f}")
    ax.axvline(c["lr_suggest"], ls="--", c="k", lw=1)
    ax.text(c["lr_suggest"], ax.get_ylim()[1], f" chosen {c['lr_suggest']:.1e}", va="top", fontsize=9)
    ax.set_xlabel("peak learning rate (heads / decoder)")
    ax.set_ylabel("smoothed multi-task loss")
    ax.set_title("Certus LR range test")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{cfg.runs}/lr_finder.png", dpi=150)
    print(f"\nchosen lr {c['lr_suggest']:.2e}, encoder ratio {c['ratio']:.2f}")


if __name__ == "__main__":
    main()
