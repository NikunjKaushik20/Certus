"""Full multi-task training. Requires runs_torch/batch_probe.json and lr_finder.json.

Validates the EMA weights every cfg.val_every steps and keeps the best checkpoint by
referable-DR AUC on the *val* split. Messidor-2 is never touched here.
Resume: python train.py --resume runs_torch/train_<stamp>/last.pt
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import load_config  # noqa: E402
from certus.data import load_manifests, make_loaders  # noqa: E402
from certus.engine import Trainer, fetch, schedule  # noqa: E402
from certus.evaluate import evaluate  # noqa: E402


def save_atomic(obj, path, keep_previous=True):
    """Write, flush, then rename. torch.save writes in place: a process killed mid-write leaves a
    truncated zip and the run is unrecoverable (this cost 2.5 hours of training once already).
    The previous checkpoint is kept as .prev so even a corrupted rename has a fallback."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    if keep_previous and os.path.exists(path):
        os.replace(path, path + ".prev")
    os.replace(tmp, path)


def validate(tr, run, M, cfg):
    """Validate the EMA weights, in a child process when that is possible.

    Validation allocates much larger blocks than a training step: whole-eye probability maps over
    3,049 eyes, against 512 px tiles. On a 6 GB card that leaves the caching allocator fragmented
    at the ceiling, and Windows then spills training into system RAM rather than raising OOM. Run
    train_20260912_211356 lost an entire night to it: 2.94 s steps became 45 s two minutes after
    the epoch-1 validation, at 100% GPU utilisation, 3% memory utilisation and 27 W of an 80 W
    budget. That is a card waiting on PCIe.

    A child process returns the whole pool to the driver when it exits, which empty_cache cannot
    do. The parent frees its own cached blocks first so the child has room; the live tensors that
    stay behind are the model, the EMA copy and the Adam moments, well under a gigabyte.

    If anything about the child goes wrong -- import error, OOM, a hang -- fall back to validating
    in process. If that fails too, return None: the caller still checkpoints and keeps training.

    That last part matters more since the sysmem fallback policy was set to "prefer no sysmem
    fallback". An overrun now raises a hard CUDA OOM instead of going slow, and validation is the
    largest allocation in the run. An unguarded failure here would kill the process at a fixed
    step, and every resume from last.pt would march back to that same step and die again. Losing
    one epoch's metrics is cheap; losing the night to a crash loop is not.
    """
    torch.cuda.empty_cache()
    weights, out = f"{run}/_val_ema.pt", f"{run}/_val_out.json"
    try:
        save_atomic({"ema": tr.ema.state_dict()}, weights, keep_previous=False)
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_once.py"),
               "--run", run, "--weights", weights, "--out", out]
        r = subprocess.run(cmd, timeout=5400)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"validate_once.py exited {r.returncode}")
        with open(out) as fh:
            R = json.load(fh)
    except Exception as e:                                        # noqa: BLE001 - deliberate catch-all
        print(f">> subprocess validation failed ({e}); validating in process", flush=True)
        try:
            torch.cuda.empty_cache()
            R = evaluate(tr.ema, M, cfg, 10 ** 9, "val")
        except Exception as e2:                                   # noqa: BLE001
            print(f">> in-process validation failed too ({e2}); skipping this epoch's metrics "
                  f"and continuing to train", flush=True)
            R = None
    finally:
        for f in (weights, out):
            if os.path.exists(f):
                os.remove(f)
        torch.cuda.empty_cache()
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume")
    ap.add_argument("--init", help="warm start from a best.pt (EMA weights only): fresh optimiser, "
                                   "fresh schedule, step counter back to 1")
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()

    cfg = load_config(**({"epochs": a.epochs} if a.epochs else {}))
    assert os.path.exists(f"{cfg.runs}/batch_probe.json"), "run probe.py first"
    assert math.isfinite(cfg.lr), "run lr_finder.py first (no measured learning rate)"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run = os.path.dirname(a.resume) if a.resume else f"{cfg.runs}/train_{datetime.now():%Y%m%d_%H%M%S}"
    assert not (a.resume and a.init), "--resume continues a run; --init starts a new one from weights"
    os.makedirs(run, exist_ok=True)
    with open(f"{run}/config.json", "w") as fh:
        fh.write(cfg.to_json())

    M = load_manifests(cfg)
    torch.manual_seed(cfg.seed)
    tr = Trainer(cfg)
    tr.enable_ema()
    start, best_auc = 1, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location="cuda", weights_only=False)
        tr.model.load_state_dict(ck["model"]); tr.ema.load_state_dict(ck["ema"])
        tr.opt.load_state_dict(ck["opt"]); tr.gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        start, best_auc = ck["step"] + 1, ck["best_auc"]
    elif a.init:
        ck = torch.load(a.init, map_location="cuda", weights_only=False)
        w = ck.get("model") or ck["ema"]                       # best.pt carries EMA weights only
        tr.model.load_state_dict(w); tr.ema.load_state_dict(ck["ema"])
        print(f"warm start from {a.init} (trained to step {ck.get('step')}), fresh optimiser", flush=True)
    iters = make_loaders(M, cfg, seed=cfg.seed + start)

    steps_per_epoch = math.ceil(len(M["grade_train"]) / (cfg.batch_grade * cfg.accum_grade))
    total = cfg.epochs * steps_per_epoch
    print(f"{total} steps ({cfg.epochs} epochs x {steps_per_epoch}), lr {cfg.lr:.2e}, "
          f"encoder ratio {cfg.enc_lr_ratio}, run dir {run}", flush=True)

    logf = open(f"{run}/log.csv", "a", newline="")
    log = csv.writer(logf)
    valf = open(f"{run}/val.csv", "a", newline="")
    val = csv.writer(valf)
    if start == 1:
        log.writerow(["step", "epoch", "lr", "total", "grade", "seg", "quality", "grad_norm",
                      "lv_grade", "lv_seg", "lv_quality", "sec"])
        val.writerow(["step", "auc", "sens_at_85spec", "qwk", "ece", "dice", "quality_acc"])

    t0 = time.perf_counter()
    for it in range(start, total + 1):
        lr = schedule(it, total, cfg)
        info = tr.step(fetch(iters, cfg), lr)
        if not math.isfinite(info["total"]):
            raise RuntimeError(f"non-finite loss at step {it}")
        log.writerow([it, f"{it / steps_per_epoch:.3f}", f"{lr:.3e}", f"{info['total']:.5f}", f"{info['grade']:.5f}",
                      f"{info['seg']:.5f}", f"{info['quality']:.5f}", f"{info['grad_norm']:.3f}",
                      *[f"{v:.3f}" for v in info["logvar"]], f"{time.perf_counter() - t0:.1f}"])
        if it % 50 == 0:
            logf.flush()
            el = time.perf_counter() - t0
            print(f"step {it}/{total} lr {lr:.2e} loss {info['total']:.4f} (g {info['grade']:.3f} "
                  f"s {info['seg']:.3f} q {info['quality']:.3f}) {el / (it - start + 1):.2f}s/step "
                  f"ETA {(total - it) * el / (it - start + 1) / 3600:.1f}h", flush=True)

        if it % 250 == 0 and it % steps_per_epoch != 0:           # cheap crash insurance between epochs
            save_atomic({"model": tr.model.state_dict(), "ema": tr.ema.state_dict(),
                         "opt": tr.opt.state_dict(), "gen": tr.gen.get_state(), "step": it,
                         "best_auc": best_auc, "cfg": cfg.to_json()}, f"{run}/last.pt")

        if it % steps_per_epoch == 0 or it == total:              # full val set at every epoch end
            R = validate(tr, run, M, cfg)
            g = R["grade"] if R else None
            if g:
                val.writerow([it, g["auc_referable"], g["sens_at_85spec"], g["qwk"], g["ece_referable"],
                              json.dumps(R.get("seg_dice")), R.get("quality", {}).get("accuracy")])
                valf.flush()
                print(f">> val {it}: AUC {g['auc_referable']:.4f} | sens@85%spec {g['sens_at_85spec']:.3f} | "
                      f"QWK {g['qwk']:.3f} | ECE {g['ece_referable']:.3f} | dice {R.get('seg_dice')}", flush=True)
            # checkpoint regardless: an epoch of training is worth keeping even when the metrics
            # that would have described it could not be computed
            ck = {"model": tr.model.state_dict(), "ema": tr.ema.state_dict(), "opt": tr.opt.state_dict(),
                  "gen": tr.gen.get_state(), "step": it,
                  "best_auc": max(best_auc, g["auc_referable"]) if g else best_auc,
                  "val": R, "cfg": cfg.to_json()}
            save_atomic(ck, f"{run}/last.pt")
            if g and g["auc_referable"] > best_auc:
                best_auc = g["auc_referable"]
                save_atomic({"ema": tr.ema.state_dict(), "step": it, "val": R,
                             "cfg": cfg.to_json()}, f"{run}/best.pt")
                print(f">> new best AUC {best_auc:.4f}", flush=True)
            tr.model.train()
    print(f"done. best val AUC {best_auc:.4f} -> {run}")


if __name__ == "__main__":
    main()
