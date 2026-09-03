"""Smoke test + measured batch-size probe. Writes runs_torch/smoke_report.txt and batch_probe.json.
Does not train."""
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certus.config import load_config  # noqa: E402
from certus.data import load_manifests, make_loaders  # noqa: E402
from certus.engine import Trainer, fetch  # noqa: E402
from certus.evaluate import evaluate  # noqa: E402
from certus.model import make_tiles, stitch_tiles  # noqa: E402

LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def measure(fn, reps=3):
    torch.cuda.synchronize()
    fn()                                                   # warm-up (cudnn autotune, allocations)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), torch.cuda.max_memory_allocated()


def grad_norm(params):
    g = [p.grad.float().norm() for p in params if p.grad is not None]
    return float(torch.stack(g).norm()) if g else 0.0


def main():
    cfg = load_config()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    total_mem = torch.cuda.get_device_properties(0).total_memory
    # Windows drivers spill to system RAM instead of raising OOM (10-100x slower): hard cap so
    # an oversized batch fails fast and the probe moves on
    torch.cuda.set_per_process_memory_fraction(0.9)
    say(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} | "
        f"{total_mem / 2**30:.2f} GB | AMP {cfg.amp_dtype}")

    # ---- data
    M = load_manifests(cfg)
    say("rows: " + ", ".join(f"{k}={len(v)}" for k, v in M.items()))
    iters = make_loaders(M, cfg)
    t = time.perf_counter()
    B = fetch(iters, cfg)
    say(f"first step fetched in {time.perf_counter() - t:.1f}s (includes worker start-up)")
    assert B[1][2].any(), "seg batch has no positive pixels"

    # ---- tiling round trip
    x = torch.rand(2, 3, cfg.canvas, cfg.canvas, device="cuda")
    assert torch.equal(stitch_tiles(make_tiles(x, cfg.grid, cfg.tile), cfg.grid, cfg.tile), x)
    say("tiling round-trip OK")

    # ---- model + gradients reach every part
    torch.manual_seed(cfg.seed)
    tr = Trainer(cfg, enc_ratio=0.1)
    m = tr.model
    say(f"taps (stage idx per stride): {m.tap_idx} | channels {m.tap_ch} | params "
        f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
    parts = {
        "backbone conv": [p for n, p in m.named_parameters() if n.startswith("backbone.") and p.ndim == 4],
        "backbone BN affine": [p for n, p in m.named_parameters() if n.startswith("backbone.") and ".bn" in n],
        "decoder": [p for n, p in m.named_parameters() if n.startswith(("lat", "dec", "skip", "seg_out"))],
        "attention": list(m.attn.parameters()),
        "grade head": list(m.grade_head.parameters()),
        "quality head": list(m.qual_head.parameters()),
    }
    for name, (fn, keys) in {
        "grade": (lambda: tr.grade_loss(*B[0][0]), ["backbone conv", "backbone BN affine", "attention", "grade head"]),
        "seg": (lambda: tr.seg_loss(*B[1]), ["backbone conv", "decoder"]),
        "quality": (lambda: tr.quality_loss(*B[2]), ["backbone conv", "quality head"]),
    }.items():
        tr.opt.zero_grad(set_to_none=True)
        L, W = fn()
        W.backward()
        assert torch.isfinite(L), f"{name} loss not finite"
        norms = {k: grad_norm(parts[k]) for k in keys}
        say(f"{name:8s} loss {L.item():.4f} | grad norms " + ", ".join(f"{k} {v:.2e}" for k, v in norms.items()))
        assert all(v > 0 for v in norms.values()), f"zero gradient in {name}"

    w0 = m.grade_head[0].weight.detach().clone()
    info = tr.step(B, 1e-4)
    dw = (m.grade_head[0].weight - w0).norm().item()
    assert math.isfinite(info["grad_norm"]) and math.isfinite(dw) and dw > 0, \
        f"optimiser step broken: grad norm {info['grad_norm']}, |dW| {dw}"
    say(f"optimiser step OK: total {info['total']:.4f}, grad norm {info['grad_norm']:.3f}, |dW| {dw:.2e}")

    # ---- batch probe (optimizer state already allocated by the step above)
    gb, sb, qb = B[0][0], B[1], B[2]
    rep = lambda lst, n: (lst * math.ceil(n / len(lst)))[:n]
    tasks = {
        "grade": ([1, 2, 3, 4], lambda n: tr.grade_loss(rep(gb[0], n), gb[1][:1].repeat(n))),
        "seg": ([4, 6, 8, 12, 16, 24], lambda n: tr.seg_loss(rep(sb[0], n), sb[1][:1].repeat(n, 1),
                                                             sb[2][:1].repeat(n, 1, 1, 1), sb[3][:1].repeat(n, 1))),
        "quality": ([8, 16, 24, 32, 48, 64], lambda n: tr.quality_loss(rep(qb[0], n), qb[1][:1].repeat(n))),
    }
    budget = 0.75 * total_mem
    results = {}
    for gc in (False, True):
        m.cfg.grad_checkpointing = gc
        for task, (sizes, fn) in tasks.items():
            for n in sizes:
                def run():
                    tr.opt.zero_grad(set_to_none=True)
                    fn(n)[1].backward()
                try:
                    t, peak = measure(run)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # cuDNN/kernel OOMs arrive as RuntimeError
                    if "out of memory" not in str(e):
                        raise
                    del e
                    torch.cuda.empty_cache()
                    say(f"  gc={gc!s:5s} {task:7s} batch {n:3d}  OUT OF MEMORY")
                    break
                ok = peak <= budget
                say(f"  gc={gc!s:5s} {task:7s} batch {n:3d}  {t:.3f}s  peak {peak / 2**30:.2f} GB {'' if ok else '(over budget)'}")
                if not ok:
                    break
                results[(gc, task, n)] = t
            torch.cuda.empty_cache()
    tr.opt.zero_grad(set_to_none=True)

    best = None
    for gc in (False, True):
        pick = {}
        for task in tasks:
            cands = [(n, t) for (g, tk, n), t in results.items() if g == gc and tk == task]
            if not cands:
                break
            pick[task] = max(cands)                              # largest batch that fits
        if len(pick) < 3:
            continue
        ng, tg = pick["grade"]
        accum = max(1, round(cfg.target_eyes_per_step / ng))
        step_t = accum * tg + pick["seg"][1] + pick["quality"][1]
        eyes_per_s = accum * ng / step_t
        say(f"gc={gc}: grade {ng} x accum {accum}, seg {pick['seg'][0]}, quality {pick['quality'][0]} "
            f"-> {step_t:.2f}s/step, {eyes_per_s:.2f} eyes/s")
        if best is None or eyes_per_s > best["eyes_per_s"]:
            best = {"grad_checkpointing": gc, "batch_grade": ng, "accum_grade": accum, "batch_seg": pick["seg"][0],
                    "batch_quality": pick["quality"][0], "sec_per_step": step_t, "eyes_per_s": eyes_per_s}
    assert best, "nothing fits in memory"
    steps = math.ceil(len(M["grade_train"]) / (best["batch_grade"] * best["accum_grade"]))
    best.update(steps_per_epoch=steps, hours_per_epoch=steps * best["sec_per_step"] / 3600,
                budget_fraction=0.75, rule="largest batch per task under 75% of VRAM (headroom for the Windows desktop); "
                "checkpointing on/off chosen by eyes per second")
    say(f"CHOSEN: {json.dumps(best)}")
    with open(f"{cfg.runs}/batch_probe.json", "w") as fh:
        json.dump(best, fh, indent=2)

    # ---- loader throughput at the chosen sizes
    cfg2 = load_config()
    it2 = make_loaders(M, cfg2, seed=123)
    fetch(it2, cfg2)
    t = time.perf_counter()
    for _ in range(10):
        fetch(it2, cfg2)
    tl = (time.perf_counter() - t) / 10
    say(f"loader {tl:.2f}s/step vs GPU {best['sec_per_step']:.2f}s/step -> "
        f"{'OK' if tl < best['sec_per_step'] else 'LOADER-BOUND: raise workers'}")
    del it2

    # ---- evaluation code path
    R = evaluate(m, M, cfg2, max_images=4)
    say(f"evaluation OK (untrained, numbers meaningless): {json.dumps(R['grade'], default=str)[:200]}...")
    say("SMOKE TEST PASSED")
    with open(f"{cfg.runs}/smoke_report.txt", "w") as fh:
        fh.write("\n".join(LOG))


if __name__ == "__main__":
    main()
