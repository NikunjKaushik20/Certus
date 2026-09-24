"""Export a trained Certus checkpoint to ONNX for MATLAB (importNetworkFromONNX).

Four graphs, matching the MATLAB model struct (matlab/model/importCertusOnnx.m):
  certus_enc.onnx    tile (n,3,512,512) in [0,1] -> seg_logits (n,6,512,512), emb (n,E)
  certus_attn.onnx   emb (n,E) -> score (n,1)
  certus_grade.onnx  [pooled, global, evidence] (n,2E+12) -> ordinal logits (n,4)
  certus_qual.onnx   emb (n,E) -> quality logits (n,3)
  certus_feat.onnx   tile (n,3,512,512) -> stride-32 activation (n,E,16,16), for Grad-CAM only.
                     emb is its spatial mean, so Grad-CAM needs only this map and d(logit)/d(emb).
Tiling, attention softmax and the evidence vector stay in MATLAB code (gradeForward.m),
exactly as in training. Every graph is checked against PyTorch with onnxruntime.
The API's CPU path (api/certus_api/net_onnx.py) runs certus_enc, certus_feat and certus_second
with onnxruntime, plus certus_runtime.json and certus_heads.npz (see export_runtime).
Usage: python export_onnx.py runs_torch/train_<stamp>/best.pt [out_dir] [--only feat|second|runtime]
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import load_model  # noqa: E402


class Enc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        f = self.m.encode(x)
        return self.m.decode(f), self.m.embed(f)


class Feat(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.encode(x)[32]


class Head(nn.Module):
    def __init__(self, h, squeeze=False):
        super().__init__()
        self.h, self.squeeze = h, squeeze

    def forward(self, x):
        y = self.h(x)
        return y.unsqueeze(-1) if self.squeeze else y


def export(module, dummy, path, inputs, outputs):
    module = module.float().eval().cpu()
    torch.onnx.export(module, dummy, path, input_names=inputs, output_names=outputs, opset_version=17,
                      dynamic_axes={n: {0: "n"} for n in inputs + outputs}, dynamo=False)
    with torch.no_grad():
        ref = module(dummy)
    ref = ref if isinstance(ref, tuple) else (ref,)
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {inputs[0]: dummy.numpy()})
    err = max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))
    print(f"{os.path.basename(path)}: max |torch - onnxruntime| = {err:.2e}")
    assert err < 1e-3, "ONNX mismatch"
    return err


def export_second(ck, out):
    """The second reader named in trust.json (torch/second_reader.py), as certus_second.onnx.

    Only the graph and its two cuts are added; the Certus graphs are left alone."""
    from certus.model import WholeImageGrader
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    t = json.load(open(os.path.join(os.path.dirname(ck), "trust.json")))
    sr = t["second_reader"]
    m = WholeImageGrader(pretrained=False)
    m.load_state_dict(torch.load(os.path.join(root, sr["checkpoint"]), map_location="cpu"))
    size = int(sr["input_size"])
    err = export(m, torch.rand(1, 3, size, size), f"{out}/certus_second.onnx", ["image"], ["logits"])
    meta_path = f"{out}/certus_meta.json"
    meta = json.load(open(meta_path))
    meta.setdefault("max_abs_err", {})["second"] = err
    meta["second_reader"] = {k: sr[k] for k in ("input_size", "clear_below_logit", "refer_above_logit", "rule")}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print("->", out)


def export_runtime(ck, out):
    """What the API's onnxruntime path needs that no graph carries, so it never imports torch:

      certus_runtime.json  the training config, step and validation metrics from the checkpoint
      certus_heads.npz     attention, grade and quality head weights. The API runs these tiny
                           heads in numpy rather than as graphs because Grad-CAM needs their
                           gradient, and differentiating a 2-layer MLP by hand is cheaper than
                           shipping autograd."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    c = torch.load(ck, map_location="cpu", weights_only=False)
    heads = {k: v.float().numpy() for k, v in c["ema"].items()
             if k.startswith(("attn.", "grade_head.", "qual_head."))}
    np.savez(f"{out}/certus_heads.npz", **heads)
    rel = os.path.relpath(os.path.abspath(ck), root).replace(os.sep, "/")
    with open(f"{out}/certus_runtime.json", "w") as fh:
        json.dump({"checkpoint": rel, "step": int(c.get("step", 0)), "cfg": json.loads(c["cfg"]),
                   "val": c.get("val") or {}}, fh, indent=2, default=float)
    print(f"certus_runtime.json, certus_heads.npz ({len(heads)} tensors) ->", out)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    args = [a for a in args if a != only]
    ck = args[0]
    out = args[1] if len(args) > 1 else os.path.join(os.path.dirname(ck), "onnx")
    os.makedirs(out, exist_ok=True)
    if only == "second":
        export_second(ck, out)
        return
    if only == "runtime":
        export_runtime(ck, out)
        return
    model, cfg, step = load_model(ck)
    model = model.float().cpu().eval()
    E = model.embed_dim
    if only == "feat":
        # Adds the Grad-CAM graph without rewriting the other four, so MATLAB's cached import
        # of those stays valid.
        err = export(Feat(model), torch.rand(1, 3, cfg.tile, cfg.tile), f"{out}/certus_feat.onnx",
                     ["tile"], ["feat32"])
        meta_path = f"{out}/certus_meta.json"
        meta = json.load(open(meta_path))
        meta.setdefault("max_abs_err", {})["feat"] = err
        json.dump(meta, open(meta_path, "w"), indent=2)
        print("->", out)
        return
    errs = {
        "enc": export(Enc(model), torch.rand(2, 3, cfg.tile, cfg.tile), f"{out}/certus_enc.onnx",
                      ["tile"], ["seg_logits", "emb"]),
        "attn": export(Head(model.attn, squeeze=True), torch.randn(9, E), f"{out}/certus_attn.onnx", ["emb"], ["score"]),
        "grade": export(Head(model.grade_head), torch.randn(2, 2 * E + cfg.evidence_dim), f"{out}/certus_grade.onnx",
                        ["features"], ["logits"]),
        "qual": export(Head(model.qual_head), torch.randn(2, E), f"{out}/certus_qual.onnx", ["emb"], ["logits"]),
        "feat": export(Feat(model), torch.rand(1, 3, cfg.tile, cfg.tile), f"{out}/certus_feat.onnx",
                       ["tile"], ["feat32"]),
    }
    meta = {"embed_dim": E, "input_scale": 1.0, "input_range": "[0,1] RGB, ImageNet normalisation inside graph",
            "seg_classes": list(cfg.seg_classes), "tile": cfg.tile, "grid": cfg.grid, "canvas": cfg.canvas,
            "global_size": cfg.global_size, "evidence_dim": cfg.evidence_dim, "max_abs_err": errs,
            "checkpoint": os.path.abspath(ck), "step": step}
    trust_path = os.path.join(os.path.dirname(ck), "trust.json")
    if os.path.exists(trust_path):
        t = json.load(open(trust_path))
        meta["temperature"] = t["temperature"]
        meta["conformal"] = {a: v["q"] for a, v in t["alphas"].items()}
        # The API stopped deciding on the conformal bands: at alpha=0.05 the clear-side band
        # auto-cleared 15.8% of referable eyes. screening_band is what actually runs, so MATLAB
        # must read the same two cuts or it will disagree with the console on the same image.
        # Lesion masks are drawn at a per-class threshold, not at 0.5: microaneurysms peak
        # near 0.02 while hard exudates sit near 0.8. MATLAB has to use the same cuts or its
        # overlay will show a different lesion count than the console for one photograph.
        meta["lesion_thresholds"] = t["lesion"]["thresholds"]
        if "screening_band" in t:
            meta["screening_band"] = {k: t["screening_band"][k]
                                      for k in ("clear_below", "refer_above")}
    with open(f"{out}/certus_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    export_runtime(ck, out)
    print("->", out)


if __name__ == "__main__":
    main()
