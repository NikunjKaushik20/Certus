"""The network on onnxruntime, for CPU servers. Never imports torch.

torch and timm alone hold ~500 MB resident before a single weight is loaded, which is half of a
1 GB VM. This path serves the same exported model (torch/export_onnx.py; every graph was checked
against PyTorch to 5e-5 at export) with onnxruntime and numpy:

  certus_enc.onnx     tile (n,3,512,512) -> seg_logits (n,6,512,512), emb (n,E)
  certus_feat.onnx    tile -> stride-32 activation (n,E,16,16); emb is its spatial mean. Used for
                      the global view (no decoder needed there) and for Grad-CAM.
  certus_second.onnx  the second reader, whole canvas at 512 px
  certus_heads.npz    attention, grade and quality heads, run here in numpy
  certus_runtime.json config, step and validation metrics of the checkpoint

Tiling, stitching, the evidence vector and the antialiased global resize mirror forward_eye in
torch/certus/model.py line for line; api/check_onnx_parity.py holds the two runtimes to the same
grades, probabilities and lesion counts.
"""
import hashlib
import json
import math
import os

import numpy as np

from .config import settings

_TARGETS = {"referable": 1, "any_dr": 0, "moderate": 2, "proliferative": 3}
_LN_EPS = 1e-5                                  # nn.LayerNorm default
_erf = np.vectorize(math.erf, otypes=[np.float32])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _aa_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) weights of PyTorch's antialiased bilinear resize (align_corners=False).

    The same triangle filter widened by the scale factor that ATen's _upsample_bilinear2d_aa uses,
    so the global view and the second reader's input match what the models were trained on."""
    scale = n_in / n_out
    support = scale if scale >= 1.0 else 1.0
    inv = 1.0 / scale if scale >= 1.0 else 1.0
    w = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        center = scale * (i + 0.5)
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        j = np.arange(lo, hi)
        k = np.clip(1.0 - np.abs((j - center + 0.5) * inv), 0.0, None)
        w[i, lo:hi] = k / k.sum()
    return w.astype(np.float32)


class OnnxNet:
    def __init__(self, onnx_dir: str, checkpoint: str, cfg_from):
        import onnxruntime as ort
        self._ort = ort
        self.dir = onnx_dir
        rt = json.load(open(os.path.join(onnx_dir, "certus_runtime.json")))
        self.cfg = cfg_from(rt["cfg"])
        self.step = int(rt["step"])
        self.val = rt.get("val") or {}
        self.checkpoint = checkpoint
        self.device, self.executor = "cpu", "onnxruntime-cpu"

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = settings.threads            # 0 = one per physical core
        so.inter_op_num_threads = 1
        # The arena keeps every buffer it ever grew to. Without it, memory between jobs falls back
        # to the weights, which is what lets a 1 GB machine hold the API and a job at once.
        so.enable_cpu_mem_arena = False
        self._so = so
        self.enc = self._session("certus_enc.onnx")
        self.feat = self._session("certus_feat.onnx")
        self.second_sess = None

        h = np.load(os.path.join(onnx_dir, "certus_heads.npz"))
        self.h = {k: h[k].astype(np.float32) for k in h.files}
        c = self.cfg
        self.g, self.t, self.D = c.grid, c.tile, c.canvas
        self.chunk = max(1, settings.batch_tiles)
        self._w_glob = _aa_weights(self.D, c.global_size)
        self._w_second = None
        self._last = None                                     # (canvas digest, emb_t, emb_g, ev)

    def _session(self, name: str):
        return self._ort.InferenceSession(os.path.join(self.dir, name), self._so,
                                          providers=["CPUExecutionProvider"])

    def attach_second(self, sr: dict) -> bool:
        if not sr.get("checkpoint"):
            return False
        path = os.path.join(self.dir, "certus_second.onnx")
        if not os.path.exists(path):
            raise RuntimeError(f"trust.json names a second reader but {path} is missing; "
                               "run torch/export_onnx.py --only second")
        # certus_meta.json records the cuts the second reader was exported with; a different pair in
        # trust.json means the graph is from another fit, and its logits would be read against the
        # wrong band.
        meta = json.load(open(os.path.join(self.dir, "certus_meta.json")))
        exported = meta.get("second_reader") or {}
        for k in ("clear_below_logit", "refer_above_logit"):
            if k in exported and abs(float(exported[k]) - float(sr[k])) > 1e-6:
                raise RuntimeError(f"certus_second.onnx was exported for a different second-reader fit "
                                   f"({k} {exported[k]} vs trust.json {sr[k]}); re-export it")
        self.second_sess = self._session("certus_second.onnx")
        size = int(sr.get("input_size", 512))
        self._w_second = self._w_glob if size == self.cfg.global_size else _aa_weights(self.D, size)
        return True

    # ------------------------------------------------------------ geometry
    def _tiles(self, x: np.ndarray) -> np.ndarray:
        """(3,D,D) -> (g*g,3,t,t), row-major tile order, as make_tiles."""
        g, t = self.g, self.t
        return np.ascontiguousarray(x.reshape(3, g, t, g, t).transpose(1, 3, 0, 2, 4).reshape(g * g, 3, t, t))

    def _stitch(self, x: np.ndarray) -> np.ndarray:
        """(g*g,c,t,t) -> (c,D,D), as stitch_tiles."""
        g, t = self.g, self.t
        c = x.shape[1]
        return np.ascontiguousarray(x.reshape(g, g, c, t, t).transpose(2, 0, 3, 1, 4).reshape(c, g * t, g * t))

    @staticmethod
    def _resize(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """(3,D,D) -> (3,s,s) with separable weights w (s,D)."""
        return np.ascontiguousarray(np.einsum("sd,cdk,tk->cst", w, x, w, optimize=True), dtype=np.float32)

    @staticmethod
    def _evidence(Pc: np.ndarray) -> np.ndarray:
        """12 numbers from stitched lesion probabilities (4,D,D), as certus.model.evidence."""
        h = Pc.shape[-1] // 2
        mass = Pc.sum((1, 2), dtype=np.float32)
        peak = Pc.max((1, 2))
        he = Pc[1]
        q = np.array([he[:h, :h].sum(), he[:h, h:].sum(), he[h:, :h].sum(), he[h:, h:].sum()], np.float32)
        q = -np.sort(-q)
        return np.concatenate([np.log1p(mass / 100), peak, np.log1p(q / 100)]).astype(np.float32)

    # ------------------------------------------------------------ heads (numpy)
    def _attn(self, e: np.ndarray):
        h = self.h
        tv = np.tanh(e @ h["attn.V.weight"].T + h["attn.V.bias"])
        su = _sigmoid(e @ h["attn.U.weight"].T + h["attn.U.bias"])
        s = (tv * su) @ h["attn.w.weight"].T + h["attn.w.bias"]
        return s[:, 0], tv, su

    def _mlp(self, name: str, z: np.ndarray):
        h = self.h
        u = z @ h[f"{name}.0.weight"].T + h[f"{name}.0.bias"]
        mu = u.mean(-1, keepdims=True)
        sd = np.sqrt(u.var(-1, keepdims=True) + _LN_EPS)
        nhat = (u - mu) / sd
        v = nhat * h[f"{name}.1.weight"] + h[f"{name}.1.bias"]
        gv = 0.5 * v * (1.0 + _erf(v / math.sqrt(2.0)))                # exact GELU, as nn.GELU()
        return gv @ h[f"{name}.4.weight"].T + h[f"{name}.4.bias"], (nhat, sd, v)

    def _grade(self, emb_t, emb_g, ev):
        s, tv, su = self._attn(emb_t)
        a = np.exp(s - s.max())
        a /= a.sum()
        pooled = (a[:, None] * emb_t).sum(0, keepdims=True)
        z = np.concatenate([pooled, emb_g, ev[None]], 1)
        logits, ln = self._mlp("grade_head", z)
        return logits[0], a, (tv, su, pooled, ln)

    # ------------------------------------------------------------ forward
    def _encode_tiles(self, tiles: np.ndarray):
        n = tiles.shape[0]
        seg = np.empty((n, len(self.cfg.seg_classes), self.t, self.t), np.float32)
        emb = None
        for i in range(0, n, self.chunk):
            s, e = self.enc.run(None, {"tile": tiles[i:i + self.chunk]})
            if emb is None:
                emb = np.empty((n, e.shape[1]), np.float32)
            seg[i:i + len(s)], emb[i:i + len(e)] = s, e
        return seg, emb

    def _digest(self, rgb01: np.ndarray) -> str:
        return hashlib.blake2b(rgb01.tobytes(), digest_size=16).hexdigest()

    def eye(self, rgb01: np.ndarray) -> dict:
        x = np.ascontiguousarray(rgb01.transpose(2, 0, 1), dtype=np.float32)
        tiles = self._tiles(x)
        seg, emb_t = self._encode_tiles(tiles)
        glob = self._resize(x, self._w_glob)[None]
        emb_g = self.feat.run(None, {"tile": glob})[0].mean((2, 3))

        fov = tiles.max(1, keepdims=True) > 0.02
        del tiles
        P = _sigmoid(seg[:, :self.cfg.n_lesion]) * fov
        Pc = self._stitch(P)
        del P
        ev = self._evidence(Pc)
        logits, a, _ = self._grade(emb_t, emb_g, ev)
        qual, _ = self._mlp("qual_head", emb_g)
        self._last = (self._digest(rgb01), emb_t, emb_g, ev)   # Grad-CAM of this eye reuses these
        return {"grade_logits": logits, "qual_logits": qual[0], "attn": a, "evidence": ev,
                "lesion_prob": Pc, "seg_logits": self._stitch(seg)}

    def second(self, rgb01: np.ndarray) -> float | None:
        if self.second_sess is None:
            return None
        x = np.ascontiguousarray(rgb01.transpose(2, 0, 1), dtype=np.float32)
        xs = self._resize(x, self._w_second)[None]
        return float(self.second_sess.run(None, {"image": xs})[0][0, 1])

    # ------------------------------------------------------------ Grad-CAM
    def _emb_grads(self, emb_t, emb_g, ev, idx: int) -> np.ndarray:
        """d(logit idx)/d(emb_i) for every tile i, by hand through the grade MLP and the attention
        pooling. Only tile i's own embedding varies; the global view and the evidence are held
        constant, exactly as api/certus_api/gradcam.py differentiates."""
        h = self.h
        logits, a, (tv, su, pooled, (nhat, sd, v)) = self._grade(emb_t, emb_g, ev)
        phi = np.exp(-0.5 * v * v) / math.sqrt(2 * math.pi)
        dv = h["grade_head.4.weight"][idx] * (0.5 * (1.0 + _erf(v / math.sqrt(2.0))) + v * phi)
        dn = dv * h["grade_head.1.weight"]
        du = (dn - dn.mean(-1, keepdims=True) - nhat * (dn * nhat).mean(-1, keepdims=True)) / sd
        gp = (du @ h["grade_head.0.weight"])[0, :emb_t.shape[1]]           # d logit / d pooled
        ds = a * (emb_t @ gp - float(pooled[0] @ gp))                       # d logit / d attn score
        w = h["attn.w.weight"][0]
        dh = (ds[:, None] * (w * su * (1 - tv * tv))) @ h["attn.V.weight"] + \
             (ds[:, None] * (w * tv * su * (1 - su))) @ h["attn.U.weight"]
        return a[:, None] * gp[None] + dh

    def grad_cam(self, rgb01: np.ndarray, target: str = "referable") -> np.ndarray:
        import cv2
        idx = _TARGETS[target] if isinstance(target, str) else int(target)
        x = np.ascontiguousarray(rgb01.transpose(2, 0, 1), dtype=np.float32)
        tiles = self._tiles(x)
        if self._last is not None and self._last[0] == self._digest(rgb01):
            _, emb_t, emb_g, ev = self._last
        else:                                                   # a canvas this process never graded
            seg, emb_t = self._encode_tiles(tiles)
            fov = tiles.max(1, keepdims=True) > 0.02
            ev = self._evidence(self._stitch(_sigmoid(seg[:, :self.cfg.n_lesion]) * fov))
            del seg
            emb_g = self.feat.run(None, {"tile": self._resize(x, self._w_glob)[None]})[0].mean((2, 3))
        grads = self._emb_grads(emb_t, emb_g, ev, idx)                      # (nt, E)

        t = self.t
        cams = np.empty((tiles.shape[0], 1, t, t), np.float32)
        for i in range(0, tiles.shape[0], self.chunk):
            act = self.feat.run(None, {"tile": tiles[i:i + self.chunk]})[0]  # (k,E,h,w)
            hw = act.shape[2] * act.shape[3]
            for j in range(act.shape[0]):
                # emb is the spatial mean of act, so d(logit)/d(act) is d(logit)/d(emb) / hw at
                # every position, and its spatial mean -- the Grad-CAM channel weight -- is the same.
                cam = np.maximum(np.tensordot(grads[i + j] / hw, act[j], axes=1), 0.0)
                cams[i + j, 0] = cv2.resize(cam, (t, t), interpolation=cv2.INTER_LINEAR)
        heat = self._stitch(cams)[0]
        heat *= rgb01.max(axis=2) > 0.02
        m = heat.max()
        if m > 1e-8:
            heat /= m
        return heat.astype(np.float32)
