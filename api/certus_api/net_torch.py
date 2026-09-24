"""The network on PyTorch: the training checkpoint (EMA weights) through the same forward_eye as
validation. Used on CUDA, and wherever onnxruntime or the ONNX export is missing.

Every method takes and returns numpy, so Engine does not know which runtime ran.
"""
import json
import os

import numpy as np
import torch

from .config import settings
from certus.model import CertusNet, WholeImageGrader   # torch_root is on sys.path (inference.py)
from . import gradcam as _gradcam


class TorchNet:
    def __init__(self, ck_path: str, cfg_from):
        self.device = self._pick_device()
        self.executor = self.device
        ck = torch.load(ck_path, map_location=self.device, weights_only=False)
        self.cfg = cfg_from(json.loads(ck["cfg"]))
        model = CertusNet(self.cfg, pretrained=False)
        model.load_state_dict(ck["ema"])                      # EMA weights: what validation scored
        self.model = model.to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.checkpoint = ck_path
        self.step = int(ck.get("step", 0))
        self.val = ck.get("val") or {}
        self.second_model = None
        self.second_size = 512
        # A GPU holds all ten tiles at once; on CPU, chunking cuts peak memory by ~0.4 GB.
        self.chunk = None if self.device == "cuda" else max(1, settings.batch_tiles)

    @staticmethod
    def _pick_device() -> str:
        want = settings.device
        if want == "cpu":
            return "cpu"
        if want == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CERTUS_DEVICE=cuda but no CUDA device is visible")
        return "cuda" if torch.cuda.is_available() else "cpu"

    def attach_second(self, sr: dict) -> bool:
        """The plain whole-image grader named in trust.json. False when none is configured."""
        if not sr.get("checkpoint"):
            return False
        path = sr["checkpoint"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(settings.torch_root), path)
        model = WholeImageGrader(pretrained=False)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model = model.to(self.device).eval()
        if self.device == "cuda":
            model = model.to(memory_format=torch.channels_last)
        self.second_model = model
        self.second_size = int(sr.get("input_size", 512))
        return True

    def _tensor(self, rgb01: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def eye(self, rgb01: np.ndarray) -> dict:
        """rgb01: (D,D,3) float32 RGB in [0,1] -> the forward_eye outputs for one eye, as numpy."""
        x = self._tensor(rgb01)
        if self.device == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=getattr(torch, self.cfg.amp_dtype)):
                out = self.model.forward_eye(x, with_maps=True)
        else:
            out = self.model.forward_eye(x, with_maps=True, chunk=self.chunk)
        return {k: out[k].float().cpu().numpy()[0]
                for k in ("grade_logits", "qual_logits", "attn", "evidence", "lesion_prob", "seg_logits")}

    @torch.no_grad()
    def second(self, rgb01: np.ndarray) -> float | None:
        """Cumulative logit k=1 of the second reader, on the unenhanced canvas it trained on."""
        if self.second_model is None:
            return None
        x = self._tensor(rgb01)
        xs = torch.nn.functional.interpolate(x.float(), size=(self.second_size,) * 2, mode="bilinear",
                                             antialias=True, align_corners=False)
        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.second_model(xs.contiguous(memory_format=torch.channels_last)).float()
        else:
            logits = self.second_model(xs)
        return float(logits[0, 1].cpu())

    def grad_cam(self, rgb01: np.ndarray, target: str = "referable") -> np.ndarray:
        return _gradcam.grad_cam(self.model, rgb01, self.cfg, target=target)
