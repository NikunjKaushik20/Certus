"""Single source of truth for paths and hyperparameters (mirrors matlab/certus_config.m).

Learning rates and batch sizes are measured, not chosen: they are loaded from
runs_torch/lr_finder.json and runs_torch/batch_probe.json when those exist.
"""
import json
import os
from dataclasses import asdict, dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep the model cache inside the checkout. Hard-coding a drive letter created a directory
# literally named "D:" when this was imported on macOS, which is where inference imports it from.
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@dataclass
class Config:
    manifests: str = f"{ROOT}/Data/manifests"
    runs: str = f"{ROOT}/runs_torch"

    # geometry
    canvas: int = 1536
    tile: int = 512
    grid: int = 3
    global_size: int = 512
    seg_classes: tuple = ("MA", "HE", "EX", "SE", "OD", "VES")
    n_lesion: int = 4                 # first 4 seg channels feed the evidence vector
    num_grades: int = 5

    # model
    backbone: str = "efficientnet_b0"
    decoder_channels: tuple = (256, 128, 64, 32)   # strides 16, 8, 4, 2
    freeze_bn: bool = True            # ImageNet BN statistics; grade batches are correlated tiles
    attn_dim: int = 128
    head_hidden: int = 256
    dropout: float = 0.3
    evidence_dim: int = 12
    grad_checkpointing: bool = False  # set by the batch probe if it buys a larger batch

    # batches per optimiser step (overwritten by batch_probe.json)
    batch_grade: int = 1
    accum_grade: int = 4
    batch_seg: int = 4
    batch_quality: int = 8
    target_eyes_per_step: int = 8

    # sampling
    grade_power: float = 0.5
    dataset_power: float = 0.5
    lesion_centred: float = 0.7
    lesion_priority: tuple = (3, 2, 1, 1)
    partial_quality: float = 0.25
    use_aux_eyepacs: bool = True      # 22.5k extra eyes from many more cameras: the domain-shift lever

    # optimisation (lr / enc_lr_ratio from lr_finder.json)
    lr: float = float("nan")
    enc_lr_ratio: float = float("nan")
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.01
    grad_clip: float = 5.0
    ema_decay: float = 0.999
    epochs: int = 10
    referable_weight: float = 2.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_gamma: float = 2.0

    # LR range test
    lrf_steps: int = 200
    lrf_min: float = 1e-6
    lrf_max: float = 1.0
    lrf_ratios: tuple = (0.1, 0.3, 1.0)
    lrf_smooth: float = 0.95
    lrf_diverge: float = 4.0

    # GPU augmentation
    aug_rotate: float = 15.0
    aug_scale: float = 0.15           # cameras frame the retina differently; FOV normalisation
    aug_shift: float = 0.05           # cannot fix framing the model never saw varied
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_gamma: float = 0.3
    aug_color: float = 0.05
    aug_blur_p: float = 0.25
    aug_blur_sigma: tuple = (0.5, 2.5)
    aug_vignette_p: float = 0.3
    aug_vignette_max: float = 0.5
    aug_noise_p: float = 0.2
    aug_noise_sigma: float = 0.03

    # runtime
    workers_grade: int = 2            # grade/quality workers only read bytes; JPEG decode is on the GPU
    workers_seg: int = 2              # seg workers decode PNG masks
    workers_quality: int = 1
    val_every: int = 1000
    val_subset: int = 800
    seed: int = 42
    amp_dtype: str = "bfloat16"

    def to_json(self):
        return json.dumps(asdict(self), indent=2)


MEASURED = {
    "batch_probe.json": ["batch_grade", "accum_grade", "batch_seg", "batch_quality", "grad_checkpointing"],
    "lr_finder.json": ["lr", "enc_lr_ratio"],
}


def load_config(**overrides) -> Config:
    cfg = Config(**overrides)
    for fname, keys in MEASURED.items():
        p = os.path.join(cfg.runs, fname)
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            for k in keys:
                if k in d and k not in overrides:
                    setattr(cfg, k, d[k])
    os.makedirs(cfg.runs, exist_ok=True)
    return cfg
