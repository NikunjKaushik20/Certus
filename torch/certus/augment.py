"""GPU augmentation (mirrors matlab/train/gpuAugment.m). All ops are batched on CUDA.

mode "full"    grading canvases: flip, small rotation, acquisition degradations
     "patch"   seg patches: + 90° turns, masks transformed with nearest sampling
     "quality" quality views: geometric + mild colour only (degradations would change the label)
"""
import math

import kornia
import torch
import torch.nn.functional as F


def _warp(x, ang, scale, shift, mode):
    """Rotate, zoom and translate in one sampling pass (two passes would blur twice).

    theta maps output coords -> input coords, so a zoom *in* is a factor < 1 here.
    """
    c, s = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([c, -s, shift[:, 0]], 1),
                         torch.stack([s, c, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def augment(x, y, cfg, mode, gen):
    N, dev = x.shape[0], x.device
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    fov = (x.amax(1, keepdim=True) > 0.02).float()
    parts = [x, fov] + ([y] if y is not None else [])
    sizes = [p.shape[1] for p in parts]
    z = torch.cat(parts, 1)

    flip = r(N) < 0.5
    z = torch.where(flip[:, None, None, None], z.flip(-1), z)
    if mode == "patch":
        k = (r(N) * 4).long()
        z = torch.stack([torch.rot90(z[i], int(k[i]), (1, 2)) for i in range(N)])
    ang = (2 * r(N) - 1) * math.radians(cfg.aug_rotate)
    scale = 1 + (2 * r(N) - 1) * cfg.aug_scale
    shift = (2 * r(N, 2) - 1) * cfg.aug_shift
    x, rest = z[:, :3], z[:, 3:]
    x = _warp(x, ang, scale, shift, "bilinear")
    rest = _warp(rest, ang, scale, shift, "nearest")
    fov = rest[:, :1]
    y = rest[:, 1:] if y is not None else None

    if mode != "quality":
        b = r(N) < cfg.aug_blur_p
        if b.any():
            lo, hi = cfg.aug_blur_sigma
            sig = (lo + r(int(b.sum())) * (hi - lo)).unsqueeze(1).repeat(1, 2)
            ks = 2 * math.ceil(3 * hi) + 1
            x[b] = kornia.filters.gaussian_blur2d(x[b], (ks, ks), sig)
        v = r(N) < cfg.aug_vignette_p
        if v.any():
            H = x.shape[-1]
            lin = torch.linspace(-1, 1, H, device=dev)
            yy, xx = torch.meshgrid(lin, lin, indexing="ij")
            c = (2 * r(N, 2) - 1) * 0.5
            d2 = (xx[None] - c[:, 0, None, None]) ** 2 + (yy[None] - c[:, 1, None, None]) ** 2
            strength = cfg.aug_vignette_max * r(N) * v.float()
            x = x * (1 - strength[:, None, None, None] * (d2 / 2).clamp(max=1)[:, None])
        nz = r(N) < cfg.aug_noise_p
        if nz.any():
            x = x + (cfg.aug_noise_sigma * r(N) * nz.float())[:, None, None, None] * \
                torch.randn(x.shape, device=dev, generator=gen)

    s = 0.5 if mode == "quality" else 1.0
    u = lambda *sh: 2 * r(*sh) - 1
    bri = 1 + s * cfg.aug_brightness * u(N, 1, 1, 1)
    con = 1 + s * cfg.aug_contrast * u(N, 1, 1, 1)
    gam = torch.exp(s * cfg.aug_gamma * u(N, 1, 1, 1))
    col = 1 + s * cfg.aug_color * u(N, 3, 1, 1)
    m = (x * fov).sum((2, 3), keepdim=True) / fov.sum((2, 3), keepdim=True).clamp_min(1)
    x = ((x - m) * con + m) * bri * col
    x = x.clamp(min=0).pow(gam).clamp(max=1) * fov
    return x, y, fov
