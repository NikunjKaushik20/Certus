"""Task losses (float32). Mirrors matlab/train/losses.m."""
import torch
import torch.nn.functional as F


def weigh(L, s):
    """Homoscedastic-uncertainty task weighting: 0.5*exp(-s)*L + 0.5*s."""
    return 0.5 * torch.exp(-s) * L + 0.5 * s


def ordinal(z, y, cfg):
    """Cumulative-link ordinal BCE: logit k models P(grade > k). k=1 == referable DR (up-weighted)."""
    z = z.float()
    K = z.shape[1]
    T = (y.view(-1, 1) > torch.arange(K, device=z.device)).float()
    w = torch.ones(K, device=z.device)
    w[1] = cfg.referable_weight
    bce = F.binary_cross_entropy_with_logits(z, T, reduction="none")
    return (bce * w).sum() / (w.sum() * z.shape[0])


def segmentation(Z, Y, V, fov, cfg):
    """Focal BCE + focal Tversky; classes not annotated for an image (V=0) are ignored.
    Z,Y: (N,C,H,W); V: (N,C); fov: (N,1,H,W)."""
    Z = Z.float()
    P = torch.sigmoid(Z)
    V = V.float()[:, :, None, None]
    W = fov * V
    bce = F.binary_cross_entropy_with_logits(Z, Y, reduction="none")
    pt = P * Y + (1 - P) * (1 - Y)
    Lf = (bce * (1 - pt) ** cfg.focal_gamma * W).sum() / W.sum().clamp_min(1)
    TP = (P * Y * fov).sum((2, 3))
    FP = (P * (1 - Y) * fov).sum((2, 3))
    FN = ((1 - P) * Y * fov).sum((2, 3))
    TI = (TP + 1) / (TP + cfg.tversky_alpha * FP + cfg.tversky_beta * FN + 1)
    Vf = V[:, :, 0, 0]
    Lt = ((1 - TI + 1e-6) ** 0.75 * Vf).sum() / Vf.sum().clamp_min(1)
    return Lf + Lt


def partial_ce(z, y):
    """y in {0,1,2}: full label; y = -1: label set {good, usable} ("not reject")."""
    z = z.float()
    M = torch.zeros_like(z)
    full = y >= 0
    M[full, y[full]] = 1
    M[~full, :2] = 1
    return -(torch.logsumexp(z.masked_fill(M == 0, -1e4), 1) - torch.logsumexp(z, 1)).mean()
