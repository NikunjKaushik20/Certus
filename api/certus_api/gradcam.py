"""Grad-CAM for CertusNet's ordinal grade head.

The eye is fed to the model as nine 512px tiles (a 3x3 grid) plus one downscaled
512px global view (see CertusNet.forward_eye in torch/certus/model.py). There is
no single "last conv layer over the whole image" the way a plain classifier has
one -- the spatial signal lives per-tile. So Grad-CAM here is computed per tile
against the last stride-32 encoder feature map (the same tensor CertusNet.embed
average-pools into the per-tile embedding that feeds the attention-MIL grade
head), then the nine tile-CAMs are stitched back into canvas coordinates with
the model's own make_tiles/stitch_tiles geometry so the result lines up with the
lesion masks pixel-for-pixel.

Target: the ordinal grade head emits cumulative-link logits, one per grade
threshold (index k = logit for P(grade > k)). Index 1 is P(grade >= 2), i.e.
referable DR -- the clinically load-bearing decision -- so that is the default
Grad-CAM target.

Memory: a naive implementation would run all 9 tiles + the global view through
the encoder in one autograd-tracked batch of 10. On a 6GB card with no autocast
that batch's saved forward activations (EfficientNet-B0 at 512px, fp32) are
enough to exhaust the card and make cuBLAS fail to allocate a workspace during
backward. Tile embeddings are architecturally independent -- the only place
tiles interact is the attention softmax that pools them -- so this instead does
one tiny batch-of-1 grad-tracked forward per tile, reusing detached (no_grad)
embeddings and attention logits for the other eight tiles to reconstruct the
exact same softmax denominator. That gives the identical gradient a full joint
backward would (nothing is approximated -- see the derivation in _tile_grad),
with peak memory for roughly one tile instead of ten.

This needs gradients for the tile under test, so that part must not run under
torch.no_grad() -- but everything not on the path to the target logit (the
lesion decoder, the other eight tiles' baseline embeddings) is deliberately
kept under no_grad to hold memory down. A forward hook captures the encoder's
stride-32 activation for whichever tile is currently being differentiated;
it is registered and removed once per tile in a finally block, so the shared
model instance is never left with an attached hook. Model is used in eval()
throughout and its training flag is restored on return.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from certus.model import CertusNet, evidence, make_tiles, stitch_tiles

_TARGETS = {"referable": 1, "any_dr": 0, "moderate": 2, "proliferative": 3}


def _tap_module(model: CertusNet, stride: int):
    """The persistent backbone submodule producing the stride-`stride` tap.

    CertusNet._stages() rebuilds its nn.Sequential wrappers on every call (it is
    called fresh inside encode()), so a hook attached to one of those wrappers
    would never fire during a later, separate encode() call. Hook the real,
    persistent backbone submodule instead: index 0 is (conv_stem, bn1), the last
    index is (conv_head, bn2), everything between is one EfficientNet block.
    """
    idx = model.tap_idx[stride]
    n_stages = len(model.backbone.blocks) + 2
    if idx == 0:
        return model.backbone.bn1
    if idx == n_stages - 1:
        return model.backbone.bn2
    return model.backbone.blocks[idx - 1]


def _tile_grad(model, tile_i, attn_logits_const, emb_const, emb_g, ev, idx, tap):
    """Gradient of the target logit w.r.t. the stride-32 activation of one tile.

    tile_i:            (1,3,t,t) pixels for this tile only.
    attn_logits_const:  (nt,) detached GatedAttention logits for ALL tiles, computed
                        once up front from the cheap no_grad pass.
    emb_const:          (nt,E) detached tile embeddings for ALL tiles, same pass.
    i:                  index of this tile within the nt tiles (0-based).

    Derivation of why reusing detached values for the other 8 tiles is exact,
    not an approximation: pooled = sum_j softmax(attn_logits)_j * emb_j. Each
    emb_j depends only on tile j's own pixels (the backbone has no cross-tile
    mixing) -- so d(emb_j)/d(tile_i) is exactly zero for j != i, and emb_j can be
    treated as a true constant when differentiating w.r.t. tile i. The only
    place tile i can influence the pooled vector through another tile's term is
    via the shared softmax denominator (attn_logits_i enters every a_j through
    the normaliser) -- and that coupling is reproduced exactly below by keeping
    attn_logits_i symbolic (from tile i's live forward pass) inside the same
    softmax that also holds the other eight constants.
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook(_m, _inp, out):
        captured["act"] = out

    handle = tap.register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            feats_i = model.encode(tile_i)
            emb_i = model.embed(feats_i)                      # (1, E), grad-tracked
            attn_logit_i = model.attn(emb_i).view(1)          # (1,), grad-tracked

            i = attn_logits_const["i"]
            logits_vec = torch.cat([attn_logits_const["vals"][:i], attn_logit_i,
                                    attn_logits_const["vals"][i + 1:]])
            a = torch.softmax(logits_vec.float(), dim=0)              # (nt,)

            emb_all = torch.cat([emb_const[:i], emb_i.float(), emb_const[i + 1:]], dim=0)  # (nt,E)
            pooled = (a.unsqueeze(-1) * emb_all).sum(0, keepdim=True)  # (1,E)

            logits = model.grade_head(torch.cat([pooled, emb_g, ev], 1))
            target_logit = logits[0, idx]

            act = captured["act"]
            grad = torch.autograd.grad(target_logit, act)[0]
        return act.detach(), grad.detach()
    finally:
        handle.remove()


def grad_cam(model: CertusNet, canvas_rgb01: np.ndarray, cfg, target: str | int = "referable") -> np.ndarray:
    """Grad-CAM heatmap for one eye canvas.

    model:         a CertusNet, any device -- eval mode throughout, restored to
                   whatever training flag it had on return; no hooks left attached.
    canvas_rgb01:  (D, D, 3) float array, RGB, values in [0, 1] -- the same canvas
                   Engine.analyse feeds to forward_eye (Engine works in BGR/uint8;
                   convert with cv2.cvtColor(..., cv2.COLOR_BGR2RGB) and /255 first).
    cfg:           the model's Config (has .grid, .tile, .global_size, .n_lesion).
    target:        "referable" (default, index 1 = P(grade>=2), the referral
                   decision), one of "any_dr"/"moderate"/"proliferative", or a
                   raw int logit index.

    Returns a float32 (D, D) array normalised to [0, 1], zero outside the field
    of view. High values mark canvas regions whose stride-32 encoder features
    pushed the target logit up.
    """
    if canvas_rgb01.ndim != 3 or canvas_rgb01.shape[2] != 3:
        raise ValueError("canvas_rgb01 must be (D, D, 3) RGB in [0,1]")
    idx = _TARGETS[target] if isinstance(target, str) else int(target)

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    D = canvas_rgb01.shape[0]
    x = torch.from_numpy(np.ascontiguousarray(canvas_rgb01)).permute(2, 0, 1).unsqueeze(0)
    x = x.to(device=device, dtype=torch.float32)

    g, t = cfg.grid, cfg.tile
    nt = g * g
    tap = _tap_module(model, 32)

    try:
        # ---- cheap constants pass: everything that does NOT need grad w.r.t. any single tile.
        with torch.no_grad():
            tiles = make_tiles(x, g, t)
            glob = F.interpolate(x, size=(cfg.global_size,) * 2, mode="bilinear",
                                 antialias=True, align_corners=False)
            feats = model.encode(torch.cat([tiles, glob]))
            emb = model.embed(feats)
            emb_t, emb_g = emb[:nt].float(), emb[nt:].float()
            attn_logits = model.attn(emb_t).view(nt).float()

            fov_t = (tiles.amax(1, keepdim=True) > 0.02).float()
            seg = model.decode({s: f[:nt] for s, f in feats.items()})
            P = torch.sigmoid(seg[:, :cfg.n_lesion].float()) * fov_t
            Pc = stitch_tiles(P, g, t)
            ev = evidence(Pc)

        cams = []
        for i in range(nt):
            # requires_grad_(True) so the encoder activation captured by the hook IS in the
            # computation graph of target_logit. Without this, act.grad_fn is None and
            # torch.autograd.grad(target_logit, act) raises RuntimeError.
            # detach().clone() first so we are NOT differentiating through the shared `tiles`
            # tensor -- we want the gradient w.r.t. this tile's own pixels only.
            tile_i = tiles[i:i + 1].detach().clone().requires_grad_(True)
            act_i, grad_i = _tile_grad(model, tile_i, {"i": i, "vals": attn_logits}, emb_t, emb_g, ev, idx, tap)
            weight = grad_i.mean(dim=(2, 3), keepdim=True)             # (1,C,1,1) channel importance
            cam = F.relu((weight * act_i).sum(dim=1, keepdim=True))    # (1,1,h,w)
            cam = F.interpolate(cam, size=(t, t), mode="bilinear", align_corners=False)
            cams.append(cam)

        cam_stack = torch.cat(cams, dim=0)                             # (nt,1,t,t), tile order == make_tiles
        canvas_cam = stitch_tiles(cam_stack, g, t)[0, 0]                # (D, D)
        if canvas_cam.shape[0] != D:
            canvas_cam = F.interpolate(canvas_cam[None, None], size=(D, D), mode="bilinear",
                                       align_corners=False)[0, 0]
        heat = canvas_cam.cpu().numpy().astype(np.float32)
    finally:
        model.train(was_training)

    fov_mask = (canvas_rgb01.max(axis=2) > 0.02)
    heat = heat * fov_mask
    m = heat.max()
    if m > 1e-8:
        heat = heat / m
    return heat.astype(np.float32)
