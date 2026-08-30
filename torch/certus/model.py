"""One network: ImageNet EfficientNet-B0 encoder shared by a U-Net lesion decoder,
an attention-MIL grade head (with stop-gradient lesion evidence) and a quality head."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def make_tiles(x, g, t):
    n, c = x.shape[:2]
    return x.view(n, c, g, t, g, t).permute(0, 2, 4, 1, 3, 5).reshape(n * g * g, c, t, t)


def stitch_tiles(x, g, t):
    c = x.shape[1]
    n = x.shape[0] // (g * g)
    return x.view(n, g, g, c, t, t).permute(0, 3, 1, 4, 2, 5).reshape(n, c, g * t, g * t)


def conv_gn(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.GroupNorm(min(8, cout), cout), nn.SiLU(inplace=True))


class GatedAttention(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.V, self.U, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)

    def forward(self, h):
        return self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)


def mlp(din, hidden, dout, p):
    return nn.Sequential(nn.Linear(din, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p),
                         nn.Linear(hidden, dout))


class CertusNet(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(cfg.backbone, pretrained=pretrained, num_classes=0, global_pool="")
        pc = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(pc["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(pc["std"]).view(1, 3, 1, 1), persistent=False)
        self._measure_taps()

        C = cfg.decoder_channels
        self.lat = nn.Conv2d(self.tap_ch[32], C[0], 1)
        self.dec = nn.ModuleList()
        self.skip = nn.ModuleList()
        prev = C[0]
        for s, c in zip((16, 8, 4, 2), C):
            self.skip.append(nn.Conv2d(self.tap_ch[s], c, 1))
            self.dec.append(nn.Sequential(conv_gn(prev + c, c), conv_gn(c, c)))
            prev = c
        self.seg_out = nn.Conv2d(prev, len(cfg.seg_classes), 1)
        nn.init.constant_(self.seg_out.bias, -4.6)             # prior p≈0.01 for rare lesion pixels

        E = self.tap_ch[32]
        self.embed_dim = E
        self.attn = GatedAttention(E, cfg.attn_dim)
        self.grade_head = mlp(2 * E + cfg.evidence_dim, cfg.head_hidden, cfg.num_grades - 1, cfg.dropout)
        self.qual_head = mlp(E, cfg.head_hidden, 3, cfg.dropout)
        self.logvar = nn.Parameter(torch.zeros(3))             # learned task weights: grade, seg, quality

    # -------------------------------------------------------------- encoder
    def _stages(self):
        b = self.backbone
        stages = [nn.Sequential(b.conv_stem, b.bn1)] + list(b.blocks)
        if hasattr(b, "bn2"):          # EfficientNet/MobileNetV2 head; MobileNetV3's conv_head runs after pooling
            stages.append(nn.Sequential(b.conv_head, b.bn2))
        return stages

    @torch.no_grad()
    def _measure_taps(self):
        """Pick the last stage output at each stride by running a dummy tile (no hard-coded indices)."""
        T = self.cfg.tile
        x = torch.zeros(1, 3, T, T)
        shapes = []
        for st in self._stages():
            x = st(x)
            shapes.append(x.shape)
        self.tap_idx, self.tap_ch = {}, {}
        for s in (2, 4, 8, 16, 32):
            idx = max(i for i, sh in enumerate(shapes) if sh[-1] == T // s)
            self.tap_idx[s], self.tap_ch[s] = idx, shapes[idx][1]

    def encode(self, x):
        x = (x - self.mean) / self.std
        feats = {}
        want = {i: s for s, i in self.tap_idx.items()}
        for i, st in enumerate(self._stages()):
            if self.cfg.grad_checkpointing and self.training and x.requires_grad:
                x = checkpoint(st, x, use_reentrant=False)
            else:
                x = st(x)
            if i in want:
                feats[want[i]] = x
        return feats

    def embed(self, feats):
        return feats[32].mean((2, 3))

    def decode(self, feats):
        y = self.lat(feats[32])
        for (s, blk, sk) in zip((16, 8, 4, 2), self.dec, self.skip):
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, sk(feats[s])], 1))
        return F.interpolate(self.seg_out(y), scale_factor=2, mode="bilinear", align_corners=False)

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_bn:                                 # keep ImageNet running statistics
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    # -------------------------------------------------------------- task forwards
    def forward_seg(self, x):
        return self.decode(self.encode(x))

    def forward_quality(self, x):
        return self.qual_head(self.embed(self.encode(x)))

    def forward_eye(self, xc, with_maps=False):
        """xc: (n,3,D,D) canvases in [0,1] -> ordinal grade logits (n,4), quality logits, extras."""
        cfg = self.cfg
        g, t = cfg.grid, cfg.tile
        n = xc.shape[0]
        nt = n * g * g
        tiles = make_tiles(xc, g, t)
        glob = F.interpolate(xc, size=(cfg.global_size,) * 2, mode="bilinear", antialias=True, align_corners=False)
        feats = self.encode(torch.cat([tiles, glob]))
        emb = self.embed(feats)
        emb_t, emb_g = emb[:nt], emb[nt:]
        seg = self.decode({s: f[:nt] for s, f in feats.items()})        # decoder on tiles only

        a = torch.softmax(self.attn(emb_t).view(n, g * g).float(), dim=1)
        pooled = (emb_t.view(n, g * g, -1).float() * a.unsqueeze(-1)).sum(1)
        fov = (tiles.amax(1, keepdim=True) > 0.02).float()
        P = torch.sigmoid(seg[:, :cfg.n_lesion].float()).detach() * fov   # stop-gradient evidence
        Pc = stitch_tiles(P, g, t)
        ev = evidence(Pc)
        logits = self.grade_head(torch.cat([pooled, emb_g.float(), ev], 1))
        out = {"grade_logits": logits, "qual_logits": self.qual_head(emb_g), "attn": a, "evidence": ev}
        if with_maps:
            out["lesion_prob"] = Pc
            out["seg_logits"] = stitch_tiles(seg.float(), g, t)
        return out


def evidence(Pc):
    """12 evidence numbers per eye from stitched lesion probabilities (n,4,D,D).
    1-4 log1p(soft area/100) · 5-8 peak probability · 9-12 HE mass per quadrant, sorted."""
    n, _, D, _ = Pc.shape
    h = D // 2
    mass = Pc.sum((2, 3))
    peak = Pc.amax((2, 3))
    he = Pc[:, 1]
    q = torch.stack([he[:, :h, :h].sum((1, 2)), he[:, :h, h:].sum((1, 2)),
                     he[:, h:, :h].sum((1, 2)), he[:, h:, h:].sum((1, 2))], 1)
    q = q.sort(1, descending=True).values
    return torch.cat([torch.log1p(mass / 100), peak, torch.log1p(q / 100)], 1)


def param_groups(model, cfg, enc_ratio):
    """AdamW groups: pretrained encoder at lr*enc_ratio, new layers at lr; no decay on 1-D params."""
    groups = {}
    for name, p in model.named_parameters():
        decay = cfg.weight_decay if p.ndim > 1 else 0.0
        if name == "logvar":
            key = ("logvar", 1.0, 0.0)
        elif name.startswith("backbone."):
            key = ("enc", enc_ratio, decay)
        else:
            key = ("new", 1.0, decay)
        groups.setdefault(key, []).append(p)
    return [{"params": ps, "name": k[0], "lr_mult": k[1], "weight_decay": k[2]} for k, ps in groups.items()]


class WholeImageGrader(nn.Module):
    """The second reader: EfficientNet-B0 on the whole canvas at 512 px, four cumulative logits.

    No tiles, no lesions, no evidence. It exists because it is more accurate than CertusNet on its
    own and makes different mistakes; the deployed decision only auto-refers or auto-clears an eye
    when both agree (torch/second_reader.py). Trained by torch/train_baseline.py.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=4)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)
