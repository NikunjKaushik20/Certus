"""Model runtime: preprocessing, grading, quality gate, lesion evidence.

The network is the training checkpoint (EMA weights) run through the same code path as
validation, so what the API reports and what the metrics claim cannot drift apart. The heavy
work is one forward pass per eye: 3x3 tiles of 512 px plus a downscaled global view.

Device is chosen at load time: CUDA when present, CPU otherwise. Nothing else in the backend
knows or cares which one ran.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from .config import settings

for _p in (settings.torch_root, settings.scripts_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from certus.config import Config           # noqa: E402  (path set above)
from certus.model import CertusNet, WholeImageGrader   # noqa: E402
import retina                              # noqa: E402
from . import gradcam as _gradcam          # noqa: E402  (gradcam.py lives in the api package)

LESIONS = ("MA", "HE", "EX", "SE")
QUALITY = ("good", "usable", "reject")
MIN_AREA = {"MA": 2, "HE": 4, "EX": 4, "SE": 4}       # drop single-pixel speckle

# Reliability gate. These three numbers are chosen together so the rule they produce can be stated
# in one sentence: a good photograph is decided on any camera; a merely usable photograph is
# decided only on a camera we have calibrated; anything worse goes to a person.
#
#   good image    quality 1.00 -> fitted 1.00 decide   unverified 0.80 decide
#   usable image  quality 0.75 -> fitted 0.75 decide   unverified 0.60 ABSTAIN
#   half-rejected quality 0.53 -> fitted 0.53 ABSTAIN  unverified 0.42 ABSTAIN
#
# Multiplicative, not a sum of penalties: two soft doubts are tolerated one at a time but not
# together, which is the behaviour a screening programme wants.
USABLE_WEIGHT = 0.75
# Plausible optic-disc diameter on a canvas whose side is the FOV diameter. Anything outside this
# means the disc segmentation failed, and every downstream fovea number would be scaled by it.
DISC_PX_MIN, DISC_PX_MAX = 110, 340

UNVERIFIED_CAMERA_FACTOR = 0.80
MIN_RELIABILITY = 0.70
DEFAULT_LESION_THR = {"MA": 0.02, "HE": 0.28, "EX": 0.78, "SE": 0.56}

# Fovea from the disc: offsets fitted on the IDRiD localisation train split (torch/fit_fovea.py),
# measured in FOV diameters rather than disc diameters (a segmented disc is a noisy ruler), then
# moved to the darkest point of the smoothed green channel within half a disc diameter. On IDRiD
# test: median 35.5 native px, mean 67 (the old fixed 2.5 DD rule: median 165).
_FOVEA_RULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "torch", "fovea_rule.json")
try:
    FOVEA_RULE = json.load(open(_FOVEA_RULE_PATH))
except OSError:
    FOVEA_RULE = {"ruler": "dd", "temporal": 2.5, "inferior": 0.1, "refine_radius_dd": None}


def _dark_refine(canvas_bgr, fx, fy, r):
    """Darkest point of the smoothed green channel within r px of (fx, fy): the foveal pit."""
    g = cv2.GaussianBlur(canvas_bgr[:, :, 1].astype(np.float32), (0, 0), max(2.0, r / 6))
    D = g.shape[0]
    x0, x1 = int(max(0, fx - r)), int(min(D, fx + r + 1))
    y0, y1 = int(max(0, fy - r)), int(min(D, fy + r + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.where((xx - fx) ** 2 + (yy - fy) ** 2 <= r * r, g[y0:y1, x0:x1], np.inf)
    iy, ix = np.unravel_index(np.argmin(patch), patch.shape)
    return float(x0 + ix), float(y0 + iy)


def _domain_table(t: dict) -> dict:
    """Per camera domain, keyed the way a device's domain_key is keyed.

    This used to return the evaluation splits instead -- "test" and "external_messidor2" -- which
    are not cameras. The Cameras page looks a device up by its domain_key, so the lookup could
    never hit and every camera read UNVERIFIED no matter what had been fitted for it.

    The numbers come from two places: the fit itself (temperature, referral threshold, how many
    val eyes it was fitted on) and the held-out tables (how it then behaved on test or Messidor-2).
    A domain that was measured but never fitted still belongs here -- knowing a camera's error rate
    without having a correction for it is exactly the state the console should be able to show."""
    out: dict = {}
    for name, fit in (t.get("per_domain") or {}).items():
        out[name] = {"temperature": fit.get("temperature"),
                     "threshold": fit.get("referral_threshold"),
                     "n_val": fit.get("n_val"), "fitted": True,
                     "screening_band": fit.get("screening_band")}
    for split in ("test", "external_messidor2"):
        for row in ((t.get(split) or {}).get("domain_table_alpha_0.05") or []):
            d = out.setdefault(row["domain"], {"fitted": False})
            d.update({"n": row.get("n"), "ece": row.get("ece"),
                      "abstain_rate": row.get("abstain_rate"),
                      "miss_rate": row.get("miss_rate_at_threshold"),
                      "measured_on": split})
    return out


class Engine:
    """Loads once, serves many. Thread-safe for a single worker: inference is under no_grad and
    the model is never mutated after load."""

    def __init__(self):
        self.device = self._pick_device()
        ck_path = settings.resolve_checkpoint()
        ck = torch.load(ck_path, map_location=self.device, weights_only=False)
        self.cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v)
                             for k, v in json.loads(ck["cfg"]).items()})
        model = CertusNet(self.cfg, pretrained=False)
        model.load_state_dict(ck["ema"])                      # EMA weights: what validation scored
        self.model = model.to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.checkpoint = ck_path
        self.step = int(ck.get("step", 0))
        self.metrics = self._metrics(ck)
        self.calibration = self._load_calibration()
        self.second = self._load_second_reader()

    # ---------------------------------------------------------------- setup
    @staticmethod
    def _pick_device() -> str:
        want = settings.device
        if want == "cpu":
            return "cpu"
        if want == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CERTUS_DEVICE=cuda but no CUDA device is visible")
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _metrics(ck) -> dict:
        v = ck.get("val") or {}
        g = v.get("grade") or {}
        keep = ("n", "auc_referable", "sens_at_85spec", "spec_at_85", "thr_at_85", "qwk", "ece_referable")
        out = {k: g[k] for k in keep if k in g}
        if v.get("seg_dice"):
            out["seg"] = v["seg_dice"]
        if v.get("quality"):
            out["quality"] = v["quality"]
        return out

    def _load_calibration(self) -> dict:
        """trust.json from calibrate.py when it exists; otherwise an explicit uncalibrated stance."""
        path = settings.resolve_trust()
        if not path or not os.path.exists(path):
            thr = self.metrics.get("thr_at_85", 0.5)
            # The checkpoint already carries per-lesion thresholds swept on its own validation set
            # (MA_thr, HE_thr, ...). Those beat the hard-coded constants, which are a last resort for
            # a checkpoint that never recorded any. Using 0.02 for MA when the checkpoint says 0.207
            # counts a lot of speckle as microaneurysms.
            seg = self.metrics.get("seg") or {}
            lesion = {k: float(seg.get(f"{k}_thr", DEFAULT_LESION_THR[k])) for k in LESIONS}
            return {"fitted": False, "temperature": 1.0, "referral_threshold": float(thr),
                    "lesion_thresholds": lesion, "conformal_q": {},
                    "source": ("checkpoint validation thresholds (calibrate.py has not run)"
                               if seg else "hard-coded defaults (checkpoint has no thresholds)")}
        t = json.load(open(path))
        lesion = (t.get("lesion") or {}).get("thresholds") or dict(DEFAULT_LESION_THR)
        test = (t.get("test") or {}).get("calibrated") or {}
        return {"fitted": True, "temperature": float(t.get("temperature", 1.0)),
                "referral_threshold": float(test.get("thr_at_85", self.metrics.get("thr_at_85", 0.5))),
                "lesion_thresholds": {k: float(v) for k, v in lesion.items()},
                "conformal_q": t.get("alphas") or {}, "screening_band": t.get("screening_band") or {},
                "per_domain_bands": {k: v["screening_band"] for k, v in (t.get("per_domain") or {}).items()
                                     if isinstance(v, dict) and v.get("screening_band")},
                "second_reader": t.get("second_reader"),
                "per_domain_second_bands": {k: v["second_reader_band"] for k, v in (t.get("per_domain") or {}).items()
                                            if isinstance(v, dict) and v.get("second_reader_band")},
                "source": path,
                "domains": _domain_table(t),
                "splits": {p: (t.get(p) or {}) for p in ("test", "external_messidor2")}}

    def _load_second_reader(self):
        """The plain whole-image grader named in trust.json, or None (Certus then decides alone)."""
        sr = self.calibration.get("second_reader") or {}
        if not sr.get("checkpoint"):
            return None
        path = sr["checkpoint"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(settings.torch_root), path)
        model = WholeImageGrader(pretrained=False)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model = model.to(self.device).eval()
        if self.device == "cuda":
            model = model.to(memory_format=torch.channels_last)
        return model

    @torch.no_grad()
    def _second_reader(self, x: torch.Tensor) -> dict | None:
        """x: (1,3,D,D) RGB in [0,1], the unenhanced canvas -- what the second reader trained on."""
        if self.second is None:
            return None
        size = int(self.calibration["second_reader"].get("input_size", 512))
        xs = torch.nn.functional.interpolate(x.float(), size=(size, size), mode="bilinear",
                                             antialias=True, align_corners=False)
        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.second(xs.contiguous(memory_format=torch.channels_last)).float()
        else:
            logits = self.second(xs)
        z = float(logits[0, 1].cpu())
        return {"logit": round(z, 6), "p_referable": round(float(1 / (1 + np.exp(-z))), 6)}

    # ---------------------------------------------------------------- preprocessing
    def preprocess(self, raw: bytes) -> dict:
        """Decode -> FOV-normalised square canvas, exactly as the training data was built."""
        buf = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)              # BGR, as cv2.imwrite produced in training
        if img is None:
            return {"ok": False, "error": "not a decodable image"}
        h, w = img.shape[:2]
        canvas, fov, _, info = retina.normalize_fov(img, self.cfg.canvas)
        if canvas is None:
            return {"ok": False, "error": "no field of view found", "width": w, "height": h,
                    "fov_ok": False}
        ok, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return {"ok": False, "error": "could not encode canvas"}
        return {"ok": True, "canvas": canvas, "fov": fov, "jpeg": enc.tobytes(),
                "width": w, "height": h, "fov_ok": bool(info.get("fov_ok", True)),
                "canvas_size": self.cfg.canvas}

    # ---------------------------------------------------------------- forward pass
    @torch.no_grad()
    def analyse(self, canvas_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(self.device)
        if self.device == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=getattr(torch, self.cfg.amp_dtype)):
                out = self.model.forward_eye(x, with_maps=True)
        else:
            out = self.model.forward_eye(x, with_maps=True)
        second = self._second_reader(x)

        cal = self.calibration
        logits = out["grade_logits"].float().cpu().numpy()[0]
        raw_p = _cummin(1 / (1 + np.exp(-logits)))
        cal_p = _cummin(1 / (1 + np.exp(-logits / cal["temperature"])))
        grade = int((cal_p > 0.5).sum())
        qual = torch.softmax(out["qual_logits"].float(), 1).cpu().numpy()[0]
        qual_label = QUALITY[int(qual.argmax())]

        # Adequacy evaluation and adaptive enhancement
        adequacy = retina.evaluate_adequacy(canvas_bgr)
        enhance_info = {"applied": False, "methods": [], "regraded": False}
        if qual_label == "usable":
            canvas_enh, enhance_info = retina.adaptive_enhance(canvas_bgr, quality_label="usable")
            if enhance_info.get("applied"):
                rgb_enh = cv2.cvtColor(canvas_enh, cv2.COLOR_BGR2RGB)
                x_enh = torch.from_numpy(rgb_enh).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(self.device)
                if self.device == "cuda":
                    x_enh = x_enh.contiguous(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=getattr(torch, self.cfg.amp_dtype)):
                        out_enh = self.model.forward_eye(x_enh, with_maps=True)
                else:
                    out_enh = self.model.forward_eye(x_enh, with_maps=True)

                logits_enh = out_enh["grade_logits"].float().cpu().numpy()[0]
                raw_p_enh = _cummin(1 / (1 + np.exp(-logits_enh)))
                cal_p_enh = _cummin(1 / (1 + np.exp(-logits_enh / cal["temperature"])))
                grade_enh = int((cal_p_enh > 0.5).sum())

                out = out_enh
                logits, raw_p, cal_p, grade = logits_enh, raw_p_enh, cal_p_enh, grade_enh
                enhance_info["regraded"] = True
                canvas_bgr = canvas_enh

        maps = out["lesion_prob"].float().cpu().numpy()[0]           # (4, D, D) after stop-gradient
        seg = torch.sigmoid(out["seg_logits"].float()).cpu().numpy()[0]
        structures = self._structures(seg, self.cfg.canvas, canvas_bgr,
                                      ves_thr=self.calibration["lesion_thresholds"].get("VES", 0.5))
        structures["adequacy"] = adequacy
        structures["enhancement"] = enhance_info

        findings = self._findings(maps, seg, structures)

        # Neovascularization (NVD / NVE) Assessment - decoupled from dr_grade
        structures["neovascularization"] = self._assess_neovascularization(seg, structures, findings)

        runtime = int((time.perf_counter() - t0) * 1000)
        return {
            "ordinal_probs": [round(float(v), 6) for v in cal_p],
            "ordinal_probs_raw": [round(float(v), 6) for v in raw_p],
            "dr_grade": grade,
            "p_referable": float(cal_p[1]),
            "p_referable_raw": float(raw_p[1]),
            "quality": {"label": qual_label,
                        "probs": {k: round(float(v), 6) for k, v in zip(QUALITY, qual)},
                        "adequacy": adequacy,
                        "recapture_advice": adequacy.get("recapture_advice", [])},
            "evidence": [round(float(v), 6) for v in out["evidence"].float().cpu().numpy()[0]],
            "attention": [round(float(v), 6) for v in out["attn"].float().cpu().numpy()[0]],
            "findings": findings,
            "structures": structures,
            "second_reader": second,
            "runtime_ms": runtime,
            "executor": self.device,
        }

    @staticmethod
    def _assess_neovascularization(seg: np.ndarray, structures: dict, findings: list[dict]) -> dict:
        """EXPERIMENTAL neovascularization indicator (NVD / NVE).

        No dataset we train or test on has NV annotations, so the thresholds below are hand-set
        and unvalidated. Shown to the reviewer as an indicator; never used in the grade or the
        referral decision. matlab/model/neovascularizationDetect.m applies the same rules.

        Decoupled from predicted DR grade: assessed strictly as an independent anatomical biomarker
        per International Clinical Diabetic Retinopathy (ICDR) and ETDRS standards:
        - NVD: new abnormal vessels on or within 1.5 disc diameters of the optic disc margin.
        - NVE: new vessels elsewhere in the retina, hallmarked by preretinal/vitreous hemorrhage
               and extensive microvascular proliferation.
        """
        D = seg.shape[-1]
        he_findings = [f for f in findings if f["lesion_type"] == "HE"]
        he_count = he_findings[0]["lesion_count"] if he_findings else 0
        preretinal = (he_findings[0].get("subtypes") or {}).get("preretinal_vitreous", 0) if he_findings else 0

        ves_info = structures.get("vessels") or {}
        ves_density = float(ves_info.get("density_pct", 0.0))

        nvd = False
        nve = False
        crit = []

        if structures.get("disc_found") and seg.shape[0] >= 6:
            ox, oy, dd = structures["disc_x"], structures["disc_y"], structures["disc_diameter_px"]
            Y, X = np.ogrid[:D, :D]
            dist_disc = np.hypot(X - ox, Y - oy)
            peridisc_mask = (dist_disc <= 1.5 * dd)
            ves_mask = (seg[5] > 0.5)                        # NV rule's own cut, unchanged
            peridisc_ves = int((ves_mask & peridisc_mask).sum())
            peridisc_area = int(peridisc_mask.sum())
            peridisc_density = float(peridisc_ves / max(peridisc_area, 1))

            # NVD threshold: abnormal peridisc microvascular proliferation independent of predicted grade
            if peridisc_density >= 0.15 or (peridisc_density >= 0.11 and he_count >= 4):
                nvd = True
                crit.append("Abnormal microvascular proliferation at optic disc margin (NVD)")

        # NVE threshold: preretinal/vitreous hemorrhage (hallmark of neovascular bleeding) or dense peripheral branching
        if preretinal > 0 or (he_count >= 20 and ves_density > 4.5):
            nve = True
            crit.append("Fine irregular branching / preretinal-vitreous hemorrhage elsewhere (NVE)")

        if not crit:
            crit.append("Normal vascular caliber without neovascular fronds")

        risk = float(np.clip(
            (0.60 if nvd else 0.0) + (0.40 if nve else 0.0) +
            (0.25 if preretinal > 0 else 0.0) + min(0.25, he_count * 0.008),
            0.0, 1.0
        ))

        return {
            "nvd_detected": bool(nvd),
            "nve_detected": bool(nve),
            "nv_risk_score": round(risk, 3),
            "criteria": "[experimental] " + "; ".join(crit),
            "experimental": True,
        }

    @staticmethod
    def _structures(seg: np.ndarray, D: int, canvas_bgr: np.ndarray | None = None,
                    ves_thr: float = 0.5) -> dict:
        """Locate optic disc, fovea, and vessel architecture from segmentation."""
        # Vessel extraction (channel 5 = VES)
        ves_info = {"found": False, "density_pct": 0.0, "vessel_pixels": 0, "caliber_mean_px": 0.0}
        if seg.shape[0] >= 6:
            ves_raw = seg[5] > ves_thr                   # 0.5; see VES_note in trust.json
            ves_px = int(ves_raw.sum())
            if ves_px > 50:
                fov_px = float((seg.max(axis=0) > 0.02).sum())
                density = round(100.0 * ves_px / max(fov_px, 1.0), 2)
                # Caliber estimation from area / skeleton length proxy
                ves_info = {
                    "found": True,
                    "density_pct": density,
                    "vessel_pixels": ves_px,
                    "caliber_mean_px": round(float(np.clip(ves_px / max(float(np.sqrt(ves_px * 12)), 1.0), 1.2, 8.5)), 2)
                }

        od = seg[4] > 0.5
        if od.sum() < 50:
            return {"disc_found": False, "fovea_found": False, "vessels": ves_info}
        n_cc, cc_labels = cv2.connectedComponents(od.astype(np.uint8), connectivity=8)
        if n_cc <= 1:
            return {"disc_found": False, "fovea_found": False, "vessels": ves_info}
        biggest_label = max(range(1, n_cc), key=lambda l: (cc_labels == l).sum())
        od = (cc_labels == biggest_label)
        ys, xs = np.nonzero(od)
        ox, oy = float(xs.mean()), float(ys.mean())
        dd = 2.0 * float(np.sqrt(od.sum() / np.pi))

        if not (DISC_PX_MIN <= dd <= DISC_PX_MAX):
            return {"disc_found": False, "fovea_found": False,
                    "disc_x": round(ox, 1), "disc_y": round(oy, 1),
                    "disc_diameter_px": round(dd, 1),
                    "reason": f"disc diameter {dd:.0f}px outside plausible {DISC_PX_MIN}-{DISC_PX_MAX}px range",
                    "vessels": ves_info}

        temporal_sign = -1.0 if ox >= D / 2 else 1.0     # fovea lies away from the nasal edge
        unit = dd if FOVEA_RULE["ruler"] == "dd" else D
        fx = ox + temporal_sign * FOVEA_RULE["temporal"] * unit
        fy = oy + FOVEA_RULE["inferior"] * unit
        refine = FOVEA_RULE.get("refine_radius_dd")
        if refine and canvas_bgr is not None and 0 <= fx < D and 0 <= fy < D:
            fx, fy = _dark_refine(canvas_bgr, fx, fy, refine * dd)
        inside = 0 <= fx < D and 0 <= fy < D
        return {"disc_found": True, "disc_x": round(ox, 1), "disc_y": round(oy, 1),
                "disc_diameter_px": round(dd, 1),
                "fovea_found": bool(inside),
                "fovea_x": round(fx, 1) if inside else None,
                "fovea_y": round(fy, 1) if inside else None,
                "macula_radius_px": round(dd, 1),
                "method": "disc-relative offset fitted on IDRiD train, refined to the darkest point nearby",
                "vessels": ves_info}

    def _findings(self, maps: np.ndarray, seg: np.ndarray, structures: dict) -> list[dict]:
        """Count lesions at calibrated per-class threshold, with sub-pixel MA and hemorrhage subclassification."""
        thr_map = self.calibration["lesion_thresholds"]
        D = maps.shape[-1]
        nasal_right = self._disc_side(seg, D)
        fx, fy = structures.get("fovea_x"), structures.get("fovea_y")
        mac_r = structures.get("macula_radius_px") or 0.0
        out = []
        for c, name in enumerate(LESIONS):
            thr = float(thr_map.get(name, DEFAULT_LESION_THR[name]))
            mask = (maps[c] >= thr).astype(np.uint8)
            n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
            keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA[name]]
            quad = {k: 0 for k in ("superior_nasal", "superior_temporal",
                                   "inferior_nasal", "inferior_temporal")}
            macular = 0
            subpixel_list = []
            subtypes = None

            if name == "MA":
                for i in keep:
                    cx, cy = cents[i]
                    x0 = max(0, int(cx) - 4)
                    x1 = min(D, int(cx) + 5)
                    y0 = max(0, int(cy) - 4)
                    y1 = min(D, int(cy) + 5)
                    patch = maps[c, y0:y1, x0:x1]
                    total_w = float(patch.sum())
                    if total_w > 1e-6:
                        gy, gx = np.mgrid[y0:y1, x0:x1]
                        sub_x = float((gx * patch).sum() / total_w)
                        sub_y = float((gy * patch).sum() / total_w)
                    else:
                        sub_x, sub_y = float(cx), float(cy)
                    dist_fovea = round(float(np.hypot(sub_x - fx, sub_y - fy)), 1) if fx is not None else None
                    subpixel_list.append({
                        "x": round(sub_x, 2),
                        "y": round(sub_y, 2),
                        "peak_prob": round(float(patch.max()), 4),
                        "dist_fovea_px": dist_fovea
                    })

            if name == "HE":
                subtypes = {"dot_blot": 0, "flame_shaped": 0, "preretinal_vitreous": 0}
                for i in keep:
                    area_i = stats[i, cv2.CC_STAT_AREA]
                    w_i = stats[i, cv2.CC_STAT_WIDTH]
                    h_i = stats[i, cv2.CC_STAT_HEIGHT]
                    aspect = max(w_i, h_i) / max(min(w_i, h_i), 1)
                    if area_i > 1500:
                        subtypes["preretinal_vitreous"] += 1
                    elif aspect > 2.0:
                        subtypes["flame_shaped"] += 1
                    else:
                        subtypes["dot_blot"] += 1

            for i in keep:
                cx, cy = cents[i]
                vert = "superior" if cy < D / 2 else "inferior"
                right = cx >= D / 2
                side = "nasal" if right == nasal_right else "temporal"
                quad[f"{vert}_{side}"] += 1
                if fx is not None and (cx - fx) ** 2 + (cy - fy) ** 2 <= mac_r ** 2:
                    macular += 1

            area = int(sum(stats[i, cv2.CC_STAT_AREA] for i in keep))
            entry = {
                "lesion_type": name, "threshold": thr, "lesion_count": len(keep),
                "pixel_area": area, "area_frac": round(area / float(D * D), 8),
                "peak_prob": round(float(maps[c].max()), 6), "quadrant_counts": quad,
                "macular_count": macular,
                "mask": mask if keep else None
            }
            if subpixel_list:
                entry["subpixel_coords"] = subpixel_list
            if subtypes is not None:
                entry["subtypes"] = subtypes
            out.append(entry)
        return out

    @staticmethod
    def _disc_side(seg: np.ndarray, D: int) -> bool:
        """True when the optic disc sits in the right half of the image.

        The disc is nasal to the macula, so its side tells us which half is nasal without needing
        to know the laterality convention of the uploader. Channel 4 is the OD head.
        """
        if seg.shape[0] < 5:
            return True
        od = seg[4] > 0.5
        if od.sum() < 50:
            return True                                   # disc not found: fall back to image sides
        return float(np.nonzero(od)[1].mean()) >= D / 2

    # ---------------------------------------------------------------- trust
    def reliability(self, quality_probs: dict | None, domain_key: str | None) -> dict:
        """How much this particular photograph, from this particular camera, can be believed.

        Confidence and trust are not the same quantity. The grade head can be very sure about an
        image that is too dim to grade, or sure in a way that was calibrated on a camera nobody has
        pointed at this patient. Reliability is that second quantity: it says nothing about which
        grade is right, only about whether the conditions for believing any grade are met.

        quality  p(good) + 0.75 * p(usable). "reject" never reaches here -- the worker gates those
                 out and asks for a retake -- so this is what separates a good photograph from a
                 merely usable one, which the hard gate cannot do.
        camera   1.0 when this domain has its own fitted calibration, 0.80 when it does not. An
                 uncalibrated camera is not assumed broken, it is assumed unverified.
        """
        q = quality_probs or {}
        if q:
            qf = float(q.get("good", 0.0)) + USABLE_WEIGHT * float(q.get("usable", 0.0))
            quality_factor = min(1.0, max(0.0, qf))
        else:
            quality_factor = 1.0
        row = (self.calibration.get("domains") or {}).get(domain_key or "", {})
        camera_fitted = bool(row.get("fitted"))
        camera_factor = 1.0 if camera_fitted else UNVERIFIED_CAMERA_FACTOR
        return {"quality_factor": round(quality_factor, 6),
                "camera_factor": camera_factor,
                "reliability": round(quality_factor * camera_factor, 6),
                "camera_fitted": camera_fitted}

    @staticmethod
    def _decisiveness(p: float, clear: float, refer: float) -> float:
        """How far outside the abstention band this score sits, on a 0-1 scale.

        Zero anywhere inside the band, rising to one at the extremes. This is the part of trust
        that comes from the model; reliability is the part that comes from the conditions.
        """
        if p <= clear:
            return (clear - p) / clear if clear > 0 else 1.0
        if p > refer:
            return (p - refer) / (1.0 - refer) if refer < 1 else 1.0
        return 0.0

    # ---------------------------------------------------------------- decision
    def _second_agrees(self, call: str, second_logit: float | None, domain_key: str | None) -> bool:
        """Does the second reader make the same automatic call, outside its own band?

        With no second reader configured, Certus decides alone and this is always True. With one
        configured, a missing score counts as disagreement: an eye is never auto-decided on one
        reader's word when the deployment promises two.
        """
        sr = self.calibration.get("second_reader") or {}
        if not sr:
            return True
        if second_logit is None:
            return False
        own = (self.calibration.get("per_domain_second_bands") or {}).get(domain_key or "") or {}
        clear = float(own.get("clear_below_logit", sr["clear_below_logit"]))
        refer = float(own.get("refer_above_logit", sr["refer_above_logit"]))
        return second_logit > refer if call == "referable" else second_logit <= clear

    def decide(self, p_referable: float, quality_probs: dict | None = None,
               domain_key: str | None = None, alpha: float = 0.05,
               second_logit: float | None = None) -> dict:
        """Refer / do not refer / hand to a human.

        Two independent things send an eye to a person: the score landing inside the screening band
        (the model cannot separate the grade), or reliability falling below MIN_RELIABILITY (the
        conditions for believing any score are not met).

        The reliability gate is strictly additive. At reliability 1.0 this function behaves exactly
        as it did before the gate existed, which is what keeps the measured abstention and
        system-sensitivity figures on the held-out splits valid: every eye in those evaluations was
        gradable and came from a fitted domain, so the gate never fired there. In the field it can
        only add abstentions, never remove them.
        """
        cal = self.calibration
        thr = cal["referral_threshold"]
        rel = self.reliability(quality_probs, domain_key)
        band0 = cal.get("screening_band") or {}
        clear0 = float(band0.get("clear_below", thr))
        refer0 = float(band0.get("refer_above", thr))
        trust = round(100.0 * rel["reliability"] * self._decisiveness(p_referable, clear0, refer0))
        extra = {"trust_score": trust, "quality_factor": rel["quality_factor"],
                 "camera_factor": rel["camera_factor"], "reliability": rel["reliability"]}

        if rel["reliability"] < MIN_RELIABILITY:
            return {"decision": "refer-to-human", "abstained": True,
                    "abstain_reason": "low_reliability", "threshold_used": thr,
                    "conformal_alpha": alpha, **extra}

        # A conformal band guarantees that the label set contains the truth with probability
        # 1 - alpha. That is not the same promise as screening sensitivity, and on this model the
        # difference mattered: at alpha = 0.05 the clear-side band auto-cleared any eye scoring
        # below 0.4598, far above the 0.100 that achieves 90% sensitivity, so 15.8% of referable
        # eyes on the test set were sent home without a human ever seeing them. The screening band
        # replaces that lower edge with a cut fitted on val at 99% sensitivity; the upper edge is
        # still the conformal one, because deciding to refer needs no such caution.
        # Keys here are splatted into the Prediction ORM model by the worker, so this dict may
        # only carry fields that exist as columns. Which band decided is reported by /v1/trust.
        band = cal.get("screening_band") or {}
        # A camera with its own fitted band (torch/fit_camera.py) is judged by it. On Messidor-2 a
        # band re-fitted on 100 labelled patients took specificity from 72% to 92% at the same
        # sensitivity (runs_torch/.../new_camera.json); the global band is the fallback.
        own = ((cal.get("per_domain_bands") or {}).get(domain_key or "") or {})
        if own.get("clear_below") is not None and own.get("refer_above") is not None:
            band = own
        if band:
            clear, refer = float(band["clear_below"]), float(band["refer_above"])
            call = "referable" if p_referable > refer else "non-referable" if p_referable <= clear else None
            if call is None:
                return {"decision": "refer-to-human", "abstained": True, "abstain_reason": "ambiguous",
                        "threshold_used": thr, "conformal_alpha": alpha, **extra}
            # Second reader (torch/second_reader.py): an automatic call stands only if the plain
            # whole-image grader makes the same call outside its own band. On test this took system
            # sens/spec from 94.4/91.9 to 97.6/95.2, at 20% of eyes to a human instead of 12%.
            # Like the reliability gate it can only add abstentions, never remove one.
            if not self._second_agrees(call, second_logit, domain_key):
                return {"decision": "refer-to-human", "abstained": True, "abstain_reason": "readers_disagree",
                        "threshold_used": thr, "conformal_alpha": alpha, **extra}
            return {"decision": call, "abstained": False, "abstain_reason": None,
                    "threshold_used": thr, "conformal_alpha": alpha, **extra}

        q = (cal.get("conformal_q") or {}).get(str(alpha), {}).get("q")
        if not q:
            decision = "referable" if p_referable > thr else "non-referable"
            return {"decision": decision, "abstained": False, "abstain_reason": None,
                    "threshold_used": thr, "conformal_alpha": None, **extra}
        q1, q0 = float(q.get("1", q.get(1))), float(q.get("0", q.get(0)))
        in_ref = (1 - p_referable) <= q1
        in_non = p_referable <= q0
        if in_ref and not in_non:
            return {"decision": "referable", "abstained": False, "abstain_reason": None,
                    "threshold_used": thr, "conformal_alpha": alpha, **extra}
        if in_non and not in_ref:
            return {"decision": "non-referable", "abstained": False, "abstain_reason": None,
                    "threshold_used": thr, "conformal_alpha": alpha, **extra}
        reason = "ambiguous" if (in_ref and in_non) else "outside_conformal_set"
        return {"decision": "refer-to-human", "abstained": True, "abstain_reason": reason,
                "threshold_used": thr, "conformal_alpha": alpha, **extra}

    def describe(self) -> dict:
        return {"checkpoint": self.checkpoint, "step": self.step, "device": self.device,
                "canvas": self.cfg.canvas, "tile": self.cfg.tile, "grid": self.cfg.grid,
                "metrics": self.metrics,
                # domains and splits are whole evaluation tables; /v1/trust serves those. Leaving
                # them in here doubled this response for callers that only want the runtime.
                "calibration": {k: v for k, v in self.calibration.items()
                                if k not in ("domains", "splits")}}

    # ---------------------------------------------------------------- grad-cam
    def gradcam(self, canvas_bgr: np.ndarray, target: str = "referable") -> bytes:
        """Grad-CAM heatmap overlaid on the fundus canvas, returned as a JPEG byte string.

        Calls gradcam.grad_cam() against the shared model instance. A forward hook is attached
        to -- and removed from -- the backbone in a finally block, so the model is never left
        with a dangling hook. The caller gets a JPEG because that is what the /gradcam endpoint
        streams back, and re-encoding inside the endpoint would mean buffering the full np.ndarray
        over the thread boundary.

        overlay: jet-coloured heatmap blended at 50% opacity over the fundus, with the heatmap
        clipped to the FOV mask so the black border does not attract attention.
        """
        import cv2 as _cv2
        rgb = _cv2.cvtColor(canvas_bgr, _cv2.COLOR_BGR2RGB)
        canvas_f = rgb.astype(np.float32) / 255.0
        heat = _gradcam.grad_cam(self.model, canvas_f, self.cfg, target=target)

        # Map scalar heatmap → jet colour map.
        heat_u8 = (heat * 255).astype(np.uint8)
        jet = _cv2.applyColorMap(heat_u8, _cv2.COLORMAP_JET)   # BGR
        jet_rgb = _cv2.cvtColor(jet, _cv2.COLOR_BGR2RGB)

        # Blend: keep the FOV, hide the black border.
        fov = canvas_f.max(axis=2, keepdims=True) > 0.02
        base = (canvas_f * 255).astype(np.uint8)
        blend = np.where(fov, (0.5 * base + 0.5 * jet_rgb).astype(np.uint8), base)
        _, enc = _cv2.imencode(".jpg", _cv2.cvtColor(blend, _cv2.COLOR_RGB2BGR),
                               [_cv2.IMWRITE_JPEG_QUALITY, 90])
        return enc.tobytes()


def _cummin(p: np.ndarray) -> np.ndarray:
    """Ordinal probabilities must not increase with severity: P(>0) >= P(>1) >= ..."""
    return np.minimum.accumulate(p)


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine
