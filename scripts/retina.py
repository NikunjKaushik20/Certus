"""Shared fundus-image utilities: FOV normalisation, hashing, and hand-crafted features.

All functions take/return OpenCV BGR uint8 images. The same operations will be
mirrored in MATLAB for inference; keep them simple and deterministic.
"""
import cv2
import numpy as np

WORK = 768  # working scale for quality features


# ---------------------------------------------------------------- FOV handling
def fov_mask_small(img, max_side=512):
    """Binary FOV mask on a downscaled copy. Returns (mask_small, scale)."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    v = small.max(axis=2).astype(np.float32)
    # Some cameras (≈5% of DDR) pad with flat grey (~17), not black: threshold above the corner level
    c = max(4, min(v.shape) // 20)
    bg = np.median(np.concatenate([v[:c, :c].ravel(), v[:c, -c:].ravel(), v[-c:, :c].ravel(), v[-c:, -c:].ravel()]))
    thr = max(8.0, bg + 10.0, 0.06 * np.percentile(v, 99.5))
    m = (v > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None, s
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (lab == big).astype(np.uint8)
    # fill holes (dark lesions / OD rim can punch holes in very dark images)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, cnts, -1, 1, cv2.FILLED)
    return m, s


def normalize_fov(img, D):
    """Crop the FOV to a centred square canvas of side D (FOV diameter -> D).

    Returns (out_img, out_mask, tf, info). tf maps raw (x, y) -> canvas via
    x' = (x - x0) * s, y' = (y - y0) * s. Images whose FOV is clipped top/bottom
    (common in APTOS) keep the true horizontal diameter.
    """
    m_small, s0 = fov_mask_small(img)
    if m_small is None:
        return None, None, None, {"fov_ok": False}
    ys, xs = np.nonzero(m_small)
    x0s, x1s, y0s, y1s = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = (x1s - x0s) / s0, (y1s - y0s) / s0
    diam = max(bw, bh)
    cx, cy = (x0s + x1s) / 2 / s0, (y0s + y1s) / 2 / s0
    x0, y0 = cx - diam / 2, cy - diam / 2
    s = D / diam

    H, W = img.shape[:2]
    ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
    side = int(np.ceil(diam))
    pad = [max(0, -iy0), max(0, iy0 + side - H), max(0, -ix0), max(0, ix0 + side - W)]
    crop = img[max(0, iy0):min(H, iy0 + side), max(0, ix0):min(W, ix0 + side)]
    crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_CONSTANT, value=0)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    out = cv2.resize(crop, (D, D), interpolation=interp)

    # canvas mask = detected FOV (resized) AND ideal circle
    m_full = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
    mc = m_full[max(0, iy0):min(H, iy0 + side), max(0, ix0):min(W, ix0 + side)]
    mc = cv2.copyMakeBorder(mc, *pad, cv2.BORDER_CONSTANT, value=0)
    mc = cv2.resize(mc, (D, D), interpolation=cv2.INTER_NEAREST)
    circ = np.zeros((D, D), np.uint8)
    cv2.circle(circ, (D // 2, D // 2), D // 2 - 1, 1, cv2.FILLED)
    mask = (mc & circ).astype(np.uint8)
    out[mask == 0] = 0

    tf = {"x0": ix0, "y0": iy0, "s": D / side}
    info = {
        "fov_ok": True,
        "raw_w": W, "raw_h": H,
        "native_fov_px": round(diam, 1),
        "fov_clip_frac": round(1 - min(bw, bh) / diam, 4),
        "fov_area_frac": round(mask.sum() / circ.sum(), 4),
        "scale": round(D / side, 5),
    }
    return out, mask, tf, info


def transform_mask(raw_mask, tf, D):
    """Apply the same crop/resize to a raw lesion mask, preserving tiny lesions."""
    m = (raw_mask > 0).astype(np.uint8) * 255
    H, W = m.shape
    side = round(D / tf["s"])
    ix0, iy0 = tf["x0"], tf["y0"]
    pad = [max(0, -iy0), max(0, iy0 + side - H), max(0, -ix0), max(0, ix0 + side - W)]
    c = m[max(0, iy0):min(H, iy0 + side), max(0, ix0):min(W, ix0 + side)]
    c = cv2.copyMakeBorder(c, *pad, cv2.BORDER_CONSTANT, value=0)
    # INTER_AREA then >0 == "any lesion pixel in the footprint": microaneurysms survive downscaling
    return (cv2.resize(c, (D, D), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)


def transform_point(x, y, tf):
    return (x - tf["x0"]) * tf["s"], (y - tf["y0"]) * tf["s"]


# ---------------------------------------------------------------- hashing
def image_hashes(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    dh = np.packbits((d[:, 1:] > d[:, :-1]).ravel())
    p = cv2.dct(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    ph = np.packbits((p > np.median(p)).ravel())
    return dh.tobytes().hex(), ph.tobytes().hex()


# ---------------------------------------------------------------- features
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def clahe_green(img):
    return _CLAHE.apply(img[..., 1])


def _masked_blur(x, m, sigma):
    num = cv2.GaussianBlur(x * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def quality_features(img, mask):
    """Focus / illumination / contrast / vessel-visibility features at WORK scale."""
    img = cv2.resize(img, (WORK, WORK), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (WORK, WORK), interpolation=cv2.INTER_NEAREST).astype(bool)
    m_in = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)  # avoid rim edge
    if m_in.sum() < 1000:
        return {}
    f = {}
    b, g, r = [img[..., i].astype(np.float32) / 255 for i in range(3)]
    v = np.maximum(np.maximum(r, g), b)
    gi = g[m_in]
    f["mean_r"], f["mean_g"], f["mean_b"] = r[m_in].mean(), gi.mean(), b[m_in].mean()
    f["r_g_ratio"] = f["mean_r"] / (f["mean_g"] + 1e-6)
    f["std_g"] = gi.std()
    f["v_p05"], f["v_p50"], f["v_p95"] = np.percentile(v[m_in], [5, 50, 95])
    f["under_exposed_frac"] = (v[m_in] < 0.10).mean()
    f["over_exposed_frac"] = (v[m_in] > 0.95).mean()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
    f["sat_mean"] = hsv[..., 1][m_in].mean()
    f["glare_frac"] = ((hsv[..., 1] < 0.15) & (hsv[..., 2] > 0.90))[m_in].mean()

    mf = m.astype(np.float32)
    bg = _masked_blur(g, mf, WORK / 20)
    f["illum_cv"] = bg[m_in].std() / (bg[m_in].mean() + 1e-6)
    gn = g / np.maximum(bg, 1e-3)  # illumination-normalised green

    lap = cv2.Laplacian(gn, cv2.CV_32F, ksize=3)
    f["lap_var"] = lap[m_in].var()
    f["noise_sigma"] = np.median(np.abs(lap[m_in])) / 0.6745
    sx, sy = cv2.Sobel(gn, cv2.CV_32F, 1, 0), cv2.Sobel(gn, cv2.CV_32F, 0, 1)
    f["tenengrad"] = (sx ** 2 + sy ** 2)[m_in].mean()
    filled = np.where(m, gn, gn[m_in].mean())
    F = np.abs(np.fft.fftshift(np.fft.fft2(filled - filled.mean()))) ** 2
    yy, xx = np.mgrid[-WORK // 2:WORK // 2, -WORK // 2:WORK // 2]
    rr = np.hypot(yy, xx) / (WORK / 2)
    f["hf_energy_ratio"] = F[rr > 0.25].sum() / (F.sum() + 1e-9)

    cg = clahe_green(img)
    hist = np.bincount((cg[m_in] // 4).ravel(), minlength=64).astype(np.float64)
    p = hist / hist.sum()
    f["entropy_g"] = -(p[p > 0] * np.log2(p[p > 0])).sum()
    f["rms_contrast"] = (cg[m_in] / 255.0).std()

    bh = cv2.morphologyEx(cg, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    t, _ = cv2.threshold(bh[m_in], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ves = (bh > t) & m_in
    f["vessel_frac"] = ves.mean() / m_in.mean()
    f["vessel_contrast"] = bh[ves].mean() / 255 if ves.any() else 0.0
    f["vessel_fragments"] = cv2.connectedComponents(ves.astype(np.uint8))[0] - 1
    return {k: float(v) for k, v in f.items()}


# Candidate-lesion parameters for a 1536 canvas. Tuned against IDRiD/DDR ground-truth
# masks by scripts/dev_tune_lesions.py (114 train images); "abs" thresholds are in CLAHE
# grey levels. Dark: best count correlation (rho≈0.21, precision≈1%) — classical MA
# detection is weak, these are baseline features only. Bright: best Dice (≈0.28).
LESION_PARAMS = {
    "dark": {"se": 7, "mode": "abs", "t": 12, "ma_max_area": 80, "he_max_area": 5000},
    "bright": {"se": 25, "mode": "std", "t": 3.0, "min_area": 5},
}


def _se(d):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def lesion_context(img, mask):
    """Everything the candidate detectors share: CLAHE green, vessels, OD, valid region."""
    D = img.shape[0]
    sc = D / 1536  # structuring elements are specified for a 1536 canvas
    odd = lambda x: max(3, int(round(x * sc)) | 1)
    cg = clahe_green(img)
    m = cv2.erode(mask, _se(odd(31))).astype(bool)
    if m.sum() < 1000:
        return None
    bh_v = cv2.morphologyEx(cg, cv2.MORPH_BLACKHAT, _se(odd(23)))
    t, _ = cv2.threshold(bh_v[m], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ves = ((bh_v > t) & m).astype(np.uint8)
    # optic disc: brightest large blob of the smoothed luminance
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    sm = cv2.GaussianBlur(lum, (0, 0), D / 40)
    sm[~m] = 0
    oy, ox = np.unravel_index(np.argmax(sm), sm.shape)
    od = np.zeros((D, D), np.uint8)
    cv2.circle(od, (int(ox), int(oy)), int(D / 10), 1, cv2.FILLED)
    return {"D": D, "sc": sc, "odd": odd, "cg": cg, "m": m, "ves": ves,
            "ves_d": cv2.dilate(ves, _se(odd(5))).astype(bool),
            "od_xy": (ox, oy), "od_contrast": (sm[oy, ox] - np.median(sm[m])) / 255,
            "valid": m & ~od.astype(bool)}


def dark_response(ctx, se):
    return cv2.morphologyEx(ctx["cg"], cv2.MORPH_BLACKHAT, _se(ctx["odd"](se))).astype(np.float32)


def bright_response(ctx, se):
    return cv2.morphologyEx(ctx["cg"], cv2.MORPH_TOPHAT, _se(ctx["odd"](se))).astype(np.float32)


def _thr(resp, region, mode, t):
    x = resp[region]
    return t if mode == "abs" else x.mean() + t * x.std()


def dark_candidates(ctx, resp, p):
    """-> (ma_mask, he_mask, he_centroids). MA: small & compact; HE: larger dark blobs."""
    region = ctx["valid"] & ~ctx["ves_d"]
    dark = ((resp > _thr(resp, region, p["mode"], p["t"])) & region).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    sc2 = ctx["sc"] ** 2
    a = st[1:, cv2.CC_STAT_AREA] / sc2
    ext = st[1:, cv2.CC_STAT_AREA] / (st[1:, cv2.CC_STAT_WIDTH] * st[1:, cv2.CC_STAT_HEIGHT])
    ma = (a >= 3) & (a <= p["ma_max_area"]) & (ext > 0.45)
    he = (a > p["ma_max_area"]) & (a <= p["he_max_area"])
    return np.r_[False, ma][lab], np.r_[False, he][lab], cen[1:][he]


def bright_candidates(ctx, resp, p):
    region = ctx["valid"]
    bright = ((resp > _thr(resp, region, p["mode"], p["t"])) & region).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    keep = st[1:, cv2.CC_STAT_AREA] / ctx["sc"] ** 2 >= p["min_area"]
    return np.r_[False, keep][lab]


def lesion_features(img, mask, params=LESION_PARAMS):
    """Classical candidate-lesion statistics on the full canvas (baseline + explainability prior).

    Not a lesion detector: these are unsupervised candidate counts used as
    hand-crafted features and as the 'single-technique' baseline in ablations.
    """
    ctx = lesion_context(img, mask)
    if ctx is None:
        return {}
    D, area = ctx["D"], ctx["m"].sum()
    ox, oy = ctx["od_xy"]
    f = {"vessel_density": ctx["ves"].sum() / area, "od_x_est": ox / D, "od_y_est": oy / D,
         "od_contrast": ctx["od_contrast"]}
    pd_, pb = params["dark"], params["bright"]
    ma, he, cen = dark_candidates(ctx, dark_response(ctx, pd_["se"]), pd_)
    f["ma_cand_count"] = cv2.connectedComponents(ma.astype(np.uint8))[0] - 1
    f["he_cand_count"] = len(cen)
    f["dark_lesion_area_frac"] = (ma | he).sum() / area
    # quadrant spread of haemorrhage-like candidates (proxy for the 4-2-1 rule)
    q = (cen[:, 0] > D / 2).astype(int) + 2 * (cen[:, 1] > D / 2).astype(int) if len(cen) else np.array([], int)
    for i in range(4):
        f[f"he_cand_q{i}"] = int((q == i).sum())
    f["he_quadrants"] = int(sum(f[f"he_cand_q{i}"] > 0 for i in range(4)))
    ex = bright_candidates(ctx, bright_response(ctx, pb["se"]), pb)
    f["ex_cand_count"] = cv2.connectedComponents(ex.astype(np.uint8))[0] - 1
    f["bright_lesion_area_frac"] = ex.sum() / area
    return {k: float(v) for k, v in f.items()}


def evaluate_adequacy(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Evaluate fundus image adequacy across focus, illumination, and field-of-view.

    Provides clinically actionable feedback for primary healthcare technicians.
    """
    H, W = img_bgr.shape[:2]
    if mask is None:
        v = np.max(img_bgr, axis=2)
        mask = v > 15
    m = mask.astype(bool)
    m_in = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if m_in.sum() < 500:
        return {
            "focus": {"score": 0.0, "adequate": False, "metric": "insufficient_fov"},
            "illumination": {"score": 0.0, "adequate": False, "metric": "insufficient_fov"},
            "field_of_view": {"score": 0.0, "adequate": False, "metric": "insufficient_fov"},
            "overall_adequate": False,
            "reasons": ["field_of_view_missing"],
            "recapture_advice": ["Retina not detected. Ensure fundus camera lens is aligned with pupil and within working distance."],
        }

    # 1. Illumination adequacy
    b, g, r = [img_bgr[..., i].astype(np.float32) / 255.0 for i in range(3)]
    v = np.maximum(np.maximum(r, g), b)
    v_in = v[m_in]
    v_p50 = float(np.percentile(v_in, 50))
    under_exposed = float((v_in < 0.10).mean())
    over_exposed = float((v_in > 0.95).mean())

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    glare_frac = float(((hsv[..., 1] < 0.15) & (hsv[..., 2] > 0.90))[m_in].mean())

    mf = m.astype(np.float32)
    bg = _masked_blur(g, mf, max(H, W) / 20)
    illum_cv = float(bg[m_in].std() / (bg[m_in].mean() + 1e-6))

    illum_adequate = bool((under_exposed < 0.35) and (over_exposed * 100 < 15) and (glare_frac < 0.06) and (v_p50 >= 0.12))
    illum_score = float(np.clip(1.0 - (under_exposed * 1.5 + over_exposed * 2.0 + glare_frac * 3.0 + illum_cv * 0.5), 0.0, 1.0))

    # 2. Focus adequacy
    gn = g / np.maximum(bg, 1e-3)
    lap = cv2.Laplacian(gn, cv2.CV_32F, ksize=3)
    lap_var = float(lap[m_in].var())
    sx = cv2.Sobel(gn, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(gn, cv2.CV_32F, 0, 1)
    tenengrad = float((sx ** 2 + sy ** 2)[m_in].mean())

    focus_adequate = bool((lap_var >= 0.0006) or (tenengrad >= 0.0012))
    focus_score = float(np.clip(tenengrad / 0.005, 0.0, 1.0))

    # 3. Field of View adequacy
    total_circle_px = np.pi * ((min(H, W) / 2.0) ** 2)
    fov_frac = float(m.sum() / max(total_circle_px, 1.0))
    fov_adequate = bool(fov_frac >= 0.65)
    fov_score = float(np.clip(fov_frac / 0.85, 0.0, 1.0))

    reasons = []
    advice = []
    if not focus_adequate:
        reasons.append("focus_blur")
        advice.append("Defocus blur detected: adjust camera focus/diopter ring until retinal vessel arcades appear crisp.")
    if under_exposed >= 0.35 or v_p50 < 0.12:
        reasons.append("underexposed")
        advice.append("Image is underexposed: increase flash intensity or verify patient pupil is sufficiently dilated.")
    if over_exposed >= 0.15:
        reasons.append("overexposed")
        advice.append("Image is overexposed: decrease flash intensity or shorten exposure time.")
    if glare_frac >= 0.06:
        reasons.append("corneal_glare")
        advice.append("Corneal reflex/glare detected: re-align camera pupil axis and ensure patient blinks prior to capture.")
    if not fov_adequate:
        reasons.append("fov_clipped")
        advice.append("Field of view clipped or off-center: re-center the macula and optic disc within the imaging aperture.")

    overall_adequate = bool(illum_adequate and focus_adequate and fov_adequate)

    return {
        "focus": {"score": round(focus_score, 4), "adequate": focus_adequate, "lap_var": round(lap_var, 6), "tenengrad": round(tenengrad, 6)},
        "illumination": {"score": round(illum_score, 4), "adequate": illum_adequate, "under_exposed_frac": round(under_exposed, 4), "glare_frac": round(glare_frac, 4), "illum_cv": round(illum_cv, 4)},
        "field_of_view": {"score": round(fov_score, 4), "adequate": fov_adequate, "fov_fraction": round(fov_frac, 4)},
        "overall_adequate": overall_adequate,
        "reasons": reasons,
        "recapture_advice": advice,
    }


def adaptive_enhance(img_bgr: np.ndarray, quality_label: str = "usable", force: bool = False) -> tuple[np.ndarray, dict]:
    """Adaptive enhancement for borderline fundus images: CLAHE, illumination normalization, and denoising.

    Strictly preserves image dimensions and colour semantics. Only applies enhancement
    if image is borderline ('usable') or explicitly forced, leaving high-quality images untouched.
    """
    if quality_label != "usable" and not force:
        return img_bgr, {"applied": False, "methods": []}

    methods = []
    out = img_bgr.copy()

    # 1. Illumination normalization & Green CLAHE
    b, g, r = cv2.split(out)
    H, W = g.shape[:2]
    sigma = max(H, W) / 25.0
    bg = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma)
    mean_bg = float(np.mean(bg[bg > 10])) if np.any(bg > 10) else 128.0
    g_norm = np.clip((g.astype(np.float32) / np.maximum(bg, 1e-3)) * mean_bg, 0, 255).astype(np.uint8)
    methods.append("illumination_normalization")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g_norm)
    methods.append("green_channel_clahe")

    # 2. Edge-preserving bilateral denoising to suppress sensor noise without blurring MAs
    denoised_g = cv2.bilateralFilter(g_clahe, d=5, sigmaColor=25, sigmaSpace=25)
    methods.append("bilateral_denoising")

    # Blend enhanced green (70% enhanced + 30% original)
    g_final = cv2.addWeighted(denoised_g, 0.70, g, 0.30, 0)
    r_clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(r)
    b_clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(b)

    out = cv2.merge([b_clahe, g_final, r_clahe])
    return out, {"applied": True, "methods": methods}

