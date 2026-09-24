"""Hold the onnxruntime path to the PyTorch one: same grade, decision and lesion counts per eye.

    python check_onnx_parity.py [image ...]      # default: every photo in demo_images/

Runs both runtimes on CPU (fp32, the reference) and exits non-zero on any disagreement a user could
see. Needs torch installed; the server it protects does not.
"""
import glob
import os
import sys

import numpy as np

from certus_api.config import settings
from certus_api.inference import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL_P = 1e-3        # probabilities; the graphs match torch to ~5e-5 in logit


def build(runtime: str) -> Engine:
    settings.runtime, settings.device = runtime, "cpu"
    return Engine()


def main() -> int:
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "demo_images", "*.jpg")))
    ref, onx = build("torch"), build("onnx")
    assert onx.executor.startswith("onnxruntime"), onx.executor
    bad = 0
    for path in paths:
        raw = open(path, "rb").read()
        pre = ref.preprocess(raw)
        a, b = ref.analyse(pre["canvas"]), onx.analyse(pre["canvas"])
        da = ref.decide(a["p_referable"], quality_probs=a["quality"]["probs"],
                        second_logit=(a["second_reader"] or {}).get("logit"))
        db = onx.decide(b["p_referable"], quality_probs=b["quality"]["probs"],
                        second_logit=(b["second_reader"] or {}).get("logit"))
        count = lambda r: {f["lesion_type"]: f["lesion_count"] for f in r["findings"]}  # noqa: E731
        ha, hb = ref.net.grad_cam(pre["canvas"][..., ::-1].astype(np.float32) / 255), \
            onx.net.grad_cam(pre["canvas"][..., ::-1].astype(np.float32) / 255)
        checks = {
            "grade": (a["dr_grade"], b["dr_grade"], a["dr_grade"] == b["dr_grade"]),
            "decision": (da["decision"], db["decision"], da["decision"] == db["decision"]),
            "quality": (a["quality"]["label"], b["quality"]["label"], a["quality"]["label"] == b["quality"]["label"]),
            "p_referable": (a["p_referable"], b["p_referable"], abs(a["p_referable"] - b["p_referable"]) < TOL_P),
            "second_logit": (a["second_reader"]["logit"], b["second_reader"]["logit"],
                             abs(a["second_reader"]["logit"] - b["second_reader"]["logit"]) < 1e-2),
            "lesions": (count(a), count(b), count(a) == count(b)),
            "evidence": ("", f"max diff {np.abs(np.subtract(a['evidence'], b['evidence'])).max():.1e}",
                         np.allclose(a["evidence"], b["evidence"], atol=1e-3)),
            "gradcam": ("", f"max diff {np.abs(ha - hb).max():.3f}, r={np.corrcoef(ha.ravel(), hb.ravel())[0, 1]:.4f}",
                        np.corrcoef(ha.ravel(), hb.ravel())[0, 1] > 0.99),
        }
        fails = [k for k, (_, _, ok) in checks.items() if not ok]
        bad += bool(fails)
        print(f"{'FAIL' if fails else 'ok  '} {os.path.basename(path)}  "
              f"grade {a['dr_grade']}  p_ref {a['p_referable']:.5f}/{b['p_referable']:.5f}  "
              f"{checks['evidence'][1]}  gradcam {checks['gradcam'][1]}  "
              f"ms torch {a['runtime_ms']} onnx {b['runtime_ms']}")
        for k in fails:
            print(f"     {k}: torch={checks[k][0]} onnx={checks[k][1]}")
    print(f"{len(paths) - bad}/{len(paths)} identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
