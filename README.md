# Certus

Explainable diabetic retinopathy screening for rural camps.
SIH 2026, problem statement 26038.

A technician photographs both eyes, the model grades them on the ICDR 0–4 scale, and every
grade arrives with the lesions it found, the tiles it looked at, a calibrated probability, and
the option to refuse to guess.

Two readers decide every eye. Certus is the one that explains itself: lesions, tiles and a
Grad-CAM a clinician can check. A plain whole-image grader is the one with no explanation but
fewer mistakes. An eye is referred or cleared automatically only when both agree; everything
else goes to an ophthalmologist, much as two human graders with adjudication would handle it.

## Architecture

The diagram is in [`docs/architecture.mmd`](docs/architecture.mmd) (Mermaid).

The same pipeline runs in MATLAB from ONNX exports (`matlab/`, checked against Python by
`torch/agreement.py`). The Simulink model in `matlab/simulink/` sizes the camp around it: staff,
cameras, edge devices and uplink.

## Held-out performance

Every threshold below was fitted on validation (3,049 eyes) and applied unchanged. Nothing is
re-tuned on the set it is scored on. Messidor-2 (Topcon TRC NW6) was never used in training,
checkpoint selection or calibration.

| | test (4,511 eyes) | Messidor-2 (1,740 eyes, unseen camera) |
|---|---|---|
| **Two readers: system sens / spec, disagreements reviewed** | **97.6% / 95.2%** | **100% / 89.1%** |
| **Two readers: sent to an ophthalmologist** | 20.1% | 45.9% |
| **Two readers, Messidor-2 bands re-fitted on 100 of its patients** | | **97.4% / 96.5%, 26.9% to a person** |
| _Certus alone, from here down_ | | |
| AUC, referable DR | 0.952 | 0.952 |
| Quadratic kappa, 5 grades | 0.834 | 0.745 |
| Sens / spec at the val threshold for 85% spec (0.078) | 91.6% / 84.9% | 98.5% / **50.0%** |
| Screening band: abstention rate | 11.7% | 26.4% |
| Screening band: system sens / spec, abstentions reviewed | 94.4% / 91.9% | 99.6% / 72.4% |
| Calibration error (ECE), raw → calibrated | 0.090 → 0.079 | 0.083 → 0.093 |

"System" counts an eye sent to a person as correctly graded, because an ophthalmologist reads
it. With two readers the system clears both targets (90% / 85%) on test with a wide margin, and
on the unseen camera once both readers' bands are re-fitted (below). The cost is workload:
one eye in five goes to a person instead of one in eight.

Certus alone sits on the targets in distribution. On a camera it has never seen, the scores shift
upwards and specificity collapses. The same thing happens inside test on IDRiD (Kowa VX-10a,
115 eyes): 100% sensitivity, 53.8% specificity at the val threshold.

### A new camera needs about 100 labelled patients

The fix is to label a small sample from the new camera and re-fit the screening band's two
cuts on it, keeping the specificity each cut had on validation. We drew k patients from
Messidor-2 (both eyes together), re-fitted, and scored the remaining patients, 1,000 times
(`torch/new_camera.py`, results in `runs_torch/train_20260912_211356/new_camera.json`):

| Labelled patients from the new camera | System sensitivity | System specificity | Abstained | Draws meeting both targets |
|---|---|---|---|---|
| 0 (val band) | 99.6% | 72.4% | 26.4% | – |
| 25 (~50 eyes) | 92.8% [84.2–97.1] | 90.9% [78.6–98.2] | 12.7% | 70% |
| 50 (~100 eyes) | 93.1% [88.4–96.9] | 91.2% [82.9–96.9] | 12.7% | 87% |
| 100 (~200 eyes) | 93.1% [90.3–96.3] | 91.6% [85.6–96.3] | 12.9% | 96% |
| 200 (~400 eyes) | 93.2% [91.1–95.3] | 91.6% [86.9–95.2] | 13.0% | 99% |

With both readers re-fitted the same way (`torch/second_reader.py`), every one of the 1,000
draws meets both targets, even with only 25 patients:

| Labelled patients | Two readers: system sens | System spec | To a person | Draws meeting both targets |
|---|---|---|---|---|
| 25 | 97.4% [94.3–99.1] | 95.8% [89.4–99.2] | 27.3% | 100% |
| 50 | 97.5% [95.5–98.9] | 96.2% [92.3–98.8] | 27.3% | 100% |
| 100 | 97.4% [96.0–98.6] | 96.5% [93.6–98.5] | 26.9% | 100% |
| 200 | 97.4% [96.4–98.5] | 96.7% [94.6–98.3] | 26.9% | 100% |

Brackets are 95% intervals over draws. "System" counts an abstained eye as correctly graded,
because it goes to an ophthalmologist. A single re-fitted threshold does not get there:
Messidor-2's ROC curve passes almost exactly through (90%, 85%), so a single cut lands on
either side about equally often. The band is what gives the margin. Recalibration also brings
ECE from 0.093 to 0.031 at 100 patients.

This is why a camera without its own calibration is marked unverified in the per-camera trust
table. In the field, a new camera's first ~100 patients go to full ophthalmologist review, which
is also what produces the labels. `torch/fit_camera.py <domain_key> <graded.csv>` then fits that
camera's bands, for both readers, with the same rule and writes them into `trust.json`; the API
decides every eye from a device with that `domain_key` by its own bands from then on.

Calibration is a single temperature, T = 1.211, fitted on validation. It changes probabilities,
not rankings, so AUC and kappa are identical before and after. Within test, ECE by camera was
0.032 (APTOS), 0.051 (IDRiD) and 0.094 (DDR). All figures are in
`runs_torch/train_20260912_211356/trust.json`.

### What each part of the architecture buys

The same frozen encoder features, with only the pooling and the head changed, each trained
from scratch for 30 epochs over 3 seeds (`torch/ablation_features.py`, `torch/ablation_heads.py`,
results in `Data/ablation/ablation.json`). Mean ± sd; sensitivity and specificity at the val
threshold for 85% specificity. These heads are re-trained on frozen features, so the
Certus row is close to, but not the same as, the deployed model in the table above.

| Head sees | Test AUC | Test kappa | Test sens / spec | Messidor-2 AUC | Messidor-2 kappa |
|---|---|---|---|---|---|
| Whole image only (512 px) | 0.920 ± 0.001 | 0.796 | 86.2% / 82.7% | 0.930 ± 0.001 | 0.741 |
| 12 lesion-evidence numbers only | 0.923 ± 0.001 | 0.747 | 84.8% / 85.8% | 0.913 ± 0.002 | 0.637 |
| Nine tiles, mean-pooled | 0.934 ± 0.005 | 0.777 | 84.4% / 86.8% | 0.905 ± 0.017 | 0.634 |
| Nine tiles, gated attention | 0.937 ± 0.005 | 0.801 | 86.2% / 86.7% | 0.911 ± 0.008 | 0.655 |
| Tiles (attention) + whole image | 0.940 ± 0.001 | 0.827 | 89.3% / 84.0% | 0.949 ± 0.002 | 0.737 |
| **Tiles + whole image + evidence (Certus)** | **0.951 ± 0.002** | **0.828** | **90.3% / 85.7%** | 0.943 ± 0.011 | 0.717 |
| _Classical: LightGBM on engineered features_ | _0.812_ | | | _0.579_ | |

Reading it honestly: the tiles are what find small lesions (test AUC 0.920 → 0.937), the whole
view is what keeps the unseen camera in check (tiles alone drop to 0.91 on Messidor-2), and the
evidence vector adds a further 0.011 on test. On Messidor-2 the
evidence vector does not help (0.943 vs 0.949, within about one sd of each other). Twelve
lesion numbers alone reach 0.923: the lesion masks carry most of the grading signal, which is
the point of making the grade depend on them. The classical pipeline (`Data/baselines/`)
generalises badly to the new camera.

### What the architecture costs

The ablation above only compares heads on Certus's own frozen features. The fairer question is
whether a plain grader, trained end to end, does better. `torch/train_baseline.py` trains
EfficientNet-B0 on the whole photograph at 512 px. It uses the same training eyes, sampling,
augmentation and ordinal loss as Certus, with checkpoint selection and the threshold both
fixed on val. It has no tiles, no lesion masks, no evidence vector and no quality gate.

| | Test AUC | Test kappa | Test sens / spec | Messidor-2 AUC | Messidor-2 kappa | Messidor-2 sens / spec |
|---|---|---|---|---|---|---|
| Whole-image baseline, end to end (1 run) | **0.964** | **0.876** | 93.7% / 84.9% | **0.968** | **0.820** | 97.6% / **68.9%** |
| Certus, deployed | 0.952 | 0.834 | 91.6% / 84.9% | 0.952 | 0.745 | 98.5% / 50.0% |

The baseline is the better grader on both test and the unseen camera. It is ahead by 0.012
AUC on test and 0.016 on Messidor-2, and it holds specificity better on the unseen camera.
Certus gives up about a point of AUC for things the baseline cannot provide:

- a grade computed from lesions it has drawn and that a clinician can check;
- exact per-tile Grad-CAM;
- a quality gate;
- a calibrated abstention band.

Whether that trade is worth it is for readers to judge (see `study/`). It should not be read as
a free improvement. This is a single run, so its seed-to-seed variation is unknown; the
ablation heads varied by 0.001–0.017 AUC across seeds. Weights and scores are in
`runs_torch/baseline_wholeimage_20260921_154239/`.

### Two readers instead of one

Certus and the baseline make different mistakes, so the deployed decision uses both.
`torch/second_reader.py` gives each reader its own screening band, cut at the same val
specificities as Certus's band, so no new parameter is tuned for the combined rule. An eye is
auto-cleared only if both readers clear it, and auto-referred only if both refer it. Anything
else, whether a disagreement or either reader unsure, goes to an ophthalmologist
(`abstain_reason = readers_disagree`).

| Test (4,511 eyes) | System sens / spec | To a person | Referable eyes auto-cleared | Healthy eyes auto-referred |
|---|---|---|---|---|
| Certus alone | 94.4% / 91.9% | 11.7% | 118 | 196 |
| Baseline alone | 95.6% / 92.1% | 9.9% | 93 | 191 |
| **Two readers** | **97.6% / 95.2%** | 20.1% | **50** | **115** |
| _OR-gate: refer if either says refer, no human_ | 96.4% / 79.0% | 0% | 75 | 505 |

The OR-gate row is the rule to use when there is no human reviewer: it buys sensitivity with
false referrals. Certus has a reviewer, so disagreements go to them instead of defaulting to
refer. That spares about 390 healthy people a trip to the district hospital and still misses
fewer sick eyes. (Every rule is at thresholds fitted on val; the OR-gate uses each reader's val
cut for 85% specificity.)

What each reader contributes: the explanation a clinician sees is still Certus's. It shows the
lesions, per-tile attention and exact Grad-CAM. The second reader contributes a vote and
nothing else. When it vetoes an automatic call, the reviewer is told that the readers disagreed,
and sees the second reader's score next to Certus's evidence.

The rule is also a trade-off, and the other settings are in
`runs_torch/baseline_wholeimage_20260921_154239/second_reader.json` (`frontier`). Narrowing both
bands to a single 85% cut sends 9.6% of test eyes to a person at 96.4% / 90.5%. Widening the
refer cut to 95% raises specificity to 97.1% at 25.5% reviewed.

Published systems, for scale only. They were tested on other data, so this is not a head-to-head
comparison:

| System | Data | Referable DR |
|---|---|---|
| IDx-DR (Abràmoff 2018, FDA pivotal) | 819 US primary-care patients | sens 87.2%, spec 90.7% |
| IDx-DR (Abràmoff 2016) | Messidor-2 | AUC 0.980, sens 96.8%, spec 87.0% |
| Gulshan et al. 2016 (JAMA) | Messidor-2 | AUC 0.990, sens 87.0%, spec 98.5% |
| Voets et al. 2019 (Gulshan replication, public data) | Messidor-2 | AUC 0.853 |
| **Certus**, no Messidor-2 data at all | Messidor-2 | AUC 0.952; band system 99.6% / 72.4% |
| **Certus**, band re-fitted on 100 patients | Messidor-2, remaining patients | system 93.1% / 91.6% |

Gulshan's model was trained on 128,000 images graded by a panel; Voets re-trained the same
recipe on public data and got 0.853 on Messidor-2. Certus is trained on public data only, so
Voets is the fairer reference.

### Anatomy and lesions

Everything the explanation panel draws is scored against pixel or landmark ground truth
(`torch/structures_eval.py` → `reports/structures.json`, `torch/lesion_froc.py` →
`reports/lesion_froc.json`). Errors are in native pixels of the original photograph.

| Structure | Data | Certus | Reference |
|---|---|---|---|
| Optic disc centre | IDRiD localisation, 115 test eyes | mean 29.2 px, median 26.6; 100% within one disc radius | IDRiD challenge winner: 21.1 px |
| Fovea | same | mean 66.9 px, median 35.4; 93.9% within one disc radius | IDRiD challenge winner: 64.5 px |
| Vessels | HRF, 15 test images | AUC 0.974; F1 0.736 at the deployed threshold 0.5 | |
| Vessels, unseen camera | DRIVE, 20 annotated images (565 px, Canon CR5) | AUC 0.876; F1 0.170 at 0.5; F1 0.571 at the best threshold chosen on these same images | DRIVE F1 is typically ~0.80 for dedicated vessel models |
| Microaneurysms, pixel AUPR | IDRiD segmentation, 30 test images | 0.369 | IDRiD winner 0.502 |
| Haemorrhages | same | 0.527 | 0.680 |
| Hard exudates | same | 0.273 | 0.885 |
| Soft exudates | same | 0.484 | 0.700 |

Where it is weak:

- **Lesion masks are below the segmentation challenge winners**, most of all for hard exudates.
  Those entries were segmentation-only models at full resolution. Certus segments on a 1536 px
  canvas (about 2.3 native px per canvas px on IDRiD) as a side task of grading. The masks are
  good enough to drive the grade (see the ablation) but should not be read as lesion-exact.
- **The microaneurysm detector over-calls.** At the deployed threshold it finds 74.6% of
  annotated MAs with 39.8 false candidates per image. Even at a threshold of 0.999 it still
  produces 15.7 per image, so a standard FROC curve (0.125–8 FP/image) cannot be reached for
  MAs at all. Read an MA count as a pointer to where to look, not a count to trust.
- **Sub-pixel MA centres** (837 matched MAs) have a median error of 2.8 native px using the
  probability-weighted centroid, against 3.7 px for the raw argmax. Quadratic and Gaussian peak
  fits (3.2–3.3 px) did no better, so the API and `matlab/model/subpixelMA.m` both use the centroid.
- **The fovea rule is fitted, not taken from a textbook.** The textbook offset (2.5 disc
  diameters temporal) gave a median error of 173 px on IDRiD train. Disc diameter is a noisy
  ruler when the disc mask is imperfect, so the offset is fitted in units of the field-of-view
  diameter on IDRiD train (`torch/fit_fovea.py`), then moved to the darkest point within half a
  disc diameter. Python and MATLAB share the rule through `torch/fovea_rule.json`.
- **A labelling bug, now fixed.** IDRiD numbers its train and test images from 001, and the
  indexer had merged the two landmark files by image name, so 206 images carried the other
  split's disc and fovea positions. Only the evaluation used these coordinates; nothing was
  trained on them. `scripts/fix_idrid_landmarks.py` repaired the manifests, and the figures
  above are from the corrected labels.
- **The vessel head only knows HRF.** It was trained on HRF alone, and on other cameras its
  probabilities run low: on the IDRiD demo photos it marks 0.3–5.6% of the field at 0.5, and
  on DRIVE it marks 1.9% where 12.5% is annotated, finding 9.8% of vessel pixels. This is not
  only a threshold problem. Even a threshold picked on DRIVE itself reaches F1 0.571, because
  DRIVE's 565 px photographs are enlarged about 2.7× onto the canvas, and the head has never
  seen vessels that blurred.
  The HRF-optimal threshold (0.85, F1 0.775, `torch/fit_vessel_threshold.py`) would drop that
  to 0–2% and switch off the neovascularisation density rule, so the deployed threshold stays
  at 0.5. Vessel density off HRF should be read as a lower bound.
  Retraining the vessel head on DRIVE, CHASE_DB1 and STARE alongside HRF is the fix. Until then,
  vessel output is for display, and only the (experimental) neovascularisation rule consumes it.
- **Our IDRiD grading test set is a split of our own**, not the official IDRiD test set.

---

## Running it

Tested on Windows and on Apple Silicon (M-series). There is no CUDA requirement — the engine
picks CUDA when it is there and CPU otherwise, and CPU inference grades an eye in a couple of
seconds. On a Mac it runs on CPU, which is deliberate: Apple's MPS backend still has gaps in
operator coverage that would surface mid-demo.

**Requirements:** Python 3.10+ and Node 18+.

### 1. Backend

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn certus_api.main:app --host 127.0.0.1 --port 8000
```

First start creates `api/certus.db` and `api/storage/`. The model checkpoint and the fitted
trust layer are already in the repo under `runs_torch/train_20260912_211356/`, and are found
automatically — no paths to configure.

Check it came up:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/trust      # should report "fitted": true
```

### 2. Frontend

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open the URL it prints (usually <http://localhost:5173>).

### 3. Sign in

Under *or enter as a demo user*, click the role you want. No API key needed — the button fills
one in and checks it against `/v1/me` before letting you through.

| Role | Sees |
|---|---|
| Technician | Capture, Status, Settings |
| Ophthalmologist | Review, Status, Settings |
| Administrator | everything, including Dashboard, Cameras, Model, Audit, People |

---

## Demo images

`demo_images/` holds six full-resolution fundus photographs (4288 × 2848) from the IDRiD
grading set. **Every one is from the held-out test split** — none was seen in training,
checkpoint selection, or calibration. The filename carries the ophthalmologist's grade, so a
live upload can be checked against ground truth on the spot.

| File | True ICDR grade | Referable |
|---|---|---|
| `grade0_no_dr_IDRiD_029.jpg` | 0 — no DR | no |
| `grade0_no_dr_IDRiD_030.jpg` | 0 — no DR | no |
| `grade1_mild_IDRiD_063.jpg` | 1 — mild NPDR | no |
| `grade2_moderate_IDRiD_008.jpg` | 2 — moderate NPDR | **yes** |
| `grade3_severe_IDRiD_006.jpg` | 3 — severe NPDR | **yes** |
| `grade4_proliferative_IDRiD_001.jpg` | 4 — proliferative DR | **yes** |

Referable DR is grade ≥ 2 — the threshold at which a patient needs an ophthalmologist.

To run a visit: sign in as a technician, open **Capture**, register a patient, attach one image
as the right eye and another as the left, then grade the visit. Grading takes a few seconds per
eye on CPU.

---

## The MATLAB deliverable

The same checkpoint also grades photographs inside MATLAB, so the design can be inspected
in MathWorks tooling rather than taken on trust:

```matlab
cd matlab
startup_certus
certus_demo        % grades demo_images/ and draws a report per eye
```

Only the weights come from the PyTorch run -- exported to ONNX and imported with
`importNetworkFromONNX`. The tiling, attention pooling, evidence vector, temperature
scaling, screening band, the two-reader rule and Grad-CAM are MATLAB code. On the six demo
images it returns the same grade, the same referral decision and the same quality class as the
API, p(referable) within 0.0021, the second reader's score within 0.012 log-odds, and Grad-CAM
maps that correlate at r ≥ 0.994. `torch/agreement.py` re-checks
this and fails if it drifts; `matlab/README.md` has the details.

Needs Deep Learning Toolbox, Image Processing Toolbox, and the Deep Learning Toolbox
Converter for ONNX Model Format. No GPU and no dataset required.

---

## Layout

```
api/          FastAPI backend: jobs, audit chain, blob storage, inference engine
torch/        model, training, calibration
  certus/     CertusNet, losses, data, trust layer
  train.py    training loop
  calibrate.py fits temperature, conformal bands, per-camera trust table
scripts/      dataset preparation; retina.py does FOV normalisation
web/          Vite + React console
matlab/       MATLAB: ONNX inference path + a full MATLAB training implementation
demo_images/  six held-out fundus photographs, graded
runs_torch/   the shipped checkpoint and its fitted trust layer
```

The dataset itself is not in the repo. It is not needed to run the demo — the API loads the
checkpoint and nothing else. Training requires it; see `scripts/`.

---

## Configuration

Everything is overridable from the environment with a `CERTUS_` prefix. Defaults resolve
relative to the checkout, so a fresh clone needs none of this.

| Variable | Default | Notes |
|---|---|---|
| `CERTUS_DEVICE` | `auto` | `auto` picks CUDA when present, else CPU. Force with `cpu` or `cuda`. |
| `CERTUS_CHECKPOINT` | newest `best.pt` under `runs_dir` | |
| `CERTUS_TRUST_JSON` | `trust.json` beside the checkpoint | |
| `CERTUS_RUNS_DIR` | `<repo>/runs_torch` | |
| `CERTUS_API_KEYS` | demo keys | `key:role` pairs |
| `CERTUS_NO_WORKER` | unset | set to `1` to disable the background grading worker |

Leave `CERTUS_NO_WORKER` alone. With it set, uploads queue jobs that nothing picks up and
grading appears to hang.

---

## What the model is

EfficientNet-B0 encoder with a U-Net decoder, trained on 26,809 eyes across APTOS, DDR, IDRiD
and EyePACS. Each eye is normalised onto a 1536 px canvas and read as nine 512 px tiles plus one
downscaled view of the whole retina; gated attention pools the tiles, so the weights that
produced a grade are the same weights shown as the explanation. Three heads: ordinal grading,
six-class lesion segmentation, and a three-class quality gate.

The grade head receives a 12-number evidence vector computed from the lesion masks through a
stop-gradient, so the grade depends on the lesions the model drew, but the model cannot learn to
draw convenient lesions to make grading easier.

The second reader is a separate EfficientNet-B0 trained end to end on the whole canvas at
512 px (`torch/train_baseline.py`, `WholeImageGrader` in `torch/certus/model.py`), on the same
training eyes. It has no tiles, lesions or quality head, and adds about 16 MB and one small
forward pass per eye.

Splits are disjoint by patient, not by image.

---

## Clinical Design & Dataset Selection Rationale

### 1. Retinal Vessel Dataset: HRF vs. DRIVE
While classical benchmarks frequently reference the DRIVE dataset ($565 \times 584$ px, 40 images), upscaling $565 \text{ px}$ images to Certus's native $1536 \times 1536 \text{ px}$ screening canvas causes severe interpolation blur, destroying 1–3 pixel capillary details. Certus utilizes the **High Resolution Fundus (HRF)** dataset ($3504 \times 2336 \text{ px}$), which mirrors the optical resolution of modern field cameras (e.g., Remidio FOP at $4288 \times 2848 \text{ px}$) and allows the 6-channel U-Net to train vessel segmentation (`VES`) at native optical fidelity without synthetic upsampling artifacts.

### 2. Anatomical Landmark Geometry for Fovea Localization
In severe diabetic retinopathy the foveal reflex is often hidden by macular oedema, dense exudates or haemorrhage, which is exactly where an intensity-only fovea detector fails. Certus places the fovea from the optic disc instead. The offset (0.379 field-of-view diameters temporal, 0.051 inferior) was fitted on IDRiD train. The estimate then moves to the darkest point of the smoothed green channel within half a disc diameter, so a visible pit is used when there is one and the geometric estimate stands when there is not. On IDRiD test the median error is 35 native px (mean 67; IDRiD challenge winner 64.5). The textbook 2.5-disc-diameter rule it replaced had a median error of 165 px on the same eyes.

### 3. Simulink Continuous-Flow Queueing vs. Discrete Entities
The telemedicine capacity model in `matlab/simulink/` simulates 100,000+ patients/year (400 patients/day over 250 camp-days). It utilizes a discrete-time fluid queue coupled with a Stateflow battery state machine to simulate rural grid outages. This provides deterministic, reproducible staffing and throughput sweeps without dependency on unverified SimEvents runtime binaries, accurately capturing camp queue dynamics and ophthalmologist review backlogs.

