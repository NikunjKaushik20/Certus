# Certus — MATLAB

Two separate things live here. **Inference** runs the shipped PyTorch checkpoint inside
MATLAB and needs no dataset and no GPU. **Training** is a full MATLAB implementation of the
same design and needs both.

---

## Inference: grading a photograph in MATLAB

```matlab
cd <repo>/matlab
startup_certus                                        % sets the path
certus_demo                                           % grades everything in demo_images/
certus_demo("../demo_images/grade3_severe_IDRiD_006.jpg")
```

The first call imports the ONNX graphs, which takes about a minute; the result is cached
next to them and reused after that. Each eye then grades in 1-3 s.

**Requires** Deep Learning Toolbox, Image Processing Toolbox, and the *Deep Learning
Toolbox Converter for ONNX Model Format* support package. Parallel Computing Toolbox is
used if present and not required.

### What is MATLAB and what is not

Only the learned weights come from Python. `torch/export_onnx.py` writes four graphs --
encoder+decoder, attention scorer, grade head, quality head -- and checks each against
PyTorch through onnxruntime (max absolute difference 4e-5 or better, asserted at export).
Everything between them is MATLAB code, and it is the same code the MATLAB training path
uses:

| Step | Where |
|---|---|
| FOV normalisation to a 1536 px canvas | `data/normalizeFov.m` |
| 3x3 tiling plus the downscaled global view | `model/makeTiles.m` |
| gated attention pooling over the nine tiles | `model/gradeForward.m` |
| 12-number lesion evidence vector, stop-gradient | `model/evidenceFeatures.m` |
| temperature scaling, cumulative-min, ordinal decode | `run/certus_demo.m` |
| screening band (auto-clear / auto-refer / abstain) | `run/certus_demo.m` |
| Grad-CAM on the referable logit | `model/certusGradCam.m` |

### Grad-CAM

`certusGradCam` computes the same map as `api/certus_api/gradcam.py`: per tile, the stride-32
activation weighted by the spatially averaged gradient of the referable logit. The tile
embedding is the spatial mean of that activation, so the gradient it needs is
d(logit)/d(embedding) / 256. `dlgradient` gets that exactly from the attention scorer, the
tile softmax and the grade head, without differentiating the encoder. The activation map
comes from a fifth ONNX graph, `certus_feat.onnx`
(`python torch/export_onnx.py <best.pt> --only feat`). It takes under a second per eye on
the GPU; the Python version runs nine encoder backward passes.

### Does it agree with the console?

It has to, or the deliverable is decoration. `torch/agreement.py` checks it and exits
non-zero if anything drifts:

```bash
matlab -batch "startup_certus; R = certus_demo(Figure=false, HeatDir='../reports/agreement'); writetable(R, '../reports/agreement/matlab.csv')"
CERTUS_DEVICE=cpu python torch/agreement.py reports/agreement      # fp32 on the Python side
```

Last run, six held-out demo images, MATLAB against `api/certus_api/inference.py` in fp32:

| | agreement |
|---|---|
| ICDR grade | 6 of 6 identical |
| referral decision | 6 of 6 identical |
| quality class | 6 of 6 identical |
| lesion counts (MA/HE/EX/SE) | 5 of 6 identical; one differs by 2 haemorrhages of 114 |
| p(referable) | within 0.0021 |
| second reader, log-odds of referable | within 0.012 |
| Grad-CAM map | Pearson r 0.994–0.999 inside the FOV, mean abs difference ≤ 0.002 |

Against the engine on CUDA, which runs under autocast, p(referable) differs by up to 0.0034
and lesion counts match on 1 of 6. The difference is precision, not a different computation.

The screening band cuts at 0.047 and 0.225, so a 0.002 wobble cannot move a decision unless
an eye sits exactly on a cut. The second reader's cuts are at -5.15 and -2.29 log-odds, so
0.012 is equally far from mattering. All six demo eyes are far from every cut, so this run
shows that the scores match. It does not show a disagreement being handled; that logic is
`model/twoReaderCall.m` against `Engine.decide`, and both follow the same three lines.

### The second reader

The referral decision uses two readers (see the main README): Certus, and a plain
EfficientNet-B0 on the whole canvas at 512 px. The second one is `certus_second.onnx`
(`python torch/export_onnx.py <best.pt> --only second`, max |torch − onnxruntime| 4.6e-5).
`model/secondReader.m` resizes the unenhanced canvas with antialiased bilinear, as torch's
`F.interpolate(antialias=True)` did in training, and returns the log-odds of referable.
`model/twoReaderCall.m` lets Certus's automatic call stand only if that score is outside its
own band on the same side. `certus_demo` prints the score as `2nd` and gives the reason when
the readers disagree. `certus_demo(Camera=...)` also loads a camera's own second-reader band
when `torch/fit_camera.py` fitted one. Without `certus_second.onnx` Certus decides alone, as
before.

Getting there needed two preprocessing details, because the network was trained on canvases
`scripts/retina.py` wrote and a different canvas is a different input:

- `data/areaResize.m` reproduces `cv2.INTER_AREA` rather than using `imresize("box")`. The
  two differ on a few pixels at a region edge; that moved the detected FOV bounding box by
  one pixel on IDRiD_008, shifted the canvas by ~8 raw pixels, and moved p(referable) from
  0.9955 to 0.9902.
- `data/nearestResize.m` reproduces `cv2.INTER_NEAREST`, which samples at
  `floor(dst * src/dst)` where MATLAB samples from pixel centres. Half a pixel, on the FOV
  mask, put the rim one pixel out.

With both in place the MATLAB and Python canvases differ on 14-134 pixels out of 7,077,888
(under 0.002%), all on the circle rim, where `cv2.circle` and a rasterised disc disagree.

---

## Training

Everything runs on the GPU (CUDA via Parallel Computing Toolbox). Data decoding runs on
`backgroundPool` threads and overlaps with GPU compute.

Needs the prepared dataset (`scripts/`) and a CUDA GPU.

### Run order
```matlab
cd <repo>/matlab
startup_certus
smoke_test      % checks env, data, gradients, measures batch sizes -> runs/batch_probe.json
lr_finder       % LR range test on the full multi-task loss         -> runs/lr_finder.json + .png
train_certus    % full training (only after the two files above exist)
```
Nothing is guessed: `certus_config.m` reads the batch sizes and learning rates from those
JSON files, and `train_certus` refuses to start without them.

### Model (one network)
| Part | What it does |
|---|---|
| Encoder | ImageNet EfficientNet-B0 on 512 px tiles; the 1536 px canvas is cut into 3x3 tiles (native lesion scale) plus one downscaled 512 px global view |
| Decoder | U-Net style, strides 32→16→8→4→2, GroupNorm; outputs MA, HE, EX, SE, OD, vessel logits at tile resolution |
| Attention MIL | Gated attention pools the 9 tile embeddings of one eye |
| Evidence | 12 numbers from the lesion maps: lesion amounts, peak probabilities, haemorrhage spread over quadrants (4-2-1 rule). Stop-gradient: the grade uses the lesion maps but cannot reshape them |
| Grade head | [pooled tiles, global view, evidence] → 4 cumulative ordinal logits P(grade > k); logit k=1 is referable DR |
| Quality head | Global view → good / usable / reject |

### Training recipe
- Losses: ordinal BCE (referable threshold ×2) · focal BCE + focal Tversky (α 0.3, β 0.7) with per-image masking of unannotated classes · partial-label cross-entropy for quality (graded clinical images count as "not reject").
- Task weights learned (homoscedastic uncertainty).
- One step = grade sub-batches (gradient accumulation) + seg patches + quality views, run sequentially so memory peaks at the largest sub-batch.
- AdamW, decoupled weight decay (weights only), encoder LR = head LR × ratio, warmup + cosine, global-norm clipping, weight EMA for evaluation.
- Encoder BN runs in inference mode (ImageNet statistics) because a grade batch is 10 correlated tiles; all encoder weights still train.
- GPU augmentation: flips, rotation, 90° turns (patches), brightness/contrast/gamma/colour, and acquisition degradations (defocus blur, vignetting, sensor noise) standing in for portable-camera images. Quality views skip the degradations because they would change the label.
- Sampling: grade and source dataset balanced with weights ∝ count^-0.5; seg patches 70% centred on a lesion (MA first).

### Validation
`evaluateModel` reports referable AUC, sensitivity at 85% specificity, sensitivity/specificity at 0.5, quadratic kappa, ECE, per-dataset AUC, lesion Dice and quality accuracy. The best checkpoint is picked on the **val** split only; Messidor-2 (`part="external"`) is kept for the final, one-time external test.

## Folders
`data/` manifests, sampling, background loading · `model/` network, tiling, evidence, forward pass · `train/` losses, augmentation, optimiser, step · `eval/` metrics · `run/` entry points.
