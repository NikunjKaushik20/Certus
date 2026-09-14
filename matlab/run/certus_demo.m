function R = certus_demo(imagePath, opts)
%CERTUS_DEMO Grade one fundus photograph in MATLAB, using the shipped PyTorch weights.
%
%   R = certus_demo                       % grades every image in demo_images/
%   R = certus_demo("path/to/eye.jpg")    % grades one photograph
%   R = certus_demo(..., Figure=false)    % skip the report figure
%   R = certus_demo(..., Camera="Topcon_TRC_NW6")  % decide by that camera's own fitted band
%                                         % (torch/fit_camera.py), as the API does
%   R = certus_demo(..., HeatDir="out")   % also write each Grad-CAM map as a 16-bit PNG
%                                         % (torch/agreement.py compares them with Python)
%
%   This is the MathWorks deliverable: the network trained in PyTorch is exported to ONNX
%   (torch/export_onnx.py), imported with importNetworkFromONNX, and driven by the same
%   MATLAB forward pass the MATLAB training path uses (gradeForward). Tiling, attention
%   pooling, the evidence vector, temperature scaling and the screening band are all
%   MATLAB code here -- only the learned weights come from the Python run.
%
%   The grade must match what the API reports for the same file. It is the same weights and
%   the same arithmetic, so a mismatch means a preprocessing drift, which is why
%   normalizeFov mirrors scripts/retina.py rather than doing anything of its own.
%
%   Returns a table: file, grade, pReferable, decision, quality, and the per-lesion counts.

arguments
    imagePath string = ""
    opts.Figure (1,1) logical = true
    opts.OnnxDir string = ""
    opts.Reimport (1,1) logical = false
    opts.HeatDir string = ""
    opts.Camera string = ""
end

% Mirrors USABLE_WEIGHT / MIN_RELIABILITY in api/certus_api/inference.py. Two copies of three
% numbers is a real risk, so they are named identically in both places and the demo prints the
% reliability it used, which is what makes a drift visible rather than silent.
USABLE_WEIGHT = 0.75;
MIN_RELIABILITY = 0.70;

cfg = startup_certus();
onnxDir = opts.OnnxDir;
if onnxDir == "", onnxDir = cfg.paths.onnx; end
assert(isfolder(onnxDir), "No ONNX export at %s. Run: python torch/export_onnx.py " + ...
       "runs_torch/train_<stamp>/best.pt", onnxDir);

files = imagePath;
if files == ""
    d = [dir(fullfile(cfg.paths.demo, "*.jpg")); dir(fullfile(cfg.paths.demo, "*.png"))];
    assert(~isempty(d), "No photographs in %s", cfg.paths.demo);
    files = string(fullfile({d.folder}, {d.name}))';
end

model = loadModel(onnxDir, opts.Reimport);
if opts.Camera ~= ""
    [model.meta.screening_band, b2] = cameraBand(fullfile(fileparts(onnxDir), "trust.json"), opts.Camera, ...
                                                 model.meta.screening_band);
    if ~isempty(b2), model.meta.second_band = b2; end
end
fprintf("Certus %s | step %d | %s\n", "v0.4", model.meta.step, executor());
fprintf("temperature %.4f | clear below %.4f | refer above %.4f\n\n", ...
        model.meta.temperature, model.meta.screening_band.clear_below, ...
        model.meta.screening_band.refer_above);
if isfield(model, "second")
    fprintf("second reader: clear below %.3f | refer above %.3f (log-odds); auto-decide only when both agree\n\n", ...
            model.meta.second_reader.clear_below_logit, model.meta.second_reader.refer_above_logit);
end

rows = cell(numel(files), 1);
for k = 1:numel(files)
    rows{k} = gradeOne(model, files(k), cfg, opts.Figure, USABLE_WEIGHT, MIN_RELIABILITY, opts.HeatDir);
end
R = vertcat(rows{:});
end


% ------------------------------------------------------------------ one photograph
function row = gradeOne(model, file, cfg, showFigure, USABLE_WEIGHT, MIN_RELIABILITY, heatDir)
t0 = tic;
img = imread(file);
if size(img, 3) == 1, img = repmat(img, 1, 1, 3); end
[canvas, fov, ~, info] = normalizeFov(img, model.meta.canvas);
assert(info.fovOk, "Could not find the retina in %s", file);

X = single(canvas) / 255;
if canUseGPU, X = gpuArray(X); end
out = gradeForward(model, X, cfg, false);
z2 = secondReader(model, X);                      % second reader sees the unenhanced canvas, as in training

% Ordinal logits are P(grade > k) for k = 0..3. Temperature scaling is monotonic, so it
% cannot change the ranking or the AUC -- it only makes the number mean what it says, and
% that is what the screening band is cut on, so it has to be applied before deciding.
logits = num(out.gradeLogits);
calP = cummin(1 ./ (1 + exp(-logits / model.meta.temperature)));
grade = sum(calP > 0.5);
pRef = calP(2);                                   % k = 1 is "grade >= 2", referable DR

% Softmax by hand rather than dlarray/softmax: the imported network's predict returns
% unlabelled data, so there is no 'C' dimension for it to reduce over.
ql = num(out.qualLogits);
qp = exp(ql - max(ql));
qp = qp / sum(qp);
qNames = ["good" "usable" "reject"];
[qBest, qi] = max(qp);
qLabel = qNames(qi);

% Adaptive enhancement for borderline ('usable') images:
% When an image is usable (borderline), feed enhanced canvas back into the model
% so the enhancement directly refines grading logits, lesion maps, and vessels!
enhInfo = struct("applied", false);
if qLabel == "usable"
    [canvasEnh, enhInfo] = adaptiveEnhance(canvas, qLabel);
    if enhInfo.applied
        canvas = canvasEnh;
        X_enh = single(canvas) / 255;
        if canUseGPU, X_enh = gpuArray(X_enh); end
        X = X_enh;                                   % Grad-CAM must see the input that produced out
        out = gradeForward(model, X, cfg, false);
        logits = num(out.gradeLogits);
        calP = cummin(1 ./ (1 + exp(-logits / model.meta.temperature)));
        grade = sum(calP > 0.5);
        pRef = calP(2);
    end
end

qualityFactor = qp(1) + USABLE_WEIGHT * qp(2);            % p(good) + 0.75 * p(usable)
cameraFactor = 1.0;
reliability = qualityFactor * cameraFactor;

band = model.meta.screening_band;
if reliability < MIN_RELIABILITY
    decision = "refer-to-human";                 % conditions for believing any score are absent
    why = "low reliability";
elseif pRef > band.refer_above
    [decision, why] = twoReaderCall(model, "referable", z2, secondBand(model));
elseif pRef <= band.clear_below
    [decision, why] = twoReaderCall(model, "non-referable", z2, secondBand(model));
else
    decision = "refer-to-human";                 % the model declines to decide
    why = "ambiguous";
end

if pRef <= band.clear_below
    % Guard matches Python _decisiveness(): "if clear > 0 else 1.0"
    decisiveness = (band.clear_below - pRef) / max(band.clear_below, 1e-9);
elseif pRef > band.refer_above
    % Guard matches Python: "if refer < 1 else 1.0"
    decisiveness = (pRef - band.refer_above) / max(1 - band.refer_above, 1e-9);
else
    decisiveness = 0;
end
trust = round(100 * reliability * decisiveness);

maps = num(out.lesionProb);            % DxDx4, MA HE EX SE
thr = model.meta.lesion_thresholds;
counts = zeros(1, 4);
for c = 1:4
    m = bwareaopen(maps(:, :, c) >= thr.(cfg.lesionClasses(c)), minArea(c), 8);
    counts(c) = bwconncomp(m, 8).NumObjects;
end

% Vessel segmentation from channel 6
vesProb = num(out.vesProb);
vesThr = 0.5;
if isfield(thr, "VES"), vesThr = thr.VES; end   % 0.5; see VES_note in trust.json
vesMask = (vesProb >= vesThr) & fov;
vesDensity = round(100 * nnz(vesMask) / max(nnz(fov), 1), 2);

% Optic Disc and Fovea localization
D = cfg.canvas;
DISC_PX_MIN = 110; DISC_PX_MAX = 340;   % matches inference.py DISC_PX_MIN/MAX
macCounts = zeros(1, 4);                 % per-class macular count
foveaStr = "fovea not located";
[struc, foveaFound, fx, fy, macR] = estimateFoveaFromOD(out.odProb, D, DISC_PX_MIN, DISC_PX_MAX, canvas);
if foveaFound
    foveaStr = sprintf("fovea at (%.0f,%.0f) DD=%.0f px", fx, fy, macR);
    for c = 1:4
        m = bwareaopen(maps(:, :, c) >= thr.(cfg.lesionClasses(c)), minArea(c), 8);
        cc = bwconncomp(m, 8);
        s  = regionprops(cc, "Centroid");
        for r = 1:cc.NumObjects
            cx = s(r).Centroid(1); cy = s(r).Centroid(2);
            if (cx-fx)^2 + (cy-fy)^2 <= macR^2
                macCounts(c) = macCounts(c) + 1;
            end
        end
    end
end

% Sub-pixel Microaneurysm (MA) localization
maMask = bwareaopen(maps(:, :, 1) >= thr.MA, minArea(1), 8);
maSubTable = subpixelMA(maps(:, :, 1), maMask, [fx, fy]);

% Hemorrhage subclassification (dot/blot vs flame vs preretinal)
heMask = bwareaopen(maps(:, :, 2) >= thr.HE, minArea(2), 8);
heCC = bwconncomp(heMask, 8);
heStats = regionprops(heCC, "Area", "BoundingBox");
heDot = 0; heFlame = 0; hePre = 0;
for hi = 1:heCC.NumObjects
    a_i = heStats(hi).Area;
    bb_i = heStats(hi).BoundingBox;
    asp = max(bb_i(3), bb_i(4)) / max(min(bb_i(3), bb_i(4)), 1);
    if a_i > 1500
        hePre = hePre + 1;
    elseif asp > 2.0
        heFlame = heFlame + 1;
    else
        heDot = heDot + 1;
    end
end
heSubtypes = struct("dot_blot", heDot, "flame", heFlame, "preretinal", hePre);

% Neovascularization clinical screening assessment
nv = neovascularizationDetect(vesProb, struc, grade, counts(2), hePre, vesDensity);

% Native MATLAB Grad-CAM calculation
[heatCam, blendCam] = certusGradCam(model, X, out, cfg, fov, canvas);
if heatDir ~= ""
    if ~isfolder(heatDir), mkdir(heatDir); end
    [~, stem] = fileparts(file);
    imwrite(uint16(round(heatCam * 65535)), fullfile(heatDir, stem + "_gradcam.png"));
end

ms = toc(t0) * 1000;

if isempty(z2), z2show = NaN; else, z2show = z2; end
fprintf("%-34s grade %d  p %.4f  2nd %+6.2f  trust %3d  %-15s quality %s (%.2f)  %s%4.0f ms\n", ...
        nameOf(file), grade, pRef, z2show, trust, decision, qNames(qi), qBest, ...
        pad(why), ms);
fprintf("                                    MA %d  HE %d  EX %d  SE %d   mac(%d %d %d %d)   %s\n", ...
        counts(1), counts(2), counts(3), counts(4), ...
        macCounts(1), macCounts(2), macCounts(3), macCounts(4), foveaStr);
fprintf("                                    vessels: %.1f%%  subpixel MAs: %d  HE: [dot %d, flame %d, pre %d]  NV: %s (risk %.2f)\n", ...
        vesDensity, height(maSubTable), heDot, heFlame, hePre, nv.criteria, nv.nv_risk_score);

row = table(nameOf(file), grade, pRef, z2show, trust, reliability, decision, string(why), qNames(qi), ...
            counts(1), counts(2), counts(3), counts(4), vesDensity, height(maSubTable), round(ms), ...
            VariableNames=["file" "grade" "pReferable" "secondLogit" "trust" "reliability" "decision" ...
                           "reason" "quality" "MA" "HE" "EX" "SE" "vesDensityPct" "subpixelMA" "ms"]);

if showFigure
    report(canvas, fov, maps, out, calP, grade, pRef, decision, ...
           qNames(qi), qBest, thr, cfg, nameOf(file), info, model.meta.screening_band, ...
           blendCam, vesMask, maSubTable, heSubtypes, nv, enhInfo, trust, reliability);
end
end


function s = pad(why)
if why == "", s = ""; else, s = sprintf("(%s) ", why); end
end


function x = num(v)
%NUM Plain double on the host. gradeForward hands back a mix: the head outputs are dlarray,
%   while the attention weights and the evidence vector have already been stripped, so one
%   unconditional extractdata would error on half of them.
if isa(v, "dlarray"), v = extractdata(v); end
x = double(gather(v));
end


function a = minArea(c)
floors = [2 4 4 4];       % MA HE EX SE
a = floors(c);
end


function n = nameOf(f)
[~, b, e] = fileparts(f);
n = b + e;
end


function s = executor()
if canUseGPU
    g = gpuDevice;
    s = sprintf("GPU %s", g.Name);
else
    s = "CPU";
end
end


function b2 = secondBand(model)
%SECONDBAND The camera's own second-reader band when one was fitted, else the global one.
b2 = [];
if isfield(model.meta, "second_band"), b2 = model.meta.second_band; end
end


function [band, band2] = cameraBand(trustPath, camera, fallback)
%CAMERABAND The screening band fitted for this camera, or the global one if it has none.
%   band2 is the camera's second-reader band when fit_camera.py fitted one, else [].
band = fallback;
band2 = [];
t = jsondecode(fileread(trustPath));
key = matlab.lang.makeValidName(camera);
if isfield(t, "per_domain") && isfield(t.per_domain, key) && isfield(t.per_domain.(key), "screening_band")
    b = t.per_domain.(key).screening_band;
    band.clear_below = b.clear_below;
    band.refer_above = b.refer_above;
    fprintf("camera %s: own band (%s)\n", camera, b.fitted_on);
    if isfield(t.per_domain.(key), "second_reader_band")
        band2 = t.per_domain.(key).second_reader_band;
    end
else
    fprintf("camera %s: no fitted band, using the global one (unverified camera)\n", camera);
end
end


% ------------------------------------------------------------------ model cache
function model = loadModel(onnxDir, reimport)
cache = fullfile(onnxDir, "certus_matlab_model.mat");
graphs = [dir(fullfile(onnxDir, "*.onnx")); dir(fullfile(onnxDir, "certus_meta.json"))];   % meta edits must invalidate too
newest = max([graphs.datenum]);
if ~reimport && isfile(cache)
    c = dir(cache);
    if c.datenum >= newest
        S = load(cache, "model");
        model = S.model;
        if canUseGPU, model = modelToGpu(model); end
        return
    end
end
fprintf("importing ONNX (about a minute, cached afterwards)...\n");
model = importCertusOnnx(onnxDir);
save(cache, "model", "-v7.3");
end


% ------------------------------------------------------------------ report figure
function report(canvas, fov, maps, out, calP, grade, pRef, decision, qName, qProb, thr, cfg, name, info, band, ...
                blendCam, vesMask, maSubTable, heSubtypes, nv, enhInfo, trust, reliability)
%REPORT Comprehensive clinical explainability report figure:
%   1. Raw fundus canvas with FOV scale
%   2. Multiclass lesion segmentation at calibrated thresholds
%   3. Native Grad-CAM attention heatmap overlay
%   4. Retinal vessel tree with sub-pixel microaneurysm centroids
%   5. Ordinal class probabilities + Stop-gradient evidence breakdown
%   6. Clinical decision summary + Hemorrhage subtypes & Neovascularization risk
colors = [1 0.25 0.25; 0.85 0 0.6; 1 0.85 0.1; 0.2 0.9 1];   % MA HE EX SE
f = figure(Name="Certus Clinical Report - " + name, Color="w", Position=[50 50 1350 780]);
if exist("theme", "file"), theme(f, "light"); end
tiledlayout(f, 2, 3, TileSpacing="compact", Padding="compact");

% Panel 1: Canvas
nexttile; imshow(canvas);
subText = "";
if isfield(enhInfo, "applied") && enhInfo.applied
    subText = " [Adaptive Enhancement: CLAHE + Denoise]";
end
title(sprintf("%s (native %.0f px)%s", name, info.nativeFovPx, subText), ...
      FontWeight="normal", Interpreter="none");

% Panel 2: Lesions at calibrated thresholds
nexttile;
ov = im2double(canvas);
for c = 1:4
    m = maps(:, :, c) >= thr.(cfg.lesionClasses(c));
    m = m & fov;
    for ch = 1:3
        layer = ov(:, :, ch);
        layer(m) = 0.35 * layer(m) + 0.65 * colors(c, ch);
        ov(:, :, ch) = layer;
    end
end
imshow(ov);
title("Lesion segmentation (calibrated thresholds)", FontWeight="normal");
legendStrings = arrayfun(@(c) sprintf("%s %.3f", cfg.lesionClasses(c), ...
                         thr.(cfg.lesionClasses(c))), 1:4, UniformOutput=false);
legendStrings = string(legendStrings);
text(10, 40, strjoin(legendStrings, "   "), Color="w", FontSize=9, Interpreter="none");

% Panel 3: Native Grad-CAM Heatmap
nexttile;
imshow(blendCam);
title("Explainability: Native Grad-CAM (referable DR)", FontWeight="normal");

% Panel 4: Retinal vessels & Sub-pixel MA centroids
nexttile;
vImg = im2double(canvas);
% Tint vessels in bright cyan
for ch = 1:3
    vCh = vImg(:, :, ch);
    cVal = [0.1, 0.9, 0.9];
    vCh(vesMask) = 0.4 * vCh(vesMask) + 0.6 * cVal(ch);
    vImg(:, :, ch) = vCh;
end
imshow(vImg); hold on;
% Plot sub-pixel microaneurysms as yellow '+' marks
if height(maSubTable) > 0
    plot(maSubTable.x_sub, maSubTable.y_sub, 'y+', 'MarkerSize', 5, 'LineWidth', 1.2);
end
hold off;
title(sprintf("Retinal vessels & Sub-pixel MAs (N=%d)", height(maSubTable)), FontWeight="normal");

% Panel 5: Ordinal probabilities and evidence
nexttile;
b = bar(0:3, calP, FaceColor="flat");
b.CData = repmat([0.6 0.7 0.8], 4, 1);
b.CData(2, :) = [0.8 0.2 0.2];                    % k=1 is the referable decision
ylim([0 1]); grid on;
xticklabels(["P(>0)" "P(>1)" "P(>2)" "P(>3)"]);
title("Ordinal Probabilities (Calibrated)", FontWeight="normal");
yline(0.5, "k--");
yline(band.clear_below, "g:", sprintf("auto-clear %.3f", band.clear_below));
yline(band.refer_above, "r:", sprintf("auto-refer %.3f", band.refer_above));

% Panel 6: Clinical Decision & Diagnostics Box
nexttile; axis off;
switch decision
    case "referable",      col = [0.75 0.1 0.1];
    case "non-referable",  col = [0.1 0.5 0.2];
    otherwise,             col = [0.85 0.55 0];
end
text(0, 0.95, sprintf("ICDR Grade %d", grade), FontSize=22, FontWeight="bold");
text(0, 0.80, upper(decision), FontSize=17, FontWeight="bold", Color=col);
text(0, 0.68, sprintf("p(referable) = %.4f | Trust = %d / 100", pRef, trust), FontSize=11);
text(0, 0.57, sprintf("Quality: %s (%.2f)  |  Reliability: %.2f", qName, qProb, reliability), FontSize=11);

% Hemorrhage subclassification & Neovascularization findings
heStr = sprintf("Hemorrhages: %d dot/blot, %d flame, %d preretinal", ...
                heSubtypes.dot_blot, heSubtypes.flame, heSubtypes.preretinal);
text(0, 0.44, heStr, FontSize=10, Color=[0.2 0.2 0.2]);

nvStr = sprintf("NV Assessment: %s (risk %.2f)", nv.criteria, nv.nv_risk_score);
nvCol = [0.1 0.5 0.2];
if nv.nvd_detected || nv.nve_detected, nvCol = [0.8 0.1 0.1]; end
text(0, 0.32, nvStr, FontSize=10, FontWeight="bold", Color=nvCol);

msg = sprintf("Screening Band (Val fit):\n  Clear <= %.4f\n  Refer > %.4f", ...
              band.clear_below, band.refer_above);
text(0, 0.12, msg, FontSize=9, Color=[0.4 0.4 0.4]);
end


function [struc, foveaFound, fx, fy, macR] = estimateFoveaFromOD(odProb, D, DISC_MIN, DISC_MAX, canvas)
%ESTIMATEFOVEAFROMOD Locate optic disc and fovea from existing OD segmentation.
foveaFound = false; fx = NaN; fy = NaN; macR = NaN; struc = struct();
if isa(odProb, "gpuArray"), odProb = gather(odProb); end
odMask = squeeze(double(odProb)) > 0.5;

if sum(odMask(:)) < 50
    struc.disc_found = false;
    return;
end

cc = bwconncomp(odMask, 8);
if cc.NumObjects == 0
    struc.disc_found = false;
    return;
end
compSizes = cellfun(@numel, cc.PixelIdxList);
[~, bigIdx] = max(compSizes);
bigMask = false(D, D);
bigMask(cc.PixelIdxList{bigIdx}) = true;

[ys, xs] = find(bigMask);
ox = mean(xs); oy = mean(ys);
dd = 2 * sqrt(sum(bigMask(:)) / pi);

struc.disc_found = true;
struc.disc_x = ox; struc.disc_y = oy; struc.disc_diameter_px = dd;

if dd < DISC_MIN || dd > DISC_MAX
    struc.fovea_found = false;
    struc.reason = sprintf("disc diameter %.0f px outside plausible %d-%d px range", dd, DISC_MIN, DISC_MAX);
    return;
end

[fx, fy] = foveaFromDisc(ox, oy, dd, D, canvas);   % fitted on IDRiD train, same rule as the API
inside = fx >= 0 && fx < D && fy >= 0 && fy < D;

if ~inside
    struc.fovea_found = false;
    return;
end

foveaFound = true;
macR = dd;    % one disc diameter radius
struc.fovea_found = true;
struc.fovea_x = fx; struc.fovea_y = fy;
struc.macula_radius_px = macR;
end



function ink = pickInk(v, vmax)
ink = "white";
if v > 0.6 * vmax, ink = "black"; end
end


function [struc, foveaFound, fx, fy, macR] = estimateFovea(model, X, cfg, D, DISC_MIN, DISC_MAX)
%ESTIMATEFOVEA Locate the optic disc and estimate the fovea by geometry.
%   Mirrors api/certus_api/inference.py Engine._structures().
%
%   The ONNX encoder has 6 segmentation channels (MA HE EX SE OD VES). gradeForward
%   only exposes the first 4 as lesionProb. We need channel 5 (OD) so we call the
%   encoder one more time with no_grad (predict, not forward).
%
%   Key decisions matching the Python implementation:
%   1. Use the LARGEST connected component of the OD mask, not the raw centroid of
%      all OD pixels. Raw means of multi-region masks are pulled toward noise on the
%      opposite side of the image, placing the fovea hundreds of pixels off.
%   2. Compute diameter from sqrt(area/pi)*2 rather than bounding-box width, which
%      vessel arcades can stretch.
%   3. Refuse to locate the fovea when dd is outside [DISC_MIN, DISC_MAX] px, and
%      return disc_found=false when OD area < 50 px (scattered noise).

foveaFound = false; fx = NaN; fy = NaN; macR = NaN; struc = struct();

g2 = cfg.grid^2; nt = g2;
tiles = makeTiles(X, cfg);                              % 512x512x3x9
glob  = imresize(X, [cfg.globalSize cfg.globalSize]);   % 512x512x3x1 — must match tile size
% Mirrors gradeForward.m line: cat(4, tiles, glob). X is 1536px; concatenating it
% directly fails because tiles are 512px and cat(4) requires matching S dimensions.
Xbatch = dlarray(cat(4, tiles, glob) * model.meta.inputScale, "SSCB");
seg = predict(model.enc, Xbatch, Outputs=model.meta.segOutput);

% seg shape: S S C(6) B(nt+1). We only need tiles (first nt batches), channel 5 = OD.
% After stitching the 4 lesion channels in gradeForward, same stitching for OD:
% extractdata strips the dlarray but the underlying tensor is still a gpuArray when the
% model is on GPU. bwconncomp / bwareaopen are CPU-only (no gpuArray support in IPT).
% double(gather(...)) matches the same pattern used by the num() helper above.
odRaw    = double(gather(extractdata(sigmoid(seg(:, :, 5, 1:nt)))));  % CPU double
odCanvas = stitchTiles(odRaw, cfg);          % D x D x 1 x 1, CPU
odMask   = squeeze(odCanvas) > 0.5;          % D x D logical, CPU

if sum(odMask(:)) < 50
    struc.disc_found = false;
    return;
end

% Largest connected component: prevents noise elsewhere from pulling the centroid.
cc = bwconncomp(odMask, 8);
if cc.NumObjects == 0
    struc.disc_found = false;
    return;
end
compSizes = cellfun(@numel, cc.PixelIdxList);
[~, bigIdx] = max(compSizes);
bigMask = false(D, D);
bigMask(cc.PixelIdxList{bigIdx}) = true;

[ys, xs] = find(bigMask);
ox = mean(xs); oy = mean(ys);
dd = 2 * sqrt(sum(bigMask(:)) / pi);

struc.disc_found = true;
struc.disc_x = ox; struc.disc_y = oy; struc.disc_diameter_px = dd;

if dd < DISC_MIN || dd > DISC_MAX
    struc.fovea_found = false;
    struc.reason = sprintf("disc diameter %.0f px outside plausible %d-%d px range", dd, DISC_MIN, DISC_MAX);
    return;
end

% Fovea sits ~2.5 disc diameters temporal to the disc centre, marginally inferior.
% temporal_sign: fovea lies away from the nasal edge. Disc on the right -> nasal.
[fx, fy] = foveaFromDisc(ox, oy, dd, D);   % fitted on IDRiD train, same rule as the API
inside = fx >= 0 && fx < D && fy >= 0 && fy < D;

if ~inside
    struc.fovea_found = false;
    return;
end

foveaFound = true;
macR = dd;    % one disc diameter radius, the clinical convention
struc.fovea_found = true;
struc.fovea_x = fx; struc.fovea_y = fy;
struc.macula_radius_px = macR;
end
