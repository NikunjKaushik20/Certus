function outPath = exportCertusReport(imagePath, opts)
%EXPORTCERTUSREPORT Generate exportable publication-grade clinical DR screening report.
%   Analyzes the fundus photograph and exports a comprehensive clinical
%   diagnostic report in PDF or high-resolution PNG format, suitable for
%   human-in-the-loop ophthalmologist validation in rural telemedicine camps.
%
%   outPath = exportCertusReport("path/to/eye.jpg")
%   outPath = exportCertusReport(..., Format="pdf")
%   outPath = exportCertusReport(..., OutDir="reports")
%
%   Organized for SIH 2026 Problem Statement 26038 (MathWorks).

arguments
    imagePath (1,1) string = "../demo_images/grade3_severe_IDRiD_006.jpg"
    opts.Format (1,1) string {mustBeMember(opts.Format, ["png", "pdf"])} = "png"
    opts.OutDir (1,1) string = ""
end

cfg = startup_certus();
cache = fullfile(cfg.paths.onnx, "certus_matlab_model.mat");
src = [dir(fullfile(cfg.paths.onnx, "*.onnx")); dir(fullfile(cfg.paths.onnx, "certus_meta.json"))];
c = dir(cache);
if isfile(cache) && c.datenum >= max([src.datenum])     % a stale cache would lack the Grad-CAM graph
    S = load(cache, "model");
    model = S.model;
    if canUseGPU, model = modelToGpu(model); end
else
    model = importCertusOnnx(cfg.paths.onnx);
end

img = imread(imagePath);
if size(img, 3) == 1, img = repmat(img, 1, 1, 3); end
[canvas, fov, ~, info] = normalizeFov(img, model.meta.canvas);
assert(info.fovOk, "Could not locate retina in %s", imagePath);

X = single(canvas) / 255;
if canUseGPU, X = gpuArray(X); end
out = gradeForward(model, X, cfg, false);

z2 = secondReader(model, X);                      % unenhanced canvas, as in training

% Probabilities and Grade
logits = num(out.gradeLogits);
calP = cummin(1 ./ (1 + exp(-logits / model.meta.temperature)));
grade = sum(calP > 0.5);
pRef = calP(2);

% Quality Gate
ql = num(out.qualLogits);
qp = exp(ql - max(ql)); qp = qp / sum(qp);
qNames = ["good" "usable" "reject"];
[qBest, qi] = max(qp);
qLabel = qNames(qi);

% Adaptive Enhancement:
% If usable (borderline), feed enhanced canvas back into the model for refined grading
enhInfo = struct("applied", false, "methods", string.empty);
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

% Screening band & Decision
band = model.meta.screening_band;
call = "";
if pRef > band.refer_above, call = "referable"; elseif pRef <= band.clear_below, call = "non-referable"; end
if call ~= "", [call, ~] = twoReaderCall(model, call, z2); end
if call == "referable"
    decision = "REFERABLE DR";
    decCol = [0.8 0.1 0.1];
    action = "URGENT REFERRAL: Consult Retina Specialist within 48-72 hours";
elseif call == "non-referable"
    decision = "NON-REFERABLE";
    decCol = [0.1 0.5 0.2];
    action = "ROUTINE SCREENING: Repeat annual non-mydriatic fundus exam";
else
    decision = "REFER-TO-HUMAN (BORDERLINE)";
    decCol = [0.85 0.55 0];
    action = "OPHTHALMOLOGIST REVIEW: Model abstains or the two readers disagree. Requires manual tele-grading.";
end

% Lesions & Quadrants
maps = num(out.lesionProb);
thr = model.meta.lesion_thresholds;
counts = zeros(1, 4);
for c = 1:4
    m = bwareaopen(maps(:, :, c) >= thr.(cfg.lesionClasses(c)), minArea(c), 8);
    counts(c) = bwconncomp(m, 8).NumObjects;
end

% Vessels & Optic Disc
vesProb = num(out.vesProb);
vesThr = 0.5;
if isfield(model.meta.lesion_thresholds, "VES"), vesThr = model.meta.lesion_thresholds.VES; end
vesMask = (vesProb >= vesThr) & fov;
vesDensity = round(100 * nnz(vesMask) / max(nnz(fov), 1), 2);

D = cfg.canvas;
[struc, foveaFound, fx, fy, macR] = estimateFoveaFromOD(out.odProb, D, 110, 340, canvas);
macCounts = zeros(1, 4);
if foveaFound
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

% Sub-pixel MA & Hemorrhages
maMask = bwareaopen(maps(:, :, 1) >= thr.MA, minArea(1), 8);
maSubTable = subpixelMA(maps(:, :, 1), maMask, [fx, fy]);

heMask = bwareaopen(maps(:, :, 2) >= thr.HE, minArea(2), 8);
heCC = bwconncomp(heMask, 8);
heStats = regionprops(heCC, "Area", "BoundingBox");
heDot = 0; heFlame = 0; hePre = 0;
for hi = 1:heCC.NumObjects
    a_i = heStats(hi).Area;
    bb_i = heStats(hi).BoundingBox;
    asp = max(bb_i(3), bb_i(4)) / max(min(bb_i(3), bb_i(4)), 1);
    if a_i > 1500, hePre = hePre + 1;
    elseif asp > 2.0, heFlame = heFlame + 1;
    else, heDot = heDot + 1; end
end

% Neovascularization & Grad-CAM
nv = neovascularizationDetect(vesProb, struc, grade, counts(2), hePre, vesDensity);
[~, blendCam] = certusGradCam(model, X, out, cfg, fov, canvas);

% ------------------------------------------------------------------ Figure Layout
f = figure(Name="Certus Clinical Report", Color="w", Position=[50 50 1400 900], Visible="off");

% Header Banner
annotation(f, "textbox", [0.03 0.92 0.94 0.06], "String", ...
           sprintf("CERTUS CLINICAL SCREENING REPORT  |  SIH 26038 (MathWorks)\nDistrict Tele-retina Program  ·  AI-Assisted Rural Triage"), ...
           "FontSize", 13, "FontWeight", "bold", "EdgeColor", "none", ...
           "Color", [0.05 0.15 0.35], "BackgroundColor", [0.93 0.95 0.98], "HorizontalAlignment", "center");

tiledlayout(f, 2, 4, TileSpacing="compact", Padding="compact");

% Plate 1: Fundus Canvas
nexttile; imshow(canvas);
title("A. Fundus Canvas (1536 px)", FontWeight="bold", Color=[0.1 0.1 0.1]);

% Plate 2: Lesion Segmentation
nexttile;
ov = im2double(canvas);
colors = [1 0.25 0.25; 0.85 0 0.6; 1 0.85 0.1; 0.2 0.9 1];
for c = 1:4
    m = (maps(:, :, c) >= thr.(cfg.lesionClasses(c))) & fov;
    for ch = 1:3
        layer = ov(:, :, ch);
        layer(m) = 0.35 * layer(m) + 0.65 * colors(c, ch);
        ov(:, :, ch) = layer;
    end
end
imshow(ov);
title("B. Lesion Masks (Calibrated)", FontWeight="bold", Color=[0.1 0.1 0.1]);

% Plate 3: Grad-CAM Heatmap
nexttile; imshow(blendCam);
title("C. Grad-CAM Attention Map", FontWeight="bold", Color=[0.1 0.1 0.1]);

% Plate 4: Vessel Tree & Sub-pixel MAs
nexttile;
vImg = im2double(canvas);
for ch = 1:3
    vCh = vImg(:, :, ch);
    vCh(vesMask) = 0.4 * vCh(vesMask) + 0.6 * 0.9;
    vImg(:, :, ch) = vCh;
end
imshow(vImg); hold on;
if height(maSubTable) > 0
    plot(maSubTable.x_sub, maSubTable.y_sub, 'y+', 'MarkerSize', 4, 'LineWidth', 1.1);
end
hold off;
title(sprintf("D. Vessels & Sub-pixel MAs (N=%d)", height(maSubTable)), FontWeight="bold", Color=[0.1 0.1 0.1]);

% Plate 5: Ordinal Severity Probabilities
nexttile;
b = bar(0:3, calP, FaceColor="flat");
b.CData = repmat([0.6 0.7 0.8], 4, 1);
b.CData(2, :) = [0.8 0.2 0.2];
ylim([0 1]); grid on;
xticklabels(["P(>0)" "P(>1)" "P(>2)" "P(>3)"]);
title("E. Ordinal Disease Probabilities", FontWeight="bold", Color=[0.1 0.1 0.1]);
yline(0.5, "k--");
yline(band.clear_below, "g:", sprintf("auto-clear %.3f", band.clear_below));
yline(band.refer_above, "r:", sprintf("auto-refer %.3f", band.refer_above));

% Plate 6: Lesion Evidence Breakdown
nexttile;
ev = num(out.evidence);
barh(ev(1:8), "FaceColor", [0.3 0.5 0.7]); grid on;
yticks(1:8);
yticklabels(["MA Area" "HE Area" "EX Area" "SE Area" "MA Peak" "HE Peak" "EX Peak" "SE Peak"]);
set(gca, YDir="reverse");
title("F. Stop-Gradient Evidence", FontWeight="bold", Color=[0.1 0.1 0.1]);

% Plate 7: Clinical Lesion Table
nexttile; axis off;
text(0, 0.95, "G. Clinical Lesion Inventory", FontWeight="bold", FontSize=12, Color=[0.05 0.05 0.05], Interpreter="none");
text(0, 0.82, sprintf("• Microaneurysms: %d total (macula: %d)", counts(1), macCounts(1)), FontSize=10, Color=[0.1 0.1 0.1], Interpreter="none");
text(0, 0.70, sprintf("• Hemorrhages: %d total (dot: %d, flame: %d)", counts(2), heDot, heFlame), FontSize=10, Color=[0.1 0.1 0.1], Interpreter="none");
text(0, 0.58, sprintf("• Preretinal Hemorrhage: %d", hePre), FontSize=10, Color=[0.1 0.1 0.1], Interpreter="none");
text(0, 0.46, sprintf("• Hard Exudates: %d (macula: %d)", counts(3), macCounts(3)), FontSize=10, Color=[0.1 0.1 0.1], Interpreter="none");
text(0, 0.34, sprintf("• Cotton Wool Spots: %d", counts(4)), FontSize=10, Color=[0.1 0.1 0.1], Interpreter="none");
text(0, 0.22, sprintf("• Vessel Density: %.1f%%", vesDensity), FontSize=10, Color=[0.1 0.1 0.1], Interpreter="none");
text(0, 0.08, sprintf("• NV Assessment: %s", nv.criteria), FontSize=9.5, FontWeight="bold", Color=[0.8 0.1 0.1], Interpreter="none");

% Plate 8: Diagnosis & Action Box
nexttile; axis off;
GRADE_NAMES = ["No Retinopathy", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
text(0, 0.95, sprintf("ICDR Level %d: %s", grade, GRADE_NAMES(grade+1)), FontSize=13, FontWeight="bold", Color=[0.05 0.05 0.05], Interpreter="none");
text(0, 0.80, sprintf("Decision: %s", decision), FontSize=12, FontWeight="bold", Color=decCol, Interpreter="none");
text(0, 0.68, sprintf("p(referable) = %.4f | Reliability = %.2f", pRef, qBest), FontSize=10, Color=[0.15 0.15 0.15], Interpreter="none");
text(0, 0.52, "RECOMMENDED ACTION:", FontSize=10, FontWeight="bold", Color=[0.05 0.05 0.05], Interpreter="none");
text(0, 0.38, action, FontSize=9.5, Color=[0.1 0.1 0.1], Interpreter="none");
text(0, 0.18, "Ophthalmologist Verification:", FontSize=9, FontWeight="bold", Color=[0.3 0.3 0.3], Interpreter="none");
text(0, 0.04, "Dr. __________________  Sign: ________  Date: ______", FontSize=8.5, Color=[0.3 0.3 0.3], Interpreter="none");

% ------------------------------------------------------------------ Save
outDir = opts.OutDir;
if outDir == "", outDir = fullfile(cfg.paths.root, "reports"); end
if ~isfolder(outDir), mkdir(outDir); end

[~, baseName] = fileparts(imagePath);
outPath = fullfile(outDir, sprintf("Certus_Report_%s.%s", baseName, opts.Format));
exportgraphics(f, outPath, Resolution=200);
close(f);
fprintf("Saved report to %s\n", outPath);
end

function x = num(v)
if isa(v, "dlarray"), v = extractdata(v); end
x = double(gather(v));
end

function a = minArea(c)
floors = [2 4 4 4];
a = floors(c);
end

function [struc, foveaFound, fx, fy, macR] = estimateFoveaFromOD(odProb, D, DISC_MIN, DISC_MAX, canvas)
foveaFound = false; fx = NaN; fy = NaN; macR = NaN; struc = struct();
if isa(odProb, "gpuArray"), odProb = gather(odProb); end
odMask = squeeze(double(odProb)) > 0.5;
if sum(odMask(:)) < 50, struc.disc_found = false; return; end
cc = bwconncomp(odMask, 8);
if cc.NumObjects == 0, struc.disc_found = false; return; end
compSizes = cellfun(@numel, cc.PixelIdxList);
[~, bigIdx] = max(compSizes);
bigMask = false(D, D); bigMask(cc.PixelIdxList{bigIdx}) = true;
[ys, xs] = find(bigMask);
ox = mean(xs); oy = mean(ys);
dd = 2 * sqrt(sum(bigMask(:)) / pi);
struc.disc_found = true; struc.disc_x = ox; struc.disc_y = oy; struc.disc_diameter_px = dd;
if dd < DISC_MIN || dd > DISC_MAX, struc.fovea_found = false; return; end
[fx, fy] = foveaFromDisc(ox, oy, dd, D, canvas);   % fitted on IDRiD train, same rule as the API
if fx < 0 || fx >= D || fy < 0 || fy >= D, struc.fovea_found = false; return; end
foveaFound = true; macR = dd;
struc.fovea_found = true; struc.fovea_x = fx; struc.fovea_y = fy; struc.macula_radius_px = macR;
end
