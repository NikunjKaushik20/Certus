function smoke_test()
%SMOKE_TEST End-to-end check of the pipeline on the GPU + measured batch-size probe.
%   Writes runs/smoke_report.txt and runs/batch_probe.json. Does not train.
cfg = startup_certus();
assert(canUseGPU, "Certus: training needs a CUDA GPU (Parallel Computing Toolbox + NVIDIA driver).");
fid = fopen(fullfile(cfg.paths.runs, "smoke_report.txt"), "w");
cleaner = onCleanup(@() fclose(fid));
say = @(varargin) logline(fid, sprintf(varargin{:}));

%% 1. environment
g = gpuDevice;
say("MATLAB %s | GPU %s (cc %s) | CUDA toolkit %s | driver %s | %.1f GB", version, g.Name, ...
    g.ComputeCapability, string(g.ToolkitVersion), string(g.DriverVersion), g.TotalMemory / 2^30);
v = string({ver().Name});
need = ["Deep Learning Toolbox" "Image Processing Toolbox" "Parallel Computing Toolbox" ...
        "Statistics and Machine Learning Toolbox" "Computer Vision Toolbox"];
missing = need(~ismember(need, v));
assert(isempty(missing), "Missing toolboxes: %s", strjoin(missing, ", "));
say("toolboxes OK");

%% 2. data
M = loadManifests(cfg); S = makeSampler(M, cfg);
say("grade train/val/test/ext = %d/%d/%d/%d | seg train/val = %d/%d | quality train/val = %d/%d", ...
    height(M.gradeTrain), height(M.gradeVal), height(M.gradeTest), height(M.gradeExt), ...
    height(M.segTrain), height(M.segVal), height(M.qualTrain), height(M.qualVal));
tic; B = fetchOutputs(parfeval(backgroundPool, @loadStep, 1, makeStepSpec(M, S, cfg), cfg));
say("one step decoded on a background thread in %.2fs (grade %s, seg %s, quality %s)", toc, ...
    mat2str(size(B.grade)), mat2str(size(B.segX)), mat2str(size(B.qualX)));
assert(any(B.segY(:)), "seg batch has no positive pixels");

%% 3. tiling round-trip
X = gpuArray.rand(cfg.canvas, cfg.canvas, 3, 2, "single");
assert(isequal(stitchTiles(makeTiles(X, cfg), cfg), X), "tile/stitch mismatch");
say("tiling round-trip OK");

%% 4. model
model = modelToGpu(buildCertusModel(cfg));
m = model.meta;
say("encoder taps: %s", strjoin(string(struct2cell(m.taps))', " | "));
say("embedDim %d | inputScale %g | params %.2fM | pretrained learnables %d/%d", m.embedDim, ...
    m.inputScale, m.numParams / 1e6, nnz(m.isPretrained), numel(m.isPretrained));

%% 5. gradients reach every part (also proves predict() is differentiable -> frozen BN works)
toGpu = @(u) single(gpuArray(u)) / 255;
[~, gE, gA, gG, ~, Lg] = dlfeval(@losses.grade, model, toGpu(B.grade(:, :, :, 1)), B.gradeY(1), cfg);
V = reshape(single(gpuArray(B.segValid)), 1, 1, size(B.segValid, 1), []);
[~, gEs, ~, Ls] = dlfeval(@losses.seg, model, toGpu(B.segX), single(gpuArray(B.segY)), V, cfg);
[~, ~, gQ, ~, Lq] = dlfeval(@losses.quality, model, toGpu(B.qualX), B.qualY, cfg);
say("initial losses: grade %.4f | seg %.4f | quality %.4f", Lg, Ls, Lq);
assert(all(isfinite([Lg Ls Lq])), "non-finite loss");
L = model.enc.Learnables;
checks = {"pretrained conv (grade loss)", gE, m.isPretrained & L.Parameter == "Weights";
          "pretrained BN offset (grade loss)", gE, m.isPretrained & L.Parameter == "Offset";
          "decoder (seg loss)", gEs, ~m.isPretrained;
          "pretrained conv (seg loss)", gEs, m.isPretrained & L.Parameter == "Weights"};
for k = 1:size(checks, 1)
    n = gradNorm(checks{k, 2}, checks{k, 3});
    say("  grad norm %-34s %.3e", checks{k, 1}, n);
    assert(n > 0 && isfinite(n), "zero/NaN gradient: %s", checks{k, 1});
end
say("  grad norm attention %.3e | grade head %.3e | quality head %.3e", ...
    gradNorm(gA), gradNorm(gG), gradNorm(gQ));

%% 6. one optimiser step changes the weights
opt = optim.init(model, cfg, 0.1);
w0 = gather(extractdata(model.grade.Learnables.Value{1}));
[model2, ~, info] = trainStep(model, opt, B, cfg, 1e-4, true);
dw = norm(gather(extractdata(model2.grade.Learnables.Value{1})) - w0, "fro");
say("trainStep OK: total loss %.4f, grad norm %.3f, |dW| %.2e", info.total, info.gradNorm, dw);
assert(dw > 0, "weights did not change");
clear model2

%% 7. batch-size probe (measured, per sub-batch type)
probe = struct();
[probe.grade, tg] = largestBatch(@(n) dlfeval(@losses.grade, model, ...
    toGpu(repmat(B.grade(:, :, :, 1), 1, 1, 1, n)), repmat(B.gradeY(1), 1, n), cfg), [1 2 3 4], say, "grade");
[probe.seg, ts] = largestBatch(@(n) dlfeval(@losses.seg, model, toGpu(repmat(B.segX(:, :, :, 1), 1, 1, 1, n)), ...
    single(gpuArray(repmat(B.segY(:, :, :, 1), 1, 1, 1, n))), repmat(V(:, :, :, 1), 1, 1, 1, n), cfg), ...
    [4 6 8 12 16 24], say, "seg");
[probe.quality, tq] = largestBatch(@(n) dlfeval(@losses.quality, model, ...
    toGpu(repmat(B.qualX(:, :, :, 1), 1, 1, 1, n)), repmat(B.qualY(1), 1, n), cfg), [8 16 24 32 48 64], say, "quality");
targetEyes = 8;   % eyes per optimiser step (effective grade batch)
probe.accumGrade = max(1, round(targetEyes / probe.grade));
probe.secPerStep = probe.accumGrade * tg + ts + tq;
steps = ceil(height(M.gradeTrain) / (probe.grade * probe.accumGrade));
probe.stepsPerEpoch = steps;
probe.hoursPerEpoch = steps * probe.secPerStep / 3600;
say("chosen: grade %d x accum %d | seg %d | quality %d -> %.2fs/step, %d steps/epoch, %.1f h/epoch (GPU compute only)", ...
    probe.grade, probe.accumGrade, probe.seg, probe.quality, probe.secPerStep, steps, probe.hoursPerEpoch);
fid2 = fopen(cfg.paths.probeFile, "w"); fprintf(fid2, "%s", jsonencode(probe, PrettyPrint=true)); fclose(fid2);

%% 8. loader keeps up?
cfg2 = certus_config();
tic; nL = 4;
f = cell(1, nL);
for k = 1:nL, f{k} = parfeval(backgroundPool, @loadStep, 1, makeStepSpec(M, makeSampler(M, cfg2), cfg2), cfg2); end
cellfun(@fetchOutputs, f);
say("loader: %d steps in %.1fs on %d threads (%.2fs/step effective) vs GPU %.2fs/step", nL, toc, ...
    backgroundPool.NumWorkers, toc / nL, probe.secPerStep);

%% 9. evaluation code paths
R = evaluateModel(model, M, cfg, 6, "val");
say("eval OK: val AUC(referable) %.3f on %d imgs (untrained - meaningless), seg dice MA %.3f, quality acc %.3f", ...
    R.grade.aucReferable, R.grade.n, R.seg.dice.MA, R.quality.accuracy);
say("SMOKE TEST PASSED");
end

function logline(fid, s)
fprintf("%s\n", s);
fprintf(fid, "%s\n", s);
end

function n = gradNorm(g, rows)
if nargin < 2, rows = true(height(g), 1); end
v = g.Value(rows);
n = sqrt(sum(cellfun(@(x) gather(sum(extractdata(x) .^ 2, "all")), v)));
end

function [best, tBest] = largestBatch(fn, sizes, say, name)
best = NaN; tBest = NaN;
for n = sizes
    try
        fn(n); wait(gpuDevice);                      % warm-up (allocations, kernels)
        t = zeros(1, 3);
        for r = 1:3, tic; fn(n); wait(gpuDevice); t(r) = toc; end
        g = gpuDevice;
        say("  probe %-7s batch %3d OK  %.3fs  free %.2f GB", name, n, median(t), g.AvailableMemory / 2^30);
        best = n; tBest = median(t);
    catch err
        if contains(err.message, ["out of memory" "Out of memory"], IgnoreCase=true)
            say("  probe %-7s batch %3d OUT OF MEMORY", name, n);
            break
        end
        rethrow(err)
    end
end
assert(~isnan(best), "even the smallest %s batch does not fit", name);
end
