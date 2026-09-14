function lr_finder()
%LR_FINDER Learning-rate range test (Smith 2017) on the full multi-task objective.
%   For each encoder/head LR ratio in cfg.lrfinder.ratios, trains from the same
%   pretrained start with the same data order while the LR grows exponentially from
%   lrMin to lrMax; records the bias-corrected smoothed loss and stops on divergence.
%   Suggested LR = min(steepest-descent LR, LR at minimum loss / 10).
%   Writes runs/lr_finder.json (read by certus_config) and runs/lr_finder.png.
cfg = startup_certus();
assert(canUseGPU, "Certus: training needs a CUDA GPU (Parallel Computing Toolbox + NVIDIA driver).");
assert(isfile(cfg.paths.probeFile), "Run smoke_test first: batch sizes must be measured.");
M = loadManifests(cfg); S = makeSampler(M, cfg);
F = cfg.lrfinder;
lrs = F.lrMin * (F.lrMax / F.lrMin) .^ ((0:F.steps - 1) / (F.steps - 1));
base = buildCertusModel(cfg);

for r = 1:numel(F.ratios)
    rng(cfg.seed, "twister"); gpurng(cfg.seed, "Philox");   % identical data + augmentation per ratio
    model = modelToGpu(base);
    opt = optim.init(model, cfg, F.ratios(r));
    pf = Prefetcher(@() makeStepSpec(M, S, cfg), cfg);
    raw = nan(1, F.steps); sm = raw; avg = 0; best = inf; t0 = tic;
    for i = 1:F.steps
        [model, opt, info] = trainStep(model, opt, pf.next(), cfg, lrs(i), false);  % task weights fixed
        raw(i) = info.total;
        avg = F.smooth * avg + (1 - F.smooth) * raw(i);
        sm(i) = avg / (1 - F.smooth ^ i);
        best = min(best, sm(i));
        if mod(i, 10) == 0
            fprintf("ratio %.2f  step %3d/%d  lr %.2e  loss %.4f  smooth %.4f  (%.1fs/step)\n", ...
                F.ratios(r), i, F.steps, lrs(i), raw(i), sm(i), toc(t0) / i);
        end
        if ~isfinite(raw(i)) || (i > 10 && sm(i) > F.divergeFactor * best)
            fprintf("ratio %.2f diverged at lr %.2e\n", F.ratios(r), lrs(i));
            break
        end
    end
    delete(pf);
    res(r) = analyse(lrs, raw, sm, F.ratios(r)); %#ok<AGROW>
    fprintf("ratio %.2f: min smoothed loss %.4f at lr %.2e | steepest %.2e | suggested %.2e\n", ...
        res(r).ratio, res(r).minLoss, res(r).lrAtMin, res(r).lrSteep, res(r).lrSuggest);
end

% lowest minimum loss wins; within 1% prefer the smaller (safer) encoder ratio
mins = [res.minLoss];
ok = find(mins <= 1.01 * min(mins));
[~, k] = min([res(ok).ratio]);
c = res(ok(k));
out = struct("lr", c.lrSuggest, "encLRRatio", c.ratio, "chosenBecause", ...
    "lowest min smoothed loss (ties within 1% -> smaller encoder ratio)", ...
    "rule", "lr = min(steepest-descent lr, lr_at_min/10)", "steps", F.steps, "runs", res);
fid = fopen(cfg.paths.lrFile, "w"); fprintf(fid, "%s", jsonencode(out, PrettyPrint=true)); fclose(fid);

fig = figure(Visible="off");
for r = 1:numel(res)
    semilogx(res(r).lrs, res(r).smooth, LineWidth=1.5, DisplayName=sprintf("enc ratio %.2f", res(r).ratio)); hold on
end
xline(c.lrSuggest, "--", sprintf("chosen %.1e (ratio %.2f)", c.lrSuggest, c.ratio), HandleVisibility="off");
xlabel("peak learning rate (heads/decoder)"); ylabel("smoothed multi-task loss"); grid on; legend(Location="best");
title("Certus LR range test");
exportgraphics(fig, fullfile(cfg.paths.runs, "lr_finder.png"), Resolution=150);
fprintf("\nchosen lr %.2e, encoder ratio %.2f -> %s\n", c.lrSuggest, c.ratio, cfg.paths.lrFile);
end

function a = analyse(lrs, raw, sm, ratio)
n = find(isfinite(sm), 1, "last");
lr = lrs(1:n); s = sm(1:n);
[a.minLoss, iMin] = min(s);
i0 = max(2, round(0.1 * n));                   % skip the smoother's warm-up
d = gradient(s, log10(lr));
if iMin > i0
    [~, k] = min(d(i0:iMin)); iSteep = i0 + k - 1;
else
    iSteep = iMin;
end
a.ratio = ratio;
a.lrAtMin = lr(iMin);
a.lrSteep = lr(iSteep);
a.lrSuggest = min(a.lrSteep, a.lrAtMin / 10);
a.lrs = lr; a.smooth = s; a.raw = raw(1:n);
end
