function train_certus()
%TRAIN_CERTUS Full multi-task training. Requires runs/lr_finder.json and runs/batch_probe.json.
%   Validates the EMA weights every cfg.valEvery steps; keeps the best checkpoint by
%   referable-DR AUC on the validation split (Messidor-2 is never touched here).
cfg = startup_certus();
assert(canUseGPU, "Certus: training needs a CUDA GPU (Parallel Computing Toolbox + NVIDIA driver).");
assert(~isnan(cfg.opt.lr), "No measured learning rate: run lr_finder first.");
assert(isfile(cfg.paths.probeFile), "No measured batch sizes: run smoke_test first.");

runDir = fullfile(cfg.paths.runs, "train_" + string(datetime("now", Format="yyyyMMdd_HHmmss")));
mkdir(runDir);
save(fullfile(runDir, "config.mat"), "cfg");

M = loadManifests(cfg); S = makeSampler(M, cfg);
model = modelToGpu(buildCertusModel(cfg));
opt = optim.init(model, cfg, cfg.opt.encLRRatio);
ema = model;

stepsPerEpoch = ceil(height(M.gradeTrain) / (cfg.batch.grade * cfg.accum.grade));
total = cfg.opt.epochs * stepsPerEpoch;
fprintf("training %d steps (%d epochs x %d), lr %.2e, encoder ratio %.2f\n", ...
    total, cfg.opt.epochs, stepsPerEpoch, cfg.opt.lr, cfg.opt.encLRRatio);

logFile = fopen(fullfile(runDir, "log.csv"), "w");
fprintf(logFile, "step,epoch,lr,total,grade,seg,quality,gradNorm,lv_grade,lv_seg,lv_quality,sec\n");
valFile = fopen(fullfile(runDir, "val.csv"), "w");
fprintf(valFile, "step,auc,sens_at_85spec,qwk,ece,dice_MA,dice_HE,dice_EX,dice_SE,quality_acc\n");
cleaner = onCleanup(@() fclose("all"));

pf = Prefetcher(@() makeStepSpec(M, S, cfg), cfg);
bestAuc = -inf; t0 = tic;
for it = 1:total
    lr = optim.schedule(it, total, cfg);
    [model, opt, info] = trainStep(model, opt, pf.next(), cfg, lr, true);
    ema = optim.emaUpdate(ema, model, cfg.opt.emaDecay);
    fprintf(logFile, "%d,%.3f,%.3e,%.5f,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f,%.3f,%.1f\n", it, it / stepsPerEpoch, ...
        lr, info.total, info.grade, info.seg, info.quality, info.gradNorm, info.logvar, toc(t0));
    if mod(it, 50) == 0
        fprintf("step %d/%d  lr %.2e  loss %.4f (g %.3f s %.3f q %.3f)  %.2fs/step\n", it, total, lr, ...
            info.total, info.grade, info.seg, info.quality, toc(t0) / it);
    end
    if ~isfinite(info.total), error("Non-finite loss at step %d", it); end

    if mod(it, cfg.valEvery) == 0 || it == total
        n = cfg.valSubset; if it == total, n = inf; end
        R = evaluateModel(ema, M, cfg, n, "val");
        g = R.grade;
        fprintf(valFile, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", it, g.aucReferable, ...
            g.sensAt085Spec, g.qwk, g.eceReferable, R.seg.dice{1, :}, R.quality.accuracy);
        fprintf(">> val step %d: AUC %.4f | sens@85%%spec %.3f | QWK %.3f | ECE %.3f\n", it, ...
            g.aucReferable, g.sensAt085Spec, g.qwk, g.eceReferable);
        ck = modelGather(ema);
        save(fullfile(runDir, "last.mat"), "ck", "it", "R", "-v7.3");
        if g.aucReferable > bestAuc
            bestAuc = g.aucReferable;
            save(fullfile(runDir, "best.mat"), "ck", "it", "R", "-v7.3");
            fprintf(">> new best (AUC %.4f)\n", bestAuc);
        end
    end
end
delete(pf);
fprintf("done. best val AUC %.4f -> %s\n", bestAuc, runDir);
end
