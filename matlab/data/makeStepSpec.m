function spec = makeStepSpec(M, S, cfg)
%MAKESTEPSPEC Draw the file lists for one optimiser step (runs on the client).
nG = cfg.batch.grade * cfg.accum.grade;
gi = randsample(height(M.gradeTrain), nG, true, S.gradeW);
spec.gradePath = M.gradeTrain.proc_path(gi);
spec.gradeY    = M.gradeTrain.dr_grade(gi)';

si = randi(height(M.segTrain), cfg.batch.seg, 1);
spec.segPath  = M.segTrain.path(si);
spec.segMasks = M.segTrain.masks(si, :);
spec.segSeed  = randi(2^31 - 1, cfg.batch.seg, 1);

nP = round(cfg.batch.quality * cfg.sampling.partialQuality);
nF = cfg.batch.quality - nP;
fi = S.qualFull(randsample(numel(S.qualFull), nF, true, S.qualFullW));
pidx = S.qualPartial(randi(numel(S.qualPartial), nP, 1));
spec.qualPath = M.qualTrain.proc_path([fi; pidx]);
spec.qualY    = [M.qualTrain.quality_label(fi); -ones(nP, 1)]';   % -1 = "not reject"
end
