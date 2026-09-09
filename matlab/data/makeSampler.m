function S = makeSampler(M, cfg)
%MAKESAMPLER Sampling weights: softened class + source-dataset balancing.
S.gradeW = balance(M.gradeTrain.dr_grade, cfg.sampling.gradePower) .* ...
           balance(M.gradeTrain.dataset, cfg.sampling.datasetPower);
S.gradeW = S.gradeW / sum(S.gradeW);

q = M.qualTrain.quality_label;
S.qualFull    = find(~isnan(q));
S.qualFullW   = balance(q(S.qualFull), cfg.sampling.gradePower);
S.qualPartial = find(isnan(q));            % gradable clinical images: label set {good, usable}
end

function w = balance(x, power)
[~, ~, k] = unique(x);
counts = accumarray(k, 1);
w = counts(k) .^ (-power);
end
