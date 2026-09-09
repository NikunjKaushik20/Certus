function B = loadStep(spec, cfg)
%LOADSTEP Decode everything one optimiser step needs (runs on a background thread).
%   Returns uint8 arrays only; conversion, augmentation and tiling happen on the GPU.
D = cfg.canvas; T = cfg.tile; Gs = cfg.globalSize; C = numel(cfg.segClasses);

nG = numel(spec.gradePath);
B.grade = zeros(D, D, 3, nG, "uint8");
for i = 1:nG
    I = imread(spec.gradePath(i));
    if size(I, 1) ~= D, I = imresize(I, [D D]); end
    B.grade(:, :, :, i) = I;
end
B.gradeY = spec.gradeY;

nS = numel(spec.segPath);
B.segX = zeros(T, T, 3, nS, "uint8");
B.segY = zeros(T, T, C, nS, "uint8");
B.segValid = false(C, nS);
for i = 1:nS
    [B.segX(:, :, :, i), B.segY(:, :, :, i), B.segValid(:, i)] = ...
        readSegCrop(spec.segPath(i), spec.segMasks(i, :), cfg, spec.segSeed(i));
end

nQ = numel(spec.qualPath);
B.qualX = zeros(Gs, Gs, 3, nQ, "uint8");
for i = 1:nQ
    B.qualX(:, :, :, i) = imresize(imread(spec.qualPath(i)), [Gs Gs]);
end
B.qualY = spec.qualY;
end
