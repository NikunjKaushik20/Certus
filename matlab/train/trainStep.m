function [model, opt, info] = trainStep(model, opt, B, cfg, lr, learnLogvar)
%TRAINSTEP One optimiser step = grade sub-batches (accumulated) + seg batch + quality batch.
%   Sub-batches run sequentially so peak GPU memory is the largest single sub-batch,
%   while the update uses the joint multi-task gradient.
toGpu = @(u) single(gpuArray(u)) / 255;
G = struct();
nG = cfg.batch.grade; A = cfg.accum.grade;
Lg = 0;
for a = 1:A
    idx = (a - 1) * nG + (1:nG);
    Xc = gpuAugment(toGpu(B.grade(:, :, :, idx)), [], cfg, "full");
    [~, gE, gA, gG, gL, l] = dlfeval(@losses.grade, model, Xc, B.gradeY(idx), cfg);
    G = optim.accumulate(G, "enc", gE, 1 / A);
    G = optim.accumulate(G, "attn", gA, 1 / A);
    G = optim.accumulate(G, "grade", gG, 1 / A);
    G = optim.accumulate(G, "logvar", gL, 1 / A);
    Lg = Lg + l / A;
end

[Xs, Ys] = gpuAugment(toGpu(B.segX), single(gpuArray(B.segY)), cfg, "patch");
V = reshape(single(gpuArray(B.segValid)), 1, 1, size(B.segValid, 1), []);
[~, gE, gL, Ls] = dlfeval(@losses.seg, model, Xs, Ys, V, cfg);
G = optim.accumulate(G, "enc", gE, 1);
G = optim.accumulate(G, "logvar", gL, 1);

Xq = gpuAugment(toGpu(B.qualX), [], cfg, "quality");
[~, gE, gQ, gL, Lq] = dlfeval(@losses.quality, model, Xq, B.qualY, cfg);
G = optim.accumulate(G, "enc", gE, 1);
G = optim.accumulate(G, "qual", gQ, 1);
G = optim.accumulate(G, "logvar", gL, 1);

[model, opt, gnorm] = optim.step(model, opt, G, lr, cfg, learnLogvar);
info = struct("grade", Lg, "seg", Ls, "quality", Lq, "total", Lg + Ls + Lq, ...
              "gradNorm", gnorm, "lr", lr, "logvar", gather(extractdata(model.logvar))');
end
