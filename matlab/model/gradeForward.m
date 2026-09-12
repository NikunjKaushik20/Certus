function out = gradeForward(model, Xc, cfg, training)
%GRADEFORWARD Whole-eye forward pass. Xc: D x D x 3 x n canvases in [0,1] (gpuArray).
%   The encoder always runs in inference mode (frozen ImageNet BN statistics); gradients
%   still flow to all encoder weights. Lesion evidence is stop-gradient by design: the
%   grade loss can use the lesion maps but cannot reshape them.
n = size(Xc, 4); g2 = cfg.grid ^ 2; nt = g2 * n; E = model.meta.embedDim;
tiles = makeTiles(Xc, cfg);
glob  = imresize(Xc, [cfg.globalSize cfg.globalSize]);
fovT  = single(any(tiles > 0.02, 3));

X = dlarray(cat(4, tiles, glob) * model.meta.inputScale, "SSCB");
[seg, emb] = predict(model.enc, X, Outputs=[model.meta.segOutput model.meta.embOutput]);
emb  = reshape(stripdims(emb), E, []);
embT = emb(:, 1:nt);
embG = emb(:, nt+1:end);

s = headFwd(model.attn, dlarray(embT, "CB"), training);
S = reshape(stripdims(s), g2, n);
a = exp(S - max(S, [], 1));
a = a ./ sum(a, 1);                                        % attention over the eye's tiles
pooled = reshape(sum(reshape(embT, E, g2, n) .* reshape(a, 1, g2, n), 2), E, n);

P  = extractdata(sigmoid(seg(:, :, 1:4, 1:nt))) .* fovT;  % stop-gradient
ev = evidenceFeatures(P, cfg, n);

out.gradeLogits = headFwd(model.grade, dlarray([pooled; embG; ev], "CB"), training);
out.qualLogits  = headFwd(model.qual, dlarray(embG, "CB"), training);
out.attn     = a;
out.evidence = ev;
if ~training
    out.embT = embT;                                      % kept for certusGradCam
    out.embG = embG;
    out.lesionProb = stitchTiles(P, cfg);                 % D x D x 4 x n, for reports
    if size(seg, 3) >= 6
        segAll = extractdata(sigmoid(seg(:, :, :, 1:nt)));
        out.odProb  = stitchTiles(segAll(:, :, 5, :), cfg);  % Optic disc
        out.vesProb = stitchTiles(segAll(:, :, 6, :), cfg);  % Retinal vessels
    end
end
end

function y = headFwd(net, x, training)
if training
    y = forward(net, x);
else
    y = predict(net, x);
end
end
