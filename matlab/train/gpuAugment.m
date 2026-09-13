function [X, Y] = gpuAugment(X, Y, cfg, mode)
%GPUAUGMENT Geometric + photometric + acquisition-degradation augmentation on the GPU.
%   mode "full"    : grading canvases (flip, small rotation, degradations)
%        "patch"   : segmentation patches (+ 90° rotations), masks transformed too
%        "quality" : quality views; NO blur/vignette/noise (they would change the label)
a = cfg.aug; N = size(X, 4); hasY = ~isempty(Y);
F = single(any(X > 0.02, 3));
for i = 1:N
    x = X(:, :, :, i); f = F(:, :, :, i);
    if hasY, y = Y(:, :, :, i); end
    if rand < 0.5
        x = flip(x, 2); f = flip(f, 2);
        if hasY, y = flip(y, 2); end
    end
    if mode == "patch"
        k = randi(4) - 1;
        x = rot90(x, k); f = rot90(f, k);
        if hasY, y = rot90(y, k); end
    end
    ang = (2 * rand - 1) * a.rotateDeg;
    x = imrotate(x, ang, "bilinear", "crop");
    f = imrotate(f, ang, "nearest", "crop");
    if hasY, y = imrotate(y, ang, "nearest", "crop"); end
    if mode ~= "quality"
        if rand < a.blurProb
            x = imgaussfilt(x, a.blurSigma(1) + rand * diff(a.blurSigma));
        end
        if rand < a.vignetteProb
            x = x .* vignette(size(x, 1), a.vignetteMax * rand);
        end
        if rand < a.noiseProb
            x = x + a.noiseSigma * rand * randn(size(x), "like", x);
        end
    end
    X(:, :, :, i) = x; F(:, :, :, i) = f;
    if hasY, Y(:, :, :, i) = y; end
end

% photometric jitter, vectorised over the batch (halved for quality views)
s = 1 - 0.5 * (mode == "quality");
u = @(sz) 2 * rand(sz, "like", X) - 1;
b   = 1 + s * a.brightness * u([1 1 1 N]);
c   = 1 + s * a.contrast   * u([1 1 1 N]);
gam = exp(s * a.gamma * u([1 1 1 N]));
col = 1 + s * a.color * u([1 1 3 N]);
m = sum(X .* F, [1 2]) ./ max(sum(F, [1 2]), 1);
X = ((X - m) .* c + m) .* b .* col;
X = min(max(X, 0) .^ gam, 1) .* F;
end

function v = vignette(n, strength)
lin = gpuArray(single(linspace(-1, 1, n)));
[xx, yy] = meshgrid(lin);
c = (2 * rand(1, 2) - 1) * 0.5;
v = 1 - strength * min(((xx - c(1)) .^ 2 + (yy - c(2)) .^ 2) / 2, 1);
end
