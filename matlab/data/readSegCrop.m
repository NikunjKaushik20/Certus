function [X, Y, valid] = readSegCrop(path, maskPaths, cfg, seed)
%READSEGCROP Lesion-aware 512px crop of a segmentation image and its masks.
%   valid(c) is false when class c is not annotated for this image (loss is masked).
s = RandStream("philox4x32_10", Seed=seed);
I = imread(path);
D = size(I, 1); T = cfg.tile; C = numel(maskPaths);
valid = maskPaths(:) ~= "";
Yf = zeros(D, D, C, "uint8");
for c = find(valid)'
    Yf(:, :, c) = uint8(imread(maskPaths(c)) > 0);
end

centre = [];
if rand(s) < cfg.sampling.lesionCentred
    L = numel(cfg.lesionClasses);
    present = find(valid(1:L) & squeeze(any(Yf(:, :, 1:L), [1 2])));
    if ~isempty(present)
        w = cfg.sampling.lesionPriority(present);
        c = present(find(rand(s) <= cumsum(w) / sum(w), 1));
        idx = find(Yf(:, :, c));
        [r, q] = ind2sub([D D], idx(randi(s, numel(idx))));
        centre = [r q] + randi(s, [-T/4 T/4], 1, 2);   % jitter: lesion not always centred
    end
end
if isempty(centre)
    idx = find(max(I, [], 3) > 10);                   % any FOV pixel
    [r, q] = ind2sub([D D], idx(randi(s, numel(idx))));
    centre = [r q];
end
r0 = min(max(centre(1) - T/2, 1), D - T + 1);
c0 = min(max(centre(2) - T/2, 1), D - T + 1);
X = I(r0:r0+T-1, c0:c0+T-1, :);
Y = Yf(r0:r0+T-1, c0:c0+T-1, :);
end
