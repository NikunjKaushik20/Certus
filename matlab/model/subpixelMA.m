function T = subpixelMA(maProb, maMask, foveaXY)
%SUBPIXELMA Localize retinal microaneurysms with sub-pixel precision.
%   Computes continuous 2D intensity-weighted spatial moments for each
%   microaneurysm (MA) candidate patch, achieving sub-pixel localization
%   accuracy beyond the discrete pixel grid.
%
%   maProb:   DxD double/single in [0, 1] (MA probability map)
%   maMask:   DxD logical binary mask of segmented microaneurysms
%   foveaXY:  [fx, fy] coordinates of estimated fovea (optional, NaN if absent)
%
%   T:        table with x_sub, y_sub, peak_prob, dist_fovea_px

arguments
    maProb (:, :) {mustBeNumeric}
    maMask (:, :) logical
    foveaXY (1, 2) double = [NaN, NaN]
end

if isa(maProb, "gpuArray"), maProb = gather(maProb); end
maProb = double(maProb);

cc = bwconncomp(maMask, 8);
n = cc.NumObjects;

if n == 0
    T = table(double.empty(0,1), double.empty(0,1), double.empty(0,1), double.empty(0,1), ...
              VariableNames=["x_sub", "y_sub", "peak_prob", "dist_fovea_px"]);
    return;
end

s = regionprops(cc, "Centroid");
[D1, D2] = size(maProb);

x_sub = zeros(n, 1);
y_sub = zeros(n, 1);
peak_prob = zeros(n, 1);
dist_fovea = nan(n, 1);

hasFovea = ~any(isnan(foveaXY));

for i = 1:n
    cx = s(i).Centroid(1);
    cy = s(i).Centroid(2);
    
    % Extract 9x9 local window centered at integer centroid
    ix = round(cx);
    iy = round(cy);
    
    x0 = max(1, ix - 4);
    x1 = min(D2, ix + 4);
    y0 = max(1, iy - 4);
    y1 = min(D1, iy + 4);
    
    patch = maProb(y0:y1, x0:x1);
    totalW = sum(patch(:));
    
    if totalW > 1e-6
        [gx, gy] = meshgrid(x0:x1, y0:y1);
        sx = sum(sum(gx .* patch)) / totalW;
        sy = sum(sum(gy .* patch)) / totalW;
    else
        sx = cx;
        sy = cy;
    end
    
    x_sub(i) = round(sx, 2);
    y_sub(i) = round(sy, 2);
    peak_prob(i) = round(max(patch(:)), 4);
    
    if hasFovea
        dist_fovea(i) = round(hypot(sx - foveaXY(1), sy - foveaXY(2)), 1);
    end
end

T = table(x_sub, y_sub, peak_prob, dist_fovea, ...
          VariableNames=["x_sub", "y_sub", "peak_prob", "dist_fovea_px"]);
end
