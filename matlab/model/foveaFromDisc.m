function [fx, fy] = foveaFromDisc(ox, oy, dd, D, canvas)
%FOVEAFROMDISC Fovea position from the optic disc, the same rule as the API engine.
%   Offsets fitted on the IDRiD localisation train split (torch/fit_fovea.py -> torch/fovea_rule.json),
%   measured in FOV diameters (the canvas side D), then moved to the darkest point of the smoothed
%   green channel within half a disc diameter. On IDRiD test: median 35.5 native px, mean 67
%   (IDRiD challenge winner, mean: 64.5). canvas is optional; without it the refinement is skipped.
persistent rule
if isempty(rule)
    f = fullfile(fileparts(fileparts(fileparts(mfilename("fullpath")))), "torch", "fovea_rule.json");
    if isfile(f)
        rule = jsondecode(fileread(f));
    else
        rule = struct("ruler", "dd", "temporal", 2.5, "inferior", 0.1, "refine_radius_dd", []);
    end
end
temporal = 1;
if ox >= D / 2, temporal = -1; end                  % fovea lies away from the nasal edge
unit = dd;
if rule.ruler == "D", unit = D; end
fx = ox + temporal * rule.temporal * unit;
fy = oy + rule.inferior * unit;

if nargin < 5 || isempty(canvas) || isempty(rule.refine_radius_dd) || ...
        fx < 0 || fx >= D || fy < 0 || fy >= D
    return
end
% Coordinates here are 1-based pixel indices, like the rest of the MATLAB demo.
r = rule.refine_radius_dd * dd;
g = imgaussfilt(single(canvas(:, :, 2)), max(2, r / 6));
x0 = max(1, floor(fx - r)); x1 = min(D, ceil(fx + r));
y0 = max(1, floor(fy - r)); y1 = min(D, ceil(fy + r));
[xx, yy] = meshgrid(x0:x1, y0:y1);
patch = g(y0:y1, x0:x1);
patch((xx - fx).^2 + (yy - fy).^2 > r^2) = Inf;
[~, k] = min(patch(:));
[iy, ix] = ind2sub(size(patch), k);
fx = x0 + ix - 1;
fy = y0 + iy - 1;
end
