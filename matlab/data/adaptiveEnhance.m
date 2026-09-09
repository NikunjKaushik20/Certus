function [enhanced, info] = adaptiveEnhance(canvas, qualityLabel, force)
%ADAPTIVEENHANCE Adaptive enhancement for borderline retinal fundus photographs.
%   Applies green-channel CLAHE (adapthisteq), illumination normalization,
%   and edge-preserving guided/bilateral denoising to borderline ('usable') images.
%
%   canvas:        DxDx3 uint8 RGB image (0-255)
%   qualityLabel:  "good" | "usable" | "reject"
%   force:         logical, default false
%
%   enhanced:      DxDx3 uint8 RGB image
%   info:          struct with applied flag and list of methods

arguments
    canvas (:, :, 3) uint8
    qualityLabel string = "usable"
    force (1,1) logical = false
end

if qualityLabel ~= "usable" && ~force
    enhanced = canvas;
    info = struct("applied", false, "methods", string.empty);
    return;
end

methods = string.empty;
enhanced = canvas;
[D, ~, ~] = size(canvas);

% 1. Illumination Normalization & CLAHE on Green Channel
% Retinal vessels and microvascular lesions have highest contrast in green.
R = canvas(:, :, 1);
G = canvas(:, :, 2);
B = canvas(:, :, 3);

G_single = single(G);
sigma = D / 25;
bg = imgaussfilt(G_single, sigma);
validMask = bg > 10;
if any(validMask(:))
    meanBg = mean(bg(validMask));
else
    meanBg = 128;
end

G_norm = uint8(min(255, max(0, (G_single ./ max(bg, 1e-3)) * meanBg)));
methods(end+1) = "illumination_normalization";

% Contrast-Limited Adaptive Histogram Equalization
G_clahe = adapthisteq(G_norm, "ClipLimit", 0.02, "NumTiles", [8 8], "Distribution", "rayleigh");
methods(end+1) = "green_channel_clahe";

% 2. Edge-preserving denoising (imguidedfilter preserves fine microaneurysm boundaries)
if exist("imguidedfilter", "file")
    G_denoised = imguidedfilter(G_clahe, "NeighborhoodSize", [5 5], "DegreeOfSmoothing", 0.01);
else
    G_denoised = medfilt2(G_clahe, [3 3]);
end
methods(end+1) = "edge_preserving_denoising";

% Blend 70% enhanced green + 30% original green to preserve natural tone
G_final = uint8(0.70 * single(G_denoised) + 0.30 * single(G));

% Subtle dynamic range equalization on Red and Blue
R_clahe = adapthisteq(R, "ClipLimit", 0.01, "NumTiles", [8 8]);
B_clahe = adapthisteq(B, "ClipLimit", 0.01, "NumTiles", [8 8]);

enhanced = cat(3, R_clahe, G_final, B_clahe);

info = struct("applied", true, "methods", methods);
end
