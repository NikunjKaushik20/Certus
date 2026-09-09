function [canvas, mask, tf, info] = normalizeFov(img, D)
%NORMALIZEFOV Crop a raw fundus photograph to a centred DxD canvas whose side is the FOV
%   diameter.
%
%   MATLAB mirror of scripts/retina.py:normalize_fov. The network was trained on canvases
%   that function produced, so this is not cosmetic preprocessing -- any drift here is a
%   different input to the same weights, and shows up as a different grade for the same
%   photograph. run/certus_demo.m measures the drift against the Python canvas rather than
%   trusting this comment.
%
%   img    HxWx3 uint8, RGB (imread order). Only per-pixel channel maxima are used to find
%          the FOV, so RGB-vs-BGR does not matter here.
%   D      canvas side in pixels (cfg.canvas, 1536).
%
%   canvas DxDx3 uint8, everything outside the FOV set to 0.
%   mask   DxD logical FOV.
%   tf     struct mapping raw (x,y) -> canvas: x' = (x - x0) * s. Zero-based, like Python.
%   info   struct; info.fovOk is false and the rest empty when no FOV could be found.

arguments
    img  uint8
    D    (1,1) double
end

% -------------------------------------------------------------- FOV mask, downscaled
% Finding the disc on a 512 px copy costs nothing and makes the morphology cheap. The
% bounding box is scaled back up afterwards, so the loss of precision is sub-pixel on the
% final canvas.
[H, W, ~] = size(img);
s0 = 512 / max(H, W);
small = areaResize(img, [round(H * s0) round(W * s0)]);
v = single(max(small, [], 3));
[hs, ws] = size(v);

% Some cameras (about 5% of DDR) pad with flat grey near 17 rather than black, so the
% threshold is lifted above whatever the corners actually contain.
c = max(4, floor(min(hs, ws) / 20));
corners = [reshape(v(1:c, 1:c), [], 1); reshape(v(1:c, end-c+1:end), [], 1)
           reshape(v(end-c+1:end, 1:c), [], 1); reshape(v(end-c+1:end, end-c+1:end), [], 1)];
thr = max([8, median(corners) + 10, 0.06 * prctile(v(:), 99.5)]);
m = v > thr;

m = imclose(imopen(m, ELLIPSE7), ELLIPSE7);
cc = bwconncomp(m, 8);
if cc.NumObjects == 0
    [canvas, mask, tf] = deal([]);
    info = struct("fovOk", false);
    return
end
[~, big] = max(cellfun(@numel, cc.PixelIdxList));
m = false(size(m));
m(cc.PixelIdxList{big}) = true;
m = imfill(m, "holes");         % dark lesions and the disc rim punch holes in dim images

% -------------------------------------------------------------- square crop box
% Zero-based throughout to match the Python, which is what wrote the training canvases.
[ys, xs] = find(m);
x0s = min(xs) - 1;  x1s = max(xs);
y0s = min(ys) - 1;  y1s = max(ys);
bw = (x1s - x0s) / s0;
bh = (y1s - y0s) / s0;
diam = max(bw, bh);             % a FOV clipped top and bottom (common in APTOS) keeps its
cx = (x0s + x1s) / 2 / s0;      % true horizontal diameter
cy = (y0s + y1s) / 2 / s0;
ix0 = floor(cx - diam / 2);
iy0 = floor(cy - diam / 2);
side = ceil(diam);

canvas = resizeCrop(img, ix0, iy0, side, H, W, D, D / diam >= 1);

% -------------------------------------------------------------- canvas mask
% What the camera actually exposed, intersected with the ideal circle, so a lens shadow or
% a clipped edge cannot leak into a tile as signal.
mFull = nearestResize(m, [H W]);
mc = resizeCrop(uint8(mFull), ix0, iy0, side, H, W, D, true, "nearest") > 0;
[gx, gy] = meshgrid(0:D-1, 0:D-1);
circ = (gx - floor(D/2)).^2 + (gy - floor(D/2)).^2 <= (D/2 - 1)^2;
mask = mc & circ;
canvas = canvas .* uint8(mask);

tf = struct("x0", ix0, "y0", iy0, "s", D / side);
info = struct("fovOk", true, "rawW", W, "rawH", H, "nativeFovPx", round(diam, 1), ...
              "fovClipFrac", round(1 - min(bw, bh) / diam, 4), ...
              "fovAreaFrac", round(nnz(mask) / nnz(circ), 4), "scale", round(D / side, 5));
end


function out = resizeCrop(img, ix0, iy0, side, H, W, D, upscaling, method)
%RESIZECROP Zero-padded square crop at zero-based (ix0, iy0), resized to DxD.
%   The crop box routinely runs off the sensor -- the FOV touches the frame edge on most
%   handheld cameras -- so the parts that fall outside are filled with black rather than
%   clamped, which would stretch the retina.
if nargin < 9
    % Match OpenCV: INTER_AREA down, INTER_CUBIC up. MATLAB antialiases bicubic by default
    % and OpenCV does not, so the upscaling branch turns it off.
    method = "area";
    if upscaling, method = "bicubic"; end
end
r0 = max(0, iy0);  r1 = min(H, iy0 + side);
c0 = max(0, ix0);  c1 = min(W, ix0 + side);
crop = img(r0+1:r1, c0+1:c1, :);
out = zeros(side, side, size(img, 3), "like", img);
out(r0-iy0+1 : r0-iy0+(r1-r0), c0-ix0+1 : c0-ix0+(c1-c0), :) = crop;
switch method
    case "bicubic", out = imresize(out, [D D], "bicubic", Antialiasing=false);
    case "area",    out = areaResize(out, [D D]);
    otherwise,      out = nearestResize(out, [D D]);
end
end


function k = ELLIPSE7()
%ELLIPSE7 strel matching cv2.getStructuringElement(MORPH_ELLIPSE, (7,7)) exactly.
%   MATLAB's strel("disk", 3) is a different set of pixels, and the difference moves the
%   bounding box, which moves every tile boundary downstream.
persistent s
if isempty(s)
    s = strel("arbitrary", logical([...
        0 0 0 1 0 0 0
        0 1 1 1 1 1 0
        1 1 1 1 1 1 1
        1 1 1 1 1 1 1
        1 1 1 1 1 1 1
        0 1 1 1 1 1 0
        0 0 0 1 0 0 0]));
end
k = s;
end
