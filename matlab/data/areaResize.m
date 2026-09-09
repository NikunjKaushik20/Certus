function out = areaResize(img, outSize)
%AREARESIZE Downscale exactly as cv2.resize(..., interpolation=cv2.INTER_AREA) does.
%
%   MATLAB's imresize(...,"box") is the same idea but not the same arithmetic: for a
%   non-integer scale factor the two disagree on a handful of pixels at the edge of a
%   region. That sounds harmless and is not. The FOV mask is a threshold on this downscaled
%   copy, so four different pixels moved the detected retina bounding box by one pixel on
%   IDRiD_008, which shifted the whole 1536 px canvas by ~8 raw pixels and moved
%   p(referable) from 0.9957 to 0.9902. Same grade, but a number that does not match the
%   console invites exactly the question you least want to answer live.
%
%   This is OpenCV's computeResizeAreaTab, built as two sparse matrices and applied
%   separably. Only downscaling is supported, which is all normalizeFov asks for.

arguments
    img
    outSize (1,2) double
end

[H, W, C] = size(img);
oh = outSize(1); ow = outSize(2);
assert(oh <= H && ow <= W, "areaResize only downscales (%dx%d -> %dx%d)", H, W, oh, ow);

x = double(reshape(img, H, W * C));
t = areaWeights(H, oh) * x;                       % rows:  oh x W*C
t = reshape(permute(reshape(t, oh, W, C), [2 1 3]), W, oh * C);
t = areaWeights(W, ow) * t;                       % cols:  ow x oh*C
out = permute(reshape(t, ow, oh, C), [2 1 3]);

if isinteger(img)
    out = cast(min(max(cvRound(out), 0), double(intmax(class(img)))), "like", img);
end
end


function A = areaWeights(inN, outN)
%AREAWEIGHTS Sparse outN x inN matrix of the area weights OpenCV would use.
scale = inN / outN;
i = zeros(0, 1); j = zeros(0, 1); v = zeros(0, 1);
for d = 0:outN-1
    fs1 = d * scale;
    fs2 = fs1 + scale;
    cellWidth = min(scale, inN - fs1);            % the last output cell can be clipped
    s1 = ceil(fs1);
    s2 = min(floor(fs2), inN - 1);
    s1 = min(s1, s2);
    % 1e-3 is OpenCV's own tolerance, not a guess: below it the partial pixel is dropped
    % entirely rather than given a near-zero weight.
    if s1 - fs1 > 1e-3
        i(end+1, 1) = d + 1;  j(end+1, 1) = s1;      v(end+1, 1) = (s1 - fs1) / cellWidth;
    end
    for s = s1:s2-1
        i(end+1, 1) = d + 1;  j(end+1, 1) = s + 1;   v(end+1, 1) = 1 / cellWidth;
    end
    if fs2 - s2 > 1e-3
        i(end+1, 1) = d + 1;  j(end+1, 1) = s2 + 1;
        v(end+1, 1) = min(min(fs2 - s2, 1), cellWidth) / cellWidth;
    end
end
A = sparse(i, j, v, outN, inN);
end


function y = cvRound(x)
%CVROUND Round half to even, which is what cvRound / saturate_cast<uchar> does. MATLAB's
%   round goes half away from zero, and the difference shows up on flat grey padding.
y = round(x);
half = abs(x - fix(x)) == 0.5;
y(half) = 2 * round(x(half) / 2);
end
