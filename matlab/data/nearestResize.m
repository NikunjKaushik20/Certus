function out = nearestResize(img, outSize)
%NEARESTRESIZE Resample exactly as cv2.resize(..., interpolation=cv2.INTER_NEAREST) does.
%
%   OpenCV takes the source pixel at floor(dst * src/dst). MATLAB's imresize("nearest")
%   samples from pixel centres, which is the more defensible choice in general and the
%   wrong one here: it lands half a pixel over, and on a FOV mask that puts the rim one
%   pixel out. The rim is where the retina meets black, so a one-pixel shift changes a few
%   thousand canvas pixels from fundus to zero -- invisible to look at, but it moves the
%   lesion blobs that touch the edge and therefore the counts.

arguments
    img
    outSize (1,2) double
end

[H, W, ~] = size(img);
sy = floor((0:outSize(1)-1) * H / outSize(1)) + 1;
sx = floor((0:outSize(2)-1) * W / outSize(2)) + 1;
out = img(sy, sx, :);
end
