function T = makeTiles(X, cfg)
%MAKETILES D x D x C x n canvases -> t x t x C x (grid^2 * n) tiles, row-major per image.
t = cfg.tile; g = cfg.grid;
[~, ~, C, n] = size(X);
T = reshape(permute(reshape(X, [t g t g C n]), [1 3 5 4 2 6]), t, t, C, g * g * n);
end
