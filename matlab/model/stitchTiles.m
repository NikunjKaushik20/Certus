function X = stitchTiles(T, cfg)
%STITCHTILES Inverse of makeTiles: t x t x C x (grid^2 * n) -> D x D x C x n.
t = cfg.tile; g = cfg.grid;
C = size(T, 3); n = size(T, 4) / (g * g);
X = reshape(permute(reshape(T, [t t C g g n]), [1 5 2 4 3 6]), t * g, t * g, C, n);
end
