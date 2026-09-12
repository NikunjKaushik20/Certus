function ev = evidenceFeatures(P, cfg, n)
%EVIDENCEFEATURES Lesion evidence vector per eye (12 x n) from lesion probabilities.
%   P: t x t x 4 x (grid^2*n) sigmoid maps for MA, HE, EX, SE (FOV-masked, no gradient).
%   rows 1-4  log1p(soft lesion area / 100)          "how much of each lesion"
%   rows 5-8  peak probability                        "is any lesion clearly present"
%   rows 9-12 HE mass per quadrant, sorted descending "haemorrhage spread (4-2-1 rule)"
Pc = stitchTiles(P, cfg);
mass = reshape(sum(Pc, [1 2]), 4, n);
peak = reshape(max(Pc, [], [1 2]), 4, n);
h = size(Pc, 1) / 2;
he = Pc(:, :, 2, :);
q = [reshape(sum(he(1:h, 1:h, :, :), [1 2]), 1, n)
     reshape(sum(he(1:h, h+1:end, :, :), [1 2]), 1, n)
     reshape(sum(he(h+1:end, 1:h, :, :), [1 2]), 1, n)
     reshape(sum(he(h+1:end, h+1:end, :, :), [1 2]), 1, n)];
ev = [log1p(mass / 100); peak; log1p(sort(q, 1, "descend") / 100)];
end
