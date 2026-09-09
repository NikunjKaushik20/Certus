function M = loadManifests(cfg)
%LOADMANIFESTS Read the cleaned CSV manifests into split-specific tables.
rd = @(f) readtable(fullfile(cfg.paths.manifests, f), TextType="string", Delimiter=",");

G = rd("grading.csv");
M.gradeTrain = G(G.split == "train", :);
M.gradeVal   = G(G.split == "val", :);
M.gradeTest  = G(G.split == "test", :);
M.gradeExt   = G(G.split == "external_test", :);   % Messidor-2: never used for training/selection

S = [segTable(rd("segmentation.csv"), cfg); segTable(rd("vessels.csv"), cfg)];
M.segTrain = S(S.split == "train", :);
M.segVal   = S(S.split == "val", :);
M.segTest  = S(S.split == "test", :);

Q = rd("quality.csv");
Q.quality_label(isnan(Q.quality_label) & Q.gradable == 0) = 2;
keep = ~isnan(Q.quality_label) | Q.gradable == 1;
Q = Q(keep, :);
M.qualTrain = Q(Q.split == "train", :);
M.qualVal   = Q(Q.split == "val" & ~isnan(Q.quality_label), :);
end

function T = segTable(T0, cfg)
% One row per image, masks(:,c) = mask path for cfg.segClasses(c) or "" when unknown.
n = height(T0);
masks = strings(n, numel(cfg.segClasses));
for c = 1:numel(cfg.segClasses)
    col = "proc_mask_" + cfg.segClasses(c);
    if ~ismember(col, T0.Properties.VariableNames), continue; end
    v = T0.(col);
    if ~isstring(v), continue; end          % all-empty column parsed as numeric
    v(ismissing(v) | v == "unknown") = "";
    masks(:, c) = v;
end
T = table(T0.uid, T0.dataset, T0.split, T0.proc_path, masks, ...
    VariableNames=["uid" "dataset" "split" "path" "masks"]);
end
