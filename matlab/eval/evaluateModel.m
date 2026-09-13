function R = evaluateModel(model, M, cfg, maxImages, part)
%EVALUATEMODEL Grading / segmentation / quality metrics on a split.
%   part: "val" (default) or "test" / "external" (grading only). maxImages caps each
%   set with a fixed-seed subsample so repeated validations are comparable.
if nargin < 4, maxImages = inf; end
if nargin < 5, part = "val"; end
switch part
    case "val",      TG = M.gradeVal;  TS = M.segVal;  TQ = M.qualVal;
    case "test",     TG = M.gradeTest; TS = M.segTest; TQ = [];
    case "external", TG = M.gradeExt;  TS = [];        TQ = [];
end
R.grade = evalGrading(model, subsample(TG, maxImages), cfg);
if ~isempty(TS) && height(TS) > 0, R.seg = evalSeg(model, subsample(TS, maxImages), cfg); end
if ~isempty(TQ) && height(TQ) > 0, R.quality = evalQuality(model, subsample(TQ, maxImages), cfg); end
end

function T = subsample(T, n)
if height(T) > n
    s = RandStream("twister", Seed=0);
    T = T(sort(randperm(s, height(T), n)), :);
end
end

function R = evalGrading(model, T, cfg)
n = height(T); bs = 2;
P = zeros(4, n); Q = zeros(3, n);
reader = ImageReader(T.proc_path, cfg);
for i = 1:bs:n
    idx = i:min(i + bs - 1, n);
    X = reader.take(numel(idx));
    out = gradeForward(model, single(gpuArray(X)) / 255, cfg, false);
    P(:, idx) = gather(extractdata(sigmoid(out.gradeLogits)));
    Q(:, idx) = gather(extractdata(softmax(out.qualLogits)));
end
P = cummin(P, 1);                          % enforce P(>0) >= P(>1) >= ...
y = T.dr_grade;
pred = sum(P > 0.5, 1)';
ref = P(2, :)';                            % P(grade >= 2)
R.n = n;
R.aucReferable = metrics.auc(y >= 2, ref);
[R.sensAt085Spec, R.specAt085, R.thrAt085] = metrics.atSpecificity(y >= 2, ref, 0.85);
[R.sensAt05, R.specAt05] = metrics.atThreshold(y >= 2, ref, 0.5);
R.qwk = metrics.qwk(y, pred, cfg.numGrades);
R.accuracy = mean(pred == y);
R.eceReferable = metrics.ece(y >= 2, ref, 10);
R.perDataset = table();
for d = unique(T.dataset)'
    m = T.dataset == d;
    R.perDataset = [R.perDataset; table(d, sum(m), metrics.auc(y(m) >= 2, ref(m)), ...
        metrics.qwk(y(m), pred(m), cfg.numGrades), VariableNames=["dataset" "n" "auc" "qwk"])];
end
R.probs = P; R.qualProbs = Q; R.uid = T.uid;
end

function R = evalSeg(model, T, cfg)
L = numel(cfg.lesionClasses);
tp = zeros(L, 1); fp = tp; fn = tp;
reader = ImageReader(T.path, cfg);
for i = 1:height(T)
    X = reader.take(1);
    out = gradeForward(model, single(gpuArray(X)) / 255, cfg, false);
    P = gather(out.lesionProb) > 0.5;
    for c = 1:L
        if T.masks(i, c) == "", continue; end
        Y = imread(T.masks(i, c)) > 0;
        tp(c) = tp(c) + nnz(P(:, :, c) & Y);
        fp(c) = fp(c) + nnz(P(:, :, c) & ~Y);
        fn(c) = fn(c) + nnz(~P(:, :, c) & Y);
    end
end
R.dice = array2table(metrics.dice(tp, fp, fn)', VariableNames=cfg.lesionClasses);
end

function R = evalQuality(model, T, cfg)
Gs = cfg.globalSize; n = height(T); bs = 16;
Q = zeros(3, n);
for i = 1:bs:n
    idx = i:min(i + bs - 1, n);
    X = zeros(Gs, Gs, 3, numel(idx), "single");
    for j = 1:numel(idx)
        X(:, :, :, j) = single(imresize(imread(T.proc_path(idx(j))), [Gs Gs])) / 255;
    end
    e = predict(model.enc, dlarray(gpuArray(X) * model.meta.inputScale, "SSCB"), Outputs="emb_gap");
    e = dlarray(reshape(stripdims(e), model.meta.embedDim, []), "CB");
    Q(:, idx) = gather(extractdata(softmax(predict(model.qual, e))));
end
[~, pred] = max(Q, [], 1);
y = T.quality_label;
R.accuracy = mean(pred' - 1 == y);
R.aucReject = metrics.auc(y == 2, Q(3, :)');
end
