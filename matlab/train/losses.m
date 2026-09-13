classdef losses
    %LOSSES Per-task loss functions, each evaluated inside dlfeval.
    %   Task weights are learned (homoscedastic uncertainty): L = 0.5*exp(-s)*L_t + 0.5*s.
    methods (Static)
        function [loss, gEnc, gAttn, gGrade, gLv, Lg] = grade(model, Xc, y, cfg)
            out = gradeForward(model, Xc, cfg, true);
            Lg = losses.ordinal(out.gradeLogits, y, cfg);
            loss = losses.weigh(Lg, model.logvar(1));
            [gEnc, gAttn, gGrade, gLv] = dlgradient(loss, model.enc.Learnables, ...
                model.attn.Learnables, model.grade.Learnables, model.logvar);
            Lg = gather(extractdata(Lg));
        end

        function [loss, gEnc, gLv, Ls] = seg(model, X, Y, V, cfg)
            Z = predict(model.enc, dlarray(X * model.meta.inputScale, "SSCB"), Outputs="seg_logits");
            F = single(any(X > 0.02, 3));
            Ls = losses.segmentation(stripdims(Z), Y, V, F, cfg);
            loss = losses.weigh(Ls, model.logvar(2));
            [gEnc, gLv] = dlgradient(loss, model.enc.Learnables, model.logvar);
            Ls = gather(extractdata(Ls));
        end

        function [loss, gEnc, gQual, gLv, Lq] = quality(model, X, y, cfg)
            e = predict(model.enc, dlarray(X * model.meta.inputScale, "SSCB"), Outputs="emb_gap");
            e = dlarray(reshape(stripdims(e), model.meta.embedDim, []), "CB");
            z = stripdims(forward(model.qual, e));                 % 3 x N
            Lq = losses.partialCE(z, y);
            loss = losses.weigh(Lq, model.logvar(3));
            [gEnc, gQual, gLv] = dlgradient(loss, model.enc.Learnables, model.qual.Learnables, model.logvar);
            Lq = gather(extractdata(Lq));
        end

        function L = weigh(Lt, s)
            L = 0.5 * exp(-s) * Lt + 0.5 * s;
        end

        function L = ordinal(z, y, cfg)
            % Cumulative-link ordinal loss: logit k models P(grade > k), k = 0..3.
            % Threshold k=1 is exactly "referable DR" and is up-weighted.
            z = stripdims(z);
            K = size(z, 1);
            T = gpuArray(single(reshape(y, 1, []) > (0:K-1)'));
            w = ones(K, 1, "single", "gpuArray");
            w(2) = cfg.loss.referableWeight;
            bce = max(z, 0) - z .* T + log(1 + exp(-abs(z)));
            L = sum(bce .* w, "all") / (sum(w) * size(z, 2));
        end

        function L = segmentation(Z, Y, V, F, cfg)
            % Focal BCE + focal Tversky; classes not annotated for an image (V=0) are ignored.
            P = sigmoid(Z);
            bce = max(Z, 0) - Z .* Y + log(1 + exp(-abs(Z)));
            pt = P .* Y + (1 - P) .* (1 - Y);
            W = F .* V;
            Lf = sum(bce .* (1 - pt) .^ cfg.loss.focalGamma .* W, "all") / max(sum(W, "all"), 1);
            TP = sum(P .* Y .* F, [1 2]);
            FP = sum(P .* (1 - Y) .* F, [1 2]);
            FN = sum((1 - P) .* Y .* F, [1 2]);
            TI = (TP + 1) ./ (TP + cfg.loss.tverskyAlpha * FP + cfg.loss.tverskyBeta * FN + 1);
            Lt = sum((1 - TI + 1e-6) .^ 0.75 .* V, "all") / max(sum(V, "all"), 1);
            L = Lf + Lt;
        end

        function L = partialCE(z, y)
            % y in {0,1,2}: full label; y = -1: label set {good, usable} ("not reject").
            N = size(z, 2);
            M = zeros(3, N, "single");
            full = y >= 0;
            M(sub2ind([3 N], y(full) + 1, find(full))) = 1;
            M(1:2, ~full) = 1;
            M = gpuArray(M);
            m = max(z, [], 1);
            e = exp(z - m);
            L = -sum(log(sum(e .* M, 1)) - log(sum(e, 1)), "all") / N;
        end
    end
end
