classdef metrics
    %METRICS Screening metrics (no toolbox-specific curve functions needed).
    methods (Static)
        function a = auc(y, s)
            y = logical(y(:)); s = double(s(:));
            np = sum(y); nn = sum(~y);
            if np == 0 || nn == 0, a = NaN; return; end
            r = tiedrank(s);
            a = (sum(r(y)) - np * (np + 1) / 2) / (np * nn);
        end

        function [sens, spec, thr] = atSpecificity(y, s, target)
            % Highest-sensitivity threshold whose specificity >= target.
            y = logical(y(:)); s = double(s(:));
            thr = quantile(s(~y), target);
            sens = mean(s(y) > thr);
            spec = mean(s(~y) <= thr);
        end

        function [sens, spec] = atThreshold(y, s, thr)
            y = logical(y(:)); s = double(s(:));
            sens = mean(s(y) > thr);
            spec = mean(s(~y) <= thr);
        end

        function k = qwk(a, b, K)
            O = accumarray([a(:) b(:)] + 1, 1, [K K]);
            W = ((0:K-1)' - (0:K-1)) .^ 2 / (K - 1) ^ 2;
            E = sum(O, 2) * sum(O, 1) / sum(O, "all");
            k = 1 - sum(W .* O, "all") / sum(W .* E, "all");
        end

        function e = ece(y, p, nBins)
            y = double(y(:)); p = double(p(:));
            edges = linspace(0, 1, nBins + 1);
            [~, ~, b] = histcounts(p, edges);
            e = 0;
            for k = 1:nBins
                m = b == k;
                if any(m), e = e + sum(m) / numel(p) * abs(mean(y(m)) - mean(p(m))); end
            end
        end

        function d = dice(tp, fp, fn)
            d = 2 * tp ./ max(2 * tp + fp + fn, 1);
        end
    end
end
