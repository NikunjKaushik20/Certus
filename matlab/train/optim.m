classdef optim
    %OPTIM AdamW with per-parameter LR factors (discriminative encoder LR), global-norm
    %   clipping, warmup + cosine schedule and weight EMA. Everything stays on the GPU.
    methods (Static)
        function opt = init(model, cfg, encRatio)
            opt.t = 0;
            for f = ["enc" "attn" "grade" "qual"]
                L = model.(f).Learnables;
                opt.(f).m = dlupdate(@(p) zeros(size(p), "like", p), L);
                opt.(f).v = opt.(f).m;
                opt.(f).decay = L;
                opt.(f).decay.Value = num2cell(single(L.Parameter == "Weights"));  % no decay on bias/norm
                fac = ones(height(L), 1, "single");
                if f == "enc", fac(model.meta.isPretrained) = encRatio; end
                opt.(f).factor = L;
                opt.(f).factor.Value = num2cell(fac);
            end
            opt.logvar.m = zeros(size(model.logvar), "like", model.logvar);
            opt.logvar.v = opt.logvar.m;
        end

        function [model, opt, gnorm] = step(model, opt, G, lr, cfg, learnLogvar)
            % G: struct of gradient tables (enc/attn/grade/qual) + logvar dlarray.
            gnorm = optim.globalNorm(G);
            s = min(1, cfg.opt.gradClip / (gnorm + 1e-6));
            opt.t = opt.t + 1;
            c = cfg.opt; t = opt.t;
            for f = ["enc" "attn" "grade" "qual"]
                if ~isfield(G, f), continue; end
                o = opt.(f);
                [model.(f), o.m, o.v] = dlupdate(@(p, g, m, v, fa, d) ...
                    optim.adamw(p, s * g, m, v, fa, d, lr, t, c), ...
                    model.(f), G.(f), o.m, o.v, o.factor, o.decay);
                opt.(f) = o;
            end
            if learnLogvar
                [model.logvar, opt.logvar.m, opt.logvar.v] = optim.adamw(model.logvar, s * G.logvar, ...
                    opt.logvar.m, opt.logvar.v, 1, 0, lr, t, c);
            end
        end

        function [p, m, v] = adamw(p, g, m, v, fa, d, lr, t, c)
            m = c.beta1 * m + (1 - c.beta1) * g;
            v = c.beta2 * v + (1 - c.beta2) * g .^ 2;
            mh = m / (1 - c.beta1 ^ t);
            vh = v / (1 - c.beta2 ^ t);
            p = p - (lr * fa) * (mh ./ (sqrt(vh) + c.eps) + d * c.weightDecay * p);
        end

        function n = globalNorm(G)
            acc = 0;
            for f = string(fieldnames(G))'
                g = G.(f);
                if istable(g)
                    for i = 1:height(g)
                        acc = acc + sum(extractdata(g.Value{i}) .^ 2, "all");
                    end
                else
                    acc = acc + sum(extractdata(g) .^ 2, "all");
                end
            end
            n = sqrt(gather(acc));
        end

        function G = accumulate(G, name, g, w)
            if istable(g)
                if isfield(G, name)
                    G.(name) = dlupdate(@(a, b) a + w * b, G.(name), g);
                else
                    G.(name) = dlupdate(@(b) w * b, g);
                end
            else
                if isfield(G, name), G.(name) = G.(name) + w * g; else, G.(name) = w * g; end
            end
        end

        function lr = schedule(it, total, cfg)
            w = max(1, round(cfg.opt.warmupFrac * total));
            if it <= w
                lr = cfg.opt.lr * it / w;
            else
                p = (it - w) / max(1, total - w);
                lr = cfg.opt.lr * (cfg.opt.minLRFrac + (1 - cfg.opt.minLRFrac) * 0.5 * (1 + cos(pi * p)));
            end
        end

        function ema = emaUpdate(ema, model, d)
            for f = ["enc" "attn" "grade" "qual"]
                ema.(f) = dlupdate(@(e, p) d * e + (1 - d) * p, ema.(f), model.(f).Learnables);
            end
            ema.logvar = model.logvar;
        end
    end
end
