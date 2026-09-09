classdef Prefetcher < handle
    %PREFETCHER Keeps `depth` optimiser steps decoding on background threads
    %   so JPEG/PNG decoding overlaps with GPU compute.
    properties
        pool
        futures = {}
        makeSpec
        cfg
    end
    methods
        function obj = Prefetcher(makeSpecFcn, cfg)
            obj.makeSpec = makeSpecFcn;
            obj.cfg = cfg;
            obj.pool = backgroundPool;
            for i = 1:cfg.prefetchDepth
                obj.push();
            end
        end

        function push(obj)
            obj.futures{end+1} = parfeval(obj.pool, @loadStep, 1, obj.makeSpec(), obj.cfg);
        end

        function B = next(obj)
            f = obj.futures{1};
            obj.futures(1) = [];
            obj.push();
            B = fetchOutputs(f);
        end

        function delete(obj)
            cellfun(@cancel, obj.futures);
        end
    end
end
