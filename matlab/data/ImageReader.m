classdef ImageReader < handle
    %IMAGEREADER In-order image reading with background-thread read-ahead (evaluation).
    properties
        paths
        futures
        nextIssue = 1
        nextTake = 1
        depth
        canvas
    end
    methods
        function obj = ImageReader(paths, cfg)
            obj.paths = paths;
            obj.canvas = cfg.canvas;
            obj.depth = 2 * cfg.prefetchDepth;
            obj.futures = cell(1, numel(paths));
            obj.issue(obj.depth);
        end

        function issue(obj, k)
            last = min(numel(obj.paths), obj.nextIssue + k - 1);
            for i = obj.nextIssue:last
                obj.futures{i} = parfeval(backgroundPool, @imread, 1, obj.paths(i));
            end
            obj.nextIssue = last + 1;
        end

        function X = take(obj, k)
            D = obj.canvas;
            X = zeros(D, D, 3, k, "uint8");
            for j = 1:k
                I = fetchOutputs(obj.futures{obj.nextTake});
                obj.futures{obj.nextTake} = [];
                if size(I, 1) ~= D, I = imresize(I, [D D]); end
                X(:, :, :, j) = I;
                obj.nextTake = obj.nextTake + 1;
                obj.issue(1);
            end
        end
    end
end
