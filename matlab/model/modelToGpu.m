function model = modelToGpu(model)
%MODELTOGPU Move all learnables and state to the GPU.
for f = intersect(["enc" "attn" "grade" "qual" "feat" "second"], string(fieldnames(model))', "stable")
    model.(f) = dlupdate(@gpuArray, model.(f));
    if ~isempty(model.(f).State)
        model.(f).State = dlupdate(@gpuArray, model.(f).State);
    end
end
model.logvar = gpuArray(model.logvar);
end
