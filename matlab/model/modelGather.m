function model = modelGather(model)
%MODELGATHER Copy learnables/state to host memory (for checkpoints).
for f = intersect(["enc" "attn" "grade" "qual" "feat" "second"], string(fieldnames(model))', "stable")
    model.(f) = dlupdate(@gather, model.(f));
    if ~isempty(model.(f).State)
        model.(f).State = dlupdate(@gather, model.(f).State);
    end
end
model.logvar = gather(model.logvar);
end
