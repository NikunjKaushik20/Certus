function model = importCertusOnnx(folder)
%IMPORTCERTUSONNX Load the PyTorch-trained Certus network (export_onnx.py) into the same
%   model struct that buildCertusModel returns, so gradeForward / evaluateModel /
%   reports run unchanged.
%
%   model = importCertusOnnx("D:/Certus/runs_torch/train_<stamp>/onnx")
%
%   Requires the Deep Learning Toolbox Converter for ONNX Model Format.
meta = jsondecode(fileread(fullfile(folder, "certus_meta.json")));

% importNetworkFromONNX writes the custom layers it generates (+certus_enc and friends)
% into the *current folder*, and the imported network cannot be used -- or loaded back from
% a .mat -- unless they are on the path. Left to itself that puts them wherever the user
% happened to be standing. Pin them next to startup_certus, which already adds that folder
% to the path, so the same packages are found on the next run.
home = fileparts(fileparts(mfilename("fullpath")));
back = cd(home);
restore = onCleanup(@() cd(back));

imp = @(f, fmt) importNetworkFromONNX(fullfile(folder, f), InputDataFormats=fmt);
model.enc   = imp("certus_enc.onnx", "BCSS");
model.attn  = imp("certus_attn.onnx", "BC");
model.grade = imp("certus_grade.onnx", "BC");
model.qual  = imp("certus_qual.onnx", "BC");
model.logvar = dlarray(zeros(3, 1, "single"));
% Stride-32 activation map, used only by certusGradCam. Older exports do not have it.
if isfile(fullfile(folder, "certus_feat.onnx"))
    model.feat = imp("certus_feat.onnx", "BCSS");
end
% Second reader (torch/second_reader.py): the plain whole-image grader. Older exports do not have it,
% and without it Certus decides alone.
if isfile(fullfile(folder, "certus_second.onnx")) && isfield(meta, "second_reader")
    model.second = imp("certus_second.onnx", "BCSS");
end

model.meta.embedDim   = meta.embed_dim;
model.meta.inputScale = meta.input_scale;          % graphs expect [0,1] input
model.meta.canvas     = meta.canvas;
model.meta.tile       = meta.tile;
model.meta.grid       = meta.grid;
model.meta.checkpoint = string(meta.checkpoint);
model.meta.step       = meta.step;
for f = ["temperature" "conformal" "screening_band" "lesion_thresholds" "second_reader"]
    if isfield(meta, f), model.meta.(f) = meta.(f); end
end
if canUseGPU, model = modelToGpu(model); end

% The importer renames graph outputs after the ops that produced them ("x_Resize_4",
% "ReduceMeanLayer1016"), so the names export_onnx.py gave them are gone by the time we
% get here. Identify them by shape rather than by position: only the segmentation head is
% spatial. Trusting OutputNames{1}/{2} would silently swap seg and emb if the importer
% ever reordered them, and that failure looks like a wrong grade, not an error.
[model.meta.segOutput, model.meta.embOutput] = encoderOutputs(model);
end


function [segName, embName] = encoderOutputs(model)
%ENCODEROUTPUTS Which encoder output is the lesion map and which is the embedding.
t = model.meta.tile;
x = dlarray(zeros(t, t, 3, 1, "single"), "SSCB");
if canUseGPU, x = gpuArray(x); end
names = string(model.enc.OutputNames);
y = cell(1, numel(names));
[y{:}] = predict(model.enc, x, Outputs=names);
spatial = cellfun(@(a) ~isempty(finddim(a, "S")), y);
assert(nnz(spatial) == 1, ...
       "Certus: expected exactly one spatial encoder output, found %d", nnz(spatial));
segName = names(spatial);
embName = names(~spatial);
emb = y{~spatial};
assert(size(emb, finddim(emb, "C")) == model.meta.embedDim, ...
       "Certus: embedding output is not %d wide", model.meta.embedDim);
end
