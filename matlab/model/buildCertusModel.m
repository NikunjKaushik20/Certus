function model = buildCertusModel(cfg)
%BUILDCERTUSMODEL One network with a shared ImageNet EfficientNet-B0 encoder.
%   enc   : tile (512px) -> 6-channel lesion/OD/vessel logits (U-Net decoder) + GAP embedding
%   attn  : gated attention over the grid^2 tile embeddings of one eye (MIL pooling)
%   grade : [attention-pooled tiles, global view, lesion evidence] -> 4 ordinal logits
%   qual  : global-view embedding -> good / usable / reject
[enc, meta] = buildEncoderDecoder(cfg);
E = meta.embedDim;
model.enc    = enc;
model.attn   = buildAttention(E, cfg.attnDim);
model.grade  = mlpHead(2 * E + cfg.evidenceDim, cfg.headHidden, cfg.numGrades - 1, cfg.dropout);
model.qual   = mlpHead(E, cfg.headHidden, 3, cfg.dropout);
model.logvar = dlarray(zeros(3, 1, "single"));   % learned task weights: grade, seg, quality
model.meta   = meta;
end

function [net, meta] = buildEncoderDecoder(cfg)
T = cfg.tile;
base = imagePretrainedNetwork(cfg.backbone);
inL = base.Layers(1);
net = initialize(replaceLayer(base, inL.Name, resizedInput(inL, T)));

% Measure every layer's output size at the new input size (no hard-coded layer names).
names = string({net.Layers.Name});
outs = cell(1, numel(names));
[outs{:}] = predict(net, dlarray(zeros(T, T, 3, 1, "single"), "SSCB"), Outputs=names);
hw = zeros(1, numel(names));
for i = 1:numel(names)
    sd = finddim(outs{i}, "S");
    if ~isempty(sd), hw(i) = size(outs{i}, sd(1)); end
end
strides = [2 4 8 16 32];
for s = strides
    idx = find(hw == T / s, 1, "last");
    assert(~isempty(idx), "No encoder layer at stride %d", s);
    meta.taps.("s" + s) = names(idx);
end
last = outs{names == meta.taps.s32};
meta.embedDim = size(last, finddim(last, "C"));
clear outs

% Drop the ImageNet classifier (everything that is no longer spatial).
net = removeLayers(net, names(hw < T / 32));
pretrained = names(hw >= T / 32);

% U-Net decoder: stride 32 -> 16 -> 8 -> 4 -> 2, then x2 to full tile resolution.
C = cfg.decoderChannels;
net = addLayers(net, convolution2dLayer(1, C(1), Name="dec_lat"));
net = connectLayers(net, meta.taps.s32, "dec_lat");
prev = "dec_lat";
for i = 1:numel(C)
    s = strides(end - i);                % 16, 8, 4, 2
    p = "dec" + s;
    g = min(8, C(i));
    net = addLayers(net, resize2dLayer(Scale=2, Method="bilinear", Name=p + "_up"));
    net = addLayers(net, convolution2dLayer(1, C(i), Name=p + "_skip"));
    net = addLayers(net, [
        depthConcatenationLayer(2, Name=p + "_cat")
        convolution2dLayer(3, C(i), Padding="same", Name=p + "_c1")
        groupNormalizationLayer(g, Name=p + "_n1")
        swishLayer(Name=p + "_a1")
        convolution2dLayer(3, C(i), Padding="same", Name=p + "_c2")
        groupNormalizationLayer(g, Name=p + "_n2")
        swishLayer(Name=p + "_a2")]);
    net = connectLayers(net, prev, p + "_up");
    net = connectLayers(net, meta.taps.("s" + s), p + "_skip");
    net = connectLayers(net, p + "_up", p + "_cat/in1");
    net = connectLayers(net, p + "_skip", p + "_cat/in2");
    prev = p + "_a2";
end
K = numel(cfg.segClasses);
net = addLayers(net, [
    convolution2dLayer(1, K, Name="seg_s2", Bias=repmat(single(-4.6), [1 1 K]))  % prior p≈0.01
    resize2dLayer(Scale=2, Method="bilinear", Name="seg_logits")]);
net = connectLayers(net, prev, "seg_s2");
net = addLayers(net, globalAveragePooling2dLayer(Name="emb_gap"));
net = connectLayers(net, meta.taps.s32, "emb_gap");
net = initialize(net);

% Name the two encoder outputs in meta rather than letting callers hard-code them: the
% ONNX export calls the embedding "emb", this builder calls it "emb_gap", and gradeForward
% has to drive either one.
meta.segOutput = "seg_logits";
meta.embOutput = "emb_gap";
meta.isPretrained = ismember(net.Learnables.Layer, pretrained);
meta.inputScale = inputScale(inL);
meta.numParams = sum(cellfun(@numel, net.Learnables.Value));
end

function L = resizedInput(inL, T)
nm = string(inL.Normalization);
avg = @(x) mean(x, [1 2]);
args = {"Name", inL.Name, "Normalization", nm};
switch nm
    case "zscore",     args = [args {"Mean", avg(inL.Mean), "StandardDeviation", avg(inL.StandardDeviation)}];
    case "zerocenter", args = [args {"Mean", avg(inL.Mean)}];
    case {"rescale-symmetric", "rescale-zero-one"}, args = [args {"Min", avg(inL.Min), "Max", avg(inL.Max)}];
end
L = imageInputLayer([T T 3], args{:});
end

function s = inputScale(inL)
% Pixel range the pretrained normalisation expects: [0,255] (MathWorks default) or [0,1].
ref = [];
if isprop(inL, "Mean") && ~isempty(inL.Mean), ref = inL.Mean; end
if startsWith(string(inL.Normalization), "rescale"), ref = inL.Max; end
s = 255;
if ~isempty(ref) && max(ref(:)) <= 1.5, s = 1; end
end

function net = buildAttention(E, A)
net = dlnetwork;
net = addLayers(net, featureInputLayer(E, Name="in"));
net = addLayers(net, [fullyConnectedLayer(A, Name="V"), tanhLayer(Name="tanh")]);
net = addLayers(net, [fullyConnectedLayer(A, Name="U"), sigmoidLayer(Name="sig")]);
net = addLayers(net, [multiplicationLayer(2, Name="gate"), fullyConnectedLayer(1, Name="w")]);
net = connectLayers(net, "in", "V");
net = connectLayers(net, "in", "U");
net = connectLayers(net, "tanh", "gate/in1");
net = connectLayers(net, "sig", "gate/in2");
net = initialize(net);
end

function net = mlpHead(inDim, hidden, outDim, p)
net = dlnetwork([
    featureInputLayer(inDim, Name="in")
    fullyConnectedLayer(hidden, Name="fc1")
    layerNormalizationLayer(Name="ln1")
    geluLayer(Name="act1")
    dropoutLayer(p, Name="drop")
    fullyConnectedLayer(outDim, Name="out")]);
end
