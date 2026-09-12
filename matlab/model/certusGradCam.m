function [heat, blendRgb] = certusGradCam(model, X, out, cfg, fov, canvas, target)
%CERTUSGRADCAM Grad-CAM for the ordinal grade head, the same computation as api/certus_api/gradcam.py.
%
%   [heat, blendRgb] = certusGradCam(model, X, out, cfg, fov, canvas)
%   [heat, blendRgb] = certusGradCam(..., target)    % "referable" (default), "any_dr",
%                                                      % "moderate", "proliferative"
%
%   model:   imported model struct (importCertusOnnx) with the certus_feat graph
%   X:       D x D x 3 canvas in [0,1], the input gradeForward was given
%   out:     gradeForward output for X (inference mode: carries embT, embG, evidence)
%   fov:     D x D logical retina mask
%   canvas:  D x D x 3 uint8, for the overlay
%
%   heat:     D x D in [0,1], zero outside the FOV
%   blendRgb: D x D x 3 uint8 overlay
%
%   Grad-CAM needs, per tile, the stride-32 activation A (16 x 16 x E) and the channel
%   weights w_c = mean over space of d(logit)/dA_c. The tile embedding is the spatial mean
%   of A (CertusNet.embed), so d(logit)/dA_c(x,y) = d(logit)/d(emb_c) / (16*16) at every
%   location, and w_c = d(logit)/d(emb_c) / (16*16). That gradient only runs through the
%   attention scorer, the softmax over tiles and the grade head, so dlgradient on those
%   small graphs gives the exact Grad-CAM weights without differentiating the encoder.
%   Each tile's embedding depends only on its own pixels, so the gradient with respect to
%   column i of embT is exactly what the Python version gets from its per-tile backward.
%   The evidence vector and the global embedding are constants here, as in Python (the
%   evidence is stop-gradient by design).

arguments
    model struct
    X (:, :, 3) {mustBeNumeric}
    out struct
    cfg struct
    fov (:, :) logical
    canvas (:, :, 3) uint8
    target string = "referable"
end
assert(isfield(model, "feat"), "certusGradCam: this export has no certus_feat.onnx. " + ...
       "Run: python torch/export_onnx.py <best.pt> --only feat");

idx = find(target == ["any_dr" "referable" "moderate" "proliferative"]);
assert(~isempty(idx), "certusGradCam: unknown target %s", target);

g2 = cfg.grid ^ 2;
t = cfg.tile;
E = model.meta.embedDim;

% ---- channel weights: d(target logit) / d(tile embeddings), E x 9
embT = dlarray(single(gather(stripdims(out.embT))), "CB");
embG = single(gather(stripdims(out.embG)));
ev = out.evidence;
if isa(ev, "dlarray"), ev = extractdata(ev); end
ev = single(gather(ev));
if canUseGPU
    embT = gpuArray(embT); embG = gpuArray(embG); ev = gpuArray(ev);
end
dEmb = dlfeval(@targetGrad, model, embT, embG, ev, idx, g2);
w = gather(extractdata(dEmb));                                  % E x 9

% ---- stride-32 activations, 16 x 16 x E x 9
tiles = makeTiles(X, cfg);
A = predict(model.feat, dlarray(tiles * model.meta.inputScale, "SSCB"));
A = gather(extractdata(A));
assert(size(A, 3) == E && size(A, 4) == g2, "certusGradCam: unexpected activation shape");
hw = size(A, 1) * size(A, 2);

cams = zeros(t, t, 1, g2, "single");
for i = 1:g2
    cam = max(0, sum(A(:, :, :, i) .* reshape(w(:, i) / hw, 1, 1, E), 3));
    cams(:, :, 1, i) = imresize(cam, [t t], "bilinear", Antialiasing=false);
end
heat = double(stitchTiles(cams, cfg));
heat = heat .* fov;
m = max(heat(:));
if m > 1e-8, heat = heat / m; end

% ---- overlay: 50% jet inside the FOV, the photograph outside
rgb = ind2rgb(round(heat * 255) + 1, jet(256));
blend = double(canvas) / 255;
mix = 0.5 * blend + 0.5 * rgb;
fov3 = repmat(fov, 1, 1, 3);
blend(fov3) = mix(fov3);
blendRgb = uint8(255 * blend);
end


function dEmb = targetGrad(model, embT, embG, ev, idx, g2)
s = predict(model.attn, embT);
s = reshape(s, 1, g2);
a = exp(s - max(s));
a = a / sum(a);
pooled = sum(stripdims(embT) .* a, 2);
logits = predict(model.grade, dlarray([pooled; embG; ev], "CB"));
dEmb = dlgradient(logits(idx), embT);
end
