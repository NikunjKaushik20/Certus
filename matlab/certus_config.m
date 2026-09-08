function cfg = certus_config()
%CERTUS_CONFIG Single source of truth for paths and hyperparameters.
%   Learning rates are NOT set here: they come from runs/lr_finder.json
%   (written by lr_finder.m). Batch sizes come from runs/batch_probe.json
%   (written by smoke_test.m) when present.

% Derived from this file's own location, not hard-coded: the repo is cloned to a different
% drive on every machine it has run on, and a wrong root fails as "no manifests" rather
% than as a path error.
root = string(strrep(fileparts(fileparts(mfilename("fullpath"))), "\\", "/"));
cfg.paths.root      = root;
cfg.paths.manifests = root + "/Data/manifests";
cfg.paths.runs      = root + "/runs";
cfg.paths.lrFile    = cfg.paths.runs + "/lr_finder.json";
cfg.paths.probeFile = cfg.paths.runs + "/batch_probe.json";
% Inference-side: the shipped PyTorch checkpoint exported for MATLAB, and the photographs
% the live demo uploads. Training does not touch either.
cfg.paths.torchRun  = root + "/runs_torch/train_20260912_211356";
cfg.paths.onnx      = cfg.paths.torchRun + "/onnx";
cfg.paths.demo      = root + "/demo_images";

% ---------------------------------------------------------------- geometry
cfg.canvas     = 1536;   % FOV diameter of preprocessed images (px)
cfg.tile       = 512;    % encoder input; canvas is cut into grid x grid tiles
cfg.grid       = 3;
cfg.globalSize = 512;    % whole-eye downscaled view (context + quality head)
cfg.segClasses    = ["MA" "HE" "EX" "SE" "OD" "VES"];
cfg.lesionClasses = ["MA" "HE" "EX" "SE"];   % first 4 seg channels feed the evidence vector
cfg.numGrades  = 5;

% ---------------------------------------------------------------- model
cfg.backbone        = "efficientnetb0";
cfg.decoderChannels = [256 128 64 32];   % decoder stages at strides 16, 8, 4, 2
cfg.freezeBN        = true;   % keep ImageNet BN statistics: grade batches are 10 correlated tiles
cfg.attnDim         = 128;
cfg.headHidden      = 256;
cfg.dropout         = 0.3;
cfg.evidenceDim     = 12;     % see evidenceFeatures.m

% ---------------------------------------------------------------- batches (per optimiser step)
cfg.batch.grade   = 1;   % images per grade sub-batch (each = grid^2 tiles + 1 global view)
cfg.accum.grade   = 4;   % grade sub-batches accumulated per step
cfg.batch.seg     = 4;   % 512px lesion/vessel patches
cfg.batch.quality = 8;   % 512px global views

% ---------------------------------------------------------------- sampling
cfg.sampling.gradePower     = 0.5;  % grade weights ∝ freq^-0.5 (softened class balancing)
cfg.sampling.datasetPower   = 0.5;  % same for source dataset (domain balance)
cfg.sampling.lesionCentred  = 0.7;  % seg patches centred on a lesion pixel
cfg.sampling.lesionPriority = [3 2 1 1];      % MA HE EX SE: rare/tiny first
cfg.sampling.partialQuality = 0.25; % share of quality batch from graded images ("not reject")

% ---------------------------------------------------------------- optimisation
cfg.opt.lr          = NaN;   % peak LR for new layers   (from lr_finder.json)
cfg.opt.encLRRatio  = NaN;   % pretrained-encoder LR = lr * ratio (from lr_finder.json)
cfg.opt.weightDecay = 1e-4;
cfg.opt.beta1 = 0.9;  cfg.opt.beta2 = 0.999;  cfg.opt.eps = 1e-8;
cfg.opt.warmupFrac  = 0.03;
cfg.opt.minLRFrac   = 0.01;
cfg.opt.gradClip    = 5;
cfg.opt.emaDecay    = 0.999;
cfg.opt.epochs      = 12;
cfg.loss.referableWeight = 2;     % extra weight on the "grade >= 2" ordinal threshold
cfg.loss.tverskyAlpha = 0.3;      % FP weight
cfg.loss.tverskyBeta  = 0.7;      % FN weight: favour recall of tiny lesions
cfg.loss.focalGamma   = 2;

% ---------------------------------------------------------------- LR range test
cfg.lrfinder.steps  = 200;
cfg.lrfinder.lrMin  = 1e-6;
cfg.lrfinder.lrMax  = 1;
cfg.lrfinder.ratios = [0.1 0.3 1];   % encoder/head LR ratios compared
cfg.lrfinder.smooth = 0.95;
cfg.lrfinder.divergeFactor = 4;

% ---------------------------------------------------------------- GPU augmentation
cfg.aug.rotateDeg  = 15;
cfg.aug.brightness = 0.2;  cfg.aug.contrast = 0.2;  cfg.aug.gamma = 0.3;  cfg.aug.color = 0.05;
cfg.aug.blurProb   = 0.25; cfg.aug.blurSigma = [0.5 2.5];   % defocus (portable cameras)
cfg.aug.vignetteProb = 0.3; cfg.aug.vignetteMax = 0.5;      % uneven illumination
cfg.aug.noiseProb  = 0.2;  cfg.aug.noiseSigma = 0.03;

% ---------------------------------------------------------------- runtime
cfg.prefetchDepth = 6;
cfg.valEvery      = 1000;   % steps
cfg.valSubset     = 800;    % images per quick validation
cfg.seed          = 42;

% ---------------------------------------------------------------- measured values
if isfile(cfg.paths.lrFile)
    s = jsondecode(fileread(cfg.paths.lrFile));
    cfg.opt.lr = s.lr;
    cfg.opt.encLRRatio = s.encLRRatio;
end
if isfile(cfg.paths.probeFile)
    p = jsondecode(fileread(cfg.paths.probeFile));
    cfg.batch.grade = p.grade;  cfg.batch.seg = p.seg;  cfg.batch.quality = p.quality;
    cfg.accum.grade = p.accumGrade;
end
end
