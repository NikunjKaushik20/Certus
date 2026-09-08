function cfg = startup_certus()
%STARTUP_CERTUS Add Certus folders to the path, seed RNGs, select the GPU.
here = fileparts(mfilename("fullpath"));
addpath(here, fullfile(here, "data"), fullfile(here, "model"), fullfile(here, "train"), ...
        fullfile(here, "eval"), fullfile(here, "run"));
cfg = certus_config();
if ~isfolder(cfg.paths.runs), mkdir(cfg.paths.runs); end
rng(cfg.seed, "twister");
% Training needs a GPU and each training entry point asserts that for itself. Inference
% does not: certus_demo runs on the CPU in a few seconds, and the demo laptop may have no
% NVIDIA card at all, so a missing GPU must not stop the path from being set up.
if canUseGPU
    gpurng(cfg.seed, "Philox");
else
    warning("Certus:noGPU", "No usable CUDA GPU: inference will run on the CPU, training cannot run.");
end
end
