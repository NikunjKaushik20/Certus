function nv = neovascularizationDetect(vesProb, struc, drGrade, heCount, preretinalCount, vesDensityPct)
%NEOVASCULARIZATIONDETECT Clinical screening assessment for Neovascularization (NV).
%   Evaluates presence and risk of Proliferative Diabetic Retinopathy (PDR) hallmarks:
%   - Neovascularization at the Disc (NVD): fine, frond-like vessel proliferation
%     on or within 1.5 disc diameters of the optic disc margin.
%   - Neovascularization Elsewhere (NVE): irregular new vessels originating away
%     from the disc, accompanied by preretinal/vitreous hemorrhages.
%
%   vesProb:          DxD double/single in [0, 1]
%   struc:            struct with disc_found, disc_x, disc_y, disc_diameter_px
%   drGrade:          scalar integer 0..4
%   heCount:          total number of hemorrhages
%   preretinalCount:  count of large preretinal/vitreous hemorrhages
%   vesDensityPct:    vessel pixels as % of the FOV (same rule as inference.py)
%
%   nv:               struct with nvd_detected, nve_detected, nv_risk_score, criteria
%
%   EXPERIMENTAL. None of the datasets we train or test on carries neovascularization
%   annotations, so the density thresholds below are hand-set and unvalidated. The output is
%   shown as an indicator for the reviewer, never used in the grade or the referral decision.
%   drGrade is accepted for interface stability and not used.

arguments
    vesProb (:, :) {mustBeNumeric}
    struc struct
    drGrade (1,1) double
    heCount (1,1) double = 0
    preretinalCount (1,1) double = 0
    vesDensityPct (1,1) double = 0
end

if isa(vesProb, "gpuArray"), vesProb = gather(vesProb); end
vesProb = double(vesProb);
[D, ~] = size(vesProb);

nvd = false;
nve = false;
crit = strings(0, 1);

vesMask = vesProb > 0.5;

if isfield(struc, "disc_found") && struc.disc_found
    ox = struc.disc_x;
    oy = struc.disc_y;
    dd = struc.disc_diameter_px;
    
    [X, Y] = meshgrid(1:D, 1:D);
    distDisc = hypot(X - ox, Y - oy);
    peridiscMask = distDisc <= (1.5 * dd);
    
    periVes = sum(vesMask(peridiscMask));
    periArea = sum(peridiscMask(:));
    peridiscDensity = periVes / max(periArea, 1);
    
    % NVD: abnormal peridisc microvascular proliferation independent of predicted grade
    if (peridiscDensity >= 0.15) || (peridiscDensity >= 0.11 && heCount >= 4)
        nvd = true;
        crit(end+1) = "Abnormal microvascular proliferation at optic disc margin (NVD)";
    end
end

% NVE: preretinal/vitreous hemorrhage (hallmark of neovascular bleeding) or extensive peripheral branching
if preretinalCount > 0 || (heCount >= 20 && vesDensityPct > 4.5)
    nve = true;
    crit(end+1) = "Fine irregular branching / preretinal-vitreous hemorrhage elsewhere (NVE)";
end

if isempty(crit)
    crit(end+1) = "Normal vascular caliber without neovascular fronds";
end

riskScore = min(1.0, max(0.0, ...
    (0.60 * double(nvd)) + (0.40 * double(nve)) + ...
    (0.25 * double(preretinalCount > 0)) + min(0.25, heCount * 0.008)));

nv = struct();
nv.nvd_detected = nvd;
nv.nve_detected = nve;
nv.nv_risk_score = round(riskScore, 3);
nv.criteria = "[experimental] " + strjoin(crit, "; ");
nv.experimental = true;
end
