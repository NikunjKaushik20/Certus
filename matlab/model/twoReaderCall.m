function [decision, why] = twoReaderCall(model, call, z, band2)
%TWOREADERCALL Let an automatic Certus call stand only if the second reader agrees.
%   call: "referable" or "non-referable" from Certus's screening band. z: secondReader score.
%   band2 (optional): a per-camera second-reader band with clear_below_logit / refer_above_logit.
%   Without a second reader the call stands; with one, a missing score or a score on the other
%   side of (or inside) its band sends the eye to a human, as Engine.decide does.
decision = call;
why = "";
if ~isfield(model, "second"), return, end
if nargin < 4 || isempty(band2)
    band2 = model.meta.second_reader;
end
if isempty(z)
    agrees = false;
elseif call == "referable"
    agrees = z > band2.refer_above_logit;
else
    agrees = z <= band2.clear_below_logit;
end
if ~agrees
    decision = "refer-to-human";
    why = "readers disagree";
end
end
