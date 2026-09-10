function out = deidentifyDicom(src, dst, opts)
%DEIDENTIFYDICOM Remove patient identifiers from a DICOM fundus image before it leaves the site.
%
%   out = deidentifyDicom(src)                 % writes <src>_deid.dcm next to the source
%   out = deidentifyDicom(src, dst)
%   out = deidentifyDicom(src, dst, Keep=["PatientAge" "PatientSex"])
%
%   A DICOM file carries identity in the header, not in the pixels: patient name, patient ID,
%   birth date, accession number, referring physician, institution, and the device's own serial
%   number and station name. Stripping the image metadata is not enough and is a different job --
%   api/certus_api/deident.py does that for JPEG and PNG uploads, losslessly, and deliberately
%   refuses DICOM rather than storing a header it cannot clean.
%
%   This uses dicomanon (Medical Imaging Toolbox), which implements the confidentiality profile
%   in DICOM PS3.15 Annex E rather than a hand-written list of tags to delete. That distinction
%   matters: the standard covers tags most people would not think of, and a screening programme
%   that has to answer for a privacy breach is on far firmer ground pointing at a standard than
%   at somebody's blocklist.
%
%   What is deliberately kept: age, sex, and the acquisition parameters a grader may need, when
%   asked for by name through Keep. Age and sex are not identifiers on their own and they are
%   clinically relevant to diabetic retinopathy risk. Nothing is kept by default.
%
%   Returns a struct: the output path, and which identifying tags were present before and after,
%   so the removal can be shown rather than asserted.

arguments
    src (1,1) string
    dst (1,1) string = ""
    opts.Keep (1,:) string = string.empty
end

assert(isfile(src), "deidentifyDicom: no such file: %s", src);
assert(~isempty(which("dicomanon")), ...
       "deidentifyDicom: needs the Medical Imaging Toolbox (dicomanon not on the path).");

if dst == ""
    [p, n] = fileparts(src);
    dst = fullfile(p, n + "_deid.dcm");
end

% Tags worth reporting on. dicomanon removes far more than this; these are the ones a reviewer
% will actually look for when asking whether the file is safe to transmit.
watch = ["PatientName" "PatientID" "PatientBirthDate" "OtherPatientIDs" "AccessionNumber" ...
         "ReferringPhysicianName" "InstitutionName" "InstitutionAddress" "StationName" ...
         "DeviceSerialNumber" "StudyDate" "StudyTime" "AcquisitionDate" "ContentDate" ...
         "OperatorName"];

before = presentTags(dicominfo(src), watch);

% dicomanon alone leaves the study date and time in place. Measured on a test header: it
% removed name, id, birth date, accession number and referring physician, and kept
% StudyDate 20260914 and StudyTime 094500. On its own a date is not an identifier; combined
% with a camp that ran on one day in one village it is close to one, and the whole point of
% screening-camp data is that those two facts travel together. Blank them unless the caller
% asks to keep them, and blank the acquisition dates that shadow them.
dateTags = ["StudyDate" "StudyTime" "SeriesDate" "SeriesTime" "AcquisitionDate" ...
            "AcquisitionTime" "ContentDate" "ContentTime"];
blank = setdiff(dateTags, opts.Keep);
update = struct();
for k = 1:numel(blank)
    update.(blank(k)) = '';
end

args = {};
if ~isempty(opts.Keep), args = [args, {"keep", cellstr(opts.Keep)}]; end
if ~isempty(fieldnames(update)), args = [args, {"update", update}]; end
dicomanon(src, dst, args{:});

after = presentTags(dicominfo(dst), watch);

out = struct("file", dst, "removed", setdiff(before, after), ...
             "remaining", intersect(before, after), "kept_by_request", opts.Keep);

fprintf("de-identified -> %s\n", dst);
fprintf("  removed  : %s\n", joinOrNone(out.removed));
fprintf("  remaining: %s\n", joinOrNone(out.remaining));
end


function present = presentTags(info, watch)
%PRESENTTAGS Which of the watched tags actually carry a value in this header.
present = string.empty;
for k = 1:numel(watch)
    if isfield(info, watch(k))
        v = info.(watch(k));
        if isstruct(v)
            v = strjoin(string(struct2cell(v)), "");    % PersonName is a struct of name parts
        end
        if ~isempty(v) && strlength(strtrim(string(v))) > 0
            present(end+1) = watch(k); %#ok<AGROW>
        end
    end
end
end


function s = joinOrNone(x)
if isempty(x), s = "(none)"; else, s = strjoin(x, ", "); end
end
