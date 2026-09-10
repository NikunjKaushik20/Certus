function test_deidentifyDicom()
%TEST_DEIDENTIFYDICOM Prove the de-identification rather than assert it.
%
%   Builds a DICOM fundus image carrying the identifiers a real one would -- patient name, ID,
%   birth date, accession number, referring physician, study date -- runs deidentifyDicom over
%   it, and prints the header before and after. Nothing is mocked: dicomwrite produces a real
%   file and dicominfo reads back what actually survived.
%
%   Run:  cd D:/Certus/matlab; startup_certus; test_deidentifyDicom

here = fileparts(mfilename("fullpath"));
addpath(here);
src = fullfile(tempdir, "certus_deid_test.dcm");

img = imresize(imread(fullfile(fileparts(here), "demo_images", ...
                               "grade3_severe_IDRiD_006.jpg")), [1024 NaN]);

% Char, not string: DICOM attribute values are character data and dicomwrite rejects a string
% scalar with "wrong data type" rather than converting it.
meta = struct();
meta.PatientName            = struct('FamilyName', 'Devi', 'GivenName', 'Sunita');
meta.PatientID              = 'UP-SITAPUR-88213';
meta.PatientBirthDate       = '19671104';
meta.PatientSex             = 'F';
meta.PatientAge             = '058Y';
meta.AccessionNumber        = 'ACC-2026-004412';
meta.ReferringPhysicianName = struct('FamilyName', 'Rao', 'GivenName', 'A');
meta.StudyDate              = '20260914';
meta.StudyTime              = '094500';
meta.Modality               = 'OP';                  % ophthalmic photography
dicomwrite(img, src, meta);

WATCH = ["PatientName" "PatientID" "PatientBirthDate" "AccessionNumber" ...
         "ReferringPhysicianName" "StudyDate" "StudyTime" "PatientAge" "PatientSex"];

info = dicominfo(src);
fprintf("\n-- header as written --\n");
report(info, WATCH);

% Age and sex are kept on purpose: neither identifies a person on its own, and both are
% clinically relevant to diabetic retinopathy risk.
out = deidentifyDicom(string(src), "", Keep=["PatientAge" "PatientSex"]);

fprintf("\n-- header after de-identification --\n");
after = dicominfo(out.file);
report(after, WATCH);

leaked = string.empty;
for f = setdiff(WATCH, ["PatientAge" "PatientSex"])
    if ~strcmp(tagValue(after, f), tagValue(info, f))
        continue                                   % changed, which is what we wanted
    end
    if ~ismember(tagValue(after, f), ["(absent)" "(empty)"])
        leaked(end+1) = f; %#ok<AGROW>
    end
end

kept = all(arrayfun(@(f) strcmp(tagValue(after, f), tagValue(info, f)), ...
                    ["PatientAge" "PatientSex"]));
pixels = isequal(dicomread(src), dicomread(out.file));

fprintf("\nidentifiers still present : %s\n", ternary(isempty(leaked), "none", strjoin(leaked, ", ")));
fprintf("age and sex preserved     : %d\n", kept);
fprintf("pixels unchanged          : %d\n", pixels);
assert(isempty(leaked), "de-identification left identifiers in the header: %s", strjoin(leaked, ", "));
assert(kept, "requested tags were not preserved");
assert(pixels, "pixel data changed, which de-identification must never do");
fprintf("\nPASS\n");
delete(src); delete(out.file);
end


function report(info, watch)
for f = watch
    fprintf("  %-24s : %s\n", f, tagValue(info, f));
end
end


function s = tagValue(info, f)
if ~isfield(info, f), s = "(absent)"; return; end
v = info.(f);
if isstruct(v), v = strjoin(string(struct2cell(v)), " "); end
s = strtrim(string(v));
if strlength(s) == 0, s = "(empty)"; end
end


function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
