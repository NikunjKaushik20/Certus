function z = secondReader(model, X)
%SECONDREADER Referable score from the second reader, the plain whole-image grader.
%   z = secondReader(model, X)   X: DxDx3 single canvas in [0,1] (unenhanced), z: log-odds of
%   grade >= 2 (raw cumulative logit k = 1). Returns [] when no second reader was imported.
%
%   The deployed decision only auto-refers or auto-clears an eye when Certus and this reader
%   both land outside their bands on the same side (torch/second_reader.py, twoReaderCall).
%   Resizing is bilinear with antialiasing, as torch's F.interpolate(antialias=True) did in
%   training.
z = [];
if ~isfield(model, "second"), return, end
n = model.meta.second_reader.input_size;
Xs = imresize(X, [n n], "bilinear", Antialiasing=true);
y = predict(model.second, dlarray(Xs, "SSCB"));
y = double(gather(extractdata(y)));
z = y(2);
end
