function p = certus_params()
%CERTUS_PARAMS Single source of truth for every parameter used by the
% Certus district DR-screening capacity model (certus_screening.slx).
%
% Every field below is either:
%   (a) MEASURED  - taken directly from a file in this repo, cited inline, or
%   (b) ASSUMPTION - a named default with the assumption stated inline,
%       exposed here so it can be swept from run_certus_screening.m.
%
% Units are stated on every line. Time is in MINUTES unless marked _s
% (seconds) or _h (hours). Rates are patients/minute unless stated.

p = struct();
m = measured_inputs();   % numbers read from files in this repo; see the function at the bottom
p.measured = m;

%% ---- Simulation time base --------------------------------------------
p.dt_min      = 1;    % ASSUMPTION: 1-minute fixed step. The model is a
                       % fluid/rate-based (not per-patient discrete-event)
                       % queueing approximation, so 1 min is fine relative
                       % to service times of 0.5-5 min.
p.T_min       = 780;  % ASSUMPTION: total simulated minutes per camp-day
                       % (8:00 camp open -> ~21:00), i.e. an 8 h session
                       % plus a 5 h evening buffer so the ophthalmologist
                       % review queue is given time to drain (tele-review
                       % continues after the camp itself closes).

%% ---- Camp session / arrivals ------------------------------------------
p.session.start_min = 0;    % camp opens at simulated t = 0 (= 08:00)
p.session.end_min   = 480;  % ASSUMPTION: 8-hour field-camp session
                             % (08:00-16:00), typical rural camp day.
p.session.ramp_min  = 30;   % ASSUMPTION: 30 min ramp-up/down in arrival
p.session.overtime_grace_min = 30;
    % ASSUMPTION: minutes past close that staff will stay to clear the queue
    % already in front of them. Anyone still queued after this goes home
    % unscreened. Without this cap the rate-stage queues are infinitely
    % patient, and a mid-session power cut costs the camp nothing at all --
    % an artefact of the queue model, not a property of the camp.
                             % rate at camp open/close (patients trickle
                             % in/out rather than a step function).

p.camp_days_per_year      = 250; % ASSUMPTION: ~5 screening days/week,
                                  % consistent with the ophthalmologist
                                  % "5 days/week" assumption below.
p.patients_per_day_target = 400; % DERIVED: 100,000 patients/yr (SIH 26038
                                  % problem-statement target: "screen
                                  % 100,000+ patients/year in a district")
                                  % / 250 camp-days/yr = 400 patients/day
                                  % district-wide aggregate.

%% ---- Stage 1: ABDM/DEPA consent -----------------------------------------
p.consent.n_stations   = 6;    % ASSUMPTION: parallel consent
                                % tablets/stations per camp-day.
p.consent.time_min     = 2;    % MEASURED (project doc): D:\Certus\sim\parameters.md
                                % "Consent (ABDM OTP) step: 2 min mean" (marked
                                % assumption there, carried through here).
p.consent.fail_rate    = 0.05; % SOURCE: sim/parameters.md, same line:
                                % "5% failure -> retry".
p.consent.retry_success = 0.6; % ASSUMPTION: fraction of failed consent
                                % attempts recovered on an in-session retry.
p.consent.dropout_rate = p.consent.fail_rate * (1 - p.consent.retry_success);
                                % DERIVED: ~2% of patients never complete
                                % consent and leave without being screened.

%% ---- Stage 3: image capture (2 eyes/patient) ---------------------------
p.capture.n_stations = 6; % ASSUMPTION: parallel non-mydriatic camera stations.
p.capture.time_min   = 5; % SOURCE: sim/parameters.md "Capture time per
                           % patient (both eyes): 5 min mean (lognormal);
                           % swept 3-8 min" (assumption there; both eyes
                           % included in this single per-patient time).

%% ---- Stage 4: quality gate + retake loop --------------------------------
p.quality.ungradable_rate = 0.07; % BASE CASE, task-specified 4-10% band.
    % SOURCES for the band (sim/parameters.md, all cited there):
    %   22.5% ungradable, population screening, NO quality handling
    %         (SMART India, Lancet Global Health 2022: 6,133/7,910 gradable)
    %   21%   ungradable, Thailand AI clinic deployment
    %         (Beede et al., CHI 2020, ~1,840 images)
    %   ~4%   ungradable, handheld camera + trained operator
    %         (Remidio Fundus-on-Phone primary-care study, 10/261)
    % Certus has an automated quality gate at capture time, so the base
    % case is set inside the 4-10% band the task specifies (7%), and is
    % exposed here as a tunable parameter for the sweep.
p.quality.retake_success_rate = 0.7; % ASSUMPTION: fraction of first-pass
    % ungradable patients who obtain a gradable image on an in-session
    % retake (loops back to the capture stage in the real workflow; here
    % the retake's capture-station time cost is folded into this pass
    % fraction rather than a second explicit queueing pass - see README
    % "assumptions and limitations").

%% ---- Stage 5: edge inference (on-device, no upload bottleneck) --------
p.inference.n_devices       = 4;   % ASSUMPTION: edge inference
                                    % devices (laptop/edge-GPU) per camp.
p.inference.eyes_per_patient = 2;  % 2 eyes/patient (task spec).
p.inference.time_per_eye_s  = m.infer_s_per_eye; % MEASURED. SOURCE:
    % reports/agreement/matlab.csv (certus_demo, full MATLAB pipeline incl.
    % Grad-CAM, RTX 3050 laptop GPU), mean of the six demo eyes excluding the
    % first (warm-up). Falls back to 3.5 s if that file is missing.

%% ---- Stage 6: screening-band decision (auto-clear/refer/abstain) ------
% With a second reader deployed (trust.json -> second_reader), an eye is auto-decided only when
% Certus and the whole-image grader agree, so more eyes reach a person. The three rates below are
% then the two-reader ones from second_reader.json (test 20.1%, Messidor-2 45.9%, re-fitted on 100
% patients 26.9%); the Certus-alone rates are kept in p.band.single_reader for comparison.
p.band.two_readers = m.two_readers;
p.band.single_reader = m.single;
p.band.abstain_rate = m.abstain_test; % MEASURED. SOURCE:
    % D:\Certus\runs_torch\train_20260912_211356\trust.json ->
    % screening_band.measured.test.abstain_rate = 0.11749057858567946
    % (n=4511, in-distribution test set). BASE CASE.
p.band.abstain_rate_unseen_camera = m.abstain_unseen; % MEASURED. SOURCE: same file
    % -> screening_band.measured.external_messidor2.abstain_rate =
    % 0.26436781609195403 (n=1740). Used for the "unseen camera" scenario.
p.band.abstain_rate_recalibrated = m.abstain_recal; % MEASURED. SOURCE:
    % runs_torch/train_20260912_211356/new_camera.json -> new_camera_messidor2
    % .by_patients."100".band_refit.abstain_rate.mean (band re-fitted on 100
    % labelled Messidor-2 patients, scored on the rest, 1000 draws).
p.band.refer_given_not_abstain = 0.08; % ASSUMPTION: fraction of
    % non-abstained patients auto-referred. Set between vision-threatening
    % DR prevalence (4.0%) and any-DR prevalence (12.5%), matching the
    % "Referable DR share used in the model: 8%" assumption in
    % sim/parameters.md (itself sourced from SMART India / Lancet Glob
    % Health 2022 prevalence figures).

%% ---- Stage 7: ophthalmologist review queue (the scarce resource) ------
p.ophth.n_doctors      = 3;   % ASSUMPTION: ophthalmologists/trained
                               % graders staffing tele-review per camp-day;
                               % this is the headline staffing sweep variable.
p.ophth.time_refer_min  = 0.5; % 30 s. SOURCE: sim/parameters.md
    % "Ophthalmologist time per case with Certus report: 30 s
    % (problem-statement target, <30s validation)" - applied to
    % auto-referred cases, where the reviewer confirms/overrides an AI call.
p.ophth.time_abstain_min = 1.5; % 90 s. SOURCE: sim/parameters.md
    % "Ophthalmologist time per case, manual grading: 90 s (assumption;
    % swept 45-180s)" - applied to abstained cases, where the model
    % offered no opinion and the reviewer grades from scratch.
p.ophth.hours_per_day  = 4;   % SOURCE: sim/parameters.md "Reviewer
    % tele-review hours: 4h/day, 5 days/week (assumption - ophthalmologists
    % also run clinics)". Used only to convert simulated queue load into
    % FTE ophthalmologist-hours/year in run_certus_screening.m.
p.ophth.days_per_week  = 5;   % SOURCE: sim/parameters.md, same line.

%% ---- Stage 6b: image upload to the remote reviewer (store-and-forward) ----
% Every auto-referred or abstained patient's two canvases go over the camp's
% link before an ophthalmologist can see them. Auto-cleared patients send only
% the small report (stage 8).
p.link.mb_per_patient = m.mb_per_patient; % MEASURED. SOURCE: mean size of the
    % 1536-px JPEG canvases Certus uploads (Data/processed/*/img, q95), x2 eyes,
    % plus the Grad-CAM overlay and report (~0.15 MB, ASSUMPTION).
p.link.tiers_name  = ["2G/EDGE" "3G" "4G" "fibre/broadband"];
p.link.tiers_mbps  = [0.05 0.5 3 10];  % ASSUMPTION: sustained UPLOAD Mbit/s
    % a camp actually gets, well below advertised download rates. Swept.
p.link.tiers_cost_per_day = [30 50 100 300]; % ASSUMPTION: INR/day for the data plan / connection.
p.link.tier = 3;                       % BASE CASE: 4G
p.link.availability = 0.85;            % ASSUMPTION: fraction of the time the
    % link is up (rural towers drop; the uploader resumes on reconnect).

%% ---- Stage 8: report upload over lossy rural link ----------------------
p.upload.n_channels          = 4;    % ASSUMPTION: parallel upload workers.
p.upload.time_s              = 3;    % ASSUMPTION: small JSON/report
    % payload (kilobytes, NOT images) over a patchy 2G/3G/4G rural link.
p.upload.permanent_loss_rate = 0.02; % ASSUMPTION: fraction of reports
    % never delivered even after automatic retries on the lossy link,
    % requiring a later manual/offline sync.

%% ---- Camp power (Stateflow: grid -> battery -> depleted/offline) ------
p.power.grid_hours_per_day    = 22.6; % MEASURED (national average).
    % SOURCE: Ministry of Power / PIB press release, Feb 2025 (cited in
    % sim/parameters.md). Context only - the discrete outage events below
    % are what actually drive the power-outage scenario.
p.power.outage_events_per_day = 1;    % MEASURED (context). SOURCE:
    % CEEW / Prayas rural electricity-supply surveys (cited in
    % sim/parameters.md): "at least 1 outage/day for two-thirds of rural
    % households".
p.power.battery_capacity_h    = 4;    % ASSUMPTION: camp inverter/UPS
    % battery runtime at full charge, hours.
p.power.recharge_time_h       = 6;    % ASSUMPTION: hours of grid
    % availability to fully recharge the battery from empty.
p.power.recharge_rate_per_min = 1 / (p.power.recharge_time_h * 60); % DERIVED
p.power.drain_rate_per_min    = 1 / (p.power.battery_capacity_h * 60); % DERIVED
p.power.outage_start_min      = 1e6;  % BASE CASE: no outage in-session
    % (sentinel far beyond T_min). Overridden by the power-outage scenario
    % in run_certus_screening.m.
p.power.outage_dur_min        = 0;    % BASE CASE: zero-length outage.

%% ---- Costs, for the resource optimiser (run_certus_optimise.m) ---------
% All ASSUMPTIONS, INR per camp-day, stated so they can be replaced with a
% district's real figures. Only relative sizes matter for the optimum.
p.cost.capture_station = 1500; % technician day wage (~800) + non-mydriatic
    % camera amortised over 5 yrs x 250 days (~400) + consumables/transport.
p.cost.consent_station = 400;  % tablet + assistant.
p.cost.edge_device     = 150;  % laptop with a small GPU amortised over 3 yrs.
p.cost.doctor_hour     = 1000; % tele-review, paid for the whole review window.
p.ophth.review_window_h = 4;   % hours each rostered reviewer is paid for (matches hours_per_day).

%% ---- Service targets the optimiser must meet ---------------------------
p.target.sent_home_frac   = 0.01; % at most 1% of the day's patients leave unphotographed
p.target.review_by_min    = 780;  % every referred/abstained eye reviewed by 21:00 (same-day result)
end


function m = measured_inputs()
%MEASURED_INPUTS Numbers this model takes from the rest of the repo rather than from assumptions.
%   Every value has a fallback (the figure it had when last read) so the model still runs
%   from a checkout without the dataset or the run folder.
repo = fileparts(fileparts(fileparts(mfilename("fullpath"))));
run  = fullfile(repo, "runs_torch", "train_20260912_211356");
m = struct("abstain_test", 0.1175, "abstain_unseen", 0.2644, "abstain_recal", 0.129, ...
           "infer_s_per_eye", 3.5, "mb_per_patient", 1.25, "sources", strings(0, 1), ...
           "two_readers", false, "single", struct());

f = fullfile(run, "trust.json");
if isfile(f)
    t = jsondecode(fileread(f));
    m.abstain_test   = t.screening_band.measured.test.abstain_rate;
    m.abstain_unseen = t.screening_band.measured.external_messidor2.abstain_rate;
    m.sources(end+1) = f;
end
f = fullfile(run, "new_camera.json");
if isfile(f)
    t = jsondecode(fileread(f));
    m.abstain_recal = t.new_camera_messidor2.by_patients.x100.band_refit.abstain_rate.mean;
    m.sources(end+1) = f;
end
m.single = struct("test", m.abstain_test, "unseen", m.abstain_unseen, "recal", m.abstain_recal);
% Second reader: if trust.json names one, the deployed rule is agree-or-human, and the rates that
% reach a person are the two-reader ones (torch/second_reader.py).
f = fullfile(run, "trust.json");
if isfile(f)
    t = jsondecode(fileread(f));
    if isfield(t, "second_reader")
        g = fullfile(repo, fileparts(t.second_reader.checkpoint), "second_reader.json");
        if isfile(g)
            s2 = jsondecode(fileread(g));
            m.abstain_test   = s2.fixed.test.twoReaders.to_human;
            m.abstain_unseen = s2.fixed.external_messidor2.twoReaders.to_human;
            m.abstain_recal  = s2.new_camera_messidor2.x100.twoReaders.to_human.mean;
            m.two_readers = true;
            m.sources(end+1) = g;
        end
    end
end
f = fullfile(repo, "reports", "agreement", "matlab.csv");
if isfile(f)
    T = readtable(f);
    if height(T) > 1
        m.infer_s_per_eye = mean(T.ms(2:end)) / 1000;
        m.sources(end+1) = f;
    end
end
d = dir(fullfile(repo, "Data", "processed", "*", "img", "*.jpg"));
if numel(d) > 100
    d = d(round(linspace(1, numel(d), 2000)));
    m.mb_per_patient = 2 * mean([d.bytes]) / 1e6 + 0.15;
    m.sources(end+1) = fullfile(repo, "Data", "processed");
end

end
