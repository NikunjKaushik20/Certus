%% RUN_CERTUS_SCREENING
% Runs certus_screening.slx across scenarios and reports the headline
% capacity-planning numbers for the Certus district DR-screening
% programme (SIH 2026 PS 26038).
%
% Scenarios:
%   1. base            - measured in-distribution abstain rate, no outage
%   2. unseen_camera   - measured external (Messidor-2) abstain rate
%   3. unseen_cam_recal - band re-fitted on 100 labelled patients from the
%                        new camera (new_camera.json)
%   4. power_outage    - a 5h midday grid outage (exceeds battery runtime)
%   5. link_3G / link_2G - the image uplink downgraded from the 4G base case
%   6. staffing_sweep  - n_doctors swept 1..6 under the base case
%
% Store-and-forward: once both eyes are photographed the patient can leave;
% the review and the report can finish later. "Sent home" therefore counts
% only patients never photographed before staff leave (close + grace). The
% review side has until p.target.review_by_min (21:00) for a same-day result.
% The resource optimiser and the 250-day Monte Carlo are in
% run_certus_optimise.m.
%
% Run with: matlab -batch "run_certus_screening"

clear; clc; close all;
mdl = 'certus_screening';
here = fileparts(mfilename('fullpath'));
cd(here);
if ~bdIsLoaded(mdl)
    load_system(fullfile(here,[mdl '.slx']));
end

scenarios = struct('name',{},'p',{});

p_base = certus_params();
scenarios(1) = struct('name','base_case','p',p_base);

p_unseen = certus_params();
p_unseen.band.abstain_rate = p_unseen.band.abstain_rate_unseen_camera;
scenarios(2) = struct('name','unseen_camera','p',p_unseen);

p_recal = certus_params();
p_recal.band.abstain_rate = p_recal.band.abstain_rate_recalibrated;
scenarios(end+1) = struct('name','unseen_cam_recal','p',p_recal);

p_outage = certus_params();
p_outage.power.outage_start_min = 180; % 11:00, mid-session
p_outage.power.outage_dur_min   = 300; % 5h outage - exceeds the 4h
                                        % battery_capacity_h, so the camp
                                        % should visibly go OFFLINE.
scenarios(end+1) = struct('name','power_outage','p',p_outage);

p_3g = certus_params(); p_3g.link.tier = 2;
scenarios(end+1) = struct('name','link_3G','p',p_3g);
p_2g = certus_params(); p_2g.link.tier = 1;
scenarios(end+1) = struct('name','link_2G','p',p_2g);

results = struct();

for i = 1:numel(scenarios)
    sc = scenarios(i);
    fprintf('\n=== Running scenario: %s ===\n', sc.name);
    results.(sc.name) = run_one(mdl, sc.p);
end

%% ---- Staffing sweep (headline: doctors needed) ----------------------------
fprintf('\n=== Running staffing sweep (base-case abstain rate) ===\n');
n_doc_list = 1:6;
sweep_util = nan(size(n_doc_list));
sweep_q95  = nan(size(n_doc_list));
sweep_qmean = nan(size(n_doc_list));
sweep_screened = nan(size(n_doc_list));
for k = 1:numel(n_doc_list)
    ps = certus_params();
    ps.ophth.n_doctors = n_doc_list(k);
    r = run_one(mdl, ps, true);
    sweep_util(k) = r.ophth_util_mean;
    sweep_q95(k)  = r.ophth_wait_p95_min;
    sweep_qmean(k) = r.ophth_wait_mean_min;
    sweep_screened(k) = r.patients_screened_day;
end

%% ---- Printed summary table -------------------------------------------------
fprintf('\n\n==================== CERTUS SCREENING CAPACITY SUMMARY ====================\n');
names = fieldnames(results);
fprintf('%-17s %8s %9s %6s %8s %6s %8s %8s %9s %8s %11s\n', ...
    'Scenario','Photo-','Results','Ophth','QWait','Over-','Lost','Lost','SentHome','Reviews','Ophth-hrs');
fprintf('%-17s %8s %9s %6s %8s %6s %8s %8s %9s %8s %11s\n', ...
    '','graphed','delivered','Util%','p95(min)','time','Consent','Quality','unphoto.','done at','/100k/yr');
for i = 1:numel(names)
    r = results.(names{i});
    fprintf('%-17s %8.1f %9.1f %5.1f%% %8.2f %6.0f %8.1f %8.1f %9.1f %8s %11.1f\n', ...
        names{i}, r.photographed_day, r.patients_screened_day, r.ophth_util_mean*100, ...
        r.ophth_wait_p95_min, r.overtime_min, ...
        r.lost_consent_day, r.lost_quality_day, r.lost_session_day, ...
        clock_str(r.review_done_min), r.ophth_hours_per_100k_per_year);
end
save(fullfile(here,'scenario_results.mat'),'results');
fprintf('=============================================================================\n');

fprintf('\n--- Staffing sweep (base case, n_doctors = 1..6) ---\n');
fprintf('%10s %10s %10s %10s %14s\n','n_doctors','Util%','QWaitMean','QWaitP95','Screened/d');
for k = 1:numel(n_doc_list)
    fprintf('%10d %9.1f%% %10.2f %10.2f %14.1f\n', n_doc_list(k), sweep_util(k)*100, ...
        sweep_qmean(k), sweep_q95(k), sweep_screened(k));
end

%% ---- Figures ---------------------------------------------------------------
f1 = figure('Name','Certus screening - base case time series','Position',[100 100 1100 700]);

r = results.base_case;
subplot(3,2,1);
plot(r.t, r.arrival,'LineWidth',1.5); grid on;
title('Patient arrival rate'); xlabel('t (min)'); ylabel('patients/min');

subplot(3,2,2);
plot(r.t, r.power,'LineWidth',1.5); hold on;
plot(r.t, r.battery,'LineWidth',1.5); grid on; ylim([-0.1 1.1]);
legend('power\_on','battery\_frac','Location','best');
title('Camp power state (Stateflow)'); xlabel('t (min)');

subplot(3,2,3);
plot(r.t, r.ophth_queue,'LineWidth',1.5); grid on;
title('Ophthalmologist review queue length'); xlabel('t (min)'); ylabel('patients waiting');

subplot(3,2,4);
plot(r.t, r.ophth_util*100,'LineWidth',1.5); grid on; ylim([0 105]);
title('Ophthalmologist utilisation'); xlabel('t (min)'); ylabel('%');

subplot(3,2,5);
plot(r.t, cumsum(r.lost_consent)*r.dt,'LineWidth',1.5); hold on;
plot(r.t, cumsum(r.lost_quality)*r.dt,'LineWidth',1.5); grid on;
legend('consent dropout (cum.)','failed recapture (cum.)','Location','best');
title('Cumulative patients lost'); xlabel('t (min)'); ylabel('patients');

subplot(3,2,6);
plot(r.t, cumsum(r.screened)*r.dt,'LineWidth',1.5); grid on;
title('Cumulative patients screened (report uploaded)'); xlabel('t (min)'); ylabel('patients');

saveas(f1, fullfile(here,'fig_base_case_timeseries.png'));

f2 = figure('Name','Certus screening - power outage scenario','Position',[100 100 900 500]);
ro = results.power_outage;
subplot(2,1,1);
plot(ro.t, ro.power,'LineWidth',1.5); hold on;
plot(ro.t, ro.battery,'LineWidth',1.5); grid on; ylim([-0.1 1.1]);
legend('power\_on','battery\_frac','Location','best');
title('Power-outage scenario: camp power state'); xlabel('t (min)');
subplot(2,1,2);
plot(ro.t, ro.ophth_queue,'LineWidth',1.5); grid on;
title('Ophthalmologist queue during/after outage'); xlabel('t (min)'); ylabel('patients waiting');
saveas(f2, fullfile(here,'fig_power_outage.png'));

f3 = figure('Name','Certus screening - staffing sweep','Position',[100 100 900 500]);
subplot(2,1,1);
plot(n_doc_list, sweep_util*100,'-o','LineWidth',1.5); grid on;
xlabel('ophthalmologists (n\_doctors)'); ylabel('utilisation %');
title('Staffing sweep: ophthalmologist utilisation');
subplot(2,1,2);
plot(n_doc_list, sweep_qmean,'-o','LineWidth',1.5); hold on;
plot(n_doc_list, sweep_q95,'-s','LineWidth',1.5); grid on;
legend('mean wait','p95 wait','Location','best');
xlabel('ophthalmologists (n\_doctors)'); ylabel('queue wait (min)');
title('Staffing sweep: review queue wait');
saveas(f3, fullfile(here,'fig_staffing_sweep.png'));

fprintf('\nFigures saved: fig_base_case_timeseries.png, fig_power_outage.png, fig_staffing_sweep.png\n');
fprintf('(in %s)\n', here);

%% =========================== LOCAL FUNCTIONS ==============================
function r = run_one(mdl, p, quiet)
    if nargin < 3; quiet = false; end
    assignin('base','p',p);
    simOut = sim(mdl,'StopTime',num2str(p.T_min),'ReturnWorkspaceOutputs','on');

    r = certus_metrics(simOut, p);

    if ~quiet
        fprintf(['  screened/day=%.1f  ophth_util=%.1f%%  qwait_p95=%.2fmin  ' ...
                 'overtime=%.0fmin  sent home=%.1f  ophth-hrs/100k/yr=%.1f\n'], ...
            r.patients_screened_day, r.ophth_util_mean*100, ...
            r.ophth_wait_p95_min, r.overtime_min, r.lost_session_day, ...
            r.ophth_hours_per_100k_per_year);
    end
end

function v = col(x)
    v = double(x(:));
end

function s = clock_str(t_min)
    % minutes after 08:00 -> "hh:mm"
    s = sprintf('%02d:%02d', floor(8 + t_min/60), round(mod(t_min, 60)));
end
