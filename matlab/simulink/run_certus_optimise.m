%% RUN_CERTUS_OPTIMISE
% Resource allocation for a district programme: the cheapest camp set-up
% that still meets the service targets, and how it holds up over a year.
%
% A camp-day meets its targets when
%   - at most p.target.sent_home_frac of the day's patients leave
%     unphotographed, and
%   - every referred/abstained eye is uploaded and reviewed by
%     p.target.review_by_min (21:00: same-day result).
% The optimum is the cheapest set-up (p.cost, all ASSUMPTIONS in
% certus_params.m) that misses a target on at most MAX_MISS_FRAC of the
% camp-days in a simulated year (a chance constraint).
%
% 1. Screen. Every combination of consent stations, camera stations, edge
%    inference devices, tele-review ophthalmologists and uplink tier is
%    simulated on one STRESS day (below). Anything failing there is dropped.
%    Exhaustive, because the space is small and discrete.
% 2. Year. Survivors, cheapest first, are each run over the same 250
%    simulated camp-days (common random numbers, so set-ups are compared on
%    identical days). The first that meets the chance constraint is the
%    optimum. Patients are lost unphotographed only upstream of capture, so
%    once a (consent, cameras) pair fails the sent-home target over the
%    year, every set-up sharing it is skipped.
% 3. The hand-picked base set-up gets the same year, for comparison.
%
% Sizing for one stress day alone was tried first and is not enough: the
% cheapest set-up passing it missed the sent-home target on 38 of 250
% simulated days.
%
% STRESS day (ASSUMPTION): 460 patients (+15%), a camera not yet calibrated
% (the measured Messidor-2 abstention rate), a 2 h outage from 11:00, link up
% 70% of the time, 10% of patients ungradable at first capture.
%
% Run with: matlab -batch "run_certus_optimise"

clear; clc; close all;
mdl = 'certus_screening';
here = fileparts(mfilename('fullpath'));
cd(here);
load_system(fullfile(here, [mdl '.slx']));
set_param(mdl, 'FastRestart', 'on');

base = certus_params();
MAX_MISS_FRAC = 0.05;

stress = base;
stress.patients_per_day_target = 460;
stress.band.abstain_rate = base.band.abstain_rate_unseen_camera;
stress.power.outage_start_min = 180;
stress.power.outage_dur_min = 120;
stress.link.availability = 0.70;
stress.quality.ungradable_rate = 0.10;

%% ---- 1. exhaustive search on the stress day ------------------------------
grid.consent = 2:6;
grid.capture = 4:9;
grid.devices = 1:4;
grid.doctors = 1:3;
grid.link    = 1:numel(base.link.tiers_mbps);
[C1, C2, C3, C4, C5] = ndgrid(grid.consent, grid.capture, grid.devices, grid.doctors, grid.link);
cand = [C1(:) C2(:) C3(:) C4(:) C5(:)];
n = size(cand, 1);
fprintf('Searching %d set-ups on the stress day...\n', n);

cost = zeros(n, 1); sent = zeros(n, 1); done = zeros(n, 1); backlog = zeros(n, 1);
tic;
for k = 1:n
    q = apply_setup(stress, cand(k, :));
    r = sim_day(mdl, q);
    cost(k) = daily_cost(q);
    sent(k) = r.lost_session_day;
    done(k) = r.review_done_min;
    backlog(k) = r.review_backlog_end;
end
fprintf('  %.0f s (%.2f s per simulation)\n', toc, toc / n);

feasible = sent <= stress.target.sent_home_frac * stress.patients_per_day_target & ...
           done <= stress.target.review_by_min & backlog < 0.5;
fprintf('  %d of %d set-ups meet both targets\n', nnz(feasible), n);
assert(any(feasible), 'No set-up in the search grid meets the targets on the stress day.');

baseSetup = [base.consent.n_stations base.capture.n_stations base.inference.n_devices ...
             base.ophth.n_doctors base.link.tier];
kb = find(all(cand == baseSetup, 2));

fprintf('\nBinding constraints on the stress day: cheapest passing set-up per camera count\n');
for c = grid.capture
    m = feasible & cand(:, 2) == c;
    if any(m)
        kk = find(m); [~, jj] = min(cost(kk));
        fprintf('  %d cameras: INR %.0f/day\n', c, cost(kk(jj)));
    else
        fprintf('  %d cameras: fails (%.1f patients sent home at best)\n', c, min(sent(cand(:, 2) == c)));
    end
end

%% ---- 2. the year, cheapest survivors first ---------------------------------------
days = base.camp_days_per_year;
rng(20260921);
draws = draw_days(base, days);
maxMiss = floor(MAX_MISS_FRAC * days);
fprintf('\nYear check: %d simulated camp-days, at most %d may miss a target\n', days, maxMiss);
idx = find(feasible);
[~, order] = sort(cost(idx));
idx = idx(order);
badPairs = zeros(0, 2);
best = [];
tried = 0;
trials = zeros(0, 4);          % [candidate index, cost, days missing sent-home, days missing review]
for k = idx'
    if ~isempty(badPairs) && any(all(badPairs == cand(k, 1:2), 2))
        continue
    end
    tried = tried + 1;
    y = run_year(mdl, base, draws, cand(k, :));
    trials(end+1, :) = [k, cost(k), y.miss_home, y.miss_review]; %#ok<AGROW>
    fprintf('  %2d. consent %d cameras %d devices %d doctors %d %-8s INR %5.0f/day: misses sent-home %3d, review %3d days\n', ...
            tried, cand(k, 1:4), base.link.tiers_name(cand(k, 5)), cost(k), y.miss_home, y.miss_review);
    if y.miss_home > maxMiss
        badPairs(end+1, :) = cand(k, 1:2); %#ok<AGROW>
    elseif y.miss_review <= maxMiss
        best = k;
        mc.optimum = y;
        break
    end
end
assert(~isempty(best), 'No set-up meets the chance constraint.');
bestSetup = cand(best, :);
mc.hand_picked = run_year(mdl, base, draws, baseSetup);

fprintf('\n%-12s %8s %8s %8s %8s %-16s %8s %12s %13s %12s\n', 'Set-up', 'consent', 'cameras', ...
        'devices', 'doctors', 'uplink', 'INR/day', 'INR lakh/yr', 'sent home/yr', 'days missed');
for s = ["optimum" "hand_picked"]
    y = mc.(s);
    fprintf('%-12s %8d %8d %8d %8d %-16s %8.0f %12.2f %13.0f %12d\n', s, y.setup(1:4), ...
            base.link.tiers_name(y.setup(5)), y.cost_per_day, y.cost_per_day * days / 1e5, ...
            y.sent_home_year, nnz(y.sent_home > base.target.sent_home_frac * [draws.patients]' | ...
                                  y.review_done_min > base.target.review_by_min));
end
fprintf('\nper year: %.0f patients; optimum photographs %.0f (%.2f%%), hand-picked %.0f (%.2f%%)\n', ...
        mc.optimum.patients_year, mc.optimum.photographed_year, ...
        100 * mc.optimum.photographed_year / mc.optimum.patients_year, ...
        mc.hand_picked.photographed_year, 100 * mc.hand_picked.photographed_year / mc.hand_picked.patients_year);
fprintf('ophthalmologist-hours spent reviewing per year (optimum): %.0f\n', mc.optimum.doctor_hours_year);

save(fullfile(here, 'optimise_results.mat'), 'cand', 'cost', 'sent', 'done', 'backlog', ...
     'feasible', 'best', 'bestSetup', 'baseSetup', 'stress', 'mc', 'draws', 'trials', 'maxMiss');

%% ---- figure ---------------------------------------------------------------------
f = figure('Name', 'Certus resource optimisation', 'Position', [100 100 1150 460], 'Color', 'w');
if exist('theme', 'file'), theme(f, 'light'); end
cams = cand(:, 2);
subplot(1, 2, 1);
hold on;
jit = @(m) 0.25 * (rand(nnz(m), 1) - 0.5);
for c = grid.capture
    m = cams == c & feasible;
    scatter(cost(m) / 1000, c + jit(m), 8, [0.2 0.45 0.8], 'filled', 'HandleVisibility', hv(c == grid.capture(1)));
    m = cams == c & ~feasible;
    scatter(cost(m) / 1000, c + jit(m), 8, [0.75 0.75 0.75], 'filled', 'HandleVisibility', hv(c == grid.capture(1)));
end
grid on; box on;
xlabel('daily cost, INR thousand (assumed)'); ylabel('camera stations');
legend('passes the stress day', 'fails the stress day', 'Location', 'southeast');
title(sprintf('Step 1: %d set-ups on one stress day', n), 'FontWeight', 'normal');

subplot(1, 2, 2);
hold on;
miss = max(trials(:, 3), trials(:, 4));
bar(1:size(trials, 1), miss, 'FaceColor', [0.55 0.65 0.8]);
iBest = find(trials(:, 1) == best);
bar(iBest, miss(iBest), 'FaceColor', [0.95 0.7 0.1]);
yline(maxMiss, 'r--', sprintf('at most %d of %d days', maxMiss, days), 'LabelHorizontalAlignment', 'left');
labels = arrayfun(@(k) sprintf('%d/%d, %.1fk', cand(k, 1), cand(k, 2), cost(k) / 1000), trials(:, 1), 'UniformOutput', false);
xticks(1:size(trials, 1)); xticklabels(labels); xtickangle(35);
grid on; box on;
xlabel('consent/cameras, INR per day (cheapest first)');
ylabel('camp-days missing a target (of 250)');
title('Step 2: year check; first to pass is the optimum', 'FontWeight', 'normal');
saveas(f, fullfile(here, 'fig_optimisation.png'));
fprintf('\nSaved optimise_results.mat and fig_optimisation.png\n');
set_param(mdl, 'FastRestart', 'off');


%% ============================ LOCAL FUNCTIONS ============================
function q = apply_setup(p, s)
    q = p;
    q.consent.n_stations = s(1);
    q.capture.n_stations = s(2);
    q.inference.n_devices = s(3);
    q.ophth.n_doctors = s(4);
    q.link.tier = s(5);
end

function y = run_year(mdl, base, draws, setup)
    days = numel(draws);
    photo = zeros(days, 1); home = zeros(days, 1); fin = zeros(days, 1); docMin = zeros(days, 1);
    for d = 1:days
        q = apply_setup(apply_draw(base, draws(d)), setup);
        r = sim_day(mdl, q);
        photo(d) = r.photographed_day;
        home(d) = r.lost_session_day;
        fin(d) = r.review_done_min;
        docMin(d) = sum(r.ophth_util) * q.ophth.n_doctors * q.dt_min;
    end
    y = struct("setup", setup, "cost_per_day", daily_cost(apply_setup(base, setup)), ...
        "patients_year", sum([draws.patients]), "photographed_year", sum(photo), ...
        "sent_home_year", sum(home), ...
        "miss_home", nnz(home > base.target.sent_home_frac * [draws.patients]'), ...
        "miss_review", nnz(fin > base.target.review_by_min), ...
        "doctor_hours_year", sum(docMin) / 60, ...
        "photographed", photo, "sent_home", home, "review_done_min", fin);
end

function c = daily_cost(p)
    c = p.consent.n_stations * p.cost.consent_station + ...
        p.capture.n_stations * p.cost.capture_station + ...
        p.inference.n_devices * p.cost.edge_device + ...
        p.ophth.n_doctors * p.ophth.review_window_h * p.cost.doctor_hour + ...
        p.link.tiers_cost_per_day(p.link.tier);
end

function r = sim_day(mdl, p)
    in = Simulink.SimulationInput(mdl);
    in = in.setVariable('p', p);
    in = in.setModelParameter('StopTime', num2str(p.T_min));
    out = sim(in, 'ShowProgress', 'off');
    r = certus_metrics(out, p);
end

function D = draw_days(p, n)
%DRAW_DAYS Day-to-day variation, all ASSUMPTIONS:
%   turnout ~ N(target, 15%); 20% of camp-days use a camera not yet
%   calibrated (Messidor-2 abstention), the rest the test-set rate; an outage
%   on 2/3 of days (CEEW/Prayas: >=1 outage/day for 2/3 of rural households),
%   starting uniformly in the session, duration exponential with mean 90 min
%   (capped at 6 h); link availability U(0.6, 0.95); first-capture
%   ungradable rate U(4%, 10%); capture time U(4, 6) min.
    D = struct('patients', {}, 'abstain', {}, 'out_start', {}, 'out_dur', {}, ...
               'link_avail', {}, 'ungradable', {}, 'capture_min', {});
    for d = 1:n
        D(d).patients = max(200, round(p.patients_per_day_target * (1 + 0.15 * randn)));
        if rand < 0.2
            D(d).abstain = p.band.abstain_rate_unseen_camera;
        else
            D(d).abstain = p.band.abstain_rate;
        end
        if rand < 2/3
            D(d).out_start = rand * p.session.end_min;
            D(d).out_dur = min(360, -90 * log(rand));
        else
            D(d).out_start = 1e6; D(d).out_dur = 0;
        end
        D(d).link_avail = 0.6 + 0.35 * rand;
        D(d).ungradable = 0.04 + 0.06 * rand;
        D(d).capture_min = 4 + 2 * rand;
    end
end

function q = apply_draw(p, d)
    q = p;
    q.patients_per_day_target = d.patients;
    q.band.abstain_rate = d.abstain;
    q.power.outage_start_min = d.out_start;
    q.power.outage_dur_min = d.out_dur;
    q.link.availability = d.link_avail;
    q.quality.ungradable_rate = d.ungradable;
    q.capture.time_min = d.capture_min;
end

function v = hv(on)
    v = 'off';
    if on, v = 'on'; end
end

function s = clock_str(t_min)
    s = sprintf('%02d:%02d', floor(8 + t_min/60), round(mod(t_min, 60)));
end
