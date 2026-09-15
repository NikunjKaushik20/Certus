function r = certus_metrics(simOut, p)
%CERTUS_METRICS Headline numbers for one simulated camp-day of certus_screening.slx.
%   Shared by run_certus_screening.m (scenarios) and run_certus_optimise.m
%   (resource search and Monte Carlo), so both report the same quantities.
    dt = p.dt_min;
    r.dt = dt;
    r.arrival      = col(simOut.get('Log_arrival'));
    r.power        = col(simOut.get('Log_power'));
    r.battery      = col(simOut.get('Log_battery'));
    r.ophth_queue  = col(simOut.get('Log_ophth_queue'));
    r.ophth_served = col(simOut.get('Log_ophth_served'));
    r.ophth_util   = col(simOut.get('Log_ophth_util'));
    r.blended_time = col(simOut.get('Log_blended_time'));
    r.lost_consent = col(simOut.get('Log_lost_consent'));
    r.lost_quality = col(simOut.get('Log_lost_quality'));
    r.screened     = col(simOut.get('Log_screened'));
    r.lost_upload  = col(simOut.get('Log_lost_upload'));
    r.captured     = col(simOut.get('Log_captured'));
    r.link_queue   = col(simOut.get('Log_link_queue'));
    n = numel(r.arrival);
    r.t = (0:n-1)' * dt;

    % --- headline metrics ---
    % Staff work to the close of the session plus a fixed grace period. Anyone
    % the pipeline has not reached by then is counted lost, not screened late:
    % the queues themselves never abandon, so without this cap the model will
    % happily screen patients at 10pm and report a mid-session power cut as
    % costing the camp nothing.
    cum_screened = cumsum(r.screened) * dt;
    t_close  = p.session.end_min;
    t_stop   = t_close + p.session.overtime_grace_min;
    i_close  = find(r.t <= t_close, 1, 'last');
    i_stop   = find(r.t <= t_stop,  1, 'last');

    % Store-and-forward: a patient is done with the camp once photographed.
    % Anyone not photographed by the time staff leave goes home unscreened --
    % the capture queue itself never abandons, so what it would have served
    % after the grace cut-off is the tail that gets counted as lost.
    cum_captured = cumsum(r.captured) * dt;
    r.photographed_day      = cum_captured(i_stop);
    r.lost_session_day      = cum_captured(end) - cum_captured(i_stop);
    % Results delivered: reports uploaded by the end of the simulated day
    % (review and upload carry on after the camp closes).
    % The model keeps serving the capture queue after staff leave; those
    % patients were counted as sent home above, so their share of reports
    % is taken back out here.
    kept = r.photographed_day / max(cum_captured(end), 1e-9);
    r.patients_screened_day = cum_screened(end) * kept;
    r.screened_after_close  = cum_screened(end) - cum_screened(i_close);
    % When the last referred/abstained eye cleared review: the image link
    % queue and the ophthalmologist queue both below half a patient.
    busy = find(r.link_queue + r.ophth_queue > 0.5, 1, 'last');
    if isempty(busy)
        r.review_done_min = t_close;
    else
        r.review_done_min = r.t(busy) + dt;
    end
    r.review_backlog_end = r.link_queue(end) + r.ophth_queue(end);
    r.link_queue_max = max(r.link_queue);

    % Overtime: time past session close that the queues were still moving.
    % Guard against empty find() when zero patients were screened (e.g. a
    % total-outage parameter test), which previously crashed with index error.
    moving = find(r.screened > 1e-9, 1, 'last');
    if isempty(moving)
        r.overtime_min = 0;
    else
        r.overtime_min = max(0, min(r.t(moving), t_stop) - t_close);
    end
    r.lost_consent_day = sum(r.lost_consent) * dt;
    r.lost_quality_day = sum(r.lost_quality) * dt;
    r.lost_upload_day  = sum(r.lost_upload) * dt;

    capacity_rate = p.ophth.n_doctors * r.power ./ max(r.blended_time,1e-6);
    wait_min = r.ophth_queue ./ max(capacity_rate,1e-6);
    wait_min(capacity_rate < 1e-6) = NaN; % offline: undefined, excluded from stats
    r.ophth_wait_mean_min = mean(wait_min,'omitnan');
    r.ophth_wait_p95_min  = prctile(wait_min(~isnan(wait_min)),95);

    % utilisation only meaningful while the camp is powered and has any
    % demand; average over the whole run (idle time correctly pulls it down)
    r.ophth_util_mean = mean(r.ophth_util(1:i_stop));
    r.ophth_util_mean_full = mean(r.ophth_util);

    % ophthalmologist-hours needed per 100,000 patients/year:
    % busy doctor-minutes this camp-day = sum(util(t) * n_doctors * dt)
    % Whole simulated day: with store-and-forward, review carries on after the
    % camp closes, and that is real doctor time.
    busy_doctor_min_day = sum(r.ophth_util) * p.ophth.n_doctors * dt;
    busy_doctor_hours_day = busy_doctor_min_day / 60;
    busy_doctor_hours_year = busy_doctor_hours_day * p.camp_days_per_year;
    patients_year = p.patients_per_day_target * p.camp_days_per_year;
    r.ophth_hours_per_100k_per_year = busy_doctor_hours_year * (100000/patients_year);
end

function v = col(x)
    v = double(x(:));
end
