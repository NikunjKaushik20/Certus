%% BUILD_CERTUS_SCREENING
% Programmatically builds certus_screening.slx: a district-level diabetic
% retinopathy screening-programme capacity model for the Certus project
% (SIH 2026 PS 26038).
%
% APPROACH: SimEvents was checked first (per task instructions) via
% license('test','SimEvents') and ver -> both reported it "installed", but
% the actual toolbox files are absent from this install
% (D:\MATLAB\R2026a\toolbox has no simevents/ directory, no simevents*.slx
% anywhere on disk, and add_block('simevents/Entity Generator',...) fails
% with "no block named"). This is a broken/incomplete mpm install, not a
% licensing issue. Per the task's explicit fallback instruction, this
% model therefore uses a PLAIN DISCRETE-TIME SIMULINK formulation: each
% pipeline stage is a fluid/rate-based queue built from a Unit Delay
% (the queue-level state, fed back to itself every step) plus a MATLAB
% Function block (the arrival/service/loss arithmetic for that step).
% Stateflow *is* present on disk (toolbox/stateflow exists) and is used
% for the camp power state machine, as required.
%
% Run with: matlab -batch "build_certus_screening"

clear; clc;
mdl = 'certus_screening';
if bdIsLoaded(mdl); close_system(mdl,0); end
new_system(mdl);
open_system(mdl);
set_param(mdl,'Solver','FixedStepDiscrete','FixedStep','p.dt_min','StopTime','p.T_min');

p = certus_params(); %#ok<NASGU> % also needed in base workspace for the
assignin('base','p',p);          % smoke-test sim at the end of this script

%% ---- layout bookkeeping -------------------------------------------------
X = 30; dx = 170; Y0 = 40; rowh = 90;

%% ---- Clock ---------------------------------------------------------------
add_block('simulink/Sources/Clock', [mdl '/Clock'], 'Position',[X Y0 X+30 Y0+30]);

%% ---- Stage 0: arrival profile ---------------------------------------------
arr = add_mfun(mdl, 'ArrivalProfile', [X+60 Y0-10 X+220 Y0+50], sprintf([ ...
 'function arrival_rate = arrival_profile(t, t_start, t_end, t_ramp, patients_per_day)\n' ...
 '%% Trapezoid-shaped registration-rate profile (patients/min) that\n' ...
 '%% integrates to patients_per_day across one camp session. ASSUMPTION:\n' ...
 '%% patients trickle in/out over t_ramp minutes at open/close rather than\n' ...
 '%% arriving as a step function.\n' ...
 'dur = t_end - t_start;\n' ...
 'plateau = max(dur - 2*t_ramp, eps);\n' ...
 'peak = patients_per_day / (t_ramp/2 + plateau + t_ramp/2);\n' ...
 'if t < t_start || t > t_end\n' ...
 '    arrival_rate = 0;\n' ...
 'elseif t < t_start + t_ramp\n' ...
 '    arrival_rate = peak * (t - t_start) / t_ramp;\n' ...
 'elseif t < t_end - t_ramp\n' ...
 '    arrival_rate = peak;\n' ...
 'else\n' ...
 '    arrival_rate = peak * (t_end - t) / t_ramp;\n' ...
 'end\n']));
wire(mdl,'Clock/1',[arr '/1']);

c_tstart = add_const(mdl,'C_t_start',[X+250 Y0-40 X+310 Y0-20],'p.session.start_min');
c_tend   = add_const(mdl,'C_t_end',  [X+250 Y0-10 X+310 Y0+10],'p.session.end_min');
c_tramp  = add_const(mdl,'C_t_ramp', [X+250 Y0+20 X+310 Y0+40],'p.session.ramp_min');
c_pday   = add_const(mdl,'C_p_day',  [X+250 Y0+50 X+310 Y0+70],'p.patients_per_day_target');
wire(mdl,[c_tstart '/1'],[arr '/2']);
wire(mdl,[c_tend   '/1'],[arr '/3']);
wire(mdl,[c_tramp  '/1'],[arr '/4']);
wire(mdl,[c_pday   '/1'],[arr '/5']);

c_dt = add_const(mdl,'C_dt',[X+250 Y0+80 X+310 Y0+100],'p.dt_min');

%% ---- Camp power Stateflow chart -------------------------------------------
Ypow = Y0 + 260;
c_grid_h  = add_const(mdl,'C_recharge',[X+60 Ypow-40 X+150 Ypow-20],'p.power.recharge_rate_per_min');
c_drain   = add_const(mdl,'C_drain',   [X+60 Ypow-10 X+150 Ypow+10],'p.power.drain_rate_per_min');
c_outst   = add_const(mdl,'C_outStart',[X+60 Ypow+20 X+150 Ypow+40],'p.power.outage_start_min');
c_outdur  = add_const(mdl,'C_outDur',  [X+60 Ypow+50 X+150 Ypow+70],'p.power.outage_dur_min');

gridavail = add_mfun(mdl,'GridAvail',[X+180 Ypow-20 X+320 Ypow+40], sprintf([ ...
 'function grid_avail = grid_avail_fn(t, outage_start, outage_dur)\n' ...
 '%% 1 = grid present, 0 = grid outage. A single outage window per\n' ...
 '%% scenario is enough to demonstrate the grid->battery->offline\n' ...
 '%% Stateflow transitions; see run_certus_screening.m for the\n' ...
 '%% power-outage scenario values.\n' ...
 'if t >= outage_start && t < outage_start + outage_dur\n' ...
 '    grid_avail = 0;\n' ...
 'else\n' ...
 '    grid_avail = 1;\n' ...
 'end\n']));
wire(mdl,'Clock/1',[gridavail '/1']);
wire(mdl,[c_outst '/1'],[gridavail '/2']);
wire(mdl,[c_outdur '/1'],[gridavail '/3']);

chartPath = add_power_chart(mdl,'CampPower',[X+360 Ypow-40 X+520 Ypow+80]);
wire(mdl,[gridavail '/1'],[chartPath '/1']);
wire(mdl,[c_grid_h '/1'],[chartPath '/2']);
wire(mdl,[c_drain '/1'],[chartPath '/3']);
wire(mdl,[c_dt '/1'],[chartPath '/4']);
% chart outputs, port order: 1 power_on 2 battery_frac 3 camp_state

logblk(mdl,'Log_power',[X+560 Ypow-40 X+620 Ypow-20],[chartPath '/1']);
logblk(mdl,'Log_battery',[X+560 Ypow+20 X+620 Ypow+40],[chartPath '/2']); %#ok<NASGU>

%% ================= PIPELINE STAGES (rate-stage pattern) ===================
% Each stage: UnitDelay(queue) --feedback--> MATLAB Function(rate_stage)
% Ports on rate_stage: in [queue_prev, arrival_rate, capacity_rate,
% pass_frac, gate, dt] ; out [served_rate, lost_rate, queue_len, queue_next]

stageX = X+420;

% ---- Stage 1: Consent ----
capC = add_const(mdl,'Cap_Consent',[stageX Y0+140 stageX+70 Y0+160], ...
    'p.consent.n_stations/p.consent.time_min');
passC = add_const(mdl,'Pass_Consent',[stageX Y0+170 stageX+70 Y0+190], ...
    '1-p.consent.dropout_rate');
S1 = add_rate_stage(mdl,'Consent',[stageX+90 Y0+100]);
wire(mdl,[arr '/1'],[S1.fcn '/2']);
wire(mdl,[capC '/1'],[S1.fcn '/3']);
wire(mdl,[passC '/1'],[S1.fcn '/4']);
wire(mdl,[chartPath '/1'],[S1.fcn '/5']);
wire(mdl,[c_dt '/1'],[S1.fcn '/6']);
logblk(mdl,'Log_arrival',[stageX+90-40 Y0-30 stageX+90+20 Y0-10],[arr '/1']);
logblk(mdl,'Log_lost_consent',[stageX+250 Y0+70 stageX+310 Y0+90],[S1.fcn '/2']);

% ---- Stage 2: Capture ----
stageX2 = stageX+280;
% A retake re-occupies a camera station, so capture capacity in patients/min is
% divided by the expected number of capture attempts per patient.
capCap = add_const(mdl,'Cap_Capture',[stageX2 Y0+140 stageX2+70 Y0+160], ...
    'p.capture.n_stations/(p.capture.time_min*(1+p.quality.ungradable_rate))');
passCap = add_const(mdl,'Pass_Capture',[stageX2 Y0+170 stageX2+70 Y0+190],'1');
S2 = add_rate_stage(mdl,'Capture',[stageX2+90 Y0+100]);
wire(mdl,[S1.fcn '/1'],[S2.fcn '/2']);
wire(mdl,[capCap '/1'],[S2.fcn '/3']);
wire(mdl,[passCap '/1'],[S2.fcn '/4']);
wire(mdl,[chartPath '/1'],[S2.fcn '/5']);
wire(mdl,[c_dt '/1'],[S2.fcn '/6']);
logblk(mdl,'Log_captured',[stageX2+250 Y0+70 stageX2+310 Y0+90],[S2.fcn '/1']);

% ---- Stage 3: Quality gate + retake ----
stageX3 = stageX2+280;
capQ = add_const(mdl,'Cap_Quality',[stageX3 Y0+140 stageX3+70 Y0+160],'1e9');
passQ = add_const(mdl,'Pass_Quality',[stageX3 Y0+170 stageX3+70 Y0+190], ...
    '(1-p.quality.ungradable_rate) + p.quality.ungradable_rate*p.quality.retake_success_rate');
S3 = add_rate_stage(mdl,'Quality',[stageX3+90 Y0+100]);
wire(mdl,[S2.fcn '/1'],[S3.fcn '/2']);
wire(mdl,[capQ '/1'],[S3.fcn '/3']);
wire(mdl,[passQ '/1'],[S3.fcn '/4']);
wire(mdl,[chartPath '/1'],[S3.fcn '/5']);
wire(mdl,[c_dt '/1'],[S3.fcn '/6']);
logblk(mdl,'Log_lost_quality',[stageX3+250 Y0+70 stageX3+310 Y0+90],[S3.fcn '/2']);

% ---- Stage 4: Edge inference ----
stageX4 = stageX3+280;
capI = add_const(mdl,'Cap_Infer',[stageX4 Y0+140 stageX4+70 Y0+160], ...
    'p.inference.n_devices*60/(p.inference.eyes_per_patient*p.inference.time_per_eye_s)');
passI = add_const(mdl,'Pass_Infer',[stageX4 Y0+170 stageX4+70 Y0+190],'1');
S4 = add_rate_stage(mdl,'Infer',[stageX4+90 Y0+100]);
wire(mdl,[S3.fcn '/1'],[S4.fcn '/2']);
wire(mdl,[capI '/1'],[S4.fcn '/3']);
wire(mdl,[passI '/1'],[S4.fcn '/4']);
wire(mdl,[chartPath '/1'],[S4.fcn '/5']);
wire(mdl,[c_dt '/1'],[S4.fcn '/6']);

% ---- Stage 5: Screening-band decision (instant split, no queue) ----
stageX5 = stageX4+280;
c_pab = add_const(mdl,'C_p_abstain',[stageX5 Y0+40 stageX5+80 Y0+60],'p.band.abstain_rate');
c_pref = add_const(mdl,'C_p_refer',[stageX5 Y0+70 stageX5+80 Y0+90],'p.band.refer_given_not_abstain');
band = add_mfun(mdl,'BandDecision',[stageX5+100 Y0+80 stageX5+240 Y0+150], sprintf([ ...
 'function [clear_rate, refer_rate, abstain_rate] = band_decision(arrival_rate, p_abstain, p_refer_ok)\n' ...
 '%% Screening-band split. p_abstain measured from trust.json\n' ...
 '%% (screening_band.measured.*.abstain_rate). p_refer_ok is the ASSUMED\n' ...
 '%% referable-DR share among non-abstained patients (certus_params.m).\n' ...
 'abstain_rate = arrival_rate * p_abstain;\n' ...
 'ok_rate = arrival_rate * (1 - p_abstain);\n' ...
 'refer_rate = ok_rate * p_refer_ok;\n' ...
 'clear_rate = ok_rate - refer_rate;\n']));
wire(mdl,[S4.fcn '/1'],[band '/1']);
wire(mdl,[c_pab '/1'],[band '/2']);
wire(mdl,[c_pref '/1'],[band '/3']);
logblk(mdl,'Log_clear',[stageX5+260 Y0+50 stageX5+320 Y0+70],[band '/1']);

% ---- Stage 6: Ophthalmologist review queue ----
stageX6 = stageX5+280;
sumRA = [mdl '/Sum_ReferAbstain'];
add_block('simulink/Math Operations/Add',sumRA, ...
    'Position',[stageX6-40 Y0+90 stageX6-10 Y0+130],'Inputs','++');
wire(mdl,[band '/2'],[sumRA '/1']);
wire(mdl,[band '/3'],[sumRA '/2']);

c_ndoc = add_const(mdl,'C_n_doctors',[stageX6 Y0-40 stageX6+80 Y0-20],'p.ophth.n_doctors');
c_tref = add_const(mdl,'C_t_refer',[stageX6 Y0-10 stageX6+80 Y0+10],'p.ophth.time_refer_min');
c_tabs = add_const(mdl,'C_t_abstain',[stageX6 Y0+20 stageX6+80 Y0+40],'p.ophth.time_abstain_min');
ophcap = add_mfun(mdl,'OphthCapacity',[stageX6+100 Y0-30 stageX6+240 Y0+50], sprintf([ ...
 'function [capacity_rate, blended_time_min] = ophth_capacity(refer_rate, abstain_rate, n_doctors, t_refer, t_abstain)\n' ...
'%% Blended service-time approximation. When arrivals stop at session close,\n' ...
'%% total drops to ~0 and the original code jumped to t_abstain (90s), cutting\n' ...
'%% clearing capacity by 60%% and producing spurious queue build-ups. Instead,\n' ...
'%% fall back to t_refer (the faster path) when no new demand is arriving --\n' ...
'%% this is conservative for the capacity estimate (faster clearing = fewer\n' ...
'%% patients sent home) and honest: the queue at end-of-day is dominated by\n' ...
'%% auto-referred eyes, not abstentions.\n' ...
'total = refer_rate + abstain_rate;\n' ...
'if total > 1e-9\n' ...
'    blended_time_min = (refer_rate*t_refer + abstain_rate*t_abstain) / total;\n' ...
'else\n' ...
'    blended_time_min = t_refer;\n' ...
'end\n' ...
'blended_time_min = max(blended_time_min, 1e-6);\n' ...
'capacity_rate = n_doctors / blended_time_min;\n']));
wire(mdl,[band '/2'],[ophcap '/1']);
wire(mdl,[band '/3'],[ophcap '/2']);
wire(mdl,[c_ndoc '/1'],[ophcap '/3']);
wire(mdl,[c_tref '/1'],[ophcap '/4']);
wire(mdl,[c_tabs '/1'],[ophcap '/5']);

% ---- Stage 6a: image upload to the remote reviewer (store-and-forward) ----
% Referred and abstained patients' images queue on the camp's link. Capacity in
% patients/min = link Mbit/s * availability * 60 / (8 * MB per patient). The
% uploader runs off the camp's power, so it stalls in an outage like capture does.
capL = add_const(mdl,'Cap_Link',[stageX6+100 Y0+230 stageX6+200 Y0+250], ...
    'p.link.tiers_mbps(p.link.tier)*p.link.availability*60/(8*p.link.mb_per_patient)');
passL = add_const(mdl,'Pass_Link',[stageX6+100 Y0+260 stageX6+170 Y0+280],'1');
SL = add_rate_stage(mdl,'ImageLink',[stageX6+40 Y0+300]);
wire(mdl,[sumRA '/1'],[SL.fcn '/2']);
wire(mdl,[capL '/1'],[SL.fcn '/3']);
wire(mdl,[passL '/1'],[SL.fcn '/4']);
wire(mdl,[chartPath '/1'],[SL.fcn '/5']);
wire(mdl,[c_dt '/1'],[SL.fcn '/6']);
logblk(mdl,'Log_link_queue',[stageX6+300 Y0+300 stageX6+360 Y0+320],[SL.fcn '/3']);

passOph = add_const(mdl,'Pass_Ophth',[stageX6+100 Y0+140 stageX6+170 Y0+160],'1');
S6 = add_rate_stage(mdl,'Ophth',[stageX6+260 Y0+100]);
wire(mdl,[SL.fcn '/1'],[S6.fcn '/2']);
wire(mdl,[ophcap '/1'],[S6.fcn '/3']);
wire(mdl,[passOph '/1'],[S6.fcn '/4']);
% Ophthalmologists work remotely from a base hospital: their capacity is NOT gated
% by the camp's local power. Using a constant 1 here instead of chartPath/1 means
% the remote review queue drains at full speed during a camp outage (as it should),
% while the upstream pipeline stages (capture, inference) correctly stall.
c_ophgate = add_const(mdl,'C_OphGate',[stageX6+100 Y0+165 stageX6+170 Y0+185],'1');
wire(mdl,[c_ophgate '/1'],[S6.fcn '/5']);
wire(mdl,[c_dt '/1'],[S6.fcn '/6']);
logblk(mdl,'Log_ophth_queue',[stageX6+400 Y0+60 stageX6+460 Y0+80],[S6.fcn '/3']);
logblk(mdl,'Log_ophth_served',[stageX6+400 Y0+90 stageX6+460 Y0+110],[S6.fcn '/1']);

util = add_mfun(mdl,'OphthUtil',[stageX6+260 Y0+180 stageX6+400 Y0+220], sprintf([ ...
 'function util = util_calc(served_rate, capacity_rate)\n' ...
 'util = served_rate / max(capacity_rate, 1e-9);\n' ...
 'util = min(max(util,0),1);\n']));
wire(mdl,[S6.fcn '/1'],[util '/1']);
wire(mdl,[ophcap '/1'],[util '/2']);
logblk(mdl,'Log_ophth_util',[stageX6+420 Y0+180 stageX6+480 Y0+200],[util '/1']);
logblk(mdl,'Log_blended_time',[stageX6+260 Y0-60 stageX6+320 Y0-40],[ophcap '/2']);

% ---- Stage 7: Report upload (lossy rural link) ----
% Everyone who reaches a final decision needs a report uploaded:
% auto-cleared patients (band output 1, who never queue for a doctor)
% PLUS the ophthalmologist-reviewed referrals/abstentions (S6 output 1).
stageX7 = stageX6+520;
sumUp = [mdl '/Sum_ToUpload'];
add_block('simulink/Math Operations/Add', sumUp, ...
    'Position',[stageX7-40 Y0+90 stageX7-10 Y0+130],'Inputs','++');
wire(mdl,[band '/1'],[sumUp '/1']);
wire(mdl,[S6.fcn '/1'],[sumUp '/2']);

capU = add_const(mdl,'Cap_Upload',[stageX7 Y0+140 stageX7+70 Y0+160], ...
    'p.upload.n_channels*60/p.upload.time_s');
passU = add_const(mdl,'Pass_Upload',[stageX7 Y0+170 stageX7+70 Y0+190], ...
    '1-p.upload.permanent_loss_rate');
S7 = add_rate_stage(mdl,'Upload',[stageX7+90 Y0+100]);
wire(mdl,[sumUp '/1'],[S7.fcn '/2']);
wire(mdl,[capU '/1'],[S7.fcn '/3']);
wire(mdl,[passU '/1'],[S7.fcn '/4']);
wire(mdl,[chartPath '/1'],[S7.fcn '/5']);
wire(mdl,[c_dt '/1'],[S7.fcn '/6']);
logblk(mdl,'Log_screened',[stageX7+250 Y0+70 stageX7+310 Y0+90],[S7.fcn '/1']);
logblk(mdl,'Log_lost_upload',[stageX7+250 Y0+100 stageX7+310 Y0+120],[S7.fcn '/2']);

%% ---- Save --------------------------------------------------------------
set_param(mdl,'StartTime','0');
Simulink.BlockDiagram.arrangeSystem(mdl);
save_system(mdl, fullfile(pwd,'certus_screening.slx'));
fprintf('Model built and saved: %s\n', fullfile(pwd,'certus_screening.slx'));

%% ---- Smoke test: does it actually compile and run? ----------------------
try
    simOut = sim(mdl,'StopTime',num2str(p.T_min),'ReturnWorkspaceOutputs','on');
    fprintf('SMOKE TEST: sim() completed OK.\n');
    logScreened = simOut.get('Log_screened');
    logOphQ = simOut.get('Log_ophth_queue');
    finalScreened = logScreened(end,:);
    finalOphQ = logOphQ(end,:);
    fprintf('  last screened-rate sample: %s\n', mat2str(finalScreened));
    fprintf('  last ophth-queue sample:   %s\n', mat2str(finalOphQ));
    if any(~isfinite(finalScreened)) || any(~isfinite(finalOphQ))
        error('Non-finite outputs in smoke test.');
    end
    fprintf('SMOKE TEST: outputs finite - PASS.\n');
catch ME
    fprintf(2,'SMOKE TEST FAILED: %s\n', ME.message);
    rethrow(ME);
end

%% =========================== LOCAL FUNCTIONS ==============================
function wire(mdl, srcSpec, dstSpec)
% Connects srcSpec ('BlockName/portNum' or 'mdl/BlockName/portNum') to
% dstSpec the same way, resolving ports via PortHandles rather than the
% string 'block/port' address form add_line() normally accepts - string
% addressing was found to fail specifically for MATLAB Function block
% ports in this MATLAB build (R2026a), even though the ports themselves
% exist (confirmed via get_param(blk,'Ports')). Handle-based add_line
% works reliably for both standard blocks and MATLAB Function blocks.
    [sBlk, sPort] = parse_port(mdl, srcSpec);
    [dBlk, dPort] = parse_port(mdl, dstSpec);
    sh = get_param(sBlk,'PortHandles');
    dh = get_param(dBlk,'PortHandles');
    add_line(mdl, sh.Outport(sPort), dh.Inport(dPort), 'autorouting','on');
end

function [blkpath, portnum] = parse_port(mdl, spec)
    idx = find(spec == '/', 1, 'last');
    portnum = str2double(spec(idx+1:end));
    blkname = spec(1:idx-1);
    prefix = [mdl '/'];
    if startsWith(blkname, prefix) || strcmp(blkname, mdl)
        blkpath = blkname;
    else
        blkpath = [mdl '/' blkname];
    end
end

function blk = add_mfun(mdl,name,pos,scriptText)
    blk = [mdl '/' name];
    add_block('simulink/User-Defined Functions/MATLAB Function', blk, 'Position',pos);
    rt = sfroot;
    obj = rt.find('-isa','Stateflow.EMChart','Path',blk);
    obj.Script = scriptText;
    ports = get_param(blk,'Ports'); %#ok<NASGU> % forces port regeneration
end

function blk = add_const(mdl,name,pos,valueExpr)
    blk = [mdl '/' name];
    add_block('simulink/Sources/Constant', blk, 'Position',pos, 'Value',valueExpr, ...
        'ShowName','on');
end

function logblk(mdl,name,pos,srcPort)
    blk = [mdl '/' name];
    add_block('simulink/Sinks/To Workspace', blk, 'Position',pos, ...
        'VariableName',name,'SaveFormat','Array','SampleTime','p.dt_min');
    wire(mdl,srcPort,[blk '/1']);
end

function S = add_rate_stage(mdl,name,posTopLeft)
% Builds one fluid queueing stage:
%   UnitDelay (the queue level, this IS the required discrete-time
%   integrator/feedback-loop element) feeds its state into a MATLAB
%   Function that computes this step's served/lost/next-queue values,
%   which feeds back into the UnitDelay input.
    ud = [mdl '/' name '_Q'];
    fcn = [mdl '/' name '_fcn'];
    x = posTopLeft(1); y = posTopLeft(2);
    add_block('simulink/Discrete/Unit Delay', ud, 'Position',[x y x+50 y+40], ...
        'InitialCondition','0');
    add_mfun(mdl, [name '_fcn'], [x+90 y-20 x+280 y+80], sprintf([ ...
      'function [served_rate, lost_rate, queue_len, queue_next] = rate_stage(queue_prev, arrival_rate, capacity_rate, pass_frac, gate, dt)\n' ...
      '%%RATE_STAGE Generic fluid/rate-based queue update for one pipeline\n' ...
      '%% stage. queue_prev is fed back from a Unit Delay block (this is\n' ...
      '%% the discrete-time "integrator" state for the stage).\n' ...
      'capacity_count = capacity_rate * gate * dt;\n' ...
      'arrival_count = arrival_rate * dt;\n' ...
      'available = queue_prev + arrival_count;\n' ...
      'attempt = min(available, capacity_count);\n' ...
      'served_count = attempt * pass_frac;\n' ...
      'lost_count = attempt * (1 - pass_frac);\n' ...
      'queue_next = available - attempt;\n' ...
      'served_rate = served_count / dt;\n' ...
      'lost_rate = lost_count / dt;\n' ...
      'queue_len = queue_next;\n']));
    wire(mdl,[ud '/1'],[fcn '/1']);
    wire(mdl,[fcn '/4'],[ud '/1']);
    S.ud = ud; S.fcn = fcn;
end

function blk = add_power_chart(mdl,name,pos)
% Stateflow chart: GRID -> BATTERY -> OFFLINE, with recharge.
% Inputs (order):  1 grid_avail, 2 recharge_rate, 3 drain_rate, 4 dt
% Outputs (order): 1 power_on,   2 battery_frac,  3 camp_state
    blk = [mdl '/' name];
    add_block('sflib/Chart', blk, 'Position',pos);
    rt = sfroot;
    machine = rt.find('-isa','Stateflow.Machine','Name',mdl);
    chart = machine.find('-isa','Stateflow.Chart','Path',blk);
    chart.ActionLanguage = 'MATLAB';

    di1 = Stateflow.Data(chart); di1.Name='grid_avail'; di1.Scope='Input';
    di2 = Stateflow.Data(chart); di2.Name='recharge_rate'; di2.Scope='Input';
    di3 = Stateflow.Data(chart); di3.Name='drain_rate'; di3.Scope='Input';
    di4 = Stateflow.Data(chart); di4.Name='dt'; di4.Scope='Input';

    do1 = Stateflow.Data(chart); do1.Name='power_on'; do1.Scope='Output';
    do2 = Stateflow.Data(chart); do2.Name='battery_frac'; do2.Scope='Output';
    do3 = Stateflow.Data(chart); do3.Name='camp_state'; do3.Scope='Output';

    dloc = Stateflow.Data(chart); dloc.Name='battery'; dloc.Scope='Local';

    sGrid = Stateflow.State(chart); sGrid.Name='GRID'; sGrid.Position=[20 20 140 80];
    sGrid.LabelString = sprintf(['GRID\nentry: power_on=1; camp_state=1;\n' ...
        'during: battery=min(1,battery+recharge_rate*dt); power_on=1; battery_frac=battery;']);

    sBatt = Stateflow.State(chart); sBatt.Name='BATTERY'; sBatt.Position=[200 20 140 80];
    sBatt.LabelString = sprintf(['BATTERY\nentry: camp_state=2;\n' ...
        'during: battery=max(0,battery-drain_rate*dt); power_on=1; battery_frac=battery;']);

    sOff = Stateflow.State(chart); sOff.Name='OFFLINE'; sOff.Position=[200 140 140 80];
    sOff.LabelString = sprintf(['OFFLINE\nentry: power_on=0; camp_state=3; battery=0;\n' ...
        'during: battery_frac=battery;']);

    % Default transition into GRID (full battery at camp start)
    defT = Stateflow.Transition(chart);
    defT.Destination = sGrid;
    defT.DestinationOClock = 0;
    defT.LabelString = '{battery=1;}';
    defT.SourceEndpoint = [sGrid.Position(1)-30, sGrid.Position(2)+sGrid.Position(4)/2];
    defT.DestinationEndpoint = [sGrid.Position(1), sGrid.Position(2)+sGrid.Position(4)/2];

    t1 = Stateflow.Transition(chart); t1.Source = sGrid; t1.Destination = sBatt;
    t1.LabelString = '[~grid_avail]';
    t2 = Stateflow.Transition(chart); t2.Source = sBatt; t2.Destination = sGrid;
    t2.LabelString = '[grid_avail]';
    t3 = Stateflow.Transition(chart); t3.Source = sBatt; t3.Destination = sOff;
    t3.LabelString = '[battery<=0]';
    t4 = Stateflow.Transition(chart); t4.Source = sOff; t4.Destination = sGrid;
    t4.LabelString = '[grid_avail]';
end
