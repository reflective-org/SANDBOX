% dump_trajectory.m
% -----------------------------------------------------------------------------
% Generate a "ground truth" multi-day trajectory by reproducing the NUMERIC CORE
% of runconcs_het.m (default scenario) in Octave, calling the unchanged
% concs_het.m / h2so4wpATfxn.m / hetgammasJPL00.m. Plotting is omitted.
%
% Key difference from runconcs_het.m: each day/night segment is integrated on a
% FIXED time grid (every DT seconds) instead of letting the solver pick output
% points. This lets the Python port (scipy solve_ivp with t_eval) be compared
% point-for-point against this reference, even though the two solvers take
% different internal steps.
%
% Output: tests/fixtures/trajectory.csv  (column 1 = time in seconds, then the
%         34 species in canonical order) and trajectory_meta.csv (the scenario).
%
% Run with:
%   octave --no-gui tests/octave/dump_trajectory.m
% -----------------------------------------------------------------------------

addpath('src-matlab');
outdir = 'tests/fixtures';

global T P M SA WTR SZA Yn2o5 count opt

% --- Scenario (matches the defaults set at the top of runconcs_het.m) ---------
conv = 760/1013.25;
T = 210;                      % temperature (K)
P = 68;                       % pressure (mbar)
M = 9.65e18*P/T*conv;         % air number density (molec/cm^3)
Yn2o5 = 0.1;                  % gamma for N2O5 + H2O
SA = 2;                       % aerosol surface area (um^2/cm^3)
WTR = 5;                      % water vapour (ppm)
opt = 1;                      % O3-photolysis workaround (as in the Science paper)
count = 0;
td = 14;                      % daytime length (hours)
tn = 10;                      % nighttime length (hours)
days = 2;                     % number of days (small => fast, deterministic fixture)
DT = 600;                     % fixed output spacing within a segment (seconds)

% --- Initial conditions for P = 68 (transcribed from runconcs_het.m) ----------
% NOTE: at P=68 the MATLAB source multiplies HNO3_i and BrO_i by 0 (lines 69-70).
%       We preserve that as-written; it is flagged in the porting plan.
ppt = 1e-12 * M;
ClO_i=10*ppt; ClONO2_i=127*ppt; HCl_i=777*ppt; NO_i=450*ppt; NO2_i=450*ppt;
O3_i=1.18e6*ppt; HNO3_i=3470*0*ppt; BrO_i=3*0*ppt;
Cl_i=0; ClOOCl_i=0; Cl2_i=0; O_i=0; O2_i=0.21*M; CH3_i=0; CH4_i=1.6e6*ppt;
H2O_i=WTR*1e6*ppt; HOCl_i=0; N2O5_i=20*ppt; NO3_i=0; OH_i=0.5*ppt; HO2_i=3*ppt;
O1D_i=0; HONO_i=0; HNO4_i=0; OClO_i=0; Br_i=0; BrONO2_i=0; BrCl_i=0; HBr_i=0;
HOBr_i=0; HNO3aq_i=0; C2H6_i=200*ppt; SO2_i=(2.4e9/2516)*ppt; H2O2_i=0;

x0 = [Cl_i; ClO_i; ClOOCl_i; ClONO2_i; HCl_i; Cl2_i; NO_i; NO2_i; O_i; O2_i; ...
      O3_i; CH3_i; CH4_i; HNO3_i; H2O_i; HOCl_i; N2O5_i; NO3_i; OH_i; HO2_i; ...
      O1D_i; HONO_i; HNO4_i; OClO_i; Br_i; BrO_i; BrONO2_i; BrCl_i; HBr_i; ...
      HOBr_i; HNO3aq_i; C2H6_i; SO2_i; H2O2_i];

% --- Solver tolerances (matches the odeset call in runconcs_het.m) ------------
% opt==1 => the O/O1D entries also use 1e-6, so the whole AbsTol vector is 1e-6.
if opt == 0; tol = 1e-3; else; tol = 1e-6; end
abstol = 1e-6 * ones(34, 1);
abstol(9)  = tol;    % O
abstol(21) = tol;    % O1D
% InitialStep: Octave's ode15s (SUNDIALS/IDA) otherwise picks too large a first step
% and fails the error test on the fast O/O1D transient at t=0 (MATLAB's ode15s notes
% the same fragility). A tiny first step is a startup detail only -- the solution is
% solver-independent (ode15s and ode23s agree on O3 to ~8 significant figures).
options = odeset('RelTol', 1e-3, 'AbsTol', abstol, 'InitialStep', 1e-10);

% --- helper: integrate one segment on a fixed grid and append to T_all/X_all --
function [tt, xx] = seg(t_start, t_end, x_start, DT, options)
    grid = (t_start:DT:t_end)';
    if grid(end) ~= t_end; grid = [grid; t_end]; end   % always include endpoint
    [tt, xx] = ode15s(@concs_het, grid, x_start, options);
end

% --- First daytime period (sets things up), then alternate night/day ----------
t0 = 0;
tf = td*3600;
SZA = 1;
[T_all, X_all] = seg(t0, tf, x0, DT, options);

for d = 1:days
    % nighttime
    SZA = 0;
    t_start = T_all(end);
    [tt, xx] = seg(t_start, t_start + tn*3600, X_all(end,:)', DT, options);
    T_all = [T_all; tt(2:end)];        % drop duplicated segment-boundary point
    X_all = [X_all; xx(2:end,:)];

    % daytime
    SZA = 1;
    t_start = T_all(end);
    [tt, xx] = seg(t_start, t_start + td*3600, X_all(end,:)', DT, options);
    T_all = [T_all; tt(2:end)];
    X_all = [X_all; xx(2:end,:)];
end

% --- Write CSV ----------------------------------------------------------------
species = {'Cl','ClO','ClOOCl','ClONO2','HCl','Cl2','NO','NO2','O','O2','O3', ...
           'CH3','CH4','HNO3','H2O','HOCl','N2O5','NO3','OH','HO2','O1D','HONO', ...
           'HNO4','OClO','Br','BrO','BrONO2','BrCl','HBr','HOBr','HNO3aq','C2H6', ...
           'SO2','H2O2'};

fid = fopen(fullfile(outdir, 'trajectory.csv'), 'w');
fprintf(fid, 'time');
for i = 1:numel(species); fprintf(fid, ',%s', species{i}); end
fprintf(fid, '\n');
for r = 1:rows(X_all)
    fprintf(fid, '%.17g', T_all(r));
    fprintf(fid, ',%.17g', X_all(r,:));
    fprintf(fid, '\n');
end
fclose(fid);

fid = fopen(fullfile(outdir, 'trajectory_meta.csv'), 'w');
fprintf(fid, 'key,value\n');
fprintf(fid, 'T,%.17g\nP,%.17g\nM,%.17g\nSA,%.17g\nWTR,%.17g\nYn2o5,%.17g\nopt,%d\n', ...
        T, P, M, SA, WTR, Yn2o5, opt);
fprintf(fid, 'td,%d\ntn,%d\ndays,%d\nDT,%d\n', td, tn, days, DT);
fclose(fid);

printf('Wrote trajectory.csv (%d rows) and trajectory_meta.csv to %s\n', rows(X_all), outdir);
