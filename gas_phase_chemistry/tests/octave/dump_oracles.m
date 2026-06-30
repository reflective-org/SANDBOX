% dump_oracles.m
% -----------------------------------------------------------------------------
% Generate "ground truth" CSV fixtures by running the ORIGINAL MATLAB functions
% in src-matlab/ through Octave. These fixtures are committed and the Python port
% is tested against them (see tests/test_aerosol.py, test_gammas.py, test_rhs.py).
%
% This script does NOT modify src-matlab/ -- it only calls those functions.
%
% Run with:
%   octave --no-gui tests/octave/dump_oracles.m
% -----------------------------------------------------------------------------

addpath('src-matlab');
outdir = 'tests/fixtures';

% =============================================================================
% 1) Aerosol composition: h2so4wpATfxn(T, P, H2O)
%    Pick (T, P, H2O) points that land in each water-activity branch of the
%    parameterization (a_W <= 0.05, 0.05 < a_W < 0.85, a_W >= 0.85).
% =============================================================================
aero_inputs = [ ...
%    T      P     H2O(ppm)
    210,   68,    5;        % cold/dry -> low water activity
    210,   68,   50;        % more water
    220,  100,    5;
    195,   46,   20;
    230,  121,  200;        % warm/wet -> higher water activity
    200,   83,   10;
    198,  121,   18;        % cold + moist -> high water-activity branch (a_W ~ 0.93)
];

fid = fopen(fullfile(outdir, 'aerosol.csv'), 'w');
fprintf(fid, 'T,P,H2O,h2so4wpAT,h2so4mlAT,a_WAT\n');
for i = 1:rows(aero_inputs)
    T = aero_inputs(i,1); P = aero_inputs(i,2); H2O = aero_inputs(i,3);
    [wp, ml, aw] = h2so4wpATfxn(T, P, H2O);
    fprintf(fid, '%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', T, P, H2O, wp, ml, aw);
end
fclose(fid);

% =============================================================================
% 2) Heterogeneous uptake coefficients: hetgammasJPL00(...)
%    Reuse the aerosol composition computed above so inputs are self-consistent.
%    HCl/ClONO2 in ppbv, radius in cm, sts = 0 (sulfate ternary solution flag).
% =============================================================================
gam_inputs = [ ...
%    T      P     H2O   HCl(ppb)  ClONO2(ppb)  radius(cm)  sts
    210,   68,    5,    0.777,    0.127,       0.1e-4,     0;
    210,   68,   50,    0.777,    0.127,       0.1e-4,     0;
    220,  100,    5,    0.294,    0.042,       0.1e-4,     0;
    195,   46,   20,    1.000,    0.200,       0.1e-4,     0;
    200,   83,   10,    0.551,    0.087,       0.1e-4,     0;
    198,  121,   18,    0.551,    0.087,       0.1e-4,     0;   % high water-activity branch
];

fid = fopen(fullfile(outdir, 'gammas.csv'), 'w');
fprintf(fid, 'T,P,H2O,HCl,ClONO2,radius,sts,h2so4wp,a_W,Yhocl,Yclnh2o,Yclnhcl\n');
for i = 1:rows(gam_inputs)
    T = gam_inputs(i,1); P = gam_inputs(i,2); H2O = gam_inputs(i,3);
    HCl = gam_inputs(i,4); ClONO2 = gam_inputs(i,5);
    radius = gam_inputs(i,6); sts = gam_inputs(i,7);
    [wp, ml, aw] = h2so4wpATfxn(T, P, H2O);
    [Yhocl, Yclnh2o, Yclnhcl] = hetgammasJPL00(T, P, wp, aw, HCl, ClONO2, radius, sts);
    fprintf(fid, '%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', ...
            T, P, H2O, HCl, ClONO2, radius, sts, wp, aw, Yhocl, Yclnh2o, Yclnhcl);
end
fclose(fid);

% =============================================================================
% 3) Right-hand side dC/dt: concs_het(t, x)
%    concs_het reads the scenario from globals. We build a realistic state vector
%    from the P=68 initial conditions and evaluate dC/dt for daytime (SZA=1) and
%    nighttime (SZA=0). Species order matches the comment at the top of concs_het.m.
% =============================================================================
global T P M SA WTR SZA Yn2o5 count opt
conv = 760/1013.25;
T = 210; P = 68; M = 9.65e18*P/T*conv;
SA = 2; WTR = 5; Yn2o5 = 0.1; opt = 1; count = 0;

ppt = 1e-12 * M;   % ppt -> molec/cm^3
% State vector in the canonical species order (values chosen to be a plausible
% mid-run mixture so most reaction terms are non-zero).
x = [ ...
    5*ppt;        % Cl
    10*ppt;       % ClO
    1*ppt;        % ClOOCl
    127*ppt;      % ClONO2
    777*ppt;      % HCl
    0.5*ppt;      % Cl2
    450*ppt;      % NO
    450*ppt;      % NO2
    1*ppt;        % O
    0.21*M;       % O2
    1.18e6*ppt;   % O3
    0.01*ppt;     % CH3
    1.6e6*ppt;    % CH4
    1100*ppt;     % HNO3
    5e6*ppt;      % H2O
    1*ppt;        % HOCl
    20*ppt;       % N2O5
    0.1*ppt;      % NO3
    0.5*ppt;      % OH
    3*ppt;        % HO2
    1e-3*ppt;     % O1D
    0.1*ppt;      % HONO
    0.1*ppt;      % HNO4
    0.1*ppt;      % OClO
    0.5*ppt;      % Br
    3*ppt;        % BrO
    0.5*ppt;      % BrONO2
    0.1*ppt;      % BrCl
    0;            % HBr
    0.1*ppt;      % HOBr
    1*ppt;        % HNO3aq
    200*ppt;      % C2H6
    (2.4e9/2516)*ppt;  % SO2
    0.1*ppt;      % H2O2
];

species = {'Cl','ClO','ClOOCl','ClONO2','HCl','Cl2','NO','NO2','O','O2','O3', ...
           'CH3','CH4','HNO3','H2O','HOCl','N2O5','NO3','OH','HO2','O1D','HONO', ...
           'HNO4','OClO','Br','BrO','BrONO2','BrCl','HBr','HOBr','HNO3aq','C2H6', ...
           'SO2','H2O2'};

SZA = 1; dCdt_day   = concs_het(0, x);
SZA = 0; dCdt_night = concs_het(0, x);

% Write inputs (scenario + state) and outputs (dC/dt) so the Python test can
% reconstruct exactly the same evaluation.
fid = fopen(fullfile(outdir, 'rhs_meta.csv'), 'w');
fprintf(fid, 'key,value\n');
fprintf(fid, 'T,%.17g\nP,%.17g\nM,%.17g\nSA,%.17g\nWTR,%.17g\nYn2o5,%.17g\nopt,%d\n', ...
        T, P, M, SA, WTR, Yn2o5, opt);
fclose(fid);

fid = fopen(fullfile(outdir, 'rhs.csv'), 'w');
fprintf(fid, 'species,x,dCdt_day,dCdt_night\n');
for i = 1:numel(species)
    fprintf(fid, '%s,%.17g,%.17g,%.17g\n', species{i}, x(i), dCdt_day(i), dCdt_night(i));
end
fclose(fid);

printf('Wrote aerosol.csv, gammas.csv, rhs.csv, rhs_meta.csv to %s\n', outdir);
