function expData = prepare_experimental_data(material_id, opts)
% PREPARE_EXPERIMENTAL_DATA  Load and preprocess LIC experimental data
%
% Syntax:
%   expData = prepare_experimental_data(material_id, opts)
%
% Inputs:
%   material_id - Material identifier:
%       0: synthetic data
%       1-3: UM datasets (UM1, UM2, UM3)
%       4-5: UT datasets (UT1, UT2)
%       6-9: Other materials (Ag2.5, Ag0.5, PA1.7, PEGDA)
%   opts - Options structure with fields:
%       .dataDir       : Data directory (default: './experiments')
%       .rho           : Density [kg/m³] (default: 1064)
%       .p_inf         : Far-field pressure [Pa] (default: 101325)
%       .epsH_rel_frac : Strain threshold fraction (default: 0.10)
%       .epsH_abs_floor: Strain absolute floor (default: 0.00)
%       .epsdot_base   : Strain rate threshold base (default: 1e5)
%       .verbose       : Display info (default: true)
%
% Outputs:
%   expData - Structure with fields:
%       .Rmatrix, .Rdotmatrix, .tmatrix  : Aligned data [nTime x nTrials]
%       .tc                              : Characteristic times [1 x nTrials]
%       .Rmax_range                      : Maximum radii per trial
%       .Req_each                        : Equilibrium radii per trial
%       .mask                            : High-information region mask
%       .sigmaR0, .sigmaRdot0            : Baseline uncertainties
%       .strain, .strainRate             : Derived quantities
%       .material_id                     : Input material ID
%       .material_name                   : Material name string
%
% Example:
%   expData = prepare_experimental_data(1);  % UM1 (10% gelatin)
%   expData = prepare_experimental_data(5, struct('verbose', false));
%
% See also: bayesian_model_selection

% Author: [Your name]
% Date: 2025

%% Parse inputs
if nargin < 2, opts = struct(); end
opts = set_defaults(opts);

%% Load dimensional radius-time data
[R_d, t_d, material_name] = load_material_data(material_id, opts);

if opts.verbose
    fprintf('=== Experimental Data Preparation ===\n');
    fprintf('Material: %s (ID=%d)\n', material_name, material_id);
    fprintf('Trials: %d | Time points: %d\n', size(R_d, 2), size(R_d, 1));
end

%% Align at peak and truncate
[R_d, t_d] = align_at_peak(R_d, t_d);

% Reset time to start at zero per trial
for k = 1:size(t_d, 2)
    t_d(:, k) = t_d(:, k) - min(t_d(:, k));
end

%% Nondimensionalize
Rmax_range = max(R_d, [], 1);
tc = Rmax_range .* sqrt(opts.rho / opts.p_inf);

Rmatrix = R_d ./ Rmax_range;   % R* = R/Rmax
tmatrix = t_d ./ tc;            % t* = t/tc

% Equilibrium radius (average of last 10% of data)
last10 = max(1, ceil(0.10 * size(R_d, 1)));
Req_each = mean(R_d(end-last10+1:end, :), 1);

%% Compute velocities with noise reduction
if exist('expDataPrep.m', 'file') == 2
    % Use existing expDataPrep if available
    [~, ~, ~, ~, R_stats, Rdot_stats, Rdotmatrix, ~, ~, ~] = ...
        dataprep(tmatrix, Rmatrix, tc, R_d, Rmax_range);
else
    % Simple gradient-based velocity
    Rdotmatrix = zeros(size(Rmatrix));
    for j = 1:size(Rmatrix, 2)
        Rdotmatrix(:, j) = gradient(Rmatrix(:, j), tmatrix(:, j));
    end
    R_stats = [];
    Rdot_stats = [];
end

%% Compute baseline uncertainties
sigmaR0 = get_sigma_column(R_stats, Rmatrix, 1e-12);
sigmaRdot0 = get_sigma_column(Rdot_stats, Rdotmatrix, 1e-12);

%% Strain and strain rate
% Dimensionless strain rate: ε̇* = -2(Ṙ*/R*)
strainRate = -2 .* Rdotmatrix ./ max(Rmatrix, 1e-12);
strainRate(~isfinite(strainRate)) = NaN;

% Dimensional strain rate [s⁻¹]
strainRate_dim = abs(strainRate) ./ tc;

% Log-Hencky strain referenced to equilibrium
strain = zeros(size(R_d));
for j = 1:size(R_d, 2)
    denom = max(Req_each(j), 1e-12);
    strain(:, j) = 0.5 .* log(max((R_d(:, j) ./ denom).^(-4), 1e-12));
end

%% Build high-information mask (elliptical gate)
mask = build_information_mask(Rmatrix, Rdotmatrix, tmatrix, tc, ...
    Req_each, Rmax_range, opts);

if opts.verbose
    fprintf('High-information points: %d / %d (%.1f%%)\n', ...
            nnz(mask), numel(mask), 100*nnz(mask)/numel(mask));
end

%% Package output
expData = struct(...
    'Rmatrix', Rmatrix, ...
    'Rdotmatrix', Rdotmatrix, ...
    'tmatrix', tmatrix, ...
    'tc', tc, ...
    'Rmax_range', Rmax_range, ...
    'Req_each', Req_each, ...
    'R_d', R_d, ...
    't_d', t_d, ...
    'mask', mask, ...
    'sigmaR0', sigmaR0, ...
    'sigmaRdot0', sigmaRdot0, ...
    'strain', strain, ...
    'strainRate', strainRate, ...
    'strainRate_dim', strainRate_dim, ...
    'R_stats', R_stats, ...
    'Rdot_stats', Rdot_stats, ...
    'material_id', material_id, ...
    'material_name', material_name, ...
    'rho', opts.rho, ...
    'p_inf', opts.p_inf);

end

%% ==================== Loading Functions ====================

function [R_d, t_d, material_name] = load_material_data(material_id, opts)
% Load dimensional radius-time data for specified material

switch material_id
    case 0
        % Synthetic data
        error('Synthetic data loader not yet implemented. Use your own synthetic data generator.');
        
    case 1  % UM1 - 10% Gelatin
        data_path = fullfile(opts.dataDir,'Processed_Data.mat');
        [R_d, t_d] = load_processed_data(data_path, 7:15);
        material_name = 'UM1_Gelatin_10pct';
        
    case 2  % UM2 - 0.2% Fibrin
        data_path = fullfile(opts.dataDir, 'UM', 'Processed_Data.mat');
        [R_d, t_d] = load_processed_data(data_path, 78:98);
        material_name = 'UM2_Fibrin_0.2pct';
        
    case 3  % UM3 - 8.0/0.26% PAAm
        data_path = fullfile(opts.dataDir, 'UM', 'Processed_Data.mat');
        [R_d, t_d] = load_processed_data(data_path, 223:234);
        material_name = 'UM3_PAAm_8.0_0.26pct';
        
    case 4  % UT1 - 10% Gelatin
        data_path = fullfile(opts.dataDir, '10percent', 'Rt_nondim_exp.mat');
        [R_d, t_d] = load_nondim_data(data_path, [1,2,3,4], opts);
        material_name = 'UT1_Gelatin_10pct';
        
    case 5  % UT2 - 5.0% Agarose
        data_path = fullfile(opts.dataDir, 'Ag5.0', 'Rt_nondim_exp.mat');
        indices = [1:5, 7:10];
        [R_d, t_d] = load_nondim_data(data_path, indices, opts);
        material_name = 'UT2_Agarose_5.0pct';
        
    case 6  % Ag2.5
        data_path = fullfile(opts.dataDir, 'Ag2.5', 'Rt_nondim_exp.mat');
        indices = [1:6, 8:11];
        [R_d, t_d] = load_nondim_data(data_path, indices, opts);
        material_name = 'Agarose_2.5pct';
        
    case 7  % Ag0.5
        data_path = fullfile(opts.dataDir, 'Ag0.5', 'Rt_nondim_exp.mat');
        indices = 2:11;
        [R_d, t_d] = load_nondim_data(data_path, indices, opts);
        material_name = 'Agarose_0.5pct';
        
    case 8  % PA1.7
        data_path = fullfile(opts.dataDir, 'PA1pt7', 'Rt_nondim_exp.mat');
        indices = 1:10;
        [R_d, t_d] = load_nondim_data(data_path, indices, opts);
        material_name = 'PAAm_1.7pct';
        
    case 9  % PEGDA
        data_path = fullfile(opts.dataDir, 'forbenchmarks', 'Rt_nondim_exp_PEGDA_S25L6.mat');
        indices = 1:7;
        [R_d, t_d] = load_nondim_data(data_path, indices, opts);
        material_name = 'PEGDA';
        
    otherwise
        error('Unknown material_id: %d', material_id);
end

end

function [R_d, t_d] = load_processed_data(filepath, indices)
% Load from Processed_Data.mat format
if ~exist(filepath, 'file')
    error('Data file not found: %s', filepath);
end

m = load(filepath);
assert(isfield(m, 'expts'), 'Expected field "expts" in %s', filepath);

k = 1;
for i = indices
    R_d(:, k) = m.expts(i).Roft; %#ok<AGROW>
    t_d(:, k) = m.expts(i).t;    %#ok<AGROW>
    k = k + 1;
end
end

function [R_d, t_d] = load_nondim_data(filepath, indices, opts)
% Load from Rt_nondim_exp.mat format (UT data)
if ~exist(filepath, 'file')
    error('Data file not found: %s', filepath);
end

m = load(filepath);
assert(isfield(m, 'R_nondim_All') && isfield(m, 't_nondim_All'), ...
    'Expected fields R_nondim_All, t_nondim_All in %s', filepath);

if isfield(m, 'P_inf')
    p_inf_local = m.P_inf;
else
    p_inf_local = opts.p_inf;
end

minLen = min(cellfun(@length, m.R_nondim_All(indices)));
k = 1;
for i = indices
    tc_local = m.RmaxList(i) * sqrt(opts.rho / p_inf_local);
    R_d(:, k) = m.R_nondim_All{i}(1:minLen) * m.RmaxList(i); %#ok<AGROW>
    t_d(:, k) = m.t_nondim_All{i}(1:minLen) * tc_local;      %#ok<AGROW>
    k = k + 1;
end
end

%% ==================== Processing Functions ====================

function [R_aligned, t_aligned] = align_at_peak(R_d, t_d)
% Align all trials at their peak radius
[~, startIdx] = max(R_d, [], 1);
nCols = size(R_d, 2);
tailLen = size(R_d, 1) - startIdx + 1;
minLen = min(tailLen);

R_aligned = zeros(minLen, nCols);
t_aligned = zeros(minLen, nCols);

for j = 1:nCols
    idx = startIdx(j):(startIdx(j) + minLen - 1);
    R_aligned(:, j) = R_d(idx, j);
    t_aligned(:, j) = t_d(idx, j);
end
end

function mask = build_information_mask(Rmatrix, Rdotmatrix, tmatrix, tc, ...
    Req_each, Rmax_range, opts)
% Build mask for high-information regions using elliptical gate

[nTime, nTrials] = size(Rmatrix);

% Compute log-Hencky strain
Req_star = repmat(Req_each, nTime, 1);
epsH_star = -2 .* log(max(Rmatrix, 1e-12) ./ max(Req_star, 1e-12));

% Compute dimensionless strain rate
tc_row = repmat(max(tc(:).', eps), nTime, 1);
epsdot_star = -2 .* (Rdotmatrix ./ max(Rmatrix, 1e-12)) .* tc_row;

% Thresholds
epsH_max_per_trial = max(abs(epsH_star), [], 1);
epsH_th_trial = max(opts.epsH_abs_floor, ...
    opts.epsH_rel_frac .* epsH_max_per_trial);
epsH_th_mat = repmat(epsH_th_trial, nTime, 1);

epsdot_th_mat = opts.epsdot_base .* tc_row;

% Base mask: finite and valid
finite_all = isfinite(tmatrix) & isfinite(Rmatrix) & isfinite(Rdotmatrix);
mask_base = finite_all;

% Elliptical gate
inHigh = ((abs(epsH_star) ./ max(epsH_th_mat, eps)).^2 + ...
          (abs(epsdot_star) ./ max(epsdot_th_mat, eps)).^2) >= 1;

mask = mask_base & inHigh;
end

function sigma_base = get_sigma_column(statsMaybe, M, floorv)
% Extract or estimate uncertainty
if ~isempty(statsMaybe) && size(statsMaybe, 2) >= 2
    sigma_base = max(statsMaybe(:, 2), floorv);
else
    sigma_base = max(std(M, 0, 2, 'omitnan'), floorv);
end
end

%% ==================== Default Options ====================

function opts = set_defaults(opts)
if ~isfield(opts, 'dataDir'),        opts.dataDir = './experiments';     end
if ~isfield(opts, 'rho'),            opts.rho = 1064;                    end
if ~isfield(opts, 'p_inf'),          opts.p_inf = 101325;                end
if ~isfield(opts, 'epsH_rel_frac'),  opts.epsH_rel_frac = 0.10;          end
if ~isfield(opts, 'epsH_abs_floor'), opts.epsH_abs_floor = 0.00;         end
if ~isfield(opts, 'epsdot_base'),    opts.epsdot_base = 1e5;             end
if ~isfield(opts, 'verbose'),        opts.verbose = true;                end
end