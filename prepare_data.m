function expData = prepare_data(material_id, opts)
% PREPARE_DATA  Load and preprocess experimental IMR data
%
% This function loads dimensional R(t) data, aligns trials at peak radius,
% nondimensionalizes, computes velocities and strain quantities, and applies
% high-information gating masks following the paper (Eq. 4-6).
%
% It also computes baseline noise scales and weights for the Bayesian IMR
% likelihood:
%   sigma0_R, sigma0_Rdot, weights_w
%
% Syntax:
%   expData = prepare_data(material_id)
%   expData = prepare_data(material_id, opts)
%
% Inputs:
%   material_id - Integer selecting material dataset:
%                 1: UM1 (10% Gelatin)
%                 2: UM2 (0.2% Fibrin)
%                 3: UM3 (8%/0.26% PAAm)
%                 4: UT1 (10% Gelatin)
%                 5: UT2 (5% Agarose)
%                 6-9: Other materials
%
%   opts - Optional settings structure:
%       .dataDir           - Path to experiments/ folder (default: './experiments')
%       .rho               - Density [kg/m³] (default: 1064)
%       .p_inf             - Far-field pressure [Pa] (default: 101325)
%       .epsH_rel_frac     - Strain threshold fraction (default: 0.10)
%       .epsdot_base       - Strain rate threshold base (default: 1e5)
%       .useEllipticalGate - Use elliptical gate (Eq. 6) (default: true)
%       .verbose           - Display progress (default: true)
%
% Outputs:
%   expData - Structure containing:
%       .Rmatrix       [nTime x nTrials] - Nondimensional radius R*
%       .Rdotmatrix    [nTime x nTrials] - Nondimensional velocity dR*/dτ
%       .tmatrix       [nTime x nTrials] - Nondimensional time τ
%       .strain        [nTime x nTrials] - Hencky strain ε* (Eq. 4)
%       .strainRate    [nTime x nTrials] - Strain rate ε̇* (Eq. 4)
%       .mask          [nTime x nTrials] - High-information gate (Eq. 6)
%       .tc            [1 x nTrials]     - Characteristic time per trial
%       .Rmax_range    [1 x nTrials]     - Peak radius per trial
%       .Req_each      [1 x nTrials]     - Equilibrium radius per trial
%       .material_name - String describing material
%       .rho, .p_inf   - Physical parameters used for nondimensionalization
%
%   Additional fields for Bayesian likelihood:
%       .sigma0_R      [nTime x nTrials] - Baseline std for R*
%       .sigma0_Rdot   [nTime x nTrials] - Baseline std for dR*/dτ
%       .weights_w     [nTime x nTrials] - Per-point weights (1 in gate, 0 out)
%
% Example:
%   expData = prepare_data(1);  % Load UM1 (10% Gelatin)
%   fprintf('Loaded %d trials with %d time steps\n', ...
%           size(expData.Rmatrix, 2), size(expData.Rmatrix, 1));
%
% See also: build_priors, train_gpr_surrogate, forward_solver_wrapper

%% Parse inputs
if nargin < 2, opts = struct(); end
opts = parse_options(opts);

if opts.verbose
    fprintf('\n=== IMR Data Preparation ===\n');
    fprintf('Material ID: %d\n', material_id);
end

%% Load dimensional data: R_d [nTime x nTrials], t_d [nTime x nTrials]
[R_d, t_d, material_name] = load_dimensional_data(material_id, opts);

if opts.verbose
    fprintf('Material: %s\n', material_name);
    fprintf('Loaded: %d trials, %d time points\n', size(R_d,2), size(R_d,1));
end

%% Align at peak radius and truncate
[R_d, t_d] = align_at_peak(R_d, t_d, opts);

%% Nondimensionalize
Rmax_range = max(R_d, [], 1);                       % [1 x nTrials]
tc         = Rmax_range .* sqrt(opts.rho / opts.p_inf); % [1 x nTrials]

Rmatrix = R_d ./ Rmax_range;                        % R* ∈ [0,1]
tmatrix = t_d ./ tc;                                % τ (dimensionless)

if opts.verbose
    fprintf('Nondimensionalized: R* ∈ [%.3f, %.3f], τ ∈ [%.3f, %.3f]\n', ...
            min(Rmatrix(:)), max(Rmatrix(:)), ...
            min(tmatrix(:)), max(tmatrix(:)));
end

%% Compute velocities (use dataprep if available, else gradient)
Rdotmatrix = compute_velocities(Rmatrix, tmatrix, tc, R_d, Rmax_range);

%% Compute equilibrium radius per trial (median of last 10% of dimensional R_d, BIMR-style)
[nTime_d, J] = size(R_d);
tailLo = max(1, floor(0.9 * nTime_d));   % last 10% of samples

Req_each = nan(1, J);  % dimensional Req per trial

for j = 1:J
    rj = R_d(:, j);
    rj = rj(isfinite(rj));  % drop NaNs/Infs
    
    if isempty(rj)
        % fallback: equilibrium = Rmax for that trial
        Req_each(j) = Rmax_range(j);
        continue;
    end

    % tail segment
    tail = rj(tailLo : min(numel(rj), nTime_d));
    if isempty(tail)
        tail = rj;
    end

    req = median(tail, 'omitnan');
    if ~isfinite(req) || req <= 0
        req = median(rj, 'omitnan');
    end
    if ~isfinite(req) || req <= 0
        req = Rmax_range(j);   % final safety fallback
    end

    Req_each(j) = req;
end

%% Compute strain and strain rate (Paper Eq. 4)
[strain, strainRate] = compute_strain_quantities(R_d, Rdotmatrix, Rmatrix, ...
                                                 Req_each, tc, opts);

%% Apply high-information gate (Paper Eq. 5-6)
mask = compute_high_info_mask(strain, strainRate, tc, opts);

if opts.verbose
    fprintf('High-info gate: %.1f%% of points retained\n', ...
            100*nnz(mask)/numel(mask));
end

%% ========== Likelihood baseline fields (for Bayesian IMR) ==========
[nTime, nTrials] = size(Rmatrix);

% BIMR-style baseline std across trials
sigmaR_time    = max(std(Rmatrix,   0, 2, 'omitnan'), 1e-12); % [nTime x 1]
sigmaRdot_time = max(std(Rdotmatrix,0, 2, 'omitnan'), 1e-12); % [nTime x 1]

sigma0_R    = repmat(sigmaR_time,    1, nTrials);  % [nTime x nTrials]
sigma0_Rdot = repmat(sigmaRdot_time, 1, nTrials);  % [nTime x nTrials]

% Heteroscedastic weights (based on strain rate)
% Parameters for logistic weighting
kappa   = opts.kappa; % Logistic steepness
m_floor = opts.m_floor; % Minimum weight 

% Strain-rate threshold per trial 
tc_row     = repmat(tc, nTime, 1);
epsdot_th  = opts.epsdot_base * tc_row;   

% Logistic activation based on distance from threshold
z = (epsdot_th - abs(strainRate)) ./ max(epsdot_th, eps);
a = 1 ./ (1 + exp(-kappa .* z));

% Map activation to weight range [m_floor, 1] and clamp
weights_w = m_floor + (1 - m_floor) .* a;
weights_w = min(max(weights_w, m_floor), 1);

% Apply high-information gate: zero weight outside mask
weights_w(~mask) = 0;
weights_w(mask)  = max(weights_w(mask), m_floor);

%% Package output
expData = struct();
expData.Rmatrix       = Rmatrix;
expData.Rdotmatrix    = Rdotmatrix;
expData.tmatrix       = tmatrix;
expData.strain        = strain;
expData.strainRate    = strainRate;
expData.mask          = mask;
expData.tc            = tc;
expData.Rmax_range    = Rmax_range;
expData.Req_each      = Req_each;
expData.material_id   = material_id;
expData.material_name = material_name;
expData.rho           = opts.rho;
expData.p_inf         = opts.p_inf;

% Average quantities for forward model
expData.tc_mean       = mean(tc,        'omitnan');
expData.Rmax_mean     = mean(Rmax_range,'omitnan');

% Likelihood-related fields
expData.sigma0_R      = sigma0_R;
expData.sigma0_Rdot   = sigma0_Rdot;
expData.weights_w     = weights_w;

if opts.verbose
    fprintf('=== Data preparation complete ===\n\n');
end

end

%% ==================== Helper Functions ====================

function opts = parse_options(opts)
% Set default options
if ~isfield(opts, 'dataDir'),           opts.dataDir = './experiments';   end
if ~isfield(opts, 'rho'),               opts.rho = 1064;                  end
if ~isfield(opts, 'p_inf'),             opts.p_inf = 101325;              end
if ~isfield(opts, 'epsH_rel_frac'),     opts.epsH_rel_frac = 0.10;        end
if ~isfield(opts, 'epsdot_base'),       opts.epsdot_base = 1e5;           end
if ~isfield(opts, 'useEllipticalGate'), opts.useEllipticalGate = true;    end
if ~isfield(opts, 'verbose'),           opts.verbose = true;              end
if ~isfield(opts, 'epsH_abs_floor'),    opts.epsH_abs_floor = 0.00;       end
if ~isfield(opts, 'kappa'),             opts.kappa   = 1;     end
if ~isfield(opts, 'm_floor'),           opts.m_floor = 0.10;  end
end

function [R_d, t_d, material_name] = load_dimensional_data(material_id, opts)
% Load dimensional R(t) data for specified material
% Returns: R_d [nTime x nTrials], t_d [nTime x nTrials], material_name (string)

switch material_id
    case 0  % Synthetic data
        load(fullfile(opts.dataDir,'synthetic_data.mat'));
        lambda_vs_Rmax_fig; close all;
        Rmax_range = linspace(Rmax-.1*Rmax_std,Rmax+.1*Rmax_std,32);
        Req_range = Rmax_range./lambda;
        for i = 1:length(synthetic_data)
            R_d(:,i) = synthetic_data{i}(:,2).*Rmax_range(i);
            t_d(:,i) = linspace(0,150e-6,256);
        end
        material_name = 'synthetic data';
    case 1  % UM1: 10% Gelatin
        load(fullfile(opts.dataDir, 'Processed_Data.mat'), 'expts');
        k = 1;
        for i = 7:15
            R_d(:,k) = expts(i).Roft;
            t_d(:,k) = expts(i).t;
            k = k + 1;
        end
        material_name = 'UM1 (10% Gelatin)';

    case 2  % UM2: 0.2% Fibrin
        load(fullfile(opts.dataDir, 'Processed_Data.mat'), 'expts');
        k = 1;
        for i = 78:98
            R_d(:,k) = expts(i).Roft;
            t_d(:,k) = expts(i).t;
            k = k + 1;
        end
        material_name = 'UM2 (0.2% Fibrin)';

    case 3  % UM3: 8%/0.26% PAAm
        load(fullfile(opts.dataDir, 'Processed_Data.mat'), 'expts');
        k = 1;
        for i = 223:234
            R_d(:,k) = expts(i).Roft;
            t_d(:,k) = expts(i).t;
            k = k + 1;
        end
        material_name = 'UM3 (8%/0.26% PAAm)';

    case 4  % UT1: 10% Gelatin
        data = load(fullfile(opts.dataDir, '10percent', 'Rt_nondim_exp.mat'));
        [R_d, t_d] = convert_from_nondim(data, [1,2,3,4], opts);
        material_name = 'UT1 (10% Gelatin)';

    case 5  % UT2: 5% Agarose
        data = load(fullfile(opts.dataDir, 'Ag5.0', 'Rt_nondim_exp.mat'));
        indices = [1:5, 7:10];
        [R_d, t_d] = convert_from_nondim(data, indices, opts);
        material_name = 'UT2 (5% Agarose)';

    case 6  % Ag2.5
        data = load(fullfile(opts.dataDir, 'Ag2.5', 'Rt_nondim_exp.mat'));
        indices = [1:6, 8:11];
        [R_d, t_d] = convert_from_nondim(data, indices, opts);
        material_name = 'Ag2.5';

    case 7  % Ag0.5
        data = load(fullfile(opts.dataDir, 'Ag0.5', 'Rt_nondim_exp.mat'));
        indices = 2:11;
        [R_d, t_d] = convert_from_nondim(data, indices, opts);
        material_name = 'Ag0.5';

    case 8  % PA1.7
        data = load(fullfile(opts.dataDir, 'PA1pt7', 'Rt_nondim_exp.mat'));
        indices = 1:10;
        [R_d, t_d] = convert_from_nondim(data, indices, opts);
        material_name = 'PA1.7';

    case 9  % PEGDA
        data = load(fullfile(opts.dataDir, 'forbenchmarks', ...
                             'Rt_nondim_exp_PEGDA_S25L6.mat'));
        indices = 1:7;
        [R_d, t_d] = convert_from_nondim(data, indices, opts);
        material_name = 'PEGDA S25L6';

    otherwise
        error('Unknown material_id: %d. Valid range: 1-9', material_id);
end

end

function [R_d, t_d] = convert_from_nondim(data, indices, opts)
% Convert nondimensional data back to dimensional
R_nondim_All = data.R_nondim_All;
t_nondim_All = data.t_nondim_All;
RmaxList     = data.RmaxList;

if isfield(data, 'P_inf')
    P_inf = data.P_inf;
else
    P_inf = opts.p_inf;
end

minLen = min(cellfun(@(x) length(x), R_nondim_All(indices)));

k = 1;
for i = indices
    tc_i       = RmaxList(i) * sqrt(opts.rho / P_inf);
    R_d(:,k)   = R_nondim_All{i}(1:minLen) * RmaxList(i);
    t_d(:,k)   = t_nondim_All{i}(1:minLen) * tc_i;
    k = k + 1;
end

end

function [R_aligned, t_aligned] = align_at_peak(R_d, t_d, opts) %#ok<INUSD>
% Align all trials at their peak radius
[~, startIdx] = max(R_d, [], 1);
nCols  = size(R_d, 2);

tailLen = size(R_d, 1) - startIdx + 1;
minLen  = min(tailLen);

R_aligned = zeros(minLen, nCols);
t_aligned = zeros(minLen, nCols);

for j = 1:nCols
    idx = startIdx(j):(startIdx(j) + minLen - 1);
    R_aligned(:,j) = R_d(idx, j);
    t_aligned(:,j) = t_d(idx, j);

    % Reset time to start at zero
    t_aligned(:,j) = t_aligned(:,j) - min(t_aligned(:,j));
end

end

function Rdotmatrix = compute_velocities(Rmatrix, tmatrix, tc, R_d, Rmax_range)
% Compute bubble wall velocities (try dataprep first, else gradient)

if exist('dataprep', 'file') == 2
    try
        [~,~,~,~,~,~,Rdotmatrix,~,~,~] = ...
            dataprep(tmatrix, Rmatrix, tc, R_d, Rmax_range);
        return;
    catch
        % Fall through to gradient
    end
end

% Fallback: simple gradient in nondimensional coordinates
Rdotmatrix = zeros(size(Rmatrix));
for j = 1:size(Rmatrix, 2)
    Rdotmatrix(:,j) = gradient(Rmatrix(:,j), tmatrix(:,j));
end

end

function [strain, strainRate] = compute_strain_quantities(R_d, Rdotmatrix, ...
                                                         Rmatrix, Req_each, tc, opts) %#ok<INUSD>
% Compute Hencky strain and strain rate (Paper Eq. 4)

[nTime, nTrials] = size(Rmatrix);

% Equilibrium-referenced log strain: ε* = 0.5 * log((R/Req)^(-4))
strain = zeros(nTime, nTrials);
for j = 1:nTrials
    denom = max(Req_each(j), 1e-12);
    strain(:,j) = 0.5 * log(max((R_d(:,j) ./ denom).^(-4), 1e-12));
end

% Strain rate: ε̇* = -2 (Ṙ/R) * tc (Paper Eq. 4)
tc_row     = repmat(tc, nTime, 1);
strainRate = -2 .* (Rdotmatrix ./ max(Rmatrix, 1e-12)) .* tc_row;

% Clean infinities / NaNs
strain(~isfinite(strain))       = NaN;
strainRate(~isfinite(strainRate)) = NaN;

end

function mask = compute_high_info_mask(strain, strainRate, tc, opts)
% Apply high-information gate (Paper Eq. 5-6)

[nTime, nTrials] = size(strain);

% Thresholds per trial (Paper Eq. 5)
epsH_max_per_trial = max(abs(strain), [], 1, 'omitnan');
epsH_th_trial = max(opts.epsH_abs_floor, ...
                    opts.epsH_rel_frac .* epsH_max_per_trial);
epsH_th_mat        = repmat(epsH_th_trial, nTime, 1);

epsdot_th_trial = opts.epsdot_base * tc;
epsdot_th_mat   = repmat(epsdot_th_trial, nTime, 1);

% Base mask: finite values
mask = isfinite(strain) & isfinite(strainRate);

% Elliptical gate (Paper Eq. 6)
if opts.useEllipticalGate
    inHigh = ((abs(strain)    ./ max(epsH_th_mat,   eps)).^2 + ...
              (abs(strainRate)./ max(epsdot_th_mat, eps)).^2) >= 1;
else
    % Rectangular gate
    inHigh = (abs(strain)    >= epsH_th_mat) | ...
             (abs(strainRate) >= epsdot_th_mat);
end

mask = mask & inHigh;

end
