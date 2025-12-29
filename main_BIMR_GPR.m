function main_BIMR_GPR(material_id, model_name, opts)
% MAIN_BIMR_GPR  Bayesian IMR with GPR-accelerated evidence integration
%
% Syntax:
%   main_BIMR_GPR(material_id, model_name)
%   main_BIMR_GPR(material_id, model_name, opts)
%
% Inputs:
%   material_id - Integer (0=synthetic, 1=UM1, 2=UM2, etc.)
%   model_name  - String: 'newtonian', 'nh', 'kv', 'qnh', 'linmax', 'qkv', 'sls'
%   opts        - Optional settings struct
%
% Example:
%   % Test on synthetic Newtonian data
%   main_BIMR_GPR(0, 'newtonian');
%
%   % Run UM1 gelatin with qKV model
%   main_BIMR_GPR(1, 'qkv');

if nargin < 3, opts = struct(); end

%% Parse options
if ~isfield(opts, 'dataDir'),     opts.dataDir = './experiments';      end
if ~isfield(opts, 'solverDir'),   opts.solverDir = '../IMRv2/src/forward_solver'; end
if ~isfield(opts, 'maxRounds'),   opts.maxRounds = 50;                 end
if ~isfield(opts, 'tolRelCI'),    opts.tolRelCI = 0.05;                end
if ~isfield(opts, 'verbose'),     opts.verbose = true;                 end
if ~isfield(opts, 'savePath'),    opts.savePath = sprintf('results_%s_mat%d.mat', model_name, material_id); end

% Add paths
addpath(opts.dataDir);
if exist(opts.solverDir, 'dir') == 7
    addpath(opts.solverDir);
end

fprintf('\n========================================\n');
fprintf('  Bayesian IMR with GPR Acceleration\n');
fprintf('========================================\n');
fprintf('Material:  %d\n', material_id);
fprintf('Model:     %s\n', upper(model_name));
fprintf('Max rounds: %d\n', opts.maxRounds);
fprintf('========================================\n\n');

%% 1. Load and prepare experimental data
tic;
fprintf('[1/4] Loading experimental data...\n');
opts_data = struct('dataDir', opts.dataDir, 'verbose', false, ...
                   'kappa', 1, 'm_floor', 0.10);
expData = prepare_data(material_id, opts_data);
fprintf('   Loaded %d trials, %d time points\n', ...
        size(expData.Rmatrix, 2), size(expData.Rmatrix, 1));
fprintf('   High-info gate: %.1f%% points retained\n', ...
        100*nnz(expData.mask)/numel(expData.mask));
t_data = toc;

%% 2. Build priors with redundancy GPR
tic;
fprintf('\n[2/4] Building priors (with redundancy GPR)...\n');
opts_prior = struct('quiet', false, 'precompute_redundancy', true, ...
                    'N_redundancy_samples', 8192);
priors = build_model_priors_gpr(expData, opts_prior);
fprintf('   Priors built for all models\n');
t_prior = toc;

%% 3. Define NLL function and bounds
fprintf('\n[3/4] Setting up model "%s"...\n', upper(model_name));

% Get parameter bounds
ranges = priors.ranges;
[xmin, xmax, param_names] = get_model_bounds(model_name, ranges);
d = numel(xmin);

fprintf('  Parameters (%dD): %s\n', d, strjoin(param_names, ', '));
fprintf('  Bounds:\n');
for i = 1:d
    fprintf('    %8s: [%.2e, %.2e]\n', param_names{i}, xmin(i), xmax(i));
end

% Wrap NLL computation
funNLL = @(theta) compute_nll(theta, model_name, expData, opts.solverDir);

%% 4. Run GPR-based Bayesian quadrature
tic;
fprintf('\n[4/4] Running active Bayesian quadrature...\n');

opts_gpr = struct();
opts_gpr.maxRounds  = opts.maxRounds;
opts_gpr.tolRelCI   = opts.tolRelCI;
opts_gpr.verbose    = opts.verbose;
opts_gpr.rngSeed    = 42;

% Pass priors and model name for prior evaluation
opts_gpr.priors     = priors;
opts_gpr.modelName  = model_name;
opts_gpr.expData    = expData;  % Needed for redundancy

out = active_integrate_logaware(funNLL, xmin, xmax, opts_gpr);
t_gpr = toc;

%% 5. Display results
fprintf('\n========================================\n');
fprintf('  RESULTS: %s\n', upper(model_name));
fprintf('========================================\n');
fprintf('Evidence:\n');
fprintf('  log10(Z) = %.6f\n', out.log10I_mean);
fprintf('  95%% CI   = [%.6f, %.6f]\n', out.log10I_CI95(1), out.log10I_CI95(2));
fprintf('\nMAP Parameters:\n');
theta_MAP = out.fromFeat(out.U(out.mapIdx,:));
for i = 1:d
    fprintf('  %8s = %.4e\n', param_names{i}, theta_MAP(i));
end
fprintf('\nDiagnostics:\n');
fprintf('  Total points:   %d\n', size(out.U, 1));
fprintf('  GP Ã (model):   %.4f\n', out.sigma_model);
fprintf('  GP Ã (RQMC):    %.4f\n', out.sigma_rqmc);
fprintf('  NLL reference:  %.6e\n', out.NLL_ref);

fprintf('\nTiming:\n');
fprintf('  Data prep:      %.2f s\n', t_data);
fprintf('  Prior build:    %.2f s\n', t_prior);
fprintf('  GPR integration:%.2f s\n', t_gpr);
fprintf('  Total:          %.2f s\n', t_data + t_prior + t_gpr);
fprintf('========================================\n\n');

%% 6. Save results
results = struct();
results.material_id   = material_id;
results.model_name    = model_name;
results.expData       = expData;
results.priors        = priors;
results.evidence      = out;
results.theta_MAP     = theta_MAP;
results.param_names   = param_names;
results.timing        = struct('data', t_data, 'prior', t_prior, 'gpr', t_gpr);

% save(opts.savePath, '-struct', 'results');
% fprintf('Results saved to: %s\n\n', opts.savePath);

end

%% ==================== Helper Functions ====================

function [xmin, xmax, param_names] = get_model_bounds(model_name, ranges)
% Get parameter bounds for specified model

switch lower(model_name)
    case {'newtonian', 'newt'}
        xmin = ranges.mu(1);
        xmax = ranges.mu(2);
        param_names = {'mu'};
        
    case 'nh'
        xmin = ranges.G(1);
        xmax = ranges.G(2);
        param_names = {'G'};
        
    case 'kv'
        xmin = [ranges.mu(1), ranges.G(1)];
        xmax = [ranges.mu(2), ranges.G(2)];
        param_names = {'mu', 'G'};
        
    case 'qnh'
        xmin = [ranges.G(1), ranges.alpha(1)];
        xmax = [ranges.G(2), ranges.alpha(2)];
        param_names = {'G', 'alpha'};
        
    case {'linmax', 'max', 'lm'}
        xmin = [ranges.mu(1), ranges.lambda1(1)];
        xmax = [ranges.mu(2), ranges.lambda1(2)];
        param_names = {'mu', 'lambda1'};
        
    case 'qkv'
        xmin = [ranges.mu(1), ranges.G(1), ranges.alpha(1)];
        xmax = [ranges.mu(2), ranges.G(2), ranges.alpha(2)];
        param_names = {'mu', 'G', 'alpha'};
        
    case 'sls'
        xmin = [ranges.mu(1), ranges.G(1), ranges.lambda1(1)];
        xmax = [ranges.mu(2), ranges.G(2), ranges.lambda1(2)];
        param_names = {'mu', 'G', 'lambda1'};
        
    otherwise
        error('Unknown model: %s', model_name);
end

end