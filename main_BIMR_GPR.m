function results_all = main_BIMR_GPR(material_id, model_name, opts)
% MAIN_BIMR_GPR  Bayesian IMR with GPR-accelerated evidence integration
%
% Syntax:
%   main_BIMR_GPR(material_id)
%   main_BIMR_GPR(material_id, model_name)
%   main_BIMR_GPR(material_id, model_name, opts)
%
% model_name can be:
%   - char/string: 'newtonian'
%   - cell array: {'newtonian','kv'}
%   - string array: ["newtonian","kv"]
%   - empty [] or omitted => runs ALL models

t_total_start = tic;

if nargin < 2, model_name = []; end
if nargin < 3, opts = struct(); end

%% Parse options
if ~isfield(opts, 'dataDir'),     opts.dataDir = './experiments'; end
if ~isfield(opts, 'solverDir'),   opts.solverDir = '../IMRv2/src/forward_solver'; end
if ~isfield(opts, 'maxRounds'),   opts.maxRounds = 50; end
if ~isfield(opts, 'tolRelCI'),    opts.tolRelCI = 0.05; end
if ~isfield(opts, 'verbose'),     opts.verbose = true; end
if ~isfield(opts, 'rngSeed'),     opts.rngSeed = 42; end

% Default: save a combined file if running multiple models
if ~isfield(opts, 'saveCombined'), opts.saveCombined = true; end
if ~isfield(opts, 'savePath')
    opts.savePath = sprintf('results_ALL_mat%d.mat', material_id);
end

% Add paths
addpath(opts.dataDir);
if exist(opts.solverDir, 'dir') == 7
    addpath(opts.solverDir);
end

%% Decide which models to run
all_models = {'newtonian','nh','kv','qnh','linmax','qkv','sls'};

if isempty(model_name)
    model_list = all_models;
else
    if ischar(model_name) || (isstring(model_name) && isscalar(model_name))
        model_list = {char(model_name)};
    elseif isstring(model_name)
        model_list = cellstr(model_name(:));
    elseif iscell(model_name)
        model_list = model_name(:);
    else
        error('model_name must be a string/char, cell array, string array, or empty.');
    end
end

% Normalize to lowercase
for i = 1:numel(model_list)
    model_list{i} = lower(string(model_list{i}));
    model_list{i} = char(model_list{i});
end

% Validate
bad = setdiff(model_list, all_models);
if ~isempty(bad)
    error('Unknown model(s): %s', strjoin(bad, ', '));
end

fprintf('\n========================================\n');
fprintf('  Bayesian IMR with GPR Acceleration\n');
fprintf('========================================\n');
fprintf('Material:   %d\n', material_id);
fprintf('Models:     %s\n', strjoin(upper(model_list), ', '));
fprintf('Max rounds: %d\n', opts.maxRounds);
fprintf('tolRelCI:   %.3g\n', opts.tolRelCI);
fprintf('========================================\n\n');

%% 1) Load and prepare experimental data ONCE
tic;
fprintf('[1/3] Loading experimental data...\n');
opts_data = struct('dataDir', opts.dataDir, 'verbose', false, ...
                   'kappa', 1, 'm_floor', 0.10);
expData = prepare_data(material_id, opts_data);
fprintf('  ✓ Loaded %d trials, %d time points\n', ...
        size(expData.Rmatrix, 2), size(expData.Rmatrix, 1));
fprintf('  ✓ High-info gate: %.1f%% points retained\n', ...
        100*nnz(expData.mask)/numel(expData.mask));
t_data = toc;

%% 2) Build priors ONCE (shared across models)
tic;
fprintf('\n[2/3] Building priors (with redundancy GPR)...\n');
opts_prior = struct('quiet', false, 'precompute_redundancy', true, ...
                    'N_redundancy_samples', 8192);
priors = build_model_priors_gpr(expData, opts_prior);
fprintf('  ✓ Priors built\n');
t_prior = toc;

%% 3) Loop models and run evidence integration
results_all = struct();
results_all.material_id = material_id;
results_all.expData     = expData;
results_all.priors      = priors;
results_all.models      = model_list;

% Pre-allocate per_model cell array instead of struct array
results_all.per_model = cell(numel(model_list), 1);

ranges = priors.ranges;

for m = 1:numel(model_list)
    model = model_list{m};

    fprintf('\n========================================\n');
    fprintf('  MODEL %d/%d: %s\n', m, numel(model_list), upper(model));
    fprintf('========================================\n');

    % Bounds
    [xmin, xmax, param_names] = get_model_bounds(model, ranges);
    d = numel(xmin);

    fprintf('  Parameters (%dD): %s\n', d, strjoin(param_names, ', '));
    fprintf('  Bounds:\n');
    for i = 1:d
        fprintf('    %8s: [%.2e, %.2e]\n', param_names{i}, xmin(i), xmax(i));
    end

    % NLL wrapper (must accept theta and beta_grid)
    funNLL = @(theta, beta_grid) compute_nll(theta, model, expData, opts.solverDir, beta_grid);

    % GPR integration opts
    opts_gpr = struct();
    opts_gpr.maxRounds = opts.maxRounds;
    opts_gpr.tolRelCI  = opts.tolRelCI;
    opts_gpr.verbose   = opts.verbose;
    opts_gpr.rngSeed   = opts.rngSeed;
    opts_gpr.Nint_final_override = 8192;
    opts_gpr.R_final_override    = 4;

    % Needed for prior + redundancy logic inside active_integrate_logaware
    opts_gpr.priors    = priors;
    opts_gpr.modelName = model;
    opts_gpr.expData   = expData;

    % Run
    tic;
    out = active_integrate_logaware(funNLL, xmin, xmax, opts_gpr);
    t_gpr = toc;

    % Report
    fprintf('\nEvidence:\n');
    fprintf('  log10(Z) = %.6f\n', out.log10I_mean);
    fprintf('  95%% CI   = [%.6f, %.6f]\n', out.log10I_CI95(1), out.log10I_CI95(2));

    theta_MAP = out.fromFeat(out.U(out.mapIdx,:));
    fprintf('\nMAP Parameters:\n');
    for i = 1:d
        fprintf('  %8s = %.4e\n', param_names{i}, theta_MAP(i));
    end

    fprintf('\nDiagnostics:\n');
    fprintf('  Total points:   %d\n', size(out.U, 1));
    fprintf('  GP σ (model):   %.4f\n', out.sigma_model);
    fprintf('  GP σ (RQMC):    %.4f\n', out.sigma_rqmc);
    fprintf('  NLL reference:  %.6e\n', out.NLL_ref);

    fprintf('\nTiming:\n');
    fprintf('  GPR integration: %.2f s\n', t_gpr);

    % Store
    pm = struct();
    pm.model_name  = model;
    pm.param_names = param_names;
    pm.theta_MAP   = theta_MAP;
    pm.evidence    = out;
    pm.timing      = struct('gpr', t_gpr);

    results_all.per_model{m} = pm;

    % Optional per-model save
    % if isfield(opts,'saveEach') && opts.saveEach
    %     savePath_i = sprintf('results_%s_mat%d.mat', model, material_id);
    %     results_i = pm; %#ok<NASGU>
    %     save(savePath_i, 'results_i');
    %     fprintf('  Saved per-model: %s\n', savePath_i);
    % end
end

results_all.timing = struct('data', t_data, 'prior', t_prior);

%% 4) Model comparison (if multiple models)
if numel(model_list) > 1
    fprintf('\n========================================\n');
    fprintf('  MODEL COMPARISON\n');
    fprintf('========================================\n');
    
    % Extract log10 evidence values
    log10_Z = zeros(numel(model_list), 1);
    for m = 1:numel(model_list)
        log10_Z(m) = results_all.per_model{m}.evidence.log10I_mean;
    end
    
    % Apply model prior P(M_i) = exp(-k_M/2 × log(N_eff)) [BIMR Eq. 28]
    kM_vec = zeros(numel(model_list), 1);
    for m = 1:numel(model_list)
        kM_vec(m) = numel(results_all.per_model{m}.param_names);
    end
    
    N_eff = 2 * nnz(expData.mask);  % Two observables: R*, Ṙ*
    log_model_prior = -(kM_vec / 2) * log(N_eff);  % Natural log
    log10_model_prior = log_model_prior / log(10);  % Convert to log10
    
    fprintf('\nModel Priors (BIC-motivated Occam penalty):\n');
    for m = 1:numel(model_list)
        fprintf('  %s (d=%d): log10 P(M) = %.4f\n', ...
            upper(model_list{m}), kM_vec(m), log10_model_prior(m));
    end
    
    % Add model prior to evidence
    log10_Z_with_prior = log10_Z + log10_model_prior;
    
    % Convert to natural log for probability calculation
    ln_Z = log10_Z_with_prior * log(10);
    
    % Compute posterior probabilities
    ln_Z_max = max(ln_Z);
    ln_Z_shifted = ln_Z - ln_Z_max;  % For numerical stability
    Z_rel = exp(ln_Z_shifted);
    posterior_probs = Z_rel / sum(Z_rel);
    
    % Sort by posterior probability
    [~, idx_sorted] = sort(posterior_probs, 'descend');
    
    fprintf('\nModel Rankings:\n');
    fprintf('%-12s  %12s  %12s  %12s  %12s  %12s\n', ...
        'Model', 'log10(P(D|M))', 'log10(P(M))', 'log10(Post)', 'Posterior', 'Rank');
    fprintf('%s\n', repmat('-', 1, 80));
    for i = 1:numel(model_list)
        m = idx_sorted(i);
        fprintf('%-12s  %12.4f  %12.4f  %12.4f  %12.6f  %12d\n', ...
            upper(model_list{m}), log10_Z(m), log10_model_prior(m), ...
            log10_Z_with_prior(m), posterior_probs(m), i);
    end
    
    % Winner
    winner_idx = idx_sorted(1);
    fprintf('\n>>> WINNING MODEL: %s <<<\n', upper(model_list{winner_idx}));
    fprintf('    Posterior probability: %.4f (%.1f%%)\n', ...
        posterior_probs(winner_idx), 100*posterior_probs(winner_idx));
    
    % Store comparison results
    results_all.comparison = struct();
    results_all.comparison.log10_evidence = log10_Z;
    results_all.comparison.log10_model_prior = log10_model_prior;
    results_all.comparison.log10_posterior_unnorm = log10_Z_with_prior;
    results_all.comparison.posterior_probs = posterior_probs;
    results_all.comparison.winner = model_list{winner_idx};
    results_all.comparison.winner_prob = posterior_probs(winner_idx);
    results_all.comparison.N_eff = N_eff;
end

fprintf('\n========================================\n');
fprintf('  DONE (material %d)\n', material_id);
fprintf('========================================\n');
fprintf('Timing:\n');
fprintf('  Data prep:   %.2f s\n', t_data);
fprintf('  Prior build: %.2f s\n', t_prior);
fprintf('  Total:       %.2f s\n', toc(t_total_start));
fprintf('========================================\n\n');

% Save combined (default)
% if opts.saveCombined
%     save(opts.savePath, 'results_all', '-v7.3');
%     fprintf('Saved combined results to: %s\n\n', opts.savePath);
% end

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