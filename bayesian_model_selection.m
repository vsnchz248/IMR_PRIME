function results = bayesian_model_selection(modelLibrary, expData, cfg)
% BAYESIAN_MODEL_SELECTION  Hierarchical Bayesian model comparison with GPR
%
% Syntax:
%   results = bayesian_model_selection(modelLibrary, expData, cfg)
%
% Inputs:
%   modelLibrary - Cell array of model specifications, each with fields:
%       .name          : Model name ('Newt', 'NH', 'KV', 'qNH', 'LM', 'qKV', 'SLS')
%       .paramNames    : Cell array of parameter names
%       .paramBounds   : [nParams x 2] array of bounds
%       .forwardSolver : Function handle @(params, expData) -> simData
%   expData - Structure from expDataPrep
%   cfg - Configuration structure (see below for fields)
%
% Outputs:
%   results - Structure containing:
%       .Models        : Array of model results (logZ, beta_MAP, etc.)
%       .posterior     : Posterior probabilities over models
%       .bestModel     : Index of most plausible model
%       .modelNames    : Names of all models
%
% Configuration (cfg) fields:
%   GPR Active Learning:
%     .gpr_n0, .gpr_maxRounds, .gpr_maxAdded, .gpr_tolRelCI
%   
%   Likelihood:
%     .useHetero, .kappa, .m_floor, .betaGrid, .useRdotInLL
%   
%   Priors:
%     .useBICprior   : Enable BIC-like complexity penalty (default: true)
%     .priors        : Prior structure from build_model_priors()
%   
%   Thresholds:
%     .Thresholds    : Strain-based gating parameters
%
% Example:
%   % Define models
%   modelLibrary = {
%       struct('name', 'Newt', 'paramNames', {{'mu'}}, ...
%              'paramBounds', [1e-4 1], 'forwardSolver', @solve_Newt),
%       struct('name', 'KV', 'paramNames', {{'mu', 'G'}}, ...
%              'paramBounds', [1e-4 1; 1e2 1e5], 'forwardSolver', @solve_KV)
%   };
%   
%   % Run model selection
%   results = bayesian_model_selection(modelLibrary, expData, cfg);
%   
%   fprintf('Best model: %s (posterior=%.4f)\n', ...
%           results.modelNames{results.bestModel}, ...
%           results.posterior(results.bestModel));
%
% See also: gpr_active_learning, build_likelihood_evaluator, build_model_priors

% Author: [Your name]
% Date: 2025

%% Initialize
tic;
nModels = numel(modelLibrary);

% Default configuration
cfg = set_default_config(cfg);

% Build likelihood evaluator (shared across all models)
likelihoodEval = build_likelihood_evaluator(expData, cfg);

% Load priors if provided
if isfield(cfg, 'priors') && ~isempty(cfg.priors)
    priors = cfg.priors;
else
    priors = struct();  % Will use uniform priors
end

% Effective sample size for BIC prior
mask = expData.mask;
cfg.N_eff = 2 * nnz(mask);

%% Process each model with GPR-based active learning
fprintf('\n=== Bayesian Model Selection with GPR ===\n');
fprintf('Number of models: %d\n', nModels);
fprintf('Effective observations: %d\n', cfg.N_eff);

Models = repmat(empty_model_struct(), 1, nModels);

for i = 1:nModels
    modelSpec = modelLibrary{i};
    modelName = modelSpec.name;
    
    fprintf('\n--- Processing Model %d/%d: %s ---\n', i, nModels, modelName);
    
    % Build forward solver wrapper that returns log-likelihood
    modelSpec.forwardSolver = wrap_solver_with_likelihood(...
        modelSpec.forwardSolver, likelihoodEval, expData);
    
    % Run GPR active learning to estimate model evidence
    gprResults = gpr_active_learning(modelSpec, expData, cfg.gpr_opts);
    
    % Extract results
    Models(i).name = modelName;
    Models(i).logZ = gprResults.I_estimate;
    Models(i).logZ_uncertainty = gprResults.I_uncertainty;
    Models(i).k_params = numel(modelSpec.paramNames);
    Models(i).nSamples = gprResults.addedTotal + cfg.gpr_opts.n0;
    Models(i).gpr = gprResults.gpr;
    Models(i).X_train = gprResults.X_train;
    Models(i).Y_train = gprResults.Y_train;
    
    % Find MAP parameters from GPR
    [~, idx_MAP] = max(gprResults.Y_train);
    Models(i).theta_MAP = gprResults.X_train(idx_MAP, :);
    Models(i).logL_MAP = gprResults.Y_train(idx_MAP);
    
    % Compute beta posterior at MAP
    simData = modelSpec.forwardSolver_orig(Models(i).theta_MAP, expData);
    beta_post = likelihoodEval.getBetaPosterior(simData);
    [~, idx_beta] = max(beta_post);
    Models(i).beta_MAP = cfg.betaGrid(idx_beta);
    Models(i).beta_mean = sum(cfg.betaGrid(:) .* beta_post(:));
    
    fprintf('  Evidence: logZ = %.5f ± %.3g\n', Models(i).logZ, Models(i).logZ_uncertainty);
    fprintf('  MAP parameters: [%s]\n', sprintf('%.3g ', Models(i).theta_MAP));
    fprintf('  Beta MAP: %.3f | Beta mean: %.3f\n', Models(i).beta_MAP, Models(i).beta_mean);
end

%% Compute model priors
logpM = compute_model_priors(Models, priors, cfg);

%% Compute posterior over models
logZ_vec = [Models.logZ];
x = logZ_vec + logpM;
Fmask = isfinite(x);

if ~any(Fmask)
    posterior = ones(1, nModels) / nModels;
    warning('All model evidences are non-finite. Using uniform posterior.');
else
    Z = safe_logsumexp(x(Fmask));
    posterior = zeros(1, nModels);
    posterior(Fmask) = exp(x(Fmask) - Z);
end

%% Display results
fprintf('\n=== Model Selection Results ===\n');
for i = 1:nModels
    fprintf('%-8s | logZ=%+10.4f | Post=%.4f | k=%d | nSamp=%4d | beta_MAP=%.3f\n', ...
            Models(i).name, Models(i).logZ, posterior(i), ...
            Models(i).k_params, Models(i).nSamples, Models(i).beta_MAP);
end

[~, bestIdx] = max(posterior);
fprintf('\nMost plausible model: %s (posterior=%.4f)\n', ...
        Models(bestIdx).name, posterior(bestIdx));

fprintf('\nTotal time: %.2f seconds\n', toc);

%% Package results
results = struct(...
    'Models', Models, ...
    'posterior', posterior, ...
    'logZ_vec', logZ_vec, ...
    'logpM', logpM, ...
    'bestModel', bestIdx, ...
    'modelNames', {arrayfun(@(m) m.name, Models, 'UniformOutput', false)}, ...
    'cfg', cfg, ...
    'expData', expData);

end

%% ==================== Helper Functions ====================

function cfg = set_default_config(cfg)
% Set default configuration values

% GPR active learning defaults
if ~isfield(cfg, 'gpr_opts'), cfg.gpr_opts = struct(); end
if ~isfield(cfg.gpr_opts, 'verbose')
    cfg.gpr_opts.verbose = true;
end

% Likelihood defaults
if ~isfield(cfg, 'useHetero'),   cfg.useHetero = true;     end
if ~isfield(cfg, 'kappa'),       cfg.kappa = 1.0;          end
if ~isfield(cfg, 'm_floor'),     cfg.m_floor = 0.10;       end
if ~isfield(cfg, 'useRdotInLL'), cfg.useRdotInLL = true;   end

% Beta grid
if ~isfield(cfg, 'betaGrid')
    cfg.betaGrid = 0.05:0.05:10.0;
end

% Thresholds
if ~isfield(cfg, 'Thresholds')
    cfg.Thresholds = struct(...
        'mode', 'auto', ...
        'auto_base', 1e5, ...
        'epsH_rel_frac', 0.10, ...
        'epsH_abs_floor', 0.00);
end

% Priors
if ~isfield(cfg, 'useBICprior'), cfg.useBICprior = true; end

end

function S = empty_model_struct()
% Template for model results
S = struct(...
    'name', '', ...
    'logZ', NaN, ...
    'logZ_uncertainty', NaN, ...
    'k_params', NaN, ...
    'nSamples', NaN, ...
    'theta_MAP', [], ...
    'logL_MAP', NaN, ...
    'beta_MAP', NaN, ...
    'beta_mean', NaN, ...
    'gpr', [], ...
    'X_train', [], ...
    'Y_train', []);
end

function solver = wrap_solver_with_likelihood(originalSolver, likelihoodEval, expData)
% Wrap forward solver to return log-likelihood instead of simData
solver = @(params) wrapper(params, originalSolver, likelihoodEval, expData);
% Store original for later use
solver.forwardSolver_orig = originalSolver;
end

function logL = wrapper(params, originalSolver, likelihoodEval, expData)
% Call forward solver and compute log-likelihood
simData = originalSolver(params, expData);
logL = likelihoodEval.computeLogL(simData);
end

function logpM = compute_model_priors(Models, priors, cfg)
% Compute prior probability for each model
nModels = numel(Models);
logpM = zeros(1, nModels);

% Data-driven axis needs (from priors if available)
if isfield(priors, 'axis_need')
    e_need = getfield_safe(priors.axis_need, 'elastic', 0);
    m_need = getfield_safe(priors.axis_need, 'maxwell', 0);
    nl_need = getfield_safe(priors.axis_need, 'nonlinear', 0);
else
    e_need = 0.5;   % Neutral defaults
    m_need = 0.5;
    nl_need = 0.5;
end

e_need = max(min(e_need, 1), 0);
m_need = max(min(m_need, 1), 0);
nl_need = max(min(nl_need, 1), 0);

% Model-specific data-driven priors (knob-free)
for i = 1:nModels
    switch lower(Models(i).name)
        case {'newtonian', 'newt'}
            f = 1;  % Baseline
        case 'nh'
            f = e_need;
        case {'kv', 'nhkv'}
            f = e_need;
        case 'qnh'
            f = e_need * nl_need;
        case 'qkv'
            f = e_need * nl_need;
        case {'linmax', 'lm', 'max'}
            f = m_need;
        case 'sls'
            f = (e_need^2) * m_need;
        otherwise
            f = 1;
    end
    logpM_data = log(max(f, 1e-12));
    
    % BIC-like complexity penalty
    if cfg.useBICprior
        logpM_BIC = -0.5 * Models(i).k_params * log(max(cfg.N_eff, 1));
        logpM(i) = logpM_BIC + logpM_data;
    else
        logpM(i) = logpM_data;
    end
end

end

function v = getfield_safe(s, f, default)
% Safe field access with default
if isfield(s, f)
    v = s.(f);
else
    v = default;
end
end

function s = safe_logsumexp(a)
% Stable log-sum-exp
if isempty(a) || all(~isfinite(a))
    s = -inf;
else
    amax = max(a(:));
    s = amax + log(sum(exp(a(:) - amax)));
end
end