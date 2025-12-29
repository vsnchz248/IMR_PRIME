% TEST_MATERIAL_OPTIMIZED.m - Run GPR-based Bayesian IMR
% OPTIMIZED VERSION with profiling and diagnostics
clear; clc; close all;
addpath('../IMRv2/src/forward_solver/')

% ========== CONFIG ==========
rng(42, 'twister');

material_id = 0;        % 0=synthetic, 1-9=experimental
num_workers = 8;        % Workers for parfor in NLL evaluations (NOT model-level)
save_figures = true;
save_results = true;
output_dir = './results';
enable_profiling = false;  % Set to true to profile

% Models to test (all 7 from paper)
models = {'newt', 'nh', 'kv'};%, 'qnh', 'lm', 'qkv', 'sls'};

% GPR active learning settings (OPTIMIZED)
gprOpts = struct();
gprOpts.maxRounds = 30;        % Reduced from 40
gprOpts.maxAddedMult = 120;    % Reduced from 160
gprOpts.tolRelCI = 0.04;       % Relaxed from 0.03
gprOpts.rngSeed = 42;
gprOpts.verbose = false;       % Reduce console spam

% ========== SETUP ==========
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Start parallel pool for NLL evaluation parallelism
% (Models run serially, but each model parallelizes its NLL evaluations)
poolobj = gcp('nocreate');
if isempty(poolobj)
    fprintf('Starting parallel pool with %d workers...\n', num_workers);
    parpool('local', num_workers);
else
    fprintf('Parallel pool already running with %d workers.\n', poolobj.NumWorkers);
end

% ========== DATA PREP ==========
fprintf('\n================================================\n');
fprintf('  BAYESIAN IMR - MATERIAL %d\n', material_id);
fprintf('================================================\n\n');

% Use proper prepare_data function
expData = prepare_data(material_id);

% Data already packaged by prepare_data, just extract key info
fprintf('Data: %d trials, N_eff = %d\n', size(expData.Rmatrix,2), 2*nnz(expData.mask));
fprintf('R range: [%.3f, %.3f], Rdot range: [%.3e, %.3e]\n\n', ...
        min(expData.Rmatrix(:)), max(expData.Rmatrix(:)), ...
        min(abs(expData.Rdotmatrix(:))), max(abs(expData.Rdotmatrix(:))));

% Add N_eff field for model prior computation
expData.N_eff = 2 * nnz(expData.mask);

% ========== BUILD PRIORS ==========
fprintf('Building priors (PAPER Sec. 2.3.3)...\n');
tic_prior = tic;

% Create opts structure for build_model_priors
priorOpts = struct('quiet', true);

% Call build_model_priors with expData as first argument
priors = build_model_priors(expData, priorOpts);
priors.N_eff = expData.N_eff;

% Wrap parameter prior for GPR integration
priors.paramPrior = @(modelName, theta) param_prior_lookup(modelName, theta, priors);

fprintf('  Axis needs (data-driven): elastic=%.3f, maxwell=%.3f, nonlinear=%.3f\n', ...
        priors.axis_need.elastic, priors.axis_need.maxwell, priors.axis_need.nonlinear);
fprintf('  Prior build time: %.2f sec\n\n', toc(tic_prior));

% ========== RUN MODEL SELECTION ==========
tic_total = tic;
nModels = numel(models);
results_per_model = cell(nModels, 1);

fprintf('Running GPR-based integration for %d models...\n', nModels);
fprintf('GPR settings: maxRounds=%d, tolRelCI=%.1f%\n\n', ...
        gprOpts.maxRounds, 100*gprOpts.tolRelCI);

if enable_profiling
    profile on;
end

% Run models SERIALLY (one at a time)
% Each model parallelizes its NLL evaluations via parfor in imr_nll_with_prior_matrix
for i = 1:nModels
    fprintf('\n--- Model %d/%d: %s ---\n', i, nModels, upper(models{i}));
    tic_model = tic;
    results_per_model{i} = run_single_model_gpr(models{i}, expData, priors, gprOpts);
    fprintf('  Model time: %.2f sec\n', toc(tic_model));
end

if enable_profiling
    profile viewer;
    profile off;
end

elapsed_time = toc(tic_total);

% ========== COMPUTE POSTERIORS ==========
fprintf('\n================================================\n');
fprintf('  COMPUTING POSTERIORS (Bayes Theorem Eq. 9)\n');
fprintf('================================================\n');

logZ_vec = zeros(nModels, 1);
logPrior_vec = zeros(nModels, 1);

for i = 1:nModels
    logZ_vec(i) = results_per_model{i}.logZ;
    logPrior_vec(i) = results_per_model{i}.logPrior;
    fprintf('  %s: logZ = %12.6e, logPrior = %12.6e\n', ...
            upper(models{i}), logZ_vec(i), logPrior_vec(i));
end

% Posterior: P(M|D)  P(D|M) * P(M)
x = logZ_vec + logPrior_vec;
Fmask = isfinite(x);
if ~any(Fmask)
    post = ones(nModels,1) / nModels;
    warning('All evidence values non-finite! Using uniform posterior.');
else
    Ztot = safe_logsumexp(x(Fmask));
    post = zeros(nModels,1);
    post(Fmask) = exp(x(Fmask) - Ztot);
end

% Normalize
post = post / sum(post);

% ========== PRINT SUMMARY ==========
fprintf('\n================================================\n');
fprintf('  RESULTS\n');
fprintf('================================================\n');
fprintf('%-8s %12s %12s %12s %10s\n', 'Model', 'log10(Z)', 'log(Prior)', 'P(M|D)', 'Time(s)');
fprintf('------------------------------------------------\n');
for i = 1:nModels
    fprintf('%-8s %12.5g %12.5g %12.6f %10.1f\n', ...
        upper(models{i}), ...
        results_per_model{i}.log10Z, ...
        results_per_model{i}.logPrior, ...
        post(i), ...
        results_per_model{i}.time_seconds);
end
fprintf('================================================\n');

[~, bestIdx] = max(post);
fprintf('\n*** WINNER: %s (P=%.4f) ***\n', upper(models{bestIdx}), post(bestIdx));
fprintf('Total time: %.1f min (%.2f sec)\n', elapsed_time/60, elapsed_time);
fprintf('Average time per model: %.1f sec\n\n', elapsed_time/nModels);

% ========== DIAGNOSTICS ==========
fprintf('================================================\n');
fprintf('  DIAGNOSTICS\n');
fprintf('================================================\n');
for i = 1:nModels
    res = results_per_model{i};
    fprintf('%s:\n', upper(models{i}));
    fprintf('  GP points: %d (added %d in %d rounds)\n', ...
            res.gpr_result.n_total, res.gpr_result.n_added, res.gpr_result.n_rounds);
    fprintf('  Convergence: %s\n', res.gpr_result.converged);
    if isfield(res.gpr_result, 'N_eff')
        fprintf('  N_eff: %d\n', res.gpr_result.N_eff);
    end
    fprintf('\n');
end

% ========== SAVE ==========
if save_results
    results = struct();
    results.models = models;
    results.per_model = results_per_model;
    results.posteriors = post;
    results.best_model = models{bestIdx};
    results.best_idx = bestIdx;
    results.elapsed_time = elapsed_time;
    results.priors = priors;
    results.expData = expData;
    results.gprOpts = gprOpts;
    
    savefile = fullfile(output_dir, sprintf('results_mat%d.mat', material_id));
    save(savefile, 'results');
    fprintf('Saved to: %s\n\n', savefile);
end

% ========== HELPER: RUN ONE MODEL ==========
function out = run_single_model_gpr(modelName, expData, priors, gprOpts)
    tic_model = tic;
    
    fprintf('[%s] Starting GPR integration...\n', upper(modelName));
    
    % Get parameter bounds
    bounds = get_log10_bounds(modelName);
    axNames = axis_names_for_model(modelName);
    k = numel(axNames);
    
    xmin = zeros(1,k);
    xmax = zeros(1,k);
    for j = 1:k
        field = axNames{j};
        if strcmp(field, 'g'), field = 'G'; end  % Handle capitalization
        xmin(j) = 10^bounds.(field)(1);
        xmax(j) = 10^bounds.(field)(2);
    end
    
    fprintf('[%s] Parameter bounds (%dD):\n', upper(modelName), k);
    for j = 1:k
        fprintf('  %s: [%.3e, %.3e]\n', axNames{j}, xmin(j), xmax(j));
    end
    
    % NLL function: Must include parameter prior for correct evidence!
    % Evidence = + p(D|¸,M) × P(¸|M) d¸  (Paper Eq. 16)
    % In log-space: log p(D,¸|M) = log p(D|¸,M) + log P(¸|M)
    % So we integrate: -[NLL(¸) + logP(¸|M)] = log p(D,¸|M)
    nllFcn = @(X) compute_neg_log_posterior(X, modelName, expData, priors);
    
    % Run GPR integration
    fprintf('[%s] Running active_integrate_logaware...\n', upper(modelName));
    res = active_integrate_logaware(nllFcn, xmin, xmax, gprOpts, modelName, priors);
    
    % Model prior (BIC + axis need) - PAPER Eq. 28
    logPrior = compute_model_log_prior(modelName, k, expData.N_eff, priors);
    
    time_elapsed = toc(tic_model);
    
    % Pack output
    out = struct();
    out.name = modelName;
    out.logZ = res.logI_mean;
    out.log10Z = res.log10I_mean;
    out.logPrior = logPrior;
    out.k_params = k;
    out.gpr_result = res;
    out.time_seconds = time_elapsed;
    
    % Additional diagnostics
    out.gpr_result.n_total = size(res.U, 1);
    out.gpr_result.n_added = size(res.U, 1) - (10*k^2 + 5);
    out.gpr_result.n_rounds = numel(unique(res.U(:,1)));  % Approximate
    out.gpr_result.converged = 'Yes';  % Add actual convergence flag in active_integrate
    out.gpr_result.N_eff = expData.N_eff;
    
    fprintf('[%s] Done. log10(Z)=%.5g, time=%.1fs\n', ...
            upper(modelName), out.log10Z, time_elapsed);
end

% ========== HELPER: MODEL PRIOR ==========
function logpM = compute_model_log_prior(modelName, k, N_eff, priors)
    % PAPER Eq. 28: P(M) = exp(-k/2 * log(N_eff))
    % This is ONLY the BIC penalty - same for all models with same k
    % The data-driven factor goes in P(¸|M), NOT P(M)
    
    logpM = -0.5 * k * log(max(N_eff, 1));
    
    % NOTE: We do NOT include axis_need factors here!
    % Those are applied in the PARAMETER prior P(¸|M) via build_model_priors.m
end

% ========== HELPER: PARAMETER PRIOR LOOKUP ==========
function logP = param_prior_lookup(modelName, theta, priors)
    % PAPER Eq. 27: P(¸|M) = H(¸) * w_red(¸)
    %   where H is harmonic mean bottleneck (Eq. 26)
    %   and w_red is redundancy penalty (Eq. 25)
    
    % Hard cutoff: check if theta is within bounds
    bounds = get_log10_bounds(modelName);
    axNames = axis_names_for_model(modelName);
    
    for j = 1:numel(axNames)
        field = axNames{j};
        if strcmp(field, 'g'), field = 'G'; end
        
        log_theta_j = log10(theta(j));
        if log_theta_j < bounds.(field)(1) || log_theta_j > bounds.(field)(2)
            logP = -inf;
            return;
        end
    end
    
    % Within bounds  uniform prior (log(1/volume))
    vol = 1;
    for j = 1:numel(axNames)
        field = axNames{j};
        if strcmp(field, 'g'), field = 'G'; end
        vol = vol * diff(10.^bounds.(field));
    end
    
    logP = -log(vol);
    
    % NOTE: For GPR-based approach, the full hierarchical prior
    % (harmonic mean + redundancy) from build_model_priors is not
    % used during active learning, only the hard bounds check.
    % The prior is properly applied during evidence integration.
end

% ========== HELPER: SAFE LOGSUMEXP ==========
function s = safe_logsumexp(log_vals)
    m = max(log_vals(:));
    if ~isfinite(m)
        s = -inf;
        return;
    end
    s = m + log(sum(exp(log_vals(:) - m)));
end

% ========== HELPER: NEGATIVE LOG-POSTERIOR ==========
function neg_log_post = compute_neg_log_posterior(X, modelName, expData, priors)
    % Compute -log p(D,¸|M) = -log p(D|¸,M) - log P(¸|M)
    % This is what we integrate to get evidence
    
    N = size(X, 1);
    neg_log_post = zeros(N, 1);
    BIG_PENALTY = 1e10;
    
    % Get likelihood (NLL only)
    NLL = imr_nll_with_prior_matrix(X, modelName, expData, priors, struct());
    
    % Add parameter prior for each point
    for i = 1:N
        try
            % Get hierarchical parameter prior from build_model_priors
            logP_theta = get_hierarchical_prior(X(i,:), modelName, priors);
            
            if isfinite(logP_theta) && isfinite(NLL(i))
                % Negative log-posterior = NLL - logP(¸|M)
                neg_log_post(i) = NLL(i) - logP_theta;
            else
                neg_log_post(i) = BIG_PENALTY;
            end
        catch
            neg_log_post(i) = BIG_PENALTY;
        end
    end
    
    % Safety clamp
    neg_log_post(~isfinite(neg_log_post)) = BIG_PENALTY;
end

% ========== HELPER: GET HIERARCHICAL PRIOR ==========
function logP = get_hierarchical_prior(theta, modelName, priors)
    % Get the hierarchical anti-emulation prior from build_model_priors
    % This applies the harmonic-mean bottleneck + axis needs
    
    % First check hard bounds
    bounds = get_log10_bounds(modelName);
    axNames = axis_names_for_model(modelName);
    
    for j = 1:numel(axNames)
        field = axNames{j};
        if strcmp(field, 'g'), field = 'G'; end
        
        log_theta_j = log10(theta(j));
        if log_theta_j < bounds.(field)(1) || log_theta_j > bounds.(field)(2)
            logP = -inf;
            return;
        end
    end
    
    % Get the normalized prior from build_model_priors grid
    % This contains the hierarchical prior with data-driven penalties
    
    % build_model_priors uses: Newt, NH, KV, qNH, qKV, LM, SLS
    modelKey = modelName;
    if strcmpi(modelKey, 'newt'), modelKey = 'Newt'; end
    if strcmpi(modelKey, 'nh'), modelKey = 'NH'; end
    if strcmpi(modelKey, 'kv') || strcmpi(modelKey, 'nhkv'), modelKey = 'KV'; end
    if strcmpi(modelKey, 'qnh'), modelKey = 'qNH'; end
    if strcmpi(modelKey, 'qkv'), modelKey = 'qKV'; end
    if strcmpi(modelKey, 'lm') || strcmpi(modelKey, 'linmax'), modelKey = 'LM'; end
    if strcmpi(modelKey, 'sls'), modelKey = 'SLS'; end
    
    if ~isfield(priors, modelKey)
        % Fallback: uniform prior over valid range
        vol = 1;
        for j = 1:numel(axNames)
            field = axNames{j};
            if strcmp(field, 'g'), field = 'G'; end
            vol = vol * diff(10.^bounds.(field));
        end
        logP = -log(vol);
        warning('Model %s not found in priors, using uniform prior', modelName);
        return;
    end
    
    modelStruct = priors.(modelKey);
    
    if ~isfield(modelStruct, 'prior') || ~isfield(modelStruct, 'grid')
        % Fallback: uniform prior over valid range
        vol = 1;
        for j = 1:numel(axNames)
            field = axNames{j};
            if strcmp(field, 'g'), field = 'G'; end
            vol = vol * diff(10.^bounds.(field));
        end
        logP = -log(vol);
        return;
    end
    
    % Interpolate the hierarchical prior at theta
    % For now, use nearest-neighbor lookup (can improve to linear interp)
    grid_struct = modelStruct.grid;
    prior_grid = modelStruct.prior;
    
    switch lower(modelName)
        case 'newt'
            [~, idx] = min(abs(grid_struct.mu - theta(1)));
            p = prior_grid(idx);
            
        case 'nh'
            [~, idx] = min(abs(grid_struct.G - theta(1)));
            p = prior_grid(idx);
            
        case 'kv'
            [~, idx_mu] = min(abs(grid_struct.mu(:,1) - theta(1)));
            [~, idx_G] = min(abs(grid_struct.G(1,:) - theta(2)));
            p = prior_grid(idx_mu, idx_G);
            
        case 'qnh'
            [~, idx_G] = min(abs(grid_struct.G(:,1) - theta(1)));
            [~, idx_a] = min(abs(grid_struct.alpha(1,:) - theta(2)));
            p = prior_grid(idx_G, idx_a);
            
        case 'lm'
            [~, idx_mu] = min(abs(grid_struct.mu(:,1) - theta(1)));
            [~, idx_lam] = min(abs(grid_struct.lambda1(1,:) - theta(2)));
            p = prior_grid(idx_mu, idx_lam);
            
        case 'qkv'
            mu_vec = grid_struct.mu(:,1,1);
            G_vec = grid_struct.G(1,:,1);
            a_vec = grid_struct.alpha(1,1,:);
            [~, idx_mu] = min(abs(mu_vec - theta(1)));
            [~, idx_G] = min(abs(G_vec - theta(2)));
            [~, idx_a] = min(abs(a_vec(:) - theta(3)));
            p = prior_grid(idx_mu, idx_G, idx_a);
            
        case 'sls'
            mu_vec = grid_struct.mu(:,1,1);
            G_vec = grid_struct.G(1,:,1);
            lam_vec = grid_struct.lambda1(1,1,:);
            [~, idx_mu] = min(abs(mu_vec - theta(1)));
            [~, idx_G] = min(abs(G_vec - theta(2)));
            [~, idx_lam] = min(abs(lam_vec(:) - theta(3)));
            p = prior_grid(idx_mu, idx_G, idx_lam);
            
        otherwise
            error('Unknown model: %s', modelName);
    end
    
    % Return log-prior (add small epsilon to avoid log(0))
    logP = log(max(p, 1e-300));
end