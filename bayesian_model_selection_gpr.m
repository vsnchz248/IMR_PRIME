function results = bayesian_model_selection_gpr(expData, priors, models, opts)
% BAYESIAN_MODEL_SELECTION_GPR  Bayesian model selection using GPR integration
%
% This implements the paper's methodology (Eqs. 9, 19-20, 28) but replaces
% the discrete grid with active GPR learning + RQMC integration.
%
% Inputs:
%   expData - from prepare_data()
%   priors  - from build_priors()
%   models  - cell array of model names: {'newt','nh','kv','qnh','lm','qkv','sls'}
%   opts    - optional settings:
%       .betaGrid      - β grid for likelihood marginalization
%       .gprOpts       - options for GPR integrator
%       .verbose       - display progress
%
% Outputs:
%   results - struct with:
%       .model_posterior   - P(M|D) for each model
%       .log_evidence      - log P(D|M) for each model
%       .theta_MAP         - MAP parameters for each model
%       .gpr_out           - full GPR output for each model
%       .models            - model names

    if nargin < 4, opts = struct(); end
    if ~isfield(opts,'betaGrid'),  opts.betaGrid  = 0.05:0.05:10; end
    if ~isfield(opts,'verbose'),   opts.verbose   = true;         end
    if ~isfield(opts,'gprOpts'),   opts.gprOpts   = struct();     end
    if ~isfield(opts,'parallel'),  opts.parallel  = false;        end
    
    % Ensure GPR opts include betaGrid for consistency
    if ~isfield(opts.gprOpts,'betaGrid')
        opts.gprOpts.betaGrid = opts.betaGrid;
    end
    
    NM = numel(models);
    log_evidence = zeros(NM,1);
    theta_MAP = cell(NM,1);
    gpr_out = cell(NM,1);
    
    if opts.verbose
        fprintf('\n========================================\n');
        fprintf('  Bayesian Model Selection (GPR-based)\n');
        if opts.parallel
            fprintf('  Using PARALLEL processing\n');
        else
            fprintf('  Using SERIAL processing\n');
        end
        fprintf('========================================\n');
        fprintf('Models: %s\n', strjoin(models, ', '));
        fprintf('Effective N: %d\n', priors.N_eff);
    end
    
    % ========== For each model, compute evidence via GPR integration ==========
    if opts.parallel
        % PARALLEL EXECUTION
        parfor i = 1:NM
            [log_evidence(i), theta_MAP{i}, gpr_out{i}] = ...
                process_model(models{i}, i, NM, expData, priors, opts);
        end
    else
        % SERIAL EXECUTION
        for i = 1:NM
            [log_evidence(i), theta_MAP{i}, gpr_out{i}] = ...
                process_model(models{i}, i, NM, expData, priors, opts);
        end
    end
    
    % ========== Apply model prior and compute posterior ==========
    log_model_prior = zeros(NM,1);
    for i = 1:NM
        modelName = models{i};
        k_params = numel(theta_MAP{i});
        log_model_prior(i) = priors.modelPrior(modelName, k_params);
    end
    
    % Log posterior: log p(M|D) ∝ log p(D|M) + log p(M)
    log_unnorm = log_evidence + log_model_prior;
    
    % Normalize (logsumexp trick)
    max_log = max(log_unnorm);
    model_posterior = exp(log_unnorm - max_log);
    model_posterior = model_posterior / sum(model_posterior);
    
    % ========== Display results ==========
    if opts.verbose
        fprintf('\n========================================\n');
        fprintf('  Summary\n');
        fprintf('========================================\n');
        fprintf('%-8s %12s %12s %12s\n', ...
                'Model', 'log10(Evid)', 'log(Prior)', 'P(M|D)');
        fprintf('----------------------------------------\n');
        for i = 1:NM
            fprintf('%-8s %12.4g %12.4g %12.6f\n', ...
                    upper(models{i}), ...
                    log_evidence(i)/log(10), ...
                    log_model_prior(i), ...
                    model_posterior(i));
        end
        fprintf('========================================\n');
        
        [~, iBest] = max(model_posterior);
        fprintf('\nMost plausible model: %s (P = %.4f)\n', ...
                upper(models{iBest}), model_posterior(iBest));
        fprintf('MAP parameters: ');
        fprintf('%.4g ', theta_MAP{iBest});
        fprintf('\n');
        
        % Performance statistics
        fprintf('\n========================================\n');
        fprintf('  Performance Statistics\n');
        fprintf('========================================\n');
        fprintf('Model     Time(sec)  Evals  Evals/sec\n');
        fprintf('----------------------------------------\n');
        for i = 1:NM
            n_evals = gpr_out{i}.n_evaluations;
            comp_time = gpr_out{i}.computation_time;
            evals_per_sec = n_evals / comp_time;
            fprintf('%-6s    %8.1f  %5d    %6.2f\n', ...
                    upper(models{i}), comp_time, n_evals, evals_per_sec);
        end
        fprintf('----------------------------------------\n');
        total_time = sum(cellfun(@(x) x.computation_time, gpr_out));
        total_evals = sum(cellfun(@(x) x.n_evaluations, gpr_out));
        fprintf('TOTAL     %8.1f  %5d    %6.2f\n', ...
                total_time, total_evals, total_evals/total_time);
    end
    
    % ========== Pack results ==========
    results = struct();
    results.models = models;
    results.model_posterior = model_posterior;
    results.log_evidence = log_evidence;
    results.log_model_prior = log_model_prior;
    results.theta_MAP = theta_MAP;
    results.gpr_out = gpr_out;
    results.expData = expData;
    results.priors = priors;
    
    % Identify most plausible model
    [results.max_posterior, results.best_idx] = max(model_posterior);
    results.best_model = models{results.best_idx};
    
    % Performance statistics (for each model)
    results.performance = struct();
    results.performance.model_times = cellfun(@(x) x.computation_time, gpr_out);
    results.performance.model_evals = cellfun(@(x) x.n_evaluations, gpr_out);
    results.performance.total_time = sum(results.performance.model_times);
    results.performance.total_evals = sum(results.performance.model_evals);
    results.performance.evals_per_sec = results.performance.model_evals ./ results.performance.model_times;
end

% ========== HELPER FUNCTION ==========

function [xmin, xmax] = get_param_bounds(modelName, priors)
    % Extract parameter bounds from priors structure
    ranges = priors.ranges;
    
    switch lower(modelName)
        case 'newt'
            xmin = ranges.mu(1);
            xmax = ranges.mu(2);
            
        case 'nh'
            xmin = ranges.G(1);
            xmax = ranges.G(2);
            
        case 'kv'
            xmin = [ranges.mu(1), ranges.G(1)];
            xmax = [ranges.mu(2), ranges.G(2)];
            
        case 'qnh'
            xmin = [ranges.G(1), ranges.alpha(1)];
            xmax = [ranges.G(2), ranges.alpha(2)];
            
        case 'lm'
            xmin = [ranges.mu(1), ranges.lambda1(1)];
            xmax = [ranges.mu(2), ranges.lambda1(2)];
            
        case 'qkv'
            xmin = [ranges.mu(1), ranges.G(1), ranges.alpha(1)];
            xmax = [ranges.mu(2), ranges.G(2), ranges.alpha(2)];
            
        case 'sls'
            xmin = [ranges.mu(1), ranges.G(1), ranges.lambda1(1)];
            xmax = [ranges.mu(2), ranges.G(2), ranges.lambda1(2)];
            
        otherwise
            error('Unknown model: %s', modelName);
    end
end


function [log_ev, map_theta, gpr_result] = process_model(modelName, idx, total, expData, priors, opts)
    % PROCESS_MODEL - Process a single model for Bayesian selection
    %   Helper function to enable parallel execution
    
    if opts.verbose
        fprintf('\n--- Model %d/%d: %s ---\n', idx, total, upper(modelName));
    end
    
    % Get parameter bounds
    [xmin, xmax] = get_param_bounds(modelName, priors);
    
    % Define NLL function (negative log posterior without model prior)
    likeOpts.useRdot = true;
    likeOpts.betaGrid = opts.betaGrid;
    funNLL = @(X) imr_nll_with_prior_matrix(X, modelName, expData, priors, likeOpts);
    
    % Run GPR-based integration
    tic;
    gpr_result = active_integrate_logaware(funNLL, xmin, xmax, opts.gprOpts);
    elapsed = toc;
    
    % Extract results
    log_ev = gpr_result.logI_mean;
    
    % Find MAP parameters
    [~, iMAP] = max(gpr_result.Y);
    map_theta = gpr_result.fromFeat(gpr_result.U(iMAP,:));
    
    % Add timing and evaluation count to gpr_result for tracking
    gpr_result.model_name = modelName;
    gpr_result.computation_time = elapsed;
    gpr_result.n_evaluations = size(gpr_result.U, 1);
    
    if opts.verbose
        fprintf('log10(evidence) = %.6g [%.6g, %.6g]\n', ...
                gpr_result.log10I_mean, ...
                gpr_result.log10I_CI95(1), gpr_result.log10I_CI95(2));
        fprintf('MAP parameters: ');
        fprintf('%.4g ', map_theta);
        fprintf('\n');
        fprintf('Computation time: %.2f sec\n', elapsed);
        fprintf('GP evaluations: %d\n', gpr_result.n_evaluations);
    end
end
