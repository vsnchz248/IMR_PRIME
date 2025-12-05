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
        fprintf('========================================\n');
        fprintf('Models: %s\n', strjoin(models, ', '));
        fprintf('Effective N: %d\n', priors.N_eff);
    end
    
    % ========== For each model, compute evidence via GPR integration ==========
    for i = 1:NM
        modelName = models{i};
        
        if opts.verbose
            fprintf('\n--- Model %d/%d: %s ---\n', i, NM, upper(modelName));
        end
        
        % Get parameter bounds
        [xmin, xmax] = get_param_bounds(modelName, priors);
        
        % Define NLL function (negative log posterior without model prior)
        % This is: -log p(data|θ,M) - log p(θ|M)
        % Uses YOUR existing wrapper function
        likeOpts.useRdot = true;
        likeOpts.betaGrid = opts.betaGrid;
        funNLL = @(X) imr_nll_with_prior_matrix(X, modelName, expData, priors, likeOpts);
        
        % Run GPR-based integration using YOUR existing active_integrate_logaware
        tic;
        gpr_result = active_integrate_logaware(funNLL, xmin, xmax, opts.gprOpts);
        elapsed = toc;
        
        % Store full GPR output
        gpr_out{i} = gpr_result;
        
        % Extract log evidence: log p(data|M) = log ∫ p(data|θ,M) p(θ|M) dθ
        log_evidence(i) = gpr_result.logI_mean;
        
        % Find MAP parameters
        [~, iMAP] = max(gpr_result.Y);  % Y is shifted log-likelihood
        theta_MAP{i} = gpr_result.fromFeat(gpr_result.U(iMAP,:));
        
        if opts.verbose
            fprintf('log10(evidence) = %.6g [%.6g, %.6g]\n', ...
                    gpr_result.log10I_mean, ...
                    gpr_result.log10I_CI95(1), gpr_result.log10I_CI95(2));
            fprintf('MAP parameters: ');
            fprintf('%.4g ', theta_MAP{i});
            fprintf('\n');
            fprintf('Computation time: %.2f sec\n', elapsed);
            fprintf('GP evaluations: %d\n', size(gpr_result.U,1));
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