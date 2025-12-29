function NLL = imr_nll_with_prior_matrix(X, modelName, expData, priors, likeOpts)
% IMR_NLL_WITH_PRIOR_MATRIX (OPTIMIZED)
%   Negative log-likelihood for Bayesian IMR (likelihood-only).
%   Optimized version with batching support and efficient parallelization.
%
% This function returns:
%   NLL(theta) = -log p(D | theta, M)
%
% The parameter prior p(theta | M) is used ONLY to reject out-of-support
% parameters (hard penalty). The actual prior weight is applied later in
% the evidence integral.
%
% OPTIMIZATIONS:
%   1. Pre-check all priors before ANY forward solver calls
%   2. Batch valid parameters to minimize overhead
%   3. Efficient parfor only over valid params
%   4. Optional vectorized forward solver support
%
% Inputs:
%   X         - [N x d] matrix of parameter vectors theta
%   modelName - string, model identifier ('newt','kv','nh', etc.)
%   expData   - experimental data struct
%   priors    - struct from build_model_priors()
%   likeOpts  - struct with fields:
%                 .useRdot  (logical, default true)
%                 .betaGrid (vector, default 0.05:0.05:10)
%
% Output:
%   NLL       - [N x 1] vector of negative log-likelihood values

    if nargin < 5 || isempty(likeOpts)
        likeOpts = struct();
    end
    if ~isfield(likeOpts,'useRdot'),   likeOpts.useRdot  = true;          end
    if ~isfield(likeOpts,'betaGrid'),  likeOpts.betaGrid = 0.05:0.05:10; end
    
    N = size(X, 1);
    NLL = zeros(N, 1);
    BIG_PENALTY = 1e10;
    
    % ============================================================
    % OPTIMIZATION 1: Pre-check ALL priors (fast, no solver calls)
    % ============================================================
    validMask = false(N, 1);
    for i = 1:N
        try
            logPrior = priors.paramPrior(modelName, X(i,:));
            validMask(i) = isfinite(logPrior) && logPrior > -1e6;
        catch
            validMask(i) = false;
        end
    end
    
    % Assign penalty to invalid params immediately
    NLL(~validMask) = BIG_PENALTY;
    
    n_valid = nnz(validMask);
    if n_valid == 0
        return;
    end
    
    fprintf('[%s] Evaluating %d valid (of %d total) parameter sets...\n', ...
            upper(modelName), n_valid, N);
    
    % Extract valid parameter sets
    X_valid = X(validMask, :);
    
    % ============================================================
    % OPTIMIZATION 2: Check if forward solver supports batching
    % ============================================================
    % Try to detect if solver can handle matrix input
    % (This requires your forward solver to be modified)
    
    solver_supports_batch = false;  % Set to true when you vectorize solver
    
    if solver_supports_batch
        % ========================================================
        % BATCH MODE: Single vectorized forward solver call
        % ========================================================
        fprintf('  Using batched forward solver...\n');
        try
            % Single call for all valid params
            sim_batch = imr_forward_solver_batch(X_valid, modelName, expData);
            
            % Compute likelihoods
            NLL_valid = zeros(n_valid, 1);
            for i = 1:n_valid
                try
                    [negLogLike, ~] = imr_negloglike(sim_batch(i), expData, likeOpts);
                    if isfinite(negLogLike)
                        NLL_valid(i) = negLogLike;
                    else
                        NLL_valid(i) = BIG_PENALTY;
                    end
                catch
                    NLL_valid(i) = BIG_PENALTY;
                end
            end
            NLL(validMask) = NLL_valid;
            
        catch ME
            warning('Batch forward solver failed: %s. Falling back to serial.', ME.message);
            solver_supports_batch = false;
        end
    end
    
    if ~solver_supports_batch
        % ========================================================
        % PARALLEL MODE: parfor over valid params only
        % ========================================================
        fprintf('  Using parallel forward solver (parfor over %d params)...\n', n_valid);
        
        NLL_valid = zeros(n_valid, 1);
        parfor i = 1:n_valid
            totalNLL = BIG_PENALTY;
            try
                % Forward simulation
                sim = imr_forward_solver(X_valid(i,:), modelName, expData);
                
                % Negative log-likelihood
                [negLogLike, ~] = imr_negloglike(sim, expData, likeOpts);
                
                if isfinite(negLogLike)
                    totalNLL = negLogLike;
                else
                    totalNLL = BIG_PENALTY;
                end
            catch
                totalNLL = BIG_PENALTY;
            end
            NLL_valid(i) = totalNLL;
        end
        NLL(validMask) = NLL_valid;
    end
    
    % Final safety clamp
    bad_mask = ~isfinite(NLL) | NLL < -1e6;
    if any(bad_mask)
        NLL(bad_mask) = BIG_PENALTY;
    end
    
    fprintf('[%s] Done. %d/%d succeeded, %d failed.\n', ...
            upper(modelName), nnz(NLL < BIG_PENALTY), N, nnz(NLL >= BIG_PENALTY));
end