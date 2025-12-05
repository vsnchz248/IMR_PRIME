function NLL = imr_nll_with_prior_matrix(X, modelName, expData, priors, likeOpts)
% IMR_NLL_WITH_PRIOR_MATRIX  Total negative log posterior for Bayesian IMR
%
% Computes: NLL(θ) = -log p(data | θ) - log p(θ | M)
%
% Uses physics-based noise floor in imr_negloglike to prevent
% unphysical perfect fits, rather than arbitrary NLL bounds.

    if nargin < 5 || isempty(likeOpts)
        likeOpts = struct();
        likeOpts.useRdot  = true;
        likeOpts.betaGrid = 0.05:0.05:10;
    end
    
    N = size(X, 1);
    NLL = zeros(N, 1);
    
    for i = 1:N
        theta_i = X(i,:);
        
        % ========== 1. Forward simulation ==========
        sim = imr_forward_solver(theta_i, modelName, expData);
        
        % ========== 2. β-marginalized likelihood (with noise floor) ==========
        [negLogLike, ~] = imr_negloglike(sim, expData, likeOpts);
        
        % ========== 3. Physics-informed prior ==========
        logp = priors.paramPrior(modelName, theta_i);
        
        % ========== 4. Total NLL ==========
        if ~isfinite(logp)
            NLL(i) = 1e10;
        else
            NLL(i) = negLogLike - logp;
        end
        
        % Safety
        if ~isfinite(NLL(i))
            NLL(i) = 1e10;
        end
    end
end
