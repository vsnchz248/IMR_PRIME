function NLL = imr_nll_with_prior_matrix(X, modelName, expData, priors, likeOpts)
% IMR_NLL_WITH_PRIOR_MATRIX
%   Negative log-likelihood for Bayesian IMR (likelihood-only).
%
% This function returns:
%   NLL(theta) = -log p(D | theta, M)
%
% The parameter prior p(theta | M) is used ONLY to:
%   - Reject parameter vectors outside the prior support (hard penalty).
% The actual prior weight is applied later in the evidence integral:
%   P(D | M) = + p(D | theta, M) p(theta | M) dtheta
%
% Inputs:
%   X         - [N x d] matrix of parameter vectors theta
%   modelName - string, model identifier ('newt','qkv', etc.)
%   expData   - experimental data struct from prepare_data()
%   priors    - struct from build_priors(), must define priors.paramPrior
%   likeOpts  - struct with fields:
%                 .useRdot  (logical, default true)
%                 .betaGrid (vector, default 0.05:0.05:10)
%
% Output:
%   NLL       - [N x 1] vector of negative log-likelihood values
%               (large penalty for out-of-support or failed sims)

    if nargin < 5 || isempty(likeOpts)
        likeOpts = struct();
    end
    if ~isfield(likeOpts,'useRdot'),   likeOpts.useRdot  = true;               end
    if ~isfield(likeOpts,'betaGrid'),  likeOpts.betaGrid = 0.05:0.05:10;      end

    N           = size(X, 1);
    NLL         = zeros(N, 1);
    BIG_PENALTY = 1e10;

    % We assume priors.paramPrior(modelName, theta) exists and returns log p(theta|M)
    % It is used only to define the support of the prior (hard cutoff).
    
    parfor i = 1:N
        theta_i  = X(i,:);
        totalNLL = BIG_PENALTY;  % default in case of failure or rejection
        
        try
            % 1) Check if parameters are within prior support
            logPrior = priors.paramPrior(modelName, theta_i);
            
            % If prior is essentially zero (out of support), assign penalty
            if ~isfinite(logPrior) || logPrior < -1e6
                totalNLL = BIG_PENALTY;
            else
                % 2) Forward simulation
                sim = imr_forward_solver(theta_i, modelName, expData);
                
                % 3) Negative log-likelihood: -log p(D | theta, M)
                [negLogLike, ~] = imr_negloglike(sim, expData, likeOpts);
                
                if ~isfinite(negLogLike)
                    totalNLL = BIG_PENALTY;
                else
                    % IMPORTANT:
                    %   totalNLL = -log p(D | theta, M)
                    % The parameter prior is NOT included here; it is
                    % applied later in the evidence integral.
                    totalNLL = negLogLike;
                end
            end
        catch
            % Any failure  large penalty
            totalNLL = BIG_PENALTY;
        end
        
        % Safety clamp
        if ~isfinite(totalNLL) || isnan(totalNLL) || totalNLL < -1e6
            totalNLL = BIG_PENALTY;
        end
        
        NLL(i) = totalNLL;
    end
end
