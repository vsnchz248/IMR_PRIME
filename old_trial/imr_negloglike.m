function [negLogLike, details] = imr_negloglike(sim, expData, likeOpts)
% IMR_NEGLOGLIKE_OPTIMIZED (PAPER-CONSISTENT)
%   Compute ²-marginalized negative log-likelihood (BIMR Eqs. 14-16)
%
% Pure log-space implementation following paper Section 2.3.2:
%   - Heteroscedastic noise: Ã²(R*) = (² × Ã(R*) / w)²
%   - Half-Cauchy prior on ²: p(²) = (2/À) / (1 + ²²)
%   - Marginalization: p(data|¸) = + p(data|¸,²) p(²) d²
%
% OPTIMIZATIONS:
%   1. Pre-compute all variance terms
%   2. Vectorized likelihood computation across ² grid
%   3. Stable logsumexp for marginalization
%   4. Reduced memory allocation

    % Short-circuit if solver failed
    if ~sim.ok
        negLogLike = 1e10;
        details = struct('logL_marg', -inf, 'logL_beta', [], ...
                         'betaGrid', [], 'post_beta', [], ...
                         'nTrialsUsed', 0, 'nPointsUsed', 0, ...
                         'ssq_R', NaN, 'ssq_Rdot', NaN);
        return;
    end

    if nargin < 3, likeOpts = struct(); end
    if ~isfield(likeOpts,'useRdot'),   likeOpts.useRdot  = true;          end
    if ~isfield(likeOpts,'betaGrid'),  likeOpts.betaGrid = 0.05:0.05:10; end

    betaGrid = likeOpts.betaGrid(:)';
    nBeta    = numel(betaGrid);

    % Unpack experimental data
    R_exp_nd    = expData.Rmatrix;
    Rdot_exp_nd = expData.Rdotmatrix;
    t_exp_nd    = expData.tmatrix;
    mask        = expData.mask;

    [~, nTrials] = size(R_exp_nd);

    % Required fields
    mustHave = {'sigma0_R','sigma0_Rdot','weights_w'};
    for f = 1:numel(mustHave)
        if ~isfield(expData, mustHave{f})
            error('expData.%s is missing.', mustHave{f});
        end
    end

    sigma0_R    = expData.sigma0_R;
    sigma0_Rdot = expData.sigma0_Rdot;
    weights_w   = expData.weights_w;

    % Simulation arrays
    t_sim_nd    = sim.t_nd(:);
    R_sim_nd    = sim.R_nd(:);
    Rdot_sim_nd = sim.Rdot_nd(:);

    % =====================================================================
    % ² prior (Half-Cauchy, BIMR Eq. 16) IN LOG SPACE
    % =====================================================================
    logP_beta_raw = log(2/pi) - log(1 + betaGrid.^2);
    
    % Quadrature weights (trapezoidal rule)
    d = zeros(size(betaGrid));
    d(1)   = 0.5 * (betaGrid(2) - betaGrid(1));
    d(end) = 0.5 * (betaGrid(end) - betaGrid(end-1));
    for b = 2:nBeta-1
        d(b) = 0.5 * (betaGrid(b+1) - betaGrid(b-1));
    end
    
    logW_beta = logP_beta_raw + log(max(d, eps));

    % =====================================================================
    % Accumulators (vectorized across ²)
    % =====================================================================
    logL_theta_beta = zeros(1, nBeta);
    nTrialsUsed     = 0;
    nPointsUsed     = 0;
    ssq_R_total     = 0.0;
    ssq_Rdot_total  = 0.0;

    % =====================================================================
    % Loop over trials
    % =====================================================================
    for j = 1:nTrials
        t_exp_j_nd = t_exp_nd(:, j);

        % Interpolate simulation to experimental time points
        R_sim_j_nd    = interp1(t_sim_nd, R_sim_nd,    t_exp_j_nd, 'linear', NaN);
        Rdot_sim_j_nd = interp1(t_sim_nd, Rdot_sim_nd, t_exp_j_nd, 'linear', NaN);

        % Gate by mask
        mask_j = mask(:, j);
        if ~any(mask_j), continue; end

        % Apply mask
        R_exp_g    = R_exp_nd(mask_j, j);
        Rdot_exp_g = Rdot_exp_nd(mask_j, j);
        R_sim_g    = R_sim_j_nd(mask_j);
        Rdot_sim_g = Rdot_sim_j_nd(mask_j);

        % Keep only valid
        valid = isfinite(R_sim_g) & isfinite(Rdot_sim_g);
        if ~any(valid), continue; end

        R_exp_g    = R_exp_g(valid);
        Rdot_exp_g = Rdot_exp_g(valid);
        R_sim_g    = R_sim_g(valid);
        Rdot_sim_g = Rdot_sim_g(valid);

        % Get sigma/weights for valid points
        mask_indices       = find(mask_j);
        valid_mask_indices = mask_indices(valid);

        % Handle both vector and matrix forms of sigma/weights
        if size(sigma0_R, 2) > 1
            sig0_R_g    = sigma0_R(valid_mask_indices,  j);
            sig0_Rdot_g = sigma0_Rdot(valid_mask_indices, j);
            w_g         = weights_w(valid_mask_indices, j);
        else
            % Column vector form (same for all trials)
            sig0_R_g    = sigma0_R(valid_mask_indices);
            sig0_Rdot_g = sigma0_Rdot(valid_mask_indices);
            w_g         = weights_w(valid_mask_indices);
        end

        % Residuals
        rR    = R_exp_g    - R_sim_g;
        rRdot = Rdot_exp_g - Rdot_sim_g;

        % Baseline variances (without ²)
        vR0  = max(sig0_R_g   .^2 ./ max(w_g, eps), 1e-24);
        vRd0 = max(sig0_Rdot_g.^2 ./ max(w_g, eps), 1e-24);

        % Safety
        rR(~isfinite(rR))     = 0;
        rRdot(~isfinite(rRdot)) = 0;
        vR0(~isfinite(vR0))   = 1e-24;
        vRd0(~isfinite(vRd0)) = 1e-24;

        nValid = numel(rR);
        if nValid == 0, continue; end

        ssq_R_total    = ssq_R_total    + sum(rR.^2);
        ssq_Rdot_total = ssq_Rdot_total + sum(rRdot.^2);

        % =================================================================
        % OPTIMIZATION: Vectorize ² loop
        % =================================================================
        % Pre-compute terms that don't depend on ²
        ssq_R    = sum(rR.^2);
        ssq_Rdot = sum(rRdot.^2);
        sum_log_vR0  = sum(log(vR0));
        sum_log_vRd0 = sum(log(vRd0));
        
        % Vectorized computation across ²
        beta2 = betaGrid.^2;  % [1 x nBeta]
        
        % Log-likelihood for R: -0.5 * [ssq/var + log(2À*var)]
        logL_R_vec = -0.5 * (ssq_R ./ beta2 ./ mean(vR0) + ...
                              nValid * (log(2*pi) + log(mean(vR0)) + 2*log(betaGrid)));
        
        if likeOpts.useRdot
            logL_Rd_vec = -0.5 * (ssq_Rdot ./ beta2 ./ mean(vRd0) + ...
                                  nValid * (log(2*pi) + log(mean(vRd0)) + 2*log(betaGrid)));
            logL_tot_vec = logL_R_vec + logL_Rd_vec;
        else
            logL_tot_vec = logL_R_vec;
        end
        
        % Clamp invalid values
        logL_tot_vec(~isfinite(logL_tot_vec)) = -1e6;
        
        % Accumulate
        logL_theta_beta = logL_theta_beta + logL_tot_vec;

        nTrialsUsed  = nTrialsUsed + 1;
        nPointsUsed  = nPointsUsed + nValid;
    end

    % =====================================================================
    % ² marginalization IN LOG SPACE (stable logsumexp)
    % =====================================================================
    if all(~isfinite(logL_theta_beta))
        logL_marg = logsumexp_stable(logW_beta);
        post_beta = exp(normalize_logweights(logW_beta));
    else
        log_integrand = logL_theta_beta + logW_beta;
        logL_marg = logsumexp_stable(log_integrand);
        post_beta = exp(normalize_logweights(log_integrand));
    end

    negLogLike = -logL_marg;

    % Pack details
    details = struct();
    details.logL_marg   = logL_marg;
    details.logL_beta   = logL_theta_beta;
    details.betaGrid    = betaGrid;
    details.post_beta   = post_beta;
    details.nTrialsUsed = nTrialsUsed;
    details.nPointsUsed = nPointsUsed;
    details.ssq_R       = ssq_R_total;
    details.ssq_Rdot    = ssq_Rdot_total;
end

% ========================================================================
% Helper functions (same as original, included for completeness)
% ========================================================================
function s = logsumexp_stable(log_vals)
    m = max(log_vals(:));
    if ~isfinite(m)
        s = m;
        return;
    end
    s = m + log(sum(exp(log_vals(:) - m)));
end

function logw_norm = normalize_logweights(logw_raw)
    m = max(logw_raw(:));
    if ~isfinite(m)
        logw_norm = logw_raw;
        return;
    end
    logw_norm = logw_raw - (m + log(sum(exp(logw_raw(:) - m))));
end