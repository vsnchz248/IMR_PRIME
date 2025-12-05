function [negLogLike, details] = imr_negloglike(sim, expData, likeOpts)
% IMR_NEGLOGLIKE  Compute β-marginalized negative log-likelihood (BIMR-style, log-space stable)
%
% Pure log-space implementation - uses actual computed values with no floors or limits.
%
% BIMR Methodology (from paper):
%   - Heteroscedastic noise: σ²(R*) = (β × σ₀(R*) / √w)²
%   - Half-Cauchy prior on β: p(β) = (2/π) / (1 + β²)
%   - Marginalization: p(data|θ) = ∫ p(data|θ,β) p(β) dβ

    % ---------- Short-circuit if solver failed ----------
    if ~sim.ok
        negLogLike = 1e10;
        details = struct('logL_marg', -inf, 'logL_beta', [], ...
                         'betaGrid', [], 'post_beta', [], ...
                         'nTrialsUsed', 0, 'nPointsUsed', 0, ...
                         'ssq_R', NaN, 'ssq_Rdot', NaN);
        return;
    end

    if nargin < 3, likeOpts = struct(); end
    if ~isfield(likeOpts,'useRdot'),   likeOpts.useRdot  = true;               end
    if ~isfield(likeOpts,'betaGrid'),  likeOpts.betaGrid = 0.05:0.05:10.0;    end

    betaGrid = likeOpts.betaGrid(:)';
    nBeta    = numel(betaGrid);

    % ---------- Unpack experimental data ----------
    R_exp_nd    = expData.Rmatrix;
    Rdot_exp_nd = expData.Rdotmatrix;
    t_exp_nd    = expData.tmatrix;
    mask        = expData.mask;

    [nTime, nTrials] = size(R_exp_nd); %#ok<NASGU>

    mustHave = {'sigma0_R','sigma0_Rdot','weights_w'};
    for f = 1:numel(mustHave)
        if ~isfield(expData, mustHave{f})
            error('expData.%s is missing.', mustHave{f});
        end
    end

    sigma0_R    = expData.sigma0_R;
    sigma0_Rdot = expData.sigma0_Rdot;
    weights_w   = expData.weights_w;

    % ---------- Simulation arrays ----------
    t_sim_nd    = sim.t_nd(:);
    R_sim_nd    = sim.R_nd(:);
    Rdot_sim_nd = sim.Rdot_nd(:);

    % ---------- β prior (Half-Cauchy, BIMR Eq. 16) IN LOG SPACE ----------
    logP_beta_raw = log(2/pi) - log(1 + betaGrid.^2);
    
    % Quadrature weights
    d = zeros(size(betaGrid));
    d(1)   = 0.5 * (betaGrid(2) - betaGrid(1));
    d(end) = 0.5 * (betaGrid(end) - betaGrid(end-1));
    for b = 2:nBeta-1
        d(b) = 0.5 * (betaGrid(b+1) - betaGrid(b-1));
    end
    
    logW_beta = logP_beta_raw + log(max(d, eps));

    % ---------- Accumulators ----------
    logL_theta_beta = zeros(1, nBeta);
    nTrialsUsed     = 0;
    nPointsUsed     = 0;
    ssq_R_total     = 0.0;
    ssq_Rdot_total  = 0.0;

    % ---------- Loop over trials ----------
    for j = 1:nTrials
        t_exp_j_nd = t_exp_nd(:, j);

        % Interpolate simulation
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

        % Get sigma/weights
        mask_indices       = find(mask_j);
        valid_mask_indices = mask_indices(valid);

        sig0_R_g    = sigma0_R(valid_mask_indices,  j);
        sig0_Rdot_g = sigma0_Rdot(valid_mask_indices, j);
        w_g         = weights_w(valid_mask_indices, j);

        % Residuals
        rR    = R_exp_g    - R_sim_g;
        rRdot = Rdot_exp_g - Rdot_sim_g;

        % Baseline variances (without β)
        vR0  = max(sig0_R_g   .^2 ./ max(w_g, eps), 1e-24);
        vRd0 = max(sig0_Rdot_g.^2 ./ max(w_g, eps), 1e-24);

        rR(~isfinite(rR))     = 0;
        rRdot(~isfinite(rRdot)) = 0;
        vR0(~isfinite(vR0))   = 1e-24;
        vRd0(~isfinite(vRd0)) = 1e-24;

        nValid = numel(rR);
        if nValid == 0, continue; end

        ssq_R_total    = ssq_R_total    + sum(rR.^2);
        ssq_Rdot_total = ssq_Rdot_total + sum(rRdot.^2);

        % ---- β loop IN LOG SPACE (PURE VALUES) ----
        for b = 1:nBeta
            beta = betaGrid(b);

            % Variance with β scaling
            vR_b = (beta^2) * vR0;

            % Log-likelihood for R
            logL_R = -0.5 * (sum((rR.^2) ./ vR_b) + sum(log(2*pi * vR_b)));

            logL_tot = logL_R;

            % Rdot contribution
            if likeOpts.useRdot
                vRd_b = (beta^2) * vRd0;
                
                logL_Rd = -0.5 * (sum((rRdot.^2) ./ vRd_b) + sum(log(2*pi * vRd_b)));
                logL_tot = logL_tot + logL_Rd;
            end

            if ~isfinite(logL_tot)
                logL_tot = -1e6;
            end

            logL_theta_beta(b) = logL_theta_beta(b) + logL_tot;
        end

        nTrialsUsed  = nTrialsUsed + 1;
        nPointsUsed  = nPointsUsed + nValid;
    end

    % ---------- β marginalization IN LOG SPACE ----------
    if all(~isfinite(logL_theta_beta))
        logL_marg = logsumexp_stable(logW_beta);
        post_beta = exp(normalize_logweights(logW_beta));
    else
        log_integrand = logL_theta_beta + logW_beta;
        logL_marg = logsumexp_stable(log_integrand);
        post_beta = exp(normalize_logweights(log_integrand));
    end

    negLogLike = -logL_marg;

    % ---------- Pack details ----------
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

% ======================================================================
% Helper functions
% ======================================================================
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