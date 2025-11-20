function likelihoodEval = build_likelihood_evaluator(expData, cfg)
% BUILD_LIKELIHOOD_EVALUATOR  Create likelihood function for IMR experiments
%
% Syntax:
%   likelihoodEval = build_likelihood_evaluator(expData, cfg)
%
% Inputs:
%   expData - Structure from expDataPrep with fields:
%       .Rmatrix, .Rdotmatrix, .tmatrix  : Experimental data [nTime x nTrials]
%       .tc                              : Characteristic times per trial
%       .mask                            : Logical mask for high-info regions
%       .sigmaR0, .sigmaRdot0            : Baseline uncertainties
%   cfg - Configuration structure with fields:
%       .useHetero        : Enable heteroscedastic weighting (default: true)
%       .kappa            : Strain-rate gate steepness (default: 1.0)
%       .m_floor          : Minimum weight floor (default: 0.10)
%       .betaGrid         : Beta values for noise marginalization
%       .Thresholds       : Strain-based gating parameters
%       .useRdotInLL      : Include velocity in likelihood (default: true)
%
% Outputs:
%   likelihoodEval - Structure with fields:
%       .computeLogL      : Function handle @(simData) -> log-likelihood
%       .getBetaPosterior : Function handle @(simData) -> beta posterior
%       .cfg              : Configuration used
%
% Example:
%   likelihoodEval = build_likelihood_evaluator(expData, cfg);
%   logL = likelihoodEval.computeLogL(simData);
%
% See also: gpr_active_learning, bayesian_model_selection

% Author: [Your name]
% Date: 2025

%% Extract experimental data
Rmatrix = expData.Rmatrix;
Rdotmatrix = expData.Rdotmatrix;
tmatrix = expData.tmatrix;
mask = expData.mask;
sigmaR0 = expData.sigmaR0;
sigmaRdot0 = expData.sigmaRdot0;
tc_vec = expData.tc(:)';

[nTime, nTrials] = size(Rmatrix);

%% Setup heteroscedastic weights
if cfg.useHetero
    % Strain rate for weighting
    tc_row = repmat(max(tc_vec, eps), nTime, 1);
    epsdot_star_full = -2 .* (Rdotmatrix ./ max(Rmatrix, 1e-12)) .* tc_row;
    epsdot_th_mat = build_srstar_thresholds(tc_vec, nTime, cfg.Thresholds);
    
    % Logistic transition around threshold
    z = (epsdot_th_mat - abs(epsdot_star_full)) ./ max(epsdot_th_mat, eps);
    a_full = 1 ./ (1 + exp(-cfg.kappa .* z));
    w_full = min(max(cfg.m_floor + (1 - cfg.m_floor) .* a_full, cfg.m_floor), 1);
else
    w_full = ones(size(Rmatrix));
end

%% Beta grid for noise marginalization
betaGrid = cfg.betaGrid(:)';
nBeta = numel(betaGrid);

% Half-Cauchy prior on beta
logw_beta = normalize_beta_weights(betaGrid, @(b) (2/pi) ./ (1 + b.^2));

%% Build likelihood evaluator
likelihoodEval = struct();
likelihoodEval.cfg = cfg;

% Main function: compute log-likelihood marginalized over beta
likelihoodEval.computeLogL = @(simData) compute_marginalized_logL(...
    simData, Rmatrix, Rdotmatrix, tmatrix, mask, ...
    sigmaR0, sigmaRdot0, w_full, betaGrid, logw_beta, cfg);

% Beta posterior (useful for diagnostics)
likelihoodEval.getBetaPosterior = @(simData) compute_beta_posterior(...
    simData, Rmatrix, Rdotmatrix, tmatrix, mask, ...
    sigmaR0, sigmaRdot0, w_full, betaGrid, logw_beta, cfg);

end

%% ==================== Core Likelihood Computation ====================

function logL_marg = compute_marginalized_logL(simData, Rmatrix, Rdotmatrix, ...
    tmatrix, mask, sigmaR0, sigmaRdot0, w_full, betaGrid, logw_beta, cfg)
% Compute log-likelihood marginalized over beta

% Interpolate simulation to experimental time stamps
[R_sim, Rdot_sim] = interpolate_simulation(simData, tmatrix);

% Compute log-likelihood for each beta
ll_beta = zeros(size(betaGrid));

for b = 1:numel(betaGrid)
    beta = betaGrid(b);
    
    % Scaled variances with heteroscedastic weighting
    vR_base = (beta^2) * sigmaR0.^2;
    vRd_base = (beta^2) * sigmaRdot0.^2;
    
    if cfg.useHetero
        vR = vR_base ./ max(w_full, cfg.m_floor);
        vRd = vRd_base ./ max(w_full, cfg.m_floor);
    else
        vR = vR_base;
        vRd = vRd_base;
    end
    
    % Compute likelihood over all trials
    ll_total = 0;
    for j = 1:size(Rmatrix, 2)
        use = mask(:, j) & isfinite(tmatrix(:, j)) & ...
              isfinite(Rmatrix(:, j)) & isfinite(Rdotmatrix(:, j));
        if ~any(use), continue; end
        
        % Residuals
        rR = Rmatrix(use, j) - R_sim(use, j);
        rRd = Rdotmatrix(use, j) - Rdot_sim(use, j);
        
        vR_use = vR(use, j);
        vRd_use = vRd(use, j);
        
        % Gaussian log-likelihood
        ll_R = -0.5 * (sum((rR.^2) ./ vR_use) + sum(log(2*pi*vR_use)));
        
        if cfg.useRdotInLL
            ll_Rd = -0.5 * (sum((rRd.^2) ./ vRd_use) + sum(log(2*pi*vRd_use)));
            ll_total = ll_total + ll_R + ll_Rd;
        else
            ll_total = ll_total + ll_R;
        end
    end
    
    ll_beta(b) = ll_total;
end

% Marginalize over beta using log-sum-exp
logL_marg = logsumexp(ll_beta + logw_beta);

end

function post_beta = compute_beta_posterior(simData, Rmatrix, Rdotmatrix, ...
    tmatrix, mask, sigmaR0, sigmaRdot0, w_full, betaGrid, logw_beta, cfg)
% Compute posterior distribution over beta for diagnostics

% Same as above but return normalized posterior
ll_beta = zeros(size(betaGrid));

[R_sim, Rdot_sim] = interpolate_simulation(simData, tmatrix);

for b = 1:numel(betaGrid)
    beta = betaGrid(b);
    
    vR_base = (beta^2) * sigmaR0.^2;
    vRd_base = (beta^2) * sigmaRdot0.^2;
    
    if cfg.useHetero
        vR = vR_base ./ max(w_full, cfg.m_floor);
        vRd = vRd_base ./ max(w_full, cfg.m_floor);
    else
        vR = vR_base;
        vRd = vRd_base;
    end
    
    ll_total = 0;
    for j = 1:size(Rmatrix, 2)
        use = mask(:, j) & isfinite(tmatrix(:, j)) & ...
              isfinite(Rmatrix(:, j)) & isfinite(Rdotmatrix(:, j));
        if ~any(use), continue; end
        
        rR = Rmatrix(use, j) - R_sim(use, j);
        rRd = Rdotmatrix(use, j) - Rdot_sim(use, j);
        
        vR_use = vR(use, j);
        vRd_use = vRd(use, j);
        
        ll_R = -0.5 * (sum((rR.^2) ./ vR_use) + sum(log(2*pi*vR_use)));
        
        if cfg.useRdotInLL
            ll_Rd = -0.5 * (sum((rRd.^2) ./ vRd_use) + sum(log(2*pi*vRd_use)));
            ll_total = ll_total + ll_R + ll_Rd;
        else
            ll_total = ll_total + ll_R;
        end
    end
    
    ll_beta(b) = ll_total;
end

% Normalize to get posterior
log_post = ll_beta + logw_beta;
post_beta = exp(log_post - logsumexp(log_post));

end

%% ==================== Helper Functions ====================

function [R_sim, Rdot_sim] = interpolate_simulation(simData, tmatrix)
% Interpolate simulation data to experimental time points
% simData should be a structure with fields: t, R, Rdot

t_sim = simData.t(:);
R_mean = simData.R(:);
Rdot_mean = simData.Rdot(:);

% Ensure monotonic time
[t_sim, uq] = unique(t_sim, 'stable');
R_mean = R_mean(uq);
Rdot_mean = Rdot_mean(uq);

% Sort if needed
if any(diff(t_sim) <= 0)
    [t_sim, is] = sort(t_sim);
    R_mean = R_mean(is);
    Rdot_mean = Rdot_mean(is);
end

% Interpolate to each trial's time points
[nTime, nTrials] = size(tmatrix);
R_sim = zeros(nTime, nTrials);
Rdot_sim = zeros(nTime, nTrials);

for j = 1:nTrials
    t_j = tmatrix(:, j);
    valid = isfinite(t_j) & (t_j >= min(t_sim)) & (t_j <= max(t_sim));
    
    if any(valid)
        R_sim(valid, j) = interp1(t_sim, R_mean, t_j(valid), 'linear', 'extrap');
        Rdot_sim(valid, j) = interp1(t_sim, Rdot_mean, t_j(valid), 'linear', 'extrap');
    end
end

end

function epsdot_th_mat = build_srstar_thresholds(tc_vec_row, nTime, Thresholds)
% Build strain rate threshold matrix
tc_vec_row = max(tc_vec_row, eps);

switch lower(string(Thresholds.mode))
    case "dim"
        epsdot_th_trial = max(Thresholds.sr_dim, 0) .* tc_vec_row;
    case "nd"
        nd_val = Thresholds.sr_nd;
        if isempty(nd_val) || ~isfinite(nd_val) || nd_val <= 0
            error('Thresholds.sr_nd must be positive when mode="nd".');
        end
        epsdot_th_trial = nd_val .* ones(size(tc_vec_row));
    otherwise
        base = Thresholds.auto_base;
        if isempty(base) || ~isfinite(base) || base <= 0
            base = 1e5;
        end
        epsdot_th_trial = base .* tc_vec_row;
end

epsdot_th_mat = repmat(epsdot_th_trial, nTime, 1);
end

function logw_norm = normalize_beta_weights(betaGrid, priorFcn)
% Normalize quadrature weights for beta integration
B = numel(betaGrid);
if B == 1
    logw_norm = 0;
    return;
end

% Trapezoidal rule spacing
d = zeros(size(betaGrid));
d(1) = 0.5 * (betaGrid(2) - betaGrid(1));
d(end) = 0.5 * (betaGrid(end) - betaGrid(end-1));
for b = 2:B-1
    d(b) = 0.5 * (betaGrid(b+1) - betaGrid(b-1));
end

% Prior-weighted quadrature
prior = priorFcn(betaGrid);
raw = prior .* max(d, eps);

% Normalize in log space
logw_raw = log(raw);
m = max(logw_raw);
logw_norm = logw_raw - (m + log(sum(exp(logw_raw - m))));
end

function s = logsumexp(a)
% Stable log-sum-exp
amax = max(a(:));
if ~isfinite(amax)
    s = amax;
    return;
end
s = amax + log(sum(exp(a(:) - amax)));
end