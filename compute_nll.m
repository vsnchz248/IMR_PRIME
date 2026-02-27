function NLL_beta = compute_nll(theta_matrix, model_name, expData, solverDir, beta_grid)
% COMPUTE_NLL  Negative log-likelihood for parameter samples with beta grid
%
% CORRECTED VERSION: Returns NLL for each (theta, beta) combination
%   NLL_beta(i, b) = -log P(D | theta(i,:), beta(b), M)
%
% PERFECTLY CONSISTENT WITH BIMR EQ. 14:
%   P(D|M,θ,β) = ∏∏ [1/√(2π Var_R)] exp[-(r_R)²/(2 Var_R)] 
%                  × [1/√(2π Var_Rdot)] exp[-(r_Rdot)²/(2 Var_Rdot)]
%
% Inputs:
%   theta_matrix - [N × d] parameter samples
%   model_name   - constitutive model name
%   expData      - experimental data struct
%   solverDir    - path to forward solver
%   beta_grid    - [1 × N_beta] noise scale values
%
% Output:
%   NLL_beta     - [N × N_beta] matrix of negative log-likelihoods

if nargin < 5
    % Default: single beta = 1 (for compatibility)
    beta_grid = 1.0;
end

N = size(theta_matrix, 1);
N_beta = numel(beta_grid);
NLL_beta = zeros(N, N_beta);

% Unpack experimental data (make local copies for parfor)
Rmatrix     = expData.Rmatrix;
Rdotmatrix  = expData.Rdotmatrix;
tmatrix     = expData.tmatrix;
mask        = expData.mask;
sigma0_R    = expData.sigma0_R;
sigma0_Rdot = expData.sigma0_Rdot;
weights_w   = expData.weights_w;

% Mean nondimensional time
tvector_nd = mean(tmatrix, 2, 'omitnan');

% Dimensional parameters
R0  = expData.Rmax_mean;
Req = mean(expData.Req_each, 'omitnan');
tc  = R0 * sqrt(expData.rho / expData.p_inf);

% Convert to dimensional time for solver
tvector_dim = tvector_nd * tc;

% Solver options
solver_opts = struct('radial', 2, 'bubtherm', 0, 'medtherm', 0, ...
                     'masstrans', 0, 'vapor', 1, 'collapse', 0, ...
                     'nt', 100, 'mt', 100);

% Add solver path
if ~isempty(solverDir) && exist(solverDir, 'dir') == 7
    addpath(solverDir);
end

nTrials = size(Rmatrix, 2);

% Compute total number of data points for normalization
n_total_data = 2 * nnz(mask);  % Both R and Rdot observations

% Compute a reasonable penalty for solver failures based on BIMR Eq. 14
% Worst case: all residuals = max observed value, variance = median baseline
% 
% From BIMR Eq. 14, the Gaussian NLL is:
%   NLL = 0.5 * [Σ(r²/σ²) + Σ log(2πσ²)]
%
% For a catastrophic failure, assume:
%   - All residuals r ≈ max(data)  [complete mismatch]
%   - Variance σ² ≈ median(σ0²)    [typical experimental noise]

max_R = median(abs(Rmatrix(mask)));
max_Rdot = median(abs(Rdotmatrix(mask)));
med_var_R = median(sigma0_R(mask).^2);
med_var_Rdot = median(sigma0_Rdot(mask).^2);

% Penalty NLL per BIMR Eq. 14 (worst case but finite)
% Note: Each term is summed over n_total_data points
penalty_nll = 0.5 * (n_total_data * max_R^2/med_var_R + ...      % Σ(r_R²/σ_R²)
                     n_total_data * max_Rdot^2/med_var_Rdot + ...% Σ(r_Rdot²/σ_Rdot²)
                     n_total_data * log(2*pi*med_var_R) + ...    % Σ log(2πσ_R²)
                     n_total_data * log(2*pi*med_var_Rdot));     % Σ log(2πσ_Rdot²)
fprintf('penalty_nll = %.4e\n', penalty_nll);
% PARALLEL LOOP over theta
parfor i = 1:N
    theta = theta_matrix(i,:);
    
    % Run solver with dimensional time
    [t_sim_nd, R_sim_nd, Rdot_sim_nd, solver_success] = ...
        run_forward_solver_parfor(solver_opts, tvector_dim, R0, Req, theta, model_name);
    
    % Initialize row for this theta
    nll_row = zeros(1, N_beta);
    
    if ~solver_success
        % Use data-scaled penalty instead of arbitrary huge number
        nll_row(:) = penalty_nll;
    else
        % Compute likelihood for EACH beta value
        for b = 1:N_beta
            beta = beta_grid(b);
            
            % Compute likelihood across trials for this beta
            ll_sum = 0;
            n_points = 0;
            
            for j = 1:nTrials
                use = mask(:,j);
                if ~any(use), continue; end
                
                t_j = tmatrix(use, j);
                
                % Interpolate
                R_th    = interp1(t_sim_nd, R_sim_nd,    t_j, 'linear', R_sim_nd(end));
                Rdot_th = interp1(t_sim_nd, Rdot_sim_nd, t_j, 'linear', Rdot_sim_nd(end));
                
                if any(~isfinite(R_th)) || any(~isfinite(Rdot_th))
                    continue;
                end
                
                % Residuals
                rR    = Rmatrix(use,j) - R_th;
                rRdot = Rdotmatrix(use,j) - Rdot_th;
                
                % Variances (SCALED BY BETA^2 per BIMR Eq. 12)
                w_j   = weights_w(use, j);
                vR    = (beta^2 * sigma0_R(use,j).^2)    ./ max(w_j, 1e-12);
                vRdot = (beta^2 * sigma0_Rdot(use,j).^2) ./ max(w_j, 1e-12);
                
                % Log-likelihood per point (BIMR Eq. 14)
                % P(D|θ,β) = ∏∏ [1/√(2πσ²)] exp[-r²/(2σ²)]
                % log P = Σ[-0.5(r²/σ² + log(2πσ²))]
                n_j = nnz(use);
                ll_R    = -0.5 * (sum((rR.^2)    ./ vR)    + sum(log(2*pi*vR)));
                ll_Rdot = -0.5 * (sum((rRdot.^2) ./ vRdot) + sum(log(2*pi*vRdot)));
                
                % Accumulate total log-likelihood (not normalized per point)
                ll_sum = ll_sum + ll_R + ll_Rdot;
                n_points = n_points + 1;
            end
            
            % Store NLL for this beta (negative of total log-likelihood)
            if n_points > 0
                nll_row(b) = -ll_sum;
            else
                nll_row(b) = penalty_nll;
            end
        end
        
        % Only sanitize NaN values
        nll_row(isnan(nll_row)) = penalty_nll;
    end
    
    % Assign complete row
    NLL_beta(i, :) = nll_row;
end

% if N <= 20 && N_beta == 1
%     fprintf('NLL range: [%.2e, %.2e]\n\n', min(NLL_beta(:)), max(NLL_beta(:)));
% end

end

%% ==================== Helper Functions ====================

function [t_nd, R_nd, Rdot_nd, success] = run_forward_solver_parfor(opts, tvector_dim, R0, Req, theta, model_name)
% Separate function for parfor compatibility

[mu, G, lambda1, alpha] = unpack_theta(theta, model_name);

switch lower(model_name)
    case {'newtonian', 'newt', 'nh', 'kv'}
        stress_idx = 1;
    case {'qnh', 'qkv'}
        stress_idx = 2;
    case {'linmax', 'max', 'lm', 'sls'}
        stress_idx = 3;
    otherwise
        stress_idx = 1;
end

success = false;

try
    % f_imr_fd: dimensional input → nondimensional output
    [t_nd, R_nd, Rdot_nd, ~] = f_imr_fd( ...
        'radial',    opts.radial, ...
        'stress',    stress_idx, ...
        'bubtherm',  opts.bubtherm, ...
        'medtherm',  opts.medtherm, ...
        'masstrans', opts.masstrans, ...
        'vapor',     opts.vapor, ...
        'tvector',   tvector_dim, ...
        'r0',        R0, ...
        'req',       Req, ...
        'mu',        mu, ...
        'g',         G, ...
        'lambda1',   lambda1, ...
        'lambda2',   0, ...
        'alphax',    alpha, ...
        'collapse',  opts.collapse, ...
        'nt',        opts.nt, ...
        'mt',        opts.mt);
    
    success = true;
    
catch
    % Silent failure in parfor
    t_nd = tvector_dim;
    R_nd = ones(size(tvector_dim));
    Rdot_nd = zeros(size(tvector_dim));
end

end

function [mu, G, lambda1, alpha] = unpack_theta(theta, model_name)

mu = 0; G = 0; lambda1 = 0; alpha = 0;

switch lower(model_name)
    case {'newtonian', 'newt'}
        mu = theta(1);
    case 'nh'
        G = theta(1);
    case 'kv'
        mu = theta(1);
        G = theta(2);
    case 'qnh'
        G = theta(1);
        alpha = theta(2);
    case {'linmax', 'max', 'lm'}
        mu = theta(1);
        lambda1 = theta(2);
    case 'qkv'
        mu = theta(1);
        G = theta(2);
        alpha = theta(3);
    case 'sls'
        mu = theta(1);
        G = theta(2);
        lambda1 = theta(3);
end

end