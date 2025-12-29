function NLL = compute_nll(theta_matrix, model_name, expData, solverDir)
% COMPUTE_NLL  Negative log-likelihood for parameter samples (PARALLELIZED)
%
% CRITICAL: f_imr_fd expects DIMENSIONAL time input, returns NONDIMENSIONAL outputs

N = size(theta_matrix, 1);
NLL = zeros(N, 1);

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
solver_opts = struct('radial', 2, 'bubtherm', 1, 'medtherm', 0, ...
                     'masstrans', 1, 'vapor', 1, 'collapse', 0, ...
                     'nt', 100, 'mt', 100);

% Add solver path (do this BEFORE parfor)
if ~isempty(solverDir) && exist(solverDir, 'dir') == 7
    addpath(solverDir);
end

nTrials = size(Rmatrix, 2);

% PARALLEL LOOP
parfor i = 1:N
    theta = theta_matrix(i,:);
    
    % Run solver with dimensional time
    [t_sim_nd, R_sim_nd, Rdot_sim_nd, solver_success] = ...
        run_forward_solver_parfor(solver_opts, tvector_dim, R0, Req, theta, model_name);
    
    if ~solver_success
        NLL(i) = 1e12;
        continue;
    end
    
    % Compute likelihood across trials
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
        
        % Variances
        w_j   = weights_w(use, j);
        vR    = (sigma0_R(use,j).^2)    ./ max(w_j, 1e-12);
        vRdot = (sigma0_Rdot(use,j).^2) ./ max(w_j, 1e-12);
        
        % Log-likelihood per point
        n_j = nnz(use);
        ll_R    = -0.5 * (sum((rR.^2)    ./ vR)    + sum(log(2*pi*vR)));
        ll_Rdot = -0.5 * (sum((rRdot.^2) ./ vRdot) + sum(log(2*pi*vRdot)));
        
        ll_sum = ll_sum + (ll_R + ll_Rdot) / n_j;
        n_points = n_points + 1;
    end
    
    if n_points > 0
        ll_avg = ll_sum / n_points;
    else
        ll_avg = -1e6;
    end
    
    NLL(i) = -ll_avg;
    
    if ~isfinite(NLL(i))
        NLL(i) = 1e12;
    end
end

if N <= 20
    fprintf('NLL range: [%.2e, %.2e]\n\n', min(NLL), max(NLL));
end

end

%% ==================== Helper Functions ====================

function [t_nd, R_nd, Rdot_nd, success] = run_forward_solver_parfor(opts, tvector_dim, R0, Req, theta, model_name)
% Separate function for parfor compatibility (no nested functions allowed)

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
    % f_imr_fd: dimensional input  nondimensional output
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
    % Silent failure in parfor (no warnings allowed)
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