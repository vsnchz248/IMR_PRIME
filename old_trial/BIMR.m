function BIMR
% BIMR  GPR-based Bayesian model selection for IMR (GPR + parallel sims)
%
% Design:
%   " Models run in SERIAL (one after another)
%   " For each model, NLL evaluations over ¸-batches are done in PARALLEL (parfor)
%   " GPR-based active integration (active_integrate_logaware) per model
%
% Output: results_gpr.mat

tic;

%% 0) Paths
thisFile = which(mfilename);
assert(~isempty(thisFile),'Cannot resolve BIMR.m on path.');
baseDir   = fileparts(thisFile);
solverDir = fullfile(baseDir,'..','IMRv2','src','forward_solver');
if exist(solverDir,'dir')==7, addpath(solverDir); end
addpath(baseDir);

%% 1) Config
filename = fullfile(baseDir,'results_gpr.mat');

% Models to test
modelNames = {'newtonian','nh','nhkv','qnh','qkv','linmax','sls'};

% GPR settings
cfg = struct();
cfg.gprOpts = struct();
cfg.gprOpts.maxRounds = 30;      % Reduced from 40
cfg.gprOpts.tolRelCI  = 0.05;    % 5% relative GP std target
cfg.gprOpts.rngSeed   = 42;
cfg.gprOpts.verbose   = true;    % Let active_integrate_logaware talk

% Likelihood settings
cfg.useBICprior = true;

%% 2) Experimental data via prepare_data
material_id = 0;
dataOpts = struct();
dataOpts.verbose = true;
dataOpts.kappa   = 1;
dataOpts.m_floor = 0.10;

fprintf('\n=== Loading experimental data ===\n');
expData = prepare_data(material_id, dataOpts);

cfg.N_eff = 2 * nnz(expData.mask);

fprintf('Material: %s\n', expData.material_name);
fprintf('Trials: %d, Effective N: %d\n\n', size(expData.Rmatrix,2), cfg.N_eff);

%% 3) Build priors
fprintf('=== Building priors ===\n');
priors_local = build_model_priors(expData);
priors_local.N_eff = cfg.N_eff;
priors_local.paramPrior = @(modelName, theta) param_prior_hard_cutoff(modelName, theta);

fprintf('Axis needs: elastic=%.3f, maxwell=%.3f, nonlinear=%.3f\n\n', ...
        priors_local.axis_need.elastic, ...
        priors_local.axis_need.maxwell, ...
        priors_local.axis_need.nonlinear);

%% 4) Score models in SERIAL (simulations parallel via parfor)
nModels = numel(modelNames);
Models(nModels) = empty_model_struct();

fprintf('=== Running GPR integration for %d models (serial models, parallel sims) ===\n', nModels);

tStart = tic;
for i = 1:nModels
    fprintf('\n--- MODEL %s (%d/%d) ---\n', upper(modelNames{i}), i, nModels);

    % Direct call (no parfeval): active_integrate_logaware will be verbose
    Models(i) = score_one_model_gpr(modelNames{i}, expData, priors_local, cfg);

    out = Models(i);
    fprintf('[%s] Complete: log10(Z)=%.5g, time=%.1fs, MAP=[', ...
            upper(out.name), out.log10Z, out.time_seconds);
    fprintf('%.4g ', out.mapTheta);
    fprintf(']\n');
end
fprintf('Total time: %.1f seconds\n\n', toc(tStart));

%% 5) Posterior over models
logZ_vec = [Models.logZ];
k_params = [Models.k_params];

e_need  = priors_local.axis_need.elastic;
m_need  = priors_local.axis_need.maxwell;
nl_need = priors_local.axis_need.nonlinear;

logpM_data = zeros(1, nModels);
for i = 1:nModels
    logpM_data(i) = compute_model_data_prior(modelNames{i}, e_need, m_need, nl_need);
end

if cfg.useBICprior
    logpM_BIC = -0.5 .* k_params .* log(max(cfg.N_eff, 1));
    logpM = logpM_BIC + logpM_data;
else
    logpM = logpM_data;
end

x = logZ_vec + logpM;
Fmask = isfinite(x);
if ~any(Fmask)
    post = ones(1, nModels) / nModels;
else
    Z = safe_logsumexp(x(Fmask));
    post = zeros(1, nModels);
    post(Fmask) = exp(x(Fmask) - Z);
end

%% 6) Print results
fprintf('\n=== RESULTS (GPR-based BIMR) ===\n');
fprintf('%-10s %12s %12s %12s %12s\n', 'Model', 'k', 'log10(Z)', 'log(Prior)', 'P(M|D)');
fprintf('---------------------------------------------------------------\n');
for i = 1:nModels
    fprintf('%-10s %12d %12.5g %12.5g %12.6f\n', ...
        upper(modelNames{i}), k_params(i), Models(i).log10Z, logpM(i), post(i));
end
fprintf('===============================================================\n');

[~, bestIdx] = max(post);
fprintf('\nWinner: %s (P=%.4f)\n', upper(modelNames{bestIdx}), post(bestIdx));

fprintf('\nMAP parameters:\n');
for i = 1:nModels
    fprintf('  %s: ', upper(modelNames{i}));
    fprintf('%.4g ', Models(i).mapTheta);
    fprintf('\n');
end

%% 7) Save
S = struct();
S.Models     = Models;
S.post       = post;
S.logZ_vec   = logZ_vec;
S.modelNames = modelNames;
S.cfg        = cfg;
S.expData    = expData;
S.priors     = priors_local;

save(filename, '-struct', 'S');
fprintf('\nSaved to: %s\n', filename);

toc;
end

%% ============ CORE SCORER ============
function out = score_one_model_gpr(modelName, expData, priors, cfg)

    tModelStart = tic;  % start timing this model
    
    bounds  = get_log10_bounds(modelName);
    axNames = axis_names_for_model(modelName);
    k       = numel(axNames);
    
    xmin = zeros(1,k);
    xmax = zeros(1,k);
    for j = 1:k
        field = axNames{j};
        if strcmp(field, 'g'), field = 'G'; end
        xmin(j) = 10^bounds.(field)(1);
        xmax(j) = 10^bounds.(field)(2);
    end
    
    % NLL function: batch evaluation (parallel over sims inside imr_nll_batch)
    nllFcn = @(X) imr_nll_batch(X, modelName, expData);
    
    % Run GPR integration
    res = active_integrate_logaware(nllFcn, xmin, xmax, cfg.gprOpts, modelName, priors);
    
    logZ   = res.logI_mean;
    log10Z = res.log10I_mean;
    
    [~, mapIdx] = max(res.Y);
    mapTheta    = res.fromFeat(res.U(mapIdx,:));
    
    out = struct();
    out.name         = modelName;
    out.logZ         = logZ;
    out.log10Z       = log10Z;
    out.k_params     = k;
    out.nGrid        = size(res.U,1);
    out.mapTheta     = mapTheta;
    out.gpr_result   = res;
    out.time_seconds = toc(tModelStart);   % elapsed time for this model
end

%% ============ NLL FUNCTION (PARALLEL OVER SAMPLES) ============
function translate_name = translate_model_name_for_solver(modelName)
    switch lower(strtrim(modelName))
        case {'newtonian','newt'}, translate_name = 'newt';
        case {'nhkv','kv'},        translate_name = 'kv';
        case 'nh',                 translate_name = 'nh';
        case 'qnh',                translate_name = 'qnh';
        case {'linmax','max','lm'},translate_name = 'lm';
        case 'qkv',                translate_name = 'qkv';
        case 'sls',                translate_name = 'sls';
        otherwise,                 translate_name = lower(modelName);
    end
end

function NLL = imr_nll_batch(X, modelName, expData)
    % IMR_NLL_BATCH  Parallel evaluation of NLL for a batch of ¸ points.
    %
    % X   : [N × d] matrix of parameter vectors
    % NLL : [N × 1] vector of NLL values

    N        = size(X,1);
    NLL      = zeros(N,1);   % sliced variable for parfor
    BIG_PEN  = 1e6;
    MAX_NLL  = 1e5;

    solver_name = translate_model_name_for_solver(modelName);

    parfor i = 1:N
        theta_i = X(i,:);
        nll_i   = BIG_PEN;   % default penalty

        try
            % Bounds check
            if ~param_out_of_bounds(modelName, theta_i)

                % Forward solve
                sim = imr_forward_solver(theta_i, solver_name, expData);

                if sim.ok
                    nll_tmp = compute_nll_from_sim(sim, expData);

                    % Guard against weird values
                    if isfinite(nll_tmp)
                        if nll_tmp < 0
                            nll_tmp = BIG_PEN;
                        elseif nll_tmp > MAX_NLL
                            nll_tmp = MAX_NLL;
                        end
                        nll_i = nll_tmp;
                    end
                end
            end
        catch
            % keep nll_i = BIG_PEN
        end

        NLL(i) = nll_i;
    end
end

function nll = compute_nll_from_sim(sim, expData)
    % Beta marginalization: -log p(D|¸) H (N/2)*log(Ç²)
    
    [~, nTrials] = size(expData.Rmatrix);
    
    t_sim_nd    = sim.t_nd;
    R_sim_nd    = sim.R_nd;
    Rdot_sim_nd = sim.Rdot_nd;
    
    chi2_total = 0;
    n_total    = 0;
    
    for j = 1:nTrials
        mask_j = expData.mask(:,j);
        if ~any(mask_j), continue; end
        
        t_exp_nd    = expData.tmatrix(mask_j, j);
        R_exp_nd    = expData.Rmatrix(mask_j, j);
        Rdot_exp_nd = expData.Rdotmatrix(mask_j, j);
        
        R_pred    = interp1(t_sim_nd, R_sim_nd,    t_exp_nd, 'linear', 'extrap');
        Rdot_pred = interp1(t_sim_nd, Rdot_sim_nd, t_exp_nd, 'linear', 'extrap');
        
        if any(~isfinite(R_pred)) || any(~isfinite(Rdot_pred))
            nll = 1e6;
            return;
        end
        
        rR    = R_exp_nd    - R_pred;
        rRdot = Rdot_exp_nd - Rdot_pred;
        
        w_j           = expData.weights_w(mask_j, j);
        sigma0_R_j    = expData.sigma0_R(mask_j, j);
        sigma0_Rdot_j = expData.sigma0_Rdot(mask_j, j);
        
        valid_R    = (w_j > 1e-6) & isfinite(rR)    & (sigma0_R_j    > 1e-10);
        valid_Rdot = (w_j > 1e-6) & isfinite(rRdot) & (sigma0_Rdot_j > 1e-10);
        
        if ~any(valid_R) && ~any(valid_Rdot), continue; end
        
        if any(valid_R)
            chi2_total = chi2_total + sum(w_j(valid_R)    .* (rR(valid_R)    ./ sigma0_R_j(valid_R)).^2);
            n_total    = n_total    + nnz(valid_R);
        end
        
        if any(valid_Rdot)
            chi2_total = chi2_total + sum(w_j(valid_Rdot) .* (rRdot(valid_Rdot) ./ sigma0_Rdot_j(valid_Rdot)).^2);
            n_total    = n_total    + nnz(valid_Rdot);
        end
    end
    
    if n_total < 2 || chi2_total <= 0
        nll = 1e6;
        return;
    end
    
    nll = 0.5 * n_total * log(chi2_total);
    
    if ~isfinite(nll) || nll < 0
        nll = 1e6;
    elseif nll > 1e5
        nll = 1e5;
    end
end

%% ============ PRIORS & HELPERS ============
function logP = param_prior_hard_cutoff(modelName, theta)
    if param_out_of_bounds(modelName, theta)
        logP = -inf;
    else
        logP = 0;
    end
end

function out_of_bounds = param_out_of_bounds(modelName, theta)
    bounds  = get_log10_bounds(modelName);
    axNames = axis_names_for_model(modelName);
    
    out_of_bounds = false;
    for j = 1:numel(axNames)
        field = axNames{j};
        if strcmp(field, 'g'), field = 'G'; end
        
        log_theta_j = log10(theta(j));
        if log_theta_j < bounds.(field)(1) || log_theta_j > bounds.(field)(2)
            out_of_bounds = true;
            return;
        end
    end
end

function logpM = compute_model_data_prior(modelName, e_need, m_need, nl_need)
    switch lower(modelName)
        case {'newtonian','newt'}, f = 1;
        case 'nh',                 f = e_need;
        case {'nhkv','kv'},        f = e_need;
        case 'qnh',                f = e_need * nl_need;
        case 'qkv',                f = e_need * nl_need;
        case {'linmax','max'},     f = m_need;
        case 'sls',                f = (e_need^2) * m_need;
        otherwise,                 f = 1;
    end
    logpM = log(max(f, 1e-12));
end

%% ============ UTILITIES ============
function S = empty_model_struct()
    S = struct('name','', ...
               'logZ',NaN, ...
               'log10Z',NaN, ...
               'k_params',NaN, ...
               'nGrid',NaN, ...
               'mapTheta',[], ...
               'gpr_result',[], ...
               'time_seconds',NaN);  % preallocate timing field
end

function s = safe_logsumexp(a)
    if isempty(a) || all(~isfinite(a))
        s = -inf;
    else
        amax = max(a(:));
        s    = amax + log(sum(exp(a(:) - amax)));
    end
end

function names = axis_names_for_model(modelName)
    switch lower(strtrim(modelName))
        case 'qkv',                names = {'mu','g','alpha'};
        case 'sls',                names = {'mu','g','lambda1'};
        case {'nhkv','kv'},        names = {'mu','g'};
        case 'qnh',                names = {'g','alpha'};
        case {'linmax','max'},     names = {'mu','lambda1'};
        case 'nh',                 names = {'g'};
        case {'newt','newtonian'}, names = {'mu'};
        otherwise, error('Unknown model: %s', modelName);
    end
end

function bounds = get_log10_bounds(modelName)
    m = lower(strtrim(modelName));
    switch m
        case {'newt','newtonian'}
            bounds.mu = [-4, 0];
        case 'nh'
            bounds.G = [2, log10(5e5)];
        case {'nhkv','kv'}
            bounds.mu = [-4, 0];
            bounds.G  = [2, log10(5e5)];
        case 'qnh'
            bounds.G     = [2, log10(5e5)];
            bounds.alpha = [-3, 1];
        case {'max','linmax'}
            bounds.mu      = [-4, 0];
            bounds.lambda1 = [-7, -3];
        case 'qkv'
            bounds.mu     = [-4, 0];
            bounds.G      = [2, log10(5e5)];
            bounds.alpha  = [-3, 1];
        case 'sls'
            bounds.mu      = [-4, 0];
            bounds.G       = [2, log10(5e5)];
            bounds.lambda1 = [-7, -3];
        otherwise
            error('Unknown model: %s', modelName);
    end
end
