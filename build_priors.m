function priors = build_priors(expData, opts)
% BUILD_PRIORS  Physics-informed priors matching BIMR paper + original grid code
%
% This version is designed to:
%   - Use the same parameter ranges and stress-based axes as in build_model_priors.m
%   - Implement the Half–Cauchy noise-scale prior P(β) (Eqs. 15–18)
%   - Implement the BIC-like model prior P(M) ∝ exp(-k_M/2 log N_eff) (Eq. 28)
%   - Provide a CONTINUOUS parameter prior P(θ|M) usable with GPR / RQMC
%
% Inputs:
%   expData : struct from prepare_data(), should contain at least:
%       .Rmatrix      [Nt x J]  nondimensional radius
%       .Rdotmatrix   [Nt x J]  nondimensional velocity
%       .tmatrix      [Nt x J]  nondimensional time
%       .tc           [1 x J]   characteristic times t_c,j
%
%     Optional fields (if present they are used exactly as in the original code):
%       .Rmax_range   [1 x J]   per-trial R_max
%       .Req_each     [1 x J]   per-trial equilibrium radius R_eq
%       .strain       [Nt x J]  precomputed Hencky strain ε*
%       .strainRate   [Nt x J]  precomputed strain rate εdot*
%
%   opts   : optional struct
%       .models     : cellstr of model tags {'newt','nh','kv','qnh','lm','qkv','sls'}
%       .paramRanges: struct with fields mu,G,lambda1,alpha (overrides defaults)
%       .betaGrid   : vector of β values (default: 0.05:0.05:10)
%       .N_eff      : effective sample size (default: 2 * #gated points)
%       .verbose    : logical
%
% Outputs:
%   priors : struct with fields
%       .models        : model info (names, param order)
%       .paramPrior    : @(modelName,theta) -> log P(θ|M)  (up to const.)
%       .modelPrior    : @(modelName,k_params) -> log P(M)
%       .noisePrior    : @(beta) P(β)   (Half–Cauchy density, unnormalized)
%       .betaGrid      : β grid (row)
%       .betaWeights   : normalized quadrature weights q_b (sum=1)
%       .kernel_norms  : struct with normA, normB, maxwell_norm handle
%       .axis_need     : struct with elastic, maxwell, nonlinear needs
%       .N_eff         : effective number of scalar observations
%       .tc_ref        : reference t_c (median)
%       .ranges        : parameter ranges (mu, G, lambda1, alpha, De)

if nargin < 2, opts = struct(); end
opts = parse_options(opts);

if opts.verbose
    fprintf('\n=== Prior Construction (BIMR-consistent) ===\n');
end

%% --- Extract data and characteristic times --------------------------------
[Rmatrix, Rdotmatrix, tmatrix, tc_vec, aux] = extract_data(expData);

tc_ref = median(tc_vec(tc_vec > 0 & isfinite(tc_vec)));  % for De = lambda1 / tc_ref
if ~isfinite(tc_ref) || tc_ref <= 0
    error('build_priors:tc_ref','tc_ref must be positive and finite.');
end

%% --- Compute kernel norms and axis "need" as in original build_model_priors
if opts.verbose
    fprintf('Computing kernel norms + axis needs from data...\n');
end

[paramRanges, ranges_with_De] = get_param_ranges(opts, tc_ref);
ranges_with_De.tc_ref = tc_ref;

% Kernel norms and storage for Maxwell solve
[kernel_norms, axis_need] = compute_norms_and_axis_need( ...
    Rmatrix, Rdotmatrix, tmatrix, tc_vec, aux, paramRanges, ranges_with_De);

if opts.verbose
    fprintf('  ||A*||_w = %.3e\n', kernel_norms.normA);
    fprintf('  ||B||_w  = %.3e\n', kernel_norms.normB);
    fprintf('  Axis needs: elastic=%.3f, maxwell=%.3f, nonlinear=%.3f\n', ...
        axis_need.elastic, axis_need.maxwell, axis_need.nonlinear);
end

%% --- Models metadata (names + parameter order) -----------------------------
models = define_models(opts.models);

%% --- Effective N_eff for model prior (Eq. 28) ------------------------------
if isempty(opts.N_eff)
    % crude default: count of finite data points across all trials
    goodMask = isfinite(Rmatrix) & isfinite(Rdotmatrix) & (Rmatrix > 0);
    opts.N_eff = 2 * nnz(goodMask);  % same spirit as your earlier choice
end

%% --- Parameter prior handle P(theta | M) (continuous) ----------------------
if opts.verbose
    fprintf('Building continuous parameter priors P(theta|M)...\n');
end

paramPriorFcn = @(modelName, theta) eval_param_prior_cont( ...
    modelName, theta, models, kernel_norms, axis_need, ranges_with_De);

%% --- Model prior P(M) ∝ exp(-k_M/2 log N_eff) (Eq. 28) --------------------
if opts.verbose
    fprintf('Building model prior P(M) with BIC-like penalty...\n');
end
modelPriorFcn = @(modelName, k_params) eval_model_prior_bic( ...
    modelName, k_params, opts.N_eff);

%% --- Noise prior: Half–Cauchy with truncated quadrature (Eqs. 15–18) ------
if opts.verbose
    fprintf('Building Half–Cauchy noise prior P(beta)...\n');
end

[betaGrid, betaWeights, noisePriorFcn] = build_noise_prior(opts.betaGrid);

%% --- Package outputs -------------------------------------------------------
priors = struct();
priors.models       = models;
priors.paramPrior   = paramPriorFcn;
priors.modelPrior   = modelPriorFcn;
priors.noisePrior   = noisePriorFcn;
priors.betaGrid     = betaGrid;
priors.betaWeights  = betaWeights;
priors.kernel_norms = kernel_norms;
priors.axis_need    = axis_need;
priors.N_eff        = opts.N_eff;
priors.tc_ref       = tc_ref;
priors.ranges       = ranges_with_De;

if opts.verbose
    fprintf('=== Prior construction complete ===\n');
    fprintf('  Models: %d\n', numel(models));
    fprintf('  N_eff : %d\n\n', opts.N_eff);
end

end

%% ======================= Helper: options + ranges ==========================
function opts = parse_options(opts)
if ~isfield(opts,'models') || isempty(opts.models)
    opts.models = {'newt','nh','kv','qnh','lm','qkv','sls'};
end

if ~isfield(opts,'paramRanges')
    % Table 2 in the paper
    opts.paramRanges = struct( ...
        'mu',      [1e-4, 1e0], ...   % viscosity [Pa·s]
        'G',       [1e2 , 1e5], ...   % shear modulus [Pa]
        'lambda1', [1e-7, 1e-3], ...  % relaxation time [s]
        'alpha',   [1e-3, 1e1]  ...   % strain-stiffening [-]
    );
end

if ~isfield(opts,'betaGrid') || isempty(opts.betaGrid)
    opts.betaGrid = 0.05:0.05:10;     % [0.05,10] truncated Half–Cauchy
end

if ~isfield(opts,'N_eff')
    opts.N_eff = [];
end

if ~isfield(opts,'verbose')
    opts.verbose = true;
end
end

function [Rmatrix, Rdotmatrix, tmatrix, tc_vec, aux] = extract_data(expData)
% Minimal required fields
req = {'Rmatrix','Rdotmatrix','tmatrix','tc'};
for i = 1:numel(req)
    assert(isfield(expData,req{i}), 'expData missing field: %s', req{i});
end

Rmatrix    = expData.Rmatrix;
Rdotmatrix = expData.Rdotmatrix;
tmatrix    = expData.tmatrix;
tc_vec     = expData.tc(:)';

% Optional fields for better matching original build_model_priors
aux = struct();
if isfield(expData,'Rmax_range'), aux.Rmax_range = expData.Rmax_range(:)'; else, aux.Rmax_range = []; end
if isfield(expData,'Req_each'),   aux.Req_each   = expData.Req_each(:)';   else, aux.Req_each   = []; end
if isfield(expData,'strain'),     aux.strain     = expData.strain;         else, aux.strain     = []; end
if isfield(expData,'strainRate'), aux.strainRate = expData.strainRate;     else, aux.strainRate = []; end
end

function [paramRanges, ranges_with_De] = get_param_ranges(opts, tc_ref)
paramRanges = opts.paramRanges;
De_rng = [paramRanges.lambda1(1)/tc_ref, paramRanges.lambda1(2)/tc_ref];
ranges_with_De = struct('mu',      paramRanges.mu, ...
                        'G',       paramRanges.G, ...
                        'lambda1', paramRanges.lambda1, ...
                        'alpha',   paramRanges.alpha, ...
                        'De',      De_rng);
end

%% ====================== Kernels + axis-need from data ======================
function [kernel_norms, axis_need] = compute_norms_and_axis_need( ...
    Rmatrix, Rdotmatrix, tmatrix, tc_vec, aux, paramRanges, ranges_with_De)

[~, J] = size(Rmatrix);

A2 = 0; B2 = 0; used = 0;
D2perp = 0;

Acell = {}; Tcell = {}; Wcell = {}; Bcell = {};

for j = 1:J
    r   = Rmatrix(:,j);
    dr  = Rdotmatrix(:,j);
    tau = tmatrix(:,j);
    
    good = isfinite(r) & isfinite(dr) & (r>0);
    r   = r(good);
    dr  = dr(good);
    tau = tau(good);
    if numel(r) < 8, continue; end
    
    % Stretch λ as in original build_model_priors:
    if ~isempty(aux.Rmax_range) && ~isempty(aux.Req_each) && ...
            numel(aux.Rmax_range) >= j && numel(aux.Req_each) >= j
        q   = aux.Rmax_range(j) / max(aux.Req_each(j),1e-12);
        lam = q * r;
    else
        % Fallback: estimate R_eq from late-time median
        tailStart = max(1, floor(0.8*numel(r)));
        req_nd    = median(r(tailStart:end), 'omitnan');
        if ~isfinite(req_nd) || req_nd <= 0, req_nd = 1; end
        lam = r / req_nd;
    end
    
    % Strain and strain rate: if not precomputed, use Hencky formulae
    if ~isempty(aux.strain) && size(aux.strain,2) >= j
        eps_star = aux.strain(good,j);
    else
        eps_star = 0.5 * log(max(lam.^(-4),1e-12));
    end
    
    if ~isempty(aux.strainRate) && size(aux.strainRate,2) >= j
        epsdot_star = aux.strainRate(good,j);
    else
        epsdot_star = -2 * (dr ./ max(r,1e-12));
    end
    
    % Elliptical gate in (ε*, εdot*) space (Eqs. 4–6)
    eps_th    = 0.10 * max(abs(eps_star));
    epsdot_th = 1e5  * max(tc_vec(j), eps);
    gate = ((eps_star ./ max(eps_th,eps)).^2 + ...
            (epsdot_star ./ max(epsdot_th,eps)).^2) >= 1;
    
    if ~any(gate) || numel(tau) <= 1
        continue;
    end
    
    % Time weights on gated τ (non-uniform) – same as original code
    dtau = [diff(tau); max(tau(end) - tau(end-1), eps)];
    w = dtau;
    w(~gate) = 0;
    s = sum(w);
    if s <= 0, continue; end
    w = w / s;
    
    % Kernels
    Astar = dr ./ max(r,1e-12);                                  % viscous A*
    Bnh   = 0.5 * (4 ./ max(lam,1e-12) + lam.^(-4) - 5);         % NH elastic B
    
    A2 = A2 + sum((Astar(gate).^2) .* w(gate));
    B2 = B2 + sum((Bnh(gate)  .^2) .* w(gate));
    used = used + 1;
    
    % Nonlinear direction D* (qNH - NH) as in comments of original code
    lam_g  = lam(gate);
    bracket = 27/40 + (1/8).*lam_g.^(-8) + (1/5).*lam_g.^(-5) + ...
              lam_g.^(-2) - 2.*lam_g;
    Dstar = -3.*Bnh(gate) + 2.*bracket;
    
    X1 = Astar(gate);
    X2 = Bnh(gate);
    W  = sqrt(w(gate));
    Xw = [W.*X1, W.*X2];
    yw = W.*Dstar;
    
    XtX = Xw.' * Xw;
    Xty = Xw.' * yw;
    coef = [0;0];
    if all(isfinite(XtX(:))) && all(isfinite(Xty))
        coef = pinv(XtX) * Xty;
    end
    Dproj = coef(1).*X1 + coef(2).*X2;
    Dperp = Dstar - Dproj;
    
    D2perp = D2perp + sum((Dperp.^2) .* w(gate));
    
    % Store for Maxwell solve
    Acell{end+1} = Astar(gate); %#ok<AGROW>
    Tcell{end+1} = tau(gate);   %#ok<AGROW>
    Wcell{end+1} = w(gate);     %#ok<AGROW>
    Bcell{end+1} = Bnh(gate);   %#ok<AGROW>
end

assert(used > 0,'No valid trials after gating.');

normA = sqrt(A2 / used);
normB = sqrt(B2 / used);
maxwell_norm = @(De) maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell);

kernel_norms = struct('normA', normA, 'normB', normB, ...
                      'maxwell_norm', maxwell_norm);

% Nonlinearity magnitude
if used > 0 && D2perp > 0
    normDperp = sqrt(D2perp / used);
else
    normDperp = 0;
end

% ---------- Axis needs (elastic, maxwell, nonlinear) -----------------------
% Elastic need via residual of B relative to span{A*, M*(De)}
De_rng  = ranges_with_De.De;
De_samp = logspace(log10(De_rng(1)), log10(De_rng(2)), 64);

res2_sum = 0; B2_sum = 0; used_em = 0;
for j = 1:numel(Acell)
    A = Acell{j};
    t = Tcell{j};
    w = Wcell{j};
    B = Bcell{j};
    
    if numel(A) < 2 || numel(B) ~= numel(A), continue; end
    W  = sqrt(w(:));
    Acol = W .* A(:);
    y    = W .* B(:);
    
    best_r2 = Inf;
    for d = 1:numel(De_samp)
        De = De_samp(d);
        M = zeros(size(A));
        for i = 2:numel(A)
            dt = t(i) - t(i-1);
            M(i) = (M(i-1) + (dt/De)*A(i)) / (1 + dt/De);
        end
        X = [Acol, W.*M(:)];
        coef = pinv(X.'*X) * (X.'*y);
        r = y - X*coef;
        r2 = sum(r.^2);
        if r2 < best_r2
            best_r2 = r2;
        end
    end
    
    if isfinite(best_r2)
        res2_sum = res2_sum + best_r2;
        B2_sum   = B2_sum   + sum((W.*B(:)).^2);
        used_em  = used_em  + 1;
    end
end

if used_em > 0 && B2_sum > 0
    e_need = max(0, min(1, res2_sum / B2_sum));
else
    e_need = 0;
end
e_need = e_need.^2;   % sharpen

% Maxwell need: compare best Maxwell kernel energy vs viscous
Mnorms = arrayfun(maxwell_norm, De_samp);
Mstar  = max(Mnorms);
m_need = (Mstar^2) / max(normA^2 + Mstar^2, eps);

% Nonlinearity need: energy in D_perp vs total
if normDperp > 0
    nl_need = (normDperp^2) / max(normA^2 + normB^2 + normDperp^2, eps);
else
    nl_need = 0;
end
nl_need = max(0, min(1, nl_need));

axis_need = struct('elastic', e_need, 'maxwell', m_need, 'nonlinear', nl_need);

end

function nm = maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell)
S2 = 0; used = 0;
for k = 1:numel(Acell)
    A = Acell{k};
    t = Tcell{k};
    w = Wcell{k};
    if numel(A) < 2, continue; end
    S = zeros(size(A));
    for i = 2:numel(A)
        dt = t(i) - t(i-1);
        S(i) = (S(i-1) + (dt/De)*A(i)) / (1 + dt/De);
    end
    S2 = S2 + sum((S.^2) .* w);
    used = used + 1;
end
if used == 0
    nm = 0;
else
    nm = sqrt(S2 / used);
end
end

%% ======================== Model metadata ===================================
function models = define_models(modelNames)
models = cell(1,numel(modelNames));
for i = 1:numel(modelNames)
    name = lower(modelNames{i});
    switch name
        case 'newt'
            models{i} = struct('name','newt', ...
                               'params',{{'mu'}}, ...
                               'k_params',1);
        case 'nh'
            models{i} = struct('name','nh', ...
                               'params',{{'G'}}, ...
                               'k_params',1);
        case 'kv'
            models{i} = struct('name','kv', ...
                               'params',{{'mu','G'}}, ...
                               'k_params',2);
        case 'qnh'
            models{i} = struct('name','qnh', ...
                               'params',{{'G','alpha'}}, ...
                               'k_params',2);
        case 'lm'
            models{i} = struct('name','lm', ...
                               'params',{{'mu','lambda1'}}, ...
                               'k_params',2);
        case 'qkv'
            models{i} = struct('name','qkv', ...
                               'params',{{'mu','G','alpha'}}, ...
                               'k_params',3);
        case 'sls'
            models{i} = struct('name','sls', ...
                               'params',{{'mu','G','lambda1'}}, ...
                               'k_params',3);
        otherwise
            error('Unknown model: %s', name);
    end
end
end

%% ===================== Parameter prior P(theta|M) ==========================
function logP = eval_param_prior_cont(modelName, theta, models, ...
    kernel_norms, axis_need, ranges)

% Find model metadata
model = [];
for i = 1:numel(models)
    if strcmpi(models{i}.name, modelName)
        model = models{i};
        break;
    end
end
if isempty(model)
    error('eval_param_prior_cont:unknownModel','Unknown model: %s',modelName);
end

theta = theta(:);  % column

% Ranges
mu_rng  = ranges.mu;
G_rng   = ranges.G;
lam_rng = ranges.lambda1;
a_rng   = ranges.alpha;
De_rng  = ranges.De;
tc_ref  = ranges.tc_ref;

normA = kernel_norms.normA;
normB = kernel_norms.normB;

% log-range map to [0,1]
log01 = @(x,xmin,xmax) (log10(x) - log10(xmin)) ./ ...
                       max(eps, (log10(xmax) - log10(xmin)));

% Axis needs
e_need  = axis_need.elastic;
m_need  = axis_need.maxwell;
nl_need = axis_need.nonlinear;

tiny = 1e-300;
logP  = log(tiny);   % default

switch lower(modelName)
    %----------------- 1-parameter models: log-uniform --------------------
    case 'newt'
        % theta = [mu]
        mu = theta(1);
        if mu < mu_rng(1) || mu > mu_rng(2)
            return;   % logP = log(tiny)
        end
        % original grid: mu grid is logspace, score = 1 -> density ∝ 1/mu
        logP = -log(mu);   % + constant (ignored)
        
    case 'nh'
        % theta = [G]
        G = theta(1);
        if G < G_rng(1) || G > G_rng(2)
            return;
        end
        % G grid logspace, score = 1 -> density ∝ 1/G
        logP = -log(G);
        
    %----------------- 2-parameter models -------------------------------
    case 'kv'
        % theta = [mu, G]
        mu = theta(1); G = theta(2);
        if mu < mu_rng(1) || mu > mu_rng(2) || ...
           G  < G_rng(1)  || G  > G_rng(2)
            return;
        end
        
        mu_t = log01(4*normA*mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
        G_t  = log01(  normB*G,   normB*G_rng(1),   normB*G_rng(2));
        
        H = hmean01(mu_t, e_need * G_t);     % anti-emulation bottleneck
        if H <= 0, return; end
        
        % log prior density ∝ H(mu_t,G_t) / (mu * G)
        logP = log(H) - log(mu) - log(G);
        
    case 'qnh'
        % theta = [G, alpha]
        G = theta(1); a = theta(2);
        if G < G_rng(1) || G > G_rng(2) || ...
           a < a_rng(1) || a > a_rng(2)
            return;
        end
        
        G_t = log01(normB*G, normB*G_rng(1), normB*G_rng(2));
        a_t = log01(a,       a_rng(1),       a_rng(2));
        
        H = hmean01(e_need * G_t, (e_need*nl_need) * a_t);
        if H <= 0, return; end
        
        % density ∝ H(G_t, a_t) / (G * a)
        logP = log(H) - log(G) - log(a);
        
    case 'lm'
        % theta = [mu, lambda1]
        mu  = theta(1);
        lam = theta(2);
        if mu  < mu_rng(1)  || mu  > mu_rng(2) || ...
           lam < lam_rng(1) || lam > lam_rng(2)
            return;
        end
        
        De_val = lam / tc_ref;
        if De_val < De_rng(1) || De_val > De_rng(2)
            return;
        end
        
        mu_t = log01(4*normA*mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
        De_t = log01(De_val,     De_rng(1),         De_rng(2));
        
        H = hmean01(mu_t, De_t);
        if H <= 0, return; end
        
        % density ∝ m_need * H(mu_t,De_t) / (mu * lambda1)
        logP = log(m_need) + log(H) - log(mu) - log(lam);
        
    %----------------- 3-parameter models -------------------------------
    case 'qkv'
        % theta = [mu, G, alpha]
        mu = theta(1); G = theta(2); a = theta(3);
        if mu < mu_rng(1) || mu > mu_rng(2) || ...
           G  < G_rng(1)  || G  > G_rng(2)  || ...
           a  < a_rng(1)  || a  > a_rng(2)
            return;
        end
        
        mu_t = log01(4*normA*mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
        G_t  = log01(  normB*G,   normB*G_rng(1),   normB*G_rng(2));
        a_t  = log01(a,           a_rng(1),         a_rng(2));
        
        H = hmean01(mu_t, e_need * G_t, (e_need*nl_need) * a_t);
        if H <= 0, return; end
        
        % density ∝ H / (mu * G * a)
        logP = log(H) - log(mu) - log(G) - log(a);
        
    case 'sls'
        % theta = [mu, G, lambda1]
        mu  = theta(1); G = theta(2); lam = theta(3);
        if mu  < mu_rng(1)  || mu  > mu_rng(2) || ...
           G   < G_rng(1)   || G   > G_rng(2)  || ...
           lam < lam_rng(1) || lam > lam_rng(2)
            return;
        end
        
        De_val = lam / tc_ref;
        if De_val < De_rng(1) || De_val > De_rng(2)
            return;
        end
        
        mu_t = log01(4*normA*mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
        G_t  = log01(  normB*G,   normB*G_rng(1),   normB*G_rng(2));
        De_t = log01(De_val,      De_rng(1),        De_rng(2));
        
        H = hmean01(mu_t, e_need * G_t, m_need * De_t);
        if H <= 0, return; end
        
        % density ∝ e_need * H / (mu * G * lambda1)
        logP = log(e_need) + log(H) - log(mu) - log(G) - log(lam);
        
    otherwise
        error('eval_param_prior_cont:unknownModel','Unknown model: %s',modelName);
end

% Clip extremely small
if ~isfinite(logP)
    logP = log(tiny);
end
end

function H = hmean01(varargin)
% Harmonic mean on [0,1] axes (as used in original code)
narginchk(1, inf);
X = cat(1, varargin{:});
X = max(0, min(1, X(:))).';  % row
n = numel(X);
H = n ./ sum(1 ./ (X + eps));
H = max(0, min(1, H));
end

%% ===================== Model prior P(M) (Eq. 28) ==========================
function logP = eval_model_prior_bic(~, k_params, N_eff)
% P(M) ∝ exp( -k_M/2 * log N_eff )
logP = -0.5 * k_params * log(max(N_eff,1));
end

%% ===================== Noise prior & quadrature ===========================
function [betaGrid, betaWeights, noisePriorFcn] = build_noise_prior(betaGrid_in)
betaGrid = betaGrid_in(:)';      % ensure row
B = numel(betaGrid);

% Half–Cauchy density on (0, ∞): P(β) = 2/π * 1/(1+β^2)
P_beta = (2/pi) * (1 ./ (1 + betaGrid.^2));

% Trapezoidal rule on a uniform (or arbitrary) grid
d = zeros(size(betaGrid));
if B == 1
    d(1) = 1;
else
    d(1)   = 0.5 * (betaGrid(2) - betaGrid(1));
    d(end) = 0.5 * (betaGrid(end) - betaGrid(end-1));
    for b = 2:B-1
        d(b) = 0.5 * (betaGrid(b+1) - betaGrid(b-1));
    end
end

weights_raw = P_beta .* d;
betaWeights = weights_raw / sum(weights_raw);

noisePriorFcn = @(beta) (2/pi) * (1 ./ (1 + beta.^2)) .* (beta > 0);
end
