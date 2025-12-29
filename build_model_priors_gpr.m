function priors = build_model_priors_gpr(expData, opts)
% BUILD_MODEL_PRIORS_GPR  Build priors with redundancy GPR for continuous evaluation
%
% This version extends build_model_priors to:
%   1. Use data from prepare_data (not expDataPrep)
%   2. Pre-compute redundancy weights on dense Sobol grid
%   3. Train auxiliary GPR for log(w_red) per model
%   4. Store continuous prior evaluator functions
%
% Inputs:
%   expData - Struct from prepare_data (provides all experimental data)
%   opts    - Options struct:
%             .quiet                  - Suppress output (default: false)
%             .precompute_redundancy  - Build redundancy GPRs (default: true)
%             .N_redundancy_samples   - Samples for redundancy (default: 8192)
%
% Output:
%   priors  - Struct with fields matching paper exactly

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'quiet'),                 opts.quiet = false;                end
if ~isfield(opts, 'precompute_redundancy'), opts.precompute_redundancy = true; end
if ~isfield(opts, 'N_redundancy_samples'),  opts.N_redundancy_samples = 8192;  end

if ~opts.quiet
    fprintf('Building priors from experimental data...\n');
end

%% Extract data from expData (instead of calling expDataPrep)
Rmatrix     = expData.Rmatrix;
Rdotmatrix  = expData.Rdotmatrix;
tmatrix     = expData.tmatrix;
Rmax_range  = expData.Rmax_range;
Req_each    = expData.Req_each;
tc          = expData.tc;

% Use precomputed strain quantities if available
if isfield(expData, 'strain')
    strain = expData.strain;
else
    strain = [];
end
if isfield(expData, 'strainRate')
    strainRate = expData.strainRate;
else
    strainRate = [];
end

[~, J] = size(Rmatrix);

% Robust characteristic time for De (paper: De = lambda1 / t_c)
tc_vec = tc(isfinite(tc) & tc>0);
assert(~isempty(tc_vec),'tc must be finite and positive for at least one trial.');
tc_ref = median(tc_vec);  % single representative t_c for prior axes

%% ---------------- Compute A*, B from data ----------------
A2 = 0; B2 = 0; used = 0;
Acell = {}; Tcell = {}; Wcell = {}; Bcell = {};

for j = 1:J
    r   = Rmatrix(:,j);          % ND radius r*
    dr  = Rdotmatrix(:,j);       % ND velocity dr*/dÄ
    tau = tmatrix(:,j);          % ND time Ä

    good = isfinite(r) & isfinite(dr) & (r>0);
    r = r(good); dr = dr(good); tau = tau(good);
    if numel(r) < 8, continue; end

    % Stretch lambda = (Rmax/Req) * r*
    q   = Rmax_range(j) / max(Req_each(j), 1e-12);
    lam = q * r;

    % Strain / strain-rate (use precomputed if available)
    if ~isempty(strain) && size(strain,2) >= j
        eps_star = strain(good, j);
    else
        eps_star = 0.5 * log(max(lam.^(-4), 1e-12));
    end
    if ~isempty(strainRate) && size(strainRate,2) >= j
        epsdot_star = strainRate(good, j);
    else
        epsdot_star = -2 * (dr ./ max(r, 1e-12));
    end

    % High-information elliptical gate in (eps*, epsdot*)
    eps_th    = 0.10 * max(abs(eps_star));
    epsdot_th = 1e5  * max(tc(j), eps);
    gate = ((eps_star ./ max(eps_th,eps)).^2 + (epsdot_star ./ max(epsdot_th,eps)).^2) >= 1;
    if ~any(gate) || numel(tau)==1, continue; end

    % Probability weights over gated Ä (non-uniform)
    dtau = [diff(tau); max(tau(end)-tau(end-1), eps)];
    w = dtau; w(~gate)=0;
    s = sum(w); if s<=0, continue; end
    w = w / s;

    % Dimensionless kernels from data
    Astar = dr ./ max(r, 1e-12);                           % viscous kernel A*
    Bnh   = 0.5 * (4 ./ max(lam,1e-12) + lam.^(-4) - 5);   % NH elastic kernel B(lambda)

    % Accumulate gated, weighted L2 norms
    A2 = A2 + sum( (Astar(gate).^2) .* w(gate) );
    B2 = B2 + sum( (Bnh(gate  ).^2) .* w(gate) );
    used = used + 1;
    
    % ---------- Data-driven "nonlinearity" direction (qNH - NH) ----------
    lam_g = lam(gate);
    bracket = 27/40 + (1/8).*lam_g.^(-8) + (1/5).*lam_g.^(-5) + lam_g.^(-2) - 2.*lam_g;
    Dstar = -3.*Bnh(gate) + 2.*bracket;
    
    % Remove component explainable by span{A*, Bnh} (weighted LS)
    X1 = Astar(gate);  X2 = Bnh(gate);
    W  = sqrt(w(gate));
    Xw = [W.*X1, W.*X2];
    yw = W.*Dstar;
    
    % Solve 2x2 normal equations safely
    XtX = Xw.'*Xw;
    Xty = Xw.'*yw;
    coef = [0;0];
    if all(isfinite(XtX(:))) && all(isfinite(Xty(:)))
        coef = pinv(XtX) * Xty;
    end
    Dproj = coef(1).*X1 + coef(2).*X2;
    Dperp = Dstar - Dproj;
    
    % Accumulate weighted ||Dperp||^2 for nonlinearity need
    if ~exist('D2perp','var'), D2perp = 0; end
    D2perp = D2perp + sum( (Dperp.^2) .* w(gate) );

    % Stash for Maxwell solve on actual (nonuniform) Ä
    Acell{end+1} = Astar(gate); %#ok<AGROW>
    Tcell{end+1} = tau(gate);   %#ok<AGROW>
    Wcell{end+1} = w(gate);     %#ok<AGROW>
    Bcell{end+1} = Bnh(gate);   %#ok<AGROW>
end
assert(used>0,'No valid trials after gating.');

normA = sqrt(A2/used);   % ||A*||_w (data-driven)
normB = sqrt(B2/used);   % ||B||_w (data-driven)

if ~opts.quiet
    fprintf('  Kernel norms: ||A*|| = %.4e, ||B|| = %.4e\n', normA, normB);
end

% Maxwell kernel norm handle
maxwell_norm = @(De) maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell);

%% ---------------- Parameter ranges (from paper Table 2) ----------------
mu_rng  = [1e-4, 1e0];
G_rng   = [1e2 , 1e5];
lam_rng = [1e-7, 1e-3];
a_rng   = [1e-3, 1e1];

% Helper: log-range map to [0,1]
log01 = @(x,xmin,xmax) (log10(x) - log10(xmin)) ./ max(eps, (log10(xmax) - log10(xmin)));

% Grid sizes (4096 per model)
n1 = 4096;   % 1D
n2 = 64;     % 2D -> 4096
n3 = 16;     % 3D -> 4096

%% ---------------- Grids per model ----------------
grid.Newt.mu = logspace(log10(mu_rng(1)),  log10(mu_rng(2)),  n1);
grid.NH.G    = logspace(log10(G_rng(1)),   log10(G_rng(2)),   n1);

[grid.KV.mu, grid.KV.G] = ndgrid( ...
    logspace(log10(mu_rng(1)), log10(mu_rng(2)), n2), ...
    logspace(log10(G_rng(1)),  log10(G_rng(2)),  n2));

[grid.qNH.G, grid.qNH.alpha] = ndgrid( ...
    logspace(log10(G_rng(1)),  log10(G_rng(2)),  n2), ...
    logspace(log10(a_rng(1)),  log10(a_rng(2)),  n2));

[grid.qKV.mu, grid.qKV.G, grid.qKV.alpha] = ndgrid( ...
    logspace(log10(mu_rng(1)), log10(mu_rng(2)), n3), ...
    logspace(log10(G_rng(1)),  log10(G_rng(2)),  n3), ...
    logspace(log10(a_rng(1)),  log10(a_rng(2)),  n3));

[grid.LM.mu, grid.LM.lambda1] = ndgrid( ...
    logspace(log10(mu_rng(1)),  log10(mu_rng(2)),  n2), ...
    logspace(log10(lam_rng(1)), log10(lam_rng(2)), n2));

[grid.SLS.mu, grid.SLS.G, grid.SLS.lambda1] = ndgrid( ...
    logspace(log10(mu_rng(1)),  log10(mu_rng(2)),  n3), ...
    logspace(log10(G_rng(1)),   log10(G_rng(2)),   n3), ...
    logspace(log10(lam_rng(1)), log10(lam_rng(2)), n3));

%% ---------------- Range-invariant axes (stress-weighted) ----------------
mu_t1d = log01(4*normA*grid.Newt.mu,  4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t1d  = log01(  normB*grid.NH.G,       normB*G_rng(1),    normB*G_rng(2));

mu_t_KV = log01(4*normA*grid.KV.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t_KV  = log01(  normB*grid.KV.G,    normB*G_rng(1),    normB*G_rng(2));

G_t_qNH = log01(  normB*grid.qNH.G,   normB*G_rng(1),    normB*G_rng(2));
a_t_qNH = log01(        grid.qNH.alpha,      a_rng(1),          a_rng(2));

mu_t_qKV = log01(4*normA*grid.qKV.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t_qKV  = log01(  normB*grid.qKV.G,   normB*G_rng(1),    normB*G_rng(2));
a_t_qKV  = log01(        grid.qKV.alpha,      a_rng(1),          a_rng(2));

mu_t_LM  = log01(4*normA*grid.LM.mu,  4*normA*mu_rng(1),  4*normA*mu_rng(2));

% PAPER-CONSISTENT De axis: De = lambda1 / t_c
De_grid_LM  = grid.LM.lambda1  / tc_ref;
De_grid_SLS = grid.SLS.lambda1 / tc_ref;
De_rng      = [lam_rng(1)/tc_ref, lam_rng(2)/tc_ref];

De_t_LM  = log01(De_grid_LM,  De_rng(1), De_rng(2));

mu_t_SLS = log01(4*normA*grid.SLS.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t_SLS  = log01(  normB*grid.SLS.G,   normB*G_rng(1),    normB*G_rng(2));
De_t_SLS = log01(De_grid_SLS, De_rng(1), De_rng(2));

%% ---------------- Compute axis "needs" ----------------
H = @(varargin) hmean01(varargin{:});

% Elastic need via residual of B relative to span{A*, M*(De)}
De_samp  = logspace(log10(De_rng(1)), log10(De_rng(2)), 64);
res2_sum = 0; B2_sum = 0; used_em = 0;

for j = 1:numel(Acell)
    A = Acell{j}; t = Tcell{j}; w = Wcell{j}; B = Bcell{j};
    if numel(A) < 2 || numel(B) ~= numel(A), continue; end
    W = sqrt(w(:));
    Acol = W .* A(:);
    y    = W .* B(:);

    best_r2 = Inf;
    for d = 1:numel(De_samp)
        De = De_samp(d);
        M = zeros(size(A));
        for i = 2:numel(A)
            dt = t(i) - t(i-1);
            M(i) = ( M(i-1) + (dt/De)*A(i) ) / (1 + dt/De);
        end
        X = [Acol, W.*M(:)];
        coef = pinv(X.'*X) * (X.'*y);
        r    = y - X*coef;
        r2   = sum(r.^2);
        if r2 < best_r2, best_r2 = r2; end
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
e_need = e_need.^2;

% Maxwell need
De_samp = logspace(log10(De_rng(1)), log10(De_rng(2)), 64);
Mnorms  = arrayfun(maxwell_norm, De_samp);
Mstar   = max(Mnorms);
m_need  = (Mstar^2) / max(normA^2 + Mstar^2, eps);

% Nonlinearity need
if ~exist('D2perp','var') || used==0
    nl_need = 0;
else
    normDperp = sqrt(D2perp / used);
    nl_need   = (normDperp^2) / max(normA^2 + normB^2 + normDperp^2, eps);
end
nl_need = max(0, min(1, nl_need));

if ~opts.quiet
    fprintf('  Axis needs: elastic=%.3f, maxwell=%.3f, nonlinear=%.3f\n', ...
            e_need, m_need, nl_need);
end

%% ---------------- Harmonic-mean scores ----------------
score.Newt = ones(size(grid.Newt.mu));
score.NH   = ones(size(grid.NH.G));
score.KV   = H(        mu_t_KV,     e_need * G_t_KV);
score.qNH  = H(e_need * G_t_qNH,    (e_need * nl_need) * a_t_qNH);
score.qKV  = H(        mu_t_qKV,    e_need * G_t_qKV,   (e_need * nl_need) * a_t_qKV);
score.LM   = (m_need) * H(mu_t_LM, De_t_LM);
score.SLS  = (e_need) * H(mu_t_SLS, e_need * G_t_SLS, m_need * De_t_SLS);

%% ---------------- Normalize to priors ----------------
priors = struct();
priors.kernel_norms = struct('normA',normA,'normB',normB);
priors.scales       = struct('tc_ref',tc_ref);
priors.ranges       = struct('mu',mu_rng,'G',G_rng,'lambda1',lam_rng,'alpha',a_rng,'De',De_rng);
priors.axis_need    = struct('elastic', e_need, 'maxwell', m_need, 'nonlinear', nl_need);

models = fieldnames(score);
for k = 1:numel(models)
    M = models{k};
    s = score.(M);
    s(~isfinite(s)) = 0; s = max(0, s);
    p = s / max(eps, sum(s(:)));
    priors.(M).grid  = grid.(M);
    priors.(M).score = s;
    priors.(M).prior = p;
end

%% ---------------- Add continuous prior functions ----------------
if ~opts.quiet
    fprintf('Creating continuous prior evaluators...\n');
end

for k = 1:numel(models)
    M = models{k};
    priors.(M).prior_fn = @(theta) evaluate_harmonic_prior(theta, M, priors);
end

%% ---------------- Pre-compute redundancy GPRs (Option A) ----------------
if opts.precompute_redundancy
    if ~opts.quiet
        fprintf('\nPre-computing redundancy GPRs (this may take 5-10 min)...\n');
    end
    
    % Generate representative meanframe
    meanframe = generate_meanframe(expData);
    
    for k = 1:numel(models)
        M = models{k};
        
        tic;
        if ~opts.quiet
            fprintf('  [%d/%d] %s...', k, numel(models), M);
        end
        
        % Train redundancy GPR
        wred_gpr = train_redundancy_gpr(M, priors, meanframe, ...
                                        opts.N_redundancy_samples, expData);
        priors.(M).wred_gpr = wred_gpr;
        
        % Update prior function to include redundancy
        priors.(M).prior_fn = @(theta) evaluate_full_prior(theta, M, priors, expData);
        
        if ~opts.quiet
            fprintf(' done (%.1f s)\n', toc);
        end
    end
    
    if ~opts.quiet
        fprintf('\n Redundancy GPRs trained for all models\n\n');
    end
end

end

%% ==================== Prior Evaluation Functions ====================

function logP = evaluate_harmonic_prior(theta, modelKey, priors)
% Harmonic-mean prior only (no redundancy)

if ~isfield(priors, modelKey)
    logP = -inf;
    return;
end

ranges = priors.ranges;

% Normalize theta to [0,1]
[theta_norm, ~] = normalize_theta(theta, modelKey, ranges, priors);

% Harmonic mean (Eq. 26 in paper)
d = numel(theta_norm);
H = d / sum(1 ./ (theta_norm + eps));

logP = log(H + eps);

end

function logP = evaluate_full_prior(theta, modelKey, priors, expData)
% Full prior: harmonic-mean + redundancy

% 1. Harmonic-mean component
logP_harm = evaluate_harmonic_prior(theta, modelKey, priors);

% 2. Redundancy component (via GPR or constant)
if isfield(priors.(modelKey), 'wred_gpr')
    gpr_obj = priors.(modelKey).wred_gpr;
    
    % Check if it's a dummy GPR
    if isstruct(gpr_obj) && isfield(gpr_obj, 'is_dummy')
        if isfield(gpr_obj, 'constant_value')
            log_wred = gpr_obj.constant_value;
        else
            log_wred = 0;  % log(1) = 0
        end
    else
        % Real GPR - predict
        log_wred = predict(gpr_obj, theta);
    end
else
    log_wred = 0;
end

logP = logP_harm + log_wred;

end

%% ==================== Helper Functions (same as before) ====================

function [theta_norm, axis_names] = normalize_theta(theta, modelKey, ranges, priors)

axis_names = get_axis_names(modelKey);
d = numel(axis_names);
theta_norm = zeros(1, d);

normA = priors.kernel_norms.normA;
normB = priors.kernel_norms.normB;

for i = 1:d
    param = axis_names{i};
    val = theta(i);
    
    switch lower(param)
        case 'mu'
            stress_val = 4 * normA * val;
            range_lo   = 4 * normA * ranges.mu(1);
            range_hi   = 4 * normA * ranges.mu(2);
            
        case 'g'
            stress_val = normB * val;
            range_lo   = normB * ranges.G(1);
            range_hi   = normB * ranges.G(2);
            
        case 'lambda1'
            tc_ref = priors.scales.tc_ref;
            stress_val = val / tc_ref;
            range_lo   = ranges.lambda1(1) / tc_ref;
            range_hi   = ranges.lambda1(2) / tc_ref;
            
        case 'alpha'
            stress_val = val;
            range_lo   = ranges.alpha(1);
            range_hi   = ranges.alpha(2);
            
        otherwise
            error('Unknown parameter: %s', param);
    end
    
    theta_norm(i) = (log10(stress_val) - log10(range_lo)) / ...
                    max(log10(range_hi) - log10(range_lo), eps);
    theta_norm(i) = max(0, min(1, theta_norm(i)));
end

end

function names = get_axis_names(modelKey)

switch lower(modelKey)
    case 'newt',        names = {'mu'};
    case 'nh',          names = {'g'};
    case 'kv',          names = {'mu', 'g'};
    case 'qnh',         names = {'g', 'alpha'};
    case {'lm','max'},  names = {'mu', 'lambda1'};
    case 'qkv',         names = {'mu', 'g', 'alpha'};
    case 'sls',         names = {'mu', 'g', 'lambda1'};
    otherwise,          names = {};
end

end

function meanframe = generate_meanframe(expData)

Rmatrix = expData.Rmatrix;
tmatrix = expData.tmatrix;

R_median = median(Rmatrix, 2, 'omitnan');
t_median = median(tmatrix, 2, 'omitnan');

Rdot_median = gradient(R_median, t_median);

meanframe = struct();
meanframe.t    = t_median;
meanframe.R    = R_median;
meanframe.Rdot = Rdot_median;
meanframe.tc   = expData.tc_mean;

end

function wred_gpr = train_redundancy_gpr(modelKey, priors, meanframe, N_samples, expData)

ranges = priors.ranges;
axis_names = get_axis_names(modelKey);
d = numel(axis_names);

% Check if model has subsets (needs redundancy prior)
subsets = get_model_subsets(modelKey);
if isempty(subsets)
    % No redundancy prior needed (base models)
    % Return dummy GPR that always returns log(1) = 0
    wred_gpr = struct('is_dummy', true);
    return;
end

xmin = zeros(1, d);
xmax = zeros(1, d);
for i = 1:d
    switch lower(axis_names{i})
        case 'mu',      xmin(i) = ranges.mu(1);      xmax(i) = ranges.mu(2);
        case 'g',       xmin(i) = ranges.G(1);       xmax(i) = ranges.G(2);
        case 'lambda1', xmin(i) = ranges.lambda1(1); xmax(i) = ranges.lambda1(2);
        case 'alpha',   xmin(i) = ranges.alpha(1);   xmax(i) = ranges.alpha(2);
    end
end

sob = scramble(sobolset(d), 'MatousekAffineOwen');
U = net(sob, N_samples);

theta_samples = zeros(N_samples, d);
for j = 1:d
    theta_samples(:,j) = 10.^(log10(xmin(j)) + U(:,j)*(log10(xmax(j))-log10(xmin(j))));
end

wred_samples = zeros(N_samples, 1);

for i = 1:N_samples
    wred_samples(i) = compute_redundancy_weight_continuous( ...
        modelKey, theta_samples(i,:), meanframe, priors, expData);
end

log_wred = log(max(wred_samples, 1e-12));

% Check for constant redundancy (happens for some models)
if std(log_wred) < 1e-6
    % Constant redundancy - return dummy GPR
    wred_gpr = struct('is_dummy', true, 'constant_value', mean(log_wred));
    return;
end

% Train GPR with safe noise floor
sigma_val = max(std(log_wred)*0.01, 1e-6);  % At least 1e-6

try
    wred_gpr = fitrgp(theta_samples, log_wred, ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true, ...
        'Sigma', sigma_val);
catch ME
    warning('GPR training failed for %s: %s. Using constant prior.', modelKey, ME.message);
    wred_gpr = struct('is_dummy', true, 'constant_value', mean(log_wred));
end

end

function w_red = compute_redundancy_weight_continuous(modelKey, theta, meanframe, priors, expData)

subsets = get_model_subsets(modelKey);

if isempty(subsets)
    w_red = 1;
    return;
end

Rm  = meanframe.R;
Rdm = meanframe.Rdot;
t   = meanframe.t;
tc  = meanframe.tc;

pars = theta_to_struct(theta, modelKey);
S_parent = compute_stress_integral(modelKey, pars, Rm, Rdm, t, tc);

if isempty(S_parent) || ~any(isfinite(S_parent))
    w_red = 1;
    return;
end

f_subset = ones(1, numel(subsets));

for s = 1:numel(subsets)
    subset = subsets{s};
    
    pars_sub = project_to_subset(modelKey, subset, pars);
    S_sub = compute_stress_integral(subset, pars_sub, Rm, Rdm, t, tc);
    
    if isempty(S_sub) || ~any(isfinite(S_sub))
        f_subset(s) = 1;
        continue;
    end
    
    norm_parent = norm(S_parent, 2);
    if norm_parent == 0
        f_subset(s) = 1;
        continue;
    end
    
    c_best = (S_sub' * S_parent) / max(sum(S_sub.^2), eps);
    residual = S_parent - c_best * S_sub;
    rel_err = norm(residual, 2) / norm_parent;
    
    tau_local = 0.01;
    f_subset(s) = rel_err^2 / (rel_err^2 + tau_local^2);
    f_subset(s) = max(1e-12, min(1, f_subset(s)));
end

w_red = min(f_subset);
w_red = max(1e-12, min(1, w_red));

N_eff = 2 * nnz(expData.mask);
w_red = w_red^N_eff;

end

function subsets = get_model_subsets(modelKey)

switch lower(modelKey)
    case 'qkv',      subsets = {'qnh', 'kv', 'nh'};
    case 'sls',      subsets = {'kv', 'lm', 'nh'};
    case 'kv',       subsets = {'nh', 'newt'};
    case 'qnh',      subsets = {'nh'};
    case {'lm','max'}, subsets = {'newt'};
    otherwise,       subsets = {};
end

end

function pars = theta_to_struct(theta, modelKey)

axis_names = get_axis_names(modelKey);
pars = struct();

for i = 1:numel(axis_names)
    pars.(axis_names{i}) = theta(i);
end

end

function pars_sub = project_to_subset(modelKey, subset, pars)

axis_names_sub = get_axis_names(subset);
pars_sub = struct();

for i = 1:numel(axis_names_sub)
    name = axis_names_sub{i};
    if isfield(pars, name)
        pars_sub.(name) = pars.(name);
    else
        pars_sub.(name) = 0;
    end
end

end

function S = compute_stress_integral(modelKey, pars, Rm, Rdm, t, tc)

dt = [diff(t); t(end)-t(end-1)];

req_nd = median(Rm(floor(0.8*numel(Rm)):end), 'omitnan');
if ~isfinite(req_nd) || req_nd <= 0, req_nd = 1; end
lambda = Rm / req_nd;

switch lower(modelKey)
    case 'newt'
        S = get_field(pars, 'mu', 0) * (Rdm ./ max(Rm, 1e-12));
        
    case 'nh'
        G = get_field(pars, 'g', 0);
        S = 0.5 * G * (4 ./ max(lambda, 1e-12) + lambda.^(-4) - 5);
        
    case 'kv'
        mu = get_field(pars, 'mu', 0);
        G  = get_field(pars, 'g', 0);
        S_v  = mu * (Rdm ./ max(Rm, 1e-12));
        S_NH = 0.5 * G * (4 ./ max(lambda, 1e-12) + lambda.^(-4) - 5);
        S = S_v + S_NH;
        
    case 'qnh'
        G = get_field(pars, 'g', 0);
        alpha = get_field(pars, 'alpha', 0);
        S_NH = 0.5 * G * (4 ./ max(lambda, 1e-12) + lambda.^(-4) - 5);
        bracket = 27/40 + (1/8)*lambda.^(-8) + (1/5)*lambda.^(-5) + lambda.^(-2) - 2*lambda;
        S = (1 - 3*alpha)*S_NH + 2*alpha*G*bracket;
        
    case {'lm', 'max'}
        mu = get_field(pars, 'mu', 0);
        lambda1 = get_field(pars, 'lambda1', 0);
        S_v = mu * (Rdm ./ max(Rm, 1e-12));
        
        S = zeros(size(S_v));
        De = lambda1 / tc;
        for k = 2:numel(S)
            a = dt(k) / De;
            S(k) = (S(k-1) + a*S_v(k)) / (1 + a);
        end
        
    case 'qkv'
        mu = get_field(pars, 'mu', 0);
        G  = get_field(pars, 'g', 0);
        alpha = get_field(pars, 'alpha', 0);
        
        S_v = mu * (Rdm ./ max(Rm, 1e-12));
        S_NH = 0.5 * G * (4 ./ max(lambda, 1e-12) + lambda.^(-4) - 5);
        bracket = 27/40 + (1/8)*lambda.^(-8) + (1/5)*lambda.^(-5) + lambda.^(-2) - 2*lambda;
        S = S_v + (1 - 3*alpha)*S_NH + 2*alpha*G*bracket;
        
    case 'sls'
        mu = get_field(pars, 'mu', 0);
        G  = get_field(pars, 'g', 0);
        lambda1 = get_field(pars, 'lambda1', 0);
        
        S_NH = 0.5 * G * (4 ./ max(lambda, 1e-12) + lambda.^(-4) - 5);
        S_v  = mu * (Rdm ./ max(Rm, 1e-12));
        
        S_M = zeros(size(S_v));
        De = lambda1 / tc;
        for k = 2:numel(S_M)
            a = dt(k) / De;
            S_M(k) = (S_M(k-1) + a*S_v(k)) / (1 + a);
        end
        
        S = S_NH + S_M;
        
    otherwise
        S = [];
end

S(~isfinite(S)) = 0;

end

function val = get_field(s, field, default)

if isfield(s, field)
    val = s.(field);
else
    val = default;
end

end

function nm = maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell)

S2 = 0; used = 0;
for k = 1:numel(Acell)
    A = Acell{k};  t = Tcell{k};  w = Wcell{k};
    if numel(A) < 2, continue; end
    S = zeros(size(A));
    for i = 2:numel(A)
        dt = t(i)-t(i-1);
        S(i) = ( S(i-1) + (dt/De)*A(i) ) / (1 + dt/De);
    end
    S2 = S2 + sum( (S.^2) .* w );
    used = used + 1;
end
nm = (used==0) * 0 + (used>0) * sqrt(S2/used);

end

function H = hmean01(varargin)

narginchk(1, inf);
base = varargin{1};
dim  = ndims(base) + 1;
X = max(0, min(1, base));
for i = 2:nargin
    X = cat(dim, X, max(0, min(1, varargin{i})));
end
n = size(X, dim);
H = n ./ sum(1 ./ (X + eps), dim);

end