function priors = build_model_priors(expData, opts)
% BUILD_MODEL_PRIORS  Hierarchical anti-emulation priors for IMR models
%
% Computes data-driven axis "needs" (elastic, maxwell, nonlinear) from
% experimental bubble dynamics and builds hierarchical priors that penalize
% model redundancy.
%
% Usage:
%   priors = build_model_priors(expData)
%   priors = build_model_priors(expData, opts)
%
% Inputs:
%   expData - struct from prepare_data() with fields:
%             .Rmatrix, .Rdotmatrix, .tmatrix, .tc, .Rmax_range, .Req_each
%             .p_inf, .rho (physical parameters)
%             Optional: .strain, .strainRate (computed if not provided)
%   opts    - optional settings:
%             .quiet     - suppress plots (default: true)
%             .n1, .n2, .n3 - grid sizes (defaults: 4096, 64, 16)
%
% Outputs:
%   priors - struct with fields:
%       .axis_need      - struct with .elastic, .maxwell, .nonlinear  [0,1]
%       .kernel_norms   - struct with .normA, .normB (data-driven scales)
%       .scales         - struct with .tc_ref (reference characteristic time)
%       .ranges         - parameter bounds (mu, G, lambda1, alpha, De)
%       .Newt, .NH, .KV, .qNH, .qKV, .LM, .SLS - per-model structs with:
%           .grid  - parameter grid arrays
%           .score - harmonic-mean bottleneck scores
%           .prior - normalized prior (sums to 1)
%
% Paper reference:
%   Dimensionless groups (Eq. 2-3):
%     Re = Á v_c R_max / ¼
%     Ca = p / G
%     De = » / t_c,  where t_c = R_max (Á/p)

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'quiet'), opts.quiet = true; end
if ~isfield(opts, 'n1'), opts.n1 = 4096; end  % 1D grid size
if ~isfield(opts, 'n2'), opts.n2 = 64;   end  % 2D grid size (64×64 = 4096)
if ~isfield(opts, 'n3'), opts.n3 = 16;   end  % 3D grid size (16×16×16 = 4096)

%% Extract experimental data
Rmatrix    = expData.Rmatrix;
Rdotmatrix = expData.Rdotmatrix;
tmatrix    = expData.tmatrix;
tc         = expData.tc;
Rmax_range = expData.Rmax_range;
Req_each   = expData.Req_each;

% Physical parameters
if isfield(expData, 'p_inf')
    p_inf = expData.p_inf;
else
    p_inf = 101325;  % Pa, default atmospheric pressure
    warning('expData.p_inf not found, using default 101325 Pa');
end

if isfield(expData, 'rho')
    rho = expData.rho;
else
    rho = 1064;  % kg/m^3, default for soft gels
    warning('expData.rho not found, using default 1064 kg/m^3');
end

% Use precomputed strain if available
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

% Reference characteristic time (median across trials)
tc_vec = tc(isfinite(tc) & tc > 0);
assert(~isempty(tc_vec), 'tc must be finite and positive for at least one trial');
tc_ref = median(tc_vec);

%% Compute data-driven kernel norms: A* (viscous), B (elastic), D (nonlinear)
A2 = 0; B2 = 0; D2perp = 0; used = 0;
Acell = {}; Tcell = {}; Wcell = {}; Bcell = {}; Lamcell = {};

for j = 1:J
    r   = Rmatrix(:,j);
    dr  = Rdotmatrix(:,j);
    tau = tmatrix(:,j);
    
    % Filter out bad data first
    good = isfinite(r) & isfinite(dr) & (r > 0);
    r = r(good); dr = dr(good); tau = tau(good);
    if numel(r) < 8, continue; end
    
    % Stretch ratio » = (Rmax/Req) × r*
    q   = Rmax_range(j) / max(Req_each(j), 1e-12);
    lam = q * r;
    
    % Strain / strain-rate (compute if not provided)
    if exist('strain','var') && size(strain,2) >= j && size(strain,1) >= sum(good)
        eps_star = strain(good, j);
    else
        eps_star = 0.5 * log(max(lam.^(-4), 1e-12));
    end
    if exist('strainRate','var') && size(strainRate,2) >= j && size(strainRate,1) >= sum(good)
        epsdot_star = strainRate(good, j);
    else
        epsdot_star = -2 * (dr ./ max(r, 1e-12));
    end
    
    % High-information elliptical gate in (eps*, epsdot*)
    eps_th    = 0.10 * max(abs(eps_star));
    epsdot_th = 1e5  * max(tc(j), eps);
    gate = ((eps_star ./ max(eps_th,eps)).^2 + (epsdot_star ./ max(epsdot_th,eps)).^2) >= 1;
    if ~any(gate) || numel(tau)==1, continue; end
    
    r = r(gate); dr = dr(gate); tau = tau(gate); lam = lam(gate);
    
    % Probability weights (non-uniform over gated Ä)
    dtau = [diff(tau); max(tau(end)-tau(end-1), eps)];
    w = dtau; w = w / max(sum(w), eps);
    
    % PAPER TABLE 1: Stress integral kernels
    % Newt: S*_v = -(4 X*)/(Re R*)
    % NH:   S*_NH = (1/2Ca)[4»^(-1) + »^(-4) - 5]
    
    Astar = dr ./ max(r, 1e-12);                         % Viscous kernel A* = X*/R*
    Bnh   = 0.5 * (4./max(lam,1e-12) + lam.^(-4) - 5);  % NH elastic kernel (unnormalized)
    
    % PAPER TABLE 1: qNH nonlinearity bracket
    bracket = 27/40 + (1/8).*lam.^(-8) + (1/5).*lam.^(-5) + lam.^(-2) - 2.*lam;
    
    % Nonlinearity direction: D* = S*_qNH(±=1) - S*_NH
    % D* = -3 S*_NH + (2/Ca) × bracket = (1/Ca)[-3 B_NH + 2 × bracket]
    Dstar = -3.*Bnh + 2.*bracket;  % Direction introduced by ±
    
    % Remove component explainable by span{A*, Bnh}
    W  = sqrt(w);
    Xw = [W.*Astar, W.*Bnh];
    yw = W.*Dstar;
    
    XtX = Xw.'*Xw;
    Xty = Xw.'*yw;
    coef = [0; 0];
    if all(isfinite(XtX(:))) && all(isfinite(Xty(:)))
        coef = pinv(XtX) * Xty;
    end
    
    Dproj = coef(1).*Astar + coef(2).*Bnh;
    Dperp = Dstar - Dproj;
    
    % Accumulate weighted L2 norms
    A2 = A2 + sum((Astar.^2) .* w);
    B2 = B2 + sum((Bnh.^2)   .* w);
    D2perp = D2perp + sum((Dperp.^2) .* w);
    used = used + 1;
    
    % Stash for later computations
    Acell{end+1} = Astar; %#ok<AGROW>
    Tcell{end+1} = tau;   %#ok<AGROW>
    Wcell{end+1} = w;     %#ok<AGROW>
    Bcell{end+1} = Bnh;   %#ok<AGROW>
    Lamcell{end+1} = lam; %#ok<AGROW>
end

assert(used > 0, 'No valid trials after gating');

normA = sqrt(A2 / used);  % ||A*||_w (data-driven)
normB = sqrt(B2 / used);  % ||B||_w (data-driven)

% Estimate typical shear modulus from elastic kernel norm
% From S*_NH = B_NH / Ca = B_NH × G / p
% The stress scale is p, so G ~ p / ||B_NH||
G_typical = p_inf / max(normB, 1e-6);  % Estimated shear modulus (Pa)

% Cauchy number: Ca = p / G (Paper Eq. 3)
Ca_typical = p_inf / G_typical;  % Should be O(1) for soft materials

if ~opts.quiet
    fprintf('  Data-driven estimates:\n');
    fprintf('    ||A*|| = %.3e\n', normA);
    fprintf('    ||B||  = %.3e\n', normB);
    fprintf('    G_est  = %.3e Pa\n', G_typical);
    fprintf('    Ca     = %.3e (p/G)\n', Ca_typical);
end

% Maxwell kernel norm: ||M*(De)||_w
maxwell_norm = @(De) maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell);

%% Parameter ranges (Table 2 in paper)
mu_rng  = [1e-4, 1e0];
G_rng   = [1e2,  1e5];
lam_rng = [1e-7, 1e-3];  % » (seconds)
a_rng   = [1e-3, 1e1];   % ±

% Log-range normalization helper
log01 = @(x, xmin, xmax) (log10(x) - log10(xmin)) ./ max(eps, log10(xmax) - log10(xmin));

%% Build parameter grids (4096 points per model)
n1 = opts.n1;
n2 = opts.n2;
n3 = opts.n3;

grid.Newt.mu = logspace(log10(mu_rng(1)), log10(mu_rng(2)), n1);
grid.NH.G    = logspace(log10(G_rng(1)),  log10(G_rng(2)),  n1);

[grid.KV.mu, grid.KV.G] = ndgrid( ...
    logspace(log10(mu_rng(1)), log10(mu_rng(2)), n2), ...
    logspace(log10(G_rng(1)),  log10(G_rng(2)),  n2));

[grid.qNH.G, grid.qNH.alpha] = ndgrid( ...
    logspace(log10(G_rng(1)), log10(G_rng(2)), n2), ...
    logspace(log10(a_rng(1)), log10(a_rng(2)), n2));

[grid.qKV.mu, grid.qKV.G, grid.qKV.alpha] = ndgrid( ...
    logspace(log10(mu_rng(1)), log10(mu_rng(2)), n3), ...
    logspace(log10(G_rng(1)),  log10(G_rng(2)),  n3), ...
    logspace(log10(a_rng(1)),  log10(a_rng(2)),  n3));

[grid.LM.mu, grid.LM.lambda1] = ndgrid( ...
    logspace(log10(mu_rng(1)),  log10(mu_rng(2)),  n2), ...
    logspace(log10(lam_rng(1)), log10(lam_rng(2)), n2));

[grid.SLS.mu, grid.SLS.G, grid.SLS.lambda1] = ndgrid( ...
    logspace(log10(mu_rng(1)), log10(mu_rng(2)), n3), ...
    logspace(log10(G_rng(1)),  log10(G_rng(2)),  n3), ...
    logspace(log10(lam_rng(1)),log10(lam_rng(2)),n3));

%% Range-invariant axes (stress-weighted)
% Stress weighting: Viscous ~ 4||A*||¼, Elastic ~ ||B||G
mu_t1d = log01(4*normA*grid.Newt.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t1d  = log01(normB*grid.NH.G,      normB*G_rng(1),    normB*G_rng(2));

mu_t_KV = log01(4*normA*grid.KV.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t_KV  = log01(normB*grid.KV.G,    normB*G_rng(1),    normB*G_rng(2));

G_t_qNH = log01(normB*grid.qNH.G,     normB*G_rng(1),  normB*G_rng(2));
a_t_qNH = log01(grid.qNH.alpha,       a_rng(1),        a_rng(2));

mu_t_qKV = log01(4*normA*grid.qKV.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t_qKV  = log01(normB*grid.qKV.G,    normB*G_rng(1),    normB*G_rng(2));
a_t_qKV  = log01(grid.qKV.alpha,      a_rng(1),          a_rng(2));

mu_t_LM = log01(4*normA*grid.LM.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));

% Deborah number axis: De = » / t_c
De_grid_LM  = grid.LM.lambda1 / tc_ref;
De_grid_SLS = grid.SLS.lambda1 / tc_ref;
De_rng      = [lam_rng(1)/tc_ref, lam_rng(2)/tc_ref];

De_t_LM  = log01(De_grid_LM,  De_rng(1), De_rng(2));

mu_t_SLS = log01(4*normA*grid.SLS.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t_SLS  = log01(normB*grid.SLS.G,    normB*G_rng(1),    normB*G_rng(2));
De_t_SLS = log01(De_grid_SLS,         De_rng(1),         De_rng(2));

%% Compute axis "needs" (data-driven)
H = @(varargin) hmean01(varargin{:});  % Harmonic mean bottleneck

% Elastic need: residual of B after projecting onto span{A*, M*(De)}
De_samp  = logspace(log10(De_rng(1)), log10(De_rng(2)), 64);
res2_sum = 0;
B2_sum   = 0;
used_em  = 0;

for j = 1:numel(Acell)
    A = Acell{j};
    t = Tcell{j};
    w = Wcell{j};
    B = Bcell{j};
    
    if numel(A) < 2 || numel(B) ~= numel(A), continue; end
    
    W = sqrt(w(:));
    Acol = W .* A(:);
    y    = W .* B(:);
    
    best_r2 = Inf;
    for d = 1:numel(De_samp)
        De = De_samp(d);
        
        % Maxwell filter: De·dS/dÄ + S = A*
        M = zeros(size(A));
        for ii = 2:numel(A)
            dt = t(ii) - t(ii-1);
            M(ii) = (M(ii-1) + (dt/De)*A(ii)) / (1 + dt/De);
        end
        
        X = [Acol, W.*M(:)];
        coef = pinv(X.'*X) * (X.'*y);
        r  = y - X*coef;
        r2 = sum(r.^2);
        
        if r2 < best_r2
            best_r2 = r2;
        end
    end
    
    if isfinite(best_r2)
        res2_sum = res2_sum + best_r2;
        B2_sum   = B2_sum + sum((W.*B(:)).^2);
        used_em  = used_em + 1;
    end
end

if used_em > 0 && B2_sum > 0
    e_need = max(0, min(1, res2_sum / B2_sum));
else
    e_need = 0;
end
e_need = e_need^2;  % Sharpen: small stays small

% Maxwell need: max||M*|| relative to ||A*||
Mnorms = arrayfun(maxwell_norm, De_samp);
Mstar  = max(Mnorms);
m_need = (Mstar^2) / max(normA^2 + Mstar^2, eps);

% Nonlinearity need: energy in D_perp
normDperp = sqrt(D2perp / used);
nl_need   = (normDperp^2) / max(normA^2 + normB^2 + normDperp^2, eps);
nl_need   = max(0, min(1, nl_need));

%% Harmonic-mean bottleneck scores
% Base 1-parameter models: uniform
score.Newt = ones(size(grid.Newt.mu));
score.NH   = ones(size(grid.NH.G));

% Composite models: penalize if data doesn't need extra axes
score.KV  = H(mu_t_KV, e_need * G_t_KV);
score.qNH = H(e_need * G_t_qNH, (e_need * nl_need) * a_t_qNH);
score.qKV = H(mu_t_qKV, e_need * G_t_qKV, (e_need * nl_need) * a_t_qKV);
score.LM  = m_need * H(mu_t_LM, De_t_LM);
score.SLS = e_need * H(mu_t_SLS, e_need * G_t_SLS, m_need * De_t_SLS);

%% Normalize to proper priors (sum = 1)
priors = struct();
priors.kernel_norms = struct('normA', normA, 'normB', normB);
priors.scales       = struct('tc_ref', tc_ref);
priors.ranges       = struct('mu', mu_rng, 'G', G_rng, 'lambda1', lam_rng, ...
                              'alpha', a_rng, 'De', De_rng);
priors.axis_need    = struct('elastic', e_need, 'maxwell', m_need, ...
                              'nonlinear', nl_need);

models = fieldnames(score);
for k = 1:numel(models)
    M = models{k};
    s = score.(M);
    s(~isfinite(s)) = 0;
    s = max(0, s);
    p = s / max(eps, sum(s(:)));  % Normalize
    
    priors.(M).grid  = grid.(M);
    priors.(M).score = s;
    priors.(M).prior = p;
end

%% Optional diagnostic plot
if ~opts.quiet && ~isempty(Acell)
    figure('Color','w');
    loglog(De_samp, Mnorms, 'b-', 'LineWidth', 1.5);
    xlabel('De = \lambda_1 / t_c', 'Interpreter', 'latex');
    ylabel('$||M^*||_w$', 'Interpreter', 'latex');
    title('Maxwell kernel L^2 norm (aggregated over trials)', 'Interpreter', 'latex');
end

end

%% ==================== Helper Functions ====================

function nm = maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell)
% Solve De·dS/dÄ + S = A* for each trial, return weighted L2 norm
S2 = 0;
used = 0;

for k = 1:numel(Acell)
    A = Acell{k};
    t = Tcell{k};
    w = Wcell{k};
    
    if numel(A) < 2, continue; end
    
    S = zeros(size(A));
    for i = 2:numel(A)
        dt = t(i) - t(i-1);
        S(i) = (S(i-1) + (dt/De)*A(i)) / (1 + dt/De);  % Implicit Euler
    end
    
    S2 = S2 + sum((S.^2) .* w);
    used = used + 1;
end

nm = sqrt(S2 / max(used, 1));
end

function H = hmean01(varargin)
% Harmonic mean of inputs in [0,1] (bottleneck behavior)
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