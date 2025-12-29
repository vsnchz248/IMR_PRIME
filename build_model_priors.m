function priors = build_model_priors(opts)
% Hierarchical anti-emulation priors for Newt, NH, KV, qNH, qKV, LM, SLS.
% - Computes A*(tau), B(lambda), and solves Maxwell S*(tau;De) from your LIC data
% - Builds range-invariant axes (log-scaled, stress-weighted where appropriate)
% - Combines axes with a harmonic-mean "bottleneck" to penalize emulation of simpler models
% - Returns proper (sum=1) priors per model grid + kernel diagnostics
%
% Paper-consistent dimensionless groups:
%   Re = rho v_c R_max / mu
%   Ca = p_inf / G                (Cauchy number; NOT a capillary number)
%   De = lambda1 / t_c,  t_c = R_max * sqrt(rho/p_inf)
%
% Notes:
% - qNH/qKV: alpha is treated as its own distinguishing axis (tilde-alpha).
%   If you provide an explicit S*_qNH(alpha) kernel, we can replace that axis
%   with a stress-weighted kernel norm exactly like B and Maxwell.

if nargin<1, opts = struct(); end
set(0,'defaultTextInterpreter','latex'); clc;

%% ---------------- 0) Load data & compute A*, B; stash sequences for Maxwell ----------------
% expDataPrep should provide: Rmatrix, Rdotmatrix, tmatrix, Rmax_range, Req_each, tc, (optional) strain, strainRate
expDataPrep;
[~, J] = size(Rmatrix);

% Robust characteristic time for De (paper: De = lambda1 / t_c)
if ~(exist('tc','var') && ~isempty(tc))
    error('expDataPrep must provide tc to form De = lambda1/tc (paper definition).');
end
tc_vec = tc(isfinite(tc) & tc>0);
assert(~isempty(tc_vec),'tc must be finite and positive for at least one trial.');
tc_ref = median(tc_vec);  % single representative t_c for prior axes

A2 = 0; B2 = 0; used = 0;
Acell = {}; Tcell = {}; Wcell = {}; Bcell = {};

for j = 1:J
    r   = Rmatrix(:,j);          % ND radius r*
    dr  = Rdotmatrix(:,j);       % ND velocity dr*/dτ
    tau = tmatrix(:,j);          % ND time τ

    good = isfinite(r) & isfinite(dr) & (r>0);
    r = r(good); dr = dr(good); tau = tau(good);
    if numel(r) < 8, continue; end

    % Stretch lambda = (Rmax/Req) * r*
    q   = Rmax_range(j) / max(Req_each(j), 1e-12);
    lam = q * r;

    % Strain / strain-rate (use precomputed if available)
    if exist('strain','var') && size(strain,2) >= j
        eps_star = strain(good, j);
    else
        eps_star = 0.5 * log(max(lam.^(-4), 1e-12));
    end
    if exist('strainRate','var') && size(strainRate,2) >= j
        epsdot_star = strainRate(good, j);
    else
        epsdot_star = -2 * (dr ./ max(r, 1e-12));
    end

    % High-information elliptical gate in (eps*, epsdot*)
    eps_th    = 0.10 * max(abs(eps_star));
    epsdot_th = 1e5  * max(tc(j), eps);
    gate = ((eps_star ./ max(eps_th,eps)).^2 + (epsdot_star ./ max(epsdot_th,eps)).^2) >= 1;
    if ~any(gate) || numel(tau)==1, continue; end

    % Probability weights over gated τ (non-uniform)
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
    
    % ---------- New: data-driven "nonlinearity" direction (qNH - NH) ----------
    % qNH stress (no-G factor): S_qNH = (1-3α) Bnh + 2α * bracket(λ)
    % Direction introduced by α (up to α·G scale) is:
    %    D*(λ) = -3*Bnh + 2*bracket(λ)
    lam_g = lam(gate);
    bracket = 27/40 + (1/8).*lam_g.^(-8) + (1/5).*lam_g.^(-5) + lam_g.^(-2) - 2.*lam_g;
    Dstar = -3.*Bnh(gate) + 2.*bracket;
    
    % Remove any component of Dstar explainable by span{A*, Bnh} (weighted LS)
    % Build weighted design with columns [A*, Bnh]
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


    % Stash for Maxwell solve on actual (nonuniform) τ
    Acell{end+1} = Astar(gate); %#ok<AGROW>
    Tcell{end+1} = tau(gate);   %#ok<AGROW>
    Wcell{end+1} = w(gate);     %#ok<AGROW>
    Bcell{end+1} = Bnh(gate);   %#ok<AGROW>
end
assert(used>0,'No valid trials after gating.');

normA = sqrt(A2/used);   % ||A*||_w (data-driven)
normB = sqrt(B2/used);   % ||B||_w (data-driven)

% Maxwell kernel norm handle: input De (dimensionless), output ||M*||_w
maxwell_norm = @(De) maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell);

%% ---------------- 1) Parameter ranges (as in your tables) ----------------
mu_rng  = [1e-4, 1e0];
G_rng   = [1e2 , 1e5];
lam_rng = [1e-7, 1e-3];     % lambda1 (s); used only to define De-range via tc_ref
a_rng   = [1e-3, 1e1];      % alpha

% Helper: log-range map to [0,1] (range-invariant, scale-invariant)
log01 = @(x,xmin,xmax) (log10(x) - log10(xmin)) ./ max(eps, (log10(xmax) - log10(xmin)));

% Grid sizes (4096 per model)
n1 = 4096;   % 1D
n2 = 64;     % 2D -> 4096
n3 = 16;     % 3D -> 4096

%% ---------------- 2) Grids per model ----------------
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

%% ---------------- 3) Range-invariant axes (stress-weighted where appropriate) ----------------
% Stress weighting: Viscous ~ (4*||A*||)*mu ; NH elastic ~ (||B||)*G
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

% PAPER-CONSISTENT De axis: De = lambda1 / t_c (use tc_ref)
De_grid_LM  = grid.LM.lambda1  / tc_ref;
De_grid_SLS = grid.SLS.lambda1 / tc_ref;
De_rng      = [lam_rng(1)/tc_ref, lam_rng(2)/tc_ref];

De_t_LM  = log01(De_grid_LM,  De_rng(1), De_rng(2));

mu_t_SLS = log01(4*normA*grid.SLS.mu, 4*normA*mu_rng(1), 4*normA*mu_rng(2));
G_t_SLS  = log01(  normB*grid.SLS.G,   normB*G_rng(1),    normB*G_rng(2));
De_t_SLS = log01(De_grid_SLS, De_rng(1), De_rng(2));

%% ---------------- 4) Harmonic-mean (bottleneck) scores: data-aware anti-emulation -----
H = @(varargin) hmean01(varargin{:});   % HM on [0,1] axes (logical-AND behavior)

% ---- Data-derived axis "need" multipliers in [0,1] (no knobs) ----
% Elastic need: compares NH kernel energy to viscous
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
        % Maxwell-filtered A* on nonuniform τ: De * dS/dτ + S = A*
        M = zeros(size(A));
        for i = 2:numel(A)
            dt = t(i) - t(i-1);
            M(i) = ( M(i-1) + (dt/De)*A(i) ) / (1 + dt/De);
        end
        X = [Acol, W.*M(:)];
        % Weighted LS (2x2) with safe pinv
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
e_need = e_need.^2;   % sharpen: tiny stays tiny; mid-values attenuate

% Maxwell need: compare best Maxwell kernel energy to viscous
De_samp = logspace(log10(De_rng(1)), log10(De_rng(2)), 64);
Mnorms  = arrayfun(maxwell_norm, De_samp);
Mstar   = max(Mnorms);      % best case across De
m_need  = (Mstar^2) / max(normA^2 + Mstar^2, eps);

% Nonlinearity need: energy in D* that cannot be emulated by {A*, Bnh}
if ~exist('D2perp','var') || used==0
    nl_need = 0;
else
    normDperp = sqrt(D2perp / used);
    nl_need   = (normDperp^2) / max(normA^2 + normB^2 + normDperp^2, eps);
end
nl_need = max(0, min(1, nl_need));

priors.axis_need = struct('elastic', e_need, 'maxwell', m_need, 'nonlinear', nl_need);

% Base 1-parameter models: neutral (uniform) over their log-spaced axes
score.Newt = ones(size(grid.Newt.mu));
score.NH   = ones(size(grid.NH.G));

% Composite models:
% KV must need BOTH viscous and elastic; elastic axis scaled by e_need
score.KV  = H(        mu_t_KV,     e_need * G_t_KV);

% qNH needs elastic + nonlinearity
score.qNH = H(e_need * G_t_qNH,    (e_need * nl_need) * a_t_qNH);

% qKV needs viscous + elastic + nonlinearity
score.qKV = H(        mu_t_qKV,    e_need * G_t_qKV,   (e_need * nl_need) * a_t_qKV);

% Multiply the whole surface by the axis "need" (no knobs).
score.LM  = (m_need) * H(mu_t_LM, De_t_LM);
score.SLS = (e_need) * H(mu_t_SLS, e_need * G_t_SLS, m_need * De_t_SLS);

%% ---------------- 5) Normalize to priors and collect outputs ----------------
priors = struct();
priors.kernel_norms = struct('normA',normA,'normB',normB);
priors.scales       = struct('tc_ref',tc_ref);
priors.ranges       = struct('mu',mu_rng,'G',G_rng,'lambda1',lam_rng,'alpha',a_rng,'De',De_rng);

models = fieldnames(score);
for k = 1:numel(models)
    M = models{k};
    s = score.(M);
    s(~isfinite(s)) = 0; s = max(0, s);
    p = s / max(eps, sum(s(:)));    % proper prior: sum = 1 over the model’s grid
    priors.(M).grid  = grid.(M);
    priors.(M).score = s;
    priors.(M).prior = p;
end

%% ---------------- (Optional) diagnostics: Maxwell norm vs De ----------------
% if ~isfield(opts,'quiet') || ~opts.quiet
%     De_samp = logspace(log10(De_rng(1)), log10(De_rng(2)), 64);
%     Mnorm   = arrayfun(maxwell_norm, De_samp);
%     figure('Color','w'); loglog(De_samp, Mnorm,'-','LineWidth',1.5);
%     xlabel('De = \lambda_1 / t_c'); ylabel('||M^*||_w (data-driven)');
%     title('Maxwell kernel L^2 norm aggregated over trials');
%     box on;
% end

end

% ============================ Helper functions ============================
function nm = maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell)
% Solve De * dS/dτ + S = A*(τ) on nonuniform τ for each trial (implicit Euler),
% return weighted L2 norm aggregated over trials.
S2 = 0; used = 0;
for k = 1:numel(Acell)
    A = Acell{k};  t = Tcell{k};  w = Wcell{k};
    if numel(A) < 2, continue; end
    S = zeros(size(A));
    for i = 2:numel(A)
        dt = t(i)-t(i-1);
        % implicit Euler (unconditionally stable for stiff De):
        % S_i = ( S_{i-1} + (dt/De)*A_i ) / (1 + dt/De)
        S(i) = ( S(i-1) + (dt/De)*A(i) ) / (1 + dt/De);
    end
    S2 = S2 + sum( (S.^2) .* w );
    used = used + 1;
end
nm = (used==0) * 0 + (used>0) * sqrt(S2/used);
end

function H = hmean01(varargin)
% Harmonic mean of inputs in [0,1], same size arrays, returns in [0,1].
% H = n / sum_i 1/(x_i + eps). Penalizes any axis near 0 (bottleneck).
narginchk(1, inf);
base = varargin{1};
dim  = ndims(base) + 1;   % new trailing dimension for stacking
X = max(0, min(1, base));
for i = 2:nargin
    X = cat(dim, X, max(0, min(1, varargin{i})));
end
n = size(X, dim);
H = n ./ sum(1 ./ (X + eps), dim);
end