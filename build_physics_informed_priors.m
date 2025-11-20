function priors = build_physics_informed_priors(expData, opts)
% BUILD_PHYSICS_INFORMED_PRIORS  Compute data-driven axis needs for model priors
%
% Syntax:
%   priors = build_physics_informed_priors(expData, opts)
%
% Inputs:
%   expData - Structure from prepare_experimental_data
%   opts    - Options (optional, for future extensions)
%
% Outputs:
%   priors - Structure with fields:
%       .axis_need     : Data-driven needs for each axis type
%           .elastic   : Need for elastic (G) axis [0,1]
%           .maxwell   : Need for Maxwell (λ₁) axis [0,1]
%           .nonlinear : Need for nonlinearity (α) axis [0,1]
%       .kernel_norms  : L² norms of stress kernels
%           .normA     : Viscous kernel ||A*||
%           .normB     : Elastic kernel ||B||
%       .scales        : Characteristic scales
%           .tc_ref    : Reference characteristic time
%
% Description:
%   Analyzes experimental bubble dynamics to quantify how much the data
%   require different constitutive features (viscosity, elasticity, 
%   relaxation, nonlinearity). These "needs" are used to:
%   - Penalize models with unnecessary complexity
%   - Weight parameter space regions appropriately
%   - Enable data-driven model selection without arbitrary knobs
%
% Example:
%   expData = prepare_experimental_data(1);
%   priors = build_physics_informed_priors(expData);
%   fprintf('Elastic need: %.3f\n', priors.axis_need.elastic);
%
% See also: bayesian_model_selection

% Author: [Your name]
% Date: 2025

if nargin < 2, opts = struct(); end

%% Extract data
Rmatrix = expData.Rmatrix;
Rdotmatrix = expData.Rdotmatrix;
tmatrix = expData.tmatrix;
tc = expData.tc;
Rmax_range = expData.Rmax_range;
Req_each = expData.Req_each;
mask = expData.mask;

[~, J] = size(Rmatrix);

% Reference characteristic time
tc_vec = tc(isfinite(tc) & tc > 0);
assert(~isempty(tc_vec), 'No valid characteristic times');
tc_ref = median(tc_vec);

%% Compute stress kernel norms from gated data
A2 = 0;  % Viscous kernel energy
B2 = 0;  % Elastic kernel energy
D2perp = 0;  % Nonlinear kernel energy (orthogonal to {A*, B})
used = 0;

% Storage for Maxwell kernel computation
Acell = {};
Tcell = {};
Wcell = {};
Bcell = {};

for j = 1:J
    r = Rmatrix(:, j);
    dr = Rdotmatrix(:, j);
    tau = tmatrix(:, j);
    
    good = isfinite(r) & isfinite(dr) & (r > 0) & mask(:, j);
    r = r(good);
    dr = dr(good);
    tau = tau(good);
    
    if numel(r) < 8, continue; end
    
    % Stretch ratio
    q = Rmax_range(j) / max(Req_each(j), 1e-12);
    lam = q * r;
    
    % Time-step weights (probability over nonuniform τ)
    dtau = [diff(tau); max(tau(end)-tau(end-1), eps)];
    w = dtau / sum(dtau);
    
    % Viscous kernel: A* = Ṙ*/R*
    Astar = dr ./ max(r, 1e-12);
    
    % Neo-Hookean kernel: B(λ) = 0.5[4λ⁻¹ + λ⁻⁴ - 5]
    Bnh = 0.5 * (4 ./ max(lam, 1e-12) + lam.^(-4) - 5);
    
    % Accumulate weighted L² norms
    A2 = A2 + sum((Astar.^2) .* w);
    B2 = B2 + sum((Bnh.^2) .* w);
    
    % Nonlinear direction: D*(λ) = ∂S_qNH/∂α (orthogonalized)
    bracket = 27/40 + (1/8).*lam.^(-8) + (1/5).*lam.^(-5) + lam.^(-2) - 2.*lam;
    Dstar_raw = -3.*Bnh + 2.*bracket;
    
    % Orthogonalize against span{A*, Bnh}
    W = sqrt(w);
    X = [W.*Astar, W.*Bnh];
    y = W.*Dstar_raw;
    
    coef = pinv(X.'*X) * (X.'*y);
    Dproj = coef(1).*Astar + coef(2).*Bnh;
    Dperp = Dstar_raw - Dproj;
    
    D2perp = D2perp + sum((Dperp.^2) .* w);
    
    % Store for Maxwell computation
    Acell{end+1} = Astar; %#ok<AGROW>
    Tcell{end+1} = tau;   %#ok<AGROW>
    Wcell{end+1} = w;     %#ok<AGROW>
    Bcell{end+1} = Bnh;   %#ok<AGROW>
    
    used = used + 1;
end

assert(used > 0, 'No valid trials after gating');

normA = sqrt(A2 / used);
normB = sqrt(B2 / used);
normDperp = sqrt(D2perp / used);

%% Elastic need: residual of B relative to span{A*, M*(De)}
% Test multiple De values and find minimum residual
De_samp = logspace(log10(1e-3), log10(65), 64);
res2_sum = 0;
B2_sum = 0;
used_em = 0;

for j = 1:numel(Acell)
    A = Acell{j};
    t = Tcell{j};
    w = Wcell{j};
    B = Bcell{j};
    
    if numel(A) < 2 || numel(B) ~= numel(A), continue; end
    
    W = sqrt(w(:));
    Acol = W .* A(:);
    y = W .* B(:);
    
    best_r2 = inf;
    for d = 1:numel(De_samp)
        De = De_samp(d);
        % Maxwell: De*dS/dτ + S = A*
        M = zeros(size(A));
        for i = 2:numel(A)
            dt = t(i) - t(i-1);
            M(i) = (M(i-1) + (dt/De)*A(i)) / (1 + dt/De);
        end
        
        X = [Acol, W.*M(:)];
        coef = pinv(X.'*X) * (X.'*y);
        r = y - X*coef;
        r2 = sum(r.^2);
        
        if r2 < best_r2, best_r2 = r2; end
    end
    
    if isfinite(best_r2)
        res2_sum = res2_sum + best_r2;
        B2_sum = B2_sum + sum(y.^2);
        used_em = used_em + 1;
    end
end

if used_em > 0 && B2_sum > 0
    e_need = res2_sum / B2_sum;  % Fraction of B variance unexplained by {A*, M*}
    e_need = max(0, min(1, e_need));
    e_need = e_need^2;  % Sharpen (small stays small, mid attenuates)
else
    e_need = 0.5;  % Neutral default
end

%% Maxwell need: best Maxwell kernel vs viscous
maxwell_norm_handle = @(De) maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell);
De_samp = logspace(log10(1e-3), log10(65), 64);
Mnorms = arrayfun(maxwell_norm_handle, De_samp);
Mstar = max(Mnorms);
m_need = (Mstar^2) / max(normA^2 + Mstar^2, eps);
m_need = max(0, min(1, m_need));

%% Nonlinearity need: orthogonalized energy
nl_need = (normDperp^2) / max(normA^2 + normB^2 + normDperp^2, eps);
nl_need = max(0, min(1, nl_need));

%% Package results
priors = struct();
priors.axis_need = struct(...
    'elastic', e_need, ...
    'maxwell', m_need, ...
    'nonlinear', nl_need);

priors.kernel_norms = struct(...
    'normA', normA, ...
    'normB', normB, ...
    'normDperp', normDperp);

priors.scales = struct(...
    'tc_ref', tc_ref);

fprintf('\n=== Physics-Informed Priors ===\n');
fprintf('Kernel norms: ||A*||=%.3g, ||B||=%.3g, ||D⊥||=%.3g\n', ...
        normA, normB, normDperp);
fprintf('Axis needs: elastic=%.3f, maxwell=%.3f, nonlinear=%.3f\n', ...
        e_need, m_need, nl_need);

end

%% ==================== Maxwell Kernel Solver ====================

function nm = maxwell_L2_norm_over_trials(De, Acell, Tcell, Wcell)
% Solve De*dS/dτ + S = A*(τ) via implicit Euler, return weighted norm
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
        S(i) = (S(i-1) + (dt/De)*A(i)) / (1 + dt/De);
    end
    
    S2 = S2 + sum((S.^2) .* w);
    used = used + 1;
end

nm = (used > 0) * sqrt(S2 / used);
end