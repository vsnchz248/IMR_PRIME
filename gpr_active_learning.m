function results = gpr_active_learning(modelSpec, expData, opts)
% GPR_ACTIVE_LEARNING  Active learning for IMR parameter space exploration
%
% Syntax:
%   results = gpr_active_learning(modelSpec, expData, opts)
%
% Inputs:
%   modelSpec - Structure with fields:
%       .name          : Model name ('Newt', 'NH', 'KV', etc.)
%       .paramNames    : Cell array of parameter names {'mu', 'G', ...}
%       .paramBounds   : [nParams x 2] array of [min, max] bounds
%       .forwardSolver : Function handle @(params) -> logL_trajectory
%   expData - Structure with experimental data (from expDataPrep)
%   opts - Options structure with fields:
%       .n0            : Initial samples (default: 10*d^2 + 5)
%       .maxRounds     : Maximum AL rounds (default: 40)
%       .maxAdded      : Maximum total samples (default: 160*d)
%       .KcapPerRound  : Samples per round (default: 6 + 4*d)
%       .tolRelCI      : Stopping tolerance (default: 0.03)
%       .verbose       : Display progress (default: true)
%
% Outputs:
%   results - Structure containing:
%       .gpr           : Final trained GPR model
%       .X_train       : Training points in feature space [0,1]^d
%       .Y_train       : Log-likelihood evaluations
%       .I_estimate    : Integral estimate (evidence)
%       .I_uncertainty : Uncertainty (combined model + RQMC)
%       .betaGrid      : Beta values for noise marginalization
%       .addedTotal    : Total samples added
%       .rounds        : Number of AL rounds completed
%
% Example:
%   modelSpec = struct('name', 'KV', ...
%                      'paramNames', {{'mu', 'G'}}, ...
%                      'paramBounds', [1e-4 1; 1e2 1e5], ...
%                      'forwardSolver', @(p) solve_KV(p, expData));
%   results = gpr_active_learning(modelSpec, expData);
%
% See also: bayesian_model_selection, build_likelihood_evaluator

% Author: [Your name]
% Date: 2025

%% Parse inputs
if nargin < 3, opts = struct(); end

% Extract dimensions
d = numel(modelSpec.paramNames);
paramBounds = modelSpec.paramBounds;
assert(size(paramBounds, 1) == d, 'paramBounds must be [%d x 2]', d);

% Default options
opts = parse_options(opts, d);

% Initialize RNG for reproducibility
if isfield(opts, 'seed'), rng(opts.seed); end

%% Domain setup: log-aware feature space U ∈ [0,1]^d
% Detect which axes need log-scaling (positive bounds with >100x range)
xmin = paramBounds(:, 1)';
xmax = paramBounds(:, 2)';
log_axes = (xmin > 0) & ((xmax ./ max(xmin, eps)) >= 100);

% Precompute log-space constants for efficiency
domain = setup_domain(xmin, xmax, log_axes);

%% Initial sampling (Sobol in feature space)
sob0 = scramble(sobolset(d), 'MatousekAffineOwen');
U = net(sob0, opts.n0);  % Feature space [0,1]^d
X = domain.fromFeat(U);  % Physical space

% Evaluate initial samples
if opts.verbose
    fprintf('Evaluating %d initial samples...\n', opts.n0);
end
Y = evaluate_samples(X, modelSpec.forwardSolver, opts.verbose);

%% Setup acquisition grid
Nacq = get_acquisition_size(d);
sobA = scramble(sobolset(d), 'MatousekAffineOwen');
Uacq = net(sobA, Nacq);
wA = domain.jacobian(Uacq) / Nacq;  % Physical weights

%% Setup final integration grid
Nint_final = 65536 * 2^(d-1);
sobF = scramble(sobolset(d), 'MatousekAffineOwen');
Uint_final = net(sobF, Nint_final);
wF = domain.jacobian(Uint_final) / Nint_final;

%% Active learning loop
addedTotal = 0;
rounds = 0;
ln10 = log(10);

% Select kernel based on dimensionality
kernelName = select_kernel(d);
basisName = 'constant';

% Initial GP fit
gpr = train_gpr(U, Y, kernelName, basisName);

% RQMC parameters for stopping criterion
[N_rqmc_small, R_rqmc_small] = get_rqmc_params(d, Nacq);

if opts.verbose
    fprintf('\n=== Active Learning for Model: %s ===\n', modelSpec.name);
    fprintf('Dimensions: %d | Initial samples: %d\n', d, opts.n0);
    fprintf('Kernel: %s | Max rounds: %d | Target CI: %.1f%%\n', ...
            kernelName, opts.maxRounds, 100*opts.tolRelCI);
end

while rounds < opts.maxRounds && addedTotal < opts.maxAdded
    rounds = rounds + 1;
    
    % Refit GPR each round (adapts to peak)
    gpr = train_gpr(U, Y, kernelName, basisName);
    
    % Predict on acquisition grid
    [muA, sdA] = predict(gpr, Uacq);
    sdA = max(sdA, 1e-9);
    
    % Transform to log10(L) space statistics
    mu_e = ln10 * muA;
    sd_e = ln10 * sdA;
    
    % Compute acquisition components
    [A, acqInfo] = compute_acquisition(muA, sdA, Y, wA, d, Nacq);
    
    % Build adaptive batch: MAP trust-region + diversification + global refresh
    [ell_vec, ~] = gp_lengthscale_vec(gpr, d);
    newU = build_adaptive_batch(Uacq, A, U, ell_vec, d, ...
                                 opts.KcapPerRound, acqInfo.C);
    
    if isempty(newU)
        if opts.verbose
            fprintf('Round %d: No new points (spacing too tight)\n', rounds);
        end
        break;
    end
    
    % Evaluate new samples
    newX = domain.fromFeat(newU);
    newY = evaluate_samples(newX, modelSpec.forwardSolver, opts.verbose);
    
    % Update dataset
    U = [U; newU];
    Y = [Y; newY];
    addedTotal = addedTotal + size(newU, 1);
    
    % Check stopping criterion (small RQMC for efficiency)
    [Ibar_small, se_rqmc, Isd_model] = ...
        rqmc_integral_stats(gpr, N_rqmc_small, R_rqmc_small, domain, d);
    
    relHalf_comb = 1.96 * sqrt(Isd_model^2 + se_rqmc^2) / max(abs(Ibar_small), eps);
    
    if opts.verbose
        fprintf(['Round %2d: +%2d samples | C=%.2f γ=%.2f | ' ...
                 'I≈%.5g | CI half-width=%.2f%% | Total=%d\n'], ...
                rounds, size(newU, 1), acqInfo.C, acqInfo.gamma, ...
                Ibar_small, 100*relHalf_comb, addedTotal);
    end
    
    if relHalf_comb <= opts.tolRelCI
        if opts.verbose
            fprintf('Converged: CI half-width ≤ %.1f%%\n', 100*opts.tolRelCI);
        end
        break;
    end
end

%% Final integration with robust RQMC
gpr = train_gpr(U, Y, kernelName, basisName);  % Final refit
R_final = (d == 1) * 16 + (d > 1) * 12;
[I_bar, se_rqmc_final, Isd_model_final] = ...
    rqmc_integral_stats(gpr, Nint_final, R_final, domain, d);

sigma_total = sqrt(Isd_model_final^2 + se_rqmc_final^2);

if opts.verbose
    fprintf('\n=== Final Results ===\n');
    fprintf('Added %d samples in %d rounds\n', addedTotal, rounds);
    fprintf('Evidence estimate: %.6g ± %.3g (95%% CI)\n', I_bar, 1.96*sigma_total);
    fprintf('Uncertainty breakdown: model=%.3g, RQMC=%.3g\n', ...
            Isd_model_final, se_rqmc_final);
end

%% Package results
results = struct(...
    'gpr', gpr, ...
    'X_train', domain.fromFeat(U), ...
    'U_train', U, ...
    'Y_train', Y, ...
    'I_estimate', I_bar, ...
    'I_uncertainty', sigma_total, ...
    'I_model_sd', Isd_model_final, ...
    'I_rqmc_se', se_rqmc_final, ...
    'addedTotal', addedTotal, ...
    'rounds', rounds, ...
    'domain', domain, ...
    'modelSpec', modelSpec);

end

%% ==================== Helper Functions ====================

function opts = parse_options(opts, d)
% Set default options
if ~isfield(opts, 'n0'),           opts.n0 = 10*d^2 + 5;      end
if ~isfield(opts, 'maxRounds'),    opts.maxRounds = 40;       end
if ~isfield(opts, 'maxAdded'),     opts.maxAdded = 160*d;     end
if ~isfield(opts, 'KcapPerRound'), opts.KcapPerRound = 6+4*d; end
if ~isfield(opts, 'tolRelCI'),     opts.tolRelCI = 0.03;      end
if ~isfield(opts, 'verbose'),      opts.verbose = true;       end
end

function domain = setup_domain(xmin, xmax, log_axes)
% Setup domain transformations and Jacobian
d = numel(xmin);

% Precompute per-axis constants
loga = zeros(1, d);
logb = zeros(1, d);
dx_du_lin = zeros(1, d);

for j = 1:d
    if log_axes(j)
        loga(j) = log10(max(xmin(j), eps));
        logb(j) = log10(max(xmax(j), xmin(j)+eps));
    else
        dx_du_lin(j) = xmax(j) - xmin(j);
    end
end

% Feature space maps
domain.fromFeat = @(U) u2x_logaware(U, xmin, xmax, log_axes, loga, logb);
domain.toFeat = @(X) x2u_logaware(X, xmin, xmax, log_axes, loga, logb);
domain.jacobian = @(U) jac_logaware(U, xmin, xmax, log_axes, loga, logb, dx_du_lin);
domain.xmin = xmin;
domain.xmax = xmax;
domain.log_axes = log_axes;
end

function Y = evaluate_samples(X, forwardSolver, verbose)
% Evaluate forward solver at sample points
n = size(X, 1);
Y = zeros(n, 1);

if verbose && n > 10
    fprintf('Evaluating %d samples: ', n);
end

for i = 1:n
    Y(i) = forwardSolver(X(i, :));
    if verbose && n > 10 && mod(i, max(1, floor(n/10))) == 0
        fprintf('%d ', i);
    end
end

if verbose && n > 10
    fprintf('Done.\n');
end
end

function kernelName = select_kernel(d)
% Select appropriate kernel based on dimensionality
if d == 1
    kernelName = 'matern52';
elseif d == 2
    kernelName = 'ardsquaredexponential';
else
    kernelName = 'ardrationalquadratic';
end
end

function Nacq = get_acquisition_size(d)
% Dimension-aware acquisition grid size
if d == 1
    Nacq = 16384;
elseif d == 2
    Nacq = 65536;
else
    Nacq = 32768;
end
end

function [N_rqmc, R_rqmc] = get_rqmc_params(d, Nacq)
% RQMC parameters for stopping criterion
if d == 1
    N_rqmc = 2^15;
    R_rqmc = 8;
else
    N_rqmc = max(8192 * 2^(d-1), floor(Nacq/2));
    R_rqmc = 6;
end
end

function gpr = train_gpr(U, Y, kernelName, basisName)
% Train GPR model with automatic hyperparameter optimization
gpr = fitrgp(U, Y, ...
    'KernelFunction', kernelName, ...
    'BasisFunction', basisName, ...
    'Standardize', true);
end

function [A, info] = compute_acquisition(muA, sdA, Y, wA, d, Nacq)
% Compute tempered variance + EI + std acquisition function
ln10 = log(10);
mu_e = ln10 * muA;
sd_e = ln10 * sdA;

% Transform to L-space
LmeanA = exp(mu_e + 0.5*sd_e.^2);
LvarA = exp(2*mu_e + sd_e.^2) .* (exp(sd_e.^2) - 1);

% Primary: variance contribution (tempered by concentration)
A_var = (wA.^2) .* LvarA;

% Compute concentration factor
kTop = max(1, round(0.01 * Nacq));
Lw_sorted = sort(LmeanA .* wA, 'descend');
shareTop = sum(Lw_sorted(1:kTop)) / max(sum(Lw_sorted), eps);
C = min(1, max(0, (shareTop - 0.01) / (0.60 - 0.01)));

% Dimension-dependent tempering
if d == 1
    gamma_hi = 0.95; gamma_lo = 0.80;
elseif d == 2
    gamma_hi = 0.90; gamma_lo = 0.60;
else
    gamma_hi = 0.85; gamma_lo = 0.45;
end
gamma = gamma_hi - (gamma_hi - gamma_lo) * C;

A_var = A_var .^ gamma;
A_var = A_var / max(A_var + eps);

% Stabilizers: EI and std
yBest = max(Y);
Z = (muA - yBest) ./ sdA;
EI = (muA - yBest) .* normcdf(Z) + sdA .* normpdf(Z);
EI(sdA <= 1e-12) = 0;
EI = EI / max(EI + eps);

A_std = sdA / max(sdA + eps);

% Auto-weighted combination
mVar = mean(A_var); mEI = mean(EI); mStd = mean(A_std);
S = mVar + mEI + mStd + eps;
aVar = mVar/S; aEI = mEI/S; aStd = mStd/S;

A = aVar*A_var + aEI*EI + aStd*A_std;

info = struct('C', C, 'gamma', gamma, 'weights', [aVar, aEI, aStd]);
end

function [ell_vec, sigmaF] = gp_lengthscale_vec(gpr, d)
% Extract length scales from GP kernel
KP = gpr.KernelInformation.KernelParameters;
if strncmpi(gpr.KernelFunction, 'ard', 3)
    ell_vec = KP(1:d)';
    sigmaF = KP(end);
else
    ell_vec = KP(1) * ones(1, d);
    sigmaF = KP(end);
end
end

function newU = build_adaptive_batch(Uacq, A, U, ell_vec, d, Kcap, C)
% Build batch: MAP trust-region + diversified + global refresh

% Cap length scales for numerical stability
ell_cap = 0.40 * sqrt(d);
ell_vec = min(ell_vec, ell_cap);

% MAP trust-region size (adaptive with concentration)
if d == 1
    step = (0.30 - 0.10*C) * ell_vec;
    qBase = 2;
elseif d == 2
    step = (0.28 - 0.10*C) * ell_vec;
    qBase = 4;
else
    step = (0.22 - 0.10*C) * ell_vec;
    qBase = 4;
end
qLocal = min(Kcap-3, max(1, qBase + round(3*C) + (d==3)*2));

% MAP location (highest predicted mean × weight)
LmeanA_approx = exp(A);  % Proxy for mean
[~, iMAP] = max(LmeanA_approx);
u0 = Uacq(iMAP, :);

% Build local candidates (axial + small diagonals)
localCands = u0;
[~, ax] = sort(ell_vec, 'descend');
maxAxes = min(d, 2);

for j = 1:maxAxes
    a = ax(j);
    u_plus = u0; u_plus(a) = min(1, u0(a) + step(a));
    u_minus = u0; u_minus(a) = max(0, u0(a) - step(a));
    localCands = [localCands; u_plus; u_minus]; %#ok<AGROW>
end

if d >= 2
    a1 = ax(1); a2 = ax(min(2, d));
    for s1 = [-1, 1]
        for s2 = [-1, 1]
            u_d = u0;
            u_d(a1) = min(1, max(0, u0(a1) + s1*0.7*step(a1)));
            u_d(a2) = min(1, max(0, u0(a2) + s2*0.7*step(a2)));
            localCands = [localCands; u_d]; %#ok<AGROW>
        end
    end
end

% Filter candidates by minimum separation
keep = true(size(localCands, 1), 1);
for i = 1:size(localCands, 1)
    if ~isempty(U)
        dists = sqrt(sum((U - localCands(i, :)).^2, 2));
        keep(i) = all(dists > 0.02 * median(ell_vec));
    end
end

localU = unique(localCands(keep, :), 'rows', 'stable');
qLocal = min(qLocal, size(localU, 1));
localU = localU(1:qLocal, :);

% Diversified batch
Krem = max(0, Kcap - qLocal - 1);
if d == 3, M = min(12000, size(Uacq, 1));
elseif d == 2, M = min(8000, size(Uacq, 1));
else, M = min(4000, size(Uacq, 1));
end

[~, ord] = sort(A, 'descend');
Ushort = Uacq(ord(1:M), :);
As = A(ord(1:M));

factors = [0.25 0.15 0.10 0.05 0.02 0.0];
newUdiv = [];
for f = factors
    minSep = f * median(ell_vec);
    newUdiv = fps_diversify(Ushort, As, U, minSep, Krem, ell_vec);
    if ~isempty(newUdiv), break; end
end

% Global refresh
Kglob = max(0, Kcap - size(localU, 1) - size(newUdiv, 1));
newUglob = [];
if Kglob > 0
    onescore = ones(size(Ushort, 1), 1);
    newUglob = fps_diversify(Ushort, onescore, U, 0.10*median(ell_vec), Kglob, ell_vec);
end

newU = [localU; newUdiv; newUglob];
end

function newU = fps_diversify(Ucand, score, Uexist, minSep, Kmax, ell_vec)
% Farthest-point sampling with ARD-aware distance
N = size(Ucand, 1);
if N == 0 || Kmax <= 0, newU = []; return; end

scale = 1 ./ max(ell_vec(:).', 1e-6);
Us = Ucand .* scale;

if ~isempty(Uexist)
    Es = Uexist .* scale;
    distE = inf(N, 1);
    for j = 1:size(Es, 1)
        distE = min(distE, sqrt(sum((Us - Es(j, :)).^2, 2)));
    end
else
    distE = inf(N, 1);
end

valid = find(distE > minSep);
if isempty(valid), newU = []; return; end

[~, k] = max(score(valid));
i0 = valid(k);
sel = i0;
newU = Ucand(i0, :);

while size(newU, 1) < Kmax
    if numel(sel) == 1
        distS = sqrt(sum((Us - Us(sel, :)).^2, 2));
    else
        distS = inf(N, 1);
        for j = 1:numel(sel)
            distS = min(distS, sqrt(sum((Us - Us(sel(j), :)).^2, 2)));
        end
    end
    
    dmin = min(distE, distS);
    candMask = dmin > minSep;
    if ~any(candMask), break; end
    
    util = score .* dmin;
    util(~candMask) = -inf;
    [best, ib] = max(util);
    if ~isfinite(best), break; end
    
    sel(end+1, 1) = ib; %#ok<AGROW>
    newU = [newU; Ucand(ib, :)]; %#ok<AGROW>
end
end

function [I_bar, se_rqmc, I_sd_model] = rqmc_integral_stats(gpr, N, R, domain, d)
% Robust RQMC integral estimation with independent random shifts
Ivals = zeros(R, 1);
Isd2 = zeros(R, 1);
sob = scramble(sobolset(d), 'MatousekAffineOwen');
ln10 = log(10);

for r = 1:R
    U0 = net(sob, N);
    shift = rand(1, d);
    U = mod(U0 + shift, 1);
    w = domain.jacobian(U) / N;
    
    [mu, sd] = predict(gpr, U);
    mu_e = ln10 * mu;
    sd_e = ln10 * sd;
    
    Lmean = exp(mu_e + 0.5*sd_e.^2);
    Lvar = exp(2*mu_e + sd_e.^2) .* (exp(sd_e.^2) - 1);
    
    I_r = sum(w .* Lmean);
    I_sd_r = sqrt(sum(w.^2 .* Lvar));
    
    Ivals(r) = I_r;
    Isd2(r) = I_sd_r^2;
end

I_bar = mean(Ivals);
se_rqmc = std(Ivals, 0) / sqrt(R);
I_sd_model = sqrt(mean(Isd2));
end

%% ==================== Domain Transformation Functions ====================

function X = u2x_logaware(U, xmin, xmax, log_axes, loga, logb)
U = double(U);
X = zeros(size(U));
for j = 1:numel(xmin)
    if log_axes(j)
        X(:, j) = 10.^(loga(j) + U(:, j).*(logb(j) - loga(j)));
    else
        X(:, j) = xmin(j) + U(:, j).*(xmax(j) - xmin(j));
    end
end
end

function U = x2u_logaware(X, xmin, xmax, log_axes, loga, logb)
X = double(X);
U = zeros(size(X));
for j = 1:numel(xmin)
    if log_axes(j)
        U(:, j) = (log10(max(X(:, j), eps)) - loga(j)) ./ max(logb(j) - loga(j), eps);
    else
        U(:, j) = (X(:, j) - xmin(j)) ./ max(xmax(j) - xmin(j), eps);
    end
end
U = min(1, max(0, U));
end

function w = jac_logaware(U, xmin, xmax, log_axes, loga, logb, dx_du_lin)
X = u2x_logaware(U, xmin, xmax, log_axes, loga, logb);
w = ones(size(U, 1), 1);
for j = 1:numel(xmin)
    if log_axes(j)
        w = w .* (log(10) * (logb(j) - loga(j)) .* X(:, j));
    else
        w = w .* dx_du_lin(j);
    end
end
end