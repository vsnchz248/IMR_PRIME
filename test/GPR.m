% Active integration on log10(L) with log-aware inputs and Jacobian-corrected
% weights. Same acquisition stack: Var(I)-driven (tempered), EI & std stabilizers,
% local MAP trust-region + diversified batch + tiny global refresh, and RQMC CIs.
% Deterministic oracle (no observation noise).
clear; clc; close all;

%% ================= USER CHOICE =================
% Example bounds; edit to your problem (any d). 1D demo:
% xmin = [1e-4];   xmax = [1]; % Example 1D
xmin = [1e2];   xmax = [1e5]; % Example 1D
% xmin = [1e-4, 1e2]; xmax = [1, 1e5]; % Example 2D
% xmin = [1e-4, 1e2, 1e-3]; xmax = [1, 1e5, 10]; % Example 3D 
xmin = xmin(:)'; xmax = xmax(:)';        % row vectors
d = numel(xmin);
rng(42);

%% ===== Domain helpers: log-aware feature space U in [0,1]^d =====
% Axes to treat in log10 (positive and ≥100× span)
log_axes = (xmin > 0) & ((xmax./max(xmin,eps)) >= 100);

% Precompute per-axis constants for U<->X and Jacobian |∂x/∂u|
loga = zeros(1,d); logb = zeros(1,d); dx_du_lin = zeros(1,d);
for j = 1:d
    if log_axes(j)
        loga(j) = log10(max(xmin(j), eps));
        logb(j) = log10(max(xmax(j), xmin(j)+eps));
    else
        dx_du_lin(j) = xmax(j) - xmin(j);
    end
end

% Feature-space maps (U<->X) and Jacobian of x(u)
fromFeat = @(U) u2x_logaware(U, xmin, xmax, log_axes, loga, logb);     % U->[X]
toFeat   = @(X) x2u_logaware(X, xmin, xmax, log_axes, loga, logb);     % X->[U]
jacobian = @(U) jac_logaware(U, xmin, xmax, log_axes, loga, logb, dx_du_lin); % |∂x/∂u|

%% ===== Toy truth (dimension-agnostic): values ~ [1e2, 1e5] for testing =====
% Build a smooth multimodal u(X)∈[0,1] in a "log box" Z
zmap = @(X) normalize01_logaware(X, xmin, xmax, log_axes);
mu_z  = linspace(0.35, 0.65, d);                 % bump centers per dim
sig_z = 0.12 * (0.85 .^ (0:d-1));                % widths per dim
quadZ = @(Z) sum(((Z - mu_z).^2) ./ (2*sig_z.^2), 2);
u01   = @(X) clip01( 0.45*(1 + sin(2*pi*sum(zmap(X),2))) ...
                   + 0.55*exp(-quadZ(zmap(X))) );
gtrue10_nd = @(X) 2 + 3*u01(X);                  % log10 L
ftrue_nd   = @(X) 10.^gtrue10_nd(X);             % L(X)
I_true_analytic = NaN;                            % unknown for this toy
ln10 = log(10);

% Quick sanity check of realized output range on a coarse net
sobChk = scramble(sobolset(d),'MatousekAffineOwen');
Uchk   = net(sobChk, 8192);
vals   = ftrue_nd(fromFeat(Uchk));
fprintf('Observed L range on coarse grid: [%.2g, %.2g] (target [1e2, 1e5])\n', min(vals), max(vals));

%% ================= DIMENSION-AWARE SIZES =================
n0 = 10*d^2 + 5;                       % 15, 45, 95 for d=1,2,3

if d==1, Nacq = 16384;       % prediction-only grid for acquisition
elseif d==2, Nacq = 65536;
else, Nacq = 32768; end

Nint_final   = 65536 * 2^(d-1);        % per-scramble GP quadrature size
maxRounds    = 40;
KcapPerRound = 6 + 4*d;                 % 10, 14, 18
maxAdded     = 160*d;
tolRelCI     = 0.03;                    % stop when 95% half-width ≤ 3%

% Per-round small RQMC for the stop rule (predictions only)
if d==1,  N_rqmc_small = 2^15; R_rqmc_small = 8;
else,     N_rqmc_small = max(8192 * 2^(d-1), floor(Nacq/2)); R_rqmc_small = 6;
end

%% ================= INITIAL DATA (Sobol in feature space) =================
sob0 = scramble(sobolset(d),'MatousekAffineOwen');
U = net(sob0, n0);                   % U∈[0,1]^d (feature space)
Y = gtrue10_nd(fromFeat(U));         % deterministic log10(L) evaluations

%% ============= ACQ & FINAL GRIDS (Sobol in feature space) =============
sobA   = scramble(sobolset(d),'MatousekAffineOwen');
Uacq   = net(sobA, Nacq);                       % feature candidates
wA     = jacobian(Uacq) / Nacq;                 % physical weights via |∂x/∂u|

sobF        = scramble(sobolset(d),'MatousekAffineOwen');
Uint_final  = net(sobF, Nint_final);            % feature nodes for final report
wF          = jacobian(Uint_final) / Nint_final;

%% ================= GP HELPERS (automatic hyperparams) =================
% Kernels: 1D Matern-5/2; 2D ARD SE; 3D ARD Rational Quadratic
if d==1,      kernelName = 'matern52';
elseif d==2,  kernelName = 'ardsquaredexponential';
else,         kernelName = 'ardrationalquadratic';
end
basisName = 'constant';

% Initial fit (fitrgp optimizes kernel params & sigma)
gpr = trainGP(U, Y, kernelName, basisName);

%% ================= ACTIVE LOOP =================
addedTotal = 0; rounds = 0;

while rounds < maxRounds && addedTotal < maxAdded
    rounds = rounds + 1;

    % Re-optimize each round (helps ARD & length-scales track the peak)
    gpr = trainGP(U, Y, kernelName, basisName);

    % Posterior on acquisition grid (feature space)
    [muA, sdA] = predict(gpr, Uacq);
    sdA = max(sdA, 1e-9);
    mu_e = ln10*muA; sd_e = ln10*sdA;

    % Primary: absolute contribution to Var(I)
    LmeanA = exp(mu_e + 0.5*sd_e.^2);
    LvarA  = exp(2*mu_e + sd_e.^2) .* (exp(sd_e.^2) - 1);
    A_var  = (wA.^2) .* LvarA;

    % Mass concentration → tempering exponent gamma (auto)
    kTop = max(1, round(0.01 * Nacq));
    Lw_sorted = sort(LmeanA .* wA, 'descend');
    shareTop  = sum(Lw_sorted(1:kTop)) / max(sum(Lw_sorted), eps); % (0,1)
    C = min(1, max(0, (shareTop - 0.01) / (0.60 - 0.01)));        % [0,1]
    switch d
        case 1, gamma_hi = 0.95; gamma_lo = 0.80;
        case 2, gamma_hi = 0.90; gamma_lo = 0.60;
        otherwise, gamma_hi = 0.85; gamma_lo = 0.45;
    end
    gamma = gamma_hi - (gamma_hi - gamma_lo)*C;
    A_var = A_var .^ gamma;
    A_var = A_var / max(A_var + eps);

    % Stabilizers: EI(log10) and raw std
    yBest = max(Y);
    Z  = (muA - yBest) ./ sdA;
    EI = (muA - yBest).*normcdf(Z) + sdA.*normpdf(Z);
    EI(sdA<=1e-12) = 0; EI = EI / max(EI + eps);
    A_std = sdA;       A_std = A_std / max(A_std + eps);

    % Auto-weights by average mass
    mVar = mean(A_var); mEI = mean(EI); mStd = mean(A_std);
    S = mVar + mEI + mStd + eps;
    aVar = mVar/S; aEI = mEI/S; aStd = mStd/S;

    % Final acquisition
    A = aVar*A_var + aEI*EI + aStd*A_std;

    % MAP trust-region micro-batch (axial + small diagonals)
    [ell_vec_raw, ~] = gp_lengthscale_vec(gpr, d);
    ell_cap = 0.40 * sqrt(d);
    ell_vec = min(ell_vec_raw, ell_cap);
    switch d
        case 1, step = (0.30 - 0.10*C) * ell_vec;
        case 2, step = (0.28 - 0.10*C) * ell_vec;
        otherwise, step = (0.22 - 0.10*C) * ell_vec;
    end
    switch d
        case 1, qBase = 2;
        case 2, qBase = 4;
        otherwise, qBase = 4;
    end
    qLocal = min(KcapPerRound-3, max(1, qBase + round(3*C) + (d==3)*2));

    [~, iMAP] = max(LmeanA); u0 = Uacq(iMAP,:);
    localCands = u0;
    [~, ax] = sort(ell_vec, 'descend'); maxAxes = min(d,2);
    for j = 1:maxAxes
        a = ax(j);
        u_plus = u0; u_minus = u0;
        u_plus(a)  = min(1, u0(a) + step(a));
        u_minus(a) = max(0, u0(a) - step(a));
        localCands = [localCands; u_plus; u_minus]; %#ok<AGROW>
    end
    if d >= 2
        a1 = ax(1); a2 = ax(min(2,d));
        for s1 = [-1,1], for s2 = [-1,1]
            u_d = u0;
            u_d(a1) = min(1, max(0, u0(a1) + s1*0.7*step(a1)));
            u_d(a2) = min(1, max(0, u0(a2) + s2*0.7*step(a2)));
            localCands = [localCands; u_d]; %#ok<AGROW>
        end, end
    end
    keep = true(size(localCands,1),1);
    for i=1:size(localCands,1)
        keep(i) = isempty(U) || all(row_dist(localCands(i,:), U) > 0.02*median(ell_vec));
    end
    localU = unique(localCands(keep,:), 'rows', 'stable');
    qLocal = min(qLocal, size(localU,1));
    localU = localU(1:qLocal,:);

    % Diversified batch for remaining slots (ARD-aware distances)
    Krem = max(0, KcapPerRound - qLocal - 1); % keep 1 for global refresh
    if d==3, M = min(12000, Nacq);
    elseif d==2, M = min(8000, Nacq);
    else, M = min(4000, Nacq); end
    [~, ord] = sort(A, 'descend');
    Ushort = Uacq(ord(1:M),:);  As = A(ord(1:M));
    factors = [0.25 0.15 0.10 0.05 0.02 0.0];
    newUdiv = [];
    for f = factors
        minSep_try = f * median(ell_vec);
        newUdiv = fps_diversify(Ushort, As, U, minSep_try, Krem, ell_vec);
        if ~isempty(newUdiv), break; end
    end

    % Tiny global refresh
    Kglob = max(0, KcapPerRound - size(localU,1) - size(newUdiv,1));
    newUglob = [];
    if Kglob > 0
        onescore = ones(size(Ushort,1),1);
        newUglob = fps_diversify(Ushort, onescore, U, 0.10*median(ell_vec), Kglob, ell_vec);
    end

    % Combined batch
    newU = [localU; newUdiv; newUglob];
    if isempty(newU)
        fprintf('r=%2d: spacing still too tight — no new points.\n', rounds);
        break;
    end

    % Deterministic oracle calls (in physical X)
    newY = gtrue10_nd(fromFeat(newU));
    U = [U; newU];  Y = [Y; newY];
    addedTotal = addedTotal + size(newU,1);

    % Small RQMC (independent shifts) for a reliable stop rule
    [Ibar_small, se_rqmc_small, Isd_model_small] = ...
        rqmc_integral_mean_and_SE(gpr, N_rqmc_small, R_rqmc_small, jacobian, d);

    relHalf_comb = 1.96 * sqrt(Isd_model_small^2 + se_rqmc_small^2) / max(abs(Ibar_small), eps);
    fprintf(['r=%2d  +%2d  C=%.2f gamma=%.2f qLoc=%d  ' ...
             'Ibar≈%.6f  modelSD=%.6f  rqmcSE=%.3g  half%%=%.2f%%  total=%d\n'], ...
            rounds, size(newU,1), C, gamma, qLocal, Ibar_small, Isd_model_small, se_rqmc_small, 100*relHalf_comb, addedTotal);

    if relHalf_comb <= tolRelCI
        fprintf('Stop: combined relative half-width ≤ %.1f%%\n', 100*tolRelCI);
        break;
    end
end

%% ================= FINAL REPORT (robust RQMC) =================
gpr = trainGP(U, Y, kernelName, basisName); % final refit

R_final = (d==1) * 16 + (d>1) * 12;
[I_bar, se_rqmc_final, Isd_model_final] = ...
    rqmc_integral_mean_and_SE(gpr, Nint_final, R_final, jacobian, d);

sigma_total = sqrt(Isd_model_final^2 + se_rqmc_final^2);
CI95 = [I_bar - 1.96*sigma_total, I_bar + 1.96*sigma_total];
absErr = abs(I_true_analytic - I_bar);
relErr = absErr / max(I_true_analytic, eps);

fprintf('\nAdded %d pts total in %d rounds\n', addedTotal, rounds);
fprintf('I ≈ %.6f  (95%% CI: [%.6f, %.6f]) | components: model sd=%.3g, RQMC se=%.3g | toy rel err = %.2f%%\n', ...
        I_bar, CI95(1), CI95(2), Isd_model_final, se_rqmc_final, 100*relErr);

% Deterministic large-net cross-check (GP-only; one scramble)
N_det = max(2^17, Nint_final);
sobChk = scramble(sobolset(d),'MatousekAffineOwen');
U_det  = net(sobChk, N_det);
w_det  = jacobian(U_det) / N_det;
[mu_det, sd_det] = predict(gpr, U_det);
[I_det, ~] = int_stats_log10(mu_det, sd_det, w_det);
fprintf('Deterministic large-net check: I_det = %.6f (diff to RQMC mean = %.3g)\n', I_det, I_det - I_bar);

%% ==================== 1D plots (log-x if positive) ====================
if d==1
    [muF_plot, sdF_plot] = predict(gpr, Uint_final);
    [~,~,~, Lvar_plot] = int_stats_log10(muF_plot, sdF_plot, wF);
    xF = fromFeat(Uint_final(:,1));
    [xF, ix] = sort(xF); muF_plot = muF_plot(ix); sdF_plot = sdF_plot(ix); Lvar_plot = Lvar_plot(ix);

    fig = figure('Color','w','Units','normalized','Position',[0.18 0.08 0.64 0.82]); movegui(fig,'center');
    subplot(2,1,1); hold on; box on;
    plot(xF, ftrue_nd([xF]), 'k--','LineWidth',1.5);
    plot(xF, 10.^muF_plot,  'r-','LineWidth',2.0);
    fill([xF; flipud(xF)], [10.^(muF_plot-1.96*sdF_plot); flipud(10.^(muF_plot+1.96*sdF_plot))], ...
         [1 0.8 0.8], 'EdgeColor','none', 'FaceAlpha',0.35);
    plot(fromFeat(U), 10.^Y, 'ko','MarkerFaceColor','y');
    legend('True L(x)','Final median','Final 95% band','Train pts','Location','best');
    xlabel('x'); ylabel('L(x)'); title(sprintf('Active Integration (d=1) | +%d pts', addedTotal));
    if xmin(1) > 0, set(gca,'XScale','log'); xlim([xmin(1) xmax(1)]);
        xticks(10.^(floor(log10(xmin(1))):ceil(log10(xmax(1)))));
        ax = gca; ax.XMinorGrid = 'on'; grid on; end

    subplot(2,1,2); hold on; box on;
    plot(xF, sqrt(max(Lvar_plot,0)),'r-','LineWidth',2.0);
    legend('Final \sigma_L(x)','Location','best');
    xlabel('x'); ylabel('\sigma_{L}(x)'); title('Pointwise uncertainty');
    if xmin(1) > 0, set(gca,'XScale','log'); xlim([xmin(1) xmax(1)]);
        xticks(10.^(floor(log10(xmin(1))):ceil(log10(xmax(1)))));
        ax = gca; ax.XMinorGrid = 'on'; grid on; end
end

%% ======================= LOCAL FUNCTIONS =======================
function mdl = trainGP(Uin, Yin, kernelName, basisName)
    mdl = fitrgp(Uin, Yin, ...
        'KernelFunction', kernelName, ...
        'BasisFunction',  basisName, ...
        'Standardize',    true);
end

function [I_est, I_sd, Lmean, Lvar] = int_stats_log10(mu10, sd10, w)
    ln10  = log(10);
    mu_e  = ln10*mu10;
    sd_e  = ln10*sd10;
    Lmean = exp(mu_e + 0.5*sd_e.^2);
    Lvar  = exp(2*mu_e + sd_e.^2) .* (exp(sd_e.^2) - 1);
    I_est = sum(w .* Lmean);
    I_sd  = sqrt(sum(w.^2 .* Lvar));
end

function [ell_vec, sigmaF] = gp_lengthscale_vec(gpr, d)
    KP = gpr.KernelInformation.KernelParameters;
    if strncmpi(gpr.KernelFunction,'ard',3)
        ell_vec = KP(1:d)'; sigmaF = KP(end);
    else
        ell_vec = KP(1)*ones(1,d); sigmaF = KP(end);
    end
end

function newU = fps_diversify(Ucand, score, Uexist, minSep, Kmax, ell_vec)
    N = size(Ucand,1);
    if N==0 || Kmax<=0, newU = []; return; end
    scale = 1 ./ max(ell_vec(:).', 1e-6);
    Us = Ucand .* scale; Es = Uexist .* scale;
    if ~isempty(Es)
        try,  distE = min(pdist2(Us, Es, 'euclidean'), [], 2);
        catch, distE = inf(N,1); for j=1:size(Es,1), distE = min(distE, sqrt(sum((Us-Es(j,:)).^2,2))); end
        end
    else, distE = inf(N,1);
    end
    valid = find(distE > minSep); if isempty(valid), newU = []; return; end
    [~, k] = max(score(valid)); i0 = valid(k);
    sel = i0; newU = Ucand(i0,:);
    while size(newU,1) < Kmax
        if numel(sel)==1, distS = sqrt(sum((Us-Us(sel,:)).^2,2));
        else
            try,  distS = min(pdist2(Us, Us(sel,:), 'euclidean'), [], 2);
            catch, distS = inf(N,1); for j=1:numel(sel), distS = min(distS, sqrt(sum((Us-Us(sel(j),:)).^2,2))); end
            end
        end
        dmin = min(distE, distS);
        candMask = dmin > minSep; if ~any(candMask), break; end
        util = score .* dmin; util(~candMask) = -inf;
        [best, ib] = max(util); if ~isfinite(best), break; end
        sel(end+1,1) = ib; %#ok<AGROW>
        newU = [newU; Ucand(ib,:)]; %#ok<AGROW>
    end
end

function d = row_dist(u, U), du = U - u; d = sqrt(sum(du.^2, 2)); end

function [I_bar, se_rqmc, I_sd_model_bar] = rqmc_integral_mean_and_SE(gpr, N, R, jacobian_fn, d)
    Ivals = zeros(R,1); Isd2 = zeros(R,1);
    sob = scramble(sobolset(d),'MatousekAffineOwen');
    for r = 1:R
        U0 = net(sob, N);
        shift = rand(1,d);
        U = mod(U0 + shift, 1);
        w = jacobian_fn(U) / N;             % physical weights via Jacobian
        [mu, sd] = predict(gpr, U);
        [I_r, I_sd_r] = int_stats_log10(mu, sd, w);
        Ivals(r) = I_r; Isd2(r) = I_sd_r.^2;
    end
    I_bar          = mean(Ivals);
    se_rqmc        = std(Ivals, 0) / sqrt(R);
    I_sd_model_bar = sqrt(mean(Isd2));
end

% -------- log-aware maps (dimension-agnostic) --------
function X = u2x_logaware(U, xmin_, xmax_, log_axes_, loga_, logb_)
    U = double(U); X = zeros(size(U));
    for j = 1:numel(xmin_)
        if log_axes_(j)
            X(:,j) = 10.^(loga_(j) + U(:,j).*(logb_(j)-loga_(j)));
        else
            X(:,j) = xmin_(j) + U(:,j).*(xmax_(j)-xmin_(j));
        end
    end
end

function U = x2u_logaware(X, xmin_, xmax_, log_axes_, loga_, logb_)
    X = double(X); U = zeros(size(X));
    for j = 1:numel(xmin_)
        if log_axes_(j)
            U(:,j) = (log10(max(X(:,j),eps)) - loga_(j))./max(logb_(j)-loga_(j),eps);
        else
            U(:,j) = (X(:,j) - xmin_(j))./max(xmax_(j)-xmin_(j),eps);
        end
    end
    U = min(1, max(0, U));
end

function w = jac_logaware(U, xmin_, xmax_, log_axes_, loga_, logb_, dx_du_lin_)
    X = u2x_logaware(U, xmin_, xmax_, log_axes_, loga_, logb_);
    w = ones(size(U,1),1);
    for j = 1:numel(xmin_)
        if log_axes_(j)
            w = w .* (log(10) * (logb_(j)-loga_(j)) .* X(:,j));   % dx/du for log axis
        else
            w = w .* dx_du_lin_(j);                               % constant for linear axis
        end
    end
end

% -------- z-space normalization used by the toy function --------
function Z = normalize01_logaware(X, xmin_, xmax_, log_axes_)
    X = double(X); dloc = numel(xmin_);
    Z = zeros(size(X,1), dloc);
    for j = 1:dloc
        if log_axes_(j)
            a = log10(max(xmin_(j), eps)); b = log10(max(xmax_(j), xmin_(j)+eps));
            Z(:,j) = (log10(max(X(:,j), eps)) - a) / max(b - a, eps);
        else
            Z(:,j) = (X(:,j) - xmin_(j)) / max(xmax_(j) - xmin_(j), eps);
        end
    end
    Z = min(1, max(0, Z));
end
function y = clip01(y), y = min(1, max(0, y)); end
