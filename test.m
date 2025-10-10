% Active integration on log10(L) with dimension-aware kernels,
% Var(I)-driven acquisition (tempered by mass concentration),
% local MAP trust-region + diversified batch + tiny global refresh,
% and robust RQMC error bars (independent digital shifts).
% Deterministic evaluations (no observation noise).
clear; clc; close all;

%% ================= USER CHOICE =================
d = 1;                       % 1, 2, or 3
rng(42);

%% ===== Domain & truth: heterogeneous input ranges, output 1e2..1e5 (dimension-agnostic) =====
% Set your bounds per-dimension (length-d). Example for d=1:
xmin = [1e-4]; xmax = [1];
% Example for d=2 with mixed decades: xmin = [1e-4, 1e2]; xmax = [1, 1e5];
xmin = xmin(:)';  xmax = xmax(:)';
d = numel(xmin);

toUnit   = @(X) (X - xmin) ./ (xmax - xmin);      % linear to [0,1]^d
fromUnit = @(U) xmin + U .* (xmax - xmin);        % back to physical domain

% Decide per-axis whether to use log10 when shaping the test function:
log_axes = (xmin > 0) & ((xmax./max(xmin,eps)) >= 100);

% Map X -> Z in [0,1]^d using linear OR log10 scaling per axis
zmap = @(X) normalize01_logaware(X, xmin, xmax, log_axes);

% Build a u(X)∈[0,1]: smooth wave across dims + anisotropic bump in Z-space
mu_z  = linspace(0.35, 0.65, d);                 % bump center per dim
sig_z = 0.12 * (0.85 .^ (0:d-1));                % widths per dim (optional taper)
quadZ = @(Z) sum(((Z - mu_z).^2) ./ (2*sig_z.^2), 2);

u01   = @(X) clip01( 0.45*(1 + sin(2*pi*sum(zmap(X),2))) ...
                   + 0.55*exp(-quadZ(zmap(X))) );

% log10 L ∈ [2,5] ⇒ L ∈ [1e2, 1e5]
gtrue10_nd = @(X) 2 + 3*u01(X);
ftrue_nd   = @(X) 10.^gtrue10_nd(X);

% No closed form integral here
I_true_analytic = NaN;

% (optional) quick sanity check of realized output range on a coarse grid:
sobChk = scramble(sobolset(d),'MatousekAffineOwen');
Uchk   = net(sobChk, 8192);
Xchk   = fromUnit(Uchk);
vals   = ftrue_nd(Xchk);
fprintf('Observed L range on coarse grid: [%.2g, %.2g] (target [1e2, 1e5])\n', min(vals), max(vals));

%% ================= DIMENSION-AWARE SIZES =================
n0 = 10*d^2 + 5;                       % 15, 45, 95 for d=1,2,3

% Candidate grid (predictions-only; cheap)
if d==1
    Nacq = 16384;
elseif d==2
    Nacq = 65536;
else
    Nacq = 32768;
end

% Final GP-only quadrature (per scramble)
Nint_final   = 65536 * 2^(d-1);
maxRounds    = 40;
KcapPerRound = 6 + 4*d;                % 10, 14, 18
maxAdded     = 160*d;
tolRelCI     = 0.03;                   % stop when 95% half-width ≤ 3%

% Cheap per-round RQMC (predictions only)
if d==1
    N_rqmc_small = 2^15;   % 32768
    R_rqmc_small = 8;
else
    N_rqmc_small = max(8192 * 2^(d-1), floor(Nacq/2));
    R_rqmc_small = 6;
end

%% ================= INITIAL DATA (Sobol) =================
sob0 = sobolset(d,'Skip',1e3,'Leap',1e2); sob0 = scramble(sob0,'MatousekAffineOwen');
U = net(sob0, n0);                      % [0,1]^d
X = fromUnit(U);
Y = gtrue10_nd(X);                      % deterministic evaluations

%% ========== ACQ & FINAL GRIDS (Sobol, equal weights) ==========
vol = prod(xmax - xmin);

sobA = sobolset(d,'Skip',2e3,'Leap',2e2); sobA = scramble(sobA,'MatousekAffineOwen');
Uacq = net(sobA, Nacq);
wA   = (vol / Nacq) * ones(Nacq,1);

sobF = sobolset(d,'Skip',5e3,'Leap',5e2); sobF = scramble(sobF,'MatousekAffineOwen');
Uint_final = net(sobF, Nint_final);
wF         = (vol / Nint_final) * ones(Nint_final,1);

%% ================= GP HELPERS (automatic hyperparams) =================
% Kernels: 1D Matern-5/2; 2D ARD SE; 3D ARD Rational Quadratic
if d==1
    kernelName = 'matern52';
elseif d==2
    kernelName = 'ardsquaredexponential';
else
    kernelName = 'ardrationalquadratic';
end
basisName = 'constant';
ln10 = log(10);

% Initial fit (fitrgp optimizes kernel params & sigma)
gpr = trainGP(U, Y, kernelName, basisName);

%% ================= ACTIVE LOOP =================
addedTotal = 0; rounds = 0;

while rounds < maxRounds && addedTotal < maxAdded
    rounds = rounds + 1;

    % Re-optimize each round
    gpr = trainGP(U, Y, kernelName, basisName);

    % Posterior on acquisition grid
    [muA, sdA] = predict(gpr, Uacq);
    sdA = max(sdA, 1e-9);
    mu_e = ln10*muA; sd_e = ln10*sdA;

    % Primary: absolute contribution to Var(I)
    LmeanA = exp(mu_e + 0.5*sd_e.^2);
    LvarA  = exp(2*mu_e + sd_e.^2) .* (exp(sd_e.^2) - 1);
    A_var  = (wA.^2) .* LvarA;

    % Mass concentration metric → tempering exponent gamma
    kTop = max(1, round(0.01 * Nacq));
    [Lw_sorted, ~] = sort(LmeanA .* wA, 'descend');
    shareTop = sum(Lw_sorted(1:kTop)) / max(sum(Lw_sorted), eps); % (0,1)
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

    % Final acquisition score
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
        for s1 = [-1,1]
            for s2 = [-1,1]
                u_d = u0;
                u_d(a1) = min(1, max(0, u0(a1) + s1*0.7*step(a1)));
                u_d(a2) = min(1, max(0, u0(a2) + s2*0.7*step(a2)));
                localCands = [localCands; u_d]; %#ok<AGROW>
            end
        end
    end
    keep = true(size(localCands,1),1);
    for i=1:size(localCands,1)
        keep(i) = isempty(U) || all(row_dist(localCands(i,:), U) > 0.02*median(ell_vec));
    end
    localU = unique(localCands(keep,:), 'rows', 'stable');
    qLocal = min(qLocal, size(localU,1));
    localU = localU(1:qLocal,:);

    % Diversified batch for remaining slots (ARD-aware distances)
    Krem = max(0, KcapPerRound - qLocal - 1); % reserve some for global refresh
    if d==3, M = min(12000, Nacq);
    elseif d==2, M = min(8000, Nacq);
    else, M = min(4000, Nacq); end
    [~, ord] = sort(A, 'descend');
    Ushort = Uacq(ord(1:M),:); As = A(ord(1:M));
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

    % Deterministic evaluations
    newX = fromUnit(newU);
    newY = gtrue10_nd(newX);

    U = [U; newU];  Y = [Y; newY];
    addedTotal = addedTotal + size(newU,1);

    % Small RQMC for a reliable stop rule (independent shifts)
    [Ibar_small, se_rqmc_small, Isd_model_small] = ...
        rqmc_integral_mean_and_SE(gpr, N_rqmc_small, R_rqmc_small, xmin, xmax);

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
% Final refit with all data
gpr = trainGP(U, Y, kernelName, basisName);

% Final R with independent digital shifts
R_final = (d==1) * 16 + (d>1) * 12;
[I_bar, se_rqmc_final, Isd_model_final] = ...
    rqmc_integral_mean_and_SE(gpr, Nint_final, R_final, xmin, xmax);

sigma_total = sqrt(Isd_model_final^2 + se_rqmc_final^2);
CI95 = [I_bar - 1.96*sigma_total, I_bar + 1.96*sigma_total];

absErr = abs(I_true_analytic - I_bar);
relErr = absErr / max(I_true_analytic, eps);

fprintf('\nAdded %d pts total in %d rounds\n', addedTotal, rounds);
fprintf('I ≈ %.6f  (95%% CI: [%.6f, %.6f]) | components: model sd=%.3g, RQMC se=%.3g | toy rel err = %.2f%%\n', ...
        I_bar, CI95(1), CI95(2), Isd_model_final, se_rqmc_final, 100*relErr);

% Deterministic big-net cross-check (optional; GP-only)
N_det = max(2^17, Nint_final);
sobChk = sobolset(d); sobChk = scramble(sobChk,'MatousekAffineOwen');
U_det  = net(sobChk, N_det);
w_det  = (prod(xmax - xmin) / N_det) * ones(N_det,1);
[mu_det, sd_det] = predict(gpr, U_det);
[I_det, ~] = int_stats_log10(mu_det, sd_det, w_det);
fprintf('Deterministic large-net check: I_det = %.6f (diff to RQMC mean = %.3g)\n', I_det, I_det - I_bar);

% 1D plots
% 1D plots
if d==1
    [muF_plot, sdF_plot] = predict(gpr, Uint_final);
    [~,~,~, Lvar_plot] = int_stats_log10(muF_plot, sdF_plot, wF);
    xF = fromUnit(Uint_final(:,1));
    [xF, ix] = sort(xF); muF_plot = muF_plot(ix); sdF_plot = sdF_plot(ix); Lvar_plot = Lvar_plot(ix);

    fig = figure('Color','w','Units','normalized','Position',[0.18 0.08 0.64 0.82]); movegui(fig,'center');

    subplot(2,1,1); hold on; box on;
    plot(xF, ftrue_nd([xF]), 'k--','LineWidth',1.5);
    plot(xF, 10.^muF_plot,  'r-','LineWidth',2.0);
    fill([xF; flipud(xF)], [10.^(muF_plot-1.96*sdF_plot); flipud(10.^(muF_plot+1.96*sdF_plot))], ...
         [1 0.8 0.8], 'EdgeColor','none', 'FaceAlpha',0.35);
    plot(fromUnit(U), 10.^Y, 'ko','MarkerFaceColor','y');
    legend('True L(x)','Final median','Final 95% band','Train pts','Location','best');
    xlabel('x'); ylabel('L(x)'); title(sprintf('Active Integration (d=1) | +%d pts', addedTotal));

    % ===== NEW: log-scale x-axis (only if x>0) =====
    if xmin(1) > 0
        set(gca,'XScale','log');
        xlim([xmin(1) xmax(1)]);
        pmin = floor(log10(xmin(1))); pmax = ceil(log10(xmax(1)));
        xticks(10.^(pmin:pmax));
        ax = gca; ax.XMinorGrid = 'on'; grid on;
    end
    % ==============================================

    subplot(2,1,2); hold on; box on;
    plot(xF, sqrt(max(Lvar_plot,0)),'r-','LineWidth',2.0);
    legend('Final \sigma_L(x)','Location','best');
    xlabel('x'); ylabel('\sigma_{L}(x)'); title('Pointwise uncertainty');

    % ===== NEW: log-scale x-axis for bottom panel too =====
    if xmin(1) > 0
        set(gca,'XScale','log');
        xlim([xmin(1) xmax(1)]);
        pmin = floor(log10(xmin(1))); pmax = ceil(log10(xmax(1)));
        xticks(10.^(pmin:pmax));
        ax = gca; ax.XMinorGrid = 'on'; grid on;
    end
    % ======================================================
end
% if d==1
%     [muF_plot, sdF_plot] = predict(gpr, Uint_final);
%     [~,~,~, Lvar_plot] = int_stats_log10(muF_plot, sdF_plot, wF);
%     xF = fromUnit(Uint_final(:,1));
%     [xF, ix] = sort(xF); muF_plot = muF_plot(ix); sdF_plot = sdF_plot(ix); Lvar_plot = Lvar_plot(ix);
%     fig = figure('Color','w','Units','normalized','Position',[0.18 0.08 0.64 0.82]); movegui(fig,'center');
% 
%     subplot(2,1,1); hold on; box on;
%     plot(xF, ftrue_nd([xF]), 'k--','LineWidth',1.5);
%     plot(xF, 10.^muF_plot,  'r-','LineWidth',2.0);
%     fill([xF; flipud(xF)], [10.^(muF_plot-1.96*sdF_plot); flipud(10.^(muF_plot+1.96*sdF_plot))], ...
%         [1 0.8 0.8], 'EdgeColor','none', 'FaceAlpha',0.35);
%     plot(fromUnit(U), 10.^Y, 'ko','MarkerFaceColor','y');
%     legend('True L(x)','Final median','Final 95% band','Train pts','Location','best');
%     xlabel('x'); ylabel('L(x)'); title(sprintf('Active Integration (d=1) | +%d pts', addedTotal));
% 
%     subplot(2,1,2); hold on; box on;
%     plot(xF, sqrt(max(Lvar_plot,0)),'r-','LineWidth',2.0);
%     legend('Final \sigma_L(x)','Location','best');
%     xlabel('x'); ylabel('\sigma_{L}(x)'); title('Pointwise uncertainty');
% end

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
    isArd = strncmpi(gpr.KernelFunction,'ard',3);
    if isArd
        ell_vec = KP(1:d)';      % ARD length-scales
        sigmaF  = KP(end);
    else
        ell_vec = KP(1)*ones(1,d);
        sigmaF  = KP(end);
    end
end

function newU = fps_diversify(Ucand, score, Uexist, minSep, Kmax, ell_vec)
    N = size(Ucand,1);
    if N==0 || Kmax<=0, newU = []; return; end
    scale = 1 ./ max(ell_vec(:).', 1e-6);
    Us = Ucand .* scale;
    Es = Uexist .* scale;

    if ~isempty(Es)
        try
            distE = min(pdist2(Us, Es, 'euclidean'), [], 2);
        catch
            distE = inf(N,1);
            for j = 1:size(Es,1)
                dj = sqrt(sum((Us - Es(j,:)).^2, 2));
                distE = min(distE, dj);
            end
        end
    else
        distE = inf(N,1);
    end

    valid = find(distE > minSep);
    if isempty(valid), newU = []; return; end

    [~, k] = max(score(valid)); i0 = valid(k);
    sel = i0; newU = Ucand(i0,:);

    while size(newU,1) < Kmax
        if numel(sel)==1
            distS = sqrt(sum((Us - Us(sel,:)).^2, 2));
        else
            try
                distS = min(pdist2(Us, Us(sel,:), 'euclidean'), [], 2);
            catch
                distS = inf(N,1);
                for j = 1:numel(sel)
                    dj = sqrt(sum((Us - Us(sel(j),:)).^2, 2));
                    distS = min(distS, dj);
                end
            end
        end
        dmin = min(distE, distS);
        candMask = dmin > minSep;
        if ~any(candMask), break; end

        util = score .* dmin;
        util(~candMask) = -inf;
        [best, ib] = max(util);
        if ~isfinite(best), break; end

        sel(end+1,1) = ib; %#ok<AGROW>
        newU = [newU; Ucand(ib,:)]; %#ok<AGROW>
    end
end

function d = row_dist(u, U)
    du = U - u; d = sqrt(sum(du.^2, 2));
end

function [I_bar, se_rqmc, I_sd_model_bar] = rqmc_integral_mean_and_SE(gpr, N, R, xmin, xmax)
% RQMC mean & SE using Owen-scrambled Sobol with an independent random
% digital shift per scramble (Cranley–Patterson). This makes scrambles i.i.d.
    d   = numel(xmin);
    vol = prod(xmax - xmin);

    Ivals = zeros(R,1);
    Isd2  = zeros(R,1);

    sob = sobolset(d);
    sob = scramble(sob,'MatousekAffineOwen');

    for r = 1:R
        U0 = net(sob, N);                 % base net
        shift = rand(1,d);                % independent random shift
        U = mod(U0 + shift, 1);           % i.i.d. scrambled set in [0,1]^d
        w = (vol / N) * ones(N,1);

        [mu, sd] = predict(gpr, U);
        [I_r, I_sd_r] = int_stats_log10(mu, sd, w);

        Ivals(r) = I_r;
        Isd2(r)  = I_sd_r.^2;
    end

    I_bar          = mean(Ivals);
    se_rqmc        = std(Ivals, 0) / sqrt(R);
    I_sd_model_bar = sqrt(mean(Isd2));
end
% -------- local helpers (dimension-agnostic) --------
function Z = normalize01_logaware(X, xmin_, xmax_, log_axes_)
    X = double(X);
    dloc = numel(xmin_);
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
