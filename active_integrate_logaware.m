function out = active_integrate_logaware(funNLL, xmin, xmax, opts)
% ACTIVE_INTEGRATE_LOGAWARE
%   Active GP-based integration of a negative log-likelihood over a box
%   [xmin, xmax] using log-aware coordinate transforms and RQMC.
%
%   This version assumes:
%       funNLL(X) = negative log-likelihood, NLL(X) = -log p(data|X)
%   (natural log). Internally we model a shifted log10-likelihood:
%
%       g(X) = log10 L_rel(X) = -(NLL(X) - N0)/ln(10),
%
%   where N0 is an arbitrary reference NLL (the min over the initial design).
%   The true evidence is reconstructed as:
%
%       I = exp(-N0) * ∫ L_rel(X) dX.
%
%   We report evidence primarily in log-space (log_e and log10) to avoid
%   underflow. Linear I is also returned but may be numerically ~0.
%
%   out = active_integrate_logaware(funNLL, xmin, xmax)
%   out = active_integrate_logaware(funNLL, xmin, xmax, opts)
%
% Inputs:
%   funNLL - function handle @(X) -> NLL(X), with X an N-by-d matrix and
%            NLL(X) an N-by-1 vector of negative log-likelihoods (natural log).
%   xmin   - 1-by-d or d-by-1 vector of lower bounds.
%   xmax   - 1-by-d or d-by-1 vector of upper bounds.
%   opts   - (optional) struct to override defaults:
%       .rngSeed       - random seed (default 42)
%       .maxRounds     - max active rounds (default 40)
%       .maxAddedMult  - maxAdded = maxAddedMult * d (default 160)
%       .tolRelCI      - stop when 95% half-width (on integral) ≤ tolRelCI (default 0.03)
%
% Output struct "out" fields:
%   Evidence (per box [xmin,xmax] with uniform prior):
%       .logI_mean     - mean log-evidence (natural log)
%       .logI_CI95     - 1-by-2 95% CI in log_e space
%       .log10I_mean   - mean log10-evidence
%       .log10I_CI95   - 1-by-2 95% CI in log10 space
%       .I_mean        - evidence in linear space (may underflow to ~0)
%       .CI95          - 1-by-2 95% CI in linear space
%       .sigma_model   - model (GP) contribution to integral std, relative-space
%       .sigma_rqmc    - RQMC standard error contribution, relative-space
%
%   GP / design:
%       .U             - final feature-space design points (N×d)
%       .Y             - final responses g = log10 L_rel
%       .gpr           - final fitrgp model on g
%       .xmin, .xmax   - original bounds
%       .log_axes      - logical mask of log-scaled axes
%       .NLL_ref       - reference N0 used for shifting
%       .toFeat        - handle: X->[U]
%       .fromFeat      - handle: U->[X]
%       .jacobian      - handle: |∂x/∂u| (Jacobian)
%
% Note:
%   The prior is uniform over [xmin,xmax] in the "physical" space X, with
%   optional log10 transforms per axis (log_axes). The evidence is for this
%   uniform prior.

    % ----------------- Basic setup -----------------
    if nargin < 4, opts = struct(); end
    if ~isfield(opts,'rngSeed'),      opts.rngSeed      = 42;   end
    if ~isfield(opts,'maxRounds'),    opts.maxRounds    = 40;   end
    if ~isfield(opts,'maxAddedMult'), opts.maxAddedMult = 160;  end
    if ~isfield(opts,'tolRelCI'),     opts.tolRelCI     = 0.03; end

    xmin = xmin(:)'; xmax = xmax(:)';  % row vectors
    d = numel(xmin);
    rng(opts.rngSeed);

    ln10 = log(10);

    % ----------------- Log-aware map: which axes are log10? -----------------
    % Criterion: axis positive and spans ≥100×
    log_axes = (xmin > 0) & ((xmax ./ max(xmin, eps)) >= 100);

    loga = zeros(1,d); logb = zeros(1,d); dx_du_lin = zeros(1,d);
    for j = 1:d
        if log_axes(j)
            loga(j) = log10(max(xmin(j), eps));
            logb(j) = log10(max(xmax(j), xmin(j)+eps));
        else
            dx_du_lin(j) = xmax(j) - xmin(j);
        end
    end

    % Maps and Jacobian
    fromFeat = @(U) u2x_logaware(U, xmin, xmax, log_axes, loga, logb); % U->[X]
    toFeat   = @(X) x2u_logaware(X, xmin, xmax, log_axes, loga, logb); % X->[U]
    jacobian = @(U) jac_logaware(U, xmin, xmax, log_axes, loga, logb, dx_du_lin);

    % ----------------- Dimension-aware sizes -----------------
    n0 = 10*d^2 + 5;                        % initial design size

    if d==1, Nacq = 16384;
    elseif d==2, Nacq = 65536;
    else,       Nacq = 32768;
    end

    Nint_final   = 65536 * 2^(d-1);         % per-scramble GP quadrature size
    maxRounds    = opts.maxRounds;
    KcapPerRound = 6 + 4*d;                 % batch cap (10,14,18)
    maxAdded     = opts.maxAddedMult * d;
    tolRelCI     = opts.tolRelCI;

    % Small RQMC for stop rule
    if d==1
        N_rqmc_small = 2^15; R_rqmc_small = 16;
    else
        N_rqmc_small = max(8192 * 2^(d-1), floor(Nacq/2));
        R_rqmc_small = 12;
    end

    % ----------------- Initial design (Sobol in U) -----------------
    sob0 = scramble(sobolset(d),'MatousekAffineOwen');
    U    = net(sob0, n0);                     % U∈[0,1]^d
    X0   = fromFeat(U);

    % Evaluate raw negative log-likelihood (natural log)
    NLL0 = funNLL(X0);                        % NLL0 is n0-by-1

    % Reference offset N0 (arbitrary; min over initial design is convenient)
    N0   = min(NLL0);

    % Likelihood
    Y = -(NLL0 - N0) / log(10);
    
    % Acquisition grid (Sobol in U)
    sobA = scramble(sobolset(d),'MatousekAffineOwen');
    Uacq = net(sobA, Nacq);
    wA   = jacobian(Uacq) / Nacq;             % physical weights

    % Final report grid (for 1D plots if d==1)
    sobF       = scramble(sobolset(d),'MatousekAffineOwen');
    Uint_final = net(sobF, Nint_final);
    wF         = jacobian(Uint_final) / Nint_final;

    % ----------------- GP kernel choice -----------------
    if d==1
        kernelName = 'matern52';
    elseif d==2
        kernelName = 'ardsquaredexponential';
    else
        kernelName = 'ardrationalquadratic';
    end
    basisName = 'constant';

    % Initial fit
    gpr = trainGP(U, Y, kernelName, basisName);

    % ----------------- Active loop -----------------
    addedTotal = 0; rounds = 0;

    while rounds < maxRounds && addedTotal < maxAdded
        rounds = rounds + 1;

        % Refit GP each round
        gpr = trainGP(U, Y, kernelName, basisName);

        % Posterior on acquisition grid: g ~ N(muA, sdA^2), where g = log10 L_rel
        [muA, sdA] = predict(gpr, Uacq);
        sdA = max(sdA, 1e-9);
        mu_e = ln10 * muA;      % mean of ln L_rel
        sd_e = ln10 * sdA;      % std  of ln L_rel

        % Log-normal mean/var of L_rel
        LmeanA = exp(mu_e + 0.5*sd_e.^2);
        LvarA  = exp(2*mu_e + sd_e.^2) .* (exp(sd_e.^2) - 1);

        % Var(I_rel) contributions
        A_var = (wA.^2) .* LvarA;

        % Mass concentration → tempering exponent gamma
        kTop = max(1, round(0.01 * Nacq));
        Lw_sorted = sort(LmeanA .* wA, 'descend');
        shareTop  = sum(Lw_sorted(1:kTop)) / max(sum(Lw_sorted), eps);
        C = min(1, max(0, (shareTop - 0.01) / (0.60 - 0.01)));

        switch d
            case 1, gamma_hi = 0.95; gamma_lo = 0.80;
            case 2, gamma_hi = 0.90; gamma_lo = 0.60;
            otherwise, gamma_hi = 0.85; gamma_lo = 0.45;
        end
        gamma = gamma_hi - (gamma_hi - gamma_lo)*C;
        A_var = A_var .^ gamma;
        A_var = A_var / max(A_var + eps);

        % EI and std stabilizers on g = log10 L_rel
        yBest = max(Y);
        Z  = (muA - yBest) ./ sdA;
        EI = (muA - yBest).*normcdf(Z) + sdA.*normpdf(Z);
        EI(sdA<=1e-12) = 0; EI = EI / max(EI + eps);
        A_std = sdA;       A_std = A_std / max(A_std + eps);

        % Auto-weights
        mVar = mean(A_var); mEI = mean(EI); mStd = mean(A_std);
        S = mVar + mEI + mStd + eps;
        aVar = mVar/S; aEI = mEI/S; aStd = mStd/S;

        % Final acquisition function
        A = aVar*A_var + aEI*EI + aStd*A_std;

        % MAP trust-region micro-batch
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

        [~, iMAP] = max(LmeanA);            % MAP in terms of L_rel mean
        u0 = Uacq(iMAP,:);
        localCands = u0;
        [~, ax] = sort(ell_vec, 'descend'); maxAxes = min(d,2);

        % Axial directions
        for j = 1:maxAxes
            a = ax(j);
            u_plus = u0; u_minus = u0;
            u_plus(a)  = min(1, u0(a) + step(a));
            u_minus(a) = max(0, u0(a) - step(a));
            localCands = [localCands; u_plus; u_minus]; %#ok<AGROW>
        end

        % Diagonals in top 2 axes
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

        % Diversified batch for remaining slots
        Krem = max(0, KcapPerRound - qLocal - 1); % leave 1 for global refresh
        if d==3,      M = min(12000, Nacq);
        elseif d==2,  M = min(8000,  Nacq);
        else,         M = min(4000,  Nacq);
        end
        [~, ord] = sort(A, 'descend');
        Ushort = Uacq(ord(1:M),:);
        As = A(ord(1:M));

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

        % Combined new batch
        newU = [localU; newUdiv; newUglob];
        if isempty(newU)
            fprintf('r=%2d: spacing too tight — no new points.\n', rounds);
            break;
        end

        % Deterministic oracle calls (NLL → shifted log10 L_rel)
        X_new   = fromFeat(newU);
        NLL_new = funNLL(X_new);
        newY    = -(NLL_new - N0) / ln10;

        U = [U; newU];
        Y = [Y; newY];
        addedTotal = addedTotal + size(newU,1);

        % Small RQMC for stop rule (relative integral)
        [logIbar_small_rel, se_rqmc_small, Isd_model_small] = ...
            rqmc_integral_mean_and_SE(gpr, N_rqmc_small, R_rqmc_small, jacobian, d);
        
       % Convergence based on RQMC uncertainty
        CV_total = sqrt(Isd_model_small^2 + se_rqmc_small^2);
        relHalf_comb = 1.96 * CV_total;  % 95% half-width as fraction of mean
        
        fprintf('Round %2d: added %2d pts | Uncertainty: %.2f%% | Total pts: %d\n', ...
                rounds, size(newU,1), 100*relHalf_comb, addedTotal);

        if relHalf_comb <= tolRelCI
            fprintf('Stop: combined relative half-width <= %.1f%%\n', 100*tolRelCI);
            break;
        end
    end

    % ----------------- Final RQMC report (relative integral) -----------------
    gpr = trainGP(U, Y, kernelName, basisName);
    
    R_final = (d==1) * 16 + (d>1) * 12;
    [logI_rel_bar, se_rqmc_final, CV_model_final] = ...
        rqmc_integral_mean_and_SE(gpr, Nint_final, R_final, jacobian, d);
    
    % log I_rel from RQMC
    logI_mean = -N0 + logI_rel_bar;   % log Z = -N0 + log I_rel
    
    % Total coefficient of variation (relative uncertainty)
    CV_total = sqrt(CV_model_final^2 + se_rqmc_final^2);
    
    % Delta method: for Y = log(X), Var(Y) H Var(X)/E[X]^2 = CV(X)^2
    if isfinite(CV_total) && CV_total > 0
        se_logI  = CV_total;  % Std error of log(I) H CV(I)
        CI95_log = [logI_mean - 1.96*se_logI, logI_mean + 1.96*se_logI];
    else
        se_logI  = NaN;
        CI95_log = [NaN, NaN];
    end
    
    % Log10 evidence
    log10I_mean = logI_mean / ln10;
    log10I_CI95 = CI95_log / ln10;
    
    % Optional linear evidence (guard against overflow)
    if logI_mean > log(realmax)
        I_mean  = Inf;
        CI95_lin = [NaN, NaN];
    elseif logI_mean < log(realmin)
        I_mean  = 0;
        CI95_lin = [0, 0];
    else
        I_mean  = exp(logI_mean);
        I_sd_abs = I_mean * CV_total;  % abs sigma = mean * CV
        CI95_lin = [I_mean - 1.96*I_sd_abs, I_mean + 1.96*I_sd_abs];
    end

    fprintf('\nAdded %d pts total in %d rounds\n', addedTotal, rounds);
    fprintf(['log10 I = %.6g  (95%% CI: [%.6g, %.6g]) | components: ' ...
             'CV_model=%.4f, CV_rqmc=%.4f\n'], ...
            log10I_mean, log10I_CI95(1), log10I_CI95(2), ...
            CV_model_final, se_rqmc_final);

    % Optional 1D plot: shifted log10-likelihood (relative)
    if d==1
        [muF_plot, sdF_plot] = predict(gpr, Uint_final);
        % center so max(mean) = 0
        muF_plot = muF_plot - max(muF_plot);
        Y_train_center = Y - max(Y);

        xF = fromFeat(Uint_final(:,1));
        [xF, ix] = sort(xF); muF_plot = muF_plot(ix); sdF_plot = sdF_plot(ix);

        fig = figure('Color','w','Units','normalized','Position',[0.18 0.20 0.60 0.55]);
        movegui(fig,'center'); hold on; box on;

        g_lo = muF_plot - 1.96*sdF_plot;
        g_hi = muF_plot + 1.96*sdF_plot;
        x_band = [xF; flipud(xF)];
        y_band = [g_lo; flipud(g_hi)];
        fill(x_band, y_band, [0.9 0.8 1.0], ...
             'EdgeColor','none', 'FaceAlpha',0.4);

        plot(xF, muF_plot, 'b-','LineWidth',2.0);

        theta_train = fromFeat(U(:,1));
        plot(theta_train, Y_train_center, 'ko','MarkerFaceColor','y','MarkerSize',5);

        xlabel('\theta (parameter)','Interpreter','latex');
        ylabel('shifted \log_{10} L_{\mathrm{rel}}(\theta)','Interpreter','latex');
        title('1D Active GP Surrogate for shifted log_{10} Likelihood','Interpreter','latex');
        legend({'95% band','GP median','Train pts'},'Location','best');

        if xmin(1) > 0
            set(gca,'XScale','log'); xlim([xmin(1) xmax(1)]);
            xticks(10.^(floor(log10(xmin(1))):ceil(log10(xmax(1)))));
            ax = gca; ax.XMinorGrid = 'on';
        end
        grid on;
    end

    % ----------------- Pack outputs -----------------
    out = struct();
    % Evidence (log-space preferred)
    out.logI_mean    = logI_mean;
    out.logI_CI95    = CI95_log;
    out.log10I_mean  = log10I_mean;
    out.log10I_CI95  = log10I_CI95;

    % Linear evidence (may underflow)
    out.I_mean       = I_mean;
    out.CI95         = CI95_lin;
    out.sigma_model  = CV_model_final;   % CHANGED: now stores CV (relative)
    out.sigma_rqmc   = se_rqmc_final;     % CHANGED: now stores CV (relative)

    % GP / design
    out.U            = U;
    out.Y            = Y;
    out.gpr          = gpr;
    out.xmin         = xmin;
    out.xmax         = xmax;
    out.log_axes     = log_axes;
    out.NLL_ref      = N0;
    out.toFeat       = toFeat;
    out.fromFeat     = fromFeat;
    out.jacobian     = jacobian;
    
    fprintf('NLL Statistics:\n');
    fprintf('  Min NLL0: %.6e\n', min(NLL0));
    fprintf('  Max NLL0: %.6e\n', max(NLL0));
    fprintf('  Median NLL0 (N0, min): %.6e\n', N0);
    fprintf('  Range: %.6e\n', max(NLL0) - min(NLL0));
    fprintf('  Shifted Y range: [%.6e, %.6e]\n', min(Y), max(Y));
end

% ====================== LOCAL FUNCTIONS ======================

function mdl = trainGP(Uin, Yin, kernelName, basisName)
    mdl = fitrgp(Uin, Yin, ...
        'KernelFunction', kernelName, ...
        'BasisFunction',  basisName, ...
        'Standardize',    true);
end

function [logI_est, I_sd, Lmean, Lvar] = int_stats_log10(mu10, sd10, w)
% INT_STATS_LOG10 - Log-space stable version without artificial clamping

    ln10  = log(10);
    mu_e  = ln10*mu10;   % mu_e = ln(L_rel)
    sd_e  = ln10*sd10;
    
    % For uncertainty estimation - NO CLAMPING
    Lmean = exp(mu_e + 0.5*sd_e.^2);
    Lvar  = exp(2*mu_e + sd_e.^2) .* (exp(sd_e.^2) - 1);
    I_sd  = sqrt(sum(w.^2 .* Lvar));
    
    % Compute integral in log-space
    log_w = log(max(w, realmin));
    log_integrand = log_w + mu_e + 0.5*sd_e.^2;
    
    % Stable logsumexp
    max_log = max(log_integrand);
    if ~isfinite(max_log)
        logI_est = -inf;
    else
        logI_est = max_log + log(sum(exp(log_integrand - max_log)));
    end
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
    Us = Ucand .* scale;
    Es = Uexist .* scale;

    if ~isempty(Es)
        try
            distE = min(pdist2(Us, Es, 'euclidean'), [], 2);
        catch
            distE = inf(N,1);
            for j=1:size(Es,1)
                distE = min(distE, sqrt(sum((Us-Es(j,:)).^2,2)));
            end
        end
    else
        distE = inf(N,1);
    end

    valid = find(distE > minSep);
    if isempty(valid), newU = []; return; end

    [~, k] = max(score(valid));
    i0 = valid(k);
    sel = i0;
    newU = Ucand(i0,:);

    while size(newU,1) < Kmax
        if numel(sel)==1
            distS = sqrt(sum((Us-Us(sel,:)).^2,2));
        else
            try
                distS = min(pdist2(Us, Us(sel,:), 'euclidean'), [], 2);
            catch
                distS = inf(N,1);
                for j=1:numel(sel)
                    distS = min(distS, sqrt(sum((Us-Us(sel(j),:)).^2,2)));
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
    du = U - u;
    d  = sqrt(sum(du.^2, 2));
end

function [logI_bar, se_rqmc, sd_model_bar] = rqmc_integral_mean_and_SE(gpr, N, R, jacobian_fn, d)
% RQMC_INTEGRAL_MEAN_AND_SE - Pure log-space uncertainty computation

    logIvals = zeros(R,1); 
    logCV_model = zeros(R,1);  % Store log(CV) for each scramble
    sob = scramble(sobolset(d),'MatousekAffineOwen');
    ln10 = log(10);
    
    for r = 1:R
        U0 = net(sob, N);
        shift = rand(1,d);
        U = mod(U0 + shift, 1);
        w = jacobian_fn(U) / N;
        [mu, sd] = predict(gpr, U);
        
        % Compute log-integral
        mu_e = ln10 * mu;
        sd_e = ln10 * sd;
        
        log_w = log(w);
        log_integrand = log_w + mu_e + 0.5*sd_e.^2;
        max_log = max(log_integrand);
        logI_r = max_log + log(sum(exp(log_integrand - max_log)));
        logIvals(r) = logI_r;
        
        % ========== Compute CV in log-space ==========
        % CV^2 = Var(I) / E[I]^2
        % log(CV^2) = log(Var(I)) - 2*log(E[I])
        % log(CV) = 0.5 * [log(Var(I)) - 2*log(E[I])]
        
        % E[I] in log-space (already computed above)
        logI_mean = logI_r;
        
        % Var(I) = sum(w_i^2 * Var(L_i))
        % In log-space:
        log_w2 = 2*log_w;  % log(w^2)
        log_varL = 2*mu_e + sd_e.^2 + log(exp(sd_e.^2) - 1);  % log(Var(L_i))
        log_varContrib = log_w2 + log_varL;  % log(w_i^2 * Var(L_i))
        
        % Sum variance contributions in log-space
        max_logvar = max(log_varContrib);
        if isfinite(max_logvar)
            logI_var = max_logvar + log(sum(exp(log_varContrib - max_logvar)));
            
            % log(CV) = 0.5 * [log(Var) - 2*log(Mean)]
            logCV_model(r) = 0.5 * (logI_var - 2*logI_mean);
        else
            logCV_model(r) = -inf;  % Zero variance
        end
    end
    
    % Average in log-space
    max_logI = max(logIvals);
    if ~isfinite(max_logI)
        logI_bar = -inf;
        se_rqmc = NaN;
        sd_model_bar = NaN;
        return;
    end
    
    logI_bar = max_logI + log(mean(exp(logIvals - max_logI)));
    
    % RQMC uncertainty (std of log values = CV)
    se_rqmc = std(logIvals, 0) / sqrt(R);
    
    % Model uncertainty: average CV across scrambles
    % Average log(CV) values in log-space, then exponentiate
    finite_logCV = logCV_model(isfinite(logCV_model));
    if ~isempty(finite_logCV)
        max_logCV = max(finite_logCV);
        mean_logCV = max_logCV + log(mean(exp(finite_logCV - max_logCV)));
        sd_model_bar = exp(mean_logCV);
    else
        sd_model_bar = 0;
    end
    
    if ~isfinite(sd_model_bar) || sd_model_bar < 0
        sd_model_bar = 0;
    end
end

% -------- log-aware maps --------
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
            U(:,j) = (log10(max(X(:,j),eps)) - loga_(j)) ./ max(logb_(j)-loga_(j), eps);
        else
            U(:,j) = (X(:,j) - xmin_(j)) ./ max(xmax_(j)-xmin_(j), eps);
        end
    end
    U = min(1, max(0, U));
end

function w = jac_logaware(U, xmin_, xmax_, log_axes_, loga_, logb_, dx_du_lin_)
    X = u2x_logaware(U, xmin_, xmax_, log_axes_, loga_, logb_);
    w = ones(size(U,1),1);
    for j = 1:numel(xmin_)
        if log_axes_(j)
            w = w .* (log(10) * (logb_(j)-loga_(j)) .* X(:,j));
        else
            w = w .* dx_du_lin_(j);
        end
    end
end
