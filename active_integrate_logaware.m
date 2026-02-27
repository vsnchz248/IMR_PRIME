function out = active_integrate_logaware(funNLL, xmin, xmax, opts)
% ACTIVE_INTEGRATE_LOGAWARE
%   Active GP-based integration with PROPER beta marginalization per BIMR
%
% CORRECTED VERSION: Matches BIMR paper exactly
%   1. For each θ: Marginalize β first -> P(D|M,θ)
%   2. Train GP on -log P(D|M,θ)  
%   3. Integrate: P(D|M) = ∫ P(D|M,θ) P(θ|M) dθ

    % ----------------- Basic setup -----------------
    if nargin < 4, opts = struct(); end
    
    if ~isfield(opts,'rngSeed'),      opts.rngSeed      = 42;   end
    if ~isfield(opts,'maxRounds'),    opts.maxRounds    = 40;   end
    if ~isfield(opts,'maxAddedMult'), opts.maxAddedMult = 160;  end
    if ~isfield(opts,'tolRelCI'),     opts.tolRelCI     = 0.03; end
    if ~isfield(opts,'verbose'),      opts.verbose      = true; end

    % Extract priors and model info from opts
    if isfield(opts, 'priors') && isfield(opts, 'modelName')
        priors = opts.priors;
        modelName = opts.modelName;
        have_priors = true;
    else
        priors = struct();
        modelName = 'unknown';
        have_priors = false;
        warning('active_integrate_logaware:noPriors', ...
                'No priors in opts - using uniform prior (may give incorrect evidence)');
    end

    xmin = xmin(:)'; xmax = xmax(:)';
    d = numel(xmin);
    rng(opts.rngSeed);

    ln10 = log(10);

    % ----------------- Beta grid for marginalization (BIMR Eq. 15-17) -----------------
    beta_min = 0.05;
    beta_max = 10.0;
    N_beta = 100;
    beta_grid = linspace(beta_min, beta_max, N_beta);
    
    % Half-Cauchy prior P(β) = (2/π) × 1/(1+β²)
    P_beta = (2/pi) ./ (1 + beta_grid.^2);
    P_beta = P_beta / sum(P_beta);  % Normalize for discrete approximation

    % ----------------- Log-aware map -----------------
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

    fromFeat = @(U) u2x_logaware(U, xmin, xmax, log_axes, loga, logb);
    toFeat   = @(X) x2u_logaware(X, xmin, xmax, log_axes, loga, logb);
    jacobian = @(U) jac_logaware(U, xmin, xmax, log_axes, loga, logb, dx_du_lin);

    % ----------------- Dimension-aware sizes -----------------
    n0 = 10*d^2 + 5;

    if d==1, Nacq = 16384;
    elseif d==2, Nacq = 65536;
    else,       Nacq = 32768;
    end

    if isfield(opts, 'Nint_final_override')
        Nint_final = opts.Nint_final_override;
    else
        Nint_final = 65536 * 2^(d-1);
    end    
    maxRounds    = opts.maxRounds;
    KcapPerRound = 6 + 4*d;
    maxAdded     = opts.maxAddedMult * d;

    fprintf('\n=== Starting Active Learning for %s (%dD) ===\n', upper(modelName), d);
    % fprintf('Initial design: %d points\n', n0);
    % fprintf('Max rounds: %d, Convergence tolerance: %.1f%%\n\n', maxRounds, 100*opts.tolRelCI);

    % ----------------- Initial design (Sobol in U) -----------------
    sob0 = scramble(sobolset(d),'MatousekAffineOwen');
    U    = net(sob0, n0);
    X0   = fromFeat(U);

    % Evaluate NLL for ALL beta values
    % fprintf('Evaluating initial design with beta marginalization...\n');
    NLL_beta = funNLL(X0, beta_grid);  % Returns [n0 × N_beta] matrix
    
    % Marginalize beta for each theta (BIMR Eq. 17)
    % P(D|M,θ) = Σ_b P(D|M,θ,β_b) P(β_b)
    % In log space: log P(D|M,θ) = logsumexp_b[-NLL(θ,β_b) + log P(β_b)]
    log_P_D_given_theta = zeros(n0, 1);
    best_beta_idx = zeros(n0, 1);
    for i = 1:n0
        log_terms = -NLL_beta(i,:) + log(P_beta);
        [~, best_beta_idx(i)] = max(log_terms);
        max_log = max(log_terms);
        log_P_D_given_theta(i) = max_log + log(sum(exp(log_terms - max_log)));
    end
    
    % Target for GP: -log10 P(D|M,θ) with offset for numerical stability
    N0 = max(log_P_D_given_theta);  % Most likely theta (MAXIMUM log probability)
    Y = -(log_P_D_given_theta - N0) / ln10;
    Y_max = min(quantile(Y(isfinite(Y) & Y >= 0), 0.95),50);
    Y = min(Y, Y_max);
    % Acquisition grid
    sobA = scramble(sobolset(d),'MatousekAffineOwen');
    Uacq = net(sobA, Nacq);
    wA   = jacobian(Uacq) / Nacq;

    % Final integration grid
    sobF       = scramble(sobolset(d),'MatousekAffineOwen');
    Uint_final = net(sobF, Nint_final);

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
    fprintf('Fitting initial GP...\n');
    gpr = trainGP(U, Y, kernelName, basisName);

    % ----------------- Active loop -----------------
    addedTotal = 0; rounds = 0;

    while rounds < maxRounds && addedTotal < maxAdded
        rounds = rounds + 1;

        % Refit GP each round
        gpr = trainGP(U, Y, kernelName, basisName);

        % Posterior on acquisition grid
        [muA, sdA] = predict(gpr, Uacq);
        sdA = max(sdA, 1e-9);

        % Expected integrand variance
        A_var = (wA.^2) .* (sdA.^2);

        % Mass concentration for tempering
        kTop = max(1, round(0.01 * Nacq));
        LmeanA = exp(-ln10 * muA);  % Convert back to P(D|M,θ) scale
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

        % EI and std stabilizers
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

        [~, iMAP] = min(muA);  % min of -log10 P = max of P
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

        % Diagonals
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

        % Diversified batch
        Krem = max(0, KcapPerRound - qLocal - 1);
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

        % Global refresh
        Kglob = max(0, KcapPerRound - size(localU,1) - size(newUdiv,1));
        newUglob = [];
        if Kglob > 0
            onescore = ones(size(Ushort,1),1);
            newUglob = fps_diversify(Ushort, onescore, U, 0.10*median(ell_vec), Kglob, ell_vec);
        end

        % Combined new batch
        newU = [localU; newUdiv; newUglob];
        if isempty(newU)
            fprintf('Round %2d: spacing too tight - no new points.\n', rounds);
            break;
        end

        % Evaluate and marginalize beta
        X_new = fromFeat(newU);
        NLL_beta_new = funNLL(X_new, beta_grid);
        
        log_P_D_given_theta_new = zeros(size(newU,1), 1);
        for i = 1:size(newU,1)
            log_terms = -NLL_beta_new(i,:) + log(P_beta);
            max_log = max(log_terms);
            log_P_D_given_theta_new(i) = max_log + log(sum(exp(log_terms - max_log)));
        end
        
        newY = -(log_P_D_given_theta_new - N0) / ln10;
        newY = min(newY, Y_max);
        U = [U; newU];
        Y = [Y; newY];
        addedTotal = addedTotal + size(newU,1);
        
        % DEBUG: Track Y statistics
        % if mod(rounds, 5) == 0
        %     fprintf('  [DEBUG] Y stats after round %d: min=%.2f, max=%.2f, mean=%.2f, std=%.2f\n', ...
        %         rounds, min(Y), max(Y), mean(Y), std(Y));
        % end

        % Convergence
        Y_range = Y_max - min(Y);
        if Y_range < eps, Y_range = 1; end
        
        max_sd = max(sdA);
        mean_sd = mean(sdA);
        max_sd_rel = max_sd / Y_range;
        mean_sd_rel = mean_sd / Y_range;
        
        fprintf('Round %2d: added %2d pts | Max GP std: %.2f%%, Mean GP std: %.2f%% | Total pts: %d\n', ...
                rounds, size(newU,1), 100*max_sd_rel, 100*mean_sd_rel, size(U,1));

        if max_sd_rel < 0.05
            fprintf('Stop: GP sufficiently accurate (max_sd < 5%% of Y range)\n');
            break;
        end
    end

    % ----------------- Final GP fit -----------------
    fprintf('\nFitting final GP with %d total points...\n', size(U,1));
    gpr = trainGP(U, Y, kernelName, basisName);

    % Final integration with high accuracy
   if isfield(opts, 'R_final_override')
       R_final = opts.R_final_override;
   elseif d==1, R_final = 32;
   elseif d==2, R_final = 16;
   else, R_final = 12;
   end
    
    fprintf('Computing final RQMC integral (N=%d, R=%d)...\n', Nint_final, R_final);
    fprintf('DEBUG N0=%.4e, min(Y)=%.2f, max(Y)=%.2f\n', N0, min(Y), max(Y));
    % Integrate with prior: ∫ P(D|M,θ) P(θ|M) dθ
    if have_priors
        [logI_rel_bar, se_rqmc_final, CV_model_final] = ...
            rqmc_integral_with_prior(gpr, Nint_final, R_final, jacobian, d, ...
                                     modelName, priors, fromFeat,Y_max);
    else
        [logI_rel_bar, se_rqmc_final, CV_model_final] = ...
            rqmc_integral_mean_and_SE(gpr, Nint_final, R_final, jacobian, d);
    end
    
    % True log evidence: log P(D|M) = N0 + logI_rel
    logI_mean = N0 + logI_rel_bar;
    
    % if opts.verbose
    %     fprintf('\n=== Evidence Calculation ===\n');
    %     fprintf('N0 (reference): %.6e\n', N0);
    %     fprintf('logI_rel: %.6e\n', logI_rel_bar);
    %     fprintf('logI_mean = N0 + logI_rel = %.6e\n', logI_mean);
    %     fprintf('log10(evidence) = %.6f\n', logI_mean / ln10);
    %     fprintf('===========================\n\n');
    % end
    
    % Total coefficient of variation
    CV_total = sqrt(CV_model_final^2 + se_rqmc_final^2);
    
    if isfinite(CV_total) && CV_total > 0
        se_logI  = CV_total;
        CI95_log = [logI_mean - 1.96*se_logI, logI_mean + 1.96*se_logI];
    else
        se_logI  = NaN;
        CI95_log = [NaN, NaN];
    end
    
    % Log10 evidence
    log10I_mean = logI_mean / ln10;
    log10I_CI95 = CI95_log / ln10;
    
    % Optional linear evidence
    if logI_mean > log(realmax)
        I_mean  = Inf;
        CI95_lin = [NaN, NaN];
    elseif logI_mean < log(realmin)
        I_mean  = 0;
        CI95_lin = [0, 0];
    else
        I_mean  = exp(logI_mean);
        I_sd_abs = I_mean * CV_total;
        CI95_lin = [I_mean - 1.96*I_sd_abs, I_mean + 1.96*I_sd_abs];
    end

    fprintf('\n=== FINAL RESULTS for %s ===\n', upper(modelName));
    fprintf('Total points: %d (added %d in %d rounds)\n', size(U,1), addedTotal, rounds);
    fprintf('log10(Z) = %.6g  (95%% CI: [%.6g, %.6g])\n', ...
            log10I_mean, log10I_CI95(1), log10I_CI95(2));
    
    % Find and report MAP
    [~, mapIdx] = min(Y);  % min of -log10 P = max of P
    out.mapIdx = mapIdx;
    mapTheta = fromFeat(U(mapIdx,:));
    fprintf('MAP parameters: ');
    for j = 1:numel(mapTheta)
        fprintf('%.4g ', mapTheta(j));
    end
    fprintf('\n');
    fprintf('================================\n\n');

    % Optional 1D plot
    if d==1 && opts.verbose
        [muF_plot, sdF_plot] = predict(gpr, Uint_final);
        muF_plot = muF_plot - min(muF_plot);  % Center around MAP
        Y_train_center = Y - min(Y);

        xF = fromFeat(Uint_final(:,1));
        [xF, ix] = sort(xF); muF_plot = muF_plot(ix); sdF_plot = sdF_plot(ix);

        figure('Color','w','Units','normalized','Position',[0.18 0.20 0.60 0.55]);
        hold on; box on;

        g_lo = muF_plot - 1.96*sdF_plot;
        g_hi = muF_plot + 1.96*sdF_plot;
        fill([xF; flipud(xF)], [g_lo; flipud(g_hi)], [0.9 0.8 1.0], ...
             'EdgeColor','none', 'FaceAlpha',0.4);

        plot(xF, muF_plot, 'b-','LineWidth',2.0);

        theta_train = fromFeat(U(:,1));
        plot(theta_train, Y_train_center, 'ko','MarkerFaceColor','y','MarkerSize',5);

        xlabel('Parameter');
        ylabel('-log10 P(D|M,\theta) (shifted)');
        title(sprintf('1D GP Surrogate - %s', upper(modelName)));
        legend({'95% band','GP mean','Training pts'},'Location','best');

        if xmin(1) > 0
            set(gca,'XScale','log'); xlim([xmin(1) xmax(1)]);
        end
        grid on;
    end

    % ----------------- Pack outputs -----------------
    out.logI_mean    = logI_mean;
    out.logI_CI95    = CI95_log;
    out.log10I_mean  = log10I_mean;
    out.log10I_CI95  = log10I_CI95;
    out.I_mean       = I_mean;
    out.CI95         = CI95_lin;
    out.sigma_model  = CV_model_final;
    out.sigma_rqmc   = se_rqmc_final;
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
end

%% ====================== LOCAL FUNCTIONS ======================

function mdl = trainGP(Uin, Yin, kernelName, basisName)
    mdl = fitrgp(Uin, Yin, ...
        'KernelFunction', kernelName, ...
        'BasisFunction',  basisName, ...
        'Standardize',    true);
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
    % RQMC integration WITHOUT prior weighting (fallback)
    logIvals = zeros(R,1);
    sob = scramble(sobolset(d),'MatousekAffineOwen');
    ln10 = log(10);

    for r = 1:R
        U0 = net(sob, N);
        shift = rand(1,d);
        U = mod(U0 + shift, 1);

        w = jacobian_fn(U) / N;
        mu = predict(gpr, U);

        % Convert from -log10 P back to P
        log_w = log(w);
        log_integrand = log_w - ln10 * mu;  % log[w × P(D|M,θ)]

        max_log = max(log_integrand);
        logIvals(r) = max_log + log(sum(exp(log_integrand - max_log)));
    end

    max_logI = max(logIvals);
    logI_bar = max_logI + log(mean(exp(logIvals - max_logI)));

    se_rqmc = std(logIvals, 0) / sqrt(R);
    sd_model_bar = 0;
end

function [logI_bar, se_rqmc, sd_model_bar] = rqmc_integral_with_prior(gpr, N, R, jacobian_fn, d, modelName, priors, fromFeat_fn,Y_max)
    % RQMC integration WITH proper prior weighting P(θ|M)
    
    logIvals = zeros(R,1);
    sob = scramble(sobolset(d),'MatousekAffineOwen');
    ln10 = log(10);

    % Get prior function for this model
    modelKey = normalize_model_name(modelName);
    if isfield(priors, modelKey) && isfield(priors.(modelKey), 'prior_fn')
        prior_fn = priors.(modelKey).prior_fn;
    else
        warning('No prior function for %s, using uniform', modelName);
        prior_fn = @(theta) 0;  % log(1) = 0
    end

    for r = 1:R
        U0 = net(sob, N);
        shift = rand(1,d);
        U = mod(U0 + shift, 1);

        w = jacobian_fn(U) / N;
        mu = min(predict(gpr,U),Y_max);

        % Evaluate prior P(θ|M) at each integration point  
        % NOTE: The prior_fn returns log of DISCRETE probability P(θ|M)
        % The jacobian already includes the volume element ΔV(θ)
        % So: integrand = [P(θ|M) / ΔV] × ΔV × P(D|M,θ) = P(θ|M) × P(D|M,θ)
        logPrior_vec = zeros(N,1);
        for i = 1:N
            theta_i = fromFeat_fn(U(i,:));
            try
                logPrior_vec(i) = min(0,prior_fn(theta_i));
            catch ME
                logPrior_vec(i) = -inf;
            end
        end
if r == 1
    fprintf('DEBUG logPrior: mean=%.2f, max=%.2f, min=%.2f\n', mean(logPrior_vec), max(logPrior_vec), min(logPrior_vec));
end
        % Log integrand: log[w × P(D|M,θ) × P(θ|M)]
        % where w = jacobian / N already includes volume element
        log_w = log(w);
        log_integrand = log_w - ln10 * mu + logPrior_vec;

        % Stable logsumexp
        valid = isfinite(log_integrand);
        if any(valid)
            max_log = max(log_integrand(valid));
            logIvals(r) = max_log + log(sum(exp(log_integrand(valid) - max_log)));
        else
            logIvals(r) = -inf;
        end
    end

    % Combine across scrambles
    valid = isfinite(logIvals);
    if ~any(valid)
        logI_bar = -inf;
        se_rqmc = inf;
        sd_model_bar = 0;
        return;
    end

    max_logI = max(logIvals(valid));
    logI_bar = max_logI + log(mean(exp(logIvals(valid) - max_logI)));

    se_rqmc = std(logIvals(valid), 0) / sqrt(sum(valid));
    sd_model_bar = 0;
end

function key = normalize_model_name(modelName)
    switch lower(modelName)
        case {'newtonian', 'newt'}, key = 'Newt';
        case 'nh',                  key = 'NH';
        case 'kv',                  key = 'KV';
        case 'qnh',                 key = 'qNH';
        case {'linmax', 'max', 'lm'}, key = 'LM';
        case 'qkv',                 key = 'qKV';
        case 'sls',                 key = 'SLS';
        otherwise,                  key = upper(modelName);
    end
end

%% ====================== Coordinate Transforms ======================

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