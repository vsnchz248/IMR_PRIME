%% run_active_gp_1d_demo.m
% 1D demo: active GP integration + visualization of fit to log10 L(θ)

clear; clc; close all;
addpath('../IMRv2/src/forward_solver/')
tic
% -------------------- 1. Load / prepare experimental data --------------------
optsData = struct();
expData  = prepare_data(0, optsData);   % e.g. UM1; change material_id as needed

% IMPORTANT:
% Here we assume expData has already been augmented with:
%   expData.sigma0_R, expData.sigma0_Rdot, expData.weights_w, expData.beta
% consistent with the paper's noise model.
% If not, you need a preprocessing step to compute those first.

% -------------------- 2. Choose model and 1D parameter bounds ----------------
% Example: 1D over a single parameter (e.g., Newtonian viscosity mu)

modelName = 'Newtonian';   % or 'NH', 'KV', 'qKV', 'SLS', etc.

% Bounds for theta (here: mu). Adjust to your prior box.
xmin = 1e-4;          % lower bound on mu
xmax = 1;             % upper bound on mu

% -------------------- 3. Vectorized negative log-likelihood wrapper ---------
% funNLL must accept an N-by-d matrix X and return N-by-1 NLL values.
% Here d=1, so X is N-by-1, each row is a scalar theta.

funNLL = @(X) arrayfun(@(i) ...
    max(forward_solver_wrapper(X(i,:), modelName, expData), 0), ...  % NLL should be ≥ 0
    (1:size(X,1))' );

% -------------------- 4. Run active integrator ------------------------------
optsInt = struct();              % use defaults (maxRounds, etc.), or set here
out = active_integrate_logaware(funNLL, xmin, xmax, optsInt);

fprintf('\nApproximate evidence integral I ≈ %.6e\n', out.I_mean);
fprintf('95%% CI: [%.6e, %.6e]\n', out.CI95(1), out.CI95(2));

% -------------------- 5. Build a fine 1D grid for plotting ------------------
Np      = 400;                         % number of plot points
U_plot  = linspace(0,1,Np)';           % feature-space grid in [0,1]
theta_plot = out.fromFeat(U_plot);     % map to physical parameter θ

% GP posterior on log10 L
[mu10, sd10] = predict(out.gpr, U_plot);

logL_med = mu10;                       % median log10 L(θ)
logL_lo  = mu10 - 1.96*sd10;          % lower 95% band
logL_hi  = mu10 + 1.96*sd10;          % upper 95% band

% Training points (θ_i, log10 L_i)
theta_train = out.fromFeat(out.U);     % N_train-by-1
logL_train  = out.Y;                   % N_train-by-1 (these are log10 L)

% -------------------- 6. Plot: log10 L(θ) surrogate + data -------------------
fig = figure('Color','w','Units','normalized','Position',[0.18 0.20 0.60 0.55]);
movegui(fig, 'center'); hold on; box on;

% Sort for nice lines
[theta_plot_sorted, idx] = sort(theta_plot);
logL_med_s = logL_med(idx);
logL_lo_s  = logL_lo(idx);
logL_hi_s  = logL_hi(idx);

% 95% band (shaded)
x_band = [theta_plot_sorted; flipud(theta_plot_sorted)];
y_band = [logL_lo_s;        flipud(logL_hi_s)];
fill(x_band, y_band, [0.9 0.8 1.0], ...
     'EdgeColor','none', 'FaceAlpha',0.4);

% Median GP prediction
plot(theta_plot_sorted, logL_med_s, 'b-', 'LineWidth', 2.0);

% Training points
plot(theta_train, logL_train, 'ko', 'MarkerFaceColor','y', 'MarkerSize',5);

xlabel('\theta (parameter)', 'Interpreter','tex');
ylabel('\log_{10} L(\theta)', 'Interpreter','tex');
title(sprintf('1D Active GP Surrogate for log_{10} Likelihood | Model: %s', modelName), ...
      'Interpreter','tex');

legend({'95% band','GP median','Train pts'}, 'Location','best');

% Log-scale x if this axis was treated as log in the integrator
if out.log_axes(1) && xmin > 0
    set(gca,'XScale','log');
    xlim([xmin xmax]);
    xticks(10.^(floor(log10(xmin)):ceil(log10(xmax))));
    ax = gca; ax.XMinorGrid = 'on';
end
grid on;

% -------------------- 7. Optional: Plot L(θ) instead of log10 L(θ) ----------
% If you want to see the likelihood itself (can be very peaky / tiny):

figure('Color','w','Units','normalized','Position',[0.18 0.20 0.60 0.55]);
movegui(gcf, 'center'); hold on; box on;

L_med = 10.^logL_med_s;
L_lo  = 10.^logL_lo_s;
L_hi  = 10.^logL_hi_s;
L_train = 10.^logL_train;

x_band = [theta_plot_sorted; flipud(theta_plot_sorted)];
y_band = [L_lo;               flipud(L_hi)];
fill(x_band, y_band, [1.0 0.9 0.9], ...
     'EdgeColor','none', 'FaceAlpha',0.4);

plot(theta_plot_sorted, L_med, 'r-', 'LineWidth', 2.0);
plot(theta_train, L_train, 'ko', 'MarkerFaceColor','y', 'MarkerSize',5);

xlabel('\theta (parameter)', 'Interpreter','tex');
ylabel('L(\theta)', 'Interpreter','tex');
title(sprintf('1D Active GP Surrogate for Likelihood L(\\theta) | Model: %s', modelName), ...
      'Interpreter','tex');

if out.log_axes(1) && xmin > 0
    set(gca,'XScale','log');
    xlim([xmin xmax]);
    xticks(10.^(floor(log10(xmin)):ceil(log10(xmax))));
    ax = gca; ax.XMinorGrid = 'on';
end
set(gca,'YScale','log');  % likelihood often spans many decades
grid on;
legend({'95% band','GP median','Train pts'}, 'Location','best');


toc