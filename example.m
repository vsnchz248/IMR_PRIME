%% EXAMPLE_MODEL_SELECTION.m
% Complete example of Bayesian model selection with GPR for IMR
%
% This script demonstrates:
% 1. Loading experimental data
% 2. Building physics-informed priors
% 3. Defining constitutive models
% 4. Running Bayesian model selection with GPR
% 5. Visualizing results
%
% Author: [Your name]
% Date: 2025

clear; clc; close all;

%% Add source directory to path
% addpath('../src');
addpath('../experiments')
addpath('../')

%% Step 1: Load and prepare experimental data
fprintf('=== Step 1: Load Experimental Data ===\n');

% Choose material (see prepare_experimental_data.m for options)
% 1: UM1 (10% Gelatin)
% 4: UT1 (10% Gelatin, cross-validation)
% 5: UT2 (5% Agarose)
material_id = 1;

% Prepare data with default options
expData = prepare_experimental_data(material_id);

% Visualize (optional)
figure('Name', 'Experimental Data');
subplot(2,1,1);
plot(expData.tmatrix, expData.Rmatrix)%, 'Color', [0.5 0.5 0.5 0.3]);
xlabel('t^* (dimensionless time)'); ylabel('R^* (dimensionless radius)');
title(sprintf('Normalized Bubble Radius - %s', expData.material_name));
grid on;

subplot(2,1,2);
plot(expData.strain(:), (expData.strainRate(:)))%, 5, ...
        % 'MarkerFaceAlpha', 0.1, 'MarkerEdgeAlpha', 0.1);
set(gca, 'YScale', 'log');
xlabel('Hencky Strain \epsilon_H'); ylabel('Strain Rate |\epsilon^.|');
title('Phase Space');
grid on;

%% Step 2: Build physics-informed priors
fprintf('\n=== Step 2: Build Physics-Informed Priors ===\n');

priors = build_physics_informed_priors(expData);

%% Step 3: Define constitutive models
fprintf('\n=== Step 3: Define Constitutive Models ===\n');

% You need to implement these forward solvers for your specific system
% Each should have signature: simData = solver(params, expData)
% where params is a vector [mu, G, lambda1, alpha] (subset depending on model)
% and simData is a structure with fields: t, R, Rdot

% For this example, we'll use placeholder solvers
% In practice, replace these with your actual IMR forward solver

modelLibrary = {
    % 1D models
    struct('name', 'Newt', ...
           'paramNames', {{'mu'}}, ...
           'paramBounds', [1e-4, 1], ...
           'forwardSolver', @(p, d) solve_Newtonian(p, d)), ...
    
    struct('name', 'NH', ...
           'paramNames', {{'G'}}, ...
           'paramBounds', [1e2, 1e5], ...
           'forwardSolver', @(p, d) solve_NeoHookean(p, d)), ...
    
    % 2D models
    struct('name', 'KV', ...
           'paramNames', {{'mu', 'G'}}, ...
           'paramBounds', [1e-4, 1; 1e2, 1e5], ...
           'forwardSolver', @(p, d) solve_KelvinVoigt(p, d)), ...
    
    struct('name', 'qNH', ...
           'paramNames', {{'G', 'alpha'}}, ...
           'paramBounds', [1e2, 1e5; 1e-3, 1e1], ...
           'forwardSolver', @(p, d) solve_QuadraticNH(p, d)), ...
    
    struct('name', 'LM', ...
           'paramNames', {{'mu', 'lambda1'}}, ...
           'paramBounds', [1e-4, 1; 1e-7, 1e-3], ...
           'forwardSolver', @(p, d) solve_LinearMaxwell(p, d)), ...
    
    % 3D models
    struct('name', 'qKV', ...
           'paramNames', {{'mu', 'G', 'alpha'}}, ...
           'paramBounds', [1e-4, 1; 1e2, 1e5; 1e-3, 1e1], ...
           'forwardSolver', @(p, d) solve_QuadraticKV(p, d)), ...
    
    struct('name', 'SLS', ...
           'paramNames', {{'mu', 'G', 'lambda1'}}, ...
           'paramBounds', [1e-4, 1; 1e2, 1e5; 1e-7, 1e-3], ...
           'forwardSolver', @(p, d) solve_SLS(p, d))
};

fprintf('Defined %d models\n', numel(modelLibrary));

%% Step 4: Configure and run Bayesian model selection
fprintf('\n=== Step 4: Run Bayesian Model Selection ===\n');

% Configuration
cfg = struct();

% GPR active learning parameters
cfg.gpr_opts = struct();
cfg.gpr_opts.maxRounds = 30;        % Maximum AL rounds
cfg.gpr_opts.tolRelCI = 0.03;       % Stop when CI half-width ≤ 3%
cfg.gpr_opts.verbose = true;        % Display progress

% Likelihood parameters
cfg.useHetero = true;               % Heteroscedastic weighting
cfg.kappa = 1.0;                    % Strain-rate gate steepness
cfg.m_floor = 0.10;                 % Minimum weight floor
cfg.betaGrid = 0.05:0.05:10.0;      % Noise scale grid
cfg.useRdotInLL = true;             % Include velocity in likelihood

% Prior parameters
cfg.useBICprior = true;             % BIC-like complexity penalty
cfg.priors = priors;                % Physics-informed priors

% Run model selection
results = bayesian_model_selection(modelLibrary, expData, cfg);

%% Step 5: Visualize results
fprintf('\n=== Step 5: Visualize Results ===\n');

% Posterior probabilities
figure('Name', 'Model Posteriors');
bar(results.posterior);
set(gca, 'XTickLabel', results.modelNames);
ylabel('Posterior Probability');
title('Model Selection Results');
grid on;

% Evidence with uncertainty
figure('Name', 'Model Evidence');
logZ = [results.Models.logZ];
logZ_err = [results.Models.logZ_uncertainty];
errorbar(1:numel(logZ), logZ, 1.96*logZ_err, 'o-', 'LineWidth', 2);
set(gca, 'XTick', 1:numel(logZ), 'XTickLabel', results.modelNames);
ylabel('log Evidence (log Z)');
title('Model Evidence with 95% CI');
grid on;

% Best model comparison
bestIdx = results.bestModel;
bestModel = results.Models(bestIdx);

figure('Name', sprintf('Best Model: %s', bestModel.name));
subplot(2,1,1);

% Simulate with MAP parameters
simData = modelLibrary{bestIdx}.forwardSolver(...
    bestModel.theta_MAP, expData);

% Plot
hold on;
plot(expData.tmatrix, expData.Rmatrix, 'Color', [0.7 0.7 0.7 0.3]);
plot(simData.t, simData.R, 'r-', 'LineWidth', 2);
xlabel('t^*'); ylabel('R^*');
title(sprintf('%s (MAP): [%s]', bestModel.name, ...
      sprintf('%.3g ', bestModel.theta_MAP)));
legend('Experimental trials', 'MAP simulation', 'Location', 'best');
grid on;

% Beta posterior
subplot(2,1,2);
beta_post = build_likelihood_evaluator(expData, cfg).getBetaPosterior(simData);
plot(cfg.betaGrid, beta_post, 'b-', 'LineWidth', 2);
xline(bestModel.beta_MAP, 'r--', 'LineWidth', 1.5);
xlabel('\beta (noise scale)');
ylabel('P(\beta | Data, MAP)');
title(sprintf('\\beta_{MAP} = %.3f, \\beta_{mean} = %.3f', ...
      bestModel.beta_MAP, bestModel.beta_mean));
grid on;

% Display summary
fprintf('\n=== Summary ===\n');
fprintf('Most plausible model: %s\n', results.modelNames{bestIdx});
fprintf('Posterior probability: %.4f\n', results.posterior(bestIdx));
fprintf('MAP parameters: [%s]\n', sprintf('%.3g ', bestModel.theta_MAP));
fprintf('Beta MAP: %.3f\n', bestModel.beta_MAP);
fprintf('Total samples used: %d\n', bestModel.nSamples);

%% Step 6: Save results
save(sprintf('results_%s.mat', expData.material_name), 'results', 'expData', 'cfg');
fprintf('\nResults saved to results_%s.mat\n', expData.material_name);

%% ==================== Placeholder Forward Solvers ====================
% These are placeholders - replace with your actual IMR forward solver
% Each should call your Keller-Miksis solver with appropriate stress integrals

function simData = solve_Newtonian(params, expData)
% Newtonian: S* = -4μ(Ṙ*/R*)/Re
mu = params(1);
simData = run_keller_miksis_solver('Newt', struct('mu', mu), expData);
end

function simData = solve_NeoHookean(params, expData)
% Neo-Hookean: S* = (1/2Ca)[4λ⁻¹ + λ⁻⁴ - 5]
G = params(1);
simData = run_keller_miksis_solver('NH', struct('G', G), expData);
end

function simData = solve_KelvinVoigt(params, expData)
% Kelvin-Voigt: S* = S_v + S_NH
mu = params(1);
G = params(2);
simData = run_keller_miksis_solver('KV', struct('mu', mu, 'G', G), expData);
end

function simData = solve_QuadraticNH(params, expData)
% Quadratic neo-Hookean
G = params(1);
alpha = params(2);
simData = run_keller_miksis_solver('qNH', struct('G', G, 'alpha', alpha), expData);
end

function simData = solve_LinearMaxwell(params, expData)
% Linear Maxwell: De*Ṡ* + S* = S_v
mu = params(1);
lambda1 = params(2);
simData = run_keller_miksis_solver('LM', struct('mu', mu, 'lambda1', lambda1), expData);
end

function simData = solve_QuadraticKV(params, expData)
% Quadratic Kelvin-Voigt
mu = params(1);
G = params(2);
alpha = params(3);
simData = run_keller_miksis_solver('qKV', ...
    struct('mu', mu, 'G', G, 'alpha', alpha), expData);
end

function simData = solve_SLS(params, expData)
% Standard Linear Solid
mu = params(1);
G = params(2);
lambda1 = params(3);
simData = run_keller_miksis_solver('SLS', ...
    struct('mu', mu, 'G', G, 'lambda1', lambda1), expData);
end

function simData = run_keller_miksis_solver(modelName, params, expData)
% Interface to your actual Keller-Miksis solver
% This is a placeholder - replace with actual implementation
%
% Expected output structure:
%   simData.t      : Time points (dimensionless)
%   simData.R      : Radius (dimensionless)
%   simData.Rdot   : Velocity (dimensionless)

error(['Keller-Miksis solver not implemented. ' ...
       'Replace run_keller_miksis_solver() with your actual forward solver.']);

% Example structure (your solver should return this format):
% simData = struct();
% simData.t = linspace(0, 5, 500)';
% simData.R = exp(-simData.t/2) .* (1 + 0.1*sin(10*simData.t));
% simData.Rdot = gradient(simData.R, simData.t);
end