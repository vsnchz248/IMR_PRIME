%% Hierarchical Bayesian Model Selection for IMR using GPR Surrogates
% Implements Algorithm 1 from the manuscript with GP-accelerated likelihood evaluation
% Victor Sanchez et al., "Hierarchical Bayesian constitutive model selection..."

clear; clc; close all;
tic
addpath('experiments/')
addpath('../IMRv2/src/forward_solver/')

%% ========================================================================
%  CONFIGURATION
%% ========================================================================

% Experimental data selection
material_id = 0;  % 0=synthetic, 1-3=UM, 4-9=UT (see Table 3 in manuscript)

% Models to compare (Table 1 in manuscript)
modelNames = {'Newtonian', 'NH', 'KV', 'qNH', 'LM', 'qKV', 'SLS'};
NM = length(modelNames);

% Adaptive GPR training parameters
% OPTION 1: Fast dimension-based (recommended for initial runs)
% OPTION 2: Convergence-based (more rigorous but slower)
USE_SIMPLE_SCALING = true;  % Set to false for convergence-based

if USE_SIMPLE_SCALING
    % Simple rule: 50 samples per dimension (very fast, good enough)
    adaptiveConfig.samplesPerDim = 50;
else
    % Convergence-based adaptive sampling
    adaptiveConfig.minSamples = 50;          % Minimum samples
    adaptiveConfig.maxSamples = 200;         % Maximum samples
    adaptiveConfig.batchSize = 25;           % Add this many samples per iteration
    adaptiveConfig.convergenceTol = 0.05;    % Stop when log-evidence change < 5%
    adaptiveConfig.nConvergeCheck = 2;       % Require stability over 2 iterations
    adaptiveConfig.mcSamplesForCheck = 2000; % MC samples for convergence check
end

% Beta (noise scale) grid for hierarchical prior (Eq. 16, Section 2.3.2)
beta_min = 0.05;
beta_max = 10;
nBeta = 50;
beta_grid = linspace(beta_min, beta_max, nBeta)';

% Half-Cauchy prior on beta (Eq. 16)
P_beta = (2/pi) ./ (1 + beta_grid.^2);
P_beta = P_beta / sum(P_beta);  % Normalize for discrete quadrature

% Effective number of observations for model prior (Eq. 28)
% Will be computed from experimental data

%% ========================================================================
%  1. LOAD AND PREPROCESS EXPERIMENTAL DATA
%% ========================================================================

fprintf('\n========================================\n');
fprintf('BAYESIAN IMR MODEL SELECTION\n');
fprintf('========================================\n\n');

fprintf('Loading experimental data (material %d)...\n', material_id);
expData = expDataPrep(material_id, 'makePlots', false);
expData.beta = 1.0;  % Will be marginalized over

% Compute effective number of observations (Section 2.3.3)
nTrials = size(expData.Rmatrix, 2);
nTimeSteps = size(expData.Rmatrix, 1);
nGatedPoints = nnz(expData.mask);
Neff = 2 * nGatedPoints;  % Factor of 2 for R and Rdot

fprintf('  Trials: %d\n', nTrials);
fprintf('  Time steps per trial: %d\n', nTimeSteps);
fprintf('  Gated points: %d / %d (%.1f%%)\n', nGatedPoints, numel(expData.mask), ...
        100*nGatedPoints/numel(expData.mask));
fprintf('  Effective observations: %d\n', Neff);

%% ========================================================================
%  2. BUILD GPR SURROGATES FOR EACH MODEL
%% ========================================================================

fprintf('\n--- Building GPR Surrogates ---\n');

% Storage for GPR models and training data
gprModels = cell(NM, 1);
trainingData = struct();

for iModel = 1:NM
    modelName = modelNames{iModel};
    fprintf('\nModel %d/%d: %s\n', iModel, NM, modelName);
    
    % Get parameter bounds for this model (Table 2 in manuscript)
    [param_bounds, param_names, nParams] = get_model_bounds(modelName);
    
    fprintf('  Parameters (%d): %s\n', nParams, strjoin(param_names, ', '));
    
    % ========== SAMPLING STRATEGY ==========
    if USE_SIMPLE_SCALING
        % SIMPLE SCALING: Fixed samples based on dimensionality
        % This is MUCH faster and works well in practice
        nSamples = adaptiveConfig.samplesPerDim * nParams;
        
        fprintf('  Using dimension-based sampling: %d samples (%d per dimension)\n', ...
                nSamples, adaptiveConfig.samplesPerDim);
        
        % Generate all samples at once via LHS
        rng(42 + iModel);
        X_unit = lhsdesign(nSamples, nParams);
        X_train = zeros(nSamples, nParams);
        
        for iParam = 1:nParams
            lb = param_bounds(iParam, 1);
            ub = param_bounds(iParam, 2);
            X_train(:, iParam) = 10.^(log10(lb) + X_unit(:, iParam) * (log10(ub) - log10(lb)));
        end
        
        % Evaluate log-likelihood at all training points
        fprintf('  Computing log-likelihoods...\n');
        logL_train = zeros(nSamples, 1);
        
        parfor iTrain = 1:nSamples
            theta = X_train(iTrain, :);
            
            % Compute log-likelihood marginalized over beta (Eq. 15)
            logL_beta = zeros(nBeta, 1);
            
            for iBeta = 1:nBeta
                expData_beta = expData;
                expData_beta.beta = beta_grid(iBeta);
                
                NLL = forward_solver_wrapper(theta, modelName, expData_beta);
                logL_beta(iBeta) = -NLL;
            end
            
            % Marginalize over beta (Eq. 17)
            max_logL = max(logL_beta);
            logL_train(iTrain) = max_logL + log(sum(exp(logL_beta - max_logL) .* P_beta));
        end
        
        nCurrent = nSamples;
        iteration = 1;
        converged = true;  % Not used in simple mode
        evidenceHistory = [];
        
        fprintf('  Log-likelihood range: [%.2f, %.2f]\n', min(logL_train), max(logL_train));
        
        % Compute normalization parameters for GPR training
        X_train_log = log10(X_train);
        X_mean_temp = mean(X_train_log, 1);
        X_std_temp = std(X_train_log, 0, 1);
        Y_mean_temp = mean(logL_train);
        Y_std_temp = std(logL_train);
        
    else
        % CONVERGENCE-BASED ADAPTIVE SAMPLING (slower but more rigorous)
        fprintf('  Starting convergence-based adaptive sampling...\n');
        fprintf('    (Set USE_SIMPLE_SCALING=true for faster fixed sampling)\n');
    
    rng(42 + iModel);  % Reproducibility per model
    
    % Initialize with minimum samples
    nCurrent = adaptiveConfig.minSamples;
    X_train = [];
    logL_train = [];
    converged = false;
    iteration = 0;
    evidenceHistory = [];
    
    while ~converged && nCurrent <= adaptiveConfig.maxSamples
        iteration = iteration + 1;
        
        % Generate new batch of samples using LHS
        X_unit_batch = lhsdesign(adaptiveConfig.batchSize, nParams);
        X_batch = zeros(adaptiveConfig.batchSize, nParams);
        
        for iParam = 1:nParams
            lb = param_bounds(iParam, 1);
            ub = param_bounds(iParam, 2);
            X_batch(:, iParam) = 10.^(log10(lb) + X_unit_batch(:, iParam) * (log10(ub) - log10(lb)));
        end
        
        % Evaluate log-likelihood at new batch
        logL_batch = zeros(adaptiveConfig.batchSize, 1);
        
        parfor iBatch = 1:adaptiveConfig.batchSize
            theta = X_batch(iBatch, :);
            
            % Compute log-likelihood marginalized over beta (Eq. 15)
            logL_beta = zeros(nBeta, 1);
            
            for iBeta = 1:nBeta
                expData_beta = expData;
                expData_beta.beta = beta_grid(iBeta);
                
                % Negative log-likelihood from forward solver
                NLL = forward_solver_wrapper(theta, modelName, expData_beta);
                logL_beta(iBeta) = -NLL;  % Convert to log-likelihood
            end
            
            % Marginalize over beta using Half-Cauchy prior (Eq. 17)
            max_logL = max(logL_beta);
            logL_batch(iBatch) = max_logL + log(sum(exp(logL_beta - max_logL) .* P_beta));
        end
        
        % Append to training set
        X_train = [X_train; X_batch];
        logL_train = [logL_train; logL_batch];
        nCurrent = size(X_train, 1);
        
        fprintf('    Iteration %d: %d samples total (log-L range: [%.2f, %.2f])\n', ...
                iteration, nCurrent, min(logL_train), max(logL_train));
        
        % Train GPR on current data
        X_train_log = log10(X_train);
        X_mean_temp = mean(X_train_log, 1);
        X_std_temp = std(X_train_log, 0, 1);
        X_train_norm = (X_train_log - X_mean_temp) ./ X_std_temp;
        
        Y_mean_temp = mean(logL_train);
        Y_std_temp = std(logL_train);
        Y_train_norm = (logL_train - Y_mean_temp) / Y_std_temp;
        
        gpr_temp = fitrgp(X_train_norm, Y_train_norm, ...
                          'KernelFunction', 'matern52', ...
                          'Sigma', 0.01, ...
                          'Standardize', false, ...
                          'ConstantSigma', false, ...
                          'OptimizeHyperparameters', 'none');  % Faster, no optimization
        
        % Check convergence: compute evidence integral with current GPR
        X_mc_unit = rand(adaptiveConfig.mcSamplesForCheck, nParams);
        X_mc = zeros(adaptiveConfig.mcSamplesForCheck, nParams);
        
        for iParam = 1:nParams
            lb = param_bounds(iParam, 1);
            ub = param_bounds(iParam, 2);
            X_mc(:, iParam) = 10.^(log10(lb) + X_mc_unit(:, iParam) * (log10(ub) - log10(lb)));
        end
        
        X_mc_log = log10(X_mc);
        X_mc_norm = (X_mc_log - X_mean_temp) ./ X_std_temp;
        
        [Y_mc_norm, ~] = predict(gpr_temp, X_mc_norm);
        logL_mc = Y_mc_norm * Y_std_temp + Y_mean_temp;
        
        % Evidence integral (log-sum-exp)
        max_logL = max(logL_mc);
        log_evidence_current = max_logL + log(mean(exp(logL_mc - max_logL)));
        evidenceHistory = [evidenceHistory; log_evidence_current];
        
        fprintf('      Current log-evidence: %.6f\n', log_evidence_current);
        
        % Check for convergence: evidence stable over last N iterations?
        if iteration >= adaptiveConfig.nConvergeCheck
            recent = evidenceHistory(end-adaptiveConfig.nConvergeCheck+1:end);
            relChange = abs(diff(recent)) / max(abs(recent));
            
            if all(relChange < adaptiveConfig.convergenceTol)
                converged = true;
                fprintf('    ✓ CONVERGED: Evidence stable within %.2f%% over %d iterations\n', ...
                        100*adaptiveConfig.convergenceTol, adaptiveConfig.nConvergeCheck);
            end
        end
        
        % Safety check
        if nCurrent >= adaptiveConfig.maxSamples
            fprintf('    ⚠ Max samples reached (%d). Stopping.\n', adaptiveConfig.maxSamples);
            break;
        end
    end
    
    fprintf('  Final training set: %d samples\n', nCurrent);
    
    end  % End if USE_SIMPLE_SCALING / else block
    
    % Train final GPR model on complete dataset with hyperparameter optimization
    fprintf('  Training final GPR with hyperparameter optimization...\n');
    
    % Use accumulated normalization parameters
    X_mean = X_mean_temp;
    X_std = X_std_temp;
    Y_mean = Y_mean_temp;
    Y_std = Y_std_temp;
    
    X_train_log = log10(X_train);
    X_train_norm = (X_train_log - X_mean) ./ X_std;
    Y_train_norm = (logL_train - Y_mean) / Y_std;
    
    % Final GPR with hyperparameter optimization for best accuracy
    % Matern52 kernel needs 2 parameters: [lengthScale; signalStd]
    gpr = fitrgp(X_train_norm, Y_train_norm, ...
                 'KernelFunction', 'matern52', ...
                 'Sigma', 0.01, ...
                 'Standardize', false, ...
                 'ConstantSigma', false, ...
                 'OptimizeHyperparameters', 'auto', ...
                 'HyperparameterOptimizationOptions', ...
                 struct('Verbose', 0, 'ShowPlots', false, 'MaxObjectiveEvaluations', 30));
    
    % Store GPR and normalization parameters
    gprModels{iModel}.gpr = gpr;
    gprModels{iModel}.X_mean = X_mean;
    gprModels{iModel}.X_std = X_std;
    gprModels{iModel}.Y_mean = Y_mean;
    gprModels{iModel}.Y_std = Y_std;
    gprModels{iModel}.param_bounds = param_bounds;
    gprModels{iModel}.param_names = param_names;
    gprModels{iModel}.nParams = nParams;
    gprModels{iModel}.nSamples = nCurrent;
    gprModels{iModel}.converged = converged;
    gprModels{iModel}.nIterations = iteration;
    gprModels{iModel}.evidenceHistory = evidenceHistory;
    
    % Store training data for diagnostics
    trainingData(iModel).modelName = modelName;
    trainingData(iModel).X_train = X_train;
    trainingData(iModel).logL_train = logL_train;
    trainingData(iModel).nSamples = nCurrent;
    trainingData(iModel).converged = converged;
    
    fprintf('  GPR trained successfully (used %d samples over %d iterations).\n', nCurrent, iteration);
end

%% ========================================================================
%  3. COMPUTE MODEL EVIDENCE INTEGRALS (Eq. 19-20)
%% ========================================================================

fprintf('\n--- Computing Model Evidence Integrals ---\n');

% Storage for results
modelEvidence = zeros(NM, 1);
modelPrior = zeros(NM, 1);
modelPosterior = zeros(NM, 1);
mapParameters = cell(NM, 1);
mapLogLikelihood = zeros(NM, 1);

% Integration grid for Monte Carlo quadrature
nMC = 10000;  % Monte Carlo samples for evidence integral

for iModel = 1:NM
    modelName = modelNames{iModel};
    fprintf('\nModel %d/%d: %s\n', iModel, NM, modelName);
    
    gprModel = gprModels{iModel};
    nParams = gprModel.nParams;
    param_bounds = gprModel.param_bounds;
    
    % Model prior (Eq. 28): P(M_i) = exp(-k_M/2 * log(Neff))
    kM = nParams;
    modelPrior(iModel) = exp(-kM/2 * log(Neff));
    fprintf('  Model prior P(M): %.6e (kM=%d)\n', modelPrior(iModel), kM);
    
    % Generate Monte Carlo samples for evidence integral
    % Using importance sampling with log-uniform prior
    rng(42 + iModel);
    X_mc_unit = rand(nMC, nParams);
    X_mc = zeros(nMC, nParams);
    
    for iParam = 1:nParams
        lb = param_bounds(iParam, 1);
        ub = param_bounds(iParam, 2);
        X_mc(:, iParam) = 10.^(log10(lb) + X_mc_unit(:, iParam) * (log10(ub) - log10(lb)));
    end
    
    % Predict log-likelihood at MC samples using GPR
    X_mc_log = log10(X_mc);
    X_mc_norm = (X_mc_log - gprModel.X_mean) ./ gprModel.X_std;
    
    [Y_mc_norm, ~] = predict(gprModel.gpr, X_mc_norm);
    logL_mc = Y_mc_norm * gprModel.Y_std + gprModel.Y_mean;
    
    % Parameter prior: uniform in log-space (normalized over bounds)
    % P(theta|M) = 1 / product(log(ub/lb))
    log_prior_volume = sum(log10(param_bounds(:,2) ./ param_bounds(:,1))) * log(10);
    log_P_theta = -log_prior_volume;  % Constant for all samples
    
    % Evidence integral using log-sum-exp trick (Eq. 20)
    % P(D|M) ≈ (1/N_MC) * sum_i P(D|M,theta_i) * P(theta_i|M) / q(theta_i)
    % where q(theta_i) = P(theta_i|M) (importance distribution = prior)
    % So: P(D|M) ≈ (1/N_MC) * sum_i P(D|M,theta_i)
    
    log_integrand = logL_mc;  % P(theta|M) cancels with q(theta)
    max_log_integrand = max(log_integrand);
    
    log_evidence = max_log_integrand + log(mean(exp(log_integrand - max_log_integrand)));
    modelEvidence(iModel) = log_evidence;
    
    fprintf('  Log-evidence: %.3f\n', log_evidence);
    
    % Find MAP parameters (maximum of posterior)
    [mapLogLikelihood(iModel), iMAP] = max(logL_mc);
    mapParameters{iModel} = X_mc(iMAP, :);
    
    fprintf('  MAP log-likelihood: %.3f\n', mapLogLikelihood(iModel));
    fprintf('  MAP parameters:\n');
    for iParam = 1:nParams
        fprintf('    %s = %.4e\n', gprModel.param_names{iParam}, mapParameters{iModel}(iParam));
    end
end

%% ========================================================================
%  4. COMPUTE MODEL POSTERIOR PROBABILITIES (Eq. 9)
%% ========================================================================

fprintf('\n--- Model Selection Results ---\n');

% Unnormalized log-posterior: log P(M|D) = log P(M) + log P(D|M) - log P(D)
log_unnorm_posterior = log(modelPrior) + modelEvidence;

% Normalize using log-sum-exp
max_log_post = max(log_unnorm_posterior);
modelPosterior = exp(log_unnorm_posterior - max_log_post);
modelPosterior = modelPosterior / sum(modelPosterior);

% Sort by posterior probability
[posteriorSorted, iSort] = sort(modelPosterior, 'descend');

fprintf('\nModel Ranking (by posterior probability):\n');
fprintf('%-15s | %10s | %12s | %12s | nParams\n', 'Model', 'Posterior', 'Log-Evidence', 'MAP LogL');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:NM
    iModel = iSort(i);
    fprintf('%-15s | %10.6f | %12.3f | %12.3f | %d\n', ...
            modelNames{iModel}, posteriorSorted(i), ...
            modelEvidence(iModel), mapLogLikelihood(iModel), ...
            gprModels{iModel}.nParams);
end

% Best model
iBest = iSort(1);
bestModel = modelNames{iBest};
fprintf('\n*** SELECTED MODEL: %s (P = %.4f) ***\n', bestModel, posteriorSorted(1));
fprintf('MAP Parameters:\n');
for iParam = 1:gprModels{iBest}.nParams
    fprintf('  %s = %.4e\n', gprModels{iBest}.param_names{iParam}, ...
            mapParameters{iBest}(iParam));
end

%% ========================================================================
%  5. VISUALIZE RESULTS
%% ========================================================================

fprintf('\n--- Generating Visualizations ---\n');

% Figure 1: Model posterior probabilities (bar chart)
fig1 = figure('Color', 'w', 'Position', [100, 100, 1200, 800]);

subplot(2,2,1)
bar(modelPosterior(iSort), 'FaceColor', [0.3 0.6 0.9]);
set(gca, 'XTickLabel', modelNames(iSort), 'XTick', 1:NM);
ylabel('Posterior Probability $P(M_i|D)$', 'Interpreter', 'latex');
title('Model Selection Results', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

subplot(2,2,2)
bar(modelEvidence(iSort), 'FaceColor', [0.9 0.6 0.3]);
set(gca, 'XTickLabel', modelNames(iSort), 'XTick', 1:NM);
ylabel('Log-Evidence $\log P(D|M_i)$', 'Interpreter', 'latex');
xlabel('Constitutive Model', 'Interpreter', 'latex');
title('Model Evidence (Occam Factor)', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Convergence diagnostics
subplot(2,2,3)
hold on;
colors = lines(NM);
for iModel = 1:NM
    if ~isempty(gprModels{iModel}.evidenceHistory)
        nIter = length(gprModels{iModel}.evidenceHistory);
        plot(1:nIter, gprModels{iModel}.evidenceHistory, 'o-', ...
             'Color', colors(iModel,:), 'LineWidth', 1.5, 'MarkerSize', 6);
    end
end
xlabel('Iteration', 'Interpreter', 'latex');
ylabel('Log-Evidence', 'Interpreter', 'latex');
title('Convergence History', 'Interpreter', 'latex', 'FontSize', 14);
legend(modelNames, 'Location', 'best', 'Interpreter', 'latex');
grid on;
set(gca, 'FontSize', 12);

subplot(2,2,4)
nSamplesUsed = cellfun(@(x) x.nSamples, gprModels);
bar(nSamplesUsed, 'FaceColor', [0.6 0.8 0.4]);
set(gca, 'XTickLabel', modelNames, 'XTick', 1:NM);
ylabel('Training Samples Used', 'Interpreter', 'latex');
xlabel('Model', 'Interpreter', 'latex');
title('Adaptive Sampling Efficiency', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);
for iModel = 1:NM
    if gprModels{iModel}.converged
        text(iModel, nSamplesUsed(iModel)+5, '\checkmark', ...
             'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', 'g');
    else
        text(iModel, nSamplesUsed(iModel)+5, '?', ...
             'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', 'r');
    end
end

% Figure 2: MAP simulation vs experimental data for best model
fig2 = figure('Color', 'w', 'Position', [200, 200, 1200, 500]);

% Run forward simulation at MAP parameters
theta_MAP = mapParameters{iBest};
[t_sim_nd, R_sim_nd, Rdot_sim_nd] = run_forward_simulation(theta_MAP, bestModel, expData);

subplot(1,2,1)
hold on;
% Plot experimental cloud
for j = 1:nTrials
    plot(expData.tmatrix(:,j), expData.Rmatrix(:,j), 'k.', 'MarkerSize', 2, 'Color', [0.7 0.7 0.7]);
end
% Plot MAP simulation
plot(t_sim_nd, R_sim_nd, 'r-', 'LineWidth', 2);
xlabel('$\tau$ (dimensionless time)', 'Interpreter', 'latex');
ylabel('$R^*$ (normalized radius)', 'Interpreter', 'latex');
title(sprintf('Best Model: %s', bestModel), 'Interpreter', 'latex');
legend('Experimental', 'MAP Simulation', 'Location', 'best', 'Interpreter', 'latex');
grid on; box on;
set(gca, 'FontSize', 12);
xlim([0 max(expData.tmatrix(:))]);

subplot(1,2,2)
hold on;
% Plot experimental cloud
for j = 1:nTrials
    plot(expData.tmatrix(:,j), expData.Rdotmatrix(:,j), 'k.', 'MarkerSize', 2, 'Color', [0.7 0.7 0.7]);
end
% Plot MAP simulation
plot(t_sim_nd, Rdot_sim_nd, 'r-', 'LineWidth', 2);
xlabel('$\tau$ (dimensionless time)', 'Interpreter', 'latex');
ylabel('$\dot{R}^*$ (normalized velocity)', 'Interpreter', 'latex');
title(sprintf('Best Model: %s', bestModel), 'Interpreter', 'latex');
grid on; box on;
set(gca, 'FontSize', 12);
xlim([0 max(expData.tmatrix(:))]);

%% ========================================================================
%  6. SAVE RESULTS
%% ========================================================================

results = struct();
results.modelNames = modelNames;
results.modelPosterior = modelPosterior;
results.modelEvidence = modelEvidence;
results.modelPrior = modelPrior;
results.mapParameters = mapParameters;
results.mapLogLikelihood = mapLogLikelihood;
results.bestModel = bestModel;
results.bestModelIndex = iBest;
results.gprModels = gprModels;
results.trainingData = trainingData;
results.expData = expData;
results.Neff = Neff;

save('bayesian_imr_results.mat', '-struct', 'results');
fprintf('\nResults saved to: bayesian_imr_results.mat\n');

toc

%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function [bounds, names, nParams] = get_model_bounds(modelName)
    % Returns parameter bounds and names for each model (Table 2)
    % Format: bounds = [lb, ub] for each parameter
    
    switch modelName
        case 'Newtonian'
            bounds = [1e-4, 1];  % mu [Pa·s]
            names = {'mu'};
            
        case 'NH'
            bounds = [1e2, 1e5];  % G [Pa]
            names = {'G'};
            
        case 'KV'
            bounds = [1e-4, 1;    % mu [Pa·s]
                      1e2, 1e5];   % G [Pa]
            names = {'mu', 'G'};
            
        case 'qNH'
            bounds = [1e2, 1e5;   % G [Pa]
                      1e-3, 10];   % alpha
            names = {'G', 'alpha'};
            
        case 'LM'
            bounds = [1e-4, 1;       % mu [Pa·s]
                      1e-7, 1e-3];   % lambda1 [s]
            names = {'mu', 'lambda1'};
            
        case 'qKV'
            bounds = [1e-4, 1;    % mu [Pa·s]
                      1e2, 1e5;   % G [Pa]
                      1e-3, 10];  % alpha
            names = {'mu', 'G', 'alpha'};
            
        case 'SLS'
            bounds = [1e-4, 1;       % mu [Pa·s]
                      1e2, 1e5;      % G [Pa]
                      1e-7, 1e-3];   % lambda1 [s]
            names = {'mu', 'G', 'lambda1'};
            
        otherwise
            error('Unknown model: %s', modelName);
    end
    
    nParams = size(bounds, 1);
end

function [t_sim_nd, R_sim_nd, Rdot_sim_nd] = run_forward_simulation(theta, modelName, expData)
    % Run forward simulation and return nondimensional results
    
    % Parse parameters
    [mu, G, lambda1, alpha, stress_model] = parse_model_parameters(theta, modelName);
    
    % IMR solver configuration
    radial = 2; bubtherm = 1; medtherm = 0; masstrans = 1; vapor = 1;
    Nt = 100; Mt = 100;
    
    % Mean experimental conditions
    Rmax_sim = expData.Rmax_mean;
    tc_sim = expData.tc_mean;
    Req_sim = mean(expData.Req_each, 'omitnan');
    
    % Time vector (dimensional)
    t_exp_mean_nd = mean(expData.tmatrix, 2);
    tvector = t_exp_mean_nd * tc_sim;
    
    % Call IMR solver
    [t_sim_nd, R_sim_nd, Rdot_sim_nd, ~] = f_imr_fd( ...
        'radial', radial, 'stress', stress_model, 'bubtherm', bubtherm, ...
        'medtherm', medtherm, 'masstrans', masstrans, 'vapor', vapor, ...
        'tvector', tvector, 'r0', Rmax_sim, 'req', Req_sim, ...
        'mu', mu, 'g', G, 'lambda1', lambda1, 'lambda2', 0, ...
        'alphax', alpha, 'collapse', 0, 'nt', Nt, 'mt', Mt);
end