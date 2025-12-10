function diagnose_bimr_gpr()
% DIAGNOSE_BIMR_GPR - Compare original BIMR vs GPR implementation
%
% This script helps identify discrepancies between the two approaches

%% Setup
fprintf('\n=== BIMR vs GPR Diagnostic ===\n\n');

% Load test data (synthetic, material 0)
material_id = 0;
expData = prepare_data(material_id, struct('verbose', false));
priors = build_priors(expData, struct('verbose', false));

% Test on a simple model first (NH - 1D)
modelName = 'nh';
fprintf('Testing model: %s\n', upper(modelName));

%% 1) Get parameter bounds and create test grid
[xmin, xmax] = get_param_bounds_from_priors(modelName, priors);
n_test = 50;
theta_test = logspace(log10(xmin), log10(xmax), n_test)';

%% 2) Evaluate NLL at test points
fprintf('\nEvaluating NLL at %d test points...\n', n_test);
likeOpts = struct('useRdot', true, 'betaGrid', 0.05:0.05:10);

NLL_vec = imr_nll_with_prior_matrix(theta_test, modelName, expData, priors, likeOpts);

%% 3) Check for numerical issues
fprintf('\n--- NLL Statistics ---\n');
fprintf('Min NLL:    %.6e\n', min(NLL_vec));
fprintf('Max NLL:    %.6e\n', max(NLL_vec));
fprintf('Mean NLL:   %.6e\n', mean(NLL_vec));
fprintf('Median NLL: %.6e\n', median(NLL_vec));
fprintf('# Inf:      %d\n', sum(isinf(NLL_vec)));
fprintf('# NaN:      %d\n', sum(isnan(NLL_vec)));

%% 4) Compare with direct prior evaluation
fprintf('\n--- Prior Evaluation ---\n');
logPrior_vec = zeros(n_test, 1);
for i = 1:n_test
    logPrior_vec(i) = priors.paramPrior(modelName, theta_test(i));
end

fprintf('Min log P(¸|M):  %.6e\n', min(logPrior_vec));
fprintf('Max log P(¸|M):  %.6e\n', max(logPrior_vec));
fprintf('# -Inf:          %d\n', sum(isinf(logPrior_vec) & logPrior_vec < 0));

%% 5) Compute evidence using quadrature (reference)
fprintf('\n--- Reference Evidence (Quadrature) ---\n');
logL_vec = -NLL_vec;  % Convert back to log-likelihood

% Apply Jacobian for log-space integration
dtheta = diff([xmin; theta_test]);  % Trapezoid widths
jacobian = theta_test * log(10);    % d(10^x)/dx = ln(10) * 10^x

% Trapezoidal quadrature weights
weights = zeros(n_test, 1);
weights(1) = 0.5 * dtheta(1) * jacobian(1);
for i = 2:n_test-1
    weights(i) = 0.5 * (dtheta(i) + dtheta(i+1)) * jacobian(i);
end
weights(end) = 0.5 * dtheta(end) * jacobian(end);

% Evidence WITHOUT prior (should be wrong)
log_integrand_no_prior = logL_vec + log(weights);
logZ_no_prior = logsumexp_stable(log_integrand_no_prior);
fprintf('log P(D|M) [NO prior]:   %.6e\n', logZ_no_prior);

% Evidence WITH prior (correct)
log_integrand_with_prior = logL_vec + logPrior_vec + log(weights);
logZ_with_prior = logsumexp_stable(log_integrand_with_prior);
fprintf('log P(D|M) [WITH prior]: %.6e\n', logZ_with_prior);

% Compare
fprintf('Difference: %.6e\n', logZ_with_prior - logZ_no_prior);

%% 6) Now test GPR integration
fprintf('\n--- GPR Integration ---\n');
gprOpts = struct('maxRounds', 30, 'tolRelCI', 0.04, 'verbose', true);

funNLL = @(X) imr_nll_with_prior_matrix(X, modelName, expData, priors, likeOpts);
gpr_result = active_integrate_logaware(funNLL, xmin, xmax, gprOpts, modelName, priors);

fprintf('GPR log10(evidence): %.6e\n', gpr_result.log10I_mean);
fprintf('GPR log(evidence):   %.6e\n', gpr_result.logI_mean);

%% 7) Compare
fprintf('\n=== COMPARISON ===\n');
fprintf('Quadrature (with prior): %.6e\n', logZ_with_prior);
fprintf('GPR result:              %.6e\n', gpr_result.logI_mean);
fprintf('Difference:              %.6e\n', abs(logZ_with_prior - gpr_result.logI_mean));

if abs(logZ_with_prior - gpr_result.logI_mean) < 1.0
    fprintf(' Results agree within 1 neper\n');
else
    fprintf(' WARNING: Large discrepancy detected!\n');
end

%% 8) Plot
figure('Position', [100 100 1200 400]);

subplot(1,3,1);
semilogx(theta_test, logL_vec, 'b.-', 'LineWidth', 1.5);
xlabel('G (Pa)');
ylabel('log L(¸)');
title('Log-Likelihood');
grid on;

subplot(1,3,2);
semilogx(theta_test, logPrior_vec, 'r.-', 'LineWidth', 1.5);
xlabel('G (Pa)');
ylabel('log P(¸|M)');
title('Log-Prior');
grid on;

subplot(1,3,3);
semilogx(theta_test, logL_vec + logPrior_vec, 'k.-', 'LineWidth', 1.5);
xlabel('G (Pa)');
ylabel('log [L(¸) × P(¸|M)]');
title('Log-Posterior (unnormalized)');
grid on;

end

%% Helper functions
function s = logsumexp_stable(log_vals)
    m = max(log_vals(:));
    if ~isfinite(m)
        s = m;
        return;
    end
    s = m + log(sum(exp(log_vals(:) - m)));
end

function [xmin, xmax] = get_param_bounds_from_priors(modelName, priors)
    % Extract parameter bounds from priors structure
    ranges = priors.ranges;
    
    switch lower(modelName)
        case 'newt'
            xmin = ranges.mu(1);
            xmax = ranges.mu(2);
        case 'nh'
            xmin = ranges.G(1);
            xmax = ranges.G(2);
        case 'kv'
            xmin = [ranges.mu(1), ranges.G(1)];
            xmax = [ranges.mu(2), ranges.G(2)];
        case 'qnh'
            xmin = [ranges.G(1), ranges.alpha(1)];
            xmax = [ranges.G(2), ranges.alpha(2)];
        case 'lm'
            xmin = [ranges.mu(1), ranges.lambda1(1)];
            xmax = [ranges.mu(2), ranges.lambda1(2)];
        case 'qkv'
            xmin = [ranges.mu(1), ranges.G(1), ranges.alpha(1)];
            xmax = [ranges.mu(2), ranges.G(2), ranges.alpha(2)];
        case 'sls'
            xmin = [ranges.mu(1), ranges.G(1), ranges.lambda1(1)];
            xmax = [ranges.mu(2), ranges.G(2), ranges.lambda1(2)];
        otherwise
            error('Unknown model: %s', modelName);
    end
end