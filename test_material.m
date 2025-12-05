% TEST_MATERIAL.m
% Test a single material with all models using Bayesian model selection
%
% Simple script to run Bayesian IMR analysis on any material.
% Just change material_id and run!

clear; clc; close all;

%% ========== CONFIGURATION (EDIT THIS) ==========

% Material to test
material_id = 0;  % 0 = synthetic, 1-9 = experimental (e.g., 1 = UM1)

% Parallel processing
use_parallel = true;  % Set to false for serial execution

% Save outputs
save_figures = true;   % Save PNG figures
save_results = true;   % Save .mat file

% Output directory
output_dir = './results';  % Where to save outputs

% Models to test (default: all 7)
models = {'newt', 'nh', 'kv', 'qnh', 'lm', 'qkv', 'sls'};
% Uncomment to test subset:
% models = {'newt', 'nh', 'kv'};  % Just 1D and 2D

% GPR options (adjust for speed vs accuracy)
gprOpts = struct();
gprOpts.maxRounds = 30;        % 15=fast, 30=moderate, 50=precise
gprOpts.maxAddedMult = 140;    % 80=fast, 140=moderate, 200=precise
gprOpts.tolRelCI = 0.04;       % 0.05=fast, 0.04=moderate, 0.02=precise
gprOpts.betaGrid = 0.05:0.05:10;

% ========== RUN ANALYSIS ==========

Create output directory
if save_figures || save_results
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
end

% Print header
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════╗\n');
fprintf('║   BAYESIAN MODEL SELECTION - MATERIAL %d          ║\n', material_id);
fprintf('╚════════════════════════════════════════════════════╝\n');
fprintf('\n');
fprintf('Configuration:\n');
fprintf('  Material ID: %d\n', material_id);
fprintf('  Models: %s\n', strjoin(upper(models), ', '));
fprintf('  Parallel: %s\n', mat2str(use_parallel));
fprintf('  Save figures: %s\n', mat2str(save_figures));
fprintf('  Save results: %s\n', mat2str(save_results));
fprintf('  Output dir: %s\n', output_dir);
fprintf('\n');

% Run analysis
try
    results = run_bayesian_imr(material_id, ...
        'models', models, ...
        'parallel', use_parallel, ...
        'gprOpts', gprOpts, ...
        'verbose', true, ...
        'saveFigs', save_figures, ...
        'saveResults', save_results, ...
        'outputDir', output_dir);
    
    % Success!
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════╗\n');
    fprintf('║   ✓ ANALYSIS COMPLETE                             ║\n');
    fprintf('╚════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    fprintf('Best Model: %s\n', upper(results.best_model));
    fprintf('Posterior: P(M|D) = %.4f\n', results.max_posterior);
    fprintf('MAP Parameters: ');
    fprintf('%.4g ', results.theta_MAP{results.best_idx});
    fprintf('\n');
    fprintf('Total Time: %.1f seconds (%.1f minutes)\n', ...
            results.elapsed_time, results.elapsed_time/60);
    fprintf('\n');
    
    if save_figures
        fprintf('Figures saved to: %s\n', output_dir);
    end
    if save_results
        fprintf('Results saved to: %s\n', output_dir);
    end
    fprintf('\n');
    
catch ME
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════╗\n');
    fprintf('║   ✗ ANALYSIS FAILED                               ║\n');
    fprintf('╚════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    fprintf('\n');
    rethrow(ME);
end