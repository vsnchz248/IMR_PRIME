function results = run_bayesian_imr(material_id, varargin)
% RUN_BAYESIAN_IMR - Main function for Bayesian model selection on IMR data
%
% Syntax:
%   results = run_bayesian_imr(material_id)
%   results = run_bayesian_imr(material_id, Name, Value)
%
% Inputs:
%   material_id - Material ID (0 = synthetic, 1-9 = experimental)
%
% Optional Name-Value Pairs:
%   'models'      - Cell array of model names (default: all 7 models)
%   'parallel'    - Use parallel processing (default: true)
%   'numWorkers'  - Number of parallel workers (default: auto)
%   'gprOpts'     - GPR options struct (default: production settings)
%   'verbose'     - Display detailed output (default: true)
%   'saveFigs'    - Save figures to disk (default: false)
%   'saveResults' - Save results to .mat file (default: false)
%   'outputDir'   - Output directory for saved files (default: pwd)
%
% Outputs:
%   results - Structure containing:
%       .model_posterior - Posterior probability P(M|D) for each model
%       .log_evidence    - Log evidence for each model
%       .theta_MAP       - MAP parameters for each model
%       .best_model      - Name of most probable model
%       .best_idx        - Index of most probable model
%       .max_posterior   - Posterior probability of best model
%       .elapsed_time    - Total computation time (seconds)
%       .material_id     - Material ID tested
%       .models          - Model names
%       .expData         - Experimental data structure
%       .priors          - Prior structure
%
% Example:
%   % Test synthetic data with all models, parallel
%   results = run_bayesian_imr(0);
%
%   % Test specific models on material 1
%   results = run_bayesian_imr(1, 'models', {'newt','nh','kv'});
%
%   % Run with custom GPR settings, save output
%   gpr = struct('maxRounds', 50, 'tolRelCI', 0.02);
%   results = run_bayesian_imr(2, 'gprOpts', gpr, 'saveFigs', true);
%
%   % Run serially (no parallel)
%   results = run_bayesian_imr(0, 'parallel', false);

% Parse inputs
p = inputParser;
addRequired(p, 'material_id', @isnumeric);
addParameter(p, 'models', {'newt','nh','kv','qnh','lm','qkv','sls'}, @iscell);
addParameter(p, 'parallel', true, @islogical);
addParameter(p, 'numWorkers', [], @(x) isempty(x) || isnumeric(x));
addParameter(p, 'gprOpts', [], @(x) isempty(x) || isstruct(x));
addParameter(p, 'verbose', true, @islogical);
addParameter(p, 'saveFigs', false, @islogical);
addParameter(p, 'saveResults', false, @islogical);
addParameter(p, 'outputDir', pwd, @ischar);
parse(p, material_id, varargin{:});

models = p.Results.models;
use_parallel = p.Results.parallel;
num_workers = p.Results.numWorkers;
gprOpts = p.Results.gprOpts;
verbose = p.Results.verbose;
save_figs = p.Results.saveFigs;
save_results = p.Results.saveResults;
output_dir = p.Results.outputDir;

% Default GPR options (production settings)
if isempty(gprOpts)
    gprOpts = struct();
    gprOpts.maxRounds = 30;
    gprOpts.maxAddedMult = 140;
    gprOpts.tolRelCI = 0.04;
    gprOpts.betaGrid = 0.05:0.05:10;
end

% Print header
if verbose
    fprintf('\n╔════════════════════════════════════════════════════╗\n');
    fprintf('║   BAYESIAN MODEL SELECTION FOR IMR                ║\n');
    fprintf('╚════════════════════════════════════════════════════╝\n');
    fprintf('\nMaterial ID: %d\n', material_id);
    fprintf('Models: %s\n', strjoin(upper(models), ', '));
    fprintf('Parallel: %s\n', mat2str(use_parallel));
end

%% Setup parallel pool
if use_parallel
    if verbose
        fprintf('\n--- Setting Up Parallel Computing ---\n');
    end
    
    % Get current pool
    pool = gcp('nocreate');
    
    if isempty(pool)
        % No pool exists, create one
        if isempty(num_workers)
            % Auto-detect number of workers
            pool = parpool();
            if verbose
                fprintf('✓ Started parallel pool with %d workers\n', pool.NumWorkers);
            end
        else
            % Use specified number
            pool = parpool(num_workers);
            if verbose
                fprintf('✓ Started parallel pool with %d workers\n', num_workers);
            end
        end
    else
        % Pool exists
        if verbose
            fprintf('✓ Using existing parallel pool (%d workers)\n', pool.NumWorkers);
        end
    end
else
    if verbose
        fprintf('\n--- Running Serially (no parallel) ---\n');
    end
end

%% Load Data
if verbose
    fprintf('\n--- Loading Data ---\n');
end

opts_data = struct();
opts_data.verbose = false;
expData = prepare_data(material_id, opts_data);

if verbose
    fprintf('✓ Loaded: %d trials, %d time points\n', ...
            size(expData.Rmatrix,2), size(expData.Rmatrix,1));
    fprintf('✓ Gated points: %d (%.1f%%)\n', ...
            nnz(expData.mask), 100*nnz(expData.mask)/numel(expData.mask));
end

%% Build Priors
if verbose
    fprintf('\n--- Building Priors ---\n');
end

opts_prior = struct();
opts_prior.verbose = false;
opts_prior.N_eff = 2 * nnz(expData.mask);
priors = build_priors(expData, opts_prior);

if verbose
    fprintf('✓ N_eff = %d\n', priors.N_eff);
    fprintf('✓ Kernel norms: ||A*||=%.2e, ||B||=%.2e\n', ...
            priors.kernel_norms.normA, priors.kernel_norms.normB);
end

%% Run Bayesian Model Selection
if verbose
    fprintf('\n╔════════════════════════════════════════════════════╗\n');
    fprintf('║   RUNNING BAYESIAN MODEL SELECTION                ║\n');
    if use_parallel
        fprintf('║   (Parallel: %d models simultaneously)            ║\n', ...
                min(numel(models), pool.NumWorkers));
    else
        fprintf('║   (Serial: one model at a time)                   ║\n');
    end
    fprintf('╚════════════════════════════════════════════════════╝\n');
end

opts = struct();
opts.gprOpts = gprOpts;
opts.verbose = verbose;
opts.parallel = use_parallel;  % Pass parallel flag to selection function

tic;
results_raw = bayesian_model_selection_gpr(expData, priors, models, opts);
elapsed = toc;

%% Package Results
results = results_raw;
results.elapsed_time = elapsed;
results.material_id = material_id;
results.models = models;
results.expData = expData;
results.priors = priors;
results.gprOpts = gprOpts;
results.parallel_used = use_parallel;

%% Display Results
if verbose
    display_results(results, models);
end

%% Validation
if verbose
    validate_results(results, models);
end

%% Save Results
if save_results
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('bayesian_imr_material%d_%s.mat', material_id, timestamp);
    filepath = fullfile(output_dir, filename);
    save(filepath, 'results');
    if verbose
        fprintf('\n✓ Saved results to: %s\n', filepath);
    end
end

%% Create Figures
if verbose || save_figs
    figs = create_figures(results, models);
    
    if save_figs
        % Save figures
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        
        % Figure 1: Model comparison
        fig1_name = sprintf('model_comparison_material%d_%s.png', material_id, timestamp);
        fig1_path = fullfile(output_dir, fig1_name);
        saveas(figs(1), fig1_path);
        if verbose
            fprintf('✓ Saved: %s\n', fig1_name);
        end
        
        % Figure 2: Parameters
        fig2_name = sprintf('map_parameters_material%d_%s.png', material_id, timestamp);
        fig2_path = fullfile(output_dir, fig2_name);
        saveas(figs(2), fig2_path);
        if verbose
            fprintf('✓ Saved: %s\n', fig2_name);
        end
    end
end

if verbose
    fprintf('\n');
end

end % main function


%% Helper Functions

function display_results(results, models)
    fprintf('\n╔════════════════════════════════════════════════════╗\n');
    fprintf('║   RESULTS                                         ║\n');
    fprintf('╚════════════════════════════════════════════════════╝\n');
    
    fprintf('\nModel Posteriors (sorted by probability):\n');
    [sorted_post, sort_idx] = sort(results.model_posterior, 'descend');
    for i = 1:numel(models)
        idx = sort_idx(i);
        fprintf('  %d. %s: P(M|D) = %.4f\n', i, upper(models{idx}), sorted_post(i));
    end
    
    fprintf('\nLog10 Evidence:\n');
    for i = 1:numel(models)
        fprintf('  %s: %.6g\n', upper(models{i}), results.log_evidence(i)/log(10));
    end
    
    fprintf('\nMAP Parameters:\n');
    param_dims = [1 1 2 2 2 3 3];
    for i = 1:numel(models)
        fprintf('  %s (%dD): ', upper(models{i}), param_dims(i));
        fprintf('%.4g ', results.theta_MAP{i});
        fprintf('\n');
    end
    
    fprintf('\nBest Model: %s (P = %.4f)\n', ...
            upper(results.best_model), results.max_posterior);
    
    fprintf('\nComputation Time: %.1f sec (%.1f min)\n', ...
            results.elapsed_time, results.elapsed_time/60);
end


function validate_results(results, models)
    fprintf('\n╔════════════════════════════════════════════════════╗\n');
    fprintf('║   VALIDATION CHECKS                               ║\n');
    fprintf('╚════════════════════════════════════════════════════╝\n');
    
    all_pass = true;
    
    % Check 1: Posteriors sum to 1
    sum_post = sum(results.model_posterior);
    if abs(sum_post - 1.0) < 1e-10
        fprintf('✓ Model posteriors sum to 1.0\n');
    else
        fprintf('✗ Model posteriors sum to %.10f (should be 1.0)\n', sum_post);
        all_pass = false;
    end
    
    % Check 2: All evidences finite
    if all(isfinite(results.log_evidence))
        fprintf('✓ All log evidences are finite\n');
    else
        fprintf('✗ Some log evidences are not finite\n');
        all_pass = false;
    end
    
    % Check 3: MAP parameters reasonable
    param_ok = true;
    for i = 1:numel(models)
        theta = results.theta_MAP{i};
        if any(~isfinite(theta)) || any(theta <= 0)
            fprintf('✗ %s: MAP parameters not positive/finite\n', upper(models{i}));
            param_ok = false;
            all_pass = false;
        end
    end
    if param_ok
        fprintf('✓ All MAP parameters are positive and finite\n');
    end
    
    % Check 4: Parameter dimensions
    expected_dims = [1 1 2 2 2 3 3];
    dim_ok = true;
    for i = 1:numel(models)
        if numel(results.theta_MAP{i}) ~= expected_dims(i)
            fprintf('✗ %s should have %d parameters, has %d\n', ...
                    upper(models{i}), expected_dims(i), numel(results.theta_MAP{i}));
            dim_ok = false;
            all_pass = false;
        end
    end
    if dim_ok
        fprintf('✓ All models have correct parameter dimensions\n');
    end
    
    if ~all_pass
        fprintf('\n⚠ Some validation checks failed\n');
    end
end


function figs = create_figures(results, models)
    % Figure 1: Model comparison
    figs(1) = figure('Position', [100 100 1200 500]);
    
    subplot(1,3,1);
    bar(results.model_posterior);
    set(gca, 'XTickLabel', upper(models), 'XTickLabelRotation', 45);
    ylabel('Posterior Probability P(M|D)');
    xlabel('Model');
    title(sprintf('Model Posterior (Material %d)', results.material_id));
    grid on;
    ylim([0 max(results.model_posterior)*1.1]);
    
    subplot(1,3,2);
    bar(results.log_evidence / log(10));
    set(gca, 'XTickLabel', upper(models), 'XTickLabelRotation', 45);
    ylabel('log_{10} Evidence');
    xlabel('Model');
    title('Model Evidence');
    grid on;
    
    subplot(1,3,3);
    [sorted_post, sort_idx] = sort(results.model_posterior, 'descend');
    bar([sorted_post(1:min(3,end)); 0]);
    labels = upper(models(sort_idx(1:min(3,end))));
    if numel(labels) < 3
        labels = [labels, repmat({''}, 1, 3-numel(labels))];
    end
    labels{end+1} = '';
    set(gca, 'XTickLabel', labels);
    ylabel('Posterior Probability');
    title('Top Models');
    grid on;
    ylim([0 1]);
    
    % Figure 2: Parameter comparison
    figs(2) = figure('Position', [100 100 1000 600]);
    param_dims = [1 1 2 2 2 3 3];
    for i = 1:numel(models)
        subplot(3,3,i);
        theta = results.theta_MAP{i};
        bar(theta);
        title(sprintf('%s (P=%.3f)', upper(models{i}), results.model_posterior(i)));
        ylabel('Parameter Value');
        xlabel('Parameter Index');
        grid on;
        if i <= numel(models)
            xlim([0.5 param_dims(i)+0.5]);
        end
    end
    sgtitle(sprintf('MAP Parameters (Material %d)', results.material_id));
end