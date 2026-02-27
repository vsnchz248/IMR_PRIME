function results = run_bayesian_imr(material_id, varargin)
% RUN_BAYESIAN_IMR
%   High-level driver for Bayesian IMR model selection.
%
%   results = run_bayesian_imr(material_id, 'Name', Value, ...)
%
% Recognized NameValue pairs:
%   'models'     - cell array of model names (e.g., {'newt','kv','qkv',...})
%   'parallel'   - logical (default: false)
%   'gprOpts'    - struct passed to active_integrate_logaware via
%                  bayesian_model_selection_gpr (field .active)
%   'verbose'    - logical (default: true)
%   'saveFigs'   - logical (default: false)  [placeholder, not heavily used]
%   'saveResults'- logical (default: true)
%   'outputDir'  - string, results directory (default: './results')
%
% Output:
%   results - struct used by test_material.m with fields:
%       .per_model      - per-model struct array from bayesian_model_selection_gpr
%       .best_model     - name of best model (lowercase)
%       .best_idx       - index of best model
%       .max_posterior  - P(M_best | D)
%       .theta_MAP      - cell array of MAP parameter vectors per model
%       .elapsed_time   - total wallclock time (seconds)
%       .N_eff, .axis_need, .kernel_norms (if available)

%% Defaults
cfg.models      = {'newt','kv'};
cfg.parallel    = false;
cfg.gprOpts     = struct();
cfg.verbose     = true;
cfg.saveFigs    = false;
cfg.saveResults = true;
cfg.outputDir   = './results';

%% Parse namevalue pairs
if mod(numel(varargin),2) ~= 0
    error('run_bayesian_imr:NameValuePairs','NameValue arguments must come in pairs.');
end

for k = 1:2:numel(varargin)
    name  = lower(varargin{k});
    value = varargin{k+1};
    switch name
        case 'models'
            cfg.models = value;
        case 'parallel'
            cfg.parallel = logical(value);
        case 'gpropts'
            cfg.gprOpts = value;
        case 'verbose'
            cfg.verbose = logical(value);
        case 'savefigs'
            cfg.saveFigs = logical(value);
        case 'saveresults'
            cfg.saveResults = logical(value);
        case 'outputdir'
            cfg.outputDir = value;
        otherwise
            warning('run_bayesian_imr:UnknownOption','Unknown option "%s" ignored.', name);
    end
end

if cfg.verbose
    fprintf('\n??????????????????????????????????????????????????????\n');
    fprintf(  '?   BAYESIAN MODEL SELECTION FOR IMR                 ?\n');
    fprintf(  '??????????????????????????????????????????????????????\n\n');
    fprintf('Material ID: %d\n', material_id);
    fprintf('Models: %s\n', strjoin(cfg.models, ', '));
    fprintf('Parallel: %s\n\n', mat2str(cfg.parallel));
end

tAll = tic;

%% 1. Prepare data
if cfg.verbose
    fprintf('--- Loading Data ---\n\n');
end

prepOpts = struct('verbose', cfg.verbose);
expData  = prepare_data(material_id, prepOpts);

if cfg.verbose
    fprintf('? Loaded: %d trials, %d time points\n', size(expData.Rmatrix,2), size(expData.Rmatrix,1));
    if isfield(expData,'mask') && ~isempty(expData.mask)
        fprintf('? Gated points: %d (%.1f%%)\n\n', ...
            nnz(expData.mask), 100*nnz(expData.mask)/numel(expData.mask));
    end
end

%% 2. Build priors
if cfg.verbose
    fprintf('--- Building Priors ---\n');
end

priorOpts.quiet = ~cfg.verbose;
priors = build_priors(expData, cfg.models, priorOpts);

if cfg.verbose
    if isfield(priors,'kernel_norms')
        fprintf('build_priors: ||A*|| = %.3e, ||B|| = %.3e\n', ...
            priors.kernel_norms.normA, priors.kernel_norms.normB);
    end
    if isfield(priors,'axis_need')
        fprintf('  axis_need: elastic=%.3f, maxwell=%.3f, nonlinear=%.3f\n', ...
            priors.axis_need.elastic, ...
            priors.axis_need.maxwell, ...
            priors.axis_need.nonlinear);
    end
    if isfield(priors,'N_eff')
        fprintf('? N_eff = %d\n', priors.N_eff);
    end
    if isfield(priors,'kernel_norms')
        fprintf('? Kernel norms: ||A*||=%.2e, ||B||=%.2e\n\n', ...
            priors.kernel_norms.normA, priors.kernel_norms.normB);
    end
end

%% 3. Run GPR-based Bayesian model selection
if cfg.verbose
    fprintf('??????????????????????????????????????????????????????\n');
    fprintf(  '?   RUNNING BAYESIAN MODEL SELECTION                 ?\n');
    fprintf(  '?   (Serial: one model at a time)                    ?\n');
    fprintf(  '??????????????????????????????????????????????????????\n');
end

bmOpts = struct();
bmOpts.parallel    = cfg.parallel;
bmOpts.useBICprior = true;
bmOpts.active      = cfg.gprOpts;

results_raw = bayesian_model_selection_gpr(expData, priors, cfg.models, bmOpts);

%% 4. Extract summary + best model
per = results_raw.per_model;
nM  = numel(per);

log10Ev = [per.log10Evidence];
post    = [per.posterior];

[~, best_idx] = max(post);
best_model    = per(best_idx).name;

theta_MAP_cell = cell(1, nM);
for k = 1:nM
    theta_MAP_cell{k} = per(k).mapTheta;
end

elapsed = toc(tAll);

%% 5. Assemble results struct for caller
results = struct();
results.per_model     = per;
results.N_eff         = results_raw.N_eff;
if isfield(results_raw,'axis_need')
    results.axis_need = results_raw.axis_need;
end
if isfield(results_raw,'kernel_norms')
    results.kernel_norms = results_raw.kernel_norms;
end
results.best_model    = best_model;
results.best_idx      = best_idx;
results.max_posterior = post(best_idx);
results.theta_MAP     = theta_MAP_cell;
results.elapsed_time  = elapsed;

%% 6. Print concise summary (only once)
if cfg.verbose
    fprintf('========================================\n');
    fprintf('  Summary\n');
    fprintf('========================================\n');
    fprintf('Model     log10(Evid)   log(Prior)       P(M|D)\n');
    fprintf('----------------------------------------\n');
    for i = 1:nM
        fprintf('%-8s %12.5g %12.5g %12.6f\n', ...
            upper(per(i).name), ...
            per(i).log10Evidence, ...
            per(i).logModelPrior, ...
            per(i).posterior);
    end
    fprintf('========================================\n\n');
end

%% 7. Save results if requested
if cfg.saveResults
    if ~exist(cfg.outputDir,'dir')
        mkdir(cfg.outputDir);
    end
    timestamp = datestr(now,'yyyymmdd_HHMMSS');
    fname     = sprintf('bayesian_imr_material%d_%s.mat', material_id, timestamp);
    fpath     = fullfile(cfg.outputDir, fname);
    save(fpath, 'results', 'results_raw', 'expData', 'priors', 'cfg');
    if cfg.verbose
        fprintf('Results saved to %s\n', fpath);
    end
end

end
