function results = bayesian_model_selection_gpr(expData, priors, modelNames, opts)
% BAYESIAN_MODEL_SELECTION_GPR
%   GPR-based Bayesian model selection for IMR using log-aware integration.
%
%   results = bayesian_model_selection_gpr(expData, priors, modelNames, opts)
%
% Inputs
%   expData    - struct from prepare_data / expDataPrep
%   priors     - struct from build_priors (BIMR-style):
%                  .kernel_norms.normA, .kernel_norms.normB
%                  .axis_need.elastic, .axis_need.maxwell, .axis_need.nonlinear
%                  .N_eff  (effective sample count)
%   modelNames - cellstr of model names, e.g. {'NEWT','KV','qKV',...}
%   opts       - struct with fields (all optional):
%                  .parallel    - logical
%                  .useBICprior - logical (default true)
%                  .active      - struct passed to active_integrate_logaware
%
% Output
%   results (scalar struct) with fields:
%       .per_model(k)   - per-model struct:
%             .name           - model name (lowercase)
%             .log10Evidence  - log10 Z_M
%             .logEvidence    - log Z_M  (natural log)
%             .logModelPrior  - log p(M)
%             .posterior      - p(M | D)
%             .mapTheta       - MAP parameter vector (row)
%             .mapNLL         - NLL at MAP
%             .nParams        - parameter count k_M
%             .nEval          - number of NLL evaluations
%             .time_sec       - wall-clock time for this model
%       .N_eff
%       .axis_need
%       .kernel_norms

if nargin < 4, opts = struct(); end
if ~isfield(opts,'parallel'),    opts.parallel    = false; end
if ~isfield(opts,'useBICprior'), opts.useBICprior = true;  end
if ~isfield(opts,'active'),      opts.active      = struct(); end

% Ensure cellstr model list
if ischar(modelNames) || isstring(modelNames)
    modelNames = cellstr(modelNames);
end

nModels = numel(modelNames);

fprintf('========================================\n');
fprintf('  Bayesian Model Selection (GPR-based)\n');
if opts.parallel
    fprintf('  Using PARALLEL processing\n');
else
    fprintf('  Using SERIAL processing\n');
end
fprintf('========================================\n');

% Effective N
if isfield(priors,'N_eff') && ~isempty(priors.N_eff)
    N_eff = priors.N_eff;
else
    if isfield(expData,'mask') && ~isempty(expData.mask)
        N_eff = 2 * nnz(expData.mask);   % BIMR-style effective count
    else
        N_eff = numel(expData.Rmatrix);
    end
end
fprintf('Models: %s\n', strjoin(lower(modelNames), ', '));
fprintf('Effective N: %d\n\n', N_eff);
fprintf('Each model will use its own N0 reference for numerical stability.\n');
fprintf('Prior P(»|M) applied as weight during integration (inside active_integrate_logaware).\n\n');

% Axis needs for p(M)
[e_need, m_need, nl_need] = get_axis_needs_from_priors(priors);

% Storage for per-model results
modelResults(nModels) = struct( ...
    'name',          '', ...
    'log10Evidence', NaN, ...
    'logEvidence',   NaN, ...
    'logModelPrior', NaN, ...
    'posterior',     NaN, ...
    'mapTheta',      [], ...
    'mapNLL',        NaN, ...
    'nParams',       NaN, ...
    'nEval',         NaN, ...
    'time_sec',      NaN );

logZ_vec      = -inf(1, nModels);
logPrior_vec  = -inf(1, nModels);

% ========= main loop over models =========
for i = 1:nModels
    modelName = lower(strtrim(modelNames{i}));
    fprintf('--- Model %d/%d: %s ---\n', i, nModels, upper(modelName));

    tStart = tic;
    try
        [log10Z, mapTheta, mapNLL, k_params, nEval] = ...
            process_single_model(modelName, expData, priors, N_eff, opts.active);
    catch ME
        warning('Model %s failed in active_integrate_logaware: %s', modelName, ME.message);
        log10Z   = -inf;
        mapTheta = [];
        mapNLL   = NaN;
        k_params = numel(axis_names_for_model(modelName));
        nEval    = NaN;
    end
    tElapsed = toc(tStart);

    logZ = log10Z * log(10);      % convert to natural log

    % Model prior log p(M) (BIC + axis-need factor)
    logpM = compute_model_log_prior(modelName, k_params, N_eff, ...
                                    e_need, m_need, nl_need, opts.useBICprior);

    modelResults(i).name          = modelName;
    modelResults(i).log10Evidence = log10Z;
    modelResults(i).logEvidence   = logZ;
    modelResults(i).logModelPrior = logpM;
    modelResults(i).mapTheta      = mapTheta;
    modelResults(i).mapNLL        = mapNLL;
    modelResults(i).nParams       = k_params;
    modelResults(i).nEval         = nEval;
    modelResults(i).time_sec      = tElapsed;

    logZ_vec(i)     = logZ;
    logPrior_vec(i) = logpM;

    fprintf('Computed log10(evidence) = %.5g, log p(M) = %.5g\n\n', log10Z, logpM);
end

% ========= posterior over models =========
x = logZ_vec + logPrior_vec;
Fmask = isfinite(x);
if ~any(Fmask)
    post = ones(1, nModels) / nModels;
else
    Ztot = safe_logsumexp(x(Fmask));
    post = zeros(1, nModels);
    post(Fmask) = exp(x(Fmask) - Ztot);
end

for i = 1:nModels
    modelResults(i).posterior = post(i);
end

% Final summary print
fprintf('========================================\n');
fprintf('  Summary\n');
fprintf('========================================\n');
fprintf('Model     log10(Evid)   log(Prior)       P(M|D)\n');
fprintf('----------------------------------------\n');
for i = 1:nModels
    fprintf('%-8s %12.5g %12.5g %12.6f\n', ...
        upper(modelResults(i).name), ...
        modelResults(i).log10Evidence, ...
        modelResults(i).logModelPrior, ...
        modelResults(i).posterior);
end
fprintf('========================================\n');

% ========= Wrap into scalar struct =========
results = struct();
results.per_model = modelResults;
results.N_eff     = N_eff;

if isfield(priors,'axis_need')
    results.axis_need = priors.axis_need;
end
if isfield(priors,'kernel_norms')
    results.kernel_norms = priors.kernel_norms;
end

end

%% ==================== Single-model worker ====================

function [log10Z, mapTheta, mapNLL, k_params, nEval] = ...
    process_single_model(modelName, expData, priors, N_eff, activeOpts)
% PROCESS_SINGLE_MODEL  Run active_integrate_logaware for one model.

modelKey = normalize_model_key(modelName);

% Try to find a matching field in priors (case-insensitive)
modelField = find_model_field(priors, modelKey);

xmin = []; xmax = [];

if ~isempty(modelField) && ...
        isfield(priors.(modelField),'grid') && ...
        isfield(priors.(modelField),'prior')

    % --- If an old BIMR-style grid exists, use it to infer bounds ---
    G = priors.(modelField).grid;
    axesCell = grid_struct_to_axes(G);
    d = numel(axesCell);
    xmin = zeros(1,d);
    xmax = zeros(1,d);
    for j = 1:d
        v = axesCell{j};
        xmin(j) = min(v(:));
        xmax(j) = max(v(:));
    end
else
    % --- Otherwise: use hard-coded model-specific parameter boxes ---
    [xmin, xmax] = get_param_box_for_model(modelName);
    fprintf('Using built-in param box for %s:\n', modelName);
    fprintf('  xmin = ['); fprintf(' %.3g', xmin); fprintf(' ]\n');
    fprintf('  xmax = ['); fprintf(' %.3g', xmax); fprintf(' ]\n');
end

k_params = numel(xmin);

% NLL(») including any parameter prior terms (handled inside imr_nll_with_prior_matrix)
nllFcn = @(theta) imr_nll_with_prior_matrix(theta, modelName, expData, priors, struct('N_eff', N_eff));

if nargin < 5 || isempty(activeOpts), activeOpts = struct(); end

% Call your existing active_integrate_logaware:
%   out = active_integrate_logaware(funNLL, xmin, xmax, opts, modelName, priors)
res = active_integrate_logaware(nllFcn, xmin, xmax, activeOpts, modelName, priors);

% Expected fields from your current implementation:
if isfield(res,'log10I_mean')
    log10Z = res.log10I_mean;
elseif isfield(res,'log10I_H')
    log10Z = res.log10I_H;
elseif isfield(res,'log10I')
    log10Z = res.log10I;
else
    error('active_integrate_logaware result lacks log10I_mean/log10I_H/log10I.');
end

% Your current active_integrate_logaware does not yet track MAP explicitly
mapTheta = [];
mapNLL   = NaN;
nEval    = NaN;

end

%% ==================== Model prior construction ====================

function logpM = compute_model_log_prior(modelName, k_params, N_eff, ...
                                         e_need, m_need, nl_need, useBIC)
% BIC-style + axis-need factor

f_data    = model_data_need_factor(modelName, e_need, m_need, nl_need);
logpMdata = log(max(f_data, 1e-12));

if useBIC
    logpM_BIC = -0.5 * k_params * log(max(N_eff, 1));
    logpM     = logpM_BIC + logpMdata;
else
    logpM     = logpMdata;
end
end

function f = model_data_need_factor(modelName, e_need, m_need, nl_need)
m = lower(strtrim(modelName));
switch m
    case {'newtonian','newt'}
        f = 1;
    case 'nh'
        f = e_need;
    case {'nhkv','kv'}
        f = e_need;
    case 'qnh'
        f = e_need * nl_need;
    case 'qkv'
        f = e_need * nl_need;
    case {'linmax','max'}
        f = m_need;
    case 'sls'
        f = (e_need^2) * m_need;
    otherwise
        f = 1;
end
f = max(min(f,1),0);
end

function [e_need, m_need, nl_need] = get_axis_needs_from_priors(priors)
e_need  = 1;
m_need  = 1;
nl_need = 0;
if isfield(priors,'axis_need')
    if isfield(priors.axis_need,'elastic')
        e_need = priors.axis_need.elastic;
    end
    if isfield(priors.axis_need,'maxwell')
        m_need = priors.axis_need.maxwell;
    end
    if isfield(priors.axis_need,'nonlinear')
        nl_need = priors.axis_need.nonlinear;
    end
end
e_need  = max(min(e_need ,1),0);
m_need  = max(min(m_need ,1),0);
nl_need = max(min(nl_need,1),0);
end

%% ==================== Param box helper ====================

function [xmin, xmax] = get_param_box_for_model(modelName)
% NOTE: these are *continuous* parameter ranges in the SAME space that
% imr_nll_with_prior_matrix expects (usually dimensional or log-aware).
% They are deliberately broad but can be tightened if you know your
% synthetic/experimental ranges precisely.

m = lower(strtrim(modelName));

switch m
    case {'newt','newtonian'}
        % Single viscosity-like parameter, MAP ~ 0.07 in your synthetic runs
        xmin = [1e-4];
        xmax = [1];        % covers ~1e-4 to 1 (log-aware map will handle this)

    case {'kv','nhkv'}
        % KelvinVoigt: [mu, G]
        % mu around ~1e-21e-1, G around ~1e21e4 in your synthetic example
        xmin = [1e-4, 1];   % [mu_min, G_min]
        xmax = [1,    1e5]; % [mu_max, G_max]

    case 'nh'
        % Neo-Hookean: single elastic modulus G
        xmin = [1];
        xmax = [1e5];

    case 'qnh'
        % Quasi-neo-Hookean: [G, alpha]
        xmin = [1,  0.1];
        xmax = [1e5, 10];

    case 'qkv'
        % Quasi-KelvinVoigt: [mu, G, alpha]
        xmin = [1e-4, 1,   0.1];
        xmax = [1,    1e5, 10];

    case {'linmax','max'}
        % Linear Maxwell: [mu, lambda1]
        xmin = [1e-4, 1e-6];
        xmax = [1,    1e-1];

    case 'sls'
        % Standard linear solid: [mu, G, lambda1]
        xmin = [1e-4, 1,   1e-6];
        xmax = [1,    1e5, 1e-1];

    otherwise
        % Generic fallback if something unexpected sneaks in
        names = axis_names_for_model(m);
        d = max(1, numel(names));
        xmin = 1e-4 * ones(1,d);
        xmax = 1    * ones(1,d);
end
end

%% ==================== Misc helpers ====================

function key = normalize_model_key(modelLower)
switch lower(strtrim(modelLower))
    case {'newt','newtonian'}, key = 'Newt';
    case 'nh',                 key = 'NH';
    case {'nhkv','kv'},        key = 'KV';
    case 'qnh',                key = 'qNH';
    case {'max','linmax'},     key = 'LM';
    case 'qkv',                key = 'qKV';
    case 'sls',                key = 'SLS';
    otherwise,                 key = '';
end
end

function fieldName = find_model_field(priors, key)
fieldName = '';
if ~isstruct(priors), return; end
f = fieldnames(priors);
for i = 1:numel(f)
    if strcmpi(f{i}, key)
        fieldName = f{i};
        return;
    end
end
end

function axesCell = grid_struct_to_axes(G)
fields = fieldnames(G);
order  = {};
if isfield(G,'mu'),      order{end+1} = 'mu';      end %#ok<AGROW>
if isfield(G,'G'),       order{end+1} = 'G';       end
if isfield(G,'lambda1'), order{end+1} = 'lambda1'; end
if isfield(G,'alpha'),   order{end+1} = 'alpha';   end
for k = 1:numel(fields)
    if ~ismember(fields{k}, order)
        order{end+1} = fields{k}; %#ok<AGROW>
    end
end
axesCell = cell(1, numel(order));
for k = 1:numel(order)
    axesCell{k} = G.(order{k});
end
end

function names = axis_names_for_model(modelName)
switch lower(strtrim(modelName))
    case 'qkv',      names = {'mu','g','alpha'};
    case 'sls',      names = {'mu','g','lambda1'};
    case {'nhkv','kv'}, names = {'mu','g'};
    case 'qnh',      names = {'g','alpha'};
    case 'linmax',   names = {'mu','lambda1'};
    case 'nh',       names = {'g'};
    case {'newt','newtonian'}, names = {'mu'};
    otherwise, names = {};
end
end

function s = safe_logsumexp(a)
if isempty(a) || all(~isfinite(a))
    s = -inf;
    return;
end
amax = max(a(:));
s    = amax + log(sum(exp(a(:) - amax)));
end
