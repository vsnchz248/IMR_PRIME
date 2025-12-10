function priors = build_priors(expData, models, opts)
% BUILD_PRIORS  Construct simple priors for Bayesian IMR model selection.
%
% Usage:
%   priors = build_priors(expData, models)
%   priors = build_priors(expData, models, opts)
%
% Inputs:
%   expData - struct from prepare_data()
%   models  - cell array of model names, e.g. {'newt','kv','nh',...}
%   opts    - (optional) struct:
%               .quiet  - suppress console output (default: false)
%
% Outputs:
%   priors - struct with fields:
%       .N_eff        - effective data count
%       .kernel_norms - struct with normA, normB
%       .axis_need    - struct with elastic, maxwell, nonlinear
%       .models       - cell array of model names
%       .modelPrior   - handle: log p(M) with BIC-like penalty
%       .paramPrior   - handle: log p(theta | M) (currently flat)

    %% ---------- Parse inputs ----------
    if nargin < 2 || isempty(models)
        models = {'newt','kv','nh','qnh','lm','qkv','sls'};
    end
    models = cellstr(models);

    if nargin < 3
        opts = struct();
    end
    if ~isfield(opts, 'quiet'), opts.quiet = false; end

    %% ---------- Effective sample size N_eff ----------
    % Use 2 x (number of gated points), matching BIMR-style (R and Rdot)
    if isfield(expData, 'mask') && ~isempty(expData.mask)
        gated_pts = nnz(expData.mask);
    else
        [nT, nJ] = size(expData.Rmatrix);
        gated_pts = nT * nJ;
    end
    N_eff = 2 * gated_pts;

    %% ---------- Kernel norms from strain / strain-rate ----------
    % Rough analogs to ||A*|| and ||B|| just for scaling / diagnostics
    if isfield(expData, 'strain') && isfield(expData, 'strainRate')
        epsH    = expData.strain;
        epsHdot = expData.strainRate;
    else
        epsH    = zeros(size(expData.Rmatrix));
        epsHdot = zeros(size(expData.Rmatrix));
    end

    if isfield(expData, 'mask') && ~isempty(expData.mask)
        mask = expData.mask;
        epsH    = epsH(mask);
        epsHdot = epsHdot(mask);
    else
        epsH    = epsH(:);
        epsHdot = epsHdot(:);
    end

    epsH    = epsH(isfinite(epsH));
    epsHdot = epsHdot(isfinite(epsHdot));

    if isempty(epsH)
        normA = 1.0;
    else
        normA = sqrt(mean(epsH.^2));
    end

    if isempty(epsHdot)
        normB = 1.0;
    else
        normB = sqrt(mean(epsHdot.^2));
    end

    kernel_norms = struct('normA', normA, 'normB', normB);

    %% ---------- Axis "need" diagnostics (purely informational) ----------
    % Simple normalized measures in [0,1]
    sumAB = normA + normB + eps;
    axis_elastic  = normA / sumAB;
    axis_maxwell  = normB / sumAB;

    % Nonlinearity: based on max strain magnitude (squashed to [0,1])
    if isempty(epsH)
        axis_nonlinear = 0.0;
    else
        max_strain = max(abs(epsH));
        axis_nonlinear = 1 - exp(-max_strain);  % monotonically increasing, saturates near 1
    end

    axis_need = struct( ...
        'elastic',  axis_elastic, ...
        'maxwell',  axis_maxwell, ...
        'nonlinear',axis_nonlinear);

    if ~opts.quiet
        fprintf('build_priors: ||A*|| = %.3e, ||B|| = %.3e\n', normA, normB);
        fprintf('  axis_need: elastic=%.3f, maxwell=%.3f, nonlinear=%.3f\n', ...
                axis_elastic, axis_maxwell, axis_nonlinear);
    end

    %% ---------- Model prior: BIC-like penalty ----------
    % log p(M)  ~  -0.5 * k * log(N_eff)
    %
    % We allow either:
    %   priors.modelPrior(modelName, k_params)
    % or
    %   priors.modelPrior(modelName, [])   -> k inferred from name
    %
    priors.N_eff        = N_eff;
    priors.kernel_norms = kernel_norms;
    priors.axis_need    = axis_need;
    priors.models       = models;

    priors.modelPrior = @(modelName, k_params) model_prior_fn(modelName, k_params, N_eff);

    %% ---------- Parameter prior: currently flat (log-uniform over region) ----------
    % Here we just return 0 for any theta, i.e. a proper prior is enforced
    % implicitly by the integration bounds passed into active_integrate_logaware.
    %
    % You can replace this later with model-specific priors if needed.
    priors.paramPrior = @(modelName, theta) param_prior_flat(modelName, theta); %#ok<NASGU>

end

% =======================================================================
% Local helper: BIC-like model prior
% =======================================================================
function logp = model_prior_fn(modelName, k_params, N_eff)
    if nargin < 2 || isempty(k_params)
        k_params = default_num_params(modelName);
    end

    % BIC-like: log p(M)  -0.5 * k * log(N_eff)
    logp = -0.5 * k_params * log(max(N_eff, 1));

    % You could normalize across models if desired, but for model comparison
    % differences in logp are what matter, so a common additive constant is irrelevant.
end

% =======================================================================
% Local helper: default number of parameters per model
% =======================================================================
function k = default_num_params(modelName)
    switch lower(strtrim(modelName))
        case 'newt'
            k = 1;   % mu
        case 'kv'
            k = 2;   % mu, G
        case 'nh'
            k = 1;   % G
        case 'qnh'
            k = 2;   % G, alpha
        case 'lm'
            k = 3;   % mu, G, lambda1
        case 'qkv'
            k = 3;   % mu, G, alpha
        case 'sls'
            k = 3;   % e.g., G1, G2, tau
        otherwise
            k = 2;   % fallback
    end
end

% =======================================================================
% Local helper: flat log-prior over theta
% =======================================================================
function logp = param_prior_flat(~, ~)
    % Flat log-prior (uniform in the region of integration)
    % You can later replace this with a model- and parameter-dependent prior.
    logp = 0;
end
