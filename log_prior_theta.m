function logp = log_prior_theta(theta, modelName, priors)
% LOG_PRIOR_THETA  Continuous log prior log p(theta | M) via interpolation
%
% Inputs:
%   theta     - parameter vector (in physical units: mu, G, lambda1, alpha)
%   modelName - string: 'newtonian','nh','nhkv','qnh','qkv','linmax','sls',...
%   priors    - struct from build_model_priors()
%
% Output:
%   logp      - scalar log prior value (−Inf if theta out of range)

tiny = 1e-300;

modelKey = normalize_model_key(lower(strtrim(modelName)));  % reuse BIMR mapping

if ~isfield(priors, modelKey) || ~isfield(priors.(modelKey),'prior') ...
        || ~isfield(priors.(modelKey),'grid')
    error('log_prior_theta: model "%s" not found in priors.', modelKey);
end

P = priors.(modelKey).prior;
Gstruct = priors.(modelKey).grid;

% Extract axis vectors depending on model dimensionality
switch modelKey
    case 'newt'   % 1D: mu
        mu = theta(1);
        mu_axis = squeeze(Gstruct.mu(:));        % 4096x1
        % Guard range
        if mu < min(mu_axis) || mu > max(mu_axis)
            logp = -Inf; return;
        end
        % 1D linear interpolation
        p = interp1(mu_axis, P(:), mu, 'linear', 0);

    case 'nh'    % 1D: G
        G = theta(1);
        G_axis = squeeze(Gstruct.G(:));
        if G < min(G_axis) || G > max(G_axis)
            logp = -Inf; return;
        end
        p = interp1(G_axis, P(:), G, 'linear', 0);

    case 'kv'    % 2D: mu, G
        mu = theta(1); G = theta(2);
        % Recover axes from ndgrid layout
        mu_axis = squeeze(Gstruct.mu(:,1));   % size nMu x 1
        G_axis  = squeeze(Gstruct.G(1,:));    % size 1 x nG
        if mu < min(mu_axis) || mu > max(mu_axis) || ...
           G  < min(G_axis)  || G  > max(G_axis)
            logp = -Inf; return;
        end
        p = interpn(mu_axis, G_axis, P, mu, G, 'linear', 0);

    case 'qnh'   % 2D: G, alpha
        G      = theta(1); 
        alpha  = theta(2);
        G_axis     = squeeze(Gstruct.G(:,1));
        alpha_axis = squeeze(Gstruct.alpha(1,:));
        if G < min(G_axis) || G > max(G_axis) || ...
           alpha < min(alpha_axis) || alpha > max(alpha_axis)
            logp = -Inf; return;
        end
        p = interpn(G_axis, alpha_axis, P, G, alpha, 'linear', 0);

    case 'qkv'   % 3D: mu, G, alpha
        mu    = theta(1); 
        G     = theta(2); 
        alpha = theta(3);
        mu_axis    = squeeze(Gstruct.mu(:,1,1));
        G_axis     = squeeze(Gstruct.G(1,:,1));
        alpha_axis = squeeze(Gstruct.alpha(1,1,:));
        if mu < min(mu_axis) || mu > max(mu_axis) || ...
           G  < min(G_axis)  || G  > max(G_axis)  || ...
           alpha < min(alpha_axis) || alpha > max(alpha_axis)
            logp = -Inf; return;
        end
        p = interpn(mu_axis, G_axis, alpha_axis, P, mu, G, alpha, 'linear', 0);

    case 'lm'    % 2D: mu, lambda1
        mu      = theta(1); 
        lambda1 = theta(2);
        mu_axis  = squeeze(Gstruct.mu(:,1));
        l1_axis  = squeeze(Gstruct.lambda1(1,:));
        if mu < min(mu_axis) || mu > max(mu_axis) || ...
           lambda1 < min(l1_axis) || lambda1 > max(l1_axis)
            logp = -Inf; return;
        end
        p = interpn(mu_axis, l1_axis, P, mu, lambda1, 'linear', 0);

    case 'sls'   % 3D: mu, G, lambda1
        mu      = theta(1);
        G       = theta(2);
        lambda1 = theta(3);
        mu_axis = squeeze(Gstruct.mu(:,1,1));
        G_axis  = squeeze(Gstruct.G(1,:,1));
        l1_axis = squeeze(Gstruct.lambda1(1,1,:));
        if mu < min(mu_axis) || mu > max(mu_axis) || ...
           G  < min(G_axis)  || G  > max(G_axis)  || ...
           lambda1 < min(l1_axis) || lambda1 > max(l1_axis)
            logp = -Inf; return;
        end
        p = interpn(mu_axis, G_axis, l1_axis, P, mu, G, lambda1, 'linear', 0);

    otherwise
        error('log_prior_theta: unknown modelKey "%s".', modelKey);
end

p = max(p, tiny);
logp = log(p);

end

function key = normalize_model_key(modelLower)
% Same mapping you used in BIMR
switch lower(modelLower)
    case 'newt', key = 'Nnwt';
    case 'nh',                 key = 'nh';
    case 'kv'
        key = 'kv';
    case 'qnh',                key = 'qnh';
    case 'qkv',                key = 'qkv';
    case 'lm'
        key = 'lm';
    case 'sls',                key = 'sls';
    otherwise,                 error('Unknown model name: %s', modelLower);
end
end
