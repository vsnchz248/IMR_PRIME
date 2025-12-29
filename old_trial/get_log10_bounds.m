function bounds = get_log10_bounds(modelName)
% GET_LOG10_BOUNDS  Return log10 parameter bounds for each model
%   Bounds from PAPER Table 2

    % Default bounds (log10 scale)
    mu_log  = [-4, 0];      % 10^-4 to 10^0 Pa·s
    G_log   = [2, 5];       % 10^2 to 10^5 Pa
    lam_log = [-7, -3];     % 10^-7 to 10^-3 s
    alpha_log = [-3, 1];    % 10^-3 to 10^1 (dimensionless)
    
    bounds = struct();
    
    switch lower(modelName)
        case {'newt', 'newtonian'}
            bounds.mu = mu_log;
            
        case {'nh', 'neo-hookean', 'neohookean'}
            bounds.G = G_log;
            
        case {'kv', 'kelvin-voigt', 'kelvinvoigt', 'nhkv'}
            bounds.mu = mu_log;
            bounds.G = G_log;
            
        case {'qnh', 'quadratic-nh', 'quadratic'}
            bounds.G = G_log;
            bounds.alpha = alpha_log;
            
        case {'lm', 'linmax', 'linear-maxwell', 'maxwell'}
            bounds.mu = mu_log;
            bounds.lambda1 = lam_log;
            
        case {'qkv', 'quadratic-kv'}
            bounds.mu = mu_log;
            bounds.G = G_log;
            bounds.alpha = alpha_log;
            
        case {'sls', 'standard-linear-solid'}
            bounds.mu = mu_log;
            bounds.G = G_log;
            bounds.lambda1 = lam_log;
            
        otherwise
            error('Unknown model name: %s', modelName);
    end
end


