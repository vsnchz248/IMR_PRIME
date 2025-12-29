function axNames = axis_names_for_model(modelName)
% AXIS_NAMES_FOR_MODEL  Return parameter names for each model
%   Order must match parameter vector ¸

    switch lower(modelName)
        case {'newt', 'newtonian'}
            axNames = {'mu'};
            
        case {'nh', 'neo-hookean', 'neohookean'}
            axNames = {'G'};
            
        case {'kv', 'kelvin-voigt', 'kelvinvoigt', 'nhkv'}
            axNames = {'mu', 'G'};
            
        case {'qnh', 'quadratic-nh', 'quadratic'}
            axNames = {'G', 'alpha'};
            
        case {'lm', 'linmax', 'linear-maxwell', 'maxwell'}
            axNames = {'mu', 'lambda1'};
            
        case {'qkv', 'quadratic-kv'}
            axNames = {'mu', 'G', 'alpha'};
            
        case {'sls', 'standard-linear-solid'}
            axNames = {'mu', 'G', 'lambda1'};
            
        otherwise
            error('Unknown model name: %s', modelName);
    end
end