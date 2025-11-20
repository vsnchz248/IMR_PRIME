function OUT = run_imr(varargin)
% run_imr  Single-simulation name/value wrapper around f_imr_fd.
% Usage (example):
%   OUT = run_imr('tvector',tvector,'R0',R0,'Req',Req, ...
%                 'mu',mu,'g',G,'lambda1',0,'alphax',alpha, ...
%                 'stress',2,'radial',2,'bubtherm',1,'medtherm',0, ...
%                 'masstrans',1,'vapor',1,'collapse',0,'Nt',100,'Mt',100);
%
% Required:
%   'tvector' : time vector (1×T or T×1)
%   'R0'      : initial radius
%   'Req'     : equilibrium radius
%
% Optional (defaults shown):
%   Flags:  'stress',2, 'radial',2, 'bubtherm',1, 'medtherm',0, ...
%           'masstrans',1, 'vapor',1, 'collapse',0, 'Nt',100, 'Mt',100
%   Params: 'mu',0, 'g',0, 'lambda1',0, 'lambda2',0, 'alphax',0
%
% Returns:
%   OUT.t, OUT.R, OUT.U, OUT.p   (numeric arrays from f_imr_fd)
%   OUT.meta.flags, OUT.meta.params (echo of inputs)

% -------- Input parsing --------
ip = inputParser; ip.FunctionName = 'run_imr';
% Required-ish
addParameter(ip,'tvector',[],@(x) isnumeric(x) && ~isempty(x));
addParameter(ip,'R0',[],@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'Req',[],@(x) isnumeric(x) && isscalar(x));
% Flags (defaults match your script)
addParameter(ip,'stress',2,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'radial',2,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'bubtherm',1,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'medtherm',0,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'masstrans',1,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'vapor',1,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'collapse',0,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'Nt',100,@(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(ip,'Mt',100,@(x) isnumeric(x) && isscalar(x) && x>0);
% Material/model params (default to 0 if unused by a given model)
addParameter(ip,'mu',0,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'g',0,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'lambda1',0,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'lambda2',0,@(x) isnumeric(x) && isscalar(x));
addParameter(ip,'alphax',0,@(x) isnumeric(x) && isscalar(x));

parse(ip,varargin{:});
S = ip.Results;

% -------- Validation --------
if isempty(S.tvector), error('run_imr:MissingInput','"tvector" is required.'); end
if isempty(S.R0),      error('run_imr:MissingInput','"R0" is required.');      end
if isempty(S.Req),     error('run_imr:MissingInput','"Req" is required.');     end
tvec = S.tvector(:).'; % row vector

% -------- Call forward solver (single run) --------
try
    [t, R, U, p] = f_imr_fd( ...
        'radial',    S.radial, ...
        'stress',    S.stress, ...
        'bubtherm',  S.bubtherm, ...
        'medtherm',  S.medtherm, ...
        'masstrans', S.masstrans, ...
        'vapor',     S.vapor, ...
        'tvector',   tvec, ...
        'r0',        S.R0, ...
        'req',       S.Req, ...
        'mu',        S.mu, ...
        'g',         S.g, ...
        'lambda1',   S.lambda1, ...
        'lambda2',   S.lambda2, ...
        'alphax',    S.alphax, ...
        'collapse',  S.collapse, ...
        'nt',        S.Nt, ...
        'mt',        S.Mt ...
    );
catch ME
    % Add context without hiding the original error
    ME2 = MException('run_imr:ForwardSolverFailed', ...
        'f_imr_fd failed for inputs: stress=%g, mu=%g, g=%g, lambda1=%g, lambda2=%g, alphax=%g', ...
         S.stress, S.mu, S.g, S.lambda1, S.lambda2, S.alphax);
    ME2 = addCause(ME2, ME);
    throw(ME2);
end

% -------- Package output --------
OUT = struct();
OUT.t = t;
OUT.R = R;
OUT.U = U;
OUT.p = p;

OUT.meta.flags = rmfield(S, {'mu','g','lambda1','lambda2','alphax','tvector','R0','Req'});
OUT.meta.params = struct('mu',S.mu,'g',S.g,'lambda1',S.lambda1,'lambda2',S.lambda2,'alphax',S.alphax);
OUT.meta.initial = struct('tvector',tvec,'R0',S.R0,'Req',S.Req);

end
