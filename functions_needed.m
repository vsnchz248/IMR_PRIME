
% Parameter ranges (columns: [min, max])
ranges = [1e-4, 1;    % mu
          1e2,  1e5;  % G
          1e-3, 1e1]; % alpha

% Suppose we have 3 samples (rows) for 3 parameters (columns)
X = [linspace(log10(ranges(1,1)),log10(ranges(1,2)),16)',...
     linspace(log10(ranges(2,1)),log10(ranges(2,2)),16)',...
     linspace(log10(ranges(3,1)),log10(ranges(3,2)),16)'];

X = 10.^X;

% Normalize all samples in log space
X_norm = normalizeParamsLog(X, ranges)

% Denormalize back
X_back = denormalizeParamsLog(X_norm, ranges)

function X_norm = normalizeParamsLog(X, ranges)
%NORMALIZEPARAMSLOG Normalize parameters to [0,1] in log10 space.
%   X       : n×d matrix of parameter values
%   ranges  : d×2 matrix of [min, max] for each parameter
%   X_norm  : n×d normalized matrix in [0,1]

    logmin = log10(ranges(:,1))';   % 1×d
    logmax = log10(ranges(:,2))';   % 1×d
    X_norm = (log10(X) - logmin) ./ (logmax - logmin);
end

function X = denormalizeParamsLog(X_norm, ranges)
%DENORMALIZEPARAMSLOG Map normalized [0,1] parameters back in log10 space.
%   X_norm  : n×d matrix in [0,1]
%   ranges  : d×2 matrix of [min, max] for each parameter
%   X       : n×d matrix in original scale

    logmin = log10(ranges(:,1))';   % 1×d
    logmax = log10(ranges(:,2))';   % 1×d
    X = 10.^( X_norm .* (logmax - logmin) + logmin );
end
