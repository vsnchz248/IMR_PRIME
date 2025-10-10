%% ================= DEMO (you can delete this block) =================
clear; clc; rng default;

% ----- Arbitrary-D synthetic data (here d = 4) -----
n = 400; d = 4;
X = rand(n,d);
f = @(x) sin(2*pi*x(:,1)) + 0.7*cos(2*pi*x(:,2)) + 0.5*x(:,3).^2 - 0.3*x(:,4);
Y = f(X) + 0.08*randn(n,1);

% ----- Grid controls -----
m = 40;                                   % points per varying dimension
gridRange = repmat([0 1], d, 1);          % [min max] per dim for prediction grid
fixedTargets = 0.5*ones(1,d);             % desired fixed values (snapped to grid)

% Example A: 2D slice (vary dims 1 & 3, fix others)
dimsToPlot = [1 3];
[gprMdl, meta] = gpr_slice_nd(X, Y, dimsToPlot, fixedTargets, m, gridRange);

% Example B: 1D slice (uncomment to try)
% dimsToPlot = 2;  % vary dim 2, fix others
% [gprMdl, meta] = gpr_slice_nd(X, Y, dimsToPlot, fixedTargets, m, gridRange);

disp('Kernel parameters (sigmaF, length scales..., sigmaNoise):');
disp(gprMdl.KernelInformation.KernelParameters.');

%% =================== GENERAL FUNCTION BELOW =========================
function [gprMdl, meta] = gpr_slice_nd(X, Y, dimsToPlot, fixedTargets, m, gridRange)
%GPR_SLICE_ND  Train N-D GPR and visualize a 1D or 2D slice.
%
% [gprMdl, meta] = gpr_slice_nd(X, Y, dimsToPlot, fixedTargets, m, gridRange)
%
% Inputs
%   X            : (n x d) training inputs
%   Y            : (n x 1) training response
%   dimsToPlot   : scalar (1D) or 2-element vector (2D) with dimensions to visualize
%   fixedTargets : (1 x d) desired fixed values for ALL dims (will be snapped)
%   m            : points per varying dimension (default 50)
%   gridRange    : (d x 2) [min max] per dimension (default [0,1] each)
%
% Behavior
% - Only the selected dims are varied on an m-point grid; all other dims are
%   snapped to the nearest grid value to fixedTargets.
% - Produces either a line (1D) or surface (2D) plot.

    arguments
        X double
        Y double
        dimsToPlot double
        fixedTargets double
        m (1,1) {mustBePositive, mustBeInteger} = 50
        gridRange double = []
    end

    [n,d] = size(X);
    assert(iscolumn(Y) && numel(Y)==n, 'Y must be n×1.');
    assert(all(dimsToPlot>=1 & dimsToPlot<=d), 'dimsToPlot must be within 1..d.');
    assert(numel(unique(dimsToPlot))==numel(dimsToPlot), 'dimsToPlot must be unique.');
    dimsToPlot = dimsToPlot(:).';        % row vector
    if isempty(gridRange)
        gridRange = repmat([0 1], d, 1);
    end
    assert(all(size(gridRange)==[d,2]), 'gridRange must be d×2.');
    assert(numel(fixedTargets)==d, 'fixedTargets must be length d.');

    % ---- Train ARD GPR ----
    gprMdl = fitrgp(X, Y, ...
        'KernelFunction','ardsquaredexponential', ...
        'Standardize',true);

    % ---- Build per-dimension grids ----
    v = cell(1,d);
    for j = 1:d
        v{j} = linspace(gridRange(j,1), gridRange(j,2), m);
    end

    % ---- Determine fixed dims and snap to nearest grid point ----
    dimsAll   = 1:d;
    dimsFixed = setdiff(dimsAll, dimsToPlot);
    kFixed    = zeros(1, numel(dimsFixed));
    for i = 1:numel(dimsFixed)
        j = dimsFixed(i);
        [~, kFixed(i)] = min(abs(v{j} - fixedTargets(j)));
    end

    % ---- Build slice design Xslice (only m or m^2 rows) ----
    if numel(dimsToPlot) == 1
        p = dimsToPlot(1);
        xp = v{p}(:);            % m×1
        Xslice = repmat(fixedTargets(:).', m, 1);
        Xslice(:,p) = xp;

        % Predict on the slice
        [Yhat, ~] = predict(gprMdl, Xslice);

        % Plot
        figure;
        plot(xp, Yhat, 'LineWidth', 1.8);
        grid on; box on;
        xlabel(sprintf('X_{%d}', p)); ylabel('Prediction');
        ttl = sprintf('GPR 1D slice (vary dim %d)', p);
        title(ttl);
        subtitle(makeFixSubtitle(dimsFixed, v, kFixed));

    elseif numel(dimsToPlot) == 2
        p = dimsToPlot(1);
        q = dimsToPlot(2);

        [Vp, Vq] = ndgrid(v{p}, v{q});      % each m×m
        Xslice = repmat(fixedTargets(:).', m*m, 1);
        Xslice(:,p) = Vp(:);
        Xslice(:,q) = Vq(:);

        % Predict on the slice
        [Yhat, ~] = predict(gprMdl, Xslice);
        Z = reshape(Yhat, [m m]);           % rows ~ v{p}, cols ~ v{q}

        % Plot
        figure;
        surf(v{p}, v{q}, Z.'); shading interp; grid on; box on;
        xlabel(sprintf('X_{%d}', p));
        ylabel(sprintf('X_{%d}', q));
        zlabel('Prediction');
        ttl = sprintf('GPR 2D slice (vary dims %d & %d)', p, q);
        title(ttl);
        subtitle(makeFixSubtitle(dimsFixed, v, kFixed));

    else
        error('dimsToPlot must have length 1 (1D) or 2 (2D).');
    end

    % ---- Fill meta ----
    meta.gridVectors = v;
    meta.dimsToPlot  = dimsToPlot;
    meta.dimsFixed   = dimsFixed;
    meta.kFixed      = kFixed;

    % fixedValues: NaN for plotted dims, snapped values for fixed dims
    fixedValues = nan(1,d);
    for i = 1:numel(dimsFixed)
        j = dimsFixed(i);
        fixedValues(j) = v{j}(kFixed(i));
    end
    meta.fixedValues = fixedValues;
    meta.m = m;
end

% ----------------- helper -----------------
function s = makeFixSubtitle(dimsFixed, v, kFixed)
    if isempty(dimsFixed)
        s = 'No fixed dimensions';
        return;
    end
    parts = strings(1,numel(dimsFixed));
    for i = 1:numel(dimsFixed)
        j = dimsFixed(i);
        parts(i) = sprintf('X_%d = %.4g', j, v{j}(kFixed(i)));
    end
    s = strjoin(parts, ', ');
end