% clear; close all; clc;
addpath('./experiments/')

% -------------------- Settings --------------------
rho = 1064;           % kg/m^3
p8  = 101325;         % Pa (far-field)
makePlots = 0;

% Plot toggles 
do.DataDimPlot    = 0;
do.DataNondimPlot = 1;
do.EnvelopePlot   = 0;
do.ConfidencePlot = 0;
do.StrainRatePlot = 1;
do.FFTPlot        = 0;
do.SpectroPlot    = 0;
do.CloudPlot      = 0;

material = 1;  % 0: synthetic data, 1: UM1, 2: UM2, 3: UM3, 4: UT1, 5: UT2, 6: Ag2.5, 7: Ag0.5, 8: PA1.7, 9: PEGDA

% -------------------- Load R_d, t_d (DIMENSIONAL) --------------------
% Result after this block:
%   R_d [N x T] dimensional radius
%   t_d [N x T] dimensional time (starts near 0 per trial)
switch material
    case 0
        load('synthetic_data.mat');
        lambda_vs_Rmax_fig; close all;
        Rmax_range = linspace(Rmax-.1*Rmax_std,Rmax+.1*Rmax_std,32);
        Req_range = Rmax_range./lambda;
        for i = 1:length(synthetic_data)
            R_d(:,i) = synthetic_data{i}(:,2).*Rmax_range(i);
            t_d(:,i) = linspace(0,150e-6,256);
        end
    case 1
        load('Processed_Data.mat')
        k = 1; 
        for i = 7:15
            R_d(:,k) = expts(i).Roft;  t_d(:,k) = expts(i).t;  k = k + 1;
        end
    case 2
        load('Processed_Data.mat')
        k = 1; 
        for i = 78:98%125:135%
            R_d(:,k) = expts(i).Roft;  t_d(:,k) = expts(i).t;  k = k + 1;
        end
    case 3
        load('Processed_Data.mat')
        k = 1; 
        for i = 223:234
            R_d(:,k) = expts(i).Roft;  t_d(:,k) = expts(i).t;  k = k + 1;
        end

    % ---- Alternate folders (build dimensional R_d,t_d from ND) ----
    case 4   % 10percent
        load('./experiments/10percent/Rt_nondim_exp.mat')     % R_nondim_All, t_nondim_All, RmaxList, P_inf
        if ~exist('P_inf','var'), P_inf = p8; end
        minLen = min(cellfun(@length, R_nondim_All));
        nTrials = [1,2,3,4];
        for k = 1:numel(nTrials)
            i = nTrials(k);
            tc = RmaxList(i)*sqrt(rho/P_inf);
            R_d(:,k) = R_nondim_All{i}(1:minLen) * RmaxList(i);
            t_d(:,k) = t_nondim_All{i}(1:minLen) * tc;
        end
    case 5   % Ag5.0
        load('./experiments/Ag5.0/Rt_nondim_exp.mat')
        if ~exist('P_inf','var'), P_inf = p8; end
        indices = [1:5, 7:10];
        minLen  = min(cellfun(@(x) length(x), R_nondim_All(indices)));
        k = 1; 
        for i = indices
            tc = RmaxList(i)*sqrt(rho/P_inf);
            R_d(:,k) = R_nondim_All{i}(1:minLen) * RmaxList(i);
            t_d(:,k) = t_nondim_All{i}(1:minLen) * tc;
            k = k + 1;
        end
    case 6     % Ag2.5
        load('./experiments/Ag2.5/Rt_nondim_exp.mat')
        if ~exist('P_inf','var'), P_inf = p8; end
        indices = [1:6, 8:11];
        minLen  = min(cellfun(@(x) length(x), R_nondim_All(indices)));
        k = 1; 
        for i = indices
            tc = RmaxList(i)*sqrt(rho/P_inf);
            R_d(:,k) = R_nondim_All{i}(1:minLen) * RmaxList(i);
            t_d(:,k) = t_nondim_All{i}(1:minLen) * tc;
            k = k + 1;
        end
    case 7     % Ag0.5
        load('./experiments/Ag0.5/Rt_nondim_exp.mat')
        if ~exist('P_inf','var'), P_inf = p8; end
        indices = 2:11;
        minLen  = min(cellfun(@(x) length(x), R_nondim_All(indices)));
        k = 1; 
        for i = indices
            tc = RmaxList(i)*sqrt(rho/P_inf);
            R_d(:,k) = R_nondim_All{i}(1:minLen) * RmaxList(i);
            t_d(:,k) = t_nondim_All{i}(1:minLen) * tc;
            k = k + 1;
        end
    case 8     % PA1pt7
        load('./experiments/PA1pt7/Rt_nondim_exp.mat')
        if ~exist('P_inf','var'), P_inf = p8; end
        indices = 1:10;
        minLen  = min(cellfun(@(x) length(x), R_nondim_All(indices)));
        k = 1; 
        for i = indices
            tc = RmaxList(i)*sqrt(rho/P_inf);
            R_d(:,k) = R_nondim_All{i}(1:minLen) * RmaxList(i);
            t_d(:,k) = t_nondim_All{i}(1:minLen) * tc;
            k = k + 1;
        end
    case 9
        load('./experiments/forbenchmarks/Rt_nondim_exp_PEGDA_S25L6.mat')
        if ~exist('P_inf','var'), P_inf = p8; end
        indices = 1:7;
        minLen  = min(cellfun(@(x) length(x), R_nondim_All(indices)));
        k = 1; 
        for i = indices
            tc = RmaxList(i)*sqrt(rho/P_inf);
            R_d(:,k) = R_nondim_All{i}(1:minLen) * RmaxList(i);
            t_d(:,k) = t_nondim_All{i}(1:minLen) * tc;
            k = k + 1;
        end
    otherwise
        error('Unknown material selection.');
end

% ---------------- Align at peak and truncate (DIMENSIONAL) --------------

[~, startIdx] = max(R_d, [], 1);
nCols = size(R_d,2);
tailLen = zeros(1,nCols);
for j = 1:nCols
    tailLen(j) = size(R_d,1) - startIdx(j) + 1;
end
minLen = min(tailLen);

R_aligned = zeros(minLen, nCols);
t_aligned = zeros(minLen, nCols);
for j = 1:nCols
    idx = startIdx(j):(startIdx(j)+minLen-1);
    R_aligned(:,j) = R_d(idx,j);
    t_aligned(:,j) = t_d(idx,j);
end

% keep dimensional copies
R_d = R_aligned;
t_d = t_aligned;

% Reset each trial's time so it starts at zero
for k = 1:size(t_d,2)
    t_d(:,k) = t_d(:,k) - ones(size(t_d,1),1).*min(t_d(:,k));
end

% ---------------- Nondimensionalize ----------------
Rmax_range = max(R_d, [], 1);
tc         = Rmax_range .* sqrt(rho/p8);

Rmatrix = R_d ./ Rmax_range;   % ND radius
tmatrix = t_d ./ tc;           % ND time

% Basic stats (both overall and per-trial)
last10    = max(1, ceil(0.10*size(R_d,1)));
Req_each  = mean(R_d(end-last10+1:end, :), 1);   % per-trial Req (dimensional)
R0        = mean(Rmax_range);                    
Req       = mean(Req_each);                      

% ---------------- Rdot / strain rate ----------------
if exist('dataprep','file') == 2
    [~,~,noisyR,~,R_stats,Rdot_stats,Rdotmatrix,~,~,~] = ...
        dataprep(tmatrix, Rmatrix, tc, R_d, Rmax_range); 
else
    Rdotmatrix = zeros(size(Rmatrix));
    for j = 1:size(Rmatrix,2)
        Rdotmatrix(:,j) = gradient(Rmatrix(:,j), tmatrix(:,j));
    end
end

strainRate_nd = abs((-2 .* Rdotmatrix) ./ Rmatrix);
strainRate_nd(~isfinite(strainRate_nd)) = NaN;

% ---- Dimensional strain rate (s^-1) from ND quantity ----
tc_each = reshape(tc, 1, []);                 % 1 x nTrials
strainRate_dim = abs(strainRate_nd ./ tc_each);      % same size as Rmatrix

tvector = mean(t_d,2); % for multirun code

% ---------------- Call plotData ----------------
if makePlots
    data = struct( ...
        't_nd', tmatrix, ...
        'R_nd', Rmatrix, ...
        't_d', t_d, ...
        'R_d', R_d, ...
        'Rdot_nd', Rdotmatrix, ...
        'strainRate_nd', strainRate_dim, ...
        'Rmax_each', Rmax_range, ...
        'tc_each', tc, ...
        'Req_each', Req_each ...
    );
    if exist('noisyR','var'), data.noisyR = noisyR; end

    % If your function is in +imr use imr.plotData; if plain function, use plotData
    figs = plotData(data, ...
    'DataDimPlot',    logical(do.DataDimPlot), ...
    'DataNondimPlot', logical(do.DataNondimPlot), ...
    'EnvelopePlot',   logical(do.EnvelopePlot), ...
    'ConfidencePlot', logical(do.ConfidencePlot), ...
    'StrainRatePlot', logical(do.StrainRatePlot), ...
    'FFTPlot',        logical(do.FFTPlot), ...
    'SpectroPlot',    logical(do.SpectroPlot), ...
    'CloudPlot',      logical(do.CloudPlot));
end
