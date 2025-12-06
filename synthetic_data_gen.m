lambda_vs_Rmax_fig
clc;close all
%
addpath('../IMRv2/src/forward_solver')

synthetic = 'synthetic_data.mat';

% if everything off, collapse must be off in function at bottom
radial = 2;
stress = 1; % 1: models 1,2,3; 2: models 4,6; 3: models 5,7
bubtherm = 1;
medtherm = 0;
masstrans = 1;
vapor = 1;
Nt = 100;
Mt = 100;

muval = logspace(-4,0,16);
Gval = logspace(2,log10(5e5),16);
alphaval = logspace(-3,1,16);

rho = 1064; % density
P_inf = 101325; % far field pressure
param = [0.05 0 0 0]; 

Rmax_range = linspace(Rmax-.1*Rmax_std,Rmax+.1*Rmax_std,32);
Req_range = Rmax_range./lambda;
tvector = linspace(0,150e-6,256);
tc = Rmax_range.*sqrt(rho/P_inf);
tmatrix = tvector'./tc;

%% Prep to run sims
N = length(Rmax_range);
futures(N) = parallel.FevalFuture; % Preallocate future array
synthetic_data = cell(N,1); % Preallocate storage for results
      
% Dispatch parallel jobs
for i = 1:N
    futures(i) = parfeval(@run_model, 2, i, radial, stress, bubtherm, ...
        medtherm, masstrans, vapor, tvector, Rmax_range(i), Req_range(i), param, Nt, Mt);
end
    
% Collect results as they complete
for i = 1:N
    try
        [~, result, idx] = fetchNext(futures);
        synthetic_data{idx} = result;
        fprintf('✓ Completed index %d\n', idx);
    catch ME
        fprintf('✗ Job %d failed: %s\n', i, ME.message);
    end
end
save(synthetic,'synthetic_data')

% === Helper Function ===
function [result, idx_out] = run_model(idx, radial, stress, bubtherm, ...
    medtherm, masstrans, vapor, tvector, R0, Req, param, Nt, Mt)

    [t, R, U, p] = f_imr_fd('radial', radial, 'stress', stress, ...
        'bubtherm', bubtherm, 'medtherm', medtherm, ...
        'masstrans', masstrans, 'vapor', vapor, 'tvector', tvector, ...
        'r0', R0, 'req', Req, 'mu', param(1), 'g', param(2), 'lambda1', param(3), ...
        'lambda2', 0, 'alphax', param(4), 'collapse', 0, 'nt', Nt, 'mt', Mt);

    result = [t, R, U, p];
    idx_out = idx;
end
