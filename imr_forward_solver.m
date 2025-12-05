function sim = imr_forward_solver(theta, modelName, expData, solverOpts)
% IMR_FORWARD_SOLVER  Run forward IMR simulation for one parameter vector.
%
% This wraps f_imr_fd and keeps the solver piece separate from the likelihood.
%
% Inputs:
%   theta      - Parameter vector [1 x k] for the chosen model.
%   modelName  - String (e.g. 'newtonian','NH','NHKV','qNH','linmax','qKV','SLS').
%   expData    - Experimental data struct from prepare_data():
%                  .Rmax_mean, .tc_mean, .Req_each, .rho, .p_inf, .tmatrix
%   solverOpts - (optional) struct for solver knobs:
%                  .radial, .bubtherm, .medtherm, .masstrans, .vapor
%                  .Nt, .Mt
%
% Output:
%   sim - Struct with fields:
%          .t_nd    [nTime_sim x 1]  nondimensional time τ
%          .R_nd    [nTime_sim x 1]  nondimensional radius R*
%          .Rdot_nd [nTime_sim x 1]  nondimensional velocity dR*/dτ
%          .ok      logical          true if solver succeeded

    if nargin < 4, solverOpts = struct(); end

    % ---------- Parse parameters based on model ----------
    [mu, G, lambda1, alpha, stress_model] = parse_model_parameters(theta, modelName);

    % ---------- Solver configuration (defaults match your wrapper) ----------
    if ~isfield(solverOpts,'radial'),    solverOpts.radial    = 2;   end
    if ~isfield(solverOpts,'bubtherm'),  solverOpts.bubtherm  = 1;   end
    if ~isfield(solverOpts,'medtherm'),  solverOpts.medtherm  = 0;   end
    if ~isfield(solverOpts,'masstrans'), solverOpts.masstrans = 1;   end
    if ~isfield(solverOpts,'vapor'),     solverOpts.vapor     = 1;   end
    if ~isfield(solverOpts,'Nt'),        solverOpts.Nt        = 100; end
    if ~isfield(solverOpts,'Mt'),        solverOpts.Mt        = 100; end

    % ---------- Use mean experimental conditions ----------
    Rmax_sim = expData.Rmax_mean;
    tc_sim   = expData.tc_mean;
    Req_sim  = mean(expData.Req_each, 'omitnan');

    % Time vector: mean nondimensional experimental τ, scaled back to dimensional
    % NOTE: expData.tmatrix is nondimensional τ; we build dimensional t for f_imr_fd.
    tau_mean = mean(expData.tmatrix, 2);        % [nTime x 1]
    tvector  = tau_mean * tc_sim;               % dimensional time

    % ---------- Call IMR solver ----------
    try
        % Assuming f_imr_fd returns NONDIMENSIONAL τ, R*, Rdot*
        [t_sim_nd, R_sim_nd, Rdot_sim_nd, ~] = f_imr_fd( ...
            'radial',    solverOpts.radial, ...
            'stress',    stress_model, ...
            'bubtherm',  solverOpts.bubtherm, ...
            'medtherm',  solverOpts.medtherm, ...
            'masstrans', solverOpts.masstrans, ...
            'vapor',     solverOpts.vapor, ...
            'tvector',   tvector, ...
            'r0',        Rmax_sim, ...
            'req',       Req_sim, ...
            'mu',        mu, ...
            'g',         G, ...
            'lambda1',   lambda1, ...
            'lambda2',   0, ...
            'alphax',    alpha, ...
            'collapse',  0, ...
            'nt',        solverOpts.Nt, ...
            'mt',        solverOpts.Mt);

        sim.t_nd    = t_sim_nd(:);
        sim.R_nd    = R_sim_nd(:);
        sim.Rdot_nd = Rdot_sim_nd(:);
        sim.ok      = true;

    catch ME
        warning('IMR solver failed for %s with θ=[%s]: %s', ...
            modelName, sprintf('%.2e ', theta), ME.message);

        sim.t_nd    = [];
        sim.R_nd    = [];
        sim.Rdot_nd = [];
        sim.ok      = false;
    end
end

% -------------------------------------------------------------------------
% Helper: parse model parameters 
% -------------------------------------------------------------------------
function [mu, G, lambda1, alpha, stress_model] = parse_model_parameters(theta, modelName)

    mu      = 0;
    G       = 0;
    lambda1 = 0;
    alpha   = 0;

    switch lower(modelName)
        case 'newt'
            mu = theta(1);
            stress_model = 1;

        case 'nh'
            G = theta(1);
            stress_model = 1;

        case 'kv'
            mu = theta(1);
            G  = theta(2);
            stress_model = 1;

        case 'qnh'
            G     = theta(1);
            alpha = theta(2);
            stress_model = 2;

        case 'lm'
            mu      = theta(1);
            lambda1 = theta(2);
            stress_model = 3;

        case 'qkv'
            mu     = theta(1);
            G      = theta(2);
            alpha  = theta(3);
            stress_model = 2;

        case 'sls'
            mu      = theta(1);
            G       = theta(2);
            lambda1 = theta(3);
            stress_model = 3;

        otherwise
            error('Unknown model name: %s', modelName);
    end
end
