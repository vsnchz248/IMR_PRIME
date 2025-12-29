function negLogLike = forward_solver_wrapper(theta, modelName, expData)
    sim = imr_forward_solver(theta, modelName, expData);
    likeOpts.useRdot  = true;
    likeOpts.betaGrid = 0.05:0.05:10;   % same as BIMR cfg.betaGrid
    [negLogLike, ~] = imr_negloglike(sim, expData, likeOpts);
end