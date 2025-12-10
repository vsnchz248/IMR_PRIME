function check_rqmc_convergence(results)
% CHECK_RQMC_CONVERGENCE - Check convergence using RQMC uncertainty
%
% When GP model uncertainty is unreliable (due to outliers), the RQMC
% sampling uncertainty is a more robust convergence metric.
%
% RQMC uncertainty < 0.001 (0.1%) is excellent
% RQMC uncertainty < 0.01 (1%) is good
% RQMC uncertainty < 0.05 (5%) is acceptable

    fprintf('\nTPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW\n');
    fprintf('Q   CONVERGENCE CHECK (RQMC-based)                  Q\n');
    fprintf('ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]\n\n');
    
    fprintf('Model   log10(E)    RQMC SE    Status\n');
    fprintf('------  ----------  ---------  -----------\n');
    
    all_converged = true;
    
    for i = 1:numel(results.models)
        gpr = results.gpr_out{i};
        
        % Extract RQMC uncertainty if available
        % This is in the raw GPR output, not always easily accessible
        % You may need to check what fields are available
        
        log10_mean = gpr.log10I_mean;
        
        % Try to extract RQMC SE from round_info
        if isfield(gpr, 'round_info') && ~isempty(gpr.round_info)
            last_round = gpr.round_info(end);
            
            if isfield(last_round, 'rqmcSE_rel')
                rqmc_se = last_round.rqmcSE_rel;
            elseif isfield(last_round, 'se_rel')
                rqmc_se = last_round.se_rel;
            else
                rqmc_se = NaN;
            end
        else
            rqmc_se = NaN;
        end
        
        % Quality assessment based on RQMC uncertainty
        if rqmc_se < 0.001
            status = ' Excellent';
        elseif rqmc_se < 0.01
            status = ' Good';
        elseif rqmc_se < 0.05
            status = '~ Acceptable';
        else
            status = ' Poor';
            all_converged = false;
        end
        
        if isnan(rqmc_se)
            status = '? Unknown';
        end
        
        fprintf('%-6s  %10.2f  %9.6f  %s\n', ...
                upper(results.models{i}), log10_mean, rqmc_se, status);
    end
    
    fprintf('\n');
    fprintf('Interpretation:\n');
    fprintf('  RQMC SE < 0.001: Excellent (0.1%% integration error)\n');
    fprintf('  RQMC SE < 0.01:  Good (1%% integration error)\n');
    fprintf('  RQMC SE < 0.05:  Acceptable (5%% integration error)\n');
    fprintf('\n');
    fprintf('Note: RQMC uncertainty measures integration accuracy,\n');
    fprintf('      independent of GP model uncertainty.\n');
    fprintf('\n');
    
    if all_converged
        fprintf(' All models have acceptable RQMC convergence\n');
        fprintf('   Results are reliable for model comparison\n');
    else
        fprintf(' Some models have poor RQMC convergence\n');
        fprintf('   Consider increasing maxRounds\n');
    end
    fprintf('\n');
    
    % Also print what convergence metric says
    fprintf('Standard Convergence Check (for comparison):\n');
    fprintf('Model   Rel.CI   Standard Check\n');
    fprintf('------  -------  --------------\n');
    for i = 1:numel(results.models)
        gpr = results.gpr_out{i};
        ci = gpr.log10I_CI95;
        rel_ci = (ci(2) - ci(1)) / abs(gpr.log10I_mean);
        
        if rel_ci < 0.05
            status = ' Pass';
        elseif rel_ci > 1e10
            status = ' Fail (infinite CI)';
        else
            status = sprintf('~ %.2e', rel_ci);
        end
        
        fprintf('%-6s  %7.4f  %s\n', ...
                upper(results.models{i}), rel_ci, status);
    end
    fprintf('\n');
    fprintf(' If standard check shows infinite CI but RQMC is good,\n');
    fprintf('  outliers are breaking GP uncertainty (use RQMC metric)\n');
end